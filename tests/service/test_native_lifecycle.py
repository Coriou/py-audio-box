"""Seeded native SOURCE journey. No installed onboarding/hardware authority."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import time
import unittest
from unittest.mock import patch

from service.core import Core
from service.dispatcher import Dispatcher
from service.execution_context import NativeQwenContexts
from service.reconciliation import Reconciliation
from service.worker_transport import Authority
from tests.service.test_native_runtime import NativeFixture
from worker import native_protocol as p
from worker.engines.qwen_execution import inventory_from_wire
from worker.transport import Refused


class NativeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.f=NativeFixture(self, clone=getattr(self,'clone_history',False));self.f.s.utc=lambda:datetime.now(timezone.utc)
        if getattr(self,'clone_history',False):
            from dataclasses import replace
            from contracts import validate as oracle
            from service.ingest import ingest_local
            # Explicit trusted fixture enrollment, not public clone onboarding.
            core=Core(self.f.f.fixture.root/'core')
            try:
                from service.admission import Admission
                # Initialize the resource protocol while the fixture root is
                # still empty, as Dispatcher does before accepting uploads.
                Admission(core,self.f.stage,private_spool=self.f.spool)
                asset=ingest_local(core,'alice',self.f.reference_raw)
            finally: core.close()
            plan=self.f.snapshot[1]['plan']; version=plan['voice_versions'][0]
            rid=asset['id']; plan['assets']=[asset]
            version.update(reference_asset_ids=[rid],reference_asset_sha256={rid:asset['sha256']})
            version['configuration']['reference_asset_id']=rid
            self.f.f.e=replace(self.f.f.e,reference_id=rid)
            self.f.snapshot[1]['execution']['descriptor']=self.f.f.e.identity()
            self.f.snapshot[1]['execution']['references']={rid:dict(path='/references/assets/'+rid,
                voice_id=self.f.f.e.voice_id,voice_revision=self.f.f.e.voice_revision,
                checkpoint=self.f.f.e.checkpoint,revision=self.f.f.e.revision,text=self.f.f.e.reference_text)}
            self.f.snapshot[2]=[rid]
            self.f.snapshot[0]['input_sha256']=oracle.plan_hash(plan)
            self.f.a=Authority(*self.f.snapshot)
            self.f.s.registrations={self.f.a.binding:dict(snapshot=copy.deepcopy(self.f.snapshot),
                execution=self.f.f.e,bundles={'bundle':self.f.f.fixture.bundle},inventory=self.f.f.inventory)}
        self.context=copy.deepcopy(self.f.snapshot[1]);self.profiles=self.f.snapshot[3]
        self.profiles[0]['limits']['disk_bytes']=2*1024**2
        self.context['plan']['resolved']['profile']['limits']['disk_bytes']=2*1024**2
        self.root=self.f.f.fixture.root/'core'
        seed=Core(self.root)
        seed.voices.register('alice','fixture-owned-version',self.context['plan']['voice_versions'][0],expected_revision=0)
        self.context['plan']['voice_versions']=[seed.voices.version('alice','vv0')]
        self.f.s.registrations[self.f.a.binding]['snapshot'][1]=copy.deepcopy(self.context)
        self.f.s.registrations[self.f.a.binding]['snapshot'][3]=copy.deepcopy(self.profiles)
        self.f.s.registrations[self.f.a.binding]['snapshot'][0]['principal_id']='alice'
        # Explicit fixture enrollment must update originals, not rely on allocations.
        self.f.s._preset_templates=copy.deepcopy(self.f.s.registrations)
        seed.close()
        self.contexts=NativeQwenContexts({('alice','qwen-mps','vv0'):self.context},self.f.f.inventory,self.profiles,self.f.native)
        self.dispatcher=None

    def wait(self, fn):
        end=time.monotonic()+5
        while time.monotonic()<end:
            value=fn()
            if value:return value
            time.sleep(.01)
        self.fail('bounded source fixture wait expired: '+repr(self.dispatcher.call('status') if self.dispatcher else None)+' requests='+repr([(q['op'],q['runtime_id']) for q in self.f.requests]))

    def open(self, *, finish=True):
        f=self.f
        def before(q):
            if q['op'] in ('observe','collect'):
                if finish and f.s.active:
                    f.finish()
                f.s.tick()
        f.before=before
        self.dispatcher=Dispatcher(self.root,staging=f.stage,runtime=f.runtime,
            profiles=self.profiles,production_contexts=self.contexts,
            **({'fault':self.extra_fault} if getattr(self,'extra_fault',None) else {}))
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        self.assertFalse(self.dispatcher.call('status')['integrity_fault'],self.dispatcher.call('status'))
        return self.dispatcher

    def close(self):
        if self.dispatcher:self.dispatcher.close();self.dispatcher=None

    def test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job(self):
        f=self.f
        with patch('worker.native_child.parent_exchange',side_effect=f.exchange):
            try:
                d=self.open()
                result=d.call('production_generate','alice',self.context['plan']['request'],key='first')
                job=self.wait(lambda:(j if (j:=d.call('job',result['job_id']))['state'] in ('failed','succeeded') else None))
                self.assertEqual(job['state'],'succeeded',job)
                self.wait(lambda:d.call('status')['phase']=='idle')
                artifact=d.call('production_artifact','alice',dict(artifact_id=job['artifact_ids'][0]))
                self.assertEqual(d.call('status')['reservations'][0]['demand']['pids'],1)
                self.assertEqual(d.call('status')['reservations'][0]['demand']['vram_bytes'],0)
                self.assertEqual(d.call('status')['reservations'][0]['state'],'retained')
                self.assertEqual(len(f.f.calls),1)
                self.close()
                # Reopen the SAME native journal (completed records only), with
                # a fresh boot and an updated trusted context for the next job.
                old_runtime=f.runtime;old_id=next(iter(old_runtime.records));old_a=old_runtime.records[old_id]['authority']
                original=f.s;original.journal.lock.close();f.f.make();f.s=f.f.s
                f.s.utc=lambda:datetime.now(timezone.utc);f.s.clock=time.monotonic
                self.assertEqual(old_runtime.collect(old_a,old_id).receipt['runtime_id'],old_id)
                self.context['execution']['native']['boot']=f.s.boot
                f.native['boot']=f.s.boot
                # Native trusted context metadata changed only through explicit
                # fixture enrollment, never a hello-driven public registration.
                template=copy.deepcopy(f.snapshot);template[0]['principal_id']='alice';template[1]=copy.deepcopy(self.context);template[3]=copy.deepcopy(self.profiles)
                from contracts import validate as oracle
                template[0]['input_sha256']=oracle.plan_hash(template[1]['plan'])
                a=Authority(*template)
                f.s.registrations={a.binding:dict(snapshot=template,execution=f.f.e,bundles={'bundle':f.f.fixture.bundle},inventory=f.f.inventory)}
                f.runtime=f.make_runtime()
                self.contexts=NativeQwenContexts({('alice','qwen-mps','vv0'):self.context},f.f.inventory,self.profiles,f.native)
                f.f.h=type(f.f.h)()
                d=self.open()
                self.assertEqual(d.call('production_artifact','alice',dict(artifact_id=artifact['id'])),artifact)
                next_job=d.call('production_generate','alice',self.context['plan']['request'],key='second')
                job2=self.wait(lambda:(j if (j:=d.call('job',next_job['job_id']))['state'] in ('failed','succeeded') else None))
                self.assertEqual(job2['state'],'succeeded',job2)
                self.wait(lambda:d.call('status')['phase']=='idle')
                self.assertEqual(len(f.f.calls),2)
            finally:self.close()

    def import_sync_failure(self, which):
        import stat
        from service import native_runtime as module
        f=self.f;injected=[];real_sync=module._sync;real_fsync=os.fsync
        def paths():
            if not f.runtime.records:return None,None,None
            r=next(iter(f.runtime.records.values()));a=r['authority']
            directory=f.spool/a.binding
            stage=f.stage/a.job['attempts'][-1]['staging_id']
            return directory,stage,r
        def ready():
            directory,stage,r=paths()
            return directory is not None and (stage/'output.wav').exists() and not (directory/'import').exists()
        def sync(path):
            directory,stage,r=paths()
            if which=='private' and path==directory and ready() and not injected:
                injected.append(which);raise OSError('single private post-unlink fsync uncertainty')
            return real_sync(path)
        def fsync(fd):
            directory,stage,r=paths()
            info=os.fstat(fd)
            if which=='stage' and ready() and stat.S_ISDIR(info.st_mode) and info.st_ino==stage.stat().st_ino and not injected:
                injected.append(which);raise OSError('single stage post-unlink fsync uncertainty')
            return real_fsync(fd)
        with patch('worker.native_child.parent_exchange',side_effect=f.exchange),patch('service.native_runtime._sync',side_effect=sync),patch('service.native_runtime.os.fsync',side_effect=fsync):
            try:
                d=self.open()
                result=d.call('production_generate','alice',self.context['plan']['request'],key='uncertain-'+which)
                self.wait(lambda:injected and f.runtime.import_closed)
                rid=next(iter(f.runtime.records));a=f.runtime.records[rid]['authority']
                before=sum(q['op']=='observe' for q in f.requests)
                self.wait(lambda:sum(q['op']=='observe' for q in f.requests)>=before+6)
                for _ in range(3):
                    self.assertEqual(f.runtime.observe(rid).state.value,'stopped-and-reaped')
                    f.runtime.terminate(rid)
                    with self.assertRaisesRegex(Refused,'uncertain'):f.runtime.collect(a,rid)
                    with self.assertRaisesRegex(Refused,'uncertain'):f.runtime.call('ack',dict(runtime_id=rid))
                    self.assertEqual(d.call('job',result['job_id'])['artifact_ids'],[])
                    self.assertEqual(d.call('status')['reservations'][0]['state'],'executing')
                self.assertTrue(f.runtime.failed);self.assertEqual(injected,[which])
                self.assertNotIn('ack',[q['op'] for q in f.requests])
                self.assertFalse(f.s.journal.value['records'][0]['acked'])
                self.close()
                core=Core(self.root)
                try:self.assertEqual(core.db.execute('SELECT * FROM publications').fetchall(),[])
                finally:core.close()
                f.runtime=f.make_runtime();self.assertFalse(f.runtime.admission_open)
                d=self.open()
                self.wait(lambda:sum(q['op']=='observe' for q in f.requests)>=before+9)
                self.assertFalse(f.runtime.admission_open)
                self.assertEqual(d.call('job',result['job_id'])['artifact_ids'],[])
                self.assertEqual(d.call('status')['reservations'][0]['state'],'executing')
                self.assertNotIn('ack',[q['op'] for q in f.requests])
            finally:self.close()

    def test_private_post_unlink_sync_uncertainty_never_hands_off(self):
        self.import_sync_failure('private')

    def test_stage_post_unlink_sync_uncertainty_never_hands_off(self):
        self.import_sync_failure('stage')

    def test_stale_future_native_registration_and_historical_decode(self):
        for seconds in (-301,1):
            context=copy.deepcopy(self.context)
            context['execution']['hardware']['devices'][0]['probed_at']=(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat()
            inventory=inventory_from_wire(context['execution']['hardware'])
            self.assertEqual(p.native_execution(context),self.f.f.e)
            with self.assertRaisesRegex(Refused,'stale'):
                NativeQwenContexts({('alice','qwen-mps','vv0'):context},inventory,self.profiles,self.f.native)
        with self.assertRaises(Refused):self.contexts.resolve('bob','speech_generate',self.context['plan']['request'])
        changed=copy.deepcopy(self.context['plan']['voice_versions'][0]);changed['name']='foreign'
        with self.assertRaises(Refused):self.contexts.resolve_registered('alice',self.context['plan']['request'],changed)

    def test_cancel_retains_reservation_until_actual_owned_receipt(self):
        f=self.f
        with patch('worker.native_child.parent_exchange',return_value=None):
            try:
                d=self.open(finish=False)
                result=d.call('production_generate','alice',self.context['plan']['request'],key='cancel')
                self.wait(lambda:f.s.active is not None and f.s.active['handle'] is not None)
                d.call('production_cancel','alice',dict(job_id=result['job_id']),key='cancel-intent')
                self.wait(lambda:f.s.active['cancel'])
                self.assertEqual(d.call('status')['reservations'][0]['state'],'executing')
                self.assertEqual(d.call('job',result['job_id'])['state'],'cancel_requested')
                f.f.h.code=-15;f.s.active['receiver'].join(1);f.s.tick()
                job=self.wait(lambda:(j if (j:=d.call('job',result['job_id']))['state']=='cancelled' else None))
                self.assertEqual(job['artifact_ids'],[])
                self.wait(lambda:d.call('status')['phase']=='idle')
                self.assertEqual(d.call('status')['reservations'][0]['state'],'retained')
            finally:self.close()

    def test_live_linux_receipt_substitution_retains_reservation_and_detached_rejects(self):
        f=self.f
        def mutate(q,value):
            if q['op']=='collect' and value.get('value',{}).get('receipt'):
                value['value']['receipt']['reap_method']='pid1-waitpid-echild'
            return value
        f.mutate=mutate
        with patch('worker.native_child.parent_exchange',side_effect=f.exchange):
            try:
                d=self.open()
                result=d.call('production_generate','alice',self.context['plan']['request'],key='bad-receipt')
                self.wait(lambda:d.call('status')['phase']=='unknown')
                self.assertEqual(d.call('status')['reservations'][0]['state'],'executing')
                self.assertEqual(d.call('job',result['job_id'])['artifact_ids'],[])
            finally:self.close()
        # Same frozen lane rejects Linux substitution in the detached validator.
        core=Core(self.root)
        try:
            row=core.db.execute('SELECT authority,runtime FROM dispatches').fetchone()
            a=Authority(*json.loads(row[0]));receipt=copy.deepcopy(f.s.journal.value['records'][0]['receipt'])
            receipt['reap_method']='pid1-waitpid-echild'
            core.db.execute('UPDATE dispatches SET receipt=?',(json.dumps(receipt),))
            scan=Reconciliation(core,f.stage)
            self.assertTrue(scan.reconcile_startup()['blocked'])
        finally:core.close()

    def test_seeded_http_cli_mcp_generation_job_artifact_and_owner(self):
        import asyncio
        import contextlib
        import io
        import httpx
        from clients.http import HTTP
        from clients.cli import main
        from clients import mcp
        from service.api import create_voice_app
        from service.auth import Credential, StaticCredentials
        outer=self;f=self.f
        with patch('worker.native_child.parent_exchange',side_effect=f.exchange):
            try:
                d=self.open()
                app=create_voice_app(d,credentials=StaticCredentials([
                    Credential('writer','alice',frozenset({'read','production-write'})),
                    Credential('bob','bob',frozenset({'read','production-write'}))]))
                class ASGIClient:
                    def __init__(inner,**kw):pass
                    def __enter__(inner):return inner
                    def __exit__(inner,*a):pass
                    def request(inner,method,path,**kw):
                        async def run():
                            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                                return await client.request(method,path,**kw)
                        return asyncio.run(run())
                    def post(inner,path,**kw):return inner.request('POST',path,**kw)
                    def get(inner,path,**kw):return inner.request('GET',path,**kw)
                def cli(args):
                    output=io.StringIO()
                    with contextlib.redirect_stdout(output):code=main(args)
                    self.assertEqual(code,0,output.getvalue());return json.loads(output.getvalue())
                with patch('clients.http.httpx.Client',ASGIClient),patch.dict(os.environ,{'SPEECH_CREDENTIAL':'writer'}):
                    transport=HTTP(credential='writer');payload=self.context['plan']['request']
                    result=transport.call('speech_generate',payload,key='same')
                    self.assertEqual(cli(['speech','generate','--request',json.dumps(payload),'--idempotency-key','same']),result)
                    m=asyncio.run(mcp.call_tool('speech_speech_generate',dict(payload,idempotency_key='same')))
                    self.assertFalse(m.isError);self.assertEqual(m.structuredContent,result)
                    self.wait(lambda:d.call('job',result['job_id'])['state']=='succeeded')
                    self.wait(lambda:d.call('status')['phase']=='idle')
                    job=transport.call('job_get',dict(job_id=result['job_id']))
                    artifact=transport.call('artifact_get',dict(artifact_id=job['artifact_ids'][0]))
                    self.assertEqual(cli(['artifact','get',artifact['id']]),artifact)
                    m=asyncio.run(mcp.call_tool('speech_artifact_get',dict(artifact_id=artifact['id'])))
                    self.assertFalse(m.isError);self.assertEqual(m.structuredContent,artifact)
                    with self.assertRaises(Exception):HTTP(credential='bob').call('job_get',dict(job_id=job['id']))
                    with self.assertRaises(Exception):HTTP(credential='bob').call('speech_generate',payload,key='foreign')
                self.assertEqual(len(f.f.calls),1)
            finally:self.close()

    def test_native_publication_storage_uncertainty_retains_reservation_until_reopen(self):
        f=self.f
        def fail(point):
            if point=='after_adoption':raise OSError('injected native publication storage failure')
        self.extra_fault=fail
        with patch('worker.native_child.parent_exchange',side_effect=f.exchange):
            try:
                d=self.open()
                result=d.call('production_generate','alice',self.context['plan']['request'],key='storage')
                self.wait(lambda:d.call('status')['persistence_fault'])
                status=d.call('status')
                self.assertEqual(status['reservations'][0]['state'],'executing')
                self.assertEqual(d.call('job',result['job_id'])['artifact_ids'],[])
                self.assertFalse(f.s.journal.value['records'][0]['acked'])
                self.close();self.extra_fault=None
                f.runtime=f.make_runtime()
                self.assertFalse(f.runtime.admission_open)
                d=self.open()
                self.assertTrue(f.runtime.admission_open)
                job=self.wait(lambda:(j if (j:=d.call('job',result['job_id']))['state']=='succeeded' else None))
                self.wait(lambda:d.call('status')['phase']=='idle')
                self.assertEqual(d.call('status')['reservations'][0]['state'],'retained')
                self.assertEqual(len(job['artifact_ids']),1);self.assertEqual(len(f.f.calls),1)
            finally:self.close()


class NativeCloneLifetimeTests(unittest.TestCase):
    """Seeded clone reference is held through cancellation and confirmed reap."""
    clone_history=True
    setUp=NativeLifecycleTests.setUp
    wait=NativeLifecycleTests.wait
    open=NativeLifecycleTests.open
    close=NativeLifecycleTests.close

    def test_cancel_keeps_reference_until_confirmed_reap(self):
        f=self.f; spec=f.f.e
        with patch('worker.native_child.parent_exchange',return_value=None):
            try:
                d=self.open(finish=False)
                result=d.call('production_generate','alice',self.context['plan']['request'],key='clone-cancel')
                self.wait(lambda:f.s.active is not None and f.s.active['handle'] is not None)
                row=f.s.journal.value['records'][0]
                native_input=f.s.directory(row)/('input-'+spec.reference_sha256)
                control_input=next(f.spool.glob('*/input'))
                owned=self.root/'assets'/spec.reference_id
                d.call('production_cancel','alice',dict(job_id=result['job_id']),key='cancel')
                self.wait(lambda:f.s.active['cancel'])
                for path in (native_input,control_input,owned):
                    self.assertEqual(path.read_bytes(),f.reference_raw)
                self.assertIsNone(f.s.journal.value['records'][0]['receipt'])
                self.assertEqual(d.call('status')['reservations'][0]['state'],'executing')
                f.f.h.code=-15;f.s.active['receiver'].join(1);f.s.tick()
                self.wait(lambda:d.call('job',result['job_id'])['state']=='cancelled')
                self.wait(lambda:d.call('status')['phase']=='idle')
                self.assertTrue(f.s.journal.value['records'][0]['receipt']['reaped'])
                for path in (native_input,control_input,owned):
                    self.assertEqual(path.read_bytes(),f.reference_raw)
            finally:self.close()
