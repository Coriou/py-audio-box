"""Offline injected CPU x-vector journey; no model import or installed qualification."""
import asyncio
import copy
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import io
import hashlib
import json
import os
import struct
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from contracts import validate as oracle
from lib.tts_backends.contract import AudioResult
from service.admission import Admission
from service.core import Core
from service.dispatch_store import DispatchStore
from service.execution_context import QwenContexts
from service.ingest import ingest_local
from service.production import Production
from tests.service import test_qwen_lifecycle as lifecycle_fixtures
from worker.engines.qwen import execute
from worker.engines.qwen_prepare import QwenPreparation, selected_reference
from worker.runtime_child import run_generation
from worker.runtime_protocol import execution, generated_source


class CPUPreparationFixture(unittest.TestCase):
    def setUp(self):
        self.b=lifecycle_fixtures.QwenLifecycleTests();self.b.setUp();self.addCleanup(self.b.doCleanups)
        b=self.b
        self.template=copy.deepcopy(b.context)
        spec=replace(b.spec,x_vector_only=True,reference_text=None,timeout_s=1800)
        v=self.template['plan']['voice_versions'][0]
        v.update(reference_text='',configuration=dict(mode='qwen_clone',reference_asset_id=b.asset['id'],reference_text='',conditioning='x_vector'))
        self.template['execution']['descriptor']=spec.identity()
        self.template['execution']['references'][b.asset['id']]['text']=None
        self.profiles=copy.deepcopy(b.profiles)
        self.profiles[0]['limits']['deadline_seconds']=1800
        self.profiles[0]['compatibility'].append(dict(operation='voice_prepare',mode='prepare',identity=v['identity']))
        self.template['plan']['resolved']['profile']=oracle.profile_snapshot(self.profiles[0])
        refs=copy.deepcopy(b.refs);refs[b.asset['id']]['text']=None
        self.contexts=QwenContexts({'alice':self.template},b.mapping,refs,b.inventory,self.profiles)
        self.request=dict(asset_id=b.asset['id'],in_tick=0,out_tick=8,reference_text='',
            audition_text='A separate audition.',identity=v['identity'])
        self.calls=[];self.collection_errors=[]

    def retained_tables(self, root):
        """Count every retained SQL table, including sqlite_sequence, after close."""
        import sqlite3
        # Core's database path is supplied by the fixture, not guessed by a reader.
        from contextlib import closing
        with closing(sqlite3.connect(Path(root).as_uri()+'?mode=ro',uri=True)) as db:
            names=[row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            return {name:db.execute('SELECT count(*) FROM "'+name.replace('"','""')+'"').fetchone()[0]
                    for name in names}

    def retained_identity(self, database):
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(Path(database).as_uri()+'?mode=ro',uri=True)) as db:
            return {name:db.execute('SELECT * FROM '+name+' ORDER BY rowid').fetchall()
                    for name in ('jobs','events','dispatches','dispatch_commands',
                                 'catalogue_requests','catalogue_versions')}

    def resolve(self, request=None, principal='alice'):
        return self.contexts.resolve_owned(self.b.core,principal,'voice_prepare',request or self.request)

    def submit(self,key='prepare'):
        return Production(self.b.core).prepare('alice',key,self.resolve(),_admission=self.b.admission,_profiles=self.profiles)

    def proposal(self,authority=None):
        b=self.b
        if authority is None:
            self.submit();authority=b.store.allocate(self.profiles,'qwen',1800)['authority']
        stage=b.stage/authority.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
        (stage/'conditioning').mkdir(mode=0o770)
        outer=self
        class Backend:
            def __init__(inner,**kw):outer.calls.append(('construct',kw))
            def synthesise_clone(inner,params,reference,transcript,xvector,timeout):
                outer.assertIs(type(reference),bytes)
                outer.assertIsNone(transcript);outer.assertIs(xvector,True)
                outer.assertGreater(timeout,0);outer.assertLessEqual(timeout,1800)
                outer.calls.append((params['text'],reference))
                return AudioResult([.25,-.25,.5],24000,metadata=dict(checkpoint=str(b.bundle),revision='rev1'))
            def close(inner):outer.calls.append('closed')
        def engine(spec,text,bundles,references,inventory,**kw):
            self.assertEqual(references,authority.context['execution']['references'])
            local=copy.deepcopy(references)
            for identity in local:local[identity]['path']=b.core.root/'assets'/identity
            kw['managed_owner_uid']=os.getuid()
            return execute(spec,text,b.mapping,local,inventory,**kw)
        sent=[]
        config=dict(job=authority.job,context=authority.context,allowed_assets=list(authority.allowed_assets),profiles=list(authority.profiles),secret='a'*64,control_uid=os.getuid(),gid=os.getgid())
        # Model the installed helper's private descriptor, not the local Core
        # publication mode, which follows the local process umask. The actual
        # fd reader still checks mode, links, access, length and hash.
        bridge=b.root/'managed-reference.wav'
        source=b.core.root/'assets'/execution(authority.context).reference_id
        fd=os.open(bridge,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(source.read_bytes())
        self.addCleanup(lambda:bridge.unlink(missing_ok=True))
        with bridge.open('rb') as reference, \
                patch('lib.tts_backends.pocket_snapshot.managed_directory',side_effect=lambda p:nullcontext()), \
                patch('worker.transport.authenticate'),patch('worker.transport.receive',return_value=authority.message()), \
                patch('worker.transport.send',side_effect=lambda c,v,t:sent.append(v)), \
                patch('worker.engines.qwen.execute',side_effect=engine):
            config['reference_fd']=reference.fileno()
            run_generation(None,config,b.stage,time.monotonic()+1790,backend_factory=Backend)
        bridge.unlink()
        self.assertEqual(list((stage/'conditioning').iterdir()),[])
        return authority,sent[0],stage

    def collected_proposal(self, authority):
        try:
            return self.proposal(authority)
        except Exception as exc:
            self.collection_errors.append(type(exc).__name__+': '+str(exc))
            raise

    def wait(self, predicate):
        end=time.monotonic()+5
        while time.monotonic()<end:
            value=predicate()
            if value:return value
            time.sleep(.01)
        self.fail(str(dict(status=self.dispatcher.call('status'),
                           collection_errors=self.collection_errors)))

    def finish(self,a,p,stage):
        b=self.b
        b.core._publish(a.job['id'],p,stage_root=b.stage,engine_uid=os.getuid())
        with b.core.transaction():b.admission.store.release_tx(a.job['id'],confirmed_reap=True)
        b.store.update(b.store.current(),acked=1,phase='terminal')
        (stage/'conditioning').rmdir()


class CPUPreparationTests(CPUPreparationFixture):
    def test_owned_riff_chunks_keep_exact_uploaded_size_and_digest(self):
        from worker.transport import Refused
        from worker.engines.pocket import _read
        # A valid mono PCM16 upload may carry an ancillary RIFF chunk.
        raw=self.b.raw
        extra=raw[:36]+b'LIST'+struct.pack('<I',5)+b'hello\0'+raw[36:]
        extra=extra[:4]+struct.pack('<I',len(extra)-8)+extra[8:]
        asset=ingest_local(self.b.core,'alice',extra)
        request=dict(self.request,asset_id=asset['id'],out_tick=8)
        context=self.contexts.resolve_owned(self.b.core,'alice','voice_prepare',request)
        spec=execution(context)
        self.assertEqual(spec.reference_bytes,len(extra))
        self.assertEqual(spec.reference_sha256,asset['sha256'])
        self.assertEqual(selected_reference(spec,extra),(extra,4))
        path=self.b.core.root/'assets'/asset['id']
        self.assertEqual(_read(path,spec.reference_bytes,spec.reference_sha256),extra)
        with self.assertRaisesRegex(Refused,'file_limit'):
            _read(path,len(extra)-1,spec.reference_sha256)
        with self.assertRaisesRegex(Refused,'file_digest'):
            _read(path,len(extra),'0'*64)
        malformed=b'BAD!'+extra[4:]
        with self.assertRaisesRegex(Refused,'reference_format'):
            selected_reference(replace(spec,reference_sha256=hashlib.sha256(malformed).hexdigest()),malformed)
        with self.assertRaises(ValueError):
            ingest_local(self.b.core,'alice',malformed)
        oversized=raw[:36]+b'JUNK'+struct.pack('<I',1048576)+b'\0'*1048576+raw[36:]
        oversized=oversized[:4]+struct.pack('<I',len(oversized)-8)+oversized[8:]
        large_asset=ingest_local(self.b.core,'alice',oversized)
        with self.assertRaisesRegex(Refused,'file_limit'):
            self.contexts.resolve_owned(self.b.core,'alice','voice_prepare',dict(request,asset_id=large_asset['id']))
        self.request=request
        self.submit()
        authority=self.b.store.allocate(self.profiles,'qwen',1800)['authority']
        stage=self.b.stage/authority.job['attempts'][-1]['staging_id']
        stage.mkdir(mode=0o770)
        (stage/'reference.wav').write_bytes(selected_reference(spec,extra)[0])
        (stage/'audition.wav').write_bytes(raw)
        self.assertEqual((stage/'reference.wav').read_bytes(),extra)
        self.b.admission.check()
        self.b.admission.check(publication=True)
        (stage/'reference.wav').write_bytes(extra+b'\0')
        with self.assertRaisesRegex(ValueError,'resource_output_bytes'):
            self.b.admission.check()

    def test_prepare_audition_byte_cap_remains_enforced(self):
        self.submit()
        authority=self.b.store.allocate(self.profiles,'qwen',1800)['authority']
        spec=execution(authority.context)
        stage=self.b.stage/authority.job['attempts'][-1]['staging_id']
        stage.mkdir(mode=0o770)
        (stage/'reference.wav').write_bytes(self.b.raw)
        (stage/'audition.wav').write_bytes(b'\0'*(spec.max_output_bytes+1))
        with self.assertRaisesRegex(ValueError,'resource_output_bytes'):
            self.b.admission.check()

    def test_detached_pair_requires_exact_typed_owner_namespace_and_links(self):
        from service.reconciliation import Reconciliation, typed_references
        from service import recovery
        b=self.b;a,p,stage=self.proposal();self.finish(a,p,stage)
        scan=Reconciliation(b.core,b.stage)
        self.assertFalse(scan.reconcile_startup()['blocked'])
        data={table:[dict(zip([column[0] for column in cursor.description],r)) for r in cursor.fetchall()]
              for table in ('jobs','assets','artifacts','publications','publication_work','dispatches','dispatch_settings','resource_reservations')
              for cursor in [b.core.db.execute('SELECT * FROM '+table)]}
        recovery._installed_detached_documents(data)
        joined=typed_references(scan,data,None,None,oracle.SCHEMA)
        contexts=joined[3];owners=list(joined[5].values())
        rows=[r for r in scan.rows.values() if r['name'] in ('reference.wav','audition.wav')]
        self.assertEqual({r['name'] for r in rows},{'reference.wav','audition.wav'})
        for row in rows:
            self.assertTrue(recovery._installed_prepared_inventory(row,contexts,owners))
            for change in (dict(owner=None),dict(owner=dict(row['owner'],job_id='foreign')),
                           dict(namespace='worker_staging/foreign'),dict(namespace='assets'),
                           dict(name='conditioning'),dict(name='other.wav')):
                with self.subTest(file=row['name'],change=change):
                    self.assertFalse(recovery._installed_prepared_inventory(dict(row,**change),contexts,owners))
            alias=copy.deepcopy(row);alias['identity'][5]=2
            self.assertFalse(recovery._installed_prepared_inventory(alias,contexts,owners))
            self.assertFalse(recovery._installed_prepared_inventory(row,{},owners))
            changed=copy.deepcopy(contexts)
            changed[row['owner']['job_id']]['plan']['reference_roles']['reference_kind']='other'
            self.assertFalse(recovery._installed_prepared_inventory(row,changed,owners))

    def test_private_descriptor_preserves_mode_and_exact_hash_fences(self):
        from worker.engines.pocket import read_reference_fd
        from worker.transport import Refused
        path=self.b.root/'descriptor-regression.wav'
        path.write_bytes(self.b.raw);path.chmod(0o644)
        with path.open('rb') as stream:
            with self.assertRaisesRegex(Refused,'managed_reference_descriptor'):
                read_reference_fd(stream.fileno(),self.b.spec,owner_uid=os.getuid())
        path.chmod(0o600)
        with path.open('rb') as stream:
            self.assertEqual(read_reference_fd(stream.fileno(),self.b.spec,owner_uid=os.getuid()),self.b.raw)
        path.write_bytes(self.b.raw[:-1]+b'x')
        with path.open('rb') as stream:
            with self.assertRaises(Refused):
                read_reference_fd(stream.fileno(),self.b.spec,owner_uid=os.getuid())

    def test_pair_save_cas_reopen_generate_replay_and_original_budget(self):
        b=self.b
        first=self.submit();self.assertEqual(first,self.submit())
        self.assertEqual(b.core.db.execute('SELECT seconds FROM dispatch_settings WHERE job_id=?',(first['id'],)).fetchone()[0],1800)
        a,p,stage=self.proposal()
        self.assertIs(type(execution(a.context)),QwenPreparation)
        ref,audition=p['manifest']['outputs']
        self.assertEqual((stage/'reference.wav').read_bytes(),b.raw)
        self.assertEqual(selected_reference(execution(a.context),b.raw),(b.raw,4))
        self.assertEqual(self.calls[1],(self.request['audition_text'],b.raw))
        self.assertNotEqual(ref['sha256'],audition['sha256'])
        self.assertEqual(ref['provenance']['source_id'],b.asset['id'])
        self.finish(a,p,stage)
        production=Production(b.core);candidate=production.results('alice',first['id'])[0]
        args=('alice','save',first['id'],candidate['id'],'cpu-clone')
        saved=production.save(*args,expected_revision=0)
        self.assertEqual(saved,production.save(*args,expected_revision=0))
        with self.assertRaises(ValueError):production.save('alice','conflict',first['id'],candidate['id'],'cpu-clone',expected_revision=0)
        version=saved['version']
        from service.execution_context import cpu_clone_context
        request=dict(b.context['plan']['request'],voice_version_id=version['id'])
        for field,value in (('settings',dict(version['settings'],temperature=.5)),
                ('reference_text','unexpected transcript'),('identity',dict(version['identity'],model='foreign'))):
            changed=copy.deepcopy(version);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                cpu_clone_context(self.template,'alice',request,ref,version=changed)
        root=b.core.root;b.core.close();b.core=Core(root)
        b.admission=Admission(b.core,b.stage);b.store=DispatchStore(b.core,b.admission)
        self.assertEqual(b.core.voices.version('alice',version['id']),version)
        request=dict(b.context['plan']['request'],voice_version_id=version['id'],text='Frozen saved voice speech.')
        context=self.contexts.resolve_owned(b.core,'alice','speech_generate',request)
        job=Production(b.core).generate('alice','generate',context,_admission=b.admission,_profiles=self.profiles)
        self.assertEqual(job,Production(b.core).generate('alice','generate',context,_admission=b.admission,_profiles=self.profiles))
        a,p,stage=self.proposal(b.store.allocate(self.profiles,'qwen',1800)['authority'])
        self.assertEqual(self.calls[-2],(request['text'],b.raw))
        self.assertEqual(a.allowed_assets,(ref['id'],));self.assertNotIn(audition['id'],a.allowed_assets)
        self.assertEqual(p['manifest']['outputs'][0]['provenance']['source_id'],generated_source(a.context))
        self.finish(a,p,stage)
        self.assertEqual(b.core.job(job['id'])['state'],'succeeded')

    def test_owner_and_transcript_segment_identity_and_bytes_fences(self):
        from jsonschema.exceptions import ValidationError
        b=self.b
        for change in (dict(reference_text='ICL is excluded'),dict(in_tick=2),dict(out_tick=6),
                dict(audition_text=' '),dict(identity=dict(self.request['identity'],model='wrong'))):
            with self.subTest(change=change),self.assertRaises((ValueError,ValidationError)):
                self.resolve(dict(self.request,**change))
        with self.assertRaises(ValueError):self.resolve(principal='bob')
        foreign=ingest_local(b.core,'bob',b.raw)
        with self.assertRaises(ValueError):self.resolve(dict(self.request,asset_id=foreign['id']))
        path=b.core.root/'assets'/b.asset['id'];path.write_bytes(b.raw[:-1]+b'x')
        with self.assertRaises(ValueError):self.resolve()
        self.assertEqual(b.core.jobs('alice'),[])

    def test_supervisor_requires_explicit_template_and_rejects_settings_substitution(self):
        from worker.runtime_supervisor import RuntimeSupervisor
        b=self.b;context=self.resolve()
        supervisor=object.__new__(RuntimeSupervisor)
        supervisor.resident_qwen=False
        supervisor.qwen_contexts={('alice','qwen-cpu','managed-clone'):self.template}
        supervisor.qwen_profiles=self.profiles;supervisor.inventory=b.inventory;supervisor.bundles=b.mapping
        supervisor.failed=False
        with patch('worker.managed_reference.acquire',return_value=41),patch('worker.engines.qwen.verify_service_inputs') as verify,patch('os.close'):
            supervisor.verify_qwen(context,self.profiles,'alice')
            self.assertEqual(verify.call_args.kwargs['reference_fd'],41)
            self.assertEqual(supervisor.verify_qwen(context,self.profiles,'alice',retain_fd=True),41)
            changed=copy.deepcopy(context);changed['execution']['descriptor']['attention']='eager'
            with self.assertRaises(ValueError):supervisor.verify_qwen(changed,self.profiles,'alice')
            with self.assertRaises(ValueError):supervisor.verify_qwen(context,self.profiles,'bob')
        with patch('worker.managed_reference.acquire',side_effect=ValueError('unknown helper reap')):
            with self.assertRaises(ValueError):supervisor.verify_qwen(context,self.profiles,'alice')
        self.assertTrue(supervisor.failed)

    def test_public_http_cli_mcp_upload_pair_save_generate_and_reopen(self):
        from clients.http import HTTP
        from clients.cli import main
        from clients import mcp
        from service.api import create_voice_app
        from service.auth import Credential,StaticCredentials
        from service.dispatcher import Dispatcher
        from service.worker_transport import TransportObservation
        from worker.lifecycle import Observation,State
        from tests.release.test_native_managed_clone import ManagedCloneTests
        outer=self;b=self.b
        from types import SimpleNamespace
        clock=[0.0];durations=[]
        def monotonic():
            clock[0]+=.04
            return clock[0]
        self.enterContext(patch('service.dispatcher.time',SimpleNamespace(monotonic=monotonic)))
        # Select the installed read-poll policy through the existing injected seam;
        # this is still fake runtime evidence, with real production SQL tables.
        self.enterContext(patch.object(Dispatcher,'_qwen_state_poll',return_value=True))
        class Runtime:
            def call(inner,op,body):return dict(quarantined=False) if op=='hello' else dict(acked=True)
            def prepare(inner,a,fixture):inner.authority=a;return 'injected/cpu-xvector'
            def start(inner,identity):
                inner.started=clock[0];inner.polled=False
                return dict(state='alive')
            def observe(inner,identity,timeout):
                if not inner.polled:
                    inner.polled=True;clock[0]+=1800
                    durations.append(clock[0]-inner.started)
                    return Observation(identity,State.ALIVE)
                return Observation(identity,State.STOPPED)
            def collect(inner,a,identity):
                if a.context['plan']['request'].get('text')=='Failed speech':
                    return dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',reaped_children=1,timely=False,original_deadline=a.job['deadline']),None
                _,p,stage=outer.collected_proposal(a);(stage/'conditioning').rmdir()
                return dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',reaped_children=1,timely=True,original_deadline=a.job['deadline']),TransportObservation('proposal','injected',message=p['message'],manifest=p['manifest'])
        b.core.close();root=b.root/'core'
        receipts={};retained=None;identity=None
        for reopen in (False,True):
            dispatcher=Dispatcher(root,staging=b.stage,runtime=Runtime(),profiles=self.profiles,production_contexts=self.contexts)
            b.core=dispatcher.core;self.dispatcher=dispatcher
            try:
                app=create_voice_app(dispatcher,credentials=StaticCredentials([Credential('writer','alice',frozenset({'read','production-write'}))]))
                self.wait(lambda:dispatcher.call('status')['reconciled'])
                def complete(receipt,expected='succeeded'):
                    self.wait(lambda:dispatcher.call('job',receipt['job_id'])['state'] in ('succeeded','failed'))
                    self.assertEqual(dispatcher.call('job',receipt['job_id'])['state'],expected)
                    self.wait(lambda:dispatcher.call('status')['phase']=='idle')
                def cli(args):
                    output=io.StringIO()
                    with redirect_stdout(output):code=main(args)
                    self.assertEqual(code,0,output.getvalue());return json.loads(output.getvalue())
                def tool(op,payload,key):
                    result=asyncio.run(mcp.call_tool('speech_'+op,dict(payload,idempotency_key=key)))
                    self.assertFalse(result.isError,result.structuredContent);return result.structuredContent
                with ManagedCloneTests.bridge(self,app),patch.dict(os.environ,{'SPEECH_CREDENTIAL':'writer'}):
                    http=HTTP(credential='writer')
                    if not reopen:
                        upload=b.root/'upload.wav';upload.write_bytes(b.raw)
                        asset=http.upload_file(upload)
                        self.request['asset_id']=asset['id']
                    prep=http.call('voice_prepare',self.request,key='public-prepare')
                    self.assertEqual(prep,cli(['voice','prepare','--request',json.dumps(self.request),'--idempotency-key','public-prepare']))
                    self.assertEqual(prep,tool('voice_prepare',self.request,'public-prepare'));complete(prep)
                    candidates=http.call('voice_results',dict(job_id=prep['job_id']))
                    candidate=candidates[0] if isinstance(candidates,list) else candidates['candidates'][0]
                    save=dict(job_id=prep['job_id'],candidate_id=candidate['id'],voice_id='public-clone',expected_revision=0)
                    saved=http.call('voice_save',save,key='public-save')
                    self.assertEqual(saved,tool('voice_save',save,'public-save'))
                    request=dict(b.context['plan']['request'],voice_version_id=saved['version']['id'],text='Public frozen speech.')
                    generated=http.call('speech_generate',request,key='public-generate');complete(generated)
                    failed=http.call('speech_generate',dict(request,text='Failed speech'),key='failed');complete(failed,'failed')
                    failed_job=dispatcher.call('job',failed['job_id'])
                    retried=http.call('job_retry',dict(job_id=failed_job['id'],input_sha256=failed_job['input_sha256']),key='retried');complete(retried,'failed')
                    extra=http.call('speech_generate',dict(request,text='Additional speech'),key='extra');complete(extra)
                    current=dict(prep=prep,saved=saved,generated=generated,failed=failed,retried=retried,extra=extra)
                    if reopen:self.assertEqual(current,receipts)
                    else:receipts=current
            finally:
                database=root/'core.sqlite'
                dispatcher.close()
            counts=self.retained_tables(database)
            self.assertLessEqual(sum(counts.values()),2048)
            self.assertEqual(len(durations),5)
            self.assertTrue(all(duration>=1800 for duration in durations))
            self.assertEqual(counts['jobs'],5)
            self.assertEqual(counts['dispatch_commands'],25)
            current_identity=self.retained_identity(database)
            if reopen:
                self.assertEqual(current_identity,identity)
                growth={name:counts[name]-retained[name] for name in counts}
                self.assertEqual(growth['p1_inventory_scans'],1)
                self.assertGreater(growth['p1_inventory_objects'],0)
                self.assertGreater(growth['p1_inventory_history'],0)
                self.assertEqual(sum(counts.values())-sum(retained.values()),
                                 sum(growth[name] for name in ('p1_inventory_scans',
                                     'p1_inventory_objects','p1_inventory_history')))
            else:retained=counts;identity=current_identity

    def test_cancel_preparation_and_generation_wait_for_reap_ack_then_retry_after_reopen(self):
        import threading
        from service.dispatcher import Dispatcher
        from service.worker_transport import TransportObservation
        from worker.lifecycle import Observation,State
        b=self.b;outer=self
        class Runtime:
            def __init__(inner,finish=False):
                inner.ready=threading.Event();inner.stopped=threading.Event();inner.finished=threading.Event()
                if finish:inner.finished.set()
                inner.cancel=False;inner.fail=False;inner.acks=0
            def call(inner,op,body):
                if op=='ack':inner.acks+=1
                return dict(quarantined=False) if op=='hello' else dict(acked=True)
            def prepare(inner,a,fixture):
                inner.authority=a;inner.ready.set();return 'injected/'+a.job['id']
            def start(inner,identity):return dict(state='alive')
            def observe(inner,identity,timeout):return Observation(identity,State.STOPPED if inner.finished.is_set() else State.ALIVE)
            def terminate(inner,identity,timeout):inner.cancel=True;inner.stopped.set()
            def collect(inner,a,identity):
                proposal=None
                if not inner.cancel and not inner.fail:
                    _,p,stage=outer.collected_proposal(a);(stage/'conditioning').rmdir()
                    proposal=TransportObservation('proposal','injected',message=p['message'],manifest=p['manifest'])
                return dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',reaped_children=1,timely=not inner.cancel,original_deadline=a.job['deadline']),proposal
        # First publish/save a real injected preparation pair for generation scope.
        a,p,stage=self.proposal();self.finish(a,p,stage)
        candidate=Production(b.core).results('alice',a.job['id'])[0]
        version=Production(b.core).save('alice','seed',a.job['id'],candidate['id'],'cancel-clone',expected_revision=0)['version']
        requests=[('production_prepare',self.request),('production_generate',dict(b.context['plan']['request'],voice_version_id=version['id']))]
        root=b.core.root;b.core.close()
        for operation,request,outcome in [(op,req,outcome) for op,req in requests for outcome in ('cancelled','failed')]:
            suffix=operation+'-'+outcome
            runtime=Runtime();runtime.fail=outcome=='failed'
            dispatcher=Dispatcher(root,staging=b.stage,runtime=runtime,profiles=self.profiles,production_contexts=self.contexts)
            b.core=dispatcher.core;self.dispatcher=dispatcher
            wait=self.wait
            try:
                wait(lambda:dispatcher.call('status')['reconciled'])
                receipt=dispatcher.call(operation,'alice',request,key='terminal-'+suffix)
                wait(runtime.ready.is_set)
                if outcome=='cancelled':
                    dispatcher.call('production_cancel','alice',dict(job_id=receipt['job_id']),key='cancel-intent-'+suffix)
                    wait(runtime.stopped.is_set)
                job=dispatcher.call('job',receipt['job_id'])
                self.assertEqual(job['state'],'cancel_requested' if outcome=='cancelled' else 'running')
                self.assertFalse(job['attempts'][-1]['reaped']);self.assertEqual(runtime.acks,0)
                retry=dict(job_id=job['id'],input_sha256=job['input_sha256'])
                with self.assertRaises(ValueError):dispatcher.call('production_retry','alice',retry,key='retry-'+suffix)
                runtime.finished.set()
                wait(lambda:dispatcher.call('job',job['id'])['state']==outcome)
                wait(lambda:dispatcher.call('status')['phase']=='idle')
                self.assertEqual(dispatcher.call('job',job['id'])['artifact_ids'],[])
                self.assertGreater(runtime.acks,0)
                frozen=runtime.authority.context
            finally:
                database=root/'core.sqlite'
                dispatcher.close()
            before_retry=self.retained_tables(database)
            self.assertLessEqual(sum(before_retry.values()),2048)
            resumed=Runtime(finish=True)
            dispatcher=Dispatcher(root,staging=b.stage,runtime=resumed,profiles=self.profiles,production_contexts=self.contexts);b.core=dispatcher.core;self.dispatcher=dispatcher
            try:
                wait(lambda:dispatcher.call('status')['reconciled'])
                self.assertTrue(dispatcher.call('status')['resource_activation'],dispatcher.call('status'))
                receipt=dispatcher.call('production_retry','alice',retry,key='retry-'+suffix)
                self.assertEqual(receipt,dispatcher.call('production_retry','alice',retry,key='retry-'+suffix))
                wait(resumed.ready.is_set)
                self.assertEqual(resumed.authority.context,frozen)
                wait(lambda:dispatcher.call('job',receipt['job_id'])['state']=='succeeded')
                wait(lambda:dispatcher.call('status')['phase']=='idle')
            finally:dispatcher.close()
            after_retry=self.retained_tables(database)
            self.assertEqual(set(before_retry),set(after_retry))
            self.assertGreater(sum(after_retry.values()),sum(before_retry.values()))
            self.assertLessEqual(sum(after_retry.values()),2048)


class CPUConditioningMonitorTests(CPUPreparationFixture):
    def test_exact_empty_removal_race_is_bounded_to_managed_qwen(self):
        b=self.b;self.submit();a=b.store.allocate(self.profiles,'qwen',1800)['authority']
        stage=b.stage/a.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
        child=stage/'conditioning';child.mkdir(mode=0o770);child.chmod(0o770)
        original=os.open;hit=[]
        from tests.worker.test_runtime_conditioning import filesystem_policy
        from worker.runtime_conditioning import remove_empty
        def interleave(path,flags,*args,**kwargs):
            if path=='conditioning' and not hit:
                hit.append(True)
                with filesystem_policy(b.stage,stage,mutate=lambda role,values:values.update(st_uid=10001) if role=='child' else None):
                    remove_empty(stage.name,20000,managed_namespace=True)
            return original(path,flags,*args,**kwargs)
        high=b.core.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0]
        with patch('service.resource_store.os.open',side_effect=interleave):b.admission.check()
        self.assertTrue(hit);self.assertFalse(child.exists())
        self.assertEqual(b.admission.store.state(),'healthy')
        self.assertGreaterEqual(b.core.db.execute('SELECT highwater FROM resource_accounting').fetchone()[0],high)
        with self.assertRaises(ValueError):
            with b.core.transaction():b.admission.store.release_tx(a.job['id'])

    def test_foreign_directory_disappearance_and_substitution_remain_faults(self):
        for scenario in ('foreign','substitution','nonempty'):
            case=CPUPreparationFixture();case.setUp();self.addCleanup(case.doCleanups)
            b=case.b;case.submit();a=b.store.allocate(case.profiles,'qwen',1800)['authority']
            stage=b.stage/a.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
            name='foreign' if scenario=='foreign' else 'conditioning'
            child=stage/name;child.mkdir(mode=0o770)
            if scenario=='nonempty':(child/'retained').write_bytes(b'opaque')
            original=os.open;hit=[]
            def interleave(path,flags,*args,**kwargs):
                if path==name and not hit:
                    hit.append(True)
                    if scenario=='substitution':
                        child.rename(stage/'retained');child.mkdir(mode=0o770)
                    else:child.rmdir()
                return original(path,flags,*args,**kwargs)
            with patch('service.resource_store.os.open',side_effect=interleave):
                with self.assertRaises((ValueError,OSError)):b.admission.check()
            self.assertEqual(b.admission.store.state(),'fault')
