"""Qwen Linux lifecycle: inert assets, injected backend/transport, no ML/processes."""
import copy
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import wave
from unittest.mock import Mock, patch
from types import SimpleNamespace
from contracts import validate as oracle
from lib.tts_backends.contract import AudioResult
from service.admission import Admission
from service.core import Core
from service.dispatch_store import DispatchStore
from service.execution_context import QwenContexts
from service.ingest import ingest_local
from service.production import Production
from worker.engines.qwen import execute as engine_execute
from worker.engines.qwen_execution import QwenExecution, inventory_wire
from worker.hardware import HardwareInventory, Host, Device, DeviceFamily
from worker.runtime_child import run_generation
from worker.runtime_protocol import execution, generated_source
from worker.transport import Refused
from tests.service.test_pocket_lifecycle import assert_configuration_mutations


class QwenLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        # This fixture exercises the retained installed Qwen v1 history family.
        self.core=Core(self.root/'core',_publication_creation=1);self.addCleanup(lambda:self.core.close())
        self.stage=self.root/'stage';self.stage.mkdir()
        self.admission=Admission(self.core,self.stage);self.store=DispatchStore(self.core,self.admission)
        out=io.BytesIO()
        with wave.open(out,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','none'));w.writeframes(b'\0\0'*4)
        self.raw=out.getvalue();self.asset=ingest_local(self.core,'alice',self.raw)
        self.bundle=self.root/'bundle';self.bundle.mkdir()
        self.configure(getattr(self,'task','clone'),getattr(self,'device','cpu'))

    def configure(self, task='clone', device='cpu'):
        for p in self.bundle.iterdir():p.chmod(0o600)
        self.bundle.chmod(0o700)
        data={'config.json':json.dumps(dict(model_type='qwen3_tts',tts_model_type='base' if task == 'clone' else 'custom_voice',tts_model_size='1b7')).encode(),
              'generation_config.json':b'{}','model.safetensors':b'inert'}
        entries=[]
        for name,raw in data.items():
            (self.bundle/name).write_bytes(raw);entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        raw=json.dumps(dict(schema_version=1,backend='qwen',checkpoint='test',revision='rev1',language='English',files=entries,
             execution=dict(sample_rate=24000,languages=['English'],presets=[] if task == 'clone' else ['Ryan'],devices=[dict(device=device,dtype='float32',attention='sdpa')]))).encode()
        (self.bundle/'manifest.json').write_bytes(raw)
        for p in self.bundle.iterdir():p.chmod(0o400)
        self.bundle.chmod(0o500)
        refs=dict(reference_id=self.asset['id'],reference_sha256=self.asset['sha256'],reference_bytes=len(self.raw),reference_frames=4,reference_rate=24000,reference_text='Original saved transcript.') if task == 'clone' else dict(preset='Ryan')
        self.spec=QwenExecution('bundle',hashlib.sha256(raw).hexdigest(),'test','rev1',task,'English','voice.0','vv0',2,1024,100,244,192*1024**2,24000,device=device,**refs)
        self.inventory=HardwareInventory(Host('linux','x86_64','docker'),(Device(DeviceFamily.CUDA if device.startswith('cuda:') else DeviceFamily.CPU,int(device.split(':')[1]) if ':' in device else 0,True,True,datetime.now(timezone.utc).isoformat()),))
        fixture=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
        self.context=fixture['context'];plan=self.context['plan'];r=plan['resolved']
        ident=dict(engine='qwen',model='test',bundle_sha256=self.spec.manifest_sha256,device='cuda' if ':' in device else 'cpu',precision='float32')
        mode='qwen_clone' if task == 'clone' else 'qwen_preset'
        settings=dict(kind=mode,language='English',seed=None,temperature=.9,top_p=1.,top_k=50,repetition_penalty=1.05,max_new_tokens=2048,instruction=None)
        config=dict(mode=mode,reference_asset_id=self.asset['id'],reference_text=self.spec.reference_text,conditioning='icl') if task == 'clone' else dict(mode=mode,speaker='Ryan')
        v=plan['voice_versions'][0]
        v.update(origin='clone' if task == 'clone' else 'preset',identity=ident,settings=settings,settings_sha256=oracle.canonical_hash(settings),configuration=config,
                 reference_asset_ids=[self.asset['id']] if task == 'clone' else [],reference_text=self.spec.reference_text or '')
        if task == 'clone':v['reference_asset_sha256']={self.asset['id']:self.asset['sha256']}
        self.version=copy.deepcopy(v)
        self.core.voices.register('alice','saved-'+task+device.replace(':','-'),v,expected_revision=0)
        plan['request'].update(identity=ident,text=' Exact\n authored text ')
        plan['assets']=[self.asset] if task == 'clone' else []
        p=fixture['profiles'][0]
        pid='qwen-cuda' if ':' in device else 'qwen-cpu'
        p.update(id=pid,device=ident['device'],support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='INJECTED-ONLY-NOT-ACTUAL')
        p['compatibility']=[dict(operation='speech_generate',mode=mode,identity=ident)]
        p['limits'].update(ram_bytes=self.spec.memory_bytes,disk_bytes=2*1024**2,threads=2,vram_bytes=4096 if ':' in device else 0)
        r.update(identity=dict(requested=ident,effective=ident,fallback='forbid',reason='',profile_id=pid),profile=oracle.profile_snapshot(p),settings=settings,settings_sha256=oracle.canonical_hash(settings))
        self.context['bundles']=[dict(instructions=True,engine='qwen',model='test',mode=mode,bundle_sha256=self.spec.manifest_sha256,languages=['English'],selectors=[] if task == 'clone' else [dict(value='Ryan',languages=['English'])])]
        self.refs={self.asset['id']:dict(path=self.core.root/'assets'/self.asset['id'],voice_id='voice.0',voice_revision='vv0',checkpoint='test',revision='rev1',text=self.spec.reference_text)} if task == 'clone' else {}
        wire_refs=copy.deepcopy(self.refs)
        for k in wire_refs:wire_refs[k]['path']='/references/assets/'+k
        self.context['execution']=dict(descriptor=self.spec.identity(),bundles={'bundle':'/models/bundles/bundle'},references=wire_refs,hardware=inventory_wire(self.inventory))
        self.profiles=[p];self.mapping={'bundle':self.bundle}
        oracle.frozen_plan(plan,self.context['bundles'])
        self.contexts=self.context_factory()

    def context_factory(self):
        return QwenContexts({'alice':self.context},self.mapping,self.refs,self.inventory,self.profiles)

    def submit(self,key='generate'):
        c=self.contexts.resolve('alice','speech_generate',self.context['plan']['request'])
        return Production(self.core).generate('alice',key,c,_admission=self.admission,_profiles=self.profiles)

    def create_stage(self, stage):
        # Publish the exact mode without changing the writer's global umask.
        with tempfile.TemporaryDirectory(dir=self.root) as private:
            pending=Path(private)/'stage'
            pending.mkdir(mode=0o700)
            pending.chmod(0o770)
            pending.rename(stage)

    def proposal(self, *, authority=None):
        if authority is None:
            self.submit();authority=self.store.allocate(self.profiles,'qwen',8)['authority']
        stage=self.stage/authority.job['attempts'][-1]['staging_id'];self.create_stage(stage)
        self.calls=[];sent=[];outer=self
        class Backend:
            def __init__(inner,**kw):outer.calls.append(kw)
            def synthesise_clone(inner,params,path,text,x_vector,timeout):
                outer.calls.append((params,Path(path).read_bytes(),text))
                return AudioResult([0.,.5,-.5],24000,metadata=dict(checkpoint=str(outer.bundle),revision='rev1'))
            def synthesise_preset(inner,params,preset,instruct,timeout):
                outer.calls.append((params,preset,instruct))
                return AudioResult([0.,.5,-.5],24000,metadata=dict(checkpoint=str(outer.bundle),revision='rev1'))
            def close(inner):outer.calls.append('closed')
        def execute(spec,text,bundles,references,inventory,**kw):
            self.assertEqual(bundles,authority.context['execution']['bundles']);self.assertEqual(references,authority.context['execution']['references'])
            kw['managed_owner_uid']=os.getuid() # fixture-only ownership injection
            return engine_execute(spec,text,self.mapping,self.refs,inventory,**kw)
        config=dict(job=authority.job,context=authority.context,allowed_assets=list(authority.allowed_assets),profiles=list(authority.profiles),secret='a'*64,control_uid=os.getuid(),gid=os.getgid())
        with patch('worker.transport.authenticate') as auth,patch('worker.transport.receive',return_value=authority.message()),patch('worker.transport.send',side_effect=lambda c,v,t:sent.append(v)),patch('worker.engines.qwen.execute',side_effect=execute):
            run_generation(None,config,self.stage,time.monotonic()+5,backend_factory=Backend)
            self.assertEqual(auth.call_args.args[1].binding,authority.binding)
        return authority,sent[0],stage

    def test_saved_reference_exact_forwarding_publication_and_restart(self):
        a,p,stage=self.proposal()
        self.assertEqual(self.calls[1][1:],(self.raw,'Original saved transcript.'))
        self.assertEqual(self.calls[1][0],dict(text=' Exact\n authored text ',language='English',temperature=.9,top_p=1.,top_k=50,repetition_penalty=1.05,max_new_tokens=2048))
        self.assertEqual(self.calls[0]['device'],'cpu');self.assertEqual(self.calls[-1],'closed')
        output=p['manifest']['outputs'][0]
        self.assertEqual(output['provenance']['source_id'],generated_source(a.context))
        self.core._publish(a.job['id'],p,stage_root=self.stage,engine_uid=os.getuid())
        self.assertEqual(self.admission.store.rows()[0]['state'],'executing')
        with self.core.transaction():self.admission.store.release_tx(a.job['id'],confirmed_reap=True)
        root=self.core.root;self.core.close();self.core=Core(root)
        self.admission=Admission(self.core,self.stage);self.store=DispatchStore(self.core,self.admission)
        self.assertEqual(self.core.voices.version('alice','vv0'),self.version)
        second=self.submit('again');self.assertEqual(second,self.submit('again'))
        self.assertEqual(self.core._load(second['id'])[1]['execution'],a.context['execution'])

    def test_refusals_before_allocation(self):
        mutations=[lambda c:c['execution']['descriptor'].update(voice_revision='other'),lambda c:c['execution']['descriptor'].update(attention='eager'),
                   lambda c:c['execution']['descriptor'].update(reference_sha256='0'*64),lambda c:c['execution']['references'][self.asset['id']].update(path='/tmp/public'),
                   lambda c:c['execution']['descriptor'].update(task='voice_design'),lambda c:c['plan']['request'].update(voice_version_id='other')]
        for mutate in mutations:
            c=copy.deepcopy(self.context);mutate(c)
            with self.assertRaises(ValueError):QwenContexts({'alice':c},self.mapping,self.refs,self.inventory,self.profiles)
        with self.assertRaises(ValueError):self.contexts.resolve('bob','speech_generate',self.context['plan']['request'])
        with self.assertRaises(ValueError):Production(self.core).generate('bob','foreign',self.context,_admission=self.admission,_profiles=self.profiles)
        for field in ('execution','context','reference_path','capability_receipt'):
            request=dict(self.context['plan']['request'],**{field:'/tmp/public'})
            with self.assertRaises(Exception):self.contexts.resolve('alice','speech_generate',request)
        self.refs[self.asset['id']]['path'].write_bytes(b'tampered')
        with self.assertRaises(ValueError):self.submit()
        for table in ('jobs','events','dispatch_settings','resource_reservations'):
            self.assertEqual(self.core.db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)

    def test_stale_probe_refuses_resolution_and_dispatch_without_expiring_history(self):
        self.submit()
        wire=self.context['execution']['hardware'];wire['devices'][0]['probed_at']=(datetime.now(timezone.utc)-timedelta(seconds=301)).isoformat()
        from worker.engines.qwen_execution import inventory_from_wire
        self.inventory=inventory_from_wire(wire)
        with self.assertRaisesRegex(ValueError,'stale'):self.context_factory()
        self.core.db.execute('UPDATE jobs SET context=?',(json.dumps(self.context),))
        self.assertEqual(execution(self.context),self.spec) # detached history is timeless
        with self.assertRaisesRegex(ValueError,'stale'):self.store.allocate(self.profiles,'qwen',8)
        self.assertEqual(self.admission.store.rows()[0]['state'],'queued')
        self.assertEqual(self.core.jobs('alice')[0]['job']['generation'],0)

    def test_supervisor_independent_registration_and_prejournal_verification(self):
        self.submit();a=self.store.allocate(self.profiles,'qwen',8)['authority']
        from worker.runtime_supervisor import RuntimeSupervisor
        s=RuntimeSupervisor.__new__(RuntimeSupervisor)
        s.resident_qwen=False
        s.bundles=a.context['execution']['bundles'];s.references=a.context['execution']['references'];s.inventory=self.inventory
        s.qwen_profiles=self.profiles;s.qwen_contexts={}
        body=dict(snapshot=json.loads(a._snapshot),binding=a.binding,fixture='qwen')
        with self.assertRaisesRegex(ValueError,'unregistered'):s.operation('prepare',body)
        s.qwen_contexts={('alice','qwen-cpu','vv0'):a.context['execution']}
        with patch('worker.engines.qwen.verify_service_inputs',side_effect=Refused('before-journal')) as check:
            with self.assertRaisesRegex(ValueError,'before-journal'):s.operation('prepare',body)
            check.assert_called_once()
        with self.assertRaisesRegex(ValueError,'unregistered'):s.verify_qwen(a.context,self.profiles,'bob')
        s.inventory=replace(self.inventory,devices=())
        with self.assertRaisesRegex(ValueError,'unregistered'):s.operation('prepare',body)

    def test_cancel_pending_retains_capacity_and_nonempty_namespace(self):
        a,p,stage=self.proposal();namespace=stage/'conditioning';(namespace/'opaque').write_bytes(b'retained')
        self.core.request_cancel(a.job['id'])
        from service.dispatcher import Dispatcher
        owner=Dispatcher.__new__(Dispatcher);owner.core=self.core;owner._fail(self.core.job(a.job['id']),'cancelled')
        with self.assertRaises(ValueError):self.core._publish(a.job['id'],p,stage_root=self.stage,engine_uid=os.getuid())
        with self.assertRaises(ValueError):
            with self.core.transaction():self.admission.store.release_tx(a.job['id'])
        self.assertEqual(self.admission.store.rows()[0]['state'],'executing')
        from service.reconciliation import Reconciliation
        self.assertTrue(Reconciliation(self.core,self.stage).reconcile_startup()['blocked'])
        from tests.worker.test_runtime_conditioning import filesystem_policy
        from worker.runtime_conditioning import remove_empty
        with filesystem_policy(self.stage,stage):
            with self.assertRaises(OSError):remove_empty(stage.name,20000)
        self.assertEqual((namespace/'opaque').read_bytes(),b'retained')

    def test_limits_before_child_backend_and_exact_cuda_reservation(self):
        from worker.runtime_child import main,contain_kokoro
        import resource
        with patch.dict(os.environ),patch('resource.setrlimit') as limits:
            contain_kokoro(self.context)
            self.assertEqual(limits.call_args_list[0].args,(resource.RLIMIT_AS,(self.spec.memory_bytes,)*2))
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'2')
        calls=[];anchor=time.monotonic()+8
        config=dict(fixture='qwen',context=self.context,qwen_cpu_cutoff=anchor,
                    job=dict(deadline=(datetime.now(timezone.utc)+timedelta(seconds=8)).isoformat(),
                             attempts=[dict(staging_id='stage')]))
        with patch('sys.argv',['worker','3']),patch('worker.runtime_child.socket.socket'),patch('worker.runtime_child.receive',return_value=config),patch('worker.runtime_child.contain_kokoro',side_effect=lambda c:calls.append('limits')),patch('worker.runtime_child.run_generation',side_effect=lambda *a:(self.assertLessEqual(a[3],anchor),calls.append('generation'))),patch('worker.runtime_child.os.chmod'):
            main()
        self.assertEqual(calls,['limits','generation'])

    def assert_closed_history(self, root, runtime, history):
        """Actual complete detached reader over the completed Pocket DB/inventory.

        Supply test enrollment and injected ownership, without changing installed
        configuration or any production reader. No historical/installed root.
        """
        import sqlite3
        from service import recovery
        from worker.runtime_journal import journal_bytes
        import shutil
        from service.reconciliation import Reconciliation
        authority=runtime.authority
        target=history.roots['control']/'core'
        # Copy only this small owned fixture's DB and explicitly named assets;
        # no production root is enrolled or altered for the reader test.
        with sqlite3.connect(target/'core.sqlite') as enrolled:
            enrollment={table:enrolled.execute('SELECT * FROM '+table).fetchall()
                        for table in ('p1_installation','p1_registrations')}
            schemas=[(n,sql) for n,sql in enrolled.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            with sqlite3.connect(root/'core.sqlite') as db:db.backup(enrolled)
            for name,sql in schemas:
                if not enrolled.execute("SELECT 1 FROM sqlite_master WHERE name=?",(name,)).fetchone():enrolled.execute(sql)
            for table,rows in enrollment.items():
                enrolled.executemany('INSERT INTO '+table+' VALUES ('+','.join('?' for _ in rows[0])+')',rows)
            receipt=json.loads(enrolled.execute('SELECT receipt FROM dispatches').fetchone()[0])
            asset_ids=[r[0] for r in enrolled.execute('SELECT id FROM assets')]
        for identity in asset_ids:
            shutil.copyfile(root/'assets'/identity,target/'assets'/identity)
            (target/'assets'/identity).chmod(0o600)
        stage_id=authority.job['attempts'][-1]['staging_id']
        stage=history.roots['staging']/stage_id;stage.mkdir(mode=0o770)
        if (self.stage/stage_id/'output.wav').exists():
            shutil.copyfile(self.stage/stage_id/'output.wav',stage/'output.wav')
        journal=dict(version=1,lane=history.lane,records=[dict(id=self.history_runtime_id,
            allocation=dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture='qwen'),
            phase='terminal',acked=True,uid=20000,boot=history.hello['boot'],receipt=receipt,proposal=None)])
        (history.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal))
        with patch.object(os,'getuid',return_value=10001):
            core=Core(target)
            try:
                scan=Reconciliation(core,history.roots['staging'])
                summary=scan.reconcile_startup()
                self.assertFalse(summary['blocked'],dict(summary=summary,rows=list(scan.rows.values())))
                for identity,row in scan.rows.items():
                    persisted=json.loads(core.db.execute('SELECT record FROM p1_inventory_objects WHERE id=?',(identity,)).fetchone()[0])
                    latest=json.loads(core.db.execute('SELECT record FROM p1_inventory_history WHERE object_id=? ORDER BY seq DESC LIMIT 1',(identity,)).fetchone()[0])
                    self.assertEqual(row,persisted);self.assertEqual(row,latest)
            finally:core.close()
        with patch.object(os,'getuid',return_value=10001), sqlite3.connect(target/'core.sqlite',isolation_level=None) as db:
            from service import install_policy as ip
            state=recovery.snapshot(history.roots,db,history.scope,policy=ip.SPLIT)
        state=json.loads(json.dumps(state))
        recovery._installed_detached_source(state,journal)
        self.assertTrue(state['allocation_coherent'])
        self.assertTrue(recovery.installed_clean_history(state,journal))
        assert_configuration_mutations(self,state,journal)
        recovery._validate_diagnostic_eligibility(state['evidence_projection'],state['affected'],[])
        self.assertFalse(state['inventory']['reasons'])

    def lifecycle_diagnostics(self):
        """Retain exceptions before Dispatcher discards their cause; always re-raise."""
        import traceback
        captures=[]
        def instrument(owner,name):
            original=getattr(owner,name)
            def capture(*args,**kwargs):
                try:return original(*args,**kwargs)
                except Exception as exc:
                    record=dict(where=owner.__name__+'.'+name,exception=type(exc).__name__,
                                message=str(exc),traceback=traceback.format_exc())
                    captures.append(record)
                    print('QWEN_DIAGNOSTIC '+json.dumps(record),flush=True)
                    raise
            self.enterContext(patch.object(owner,name,new=capture))
        instrument(Admission,'check');instrument(Core,'_publish')
        # The dispatcher's own resource_violation is outside these three calls.
        # Observe its still-active exception at the terminal mapping boundary.
        from service.dispatcher import Dispatcher
        import sys
        original_fail=Dispatcher._fail
        def failing(dispatcher,job,stage):
            if stage=='verification_failed':
                exc=sys.exception()
                record=dict(where='Dispatcher.verification_failed',
                            exception=type(exc).__name__,message=str(exc),
                            traceback=''.join(traceback.format_exception(exc)))
                captures.append(record)
                print('QWEN_DIAGNOSTIC '+json.dumps(record),flush=True)
            return original_fail(dispatcher,job,stage)
        self.enterContext(patch.object(Dispatcher,'_fail',new=failing))
        return captures,instrument

    def assert_scanner_interleavings(self):
        """Force each namespace race once using an actual CPU ICL allocation."""
        t=QwenLifecycleTests();t.setUp()
        try:
            t.submit();a=t.store.allocate(t.profiles,'qwen',8)['authority']
            stage=t.stage/a.job['attempts'][-1]['staging_id']
            t.create_stage(stage)
            namespace=stage/'conditioning';namespace.mkdir(mode=0o700)
            original=os.open;seen=[]
            def removing(path,*args,**kwargs):
                if path=='conditioning' and kwargs.get('dir_fd') is not None:
                    namespace.rmdir();seen.append('removed-before-open')
                return original(path,*args,**kwargs)
            with patch.object(os,'open',side_effect=removing):
                t.admission.store._measure_staging(t.stage)
            self.assertEqual(seen,['removed-before-open'])
            print('QWEN_RACE '+json.dumps(dict(interleaving=seen,accepted=True)),flush=True)
            stage.chmod(0o750);seen=[]
            def changing(path,*args,**kwargs):
                if path==stage.name and kwargs.get('dir_fd') is not None:
                    stage.chmod(0o770);seen.append('chmod-between-stat-and-open')
                return original(path,*args,**kwargs)
            with patch.object(os,'open',side_effect=changing):
                with self.assertRaisesRegex(ValueError,'^resource_namespace_changed$') as caught:
                    t.admission.store._measure_staging(t.stage)
            self.assertEqual(seen,['chmod-between-stat-and-open'])
            print('QWEN_RACE '+json.dumps(dict(interleaving=seen,cause=str(caught.exception))),flush=True)
        finally:t.doCleanups()

    def test_stage_published_only_after_exact_mode_is_set(self):
        stage=self.stage/'atomic-stage';original=Path.chmod;seen=[]
        def changing(path,mode,*args,**kwargs):
            if path.name=='stage' and path.parent!=self.root:
                self.assertFalse(stage.exists())
                self.admission.store._measure_staging(self.stage)
                seen.append('private-mode-change')
            return original(path,mode,*args,**kwargs)
        with patch.object(Path,'chmod',new=changing):self.create_stage(stage)
        self.assertEqual(seen,['private-mode-change'])
        self.assertEqual(stage.stat().st_mode & 0o777,0o770)
        self.assertEqual(stage.stat().st_uid,os.getuid())
        self.admission.store._measure_staging(self.stage)

    def test_icl_scanner_refuses_namespace_substitution_and_payload_loss(self):
        for defect in ('inode','mode','nonempty','file'):
            with self.subTest(defect=defect):
                t=QwenLifecycleTests();t.setUp()
                try:
                    t.submit();a=t.store.allocate(t.profiles,'qwen',8)['authority']
                    stage=t.stage/a.job['attempts'][-1]['staging_id']
                    t.create_stage(stage)
                    namespace=stage/'conditioning';namespace.mkdir(mode=0o700)
                    if defect=='mode':namespace.chmod(0o750)
                    if defect in ('nonempty','file'):(namespace/'unexpected').write_bytes(b'retain')
                    original=os.open
                    def change(path,*args,**kwargs):
                        if path=='conditioning' and kwargs.get('dir_fd') is not None:
                            if defect=='inode':
                                namespace.rename(stage/'substituted')
                                namespace.mkdir(mode=0o700)
                            elif defect=='file':
                                (namespace/'unexpected').unlink()
                                # A disappearing file has no directory waiver.
                                raise FileNotFoundError('unexpected payload removal')
                            else:namespace.rmdir()
                        return original(path,*args,**kwargs)
                    with patch.object(os,'open',side_effect=change):
                        with self.assertRaises((ValueError,OSError)):
                            t.admission.store._measure_staging(t.stage)
                    if defect=='nonempty':self.assertEqual((namespace/'unexpected').read_bytes(),b'retain')
                finally:t.doCleanups()

    def test_dispatcher_generate_artifact_and_cancel_reap(self):
        self.assert_scanner_interleavings()
        captures,instrument=self.lifecycle_diagnostics()
        from service.dispatcher import Dispatcher
        from service.worker_transport import TransportObservation
        from worker.lifecycle import Observation, State
        import threading
        outer=self
        class Runtime:
            def __init__(inner):
                inner.state=State.ALIVE;inner.authority=None
                inner.finish=threading.Event();inner.stopped=threading.Event();inner.cancel=False
            def call(inner,op,body):
                return dict(quarantined=False) if op=='hello' else dict(acked=True)
            def prepare(inner,authority,fixture):
                outer.assertEqual(fixture,'qwen');inner.authority=authority
                outer.history_runtime_id=history.lane+'/'+history.hello['boot']+'/'+hashlib.sha256(authority.job['id'].encode()).hexdigest()[:32]
                return outer.history_runtime_id
            def start(inner,identity):return dict(state='alive')
            def observe(inner,identity,timeout):
                return Observation(identity,State.STOPPED if inner.finish.is_set() else State.ALIVE)
            def terminate(inner,identity,timeout):inner.cancel=True;inner.stopped.set()
            def collect(inner,authority,identity):
                receipt=dict(runtime_id=identity,binding=authority.binding,reaped=True,engine_uid=20000,reap_method='pid1-waitpid-echild',reaped_children=0,timely=not inner.cancel,original_deadline=authority.job['deadline'])
                from tests.worker.test_runtime_conditioning import filesystem_policy
                from worker.runtime_conditioning import remove_empty
                stage=outer.stage/authority.job['attempts'][-1]['staging_id']
                if inner.cancel:
                    (stage/'conditioning').rmdir() if (stage/'conditioning').exists() else None
                    return receipt,None
                _,proposal,stage=outer.proposal(authority=authority)
                (stage/'conditioning').rmdir() if (stage/'conditioning').exists() else None
                return receipt,TransportObservation('proposal','injected-test-only',message=proposal['message'],manifest=proposal['manifest'])
        instrument(Runtime,'collect')
        def wait(predicate):
            end=time.monotonic()+5
            while time.monotonic()<end:
                value=predicate()
                if value:return value
                time.sleep(.02)
            self.fail(dict(reason='bounded dispatcher wait expired',captures=captures,status=dispatcher.call('status')))
        # Separate persistent owner root, seeded with one explicitly owned preset.
        from service.core import Core
        for cancel in (False,True):
            # Enrollment is inert test support only; installed config stays Kokoro.
            from tests.service.test_installed_registration import Registration
            history=Registration();history.setUp()
            self.addCleanup(history.doCleanups)
            history.run_registration()
            base_translate=history.fixture.translate
            def engine_translate(info,path=None):
                if path is not None and path.name=='output.wav' and self.stage in path.parents:
                    return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=20000,st_gid=20000))
                return base_translate(info,path)
            history.fixture.translate=engine_translate
            # Each scenario reopens its exact completed root and staging root.
            self.stage=self.root/('stage-cancel' if cancel else 'stage-success')
            self.stage.mkdir()
            root=self.root/('dispatcher-cancel' if cancel else 'dispatcher-success')/'core'
            seed=Core(root,_publication_creation=1);self.core.db.backup(seed.db);(seed.root/'assets'/self.asset['id']).write_bytes(self.raw);seed.close()
            runtime=Runtime()
            contexts=self.context_factory()
            dispatcher=Dispatcher(root,staging=self.stage,runtime=runtime,profiles=self.profiles,production_contexts=contexts)
            try:
                wait(lambda:dispatcher.call('status')['reconciled'])
                self.assertFalse(dispatcher.call('status')['integrity_fault'],dispatcher.call('status'))
                result=dispatcher.call('production_generate','alice',self.context['plan']['request'],key='cancel' if cancel else 'success')
                wait(lambda:runtime.authority is not None)
                self.assertEqual(dispatcher.call('status')['reservations'][-1]['state'],'executing')
                if cancel:
                    attempt_stage=self.stage/runtime.authority.job['attempts'][-1]['staging_id']
                    self.create_stage(attempt_stage)
                    namespace=attempt_stage/'conditioning';namespace.mkdir(mode=0o700)
                    dispatcher.call('production_cancel','alice',dict(job_id=result['job_id']),key='cancel-intent')
                    wait(runtime.stopped.is_set)
                    self.assertEqual(dispatcher.call('status')['reservations'][-1]['state'],'executing')
                    self.assertEqual(dispatcher.call('job',result['job_id'])['state'],'cancel_requested')
                    self.assertTrue(namespace.is_dir())
                runtime.finish.set()
                job=wait(lambda:(j if (j:=dispatcher.call('job',result['job_id']))['state'] in ('succeeded','cancelled','failed') else None))
                self.assertEqual(job['state'],'cancelled' if cancel else 'succeeded',dict(job=job,captures=captures,status=dispatcher.call('status')))
                wait(lambda:dispatcher.call('status')['reservations'][-1]['state']=='retained')
                wait(lambda:dispatcher.call('status')['phase']=='idle')
                if not cancel:
                    artifact=dispatcher.call('production_artifact','alice',dict(artifact_id=job['artifact_ids'][0]))
                    self.assertEqual(len(artifact['asset_ids']),1)
            finally:dispatcher.close()
            self.assert_closed_history(root, runtime, history)
            reopened_runtime=Runtime()
            reopened=Dispatcher(root,staging=self.stage,runtime=reopened_runtime,profiles=self.profiles,production_contexts=contexts)
            try:
                status=wait(lambda:(v if (v:=reopened.call('status'))['reconciled'] else None))
                self.assertFalse(status['integrity_fault'],status)
                self.assertEqual(reopened.call('job',result['job_id'])['state'],job['state'])
                self.assertEqual(reopened.call('voice_version','alice',self.version['id']),self.version)
                with self.assertRaises(ValueError):reopened.call('voice_version','bob',self.version['id'])
                self.assertFalse((self.stage/runtime.authority.job['attempts'][-1]['staging_id']/'conditioning').exists())
                if not cancel:
                    self.assertEqual(reopened.call('production_artifact','alice',dict(artifact_id=job['artifact_ids'][0])),artifact)
                next_job=reopened.call('production_generate','alice',self.context['plan']['request'],key='after-reopen')
                self.assertNotEqual(next_job['job_id'],result['job_id'])
                wait(lambda:reopened_runtime.authority is not None)
                self.assertEqual(reopened.call('status')['reservations'][-1]['state'],'executing')
            finally:reopened.close()
            history.doCleanups()

    def test_preset_and_explicit_cuda_through_actual_child(self):
        from service.admission import DEFAULT_BUDGET
        for task,device in [('custom_voice','cpu'),('custom_voice','cuda:1'),('clone','cuda:1')]:
            with self.subTest(task=task,device=device):
                t=QwenLifecycleTests();t.task=task;t.device=device;t.setUp()
                try:
                    t.admission=Admission(t.core,t.stage,dict(DEFAULT_BUDGET,vram_bytes=8192));t.store=DispatchStore(t.core,t.admission)
                    a,p,stage=t.proposal()
                    self.assertEqual(t.calls[0]['device'],device)
                    self.assertEqual(t.admission.store.rows()[0]['demand']['vram_bytes'],4096 if ':' in device else 0)
                    self.assertEqual(p['manifest']['outputs'][0]['sample_rate'],24000)
                    t.core._publish(a.job['id'],p,stage_root=t.stage,engine_uid=os.getuid())
                    self.assertEqual(t.core.asset(p['manifest']['outputs'][0]['id'])['sha256'],p['manifest']['outputs'][0]['sha256'])
                finally:t.doCleanups()

    def test_closed_hardware_and_native_metal_service_refusal(self):
        from worker.engines.qwen_execution import inventory_from_wire, current_hardware
        for mutate in (lambda w:w.update(extra=True),lambda w:w['devices'][0].update(probed_at='opaque-receipt'),
                       lambda w:w['devices'][0].update(available=1),lambda w:w['devices'][0].update(index=True)):
            wire=inventory_wire(self.inventory);mutate(wire)
            with self.assertRaises(ValueError):inventory_from_wire(wire)
        metal=replace(self.spec,device='mps')
        native=HardwareInventory(Host('darwin','arm64','native'),(Device(DeviceFamily.MPS,0,True,True,datetime.now(timezone.utc).isoformat()),))
        with self.assertRaisesRegex(ValueError,'native_transport'):current_hardware(metal,native)
        context=copy.deepcopy(self.context);context['execution']['descriptor']=metal.identity()
        with self.assertRaisesRegex(ValueError,'generation_only'):execution(context)

    def test_full_descriptor_rate_provenance_and_qualification_fences(self):
        self.profiles[0]['verification']='synthetic'
        with self.assertRaisesRegex(ValueError,'qualification'):self.submit()
        self.assertEqual(self.core.jobs('alice'),[])
        self.profiles[0]['verification']='measured'
        a,p,stage=self.proposal()
        for mutate in (lambda p:p['manifest']['outputs'][0].update(sample_rate=22050),
                       lambda p:p['manifest']['outputs'][0]['provenance'].update(source_id='unbound'),
                       lambda p:p['message'].update(input_sha256='0'*64)):
            changed=copy.deepcopy(p);mutate(changed)
            with self.assertRaises(ValueError):self.core._verified(self.core.job(a.job['id']),changed,stage_root=self.stage,engine_uid=os.getuid())
        for key,value in [('attention','eager'),('temperature',.5),('max_new_tokens',2000)]:
            changed=copy.deepcopy(a.context);changed['execution']['descriptor'][key]=value
            if key != 'attention':
                settings=changed['plan']['resolved']['settings'];settings[key]=value
                changed['plan']['voice_versions'][0]['settings'][key]=value
                digest=oracle.canonical_hash(settings)
                changed['plan']['resolved']['settings_sha256']=digest;changed['plan']['voice_versions'][0]['settings_sha256']=digest
            self.assertNotEqual(generated_source(changed),generated_source(a.context))
        self.bundle.chmod(0o700)
        with self.assertRaisesRegex(ValueError,'sealed'):self.context_factory()
        self.bundle.chmod(0o500)

    def test_http_cli_mcp_saved_reference_generation_and_authorization(self):
        captures,instrument=self.lifecycle_diagnostics()
        import asyncio
        import contextlib
        import httpx
        from clients.http import HTTP
        from clients.cli import main
        from clients import mcp
        from service.api import create_voice_app
        from service.auth import Credential, StaticCredentials
        from service.dispatcher import Dispatcher
        from service.worker_transport import TransportObservation
        from worker.lifecycle import Observation, State
        from tests.service.test_production_api import ProductionAPIFixture
        outer=self
        class Runtime:
            def __init__(inner):inner.authority=None
            def call(inner,op,body):return dict(quarantined=False) if op=='hello' else dict(acked=True)
            def prepare(inner,a,fixture):
                outer.assertEqual(fixture,'qwen');inner.authority=a
                return 'injected/qwen'
            def start(inner,identity):return dict(state='alive')
            def observe(inner,identity,timeout):return Observation(identity,State.STOPPED)
            def collect(inner,a,identity):
                _,p,stage=outer.proposal(authority=a)
                from tests.worker.test_runtime_conditioning import filesystem_policy
                from worker.runtime_conditioning import remove_empty
                (stage/'conditioning').rmdir() if (stage/'conditioning').exists() else None
                return dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',reaped_children=3,timely=True,original_deadline=a.job['deadline']),TransportObservation('proposal','injected',message=p['message'],manifest=p['manifest'])
        instrument(Runtime,'collect')
        self.runtime=Runtime()
        self.core.close()
        self.dispatcher=Dispatcher(self.root/'core',staging=self.stage,runtime=self.runtime,profiles=self.profiles,production_contexts=self.contexts)
        self.addCleanup(self.dispatcher.close)
        self.app=create_voice_app(self.dispatcher,credentials=StaticCredentials([
            Credential('writer','alice',frozenset({'read','production-write'})),Credential('bob','bob',frozenset({'read','production-write'})),Credential('reader','alice',frozenset({'read'}))]))
        def diagnostic_wait(fn):
            try:return ProductionAPIFixture.wait(self,fn)
            except Exception:
                print('QWEN_WAIT '+json.dumps(dict(captures=captures,status=self.dispatcher.call('status'))),flush=True)
                raise
        self.wait=diagnostic_wait
        request=lambda method,path,**kw:ProductionAPIFixture.request(self,method,path,**kw)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        payload=self.context['plan']['request'];endpoint='/v1/operations/speech_generate'
        for token in ('reader','bob'):
            self.assertNotEqual(request('POST',endpoint,payload=payload,key='denied-'+token,token=token).status_code,200)
        for changes in (dict(voice_version_id='missing'),dict(context={}),dict(identity=dict(payload['identity'],device='cuda'))):
            self.assertNotEqual(request('POST',endpoint,payload=dict(payload,**changes),key='bad').status_code,200)
        self.assertIsNone(self.runtime.authority)
        class ASGIClient:
            def __init__(inner,**kw):pass
            def __enter__(inner):return inner
            def __exit__(inner,*a):pass
            def request(inner,method,path,**kw):
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer.app),base_url='http://test') as client:return await client.request(method,path,**kw)
                return asyncio.run(run())
            def post(inner,path,**kw):return inner.request('POST',path,**kw)
            def get(inner,path,**kw):return inner.request('GET',path,**kw)
        def cli(args):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):code=main(args)
            self.assertEqual(code,0,output.getvalue());return json.loads(output.getvalue())
        with patch('clients.http.httpx.Client',ASGIClient),patch.dict(os.environ,{'SPEECH_CREDENTIAL':'writer'}):
            transport=HTTP(credential='writer')
            receipt=transport.call('speech_generate',payload,key='same')
            self.assertEqual(cli(['speech','generate','--request',json.dumps(payload),'--idempotency-key','same']),receipt)
            result=asyncio.run(mcp.call_tool('speech_speech_generate',dict(payload,idempotency_key='same')))
            self.assertFalse(result.isError,result.structuredContent);self.assertEqual(result.structuredContent,receipt)
            self.wait(lambda:self.dispatcher.call('job',receipt['job_id'])['state']=='succeeded')
            self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
            job=transport.call('job_get',dict(job_id=receipt['job_id']))
            artifact=transport.call('artifact_get',dict(artifact_id=job['artifact_ids'][0]))
            self.assertEqual(cli(['artifact','get',artifact['id']]),artifact)
            result=asyncio.run(mcp.call_tool('speech_artifact_get',dict(artifact_id=artifact['id'])))
            self.assertFalse(result.isError);self.assertEqual(result.structuredContent,artifact)
        asset=artifact['asset_ids'][0]
        response=request('GET','/v1/assets/'+asset+'/content?start=0&end_exclusive=50')
        self.assertEqual(response.status_code,200)
        import base64
        self.assertTrue(base64.b64decode(response.json()['content_base64']).startswith(b'RIFF'))
        self.assertEqual(request('GET','/v1/assets/'+asset,token='bob').status_code,404)
        self.assertEqual(self.runtime.authority.allowed_assets,(self.asset['id'],))
        self.assertEqual(self.runtime.authority.context['plan']['voice_versions'][0],self.version)
        self.assertEqual(self.runtime.authority.context['execution']['descriptor']['reference_text'],'Original saved transcript.')


from tests.worker import test_runtime_conditioning as conditioning_tests

class QwenDescendantGateTests(conditioning_tests.ConditioningSupervisorTests):
    def setUp(self):
        super().setUp()
        self.row['allocation']['fixture']='qwen'
        self.enterContext(patch('worker.runtime_protocol.execution',return_value=SimpleNamespace(
            engine='qwen',device='cpu',task='clone',x_vector_only=True,reference_text=None)))
