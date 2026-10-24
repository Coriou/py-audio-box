"""Saved-reference vertical slice; injected transport/backend, no ML or processes."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import wave
from unittest.mock import patch
from contracts import validate as oracle
from lib.tts_backends.contract import AudioResult
from service.admission import Admission
from service.core import Core
from service.dispatch_store import DispatchStore
from service.execution_context import PocketContexts
from service.ingest import ingest_local
from service.production import Production
from worker.engines.pocket import execute as engine_execute
from worker.engines.pocket_execution import PocketExecution
from worker.runtime_child import run_generation
from worker.runtime_protocol import execution, generated_source
from worker.transport import Refused


class PocketLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.core = Core(self.root/'core'); self.addCleanup(lambda: self.core.close())
        self.stage = self.root/'stage'; self.stage.mkdir()
        self.admission = Admission(self.core, self.stage); self.store = DispatchStore(self.core,self.admission)
        b = io.BytesIO()
        with wave.open(b,'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b'\0\0'*4)
        self.raw = b.getvalue(); self.asset = ingest_local(self.core,'alice',self.raw)
        self.bundle = self.root/'bundle'; self.bundle.mkdir()
        entries=[]
        for name in ('config.yaml','model.safetensors'):
            data=b'inert'; (self.bundle/name).write_bytes(data)
            entries.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        raw=json.dumps(dict(schema_version=1,backend='pocket-tts',checkpoint='test',revision='rev1',language='fr',files=entries)).encode()
        (self.bundle/'manifest.json').write_bytes(raw)
        self.spec=PocketExecution('bundle',hashlib.sha256(raw).hexdigest(),'test','rev1','fr',2,1024,100,244,192*1024**2,self.asset['id'],self.asset['sha256'],len(self.raw),4,24000,22050)
        fixture=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())
        self.context=fixture['context']; plan=self.context['plan']; r=plan['resolved']
        ident=dict(engine='pocket-tts',model='test',bundle_sha256=self.spec.manifest_sha256,device='cpu',precision='float32')
        settings=dict(kind='pocket_clone',language='fr',seed=None)
        version=plan['voice_versions'][0]
        version.update(origin='clone',identity=ident,settings=settings,settings_sha256=oracle.canonical_hash(settings),configuration=dict(mode='pocket_clone',language='fr',reference_asset_id=self.asset['id']),reference_asset_ids=[self.asset['id']],reference_asset_sha256={self.asset['id']:self.asset['sha256']})
        self.version=copy.deepcopy(version)
        self.core.voices.register('alice','saved',version,expected_revision=0)
        plan['request'].update(identity=ident,text=' Exact\n texte ')
        plan['assets']=[self.asset]
        profile=fixture['profiles'][0]
        profile.update(id='pocket-cpu',support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='INJECTED-ONLY-NOT-ACTUAL')
        profile['compatibility']=[dict(operation='speech_generate',mode='pocket_clone',identity=ident)]
        profile['limits'].update(ram_bytes=self.spec.memory_bytes,disk_bytes=2*1024**2,threads=2)
        r.update(identity=dict(requested=ident,effective=ident,fallback='forbid',reason='',profile_id='pocket-cpu'),profile=oracle.profile_snapshot(profile),settings=settings,settings_sha256=oracle.canonical_hash(settings))
        self.context['bundles']=[dict(instructions=False,engine='pocket-tts',model='test',mode='pocket_clone',bundle_sha256=self.spec.manifest_sha256,languages=['fr'],selectors=[])]
        self.context['execution']=dict(descriptor=self.spec.identity(),bundles={'bundle':'/models/bundles/bundle'},references={self.asset['id']:'/references/assets/'+self.asset['id']})
        self.profiles=[profile]; self.mapping={'bundle':self.bundle}; self.refs={self.asset['id']:self.core.root/'assets'/self.asset['id']}
        oracle.frozen_plan(plan,self.context['bundles'])
        self.contexts=PocketContexts({'alice':self.context},self.mapping,self.refs)

    def submit(self,key='generate'):
        c=self.contexts.resolve('alice','speech_generate',self.context['plan']['request'])
        return Production(self.core).generate('alice',key,c,_admission=self.admission,_profiles=self.profiles)

    def proposal(self, *, authority=None):
        if authority is None:
            self.submit(); authority=self.store.allocate(self.profiles,'pocket',8)['authority']
        stage=self.stage/authority.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
        self.calls=[];sent=[];outer=self
        class Backend:
            def __init__(inner,root,**kw):outer.calls.append((root,kw))
            def synthesise(inner,text,**kw):
                outer.calls.append((text,kw['reference'].read_bytes()))
                return AudioResult([0.,.5,-.5],22050)
            def close(inner):outer.calls.append('closed')
        def execute(spec,text,bundles,references,**kw):
            self.assertEqual(bundles,authority.context['execution']['bundles']);self.assertEqual(references,authority.context['execution']['references'])
            kw['managed_owner_uid']=os.getuid()  # injected fixture ownership; installed owner is 10001
            return engine_execute(spec,text,self.mapping,self.refs,**kw)
        config=dict(job=authority.job,context=authority.context,allowed_assets=list(authority.allowed_assets),profiles=list(authority.profiles),secret='a'*64,control_uid=os.getuid(),gid=os.getgid())
        with patch('worker.transport.authenticate') as auth,patch('worker.transport.receive',return_value=authority.message()),patch('worker.transport.send',side_effect=lambda c,v,t:sent.append(v)),patch('worker.engines.pocket.execute',side_effect=execute):
            run_generation(None,config,self.stage,time.monotonic()+5,backend_factory=Backend)
            self.assertEqual(auth.call_args.args[1].binding,authority.binding)
        return authority,sent[0],stage

    def test_saved_upload_generation_measured_rate_and_reuse_restart(self):
        authority,proposal,stage=self.proposal()
        self.assertEqual(self.calls[1],(' Exact\n texte ',self.raw));self.assertEqual(self.calls[-1],'closed')
        asset=proposal['manifest']['outputs'][0]
        self.assertEqual((asset['sample_rate'],asset['frames']),(22050,3))
        self.assertEqual(asset['provenance']['source_id'],generated_source(authority.context))
        self.core._publish(authority.job['id'],proposal,stage_root=self.stage,engine_uid=os.getuid())
        self.assertEqual(self.core.asset(asset['id'])['sha256'],asset['sha256'])
        self.assertEqual(self.admission.store.rows()[0]['state'],'executing')
        with self.core.transaction():self.admission.store.release_tx(authority.job['id'],confirmed_reap=True)
        root=self.core.root;self.core.close();self.core=Core(root)
        self.admission=Admission(self.core,self.stage);self.store=DispatchStore(self.core,self.admission)
        self.assertEqual(self.core.voices.version('alice',self.version['id']),self.version)
        second=self.submit('again');self.assertEqual(second,self.submit('again'))
        self.assertEqual(self.core._load(second['id'])[1]['execution']['descriptor'],self.spec.identity())

    def test_closed_identity_foreign_missing_tampered(self):
        for mutate in (lambda c:c['execution']['descriptor'].update(reference_sha256='0'*64),lambda c:c['execution']['references'].update({self.asset['id']:'/tmp/arbitrary'}),lambda c:c['execution']['descriptor'].update(revision='other',path='bad'),lambda c:c['plan']['request'].update(fallback='allow')):
            c=copy.deepcopy(self.context);mutate(c)
            with self.assertRaises(ValueError):PocketContexts({'alice':c},self.mapping,self.refs)
        with self.assertRaises(ValueError):self.contexts.resolve('bob','speech_generate',self.context['plan']['request'])
        with self.assertRaises(ValueError):Production(self.core).generate('bob','foreign',self.context,_admission=self.admission,_profiles=self.profiles)
        for path in (self.refs[self.asset['id']],self.bundle/'manifest.json',self.bundle/'model.safetensors'):
            raw=path.read_bytes();path.write_bytes(b'tampered')
            with self.assertRaises(ValueError):PocketContexts({'alice':self.context},self.mapping,self.refs)
            path.write_bytes(raw)
        self.refs[self.asset['id']].unlink()
        with self.assertRaises((ValueError,OSError)):self.submit()
        self.assertEqual(self.core.jobs('alice'),[])

    def test_cancel_reap_retains_namespace_and_refuses_stale_output(self):
        authority,proposal,stage=self.proposal()
        namespace=stage/'conditioning';self.assertEqual(namespace.stat().st_mode&0o777,0o700)
        (namespace/'descendant-sentinel').write_bytes(b'retained')
        self.core.request_cancel(authority.job['id'])
        from service.dispatcher import Dispatcher
        owner=Dispatcher.__new__(Dispatcher);owner.core=self.core;owner._fail(self.core.job(authority.job['id']),'cancelled')
        with self.assertRaises(ValueError):self.core._publish(authority.job['id'],proposal,stage_root=self.stage,engine_uid=os.getuid())
        with self.assertRaises(ValueError):
            with self.core.transaction():self.admission.store.release_tx(authority.job['id'])
        self.assertTrue((namespace/'descendant-sentinel').exists())
        with self.core.transaction():self.admission.store.release_tx(authority.job['id'],confirmed_reap=True)
        self.assertTrue((namespace/'descendant-sentinel').exists())

    def test_oracle_provenance_rate_and_unknown_namespace(self):
        authority,proposal,stage=self.proposal()
        for mutate in (lambda p:p['manifest']['outputs'][0].update(sample_rate=24000),
                       lambda p:p['manifest']['outputs'][0]['provenance'].update(source_id='unbound'),
                       lambda p:p['message'].update(input_sha256='0'*64)):
            changed=copy.deepcopy(proposal);mutate(changed)
            with self.assertRaises(ValueError):self.core._verified(self.core.job(authority.job['id']),changed,stage_root=self.stage,engine_uid=os.getuid())
        from service.reconciliation import Reconciliation
        (stage/'conditioning'/'opaque').write_bytes(b'retain')
        summary=Reconciliation(self.core,self.stage).reconcile_startup()
        self.assertTrue(summary['blocked']);self.assertIn('unknown_namespace',summary['reasons'])
        self.assertTrue((stage/'conditioning'/'opaque').exists())

    def test_qualification_and_dispatch_dependency_recheck(self):
        self.profiles[0]['verification']='synthetic'
        with self.assertRaisesRegex(ValueError,'qualification'):self.submit()
        self.profiles[0]['verification']='measured';self.submit()
        with self.assertRaises(ValueError):self.store.allocate(self.profiles,'kokoro',8)
        self.refs[self.asset['id']].write_bytes(b'tamper')
        with self.assertRaises(ValueError):self.store.allocate(self.profiles,'pocket',8)
        self.assertEqual(self.admission.store.rows()[0]['state'],'queued')

    def test_public_overrides_and_managed_owner_refuse(self):
        from jsonschema.exceptions import ValidationError
        for field in ('execution','reference_path','reference_overrides','context'):
            request=copy.deepcopy(self.context['plan']['request']);request[field]='/tmp/caller'
            with self.assertRaises(ValidationError):self.contexts.resolve('alice','speech_generate',request)
        from worker.engines.pocket import verify_inputs
        with self.assertRaisesRegex(Refused,'ownership'):
            verify_inputs(self.spec,self.mapping,self.refs,owner_uid=os.getuid()+1)
        resolved=self.contexts.resolve('alice','speech_generate',self.context['plan']['request'])
        resolved['execution']['descriptor']['reference_sha256']='0'*64
        self.assertEqual(execution(self.contexts.resolve('alice','speech_generate',self.context['plan']['request'])),self.spec)

    def test_child_limits_precede_generation_without_actual_rlimits(self):
        from worker.runtime_child import contain_kokoro, main
        import resource
        with patch.dict(os.environ),patch('resource.setrlimit') as limits:
            contain_kokoro(self.context)
            self.assertEqual(limits.call_args_list[0].args,(resource.RLIMIT_AS,(self.spec.memory_bytes,)*2))
            self.assertEqual(limits.call_args_list[1].args,(resource.RLIMIT_FSIZE,(max(self.spec.max_output_bytes,self.spec.reference_bytes),)*2))
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'2')
        calls=[]
        config=dict(fixture='pocket',context=self.context,job=dict(attempts=[dict(staging_id='stage')]))
        with patch('sys.argv',['worker','3']),patch('worker.runtime_child.socket.socket'),patch('worker.runtime_child.receive',return_value=config),patch('worker.runtime_child.contain_kokoro',side_effect=lambda c:calls.append('limits')),patch('worker.runtime_child.run_generation',side_effect=lambda *a:calls.append('generation')),patch('worker.runtime_child.os.chmod'):
            main()
        self.assertEqual(calls,['limits','generation'])

    def test_supervisor_independent_mapping_and_fixture(self):
        self.submit();a=self.store.allocate(self.profiles,'pocket',8)['authority']
        from worker.runtime_supervisor import RuntimeSupervisor
        s=RuntimeSupervisor.__new__(RuntimeSupervisor);s.bundles=a.context['execution']['bundles'];s.references={}
        body=dict(snapshot=json.loads(a._snapshot),binding=a.binding,fixture='pocket')
        with self.assertRaisesRegex(Refused,'reference_mapping'):s.operation('prepare',body)
        s.references=a.context['execution']['references']
        from worker.engines.pocket import verify_inputs as real_verify
        def verify(spec,bundles,references,**kwargs):
            self.assertEqual(kwargs,dict(owner_uid=10001));self.assertEqual(references,s.references)
            real_verify(spec,self.mapping,self.refs)
            raise Refused('inert_verified_before_journal')
        with patch('worker.engines.pocket.verify_inputs',side_effect=verify):
            with self.assertRaisesRegex(Refused,'inert_verified'):s.operation('prepare',body)
        body['fixture']='kokoro'
        with self.assertRaisesRegex(Refused,'dispatch'):s.operation('prepare',body)

    def test_dispatcher_generate_artifact_and_cancel_reap(self):
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
                outer.assertEqual(fixture,'pocket');inner.authority=authority
                return 'lane/boot/allocation'
            def start(inner,identity):return dict(state='alive')
            def observe(inner,identity,timeout):
                return Observation(identity,State.STOPPED if inner.finish.is_set() else State.ALIVE)
            def terminate(inner,identity,timeout):inner.cancel=True;inner.stopped.set()
            def collect(inner,authority,identity):
                receipt=dict(runtime_id=identity,binding=authority.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',timely=not inner.cancel,original_deadline=authority.job['deadline'])
                if inner.cancel:return receipt,None
                _,proposal,_=outer.proposal(authority=authority)
                return receipt,TransportObservation('proposal','injected-test-only',message=proposal['message'],manifest=proposal['manifest'])
        def wait(predicate):
            end=time.monotonic()+5
            while time.monotonic()<end:
                value=predicate()
                if value:return value
                time.sleep(.02)
            self.fail('bounded dispatcher wait expired')
        # Separate persistent owner root, seeded with one explicitly owned preset.
        from service.core import Core
        root=self.root/'dispatcher'
        seed=Core(root);self.core.db.backup(seed.db);(seed.root/'assets'/self.asset['id']).write_bytes(self.raw);seed.close()
        for cancel in (False,True):
            runtime=Runtime()
            contexts=PocketContexts({'alice':self.context},self.mapping,self.refs)
            dispatcher=Dispatcher(root,staging=self.stage,runtime=runtime,profiles=self.profiles,production_contexts=contexts)
            try:
                wait(lambda:dispatcher.call('status')['reconciled'])
                self.assertFalse(dispatcher.call('status')['integrity_fault'],dispatcher.call('status'))
                result=dispatcher.call('production_generate','alice',self.context['plan']['request'],key='cancel' if cancel else 'success')
                wait(lambda:runtime.authority is not None)
                self.assertEqual(dispatcher.call('status')['reservations'][-1]['state'],'executing')
                if cancel:
                    dispatcher.call('production_cancel','alice',dict(job_id=result['job_id']),key='cancel-intent')
                    wait(runtime.stopped.is_set)
                    self.assertEqual(dispatcher.call('status')['reservations'][-1]['state'],'executing')
                    self.assertEqual(dispatcher.call('job',result['job_id'])['state'],'cancel_requested')
                runtime.finish.set()
                job=wait(lambda:(j if (j:=dispatcher.call('job',result['job_id']))['state'] in ('succeeded','cancelled') else None))
                self.assertEqual(job['state'],'cancelled' if cancel else 'succeeded')
                wait(lambda:dispatcher.call('status')['reservations'][-1]['state']=='retained')
                if not cancel:
                    artifact=dispatcher.call('production_artifact','alice',dict(artifact_id=job['artifact_ids'][0]))
                    self.assertEqual(len(artifact['asset_ids']),1)
            finally:dispatcher.close()


if __name__=='__main__':unittest.main()
