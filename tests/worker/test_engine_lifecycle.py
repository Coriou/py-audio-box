"""Authenticated lifecycle/publication with injected local backend; no ML inference."""
import copy
import hashlib
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from contracts import validate as oracle
from service.worker_transport import Authority
from tests.service import test_engine_admission as fixtures
from worker.runtime_child import run_kokoro
from worker.runtime_protocol import generated_source
from worker.engines.kokoro import execute as engine_execute
from worker.transport import Refused


class EngineLifecycleTests(unittest.TestCase):
    setUp = fixtures.EngineAdmissionTests.setUp
    submit = fixtures.EngineAdmissionTests.submit
    def proposal(self, *, authority=None, samples=None, mutate=None, authentication_error=None):
        if authority is None:
            self.submit();authority=self.store.allocate(self.profiles,'kokoro',8)['authority']
        stage=self.stage/authority.job['attempts'][-1]['staging_id']
        stage.mkdir(mode=0o770,exist_ok=True);stage.chmod(0o770)
        calls=[];sent=[]
        class Backend:
            def __init__(inner, root, **kw):calls.append((root,kw))
            def synthesise(inner,text,**kw):
                calls.append((text,kw))
                from lib.tts_backends.contract import AudioResult
                return AudioResult([0., .25, -.5] if samples is None else samples,24000)
            def close(inner):calls.append('closed')
        def execute(spec,text,bundles,**kw):
            self.assertEqual(bundles,authority.context['execution']['bundles'])
            return engine_execute(spec,text,self.mapping,**kw)
        config=dict(job=authority.job,context=authority.context,allowed_assets=list(authority.allowed_assets),profiles=list(authority.profiles),secret='a'*64,control_uid=os.getuid(),gid=os.getgid())
        message=authority.message()
        if mutate:mutate(message)
        with patch('worker.transport.authenticate',side_effect=authentication_error) as auth, patch('worker.transport.receive',return_value=message), patch('worker.transport.send',side_effect=lambda c,v,t:sent.append(v)), patch('worker.engines.kokoro.execute',side_effect=execute):
            run_kokoro(None,config,self.stage,time.monotonic()+5,backend_factory=Backend)
            self.assertEqual(auth.call_args.args[1].binding,authority.binding)
        return authority,sent[0],calls

    def test_measured_publication_and_schema(self):
        authority,proposal,calls=self.proposal()
        asset=proposal['manifest']['outputs'][0]
        oracle.validate('asset',asset)
        self.assertEqual(asset['provenance']['kind'],'generated')
        self.assertEqual(asset['provenance']['source_id'],generated_source(authority.context))
        self.assertEqual(calls[1],(' Exact\n authored text ',dict(preset='af_heart')))
        self.assertEqual(calls[-1],'closed')
        data,_,_,_=self.core._verified(self.core.job(authority.job['id']),proposal,stage_root=self.stage,engine_uid=os.getuid())
        self.assertEqual(len(data),50);self.assertEqual(asset['frames'],3)
        self.assertEqual(asset['sha256'],hashlib.sha256(data).hexdigest())
        self.core._publish(authority.job['id'],proposal,stage_root=self.stage,engine_uid=os.getuid())
        self.assertEqual(self.core.job(authority.job['id'])['state'],'succeeded')
        self.assertEqual(self.core.asset(asset['id'])['sha256'],asset['sha256'])
        # Publication alone never supplies supervisor death proof/capacity release.
        self.assertEqual(self.admission.store.rows()[0]['state'],'executing')
        with self.assertRaises(ValueError):
            with self.core.transaction():self.admission.store.release_tx(authority.job['id'])
        with self.core.transaction():self.admission.store.release_tx(authority.job['id'],confirmed_reap=True)

    def test_provenance_enum_retains_existing_values(self):
        from jsonschema.exceptions import ValidationError
        for kind in ('synthetic','upload','microphone','https','youtube','legacy','generated'):
            oracle.validate('provenance',dict(kind=kind,source_sha256='0'*64,source_id='schema-only',missing_fields=[]))
        with self.assertRaises(ValidationError):
            oracle.validate('provenance',dict(kind='caller-invented',source_sha256='0'*64,source_id='schema-only',missing_fields=[]))

    def test_source_hash_frame_and_stale_identity_refuse(self):
        authority,proposal,_=self.proposal()
        for mutate in (lambda p:p['manifest']['outputs'][0]['provenance'].update(source_id='caller-source'),lambda p:p['manifest']['outputs'][0].update(sha256='0'*64),lambda p:p['manifest']['outputs'][0].update(frames=4),lambda p:p['message'].update(generation=2),lambda p:p['manifest'].update(attempt_id='other-attempt')):
            changed=copy.deepcopy(proposal);mutate(changed)
            with self.assertRaises(ValueError):self.core._verified(self.core.job(authority.job['id']),changed,stage_root=self.stage,engine_uid=os.getuid())
        self.assertEqual(self.core.db.execute('SELECT count(*) FROM publications').fetchone()[0],0)

    def test_input_substitution(self):
        with self.assertRaises(ValueError):self.proposal(mutate=lambda m:m['plan']['request'].update(text='substitute'))
        self.assertEqual(list(self.stage.rglob('output.wav')),[])

    def test_oversized_backend_output_never_stages(self):
        with self.assertRaises(Refused):self.proposal(samples=[0.]*101)
        self.assertEqual(list(self.stage.rglob('output.wav')),[])

    def test_no_session_authority_never_calls_backend(self):
        with self.assertRaises(Refused):self.proposal(authentication_error=Refused('unbound_session'))
        self.assertEqual(list(self.stage.rglob('output.wav')),[])

    def test_cancelled_result_cannot_publish_or_release_early(self):
        authority,proposal,_=self.proposal()
        job=self.core.request_cancel(authority.job['id'])
        self.assertEqual(job['state'],'cancel_requested')
        # Existing cancellation is intent until reap; after confirmed cancellation
        # the stale proposal cannot resurrect publication.
        from service.dispatcher import Dispatcher
        owner=Dispatcher.__new__(Dispatcher);owner.core=self.core
        owner._fail(job,'cancelled')
        with self.assertRaises(ValueError):self.core._publish(job['id'],proposal,stage_root=self.stage,engine_uid=os.getuid())
        self.assertEqual(self.admission.store.rows()[0]['state'],'executing')
        with self.assertRaises(ValueError):
            with self.core.transaction():self.admission.store.release_tx(job['id'])

    def test_supervisor_requires_independently_registered_mount(self):
        from worker.runtime_supervisor import RuntimeSupervisor
        self.submit();authority=self.store.allocate(self.profiles,'kokoro',8)['authority']
        supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor);supervisor.bundles={}
        import json
        with self.assertRaisesRegex(Refused,'mount'):
            supervisor.operation('prepare',dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture='kokoro'))

class DispatcherEngineTests(unittest.TestCase):
    setUp=fixtures.EngineAdmissionTests.setUp
    proposal=EngineLifecycleTests.proposal
    submit=fixtures.EngineAdmissionTests.submit
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
                outer.assertEqual(fixture,'kokoro');inner.authority=authority
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
        root=Path(self.fixture.temp.name)/'dispatcher'
        seed=Core(root);seed.voices.register('alice','register',self.version,expected_revision=0);seed.close()
        for cancel in (False,True):
            runtime=Runtime()
            contexts=__import__('service.execution_context',fromlist=['KokoroContexts']).KokoroContexts({'alice':self.context},self.mapping)
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
