"""Unseeded public clone journey with real Core and inert TLS/backend/child function."""
import asyncio
import base64
import copy
import hashlib
import json
import time
import unittest
from unittest.mock import patch
from clients.http import HTTP
from clients import mcp
from contracts import validate as oracle
from service.catalogue import Failure
from tests.release import test_native_onboarding as onboarding
from tests.release import test_native_enrollment as enrollment
from tests.service.test_native_runtime import NativeFixture
from worker.engines.qwen_execution import inventory_wire
from worker.transport import canonical, Refused


class ManagedCloneTests(unittest.TestCase):
    write = enrollment.NativeEnrollmentTests.write
    enroll = enrollment.NativeEnrollmentTests.enroll
    app = onboarding.NativeOnboardingTests.app
    connector = onboarding.NativeOnboardingTests.connector
    invoke_cli = onboarding.NativeOnboardingTests.invoke_cli
    wait = onboarding.NativeOnboardingTests.wait
    prepare = onboarding.NativeOnboardingTests.prepare

    def bridge(self, app):
        import httpx
        class Client:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def request(self, method, path, **kwargs):
                if 'content' in kwargs and not isinstance(kwargs['content'],bytes):
                    kwargs['content']=b''.join(kwargs['content'])
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://inert') as client:
                        return await client.request(method,path,**kwargs)
                return asyncio.run(run())
            def get(self,path,**kwargs):return self.request('GET',path,**kwargs)
            def post(self,path,**kwargs):return self.request('POST',path,**kwargs)
        return patch('clients.http.httpx.Client',Client)

    def setUp(self):
        from service import native_package as n
        from service.admission import DEFAULT_BUDGET
        self.f = NativeFixture(self,clone=True)
        self.root = self.f.f.fixture.root; self.root.chmod(0o700)
        self.owner = self.root/'owner'; self.owner.mkdir(mode=0o700)
        for name in ('ca','cert','key'): self.write(name,b'INERT')
        self.config = dict(version=1,variant=n.VARIANT,native=self.f.native,host='127.0.0.1',port=4443,
            peer_sha256=self.f.pin,cafile=str(self.root/'ca'),certfile=str(self.root/'cert'),keyfile=str(self.root/'key'),
            spool=str(self.f.spool),staging=str(self.f.stage))
        snapshot=self.f.snapshot; identity=snapshot[1]['plan']['resolved']['identity']['effective']
        # One trusted provisioned Base template/profile. No catalogue row is seeded.
        snapshot[3][0]['compatibility'].append(dict(operation='voice_prepare',mode='prepare',identity=identity))
        snapshot[1]['plan']['resolved']['profile']=oracle.profile_snapshot(snapshot[3][0])
        self.templates=dict(version=1,variant=n.VARIANT,profiles=snapshot[3],budget=DEFAULT_BUDGET,
            inventory=inventory_wire(self.f.f.inventory),registrations=[dict(principal='alice',profile='qwen-mps',voice_version='vv0',context=snapshot[1])])
        self.write('control',canonical(self.config)); self.write('templates',canonical(self.templates))
        self.record=self.root/'enrollment'
        self.kw=dict(configuration_file=str(self.root/'control'),templates_file=str(self.root/'templates'),owner_root=str(self.owner),connector=self.f.connect)
        tls=patch.object(n.p,'tls_context',return_value=None);tls.start();self.addCleanup(tls.stop)

    def public_journey(self, terminal=None):
        app=self.prepare()
        transcript='  Exact user transcript.\nBonjour ! '
        audition='  A separately authored audition.\nEncore ! '
        def journey(reopen):
            with self.bridge(app),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'x'*64}):
                http=HTTP(credential='x'*64)
                def tool(op,payload,key=None):
                    args=dict(payload)
                    if key:args['idempotency_key']=key
                    result=asyncio.run(mcp.call_tool('speech_'+op,args))
                    self.assertFalse(result.isError,result.structuredContent)
                    return result.structuredContent
                def complete(receipt):
                    end=time.monotonic()+5
                    while time.monotonic()<end:
                        job=http.call('job_get',dict(job_id=receipt['job_id']))
                        if job['state'] in ('succeeded','failed'):break
                        time.sleep(.01)
                    else:self.fail(repr((job,app.state.dispatcher.call('status'),[(q['op'],q['runtime_id']) for q in self.f.requests],self.f.s.failed)))
                    self.assertEqual(job['state'],'succeeded',job)
                    self.wait(lambda:app.state.dispatcher.call('status')['phase']=='idle')
                    return job
                if not reopen:
                    self.assertEqual(http.call('voices_list',dict(limit=20,cursor=''))['items'],[])
                    self.assertIn('voice_prepare',http.get('/skeleton')['configured_capabilities']['operations'])
                    path=self.root/'public-reference.wav';path.write_bytes(self.f.reference_raw)
                    upload=http.upload_file(path)
                    self.upload=upload
                    self.request=dict(asset_id=upload['id'],in_tick=0,out_tick=8,reference_text=transcript,audition_text=audition,
                        identity=self.f.snapshot[1]['plan']['request']['identity'])
                    for change in (dict(reference_text=''),dict(audition_text=' '),dict(out_tick=6),dict(identity=dict(self.request['identity'],model='wrong'))):
                        with self.assertRaises(Failure):http.call('voice_prepare',dict(self.request,**change),key='invalid')
                    self.prep=self.invoke_cli(['voice','prepare','--request',json.dumps(self.request),'--idempotency-key','prepare'])
                    self.assertEqual(self.prep,tool('voice_prepare',self.request,'prepare'))
                    self.prep_job=complete(self.prep)
                    candidates=http.call('voice_results',dict(job_id=self.prep['job_id']))
                    self.assertEqual(candidates,tool('voice_results',dict(job_id=self.prep['job_id'])))
                    candidate=candidates[0] if isinstance(candidates,list) else candidates['candidates'][0]
                    roles=candidate['reference_roles'];reference=roles['reference']['asset'];spoken=roles['audition']['asset']
                    self.assertEqual(roles['reference']['transcript'],transcript)
                    self.assertEqual(roles['audition']['text'],audition)
                    self.assertEqual(roles['reference']['original'],upload)
                    self.assertEqual(reference['sha256'],upload['sha256']);self.assertNotEqual(reference['sha256'],spoken['sha256'])
                    self.save=dict(job_id=self.prep['job_id'],candidate_id=candidate['id'],voice_id='my.clone',expected_revision=0)
                    self.saved=self.invoke_cli(['voice','save','--request',json.dumps(self.save),'--idempotency-key','save'])
                    self.assertEqual(self.saved,tool('voice_save',self.save,'save'))
                    for payload,key in ((dict(self.save,expected_revision=1),'save'),(self.save,'cas')):
                        with self.assertRaises(Failure):http.call('voice_save',payload,key=key)
                    version=self.saved['version']
                    self.generate=dict(self.f.snapshot[1]['plan']['request'],voice_version_id=version['id'],text='Now reuse this immutable clone.')
                    self.generated=http.call('speech_generate',self.generate,key='generate')
                    self.generated_job=complete(self.generated)
                    artifact=http.call('artifact_get',dict(artifact_id=self.generated_job['artifact_ids'][0]))
                    aid=artifact['asset_ids'][0];asset=http.call('asset_get',dict(asset_id=aid))
                    first=http.call('asset_content',dict(asset_id=aid,start=0,end_exclusive=1))
                    data=http.call('asset_content',dict(asset_id=aid,start=0,end_exclusive=first['total_bytes']))
                    self.assertEqual(hashlib.sha256(base64.b64decode(data['content_base64'])).hexdigest(),asset['sha256'])
                    with self.assertRaises(Failure):http.call('voice_prepare',dict(self.request,reference_text='changed'),key='prepare')
                    for credential in (None,'wrong','r'*64):
                        with self.assertRaises(Failure):HTTP(credential=credential).call('voice_prepare',self.request,key='denied')
                    self.assertTrue((self.owner/'assets'/upload['id']).exists())
                    self.assertTrue((self.owner/'assets'/reference['id']).exists())
                    if terminal:
                        self.f.f.h=type(self.f.f.h)()
                        before=self.f.before
                        self.f.before=lambda q:self.f.s.tick()
                        with patch('worker.native_child.parent_exchange',return_value=None):
                            stopped=http.call('speech_generate',self.generate,key='terminal')
                            self.terminal_id=stopped['job_id']
                            self.wait(lambda:self.f.s.active and self.f.s.active['handle'])
                            if terminal == 'cancelled':
                                http.call('job_cancel',dict(job_id=self.terminal_id),key='cancel-terminal')
                                self.wait(lambda:self.f.s.active['cancel'])
                            self.f.s.active['receiver'].join(1)
                            self.f.f.h.code=-15 if terminal == 'cancelled' else 7
                            self.f.s.tick()
                            self.wait(lambda:http.call('job_get',dict(job_id=self.terminal_id))['state']==terminal)
                            self.wait(lambda:app.state.dispatcher.call('status')['phase']=='idle')
                        self.f.before=before

                else:
                    self.assertEqual(self.prep,http.call('voice_prepare',self.request,key='prepare'))
                    self.assertEqual(self.saved,http.call('voice_save',self.save,key='save'))
                    self.assertEqual(self.generated,http.call('speech_generate',self.generate,key='generate'))
                    self.assertEqual(self.generated_job,http.call('job_get',dict(job_id=self.generated['job_id'])))
                    self.assertEqual(self.prep_job,http.call('job_get',dict(job_id=self.prep['job_id'])))
                    if terminal:
                        self.assertEqual(http.call('job_wait',dict(job_id=self.terminal_id,cursor='',wait_seconds=0))['job']['state'],terminal)
                        self.assertEqual(http.call('voice_version_get',dict(voice_version_id=self.saved['version']['id'])),self.saved['version'])
                        self.f.f.h=type(self.f.f.h)()
                        complete(http.call('speech_generate',self.generate,key='after-terminal'))
                        self.assertFalse(app.state.dispatcher.call('status')['retry_available'])

        async def run():
            for reopen in (False,True):
                async with app.router.lifespan_context(app):await asyncio.to_thread(journey,reopen)
                self.assertFalse(app.state.dispatcher.thread.is_alive())
        with self.connector(),patch('worker.native_child.parent_exchange',side_effect=self.f.exchange):asyncio.run(run())
        self.assertEqual(len(self.f.f.calls),4 if terminal else 2)
        calls=self.f.f.fixture.backend.synthesise_clone.call_args_list
        self.assertEqual(len(calls),3 if terminal else 2)
        self.assertEqual(calls[0].args[0]['text'],audition)
        self.assertEqual(calls[1].args[0]['text'],self.generate['text'])
        self.assertEqual([call.args[2] for call in calls],[transcript]*(3 if terminal else 2))
        self.assertEqual(len(self.f.s._preset_templates),1)
        return app

    def test_public_upload_prepare_results_save_generate_reopen_replay(self):
        self.public_journey()

    def test_saved_clone_cancel_reopen_and_new_job(self):
        self.public_journey(terminal='cancelled')

    def test_saved_clone_failure_reopen_and_new_job(self):
        self.public_journey(terminal='failed')

    def test_reopen_refuses_tampered_preparation_pair_without_repair(self):
        app=self.public_journey()
        stage=self.f.stage/self.prep_job['attempts'][0]['staging_id']
        path=stage/'reference.wav';raw=path.read_bytes();path.write_bytes(raw[:-2]+b'xx')
        before={p:p.read_bytes() for root in (self.owner,self.f.spool,self.f.stage) for p in root.rglob('*') if p.is_file()}
        acks=sum(q['op']=='ack' for q in self.f.requests)
        async def reopen():
            with self.assertRaises(Exception):
                async with app.router.lifespan_context(app):self.fail('tampered preparation accepted')
        with self.connector():asyncio.run(reopen())
        self.assertEqual(acks,sum(q['op']=='ack' for q in self.f.requests))
        self.assertEqual(before,{p:p.read_bytes() for root in (self.owner,self.f.spool,self.f.stage) for p in root.rglob('*') if p.is_file()})

    def test_cancelled_preparation_retains_reference_and_reopens(self):
        app=self.prepare()
        self.f.before=lambda q:self.f.s.tick()
        def journey():
            with self.bridge(app):
                http=HTTP(credential='x'*64)
                path=self.root/'cancel-reference.wav';path.write_bytes(self.f.reference_raw)
                asset=http.upload_file(path)
                request=dict(asset_id=asset['id'],in_tick=0,out_tick=8,reference_text='Exact cancellation source.',
                    audition_text='Separate audition.',identity=self.f.snapshot[1]['plan']['request']['identity'])
                receipt=http.call('voice_prepare',request,key='cancel')
                self.wait(lambda:self.f.s.active and self.f.s.active['handle'])
                row=self.f.s.journal.value['records'][0]
                paths=(self.owner/'assets'/asset['id'],next(self.f.spool.glob('*/input')),
                    self.f.s.directory(row)/('input-'+asset['sha256']))
                http.call('job_cancel',dict(job_id=receipt['job_id']),key='cancel-intent')
                self.wait(lambda:self.f.s.active['cancel'])
                self.assertIsNone(self.f.s.journal.value['records'][0]['receipt'])
                self.assertEqual(app.state.dispatcher.call('status')['reservations'][0]['state'],'executing')
                for p in paths:self.assertEqual(p.read_bytes(),self.f.reference_raw)
                self.f.s.active['receiver'].join(1);self.f.f.h.code=-15;self.f.s.tick()
                self.wait(lambda:http.call('job_get',dict(job_id=receipt['job_id']))['state']=='cancelled')
                self.wait(lambda:app.state.dispatcher.call('status')['phase']=='idle')
                for p in paths:self.assertEqual(p.read_bytes(),self.f.reference_raw)
                self.assertFalse(list(self.f.stage.glob('*/reference.wav')))
        async def run():
            async with app.router.lifespan_context(app):await asyncio.to_thread(journey)
            acks=sum(q['op']=='ack' for q in self.f.requests)
            async with app.router.lifespan_context(app):
                self.assertFalse(app.state.dispatcher.call('status')['quarantined'])
            self.assertEqual(acks,sum(q['op']=='ack' for q in self.f.requests))
        with self.connector(),patch('worker.native_child.parent_exchange',return_value=None):asyncio.run(run())


class ManagedCloneDerivationTests(unittest.TestCase):
    def setUp(self):
        from service.core import Core
        from service.admission import Admission
        from service.ingest import ingest_local
        from service.execution_context import NativeQwenContexts
        self.f=NativeFixture(self,clone=True)
        snapshot=self.f.snapshot; snapshot[0]['principal_id']='alice'
        identity=snapshot[1]['plan']['resolved']['identity']['effective']
        profile=snapshot[3][0];profile['limits']['disk_bytes']=2*1024**2
        profile['compatibility'].append(dict(operation='voice_prepare',mode='prepare',identity=identity))
        snapshot[1]['plan']['resolved']['profile']=oracle.profile_snapshot(profile)
        self.f.s.registrations[self.f.a.binding]['snapshot']=copy.deepcopy(snapshot)
        self.f.s._preset_templates=copy.deepcopy(self.f.s.registrations)
        self.core=Core(self.f.f.fixture.root/'owned');self.addCleanup(self.core.close)
        Admission(self.core,self.f.stage,private_spool=self.f.spool)
        self.asset=ingest_local(self.core,'alice',self.f.reference_raw)
        self.foreign=ingest_local(self.core,'bob',self.f.reference_raw)
        self.contexts=NativeQwenContexts({('alice','qwen-mps','vv0'):snapshot[1]},self.f.f.inventory,snapshot[3],self.f.native)
        self.contexts.bind_availability(self.f.runtime)
        self.request=dict(asset_id=self.asset['id'],in_tick=0,out_tick=8,reference_text=' Exact source.\n',
            audition_text=' Different audition.\n',identity=identity)

    def prepared(self):
        context=self.contexts.resolve_owned(self.core,'alice','voice_prepare',self.request)
        token=oracle.canonical_hash(['alice','prepare',self.request])
        context['plan']['reference_roles'].update(reference_asset_id='reference-'+token,audition_asset_id='audition-'+token)
        context['plan']['resolved']['reuse_configuration']['reference_asset_id']='reference-'+token
        snapshot=copy.deepcopy(self.f.snapshot);snapshot[1]=context;snapshot[2]=[self.asset['id']]
        snapshot[0].update(idempotency_key='prepare',operation='voice_prepare',input_sha256=oracle.plan_hash(context['plan']))
        return snapshot

    def test_owner_transcript_model_segment_and_owned_byte_refusals(self):
        for change in (dict(asset_id=self.foreign['id']),dict(reference_text=' '),dict(audition_text=' '),
                dict(in_tick=2),dict(out_tick=6),dict(identity=dict(self.request['identity'],model='wrong'))):
            with self.subTest(change=change),self.assertRaises((ValueError,Refused)):
                self.contexts.resolve_owned(self.core,'alice','voice_prepare',dict(self.request,**change))
        with self.assertRaises(Refused):self.contexts.resolve_owned(self.core,'bob','voice_prepare',self.request)
        self.assertEqual(self.core.voices.list('alice'),[])
        path=self.core.root/'assets'/self.asset['id'];raw=path.read_bytes();path.write_bytes(raw[:-2]+b'xx')
        with self.assertRaises((ValueError,Refused)):self.contexts.resolve_owned(self.core,'alice','voice_prepare',self.request)

    def test_supervisor_reconstructs_original_only_and_refuses_transcript_model_trust_tamper(self):
        from service.worker_transport import Authority
        snapshot=self.prepared();supervisor=self.f.s
        binding=supervisor.register_snapshot(snapshot)
        self.assertEqual(binding,Authority(*snapshot).binding)
        for mutate in (lambda s:s[0].update(principal_id='bob'),
                lambda s:s[1]['execution']['descriptor'].update(reference_text='different'),
                lambda s:s[1]['execution']['descriptor'].update(checkpoint='wrong'),
                lambda s:s[1]['execution']['native'].update(boot='e'*32),
                lambda s:s[1]['plan']['resolved']['reuse_configuration'].update(reference_text='different'),
                lambda s:s[1]['plan']['reference_roles'].update(audition_asset_id='forged'),
                lambda s:s[1]['execution']['descriptor'].update(top_p=1)):
            changed=copy.deepcopy(snapshot);mutate(changed)
            with self.subTest(mutation=mutate),self.assertRaises((Refused,ValueError)):
                changed[0]['input_sha256']=oracle.plan_hash(changed[1]['plan'])
                supervisor.register_snapshot(changed)
        supervisor._preset_templates={}
        changed=copy.deepcopy(snapshot);changed[0]['id']='another-job'
        with self.assertRaises(Refused):supervisor.register_snapshot(changed)
        self.assertIn(binding,supervisor.registrations)

    def test_preparation_proposal_binds_pair_transcript_model_and_attempt(self):
        import io,wave
        from worker import native_child,native_protocol as p
        snapshot=self.prepared();spec=p.native_execution(snapshot[1])
        out=io.BytesIO()
        with wave.open(out,'wb') as w:w.setparams((1,2,spec.sample_rate,0,'NONE','none'));w.writeframes(b'\x01\x00'*3)
        raw=out.getvalue();result=dict(data=raw,frames=3,sha256=hashlib.sha256(raw).hexdigest())
        proposal=native_child.proposal(snapshot,result,spec)
        native_child.validate_proposal(snapshot,proposal,raw,spec)
        for mutate in (lambda v:v['manifest'].update(job_id='wrong'),lambda v:v['manifest'].update(operation='speech_generate'),
                lambda v:v['message'].update(kind='execute'),
                lambda v:v['manifest']['outputs'][0].update(sha256='0'*64),
                lambda v:v['manifest']['candidates'][0]['version'].update(reference_text='inferred'),
                lambda v:v['manifest']['candidates'][0]['version']['identity'].update(model='wrong'),
                lambda v:v['manifest']['candidates'][0]['reference_roles']['audition'].update(text='source transcript'),
                lambda v:v['manifest']['candidates'][0].update(audition_asset_id=v['manifest']['outputs'][0]['id'])):
            changed=copy.deepcopy(proposal);mutate(changed)
            with self.subTest(mutation=mutate),self.assertRaises((ValueError,Refused)):
                native_child.validate_proposal(snapshot,changed,raw,spec)

    def test_real_preparation_adapter_transports_audition_and_preserves_reference(self):
        from worker import native_child
        from worker.engines.qwen_prepare import QwenPreparation
        real_infer = native_child.infer
        observed = []
        def infer(spec, *args, **kwargs):
            self.assertIs(type(spec), QwenPreparation)
            result = real_infer(spec, *args, **kwargs)
            self.assertEqual(set(result), {'data', 'frames', 'sha256', 'size_bytes', 'identity'})
            self.assertEqual(result['frames'], 1)
            self.assertEqual(result['size_bytes'], 46)
            self.assertEqual(result['sha256'], hashlib.sha256(result['data']).hexdigest())
            self.assertNotEqual(result['data'], self.f.reference_raw)
            self.assertNotEqual(result['sha256'], self.asset['sha256'])
            observed.append((result, kwargs['reference_temp_dir']))
            return result
        # Only the backend is inert: actual infer, child framing, parent exchange,
        # proposal validation and supervisor collection all execute.
        with patch('worker.native_child.infer', side_effect=infer):
            authority, rid = self.launch_preparation()
        self.assertEqual(len(observed), 1)
        result, conditioning = observed[0]
        from pathlib import Path
        self.assertFalse(Path(conditioning).exists())
        pending = self.f.runtime.collect(authority, rid)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.raw, result['data'])
        reference, audition = pending.proposal['manifest']['outputs']
        self.assertEqual((reference['sha256'], reference['frames']), (self.asset['sha256'], 4))
        self.assertEqual((audition['sha256'], audition['frames']), (result['sha256'], 1))
        directory = self.f.spool/authority.binding
        self.assertEqual((directory/'import-reference').read_bytes(), self.f.reference_raw)
        self.assertEqual((directory/'import').read_bytes(), result['data'])

    def test_preparation_proposal_keeps_distinct_role_media_metadata(self):
        from worker import native_child, native_protocol as p
        snapshot = self.prepared(); spec = p.native_execution(snapshot[1])
        # Deliberately distinct digest and frame count prevent reference-as-audition
        # adapters from satisfying the role contract accidentally.
        result = dict(sha256='a'*64, frames=3)
        value = native_child.proposal(snapshot, result, spec)
        reference, audition = value['manifest']['outputs']
        self.assertEqual((reference['sha256'], reference['frames']), (self.asset['sha256'], 4))
        self.assertEqual((audition['sha256'], audition['frames']), ('a'*64, 3))
        roles = value['manifest']['candidates'][0]['reference_roles']
        self.assertEqual(roles['reference']['asset'], reference)
        self.assertEqual(roles['audition']['asset'], audition)
        self.assertEqual(roles['reference']['transcript'], self.request['reference_text'])
        self.assertEqual(roles['audition']['text'], self.request['audition_text'])

    def test_saved_reuse_rederives_exact_reference_and_settings(self):
        from worker.native_protocol import owned_clone_context
        from worker.synthetic import candidates
        from worker import native_protocol as p
        snapshot=self.prepared();spec=p.native_execution(snapshot[1])
        candidate=candidates(dict(plan=snapshot[1]['plan'],job_id='candidate-job'),
            dict(id='audition',sha256='0'*64),dict(self.asset))
        # Use the exact immutable shape created by VoiceStore, without registering it.
        version=copy.deepcopy(candidate[0]['version']);version.update(id='saved-clone',voice_id='mine',reference_asset_sha256={self.asset['id']:self.asset['sha256']})
        version['configuration']['reference_asset_id']=self.asset['id']
        request=dict(self.f.snapshot[1]['plan']['request'],voice_version_id=version['id'])
        context=owned_clone_context(self.f.snapshot[1],'alice',request,self.asset,version=version)
        self.assertEqual(p.native_execution(context).reference_text,self.request['reference_text'])
        for mutate in (lambda v:v['configuration'].update(reference_text='different'),
                lambda v:v['identity'].update(model='wrong'),lambda v:v['settings'].update(top_k=2),
                lambda v:v.update(reference_asset_sha256={self.asset['id']:'0'*64}),lambda v:v['provenance'].update(source_id='forged')):
            changed=copy.deepcopy(version);mutate(changed)
            with self.assertRaises((ValueError,Refused)):
                owned_clone_context(self.f.snapshot[1],'alice',request,self.asset,version=changed)

    def launch_preparation(self, *, execute=True):
        from service.worker_transport import Authority
        authority=Authority(*self.prepared())
        raw=self.f.runtime.detach(authority,self.core)
        rid=self.f.runtime.prepare(authority,detached=raw)
        with patch('worker.native_child.parent_exchange',side_effect=self.f.exchange if execute else lambda *args:None):
            self.f.runtime.start(rid)
            if execute:self.f.finish()
        return authority,rid

    def test_native_rechecks_reference_bytes_before_launch(self):
        from service.worker_transport import Authority
        authority=Authority(*self.prepared());runtime=self.f.runtime
        raw=runtime.detach(authority,self.core)
        with self.assertRaises(Refused):runtime.prepare(authority,detached=raw[:-2]+b'xx')
        rid=runtime.prepare(authority,detached=raw)
        reg=self.f.s.journal.value['records'][0]
        native=self.f.s.directory(reg)/('input-'+self.asset['sha256'])
        native.write_bytes(raw[:-2]+b'xx')
        with self.assertRaises(Refused):runtime.start(rid)
        self.assertEqual(self.f.f.calls,[])
        self.assertEqual((self.core.root/'assets'/self.asset['id']).read_bytes(),raw)

    def test_preparation_reference_retained_until_confirmed_cancel_reap(self):
        authority,rid=self.launch_preparation(execute=False)
        self.f.s.active['receiver'].join(1)
        row=self.f.s.journal.value['records'][0]
        paths=(self.core.root/'assets'/self.asset['id'],self.f.spool/authority.binding/'input',
            self.f.s.directory(row)/('input-'+self.asset['sha256']))
        self.f.runtime.terminate(rid)
        self.assertIsNone(self.f.s.journal.value['records'][0]['receipt'])
        self.assertEqual(self.f.runtime.collect(authority,rid),None)
        for path in paths:self.assertEqual(path.read_bytes(),self.f.reference_raw)
        self.f.f.h.code=-15;self.f.s.tick()
        receipt,observation=self.f.runtime.collect(authority,rid)
        self.assertTrue(receipt['reaped']);self.assertFalse(receipt['timely']);self.assertIsNone(observation)
        for path in paths:self.assertEqual(path.read_bytes(),self.f.reference_raw)
        self.assertFalse(list(self.f.stage.glob('*/reference.wav')))

    def test_pair_import_reference_identity_failure_retains_uncertainty(self):
        authority,rid=self.launch_preparation()
        pending=self.f.runtime.collect(authority,rid)
        directory=self.f.spool/authority.binding
        reference=directory/'import-reference';raw=reference.read_bytes();reference.write_bytes(raw[:-2]+b'xx')
        with self.assertRaisesRegex(Refused,'native_control_storage'):self.f.runtime.finalize_collect(pending)
        self.assertTrue(self.f.runtime.failed)
        self.assertIn(authority.binding,self.f.runtime.import_closed)
        self.assertFalse((directory/'import-complete').exists())
        before={p:p.read_bytes() for p in directory.iterdir()}
        reopened=self.f.make_runtime()
        self.assertFalse(reopened.admission_open)
        self.assertEqual(before,{p:p.read_bytes() for p in directory.iterdir()})

    def test_bounded_canonical_pcm_only_and_explicit_provisioned_capability(self):
        from worker.engines.qwen_prepare import QwenPreparation, selected_reference
        from worker import native_protocol as p
        spec=p.native_execution(self.prepared()[1])
        self.assertEqual(selected_reference(spec,self.f.reference_raw),(self.f.reference_raw,4))
        for changes in (dict(x_vector_only=True),dict(phase='generate'),dict(reference_bytes=1048577),dict(task='custom_voice')):
            with self.assertRaises(Refused):QwenPreparation(**dict(spec.identity(),**changes))
        # A RIFF header with a different PCM marker is a byte mismatch, even when
        # every media shape field still agrees.
        with self.assertRaises(Refused):selected_reference(spec,self.f.reference_raw[:-2]+b'xx')
        contexts=self.contexts
        self.assertIn('voice_prepare',contexts.capabilities('alice')['operations'])
        original=copy.deepcopy(contexts._contexts)
        context=json.loads(next(iter(original.values())))
        context['plan']['resolved']['profile']['compatibility']=[c for c in context['plan']['resolved']['profile']['compatibility'] if c['operation']!='voice_prepare']
        contexts._contexts={('alice','qwen-mps','vv0'):json.dumps(context)}
        self.assertNotIn('voice_prepare',contexts.capabilities('alice')['operations'])
        with self.assertRaises(Refused):contexts.resolve_owned(self.core,'alice','voice_prepare',self.request)
        contexts._contexts=original|{('alice','qwen-mps','ambiguous'):next(iter(original.values()))}
        self.assertNotIn('voice_prepare',contexts.capabilities('alice')['operations'])
        with self.assertRaises(Refused):contexts.resolve_owned(self.core,'alice','voice_prepare',self.request)
