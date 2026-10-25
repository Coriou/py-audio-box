"""Empty catalogue public native journey; fake TLS/process/backend only."""
import asyncio
import copy
import hashlib
import base64
import json
import time
import unittest
from unittest.mock import patch
from clients.http import HTTP
from clients import mcp
from contracts.scenarios import preset_registration
from service.catalogue import Failure
from tests.release import test_native_launch as launch
from tests.service import test_installed_preset as installed
from worker.transport import canonical, Refused


class NativeOnboardingTests(unittest.TestCase):
    def setUp(self):
        launch.NativeLaunchTests.setUp(self)
        # NativeFixture binds its fake supervisor's native identity after the
        # constructor; mirror that explicit fixture enrollment in both stores.
        self.f.s._preset_templates = copy.deepcopy(self.f.s.registrations)
    write = launch.NativeLaunchTests.write
    enroll = launch.NativeLaunchTests.enroll
    app = launch.NativeLaunchTests.app
    connector = launch.NativeLaunchTests.connector
    bridge = installed.InstalledPresetTests.bridge
    invoke_cli = installed.InstalledPresetTests.invoke_cli

    def prepare(self):
        self.f.snapshot[0]['principal_id'] = 'alice'
        self.f.snapshot[3][0]['limits']['disk_bytes'] = 2*1024**2
        self.f.snapshot[1]['plan']['resolved']['profile']['limits']['disk_bytes'] = 2*1024**2
        self.f.s.registrations[self.f.a.binding]['snapshot'] = copy.deepcopy(self.f.snapshot)
        self.f.s._preset_templates = copy.deepcopy(self.f.s.registrations)
        self.write('templates', canonical(self.templates))
        self.f.s.utc = lambda: __import__('datetime').datetime.now(__import__('datetime').timezone.utc)
        def before(q):
            if q['op'] in ('observe','collect'):
                if self.f.s.active: self.f.finish()
                self.f.s.tick()
        self.f.before = before
        self.app()
        self.write('credentials',canonical(dict(version=1,credentials=[
            dict(token='x'*64,principal='alice',scopes=['read','production-write']),
            dict(token='r'*64,principal='alice',scopes=['read'])])))
        return launch.launch.create_native_app(record_file=str(self.record),
            configuration_file=str(self.root/'control'),templates_file=str(self.root/'templates'),
            owner_root=str(self.owner),staging=str(self.f.stage),credential_file=str(self.credential))

    def wait(self, fn):
        end = time.monotonic()+5
        while time.monotonic()<end:
            if value := fn(): return value
            time.sleep(.01)
        self.fail('bounded fixture wait expired')

    def test_fresh_public_registration_generation_artifact_restart_replay(self):
        app = self.prepare()
        def journey(reopen=False):
            with self.bridge(app), patch.dict('os.environ', {'SPEECH_CREDENTIAL':'x'*64}):
                http = HTTP(credential='x'*64)
                request = preset_registration(self.f.f.e.preset, 'my.voice')
                if not reopen:
                    self.assertEqual(http.call('voices_list',dict(limit=20,cursor=''))['items'], [])
                    self.assertEqual(http.get('/skeleton')['configured_presets'][0]['engine'],'qwen')
                saved = self.invoke_cli(['voice','preset-register','--request',json.dumps(request),'--idempotency-key','onboard'])
                m = asyncio.run(mcp.call_tool('speech_voice_preset_register',dict(request,idempotency_key='onboard')))
                self.assertFalse(m.isError,m.structuredContent); self.assertEqual(saved,m.structuredContent)
                self.assertEqual(saved,http.call('voice_preset_register',request,key='onboard'))
                version=saved['version']
                self.assertEqual(version['configuration']['mode'],'qwen_preset')
                payload=dict(self.f.snapshot[1]['plan']['request'],voice_version_id=version['id'])
                receipt=http.call('speech_generate',payload,key='generate')
                if reopen: self.assertEqual(receipt,self.receipt)
                self.receipt=receipt
                job=self.wait(lambda:(j if (j:=http.call('job_get',dict(job_id=receipt['job_id'])))['state'] in ('succeeded','failed') else None))
                self.assertEqual(job['state'],'succeeded',job)
                self.wait(lambda:app.state.dispatcher.call('status')['phase']=='idle')
                artifact=http.call('artifact_get',dict(artifact_id=job['artifact_ids'][0]))
                asset=http.call('asset_get',dict(asset_id=artifact['asset_ids'][0]))
                chunk=http.call('asset_content',dict(asset_id=asset['id'],start=0,end_exclusive=1))
                content=http.call('asset_content',dict(asset_id=asset['id'],start=0,end_exclusive=chunk['total_bytes']))
                self.assertEqual(hashlib.sha256(base64.b64decode(content['content_base64'])).hexdigest(),asset['sha256'])
                for cred in (None,'wrong','r'*64):
                    with self.assertRaises(Failure):HTTP(credential=cred).call('voice_preset_register',request,key='denied')
                for change,key,code in ((dict(preset='wrong'),'wrong','unsupported_control'),
                        (dict(voice_id='other'),'onboard','idempotency_conflict'),
                        (dict(expected_revision=1),'onboard','idempotency_conflict'),
                        ({},'new-key','version_conflict'),
                        (dict(voice_id='other',expected_revision=1),'cas','revision_conflict')):
                    with self.assertRaises(Failure) as caught:http.call('voice_preset_register',dict(request,**change),key=key)
                    self.assertEqual(caught.exception.body['code'],code)
                contexts=app.state.dispatcher.production_contexts
                with self.assertRaises(Refused):contexts.preset_version('bob',request['preset'],'my.voice')
                for mutate in (lambda v:v.update(id='forged'),lambda v:v['identity'].update(model='wrong'),lambda v:v['configuration'].update(speaker='wrong')):
                    changed=copy.deepcopy(version);mutate(changed)
                    with self.assertRaises(Refused):contexts.resolve_registered('alice',payload,changed)
        async def run():
            for reopen in (False,True):
                async with app.router.lifespan_context(app): await asyncio.to_thread(journey,reopen)
                self.assertFalse(app.state.dispatcher.thread.is_alive())
        with self.connector(), patch('worker.native_child.parent_exchange',side_effect=self.f.exchange): asyncio.run(run())
        self.assertEqual(len(self.f.f.calls),1)

    def test_supervisor_derivation_is_exact_principal_model_profile_and_trust(self):
        from worker.native_protocol import owned_preset_context
        from service.worker_transport import Authority
        from contracts import validate as oracle
        template=copy.deepcopy(self.f.snapshot)
        principal=template[0]['principal_id']
        template[1]=owned_preset_context(template[1],principal,self.f.f.e.preset,'fresh')
        template[0]['input_sha256']=oracle.plan_hash(template[1]['plan'])
        def forged_version(s):
            s[1]['plan']['voice_versions'][0]['id']='forged'
            s[1]['plan']['request']['voice_version_id']='forged'
            s[1]['execution']['descriptor']['voice_revision']='forged'
        for mutate in (lambda s:s[0].update(principal_id='foreign'),
                lambda s:s[1]['execution']['native'].update(boot='e'*32),
                lambda s:s[1]['execution']['descriptor'].update(preset='wrong'),
                lambda s:s[1]['plan']['resolved']['profile'].update(config_revision='forged'),
                forged_version):
            changed=copy.deepcopy(template);mutate(changed)
            changed[0]['input_sha256']=oracle.plan_hash(changed[1]['plan'])
            with self.assertRaises((Refused,ValueError)):self.f.s.register_snapshot(changed)
        self.assertEqual(self.f.s.register_snapshot(template),Authority(*template).binding)
        self.assertEqual(len(self.f.s._preset_templates),1)

    def test_original_and_owned_registration_require_canonical_primitives(self):
        from worker.native_protocol import owned_preset_context
        from contracts import validate as oracle
        original = copy.deepcopy(self.f.snapshot)
        owned = copy.deepcopy(original)
        owned[1] = owned_preset_context(original[1], original[0]['principal_id'],
                                       self.f.f.e.preset, 'canonical')
        owned[0]['input_sha256'] = oracle.plan_hash(owned[1]['plan'])
        for snapshot in (original, owned):
            changed = copy.deepcopy(snapshot)
            self.assertIs(type(changed[1]['execution']['descriptor']['top_p']), float)
            changed[1]['execution']['descriptor']['top_p'] = 1
            self.assertEqual(snapshot, changed)
            self.assertNotEqual(canonical(snapshot), canonical(changed))
            for enrolled in (False, True, True):
                if enrolled:
                    binding = self.f.s.register_snapshot(snapshot)
                    self.assertEqual(self.f.s.register_snapshot(snapshot), binding)
                with self.assertRaises(Refused) as caught:
                    self.f.s.register_snapshot(changed)
                self.assertEqual(str(caught.exception), 'native_registration')
        self.assertEqual(len(self.f.s._preset_templates), 1)

    def test_existing_binding_snapshot_join_requires_canonical_primitives(self):
        binding = self.f.s.register_snapshot(self.f.snapshot)
        changed = copy.deepcopy(self.f.snapshot)
        changed[1]['execution']['descriptor']['top_p'] = 1
        with self.assertRaises(Refused):
            self.f.s.registration(dict(binding=binding, snapshot=changed))
        self.f.s.registration(dict(binding=binding, snapshot=self.f.snapshot))

    def test_saved_original_and_owned_versions_require_canonical_primitives(self):
        from service.execution_context import NativeQwenContexts
        original = copy.deepcopy(self.templates['registrations'][0]['context'])
        principal = self.f.snapshot[0]['principal_id']
        contexts = NativeQwenContexts({(principal, 'qwen-mps', 'vv0'): original},
            self.f.f.inventory, self.templates['profiles'], self.f.native)
        for version in (original['plan']['voice_versions'][0],
                        contexts.preset_version(principal, self.f.f.e.preset, 'canonical')):
            request = dict(original['plan']['request'], voice_version_id=version['id'])
            contexts.resolve_registered(principal, request, version)
            changed = copy.deepcopy(version)
            self.assertIs(type(changed['settings']['top_p']), float)
            changed['settings']['top_p'] = 1
            self.assertEqual(version, changed)
            self.assertNotEqual(canonical(version), canonical(changed))
            for _ in range(2):
                with self.assertRaises(Refused):
                    contexts.resolve_registered(principal, request, changed)
                contexts.resolve_registered(principal, request, version)

    def test_ambiguous_selector_and_clone_template_never_become_presets(self):
        from service.execution_context import NativeQwenContexts
        original=copy.deepcopy(self.templates['registrations'][0]['context'])
        contexts=NativeQwenContexts({('alice','qwen-mps','vv0'):original},self.f.f.inventory,self.templates['profiles'],self.f.native)
        contexts._contexts[('alice','qwen-mps','another')]=json.dumps(original)
        self.assertEqual(contexts.presets('alice'),[])
        with self.assertRaises(Refused):contexts.preset_version('alice',self.f.f.e.preset,'fresh')
        from dataclasses import replace
        from worker.native_protocol import owned_preset_context
        clone=replace(self.f.f.e,task='clone',preset=None,reference_id='ref',reference_sha256='f'*64,
            reference_bytes=46,reference_frames=1,reference_rate=24000,reference_text='reference')
        original['execution']['descriptor']=clone.identity()
        with self.assertRaises(Refused):owned_preset_context(original,'alice',None,'fresh')
