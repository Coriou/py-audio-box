"""Fresh installed-fixture public bootstrap; injected host/runtime only, no ML/listener."""
import asyncio
import base64
import contextlib
import copy
import hashlib
import io
import json
import unittest
from unittest.mock import patch
import httpx
from clients import cli, mcp
from clients.http import HTTP
from contracts.scenarios import preset_registration
from service.api import create_voice_app
from service.auth import Credential, StaticCredentials
from service.catalogue import Failure, OPERATIONS
from jsonschema.exceptions import ValidationError
from tests.service import test_installed_activation as activation


class InstalledPresetTests(unittest.TestCase):
    bootstrap = False
    setUp = activation.ActivationTests.setUp
    notification_directory = activation.ActivationTests.notification_directory
    proposal = activation.ActivationTests.proposal
    owner = activation.ActivationTests.owner
    wait = activation.ActivationTests.wait

    def open(self):
        owner = self.owner()
        self.addCleanup(lambda: owner.close() if owner.thread.is_alive() else None)
        self.wait(lambda: owner.call('status')['persistent_activation'])
        credentials = StaticCredentials([
            Credential('writer', 'alice', frozenset({'read','production-write'})),
            Credential('reader', 'alice', frozenset({'read'})),
            Credential('foreign', 'bob', frozenset({'read','production-write'}))])
        app = create_voice_app(owner, credentials=credentials, installed=self.fixture.config)
        return owner, app

    def bridge(self, app):
        class Client:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def request(self, method, path, **kwargs):
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://inert') as client:
                        return await client.request(method, path, **kwargs)
                return asyncio.run(run())
            def get(self, path, **kwargs): return self.request('GET',path,**kwargs)
            def post(self, path, **kwargs): return self.request('POST',path,**kwargs)
        return patch('clients.http.httpx.Client', Client)

    def request(self):
        return preset_registration(self.context['execution']['descriptor']['preset'], 'first-voice')

    def invoke_cli(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output): code = cli.main(args)
        self.assertEqual(code, 0, output.getvalue())
        return json.loads(output.getvalue())

    def test_empty_public_journey_same_root_reopen_and_frozen_job(self):
        owner, app = self.open()
        with self.bridge(app), patch.dict('os.environ', {'SPEECH_CREDENTIAL':'writer'}):
            http = HTTP(credential='writer')
            self.assertEqual(http.call('voices_list',dict(limit=20,cursor=''))['items'], [])
            with self.assertRaises(Failure) as missing:
                http.call('speech_generate', self.context['plan']['request'], key='before-bootstrap')
            self.assertEqual(missing.exception.body['code'],'not_found')
            status = http.get('/skeleton')
            self.assertFalse(status['clone_prepare'])
            self.assertEqual(status['configured_presets'][0]['preset'],self.request()['preset'])
            self.assertNotIn('voice_prepare',http.call('discover',{})['operations'])
            detail = http.get('/skeleton/operations/voice_prepare')
            self.assertEqual(detail['availability'],'unavailable_configured_profile')
            request = self.request()
            saved = self.invoke_cli(['voice','preset-register','--request',json.dumps(request),'--idempotency-key','bootstrap'])
            replay = asyncio.run(mcp.call_tool('speech_voice_preset_register',dict(request,idempotency_key='bootstrap')))
            self.assertFalse(replay.isError,replay.structuredContent)
            self.assertEqual(saved,replay.structuredContent)
            self.assertEqual(saved,http.call('voice_preset_register',request,key='bootstrap'))
            version = saved['version']
            self.assertEqual(http.call('voices_list',dict(limit=20,cursor=''))['items'],['first-voice'])
            self.assertEqual(http.call('voice_version_get',dict(voice_version_id=version['id'])),version)
            self.assertEqual(http.call('voice_versions_list',dict(voice_id='first-voice',limit=20,cursor=''))['items'],[version['id']])
            generation = dict(self.context['plan']['request'],voice_version_id=version['id'])
            receipt = http.call('speech_generate',generation,key='first-generation')
            self.wait(lambda:self.runtime.authority)
            frozen = copy.deepcopy(self.runtime.authority.context)
            self.runtime.finish.set()
            job = self.wait(lambda:(j if (j:=http.call('job_get',dict(job_id=receipt['job_id'])))['state']=='succeeded' else None))
            self.wait(lambda:owner.call('status')['phase']=='idle')
            events = self.invoke_cli(['job','events',job['id']])
            self.assertTrue(events['items'] if 'items' in events else events['events'])
            waited = asyncio.run(mcp.call_tool('speech_job_wait',dict(job_id=job['id'],cursor='',wait_seconds=0)))
            self.assertFalse(waited.isError, waited.structuredContent)
            artifact = http.call('artifact_get',dict(artifact_id=job['artifact_ids'][0]))
            asset = http.call('asset_get',dict(asset_id=artifact['asset_ids'][0]))
            chunk = http.call('asset_content',dict(asset_id=asset['id'],start=0,end_exclusive=1))
            content = http.call('asset_content',dict(asset_id=asset['id'],start=0,end_exclusive=chunk['total_bytes']))
            self.assertEqual(hashlib.sha256(base64.b64decode(content['content_base64'])).hexdigest(),asset['sha256'])
            second = http.call('voice_preset_register',dict(request,voice_id='second-voice'),key='second')
            self.assertNotEqual(second['version']['id'],version['id'])
            self.assertEqual(self.runtime.authority.context,frozen)
        owner.close()
        # Reopen this exact completed installed root, retaining actual SQLite job,
        # catalogue, dispatch/reap/ack history rather than rebuilding a fixture.
        self.runtime.authority = None
        owner2, app2 = self.open()
        with self.bridge(app2):
            http = HTTP(credential='writer')
            self.assertEqual(http.call('voice_version_get',dict(voice_version_id=version['id'])),version)
            self.assertEqual(http.call('job_get',dict(job_id=job['id'])),job)
            replay = http.call('speech_generate',generation,key='first-generation')
            self.assertEqual(replay,receipt)
            self.assertIsNone(self.runtime.authority)
            self.assertEqual(http.call('artifact_get',dict(artifact_id=artifact['id'])),artifact)
            receipt2 = http.call('speech_generate',generation,key='after-restart')
            self.wait(lambda:self.runtime.authority)
            self.assertEqual(self.runtime.authority.context['plan']['voice_versions'],[version])
            self.wait(lambda:http.call('job_get',dict(job_id=receipt2['job_id']))['state']=='succeeded')
            self.wait(lambda:owner2.call('status')['phase']=='idle')

    def test_replay_conflicts_auth_selection_and_activation(self):
        owner, app = self.open()
        with self.bridge(app):
            http = HTTP(credential='writer'); request=self.request()
            for credential in (None,'reader','foreign'):
                with self.subTest(credential=credential), self.assertRaises(Failure):
                    HTTP(credential=credential).call('voice_preset_register',request,key='bootstrap')
            for payload in (dict(request,preset='wrong'),dict(request,model='arbitrary'),dict(request,path='/arbitrary'),dict(request,readiness='ready')):
                with self.subTest(payload=payload), self.assertRaises((Failure, ValidationError)):
                    http.call('voice_preset_register',payload,key='bad')
            saved=http.call('voice_preset_register',request,key='bootstrap')
            for payload,key,code in ((dict(request,voice_id='other'),'bootstrap','idempotency_conflict'),
                                     (dict(request,expected_revision=1),'bootstrap','idempotency_conflict'),
                                     (request,'different-key','version_conflict')):
                with self.assertRaises(Failure) as conflict:http.call('voice_preset_register',payload,key=key)
                self.assertEqual(conflict.exception.body['code'],code)
            with self.assertRaises(Failure):HTTP(credential='foreign').call('voice_version_get',dict(voice_version_id=saved['version']['id']))
            owner.call('close')
            owner.thread.join(4)
            self.assertFalse(owner.thread.is_alive())
        # Context-level mismatched model/identity cannot inherit preset authority.
        for mutate in (lambda v:v['identity'].update(model='wrong'),
                       lambda v:v.update(id='arbitrary-import'),
                       lambda v:v['configuration'].update(preset='wrong')):
            version=copy.deepcopy(saved['version']);mutate(version)
            with self.assertRaisesRegex(ValueError,'unsupported_control'):
                self.contexts.resolve_registered('alice',dict(self.context['plan']['request'],voice_version_id=version['id']),version)

    def test_unavailable_activation_has_no_registration(self):
        self.runtime.changed_at=5
        owner=self.owner();self.addCleanup(owner.close)
        self.wait(lambda:owner.call('status')['reconciled'])
        self.assertFalse(owner.call('status')['persistent_activation'])
        with self.assertRaisesRegex(ValueError,'not_ready'):
            owner.call('production_preset_register','alice',self.request(),key='closed')
        self.assertEqual(owner.call('voice_list','alice'),[])


if __name__ == '__main__': unittest.main()
