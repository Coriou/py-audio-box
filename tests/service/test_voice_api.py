"""In-process ASGI and actual local owner; no listeners or inference."""
import asyncio
import json
import queue
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import httpx
from contracts.validate import validate
from service.api import app, create_voice_app
from service.auth import Credential, StaticCredentials
from service.core import Core
from service.dispatcher import Dispatcher
from test_voice_dispatcher import VoiceDispatcherFixture

class VoiceAPIFixture(VoiceDispatcherFixture):
    def setUp(self):
        super().setUp()
        self.credentials = StaticCredentials([
            Credential('alice-secret', 'alice', frozenset({'read'})),
            Credential('bob-secret', 'bob', frozenset({'read'})),
            Credential('write-secret', 'alice', frozenset({'production-write'})),
            Credential('rw-secret', 'alice', frozenset({'read', 'production-write'}))])
        self.app = create_voice_app(self.dispatcher, credentials=self.credentials)

    def request(self, path, token='alice-secret', target=None):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=target or self.app),base_url='http://test') as client:
                return await client.get(path, headers={'Authorization':'Bearer '+token} if token else {})
        return asyncio.run(run())

class VoiceAPITests(VoiceAPIFixture, unittest.TestCase):
    def test_status_alias_auth_query_and_envelope_parity(self):
        for target in (self.app, app):
            for token in ('alice-secret', '', 'bad', 'write-secret'):
                for query in ('', '?scan=true'):
                    with self.subTest(fixture=target is app, token=token, query=query):
                        canonical = self.request('/v1/status' + query, token, target)
                        old = self.request('/skeleton' + query, token, target)
                        self.assertEqual(canonical.status_code, old.status_code)
                        self.assertEqual(canonical.json(), old.json())
                        self.assertLessEqual(len(canonical.content), 2048)
                        if target is self.app and token != 'alice-secret':
                            self.assertEqual(canonical.status_code, 403)
                        elif query:
                            self.assertEqual(canonical.status_code, 422)
                        else:
                            self.assertEqual(canonical.status_code, 200)
                            self.assertFalse(canonical.json()['ready_for_speech'])
        self.credentials.revoke('alice-secret')
        for path in ('/v1/status', '/skeleton'):
            self.assertEqual(self.request(path).status_code, 403)

    def advertised_errors(self, operation):
        response = self.request('/skeleton/operations/'+operation)
        self.assertEqual(response.status_code, 200)
        detail = response.json()
        self.assertFalse(detail['fixture'])
        self.assertEqual(detail['operation']['id'], operation)
        return detail['operation']['errors']

    def assert_operation_error(self, response, operation, code, status):
        self.assertEqual(response.status_code, status)
        body = validate('error', response.json())
        self.assertEqual(body['code'], code)
        self.assertIn(body['code'], self.advertised_errors(operation))
        self.assertLess(len(response.content), 2048)
        self.assertNotIn('private-path', response.text)

    def test_read_error_declarations_and_sanitized_readiness(self):
        routes = {
            'voices_list': '/v1/voices',
            'voice_get': '/v1/voices/'+self.version['voice_id'],
            'voice_version_get': '/v1/voice-versions/'+self.version['id'],
        }
        for operation, path in routes.items():
            with self.subTest(operation=operation):
                self.assertEqual(self.advertised_errors(operation),
                                 ['invalid_request', 'forbidden', 'not_found',
                                  'not_ready', 'resource_limit'])
                for failure in (OSError, queue.Full, TimeoutError):
                    with self.subTest(failure=failure.__name__):
                        with patch.object(self.dispatcher, 'call',
                                          side_effect=failure('private-path')):
                            response = self.request(path)
                        self.assert_operation_error(response, operation, 'not_ready', 503)

    def test_auth_principal_revocation_and_fixture_separation(self):
        for token in ('', 'bad', 'write-secret'):
            response = self.request('/v1/voices', token)
            self.assertEqual(response.status_code,403)
            self.assertEqual(response.json()['code'],'forbidden')
        self.assertEqual(self.request('/v1/voices').json()['items'],[self.version['voice_id']])
        self.assertEqual(self.request('/v1/voices/'+self.version['voice_id'],'bob-secret').status_code,404)
        self.assertEqual(self.request('/v1/voice-versions/'+self.version['id'],'bob-secret').status_code,404)
        self.credentials.revoke('alice-secret')
        self.assertEqual(self.request('/v1/capabilities').status_code,403)
        fixture = self.request('/v1/voices', '', app)
        self.assertEqual(fixture.json()['items'],[])
        self.assertIn('X-Speech-Fixture',fixture.headers)
        self.assertEqual(self.request('/v1/assets/short','bob-secret').status_code,404)

    def test_page_validation_detail_and_bounded_failures(self):
        first = self.request('/v1/voices?limit=1').json()
        self.assertEqual(len(first['items']),1)
        self.assertEqual(self.request('/v1/voices?limit=1&cursor='+first['next_cursor']).json()['items'],[])
        for query in ('limit=21','cursor=bad','principal=bob','cursor=99999999999999999'):
            self.assertEqual(self.request('/v1/voices?'+query).status_code,422)
        self.assertEqual(self.request('/v1/voices/'+self.version['voice_id']).json()['revision'],1)
        self.assertEqual(self.request('/v1/voice-versions/'+self.version['id']).json(),self.version)
        # Real oversized metadata, seeded while no Dispatcher owns the root.
        self.dispatcher.close()
        core = Core(self.root)
        try:
            aliases = ['a'+str(i).zfill(3)+'x'*100 for i in range(30)]
            core.voices.aliases('alice','aliases',self.version['voice_id'],aliases,expected_revision=1)
        finally:
            core.close()
        self.dispatcher = Dispatcher(self.root,staging=Path(self.tmp.name)/'staging')
        self.app = create_voice_app(self.dispatcher,credentials=self.credentials)
        response = self.request('/v1/voices/'+self.version['voice_id'])
        self.assert_operation_error(response, 'voice_get', 'resource_limit', 429)

    def test_resolve_names_errors_isolation_and_validation(self):
        voice_id = self.version['voice_id']
        # Seed aliases and archive only while no Dispatcher owns the root.
        self.dispatcher.close()
        core = Core(self.root)
        try:
            core.voices.aliases('alice','resolve-alias',voice_id,['default-reading'],expected_revision=1)
        finally:
            core.close()
        self.dispatcher = Dispatcher(self.root,staging=Path(self.tmp.name)/'staging')
        self.app = create_voice_app(self.dispatcher,credentials=self.credentials)
        expected = self.request('/v1/voices/'+voice_id).json()
        for name in (voice_id,'default-reading'):
            self.assertEqual(self.request('/v1/voice-names/'+name).json(),expected)
            self.assert_operation_error(self.request('/v1/voice-names/'+name,'bob-secret'),'voice_resolve','not_found',404)
        self.assert_operation_error(self.request('/v1/voice-names/unknown'),'voice_resolve','not_found',404)
        for suffix in ('bad!','default-reading?principal=bob'):
            self.assert_operation_error(self.request('/v1/voice-names/'+suffix),'voice_resolve','invalid_request',422)
        self.assert_operation_error(self.request('/v1/voice-names/default-reading','write-secret'),'voice_resolve','forbidden',403)
        with patch('service.catalogue.VOICE_RESPONSE_BYTES',1):
            self.assert_operation_error(self.request('/v1/voice-names/default-reading'),'voice_resolve','resource_limit',429)
        async def mutate(operation, payload, key):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://test') as client:
                return await client.post('/v1/operations/'+operation,
                    headers={'Authorization':'Bearer rw-secret','Idempotency-Key':key},json=payload)
        archived = asyncio.run(mutate('voice_archive',dict(voice_id=voice_id,expected_revision=2),'resolve-archive'))
        self.assertEqual(archived.status_code,200,archived.text)
        self.assertEqual((archived.json()['revision'],archived.json()['aliases']),(3,[]))
        self.assert_operation_error(self.request('/v1/voice-names/'+voice_id),'voice_resolve','voice_archived',409)
        self.assert_operation_error(self.request('/v1/voice-names/default-reading'),'voice_resolve','not_found',404)
        self.dispatcher.close()
        core = Core(self.root)
        try:
            core.voices.register('alice','replacement',dict(self.version,id='replacement-version',voice_id='replacement'),expected_revision=0)
        finally:
            core.close()
        self.dispatcher = Dispatcher(self.root,staging=Path(self.tmp.name)/'staging')
        self.app = create_voice_app(self.dispatcher,credentials=self.credentials)
        response = asyncio.run(mutate('voice_aliases',
            dict(voice_id='replacement',aliases=['default-reading'],expected_revision=1),'reuse'))
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.request('/v1/voice-names/default-reading').json()['id'],'replacement')
        self.assert_operation_error(self.request('/v1/voice-names/default-reading','bob-secret'),'voice_resolve','not_found',404)
        self.assertEqual(self.request('/v1/voice-versions/'+self.version['id']).json(),self.version)

    def test_discovery_and_lazy_schema(self):
        reads={('configured_capabilities','alice'):None,
               ('configured_profiles','alice'):[],
               ('audio_export_available','alice'):False,
               ('status',):dict(retry_available=False)}
        def configured_read(*args, **kwargs):
            if args in reads and not kwargs:return reads[args]
            raise AssertionError('discovery accessed unexpected owner operation')
        with patch.object(self.dispatcher,'call',side_effect=configured_read) as owner:
            body = self.request('/v1/capabilities').json()
            self.assertEqual(body['profiles'],[])
            self.assertIn('voice_get',body['operations'])
            self.assertIn('voice_resolve',body['operations'])
            self.assertIn('asset_get',body['operations'])
            self.assertFalse(self.request('/skeleton/operations/voice_get').json()['fixture'])
            self.assertNotIn('audio_export',body['operations'])
            self.assertNotIn('job_retry',body['operations'])
            self.assertEqual([c.args for c in owner.call_args_list],
                [('configured_capabilities','alice'),('configured_profiles','alice'),
                 ('audio_export_available','alice'),('status',),('configured_capabilities','alice')])
        code = "import sys; import service.api; assert not any(x in sys.modules for x in ('torch','transformers','service.core','service.dispatcher'))"
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
