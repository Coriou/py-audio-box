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
            Credential('write-secret', 'alice', frozenset({'production-write'}))])
        self.app = create_voice_app(self.dispatcher, credentials=self.credentials)

    def request(self, path, token='alice-secret', target=None):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=target or self.app),base_url='http://test') as client:
                return await client.get(path, headers={'Authorization':'Bearer '+token} if token else {})
        return asyncio.run(run())

class VoiceAPITests(VoiceAPIFixture, unittest.TestCase):
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

    def test_discovery_and_lazy_schema(self):
        with patch.object(self.dispatcher,'call',side_effect=AssertionError('discovery accessed owner')):
            body = self.request('/v1/capabilities').json()
            self.assertEqual(body['profiles'],[])
            self.assertIn('voice_get',body['operations'])
            self.assertIn('asset_get',body['operations'])
            self.assertFalse(self.request('/skeleton/operations/voice_get').json()['fixture'])
        code = "import sys; import service.api; assert not any(x in sys.modules for x in ('torch','transformers','service.core','service.dispatcher'))"
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
