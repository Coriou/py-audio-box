"""Named adapters against actual in-process ASGI HTTP and temporary owner."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'service'))
from test_voice_api import VoiceAPIFixture
from clients.http import HTTP
from clients.cli import main
from clients import mcp
from service.catalogue import Failure

class VoiceClientTests(VoiceAPIFixture, unittest.TestCase):
    def test_client_parity_credentials_and_errors(self):
        outer = self
        requested_paths = []
        class ASGIClient:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def get(self,path,params=None,headers=None):
                requested_paths.append(path)
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer.app),base_url='http://test') as client:
                        return await client.get(path,params=params,headers=headers)
                return asyncio.run(run())
        with patch('clients.http.httpx.Client',ASGIClient):
            transport=HTTP(credential='alice-secret')
            expected=transport.call('voice_get',{'voice_id':self.version['voice_id']})
            self.assertEqual(transport.call('voice_version_get',{'voice_version_id':self.version['id']}),self.version)
            with self.assertRaises(Failure):
                HTTP().call('voice_get',{'voice_id':self.version['voice_id']})
            with patch.dict('os.environ',{'SPEECH_CREDENTIAL':'alice-secret'}):
                for args, operation, payload in [
                    (['voice','resolve',self.version['voice_id']],'voice_resolve',{'name':self.version['voice_id']}),
                    (['voices','list'],'voices_list',{'limit':20,'cursor':''}),
                    (['voice','get',self.version['voice_id']],'voice_get',{'voice_id':self.version['voice_id']}),
                    (['voice','version','get',self.version['id']],'voice_version_get',{'voice_version_id':self.version['id']})]:
                    output=io.StringIO()
                    with contextlib.redirect_stdout(output): self.assertEqual(main(args),0)
                    body=json.loads(output.getvalue())
                    self.assertEqual(body,transport.call(operation,payload))
                    tool='speech_'+operation
                    response=asyncio.run(mcp.call_tool(tool,payload))
                    self.assertFalse(response.isError)
                    self.assertEqual(response.structuredContent,body)
                output = io.StringIO()
                with contextlib.redirect_stdout(output): self.assertEqual(main(['status']), 0)
                status = json.loads(output.getvalue())
                self.assertEqual(status, transport.get('/v1/status'))
                for spelling in ('speech_status', 'speech_skeleton_status'):
                    response = asyncio.run(mcp.call_tool(spelling, {}))
                    self.assertFalse(response.isError)
                    self.assertEqual(response.structuredContent, status)
                    response = asyncio.run(mcp.call_tool(spelling, {'extra': 1}))
                    self.assertTrue(response.isError)
                    self.assertEqual(response.structuredContent['code'], 'invalid_request')
                for operation_id in ('voice_get', 'missing'):
                    requested_paths.clear()
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main(['operation-detail', operation_id])
                    response = asyncio.run(mcp.call_tool('speech_operation_detail', {'operation_id': operation_id}))
                    self.assertEqual(code, 0 if operation_id == 'voice_get' else 2)
                    self.assertEqual(response.isError, operation_id == 'missing')
                    self.assertEqual(json.loads(output.getvalue()), response.structuredContent)
                    self.assertEqual(requested_paths, ['/v1/operations/' + operation_id] * 2)
                names={t.name for t in mcp.tools()}
                self.assertIn('speech_status', names)
                self.assertNotIn('speech_skeleton_status', names)
                self.assertIn('speech_voice_version_get',names)
                self.assertIn('speech_voice_resolve',names)
                self.assertNotIn('idempotency_key',mcp.input_schema('voice_resolve')['properties'])
            with patch.dict('os.environ',{'SPEECH_CREDENTIAL':'write-secret'}):
                for spelling in ('speech_status', 'speech_skeleton_status'):
                    response = asyncio.run(mcp.call_tool(spelling, {}))
                    self.assertTrue(response.isError)
                    self.assertEqual(response.structuredContent['code'], 'forbidden')
                response=asyncio.run(mcp.call_tool('speech_voice_get',{'voice_id':self.version['voice_id']}))
                self.assertTrue(response.isError)
                self.assertEqual(response.structuredContent['code'],'forbidden')
