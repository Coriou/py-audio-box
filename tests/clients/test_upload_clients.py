"""Client-host no-follow upload permission and actual ASGI parity."""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'service'))
from test_upload_api import UploadFixture, wav_bytes
from clients.http import HTTP
from clients.cli import main
from clients import mcp
from service.catalogue import Failure


class UploadClientTests(UploadFixture, unittest.TestCase):
    def test_http_cli_mcp_actual_upload_and_errors(self):
        outer=self
        class ASGIClient:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def post(self,path,content,headers):
                async def run():
                    async def chunks():
                        for chunk in content: yield chunk
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer.app),base_url='http://test') as client:
                        return await client.post(path,content=chunks(),headers=headers)
                return asyncio.run(run())
            def get(self,path,params=None,headers=None):
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer.app),base_url='http://test') as client:
                        return await client.get(path,params=params,headers=headers)
                return asyncio.run(run())
        root=Path(self.tmp.name).resolve()/'upload';root.mkdir()
        source=root/'ref.wav';source.write_bytes(wav_bytes())
        with patch('clients.http.httpx.Client',ASGIClient):
            transport=HTTP(credential='writer')
            asset=transport.upload_file(source)
            self.assertEqual(transport.call('asset_get',{'asset_id':asset['id']}),asset)
            with patch.dict(os.environ,{'SPEECH_CREDENTIAL':'writer','SPEECH_UPLOAD_ROOT':str(root)}):
                output=io.StringIO()
                with contextlib.redirect_stdout(output): self.assertEqual(main(['asset','upload',str(source)]),0)
                self.assertEqual(json.loads(output.getvalue())['sha256'],asset['sha256'])
                response=asyncio.run(mcp.call_tool('speech_asset_upload_file',{'path':'ref.wav'}))
                self.assertFalse(response.isError,response)
                self.assertEqual(response.structuredContent['sha256'],asset['sha256'])
                self.assertIn('speech_asset_upload_file',{tool.name for tool in mcp.tools()})
                versions=transport.call('voice_versions_list',dict(voice_id=self.version['voice_id'],limit=20,cursor=''))
                response=asyncio.run(mcp.call_tool('speech_voice_versions_list',dict(voice_id=self.version['voice_id'],limit=20,cursor='')))
                self.assertEqual(response.structuredContent,versions)
                output=io.StringIO()
                with contextlib.redirect_stdout(output): self.assertEqual(main(['voice','versions',self.version['voice_id']]),0)
                self.assertEqual(json.loads(output.getvalue()),versions)
                for path in ('../ref.wav',str(source)):
                    response=asyncio.run(mcp.call_tool('speech_asset_upload_file',{'path':path}))
                    self.assertTrue(response.isError)
                    self.assertEqual(response.structuredContent['code'],'invalid_request')
            with self.assertRaises(Failure) as caught: HTTP(credential='alice-secret').upload_file(source)
            self.assertEqual(caught.exception.body['code'],'forbidden')
            self.credentials.revoke('writer')
            with self.assertRaises(Failure) as caught: transport.upload_file(source)
            self.assertEqual(caught.exception.body['code'],'forbidden')

    def test_client_local_root_symlink_nonregular_and_default_off(self):
        root=Path(self.tmp.name).resolve()/'upload'; root.mkdir()
        (root/'ref.wav').write_bytes(wav_bytes())
        (root/'link.wav').symlink_to(root/'ref.wav')
        (root/'directory').mkdir()
        (root/'dirlink').symlink_to(root/'directory')
        os.mkfifo(root/'pipe')
        with patch('clients.http.httpx.Client',side_effect=AssertionError('unsafe local file reached HTTP')):
            for path in ('../ref.wav','link.wav','dirlink/ref.wav','directory','pipe','missing.wav'):
                with self.subTest(path=path),self.assertRaises(Failure):
                    HTTP(credential='writer').upload_file(path,root=root)
        with patch.dict(os.environ,{'SPEECH_UPLOAD_ROOT':''}):
            self.assertNotIn('speech_asset_upload_file',{tool.name for tool in mcp.tools()})
            response=asyncio.run(mcp.call_tool('speech_asset_upload_file',{'path':'ref.wav'}))
            self.assertEqual(response.structuredContent['code'],'forbidden')
