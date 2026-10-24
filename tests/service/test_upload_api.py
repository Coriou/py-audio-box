"""Actual ASGI/owner upload boundaries; generated WAVs and temporary Core only."""
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
import queue
import unittest
import wave
from unittest.mock import patch
import httpx
from test_voice_api import VoiceAPIFixture
from service.api import create_voice_app
from service.auth import Credential
from service.catalogue import OPERATIONS
from service.core import Core
from service.dispatcher import Dispatcher
from service.voice_store import VoiceStore
from contracts.validate import validate


def wav_bytes():
    out = io.BytesIO()
    with wave.open(out,'wb') as stream:
        stream.setparams((1,2,8000,0,'NONE','not compressed'))
        stream.writeframes(b'\0\0'*800)
    return out.getvalue()


class UploadFixture(VoiceAPIFixture):
    def setUp(self):
        super().setUp()
        self.credentials.replace([
            Credential('alice-secret','alice',frozenset({'read'})),
            Credential('writer','alice',frozenset({'read','production-write'})),
            Credential('bob-secret','bob',frozenset({'read','production-write'}))])

    async def post(self, chunks, *, headers=None, token='writer', app=None):
        async def content():
            for chunk in chunks:
                yield chunk
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app or self.app),base_url='http://test') as client:
            return await client.post('/v1/assets', content=content(), headers={
                'Authorization':'Bearer '+token,'Content-Type':'audio/wav',**(headers or {})})

    def assert_empty_media(self):
        self.assertEqual(list((self.root/'assets').iterdir()),[])
        self.assertEqual(list((self.root/'staging').iterdir()),[])
        self.dispatcher.close()
        core=Core(self.root)
        try:
            self.assertEqual(core.db.execute('SELECT count(*) FROM assets').fetchone()[0],0)
        finally:
            core.close()
            self.dispatcher=Dispatcher(self.root,staging=Path(self.tmp.name)/'staging')
        self.app=create_voice_app(self.dispatcher,credentials=self.credentials)

    def error(self, response, code):
        body=validate('error',response.json())
        self.assertEqual(body['code'],code,response.text)
        self.assertIn(code,OPERATIONS['asset_upload']['errors'])
        self.assertNotIn('private-path',response.text)


class UploadAPITests(UploadFixture, unittest.TestCase):
    def test_upload_register_compact_restart_and_download(self):
        data=wav_bytes()
        import threading
        from service.ingest import ingest_local
        observed=[]
        def ingest(core,principal,source):
            observed.append((threading.get_ident(),principal,type(source)))
            return ingest_local(core,principal,source)
        with patch('service.ingest.ingest_local',ingest):
            response=asyncio.run(self.post([data[:33],data[33:]],headers={'Content-Length':str(len(data))}))
        self.assertEqual(observed,[(self.dispatcher.thread.ident,'alice',bytes)])
        self.assertEqual(response.status_code,200,response.text)
        asset=validate('asset',response.json())
        self.assertEqual(asset['sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(asset['principal_id'],'alice')
        self.assertEqual(self.request('/v1/assets/'+asset['id'],'bob-secret').status_code,404)
        self.assertEqual(self.request('/v1/assets/short').status_code,404)
        self.dispatcher.close()
        core=Core(self.root)
        try:
            version=json.loads(Path('contracts/fixtures/v1/voice-catalogue.json').read_text())['register']['version']
            # Actual trusted fixture registration freezes uploaded reference hash.
            version['voice_id']='uploaded-voice'; version['id']='uploaded-v1'
            version.update(origin='clone',reference_asset_ids=[asset['id']],reference_text=' Exact reference. ', configuration=dict(mode='qwen_clone',reference_asset_id=asset['id'],reference_text=' Exact reference. ',conditioning='icl'))
            version['settings']['kind']='qwen_clone'
            from contracts.validate import canonical_hash
            version['settings_sha256']=canonical_hash(version['settings'])
            registered=core.voices.register('alice','uploaded',version,expected_revision=0)['version']
            for number in range(1,41):
                version['id']='uploaded-v'+str(number+1)+'x'*90
                core.voices.register('alice','history'+str(number),version,expected_revision=number)
        finally:
            core.close()
            self.dispatcher=Dispatcher(self.root,staging=Path(self.tmp.name)/'staging')
        self.app=create_voice_app(self.dispatcher,credentials=self.credentials)
        self.assertEqual(self.request('/v1/assets/'+asset['id']).json(),asset)
        content=self.request('/v1/assets/'+asset['id']+'/content?start=0&end_exclusive='+str(len(data))).json()
        self.assertEqual(base64.b64decode(content['content_base64']),data)
        self.assertEqual(content['content_sha256'],asset['sha256'])
        with patch.object(VoiceStore,'get',side_effect=AssertionError('full get called')):
            self.assertIn('uploaded-voice',self.request('/v1/voices').json()['items'])
            versions=self.request('/v1/voices/uploaded-voice/versions?limit=10').json()
            self.assertEqual(len(versions['items']),10)
            self.assertEqual(versions['items'][0],registered['id'])
            following=self.request('/v1/voices/uploaded-voice/versions?limit=10&cursor='+versions['next_cursor']).json()
            self.assertEqual(len(following['items']),10)
        self.assertLess(len(json.dumps(versions)),2048)
        self.assertEqual(self.request('/v1/voices/uploaded-voice').json()['code'],'resource_limit')
        self.assertEqual(self.request('/v1/voices/uploaded-voice/versions','bob-secret').status_code,404)

    def test_auth_invalid_transport_and_media(self):
        data=wav_bytes()
        for token in ('alice-secret','bad'):
            with patch.object(self.dispatcher,'call',side_effect=AssertionError('unauthorized dispatch')):
                self.error(asyncio.run(self.post([data],token=token)),'forbidden')
        for chunks,headers,code in [
            ([data],{'Content-Type':'application/json'},'invalid_request'),
            ([data],{'Content-Length':'3'},'invalid_request'),
            ([data],{'Content-Length':str(len(data)+1)},'invalid_request'),
            ([data],{'Content-Length':'25165825'},'resource_limit'),
            ([b'x'*65537],{},'resource_limit'),
            ([],{},'invalid_request')]:
            with patch.object(self.dispatcher,'call',side_effect=AssertionError('partial dispatch')):
                self.error(asyncio.run(self.post(chunks,headers=headers)),code)
        self.error(asyncio.run(self.post([b'not RIFF'])),'invalid_media')
        self.assert_empty_media()
        self.credentials.revoke('writer')
        self.error(asyncio.run(self.post([data])),'forbidden')

    def test_actual_chunk_counter_and_maximum_boundary(self):
        # No Content-Length; one byte over cap refuses before any owner dispatch.
        async def run():
            with patch.object(self.dispatcher,'call',side_effect=AssertionError('oversize dispatch')):
                self.error(await self.post([b'x'*65536]*384+[b'x']),'resource_limit')
        asyncio.run(run())
        import struct
        data=wav_bytes()
        padding=24*1024*1024-len(data)-8
        maximum=data+b'JUNK'+struct.pack('<I',padding)+b'\0'*padding
        maximum=maximum[:4]+struct.pack('<I',len(maximum)-8)+maximum[8:]
        response=asyncio.run(self.post([maximum[i:i+65536] for i in range(0,len(maximum),65536)]))
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['sha256'],hashlib.sha256(maximum).hexdigest())

    def test_deadline_disconnect_and_concurrent_admission(self):
        async def run():
            app=create_voice_app(self.dispatcher,credentials=self.credentials,upload_deadline=.03)
            entered=asyncio.Event(); release=asyncio.Event()
            async def slow():
                entered.set()
                await release.wait()
                yield wav_bytes()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                with patch.object(self.dispatcher,'call',side_effect=AssertionError('incomplete dispatch')):
                    task=asyncio.create_task(client.post('/v1/assets',content=slow(),headers={'Authorization':'Bearer writer','Content-Type':'audio/wav'}))
                    await entered.wait()
                    self.error(await self.post([wav_bytes()],app=app),'resource_limit')
                    self.error(await task,'invalid_request')
            # Raw ASGI receive supplies a partial body then disconnect, no HTTP listener.
            messages=iter([{'type':'http.request','body':b'RIFF','more_body':True},{'type':'http.disconnect'}])
            sent=[]
            async def receive():
                try: return next(messages)
                except StopIteration: await asyncio.Event().wait()
            async def send(message): sent.append(message)
            scope={'type':'http','asgi':{'version':'3.0'},'method':'POST','path':'/v1/assets','raw_path':b'/v1/assets','query_string':b'', 'headers':[(b'authorization',b'Bearer writer'),(b'content-type',b'audio/wav')],'scheme':'http','server':('test',80),'client':('test',1),'http_version':'1.1','root_path':''}
            with patch.object(self.dispatcher,'call',side_effect=AssertionError('disconnect dispatch')):
                await app(scope,receive,send)
            self.assertEqual(next(m['status'] for m in sent if m['type']=='http.response.start'),422)
        asyncio.run(run())
        self.assert_empty_media()

    def test_owner_failures_sanitized_no_receipt_and_fail_closed(self):
        for failure in (queue.Full,TimeoutError,OSError):
            app=create_voice_app(self.dispatcher,credentials=self.credentials)
            with patch.object(self.dispatcher,'call',side_effect=failure('private-path')):
                self.error(asyncio.run(self.post([wav_bytes()],app=app)),'not_ready')
            self.error(asyncio.run(self.post([wav_bytes()],app=app)),'resource_limit')

    def test_asset_read_error_contract_membership(self):
        asset=asyncio.run(self.post([wav_bytes()])).json()
        for operation, path in (
            ('asset_get','/v1/assets/'+asset['id']),
            ('asset_content','/v1/assets/'+asset['id']+'/content?start=0&end_exclusive=1'),
            ('voice_versions_list','/v1/voices/'+self.version['voice_id']+'/versions')):
            declared=self.request('/skeleton/operations/'+operation).json()['operation']['errors']
            for failure in (queue.Full,TimeoutError,OSError):
                with patch.object(self.dispatcher,'call',side_effect=failure('private-path')):
                    response=self.request(path)
                body=validate('error',response.json())
                self.assertEqual(body['code'],'not_ready')
                self.assertIn(body['code'],declared)
                self.assertNotIn('private-path',response.text)
        (self.root/'assets'/asset['id']).write_bytes(b'tampered')
        self.assertEqual(self.request('/v1/assets/'+asset['id']).json()['code'],'missing_media')
        self.assertEqual(self.request('/v1/assets/'+asset['id'],'bob-secret').status_code,404)
