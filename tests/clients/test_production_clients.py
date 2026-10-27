"""HTTP/CLI/MCP parity against real ASGI/Core/owner, no listener."""
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
from test_production_api import ProductionAPIFixture, audio
from clients.http import HTTP
from clients.cli import main
from clients import mcp
from service.catalogue import PRODUCTION_OPERATIONS, OPERATIONS


class ProductionClientTests(ProductionAPIFixture,unittest.TestCase):
    def test_prepare_audition_is_bounded_named_and_closed(self):
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError
        tool=next(t for t in mcp.tools() if t.name=='speech_voice_prepare')
        schema=tool.inputSchema
        self.assertEqual(schema['properties']['audition_text'],dict(type='string',minLength=1,maxLength=10000))
        self.assertFalse(schema['additionalProperties'])
        payload=dict(self.prepare['plan']['request'],audition_text='  Exact\n audition! ',idempotency_key='key')
        Draft202012Validator(schema).validate(payload)
        for change in ({'audition_text':''},{'audition_text':'x'*10001},{'execution':{}},{'path':'/tmp/caller'}):
            with self.assertRaises(ValidationError):Draft202012Validator(schema).validate(dict(payload,**change))

    def test_preset_tool_is_named_closed_and_requires_key(self):
        tool=next(t for t in mcp.tools() if t.name=='speech_voice_preset_register')
        self.assertEqual(set(tool.inputSchema['properties']),{'preset','voice_id','expected_revision','idempotency_key'})
        self.assertEqual(set(tool.inputSchema['required']),set(tool.inputSchema['properties']))
        self.assertFalse(tool.inputSchema['additionalProperties'])

    def test_all_shipped_operations_and_reference_journey(self):
        outer=self
        class ASGIClient:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def request(self,method,path,**kwargs):
                content=kwargs.get('content')
                if content is not None and not isinstance(content,bytes):kwargs['content']=b''.join(content)
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer.app),base_url='http://test') as client:
                        return await client.request(method,path,**kwargs)
                return asyncio.run(run())
            def get(self,path,**kwargs):return self.request('GET',path,**kwargs)
            def post(self,path,**kwargs):return self.request('POST',path,**kwargs)
        def cli(args):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):code=main(args)
            self.assertEqual(code,0,output.getvalue())
            return json.loads(output.getvalue())
        def tool(operation,payload,key=None):
            arguments=dict(payload)
            if key:arguments['idempotency_key']=key
            result=asyncio.run(mcp.call_tool(OPERATIONS[operation]['mcp'],arguments))
            self.assertFalse(result.isError,result.structuredContent)
            return result.structuredContent
        with patch('clients.http.httpx.Client',ASGIClient),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'writer','SPEECH_UPLOAD_ROOT':self.tmp.name}):
            transport=HTTP(credential='writer')
            path=Path(self.tmp.name).resolve()/'reference.wav';path.write_bytes(audio())
            asset=transport.upload_file(path)
            payload=dict(self.prepare['plan']['request'],asset_id=asset['id'])
            receipt=cli(['voice','prepare','--request',json.dumps(payload),'--idempotency-key','prepare'])
            self.assertEqual(tool('voice_prepare',payload,'prepare'),receipt)
            self.assertEqual(transport.call('voice_prepare',payload,key='prepare'),receipt)
            job_id=receipt['job_id'];self.wait(lambda:self.dispatcher.call('job',job_id)['state']=='succeeded')
            self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
            result=cli(['voice','results',job_id]);self.assertEqual(tool('voice_results',dict(job_id=job_id)),result)
            save=dict(job_id=job_id,candidate_id=result['candidates'][0]['id'],voice_id='voice',expected_revision=0)
            saved=cli(['voice','save','--request',json.dumps(save),'--idempotency-key','save'])
            self.assertEqual(tool('voice_save',save,'save'),saved)
            generation=dict(self.generate['plan']['request'],voice_version_id=saved['version']['id'])
            speech=cli(['speech','generate','--request',json.dumps(generation),'--idempotency-key','generate'])
            self.assertEqual(tool('speech_generate',generation,'generate'),speech)
            self.wait(lambda:self.dispatcher.call('job',speech['job_id'])['state']=='succeeded')
            self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
            job=cli(['job','get',speech['job_id']]);self.assertEqual(tool('job_get',dict(job_id=job['id'])),job)
            artifact=cli(['artifact','get',job['artifact_ids'][0]])
            self.assertEqual(tool('artifact_get',dict(artifact_id=artifact['id'])),artifact)
            listing=cli(['jobs','list','--limit','1'])
            self.assertEqual(tool('jobs_list',dict(cursor='',limit=1)),listing)
            self.assertEqual(transport.call('jobs_list',dict(cursor='',limit=1)),listing)
            events=cli(['job','events',job_id])
            self.assertEqual(tool('job_events',dict(job_id=job_id,cursor='',limit=20)),events)
            waited=cli(['job','wait',job_id,'--cursor',events['next_cursor'],'--wait-seconds','0'])
            self.assertEqual(tool('job_wait',dict(job_id=job_id,cursor=events['next_cursor'],wait_seconds=0)),waited)
            # Profile detail parity is checked even when discovery is empty.
            profile_id=self.profiles[0]['id']
            with patch('service.production_selection.discovery',return_value=self.profiles):
                profile=cli(['profile','get',profile_id])
                self.assertEqual(tool('profile_get',dict(profile_id=profile_id)),profile)
                self.assertEqual(transport.get('/skeleton/profiles/'+profile_id),profile)
            version=transport.call('voice_version_get',dict(voice_version_id=saved['version']['id']))
            aliases=dict(voice_id='voice',aliases=['speaker'],expected_revision=1)
            changed=cli(['voice','aliases','--request',json.dumps(aliases),'--idempotency-key','aliases'])
            self.assertEqual(tool('voice_aliases',aliases,'aliases'),changed)
            from service.catalogue import Failure
            for bad,key,code in ((dict(aliases,aliases=['different']),'aliases','idempotency_conflict'),
                                 (dict(aliases,aliases=['fresh']),'stale','revision_conflict'),
                                 (dict(voice_id='voice',aliases=['voice'],expected_revision=2),'self-alias','invalid_request')):
                with self.assertRaises(Failure) as error: transport.call('voice_aliases',bad,key=key)
                self.assertEqual(error.exception.body['code'],code)
            other=transport.call('voice_save',dict(save,voice_id='other'),key='save-other')
            with self.assertRaises(Failure) as error:
                transport.call('voice_aliases',dict(voice_id='voice',aliases=['other'],expected_revision=2),key='conflict')
            self.assertEqual(error.exception.body['code'],'alias_conflict')
            for token in ('bob','reader'):
                with self.assertRaises(Failure):HTTP(credential=token).call('voice_archive',dict(voice_id='voice',expected_revision=2),key='foreign')
            archive=dict(voice_id='voice',expected_revision=changed['revision'])
            archived=cli(['voice','archive','--request',json.dumps(archive),'--idempotency-key','archive'])
            self.assertEqual(tool('voice_archive',archive,'archive'),archived)
            self.assertTrue(archived['archived'])
            self.assertEqual(transport.call('voice_version_get',dict(voice_version_id=version['id'])),version)
            # Immutable archived versions remain valid generation references.
            historical=transport.call('speech_generate',generation,key='historical')
            self.wait(lambda:self.dispatcher.call('job',historical['job_id'])['state']=='succeeded')
            self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
            cancelled=cli(['job','cancel',job_id,'--idempotency-key','cancel'])
            self.assertEqual(tool('job_cancel',dict(job_id=job_id),'cancel'),cancelled)
            self.runtime.hold=True
            held=transport.call('voice_prepare',payload,key='held')
            self.wait(lambda:self.dispatcher.call('job',held['job_id'])['state']=='running')
            cli(['job','cancel',held['job_id'],'--idempotency-key','held-cancel'])
            self.wait(lambda:self.dispatcher.call('job',held['job_id'])['state']=='cancelled')
            self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
            self.runtime.hold=False
            retried=cli(['job','retry',held['job_id'],'--input-sha256',held['input_sha256'],'--idempotency-key','retry'])
            retry_payload=dict(job_id=held['job_id'],input_sha256=held['input_sha256'])
            self.assertEqual(tool('job_retry',retry_payload,'retry'),retried)
            self.assertEqual(transport.call('job_retry',retry_payload,key='retry'),retried)
            self.wait(lambda:self.dispatcher.call('job',retried['job_id'])['state']=='succeeded')
            self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
            # Reopen the same durable owner and re-read through all adapters.
            from service.dispatcher import Dispatcher
            from service.api import create_voice_app
            self.dispatcher.close()
            self.dispatcher=Dispatcher(self.root,staging=self.stage,runtime=self.runtime,profiles=self.profiles,production_contexts=self.contexts)
            self.app=create_voice_app(self.dispatcher,credentials=self.credentials)
            self.wait(lambda:self.dispatcher.call('status')['reconciled'])
            self.assertEqual(cli(['artifact','get',artifact['id']]),artifact)
            self.assertEqual(tool('voice_get',dict(voice_id='voice')),archived)
            self.assertEqual(transport.call('voice_version_get',dict(voice_version_id=version['id'])),version)
            self.assertEqual(tool('job_events',dict(job_id=job_id,cursor='',limit=20)),events)
            self.assertEqual(tool('job_retry',retry_payload,'retry'),retried)
            advertised=set(transport.call('discover',{})['operations'])
            tools={t.name:t for t in mcp.tools()}
            for operation in advertised-{'asset_upload'}:
                self.assertIn(OPERATIONS[operation]['mcp'],tools)
            for operation in PRODUCTION_OPERATIONS:
                if operation=='audio_export':
                    self.assertNotIn(operation,advertised)  # No converter registered in this speech fixture.
                    continue
                self.assertIn(operation,advertised)
                self.assertIn('idempotency_key',tools[OPERATIONS[operation]['mcp']].inputSchema['required']) if OPERATIONS[operation]['idempotency'] else None
            bad=asyncio.run(mcp.call_tool('speech_voice_prepare',payload))
            self.assertTrue(bad.isError);self.assertEqual(bad.structuredContent['code'],'invalid_request')
            self.assertLess(len(json.dumps(result)),12288)

if __name__=='__main__':unittest.main()


class AudioExportClientTests(unittest.TestCase):
    def test_http_cli_mcp_export_same_job_and_verified_download(self):
        import test_audio_export as export_fixture
        fixture=export_fixture.AudioExportTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        class ASGIClient:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def request(self,method,path,**kwargs):
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fixture.app),base_url='http://test') as client:
                        return await client.request(method,path,**kwargs)
                return asyncio.run(run())
            def get(self,path,**kw):return self.request('GET',path,**kw)
            def post(self,path,**kw):return self.request('POST',path,**kw)
        _,_,request=fixture.upload()
        with patch('clients.http.httpx.Client',ASGIClient),patch.dict('os.environ',{'SPEECH_CREDENTIAL':'writer'}):
            transport=HTTP(credential='writer')
            receipt=transport.call('audio_export',request,key='master')
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                code=main(['audio','export','--request',json.dumps(request),'--idempotency-key','master'])
            self.assertEqual(code,0);self.assertEqual(json.loads(output.getvalue()),receipt)
            result=asyncio.run(mcp.call_tool('speech_audio_export',dict(request,idempotency_key='master')))
            self.assertFalse(result.isError);self.assertEqual(result.structuredContent,receipt)
            tool=next(t for t in mcp.tools() if t.name=='speech_audio_export')
            self.assertEqual(set(tool.inputSchema['required']),{'asset_id','rate','channels','loudness','idempotency_key'})
            self.assertFalse(tool.inputSchema['additionalProperties'])
            job=fixture.terminal(receipt)
            artifact=transport.call('artifact_get',dict(artifact_id=job['artifact_ids'][0]))
            self.assertEqual(artifact['audio_export']['settings'],request)
            asset_id=artifact['asset_ids'][0]
            target=Path(fixture.tmp.name)/'master.wav'
            with contextlib.redirect_stdout(io.StringIO()):code=main(['asset','content',asset_id,'--output',str(target)])
            self.assertEqual(code,0)
            import hashlib
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),transport.call('asset_get',dict(asset_id=asset_id))['sha256'])
            self.assertEqual(len(fixture.runtime.authorities),1)
