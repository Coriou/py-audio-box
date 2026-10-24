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
            cancelled=cli(['job','cancel',job_id,'--idempotency-key','cancel'])
            self.assertEqual(tool('job_cancel',dict(job_id=job_id),'cancel'),cancelled)
            advertised=set(transport.call('discover',{})['operations'])
            tools={t.name:t for t in mcp.tools()}
            for operation in advertised-{'asset_upload'}:
                self.assertIn(OPERATIONS[operation]['mcp'],tools)
            for operation in PRODUCTION_OPERATIONS:
                self.assertIn(operation,advertised)
                self.assertIn('idempotency_key',tools[OPERATIONS[operation]['mcp']].inputSchema['required']) if OPERATIONS[operation]['idempotency'] else None
            bad=asyncio.run(mcp.call_tool('speech_voice_prepare',payload))
            self.assertTrue(bad.isError);self.assertEqual(bad.structuredContent['code'],'invalid_request')
            self.assertLess(len(json.dumps(result)),12288)

if __name__=='__main__':unittest.main()
