"""Inert ASGI/owner control tests; no child, listener or model."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_production_api import ProductionAPIFixture, audio
from service.job_control import cursor_encode, wait_job
from service.domain import production_call
from service.core import Core
from service.catalogue import Failure


class PublicJobControlTests(ProductionAPIFixture,unittest.TestCase):
    def held(self,key='prepare'):
        self.runtime.hold=True
        asset=self.request('POST','/v1/assets',content=audio()).json()
        payload=dict(self.prepare['plan']['request'],asset_id=asset['id'])
        receipt=self.submit('voice_prepare',payload,key)
        self.wait(lambda:self.runtime.authorities)
        return receipt['job_id'],payload

    def read(self,operation,payload,principal='alice'):
        return production_call(self.dispatcher,principal,operation,payload)

    def test_pages_cursors_scope_deadline_and_disconnect(self):
        job,payload=self.held()
        second=self.submit('voice_prepare',payload,'second')['job_id']
        page=self.read('jobs_list',dict(cursor='',limit=1))
        self.assertEqual([j['id'] for j in page['items']],[job])
        third=self.submit('voice_prepare',payload,'third')['job_id']
        self.assertEqual(self.read('jobs_list',dict(cursor=page['next_cursor'],limit=1))['items'][0]['id'],second)
        events=self.read('job_events',dict(job_id=job,cursor='',limit=20))
        for cursor in ('x','0','0:0',cursor_encode(99999,'alice','jobs',''),cursor_encode(1,'bob','jobs','')):
            with self.assertRaises(Failure): self.read('jobs_list',dict(cursor=cursor,limit=1))
        with self.assertRaises(Failure):self.read('job_events',dict(job_id=second,cursor=events['next_cursor'],limit=20))
        with self.assertRaises(Failure):self.read('job_events',dict(job_id=job,cursor=cursor_encode(99999,'alice','events',job),limit=20))
        self.assertEqual(self.read('jobs_list',dict(cursor='',limit=20),'bob')['items'],[])
        with self.assertRaises(Failure):self.read('job_events',dict(job_id=job,cursor='',limit=20),'bob')
        async def exercise():
            async def connected():return False
            payload=dict(job_id=job,cursor=events['next_cursor'],wait_seconds=0)
            result=await wait_job(self.dispatcher,'alice',payload,connected)
            self.assertTrue(result['deadline_reached']);self.assertEqual(result['next_cursor'],events['next_cursor'])
            async def disconnected():return True
            with self.assertRaises(Failure):await wait_job(self.dispatcher,'alice',dict(payload,wait_seconds=5),disconnected)
            # A pending wait releases the writer for cancellation and confirmed reap.
            task=asyncio.create_task(wait_job(self.dispatcher,'alice',dict(payload,wait_seconds=5),connected))
            await asyncio.sleep(.05)
            await asyncio.to_thread(production_call,self.dispatcher,'alice','job_cancel',dict(job_id=job),'cancel')
            result=await task
            self.assertTrue(result['events'])
        asyncio.run(exercise())
        self.wait(lambda:self.dispatcher.call('job',job)['state']=='cancelled')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle' or len(self.runtime.authorities)>1)
        self.runtime.hold=False
        self.wait(lambda:self.dispatcher.call('job',third)['state']=='succeeded')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        terminal=self.dispatcher.call('job',job)
        self.assertTrue(all(a['reaped'] for a in terminal['attempts']))
        final=self.read('job_events',dict(job_id=job,cursor='',limit=20))
        # Close owner and reopen SQLite read-only through Core, preserving durable rows.
        self.dispatcher.close()
        with patch.object(self.dispatcher,'close'):
            core=Core(self.root,historical_local=True)
            try:
                self.assertEqual(core.job(job),terminal)
                self.assertEqual([r['state'] for r in core.events(job)],[r['state'] for r in final['events']])
            finally:core.close()
        # Fixture cleanup owns only its temporary tree; avoid a second owner close.
        self.dispatcher.close=lambda:None

    def test_retry_exact_linkage_duplicates_authority_and_native_refusal(self):
        job,payload=self.held()
        with self.assertRaises(Failure):production_call(self.dispatcher,'alice','job_retry',dict(job_id=job,input_sha256=self.dispatcher.call('job',job)['input_sha256']),'retry')
        production_call(self.dispatcher,'alice','job_cancel',dict(job_id=job),'cancel')
        self.wait(lambda:self.dispatcher.call('job',job)['state']=='cancelled')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        self.runtime.hold=False
        retry=production_call(self.dispatcher,'alice','job_retry',dict(job_id=job,input_sha256=self.dispatcher.call('job',job)['input_sha256']),'retry')
        self.assertEqual(production_call(self.dispatcher,'alice','job_retry',dict(job_id=job,input_sha256=self.dispatcher.call('job',job)['input_sha256']),'retry'),retry)
        self.assertEqual(retry['input_sha256'],self.dispatcher.call('job',job)['input_sha256'])
        self.wait(lambda:self.dispatcher.call('job',retry['job_id'])['state']=='succeeded')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        retried=self.dispatcher.call('job',retry['job_id'])
        self.assertEqual(retried['retry_of'],job)
        with self.assertRaises(Failure):production_call(self.dispatcher,'bob','job_retry',dict(job_id=job,input_sha256=self.dispatcher.call('job',job)['input_sha256']),'foreign')
        with self.assertRaises(Failure):production_call(self.dispatcher,'alice','job_retry',dict(job_id=retry['job_id'],input_sha256=retry['input_sha256']),'retry')
        with self.assertRaises(Failure):production_call(self.dispatcher,'alice','job_retry',dict(job_id=job,input_sha256='0'*64),'changed')
        before=self.read('jobs_list',dict(cursor='',limit=20))
        with patch.object(self.dispatcher.admission,'demand',side_effect=ValueError('resource_runtime_unsupported')):
            with self.assertRaises(Failure):production_call(self.dispatcher,'alice','job_retry',dict(job_id=job,input_sha256=retry['input_sha256']),'admission')
        self.assertEqual(before,self.read('jobs_list',dict(cursor='',limit=20)))

        self.runtime.native={}
        try:
            response=self.request('GET','/skeleton/operations/job_retry')
            self.assertEqual(response.json()['availability'],'unavailable_lifecycle')
            self.assertNotIn('job_retry',self.request('GET','/v1/capabilities').json()['operations'])
            with self.assertRaises(Failure):production_call(self.dispatcher,'alice','job_retry',dict(job_id=job,input_sha256=self.dispatcher.call('job',job)['input_sha256']),'native')
        finally:del self.runtime.native
        self.assertEqual(before,self.read('jobs_list',dict(cursor='',limit=20)))

    def test_finite_contracts_and_generator_filename_byte_determinism(self):
        import tempfile
        from contracts import generate
        from contracts import validate as oracle
        from service.catalogue import OPERATIONS
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(oracle.SCHEMA)
        fixtures=Path('contracts/fixtures/v1')
        requests=json.loads((fixtures/'operation-requests.json').read_text())
        self.assertEqual(set(requests),set(OPERATIONS))
        for name,op in OPERATIONS.items():
            oracle.validate(op['request'],oracle.fixture_request(op['request'],requests[name]))
        expected={p.name:p.read_bytes() for p in fixtures.iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'contracts'
            (root/'v1').mkdir(parents=True)
            for name in ('schema.json','operations.json'):
                (root/'v1'/name).write_bytes((Path('contracts/v1')/name).read_bytes())
            generated=root/'fixtures/v1'
            generated.mkdir(parents=True)
            for name,data in expected.items(): (generated/name).write_bytes(data)
            with patch.object(generate,'ROOT',generated):generate.generate()
            self.assertEqual(expected,{p.name:p.read_bytes() for p in generated.iterdir() if p.is_file()})

    def test_http_query_contract(self):
        for route in ('/v1/jobs?cursor=&limit=0','/v1/jobs?cursor=&limit=1&limit=2','/v1/jobs?cursor=&limit=1&extra=x','/v1/jobs/path/events?job_id=query&cursor=&limit=1'):
            self.assertEqual(self.request('GET',route).json()['code'],'invalid_request')

if __name__=='__main__':unittest.main()
