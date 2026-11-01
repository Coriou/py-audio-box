"""Real owner/ASGI, injected synthetic runtime. No hardware evidence."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_production_api import ProductionAPIFixture
from lib.spoken_text import speech_text


class SpokenDispatchTests(ProductionAPIFixture,unittest.TestCase):
    def generation(self):
        _,_,receipt,result=self.prepared()
        saved=self.submit('voice_save',dict(job_id=receipt['job_id'],candidate_id=result['candidates'][0]['id'],voice_id='spoken',expected_revision=0),'save-spoken')
        request=dict(self.generate['plan']['request'],voice_version_id=saved['version']['id'],text='A-2047')
        request['text_rendering']=dict(profile='numeric-v1',version=1,locale='fr-FR',annotations=[dict(type='identifier',start=0,end=6,literal='A-2047',policy='spell-letters-digits')])
        return request

    def test_frozen_publication_duplicate_without_resolver_and_conflict(self):
        request=self.generation();receipt=self.submit('speech_generate',request,'spoken')
        job_id=receipt['job_id'];self.wait(lambda:self.dispatcher.call('job',job_id)['state']=='succeeded')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        authority=self.runtime.authorities[-1]
        self.assertEqual(speech_text(authority.context['plan']),'a, deux zéro quatre sept')
        self.assertEqual(authority.context['plan']['request'],request)
        artifact=self.dispatcher.call('job',job_id)['artifact_ids'][0]
        self.assertEqual(self.request('GET','/v1/artifacts/'+artifact).status_code,200)
        before=len(self.runtime.authorities)
        self.dispatcher.production_contexts=None
        duplicate=self.submit('speech_generate',request,'spoken')
        self.assertEqual(duplicate['input_sha256'],receipt['input_sha256'])
        self.assertEqual(duplicate['job_id'],job_id)
        self.assertEqual(len(self.runtime.authorities),before)
        changed=copy.deepcopy(request);changed['text_rendering']['annotations']=[]
        response=self.request('POST','/v1/operations/speech_generate',payload=changed,key='spoken')
        self.assertNotEqual(response.status_code,200)

    def test_expansion_refused_before_reservation_or_dispatch(self):
        request=self.generation();request['text']='W'*4096
        request['text_rendering']['annotations']=[dict(type='identifier',start=0,end=4096,literal=request['text'],policy='spell-letters-digits')]
        before=self.dispatcher.call('status')['reservations'];calls=len(self.runtime.authorities)
        response=self.request('POST','/v1/operations/speech_generate',payload=request,key='too-large')
        self.assertEqual(response.json()['code'],'invalid_request')
        self.assertEqual(self.dispatcher.call('status')['reservations'],before)
        self.assertEqual(len(self.runtime.authorities),calls)

    def test_cancel_retry_retains_input_and_context(self):
        request=self.generation();self.runtime.hold=True
        receipt=self.submit('speech_generate',request,'cancel-spoken');job_id=receipt['job_id']
        self.wait(lambda:self.dispatcher.call('job',job_id)['state']=='running')
        self.submit('job_cancel',dict(job_id=job_id),'cancel')
        self.wait(lambda:self.dispatcher.call('job',job_id)['state']=='cancelled')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        original=copy.deepcopy(self.runtime.authorities[-1].context)
        self.runtime.hold=False
        retry=self.submit('job_retry',dict(job_id=job_id,input_sha256=receipt['input_sha256']),'retry-spoken')
        self.wait(lambda:self.dispatcher.call('job',retry['job_id'])['state']=='succeeded')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        self.assertEqual(retry['input_sha256'],receipt['input_sha256'])
        self.assertEqual(self.runtime.authorities[-1].context,original)
        self.dispatcher.production_contexts=None
        replay=self.submit('job_retry',dict(job_id=job_id,input_sha256=receipt['input_sha256']),'retry-spoken')
        self.assertEqual(replay['job_id'],retry['job_id'])

    def test_publication_reopen_and_identity_tamper_refuses_without_repair(self):
        from service.core import Core,encode
        request=self.generation();receipt=self.submit('speech_generate',request,'reopen-spoken')
        self.wait(lambda:self.dispatcher.call('job',receipt['job_id'])['state']=='succeeded')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        artifact=self.dispatcher.call('job',receipt['job_id'])['artifact_ids'][0]
        self.dispatcher.close()
        # The owner is now closed; inherited cleanup resolves this handle at
        # cleanup time. Do not call the genuine production close twice.
        self.dispatcher=ExitStack()
        core=Core(self.root)
        try:
            self.assertEqual(core.artifact(artifact)['input_sha256'],receipt['input_sha256'])
            job,context=core._load(receipt['job_id'])
            context['plan']['speech_text']['spoken_text']='substitution'
            raw=encode(context)
            core.db.execute('UPDATE jobs SET context=? WHERE id=?',(raw,job['id']));core.db.commit()
            with self.assertRaises(ValueError):core.artifact(artifact)
            self.assertEqual(core.db.execute('SELECT context FROM jobs WHERE id=?',(job['id'],)).fetchone()[0],raw)
        finally:core.close()
