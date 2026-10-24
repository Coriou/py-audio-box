"""In-process ASGI + real owner/Core; synthetic receipt evidence only."""
import asyncio
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import wave
import httpx
from contracts import validate as oracle
from service.api import create_voice_app
from service.auth import Credential, StaticCredentials
from service.dispatcher import Dispatcher
from service.domain import SyntheticContexts
from service.worker_transport import TransportObservation
from worker.lifecycle import Observation, State
from worker.synthetic import candidates


def audio():
    buffer = io.BytesIO()
    with wave.open(buffer,'wb') as wav:
        wav.setparams((1,2,48000,0,'NONE','not compressed')); wav.writeframes(b'\0'*9600)
    return buffer.getvalue()


def configuration():
    fixture = json.loads(Path('contracts/fixtures/v1/transcript-voice_prepare.json').read_text())
    prepare = fixture['context']; generate = copy.deepcopy(prepare)
    plan = generate['plan']; r = plan['resolved']
    plan['operation'] = 'speech_generate'
    plan['request'] = dict(voice_version_id='unused',text=' Exact\n text ',identity=r['identity']['requested'],fallback='forbid',variants=1,reuse='never')
    r.update(kind='speech',settings=r['reuse_settings'],reuse_settings=None,reuse_configuration=None)
    r['settings_sha256'] = oracle.canonical_hash(r['settings'])
    profile = fixture['profiles'][0]
    profile['compatibility'].append(dict(operation='speech_generate',mode='qwen_clone',identity=copy.deepcopy(r['identity']['effective'])))
    profile.update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='E1 MOCK synthetic only, no inference qualification')
    for context in (prepare,generate): context['plan']['resolved']['profile'] = oracle.profile_snapshot(profile)
    return prepare,generate,[profile]


class SyntheticRuntime:
    """No process/listener; fabricated reap receipt does not prove containment."""
    def __init__(self, staging):
        self.staging=staging; self.authorities=[]; self.hold=False; self.cancelled=False
    def call(self,op,body): return {'quarantined':False} if op=='hello' else {'acked':True}
    def prepare(self,authority,fixture):
        authority.check(authority.message(),'execute')
        self.authorities.append(authority); self.cancelled=False
        return 'synthetic/'+str(len(self.authorities))
    def start(self,identity): return {'state':'alive'}
    def observe(self,identity,timeout):
        return Observation(identity,State.ALIVE if self.hold and not self.cancelled else State.STOPPED)
    def terminate(self,identity,timeout): self.cancelled=True
    def collect(self,a,identity):
        receipt=dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',timely=not self.cancelled,original_deadline=a.job['deadline'])
        if self.cancelled: return receipt,None
        message=a.message(); plan=message['plan']; data=audio(); digest=hashlib.sha256(data).hexdigest()
        stage=self.staging/message['staging_id']; stage.mkdir(mode=0o770); stage.chmod(0o770)
        (stage/'output.wav').write_bytes(data)
        output=dict(id=plan['resolved']['reuse_configuration']['reference_asset_id'] if plan['operation']=='voice_prepare' else 'output-'+message['job_id'],sha256=digest,frames=4800,sample_rate=48000,channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='synthetic',source_sha256=digest,source_id='phase-1a-silence',missing_fields=[]))
        manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
        manifest.update(operation=plan['operation'],artifact_id='artifact-'+message['job_id'],outputs=[output],candidates=candidates(message,output))
        message.update(kind='result',stage='completed',result_asset_ids=[output['id']])
        a.check(message,'result')
        return receipt,TransportObservation('proposal','E1 synthetic',message=message,manifest=manifest)


class ProductionAPIFixture:
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='pab-r186-'); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'core'; self.stage=Path(self.tmp.name)/'stage'; self.stage.mkdir()
        self.prepare,self.generate,self.profiles=configuration()
        self.contexts=SyntheticContexts({('alice','voice_prepare'):self.prepare,('alice','speech_generate'):self.generate})
        self.runtime=SyntheticRuntime(self.stage)
        self.dispatcher=Dispatcher(self.root,staging=self.stage,runtime=self.runtime,profiles=self.profiles,production_contexts=self.contexts)
        self.addCleanup(lambda:self.dispatcher.close())
        self.credentials=StaticCredentials([Credential('writer','alice',frozenset({'read','production-write'})),Credential('reader','alice',frozenset({'read'})),Credential('bob','bob',frozenset({'read','production-write'})),Credential('write-only','alice',frozenset({'production-write'}))])
        self.app=create_voice_app(self.dispatcher,credentials=self.credentials)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])

    def wait(self,fn):
        end=time.monotonic()+5
        while time.monotonic()<end:
            value=fn()
            if value:return value
            time.sleep(.01)
        self.fail(str(self.dispatcher.call('status')))

    def request(self,method,path,*,payload=None,key=None,token='writer',content=None):
        async def run():
            headers={'Authorization':'Bearer '+token}
            if key is not None:headers['Idempotency-Key']=key
            if content is not None:headers['Content-Type']='audio/wav'
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://test') as client:
                return await client.request(method,path,json=payload,content=content,headers=headers)
        return asyncio.run(run())

    def submit(self,op,payload,key):
        response=self.request('POST','/v1/operations/'+op,payload=payload,key=key)
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def prepared(self):
        asset=self.request('POST','/v1/assets',content=audio()).json()
        payload=dict(self.prepare['plan']['request'],asset_id=asset['id'])
        receipt=self.submit('voice_prepare',payload,'prepare')
        job_id=receipt['job_id']
        self.wait(lambda:self.dispatcher.call('job',job_id)['state']=='succeeded')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        result=self.request('GET',f'/v1/jobs/{job_id}/voice-results').json()
        return asset,payload,receipt,result


class ProductionAPITests(ProductionAPIFixture,unittest.TestCase):
    def test_reference_prepare_select_save_generate_and_ownership(self):
        asset,payload,receipt,result=self.prepared(); job_id=receipt['job_id']
        before=self.dispatcher.call('status')['reservations']
        self.assertEqual(self.submit('voice_prepare',payload,'prepare'),receipt)
        self.assertEqual(self.dispatcher.call('status')['reservations'],before)
        self.assertEqual(len(before),1)
        authority=self.runtime.authorities[0]
        self.assertEqual(authority.allowed_assets,(asset['id'],))
        self.assertEqual(authority.message()['input_asset_ids'],[asset['id']])
        audition=result['candidates'][0]['audition_asset_id']
        self.assertEqual(self.request('GET','/v1/assets/'+audition).status_code,200)
        self.assertEqual(self.request('GET','/v1/assets/'+audition,token='bob').status_code,404)
        self.assertEqual(self.request('GET',f'/v1/assets/{audition}/content?start=0&end_exclusive=3072').status_code,200)
        save=dict(job_id=job_id,candidate_id=result['candidates'][0]['id'],voice_id='voice',expected_revision=0)
        saved=self.submit('voice_save',save,'save')
        self.assertEqual(self.submit('voice_save',save,'save'),saved)
        generation=dict(self.generate['plan']['request'],voice_version_id=saved['version']['id'])
        speech=self.submit('speech_generate',generation,'generate')
        self.wait(lambda:self.dispatcher.call('job',speech['job_id'])['state']=='succeeded')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        self.assertEqual(self.runtime.authorities[-1].allowed_assets,(result['candidates'][0]['audition_asset_id'],))
        self.assertEqual(self.runtime.authorities[-1].context['plan']['request']['text'],generation['text'])
        job=self.request('GET','/v1/jobs/'+speech['job_id']).json()
        artifact=self.request('GET','/v1/artifacts/'+job['artifact_ids'][0]).json()
        self.assertEqual(artifact['job_id'],job['id'])
        self.assertEqual(self.request('GET','/v1/assets/'+artifact['asset_ids'][0]).status_code,200)
        for path in (f'/v1/jobs/{job_id}',f'/v1/jobs/{job_id}/voice-results','/v1/artifacts/'+artifact['id']):
            self.assertEqual(self.request('GET',path,token='bob').status_code,404)
        self.assertEqual(self.request('POST','/v1/operations/voice_save',payload=dict(save,expected_revision=1),key='save2').status_code,409)
        self.assertEqual(self.request('POST','/v1/operations/voice_prepare',payload=dict(payload,reference_text='changed'),key='prepare').status_code,409)

    def test_default_not_ready_refusal_and_immutable_operator_context(self):
        # Mutation of constructor inputs cannot retarget the immutable provider.
        self.prepare['plan']['resolved']['identity']['requested']['model']='changed'
        original=self.contexts.resolve('alice','voice_prepare',configuration()[0]['plan']['request'])
        self.assertNotEqual(original['plan']['resolved']['identity']['requested']['model'],'changed')
        for provider,profiles in ((None,self.profiles),(self.contexts,[dict(self.profiles[0],readiness='not_ready')])):
            self.dispatcher.close()
            self.dispatcher=Dispatcher(self.root,staging=self.stage,runtime=self.runtime,profiles=profiles,production_contexts=provider)
            self.app=create_voice_app(self.dispatcher,credentials=self.credentials)
            self.wait(lambda:self.dispatcher.call('status')['reconciled'])
            asset=self.request('POST','/v1/assets',content=audio()).json()
            payload=dict(configuration()[0]['plan']['request'],asset_id=asset['id'])
            response=self.request('POST','/v1/operations/voice_prepare',payload=payload,key='refused')
            self.assertEqual(response.json()['code'],'not_ready')
            self.assertEqual(self.dispatcher.call('jobs','alice'),[])
            self.assertEqual(self.dispatcher.call('status')['reservations'],[])

    def test_all_generator_request_and_exchange_consumers(self):
        from service.catalogue import OPERATIONS
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(oracle.SCHEMA)
        fixtures=Path('contracts/fixtures/v1')
        requests=json.loads((fixtures/'operation-requests.json').read_text())
        for operation,payload in requests.items():
            op=OPERATIONS[operation]
            oracle.validate(op['request'],oracle.fixture_request(op['request'],payload))
        for name in ('voice_prepare','voice_design','speech_generate','project_render'):
            fixture=json.loads((fixtures/('transcript-'+name+'.json')).read_text())
            for exchange in fixture['exchanges']:
                op=OPERATIONS[exchange['operation']]
                oracle.validate(op['request'],exchange['request'])
                oracle.validate(op['response'],exchange['response'])

    def test_scopes_validation_configuration_and_cancel(self):
        for token in ('reader','write-only','invalid'):
            self.assertEqual(self.request('POST','/v1/operations/voice_prepare',payload=self.prepare['plan']['request'],key='k',token=token).status_code,403)
        self.assertEqual(self.request('POST','/v1/operations/voice_prepare',payload=self.prepare['plan']['request']).status_code,422)
        self.assertEqual(self.request('POST','/v1/operations/voice_prepare',payload=dict(self.prepare['plan']['request'],context={}),key='k').status_code,422)
        identity=dict(self.prepare['plan']['request']['identity'],model='unsupported')
        response=self.request('POST','/v1/operations/voice_prepare',payload=dict(self.prepare['plan']['request'],identity=identity),key='bad')
        self.assertEqual(response.json()['code'],'unsupported_control')
        self.runtime.hold=True
        asset=self.request('POST','/v1/assets',content=audio()).json()
        payload=dict(self.prepare['plan']['request'],asset_id=asset['id'])
        first=self.submit('voice_prepare',payload,'first'); self.wait(lambda:self.runtime.authorities)
        second=self.submit('voice_prepare',payload,'second')
        cancelled=self.submit('job_cancel',dict(job_id=second['job_id']),'cancel')
        self.assertEqual(cancelled['state'],'cancelled')
        self.assertEqual(self.submit('job_cancel',dict(job_id=second['job_id']),'cancel')['state'],'cancelled')
        self.assertEqual(self.request('POST','/v1/operations/job_cancel',payload=dict(job_id=first['job_id']),key='cancel').status_code,409)
        self.submit('job_cancel',dict(job_id=first['job_id']),'cancel-first')
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        caps=self.request('GET','/v1/capabilities').json()
        self.assertEqual(caps['profiles'],[]); self.assertNotIn('voice_design',caps['operations'])
        for operation in caps['operations']:
            self.assertEqual(self.request('GET','/skeleton/operations/'+operation).status_code,200)

if __name__=='__main__':unittest.main()
