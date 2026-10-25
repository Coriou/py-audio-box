"""Tiny local SQLite/ASGI lifecycle; injected PCM is NOT conversion evidence."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch
from contracts import validate as oracle
from service.audio_export import AudioExports
from service.dispatcher import Dispatcher
from service.api import create_voice_app
from service.auth import Credential,StaticCredentials
from service.worker_transport import TransportObservation
from worker.lifecycle import Observation,State
from worker.audio_export import pcm,provenance
from test_production_api import ProductionAPIFixture


def audio(rate=24000,channels=1,width=2,frames=2400):
    out=io.BytesIO()
    with wave.open(out,'wb') as f:
        f.setparams((channels,width,rate,0,'NONE','not compressed'));f.writeframes(b'\0'*(frames*channels*width))
    return out.getvalue()


class InertExportRuntime:
    def __init__(self,stage):
        self.stage=stage;self.authorities=[];self.cancelled=False;self.hold=False;self.unknown=False;self.fault=None
    def call(self,op,body):return {'quarantined':False} if op=='hello' else {'acked':True}
    def prepare(self,a,fixture):
        if fixture!='audio_export':raise ValueError('mode')
        a.check(a.message(),'execute');self.authorities.append(a);self.cancelled=False
        return 'fixture/'+str(len(self.authorities))
    def start(self,i):return {'state':'alive'}
    def observe(self,i,timeout):return Observation(i,State.UNKNOWN if self.unknown else State.ALIVE if self.hold and not self.cancelled else State.STOPPED)
    def terminate(self,i,timeout):self.cancelled=True
    def collect(self,a,i):
        receipt=dict(runtime_id=i,binding=a.binding,reaped=True,engine_uid=os.getuid(),reap_method='pid1-waitpid-echild',reaped_children=2,timely=not self.cancelled and self.fault!='failure',original_deadline=a.job['deadline'])
        if self.cancelled or self.fault=='failure':return receipt,None
        m=a.message();p=a.context['plan'];source=p['assets'][0];r=p['request'];frames=(source['frames']*r['rate']+source['sample_rate']-1)//source['sample_rate']
        raw=audio(r['rate'],r['channels'],3,frames)
        output=dict(id='output-'+a.job['id'],**pcm(raw),provenance=provenance(p))
        stage=self.stage/m['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
        (stage/'output.wav').write_bytes(raw if self.fault!='malformed' else raw[:-1])
        manifest={k:m[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
        manifest.update(operation='audio_export',artifact_id='artifact-'+a.job['id'],outputs=[output],candidates=[],audio_export=dict(source_asset_id=source['id'],source_sha256=source['sha256'],settings=r,converter=p['resolved']['identity']['converter']))
        m.update(kind='result',stage='completed',result_asset_ids=[output['id']])
        if self.fault=='stale':manifest['generation']+=1
        return receipt,TransportObservation('proposal','INERT bytes; no FFmpeg',message=m,manifest=manifest)


class AudioExportTests(ProductionAPIFixture,unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='pab-r366-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'core';self.stage=Path(self.tmp.name)/'stage';self.stage.mkdir()
        fixture=json.loads(Path('contracts/fixtures/v1/transcript-audio_export.json').read_text())
        self.profile=fixture['profiles'][0]
        self.profile.update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='E1 INERT fixture admission; no actual converter qualification')
        self.exports=AudioExports(fixture['context']['execution']['converter'],self.profile,['alice'])
        self.credentials=StaticCredentials([Credential('writer','alice',frozenset({'read','production-write'})),Credential('reader','alice',frozenset({'read'})),Credential('bob','bob',frozenset({'read','production-write'}))])
        self.start()
        self.addCleanup(lambda:self.dispatcher.close())
    def start(self,exports=True):
        self.runtime=InertExportRuntime(self.stage)
        budget=dict(slots=1,resident_models=1,ram_bytes=192*1024**2,vram_bytes=0,pids=48,queue=32,disk_bytes=512*1024**2,free_bytes=1024**2)
        self.dispatcher=Dispatcher(self.root,staging=self.stage,runtime=self.runtime,profiles=[self.profile],budget=budget,audio_exports=self.exports if exports else None)
        self.app=create_voice_app(self.dispatcher,credentials=self.credentials)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
    def upload(self):
        raw=audio();response=self.request('POST','/v1/assets',content=raw)
        self.assertEqual(response.status_code,200,response.text)
        asset=response.json();return raw,asset,dict(asset_id=asset['id'],rate=48000,channels=2,loudness=None)
    def terminal(self,receipt,state='succeeded'):
        j=self.wait(lambda:(j if (j:=self.dispatcher.call('job',receipt['job_id']))['state'] in ('succeeded','failed','cancelled') else None))
        self.assertEqual(j['state'],state,str(j));self.wait(lambda:self.dispatcher.call('status')['phase']=='idle');return j
    def test_public_upload_export_wait_artifact_hash_download_reopen_replay(self):
        raw,asset,request=self.upload();receipt=self.submit('audio_export',request,'master')
        job=self.terminal(receipt)
        response=self.request('GET',f"/v1/jobs/{job['id']}/wait?cursor=&wait_seconds=0")
        self.assertEqual(response.status_code,200,response.text)
        artifact=self.request('GET','/v1/artifacts/'+job['artifact_ids'][0]).json()
        output=self.request('GET','/v1/assets/'+artifact['asset_ids'][0]).json()
        self.assertEqual((output['sample_rate'],output['channels'],output['media_subtype']),(48000,2,'PCM_24'))
        data=self.request('GET',f"/v1/assets/{output['id']}/content?start=0&end_exclusive={44+4800*6}").json()
        import base64
        self.assertEqual(hashlib.sha256(base64.b64decode(data['content_base64'])).hexdigest(),output['sha256'])
        self.assertEqual((self.root/'assets'/asset['id']).read_bytes(),raw)
        self.assertEqual(self.submit('audio_export',request,'master'),receipt)
        self.dispatcher.close();self.start(exports=False)
        self.assertFalse(self.dispatcher.call('status')['integrity_fault'])
        self.assertEqual(self.submit('audio_export',request,'master'),receipt)
        self.assertEqual(self.request('POST','/v1/operations/audio_export',payload=dict(request,rate=96000),key='master').json()['code'],'idempotency_conflict')
        self.assertEqual(self.request('POST','/v1/operations/audio_export',payload=request,key='new').json()['code'],'not_ready')
    def test_invalid_foreign_missing_unavailable_and_permissions(self):
        _,asset,r=self.upload()
        for payload in (dict(r,rate=22050),dict(r,channels=True),dict(r,path='/tmp/a'),{k:v for k,v in r.items() if k!='loudness'}):
            response=self.request('POST','/v1/operations/audio_export',payload=payload,key='invalid');self.assertEqual(response.json()['code'],'invalid_request')
        self.assertEqual(self.request('POST','/v1/operations/audio_export',payload=r,key='read',token='reader').status_code,403)
        self.assertEqual(self.request('POST','/v1/operations/audio_export',payload=dict(r,asset_id='absent'),key='missing').json()['code'],'not_found')
        # Allow a second principal to resolve; ownership still independently refuses.
        self.exports.principals=frozenset(['alice','bob'])
        self.assertEqual(self.request('POST','/v1/operations/audio_export',payload=r,key='foreign',token='bob').json()['code'],'not_found')
        (self.root/'assets'/asset['id']).write_bytes(b'corrupt')
        self.assertEqual(self.request('POST','/v1/operations/audio_export',payload=r,key='corrupt').json()['code'],'missing_media')
        self.assertEqual(self.dispatcher.call('jobs','alice'),[])
    def test_failure_stale_malformed_never_publish(self):
        for fault in ('failure','stale','malformed'):
            self.runtime.fault=fault
            _,_,r=self.upload();j=self.terminal(self.submit('audio_export',r,'key-'+fault),'failed')
            self.assertEqual(j['artifact_ids'],[])
            self.assertTrue(j['attempts'][0]['reaped'])
    def test_cancel_reap_and_unknown_refuses_publication_capacity_release(self):
        self.runtime.hold=True;_,_,r=self.upload();receipt=self.submit('audio_export',r,'cancel')
        self.wait(lambda:self.dispatcher.call('job',receipt['job_id'])['state']=='running')
        self.runtime.unknown=True
        self.wait(lambda:self.dispatcher.call('status')['quarantined'])
        rows=self.dispatcher.call('status')['reservations'];self.assertEqual(rows[0]['state'],'executing')
        self.assertEqual(self.dispatcher.call('job',receipt['job_id'])['artifact_ids'],[])
        self.runtime.unknown=False
        self.wait(lambda:not self.dispatcher.call('status')['quarantined'])
        response=self.request('POST','/v1/operations/job_cancel',payload=dict(job_id=receipt['job_id']),key='cancel-job')
        self.assertEqual(response.status_code,200,response.text)
        self.terminal(receipt,'cancelled')
        self.assertEqual(self.dispatcher.call('status')['reservations'][0]['state'],'retained')
    def test_metadata_discovery_requires_converter_registration(self):
        detail=self.request('GET','/skeleton/operations/audio_export').json();self.assertEqual(detail['availability'],'implemented')
        self.dispatcher.close();self.start(exports=False)
        detail=self.request('GET','/skeleton/operations/audio_export').json();self.assertEqual(detail['availability'],'unavailable_converter')
        self.assertNotIn('audio_export',self.request('GET','/v1/capabilities').json()['operations'])

    def test_source_mutation_after_admission_refuses_publication(self):
        self.runtime.hold=True
        _,asset,r=self.upload();receipt=self.submit('audio_export',r,'mutation')
        self.wait(lambda:self.dispatcher.call('job',receipt['job_id'])['state']=='running')
        (self.root/'assets'/asset['id']).write_bytes(b'changed after admission')
        self.runtime.hold=False
        job=self.terminal(receipt,'failed')
        self.assertEqual(job['artifact_ids'],[])

    def test_oversize_plan_refusal_is_atomic(self):
        self.dispatcher.close()
        self.profile['limits']['upload_bytes']=128
        self.exports.profile=copy.deepcopy(self.profile)
        self.start()
        _,_,r=self.upload()
        response=self.request('POST','/v1/operations/audio_export',payload=r,key='too-big')
        self.assertEqual(response.json()['code'],'resource_limit',response.text)
        self.assertEqual(self.dispatcher.call('jobs','alice'),[])
        self.assertEqual(self.dispatcher.call('status')['reservations'],[])

    def test_existing_retry_reuses_frozen_export_and_replays(self):
        self.runtime.fault='failure';_,_,r=self.upload()
        old=self.terminal(self.submit('audio_export',r,'original'),'failed')
        self.runtime.fault=None
        request=dict(job_id=old['id'],input_sha256=old['input_sha256'])
        retried=self.submit('job_retry',request,'retry')
        job=self.terminal(retried)
        self.assertEqual(job['retry_of'],old['id'])
        self.assertEqual(job['input_sha256'],old['input_sha256'])
        self.assertEqual(self.submit('job_retry',request,'retry'),retried)
