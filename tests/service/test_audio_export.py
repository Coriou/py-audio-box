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
        if self.fault in ('late_partial','late_oversized'):
            from worker.audio_export import execution
            stage=self.stage/a.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
            (stage/'output.wav').write_bytes(b'incomplete' if self.fault=='late_partial' else b'x'*(execution(a.context).max_output_bytes+1))
            (stage/'output.wav').chmod(0o600)
            receipt['timely']=False
            return receipt,None
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
    def test_full_converter_closure_artifact_read_preserves_both_provenance_copies(self):
        # Model the complete 214-library pinned closure with inert absolute paths.
        # No converter is executed and these paths are never opened by this fixture.
        from service.catalogue import compact, PRODUCTION_RESPONSE_BYTES, ARTIFACT_RESPONSE_BYTES
        converter=copy.deepcopy(self.exports.converter)
        converter['runtime_closure']=['/usr/lib/x86_64-linux-gnu/libfixture-runtime-'+format(i,'03d')+'.so.1' for i in range(214)]
        self.dispatcher.close()
        self.exports=AudioExports(converter,self.profile,['alice'])
        self.start()
        _,_,request=self.upload()
        job=self.terminal(self.submit('audio_export',request,'full-converter-closure'))
        response=self.request('GET','/v1/artifacts/'+job['artifact_ids'][0])
        self.assertEqual(response.status_code,200,response.text)
        artifact=response.json()
        self.assertGreater(len(compact(artifact).encode('utf-8')),PRODUCTION_RESPONSE_BYTES)
        self.assertLessEqual(len(compact(artifact).encode('utf-8')),ARTIFACT_RESPONSE_BYTES)
        self.assertEqual(artifact['identity']['converter'],converter)
        self.assertEqual(artifact['audio_export']['converter'],converter)
        # Asset metadata still uses its original compact provenance and 2 KiB cap.
        asset=self.request('GET','/v1/assets/'+artifact['asset_ids'][0])
        self.assertEqual(asset.status_code,200,asset.text)
        self.assertLessEqual(len(compact(asset.json()).encode('utf-8')),2048)

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

    def tampered_public_copy(self, path, value=None, remove=False):
        import sqlite3
        from service.core import Core, encode
        from service.reconciliation import Reconciliation
        _,_,request=self.upload()
        job=self.terminal(self.submit('audio_export',request,'tamper'))
        artifact_id=job['artifact_ids'][0]
        self.assertEqual(self.request('GET','/v1/artifacts/'+artifact_id).status_code,200)
        with sqlite3.connect(self.root/'core.sqlite') as db:
            original=db.execute('SELECT metadata FROM artifacts WHERE id=?',(artifact_id,)).fetchone()[0]
            trusted={t:db.execute('SELECT * FROM '+t).fetchall() for t in ('jobs','publications','publication_work')}
            changed=json.loads(original);target=changed
            for key in path[:-1]:target=target[key]
            if remove:del target[path[-1]]
            else:target[path[-1]]=value
            db.execute('UPDATE artifacts SET metadata=? WHERE id=?',(encode(changed),artifact_id))
        response=self.request('GET','/v1/artifacts/'+artifact_id)
        self.assertEqual(response.status_code,503,response.text)
        self.dispatcher.close()
        core=Core(self.root)
        try:
            result=Reconciliation(core,self.stage).reconcile_startup()
            self.assertTrue(result['blocked'],result)
            self.assertIn('public_artifact_mismatch',result['reasons'])
            for table,rows in trusted.items():self.assertEqual(core.db.execute('SELECT * FROM '+table).fetchall(),rows)
            self.assertEqual(core.db.execute('SELECT metadata FROM artifacts WHERE id=?',(artifact_id,)).fetchone()[0],encode(changed))
        finally:core.close()
        self.start(exports=False)
        self.assertTrue(self.dispatcher.call('status')['integrity_fault'])
        self.assertEqual(self.request('GET','/v1/artifacts/'+artifact_id).status_code,503)

    def test_public_converter_tamper_refused(self):
        self.tampered_public_copy(['audio_export','converter','reported_build'],'TAMPERED BUILD CLAIM')
    def test_public_settings_tamper_refused(self):
        self.tampered_public_copy(['audio_export','settings','rate'],96000)
    def test_public_source_tamper_refused(self):
        self.tampered_public_copy(['audio_export','source_sha256'],'f'*64)
    def test_public_missing_provenance_refused(self):
        self.tampered_public_copy(['audio_export'],remove=True)
    def test_public_id_tamper_refused(self):
        self.tampered_public_copy(['id'],'foreign-artifact')
    def test_public_job_tamper_refused(self):
        self.tampered_public_copy(['job_id'],'foreign-job')
    def test_public_asset_tamper_refused(self):
        self.tampered_public_copy(['asset_ids'],['foreign-asset'])
    def test_public_numeric_representation_tamper_refused(self):
        self.tampered_public_copy(['generation'],0.0)

    def test_export_of_export_retained_history_healthy(self):
        from service.core import Core
        from service.reconciliation import Reconciliation
        _,_,r=self.upload();first=self.terminal(self.submit('audio_export',r,'first'))
        a=self.request('GET','/v1/artifacts/'+first['artifact_ids'][0]).json()
        second=self.terminal(self.submit('audio_export',dict(r,asset_id=a['asset_ids'][0],rate=24000,channels=1),'second'))
        self.dispatcher.close()
        core=Core(self.root)
        try:
            result=Reconciliation(core,self.stage).reconcile_startup()
            self.assertTrue(result['complete'],result);self.assertFalse(result['blocked'],result)
        finally:core.close()
        self.start(exports=False)
        for job in (first,second):self.assertEqual(self.request('GET','/v1/artifacts/'+job['artifact_ids'][0]).status_code,200)
    def test_failure_stale_malformed_never_publish(self):
        for fault in ('failure','stale','malformed'):
            self.runtime.fault=fault
            _,_,r=self.upload();j=self.terminal(self.submit('audio_export',r,'key-'+fault),'failed')
            self.assertEqual(j['artifact_ids'],[])
            self.assertTrue(j['attempts'][0]['reaped'])
    def test_late_partial_is_accounted_without_decoding_or_publication(self):
        self.runtime.fault='late_partial'
        _,_,r=self.upload()
        with patch('service.admission.os.open', wraps=os.open) as opened:
            j=self.terminal(self.submit('audio_export',r,'late-partial'),'failed')
        self.assertEqual(j['stage'],'deadline_or_reaped')
        self.assertEqual(j['artifact_ids'],[])
        self.assertFalse(any(Path(call.args[0]).name=='output.wav' for call in opened.call_args_list if isinstance(call.args[0],(str,Path))))
        self.assertFalse(self.dispatcher.call('status')['quarantined'])
    def test_late_oversized_partial_still_faults_resource_fence(self):
        self.runtime.fault='late_oversized'
        _,_,r=self.upload();j=self.terminal(self.submit('audio_export',r,'late-oversized'),'failed')
        self.assertEqual(j['stage'],'verification_failed')
        self.assertEqual(j['artifact_ids'],[])
        self.assertTrue(self.dispatcher.call('status')['quarantined'])
    def test_eligible_unreadable_proposal_still_refused(self):
        _,_,r=self.upload();original=os.open
        def deny_output(path,*args,**kwargs):
            if isinstance(path,(str,Path)) and Path(path).name=='output.wav':raise PermissionError('INERT unreadable eligible output')
            return original(path,*args,**kwargs)
        with patch('service.admission.os.open',side_effect=deny_output):
            j=self.terminal(self.submit('audio_export',r,'unreadable'),'failed')
        self.assertEqual(j['stage'],'verification_failed')
        self.assertEqual(j['artifact_ids'],[])
        self.assertTrue(self.dispatcher.call('status')['quarantined'])
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


class ProductionResponseBudgetTests(unittest.TestCase):
    def test_artifact_response_still_refuses_above_64_kib(self):
        from unittest.mock import Mock
        from service.catalogue import ARTIFACT_RESPONSE_BYTES, Failure, compact
        from service.domain import production_call
        artifact=json.loads(Path('contracts/fixtures/v1/artifact.json').read_bytes())
        artifact['asset_ids']=['a'+format(i,'04d')+'x'*120 for i in range(600)]
        oracle.validate('artifact',artifact)
        self.assertGreater(len(compact(artifact).encode('utf-8')),ARTIFACT_RESPONSE_BYTES)
        dispatcher=Mock();dispatcher.call.return_value=artifact
        with self.assertRaises(Failure) as caught:
            production_call(dispatcher,'alice','artifact_get',dict(artifact_id=artifact['id']))
        self.assertEqual(caught.exception.body['code'],'resource_limit')
        dispatcher.call.assert_called_once_with('production_artifact','alice',dict(artifact_id=artifact['id']),key=None)

    def test_other_production_response_keeps_12_kib_budget(self):
        from unittest.mock import Mock
        from service.catalogue import PRODUCTION_RESPONSE_BYTES, ARTIFACT_RESPONSE_BYTES, Failure, compact
        from service.domain import production_call
        job=json.loads(Path('contracts/fixtures/v1/job.json').read_bytes())
        for field in ('id','principal_id','idempotency_key'):job[field]='j'*128
        for field in ('id','owner_id','staging_id'):job['attempts'][0][field]='a'*128
        page=dict(items=[copy.deepcopy(job) for _ in range(20)],next_cursor=None)
        oracle.validate('jobs_page',page)
        size=len(compact(page).encode('utf-8'))
        self.assertGreater(size,PRODUCTION_RESPONSE_BYTES)
        self.assertLess(size,ARTIFACT_RESPONSE_BYTES)
        dispatcher=Mock();dispatcher.call.return_value=page
        with self.assertRaises(Failure) as caught:
            production_call(dispatcher,'alice','jobs_list',dict(cursor='',limit=20))
        self.assertEqual(caught.exception.body['code'],'resource_limit')
