"""Seeded CPU public source contracts; inert model/handles/TLS, never hardware proof."""
from contextlib import contextmanager
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import unittest
from unittest.mock import Mock,patch

from service.native_cpu_activation import NativeCPUActivation
from service.native_cpu_install import Prepared
from service.native_cpu_context import NativeCPUContexts
from service.native_cpu_runtime import NativeCPURuntime
from service.dispatcher import Dispatcher
from service.worker_transport import Authority
from tests.worker import test_native_cpu_supervisor as supervisor_tests
from tests.service.test_native_runtime import BytesChannel
from worker import native_cpu_protocol as p
from worker.native_cpu_qualification import Session,VERSION,SECONDS,MAX_JOBS
from worker.transport import canonical,Refused


class CPUPublicSourceTests(unittest.TestCase):
    def setUp(self):
        fixture=supervisor_tests.CPUSupervisorTests('test_memory_limit_stops_without_an_oom_claim');fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.f=fixture.f;self.s=fixture.s;self.d=None
        self.root=self.f.bundle.parent.parent/'cpu-public-source';self.root.mkdir(mode=0o700)
        self.control=self.root/'control';self.control.mkdir(mode=0o700)
        self.spool=self.root/'spool';self.spool.mkdir(mode=0o700)
        self.stage=self.root/'staging';self.stage.mkdir(mode=0o770);self.stage.chmod(0o770)
        self.cert=b'inert-source-TLS-peer';self.pin=hashlib.sha256(self.cert).hexdigest()
        self.s.pin=self.pin;self.s.clock=time.monotonic;self.s.utc=lambda:datetime.now(timezone.utc)
        self.s.availability.clock=time.monotonic
        self.f.profiles[0].update(support='unqualified',readiness='not_ready',verification='synthetic',qualification_receipt='')
        self.native=copy.deepcopy(self.f.context['execution']['native'])
        value=dict(version=VERSION,session_id='1'*32,boot_id='00000000-0000-0000-0000-000000000001',
            started_monotonic=time.monotonic(),seconds=SECONDS,max_jobs=MAX_JOBS)
        self.session=Session(value,configuration_sha256='b'*64,runtime_manifest_sha256='d'*64,deployment_id='a'*32,
            profile=self.f.profiles[0],manifest_sha256=self.f.spec.manifest_sha256,principal='local',boot=lambda:value['boot_id'])
        self.s.qualification=self.session
        snapshot=self.f.snapshot_for(self.f.context);snapshot[0]['principal_id']='local'
        a=Authority(*snapshot)
        self.s.registrations={a.binding:dict(snapshot=snapshot,execution=self.f.spec,bundles={self.f.spec.bundle_id:self.f.bundle},inventory=self.f.inventory)}
        self.s._preset_templates=copy.deepcopy(self.s.registrations)
        self.runtime=NativeCPURuntime(host='127.0.0.1',port=4443,tls=None,peer_sha256=self.pin,native=self.native,
            spool=self.spool,staging=self.stage,local_import_uid=os.getuid(),connector=self.connect)
        self.contexts=NativeCPUContexts({'local':self.f.context},{self.f.spec.bundle_id:self.f.bundle})
        self.contexts.bind_availability(self.runtime);self.contexts.qualification=self.session
        self.prepared=Prepared.__new__(Prepared)
        self.prepared.qualification=self.session;self.prepared.config=dict(principal='local')
        self.prepared.control=self.control;self.prepared.staging=self.stage;self.prepared.runtime=self.runtime
        self.prepared.contexts=self.contexts;self.prepared.profiles=self.f.profiles
        self.prepared.supervisor=self.s;self.prepared.inventory_wire=p.inventory_wire(self.f.inventory)
        self.prepared.revalidate=Mock(return_value='b'*64)
        self.prepared.budget=dict(slots=1,resident_models=1,ram_bytes=2*1024**3,vram_bytes=0,pids=1,queue=2,disk_bytes=128*1024**2,free_bytes=16*1024**2)
        self.finish=True;self.requests=[];self.worker_errors=[];self.addCleanup(self.close)

    @contextmanager
    def connect(self,cutoff):
        owner=self
        class TLS(BytesChannel):
            def version(self):return 'TLSv1.3'
            def getpeercert(self,binary_form=False):return owner.cert
            def sendall(self,data):
                q=json.loads(data[4:]);owner.requests.append(q)
                if q['op'] in ('observe','collect') and owner.s.active:
                    active=owner.s.active
                    if active.get('sampler') is not None:active['sampler'].stamp=time.monotonic()
                    if owner.finish:
                        active['receiver'].join(1);owner.fixture.h.code=0
                    elif active['cancel'] and active['term'] is not None:owner.fixture.h.code=-15
                    owner.s.tick()
                envelope={k:v for k,v in q.items() if k!='body'}
                try:envelope.update(ok=True,value=owner.s.dispatch(q))
                except Refused as exc:owner.worker_errors.append((q['op'],str(exc)));envelope.update(ok=False,error='native_refused')
                raw=canonical(envelope);self.raw=bytearray(struct.pack('!I',len(raw))+raw)
        try:yield TLS()
        except Exception as exc:
            self.worker_errors.append(('transport',type(exc).__name__,str(exc)))
            raise

    def exchange(self,connection,config,cutoff,on_loaded):
        # Existing engine verifies the tiny immutable fixture and encodes PCM.
        from lib.tts_backends.contract import AudioResult
        from worker.engines.kokoro import execute
        from worker.native_child import proposal
        from lib.spoken_text import speech_text
        import numpy as np
        backend=Mock();backend.synthesise.return_value=AudioResult(np.asarray([.1,-.1],dtype='float32'),24000)
        on_loaded(min(config['cutoff'],time.monotonic()+30))
        spec=p.native_execution(config['snapshot'][1])
        result=execute(spec.model,speech_text(config['snapshot'][1]['plan']),config['bundles'],backend_factory=lambda *a,**kw:backend)
        return proposal(config['snapshot'],result,spec,protocol=p),result['data']

    def open(self):
        self.activation=NativeCPUActivation(self.prepared)
        self.d=Dispatcher(self.control,staging=self.stage,installed_activation=self.activation)
        self.wait(lambda:self.d.call('status')['reconciled'])
        self.assertFalse(self.d.call('status')['integrity_fault'])
        self.assertFalse(self.d.call('status')['quarantined'])
        return self.d

    def close(self):
        if self.d is not None:self.d.close();self.d=None;self.activation.release()

    def wait(self,fn):
        end=time.monotonic()+4
        while time.monotonic()<end:
            value=fn()
            if value:return value
            time.sleep(.01)
        self.fail('bounded source wait expired: '+repr(self.d.call('status'))+' ops='+repr([q['op'] for q in self.requests][-12:])+' phases='+repr([(r['phase'],r['receipt']) for r in self.s.journal.value['records']])+' worker_errors='+repr(self.worker_errors)+' owner='+repr(getattr(self.s,'owner_error',None))+' receiver='+repr(None if self.s.active is None else self.s.active.get('receiver_error')))

    def request(self,version):
        return dict(voice_version_id=version['id'],text='Hello. The test is complete.',identity=version['identity'],fallback='forbid',variants=1,reuse='never')

    def reopen_transport(self,*,new_owner=False):
        if new_owner:
            from worker.native_cpu_supervisor import NativeCPUSupervisor
            old=self.s;old.journal.lock.close()
            self.native['boot']='9'*32
            self.f.context['execution']['native']=copy.deepcopy(self.native)
            snapshot=self.f.snapshot_for(self.f.context);snapshot[0]['principal_id']='local'
            authority=Authority(*snapshot)
            self.s=NativeCPUSupervisor(old.journal.root,deployment_id='a'*32,configuration_sha256='b'*64,
                peer_pin=self.pin,runtime_manifest_sha256='d'*64,python=old.python,child_entry=old.entry,
                registrations={authority.binding:dict(snapshot=snapshot,execution=self.f.spec,bundles={self.f.spec.bundle_id:self.f.bundle},inventory=self.f.inventory)},
                qualification=self.session,_boot=self.native['boot'],popen=old.popen,pair=old.pair,
                signal_group=old.signal_group,identity=old.identity,clock=time.monotonic,utc=lambda:datetime.now(timezone.utc),
                free_bytes=old.free_bytes,free_ram=old.free_ram,availability_provider=self.f.observed,sampler_factory=old.sampler_factory)
            self.addCleanup(self.s.journal.lock.close);self.prepared.supervisor=self.s
        self.runtime=NativeCPURuntime(host='127.0.0.1',port=4443,tls=None,peer_sha256=self.pin,native=self.native,
            spool=self.spool,staging=self.stage,local_import_uid=os.getuid(),connector=self.connect)
        self.prepared.runtime=self.runtime
        self.contexts=NativeCPUContexts({'local':self.f.context},{self.f.spec.bundle_id:self.f.bundle})
        self.contexts.bind_availability(self.runtime);self.contexts.qualification=self.session
        self.prepared.contexts=self.contexts

    def test_checked_owner_catalogue_alias_and_generate_source(self):
        from service.native_cpu_configuration import ALIAS
        with patch.object(self.s,'exchange',side_effect=self.exchange):
            d=self.open()
            voice=d.call('voice_resolve','local',ALIAS)
            version=d.call('voice_version','local',voice['version_ids'][-1])
            registered=d.call('production_preset_register','local',dict(preset='af_heart',voice_id='source.extra',expected_revision=0),key='source-register')
            self.assertEqual(registered['voice']['id'],'source.extra')
            submitted=d.call('production_generate','local',self.request(version),key='source-generate')
            job=self.wait(lambda:(j if (j:=d.call('job',submitted['job_id']))['state']=='succeeded' else None))
            self.wait(lambda:d.call('status')['phase']=='idle')
            artifact=d.call('production_artifact','local',dict(artifact_id=job['artifact_ids'][0]))
            asset=d.call('asset',artifact['asset_ids'][0]);self.assertGreater(asset['frames'],0)
            status=d.call('status');self.assertTrue(status['persistent_activation'])
            self.assertEqual(status['reservations'][0]['demand']['pids'],1)
            self.assertEqual(d.call('configured_profiles')[0]['support'],'unqualified')

    def test_ordinary_installation_metadata_does_not_admit_speech(self):
        from service.native_cpu_configuration import ALIAS
        self.prepared.qualification=None;self.s.qualification=None;self.contexts.qualification=None
        d=self.open();voice=d.call('voice_resolve','local',ALIAS)
        version=d.call('voice_version','local',voice['version_ids'][-1])
        with self.assertRaisesRegex(ValueError,'not_ready'):d.call('production_generate','local',self.request(version),key='must-refuse')
        self.assertFalse(d.call('status')['resource_activation'])
        self.assertEqual(self.fixture.calls,[]);self.assertEqual(self.s.journal.value['records'],[])

    def test_completed_public_state_reopens_without_reprovisioning(self):
        from service.native_cpu_configuration import ALIAS
        with patch.object(self.s,'exchange',side_effect=self.exchange):
            d=self.open()
            voice=d.call('voice_resolve','local',ALIAS)
            version=d.call('voice_version','local',voice['version_ids'][-1])
            submitted=d.call('production_generate','local',self.request(version),key='retained-source')
            job=self.wait(lambda:(j if (j:=d.call('job',submitted['job_id']))['state']=='succeeded' else None))
            self.wait(lambda:d.call('status')['phase']=='idle')
            records=copy.deepcopy(self.s.journal.value['records'])
            self.close()
            # A process restart constructs a fresh transport and accounts retained
            # import markers; reusing the old in-memory transport would not test it.
            self.reopen_transport(new_owner=True)
            d=self.open()
            self.assertEqual(d.call('job',job['id']),job)
            self.assertEqual(d.call('voice_resolve','local',ALIAS),voice)
            self.assertEqual(self.s.journal.value['records'],records)
            self.assertEqual(len(self.fixture.calls),1)

    def test_public_http_cli_mcp_share_one_generation_and_download(self):
        import asyncio,contextlib,io
        import httpx
        from clients.http import HTTP
        from clients.cli import main
        from clients import mcp
        from service.api import create_voice_app
        from service.auth import Credential,StaticCredentials
        from service.native_cpu_configuration import ALIAS
        with patch.object(self.s,'exchange',side_effect=self.exchange):
            d=self.open()
            app=create_voice_app(d,credentials=StaticCredentials([
                Credential('source-writer','local',frozenset({'read','production-write'})),
                Credential('source-other','other',frozenset({'read','production-write'}))]),native_installed=True)
            class ASGIClient:
                def __init__(inner,**kw):pass
                def __enter__(inner):return inner
                def __exit__(inner,*a):pass
                def request(inner,method,path,**kw):
                    async def run():
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                            return await client.request(method,path,**kw)
                    return asyncio.run(run())
                def get(inner,path,**kw):return inner.request('GET',path,**kw)
                def post(inner,path,**kw):return inner.request('POST',path,**kw)
            def cli(args):
                output=io.StringIO()
                with contextlib.redirect_stdout(output):code=main(args)
                self.assertEqual(code,0,output.getvalue())
                return json.loads(output.getvalue())
            voice=d.call('voice_resolve','local',ALIAS)
            payload=self.request(d.call('voice_version','local',voice['version_ids'][-1]))
            with patch('clients.http.httpx.Client',ASGIClient),patch.dict(os.environ,{'SPEECH_CREDENTIAL':'source-writer'}):
                transport=HTTP(credential='source-writer')
                status=transport.get('/v1/status')
                self.assertFalse(status['ready_for_speech']);self.assertTrue(status['installed'])
                result=transport.call('speech_generate',payload,key='shared-public')
                self.assertEqual(cli(['speech','generate','--request',json.dumps(payload),'--idempotency-key','shared-public']),result)
                response=asyncio.run(mcp.call_tool('speech_speech_generate',dict(payload,idempotency_key='shared-public')))
                self.assertFalse(response.isError);self.assertEqual(response.structuredContent,result)
                job=self.wait(lambda:(j if (j:=d.call('job',result['job_id']))['state']=='succeeded' else None))
                self.wait(lambda:d.call('status')['phase']=='idle')
                artifact=transport.call('artifact_get',dict(artifact_id=job['artifact_ids'][0]))
                asset=artifact['asset_ids'][0];target=self.root/'public.wav'
                cli(['asset','content',asset,'--output',str(target)])
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),d.call('asset',asset)['sha256'])
                response=asyncio.run(mcp.call_tool('speech_artifact_get',dict(artifact_id=artifact['id'])))
                self.assertFalse(response.isError);self.assertEqual(response.structuredContent,artifact)
                with self.assertRaises(Exception):HTTP(credential='source-other').call('job_get',dict(job_id=job['id']))
            self.assertEqual(len(self.fixture.calls),1)

    def test_running_cancel_reaps_without_publishing_audio(self):
        from service.native_cpu_configuration import ALIAS
        self.finish=False
        with patch.object(self.s,'exchange',side_effect=self.exchange):
            d=self.open();voice=d.call('voice_resolve','local',ALIAS)
            version=d.call('voice_version','local',voice['version_ids'][-1])
            submitted=d.call('production_generate','local',self.request(version),key='cancel-source')
            self.wait(lambda:self.s.active is not None)
            d.call('cancel',submitted['job_id'])
            job=self.wait(lambda:(j if (j:=d.call('job',submitted['job_id']))['state']=='cancelled' else None))
            self.wait(lambda:d.call('status')['phase']=='idle')
            self.assertEqual(job['artifact_ids'],[])
            receipt=self.s.journal.value['records'][0]['receipt']
            self.assertFalse(receipt['timely']);self.assertTrue(receipt['reaped'])
            self.assertEqual(receipt['termination']['reason'],'cancelled')
            self.assertEqual(self.fixture.h.waited,1)

    def test_never_started_cancel_reopens_without_inventing_worker_receipt(self):
        from service.native_cpu_configuration import ALIAS
        d=self.open();voice=d.call('voice_resolve','local',ALIAS)
        version=d.call('voice_version','local',voice['version_ids'][-1])
        with patch.object(Dispatcher,'_tick'):
            submitted=d.call('production_generate','local',self.request(version),key='queued-cancel-source')
            job=d.call('cancel',submitted['job_id'])
            self.assertEqual((job['state'],job['stage'],job['attempts']),('cancelled','never_started',[]))
            self.close()
        self.reopen_transport(new_owner=True);d=self.open()
        self.assertEqual(d.call('job',job['id']),job)
        self.assertEqual(self.s.journal.value['records'],[])
        self.assertEqual(self.fixture.calls,[])


if __name__=='__main__':unittest.main()
