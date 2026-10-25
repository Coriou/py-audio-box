"""INERT installed owner journey; no real Docker/converter/UID/reap evidence."""
import copy
import hashlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from service import entrypoint as ep, installed_export as ex
from service.dispatcher import Dispatcher
from service.api import create_voice_app
from service.auth import Credential, StaticCredentials
from tests.service.test_daemon_volumes import Session
from test_audio_export import InertExportRuntime, audio
from test_production_api import ProductionAPIFixture


class ExportOwnerTests(ProductionAPIFixture, unittest.TestCase):
    def setUp(self): pass

    def setup_family(self, *, qualified=True):
        s=self.s=Session(self,family=ex.FAMILY)
        s.config['budget'].update(disk_bytes=512*1024**2,free_bytes=1024**2)
        # Explicit injected external qualification. No product default or receipt.
        for p in ([*s.config['profiles'],s.config['audio_export']['profile']] if qualified else []):
            p.update(support='qualified',installation='installed',configuration='valid',
                     readiness='ready',verification='measured',
                     qualification_receipt='INERT external fixture, NOT qualification')
        # Context templates retain their reviewed Kokoro profile snapshot.
        from contracts import validate as oracle
        for context in s.config['contexts'].values():
            context['plan']['resolved']['profile']=oracle.profile_snapshot(s.config['profiles'][0])
        exe=s.root/'inert-converter';exe.write_bytes(b'INERT never executed');exe.chmod(0o555)
        lib=s.root/'inert-runtime';lib.write_bytes(b'INERT closure');lib.chmod(0o444)
        s.config['audio_export']['converter'].update(executable=str(exe),
            executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),runtime_closure=[str(lib)])
        from worker.runtime_supervisor import export_startup, RuntimeSupervisor
        checked=export_startup(s.config,s.config['images']['worker'],owner_uid=os.getuid())
        worker=RuntimeSupervisor.__new__(RuntimeSupervisor)
        worker._export_configuration=checked;worker.failed=False;worker.active=None;worker.launches=0
        def startup():
            journal=json.loads((s.roots['receipts']/'journal.json').read_bytes())['payload']
            worker.journal=SimpleNamespace(value=journal,maintenance=None);worker.boot=s.boot
            return dict(hello=worker.operation('hello',{}),local=s.facts('worker'))
        s.startup=startup
        s.worker=worker;worker.export_reference_roots={'alice':'/references/assets'}
        for path in (exe,lib):
            st=path.stat();s.f.virtual[st.st_dev,st.st_ino]=(0,0)
        s.b['configuration_digest']=ex._digest(s.config)
        # Install fixture input must reflect the exact immutable selected config.
        for i,entry in enumerate(s.inputs):
            if entry['role']=='config':
                raw=json.dumps(s.config,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
                s.bytes[i]=raw;entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        s.run('install');s.run('init');s.run('up')
        translate=s.f.translate
        def ownership(info,path=None):
            if path is not None and s.roots['staging'] in path.parents:
                import stat
                uid=10001 if stat.S_ISDIR(info.st_mode) else 20000
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=uid,st_gid=10001))
            return translate(info,path)
        s.f.translate=ownership
        self.root=s.roots['control']/'core';self.stage=s.roots['staging']
        self.credentials=StaticCredentials([Credential('writer','alice',frozenset({'read','production-write'}))])
        self.enterContext(patch.multiple(os,getuid=lambda:10001,geteuid=lambda:10001,
            getgid=lambda:10001,getegid=lambda:10001,getgroups=lambda:[]))
        self.enterContext(patch.object(ep,'local_incarnation',side_effect=lambda:{
            k:s.facts('control')[k] for k in ('start_ticks','pid_namespace','cgroup')}))

    def open_owner(self, *, admission=True):
        s=self.s
        class Runtime(InertExportRuntime):
            acks=0
            def __init__(self,stage):
                super().__init__(stage);self.completed={}
            def call(self,op,body):
                if op=='ack':
                    self.acks+=1
                    a,receipt=self.completed[body['runtime_id']]
                    path=s.roots['receipts']/'journal.json'
                    journal=json.loads(path.read_bytes())['payload']
                    journal['records'].append(dict(id=body['runtime_id'],
                        allocation=dict(snapshot=json.loads(a._snapshot),binding=a.binding,
                            fixture='audio_export' if a.job['operation']=='audio_export' else 'kokoro'),
                        phase='terminal',acked=True,uid=20000,boot=s.boot,receipt=receipt,proposal=None))
                    from worker.runtime_journal import journal_bytes
                    path.write_bytes(journal_bytes(journal))
                return s.startup()['hello'] if op=='hello' else super().call(op,body)
            def collect(self,a,i):
                if a.job['operation']=='speech_generate':
                    from worker.runtime_protocol import generated_source
                    from service.worker_transport import TransportObservation
                    from worker.audio_export import pcm
                    m=a.message();raw=audio(frames=100)
                    metadata=pcm(raw)
                    asset=dict(id='speech-'+a.job['id'],**metadata,provenance=dict(kind='generated',
                        source_sha256=metadata['sha256'],source_id=generated_source(a.context),missing_fields=[]))
                    stage=self.stage/m['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
                    (stage/'output.wav').write_bytes(raw)
                    manifest={k:m[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
                    manifest.update(operation='speech_generate',artifact_id='artifact-'+a.job['id'],outputs=[asset],candidates=[])
                    m.update(kind='result',stage='completed',result_asset_ids=[asset['id']])
                    receipt=dict(runtime_id=i,binding=a.binding,reaped=True,engine_uid=20000,
                        reap_method='pid1-waitpid-echild',reaped_children=2,timely=True,original_deadline=a.job['deadline'])
                    self.completed[i]=(a,receipt)
                    return receipt,TransportObservation('proposal','INERT Kokoro PCM, no inference',message=m,manifest=manifest)
                receipt,proposal=super().collect(a,i)
                receipt["engine_uid"]=20000
                self.completed[i]=(a,receipt)
                return receipt,proposal
            def prepare(self,a,fixture):
                if fixture=='audio_export':
                    from worker import managed_reference
                    def acquire(asset_id):
                        return os.open(owner.root/'assets'/asset_id,os.O_RDONLY)
                    with patch.object(managed_reference,'acquire',side_effect=acquire):
                        fd=s.worker.verify_export(a.context,a.job['principal_id'])
                        os.close(fd)
                    super().prepare(a,fixture)
                else:
                    if fixture!='kokoro': raise ValueError('fixture')
                    a.check(a.message(),'execute');self.authorities.append(a)
                return s.gate['registration']['lane']+'/'+s.boot+'/'+format(len(self.authorities),'032x')
        owner=self
        self.runtime=Runtime(self.stage)
        self.runtime.admission_open=admission
        from service.execution_context import KokoroContexts
        contexts=KokoroContexts.__new__(KokoroContexts)
        contexts._contexts={k:json.dumps(v) for k,v in s.config['contexts'].items()}
        directory=s.roots['control']/('notification-'+str(len(getattr(self,'notifications',[]))))
        waiting=ep.WaitingControl(directory=directory,family=ex.FAMILY)
        self.notifications=getattr(self,'notifications',[])+[waiting]
        self.addCleanup(waiting.close)
        ep.publish(ex._encoded(s.gate),directory=directory,family=ex.FAMILY)
        activation=ep.InstalledActivation(waiting,s.config,launch_policy=ex.LAUNCH,family=ex.FAMILY)
        activation.prepare(self.runtime,self.root)
        self.dispatcher=Dispatcher(self.root,staging=self.stage,runtime=self.runtime,
            profiles=ex.admission_profiles(s.config),budget=s.config['budget'],
            production_contexts=contexts,installed_activation=activation)
        acquired=self.dispatcher
        self.addCleanup(lambda: acquired.close() if acquired.thread.is_alive() else None)
        self.app=create_voice_app(self.dispatcher,credentials=self.credentials,installed=s.config)
        self.wait(lambda:self.dispatcher.call('status')['reconciled'])
        self.assertFalse(self.dispatcher.call('status')['integrity_fault'], self.dispatcher.integrity)
        self.assertTrue(self.dispatcher.installed_checked)
        return activation

    def test_fresh_owner_public_large_export_reopen_replay(self):
        self.setup_family();self.open_owner()
        raw=audio(frames=96000)
        response=self.request('POST','/v1/assets',content=raw)
        self.assertEqual(response.status_code,200,response.text)
        asset=response.json()
        request=dict(asset_id=asset['id'],rate=48000,channels=2,loudness=None)
        response=self.request('POST','/v1/operations/audio_export',payload=request,key='installed')
        self.assertEqual(response.status_code,200,str((response.text,self.dispatcher.call('status'),self.dispatcher.audio_exports,self.dispatcher.profiles)))
        receipt=response.json()
        job=self.wait(lambda:(j if (j:=self.dispatcher.call('job',receipt['job_id']))['state'] in ('succeeded','failed','cancelled') else None))
        self.assertEqual(job['state'],'succeeded',str(job))
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        artifact=self.request('GET','/v1/artifacts/'+job['artifact_ids'][0]).json()
        output=self.request('GET','/v1/assets/'+artifact['asset_ids'][0]).json()
        content=(self.root/'assets'/output['id']).read_bytes()
        self.assertGreater(len(content),1024**2)
        self.assertEqual(hashlib.sha256(content).hexdigest(),output['sha256'])
        self.assertEqual((output['sample_rate'],output['channels'],output['media_subtype']),(48000,2,'PCM_24'))
        import base64
        reconstructed=b''
        for start in range(0,len(content),65536):
            end=min(start+65536,len(content))
            result=self.request('GET',f"/v1/assets/{output['id']}/content?start={start}&end_exclusive={end}")
            self.assertEqual(result.status_code,200,result.text)
            reconstructed+=base64.b64decode(result.json()['content_base64'])
        self.assertEqual(reconstructed,content)
        for start,end in [(41,103)]:
            result=self.request('GET',f"/v1/assets/{output['id']}/content?start={start}&end_exclusive={end}")
            self.assertEqual(result.status_code,200,result.text)
            self.assertEqual(base64.b64decode(result.json()['content_base64']),content[start:end])
        self.assertEqual((self.root/'assets'/asset['id']).read_bytes(),raw)
        self.client_parity(request,receipt,output,content)
        reservations=self.dispatcher.call('status')['reservations']
        self.dispatcher.close()
        public=json.loads((self.s.roots['receipts']/'p1-init.json').read_bytes())['public']
        state=ex.retained_snapshot(self.s.roots,public,family=ex.FAMILY)
        self.assertFalse(state['faults'])
        # No configuration/enrollment mutation: readiness loss is live admission.
        self.open_owner(admission=False)
        self.assertEqual(self.submit('audio_export',request,'installed'),receipt)
        self.assertEqual(self.runtime.authorities,[])
        self.assertEqual(self.dispatcher.call('status')['reservations'],reservations)
        self.assertEqual(self.request('POST','/v1/operations/audio_export',payload=request,key='new').json()['code'],'not_ready')

    def test_helper_status_stop_reopen(self):
        self.setup_family()
        self.assertEqual(self.s.run('status')['phase'],'ready')
        self.assertEqual(self.s.run('stop')['phase'],'stopped')
        self.assertEqual(self.s.run('status')['phase'],'stopped')
        self.assertEqual(self.s.run('up')['phase'],'ready')

    def client_parity(self,request,receipt,output,content):
        import asyncio,contextlib,io,httpx
        from pathlib import Path
        from clients.http import HTTP
        from clients.cli import main
        from clients import mcp
        owner=self
        class Client:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def request(self,method,path,**kwargs):
                async def run():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=owner.app),base_url='http://test') as client:
                        return await client.request(method,path,**kwargs)
                return asyncio.run(run())
            def get(self,path,**kwargs): return self.request('GET',path,**kwargs)
            def post(self,path,**kwargs): return self.request('POST',path,**kwargs)
        with patch('clients.http.httpx.Client',Client),patch.dict(os.environ,{'SPEECH_CREDENTIAL':'writer'}):
            self.assertEqual(HTTP(credential='writer').call('audio_export',request,key='installed'),receipt)
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(['audio','export','--request',json.dumps(request),'--idempotency-key','installed']),0)
            self.assertEqual(json.loads(stdout.getvalue()),receipt)
            result=asyncio.run(mcp.call_tool('speech_audio_export',dict(request,idempotency_key='installed')))
            self.assertFalse(result.isError);self.assertEqual(result.structuredContent,receipt)
            target=self.s.root/'download.wav'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['asset','content',output['id'],'--output',str(target)]),0)
            self.assertEqual(target.read_bytes(),content)
            self.assertEqual(len(self.runtime.authorities),1)

    def uploaded_request(self):
        response=self.request('POST','/v1/assets',content=audio(frames=96000))
        self.assertEqual(response.status_code,200,response.text)
        return dict(asset_id=response.json()['id'],rate=48000,channels=2,loudness=None)

    def terminal(self,receipt,state):
        job=self.wait(lambda:(j if (j:=self.dispatcher.call('job',receipt['job_id']))['state'] in ('succeeded','failed','cancelled') else None))
        self.assertEqual(job['state'],state,str(job))
        self.wait(lambda:self.dispatcher.call('status')['phase']=='idle')
        return job

    def test_generated_kokoro_asset_uses_same_export_resolver(self):
        self.setup_family();self.open_owner()
        config=self.s.config['contexts']['alice']
        saved=self.submit('voice_preset_register',dict(preset='af_heart',voice_id='voice-owned',expected_revision=0),'preset')
        generation=dict(config['plan']['request'],voice_version_id=saved['version']['id'])
        speech=self.terminal(self.submit('speech_generate',generation,'speech'),'succeeded')
        artifact=self.request('GET','/v1/artifacts/'+speech['artifact_ids'][0]).json()
        source=artifact['asset_ids'][0]
        raw=(self.root/'assets'/source).read_bytes()
        self.assertLess(len(raw),1024**2)
        result=self.terminal(self.submit('audio_export',dict(asset_id=source,rate=48000,channels=2,loudness=None),'speech-export'),'succeeded')
        exported=self.request('GET','/v1/artifacts/'+result['artifact_ids'][0]).json()
        self.assertEqual(exported['audio_export']['source_asset_id'],source)
        self.assertEqual((self.root/'assets'/source).read_bytes(),raw)

    def failed_export(self,fault):
        self.setup_family();self.open_owner()
        self.runtime.fault=fault
        request=self.uploaded_request()
        job=self.terminal(self.submit('audio_export',request,fault),'failed')
        self.assertEqual(job['artifact_ids'],[])
        self.assertTrue(job['attempts'][0]['reaped'])
        self.runtime.admission_open=False
        response=self.request('POST','/v1/operations/job_retry',
            payload=dict(job_id=job['id'],input_sha256=job['input_sha256']),key='retry')
        self.assertEqual(response.json()['code'],'not_ready')
        self.assertEqual(len(self.runtime.authorities),1)

    def test_timeout_receipt_no_publication_new_retry_refused(self):
        self.failed_export('failure')

    def test_bad_output_no_publication_new_retry_refused(self):
        self.failed_export('malformed')

    def test_stale_output_no_publication_new_retry_refused(self):
        self.failed_export('stale')

    def test_unknown_descendant_retains_capacity_until_cancel_reaped(self):
        self.setup_family();self.open_owner()
        self.runtime.hold=True
        receipt=self.submit('audio_export',self.uploaded_request(),'unknown')
        self.wait(lambda:self.dispatcher.call('job',receipt['job_id'])['state']=='running')
        self.runtime.unknown=True
        self.wait(lambda:self.dispatcher.call('status')['quarantined'])
        self.assertEqual(self.runtime.acks,0)
        self.assertEqual(self.dispatcher.call('status')['reservations'][0]['state'],'executing')
        self.assertEqual(self.dispatcher.call('job',receipt['job_id'])['artifact_ids'],[])
        self.runtime.unknown=False
        self.wait(lambda:not self.dispatcher.call('status')['quarantined'])
        self.submit('job_cancel',dict(job_id=receipt['job_id']),'cancel')
        self.assertEqual(self.terminal(receipt,'cancelled')['artifact_ids'],[])
        self.assertEqual(self.runtime.acks,1)

    def test_owner_seal_and_notification_changes_refuse(self):
        self.setup_family();activation=self.open_owner()
        self.dispatcher.close()
        for key in ('converter_digest','profile_digest','principals_digest','worker_image_digest','configuration_digest'):
            hello=self.s.startup()['hello'];hello['export_seal'][key]='f'*64
            with self.assertRaises(ValueError):activation.check_hello(hello)
        hello=self.s.startup()['hello'];hello['boot']='f'*32
        with self.assertRaises(ValueError):activation.check_hello(hello)
        for mutation in ('missing','format','extra'):
            gate=copy.deepcopy(self.s.gate)
            hello=gate['observation']['hello']
            if mutation=='missing': hello.pop('export_seal')
            elif mutation=='format': hello['export_seal']['publication_format']=2.0
            else: hello['extra']=True
            with self.assertRaises(ValueError): ep.decode_projection(ex._encoded(gate),family=ex.FAMILY)
        notification=self.notifications[-1]
        path=notification.directory/ep.GATE_NAME
        path.write_bytes(ex._encoded(dict(self.s.gate,configuration_digest='f'*64)))
        with self.assertRaisesRegex(ValueError,'gate_replaced'):notification.read()

    def test_explicit_family_and_caller_exports_refuse(self):
        self.setup_family()
        with self.assertRaises(ValueError):ep.InstalledActivation(None,self.s.config)
        with self.assertRaises(ValueError):ep.InstalledActivation(None,self.s.config,family=object(),launch_policy=ex.LAUNCH)
        activation=self.open_owner();self.dispatcher.close()
        runtime=self.runtime
        activation=ep.InstalledActivation(self.notifications[-1],self.s.config,family=ex.FAMILY,launch_policy=ex.LAUNCH)
        with self.assertRaisesRegex(ValueError,'caller_export_mapping'):
            Dispatcher(self.root,staging=self.stage,runtime=runtime,profiles=ex.admission_profiles(self.s.config),
                budget=self.s.config['budget'],production_contexts=object(),installed_activation=activation,audio_exports=object())

    def test_unqualified_configuration_does_not_admit_or_advertise_export(self):
        self.setup_family(qualified=False);self.open_owner()
        response=self.request('GET','/v1/capabilities')
        self.assertEqual(response.status_code,200,response.text)
        self.assertNotIn('audio_export',response.json()['operations'])
        self.assertEqual(self.request('GET','/skeleton/operations/audio_export').json()['availability'],'unavailable_converter')
        self.assertEqual(self.dispatcher.call('status')['reservations'],[])

    def test_factory_uses_same_lifespan_and_checked_configuration(self):
        import asyncio
        self.setup_family();self.open_owner()
        contexts=self.dispatcher.production_contexts
        self.dispatcher.close()
        # Only fixture mount resolution is injected. Configuration, credentials,
        # notification, activation, Core and lifespan remain production functions.
        with patch('service.execution_context.installed_mapping',return_value={}),patch('service.execution_context.installed_contexts',return_value=contexts):
            app=ep.create_installed_app(configuration_file=self.s.roots['config']/'install.json',
                credential_file=self.s.roots['credential']/'control.key',root=self.root,staging=self.stage,
                runtime=self.runtime,waiting=self.notifications[-1],family=ex.FAMILY,launch_policy=ex.LAUNCH)
        async def run():
            async with app.router.lifespan_context(app):
                owner=app.state.dispatcher
                self.assertTrue(owner.installed_checked)
                self.assertEqual(owner.audio_exports.converter,self.s.config['audio_export']['converter'])
                self.assertEqual(owner.current_profiles,ex.admission_profiles(self.s.config))
            self.assertTrue(app.state.installed_lifecycle['closed'])
        # Lifespan owns notification.close; avoid a second fixture close.
        self._cleanups=[item for item in self._cleanups if item[0] != self.notifications[-1].close]
        asyncio.run(run())
