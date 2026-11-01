"""Installed format 2 publication through the real dispatcher and detached reader."""
import copy
import hashlib
import io
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import patch
import wave

from contracts import validate as oracle
from service import host_install, install_policy, persistence, recovery
from service.core import Core
from service.dispatcher import Dispatcher
from service.worker_transport import TransportObservation
from tests.service.test_shared_model_installation import SharedInstallationTests
from tests.service.test_split_init import _semantic_inventory_clock
from worker.lifecycle import Observation, State
from worker.runtime_journal import journal_bytes


class InstalledQwenRegistrationCheckpoint(__import__('unittest').TestCase):
    def test_real_qwen_registration_on_installed_root(self):
        fixture = SharedInstallationTests(methodName='runTest')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        session = fixture.install('alice', shared=False)
        fixture.command(0, 'up')
        gate = session.gate
        self.assertEqual(gate['registration']['scope'], session.scope)
        self.assertEqual(gate['registration']['evidence'], 'daemon-volumes-inspection-v1')
        self.assertEqual(fixture.startup(session)['registration_id'], gate['registration_id'])

        from service import entrypoint as entry, installed_qwen_daemon as qd
        from service.installed_pocket_daemon import launch
        root = session.roots['control'] / 'core'
        local = gate['observation']['local']['control']
        incarnation = {key: local[key] for key in ('start_ticks', 'pid_namespace', 'cgroup')}
        waiting = SimpleNamespace(wait=lambda: copy.deepcopy(gate), read=lambda: copy.deepcopy(gate))
        runtime = SimpleNamespace(family=qd.FAMILY, call=lambda operation, body: session.startup()['hello'])
        activation = entry.InstalledActivation(waiting, session.config,
            launch_policy=launch(qd.FAMILY), family=qd.FAMILY)
        with patch.object(entry, '_control_identity'), patch.object(entry, 'local_incarnation', return_value=incarnation):
            activation.prepare(runtime, root)
            self.assertEqual(activation.checked['registration_id'], gate['registration_id'])
            with patch.multiple(os, getuid=lambda: 10001, geteuid=lambda: 10001,
                                getgid=lambda: 10001, getegid=lambda: 10001, getgroups=lambda: []):
                with patch.object(activation, 'precheck', wraps=activation.precheck) as precheck, \
                     patch.object(activation, 'postcheck', wraps=activation.postcheck) as postcheck:
                    dispatcher = Dispatcher(root, staging=session.roots['staging'], runtime=runtime,
                        profiles=session.config['profiles'], budget=session.config['budget'],
                        production_contexts=SimpleNamespace(), installed_activation=activation)
                precheck.assert_called_once_with(dispatcher)
                postcheck.assert_called_once_with(dispatcher)
                try:
                    self.assertTrue(dispatcher.installed_checked)
                    self.assertTrue(dispatcher.enrolled)
                    self.assertEqual(dispatcher.call('status')['phase'], 'idle')
                finally:
                    dispatcher.close()


    def test_same_root_activated_long_publication(self):
        from service import entrypoint as entry, installed_qwen_daemon as qd
        from service.installed_pocket_daemon import launch
        from service.execution_context import cpu_clone_context
        from service.ingest import ingest_local
        from worker.runtime_protocol import generated_source
        old_umask=os.umask(0o077);self.addCleanup(os.umask,old_umask)
        fixture=SharedInstallationTests(methodName='runTest')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=fixture.fixture.config
        template=config['contexts']['alice']
        template['execution']['descriptor'].update(max_frames=700000,max_output_bytes=1400044)
        config['profiles'][0]['limits'].update(upload_bytes=1400044,disk_bytes=4000000,decoded_ticks=1400000)
        template['plan']['resolved']['profile']=oracle.profile_snapshot(config['profiles'][0])
        session=fixture.install('alice',shared=False)
        root=session.roots['control']/'core';stage_root=session.roots['staging']
        ids=patch.multiple(os,getuid=lambda:10001,geteuid=lambda:10001,getgid=lambda:10001,getegid=lambda:10001,getgroups=lambda:[])
        with ids:
            core=Core(root)
            asset=ingest_local(core,'alice',fixture.fixture.b.raw)
            template=session.config['contexts']['alice']
            version=copy.deepcopy(template['plan']['voice_versions'][0])
            version.update(provenance=asset['provenance'],reference_asset_ids=[asset['id']],reference_asset_sha256={asset['id']:asset['sha256']})
            version['configuration']['reference_asset_id']=asset['id']
            core.voices.register('alice','fixture-voice',version,expected_revision=0)
            core.close()
        fixture.command(0,'up')
        gate=session.gate;hello=session.startup()['hello']
        local=gate['observation']['local']['control']
        incarnation={k:local[k] for k in ('start_ticks','pid_namespace','cgroup')}
        waiting=SimpleNamespace(wait=lambda:copy.deepcopy(gate),read=lambda:copy.deepcopy(gate))
        outer=self
        class Runtime:
            family=qd.FAMILY
            authority=None;receipt=None;acked=False
            def call(inner,op,body):
                if op=='hello':return copy.deepcopy(hello)
                outer.assertEqual(op,'ack');inner.acked=True;return dict(acked=True)
            def prepare(inner,a,fixture):
                inner.authority=a;inner.identity=hello['lane']+'/'+hello['boot']+'/'+hashlib.sha256(a.job['id'].encode()).hexdigest()[:32]
                return inner.identity
            def start(inner,identity):return dict(state='alive')
            def observe(inner,identity,timeout):return Observation(identity,State.STOPPED)
            def collect(inner,a,identity):
                stage=stage_root/a.job['attempts'][-1]['staging_id'];stage.mkdir(mode=0o770);stage.chmod(0o770)
                buf=io.BytesIO()
                with wave.open(buf,'wb') as wav:
                    wav.setparams((1,2,24000,0,'NONE','none'));wav.writeframes(b'\0\0'*600000)
                raw=buf.getvalue();(stage/'output.wav').write_bytes(raw)
                digest=hashlib.sha256(raw).hexdigest()
                output=dict(id='asset-installed-long',sha256=digest,frames=600000,sample_rate=24000,channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='generated',source_sha256=digest,source_id=generated_source(a.context),missing_fields=[]))
                message=a.message();message.update(kind='result',stage='completed',result_asset_ids=[output['id']])
                manifest={k:message[k] for k in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256')}
                manifest.update(operation='speech_generate',artifact_id='artifact-installed-long',outputs=[output],candidates=[])
                inner.receipt=dict(runtime_id=identity,binding=a.binding,reaped=True,engine_uid=20000,reap_method='pid1-waitpid-echild',reaped_children=0,timely=True,original_deadline=a.job['deadline'])
                return inner.receipt,TransportObservation('proposal','injected-backend',message=message,manifest=manifest)
        runtime=Runtime()
        failures=[]
        original_publish=Core._publish
        def publish(core,*args,**kw):
            try:return original_publish(core,*args,**kw)
            except Exception as exc:
                failures.append(repr(exc));raise
        self.enterContext(patch.object(Core,'_publish',publish))
        original=fixture.f.translate
        def translate(info,path=None):
            if path is not None and stage_root in path.parents:
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=20000 if path.name=='output.wav' else 10001,st_gid=10001))
            return original(info,path)
        fixture.f.translate=translate
        def resolve(core,principal,operation,request):
            return cpu_clone_context(template,principal,request,core.asset(asset['id']),version=core.voices.version(principal,version['id']),reference_bytes=len(fixture.fixture.b.raw))
        contexts=SimpleNamespace(resolve_owned=resolve)
        activation=entry.InstalledActivation(waiting,session.config,launch_policy=launch(qd.FAMILY),family=qd.FAMILY)
        with patch.object(entry,'_control_identity'),patch.object(entry,'local_incarnation',return_value=incarnation),patch.multiple(os,getuid=lambda:10001,geteuid=lambda:10001,getgid=lambda:10001,getegid=lambda:10001,getgroups=lambda:[]):
            with patch.object(activation,'precheck',wraps=activation.precheck) as pre,patch.object(activation,'postcheck',wraps=activation.postcheck) as post:
                d=Dispatcher(root,staging=stage_root,runtime=runtime,profiles=session.config['profiles'],budget=session.config['budget'],production_contexts=contexts,installed_activation=activation)
            try:
                pre.assert_called_once();post.assert_called_once();self.assertTrue(d.enrolled and d.installed_checked)
                end=time.monotonic()+3
                while not d.call('status')['reconciled'] and time.monotonic()<end:time.sleep(.02)
                self.assertTrue(d.call('status')['persistent_activation'],d.call('status'))
                request=copy.deepcopy(template['plan']['request']);request['voice_version_id']=version['id']
                receipt=d.call('production_generate','alice',request,key='long')
                end=time.monotonic()+5
                while time.monotonic()<end:
                    job=d.call('job',receipt['job_id'])
                    if job['state'] in ('succeeded','failed') and d.call('status')['phase']=='idle':break
                    time.sleep(.02)
                self.assertEqual(job['state'],'succeeded',(job,failures,d.call('status')))
                artifact=d.call('production_artifact','alice',dict(artifact_id=job['artifact_ids'][0]))
                self.assertGreater((root/'assets'/artifact['asset_ids'][0]).stat().st_size,1048576)
            finally:d.close()
            self.assertTrue(runtime.acked)
            core=Core(root)
            try:
                self.assertEqual(core.artifact(artifact['id']),artifact)
                a=runtime.authority
                journal=dict(version=1,lane=hello['lane'],records=[dict(id=runtime.identity,allocation=dict(snapshot=json.loads(a._snapshot),binding=a.binding,fixture='qwen'),phase='terminal',acked=True,uid=20000,boot=hello['boot'],receipt=runtime.receipt,proposal=None)])
                (session.roots['receipts']/'journal.json').write_bytes(journal_bytes(journal))
                with _semantic_inventory_clock():state=recovery.snapshot({r:session.roots[r] for r in install_policy.ROLES},core.db,session.scope,policy=install_policy.QWEN_SPLIT)
                self.assertFalse(state['inventory']['reasons'],state['evidence_projection']['inventory'])
                recovery._installed_detached_source(json.loads(json.dumps(state)),journal)
                self.assertTrue(recovery.installed_clean_history(json.loads(json.dumps(state)),journal))
                corrupted=json.loads(json.dumps(state))
                corrupted['evidence_projection']['data']['p1_format'][0][0]=1
                with self.assertRaises(ValueError):recovery._installed_detached_source(corrupted,journal)
            finally:core.close()


    def test_installed_precheck_is_required(self):
        from service.entrypoint import InstalledActivation
        with patch.object(InstalledActivation,'precheck',side_effect=RuntimeError('precheck-required')) as gate:
            with self.assertRaisesRegex(RuntimeError,'precheck-required'):
                self.test_same_root_activated_long_publication()
            self.assertEqual(gate.call_count,1)

    def test_installed_postcheck_is_required(self):
        from service.entrypoint import InstalledActivation
        with patch.object(InstalledActivation,'postcheck',side_effect=RuntimeError('postcheck-required')) as gate:
            with self.assertRaisesRegex(RuntimeError,'postcheck-required'):
                self.test_same_root_activated_long_publication()
            self.assertEqual(gate.call_count,1)

    def test_existing_v1_reopen_does_not_upgrade(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'legacy-core'
            core=Core(root,_publication_creation=1)
            try:self.assertEqual(core.publication_format,1)
            finally:core.close()
            reopened=Core(root)
            try:self.assertEqual(reopened.publication_format,1)
            finally:reopened.close()
