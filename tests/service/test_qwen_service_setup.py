"""Inert setup contract: exact generated installer input and credential connection."""
import json
import asyncio
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import wave

from service import installed_qwen_daemon as qd
from service import qwen_service_setup as setup
from service import qwen_local_config as local
from service.daemon_client import manifest
from service.provisioning import bundle_manifest, manifest_bytes


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();bundles={}
        for name,suffix in (('clone','base'),('Ryan','customvoice-cpu')):
            expected=json.loads((setup.ROOT/'requirements/tts'/('qwen-0.6b-'+suffix+'.expected.json')).read_text())
            path=self.root/name;path.mkdir()
            (path/'manifest.json').write_bytes(manifest_bytes(bundle_manifest(expected)))
            bundles[name]=str(path)
        reference=self.root/'reference.wav'
        with wave.open(str(reference),'wb') as wav:
            wav.setparams((1,2,24000,0,'NONE','none'))
            wav.writeframes(b'\0\0'*4)
        self.request=dict(principal='alice',bundles=bundles,docker_context='local-test',
            images={key:'sha256:'+digit*64 for key,digit in (('control','1'),('worker','2'),('helper','3'))},
            root=str(self.root/'local'),port=8765,custody='/sys/fs/cgroup/speech-box-qwen/'+'a'*32,
            reference=dict(wav=str(reference),text='Exact template transcript.'),
            measurement=dict(receipt='operator-measurement-123',checked_at='2026-10-29T00:00:00+00:00',
                             memory_bytes=192*1024**2,
                             disk_bytes=1024**3,deadline_seconds=600),
            ram_bytes=12*1024**3,pids=64,threads=2,
            cpu=dict(control=10**9,worker=2*10**9),
            tmpfs={key:dict(bytes=65536,inodes=64) for key in ('control','worker')})

    def test_generated_request_passes_existing_validators(self):
        result=setup.generate(self.request,verifier=lambda *args: None)
        self.assertEqual(set(result),{'configuration','helper_image','inputs','cpu','tmpfs','control_port','custody'})
        self.assertEqual(len([e for e in result['inputs'] if e['role']=='models']),
                         sum(len(json.loads((setup.ROOT/'requirements/tts'/('qwen-0.6b-'+suffix+'.expected.json')).read_text())['files'])+1
                             for suffix in ('base','customvoice-cpu')))
        root=local.create(self.request['root'],port=8765,principal='alice')
        self.assertEqual(local.discover(root)['endpoint'],'http://127.0.0.1:8765')
        from service.auth import decode_credentials
        credential=root/'control.key'
        self.assertEqual(decode_credentials(credential.read_bytes(),result['configuration']['contexts']).resolve(
            'Bearer '+json.loads(credential.read_text())['credentials'][0]['token']),'alice')
        result['inputs'] += [dict(role='config',path='install.json',source=str(root/'install.json'),bytes=1,sha256='a'*64),
                             dict(role='credential',path='control.key',source=str(credential),bytes=credential.stat().st_size)]
        manifest(result['inputs'],family=qd.FAMILY,service=True)

    def test_setup_rejects_missing_measurement_before_outputs(self):
        self.request['measurement']['receipt']=''
        with self.assertRaisesRegex(ValueError,'setup_real_profile_measurement_required'):
            setup.generate(self.request,verifier=lambda *args: None)
        self.assertFalse((self.root/'local').exists())

    def test_read_only_docker_prerequisites(self):
        class Docker:
            frozen={'id':'fixture'}
            def __init__(self): self.calls=[]
            def call(self,args):
                self.calls.append(args)
                if args[:2]==['info','--format']: return 'cgroupfs'
                if args[:2]==['image','inspect']: return args[2]
                raise AssertionError(args)
            def check(self): self.calls.append(['check'])
        docker=Docker()
        setup.preflight_docker(self.request,docker=docker)
        self.assertEqual(len(docker.calls),5)
        self.assertTrue(all(call[0] in ('info','image','check') for call in docker.calls))
        class Wrong(Docker):
            def call(self,args): return 'systemd'
        with self.assertRaisesRegex(ValueError,'setup_cgroupfs_driver_required'):
            setup.preflight_docker(self.request,docker=Wrong())

    def test_flags_build_validated_request_from_measured_file_and_named_images(self):
        measurement=self.root/'measurement.json'
        measurement.write_text(json.dumps(self.request['measurement']))
        names={role:'local/'+role+':reviewed' for role in ('control','worker','helper')}
        class Docker:
            def __init__(self): self.calls=[]
            def call(self,args):
                self.calls.append(args)
                return self_ids[args[2].split('/')[1].split(':')[0]]
            def check(self): self.calls.append(['check'])
        self_ids=self.request['images']
        docker=Docker()
        a=SimpleNamespace(context=self.request['docker_context'],control_image=names['control'],
            worker_image=names['worker'],helper_image=names['helper'],measurement=str(measurement),
            threads=2,principal='alice',clone_bundle=self.request['bundles']['clone'],
            ryan_bundle=self.request['bundles']['Ryan'],root=self.request['root'],port=8765,
            custody=self.request['custody'],reference_wav=self.request['reference']['wav'],
            reference_text=self.request['reference']['text'],ram_bytes=self.request['ram_bytes'])
        request=setup.flag_request(a,docker=docker)
        self.assertEqual(request,self.request)
        with patch('service.qwen_local_config.DEFAULT_ROOT',self.root/'default'), \
             patch.dict(os.environ,{},clear=True),patch.object(setup,'_image_ids',return_value=self_ids):
            a.root=None;a.port=8000
            defaults=setup.flag_request(a)
        self.assertEqual((defaults['root'],defaults['port']),
                         (str(self.root/'default'),8000))
        a.root=self.request['root'];a.port=8765
        self.assertEqual([call[0] for call in docker.calls],['image','image','image','check'])
        self.assertEqual(setup.generate(request,verifier=lambda *args:None)['control_port'],8765)
        output=io.StringIO()
        with patch.object(setup,'_image_ids',return_value=self_ids), \
             patch.object(setup,'generate',return_value=setup.generate(request,verifier=lambda *args:None)), \
             contextlib.redirect_stdout(output):
            setup.main(['--clone-bundle',a.clone_bundle,'--ryan-bundle',a.ryan_bundle,
                '--context',a.context,'--control-image',a.control_image,'--worker-image',a.worker_image,
                '--helper-image',a.helper_image,'--custody',a.custody,'--reference-wav',a.reference_wav,
                '--reference-text',a.reference_text,'--measurement',a.measurement,
                '--ram-bytes',str(a.ram_bytes),'--threads','2','--principal','alice','--root',a.root,
                '--port','8765','--handle',str(self.root/'handle'),'--prepare-only'])
        self.assertEqual(json.loads(output.getvalue())['client_command'],
                         ['python3','-m','clients.cli','--installation',str(self.root/'local')])
        self.assertEqual(json.loads((self.root/'local'/'install-request.json').read_text())['configuration'],
                         setup.generate(request,verifier=lambda *args:None)['configuration'])

    def test_default_command_writes_exact_request_and_invokes_install_up(self):
        generated=setup.generate(self.request,verifier=lambda *args: None)
        request_file=self.root/'operator.json';request_file.write_text(json.dumps(self.request))
        handle=str(self.root/'retained-handle.json')
        with patch.object(setup,'generate',return_value=generated), \
             patch.object(setup,'preflight_docker') as preflight, \
             patch('service.daemon_client.qwen_main') as launch:
            setup.main(['--request',str(request_file),'--handle',handle])
        preflight.assert_called_once_with(self.request)
        self.assertEqual([call.args[0][0] for call in launch.call_args_list],['install','up'])
        self.assertEqual(launch.call_args_list[0].args[0],
            ['install','--context',self.request['docker_context'],'--request',
             str(self.root/'local'/'install-request.json'),'--handle',handle])
        self.assertEqual(launch.call_args_list[1].args[0],['up','--handle',handle])
        installed=json.loads((self.root/'local'/'install-request.json').read_text())
        self.assertEqual(installed['configuration'],generated['configuration'])
        self.assertEqual(local.discover(self.root/'local')['principal'],'alice')

    def test_installed_discovery_agrees_with_qwen_status(self):
        try:
            import httpx
            from service.api import create_voice_app
        except ImportError:
            self.skipTest('FastAPI is not installed in the bounded development environment')
        from service.auth import Credential, StaticCredentials
        from types import SimpleNamespace
        config=setup.generate(self.request,verifier=lambda *args: None)['configuration']
        profile=config['profiles'][0]
        operations=['voice_prepare','voice_results','voice_save','voice_preset_register','speech_generate']
        configured=dict(engine='qwen',profile=profile,operations=operations,next_action='Enroll or register Ryan')
        class Dispatcher:
            installed_checked=True
            installed_activation=SimpleNamespace(configuration=config)
            def call(self,operation,*args):
                if operation=='configured_capabilities': return configured
                if operation=='configured_profiles': return [profile]
                if operation=='configured_presets': return [dict(preset='Ryan')]
                if operation=='audio_export_available': return False
                if operation=='status': return dict(persistent_activation=True,retry_available=False)
                raise AssertionError(operation)
        app=create_voice_app(Dispatcher(),credentials=StaticCredentials([
            Credential('a'*64,'alice',frozenset(('read','production-write')))]),installed=config)
        async def get(path):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                return await client.get(path,headers={'Authorization':'Bearer '+'a'*64})
        caps=asyncio.run(get('/v1/capabilities')).json()
        status=asyncio.run(get('/v1/status')).json()
        self.assertTrue(set(operations)<=set(caps['operations']))
        self.assertEqual(caps['profiles'],status['profiles'])
        self.assertEqual(caps['engines'],['qwen'])
        self.assertTrue(status['clone_prepare'])


if __name__=='__main__': unittest.main()
