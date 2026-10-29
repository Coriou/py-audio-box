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
import subprocess
import sys
import zipfile

from service import installed_qwen_daemon as qd
from service import qwen_service_setup as setup
from service import daemon_volumes as volumes, recovery
from service import qwen_local_config as local
from service.daemon_client import manifest
from service.provisioning import bundle_manifest, manifest_bytes
from service.qwen_local_lane import context_allowed
from contracts import validate as oracle


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
            tmpfs=dict(control=dict(bytes=setup.CONTROL_TMPFS_BYTES,inodes=64),
                       worker=dict(bytes=65536,inodes=64)))

    def test_private_sqlite_copy_fits_generated_helper_budget(self):
        from tests.service.test_split_init import SplitInit
        fixture=SplitInit();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.run_init()  # Creates and validates a real persisted SQLite core.
        database=fixture.roots['control']/'core'/'core.sqlite'
        source_bytes=sum((database.parent/name).stat().st_size for name in
                         ('core.sqlite','core.sqlite-wal','core.sqlite-shm')
                         if (database.parent/name).exists())
        self.assertGreater(source_bytes,65536)
        self.assertLessEqual(source_bytes,recovery.MAX_BYTES)
        generated=setup.generate(self.request,verifier=lambda *args:None)
        budget=generated['tmpfs']['control']['bytes']
        self.assertGreaterEqual(budget,2*recovery.MAX_BYTES)
        self.assertEqual(generated['tmpfs']['worker'],dict(bytes=65536,inodes=64))
        from service import install_policy
        with recovery.private_database(database,policy=install_policy.SPLIT) as copied:
            self.assertEqual(copied.read_bytes(),database.read_bytes())
        config=generated['configuration']
        installation='a'*32
        b=dict(policy=volumes.POLICY,installation=installation,
               daemon=dict(endpoint='unix:///fixture/docker.sock',id='fixture-linux',
                           os='linux',architecture='amd64',security=['name=seccomp,profile=builtin'],cgroup='2'),
               volumes={r:dict(Name='pab-'+installation+'-'+r,Driver='local',Options=None,
                               Scope='local',Mountpoint='/opaque/'+r,CreatedAt='2026-10-29T00:00:00Z',
                               Labels={'speech-box.installation':installation,'speech-box.role':r})
                        for r in volumes.ROLES},
               helper_image=generated['helper_image'],configuration_digest=volumes.digest(config),
               tmpfs=generated['tmpfs'],cpu=generated['cpu'],control_port=generated['control_port'])
        for role,phase in (('helper','init'),('control',None)):
            with self.subTest(role=role):
                args=volumes.command(role,config,b,phase=phase,
                                     nonce='a'*32 if phase else None)
                expected='/tmp:rw,noexec,nosuid,nodev,size='+str(budget)+',nr_inodes=64,mode=1777'
                self.assertEqual(args[args.index('--tmpfs')+1],expected)
                self.assertEqual(args[args.index('--memory')+1],str(config['budget']['ram_bytes']))
                self.assertEqual(args[args.index('--memory-swap')+1],str(config['budget']['ram_bytes']))
                helper=role=='helper';mounts=volumes.mounts(role,b,phase)
                inspection=dict(Id='1'*64,Image=b['helper_image'] if helper else config['images'][role],
                    Config=dict(User='0:0' if helper else '10001:10001',Entrypoint=['python'],
                                Cmd=args[args.index('--entrypoint')+3:]),
                    HostConfig=dict(Privileged=False,ReadonlyRootfs=True,PidMode='',IpcMode='private',
                        CgroupnsMode='private',UsernsMode='',NetworkMode='none' if helper else 'bridge',
                        CapDrop=['ALL'],CapAdd=list(volumes.INIT_CAPS) if helper else [],
                        SecurityOpt=['no-new-privileges:true'],RestartPolicy=dict(Name='no',MaximumRetryCount=0),
                        Devices=[],DeviceRequests=None,Binds=None,Memory=config['budget']['ram_bytes'],
                        MemorySwap=config['budget']['ram_bytes'],PidsLimit=config['budget']['pids'],
                        Mounts=[dict(Type='volume',Source=m['Name'],Target=m['Destination'],
                                     ReadOnly=not m['RW'],VolumeOptions=dict(NoCopy=True)) for m in mounts],
                        NanoCpus=b['cpu']['control'],Tmpfs={'/tmp':expected.removeprefix('/tmp:')},
                        PortBindings={} if helper else {'8000/tcp':[dict(HostIp='127.0.0.1',HostPort='8765')]}),
                    State=dict(Running=False,Restarting=False,ExitCode=0,Status='created',
                               StartedAt='0001-01-01T00:00:00Z',Pid=0),RestartCount=0,Mounts=mounts)
                self.assertEqual(inspection['HostConfig']['Tmpfs']['/tmp'],expected.removeprefix('/tmp:'))
                volumes.inspect_container(inspection,role,config,b,
                                          phase=phase,nonce='a'*32 if phase else None)

    def test_control_tmpfs_below_private_copy_budget_refused(self):
        self.request['tmpfs']['control']['bytes']=65536
        with self.assertRaisesRegex(ValueError,'setup_control_tmpfs_below_private_database_budget'):
            setup.generate(self.request,verifier=lambda *args:None)

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
        self.request['measurement']['receipt']='ben-box-smoke-estimate-not-a-qualification'
        with self.assertRaisesRegex(ValueError,'setup_real_profile_measurement_required'):
            setup.generate(self.request,verifier=lambda *args: None)

    def test_read_only_docker_prerequisites(self):
        class Docker:
            frozen={'id':'fixture'}
            endpoint='unix:///var/run/docker.sock'
            def __init__(self): self.calls=[]
            def call(self,args):
                self.calls.append(args)
                if args[:2]==['info','--format'] and args[2]=='{{.Name}}':
                    return 'test-host'
                if args[:2]==['info','--format']: return 'cgroupfs'
                if args[:2]==['image','inspect']: return args[2]
                raise AssertionError(args)
            def check(self): self.calls.append(['check'])
        docker=Docker()
        with patch.object(setup,'preflight_host_cgroup'), patch.object(setup.platform,'node',return_value='test-host'):
            setup.preflight_docker(self.request,docker=docker)
        self.assertEqual(len(docker.calls),6)
        self.assertTrue(all(call[0] in ('info','image','check') for call in docker.calls))
        class Wrong(Docker):
            def call(self,args): return 'systemd'
        with patch.object(setup,'preflight_host_cgroup'), patch.object(setup.platform,'node',return_value='test-host'):
            with self.assertRaisesRegex(ValueError,'setup_docker_daemon_host_mismatch'):
                setup.preflight_docker(self.request,docker=Wrong())

    def test_local_unqualified_templates_admit_both_models_without_receipt(self):
        self.request['measurement']=None
        result=setup.generate(self.request,verifier=lambda *args: None)
        config=result['configuration'];profile=config['profiles'][0]
        self.assertEqual((profile['support'],profile['verification'],profile['qualification_receipt']),
                         ('unqualified','synthetic',''))
        self.assertLessEqual(profile['limits']['ram_bytes'],4*1024**3)
        for context in config['contexts']['alice'].values():
            self.assertTrue(context_allowed(context,profile))
            plan=context['plan']
            oracle.admission(dict(operation=plan['operation'],input_sha256=oracle.plan_hash(plan)),context,[profile])
            from service.admission import Admission
            class InstalledAvailability:
                family=qd.FAMILY
                def check_availability(self, context): pass
            demand=Admission.demand(SimpleNamespace(native_availability=InstalledAvailability().check_availability),context,[profile])
            self.assertEqual((demand['slots'],demand['ram_bytes'],demand['pids']),
                             (1,4*1024**3,48))
            self.assertLess(demand['disk_bytes'], profile['limits']['disk_bytes'])
            self.assertEqual(demand['disk_bytes'], demand['attempt_disk_bytes'])
        other=dict(profile,config_revision='production-v1')
        self.assertFalse(context_allowed(config['contexts']['alice']['Ryan'],other))
        with self.assertRaises(Exception):
            plan=config['contexts']['alice']['Ryan']['plan']
            oracle.admission(dict(operation=plan['operation'],input_sha256=oracle.plan_hash(plan)),
                             config['contexts']['alice']['Ryan'],[other])

    def test_generated_ryan_request_reaches_worker_only_in_installed_local_lane(self):
        from service import install_policy as ip
        from service.execution_context import installed_preset_context
        from worker.runtime_supervisor import RuntimeSupervisor
        from worker.engines.qwen_execution import inventory_from_wire
        from worker.transport import Refused
        self.request['measurement']=None
        config=setup.generate(self.request,verifier=lambda *args: None)['configuration']
        profile=config['profiles'][0];template=config['contexts']['alice']['Ryan']
        context=installed_preset_context(template,'alice','Ryan')
        worker=object.__new__(RuntimeSupervisor)
        worker.resident_qwen=True;worker.installed_policy=ip.QWEN_SPLIT
        worker.qwen_contexts={('alice','qwen-cpu','Ryan'):template}
        worker.inventory=inventory_from_wire(context['execution']['hardware'])
        worker.qwen_profiles=[profile];worker.bundles={};worker.references={}
        with patch('worker.runtime_protocol.resident_qwen_cpu_hardware'), \
             patch('worker.engines.qwen.verify_service_inputs') as inert_engine:
            worker.verify_qwen(context,[profile],'alice')
            inert_engine.assert_called_once()
        worker.installed_policy=None
        with self.assertRaisesRegex(Refused,'real_qualification_required'):
            worker.verify_qwen(context,[profile],'alice')

    def test_qwen_up_entrypoint_prepares_local_lane_without_receipt(self):
        from service.package_install import main as command
        args=['qwen-up','--clone-bundle',self.request['bundles']['clone'],
              '--ryan-bundle',self.request['bundles']['Ryan'],'--context','local-test',
              '--control-image','control:local','--worker-image','worker:local',
              '--helper-image','helper:local','--custody',self.request['custody'],
              '--reference-wav',self.request['reference']['wav'],
              '--reference-text',self.request['reference']['text'],
              '--root',self.request['root'],'--handle',str(self.root/'handle'),
              '--prepare-only']
        with patch.object(setup,'_image_ids',return_value=self.request['images']), \
             patch('worker.engines.qwen.verify_bundle'), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            command(args)
        self.assertEqual(json.loads(output.getvalue())['qualification'],'unqualified')
        installed=json.loads((self.root/'local'/'install-request.json').read_text())
        self.assertEqual(installed['configuration']['profiles'][0]['qualification_receipt'],'')
        sudo_root=self.root/'sudo-client'
        sudo_args=args.copy();sudo_args[sudo_args.index(self.request['root'])]=str(sudo_root)
        sudo_uid=12345;sudo_gid=12346
        sudo_user=SimpleNamespace(pw_uid=sudo_uid,pw_gid=sudo_gid,pw_dir=str(self.root))
        with patch.object(setup,'_image_ids',return_value=self.request['images']), \
             patch('worker.engines.qwen.verify_bundle'), \
             patch.object(setup.os,'geteuid',return_value=0), \
             patch.object(setup.pwd,'getpwuid',return_value=sudo_user) as passwd, \
             patch.object(setup,'prepare_client_parent') as prepare_parent, \
             patch.object(setup.os,'chown') as chown, \
             patch.dict(os.environ,{'SUDO_UID':str(sudo_uid),'SUDO_GID':str(sudo_gid)}), \
             contextlib.redirect_stdout(io.StringIO()):
            command(sudo_args)
        passwd.assert_called_once_with(sudo_uid)
        prepare_parent.assert_called_once_with(sudo_root,(sudo_uid,sudo_gid,str(self.root)))
        self.assertEqual([call.args[0] for call in chown.call_args_list],
                         [sudo_root/'control.key',sudo_root/'client.json',sudo_root])
        self.assertTrue(all(call.args[1:] == (sudo_uid,sudo_gid) and
                            call.kwargs == {'follow_symlinks':False}
                            for call in chown.call_args_list))

    def test_public_up_routes_local_and_legacy_up_keeps_explicit_scope(self):
        from service.package_install import main as command
        args=['up','--clone-bundle',self.request['bundles']['clone'],
              '--ryan-bundle',self.request['bundles']['Ryan'],
              '--control-image','control:local','--worker-image','worker:local',
              '--helper-image','helper:local','--reference-wav',self.request['reference']['wav'],
              '--reference-text',self.request['reference']['text'],
              '--root',self.request['root'],'--handle',str(self.root/'handle'),'--prepare-only']
        with patch.object(setup,'_image_ids',return_value=self.request['images']), \
             patch('worker.engines.qwen.verify_bundle'), contextlib.redirect_stdout(io.StringIO()) as output:
            command(args)
        self.assertEqual(json.loads(output.getvalue())['qualification'],'unqualified')
        with patch.object(setup,'main') as local_main:
            with self.assertRaises(SystemExit):
                command(['up','--configuration','/abs/config','--scope','/abs/scope'])
            local_main.assert_not_called()

    def test_invalid_sudo_identity_precedes_private_file_creation(self):
        with patch.object(setup.os,'geteuid',return_value=0), \
             patch.dict(os.environ,{'SUDO_UID':'123456789','SUDO_GID':'123456789'}), \
             patch.object(setup,'_image_ids') as images:
            with self.assertRaisesRegex(ValueError,'setup_sudo_client_identity_required'):
                setup.main(['--local-unqualified','--handle',str(self.root/'handle'),
                            '--root',str(self.root/'never-created'),'--prepare-only'])
        images.assert_not_called()
        self.assertFalse((self.root/'never-created').exists())
        with patch.object(setup.os,'geteuid',return_value=0), \
             patch.dict(os.environ,{'SUDO_UID':str(os.getuid()),'SUDO_GID':str(os.getgid()+1)}), \
             patch.object(setup,'_image_ids') as images:
            with self.assertRaisesRegex(ValueError,'setup_sudo_client_identity_required'):
                setup.main(['--local-unqualified','--handle',str(self.root/'handle'),
                            '--root',str(self.root/'never-created'),'--prepare-only'])
        images.assert_not_called()
        self.assertFalse((self.root/'never-created').exists())

    def test_extracted_archive_public_up_help_imports_complete_local_lane(self):
        from service.package_install import archive
        package=self.root/'speech.zip'
        archive(setup.ROOT,package)
        with zipfile.ZipFile(package) as source:
            source.extractall(self.root/'extracted')
        command=self.root/'extracted'/'speech-box'/'speech-box'
        result=subprocess.run([sys.executable,str(command),'up','--help'],
                              capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--clone-bundle',result.stdout)
        request=self.root/'extracted-request.json'
        data=dict(self.request,measurement=None,ram_bytes=6*1024**3)
        request.write_text(json.dumps(data))
        script=("import json,sys;sys.path.insert(0,sys.argv[1]);"
                "from service import qwen_service_setup as s;"
                "r=s.generate(json.load(open(sys.argv[2])),verifier=lambda *args:None);"
                "print(r['configuration']['profiles'][0]['support'])")
        generated=subprocess.run([sys.executable,'-c',script,str(command.parent),str(request)],
                                 capture_output=True,text=True,timeout=15)
        self.assertEqual(generated.returncode,0,generated.stderr)
        self.assertEqual(generated.stdout.strip(),'unqualified')

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


class UpDispatchTests(unittest.TestCase):
    def test_legacy_discriminators_route_separate_equal_and_abbreviated_forms(self):
        from service.package_install import main
        options = {'--configuration': '--conf', '--scope': '--sc',
                   '--setup-bundle': '--setup', '--host-policy': '--host-p',
                   '--initialization-digest': '--init'}
        for full, abbreviated in options.items():
            for name in (full, abbreviated):
                for tail in ([name, '/missing'], [name + '=/missing']):
                    with self.subTest(tail=tail), patch.object(setup, 'main') as local_main:
                        with self.assertRaises(SystemExit):
                            main(['up', *tail])
                        local_main.assert_not_called()

    def test_mixed_legacy_forms_route_without_starting_local_setup(self):
        from service.package_install import main
        for tail in (['--configuration=/missing', '--scope', '/missing'],
                     ['--configuration', '/missing', '--scope=/missing']):
            with self.subTest(tail=tail), patch.object(setup, 'main') as local_main:
                with self.assertRaises(SystemExit):
                    main(['up', *tail])
                local_main.assert_not_called()

    def test_local_values_containing_legacy_option_names_stay_local(self):
        from service.package_install import main
        tail = ['--reference-text=--configuration=config.json',
                '--root', '/tmp/--scope=scope.json',
                '--clone-bundle=contains--setup-bundle=x']
        with patch.object(setup, 'main', return_value='local') as local_main:
            self.assertEqual(main(['up', *tail]), 'local')
            local_main.assert_called_once_with(['--local-unqualified', *tail])


if __name__=='__main__': unittest.main()
