"""Inert installed two-model wiring; no Docker or model qualification."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from contracts import validate as oracle
from service import installed_qwen_daemon as qd, install_policy as ip
from service import execution_context as ec, daemon_volumes as dv
from service.qwen_service_config import build
from service import daemon_client as dc
from worker.runtime_protocol import execution


class ServicePathTests(unittest.TestCase):
    def setUp(self):
        from tests.service.test_qwen_preparation import CPUPreparationFixture
        from tests.service.test_qwen_lifecycle import QwenLifecycleTests
        from tests.service.test_installed_qwen import configuration
        self.clone_fixture=CPUPreparationFixture();self.clone_fixture.setUp()
        self.addCleanup(self.clone_fixture.doCleanups)
        self.ryan_fixture=QwenLifecycleTests();self.ryan_fixture.task='custom_voice'
        self.ryan_fixture.setUp();self.addCleanup(self.ryan_fixture.doCleanups)
        base=configuration(self.clone_fixture)
        self.clone=base['contexts']['alice'];self.profile=base['profiles'][0]
        self.clone['execution']['descriptor'].update(x_vector_only=False,reference_text='Trusted template transcript.')
        self.clone['plan']['voice_versions'][0]['reference_text']='Trusted template transcript.'
        self.clone['plan']['voice_versions'][0]['configuration'].update(
            reference_text='Trusted template transcript.',conditioning='icl')
        for reference in self.clone['execution']['references'].values():
            reference['text']='Trusted template transcript.'
        r=self.ryan_fixture
        self.ryan=json.loads(json.dumps(r.context).replace('rev1',qd.CUSTOM_REVISION)
            .replace('test',qd.PUBLIC_CUSTOM_MODEL)
            .replace(r.spec.manifest_sha256,qd.CUSTOM_MANIFEST_SHA256))
        self.ryan['execution']['descriptor']['checkpoint']=qd.CUSTOM_CHECKPOINT
        self.ryan['execution']['hardware']=copy.deepcopy(self.clone['execution']['hardware'])
        self.profile['compatibility'].append(dict(operation='speech_generate',mode='qwen_preset',
            identity=self.ryan['plan']['resolved']['identity']['effective']))
        for context in (self.clone,self.ryan):
            context['plan']['resolved']['profile']=oracle.profile_snapshot(self.profile)
        self.clone_digest=execution(self.clone).manifest_sha256
        self.enterContext(patch.dict(qd.SERVICE_MODELS,{'clone':(qd.CHECKPOINT,qd.REVISION,self.clone_digest)}))
        self.enterContext(patch.object(qd,'MANIFEST_SHA256',self.clone_digest))
        self.config=build('alice',self.clone,self.ryan,self.profile,base['budget'],base['images'])

    def test_resident_driver_gate_precedes_docker_mutation(self):
        image=self.config['images']['control']
        request=dict(configuration=self.config,helper_image=image,
            inputs=[dict(role='config',source='config',bytes=100),dict(role='credential',source='credential',bytes=100)],
            cpu=dict(control=10**9,worker=2*10**9),
            tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},
            control_port=8765,custody='/sys/fs/cgroup/speech-box-qwen/'+'b'*32)
        info=dict(ID='daemon',OSType='linux',Architecture='amd64',SecurityOptions=[],CgroupVersion='2')
        context=[dict(Endpoints=dict(docker=dict(Host='unix:///run/docker.sock')))]
        credentials=SimpleNamespace(_credentials=[SimpleNamespace(principal='alice',scopes={'read','production-write'})])
        for driver in (None,'systemd','unknown','cgroupfs'):
            with self.subTest(driver=driver), \
                 patch.object(dc,'read',side_effect=lambda path,*_:dc.encoded(request) if path=='request' else b'credential'), \
                 patch.object(dc,'manifest',return_value=request['inputs']), \
                 patch('service.auth.decode_credentials',return_value=credentials), \
                 patch.object(dc.Docker,'json',side_effect=[context,dict(info,**({'CgroupDriver':driver} if driver is not None else {}))]), \
                 patch.object(dc.Docker,'call',side_effect=RuntimeError('first Docker call after info')) as call:
                with self.assertRaises(SystemExit) as failure:
                    dc.main(['install','--context','local','--request','request','--handle','handle'],family=qd.FAMILY)
                if driver=='cgroupfs':
                    self.assertEqual(call.call_args.args[0][:2],['image','inspect'])
                else:
                    self.assertIn('CgroupDriver must be cgroupfs',str(failure.exception))
                    call.assert_not_called()

    def test_retained_resident_handle_rechecks_driver_before_mutation(self):
        installation='a'*32
        b=dict(policy=qd.LAUNCH,installation=installation,
            daemon=dict(endpoint='unix:///run/docker.sock',id='daemon',os='linux',architecture='amd64',security=[],cgroup='2'),
            volumes={r:dict(Name='pab-'+installation+'-'+r,Driver='local',Options=None,Scope='local',
                Mountpoint='/var/lib/docker/volumes/'+r,CreatedAt='2026-10-25T00:00:00Z',
                Labels={'speech-box.installation':installation,'speech-box.role':r}) for r in dv.ROLES},
            helper_image=self.config['images']['control'],configuration_digest=dv.digest(self.config),
            cpu=dict(control=10**9,worker=2*10**9),
            tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},
            control_port=8765,custody='/sys/fs/cgroup/speech-box-qwen/'+'b'*32)
        retained=dict(binding=b,configuration=self.config)
        info=dict(ID='daemon',OSType='linux',Architecture='amd64',SecurityOptions=[],CgroupVersion='2',CgroupDriver='systemd')
        with patch.object(dc,'read',return_value=dc.encoded(retained)), \
             patch.object(dc.Docker,'json',return_value=info), \
             patch.object(dc.Docker,'call') as call:
            with self.assertRaises(SystemExit) as failure:
                dc.main(['status','--handle','handle'],family=qd.FAMILY)
            self.assertIn('CgroupDriver must be cgroupfs',str(failure.exception))
            call.assert_not_called()

    def test_driver_is_not_added_to_legacy_daemon_record(self):
        info=dict(ID='daemon',OSType='linux',Architecture='amd64',SecurityOptions=[],CgroupVersion='2',CgroupDriver='systemd')
        with patch.object(dc.Docker,'json',return_value=info):
            d=dc.Docker(endpoint='unix:///run/docker.sock',family=qd.FAMILY)
            self.assertEqual(set(d.frozen),{'endpoint','id','os','architecture','security','cgroup'})
            self.assertEqual(d.cgroup_driver,'systemd')
            d.check()
            with self.assertRaisesRegex(ValueError,'CgroupDriver must be cgroupfs'):
                d.require_resident_cgroupfs()

        with patch.object(dc.Docker,'json',side_effect=[dict(info,CgroupDriver='cgroupfs'),info]):
            d=dc.Docker(endpoint='unix:///run/docker.sock',family=qd.FAMILY)
            d.require_resident_cgroupfs()
            with self.assertRaisesRegex(ValueError,'CgroupDriver must be cgroupfs'):
                d.check()

    def test_builder_configuration_launch_and_one_principal(self):
        self.assertTrue(ip.qwen_service(self.config))
        self.assertEqual(set(self.config['contexts']),{'alice'})
        self.assertEqual(set(self.config['bundles']),{self.clone_digest,qd.CUSTOM_MANIFEST_SHA256})
        self.assertEqual(len(list(ip.context_values(self.config))),2)
        installation='a'*32
        b=dict(policy=qd.LAUNCH,installation=installation,
            daemon=dict(endpoint='unix:///run/docker.sock',id='fixture-daemon',os='linux',architecture='amd64',security=[],cgroup='2'),
            volumes={r:dict(Name='pab-'+installation+'-'+r,Driver='local',Options=None,Scope='local',
                Mountpoint='/var/lib/docker/volumes/'+r,CreatedAt='2026-10-25T00:00:00Z',
                Labels={'speech-box.installation':installation,'speech-box.role':r}) for r in dv.ROLES},
            helper_image=self.config['images']['control'],configuration_digest=dv.digest(self.config),
            cpu=dict(control=10**9,worker=2*10**9),
            tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},control_port=8765,
            custody='/sys/fs/cgroup/speech-box-qwen/'+'b'*32)
        args=dv.command('worker',self.config,b,family=qd.FAMILY)
        self.assertEqual(args[-2:],["--resident-qwen-policy",qd.SERVICE_POLICY])
        self.assertIn('--cgroup-parent',args)
        missing=copy.deepcopy(b);missing.pop('custody')
        with self.assertRaisesRegex(ValueError,'qwen_service_delegated_cgroup_required'):
            qd.binding_configuration(missing,self.config)
        custody=next(m for m in dv.mounts('worker',b,family=qd.FAMILY) if m['Destination']=='/run/attempt-cgroups')
        self.assertEqual((custody['Type'],custody['Source'],custody['RW']),('bind',b['custody'],True))
        selected=dv.mounts('worker',b,family=qd.FAMILY)
        requested=[dict(Type=m['Type'],Source=m['Source'] if m['Type']=='bind' else m['Name'],
            Target=m['Destination'],ReadOnly=not m['RW'],
            **(dict(BindOptions=dict(Propagation='rprivate')) if m['Type']=='bind' else
               dict(VolumeOptions=dict(NoCopy=True,**({'Subpath':'core/assets'} if m['Destination']=='/references/assets' else {})))))
            for m in selected]
        t=b['tmpfs']['worker']
        inspection=dict(Id='2'*64,Image=self.config['images']['worker'],
            Config=dict(User='0:10001',Entrypoint=['python'],Cmd=args[args.index('--entrypoint')+3:]),
            HostConfig=dict(Privileged=False,ReadonlyRootfs=True,Init=None,PidMode='',IpcMode='private',
                CgroupnsMode='host',UsernsMode='',NetworkMode='none',CapDrop=['ALL'],CapAdd=list(dv.WORKER_CAPS),
                SecurityOpt=['no-new-privileges:true'],RestartPolicy=dict(Name='no',MaximumRetryCount=0),
                Devices=[],DeviceRequests=None,Binds=None,CgroupParent=b['custody'].removeprefix('/sys/fs/cgroup'),
                Memory=self.config['budget']['ram_bytes'],MemorySwap=self.config['budget']['ram_bytes'],
                PidsLimit=self.config['budget']['pids'],Mounts=requested,NanoCpus=b['cpu']['worker'],
                Tmpfs={'/tmp':'rw,noexec,nosuid,nodev,size='+str(t['bytes'])+',nr_inodes='+str(t['inodes'])+',mode=1777'},
                PortBindings={}),
            State=dict(Running=False,Restarting=False,ExitCode=0,Status='created',StartedAt='0001-01-01T00:00:00Z'),
            RestartCount=0,Mounts=selected)
        dv.inspect_container(inspection,'worker',self.config,b,family=qd.FAMILY)
        changed=copy.deepcopy(inspection)
        changed['HostConfig']['CgroupParent']='/wrong'
        with self.assertRaises(ValueError):dv.inspect_container(changed,'worker',self.config,b,family=qd.FAMILY)
        changed=copy.deepcopy(inspection)
        changed['HostConfig']['CgroupnsMode']='private'
        with self.assertRaises(ValueError):dv.inspect_container(changed,'worker',self.config,b,family=qd.FAMILY)
        from service.daemon_client import manifest
        entries=[dict(role='models',path='bundles/'+key+'/'+name,source='/inert/'+name+key,
            bytes=size,sha256='a'*64) for key in self.config['bundles'] for name,size in
            (('model.safetensors',1800*1024**2),('speech_tokenizer/model.safetensors',600*1024**2))]
        entries.extend((dict(role='config',path='install.json',source='/inert/config',bytes=100,sha256='b'*64),
                        dict(role='credential',path='control.key',source='/inert/credential',bytes=100)))
        self.assertEqual(len(manifest(entries,family=qd.FAMILY,service=True)),6)
        with self.assertRaises(ValueError):manifest(entries,family=qd.FAMILY)
        changed=copy.deepcopy(inspection)
        changed['HostConfig']['Mounts'][-1]['Source']='/sys/fs/cgroup/other'
        with self.assertRaises(ValueError):dv.inspect_container(changed,'worker',self.config,b,family=qd.FAMILY)
        bad=copy.deepcopy(self.config)
        bad['contexts']['alice']['Ryan']['execution']['descriptor']['preset']='Other'
        with self.assertRaises(ValueError):qd.configuration(bad)

    def test_core_contexts_select_ryan_and_clone(self):
        mapping={key:'/models/bundles/'+key for key in self.config['bundles']}
        inventory=self.clone_fixture.b.inventory
        with patch('worker.engines.qwen.verify_bundle'),patch.object(ec,'verify_qwen_local'), \
                patch('sys.platform','linux'),patch('os.sched_getaffinity',return_value={0,1},create=True):
            contexts=ec.QwenContexts(self.config['contexts'],mapping,{},inventory,[self.profile],installed_family=qd.FAMILY)
            self.assertEqual([p['preset'] for p in contexts.presets('alice')],['Ryan'])
            version=contexts.preset_version('alice','Ryan','voice-ryan')
            request=copy.deepcopy(self.ryan['plan']['request'])
            request['voice_version_id']=version['id']
            selected=contexts.resolve_registered('alice',request,version)
            self.assertEqual(execution(selected).checkpoint,qd.CUSTOM_CHECKPOINT)
            self.assertEqual(contexts.capabilities('alice')['engine'],'qwen')
            self.assertIn('voice_prepare',contexts.capabilities('alice')['operations'])
            self.assertIn('voice_preset_register',contexts.capabilities('alice')['operations'])
            from service.dispatcher import Dispatcher
            dispatcher=Dispatcher.__new__(Dispatcher)
            dispatcher.production_contexts=contexts
            dispatcher.profiles=[self.profile]
            self.assertEqual([p['preset'] for p in dispatcher._command('configured_presets','alice')],['Ryan'])
            self.assertIn('voice_preset_register',dispatcher._command('configured_capabilities','alice')['operations'])

    def test_base_transcript_prepare_and_one_principal_credentials(self):
        from service.auth import decode_credentials
        template=copy.deepcopy(self.config['contexts']['alice']['clone'])
        descriptor=template['execution']['descriptor']
        descriptor.update(x_vector_only=False,reference_text='Trusted template transcript.')
        version=template['plan']['voice_versions'][0]
        version['reference_text']='Trusted template transcript.'
        version['configuration'].update(reference_text='Trusted template transcript.',conditioning='icl')
        for reference in template['execution']['references'].values():
            reference['text']='Trusted template transcript.'
        request=copy.deepcopy(self.clone_fixture.request)
        request.update(reference_text='Bruce and John reference transcript.',
                       identity=template['plan']['resolved']['identity']['effective'])
        result=ec.cpu_clone_context(template,'alice',request,self.clone_fixture.b.asset)
        self.assertEqual(execution(result).reference_text,request['reference_text'])
        self.assertEqual(result['plan']['resolved']['reuse_configuration']['conditioning'],'icl')
        credential=json.dumps(dict(version=1,credentials=[dict(token='a'*64,principal='alice',
            scopes=['read','production-write'])])).encode()
        self.assertEqual(decode_credentials(credential,self.config['contexts']).resolve('Bearer '+'a'*64), 'alice')

    def test_independent_worker_reconstructs_both_model_authorities(self):
        from worker.runtime_supervisor import RuntimeSupervisor
        from worker.runtime_protocol import resident_attempt_authority
        supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
        supervisor.resident_qwen=True
        supervisor.installed_policy=ip.QWEN_SPLIT
        supervisor.bundles={key:'/models/bundles/'+key for key in self.config['bundles']}
        supervisor.qwen_profiles=[self.profile]
        supervisor.inventory=self.clone_fixture.b.inventory
        supervisor.references={}
        clone=self.config['contexts']['alice']['clone']
        ryan=self.config['contexts']['alice']['Ryan']
        supervisor.qwen_contexts={('alice','qwen-cpu','managed-clone'):clone,
                                  ('alice','qwen-cpu','Ryan'):ryan}
        request=copy.deepcopy(self.clone_fixture.request)
        request.update(reference_text='Bruce and John exact transcript.',
                       identity=clone['plan']['resolved']['identity']['effective'])
        prepared=ec.cpu_clone_context(clone,'alice',request,self.clone_fixture.b.asset)
        preset=ec.installed_preset_context(ryan,'alice','voice-ryan')
        with patch('worker.runtime_protocol.resident_qwen_cpu_hardware'), \
                patch('worker.engines.qwen.verify_service_inputs') as verified, \
                patch('worker.managed_reference.acquire',return_value=42),patch('os.close'):
            self.assertEqual(supervisor.verify_qwen(prepared,[self.profile],'alice',retain_fd=True),42)
            supervisor.verify_qwen(preset,[self.profile],'alice')
        self.assertEqual(verified.call_count,2)
        roots={'clone':dict(path='/models/bundles/'+self.clone_digest,revision=qd.REVISION),
               'Ryan':dict(path='/models/bundles/'+qd.CUSTOM_MANIFEST_SHA256,revision=qd.CUSTOM_REVISION)}
        clone_call=resident_attempt_authority(execution(prepared),prepared['plan'],roots,1000.,'a'*64)
        ryan_call=resident_attempt_authority(execution(preset),preset['plan'],roots,1000.,'b'*64)
        self.assertEqual(clone_call['reference_text'],request['reference_text'])
        self.assertEqual(ryan_call['preset'],'Ryan')
        self.assertNotEqual(clone_call['settings']['model_name'],ryan_call['settings']['model_name'])


if __name__ == '__main__': unittest.main()
