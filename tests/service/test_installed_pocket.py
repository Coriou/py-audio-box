"""Installed Pocket SOURCE fixtures. Host/UID/reap/backend are injected.

No ML, dependency install, Docker, network, listener, kernel UID transition or
actual helper subprocess. Real temporary files, Core enrollment and public APIs.
"""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import unittest
from unittest.mock import patch
from service import entrypoint as e, install_policy as ip, pocket_install
from service.core import encode
from service.execution_context import installed_contexts
from service.provisioning import bundle_manifest, manifest_bytes, _pocket_config
from tests.service.test_pocket_preparation import PocketPublicJourneyTests, PreparationFixture
from tests.service import test_provisioning, test_split_init, test_installed_registration
from tests.release.test_headless_install import GateTests


def configuration(context, profiles):
    from service.admission import DEFAULT_BUDGET
    spec=context['execution']['descriptor'];bid=spec['bundle_id']
    return dict(version=1, ownership_policy=ip.POCKET_POLICY, profiles=copy.deepcopy(profiles),
        budget=copy.deepcopy(DEFAULT_BUDGET), contexts={'alice':copy.deepcopy(context)},
        bundles={bid:dict(manifest_sha256=spec['manifest_sha256'],checkpoint=spec['checkpoint'],revision=spec['revision'],path='/models/bundles/'+bid)},
        images=dict(control='sha256:'+'1'*64,worker='sha256:'+'2'*64),
        mounts=dict(control='/control',staging='/staging',receipts='/receipts',ipc='/ipc',models='/models'))


class InstalledPocketJourney(PocketPublicJourneyTests):
    installed_journey = True

    def setUp(self):
        PreparationFixture.setUp(self)
        b=self.base
        pins=test_provisioning.PocketProvisioningTests();pins.setUp();self.addCleanup(pins.doCleanups)
        pins.expected.update(checkpoint='test',language='fr',versions=pocket_install.runtime_versions())
        manifest=bundle_manifest(pins.expected);raw=manifest_bytes(manifest)
        # Materialize only inert fixture bytes; never call provisioning/download.
        for p in b.bundle.iterdir():p.unlink()
        for name,data in {**pins.data,'config.yaml':_pocket_config(pins.expected),'manifest.json':raw}.items():
            (b.bundle/name).write_bytes(data);(b.bundle/name).chmod(0o444)
        b.bundle.chmod(0o555)
        self.addCleanup(lambda:b.bundle.chmod(0o755))
        self.spec=replace(self.spec,manifest_sha256=hashlib.sha256(raw).hexdigest(),revision=manifest['revision'])
        identity=self.context['plan']['resolved']['identity']['effective']
        identity['bundle_sha256']=self.spec.manifest_sha256
        self.context['plan']['resolved']['identity']['requested']=copy.deepcopy(identity)
        self.context['plan']['request']['identity']=copy.deepcopy(identity)
        b.context['plan']['request']['identity']=copy.deepcopy(identity)
        for c in self.profiles[0]['compatibility']:c['identity']=copy.deepcopy(identity)
        from contracts import validate as oracle
        self.context['plan']['resolved']['profile']=oracle.profile_snapshot(self.profiles[0])
        self.context['bundles'][0]['bundle_sha256']=self.spec.manifest_sha256
        self.context['execution']['descriptor']=self.spec.identity()
        self.config=configuration(self.context,self.profiles)
        outer=self
        class Registration(test_installed_registration.Registration):
            def new_split_fixture(inner):
                f=test_split_init.SplitInit();f.setUp();f.config=copy.deepcopy(outer.config);return f
        def registered():
            r=Registration();r.setUp();return r
        self.gate_fixture=gate=GateTests();gate.new_registration=registered;gate.setUp();self.addCleanup(gate.doCleanups)
        f=gate.fixture.fixture
        original=f.translate
        def translated(info,path=None):
            if path is not None and any(path==p or p in path.parents for p in (b.bundle,f.roots['staging'])):
                from types import SimpleNamespace
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=10001,st_gid=10001))
            return original(info,path)
        f.translate=translated
        real_lstat=os.lstat
        self.enterContext(patch.object(os,'lstat',side_effect=lambda path,*a,**kw: translated(real_lstat(path,*a,**kw),Path(path))))
        for name in ('getuid','geteuid','getgid','getegid'):self.enterContext(patch.object(os,name,return_value=10001))
        self.enterContext(patch.object(os,'getgroups',return_value=[]))
        self.enterContext(patch.object(e,'local_incarnation',return_value=gate.incarnation))
        old=os.umask(0o077);self.addCleanup(os.umask,old)
        directory=gate.notification_directory()
        self.waiting=e.WaitingControl(directory=directory);self.addCleanup(self.waiting.close)
        e.publish(encode(gate.gate).encode(),directory=directory)

    def journey_paths(self):
        return self.gate_fixture.root,self.gate_fixture.fixture.roots['staging']

    def journey_hello(self):return copy.deepcopy(self.gate_fixture.fixture.hello)

    def journey_identity(self,count):
        h=self.journey_hello();return h['lane']+'/'+h['boot']+'/'+format(count,'032x')

    def journey_contexts(self):
        from service.execution_context import installed_mapping
        # Only fixed container-to-host fixture path translation is injected.
        from service.execution_context import verify_local
        with patch('service.execution_context.verify_local',side_effect=lambda c,m,policy:verify_local(c,self.base.mapping,policy=policy)) as verify:
            installed_mapping(self.config,policy=ip.SPLIT)
            self.assertEqual(verify.call_count,1)
        return installed_contexts(self.config,self.base.mapping)

    def journey_owner(self,root,stage,runtime,configured):
        from service.dispatcher import Dispatcher
        return Dispatcher(root,staging=stage,runtime=runtime,profiles=self.profiles,budget=self.config['budget'],
            production_contexts=configured,installed_activation=e.InstalledActivation(self.waiting,self.config))

    def journey_app(self,owner,credentials):
        from service.api import create_voice_app
        return create_voice_app(owner,credentials=credentials,installed=self.config)

    def journey_supervisor(self, supervisor, authority, root):
        supervisor.installed_pocket=True
        supervisor.pocket_reference_roots={'alice':'/references/assets'}
        def acquire(identity):
            return os.open(root/'assets'/identity,os.O_RDONLY|os.O_NOFOLLOW)
        with patch('worker.managed_reference.acquire',side_effect=acquire):
            supervisor.verify_pocket(authority.context,authority.job['principal_id'])

    def journey_execute(self, runner, spec, text, bundles, references, kwargs):
        fd=os.open(references[spec.reference_id],os.O_RDONLY|os.O_NOFOLLOW)
        try:
            kwargs['reference_fd']=fd
            return runner(spec,text,bundles,{spec.reference_id:'/references/assets/'+spec.reference_id},**kwargs)
        finally:os.close(fd)

    def test_fresh_v2_enrollment_registration_and_legacy_immutability(self):
        from service import persistent_init, host_install
        config = copy.deepcopy(self.config)
        config['ownership_policy'] = ip.POCKET_V2_POLICY
        self.assertIs(ip.configuration_policy(config), ip.POCKET_SPLIT)
        self.assertIsNot(ip.POCKET_SPLIT, ip.EXPORT_SPLIT)
        self.assertEqual(ip.expectation(ip.POCKET_SPLIT, 'staging', directory=True), (0,10001,0o1753))
        self.assertEqual(ip.expectation(ip.SPLIT, 'staging', directory=True), (0,0,0o1733))
        for role in ('control','receipts','ipc'):
            self.assertEqual(ip.expectation(ip.POCKET_SPLIT,role,directory=True),ip.expectation(ip.SPLIT,role,directory=True))
        class Registration(test_installed_registration.Registration):
            def new_split_fixture(inner):
                f=test_split_init.SplitInit();f.setUp();f.config=copy.deepcopy(config)
                f.run_init=lambda **kw: persistent_init.initialize_split(f.roots,f.scope,f.config,
                    authority=f.authority,policy=ip.POCKET_SPLIT,launch_child=f.child,**kw)
                return f
        r=Registration();r.setUp();self.addCleanup(r.doCleanups)
        self.assertEqual(r.public['scope'],ip.POCKET_V2_SCOPE)
        self.assertEqual(r.public['roles']['staging']['gid'],10001)
        self.assertEqual(r.public['roles']['staging']['mode'],0o1753)
        ip.installation(r.public)
        forged=copy.deepcopy(r.public);forged['roles']['staging'].update(gid=0,mode=0o1733)
        with self.assertRaises(ValueError):ip.installation(forged)
        with self.assertRaises(ValueError):
            persistent_init.initialize_split(r.roots,r.scope,config,authority=r.fixture.authority,
                policy=ip.SPLIT,launch_child=r.fixture.child)
        # Registration reads the full staging inventory under the explicit token.
        rid=r.run_registration()
        # Control startup binds the protected configuration to retained enrollment.
        record=json.loads(r.rows()[0][1])
        from service import recovery
        from service.persistent_init import digest
        processes=[dict(id=c['id'],image=c['image'],started=c['started'],pid=90+i,
            start_ticks=100+i,pid_namespace=200+i,cgroup='/docker/'+c['id'])
            for i,c in enumerate(record['containers'])]
        gate=dict(version=1,registration_id=rid,registration=record,
            configuration_digest=digest(config),enrollment_digest=digest(r.public),
            provenance_digest='e'*64,processes=processes)
        args=(r.roots['control']/'core',config,gate,r.hello,
              dict(start_ticks=100,pid_namespace=200,cgroup='0::/'))
        self.assertEqual(recovery.installed_startup_read(*args,policy=ip.POCKET_SPLIT)['registration_id'],rid)
        with self.assertRaises(ValueError):recovery.installed_startup_read(*args,policy=ip.SPLIT)
        state=host_install._registration_snapshot(r.roots,r.scope,policy=ip.POCKET_SPLIT)
        self.assertFalse(state['faults'])
        with self.assertRaises(ValueError):host_install._registration_snapshot(r.roots,r.scope,policy=ip.SPLIT)
        self.assertEqual(self.config['ownership_policy'],ip.POCKET_POLICY)
        legacy=self.gate_fixture.fixture.fixture
        retained=legacy.bytes()
        with self.assertRaisesRegex(ValueError,'new_empty_installation_required'):
            persistent_init.initialize_split(legacy.roots,legacy.scope,config,
                authority=legacy.authority,policy=ip.POCKET_SPLIT,launch_child=legacy.child)
        self.assertEqual(legacy.bytes(),retained)
        legacy_public=json.loads((legacy.roots['receipts']/'p1-init.json').read_bytes())['public']
        self.assertEqual((legacy_public['roles']['staging']['gid'],legacy_public['roles']['staging']['mode']),(0,0o1733))


    def test_closed_policy_enrollment_and_reference_free_factory(self):
        self.assertEqual(ip.configuration(self.config),self.config)
        public=self.gate_fixture.fixture.public
        self.assertEqual(public['scope'],ip.POCKET_SCOPE)
        self.assertEqual(ip.installation(public),public)
        for change in ({'ownership_policy':ip.POLICY},{'contexts':{}},{'extra':True},{'profiles':self.profiles*2}):
            with self.assertRaises((ValueError,KeyError)):
                ip.configuration(dict(self.config,**change))
        bad=copy.deepcopy(public);bad['scope']=ip.SCOPE
        with self.assertRaises(ValueError):ip.installation(bad)
        factory=self.journey_contexts()
        self.assertTrue(factory.capabilities('alice')['installed'])
        self.assertIsNone(factory.capabilities('foreign'))
        manifest=json.loads((self.base.bundle/'manifest.json').read_bytes())
        pocket_install.manifest(manifest)
        for change in ({'versions':dict(manifest['versions'],torch='invented')},{'extra':True}):
            with self.assertRaises(ValueError):pocket_install.manifest(dict(manifest,**change))

    def test_exact_readonly_mount_and_launcher_policy(self):
        from tests.service.test_installed_host_policy import InstalledHost
        from service import host_install as h, package_install as package
        f=InstalledHost();f.setUp();self.addCleanup(f.doCleanups)
        f.config=copy.deepcopy(self.config)
        f.policy['configuration_digest']=h.digest(f.config)
        f.policy['cpu']['worker']=self.spec.threads*10**9
        f.containers[1]['nanocpus']=f.policy['cpu']['worker']
        reference=dict(type='bind',name='',source=f.policy['roots']['control']['source']+'/core/assets',destination='/references/assets',rw=False)
        f.containers[1]['mounts'].append(reference)
        f.validate()
        command=package.command('worker',f.config,f.policy,f.scope)
        self.assertIn(ip.POCKET_POLICY,command)
        self.assertIn('type=bind,source='+reference['source']+',destination=/references/assets,readonly',command)
        self.assertNotIn('--cap-add=DAC_READ_SEARCH',command)
        v2=copy.deepcopy(f.config);v2['ownership_policy']=ip.POCKET_V2_POLICY
        policy=copy.deepcopy(f.policy);policy['configuration_digest']=h.digest(v2)
        command_v2=package.command('worker',v2,policy,f.scope)
        self.assertIn(ip.POCKET_V2_POLICY,command_v2)
        self.assertIn('type=bind,source='+reference['source']+',destination=/references/assets,readonly',command_v2)
        for field,value in (('rw',True),('source',f.policy['roots']['control']['source']),('destination','/control'),('type','volume')):
            old=reference[field];reference[field]=value
            try:
                with self.assertRaises(ValueError):f.validate()
            finally:reference[field]=old
        f.containers[1]['mounts'].remove(reference)
        with self.assertRaisesRegex(ValueError,'reference_mount_coverage'):f.validate()

    def supervisor_fixture(self):
        import threading
        from worker.runtime_supervisor import RuntimeSupervisor
        s=RuntimeSupervisor.__new__(RuntimeSupervisor)
        s.bundles=self.base.mapping;s.references={};s.pocket_reference_roots={'alice':'/references/assets'};s.installed_pocket=True
        s.failed=False;s.resident_qwen=False;s.tree_lock=threading.Lock()
        path=self.gate_fixture.root/'assets'/self.spec.reference_id
        path.write_bytes(self.raw);path.chmod(0o600)
        return s,path

    def test_independent_descriptor_mapping_metadata_and_hash_refusals(self):
        from worker.transport import Refused
        s,path=self.supervisor_fixture()
        def acquire(identity):return os.open(path.parent/identity,os.O_RDONLY|os.O_NOFOLLOW)
        with patch('worker.managed_reference.acquire',side_effect=acquire):
            s.verify_pocket(self.context,'alice')
            with self.assertRaises(Refused):s.verify_pocket(self.context,'foreign')
            bad=copy.deepcopy(self.context);bad['execution']['references'][self.spec.reference_id]='/public/reference.wav'
            with self.assertRaises(Refused):s.verify_pocket(bad,'alice')
            path.chmod(0o644)
            with self.assertRaisesRegex(Refused,'descriptor'):s.verify_pocket(self.context,'alice')
            path.chmod(0o600);path.write_bytes(self.raw[:-1]+b'x')
            with self.assertRaisesRegex(Refused,'digest'):s.verify_pocket(self.context,'alice')
            path.unlink()
            with self.assertRaises(OSError):s.verify_pocket(self.context,'alice')
            self.assertTrue(s.failed)
            path.symlink_to(self.base.core.root/'assets'/self.spec.reference_id)
            with self.assertRaises(OSError):s.verify_pocket(self.context,'alice')

    def test_supervisor_launch_inherits_only_socket_and_selected_readonly_fd(self):
        from types import SimpleNamespace
        import time
        import threading
        from unittest.mock import Mock
        s,path=self.supervisor_fixture()
        row=dict(id='lane/boot/allocation',uid=20000,phase='prepared',allocation=dict(fixture='pocket',binding='frozen',
            snapshot=[dict(principal_id='alice',attempts=[dict(staging_id='stage-a')]),self.context,[],self.profiles]))
        s.lookup=lambda identity:row
        def change(current,**kw):current.update(kw);return current
        s.change=change;s.active=dict(id=row['id'],cutoff=time.monotonic()+5);s.launches=0
        s.journal=SimpleNamespace(value=dict(records=[row]))
        parent=Mock();child=Mock();child.fileno.return_value=1234
        descriptors=[]
        def acquire(identity):
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);descriptors.append(fd);return fd
        with patch('worker.managed_reference.acquire',side_effect=acquire), \
             patch('worker.runtime_supervisor.subprocess.run') as setup, \
             patch('worker.runtime_supervisor.subprocess.Popen',return_value=SimpleNamespace(pid=42)) as launch, \
             patch('worker.runtime_supervisor.socket.socketpair',return_value=(parent,child)), \
             patch('worker.runtime_supervisor.threading.Thread'):
            s.operation('start',dict(runtime_id=row['id']))
        self.assertEqual(row['phase'],'started')
        kw=launch.call_args.kwargs
        self.assertEqual((kw['user'],kw['group'],kw['extra_groups']),(20000,20000,[]))
        self.assertEqual(kw['pass_fds'],(1234,descriptors[-1]))
        for fd in descriptors:
            with self.assertRaises(OSError):os.fstat(fd)
        self.assertEqual(setup.call_args.kwargs['user'],10001)
        self.assertEqual(s.launches,1)

    def test_host_streams_exact_sealed_pocket_tree_and_rejects_mutation(self):
        from service import host_install as h
        import time
        models=self.base.root/'installed-models';bundle=models/'bundles'/'bundle';bundle.mkdir(parents=True)
        for source in self.base.bundle.iterdir():
            target=bundle/source.name;target.write_bytes(source.read_bytes());target.chmod(0o444)
        for target in (bundle,bundle.parent,models):target.chmod(0o555)
        self.addCleanup(lambda:[p.chmod(0o755) for p in (models,bundle.parent,bundle)])
        f=self.gate_fixture.fixture.fixture;previous=f.translate
        def owner(info,path=None):
            if path is not None and (path==models or models in path.parents):
                from types import SimpleNamespace
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=10001,st_gid=10001))
            return previous(info,path)
        f.translate=owner
        reads=[];real=os.read
        def read(fd,size):reads.append(size);return real(fd,size)
        with patch.object(os,'read',side_effect=read):
            result=h._model_tree(str(models),self.config,time.monotonic()+5)
        self.assertEqual(len(result),64);self.assertTrue(reads);self.assertLessEqual(max(reads),1024**2)
        target=bundle/'model.safetensors';target.chmod(0o644);target.write_bytes(b'changed');target.chmod(0o444)
        with self.assertRaisesRegex(ValueError,'model_manifest'):h._model_tree(str(models),self.config,time.monotonic()+5)



class DiscoveryTests(unittest.TestCase):
    def test_installed_activation_does_not_promote_pocket_qualification(self):
        import asyncio
        import httpx
        from types import SimpleNamespace
        from service.api import create_voice_app
        from service.auth import Credential,StaticCredentials
        fixture=PreparationFixture();fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=configuration(fixture.context,fixture.profiles)
        config['profiles'][0].update(support='unqualified',readiness='not_ready',verification='synthetic',qualification_receipt='')
        class Dispatcher:
            installed_checked=True
            installed_activation=SimpleNamespace(configuration=config)
            def call(inner,operation,*args):
                if operation=='configured_capabilities':
                    return dict(engine='pocket-tts',profile=config['profiles'][0],operations=['voice_prepare','voice_results','voice_save','speech_generate'],next_action='upload → prepare → save')
                if operation=='configured_profiles':return copy.deepcopy(config['profiles'])
                if operation=='configured_presets':return []
                if operation=='audio_export_available':return False
                if operation=='status':return dict(persistent_activation=True,retry_available=True)
                raise AssertionError(operation)
        credentials=StaticCredentials([Credential('read','alice',frozenset({'read'}))])
        app=create_voice_app(Dispatcher(),credentials=credentials,installed=config)
        async def read():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers={'Authorization':'Bearer read'}) as client:
                return ((await client.get('/skeleton')).json(), (await client.get('/v1/capabilities')).json(),
                        {op: (await client.get('/skeleton/operations/'+op)).json()
                         for op in ('voice_prepare','voice_results','voice_save','voice_preset_register')})
        status,caps,details=asyncio.run(read())
        self.assertTrue(status['installed']);self.assertTrue(status['activation_ready']);self.assertFalse(status['ready_for_speech'])
        self.assertTrue(status['clone_prepare']);self.assertEqual(status['configured_presets'],[])
        self.assertIn('voice_prepare',caps['operations']);self.assertNotIn('voice_preset_register',caps['operations'])
        self.assertEqual(caps['profiles'][0]['readiness'],'not_ready')
        for op in ('voice_prepare','voice_results','voice_save'):
            self.assertIn(op,caps['operations'])
            self.assertEqual(details[op]['availability'], 'implemented_read' if op=='voice_results' else 'implemented')
        self.assertEqual(details['voice_preset_register']['availability'],'unavailable_configured_profile')
