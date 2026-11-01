"""INJECTED Linux/Docker/UID observations; real temp files, no host qualification."""
import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from service import host_install as h, install_policy as ip, recovery
from service.core import encode
from service.persistent_init import digest
from service.admission import DEFAULT_BUDGET
from tests.service.test_engine_admission import configuration
from tests.service import test_provisioning as provisioning


class InstalledHost(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=getattr(self,'temporary_parent',None)); self.addCleanup(self.cleanup)
        self.root=Path(self.temp.name).resolve(); self.owners_map={}
        fixture=provisioning.ProvisioningTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        context,profiles=configuration(fixture.expected); spec=context['execution']['descriptor']; bid=spec['bundle_id']
        self.config=dict(version=1,ownership_policy=ip.POLICY,profiles=profiles,budget=copy.deepcopy(DEFAULT_BUDGET),
            contexts={'alice':context},bundles={bid:dict(manifest_sha256=spec['manifest_sha256'],checkpoint=spec['checkpoint'],revision=spec['revision'],path='/models/bundles/'+bid)},
            images=dict(control='sha256:'+'1'*64,worker='sha256:'+'2'*64),
            mounts=dict(control='/control',staging='/staging',receipts='/receipts',ipc='/ipc',models='/models'))
        self.scope=dict(daemon='injected-linux',context='injected',volumes={r:dict(name=r,driver='local',created='now') for r in ip.ROLES})
        self.policy=dict(version=1,kind=ip.HOST_POLICY,configuration_digest=digest(self.config),host=self.scope,
                         roots={},inputs={},cpu=dict(control=10**9,worker=spec['threads']*10**9),
                         tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},control_port=8765)
        for role in ip.ROLES:
            p=self.root/role; p.mkdir(mode=0o700)
            uid,gid,mode=ip.expectation(ip.SPLIT,role,directory=True);p.chmod(mode);self.owner(p,uid,gid);s=p.stat()
            self.policy['roots'][role]=dict(source=str(p),device=s.st_dev,inode=s.st_ino)
        models=self.root/'models'; bundle=models/'bundles'/bid; bundle.mkdir(parents=True)
        for n,b in fixture.data.items(): (bundle/n).write_bytes(b)
        (bundle/'manifest.json').write_text(encode(fixture.expected))
        # Same canonical manifest bytes as the enrolled fixture.
        from service.provisioning import manifest_bytes
        (bundle/'manifest.json').write_bytes(manifest_bytes(fixture.expected))
        for p in [models,*models.rglob('*')]:
            p.chmod(0o555 if p.is_dir() else 0o444); self.owner(p,10001,10001)
        config=self.root/'install.json'; config.write_text(encode(self.config)); config.chmod(0o440);self.owner(config,0,10001)
        credential=self.root/'control.key'; credential.write_bytes(b'NEVER-RETURN-THIS-CREDENTIAL');credential.chmod(0o640);self.owner(credential,0,10001)
        for name,p in [('models',models),('config',config),('credential',credential)]:
            s=p.stat();uid,gid=self.owners_map[s.st_dev,s.st_ino]
            record=dict(source=str(p),device=s.st_dev,inode=s.st_ino,uid=uid,gid=gid,mode=stat.S_IMODE(s.st_mode),size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
            if name!='credential':record['sha256']='0'*64
            self.policy['inputs'][name]=record
        self.real_stat=os.stat; self.real_fstat=os.fstat
        self.enterContext(patch.object(os,'stat',side_effect=lambda *a,**k:self.virtual(self.real_stat(*a,**k))))
        self.enterContext(patch.object(os,'fstat',side_effect=lambda fd:self.virtual(self.real_fstat(fd))))
        self.policy['inputs']['models']['sha256']=h._model_tree(str(models),self.config,time.monotonic()+5)
        self.policy['inputs']['config']['sha256']=hashlib.sha256(config.read_bytes()).hexdigest()
        device=self.policy['roots']['control']['device']
        self.mounts=[dict(id='1',device=str(os.major(device))+':'+str(os.minor(device)),root='/',target='/',options=['rw'])]
        self.volumes=[dict(role=r,**self.scope['volumes'][r],**v) for r,v in self.policy['roots'].items()]
        self.owners=[]; self.containers=[]
        for i,role in enumerate(('control','worker')):
            owner=dict(id=str(i+1)*64,image=self.config['images'][role],started='2026-10-24T01:00:00Z',role=role,
                       volumes=['control','ipc','staging'] if role=='control' else ['ipc','receipts','staging'])
            self.owners.append(owner)
            tmp=self.policy['tmpfs'][role]
            c=dict(id=owner['id'],image=owner['image'],started=owner['started'],finished='0001-01-01T00:00:00Z',running=True,restarting=False,pid=123+i,
                restart='no',pidmode='',privileged=False,capadd=[] if role=='control' else ['SETUID','SETGID','KILL'],cgroupns='private',
                user='10001:10001' if role=='control' else '0:10001',capdrop=['ALL'],security=['no-new-privileges:true'],readonly=True,
                network='bridge' if role=='control' else 'none',groups=[],devices=[],device_requests=[],device_rules=[],userns='',ipcns='private',utsns='',
                memory=self.config['budget']['ram_bytes'],swap=self.config['budget']['ram_bytes'],pids=self.config['budget']['pids'],nanocpus=self.policy['cpu'][role],period=0,quota=0,cpuset='',
                tmpfs={'/tmp':'rw,noexec,nosuid,nodev,size='+str(tmp['bytes'])+',nr_inodes='+str(tmp['inodes'])+',mode=1777'},
                ports={'8000/tcp':[dict(HostIp='127.0.0.1',HostPort='8765')]} if role=='control' else {},publish_all=False,mounts=[])
            for r in owner['volumes']:
                c['mounts'].append(dict(type='volume',name=r,source=self.policy['roots'][r]['source'],destination='/'+r,rw=role!='control' or r=='control'))
            for name,destination in [('models','/models'),('config',ip.CONFIG_DESTINATION)]+([('credential',ip.CREDENTIAL_DESTINATION)] if role=='control' else []):
                c['mounts'].append(dict(type='bind',name='',source=self.policy['inputs'][name]['source'],destination=destination,rw=False))
            self.containers.append(c)

    def cleanup(self):
        for p in self.root.rglob('*'):
            if p.is_dir() and not p.is_symlink():p.chmod(0o755)
        self.temp.cleanup()

    def owner(self,p,uid,gid):
        s=p.stat();self.owners_map[s.st_dev,s.st_ino]=(uid,gid)

    def virtual(self,s):
        owner=self.owners_map.get((s.st_dev,s.st_ino))
        return s if owner is None else SimpleNamespace(**{k:getattr(s,k) for k in dir(s) if k.startswith('st_') and k not in ('st_uid','st_gid')},st_uid=owner[0],st_gid=owner[1])

    def enrollment(self):
        return dict(deployment_id='a'*32,revision=1,scope=ip.SCOPE,host=self.scope,configuration=self.config,
                    configuration_digest=digest(self.config),ownership_policy=ip.POLICY,
                    roles={r:dict(device=v['device'],inode=v['inode'],uid=ip.expectation(ip.SPLIT,r,directory=True)[0],
                                  gid=ip.expectation(ip.SPLIT,r,directory=True)[1],mode=ip.expectation(ip.SPLIT,r,directory=True)[2]) for r,v in self.policy['roots'].items()})

    def topology(self):
        return dict(scope=self.scope,volumes=self.volumes,consumers=self.containers,policy=ip.HOST_POLICY,configuration_digest=digest(self.config),host_policy_digest=digest(self.policy))

    def validate(self):
        return recovery.validate_host_topology(self.topology(),self.scope,self.owners,stopped=False,installed_policy=self.policy,configuration=self.config)

    def test_injected_positive_exact_policy_and_real_sources(self):
        ip.host_policy(self.policy,self.config,self.scope); self.validate()
        h._installed_mount_proof(self.policy,self.mounts)
        result=h._installed_sources(self.policy,self.config)
        self.assertEqual(len(result),7)
        self.assertNotIn('NEVER-RETURN',encode(self.topology()))
        with self.assertRaises(ValueError): recovery.validate_host_topology(self.topology(),self.scope,self.owners,stopped=False)

    def test_closed_policy_configuration_cpu_tmpfs(self):
        original=copy.deepcopy(self.policy)
        changes=[lambda p:p.update(extra=True),lambda p:p.pop('cpu'),lambda p:p['cpu'].update(worker=1),
                 lambda p:p['cpu'].update(control=True),lambda p:p['tmpfs']['worker'].update(bytes=0),
                 lambda p:p['tmpfs']['worker'].update(inodes=100000),lambda p:p.update(configuration_digest='a'*64),
                 lambda p:p['inputs']['credential'].update(sha256='a'*64)]
        for change in changes:
            self.policy=copy.deepcopy(original);change(self.policy)
            with self.subTest(policy=self.policy),self.assertRaises(ValueError):ip.host_policy(self.policy,self.config,self.scope)

    def test_container_negative_matrix(self):
        for field,value in [('user','root'),('capadd',['SYS_ADMIN']),('capdrop',[]),('privileged',True),('groups',['10001']),
                ('devices',[{}]),('device_requests',[{}]),('device_rules',['a']),('pidmode','host'),('ipcns','host'),('utsns','host'),
                ('cgroupns','host'),('userns','host'),('security',[]),('readonly',False),('network','host'),('restart','always'),
                ('memory',0),('swap',-1),('pids',0),('nanocpus',1),('period',100),('quota',100),('cpuset','0'),
                ('tmpfs',{'/tmp':'size=0'}),('publish_all',True),('ports',{}),('image','sha256:'+'a'*64)]:
            with self.subTest(field=field):
                original=self.containers[0][field];self.containers[0][field]=value
                with self.assertRaises(ValueError):self.validate()
                self.containers[0][field]=original

    def test_mount_negative_matrix(self):
        original=copy.deepcopy(self.containers)
        changes=[lambda cs:cs[1]['mounts'].append(copy.deepcopy(cs[0]['mounts'][-1])),
                 lambda cs:cs[0]['mounts'][-1].update(rw=True),lambda cs:cs[0]['mounts'][-1].update(source='/other'),
                 lambda cs:cs[0]['mounts'].pop(),lambda cs:cs[0]['mounts'][0].update(type='bind'),
                 lambda cs:cs[0]['mounts'][0].update(source=self.policy['roots']['ipc']['source'])]
        for change in changes:
            self.containers=copy.deepcopy(original);change(self.containers)
            with self.assertRaises(ValueError):self.validate()

    def test_unknown_missing_or_changed_owner(self):
        for owners in ([],self.owners[:1],[dict(self.owners[0],id='f'*64),self.owners[1]]):
            with self.assertRaises(ValueError):h.validate_installed_topology(self.topology(),self.scope,owners,self.config,self.policy)

    def test_mountinfo_alias_nested_remote_device_refuse(self):
        for extra in [dict(self.mounts[0],id='2',target='/alias',root='/'),
                      dict(self.mounts[0],id='2',target=self.policy['inputs']['models']['source']+'/nested'),
                      dict(self.mounts[0],id='2',target=str(self.root),root='/subtree')]:
            with self.assertRaises(ValueError):h._installed_mount_proof(self.policy,self.mounts+[extra])
        with self.assertRaises(ValueError):h._installed_mount_proof(self.policy,[dict(self.mounts[0],device='1:99')])

    def test_secret_metadata_change_and_no_secret_read(self):
        real_read=os.read; secret=self.policy['inputs']['credential']
        def read(fd,n):
            s=os.fstat(fd)
            if (s.st_dev,s.st_ino)==(secret['device'],secret['inode']):raise AssertionError('credential contents read')
            return real_read(fd,n)
        with patch.object(os,'read',side_effect=read):h._installed_sources(self.policy,self.config)
        Path(secret['source']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'installed_input_changed'):h._installed_sources(self.policy,self.config)

    def test_config_bytes_and_model_bytes_changes_refuse(self):
        config=Path(self.policy['inputs']['config']['source']);config.chmod(0o640);config.write_text('{}');config.chmod(0o440)
        with self.assertRaises(ValueError):h._installed_sources(self.policy,self.config)
        model=next((self.root/'models').rglob('model.onnx'));model.chmod(0o644);model.write_bytes(b'wrong');model.chmod(0o444)
        with self.assertRaises(ValueError):h._model_tree(str(self.root/'models'),self.config,time.monotonic()+5)

    def test_symlink_hardlink_wrong_mode_and_overlap_refuse(self):
        config=Path(self.policy['inputs']['config']['source']);alias=self.root/'alias';alias.symlink_to(config)
        with self.assertRaises(OSError):h._public_file(alias,uid=0,gid=10001,mode=0o440,limit=65536,deadline=time.monotonic()+2)
        os.link(config,self.root/'hardlink')
        with self.assertRaises(ValueError):h._installed_sources(self.policy,self.config)
        (self.root/'hardlink').unlink();config.chmod(0o444)
        with self.assertRaises(ValueError):h._installed_sources(self.policy,self.config)
        config.chmod(0o440)
        s=config.stat()
        for k in ('size','mtime_ns','ctime_ns'):self.policy['inputs']['config'][k]=getattr(s,'st_'+k)
        self.policy['roots']['ipc']=dict(self.policy['roots']['control'])
        with self.assertRaisesRegex(ValueError,'overlap'):h._installed_sources(self.policy,self.config)

    def observer(self,containers=None,security=None):
        test=self
        class Docker:
            context='injected'
            def _read(self,args):
                if args[0]=='context':return json.dumps('unix:///var/run/docker.sock')
                if args[0]=='info':
                    if 'SecurityOptions' in args[-1]:return encode(security or dict(security=['name=seccomp'],cgroup='2',name='linux-host',os='Linux'))
                    return encode(dict(id=test.scope['daemon'],os='linux'))
                if args[:2]==['volume','inspect']:
                    r=args[2];return encode(dict(test.scope['volumes'][r],source=test.policy['roots'][r]['source'],options={}))
                if args[:2]==['container','ls']:return '\n'.join(c['id'] for c in (test.containers if containers is None else containers))
                if args[:2]==['container','inspect']:
                    current=next((c for c in (test.containers if containers is None else containers) if c['id']==args[2]),None)
                    if current is None:raise ValueError('injected_missing_container')
                    return encode(current)
                raise AssertionError(args)
        return h.InstalledObserver(Docker(),self.config,self.scope,self.policy,mountinfo=lambda:copy.deepcopy(self.mounts),proc=lambda pid,identity,ipc:dict(pid=pid,start_ticks=100,pid_namespace=pid,cgroup='/docker/'+identity))

    def test_real_adapters_injected_inventory_proc_and_init_callback(self):
        with patch.object(h.sys,'platform','linux'):
            observed=self.observer().observe_owners(self.owners,self.enrollment())
            self.assertEqual(len(observed['processes']),2)
            self.assertNotIn('actual-host-inspection',encode(observed))
            result=self.observer([]).initialize_authority()
            self.assertEqual(result,dict(host=self.scope,roots={r:[v['device'],v['inode']] for r,v in self.policy['roots'].items()},consumers=[]))
            with self.assertRaises(ValueError):self.observer().initialize_authority()

    def test_rootless_desktop_remote_daemon_and_unrelated_storage(self):
        with patch.object(h.sys,'platform','linux'):
            for security in [dict(security=['name=rootless'],cgroup='2',name='x',os='Linux'),dict(security=['name=userns'],cgroup='2',name='x',os='Linux'),dict(security=[],cgroup='1',name='x',os='Linux'),dict(security=[],cgroup='2',name='docker-desktop',os='Linux')]:
                with self.assertRaises(ValueError):self.observer(security=security).inventory(self.owners)
            unrelated=self.root/'unrelated';unrelated.mkdir()
            self.containers[0]['mounts'].append(dict(type='bind',name='',source=str(unrelated),destination='/unrelated',rw=False))
            with self.assertRaisesRegex(ValueError,'unsupported_host_storage_disjointness'):self.observer().inventory(self.owners)

    def test_inventory_and_process_races(self):
        with patch.object(h.sys,'platform','linux'):
            observer=self.observer(); original=observer.docker._read; n=0
            def read(args):
                nonlocal n
                raw=original(args)
                if args[:2]==['container','ls']:
                    n+=1
                    if n==2:return ''
                return raw
            observer.docker._read=read
            with self.assertRaisesRegex(ValueError,'observation_changed'):observer.inventory(self.owners)
            observer=self.observer(); n=0
            def proc(pid,identity,ipc):
                nonlocal n
                n+=1;return dict(pid=pid,start_ticks=n,pid_namespace=pid,cgroup='/docker/'+identity)
            observer.proc=proc
            with self.assertRaisesRegex(ValueError,'bracket_changed'):observer.observe_owners(self.owners,self.enrollment())

    def authority_fixture(self):
        base=self.root/'authority'; directory=base/'installed'/digest(self.scope)
        directory.mkdir(parents=True,mode=0o700)
        lock=directory/'launch.lock';lock.write_bytes(b'');lock.chmod(0o600)
        policy=directory/'host-policy.json';policy.write_text(encode(self.policy));policy.chmod(0o600)
        self.owner(lock,0,0);self.owner(policy,0,0)
        for p in (directory,*directory.parents):self.owner(p,0,0)
        real=h._root_authority_directory; fstat=os.fstat
        ancestors={(p.stat().st_dev,p.stat().st_ino) for p in directory.parents}
        def trusted(path):
            def observation(fd):
                s=fstat(fd)
                if (s.st_dev,s.st_ino) in ancestors:
                    return SimpleNamespace(**{k:getattr(s,k) for k in dir(s) if k.startswith('st_') and k!='st_mode'},st_mode=s.st_mode & ~0o022)
                return s
            # Inject trust for temp fixture ancestors only; production validator runs.
            with patch.object(os,'fstat',side_effect=observation):return real(path)
        stack=self.enterContext(ExitStack())
        stack.enter_context(patch.object(h.recovery,'HOST_AUTHORITY_DIRECTORY',str(base)))
        stack.enter_context(patch.object(h.sys,'platform','linux'))
        stack.enter_context(patch.multiple(h.os,getuid=lambda:0,geteuid=lambda:0))
        stack.enter_context(patch.object(h,'_root_authority_directory',side_effect=trusted))
        return directory

    def test_protected_authority_exclusion_and_callback_lifetime(self):
        self.authority_fixture()
        expected=dict(host=self.scope,roots={},consumers=[])
        fake=SimpleNamespace(initialize_authority=lambda:expected)
        with patch.object(h,'InstalledObserver',return_value=fake):
            with h.installed_launch_authority(self.config,self.scope) as authority:
                self.assertEqual(authority.initialize_authority(),expected)
                with self.assertRaisesRegex(ValueError,'installed_launch_busy'):
                    with h.installed_launch_authority(self.config,self.scope):pass
            with self.assertRaisesRegex(ValueError,'authority_closed'):authority.initialize_authority()

    def test_protected_policy_exchange_refuses_without_any_docker(self):
        directory=self.authority_fixture()
        with self.assertRaisesRegex(ValueError,'policy_changed'):
            with h.installed_launch_authority(self.config,self.scope) as authority:
                path=directory/'host-policy.json'; path.rename(directory/'retained.json')
                path.write_text(encode(self.policy));path.chmod(0o600);self.owner(path,0,0)
                authority.initialize_authority()

    def test_protected_lock_mode_and_root_requirement(self):
        directory=self.authority_fixture();(directory/'launch.lock').chmod(0o644)
        with self.assertRaisesRegex(ValueError,'lock_changed'):
            with h.installed_launch_authority(self.config,self.scope):pass
        with patch.object(h.os,'geteuid',return_value=10001):
            with self.assertRaisesRegex(ValueError,'root_authority_required'):
                with h.installed_launch_authority(self.config,self.scope):pass

    def test_process_content_cannot_smuggle_provenance_or_credentials(self):
        with patch.object(h.sys,'platform','linux'):
            observer=self.observer()
            observer.proc=lambda *args:dict(pid=123,start_ticks=1,pid_namespace=2,cgroup='/bad',credential='NEVER-RETURN')
            with self.assertRaises(ValueError):observer.observe_owners(self.owners,self.enrollment())


    def test_observer_callback_supplies_actual_split_initializer_injected_child(self):
        from service import persistent_init
        from tests.service import test_split_init as split_tests
        fixture=split_tests.SplitInit();fixture.temporary_parent=self.root.parent
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.scope=self.scope;fixture.config=self.config
        self.policy['roots']={r:dict(source=str(p),device=p.stat().st_dev,inode=p.stat().st_ino) for r,p in fixture.roots.items()}
        self.volumes=[dict(role=r,**self.scope['volumes'][r],**v) for r,v in self.policy['roots'].items()]
        with patch.object(h.sys,'platform','linux'):
            observer=self.observer([])
            result=persistent_init.initialize_split(fixture.roots,self.scope,self.config,
                authority=observer.initialize_authority,policy=ip.SPLIT,launch_child=fixture.child)
        self.assertEqual(len(result['deployment_id']),32)
        self.assertTrue((fixture.roots['receipts']/'p1-enrolled.json').is_file())

    def test_remote_endpoint_and_changed_daemon_refuse(self):
        with patch.object(h.sys,'platform','linux'):
            for endpoint in ('ssh://remote','unix:///run/user/1000/docker.sock'):
                observer=self.observer();read=observer.docker._read
                observer.docker._read=lambda args:json.dumps(endpoint) if args[0]=='context' else read(args)
                with self.assertRaisesRegex(ValueError,'unsupported_host_pid_provenance'):observer.inventory(self.owners)
            observer=self.observer();read=observer.docker._read
            observer.docker._read=lambda args:encode(dict(id='different',os='linux')) if args[0]=='info' and 'SecurityOptions' not in args[-1] else read(args)
            with self.assertRaisesRegex(ValueError,'daemon_mismatch'):observer.inventory(self.owners)

    def test_source_read_exchange_and_ancestor_symlink(self):
        path=Path(self.policy['inputs']['config']['source']);replacement=self.root/'replacement'
        replacement.write_bytes(path.read_bytes());replacement.chmod(0o440);self.owner(replacement,0,10001)
        read=os.read;exchanged=False
        def mutate(fd,n):
            nonlocal exchanged
            raw=read(fd,n)
            s=os.fstat(fd)
            if not exchanged and s.st_ino==path.stat().st_ino:
                exchanged=True;path.rename(self.root/'retained');replacement.rename(path)
            return raw
        with patch.object(os,'read',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'source_changed'):h._installed_sources(self.policy,self.config)
        alias=self.root/'alias';alias.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(OSError):h._source_stat(alias/'install.json',directory=False)

    def test_closed_expected_enrollment_configuration_binding(self):
        for change in (lambda p:p.update(unknown=True),lambda p:p.update(configuration_digest='0'*64),
                       lambda p:p['roles']['ipc'].update(inode=1),lambda p:p.update(scope='owned-synthetic')):
            public=copy.deepcopy(self.enrollment());change(public)
            with self.assertRaises(ValueError):self.observer().observe_owners(self.owners,public)


if __name__=='__main__':unittest.main()


class ModelEnumerationBounds(unittest.TestCase):
    """Lazy iterators fail loudly if collection passes the expected stop point."""
    def collect(self, *, root_names=None, deadline=10, clock=lambda: 0,
                guard=4096):
        from contextlib import nullcontext
        consumed={}; opened=[]
        directory=SimpleNamespace(st_mode=stat.S_IFDIR|0o555,st_uid=10001,st_gid=10001)
        class Entries:
            def __init__(self, path): self.path=path; self.index=0
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def __iter__(self): return self
            def __next__(self):
                if self.path == Path('/models') and root_names is not None:
                    if self.index == len(root_names): raise StopIteration
                    name=root_names[self.index]
                else: name='entry'+str(self.index)
                self.index+=1; consumed[self.path]=self.index
                if self.index>guard: raise AssertionError('iterator consumed beyond failguard')
                return SimpleNamespace(name=name)
        def scan(path): opened.append(path); return Entries(path)
        with patch.object(h,'_source_stat',return_value=directory), \
             patch.object(ip,'identity',return_value=(1,2)), \
             patch.object(ip,'parent_fd',side_effect=lambda path:nullcontext((0,path.name))), \
             patch.object(h.os,'stat',return_value=directory), \
             patch.object(h.os,'scandir',side_effect=scan), \
             patch.object(h.time,'monotonic',side_effect=clock):
            with self.assertRaisesRegex(ValueError,'^installed_source_budget$'):
                h._model_tree(Path('/models'),{'ownership_policy':ip.POLICY},deadline)
        return consumed,opened

    def test_lazy_oversized_stops_at_global_entry_bound(self):
        consumed,opened=self.collect()
        self.assertEqual(consumed,{Path('/models'):4096})
        self.assertEqual(opened,[Path('/models')])

    def test_already_expired_never_opens_or_consumes(self):
        consumed,opened=self.collect(deadline=-1,guard=0)
        self.assertEqual((consumed,opened),({},[]))

    def test_nested_scan_uses_remaining_global_budget(self):
        consumed,opened=self.collect(root_names=['a','b'],guard=4094)
        self.assertEqual(consumed,{Path('/models'):2,Path('/models/b'):4094})
        self.assertEqual(opened,[Path('/models'),Path('/models/b')])

    def test_deadline_expiring_during_scan_stops_early(self):
        calls=iter([0,0,0,11])
        consumed,opened=self.collect(clock=lambda:next(calls,11),guard=1)
        self.assertEqual(consumed,{Path('/models'):1})
        self.assertEqual(opened,[Path('/models')])
