"""SOURCE fixtures only: no Docker, child, UID transition, listener or ML.

Real files/Core/SQLite; Docker inspection, mountinfo and identity enforcement
are explicitly adapted. These receipts cannot qualify Desktop/OrbStack/Linux.
"""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from service import daemon_volumes as dv, daemon_helper as dh, daemon_client as dc
from service import install_child as ic, persistent_init as pi, install_policy as ip, entrypoint as ep, recovery
from service.core import encode
from tests.service.test_split_init import SplitInit
from service.provisioning import manifest_bytes


def binding(config):
    installation='a'*32
    return dict(policy=dv.POLICY,installation=installation,daemon=dict(endpoint='unix:///Users/operator/.docker/run/docker.sock',
        id='fixture-desktop-linux',os='linux',architecture='amd64',security=['name=seccomp,profile=builtin'],cgroup='2'),
        volumes={r:dict(Name='pab-'+installation+'-'+r,Driver='local',Options=None,Scope='local',
            Mountpoint='/opaque-daemon/volumes/'+r,CreatedAt='2026-10-25T00:00:00Z',
            Labels={'speech-box.installation':installation,'speech-box.role':r}) for r in dv.ROLES},
        helper_image=config['images']['control'],configuration_digest=dv.digest(config),cpu=dict(control=10**9,
            worker=max(c['execution']['descriptor']['threads'] for c in config['contexts'].values())*10**9),
        tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},control_port=8765)


def container(role,config,b,*,phase=None,nonce=None,identity=None):
    args=dv.command(role,config,b,phase=phase,nonce=nonce)
    helper=role=='helper'; t=b['tmpfs']['control' if helper else role]
    return dict(Id=identity or ('1' if role=='control' else '2')*64,Image=b['helper_image'] if helper else config['images'][role],
        Config=dict(User='0:0' if helper else '10001:10001' if role=='control' else '0:10001',Entrypoint=['python'],
            Cmd=args[args.index('--entrypoint')+3:]),
        HostConfig=dict(Privileged=False,ReadonlyRootfs=True,Init=None,PidMode='',IpcMode='private',CgroupnsMode='private',UsernsMode='',
            NetworkMode='bridge' if role=='control' else 'none',CapDrop=['ALL'],
            CapAdd=list(dv.INIT_CAPS if phase in ('install','init') else dv.LATER_CAPS) if helper else [] if role=='control' else list(dv.WORKER_CAPS),
            SecurityOpt=['no-new-privileges:true'],RestartPolicy=dict(Name='no',MaximumRetryCount=0),Devices=[],DeviceRequests=None,Binds=None,
            Memory=config['budget']['ram_bytes'],MemorySwap=config['budget']['ram_bytes'],PidsLimit=config['budget']['pids'],
            Mounts=[dict(Type='volume',Source=m['Name'],Target=m['Destination'],ReadOnly=not m['RW'],VolumeOptions=dict(NoCopy=True)) for m in dv.mounts(role,b,phase)],
            NanoCpus=b['cpu']['control' if helper else role],Tmpfs={'/tmp':'rw,noexec,nosuid,nodev,size='+str(t['bytes'])+',nr_inodes='+str(t['inodes'])+',mode=1777'},
            PortBindings={'8000/tcp':[dict(HostIp='127.0.0.1',HostPort=str(b['control_port']))]} if role=='control' else {}),
        State=dict(Running=False,Restarting=False,ExitCode=0,Status='created',StartedAt='0001-01-01T00:00:00Z',Pid=0),RestartCount=0,
        Mounts=dv.mounts(role,b,phase))


class Session:
    def __init__(self,test):
        self.test=test; self.f=SplitInit(); self.f.setUp(); test.addCleanup(self.f.doCleanups)
        self.root=self.f.root; self.config=self.f.config; self.b=binding(self.config); self.scope=dv.scope(self.b)
        self.roots=dict(self.f.roots); self.cs={}; self.n=0; self.calls=[]; self.marker=dict(phase='preparing'); self.gate=None
        self.boot='b'*32; self.starts=0; self.f.scope=self.scope
        for role in ('authority','models','config','credential'):
            path=self.root/role; path.mkdir(mode=0o755); self.roots[role]=path
            st=path.stat(); self.f.virtual[st.st_dev,st.st_ino]=(0,0)
        translate=self.f.translate
        def owner(info,path=None):
            if path is not None and (path==self.roots['authority'] or self.roots['authority'] in path.parents):
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=0,st_gid=0))
            return translate(info,path)
        self.f.translate=owner
        # Already supplied tiny model bytes; no acquisition/imports of ML modules.
        from tests.service.test_provisioning import ProvisioningTests
        model=ProvisioningTests(); model.setUp(); test.addCleanup(model.doCleanups)
        bid=next(iter(self.config['bundles']))
        data=[('models','bundles/'+bid+'/'+name,raw) for name,raw in model.data.items()]
        data += [('models','bundles/'+bid+'/manifest.json',manifest_bytes(model.expected)),('config','install.json',encode(self.config).encode()),
            ('credential','control.key',encode(dict(version=1,credentials=[dict(token='secret-fixture-only-'+'x'*32,principal='alice',scopes=['read','production-write'])])).encode())]
        self.inputs=[]; self.bytes=[]
        for role,path,raw in data:
            entry=dict(role=role,path=path,bytes=len(raw))
            if role!='credential': entry['sha256']=hashlib.sha256(raw).hexdigest()
            self.inputs.append(entry); self.bytes.append(raw)
        self.fail_action=None; self.mutate=None
        def writable():
            for p in self.root.rglob('*'):
                if p.is_dir() and not p.is_symlink(): p.chmod(0o755)
        test.addCleanup(writable)

    def facts(self,role):
        names=('control','ipc','staging','models','config','credential') if role=='control' else ('receipts','ipc','staging','models','config')
        return dict(role=role,pid=1,start_ticks=100+self.starts,pid_namespace=200+(role=='worker'),cgroup='0::/',
            uid=10001 if role=='control' else 0,gid=10001,groups=[],capabilities=0 if role=='control' else 224,
            roots={r:[self.roots[r].stat().st_dev,self.roots[r].stat().st_ino] for r in names})

    def startup(self):
        journal=json.loads((self.roots['receipts']/'journal.json').read_bytes())['payload']
        return dict(hello=dict(lane=journal['lane'],boot=self.boot,quarantined=False,launches=0),local=self.facts('worker'))

    def action(self,action,payload):
        self.calls.append((action,copy.deepcopy(payload)))
        if self.fail_action==action: raise OSError('injected '+action)
        if action=='snapshot':
            return dict(daemon=copy.deepcopy(self.b['daemon']),volumes=copy.deepcopy(self.b['volumes']),consumers=copy.deepcopy(list(self.cs.values())))
        if action=='input':
            raw=self.bytes[payload['index']][payload['offset']:payload['offset']+49152]
            return dict(data=base64.b64encode(raw).decode())
        if action=='create':
            role=payload['role']; c=container(role,self.config,self.b); self.cs[c['Id']]=c; return copy.deepcopy(c)
        if action=='notify':
            self.gate=copy.deepcopy(payload['gate']); self.marker=dict(phase='ready',registration_id=self.gate['registration_id']); return {}
        identity=payload['id']; role=payload['role']; c=self.cs[identity]
        if action=='start':
            if role=='control': self.starts+=1; self.marker=dict(phase='waiting')
            else: self.boot=hex(self.starts+10)[2:]*32
            c['State'].update(Running=True,Status='running',StartedAt='2026-10-25T00:00:0'+str(self.starts)+'Z')
        elif action=='term': c['State'].update(Running=False,Status='exited')
        elif action=='marker': return copy.deepcopy(self.marker)
        elif action=='facts': return self.facts('control')
        elif action!='inspect': raise AssertionError(action)
        if self.mutate: self.mutate(c)
        return copy.deepcopy(c)

    def child(self,kind,request,fds):
        self.test.assertNotIn('seed',request)
        if kind=='hello': return self.startup()
        with patch.object(ic,'CONTAINER_CONTROL',self.roots['control']),patch.object(ic,'CONTAINER_STAGING',self.roots['staging']):
            control,staging=ic.prepare_container(request['public'],request['control_fd'],request['staging_fd'])
        with patch.object(ic.LinuxMounts,'unshare',side_effect=AssertionError('unshare')),patch.object(ic.LinuxMounts,'mount',side_effect=AssertionError('mount')):
            with patch.multiple(os,getuid=lambda:10001,geteuid=lambda:10001,getgid=lambda:10001,getegid=lambda:10001,getgroups=lambda:[]):
                if kind=='core':
                    self.f.calls.append(request); return pi._split_core(request['public'],control,staging)
                return ic.registration_writer(request['public'],request['registration'],request['prestate'],control,request['lock_fd'])

    def run(self,phase):
        self.n+=1; nonce=format(self.n,'032x'); identity=format(self.n+10,'064x')
        c=container('helper',self.config,self.b,phase=phase,nonce=nonce,identity=identity)
        c['State'].update(Running=True,Status='running',StartedAt='2026-10-25T00:00:00Z'); self.cs[identity]=c
        outer=self
        class Wire:
            def send(self,action,payload): self.pending=(action,payload)
            def receive(self):
                action,payload=self.pending; return action,outer.action(action,payload)
        roots={r:self.roots[r] for r in dv.HELPER_ROLES[phase]}
        h=dh.Helper(phase,nonce,dict(configuration=self.config,binding=self.b,helper=copy.deepcopy(c),manifest=self.inputs if phase=='install' else []),Wire(),roots=roots,child_runner=self.child)
        self.helper=h
        old=os.umask(0o077)
        try:
            with patch.object(h,'mounts_checked'): result=h.run()
        finally: os.umask(old)
        c['State'].update(Running=False,Status='exited')
        return result

    def install(self): self.run('install'); self.run('init'); return self.run('up')


class DaemonTests(unittest.TestCase):
    def setUp(self): self.s=Session(self)

    def test_install_up_status_stop_same_id_up(self):
        s=self.s; first=s.install(); ids=first['containers']; immutable=(s.roots['authority']/('registration-'+first['registration_id']+'.json')).read_bytes()
        self.assertEqual(len(s.f.calls),1)
        self.assertEqual(s.run('status')['phase'],'ready')
        self.assertEqual(s.run('stop')['phase'],'stopped')
        self.assertEqual(s.run('status')['phase'],'stopped')
        second=s.run('up'); self.assertEqual(second['containers'],ids); self.assertNotEqual(first['registration_id'],second['registration_id'])
        self.assertEqual((s.roots['authority']/('registration-'+first['registration_id']+'.json')).read_bytes(),immutable)
        self.assertEqual(len(s.f.calls),1)
        self.assertEqual([p['role'] for a,p in s.calls if a=='term'],['control','worker'])
        self.assertEqual([a for a,_ in s.calls].count('create'),2)
        self.assertNotIn('secret-fixture-only',encode([p for a,p in s.calls if a!='input']))
        self.assertNotIn('secret-fixture-only',b''.join(p.read_bytes() for p in s.roots['authority'].iterdir()).decode())
        self.assertEqual({p.name for p in s.roots['control'].iterdir()},{'core'})
        self.assertEqual(s.roots['authority'].stat().st_uid,0)  # injected owner, not kernel evidence

    def test_partial_create_retains_no_retry_or_start(self):
        s=self.s; s.run('install'); s.run('init'); s.fail_action='create'
        with self.assertRaises(OSError): s.run('up')
        self.assertTrue((s.roots['authority']/'container-creation-intent.json').exists())
        s.fail_action=None
        with self.assertRaisesRegex(ValueError,'incomplete_operation'): s.run('up')
        self.assertFalse(any(a=='start' for a,_ in s.calls))

    def test_import_hash_failure_is_retained(self):
        s=self.s; s.bytes[0]=b'x'*len(s.bytes[0])
        with self.assertRaisesRegex(ValueError,'content_changed'): s.run('install')
        self.assertTrue((s.roots['authority']/'import-intent.json').exists())
        with self.assertRaises(ValueError): s.run('init')
        self.assertFalse((s.roots['control']/'core').exists())

    def test_notification_failure_retains_registration_and_no_ready(self):
        s=self.s; s.run('install'); s.run('init'); s.fail_action='notify'
        with self.assertRaises(OSError): s.run('up')
        self.assertEqual(len(list(s.roots['authority'].glob('registration-*.json'))),1)
        self.assertEqual(s.marker['phase'],'waiting')
        s.fail_action=None
        with self.assertRaisesRegex(ValueError,'incomplete_operation'): s.run('status')

    def test_unknown_consumer_and_changed_volume_refuse(self):
        for kind in ('consumer','volume'):
            with self.subTest(kind=kind):
                s=self.s
                if kind=='consumer':
                    c=container('control',s.config,s.b,identity='f'*64); s.cs[c['Id']]=c
                else: s.b['volumes']['models']['Options']={'type':'nfs'}
                with self.assertRaises(ValueError): s.run('install')
                if kind=='consumer': s.cs.pop('f'*64)

    def test_context_and_volume_policy_matrix(self):
        b=self.s.b
        for endpoint in ('ssh://host','tcp://localhost:2375','https://localhost','unix://relative'):
            with self.subTest(endpoint=endpoint),self.assertRaises(ValueError): dv.daemon(dict(b['daemon'],endpoint=endpoint))
        for key,value in [('os','darwin'),('architecture','aarch64'),('cgroup','1'),('security',['name=rootless']),('security',['name=userns'])]:
            with self.subTest(key=key),self.assertRaises(ValueError): dv.daemon(dict(b['daemon'],**{key:value}))
        for field,value in [('Driver','nfs'),('Options',{'device':'/host'}),('Scope','global'),('Labels',{})]:
            v=copy.deepcopy(b); v['volumes']['control'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): dv.binding(v)
        for endpoint in ('unix:///Users/operator/.orbstack/run/docker.sock','unix:///var/run/docker.sock'):
            dv.daemon(dict(b['daemon'],endpoint=endpoint))

    def test_security_mount_negative_matrix_and_commands(self):
        s=self.s
        for role in ('control','worker'):
            good=container(role,s.config,s.b); dv.inspect_container(good,role,s.config,s.b)
            for field,value in [('Privileged',True),('Init',True),('PidMode','host'),('IpcMode','host'),('CgroupnsMode','host'),('UsernsMode','host'),
                    ('NetworkMode','host'),('CapAdd',['SYS_ADMIN']),('CapDrop',[]),('Binds',['/var/run/docker.sock:/socket']),('MemorySwap',-1),('PidsLimit',-1)]:
                c=copy.deepcopy(good); c['HostConfig'][field]=value
                with self.subTest(role=role,field=field),self.assertRaises(ValueError): dv.inspect_container(c,role,s.config,s.b)
            for change in (lambda c:c['Mounts'].append(dict(Type='bind',Source='/',Destination='/host',RW=True)),
                    lambda c:c['Mounts'][0].update(RW=not c['Mounts'][0]['RW']),lambda c:c['Mounts'][0].update(Source='/other')):
                c=copy.deepcopy(good); change(c)
                with self.assertRaises(ValueError): dv.inspect_container(c,role,s.config,s.b)
        for phase in dv.PHASES:
            args=dv.command('helper',s.config,s.b,phase=phase,nonce='a'*32)
            self.assertIn('--pull=never',args); self.assertIn('none',args)
            self.assertNotIn('--privileged',args); self.assertNotIn('--cap-add=SYS_ADMIN',args)
            self.assertTrue(all('type=bind' not in a and 'docker.sock' not in a for a in args))
            self.assertTrue(all('volume-nocopy' in args[i+1] for i,a in enumerate(args) if a=='--mount'))
            roles=dv.HELPER_ROLES[phase]
            self.assertEqual(len(dv.mounts('helper',s.b,phase)),len(roles))
            if phase=='init': self.assertFalse({'models','config','credential'} & set(roles))
        for role in ('control','worker'):
            self.assertNotIn('/authority',[m['Destination'] for m in dv.mounts(role,s.b)])
            if role=='worker': self.assertNotIn('--cap-add=DAC_OVERRIDE',dv.command(role,s.config,s.b))

    def test_manifest_secrets_traversal_duplicates_and_limits(self):
        valid=[dict(e,source='/preprovided') for e in self.s.inputs]; dc.manifest(valid)
        for change in (lambda v:v[0].update(path='../escape'),lambda v:v.append(v[0]),lambda v:v[-1].update(sha256='f'*64),
                lambda v:v[0].update(bytes=True),lambda v:v[0].update(bytes=dc.TOTAL+1),lambda v:v[0].update(path='x//y')):
            value=copy.deepcopy(valid); change(value)
            with self.assertRaises(ValueError): dc.manifest(value)
        serialized=encode(self.s.inputs)
        self.assertNotIn('secret-fixture-only',serialized)
        credential=next(e for e in self.s.inputs if e['role']=='credential'); self.assertNotIn('sha256',credential)

    def test_local_facts_no_host_pid_equality_or_global_namespace_uniqueness(self):
        s=self.s; s.install(); gate=s.gate; public=json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']
        # State.Pid is diagnostic; both local processes legitimately have PID1.
        for c in gate['observation']['inspections'].values(): c['State']['Pid']=0
        gate['observation']['local']['worker']['pid_namespace']=gate['observation']['local']['control']['pid_namespace']
        dv.observation(gate['observation'],gate['registration'],s.config,s.b,public)
        for change in (lambda v:v['local']['worker'].update(pid=2),lambda v:v['hello'].update(boot='f'*32),
                lambda v:v['local']['worker']['roots'].update(ipc=[1,2])):
            value=copy.deepcopy(gate['observation']); change(value)
            with self.assertRaises(ValueError): dv.observation(value,gate['registration'],s.config,s.b,public)


class ProtocolTests(unittest.TestCase):
    def pipes(self):
        a,b=os.pipe(); c,d=os.pipe()
        files=[os.fdopen(a,'rb',buffering=0),os.fdopen(b,'wb',buffering=0),os.fdopen(c,'rb',buffering=0),os.fdopen(d,'wb',buffering=0)]
        for f in files:self.addCleanup(f.close)
        return files
    def test_nonce_phase_step_and_extra_fields(self):
        for change in (lambda v:v.update(nonce='b'*32),lambda v:v.update(phase='stop'),lambda v:v.update(step=2),lambda v:v.update(extra=True)):
            reader,writer,_,out=self.pipes(); wire=dc.Wire(reader,out,'a'*32,'up',deadline=time.monotonic()+1)
            value=dict(version=1,nonce='a'*32,phase='up',step=1,action='snapshot',payload={}); change(value)
            writer.write(dc.encoded(value)+b'\n')
            with self.assertRaises(ValueError): wire.receive()
    def test_attached_exchange_and_eof_never_success(self):
        ar,aw,br,bw=self.pipes(); client=dc.Wire(ar,bw,'a'*32,'up'); helper=dc.Wire(br,aw,'a'*32,'up')
        client.send('begin',{}); self.assertEqual(helper.receive(),('begin',{}))
        helper.send('snapshot',{}); self.assertEqual(client.receive(),('snapshot',{}))
        aw.close()
        with self.assertRaisesRegex(ValueError,'eof'): client.receive()
    def test_duplicate_json_fields_refuse(self):
        with self.assertRaises(ValueError): dc.parse(b'{"nonce":1,"nonce":2}')


class ChildAndClientTests(unittest.TestCase):
    def test_fixed_container_child_permanent_drop_no_namespace_calls(self):
        request=dict(public={},control_fd=10,staging_fd=11)
        events=[]
        with patch.object(ic.sys,'platform','linux'),patch.object(ic.sys,'stdin',SimpleNamespace(buffer=io.BytesIO(encode(request).encode()))), \
             patch.multiple(os,getuid=lambda:0,geteuid=lambda:0,getgroups=lambda:[],umask=lambda _:0o077), \
             patch.object(os,'getresuid',create=True,return_value=(10001,)*3),patch.object(os,'getresgid',create=True,return_value=(10001,)*3), \
             patch.object(os,'setgroups',side_effect=lambda value:events.append(('groups',value))), \
             patch.object(os,'setresgid',create=True,side_effect=lambda *value:events.append(('gid',value))), \
             patch.object(os,'setresuid',create=True,side_effect=lambda *value:events.append(('uid',value))), \
             patch.object(ic,'prepare_container',return_value=(Path('/control'),Path('/staging'))) as prepare, \
             patch.object(ic,'close_other_descriptors') as close,patch.object(ic,'verify_dropped_capabilities') as caps, \
             patch.object(pi,'_split_core') as core,patch.object(ic.LinuxMounts,'unshare',side_effect=AssertionError('unshare')), \
             patch.object(ic.LinuxMounts,'mount',side_effect=AssertionError('mount')):
            ic.container_entry('core')
            prepare.assert_called_once_with({},10,11); close.assert_called_once_with(())
            caps.assert_called_once(); core.assert_called_once_with({},Path('/control'),Path('/staging'))
        self.assertEqual(events,[('groups',[]),('gid',(10001,)*3),('uid',(10001,)*3)])

    def test_retained_capabilities_or_saved_identity_fail(self):
        text='CapInh:\t0\nCapPrm:\t0\nCapEff:\t0\nCapAmb:\t0\nNoNewPrivs:\t1\n'
        with patch.object(Path,'read_text',return_value=text): ic.verify_dropped_capabilities()
        for source in (text.replace('CapPrm:\t0','CapPrm:\t1'),text.replace('NoNewPrivs:\t1','NoNewPrivs:\t0')):
            with patch.object(Path,'read_text',return_value=source),self.assertRaises(ValueError): ic.verify_dropped_capabilities()
        with patch.object(os,'setgroups'),patch.object(os,'setresgid',create=True),patch.object(os,'setresuid',create=True), \
             patch.object(os,'getresuid',create=True,return_value=(10001,10001,0)),patch.object(os,'getresgid',create=True,return_value=(10001,)*3), \
             patch.object(os,'getgroups',return_value=[]),patch.object(os,'umask'),self.assertRaises(ValueError): ic.drop_control()

    def test_context_freezes_forwarded_endpoint_ignores_ambient_and_detects_change(self):
        info=dict(ID='daemon',OSType='linux',Architecture='amd64',SecurityOptions=[],CgroupVersion='2')
        endpoint='unix:///Users/operator/.orbstack/run/docker.sock'
        with patch.dict(os.environ,{'DOCKER_HOST':'tcp://wrong:2375','DOCKER_CONTEXT':'wrong'}), \
             patch.object(dc.Docker,'json',side_effect=[[dict(Endpoints=dict(docker=dict(Host=endpoint)))],info,dict(info,ID='changed')]):
            d=dc.Docker('orbstack'); self.assertEqual(d.argv(['info']),['docker','--host',endpoint,'info'])
            self.assertNotIn('DOCKER_HOST',d.env); self.assertNotIn('DOCKER_CONTEXT',d.env)
            with self.assertRaisesRegex(ValueError,'identity_changed'): d.check()

    def test_real_packaged_client_commands_with_fake_daemon_and_same_helper(self):
        s=Session(self); input_dir=s.root/'preprovided'; input_dir.mkdir(); inputs=[]
        for i,(entry,raw) in enumerate(zip(s.inputs,s.bytes)):
            source=input_dir/str(i); source.write_bytes(raw); inputs.append(dict(entry,source=str(source)))
        request=dict(configuration=s.config,helper_image=s.b['helper_image'],inputs=inputs,cpu=s.b['cpu'],tmpfs=s.b['tmpfs'],control_port=s.b['control_port'])
        request_file=s.root/'request.json'; request_file.write_text(encode(request)); handle=s.root/'handle.json'
        class FakeDocker:
            def __init__(self,context=None,*,endpoint=None): self.frozen=s.b['daemon']; self.names=set()
            def check(self): pass
            def call(self,args):
                if args[:2]==['volume','ls']: return '\n'.join(self.names)
                if args[:2]==['volume','create']: self.names.add(args[-1]); return args[-1]
                if args[:2]==['image','inspect']: return args[2]
                if args[:2]==['container','create']: return format(s.n+11,'064x')
                raise AssertionError(args)
            def json(self,args):
                if args[:2]==['volume','inspect']:
                    return [copy.deepcopy(next(v for v in s.b['volumes'].values() if v['Name']==args[2]))]
                raise AssertionError(args)
            def exchange(self,identity,phase,nonce,config,b,source):
                if b!=s.b or config!=s.config: raise AssertionError('binding changed')
                return s.run(phase)
        output=io.StringIO()
        from contextlib import redirect_stdout
        with patch.object(dc,'Docker',FakeDocker),patch.object(dc,'secrets',SimpleNamespace(token_hex=Mock(side_effect=['a'*32,'1'*32,'2'*32,'3'*32,'4'*32,'5'*32]))),redirect_stdout(output):
            dc.main(['install','--context','desktop-linux','--request',str(request_file),'--handle',str(handle)])
            dc.main(['up','--handle',str(handle)])
            dc.main(['status','--handle',str(handle)])
            dc.main(['stop','--handle',str(handle)])
        self.assertEqual([json.loads(line)['phase'] for line in output.getvalue().splitlines()],['initialized','ready','ready','stopped'])
        self.assertNotIn('secret-fixture-only',handle.read_text())
        self.assertNotIn('secret-fixture-only',Path(str(handle)+'.binding').read_text())
        with patch.object(dc,'Docker',FakeDocker),patch.object(dc.secrets,'token_hex',return_value='a'*32),self.assertRaises(SystemExit):
            dc.main(['install','--context','desktop-linux','--request',str(request_file),'--handle',str(handle)])


from tests.service.test_installed_preset import InstalledPresetTests
from tests.service import test_installed_activation as activation
from tests.release import test_headless_install as gate_tests
from service.dispatcher import Dispatcher


class DaemonPublicTests(InstalledPresetTests):
    """Same helper-created Core/enrollment/registration enters real Dispatcher/API.

    Runtime attempts/media are injected as in the existing installed public fixture.
    The fixture models a PID1-owned UID20000+ journal and clean stop/up after
    completed published work, without executing an OS child or real Docker.
    """
    def setUp(self):
        def gate_setup(test):
            previous_umask=os.umask(0o077); test.addCleanup(os.umask,previous_umask)
            s=Session(test); s.install(); s.run('stop'); s.run('up'); test.session=s
            public=json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']
            test.fixture=SimpleNamespace(fixture=s.f,config=s.config,roots=s.roots,public=public,hello=s.startup()['hello'],lane=s.startup()['hello']['lane'])
            test.gate=copy.deepcopy(s.gate); test.rid=s.gate['registration_id']; test.root=s.roots['control']/'core'
            test.incarnation={k:s.facts('control')[k] for k in ('start_ticks','pid_namespace','cgroup')}
            test.original_translate=s.f.translate
        with patch.object(gate_tests.GateTests,'setUp',gate_setup): activation.ActivationTests.setUp(self)
        self.open_owners=[]; self.runtime.records=[]; self.stage_uids={}
        old_prepare=self.runtime.prepare; old_collect=self.runtime.collect; old_call=self.runtime.call
        from worker.runtime_journal import journal_bytes
        def persist():
            value=dict(version=1,lane=self.fixture.lane,records=self.runtime.records)
            (self.session.roots['receipts']/'journal.json').write_bytes(journal_bytes(value))
        def prepare(authority,fixture):
            old_prepare(authority,fixture)
            uid=20000+len(self.runtime.records); identity=self.fixture.lane+'/'+self.fixture.hello['boot']+'/'+format(uid,'032x')
            self.stage_uids[authority.job['attempts'][-1]['staging_id']]=uid
            self.runtime.records.append(dict(id=identity,allocation=dict(snapshot=json.loads(authority._snapshot),binding=authority.binding,fixture=fixture),
                phase='prepared',acked=False,uid=uid,boot=self.fixture.hello['boot']))
            persist(); return identity
        def collect(authority,identity):
            receipt,proposal=old_collect(authority,identity)
            row=next(r for r in self.runtime.records if r['id']==identity)
            receipt['engine_uid']=row['uid']
            row.update(phase='terminal',receipt=copy.deepcopy(receipt),proposal=None if proposal is None else dict(message=proposal.message,manifest=proposal.manifest))
            persist(); return receipt,proposal
        def call(op,body):
            value=old_call(op,body)
            if op=='ack':
                row=next(r for r in self.runtime.records if r['id']==body['runtime_id']); row.update(acked=True,proposal=None); persist()
            return value
        self.runtime.prepare=prepare; self.runtime.collect=collect; self.runtime.call=call
        previous=self.session.f.translate
        def ownership(info,path=None):
            if path==self.stage: return self.original_translate(info,path)
            if path is not None and self.stage in path.parents:
                relative=path.relative_to(self.stage)
                import stat
                uid=10001 if stat.S_ISDIR(info.st_mode) else self.stage_uids.get(relative.parts[0],10001)
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=uid,st_gid=10001))
            return previous(info,path)
        self.session.f.translate=ownership
        previous_lstat=os.lstat
        self.enterContext(patch.object(os,'lstat',side_effect=lambda p,*a,**kw:ownership(previous_lstat(p,*a,**kw),Path(p))))

    def open(self):
        owner,app=super().open(); self.open_owners.append(owner); return owner,app

    def test_empty_public_journey_same_root_reopen_and_frozen_job(self):
        super().test_empty_public_journey_same_root_reopen_and_frozen_job()
        for owner in self.open_owners:
            if owner.thread.is_alive(): owner.close()
        request=copy.deepcopy(self.runtime.records[0]['allocation']['snapshot'][1]['plan']['request'])
        job_id=self.runtime.records[0]['allocation']['snapshot'][0]['id']
        self.assertEqual(self.session.run('stop')['phase'],'stopped')
        self.assertEqual(self.session.run('up')['phase'],'ready')
        self.assertEqual(len(self.session.f.calls),1)
        # Keep the old reader open until its registered cleanup; new notification
        # uses a distinct modeled incarnation tmpfs directory.
        directory=self.session.root/'notification-worker-restart'
        translate=self.session.f.translate
        def owner(info,path=None):
            if path is not None and (path==directory or directory in path.parents):
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=10001,st_gid=10001))
            return translate(info,path)
        self.session.f.translate=owner
        old_waiting=self.waiting; self.addCleanup(old_waiting.close)
        self.waiting=ep.WaitingControl(directory=directory)
        self.gate=copy.deepcopy(self.session.gate); ep.publish(encode(self.gate).encode(),directory=directory)
        self.incarnation.update({k:self.session.facts('control')[k] for k in self.incarnation})
        self.fixture.hello=self.session.startup()['hello']; self.runtime.authority=None
        owner3,app=self.open()
        from clients.http import HTTP
        with self.bridge(app):
            http=HTTP(credential='writer'); replay=http.call('speech_generate',request,key='first-generation')
            self.assertEqual(replay['job_id'],job_id); self.assertIsNone(self.runtime.authority)
            old=http.call('job_get',dict(job_id=job_id)); self.assertEqual(old['state'],'succeeded')
            self.assertTrue(http.call('artifact_get',dict(artifact_id=old['artifact_ids'][0]))['asset_ids'])
            fresh=http.call('speech_generate',request,key='after-worker-restart')
            self.wait(lambda:self.runtime.authority)
            self.wait(lambda:http.call('job_get',dict(job_id=fresh['job_id']))['state']=='succeeded')
            self.wait(lambda:owner3.call('status')['phase']=='idle')
        owner3.close()

    def owner(self):
        return Dispatcher(self.root,staging=self.stage,runtime=self.runtime,profiles=self.profiles,
            budget=self.fixture.config['budget'],production_contexts=self.contexts,
            installed_activation=ep.InstalledActivation(self.waiting,self.fixture.config,launch_policy=dv.POLICY))

    def test_mixed_host_evidence_and_unselected_new_policy_refuse(self):
        with self.assertRaisesRegex(ValueError,'explicit_launch_policy'):
            recovery.installed_startup_read(self.root,self.fixture.config,self.gate,self.fixture.hello,self.incarnation,policy=ip.SPLIT)
        changed=copy.deepcopy(self.gate); changed['registration']['evidence']='actual-host-inspection'
        changed['registration_id']=pi.digest(changed['registration'])
        with self.assertRaises(ValueError): ep.projection(changed)
        with self.assertRaises(ValueError): recovery.host_registration_union([self.gate['registration']],self.fixture.public['host'],self.fixture.public['deployment_id'])

    def test_stale_registration_and_unsafe_history_refuse(self):
        changed=copy.deepcopy(self.gate); changed['observation']['local']['control']['start_ticks']+=1
        with self.assertRaises(ValueError):
            recovery.installed_startup_read(self.root,self.fixture.config,changed,self.fixture.hello,self.incarnation,policy=ip.SPLIT,launch_policy=dv.POLICY)
        with sqlite3.connect(self.root/'core.sqlite') as db: db.execute("UPDATE resource_monitor SET state='fault'")
        with self.assertRaises(ValueError): self.session.helper.clean(lane_lock=True)


    def test_unknown_unreaped_unacked_and_partial_publication_history_refuse(self):
        owner,app=self.open()
        from clients.http import HTTP
        with self.bridge(app):
            http=HTTP(credential='writer')
            saved=http.call('voice_preset_register',self.request(),key='history-preset')
            request=dict(self.context['plan']['request'],voice_version_id=saved['version']['id'])
            receipt=http.call('speech_generate',request,key='history-generation')
            self.wait(lambda:self.runtime.authority); self.runtime.finish.set()
            self.wait(lambda:http.call('job_get',dict(job_id=receipt['job_id']))['state']=='succeeded')
            self.wait(lambda:owner.call('status')['phase']=='idle')
        owner.close()
        from worker.runtime_journal import journal_bytes
        path=self.session.roots['receipts']/'journal.json'; original=path.read_bytes(); journal=json.loads(original)['payload']
        for fault in ('unacked','unreaped','unknown'):
            changed=copy.deepcopy(journal); row=changed['records'][0]
            if fault=='unacked': row['acked']=False
            elif fault=='unreaped': row['receipt']['reaped']=False
            else: row.update(phase='started',acked=False); row.pop('receipt'); row.pop('proposal')
            broken=journal_bytes(changed); path.write_bytes(broken)
            with self.subTest(fault=fault),self.assertRaises(ValueError): self.session.helper.clean(lane_lock=True)
            self.assertEqual(path.read_bytes(),broken)
            path.write_bytes(original)
        with sqlite3.connect(self.root/'core.sqlite') as db:
            db.execute('UPDATE publication_work SET binding=? WHERE job_id=?',('{}',receipt['job_id']))
        with self.assertRaises((ValueError,KeyError)): self.session.helper.clean(lane_lock=True)


class FailureAndPackagingTests(unittest.TestCase):
    def test_init_child_failure_quarantines_core_and_no_second_constructor(self):
        s=Session(self); s.run('install')
        def fail(*_): raise OSError('injected child completion')
        s.child=fail
        with self.assertRaises(OSError): s.run('init')
        self.assertTrue((s.roots['receipts']/'p1-core-intent').exists())
        self.assertFalse((s.roots['receipts']/'p1-enrolled.json').exists())
        with self.assertRaisesRegex(ValueError,'incomplete_operation'): s.run('init')
        self.assertEqual(s.f.calls,[])

    def test_writer_and_final_bracket_failure_leave_pending_no_success(self):
        for point in ('writer','final'):
            with self.subTest(point=point):
                s=Session(self); s.run('install'); s.run('init')
                if point=='writer':
                    original=s.child
                    def child(kind,*args):
                        if kind=='writer': raise OSError('injected writer')
                        return original(kind,*args)
                    s.child=child
                else:
                    original=s.action
                    def action(name,payload):
                        if name=='snapshot' and s.marker['phase']=='ready': raise OSError('injected final bracket')
                        return original(name,payload)
                    s.action=action
                with self.assertRaises(OSError): s.run('up')
                self.assertEqual(len(list(s.roots['authority'].glob('registration-*.json'))),1)
                self.assertFalse((s.roots['authority']/('operation-'+format(s.n,'032x')+'.done')).exists())

    def test_launch_lock_contention_and_mountinfo_nested_alias_refuse(self):
        s=Session(self); s.run('install'); s.run('init')
        fd=os.open(s.roots['authority'],os.O_RDONLY|os.O_DIRECTORY)
        try:
            import fcntl
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'lock_busy'): s.run('up')
        finally: os.close(fd)
        h=s.helper
        entries=[dict(target=str(h.root[r]),options=['rw'],root='/volumes/'+r,device='1:1') for r in dv.HELPER_ROLES['up']]
        with patch.object(recovery,'_host_mount_topology',return_value=entries): h.mounts_checked()
        for changed in (entries+[dict(target=str(h.root['control'])+'/nested',options=['rw'],root='/nested',device='1:1')],
                [dict(e,root='/same') for e in entries]):
            with patch.object(recovery,'_host_mount_topology',return_value=changed),self.assertRaises(ValueError): h.mounts_checked()

    def test_symlink_hardlink_nonempty_ownership_and_credential_negatives(self):
        for fault in ('symlink','hardlink','nonempty','owner','credential'):
            with self.subTest(fault=fault):
                s=Session(self)
                if fault=='credential':
                    i=next(i for i,e in enumerate(s.inputs) if e['role']=='credential'); s.bytes[i]=b'x'*s.inputs[i]['bytes']
                elif fault=='nonempty': (s.roots['models']/'foreign').write_bytes(b'foreign')
                elif fault=='owner':
                    st=s.roots['models'].stat(); s.f.virtual[st.st_dev,st.st_ino]=(10001,10001)
                elif fault=='hardlink':
                    # Hardlinked source files are refused at the stdlib client boundary.
                    path=s.root/'source'; path.write_bytes(b'x'); os.link(path,s.root/'alias')
                    with self.assertRaises(ValueError): dc.read(path)
                    continue
                else:
                    path=s.roots['models']; path.rmdir(); path.symlink_to(s.roots['config'],target_is_directory=True)
                with self.assertRaises((ValueError,OSError)): s.run('install')
                self.assertFalse((s.roots['control']/'core').exists())

    def test_distribution_exact_entries_bytes_determinism_no_child(self):
        import zipfile
        from service.package_install import archive
        root=Path(__file__).resolve().parents[2]; s=Session(self)
        a=s.root/'one.zip'; b=s.root/'two.zip'
        self.assertEqual(archive(root,a),archive(root,b))
        manifest=json.loads((root/'headless-dist-manifest.json').read_bytes())
        with zipfile.ZipFile(a) as z:
            self.assertEqual(set(z.namelist()),{'speech-box/'+p for p in manifest['files']}|{'speech-box/speech-box'})
            for path in manifest['files']: self.assertEqual(z.read('speech-box/'+path),(root/path).read_bytes())
        self.assertIn('service/daemon_helper.py',manifest['files'])
        self.assertIn('service/compose.daemon-volumes.yml',manifest['files'])

    def test_authenticated_startup_operation_is_closed(self):
        from worker.runtime_supervisor import RuntimeSupervisor
        from worker.runtime_protocol import request, VERSION
        from worker.transport import Refused
        owner=RuntimeSupervisor.__new__(RuntimeSupervisor)
        owner.boot='b'*32; owner.journal=SimpleNamespace(value=dict(lane='a'*32,records=[]),maintenance=None)
        owner.failed=False; owner.launches=0; owner.active=None
        facts=dict(injected='PID1 facts hook only')
        with patch('service.daemon_runtime.facts',return_value=facts):
            result=owner.operation('startup',{})
        self.assertEqual(result['local'],facts); self.assertEqual(result['hello']['boot'],'b'*32)
        with self.assertRaises(Refused): owner.operation('startup',dict(extra=True))
        self.assertEqual(request(dict(version=VERSION,op='startup',body={})),('startup',{}))


class AdapterBoundsTests(ProtocolTests):
    """Actual adapters with inert pipes and controlled fake reap transitions."""
    def process(self, data=b'', wait=None):
        reader,writer,input_reader,input_writer=self.pipes()
        if data: writer.write(data)
        writer.close()
        p=SimpleNamespace(stdout=reader,stdin=input_writer,returncode=None)
        def done(timeout):
            if wait: wait(p,timeout)
            else: p.returncode=0
            return p.returncode
        p.wait=done
        return p

    def test_child_timeout_holds_every_root_until_reap(self):
        import fcntl
        s=Session(self); s.run('install'); s.child=dh.child; calls=[]
        def wait(p,timeout):
            self.assertEqual(timeout,.1)
            for role in ('authority','control','receipts','staging','ipc'):
                fd=os.open(s.roots[role],os.O_RDONLY|os.O_DIRECTORY)
                try:
                    with self.assertRaises(BlockingIOError): fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                finally: os.close(fd)
            self.assertTrue((s.roots['receipts']/'p1-core-intent').exists())
            calls.append('wait')
            if len(calls)<3: raise dc.subprocess.TimeoutExpired('inert',timeout)
            p.returncode=0
        p=self.process(wait=wait); original=dc.collect
        def expired(*args): return original(*args[:3],-1,args[4])
        with patch.object(dh.subprocess,'Popen',return_value=p),patch.object(dh,'collect',side_effect=expired):
            with self.assertRaisesRegex(ValueError,'child_timeout_retained'): s.run('init')
        self.assertEqual(len(calls),3)
        self.assertTrue((s.roots['receipts']/'p1-core-intent').exists())
        for role in ('authority','control','receipts','staging','ipc'):
            fd=os.open(s.roots[role],os.O_RDONLY|os.O_DIRECTORY)
            try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally: os.close(fd)

    def test_both_adapters_live_overflow_bounded_and_reaped(self):
        import threading
        for helper,limit in ((True,65536),(False,dc.LIMIT)):
            with self.subTest(helper=helper):
                reader,writer,input_reader,input_writer=self.pipes()
                sent=[]; reaped=[]; peak=[]
                def produce():
                    try:
                        for _ in range(limit//4096+16): writer.write(b'x'*4096)
                        sent.append(True)
                    finally: writer.close()
                thread=threading.Thread(target=produce); thread.start()
                p=SimpleNamespace(stdout=reader,stdin=input_writer if helper else None,returncode=None)
                def wait(timeout):
                    thread.join(1); self.assertFalse(thread.is_alive()); reaped.append(True); p.returncode=0; return 0
                p.wait=wait
                class Capped(bytearray):
                    def extend(self,data):
                        super().extend(data); peak.append(len(self)); self_test.assertLessEqual(len(self),limit)
                self_test=self
                d=object.__new__(dc.Docker); d.endpoint='unix:///fixture'; d.env={}; d.end=time.monotonic()+3; d.retained=[]
                with patch.object(dc.subprocess,'Popen',return_value=p) as popen,patch.object(dc,'bytearray',Capped,create=True):
                    with self.assertRaisesRegex(ValueError,'output_budget'):
                        if helper: dh.child('hello',{},())
                        else: d.call(['fixture'])
                self.assertEqual(popen.call_args.kwargs['stdout'],dc.subprocess.PIPE)
                self.assertTrue(sent and reaped); self.assertLessEqual(max(peak,default=0),limit)
                self.assertEqual(d.retained,[])

    def test_client_timeout_reaps_before_return_and_releases_reference(self):
        calls=[]
        def wait(p,timeout):
            calls.append(timeout)
            if len(calls)<3: raise dc.subprocess.TimeoutExpired('inert',timeout)
            p.returncode=0
        p=self.process(wait=wait); p.stdin.close(); p.stdin=None
        d=object.__new__(dc.Docker); d.endpoint='unix:///fixture'; d.env={}; d.end=-1; d.retained=[]
        with patch.object(dc.subprocess,'Popen',return_value=p):
            with self.assertRaisesRegex(ValueError,'client_timeout_retained'): d.call(['fixture'])
        self.assertEqual(calls,[.1]*3); self.assertEqual(d.retained,[])

    def test_exchange_exact_eof_pending_and_subsequent_trailing(self):
        s=Session(self); nonce='a'*32; identity='3'*64
        c=container('helper',s.config,s.b,phase='init',nonce=nonce,identity=identity)
        c['State'].update(Running=True,Status='running',StartedAt='2026-10-25T00:00:00Z')
        frame=dc.encoded(dict(version=1,nonce=nonce,phase='init',step=2,action='complete',payload=dict(result={'phase':'initialized'})))+b'\n'
        for suffix,split in ((b'',False),(b'junk',False),(frame,False),(b'junk',True)):
            with self.subTest(suffix=suffix,split=split):
                p=self.process(frame+suffix)
                d=object.__new__(dc.Docker); d.endpoint='unix:///fixture'; d.env={}; d.end=time.monotonic()+2; d.retained=[]
                final=copy.deepcopy(c); final['State'].update(Running=False,Status='exited')
                d.container=Mock(side_effect=[c,c,final]); d.check=Mock()
                read=os.read
                def split_read(fd,n): return read(fd,min(n,len(frame)))
                with patch.object(dc.subprocess,'Popen',return_value=p),patch.object(dc.os,'read',side_effect=split_read if split else read):
                    if suffix:
                        with self.assertRaisesRegex(ValueError,'trailing_protocol'): d.exchange(identity,'init',nonce,s.config,s.b,[])
                    else: self.assertEqual(d.exchange(identity,'init',nonce,s.config,s.b,[]),{'phase':'initialized'})
                if suffix: d.check.assert_not_called()

    def test_terminal_eof_timeout(self):
        reader,writer,_,out=self.pipes(); wire=dc.Wire(reader,out,'a'*32,'up',deadline=time.monotonic()+.01)
        with self.assertRaisesRegex(ValueError,'terminal_eof_timeout'): wire.finish()

    def test_good_output_and_nonzero_exit(self):
        for helper in (False,True):
            for status in (0,2):
                with self.subTest(helper=helper,status=status):
                    p=self.process(b'{"ok":true}' if helper else b'  result\n')
                    def wait(timeout): p.returncode=status; return status
                    p.wait=wait
                    d=object.__new__(dc.Docker); d.endpoint='unix:///fixture'; d.env={}; d.end=time.monotonic()+1; d.retained=[]
                    with patch.object(dc.subprocess,'Popen',return_value=p):
                        if status:
                            with self.assertRaisesRegex(ValueError,'failed_retained'):
                                if helper: dh.child('hello',{},())
                                else: d.call(['fixture'])
                        elif helper: self.assertEqual(dh.child('hello',{},()),{'ok':True})
                        else: self.assertEqual(d.call(['fixture']),'result')
                    self.assertEqual(d.retained,[])

    def test_setup_error_and_unknown_wait_cannot_unwind_before_reap(self):
        p=self.process(); calls=[]
        def wait(timeout):
            calls.append(timeout)
            if len(calls)==1: return None
            p.returncode=0; return 0
        p.wait=wait
        with patch.object(dc.os,'set_blocking',side_effect=OSError('inert setup fault')):
            with self.assertRaisesRegex(OSError,'inert setup fault'):
                dc.collect(p,b'',64,time.monotonic()+1,'fixture')
        self.assertEqual(calls,[.1,.1]); self.assertTrue(p.stdout.closed and p.stdin.closed)
