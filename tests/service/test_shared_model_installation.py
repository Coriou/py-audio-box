"""Real catalog/files/Core/registration with injected Docker and UID boundary.

No dependency substitutes, Docker execution or model runtime. Session's dropped
writer adapter is copied from the existing daemon fixture to avoid its unrelated
module-level MCP test aggregation.
"""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from service import daemon_client as dc, daemon_helper as dh, daemon_volumes as dv
from service import shared_models as sm, shared_model_helper as sh, shared_model_client as sc
from service import install_policy as ip, persistent_init as pi, install_child as ic, recovery
from service import installed_qwen_daemon as qd
from service.core import encode
from tests.service import test_split_init as split_fixture
from tests.service import test_installed_qwen as qwen_fixture
from tests.service.test_shared_model_binding import binding, container
from tests.service.test_shared_model_transport import FakeDocker, LocalClient, client, exit_container
from tests.service.test_shared_models import fixture


class Session:
    def __init__(self,test,f,config,b,roots,cs):
        self.test=test; self.f=f; self.config=config; self.b=b; self.roots=roots; self.cs=cs
        self.family=qd.FAMILY; self.scope=dv.scope(b,family=qd.FAMILY)
        self.calls=[]; self.fail_action=None; self.mutate=None; self.starts=0; self.boot='b'*32
        self.marker=dict(phase='preparing'); self.gate=None
    def facts(self,role):
        names=('control','ipc','staging','models','config','credential') if role=='control' else ('receipts','ipc','staging','models','config')
        roots={r:[self.roots[r].stat().st_dev,self.roots[r].stat().st_ino] for r in names}
        if self.family is not None:
            assets=self.roots['control']/'core/assets'; roots['assets']=[assets.stat().st_dev,assets.stat().st_ino]
        return dict(role=role,pid=1,start_ticks=100+self.starts,pid_namespace=200+(role=='worker'),cgroup='0::/',
            uid=10001 if role=='control' else 0,gid=10001,groups=[],capabilities=0 if role=='control' else 224,
            roots=roots)

    def startup(self):
        journal=json.loads((self.roots['receipts']/'journal.json').read_bytes())['payload']
        hello=dict(lane=journal['lane'],boot=self.boot,quarantined=False,launches=0)
        from service.installed_export import FAMILY
        if self.family is FAMILY:
            from service.installed_export import export_seal
            hello['export_seal']=export_seal(self.config)
        return dict(hello=hello,local=self.facts('worker'))

    def action(self,action,payload):
        self.calls.append((action,copy.deepcopy(payload)))
        if self.fail_action==action: raise OSError('injected '+action)
        if action=='snapshot':
            return self.docker.snapshot(self.b)
        if action=='input':
            raw=self.bytes[payload['index']][payload['offset']:payload['offset']+49152]
            return dict(data=base64.b64encode(raw).decode())
        if action=='create':
            role=payload['role']; c=container(role,self.config,self.b,identity=format(len(self.docker.containers)+1,'064x'),family=self.family); self.cs[c['Id']]=c; self.docker.containers[c['Id']]=c; return copy.deepcopy(c)
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
        if 'model_storage' in self.b:
            import subprocess,sys
            paths=[self.roots['catalog']/'locks'/self.b['model_storage']['manifest_sha256'],self.roots['authority']]
            code="import os,fcntl,sys; fs=[os.open(p,os.O_RDONLY) for p in sys.argv[1:]]\nfor f in fs:\n try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print('unlocked')\n except BlockingIOError: print('blocked')"
            observed=subprocess.run([sys.executable,'-B','-c',code,*map(str,paths)],capture_output=True,timeout=3,check=True)
            self.test.assertEqual(observed.stdout,b'blocked\nblocked\n')
        with patch.object(ic,'CONTAINER_CONTROL',self.roots['control']),patch.object(ic,'CONTAINER_STAGING',self.roots['staging']):
            control,staging=ic.prepare_container(request['public'],request['control_fd'],request['staging_fd'],family=self.family)
        with patch.object(ic.LinuxMounts,'unshare',side_effect=AssertionError('unshare')),patch.object(ic.LinuxMounts,'mount',side_effect=AssertionError('mount')):
            with patch.multiple(os,getuid=lambda:10001,geteuid=lambda:10001,getgid=lambda:10001,getegid=lambda:10001,getgroups=lambda:[]):
                if kind=='core':
                    self.f.calls.append(request); return pi._split_core(request['public'],control,staging,family=self.family)
                return ic.registration_writer(request['public'],request['registration'],request['prestate'],control,request['lock_fd'],family=self.family)



class InstallationDocker(FakeDocker):
    snapshot=dc.Docker.snapshot
    def __init__(self,root,test,f):
        super().__init__(root); self.test=test; self.f=f; self.sessions={}; self.family=qd.FAMILY
        self.current=None; self.input_bytes=0
    def identity(self): return self.frozen
    def call(self,args):
        if args[:2]==['image','inspect']: return args[2]
        if args[:2]==['volume','create'] and any(x.startswith('speech-box.installation=') for x in args):
            name=args[-1]; path=self.root/name; path.mkdir(mode=0o755)
            labels=dict(args[i+1].split('=',1) for i,a in enumerate(args) if a=='--label')
            self.volumes[name]=dict(Name=name,Driver='local',Options=None,Scope='local',Mountpoint=str(path),CreatedAt='2026-10-27T00:00:00Z',Labels=labels)
            return name
        if args[:2]==['container','create'] and any('from service.daemon_helper' in x for x in args):
            for retained in self.root.glob('handle-*.binding'):
                value=dc.parse(dc.read(retained)); selected=value['binding']
                if any('source='+selected['volumes']['authority']['Name']+',' in x for x in args):
                    self.current=(selected,value['configuration']); break
            b,config=self.current
            # Exact helper command tail is phase/nonce after the fixed -c entry.
            phase,nonce=args[-2:]
            identity=format(len(self.containers)+1,'064x')
            c=container('helper',config,b,phase=phase,nonce=nonce,identity=identity,family=qd.FAMILY)
            self.containers[identity]=c; return identity
        return super().call(args)
    def exchange(self,identity,phase,nonce,config,b,source):
        c=self.containers[identity]
        dv.inspect_container(c,'helper',config,b,phase=phase,nonce=nonce,family=qd.FAMILY)
        installation=b['installation']
        roots={r:Path(v['Mountpoint']) for r,v in b['volumes'].items()}
        if 'model_storage' in b: roots['catalog']=Path(b['model_storage']['catalog']['Mountpoint'])
        if installation not in self.sessions:
            self.sessions[installation]=Session(self.test,self.f,config,b,roots,{})
            self.sessions[installation].docker=self
        s=self.sessions[installation]; s.cs[identity]=c
        c['State'].update(Running=True,Status='running',StartedAt='2026-10-27T10:00:00Z')
        s.inputs=[{k:v for k,v in e.items() if k!='source'} for e in source]
        s.bytes=[Path(e['source']).read_bytes() for e in source]
        self.input_bytes+=sum(len(raw) for raw in s.bytes)
        class Wire:
            def send(self,action,payload): self.pending=(action,payload)
            def receive(self):
                action,payload=self.pending; return action,s.action(action,payload)
        h=dh.Helper(phase,nonce,dict(configuration=config,binding=b,helper=copy.deepcopy(c),manifest=s.inputs),
            Wire(),roots={r:roots[r] for r in dv.helper_roles(b,phase)},child_runner=s.child,family=qd.FAMILY)
        s.helper=h; h.end=self.end
        old=os.umask(0o077)
        try:
            with patch.object(h,'mounts_checked'): result=h.run()
        except BaseException:
            exit_container(c,1); raise
        finally: os.umask(old)
        exit_container(c); return result


class InstallationClient(LocalClient):
    def input(self,payload):
        result=super().input(payload)
        self.docker.installation_model_bytes=getattr(self.docker,'installation_model_bytes',0)+len(base64.b64decode(result['data']))
        return result
    def coordinate(self,operation,selection):
        result=super().coordinate(operation,selection)
        if operation in ('bind','attachment-intent'):
            if 'binding' in selection: self.docker.current=(selection['binding'],selection['configuration'])
        return result


class SharedInstallationTests(unittest.TestCase):
    def setUp(self):
        self.f=split_fixture.SplitInit(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        # One virtual UID map covers both independent installations and the store.
        self.original_translate=self.f.translate
        def translate(info,path=None):
            if path is not None:
                for root in getattr(self,'controls',[]):
                    if (path==root or root in path.parents) and (info.st_dev,info.st_ino) not in self.f.virtual:
                        from types import SimpleNamespace
                        return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=10001,st_gid=10001))
            if path is not None and any(p.name==sm.CATALOG or p.name.endswith('-authority') for p in (path,*path.parents)):
                from types import SimpleNamespace
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=0,st_gid=0))
            return self.original_translate(info,path)
        self.f.translate=translate; self.controls=[]
        self.fixture=qwen_fixture.InstalledQwenTests(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.expected,self.data=fixture(); self.key=sm.descriptor(self.expected)[0]
        self.enterContext(patch.object(qd,'MANIFEST_SHA256',self.key))
        self.d=InstallationDocker(self.f.root,self,self.f)
        self.enterContext(patch.object(dc,'Docker',return_value=self.d))
        self.enterContext(patch.object(sc,'Client',InstallationClient))
        self.enterContext(patch.object(sm,'CATALOG_OWNER',(0,0)))
        # Existing ownership adapter records chown in the real inode map.
        def create_call(args):
            result=original_call(args)
            if args[:2]==['volume','create']:
                st=Path(self.d.volumes[result]['Mountpoint']).stat(); self.f.virtual[(st.st_dev,st.st_ino)]=(0,0)
            return result
        original_call=self.d.call; self.d.call=create_call
        c=client(self.f.root,self.d,self.expected,self.data,1)
        self.assertEqual(c.run('init')['schema'],sm.SCHEMA)
        self.published=c.run('import'); self.payload_bytes=c.bytes
        self.selection=self.f.root/'selection.json'
        self.selection.write_bytes(dc.encoded(dict(descriptor=self.expected,catalog=self.published['catalog'],models=self.published['candidate'],publication_sha256=sm.digest(self.published))))
        self.handles=[]
        self.addCleanup(self.writable)
    def writable(self):
        for root,dirs,files in os.walk(self.f.root):
            os.chmod(root,0o755)
    def configuration(self,principal):
        config=copy.deepcopy(self.fixture.config)
        old_hash=config['contexts']['alice']['execution']['descriptor']['manifest_sha256']
        old_id=next(iter(config['bundles']))
        def rename(v):
            if isinstance(v,dict): return {(self.key if k==old_id else principal if k=='alice' else k):rename(x) for k,x in v.items()}
            if isinstance(v,list): return [rename(x) for x in v]
            if isinstance(v,str):
                if v in (old_hash,old_id): return self.key
                if v=='alice': return principal
                return v.replace('/bundles/'+old_id,'/bundles/'+self.key)
            return v
        return rename(config)
    def install(self,principal,*,shared=True):
        config=self.configuration(principal); index=len(self.handles)
        handle=str(self.f.root/('handle-'+str(index)))
        inputs=[]
        credential=dc.encoded(dict(version=1,credentials=[dict(token=principal+'-private-fixture-'+'x'*32,principal=principal,scopes=['read','production-write'])]))
        for role,name,raw in [('config','install.json',dc.encoded(config)),('credential','control.key',credential)]:
            path=self.f.root/(str(index)+'-'+name); path.write_bytes(raw)
            entry=dict(role=role,path=name,source=str(path),bytes=len(raw))
            if role!='credential': entry['sha256']=hashlib.sha256(raw).hexdigest()
            inputs.append(entry)
        if not shared:
            raw_manifest=sm.descriptor(self.expected)[1]
            for name,raw in {**self.data,'manifest.json':raw_manifest}.items():
                path=self.f.root/('legacy-'+name.replace('/','-')); path.write_bytes(raw)
                inputs.append(dict(role='models',path='bundles/'+self.key+'/'+name,source=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        request=self.f.root/('request-'+str(index))
        request.write_bytes(dc.encoded(dict(configuration=config,helper_image=config['images']['control'],inputs=inputs,
            cpu=dict(control=10**9,worker=2*10**9),tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},control_port=8765+index)))
        # Register the new control root's UID view immediately after allocation.
        old_call=self.d.call
        def call(args):
            value=old_call(args)
            if args[:2]==['volume','create'] and value.endswith('-control'): self.controls.append(Path(self.d.volumes[value]['Mountpoint']))
            return value
        self.d.call=call
        self.invoke(['install','--context','fixture','--request',str(request),'--handle',handle]+(['--shared-models',str(self.selection)] if shared else []))
        self.handles.append(handle)
        self.assertFalse(Path(handle+'.shared-selection').exists())
        b=dc.parse(dc.read(handle+'.binding'))['binding']
        return self.d.sessions[b['installation']]
    def invoke(self,args):
        # CLI intentionally redacts errors. Retain the original fixture exception
        # for a useful test traceback without changing production diagnostics.
        import sys
        errors=[]
        def trace(frame,event,arg):
            if event=='exception' and frame.f_code.co_filename==dc.__file__ and frame.f_code.co_name=='main' and isinstance(arg[1],Exception): errors.append(arg)
            return trace if frame.f_code.co_filename==dc.__file__ and frame.f_code.co_name=='main' else None
        previous=sys.gettrace(); sys.settrace(trace)
        try:
            try: return dc.qwen_main(args)
            except SystemExit:
                if errors: raise errors[-1][1].with_traceback(errors[-1][2])
                raise
        finally: sys.settrace(previous)
    def command(self,index,phase):
        self.invoke([phase,'--handle',self.handles[index]])
    def startup(self,s,gate=None):
        from service.entrypoint import projection
        g=projection(copy.deepcopy(s.gate if gate is None else gate),family=qd.FAMILY)
        local=g['observation']['local']['control']
        incarnation={k:local[k] for k in ('start_ticks','pid_namespace','cgroup')}
        return recovery.installed_startup_read(s.roots['control']/'core',s.config,g,s.startup()['hello'],incarnation,
            policy=ip.QWEN_SPLIT,launch_policy=qd.LAUNCH,family=qd.FAMILY)
    def retained_job(self,s,principal):
        from service.core import Core
        from service.ingest import ingest_local
        from service.admission import Admission
        from worker.runtime_protocol import execution,dispatch_mode
        with patch.multiple(os,getuid=lambda:10001,geteuid=lambda:10001,getgid=lambda:10001,getegid=lambda:10001,getgroups=lambda:[]):
            core=Core(s.roots['control']/'core',_publication_creation=1)
            try:
                asset=ingest_local(core,principal,self.fixture.b.raw)
                context=copy.deepcopy(s.config['contexts'][principal])
                old=context['execution']['descriptor']['reference_id']
                def rename(v):
                    if isinstance(v,dict): return {(asset['id'] if k==old else k):rename(x) for k,x in v.items()}
                    if isinstance(v,list): return [rename(x) for x in v]
                    if isinstance(v,str): return asset['id'] if v==old else v.replace('/'+old,'/'+asset['id'])
                    return v
                context=rename(context); context['plan']['assets']=[asset]
                core.voices.register(principal,'saved-voice',context['plan']['voice_versions'][0],expected_revision=0)
                admission=Admission(core,s.roots['staging'],s.config['budget'])
                job=core.submit_execution(principal,'retained-job',context,_admission=admission,_profiles=s.config['profiles'],
                    _dispatch_settings=(dispatch_mode(execution(context)),1800))
                cancelled=core.request_cancel(job['id'],_cancel_unstarted=True,_admission=admission)
                self.assertEqual(cancelled['state'],'cancelled')
                return cancelled
            finally: core.close()

    def test_two_packaged_installs_real_core_registration_recovery_restart(self):
        a=self.install('alice'); self.command(0,'up')
        b=self.install('bob'); self.command(1,'up')
        self.assertEqual(self.d.allocations,1)
        self.assertEqual(getattr(self.d,'installation_model_bytes',0),0)
        self.assertEqual(self.payload_bytes,sum(len(v) for v in self.data.values()))
        self.assertEqual(len([v for v in self.d.volumes.values() if 'speech-box.installation' in v['Labels']]),14)
        self.assertEqual(a.roots['models'].stat().st_ino,b.roots['models'].stat().st_ino)
        self.assertTrue(all(a.roots[r]!=b.roots[r] for r in dv.ROLES if r!='models'))
        self.assertNotEqual((a.roots['credential']/'control.key').read_bytes(),(b.roots['credential']/'control.key').read_bytes())
        a_job=self.retained_job(a,'alice'); b_job=self.retained_job(b,'bob')
        self.assertNotEqual(a_job['id'],b_job['id'])
        import sqlite3
        def jobs(s):
            with sqlite3.connect(s.roots['control']/'core/core.sqlite') as db: return list(db.execute('SELECT job,context FROM jobs'))
        b_jobs=jobs(b)
        before=sm.verify_tree(a.roots['models'],self.expected,time.monotonic()+10)
        self.selection.rename(self.f.root/'retained-original-selection.json')
        b_gate=copy.deepcopy(b.gate); b_files={p.relative_to(b.roots['authority']):p.read_bytes() for p in b.roots['authority'].iterdir()}
        self.assertEqual(self.startup(a)['registration_id'],a.gate['registration_id'])
        self.assertEqual(self.startup(b)['registration_id'],b.gate['registration_id'])
        self.command(0,'stop'); self.command(0,'up')
        self.assertEqual(b.gate,b_gate)
        self.assertEqual(jobs(b),b_jobs)
        self.assertEqual(b_files,{p.relative_to(b.roots['authority']):p.read_bytes() for p in b.roots['authority'].iterdir()})
        self.assertEqual(self.startup(a)['registration_id'],a.gate['registration_id'])
        self.assertEqual(self.startup(b)['registration_id'],b.gate['registration_id'])
        self.assertEqual(sm.verify_tree(a.roots['models'],self.expected,time.monotonic()+10),before)
        with self.assertRaises(ValueError): self.startup(b,a.gate)
        for field in ('configuration_digest','enrollment_digest'):
            bad=copy.deepcopy(a.gate); bad[field]='f'*64
            with self.subTest(field=field),self.assertRaises(ValueError): self.startup(a,bad)
        # Existing control-readable startup explicitly treats provenance_digest
        # as an opaque binding label; private provenance remains helper authority.
        bad=copy.deepcopy(a.gate); bad['provenance_digest']='f'*64
        self.assertEqual(self.startup(a,bad)['registration_id'],a.gate['registration_id'])
        bad=copy.deepcopy(a.gate); bad['observation']['binding']='f'*64
        with self.assertRaises(ValueError): self.startup(a,bad)
        bad=copy.deepcopy(a.gate); bad['observation']['local']['worker']['roots']['models'][1]+=1
        with self.assertRaises(ValueError): self.startup(a,bad)

    def catalog(self):
        return sm.Catalog(self.published['catalog']['Mountpoint'],self.d.frozen,self.published['catalog'],time.monotonic()+20)

    def test_catalog_authentication_and_unfiltered_unsafe_consumers(self):
        a=self.install('alice'); self.command(0,'up')
        with self.catalog().held(self.key) as store:
            authority=sm.Installations(store)
            readers=authority.readers()
            live=[c for c in self.d.snapshot(a.b)['consumers'] if c['State']['Running']]
            for c in live: authority.classify(c,readers)
            c=copy.deepcopy(live[0])
            for mutation in ('unknown','rw','requested-rw','alias','nonce','private-alias'):
                bad=copy.deepcopy(c)
                if mutation=='unknown': bad['Id']='f'*64
                elif mutation=='rw': next(m for m in bad['Mounts'] if m['Destination']=='/models')['RW']=True
                elif mutation=='requested-rw': next(m for m in bad['HostConfig']['Mounts'] if m['Target']=='/models')['ReadOnly']=False
                elif mutation=='alias': bad['Mounts'].append(dict(bad['Mounts'][0],Destination='/alias'))
                elif mutation=='nonce': bad['Config']['Cmd'][-1]='f'*32
                else: bad['Mounts'].append(dict(Type='bind',Source='/'+a.b['volumes']['control']['Mountpoint'],Destination='/hidden',RW=False))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError): authority.classify(bad,readers)
            # Remove real protected authority one record at a time. These are
            # disposable fixture records; restore only to exercise the next
            # independent missing-record predicate in the same catalog.
            created=next(n for n in (Path(store.catalog.path)/'references').iterdir() if n.name.startswith('install-') and n.name.endswith('-created.json'))
            for event in ('reserved','bound','enrolled',None):
                path=created if event is None else Path(store.catalog.path)/'references'/authority.name(a.b['installation'],event)
                retained=Path(store.catalog.path).parent/('retained-'+path.name)
                path.rename(retained)
                try:
                    with self.subTest(missing=event),self.assertRaises((ValueError,FileNotFoundError)): authority.readers()
                finally: retained.rename(path)
            # Root-protected records, rather than labels or a supplied b, grant
            # reader authority. A valid-shaped foreign digest still refuses.
            bound=Path(store.catalog.path)/'references'/authority.name(a.b['installation'],'bound')
            original=bound.read_bytes(); bad=dc.parse(original); bad['binding']['model_storage']['publication_sha256']='f'*64
            bound.write_bytes(dc.encoded(bad)); bound.chmod(0o600)
            with self.assertRaises(ValueError): authority.readers()
        # The daemon scan must discover aliases even with unknown volume names.
        bad=copy.deepcopy(c); bad['Id']='e'*64
        bad['Mounts']=[dict(Type='bind',Source='/'+a.b['volumes']['models']['Mountpoint']+'//bundles',Destination='/models',RW=False)]
        self.d.containers[bad['Id']]=bad
        self.assertIn(bad['Id'],{v['Id'] for v in self.d.snapshot(a.b)['consumers']})

    def test_invalid_shared_resources_refuse_before_reservation_or_allocation(self):
        config=self.configuration('alice')
        request=self.f.root/'bad-request'
        entries=[]
        for role,path,raw in [('config','install.json',dc.encoded(config)),('credential','control.key',b'inert-not-read')]:
            source=self.f.root/('bad-'+path); source.write_bytes(raw)
            entry=dict(role=role,path=path,source=str(source),bytes=len(raw))
            if role!='credential':entry['sha256']=hashlib.sha256(raw).hexdigest()
            entries.append(entry)
        request.write_bytes(dc.encoded(dict(configuration=config,helper_image=config['images']['control'],inputs=entries,
            cpu=dict(control=0,worker=2*10**9),tmpfs={r:dict(bytes=65536,inodes=64) for r in ('control','worker')},control_port=8765)))
        before=(set(self.d.containers),set(self.d.volumes))
        with self.assertRaises(ValueError): self.invoke(['install','--context','fixture','--request',str(request),
            '--handle',str(self.f.root/'invalid-handle'),'--shared-models',str(self.selection)])
        self.assertEqual((set(self.d.containers),set(self.d.volumes)),before)
        self.assertFalse((self.f.root/'invalid-handle').exists())

    def test_full_tamper_refuses_up_before_owner_creation(self):
        a=self.install('alice')
        payload=a.roots['models']/'bundles'/self.key/self.expected['files'][-1]['path']
        raw=payload.read_bytes(); payload.chmod(0o644); payload.write_bytes(raw[:-1]+bytes([raw[-1]^1])); payload.chmod(0o444)
        before={c['Id'] for c in self.d.containers.values() if c['Config']['User']!='0:0'}
        with self.assertRaisesRegex(ValueError,'shared_file_hash'): self.command(0,'up')
        after={c['Id'] for c in self.d.containers.values() if c['Config']['User']!='0:0'}
        self.assertEqual(before,after)
        self.assertFalse((a.roots['authority']/'containers.json').exists())
        self.assertTrue(list(a.roots['authority'].glob('operation-*.done')))

    def test_incomplete_shared_refuses_retry_and_store_recovery_cannot_enroll(self):
        with patch.object(dh.Helper,'initialize',side_effect=ValueError('inert_init_failure')):
            with self.assertRaisesRegex(ValueError,'inert_init_failure'): self.install('alice')
        s=next(iter(self.d.sessions.values())); handle=str(self.f.root/'handle-0'); self.handles=[handle]
        before={p.name:p.read_bytes() for p in s.roots['authority'].iterdir()}
        c=client(self.f.root,self.d,self.expected,{},99)
        self.assertEqual(c.run('recover-import',self.published['attempt']),self.published)
        self.assertEqual(before,{p.name:p.read_bytes() for p in s.roots['authority'].iterdir()})
        with self.assertRaisesRegex(ValueError,'daemon_incomplete_operation_retained'): self.command(0,'up')
        self.assertFalse((s.roots['authority']/'init-done.json').exists())
        self.assertFalse((s.roots['receipts']/'p1-init.done').exists())

    def test_legacy_packaged_journey_and_incomplete_refusal(self):
        a=self.install('alice',shared=False)
        self.assertNotIn('model_storage',a.b)
        self.assertEqual(len([v for v in self.d.volumes.values() if 'speech-box.installation' in v['Labels']]),8)
        self.command(0,'up'); self.assertEqual(self.startup(a)['registration_id'],a.gate['registration_id'])
        self.command(0,'stop'); self.command(0,'up')
        self.assertEqual(self.startup(a)['registration_id'],a.gate['registration_id'])
        with patch.object(dh.Helper,'runtime',side_effect=ValueError('inert_retained_failure')):
            with self.assertRaisesRegex(ValueError,'inert_retained_failure'): self.command(0,'status')
        with self.assertRaisesRegex(ValueError,'daemon_incomplete_operation_retained'): self.command(0,'up')
