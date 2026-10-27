"""Real disposable SQLite/files, injected OS UID boundary. NO actual Linux UID proof."""
import sys
import copy
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from service import install_policy as ip, persistent_init as init, recovery, install_child
from tests.service.test_install_child import Namespace
from service.admission import DEFAULT_BUDGET
from service.core import encode
from tests.service.test_engine_admission import configuration as engine_configuration
from tests.service import test_provisioning as provisioning_fixtures


class SplitInit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=getattr(self,'temporary_parent','/private/tmp' if sys.platform == 'darwin' else '/tmp')); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.roots={r:self.root/r for r in ip.ROLES}
        for p in self.roots.values(): p.mkdir(mode=0o700)
        self.scope=dict(daemon='injected-local-linux',context='injected-context',volumes={r:dict(name=r,driver='local',created='now') for r in ip.ROLES})
        fixture=provisioning_fixtures.ProvisioningTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        context,profiles=engine_configuration(fixture.expected)
        spec=context['execution']['descriptor']; bid=spec['bundle_id']
        self.config=dict(version=1,ownership_policy=ip.POLICY,profiles=profiles,budget=copy.deepcopy(DEFAULT_BUDGET),
            contexts={'alice':context},bundles={bid:dict(manifest_sha256=spec['manifest_sha256'],checkpoint=spec['checkpoint'],revision=spec['revision'],path='/models/bundles/'+bid)},
            images=dict(control='sha256:'+'1'*64,worker='sha256:'+'2'*64),
            mounts=dict(control='/control',staging='/staging',receipts='/receipts',ipc='/ipc',models='/models'))
        self.virtual={}; self.special={}; self.fds={}; self.calls=[]
        self.real_chmod=os.fchmod;self.real_stat=os.stat;self.real_fstat=os.fstat;self.real_open=os.open;self.real_close=os.close
        for p in self.roots.values():
            s=self.real_stat(p);self.virtual[(s.st_dev,s.st_ino)]=(0,0)
        stack=self.enterContext(ExitStack())
        stack.enter_context(patch.object(os,'open',side_effect=self.open))
        stack.enter_context(patch.object(os,'close',side_effect=self.close))
        stack.enter_context(patch.object(os,'stat',side_effect=self.stat))
        stack.enter_context(patch.object(os,'fstat',side_effect=self.fstat))
        stack.enter_context(patch.object(os,'fchown',side_effect=self.chown))
        stack.enter_context(patch.object(os,'fchmod',side_effect=self.chmod))

    def translate(self,s,path=None):
        owner=self.virtual.get((s.st_dev,s.st_ino))
        if owner is None and path is not None and (path==self.roots['control'] or self.roots['control'] in path.parents):owner=(10001,10001)
        if owner is None:return s
        return SimpleNamespace(**dict({k:getattr(s,k) for k in dir(s) if k.startswith('st_')},st_uid=owner[0],st_gid=owner[1],st_mode=s.st_mode|self.special.get((s.st_dev,s.st_ino),0)))

    def open(self,path,flags,*args,**kwargs):
        fd=self.real_open(path,flags,*args,**kwargs)
        p=Path(path)
        if not p.is_absolute():p=self.fds.get(kwargs.get('dir_fd'),Path.cwd())/p
        self.fds[fd]=p;return fd

    def close(self,fd):
        self.fds.pop(fd,None);return self.real_close(fd)

    def stat(self,path,*args,**kwargs):
        s=self.real_stat(path,*args,**kwargs)
        if isinstance(path,int):return self.translate(s,self.fds.get(path))
        p=Path(path)
        if not p.is_absolute():p=self.fds.get(kwargs.get('dir_fd'),Path.cwd())/p
        return self.translate(s,p)

    def fstat(self,fd):return self.translate(self.real_fstat(fd),self.fds.get(fd))
    def chmod(self,fd,mode):
        self.real_chmod(fd,mode);s=self.real_fstat(fd);self.special[(s.st_dev,s.st_ino)]=mode & 0o7000

    def chown(self,fd,uid,gid):
        s=self.real_fstat(fd);self.virtual[(s.st_dev,s.st_ino)]=(uid,gid)

    def authority(self):
        return dict(host=copy.deepcopy(self.scope),roots={r:[p.stat().st_dev,p.stat().st_ino] for r,p in self.roots.items()},consumers=[])

    def child(self,raw):
        request=json.loads(raw);self.assertEqual(set(request),{'public','control_fd','staging_fd'})
        self.assertNotIn('seed',request);self.assertNotIn('seed',request['public']);self.calls.append(raw)
        old=os.umask(0o077)
        try:
            with ExitStack() as identity:
                for name in ('getuid','geteuid','getgid','getegid'):identity.enter_context(patch.object(os,name,return_value=10001))
                identity.enter_context(patch.object(os,'getgroups',return_value=[]))
                namespace = Namespace(self)
                with namespace.view():
                    control, staging = install_child.prepare(request['public'], request['control_fd'], request['staging_fd'], ops=namespace)
                    self.roots.update(control=control, staging=staging)
                    init._split_core(request['public'], control, staging)
                self.assertEqual(namespace.events, ['unshare','private','tmpfs','bind-control','bind-staging'])
        finally:os.umask(old)

    def run_init(self,**kwargs):
        return init.initialize_split(self.roots,self.scope,self.config,authority=self.authority,policy=ip.SPLIT,launch_child=self.child,**kwargs)

    def bytes(self):return {str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
    def completion(self):return init.digest(json.loads((self.roots['receipts']/'p1-init.json').read_bytes()))
    def crash(self,point):
        def fault(where):
            if where==point:raise OSError('injected '+point)
        with self.assertRaisesRegex(OSError,'injected'):self.run_init(fault=fault)

    def adapter_request(self):
        self.crash('core_intent')
        public=json.loads((self.roots['receipts']/'p1-init.json').read_bytes())['public']
        fds=[self.real_open(self.roots[r],os.O_RDONLY|os.O_DIRECTORY) for r in ('control','staging')]
        for fd,role in zip(fds,('control','staging')):
            self.fds[fd]=self.roots[role];self.addCleanup(self.close,fd)
        return public,fds

    def test_linux_entry_prepares_before_permanent_drop(self):
        import io
        import sys
        public,fds=self.adapter_request()
        request=dict(public=public,control_fd=fds[0],staging_fd=fds[1])
        events=[]
        namespace=Namespace(self)
        prepare=install_child.prepare
        def setup(*args):
            events.append('prepare')
            return prepare(*args,ops=namespace)
        with namespace.view(), ExitStack() as stack:
            stack.enter_context(patch.object(sys,'platform','linux'))
            stack.enter_context(patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(encode(request).encode()))))
            for name in ('getuid','geteuid'):stack.enter_context(patch.object(os,name,return_value=0))
            for name,args in (('setgroups',[]),('setresgid',(10001,)*3),('setresuid',(10001,)*3)):
                def call(*values,n=name):events.append((n,values))
                stack.enter_context(patch.object(os,name,create=True,side_effect=call))
            for name in ('getresuid','getresgid'):stack.enter_context(patch.object(os,name,create=True,return_value=(10001,)*3))
            stack.enter_context(patch.object(os,'getgroups',return_value=[]))
            stack.enter_context(patch.object(os,'umask',side_effect=lambda mode:events.append(('umask',mode))))
            stack.enter_context(patch.object(install_child,'prepare',side_effect=setup))
            core=stack.enter_context(patch.object(init,'_split_core'))
            init._linux_split_child()
            core.assert_called_once_with(public,install_child.CONTROL,install_child.STAGING)
        self.assertEqual(events,['prepare',('setgroups',([],)),('setresgid',(10001,)*3),('setresuid',(10001,)*3),('umask',0o077)])

    def test_linux_entry_setup_failure_never_drops_or_opens_core(self):
        import io
        import sys
        public,fds=self.adapter_request()
        request=dict(public=public,control_fd=fds[0],staging_fd=fds[1])
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys,'platform','linux'))
            stack.enter_context(patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(encode(request).encode()))))
            for name in ('getuid','geteuid'):stack.enter_context(patch.object(os,name,return_value=0))
            drops=[stack.enter_context(patch.object(os,name,create=True)) for name in ('setgroups','setresgid','setresuid')]
            stack.enter_context(patch.object(install_child,'prepare',side_effect=OSError('namespace denied')))
            core=stack.enter_context(patch.object(init,'_split_core'))
            with self.assertRaisesRegex(OSError,'namespace denied'):init._linux_split_child()
            for drop in drops:drop.assert_not_called()
            core.assert_not_called()

    def test_adapter_sqlite_canonical_database_wal_shm(self):
        public,fds=self.adapter_request()
        namespace=Namespace(self)
        with namespace.view():
            control,staging=install_child.prepare(public,*fds,ops=namespace)
            database=control/'probe.sqlite'
            import sqlite3
            db=sqlite3.connect(database)
            try:
                db.execute('pragma journal_mode=wal')
                db.execute('create table probe(value)');db.commit()
                self.assertEqual(db.execute('pragma database_list').fetchone()[2],str(database))
                for suffix in ('','-wal','-shm'):self.assertTrue(Path(str(database)+suffix).is_file())
                self.assertEqual(control.stat().st_ino,public['roles']['control']['inode'])
            finally:db.close()

    def test_adapter_failures_stop_before_core(self):
        public,fds=self.adapter_request()
        before=self.bytes()
        for point in ('unshare','private','tmpfs','bind-control','bind-staging'):
            namespace=Namespace(self,fail=point)
            with namespace.view(),self.assertRaisesRegex(OSError,'injected'):
                install_child.prepare(public,*fds,ops=namespace)
            self.assertEqual(namespace.events[-1],point)
        self.assertEqual(before,self.bytes())

    def test_adapter_refuses_changed_fd_and_duplicate_before_namespace(self):
        public,fds=self.adapter_request()
        namespace=Namespace(self)
        bad=copy.deepcopy(public);bad['roles']['control']['inode']+=1
        for record,descriptors in ((bad,fds),(public,[fds[0],fds[0]]),(public,[True,fds[1]])):
            with self.assertRaises(ValueError):install_child.prepare(record,*descriptors,ops=namespace)
        self.assertEqual(namespace.events,[])

    def test_adapter_refuses_unsafe_mountpoint_and_bind_identity(self):
        public,fds=self.adapter_request()
        namespace=Namespace(self)
        with namespace.view():
            with patch.object(namespace,'stat',return_value=SimpleNamespace(st_uid=0,st_mode=0o120777)):
                with self.assertRaisesRegex(ValueError,'unsafe_initializer_mountpoint'):
                    install_child.prepare(public,*fds,ops=namespace)
            self.assertEqual(namespace.events,[])
            original=namespace.stat
            def changed(path):
                if path != namespace.base:return SimpleNamespace(st_dev=-1,st_ino=-1,st_mode=0o40700)
                return original(path)
            with patch.object(namespace,'stat',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'bind_identity_changed'):
                    install_child.prepare(public,*fds,ops=namespace)

    def test_fresh_closed_records_private_copy_inspection_same_bytes_reopen(self):
        result=self.run_init();self.assertEqual(len(self.calls),1)
        public=json.loads((self.roots['receipts']/'p1-init.json').read_bytes())['public']
        self.assertEqual(public['scope'],ip.SCOPE);self.assertEqual(public['configuration_digest'],init.digest(self.config))
        for role,r in public['roles'].items():
            self.assertEqual((r['uid'],r['gid'],r['mode']),ip.expectation(ip.SPLIT,role,directory=True))
        before=self.bytes()
        for _ in range(2):
            state=recovery.inspect(self.roots,self.scope,policy=ip.SPLIT)
            self.assertEqual(state['deployment_id'],result['deployment_id']);self.assertEqual(state['inventory']['reasons'],[])
            self.assertEqual(state['registrations'],[]);self.assertFalse(state['apply_available'])
        self.assertEqual(before,self.bytes())
        with self.assertRaisesRegex(ValueError,'already_enrolled'):self.run_init(complete=result['initialization_digest'])
        self.assertEqual(before,self.bytes())
        with self.assertRaises(ValueError):recovery.register(self.roots,self.scope,{})
        with self.assertRaises(ValueError):recovery.enrollment_gate(self.roots['control']/'core','x',{},self.config['profiles'],self.config['budget'])

    def test_deterministic_completion_before_core(self):
        self.crash('lifecycle.key');before=self.bytes();token=self.completion()
        self.run_init(complete=token)
        self.assertEqual(len(self.calls),1)
        for k,v in before.items():self.assertEqual(self.bytes()[k],v)

    def test_missing_attempted_file_never_reminted(self):
        self.crash('intent_lifecycle.key');before=self.bytes()
        with self.assertRaisesRegex(ValueError,'missing_initialized_file_quarantine'):self.run_init(complete=self.completion())
        self.assertEqual(before,self.bytes());self.assertFalse(self.calls)

    def test_ambiguous_core_never_reconstructed(self):
        self.crash('core_intent');before=self.bytes()
        with self.assertRaisesRegex(ValueError,'partial_core_quarantine'):self.run_init(complete=self.completion())
        self.assertEqual(before,self.bytes());self.assertFalse(self.calls)

    def test_reaped_but_unmarked_core_quarantined(self):
        self.crash('child_reaped');before=self.bytes()
        with self.assertRaisesRegex(ValueError,'partial_core_quarantine'):self.run_init(complete=self.completion())
        self.assertEqual(len(self.calls),1);self.assertEqual(before,self.bytes())

    def test_validated_database_completion_does_not_run_child_twice(self):
        self.crash('database');before=self.bytes();self.run_init(complete=self.completion())
        self.assertEqual(len(self.calls),1)
        for k,v in before.items():self.assertEqual(self.bytes()[k],v)

    def test_config_tamper_and_closed_records(self):
        self.crash('init_record');before=self.bytes();token=self.completion()
        self.config['images']['worker']='sha256:'+'3'*64
        with self.assertRaisesRegex(ValueError,'scope_mismatch'):self.run_init(complete=token)
        self.assertEqual(before,self.bytes())
        for mutate in (lambda c:c.update(extra=1),lambda c:c['mounts'].update(control='/other'),
                       lambda c:c['images'].update(control='latest'),lambda c:c['contexts']['alice']['execution']['descriptor'].update(revision='other'),
                       lambda c:c['budget'].update(pids=1)):
            c=copy.deepcopy(self.config);mutate(c)
            with self.assertRaises(Exception):ip.configuration(c)

    def test_live_directory_exclusion_and_authority(self):
        with ExitStack() as stack:
            fd=os.open(self.roots['control'],os.O_RDONLY|os.O_DIRECTORY);stack.callback(os.close,fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'initialization_lock_busy'):self.run_init()
        with patch.object(self,'authority',return_value=dict(host=self.scope,roots={},consumers=['unknown'])):
            with self.assertRaisesRegex(ValueError,'fresh_host_authority'):self.run_init()
        self.assertEqual(self.bytes(),{})

    def test_inode_exchange_and_partial_bytes_retained(self):
        self.crash('execution.key');token=self.completion()
        path=self.roots['ipc']/'execution.key';path.write_bytes(b'partial');before=self.bytes()
        with self.assertRaises(ValueError):self.run_init(complete=token)
        self.assertEqual(before,self.bytes())
        old=self.roots['staging'];old.rename(self.root/'retained');old.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError,'scope_mismatch'):self.run_init(complete=token)

    def test_child_failure_retained_and_no_enrollment(self):
        def failed(raw):
            core=self.roots['control']/'core';core.mkdir(mode=0o700)
            (core/'core.sqlite').write_bytes(b'partial SQLite')
            raise ValueError('injected child failure after reap')
        with self.assertRaisesRegex(ValueError,'child failure'):
            init.initialize_split(self.roots,self.scope,self.config,authority=self.authority,policy=ip.SPLIT,launch_child=failed)
        before=self.bytes()
        with self.assertRaisesRegex(ValueError,'partial_core_quarantine'):self.run_init(complete=self.completion())
        self.assertEqual(before,self.bytes());self.assertFalse((self.roots['receipts']/'p1-enrolled.json').exists())

    def test_legacy_unchanged_and_fresh_unknown_refused(self):
        (self.roots['control']/'old').write_bytes(b'legacy')
        before=self.bytes()
        with self.assertRaisesRegex(ValueError,'new_empty'):self.run_init()
        self.assertEqual(before,self.bytes())

    def test_original_sqlite_writes_only_inside_injected_child(self):
        import sqlite3
        real=sqlite3.connect; opened=[]; inside=False
        child=self.child
        def launch(raw):
            nonlocal inside
            inside=True
            try:child(raw)
            finally:inside=False
        def connect(path,*args,**kw):
            opened.append((str(path),inside));return real(path,*args,**kw)
        with patch.object(sqlite3,'connect',side_effect=connect):
            init.initialize_split(self.roots,self.scope,self.config,authority=self.authority,policy=ip.SPLIT,launch_child=launch)
            recovery.inspect(self.roots,self.scope,policy=ip.SPLIT)
        originals=[(p,c) for p,c in opened if p.endswith('/control/core/core.sqlite')]
        self.assertEqual(len(originals),1);self.assertTrue(originals[0][1])
        self.assertTrue(all(c or p == ':memory:' or 'p1-inspect-' in p for p,c in opened))
        for p in (self.roots['control']/'core').glob('core.sqlite*'):
            self.assertEqual((p.stat().st_uid,p.stat().st_gid),(10001,10001))

    def test_core_live_lock_refuses_completion_without_changes(self):
        self.crash('database');token=self.completion();before=self.bytes()
        with open(self.roots['control']/'core'/'control.lock','rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'maintenance_lock_busy'):self.run_init(complete=token)
        self.assertEqual(before,self.bytes());self.assertEqual(len(self.calls),1)

    def test_parent_exclusions_held_at_child_boundary(self):
        def launch(raw):
            for path in (self.roots['control'],self.roots['receipts'],self.roots['receipts']/'lane.lock'):
                fd=os.open(path,os.O_RDONLY)
                try:
                    with self.assertRaises(BlockingIOError):fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                finally:os.close(fd)
            self.child(raw)
        init.initialize_split(self.roots,self.scope,self.config,authority=self.authority,policy=ip.SPLIT,launch_child=launch)

    def test_forged_done_cannot_validate_partial_schema(self):
        import sqlite3
        self.crash('child_reaped');token=self.completion()
        public=json.loads((self.roots['receipts']/'p1-init.json').read_bytes())['public']
        with sqlite3.connect(self.roots['control']/'core'/'core.sqlite') as db:db.execute('DROP TABLE dispatches')
        init._split_create(self.roots['receipts']/'p1-core-done',init.digest(public).encode(),'receipts')
        before=self.bytes()
        with self.assertRaisesRegex(ValueError,'partial_database'):self.run_init(complete=token)
        self.assertEqual(before,self.bytes());self.assertEqual(len(self.calls),1)

    def test_closed_installation_and_config_tamper_refused_on_inspection(self):
        self.run_init();public=json.loads((self.roots['receipts']/'p1-init.json').read_bytes())['public']
        for field in ('scope','ownership_policy','configuration_digest','roles'):
            changed=copy.deepcopy(public);changed[field]='forged'
            with self.subTest(field=field),self.assertRaises(ValueError):ip.installation(changed)
        changed=copy.deepcopy(public);changed['configuration']['contexts']['alice']['execution']['descriptor']['revision']='changed'
        with self.assertRaises(Exception):ip.installation(changed)
        path=self.roots['receipts']/'p1-init.json';value=json.loads(path.read_bytes())
        value['public']['configuration']['images']['control']='sha256:'+'f'*64
        path.write_text(encode(value));before=self.bytes()
        with self.assertRaises(ValueError):recovery.inspect(self.roots,self.scope,policy=ip.SPLIT)
        self.assertEqual(before,self.bytes())

    def test_symlink_alias_and_policy_refuse_before_create(self):
        bad=dict(self.roots);bad['control']=self.root/'alias';bad['control'].symlink_to(self.roots['control'],target_is_directory=True)
        with self.assertRaises(OSError):
            init.initialize_split(bad,self.scope,self.config,authority=self.authority,policy=ip.SPLIT,launch_child=self.child)
        bad=dict(self.roots);bad['ipc']=bad['control']
        with self.assertRaisesRegex(ValueError,'distinct_non_nested'):
            init.initialize_split(bad,self.scope,self.config,authority=self.authority,policy=ip.SPLIT,launch_child=self.child)
        with self.assertRaises(ValueError):
            init.initialize_split(self.roots,self.scope,self.config,authority=self.authority,policy=ip.POLICY,launch_child=self.child)
        self.assertEqual(self.bytes(),{})

    def test_role_exchange_after_child_refuses_enrollment(self):
        def launch(raw):
            self.child(raw)
            self.roots['control'].rename(self.root/'retained-control')
            self.roots['control'].mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError,'installed_root_changed'):
            init.initialize_split(self.roots,self.scope,self.config,authority=self.authority,policy=ip.SPLIT,launch_child=launch)
        self.assertTrue((self.root/'retained-control'/'core'/'core.sqlite').exists())
        self.assertFalse((self.roots['receipts']/'p1-enrolled.json').exists())

    def test_changed_initialization_bytes_refuse_completion(self):
        self.crash('init_record');token=self.completion()
        path=self.roots['receipts']/'p1-init.json';path.write_bytes(path.read_bytes()+b'\n');before=self.bytes()
        with self.assertRaisesRegex(ValueError,'matching_initialization'):self.run_init(complete=token)
        self.assertEqual(before,self.bytes())

    def test_final_config_tamper_never_enrolled(self):
        def fault(point):
            if point=='database':
                path=self.roots['receipts']/'p1-init.json'
                path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'initialization_record_changed'):self.run_init(fault=fault)
        self.assertFalse((self.roots['receipts']/'p1-enrolled.json').exists())

    def test_fsync_failure_retains_intent_without_remint(self):
        real=os.fsync;count=0
        def failed(fd):
            nonlocal count
            count+=1
            if count==1:raise OSError('injected fsync')
            return real(fd)
        with patch.object(os,'fsync',side_effect=failed),self.assertRaisesRegex(OSError,'injected fsync'):self.run_init()
        before=self.bytes();self.assertIn('receipts/p1-init.json',before)
        with self.assertRaisesRegex(ValueError,'new_empty'):self.run_init()
        self.assertEqual(before,self.bytes());self.assertFalse(self.calls)

    def test_enrolled_fault_never_allows_completion(self):
        self.crash('enrolled');before=self.bytes()
        with self.assertRaisesRegex(ValueError,'already_enrolled'):self.run_init(complete=self.completion())
        self.assertEqual(before,self.bytes());self.assertEqual(len(self.calls),1)

    def test_core_writer_rejects_root_before_construction(self):
        with patch.object(init,'Core',side_effect=AssertionError('must not construct')):
            with self.assertRaisesRegex(ValueError,'permanent_control_identity'):
                init._split_core({},self.roots['control'],self.roots['staging'])


def _completion_case(point):
    def case(self):
        self.crash(point);before=self.bytes();self.run_init(complete=self.completion())
        for k,v in before.items():self.assertEqual(self.bytes()[k],v)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(recovery.inspect(self.roots,self.scope,policy=ip.SPLIT)['inventory']['reasons'],[])
    return case


for _point in ('init_record','role_control','role_receipts','role_ipc','role_staging',
               'execution.key','journal.json','initialized','lane.lock'):
    setattr(SplitInit,'test_completion_at_'+_point.replace('.','_'),_completion_case(_point))


if __name__=='__main__':unittest.main()
