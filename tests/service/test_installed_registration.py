"""Real temporary SQLite; injected UID, namespace, hello and host observations."""
import copy
from contextlib import ExitStack, closing
import fcntl
import json
import os
import sqlite3
import unittest
from unittest.mock import patch

from service import host_install as h, install_policy as ip, install_child
from service.core import encode
from service.persistent_init import digest
from tests.service import test_split_init as split_fixtures
from tests.service.test_install_child import Namespace


class Registration(unittest.TestCase):
    def new_split_fixture(self):
        fixture=split_fixtures.SplitInit();fixture.setUp();return fixture

    def setUp(self):
        self.fixture=self.new_split_fixture();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;f.run_init();self.roots=f.roots;self.scope=f.scope;self.config=f.config
        self.public=json.loads((self.roots['receipts']/'p1-init.json').read_bytes())['public']
        self.authority=f.root/'authority';self.authority.mkdir(mode=0o700)
        original_translate=f.translate
        def translate(info,path=None):
            if path is not None and (path==self.authority or self.authority in path.parents):
                from types import SimpleNamespace
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=0,st_gid=0))
            return original_translate(info,path)
        f.translate=translate
        self.fd=os.open(self.authority,os.O_RDONLY|os.O_DIRECTORY);self.addCleanup(os.close,self.fd)
        self.owners=[dict(id=str(i)*64,image=self.config['images'][role],role=role,
            started='2026-10-24T01:00:00Z',volumes=['control','ipc','staging'] if role=='control' else ['ipc','receipts','staging'])
            for i,role in ((1,'control'),(2,'worker'))]
        self.lane=json.loads((self.roots['receipts']/'journal.json').read_bytes())['payload']['lane']
        self.hello=dict(lane=self.lane,boot='b'*32,quarantined=False,launches=0)
        self.seen=[];self.children=[]
        test=self
        class Observer:
            scope=test.scope;config=test.config
            policy=dict(roots={r:dict(source=str(p),device=p.stat().st_dev,inode=p.stat().st_ino) for r,p in test.roots.items()})
            def current_owners(self,claims):
                return copy.deepcopy(claims)
            def observe_owners(self,owners,public):
                test.seen.append(copy.deepcopy(owners))
                return dict(topology=dict(consumers=[dict(c,running=True) for c in owners]),processes=[])
        self.observer=Observer()
        # Content validator is tested independently by installed-host-policy tests.
        # Here injected observations exercise the transaction with real copies/DB.
        self.enrollment_patch=patch.object(ip,'host_enrollment',return_value=digest(self.public))
        self.enterContext(self.enrollment_patch)
        self.provenance_patch=patch.object(h,'_installed_provenance',side_effect=lambda v,*_:v)
        self.enterContext(self.provenance_patch)


    def read_provenance(self,fd,name):
        return (self.authority/name).read_bytes()

    def child(self,kind,request,fds):
        self.children.append(kind)
        self.assertNotIn('seed',request);self.assertNotIn('seed',request.get('public',{}));self.assertNotIn('sql',request)
        with self.assertRaises(BlockingIOError):
            fd=os.open(self.roots['control']/'core/control.lock',os.O_RDONLY)
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:os.close(fd)
        if kind=='hello':return copy.deepcopy(self.hello)
        with ExitStack() as stack:
            for n in ('getuid','geteuid','getgid','getegid'):stack.enter_context(patch.object(os,n,return_value=10001))
            stack.enter_context(patch.object(os,'getgroups',return_value=[]))
            namespace=Namespace(self.fixture)
            with namespace.view():
                control,staging=install_child.prepare(request['public'],request['control_fd'],request['staging_fd'],ops=namespace)
                self.fixture.roots.update(control=control,staging=staging)
                self.fixture.fds[request["lock_fd"]]=control/"core/control.lock"
                install_child.registration_writer(request['public'],request['registration'],request['prestate'],control,request['lock_fd'],fault=getattr(self,'writer_fault',lambda _:None))
            self.assertEqual(namespace.events,['unshare','private','tmpfs','bind-control','bind-staging'])

    def run_registration(self,**kw):
        return h._register_installed(self.observer,self.owners,self.fd,lambda:None,child=self.child,**kw)

    def rows(self):
        with closing(sqlite3.connect(self.roots['control']/'core/core.sqlite')) as db:
            return db.execute('SELECT id,record,digest FROM p1_registrations').fetchall()

    def test_registration_and_exact_retry(self):
        rid=self.run_registration();self.assertEqual(len(self.rows()),1)
        raw=(self.authority/('registration-'+rid+'.json')).read_bytes()
        self.assertEqual(self.run_registration(),rid)
        self.assertEqual((self.authority/('registration-'+rid+'.json')).read_bytes(),raw)
        self.assertEqual(self.children,['hello','hello','writer','hello']*2)

    def interruption(self,point):
        def fault(where):
            if where==point:raise OSError(point)
        with self.assertRaisesRegex(OSError,point):self.run_registration(fault=fault)
        self.assertTrue(list(self.authority.glob('registration-*.json')))
        self.assertEqual(len(self.rows()),0 if point=='provenance' else 1)
        self.run_registration()

    def test_interrupt_preinsert(self):self.interruption('provenance')
    def test_interrupt_postinsert(self):self.interruption('insert')
    def test_interrupt_postbracket(self):self.interruption('bracket')

    def test_worker_boot_change_after_insert_refuses_retains(self):
        original=self.child
        def child(kind,request,fds):
            result=original(kind,request,fds)
            if kind=='writer':self.hello['boot']='c'*32
            return result
        self.child=child
        with self.assertRaisesRegex(ValueError,'hello_changed'):self.run_registration()
        self.assertEqual(len(self.rows()),1)

    def test_control_process_change_after_insert_refuses(self):
        original=self.observer.observe_owners
        def observe(*args):
            result=original(*args)
            if 'writer' in self.children:result['processes']=[dict(pid=999)]
            return result
        self.observer.observe_owners=observe
        with self.assertRaisesRegex(ValueError,'postbracket'):self.run_registration()
        self.assertEqual(len(self.rows()),1)

    def test_live_core_exclusion_before_child(self):
        fd=os.open(self.roots['control']/'core/control.lock',os.O_RDONLY)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'lock_busy'):self.run_registration()
        finally:os.close(fd)
        self.assertEqual(self.children,[])

    def test_wrong_enrollment_before_child(self):
        (self.roots['receipts']/'p1-enrolled.json').write_text('{}')
        with self.assertRaises(ValueError):self.run_registration()
        self.assertEqual(self.children,[])

    def test_wrong_candidate_scope_before_child(self):
        self.owners[0]['image']='bad'
        with self.assertRaises(ValueError):self.run_registration()
        self.assertEqual(self.children,[])

    def test_writer_stale_prestate_rolls_back(self):
        def child(kind,request,fds):
            if kind=='writer':request['prestate']=[['x','x','x']]
            return Registration.child(self,kind,request,fds)
        self.child=child
        with self.assertRaisesRegex(ValueError,'prestate_changed'):self.run_registration()
        self.assertEqual(self.rows(),[])

    def test_unreaped_retains_control_lock(self):
        def child(*args):raise h._RegistrationChildUnreaped('injected')
        self.child=child;start=len(h._registration_retained)
        try:
            with self.assertRaises(h._RegistrationChildUnreaped):self.run_registration()
            fd=os.open(self.roots['control']/'core/control.lock',os.O_RDONLY)
            try:
                with self.assertRaises(BlockingIOError):fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            finally:os.close(fd)
        finally:
            for fd in h._registration_retained[start:]:os.close(fd)
            del h._registration_retained[start:]

    def use_closed_observer(self):
        from tests.service import test_installed_host_policy as host_fixtures
        host=host_fixtures.InstalledHost();host.setUp();self.addCleanup(host.doCleanups)
        self.assertEqual(host.config,self.config)
        host.scope=self.scope;host.policy['host']=self.scope
        host.policy['roots']={r:dict(source=str(p),device=p.stat().st_dev,inode=p.stat().st_ino) for r,p in self.roots.items()}
        host.volumes=[dict(role=r,**self.scope['volumes'][r],**v) for r,v in host.policy['roots'].items()]
        for container in host.containers:
            for mount in container['mounts']:
                if mount['type']=='volume':mount['source']=str(self.roots[mount['name']])
        self.provenance_patch.__exit__(None,None,None);self.enrollment_patch.__exit__(None,None,None)
        self.observer=host.observer();self.observer.docker.context=self.scope["context"];self.owners=host.owners
        return host

    def test_real_closed_observer_transaction(self):
        self.use_closed_observer()
        with patch.object(h.sys,'platform','linux'):
            rid=self.run_registration()
            self.assertEqual(self.run_registration(),rid)
            self.assertEqual(len(self.rows()),1)

    def test_complete_union_includes_orphan_provenance(self):
        old=copy.deepcopy(self.owners)
        old[0]['id']='3'*64;old[1]['id']='4'*64
        registration=dict(deployment_id=self.public['deployment_id'],scope=self.scope,containers=old,
            lane=self.lane,boot='a'*32,evidence='actual-host-inspection')
        value=dict(kind='installed-registration-v1',registration=registration,hello=dict(self.hello,boot='a'*32),
            observation={},ipc_identity=[])
        (self.authority/('registration-'+digest(registration)+'.json')).write_text(encode(value))
        (self.authority/('registration-'+digest(registration)+'.json')).chmod(0o600)
        self.run_registration()
        self.assertEqual({c['id'] for c in self.seen[0]},set('1234'[i]*64 for i in range(4)))
        self.assertEqual(len(self.rows()),1)

    def test_provenance_tamper_refuses_before_writer(self):
        self.run_registration();self.children=[]
        path=next(self.authority.glob('registration-*.json'))
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'provenance_changed'):self.run_registration()
        self.assertEqual(self.children,[])

    def replace_provenance(self):
        path=next(self.authority.glob('registration-*.json'))
        before=path.stat().st_ino
        replacement=self.authority/'replacement'
        replacement.write_bytes(path.read_bytes());replacement.chmod(0o600)
        os.replace(replacement,path)
        self.assertNotEqual(path.stat().st_ino,before)

    def retry_replacement(self,seam):
        rid=self.run_registration();self.children=[];returned=[]
        if seam=='hello':
            original=self.child
            def child(kind,request,fds):
                result=original(kind,request,fds)
                if self.children==['hello']:self.replace_provenance()
                return result
            self.child=child
        elif seam=='observation':
            original=self.observer.observe_owners;replaced=[]
            def observe(*args):
                result=original(*args)
                if not replaced:
                    self.replace_provenance();replaced.append(True)
                return result
            self.observer.observe_owners=observe
        else:
            original=h._registration_read_at;replaced=[]
            def read(*args):
                result=original(*args)
                if not replaced:
                    self.replace_provenance();replaced.append(True)
                return result
            self.enterContext(patch.object(h,'_registration_read_at',side_effect=read))
        with self.assertRaisesRegex(ValueError,'provenance_changed'):
            returned.append(self.run_registration())
        self.assertEqual(returned,[]);self.assertNotIn('writer',self.children)
        self.assertEqual([r[0] for r in self.rows()],[rid])

    def test_retry_identical_replacement_during_prewrite_hello_refuses(self):
        self.retry_replacement('hello')

    def test_retry_identical_replacement_during_observation_refuses(self):
        self.retry_replacement('observation')

    def test_retry_identical_replacement_between_read_and_baseline_refuses(self):
        self.retry_replacement('read')

    def test_created_provenance_replacement_before_path_check_refuses(self):
        original=os.fsync;replaced=[];returned=[]
        def fsync(fd):
            result=original(fd)
            if fd==self.fd and not replaced:
                self.replace_provenance();replaced.append(True)
            return result
        with patch.object(os,'fsync',side_effect=fsync):
            with self.assertRaisesRegex(ValueError,'provenance_changed'):
                returned.append(self.run_registration())
        self.assertEqual(returned,[]);self.assertNotIn('writer',self.children)
        self.assertEqual(self.rows(),[]);self.assertTrue(replaced)

    def test_provenance_identical_replacement_during_writer_refuses(self):
        original=self.child
        def child(kind,request,fds):
            result=original(kind,request,fds)
            if kind=='writer':
                path=next(self.authority.glob('registration-*.json'))
                replacement=self.authority/'replacement';replacement.write_bytes(path.read_bytes());replacement.chmod(0o600)
                os.replace(replacement,path)
            return result
        self.child=child
        with self.assertRaisesRegex(ValueError,'provenance_changed'):self.run_registration()
        self.assertEqual(len(self.rows()),1)

    def writer_interruption(self,point):
        def fault(where):
            if where==point:raise OSError(point)
        self.writer_fault=fault
        with self.assertRaisesRegex(OSError,point):self.run_registration()
        self.assertEqual(len(self.rows()),int(point=='committed'))
        del self.writer_fault
        self.run_registration()

    def test_writer_preinsert_rollback(self):self.writer_interruption('preinsert')
    def test_writer_inserted_rollback(self):self.writer_interruption('inserted')
    def test_writer_precommit_rollback(self):self.writer_interruption('precommit')
    def test_writer_committed_retains(self):self.writer_interruption('committed')

    def test_writer_existing_database_only(self):
        original=self.child
        def child(kind,request,fds):
            if kind=='writer':(self.roots['control']/'core/core.sqlite').rename(self.roots['control']/'core/retained.sqlite')
            return original(kind,request,fds)
        self.child=child
        with self.assertRaises(FileNotFoundError):self.run_registration()
        self.assertFalse((self.roots['control']/'core/core.sqlite').exists())

    def test_writer_wrong_installation_transaction_refuses(self):
        original=self.child
        def child(kind,request,fds):
            if kind=='writer':request['public']['deployment_id']='f'*32
            return original(kind,request,fds)
        self.child=child
        with self.assertRaisesRegex(ValueError,'registration_binding'):self.run_registration()
        self.assertEqual(self.rows(),[])

    def test_parent_only_connects_private_copy(self):
        original=sqlite3.connect;opened=[]
        def connect(path,*args,**kw):
            opened.append((str(path),os.getuid()))
            return original(path,*args,**kw)
        with patch.object(sqlite3,'connect',side_effect=connect):self.run_registration()
        originals=[(p,u) for p,u in opened if 'namespace/mnt/pab-install/control/core/core.sqlite' in p]
        self.assertEqual(len(originals),1);self.assertEqual(originals[0][1],10001)
        self.assertTrue(all('p1-inspect-' in p or (p,u) in originals for p,u in opened))

    def test_writer_entry_prepares_fixed_paths_before_drop(self):
        import io
        import sys
        from types import SimpleNamespace
        fds=[os.open(self.roots[r],os.O_RDONLY|os.O_DIRECTORY) for r in ('control','staging')]
        lock=os.open(self.roots['control']/'core/control.lock',os.O_RDONLY);fds.append(lock)
        for fd in fds:self.addCleanup(os.close,fd)
        request=dict(public=self.public,registration=dict(deployment_id=self.public['deployment_id'],scope=self.scope,
            containers=self.owners,lane=self.lane,boot='b'*32,evidence='actual-host-inspection'),prestate=[],
            control_fd=fds[0],staging_fd=fds[1],lock_fd=fds[2])
        events=[]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys,'platform','linux'))
            stack.enter_context(patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(encode(request).encode()))))
            for name in ('getuid','geteuid'):stack.enter_context(patch.object(os,name,return_value=0))
            def prepare(*args):events.append('prepare');return install_child.CONTROL,install_child.STAGING
            stack.enter_context(patch.object(install_child,'prepare',side_effect=prepare))
            stack.enter_context(patch.object(install_child,'drop_control',side_effect=lambda:events.append('drop')))
            writer=stack.enter_context(patch.object(install_child,'registration_writer',side_effect=lambda *args:events.append('writer')))
            install_child.registration_entry('writer')
            self.assertEqual(events,['prepare','drop','writer'])
            self.assertEqual(writer.call_args.args[3],install_child.CONTROL)

    def test_writer_sidecar_seam(self):
        observed=[]
        def fault(where):
            if where=='inserted':
                for name in ('core.sqlite-wal','core.sqlite-shm'):
                    info=(self.roots['control']/'core'/name).stat()
                    self.assertEqual((info.st_uid,info.st_gid),(10001,10001))
                    self.assertEqual(info.st_mode & 0o777,0o600)
                    observed.append(name)
        self.writer_fault=fault;self.run_registration()
        self.assertEqual(len(observed),2)

    def test_unknown_authority_file_refuses(self):
        (self.authority/'unknown').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'unknown_authority'):self.run_registration()
        self.assertEqual(self.children,[])


class ChildTransport(unittest.TestCase):
    def test_finite_kill_reap_failure_retains_child(self):
        import subprocess
        from unittest.mock import Mock
        child=Mock();child.communicate.side_effect=subprocess.TimeoutExpired('injected',10)
        child.wait.side_effect=subprocess.TimeoutExpired('injected',2)
        start=len(h._registration_retained)
        try:
            with patch.object(h.sys,'platform','linux'),patch.object(os,'getuid',return_value=0),patch.object(os,'geteuid',return_value=0),patch.object(h.subprocess,'Popen',return_value=child) as spawn:
                with self.assertRaises(h._RegistrationChildUnreaped):h._registration_child('hello',dict(ipc_fd=9),(9,))
            child.kill.assert_called_once();child.wait.assert_called_once_with(timeout=2)
            self.assertIs(h._registration_retained[-1],child)
            self.assertEqual(spawn.call_args.kwargs['pass_fds'],(9,))
            self.assertEqual(spawn.call_args.kwargs['env'],{})
            self.assertEqual(spawn.call_args.args[0][1:3],['-I','-B'])
        finally:del h._registration_retained[start:]

    def test_timeout_reaped_still_refuses(self):
        import subprocess
        from unittest.mock import Mock
        child=Mock();child.communicate.side_effect=subprocess.TimeoutExpired('injected',10)
        child.wait.return_value=-9
        with patch.object(h.sys,'platform','linux'),patch.object(os,'getuid',return_value=0),patch.object(os,'geteuid',return_value=0),patch.object(h.subprocess,'Popen',return_value=child):
            with self.assertRaises(subprocess.TimeoutExpired):h._registration_child('hello',dict(ipc_fd=9),(9,))
        child.wait.assert_called_once_with(timeout=2)
