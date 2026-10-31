"""SOURCE ONLY: real disposable files/SQLite; injected privilege/host/hello.

No Docker, namespace, UID transition, listener, model inference or guard runner.
"""
import copy
import io
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from service import entrypoint as e, recovery, host_install as h, install_policy as ip
from service.core import encode
from service.persistent_init import digest
from tests.service import test_installed_registration as fixtures


class GateTests(unittest.TestCase):
    def new_registration(self):
        fixture=fixtures.Registration();fixture.setUp();return fixture

    def setUp(self):
        f = self.fixture = self.new_registration() if hasattr(self,'new_registration') else GateTests.new_registration(self)
        self.addCleanup(f.doCleanups)
        self.rid = f.run_registration()
        record = json.loads(f.rows()[0][1])
        self.processes = [dict(id=c['id'],image=c['image'],started=c['started'],pid=90+i,
            start_ticks=100+i,pid_namespace=200+i,cgroup='/docker/'+c['id'])
            for i,c in enumerate(record['containers'])]
        self.gate = dict(version=1,registration_id=self.rid,registration=record,
            configuration_digest=digest(f.config),enrollment_digest=digest(f.public),
            provenance_digest='e'*64,processes=self.processes)
        self.incarnation = dict(start_ticks=100,pid_namespace=200,cgroup='0::/')
        self.root = f.roots['control']/'core'

    def read(self, **kwargs):
        return recovery.installed_startup_read(self.root,kwargs.get('config',self.fixture.config),
            kwargs.get('gate',self.gate),kwargs.get('hello',self.fixture.hello),
            kwargs.get('incarnation',self.incarnation),policy=ip.SPLIT)

    def test_exact_private_copy_gate(self):
        original = sqlite3.connect; paths = []
        def connect(path,*a,**kw):
            paths.append(str(path)); self.assertNotIn(str(self.root),str(path))
            return original(path,*a,**kw)
        before = self.fixture.rows()
        with patch.object(sqlite3,'connect',side_effect=connect): result = self.read()
        self.assertEqual(result['registration_id'],self.rid)
        self.assertEqual(before,self.fixture.rows()); self.assertTrue(paths)

    def test_wrong_config(self):
        config=copy.deepcopy(self.fixture.config);config['images']['control']='sha256:'+'a'*64
        with self.assertRaisesRegex(ValueError,'configuration'): self.read(config=config)

    def test_wrong_scope(self):
        public = self.fixture.public
        legacy = dict(deployment_id=public['deployment_id'],revision=1,scope='owned-synthetic',
            host=public['host'],profile_budget_digest='a'*64)
        with sqlite3.connect(self.root/'core.sqlite') as db:
            db.execute('UPDATE p1_installation SET record=?,digest=?',(encode(legacy),digest(legacy)))
        with self.assertRaisesRegex(ValueError,'configuration'): self.read()

    def test_boot_and_health_refuse(self):
        for field,value in (('boot','a'*32),('lane','a'*32),('quarantined',True),('launches',64)):
            with self.subTest(field=field):
                hello=dict(self.fixture.hello,**{field:value})
                with self.assertRaises(ValueError): self.read(hello=hello)

    def test_current_process_mismatch(self):
        for field,value in (('start_ticks',101),('pid_namespace',201),('cgroup','0::/docker/other')):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError,'incarnation'):
                    self.read(incarnation=dict(self.incarnation,**{field:value}))

    def test_gate_binding_mismatch(self):
        for field in ('registration_id','configuration_digest','enrollment_digest'):
            gate=copy.deepcopy(self.gate);gate[field]='f'*64
            with self.subTest(field=field), self.assertRaises(ValueError): self.read(gate=gate)

    def test_process_projection_mismatch(self):
        for field,value in (('id','c'*64),('image','sha256:'+'c'*64),('started','different'),
                            ('cgroup','/wrong'),('pid_namespace',201),('start_ticks',False)):
            gate=copy.deepcopy(self.gate);gate['processes'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.read(gate=gate)

    def test_exact_stored_registration_required(self):
        with sqlite3.connect(self.root/'core.sqlite') as db:
            db.execute('UPDATE p1_registrations SET digest=?',('a'*64,))
        with self.assertRaisesRegex(ValueError,'registration'): self.read()

    def test_recovery_plan_refuses(self):
        with sqlite3.connect(self.root/'core.sqlite') as db:
            db.execute('INSERT INTO p1_recovery_plans VALUES(?,?,?)',('x','{}','x'))
        with self.assertRaisesRegex(ValueError,'recovery_obligation'): self.read()

    def test_b2_namespace_refuses(self):
        with sqlite3.connect(self.root/'core.sqlite') as db:
            db.execute('CREATE TABLE p1_b2_format(id INTEGER, revision INTEGER)')
        with self.assertRaisesRegex(ValueError,'recovery_obligation'): self.read()

    def test_legacy_startup_stays_closed(self):
        with patch.object(os,'getuid',return_value=10001), recovery._startup_read_lock(self.root):
            with self.assertRaises(ValueError): recovery.startup_read(self.root)

    def test_held_owner_exclusion(self):
        path=self.notification_directory();wait=e.WaitingControl(directory=path);self.addCleanup(wait.close)
        e.publish(encode(self.gate).encode(),directory=path)
        from unittest.mock import Mock
        runtime=Mock()
        with ip.locked_file(self.root/'control.lock',policy=ip.SPLIT,role='control'):
            with self.assertRaisesRegex(ValueError,'maintenance_lock_busy'):
                e.checked_waiting_gate(wait,self.fixture.config,runtime,root=self.root)
        runtime.call.assert_not_called()

    def test_stored_enrollment_columns_match_record(self):
        with sqlite3.connect(self.root/'core.sqlite') as db:
            db.execute("UPDATE p1_installation SET scope='owned-synthetic'")
        with self.assertRaisesRegex(ValueError,'configuration'):self.read()

    def test_gate_directory_mode_change_refuses(self):
        path=self.notification_directory();wait=e.WaitingControl(directory=path);self.addCleanup(wait.close)
        e.publish(encode(self.gate).encode(),directory=path);path.chmod(0o750)
        with self.assertRaisesRegex(ValueError,'directory_changed'):wait.read()

    def notification_directory(self):
        path=self.fixture.fixture.root/'notification'
        f=self.fixture.fixture; translate=f.translate
        def ownership(info,current=None):
            if current is not None and (current==path or path in current.parents):
                return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_uid=10001,st_gid=10001))
            return translate(info,current)
        f.translate=ownership
        self.enterContext(patch.object(e,'_control_identity'))
        return path

    def test_exclusive_notification_and_exact_reread(self):
        path=self.notification_directory();wait=e.WaitingControl(directory=path)
        self.addCleanup(wait.close)
        raw=encode(self.gate).encode();e.publish(raw,directory=path)
        self.assertEqual(wait.wait(),self.gate)
        with self.assertRaises(FileExistsError): e.publish(raw,directory=path)
        self.assertEqual(wait.read(),self.gate)
        with self.assertRaises(FileExistsError): e.WaitingControl(directory=path)

    def test_byte_identical_gate_replacement(self):
        path=self.notification_directory();wait=e.WaitingControl(directory=path);self.addCleanup(wait.close)
        raw=encode(self.gate).encode();e.publish(raw,directory=path);wait.read()
        new=path/'replacement';new.write_bytes(raw);new.chmod(0o600);new.replace(path/e.GATE_NAME)
        with self.assertRaisesRegex(ValueError,'replaced'): wait.read()

    def test_symlink_or_hardlink_notification_refuses(self):
        path=self.notification_directory();wait=e.WaitingControl(directory=path);self.addCleanup(wait.close)
        target=path/'other';target.write_bytes(encode(self.gate).encode());target.chmod(0o600)
        (path/e.GATE_NAME).symlink_to(target)
        with self.assertRaises(OSError):wait.read()
        (path/e.GATE_NAME).unlink();os.link(target,path/e.GATE_NAME)
        with self.assertRaisesRegex(ValueError,'gate_file'):wait.read()

    def test_notification_timeout(self):
        path=self.notification_directory();wait=e.WaitingControl(directory=path,seconds=.001);self.addCleanup(wait.close)
        with self.assertRaisesRegex(ValueError,'timeout'):wait.wait()
        self.assertTrue(path.exists())

    def test_notification_closed_bounded_no_secret_shape(self):
        raw=encode(self.gate).encode()
        for bad in (raw+b' ',b'x'*(e.GATE_LIMIT+1),b'{"version":1,"version":1}',
                    encode(dict(self.gate,credential='secret')).encode()):
            with self.subTest(size=len(bad)),self.assertRaises(ValueError):e.decode_projection(bad)

    def test_checked_waiting_gate_brackets_before_owner(self):
        path=self.notification_directory();wait=e.WaitingControl(directory=path);self.addCleanup(wait.close)
        e.publish(encode(self.gate).encode(),directory=path)
        test=self
        class Runtime:
            def call(self,op,body):
                test.assertEqual((op,body),('hello',{}));return copy.deepcopy(test.fixture.hello)
        with patch.object(e,'local_incarnation',return_value=self.incarnation):
            result=e.checked_waiting_gate(wait,self.fixture.config,Runtime(),root=self.root)
        self.assertEqual(result['registration_id'],self.rid)

    def test_changed_second_hello_refuses(self):
        path=self.notification_directory();wait=e.WaitingControl(directory=path);self.addCleanup(wait.close)
        e.publish(encode(self.gate).encode(),directory=path)
        from unittest.mock import Mock
        runtime=Mock();runtime.call.side_effect=[self.fixture.hello,dict(self.fixture.hello,boot='a'*32)]
        with patch.object(e,'local_incarnation',return_value=self.incarnation):
            with self.assertRaisesRegex(ValueError,'bracket_changed'):
                e.checked_waiting_gate(wait,self.fixture.config,runtime,root=self.root)

    def test_host_handoff_uses_successful_retained_record(self):
        f=self.fixture;path=f.authority/('registration-'+self.rid+'.json')
        value=json.loads(path.read_bytes());value['observation']['processes']=self.processes
        path.write_bytes(encode(value).encode())
        f.observer.observe_owners=lambda owners,public:copy.deepcopy(value['observation'])
        sent=[]
        result=h._installed_handoff(f.observer,f.owners,self.rid,f.fd,lambda:None,
            notify=lambda context,control,raw:sent.append((context,control,e.decode_projection(raw))))
        self.assertEqual(result,self.rid);self.assertEqual(len(sent),1)
        self.assertEqual(sent[0][1],f.owners[0]['id'])
        self.assertEqual(sent[0][2]['provenance_digest'],digest(value))
        self.assertNotIn('observation',sent[0][2]);self.assertNotIn('seed',sent[0][2])

    def test_host_changed_incarnation_does_not_notify(self):
        f=self.fixture;sent=[]
        f.observer.observe_owners=lambda *args:dict(changed=True)
        with self.assertRaisesRegex(ValueError,'incarnation_changed'):
            h._installed_handoff(f.observer,f.owners,self.rid,f.fd,lambda:None,
                notify=lambda *args:sent.append(args))
        self.assertEqual(sent,[])

    def test_host_unknown_record_does_not_notify(self):
        f=self.fixture;sent=[]
        with self.assertRaises((OSError,ValueError)):
            h._installed_handoff(f.observer,f.owners,'f'*64,f.fd,lambda:None,
                notify=lambda *args:sent.append(args))
        self.assertEqual(sent,[])

    def test_fixed_exec_stdin_only(self):
        calls=[]
        class Child:
            returncode=0;stdin=io.BytesIO()
            def communicate(child,raw,timeout):calls.append((raw,timeout))
        def spawn(command,**kw):
            calls.append((command,kw));return Child()
        with patch.object(h.subprocess,'Popen',side_effect=spawn):
            h._installed_notify('local',self.gate['registration']['containers'][0]['id'],encode(self.gate).encode())
        command,kw=calls[0]
        self.assertEqual(command[:7],['docker','--context','local','exec','--user','10001:10001','-i'])
        self.assertNotIn(encode(self.gate),command);self.assertEqual(calls[1][1],5)
        self.assertEqual(kw['stderr'],h.subprocess.DEVNULL)


class NotificationCleanupTests(unittest.TestCase):
    setUp = GateTests.setUp
    # Reuse the real registration/projection fixture without inheriting its tests.
    def notification_failure(self, *, unreaped=False, pipe=False, output=False, primary=True):
        error = h.subprocess.TimeoutExpired('injected-notification', 5)
        class Pipe:
            def close(self):
                if pipe: raise OSError('injected pipe close')
        class Child:
            stdin=Pipe(); returncode=0
            def communicate(self, raw, timeout):
                if primary: raise error
            def kill(self): pass
            def wait(self, timeout):
                if unreaped: raise h.subprocess.TimeoutExpired('injected-reap', 2)
        real = h.tempfile.TemporaryFile
        class Output:
            def __init__(self): self.file=real()
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
            def fileno(self): return self.file.fileno()
            def close(self):
                self.file.close()
                if output: raise OSError('injected output close')
        child=Child()
        return error, child, patch.object(h.subprocess,'Popen',return_value=child), patch.object(h.tempfile,'TemporaryFile',Output)

    def check_failure(self, **kw):
        error, child, spawn, output = self.notification_failure(**kw)
        start=len(h._registration_retained)
        try:
            with spawn, output:
                with self.assertRaises(BaseException) as caught:
                    h._installed_notify('local',self.fixture.owners[0]['id'],encode(self.gate).encode())
            expected=h._RegistrationChildUnreaped if kw.get('unreaped') else (type(error) if kw.get('primary',True) else OSError)
            self.assertIsInstance(caught.exception,expected)
            if kw.get('unreaped'): self.assertIs(caught.exception.__cause__,error)
            elif kw.get('primary',True): self.assertIs(caught.exception,error)
            notes=' '.join(getattr(caught.exception,'__notes__',[]))
            for name in ('pipe','output'):
                if kw.get(name) and (kw.get('primary',True) or name=='output' and kw.get('pipe')):
                    self.assertIn(name,notes)
        finally:
            del h._registration_retained[start:]  # Only this inert injected child.

    def test_reaped_primary_pipe_close(self): self.check_failure(pipe=True)
    def test_reaped_primary_output_close(self): self.check_failure(output=True)
    def test_unreaped_output_close(self): self.check_failure(unreaped=True,output=True)
    def test_success_output_close_refuses(self): self.check_failure(primary=False,output=True)
    def test_success_pipe_and_output_close_refuses(self): self.check_failure(primary=False,pipe=True,output=True)

    def test_unreaped_both_closes_retains_authority(self):
        from tests.service.test_installed_host_policy import InstalledHost
        f=InstalledHost();f.setUp()
        try:
            f.authority_fixture()
            error,child,spawn,output=self.notification_failure(unreaped=True,pipe=True,output=True)
            start=len(h._registration_retained)
            try:
                with patch.object(h,'InstalledObserver',return_value=f.observer()), spawn, output:
                    with self.assertRaises(h._RegistrationChildUnreaped) as caught:
                        with h.installed_launch_authority(f.config,f.scope):
                            h._installed_notify('local',self.fixture.owners[0]['id'],encode(self.gate).encode())
                    self.assertIs(caught.exception.__cause__,error)
                    self.assertIn('pipe',' '.join(caught.exception.__notes__))
                    self.assertIn('output',' '.join(caught.exception.__notes__))
                    retained=h._registration_retained[start:]
                    self.assertIs(retained[0],child)
                    self.assertEqual(len(retained),2)
                    self.assertEqual(len(retained[1]),2)
                    for fd in retained[1]: os.fstat(fd)
                    with self.assertRaisesRegex(ValueError,'installed_launch_busy'):
                        with h.installed_launch_authority(f.config,f.scope): pass
            finally:
                for item in h._registration_retained[start:]:
                    if isinstance(item,tuple):
                        for fd in item: os.close(fd)
                del h._registration_retained[start:]
        finally: f.doCleanups()


if __name__ == '__main__':
    unittest.main()
