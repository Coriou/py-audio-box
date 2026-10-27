"""Inert FD/kernel-empty tests with injected ownership policy, no UID switch."""
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import queue
import stat
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from worker import runtime_conditioning as c
from worker.runtime_supervisor import RuntimeSupervisor


@contextmanager
def filesystem_policy(root, stage, uid=20000, mutate=None):
    """Redirect only fixed root; inject installed metadata, preserve real inodes.

    This proves FD operations and same-UID rmdir, never cross-UID kernel access.
    """
    real_open, real_stat, real_fstat = os.open, os.stat, os.fstat
    def metadata(s, role):
        values = dict(st_dev=s.st_dev, st_ino=s.st_ino, st_mode=s.st_mode,
                      st_uid=s.st_uid, st_gid=s.st_gid)
        if role == 'root': values.update(st_uid=0, st_gid=0, st_mode=stat.S_IFDIR|0o1733)
        elif role == 'stage': values.update(st_uid=10001, st_gid=uid)
        else: values.update(st_uid=uid, st_gid=uid)
        if mutate: mutate(role, values)
        return SimpleNamespace(**values)
    root_inode=real_stat(root).st_ino; stage_inode=real_stat(stage).st_ino
    def role(s): return 'root' if s.st_ino==root_inode else 'stage' if s.st_ino==stage_inode else 'child'
    def opened(path, flags, *args, **kw):
        if path == '/staging':
            # macOS test host lacks O_PATH; production always requests it.
            return real_open(root, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        return real_open(path, flags, *args, **kw)
    def stated(path, *args, **kw):
        s=real_stat(root if path=='/staging' else path,*args,**kw)
        return metadata(s,role(s))
    with ExitStack() as stack:
        stack.enter_context(patch.object(os,'O_PATH',getattr(os,'O_PATH',0x200000),create=True))
        stack.enter_context(patch.object(os,'open',side_effect=opened))
        stack.enter_context(patch.object(os,'stat',side_effect=stated))
        stack.enter_context(patch.object(os,'fstat',side_effect=lambda fd: (lambda s:metadata(s,role(s)))(real_fstat(fd))))
        yield


class ConditioningFilesystemTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        self.root=Path(t.name); self.stage=self.root/'attempt'; self.stage.mkdir();self.stage.chmod(0o770)
        self.child=self.stage/'conditioning';self.child.mkdir();self.child.chmod(0o700)
        (self.stage/'output.wav').write_bytes(b'untouched')
    def run_helper(self, mutate=None):
        with filesystem_policy(self.root,self.stage,mutate=mutate): c.remove_empty('attempt',20000)
    def test_pocket_v2_requires_explicit_fixed_selector_and_empty_directory(self):
        from service import install_policy as ip
        def v2(role, values):
            if role=='root':values.update(st_gid=10001,st_mode=stat.S_IFDIR|0o1753)
        with filesystem_policy(self.root,self.stage,mutate=v2):
            with self.assertRaises(c.ConditioningRefused):c.remove_empty('attempt',20000)
            with self.assertRaises(c.ConditioningRefused):c.remove_empty('attempt',20000,policy=ip.EXPORT_SPLIT)
            (self.child/'retained').write_bytes(b'x')
            with self.assertRaises(OSError):c.remove_empty('attempt',20000,policy=ip.POCKET_SPLIT)
            (self.child/'retained').unlink()
            c.remove_empty('attempt',20000,policy=ip.POCKET_SPLIT)
        self.assertFalse(self.child.exists())
        self.assertEqual((self.stage/'output.wav').read_bytes(),b'untouched')

    def test_empty_and_verified_absence(self):
        with patch.object(os,'rmdir',wraps=os.rmdir) as remove, patch.object(os,'fsync',wraps=os.fsync) as sync:
            self.run_helper(); self.run_helper()
            self.assertEqual(remove.call_count,1);self.assertEqual(remove.call_args.args,('conditioning',))
            self.assertEqual(set(remove.call_args.kwargs),{'dir_fd'});sync.assert_called_once()
        self.assertEqual((self.stage/'output.wav').read_bytes(),b'untouched')
    def test_named_snapshot_leftovers_refuse_without_removal(self):
        for name in ('wrapper.wav','backend.wav','translated.yaml'):
            with self.subTest(name=name):
                sentinel=self.child/name;sentinel.write_bytes(b'opaque')
                with self.assertRaises(OSError):self.run_helper()
                self.assertEqual(sentinel.read_bytes(),b'opaque');sentinel.unlink()
    def test_symlink_and_regular_file_refuse(self):
        self.child.rmdir();self.child.symlink_to(self.stage/'output.wav')
        with self.assertRaises(c.ConditioningRefused):self.run_helper()
        self.child.unlink();self.child.write_bytes(b'opaque')
        with self.assertRaises(c.ConditioningRefused):self.run_helper()
        self.assertEqual(self.child.read_bytes(),b'opaque')
    def test_policy_foreign_mode_device_and_stage_substitution(self):
        for role,key,value in [('root','st_uid',1),('root','st_mode',stat.S_IFDIR|0o777),('stage','st_uid',1),('stage','st_gid',1),('stage','st_mode',stat.S_IFDIR|0o777),('stage','st_dev',-1),('child','st_uid',1),('child','st_gid',1),('child','st_mode',stat.S_IFDIR|0o770),('child','st_dev',-1)]:
            with self.subTest(role=role,key=key):
                def mutate(r,v):
                    if r==role:v[key]=value
                with self.assertRaises(c.ConditioningRefused):self.run_helper(mutate)
                self.assertTrue(self.child.exists())
        calls=0
        def changed(role,v):
            nonlocal calls
            if role=='stage':
                calls+=1
                if calls>1:v['st_ino']+=1
        with self.assertRaises(c.ConditioningRefused):self.run_helper(changed)
        self.assertTrue(self.child.exists())
    def test_fsync_and_rmdir_failure_refuse(self):
        with patch.object(os,'rmdir',side_effect=PermissionError):
            with self.assertRaises(PermissionError):self.run_helper()
        self.assertTrue(self.child.exists())
        with patch.object(os,'fsync',side_effect=OSError):
            with self.assertRaises(OSError):self.run_helper()
        self.assertFalse(self.child.exists())
    def test_post_fsync_stage_identity_change_is_refused(self):
        real_sync=os.fsync
        def changed(fd):
            real_sync(fd)
            self.stage.rename(self.root/'retained')
            self.stage.mkdir(mode=0o770)
        with patch.object(os,'fsync',side_effect=changed):
            with self.assertRaises(c.ConditioningRefused):self.run_helper()
        self.assertEqual((self.root/'retained'/'output.wav').read_bytes(),b'untouched')
    def test_child_identity_change_before_rmdir_refuses(self):
        calls=0
        def changed(role,values):
            nonlocal calls
            if role=='child':
                calls+=1
                if calls>1:values['st_ino']+=1
        with patch.object(os,'rmdir',wraps=os.rmdir) as remove:
            with self.assertRaises(c.ConditioningRefused):self.run_helper(changed)
            remove.assert_not_called()
        self.assertTrue(self.child.is_dir())
    def test_no_stage_is_not_absence_and_bad_names_refuse(self):
        with filesystem_policy(self.root,self.stage):
            with self.assertRaises(FileNotFoundError):c.remove_empty('missing',20000)
        for name in ('','..','a/b','a\\b'):
            with self.assertRaises(c.ConditioningRefused):c.remove_empty(name,20000)

    def test_root_open_flags_and_exact_main_credentials(self):
        with filesystem_policy(self.root,self.stage):
            with patch.object(os,'open',wraps=os.open) as opened:
                c.remove_empty('attempt',20000)
                root_call=opened.call_args_list[0]
                self.assertEqual(root_call.args[0],'/staging')
                self.assertEqual(root_call.args[1]& (os.O_PATH|os.O_NOFOLLOW|os.O_DIRECTORY),os.O_PATH|os.O_NOFOLLOW|os.O_DIRECTORY)
                self.assertEqual(opened.call_count,2)  # Never opens conditioning/descendants.
        with ExitStack() as stack:
            stack.enter_context(patch('sys.argv',['helper','attempt','20000']))
            for name,value in [('getuid',10001),('geteuid',10001),('getgid',20000),('getegid',20000),('getgroups',[])]:
                stack.enter_context(patch.object(os,name,return_value=value))
            remove=stack.enter_context(patch.object(c,'remove_empty'))
            c.main();remove.assert_called_once_with('attempt',20000,policy=None)
            with patch.object(os,'getgroups',return_value=[10001]):
                with self.assertRaises(c.ConditioningRefused):c.main()
    def test_stage_symlink_and_substitution_leave_original_untouched(self):
        alias=self.root/'alias';alias.symlink_to(self.stage,target_is_directory=True)
        with filesystem_policy(self.root,self.stage):
            with self.assertRaises(OSError):c.remove_empty('alias',20000)
        calls=0
        with filesystem_policy(self.root,self.stage):
            policy_stat=os.stat
            def substituted(path,*args,**kw):
                nonlocal calls
                if path=='attempt':
                    calls+=1
                    if calls==1:
                        self.stage.rename(self.root/'retained')
                        self.stage.mkdir(mode=0o770)
                return policy_stat(path,*args,**kw)
            with patch.object(os,'stat',side_effect=substituted):
                with self.assertRaises(c.ConditioningRefused):c.remove_empty('attempt',20000)
        self.assertTrue((self.root/'retained'/'conditioning').is_dir())


class ConditioningSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.s=RuntimeSupervisor.__new__(RuntimeSupervisor)
        self.row=dict(id='lane/boot/allocation',uid=20000,phase='started',allocation=dict(fixture='pocket',binding='frozen',snapshot=[dict(deadline='original',attempts=[dict(staging_id='attempt')]),{}]))
        self.enterContext(patch('worker.runtime_protocol.execution',return_value=SimpleNamespace(engine='pocket-tts')))
        self.s.journal=SimpleNamespace(value=dict(records=[self.row]))
        self.s.failed=False;self.s.halt=False;self.s.tree_lock=threading.Lock();self.s.received=queue.Queue()
        done=threading.Event();done.set()
        self.s.active=dict(id=self.row['id'],cutoff=100,started=True,stopping=None,leader=42,process=SimpleNamespace(returncode=None),reaped_children=0,receiver_done=done,proposal={'engine':'proposal'},proposed_at=90)
        self.s.change=Mock();self.s.signal_namespace=Mock()
    def tick(self, waits=None):
        with patch('worker.runtime_supervisor.os.waitpid',side_effect=waits or [ChildProcessError()]),patch('worker.runtime_supervisor.time.monotonic',return_value=95):self.s.tick()
    def test_only_trusted_pocket_v2_supervisor_adds_selector(self):
        from service import install_policy as ip
        self.s.installed_policy=ip.POCKET_SPLIT
        proc=Mock();proc.wait.return_value=0
        with patch('worker.runtime_supervisor.subprocess.Popen',return_value=proc) as launch:
            self.s.conditioning(self.row)
        self.assertEqual(launch.call_args.args[0][-1],ip.POCKET_V2_POLICY)

    def test_before_echild_and_receiver_pending_never_launch(self):
        with patch('worker.runtime_supervisor.subprocess.Popen') as launch:
            self.tick([(42,0),(0,0)]);launch.assert_not_called();self.s.change.assert_not_called()
            self.assertEqual(self.s.active['reaped_children'],1)
            self.s.active['receiver_done'].clear();self.tick();launch.assert_not_called()
    def test_success_frozen_receipt_credentials_and_watchdog(self):
        proc=Mock();proc.wait.return_value=0
        def launch(*a,**kw):
            self.assertFalse(self.s.tree_lock.locked());self.assertIsNotNone(self.s.active)
            self.s.watchdog_tick();self.s.signal_namespace.assert_called_once() # leader drain only
            return proc
        with patch('worker.runtime_supervisor.subprocess.Popen',side_effect=launch) as popen:
            self.tick([(42,0),ChildProcessError()])
        self.assertEqual(popen.call_args.kwargs,dict(user=10001,group=20000,extra_groups=[],close_fds=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
        self.assertEqual(popen.call_args.args[0],['python','-c','from worker.runtime_conditioning import pocket_main; pocket_main()','attempt','20000'])
        receipt=self.s.change.call_args.kwargs['receipt']
        self.assertEqual(receipt,dict(runtime_id=self.row['id'],binding='frozen',reaped=True,engine_uid=20000,reap_method='pid1-waitpid-echild',reaped_children=1,timely=True,original_deadline='original'))
        self.assertIsNone(self.s.active)
    def test_failure_timeout_unknown_and_no_next_tick_terminal(self):
        for waits,unknown in [([1],False),([subprocess.TimeoutExpired('helper',1),-9],False),([subprocess.TimeoutExpired('helper',1),subprocess.TimeoutExpired('helper',1)],True),([OSError(),-9],False)]:
            with self.subTest(waits=waits):
                self.setUp();original_active=self.s.active;proc=Mock();proc.wait.side_effect=waits
                with patch('worker.runtime_supervisor.subprocess.Popen',return_value=proc):self.tick()
                self.assertTrue(self.s.failed);self.s.change.assert_not_called()
                self.assertEqual(original_active['cutoff'],100)
                self.assertEqual(original_active['reaped_children'],0)
                self.assertEqual(self.row['allocation']['binding'],'frozen')
                self.assertEqual(self.s.active is not None,unknown)
                self.tick();self.s.change.assert_not_called()
                if len(waits)>1:proc.kill.assert_called_once()
                self.assertEqual(self.row['phase'],'started')
                from worker.transport import Refused
                with self.assertRaisesRegex(Refused,'terminal_receipt_required'):
                    self.s.operation('ack',dict(runtime_id=self.row['id']))
                self.assertIsNone(self.s.operation('collect',dict(runtime_id=self.row['id']))['receipt'])
    def test_journal_failure_after_cleanup_never_terminalizes_again(self):
        self.s.change.side_effect=OSError('journal')
        with patch.object(self.s,'conditioning') as helper:self.tick();helper.assert_called_once_with(self.row)
        self.assertTrue(self.s.failed);self.assertEqual(self.row['phase'],'started')
        self.tick();self.assertEqual(self.s.change.call_count,1)
    def test_housekeeping_elapsed_time_does_not_change_engine_timeliness(self):
        clock=[95]
        def slow(row):clock[0]=200
        with patch.object(self.s,'conditioning',side_effect=slow),patch('worker.runtime_supervisor.os.waitpid',side_effect=ChildProcessError),patch('worker.runtime_supervisor.time.monotonic',side_effect=lambda:clock[0]):
            self.s.tick()
        receipt=self.s.change.call_args.kwargs['receipt']
        self.assertTrue(receipt['timely']);self.assertEqual(receipt['original_deadline'],'original')
        self.assertEqual(receipt['binding'],'frozen');self.assertEqual(receipt['reaped_children'],0)
    def test_journal_failure_after_real_rmdir_keeps_nonterminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'attempt';stage.mkdir(mode=0o770);stage.chmod(0o770)
            (stage/'conditioning').mkdir(mode=0o700)
            self.s.change.side_effect=OSError('journal')
            def remove(row):
                with filesystem_policy(root,stage):c.remove_empty('attempt',20000)
            with patch.object(self.s,'conditioning',side_effect=remove):self.tick()
            self.assertFalse((stage/'conditioning').exists())
            self.assertTrue(self.s.failed);self.assertEqual(self.row['phase'],'started')
            self.tick();self.assertEqual(self.s.change.call_count,1)
    def test_prestart_stop_no_stage_helper(self):
        self.row['phase']='prepared';self.s.active.update(started=False,stopping=90)
        with patch.object(self.s,'conditioning') as helper:self.tick();helper.assert_not_called()
        self.s.change.assert_called_once()
