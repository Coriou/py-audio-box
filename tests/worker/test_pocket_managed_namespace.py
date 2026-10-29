"""Inert filesystem/tuple and cache-environment tests; no model or UID transition."""
from contextlib import ExitStack
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lib.tts_backends.pocket_snapshot import managed_directory
from tests.worker.test_runtime_conditioning import filesystem_policy
from worker import runtime_conditioning
from worker.engines import pocket


class ManagedPocketNamespaceTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name).resolve();self.stage=self.root/'attempt';self.stage.mkdir(mode=0o770);self.stage.chmod(0o770)
        self.child=self.stage/'conditioning';self.child.mkdir(mode=0o770);self.child.chmod(0o770)

    def context(self,mutate=None,uid=20000):
        stack=ExitStack()
        def metadata(role,values):
            if role=='child':values.update(st_uid=10001,st_gid=20000,st_mode=stat.S_IFDIR|0o770)
            if mutate:mutate(role,values)
        stack.enter_context(filesystem_policy(self.root,self.stage,mutate=metadata))
        for name,value in (('getuid',uid),('geteuid',uid),('getgid',uid),('getegid',uid),('getgroups',[])):
            stack.enter_context(patch.object(os,name,return_value=value))
        return stack

    def test_exact_core_owned_tuple_and_core_traversal(self):
        with self.context():
            with managed_directory(self.child) as directory:self.assertEqual(directory,self.child)
            self.assertEqual(os.stat(self.child).st_uid,10001)
            self.assertEqual(stat.S_IMODE(os.stat(self.child).st_mode),0o770)
            self.assertEqual(list(self.child.iterdir()),[])

    def test_wrong_owner_group_mode_sibling_uid_and_symlink_refuse(self):
        for role,key,value in (('child','st_uid',20000),('child','st_gid',20001),('child','st_mode',stat.S_IFDIR|0o700),('stage','st_uid',20000),('stage','st_gid',20001),('stage','st_mode',stat.S_IFDIR|0o777)):
            with self.subTest(role=role,key=key),self.context(lambda r,v:v.update({key:value}) if r==role else None):
                with self.assertRaises(ValueError):
                    with managed_directory(self.child):pass
        for uid in (10001,19999,20001,20064):
            with self.subTest(uid=uid),self.context(uid=uid),self.assertRaises(ValueError):
                with managed_directory(self.child):pass
        alias=self.stage/'alias';alias.symlink_to(self.child,target_is_directory=True)
        with self.context(),self.assertRaises(ValueError):
            with managed_directory(alias):pass

    def test_managed_empty_only_cleanup_and_nonempty_refusal(self):
        with self.context():runtime_conditioning.remove_empty('attempt',20000,managed_namespace=True)
        self.assertFalse(self.child.exists())
        self.child.mkdir(mode=0o770);self.child.chmod(0o770)
        (self.child/'unexpected-cache').write_bytes(b'retained')
        with self.context(),self.assertRaises(OSError):runtime_conditioning.remove_empty('attempt',20000,managed_namespace=True)
        self.assertEqual((self.child/'unexpected-cache').read_bytes(),b'retained')

    def test_managed_backend_flag_and_cache_restored_on_constructor_failure(self):
        seen=[]
        def failing(*args,**kwargs):
            seen.append((dict(kwargs),os.environ.get('TORCHINDUCTOR_CACHE_DIR')))
            raise RuntimeError('inert constructor failure')
        spec=SimpleNamespace(language='fr',threads=2)
        for prior in (None,'/previous-cache'):
            with patch.dict(os.environ,{},clear=False):
                if prior is None:os.environ.pop('TORCHINDUCTOR_CACHE_DIR',None)
                else:os.environ['TORCHINDUCTOR_CACHE_DIR']=prior
                with self.context(),self.assertRaises(RuntimeError):
                    pocket._synthesise(spec,'text',self.root,b'reference',self.child,failing,managed_namespace=True)
                self.assertEqual(os.environ.get('TORCHINDUCTOR_CACHE_DIR'),prior)
                self.assertEqual(seen[-1][0]['managed_namespace'],True)
                self.assertEqual(seen[-1][1],str(self.child))

    def test_default_owned_0700_policy_remains_distinct(self):
        self.child.chmod(0o700)
        seen=[]
        def failing(*args,**kwargs):
            seen.append(kwargs)
            self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],str(self.child))
            raise RuntimeError('inert default failure')
        with self.assertRaises(RuntimeError):
            pocket._synthesise(SimpleNamespace(language='fr',threads=2),'text',self.root,b'reference',self.child,failing)
        self.assertNotIn('managed_namespace',seen[-1])
        self.child.chmod(0o770)
        with self.assertRaises(ValueError):
            pocket._synthesise(SimpleNamespace(language='fr',threads=2),'text',self.root,b'reference',self.child,failing)

    def test_cache_stays_scoped_through_synthesis_close_and_failures(self):
        for close_fails in (False,True):
            events=[];namespace=str(self.child)
            class Backend:
                def __init__(inner,*args,**kwargs):
                    events.append('construct')
                    self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],namespace)
                def synthesise(inner,*args,**kwargs):
                    events.append('synthesise')
                    self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],namespace)
                    raise RuntimeError('inert synthesis failure')
                def close(inner):
                    events.append('close')
                    self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],namespace)
                    if close_fails:raise RuntimeError('inert close failure')
            with self.subTest(close_fails=close_fails),patch.dict(os.environ,{'TORCHINDUCTOR_CACHE_DIR':'/retained-prior'}),self.context():
                with self.assertRaises(RuntimeError):
                    pocket._synthesise(SimpleNamespace(language='fr',threads=2),'text',self.root,b'reference',self.child,Backend,managed_namespace=True)
                self.assertEqual(events,['construct','synthesise','close'])
                self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],'/retained-prior')

    def test_supervisor_setup_precedes_launch_and_managed_pocket_and_qwen_clone_get_literal_namespace(self):
        from unittest.mock import Mock
        from worker.runtime_supervisor import RuntimeSupervisor
        class SetupComplete(Exception):pass
        from dataclasses import replace
        from worker.engines.qwen_execution import QwenExecution
        preset=QwenExecution('bundle','a'*64,'operator-checkpoint','exact-revision',
            'custom_voice','English','voice','v1',2,100,100,244,4096,24000,preset='Ryan')
        clone=replace(preset,task='clone',preset=None,reference_id='ref',
            reference_sha256='b'*64,reference_bytes=52,reference_frames=4,
            reference_rate=24000,reference_text=None,x_vector_only=True)
        cases=(('pocket',SimpleNamespace(engine='pocket-tts'),True),
               ('qwen-preset',preset,False),('qwen-clone',clone,True),
               ('kokoro',SimpleNamespace(engine='kokoro'),False))
        for stage_id,spec,conditioning in cases:
            engine=spec.engine
            stage=self.root/stage_id
            supervisor=RuntimeSupervisor.__new__(RuntimeSupervisor)
            row=dict(id='runtime',phase='prepared',uid=20000,allocation=dict(snapshot=[dict(principal_id='alice',attempts=[dict(staging_id=stage_id)]),{},[],[]]))
            supervisor.lookup=lambda identity:row
            supervisor.failed=False;supervisor.resident_qwen=False;supervisor.active=dict(id='runtime',cutoff=100)
            supervisor.change=Mock(side_effect=lambda record,**changes:dict(record,**changes))
            supervisor.verify_pocket=Mock();supervisor.verify_qwen=Mock()
            def setup(arguments,**kwargs):
                self.assertEqual(kwargs,dict(user=10001,group=20000,extra_groups=[],close_fds=True,check=True,timeout=1))
                supervisor.change.assert_called_once_with(row,phase='started')
                # Execute only the finite mkdir program as this test's UID.
                # This checks names/modes/order, never a kernel UID transition.
                prior=os.umask(0o077)
                try:
                    with patch('sys.argv',['setup',str(stage)]):exec(arguments[2],{})
                finally:os.umask(prior)
                self.assertEqual(stat.S_IMODE(stage.stat().st_mode),0o770)
                self.assertEqual((stage/'conditioning').exists(),conditioning)
                if conditioning:self.assertEqual(stat.S_IMODE((stage/'conditioning').stat().st_mode),0o770)
                raise SetupComplete()
            with self.subTest(case=stage_id),patch('worker.runtime_protocol.execution',return_value=spec),patch('worker.runtime_supervisor.time.monotonic',return_value=0),patch('worker.runtime_supervisor.subprocess.run',side_effect=setup),patch('worker.runtime_supervisor.subprocess.Popen') as launch:
                with self.assertRaises(SetupComplete):supervisor.operation('start',dict(runtime_id='runtime'))
                launch.assert_not_called()
                context=row['allocation']['snapshot'][1]
                if engine=='pocket-tts':supervisor.verify_pocket.assert_called_once_with(context,'alice')
                else:supervisor.verify_pocket.assert_not_called()
                if engine=='qwen':supervisor.verify_qwen.assert_called_once_with(context,[],'alice')
                else:supervisor.verify_qwen.assert_not_called()
