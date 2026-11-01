"""r382 inert worker tests; image/UID/process facts are injected, never qualified."""
import sys
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from contracts import validate as oracle
from service import installed_export as ex
from service.host_install import _registration_hello
from service.supervisor_runtime import SupervisorRuntime
from tests.service.test_installed_export import InstalledExportContractTests
from tests.worker.test_worker_audio_export import fixture
from worker import managed_reference as m
from worker.runtime_supervisor import RuntimeSupervisor, export_startup, main


class ExportSealTests(unittest.TestCase):
    def setUp(self):
        f=InstalledExportContractTests(); f.setUp(); self.addCleanup(f.doCleanups)
        self.tmp=tempfile.TemporaryDirectory(dir='/private/tmp' if sys.platform == 'darwin' else '/tmp'); self.addCleanup(self.tmp.cleanup)
        self.config,self.exe,self.lib=f.local_converter(Path(self.tmp.name))
        self.image=self.config['images']['worker']
        self.s=RuntimeSupervisor.__new__(RuntimeSupervisor)
        self.s._export_configuration=copy.deepcopy(self.config)
        self.s.export_reference_roots={'alice':'/references/assets'}
        self.s.converter_image_digest=self.image
        self.s.failed=False; self.s.active=None; self.s.boot='a'*32; self.s.launches=0
        self.s.journal=SimpleNamespace(value=dict(lane='b'*32,records=[]),maintenance=None)
        self.context=fixture()['context']
        reg=copy.deepcopy(self.config['audio_export']['converter'])
        self.context['execution']['converter']=reg
        self.context['plan']['resolved']['identity']['converter']=copy.deepcopy(reg)

    def qualified(self):
        # Injected assertion only, never emitted as an operational receipt.
        self.s._export_configuration['audio_export']['profile'].update(support='qualified',installation='installed',readiness='ready',verification='measured',qualification_receipt='INJECTED r382')

    def test_startup_requires_trusted_image_and_actual_immutable_closure(self):
        got=export_startup(self.config,self.image,owner_uid=os.getuid())
        self.assertEqual(got,self.config); self.assertIsNot(got,self.config)
        for image in (None,'sha256:'+'f'*64,True):
            with self.assertRaises(ValueError): export_startup(self.config,image,owner_uid=os.getuid())
        for path,mode in ((self.exe,0o755),(self.lib,0o644)):
            path.chmod(mode)
            with self.assertRaises(ValueError): export_startup(self.config,self.image,owner_uid=os.getuid())
            path.chmod(0o555 if path==self.exe else 0o444)
        c=copy.deepcopy(self.config); c['audio_export']['converter']['executable_sha256']='f'*64
        with self.assertRaises(ValueError): export_startup(c,self.image,owner_uid=os.getuid())

    def test_closed_new_hello_old_exact_and_readiness_refusal(self):
        hello=self.s.operation('hello',{})
        self.assertEqual(set(hello),{'lane','boot','quarantined','launches','export_seal'})
        self.assertEqual(_registration_hello(hello,family=ex.FAMILY,configuration=self.config),hello)
        with self.assertRaises(ValueError): _registration_hello(hello)
        for key in hello['export_seal']:
            h=copy.deepcopy(hello); h['export_seal'][key]=True if key=='publication_format' else 'f'*64
            with self.assertRaises(ValueError): _registration_hello(h,family=ex.FAMILY,configuration=self.config)
        for v in (2.0,True):
            h=copy.deepcopy(hello);h['export_seal']['publication_format']=v
            with self.assertRaises(ValueError): _registration_hello(h,family=ex.FAMILY,configuration=self.config)
        for change in ({'quarantined':True},{'extra':0}):
            with self.assertRaises(ValueError): _registration_hello(dict(hello,**change),family=ex.FAMILY,configuration=self.config)
        self.s._export_configuration=None
        old=self.s.operation('hello',{})
        self.assertEqual(set(old),{'lane','boot','quarantined','launches'})
        _registration_hello(old)
        with self.assertRaises(ValueError): _registration_hello(old,family=ex.FAMILY,configuration=self.config)

    def test_full_frozen_profile_principal_and_converter_before_helper(self):
        self.qualified()
        cases=[]
        c=copy.deepcopy(self.context); c['plan']['resolved']['profile']['limits']['threads']+=1; cases.append((c,'alice'))
        c=copy.deepcopy(self.context); c['plan']['resolved']['profile']['limits']['threads']=2.0; cases.append((c,'alice'))
        cases.append((self.context,'bob'))
        c=copy.deepcopy(self.context); c['execution']['converter']['reported_build']='caller';c['plan']['resolved']['identity']['converter']=copy.deepcopy(c['execution']['converter']);cases.append((c,'alice'))
        c=copy.deepcopy(self.context); c['execution']['converter']['worker_image_digest']='sha256:'+'f'*64;c['plan']['resolved']['identity']['converter']=copy.deepcopy(c['execution']['converter']);cases.append((c,'alice'))
        with patch.object(m,'acquire') as acquire:
            for c,p in cases:
                with self.assertRaises(ValueError):self.s.verify_export(c,p)
            acquire.assert_not_called()

    def test_unqualified_profile_cannot_acquire_or_dispatch(self):
        with patch.object(m,'acquire') as acquire:
            with self.assertRaisesRegex(ValueError,'not_ready'):self.s.verify_export(self.context,'alice')
            acquire.assert_not_called()

    def source(self):
        path=Path(self.tmp.name)/'source';path.write_bytes(Path('contracts/fixtures/v1/short.wav').read_bytes());path.chmod(0o600)
        return path

    def test_helper_fd_verified_retained_readonly_and_closed_on_bad_hash(self):
        self.qualified(); path=self.source()
        real=ex.verify_local_converter
        check=m.check_fd
        fstat=os.fstat
        def owned(fd):
            info=fstat(fd)
            return SimpleNamespace(**dict({k:getattr(info,k) for k in dir(info) if k.startswith('st_')},st_gid=os.getuid()))
        def checked(fd,**kw):
            with patch.object(m.os,'fstat',side_effect=owned):return check(fd,**kw)
        with patch.object(m,'check_fd',side_effect=checked), patch.object(ex,'verify_local_converter',side_effect=lambda c:real(c,owner_uid=os.getuid())), patch.object(m,'acquire',side_effect=lambda identity:os.open(path,os.O_RDONLY)) as acquire:
            fd=self.s.verify_export(self.context,'alice',owner_uid=os.getuid())
            try:
                self.assertEqual(os.lseek(fd,0,1),0)
                with self.assertRaises(OSError):os.write(fd,b'x')
            finally:os.close(fd)
            acquire.assert_called_once_with('short')
            bad=copy.deepcopy(self.context);bad['plan']['assets'][0]['sha256']='f'*64
            with self.assertRaises(ValueError):self.s.verify_export(bad,'alice',owner_uid=os.getuid())

    def test_helper_uncertainty_quarantines_without_second_scheduler(self):
        self.qualified()
        with patch.object(ex,'verify_local_converter',return_value=os.open(self.exe,os.O_RDONLY)), patch.object(m,'acquire',side_effect=ValueError('managed_reference_helper_unreaped')):
            with self.assertRaisesRegex(ValueError,'unreaped'):self.s.verify_export(self.context,'alice')
        self.assertTrue(self.s.failed)
        self.assertTrue(self.s.operation('hello',{})['quarantined'])

    def test_main_explicit_family_selection_only(self):
        import service.execution_context as ec
        import service.install_policy as ip
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name,value in (('getuid',0),('geteuid',0),('getgid',10001),('getegid',10001),('getgroups',[])):
                stack.enter_context(patch('worker.runtime_supervisor.os.'+name,return_value=value))
            stack.enter_context(patch('service.auth.private_input',return_value=b'{}'))
            stack.enter_context(patch('service.recovery.host_json',return_value=self.config))
            stack.enter_context(patch('worker.runtime_supervisor.export_startup',return_value=self.config))
            mapping=stack.enter_context(patch.object(ec,'installed_mapping',return_value={}))
            stack.enter_context(patch.object(ip,'directory_identity'))
            stack.enter_context(patch.object(ip,'read_private'))
            runtime=stack.enter_context(patch('worker.runtime_supervisor.RuntimeSupervisor'))
            stack.enter_context(patch('sys.argv',['worker','--installed-policy',ex.POLICY,'--configuration-file','/config','--expected-worker-image',self.image]))
            main()
            self.assertEqual(runtime.call_args.kwargs['export_configuration'],self.config)
            self.assertEqual(runtime.call_args.kwargs['expected_worker_image'],self.image)
            self.assertNotIn('audio_export',mapping.call_args.args[0])
            runtime.return_value.serve.assert_called_once()

    def test_closed_startup_inputs_and_transport_family(self):
        for value in (2.0,True):
            with self.assertRaises(ValueError):export_startup(dict(self.config,version=value),self.image)
        with self.assertRaises(ValueError):export_startup(dict(self.config,extra=1),self.image)
        with self.assertRaises(ValueError):SupervisorRuntime(family=object(),configuration=self.config)
        runtime=SupervisorRuntime(family=ex.FAMILY,configuration=self.config)
        self.config['audio_export']['principals'].clear()
        self.assertEqual(runtime.configuration['audio_export']['principals'],['alice'])

    def test_actual_acquire_unknown_wait_retains_process_and_closes_fd(self):
        import array
        import socket
        fd=os.open(self.source(),os.O_RDONLY)
        parent=Mock();child=Mock();child.fileno.return_value=17
        parent.recvmsg.return_value=(b'R',[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[fd]).tobytes())],0,None)
        proc=Mock(returncode=None);proc.wait.side_effect=RuntimeError('INJECTED unknown wait')
        retained=[]
        with patch.object(m,'_retained',retained), patch.object(socket,'socketpair',return_value=(parent,child)), patch.object(socket,'MSG_CMSG_CLOEXEC',getattr(socket,'MSG_CMSG_CLOEXEC',0),create=True), patch.object(m.subprocess,'Popen',return_value=proc) as launch:
            with self.assertRaisesRegex(ValueError,'helper_unreaped'):m.acquire('short')
            self.assertEqual(retained,[proc])
            proc.kill.assert_called_once()
            self.assertEqual(proc.wait.call_count,2)
            self.assertEqual((launch.call_args.kwargs['user'],launch.call_args.kwargs['group'],launch.call_args.kwargs['extra_groups']),(10001,10001,[]))
            with self.assertRaises(OSError):os.fstat(fd)

    def test_transport_requires_matching_seal(self):
        import service.supervisor_runtime as sr
        runtime=SupervisorRuntime(family=ex.FAMILY,configuration=self.config)
        hello=self.s.operation('hello',{})
        connection=Mock();connection.__enter__=Mock(return_value=connection);connection.__exit__=Mock(return_value=False)
        with patch.object(Path,'read_text',return_value='a'*64),patch.object(sr.socket,'socket',return_value=connection),patch.object(sr,'authenticate'),patch.object(sr,'send'),patch.object(sr,'receive',return_value=dict(ok=True,value=hello)):
            self.assertEqual(runtime.call('hello',{}),hello)
            hello['export_seal']['worker_image_digest']='sha256:'+'f'*64
            with self.assertRaises(ValueError):runtime.call('hello',{})

    def test_constructor_refuses_parallel_caller_maps(self):
        for kw in (dict(audio_converters={('alice','id'): {}}),dict(converter_image_digest=self.image),dict(export_reference_roots={'alice':'/other'}),dict(installed_pocket=True)):
            with patch('worker.runtime_supervisor.os.getpid',return_value=1),patch('worker.runtime_supervisor.export_startup',return_value=copy.deepcopy(self.config)):
                with self.assertRaisesRegex(ValueError,'sole_authority'):
                    RuntimeSupervisor(export_configuration=self.config,expected_worker_image=self.image,**kw)

    def test_local_closure_rechecked_before_source_acquisition(self):
        self.qualified()
        real=ex.verify_local_converter
        self.lib.chmod(0o644)
        with patch.object(ex,'verify_local_converter',side_effect=lambda c:real(c,owner_uid=os.getuid())),patch.object(m,'acquire') as acquire:
            with self.assertRaises(ValueError):self.s.verify_export(self.context,'alice')
            acquire.assert_not_called()
