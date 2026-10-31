"""Private FD transfer SOURCE tests; subprocess/UID observations injected."""
import array
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from worker import managed_reference as m
from worker.engines.pocket import read_reference_fd
from worker.transport import Refused


class ManagedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.path=self.root/'asset-a'
        import io,wave
        data=io.BytesIO()
        with wave.open(data,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','not compressed'));w.writeframes(b'\1\0'*4)
        self.raw=data.getvalue();self.path.write_bytes(self.raw);self.path.chmod(0o600)
        self.root.chmod(0o700)
        self.spec=SimpleNamespace(reference_bytes=len(self.raw),reference_sha256=hashlib.sha256(self.raw).hexdigest(),reference_rate=24000,reference_frames=4)
        real=os.fstat
        def owned(fd):
            s=real(fd)
            return SimpleNamespace(**dict({k:getattr(s,k) for k in dir(s) if k.startswith('st_')},st_uid=10001,st_gid=10001))
        self.enterContext(patch.object(os,'fstat',side_effect=owned))

    def test_open_one_private_no_follow_asset_and_readonly_child_descriptor(self):
        with patch.object(m,'ROOT',str(self.root)):
            fd=m.open_reference('asset-a')
        try:
            self.assertEqual(read_reference_fd(fd,self.spec,owner_uid=10001),self.raw)
            with self.assertRaises(OSError):os.write(fd,b'x')
            self.assertEqual(os.fstat(fd).st_mode & 0o777,0o600)
            self.spec.reference_sha256='0'*64
            with self.assertRaisesRegex(Refused,'digest'):read_reference_fd(fd,self.spec,owner_uid=10001)
        finally:os.close(fd)

    def test_wrong_modes_owner_hardlink_symlink_and_traversal_refuse(self):
        with patch.object(m,'ROOT',str(self.root)):
            for identity in ('../asset-a','a/b','', '/asset-a'):
                with self.assertRaises(Refused):m.open_reference(identity)
            self.path.chmod(0o644)
            with self.assertRaises(Refused):m.open_reference('asset-a')
            self.path.chmod(0o600)
            self.root.chmod(0o755)
            with self.assertRaises(Refused):m.open_reference('asset-a')
            self.root.chmod(0o700)
            os.link(self.path,self.root/'linked')
            with self.assertRaises(Refused):m.open_reference('asset-a')
            (self.root/'linked').unlink()
            (self.root/'link').symlink_to(self.path)
            with self.assertRaises(OSError):m.open_reference('link')
            real=os.fstat
            def foreign(fd):
                s=real(fd);return SimpleNamespace(**dict(vars(s),st_uid=20000))
            with patch.object(os,'fstat',side_effect=foreign):
                with self.assertRaisesRegex(Refused,'ownership|root'):m.open_reference('asset-a')

    def test_writable_descriptor_refused_even_with_correct_metadata(self):
        fd=os.open(self.path,os.O_RDWR)
        try:
            with self.assertRaisesRegex(Refused,'descriptor'):m.check_fd(fd)
        finally:os.close(fd)

    def transfer(self,code=0):
        fd=os.open(self.path,os.O_RDONLY)
        parent=Mock();child=Mock();child.fileno.return_value=19
        parent.recvmsg.return_value=(b'R',[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[fd]).tobytes())],0,None)
        proc=Mock(returncode=None)
        def reaped(**kwargs):proc.returncode=code;return code
        proc.wait.side_effect=reaped
        with patch.object(socket,'socketpair',return_value=(parent,child)), \
             patch.object(socket,'MSG_CMSG_CLOEXEC',getattr(socket,'MSG_CMSG_CLOEXEC',0),create=True), \
             patch.object(m.subprocess,'Popen',return_value=proc) as launch:
            if code:
                with self.assertRaises(Refused):m.acquire('asset-a')
                with self.assertRaises(OSError):os.fstat(fd)
            else:
                result=m.acquire('asset-a');self.assertEqual(result,fd);os.close(result)
            kw=launch.call_args.kwargs
            self.assertEqual((kw['user'],kw['group'],kw['extra_groups'],kw['pass_fds']),(10001,10001,[],(19,)))
            self.assertIn('asset-a',launch.call_args.args[0])
            proc.wait.assert_called_once_with(timeout=1)
            parent.close.assert_called_once()

    def test_reaped_helper_transfers_exact_fd(self):self.transfer()
    def test_helper_failure_closes_fd(self):self.transfer(code=2)
