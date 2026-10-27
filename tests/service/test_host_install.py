"""Injected operator adapter checks, not actual Docker/Linux qualification."""
import importlib.util
from pathlib import Path
import unittest

from service import host_install


class HostInstall(unittest.TestCase):
    def test_cli_adapter_uses_extracted_implementation(self):
        spec = importlib.util.spec_from_file_location('operator_adapter', Path('scripts/p1-operator.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertIs(module.main, host_install.main)
        for name in ('host_state', 'register_host', 'host_inventory', 'host_preflight', 'freeze_host'):
            self.assertEqual(getattr(host_install, name).__module__, 'service.host_install')

    def test_fixed_docker_context_and_deadline_before_subprocess(self):
        for value in ('', '-host', 'a b', 'x;echo', None):
            with self.assertRaises(ValueError): host_install.Docker(value)
        observer = host_install.Docker('local'); observer.deadline = 0
        with self.assertRaisesRegex(ValueError, 'docker_global_deadline'):
            observer._read(['container', 'ls'])

    def test_legacy_mount_disjointness_is_still_closed(self):
        # Real temp inode; unrelated bind storage remains an explicit refusal.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            mount = dict(name='', source=str(Path(directory).resolve()), type='bind')
            with self.assertRaisesRegex(ValueError, 'unsupported_host_storage_disjointness'):
                host_install._mount_touches(mount, [])

    def test_fd_hello_refuses_root_before_any_key_or_socket(self):
        from unittest.mock import patch
        from service import supervisor_runtime as runtime
        with patch.object(runtime.sys, 'platform', 'linux'), patch.object(runtime.os, 'getuid', return_value=0), patch.object(runtime.os, 'open', side_effect=AssertionError('key opened')):
            with self.assertRaisesRegex(ValueError, 'dropped_control_identity_required'):
                runtime.installed_hello_at(123)

    def test_fd_hello_injected_identity_preserves_peer_auth_and_hello_only(self):
        # No setuid, socket bind/listen, or actual peer credential assertion here.
        import os
        import stat
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch, MagicMock
        from service import supervisor_runtime as runtime, install_policy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); root.chmod(0o750)
            key = root/'lifecycle.key'; key.write_bytes(b'a'*64); key.chmod(0o640)
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                info = key.stat()
                endpoint = SimpleNamespace(st_mode=stat.S_IFSOCK | 0o660, st_uid=0, st_gid=10001,
                    st_nlink=1, st_dev=info.st_dev, st_ino=99, st_size=0, st_mtime_ns=0, st_ctime_ns=0)
                real_stat = os.stat; real_check = install_policy.check
                def observed(path, *a, **kw):
                    return endpoint if path == 'lifecycle.sock' else real_stat(path, *a, **kw)
                def check(info, policy, role, **kw):
                    uid, gid, mode = install_policy.expectation(policy, role, **kw)
                    real_check(SimpleNamespace(st_uid=uid, st_gid=gid, st_mode=info.st_mode, st_nlink=info.st_nlink), policy, role, **kw)
                hello = dict(lane='b'*32, boot='c'*32, quarantined=False, launches=0)
                connection = MagicMock(); connection.__enter__.return_value = connection
                with patch.object(runtime.sys, 'platform', 'linux'), patch.multiple(runtime.os,
                        getuid=lambda:10001, geteuid=lambda:10001, getgid=lambda:10001,
                        getegid=lambda:10001, getgroups=lambda:[]), patch.object(runtime.os, 'stat', side_effect=observed), patch.object(install_policy, 'check', side_effect=check), patch.object(runtime.socket, 'socket', return_value=connection), patch.object(runtime, 'authenticate') as auth, patch.object(runtime, 'send') as send, patch.object(runtime, 'receive', return_value=dict(ok=True,value=hello)):
                    self.assertEqual(runtime.installed_hello_at(fd), hello)
                    connection.connect.assert_called_once_with('/proc/self/fd/'+str(fd)+'/lifecycle.sock')
                    auth.assert_called_once()
                    self.assertEqual(auth.call_args.args[1], runtime.session('a'*64, 0, 10001))
                    self.assertIs(auth.call_args.kwargs['control'], True)
                    self.assertEqual(send.call_args.args[1], dict(version=runtime.VERSION,op='hello',body={}))
            finally:
                os.close(fd)

    def test_historical_capsule_projection_and_refusals_direct_unittest(self):
        # Call only named injected historical cases; no pytest runner/guard.
        import tempfile
        import pytest
        from tests.service import test_recovery as old
        faults=['unknown','nested','bool','nonfinite','duplicate-registration','missing-wait','duplicate-wait',
                'wait-failed','E1','forged-hash','forged-label','omitted-provenance','bad-cgroup','same-pidns','expiry','oversize','duration','finished']
        for fault in ['positive',*faults]:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory,pytest.MonkeyPatch.context() as monkeypatch:
                root=Path(directory).resolve();installation=old.installation.__wrapped__(root)
                host=old.b23_host.__wrapped__(installation,root,monkeypatch)
                if fault=='positive':old.test_b23_closed_capsule_freeze_and_apply_closed(host)
                else:old.test_b23_closed_capsule_negative_matrix(host,fault)

    def test_historical_bounded_docker_and_injected_proc(self):
        import tempfile
        import pytest
        from tests.service import test_recovery as old
        for fault in ['none','stdout','stderr','exit','spawn','global-timeout','process-timeout']:
            # Each historical Popen is injected.
            with self.subTest(adapter='docker',fault=fault),pytest.MonkeyPatch.context() as monkeypatch:
                old.test_b23_fixed_subprocess_bounds_and_sanitized_errors(monkeypatch,fault)
        for fault in ['none','pid-reuse','foreign-cgroup','shared-namespace','child-namespace','child-escape','ipc-alias','cgroup-symlink','oversize','deadline','no-cgroup-v2']:
            with self.subTest(adapter='proc',fault=fault),tempfile.TemporaryDirectory() as directory,pytest.MonkeyPatch.context() as monkeypatch:
                old.test_b23_local_proc_tree_os_boundary(Path(directory),monkeypatch,fault)


if __name__ == '__main__': unittest.main()
