"""Source/injected ownership observations; NOT real multi-UID permission evidence."""
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from service import install_policy as policy
from service import persistent_init, recovery


class InstallOwnership(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def info(self, role, *, directory=False, **changes):
        uid, gid, mode = policy.expectation(policy.SPLIT, role, directory=directory)
        value = dict(st_uid=uid, st_gid=gid,
                     st_mode=(stat.S_IFDIR if directory else stat.S_IFREG) | mode, st_nlink=1)
        value.update(changes)
        return SimpleNamespace(**value)

    def test_closed_fixed_roles_and_modes(self):
        for role in policy.ROLES:
            policy.check(self.info(role, directory=True), policy.SPLIT, role, directory=True)
        for role in ('control', 'receipts', 'ipc'):
            policy.check(self.info(role), policy.SPLIT, role)
            for field, value in [('st_uid', 20000), ('st_gid', 20000),
                                 ('st_mode', stat.S_IFREG | 0o666), ('st_nlink', 2)]:
                with self.subTest(role=role, field=field), self.assertRaises(ValueError):
                    policy.check(self.info(role, **{field:value}), policy.SPLIT, role)
        with self.assertRaises(ValueError): policy.expectation(policy.SPLIT, 'staging')
        for selection in (policy.POLICY, policy.SCOPE, object(), policy._SplitPolicy()):
            with self.assertRaises(ValueError): policy.require(selection)
        with self.assertRaises(ValueError):
            policy.check(self.info('receipts', directory=True, st_mode=stat.S_IFDIR | 0o700),
                         policy.SPLIT, 'receipts', directory=True)

    def injected_owner(self, info, selected, role, **kwargs):
        # Only UID/GID are injected. Real descriptor type/mode/link checks remain.
        uid, gid, _ = policy.expectation(selected, role, **kwargs)
        observed = SimpleNamespace(st_uid=uid, st_gid=gid, st_mode=info.st_mode,
                                   st_nlink=info.st_nlink)
        return self.real_check(observed, selected, role, **kwargs)

    def owner_patch(self):
        self.real_check = policy.check
        return patch.object(policy, 'check', side_effect=self.injected_owner)

    def file(self, name='private', data=b'private'):
        path = self.root/name; path.write_bytes(data); path.chmod(0o600)
        return path

    def test_private_read_and_legacy_default(self):
        path = self.file()
        self.assertEqual(persistent_init.read_file(path), b'private')
        with self.owner_patch():
            self.assertEqual(persistent_init.read_file(path, policy=policy.SPLIT, role='control'), b'private')
            with self.assertRaises(ValueError):
                persistent_init.read_file(path, 2, policy=policy.SPLIT, role='control')
        with self.assertRaises(ValueError): persistent_init.read_file(path, role='control')

    def test_symlink_ancestor_leaf_and_hardlink_refuse(self):
        path = self.file()
        link = self.root/'link'; link.symlink_to(path)
        alias = self.root/'alias'; alias.symlink_to(self.root, target_is_directory=True)
        with self.owner_patch():
            for candidate in (link, alias/'private'):
                with self.assertRaises(OSError):
                    policy.read_private(candidate, 100, policy=policy.SPLIT, role='control')
            os.link(path, self.root/'hard')
            with self.assertRaises(ValueError):
                policy.read_private(path, 100, policy=policy.SPLIT, role='control')

    def test_exchange_during_read_refuses_and_preserves_both_files(self):
        path = self.file(); replacement = self.file('replacement', b'changed')
        original_read = os.read; swapped = False
        def read(fd, size):
            nonlocal swapped
            data = original_read(fd, size)
            if not swapped:
                swapped = True; path.rename(self.root/'retained'); replacement.rename(path)
            return data
        with self.owner_patch(), patch.object(os, 'read', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'private_file_changed'):
                policy.read_private(path, 100, policy=policy.SPLIT, role='control')
        self.assertEqual((self.root/'retained').read_bytes(), b'private')
        self.assertEqual(path.read_bytes(), b'changed')

    def test_private_sqlite_copy_never_opens_original_and_detects_mutation(self):
        path = self.root/'core.sqlite'
        with sqlite3.connect(path) as db: db.execute('CREATE TABLE sample(id)')
        path.chmod(0o600); before = path.read_bytes()
        with self.owner_patch():
            with recovery.private_database(path, policy=policy.SPLIT) as private:
                self.assertNotEqual(private, path)
                self.assertEqual(stat.S_IMODE(private.parent.stat().st_mode), 0o700)
                with sqlite3.connect(private) as db:
                    self.assertEqual(db.execute('SELECT * FROM sample').fetchall(), [])
            self.assertFalse(private.exists())
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'snapshot_source_changed'):
                with recovery.private_database(path, policy=policy.SPLIT):
                    with path.open('ab') as stream: stream.write(b'mutation')
        self.assertEqual(path.read_bytes(), before+b'mutation')

    def test_no_enrollment_or_activation_from_policy_primitives(self):
        roots = {role:self.root/role for role in policy.ROLES}
        for path in roots.values(): path.mkdir(mode=0o700)
        with self.assertRaises(ValueError): persistent_init.roots_checked(roots, policy=policy.SPLIT)
        self.assertTrue(all(not list(path.iterdir()) for path in roots.values()))


if __name__ == '__main__': unittest.main()
