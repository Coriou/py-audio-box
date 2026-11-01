"""Injected namespace operations, real SQLite canonicalization; no privileged OS calls."""
from contextlib import contextmanager
import sys
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from service import install_child as child


class Namespace:
    """Emulate bind aliases with directory relocation, never symlink aliases.

    A temporary prefix represents the child namespace's /. SQLite must resolve
    inside that view. Relocation preserves the inherited device/inode identity.
    This is not a kernel mount, traversal-permission, or multi-UID proof.
    """
    def __init__(self, fixture, fail=None):
        self.fixture = fixture
        self.base = fixture.root / 'namespace' / 'mnt'
        self.base.mkdir(parents=True)
        self.events = []
        self.moves = []
        self.fail = fail

    def event(self, name):
        self.events.append(name)
        if self.fail == name: raise OSError('injected ' + name)

    def stat(self, path):
        if path == self.base:
            return SimpleNamespace(st_uid=0, st_mode=0o40755)
        return os.stat(path, follow_symlinks=False)

    def unshare(self): self.event('unshare')
    def mkdir(self, path): os.mkdir(path, 0o755)
    def mount(self, source, target, kind, flags, data=None):
        self.event(('private' if target == '/' else 'tmpfs') if flags != 4096 else 'bind-' + Path(target).name)
        if flags == 4096:
            fd = int(source.rsplit('/', 1)[1])
            original = self.fixture.fds[fd]
            os.rmdir(target)
            os.rename(original, target)
            self.moves.append((original, target))
            self.fixture.fds[fd] = Path(target)

    @contextmanager
    def view(self):
        old_roots = self.fixture.roots.copy()
        old_fds = self.fixture.fds.copy()
        try:
            with patch.multiple(child, BASE=self.base, CONTROL=self.base/'pab-install/control',
                                STAGING=self.base/'pab-install/staging'):
                yield self
        finally:
            for original, target in reversed(self.moves): os.rename(target, original)
            self.fixture.roots.update(old_roots)
            self.fixture.fds.update(old_fds)
            import shutil
            shutil.rmtree(self.base.parent)


class Canonicalization(unittest.TestCase):
    def test_fd_style_symlink_canonicalizes_to_backing_path(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir='/private/tmp' if sys.platform == 'darwin' else '/tmp') as temp:
            root = Path(temp); (root/'backing').mkdir()
            (root/'fd-alias').symlink_to(root/'backing', target_is_directory=True)
            with sqlite3.connect(root/'fd-alias/db') as db:
                self.assertEqual(db.execute('pragma database_list').fetchone()[2], str(root/'backing/db'))

    def test_linux_guard_has_no_privileged_call_on_other_platform(self):
        with patch.object(child.sys, 'platform', 'darwin'):
            with self.assertRaisesRegex(ValueError, 'linux_root'): child.LinuxMounts()
