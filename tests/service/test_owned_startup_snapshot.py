"""Real SQLite owner/backup tests; Linux lock evidence is container-local only."""
import builtins
from contextlib import ExitStack
import io
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from service import recovery


class OwnedStartupSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='owned-startup-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.path=self.root/'core.sqlite'
        self.db=sqlite3.connect(self.path,isolation_level=None)
        self.addCleanup(self.db.close)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE facts(value TEXT)')
        self.db.execute('INSERT INTO facts VALUES (?)',('retained-value',))
        self.originals={self.root/name for name in ('core.sqlite','core.sqlite-wal','core.sqlite-shm')}
        for path in self.originals:
            if path.exists():path.chmod(0o600)

    def usable(self):
        self.assertEqual(self.db.execute('SELECT value FROM facts').fetchall(),[('retained-value',)])
        self.db.execute('INSERT INTO facts VALUES (?)',('continued-owner',))
        self.db.execute("DELETE FROM facts WHERE value='continued-owner'")

    def locks(self):
        inodes={str(path.stat().st_ino) for path in self.originals if path.exists()}
        with open('/proc/locks') as stream: lines=stream.read().splitlines()
        return {tuple(fields[1:]) for line in lines if len(fields:=line.split())>=8 and
                fields[4]==str(os.getpid()) and fields[5].split(':')[-1] in inodes}

    def test_backup_never_raw_opens_originals_and_keeps_owner_locks(self):
        before=self.locks() if sys.platform=='linux' else None
        if before is not None:self.assertTrue(before,'SQLite owner should hold local POSIX locks')
        with ExitStack() as stack:
            def guarded(original):
                def call(path,*args,**kwargs):
                    if not isinstance(path,int) and Path(os.fspath(path)).absolute() in self.originals:
                        raise AssertionError('raw original descriptor forbidden')
                    return original(path,*args,**kwargs)
                return call
            stack.enter_context(patch.object(builtins,'open',side_effect=guarded(builtins.open)))
            stack.enter_context(patch.object(io,'open',side_effect=guarded(io.open)))
            stack.enter_context(patch.object(os,'open',side_effect=guarded(os.open)))
            stack.enter_context(patch.object(recovery,'read_file',side_effect=AssertionError('raw snapshot forbidden')))
            original_connect=sqlite3.connect
            def connect(path,*args,**kwargs):
                if str(path).split('?',1)[0].removeprefix('file:')==str(self.path):
                    raise AssertionError('second source SQLite connection forbidden')
                return original_connect(path,*args,**kwargs)
            stack.enter_context(patch.object(sqlite3,'connect',side_effect=connect))
            with recovery.startup_database(self.path,owned_db=self.db) as target:
                self.assertNotEqual(target,self.path)
                with sqlite3.connect(target.as_uri()+'?mode=ro',uri=True) as copy:
                    self.assertEqual(copy.execute('SELECT value FROM facts').fetchall(),[('retained-value',)])
            if before is not None:self.assertEqual(self.locks(),before)
        self.usable()

    def test_wrong_main_attached_database_and_transaction_refuse(self):
        other=sqlite3.connect(self.root/'other.sqlite',isolation_level=None)
        try:
            with self.assertRaisesRegex(ValueError,'owned_snapshot_main_database_required'):
                with recovery.startup_database(self.path,owned_db=other):self.fail('wrong main admitted')
        finally:other.close()
        self.db.execute("ATTACH DATABASE ':memory:' AS other")
        with self.assertRaisesRegex(ValueError,'owned_snapshot_main_database_required'):
            with recovery.startup_database(self.path,owned_db=self.db):self.fail('attachment admitted')
        self.db.execute('DETACH DATABASE other')
        self.db.execute('BEGIN')
        with self.assertRaisesRegex(ValueError,'owned_snapshot_connection_required'):
            with recovery.startup_database(self.path,owned_db=self.db):self.fail('transaction admitted')
        self.db.rollback()
        self.usable()

    def test_integrity_check_implicit_empty_temp_preserves_owner_backup(self):
        self.assertEqual(self.db.execute('PRAGMA integrity_check').fetchall(), [('ok',)])
        self.assertIn((1,'temp',''), self.db.execute('PRAGMA database_list').fetchall())
        before=self.locks() if sys.platform=='linux' else None
        with recovery.startup_database(self.path,owned_db=self.db) as target:
            with sqlite3.connect(target.as_uri()+'?mode=ro',uri=True) as copy:
                self.assertEqual(copy.execute('SELECT value FROM facts').fetchall(), [('retained-value',)])
        if before is not None:self.assertEqual(self.locks(),before)
        self.usable()
        self.db.execute('CREATE TEMP TABLE transient(value)')
        with self.assertRaisesRegex(ValueError,'owned_snapshot_main_database_required'):
            with recovery.startup_database(self.path,owned_db=self.db):self.fail('temporary objects admitted')

    def test_changed_source_identity_refuses_without_closing_owner(self):
        original=os.stat; changed=False
        def observed(path,*args,**kwargs):
            info=original(path,*args,**kwargs)
            if changed and not isinstance(path,int) and Path(path).absolute()==self.path:
                return SimpleNamespace(**dict({name:getattr(info,name) for name in dir(info) if name.startswith('st_')},st_ino=info.st_ino+1))
            return info
        with patch.object(os,'stat',side_effect=observed), self.assertRaisesRegex(ValueError,'snapshot_source_changed'):
            with recovery.startup_database(self.path,owned_db=self.db):changed=True
        self.usable()

    def test_physical_size_cap_refuses_and_owner_remains_usable(self):
        with patch.object(recovery,'MAX_BYTES',1), self.assertRaisesRegex(ValueError,'snapshot_byte_budget'):
            with recovery.startup_database(self.path,owned_db=self.db):self.fail('oversize admitted')
        self.usable()

    def test_backup_deadline_refuses_and_owner_remains_usable(self):
        # Capture uses several time checks; expire at the backup progress callback.
        original=sqlite3.connect
        def connecting(*args,**kwargs):
            result=original(*args,**kwargs)
            clock[0]=3.0
            return result
        clock=[0.0]
        with patch.object(recovery.time,'monotonic',side_effect=lambda:clock[0]), patch.object(sqlite3,'connect',side_effect=connecting):
            with self.assertRaisesRegex(ValueError,'snapshot_deadline'):
                with recovery.startup_database(self.path,owned_db=self.db):self.fail('expired backup admitted')
        self.usable()

    def test_preowner_reader_remains_the_existing_raw_copy(self):
        sentinel=self.root/'private.sqlite'
        from contextlib import contextmanager
        @contextmanager
        def existing(path,*,policy=None):
            self.assertEqual(path,self.path)
            yield sentinel
        with patch.object(recovery,'private_database',side_effect=existing) as reader:
            with recovery.startup_database(self.path) as target:self.assertEqual(target,sentinel)
        reader.assert_called_once_with(self.path,policy=None)
