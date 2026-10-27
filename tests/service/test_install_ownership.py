"""Source/injected ownership observations; NOT real multi-UID permission evidence."""
import os
import fcntl
import json
from pathlib import Path
import sqlite3
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from service import install_policy as policy
from service import persistent_init, recovery
from worker.runtime_journal import read_snapshot, journal_bytes


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

    def split_fixture(self):
        # Real files and SQLite; only owner observations are translated to this
        # unprivileged Mac test identity. No setuid/setgid/chown is performed.
        roots = {role:self.root/role for role in policy.ROLES}
        for path in roots.values():path.mkdir(mode=0o700)
        (roots['control']/'core').mkdir(mode=0o700)
        for role,path in roots.items():
            path.chmod(policy.expectation(policy.SPLIT,role,directory=True)[2])
        for path in (roots['control']/'core'/'control.lock',roots['receipts']/'lane.lock'):
            path.touch(mode=0o600)
        path=roots['control']/'core'/'core.sqlite'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE retained(value TEXT)')
            db.execute('INSERT INTO retained VALUES(?)',('saved voice/artifact sentinel',))
        path.chmod(0o600)
        (roots['receipts']/'journal.json').write_bytes(journal_bytes(dict(version=1,lane='a'*32,records=[])))
        (roots['receipts']/'journal.json').chmod(0o600)
        return roots

    def translated_roles(self):
        fixed=policy.expectation
        uid,gid=os.getuid(),os.getgid()
        def observed(selected,role,**kwargs):
            _,_,mode=fixed(selected,role,**kwargs)
            return uid,gid,mode
        return patch.object(policy,'expectation',side_effect=observed)

    def retained_bytes(self,roots):
        return {str(p):p.read_bytes() for root in roots.values() for p in root.rglob('*') if p.is_file()}

    def test_split_offline_copy_restart_and_journal_are_read_only(self):
        roots=self.split_fixture();before=self.retained_bytes(roots)
        original_connect=sqlite3.connect;opened=[]
        def connect(path,*args,**kwargs):
            opened.append(str(path));return original_connect(path,*args,**kwargs)
        with self.translated_roles(),patch.object(sqlite3,'connect',side_effect=connect):
            for _ in range(2):
                with recovery.offline(roots,policy=policy.SPLIT) as (_,db):
                    self.assertEqual(db.execute('SELECT value FROM retained').fetchall(),[('saved voice/artifact sentinel',)])
                    with self.assertRaises(sqlite3.OperationalError):db.execute('DELETE FROM retained')
                    value,maintenance=read_snapshot(roots['receipts'],policy=policy.SPLIT)
                    self.assertEqual(value,dict(version=1,lane='a'*32,records=[]));self.assertIsNone(maintenance)
            self.assertTrue(opened)
            self.assertTrue(all('p1-inspect-' in path for path in opened))
        self.assertEqual(before,self.retained_bytes(roots))

    def test_split_write_refuses_before_any_filesystem_access(self):
        with patch.object(recovery,'_locked_roots',side_effect=AssertionError('opened roots')):
            with self.assertRaisesRegex(ValueError,'installed_original_writer_required'):
                with recovery.offline({},write=True,policy=policy.SPLIT):self.fail('entered')

    def test_split_control_then_lane_exclusion_and_exact_modes(self):
        roots=self.split_fixture();before=self.retained_bytes(roots)
        for role,path in [('control',roots['control']/'core'/'control.lock'),('receipts',roots['receipts']/'lane.lock')]:
            with self.subTest(role=role),path.open('rb') as lock,self.translated_roles():
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'maintenance_lock_busy'):
                    with recovery.offline(roots,policy=policy.SPLIT):self.fail('entered')
        # Registration may omit only the lane lock, never control exclusion.
        with (roots['receipts']/'lane.lock').open('rb') as lock,self.translated_roles():
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with recovery.offline(roots,policy=policy.SPLIT,journal_lock=False):pass
        path=roots['receipts']/'lane.lock';path.chmod(0o644)
        with self.translated_roles(),self.assertRaisesRegex(ValueError,'installed_role_identity'):
            with recovery.offline(roots,policy=policy.SPLIT):self.fail('entered')
        path.chmod(0o600);self.assertEqual(before,self.retained_bytes(roots))

    def test_split_lock_exchange_root_exchange_and_hardlink_refuse(self):
        roots=self.split_fixture()
        with self.translated_roles():
            lock=roots['control']/'core'/'control.lock'
            with self.assertRaisesRegex(ValueError,'snapshot_source_changed') as raised:
                with recovery.offline(roots,policy=policy.SPLIT):
                    lock.rename(lock.with_name('retained.lock'));lock.touch(mode=0o600)
            self.assertIn('lock revalidation failed: installed_lock_changed',raised.exception.__notes__)
            self.assertTrue(lock.with_name('retained.lock').exists())
            with self.assertRaisesRegex(ValueError,'installed_root_changed'):
                with recovery.offline(roots,policy=policy.SPLIT):
                    roots['staging'].rename(self.root/'old-staging')
                    roots['staging'].mkdir(mode=0o1733);roots['staging'].chmod(0o1733)
            os.link(lock,lock.with_name('alias.lock'))
            with self.assertRaisesRegex(ValueError,'installed_role_identity'):
                with recovery.offline(roots,policy=policy.SPLIT):self.fail('entered')

    def test_split_journal_reuses_closed_validator_and_preserves_partial(self):
        roots=self.split_fixture();path=roots['receipts']/'journal.json'
        partial=b'{partial';path.write_bytes(partial)
        with self.translated_roles(),self.assertRaises(json.JSONDecodeError):
            read_snapshot(roots['receipts'],policy=policy.SPLIT)
        self.assertEqual(path.read_bytes(),partial)
        value=dict(version=1,lane='a'*32,records=[dict(id='bad')])
        path.write_bytes(journal_bytes(value));before=path.read_bytes()
        with self.translated_roles(),self.assertRaisesRegex(ValueError,'invalid_record'):
            read_snapshot(roots['receipts'],policy=policy.SPLIT)
        self.assertEqual(path.read_bytes(),before)

    def test_root_reader_reconciliation_uses_control_context_only(self):
        from service.admission import DEFAULT_BUDGET
        from service.reconciliation import Reconciliation
        roots={role:self.root/role for role in policy.ROLES}
        for path in roots.values():path.mkdir(mode=0o700)
        scope=dict(daemon='d',context='c',volumes={role:dict(name=role,driver='local',created='now') for role in roots})
        profiles=json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['profiles']
        persistent_init.initialize(roots,scope,profiles,DEFAULT_BUDGET)
        for role,path in roots.items():path.chmod(policy.expectation(policy.SPLIT,role,directory=True)[2])
        # Legacy Core follows the caller umask; this fixture supplies installed
        # modes explicitly. This is not a split initialization receipt.
        for path in roots['control'].rglob('*'):path.chmod(0o700 if path.is_dir() else 0o600)
        with self.translated_roles():
            with recovery.offline(roots,policy=policy.SPLIT) as (_,db):
                scanner=Reconciliation.__new__(Reconciliation)
                scanner.core=SimpleNamespace(root=roots['control']/'core');scanner.db=db;scanner.staging=roots['staging']
                scanner.entry_limit=2048;scanner.byte_limit=16*1024**2;scanner.seconds=1
                # Keep control and root identities distinct even in root CI.
                # Inject only ownership; retain real modes, links, inode/device,
                # timestamps and canonical descriptor/path validation.
                control_uid,control_gid,root_uid=10001,10001,0
                real_stat,real_fstat=os.stat,os.fstat
                expectation=policy.expectation
                def owned(observe):
                    def observed(*args,**kwargs):
                        info=observe(*args,**kwargs)
                        fields={name:getattr(info,name) for name in dir(info) if name.startswith('st_')}
                        fields.update(st_uid=control_uid,st_gid=control_gid)
                        return SimpleNamespace(**fields)
                    return observed
                def control_expectation(selected,role,**kwargs):
                    return control_uid,control_gid,expectation(selected,role,**kwargs)[2]
                with patch.object(os,'getuid',return_value=root_uid), \
                     patch.object(os,'stat',side_effect=owned(real_stat)), \
                     patch.object(os,'fstat',side_effect=owned(real_fstat)), \
                     patch.object(policy,'expectation',side_effect=control_expectation):
                    scanner._collect(policy=policy.SPLIT)
                    self.assertEqual(scanner.reasons,set())
                    scanner._collect()
                    self.assertIn('unknown_control_namespace',scanner.reasons)

    def test_legacy_maintenance_entry_refusals_do_not_open_or_spawn(self):
        roots=self.split_fixture()
        marker=roots['receipts']/'p1-init.json'
        marker.write_text(json.dumps(dict(scope=policy.SCOPE)));marker.chmod(0o600)
        before=self.retained_bytes(roots)
        # These calls must refuse before reaching any HELD/maintenance work.
        calls=[
            lambda:recovery._capture_original_snapshot(roots,None,1),
            lambda:recovery._maintenance_session(roots,{},None).__enter__(),
            lambda:recovery._run_bounded_maintenance(roots,{},None),
            lambda:recovery._pending_prefix(roots,{}, {},{}),
            lambda:recovery._pending_prefix_locked(roots,None,{}, {},{},now=1),
            lambda:recovery._prepare_abandon(roots,{},profiles=[],budget={},diagnostic_decisions=[]),
            lambda:recovery._rehearse_preparation(SimpleNamespace(roots=roots,db=None),None,None,None),
        ]
        with patch.object(sqlite3,'connect',side_effect=AssertionError('SQLite opened')):
            for call in calls:
                with self.subTest(entry=call),self.assertRaisesRegex(ValueError,'legacy_recovery_scope_required'):call()
            with self.translated_roles(),self.assertRaisesRegex(ValueError,'legacy_recovery_scope_required'):
                with recovery.offline(roots,write=True):self.fail('entered')
        self.assertEqual(before,self.retained_bytes(roots))

    def test_legacy_writer_rejects_split_database_before_original_connection(self):
        roots=self.split_fixture();path=roots['control']/'core'/'core.sqlite'
        marker=roots['receipts']/'p1-init.json'
        marker.write_text(json.dumps(dict(scope='owned-synthetic')));marker.chmod(0o600)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE p1_installation(scope TEXT,record TEXT)')
            db.execute('INSERT INTO p1_installation VALUES(?,?)',(policy.SCOPE,json.dumps(dict(scope=policy.SCOPE))))
        # The default legacy root validator requires uniform ownership only.
        before=self.retained_bytes(roots);original=sqlite3.connect;opened=[]
        def connect(path,*args,**kwargs):
            opened.append(str(path));return original(path,*args,**kwargs)
        with patch.object(sqlite3,'connect',side_effect=connect):
            with self.assertRaisesRegex(ValueError,'legacy_recovery_scope_required'):
                with recovery.offline(roots,write=True):self.fail('entered')
        self.assertTrue(opened);self.assertTrue(all('p1-inspect-' in p for p in opened))
        self.assertEqual(before,self.retained_bytes(roots))


if __name__ == '__main__': unittest.main()
