"""Stopped SOURCE preparation only. Tiny files/SQLite, injected TLS/processes."""
import copy
import os
import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from service import native_renewal as r
from service import native_package as n
from service.core import Core, native_maintenance
from tests.release.test_native_history import NativeHistoryTests
from worker.transport import Refused, canonical
from worker.native_supervisor import Journal


class RenewalFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.root.chmod(0o700)

    def test_publish_exact_resume_and_conflict_retention(self):
        path=self.root/'document'
        r.publish(path,dict(value=1))
        r.publish(path,dict(value=1))
        saved=path.read_bytes()
        with self.assertRaises(Refused): r.publish(path,dict(value=True))
        self.assertEqual(path.read_bytes(),saved)

    def test_partial_stage_retained(self):
        path=self.root/'document'; stage=self.root/'document.stage'
        stage.write_bytes(b'{'); stage.chmod(0o600)
        with self.assertRaises(Refused): r.publish(path,dict(value=1))
        self.assertEqual(stage.read_bytes(),b'{'); self.assertFalse(path.exists())

    def test_exact_stage_resumes(self):
        path=self.root/'document'; stage=self.root/'document.stage'
        stage.write_bytes(canonical(dict(value=1))); stage.chmod(0o600)
        r.publish(path,dict(value=1))
        self.assertFalse(stage.exists()); self.assertEqual(r.document(path),dict(value=1))

    def test_rename_failure_preserves_owned_staging(self):
        path=self.root/'document'
        with patch.object(r.os,'rename',side_effect=OSError('injected rename')):
            with self.assertRaises(OSError): r.publish(path,dict(value=1))
        self.assertEqual((self.root/'document.stage').read_bytes(),canonical(dict(value=1)))
        r.publish(path,dict(value=1))

    def test_fsync_failure_retains_then_explicit_resume(self):
        path=self.root/'document'
        with patch.object(r.os,'fsync',side_effect=OSError('injected fsync')):
            with self.assertRaises(OSError): r.publish(path,dict(value=1))
        self.assertTrue((self.root/'document.stage').exists())
        r.publish(path,dict(value=1))

    def test_write_failure_keeps_partial_without_repair(self):
        path=self.root/'document'
        real=os.fdopen
        class FailedWrite:
            def __init__(self, stream): self.stream=stream
            def __enter__(self): return self
            def __exit__(self,*args): self.stream.close()
            def write(self,raw): raise OSError('injected write')
        with patch.object(r.os,'fdopen',side_effect=lambda *a,**k:FailedWrite(real(*a,**k))):
            with self.assertRaises(OSError): r.publish(path,dict(value=1))
        self.assertEqual((self.root/'document.stage').read_bytes(),b'')
        with self.assertRaises(Refused): r.publish(path,dict(value=1))

    def test_visible_final_fsync_failure_explicit_resume(self):
        path=self.root/'document';real=os.fsync;calls=0
        def fail(fd):
            nonlocal calls
            calls+=1
            if calls==3: raise OSError('injected final fsync')
            return real(fd)
        with patch.object(r.os,'fsync',side_effect=fail):
            with self.assertRaises(OSError): r.publish(path,dict(value=1))
        self.assertTrue(path.exists());self.assertFalse((self.root/'document.stage').exists())
        r.publish(path,dict(value=1))

    def test_symlink_hardlink_and_duplicate_keys(self):
        source=self.root/'source'; source.write_bytes(b'{"a":1,"a":1}');source.chmod(0o600)
        with self.assertRaises(Refused): r.document(source)
        path=self.root/'document';path.symlink_to(source)
        with self.assertRaises(OSError) as raised: r.publish(path,dict(value=1))
        import errno
        self.assertEqual(raised.exception.errno,errno.ELOOP)
        self.assertTrue(path.is_symlink())
        self.assertEqual(source.read_bytes(),b'{"a":1,"a":1}')
        path.unlink();os.link(source,path)
        with self.assertRaises(Refused): r.publish(path,dict(value=1))

    def test_empty_unknown_namespace_blocks_core_before_schema(self):
        owner=self.root/'owner'; core=Core(owner);core.close()
        directory=owner/'native-renewal';directory.mkdir(mode=0o700)
        with self.assertRaises(Refused): Core(owner)
        marker=directory/'foreign';marker.write_bytes(b'RETAIN')
        with self.assertRaises(Refused): Core(owner)
        self.assertEqual(marker.read_bytes(),b'RETAIN')

    def test_maintenance_same_registry_readonly_and_no_second_core(self):
        owner=self.root/'owner';core=Core(owner)
        with self.assertRaisesRegex(ValueError,'control_busy'):
            with native_maintenance(owner): pass
        core.close()
        with native_maintenance(owner) as view:
            with self.assertRaises(sqlite3.OperationalError): view.db.execute('CREATE TABLE forbidden(x)')
            with self.assertRaisesRegex(ValueError,'control_busy'): Core(owner)
        core=Core(owner);core.close()

    def test_partial_retirement_blocks_journal(self):
        root=self.root/'host';root.mkdir(mode=0o700)
        journal=Journal(root,'a'*32,'b'*64);journal.lock.close()
        marker=root/'retirement.stage';marker.write_bytes(b'{');marker.chmod(0o600)
        with self.assertRaisesRegex(Refused,'native_root_retired'): Journal(root,'a'*32,'b'*64)
        self.assertEqual(marker.read_bytes(),b'{')

    def test_capacity_actionable_and_canonical_types(self):
        r.capacity('history',62,63)
        with self.assertRaisesRegex(Refused,'native_retention_capacity.*used=63.*retain owner'):
            r.capacity('history',63,63)
        with self.assertRaises(Refused): r.capacity('history',True,63)


class RenewalHistoryTests(unittest.TestCase):
    setUp = NativeHistoryTests.setUp
    prepare = NativeHistoryTests.prepare
    wait = NativeHistoryTests.wait
    open = NativeHistoryTests.open
    close = NativeHistoryTests.close

    def begin(self):
        self.intent = self.root/'native-renewal'/'001.intent'
        return r.begin(self.record,configuration_file=self.kw['configuration_file'],
            templates_file=self.kw['templates_file'],owner_root=self.root,request_id='e'*32)

    def retire(self):
        self.f.s.journal.lock.close()
        return r.retire(self.intent,host_root=self.f.s.journal.root)

    def test_stopped_offline_success_seal_and_closed_commit(self):
        self.prepare(); old=self.record.read_bytes()
        intent=self.begin();self.assertEqual(self.begin(),intent)
        export=self.retire();self.assertEqual(self.retire(),export)
        self.kw['connector']=lambda _: (_ for _ in ()).throw(AssertionError('old TLS unavailable'))
        with patch.object(n.p,'tls_context',side_effect=AssertionError('no old private key required')):
            seal=r.seal(self.intent,export_file=self.f.s.journal.root/'retirement-export')
            self.assertEqual(r.seal(self.intent,export_file=self.f.s.journal.root/'retirement-export'),seal)
        with self.assertRaisesRegex(Refused,'native_renewal_successor_unavailable'):
            r.commit(self.intent,export_file=self.f.s.journal.root/'retirement-export')
        self.assertEqual(self.record.read_bytes(),old)
        with self.assertRaisesRegex(Refused,'native_renewal_pending'): n.native_startup_inputs(self.record,**self.kw)
        with self.assertRaisesRegex(Refused,'native_root_retired'):
            self.f.s.dispatch(dict(op='hello'))

    def test_live_host_lock_refuses_retirement_without_marker(self):
        self.prepare();self.begin()
        with self.assertRaises(BlockingIOError): r.retire(self.intent,host_root=self.f.s.journal.root)
        self.assertFalse((self.f.s.journal.root/'retirement').exists())

    def test_changed_request_cannot_fork_pending(self):
        self.prepare();self.begin();saved=self.intent.read_bytes()
        with self.assertRaises(Refused):
            r.begin(self.record,configuration_file=self.kw['configuration_file'],
                templates_file=self.kw['templates_file'],owner_root=self.root,request_id='f'*32)
        self.assertEqual(self.intent.read_bytes(),saved)

    def test_unacked_record_refuses_without_retirement(self):
        self.prepare();self.begin()
        record=self.f.s.journal.value['records'][0]
        self.f.s.persist(record,acked=False)
        self.f.s.journal.lock.close()
        with self.assertRaises(Refused): r.retire(self.intent,host_root=self.f.s.journal.root)
        self.assertFalse((self.f.s.journal.root/'retirement').exists())

    def test_unknown_host_evidence_retained(self):
        self.prepare();self.begin();root=self.f.s.journal.root
        file=root/'native.tmp';file.write_bytes(b'RETAIN');file.chmod(0o600)
        self.f.s.journal.lock.close()
        with self.assertRaises(Refused): r.retire(self.intent,host_root=root)
        self.assertEqual(file.read_bytes(),b'RETAIN')

    def test_altered_membership_refuses(self):
        self.prepare();intent=self.begin();intent['membership']=[]
        path=self.intent.parent/'forged';path.write_bytes(canonical(intent));path.chmod(0o600)
        self.f.s.journal.lock.close()
        with self.assertRaises(Refused): r.retire(path,host_root=self.f.s.journal.root)
        self.assertFalse((self.f.s.journal.root/'retirement').exists())

    def test_changed_retained_media_refuses_seal(self):
        self.prepare();self.begin();self.retire()
        file=next((self.root/'assets').iterdir());raw=file.read_bytes();file.write_bytes(raw+b'RETAIN')
        with self.assertRaises(Exception): r.seal(self.intent,export_file=self.f.s.journal.root/'retirement-export')
        self.assertFalse((self.intent.parent/'001.seal').exists())
        self.assertEqual(file.read_bytes(),raw+b'RETAIN')

    def test_canonical_compatibility_refuses_profile_and_types(self):
        self.prepare()
        config=n.document(self.kw['configuration_file']);templates=n.document(self.kw['templates_file'])
        for field,value in (('version',True),('version',1.0)):
            changed=copy.deepcopy(config);changed[field]=value
            with self.assertRaises(Refused): r.compatible(config,templates,changed,templates)
        changed=copy.deepcopy(templates);changed['profiles'][0]['limits']['disk_bytes']+=1
        with self.assertRaises(Refused): r.compatible(config,templates,config,changed)
        changed=copy.deepcopy(config);changed['native']['runtime_manifest_sha256']='f'*64
        with self.assertRaises(Refused): r.compatible(config,templates,changed,templates)


class RenewalNegativeHistoryTests(RenewalHistoryTests):
    terminal_mode='cancelled'
    # Select only the offline proof method explicitly; inherited loader is unused.


class RenewalFailedHistoryTests(RenewalHistoryTests):
    terminal_mode='failed'
