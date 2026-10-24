"""Actual isolated Core/SQLite/PCM bytes; no models, pytest or workers."""
import errno
import hashlib
import io
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from service.core import Core
from service.ingest import ingest_local, MAX_BYTES


def audio(frames=24000, rate=24000, channels=1, width=2):
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setnchannels(channels); wav.setsampwidth(width); wav.setframerate(rate)
        wav.writeframes(b'\x00' * frames * channels * width)
    return out.getvalue()


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pab-h1-ingest-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'data'
        self.core = Core(self.root)
        self.addCleanup(lambda: self.core.close())

    def empty(self):
        self.assertEqual(list((self.root/'staging').iterdir()), [])
        self.assertEqual(list((self.root/'assets').iterdir()), [])
        self.assertEqual(self.core.db.execute('SELECT * FROM assets').fetchall(), [])

    def test_valid_restart_original_principal_provenance_and_private_files(self):
        data = audio()
        asset = ingest_local(self.core, 'alice', io.BytesIO(data))
        self.assertEqual(asset['sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(asset['provenance'], dict(kind='upload', source_id=asset['id'], source_sha256=asset['sha256'], missing_fields=[]))
        self.assertEqual(asset['principal_id'], 'alice')
        path = self.root/'assets'/asset['id']
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.stat().st_nlink, 1)
        self.core.close(); self.core = Core(self.root)
        self.assertEqual(self.core.asset(asset['id']), asset)
        self.assertEqual(list((self.root/'staging').iterdir()), [])

    def test_duplicate_content_is_new_owned_asset_never_foreign_adoption(self):
        first = ingest_local(self.core, 'alice', audio())
        other = ingest_local(self.core, 'bob', audio())
        again = ingest_local(self.core, 'alice', audio())
        self.assertEqual(first['sha256'], other['sha256'])
        self.assertEqual(len({first['id'], other['id'], again['id']}), 3)
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            self.core.voices._asset('alice', other['id'])
        self.assertEqual(self.core.voices._asset('alice', first['id']), first['sha256'])

    def test_large_reference_budget_and_tamper(self):
        data = audio(frames=48000*60, rate=48000, channels=2, width=4)
        asset = ingest_local(self.core, 'alice', data)
        self.assertGreater(len(data), 1024*1024)
        self.assertLess(len(data), MAX_BYTES)
        self.assertEqual(self.core.asset(asset['id']), asset)
        with (self.root/'assets'/asset['id']).open('r+b') as stream:
            stream.seek(-1, 2); stream.write(b'\x01')
        with self.assertRaisesRegex(ValueError, 'missing_media'): self.core.asset(asset['id'])

    def test_invalid_audio_formats_truncation_limits_and_paths(self):
        invalid = [b'', b'not WAV', b'RIFF\xff\xff\xff\xffWAVE', audio()[:-3],
                   audio(frames=0), audio(rate=7999), audio(rate=96000),
                   audio(channels=3), audio(width=1), audio(frames=24000*60+1),
                   b'x'*(MAX_BYTES+1)]
        floating = bytearray(audio()); struct.pack_into('<H', floating, 20, 3)
        floating[44:48] = struct.pack('<f', float('nan'))
        invalid.append(bytes(floating))
        for data in invalid:
            with self.subTest(length=len(data)):
                with self.assertRaises(ValueError): ingest_local(self.core, 'alice', data)
                self.empty()
        for value in ('/tmp/source.wav', Path('/tmp/source.wav'), {'principal_id': 'alice'}):
            with self.assertRaisesRegex(ValueError, 'invalid_request'): ingest_local(self.core, 'alice', value)
        self.empty()

    def test_bounded_stream_and_mid_read_failure(self):
        class Oversize:
            def read(self, size): return b'x'*(size+1)
        with self.assertRaisesRegex(ValueError, 'resource_limit'):
            ingest_local(self.core, 'alice', Oversize())
        class Broken:
            calls = 0
            def read(self, size):
                self.calls += 1
                if self.calls == 1: return b'RIFF'
                raise OSError(errno.EIO, 'injected read')
        with self.assertRaises(OSError): ingest_local(self.core, 'alice', Broken())
        self.empty()

    def test_low_disk_and_publish_failure_remove_only_own_files(self):
        sentinel = self.root/'staging'/'keep'; sentinel.write_bytes(b'keep')
        class Full:
            f_bavail = 0
            f_frsize = 4096
        with patch('service.ingest.os.fstatvfs', return_value=Full()):
            with self.assertRaisesRegex(ValueError, 'resource_limit'): ingest_local(self.core, 'alice', audio())
        with patch('service.ingest.os.link', side_effect=OSError(errno.EIO, 'publish')):
            with self.assertRaises(OSError): ingest_local(self.core, 'alice', audio())
        self.assertEqual(sentinel.read_bytes(), b'keep')
        sentinel.unlink(); self.empty()

    def test_fsync_and_database_insert_failure_cleanup(self):
        with patch('service.ingest.os.fsync', side_effect=OSError(errno.EIO, 'sync')):
            with self.assertRaises(OSError): ingest_local(self.core, 'alice', audio())
        self.empty()
        def deny(action, a, b, db, origin):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and a == 'assets' else sqlite3.SQLITE_OK
        self.core.db.set_authorizer(deny)
        try:
            with self.assertRaises(sqlite3.DatabaseError): ingest_local(self.core, 'alice', audio())
        finally: self.core.db.set_authorizer(None)
        self.empty()

    def test_nofollow_namespaces_and_managed_file(self):
        for name in ('assets', 'staging'):
            path = self.root/name
            saved = self.root/(name+'-saved'); path.rename(saved); path.symlink_to(saved, target_is_directory=True)
            try:
                with self.assertRaises(OSError): ingest_local(self.core, 'alice', audio())
                self.assertEqual(list(saved.iterdir()), [])
            finally: path.unlink(); saved.rename(path)
        asset = ingest_local(self.core, 'alice', audio())
        path = self.root/'assets'/asset['id']; path.unlink(); path.symlink_to(self.root/'core.sqlite')
        with self.assertRaises(OSError): self.core.asset(asset['id'])

    def test_old_assets_keep_small_cap_and_missing_file_guarantees(self):
        asset = ingest_local(self.core, 'alice', audio(frames=600000))
        import json
        legacy = dict(asset); legacy.pop('principal_id')
        with self.core.transaction(): self.core.db.execute('UPDATE assets SET metadata=? WHERE id=?', (json.dumps(legacy), asset['id']))
        with self.assertRaisesRegex(ValueError, 'foreign_media'): self.core.asset(asset['id'])

    def test_pre_p1_and_nested_transaction_no_files(self):
        self.core.p1 = False
        try:
            with self.assertRaisesRegex(ValueError, 'pre_p1_inspect_only'): ingest_local(self.core, 'alice', audio())
        finally: self.core.p1 = True
        with self.core.transaction():
            with self.assertRaisesRegex(ValueError, 'invalid_request'): ingest_local(self.core, 'alice', audio())
        self.empty()

    def test_post_publish_sync_failure_cleans_own_final(self):
        original = os.fsync
        calls = []
        def sync(fd):
            calls.append(fd)
            if len(calls) == 3: raise OSError(errno.EIO, 'asset directory sync')
            return original(fd)
        with patch('service.ingest.os.fsync', side_effect=sync):
            with self.assertRaises(OSError): ingest_local(self.core, 'alice', audio())
        self.empty()

    def test_commit_refusal_retains_complete_unreferenced_orphan(self):
        def deny(action, a, b, db, origin):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_TRANSACTION and a == 'COMMIT' else sqlite3.SQLITE_OK
        self.core.db.set_authorizer(deny)
        try:
            with self.assertRaises(sqlite3.DatabaseError): ingest_local(self.core, 'alice', audio())
        finally: self.core.db.set_authorizer(None)
        files = list((self.root/'assets').iterdir())
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_bytes(), audio())
        self.assertEqual(self.core.db.execute('SELECT * FROM assets').fetchall(), [])
        self.assertEqual(list((self.root/'staging').iterdir()), [])
        self.core.close(); self.core = Core(self.root)
        with self.assertRaisesRegex(ValueError, 'not_found'): self.core.asset(files[0].name)
        fresh = ingest_local(self.core, 'alice', audio())
        self.assertNotEqual(fresh['id'], files[0].name)

    def test_pcm24_stereo_and_pcm32_and_short_reads(self):
        for width in (2, 3, 4):
            class Short(io.BytesIO):
                def read(self, size=-1): return super().read(min(size, 17))
            data = audio(width=width, channels=2)
            metadata = ingest_local(self.core, 'alice', Short(data))
            self.assertEqual(metadata['media_subtype'], {2:'PCM_16',3:'PCM_24',4:'PCM_32'}[width])
            self.assertEqual(metadata['channels'], 2)
            self.assertEqual(self.core.asset(metadata['id']), metadata)
