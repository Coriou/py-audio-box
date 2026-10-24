"""Isolated SQLite/media tests; also runnable without pytest/conftest via unittest.

python -m unittest discover -s tests/service -p test_voice_store.py -v
No subprocesses, maintenance imports, snapshots, workers or ML.
"""
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

from contracts import validate as oracle
from service.core import Core, encode
from service.voice_store import initialize

FIXTURES = Path(__file__).resolve().parents[2] / 'contracts/fixtures/v1'


class VoiceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pab-h0-voice-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'data'
        self.core = Core(self.root)
        self.addCleanup(lambda: self.core.close())
        self.version = json.loads((FIXTURES/'voice-catalogue.json').read_text())['register']['version']

    @property
    def store(self):
        return self.core.voices

    def register(self, key='save', revision=0, version=None):
        return self.store.register('alice',key,version or self.version,expected_revision=revision)

    def media(self, *, owner='alice', managed=True):
        data=(FIXTURES/'short.wav').read_bytes()
        metadata=dict(id='reference',sha256=hashlib.sha256(data).hexdigest(),frames=1,sample_rate=24000,channels=1,
                      provenance=dict(kind='upload',source_sha256=hashlib.sha256(data).hexdigest(),source_id='upload-1',missing_fields=[]),media_format='wav',media_subtype='PCM_16')
        if owner is not None:
            metadata['principal_id']=owner
        (self.root/'assets'/'reference').write_bytes(data)
        if managed:
            with self.core.transaction():
                self.core.db.execute('INSERT INTO assets VALUES(?,?)',('reference',encode(metadata)))
        v=copy.deepcopy(self.version)
        v.update(origin='clone',reference_asset_ids=['reference'],reference_text=' Exact reference. ',
                 configuration=dict(mode='qwen_clone',reference_asset_id='reference',reference_text=' Exact reference. ',conditioning='icl'))
        v['settings']['kind']='qwen_clone';v['settings']['instruction']=None
        v['settings_sha256']=oracle.canonical_hash(v['settings'])
        return v, metadata

    def reopen(self):
        self.core.close()
        self.core=Core(self.root)

    def test_restart_frozen_versions_aliases_and_exact_identity(self):
        first=self.register()
        self.store.aliases('alice','alias',self.version['voice_id'],['favorite'],expected_revision=1)
        v=copy.deepcopy(self.version);v['id']='version-2';v['settings']['temperature']=0.75
        v['settings_sha256']=oracle.canonical_hash(v['settings'])
        self.register('second',2,v)
        self.reopen()
        self.assertEqual(self.store.resolve('alice','favorite')['revision'],3)
        self.assertEqual(self.store.version('alice',self.version['id']),first['version'])
        self.assertEqual(self.store.version('alice','version-2')['identity'],v['identity'])
        self.assertEqual(len(self.store.versions('alice',v['voice_id'])),2)
        self.assertEqual(self.core.jobs('alice'),[])

    def test_idempotency_precedes_revision_and_survives_restart(self):
        first=self.register();self.reopen()
        self.assertEqual(first,self.register())
        changed=copy.deepcopy(self.version);changed['settings']['temperature']=0.81;changed['settings_sha256']=oracle.canonical_hash(changed['settings'])
        with self.assertRaisesRegex(ValueError,'idempotency_conflict'):
            self.register(version=changed)
        with self.assertRaisesRegex(ValueError,'version_conflict'):
            self.register('new-key',1)
        self.assertEqual(self.store.get('alice',self.version['voice_id'])['revision'],1)

    def test_serialized_revision_race_and_alias_collision_rollback(self):
        self.register();voice=self.version['voice_id']
        self.store.aliases('alice','winner',voice,['shared'],expected_revision=1)
        with self.assertRaisesRegex(ValueError,'revision_conflict'):
            self.store.aliases('alice','loser',voice,['lost'],expected_revision=1)
        other=copy.deepcopy(self.version);other.update(id='other-version',voice_id='other')
        self.register('other',0,other)
        with self.assertRaisesRegex(ValueError,'alias_conflict'):
            self.store.aliases('alice','collision','other',['free','shared'],expected_revision=1)
        self.assertEqual(self.store.get('alice','other')['revision'],1)
        with self.assertRaisesRegex(ValueError,'not_found'):self.store.resolve('alice','free')
        with self.assertRaisesRegex(ValueError,'alias_conflict'):
            self.store.aliases('alice','id-collision','other',[voice],expected_revision=1)

    def test_archive_preserves_old_references_and_reserves_names(self):
        saved=self.register();voice=self.version['voice_id']
        self.store.aliases('alice','alias',voice,['favorite'],expected_revision=1)
        archived=self.store.archive('alice','archive',voice,expected_revision=2)
        self.assertEqual(archived,self.store.archive('alice','archive',voice,expected_revision=2))
        self.reopen()
        self.assertEqual(self.store.version('alice',self.version['id']),saved['version'])
        self.assertEqual(self.store.list('alice'),[])
        self.assertTrue(self.store.list('alice',include_archived=True)[0]['voice']['archived'])
        with self.assertRaisesRegex(ValueError,'voice_archived'):self.store.resolve('alice','favorite')
        v=copy.deepcopy(self.version);v['id']='later'
        with self.assertRaisesRegex(ValueError,'voice_archived'):self.register('later',3,v)
        v.update(voice_id='favorite')
        with self.assertRaisesRegex(ValueError,'alias_conflict'):self.register('reuse',0,v)

    def test_managed_asset_verified_and_hash_frozen(self):
        v,asset=self.media();saved=self.register(version=v)['version']
        self.assertEqual(saved['reference_asset_sha256'],{'reference':asset['sha256']})
        v['reference_text']='caller changed'
        self.reopen()
        self.assertEqual(self.store.version('alice',saved['id'])['reference_text'],' Exact reference. ')
        (self.root/'assets'/'reference').write_bytes(b'tampered')
        v=copy.deepcopy(saved);v['id']='another'
        with self.assertRaisesRegex(ValueError,'missing_media'):self.register('tamper',1,v)
        self.assertEqual(self.store.get('alice',saved['voice_id'])['revision'],1)

    def test_absent_unmanaged_foreign_and_ownerless_rejected(self):
        v,_=self.media(managed=False)
        with self.assertRaisesRegex(ValueError,'not_found'):self.register(version=v)
        _,metadata=self.media(owner='bob')
        with self.assertRaisesRegex(ValueError,'forbidden'):self.register(version=v)
        metadata.pop('principal_id')
        with self.core.transaction():self.core.db.execute('UPDATE assets SET metadata=?',(encode(metadata),))
        with self.assertRaisesRegex(ValueError,'forbidden'):self.register(version=v)
        self.assertEqual(self.store.list('alice'),[])

    def test_immutable_sql_rows_and_return_values(self):
        saved=self.register()['version'];saved['identity']['model']='changed'
        with self.assertRaisesRegex(sqlite3.IntegrityError,'immutable_voice_version'):
            with self.core.transaction():self.core.db.execute("UPDATE catalogue_versions SET record='{}'")
        with self.assertRaisesRegex(sqlite3.IntegrityError,'immutable_voice_version'):
            with self.core.transaction():self.core.db.execute('DELETE FROM catalogue_versions')
        self.assertEqual(self.store.version('alice',self.version['id'])['identity'],self.version['identity'])
        with self.assertRaisesRegex(ValueError,'not_found'):self.store.version('bob',self.version['id'])

    def test_migration_preserves_old_rows_repeat_open_and_rollback(self):
        # Simulate the pre-H0 additive boundary, retaining existing core records.
        with self.core.transaction():
            for table in ('catalogue_requests','catalogue_versions','catalogue_names','catalogue_voices','voice_catalogue_format'):
                self.core.db.execute('DROP TABLE '+table)
            self.core.db.execute("INSERT INTO revisions VALUES('prior',1,'{}')")
        def deny(action, arg1, arg2, dbname, source):
            return sqlite3.SQLITE_DENY if action==sqlite3.SQLITE_CREATE_TABLE and arg1=='catalogue_names' else sqlite3.SQLITE_OK
        self.core.db.set_authorizer(deny)
        try:
            with self.assertRaises(sqlite3.DatabaseError):initialize(self.core)
        finally:self.core.db.set_authorizer(None)
        self.assertIsNone(self.core.db.execute("SELECT name FROM sqlite_master WHERE name='catalogue_voices'").fetchone())
        initialize(self.core);self.register();self.reopen();self.reopen()
        self.assertEqual(self.core.db.execute("SELECT snapshot FROM revisions WHERE project='prior'").fetchone(),('{}',))
        self.assertEqual(self.store.get('alice',self.version['voice_id'])['revision'],1)
        self.assertEqual(self.core.db.execute('PRAGMA user_version').fetchone(),(1,))

    def test_bounded_pages_no_hidden_ml_or_network_import(self):
        class Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('torch','transformers','qwen_tts','lib','requests','httpx'):
                    raise AssertionError('unexpected heavy import '+fullname)
        blocker=Refuse();sys.meta_path.insert(0,blocker)
        try:
            for i in range(3):
                v=copy.deepcopy(self.version);v.update(id=f'version-{i}',voice_id=f'voice-{i}')
                self.register(f'save-{i}',0,v)
            first=self.store.list('alice',limit=2)
            second=self.store.list('alice',after=first[-1]['cursor'],limit=2)
            self.assertEqual(len(first),2);self.assertEqual(len(second),1)
            self.assertEqual(self.store.list('bob'),[])
            for limit in (0,21,True):
                with self.assertRaisesRegex(ValueError,'invalid_request'):self.store.list('alice',limit=limit)
        finally:sys.meta_path.remove(blocker)

    def test_contract_fixtures(self):
        fixture=json.loads((FIXTURES/'voice-catalogue.json').read_text())
        oracle.validate('voice_register_request',fixture['register'])
        oracle.validate('voice_aliases_request',fixture['aliases'])
        oracle.validate('voice_register_result',self.register())
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(oracle.SCHEMA)
        operations=json.loads((FIXTURES.parents[1]/'v1/operations.json').read_text())['operations']
        requests=json.loads((FIXTURES/'operation-requests.json').read_text())
        self.assertEqual(set(requests),{o['id'] for o in operations})
        for operation in operations:
            oracle.validate(operation['request'],requests[operation['id']])
        errors=json.loads((FIXTURES.parents[1]/'v1/errors.json').read_text())
        self.assertEqual(set(errors['http_status']),set(oracle.SCHEMA['$defs']['error']['properties']['code']['enum']))

    def test_symlink_and_missing_managed_file_rejected(self):
        v,_=self.media()
        path=self.root/'assets'/'reference';path.unlink()
        with self.assertRaisesRegex(ValueError,'missing_media'):self.register(version=v)
        path.symlink_to(FIXTURES/'short.wav')
        with self.assertRaisesRegex(ValueError,'missing_media'):self.register(version=v)
        self.assertEqual(self.store.list('alice'),[])

    def test_pre_h0_published_asset_inherits_job_owner(self):
        v,_=self.media(owner=None)
        context=json.loads((FIXTURES/'transcript-speech_generate.json').read_text())['context']
        with self.core.transaction():
            self.core.db.execute('INSERT INTO jobs(id,principal,operation,key,fingerprint,job,context) VALUES(?,?,?,?,?,?,?)',
                                 ('prior-job','alice','speech_generate','old','hash','{}',encode(context)))
            self.core.db.execute('INSERT INTO artifacts VALUES(?,?)',('prior-artifact',encode(dict(job_id='prior-job',asset_ids=['reference']))))
        self.assertEqual(self.register(version=v)['version']['reference_asset_ids'],['reference'])

    def test_partial_migration_refuses_without_repair_and_old_root_stays_fenced(self):
        with self.core.transaction():self.core.db.execute('DROP TABLE catalogue_requests')
        with self.assertRaisesRegex(ValueError,'partial_voice_store_schema'):initialize(self.core)
        self.assertIsNone(self.core.db.execute("SELECT name FROM sqlite_master WHERE name='catalogue_requests'").fetchone())
        self.core.p1=False
        initialize(self.core)
        with self.assertRaisesRegex(ValueError,'pre_p1_inspect_only'):self.register()
        self.core.p1=True
