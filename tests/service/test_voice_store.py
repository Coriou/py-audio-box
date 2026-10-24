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

    def test_same_column_schema_refusal_preserves_all_rows(self):
        self.register()
        order = ('voice_catalogue_format', 'catalogue_voices', 'catalogue_names',
                 'catalogue_versions', 'catalogue_requests')
        ddl = {name: self.core.db.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
               for name in order}
        ddl = {name: ' '.join(sql.split()) for name, sql in ddl.items()}
        triggers = [r[0] for r in self.core.db.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name='catalogue_versions'")]
        rows = {name: self.core.db.execute('SELECT * FROM '+name).fetchall() for name in order}
        def snapshot():
            return (self.core.db.execute('SELECT * FROM sqlite_master ORDER BY name').fetchall(),
                    {name: self.core.db.execute('SELECT * FROM '+name).fetchall() for name in order},
                    self.core.db.execute('SELECT * FROM revisions').fetchall())
        with self.core.transaction():
            self.core.db.execute("INSERT INTO revisions VALUES('prior',1,'{}')")
        variants = []
        for name in order:
            variants.append((name+' wrong type', name, ddl[name].replace('INTEGER', 'BLOB', 1)
                             if 'INTEGER' in ddl[name] else ddl[name].replace('TEXT', 'BLOB', 1)))
            variants.append((name+' nullable', name, ddl[name].replace(' NOT NULL', '', 1)))
        for name in ('catalogue_voices', 'catalogue_versions'):
            variants.append((name+' missing PK', name, ddl[name].replace(' PRIMARY KEY', '', 1)))
            variants.append((name+' missing unique', name, ddl[name].replace(', UNIQUE(principal,id)', '')))
        for name in ('catalogue_names', 'catalogue_versions'):
            variants.append((name+' missing FK', name, ddl[name][:ddl[name].index(', FOREIGN KEY')] + ')'))
        for name, constraint in (('catalogue_names', 'PRIMARY KEY(principal,name)'),
                                 ('catalogue_requests', 'PRIMARY KEY(principal,operation,key)')):
            variants.append((name+' missing PK', name, ddl[name].replace(constraint+', ', '').replace(', '+constraint, '')))
        # No CHECK is declared by H0. Added checks are incompatible too.
        variants.append(('added check', 'catalogue_voices', ddl['catalogue_voices'].replace('archived INTEGER NOT NULL', 'archived INTEGER NOT NULL CHECK(archived=0)')))
        for label, changed, replacement in variants:
            with self.subTest(layout=label):
                with self.core.transaction():
                    for name in reversed(order):
                        self.core.db.execute('DROP TABLE '+name)
                    for name in order:
                        self.core.db.execute(replacement if name == changed else ddl[name])
                    # Missing parent uniqueness prevents FK inserts; retain the
                    # parent/core rows and leave child tables empty in that case.
                    for name in order:
                        if 'missing unique' in label and name in ('catalogue_names', 'catalogue_versions'):
                            continue
                        for row in rows[name]:
                            self.core.db.execute('INSERT INTO '+name+' VALUES('+','.join('?' for _ in row)+')', row)
                    for sql in triggers:
                        self.core.db.execute(sql)
                before = snapshot()
                with self.assertRaisesRegex(ValueError, 'unsupported_voice_store_schema'):
                    initialize(self.core)
                self.assertFalse(self.core.db.in_transaction)
                self.assertEqual(snapshot(), before)
        # Restore only our test fixture, then prove a valid schema reopens.
        with self.core.transaction():
            for name in reversed(order): self.core.db.execute('DROP TABLE '+name)
            for name in order:
                self.core.db.execute(ddl[name])
                for row in rows[name]:
                    self.core.db.execute('INSERT INTO '+name+' VALUES('+','.join('?' for _ in row)+')', row)
            for sql in triggers: self.core.db.execute(sql)
        self.reopen()
        self.assertEqual(self.store.get('alice', self.version['voice_id'])['revision'], 1)

    def test_r160_null_seq_counterexample_refused(self):
        with self.core.transaction():
            self.core.db.execute('DROP TABLE catalogue_versions')
            self.core.db.execute('DROP TABLE catalogue_names')
            self.core.db.execute('DROP TABLE catalogue_voices')
            self.core.db.execute('CREATE TABLE catalogue_voices(seq INTEGER, principal TEXT, id TEXT, revision INTEGER, archived INTEGER, UNIQUE(principal,id))')
            self.core.db.execute('CREATE TABLE catalogue_names(principal TEXT NOT NULL, name TEXT NOT NULL, voice_id TEXT NOT NULL, PRIMARY KEY(principal,name), FOREIGN KEY(principal,voice_id) REFERENCES catalogue_voices(principal,id))')
            self.core.db.execute('CREATE TABLE catalogue_versions(seq INTEGER PRIMARY KEY, principal TEXT NOT NULL, id TEXT NOT NULL, voice_id TEXT NOT NULL, record TEXT NOT NULL, UNIQUE(principal,id), FOREIGN KEY(principal,voice_id) REFERENCES catalogue_voices(principal,id))')
            for action in ('UPDATE', 'DELETE'):
                self.core.db.execute(f"CREATE TRIGGER catalogue_versions_no_{action.lower()} BEFORE {action} ON catalogue_versions BEGIN SELECT RAISE(ABORT,'immutable_voice_version'); END")
        # The already-held trusted store demonstrates why admission must refuse.
        self.register()
        self.assertEqual(self.core.db.execute('SELECT seq FROM catalogue_voices').fetchall(), [(None,)])
        self.assertEqual(self.store.list('alice'), [])
        before = list(self.core.db.iterdump())
        with self.assertRaisesRegex(ValueError, 'unsupported_voice_store_schema'):
            initialize(self.core)
        self.assertEqual(list(self.core.db.iterdump()), before)
        self.core.close()
        with self.assertRaisesRegex(ValueError, 'unsupported_voice_store_schema'):
            Core(self.root)

    def test_trigger_and_index_identity_refusal(self):
        self.register()
        for action in ('update', 'delete'):
            with self.subTest(action=action):
                name = 'catalogue_versions_no_'+action
                sql = self.core.db.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()[0]
                with self.core.transaction():
                    self.core.db.execute('DROP TRIGGER '+name)
                    self.core.db.execute(f'CREATE TRIGGER {name} BEFORE {action} ON catalogue_versions BEGIN SELECT 1; END')
                before = list(self.core.db.iterdump())
                with self.assertRaisesRegex(ValueError, 'unsupported_voice_store_schema'): initialize(self.core)
                self.assertEqual(list(self.core.db.iterdump()), before)
                with self.core.transaction():
                    self.core.db.execute('DROP TRIGGER '+name)
                    self.core.db.execute(sql)
        with self.core.transaction(): self.core.db.execute('CREATE UNIQUE INDEX extra_voice_constraint ON catalogue_voices(id)')
        with self.assertRaisesRegex(ValueError, 'unsupported_voice_store_schema'): initialize(self.core)
        with self.core.transaction(): self.core.db.execute('DROP INDEX extra_voice_constraint')
        self.reopen()
        self.assertEqual(self.store.version('alice', self.version['id'])['id'], self.version['id'])

    def test_schema_tokens_accept_spacing_and_case_but_preserve_literals(self):
        with self.core.transaction():
            self.core.db.execute('DROP TRIGGER catalogue_versions_no_update')
            self.core.db.execute("create trigger catalogue_versions_no_update before update on catalogue_versions begin select raise(abort, 'immutable_voice_version'); end")
        initialize(self.core)
        self.register()
        with self.core.transaction():
            self.core.db.execute('DROP TRIGGER catalogue_versions_no_update')
            self.core.db.execute("CREATE TRIGGER catalogue_versions_no_update BEFORE UPDATE ON catalogue_versions BEGIN SELECT RAISE(ABORT,'immutable voice version'); END")
        with self.assertRaisesRegex(ValueError, 'unsupported_voice_store_schema'): initialize(self.core)
        with self.core.transaction(): self.core.db.execute('DROP TRIGGER catalogue_versions_no_update')
        with self.assertRaisesRegex(ValueError, 'unsupported_voice_store_schema'): initialize(self.core)
        self.assertIsNone(self.core.db.execute("SELECT name FROM sqlite_master WHERE name='catalogue_versions_no_update'").fetchone())

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
            oracle.validate(operation['request'],oracle.fixture_request(operation['request'],requests[operation['id']]))
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

    def test_compact_pages_and_versions_never_full_get(self):
        from unittest.mock import patch
        self.register()
        for i in range(3):
            v=copy.deepcopy(self.version); v['id']='page-'+str(i)
            self.register('page-'+str(i), i+1, v)
        with patch.object(self.store, 'get', side_effect=AssertionError('full history')):
            self.assertEqual(self.store.summary('alice', self.version['voice_id'])['version_count'], 4)
            self.assertEqual(len(self.store.summaries('alice')), 1)
            first=self.store.version_summaries('alice', self.version['voice_id'], limit=2)
            rest=self.store.version_summaries('alice', self.version['voice_id'], after=first[-1]['cursor'], limit=2)
            self.assertEqual(len(first)+len(rest), 4)
            self.assertEqual(len(self.store.versions('alice', self.version['voice_id'], limit=2)), 2)
        with self.assertRaisesRegex(ValueError, 'not_found'):
            self.store.version_summaries('bob', self.version['voice_id'])

    def test_new_registration_text_budget_preserves_legacy_reads_and_replay(self):
        v,_=self.media()
        v['reference_text']='x'*16385
        v['configuration']['reference_text']=v['reference_text']
        with self.assertRaisesRegex(ValueError, 'resource_limit'): self.register(version=v)
        v['reference_text']='x'*16384
        v['configuration']['reference_text']=v['reference_text']
        saved=self.register(version=v)
        self.assertEqual(saved,self.register(version=v))
        self.assertEqual(self.store.version('alice',v['id'])['reference_text'],v['reference_text'])

    def test_ingested_asset_registers_and_freezes_verified_hash(self):
        from service.ingest import ingest_local
        import io
        import wave
        output=io.BytesIO()
        with wave.open(output,'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
            wav.writeframes(b'\0\0'*600000)
        asset=ingest_local(self.core,'alice',output.getvalue())
        v=copy.deepcopy(self.version)
        v.update(origin='clone', reference_asset_ids=[asset['id']], reference_text=' Exact reference. ',
                 configuration=dict(mode='qwen_clone',reference_asset_id=asset['id'],reference_text=' Exact reference. ',conditioning='icl'))
        v['settings']['kind']='qwen_clone'; v['settings']['instruction']=None
        v['settings_sha256']=oracle.canonical_hash(v['settings'])
        saved=self.register(version=v)['version']
        self.reopen()
        self.assertEqual(self.store.version('alice',v['id']),saved)
        self.assertEqual(saved['reference_asset_sha256'],{asset['id']:asset['sha256']})
