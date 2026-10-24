"""Trusted H0 metadata catalogue. No acquisition, inference or public dispatcher.

Uses the Core owning thread/transaction and verified managed assets. Old roots
remain inspect-only. Versions and their asset digests are append-only; aliases
are principal-local names. Archive reserves names and preserves all references.
"""
import errno
import json
import re
from contracts import validate as oracle

_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError('invalid_request')
    return value


# One supported DDL declaration drives installation and schema validation.
_TABLE_DDL = (
    '''CREATE TABLE voice_catalogue_format(revision INTEGER NOT NULL)''',
    '''CREATE TABLE catalogue_voices(
            seq INTEGER PRIMARY KEY, principal TEXT NOT NULL, id TEXT NOT NULL,
            revision INTEGER NOT NULL, archived INTEGER NOT NULL,
            UNIQUE(principal,id))''',
    '''CREATE TABLE catalogue_names(
            principal TEXT NOT NULL, name TEXT NOT NULL, voice_id TEXT NOT NULL,
            PRIMARY KEY(principal,name), FOREIGN KEY(principal,voice_id)
            REFERENCES catalogue_voices(principal,id))''',
    '''CREATE TABLE catalogue_versions(
            seq INTEGER PRIMARY KEY, principal TEXT NOT NULL, id TEXT NOT NULL,
            voice_id TEXT NOT NULL, record TEXT NOT NULL, UNIQUE(principal,id),
            FOREIGN KEY(principal,voice_id) REFERENCES catalogue_voices(principal,id))''',
    '''CREATE TABLE catalogue_requests(
            principal TEXT NOT NULL, operation TEXT NOT NULL, key TEXT NOT NULL,
            fingerprint TEXT NOT NULL, result TEXT NOT NULL,
            PRIMARY KEY(principal,operation,key))''',
)
_TRIGGER_DDL = tuple(
    f"CREATE TRIGGER catalogue_versions_no_{action.lower()} "
    f"BEFORE {action} ON catalogue_versions BEGIN "
    "SELECT RAISE(ABORT,'immutable_voice_version'); END"
    for action in ('UPDATE', 'DELETE')
)
_TABLES = frozenset(sql.split('(')[0].split()[-1] for sql in _TABLE_DDL)
_SQL_TOKEN = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^]]*\]|[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\s]")


def _sql_identity(sql):
    # Token identity preserves boundaries and quoted contents, unlike removing
    # whitespace. Only unquoted keyword/identifier case is insignificant.
    return tuple(token if token[0] in "'\"`[" else token.casefold()
                 for token in _SQL_TOKEN.findall(sql or ''))


def _schema_identity(db):
    # Includes implicit constraint indexes and extra indexes/triggers on every
    # owned table, while leaving unrelated Core schema objects outside scope.
    return sorted((kind, name, table, _sql_identity(sql))
                  for kind, name, table, sql in db.execute(
                      'SELECT type,name,tbl_name,sql FROM sqlite_master')
                  if table in _TABLES or name in _TABLES)


def initialize(core):
    """Atomically install or refuse the exact supported catalogue schema."""
    if not core.p1:
        return
    import sqlite3
    with core.transaction():
        tables = {r[0] for r in core.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        existing = tables & _TABLES
        if existing and existing != _TABLES:
            raise ValueError('partial_voice_store_schema')
        # SQLite derives the expected implicit indexes from our supported DDL.
        # This memory-only reference never reads/modifies application rows.
        reference = sqlite3.connect(':memory:')
        try:
            for sql in (*_TABLE_DDL, *_TRIGGER_DDL):
                reference.execute(sql)
            expected = _schema_identity(reference)
        finally:
            reference.close()
        if not existing:
            for sql in (*_TABLE_DDL, *_TRIGGER_DDL):
                core.db.execute(sql)
            core.db.execute('INSERT INTO voice_catalogue_format VALUES(1)')
        if _schema_identity(core.db) != expected:
            raise ValueError('unsupported_voice_store_schema')
        if core.db.execute('SELECT revision FROM voice_catalogue_format').fetchall() != [(1,)]:
            raise ValueError('unsupported_voice_store_version')


class VoiceStore:
    def __init__(self, core):
        self.core = core

    @property
    def db(self):
        return self.core.db

    def _mutate(self, principal, operation, key, payload, apply):
        _id(principal); _id(key)
        fingerprint = oracle.fingerprint(principal, operation, payload)
        with self.core.transaction():
            row = self.db.execute('SELECT fingerprint,result FROM catalogue_requests WHERE principal=? AND operation=? AND key=?', (principal,operation,key)).fetchone()
            if row:
                oracle.require(row[0] == fingerprint, 'idempotency_conflict')
                return json.loads(row[1])
            result = apply()
            self.db.execute('INSERT INTO catalogue_requests VALUES(?,?,?,?,?)',
                            (principal,operation,key,fingerprint,_encode(result)))
            return result

    def get(self, principal, voice_id):
        _id(principal); _id(voice_id)
        row = self.db.execute('SELECT id,revision,archived FROM catalogue_voices WHERE principal=? AND id=?', (principal,voice_id)).fetchone()
        if row is None:
            raise ValueError('not_found')
        return dict(id=row[0], revision=row[1], archived=bool(row[2]),
                    aliases=[r[0] for r in self.db.execute('SELECT name FROM catalogue_names WHERE principal=? AND voice_id=? AND name!=? ORDER BY name', (principal,voice_id,voice_id))],
                    version_ids=[r[0] for r in self.db.execute('SELECT id FROM catalogue_versions WHERE principal=? AND voice_id=? ORDER BY seq', (principal,voice_id))])

    def resolve(self, principal, name):
        _id(principal); _id(name)
        row = self.db.execute('SELECT voice_id FROM catalogue_names WHERE principal=? AND name=?', (principal,name)).fetchone()
        if row is None:
            raise ValueError('not_found')
        voice = self.get(principal, row[0])
        oracle.require(not voice['archived'], 'voice_archived')
        return voice

    def version(self, principal, version_id):
        _id(principal); _id(version_id)
        row = self.db.execute('SELECT record FROM catalogue_versions WHERE principal=? AND id=?', (principal,version_id)).fetchone()
        if row is None:
            raise ValueError('not_found')
        return json.loads(row[0])

    def list(self, principal, *, after=0, limit=20, include_archived=False):
        _id(principal)
        oracle.require(type(after) is int and after >= 0 and type(limit) is int and 1 <= limit <= 20 and type(include_archived) is bool, 'invalid_request')
        return [dict(cursor=r[0], voice=self.get(principal,r[1])) for r in self.db.execute(
            'SELECT seq,id FROM catalogue_voices WHERE principal=? AND seq>? AND (archived=0 OR ?=1) ORDER BY seq LIMIT ?', (principal,after,int(include_archived),limit))]

    def summary(self, principal, voice_id):
        """Compact identity with counts; never loads authored text/history."""
        _id(principal); _id(voice_id)
        row = self.db.execute('SELECT id,revision,archived FROM catalogue_voices WHERE principal=? AND id=?', (principal,voice_id)).fetchone()
        if row is None: raise ValueError('not_found')
        count = self.db.execute('SELECT count(*) FROM catalogue_versions WHERE principal=? AND voice_id=?', (principal,voice_id)).fetchone()[0]
        return dict(id=row[0], revision=row[1], archived=bool(row[2]), version_count=count)

    def summaries(self, principal, *, after=0, limit=20, include_archived=False):
        _id(principal)
        oracle.require(type(after) is int and after >= 0 and type(limit) is int and 1 <= limit <= 20 and type(include_archived) is bool, 'invalid_request')
        return [dict(cursor=r[0], voice=self.summary(principal,r[1])) for r in self.db.execute(
            'SELECT seq,id FROM catalogue_voices WHERE principal=? AND seq>? AND (archived=0 OR ?=1) ORDER BY seq LIMIT ?', (principal,after,int(include_archived),limit))]

    def version_summaries(self, principal, voice_id, *, after=0, limit=20):
        _id(principal); _id(voice_id)
        oracle.require(self.db.execute('SELECT 1 FROM catalogue_voices WHERE principal=? AND id=?', (principal,voice_id)).fetchone() is not None, 'not_found')
        oracle.require(type(after) is int and after >= 0 and type(limit) is int and 1 <= limit <= 20, 'invalid_request')
        return [dict(cursor=r[0], id=r[1], voice_id=voice_id) for r in self.db.execute(
            'SELECT seq,id FROM catalogue_versions WHERE principal=? AND voice_id=? AND seq>? ORDER BY seq LIMIT ?', (principal,voice_id,after,limit))]

    def versions(self, principal, voice_id, *, after=0, limit=20):
        _id(principal); _id(voice_id)
        oracle.require(self.db.execute('SELECT 1 FROM catalogue_voices WHERE principal=? AND id=?', (principal,voice_id)).fetchone() is not None, 'not_found')
        oracle.require(type(after) is int and after>=0 and type(limit) is int and 1<=limit<=20, 'invalid_request')
        return [dict(cursor=r[0], version=json.loads(r[1])) for r in self.db.execute(
            'SELECT seq,record FROM catalogue_versions WHERE principal=? AND voice_id=? AND seq>? ORDER BY seq LIMIT ?', (principal,voice_id,after,limit))]

    def _asset(self, principal, asset_id):
        try:
            metadata = self.core.asset(asset_id)  # DB membership, schema, no-follow, inode, bytes/hash.
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ELOOP):
                raise ValueError('missing_media') from None
            raise
        owner = metadata.get('principal_id')
        if owner is None:
            # Existing published outputs inherit the producing job's principal.
            owners = set()
            # Compatibility fallback is explicitly capped. H1 ingest writes
            # principal_id in trusted managed metadata; no caller owner claims.
            rows = self.db.execute('SELECT metadata FROM artifacts ORDER BY id LIMIT 1025').fetchall()
            oracle.require(len(rows)<=1024, 'resource_limit')
            for (raw,) in rows:
                artifact = json.loads(raw)
                if asset_id in artifact['asset_ids']:
                    row = self.db.execute('SELECT principal FROM jobs WHERE id=?', (artifact['job_id'],)).fetchone()
                    if row:
                        owners.add(row[0])
            oracle.require(owners == {principal}, 'forbidden')
        else:
            oracle.require(owner == principal, 'forbidden')
        return metadata['sha256']

    def register(self, principal, key, version, *, expected_revision):
        # Canonical round-trip isolates caller mutation, rejects non-JSON/NaN.
        version = json.loads(_encode(version))
        oracle.validate('voice_version', version)
        oracle.require(type(expected_revision) is int and expected_revision >= 0, 'invalid_request')
        payload = dict(version=version, expected_revision=expected_revision)
        def apply():
            # New registrations only: old records/read/replays remain exact.
            oracle.require(len(_encode(version).encode('utf-8')) <= 65536, 'resource_limit')
            def bounded(value):
                if isinstance(value, str):
                    oracle.require(len(value.encode('utf-8')) <= 16384, 'resource_limit')
                elif isinstance(value, dict):
                    for key, item in value.items(): bounded(key); bounded(item)
                elif isinstance(value, list):
                    for item in value: bounded(item)
            bounded(version)
            voice_id = version['voice_id']
            prior = self.db.execute('SELECT record FROM catalogue_versions WHERE principal=? AND id=?', (principal,version['id'])).fetchone()
            oracle.require(prior is None, 'version_conflict')
            exists = self.db.execute('SELECT 1 FROM catalogue_voices WHERE principal=? AND id=?', (principal,voice_id)).fetchone()
            if exists:
                voice = self.get(principal,voice_id)
                oracle.require(not voice['archived'], 'voice_archived')
                oracle.revision(expected_revision,voice['revision'])
            else:
                oracle.revision(expected_revision,0)
                oracle.require(not self.db.execute('SELECT 1 FROM catalogue_names WHERE principal=? AND name=?', (principal,voice_id)).fetchone(), 'alias_conflict')
            hashes = {a: self._asset(principal,a) for a in version['reference_asset_ids']}
            if 'reference_asset_sha256' in version:
                oracle.require(version['reference_asset_sha256'] == hashes, 'reference_mismatch')
            version['reference_asset_sha256'] = hashes
            if not exists:
                self.db.execute('INSERT INTO catalogue_voices(principal,id,revision,archived) VALUES(?,?,0,0)', (principal,voice_id))
                self.db.execute('INSERT INTO catalogue_names VALUES(?,?,?)', (principal,voice_id,voice_id))
            self.db.execute('INSERT INTO catalogue_versions(principal,id,voice_id,record) VALUES(?,?,?,?)', (principal,version['id'],voice_id,_encode(version)))
            self.db.execute('UPDATE catalogue_voices SET revision=revision+1 WHERE principal=? AND id=?', (principal,voice_id))
            return dict(voice=self.get(principal,voice_id), version=version)
        return self._mutate(principal,'voice_register',key,payload,apply)

    def aliases(self, principal, key, voice_id, aliases, *, expected_revision):
        _id(voice_id)
        oracle.require(isinstance(aliases,list) and len(aliases)<=32, 'invalid_request')
        aliases = sorted(_id(a) for a in aliases)
        oracle.require(len(set(aliases))==len(aliases) and voice_id not in aliases, 'invalid_request')
        def apply():
            voice = self.get(principal,voice_id)
            oracle.require(not voice['archived'], 'voice_archived')
            oracle.revision(expected_revision,voice['revision'])
            for name in aliases:
                row = self.db.execute('SELECT voice_id FROM catalogue_names WHERE principal=? AND name=?', (principal,name)).fetchone()
                oracle.require(row is None or row[0]==voice_id, 'alias_conflict')
            self.db.execute('DELETE FROM catalogue_names WHERE principal=? AND voice_id=? AND name!=?', (principal,voice_id,voice_id))
            for name in aliases:
                self.db.execute('INSERT INTO catalogue_names VALUES(?,?,?)', (principal,name,voice_id))
            self.db.execute('UPDATE catalogue_voices SET revision=revision+1 WHERE principal=? AND id=?', (principal,voice_id))
            return self.get(principal,voice_id)
        oracle.require(type(expected_revision) is int and expected_revision>=0, 'invalid_request')
        return self._mutate(principal,'voice_aliases',key,dict(voice_id=voice_id,aliases=aliases,expected_revision=expected_revision),apply)

    def archive(self, principal, key, voice_id, *, expected_revision):
        _id(voice_id)
        oracle.require(type(expected_revision) is int and expected_revision>=0, 'invalid_request')
        def apply():
            voice = self.get(principal,voice_id)
            oracle.revision(expected_revision,voice['revision'])
            oracle.require(not voice['archived'], 'voice_archived')
            self.db.execute('UPDATE catalogue_voices SET archived=1,revision=revision+1 WHERE principal=? AND id=?', (principal,voice_id))
            return self.get(principal,voice_id)
        return self._mutate(principal,'voice_archive',key,dict(voice_id=voice_id,expected_revision=expected_revision),apply)


def _encode(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False)
