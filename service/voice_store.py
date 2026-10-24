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


def initialize(core):
    """Atomic additive migration; never enroll or un-fence a pre-P1 root."""
    if not core.p1:
        return
    expected = {
        'voice_catalogue_format': ['revision'],
        'catalogue_voices': ['seq','principal','id','revision','archived'],
        'catalogue_names': ['principal','name','voice_id'],
        'catalogue_versions': ['seq','principal','id','voice_id','record'],
        'catalogue_requests': ['principal','operation','key','fingerprint','result'],
    }
    tables = {r[0] for r in core.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    existing = tables & expected.keys()
    if existing:
        if existing != expected.keys():
            raise ValueError('partial_voice_store_schema')
        for table, columns in expected.items():
            if [r[1] for r in core.db.execute('PRAGMA table_info('+table+')')] != columns:
                raise ValueError('unsupported_voice_store_schema')
        if core.db.execute('SELECT revision FROM voice_catalogue_format').fetchall() != [(1,)]:
            raise ValueError('unsupported_voice_store_version')
    with core.transaction():
        core.db.execute('CREATE TABLE IF NOT EXISTS voice_catalogue_format(revision INTEGER NOT NULL)')
        rows = core.db.execute('SELECT revision FROM voice_catalogue_format').fetchall()
        if rows and rows != [(1,)]:
            raise ValueError('unsupported_voice_store_version')
        if not rows:
            core.db.execute('INSERT INTO voice_catalogue_format VALUES(1)')
        core.db.execute('''CREATE TABLE IF NOT EXISTS catalogue_voices(
            seq INTEGER PRIMARY KEY, principal TEXT NOT NULL, id TEXT NOT NULL,
            revision INTEGER NOT NULL, archived INTEGER NOT NULL,
            UNIQUE(principal,id))''')
        core.db.execute('''CREATE TABLE IF NOT EXISTS catalogue_names(
            principal TEXT NOT NULL, name TEXT NOT NULL, voice_id TEXT NOT NULL,
            PRIMARY KEY(principal,name), FOREIGN KEY(principal,voice_id)
            REFERENCES catalogue_voices(principal,id))''')
        core.db.execute('''CREATE TABLE IF NOT EXISTS catalogue_versions(
            seq INTEGER PRIMARY KEY, principal TEXT NOT NULL, id TEXT NOT NULL,
            voice_id TEXT NOT NULL, record TEXT NOT NULL, UNIQUE(principal,id),
            FOREIGN KEY(principal,voice_id) REFERENCES catalogue_voices(principal,id))''')
        core.db.execute('''CREATE TABLE IF NOT EXISTS catalogue_requests(
            principal TEXT NOT NULL, operation TEXT NOT NULL, key TEXT NOT NULL,
            fingerprint TEXT NOT NULL, result TEXT NOT NULL,
            PRIMARY KEY(principal,operation,key))''')
        # Defense against accidental UPDATE/DELETE by future domain code.
        for action in ('UPDATE', 'DELETE'):
            core.db.execute(f'''CREATE TRIGGER IF NOT EXISTS catalogue_versions_no_{action.lower()}
                BEFORE {action} ON catalogue_versions BEGIN
                SELECT RAISE(ABORT,'immutable_voice_version'); END''')


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

    def versions(self, principal, voice_id, *, after=0, limit=20):
        self.get(principal,voice_id)
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
