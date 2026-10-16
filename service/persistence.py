"""Private A1 publication format; not deployment enrollment or migration."""
import hashlib
import os
import stat
import sqlite3
from functools import wraps


class PublicationQuarantine(ValueError):
    """Ambiguous namespace/provenance: retain intent and every name."""


def initialize(db, *, fresh):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check_recovery_open(db)
    private = tables & {'p1_format', 'publication_work'}
    if private:
        if private != {'p1_format', 'publication_work'}:
            raise PublicationQuarantine('partial_p1_format')
        if db.execute('SELECT revision,scope FROM p1_format').fetchall() != [(1, 'synthetic-core')]:
            raise PublicationQuarantine('unsupported_p1_format')
        expected = {
            'p1_format': ['revision', 'scope'],
            'publication_work': ['job_id', 'binding', 'destination', 'temporary', 'sha256', 'byte_length'],
        }
        for table, columns in expected.items():
            if [r[1] for r in db.execute('PRAGMA table_info('+table+')')] != columns:
                raise PublicationQuarantine('partial_p1_schema')
        return True
    if not fresh:
        return False  # Never stamp an old root, even an empty old database.
    db.executescript('''BEGIN IMMEDIATE;
        CREATE TABLE p1_format(revision INTEGER NOT NULL, scope TEXT NOT NULL);
        INSERT INTO p1_format VALUES (1,'synthetic-core');
        CREATE TABLE publication_work(
          job_id TEXT PRIMARY KEY REFERENCES publications(job_id),
          binding TEXT NOT NULL, destination TEXT NOT NULL UNIQUE,
          temporary TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,
          byte_length INTEGER NOT NULL CHECK(byte_length BETWEEN 1 AND 1048576));
        COMMIT;''')
    return True


def mkdir_durable(path, sync_dir):
    """Persist each new name in its parent, including a newly created root."""
    if not path.exists():
        mkdir_durable(path.parent, sync_dir)
        path.mkdir(mode=0o700)
        sync_dir(path)
        sync_dir(path.parent)


def write_complete(stream, data):
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0 or written > len(view):
            raise OSError('incomplete_publication_write')
        view = view[written:]
    stream.flush()
    if os.fstat(stream.fileno()).st_size != len(data):
        raise OSError('incomplete_publication_length')
    os.fsync(stream.fileno())


def verify_file(path, digest, length, *, links):
    if not 1 <= length <= 1048576:
        raise PublicationQuarantine('publication_length_limit')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != links or info.st_size != length):
            raise PublicationQuarantine('foreign_publication_inode')
        data = bytearray()
        while len(data) < length + 1:
            chunk = os.read(fd, min(65536, length + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) != length or hashlib.sha256(data).hexdigest() != digest:
            raise PublicationQuarantine('publication_bytes_mismatch')
        os.fsync(fd)
        return info
    finally:
        os.close(fd)


def storage_session(method):
    """A failed storage operation needs reopen/read-state, never live retry."""
    @wraps(method)
    def guarded(core, *args, **kwargs):
        if getattr(core, '_publication_storage_failed', False):
            raise OSError('publication_reopen_required')
        try:
            job_id = args[0] if args else kwargs['job_id']
            check_publication_allowed(core.db, job_id)
            return method(core, *args, **kwargs)
        except (OSError, sqlite3.Error, PublicationQuarantine):
            core._publication_storage_failed = True
            raise
    return guarded


def initialize_inventory(core):
    """Additive private revision-1 diagnostics, only on an existing A1 root.

    This does not enroll an old/populated pre-P1 installation or create authority.
    """
    if not core.p1:
        raise PublicationQuarantine('pre_p1_inspect_only')
    expected = {
        'p1_inventory_scans': ['id','observed','complete','fingerprint','summary'],
        'p1_inventory_objects': ['id','first_observed','last_observed','scan_id','record'],
        'p1_inventory_history': ['seq','scan_id','object_id','observed','record'],
    }
    tables = {r[0] for r in core.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    existing = tables & expected.keys()
    if existing:
        if existing != expected.keys(): raise PublicationQuarantine('partial_inventory_schema')
        for table, columns in expected.items():
            if [r[1] for r in core.db.execute('PRAGMA table_info('+table+')')] != columns:
                raise PublicationQuarantine('unsupported_inventory_schema')
        initialize_integrity(core)
        return
    with core.transaction():
        core.db.execute('CREATE TABLE p1_inventory_scans(id TEXT PRIMARY KEY, observed REAL NOT NULL, complete INTEGER NOT NULL, fingerprint TEXT, summary TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_inventory_objects(id TEXT PRIMARY KEY, first_observed REAL NOT NULL, last_observed REAL NOT NULL, scan_id TEXT NOT NULL REFERENCES p1_inventory_scans(id), record TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_inventory_history(seq INTEGER PRIMARY KEY, scan_id TEXT NOT NULL REFERENCES p1_inventory_scans(id), object_id TEXT NOT NULL, observed REAL NOT NULL, record TEXT NOT NULL)')

    initialize_integrity(core)


def initialize_integrity(core):
    """Revision 1: missing populated diagnostics are unresolved, never healthy."""
    db = core.db
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='p1_integrity'").fetchone()
    if exists:
        if [r[1] for r in db.execute('PRAGMA table_info(p1_integrity)')] != ['id','revision','state','token']:
            raise PublicationQuarantine('unsupported_integrity_schema')
        if db.execute('SELECT id,revision FROM p1_integrity').fetchall() != [(1,1)]:
            raise PublicationQuarantine('unsupported_integrity_revision')
        return
    populated = any(db.execute('SELECT 1 FROM '+t+' LIMIT 1').fetchone() for t in
                    ('p1_inventory_scans','p1_inventory_objects','p1_inventory_history'))
    with core.transaction():
        db.execute('CREATE TABLE p1_integrity(id INTEGER PRIMARY KEY, revision INTEGER NOT NULL, state TEXT NOT NULL, token TEXT)')
        db.execute('INSERT INTO p1_integrity VALUES(1,1,?,NULL)', ('unresolved' if populated else 'healthy',))


# B1 records deliberately contain no resource epoch or cleanup authority.
B1_COLUMNS = {
    'p1_installation': 'id revision scope record digest',
    'p1_registrations': 'id record digest',
    'p1_recovery_plans': 'id record digest',
}


def initialize_enrollment(core, record):
    """Only the create-only initializer calls this, after all predecessors exist."""
    from service.core import encode
    tables = {r[0] for r in core.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables & B1_COLUMNS.keys():
        for table, columns in B1_COLUMNS.items():
            if [r[1] for r in core.db.execute('PRAGMA table_info('+table+')')] != columns.split():
                raise PublicationQuarantine('partial_enrollment_schema')
        rows = core.db.execute('SELECT record,digest FROM p1_installation').fetchall()
        if rows != [(encode(record), hashlib.sha256(encode(record).encode()).hexdigest())]:
            raise PublicationQuarantine('enrollment_mismatch')
        return
    with core.transaction():
        core.db.execute('CREATE TABLE p1_installation(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, scope TEXT NOT NULL, record TEXT NOT NULL, digest TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_registrations(id TEXT PRIMARY KEY, record TEXT NOT NULL, digest TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_recovery_plans(id TEXT PRIMARY KEY, record TEXT NOT NULL, digest TEXT NOT NULL)')
        core.db.execute('INSERT INTO p1_installation VALUES(?,1,?,?,?)',
                        (record['deployment_id'], 'owned-synthetic', encode(record), hashlib.sha256(encode(record).encode()).hexdigest()))


# Optional B2 extension. Normal constructors validate it, never install it.
B2_COLUMNS = {
    'p1_b2_format': 'id revision',
    'p1_abandon_plans': 'id record digest',
    'p1_recoveries': 'id plan_id record digest',
    'p1_recovery_steps': 'recovery_id step record digest',
    'dispatch_resolutions': 'dispatch_seq recovery_id record digest',
    'p1_retired_publications': 'job_id recovery_id record digest',
    'p1_integrity_resolutions': 'object_id recovery_id record digest',
}
B2_DDL = (
    'CREATE TABLE p1_b2_format(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL CHECK(revision=1))',
    'CREATE TABLE p1_abandon_plans(id TEXT PRIMARY KEY, record TEXT NOT NULL, digest TEXT NOT NULL)',
    'CREATE TABLE p1_recoveries(id TEXT PRIMARY KEY, plan_id TEXT NOT NULL UNIQUE REFERENCES p1_abandon_plans(id), record TEXT NOT NULL, digest TEXT NOT NULL)',
    'CREATE TABLE p1_recovery_steps(recovery_id TEXT NOT NULL REFERENCES p1_recoveries(id), step TEXT NOT NULL, record TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(recovery_id,step))',
    'CREATE TABLE dispatch_resolutions(dispatch_seq INTEGER NOT NULL REFERENCES dispatches(seq), recovery_id TEXT NOT NULL REFERENCES p1_recoveries(id), record TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(dispatch_seq,recovery_id))',
    'CREATE TABLE p1_retired_publications(job_id TEXT NOT NULL REFERENCES publications(job_id), recovery_id TEXT NOT NULL REFERENCES p1_recoveries(id), record TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(job_id,recovery_id))',
    'CREATE TABLE p1_integrity_resolutions(object_id TEXT NOT NULL REFERENCES p1_inventory_objects(id), recovery_id TEXT NOT NULL REFERENCES p1_recoveries(id), record TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(object_id,recovery_id))',
)


def b2_schema(db):
    names = db.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 2049").fetchall()
    if len(names)>2048: raise PublicationQuarantine('b2_schema_budget')
    tables = {r[0] for r in names}
    present = tables & B2_COLUMNS.keys()
    if not present:
        return False
    if present != B2_COLUMNS.keys():
        raise PublicationQuarantine('partial_b2_schema')
    for table, ddl in zip(B2_COLUMNS, B2_DDL):
        if db.execute('SELECT length(sql) FROM sqlite_master WHERE name=?',(table,)).fetchone()[0]>65536:
            raise PublicationQuarantine('b2_schema_budget')
        sql = db.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone()[0]
        if ' '.join(sql.split()) != ' '.join(ddl.split()):
            raise PublicationQuarantine('unsupported_b2_schema')
    if db.execute('SELECT id,revision FROM p1_b2_format').fetchall() != [(1,1)]:
        raise PublicationQuarantine('unsupported_b2_format')
    return True


def check_recovery_open(db):
    """DB mutation authority only; cannot attest the receipt volume."""
    if not b2_schema(db):
        return None
    from service.recovery import validate_b2_database
    try:
        state = validate_b2_database(db)
        completed = {r['recovery_id']: r for r in state['p1_recovery_steps'] if r['step']=='core-complete'}
        for row in state['p1_recoveries']:
            if row['id'] not in completed:
                raise ValueError('b2_recovery_pending')
        return state
    except (ValueError, TypeError, KeyError) as exc:
        raise PublicationQuarantine('b2_'+str(exc)) from exc


def check_publication_allowed(db, job_id):
    state = check_recovery_open(db)
    if state and any(r['job_id']==job_id for r in state['p1_retired_publications']):
        raise PublicationQuarantine('b2_retired_publication')
