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
        return
    with core.transaction():
        core.db.execute('CREATE TABLE p1_inventory_scans(id TEXT PRIMARY KEY, observed REAL NOT NULL, complete INTEGER NOT NULL, fingerprint TEXT, summary TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_inventory_objects(id TEXT PRIMARY KEY, first_observed REAL NOT NULL, last_observed REAL NOT NULL, scan_id TEXT NOT NULL REFERENCES p1_inventory_scans(id), record TEXT NOT NULL)')
        core.db.execute('CREATE TABLE p1_inventory_history(seq INTEGER PRIMARY KEY, scan_id TEXT NOT NULL REFERENCES p1_inventory_scans(id), object_id TEXT NOT NULL, observed REAL NOT NULL, record TEXT NOT NULL)')
