"""Owner-only bounded P1A2 diagnostics. No deletion or resource attestation.

Paths remain private SQLite evidence; callers receive IDs and bounded summaries.
Unknown tables are deliberately unsupported until their reference adapter is added.
"""
from collections import Counter
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import sqlite3
import threading
import time
import uuid

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from service.core import encode
from service import persistence

# Approved B1-only exact-column adapter; no recovery/apply authority here.
from service.worker_transport import Authority


TABLES = {
    'sqlite_sequence', 'jobs', 'revisions', 'events', 'publications', 'assets',
    'artifacts', 'p1_format', 'publication_work', 'dispatch_settings', 'dispatches',
    'dispatch_commands', 'resource_reservations', 'resource_accounting',
    'resource_faults', 'resource_monitor', 'resource_protocol',
    'p1_inventory_scans', 'p1_inventory_objects', 'p1_inventory_history', 'p1_integrity',
}
from service.voice_store import _TABLE_DDL, _TRIGGER_DDL, _TABLES as CATALOGUE_TABLES, _schema_identity
TABLES.update(CATALOGUE_TABLES)
TABLES.update(persistence.B1_COLUMNS)
TABLES.update(persistence.B2_COLUMNS)
DIAGNOSTICS = {'p1_inventory_scans', 'p1_inventory_objects', 'p1_inventory_history', 'p1_integrity'}

COLUMNS = {
    'jobs': 'seq id principal operation key fingerprint job context retry_of',
    'revisions': 'project revision snapshot', 'events': 'seq job_id state',
    'publications': 'job_id manifest artifact message', 'assets': 'id metadata',
    'artifacts': 'id metadata', 'publication_work': 'job_id binding destination temporary sha256 byte_length',
    'p1_format': 'revision scope', 'dispatch_settings': 'job_id fixture seconds',
    'dispatches': 'seq job_id authority binding fixture runtime phase receipt outcome acked',
    'dispatch_commands': 'seq dispatch_id op state',
    'resource_reservations': 'id category demand state', 'resource_accounting': 'id highwater',
    'resource_faults': 'reason', 'resource_monitor': 'id state', 'resource_protocol': 'id revision',
}

# Columns derive from VoiceStore's sole supported DDL, not table-name trust.
_catalogue_reference = sqlite3.connect(':memory:')
try:
    for _sql in (*_TABLE_DDL, *_TRIGGER_DDL): _catalogue_reference.execute(_sql)
    CATALOGUE_IDENTITY = _schema_identity(_catalogue_reference)
    COLUMNS.update({t:' '.join(r[1] for r in _catalogue_reference.execute('PRAGMA table_info('+t+')')) for t in CATALOGUE_TABLES})
finally:
    _catalogue_reference.close()

COLUMNS.update(persistence.B1_COLUMNS)


def stable(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class Incomplete(ValueError):
    pass


class Reconciliation:
    def __init__(self, core, staging=None, *, entries=2048, byte_budget=16*1024**2, seconds=1):
        if type(entries)!=int or not 1<=entries<=10000 or type(byte_budget)!=int or not 1<=byte_budget<=64*1024**2 or type(seconds) not in (int,float) or not 0<=seconds<=1:
            raise ValueError('finite_inventory_budget')
        self.core, self.db = core, core.db
        self.staging = Path(staging).absolute() if staging is not None else None
        self.owner = threading.get_ident()
        self.entry_limit, self.byte_limit, self.seconds = entries, byte_budget, seconds
        self.scalar_limit = 16*1024**2 if core.publication_format==2 else byte_budget
        self.media_limit = 160*1024**2 if core.publication_format==2 else byte_budget
        self.separate_media = core.publication_format==2
        self.scan_id = None
        self.cursors = {}
        persistence.initialize_inventory(core)
        self.summary = dict(complete=False, blocked=True, reasons=['not_scanned'], counts={})

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise ValueError('inventory_owner')
        if self.db.in_transaction:
            raise ValueError('inventory_transaction')

    def _charge(self, size=0, *, media=False):
        self.entries += 1
        if media and self.separate_media: self.media_bytes += size
        else: self.bytes += size
        if self.entries > self.entry_limit or self.bytes > min(self.byte_limit,self.scalar_limit) or self.media_bytes > self.media_limit or time.monotonic() > self.end:
            raise Incomplete('inventory_budget')

    def _row(self, namespace, name, kind, reason, *, identity=None, owner=None, refs=()):
        key = hashlib.sha256(encode([namespace, name]).encode()).hexdigest()
        self.rows[key] = dict(id=key, namespace=namespace, name=name, classification=kind,
            reason=reason, identity=identity, owner=owner, references=sorted(set(refs)))

    def _collect(self, *, _before=None, policy=None, _export_public=None, _family=None):
        # Trusted offline read context only; never infer ownership from a row.
        # Worker attempt ownership continues to use its existing allocation map.
        if policy is not None:
            from service import install_policy
            install_policy.require(policy)
        def control_owned(info, *, directory=False):
            if policy is None: return info.st_uid == os.getuid()
            uid,gid,mode = install_policy.expectation(policy,'control',directory=directory)
            return (info.st_uid,info.st_gid,stat.S_IMODE(info.st_mode)) == (uid,gid,mode)
        # Offline/native callers intentionally bypass the mutating constructor.
        self.scalar_limit = self.byte_limit
        self.media_limit = self.byte_limit
        self.separate_media = False
        self._export_family = _family
        self.entries = self.bytes = self.media_bytes = 0
        self.end = time.monotonic() + self.seconds
        self.rows, self.reasons = {}, set()
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        schema = self.db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name").fetchall()
        self._charge(len(encode(schema).encode()))
        self.fingerprint = hashlib.sha256(encode(schema).encode())
        if tables - TABLES:
            self.reasons.add('unknown_reference_schema')
        try: self.publication_format = persistence.store_format(self.db)
        except (ValueError,sqlite3.Error): raise Incomplete('unknown_publication_format') from None
        if self.publication_format==2 and policy is None and not tables & persistence.B1_COLUMNS.keys():
            self.media_limit=160*1024**2
            self.separate_media=True
            self.scalar_limit=16*1024**2
        from service import installed_export as ex
        if _family is not None or _export_public is not None:
            if _family is not ex.FAMILY or policy is not ex.ip.SPLIT:
                raise Incomplete('explicit_export_family_required')
            ex.database(self.db, _export_public)
            self.media_limit=ex.MEDIA_BYTES
            self.scalar_limit=ex.SCALAR_BYTES
            self.separate_media=True
        b2 = None
        if tables & persistence.B2_COLUMNS.keys():
            from service.recovery import validate_b2_database
            b2 = validate_b2_database(self.db, self.core.root, _uncertified=_before['id'] if _before else None)
        if tables & CATALOGUE_TABLES:
            if tables & CATALOGUE_TABLES != CATALOGUE_TABLES or _schema_identity(self.db) != CATALOGUE_IDENTITY:
                raise Incomplete('unknown_reference_schema')
            if self.db.execute('SELECT revision FROM voice_catalogue_format').fetchall() != [(1,)]:
                raise Incomplete('unknown_reference_schema')
        data = {}
        for table in sorted(tables & TABLES - DIAGNOSTICS - {'sqlite_sequence'}):
            columns = [r[1] for r in self.db.execute('PRAGMA table_info('+table+')')]
            if columns != {**COLUMNS, **persistence.B2_COLUMNS}[table].split():
                raise Incomplete('unknown_reference_schema')
            records = []
            # Reject oversized rows before materializing their content in Python.
            sizes = '+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in columns)
            if self.db.execute('SELECT 1 FROM '+table+' WHERE '+sizes+' > ? LIMIT 1', (self.byte_limit,)).fetchone():
                raise Incomplete('inventory_budget')
            for raw in self.db.execute('SELECT * FROM '+table+' ORDER BY rowid'):
                serialized = encode(raw)
                self._charge(len(serialized.encode()))
                self.fingerprint.update(serialized.encode())
                records.append(dict(zip(columns, raw)))
            data[table] = records
        if _before is not None:
            # Only the recovery projection oracle calls this, after verifying
            # every declared after row. Re-render its exact original joins/refs.
            from service.recovery import CORE_OUTPUT_TABLES, RESOLUTION_TABLES
            for table in CORE_OUTPUT_TABLES:
                data[table]=[dict(zip(COLUMNS[table].split(),r)) for r in _before['before']['source']['data'][table]]
            for table in RESOLUTION_TABLES:
                data[table]=[r for r in data[table] if r['recovery_id']!=_before['id']]
        document = json.loads(Path('contracts/v1/schema.json').read_text())
        (refs, assets, jobs, contexts, attempts, owners, work, job_owner, work_owner, reservations, dispatches) = typed_references(self, data, b2, _before, document)
        seen = set()
        def file_entry(directory_fd, name, namespace, owner=None):
            self._charge()
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            identity = [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
                        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            kind, reason = 'unknown_foreign', 'unknown_namespace'
            regular = stat.S_ISREG(info.st_mode)
            digest = None
            if regular:
                # Charge each name, including aliases. Never claim hardlink credit.
                self._charge(info.st_size,media=True)
                if namespace == 'assets':
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                    try:
                        if stable(os.fstat(fd)) != stable(info): raise Incomplete('inventory_changed')
                        h = hashlib.sha256()
                        remaining = info.st_size
                        while remaining:
                            if time.monotonic() > self.end: raise Incomplete('inventory_budget')
                            chunk = os.read(fd, min(65536, remaining))
                            if not chunk: raise Incomplete('inventory_changed')
                            h.update(chunk); remaining -= len(chunk)
                        if stable(os.fstat(fd)) != stable(info) or stable(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)) != stable(info): raise Incomplete('inventory_changed')
                        digest = h.hexdigest()
                    finally: os.close(fd)
            if namespace == 'assets':
                owner = work_owner(name)
                seen.add(name)
                if name in assets:
                    kind, reason = 'committed_referenced', 'retained_metadata'
                    if not regular or not control_owned(info) or info.st_nlink != 1 or digest != assets[name]['sha256']:
                        kind, reason = 'missing_hash_mismatched_terminal', 'unavailable'
                elif name in work:
                    kind, reason = 'pending_adoption_known_alias', 'publication_work'
                    owner = work_owner(name)
                    if (not regular or not control_owned(info) or info.st_nlink not in (1,2)
                        or (name == work[name]['destination'] and
                            (digest != work[name]['sha256'] or info.st_size != work[name]['byte_length']))):
                        kind, reason = 'unknown_foreign', 'foreign_adoption'
                elif re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}', name) and not name.startswith('adoption-') and regular and control_owned(info) and info.st_nlink == 1:
                    kind, reason = 'adopted_orphan', 'no_publication_provenance'
            elif (owner and (regular or stat.S_ISDIR(info.st_mode))
                  and (not regular or info.st_nlink == 1)
                  and (namespace == 'owners' or namespace in ('internal_staging','worker_staging') or name == 'output.wav' or
                       (name in ('reference.wav','audition.wav') and
                        contexts.get(owner['job_id'],{}).get('plan',{}).get('reference_roles',{}).get('reference_kind')=='prepared_reference'))):
                kind, reason = 'owned_synthetic_transient', 'attempt_record'
            # Conditioning directories have no closed durable retained-namespace
            # representation. Preserve UNKNOWN even when currently empty; stat
            # identity alone cannot attest emptiness to detached history readers.
            if kind in ('unknown_foreign','missing_hash_mismatched_terminal','adopted_orphan'):
                self.reasons.add(reason)
            self._row(namespace, name,
                kind, reason, identity=identity+[digest], owner=owner, refs=refs.get(name, ()))
            return info
        def directory(path, namespace, known, *, files=False, parent_owner=None, parent_fd=None):
            # Each descent is descriptor-relative and NOFOLLOW; even a renamed
            # parent cannot redirect this scan into an arbitrary host directory.
            self._charge()
            info = os.stat(path, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode): raise Incomplete('foreign_inventory_root')
            if policy is not None and namespace in ('assets','owners','internal_staging'):
                if not control_owned(info,directory=True): raise Incomplete('foreign_inventory_root')
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                if stable(os.fstat(fd)) != stable(info): raise Incomplete('inventory_changed')
                with os.scandir(fd) as entries:
                    for entry in entries:
                        owner = known.get(entry.name, parent_owner)
                        child = file_entry(fd, entry.name, namespace, owner)
                        if not files and owner and stat.S_ISDIR(child.st_mode):
                            if child.st_dev != info.st_dev: raise Incomplete('inventory_mount_crossing')
                            directory(entry.name, namespace+'/'+entry.name, {}, files=True, parent_owner=owner, parent_fd=fd)
                if stable(os.fstat(fd)) != stable(info) or stable(os.stat(path, dir_fd=parent_fd, follow_symlinks=False)) != stable(info):
                    raise Incomplete('inventory_changed')
            finally: os.close(fd)
        archives = {}
        if b2 is not None:
            from service.recovery import archive_names
            for intent in b2['p1_recoveries']:
                for name, kind in archive_names(intent).items():
                    if name in archives: raise Incomplete('duplicate_b2_archive')
                    archives[name] = (kind, intent)
        if policy is not None:
            with install_policy.parent_fd(self.core.root) as (parent,name):
                root_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        else:
            root_fd = os.open(self.core.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            root_before = os.fstat(root_fd)
            if policy is not None: install_policy.check(root_before,policy,'control',directory=True)
            with os.scandir(root_fd) as root_entries:
                for entry in root_entries:
                    self._charge()
                    info = os.stat(entry.name, dir_fd=root_fd, follow_symlinks=False)
                    if entry.name == 'native-renewal' and policy is None:
                        from service.native_renewal import namespace
                        evidence = namespace(self.core.root,resume=getattr(self,'native_renewal_resume',False))
                        for name, value in evidence.items():
                            raw = encode(value).encode()
                            self._charge(len(raw))
                            self._row('native-renewal', name, 'committed_referenced',
                                      'validated_native_renewal_preparation',
                                      identity=[hashlib.sha256(raw).hexdigest()])
                        continue
                    if entry.name in archives:
                        kind, intent = archives[entry.name]
                        if not stat.S_ISREG(info.st_mode) or not control_owned(info) or info.st_nlink!=1 or info.st_mode & 0o077:
                            raise Incomplete('foreign_b2_archive')
                        from service.persistent_init import read_file
                        from service.recovery import validate_archive_document
                        content = read_file(self.core.root/entry.name, self.byte_limit-self.bytes,
                            **(dict(policy=policy,role='control') if policy is not None else {}))
                        self._charge(len(content))
                        if kind!='snapshot': validate_archive_document(kind,json.loads(content),intent)
                        self._row('recovery',entry.name,'recovery_evidence','retained_recovery_evidence',
                            identity=[*stable(info),hashlib.sha256(content).hexdigest()],refs=['p1_recoveries:'+intent['id']])
                        continue
                    is_directory = entry.name in ('assets','staging','owners')
                    is_control = entry.name in ('core.sqlite','core.sqlite-wal','core.sqlite-shm','control.lock')
                    if not ((is_directory and stat.S_ISDIR(info.st_mode) and (policy is None or control_owned(info,directory=True))) or
                            (is_control and stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and control_owned(info))):
                        self.reasons.add('unknown_control_namespace')
                        self._row('control', entry.name, 'unknown_foreign', 'unknown_control_namespace', identity=list(stable(info)))
            directory('assets', 'assets', {}, files=True, parent_fd=root_fd)
            directory('owners', 'owners', owners, files=True, parent_fd=root_fd)
            directory('staging', 'internal_staging', attempts, parent_fd=root_fd)
            if policy is not None:
                with install_policy.parent_fd(self.core.root) as (parent,name):
                    if (stable(os.fstat(root_fd)) != stable(root_before) or
                            stable(os.stat(name,dir_fd=parent,follow_symlinks=False)) != stable(root_before)):
                        raise Incomplete('inventory_changed')
        finally: os.close(root_fd)
        if self.staging is not None and self.staging != self.core.root/'staging':
            directory(self.staging, 'worker_staging', attempts)
        for asset_id in refs.keys() | assets.keys():
            if asset_id not in seen:
                if asset_id in work and asset_id not in assets:
                    self._row('assets', asset_id, 'pending_adoption_known_alias', 'incomplete_adoption', refs=refs.get(asset_id, ()))
                    continue
                self.reasons.add('unavailable')
                self._row('assets', asset_id, 'missing_hash_mismatched_terminal', 'unavailable', owner=work_owner(asset_id), refs=refs.get(asset_id, ()))
        for row in sorted(self.rows.values(), key=lambda r:r['id']):
            self.fingerprint.update(encode(row).encode())
        return self.fingerprint.hexdigest()

    def _begin(self):
        self._owner()
        if getattr(self, 'failed_epoch', False): return False
        try:
            with self.core.transaction():
                self.inherited_fault = self.db.execute('SELECT state FROM p1_integrity WHERE id=1').fetchone()[0] == 'unresolved'
                token = uuid.uuid4().hex
                changed = self.db.execute("UPDATE p1_integrity SET state='checking',token=? WHERE id=1 AND revision=1 AND state IN ('healthy','unresolved')", (token,)).rowcount
                if changed != 1: raise Incomplete('unacknowledged_integrity')
            self.epoch = token
            return True
        except (OSError, ValueError, sqlite3.Error) as exc:
            self.failed_epoch = True
            self.summary = dict(complete=False, blocked=True, defect_latched=True, counts={},
                reasons=['unacknowledged_integrity' if isinstance(exc, Incomplete) else 'inventory_persistence_failed'], cleanup_allowed=False)
            return False

    def _resolved_diagnostic(self,row):
        if row['classification'] not in ('adopted_orphan','missing_hash_mismatched_terminal','unresolved_owner_receipt_reservation'):
            return False
        from service.recovery import validate_b2_database, diagnostic_digest
        state=validate_b2_database(self.db,self.core.root)
        if not state:return False
        completed={s['recovery_id'] for s in state['p1_recovery_steps'] if s['step']=='core-complete'}
        content=diagnostic_digest(row)
        if row['classification']=='unresolved_owner_receipt_reservation':
            if row['namespace']!='dispatch':return False
            for resolution in state['dispatch_resolutions']:
                if resolution['recovery_id'] not in completed or str(resolution['dispatch_seq'])!=row['name']:continue
                accepted=next(r for r in state['p1_recoveries'] if r['id']==resolution['recovery_id'])
                original=next((json.loads(r[4]) for r in accepted['before']['source']['data']['p1_inventory_objects'] if r[0]==row['id']),None)
                if original and original['reason']==row['reason'] and diagnostic_digest(original)==content:return True
            return False
        return any(r['recovery_id'] in completed and r['object_id']==row['id'] and r['reason']==row['reason'] and r['content_digest']==content for r in state['p1_integrity_resolutions'])

    def reconcile_startup(self):
        self._owner()
        if not self._begin(): return dict(self.summary)
        self.scan_id = uuid.uuid4().hex
        self.cursors = {}
        complete = False
        self.rows, self.reasons = {}, set()
        try:
            self.db.set_progress_handler(lambda: int(time.monotonic() > self.end), 1000)
            from service import installed_export as ex, install_policy as ip
            expected = getattr(self.core, "_installed_family", None)
            if type(expected) is ex._Expectation:
                fingerprint = self._collect(policy=ip.SPLIT, _family=ex.FAMILY,
                    _export_public=json.loads(expected._public))
            else:
                fingerprint = self._collect()
            complete = True
        except (OSError, sqlite3.Error, ValueError, ValidationError, KeyError, IndexError, TypeError, RecursionError) as exc:
            self.reasons.add(str(exc) if isinstance(exc, Incomplete) else 'inventory_read_failed')
            fingerprint = None
        finally:
            self.db.set_progress_handler(None, 0)
        # A2 cannot acknowledge an observed defect, even when somebody replaces
        # bytes/removes a name. B2 will own explicit diagnostic resolutions.
        # Bound aggregate serialized diagnostic bytes BEFORE fetching/decoding.
        try:
            count, size = self.db.execute('SELECT count(*),coalesce(sum(length(CAST(record AS BLOB))),0) FROM p1_inventory_objects').fetchone()
            summary_size = self.db.execute('SELECT coalesce(sum(length(CAST(summary AS BLOB))),0) FROM p1_inventory_scans').fetchone()[0]
            history_count, history_size = self.db.execute('SELECT count(*),coalesce(sum(length(CAST(record AS BLOB))),0) FROM p1_inventory_history').fetchone()
            if count + history_count > self.entry_limit or size + summary_size + history_size > max(0, self.byte_limit - self.bytes):
                raise Incomplete('inventory_history_budget')
            prior = self.db.execute('SELECT record FROM p1_inventory_objects')
            if any((row:=json.loads(raw))['classification'] in ('unknown_foreign','adopted_orphan','missing_hash_mismatched_terminal','unresolved_owner_receipt_reservation') and not self._resolved_diagnostic(row) for (raw,) in prior):
                self.reasons.add('unacknowledged_integrity')
        except (sqlite3.Error, ValueError, TypeError, KeyError):
            complete = False
            self.reasons.add('inventory_history_budget')
        for reason in ('unavailable','no_publication_provenance'):
            related=[r for r in self.rows.values() if r['reason']==reason]
            if related and all(self._resolved_diagnostic(r) for r in related):self.reasons.discard(reason)
        if self.inherited_fault: self.reasons.add('unacknowledged_integrity')
        counts = dict(Counter(r['classification'] for r in self.rows.values()))
        defect_latched = bool(self.reasons & {'unacknowledged_integrity','unknown_reference_schema','unknown_reference_kind'}) or any(r['classification'] in ('unknown_foreign','adopted_orphan','missing_hash_mismatched_terminal','unresolved_owner_receipt_reservation') and not self._resolved_diagnostic(r) for r in self.rows.values())
        self.summary = dict(scan_id=self.scan_id, complete=complete, defect_latched=defect_latched,
            blocked=not complete or bool(self.reasons), counts=counts, reasons=sorted(self.reasons)[:20],
            cleanup_allowed=False)
        if getattr(self, 'page_expected', fingerprint) != fingerprint:
            self.reasons.add('inventory_changed')
            self.summary.update(complete=False, blocked=True, reasons=['inventory_changed'])
            complete = False
        now = time.time()
        try:
            with self.core.transaction():
                self.db.execute('INSERT INTO p1_inventory_scans VALUES(?,?,?,?,?)',
                    (self.scan_id, now, int(complete), fingerprint, encode(self.summary)))
                for row in self.rows.values():
                    previous = self.db.execute('SELECT first_observed FROM p1_inventory_objects WHERE id=?',(row['id'],)).fetchone()
                    self.db.execute('INSERT OR REPLACE INTO p1_inventory_objects VALUES(?,?,?,?,?)',
                        (row['id'], previous[0] if previous else now, now, self.scan_id, encode(row)))
                    self.db.execute('INSERT INTO p1_inventory_history(scan_id,object_id,observed,record) VALUES(?,?,?,?)',
                        (self.scan_id, row['id'], now, encode(row)))
                if getattr(self, 'failed_epoch', False): raise Incomplete('integrity_epoch_interrupted')
                state = 'healthy' if complete and not self.reasons else 'unresolved'
                if self.db.execute('UPDATE p1_integrity SET state=?,token=NULL WHERE id=1 AND revision=1 AND state="checking" AND token=?', (state,self.epoch)).rowcount != 1:
                    raise Incomplete('integrity_epoch_lost')
        except (OSError, ValueError, sqlite3.Error):
            self.failed_epoch = True
            self.summary.update(complete=False, blocked=True, defect_latched=True, reasons=['inventory_persistence_failed'])
        self.expected = fingerprint
        return dict(self.summary)

    def inventory_page(self, scan_id, cursor=None, limit=20):
        self._owner()
        if type(limit) != int or not 1 <= limit <= 20: raise ValueError('inventory_page_limit')
        if scan_id != self.scan_id or not self.summary['complete']: raise ValueError('invalid_inventory_scan')
        if cursor is not None and cursor not in self.cursors: raise ValueError('invalid_inventory_cursor')
        expected, generation, cursors = self.expected, self.scan_id, self.cursors
        self.page_expected = expected
        try: result = self.reconcile_startup()
        finally: del self.page_expected
        unchanged = result['complete'] and self.expected == expected
        self.scan_id, self.cursors = generation, cursors
        self.summary['scan_id'] = generation
        if not unchanged:
            # A fingerprint change is itself defect-bearing, even a healthy new image.
            self.summary.update(complete=False, blocked=True, reasons=['inventory_changed'])
            self.failed_epoch = True
            raise ValueError('stale_inventory_cursor')
        offset = self.cursors[cursor] if cursor is not None else 0
        rows = sorted(self.rows.values(), key=lambda r:r['id'])
        selected = rows[offset:offset+limit]
        token = None
        if offset+limit < len(rows):
            token = next((k for k,v in self.cursors.items() if v == offset+limit), None) or uuid.uuid4().hex
            self.cursors[token] = offset+limit
        # No arbitrary host paths or giant metadata; exact private records remain in DB.
        return dict(scan_id=scan_id, entries=[dict({k:r[k] for k in ('id','classification','reason','owner')}, references=r['references'][:20]) for r in selected], next_cursor=token)


def typed_references(self, data, b2, _before, document):
    """Existing bounded typed joins; callbacks collect diagnostics, no filesystem."""
    completed={r['recovery_id'] for r in (b2 or {}).get('p1_recovery_steps',[]) if r['step']=='core-complete'}
    resolved={r['dispatch_seq']:r for r in (b2 or {}).get('dispatch_resolutions',[]) if r['recovery_id'] in completed and (_before is None or r['recovery_id']!=_before['id'])}
    refs = {}
    # Closed contract locations, never scalar spelling or user-authored keys.
    schema = document['$defs']
    validator = Draft202012Validator(document)
    reference_keys = {'asset_id','asset_ids','input_asset_ids','result_asset_ids',
                      'source_asset_id','audition_asset_id','reference_asset_id','reference_asset_ids'}
    def add(value, source):
        if isinstance(value, list):
            for item in value: add(item, source)
        elif isinstance(value, str):
            refs.setdefault(value, set()).add(source if len(source)<=256 else source.split(':')[0]+':'+hashlib.sha256(source.encode()).hexdigest())
        else: self.reasons.add('unknown_reference_kind')
    def walk(value, definition, source, path=(), planned_prepare=False):
        if definition == 'asset' and isinstance(value, dict): add(value.get('id'), source)
        node = schema[definition] if isinstance(definition, str) else definition
        if '$ref' in node:
            name = node['$ref'].split('/')[-1]
            if name == 'asset' and isinstance(value, dict): add(value.get('id'), source)
            return walk(value, name, source, path, planned_prepare)
        for union in ('oneOf', 'anyOf', 'allOf'):
            if union in node:
                # Delegate type, $ref and discriminator applicability to the
                # same Draft 2020-12 implementation used by the contract oracle.
                branches = [n for n in node[union]
                            if validator.evolve(schema=n).is_valid(value)]
                if union == 'oneOf' and len(branches) != 1:
                    self.reasons.add('unknown_reference_kind')
                if union == 'allOf' and len(branches) != len(node[union]):
                    self.reasons.add('unknown_reference_kind')
                if not branches: self.reasons.add('unknown_reference_kind')
                for branch in branches: walk(value, branch, source, path, planned_prepare)
                return
        if 'type' in node and not validator.evolve(schema={'type': node['type']}).is_valid(value):
            self.reasons.add('unknown_reference_kind')
            return
        if not isinstance(value, (dict, list)) and not validator.evolve(schema=node).is_valid(value):
            self.reasons.add('unknown_reference_kind')
        if isinstance(value, list):
            if 'items' in node:
                for item in value: walk(item, node['items'], source, path, planned_prepare)
        elif isinstance(value, dict):
            properties = node.get('properties', {})
            if set(value) - properties.keys(): self.reasons.add('unknown_reference_schema')
            for key, item in value.items():
                if key not in properties: continue
                # Only these closed prepare-plan fields declare future outputs.
                # Published manifests/artifacts and saved versions are walked
                # independently and continue to require their actual assets.
                if planned_prepare and ((path == ('reference_roles',) and key in ('reference_asset_id','audition_asset_id')) or
                        (path == ('resolved','reuse_configuration') and key == 'reference_asset_id')):
                    continue
                if key == 'reference_asset_sha256' and definition == 'voice_version':
                    if validator.evolve(schema=properties[key]).is_valid(item): add(list(item), source)
                    else: self.reasons.add('unknown_reference_kind')
                elif key in reference_keys:
                    if validator.evolve(schema=properties[key]).is_valid(item): add(item, source)
                    else: self.reasons.add('unknown_reference_kind')
                else: walk(item, properties[key], source, path+(key,), planned_prepare)
    def context(value, source):
        if set(value) == {'plan','bundles','execution'}:
            from worker.runtime_protocol import execution
            try:
                if execution(value) is None: raise ValueError('execution_required')
            except (ValueError, KeyError, TypeError):
                raise Incomplete('unknown_reference_kind') from None
        elif set(value) != {'plan','bundles'}:
            self.reasons.add('unknown_reference_schema')
        elif value['plan']['resolved']['profile']['id'] in ('kokoro-cpu','pocket-cpu','qwen-cpu','qwen-cuda','qwen-mps'):
            raise Incomplete('unknown_reference_kind')
        plan=value['plan']
        planned_prepare=False
        if plan.get('operation')=='voice_prepare' and 'reference_roles' in plan:
            from contracts import validate as oracle
            try:
                oracle.validate('worker_plan',plan)
                oracle.frozen_plan(plan,value['bundles'])
            except (ValueError,KeyError,TypeError,ValidationError):
                raise Incomplete('unknown_reference_kind') from None
            planned_prepare=plan['reference_roles']['reference_kind']=='prepared_reference'
        walk(plan, 'worker_plan', source, planned_prepare=planned_prepare)
        for bundle in value['bundles']: walk(bundle, 'bundle_catalogue_entry', source)
    def catalogue_version(value, principal, source):
        from contracts import validate as oracle
        try: oracle.validate('voice_version', value)
        except (ValueError, ValidationError): raise Incomplete('unknown_reference_kind') from None
        if not any(r['principal']==principal and r['id']==value['voice_id'] for r in data.get('catalogue_voices', [])):
            raise Incomplete('unknown_reference_kind')
        if set(value.get('reference_asset_sha256',{})) != set(value['reference_asset_ids']):
            raise Incomplete('unknown_reference_kind')
        for asset_id in value['reference_asset_ids']:
            asset = next((json.loads(r['metadata']) for r in data.get('assets',[]) if r['id']==asset_id), None)
            if asset is None: raise Incomplete('unknown_reference_kind')
            owner = asset.get('principal_id')
            if owner is None:
                producing = {r['principal'] for r in data.get('jobs',[]) for a in data.get('artifacts',[])
                             if json.loads(a['metadata'])['job_id']==r['id'] and asset_id in json.loads(a['metadata'])['asset_ids']}
                if producing != {principal}: raise Incomplete('unknown_reference_kind')
            elif owner != principal: raise Incomplete('unknown_reference_kind')
            if value.get('reference_asset_sha256',{}).get(asset_id) != asset['sha256']:
                raise Incomplete('unknown_reference_kind')
        walk(value, 'voice_version', source)

    adapters = {
        'jobs': {'job':'job', 'context':None}, 'revisions': {'snapshot':'project'},
        'assets': {'metadata':'asset'}, 'artifacts': {'metadata':'artifact'},
        'publications': {'manifest':'result_manifest','artifact':'artifact','message':'worker_message'},
    }
    for table, records in data.items():
        for index, record in enumerate(records):
            source = table+':'+str(record.get('id', record.get('job_id', record.get('project', index))))+(':'+str(record['revision']) if 'revision' in record else '')
            if table in persistence.B2_COLUMNS and table not in ('p1_b2_format','p1_abandon_plans','p1_recoveries'):
                source = table+':'+str(record.get('recovery_id'))+':'+str(record.get('step',record.get('dispatch_seq',record.get('job_id',record.get('object_id')))))
            for column, definition in adapters.get(table, {}).items():
                value = json.loads(record[column])
                if column == 'context': context(value, source)
                else: walk(value, definition, source)
            if table == 'artifacts':
                try: persistence.public_artifact(record, data,publication_format=self.publication_format)
                except persistence.PublicationQuarantine:
                    raise Incomplete('public_artifact_mismatch') from None
            if table in CATALOGUE_TABLES - {'voice_catalogue_format'}:
                from service.voice_store import _id
                try:
                    _id(record['principal'])
                    for field in ('id','voice_id','name','key'):
                        if field in record: _id(record[field])
                except ValueError: raise Incomplete('unknown_reference_kind') from None
            if table == 'catalogue_voices':
                if type(record['revision']) is not int or record['revision'] < 1 or record['archived'] not in (0,1):
                    raise Incomplete('unknown_reference_kind')
            if table == 'catalogue_names':
                if not any(r['principal']==record['principal'] and r['id']==record['voice_id'] for r in data.get('catalogue_voices', [])):
                    raise Incomplete('unknown_reference_kind')
            if table == 'catalogue_versions':
                value = json.loads(record['record'])
                if value.get('id') != record['id'] or value.get('voice_id') != record['voice_id']:
                    raise Incomplete('unknown_reference_kind')
                catalogue_version(value, record['principal'], source)
            if table == 'catalogue_requests':
                if not re.fullmatch('[a-f0-9]{64}',record['fingerprint']): raise Incomplete('unknown_reference_kind')
                value = json.loads(record['result'])
                definitions = {'voice_register':'voice_register_result', 'voice_aliases':'voice',
                               'voice_archive':'voice', 'job_cancel':'job_cancel_request'}
                definition = definitions.get(record['operation'])
                if definition is None: raise Incomplete('unknown_reference_kind')
                from contracts import validate as oracle
                try: oracle.validate(definition, value)
                except (ValueError, ValidationError): raise Incomplete('unknown_reference_kind') from None
                if record['operation'] == 'voice_register': catalogue_version(value['version'],record['principal'],source)
                if record['operation'] == 'job_cancel':
                    if not any(r['id']==value['job_id'] and r['principal']==record['principal'] for r in data.get('jobs', [])):
                        raise Incomplete('unknown_reference_kind')
                walk(value, definition, source)
            if table in persistence.B1_COLUMNS:
                from service.persistent_init import digest, validate_scope
                from service.recovery import validate_plan, validate_registration, closed
                value = json.loads(record['record'])
                if digest(value) != record['digest']:
                    raise Incomplete('invalid_b1_digest')
                if table == 'p1_installation':
                    from service.install_policy import installation
                    from service import installed_export as ex
                    (ex.installation if getattr(self,'_export_family',None) is ex.FAMILY else installation)(value)
                    if (value['deployment_id'] != record['id'] or record['revision'] != 1 or
                        value['revision'] != 1 or record['scope'] != value['scope']):
                        raise Incomplete('invalid_b1_installation')
                elif table == 'p1_registrations':
                    from service import installed_export as ex
                    (ex.host_registration if getattr(self,'_export_family',None) is ex.FAMILY else validate_registration)(value)
                    if record['id'] != digest(value): raise Incomplete('invalid_b1_registration')
                else:
                    validate_plan(value)
                    if record['id'] != value['id']: raise Incomplete('invalid_b1_plan')
                    # Typed references retain the referenced canonical documents.
                    # Plan metadata/prose and arbitrary scalar strings are not assets.
                    for job_id in value['affected']['jobs']:
                        target = next((r for r in data.get('jobs', []) if r['id'] == job_id), None)
                        if target is None: raise Incomplete('unknown_b1_job_reference')
                        walk(json.loads(target['job']), 'job', source)
                        context(json.loads(target['context']), source)
                    for dispatch_id in value['affected']['dispatches']:
                        target = next((r for r in data.get('dispatches', []) if r['seq'] == dispatch_id), None)
                        if target is None: raise Incomplete('unknown_b1_dispatch_reference')
                        authority = json.loads(target['authority'])
                        walk(authority[0], 'job', source); context(authority[1], source); add(authority[2], source)
                    for job_id in value['affected']['publications']:
                        target = next((r for r in data.get('publications', []) if r['job_id'] == job_id), None)
                        if target is None: raise Incomplete('unknown_b1_publication_reference')
                        for column, definition in adapters['publications'].items():
                            walk(json.loads(target[column]), definition, source)
                installations = data.get('p1_installation', [])
                if (len(installations) != 1 or value['deployment_id'] != installations[0]['id']):
                    raise Incomplete('unknown_b1_deployment_reference')
            if table in persistence.B2_COLUMNS and table != 'p1_b2_format':
                value = json.loads(record['record'])
                # Follow only closed typed references. Resolutions have no
                # reconciliation effect until the later kernel lot.
                affected = value.get('affected', {})
                for job_id in set(affected.get('jobs', []) + affected.get('reservations', []) + ([value['job_id']] if 'job_id' in value else [])):
                    target = next((r for r in data.get('jobs', []) if r['id']==job_id), None)
                    if target is None: raise Incomplete('unknown_b2_job_reference')
                    walk(json.loads(target['job']), 'job', source)
                    context(json.loads(target['context']), source)
                for dispatch_id in affected.get('dispatches', []) + ([value['dispatch_seq']] if 'dispatch_seq' in value else []):
                    target = next((r for r in data.get('dispatches', []) if r['seq']==dispatch_id), None)
                    if target is None: raise Incomplete('unknown_b2_dispatch_reference')
                    authority = json.loads(target['authority'])
                    walk(authority[0], 'job', source); context(authority[1], source); add(authority[2], source)
                for job_id in affected.get('publications', []) + ([value['job_id']] if table=='p1_retired_publications' else []):
                    target = next((r for r in data.get('publications', []) if r['job_id']==job_id), None)
                    if target is None: raise Incomplete('unknown_b2_publication_reference')
                    for column, definition in adapters['publications'].items(): walk(json.loads(target[column]), definition, source)
            if table == 'publication_work':
                from service.persistence import publication_record
                artifact,entries=publication_record(record,data,publication_format=self.publication_format)
                for entry in entries: add(entry['destination'], source)
                walk(artifact, 'artifact', source)
            if table == 'dispatches':
                authority = json.loads(record['authority'])
                walk(authority[0], 'job', source)
                context(authority[1], source)
                for profile in authority[3]: walk(profile, 'profile', source)
                add(authority[2], source)
                if record['outcome']: walk(json.loads(record['outcome']), 'job', source)
                if record['receipt']:
                    receipt = json.loads(record['receipt'])
                    from worker.native_protocol import terminal_receipt
                    terminal_receipt(receipt, authority=Authority(*authority), runtime_id=record['runtime'])
            if table == 'resource_reservations':
                if set(json.loads(record['demand'])) != {'configuration','slots','ram_bytes','vram_bytes','pids','disk_bytes','attempt_disk_bytes'}:
                    self.reasons.add('unknown_reference_schema')
    assets = {r['id']: json.loads(r['metadata']) for r in data.get('assets', [])}
    jobs = {r['id']: json.loads(r['job']) for r in data.get('jobs', [])}
    contexts = {r['id']: json.loads(r['context']) for r in data.get('jobs', [])}
    attempts, owners = {}, {}
    for job in jobs.values():
        for a in job['attempts']:
            owner = {k: a[k] for k in ('id', 'generation', 'owner_id', 'staging_id')}
            owner.update(job_id=job['id'], input_sha256=job['input_sha256'])
            if any(not isinstance(v,(str,int)) or len(str(v))>128 for v in owner.values()):
                raise Incomplete('invalid_owner_tuple')
            attempts[a['staging_id']] = owner
            owners[a['owner_id']] = owner
    work = {}
    for r in data.get('publication_work', []):
        from service.persistence import publication_record
        artifact,entries=publication_record(r,data,publication_format=self.publication_format)
        for entry in entries:
            expanded=dict(r,**entry);expanded['binding']=json.dumps(artifact)
            work[entry['destination']]=expanded
            work[entry['temporary']]=expanded
    def job_owner(job_id):
        known = jobs.get(job_id, {}).get('attempts', [])
        return attempts.get(known[-1]['staging_id']) if known else None
    def work_owner(name):
        return attempts.get(json.loads(work[name]['binding'])['staging_id']) if name in work else None
    reservations = {r['id']: r for r in data.get('resource_reservations', [])}
    dispatches = data.get('dispatches', [])
    for r in dispatches:
        if r['acked']: continue
        dispatch_reasons = set()
        try:
            a = Authority(*json.loads(r['authority']))
            job = jobs[r['job_id']]
            if (a.binding != r['binding'] or a.job['id'] != job['id'] or
                a.job['attempts'][-1] != {**job['attempts'][-1], 'reaped': a.job['attempts'][-1]['reaped']} or
                a.job['generation'] != job['generation'] or a.context != contexts[job['id']] or
                a.job['deadline'] != job['deadline']):
                dispatch_reasons.add('dispatch_owner_mismatch')
            if r['runtime'] is None: dispatch_reasons.add('unbound_dispatch')
            reservation = reservations.get(job['id'])
            if reservation:
                demand = json.loads(reservation['demand'])
                from worker.runtime_protocol import execution
                frozen_value = a.context['execution'] if execution(a.context) is not None else a.context['plan']['resolved']['identity']['effective']
                frozen_configuration = hashlib.sha256(encode(frozen_value).encode()).hexdigest()
                if (reservation['category'] != 'inference' or demand['configuration'] != frozen_configuration or
                    demand['disk_bytes'] < demand['attempt_disk_bytes'] * len(job['attempts'])):
                    dispatch_reasons.add('reservation_binding_mismatch')
            if not reservation or reservation['state'] not in ('executing', 'retained'):
                dispatch_reasons.add('missing_or_invalid_reservation')
            if reservation and ((reservation['state'] == 'retained' and not r['outcome']) or (reservation['state'] == 'executing' and r['outcome'])):
                dispatch_reasons.add('reservation_outcome_mismatch')
            if r['receipt']:
                receipt = json.loads(r['receipt'])
                from worker.native_protocol import terminal_receipt
                terminal_receipt(receipt, authority=a, runtime_id=r['runtime'])
            elif r['outcome'] or job['state'] in ('succeeded','failed','cancelled'):
                dispatch_reasons.add('terminal_receipt_missing')
            publication = next((p for p in data.get('publications', []) if p['job_id'] == job['id']), None)
            if publication:
                artifact = json.loads(publication['artifact'])
                for key in ('attempt_id','owner_id','staging_id','generation'):
                    expected = a.job['generation'] if key == 'generation' else a.job['attempts'][-1]['id' if key == 'attempt_id' else key]
                    if artifact[key] != expected: dispatch_reasons.add('publication_owner_mismatch')
        except (ValueError, ValidationError, KeyError, IndexError, TypeError, RecursionError):
            dispatch_reasons.add('invalid_dispatch_identity')
        resolution=resolved.get(r['seq'])
        # Consume old mismatches only after independently checking all
        # current joins. Allowed reasons cannot erase a new strict defect.
        permitted = {'unbound_dispatch', 'reservation_outcome_mismatch', 'terminal_receipt_missing'}
        if resolution:
            accepted = next(v for v in b2['p1_recoveries'] if v['id'] == resolution['recovery_id'])
            expected = accepted['expected'][3]['output']['rows']
            job_row = next(v for v in expected['jobs'] if v[1] == r['job_id'])
            reservation_row = next(v for v in expected['resource_reservations'] if v[0] == r['job_id'])
            current_job_row = next(v for v in data['jobs'] if v['id'] == r['job_id'])
            if (current_job_row != dict(zip(COLUMNS['jobs'].split(), job_row)) or
                reservation != dict(zip(COLUMNS['resource_reservations'].split(), reservation_row))):
                dispatch_reasons.add('recovered_job_reservation_mismatch')
            if not dispatch_reasons - permitted:
                dispatch_reasons.difference_update(permitted)
        if dispatch_reasons:
            self.reasons.update(dispatch_reasons)
            # Retain one stable object; summary retains every reason.
            reason = sorted(dispatch_reasons - permitted or dispatch_reasons)[0]
            self._row('dispatch', str(r['seq']), 'unresolved_owner_receipt_reservation', reason, owner=job_owner(r['job_id']))
    if 'resource_reservations' in data:
        for job_id in jobs.keys() - reservations.keys():
            self.reasons.add('missing_reservation')
            self._row('reservation', job_id, 'unresolved_owner_receipt_reservation', 'missing_reservation', owner=job_owner(job_id))
    for reservation in reservations.values():
        if reservation['id'] not in jobs:
            self.reasons.add('reservation_job_missing')
            self._row('reservation', reservation['id'], 'unresolved_owner_receipt_reservation', 'reservation_job_missing')
        if reservation['state'] == 'executing' and not any(r['job_id']==reservation['id'] and not r['acked'] for r in dispatches):
            self.reasons.add('unexplained_executing_reservation')
            self._row('reservation', reservation['id'], 'unresolved_owner_receipt_reservation', 'unexplained_executing_reservation', owner=job_owner(reservation['id']))
    return refs, assets, jobs, contexts, attempts, owners, work, job_owner, work_owner, reservations, dispatches
