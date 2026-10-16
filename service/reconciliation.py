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

from jsonschema.exceptions import ValidationError

from service.core import encode
from service import persistence
from service.worker_transport import Authority


TABLES = {
    'sqlite_sequence', 'jobs', 'revisions', 'events', 'publications', 'assets',
    'artifacts', 'p1_format', 'publication_work', 'dispatch_settings', 'dispatches',
    'dispatch_commands', 'resource_reservations', 'resource_accounting',
    'resource_faults', 'resource_monitor', 'resource_protocol',
    'p1_inventory_scans', 'p1_inventory_objects', 'p1_inventory_history', 'p1_integrity',
}
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
        self.scan_id = None
        self.cursors = {}
        persistence.initialize_inventory(core)
        self.summary = dict(complete=False, blocked=True, reasons=['not_scanned'], counts={})

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise ValueError('inventory_owner')
        if self.db.in_transaction:
            raise ValueError('inventory_transaction')

    def _charge(self, size=0):
        self.entries += 1
        self.bytes += size
        if self.entries > self.entry_limit or self.bytes > self.byte_limit or time.monotonic() > self.end:
            raise Incomplete('inventory_budget')

    def _row(self, namespace, name, kind, reason, *, identity=None, owner=None, refs=()):
        key = hashlib.sha256(encode([namespace, name]).encode()).hexdigest()
        self.rows[key] = dict(id=key, namespace=namespace, name=name, classification=kind,
            reason=reason, identity=identity, owner=owner, references=sorted(set(refs)))

    def _collect(self):
        self.entries = self.bytes = 0
        self.end = time.monotonic() + self.seconds
        self.rows, self.reasons = {}, set()
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        schema = self.db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name").fetchall()
        self._charge(len(encode(schema).encode()))
        self.fingerprint = hashlib.sha256(encode(schema).encode())
        if tables - TABLES:
            self.reasons.add('unknown_reference_schema')
        data = {}
        for table in sorted(tables & TABLES - DIAGNOSTICS - {'sqlite_sequence'}):
            columns = [r[1] for r in self.db.execute('PRAGMA table_info('+table+')')]
            if columns != COLUMNS[table].split():
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
        refs = {}
        # Closed contract locations, never scalar spelling or user-authored keys.
        schema = json.loads(Path('contracts/v1/schema.json').read_text())['$defs']
        reference_keys = {'asset_id','asset_ids','input_asset_ids','result_asset_ids',
                          'audition_asset_id','reference_asset_ids'}
        def add(value, source):
            if isinstance(value, list):
                for item in value: add(item, source)
            elif isinstance(value, str):
                refs.setdefault(value, set()).add(source if len(source)<=256 else source.split(':')[0]+':'+hashlib.sha256(source.encode()).hexdigest())
            else: self.reasons.add('unknown_reference_kind')
        def walk(value, definition, source):
            if value is None: return
            if definition == 'asset' and isinstance(value, dict): add(value.get('id'), source)
            node = schema[definition] if isinstance(definition, str) else definition
            if '$ref' in node:
                name = node['$ref'].split('/')[-1]
                if name == 'asset' and isinstance(value, dict): add(value.get('id'), source)
                return walk(value, name, source)
            for union in ('oneOf', 'anyOf', 'allOf'):
                if union in node:
                    # Closed union branches are selected by their declared constants.
                    branches = [n for n in node[union] if all(
                        not isinstance(value, dict) or k not in value or
                        ('const' not in v or value[k] == v['const'])
                        for k,v in n.get('properties', {}).items())]
                    if not branches: self.reasons.add('unknown_reference_kind')
                    for branch in branches: walk(value, branch, source)
                    return
            if isinstance(value, list):
                if 'items' in node:
                    for item in value: walk(item, node['items'], source)
            elif isinstance(value, dict):
                properties = node.get('properties', {})
                if set(value) - properties.keys(): self.reasons.add('unknown_reference_schema')
                for key, item in value.items():
                    if key not in properties: continue
                    if key in reference_keys: add(item, source)
                    else: walk(item, properties[key], source)
        def context(value, source):
            if set(value) != {'plan','bundles'}: self.reasons.add('unknown_reference_schema')
            walk(value['plan'], 'worker_plan', source)
            for bundle in value['bundles']: walk(bundle, 'bundle_catalogue_entry', source)
        adapters = {
            'jobs': {'job':'job', 'context':None}, 'revisions': {'snapshot':'project'},
            'assets': {'metadata':'asset'}, 'artifacts': {'metadata':'artifact'},
            'publications': {'manifest':'result_manifest','artifact':'artifact','message':'worker_message'},
        }
        for table, records in data.items():
            for index, record in enumerate(records):
                source = table+':'+str(record.get('id', record.get('job_id', record.get('project', index))))+(':'+str(record['revision']) if 'revision' in record else '')
                for column, definition in adapters.get(table, {}).items():
                    value = json.loads(record[column])
                    if column == 'context': context(value, source)
                    else: walk(value, definition, source)
                if table == 'publication_work':
                    add(record['destination'], source)
                    walk(json.loads(record['binding']), 'artifact', source)
                if table == 'dispatches':
                    authority = json.loads(record['authority'])
                    walk(authority[0], 'job', source)
                    context(authority[1], source)
                    for profile in authority[3]: walk(profile, 'profile', source)
                    add(authority[2], source)
                    if record['outcome']: walk(json.loads(record['outcome']), 'job', source)
                    if record['receipt']:
                        receipt = json.loads(record['receipt'])
                        if set(receipt) - {'runtime_id','binding','reaped','engine_uid','reap_method','reaped_children','timely','original_deadline'}:
                            self.reasons.add('unknown_reference_schema')
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
            work[r['destination']] = r
            work[r['temporary']] = r
        def job_owner(job_id):
            known = jobs.get(job_id, {}).get('attempts', [])
            return attempts.get(known[-1]['staging_id']) if known else None
        def work_owner(name):
            return attempts.get(json.loads(work[name]['binding'])['staging_id']) if name in work else None
        reservations = {r['id']: r for r in data.get('resource_reservations', [])}
        dispatches = data.get('dispatches', [])
        for r in dispatches:
            if r['acked']: continue
            reason = None
            try:
                a = Authority(*json.loads(r['authority']))
                job = jobs[r['job_id']]
                if (a.binding != r['binding'] or a.job['id'] != job['id'] or
                    a.job['attempts'][-1] != {**job['attempts'][-1], 'reaped': a.job['attempts'][-1]['reaped']} or
                    a.job['generation'] != job['generation'] or a.context != contexts[job['id']] or
                    a.job['deadline'] != job['deadline']):
                    reason = 'dispatch_owner_mismatch'
                if r['runtime'] is None: reason = 'unbound_dispatch'
                reservation = reservations.get(job['id'])
                if reservation:
                    demand = json.loads(reservation['demand'])
                    frozen_configuration = hashlib.sha256(encode(a.context['plan']['resolved']['identity']['effective']).encode()).hexdigest()
                    if (reservation['category'] != 'inference' or demand['configuration'] != frozen_configuration or
                        demand['disk_bytes'] < demand['attempt_disk_bytes'] * len(job['attempts'])):
                        reason = 'reservation_binding_mismatch'
                if not reservation or reservation['state'] not in ('executing', 'retained'):
                    reason = 'missing_or_invalid_reservation'
                if reservation and ((reservation['state'] == 'retained' and not r['outcome']) or (reservation['state'] == 'executing' and r['outcome'])):
                    reason = 'reservation_outcome_mismatch'
                if r['receipt']:
                    receipt = json.loads(r['receipt'])
                    if (receipt.get('runtime_id') != r['runtime'] or receipt.get('binding') != a.binding or
                        receipt.get('original_deadline') != a.job['deadline'] or receipt.get('reaped') is not True or
                        receipt.get('reap_method') != 'pid1-waitpid-echild'):
                        reason = 'receipt_mismatch'
                elif r['outcome'] or job['state'] in ('succeeded','failed','cancelled'):
                    reason = 'terminal_receipt_missing'
                publication = next((p for p in data.get('publications', []) if p['job_id'] == job['id']), None)
                if publication:
                    artifact = json.loads(publication['artifact'])
                    for key in ('attempt_id','owner_id','staging_id','generation'):
                        expected = a.job['generation'] if key == 'generation' else a.job['attempts'][-1]['id' if key == 'attempt_id' else key]
                        if artifact[key] != expected: reason = 'publication_owner_mismatch'
            except (ValueError, ValidationError, KeyError, IndexError, TypeError, RecursionError):
                reason = 'invalid_dispatch_identity'
            if reason:
                self.reasons.add(reason)
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
                self._charge(info.st_size)
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
                    if not regular or info.st_uid != os.getuid() or info.st_nlink != 1 or digest != assets[name]['sha256']:
                        kind, reason = 'missing_hash_mismatched_terminal', 'unavailable'
                elif name in work:
                    kind, reason = 'pending_adoption_known_alias', 'publication_work'
                    owner = attempts.get(json.loads(work[name]['binding'])['staging_id'])
                    if (not regular or info.st_uid != os.getuid() or info.st_nlink not in (1,2)
                        or (name == work[name]['destination'] and
                            (digest != work[name]['sha256'] or info.st_size != work[name]['byte_length']))):
                        kind, reason = 'unknown_foreign', 'foreign_adoption'
                elif re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}', name) and not name.startswith('adoption-') and regular and info.st_uid == os.getuid() and info.st_nlink == 1:
                    kind, reason = 'adopted_orphan', 'no_publication_provenance'
            elif (owner and (regular or stat.S_ISDIR(info.st_mode))
                  and (not regular or info.st_nlink == 1)
                  and (namespace == 'owners' or namespace in ('internal_staging','worker_staging') or name == 'output.wav')):
                kind, reason = 'owned_synthetic_transient', 'attempt_record'
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
        root_fd = os.open(self.core.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with os.scandir(root_fd) as root_entries:
                for entry in root_entries:
                    self._charge()
                    info = os.stat(entry.name, dir_fd=root_fd, follow_symlinks=False)
                    is_directory = entry.name in ('assets','staging','owners')
                    is_control = entry.name in ('core.sqlite','core.sqlite-wal','core.sqlite-shm','control.lock')
                    if not ((is_directory and stat.S_ISDIR(info.st_mode)) or
                            (is_control and stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.getuid())):
                        self.reasons.add('unknown_control_namespace')
                        self._row('control', entry.name, 'unknown_foreign', 'unknown_control_namespace', identity=list(stable(info)))
        finally: os.close(root_fd)
        directory(self.core.root/'assets', 'assets', {}, files=True)
        directory(self.core.root/'owners', 'owners', owners, files=True)
        directory(self.core.root/'staging', 'internal_staging', attempts)
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

    def reconcile_startup(self):
        self._owner()
        if not self._begin(): return dict(self.summary)
        self.scan_id = uuid.uuid4().hex
        self.cursors = {}
        complete = False
        self.rows, self.reasons = {}, set()
        try:
            self.db.set_progress_handler(lambda: int(time.monotonic() > self.end), 1000)
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
            if any(json.loads(raw)['classification'] in ('unknown_foreign','adopted_orphan','missing_hash_mismatched_terminal','unresolved_owner_receipt_reservation') for (raw,) in prior):
                self.reasons.add('unacknowledged_integrity')
        except (sqlite3.Error, ValueError, TypeError, KeyError):
            complete = False
            self.reasons.add('inventory_history_budget')
        if self.inherited_fault: self.reasons.add('unacknowledged_integrity')
        counts = dict(Counter(r['classification'] for r in self.rows.values()))
        defect_latched = bool(self.reasons & {'unacknowledged_integrity','unknown_reference_schema','unknown_reference_kind'}) or any(r['classification'] in ('unknown_foreign','adopted_orphan','missing_hash_mismatched_terminal','unresolved_owner_receipt_reservation') for r in self.rows.values())
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
