"""Bounded offline planning and private pending evidence prefix; apply unavailable."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import tempfile
from types import SimpleNamespace
import uuid

from service.core import encode, sync_dir
from service.persistent_init import digest, read_file, roots_checked, validate_scope
from service.persistence import B1_COLUMNS, B2_COLUMNS
from service.reconciliation import Reconciliation, COLUMNS

MAX_ROWS = 2048
MAX_BYTES = 16 * 1024**2
TTL = 300
HEX = re.compile(r'[0-9a-f]{64}')


def closed(value, keys):
    if not isinstance(value,dict) or set(value) != set(keys.split()): raise ValueError('unknown_record_fields')


def validate_registration(record):
    closed(record, 'deployment_id scope containers lane boot evidence')
    validate_scope(record['scope'])
    if record['evidence'] not in ('E1-mock','actual-host-inspection'): raise ValueError('unknown_host_evidence')
    if not isinstance(record['containers'],list) or not 2<=len(record['containers'])<=16: raise ValueError('bounded_container_set')
    roles = set(); ids = set()
    for c in record['containers']:
        closed(c, 'id image role started volumes')
        if (not isinstance(c['id'],str) or not HEX.fullmatch(c['id']) or c['id'] in ids or
            not isinstance(c['image'],str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',c['image']) or
            c['role'] not in ('control','worker') or not isinstance(c['started'],str) or not 1<=len(c['started'])<=128):
            raise ValueError('exact_container_incarnation_required')
        if not isinstance(c['volumes'],list) or not c['volumes'] or len(set(c['volumes'])) != len(c['volumes']) or set(c['volumes'])-set(record['scope']['volumes']):
            raise ValueError('container_volume_scope')
        required = {'control','staging','ipc'} if c['role']=='control' else {'staging','receipts','ipc'}
        if set(c['volumes']) != required: raise ValueError('container_topology')
        ids.add(c['id']); roles.add(c['role'])
    if roles != {'control','worker'}: raise ValueError('complete_container_roles')
    for k in ('deployment_id','lane','boot'):
        if not isinstance(record[k],str) or not re.fullmatch(r'[0-9a-f]{32}',record[k]): raise ValueError('exact_lane_boot_required')


def validate_plan(record):
    closed(record, 'id deployment_id revision source_digest proof_digest affected count bytes created expires policy apply_available')
    if (not isinstance(record['id'],str) or not re.fullmatch(r'[0-9a-f]{32}',record['id']) or
        record['revision'] != 1 or record['policy'] != 'abandon-uncertain-synthetic-v1' or record['apply_available'] is not False or
        not isinstance(record['source_digest'],str) or not HEX.fullmatch(record['source_digest']) or
        not isinstance(record['proof_digest'],str) or not HEX.fullmatch(record['proof_digest'])):
        raise ValueError('unsupported_plan')
    closed(record['affected'], 'jobs dispatches publications')
    if any(not isinstance(v,list) or len(v)>MAX_ROWS for v in record['affected'].values()): raise ValueError('bounded_affected_set')
    if record['count'] != sum(map(len,record['affected'].values())) or not 0<=record['count']<=MAX_ROWS:
        raise ValueError('affected_count')
    if type(record['bytes'])!=int or not 0<=record['bytes']<=MAX_BYTES: raise ValueError('plan_byte_limit')
    if any(type(record[k]) not in (int,float) for k in ('created','expires')) or not 0<record['expires']-record['created']<=TTL:
        raise ValueError('plan_expiry')
    for key,values in record['affected'].items():
        if len(set(values)) != len(values): raise ValueError('duplicate_affected_id')
        if any((type(v)!=int or v<1) if key=='dispatches' else (not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}',v)) for v in values):
            raise ValueError('typed_affected_id')


@contextmanager
def private_database(path):
    """Bounded stable WAL-aware copy under the normal external owner lock.

    SQLite opens only the private 0700 namespace. Cooperative writers are
    excluded by control.lock; detected source changes refuse the result.
    """
    deadline = time.monotonic()+2
    def capture():
        result = {}; total = 0
        namespace = []
        with os.scandir(path.parent) as entries:
            for entry in entries:
                if len(namespace)>=MAX_ROWS or time.monotonic()>deadline:
                    raise ValueError('snapshot_namespace_budget')
                namespace.append(entry.name)
        namespace.sort()
        for name in ('core.sqlite', 'core.sqlite-wal', 'core.sqlite-shm'):
            source = path.parent/name
            if source.exists() or source.is_symlink():
                if time.monotonic()>deadline: raise ValueError('snapshot_deadline')
                data = read_file(source, MAX_BYTES-total)
                total += len(data)
                if total > MAX_BYTES: raise ValueError('snapshot_byte_budget')
                st = source.stat(follow_symlinks=False)
                result[name] = (data, (st.st_dev, st.st_ino, st.st_mode,
                    st.st_uid, st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        if 'core.sqlite' not in result: raise sqlite3.OperationalError('missing_database')
        return namespace, result
    if not path.parent.is_dir(): raise sqlite3.OperationalError('missing_database')
    before = capture()
    with tempfile.TemporaryDirectory(prefix='p1-inspect-') as directory:
        target = Path(directory)/'core.sqlite'
        for name, (data, _) in before[1].items():
            fd = os.open(Path(directory)/name, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                if stream.write(data) != len(data): raise OSError('short_private_copy')
        if capture() != before: raise ValueError('snapshot_source_changed')
        try:
            yield target
        finally:
            if capture() != before: raise ValueError('snapshot_source_changed')


@contextmanager
def offline(roots, *, write=False, journal_lock=True):
    roots = roots_checked(roots)
    locks = []
    db = None
    private = None
    try:
        lock_paths=[roots['control']/'core'/'control.lock']
        if journal_lock: lock_paths.append(roots['receipts']/'lane.lock')
        for p in lock_paths:
            fd = os.open(p,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            locks.append(fd)
            s=os.fstat(fd)
            if s.st_uid!=os.getuid() or s.st_nlink!=1: raise ValueError('foreign_lock')
            try: fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise ValueError('maintenance_lock_busy') from None
        path=roots['control']/'core'/'core.sqlite'
        if path.is_symlink(): raise ValueError('foreign_database')
        if not write:
            private = private_database(path)
            path = private.__enter__()
        db=sqlite3.connect(path.as_uri()+('?mode=rw' if write else '?mode=ro'),uri=True,timeout=0,isolation_level=None)
        db.execute('PRAGMA trusted_schema=OFF')
        if write: db.execute('PRAGMA synchronous=FULL')
        if not write: db.execute('PRAGMA query_only=ON')
        end=time.monotonic()+2
        db.set_progress_handler(lambda:int(time.monotonic()>end),1000)
        yield roots,db
    finally:
        if db is not None: db.close()
        try:
            if private is not None: private.__exit__(None,None,None)
        finally:
            for fd in reversed(locks): os.close(fd)


def snapshot(roots, db, scope, *, exclude_plan=None, exclude_abandon=None, transaction=True, _prefix=None, _journal_after=False, _core_after=False):
    """Read-only transaction including WAL; never calls an observation fence."""
    validate_scope(scope)
    if transaction: db.execute('BEGIN')
    try:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        expected=dict({k:v for k,v in COLUMNS.items() if k not in B2_COLUMNS},**B1_COLUMNS,
            p1_inventory_scans='id observed complete fingerprint summary',
            p1_inventory_objects='id first_observed last_observed scan_id record',
            p1_inventory_history='seq scan_id object_id observed record',
            p1_integrity='id revision state token',sqlite_sequence='name seq')
        if tables & B2_COLUMNS.keys():
            validate_b2_database(db, roots['control']/'core', _uncertified=_prefix['id'] if _core_after else None)
            expected.update(B2_COLUMNS)
            deadline=time.monotonic()+2
            db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
        if tables != set(expected): raise ValueError('unknown_or_missing_schema')
        data={}; size=0; count=0
        schema=db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name').fetchall()
        size+=len(encode(schema).encode())
        for table in sorted(tables):
            columns=expected[table].split()
            if [r[1] for r in db.execute('PRAGMA table_info('+table+')')] != columns: raise ValueError('unknown_columns')
            sizes='+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in columns)
            n,b=db.execute('SELECT count(*),coalesce(sum('+sizes+'),0) FROM '+table).fetchone()
            count+=n; size+=b
            if count>MAX_ROWS or size>MAX_BYTES: raise ValueError('inspection_budget')
            data[table]=db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall()
        if data['p1_format'] != [(1,'synthetic-core')] or data['resource_protocol'] != [(1,2)] or db.execute('PRAGMA user_version').fetchone() != (1,):
            raise ValueError('unsupported_versions')
        if len(data['p1_installation'])!=1: raise ValueError('not_enrolled')
        identity,revision,kind,raw,h=data['p1_installation'][0]
        installation=json.loads(raw)
        closed(installation,'deployment_id revision scope host profile_budget_digest')
        if (digest(installation)!=h or identity!=installation['deployment_id'] or revision!=1 or kind!='owned-synthetic' or
            installation['revision']!=1 or installation['scope']!='owned-synthetic' or installation['host']!=scope):
            raise ValueError('deployment_scope_mismatch')
        private=json.loads(read_file(roots['receipts']/'p1-init.json'))
        closed(private,'deployment_id revision scope host profile_budget_digest roots seed')
        if ({k:v for k,v in private.items() if k not in ('seed','roots')} != installation or
            private['roots'] != {k:[p.stat().st_dev,p.stat().st_ino] for k,p in roots.items()}):
            raise ValueError('volume_role_identity_mismatch')
        import hashlib
        for key in ('lifecycle','execution'):
            if read_file(roots['ipc']/(key+'.key'),64) != hashlib.sha256((private['seed']+key).encode()).hexdigest().encode():
                raise ValueError('installation_key_missing_or_changed')
        marker=json.loads(read_file(roots['receipts']/'p1-enrolled.json'))
        if marker!=dict(deployment_id=identity,digest=h): raise ValueError('incomplete_enrollment')
        journal_before=bytes.fromhex(_prefix['before']['journal']['bytes_hex']) if _journal_after else None
        damaged=_prefix is not None and _prefix['before']['journal']['state']!='valid'
        if damaged:
            try: journal_raw=journal_before if _journal_after else read_file(roots['receipts']/'journal.json',1024**2)
            except FileNotFoundError:
                if _prefix['before']['journal']['state']!='missing': raise
                journal_raw=b''
            else:
                if _prefix['before']['journal']['state']=='missing' and not _journal_after: raise ValueError('b2_expected_missing_journal')
            if journal_raw.hex()!=_prefix['before']['journal']['bytes_hex']: raise ValueError('b2_prefix_journal_changed')
            # Coverage comes from coherent Core allocations and ALL registrations;
            # these placeholders are never a clean-receipt observation.
            payload=dict(version=1,lane=_prefix['old_lane'],records=[])
            journal=dict(payload=payload,sha256=digest(payload))
        else:
            journal_raw=journal_before if _journal_after else read_file(roots['receipts']/'journal.json',1024**2)
            journal=json.loads(journal_raw)
        closed(journal,'payload sha256')
        payload_keys='version lane records maintenance' if 'maintenance' in journal['payload'] else 'version lane records'
        closed(journal['payload'],payload_keys)
        from worker.runtime_journal import read_maintenance
        receipt_completion=journal['payload'].get('maintenance') if _prefix is not None else read_maintenance(roots['receipts'])
        if receipt_completion!=journal['payload'].get('maintenance'): raise ValueError('journal_maintenance_binding')
        payload=journal['payload']
        if digest(payload)!=journal['sha256'] or payload['version']!=1 or not isinstance(payload['records'],list) or len(payload['records'])>64:
            raise ValueError('journal_identity_unreadable')
        registrations=[]
        for rid,raw,h in data['p1_registrations']:
            r=json.loads(raw); validate_registration(r)
            if digest(r)!=h or r['deployment_id']!=identity or r['scope']!=scope or rid!=digest(r): raise ValueError('registration_digest_scope')
            registrations.append(r)
        journal_ids=set()
        for journal_record in payload['records']:
            fields={'id','allocation','phase','acked','uid','boot'}
            if not isinstance(journal_record,dict) or set(journal_record) not in (fields,fields|{'receipt','proposal'}):
                raise ValueError('unknown_journal_record')
            allocation=journal_record['allocation']
            closed(allocation,'snapshot binding fixture')
            if (digest(allocation['snapshot'])!=allocation['binding'] or journal_record['phase'] not in ('prepared','started','terminal') or
                type(journal_record['acked'])!=bool or not isinstance(journal_record['id'],str) or
                not journal_record['id'].startswith(payload['lane']+'/'+journal_record['boot']+'/')):
                raise ValueError('journal_binding_mismatch')
            from service.worker_transport import Authority
            from contracts import validate as oracle
            authority=Authority(*allocation['snapshot'])
            if (authority.binding!=allocation['binding'] or journal_record['id'] in journal_ids or
                type(journal_record['uid'])!=int or not 20000<=journal_record['uid']<20064):
                raise ValueError('invalid_journal_authority')
            journal_ids.add(journal_record['id'])
            if journal_record['phase']=='terminal':
                receipt=journal_record.get('receipt')
                closed(receipt,'runtime_id binding reaped timely original_deadline engine_uid reap_method reaped_children')
                if (receipt['runtime_id']!=journal_record['id'] or receipt['binding']!=authority.binding or
                    receipt['reaped'] is not True or receipt['reap_method']!='pid1-waitpid-echild' or
                    type(receipt['reaped_children'])!=int or receipt['reaped_children']<0 or
                    receipt['engine_uid']!=journal_record['uid'] or type(receipt['timely'])!=bool or
                    receipt['original_deadline']!=authority.job['deadline']):
                    raise ValueError('invalid_journal_receipt')
                proposal=journal_record.get('proposal')
                if not journal_record['acked'] and receipt['timely'] and proposal is None: raise ValueError('missing_journal_proposal')
                if proposal is not None:
                    closed(proposal,'message manifest')
                    authority.check(proposal['message'],'result'); oracle.validate('result_manifest',proposal['manifest'])
                    for field in ('job_id','attempt_id','generation','owner_id','staging_id','input_sha256'):
                        if proposal['manifest'][field]!=proposal['message'][field]: raise ValueError('journal_manifest_mismatch')
            elif journal_record['acked'] or 'receipt' in journal_record or 'proposal' in journal_record:
                raise ValueError('invalid_journal_phase')
            matching=[r for r in data['dispatches'] if r[5]==journal_record['id'] or (r[5] is None and r[3]==authority.binding)]
            if len(matching)!=1 or matching[0][3]!=authority.binding:
                raise ValueError('journal_dispatch_coverage_unknown')
            if not any(r['lane']==payload['lane'] and r['boot']==journal_record['boot'] for r in registrations):
                raise ValueError('unregistered_journal_incarnation')
        for pid,raw,h in data['p1_recovery_plans']:
            p=json.loads(raw); validate_plan(p)
            if digest(p)!=h or p['id']!=pid or p['deployment_id']!=identity: raise ValueError('plan_digest_scope')
        if _core_after:_verify_core_rows(db,_prefix)
        scanner=Reconciliation.__new__(Reconciliation)
        scanner.core=SimpleNamespace(root=roots['control']/'core'); scanner.db=db; scanner.staging=roots['staging']
        scanner.entry_limit=MAX_ROWS; scanner.byte_limit=MAX_BYTES; scanner.seconds=1
        scanner._collect(_before=_prefix if _core_after else None)
        if tables & B2_COLUMNS.keys():
            size += scanner.bytes + len(journal_raw)
            if size>MAX_BYTES or count+scanner.entries+len(payload['records'])>MAX_ROWS:
                raise ValueError('b2_aggregate_budget')
        # Exclude only this proposal. Later appends invalidate older plans.
        state=dict(data)
        state['p1_recovery_plans']=[r for r in data['p1_recovery_plans'] if r[0]!=exclude_plan]
        if 'p1_abandon_plans' in data:
            state['p1_abandon_plans']=[r for r in data['p1_abandon_plans'] if r[0]!=exclude_abandon]
        if _prefix is not None:
            state['p1_recoveries']=[r for r in data['p1_recoveries'] if r[0]!=_prefix['id']]
            state['p1_recovery_steps']=[r for r in data['p1_recovery_steps'] if r[0]!=_prefix['id']]
            if _core_after:
                _verify_core_rows(db,_prefix)
                for table in (*CORE_OUTPUT_TABLES,'p1_integrity'):
                    state[table]=_prefix['before']['source']['data'][table]
                for table in RESOLUTION_TABLES:
                    state[table]=[r for r in data[table] if r[1]!=_prefix['id']]
        inventory=[]
        for row in sorted(scanner.rows.values(),key=lambda r:r['id']):
            if _prefix is not None and row['namespace']=='recovery' and row['name'] in {n for n,k in archive_names(_prefix).items() if k!='certificate' or _core_after}:
                continue
            inventory.append({**row,'references':[r for r in row['references'] if r not in ('p1_recovery_plans:'+str(exclude_plan), 'p1_abandon_plans:'+str(exclude_abandon)) and not (_prefix is not None and r in _prefix_reference_labels(_prefix))]})
        journal_hash=__import__('hashlib').sha256(journal_raw).hexdigest() if damaged else digest(journal)
        projection=dict(schema=schema,data=state,inventory=inventory,journal=journal_hash,marker=marker)
        fingerprint=digest(projection)
        affected=dict(jobs=[r[1] for r in data['jobs']], dispatches=[r[0] for r in data['dispatches'] if not r[-1]], publications=[r[0] for r in data['publications']])
        if sum(map(len,affected.values()))>MAX_ROWS: raise ValueError('affected_budget')
        completions=[json.loads(r[2]) for r in data.get('p1_recovery_steps',[]) if r[1]=='core-complete']
        return dict(prior_epoch=completions[-1]['epoch'] if completions else identity,
            deployment_id=identity,source_digest=fingerprint,affected=affected,bytes=size,
            evidence_projection=projection,
            monitor=data['resource_monitor'],protocol=2,integrity=data['p1_integrity'],
            reservations=[dict(zip(('job_id','category','demand','state'),(r[0],r[1],json.loads(r[2]),r[3]))) for r in data['resource_reservations']],
            faults=[r[0] for r in data['resource_faults']],
            dispatches=[dict(seq=r[0],job_id=r[1],binding=r[3],runtime_id=r[5],phase=r[6],receipt_digest=digest(json.loads(r[7])) if r[7] else None,acked=bool(r[9])) for r in data['dispatches']],
            jobs=[{k:v for k,v in json.loads(r[6]).items() if k in ('id','state','generation','deadline','attempts')} for r in data['jobs']],
            publications=[dict(job_id=r[0],manifest_digest=digest(json.loads(r[1]))) for r in data['publications']],
            allocation_coherent=_allocation_coherent(state if _core_after else data),
            inventory=dict(counts=dict(__import__('collections').Counter(r['classification'] for r in scanner.rows.values())),reasons=sorted(scanner.reasons)),
            journal_digest=journal_hash,lane=payload['lane'],
            journal_records=None if damaged else [dict(runtime_id=r['id'],boot=r['boot'],phase=r['phase'],binding=r['allocation']['binding']) for r in payload['records']],
            registrations=registrations,
            versions=dict(contract='1.3.0',core=1,private=1,resource=2),apply_available=False)
    finally:
        if transaction: db.execute('ROLLBACK')


def inspect(roots,scope):
    with offline(roots) as (roots,db): return snapshot(roots,db,scope)


def check_consumers(state, observation):
    closed(observation,'scope containers evidence states')
    if not state['registrations']: raise ValueError('missing_registration')
    if observation['evidence'] not in ('E1-mock','actual-host-inspection'): raise ValueError('unknown_host_evidence')
    known={}
    for r in state['registrations']:
        if r['scope']!=observation['scope'] or r['evidence']!=observation['evidence']:
            raise ValueError('registration_scope_evidence_mismatch')
        for c in r['containers']:
            if c['id'] in known and known[c['id']]!=c: raise ValueError('ambiguous_incarnation')
            known[c['id']]=c
    actual=observation['containers']
    if (not isinstance(actual,list) or len(actual)!=len(known) or
        sorted(actual,key=lambda c:c['id'])!=sorted(known.values(),key=lambda c:c['id'])):
        raise ValueError('unknown_or_changed_consumers')
    states=observation['states']
    if not isinstance(states,list) or len(states)!=len(known): raise ValueError('container_state_coverage')
    seen=set()
    for current in states:
        closed(current,'id running restarting pid started finished')
        if (current['id'] not in known or current['id'] in seen or
            current['started']!=known[current['id']]['started'] or type(current['running'])!=bool or
            type(current['restarting'])!=bool or type(current['pid'])!=int or current['pid']<0 or
            not isinstance(current['finished'],str) or len(current['finished'])>128):
            raise ValueError('container_state_mismatch')
        if current['running'] or current['restarting'] or current['pid']!=0:
            raise ValueError('live_volume_consumer')
        seen.add(current['id'])
    if not any(r['lane']==state['lane'] for r in state['registrations']): raise ValueError('unregistered_lane')
    return known



def register(roots,scope,registration):
    validate_registration(registration)
    with offline(roots,write=True,journal_lock=False) as (roots,db):
        state=snapshot(roots,db,scope)
        if registration['deployment_id']!=state['deployment_id'] or registration['scope']!=scope or registration['lane']!=state['lane']:
            raise ValueError('registration_scope_mismatch')
        if digest(json.loads(read_file(roots['receipts']/'journal.json'))) != state['journal_digest']:
            raise ValueError('journal_changed_during_registration')
        rid=digest(registration)
        db.execute('BEGIN IMMEDIATE')
        try:
            db.execute('INSERT INTO p1_registrations VALUES(?,?,?)',(rid,encode(registration),rid)); db.execute('COMMIT')
        except BaseException:
            db.execute('ROLLBACK'); raise
        sync_dir(roots['control']/'core'); sync_dir(roots['control'])
        return rid


def plan(roots,scope,observation, *, now=None):
    now=time.time() if now is None else now
    with offline(roots,write=True) as (roots,db):
        state=snapshot(roots,db,scope); check_consumers(state,observation)
        if (set(state['inventory']['reasons']) & {'unknown_reference_schema','unknown_reference_kind'} or
            state['inventory']['counts'].get('unknown_foreign',0)):
            raise ValueError('unknown_recovery_scope')
        value=dict(id=uuid.uuid4().hex,deployment_id=state['deployment_id'],revision=1,
            source_digest=state['source_digest'],proof_digest=digest(observation),affected=state['affected'],
            count=sum(map(len,state['affected'].values())),bytes=state['bytes'],created=now,expires=now+TTL,
            policy='abandon-uncertain-synthetic-v1',apply_available=False)
        validate_plan(value)
        db.execute('INSERT INTO p1_recovery_plans VALUES(?,?,?)',(value['id'],encode(value),digest(value)))
        sync_dir(roots['control']/'core'); sync_dir(roots['control'])
        return dict(plan=value,digest=digest(value),evidence=observation['evidence'])


def verify_plan(roots,scope,observation,plan_id,expected_digest, *, now=None):
    now=time.time() if now is None else now
    with offline(roots) as (roots,db):
        row=db.execute('SELECT record,digest FROM p1_recovery_plans WHERE id=?',(plan_id,)).fetchone()
        if not row: raise ValueError('unknown_plan')
        value=json.loads(row[0]); validate_plan(value)
        if digest(value)!=row[1] or row[1]!=expected_digest: raise ValueError('plan_digest_mismatch')
        if not value['created']<=now<value['expires']: raise ValueError('expired_plan')
        state=snapshot(roots,db,scope,exclude_plan=plan_id); check_consumers(state,observation)
        if value['source_digest']!=state['source_digest'] or value['proof_digest']!=digest(observation) or value['affected']!=state['affected']:
            raise ValueError('stale_plan')
        return value


def apply(*args,**kwargs):
    raise NotImplementedError('P1B2_apply_unavailable')


def enrollment_gate(root, registration_id, hello, profiles, budget):
    """Read-only pre-Core startup check. No Docker access or new health epoch."""
    path=Path(root).absolute()/'core.sqlite'
    with private_database(path) as private_path, sqlite3.connect(private_path.as_uri()+'?mode=ro',uri=True,timeout=0) as db:
        db.execute('PRAGMA query_only=ON')
        deadline=time.monotonic()+2
        db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
        count=size=0
        for table,columns in B1_COLUMNS.items():
            if [r[1] for r in db.execute('PRAGMA table_info('+table+')')] != columns.split(): raise ValueError('unsupported_enrollment_schema')
            n,b=db.execute('SELECT count(*),coalesce(sum(length(record)),0) FROM '+table).fetchone()
            count+=n; size+=b
            if count>MAX_ROWS or size>MAX_BYTES: raise ValueError('enrollment_read_budget')
        row=db.execute('SELECT record,digest FROM p1_installation').fetchall()
        if len(row)!=1: raise ValueError('not_enrolled')
        installation=json.loads(row[0][0])
        closed(installation,'deployment_id revision scope host profile_budget_digest')
        validate_scope(installation['host'])
        if installation['revision']!=1 or installation['scope']!='owned-synthetic': raise ValueError('unsupported_enrollment')
        if digest(installation)!=row[0][1] or installation['profile_budget_digest']!=digest(dict(profiles=profiles,budget=budget)):
            raise ValueError('enrollment_configuration_mismatch')
        row=db.execute('SELECT record,digest FROM p1_registrations WHERE id=?',(registration_id,)).fetchone()
        if not row or len(row[0])>65536: raise ValueError('missing_registration')
        record=json.loads(row[0]); validate_registration(record)
        check_completion_hello(db,hello,Path(root))
        if (digest(record)!=row[1] or row[1]!=registration_id or record['deployment_id']!=installation['deployment_id'] or
            record['scope']!=installation['host'] or record['evidence']!='actual-host-inspection' or
            record['lane']!=hello.get('lane') or record['boot']!=hello.get('boot') or hello.get('quarantined') is not False):
            raise ValueError('persistent_registration_mismatch')
    return installation['deployment_id']


def archive_names(intent):
    """Fixed control evidence namespace, never caller-selected paths."""
    if not re.fullmatch('[0-9a-f]{32}',intent['archive_id']): raise ValueError('b2_archive_id')
    prefix='p1-recovery-'+intent['archive_id']
    return {prefix+'.sqlite':'snapshot',prefix+'.manifest.json':'manifest',prefix+'.certificate.json':'certificate'}


def archive_manifest(intent):
    output=intent['expected'][1]['output']
    return dict(revision=1,recovery_id=intent['id'],source_digest=intent['source_digest'],
        before_digest=intent['before_digest'],proof_digest=intent['proof_digest'],
        snapshot_digest=output['snapshot_digest'],journal=intent['before']['journal'],journal_state=output['journal_state'],
        journal_digest=output['journal_digest'])


def _certificate(intent,step):
    return dict(deployment_id=intent['deployment_id'],recovery_id=intent['id'],
        plan_digest=intent['plan_digest'],proof_digest=intent['proof_digest'],
        lane=intent['new_lane'],epoch=intent['new_epoch'],certificate_digest=digest(step))


def validate_archive_document(kind,value,intent):
    if kind=='manifest':
        if encode(value)!=encode(archive_manifest(intent)): raise ValueError('b2_archive_manifest_binding')
    elif kind=='certificate':
        expected=intent['expected'][3]
        step=dict(recovery_id=intent['id'],**expected,lane=intent['new_lane'],epoch=intent['new_epoch'])
        if encode(value)!=encode(_certificate(intent,step)): raise ValueError('b2_archive_certificate_binding')
    else: raise ValueError('unknown_b2_archive_document')


def completion_binding(db, root=None):
    from service.persistence import check_recovery_open
    data=check_recovery_open(db, root)
    if data is None: return None
    completions=[s for s in data['p1_recovery_steps'] if s['step']=='core-complete']
    if not completions: return None
    step=completions[-1]
    intent=next(r for r in data['p1_recoveries'] if r['id']==step['recovery_id'])
    return dict(deployment_id=intent['deployment_id'],recovery_id=intent['id'],
        plan_digest=intent['plan_digest'],proof_digest=intent['proof_digest'],
        lane=step['lane'],epoch=step['epoch'],certificate_digest=digest(step))


def check_completion_hello(db,hello,root=None):
    if not isinstance(hello,dict) or len(encode(hello).encode())>65536 or hello.get('quarantined') is not False:
        raise ValueError('b2_worker_hello_refused')
    expected=completion_binding(db,root)
    if hello.get('maintenance')!=expected or (expected is not None and hello.get('lane')!=expected['lane']):
        raise ValueError('b2_cross_volume_completion_mismatch')
    return expected


def startup_read(root):
    """Pre-Core copy, never opens the original SQLite namespace with SQLite."""
    path=Path(root).absolute()/'core.sqlite'
    if not path.exists(): return None
    with private_database(path) as private_path, sqlite3.connect(private_path.as_uri()+'?mode=ro',uri=True,timeout=0) as db:
        db.execute('PRAGMA query_only=ON')
        end=time.monotonic()+2
        db.set_progress_handler(lambda:int(time.monotonic()>end),1000)
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables & (set(B1_COLUMNS)|set(B2_COLUMNS)): return None
        for table,columns in B1_COLUMNS.items():
            if [r[1] for r in db.execute('PRAGMA table_info('+table+')')]!=columns.split(): raise ValueError('partial_enrollment_schema')
        n,b=db.execute('SELECT count(*),coalesce(sum(length(record)),0) FROM p1_installation').fetchone()
        if n!=1 or b>65536: raise ValueError('bounded_enrollment_required')
        identity,raw,h=db.execute('SELECT id,record,digest FROM p1_installation').fetchone()
        value=json.loads(raw)
        closed(value,'deployment_id revision scope host profile_budget_digest'); validate_scope(value['host'])
        if digest(value)!=h or identity!=value['deployment_id'] or value['revision']!=1 or value['scope']!='owned-synthetic': raise ValueError('invalid_enrollment')
        return dict(deployment_id=identity,completion=completion_binding(db,Path(root)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('inspect','plan','apply'))
    parser.add_argument('--request',required=True,help='private bounded structured request file')
    args=parser.parse_args()
    if args.command=='apply': apply()
    request=json.loads(read_file(Path(args.request)))
    closed(request,'roots scope' if args.command=='inspect' else 'roots scope observation')
    result=inspect(request['roots'],request['scope']) if args.command=='inspect' else plan(request['roots'],request['scope'],request['observation'])
    # Detailed private state remains local; no keys, paths or raw Docker receipts.
    print(encode(result))





# B2.1 frozen private records. No transition writer is exposed in this lot.
POLICY = 'abandon-uncertain-synthetic-v1'
STEPS = ('pending', 'evidence', 'journal', 'core-complete', 'receipt-complete')
PLAN_KEYS = ('id deployment_id revision policy source_digest observation_digest proof_digest affected '
             'prior_epoch new_epoch old_lane new_lane journal_state journal_digest budget_before '
             'budget_after diagnostic_decisions count bytes created expires')
RECOVERY_KEYS = ('id deployment_id plan_id plan_digest source_digest observation_digest proof_digest '
                 'proof_kind prior_epoch new_epoch old_lane new_lane affected before_digest monitor_digest '
                 'faults_digest highwater budget_before budget_after actions diagnostic_decisions archive_id expected created before proof')


def _text(value):
    if not isinstance(value,str) or not 1<=len(value)<=256 or not re.fullmatch(r'[A-Za-z0-9_.:-]+',value):
        raise ValueError('b2_identity')


def _uuid(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{32}',value): raise ValueError('b2_exact_identity')


def _hash(value):
    if not isinstance(value,str) or not HEX.fullmatch(value): raise ValueError('b2_digest')


def _integer(value, maximum=MAX_BYTES):
    if type(value)!=int or not 0<=value<=maximum: raise ValueError('b2_integer')


def _time(value):
    import math
    if type(value) not in (int,float) or not math.isfinite(value) or value<0: raise ValueError('b2_time')


def _affected(value):
    closed(value,'jobs dispatches publications reservations')
    count=0
    for kind,items in value.items():
        if not isinstance(items,list) or len(items)>MAX_ROWS: raise ValueError('b2_affected_limit')
        for item in items:
            if kind=='dispatches':
                _integer(item,2**53-1)
                if item==0: raise ValueError('b2_dispatch_id')
            else: _text(item)
        if len(set(items))!=len(items): raise ValueError('b2_duplicate_reference')
        count+=len(items)
    if count>MAX_ROWS: raise ValueError('b2_affected_limit')
    return count


def _budget(value):
    from service.admission import finite_budget
    if not isinstance(value,dict): raise ValueError('b2_budget')
    finite_budget(value)


def _decisions(value):
    if not isinstance(value,list) or len(value)>MAX_ROWS: raise ValueError('b2_decision_limit')
    seen=set()
    for row in value:
        closed(row,'object_id reason content_digest decision availability')
        _hash(row['object_id']); _text(row['reason']); _hash(row['content_digest'])
        if row['decision']!='accept-degraded' or row['availability'] not in ('unavailable','retained') or row['object_id'] in seen:
            raise ValueError('b2_diagnostic_decision')
        seen.add(row['object_id'])


def validate_abandon_plan(value):
    closed(value,PLAN_KEYS)
    for key in ('id','deployment_id','prior_epoch','new_epoch','old_lane','new_lane'): _uuid(value[key])
    for key in ('source_digest','observation_digest','proof_digest','journal_digest'): _hash(value[key])
    if type(value['revision'])!=int or value['revision']!=1 or value['policy']!=POLICY: raise ValueError('b2_policy_revision')
    if value['journal_state'] not in ('valid','missing','unreadable'): raise ValueError('b2_journal_state')
    if value['new_epoch']==value['prior_epoch'] or value['old_lane']==value['new_lane']: raise ValueError('b2_reused_identity')
    _integer(value['count'],MAX_ROWS); _integer(value['bytes']); _time(value['created']); _time(value['expires'])
    if not 0<value['expires']-value['created']<=TTL: raise ValueError('b2_expiry')
    if _affected(value['affected'])!=value['count']: raise ValueError('b2_count')
    _budget(value['budget_before']); _budget(value['budget_after']); _decisions(value['diagnostic_decisions'])


def _evidence_columns():
    return dict(COLUMNS,**B2_COLUMNS,
        p1_inventory_scans='id observed complete fingerprint summary',
        p1_inventory_objects='id first_observed last_observed scan_id record',
        p1_inventory_history='seq scan_id object_id observed record',
        p1_integrity='id revision state token',sqlite_sequence='name seq')


def _scalar(value):
    import math
    if value is not None and (type(value) not in (str,int,float) or
        (type(value)==float and not math.isfinite(value))): raise ValueError('b2_evidence_scalar')


def _validate_source(source):
    closed(source,'schema data inventory journal marker')
    columns=_evidence_columns()
    closed(source['data'],' '.join(columns))
    count=0
    for table,rows in source['data'].items():
        if not isinstance(rows,list): raise ValueError('b2_before_rows')
        count+=len(rows)
        for row in rows:
            if not isinstance(row,list) or len(row)!=len(columns[table].split()): raise ValueError('b2_before_columns')
            for v in row: _scalar(v)
    if not isinstance(source['schema'],list): raise ValueError('b2_before_schema')
    for row in source['schema']:
        if not isinstance(row,list) or len(row)!=4: raise ValueError('b2_before_schema')
        for v in row:_scalar(v)
    if not isinstance(source['inventory'],list): raise ValueError('b2_before_inventory')
    count+=len(source['inventory'])
    for row in source['inventory']:
        closed(row,'id namespace name classification reason identity owner references')
        for key in ('id','namespace','name','classification','reason'):
            if not isinstance(row[key],str) or len(row[key])>256: raise ValueError('b2_before_inventory')
        _hash(row['id'])
        if row['identity'] is not None:
            if not isinstance(row['identity'],list) or len(row['identity'])>16: raise ValueError('b2_before_identity')
            for v in row['identity']:_scalar(v)
        if row['owner'] is not None:
            closed(row['owner'],'id generation owner_id staging_id job_id input_sha256')
            for v in row['owner'].values():_scalar(v)
        if not isinstance(row['references'],list) or len(row['references'])>MAX_ROWS: raise ValueError('b2_before_references')
        for v in row['references']:
            if not isinstance(v,str) or len(v)>256: raise ValueError('b2_before_references')
        count+=len(row['references'])
    _hash(source['journal']);closed(source['marker'],'deployment_id digest')
    _uuid(source['marker']['deployment_id']);_hash(source['marker']['digest'])
    if count>MAX_ROWS or len(encode(source).encode())>MAX_BYTES: raise ValueError('b2_evidence_budget')


CORE_OUTPUT_TABLES=('jobs','resource_reservations','resource_monitor','resource_faults','resource_accounting')


def _validate_core_output(rows,r):
    closed(rows,' '.join(CORE_OUTPUT_TABLES))
    columns=_evidence_columns();before=r['before']['source']['data'];count=0
    for table,values in rows.items():
        if not isinstance(values,list):raise ValueError('b2_output_rows')
        count+=len(values)
        for row in values:
            if not isinstance(row,list) or len(row)!=len(columns[table].split()):raise ValueError('b2_output_columns')
            for v in row:_scalar(v)
    if count>MAX_ROWS:raise ValueError('b2_output_budget')
    if rows['resource_monitor']!=[[1,'healthy']] or rows['resource_faults']:raise ValueError('b2_output_authority')
    if len(rows['jobs'])!=len(before['jobs']) or len(rows['resource_reservations'])!=len(before['resource_reservations']):raise ValueError('b2_output_coverage')
    actions={a['job_id']:a['action'] for a in r['actions']}
    for old,new in zip(before['jobs'],rows['jobs']):
        if old[:6]!=new[:6] or old[7:]!=new[7:]:raise ValueError('b2_output_job_identity')
        job=json.loads(new[6]);prior=json.loads(old[6]);action=actions[old[1]]
        # Validate both complete canonical documents before comparing a fixed
        # unsuccessful transition. Hash agreement alone is not job authority.
        from contracts import validate as oracle
        from jsonschema.exceptions import ValidationError
        import copy
        try:
            context=json.loads(old[7])
            closed(context,'plan bundles')
            for value in (prior,job):
                oracle.validate('job',value)
                oracle.stored_context(value,context)
                if value['attempts']:oracle.active_attempt(value)
        except (ValidationError,KeyError,TypeError,ValueError) as exc:
            raise ValueError('b2_output_job_contract') from exc
        if action=='retain-terminal':
            if old!=new:raise ValueError('b2_output_terminal_history')
        else:
            # Separate validated operator proof (never a PID-1 receipt) supports
            # only terminal state and the current attempt's proven reap bit.
            # Every other field, all prior attempts and the context stay frozen.
            expected=copy.deepcopy(prior)
            expected['state']='failed' if action=='fail' else 'cancelled'
            if expected['attempts']:expected['attempts'][-1]['reaped']=True
            if job!=expected:raise ValueError('b2_output_job_outcome')
    for old,new in zip(before['resource_reservations'],rows['resource_reservations']):
        if old[:3]!=new[:3] or new[3]!='retained':raise ValueError('b2_output_reservation')


def _validate_journal_evidence(evidence,r):
    import hashlib
    closed(evidence,'state bytes_hex')
    state=evidence['state'];raw=evidence['bytes_hex']
    if state not in ('valid','missing','unreadable') or not isinstance(raw,str) or len(raw)>2*1024**2 or not re.fullmatch('(?:[0-9a-f]{2})*',raw):raise ValueError('b2_journal_evidence')
    raw=bytes.fromhex(raw)
    output=r['expected'][1]['output']
    if state!=output['journal_state']:raise ValueError('b2_journal_evidence_state')
    if state=='valid':
        document=json.loads(raw);closed(document,'payload sha256');payload=document['payload']
        closed(payload,'version lane records maintenance' if 'maintenance' in payload else 'version lane records')
        if type(payload['version'])!=int or payload['version']!=1 or payload['lane']!=r['old_lane'] or not isinstance(payload['records'],list) or len(payload['records'])>64 or digest(payload)!=document['sha256']:raise ValueError('b2_journal_evidence_binding')
        from service.worker_transport import Authority
        for row in payload['records']:
            closed(row,'id allocation phase acked uid boot receipt proposal' if 'receipt' in row else 'id allocation phase acked uid boot')
            closed(row['allocation'],'snapshot binding fixture')
            authority=Authority(*row['allocation']['snapshot'])
            if digest(row['allocation']['snapshot'])!=row['allocation']['binding'] or authority.binding!=row['allocation']['binding']:raise ValueError('b2_journal_allocation')
        actual=digest(document)
    elif state=='missing':
        if raw:raise ValueError('b2_missing_journal_bytes')
        actual=hashlib.sha256(raw).hexdigest()
    else:actual=hashlib.sha256(raw).hexdigest()
    if actual!=output['journal_digest']:raise ValueError('b2_journal_evidence_digest')


def _validate_evidence(r):
    # Documents are finite read-only descriptions; no arbitrary patch or path fields.
    if len(encode(r).encode())>MAX_BYTES: raise ValueError('b2_evidence_budget')
    if r['budget_after']!=r['budget_before']:raise ValueError('b2_budget_change_unsupported')
    before=r['before'];closed(before,'source prior_certificate journal')
    source=before['source'];_validate_source(source);data=source['data']
    _validate_journal_evidence(before['journal'],r)
    if digest(source)!=r['source_digest'] or digest(before)!=r['before_digest']:
        raise ValueError('b2_before_binding')
    if (digest(data['resource_monitor'])!=r['monitor_digest'] or digest(data['resource_faults'])!=r['faults_digest'] or
        data['resource_accounting']!=[[1,r['highwater']]]): raise ValueError('b2_monitor_fault_binding')
    prior=before['prior_certificate']
    histories=[json.loads(row[2]) for row in data['p1_recovery_steps'] if row[1]=='core-complete']
    if histories:
        last=histories[-1];old=next(json.loads(row[2]) for row in data['p1_recoveries'] if row[0]==last['recovery_id'])
        if prior!=_certificate(old,last) or prior['epoch']!=r['prior_epoch'] or prior['lane']!=r['old_lane']:
            raise ValueError('b2_prior_certificate')
    elif prior is not None or r['prior_epoch']!=r['deployment_id']: raise ValueError('b2_prior_certificate')
    proof=r['proof'];closed(proof,'kind observation prior_epoch old_lane')
    if (proof['kind']!=r['proof_kind'] or proof['prior_epoch']!=r['prior_epoch'] or proof['old_lane']!=r['old_lane'] or
        digest(proof)!=r['proof_digest'] or digest(proof['observation'])!=r['observation_digest']): raise ValueError('b2_proof_binding')
    registrations=[]
    for _,raw,h in data['p1_registrations']:
        reg=json.loads(raw);validate_registration(reg)
        if digest(reg)!=h: raise ValueError('b2_before_registration')
        registrations.append(reg)
    check_consumers(dict(registrations=registrations,lane=r['old_lane']),proof['observation'])
    jobs={row[1]:json.loads(row[6]) for row in data['jobs']}
    for action in r['actions']:
        job=jobs[action['job_id']]
        expected=('retain-terminal' if job['state'] in ('succeeded','failed','cancelled') else
                  'cancel-never-started' if not job['attempts'] else
                  'cancel-requested' if job['state']=='cancel_requested' else 'fail')
        if action['before_digest']!=digest(job) or action['action']!=expected: raise ValueError('b2_action_before_binding')
    inventory={v['id']:v for v in source['inventory']}
    for decision in r['diagnostic_decisions']:
        row=inventory.get(decision['object_id'])
        if (not row or row['classification'] not in ('missing_hash_mismatched_terminal','adopted_orphan') or
            decision['reason']!=row['reason'] or decision['content_digest']!=diagnostic_digest(row) or
            decision['availability']!=('unavailable' if row['classification']=='missing_hash_mismatched_terminal' else 'retained')):
            raise ValueError('b2_diagnostic_before_binding')
    previous=r['before_digest']
    for descriptor in r['expected']:
        output=descriptor['output'];step=descriptor['step']
        keys={
            'pending':'recovery_id plan_digest proof_digest before_digest',
            'evidence':'archive_id snapshot_digest manifest_digest journal_state journal_digest',
            'journal':'old_lane new_lane journal',
            'core-complete':'before_digest prior_epoch new_epoch actions diagnostic_decisions highwater budget_after monitor_state rows',
            'receipt-complete':'recovery_id lane epoch core_output_digest'}[step]
        closed(output,keys)
        if descriptor['input_digest']!=previous or digest(dict(step=step,input_digest=previous,output=output))!=descriptor['output_digest']: raise ValueError('b2_output_binding')
        if step=='pending': expected={k:r[k] for k in ('plan_digest','proof_digest','before_digest')};expected['recovery_id']=r['id']
        elif step=='evidence':
            for k in ('snapshot_digest','manifest_digest','journal_digest'):_hash(output[k])
            expected=dict(output)
            expected.update(archive_id=r['archive_id'],journal_digest=source['journal'])
            if output['journal_state'] not in ('valid','missing','unreadable'):raise ValueError('b2_evidence_journal')
            if output['manifest_digest']!=digest(archive_manifest(r)):raise ValueError('b2_manifest_digest')
        elif step=='journal':expected=dict(old_lane=r['old_lane'],new_lane=r['new_lane'],journal=dict(version=1,lane=r['new_lane'],records=[]))
        elif step=='core-complete':
            _validate_core_output(output['rows'],r)
            _integer(output['highwater'],2**53-1)
            expected=dict(before_digest=r['before_digest'],prior_epoch=r['prior_epoch'],new_epoch=r['new_epoch'],actions=r['actions'],diagnostic_decisions=r['diagnostic_decisions'],highwater=output['highwater'],budget_after=r['budget_after'],monitor_state='healthy',rows=output['rows'])
            if type(output['highwater'])!=int or output['highwater']<r['highwater'] or output['rows']['resource_accounting']!=[[1,output['highwater']]]:raise ValueError('b2_output_highwater')
        else:expected=dict(recovery_id=r['id'],lane=r['new_lane'],epoch=r['new_epoch'],core_output_digest=r['expected'][3]['output_digest'])
        if output!=expected:raise ValueError('b2_step_output')
        previous=descriptor['output_digest']


@contextmanager
def _archive_database(raw):
    with tempfile.TemporaryDirectory(prefix='p1-evidence-') as directory:
        path=Path(directory)/'snapshot.sqlite'
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:
            if stream.write(raw)!=len(raw):raise OSError('short_evidence_copy')
        db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:yield db
        finally:db.close()


def _validate_archives(root,data,entries,used,uncertified=None,retained_receipts=None):
    import hashlib
    deadline=time.monotonic()+2
    for r in data['p1_recoveries']:
        steps={s['step']:s for s in data['p1_recovery_steps'] if s['recovery_id']==r['id']}
        if 'evidence' not in steps:continue
        content={}
        for name,kind in archive_names(r).items():
            if kind=='certificate' and 'core-complete' not in steps:continue
            # Only the latest unfinished history may have committed Core before
            # its first certificate attempt. Attempted or completed histories
            # retain the unconditional actual-certificate obligation.
            unattempted=(retained_receipts is not None and r is data['p1_recoveries'][-1]
                and 'receipt-complete' not in steps
                and not (retained_receipts/('p1-recovery-'+r['id']+'.core-certificate.intent.json')).exists()
                and not (retained_receipts/('p1-recovery-'+r['id']+'.core-certificate.intent.json')).is_symlink()
                and not (root/name).exists() and not (root/name).is_symlink())
            if kind=='certificate' and (r['id']==uncertified or unattempted):
                # Reserve the exact not-yet-created certificate before COMMIT.
                used+=len(encode(_certificate(r,steps['core-complete'])).encode());entries+=1
                if used>MAX_BYTES or entries>MAX_ROWS:raise ValueError('b2_archive_budget')
                continue
            if (root/name).stat(follow_symlinks=False).st_mode & 0o077:raise ValueError('b2_archive_permissions')
            raw=read_file(root/name,MAX_BYTES-used)
            used+=len(raw);entries+=1
            if used>MAX_BYTES or entries>MAX_ROWS or time.monotonic()>deadline:raise ValueError('b2_archive_budget')
            content[kind]=raw
            if kind!='snapshot':validate_archive_document(kind,json.loads(raw),r)
        output=r['expected'][1]['output']
        if hashlib.sha256(content['snapshot']).hexdigest()!=output['snapshot_digest']:raise ValueError('b2_snapshot_digest')
        if digest(json.loads(content['manifest']))!=output['manifest_digest']:raise ValueError('b2_manifest_digest')
        # SQLite reads a bounded private copy; retained archives are never opened writable.
        with _archive_database(content['snapshot']) as archive:
            archive.execute('PRAGMA trusted_schema=OFF')
            archive.execute('PRAGMA query_only=ON')
            archive.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
            columns=_evidence_columns()
            if {row[0] for row in archive.execute("SELECT name FROM sqlite_master WHERE type='table'")}!=set(columns):raise ValueError('b2_snapshot_schema')
            if archive.execute('PRAGMA user_version').fetchone()!=(1,):raise ValueError('b2_snapshot_version')
            schema=archive.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name').fetchall()
            source=r['before']['source']
            if encode(schema)!=encode(source['schema']):raise ValueError('b2_snapshot_schema')
            for table,cols in columns.items():
                if [v[1] for v in archive.execute('PRAGMA table_info('+table+')')]!=cols.split():raise ValueError('b2_snapshot_columns')
                n,b=archive.execute('SELECT count(*),coalesce(sum('+ '+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in cols.split())+'),0) FROM '+table).fetchone()
                entries+=n
                if entries>MAX_ROWS or b>MAX_BYTES-used or time.monotonic()>deadline:raise ValueError('b2_archive_budget')
                rows=archive.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall()
                if table=='p1_abandon_plans':
                    current=[v for v in rows if v[0]==r['plan_id']]
                    plan=next(p for p in data['p1_abandon_plans'] if p['id']==r['plan_id'])
                    if len(current)!=1 or encode(current[0])!=encode([r['plan_id'],encode(plan),r['plan_digest']]):raise ValueError('b2_snapshot_plan')
                    rows=[v for v in rows if v[0]!=r['plan_id']]
                if encode(rows)!=encode(source['data'][table]):raise ValueError('b2_snapshot_before')


def validate_b2_record(table,value):
    if table=='p1_abandon_plans':
        validate_abandon_plan(value); return
    if table=='p1_recoveries':
        closed(value,RECOVERY_KEYS)
        for key in ('id','deployment_id','plan_id','prior_epoch','new_epoch','old_lane','new_lane','archive_id'): _uuid(value[key])
        archive_names(value)
        for key in ('plan_digest','source_digest','observation_digest','proof_digest','before_digest','monitor_digest','faults_digest'): _hash(value[key])
        if value['proof_kind'] not in ('E1-mock','host-admin-container-teardown-v1'): raise ValueError('b2_proof_kind')
        _affected(value['affected']); _integer(value['highwater'],2**53-1); _time(value['created'])
        _budget(value['budget_before']); _budget(value['budget_after']); _decisions(value['diagnostic_decisions'])
        if not isinstance(value['actions'],list) or len(value['actions'])>MAX_ROWS: raise ValueError('b2_actions')
        seen=set()
        for action in value['actions']:
            closed(action,'job_id action before_digest')
            _text(action['job_id']); _hash(action['before_digest'])
            if action['action'] not in ('retain-terminal','fail','cancel-requested','cancel-never-started') or action['job_id'] in seen: raise ValueError('b2_action')
            seen.add(action['job_id'])
        if seen!=set(value['affected']['jobs']): raise ValueError('b2_action_coverage')
        if not isinstance(value['expected'],list) or [r.get('step') for r in value['expected']]!=list(STEPS): raise ValueError('b2_step_set')
        for previous,current in zip(value['expected'],value['expected'][1:]):
            if previous['output_digest']!=current['input_digest']: raise ValueError('b2_expected_chain')
        for row in value['expected']:
            closed(row,'step input_digest output_digest output'); _hash(row['input_digest']); _hash(row['output_digest'])
        _validate_evidence(value)
        return
    if table=='p1_recovery_steps':
        closed(value,'recovery_id step input_digest output_digest output lane epoch')
        if value['step'] not in STEPS: raise ValueError('b2_step')
        for key in ('input_digest','output_digest'): _hash(value[key])
        for key in ('recovery_id','lane','epoch'): _uuid(value[key])
        return
    if table=='dispatch_resolutions':
        closed(value,'dispatch_seq recovery_id job_id binding original_digest decision')
        _integer(value['dispatch_seq'],2**53-1); _text(value['job_id']); _hash(value['binding']); _hash(value['original_digest'])
        if value['decision']!='abandoned': raise ValueError('b2_resolution')
    elif table=='p1_retired_publications':
        closed(value,'job_id recovery_id attempt_id binding publication_digest work_digest decision')
        for key in ('job_id','attempt_id'): _text(value[key])
        for key in ('binding','publication_digest','work_digest'): _hash(value[key])
        if value['decision']!='retired': raise ValueError('b2_retirement')
    elif table=='p1_integrity_resolutions':
        closed(value,'object_id recovery_id reason content_digest decision availability')
        _decisions([{k:v for k,v in value.items() if k!='recovery_id'}])
    else: raise ValueError('unknown_b2_record')
    _uuid(value['recovery_id'])


def _validate_b2_database(db, root, uncertified=None, retained_receipts=None):
    """Bound the complete logical population before decoding any JSON."""
    from service.persistence import B2_COLUMNS, b2_schema
    if not b2_schema(db): return None
    end=time.monotonic()+2
    expected=dict(COLUMNS,**B2_COLUMNS,
        p1_inventory_scans='id observed complete fingerprint summary',
        p1_inventory_objects='id first_observed last_observed scan_id record',
        p1_inventory_history='seq scan_id object_id observed record',
        p1_integrity='id revision state token',sqlite_sequence='name seq')
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables!=set(expected) or db.execute("SELECT 1 FROM sqlite_master WHERE type IN ('trigger','view') OR (type='index' AND sql IS NOT NULL) LIMIT 1").fetchone(): raise ValueError('b2_unknown_schema')
    count=size=0
    for table in sorted(tables):
        columns=expected[table].split()
        if [r[1] for r in db.execute('PRAGMA table_info('+table+')')]!=columns: raise ValueError('b2_unknown_columns')
        sizes='+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in columns)
        n,b=db.execute('SELECT count(*),coalesce(sum('+sizes+'),0) FROM '+table).fetchone()
        count+=n; size+=b
        if count>MAX_ROWS or size>MAX_BYTES or time.monotonic()>end: raise ValueError('b2_record_budget')
    installation=db.execute('SELECT id,record,digest FROM p1_installation').fetchall()
    if len(installation)!=1: raise ValueError('b2_not_enrolled')
    deployment,raw,h=installation[0]
    if digest(json.loads(raw))!=h: raise ValueError('b2_installation_digest')
    data={}
    for table in B2_COLUMNS:
        data[table]=[]
        if table=='p1_b2_format': continue
        for row in db.execute('SELECT * FROM '+table+' ORDER BY rowid'):
            value=json.loads(row[-2]); validate_b2_record(table,value)
            if encode(value)!=row[-2]: raise ValueError('b2_noncanonical_record')
            if digest(value)!=row[-1]: raise ValueError('b2_record_digest')
            key={'p1_abandon_plans':'id','p1_recoveries':'id','p1_recovery_steps':'recovery_id','dispatch_resolutions':'dispatch_seq','p1_retired_publications':'job_id','p1_integrity_resolutions':'object_id'}[table]
            if value[key]!=row[0]: raise ValueError('b2_row_identity')
            if table not in ('p1_abandon_plans','p1_recoveries') and value['recovery_id']!=row[1 if table!='p1_recovery_steps' else 0]: raise ValueError('b2_recovery_reference')
            if table=='p1_recoveries' and value['plan_id']!=row[1]: raise ValueError('b2_plan_reference')
            if table=='p1_recovery_steps' and value['step']!=row[1]: raise ValueError('b2_step_identity')
            data[table].append(value)
    plans={p['id']:p for p in data['p1_abandon_plans']}
    recoveries={r['id']:r for r in data['p1_recoveries']}
    targets={kind:{r[0] for r in db.execute('SELECT '+column+' FROM '+table)} for kind,table,column in
        (('jobs','jobs','id'),('dispatches','dispatches','seq'),('publications','publications','job_id'),('reservations','resource_reservations','id'))}
    for plan_value in plans.values():
        if plan_value['deployment_id']!=deployment: raise ValueError('b2_deployment_reference')
        for kind,ids in plan_value['affected'].items():
            if not set(ids)<=targets[kind]: raise ValueError('unknown_b2_'+kind+'_reference')
        for decision in plan_value['diagnostic_decisions']:
            if not db.execute('SELECT 1 FROM p1_inventory_objects WHERE id=?',(decision['object_id'],)).fetchone(): raise ValueError('b2_diagnostic_reference')
    for recovery in recoveries.values():
        p=plans.get(recovery['plan_id'])
        if not p or digest(p)!=recovery['plan_digest']: raise ValueError('b2_plan_reference')
        if recovery['expected'][1]['output']['journal_state']!=p['journal_state'] or recovery['expected'][1]['output']['journal_digest']!=p['journal_digest']:raise ValueError('b2_plan_journal_binding')
        for key in ('deployment_id','source_digest','observation_digest','proof_digest','prior_epoch','new_epoch','old_lane','new_lane','affected','budget_before','budget_after','diagnostic_decisions'):
            if recovery[key]!=p[key]: raise ValueError('b2_intent_binding')
    epoch=deployment; lane=None; identities=set()
    for intent in data['p1_recoveries']:
        if intent['prior_epoch']!=epoch or (lane is not None and intent['old_lane']!=lane): raise ValueError('b2_epoch_history')
        if intent['new_epoch'] in identities or intent['new_lane'] in identities: raise ValueError('b2_duplicate_epoch_lane')
        identities.update((intent['new_epoch'],intent['new_lane']))
        if any(s['recovery_id']==intent['id'] and s['step']=='core-complete' for s in data['p1_recovery_steps']):
            epoch=intent['new_epoch']; lane=intent['new_lane']
        elif intent is not data['p1_recoveries'][-1]: raise ValueError('b2_prior_pending_recovery')
    for step in data['p1_recovery_steps']:
        r=recoveries.get(step['recovery_id'])
        if not r: raise ValueError('b2_step_reference')
        expected_step=r['expected'][STEPS.index(step['step'])]
        if any(step[k]!=expected_step[k] for k in ('input_digest','output_digest','output')) or step['lane']!=r['new_lane'] or step['epoch']!=r['new_epoch']: raise ValueError('b2_step_binding')
        preceding=STEPS[:STEPS.index(step['step'])]
        if any(not any(s['recovery_id']==r['id'] and s['step']==name for s in data['p1_recovery_steps']) for name in preceding): raise ValueError('b2_step_chain')
    for table,kind,key in (('dispatch_resolutions','dispatches','dispatch_seq'),('p1_retired_publications','publications','job_id'),('p1_integrity_resolutions',None,'object_id')):
        for resolution in data[table]:
            r=recoveries.get(resolution['recovery_id'])
            if not r or (kind and resolution[key] not in r['affected'][kind]): raise ValueError('b2_resolution_reference')
            if table=='dispatch_resolutions':
                raw=db.execute('SELECT * FROM dispatches WHERE seq=?',(resolution[key],)).fetchone()
                if resolution['job_id']!=raw[1] or resolution['binding']!=raw[3] or resolution['original_digest']!=digest(raw): raise ValueError('b2_dispatch_binding')
            elif table=='p1_retired_publications':
                raw=db.execute('SELECT * FROM publications WHERE job_id=?',(resolution[key],)).fetchone()
                work=db.execute('SELECT * FROM publication_work WHERE job_id=?',(resolution[key],)).fetchone()
                job=json.loads(db.execute('SELECT job FROM jobs WHERE id=?',(resolution[key],)).fetchone()[0])
                if (not work or digest(raw)!=resolution['publication_digest'] or digest(work)!=resolution['work_digest'] or
                    digest(json.loads(work[1]))!=resolution['binding'] or not job['attempts'] or job['attempts'][-1]['id']!=resolution['attempt_id']): raise ValueError('b2_publication_binding')
            elif {k:v for k,v in resolution.items() if k!='recovery_id'} not in r['diagnostic_decisions']: raise ValueError('b2_diagnostic_binding')
    # A completion binds its entire finite resolution set, not merely whatever
    # subset happens to remain in the tables on a later startup.
    for intent in data['p1_recoveries']:
        if any(s['recovery_id']==intent['id'] and s['step']=='core-complete' for s in data['p1_recovery_steps']):
            for table,expected_records in _resolution_outputs(intent).items():
                actual=[r for r in data[table] if r['recovery_id']==intent['id']]
                if actual!=expected_records:raise ValueError('b2_completed_resolution_set')
    # Embedded history and archive populations share the original aggregate ceilings.
    for intent in data['p1_recoveries']:
        source=intent['before']['source']
        count+=sum(len(rows) for rows in source['data'].values())+len(source['inventory'])+len(source['schema'])
        count+=sum(len(row['references']) for row in source['inventory'])+len(intent['actions'])+len(intent['expected'])
        count+=sum(len(rows) for rows in intent['expected'][3]['output']['rows'].values())
        if count>MAX_ROWS:raise ValueError('b2_evidence_budget')
        # Prior immutable rows cannot be omitted or relabelled in a later certificate.
        for table in (*B1_COLUMNS,*B2_COLUMNS):
            if table=='p1_b2_format':continue
            current=db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall()
            for old in source['data'][table]:
                if not any(encode(old)==encode(row) for row in current):raise ValueError('b2_before_history_binding')
    _validate_archives(root,data,count,size,uncertified,retained_receipts)
    return data


def validate_b2_database(db, root=None, *, _uncertified=None, _retained_receipts=None):
    # These owning seams have no live resource observation. Restore no handler;
    # offline snapshot installs its own deadline for its remaining SQL work.
    deadline=time.monotonic()+2
    db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
    try:
        if root is None:
            root=Path(next(r[2] for r in db.execute('PRAGMA database_list') if r[1]=='main')).parent
        return _validate_b2_database(db,Path(root),_uncertified,_retained_receipts)
    except (RecursionError,TypeError,KeyError,IndexError) as exc:
        raise ValueError('b2_malformed_evidence') from exc
    finally: db.set_progress_handler(None,0)


def _b2_state(roots,db,scope,exclude_plan=None):
    return snapshot(roots,db,scope,exclude_abandon=exclude_plan,transaction=False)


def plan_abandon(roots,scope,observation,*,proof_digest,profiles,budget,now=None):
    """Offline proposal only. A supplied digest confers no death/apply authority."""
    from service.persistence import B2_DDL, b2_schema
    _hash(proof_digest)
    now=time.time() if now is None else now; _time(now)
    with offline(roots,write=True) as (roots,db):
        _budget(budget)
        n,b=db.execute('SELECT count(*),coalesce(sum(length(record)),0) FROM p1_installation').fetchone()
        if n!=1 or b>65536: raise ValueError('b2_installation_budget')
        installation=json.loads(db.execute('SELECT record FROM p1_installation').fetchone()[0])
        if installation['profile_budget_digest']!=digest(dict(profiles=profiles,budget=budget)):
            raise ValueError('b2_configuration_mismatch')
        if not b2_schema(db):
            predecessor=snapshot(roots,db,scope); check_consumers(predecessor,observation)
            _coherent_b2(predecessor)
            db.execute('BEGIN IMMEDIATE')
            try:
                for ddl in B2_DDL: db.execute(ddl)
                db.execute('INSERT INTO p1_b2_format VALUES(1,1)'); db.execute('COMMIT')
            except BaseException:
                if db.in_transaction: db.execute('ROLLBACK')
                raise
        from service.persistence import check_recovery_open
        check_recovery_open(db)
        state=_b2_state(roots,db,scope); check_consumers(state,observation); _coherent_b2(state)
        # Enrollment freezes the configuration. B2.1 permits no budget change.
        prior=state['prior_epoch']
        value=dict(id=uuid.uuid4().hex,deployment_id=state['deployment_id'],revision=1,policy=POLICY,
            source_digest=state['source_digest'],observation_digest=digest(observation),proof_digest=proof_digest,
            affected={**state['affected'],'reservations':[r['job_id'] for r in state['reservations']]},
            prior_epoch=prior,new_epoch=uuid.uuid4().hex,old_lane=state['lane'],new_lane=uuid.uuid4().hex,
            journal_state='valid',journal_digest=state['journal_digest'],budget_before=budget,budget_after=budget,
            diagnostic_decisions=[],count=0,bytes=state['bytes'],created=now,expires=now+TTL)
        value['count']=sum(map(len,value['affected'].values())); validate_abandon_plan(value)
        db.execute('INSERT INTO p1_abandon_plans VALUES(?,?,?)',(value['id'],encode(value),digest(value)))
        sync_dir(roots['control']/'core'); sync_dir(roots['control'])
        return dict(plan=value,digest=digest(value),evidence=observation['evidence'])


def _allocation_coherent(data):
    """Check provenance joins directly; inventory reasons can overwrite each other."""
    from service.worker_transport import Authority
    try:
        jobs={r[1]:(json.loads(r[6]),json.loads(r[7])) for r in data['jobs']}
        reservations={r[0]:r for r in data['resource_reservations']}
        if set(jobs)!=set(reservations): return False
        for job_id,(job,context) in jobs.items():
            reservation=reservations[job_id]
            dispatches=[r for r in data['dispatches'] if r[1]==job_id]
            if reservation[3]=='queued':
                if job['attempts'] or dispatches or job['state'] not in ('queued','cancel_requested'): return False
            elif reservation[3]=='executing':
                if not job['attempts'] or not any(not r[9] for r in dispatches): return False
            elif reservation[3]!='retained' or job['state'] not in ('succeeded','failed','cancelled'): return False
            for row in dispatches:
                authority=Authority(*json.loads(row[2]))
                if (authority.binding!=row[3] or authority.job['id']!=job_id or
                    authority.context!=context or authority.job['generation']!=job['generation'] or
                    authority.job['deadline']!=job['deadline'] or not job['attempts'] or
                    authority.job['attempts'][-1]!={**job['attempts'][-1],'reaped':authority.job['attempts'][-1]['reaped']}): return False
                demand=json.loads(reservation[2])
                if (reservation[1]!='inference' or demand['configuration']!=digest(authority.context['plan']['resolved']['identity']['effective']) or
                    demand['disk_bytes']<demand['attempt_disk_bytes']*len(job['attempts'])): return False
                if not row[9] and (reservation[3] not in ('executing','retained') or bool(row[8])!=(reservation[3]=='retained')): return False
                if row[7]:
                    receipt=json.loads(row[7])
                    if (receipt.get('runtime_id')!=row[5] or receipt.get('binding')!=row[3] or
                        receipt.get('original_deadline')!=job['deadline'] or receipt.get('reaped') is not True or
                        receipt.get('reap_method')!='pid1-waitpid-echild'): return False
                elif row[8] or job['state'] in ('succeeded','failed','cancelled'): return False
                for publication in data['publications']:
                    if publication[0]!=job_id: continue
                    artifact=json.loads(publication[2]); attempt=authority.job['attempts'][-1]
                    if any(artifact[k]!=(job['generation'] if k=='generation' else attempt['id' if k=='attempt_id' else k])
                           for k in ('attempt_id','owner_id','staging_id','generation')): return False
        return True
    except (ValueError,KeyError,IndexError,TypeError):
        return False


def _coherent_b2(state):
    if (not state['allocation_coherent'] or
        set(state['inventory']['reasons']) & {'unknown_reference_schema','unknown_reference_kind'} or
        state['inventory']['counts'].get('unknown_foreign',0)):
        raise ValueError('incoherent_b2_predecessor')


def verify_abandon_plan(roots,db,scope,observation,plan_id,expected_digest,*,proof_digest,now=None):
    """Caller owns maintenance locks. Pure read, no lock/transaction acquisition."""
    now=time.time() if now is None else now; _time(now); _hash(proof_digest)
    data=validate_b2_database(db,roots['control']/'core')
    value=next((p for p in data['p1_abandon_plans'] if p['id']==plan_id),None) if data else None
    if value is None: raise ValueError('unknown_b2_plan')
    if any(r['plan_id']==plan_id for r in data['p1_recoveries']): raise ValueError('b2_accepted_plan_requires_resume')
    if digest(value)!=expected_digest: raise ValueError('b2_plan_digest')
    if not value['created']<=now<value['expires']: raise ValueError('expired_b2_plan')
    state=_b2_state(roots,db,scope,exclude_plan=plan_id); check_consumers(state,observation); _coherent_b2(state)
    affected={**state['affected'],'reservations':[r['job_id'] for r in state['reservations']]}
    if (value['source_digest']!=state['source_digest'] or value['affected']!=affected or
        value['observation_digest']!=digest(observation) or value['proof_digest']!=proof_digest): raise ValueError('stale_b2_plan')
    return value


def _prefix_reference_labels(intent):
    return {'p1_recoveries:'+intent['id']} | {
        'p1_recovery_steps:'+intent['id']+':'+name for name in STEPS[:4]}


def _prefix_file(path, raw, fault, point):
    """Exclusive creation only; a partial durable artifact is quarantined on resume."""
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        fault(point+'-create')
        if stream.write(raw)!=len(raw): raise OSError('b2_short_write')
        stream.flush(); fault(point+'-write')
        os.fsync(stream.fileno()); fault(point+'-fsync')
    sync_dir(path.parent); fault(point+'-parent')


def _prefix_exact(path, raw):
    if path.stat(follow_symlinks=False).st_mode & 0o777 != 0o600: raise ValueError('b2_prefix_permissions')
    if read_file(path,MAX_BYTES)!=raw: raise ValueError('b2_prefix_quarantine')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try: os.fsync(fd)
    finally: os.close(fd)
    sync_dir(path.parent)


def _prefix_transaction(db, operation, fault, point):
    # No retry after an ambiguous COMMIT: offline closes the connection on exit.
    db.execute('BEGIN IMMEDIATE')
    try:
        fault(point+'-begin'); operation(); fault(point+'-write')
        db.execute('COMMIT'); fault(point+'-commit')
    except BaseException:
        if db.in_transaction:
            try:db.execute('ROLLBACK')
            except sqlite3.Error:pass  # preserve the original ambiguity; owner closes
        raise


def _prefix_receipt_namespace(receipts, intents, *, retained_names=()):
    allowed={'p1-init.json','p1-core-intent','journal.json','initialized','p1-enrolled.json','lane.lock',
        'p1-b2-required.json','p1-maintenance.json',
        'p1-created-ipc-lifecycle.key','p1-created-ipc-execution.key',
        'p1-created-receipts-journal.json','p1-created-receipts-initialized','p1-created-receipts-lane.lock'}
    for intent in intents:
        prefix='p1-recovery-'+intent['id']
        allowed.update(prefix+suffix for suffix in ('.intent.json','.journal','.certificate.json',
            '.snapshot.intent.json','.manifest.intent.json','.original-journal.intent.json',
            '.snapshot.bundle','.bundle.intent.json','.rotation.intent.json','.core-certificate.intent.json',
            '.receipt-certificate.intent.json','.completion-journal.intent.json','.completion-fence.intent.json'))
    allowed.update(retained_names)
    used=count=0; deadline=time.monotonic()+2
    with os.scandir(receipts) as entries:
        for entry in entries:
            count+=1
            if count>MAX_ROWS or entry.name not in allowed or time.monotonic()>deadline:
                raise ValueError('b2_prefix_receipt_namespace')
            used+=len(read_file(receipts/entry.name,MAX_BYTES-used))
            if used>MAX_BYTES: raise ValueError('b2_prefix_receipt_budget')
    return used,count


def _receipt_only_successor(db, roots, scope, data):
    """Read one receipt-first acceptance, never finish it or scan current media.

    Discovery is bounded before decoding. A filename only selects a candidate;
    the closed record, existing plan, complete SQL before-image and predecessor
    binding authorize its exact reached names, not a recovery-name wildcard.
    """
    import hashlib
    receipts=roots['receipts'];root=roots['control']/'core'
    known={r['id'] for r in data['p1_recoveries']}
    candidates=[];used=count=0;deadline=time.monotonic()+2
    with os.scandir(receipts) as entries:
        for entry in entries:
            count+=1
            if count>MAX_ROWS or time.monotonic()>deadline:raise ValueError('b2_prefix_receipt_budget')
            raw=read_file(receipts/entry.name,MAX_BYTES-used);used+=len(raw)
            match=re.fullmatch(r'p1-recovery-([0-9a-f]{32})\.intent\.json',entry.name)
            if match and match[1] not in known:
                candidates.append((entry.name,raw))
                if len(candidates)>1:raise ValueError('b2_receipt_only_ambiguous')
    if not candidates:return set()
    name,raw=candidates[0];intent=json.loads(raw)
    try:validate_b2_record('p1_recoveries',intent)
    except (RecursionError,TypeError,KeyError,IndexError) as exc:raise ValueError('b2_malformed_evidence') from exc
    prefix='p1-recovery-'+intent['id']
    if name!=prefix+'.intent.json':raise ValueError('b2_receipt_only_identity')
    _retained_exact(receipts/name,encode(intent).encode())
    plan=next((p for p in data['p1_abandon_plans'] if p['id']==intent['plan_id']),None)
    if plan is None or digest(plan)!=intent['plan_digest']:raise ValueError('b2_prefix_plan_changed')
    for key in ('deployment_id','source_digest','observation_digest','proof_digest','prior_epoch','new_epoch','old_lane','new_lane','affected','budget_before','budget_after','diagnostic_decisions'):
        if intent[key]!=plan[key]:raise ValueError('b2_prefix_plan_binding')
    if any(intent['expected'][1]['output']['journal_'+k]!=plan['journal_'+k] for k in ('state','digest')):
        raise ValueError('b2_prefix_plan_journal')
    latest=data['p1_recoveries'][-1]
    if not any(s['recovery_id']==latest['id'] and s['step']=='receipt-complete' for s in data['p1_recovery_steps']):
        raise ValueError('b2_receipt_only_predecessor')
    if intent['before']['prior_certificate']!=_core_result(latest)['certificate']:
        raise ValueError('b2_receipt_only_predecessor')
    identities={r[k] for r in data['p1_recoveries'] for k in ('new_lane','new_epoch')}
    if intent['new_lane'] in identities or intent['new_epoch'] in identities:raise ValueError('b2_duplicate_epoch_lane')
    if intent['proof']['observation']['scope']!=scope:raise ValueError('b2_prefix_proof_changed')
    source=intent['before']['source']
    population=1+sum(len(rows) for rows in source['data'].values())
    for retained in [*data['p1_recoveries'],intent]:
        before=retained['before']['source']
        population+=sum(len(rows) for rows in before['data'].values())+len(before['inventory'])+len(before['schema'])
        population+=sum(len(row['references']) for row in before['inventory'])+len(retained['actions'])+len(retained['expected'])
        population+=sum(len(rows) for rows in retained['expected'][3]['output']['rows'].values())
    if population>MAX_ROWS:raise ValueError('b2_evidence_budget')
    if encode(db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name').fetchall())!=encode(source['schema']):
        raise ValueError('b2_receipt_only_before')
    # The database was already bounded/validated. Only this plan's own row is
    # self-excluded; every previous plan, registration, step and resolution stays.
    for table,rows in source['data'].items():
        if time.monotonic()>deadline:raise ValueError('b2_receipt_only_deadline')
        current=db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall()
        if table=='p1_abandon_plans':current=[r for r in current if r[0]!=plan['id']]
        if encode(current)!=encode(rows):raise ValueError('b2_receipt_only_before')
    private=json.loads(read_file(receipts/'p1-init.json'))
    closed(private,'deployment_id revision scope host profile_budget_digest roots seed')
    installation=json.loads(source['data']['p1_installation'][0][3])
    if ({k:v for k,v in private.items() if k not in ('roots','seed')}!=installation or installation['host']!=scope or
        private['roots']!={k:[p.stat().st_dev,p.stat().st_ino] for k,p in roots.items()}):
        raise ValueError('deployment_scope_mismatch')
    for key in ('lifecycle','execution'):
        if read_file(roots['ipc']/(key+'.key'),64)!=hashlib.sha256((private['seed']+key).encode()).hexdigest().encode():
            raise ValueError('installation_key_missing_or_changed')
    _retained_exact(receipts/'p1-enrolled.json',encode(source['marker']).encode())
    _retained_exact(receipts/'p1-b2-required.json',encode(dict(revision=1,deployment_id=intent['deployment_id'])).encode())
    pending=dict(revision=1,**{k:intent[k] for k in ('deployment_id','plan_id','plan_digest','proof_digest','old_lane','new_lane')},recovery_id=intent['id'],step='pending',certificate=None)
    fence=read_file(receipts/'p1-maintenance.json',65536)
    prior=dict(revision=1,**{k:latest[k] for k in ('deployment_id','plan_id','plan_digest','proof_digest','old_lane','new_lane')},recovery_id=latest['id'],step='complete',certificate=_core_result(latest)['certificate'])
    if fence not in (encode(pending).encode(),encode(prior).encode()):raise ValueError('b2_receipt_only_fence')
    _retained_exact(receipts/'p1-maintenance.json',fence)
    _retained_exact(receipts/'journal.json',bytes.fromhex(intent['before']['journal']['bytes_hex']))
    # Pre-Core can reach only acceptance, fence replacement and bundle creation.
    # Replacement temporaries remain quarantine, even complete, as on B resume.
    names={name};descriptor=receipts/(prefix+'.bundle.intent.json');bundle=receipts/(prefix+'.snapshot.bundle')
    exists=lambda p:p.exists() or p.is_symlink()
    if exists(descriptor):
        if fence!=encode(pending).encode():raise ValueError('b2_retained_creation_prerequisite')
        expected=encode(dict(revision=1,recovery_id=intent['id'],intent_digest=digest(intent),source_digest=intent['source_digest'],snapshot_digest=intent['expected'][1]['output']['snapshot_digest'])).encode()
        _retained_exact(descriptor,expected)
        content=read_file(bundle,MAX_BYTES);_retained_exact(bundle,content)
        if hashlib.sha256(content).hexdigest()!=intent['expected'][1]['output']['snapshot_digest']:raise ValueError('b2_prefix_snapshot_changed')
        # Reuse the actual archive oracle on private evidence copies. No future
        # Core row/artifact is demanded or written in either owned volume.
        with tempfile.TemporaryDirectory(prefix='p1-retained-bundle-') as temporary:
            temporary=Path(temporary)
            for archive,kind in archive_names(intent).items():
                if kind=='certificate':continue
                path=temporary/archive
                path.write_bytes(content if kind=='snapshot' else encode(archive_manifest(intent)).encode());path.chmod(0o600)
            witness=dict(recovery_id=intent['id'],**intent['expected'][1],lane=intent['new_lane'],epoch=intent['new_epoch'])
            _validate_archives(temporary,dict(p1_recoveries=[intent],p1_abandon_plans=[plan],p1_recovery_steps=[witness]),count,used)
        names.update((descriptor.name,bundle.name))
    elif exists(bundle):raise ValueError('b2_prefix_unrecorded_bundle_quarantine')
    if any(exists(root/n) for n in archive_names(intent)):raise ValueError('b2_receipt_only_premature_archive')
    return names


def _retained_exact(path, raw):
    """Protected immutable comparison, with no file creation or re-barrier."""
    if path.stat(follow_symlinks=False).st_mode & 0o777!=0o600:raise ValueError('b2_prefix_permissions')
    if read_file(path,MAX_BYTES)!=raw:raise ValueError('b2_prefix_quarantine')


def _pending_through_evidence(roots, scope, intent, observation, *, now=None, snapshot_bytes=None, fault=lambda point:None):
    return _pending_prefix(roots,scope,intent,observation,now=now,snapshot_bytes=snapshot_bytes,fault=fault)


def _pending_through_journal(roots, scope, intent, observation, *, now=None, fault=lambda point:None):
    """Private fixed journal continuation; never grants completion or activation."""
    return _pending_prefix(roots,scope,intent,observation,now=now,fault=fault,through_journal=True)


def _pending_through_core(roots, scope, intent, observation, *, now=None, fault=lambda point:None):
    """Private B2.2c continuation; receipt authority stays PENDING."""
    return _pending_prefix(roots,scope,intent,observation,now=now,fault=fault,through_journal=True,through_core=True)


def _complete_receipts(roots, scope, intent, observation, *, now=None, fault=lambda point:None):
    """Private final continuation; public apply and activation remain unavailable."""
    return _pending_prefix(roots,scope,intent,observation,now=now,fault=fault,
        through_journal=True,through_core=True,through_receipt=True)


def _pending_prefix(roots, scope, intent, observation, *, now=None, snapshot_bytes=None, fault=lambda point:None, through_journal=False, through_core=False, through_receipt=False):
    """Private bounded prefix. Supplied proof is NOT host authentication.

    Both normal locks cover the named continuation. Completion is private;
    public apply and enrolled activation remain unavailable.
    """
    import hashlib
    intent=json.loads(encode(intent))
    validate_b2_record('p1_recoveries',intent)
    now=time.time() if now is None else now; _time(now)
    validate_scope(scope)
    if observation!=intent['proof']['observation'] or observation['scope']!=scope: raise ValueError('b2_prefix_proof_changed')
    with offline(roots,write=True) as (roots,db):
        root=roots['control']/'core'; receipts=roots['receipts']
        data=validate_b2_database(db,root,_uncertified=intent['id'] if through_core else None,_retained_receipts=receipts)
        plan=next((p for p in data['p1_abandon_plans'] if p['id']==intent['plan_id']),None) if data else None
        if plan is None or digest(plan)!=intent['plan_digest']: raise ValueError('b2_prefix_plan_changed')
        for key in ('deployment_id','source_digest','observation_digest','proof_digest','prior_epoch','new_epoch','old_lane','new_lane','affected','budget_before','budget_after','diagnostic_decisions'):
            if intent[key]!=plan[key]: raise ValueError('b2_prefix_plan_binding')
        if intent['expected'][1]['output']['journal_state']!=plan['journal_state'] or intent['expected'][1]['output']['journal_digest']!=plan['journal_digest']:
            raise ValueError('b2_prefix_plan_journal')
        existing=next((r for r in data['p1_recoveries'] if r['id']==intent['id']),None)
        if existing is not None and existing!=intent: raise ValueError('b2_prefix_intent_changed')
        for prior in data['p1_recoveries']:
            if prior['id']!=intent['id'] and any(s['recovery_id']==prior['id'] and s['step']=='receipt-complete' for s in data['p1_recovery_steps']):
                _completed_prefix_receipts(roots,prior)
                _verify_finished_receipt_evidence(roots,prior)
        completed=any(r['recovery_id']==intent['id'] and r['step']=='core-complete' for r in data['p1_recovery_steps'])
        certificate_path=next(root/n for n,k in archive_names(intent).items() if k=='certificate')
        creation=receipts/('p1-recovery-'+intent['id']+'.core-certificate.intent.json')
        if through_core and completed and (certificate_path.exists() or creation.exists()):
            # Immutable completion is separate from current startup eligibility.
            # Later ordinary rows/journal activity must never replay this epoch.
            retained_names=_receipt_only_successor(db,roots,scope,data)
            _prefix_receipt_namespace(receipts,data['p1_recoveries'],retained_names=retained_names)
            for prior in data['p1_recoveries']:
                if prior['id']==intent['id']:continue
                stages={s['step'] for s in data['p1_recovery_steps'] if s['recovery_id']==prior['id']}
                # A successor may still be pending. Its declared durable prefix
                # requires its actual evidence, never a future receipt artifact.
                if data['p1_recoveries'].index(prior)>data['p1_recoveries'].index(intent) and 'receipt-complete' not in stages:
                    _retained_pending_receipts(roots,prior,stages)
                else:
                    chain=[dict(recovery_id=prior['id'],**e,lane=prior['new_lane'],epoch=prior['new_epoch']) for e in prior['expected'][:4]]
                    _prefix_exact(receipts/('p1-recovery-'+prior['id']+'.certificate.json'),encode(dict(certificate=_certificate(prior,chain[-1]),steps=chain)).encode())
                    _prefix_exact(receipts/('p1-recovery-'+prior['id']+'.journal'),bytes.fromhex(prior['before']['journal']['bytes_hex']))
            if through_receipt and not any(s['recovery_id']==intent['id'] and s['step']=='receipt-complete' for s in data['p1_recovery_steps']):
                # Core completion alone is not an immutable completed replay.
                # Both locks exclude concurrent owners, but do not prove that a
                # registered incarnation stayed stopped between invocations.
                registrations=db.execute('SELECT * FROM p1_registrations ORDER BY rowid').fetchall()
                if encode(registrations)!=encode(intent['before']['source']['data']['p1_registrations']):
                    raise ValueError('b2_prefix_registrations_changed')
                current=[json.loads(row[1]) for row in registrations]
                for registration in current:validate_registration(registration)
                check_consumers(dict(registrations=current,lane=intent['old_lane']),observation)
            _completed_prefix_receipts(roots,intent)
            if through_receipt and any(s['recovery_id']==intent['id'] and s['step']=='receipt-complete' for s in data['p1_recovery_steps']):
                _verify_finished_receipt_evidence(roots,intent)
                if data['p1_recoveries'][-1]['id']==intent['id'] and not retained_names:
                    fence=dict(revision=1,**{k:intent[k] for k in ('deployment_id','plan_id','plan_digest','proof_digest','old_lane','new_lane')},recovery_id=intent['id'],step='complete',certificate=_core_result(intent)['certificate'])
                    _retained_exact(receipts/'p1-maintenance.json',encode(fence).encode())
                return dict(_core_result(intent),status='complete')
            _core_certificate(root,receipts,intent,fault)
            validate_b2_database(db,root,_retained_receipts=receipts)
            return _finish_receipts(db,roots,intent,fault) if through_receipt else _core_result(intent)
        if any(r['id']!=intent['id'] and not any(s['recovery_id']==r['id'] and s['step']=='core-complete' for s in data['p1_recovery_steps']) for r in data['p1_recoveries']):
            raise ValueError('b2_prefix_other_pending')
        steps=[s for s in data['p1_recovery_steps'] if s['recovery_id']==intent['id']]
        if any(s['step'] not in STEPS[:4 if through_core else 3 if through_journal else 2] for s in steps): raise ValueError('b2_prefix_already_advanced')
        marker=dict(revision=1,deployment_id=intent['deployment_id'])
        fence=dict(revision=1,**{k:intent[k] for k in ('deployment_id','plan_id','plan_digest','proof_digest','old_lane','new_lane')},recovery_id=intent['id'],step='pending',certificate=None)
        from worker.runtime_journal import maintenance_record
        maintenance_record(fence)
        marker_path=receipts/'p1-b2-required.json'; fence_path=receipts/'p1-maintenance.json'
        accepted_path=receipts/('p1-recovery-'+intent['id']+'.intent.json')
        marker_raw=encode(marker).encode(); fence_raw=encode(fence).encode(); accepted_raw=encode(intent).encode()
        _prefix_receipt_namespace(receipts,[*data['p1_recoveries'],intent])
        # All completed predecessors remain immutable even after their fence is
        # superseded. Derive exact receipt documents from validated SQL history.
        for prior in data['p1_recoveries']:
            if prior['id']==intent['id']: continue
            chain=[dict(recovery_id=prior['id'],**e,lane=prior['new_lane'],epoch=prior['new_epoch']) for e in prior['expected'][:4]]
            certificate=_certificate(prior,chain[-1])
            prefix='p1-recovery-'+prior['id']
            certificate_raw=encode(dict(certificate=certificate,steps=chain)).encode()
            if len(certificate_raw)>65536: raise ValueError('b2_prefix_receipt_budget')
            _prefix_exact(receipts/(prefix+'.certificate.json'),certificate_raw)
            _prefix_exact(receipts/(prefix+'.journal'),bytes.fromhex(prior['before']['journal']['bytes_hex']))
        accepted=accepted_path.exists()
        if existing is not None and not fence_path.exists(): raise ValueError('b2_prefix_receipt_quarantine')
        previous_fence=False
        if fence_path.exists() and read_file(fence_path,65536)!=fence_raw:
            from worker.runtime_journal import read_maintenance
            prior=read_maintenance(receipts)
            if existing is not None or prior is None or prior!=intent['before']['prior_certificate'] or prior!=completion_binding(db,root):
                raise ValueError('b2_prefix_previous_fence')
            previous_fence=True
        if accepted:
            _prefix_exact(marker_path,marker_raw); _prefix_exact(accepted_path,accepted_raw)
            if fence_path.exists() and not previous_fence: _prefix_exact(fence_path,fence_raw)
        elif existing is not None or ((marker_path.exists() or fence_path.exists()) and not previous_fence):
            raise ValueError('b2_prefix_receipt_quarantine')
        if not accepted and not plan['created']<=now<plan['expires']: raise ValueError('expired_b2_plan')
        # Exact own rows and derived archive labels only are removed from this
        # projection. Every older plan/history, media identity and namespace stays.
        rotation=receipts/('p1-recovery-'+intent['id']+'.rotation.intent.json')
        from worker.runtime_journal import journal_bytes
        new_raw=journal_bytes(intent['expected'][2]['output']['journal'])
        rotation_raw=encode(dict(revision=1,recovery_id=intent['id'],intent_digest=digest(intent),
            step=intent['expected'][2],before=intent['before']['journal'],sha256=hashlib.sha256(new_raw).hexdigest())).encode()
        journal_after=False
        if rotation.exists():
            if not through_journal: raise ValueError('b2_prefix_already_advanced')
            _prefix_exact(rotation,rotation_raw)
            # A recorded attempt may only resume its exact output, never retry
            # replacement from an absent/old/partial output.
            _prefix_exact(receipts/'journal.json',new_raw)
            journal_after=True
        elif any(s['step']=='journal' for s in steps): raise ValueError('b2_missing_rotation_intent')
        core_after=any(s['step']=='core-complete' for s in steps)
        state=snapshot(roots,db,scope,exclude_abandon=plan['id'],transaction=False,_prefix=intent,_journal_after=journal_after,_core_after=core_after)
        check_consumers(state,observation); _coherent_b2(state)
        if encode(state['evidence_projection'])!=encode(intent['before']['source']): raise ValueError('b2_prefix_prestate_changed')
        if {**state['affected'],'reservations':[r['job_id'] for r in state['reservations']]}!=plan['affected']:
            raise ValueError('b2_prefix_affected_changed')
        if not journal_after and intent['before']['journal']['state']!='missing' and read_file(receipts/'journal.json',1024**2).hex()!=intent['before']['journal']['bytes_hex']:
            raise ValueError('b2_prefix_journal_changed')
        names={kind:root/name for name,kind in archive_names(intent).items()}
        if (names['certificate'].exists() and not core_after) or (receipts/('p1-recovery-'+intent['id']+'.certificate.json')).exists(): raise ValueError('b2_prefix_premature_certificate')
        if core_after:
            _completed_prefix_receipts(roots,intent)
            _core_certificate(root,receipts,intent,fault)
            validate_b2_database(db,root,_retained_receipts=receipts)
            return _finish_receipts(db,roots,intent,fault) if through_receipt else _core_result(intent)
        bundle=receipts/('p1-recovery-'+intent['id']+'.snapshot.bundle')
        bundle_intent=receipts/('p1-recovery-'+intent['id']+'.bundle.intent.json')
        bundle_descriptor=encode(dict(revision=1,recovery_id=intent['id'],intent_digest=digest(intent),
            source_digest=intent['source_digest'],snapshot_digest=intent['expected'][1]['output']['snapshot_digest'])).encode()
        snapshot_raw=snapshot_bytes
        if bundle_intent.exists():
            _prefix_exact(bundle_intent,bundle_descriptor)
            if not bundle.exists(): raise ValueError('b2_prefix_missing_bundle_quarantine')
            snapshot_raw=read_file(bundle,MAX_BYTES)
            _prefix_exact(bundle,snapshot_raw)
            if snapshot_bytes is not None and snapshot_bytes!=snapshot_raw: raise ValueError('b2_prefix_snapshot_changed')
        elif bundle.exists() or bundle.is_symlink() or existing is not None:
            raise ValueError('b2_prefix_unrecorded_bundle_quarantine')
        if snapshot_raw is not None:
            if not isinstance(snapshot_raw,bytes) or len(snapshot_raw)>MAX_BYTES or hashlib.sha256(snapshot_raw).hexdigest()!=intent['expected'][1]['output']['snapshot_digest']:
                raise ValueError('b2_prefix_snapshot_changed')
        if not accepted or (existing is None and snapshot_raw is None):
            if any(p.exists() or p.is_symlink() for p in names.values()): raise ValueError('b2_prefix_foreign_archive')
            # backup incorporates committed WAL into a bounded private snapshot;
            # source is never restored or copied over the live authority database.
            with tempfile.TemporaryDirectory(prefix='p1-evidence-') as temporary:
                path=Path(temporary)/'snapshot.sqlite'
                deadline=time.monotonic()+2
                def progress(status,remaining,total):
                    if total*db.execute('PRAGMA page_size').fetchone()[0]>MAX_BYTES or time.monotonic()>deadline:
                        raise ValueError('b2_prefix_snapshot_budget')
                with private_database(root/'core.sqlite') as private_path:
                    with sqlite3.connect(private_path.as_uri()+'?mode=ro',uri=True) as source_db, sqlite3.connect(path) as target:
                        source_db.backup(target,pages=64,progress=progress)
                snapshot_raw=read_file(path,MAX_BYTES)
                if hashlib.sha256(snapshot_raw).hexdigest()!=intent['expected'][1]['output']['snapshot_digest']:
                    raise ValueError('b2_prefix_snapshot_changed')
                (Path(temporary)/names['snapshot'].name).write_bytes(snapshot_raw)
                manifest=Path(temporary)/names['manifest'].name
                manifest.write_text(encode(archive_manifest(intent))); manifest.chmod(0o600)
                (Path(temporary)/names['snapshot'].name).chmod(0o600)
                witness=dict(recovery_id=intent['id'],**intent['expected'][1],lane=intent['new_lane'],epoch=intent['new_epoch'])
                _validate_archives(Path(temporary),dict(p1_recoveries=[intent],p1_abandon_plans=[plan],p1_recovery_steps=[witness]),0,0)
                with sqlite3.connect(path) as candidate:
                    candidate.execute('INSERT INTO p1_recoveries VALUES(?,?,?,?)',(intent['id'],intent['plan_id'],encode(intent),digest(intent)))
                    for descriptor in intent['expected'][:2]:
                        step=dict(recovery_id=intent['id'],**descriptor,lane=intent['new_lane'],epoch=intent['new_epoch'])
                        candidate.execute('INSERT INTO p1_recovery_steps VALUES(?,?,?,?)',(intent['id'],step['step'],encode(step),digest(step)))
                    candidate.commit()
                    used=len(snapshot_raw)+len(encode(intent).encode())
                    for prior in data['p1_recoveries']:
                        for name in archive_names(prior):
                            raw=read_file(root/name,MAX_BYTES-used); used+=len(raw)
                            target=Path(temporary)/name; target.write_bytes(raw); target.chmod(0o600)
                    validate_b2_database(candidate,Path(temporary))
        if not accepted:
            if marker_path.exists(): _prefix_exact(marker_path,marker_raw)
            else: _prefix_file(marker_path,marker_raw,fault,'marker')
            _prefix_file(accepted_path,accepted_raw,fault,'accepted')
        if snapshot_raw is not None:
            with tempfile.TemporaryDirectory(prefix='p1-evidence-check-') as temporary:
                for kind,raw in (('snapshot',snapshot_raw),('manifest',encode(archive_manifest(intent)).encode())):
                    path=Path(temporary)/names[kind].name
                    path.write_bytes(raw); path.chmod(0o600)
                witness=dict(recovery_id=intent['id'],**intent['expected'][1],lane=intent['new_lane'],epoch=intent['new_epoch'])
                _validate_archives(Path(temporary),dict(p1_recoveries=[intent],p1_abandon_plans=[plan],p1_recovery_steps=[witness]),0,0)
        if previous_fence:
            pending=receipts/'p1-maintenance.pending.json'
            _prefix_file(pending,fence_raw,fault,'fence')
            os.replace(pending,fence_path); sync_dir(receipts); fault('fence-replace-parent')
        elif not fence_path.exists(): _prefix_file(fence_path,fence_raw,fault,'fence')
        if not bundle_intent.exists():
            if snapshot_raw is None: raise ValueError('b2_prefix_missing_bundle_quarantine')
            # Immutable creation intent precedes the exact original bytes. Once
            # recorded, missing/partial output quarantines; never recapture it.
            _prefix_file(bundle_intent,bundle_descriptor,fault,'bundle-intent')
            fault('bundle-intent-recorded')
            _prefix_file(bundle,snapshot_raw,fault,'bundle')
        _prefix_exact(bundle_intent,bundle_descriptor)
        _prefix_exact(bundle,snapshot_raw)
        # Charge the new durable restart input against the receipt population.
        _prefix_receipt_namespace(receipts,[*data['p1_recoveries'],intent])
        if existing is None:
            _prefix_transaction(db,lambda:db.execute('INSERT INTO p1_recoveries VALUES(?,?,?,?)',
                (intent['id'],intent['plan_id'],encode(intent),digest(intent))),fault,'core-intent')
            sync_dir(root); fault('core-intent-parent')
        def record(index):
            descriptor=intent['expected'][index]
            step=dict(recovery_id=intent['id'],**descriptor,lane=intent['new_lane'],epoch=intent['new_epoch'])
            row=db.execute('SELECT record,digest FROM p1_recovery_steps WHERE recovery_id=? AND step=?',(intent['id'],step['step'])).fetchone()
            if row is not None:
                if row!=(encode(step),digest(step)): raise ValueError('b2_prefix_step_changed')
                return
            _prefix_transaction(db,lambda:db.execute('INSERT INTO p1_recovery_steps VALUES(?,?,?,?)',
                (intent['id'],step['step'],encode(step),digest(step))),fault,step['step']+'-receipt')
            sync_dir(root); fault(step['step']+'-receipt-parent')
        record(0)
        outputs=[(names['snapshot'],snapshot_raw,'snapshot'),
                 (names['manifest'],encode(archive_manifest(intent)).encode(),'manifest'),
                 (receipts/('p1-recovery-'+intent['id']+'.journal'),bytes.fromhex(intent['before']['journal']['bytes_hex']),'original-journal')]
        for path,raw,kind in outputs:
            creation=receipts/('p1-recovery-'+intent['id']+'.'+kind+'.intent.json')
            expected_hash=intent['expected'][1]['output']['snapshot_digest'] if kind=='snapshot' else hashlib.sha256(raw).hexdigest()
            descriptor=encode(dict(revision=1,recovery_id=intent['id'],kind=kind,sha256=expected_hash)).encode()
            if creation.exists():
                _prefix_exact(creation,descriptor)
                if not path.exists(): raise ValueError('b2_prefix_missing_artifact_quarantine')
            else:
                if path.exists() or path.is_symlink(): raise ValueError('b2_prefix_unrecorded_artifact')
                if raw is None: raise ValueError('b2_prefix_missing_snapshot_quarantine')
                _prefix_file(creation,descriptor,fault,kind+'-intent')
                fault(kind+'-intent-recorded')
                _prefix_file(path,raw,fault,kind)
            actual=read_file(path,MAX_BYTES)
            if hashlib.sha256(actual).hexdigest()!=expected_hash: raise ValueError('b2_prefix_artifact_quarantine')
            _prefix_exact(path,actual)
        witness=dict(recovery_id=intent['id'],**intent['expected'][1],lane=intent['new_lane'],epoch=intent['new_epoch'])
        _validate_archives(root,dict(p1_recoveries=[intent],p1_abandon_plans=[plan],p1_recovery_steps=[witness]),0,0)
        record(1)
        validate_b2_database(db,root)
        if through_journal:
            if not journal_after:
                used,count=_prefix_receipt_namespace(receipts,[*data['p1_recoveries'],intent])
                # Include the replacement temporary and immutable descriptor in
                # the same finite receipt population before attempting creation.
                if count+2>MAX_ROWS or used+len(rotation_raw)+len(new_raw)>MAX_BYTES:
                    raise ValueError('b2_prefix_receipt_budget')
                _prefix_file(rotation,rotation_raw,fault,'journal-intent')
                fault('journal-intent-recorded')
                from worker.runtime_journal import _maintenance_replace
                _maintenance_replace(receipts,intent['expected'][2]['output']['journal'],fault=fault)
            _prefix_exact(receipts/'journal.json',new_raw)
            record(2)
            validate_b2_database(db,root)
        if through_core:
            _complete_core(db,roots,intent,fault)
            return _finish_receipts(db,roots,intent,fault) if through_receipt else _core_result(intent)
        return dict(status='pending-through-journal' if through_journal else 'pending-through-evidence',recovery_id=intent['id'],lane=intent['new_lane'],epoch=intent['new_epoch'])


RESOLUTION_TABLES=('dispatch_resolutions','p1_retired_publications','p1_integrity_resolutions')


def diagnostic_digest(row):
    # Typed references remain retained independently; adding an audit reference
    # does not change the original defect's tuple, bytes or availability.
    return digest({k:v for k,v in row.items() if k!='references'})


def _resolution_outputs(intent):
    before=intent['before']['source']['data'];rid=intent['id']
    dispatches=[dict(dispatch_seq=r[0],recovery_id=rid,job_id=r[1],binding=r[3],original_digest=digest(r),decision='abandoned')
        for r in before['dispatches'] if r[0] in intent['affected']['dispatches']]
    work={r[0]:r for r in before['publication_work']};jobs={r[1]:json.loads(r[6]) for r in before['jobs']}
    publications=[dict(job_id=r[0],recovery_id=rid,attempt_id=jobs[r[0]]['attempts'][-1]['id'],
        binding=digest(json.loads(work[r[0]][1])),publication_digest=digest(r),work_digest=digest(work[r[0]]),decision='retired')
        for r in before['publications'] if r[0] in intent['affected']['publications']]
    diagnostics=[dict(**r,recovery_id=rid) for r in intent['diagnostic_decisions']]
    return dict(zip(RESOLUTION_TABLES,(dispatches,publications,diagnostics)))


def _verify_core_rows(db,intent):
    # Explicit closed after-image; every unaffected table is compared by snapshot.
    for table,rows in intent['expected'][3]['output']['rows'].items():
        if encode(db.execute('SELECT * FROM '+table+' ORDER BY rowid').fetchall())!=encode(rows):raise ValueError('b2_core_output_changed')
    if db.execute('SELECT * FROM p1_integrity').fetchall()!=[(1,1,'healthy',None)]:raise ValueError('b2_core_integrity_changed')
    for table,records in _resolution_outputs(intent).items():
        expected=[[r.get('dispatch_seq',r.get('job_id',r.get('object_id'))),intent['id'],encode(r),digest(r)] for r in records]
        actual=db.execute('SELECT * FROM '+table+' WHERE recovery_id=? ORDER BY rowid',(intent['id'],)).fetchall()
        if encode(actual)!=encode(expected):raise ValueError('b2_core_resolution_changed')


def _core_result(intent):
    step=dict(recovery_id=intent['id'],**intent['expected'][3],lane=intent['new_lane'],epoch=intent['new_epoch'])
    return dict(status='core-complete-receipt-pending',recovery_id=intent['id'],lane=intent['new_lane'],epoch=intent['new_epoch'],certificate=_certificate(intent,step))


def _completed_prefix_receipts(roots,intent, *, _stages=None):
    """Verify immutable owned inputs without another inventory/resource scan."""
    import hashlib
    receipts=roots['receipts'];root=roots['control']/'core';prefix='p1-recovery-'+intent['id']
    private=json.loads(read_file(receipts/'p1-init.json'))
    closed(private,'deployment_id revision scope host profile_budget_digest roots seed')
    installation=json.loads(intent['before']['source']['data']['p1_installation'][0][3])
    if {k:v for k,v in private.items() if k not in ('roots','seed')}!=installation:raise ValueError('deployment_scope_mismatch')
    _prefix_exact(receipts/(prefix+'.intent.json'),encode(intent).encode())
    _prefix_exact(receipts/'p1-b2-required.json',encode(dict(revision=1,deployment_id=intent['deployment_id'])).encode())
    if json.loads(read_file(receipts/'p1-enrolled.json'))!=intent['before']['source']['marker']:raise ValueError('incomplete_enrollment')
    from worker.runtime_journal import journal_bytes
    rotation=encode(dict(revision=1,recovery_id=intent['id'],intent_digest=digest(intent),step=intent['expected'][2],before=intent['before']['journal'],sha256=hashlib.sha256(journal_bytes(intent['expected'][2]['output']['journal'])).hexdigest())).encode()
    if _stages is None or 'journal' in _stages or (receipts/(prefix+'.rotation.intent.json')).exists():
        _prefix_exact(receipts/(prefix+'.rotation.intent.json'),rotation)
    if private['roots']!={k:[p.stat().st_dev,p.stat().st_ino] for k,p in roots.items()}:raise ValueError('volume_role_identity_mismatch')
    for key in ('lifecycle','execution'):
        if read_file(roots['ipc']/(key+'.key'),64)!=hashlib.sha256((private['seed']+key).encode()).hexdigest().encode():raise ValueError('installation_key_missing_or_changed')
    raw=read_file(receipts/(prefix+'.snapshot.bundle'),MAX_BYTES)
    descriptor=encode(dict(revision=1,recovery_id=intent['id'],intent_digest=digest(intent),source_digest=intent['source_digest'],snapshot_digest=intent['expected'][1]['output']['snapshot_digest'])).encode()
    _prefix_exact(receipts/(prefix+'.bundle.intent.json'),descriptor)
    if hashlib.sha256(raw).hexdigest()!=intent['expected'][1]['output']['snapshot_digest']:raise ValueError('b2_prefix_snapshot_changed')
    _prefix_exact(receipts/(prefix+'.snapshot.bundle'),raw)
    names={kind:root/name for name,kind in archive_names(intent).items()}
    for path,content,kind in ((names['snapshot'],raw,'snapshot'),(names['manifest'],encode(archive_manifest(intent)).encode(),'manifest'),(receipts/(prefix+'.journal'),bytes.fromhex(intent['before']['journal']['bytes_hex']),'original-journal')):
        creation=encode(dict(revision=1,recovery_id=intent['id'],kind=kind,sha256=hashlib.sha256(content).hexdigest())).encode()
        if _stages is not None and 'evidence' not in _stages and not (receipts/(prefix+'.'+kind+'.intent.json')).exists():
            if path.exists() or path.is_symlink():raise ValueError('b2_prefix_unrecorded_artifact')
            continue
        _prefix_exact(receipts/(prefix+'.'+kind+'.intent.json'),creation)
        _prefix_exact(path,content)


def _retained_pending_receipts(roots,intent,stages):
    """Validate retained production inputs at their actual durable stage."""
    from worker.runtime_journal import journal_bytes
    receipts=roots['receipts'];prefix='p1-recovery-'+intent['id']
    def attempted(kind):
        path=receipts/(prefix+'.'+kind+'.intent.json')
        return path.exists() or path.is_symlink()
    # Closed production order: an output may precede its OWN SQL receipt,
    # but a later attempt proves that every predecessor receipt/output existed.
    evidence=('snapshot','manifest','original-journal')
    for index,kind in enumerate(evidence):
        if attempted(kind) and ('pending' not in stages or any(not attempted(k) for k in evidence[:index])):
            raise ValueError('b2_retained_creation_prerequisite')
    if 'evidence' in stages and not all(attempted(k) for k in evidence):
        raise ValueError('b2_retained_creation_prerequisite')
    if attempted('rotation') and 'evidence' not in stages:
        raise ValueError('b2_retained_creation_prerequisite')
    if attempted('core-certificate') and 'core-complete' not in stages:
        raise ValueError('b2_retained_creation_prerequisite')
    if attempted('receipt-certificate') and ('core-complete' not in stages or not attempted('core-certificate')):
        raise ValueError('b2_retained_creation_prerequisite')
    _completed_prefix_receipts(roots,intent,_stages=stages)
    certificate=_core_result(intent)['certificate']
    raw=encode(certificate).encode()
    creation=receipts/(prefix+'.core-certificate.intent.json')
    if attempted('core-certificate'):
        _prefix_exact(creation,_receipt_descriptor(intent,raw))
        name=next(n for n,k in archive_names(intent).items() if k=='certificate')
        _prefix_exact(roots['control']/'core'/name,raw)
    else:
        name=next(n for n,k in archive_names(intent).items() if k=='certificate')
        path=roots['control']/'core'/name
        if path.exists() or path.is_symlink():raise ValueError('b2_core_certificate_unrecorded')
    chain=[dict(recovery_id=intent['id'],**e,lane=intent['new_lane'],epoch=intent['new_epoch']) for e in intent['expected'][:4]]
    archive=encode(dict(certificate=certificate,steps=chain)).encode()
    fence=dict(revision=1,**{k:intent[k] for k in ('deployment_id','plan_id','plan_digest','proof_digest','old_lane','new_lane')},recovery_id=intent['id'],step='complete',certificate=certificate)
    final_attempts=[]
    for kind,raw,path in (
        ('receipt-certificate',archive,receipts/(prefix+'.certificate.json')),
        ('completion-journal',journal_bytes(dict(intent['expected'][2]['output']['journal'],maintenance=certificate)),receipts/'journal.json'),
        ('completion-fence',encode(fence).encode(),receipts/'p1-maintenance.json')):
        creation=receipts/(prefix+'.'+kind+'.intent.json')
        final_attempts.append(creation.exists() or creation.is_symlink())
        if creation.exists() or creation.is_symlink():
            if 'core-complete' not in stages:raise ValueError('b2_prefix_premature_certificate')
            _prefix_exact(creation,_receipt_descriptor(intent,raw));_prefix_exact(path,raw)
        elif kind=='receipt-certificate' and (path.exists() or path.is_symlink()):
            raise ValueError('b2_receipt_unrecorded_artifact')
    if (final_attempts[2] and not final_attempts[1]) or (final_attempts[1] and not final_attempts[0]):
        raise ValueError('b2_receipt_creation_chain')
    # The latest unfinished history still owns these live receipt outputs.
    pending=encode(dict(fence,step='pending',certificate=None)).encode()
    if not (receipts/(prefix+'.completion-fence.intent.json')).exists():
        _prefix_exact(receipts/'p1-maintenance.json',pending)
    if not (receipts/(prefix+'.completion-journal.intent.json')).exists():
        if 'journal' in stages or (receipts/(prefix+'.rotation.intent.json')).exists():
            _prefix_exact(receipts/'journal.json',journal_bytes(intent['expected'][2]['output']['journal']))
        elif intent['before']['journal']['state']!='missing':
            _prefix_exact(receipts/'journal.json',bytes.fromhex(intent['before']['journal']['bytes_hex']))
        elif (receipts/'journal.json').exists() or (receipts/'journal.json').is_symlink():
            raise ValueError('b2_prefix_journal_changed')


def _core_certificate(root,receipts,intent,fault):
    raw=encode(_core_result(intent)['certificate']).encode()
    name=next(n for n,k in archive_names(intent).items() if k=='certificate');path=root/name
    creation=receipts/('p1-recovery-'+intent['id']+'.core-certificate.intent.json')
    descriptor=encode(dict(revision=1,recovery_id=intent['id'],intent_digest=digest(intent),sha256=__import__('hashlib').sha256(raw).hexdigest())).encode()
    if creation.exists():
        _prefix_exact(creation,descriptor)
        if not path.exists():raise ValueError('b2_core_certificate_missing_quarantine')
    else:
        if path.exists() or path.is_symlink():raise ValueError('b2_core_certificate_unrecorded')
        _prefix_file(creation,descriptor,fault,'core-certificate-intent')
        fault('core-certificate-intent-recorded')
        _prefix_file(path,raw,fault,'core-certificate')
    _prefix_exact(path,raw)


def _complete_core(db,roots,intent,fault):
    from service.resource_store import _recovery_scan
    root=roots['control']/'core'
    # Full bounded inventory/prestate was just validated under BOTH locks. The
    # resource observation is separate from ordinary R1/A2 inherited tokens.
    decisions={r['object_id']:r for r in intent['diagnostic_decisions']}
    for row in intent['before']['source']['inventory']:
        if row['classification'] in ('missing_hash_mismatched_terminal','adopted_orphan') and row['id'] not in decisions:
            raise ValueError('b2_unresolved_diagnostic')
    for old in intent['before']['source']['data']['p1_inventory_objects']:
        row=json.loads(old[4])
        if row['classification'] in ('unknown_foreign','unresolved_owner_receipt_reservation'):
            # Dispatch mismatch is accounted by exact original dispatch history;
            # arbitrary reservation/schema defects cannot be acknowledged away.
            if row['classification']=='unknown_foreign' or row['namespace']!='dispatch' or not any(str(v)==row['name'] for v in intent['affected']['dispatches']):
                raise ValueError('b2_unresolved_diagnostic')
        elif row['classification'] in ('missing_hash_mismatched_terminal','adopted_orphan'):
            decision=decisions.get(row['id'])
            if not decision or decision['content_digest']!=diagnostic_digest(row):raise ValueError('b2_unresolved_diagnostic')
    used,count=_prefix_receipt_namespace(roots['receipts'],[json.loads(r[0]) for r in db.execute('SELECT record FROM p1_recoveries ORDER BY rowid')])
    certificate_raw=encode(_core_result(intent)['certificate']).encode()
    descriptor=encode(dict(revision=1,recovery_id=intent['id'],intent_digest=digest(intent),sha256=__import__('hashlib').sha256(certificate_raw).hexdigest())).encode()
    if count+1>MAX_ROWS or used+len(descriptor)>MAX_BYTES:raise ValueError('b2_prefix_receipt_budget')
    _recovery_scan(db,(root,roots['staging']),intent);fault('core-scan')
    if intent['budget_after']!=intent['budget_before']:raise ValueError('b2_budget_change_unsupported')
    def write():
        rows=intent['expected'][3]['output']['rows']
        for row in rows['jobs']:
            db.execute('UPDATE jobs SET job=? WHERE id=?',(row[6],row[1]))
        db.execute("UPDATE resource_reservations SET state='retained'")
        db.execute('UPDATE resource_accounting SET highwater=? WHERE id=1',(rows['resource_accounting'][0][1],))
        db.execute('DELETE FROM resource_faults')
        db.execute("UPDATE resource_monitor SET state='healthy' WHERE id=1")
        db.execute("UPDATE p1_integrity SET state='healthy',token=NULL WHERE id=1 AND revision=1")
        for table,records in _resolution_outputs(intent).items():
            for r in records:
                key=r.get('dispatch_seq',r.get('job_id',r.get('object_id')))
                db.execute('INSERT INTO '+table+' VALUES(?,?,?,?)',(key,intent['id'],encode(r),digest(r)))
        step=dict(recovery_id=intent['id'],**intent['expected'][3],lane=intent['new_lane'],epoch=intent['new_epoch'])
        db.execute('INSERT INTO p1_recovery_steps VALUES(?,?,?,?)',(intent['id'],'core-complete',encode(step),digest(step)))
        _verify_core_rows(db,intent)
        validate_b2_database(db,root,_uncertified=intent['id'])
    _prefix_transaction(db,write,fault,'core-complete')
    sync_dir(root);fault('core-complete-parent')
    _completed_prefix_receipts(roots,intent)
    _core_certificate(root,roots['receipts'],intent,fault)
    validate_b2_database(db,root)


def _receipt_descriptor(intent,raw):
    import hashlib
    return encode(dict(revision=1,recovery_id=intent['id'],intent_digest=digest(intent),sha256=hashlib.sha256(raw).hexdigest())).encode()


def _verify_finished_receipt_evidence(roots,intent):
    from worker.runtime_journal import journal_bytes
    chain=[dict(recovery_id=intent['id'],**e,lane=intent['new_lane'],epoch=intent['new_epoch']) for e in intent['expected'][:4]]
    certificate=_certificate(intent,chain[-1])
    archive=encode(dict(certificate=certificate,steps=chain)).encode()
    fence=dict(revision=1,**{k:intent[k] for k in ('deployment_id','plan_id','plan_digest','proof_digest','old_lane','new_lane')},recovery_id=intent['id'],step='complete',certificate=certificate)
    bound=journal_bytes(dict(intent['expected'][2]['output']['journal'],maintenance=certificate))
    prefix='p1-recovery-'+intent['id'];receipts=roots['receipts']
    core_raw=encode(certificate).encode()
    core_name=next(n for n,k in archive_names(intent).items() if k=='certificate')
    _prefix_exact(roots['control']/'core'/core_name,core_raw)
    _prefix_exact(receipts/(prefix+'.core-certificate.intent.json'),_receipt_descriptor(intent,core_raw))
    _prefix_exact(receipts/(prefix+'.certificate.json'),archive)
    for kind,raw in (('receipt-certificate',archive),('completion-journal',bound),('completion-fence',encode(fence).encode())):
        _prefix_exact(receipts/(prefix+'.'+kind+'.intent.json'),_receipt_descriptor(intent,raw))


def _finish_receipts(db,roots,intent,fault):
    """Core certificate -> receipt certificate -> bound journal -> fence -> SQL.

    Every creation attempt is immutable. Missing/partial output after an attempt
    quarantines; only an exact durable output can be re-barriered and recorded.
    A completed replay validates immutable evidence, never current work or scans.
    """
    from worker.runtime_journal import journal_bytes, maintenance_record
    root=roots['control']/'core';receipts=roots['receipts'];prefix='p1-recovery-'+intent['id']
    validate_b2_database(db,root,_retained_receipts=receipts)  # own durable fourth step AND Core certificate
    chain=[dict(recovery_id=intent['id'],**e,lane=intent['new_lane'],epoch=intent['new_epoch']) for e in intent['expected'][:4]]
    certificate=_certificate(intent,chain[-1])
    fence=dict(revision=1,**{k:intent[k] for k in ('deployment_id','plan_id','plan_digest','proof_digest','old_lane','new_lane')},recovery_id=intent['id'],step='complete',certificate=certificate)
    maintenance_record(fence)
    fence_raw=encode(fence).encode()
    pending=encode(dict(fence,step='pending',certificate=None)).encode()
    archive=encode(dict(certificate=certificate,steps=chain)).encode()
    if len(archive)>65536:raise ValueError('b2_prefix_receipt_budget')
    bound=journal_bytes(dict(intent['expected'][2]['output']['journal'],maintenance=certificate))
    old=journal_bytes(intent['expected'][2]['output']['journal'])
    step=dict(recovery_id=intent['id'],**intent['expected'][4],lane=intent['new_lane'],epoch=intent['new_epoch'])
    row=db.execute('SELECT record,digest FROM p1_recovery_steps WHERE recovery_id=? AND step=?',(intent['id'],'receipt-complete')).fetchone()
    if row is not None and row!=(encode(step),digest(step)):raise ValueError('b2_prefix_step_changed')
    def descriptor(raw):
        return _receipt_descriptor(intent,raw)
    def artifact(path,raw,kind):
        creation=receipts/(prefix+'.'+kind+'.intent.json');expected=descriptor(raw)
        if creation.exists():
            _prefix_exact(creation,expected)
            if not path.exists():raise ValueError('b2_receipt_missing_artifact_quarantine')
        else:
            if path.exists() or path.is_symlink():raise ValueError('b2_receipt_unrecorded_artifact')
            _prefix_file(creation,expected,fault,kind+'-intent');fault(kind+'-intent-recorded')
            _prefix_file(path,raw,fault,kind)
        _prefix_exact(path,raw)
    # Bound the whole additional population before any finalization write.
    used,count=_prefix_receipt_namespace(receipts,[json.loads(r[0]) for r in db.execute('SELECT record FROM p1_recoveries ORDER BY rowid')])
    if row is not None:
        for name in ('p1-maintenance.complete.tmp','p1-maintenance.journal.tmp'):
            if (receipts/name).exists() or (receipts/name).is_symlink():raise ValueError('b2_receipt_temporary_quarantine')
        latest=json.loads(db.execute('SELECT record FROM p1_recoveries ORDER BY rowid DESC LIMIT 1').fetchone()[0])
        if latest['id']==intent['id']:_prefix_exact(receipts/'p1-maintenance.json',fence_raw)
        _prefix_exact(receipts/(prefix+'.certificate.json'),archive)
        for kind,raw in (('receipt-certificate',archive),('completion-journal',bound),('completion-fence',fence_raw)):
            _prefix_exact(receipts/(prefix+'.'+kind+'.intent.json'),descriptor(raw))
        return dict(status='complete',recovery_id=intent['id'],lane=intent['new_lane'],epoch=intent['new_epoch'],certificate=certificate)
    if used+len(archive)+len(bound)+len(fence_raw)+3*1024>MAX_BYTES or count+6>MAX_ROWS:raise ValueError('b2_prefix_receipt_budget')
    # Validate every existing creation attempt before creating any later file.
    # An attempted replacement must already have its exact target, even when
    # the temporary never appeared. Neither half-output may be reminted.
    attempted=[]
    for kind,raw,path,before in (
        ('receipt-certificate',archive,receipts/(prefix+'.certificate.json'),None),
        ('completion-journal',bound,receipts/'journal.json',old),
        ('completion-fence',fence_raw,receipts/'p1-maintenance.json',pending)):
        creation=receipts/(prefix+'.'+kind+'.intent.json')
        attempted.append(creation.exists())
        if creation.exists():
            _prefix_exact(creation,descriptor(raw));_prefix_exact(path,raw)
        elif before is not None:_prefix_exact(path,before)
        elif path.exists() or path.is_symlink():raise ValueError('b2_receipt_unrecorded_artifact')
    if (attempted[2] and not attempted[1]) or (attempted[1] and not attempted[0]):raise ValueError('b2_receipt_creation_chain')
    artifact(receipts/(prefix+'.certificate.json'),archive,'receipt-certificate')
    # Verify the exact receipt document against the durable Core binding before
    # publishing either worker authority file. No self-hashing fifth certificate.
    from worker.runtime_journal import validate_maintenance_archive
    validate_maintenance_archive(fence,json.loads(archive),bytes.fromhex(intent['before']['journal']['bytes_hex']))
    for kind,raw,path,temp,before in (
        ('completion-journal',bound,receipts/'journal.json',receipts/'p1-maintenance.journal.tmp',old),
        ('completion-fence',fence_raw,receipts/'p1-maintenance.json',receipts/'p1-maintenance.complete.tmp',pending)):
        creation=receipts/(prefix+'.'+kind+'.intent.json');expected=descriptor(raw)
        if creation.exists():
            _prefix_exact(creation,expected)
            # Only an actual completed replacement resumes. A leftover temporary
            # or unchanged target after an attempted write is quarantined.
            if temp.exists() or temp.is_symlink():raise ValueError('b2_receipt_temporary_quarantine')
            _prefix_exact(path,raw)
        else:
            _prefix_exact(path,before)
            if temp.exists() or temp.is_symlink():raise ValueError('b2_receipt_temporary_quarantine')
            _prefix_file(creation,expected,fault,kind+'-intent');fault(kind+'-intent-recorded')
            _prefix_file(temp,raw,fault,kind)
            fault(kind+'-before-replace')
            os.replace(temp,path);fault(kind+'-replace')
            sync_dir(receipts);fault(kind+'-dirsync')
            _prefix_exact(path,raw)
    def write():
        db.execute('INSERT INTO p1_recovery_steps VALUES(?,?,?,?)',(intent['id'],'receipt-complete',encode(step),digest(step)))
        validate_b2_database(db,root)
    _prefix_transaction(db,write,fault,'receipt-complete')
    sync_dir(root);fault('receipt-complete-parent')
    return dict(status='complete',recovery_id=intent['id'],lane=intent['new_lane'],epoch=intent['new_epoch'],certificate=certificate)


if __name__=='__main__': main()


# B2.3a is deliberately independent of the private transition kernel above.
# These constants describe the C1 launcher contract, not installed packaging.
HOST_AUTHORITY_DIRECTORY = '/var/lib/py-audio-box/p1-host-authority'
HOST_CAPSULE_MOUNT = '/run/p1-host-authority'
HOST_MAPPED_UID = 0
HOST_CAPSULE_BYTES = 1024 * 1024
HOST_OWNER_LIMIT = 128
HOST_COLLECTION_SECONDS = 120
HOST_CAPSULE_KIND = 'p1-closed-host-authority-v1'


def host_json(raw):
    """Bound before parsing; duplicate JSON keys and nonfinite values refuse."""
    if not isinstance(raw, bytes) or len(raw) > HOST_CAPSULE_BYTES:
        raise ValueError('host_document_budget')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('host_duplicate_key')
            result[key] = value
        return result
    def constant(_): raise ValueError('host_nonfinite')
    def floating(value):
        import math
        result=float(value)
        if not math.isfinite(result): raise ValueError('host_nonfinite')
        return result
    try: return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)
    except (UnicodeError, RecursionError, json.JSONDecodeError):
        raise ValueError('host_document_invalid') from None


def _host_time(value):
    import math
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('host_time')


def _host_id(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('host_identity')


def host_timestamp(value):
    from datetime import datetime
    if not isinstance(value, str) or not re.fullmatch(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?(?:Z|[+-][0-9]{2}:[0-9]{2})', value):
        raise ValueError('host_incarnation_time')
    try: result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError: raise ValueError('host_incarnation_time') from None
    if result.tzinfo is None: raise ValueError('host_incarnation_time')
    fraction=re.search(r'\.([0-9]{1,9})',value)
    nanos=int(fraction[1].ljust(9,'0')) if fraction else 0
    return result.replace(microsecond=0),nanos


def host_registration_union(registrations, scope, deployment_id):
    validate_scope(scope); _host_id(deployment_id)
    if not isinstance(registrations, list) or not 1 <= len(registrations) <= HOST_OWNER_LIMIT:
        raise ValueError('host_registration_budget')
    if len(encode(registrations).encode()) > HOST_CAPSULE_BYTES:
        raise ValueError('host_registration_budget')
    known = {}; seen = set()
    for registration in registrations:
        validate_registration(registration)
        if (registration['scope'] != scope or registration['deployment_id'] != deployment_id or
                registration['evidence'] != 'actual-host-inspection'):
            raise ValueError('host_registration_origin')
        identity = digest(registration)
        if identity in seen: raise ValueError('host_duplicate_registration')
        seen.add(identity)
        for owner in registration['containers']:
            host_timestamp(owner['started'])
            if owner['id'] in known and known[owner['id']] != owner:
                raise ValueError('host_ambiguous_incarnation')
            known[owner['id']] = owner
    if len(known) > HOST_OWNER_LIMIT: raise ValueError('host_owner_budget')
    return [known[k] for k in sorted(known)]


def host_registration_digest(registrations):
    return digest(sorted(registrations, key=digest))


def validate_host_topology(topology, scope, owners, *, stopped):
    closed(topology, 'scope volumes consumers')
    if topology['scope'] != scope: raise ValueError('host_scope_changed')
    volumes = topology['volumes']
    if not isinstance(volumes, list) or len(volumes) != 4: raise ValueError('host_volume_coverage')
    sources = set(); roles = set(); identities = set()
    for volume in volumes:
        closed(volume, 'role name driver created source device inode')
        role = volume['role']
        if role not in scope['volumes'] or role in roles: raise ValueError('host_volume_coverage')
        if {k:volume[k] for k in ('name','driver','created')} != scope['volumes'][role]:
            raise ValueError('host_volume_changed')
        source = volume['source']
        # Only ordinary local Linux Docker volumes. Options/plugins/aliases refuse.
        if (volume['driver'] != 'local' or not isinstance(source, str) or
                not source.startswith('/') or '..' in source.split('/') or
                source in sources or any(source.startswith(s+'/') or s.startswith(source+'/') for s in sources)):
            raise ValueError('host_volume_alias')
        identity=(volume['device'],volume['inode'])
        if any(type(n) is not int or not 1<=n<2**64 for n in identity) or identity in identities:
            raise ValueError('host_volume_alias')
        identities.add(identity); sources.add(source); roles.add(role)
    consumers = topology['consumers']
    if not isinstance(consumers, list) or len(consumers) != len(owners): raise ValueError('host_consumer_coverage')
    expected = {c['id']:c for c in owners}; seen = set()
    for current in consumers:
        closed(current, 'id image started finished running restarting pid restart pidmode privileged capadd cgroupns mounts')
        identity = current['id']
        if not isinstance(identity,str) or identity not in expected or identity in seen:
            raise ValueError('host_unknown_consumer')
        seen.add(identity); owner = expected[identity]
        if any(current[k] != owner[k] for k in ('id','image','started')):
            raise ValueError('host_incarnation_changed')
        start = host_timestamp(current['started'])
        if (current['restart'] != 'no' or current['pidmode'] != '' or current['privileged'] is not False or
                current['capadd'] not in (None,[]) or current['cgroupns'] not in ('','private') or
                type(current['running']) is not bool or type(current['restarting']) is not bool or
                current['restarting'] or type(current['pid']) is not int or not 0<=current['pid']<2**31):
            raise ValueError('host_unsafe_topology')
        if current['running'] != (current['pid'] > 0): raise ValueError('host_pid_state')
        if not isinstance(current['finished'],str) or len(current['finished'])>128:
            raise ValueError('host_finished_time')
        if stopped or not current['running']:
            if current['running'] or current['pid'] != 0 or host_timestamp(current['finished']) <= start:
                raise ValueError('host_not_finished')
        mounts = current['mounts']
        if not isinstance(mounts,list) or len(mounts) != len(owner['volumes']): raise ValueError('host_mount_coverage')
        mounted = set()
        for mount in mounts:
            closed(mount, 'type name source destination rw')
            matches = [v for v in volumes if v['name'] == mount['name']]
            if len(matches) != 1: raise ValueError('host_unknown_mount')
            volume = matches[0]; role = volume['role']
            if (role in mounted or role not in owner['volumes'] or mount['type'] != 'volume' or
                    mount['source'] != volume['source'] or mount['destination'] != '/'+role or
                    type(mount['rw']) is not bool or
                    mount['rw'] != (owner['role'] != 'control' or role not in ('ipc','staging'))):
                raise ValueError('host_mount_changed')
            mounted.add(role)
    return topology


def validate_host_provenance(value, registration):
    closed(value, 'revision kind registration_digest ipc_identity hello owners')
    if (type(value['revision']) is not int or value['revision'] != 1 or
            value['kind'] != 'local-linux-proc-v1' or value['registration_digest'] != digest(registration)):
        raise ValueError('host_provenance_origin')
    closed(value['hello'], 'lane boot')
    if value['hello'] != {k:registration[k] for k in ('lane','boot')}: raise ValueError('host_hello_binding')
    if (not isinstance(value['ipc_identity'],list) or len(value['ipc_identity']) != 2 or
            any(type(n) is not int or not 1<=n<2**64 for n in value['ipc_identity'])): raise ValueError('host_ipc_identity')
    owners = value['owners']
    if not isinstance(owners,list) or len(owners) != len(registration['containers']): raise ValueError('host_proc_coverage')
    expected = {c['id']:c for c in registration['containers']}; seen=set()
    for owner in owners:
        closed(owner, 'id image started pid start_ticks pid_namespace cgroup')
        if not isinstance(owner['id'],str) or owner['id'] not in expected or owner['id'] in seen:
            raise ValueError('host_proc_coverage')
        seen.add(owner['id'])
        if any(owner[k] != expected[owner['id']][k] for k in ('image','started')):
            raise ValueError('host_proc_incarnation')
        if any(type(owner[k]) is not int or not 1<=owner[k]<2**64 for k in ('pid','start_ticks','pid_namespace')) or owner['pid']>=2**31:
            raise ValueError('host_proc_identity')
        if owner['cgroup'] not in ('/docker/'+owner['id'], '/system.slice/docker-'+owner['id']+'.scope'):
            raise ValueError('host_cgroup_association')
    if len({o['pid_namespace'] for o in owners}) != len(owners): raise ValueError('host_shared_pid_namespace')


def validate_host_capsule(capsule, *, now=None):
    """Content validation only. It does not authenticate caller-supplied JSON."""
    deadline=time.monotonic()+2
    if len(encode(capsule).encode()) > HOST_CAPSULE_BYTES: raise ValueError('host_document_budget')
    closed(capsule, 'revision kind invocation_id original_invocation deployment_id scope registrations registration_digest kernel_proof kernel_proof_digest purpose issued expires collection_seconds topology provenance waits')
    if type(capsule['revision']) is not int or capsule['revision'] != 1 or capsule['kind'] != HOST_CAPSULE_KIND:
        raise ValueError('host_capsule_kind')
    _host_id(capsule['invocation_id'])
    if capsule['original_invocation'] is not None:
        _host_id(capsule['original_invocation'])
        if capsule['original_invocation']==capsule['invocation_id']: raise ValueError('host_resume_identity')
    owners = host_registration_union(capsule['registrations'], capsule['scope'], capsule['deployment_id'])
    if capsule['registration_digest'] != host_registration_digest(capsule['registrations']):
        raise ValueError('host_registration_digest')
    proof = capsule['kernel_proof']; closed(proof, 'kind observation prior_epoch old_lane')
    if proof['kind'] != 'host-admin-container-teardown-v1': raise ValueError('host_proof_kind')
    _host_id(proof['prior_epoch']); _host_id(proof['old_lane'])
    if capsule['kernel_proof_digest'] != digest(proof): raise ValueError('host_proof_digest')
    check_consumers(dict(registrations=capsule['registrations'],lane=proof['old_lane']),proof['observation'])
    validate_host_topology(capsule['topology'], capsule['scope'], owners, stopped=True)
    observation = dict(scope=capsule['scope'],containers=owners,evidence='actual-host-inspection',
        states=[{k:c[k] for k in ('id','running','restarting','pid','started','finished')} for c in capsule['topology']['consumers']])
    if proof['observation'] != observation: raise ValueError('host_proof_topology_binding')
    purpose=capsule['purpose']; closed(purpose, 'kind plan_id plan_digest')
    if purpose['kind'] == 'pre-plan':
        if purpose['plan_id'] is not None or purpose['plan_digest'] is not None: raise ValueError('host_preplan_purpose')
    elif purpose['kind'] in ('initial','resume'):
        _host_id(purpose['plan_id']); _hash(purpose['plan_digest'])
    else: raise ValueError('host_purpose')
    if (purpose['kind']=='resume') != (capsule['original_invocation'] is not None):
        raise ValueError('host_resume_identity')
    for k in ('issued','expires','collection_seconds'): _host_time(capsule[k])
    if not 0 < capsule['expires']-capsule['issued'] <= TTL or capsule['collection_seconds'] > HOST_COLLECTION_SECONDS:
        raise ValueError('host_capsule_expiry')
    now = time.time() if now is None else now; _host_time(now)
    if not capsule['issued'] <= now < capsule['expires']: raise ValueError('host_capsule_expired')
    provenance = capsule['provenance']
    if not isinstance(provenance,list) or len(provenance) != len(capsule['registrations']): raise ValueError('host_provenance_coverage')
    by_digest = {digest(r):r for r in capsule['registrations']}; seen=set()
    for value in provenance:
        if not isinstance(value,dict) or value.get('registration_digest') not in by_digest or value['registration_digest'] in seen:
            raise ValueError('host_provenance_coverage')
        seen.add(value['registration_digest']); validate_host_provenance(value, by_digest[value['registration_digest']])
    # A retained registration cannot vouch for two boot/lane identities of the
    # same worker incarnation. No hash/label resolves that ambiguity.
    worker_bindings={}
    for registration in capsule['registrations']:
        for owner in registration['containers']:
            if owner['role'] == 'worker':
                binding=(registration['lane'],registration['boot'])
                if owner['id'] in worker_bindings and worker_bindings[owner['id']] != binding:
                    raise ValueError('host_ambiguous_worker_boot')
                worker_bindings[owner['id']]=binding
    waits=capsule['waits']
    if not isinstance(waits,list) or len(waits) != len(owners): raise ValueError('host_wait_coverage')
    seen=set()
    for wait in waits:
        closed(wait, 'id command_status container_status')
        if (not isinstance(wait['id'],str) or wait['id'] not in {o['id'] for o in owners} or wait['id'] in seen or
                type(wait['command_status']) is not int or wait['command_status'] != 0 or
                type(wait['container_status']) is not int or not 0 <= wait['container_status'] <= 255):
            raise ValueError('host_wait_failed')
        seen.add(wait['id'])
    if time.monotonic()>deadline: raise ValueError('host_validation_deadline')
    return capsule


def _host_file_identity(st):
    return (st.st_dev,st.st_ino,st.st_mode,st.st_uid,st.st_gid,st.st_nlink,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def _host_read_at(directory_fd, name, owner):
    import stat
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory_fd)
    try:
        before=os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o600 or
                before.st_uid!=owner or before.st_nlink!=1 or before.st_size>HOST_CAPSULE_BYTES):
            raise ValueError('host_unsafe_file')
        raw=os.read(fd,HOST_CAPSULE_BYTES+1)
        if (len(raw)!=before.st_size or _host_file_identity(os.fstat(fd))!=_host_file_identity(before) or
                _host_file_identity(os.stat(name,dir_fd=directory_fd,follow_symlinks=False))!=_host_file_identity(before)):
            raise ValueError('host_file_changed')
        value=host_json(raw)
        if (_host_file_identity(os.fstat(fd))!=_host_file_identity(before) or
                _host_file_identity(os.stat(name,dir_fd=directory_fd,follow_symlinks=False))!=_host_file_identity(before)):
            raise ValueError('host_file_changed')
        return value
    finally: os.close(fd)


def _host_mount_topology():
    # OS boundary intentionally factored for E1 monkeypatching, never a CLI flag.
    with open('/proc/self/mountinfo','rb') as stream: raw=stream.read(HOST_CAPSULE_BYTES+1)
    if len(raw)>HOST_CAPSULE_BYTES: raise ValueError('host_mount_budget')
    mounts=[]
    for line in raw.decode('ascii').splitlines():
        fields=line.split(); split=fields.index('-')
        if len(fields)<10 or split<6: raise ValueError('host_mount_invalid')
        def unescape(value):
            return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),value)
        mounts.append(dict(id=fields[0],device=fields[2],root=unescape(fields[3]),
            target=unescape(fields[4]),options=fields[5].split(',')))
        if len(mounts)>MAX_ROWS: raise ValueError('host_mount_budget')
    return mounts


def _host_role_identity(role):
    st=os.stat('/'+role,follow_symlinks=False)
    import stat
    if not stat.S_ISDIR(st.st_mode): raise ValueError('host_role_directory')
    return st.st_dev,st.st_ino


def read_host_capsule():
    """Only the fixed C1 RO transport authenticates origin. No local fallback.

    C1 must qualify host-admin ownership separately and an explicit mapped UID
    0 maintenance/root ownership model. This reader never chowns existing data.
    Reading retains the evidence and grants no kernel/apply capability.
    """
    import stat
    mounts=_host_mount_topology()
    selected=[m for m in mounts if m['target']==HOST_CAPSULE_MOUNT]
    if len(selected)!=1: raise ValueError('host_trusted_mount_absent')
    mount=selected[0]
    prefix=HOST_AUTHORITY_DIRECTORY+'/invocations/'
    if (not mount['root'].startswith(prefix) or not re.fullmatch('[0-9a-f]{32}',mount['root'][len(prefix):]) or
            'ro' not in mount['options'] or 'rw' in mount['options']): raise ValueError('host_trusted_mount_invalid')
    for other in mounts:
        if other is mount: continue
        if (other['target'].startswith(HOST_CAPSULE_MOUNT+'/') or
                (other['device']==mount['device'] and
                 (other['root']==mount['root'] or other['root'].startswith(mount['root']+'/') or
                  mount['root'].startswith(other['root'].rstrip('/')+'/')))):
            raise ValueError('host_proof_mount_alias')
    # Require separate role mounts, and no proof path inside an ordinary role.
    for role in ('control','staging','receipts','ipc'):
        matches=[m for m in mounts if m['target']=='/'+role]
        if len(matches)!=1 or matches[0]['id']==mount['id']: raise ValueError('host_role_mount_absent')
    fd=os.open(HOST_CAPSULE_MOUNT,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if (before.st_uid!=HOST_MAPPED_UID or stat.S_IMODE(before.st_mode)!=0o700 or
                mount['device']!=str(os.major(before.st_dev))+':'+str(os.minor(before.st_dev))):
            raise ValueError('host_unsafe_directory')
        with os.scandir(fd) as entries:
            if [entry.name for entry in entries] != ['capsule.json']: raise ValueError('host_invocation_namespace')
        capsule=_host_read_at(fd,'capsule.json',HOST_MAPPED_UID)
        if (capsule.get('invocation_id')!=mount['root'][len(prefix):] or
                _host_file_identity(os.fstat(fd))!=_host_file_identity(before) or mounts!=_host_mount_topology()):
            raise ValueError('host_transport_changed')
        validate_host_capsule(capsule)
        for volume in capsule['topology']['volumes']:
            role_mount=next(m for m in mounts if m['target']=='/'+volume['role'])
            device,inode=_host_role_identity(volume['role'])
            if (role_mount['root']!=volume['source'] or
                    role_mount['device']!=str(os.major(device))+':'+str(os.minor(device)) or
                    (device,inode)!=(volume['device'],volume['inode'])):
                raise ValueError('host_role_mount_changed')
        if mounts!=_host_mount_topology(): raise ValueError('host_transport_changed')
        return capsule
    finally: os.close(fd)
