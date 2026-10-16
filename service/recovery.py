"""Bounded offline B1 inspection/plans. Apply has no implementation or authority."""
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


def snapshot(roots, db, scope, *, exclude_plan=None, exclude_abandon=None, transaction=True):
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
            validate_b2_database(db, roots['control']/'core')
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
        journal_raw=read_file(roots['receipts']/'journal.json',1024**2)
        journal=json.loads(journal_raw)
        closed(journal,'payload sha256')
        payload_keys='version lane records maintenance' if 'maintenance' in journal['payload'] else 'version lane records'
        closed(journal['payload'],payload_keys)
        from worker.runtime_journal import read_maintenance
        receipt_completion=read_maintenance(roots['receipts'])
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
        scanner=Reconciliation.__new__(Reconciliation)
        scanner.core=SimpleNamespace(root=roots['control']/'core'); scanner.db=db; scanner.staging=roots['staging']
        scanner.entry_limit=MAX_ROWS; scanner.byte_limit=MAX_BYTES; scanner.seconds=1
        scanner._collect()
        if tables & B2_COLUMNS.keys():
            size += scanner.bytes + len(journal_raw)
            if size>MAX_BYTES or count+scanner.entries+len(payload['records'])>MAX_ROWS:
                raise ValueError('b2_aggregate_budget')
        # Exclude only this proposal. Later appends invalidate older plans.
        state=dict(data)
        state['p1_recovery_plans']=[r for r in data['p1_recovery_plans'] if r[0]!=exclude_plan]
        if 'p1_abandon_plans' in data:
            state['p1_abandon_plans']=[r for r in data['p1_abandon_plans'] if r[0]!=exclude_abandon]
        inventory=[]
        for row in sorted(scanner.rows.values(),key=lambda r:r['id']):
            inventory.append({**row,'references':[r for r in row['references'] if r not in ('p1_recovery_plans:'+str(exclude_plan), 'p1_abandon_plans:'+str(exclude_abandon))]})
        projection=dict(schema=schema,data=state,inventory=inventory,journal=digest(journal),marker=marker)
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
            allocation_coherent=_allocation_coherent(data),
            inventory=dict(counts=dict(__import__('collections').Counter(r['classification'] for r in scanner.rows.values())),reasons=sorted(scanner.reasons)),
            journal_digest=digest(journal),lane=payload['lane'],
            journal_records=[dict(runtime_id=r['id'],boot=r['boot'],phase=r['phase'],binding=r['allocation']['binding']) for r in payload['records']],
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
        if action=='retain-terminal':
            if old!=new:raise ValueError('b2_output_terminal_history')
        elif job['state']!=('failed' if action=='fail' else 'cancelled') or job['id']!=prior['id'] or job['generation']!=prior['generation'] or job['deadline']!=prior['deadline']:
            raise ValueError('b2_output_job_outcome')
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
                  'cancel-requested' if job.get('cancel_requested') else 'fail')
        if action['before_digest']!=digest(job) or action['action']!=expected: raise ValueError('b2_action_before_binding')
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


def _validate_archives(root,data,entries,used):
    import hashlib
    deadline=time.monotonic()+2
    for r in data['p1_recoveries']:
        steps={s['step']:s for s in data['p1_recovery_steps'] if s['recovery_id']==r['id']}
        if 'evidence' not in steps:continue
        content={}
        for name,kind in archive_names(r).items():
            if kind=='certificate' and 'core-complete' not in steps:continue
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


def _validate_b2_database(db, root):
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
    _validate_archives(root,data,count,size)
    return data


def validate_b2_database(db, root=None):
    # These owning seams have no live resource observation. Restore no handler;
    # offline snapshot installs its own deadline for its remaining SQL work.
    deadline=time.monotonic()+2
    db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
    try:
        if root is None:
            root=Path(next(r[2] for r in db.execute('PRAGMA database_list') if r[1]=='main')).parent
        return _validate_b2_database(db,Path(root))
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
            elif reservation[3]!='retained': return False
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


if __name__=='__main__': main()
