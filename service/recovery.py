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
from types import SimpleNamespace
import uuid

from service.core import encode, sync_dir
from service.persistent_init import digest, read_file, roots_checked, validate_scope
from service.persistence import B1_COLUMNS
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
def offline(roots, *, write=False, journal_lock=True):
    roots = roots_checked(roots)
    locks = []
    db = None
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
        db=sqlite3.connect(path.as_uri()+('?mode=rw' if write else '?mode=ro'),uri=True,timeout=0,isolation_level=None)
        db.execute('PRAGMA trusted_schema=OFF')
        if write: db.execute('PRAGMA synchronous=FULL')
        if not write: db.execute('PRAGMA query_only=ON')
        end=time.monotonic()+2
        db.set_progress_handler(lambda:int(time.monotonic()>end),1000)
        yield roots,db
    finally:
        if db is not None: db.close()
        for fd in reversed(locks): os.close(fd)


def snapshot(roots, db, scope):
    """Read-only transaction including WAL; never calls an observation fence."""
    validate_scope(scope)
    db.execute('BEGIN')
    try:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        expected=dict(COLUMNS,**B1_COLUMNS,
            p1_inventory_scans='id observed complete fingerprint summary',
            p1_inventory_objects='id first_observed last_observed scan_id record',
            p1_inventory_history='seq scan_id object_id observed record',
            p1_integrity='id revision state token',sqlite_sequence='name seq')
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
        journal_raw=read_file(roots['receipts']/'journal.json')
        journal=json.loads(journal_raw)
        closed(journal,'payload sha256'); closed(journal['payload'],'version lane records')
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
        # Plans are immutable proposals, not source authority. Adding a proposal
        # must not invalidate itself; every other row including diagnostics binds.
        state={k:v for k,v in data.items() if k!='p1_recovery_plans'}
        inventory=[]
        for row in sorted(scanner.rows.values(),key=lambda r:r['id']):
            inventory.append({**row,'references':[r for r in row['references'] if not r.startswith('p1_recovery_plans:')]})
        fingerprint=digest(dict(schema=schema,data=state,inventory=inventory,journal=digest(journal),marker=marker))
        affected=dict(jobs=[r[1] for r in data['jobs']], dispatches=[r[0] for r in data['dispatches'] if not r[-1]], publications=[r[0] for r in data['publications']])
        if sum(map(len,affected.values()))>MAX_ROWS: raise ValueError('affected_budget')
        return dict(deployment_id=identity,source_digest=fingerprint,affected=affected,bytes=size,
            monitor=data['resource_monitor'],protocol=2,integrity=data['p1_integrity'],
            reservations=[dict(zip(('job_id','category','demand','state'),(r[0],r[1],json.loads(r[2]),r[3]))) for r in data['resource_reservations']],
            faults=[r[0] for r in data['resource_faults']],
            dispatches=[dict(seq=r[0],job_id=r[1],binding=r[3],runtime_id=r[5],phase=r[6],receipt_digest=digest(json.loads(r[7])) if r[7] else None,acked=bool(r[9])) for r in data['dispatches']],
            jobs=[{k:v for k,v in json.loads(r[6]).items() if k in ('id','state','generation','deadline','attempts')} for r in data['jobs']],
            publications=[dict(job_id=r[0],manifest_digest=digest(json.loads(r[1]))) for r in data['publications']],
            inventory=dict(counts=dict(__import__('collections').Counter(r['classification'] for r in scanner.rows.values())),reasons=sorted(scanner.reasons)),
            journal_digest=digest(journal),lane=payload['lane'],
            journal_records=[dict(runtime_id=r['id'],boot=r['boot'],phase=r['phase'],binding=r['allocation']['binding']) for r in payload['records']],
            registrations=registrations,
            versions=dict(contract='1.3.0',core=1,private=1,resource=2),apply_available=False)
    finally: db.execute('ROLLBACK')


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
        state=snapshot(roots,db,scope); check_consumers(state,observation)
        if value['source_digest']!=state['source_digest'] or value['proof_digest']!=digest(observation) or value['affected']!=state['affected']:
            raise ValueError('stale_plan')
        return value


def apply(*args,**kwargs):
    raise NotImplementedError('P1B2_apply_unavailable')


def enrollment_gate(root, registration_id, hello, profiles, budget):
    """Read-only pre-Core startup check. No Docker access or new health epoch."""
    path=Path(root).absolute()/'core.sqlite'
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=0) as db:
        db.execute('PRAGMA query_only=ON')
        for table,columns in B1_COLUMNS.items():
            if [r[1] for r in db.execute('PRAGMA table_info('+table+')')] != columns.split(): raise ValueError('unsupported_enrollment_schema')
            n,b=db.execute('SELECT count(*),coalesce(sum(length(record)),0) FROM '+table).fetchone()
            if n>MAX_ROWS or b>MAX_BYTES: raise ValueError('enrollment_read_budget')
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
        if (digest(record)!=row[1] or row[1]!=registration_id or record['deployment_id']!=installation['deployment_id'] or
            record['scope']!=installation['host'] or record['evidence']!='actual-host-inspection' or
            record['lane']!=hello.get('lane') or record['boot']!=hello.get('boot') or hello.get('quarantined') is not False):
            raise ValueError('persistent_registration_mismatch')
    return installation['deployment_id']


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


if __name__=='__main__': main()
