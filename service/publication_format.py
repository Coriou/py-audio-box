"""Private, fixed clean-local publication-v1 -> v2 operator transition.

No enrollment, dispatch recovery, cleanup or public service route. Calling this
module on a real root requires separate operator authority.
"""
import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
from types import SimpleNamespace

from service import persistence as p
from service.core import encode

MAX_ROWS = 2048
MAX_BYTES = 16*1024**2
_INVENTORY = frozenset(('p1_inventory_scans','p1_inventory_objects','p1_inventory_history','p1_integrity'))
_RESOURCE = frozenset(('resource_reservations','resource_accounting','resource_faults','resource_monitor','resource_protocol'))
_DISPATCH = frozenset(('dispatch_settings','dispatches','dispatch_commands'))
_BASE = frozenset(('jobs','events','revisions','publications','assets','artifacts','p1_format','publication_work'))


class Refusal(ValueError):
    pass


def _refused(exc):
    message=str(exc)[:180]
    if message.startswith('invalid-prestate:'):
        return 2,message
    if (isinstance(exc,(p.PublicationQuarantine,sqlite3.DatabaseError,KeyError,TypeError,json.JSONDecodeError)) or
            isinstance(exc,ValueError) and not isinstance(exc,Refusal)):
        return 2,'invalid-prestate:'+message
    return 2,'refused:'+message


def _identity(rows):
    from service.voice_store import _sql_identity
    return sorted((a,b,c,_sql_identity(d)) for a,b,c,d in rows)


def _model(tables, version):
    """Only complete initializer groups, built from canonical literals."""
    from service.recovery import _INSTALLED_V1_TABLE_DDL
    from service.voice_store import _TABLE_DDL, _TRIGGER_DDL, _TABLES
    required = _BASE | _TABLES | {'sqlite_sequence'}
    allowed = required | _INVENTORY | _RESOURCE | _DISPATCH
    if not required <= tables or tables - allowed:
        raise Refusal('scope-unsupported:unknown-or-enrolled-schema')
    for group in (_INVENTORY,_RESOURCE,_DISPATCH):
        if tables & group and not group <= tables: raise Refusal('invalid-prestate:partial-group')
    db=sqlite3.connect(':memory:')
    try:
        db.execute('PRAGMA trusted_schema=OFF')
        for sql in _INSTALLED_V1_TABLE_DDL:
            name=sql.split('(')[0].split()[-1]
            if name not in tables: continue
            if name=='publication_work': sql=p.PUBLICATION_V1 if version==1 else p.PUBLICATION_V2
            db.execute(sql)
        for sql in (*_TABLE_DDL,*_TRIGGER_DDL): db.execute(sql)
        return db
    except BaseException:
        db.close();raise


def _typed_equal(a,b):
    return encode(a)==encode(b)


def _validate(db, root):
    end=time.monotonic()+2
    db.set_progress_handler(lambda:int(time.monotonic()>end),1000)
    n,size=db.execute('SELECT count(*),coalesce(sum(coalesce(length(sql),0)+length(name)+length(tbl_name)+length(type)),0) FROM sqlite_master').fetchone()
    if n>MAX_ROWS or size>MAX_BYTES:raise Refusal('inspection-budget')
    schema=list(db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master LIMIT 2049'))
    if len(schema)>MAX_ROWS or len(encode(schema).encode())>MAX_BYTES: raise Refusal('inspection-budget')
    tables={r[1] for r in schema if r[0]=='table'}
    # Check closed scope before interpreting markers or any enrolled documents.
    with closing(_model(tables,1)):
        pass
    version=p.store_format(db)
    total=len(encode(schema).encode());count=0;data={};rows={};columns={}
    with closing(_model(tables,version)) as model:
        model.set_progress_handler(lambda:int(time.monotonic()>end),1000)
        if _identity(schema)!=_identity(model.execute('SELECT type,name,tbl_name,sql FROM sqlite_master')):
            raise Refusal('invalid-prestate:schema')
        if list(db.execute('PRAGMA integrity_check(1)'))!=[('ok',)] or list(db.execute('PRAGMA foreign_key_check')):
            raise Refusal('invalid-prestate:integrity')
        for table in sorted(tables):
            fields=[r[1] for r in model.execute('PRAGMA table_info('+table+')')];columns[table]=fields
            sizes='+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in fields)
            n,b=db.execute('SELECT count(*),coalesce(sum('+sizes+'),0) FROM '+table).fetchone()
            count+=n;total+=b
            if count>MAX_ROWS or total>MAX_BYTES or time.monotonic()>end: raise Refusal('inspection-budget')
            rows[table]=list(db.execute('SELECT rowid,* FROM '+table+' ORDER BY rowid'))
            data[table]=[dict(zip(fields,r[1:])) for r in rows[table]]
            if table=='sqlite_sequence': continue
            model.executemany('INSERT INTO '+table+'(rowid,'+','.join(fields)+') VALUES('+','.join('?' for _ in range(len(fields)+1))+')',rows[table])
            if not _typed_equal(list(model.execute('SELECT rowid,* FROM '+table+' ORDER BY rowid')),rows[table]):
                raise Refusal('invalid-prestate:affinity')
        if list(model.execute('PRAGMA foreign_key_check')): raise Refusal('invalid-prestate:foreign-key')
        sequence=data['sqlite_sequence'];names=[r['name'] for r in sequence]
        if len(names)!=len(set(names)) or any(r['name'] not in ('jobs','events') or type(r['seq']) is not int or r['seq']<0 for r in sequence):
            raise Refusal('invalid-prestate:sequence')
        for table in ('jobs','events'):
            maximum=max((r['seq'] for r in data[table]),default=0)
            if maximum and not any(r['name']==table and r['seq']>=maximum for r in sequence):
                raise Refusal('invalid-prestate:sequence')
    if any(data.get(t) for t in _DISPATCH): raise Refusal('dispatch-history')
    from service.recovery import _installed_detached_documents
    _installed_detached_documents(data,publication_format=version)
    from contracts import validate as oracle
    for row in data['jobs']:
        job=json.loads(row['job']);context=json.loads(row['context'])
        from worker.native_family import frozen_native
        if frozen_native(context):raise Refusal('scope-unsupported:native-history')
        oracle.plan_hash(context['plan']);oracle.frozen_plan(context['plan'],context['bundles'])
        if (job['state'] not in ('succeeded','failed','cancelled') or
                any(a['reaped'] is not True for a in job['attempts'])): raise Refusal('active-job-or-attempt')
        if job['input_sha256']!=oracle.plan_hash(context['plan']): raise Refusal('invalid-prestate:job-plan')
        request=context['plan']['request']
        fingerprints={oracle.fingerprint(row['principal'],row['operation'],request)}
        if row['retry_of']:
            fingerprints.add(oracle.fingerprint(row['principal'],row['operation'],dict(request=request,retry_of=row['retry_of'])))
            previous=next((r for r in data['jobs'] if r['id']==row['retry_of']),None)
            if previous is None or previous['principal']!=row['principal'] or previous['operation']!=row['operation'] or previous['context']!=row['context']:
                raise Refusal('invalid-prestate:retry-lineage')
        if row['fingerprint'] not in fingerprints or job.get('retry_of',row['retry_of'])!=row['retry_of']:
            raise Refusal('invalid-prestate:idempotency')
    for row in data['artifacts']: p.public_artifact(row,data,publication_format=version)
    published={json.loads(r['metadata'])['job_id'] for r in data['artifacts']}
    if {r['job_id'] for r in data['publications']} != published or {r['job_id'] for r in data['publication_work']} != published:
        raise Refusal('pending-publication')
    if _RESOURCE <= tables:
        if (data['resource_faults'] or data['resource_monitor']!=[dict(id=1,state='healthy')] or
                data['resource_protocol']!=[dict(id=1,revision=2)]): raise Refusal('unhealthy-resource')
        accounting=data['resource_accounting']
        if len(accounting)!=1 or accounting[0]['id']!=1 or type(accounting[0]['highwater']) is not int or accounting[0]['highwater']<0:
            raise Refusal('invalid-prestate:accounting')
        for r in data['resource_reservations']:
            if r['state']!='retained' or not any(j['id']==r['id'] for j in data['jobs']): raise Refusal('active-or-unbound-reservation')
            stored=next(j for j in data['jobs'] if j['id']==r['id'])
            context=json.loads(stored['context']);job=json.loads(stored['job']);plan=context['plan']
            from worker.runtime_protocol import execution
            spec=execution(context);limits=plan['resolved']['profile']['limits']
            # Pure retained counterpart of Admission.demand: no hardware refresh,
            # observations, release or resource reconstruction is performed.
            if spec is None:
                if plan['resolved']['profile']['id']!='cpu-synthetic':raise Refusal('unsupported-retained-resource')
                configuration=oracle.canonical_hash(plan['resolved']['identity']['effective'])
                ram=192*1024**2;vram=0;disk=2*1024**2
            else:
                if getattr(spec,'device',None)=='mps':raise Refusal('scope-unsupported:native-history')
                configuration=oracle.canonical_hash(context['execution']);ram=spec.memory_bytes
                vram=limits['vram_bytes'] if spec.engine=='qwen' and spec.device.startswith('cuda:') else 0
                disk=limits['disk_bytes']
            demand=dict(configuration=configuration,slots=1,ram_bytes=ram,vram_bytes=vram,pids=48,
                        disk_bytes=disk*max(1,len(job['attempts'])),attempt_disk_bytes=disk)
            if r['category']!='inference' or encode(json.loads(r['demand']))!=encode(demand):
                raise Refusal('invalid-prestate:reservation-binding')
    if _INVENTORY <= tables:
        if data['p1_integrity']!=[dict(id=1,revision=1,state='healthy',token=None)]: raise Refusal('unhealthy-integrity')
        for r in data['p1_inventory_scans']:
            value=json.loads(r['summary'])
            if (r['complete']!=1 or type(r['complete']) is not int or
                    set(value)!={'scan_id','complete','defect_latched','blocked','counts','reasons','cleanup_allowed'} or
                    value['scan_id']!=r['id'] or value['complete'] is not True or value['blocked'] is not False or
                    value['defect_latched'] is not False or value['cleanup_allowed'] is not False or value['reasons']!=[] or
                    not isinstance(value['counts'],dict) or any(type(v) is not int or v<0 for v in value['counts'].values())):
                raise Refusal('unhealthy-inventory-scan')
        for r in (*data['p1_inventory_objects'],*data['p1_inventory_history']):
            value=json.loads(r['record'])
            if (set(value)!={'id','namespace','name','classification','reason','identity','owner','references'} or
                    any(type(value[k]) is not str or len(value[k])>256 for k in ('id','namespace','name','classification','reason')) or
                    value['id']!=hashlib.sha256(encode([value['namespace'],value['name']]).encode()).hexdigest() or
                    value['id']!=r.get('object_id',r.get('id')) or
                    not isinstance(value['references'],list) or len(value['references'])>MAX_ROWS or
                    any(type(v) is not str or len(v)>256 for v in value['references'])):
                raise Refusal('invalid-prestate:inventory-document')
            if value['classification'] in ('unknown_foreign','adopted_orphan','missing_hash_mismatched_terminal','unresolved_owner_receipt_reservation','pending_adoption_known_alias'):
                raise Refusal('unhealthy-inventory')
    # Pure existing inventory collector: no constructor, scan commit or cleanup.
    from service.reconciliation import Reconciliation
    scan=Reconciliation.__new__(Reconciliation)
    scan.core=SimpleNamespace(root=root,db=db,publication_format=version)
    scan.db=db;scan.staging=None;scan.entry_limit=MAX_ROWS;scan.byte_limit=MAX_BYTES
    scan.scalar_limit=MAX_BYTES;scan.media_limit=160*1024**2 if version==2 else MAX_BYTES;scan.seconds=1
    scan._collect()
    if scan.reasons: raise Refusal('namespace-or-reference:'+','.join(sorted(scan.reasons)))
    if any(r['classification']=='pending_adoption_known_alias' for r in scan.rows.values()): raise Refusal('pending-adoption')
    return version,schema,rows


def _namespace(root):
    """No-follow chain and fixed local owner. Never repairs permissions."""
    from service.install_policy import parent_fd
    ancestors=[]
    chain=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in (*root.parts[1:-1],None):
            item=os.fstat(chain)
            if item.st_uid not in (0,os.getuid()) or (item.st_mode & 0o022 and not (item.st_uid==0 and item.st_mode & stat.S_ISVTX)):
                raise Refusal('foreign-or-writable-ancestor')
            ancestors.append((item.st_dev,item.st_ino,item.st_mode,item.st_uid,item.st_gid))
            if part is not None:
                next_fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=chain)
                os.close(chain);chain=next_fd
    finally:os.close(chain)
    with parent_fd(root) as (parent,name):
        fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
        try:
            info=os.fstat(fd)
            if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700: raise Refusal('private-local-root-required')
            names=os.listdir(fd)
            if len(names)>MAX_ROWS or set(names)-{'core.sqlite','core.sqlite-wal','core.sqlite-shm','control.lock','assets','staging','owners'}:
                raise Refusal('scope-unsupported:namespace')
            identities={}
            for leaf in names:
                item=os.stat(leaf,dir_fd=fd,follow_symlinks=False)
                directory=leaf in ('assets','staging','owners')
                if (item.st_uid!=os.getuid() or stat.S_IMODE(item.st_mode)!=(0o700 if directory else 0o600) or
                        not (stat.S_ISDIR(item.st_mode) if directory else stat.S_ISREG(item.st_mode) and item.st_nlink==1)):
                    raise Refusal('foreign-local-namespace')
                identities[leaf]=(item.st_dev,item.st_ino,item.st_mode,item.st_uid,item.st_gid,item.st_nlink)
            if not {'core.sqlite','control.lock','assets','staging','owners'}<=set(names): raise Refusal('incomplete-local-root')
            return (info.st_dev,info.st_ino),identities,ancestors
        finally: os.close(fd)


def _owner_info(fd):
    info=os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o600 or
            info.st_uid!=os.getuid() or info.st_nlink!=1 or info.st_size!=0):
        raise Refusal('foreign-owner-file')
    return info


def _same(root,before,owner=None):
    now=_namespace(root)
    if now[0]!=before[0] or now[2]!=before[2] or any(now[1].get(k)!=v for k,v in before[1].items()): raise Refusal('local-identity-changed')
    if owner is not None:
        for stream,name in zip(owner.locks,owner.lock_names):
            info=_owner_info(stream.fileno())
            other=os.stat(root/'owners'/name,follow_symlinks=False)
            if (info.st_dev,info.st_ino)!=(other.st_dev,other.st_ino):raise Refusal('owner-changed')
    return now


def _hold_owners(root, db, owner):
    known={a['owner_id'] for (raw,) in db.execute('SELECT job FROM jobs') for a in json.loads(raw)['attempts']}
    directory=os.open(root/'owners',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        names=os.listdir(directory)
        if len(names)>MAX_ROWS:raise Refusal('owner-budget')
        for name in names:
            if name not in known:raise Refusal('unknown-owner')
            fd=os.open(name,os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
            stream=os.fdopen(fd,'r+b');owner.locks.append(stream);owner.lock_names.append(name)
            info=_owner_info(fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            _owner_info(fd)
            other=os.stat(name,dir_fd=directory,follow_symlinks=False)
            if (info.st_dev,info.st_ino)!=(other.st_dev,other.st_ino):raise Refusal('owner-changed')
    finally:os.close(directory)


def migrate(root, *, expect_v1, _fault=lambda point:None):
    """Return (exit code, finite result). Private fault seam, never CLI input."""
    from service import core as c
    from service.recovery import private_database
    root=Path(root)
    if expect_v1 is not True or not root.is_absolute() or '..' in root.parts: return 2,'refused:explicit-existing-absolute-root-and-expect-v1-required'
    try:
        with c._core_maintenance_lease():
            # The lease already holds bookkeeping. Do not reserve/construct Core
            # or call registry cleanup while inside it. A retained sentinel fences
            # all future Core/maintenance admission if close becomes uncertain.
            owner=SimpleNamespace(root_key=os.path.normcase(str(root)),root_identity=None,
                constructor_failed=False,state='maintenance',db=None,owner=None,private=None,locks=[],lock_names=[])
            c._core_registry[0]=owner
            result=(2,'refused');committing=False;begun=False;db=None;lock=None;close_uncertain=False;mutated=False
            try:
                before=_namespace(root);owner.root_identity=before[0]
                fd=os.open(root/'control.lock',os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK)
                lock=owner.owner=os.fdopen(fd,'r+b')
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                st=os.fstat(fd)
                if (st.st_dev,st.st_ino,st.st_mode,st.st_uid,st.st_gid,st.st_nlink)!=before[1]['control.lock']: raise Refusal('control-lock-changed')
                _same(root,before,owner)
                owner.private=private_database(root/'core.sqlite')
                private=owner.private.__enter__()
                db=owner.db=sqlite3.connect(private.as_uri()+'?mode=ro',uri=True,timeout=0,isolation_level=None)
                db.execute('PRAGMA trusted_schema=OFF');db.execute('PRAGMA query_only=ON')
                predecessor=_validate(db,root)
                _hold_owners(root,db,owner)
                try:
                    db.close();owner.db=None;db=None
                    owner.private.__exit__(None,None,None);owner.private=None
                except BaseException:
                    close_uncertain=True;raise
                _same(root,before,owner)
                db=owner.db=sqlite3.connect((root/'core.sqlite').as_uri()+'?mode=rw',uri=True,timeout=0,isolation_level=None)
                db.execute('PRAGMA trusted_schema=OFF');db.execute('PRAGMA foreign_keys=ON');db.execute('PRAGMA synchronous=FULL')
                db.execute('PRAGMA wal_autocheckpoint=0')
                if db.execute('PRAGMA journal_mode').fetchone()!=('wal',): raise Refusal('existing-wal-required')
                db.execute('BEGIN IMMEDIATE');begun=True
                current=_validate(db,root)
                if not _typed_equal(current,predecessor): raise Refusal('prestate-changed')
                before=_same(root,before,owner)
                if current[0]==2:
                    db.execute('ROLLBACK');begun=False;result=(0,'already-v2')
                else:
                    _fault('before-rename')
                    db.execute('ALTER TABLE publication_work RENAME TO publication_work_v1_transition')
                    mutated=True
                    _fault('after-rename')
                    db.execute(p.PUBLICATION_V2);_fault('after-create')
                    db.execute('INSERT INTO publication_work(rowid,job_id,binding,destination,temporary,sha256,byte_length) SELECT rowid,job_id,binding,destination,temporary,sha256,byte_length FROM publication_work_v1_transition ORDER BY rowid')
                    _fault('after-copy')
                    db.execute('DROP TABLE publication_work_v1_transition');_fault('after-drop')
                    if db.execute("UPDATE p1_format SET revision=2 WHERE typeof(revision)='integer' AND revision=1 AND scope='synthetic-core'").rowcount!=1: raise Refusal('marker-precondition')
                    _fault('after-marker')
                    db.execute('PRAGMA user_version=2');_fault('after-version')
                    post=_validate(db,root)
                    expected=dict(current[2]);expected['p1_format']=[(r[0],2,r[2]) for r in expected['p1_format']]
                    if not _typed_equal(post[2],expected): raise Refusal('post-row-preservation')
                    _same(root,before,owner);_fault('before-commit')
                    committing=True
                    db.execute('COMMIT');begun=False
                    _fault('after-commit')
                    c.sync_dir(root);c.sync_dir(root.parent)
                    if p.store_format(db)!=2: raise Refusal('committed-pair')
                    _same(root,before,owner);_fault('after-verify')
                    result=(0,'migrated')
            except BaseException as exc:
                if committing or close_uncertain:
                    result=(4,'outcome-unknown:reopen-required:'+type(exc).__name__)
                elif begun:
                    try:
                        db.execute('ROLLBACK');begun=False
                        result=(_refused(exc) if not mutated and isinstance(exc,(Refusal,p.PublicationQuarantine)) else (3,'failed-rolled-back:'+type(exc).__name__))
                    except BaseException:
                        result=(4,'outcome-unknown:rollback-unconfirmed')
                else:
                    result=_refused(exc)
            finally:
                try:
                    if close_uncertain:raise OSError('prior-close-unconfirmed')
                    _fault('before-close')
                    if db is not None: db.close();owner.db=None
                    if owner.private is not None:owner.private.__exit__(None,None,None);owner.private=None
                    while owner.locks:
                        owner.locks[-1].close();owner.locks.pop()
                    if lock is not None: lock.close();owner.owner=None
                    c._core_registry[0]=None
                except BaseException:
                    owner.state='uncertain'
                    result=(4,'outcome-unknown:close-unconfirmed-restart-required')
            return result
    except (ValueError,OSError) as exc:
        return 2,'refused:'+str(exc)[:180]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transition',choices=['migrate-v1-to-v2'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--expect-v1',action='store_true',required=True)
    args=parser.parse_args(argv)
    code,result=migrate(args.root,expect_v1=args.expect_v1)
    print(result)
    return code


if __name__=='__main__':
    raise SystemExit(main())
