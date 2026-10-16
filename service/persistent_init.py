"""Create-only owned synthetic installation. No legacy migration or key reminting."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat

from service.core import Core, encode, sync_dir
from service import persistence
from service.admission import Admission, finite_budget
from service.dispatch_store import DispatchStore

ROLES = ('control', 'staging', 'receipts', 'ipc')
LIMIT = 1024 * 1024


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def read_file(path, limit=LIMIT):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid != os.getuid() or s.st_size > limit:
            raise ValueError('unsafe_private_file')
        data = os.read(fd, limit+1)
        identity=lambda v:(v.st_dev,v.st_ino,v.st_mode,v.st_uid,v.st_gid,v.st_nlink,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
        if len(data) != s.st_size or identity(os.fstat(fd)) != identity(s) or identity(os.stat(path,follow_symlinks=False)) != identity(s):
            raise ValueError('private_file_changed')
        return data
    finally:
        os.close(fd)


def exclusive(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        persistence.write_complete(f, data)
    sync_dir(path.parent)
    sync_dir(path.parent.parent)


def roots_checked(roots):
    if set(roots) != set(ROLES): raise ValueError('exact_volume_roles_required')
    if any(Path(v).is_symlink() for v in roots.values()): raise ValueError('symlink_volume')
    result = {k: Path(v).resolve() for k,v in roots.items()}
    for p in result.values():
        if p.is_symlink() or not p.is_dir() or p.stat().st_uid != os.getuid():
            raise ValueError('owned_existing_volume_required')
        if any(parent.is_symlink() for parent in p.parents): raise ValueError('symlink_parent')
    if len({(p.stat().st_dev,p.stat().st_ino) for p in result.values()}) != 4:
        raise ValueError('distinct_volume_roles_required')
    if any(a in b.parents for a in result.values() for b in result.values() if a != b):
        raise ValueError('nested_volume_roles')
    return result


def validate_scope(scope):
    if not isinstance(scope, dict) or set(scope) != {'daemon','context','volumes'}:
        raise ValueError('closed_installation_scope')
    if any(not isinstance(scope[k],str) or not 1 <= len(scope[k]) <= 256 for k in ('daemon','context')):
        raise ValueError('invalid_daemon_scope')
    if set(scope['volumes']) != set(ROLES): raise ValueError('exact_volume_roles_required')
    for v in scope['volumes'].values():
        if not isinstance(v,dict) or set(v) != {'name','driver','created'} or any(not isinstance(x,str) or not 1<=len(x)<=256 for x in v.values()):
            raise ValueError('exact_volume_identity_required')
    if len({v['name'] for v in scope['volumes'].values()}) != 4: raise ValueError('distinct_volume_identity')
    return scope


def initialize(roots, scope, profiles, budget, *, complete=None, fault=lambda _:None):
    """Caller owns fresh volumes. Resume requires the exact private record digest.

    Ambiguous partial Core construction is quarantined, never reconstructed.
    The private seed permits only missing *not-yet-created* deterministic files;
    any existing differing file is retained and refuses completion.
    """
    roots = roots_checked(roots); validate_scope(scope); budget = finite_budget(budget)
    if not isinstance(profiles,list) or len(profiles)!=1 or len(encode(profiles).encode())>48000: raise ValueError('bounded_synthetic_profiles_required')
    from contracts import validate as oracle
    oracle.validate('profile',profiles[0])
    if profiles[0]['id']!='cpu-synthetic': raise ValueError('fixed_synthetic_profile_required')
    record_path = roots['receipts']/'p1-init.json'
    if complete is None:
        if any(any(p.iterdir()) for p in roots.values()): raise ValueError('new_empty_installation_required')
        private = dict(deployment_id=secrets.token_hex(16), revision=1, scope='owned-synthetic',
            host=scope, profile_budget_digest=digest(dict(profiles=profiles,budget=budget)),
            roots={k:[p.stat().st_dev,p.stat().st_ino] for k,p in roots.items()}, seed=secrets.token_hex(32))
        exclusive(record_path, encode(private).encode()); fault('init_record')
    else:
        private = json.loads(read_file(record_path))
        if digest(private) != complete: raise ValueError('matching_initialization_record_required')
    if (set(private) != {'deployment_id','revision','scope','host','profile_budget_digest','roots','seed'} or
            private['revision'] != 1 or private['scope'] != 'owned-synthetic' or private['host'] != scope or
            private['profile_budget_digest'] != digest(dict(profiles=profiles,budget=budget)) or
            private['roots'] != {k:[p.stat().st_dev,p.stat().st_ino] for k,p in roots.items()}):
        raise ValueError('initialization_scope_mismatch')
    public = {k:v for k,v in private.items() if k not in ('seed','roots')}
    lane = hashlib.sha256((private['seed']+'lane').encode()).hexdigest()[:32]
    payload = dict(version=1,lane=lane,records=[])
    files = {
        ('ipc','lifecycle.key'): hashlib.sha256((private['seed']+'lifecycle').encode()).hexdigest().encode(),
        ('ipc','execution.key'): hashlib.sha256((private['seed']+'execution').encode()).hexdigest().encode(),
        ('receipts','journal.json'): encode(dict(payload=payload,sha256=digest(payload))).encode(),
        ('receipts','initialized'): b'',
        ('receipts','lane.lock'): b'',
    }
    allowed = {'control':{'core'},'staging':set(),'ipc':{'lifecycle.key','execution.key'},
               'receipts':{'p1-init.json','p1-core-intent','journal.json','initialized','p1-enrolled.json','lane.lock'}}
    allowed['receipts'].update('p1-created-'+role+'-'+name for role,name in files)
    for role,p in roots.items():
        if {x.name for x in p.iterdir()} - allowed[role]: raise ValueError('unexplained_initialization_entries')
    # Never complete after runtime use or remint an enrolled missing file.
    enrolled = roots['receipts']/'p1-enrolled.json'
    if enrolled.exists():
        raise ValueError('installation_already_enrolled')
    for (role,name),data in files.items():
        path = roots[role]/name
        if path.exists() or path.is_symlink():
            if read_file(path) != data: raise ValueError('partial_initialization_mismatch')
        else:
            intent = roots['receipts']/('p1-created-'+role+'-'+name)
            if intent.exists(): raise ValueError('missing_initialized_file_quarantine')
            exclusive(intent, digest(dict(role=role,name=name,sha256=hashlib.sha256(data).hexdigest())).encode())
            exclusive(path,data)
        fault(name)
    core_path = roots['control']/'core'
    core_intent = roots['receipts']/'p1-core-intent'
    intent_bytes = digest(dict(deployment_id=private['deployment_id'], namespace='control/core')).encode()
    if core_intent.exists():
        if read_file(core_intent) != intent_bytes: raise ValueError('core_intent_quarantine')
        # Refuse before Core can create namespace or schema.
        if (core_path.is_symlink() or not core_path.is_dir() or
                any(not (core_path/name).is_dir() or (core_path/name).is_symlink()
                    for name in ('assets','staging','owners')) or
                not (core_path/'core.sqlite').is_file() or (core_path/'core.sqlite').is_symlink() or
                (core_path/'core.sqlite').stat().st_size == 0):
            raise ValueError('core_namespace_quarantine')
    else:
        if core_path.exists() or core_path.is_symlink(): raise ValueError('core_without_intent_quarantine')
        exclusive(core_intent, intent_bytes)
        fault('core_intent')
    existing_core = core_path.exists()
    if existing_core:
        # A constructor is a writer: validate the retained authority first using
        # the same source-preserving private copy as offline inspection.
        from service.recovery import offline
        from service.reconciliation import COLUMNS
        with offline(roots) as (_, db):
            required = dict(COLUMNS, **persistence.B1_COLUMNS,
                p1_inventory_scans='id observed complete fingerprint summary',
                p1_inventory_objects='id first_observed last_observed scan_id record',
                p1_inventory_history='seq scan_id object_id observed record',
                p1_integrity='id revision state token', sqlite_sequence='name seq')
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables != set(required) or any(
                    [r[1] for r in db.execute('PRAGMA table_info('+t+')')] != columns.split()
                    for t,columns in required.items()):
                raise ValueError('partial_database_initialization_quarantine')
            if (db.execute('SELECT * FROM resource_protocol').fetchall() != [(1,2)] or
                    db.execute('SELECT * FROM resource_monitor').fetchall() != [(1,'healthy')] or
                    db.execute('SELECT * FROM p1_integrity').fetchall() != [(1,1,'healthy',None)] or
                    db.execute('SELECT record FROM p1_installation').fetchall() != [(encode(public),)]):
                raise ValueError('partial_authority_initialization_quarantine')
    core = Core(roots['control']/'core')
    try:
        if any(core.db.execute('SELECT 1 FROM '+t+' LIMIT 1').fetchone() for t in ('jobs','assets','publications','artifacts','revisions')):
            raise ValueError('populated_initialization_refused')
        if existing_core:
            required = set(persistence.B1_COLUMNS) | {'resource_protocol','resource_monitor','p1_integrity','dispatches'}
            tables = {r[0] for r in core.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not required <= tables or core.db.execute('SELECT * FROM resource_protocol').fetchall() != [(1,2)]:
                raise ValueError('partial_database_initialization_quarantine')
        Admission(core,roots['staging'],budget)
        DispatchStore(core)
        persistence.initialize_inventory(core)
        persistence.initialize_enrollment(core,public)
        if (core.db.execute('SELECT * FROM resource_monitor').fetchall()!=[(1,'healthy')] or
            core.db.execute('SELECT * FROM p1_integrity').fetchall()!=[(1,1,'healthy',None)] or
            any(core.db.execute('SELECT 1 FROM '+t+' LIMIT 1').fetchone() for t in
                ('dispatches','dispatch_commands','resource_faults','resource_reservations','p1_registrations','p1_recovery_plans'))):
            raise ValueError('partial_authority_initialization_quarantine')
        from service.reconciliation import Reconciliation
        scan=Reconciliation.__new__(Reconciliation)
        scan.core=core; scan.db=core.db; scan.staging=roots['staging']
        scan.entry_limit=2048; scan.byte_limit=16*1024**2; scan.seconds=1
        scan._collect()
        if scan.reasons: raise ValueError('initialization_inventory_quarantine')
        fault('database')
    finally: core.close()
    sync_dir(roots['control']/'core'); sync_dir(roots['control'])
    exclusive(enrolled,encode(dict(deployment_id=public['deployment_id'],digest=digest(public))).encode())
    fault('enrolled')
    return dict(deployment_id=public['deployment_id'], digest=digest(public), initialization_digest=digest(private))


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('new','complete'))
    parser.add_argument('--request',required=True)
    parser.add_argument('--initialization-digest')
    args=parser.parse_args()
    request=json.loads(read_file(Path(args.request)))
    if not isinstance(request,dict) or set(request)!={'roots','scope','profiles','budget'}:
        raise ValueError('closed_initialization_request')
    if (args.command=='complete') != (args.initialization_digest is not None):
        raise ValueError('explicit_completion_digest_required')
    print(encode(initialize(request['roots'],request['scope'],request['profiles'],request['budget'],complete=args.initialization_digest)))


if __name__=='__main__': main()
