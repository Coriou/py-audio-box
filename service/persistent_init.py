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


def read_file(path, limit=LIMIT, *, policy=None, role=None):
    if policy is not None:
        from service.install_policy import read_private
        return read_private(path, limit, policy=policy, role=role)
    if role is not None:
        raise ValueError('role_requires_fixed_policy')
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


def roots_checked(roots, *, policy=None):
    if policy is not None:
        from service.install_policy import roots_checked as installed_roots
        return installed_roots(roots, policy=policy)
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
            if persistence.store_format(db)!=1:
                raise ValueError('installed_native_publication_v2_unsupported')
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
    core = Core(roots['control']/'core', _publication_creation=1)
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



# Internal installed initialization. No CLI/activation or host-registration adapter.
def _split_core(public, control, staging, *, family=None):
    """Only entered after the child privilege boundary (injected in source tests)."""
    if (os.getuid(),os.geteuid(),os.getgid(),os.getegid()) != (10001,)*4 or os.getgroups():
        raise ValueError('permanent_control_identity_required')
    from service import install_policy as ip, installed_export as ex
    ip.family_installation(public, family)
    checked = ex.expectation(public, family) if family is ex.FAMILY else None
    # Creation is exclusive. Existing roots must be fully enrolled exact stores;
    # unsupported interruption inside Core construction remains quarantined.
    core = Core(Path(control)/'core', _publication_creation=2 if checked else 1,
                _installed_family=checked)
    try:
        Admission(core, Path(staging), public['configuration']['budget'])
        DispatchStore(core)
        persistence.initialize_inventory(core)
        persistence.initialize_enrollment(core, public)
        if checked: ex.database(core.db, public, fresh=True)
    finally:
        core.close()
    sync_dir(Path(control)/'core'); sync_dir(Path(control))


def _linux_split_child():
    """Fresh exec: stdin is bounded PUBLIC data; no seed or receipts descriptor."""
    import sys
    if sys.platform != 'linux' or os.getuid() != 0 or os.geteuid() != 0:
        raise ValueError('linux_root_initializer_required')
    raw = sys.stdin.buffer.read(65537)
    if len(raw) > 65536: raise ValueError('initializer_pipe_budget')
    request = json.loads(raw)
    from service import install_policy as ip
    from service.install_child import prepare
    ip._closed(request, 'public control_fd staging_fd')
    control, staging = prepare(request['public'], request['control_fd'], request['staging_fd'])
    os.setgroups([])
    os.setresgid(10001, 10001, 10001)
    os.setresuid(10001, 10001, 10001)
    os.umask(0o077)
    if os.getresuid() != (10001,)*3 or os.getresgid() != (10001,)*3 or os.getgroups():
        raise ValueError('permanent_control_identity_required')
    _split_core(request['public'], control, staging)


def _launch_split_child(raw):
    """Synchronous launcher retains parent exclusions until exit AND reap."""
    import subprocess
    import sys
    if sys.platform != 'linux' or os.geteuid() != 0:
        raise ValueError('linux_root_initializer_required')
    if len(raw) > 65536: raise ValueError('initializer_pipe_budget')
    request=json.loads(raw)
    from service.install_policy import _closed
    _closed(request,'public control_fd staging_fd')
    descriptors=(request['control_fd'],request['staging_fd'])
    if any(type(fd) is not int or fd<3 for fd in descriptors) or len(set(descriptors))!=2:
        raise ValueError('closed_initializer_descriptors')
    # Only two role fds survive exec; receipts, exclusion fds and seed do not.
    child = subprocess.Popen([sys.executable, '-I', '-B', '-c',
        'import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent.parent))+');'
        'from service.persistent_init import _linux_split_child;_linux_split_child()'],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, pass_fds=descriptors, cwd='/', env={})
    try:
        child.communicate(raw, timeout=30)
    except BaseException as primary:
        try: child.kill()
        except BaseException as cleanup: primary.add_note('child kill failed: '+str(cleanup))
        while child.returncode is None:
            try: child.wait()
            except (InterruptedError,KeyboardInterrupt) as cleanup:
                primary.add_note('child reap interrupted: '+str(cleanup))
        raise
    if child.returncode != 0: raise ValueError('split_child_failed')


def _split_create(path, data, role, *, parent_identity=None):
    """Exclusive, fd-checked durable creation. No replacement or chown repair."""
    from service import install_policy as ip
    with ip.parent_fd(path) as (parent, name):
        observed=os.fstat(parent)
        if parent_identity is not None and (observed.st_dev,observed.st_ino) != parent_identity:
            raise ValueError('installed_root_changed')
        fd = os.open(name, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            uid, gid, mode = ip.expectation(ip.SPLIT, role)
            # New empty inode only; never an existing data file.
            os.fchown(fd, uid, gid); os.fchmod(fd, mode)
            before = os.fstat(fd); ip.check(before, ip.SPLIT, role)
            with os.fdopen(os.dup(fd), 'wb') as stream:
                persistence.write_complete(stream, data)
            after = os.fstat(fd); ip.check(after, ip.SPLIT, role)
            if ((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or
                    ip.identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != ip.identity(after)):
                raise ValueError('installed_creation_changed')
            os.fsync(fd); os.fsync(parent)
        finally: os.close(fd)


def initialize_split(roots, scope, config, *, authority, policy, complete=None,
                     launch_child=_launch_split_child, fault=lambda _: None, daemon_binding=None, family=None):
    """Create-only internal transaction; authority is a trusted host inspection callback.

    The callback must freshly inspect daemon/context/named-volume identities and
    consumers while the caller's host launch exclusion is held. It returns only
    the closed observation below; this unit does not implement Docker observation.
    Injected launchers are synchronous: return only after child close/reap.
    """
    import fcntl
    from contextlib import ExitStack
    from service import install_policy as ip
    ip.require(policy); validate_scope(scope); config = ip.family_configuration(config, family)
    from service import installed_export as ex
    export = family is ex.FAMILY
    if export and (daemon_binding is None or launch_child is _launch_split_child):
        raise ValueError('export_private_initializer_binding_required')
    if daemon_binding is not None:
        from service import daemon_volumes as dv
        (ex.binding if export else dv.binding)(daemon_binding)
        if scope != (ex.host_scope if export else dv.scope)(daemon_binding) or digest(config) != daemon_binding['configuration_digest'] or config['ownership_policy'] != (ex.POLICY if export else ip.POLICY):
            raise ValueError('daemon_initialization_binding')
    if launch_child is _launch_split_child:
        import sys
        if sys.platform != 'linux' or os.geteuid() != 0:
            raise ValueError('linux_root_initializer_required')
    if type(roots) is not dict or set(roots) != set(ROLES): raise ValueError('exact_volume_roles_required')
    roots = {k: Path(v) for k,v in roots.items()}
    with ExitStack() as stack:
        held = {}; original = {}; roles_ready = False
        # Directory locks exist before any file. Fixed order includes control first.
        for role in ('control', 'receipts', 'ipc', 'staging'):
            with ip.parent_fd(roots[role]) as (parent, name):
                fd = os.open(name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=parent)
                stack.callback(os.close, fd)
                info = os.fstat(fd)
                if (info.st_dev,info.st_ino) in original.values():
                    raise ValueError('distinct_non_nested_roots_required')
                try: fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: raise ValueError('initialization_lock_busy') from None
                info = os.fstat(fd)
                if ip.identity(os.stat(name,dir_fd=parent,follow_symlinks=False)) != ip.identity(info):
                    raise ValueError('installed_root_changed')
                held[role] = fd; original[role] = (info.st_dev, info.st_ino)
        if len(set(original.values())) != 4 or any(a in b.parents for a in roots.values() for b in roots.values() if a != b):
            raise ValueError('distinct_non_nested_roots_required')
        def stable():
            for role, fd in held.items():
                with ip.parent_fd(roots[role]) as (parent, name):
                    s = os.stat(name,dir_fd=parent,follow_symlinks=False)
                    if (s.st_dev,s.st_ino) != original[role] or (os.fstat(fd).st_dev,os.fstat(fd).st_ino) != original[role]:
                        raise ValueError('installed_root_changed')
                    if roles_ready:
                        ip.check(s,policy,role,directory=True)
                        ip.check(os.fstat(fd),policy,role,directory=True)
        def observe():
            observation = authority()
            ip._closed(observation, 'host roots consumers')
            if (observation['host'] != scope or observation['roots'] != {k:list(v) for k,v in original.items()} or
                    observation['consumers'] != []):
                raise ValueError('fresh_host_authority_required')
            stable()
        def create(path,data,role):
            stable()
            _split_create(path,data,role,parent_identity=original[role])
            stable()
        observe()
        receipts = roots['receipts']; record_path = receipts/'p1-init.json'
        def read(path, role): return read_file(path, policy=policy, role=role)
        if complete is None:
            if any(ip.directory_entries(fd) for fd in held.values()): raise ValueError('new_empty_installation_required')
            for fd in held.values():
                s = os.fstat(fd)
                if (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)) != (0,0,0o700):
                    raise ValueError('fresh_root_identity_required')
            target = {role:dict(device=dev,inode=ino,uid=ip.expectation(policy,role,directory=True)[0],
                gid=ip.expectation(policy,role,directory=True)[1],mode=ip.expectation(policy,role,directory=True)[2])
                for role,(dev,ino) in original.items()}
            public = dict(deployment_id=daemon_binding['installation'] if export else secrets.token_hex(16), revision=1, scope=ex.SCOPE if export else ip.scope_for(config), host=scope,
                configuration=config, configuration_digest=digest(config), ownership_policy=config['ownership_policy'], roles=target)
            if daemon_binding is not None: public['launch'] = daemon_binding
            private = dict(public=public, seed=secrets.token_hex(32))
            if export: ex.installation(public)
            else: ip.initialization(private)
            # Receipts is still root:root 0700 here. Intent precedes ANY transition.
            create(record_path, encode(private).encode(), 'receipts'); fault('init_record')
        else:
            raw_private = read(record_path, 'receipts')
            private = json.loads(raw_private)
            if raw_private != encode(private).encode() or hashlib.sha256(raw_private).hexdigest() != complete: raise ValueError('matching_initialization_record_required')
            ip._closed(private, 'public seed'); public = private['public']
        if export:
            ex.installation(public)
            ip._closed(private, 'public seed')
        else: ip.initialization(private)
        if (public.get('launch') != daemon_binding or public['host'] != scope or encode(public['configuration']) != encode(config) or
                any((r['device'],r['inode']) != original[k] for k,r in public['roles'].items()) or
                not isinstance(private['seed'],str) or len(private['seed']) != 64):
            raise ValueError('initialization_scope_mismatch')
        if (receipts/'p1-enrolled.json').exists(): raise ValueError('installation_already_enrolled')
        # Only empty role inodes (receipts contains the initial intent) transition.
        for role, fd in held.items():
            uid,gid,mode = ip.expectation(policy,role,directory=True)
            s = os.fstat(fd)
            if (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)) != (uid,gid,mode):
                if ((s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)) != (0,0,0o700) or
                        set(ip.directory_entries(fd)) != ({'p1-init.json'} if role == 'receipts' else set())):
                    raise ValueError('partial_role_transition_quarantine')
                os.fchown(fd,uid,gid); os.fchmod(fd,mode); os.fsync(fd)
            ip.check(os.fstat(fd),policy,role,directory=True); stable(); fault('role_'+role)
        roles_ready = True
        if ip.role_records(roots) != public['roles']: raise ValueError('installed_root_changed')
        lane = hashlib.sha256((private['seed']+'lane').encode()).hexdigest()[:32]
        payload = dict(version=1,lane=lane,records=[])
        files = {('ipc','lifecycle.key'):hashlib.sha256((private['seed']+'lifecycle').encode()).hexdigest().encode(),
            ('ipc','execution.key'):hashlib.sha256((private['seed']+'execution').encode()).hexdigest().encode(),
            ('receipts','journal.json'):encode(dict(payload=payload,sha256=digest(payload))).encode(),
            ('receipts','initialized'):b'', ('receipts','lane.lock'):b''}
        allowed = dict(control={'core'}, staging=set(), ipc={'lifecycle.key','execution.key'},
            receipts={'p1-init.json','p1-enrolled.json','p1-core-intent','p1-core-done','journal.json','initialized','lane.lock'})
        allowed['receipts'].update('p1-created-'+r+'-'+n for r,n in files)
        for role,fd in held.items():
            if set(ip.directory_entries(fd))-allowed[role]: raise ValueError('unexplained_initialization_entries')
        for (role,name),data in files.items():
            path = roots[role]/name; intent = receipts/('p1-created-'+role+'-'+name)
            binding = digest(dict(role=role,name=name,sha256=hashlib.sha256(data).hexdigest())).encode()
            if path.exists() or path.is_symlink():
                if read(path,role) != data or read(intent,'receipts') != binding: raise ValueError('partial_initialization_mismatch')
            else:
                if intent.exists() or intent.is_symlink(): raise ValueError('missing_initialized_file_quarantine')
                create(intent,binding,'receipts'); fault('intent_'+name)
                create(path,data,role)
            stable(); fault(name)
        # Hold lane exclusion through child reap and final verification. Core owns
        # its lifetime lock during construction; parent holds every root lock.
        stack.enter_context(ip.locked_file(receipts/'lane.lock',policy=policy,role='receipts'))
        completion_lock = False
        core_intent = receipts/'p1-core-intent'; done = receipts/'p1-core-done'
        binding = digest(public).encode()
        if core_intent.exists() or core_intent.is_symlink():
            if read(core_intent,'receipts') != binding:
                raise ValueError('partial_core_quarantine')
            if export and not done.exists() and not done.is_symlink():
                # Only a complete exact fresh store can finish the lost child
                # reply. Never reconstruct missing enrollment or partial DDL.
                stack.enter_context(ip.locked_file(roots['control']/'core'/'control.lock',policy=policy,role='control'))
                completion_lock = True
                _validate_split_database(roots, public, policy, family=family)
                create(done,binding,'receipts')
            elif not done.exists() or read(done,'receipts') != binding:
                raise ValueError('partial_core_quarantine')
        else:
            if ((roots['control']/'core').exists() or (roots['control']/'core').is_symlink() or done.exists() or done.is_symlink()):
                raise ValueError('core_without_intent_quarantine')
            create(core_intent,binding,'receipts'); fault('core_intent')
            request = encode(dict(public=public,control_fd=held['control'],staging_fd=held['staging'])).encode()
            if len(request)>65536: raise ValueError('initializer_pipe_budget')
            launch_child(request)  # MUST be synchronous, including failed reap.
            stable(); fault('child_reaped')
            create(done,binding,'receipts')
        if not completion_lock:
            stack.enter_context(ip.locked_file(roots['control']/'core'/'control.lock',policy=policy,role='control'))
        _validate_split_database(roots, public, policy, family=family)
        observe(); fault('database')
        if read(record_path,'receipts') != encode(private).encode():
            raise ValueError('initialization_record_changed')
        for (role,name),data in files.items():
            if read(roots[role]/name,role) != data: raise ValueError('partial_initialization_mismatch')
        if read(core_intent,'receipts') != binding or read(done,'receipts') != binding:
            raise ValueError('partial_core_quarantine')
        create(receipts/'p1-enrolled.json',encode(dict(deployment_id=public['deployment_id'],digest=digest(public))).encode(),'receipts')
        stable(); fault('enrolled')
        return dict(deployment_id=public['deployment_id'],digest=digest(public),initialization_digest=digest(private))


def _validate_split_database(roots, public, policy, *, family=None):
    """Private-copy schema/authority/inventory validation, never original SQLite."""
    from service.recovery import private_database
    from service.reconciliation import COLUMNS, Reconciliation
    from types import SimpleNamespace
    from contextlib import closing
    import sqlite3
    with private_database(roots['control']/'core'/'core.sqlite', policy=policy) as copied, closing(sqlite3.connect(copied.as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('PRAGMA trusted_schema=OFF'); db.execute('PRAGMA query_only=ON')
        import time
        deadline=time.monotonic()+2
        db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
        if family is not None:
            from service import installed_export as ex
            ex.expectation(public, family)
            ex.database(db, public, fresh=True)
            ex.fresh_namespace(roots)
            return
        expected = dict(COLUMNS, **persistence.B1_COLUMNS,
            p1_inventory_scans='id observed complete fingerprint summary',
            p1_inventory_objects='id first_observed last_observed scan_id record',
            p1_inventory_history='seq scan_id object_id observed record',
            p1_integrity='id revision state token',sqlite_sequence='name seq')
        if {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} != set(expected):
            raise ValueError('partial_database_initialization_quarantine')
        for table,columns in expected.items():
            if [r[1] for r in db.execute('PRAGMA table_info('+table+')')] != columns.split():
                raise ValueError('partial_database_initialization_quarantine')
        fixed = dict(p1_format=[(1,'synthetic-core')], voice_catalogue_format=[(1,)], resource_accounting=[(1,0)], resource_protocol=[(1,2)],
            resource_monitor=[(1,'healthy')], p1_integrity=[(1,1,'healthy',None)],
            p1_installation=[(public['deployment_id'],1,public['scope'],encode(public),digest(public))])
        for table in expected:
            rows=db.execute('SELECT * FROM '+table+' LIMIT 2').fetchall()
            if rows != fixed.get(table,[]): raise ValueError('nonfresh_split_database:'+table)
        if db.execute('PRAGMA user_version').fetchone() != (1,): raise ValueError('unsupported_versions')
        scan=Reconciliation.__new__(Reconciliation)
        scan.core=SimpleNamespace(root=roots['control']/'core'); scan.db=db; scan.staging=roots['staging']
        scan.entry_limit=2048; scan.byte_limit=16*1024**2; scan.seconds=1
        scan._collect(policy=policy)
        if scan.reasons: raise ValueError('initialization_inventory_quarantine:'+repr(scan.reasons))


if __name__=='__main__': main()
