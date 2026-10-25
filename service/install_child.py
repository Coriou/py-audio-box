"""Private Linux initializer mount view. No host mount or general path interface.

Only the fresh exec child calls LinuxMounts. Tests inject namespace operations;
those tests establish pathname semantics, never kernel permission qualification.
"""
import ctypes
import os
from pathlib import Path
import stat
import sys

BASE = Path('/mnt')
CONTROL = BASE / 'pab-install/control'
STAGING = BASE / 'pab-install/staging'


class LinuxMounts:
    def __init__(self):
        if sys.platform != 'linux' or os.getuid() != 0 or os.geteuid() != 0:
            raise ValueError('linux_root_initializer_required')
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.unshare.argtypes = [ctypes.c_int]
        self.libc.unshare.restype = ctypes.c_int
        self.libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]
        self.libc.mount.restype = ctypes.c_int

    def _check(self, result):
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    def unshare(self):
        self._check(self.libc.unshare(0x00020000))  # CLONE_NEWNS only

    def mount(self, source, target, kind, flags, data=None):
        def b(value): return None if value is None else os.fsencode(value)
        self._check(self.libc.mount(b(source), b(target), b(kind), flags, b(data)))

    def mkdir(self, path):
        os.mkdir(path, 0o755)

    def stat(self, path):
        return os.stat(path, follow_symlinks=False)


def prepare(public, control_fd, staging_fd, *, ops=None):
    """Build fixed canonical paths before UID drop; failure kills this helper.

    No chroot, recursive bind, caller mount argument, host mkdir, or unmount.
    The namespace and its mounts disappear when this sole helper exits.
    """
    from service import install_policy as ip
    ip.installation(public)
    descriptors = (control_fd, staging_fd)
    if any(type(fd) is not int or fd < 3 for fd in descriptors) or len(set(descriptors)) != 2:
        raise ValueError('closed_initializer_descriptors')
    identities = []
    for role, fd in zip(('control', 'staging'), descriptors):
        info = os.fstat(fd)
        ip.check(info, ip.SPLIT, role, directory=True)
        record = public['roles'][role]
        identity = (info.st_dev, info.st_ino)
        if identity != (record['device'], record['inode']):
            raise ValueError('installed_root_changed')
        identities.append(identity)
    if identities[0] == identities[1]:
        raise ValueError('installed_root_alias')
    ops = LinuxMounts() if ops is None else ops
    # Refuse missing/symlink/non-root-owned mountpoint. No creation on host.
    info = ops.stat(BASE)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError('unsafe_initializer_mountpoint')
    ops.unshare()
    ops.mount(None, '/', None, 0x4000 | 0x40000)  # MS_REC | MS_PRIVATE
    # Private bounded tmpfs hides /mnt only, leaving interpreter/modules intact.
    ops.mount('tmpfs', BASE, 'tmpfs', 2 | 4, 'mode=0755,size=64k,nr_inodes=16')
    ops.mkdir(BASE / 'pab-install')
    for fd, target, identity in zip(descriptors, (CONTROL, STAGING), identities):
        ops.mkdir(target)
        ops.mount('/proc/self/fd/' + str(fd), target, None, 4096)  # MS_BIND
        info = ops.stat(target)
        if (info.st_dev, info.st_ino) != identity or not stat.S_ISDIR(info.st_mode):
            raise ValueError('initializer_bind_identity_changed')
    return CONTROL, STAGING


def drop_control():
    """Only in a fresh synchronous exec child; verify all saved identity slots."""
    os.setgroups([])
    os.setresgid(10001, 10001, 10001)
    os.setresuid(10001, 10001, 10001)
    os.umask(0o077)
    if os.getresuid() != (10001,)*3 or os.getresgid() != (10001,)*3 or os.getgroups():
        raise ValueError('permanent_control_identity_required')


def registration_writer(public, registration, prestate, control, lock_fd, *, fault=lambda _:None, family=None):
    """No DDL, Core constructor, receipt access, generic SQL or action input."""
    import sqlite3
    from service import install_policy as ip, recovery
    from service.core import encode, sync_dir
    from service.persistent_init import digest
    if ((os.getuid(),os.geteuid(),os.getgid(),os.getegid()) != (10001,)*4 or
            os.getgroups()):
        raise ValueError('permanent_control_identity_required')
    from service import installed_export as ex, persistence
    export = family is ex.FAMILY
    ip.family_installation(public, family)
    if export: ex.bound_registration(registration, public)
    else: recovery.validate_registration(registration)
    if (public['scope'] not in ((ex.SCOPE,) if export else ip.SCOPES) or registration['scope'] != public['host'] or
            registration['deployment_id'] != public['deployment_id'] or
            registration['evidence'] != (ex.EVIDENCE if export else ('daemon-volumes-inspection-v1' if 'launch' in public else 'actual-host-inspection'))):
        raise ValueError('installed_registration_binding')
    if type(prestate) is not list or len(encode(prestate).encode()) > recovery.HOST_CAPSULE_BYTES:
        raise ValueError('installed_registration_prestate')
    root=Path(control)/'core'
    ip.directory_identity(root,policy=ip.SPLIT,role='control')
    if export: ex.expectation(public, family).check_root(root)
    held=os.fstat(lock_fd); ip.check(held,ip.SPLIT,'control')
    if ip.identity(held)!=ip.identity(os.stat(root/'control.lock',follow_symlinks=False)):
        raise ValueError('installed_inherited_lock_changed')
    if export:
        # An independent open must observe inherited exclusion, not acquire it.
        import fcntl
        probe = os.open(root/'control.lock', os.O_RDONLY|os.O_NOFOLLOW)
        try:
            try: fcntl.flock(probe, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: pass
            else: raise ValueError('installed_inherited_lock_not_held')
        finally: os.close(probe)
    before={}
    for name in ('core.sqlite','core.sqlite-wal','core.sqlite-shm'):
        path=root/name
        try: info=os.stat(path,follow_symlinks=False)
        except FileNotFoundError:
            if name=='core.sqlite': raise
            continue
        ip.check(info,ip.SPLIT,'control'); before[name]=(info.st_dev,info.st_ino)
    db=sqlite3.connect((root/'core.sqlite').as_uri()+'?mode=rw',uri=True,timeout=0,isolation_level=None)
    try:
        db.execute('PRAGMA trusted_schema=OFF'); db.execute('PRAGMA synchronous=FULL')
        import time
        deadline=time.monotonic()+2
        db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
        def validate():
            if export:
                ex.expectation(public, family).check_root(root)
                if (ip.identity(os.fstat(lock_fd)) != ip.identity(held) or
                        ip.identity(os.stat(root/'control.lock',follow_symlinks=False)) != ip.identity(held)):
                    raise ValueError('installed_inherited_lock_changed')
                for name in ('core.sqlite','core.sqlite-wal','core.sqlite-shm'):
                    try: info=os.stat(root/name,follow_symlinks=False)
                    except FileNotFoundError:
                        if name in before: raise ValueError('installed_writer_database_changed')
                        continue
                    ip.check(info,ip.SPLIT,'control')
                    if name in before and before[name] != (info.st_dev,info.st_ino):
                        raise ValueError('installed_writer_database_changed')
                ex.database(db, public)
            elif persistence.store_format(db) != 1:
                raise ValueError('installed_writer_v1_required')
        validate()
        db.execute('BEGIN IMMEDIATE')
        try:
            validate()
            rows=db.execute('SELECT id,revision,scope,record,digest FROM p1_installation LIMIT 2').fetchall()
            expected=[(public['deployment_id'],1,public['scope'],encode(public),digest(public))]
            if rows!=expected: raise ValueError('installed_writer_installation_changed')
            count,size=db.execute('SELECT count(*),coalesce(sum(length(record)),0) FROM p1_registrations').fetchone()
            if count>recovery.HOST_OWNER_LIMIT or size>recovery.HOST_CAPSULE_BYTES:
                raise ValueError('installed_registration_budget')
            rows=db.execute('SELECT id,record,digest FROM p1_registrations ORDER BY rowid').fetchall()
            if encode([list(r) for r in rows])!=encode(prestate): raise ValueError('installed_writer_prestate_changed')
            rid=digest(registration); row=(rid,encode(registration),rid)
            found=[r for r in rows if r[0]==rid]
            if found and found!=[row]: raise ValueError('installed_writer_registration_changed')
            fault('preinsert')
            if export: validate()
            if not found:
                if count>=recovery.HOST_OWNER_LIMIT or size+len(row[1].encode())>recovery.HOST_CAPSULE_BYTES:
                    raise ValueError('installed_registration_budget')
                db.execute('INSERT INTO p1_registrations VALUES(?,?,?)',row)
            fault('inserted')
            validate()
            fault('precommit')
            validate()
            db.execute('COMMIT'); fault('committed')
        except BaseException:
            if db.in_transaction: db.execute('ROLLBACK')
            raise
    finally: db.close()
    for name in ('core.sqlite','core.sqlite-wal','core.sqlite-shm'):
        try: info=os.stat(root/name,follow_symlinks=False)
        except FileNotFoundError:
            if name=='core.sqlite': raise
            continue
        ip.check(info,ip.SPLIT,'control')
        if name in before and before[name]!=(info.st_dev,info.st_ino):
            raise ValueError('installed_writer_database_changed')
    if ip.identity(os.fstat(lock_fd))!=ip.identity(held): raise ValueError('installed_inherited_lock_changed')
    sync_dir(root); sync_dir(Path(control))


def registration_entry(kind):
    """Two fixed entrypoints, bounded public stdin, no operation selected by stdin."""
    import json
    from service import install_policy as ip
    from service.core import encode
    if sys.platform!='linux' or os.getuid()!=0 or os.geteuid()!=0:
        raise ValueError('linux_root_initializer_required')
    raw=sys.stdin.buffer.read(65537)
    if len(raw)>65536: raise ValueError('installed_child_request_budget')
    request=json.loads(raw)
    if kind=='hello':
        ip._closed(request,'ipc_fd')
        fd=request['ipc_fd']
        if type(fd) is not int or fd<3: raise ValueError('installed_child_descriptor')
        ip.check(os.fstat(fd),ip.SPLIT,'ipc',directory=True)
        drop_control()
        from service.supervisor_runtime import installed_hello_at
        print(encode(installed_hello_at(fd)))
    elif kind=='writer':
        ip._closed(request,'public registration prestate control_fd staging_fd lock_fd')
        ip.installation(request['public'])
        from service import recovery
        recovery.validate_registration(request['registration'])
        descriptors=[request[k] for k in ('control_fd','staging_fd','lock_fd')]
        if any(type(fd) is not int or fd<3 for fd in descriptors) or len(set(descriptors))!=3:
            raise ValueError('installed_child_descriptor')
        control,_=prepare(request['public'],*descriptors[:2])
        drop_control()
        registration_writer(request['public'],request['registration'],request['prestate'],control,descriptors[2])
    else: raise ValueError('installed_child_entrypoint')


CONTAINER_CONTROL = Path('/control')
CONTAINER_STAGING = Path('/staging')


def prepare_container(public, control_fd, staging_fd):
    """Existing fixed mount view; never unshare/mount/chroot or host paths.

    Only the daemon helper's Core/writer phases mount these roots. Models and
    credentials are absent from that helper, not hidden by pathname convention.
    """
    from service import install_policy as ip
    from service import daemon_volumes as dv
    ip.installation(public)
    if public.get('launch',{}).get('policy')!=dv.POLICY:
        raise ValueError('explicit_daemon_container_view_required')
    descriptors=(control_fd,staging_fd)
    if any(type(fd) is not int or fd<3 for fd in descriptors) or len(set(descriptors))!=2:
        raise ValueError('closed_initializer_descriptors')
    for role,fd in zip(('control','staging'),descriptors):
        info=os.fstat(fd); ip.check(info,ip.SPLIT,role,directory=True)
        record=public['roles'][role]
        if (info.st_dev,info.st_ino)!=(record['device'],record['inode']): raise ValueError('installed_root_changed')
        path=CONTAINER_CONTROL if role=='control' else CONTAINER_STAGING
        with ip.parent_fd(path) as (parent,name):
            actual=os.stat(name,dir_fd=parent,follow_symlinks=False)
            if (actual.st_dev,actual.st_ino)!=(info.st_dev,info.st_ino) or not stat.S_ISDIR(actual.st_mode):
                raise ValueError('container_view_identity_changed')
    return CONTAINER_CONTROL,CONTAINER_STAGING


def close_other_descriptors(keep=()):
    # Linux-only fresh exec. Enumeration is bounded; unexpected excess refuses.
    names=os.listdir('/proc/self/fd')
    if len(names)>64: raise ValueError('container_child_descriptor_budget')
    for name in names:
        fd=int(name)
        if fd>2 and fd not in keep:
            try: os.close(fd)
            except OSError as exc:
                if exc.errno!=9: raise


def verify_dropped_capabilities():
    fields=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if (any(int(fields[k].strip(),16) for k in ('CapInh','CapPrm','CapEff','CapAmb')) or
            fields['NoNewPrivs'].strip()!='1'):
        raise ValueError('container_child_retained_capabilities')


def container_entry(kind):
    """Three finite trusted entrypoints; no arbitrary action/path/SQL protocol."""
    import json
    from service import install_policy as ip
    from service.core import encode
    if sys.platform!='linux' or os.getuid()!=0 or os.geteuid()!=0:
        raise ValueError('linux_root_initializer_required')
    raw=sys.stdin.buffer.read(65537)
    if len(raw)>65536: raise ValueError('installed_child_request_budget')
    request=json.loads(raw)
    if kind=='hello':
        ip._closed(request,'ipc_fd')
        fd=request['ipc_fd']
        if type(fd) is not int or fd<3: raise ValueError('installed_child_descriptor')
        ip.check(os.fstat(fd),ip.SPLIT,'ipc',directory=True)
        close_other_descriptors((fd,)); drop_control(); verify_dropped_capabilities()
        from service.supervisor_runtime import installed_hello_at
        print(encode(installed_hello_at(fd,startup=True)))
        return
    if kind=='core': ip._closed(request,'public control_fd staging_fd')
    elif kind=='writer': ip._closed(request,'public registration prestate control_fd staging_fd lock_fd')
    else: raise ValueError('installed_child_entrypoint')
    control,staging=prepare_container(request['public'],request['control_fd'],request['staging_fd'])
    keep=(request['lock_fd'],) if kind=='writer' else ()
    if keep and (type(keep[0]) is not int or keep[0]<3 or keep[0] in (request['control_fd'],request['staging_fd'])):
        raise ValueError('installed_child_descriptor')
    close_other_descriptors(keep); drop_control(); verify_dropped_capabilities()
    if kind=='core':
        from service.persistent_init import _split_core
        _split_core(request['public'],control,staging)
    else: registration_writer(request['public'],request['registration'],request['prestate'],control,keep[0])
