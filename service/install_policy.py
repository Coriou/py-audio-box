"""Closed installed filesystem policy. Selection is trusted entrypoint authority.

These primitives do not enroll, migrate, activate, or infer policy from disk.
The split enrollment transaction must corroborate retained records separately.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import fcntl

SCOPE = 'owned-kokoro-v1'
POLICY = 'docker-kokoro-split-v1'
ROLES = ('control', 'staging', 'receipts', 'ipc')


@dataclass(frozen=True)
class _SplitPolicy:
    scope: str = SCOPE
    name: str = POLICY


SPLIT = _SplitPolicy()


def require(policy):
    # No caller-supplied UID map, equivalent object, or metadata-selected string.
    if policy is not SPLIT:
        raise ValueError('explicit_fixed_install_policy_required')


def expectation(policy, role, *, directory=False):
    require(policy)
    if role not in ROLES:
        raise ValueError('exact_volume_roles_required')
    if directory:
        return {'control': (10001, 10001, 0o700),
                'receipts': (0, 0, 0o2700),
                'ipc': (0, 10001, 0o750),
                'staging': (0, 0, 0o1733)}[role]
    if role == 'staging':
        # Attempt directories have a separate allocation policy, never this one.
        raise ValueError('attempt_policy_required')
    return {'control': (10001, 10001, 0o600),
            'receipts': (0, 0, 0o600),
            'ipc': (0, 10001, 0o640)}[role]


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
            info.st_gid, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def check(info, policy, role, *, directory=False):
    uid, gid, mode = expectation(policy, role, directory=directory)
    if ((info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (uid, gid, mode) or
            not (stat.S_ISDIR(info.st_mode) if directory else
                 stat.S_ISREG(info.st_mode) and info.st_nlink == 1)):
        raise ValueError('installed_role_identity')


@contextmanager
def parent_fd(path):
    """Walk the literal absolute chain; never resolve away a symlink first."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path.name in ('', '.', '..'):
        raise ValueError('literal_absolute_install_path_required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd, path.name
    finally:
        os.close(fd)


def read_private(path, limit, *, policy, role):
    require(policy)
    if type(limit) is not int or limit < 0 or limit > 16 * 1024**2:
        raise ValueError('installed_read_budget')
    with parent_fd(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            check(before, policy, role)
            if before.st_size > limit:
                raise ValueError('installed_read_budget')
            chunks = []; size = 0
            while size <= before.st_size:
                data = os.read(fd, min(65536, before.st_size + 1 - size))
                if not data:
                    break
                chunks.append(data); size += len(data)
            if (size != before.st_size or identity(os.fstat(fd)) != identity(before) or
                    identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != identity(before)):
                raise ValueError('private_file_changed')
        finally:
            os.close(fd)
    # Also reject a renamed ancestor between descriptor traversal and return.
    with parent_fd(path) as (parent, name):
        if identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != identity(before):
            raise ValueError('private_file_changed')
    return b''.join(chunks)


def roots_checked(roots, *, policy):
    require(policy)
    if not isinstance(roots, dict) or set(roots) != set(ROLES):
        raise ValueError('exact_volume_roles_required')
    result = {}; seen = set()
    for role, value in roots.items():
        path = Path(value)
        with parent_fd(path) as (parent, name):
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                info = os.fstat(fd); check(info, policy, role, directory=True)
                if identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != identity(info):
                    raise ValueError('installed_root_changed')
            finally:
                os.close(fd)
        key = info.st_dev, info.st_ino
        if key in seen:
            raise ValueError('distinct_volume_roles_required')
        seen.add(key); result[role] = path
    if any(a in b.parents for a in result.values() for b in result.values() if a != b):
        raise ValueError('nested_volume_roles')
    return result


def directory_identity(path, *, policy, role):
    """Fixed role directory, including a literal no-follow ancestor walk."""
    require(policy)
    with parent_fd(path) as (parent,name):
        fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
        try:
            info=os.fstat(fd);check(info,policy,role,directory=True)
            if identity(os.stat(name,dir_fd=parent,follow_symlinks=False))!=identity(info):
                raise ValueError('installed_root_changed')
            return identity(info)
        finally:os.close(fd)


@contextmanager
def locked_file(path, *, policy, role):
    """Existing role lock only. Keep exclusion through the final identity check.

    No creation, ownership repair, or arbitrary UID exception. The caller owns
    ordering when taking more than one lock (control before receipts).
    """
    require(policy)
    with parent_fd(path) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            check(before, policy, role)
            if identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != identity(before):
                raise ValueError('installed_lock_changed')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('maintenance_lock_busy') from None
            def revalidate():
                check(os.fstat(fd), policy, role)
                with parent_fd(path) as (current, leaf):
                    if (identity(os.fstat(fd)) != identity(before) or
                            identity(os.stat(leaf, dir_fd=current, follow_symlinks=False)) != identity(before)):
                        raise ValueError('installed_lock_changed')
            revalidate()
            try:
                yield fd
            except BaseException as primary:
                try:
                    revalidate()
                except BaseException as cleanup:
                    primary.add_note('lock revalidation failed: '+str(cleanup))
                raise
            else:
                revalidate()
        finally:
            os.close(fd)
