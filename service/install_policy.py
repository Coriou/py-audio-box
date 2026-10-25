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
POCKET_POLICY = 'docker-pocket-split-v1'
POCKET_SCOPE = 'owned-pocket-v1'
SCOPES = (SCOPE, POCKET_SCOPE)


def pocket(config):
    return config['ownership_policy'] == POCKET_POLICY


def scope_for(config):
    return POCKET_SCOPE if pocket(config) else SCOPE
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


def _closed(value, fields):
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError('closed_installed_record')


def configuration(value):
    """Detached public configuration. No local paths, credentials or seed."""
    import json
    import re
    from service.core import encode
    from service.admission import finite_budget
    from contracts import validate as oracle
    from worker.runtime_protocol import execution
    _closed(value, 'version ownership_policy profiles budget contexts bundles images mounts')
    if len(encode(value).encode()) > 48000:
        raise ValueError('installed_configuration_budget')
    value = json.loads(encode(value))
    if type(value['version']) is not int or value['version'] != 1 or value['ownership_policy'] not in (POLICY, POCKET_POLICY):
        raise ValueError('installed_configuration_version')
    finite_budget(value['budget'])
    profiles = value['profiles']
    if type(profiles) is not list or len(profiles) != 1 or profiles[0]['id'] != ('pocket-cpu' if pocket(value) else 'kokoro-cpu'):
        raise ValueError('installed_profiles')
    oracle.validate('profile', profiles[0])
    if profiles[0]['device'] != 'cpu' or profiles[0]['topology'] != 'networkless_container':
        raise ValueError('installed_profile_topology')
    if pocket(value) and {(c['operation'], c['mode'], c['identity']['engine']) for c in profiles[0]['compatibility']} != {('voice_prepare','prepare','pocket-tts'),('speech_generate','pocket_clone','pocket-tts')}:
        raise ValueError('installed_pocket_capabilities')
    _closed(value['images'], 'control worker')
    if any(not isinstance(v, str) or not re.fullmatch('sha256:[0-9a-f]{64}', v) for v in value['images'].values()):
        raise ValueError('installed_image_digest')
    if value['mounts'] != dict(control='/control', staging='/staging', receipts='/receipts', ipc='/ipc', models='/models'):
        raise ValueError('installed_mount_mapping')
    contexts = value['contexts']; bundles = value['bundles']
    if type(contexts) is not dict or not 1 <= len(contexts) <= 32 or type(bundles) is not dict or len(bundles) != 1:
        raise ValueError('installed_contexts')
    for principal, context in contexts.items():
        if not isinstance(principal, str) or not re.fullmatch('[a-zA-Z0-9_-]{1,128}', principal):
            raise ValueError('installed_principal')
        spec = execution(context)
        if spec is None or context['plan']['resolved']['profile'] != oracle.profile_snapshot(profiles[0]):
            raise ValueError('installed_context_profile')
        if pocket(value):
            if spec.engine != 'pocket-tts' or context['execution']['references'] != {spec.reference_id:'/references/assets/'+spec.reference_id}:
                raise ValueError('installed_pocket_context')
        elif spec.engine != 'kokoro':
            raise ValueError('installed_kokoro_context')
        oracle.frozen_plan(context['plan'], context['bundles'])
        expected = dict(manifest_sha256=spec.manifest_sha256, checkpoint=spec.checkpoint,
                        revision=spec.revision, path='/models/bundles/'+spec.bundle_id)
        if bundles != {spec.bundle_id: expected}:
            raise ValueError('installed_bundle_binding')
        if (spec.memory_bytes > value['budget']['ram_bytes'] or
                spec.max_output_bytes*2 > value['budget']['disk_bytes'] or value['budget']['pids'] < 48):
            raise ValueError('installed_context_budget')
    return value


def installation(value):
    """Shared closed parser; legacy remains byte/shape compatible."""
    import re
    from service.persistent_init import digest, validate_scope
    if type(value) is not dict:
        raise ValueError('closed_installation')
    if value.get('scope') == 'owned-synthetic':
        _closed(value, 'deployment_id revision scope host profile_budget_digest')
        hashes = [value['profile_budget_digest']]
    elif value.get('scope') in SCOPES:
        _closed(value, 'deployment_id revision scope host configuration configuration_digest ownership_policy roles'+(' launch' if 'launch' in value else ''))
        if 'launch' in value:
            from service import daemon_volumes as dv
            dv.binding(value['launch'])
            if (value['host'] != dv.scope(value['launch']) or value['configuration_digest'] != value['launch']['configuration_digest'] or
                    value['ownership_policy'] != POLICY):
                raise ValueError('daemon_enrollment_binding')
        config = configuration(value['configuration'])
        if (value['scope'] != scope_for(config) or value['ownership_policy'] != config['ownership_policy'] or value['configuration_digest'] != digest(config)):
            raise ValueError('installed_configuration_digest')
        _closed(value['roles'], 'control staging receipts ipc')
        seen = set()
        for role, record in value['roles'].items():
            _closed(record, 'device inode uid gid mode')
            if (any(type(v) is not int or v < 0 for v in record.values()) or record['inode'] == 0 or
                    (record['uid'], record['gid'], record['mode']) != expectation(SPLIT, role, directory=True)):
                raise ValueError('installed_role_record')
            key = record['device'], record['inode']
            if key in seen: raise ValueError('distinct_volume_roles_required')
            seen.add(key)
        hashes = [value['configuration_digest']]
    else:
        raise ValueError('unsupported_enrollment')
    validate_scope(value['host'])
    if value['scope'] in SCOPES and any(v['driver'] != 'local' for v in value['host']['volumes'].values()):
        raise ValueError('local_named_volumes_required')
    if (type(value['revision']) is not int or value['revision'] != 1 or
            not isinstance(value['deployment_id'], str) or not re.fullmatch('[0-9a-f]{32}', value['deployment_id']) or
            any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes)):
        raise ValueError('invalid_installation_identity')
    return value


def role_records(roots):
    result = {}
    for role, path in roots_checked(roots, policy=SPLIT).items():
        s = directory_identity(path, policy=SPLIT, role=role)
        result[role] = dict(device=s[0], inode=s[1], uid=s[3], gid=s[4], mode=stat.S_IMODE(s[2]))
    return result


def initialization(value):
    """Closed split private record, kept exclusively in receipts."""
    import re
    _closed(value, 'public seed')
    installation(value['public'])
    if (value['public']['scope'] not in SCOPES or not isinstance(value['seed'],str) or
            not re.fullmatch('[0-9a-f]{64}',value['seed'])):
        raise ValueError('invalid_split_initialization')
    return value


HOST_POLICY = 'installed-local-docker-v1'
CONFIG_DESTINATION = '/etc/speech-box/install.json'
CREDENTIAL_DESTINATION = '/etc/speech-box/control.key'


def host_policy(value, config, scope):
    """Protected per-install authority, never selected from observed metadata.

    CPU is in integer nanocpus; worker allocation covers every enrolled thread.
    Tmpfs bytes/inodes are explicit, finite, and charged within memory budget.
    Source records contain identity and public hashes, never credential bytes.
    """
    from service.persistent_init import digest, validate_scope
    import re
    config = configuration(config); validate_scope(scope)
    _closed(value, 'version kind configuration_digest host roots inputs cpu tmpfs control_port')
    if (type(value['version']) is not int or value['version'] != 1 or
            value['kind'] != HOST_POLICY or value['configuration_digest'] != digest(config) or
            value['host'] != scope):
        raise ValueError('installed_host_policy_binding')
    _closed(value['roots'], 'control staging receipts ipc')
    _closed(value['inputs'], 'models config credential')
    _closed(value['cpu'], 'control worker'); _closed(value['tmpfs'], 'control worker')
    if type(value['control_port']) is not int or not 1 <= value['control_port'] <= 65535:
        raise ValueError('installed_control_port')
    for role, root in value['roots'].items():
        _closed(root, 'source device inode')
        _source_record(root)
    for name, source in value['inputs'].items():
        _closed(source, 'source device inode mode uid gid size mtime_ns ctime_ns sha256' if name != 'credential'
                else 'source device inode mode uid gid size mtime_ns ctime_ns')
        _source_record(source)
        if any(type(source[k]) is not int or source[k] < 0 for k in ('size','mtime_ns','ctime_ns')):
            raise ValueError('installed_input_identity')
        if any(type(source[k]) is not int for k in ('uid','gid','mode')):
            raise ValueError('installed_input_identity')
        expected = {'models': (10001, 10001, 0o555), 'config': (0, 10001, 0o440),
                    'credential': (0, 10001, 0o640)}[name]
        if (source['uid'], source['gid'], source['mode']) != expected:
            raise ValueError('installed_input_policy')
        if name != 'credential' and (not isinstance(source['sha256'], str) or
                not re.fullmatch('[0-9a-f]{64}', source['sha256'])):
            raise ValueError('installed_input_hash')
    for role in ('control', 'worker'):
        cpu = value['cpu'][role]
        if type(cpu) is not int or not 1 <= cpu <= 1024 * 10**9:
            raise ValueError('installed_cpu_budget')
        tmpfs = value['tmpfs'][role]; _closed(tmpfs, 'bytes inodes')
        if (any(type(n) is not int for n in tmpfs.values()) or
                not 1 <= tmpfs['bytes'] <= config['budget']['ram_bytes'] or
                not 1 <= tmpfs['inodes'] <= 65536):
            raise ValueError('installed_tmpfs_budget')
    threads = max(c['execution']['descriptor']['threads'] for c in config['contexts'].values())
    if value['cpu']['worker'] < threads * 10**9:
        raise ValueError('installed_worker_cpu_threads')
    return value


def _source_record(value):
    path = value['source']
    if (not isinstance(path, str) or not path.startswith('/') or
            str(Path(path)) != path or '..' in Path(path).parts or path == '/' or
            any(type(value[k]) is not int or not 1 <= value[k] < 2**64 for k in ('device', 'inode'))):
        raise ValueError('installed_source_record')


def host_enrollment(public, policy, config, scope):
    """Corroborate closed retained public enrollment against protected authority."""
    from service.persistent_init import digest
    installation(public); host_policy(policy,config,scope)
    if 'launch' in public: raise ValueError('strict_host_launch_policy_required')
    if (public['scope']!=scope_for(config) or public['host']!=scope or
            public['configuration']!=config or public['configuration_digest']!=policy['configuration_digest']):
        raise ValueError('installed_expected_enrollment_binding')
    for role,source in policy['roots'].items():
        if (public['roles'][role]['device'],public['roles'][role]['inode'])!=(source['device'],source['inode']):
            raise ValueError('installed_expected_enrollment_roots')
    return digest(public)


def family_configuration(value, family=None):
    if family is None:
        return configuration(value)
    from service import installed_export as ex
    if family is not ex.FAMILY:
        raise ValueError('explicit_installed_family_required')
    return ex.configuration(value)


def family_installation(value, family=None):
    if family is None:
        return installation(value)
    from service import installed_export as ex
    if family is not ex.FAMILY:
        raise ValueError('explicit_installed_family_required')
    return ex.installation(value)
