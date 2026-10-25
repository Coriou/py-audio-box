"""Daemon-side, create-only Kokoro policy generation. Never provisions or starts Docker."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

from service import host_install as h, install_policy as ip
from service.core import encode
from service.persistent_init import digest, validate_scope


def generate(request, *, docker=None):
    """Inspect only explicit paths/volumes; return no credential bytes or readiness."""
    if sys.platform != 'linux' or os.getuid() != 0 or os.geteuid() != 0:
        raise ValueError('setup_run_as_root_inside_linux_daemon_host_or_vm')
    h.recovery.closed(request, 'layout daemon_root configuration scope inputs cpu tmpfs control_port')
    if request['layout'] not in ('linux-local', 'intel-mac-linux-vm'):
        raise ValueError('setup_layout_requires_linux_local_or_intel_mac_linux_vm')
    inputs = request['inputs']; h.recovery.closed(inputs, 'models config credential')
    scope = validate_scope(request['scope'])
    raw = h._public_file(inputs['config'], uid=0, gid=10001, mode=0o440,
                         limit=65536, deadline=time.monotonic()+5)
    config = ip.configuration(h.recovery.host_json(raw))
    if config != request['configuration'] or ip.pocket(config):
        raise ValueError('setup_explicit_kokoro_configuration_required_no_fallback')
    docker = docker or h.Docker(scope['context'])
    # Includes exact local socket, Linux daemon ID and local volume metadata.
    root = request['daemon_root']
    if not isinstance(root, str) or not root.startswith('/') or root == '/' or str(Path(root)) != root or '..' in Path(root).parts:
        raise ValueError('setup_exact_daemon_root_required')
    actual = docker._read(['info','--format','{{.DockerRootDir}}'])
    if actual != root: raise ValueError('setup_daemon_root_mismatch')
    architecture = docker._read(['info','--format','{{.Architecture}}'])
    if architecture not in ('x86_64','amd64'): raise ValueError('setup_x86_64_linux_daemon_required')
    volumes = h._host_scope(docker, scope)
    if any(Path(root) not in Path(v['source']).parents for v in volumes):
        raise ValueError('setup_volumes_outside_daemon_root')
    if any(v['driver'] != 'local' for v in volumes):
        raise ValueError('setup_plain_local_volumes_required')
    roots = {v['role']: {k:v[k] for k in ('source','device','inode')} for v in volumes}
    for record in roots.values():
        info = h._source_stat(record['source'], directory=True)
        if (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (0,0,0o700):
            raise ValueError('setup_fresh_roots_require_root_root_0700')
        if any(Path(record['source']).iterdir()):
            raise ValueError('setup_fresh_empty_roots_required')
    records = {}
    for name, path in inputs.items():
        s = h._source_stat(path, directory=name == 'models')
        records[name] = dict(source=path, device=s.st_dev, inode=s.st_ino, uid=s.st_uid,
            gid=s.st_gid, mode=stat.S_IMODE(s.st_mode), size=s.st_size,
            mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
        if name != 'credential':
            records[name]['sha256'] = (hashlib.sha256(raw).hexdigest() if name == 'config' else
                h._model_tree(path, config, time.monotonic()+5))
    policy = dict(version=1, kind=ip.HOST_POLICY, configuration_digest=digest(config),
        host=scope, roots=roots, inputs=records, cpu=request['cpu'], tmpfs=request['tmpfs'],
        control_port=request['control_port'])
    ip.host_policy(policy, config, scope)
    # Existing exact mount/source/ownership/credential/daemon/consumer fences.
    # Check even empty roots: no implicit chown, adoption, reminting or repair.
    observer = h.InstalledObserver(docker, config, scope, policy)
    observer.initialize_authority()
    for role in ('control', 'worker'):
        if docker._read(['image','inspect',config['images'][role],'--format','{{.Id}}']) != config['images'][role]:
            raise ValueError('setup_exact_local_image_required_no_pull')
    observer.initialize_authority()
    if (docker._read(['info','--format','{{.DockerRootDir}}']) != root or
            docker._read(['info','--format','{{.Architecture}}']) != architecture):
        raise ValueError('setup_daemon_metadata_changed')
    return dict(phase='configuration-only', layout=request['layout'], configuration=config,
                scope=scope, host_policy=policy)


def write_bundle(result, output):
    """One exclusive public result file; no success output on failed validation."""
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encode(result).encode()); stream.flush(); os.fsync(stream.fileno())


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Inspect explicit CPU inputs on the Linux daemon host/VM; never install/start/pull.')
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    # Only fixed actionable categories cross this CLI; arbitrary exceptions may contain secrets.
    try:
        fd = os.open(args.request, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('setup_request_regular_file_required')
            raw = stream.read(65537)
        if len(raw) > 65536: raise ValueError('setup_request_budget')
        result = generate(h.recovery.host_json(raw))
        write_bundle(result, args.output)
    except (Exception, KeyboardInterrupt) as exc:
        code = str(exc)
        allowed = ('setup_', 'installed_', 'docker_', 'unsupported_', 'owned_', 'distinct_', 'nested_')
        safe = code if code.startswith(allowed) and code.replace('_','').isalnum() else 'setup_refused_retained_inputs_require_review'
        raise SystemExit(safe) from None
    print('configuration-only; no installation or runtime qualification')


if __name__ == '__main__': main()
