"""Real bounded UDS lifecycle adapter. No host operator or daemon authority."""
import json
import os
import stat
import sys
from pathlib import Path
import socket
import time
from worker.lifecycle import Observation, State
from worker.runtime_protocol import VERSION, session, authenticate, send, receive
from service.worker_transport import TransportObservation


class SupervisorRuntime:
    def __init__(self, ipc='/ipc', interrupt='none', *, family=None, configuration=None):
        if family is not None or configuration is not None:
            from service import installed_export as ex, installed_pocket_daemon as pd, install_policy as ip
            pd.require(family)
            if family is None or configuration is None: raise ValueError('explicit_installed_family_required')
            configuration = ip.family_configuration(configuration, family)
        self.family = family
        self.configuration = configuration
        self.ipc = Path(ipc)
        self.interrupt = interrupt

    def call(self, op, body, timeout=1.5):
        if not 0 < timeout <= 2:
            raise ValueError('bounded_communication_required')
        cutoff = time.monotonic()+timeout
        secret = (self.ipc/'lifecycle.key').read_text().strip()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(self.ipc/'lifecycle.sock'))
            authenticate(connection, session(secret, 0, 10001), cutoff, control=True)
            send(connection, dict(version=VERSION, op=op, body=body), cutoff)
            if self.interrupt == op+'-sent':
                import os
                os._exit(17)
            result = receive(connection, cutoff)
        if set(result) != {'ok', 'value'} or result['ok'] is not True:
            raise RuntimeError('lifecycle_refused_or_quarantined')
        value = result['value']
        if self.family is not None and op in ('hello', 'startup'):
            from service.host_install import _registration_hello
            _registration_hello(value['hello'] if op == 'startup' else value,
                                family=self.family, configuration=self.configuration)
        return value

    def prepare(self, authority, fixture='silence'):
        return self.call('prepare', dict(snapshot=json.loads(authority._snapshot),
            binding=authority.binding, fixture=fixture))['runtime_id']

    def start(self, identity):
        return self.call('start', dict(runtime_id=identity))

    def observe(self, runtime_id, timeout):
        value = self.call('observe', dict(runtime_id=runtime_id), min(timeout, 2))
        if set(value) != {'runtime_id', 'state'} or value['runtime_id'] != runtime_id:
            raise ValueError('unbound_observation')
        return Observation(runtime_id, State(value['state']))

    def terminate(self, runtime_id, timeout):
        self.call('stop', dict(runtime_id=runtime_id), min(timeout, 2))

    def kill(self, runtime_id, timeout):
        # Escalation remains the parent watchdog's fixed grace, never client signals.
        self.terminate(runtime_id, timeout)

    def collect(self, authority, runtime_id):
        value = self.call('collect', dict(runtime_id=runtime_id))
        receipt = value['receipt']
        if receipt is None:
            return None
        from worker.native_protocol import terminal_receipt, frozen_native
        if frozen_native(authority.context): raise ValueError('native_transport_required')
        terminal_receipt(receipt,authority=authority,runtime_id=runtime_id)
        proposal = value['proposal']
        if proposal is None:
            return receipt, None
        authority.check(proposal['message'], 'result')
        from contracts import validate as oracle
        oracle.validate('result_manifest', proposal['manifest'])
        message, manifest = proposal['message'], proposal['manifest']
        for key in ('job_id','attempt_id','generation','owner_id','input_sha256','staging_id'):
            if manifest[key] != message[key]:
                raise ValueError('manifest_tuple')
        return receipt, TransportObservation('proposal', 'durable_parent_receipt', None,
                                            message, manifest, time.monotonic())


def installed_hello_at(ipc_fd, *, timeout=1.5, startup=False, family=None, configuration=None):
    """Hello only, for a permanently dropped host helper with a validated fd.

    The caller owns child creation/reap. This never drops the current process,
    reads Core, or exposes keys. Authentication retains the existing peer pair.
    """
    from service import install_policy
    if (sys.platform != 'linux' or os.getuid() != 10001 or os.geteuid() != 10001 or
            os.getgid() != 10001 or os.getegid() != 10001 or os.getgroups() != []):
        raise ValueError('dropped_control_identity_required')
    if type(ipc_fd) is not int or type(timeout) not in (int, float) or not 0 < timeout <= 2:
        raise ValueError('bounded_communication_required')
    root = os.fstat(ipc_fd)
    install_policy.check(root, install_policy.SPLIT, 'ipc', directory=True)
    cutoff = time.monotonic() + timeout
    fd = os.open('lifecycle.key', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=ipc_fd)
    try:
        before = os.fstat(fd)
        install_policy.check(before, install_policy.SPLIT, 'ipc')
        if before.st_size != 64:
            raise ValueError('installed_lifecycle_key')
        raw = os.read(fd, 65)
        if (len(raw) != 64 or install_policy.identity(os.fstat(fd)) != install_policy.identity(before) or
                install_policy.identity(os.stat('lifecycle.key', dir_fd=ipc_fd, follow_symlinks=False)) != install_policy.identity(before)):
            raise ValueError('installed_lifecycle_key_changed')
        secret = raw.decode('ascii')
        if any(c not in '0123456789abcdef' for c in secret):
            raise ValueError('installed_lifecycle_key')
    finally:
        os.close(fd)
    endpoint = os.stat('lifecycle.sock', dir_fd=ipc_fd, follow_symlinks=False)
    if (not stat.S_ISSOCK(endpoint.st_mode) or endpoint.st_uid != 0 or endpoint.st_gid != 10001 or
            stat.S_IMODE(endpoint.st_mode) != 0o660 or endpoint.st_nlink != 1):
        raise ValueError('installed_lifecycle_socket')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        # AF_UNIX resolves this magic-link path in this helper's own proc view.
        # Unlike SQLite, connect does not canonicalize it back through Docker's
        # inaccessible host ancestors. The inherited directory remains open;
        # key/socket identity checks and authenticated server peer checks bracket it.
        connection.connect('/proc/self/fd/'+str(ipc_fd)+'/lifecycle.sock')
        authenticate(connection, session(secret, 0, 10001), cutoff, control=True)
        send(connection, dict(version=VERSION, op='startup' if startup else 'hello', body={}), cutoff)
        result = receive(connection, cutoff)
    if (install_policy.identity(os.fstat(ipc_fd)) != install_policy.identity(root) or
            install_policy.identity(os.stat('lifecycle.sock', dir_fd=ipc_fd, follow_symlinks=False)) != install_policy.identity(endpoint)):
        raise ValueError('installed_ipc_changed')
    if (not isinstance(result, dict) or set(result) != {'ok', 'value'} or result['ok'] is not True or
            not isinstance(result['value'], dict) or len(json.dumps(result['value'], allow_nan=False).encode()) > 65536):
        raise RuntimeError('lifecycle_refused_or_quarantined')
    value = result['value']
    if startup:
        from service import daemon_volumes as dv
        dv.closed(value,'hello local')
        dv.local(value['local'],'worker',family=family)
        from service.host_install import _registration_hello
        _registration_hello(value['hello'], family=family, configuration=configuration)
        return value
    if family is not None or configuration is not None:
        from service.host_install import _registration_hello
        return _registration_hello(value, family=family, configuration=configuration)
    if (set(value) != {'lane', 'boot', 'quarantined', 'launches'} or
            any(not isinstance(value[k], str) or len(value[k]) != 32 or
                any(c not in '0123456789abcdef' for c in value[k]) for k in ('lane', 'boot')) or
            type(value['quarantined']) is not bool or type(value['launches']) is not int or
            not 0 <= value['launches'] <= 64):
        raise ValueError('installed_hello_shape_or_maintenance')
    return value
