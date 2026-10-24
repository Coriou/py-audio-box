"""Installed control's private launch notification boundary.

No listener, Core owner, Docker authority, or public gate-publishing API lives
here. Notification is corroborated against SQLite and live identity elsewhere.
"""
import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

from service import install_policy as ip
from service.core import encode
from service.persistent_init import digest

GATE_DIRECTORY = Path('/tmp/speech-box-activation')
GATE_NAME = 'gate.json'
GATE_LIMIT = 16384


def projection(value):
    from service.recovery import closed, validate_registration
    closed(value, 'version registration_id registration configuration_digest enrollment_digest provenance_digest processes')
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('installed_gate_version')
    if len(encode(value).encode()) > GATE_LIMIT:
        raise ValueError('installed_gate_budget')
    for key in ('registration_id', 'configuration_digest', 'enrollment_digest', 'provenance_digest'):
        if not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
            raise ValueError('installed_gate_digest')
    record = value['registration']; validate_registration(record)
    if (digest(record) != value['registration_id'] or record['evidence'] != 'actual-host-inspection' or
            len(record['containers']) != 2):
        raise ValueError('installed_gate_registration')
    if type(value['processes']) is not list or len(value['processes']) != 2:
        raise ValueError('installed_gate_processes')
    seen = set(); namespaces = set()
    owners = {c['id']: c for c in record['containers']}
    if {c['role'] for c in owners.values()} != {'control', 'worker'}:
        raise ValueError('installed_gate_roles')
    for process in value['processes']:
        closed(process, 'id image started pid start_ticks pid_namespace cgroup')
        owner = owners.get(process['id'])
        if owner is None or process['id'] in seen or any(process[k] != owner[k] for k in ('id','image','started')):
            raise ValueError('installed_gate_process_binding')
        if any(type(process[k]) is not int or not 1 <= process[k] < 2**64 for k in ('pid','start_ticks','pid_namespace')):
            raise ValueError('installed_gate_process_identity')
        if (process['pid_namespace'] in namespaces or process['cgroup'] not in
                ('/docker/'+process['id'], '/system.slice/docker-'+process['id']+'.scope')):
            raise ValueError('installed_gate_process_identity')
        namespaces.add(process['pid_namespace']); seen.add(process['id'])
    return value


def decode_projection(raw):
    from service.recovery import host_json
    if type(raw) is not bytes or len(raw) > GATE_LIMIT:
        raise ValueError('installed_gate_budget')
    value = projection(host_json(raw))
    if raw != encode(value).encode():
        raise ValueError('installed_gate_encoding')
    return value


def _control_identity():
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (10001,)*4:
        raise ValueError('installed_control_identity')
    if os.getgroups() not in ([], [10001]):
        raise ValueError('installed_control_groups')


def _directory(path):
    # The fixed parent is the exact bounded /tmp tmpfs checked by the host.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        if (not stat.S_ISDIR(s.st_mode) or (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode)) != (10001,10001,0o700) or
                ip.identity(os.stat(path, follow_symlinks=False)) != ip.identity(s)):
            raise ValueError('installed_gate_directory')
        return fd
    except BaseException:
        os.close(fd); raise


def _directory_stable(fd, path):
    # Directory timestamps legitimately change on exclusive publication.
    a = os.fstat(fd); b = os.stat(path, follow_symlinks=False)
    if (a.st_uid,a.st_gid,stat.S_IMODE(a.st_mode)) != (10001,10001,0o700):
        raise ValueError('installed_gate_directory_changed')
    fields = ('st_dev','st_ino','st_mode','st_uid','st_gid')
    if any(getattr(a,k) != getattr(b,k) for k in fields):
        raise ValueError('installed_gate_directory_changed')


def publish(raw, *, directory=GATE_DIRECTORY):
    """Fixed exec helper only: bounded stdin, exclusive file, no path/action input."""
    _control_identity(); decode_projection(raw)
    fd = _directory(directory)
    try:
        out = os.open(GATE_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        with os.fdopen(out, 'wb') as stream:
            s = os.fstat(stream.fileno())
            if (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode),s.st_nlink) != (10001,10001,0o600,1):
                raise ValueError('installed_gate_file')
            if stream.write(raw) != len(raw): raise OSError('installed_gate_short_write')
            stream.flush(); os.fsync(stream.fileno())
            identity = ip.identity(os.fstat(stream.fileno()))
        os.fsync(fd)
        if ip.identity(os.stat(GATE_NAME,dir_fd=fd,follow_symlinks=False)) != identity:
            raise ValueError('installed_gate_replaced')
        _directory_stable(fd,directory)
    finally:
        os.close(fd)


class WaitingControl:
    """One incarnation, one notification; no stale-directory reuse or repair."""
    def __init__(self, *, directory=GATE_DIRECTORY, seconds=30):
        _control_identity()
        if type(seconds) not in (int,float) or not 0 < seconds <= 30:
            raise ValueError('installed_wait_budget')
        self.directory = Path(directory); self.deadline = time.monotonic()+seconds
        self.directory.mkdir(mode=0o700)  # Existing gate/directory always refuses.
        self.fd = _directory(self.directory)
        self.saved = None

    def read(self):
        _directory_stable(self.fd,self.directory)
        fd = os.open(GATE_NAME,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,dir_fd=self.fd)
        try:
            s = os.fstat(fd)
            if (not stat.S_ISREG(s.st_mode) or (s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode),s.st_nlink) != (10001,10001,0o600,1) or
                    not 1 <= s.st_size <= GATE_LIMIT):
                raise ValueError('installed_gate_file')
            raw = os.read(fd,GATE_LIMIT+1)
            identity = ip.identity(s)
            if (ip.identity(os.fstat(fd)) != identity or
                    ip.identity(os.stat(GATE_NAME,dir_fd=self.fd,follow_symlinks=False)) != identity):
                raise ValueError('installed_gate_replaced')
            value = decode_projection(raw)
            pair = raw,identity
            if self.saved is not None and self.saved != pair:
                raise ValueError('installed_gate_replaced')
            self.saved = pair
            _directory_stable(self.fd,self.directory)
            return value
        finally:
            os.close(fd)

    def wait(self):
        while time.monotonic() < self.deadline:
            try: return self.read()
            except FileNotFoundError:
                # A missing directory is replacement, not an absent notification.
                _directory_stable(self.fd,self.directory)
                time.sleep(.02)
        raise ValueError('installed_activation_timeout')

    def close(self):
        os.close(self.fd)  # Retain notification; never erase failed evidence.


def local_incarnation():
    """Current PID1 in the private PID/cgroup namespaces; no host /proc access."""
    if sys.platform != 'linux' or os.getpid() != 1:
        raise ValueError('installed_control_pid1_required')
    def read(name):
        with open('/proc/1/'+name,'rb') as stream: raw = stream.read(65537)
        if len(raw)>65536: raise ValueError('installed_proc_budget')
        return raw.decode('ascii')
    def capture():
        raw = read('stat'); fields = raw[raw.rfind(')')+2:].split()
        return dict(start_ticks=int(fields[19]),pid_namespace=os.stat('/proc/1/ns/pid').st_ino,
                    cgroup=read('cgroup').strip())
    result = capture()
    if result != capture() or result['cgroup'] != '0::/':
        raise ValueError('installed_control_incarnation')
    return result


def checked_waiting_gate(waiting, configuration, runtime, *, root='/control/core'):
    """Pre-owner check only. The caller must repeat binding after Core acquisition.

    Holding the existing control lock excludes registrars/owners through all
    private copies and live brackets. This function never constructs an owner.
    """
    from service.recovery import installed_startup_read
    _control_identity()
    gate = waiting.wait()
    with ip.locked_file(Path(root)/'control.lock',policy=ip.SPLIT,role='control'):
        before = local_incarnation()
        hello = runtime.call('hello', {})
        checked = installed_startup_read(root,configuration,gate,hello,before,policy=ip.SPLIT)
        if (runtime.call('hello', {}) != hello or local_incarnation() != before or
                waiting.read() != gate):
            raise ValueError('installed_startup_bracket_changed')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('notify',))
    parser.parse_args()
    try:
        publish(sys.stdin.buffer.read(GATE_LIMIT+1))
    except (OSError,ValueError,TypeError,KeyError):
        # No raw request, credential, path, or protected provenance in logs.
        raise SystemExit('installed notification refused') from None


if __name__ == '__main__':
    main()
