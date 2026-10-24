"""Installed Pocket private-media bridge; no DB, public paths or permissions repair.

A nonresident UID10001 helper opens one protocol-validated managed asset under
one fixed read-only mount. The supervisor reaps it before trusting the descriptor.
Only that O_RDONLY descriptor is inherited by the isolated inference child.
"""
import array
import fcntl
import os
import re
import socket
import stat
import subprocess
import sys
from worker.transport import Refused

ROOT = '/references/assets'


def asset_id(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9_-]{1,256}', value):
        raise Refused('managed_reference_identity')
    return value


def check_fd(fd, *, owner_uid=10001):
    s = os.fstat(fd)
    if (not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or
            (s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode)) != (owner_uid, owner_uid, 0o600) or
            fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
        raise Refused('managed_reference_descriptor')
    return s


def open_reference(identity):
    from worker.engines.pocket import _open
    identity = asset_id(identity)
    root = _open(ROOT, directory=True, owner_uid=10001)
    try:
        s = os.fstat(root)
        if (s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode)) != (10001, 10001, 0o700):
            raise Refused('managed_reference_root')
        fd = os.open(identity, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=root)
        try: check_fd(fd)
        except BaseException:
            os.close(fd)
            raise
        return fd
    finally:
        os.close(root)


def acquire(identity):
    """Finite exact helper, no listener; failures retain unknown process handles."""
    identity = asset_id(identity)
    parent, child = socket.socketpair()
    fd = None
    proc = None
    try:
        parent.settimeout(2)
        proc = subprocess.Popen(['python', '-I', '-B', '-c',
            "import sys;sys.path.insert(0,'/opt/speech');from worker.managed_reference import main;main()",
            str(child.fileno()), identity], user=10001, group=10001, extra_groups=[],
            close_fds=True, pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child.close()
        data, ancillary, flags, _ = parent.recvmsg(1, socket.CMSG_SPACE(array.array('i').itemsize), socket.MSG_CMSG_CLOEXEC)
        received = []
        for level, kind, raw in ancillary:
            if (level, kind) == (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                values = array.array('i'); values.frombytes(raw); received.extend(values)
        if len(received) != 1 or data != b'R' or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
            for value in received: os.close(value)
            raise Refused('managed_reference_transfer')
        fd = received[0]
        if proc.wait(timeout=1) != 0: raise Refused('managed_reference_helper')
        check_fd(fd)
        result, fd = fd, None
        return result
    except BaseException as primary:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill(); proc.wait(timeout=1)
            except BaseException:
                _retained.append(proc)
                raise Refused('managed_reference_helper_unreaped') from primary
        raise
    finally:
        if fd is not None: os.close(fd)
        parent.close(); child.close()


_retained = []


def main():
    if (len(sys.argv) != 3 or
            (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (10001,)*4 or os.getgroups()):
        raise Refused('managed_reference_helper_identity')
    fd = open_reference(sys.argv[2])
    try:
        with socket.socket(fileno=int(sys.argv[1])) as connection:
            if connection.sendmsg([b'R'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))]) != 1:
                raise Refused('managed_reference_transfer')
    finally:
        os.close(fd)
