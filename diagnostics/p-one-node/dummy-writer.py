"""Non-product qualification PID1. Fixed tiny files; no child or product import."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
import uuid

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dummy_finite', D / 'finite-driver.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
CAP = 1024 * 1024
SIZE = 4096
BINARY = bytes(range(256)) * 4


def payload(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    result = hashlib.sha256(raw).hexdigest().encode() + b'\n' + raw + b'\n'
    F.require(len(result) <= SIZE, 'STOP dummy record size')
    return result.ljust(SIZE, b'\0')


class Proof:
    def __init__(self, root):
        self.root, self.written = Path(root), 0
    def write(self, name, raw, *, update=False):
        F.require(self.written + len(raw) <= CAP, 'STOP dummy write counter')
        self.written += len(raw)
        path = self.root / name
        if update:
            fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
            try:
                s = os.fstat(fd)
                F.require(stat.S_ISREG(s.st_mode) and s.st_size == SIZE and s.st_nlink == 1
                          and stat.S_IMODE(s.st_mode) == 0o600, 'STOP dummy slot identity')
                F.require(os.pwrite(fd, raw, 0) == len(raw), 'STOP dummy partial update')
                os.fsync(fd)
            finally:
                os.close(fd)
        else:
            with F.private(path) as f:
                F.require(f.write(raw) == len(raw), 'STOP dummy short write')
                os.fsync(f.fileno())


def body(root, token, invocation):
    """Real IO core, HOST uses a fresh private directory; no identity bypass in CLI."""
    proof = Proof(root)
    proof.write('pab-terminal.slot', payload(dict(state='PENDING', token=token, invocation=invocation)))
    proof.write('retention-sentinel', token.encode() + b'\x00\xff\x80\r\n')
    proof.write('dummy.bin', BINARY)
    final = dict(state='TERMINAL', token=token, invocation=invocation, qualification=True,
                 acceptance=False, product_dispatched=False, descendants=[],
                 bytes_written=proof.written + SIZE, write_cap=CAP,
                 identity=dict(pid=os.getpid(), uid=os.getuid(), gid=os.getgid(),
                               python=sys.version, optimize=sys.flags.optimize))
    proof.write('pab-terminal.slot', payload(final), update=True)
    F.require(F.slot_read(F.read_regular(Path(root) / 'pab-terminal.slot', SIZE, 0o600)) == final,
              'STOP dummy readback')
    return final


def main():
    F.enabled()
    F.require(len(sys.argv) == 3, 'STOP fixed dummy argv')
    token, invocation = sys.argv[1:]
    F.require(re.fullmatch('[0-9a-f]{48}', token) is not None and str(uuid.UUID(invocation)) == invocation,
              'STOP dummy owner')
    F.require(os.getpid() == 1 and os.getuid() == os.getgid() == 10001, 'STOP dummy PID1 UID')
    os.umask(0o077)
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, 30)
    root = Path('/tmp')
    F.initialized(root)
    F.verify_source()  # hashes source as bytes; never imports product/runtime identity
    final = body(root, token, invocation)
    F.require(time.monotonic() - start < 30 and final['bytes_written'] <= CAP, 'STOP dummy deadline')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
