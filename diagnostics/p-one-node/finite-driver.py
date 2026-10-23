"""Task-specific finite PID1 and bounded proof readers; no product imports here."""
import hashlib
import contextlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import selectors
import stat
import subprocess
import sys
import tarfile
import threading
import time

GIB = 1024 ** 3
LOGICAL = GIB
LEAF = 64 * 1024 ** 2
ENTRIES = 200000
EXPORT = 2 * GIB
META = 16 * 1024 ** 2
SLOT = 512 * 1024
SEED = b'pab-p-one-local-r16\x00\xff\x80\n'
ENTRY = ['python', '/app/diagnostics/p-one-node/entry.py', '--one-node']
NODE = 'tests/service/test_recovery.py::test_t1_real_blocked_pipe_backpressure_deadline'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def enabled():
    require(__debug__ and sys.flags.optimize == 0 and not os.environ.get('PYTHONOPTIMIZE'),
            'STOP optimization')


def canonical(path):
    p = Path(os.path.abspath(path))
    cursor = Path(p.anchor)
    for part in p.parts[1:]:
        cursor /= part
        s = cursor.lstat()
        require(not stat.S_ISLNK(s.st_mode), 'STOP symlink component')
    return p


def private(path):
    canonical(Path(path).parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_IMODE(os.fstat(fd).st_mode) == 0o600, 'STOP private mode')
        return os.fdopen(fd, 'wb', buffering=0)
    except BaseException:
        os.close(fd)
        raise


def read_regular(path, cap, mode=None):
    canonical(Path(path).parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'STOP regular bound')
        if mode is not None:
            require(stat.S_IMODE(before.st_mode) == mode, 'STOP mode')
        out = bytearray()
        while True:
            b = os.read(fd, min(65536, cap + 1 - len(out)))
            if not b:
                break
            out.extend(b)
            require(len(out) <= cap, 'STOP read cap')
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                and len(out) == before.st_size, 'STOP unstable read')
        return bytes(out)
    finally:
        os.close(fd)


def slot_create(path):
    with private(path) as f:
        # Real bounded writes initialize the reserve on both Darwin HOST and Linux.
        require(f.write(b'\0' * SLOT) == SLOT, 'STOP slot reserve short write')
        os.fsync(f.fileno())
        require(os.fstat(f.fileno()).st_size == SLOT, 'STOP reserve')
    slot_write(path, {'state': 'PENDING'})


def slot_write(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    require(len(raw) < SLOT - 100, 'STOP slot cap')
    payload = hashlib.sha256(raw).hexdigest().encode() + b'\n' + raw + b'\n'
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_size == SLOT and s.st_nlink == 1
                and stat.S_IMODE(s.st_mode) == 0o600, 'STOP slot identity')
        require(os.pwrite(fd, payload + b'\0' * (SLOT - len(payload)), 0) == SLOT,
                'STOP partial slot')
        os.fsync(fd)
    finally:
        os.close(fd)


def slot_read(raw):
    digest, body, padding = raw.split(b'\n', 2)
    require(not padding.strip(b'\0') and hashlib.sha256(body).hexdigest().encode() == digest,
            'STOP torn slot')
    return json.loads(body)


def safe_name(name):
    require(isinstance(name, str) and len(name.encode()) <= 4096 and '\0' not in name,
            'STOP name bound')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts, 'STOP unsafe name')
    return p.as_posix()


def census(root, *, uid=10001, clock=time.monotonic, budget=0.5, gid=None):
    """No follow traversal; bounded cooperative observation, never a physical quota."""
    gid = uid if gid is None else gid
    start = clock()
    total = 0
    rows = []
    pending = [(Path(root), '.')]
    while pending:
        require(clock() - start <= budget, 'STOP census deadline')
        path, name = pending.pop()
        s = path.lstat()
        require(s.st_uid == uid and s.st_gid == gid, 'STOP proof UID')
        kind = 'file' if stat.S_ISREG(s.st_mode) else 'dir' if stat.S_ISDIR(s.st_mode) else 'link' if stat.S_ISLNK(s.st_mode) else None
        require(kind is not None, 'STOP proof type')
        safe_name(name)
        row = dict(name=name, kind=kind, size=s.st_size, mode=stat.S_IMODE(s.st_mode),
                   uid=s.st_uid, gid=s.st_gid)
        if kind == 'file':
            require(s.st_size <= LEAF, 'STOP leaf logical cap')
            total += s.st_size
            require(total <= LOGICAL, 'STOP aggregate logical cap')
        elif kind == 'link':
            row['link'] = os.readlink(path)
            require(len(row['link'].encode()) <= 4096, 'STOP link cap')
        else:
            with os.scandir(path) as it:
                for child in it:
                    require(len(rows) + len(pending) < ENTRIES, 'STOP inventory cap')
                    pending.append((Path(child.path), child.name if name == '.' else name + '/' + child.name))
                    require(clock() - start <= budget, 'STOP census deadline')
        rows.append(row)
        require(len(rows) <= ENTRIES, 'STOP inventory cap')
    free = os.statvfs(root)
    available = free.f_bavail * free.f_frsize
    require(available >= 2 * GIB, 'STOP daemon pressure')
    return dict(logical=total, entries=len(rows), available=available, rows=rows,
                seconds=clock() - start, physical_quota=False)


def initialized(root, *, uid=10001, gid=None):
    gid = uid if gid is None else gid
    root = canonical(root)
    s = root.stat()
    require(stat.S_IMODE(s.st_mode) == 0o700 and s.st_uid == uid and s.st_gid == gid,
            'STOP private volume initialization')
    require({p.name for p in root.iterdir()} == {'pab-seed'}, 'STOP nonfresh volume')
    require(read_regular(root / 'pab-seed', 128, 0o600) == SEED, 'STOP seed')
    result = census(root, uid=uid, gid=gid)
    require(result['available'] >= 6 * GIB, 'STOP daemon admission 6GiB')
    return result


class Limited:
    def __init__(self, stream, cap):
        self.stream, self.cap, self.count = stream, cap, 0
    def write(self, raw):
        require(self.count + len(raw) <= self.cap, 'STOP export overflow')
        n = self.stream.write(raw)
        require(n == len(raw), 'STOP short export')
        self.count += n
        return n
    def flush(self):
        self.stream.flush()


def regular_hash(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        s = os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_size <= LEAF, 'STOP hash file')
        h = hashlib.sha256()
        count = 0
        while True:
            b = os.read(fd, 65536)
            if not b:
                break
            count += len(b)
            require(count <= s.st_size, 'STOP changed hash')
            h.update(b)
        t = os.fstat(fd)
        require(count == s.st_size and (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) ==
                (t.st_ino, t.st_size, t.st_mtime_ns, t.st_ctime_ns), 'STOP changed hash')
        return h.hexdigest()
    finally:
        os.close(fd)


def reader(root, mode, output, *, uid=10001, gid=None):
    enabled()
    inventory = census(root, uid=uid, gid=gid)
    sink = Limited(output, EXPORT if mode == 'archive' else META)
    if mode == 'archive':
        with tarfile.open(fileobj=sink, mode='w|', format=tarfile.USTAR_FORMAT) as tf:
            for row in inventory['rows']:
                path = Path(root) / row['name']
                info = tarfile.TarInfo(row['name'])
                info.uid = info.gid = uid
                info.mode = row['mode']
                info.mtime = 0
                if row['kind'] == 'dir':
                    info.type = tarfile.DIRTYPE
                    tf.addfile(info)
                elif row['kind'] == 'link':
                    info.type = tarfile.SYMTYPE
                    info.linkname = row['link']
                    tf.addfile(info)
                else:
                    info.size = row['size']
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, 'rb') as f:
                        s = os.fstat(f.fileno())
                        require(stat.S_ISREG(s.st_mode) and s.st_size == info.size, 'STOP archive race')
                        tf.addfile(info, f)
    else:
        require(mode == 'hash', 'STOP reader mode')
        for row in inventory['rows']:
            if row['kind'] == 'file':
                sink.write((json.dumps({'name': row['name'], 'sha256': regular_hash(Path(root) / row['name'])},
                                       separators=(',', ':')) + '\n').encode())
    sink.flush()


def namespace_pids():
    """Used only as PID1 in a private PID namespace, after test phase for reaping."""
    require(os.getpid() == 1, 'STOP PID namespace')
    return {int(p.name) for p in Path('/proc').iterdir() if p.name.isdecimal() and int(p.name) != 1}


def quiesce(deadline):
    observations = []
    for sig, allowance in ((signal.SIGTERM, 10), (signal.SIGKILL, 20)):
        pids = namespace_pids()
        observations.append({'signal': int(sig), 'pids': sorted(pids)})
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        end = min(deadline, time.monotonic() + allowance)
        while time.monotonic() < end:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                observations.append({'pid': pid, 'raw': status})
            if not namespace_pids():
                return {'complete': True, 'waits': observations}
            time.sleep(0.05)
    return {'complete': not namespace_pids(), 'waits': observations}


def capture(argv, outputs, deadline, *, env=None, observe=lambda: None):
    """Cap before each write. Reap only our entry/check leader during test phase."""
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(private(p)) for p in outputs]
        p = subprocess.Popen(argv, cwd='/app', env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
        # Own BOTH pipes before any fallible selector setup.
        streams = [p.stdout, p.stderr]
        for stream in streams:
            stack.callback(stream.close)
        try:
            sel = stack.enter_context(selectors.DefaultSelector())
            for index, stream in enumerate(streams):
                os.set_blocking(stream.fileno(), False)
                sel.register(stream, selectors.EVENT_READ, index)
            while sel.get_map() or p.poll() is None:
                require(time.monotonic() < deadline, 'STOP remote capture timeout')
                observe()
                for key, _ in sel.select(0.1):
                    raw = os.read(key.fileobj.fileno(), 65536)
                    if not raw:
                        sel.unregister(key.fileobj)
                        continue
                    f = files[key.data if len(files) == 2 else 0]
                    require(f.tell() + len(raw) <= META, 'STOP remote stream cap')
                    require(f.write(raw) == len(raw), 'STOP remote short write')
            return p.wait(timeout=0.1), p.pid
        except BaseException:
            # Namespace descendant cleanup is performed by writer only after test ends.
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except BaseException:
                pass
            try:
                p.wait(timeout=0.1)
            except BaseException:
                pass
            raise


def checked_program(argv, output, deadline):
    rc, _ = capture(argv, [output], deadline)
    require(rc == 0, 'STOP identity/source check')


def verify_source(root='/app'):
    data = json.loads(read_regular(Path(root) / 'manifest.json', META))
    for name, digest in {**data['files'], **data['derivative_files']}.items():
        require(regular_hash(Path(root) / safe_name(name)) == digest, 'STOP source drift')


def identity_program():
    # Original helper stays byte-identical; this bounded process adds the old launcher pins.
    return """import sys,os
if not __debug__ or sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'): raise SystemExit('STOP optimization')
exec(compile(open('/app/diagnostics/p-one-node/runtime-identity.py','rb').read(), '/app/diagnostics/p-one-node/runtime-identity.py','exec'))
assert sqlite3.sqlite_version=='3.40.1'
assert hashlib.sha256(lib.read_bytes()).hexdigest()=='2e6eef9a727f081f0d453b4e5e6cbd8b9ef8b6f86cbf7681cbad444d3b0b55c8'
assert subprocess.check_output(['dpkg-query','-W','-f=${Version} ${Architecture}','libsqlite3-0'],text=True)=='3.40.1-2+deb12u2 amd64'
with sqlite3.connect(':memory:') as db: assert db.execute('select sqlite_source_id()').fetchone()[0]=='2022-12-28 14:03:47 df5c253c0b3dd24916e4ec7cf77d3db5294cc9fd45ae7b9c5e82ad8197f3alt1'
print('assertion-enabled', __debug__, sys.flags.optimize)
"""


def writer(root, token, invocation):
    enabled()
    require(os.getpid() == 1 and os.getuid() == os.getgid() == 10001, 'STOP writer identity')
    require(re.fullmatch('[0-9a-f]{48}', token) is not None, 'STOP token')
    start = time.monotonic()
    # Signal watchdog does not depend on host attachment; no product handler changes.
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, 1350)
    state = dict(state='PENDING', token=token, invocation=invocation, entry=ENTRY,
                 node=NODE, expected_pytest_exit=1, acceptance=False, physical_quota=False)
    terminal = root / 'pab-terminal.slot'
    child = None
    monitor_stop = threading.Event()
    monitor_error = []
    maxima = {'logical': 0, 'entries': 0, 'scan_seconds': 0}
    try:
        initialized(root)
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        require(memory['MemAvailable'] >= 4 * GIB, 'STOP memory admission')
        for name in ('pab-intent.slot', 'pab-terminal.slot'):
            slot_create(root / name)
        slot_write(root / 'pab-intent.slot', state)
        with private(root / 'retention-sentinel') as f:
            f.write(token.encode() + b'\x00\xff\x80\r\n')
        verify_source()
        checked_program(['python', '-c', identity_program()], root / 'identity-before', start + 60)
        def monitor():
            while not monitor_stop.is_set():
                try:
                    observed = census(root)
                    for key in ('logical', 'entries'):
                        maxima[key] = max(maxima[key], observed[key])
                    maxima['scan_seconds'] = max(maxima['scan_seconds'], observed['seconds'])
                except BaseException as exc:
                    monitor_error.append(type(exc).__name__)
                    return
                monitor_stop.wait(0.5)
        require(time.monotonic() < start + 60, 'STOP pre-import phase overrun')
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='/app/diagnostics/p-one-node',
                   PAB_MAINTENANCE_WAIT_LOG='/tmp/waits.jsonl')
        def observe():
            require(not monitor_error, 'STOP monitor error')
        child_deadline = min(time.monotonic() + 1200, start + 1260)
        rc, pid = capture(ENTRY, [root / 'pytest.stdout', root / 'pytest.stderr'],
                          child_deadline, env=env, observe=observe)
        state['child_pid'], state['pytest_exit'] = pid, rc
        require(not monitor_error, 'STOP monitor error')
        state['test_phase_finished'] = True
    except BaseException as exc:
        state['failure'] = type(exc).__name__
    finally:
        # No namespace waitpid competes with pytest during its protected test phase.
        state['test_phase_finished'] = True
        try:
            state['descendants'] = quiesce(min(start + 1290, time.monotonic() + 30))
            require(state['descendants']['complete'], 'STOP unresolved descendants')
            post_deadline = min(time.monotonic() + 60, start + 1350)
            monitor_stop.set()
            if 'thread' in locals():
                thread.join(timeout=1)
                require(not thread.is_alive(), 'STOP unresolved monitor')
            require(time.monotonic() < post_deadline, 'STOP postcheck phase overrun')
            state['maxima'] = maxima
            require(not monitor_error, 'STOP incomplete census')
            verify_source()
            require(time.monotonic() < post_deadline, 'STOP postcheck phase overrun')
            checked_program(['python', '-c', identity_program()], root / 'identity-after', post_deadline)
            require(read_regular(root / 'identity-before', META) == read_regular(root / 'identity-after', META),
                    'STOP identity drift')
            require(state.get('pytest_exit') == 1 and 'failure' not in state, 'STOP unexpected test lifecycle')
            state['state'] = 'TERMINAL'
            state['classification'] = 'EXPECTED-RED-REQUIRES-INDEPENDENT-47-FIELD-REVIEW'
        except BaseException as exc:
            state['state'] = 'UNKNOWN' if not state.get('descendants', {}).get('complete') else 'STOP'
            state['terminal_failure'] = type(exc).__name__
        try:
            # Publication belongs to the same phase, including failure publication.
            if 'post_deadline' not in locals():
                post_deadline = min(time.monotonic() + 60, start + 1350)
            require(time.monotonic() < post_deadline, 'STOP publication phase overrun')
            state['elapsed'] = time.monotonic() - start
            slot_write(terminal, state)
            require(time.monotonic() < post_deadline, 'STOP publication phase overrun')
        except BaseException:
            # Preallocated PENDING/torn slot remains authoritative uncertainty.
            state['state'] = 'UNKNOWN'
    return 0 if state['state'] == 'TERMINAL' else 1


def main():
    enabled()
    require(len(sys.argv) == 4 and sys.argv[1] in ('writer', 'archive', 'hash'), 'STOP fixed arguments')
    mode, token, invocation = sys.argv[1:]
    require(re.fullmatch('[0-9a-f]{48}', token) is not None, 'STOP token')
    import uuid
    require(str(uuid.UUID(invocation)) == invocation, 'STOP UUID')
    root = Path('/tmp')
    if mode == 'writer':
        return writer(root, token, invocation)
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, 60)
    require(os.getpid() == 1 and os.getuid() == os.getgid() == 10001, 'STOP helper identity')
    reader(root, mode, sys.stdout.buffer)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
