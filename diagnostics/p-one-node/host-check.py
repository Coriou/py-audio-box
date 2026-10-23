"""Fixed <=32 case /300s HOST matrix. FAKE EXTERNAL COMMANDS; NO PRODUCT CLAIM."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import uuid

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('p_one_launcher', D / 'launcher.py')
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
F = L.F
OWNER = {'token': 'ab' * 24, 'invocation': '1e60f6fa-9f38-4805-a918-8ae94112e286',
         'source': L.SOURCE, 'documentation': L.DOCUMENTATION, 'pid': 777}
IMAGE = 'sha256:' + 'cd' * 32
CID = {'writer': '1' * 64, 'archive': '2' * 64, 'hash': '3' * 64}
VOL = 'pab-p-one-' + OWNER['invocation']
CASES = []
ROOT = None
OBSERVATIONS = []
FAKES = []


def case(name):
    def add(fn):
        CASES.append((name, fn))
        return fn
    return add


def expect(fn, kind=Exception, message=None):
    try:
        fn()
    except kind as exc:
        if message is not None:
            check(message in str(exc), 'wrong rejection: ' + str(exc))
        return
    raise RuntimeError('expected rejection did not occur')


def fresh(name):
    p = ROOT / name
    p.mkdir(mode=0o700)
    return p


class Proc:
    def __init__(self, raw=b'', err=b'', rc=0, *, open_pipe=False, wait_failure=False):
        self.pid = 987654321  # never passed to a real signal API
        self.rc, self.wait_failure = rc, wait_failure
        self.stdout, self.stderr = None, None
        self.writers, self.threads = [], []
        for key, payload in (('stdout', raw), ('stderr', err)):
            r, w = os.pipe()
            setattr(self, key, os.fdopen(r, 'rb', buffering=0))
            if open_pipe:
                self.writers.append(w)
            else:
                def feed(fd=w, data=payload):
                    try:
                        offset = 0
                        while offset < len(data):
                            offset += os.write(fd, data[offset:offset + 4096])
                    except BrokenPipeError:
                        pass
                    finally:
                        os.close(fd)
                t = threading.Thread(target=feed)
                t.start()
                self.threads.append(t)
    def poll(self):
        return self.rc
    def wait(self, timeout):
        check(0 < timeout <= 1, 'unbounded wait')
        if self.wait_failure:
            raise subprocess.TimeoutExpired('FAKE', timeout)
        return self.rc
    def observe(self):
        check(self.stdout.closed and self.stderr.closed, 'FD leak before rescue')
        return {'stdout_closed': self.stdout.closed, 'stderr_closed': self.stderr.closed}
    def rescue(self):
        for fd in self.writers:
            os.close(fd)
        for t in self.threads:
            t.join(timeout=1)
            check(not t.is_alive(), 'fake pipe feeder unresolved')


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


class Fake:
    """Exact finite queue, no fallback. No callable can invoke a real command."""
    def __init__(self, script, *, low=False, tick=0.0001):
        FAKES.append(self)
        self.finished = False
        self.script = list(script)
        self.calls, self.processes, self.kills = [], [], []
        self.t, self.tick, self.low = 0.0, tick, low
    def clock(self):
        self.t += self.tick
        return self.t
    def free(self, path):
        check(Path(path).is_relative_to(ROOT), 'fake free outside test root')
        return (1 if self.low else 20) * F.GIB
    def spawn(self, argv, **kw):
        check(self.script, 'unexpected external command')
        expected, response = self.script.pop(0)
        check(argv == expected, 'unexpected argv: ' + repr(argv))
        check(kw == dict(cwd=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True),
              'unexpected cwd/fd/spawn kwargs')
        self.calls.append(argv)
        if isinstance(response, BaseException):
            raise response
        p = Proc(**response)
        self.processes.append(p)
        return p
    def killpg(self, pid, sig):
        check(pid == 987654321 and sig == signal.SIGKILL, 'foreign signal')
        self.kills.append((pid, int(sig)))
    def done(self, *, remaining=False):
        observations = [p.observe() for p in self.processes]
        # Observe first, rescue fake-owned feeder descriptors second.
        for p in self.processes:
            p.rescue()
        check(remaining or not self.script, 'unconsumed expected commands')
        self.finished = True
        OBSERVATIONS.append({'commands': len(self.calls), 'kills': self.kills,
                             'pre_rescue_fd_observations': observations})


def invoke(root, label, fake, *, cap=F.META, deadline=10):
    return L.call(root, label, ['FAKE', label], L.Budget(), fake, deadline=deadline, cap=cap)


@case('01-private-IO-022-exclusive-symlink')
def _():
    r = fresh('private')
    b = L.Budget()
    with L.checkout_umask():
        L.publish(r / 'ok', {'binary': '\u0000\u00ff'}, b)
        check(stat.S_IMODE((r / 'ok').stat().st_mode) == 0o600, 'private0600')
    expect(lambda: L.publish(r / 'ok', {}, b))
    (r / 'link').symlink_to(r / 'ok')
    before = (r / 'ok').read_bytes()
    expect(lambda: L.publish(r / 'link', {}, b))
    check((r / 'ok').read_bytes() == before, 'truncated existing leaf')


@case('02-stable-read-caps-ancestor-and-scope')
def _():
    r = fresh('stable')
    with F.private(r / 'f') as f:
        f.write(b'\0\xff\x80')
    check(F.read_regular(r / 'f', 3, 0o600) == b'\0\xff\x80', 'binary read')
    expect(lambda: F.read_regular(r / 'f', 2))
    expect(lambda: F.read_regular(r / 'f', 3, 0o644))
    (r / 'alias').symlink_to(r, target_is_directory=True)
    expect(lambda: F.read_regular(r / 'alias' / 'f', 3))
    try:
        with L.checkout_umask():
            raise ValueError('fixture')
    except ValueError:
        pass
    previous = os.umask(0o077)
    check(previous == 0o077, 'umask not restored')


@case('03-real-slots-pending-torn-final-publication')
def _():
    r = fresh('slots')
    F.slot_create(r / 'state')
    check(F.slot_read(F.read_regular(r / 'state', F.SLOT)) == {'state': 'PENDING'}, 'pending slot')
    F.slot_write(r / 'state', {'state': 'TERMINAL', 'token': OWNER['token']})
    raw = F.read_regular(r / 'state', F.SLOT)
    check(F.slot_read(raw)['state'] == 'TERMINAL', 'terminal slot')
    expect(lambda: F.slot_read(raw[:80] + b'!' + raw[81:]))
    expect(lambda: L.publish(r / 'fail', {}, L.Budget(0)))
    check(not (r / 'fail').exists(), 'write after budget refusal')


@case('04-real-pipes-binary-EOF-nonzero-exactcap')
def _():
    for label, raw, rc in [('binary', b'\0\xff\x80', 0), ('nonzero', b'raw', 7), ('eof', b'', 0)]:
        r = fresh(label)
        fake = Fake([(['FAKE', label], {'raw': raw, 'err': b'\xff', 'rc': rc})])
        if rc:
            expect(lambda: invoke(r, label, fake, cap=max(1, len(raw))))
        else:
            check(invoke(r, label, fake, cap=max(1, len(raw))) == raw, 'raw capture')
        fake.done()


@case('05-capture-overflow-before-write')
def _():
    r = fresh('overflow')
    fake = Fake([(['FAKE', 'overflow'], {'raw': b'abcd'})])
    expect(lambda: invoke(r, 'overflow', fake, cap=3))
    check((r / 'overflow.stdout').stat().st_size <= 3 and len(fake.kills) == 1, 'overflow branch')
    fake.done()


@case('06-exited-leader-open-pipe-timeout-groupkill')
def _():
    r = fresh('leader')
    fake = Fake([(['FAKE', 'leader'], {'open_pipe': True})], tick=1)
    expect(lambda: invoke(r, 'leader', fake, deadline=4))
    check(len(fake.kills) == 1, 'exited leader kill omitted')
    fake.done()


@case('07-spawn-failure-and-second-capture-open')
def _():
    r = fresh('spawn')
    fake = Fake([(['FAKE', 'spawn'], OSError('fake spawn failure'))])
    expect(lambda: invoke(r, 'spawn', fake))
    fake.done()
    r = fresh('second-open')
    (r / 'second.stderr').touch(mode=0o600)
    fake = Fake([])
    expect(lambda: invoke(r, 'second', fake))
    check(not fake.calls, 'dispatch after capture-open failure')
    fake.done()


@case('08-capture-read-failure')
def _():
    r = fresh('readfail')
    fake = Fake([(['FAKE', 'readfail'], {'raw': b'abc'})])
    original = os.read
    def fail(fd, n):
        if any(fd in (p.stdout.fileno(), p.stderr.fileno()) for p in fake.processes):
            raise OSError('fake read')
        return original(fd, n)
    with patch.object(os, 'read', fail):
        expect(lambda: invoke(r, 'readfail', fake))
    check(len(fake.kills) == 1, 'read failure no kill')
    fake.done()


@case('09-selector-failure')
def _():
    r = fresh('selectfail')
    fake = Fake([(['FAKE', 'selectfail'], {'raw': b'abc'})])
    with patch.object(L.selectors.DefaultSelector, 'select', side_effect=OSError('fake select')):
        expect(lambda: invoke(r, 'selectfail', fake))
    fake.done()


# F2 additions remain in the existing capture scenario group09.
_old_capture_case = CASES[-1][1]
def capture_setup_failures():
    _old_capture_case()
    for mode in ('spawn-publication', 'selector-constructor', 'first-register', 'set-blocking'):
        r = fresh('host-setup-' + mode)
        fake = Fake([(['FAKE', 'setup'], {'open_pipe': True})])
        original = L.publish
        def publication(path, value, budget):
            if path.name.endswith('.spawn.json'):
                raise OSError('FAKE spawn publication')
            return original(path, value, budget)
        hook = (patch.object(L, 'publish', publication) if mode == 'spawn-publication' else
                patch.object(L.selectors, 'DefaultSelector', side_effect=OSError('FAKE constructor'))
                if mode == 'selector-constructor' else
                patch.object(L.selectors.DefaultSelector, 'register', side_effect=OSError('FAKE register'))
                if mode == 'first-register' else
                patch.object(os, 'set_blocking', side_effect=OSError('FAKE blocking')))
        with hook:
            expect(lambda: invoke(r, 'setup', fake))
        receipt = json.loads((r / 'setup.complete.json').read_bytes())
        check(receipt['state'] == 'UNKNOWN' and receipt['group_kill_attempted']
              and receipt['leader_reaped'] and not receipt['remote_completion'], 'host failure latch')
        check(len(fake.kills) == 1, 'host kill count')
        fake.done()
    for mode in ('selector-constructor', 'first-register', 'set-blocking'):
        r = fresh('remote-setup-' + mode)
        proc = Proc(open_pipe=True)
        kills, waits = [], []
        original_wait = proc.wait
        def wait(timeout):
            waits.append(timeout)
            return original_wait(timeout)
        hook = (patch.object(F.selectors, 'DefaultSelector', side_effect=OSError('FAKE constructor'))
                if mode == 'selector-constructor' else
                patch.object(F.selectors.DefaultSelector, 'register', side_effect=OSError('FAKE register'))
                if mode == 'first-register' else
                patch.object(os, 'set_blocking', side_effect=OSError('FAKE blocking')))
        with patch.object(F.subprocess, 'Popen', return_value=proc), hook, \
             patch.object(F.os, 'killpg', side_effect=lambda pid, sig: kills.append((pid, int(sig)))), \
             patch.object(proc, 'wait', side_effect=wait):
            expect(lambda: F.capture(['FAKE'], [r / 'out'], 10), message='FAKE')
        check(kills == [(proc.pid, int(signal.SIGKILL))] and waits == [0.1], 'remote kill/reap')
        observation = proc.observe()
        OBSERVATIONS.append({'remote_setup': mode, 'kills': kills, 'waits': waits,
                             'pre_rescue_fd_observations': [observation]})
        proc.rescue()
CASES[-1] = (CASES[-1][0], capture_setup_failures)


@case('10-capture-write-failure')
def _():
    r = fresh('writefail')
    fake = Fake([(['FAKE', 'writefail'], {'raw': b'abc'})])
    original = F.private
    class Bad:
        def __init__(self, f): self.f = f
        def write(self, _): raise OSError('fake capture write')
        def close(self): return self.f.close()
        @property
        def closed(self): return self.f.closed
    def opened(path):
        f = original(path)
        return Bad(f) if str(path).endswith('.stdout') else f
    with patch.object(F, 'private', opened):
        expect(lambda: invoke(r, 'writefail', fake))
    fake.done()


@case('11-capture-close-failure')
def _():
    r = fresh('closefail')
    fake = Fake([(['FAKE', 'closefail'], {'raw': b'abc'})])
    original = F.private
    class Bad:
        def __init__(self, f): self.f = f
        def write(self, b): return self.f.write(b)
        def close(self):
            self.f.close()
            raise OSError('fake close')
        @property
        def closed(self): return self.f.closed
    def opened(path):
        f = original(path)
        return Bad(f) if str(path).endswith('.stdout') else f
    with patch.object(F, 'private', opened):
        expect(lambda: invoke(r, 'closefail', fake))
    fake.done()


@case('12-leader-reap-failure-explicit-unknown')
def _():
    r = fresh('waitfail')
    fake = Fake([(['FAKE', 'waitfail'], {'wait_failure': True})])
    expect(lambda: invoke(r, 'waitfail', fake))
    receipt = json.loads((r / 'waitfail.complete.json').read_bytes())
    check(receipt['state'] == 'UNKNOWN' and not receipt['leader_reaped'], 'reap uncertainty')
    fake.done()


@case('13-optimization-refuses-before-acquisition')
def _():
    flags = SimpleNamespace(optimize=1)
    with patch.object(F.sys, 'flags', flags):
        expect(F.enabled)
    with patch.dict(os.environ, {'PYTHONOPTIMIZE': '1'}):
        expect(F.enabled)


@case('14-startup-configuration-admissions')
def _():
    L.startup({'Env': ['PATH=/usr/bin']})
    for key in ('Entrypoint', 'OnBuild', 'Healthcheck', 'Volumes'):
        expect(lambda key=key: L.startup({key: ['unexpected']}))
    for var in ('PYTHONOPTIMIZE=1', 'PYTHONHOME=/bad', 'LD_PRELOAD=bad'):
        expect(lambda var=var: L.startup({'Env': [var]}))


def volume():
    return {'Name': VOL, 'Driver': 'local', 'Options': None, 'Scope': 'local',
            'CreatedAt': '2026-10-23T12:00:00Z', 'Mountpoint': '/fake/daemon/path',
            'Labels': {'pab.owner': OWNER['token'], 'pab.invocation': OWNER['invocation']}}


@case('15-volume-options-collision-not-immutable')
def _():
    check(L.volume_binding(volume(), OWNER, VOL)['Name'] == VOL, 'local volume')
    for change in ({'Options': {'type': 'tmpfs'}}, {'Driver': 'nfs'}, {'Labels': {}}, {'CreatedAt': ''}):
        expect(lambda change=change: L.volume_binding(dict(volume(), **change), OWNER, VOL))


@case('16-real-private-seed-and-wrong-UID-mode')
def _():
    r = fresh('seed')
    with F.private(r / 'pab-seed') as f: f.write(F.SEED)
    check(F.initialized(r, uid=os.getuid(), gid=r.stat().st_gid)['entries'] == 2, 'seed initialization')
    expect(lambda: F.initialized(r, uid=10001))
    r.chmod(0o755)
    expect(lambda: F.initialized(r, uid=os.getuid(), gid=r.stat().st_gid))
    r.chmod(0o700)


@case('17-host-low-space-no-dispatch')
def _():
    r = lifecycle_root('lowspace')
    fake = Fake([], low=True)
    obj = L.Lifecycle(r, r / 'lane', IMAGE, OWNER, fake)
    check(not obj.run() and not fake.calls and not (r / 'lane').exists(), 'lowspace dispatched')
    fake.done()
    for name in ('occupied', 'partial'):
        r = lifecycle_root(name)
        (r / 'lane').mkdir(mode=0o700)
        if name == 'occupied':
            L.publish(r / 'lane' / 'owner.json', {'foreign': True}, L.Budget())
        fake = Fake([])
        obj = L.Lifecycle(r, r / 'lane', IMAGE, OWNER, fake)
        check(not obj.run() and (r / 'lane').exists() and not fake.calls, 'collision adopted')
        fake.done()


@case('18-logical-accounting-exact-and-overflow')
def _():
    b = L.Budget()
    b.charge(L.HOST)
    expect(lambda: b.charge(1))
    r = fresh('logical')
    for n in range(17):
        with F.private(r / str(n)): pass
    original = Path.lstat
    def size_fixture(path):
        s = original(path)
        if path.parent == r and path.name.isdecimal():
            values = list(s)
            values[6] = F.LEAF
            return os.stat_result(values)
        return s
    with patch.object(Path, 'lstat', size_fixture):
        expect(lambda: F.census(r, uid=os.getuid(), gid=r.stat().st_gid, budget=1), message='aggregate logical cap')


@case('19-export-bound-before-write-no-large-allocation')
def _():
    out = io.BytesIO()
    limited = F.Limited(out, F.EXPORT)
    limited.count = F.EXPORT - 1  # Counter boundary fixture, no fake storage quota claim.
    limited.write(b'a')
    expect(lambda: limited.write(b'b'))
    check(out.getvalue() == b'a', 'overflow emitted bytes')


@case('20-monitor-deadline-error-pressure')
def _():
    r = fresh('monitor')
    ticks = iter([0, 1, 2])
    expect(lambda: F.census(r, uid=os.getuid(), gid=r.stat().st_gid, clock=lambda: next(ticks)), message='census deadline')
    os.mkfifo(r / 'fifo', 0o600)
    expect(lambda: F.census(r, uid=os.getuid(), gid=r.stat().st_gid), message='proof type')
    (r / 'fifo').unlink()
    with patch.object(F.os, 'statvfs', return_value=SimpleNamespace(f_bavail=1, f_frsize=1)):
        expect(lambda: F.census(r, uid=os.getuid(), gid=r.stat().st_gid), message='daemon pressure')


def tar_bytes(files, *, bad=None):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w', format=tarfile.USTAR_FORMAT) as tf:
        root = tarfile.TarInfo('.')
        root.type = tarfile.DIRTYPE
        root.uid = root.gid = 10001
        root.mode = 0o700
        tf.addfile(root)
        for name, raw in files:
            m = tarfile.TarInfo(name)
            m.uid = m.gid = 10001
            m.mode, m.size = 0o600, len(raw)
            if bad == 'uid': m.uid = 0
            if bad == 'link': m.type, m.linkname, m.size = tarfile.LNKTYPE, 'f', 0
            tf.addfile(m, io.BytesIO(raw) if m.isfile() else None)
    return out.getvalue()


@case('21-stream-tar-valid-duplicate-escape-link-UID-truncation')
def _():
    reader_root = fresh('actual-reader')
    with F.private(reader_root / 'binary') as f: f.write(b'\0\xff\x80')
    archive, hashes = io.BytesIO(), io.BytesIO()
    F.reader(reader_root, 'archive', archive, uid=os.getuid(), gid=reader_root.stat().st_gid)
    F.reader(reader_root, 'hash', hashes, uid=os.getuid(), gid=reader_root.stat().st_gid)
    with tarfile.open(fileobj=io.BytesIO(archive.getvalue())) as tf:
        check(tf.extractfile('binary').read() == b'\0\xff\x80', 'actual reader binary')
    check(json.loads(hashes.getvalue())['sha256'] == hashlib.sha256(b'\0\xff\x80').hexdigest(),
          'actual independent reader hash')
    for index, (files, bad, truncate) in enumerate([
            ([('./f', b'\0\xff\x80')], None, False),
            ([('f', b'a'), ('./f', b'b')], None, False),
            ([('../f', b'a')], None, False), ([('f', b'a')], 'link', False),
            ([('f', b'a')], 'uid', False), ([('f', b'a')], None, True)]):
        r = fresh('tar-' + str(index))
        raw = tar_bytes(files, bad=bad)
        with F.private(r / 'tar') as f: f.write(raw[:700] if truncate else raw)
        fn = lambda: L.parse_archive(r / 'tar', r / 'inventory', L.Budget())
        if index == 0:
            check(fn()['hashes']['f'] == hashlib.sha256(b'\0\xff\x80').hexdigest(), 'tar binary hash')
        else: expect(fn)


@case('22-map-duplicates-malformed-mismatch')
def _():
    r = fresh('map')
    raw = L.encode({'name': './f', 'sha256': 'a' * 64}) + b'\n'
    with F.private(r / 'ok') as f: f.write(raw)
    check(L.parse_map(r / 'ok') == {'f': 'a' * 64}, 'map normalized')
    for name, data in [('duplicate', raw + raw), ('bad', b'{}'), ('escape', raw.replace(b'./f', b'../'))]:
        with F.private(r / name) as f: f.write(data)
        expect(lambda name=name: L.parse_map(r / name))
    check(L.parse_map(r / 'ok') != {'f': 'b' * 64}, 'map mismatch lost')


def records():
    rows = []
    for stage in range(1, 5):
        r = [0] * 47
        r[0:4] = [1, 1, stage, 10]
        r[4], r[5], r[6], r[7], r[12] = (11 if stage == 1 else 10), 10, 11, 18944, 1
        r[16] = r[26] = r[31] = 1
        rows.append(','.join(map(str, r)).encode() + b'\n')
    return b''.join(rows)


def lifecycle_root(name):
    r = fresh(name)
    for slot in ('intent.slot', 'final.slot'): F.slot_create(r / slot)
    return r


def container(mode, status):
    return {'Id': CID[mode], 'Name': '/pab-p-one-' + OWNER['invocation'] + '-' + mode, 'Image': IMAGE,
            'Config': {'Labels': {'pab.owner': OWNER['token'], 'pab.invocation': OWNER['invocation']},
                       'User': '10001:10001', 'WorkingDir': '/app', 'Cmd': L.command(mode, OWNER),
                       'Env': ['PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/app/diagnostics/p-one-node',
                               'PAB_MAINTENANCE_WAIT_LOG=/tmp/waits.jsonl', 'TIKTOKEN_CACHE_DIR=/tokenizer-cache']},
            'HostConfig': {'NanoCpus': 2000000000, 'Memory': 2147483648, 'PidsLimit': 256,
                           'ReadonlyRootfs': True, 'NetworkMode': 'none', 'CapDrop': ['ALL'],
                           'SecurityOpt': ['no-new-privileges'], 'RestartPolicy': {'Name': 'no'},
                           'LogConfig': {'Type': 'none'}},
            'Mounts': [{'Type': 'volume', 'Name': VOL, 'Destination': '/tmp', 'RW': mode == 'writer'}],
            'State': {'Status': status, 'Running': False, 'ExitCode': 0, 'Pid': 0, 'OOMKilled': False}}


def lifecycle_script(failure=None):
    tmp = fresh('fixture-' + uuid.uuid4().hex)
    terminal = {'state': 'TERMINAL', 'token': OWNER['token'], 'invocation': OWNER['invocation'],
                'pytest_exit': 1, 'descendants': {'complete': True}, 'acceptance': False}
    F.slot_create(tmp / 'terminal')
    F.slot_write(tmp / 'terminal', terminal)
    files = [('pab-terminal.slot', F.read_regular(tmp / 'terminal', F.SLOT)),
             ('retention-sentinel', OWNER['token'].encode() + b'\0\xff\x80\r\n'),
             ('pab-p-one-node-observer/records', records())]
    archive = tar_bytes(files)
    mapraw = b''.join(L.encode({'name': name, 'sha256': hashlib.sha256(raw).hexdigest()}) + b'\n'
                      for name, raw in files)
    seq = []
    def add(argv, raw=b''):
        seq.append((argv, {'raw': raw}))
    def obj(argv, value): add(argv, L.encode([value]))
    obj(['docker', 'image', 'inspect', IMAGE], {'Id': IMAGE, 'Os': 'linux', 'Architecture': 'amd64', 'Config': {}})
    add(['docker', 'volume', 'ls', '--format', '{{.Name}}', '--filter', 'name=^' + VOL + '$'])
    add(['docker', 'volume', 'create', '--driver', 'local', '--label', 'pab.owner=' + OWNER['token'],
         '--label', 'pab.invocation=' + OWNER['invocation'], VOL], VOL.encode())
    obj(['docker', 'volume', 'inspect', VOL], volume())
    for mode in ('writer', 'archive', 'hash'):
        name = 'pab-p-one-' + OWNER['invocation'] + '-' + mode
        add(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'])
        add(L.create_argv(name, VOL, IMAGE, mode, OWNER), CID[mode].encode())
        obj(['docker', 'inspect', CID[mode]], container(mode, 'created'))
        add(['docker', 'start', '--attach', CID[mode]], archive if mode == 'archive' else mapraw if mode == 'hash' else b'')
        obj(['docker', 'inspect', CID[mode]], container(mode, 'exited'))
    for mode in ('writer', 'archive', 'hash'):
        obj(['docker', 'inspect', CID[mode]], container(mode, 'exited'))
        add(['docker', 'rm', CID[mode]])
        add(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id=' + CID[mode]])
    if failure:
        kind, target = failure
        index = next(i for i, (argv, _) in enumerate(seq) if argv == target)
        seq = seq[:index + 1]
        if kind == 'transport':
            seq[index] = (target, OSError('FAKE lost acknowledgement'))
            if target[:3] == ['docker', 'volume', 'create']:
                obj(['docker', 'volume', 'inspect', VOL], volume())
            else:
                mode = next(m for m in CID if CID[m] in target or any(m in a for a in target if a.startswith('pab-p-one-')))
                identifier = CID[mode] if 'start' in target or 'rm' in target else 'pab-p-one-' + OWNER['invocation'] + '-' + mode
                obj(['docker', 'inspect', identifier], container(mode, 'exited'))
        elif kind == 'nonzero':
            seq[index][1]['rc'] = 1
            # mutation failure -> one exact discovery, no retry
            if target[:2] == ['docker', 'start']:
                mode = next(m for m in CID if CID[m] in target)
                obj(['docker', 'inspect', CID[mode]], container(mode, 'exited'))
        elif kind == 'wrong-state':
            mode = next(m for m in CID if CID[m] in target)
            value = container(mode, 'running')
            value['State']['Running'] = True
            seq[index][1]['raw'] = L.encode([value])
    return seq


def full(name, failure=None):
    r = lifecycle_root(name)
    fake = Fake(lifecycle_script(failure))
    obj = L.Lifecycle(r, r / 'lane', IMAGE, OWNER, fake)
    ok = obj.run()
    fake.done()
    return r, obj, ok


@case('23-full-scripted-success-real-retention-and-handback')
def _():
    r, obj, ok = full('full')
    check(ok and obj.retained and not (r / 'lane').exists(), 'full lifecycle did not finish')
    check(F.slot_read(F.read_regular(r / 'final.slot', F.SLOT))['acceptance'] is False, 'self acceptance')
    check(not any('exec' in cmd or (cmd[:3] == ['docker', 'volume', 'rm']) for cmd in obj.a.calls),
          'forbidden operation')


@case('24-lost-volume-create-ack-no-retry-hold')
def _():
    target = ['docker', 'volume', 'create', '--driver', 'local', '--label', 'pab.owner=' + OWNER['token'],
              '--label', 'pab.invocation=' + OWNER['invocation'], VOL]
    r, obj, ok = full('lost-volume', ('transport', target))
    check(not ok and (r / 'lane').exists() and obj.operations['volume-create']['state'] == 'SENT-UNKNOWN',
          'lost create released')
    check(obj.a.calls.count(target) == 1, 'mutation retry')


@case('25-lost-writer-start-ack-no-reader')
def _():
    target = ['docker', 'start', '--attach', CID['writer']]
    r, obj, ok = full('lost-start', ('transport', target))
    check(not ok and (r / 'lane').exists() and not any(CID['archive'] in cmd for cmd in obj.a.calls),
          'reader after uncertainty')


@case('26-stopped-writer-export-failure-preserves-volume')
def _():
    r, obj, ok = full('export-fail', ('nonzero', ['docker', 'start', '--attach', CID['archive']]))
    check(not ok and obj.objects['writer']['state'] == 'STOPPED' and (r / 'lane').exists(), 'stopped hold')
    check(not any(cmd[:2] == ['docker', 'rm'] or cmd[:3] == ['docker', 'volume', 'rm'] for cmd in obj.a.calls),
          'destruction after failed retention')
    r = lifecycle_root('hash-mismatch')
    seq = lifecycle_script()
    idx = next(i for i, (cmd, _) in enumerate(seq) if cmd == ['docker', 'start', '--attach', CID['hash']])
    rows = seq[idx][1]['raw'].splitlines()
    changed = json.loads(rows[0]); changed['sha256'] = '0' * 64
    seq[idx][1]['raw'] = L.encode(changed) + b'\n' + b'\n'.join(rows[1:]) + b'\n'
    seq = seq[:idx + 2]  # Stopped hash receipt, then mismatch forbids cleanup.
    fake = Fake(seq)
    obj = L.Lifecycle(r, r / 'lane', IMAGE, OWNER, fake)
    check(not obj.run() and not obj.retained and (r / 'lane').exists(), 'hash mismatch accepted')
    fake.done()


@case('27-remove-uncertainty-and-active-state')
def _():
    r, obj, ok = full('remove-fail', ('transport', ['docker', 'rm', CID['writer']]))
    check(not ok and obj.retained and (r / 'lane').exists(), 'remove failure released lane')
    bad = container('writer', 'exited')
    bad['State']['OOMKilled'] = True
    expect(lambda: L.stopped(bad))
    bad = container('writer', 'exited')
    bad['Config']['User'] = '0'
    expect(lambda: L.container_binding(bad, OWNER, 'pab-p-one-' + OWNER['invocation'] + '-writer', VOL, IMAGE, 'writer'))


@case('28-final-publication-failure-keeps-pending')
def _():
    r = lifecycle_root('finalfail')
    fake = Fake(lifecycle_script())
    obj = L.Lifecycle(r, r / 'lane', IMAGE, OWNER, fake)
    original = L.publish
    def failed(path, value, budget):
        if path.name == 'evidence-hashes.json': raise OSError('FAKE final publication')
        return original(path, value, budget)
    with patch.object(L, 'publish', failed):
        check(not obj.run(), 'finalization accepted')
    check(F.slot_read(F.read_regular(r / 'final.slot', F.SLOT))['state'] == 'PENDING'
          and (r / 'lane').exists(), 'stale final success')
    fake.done()


@case('29-actual-source-context-private-copy-no-build')
def _():
    r = fresh('context')
    context = L.prepare_context(D.parents[1], r, L.Budget(), OWNER)
    recipe = (context / 'Dockerfile').read_text()
    check('COPY --chown=10001:10001' in recipe and 'RUN chown 10001:10001 /tmp' in recipe,
          'ownership recipe')
    check('install' not in recipe and recipe.startswith('FROM ' + L.base_reference(OWNER) + '\n'), 'unexpected build operation')
    L.validate_recipe(context, OWNER, L.base_steps(OWNER))
    (context / 'Dockerfile').write_text(recipe.replace(L.base_reference(OWNER), L.BASE))
    expect(lambda: L.validate_recipe(context, OWNER, L.base_steps(OWNER)))
    (context / 'Dockerfile').write_text(recipe)
    for p in context.rglob('*'):
        check(stat.S_IMODE(p.stat().st_mode) == (0o700 if p.is_dir() else 0o600), 'context private modes')


@case('30-actual-47-field-validator-synthetic-only')
def _():
    check(L.validate_records(records())['acceptance'] is False, 'synthetic accepted product')
    expect(lambda: L.validate_records(records().split(b'\n', 1)[1]))
    expect(lambda: L.validate_records(records().replace(b',18944,', b',0,')))
    expect(lambda: L.validate_records(b'1,' * 52))


@case('31-bounded-post-test-descendant-cleanup-fixtures')
def _():
    # Exercise actual namespace enumeration and quiesce, never a real PID signal/reap.
    clock = iter([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    killed = []
    def entries(_):
        return [Path('/proc/1')] + ([] if killed else [Path('/proc/42')])
    with patch.object(F.os, 'getpid', return_value=1), patch.object(Path, 'iterdir', entries), \
         patch.object(F.os, 'kill', side_effect=lambda pid, sig: killed.append((pid, int(sig)))), \
         patch.object(F.os, 'waitpid', side_effect=ChildProcessError), \
         patch.object(F.time, 'monotonic', side_effect=lambda: next(clock)):
        result = F.quiesce(1)
    check(result['complete'] and killed == [(42, int(signal.SIGTERM))], 'descendant cleanup')
    r = fresh('supervisor')
    class Worker:
        pid = 888888888
        def __init__(self, target): self.status = None
        def poll(self): return self.status
        def terminate(self, sig): self.status = -int(sig)
    tick = [0]
    def now():
        tick[0] += 0.01
        return tick[0]
    check(not L.supervise(r, lambda: True, factory=Worker, clock=now, sleep=lambda _: None,
                          free=lambda _: 1), 'pressure supervisor accepted')
    check(json.loads((r / 'supervisor-complete.json').read_bytes())['state'] == 'UNKNOWN',
          'supervisor inferred remote completion')



@case('32-writer-refuses-non-PID1-before-mutations')
def _():
    r = fresh('writer-refusal')
    expect(lambda: F.writer(r, OWNER['token'], OWNER['invocation']))
    check(not list(r.iterdir()), 'writer mutation before PID1 admission')


_old_writer_case = CASES[-1][1]
def writer_scheduling():
    _old_writer_case()
    # Execute the ACTUAL writer scheduling function; only environment/IO/child
    # boundaries are fixtures. No synthetic replacement writer terminal is used.
    for scenario in ('child-limit', 'post-limit', 'source-limit', 'publication-limit', 'success'):
        r = fresh('schedule-' + scenario)
        tick = [0.0]
        captures, publications, quiescence = [], [], []
        def now(): return tick[0]
        def checked(argv, output, deadline):
            if output.name == 'identity-before':
                tick[0] = 5.0
            else:
                captures.append(('post', tick[0], deadline))
                tick[0] += 60 if scenario == 'post-limit' else 1
                F.require(tick[0] < deadline, 'STOP fixture checked timeout')
        def capture(argv, outputs, deadline, **kw):
            captures.append(('child', tick[0], deadline))
            if scenario == 'child-limit':
                tick[0] = deadline
                F.require(tick[0] < deadline, 'STOP fixture child timeout')
            tick[0] += 1
            return 1, 42
        source_calls = []
        def verify():
            source_calls.append(tick[0])
            if len(source_calls) == 2 and scenario == 'source-limit': tick[0] += 61
        def quiet(deadline):
            quiescence.append(deadline)
            return {'complete': True}
        def write(path, state):
            publications.append(dict(state))
            if path.name == 'pab-terminal.slot' and scenario == 'publication-limit': tick[0] += 61
        class Thread:
            def __init__(self, **kw): pass
            def start(self): pass
            def join(self, timeout): check(timeout == 1, 'join bound')
            def is_alive(self): return False
        original_read = Path.read_text
        def read(path, *a, **kw):
            return 'MemAvailable: 8388608 kB' if str(path) == '/proc/meminfo' else original_read(path, *a, **kw)
        with contextlib.ExitStack() as stack:
            for obj, name, value in [(F.os, 'getpid', lambda: 1), (F.os, 'getuid', lambda: 10001),
                    (F.os, 'getgid', lambda: 10001), (F.time, 'monotonic', now),
                    (F.signal, 'signal', lambda *a: None), (F.signal, 'setitimer', lambda *a: None),
                    (F, 'initialized', lambda *a: None), (F, 'slot_create', lambda *a: None),
                    (F, 'slot_write', write), (F, 'verify_source', verify),
                    (F, 'checked_program', checked), (F, 'capture', capture), (F, 'quiesce', quiet),
                    (F, 'read_regular', lambda *a: b'identical'), (F.threading, 'Thread', Thread),
                    (Path, 'read_text', read)]:
                stack.enter_context(patch.object(obj, name, value))
            result = F.writer(r, OWNER['token'], OWNER['invocation'])
        check(captures[0] == ('child', 5.0, 1205.0), 'early precheck lent child time')
        check(quiescence[0] <= 1290 and quiescence[0] <= (1205 if scenario == 'child-limit' else 6) + 30,
              'quiesce allowance changed')
        if scenario not in ('source-limit',):
            post = captures[1]
            check(post[2] == min(post[1] + 60, 1350), 'early child lent postcheck time')
        check(result == (0 if scenario == 'success' else 1), 'phase overrun accepted')
        if scenario in ('post-limit', 'source-limit'):
            check(len(publications) == 1, 'publication after postcheck deadline')
            check(not any(x.get('state') == 'TERMINAL' for x in publications), 'terminal after timeout')
        OBSERVATIONS.append({'actual_writer_schedule': scenario, 'captures': captures,
                             'quiesce_deadlines': quiescence, 'return': result,
                             'publication_states': [x['state'] for x in publications]})
CASES[-1] = (CASES[-1][0], writer_scheduling)


def main():
    F.enabled()
    os.umask(0o077)
    check(len(CASES) <= 32, 'matrix bound')
    selected = CASES
    prior = None
    if sys.argv[1:]:
        check(len(sys.argv) == 3 and sys.argv[1] == '--repair-scope', 'HOST fixed repair arguments')
        prior = json.loads(F.read_regular(sys.argv[2], F.META))
        failed = {r['case'] for r in prior['cases'] if r['result'] == 'RED'}
        # F1/F2 plus supervisor budget reservation and fresh source binding.
        affected = {'09-selector-failure', '32-writer-refuses-non-PID1-before-mutations',
                    '23-full-scripted-success-real-retention-and-handback',
                    '26-stopped-writer-export-failure-preserves-volume',
                    '28-final-publication-failure-keeps-pending',
                    '29-actual-source-context-private-copy-no-build',
                    '31-bounded-post-test-descendant-cleanup-fixtures'}
        selected = [(n, fn) for n, fn in CASES if n in failed | affected]

    start = time.monotonic()
    global ROOT
    ROOT = Path(tempfile.mkdtemp(prefix='pab-p-one-r16-HOST-', dir='/private/tmp'))
    ROOT.chmod(0o700)
    frozen = {p.name: L.stream_hash(p) for p in (D / 'launcher.py', D / 'finite-driver.py',
              D / 'host-check.py', D / 'source-manifest.json')}
    results = []
    # Guard the actual external entry points in addition to the fake adapter.
    def forbidden(*a, **kw): raise RuntimeError('HOST real external command forbidden')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('HOST case9s')))
    with patch.object(subprocess, 'Popen', forbidden), patch.object(L.Real, 'spawn', forbidden), \
         patch.object(os, 'system', forbidden), patch.object(os, 'killpg', forbidden):
        for name, fn in selected:
            check(time.monotonic() - start < 300, 'whole matrix budget')
            before = time.monotonic()
            try:
                signal.setitimer(signal.ITIMER_REAL, 9)
                fn()
                results.append({'case': name, 'result': 'PASS', 'seconds': time.monotonic() - before})
            except BaseException as exc:
                import traceback
                results.append({'case': name, 'result': 'RED', 'error': repr(exc), 'traceback': traceback.format_exc(),
                                'seconds': time.monotonic() - before})
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for fake in FAKES:
                    if not fake.finished:
                        try:
                            fake.done(remaining=True)
                        except BaseException as exc:
                            results[-1]['fd_cleanup_failure'] = repr(exc)
                            results[-1]['result'] = 'RED'
                            for proc in fake.processes:
                                proc.rescue()
                            fake.finished = True
    check(frozen == {p.name: L.stream_hash(p) for p in (D / 'launcher.py', D / 'finite-driver.py',
          D / 'host-check.py', D / 'source-manifest.json')}, 'source changed during matrix')
    receipt = {'tier': 'HOST-ONLY / FAKE EXTERNAL COMMANDS / NO PRODUCT CLAIM', 'root': str(ROOT),
               'python': sys.version, 'optimize': sys.flags.optimize, 'debug': __debug__, 'hashes': frozen,
               'maximum_scenarios': 32, 'maximum_seconds': 300, 'maximum_case_seconds': 9, 'cases': results, 'prior_receipt': sys.argv[2] if prior else None,
               'scope': 'F1/F2 plus supervisor reserve, latest full success/failure, source binding; prior RED retained' if prior else 'full fixed32',
               'seconds': time.monotonic() - start, 'fd_observations': OBSERVATIONS,
               'result': 'PASS' if all(r['result'] == 'PASS' for r in results) else 'RED',
               'actual_Docker_Git_mutations_product_imports_SQLite_pytest_network': 0,
               'physical_quota_claimed': False, 'product_acceptance': False}
    L.publish(ROOT / 'HOST-receipt.json', receipt, L.Budget())
    print(json.dumps({'root': str(ROOT), 'result': receipt['result'], 'cases': len(results),
                      'red': [r['case'] for r in results if r['result'] == 'RED']}))
    return 0 if receipt['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
