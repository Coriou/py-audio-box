"""Bounded operator overlay tests; no Docker or model imports."""
import ast
import contextlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/qwen_private_trace_overlay.py'
SOURCE = ROOT / 'worker/runtime_supervisor.py'


def helper(tmp_path):
    source = tmp_path / 'head.py'
    # CI runs from an exported archive without .git or ancestor objects.
    source.write_bytes(SOURCE.read_bytes())
    context = tmp_path / 'ctx'
    subprocess.run([sys.executable, str(SCRIPT), '--source', str(source), '--image', 'verified-local:base',
                    '--output', str(context)], check=True, capture_output=True)
    patched = (context / 'runtime_supervisor.py').read_text()
    compile(patched, '<overlay>', 'exec')
    assert patched.count('with private_child_stderr(') == 2
    assert "private_child_stderr('resident')" in patched
    assert "private_child_stderr('attempt')" in patched
    assert (context / 'Dockerfile').read_text().startswith('FROM verified-local:base\n')
    module = ast.parse(patched)
    nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and
             n.name in ('_trace_write', '_trace_state', 'private_child_stderr')]
    ns = {'os': os, 'threading': threading, 'contextmanager': contextlib.contextmanager,
          '_TRACE_LIMIT': 8192, '_TRACE_RESERVE': 32768, '_trace_lock': threading.Lock(),
          '_trace_owners': {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<helper>', 'exec'), ns)
    fn = ns['private_child_stderr'].__wrapped__
    consts = tuple(str(tmp_path) if c == '/tmp' else
                   str(tmp_path) + c[4:] if isinstance(c, str) and c.startswith('/tmp/qwen-private-') else c
                   for c in fn.__code__.co_consts)
    ns['private_child_stderr'] = contextlib.contextmanager(types.FunctionType(
        fn.__code__.replace(co_consts=consts), ns))
    return ns


def settled(ns, role):
    owner = ns['_trace_owners'][role]
    owner.join(2)
    assert not owner.is_alive()
    return role


def test_dual_concurrent_overflow_and_live_bytes(tmp_path):
    ns = helper(tmp_path)
    with ns['private_child_stderr']('resident') as a, ns['private_child_stderr']('attempt') as b:
        os.write(a, b'A' * 20000)
        os.write(b, b'B' * 20000)
        for role in ('resident', 'attempt'):
            path = tmp_path / f'qwen-private-{role}.state'
            deadline = time.monotonic() + 2
            while 'bytes=20000' not in path.read_text() and time.monotonic() < deadline:
                time.sleep(.01)
            assert 'open bytes=20000' in path.read_text()
    for role, byte in (('resident', b'A'), ('attempt', b'B')):
        settled(ns, role)
        assert (tmp_path / f'qwen-private-{role}.tail').read_bytes() == byte * 8192
        assert 'complete bytes=20000 truncated=1' in (tmp_path / f'qwen-private-{role}.state').read_text()


def test_old_descendant_reader_refuses_relaunch(tmp_path):
    ns = helper(tmp_path)
    with ns['private_child_stderr']('resident') as fd:
        old = os.dup(fd)
        os.write(fd, b'OLD')
    try:
        with pytest.raises(RuntimeError, match='previous_reader_alive'):
            with ns['private_child_stderr']('resident'):
                pass
        assert (tmp_path / 'qwen-private-resident.tail').read_bytes() == b'OLD'
    finally:
        os.close(old)
        settled(ns, 'resident')
    with ns['private_child_stderr']('resident') as fd:
        os.write(fd, b'NEW')
    settled(ns, 'resident')
    assert (tmp_path / 'qwen-private-resident.tail').read_bytes() == b'NEW'


def test_short_and_zero_writes(tmp_path):
    ns = helper(tmp_path)
    real = os.write
    def short(fd, data):
        return real(fd, data[:max(1, len(data)//3)])
    ns['os'] = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith('__')})
    ns['os'].write = short
    with ns['private_child_stderr']('attempt') as fd:
        real(fd, b'X' * 9000)
    settled(ns, 'attempt')
    assert (tmp_path / 'qwen-private-attempt.tail').read_bytes() == b'X' * 8192
    ns['os'].write = lambda fd, data: 0
    with pytest.raises(OSError, match='no_progress'):
        with ns['private_child_stderr']('attempt'):
            pass
    assert not (tmp_path / 'qwen-private-attempt.state').read_text().startswith('complete')


def test_launch_failure_and_low_budget(tmp_path):
    ns = helper(tmp_path)
    with pytest.raises(ValueError, match='launch'):
        with ns['private_child_stderr']('attempt'):
            raise ValueError('launch')
    settled(ns, 'attempt')
    assert (tmp_path / 'qwen-private-attempt.state').read_text() == 'error launch\n'
    real = ns['os']
    fake = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith('__')})
    fake.statvfs = lambda path: types.SimpleNamespace(f_bavail=1, f_frsize=4096, f_favail=1)
    ns['os'] = fake
    with pytest.raises(RuntimeError, match='budget'):
        with ns['private_child_stderr']('resident'):
            pass
    ns['os'] = real


class TrackedOS:
    def __init__(self):
        self.live = set()
        self.paths = {}
        self.fail = None

    def __getattr__(self, name):
        return getattr(os, name)

    def open(self, path, flags, mode=0o777):
        fd = os.open(path, flags, mode)
        self.live.add(fd)
        self.paths[fd] = str(path)
        return fd

    def pipe(self):
        pair = os.pipe()
        self.live.update(pair)
        return pair

    def close(self, fd):
        os.close(fd)
        self.live.remove(fd)
        self.paths.pop(fd, None)

    def write(self, fd, data):
        if self.fail == 'sink' and self.paths.get(fd, '').endswith('.tail'):
            raise OSError('injected sink')
        return os.write(fd, data)

    def read(self, fd, size):
        if self.fail == 'reader':
            raise OSError('injected reader')
        return os.read(fd, size)


@pytest.mark.parametrize('failure', ['sink', 'reader', 'thread_start', 'launch'])
def test_failure_state_and_descriptor_ownership(tmp_path, failure):
    ns = helper(tmp_path)
    tracked = TrackedOS()
    ns['os'] = tracked
    if failure == 'thread_start':
        class RefusingThread:
            def __init__(self, **kwargs): pass
            def start(self): raise RuntimeError('injected start')
            def is_alive(self): return False
        ns['threading'] = types.SimpleNamespace(Event=threading.Event, Thread=RefusingThread)
        with pytest.raises(RuntimeError, match='injected start'):
            with ns['private_child_stderr']('attempt'): pass
    elif failure == 'launch':
        with pytest.raises(RuntimeError, match='injected launch'):
            with ns['private_child_stderr']('attempt'):
                raise RuntimeError('injected launch')
        settled(ns, 'attempt')
    else:
        tracked.fail = failure
        with ns['private_child_stderr']('attempt') as fd:
            if failure == 'sink': os.write(fd, b'EXCEPTION')
        settled(ns, 'attempt')
    assert not tracked.live
    state = (tmp_path / 'qwen-private-attempt.state').read_text()
    assert state == {'sink': 'error sink\n', 'reader': 'error reader\n',
                     'thread_start': 'error setup\n', 'launch': 'error launch\n'}[failure]


@pytest.mark.parametrize('available,inodes', [(32767, 64), (65536, 3)])
def test_budget_refuses_before_allocating(tmp_path, available, inodes):
    ns = helper(tmp_path)
    tracked = TrackedOS()
    tracked.statvfs = lambda path: types.SimpleNamespace(f_bavail=available, f_frsize=1, f_favail=inodes)
    ns['os'] = tracked
    with pytest.raises(RuntimeError, match='budget_unavailable'):
        with ns['private_child_stderr']('attempt'): pass
    assert not tracked.live
    assert not list(tmp_path.glob('qwen-private-*'))


def test_subprocess_capture_allocation_and_cleanup(tmp_path):
    ns = helper(tmp_path)
    tracked = TrackedOS()
    ns['os'] = tracked
    children = []
    try:
        with ns['private_child_stderr']('resident') as a, ns['private_child_stderr']('attempt') as b:
            for fd in (a, b):
                children.append(subprocess.Popen([sys.executable, '-c',
                    'import os; os.write(2, b"E" * 131072)'], stderr=fd,
                    stdout=subprocess.DEVNULL))
            for child in children:
                assert child.wait(timeout=5) == 0
    finally:
        for child in children:
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
        for role in tuple(ns['_trace_owners']): settled(ns, role)
    assert not tracked.live
    files = list(tmp_path.glob('qwen-private-*'))
    assert len(files) == 4
    assert sum(p.stat().st_blocks * 512 for p in files) <= 32768
    assert sum(p.stat().st_size for p in files) < 17000
    for role in ('attempt', 'resident'):
        assert (tmp_path / f'qwen-private-{role}.tail').read_bytes() == b'E' * 8192
        assert (tmp_path / f'qwen-private-{role}.state').read_text() == 'complete bytes=131072 truncated=1\n'
