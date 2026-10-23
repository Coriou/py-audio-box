"""Prospective evidence guard tests. Signal statuses here are synthetic only."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import _maintenance_wait_guard as harness


MODEL = '''
import os, sys
from contextlib import contextmanager
from service import recovery
sys.platform = 'linux'
recovery._install_maintenance_limit = lambda: None
@contextmanager
def session(*a, **k): yield None
recovery._maintenance_session = session
'''


def isolated_guard(tmp_path):
    node = SimpleNamespace(nodeid='synthetic-fresh', originalname='synthetic_fresh')
    guard = harness.MaintenanceWaitGuard(node, SimpleNamespace(shouldstop=False))
    return guard, harness.activate(guard, tmp_path / 'helpers')


@pytest.mark.parametrize('mode', ['normal', 'semantic', 'nested'])
def test_fresh_bootstrap_owned_receipts(tmp_path, mode):
    guard, active = isolated_guard(tmp_path)
    body = MODEL + '''
unrelated = os.fork()
if unrelated == 0: os._exit(0)
def operation(value):
    unrelated_inside = os.fork()
    if unrelated_inside == 0: os._exit(0)
    assert os.waitpid(unrelated_inside, 0) == (unrelated_inside, 0)
    return True
'''
    if mode == 'semantic':
        body += '''
def operation(value): raise ValueError('test semantic boundary')
try: recovery._run_bounded_maintenance({}, {}, operation)
except ValueError: pass
else: raise AssertionError('refusal absent')
'''
    else:
        body += 'assert recovery._run_bounded_maintenance({}, {}, operation)\n'
    body += 'assert os.waitpid(unrelated, 0) == (unrelated, 0)\n'
    if mode == 'nested':
        body += ('from _maintenance_wait_guard import fresh_run\n'
                 'assert fresh_run([sys.executable, "-c", ' + repr(MODEL +
                 'assert recovery._run_bounded_maintenance({}, {}, lambda value: True)') +
                 ']).returncode == 0\n')
    with active:
        result = harness.fresh_run([sys.executable, '-c', body], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not guard.failures
    assert len(guard.receipts) == (2 if mode == 'nested' else 1)
    assert all(status == ((74 << 8) if mode == 'semantic' else 0) for _, status in guard.receipts)
    directories = list((tmp_path / 'helpers').rglob('ack.json'))
    assert len(directories) == (2 if mode == 'nested' else 1)
    for ack in directories:
        assert harness.aggregate(ack.parent, harness.metadata(ack.parent / "launch.json")["identity"])


@pytest.mark.parametrize('mode', ['signal', 'missing-semantic', 'missing-done', 'orphan'])
def test_fresh_fail_closed_parent_latch(tmp_path, mode):
    guard, active = isolated_guard(tmp_path)
    if mode == 'missing-done':
        body = 'import os; os._exit(0)'
    else:
        body = 'import _maintenance_wait_guard as h\ng = h._CURRENT\ng.owned.add(123)\ng.event("enroll", pid=123)\n'
        if mode != 'orphan':
            body += 'g.observe((123, ' + ('11' if mode == 'signal' else str(74 << 8)) + '))\n'
    with active:
        result = harness.fresh_run([sys.executable, '-c', body], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert guard.failures and guard.session.shouldstop
        with pytest.raises(RuntimeError):
            harness.fresh_run([sys.executable, '-c', 'raise AssertionError("must not launch")'])
    assert 'fresh helper receipt failure' in guard.failures[0]


@pytest.mark.parametrize('defect', ['missing-ack', 'missing-terminal', 'unowned', 'duplicate', 'oversize'])
def test_receipt_validation(tmp_path, defect):
    harness.atomic(tmp_path / 'ack.json', dict(pid=1, node='synthetic', token='a'*32))
    harness.atomic(tmp_path / 'done.json', dict(pid=1, node='synthetic', token='a'*32))
    if defect == 'missing-ack': (tmp_path / 'ack.json').unlink()
    if defect in ('missing-terminal', 'duplicate'):
        harness.atomic(tmp_path / '1-1.json', dict(kind='enroll', owner=1, pid=2))
    if defect in ('unowned', 'duplicate'):
        harness.atomic(tmp_path / '1-2.json', dict(kind='terminal', owner=1, pid=2, status=0, accepted=True, phase=''))
    if defect == 'duplicate':
        harness.atomic(tmp_path / '1-3.json', dict(kind='terminal', owner=1, pid=2, status=0, accepted=True, phase=''))
    if defect == 'oversize': (tmp_path / '1-1.json').write_bytes(b'x' * 4097)
    with pytest.raises((ValueError, OSError)):
        harness.aggregate(tmp_path, dict(pid=1, node='synthetic', token='a'*32))


def test_concurrent_fresh_helpers_keep_receipts_separate(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    guard, active = isolated_guard(tmp_path)
    body = MODEL + 'assert recovery._run_bounded_maintenance({}, {}, lambda value: True)\n'
    with active, ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: harness.fresh_run(
            [sys.executable, '-c', body], capture_output=True, text=True, timeout=15), range(2)))
    assert all(result.returncode == 0 for result in results)
    assert not guard.failures and len(guard.receipts) == 2
    assert len({pid for pid, _ in guard.receipts}) == 2
    assert len(list((tmp_path / 'helpers').glob('helper-*/done.json'))) == 2


def test_fresh_assertion_cannot_be_semantic_negative(tmp_path):
    guard, active = isolated_guard(tmp_path)
    sentinel = tmp_path / 'writer'
    body = MODEL + '''
def operation(value): raise AssertionError('test sentinel')
try: recovery._run_bounded_maintenance({}, {}, operation)
except ValueError: pass
''' + f'from pathlib import Path; Path({str(sentinel)!r}).write_text("continued")\n'
    with active:
        result = harness.fresh_run([sys.executable, '-c', body], capture_output=True, text=True, timeout=15)
    assert result.returncode == 86
    assert not sentinel.exists()
    assert guard.failures and guard.session.shouldstop
    assert 'status=18944' in guard.failures[0]


@pytest.mark.parametrize('defect', ['foreign-pair', 'ack-node', 'ack-pid', 'ack-bool',
    'ack-zero', 'ack-large-pid', 'ack-extra', 'ack-token', 'ack-oversize', 'done-oversize',
    'ack-malformed', 'done-malformed', 'duplicate-key', 'duplicate-enroll',
    'owner-bool', 'child-bool', 'status-bool', 'status-bound', 'status-stopped',
    'extra-field', 'crossdir', 'nested-owner', 'nested-node', 'nested-pair'])
def test_complete_receipts_bound_to_launch(tmp_path, defect):
    expected = dict(pid=100, node='owned-helper', token='a'*32)
    ack = dict(expected)
    if defect == 'ack-node': ack['node'] = 'another-node'
    if defect == 'ack-pid': ack['pid'] = 101
    if defect == 'ack-bool': ack['pid'] = True
    if defect == 'ack-zero': ack['pid'] = 0
    if defect == 'ack-large-pid': ack['pid'] = 2**31
    if defect == 'ack-extra': ack['extra'] = 0
    if defect == 'ack-token': ack['token'] = 'b'*32
    harness.atomic(tmp_path/'ack.json', ack)
    harness.atomic(tmp_path/'done.json', ack)
    enroll = dict(kind='enroll', owner=100, pid=777)
    terminal = dict(kind='terminal', owner=100, pid=777, status=0, accepted=True, phase='')
    if defect == 'foreign-pair': enroll['owner'] = terminal['owner'] = 999
    if defect == 'owner-bool': enroll['owner'] = terminal['owner'] = True
    if defect == 'child-bool': enroll['pid'] = terminal['pid'] = True
    if defect == 'status-bool': terminal['status'] = False
    if defect == 'status-bound': terminal['status'] = 65536
    if defect == 'status-stopped': terminal['status'] = 0x137f
    if defect == 'extra-field': terminal['extra'] = 0
    harness.atomic(tmp_path/'100-1.json', enroll)
    harness.atomic(tmp_path/'100-2.json', terminal)
    if defect == 'duplicate-enroll': harness.atomic(tmp_path/'100-3.json', enroll)
    if defect.endswith('-oversize'):
        (tmp_path/(defect.split('-')[0]+'.json')).write_bytes(b' '*4097)
    if defect.endswith('-malformed'):
        (tmp_path/(defect.split('-')[0]+'.json')).write_bytes(b'{')
    if defect == 'duplicate-key':
        (tmp_path/'ack.json').write_text('{"pid":100,"pid":100,"node":"owned-helper","token":"'+'a'*32+'"}')
    if defect in ('crossdir', 'nested-owner', 'nested-node', 'nested-pair'):
        child = tmp_path/'nested'/'helper-child'
        child.mkdir(parents=True, mode=0o700)
        child_id = dict(pid=200, node=expected['node'], token='b'*32)
        launch_id = dict(child_id)
        if defect == 'nested-node': launch_id['node'] = 'foreign-node'
        harness.atomic(child/'launch.json', dict(parent=999 if defect == 'nested-owner' else 100, identity=launch_id))
        harness.atomic(child/'ack.json', child_id)
        harness.atomic(child/'done.json', child_id)
        if defect == 'crossdir':
            (tmp_path/'100-2.json').replace(child/'100-2.json')
        else:
            harness.atomic(child/'200-1.json', dict(kind='enroll', owner=999 if defect == 'nested-pair' else 200, pid=888))
            harness.atomic(child/'200-2.json', dict(kind='terminal', owner=999 if defect == 'nested-pair' else 200, pid=888, status=0, accepted=True, phase=''))
    with pytest.raises(ValueError): harness.aggregate(tmp_path, expected)


@pytest.mark.parametrize('mode', ['normal', 'assertion', 'wrong-phase', 'wrong-class',
    'wrong-errno', 'missing-symlink', 'missing-witness', 'wrong-node', 'wrong-param'])
def test_journal_protected_read_fresh_phase(tmp_path, mode):
    nodeid = 'tests/service/test_recovery.py::test_lotp_early_refusals_precede_original_writer[journal-symlink]'
    node = SimpleNamespace(nodeid=nodeid,
        originalname='test_lotp_early_refusals_precede_original_writer',
        callspec=SimpleNamespace(params={'failure': 'journal-symlink'}))
    if mode == 'wrong-node': node.nodeid = nodeid.replace('journal-symlink]', 'another]')
    if mode == 'wrong-param': node.callspec.params['failure'] = 'another'
    guard = harness.MaintenanceWaitGuard(node, SimpleNamespace(shouldstop=False))
    sentinel = tmp_path/'continued'
    # The unchanged real row retains its writer trap, logical SQL and source-byte
    # assertions. --noconftest keeps this explicit helper's guard as sole owner.
    body = """
import errno, os, pytest
from pathlib import Path
from service import recovery
import _maintenance_wait_guard as h
mode = MODE
class Plugin:
    @pytest.fixture(autouse=True)
    def fault(self, monkeypatch):
        if mode == 'wrong-phase':
            def wrong(*a, **k): raise OSError(errno.ELOOP, 'synthetic phase')
            monkeypatch.setattr(recovery, '_preparation_prestate', wrong)
        if mode == 'missing-witness':
            observe = h._CURRENT.observe
            def missing(result, **kw):
                kw['witness'] = b''
                return observe(result, **kw)
            monkeypatch.setattr(h._CURRENT, 'observe', missing)
        open0 = os.open
        class OtherOS(OSError): pass
        def opened(path, flags, *a, **k):
            if str(path).endswith('/journal.json') and Path(path).is_symlink():
                if mode == 'assertion': raise AssertionError('synthetic class')
                if mode == 'wrong-class': raise OtherOS(errno.ELOOP, 'synthetic subclass')
                if mode == 'wrong-errno': raise OSError(errno.EIO, 'synthetic errno')
                if mode == 'missing-symlink':
                    Path(path).unlink(); Path(path).write_bytes(b'{}')
                    raise OSError(errno.ELOOP, 'synthetic absent symlink')
            return open0(path, flags, *a, **k)
        monkeypatch.setattr(os, 'open', opened)
result = pytest.main(['--noconftest', '-p', 'no:cacheprovider', NODE,
    '-q', '-rs', '-x', '--basetemp='+BASE], plugins=[Plugin()])
if h._CURRENT.failures: raise SystemExit(86)
assert result == 0
Path(SENTINEL).write_text('continued')
""".replace('MODE', repr(mode)).replace('NODE', repr(nodeid)).replace('BASE', repr(str(tmp_path/'inner'))).replace('SENTINEL', repr(str(sentinel)))
    with harness.activate(guard, tmp_path/'helpers'):
        result = harness.fresh_run([sys.executable, '-c', body], capture_output=True, text=True, timeout=30)
    if mode == 'normal':
        assert result.returncode == 0, result.stdout + result.stderr
        assert sentinel.exists() and not guard.failures
        assert len(guard.receipts) == 1 and guard.receipts[0][1] == 74 << 8
        events = [json.loads(p.read_text()) for p in (tmp_path/'helpers').glob('helper-*/[0-9]*.json')]
        assert any(e.get('phase') == 'J:prewrite-journal-nofollow-ELOOP;' for e in events)
    else:
        assert result.returncode == 86, result.stdout + result.stderr
        assert not sentinel.exists()
        assert guard.failures and guard.session.shouldstop
        assert 'status=18944' in guard.failures[0]


@pytest.mark.parametrize('record', ['ack', 'done'])
def test_identity_size_refused_before_read(tmp_path, monkeypatch, record):
    expected = dict(pid=100, node='owned-helper', token='a'*32)
    for name in ('ack', 'done'): harness.atomic(tmp_path/(name+'.json'), expected)
    target = tmp_path/(record+'.json')
    target.write_bytes(b' '*4097)
    inode = target.stat().st_ino
    read = os.read
    def checked(fd, length):
        assert os.fstat(fd).st_ino != inode, 'oversize identity read before bound'
        return read(fd, length)
    monkeypatch.setattr(os, 'read', checked)
    with pytest.raises(ValueError, match='bound'): harness.aggregate(tmp_path, expected)


@pytest.mark.parametrize('field', ['node', 'pid', 'token'])
def test_fresh_matching_forged_identity_rejected(tmp_path, field):
    guard, active = isolated_guard(tmp_path)
    body = '''
import _maintenance_wait_guard as h
value = h.metadata(h._CURRENT.directory/'ack.json')
value[FIELD] = REPLACEMENT
original = h.atomic
original(h._CURRENT.directory/'ack.json', value)
def altered(path, record):
    if path.name == 'done.json': record = value
    return original(path, record)
h.atomic = altered
'''.replace('FIELD', repr(field)).replace('REPLACEMENT', repr({'node':'foreign', 'pid':999, 'token':'b'*32}[field]))
    with active:
        result = harness.fresh_run([sys.executable, '-c', body], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0
    assert guard.failures and guard.session.shouldstop
    assert 'launch/acknowledgement/completion mismatch' in guard.failures[0]


def test_fresh_timeout_preserves_exception_and_rejects_missing_completion(tmp_path):
    import subprocess
    guard, active = isolated_guard(tmp_path)
    with active, pytest.raises(subprocess.TimeoutExpired):
        harness.fresh_run([sys.executable, '-c', 'import time; time.sleep(10)'],
            capture_output=True, text=True, timeout=1)
    assert guard.failures and guard.session.shouldstop
    directory, = (tmp_path/'helpers').glob('helper-*')
    assert harness.metadata(directory/'ack.json')['pid'] == harness.metadata(directory/'launch.json')['identity']['pid']
    assert not (directory/'done.json').exists()


@pytest.mark.parametrize('fault', ['EPIPE', 'EAGAIN', 'assertion', 'unrelated-os',
    'wrong-phase', 'wrong-node', 'missing', 'signal'])
def test_backpressure_exact_source_phase(fault):
    import errno
    import types
    import test_recovery
    node = SimpleNamespace(nodeid='tests/service/test_recovery.py::test_t1_actual_pipe_transport[backpressure]',
        originalname='test_t1_actual_pipe_transport', callspec=SimpleNamespace(params={'mode':'backpressure'}))
    function = test_recovery.test_t1_actual_pipe_transport
    code, = [c for c in function.__code__.co_consts if isinstance(c, types.CodeType) and c.co_name == 'changed_write']
    error = {'EPIPE': BrokenPipeError(errno.EPIPE, 'unit'),
        'EAGAIN': BlockingIOError(errno.EAGAIN, 'unit'),
        'assertion': AssertionError('unit'), 'unrelated-os': OSError(errno.EIO, 'unit')}.get(fault, BrokenPipeError(errno.EPIPE, 'unit'))
    def write(*args): raise error
    values = dict(state={'write':0}, mode='backpressure', write=write,
        original_select=None, time=None, signal=None)
    def cell(value): return (lambda: value).__closure__[0]
    changed = types.FunctionType(code, function.__globals__, closure=tuple(cell(values[n]) for n in code.co_freevars))
    try:
        if fault == 'wrong-phase': raise error
        changed(123, b'unit')
    except BaseException as exc:
        if fault == 'wrong-node': node.nodeid = 'foreign'
        witness = harness.backpressure_phase(node, exc)
    else: raise AssertionError('unit fault missing')
    if fault == 'missing': witness = b''
    guard = harness.MaintenanceWaitGuard(node, SimpleNamespace(shouldstop=False))
    guard.owned.add(123)
    status = 11 if fault == 'signal' else 74 << 8
    actual = (123, status)
    assert guard.observe(actual, witness=witness,
        expected_exit=74 if witness in harness._BACKPRESSURE_PHASES else None) is actual
    if fault in ('EPIPE', 'EAGAIN'):
        assert witness in harness._BACKPRESSURE_PHASES and not guard.failures
    else:
        assert guard.failures and guard.session.shouldstop


@pytest.mark.parametrize('mode', ['normal', 'assertion', 'other-sqlite', 'wrong-args',
    'wrong-function', 'outside-private', 'original-db', 'wrong-node', 'wrong-param',
    'missing-marker', 'forged-code-name', 'cleanup-replacement', 'signal-status',
    'wrong-owner', 'missing-helper-done'])
def test_modeled_rehearsal_completion_fresh_contract(tmp_path, mode):
    nodeid = 'tests/service/test_recovery.py::test_lotp_late_refusal_retains_unaccepted_plan[rehearsal]'
    node = SimpleNamespace(nodeid=nodeid,
        originalname='test_lotp_late_refusal_retains_unaccepted_plan',
        callspec=SimpleNamespace(params={'failure': 'rehearsal'}))
    if mode == 'wrong-node': node.nodeid = 'foreign::' + nodeid
    if mode == 'wrong-param': node.callspec.params['failure'] = 'snapshot'
    guard = harness.MaintenanceWaitGuard(node, SimpleNamespace(shouldstop=False))
    harness.atomic(tmp_path/'modeled-control.json', dict(mode=mode,
        synthetic_wait_status=mode == 'signal-status'))
    sentinel = tmp_path/'continued'
    body = r'''
import os, sys, sqlite3, types, pytest
from pathlib import Path
from contextlib import contextmanager
from service import recovery
import _maintenance_wait_guard as h
mode = MODE
class Plugin:
    @pytest.fixture(autouse=True)
    def fault(self, monkeypatch):
        run = recovery._run_bounded_maintenance
        rehearsal = recovery._rehearse_preparation
        session = recovery._maintenance_session
        def changed_run(roots, scope, operation, **kw):
            installed = recovery._write_core_completion
            if mode in ('assertion', 'other-sqlite', 'wrong-args'):
                error = {'assertion': AssertionError('modeled private capacity failure'),
                    'other-sqlite': sqlite3.DatabaseError('modeled private capacity failure'),
                    'wrong-args': sqlite3.OperationalError('different modeled args')}[mode]
                # Same collected code object, different raised class/args. This
                # challenges the exception predicates independently of code identity.
                proxy = types.SimpleNamespace(OperationalError=lambda *a: error)
                installed = types.FunctionType(installed.__code__,
                    {**installed.__globals__, 'sqlite3': proxy})
                monkeypatch.setattr(recovery, '_write_core_completion', installed)
            if mode in ('wrong-function', 'forged-code-name'):
                def wrong(*args): raise sqlite3.OperationalError('modeled private capacity failure')
                if mode == 'forged-code-name':
                    wrong = types.FunctionType(wrong.__code__.replace(
                        co_name=installed.__code__.co_name,
                        co_qualname=installed.__code__.co_qualname), globals())
                monkeypatch.setattr(recovery, '_write_core_completion', wrong)
            if mode == 'original-db':
                connect = sqlite3.connect
                def original_database(path, *a, **kw):
                    if '/p1-rehearsal-' in str(path) and str(path).endswith('/core.sqlite?mode=rw&vfs=unix'):
                        path = (roots['control']/'core'/'core.sqlite').as_uri()+'?mode=rw&vfs=unix'
                    return connect(path, *a, **kw)
                monkeypatch.setattr(sqlite3, 'connect', original_database)
            if mode == 'outside-private':
                def outside(value, plan, intent, frozen):
                    recovery._write_core_completion(value.db, roots['control']/'core', intent)
                monkeypatch.setattr(recovery, '_rehearse_preparation', outside)
            return run(roots, scope, operation, **kw)
        monkeypatch.setattr(recovery, '_run_bounded_maintenance', changed_run)
        if mode == 'cleanup-replacement':
            # A real seam marker is emitted first; session cleanup then replaces
            # it with ValueError, which has an otherwise accepted legacy V phase.
            @contextmanager
            def replacement(*a, **kw):
                with session(*a, **kw) as value:
                    try: yield value
                    except sqlite3.OperationalError:
                        raise ValueError('synthetic replacing cleanup')
            monkeypatch.setattr(recovery, '_maintenance_session', replacement)
        if mode in ('missing-marker', 'signal-status'):
            observe = h._CURRENT.observe
            def changed(result, **kw):
                if mode == 'missing-marker': kw['witness'] = b''
                # Synthetic wait status only; no signal is generated or sent.
                if mode == 'signal-status': result = (result[0], 11)
                return observe(result, **kw)
            monkeypatch.setattr(h._CURRENT, 'observe', changed)
        if mode == 'wrong-owner':
            event = h._CURRENT.event
            def foreign(kind, **data):
                if kind in ('enroll', 'terminal'):
                    h._CURRENT.sequence += 1
                    h.atomic(h._CURRENT.directory/f'{os.getpid()}-{h._CURRENT.sequence}.json',
                        dict(kind=kind, owner=os.getpid()+100000, **data))
                else: event(kind, **data)
            monkeypatch.setattr(h._CURRENT, 'event', foreign)
        if mode == 'missing-helper-done':
            atomic = h.atomic
            def missing(path, value):
                if path.name != 'done.json': return atomic(path, value)
            # Persist through pytest fixture teardown until bootstrap atexit.
            h.atomic = missing
result = pytest.main(['--noconftest', '-p', 'no:cacheprovider', NODE,
    '-q', '-rs', '-x', '--basetemp='+BASE], plugins=[Plugin()])
if h._CURRENT.failures:
    assert h._CURRENT.session.shouldstop
    # Parent ValueError was caught by the target; this latch lives outside it.
    with pytest.raises(pytest.exit.Exception):
        recovery._run_bounded_maintenance({}, {}, lambda value: True)
    raise SystemExit(86)
assert result == 0
Path(SENTINEL).write_text('continued')
'''.replace('MODE', repr(mode)).replace('NODE', repr(nodeid)).replace('BASE', repr(str(tmp_path/'inner'))).replace('SENTINEL', repr(str(sentinel)))
    with harness.activate(guard, tmp_path/'helpers'):
        result = harness.fresh_run([sys.executable, '-c', body], capture_output=True, text=True, timeout=30)
        if mode == 'normal':
            assert result.returncode == 0, result.stdout + result.stderr
            assert sentinel.exists() and not guard.failures
            assert len(guard.receipts) == 1 and guard.receipts[0][1] == 74 << 8
        else:
            assert result.returncode == (0 if mode in ('wrong-owner', 'missing-helper-done') else 86), result.stdout + result.stderr
            assert guard.failures and guard.session.shouldstop
            with pytest.raises(RuntimeError):
                harness.fresh_run([sys.executable, '-c', 'raise AssertionError("must not launch")'])
            if mode not in ('wrong-owner', 'missing-helper-done'): assert not sentinel.exists()
    directory, = (tmp_path/'helpers').glob('helper-*')
    events = [json.loads(p.read_text()) for p in directory.glob('[0-9]*.json')]
    terminals = [e for e in events if e['kind'] == 'terminal']
    assert len(terminals) == 1
    terminal, = terminals
    if mode in ('normal', 'wrong-owner', 'missing-helper-done'):
        assert harness._MODELED_REHEARSAL_FINAL.decode() in terminal['phase']
        assert terminal['status'] == 74 << 8 and terminal['accepted']
    elif mode == 'cleanup-replacement':
        assert 'completion-seam;' in terminal['phase']
        assert 'completion-final;' not in terminal['phase']
        assert not terminal['accepted']
    elif mode == 'signal-status':
        assert harness._MODELED_REHEARSAL_FINAL.decode() in terminal['phase']
        assert terminal['status'] == 11 and not terminal['accepted']
    else:
        assert harness._MODELED_REHEARSAL_FINAL.decode() not in terminal['phase']
        assert not terminal['accepted']
