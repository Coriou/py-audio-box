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


# Family P only. Observer controls below are synthetic rejection evidence;
# the separate fresh normal case executes the unchanged five-step fixture.
@pytest.mark.parametrize('defect', ['normal', 'missing', 'no-seam', 'no-final',
    'final-only', 'duplicate', 'truncated', 'overflow', 'wrong-order', 'exit0', 'signal11'])
def test_P_ordered_armed_transcript(defect):
    node = SimpleNamespace(nodeid='P-observer', originalname='P-observer')
    guard = harness.MaintenanceWaitGuard(node, SimpleNamespace(shouldstop=False))
    guard.owned.add(123)
    tokens = b'P:a;P:s;P:f;'
    witness = {'missing': b'V', 'no-seam': b'P:a;P:f;V',
        'no-final': b'P:a;P:s;V', 'final-only': b'P:f;V',
        'duplicate': tokens + b'P:f;', 'truncated': tokens[:-1],
        'overflow': tokens + b'V'*257, 'wrong-order': b'P:a;P:f;P:s;'}.get(defect, tokens)
    status = 0 if defect == 'exit0' else 11 if defect == 'signal11' else 74 << 8
    raw = (123, status)
    assert guard.observe(raw, witness=witness, armed_P=True, expected_exit=74,
        expected_signal=11) is raw
    assert guard.receipts == [raw] and not guard.owned
    assert bool(guard.failures) == (defect != 'normal')
    assert bool(guard.session.shouldstop) == (defect != 'normal')


@pytest.mark.parametrize('mode', ['normal', 'wrong-class', 'subclass', 'wrong-args',
    'real-EPIPE', 'wrong-code', 'wrong-node', 'wrong-collected', 'missing-collected', 'precompletion',
    'durable-id', 'missing-sentinel', 'wrong-fd', 'reused-fd', 'witness-pipe',
    'operation-failure', 'cleanup-value', 'cleanup-nonsemantic', 'cleanup-new-object',
    'changed-route', 'changed-attributes', 'wrong-final-caller', 'bad-offset',
    'bad-slice', 'frame-budget', 'pipe-mode', 'wrong-operation-code',
    'missing-markers', 'signal11', 'foreign-owner', 'missing-done', 'operation-value', 'setup-evidence-failure'])
def test_P_durable_lost_result_fresh_contract(tmp_path, mode):
    nodeid = 'tests/service/test_recovery.py::test_t1_durable_completion_lost_result_no_false_success'
    node = SimpleNamespace(nodeid=nodeid, originalname=nodeid.split('::')[-1],
        callspec=SimpleNamespace(params={}))
    guard = harness.MaintenanceWaitGuard(node, SimpleNamespace(shouldstop=False))
    sentinel = tmp_path/'continued-P'
    proof = tmp_path/'directed-P'
    proof.mkdir(mode=0o700)
    # Evidence is bounded, test-owned, and separate from the receipt schema.
    # Predicate-only: bad-offset/slice/frame-budget, durable-id,
    # wrong-operation-code/final-caller. Real-route representatives remain below.
    body = r'''
import os, sys, types, errno, stat, pytest
from pathlib import Path
from service import recovery
import _maintenance_wait_guard as h
mode = MODE
steps = []
proof_owner = os.getpid()
def evidence(label):
    global proof_owner, steps
    if proof_owner != os.getpid():
        proof_owner, steps = os.getpid(), []
    assert label not in steps and len(steps) < 24
    steps.append(label)
    h.atomic(Path(PROOF)/(str(os.getpid())+'.json'), dict(mode=mode, steps=steps))
setup_reports = []
class Plugin:
    def pytest_runtest_makereport(self, item, call):
        if mode != 'setup-evidence-failure': return
        setup_reports.append(call.when)
        assert call.when in ('setup', 'teardown')  # No target call is allowed.
        if call.when == 'setup':
            assert call.excinfo is not None and call.excinfo.type is OSError
            assert call.excinfo.value.args == (errno.ENOSPC, 'P directed setup evidence failure')
            # The injected writer fails once; classification uses the original.
            evidence('setup-failure-classified')
        else:
            assert call.excinfo is None
    @pytest.fixture(autouse=True)
    def fault_P(self, request):
        patch = pytest.MonkeyPatch()
        run = recovery._run_bounded_maintenance
        trace0 = sys.gettrace()
        write0, exit0 = os.write, os._exit
        cls = h.DurableLostResultWitness_P
        original_init, original_clear = cls.__init__, cls.clear_P
        original_seam, original_exit, original_trace = cls.at_seam_P, cls.at_exit_P, cls.trace_P
        original_continue = recovery._MaintenanceSession.continue_kernel
        original_locked = recovery._prepare_abandon_locked
        replacements = {}
        predicate = {'durable-id', 'bad-offset', 'bad-slice', 'frame-budget', 'wrong-operation-code'}
        cleanup = {'cleanup-value', 'cleanup-nonsemantic', 'cleanup-new-object', 'changed-route'}
        fd_modes = {'wrong-fd', 'reused-fd', 'witness-pipe', 'pipe-mode'}
        node0 = h._CURRENT.node
        atomic0 = h.atomic
        nodeid0 = node0.nodeid
        def restore_P():
            # Registered before ANY manual binding or patch mutation. Pytest
            # runs this even when setup fails before the generator's yield.
            patch.undo()
            assert h._CURRENT.node is node0 and node0.nodeid == nodeid0
            assert sys.gettrace() is trace0 and os.write is write0 and os._exit is exit0
            assert cls.__init__ is original_init and cls.at_seam_P is original_seam
            assert cls.at_exit_P is original_exit and cls.trace_P is original_trace and cls.clear_P is original_clear
            assert recovery._MaintenanceSession.continue_kernel is original_continue
            assert recovery._prepare_abandon_locked is original_locked
            assert recovery._run_bounded_maintenance is run
            assert not hasattr(h._CURRENT.node, 'obj')
            if mode != 'missing-done': assert h.atomic is atomic0
            evidence('hooks-restored')
        request.addfinalizer(restore_P)
        # Bootstrap has no collected callable. Bind THIS inner pytest item
        # before its invocation; never recover authority from the module.
        assert not hasattr(h._CURRENT.node, 'obj')
        assert request.node.nodeid == NODE
        collected = request.node.obj
        assert callable(collected)
        assert collected is sys.modules['test_recovery'].test_t1_durable_completion_lost_result_no_false_success
        if mode == 'missing-collected':
            assert not hasattr(h._CURRENT.node, 'obj')
            evidence('collected-withheld')
        else:
            patch.setattr(h._CURRENT.node, 'obj', collected, raising=False)
            assert h._CURRENT.node.obj is collected
        if mode == 'wrong-node':
            patch.setattr(h._CURRENT.node, 'nodeid', 'foreign::' + NODE)
            assert h._CURRENT.node.nodeid == 'foreign::' + NODE
            evidence('node-mutated')
        if mode == 'wrong-collected':
            h._CURRENT.node.obj = lambda: None
            assert h._CURRENT.node.obj.__code__ is not sys.modules['test_recovery'].test_t1_durable_completion_lost_result_no_false_success.__code__
            evidence('collected-mutated')
        def init(witness, *a, **kw):
            original_init(witness, *a, **kw)
            assert witness.eligible
            if mode == 'missing-collected':
                assert not hasattr(a[0], 'obj')
            invalid = mode in {'wrong-code', 'wrong-node', 'wrong-collected', 'missing-collected', 'operation-failure', 'operation-value'}
            assert witness.valid is not invalid
            evidence('armed-invalid' if invalid else 'armed-valid')
        patch.setattr(cls, '__init__', init)
        def trace(witness, frame, event, arg):
            if frame.f_code is witness.continuation and event == 'call':
                evidence('continuation-enter')
            answer = original_trace(witness, frame, event, arg)
            if frame.f_code is witness.continuation and event == 'return' and witness.completed is not None:
                assert witness.valid is (mode not in {'wrong-node', 'wrong-collected', 'missing-collected'})
                evidence('completed')
            if frame.f_code is witness.finish and event == 'return' and witness.completed is not None:
                assert witness.valid is (mode not in {'wrong-node', 'wrong-collected', 'missing-collected'})
                assert witness.sentinel_P.read_text() == witness.completed[2]
                evidence('finish')
            return trace.__get__(witness, cls) if answer is not None else None
        patch.setattr(cls, 'trace_P', trace)
        def probe(witness, exc, frame):
            saved = witness.exception, witness.route
            try: return original_seam(witness, exc, frame)
            finally: witness.exception, witness.route = saved
        def post_seam(exc):
            # This raise escapes the trace callback into the ORIGINAL runner's
            # catch-all. It occurs after the genuine result-write seam, after
            # session exit. It deliberately invalidates the route even when the
            # same object is reraised; no arbitrary-route acceptance is added.
            try: raise exc
            finally:
                if mode != 'changed-route': raise replacements['final']
        def seam(witness, exc, frame):
            if frame.f_locals.get('pid') != 0:
                assert not original_seam(witness, exc, frame)
                return False  # Parent refusal is not the child's write seam.
            assert frame is witness.runner_frame
            if mode in {'wrong-code', 'operation-failure', 'operation-value', 'precompletion'}:
                assert witness.completed is None and witness.exception is None
                assert type(exc) is (ValueError if mode in {'wrong-code', 'operation-value'} else BrokenPipeError)
                assert not original_seam(witness, exc, frame)
                evidence('early-seam-rejected')
                return False
            if mode in predicate or mode in cleanup:
                assert probe(witness, exc, frame)
                evidence('baseline-seam')
            if mode == 'durable-id':
                witness.completed = (*witness.completed[:2], '0'*32)
                assert witness.completed[2] != witness.completed[1]['intent']['id']
            if mode == 'bad-offset':
                frame.f_locals['offset'] = len(frame.f_locals['raw'])
                assert frame.f_locals['offset'] == len(frame.f_locals['raw'])
            if mode == 'frame-budget':
                frame.f_locals['raw'] = bytearray(1048577)
                assert len(frame.f_locals['raw']) == 1048577
            if mode == 'bad-slice':
                inner = exc.__traceback__.tb_next.tb_frame
                inner.f_locals['a'] = (frame.f_locals['write_fd'], memoryview(b'wrong slice'))
                assert bytes(inner.f_locals['a'][1]) == b'wrong slice'
                assert inner.f_locals['a'][1].obj is not frame.f_locals['raw']
            if mode == 'wrong-operation-code':
                witness.prepare_code = seam.__code__
                assert witness.prepare_code is not frame.f_locals['operation'].__code__
            if mode in predicate: evidence('predicate-mutated')
            result = original_seam(witness, exc, frame)
            if mode in cleanup:
                assert result and witness.exception is exc
                witness.emit_P(witness.seam)
                evidence('seam')
                replacements['original'] = exc
                replacements['final'] = (ValueError('P cleanup') if mode == 'cleanup-value' else
                    RuntimeError('P cleanup') if mode == 'cleanup-nonsemantic' else
                    BrokenPipeError(*exc.args) if mode == 'cleanup-new-object' else exc)
                assert (replacements['final'] is exc) == (mode == 'changed-route')
                evidence('unwind-mutated')
                post_seam(exc)
            rejected = mode in predicate | fd_modes | {
                'wrong-class', 'subclass', 'wrong-args', 'real-EPIPE', 'missing-sentinel',
                'wrong-node', 'wrong-collected', 'missing-collected'}
            if rejected:
                assert not result
                evidence('seam-rejected')
            else:
                assert result
                evidence('seam')
            return result
        patch.setattr(cls, 'at_seam_P', seam)
        def final(witness, exc, caller):
            # observed_exit reads sys.exception itself; this audit only verifies
            # that real active object/caller, except the labelled predicate probe.
            assert sys.exception() is exc and caller is witness.runner_frame
            if mode in cleanup:
                assert exc is replacements['final']
                assert (exc is witness.exception) == (mode == 'changed-route')
                if mode == 'changed-route':
                    route, tb = [], exc.__traceback__
                    while tb is not None:
                        route.append((tb.tb_frame, tb.tb_lasti)); tb = tb.tb_next
                    assert tuple(route) != witness.route
                evidence('final-identity')
            if mode == 'changed-attributes':
                assert exc is witness.exception and exc.args == ('modeled lost result pipe',)
                exc.args = ('changed',)
                assert exc.args == ('changed',)
                evidence('attributes-mutated')
            if mode == 'wrong-final-caller':
                assert original_exit(witness, exc, caller) == witness.final
                caller = sys._getframe()  # Predicate-only; changed-route is real.
                evidence('caller-predicate-mutated')
            answer = original_exit(witness, exc, caller)
            accepted = mode in {'normal', 'missing-markers', 'signal11', 'foreign-owner', 'missing-done'}
            assert answer == (witness.final if accepted else b'')
            evidence('final-accepted' if accepted else 'final-rejected')
            return answer
        patch.setattr(cls, 'at_exit_P', final)
        def clear(witness):
            original_clear(witness)
            assert witness.exception is witness.route is witness.completed is witness.runner_frame is witness.sentinel_P is None
            replacements.clear()
            evidence('references-cleared')
        patch.setattr(cls, 'clear_P', clear)
        def changed_run(roots, scope, operation, **kw):
            if mode == 'missing-collected':
                assert not hasattr(h._CURRENT.node, 'obj')
            installed = recovery._prepare_abandon_locked
            if mode in {'wrong-class', 'subclass', 'wrong-args', 'real-EPIPE', 'missing-sentinel'} | fd_modes:
                class SubPipe(BrokenPipeError): pass
                def error(*args):
                    broken_frame = sys._getframe(1)
                    runner = broken_frame.f_back
                    witness = sys.gettrace().__self__
                    assert runner is witness.runner_frame and witness.valid
                    assert witness.completed is not None and witness.sentinel_P.read_text() == witness.completed[2]
                    fd = runner.f_locals['write_fd']
                    old = witness.pipe_identity(fd)
                    witness_identity = witness.pipe_identity(witness.witness_fd)
                    assert old == witness.result_identity and old != witness_identity
                    assert runner.f_locals['identities'][fd] == old
                    write_args = broken_frame.f_locals['a']
                    assert write_args[0] == fd and write_args[1].obj is runner.f_locals['raw']
                    assert bytes(write_args[1]) == runner.f_locals['raw'][runner.f_locals['offset']:runner.f_locals['offset']+65536]
                    evidence('error-factory-reached')
                    if mode == 'missing-sentinel':
                        witness.sentinel_P.unlink()
                        assert not witness.sentinel_P.exists()
                    if mode in fd_modes:
                        if mode == 'wrong-fd': other = os.open('/dev/null', os.O_WRONLY)
                        elif mode == 'witness-pipe': other = os.dup(witness.witness_fd)
                        else:
                            read, write = os.pipe()
                            other = read if mode == 'pipe-mode' else write
                            os.close(write if mode == 'pipe-mode' else read)
                        try:
                            replacement = witness.pipe_identity(other)
                            assert replacement != old
                            assert replacement[3] == (os.O_RDONLY if mode == 'pipe-mode' else os.O_WRONLY)
                            assert stat.S_ISCHR(replacement[2]) if mode == 'wrong-fd' else stat.S_ISFIFO(replacement[2])
                            assert (replacement == witness_identity) == (mode == 'witness-pipe')
                            os.dup2(other, fd)
                            assert witness.pipe_identity(fd) == replacement
                            assert (replacement[:2] != old[:2])
                            assert witness.result_identity == old
                            assert witness.pipe_identity(witness.witness_fd) == witness_identity
                        finally: os.close(other)
                        with pytest.raises(OSError) as closed: os.fstat(other)
                        assert closed.value.errno == errno.EBADF
                        evidence('fd-mutated-and-temp-closed')
                    replacement = (ValueError(*args) if mode == 'wrong-class' else
                        SubPipe(*args) if mode == 'subclass' else
                        BrokenPipeError('different') if mode == 'wrong-args' else
                        BrokenPipeError(errno.EPIPE, 'modeled lost result pipe') if mode == 'real-EPIPE' else
                        BrokenPipeError(*args))
                    if mode == 'wrong-class': assert type(replacement) is ValueError
                    if mode == 'subclass': assert type(replacement) is SubPipe
                    if mode == 'wrong-args': assert replacement.args == ('different',)
                    if mode == 'real-EPIPE': assert replacement.errno == errno.EPIPE
                    evidence('error-mutated')
                    return replacement
                changed = types.FunctionType(installed.__code__,
                    {**installed.__globals__, 'BrokenPipeError': error},
                    installed.__name__, installed.__defaults__, installed.__closure__)
                assert changed.__code__ is installed.__code__
                patch.setattr(recovery, '_prepare_abandon_locked', changed)
            if mode == 'wrong-code':
                def wrong(*a, **k):
                    evidence('installed-operation-reached')
                    raise ValueError('wrong installed operation')
                patch.setattr(recovery, '_prepare_abandon_locked', wrong)
            if mode == 'precompletion':
                def early(session, intent, *, prepared):
                    assert prepared._session is session and not prepared._used
                    evidence('continuation-precompletion-reached')
                    raise BrokenPipeError('modeled lost result pipe')
                patch.setattr(recovery._MaintenanceSession, 'continue_kernel', early)
            if mode in {'operation-failure', 'operation-value'}:
                def outer(value):
                    assert value._live
                    evidence('outer-operation-reached')
                    if mode == 'operation-value': raise ValueError('malformed actual operation')
                    raise BrokenPipeError('modeled lost result pipe')
                assert outer.__code__ is not operation.__code__
                operation = outer  # Actual callable, before parent arming.
            return run(roots, scope, operation, **kw)
        patch.setattr(recovery, '_run_bounded_maintenance', changed_run)
        if mode == 'setup-evidence-failure':
            assert h._CURRENT.node.obj is collected
            assert cls.__init__ is init and cls.trace_P is trace
            assert cls.at_seam_P is seam and cls.at_exit_P is final and cls.clear_P is clear
            assert recovery._run_bounded_maintenance is changed_run
            def failed_evidence(path, value):
                assert path == Path(PROOF)/(str(os.getpid())+'.json')
                assert value == dict(mode=mode, steps=['setup-evidence-attempt'])
                # Leave later report/finalizer evidence writable, but keep this
                # patch owned so undo must still restore the exact original.
                patch.setattr(h, 'atomic', atomic0)
                raise OSError(errno.ENOSPC, 'P directed setup evidence failure')
            patch.setattr(h, 'atomic', failed_evidence)
            evidence('setup-evidence-attempt')
            raise AssertionError('setup evidence failure did not escape')
        if mode in ('missing-markers', 'signal11'):
            observe = h._CURRENT.observe
            def changed(result, **kw):
                assert result[1] == 74 << 8 and kw['witness'] == b'P:a;P:s;P:f;' and kw['armed_P']
                if mode == 'missing-markers': kw['witness'] = b'V'
                if mode == 'signal11': result = (result[0], 11)  # Synthetic observer only.
                evidence('observer-mutated')
                return observe(result, **kw)
            patch.setattr(h._CURRENT, 'observe', changed)
        if mode == 'foreign-owner':
            event = h._CURRENT.event
            def foreign(kind, **data):
                if kind in ('enroll', 'terminal'):
                    h._CURRENT.sequence += 1
                    h.atomic(h._CURRENT.directory/f'{os.getpid()}-{h._CURRENT.sequence}.json',
                        dict(kind=kind, owner=os.getpid()+100000, **data))
                else: event(kind, **data)
            patch.setattr(h._CURRENT, 'event', foreign)
        if mode == 'missing-done':
            atomic = h.atomic
            def missing(path, value):
                if path.name != 'done.json': return atomic(path, value)
            h.atomic = missing  # Through bootstrap atexit; local helper only.
        yield
result = pytest.main(['--noconftest', '-p', 'no:cacheprovider', NODE,
    '-q', '-rs', '-x', '--basetemp='+BASE], plugins=[Plugin()])
if mode == 'setup-evidence-failure':
    assert result == pytest.ExitCode.TESTS_FAILED
    assert setup_reports == ['setup', 'teardown']
    assert steps == ['setup-evidence-attempt', 'setup-failure-classified', 'hooks-restored']
    assert not h._CURRENT.receipts and not h._CURRENT.owned and not h._CURRENT.failures
    assert not h._CURRENT.session.shouldstop and not hasattr(h._CURRENT.node, 'obj')
    raise SystemExit(87)
if h._CURRENT.failures:
    assert h._CURRENT.session.shouldstop
    with pytest.raises(pytest.exit.Exception):
        recovery._run_bounded_maintenance({}, {}, lambda value: True)
    evidence('next-maintenance-refused')
    raise SystemExit(86)
assert result == 0
Path(SENTINEL).write_text('continued')
'''.replace('MODE', repr(mode)).replace('NODE', repr(nodeid)).replace('BASE', repr(str(tmp_path/'inner-P'))).replace('SENTINEL', repr(str(sentinel))).replace('PROOF', repr(str(proof)))
    aggregation = mode in ('foreign-owner', 'missing-done')
    with harness.activate(guard, tmp_path/'helpers-P'):
        result = harness.fresh_run([sys.executable, '-c', body], capture_output=True, text=True, timeout=30)
        if mode == 'setup-evidence-failure':
            assert result.returncode == 87, result.stdout + result.stderr
            assert not sentinel.exists() and not guard.receipts and not guard.owned and not guard.failures
            assert not guard.session.shouldstop
        elif mode == 'normal':
            assert result.returncode == 0, result.stdout + result.stderr
            assert sentinel.exists() and not guard.failures
            # _kfifth_fresh launches TWO genuine checked observational replays.
            assert [status for _, status in guard.receipts] == [74 << 8, 0, 0]
        else:
            assert result.returncode == (0 if aggregation else 86), result.stdout + result.stderr
            assert guard.failures and guard.session.shouldstop, result.stdout + result.stderr
            with pytest.raises(RuntimeError):
                harness.fresh_run([sys.executable, '-c', 'raise AssertionError("must not launch")'])
            assert sentinel.exists() is aggregation
    directory, = (tmp_path/'helpers-P').glob('helper-*')
    events = [json.loads(p.read_text()) for p in directory.glob('[0-9]*.json')]
    terminals = [e for e in events if e['kind'] == 'terminal']
    if mode == 'setup-evidence-failure':
        assert events == [] and terminals == []
        ack = json.loads((directory/'ack.json').read_text())
        assert ack == json.loads((directory/'done.json').read_text())
        assert json.loads((directory/'launch.json').read_text())['identity'] == ack
        assert not list((directory/'nested').glob('helper-*'))
        records = [json.loads(p.read_text()) for p in proof.glob('*.json')]
        assert records == [dict(mode=mode, steps=[
            'setup-evidence-attempt', 'setup-failure-classified', 'hooks-restored'])]
        return  # Setup-only restoration control, never a P refusal receipt.
    assert len(terminals) == 1
    terminal, = terminals
    positive = mode == 'normal' or aggregation
    phase = ('P:a;P:s;P:f;' if positive or mode == 'signal11' else
        'V' if mode == 'missing-markers' else
        'P:a;V:operation;V:session;' if mode in ('wrong-code', 'operation-value') else
        'P:a;P:s;' if mode.startswith('cleanup-') or mode in ('changed-route', 'changed-attributes', 'wrong-final-caller') else 'P:a;')
    assert terminal['phase'] == phase
    assert terminal['status'] == (11 if mode == 'signal11' else 74 << 8)
    assert terminal['accepted'] is positive
    if mode == 'normal':
        nested = list((directory/'nested').glob('helper-*'))
        assert len(nested) == 2
        for child in nested:
            rows = [json.loads(p.read_text()) for p in child.glob('[0-9]*.json')]
            wait, = [e for e in rows if e['kind'] == 'terminal']
            assert wait['phase'] == '' and wait['status'] == 0 and wait['accepted']
            assert json.loads((child/'ack.json').read_text()) == json.loads((child/'done.json').read_text())
    if mode.startswith('cleanup-') or mode == 'changed-route':
        assert 'P:s;' in terminals[0]['phase'] and 'P:f;' not in terminals[0]['phase']
        assert not terminals[0]['accepted']
    if not positive:
        failure, = [e for e in events if e['kind'] == 'failure']
        assert failure['message'].endswith('incomplete or invalid armed P contract')
    if mode == 'foreign-owner':
        ack = json.loads((directory/'ack.json').read_text())
        pair = [e for e in events if e['kind'] in ('enroll', 'terminal')]
        assert len(pair) == 2 and {e['owner'] for e in pair} == {ack['pid']+100000}
        assert {e['pid'] for e in pair} == {terminal['pid']}
        assert (directory/'done.json').exists()
        assert len(guard.failures) == 1 and guard.failures[0].endswith('fresh helper receipt failure: foreign helper event owner')
    if mode == 'missing-done':
        # Demand the precise unchanged aggregator failure, not an arbitrary RED.
        assert not (directory/'done.json').exists()
        with pytest.raises(FileNotFoundError) as absent:
            harness.aggregate(directory, json.loads((directory/'ack.json').read_text()))
        assert guard.failures == [nodeid + ': fresh helper receipt failure: ' + str(absent.value)[:1024]]
    records = [json.loads(p.read_text()) for p in proof.glob('*.json')]
    assert len(records) == 2 and all(r['mode'] == mode for r in records)
    parent, = [r['steps'] for r in records if 'hooks-restored' in r['steps']]
    child, = [r['steps'] for r in records if 'final-accepted' in r['steps'] or 'final-rejected' in r['steps']]
    invalid = mode in ('wrong-code', 'wrong-node', 'wrong-collected', 'missing-collected', 'operation-failure', 'operation-value')
    assert parent == (['node-mutated'] if mode == 'wrong-node' else ['collected-mutated'] if mode == 'wrong-collected' else ['collected-withheld'] if mode == 'missing-collected' else []) + [('armed-invalid' if invalid else 'armed-valid')] + (['observer-mutated'] if mode in ('missing-markers', 'signal11') else []) + ['references-cleared', 'hooks-restored'] + ([] if positive else ['next-maintenance-refused'])
    early = {'wrong-code': 'installed-operation-reached', 'operation-value': 'outer-operation-reached',
        'operation-failure': 'outer-operation-reached', 'precompletion': 'continuation-precompletion-reached'}
    if mode in early:
        assert child == [early[mode], 'early-seam-rejected', 'final-rejected']
    else:
        expected = ['continuation-enter', 'completed', 'finish']
        fd_modes = {'wrong-fd', 'reused-fd', 'witness-pipe', 'pipe-mode'}
        error_modes = fd_modes | {'wrong-class', 'subclass', 'wrong-args', 'real-EPIPE', 'missing-sentinel'}
        predicates = {'durable-id', 'bad-offset', 'bad-slice', 'frame-budget', 'wrong-operation-code'}
        cleanup = mode.startswith('cleanup-') or mode == 'changed-route'
        if mode in error_modes:
            expected += ['error-factory-reached'] + (['fd-mutated-and-temp-closed'] if mode in fd_modes else []) + ['error-mutated']
        if mode in predicates or cleanup: expected += ['baseline-seam']
        if mode in predicates: expected += ['predicate-mutated']
        expected += ['seam-rejected' if mode in error_modes | predicates | {'wrong-node', 'wrong-collected', 'missing-collected'} else 'seam']
        if cleanup: expected += ['unwind-mutated', 'final-identity']
        if mode == 'changed-attributes': expected += ['attributes-mutated']
        if mode == 'wrong-final-caller': expected += ['caller-predicate-mutated']
        expected += ['final-accepted' if positive or mode in ('missing-markers', 'signal11') else 'final-rejected']
        assert child == expected
