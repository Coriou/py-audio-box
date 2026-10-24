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
    'missing-markers', 'signal11', 'foreign-owner', 'missing-done', 'operation-value', 'setup-evidence-failure', 'cleanup-ordering'])
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
setup_only = mode in {'setup-evidence-failure', 'cleanup-ordering'}
class Plugin:
    def pytest_runtest_makereport(self, item, call):
        if not setup_only: return
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
        # This slot owns only our exact installed wrapper. Never put the
        # target's finish closure into the manual MonkeyPatch undo stack:
        # its separate fixture can already have restored production by then.
        locked_owner = [None]
        def install_locked(value):
            assert locked_owner[0] is None
            locked_owner[0] = value
            recovery._prepare_abandon_locked = value
        def restore_locked():
            owned = locked_owner[0]
            if owned is not None and recovery._prepare_abandon_locked is owned:
                recovery._prepare_abandon_locked = original_locked
            locked_owner[0] = None
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
            restore_locked()
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
                install_locked(changed)
            if mode == 'wrong-code':
                def wrong(*a, **k):
                    evidence('installed-operation-reached')
                    raise ValueError('wrong installed operation')
                install_locked(wrong)
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
        if mode == 'cleanup-ordering':
            # Use the SAME pytest fixture the target requests, and its real
            # undo(), before our registered finalizer. No maintenance runs.
            target_patch = request.getfixturevalue('monkeypatch')
            def target_finish(*a, **kw): raise AssertionError('control must not run')
            def owned_wrapper(*a, **kw): raise AssertionError('control must not run')
            def foreign_wrapper(*a, **kw): raise AssertionError('control must not run')
            assert recovery._prepare_abandon_locked is original_locked
            target_patch.setattr(recovery, '_prepare_abandon_locked', target_finish)
            install_locked(owned_wrapper)
            assert recovery._prepare_abandon_locked is owned_wrapper
            restore_locked()
            assert recovery._prepare_abandon_locked is original_locked
            evidence('owned-wrapper-restored')
            target_patch.undo()
            assert recovery._prepare_abandon_locked is original_locked
            target_patch.setattr(recovery, '_prepare_abandon_locked', target_finish)
            install_locked(owned_wrapper)
            recovery._prepare_abandon_locked = foreign_wrapper
            restore_locked()
            assert recovery._prepare_abandon_locked is foreign_wrapper
            evidence('foreign-owner-preserved')
            target_patch.undo()
            assert recovery._prepare_abandon_locked is original_locked
            target_patch.setattr(recovery, '_prepare_abandon_locked', target_finish)
            install_locked(owned_wrapper)
            assert recovery._prepare_abandon_locked is owned_wrapper
            evidence('target-wrapper-installed')
            target_patch.undo()
            assert recovery._prepare_abandon_locked is original_locked
            assert locked_owner[0] is owned_wrapper
            evidence('target-fixture-undone')
            # Leave the ownership slot populated: the plugin finalizer must
            # see production already restored and must not resurrect target_finish.
        if setup_only:
            assert h._CURRENT.node.obj is collected
            assert cls.__init__ is init and cls.trace_P is trace
            assert cls.at_seam_P is seam and cls.at_exit_P is final and cls.clear_P is clear
            assert recovery._run_bounded_maintenance is changed_run
            def failed_evidence(path, value):
                assert path == Path(PROOF)/(str(os.getpid())+'.json')
                if mode == 'cleanup-ordering':
                    assert value == dict(mode=mode, steps=['owned-wrapper-restored',
                        'foreign-owner-preserved', 'target-wrapper-installed', 'target-fixture-undone',
                        'setup-evidence-attempt'])
                else:
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
if setup_only:
    assert result == pytest.ExitCode.TESTS_FAILED
    assert setup_reports == ['setup', 'teardown']
    if mode == 'cleanup-ordering':
        assert steps == ['owned-wrapper-restored', 'foreign-owner-preserved',
            'target-wrapper-installed', 'target-fixture-undone',
            'setup-evidence-attempt', 'setup-failure-classified', 'hooks-restored']
    else:
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
        if mode == 'setup-evidence-failure' or mode == 'cleanup-ordering':
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
    if mode == 'setup-evidence-failure' or mode == 'cleanup-ordering':
        assert events == [] and terminals == []
        ack = json.loads((directory/'ack.json').read_text())
        assert ack == json.loads((directory/'done.json').read_text())
        assert json.loads((directory/'launch.json').read_text())['identity'] == ack
        assert not list((directory/'nested').glob('helper-*'))
        records = [json.loads(p.read_text()) for p in proof.glob('*.json')]
        if mode == 'cleanup-ordering':
            assert records == [dict(mode=mode, steps=['owned-wrapper-restored',
                'foreign-owner-preserved', 'target-wrapper-installed', 'target-fixture-undone',
                'setup-evidence-attempt', 'setup-failure-classified', 'hooks-restored'])]
        else:
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


# B r44 directed controls. APPEND ONLY. Run under --noconftest in a separately
# authorized control lane: the frozen singleton Observer rejects other nodes.
# No test below is executed by authoring/compiling this file.
B_CONTROL_NODE = 'tests/service/test_recovery.py::test_t1_real_blocked_pipe_backpressure_deadline'
B_CONTROL_PARSER = (
    '74', '9-no-f', '9-f', 'missing-a', 'missing-e', 'missing-f',
    'missing-t', 'missing-k', 'duplicate', 'partial', 'order', 'overflow',
    'legacy', 'exit0', 'arbitrary74', 'signal11', 'overlap-P',
)
B_CONTROL_SEAMS = (
    'original74-real-observer', 'collected-control', 'missing-obj', 'wrong-obj',
    'wrong-node', 'parameter', 'wrong-code', 'closure', 'operation',
    'wrong-errno', 'subclass', 'missing-eagain', 'fd-other', 'fd-reuse',
    'fd-witness-alias', 'fd-read-access', 'raw-budget', 'offset', 'view-owner',
    'view-slice', 'premature', 'route561', 'raise-offset', 'call-offset',
    'absent-route', 'extra-route', 'reordered-route', 'wrong-args',
    'replacement', 'same-object-extra-frame', 'trace-disabled', 'writer-failure',
    'direct-exit74', 'actual-overflow', 'schedule9-no-f', 'schedule9-f',
    'previous-trace', 'setup-failure', 'fixture-undo',
)


def _B_refused_launches(guard, tmp_path):
    """Demand the existing two latch consumers; never launch after refusal."""
    # The guard's real installed wrapper is supplied by the isolated seam
    # process below. Parser-only controls use a source projection of its exact
    # preflight because installing it requires an observer/production import.
    import ast
    source = Path(harness.__file__).read_text()
    tree = ast.parse(source)
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    run = next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == 'guarded_run')
    preflight = run.body[0]
    assert ast.unparse(preflight) == 'if guard.failures:\n    pytest.exit(guard.failures[0], returncode=2)'
    # Exact source, no rewritten predicate and no call into production.
    with pytest.raises(pytest.exit.Exception) as stopped:
        exec(compile(ast.Module(body=[preflight], type_ignores=[]), harness.__file__, 'exec'),
             {'guard': guard, 'pytest': pytest})
    assert str(stopped.value) == guard.failures[0]
    with harness.activate(guard, tmp_path/'refused-helpers'):
        with pytest.raises(RuntimeError) as helper:
            harness.fresh_run([sys.executable, '-c', 'raise AssertionError("must not launch")'])
        assert str(helper.value) == guard.failures[0]
    assert list((tmp_path/'refused-helpers').iterdir()) == []


@pytest.mark.parametrize('control', B_CONTROL_PARSER, ids=lambda c: 'B-T-'+c)
def test_B_closed_transcript_source_predicate(tmp_path, control):
    """Synthetic raw statuses/parser inputs; no child/kill/timeout certificate."""
    cls = harness.BlockedPipeWitness_B
    baseline = b'B:a;B:e;B:f;B:t;'
    assert cls.transcript(baseline, 74 << 8)
    assert cls.transcript(b'B:a;B:e;B:t;B:k;', 9)
    assert cls.transcript(baseline+b'B:k;', 9)
    witness, status = baseline, 74 << 8
    if control == '9-no-f': witness, status = b'B:a;B:e;B:t;B:k;', 9
    elif control == '9-f': witness, status = baseline+b'B:k;', 9
    elif control.startswith('missing-'):
        token = ('B:'+control[-1]+';').encode()
        witness = (baseline+b'B:k;' if control == 'missing-k' else baseline).replace(token, b'')
        if control == 'missing-k': status = 9
        assert token not in witness
    elif control == 'duplicate': witness += b'B:f;'
    elif control == 'partial': witness = baseline[:-1]
    elif control == 'order': witness = b'B:a;B:f;B:e;B:t;'
    elif control == 'overflow': witness = baseline+b'x'*257
    elif control == 'legacy': witness = b'V:session;V:operation;'
    elif control == 'exit0': status = 0
    elif control == 'arbitrary74': witness = b'B:a;'
    elif control == 'signal11': status = 11
    positive = control in ('74', '9-no-f', '9-f')
    assert cls.transcript(witness, status) is (positive or control == 'overlap-P')
    guard = harness.MaintenanceWaitGuard(SimpleNamespace(nodeid='B-source-parser',
        originalname='B-source-parser'), SimpleNamespace(shouldstop=False))
    guard.owned.add(123)
    raw = (123, status)
    assert guard.observe(raw, witness=witness, armed_B=True, armed_P=control == 'overlap-P',
        expected_exit=74, expected_signal=status) is raw
    assert guard.receipts == [raw] and not guard.owned
    if positive:
        assert guard.failures == [] and guard.session.shouldstop is False
    else:
        assert len(guard.failures) == 1
        assert guard.failures[0].endswith('incomplete or invalid armed B contract')
        assert guard.session.shouldstop == guard.failures[0]
        _B_refused_launches(guard, tmp_path)


# This program runs only when a separately authorized test invokes it. The
# control process owns every substitution. The guard source is imported intact.
_B_SEAM_PROGRAM = r'''
import errno, os, sys, types, stat, pytest
from pathlib import Path
import _maintenance_wait_guard as h
from service import recovery
mode, root = CONTROL, Path(ROOT)
root.mkdir(mode=0o700)
steps = []

# Schema and storage are prepared before target clock/trace installation.
from types import MappingProxyType
BASELINE_IDENTITIES = MappingProxyType({
    'original74-real-observer': 'ORIGINAL74_REAL_OBSERVER',
    'collected-control': 'COLLECTED_CONTROL'})
assert mode in ('original74-real-observer', 'collected-control', 'missing-obj', 'wrong-obj', 'wrong-node', 'parameter', 'wrong-code', 'closure', 'operation', 'wrong-errno', 'subclass', 'missing-eagain', 'fd-other', 'fd-reuse', 'fd-witness-alias', 'fd-read-access', 'raw-budget', 'offset', 'view-owner', 'view-slice', 'premature', 'route561', 'raise-offset', 'call-offset', 'absent-route', 'extra-route', 'reordered-route', 'wrong-args', 'replacement', 'same-object-extra-frame', 'trace-disabled', 'writer-failure', 'direct-exit74', 'actual-overflow', 'schedule9-no-f', 'schedule9-f', 'previous-trace', 'setup-failure', 'fixture-undo')
expected_control = BASELINE_IDENTITIES.get(mode)
diagnostic_enabled = expected_control is not None
NOT_REACHED = 'NOT_REACHED'
QUALIFICATION_KEYS = ('terminal', 'observe_status', 'observe_witness', 'observe_return', 'receipt_owned')
def qualification_record():
    return dict(policy='BASELINE_AFTER_OBSERVE', **dict.fromkeys(QUALIFICATION_KEYS, NOT_REACHED),
        observe_calls=0, assertions='PENDING', first_failed='NONE')
def validate_identity(expected):
    if type(expected) is not str or expected not in BASELINE_IDENTITIES.values():
        raise ValueError('external baseline identity')
def validate_qualification(record):
    if type(record) is not dict or record.keys() != qualification_record().keys():
        raise ValueError('qualification schema')
    if record['policy'] != 'BASELINE_AFTER_OBSERVE': raise ValueError('qualification policy')
    for key in QUALIFICATION_KEYS:
        value = record[key]
        if not (type(value) is bool or type(value) is str and value == NOT_REACHED):
            raise ValueError('qualification operand')
    if type(record['observe_calls']) is not int or record['observe_calls'] not in (0,1):
        raise ValueError('qualification call count')
    if record['assertions'] not in ('PENDING','PASSED','FAILED','NOT_REACHED') or record['first_failed'] not in (*QUALIFICATION_KEYS,'NONE'):
        raise ValueError('qualification assertion enum')
    failed = record['first_failed']
    if record['assertions'] == 'PASSED':
        if failed != 'NONE' or record['observe_calls'] != 1 or any(record[k] is not True for k in QUALIFICATION_KEYS):
            raise ValueError('qualification false PASS')
    elif record['assertions'] == 'FAILED':
        if failed == 'NONE' or record[failed] is True or any(record[k] is not True for k in QUALIFICATION_KEYS[:QUALIFICATION_KEYS.index(failed)]):
            raise ValueError('qualification first failure')
    elif failed != 'NONE': raise ValueError('qualification premature failure')
def assert_qualification(record):
    # Saved primitives only: no late witness/receipt/state consultation.
    for key in QUALIFICATION_KEYS:
        if record[key] is not True:
            record['assertions'], record['first_failed'] = 'FAILED', key
            raise AssertionError('baseline qualification: '+key)
    record['assertions'] = 'PASSED'
def callback_record(role, expected):
    validate_identity(expected)
    return dict(schema=2, control=expected, role=role,
        first_site=0, kind='UNREACHED', gate=0, hook_gate=0,
        valid=NOT_REACHED, phase=NOT_REACHED, final_called=NOT_REACHED,
        final_result=NOT_REACHED, terminal_called=NOT_REACHED,
        terminal_result=NOT_REACHED, status=NOT_REACHED, child_bytes=NOT_REACHED,
        suffix=NOT_REACHED, raw18944=NOT_REACHED, exact_ef=NOT_REACHED,
        exact_t=NOT_REACHED, observe_entered=NOT_REACHED,
        observe_completed=NOT_REACHED, observe_status=NOT_REACHED,
        observe_armed=NOT_REACHED, observe_witness=NOT_REACHED,
        observe_return=NOT_REACHED, assertion=NOT_REACHED, qualification=qualification_record())

CALLBACK_ENUMS = {
    'schema': (2,), 'control': tuple(BASELINE_IDENTITIES.values()), 'role': ('CHILD','PARENT'),
    'first_site': tuple(range(14)), 'gate': tuple(range(16)), 'hook_gate': tuple(range(25)),
    'kind': ('FALSE','CAUGHT_EXCEPTION','WRAPPER_ASSERT','UNREACHED'),
    'phase': (NOT_REACHED,'ARMED','BACKPRESSURE','TIMEOUT','OTHER'),
    'terminal_result': (NOT_REACHED,'EMPTY','T','TK','OTHER'),
    'child_bytes': (NOT_REACHED,'EMPTY','E','EF','OTHER'),
    'suffix': (NOT_REACHED,'EMPTY','T','TK','OTHER'),
    'assertion': (NOT_REACHED,'FINAL_PREREQUISITE','TERMINAL_PREREQUISITE',
        'OBSERVE_STATUS','OBSERVE_WITNESS','OBSERVE_RETURN')}
CALLBACK_KEYS = frozenset(callback_record('CHILD', next(iter(BASELINE_IDENTITIES.values()))))
def validate_callback(record, expected):
    validate_identity(expected)
    if record.get("control") != expected: raise ValueError("callback identity binding")
    if record.keys() != CALLBACK_KEYS: raise ValueError('diagnostic schema')
    for key, value in record.items():
        if key == 'qualification':
            validate_qualification(value)
            continue
        if key == 'status':
            valid = type(value) is int and 0 <= value <= 65535 or type(value) is str and value in (NOT_REACHED,'OTHER')
        elif key in CALLBACK_ENUMS:
            valid = any(type(value) is type(allowed) and value == allowed for allowed in CALLBACK_ENUMS[key])
        else: valid = type(value) is bool or type(value) is str and value in (NOT_REACHED,'OTHER')
        if not valid: raise ValueError('diagnostic enum')

# Exact v2 child-survival leaf; the historical filename does not imply refusal.
MILESTONES = ('real-eagain-baseline', 'eagain-accepted', 'actual-child558-timeout',
    'final-baseline', 'final-accepted')
def decode_survival(raw, delivered, expected):
    validate_identity(expected)
    record = dict(schema=2, control=expected, role='CHILD',
        wait='DELIVERED' if delivered else 'WAIT_UNOBSERVED',
        raw=raw.hex() if raw is not None else None, state='UNKNOWN',
        result_complete=False, first_site=0, final_gate=0, hook_gate=0,
        hook_stage=0, milestones=[], error=0)
    if not delivered:
        if raw is not None: raise ValueError('read without original wait')
        return record
    if type(raw) is not bytes or len(raw) != 64: raise ValueError('survival length')
    if raw[:4] != b'BDS\x02' or any(raw[36:]): return record
    def pair(offset, maximum):
        v, complement = raw[offset:offset+2]
        if (v, complement) == (0, 0): return 0
        return v if 1 <= v <= maximum and complement == 255-v else -1
    role, entry, site = pair(4,1), pair(6,1), pair(8,13)
    final, hook, stage, error = pair(10,15), pair(12,24), pair(14,5), pair(34,1)
    record.update(first_site=max(site,0), final_gate=max(final,0),
        hook_gate=max(hook,0), hook_stage=max(stage,0), error=max(error,0))
    # Retain only the exact committed prefix, even when a later slot tears.
    milestone_bad = False
    ended = False
    for index in range(5):
        value = pair(24+2*index,5)
        if value == index+1 and not ended: record['milestones'].append(MILESTONES[index])
        else:
            if value != 0: milestone_bad = True
            ended = True
    prefix = len(record['milestones'])
    if role != 1 or site < 0 or error != 0: return record
    if entry == 0 and (any(raw[10:24]) or prefix > 3): return record
    if entry == 0:
        if not milestone_bad: record['state'] = 'NOT_ENTERED_RECORDED'
        return record
    if entry < 0:
        record['state'] = 'INCOMPLETE'
        return record
    # An entered final follows the three unchanged prerequisite milestones.
    if prefix < 3: return record
    if final < 0 or hook < 0 or stage < 0 or milestone_bad:
        record['state'] = 'INCOMPLETE' if raw[22:24] != b'\xa5\x5a' else 'UNKNOWN'
        return record
    if (final < 2 and (hook or stage) or hook and stage == 0 or
            stage == 1 and hook != 0 or stage == 3 and hook != 24 or
            stage == 4 and not hook or final > 2 and stage != 3): return record
    commit = raw[22:24]
    if commit != b'\xa5\x5a':
        # No payload prefix, including A5 alone, is a completed result.
        if prefix > 3: return record
        record['state'] = 'INCOMPLETE'
        return record
    kind, result, gate, detail, valid, phase = raw[16:22]
    if valid not in (1,2) or phase not in (1,2,3): return record
    if kind == 3:
        legal = (result == 2 and gate == detail == 0 and final == 15 and
            stage == 3 and hook == 24 and valid == 2 and phase == 3 and site == 0)
        state = 'TRUE'
    elif kind in (1,2):
        legal = (result == 1 and 0 <= gate <= 15 and gate == final and
            (kind != 1 or gate > 0) and (kind != 2 or valid == 1) and
            (gate == 2 and detail == (0 if stage == 3 else hook) and stage in (3,4,5) or
             gate != 2 and detail == 0) and prefix <= 3 and
            (gate <= 4 or phase == 3))
        if kind == 1:
            legal = legal and (gate != 1 or valid == 1) and (gate <= 1 or valid == 2)
            if gate == 2: legal = legal and stage == 4
            if gate == 4: legal = legal and phase != 3
            if gate > 1: legal = legal and site == 0
        state = 'FALSE' if kind == 1 else 'CAUGHT_EXCEPTION'
    else: return record
    if legal: record.update(state=state, result_complete=True)
    return record

def validate_survival(record, expected):
    validate_identity(expected)
    if record.get("control") != expected: raise ValueError("survival identity binding")
    if type(record) is not dict or record.get('schema') != 2 or type(record['schema']) is not int:
        raise ValueError('survival schema')
    if record.get('wait') not in ('DELIVERED','WAIT_UNOBSERVED'): raise ValueError('survival wait')
    encoded = record.get('raw')
    if encoded is None: raw = None
    elif type(encoded) is str and len(encoded) == 128 and all(c in '0123456789abcdef' for c in encoded):
        raw = bytes.fromhex(encoded)
    else: raise ValueError('survival bytes')
    expected = decode_survival(raw, record['wait'] == 'DELIVERED', expected)
    # Type-exact comparison rejects bool/int aliases as well as extra fields.
    import json
    if json.dumps(record, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('survival crosscheck')

child_diagnostic, parent_diagnostic = callback_record('CHILD', expected_control) if diagnostic_enabled else None, callback_record('PARENT', expected_control) if diagnostic_enabled else None
diagnostic_io_failed = False
teardown_snapshot = None
survival = None
binding0 = h._B_SURVIVAL
wait_delivered = False
child_steps = []
parent_steps = []
child_step_count = 0
def publish_diagnostic(name, record):
    global diagnostic_io_failed
    if diagnostic_enabled:
        try:
            if name == 'child-refusal.json': validate_survival(record, expected_control)
            elif name != 'directed-outcome.json': validate_callback(record, expected_control)
            h.atomic(root/name, record)
        except (OSError, RuntimeError, TypeError, ValueError): diagnostic_io_failed = True

def witness_snapshot(w, record):
    slot = w.__dict__.get('_refusal')
    if slot:
        for key in ('first_site', 'gate', 'hook_gate'): record[key] = slot[key]
        if slot['kind'] != 'UNREACHED': record['kind'] = slot['kind']
    record['valid'] = w.valid if type(w.valid) is bool else 'OTHER'
    record['phase'] = {'armed':'ARMED', 'backpressure':'BACKPRESSURE', 'timeout':'TIMEOUT'}.get(w.phase, 'OTHER')

def guard_snapshot(g):
    forms = {NODE+': incomplete or invalid escaping B parent contract':'ESCAPING_B',
        NODE+': missing terminal maintenance receipt':'MISSING_RECEIPT'}
    for pid, status in g.receipts:
        forms[NODE+': '+NODE+f': maintenance terminal wait pid={pid} status={status}: incomplete or invalid armed B contract'] = 'ARMED_B'
    failures = [forms.get(value, 'OTHER') for value in g.failures[:8]]
    stop = g.session.shouldstop
    first = bool(g.failures) and stop == g.failures[0]
    last = bool(g.failures) and stop == g.failures[-1]
    statuses = [status if type(status) is int and status in (9,18944) else 'OTHER'
        for pid, status in g.receipts[:2]]
    return dict(failures=failures, failure_overflow=len(g.failures)>8,
        shouldstop=('FALSE' if stop is False else 'MATCH_BOTH' if first and last else
            'MATCH_FIRST' if first else 'MATCH_LAST' if last else 'OTHER'),
        owned=min(len(g.owned),9), owned_overflow=len(g.owned)>9,
        receipts=min(len(g.receipts),9), receipt_count_overflow=len(g.receipts)>9,
        receipt_statuses=statuses, receipt_status_overflow=len(g.receipts)>2)

def snapshot_complete(snapshot):
    return (snapshot is not None and
        all(not snapshot[key] for key in ('failure_overflow', 'owned_overflow',
            'receipt_count_overflow', 'receipt_status_overflow')) and
        all(value in ('ESCAPING_B', 'MISSING_RECEIPT', 'ARMED_B') for value in snapshot['failures']) and
        snapshot['shouldstop'] in ('FALSE', 'MATCH_FIRST', 'MATCH_LAST', 'MATCH_BOTH') and
        all(type(value) is int and value in (9, 18944) for value in snapshot['receipt_statuses']))

def bounded_baseline_snapshot(value):
    keys = {'failures','failure_overflow','shouldstop','owned','owned_overflow','receipts',
        'receipt_count_overflow','receipt_statuses','receipt_status_overflow'}
    return (type(value) is dict and set(value) == keys and
        type(value['failures']) is list and len(value['failures']) <= 8 and
        type(value['receipt_statuses']) is list and len(value['receipt_statuses']) <= 2 and
        all(type(value[k]) is bool for k in ('failure_overflow','owned_overflow','receipt_count_overflow','receipt_status_overflow')) and
        all(type(value[k]) is int and 0 <= value[k] <= 9 for k in ('owned','receipts')) and snapshot_complete(value))

def validate_baseline_records(records, expected, node, binding):
    validate_identity(expected)
    if (type(binding) is not dict or binding != dict(schema=2, control=expected, node=node) or
            node != NODE): raise ValueError('external node/private binding')
    if type(records) is not dict or set(records) != {'child-refusal.json','parent-callback.json','directed-outcome.json'}:
        raise ValueError('baseline leaves')
    child, parent, outcome = (records[k] for k in ('child-refusal.json','parent-callback.json','directed-outcome.json'))
    validate_survival(child, expected)
    validate_callback(parent, expected)
    if parent['role'] != 'PARENT': raise ValueError('parent role')
    qualification = parent['qualification']
    if type(qualification['terminal']) is bool:
        raw, ef, suffix = (parent[k] for k in ('raw18944','exact_ef','exact_t'))
        if (type(raw) is not bool or type(parent['status']) is not int or raw != (parent['status'] == 18944) or
                (type(ef) is not bool if raw else ef != NOT_REACHED) or
                (type(suffix) is not bool if raw and ef is True else suffix != NOT_REACHED) or
                qualification['terminal'] != (raw is True and ef is True and suffix is True) or
                parent['terminal_called'] is not True): raise ValueError('terminal capture crosscheck')
    for key in ('observe_status','observe_return'):
        if type(qualification[key]) is bool and (parent[key] is not qualification[key] or parent['observe_entered'] is not True):
            raise ValueError('observe capture crosscheck')
    if type(qualification['observe_witness']) is bool:
        armed, witness = parent['observe_armed'], parent['observe_witness']
        if (type(armed) is not bool or (type(witness) is not bool if armed else witness != NOT_REACHED) or
                qualification['observe_witness'] != (armed is True and witness is True)):
            raise ValueError('witness capture crosscheck')
    keys = {'schema','control','role','qualification','pytest_exit','phases','duplicate_phase',
        'unknown_phase','teardown','final','io_failed','callback_other','missing_records','diagnostic'}
    if type(outcome) is not dict or set(outcome) != keys: raise ValueError('outcome schema')
    if type(outcome['schema']) is not int or outcome['schema'] != 2 or outcome['control'] != expected or outcome['role'] != 'PARENT':
        raise ValueError('outcome identity')
    validate_qualification(outcome['qualification'])
    if outcome['qualification'] != parent['qualification']: raise ValueError('qualification crosscheck')
    for key in ('duplicate_phase','unknown_phase','io_failed','callback_other'):
        if type(outcome[key]) is not bool: raise ValueError('outcome flag')
    if (type(outcome['phases']) is not dict or set(outcome['phases']) != {'setup','call','teardown'} or
            any(v not in ('PASSED','FAILED',NOT_REACHED,'OTHER') for v in outcome['phases'].values())):
        raise ValueError('outcome phases')
    if type(outcome['missing_records']) is not list or any(v not in ('child-refusal.json','parent-callback.json') for v in outcome['missing_records']):
        raise ValueError('missing records')
    exit_code = outcome['pytest_exit']
    complete = (type(exit_code) is int and exit_code in range(6) and child['result_complete'] and
        not any(outcome[k] for k in ('duplicate_phase','unknown_phase','io_failed','callback_other','missing_records')) and
        bounded_baseline_snapshot(outcome['teardown']) and bounded_baseline_snapshot(outcome['final']) and
        all(v in ('PASSED','FAILED') for v in outcome['phases'].values()) and
        all(type(parent['qualification'][k]) is bool for k in QUALIFICATION_KEYS) and
        parent['qualification']['observe_calls'] == 1 and parent['qualification']['assertions'] in ('PASSED','FAILED'))
    if outcome['diagnostic'] != ('COMPLETE' if complete else 'RED'): raise ValueError('diagnostic completeness')
    passed = (complete and exit_code == 0 and all(v == 'PASSED' for v in outcome['phases'].values()) and
        parent['qualification']['assertions'] == 'PASSED' and child['state'] == 'TRUE' and child['milestones'] == list(MILESTONES) and
        all(not outcome[k]['failures'] and outcome[k]['shouldstop'] == 'FALSE' and outcome[k]['owned'] == 0 and
            outcome[k]['receipts'] == 1 and outcome[k]['receipt_statuses'] == [18944] for k in ('teardown','final')))
    if exit_code == 0 and not passed: raise ValueError('false baseline PASS')
    return dict(diagnostic='COMPLETE' if complete else 'RED', test_status='PASS' if passed else 'FAIL')

def publish_outcome(exit_code):
    if not diagnostic_enabled: return
    phases = dict.fromkeys(('setup','call','teardown'), NOT_REACHED)
    duplicate = unknown = False
    for phase, passed in reports:
        if phase not in phases: unknown = True; continue
        if phases[phase] != NOT_REACHED: duplicate = True
        phases[phase] = 'PASSED' if passed is True else 'FAILED' if passed is False else 'OTHER'
    snapshot = guard_snapshot(guards[0]) if len(guards)==1 else None
    if type(exit_code) is int and exit_code not in range(6): exit_code = 'OTHER'
    missing = [name for name in ('child-refusal.json','parent-callback.json') if not (root/name).is_file()]
    callback_other = False
    for name in ('child-refusal.json','parent-callback.json'):
        if name in missing: continue
        try:
            callback = h.metadata(root/name)
            if name == 'child-refusal.json':
                validate_survival(callback, expected_control)
                callback_other = callback_other or not callback['result_complete']
            else:
                validate_callback(callback, expected_control)
                callback_other = callback_other or any(value == 'OTHER' for value in callback.values())
        except (OSError, RuntimeError, TypeError, ValueError): callback_other = True
    record = dict(schema=2, control=expected_control, role='PARENT',
        qualification=parent_diagnostic['qualification'].copy(),
        pytest_exit=exit_code, phases=phases, duplicate_phase=duplicate,
        unknown_phase=unknown, teardown=teardown_snapshot if teardown_snapshot is not None else NOT_REACHED, final=snapshot if snapshot is not None else NOT_REACHED,
        io_failed=diagnostic_io_failed, callback_other=callback_other, missing_records=missing, diagnostic='RED')
    # These observations never change pytest/result/guard acceptance.
    if (not missing and not diagnostic_io_failed and not callback_other and not duplicate and not unknown and
            snapshot_complete(snapshot) and snapshot_complete(teardown_snapshot) and
            all(value in ('PASSED','FAILED') for value in phases.values()) and type(exit_code) is int and
            all(type(parent_diagnostic['qualification'][k]) is bool for k in QUALIFICATION_KEYS) and
            parent_diagnostic['qualification']['observe_calls'] == 1 and
            parent_diagnostic['qualification']['assertions'] in ('PASSED','FAILED')): record['diagnostic'] = 'COMPLETE'
    try:
        records = {name: h.metadata(root/name) for name in ('child-refusal.json','parent-callback.json')}
        records['directed-outcome.json'] = record
        validate_baseline_records(records, expected_control, NODE, dict(schema=2, control=expected_control, node=NODE))
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        diagnostic_io_failed = True
        record['io_failed'], record['diagnostic'] = True, 'RED'
    publish_diagnostic('directed-outcome.json', record)
owner = os.getpid()
parent_owner = owner
def evidence(label):
    global owner, steps, child_step_count
    if diagnostic_enabled:
        # Preserve the existing role observation, without a new identity probe.
        current = os.getpid()
        if owner != current:
            owner, steps = current, []
            child_step_count = 0
        if current != parent_owner:
            try:
                assert label not in steps and len(steps) < 24
                assert child_step_count < 5 and label == MILESTONES[child_step_count]
            except BaseException:
                try:
                    survival[34] = 1
                    survival[35] = 254
                except BaseException: pass
                raise
            steps.append(label)
            index = child_step_count
            child_step_count += 1
            try:
                survival[24+2*index] = index+1
                survival[25+2*index] = 254-index
            except BaseException:
                try:
                    survival[34] = 1
                    survival[35] = 254
                except BaseException: pass
            return
        assert label not in steps and len(steps) < 24
        steps.append(label)
        parent_steps.append(label)
        return
    if owner != os.getpid(): owner, steps = os.getpid(), []
    assert label not in steps and len(steps) < 24
    steps.append(label)
    h.atomic(root/(str(os.getpid())+'.json'), dict(control=mode, steps=steps))
import one_node_observer as observer
observer0 = observer.Observer
import atexit
def restore_observer():
    observer.Observer = observer0
    assert observer.Observer is observer0
atexit.register(restore_observer)
class InertObserver:
    # Observational sink ONLY: a real distinct regular FD remains available for
    # B's pipe-exclusion check. No certificate/predicate/guard is replaced.
    def __init__(self, request, code):
        self.fd = os.open(root/'inert-observer', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        request.addfinalizer(self.close)
    def sample(self, *a, **kw): pass
    def close(self):
        fd, self.fd = self.fd, None
        if fd is not None: os.close(fd)
if mode != 'original74-real-observer': observer.Observer = InertObserver
def previous_trace(frame, event, arg): return None
if mode == 'previous-trace': sys.settrace(previous_trace)
cls = h.BlockedPipeWitness_B
original = {name: getattr(cls, name) for name in
    ('__init__', 'eagain', 'trace_B', 'final_child', 'terminal', 'clear')}
guards, reports = [], []
predicate_e = {'wrong-errno', 'subclass', 'raw-budget', 'offset', 'view-owner', 'view-slice'}
predicate_f = {'premature', 'route561', 'raise-offset', 'call-offset',
    'absent-route', 'extra-route', 'reordered-route', 'wrong-args'}
admission = {'missing-obj', 'wrong-obj', 'wrong-node', 'parameter', 'wrong-code', 'closure', 'operation'}
def probe_e(w, exc, frame):
    saved = w.__dict__.copy()
    try: return original['eagain'](w, exc, frame)
    finally: w.__dict__.clear(); w.__dict__.update(saved)
def extra(exc): raise exc
class Plugin:
    def pytest_runtest_makereport(self, item, call):
        reports.append((call.when, call.excinfo is None))
        if call.when == 'setup':
            assert call.excinfo is None or (mode == 'setup-failure' and
                call.excinfo.type is OSError and call.excinfo.value.args == (errno.ENOSPC, 'B setup control'))
        if mode == 'setup-failure' and call.when == 'call':
            assert call.excinfo is not None and call.excinfo.type is OSError
            assert call.excinfo.value.args == (errno.ENOSPC, 'B setup control')
            evidence('invocation-setup-failure-classified')
        if call.when == 'call' and call.excinfo is None: evidence('original-assertions-completed')
    @pytest.fixture(autouse=True)
    def directed(self, request):
        import gc, weakref
        patch = pytest.MonkeyPatch()
        witness_refs = []
        bind0 = cls.bind_hooks
        trace0 = sys.gettrace()
        hooks0 = (os.fork, os.waitpid, os.write, os._exit, recovery._run_bounded_maintenance)
        node = request.node
        assert node.nodeid == NODE and callable(node.obj)
        assert node.obj is sys.modules['test_recovery'].test_t1_real_blocked_pipe_backpressure_deadline
        guard = h.MaintenanceWaitGuard(node, types.SimpleNamespace(shouldstop=False))
        guards.append(guard)
        observe0 = guard.observe
        def observe(result, **kw):
            if diagnostic_enabled:
                qualification = parent_diagnostic['qualification']
                if qualification['observe_calls'] != 0: raise AssertionError('duplicate baseline observe')
                qualification['observe_calls'] = 1
                parent_diagnostic['observe_entered'] = True
                parent_diagnostic['assertion'] = 'OBSERVE_STATUS'
                if result[1] == (9 if mode.startswith('schedule9-') else 74 << 8):
                    status_ok = True
                else:
                    status_ok = False
                parent_diagnostic['observe_status'] = status_ok
                # Publication is deferred to handback.
                parent_diagnostic['qualification']['observe_status'] = status_ok
            else:
                assert result[1] == (9 if mode.startswith('schedule9-') else 74 << 8)
            if mode == 'actual-overflow': expected = b'B:overflow;'
            elif mode in admission | predicate_e or mode.startswith('fd-') or mode == 'missing-eagain': expected = b'B:a;'
            elif mode in predicate_f or mode in {'replacement', 'same-object-extra-frame',
                    'trace-disabled', 'writer-failure', 'direct-exit74'}: expected = b'B:a;B:e;'
            elif mode == 'schedule9-no-f': expected = b'B:a;B:e;B:t;B:k;'
            elif mode == 'schedule9-f': expected = b'B:a;B:e;B:f;B:t;B:k;'
            else: expected = b'B:a;B:e;B:f;B:t;'
            if diagnostic_enabled:
                parent_diagnostic['assertion'] = 'OBSERVE_WITNESS'
                if kw['armed_B']:
                    parent_diagnostic['observe_armed'] = True
                    if kw['witness'] == expected:
                        witness_ok = True
                    else:
                        witness_ok = False
                    parent_diagnostic['observe_witness'] = witness_ok
                else:
                    parent_diagnostic['observe_armed'] = False
                    witness_ok = False
                # Publication is deferred to handback.
                qualification['observe_witness'] = witness_ok
            else:
                assert kw['armed_B'] and kw['witness'] == expected
            observed = observe0(result, **kw)
            if diagnostic_enabled:
                qualification['observe_return'] = observed is result
                qualification['receipt_owned'] = bool(guard.receipts[-1] == result and result[0] not in guard.owned)
                parent_diagnostic['observe_completed'] = True
                parent_diagnostic['observe_return'] = qualification['observe_return']
                parent_diagnostic['assertion'] = 'OBSERVE_RETURN'
                if all(qualification[key] is True for key in QUALIFICATION_KEYS):
                    evidence('exact-terminal-readback')
                return observed
            assert observed is result
            assert guard.receipts[-1] == result and result[0] not in guard.owned
            evidence('exact-terminal-readback')
            return result
        patch.setattr(guard, 'observe', observe)
        finals = []
        def finish():
            # Target's MonkeyPatch fixture may already have undone its hooks.
            # Our stack never records any target-owned observed closure.
            patch.undo()
            for final in reversed(finals): final()
            assert sys.gettrace() is trace0
            assert (os.fork, os.waitpid, os.write, os._exit, recovery._run_bounded_maintenance) == hooks0
            assert all(getattr(cls, n) is f for n, f in original.items())
            assert cls.bind_hooks is bind0
            if mode != 'setup-failure':
                gc.collect()
                assert all(ref() is None for ref in witness_refs)
            evidence('hooks-restored')
        request.addfinalizer(finish)
        proxy = types.SimpleNamespace(node=node, addfinalizer=finals.append)
        h.install(proxy, patch, guard)
        wrapped_run = recovery._run_bounded_maintenance
        def init(w, *a, **kw):
            # Establish admission first; then mutate and reconstruct on the same
            # real target caller. Missing obj is never recovered from the module.
            original['__init__'](w, *a, **kw)
            witness_refs.append(weakref.ref(w))
            assert w.eligible and w.valid
            if diagnostic_enabled:
                w._refusal = dict(first_site=0, kind='UNREACHED', gate=0, hook_gate=0, final_active=False)
            evidence('admission-baseline')
            if mode in admission:
                a = list(a); n = a[0]
                if mode == 'missing-obj':
                    n = types.SimpleNamespace(nodeid=node.nodeid, originalname=node.originalname)
                    assert not hasattr(n, 'obj')
                elif mode == 'wrong-obj':
                    n = types.SimpleNamespace(nodeid=node.nodeid, originalname=node.originalname, obj=lambda: None)
                    assert n.obj is not node.obj
                elif mode == 'wrong-node':
                    n = types.SimpleNamespace(nodeid='foreign::'+node.nodeid, originalname=node.originalname, obj=node.obj)
                    assert n.nodeid != node.nodeid
                elif mode == 'parameter':
                    n = types.SimpleNamespace(nodeid=node.nodeid, originalname=node.originalname,
                        obj=node.obj, callspec=types.SimpleNamespace(params={'foreign': True}))
                    assert n.callspec.params
                elif mode == 'wrong-code':
                    a[1] = lambda *a: None
                    assert a[1].__code__ is not w.runner
                elif mode == 'operation':
                    a[2] = lambda value: 'foreign'
                    assert a[2].__code__ is not w.operation.__code__
                elif mode == 'closure':
                    frame = a[3]; real = frame.f_locals['observed']
                    def cell(value): return (lambda: value).__closure__[0]
                    cells = tuple(cell((lambda *a: 1) if name == 'write' else
                        cls.closure(real)[name]) for name in real.__code__.co_freevars)
                    clone = types.FunctionType(real.__code__, real.__globals__, closure=cells)
                    assert clone.__code__ is real.__code__ and cls.closure(clone)['write'] is not w.writer
                    frame.f_locals['observed'] = clone
                    assert frame.f_locals['observed'] is clone
                a[0] = n
                original['__init__'](w, *a, **kw)
                assert w.eligible and not w.valid
                evidence('admission-mutated-rejected')
        patch.setattr(cls, '__init__', init)
        def eagain(w, exc, frame):
            assert probe_e(w, exc, frame), 'EAGAIN prerequisite absent; never negative PASS'
            evidence('real-eagain-baseline')
            if mode.startswith('fd-'):
                fd = frame.f_locals['write_fd']; old = w.identity(fd)[0]
                assert old == w.pipe and old[:2] != w.identity(w.witness_fd)[0][:2]
                if mode == 'fd-witness-alias': other = os.dup(w.witness_fd)
                elif mode == 'fd-other': other = os.open('/dev/null', os.O_WRONLY)
                else:
                    r, v = os.pipe(); other = r if mode == 'fd-read-access' else v
                    os.close(v if mode == 'fd-read-access' else r)
                replacement = w.identity(other)[0]
                try:
                    if mode == 'fd-reuse': os.close(fd)
                    os.dup2(other, fd)
                    assert w.identity(fd)[0] == replacement and replacement[:2] != old[:2]
                    assert replacement[3] == (os.O_RDONLY if mode == 'fd-read-access' else os.O_WRONLY)
                    if mode == 'fd-witness-alias': assert replacement[:2] == w.identity(w.witness_fd)[0][:2]
                finally: os.close(other)
                with pytest.raises(OSError) as closed: os.fstat(other)
                assert closed.value.errno == errno.EBADF
                evidence('actual-dup2-readback-temp-closed')
            if mode in predicate_e:
                inner = exc.__traceback__.tb_next.tb_frame
                if mode == 'wrong-errno': exc.errno = errno.EPIPE; assert exc.errno == errno.EPIPE
                elif mode == 'subclass':
                    class Sub(BlockingIOError): pass
                    exc = Sub(errno.EAGAIN, 'control').with_traceback(exc.__traceback__)
                    assert type(exc) is Sub
                elif mode == 'raw-budget':
                    frame.f_locals['raw'] = bytearray(1048577)
                    w.framed_id = id(frame.f_locals['raw'])
                    offset = frame.f_locals['offset']
                    inner.f_locals['raw'] = memoryview(frame.f_locals['raw'])[offset:offset+65536]
                    assert len(frame.f_locals['raw']) == 1048577
                    assert id(frame.f_locals['raw']) == w.framed_id
                    assert inner.f_locals['raw'].obj is frame.f_locals['raw']
                elif mode == 'offset':
                    frame.f_locals['offset'] = -1; assert frame.f_locals['offset'] == -1
                elif mode == 'view-owner':
                    inner.f_locals['raw'] = memoryview(bytes(inner.f_locals['raw']))
                    assert inner.f_locals['raw'].obj is not frame.f_locals['raw']
                elif mode == 'view-slice':
                    inner.f_locals['raw'] = memoryview(frame.f_locals['raw'])[:1]
                    assert inner.f_locals['raw'].nbytes == 1
                evidence('source-predicate-mutated')
            answer = original['eagain'](w, exc, frame)
            reject = mode.startswith('fd-') or mode in predicate_e or mode == 'missing-eagain'
            if mode == 'missing-eagain': answer = False  # Withheld certificate control, no fake kernel EAGAIN.
            assert answer is not reject
            if mode.startswith('fd-'):
                os.close(fd)
                with pytest.raises(OSError) as closed_result: os.fstat(fd)
                assert closed_result.value.errno == errno.EBADF
                evidence('replacement-result-fd-closed')
            evidence('eagain-rejected' if reject else 'eagain-accepted')
            return answer
        # Repeated EAGAIN retries must not create repeated evidence or mutations.
        seen = [False]
        def once_e(w, exc, frame):
            if seen[0]: return original['eagain'](w, exc, frame)
            seen[0] = True
            return eagain(w, exc, frame)
        patch.setattr(cls, 'eagain', once_e)
        timeout_ids = [0, 0]
        def trace(w, frame, event, arg):
            answer = original['trace_B'](w, frame, event, arg)
            if (mode == 'direct-exit74' and frame.f_code is w.runner and event == 'exception'
                    and frame.f_locals.get('pid') == 0 and w.phase == 'backpressure' and w.valid):
                evidence('unrelated-exit74-after-eagain')
                os._exit(74)
            if (mode == 'schedule9-no-f' and frame.f_code is w.runner and event == 'exception'
                    and frame.f_locals.get('pid') == 0 and w.phase == 'backpressure' and w.valid):
                # Scheduling intervention ONLY, after original trace emitted e.
                evidence('scheduling-after-real-eagain')
                import time
                time.sleep(1)
            if frame.f_code is w.runner and event == 'exception' and frame.f_locals.get('pid') == 0 and w.phase == 'timeout':
                assert w.valid and w.timeout_id == id(arg[1])
                timeout_ids[0] = id(arg[1])
                evidence('actual-child558-timeout')
                if mode == 'replacement':
                    replacement = ValueError(*arg[1].args)
                    assert replacement is not arg[1] and replacement.args == arg[1].args
                    timeout_ids[1] = id(replacement)
                    evidence('actual-replacement-raised')
                    raise replacement
                if mode == 'same-object-extra-frame':
                    evidence('actual-same-object-reraise')
                    extra(arg[1])
                if mode == 'trace-disabled':
                    sys.settrace(None); assert sys.gettrace() is None
                    evidence('trace-disabled-readback')
            return answer
        patch.setattr(cls, 'trace_B', trace)
        def final(w, caller):
            if mode in admission or mode.startswith('fd-') or mode in predicate_e or mode == 'missing-eagain':
                assert not original['final_child'](w, caller)
                evidence('final-rejected'); return False
            if mode in {'replacement', 'same-object-extra-frame', 'trace-disabled', 'direct-exit74'}:
                if mode != 'direct-exit74':
                    current = sys.exception()
                    assert type(current) is ValueError and current.args == ('b2_maintenance_timeout',)
                    assert id(current) == timeout_ids[1 if mode == 'replacement' else 0]
                    route, tb = [], current.__traceback__
                    while tb is not None:
                        route.append(tb.tb_frame.f_code); tb = tb.tb_next
                    if mode == 'replacement': assert trace.__code__ in route and timeout_ids[0] != timeout_ids[1]
                    if mode == 'same-object-extra-frame': assert extra.__code__ in route
                    if mode == 'trace-disabled': assert sys.gettrace() is None
                    del current, tb, route
                    evidence('active-exception-route-readback')
                assert not original['final_child'](w, caller)
                evidence('final-rejected'); return False
            answer = original['final_child'](w, caller)
            assert answer, 'actual74 final prerequisite absent'
            evidence('final-baseline')
            if mode in predicate_f:
                exc = sys.exception(); tb = exc.__traceback__
                if mode == 'premature': w.deadline += 1000000; assert not w.expired(caller)
                elif mode == 'route561':
                    import dis
                    site, = [i.offset for i in dis.get_instructions(w.runner)
                        if i.opname == 'RAISE_VARARGS' and i.positions.lineno == 561]
                    exc.__traceback__ = types.TracebackType(None, caller, site, 561)
                    assert exc.__traceback__.tb_lasti == site and site not in w.raise_child
                elif mode == 'raise-offset': w.raise_child = frozenset({-1}); assert tb.tb_lasti not in w.raise_child
                elif mode == 'call-offset': w.exit_sites = frozenset({-1}); assert caller.f_lasti not in w.exit_sites
                elif mode == 'absent-route': exc.__traceback__ = None; assert exc.__traceback__ is None
                elif mode == 'extra-route':
                    exc.__traceback__ = types.TracebackType(tb, caller, tb.tb_lasti, tb.tb_lineno)
                    assert exc.__traceback__.tb_next is tb
                elif mode == 'reordered-route':
                    exc.__traceback__ = types.TracebackType(tb, sys._getframe(), 0, 1)
                    assert exc.__traceback__.tb_frame is not caller
                elif mode == 'wrong-args': exc.args = ('foreign',); assert exc.args == ('foreign',)
                evidence('final-source-predicate-mutated')
                assert not original['final_child'](w, caller)
                evidence('final-rejected'); return False
            if mode == 'writer-failure':
                # Fail the captured B emitter at its final write, after the real
                # final predicate. No observer or attestor result is substituted.
                state = w.emit.__closure__
                bindings = cls.closure(w.emit)
                assert bindings['write'] is h._RAW_WRITE
                def failed(fd, data):
                    assert data == b'B:f;'; evidence('final-writer-failed')
                    raise OSError(errno.ENOSPC, 'B final writer control')
                for name, cell in zip(w.emit.__code__.co_freevars, state):
                    if name == 'write': cell.cell_contents = failed
                assert cls.closure(w.emit)['write'] is failed
            if mode == 'actual-overflow':
                w.emit(b'x'*257)
                evidence('actual-overflow-written')
            if mode == 'schedule9-f':
                # Delay only captured exit, after the original observed_exit
                # has emitted f. Guard/attestor and builtin kill stay intact.
                exit_fn = os._exit
                cells = dict(zip(exit_fn.__code__.co_freevars, exit_fn.__closure__))
                exit0 = cells['exit0'].cell_contents
                def delayed(code):
                    assert code == 74
                    evidence('scheduling-after-real-final-token')
                    import time
                    time.sleep(1)
                    exit0(code)
                cells['exit0'].cell_contents = delayed
                assert cls.closure(exit_fn)['exit0'] is delayed
            evidence('final-accepted'); return True
        patch.setattr(cls, 'final_child', final)
        def terminal(w, result, caller, enrolled, child_bytes):
            global wait_delivered
            if diagnostic_enabled: wait_delivered = True
            answer = original['terminal'](w, result, caller, enrolled, child_bytes)
            if diagnostic_enabled:
                # Original left-to-right qualification, evaluated once inline.
                raw_ok = ef_ok = t_ok = NOT_REACHED
                if ((raw_ok := result[1] == 74 << 8) and
                        (ef_ok := child_bytes == b'B:e;B:f;') and (t_ok := answer == b'B:t;')):
                    terminal_ok = True
                else: terminal_ok = False
                parent_diagnostic['qualification']['terminal'] = terminal_ok
                witness_snapshot(w, parent_diagnostic)
                parent_diagnostic.update(terminal_called=True,
                    terminal_result={b'':'EMPTY',b'B:t;':'T',b'B:t;B:k;':'TK'}.get(answer,'OTHER'),
                    status=result[1] if type(result[1]) is int and 0 <= result[1] <= 65535 else 'OTHER',
                    child_bytes={b'':'EMPTY',b'B:e;':'E',b'B:e;B:f;':'EF'}.get(child_bytes,'OTHER'),
                    suffix={b'':'EMPTY',b'B:t;':'T',b'B:t;B:k;':'TK'}.get(answer,'OTHER'),
                    raw18944=raw_ok, exact_ef=ef_ok, exact_t=t_ok)
                evidence('terminal-reached')
                if terminal_ok: evidence('actual74-parent-qualified')
                return answer
            evidence('terminal-reached')
            if mode in {'schedule9-no-f', 'schedule9-f'}:
                assert result[1] == 9 and w.killed and caller.f_locals['killed'] is True
                assert child_bytes == (b'B:e;' if mode == 'schedule9-no-f' else b'B:e;B:f;')
                assert answer == b'B:t;B:k;'
                evidence('actual-owned-kill-qualified')
            elif mode not in admission | predicate_e | predicate_f and not mode.startswith('fd-') and mode not in {
                'replacement', 'same-object-extra-frame', 'trace-disabled', 'writer-failure', 'missing-eagain'}:
                assert result[1] == 74 << 8 and child_bytes == b'B:e;B:f;' and answer == b'B:t;'
                evidence('actual74-parent-qualified')
            return answer
        patch.setattr(cls, 'terminal', terminal)
        def clear(w):
            if diagnostic_enabled: witness_snapshot(w, parent_diagnostic)
            assert not any(isinstance(value, (types.FrameType, types.TracebackType,
                BaseException, memoryview)) for value in w.__dict__.values())
            original['clear'](w)
            assert w.__dict__ == {}
            evidence('all-witness-references-cleared')
        patch.setattr(cls, 'clear', clear)
        if mode == 'setup-failure':
            def fail_bind(w, *a):
                assert os._exit is not hooks0[3]
                evidence('setup-hooks-installed')
                raise OSError(errno.ENOSPC, 'B setup control')
            patch.setattr(cls, 'bind_hooks', fail_bind)
        if mode == 'fixture-undo':
            target = request.getfixturevalue('monkeypatch')
            before = recovery._maintenance_result_frame
            def owned(*a): raise AssertionError('must not run')
            target.setattr(recovery, '_maintenance_result_frame', owned)
            assert recovery._maintenance_result_frame is owned
            target.undo()
            assert recovery._maintenance_result_frame is before
            evidence('target-fixture-undone')
        yield
        if diagnostic_enabled:
            global teardown_snapshot
            teardown_snapshot = guard_snapshot(guard)
            # Publication is deferred to handback.
        try:
            if guard.failures:
                assert guard.session.shouldstop == guard.failures[-1]
                assert any(s.endswith('incomplete or invalid armed B contract') for s in guard.failures)
                with pytest.raises(pytest.exit.Exception): wrapped_run({}, {}, lambda value: True)
                with h.activate(guard, root/'refused'):
                    with pytest.raises(RuntimeError): h.fresh_run([sys.executable, '-c', 'raise AssertionError("must not launch")'])
                    assert not list((root/'refused').iterdir())
                evidence('both-next-launches-refused')
        finally:
            if diagnostic_enabled: assert_qualification(parent_diagnostic['qualification'])
result = 'ESCAPED'
try:
    if diagnostic_enabled:
        import mmap
        anon = mmap.MAP_ANON if hasattr(mmap, 'MAP_ANON') else mmap.MAP_ANONYMOUS
        survival = mmap.mmap(-1, 64, flags=mmap.MAP_SHARED | anon,
            prot=mmap.PROT_READ | mmap.PROT_WRITE)
        survival[:] = b'BDS\x02' + bytes(60)
        h._B_SURVIVAL = survival
    result = pytest.main(['--noconftest', '-p', 'no:cacheprovider', '-q', '-x', NODE,
        '--basetemp='+str(root/'pytest')], plugins=[Plugin()])
finally:
    try:
        if diagnostic_enabled:
            # Never inspect the mapping unless the original wait was delivered.
            raw_survival = bytes(survival[:]) if wait_delivered else None
            child_diagnostic = decode_survival(raw_survival, wait_delivered, expected_control)
            child_steps = child_diagnostic['milestones']
            publish_diagnostic('child-refusal.json', child_diagnostic)
            publish_diagnostic('parent-callback.json', parent_diagnostic)
            try:
                h.atomic(root/(str(parent_owner)+'.json'), dict(control=mode, steps=parent_steps))
            except (OSError, RuntimeError, TypeError, ValueError): diagnostic_io_failed = True
        publish_outcome(int(result) if type(result) is int or isinstance(result, int) else 'ESCAPED')
    finally:
        h._B_SURVIVAL = binding0
        if survival is not None: survival.close()
        survival = None
if diagnostic_enabled: assert not diagnostic_io_failed, 'baseline diagnostic publication/validation failed'
assert len(guards) == 1
g = guards[0]
restore_observer()
if mode == 'setup-failure':
    # Failure is in invocation setup, so target call is RED, never a negative
    # certificate. Restoration is the only expected success of this control.
    assert result != 0 and not g.receipts and not g.owned
    assert not g.failures and not g.session.shouldstop
else:
    positive = mode in {'original74-real-observer', 'collected-control', 'fixture-undo', 'previous-trace', 'schedule9-no-f', 'schedule9-f'}
    assert result == 0, 'target assertions must complete even for guard negatives'
    assert bool(g.failures) is not positive
    assert len(g.receipts) == 1 and not g.owned
    if positive:
        assert g.receipts[0][1] == (9 if mode.startswith('schedule9-') else 74 << 8)
        assert not g.session.shouldstop
records = [h.metadata(p) for p in root.glob('[0-9]*.json')]
parent, = [r['steps'] for r in records if 'hooks-restored' in r['steps']]
assert 'all-witness-references-cleared' in parent
if mode == 'setup-failure':
    assert parent == ['admission-baseline', 'setup-hooks-installed',
        'all-witness-references-cleared', 'invocation-setup-failure-classified', 'hooks-restored']
if mode != 'setup-failure':
    if diagnostic_enabled:
        child, = [child_steps] if 'final-accepted' in child_steps else []
    else:
        child, = [r['steps'] for r in records if 'final-rejected' in r['steps'] or 'final-accepted' in r['steps'] or 'scheduling-after-real-eagain' in r['steps']]
    assert 'admission-baseline' in parent
    if mode not in admission: assert 'real-eagain-baseline' in child
    if mode in {'replacement', 'same-object-extra-frame', 'trace-disabled'}: assert 'actual-child558-timeout' in child
    if mode in predicate_f: assert 'final-baseline' in child and 'final-source-predicate-mutated' in child
    if not positive: assert 'both-next-launches-refused' in parent
'''


@pytest.mark.parametrize('control', B_CONTROL_SEAMS, ids=lambda c: 'B-S-'+c)
def test_B_directed_isolated_seams(tmp_path, control):
    """Original target, finite one-shot controls; scheduling failures stay RED."""
    import subprocess
    source = _B_SEAM_PROGRAM.replace('mode, root = CONTROL, Path(ROOT)',
        'mode, root = '+repr(control)+', Path('+repr(str(tmp_path/'proof'))+')')
    source = source.replace('NODE', repr(B_CONTROL_NODE))
    # Explicit path, no installed/runtime equivalence claim. No fresh_run here:
    # real-observer bootstrap's missing addfinalizer is a recorded source block.
    prefix = 'import sys; sys.path[:0] = '+repr([
        str(Path(harness.__file__).parent), str(Path(harness.__file__).parent.parent/'diagnostics'/'p-one-node')])+'\n'
    result = subprocess.run([sys.executable, '-B', '-c', prefix+source],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr


B_CONTROL_PARENT = (
    '624', '627-positive-remaining', '74-kill-attempt', '9-no-f', '9-f',
    '624-positive-remaining', '627-no-select', 'wrong-active', 'route636',
    'route648', 'raw-nonempty', 'deadline', 'wrong-pid', 'unenrolled',
    'wrong-caller', 'wrong-parent', 'kill-absent', 'kill-wrong-child',
    'kill-wrong-signal', '74-no-f', 'child-owned-t', 'child-owned-a',
    'child-owned-k',
)


def _B_method_ast(name):
    import ast
    tree = ast.parse(Path(harness.__file__).read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BlockedPipeWitness_B')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _B_eval_source(expression, values):
    """Exact AST expression projection, explicit synthetic inputs, no imports/IO."""
    import ast
    return eval(compile(ast.Expression(body=expression), harness.__file__, 'eval'), values)


@pytest.mark.parametrize('control', B_CONTROL_PARENT, ids=lambda c: 'B-P-'+c)
def test_B_parent_source_predicates(tmp_path, control):
    """Synthetic predicate controls, NEVER terminal/kill scheduling evidence.

    Extract the exact original terminal admission, route and kill expressions.
    Capability/frame fixtures are synthetic. Real frame/active-handler/kill
    behavior belongs to B-S controls; this projection only isolates dimensions.
    """
    import ast
    import signal
    method = _B_method_ast('terminal')
    body = next(n for n in method.body if isinstance(n, ast.Try)).body
    admission = next(n.test for n in body if isinstance(n, ast.If))
    # The first expression is `not (all prerequisites)`: False admits.
    assert isinstance(admission, ast.UnaryOp) and isinstance(admission.op, ast.Not)
    status_if = next(n for n in body if isinstance(n, ast.If) and
        isinstance(n.test, ast.BoolOp) and 'status == 74 << 8' in ast.unparse(n.test))
    kill_test = status_if.orelse[0].test
    expire = next(n.value for n in _B_method_ast('expired').body if isinstance(n, ast.Return))
    # Skip the exception-path False reset; select the full opcode predicate.
    kill_pending = next(n.value for n in ast.walk(_B_method_ast('trace_B'))
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.BoolOp) and
        any(isinstance(t, ast.Attribute) and t.attr == 'kill_pending' for t in n.targets))
    runner, reap, other = object(), object(), object()
    parent = SimpleNamespace(f_code=runner, f_lasti=661, f_locals={})
    caller = SimpleNamespace(f_code=reap, f_lasti=581, f_back=parent,
        f_locals={'pid': 123, 'killed': True})
    # Synthetic traceback is an INPUT to the extracted expression, not installed
    # on a real exception and not presented to the complete attestor.
    tb = SimpleNamespace(tb_frame=parent, tb_next=None, tb_lasti=624)
    exc = ValueError('b2_maintenance_timeout')
    local = dict(pid=123, original=exc, raw=bytearray(), deadline=10., remaining=-1.)
    parent.f_locals = local
    clock = lambda: 11.
    w = SimpleNamespace(valid=True, hooks=lambda: True, trace_callback=object(),
        reap=reap, runner=runner, runner_id=id(parent), reap_sites={661},
        wait_first={581}, wait_last={604}, child=123, owner=os.getpid(),
        timeout=harness.BlockedPipeWitness_B.timeout, raise_parent={624}, raise_empty={627},
        empty_select=True, killed=False, deadline=10., clock=clock,
        recovery=SimpleNamespace(time=SimpleNamespace(monotonic=clock)))
    projected_sys = SimpleNamespace(gettrace=lambda: w.trace_callback, exception=lambda: exc)
    values = dict(self=w, caller=caller, parent=parent, local=local, exc=exc, tb=tb,
        result=(123, 74 << 8), enrolled={123}, child_bytes=b'B:e;B:f;',
        sys=projected_sys, os=os, signal=signal, frame=parent, math=__import__('math'))
    def expired(frame):
        return _B_eval_source(expire, dict(values, frame=frame, deadline=frame.f_locals.get('deadline')))
    w.expired = expired
    assert _B_eval_source(admission, values) is False
    assert expired(parent)
    assert _B_eval_source(status_if.test, dict(values, status=74 << 8))
    if control in ('627-positive-remaining', '627-no-select'):
        tb.tb_lasti = 627; local['remaining'] = 2.
        assert _B_eval_source(admission, values) is False
        assert local['remaining'] > 0 and expired(parent)
    if control in ('74-kill-attempt', '9-no-f', '9-f', 'kill-absent', 'kill-wrong-child', 'kill-wrong-signal'):
        # Exact kill opcode predicate baseline, with all synthetic capabilities
        # and owned caller established before a single independent mutation.
        kill0 = os.kill
        w.kill = kill0
        killframe = SimpleNamespace(f_lasti=593, f_locals={'pid': 123, 'signal': signal},
            f_globals={'os': os})
        kv = dict(self=w, frame=killframe, parent=parent, os=os, signal=signal)
        assert _B_eval_source(kill_pending, kv)
        if control == 'kill-wrong-child':
            killframe.f_locals['pid'] = 124
            assert killframe.f_locals['pid'] != w.child
        elif control == 'kill-wrong-signal':
            killframe.f_locals['signal'] = SimpleNamespace(SIGKILL=15)
            assert killframe.f_locals['signal'] is not signal
        w.killed = control != 'kill-absent' and _B_eval_source(kill_pending, kv)
        caller.f_lasti = 604
        assert w.killed is (control not in ('kill-absent', 'kill-wrong-child', 'kill-wrong-signal'))
    if control == '624-positive-remaining': local['remaining'] = 2.; assert local['remaining'] > 0
    elif control == '627-no-select': w.empty_select = False; assert not w.empty_select
    elif control == 'wrong-active':
        active = ValueError(*exc.args)
        projected_sys.exception = lambda: active
        assert projected_sys.exception() is not local['original']
    elif control in ('route636', 'route648'):
        tb.tb_lasti = int(control[-3:]); assert tb.tb_lasti not in w.raise_parent | w.raise_empty
    elif control == 'raw-nonempty': local['raw'].extend(b'x'); assert len(local['raw']) == 1
    elif control == 'deadline': local['deadline'] = 12.; assert not expired(parent)
    elif control == 'wrong-pid': values['result'] = (124, 74 << 8); assert values['result'][0] != w.child
    elif control == 'unenrolled': values['enrolled'] = set(); assert w.child not in values['enrolled']
    elif control == 'wrong-caller': caller.f_code = other; assert caller.f_code is not w.reap
    elif control == 'wrong-parent': parent.f_code = other; assert parent.f_code is not w.runner
    elif control == '74-no-f': values['child_bytes'] = b'B:e;'; assert b'B:f;' not in values['child_bytes']
    elif control.startswith('child-owned-'):
        values['child_bytes'] += ('B:'+control[-1]+';').encode()
        assert values['child_bytes'] not in (b'B:e;', b'B:e;B:f;')
    status = 9 if control.startswith('9-') or control.startswith('kill-') else 74 << 8
    values['result'] = (values['result'][0], status)
    if control == '9-no-f': values['child_bytes'] = b'B:e;'
    admitted = not _B_eval_source(admission, values)
    suffix = b''
    if admitted:
        if _B_eval_source(status_if.test, dict(values, status=status)): suffix = b'B:t;'
        elif _B_eval_source(kill_test, dict(values, status=status)): suffix = b'B:t;B:k;'
    positive = control in ('624', '627-positive-remaining', '74-kill-attempt', '9-no-f', '9-f')
    assert bool(suffix) is positive
    witness = b'B:a;'+values['child_bytes']+suffix
    guard = harness.MaintenanceWaitGuard(SimpleNamespace(nodeid='B-parent-projection',
        originalname='B-parent-projection'), SimpleNamespace(shouldstop=False))
    guard.owned.add(123)
    raw = (123, status)
    assert guard.observe(raw, witness=witness, armed_B=True, expected_exit=74, expected_signal=9) is raw
    assert guard.receipts == [raw] and not guard.owned
    assert bool(guard.failures) is not positive
    if not positive:
        assert guard.failures[0].endswith('incomplete or invalid armed B contract')
        _B_refused_launches(guard, tmp_path)


@pytest.mark.parametrize('control', ('foreign-owner', 'missing-done'), ids=lambda c: 'B-O-'+c)
def test_B_complete_receipt_ownership(tmp_path, control):
    """Synthetic complete B receipt; precise aggregation error, no child claim."""
    directory = tmp_path/'receipt'
    directory.mkdir(mode=0o700)
    expected = dict(pid=100, node=B_CONTROL_NODE, token='a'*32)
    harness.atomic(directory/'ack.json', expected)
    harness.atomic(directory/'done.json', expected)
    enroll = dict(kind='enroll', owner=100, pid=123)
    terminal = dict(kind='terminal', owner=100, pid=123, status=74 << 8,
        accepted=True, phase='B:a;B:e;B:f;B:t;')
    harness.atomic(directory/'100-1.json', enroll)
    harness.atomic(directory/'100-2.json', terminal)
    assert harness.aggregate(directory, expected) == [(123, 74 << 8)]
    if control == 'foreign-owner':
        enroll['owner'] = terminal['owner'] = 999
        harness.atomic(directory/'100-1.json', enroll)
        harness.atomic(directory/'100-2.json', terminal)
        assert harness.metadata(directory/'100-1.json')['owner'] == harness.metadata(directory/'100-2.json')['owner'] == 999
        with pytest.raises(ValueError) as caught: harness.aggregate(directory, expected)
        assert str(caught.value) == 'foreign helper event owner'
    else:
        (directory/'done.json').unlink()
        assert not (directory/'done.json').exists()
        with pytest.raises(FileNotFoundError) as caught: harness.aggregate(directory, expected)
        assert caught.value.filename == str(directory/'done.json')
    # Project exact fresh_run aggregation-failure handler with the actual caught
    # error. No simulated successful subprocess or forged helper is launched.
    import ast
    tree = ast.parse(Path(harness.__file__).read_bytes())
    fresh = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fresh_run')
    handler = next(n for n in ast.walk(fresh) if isinstance(n, ast.ExceptHandler) and
        'fresh helper receipt failure: ' in ast.unparse(n))
    guard = harness.MaintenanceWaitGuard(SimpleNamespace(nodeid=B_CONTROL_NODE,
        originalname=B_CONTROL_NODE.split('::')[-1]), SimpleNamespace(shouldstop=False))
    exec(compile(ast.Module(body=handler.body, type_ignores=[]), harness.__file__, 'exec'),
        dict(guard=guard, exc=caught.value))
    assert guard.failures == [B_CONTROL_NODE+': fresh helper receipt failure: '+str(caught.value)[:1024]]
    assert guard.session.shouldstop == guard.failures[0]
    _B_refused_launches(guard, tmp_path)


_B_HELPER_SINK = r'''
# Control-process-only sitecustomize. Observer replacement, NOT an attestor.
import atexit, os, tempfile
import one_node_observer as _observer
_original_observer = _observer.Observer
class _ControlSink:
    def __init__(self, request, code):
        self.file = tempfile.TemporaryFile()
        self.fd = self.file.fileno()
        atexit.register(self.close)
    def sample(self, *a, **kw): pass
    def close(self):
        self.file.close()
        self.fd = None
_observer.Observer = _ControlSink
def _restore_observer():
    _observer.Observer = _original_observer
    assert _observer.Observer is _original_observer
atexit.register(_restore_observer)
'''

_B_HELPER_BODY = r'''
import os, sys, pytest
from pathlib import Path
import _maintenance_wait_guard as h
from service import recovery
mode = CONTROL
steps = []
class Plugin:
    @pytest.fixture(autouse=True)
    def bind(self, request):
        patch = pytest.MonkeyPatch()
        original = h.BlockedPipeWitness_B.__init__
        helper_node = h._CURRENT.node
        saved = {name: (hasattr(helper_node, name), getattr(helper_node, name, None))
                 for name in ('obj', 'config')}
        def finish():
            patch.undo()
            for name, (present, value) in saved.items():
                assert hasattr(helper_node, name) is present
                if present: assert getattr(helper_node, name) is value
            assert h.BlockedPipeWitness_B.__init__ is original
            steps.append('binding-restored')
        request.addfinalizer(finish)
        assert request.node.nodeid == NODE
        collected = request.node.obj
        assert callable(collected)
        assert collected is sys.modules['test_recovery'].test_t1_real_blocked_pipe_backpressure_deadline
        assert not hasattr(h._CURRENT.node, 'obj')
        patch.setattr(helper_node, 'config', request.node.config, raising=False)
        assert helper_node.config is request.node.config
        if mode == 'bound':
            patch.setattr(h._CURRENT.node, 'obj', collected, raising=False)
            assert h._CURRENT.node.obj is collected
            steps.append('real-inner-item-bound')
        else:
            assert mode == 'withheld' and not hasattr(h._CURRENT.node, 'obj')
            steps.append('binding-withheld')
        def init(w, *a, **kw):
            baseline = list(a)
            baseline[0] = request.node
            original(w, *baseline, **kw)
            assert w.eligible and w.valid
            steps.append('valid-collected-prerequisite')
            original(w, *a, **kw)
            assert w.eligible and w.valid is (mode == 'bound')
            steps.append('admission-readback')
        patch.setattr(h.BlockedPipeWitness_B, '__init__', init)
        yield
result = pytest.main(['--noconftest', '-p', 'no:cacheprovider', '-q', '-x', NODE,
    '--basetemp='+BASE], plugins=[Plugin()])
assert steps == [('real-inner-item-bound' if mode == 'bound' else 'binding-withheld'),
    'valid-collected-prerequisite', 'admission-readback', 'binding-restored']
g = h._CURRENT
assert not g.owned and len(g.receipts) == 1
if mode == 'withheld':
    assert result != 0 and g.session.shouldstop
    assert any(f.endswith('incomplete or invalid armed B contract') for f in g.failures)
    with pytest.raises(pytest.exit.Exception): recovery._run_bounded_maintenance({}, {}, lambda value: True)
    with pytest.raises(RuntimeError): h.fresh_run([sys.executable, '-c', 'raise AssertionError("must not launch")'])
    raise SystemExit(86)
assert result == 0 and not g.failures and not g.session.shouldstop
assert g.receipts[0][1] == 74 << 8
'''


@pytest.mark.parametrize('control', ('bound', 'withheld'), ids=lambda c: 'B-H-'+c)
def test_B_fresh_real_inner_binding_control_sink(tmp_path, control):
    """Fresh bootstrap with isolated inert sink; no real-observer credit.

    The unchanged real-observer bootstrap cannot satisfy addfinalizer. Keep that
    source finding visible; this control does not repair or qualify that path.
    """
    import subprocess
    custom = tmp_path/'control-site'
    custom.mkdir(mode=0o700)
    (custom/'sitecustomize.py').write_text(_B_HELPER_SINK)
    paths = [str(custom), str(Path(harness.__file__).parent),
        str(Path(harness.__file__).parent.parent/'diagnostics'/'p-one-node')]
    body = _B_HELPER_BODY.replace('CONTROL', repr(control)).replace('NODE', repr(B_CONTROL_NODE))
    body = body.replace('BASE', repr(str(tmp_path/'inner-helper')))
    # This outer control process has no observer installation. Only its actual
    # fresh_run child loads the explicitly private sitecustomize substitution.
    outer = r'''
import os, sys
from pathlib import Path
from types import SimpleNamespace
import _maintenance_wait_guard as h
g = h.MaintenanceWaitGuard(SimpleNamespace(nodeid=NODE,
    originalname=NODE.split('::')[-1]), SimpleNamespace(shouldstop=False))
env = dict(os.environ, PYTHONPATH=PATHS)
with h.activate(g, Path(HELPERS)):
    result = h.fresh_run([sys.executable, '-c', BODY], env=env,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if CONTROL == 'bound' else 86), result.stdout+result.stderr
    assert bool(g.failures) is (CONTROL == 'withheld')
    if g.failures:
        assert g.session.shouldstop == g.failures[0]
        with __import__('pytest').raises(RuntimeError):
            h.fresh_run([sys.executable, '-c', 'raise AssertionError("must not launch")'])
directory, = Path(HELPERS).glob('helper-*')
rows = [h.metadata(p) for p in directory.glob('[0-9]*.json')]
terminal, = [r for r in rows if r['kind'] == 'terminal']
assert terminal['status'] == 74 << 8
assert terminal['accepted'] is (CONTROL == 'bound')
assert terminal['phase'] == ('B:a;B:e;B:f;B:t;' if CONTROL == 'bound' else 'B:a;')
expected = h.metadata(directory/'launch.json')['identity']
if CONTROL == 'bound':
    assert h.aggregate(directory, expected) == g.receipts
else:
    try: h.aggregate(directory, expected)
    except ValueError as exc:
        assert g.failures == [NODE+': fresh helper receipt failure: '+str(exc)[:1024]]
    else: raise AssertionError('unaccepted B helper aggregated')
'''
    # Literal token replacement uses longest identifiers first; values are
    # Python literals, never interpolated shell commands.
    replacements = dict(HELPERS=str(tmp_path/'helpers'), PATHS=os.pathsep.join(paths),
        BODY=body, CONTROL=control, NODE=B_CONTROL_NODE)
    for token, value in replacements.items(): outer = outer.replace(token, repr(value))
    prefix = 'import sys; sys.path.insert(0, '+repr(str(Path(harness.__file__).parent))+')\n'
    result = subprocess.run([sys.executable, '-B', '-c', prefix+outer],
        capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize('control', ('class', 'subclass', 'args-count', 'args-type'),
    ids=lambda c: 'B-X-'+c)
def test_B_timeout_shape_source_predicate(tmp_path, control):
    """Exact timeout shape only; no active handler or route certificate."""
    cls = harness.BlockedPipeWitness_B
    baseline = ValueError('b2_maintenance_timeout')
    assert cls.timeout(baseline)
    class SubTimeout(ValueError): pass
    class Text(str): pass
    if control == 'class':
        changed = RuntimeError(*baseline.args)
        assert changed.args == baseline.args and type(changed) is RuntimeError
    elif control == 'subclass':
        changed = SubTimeout(*baseline.args)
        assert changed.args == baseline.args and isinstance(changed, ValueError)
        assert type(changed) is not ValueError
    elif control == 'args-count':
        changed = ValueError(*baseline.args, 'extra')
        assert len(changed.args) == 2
    else:
        changed = ValueError(Text(baseline.args[0]))
        assert changed.args == baseline.args and type(changed.args[0]) is not str
    assert not cls.timeout(changed)
    guard = harness.MaintenanceWaitGuard(SimpleNamespace(nodeid='B-timeout-shape',
        originalname='B-timeout-shape'), SimpleNamespace(shouldstop=False))
    guard.owned.add(123)
    raw = (123, 74 << 8)
    assert guard.observe(raw, armed_B=True, witness=b'B:a;B:e;', expected_exit=74) is raw
    assert guard.receipts == [raw] and not guard.owned
    assert len(guard.failures) == 1 and guard.failures[0].endswith('incomplete or invalid armed B contract')
    _B_refused_launches(guard, tmp_path)


# Combined successor C2: finite pre-capture controls. These isolated inert-sink
# processes are guard controls, never real-observer diagnostic qualification.
B_CONTROL_B1 = ('clock1000', 'observed', 'ready', 'select_blocked', 'operation', 'outer-binding')


def _B_precapture_program(control):
    """Derive only the appended control harness; frozen target bytes stay fixed."""
    source = _B_SEAM_PROGRAM
    source = source.replace("admission = {", "admission = {"+repr(control)+", ", 1)
    anchor = "        guard = h.MaintenanceWaitGuard(node, types.SimpleNamespace(shouldstop=False))"
    setup = r'''
        import ast
        function = node.obj
        code0 = function.__code__
        plain, authority, roles, tree = cls.target_source(function, node)
        assert cls.code_projection(code0) == cls.code_projection(function.__code__)
        # Restoration is registered before changing the real collected object.
        def restore_code():
            function.__code__ = code0
            assert node.obj is function and function.__code__ is code0
        request.addfinalizer(restore_code)
        if mode == 'outer-binding':
            target = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == cls.name)
            call, = [n for n in ast.walk(target) if isinstance(n, ast.Call) and
                ast.unparse(n.func) == 'recovery._run_bounded_maintenance']
            assert isinstance(call.args[0], ast.Dict) and not call.args[0].keys
            call.args[0] = ast.copy_location(ast.Dict(keys=[ast.Constant('B1')],
                values=[ast.Constant(None)]), call.args[0])
            ast.fix_missing_locations(tree)
            if node.config.getoption('assertmode') == 'rewrite':
                from _pytest.assertion.rewrite import rewrite_asserts
                data = Path(code0.co_filename).read_bytes()
                rewrite_asserts(tree, data, code0.co_filename, node.config)
            changed = cls.nested(compile(tree, code0.co_filename, 'exec', dont_inherit=True), cls.name)
            nested = { (c.co_name, c.co_firstlineno): c for c in code0.co_consts if isinstance(c, types.CodeType) }
            changed = changed.replace(co_consts=tuple(nested.get((c.co_name, c.co_firstlineno), c)
                if isinstance(c, types.CodeType) else c for c in changed.co_consts))
            assert all(any(c is v for c in changed.co_consts) for v in nested.values())
            assert changed.co_code != code0.co_code
        else:
            name = {'clock1000': '<lambda>', 'operation': '<lambda>'}.get(mode, mode)
            candidates = [c for c in code0.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
            if name == '<lambda>':
                role = roles['clock' if mode == 'clock1000' else 'operation']
                selected, = [c for c in candidates if cls.code_projection(c) == cls.code_projection(role)]
            else: selected, = candidates
            if mode == 'clock1000':
                assert selected.co_consts == (100,)
                mutated = selected.replace(co_consts=(1000,))
                assert mutated.co_code == selected.co_code
                assert tuple(mutated.co_positions()) == tuple(selected.co_positions())
                assert mutated.co_freevars == selected.co_freevars
            else:
                # Inert extra typed constant changes code authority while keeping
                # the actual original capability behavior and FD/hooks intact.
                mutated = selected.replace(co_consts=selected.co_consts+('B1-pre-capture',))
            changed = code0.replace(co_consts=tuple(mutated if c is selected else c for c in code0.co_consts))
            assert any(c is mutated for c in changed.co_consts)
            assert not any(c is selected for c in changed.co_consts)
        function.__code__ = changed
        assert node.obj is function and function.__code__ is changed and changed is not code0
        assert cls.code_projection(changed) != cls.code_projection(code0)
        with pytest.raises(ValueError, match='^B changed collected target code$'):
            cls.target_source(function, node)
        evidence('pre-capture-source-readback')
'''
    assert source.count(anchor) == 1
    source = source.replace(anchor, setup+'\n'+anchor)
    old = """            assert w.eligible and w.valid
            evidence('admission-baseline')
            if mode in admission:"""
    new = """            assert a[0].obj is function and a[3].f_code is changed
            assert a[3].f_globals is function.__globals__
            local = a[3].f_locals
            assert os.write is local['observed'] and recovery.time.monotonic.__code__ in changed.co_consts
            assert local['write'] is a[5] and os.waitpid is a[7]
            assert cls.identity(a[8])[0] != cls.identity(a[9])[0]
            assert w.eligible and not w.valid
            evidence('admission-baseline')
            evidence('admission-mutated-rejected')
            return
            if mode in admission:"""
    assert source.count(old) == 1
    return source.replace(old, new)


@pytest.mark.parametrize('control', B_CONTROL_B1, ids=lambda c: 'B-C-'+c)
def test_B_precapture_same_collected_function(tmp_path, control):
    """Valid collected invocation first, then same-object pre-capture refusal.

    Both processes must complete original assertions. The baseline is an
    independent invocation, not an invalid constructor relabelled as positive.
    """
    import subprocess
    (tmp_path/'baseline').mkdir()
    test_B_directed_isolated_seams(tmp_path/'baseline', 'collected-control')
    source = _B_precapture_program(control).replace('CONTROL', repr(control))
    source = source.replace('ROOT', repr(str(tmp_path/'mutated-proof'))).replace('NODE', repr(B_CONTROL_NODE))
    prefix = 'import sys; sys.path[:0] = '+repr([str(Path(harness.__file__).parent),
        str(Path(harness.__file__).parent.parent/'diagnostics'/'p-one-node')])+'\n'
    result = subprocess.run([sys.executable, '-B', '-c', prefix+source],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr


B_CONTROL_AUTHORITY = ('wrong-source', 'wrong-digest', 'filename', 'qualname',
    'flags', 'typed-constant', 'lambda-roles')


@pytest.mark.parametrize('control', B_CONTROL_AUTHORITY, ids=lambda c: 'B-A-'+c)
def test_B_independent_authority_source_predicates(monkeypatch, control):
    """Baseline-first exact source predicates; no child/transport/latch credit."""
    import ast, types
    cls = harness.BlockedPipeWitness_B
    path = Path(harness.__file__).parent/'service'/'test_recovery.py'
    data = path.read_bytes()
    module = compile(data, str(path), 'exec', dont_inherit=True)
    code = cls.nested(module, cls.name)
    # No module execution or product import: a function is only a code carrier.
    function = types.FunctionType(code, {})
    monkeypatch.setattr(function, '__code__', code)  # Register restoration first.
    config = SimpleNamespace(getoption=lambda name: 'plain' if name == 'assertmode' else None)
    node = SimpleNamespace(config=config)
    _, original, roles, _ = cls.target_source(function, node)
    assert cls.code_projection(original) == cls.code_projection(code)
    roles = {name: next(c for c in code.co_consts if isinstance(c, types.CodeType) and
        cls.code_projection(c) == cls.code_projection(role)) for name, role in roles.items()}
    assert roles['clock'] is not roles['operation']
    assert cls.code_projection(roles['clock']) != cls.code_projection(roles['operation'])
    if control == 'wrong-source':
        function.__code__ = code.replace(co_filename=str(path.parent/'foreign.py'))
        with pytest.raises(ValueError, match='^B wrong target source$'): cls.target_source(function, node)
    elif control == 'wrong-digest':
        read = Path.read_bytes
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'read_bytes', lambda self: data+b'\n' if self == path else read(self))
            with pytest.raises(ValueError, match='^B changed target source$'): cls.target_source(function, node)
        assert path.read_bytes() == data
    else:
        if control == 'filename': changed = code.replace(co_filename=str(path.parent/'..'/'service'/'test_recovery.py'))
        elif control == 'qualname': changed = code.replace(co_qualname=code.co_qualname+'.foreign')
        elif control == 'flags': changed = code.replace(co_flags=code.co_flags ^ 0x1000000)
        elif control == 'typed-constant':
            clock = roles['clock']; assert clock.co_consts == (100,)
            changed_clock = clock.replace(co_consts=(100.0,))
            assert changed_clock.co_consts == clock.co_consts
            assert cls.code_projection(changed_clock) != cls.code_projection(clock)
            changed = code.replace(co_consts=tuple(changed_clock if c is clock else c for c in code.co_consts))
        else:
            assert control == 'lambda-roles'
            changed = code.replace(co_consts=tuple(roles['operation'] if c is roles['clock'] else
                roles['clock'] if c is roles['operation'] else c for c in code.co_consts))
        assert cls.code_projection(changed) != cls.code_projection(code)
        function.__code__ = changed
        assert function.__code__ is changed
        with pytest.raises(ValueError, match='^B changed collected target code$'): cls.target_source(function, node)
    function.__code__ = code
    assert cls.code_projection(cls.target_source(function, node)[1]) == cls.code_projection(code)


def test_B_overhead_extracted_HOST_contract():
    """Separate future test entry; only extracted/synthetic stdlib HOST credit.

    Does not join or alter B100. Complete guard setup, fixture undo, rewritten
    collection and actual trace restoration still require their existing gates.
    """
    import subprocess, sys
    path = Path(harness.__file__).resolve().parents[1]/'diagnostics/b-controls/HOST-overhead.py'
    completed = subprocess.run([sys.executable, str(path)], capture_output=True,
        text=True, timeout=30, check=False)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert completed.stdout == ('HOST-overhead PASS 32 extracted/synthetic checks; '
        'full setup/fixture/trace restoration PENDING\n')
    assert completed.stderr == ''
