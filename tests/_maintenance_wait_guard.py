"""Prospective maintenance-child evidence guard; never reaps or signals."""
import os
import errno
import stat
import json
import signal
import sys
from contextlib import contextmanager

import pytest
from pathlib import Path
from types import SimpleNamespace
import atexit
import subprocess
import tempfile
import uuid

_RAW_WRITE = os.write


class MaintenanceWaitGuard:
    """Only terminal waits of PIDs enrolled during an owned maintenance call."""
    def __init__(self, node, session):
        self.node, self.session = node, session
        self.owned = set()
        self.failures = []
        self.receipts = []
        self.directory = None
        self.sequence = 0

    def event(self, kind, **data):
        if self.directory is not None:
            self.sequence += 1
            if self.sequence > 4096:
                raise RuntimeError("test guard metadata bound")
            atomic(self.directory / f"{os.getpid()}-{self.sequence}.json", dict(kind=kind, owner=os.getpid(), **data))

    def fail(self, message):
        message = f"{self.node.nodeid}: {message}"
        self.failures.append(message)
        self.session.shouldstop = message
        self.event("failure", message=message[:1024])

    def observe(self, result, *, witness=b'', expected_signal=None, expected_exit=None):
        pid, status = result
        if pid not in self.owned or not (os.WIFEXITED(status) or os.WIFSIGNALED(status)):
            return result
        self.owned.remove(pid)
        self.receipts.append((pid, status))
        log = os.environ.get('PAB_MAINTENANCE_WAIT_LOG')
        if log:
            record = json.dumps(dict(node=self.node.nodeid, owner=os.getpid(), pid=pid, status=status,
                phase=witness.decode('ascii', errors='replace')[:256]))+'\n'
            fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try: _RAW_WRITE(fd, record.encode())
            finally: os.close(fd)
        error = None
        if os.WIFSIGNALED(status):
            if os.WTERMSIG(status) != expected_signal:
                error = 'unexpected signal '+str(os.WTERMSIG(status))
        elif os.WEXITSTATUS(status) == 74:
            if b'R:modeled-private-rehearsal-completion-seam;' in witness and _MODELED_REHEARSAL_FINAL not in witness:
                error = 'modeled rehearsal exception replaced before final exit'
            elif b'V' not in witness and b'J:prewrite-journal-nofollow-ELOOP;' not in witness and _MODELED_REHEARSAL_FINAL not in witness and expected_exit != 74:
                error = 'exit 74 without semantic ValueError phase witness'
        elif os.WEXITSTATUS(status) == 73:
            if b'D' not in witness:
                error = 'exit 73 without declared death phase witness'
        elif os.WEXITSTATUS(status) != 0:
            error = 'unexpected exit '+str(os.WEXITSTATUS(status))
        self.event("terminal", pid=pid, status=status, accepted=error is None,
            phase=witness.decode("ascii", errors="replace")[:256])
        if error:
            message = f'{self.node.nodeid}: maintenance terminal wait pid={pid} status={status}: {error}'
            self.fail(message)
        return result


# Exact test/phase declarations, not a signal-wide exemption. SIGSEGV is never expected.
def _expected_cleanup_signal(node, cleanup):
    name = node.originalname
    params = getattr(getattr(node, 'callspec', None), 'params', {})
    if name == 'test_t1_actual_pipe_transport':
        mode = params.get('mode')
        if mode == 'signal' or (cleanup and mode in {
            'EPIPE', 'zero-write', 'extra', 'second', 'partial', 'empty', 'post-EOF-alive',
            'never-EOF', 'slow', 'exact-T-alive'}):
            return signal.SIGKILL
    if cleanup and name in {
        'test_t1_real_blocked_pipe_backpressure_deadline',
        'test_t1_actual_pipe_overflow_precedes_decoder',
        'test_t1_close_ownership_cleanup', 'test_t1_reap_interruption_ownership',
        'test_t1_cleanup_missing_process_receipts',
    }:
        return signal.SIGKILL
    return None


_DEATH_PHASES = {
    'test_lotk_actual_death_fresh_owned_resume.<locals>.operation.<locals>.die',
    'test_lotp_production_intent_actual_death_fresh_resume.<locals>.die.<locals>.fault',
    'test_lotp_single_immutable_plan_proof_first_and_preaccept_exit.<locals>.die',
    '_kpre_accept.<locals>.die.<locals>.fault',
    '_kpre_attempt.<locals>.die.<locals>.fault',
    '_kfifth_successor.<locals>.die.<locals>.fault',
}


_HELPER_DEATH_NODES = {
    'test_kpre_actual_death_owned_fresh_resume',
    'test_kpre_acceptance_barriers_journal',
    'test_kpre_recapture_actual_death_fresh_retry',
    'test_kpre_no_attempt_closed_refusal_twice',
    'test_kpre_capture_failures_burn_without_outputs',
    'test_kpre_history_and_observational_replay',
    'test_kpre_predecessor_damage_no_capture',
    'test_kpre_after_attempt_never_recaptures',
    'test_kpre_deleted_bundle_sql_is_not_no_attempt',
    'test_kpre_recapture_connections_at_actual_use_and_close',
    'test_kpre_no_attempt_both_busy_locks',
    'test_kpre_additional_closed_inputs',
    'test_kpre_pending_replacement_temporary_quarantines',
    'test_kpre_each_no_attempt_window_rechecks',
    'test_kpre_history_identity_collision_refuses_before_capture',
    'test_kfifth_actual_death_monitor_refuses_before_writers',
    'test_kfifth_reached_actual_death_prefix',
    'test_kfifth_full_current_source_before_writers',
    'test_kfifth_production_history_actual_prefix',
    'test_kfifth_every_completed_descriptor_both_directions',
    'test_kfifth_completed_current_changes_observational',
    'test_kfifth_attempted_output_quarantines_before_writers',
    'test_kfifth_fourth_query_cannot_bypass_unfinished_finish',
    'test_kfifth_production_owner_changes_before_writers',
    'test_kfifth_fourth_query_missing_certificate_cannot_write',
    'test_kfifth_production_completed_replay_after_valid_hello',
    'test_t1_nonempty_B_C_actual_prefix_history',
    'test_t1_nonempty_C_current_source_before_any_writer',
    'test_t1_additional_successor_refusals',
}


_BACKPRESSURE_PHASES = {
    b'T:backpressure-changed_write-BrokenPipeError-EPIPE;',
    b'T:backpressure-changed_write-BlockingIOError-EAGAIN;',
}


def backpressure_phase(node, exc):
    """Attest only the unchanged test-owned raw nonblocking write loop."""
    if (node.originalname != 'test_t1_actual_pipe_transport' or
            not node.nodeid.endswith('::test_t1_actual_pipe_transport[backpressure]') or
            getattr(getattr(node, 'callspec', None), 'params', {}).get('mode') != 'backpressure'):
        return b''
    classes = {(BrokenPipeError, errno.EPIPE): b'BrokenPipeError-EPIPE',
        (BlockingIOError, errno.EAGAIN): b'BlockingIOError-EAGAIN'}
    suffix = classes.get((type(exc), getattr(exc, 'errno', None)))
    if suffix is None: return b''
    # Resolve the actual collected source function, not a filename/name assertion.
    module = sys.modules.get('test_recovery')
    function = getattr(module, 'test_t1_actual_pipe_transport', None)
    if function is None: return b''
    codes = [c for c in function.__code__.co_consts
        if isinstance(c, type(function.__code__)) and c.co_name == 'changed_write']
    trace = exc.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if (len(codes) == 1 and frame.f_code is codes[0] and
                frame.f_locals.get('mode') == 'backpressure'):
            import linecache
            if linecache.getline(frame.f_code.co_filename, trace.tb_lineno).strip() == "while True:write(fd,b'x'*65536)":
                return b'T:backpressure-changed_write-' + suffix + b';'
        trace = trace.tb_next
    return b''


_MODELED_REHEARSAL_FINAL = b'R:modeled-private-rehearsal-completion-final;'


class ModeledRehearsalCompletionWitness:
    """Test-owned injection attestation; a seam token alone never admits exit74."""
    def __init__(self, node, roots, rehearsal, transaction):
        self.node, self.roots = node, roots
        self.rehearsal, self.transaction = rehearsal, transaction
        self.token = None

    def fail_code(self, installed):
        node = self.node
        module = sys.modules.get('test_recovery')
        function = getattr(module, 'test_lotp_late_refusal_retains_unaccepted_plan', None)
        if (node.originalname != 'test_lotp_late_refusal_retains_unaccepted_plan' or
                node.nodeid != 'tests/service/test_recovery.py::test_lotp_late_refusal_retains_unaccepted_plan[rehearsal]' or
                getattr(getattr(node, 'callspec', None), 'params', {}).get('failure') != 'rehearsal' or
                function is None or getattr(node, 'obj', function) is not function):
            return None
        codes = [c for c in function.__code__.co_consts
            if isinstance(c, type(function.__code__)) and
            'modeled private capacity failure' in c.co_consts]
        if len(codes) != 1 or getattr(installed, '__code__', None) is not codes[0]:
            return None
        return codes[0]

    def at_seam(self, exc, installed, wrapper_code, caller):
        import sqlite3
        self.token = None
        code = self.fail_code(installed)
        if (code is None or type(exc) is not sqlite3.OperationalError or
                exc.args != ('modeled private capacity failure',)):
            return False
        lambdas = [c for c in self.rehearsal.__code__.co_consts
            if isinstance(c, type(code)) and '_write_core_completion' in c.co_names]
        if len(lambdas) != 1: return False
        frames = []
        frame = caller
        for expected in (lambdas[0], self.transaction.__code__, self.rehearsal.__code__):
            if frame is None or frame.f_code is not expected: return False
            frames.append(frame)
            frame = frame.f_back
        local, transaction, rehearsal = [f.f_locals for f in frames]
        db, private, intent = local.get('db'), local.get('private'), local.get('intent')
        session = rehearsal.get('session')
        if (session is None or session.roots != self.roots or session.db is db or type(db) is not sqlite3.Connection or
                getattr(rehearsal.get('frozen'), '_session', None) is not session or
                transaction.get('db') is not db or transaction.get('point') != 'core-complete' or
                rehearsal.get('db') is not db or rehearsal.get('private') is not private or
                rehearsal.get('intent') is not intent or not isinstance(private, Path) or
                private != Path(rehearsal.get('temporary', '')) or
                private == self.roots['control'] / 'core'):
            return False
        trace = exc.__traceback__
        if (trace is None or trace.tb_frame.f_code is not wrapper_code or
                trace.tb_next is None or trace.tb_next.tb_frame.f_code is not code or
                trace.tb_next.tb_next is not None or
                trace.tb_next.tb_frame.f_locals.get('args') != (db, private, intent)):
            return False
        info = private.lstat()
        database = private / 'core.sqlite'
        file_info = database.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or
                not stat.S_ISREG(file_info.st_mode) or file_info.st_uid != os.getuid() or
                file_info.st_nlink != 1 or
                db.execute('PRAGMA database_list').fetchall() != [(0, 'main', str(database))]):
            return False
        # Only code objects and the exact exception survive cleanup. No exception
        # payload, DB contents or paths are serialized into the witness pipe.
        self.token = (exc, (self.rehearsal.__code__, self.transaction.__code__,
            lambdas[0], wrapper_code, code))
        return True

    def at_exit(self, exc):
        if self.token is None or exc is not self.token[0]: return b''
        import sqlite3
        if type(exc) is not sqlite3.OperationalError or exc.args != ('modeled private capacity failure',):
            return b''
        route = []
        trace = exc.__traceback__
        while trace is not None:
            route.append(trace.tb_frame.f_code)
            trace = trace.tb_next
        expected = self.token[1]
        if len(route) < len(expected) or tuple(route[-len(expected):]) != expected:
            return b''
        return _MODELED_REHEARSAL_FINAL


def install(request, monkeypatch, guard):
    from service import recovery
    run, fork, wait = recovery._run_bounded_maintenance, os.fork, os.waitpid
    rehearsal0, transaction0 = recovery._rehearse_preparation, recovery._prefix_transaction
    pipe, read, write, close, exit0 = os.pipe, os.read, os.write, os.close, os._exit
    state = dict(active=False, read=None, write=None, witness=b'', enrolling=False)

    def tracked_fork():
        enrolling = state['active'] and state['enrolling']
        state['enrolling'] = False
        pid = fork()
        if enrolling and pid:
            guard.owned.add(pid)
            guard.event('enroll', pid=pid)
        return pid

    def tracked_wait(pid, flags):
        result = wait(pid, flags)
        if result[0] in guard.owned:
            try: state['witness'] = (state['witness'] + read(state['read'], 256))[:256]
            except BlockingIOError: pass
            frame = sys._getframe(1)
            cleanup = False
            while frame is not None:
                if frame.f_code.co_name == 'reap_owned': cleanup = True; break
                frame = frame.f_back
            # Backpressure cleanup must attest the actual parent refusal, not
            # merely the node name or presence of a cleanup stack frame.
            cleanup_phase = b''
            if cleanup and request.node.originalname == 'test_t1_actual_pipe_transport' and getattr(
                    getattr(request.node, 'callspec', None), 'params', {}).get('mode') == 'backpressure':
                parent = frame.f_back
                while parent is not None:
                    if parent.f_code is run.__code__:
                        original = parent.f_locals.get('original')
                        raw = parent.f_locals.get('raw')
                        if type(original) is ValueError and type(raw) is bytearray:
                            if original.args == ('b2_maintenance_result_budget',) and len(raw) == recovery.MAINTENANCE_RESULT_LIMIT + 1:
                                cleanup_phase = b'C:backpressure-parent-T+1-ValueError-budget;'
                            elif original.args == ('b2_maintenance_timeout',) and parent.f_locals.get('remaining', 1) <= 0:
                                cleanup_phase = b'C:backpressure-parent-deadline-ValueError-timeout;'
                        break
                    parent = parent.f_back
                if cleanup_phase: state['witness'] = (state['witness'] + cleanup_phase)[:256]
            expected_exit = 74 if (any(p in state['witness'] for p in _BACKPRESSURE_PHASES) or
                any(p in state['witness'] for p in (b'T:install-setlimit;', b'T:transport;', b'T:child-read-close;'))) else None
            guard.observe(result, witness=state['witness'],
                expected_signal=signal.SIGKILL if cleanup_phase else _expected_cleanup_signal(request.node, cleanup), expected_exit=expected_exit)
        return result

    def guarded_run(roots, scope, operation, **kwargs):
        if guard.failures:
            pytest.exit(guard.failures[0], returncode=2)
        r, w = pipe()
        os.set_blocking(r, False)
        state.update(active=True, read=r, write=w, witness=b'', enrolling=True)
        sent = set()
        completion = recovery._write_core_completion
        modeled = ModeledRehearsalCompletionWitness(request.node, roots,
            rehearsal0, transaction0)
        def observed_completion(*args):
            try: return completion(*args)
            except BaseException as exc:
                if modeled.at_seam(exc, completion, observed_completion.__code__, sys._getframe(1)):
                    mark(b'R:modeled-private-rehearsal-completion-seam;')
                raise
        def mark(data):
            if data not in sent:
                sent.add(data)
                write(w, data)
        # This witness is emitted only by the actual protected read, before the
        # original writer. No OSError from the enclosing session is accepted.
        open0 = os.open
        def journal_open(path, flags, *args, **kw):
            params = getattr(getattr(request.node, 'callspec', None), 'params', {})
            eligible = (request.node.originalname == 'test_lotp_early_refusals_precede_original_writer'
                and request.node.nodeid.endswith('::test_lotp_early_refusals_precede_original_writer[journal-symlink]')
                and params.get('failure') == 'journal-symlink'
                and path == roots['receipts'] / 'journal.json'
                and flags == os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            before = None
            if eligible:
                route = []
                frame = sys._getframe(1)
                while frame is not None and len(route) < 5:
                    route.append(frame.f_code)
                    frame = frame.f_back
                del frame
                prewrite_codes = [c for c in recovery._prepare_abandon.__code__.co_consts
                    if isinstance(c, type(recovery._prepare_abandon.__code__)) and c.co_name == 'prewrite']
                eligible = len(prewrite_codes) == 1 and route == [
                    recovery.read_file.__code__, recovery._observe_b2_journal.__code__,
                    recovery._preparation_prestate.__code__, prewrite_codes[0], session.__wrapped__.__code__]
                if eligible:
                    try: before = os.lstat(path)
                    except OSError: pass
            try:
                return open0(path, flags, *args, **kw)
            except OSError as exc:
                if eligible and type(exc) is OSError and exc.errno == errno.ELOOP:
                    after = os.lstat(path)
                    if (before is not None and stat.S_ISLNK(before.st_mode) and
                            stat.S_ISLNK(after.st_mode) and
                            (before.st_dev, before.st_ino, before.st_ctime_ns) ==
                            (after.st_dev, after.st_ino, after.st_ctime_ns)):
                        mark(b'J:prewrite-journal-nofollow-ELOOP;')
                raise
        session = recovery._maintenance_session
        install = recovery._install_maintenance_limit
        frame_result = recovery._maintenance_result_frame
        def semantic_phase(function):
            def observed(*args, **kw):
                try: return function(*args, **kw)
                except ValueError:
                    mark(b'V')
                    raise
                except OSError:
                    params = getattr(getattr(request.node, 'callspec', None), 'params', {})
                    if (function is install and
                            request.node.originalname == 'test_lotk_refuses_before_original_writable_open' and
                            params.get('failure') == 'setlimit'):
                        mark(b'T:install-setlimit;')
                    raise
            return observed
        @contextmanager
        def observed_session(*args, **kw):
            try:
                with session(*args, **kw) as value: yield value
            except ValueError:
                mark(b'V:session;')  # bounded test-only phase; no exception payload
                raise
        def observed_operation(value):
            try: return operation(value)
            except ValueError:
                mark(b'V:operation;')
                raise
        def observed_exit(code):
            if code == 74:
                params = getattr(getattr(request.node, 'callspec', None), 'params', {})
                exc = sys.exception()
                mode = params.get('mode')
                final = modeled.at_exit(exc)
                if final: mark(final)
                phase = backpressure_phase(request.node, exc)
                if phase: mark(phase)
                if request.node.originalname == 'test_t1_actual_pipe_transport':
                    types = {'EPIPE': BrokenPipeError, 'zero-write': OSError,
                        'extra': BrokenPipeError, 'second': BrokenPipeError}
                    if ((mode in ('nonzero', 'slow') and exc is None) or
                            (mode in types and isinstance(exc, types[mode]))):
                        mark(b'T:transport;')
                if (request.node.originalname == 'test_t1_close_ownership_cleanup' and
                        params.get('site') == 'child-read' and
                        isinstance(exc, (InterruptedError, KeyboardInterrupt))):
                    mark(b'T:child-read-close;')
            if code == 73:
                frame = sys._getframe(1)
                if (frame.f_code.co_filename.endswith('/tests/service/test_recovery.py') and
                        frame.f_code.co_qualname in _DEATH_PHASES and
                        (frame.f_code.co_qualname.split('.<locals>')[0] == request.node.originalname or
                         request.node.originalname in _HELPER_DEATH_NODES) and
                        (frame.f_locals.get('where') == frame.f_locals.get('point') or
                         frame.f_code.co_qualname == 'test_lotp_single_immutable_plan_proof_first_and_preaccept_exit.<locals>.die')):
                    phase = frame.f_locals.get('where', 'preaccept')
                    mark(b'D:' + str(phase).encode('ascii')[:128])
                elif (request.node.originalname == 'test_kpre_recapture_actual_death_fresh_retry' and
                        frame.f_code.co_filename == '<string>' and
                        frame.f_code.co_qualname == 'resume.<locals>.fault' and
                        frame.f_locals.get('p') == request.node.callspec.params.get('point') and
                        frame.f_locals.get('p') in {'recapture-before', 'recapture-after',
                            'recapture-hash', 'recapture-oracle', 'recapture-verified'}):
                    mark(b'D:' + frame.f_locals['p'].encode('ascii'))
            exit0(code)
        try:
            with monkeypatch.context() as patch:
                if modeled.fail_code(completion) is not None:
                    patch.setattr(recovery, '_write_core_completion', observed_completion)
                patch.setattr(recovery, '_maintenance_session', observed_session)
                patch.setattr(recovery, '_install_maintenance_limit', semantic_phase(install))
                patch.setattr(recovery, '_maintenance_result_frame', semantic_phase(frame_result))
                patch.setattr(os, '_exit', observed_exit)
                patch.setattr(os, 'open', journal_open)
                return run(roots, scope, observed_operation, **kwargs)
        finally:
            state['active'] = False
            close(r); close(w)
            if guard.owned:
                guard.fail('missing terminal maintenance receipt')
            if guard.directory is not None and guard.failures:
                raise SystemExit(86)
    monkeypatch.setattr(os, 'fork', tracked_fork)
    monkeypatch.setattr(os, 'waitpid', tracked_wait)
    monkeypatch.setattr(recovery, '_run_bounded_maintenance', guarded_run)
    return guard



_CURRENT = None
_ROOT = None


def atomic(path, value):
    """One bounded metadata record, published atomically in a private test directory."""
    raw = json.dumps(value).encode()
    if len(raw) > 4096:
        raise RuntimeError('test guard metadata bound')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        stream.write(raw)
    temporary.replace(path)


@contextmanager
def activate(guard, root):
    global _CURRENT, _ROOT
    previous = _CURRENT, _ROOT
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _CURRENT, _ROOT = guard, root
    try: yield
    finally: _CURRENT, _ROOT = previous


def metadata(path):
    """Private bounded regular-file read; bound checked before parsing or reading."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
                info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size > 4096):
            raise ValueError('helper record bound or identity')
        raw = os.read(fd, 4097)
        if len(raw) != info.st_size or len(raw) > 4096:
            raise ValueError('helper record bound or changed')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError('duplicate metadata key')
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=unique)
        if type(value) is not dict: raise ValueError('helper record schema')
        return value
    finally:
        os.close(fd)


def valid_pid(value):
    return type(value) is int and 0 < value <= 2**31 - 1


def identity(value):
    if (type(value) is not dict or set(value) != {'pid', 'node', 'token'} or not valid_pid(value['pid']) or
            type(value['node']) is not str or not 0 < len(value['node']) <= 1024 or
            type(value['token']) is not str or len(value['token']) != 32 or
            any(c not in '0123456789abcdef' for c in value['token'])):
        raise ValueError('invalid helper acknowledgement')


def aggregate(directory, expected, *, depth=0):
    """Bind each private directory to its launcher, including nested owners."""
    identity(expected)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('helper directory identity')
    if depth > 16: raise ValueError('nested helper depth bound')
    ack, done = metadata(directory / 'ack.json'), metadata(directory / 'done.json')
    identity(ack)
    identity(done)
    if ack != expected or done != expected:
        raise ValueError('helper launch/acknowledgement/completion mismatch')
    events = list(directory.glob('[0-9]*.json'))
    if len(events) > 4096: raise ValueError('helper metadata bound')
    owned, terminal, receipts, failures = set(), set(), [], []
    for path in events:
        value = metadata(path)
        kind = value.get('kind')
        if not valid_pid(value.get('owner')) or value['owner'] != ack['pid']:
            raise ValueError('foreign helper event owner')
        schemas = {'enroll': {'kind', 'owner', 'pid'},
            'terminal': {'kind', 'owner', 'pid', 'status', 'accepted', 'phase'},
            'failure': {'kind', 'owner', 'message'}}
        if kind not in schemas or set(value) != schemas[kind]:
            raise ValueError('helper event schema')
        if kind in ('enroll', 'terminal') and not valid_pid(value['pid']):
            raise ValueError('helper child PID')
        key = value.get('pid')
        if kind == 'enroll':
            if key in owned: raise ValueError('duplicate enrollment')
            owned.add(key)
        elif kind == 'terminal':
            if (type(value['status']) is not int or not 0 <= value['status'] <= 65535 or
                    not (os.WIFEXITED(value['status']) or os.WIFSIGNALED(value['status'])) or
                    type(value['accepted']) is not bool or type(value['phase']) is not str or
                    len(value['phase']) > 256):
                raise ValueError('helper terminal schema')
            if key in terminal: raise ValueError('duplicate terminal receipt')
            terminal.add(key)
            receipts.append((key, value['status']))
            if value['accepted'] is not True:
                failures.append('unaccepted terminal status=' + str(value['status']))
        else:
            if type(value['message']) is not str or len(value['message']) > 1024:
                raise ValueError('helper failure schema')
            failures.append(value['message'])
    if owned != terminal: raise ValueError('orphan or missing terminal receipt')
    if failures: raise ValueError('; '.join(failures))
    nested_root = directory / 'nested'
    if nested_root.is_symlink(): raise ValueError('nested helper directory')
    nested = list(nested_root.glob('helper-*'))
    if len(nested) > 256: raise ValueError('nested helper bound')
    for child in nested:
        if child.is_symlink() or not child.is_dir(): raise ValueError('nested helper directory')
        launch = metadata(child / 'launch.json')
        if set(launch) != {'parent', 'identity'} or launch['parent'] != ack['pid'] or not valid_pid(launch['parent']):
            raise ValueError('nested launch owner')
        identity(launch['identity'])
        if launch['identity']['node'] != ack['node']: raise ValueError('nested launch node')
        receipts.extend(aggregate(child, launch['identity'], depth=depth+1))
    return receipts


def bootstrap(directory, nodeid, originalname, params, token):
    """Called explicitly before a recovery test's fresh python -c body."""
    global _CURRENT, _ROOT
    directory = Path(directory)
    node = SimpleNamespace(nodeid=nodeid, originalname=originalname,
        callspec=SimpleNamespace(params=params))
    guard = MaintenanceWaitGuard(node, SimpleNamespace(shouldstop=False))
    guard.directory = directory
    patch = pytest.MonkeyPatch()
    install(SimpleNamespace(node=node), patch, guard)
    _CURRENT, _ROOT = guard, directory / 'nested'
    _ROOT.mkdir(mode=0o700)
    identity = dict(pid=os.getpid(), node=nodeid, token=token)
    atomic(directory / 'ack.json', identity)
    owner = os.getpid()
    def finish():
        if os.getpid() != owner: return
        if guard.owned: guard.fail('orphan owned maintenance PIDs')
        atomic(directory / 'done.json', identity)
    atexit.register(finish)


def fresh_run(argv, **kwargs):
    """Explicit recovery-test launcher; never changes ambient PYTHONPATH or production."""
    guard = _CURRENT
    if guard is None:
        raise RuntimeError('fresh recovery helper requires active test guard')
    if guard.failures:
        raise RuntimeError(guard.failures[0])
    if not (len(argv) >= 3 and argv[0] == sys.executable and argv[1] == '-c'):
        raise ValueError('test guard requires explicit fresh python -c')
    directory = Path(tempfile.mkdtemp(prefix='helper-', dir=_ROOT))
    params = getattr(getattr(guard.node, 'callspec', None), 'params', {})
    policy = {k: v for k, v in params.items() if k in ('mode', 'site', 'point', 'failure') and isinstance(v, str)}
    token = uuid.uuid4().hex
    prefix = ('import sys; sys.path.insert(0, ' + repr(str(Path(__file__).parent)) + '); '
        'from _maintenance_wait_guard import bootstrap; bootstrap(' +
        repr(str(directory)) + ', ' + repr(guard.node.nodeid) + ', ' +
        repr(guard.node.originalname) + ', ' + repr(policy) + ', ' + repr(token) + ')\n')
    expected = None
    try:
        # Mirror subprocess.run's input/capture/check/timeout behavior while retaining
        # the PID from the actual launch. No waitpid hook or unrelated interception.
        input0 = kwargs.pop('input', None)
        capture = kwargs.pop('capture_output', False)
        timeout = kwargs.pop('timeout', None)
        check = kwargs.pop('check', False)
        if input0 is not None:
            if kwargs.get('stdin') is not None: raise ValueError('stdin and input arguments')
            kwargs['stdin'] = subprocess.PIPE
        if capture:
            if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
                raise ValueError('capture_output and stdout/stderr arguments')
            kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = [*argv[:2], prefix + argv[2], *argv[3:]]
        with subprocess.Popen(command, **kwargs) as process:
            expected = dict(pid=process.pid, node=guard.node.nodeid, token=token)
            atomic(directory / 'launch.json', dict(parent=os.getpid(), identity=expected))
            try:
                stdout, stderr = process.communicate(input0, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                stdout, stderr = process.communicate()
                exc.stdout, exc.stderr = stdout, stderr
                raise
            except BaseException:
                process.kill()
                raise
            result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if check: result.check_returncode()
            return result
    finally:
        try:
            if expected is None: raise ValueError('helper launch absent')
            receipts = aggregate(directory, expected)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            guard.fail('fresh helper receipt failure: ' + str(exc)[:1024])
        else:
            guard.receipts.extend(receipts)
