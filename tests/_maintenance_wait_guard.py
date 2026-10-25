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

# Directed survival storage is fixture-owned, never witness-owned.
_B_SURVIVAL = None
_B_GAP_ENABLED = False
_B_GAP_CHILD = False
_B_GAP_CLOCK = None
_B_GAP_PARENT = None


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

    def observe(self, result, *, witness=b'', expected_signal=None, expected_exit=None, armed_P=False, armed_B=False):
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
        if armed_B or b'B:' in witness:
            if armed_P or not BlockedPipeWitness_B.transcript(witness, status):
                error = 'incomplete or invalid armed B contract'
        elif armed_P or b'P:' in witness:
            if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 74 and
                    DurableLostResultWitness_P.transcript_P(witness)):
                error = 'incomplete or invalid armed P contract'
        elif os.WIFSIGNALED(status):
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
                db.execute('PRAGMA database_list').fetchall() != [(0, 'main', str(database.resolve()))]):
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



# Reserved grouped convention: A/H/P/S, then armed/seam/final. P emits 12 bytes.
# The P invocation can additionally emit V, V:session;, V:operation; (23 bytes).
# Future H/S triples of the same four-byte format plus P and these legacy phases
# total 59 bytes; even all existing R/J/T/C alternatives together stay <=256
# only under their existing mutually exclusive node declarations (see handoff).
def set_frame_local(frame, name, value):
    """Apply an explicit predicate mutation on supported CPython test runtimes."""
    frame.f_locals[name] = value
    if sys.version_info < (3, 13):
        # Before PEP 667, f_locals is a snapshot; writing it alone silently
        # leaves the tested predicate unchanged. Synchronize the labelled probe.
        import ctypes
        sync = ctypes.pythonapi.PyFrame_LocalsToFast
        sync.argtypes = (ctypes.py_object, ctypes.c_int)
        sync.restype = None
        sync(frame, 0)


class DurableLostResultWitness_P:
    armed, seam, final = b'P:a;', b'P:s;', b'P:f;'

    def __init__(self, node, installed, runner, continuation, prepare_code, witness_fd, operation_code, *, replay=False):
        self.runner, self.continuation = runner, continuation
        self.witness_fd = witness_fd
        self.exception = self.route = self.completed = self.runner_frame = None
        self.event_trace_P = None
        self.result_identity = self.sentinel_P = None
        self.valid = True
        module = sys.modules.get('test_recovery')
        function = getattr(module, 'test_t1_durable_completion_lost_result_no_false_success', None)
        name = 'test_t1_durable_completion_lost_result_no_false_success'
        # Designation and validation are independent: malformed collected/node/
        # operation identity cannot turn a designated failure into legacy V.
        self.eligible = not replay and (node.originalname == name or
            node.nodeid == 'tests/service/test_recovery.py::' + name or
            getattr(node, 'obj', None) is function and function is not None)
        self.finish = self.broken = None
        if function is not None:
            codes = [c for c in function.__code__.co_consts if isinstance(c, type(function.__code__)) and c.co_name == 'finish_then_lose']
            if len(codes) == 1:
                self.finish = codes[0]
                codes = [c for c in self.finish.co_consts if isinstance(c, type(self.finish)) and c.co_name == 'broken']
                if len(codes) == 1: self.broken = codes[0]
        self.valid = (self.eligible and node.originalname == name and
            node.nodeid == 'tests/service/test_recovery.py::' + name and
            getattr(getattr(node, 'callspec', None), 'params', {}) == {} and
            function is not None and getattr(node, 'obj', None) is function and
            self.finish is not None and self.broken is not None and
            getattr(installed, '__code__', None) is self.finish and
            prepare_code is not None and operation_code is prepare_code)
        # The exact _prepare_abandon lambda is the only allowed runner operation.
        self.prepare_code = prepare_code
        self.write_offsets = self.call_offsets(runner, 'write')
        self.exit_offsets = self.call_offsets(runner, '_exit')
        import dis
        self.raise_offsets = frozenset(i.offset for i in dis.get_instructions(self.broken)
            if i.opname == 'RAISE_VARARGS') if self.broken is not None else frozenset()

    @staticmethod
    def replay_P(operation, installed, original_locked, caller):
        # Positive exclusion for the actual checked _kfifth_fresh subprocess.
        # Compare its actual module-owned code constant to the unchanged source
        # literal, not just a function name/filename or "not preparation".
        if (installed is not original_locked or caller.f_code.co_name != '<module>' or
                caller.f_globals.get('resume') is not operation or
                getattr(operation, '__globals__', None) is not caller.f_globals or
                not any(c is getattr(operation, '__code__', None) for c in caller.f_code.co_consts)):
            return False
        import ast
        try:
            source = Path(__file__).parent / 'service' / 'test_recovery.py'
            tree = ast.parse(source.read_bytes())
            literals = [n.value for n in tree.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == '_KPRE_FRESH' for t in n.targets)]
            if len(literals) != 1: return False
            # fresh_run prepends exactly one bootstrap source line.
            code = compile('\n' + ast.literal_eval(literals[0]), '<string>', 'exec')
            candidates = [c for c in code.co_consts if isinstance(c, type(code)) and c.co_name == 'resume']
            return len(candidates) == 1 and operation.__code__ == candidates[0]
        except (OSError, SyntaxError, ValueError, TypeError):
            return False  # No replay exemption on unavailable/malformed source.

    @staticmethod
    def call_offsets(code, attribute):
        import dis
        instructions = list(dis.get_instructions(code, show_caches=True))
        result = []
        for i, instruction in enumerate(instructions):
            if instruction.opname in ('LOAD_ATTR', 'LOAD_METHOD') and instruction.argval == attribute:
                # The last _exit site is the captured runner's catch-all74;
                # the earlier success _exit(0) is never final authority.
                result = []
                for j in range(i + 1, len(instructions)):
                    if (instructions[j].opname.startswith('CALL') and
                            instructions[j].positions.lineno == instruction.positions.lineno and
                            instructions[j].positions.col_offset == instruction.positions.col_offset):
                        result.append(instructions[j].offset)
                        k = j + 1
                        while k < len(instructions) and instructions[k].opname == 'CACHE':
                            result.append(instructions[k].offset); k += 1
                        break
        return frozenset(result)

    @staticmethod
    def pipe_identity(fd):
        import fcntl
        info = os.fstat(fd)
        return info.st_dev, info.st_ino, info.st_mode, fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE

    def trace_P(self, frame, event, arg):
        if frame.f_code not in (self.runner, self.continuation, self.finish): return None
        if frame.f_code is self.runner and event == 'line':
            local = frame.f_locals
            if local.get('pid') == 0 and self.runner_frame is None:
                self.runner_frame = frame
                fd = local.get('write_fd')
                self.result_identity = local.get('identities', {}).get(fd)
                if (self.result_identity is None or not stat.S_ISFIFO(self.result_identity[2]) or
                        self.result_identity[3] != os.O_WRONLY or
                        self.result_identity == self.pipe_identity(self.witness_fd)):
                    self.valid = False
        if event == 'return' and frame.f_code is self.continuation:
            caller = frame.f_back
            if caller is not None and caller.f_code is self.finish:
                r = caller.f_locals.get('r')
                session = caller.f_locals.get('session')
                local = frame.f_locals
                if (type(arg) is dict and arg.get('status') == 'complete' and r is not None and
                        local.get('self') is session and local.get('prepared') is r._snapshot and
                        r._snapshot._session is session and r._snapshot._used and
                        local.get('intent') == r['intent'] and
                        session.db.execute('SELECT count(*) FROM p1_recoveries').fetchone() == (1,) and
                        session.db.execute('SELECT count(*) FROM p1_recovery_steps').fetchone() == (5,) and
                        session.db.execute('SELECT id FROM p1_recoveries').fetchone() == (r['intent']['id'],) and
                        arg.get('recovery_id') == r['intent']['id']):
                    self.completed = (session, r, r['intent']['id'])
                else: self.valid = False
        if event == 'return' and frame.f_code is self.finish:
            if self.completed is None or arg is not self.completed[1]: self.valid = False
            else:
                path = frame.f_locals.get('tmp_path')
                self.sentinel_P = None if path is None else path / 'durable-completion'
                if self.sentinel_P is None or self.sentinel_P.read_text() != self.completed[2]: self.valid = False
        if event == 'exception' and frame.f_code is self.runner:
            exc = arg[1]
            # CPython 3.11 supplies the traceback in the trace event before it
            # attaches it to the exception. Keep only this exact observed event.
            self.event_trace_P = (exc, frame, arg[2])
            try:
                if self.at_seam_P(exc, frame): self.emit_P(self.seam)
            finally:
                self.event_trace_P = None
        return self.trace_P

    def at_seam_P(self, exc, frame):
        if (not self.valid or self.exception is not None or self.completed is None or
                frame is not self.runner_frame or type(exc) is not BrokenPipeError or
                exc.args != ('modeled lost result pipe',) or exc.errno is not None): return False
        try:
            if self.sentinel_P is None or self.sentinel_P.read_text() != self.completed[2]: return False
        except OSError: return False
        observed = self.event_trace_P
        trace = observed[2] if observed is not None and observed[0] is exc and observed[1] is frame else exc.__traceback__
        if (trace is None or trace.tb_frame is not frame or trace.tb_lasti not in self.write_offsets or
                trace.tb_next is None or trace.tb_next.tb_frame.f_code is not self.broken or
                trace.tb_next.tb_lasti not in self.raise_offsets or
                trace.tb_next.tb_next is not None): return False
        local = frame.f_locals
        fd, raw, offset = local.get('write_fd'), local.get('raw'), local.get('offset')
        args = trace.tb_next.tb_frame.f_locals.get('a')
        session, result, identity = self.completed
        operation = local.get('operation')
        if (getattr(operation, '__code__', None) is not self.prepare_code or
                local.get('result') is not result or local.get('session') is not session or
                result['intent']['id'] != identity or type(raw) is not bytearray or
                not 0 < len(raw) <= 1048576 or type(offset) is not int or not 0 <= offset < len(raw) or
                self.pipe_identity(fd) != self.result_identity or fd == self.witness_fd or
                type(args) is not tuple or len(args) != 2 or args[0] != fd or
                type(args[1]) is not memoryview or args[1].obj is not raw or
                bytes(args[1]) != raw[offset:offset + 65536]): return False
        self.exception = exc
        self.route = ((trace.tb_frame, trace.tb_lasti), (trace.tb_next.tb_frame, trace.tb_next.tb_lasti))
        return True

    def at_exit_P(self, exc, caller):
        if (exc is not self.exception or exc is None or not self.valid or
                type(exc) is not BrokenPipeError or exc.args != ('modeled lost result pipe',) or
                exc.errno is not None or caller is not self.runner_frame or
                caller.f_lasti not in self.exit_offsets): return b''
        route = []
        trace = exc.__traceback__
        while trace is not None:
            route.append((trace.tb_frame, trace.tb_lasti)); trace = trace.tb_next
        return self.final if tuple(route) == self.route else b''

    def clear_P(self):
        self.event_trace_P = None
        self.exception = self.route = self.completed = self.runner_frame = self.sentinel_P = None

    @classmethod
    def transcript_P(cls, witness):
        # Closed grammar: exact ordered family tokens followed/interleaved only
        # by the three existing semantic phases. Unknown/truncated data rejects.
        family = []
        remaining = witness
        tokens = (cls.armed, cls.seam, cls.final, b'V:session;', b'V:operation;', b'V')
        counts = {}
        while remaining:
            token = next((t for t in tokens if remaining.startswith(t)), None)
            if token is None: return False
            counts[token] = counts.get(token, 0) + 1
            if counts[token] != 1: return False
            if token in tokens[:3]: family.append(token)
            remaining = remaining[len(token):]
        return family == list(tokens[:3]) and len(witness) <= 256


class BlockedPipeWitness_B:
    """Invocation-local singleton certificate. Never owns a frame or exception.

    Source positions select statements; executing code/disassembly select sites.
    All event arguments are transient. Only primitives and callable capabilities
    survive a callback. No result pipe reads, probe writes, signals or reaps.
    """
    name = 'test_t1_real_blocked_pipe_backpressure_deadline'
    nodeid = 'tests/service/test_recovery.py::' + name

    @staticmethod
    def closure(function):
        return dict(zip(function.__code__.co_freevars,
            (cell.cell_contents for cell in (function.__closure__ or ()))))

    @staticmethod
    def nested(code, name):
        values = [c for c in code.co_consts if isinstance(c, type(code)) and c.co_name == name]
        if len(values) != 1: raise ValueError('B ambiguous code capability')
        return values[0]

    @staticmethod
    def code_projection(code, _memo=None):
        """Closed stored-code schema, including typed recursive constants.

        No method/introspection inventory, ignored constants or line-only proof.
        The bytecode and exception/position tables also close every outer site.
        """
        import types
        if _memo is None: _memo = {}
        if id(code) in _memo: return _memo[id(code)][1]
        def constant(value):
            kind = type(value)
            if kind is types.CodeType:
                return (kind, BlockedPipeWitness_B.code_projection(value, _memo))
            if kind is tuple:
                return (kind, tuple(constant(v) for v in value))
            if kind is frozenset:
                return (kind, frozenset(constant(v) for v in value))
            if kind not in (type(None), bool, int, float, complex, str, bytes, type(Ellipsis)):
                raise ValueError('B unsupported code constant')
            return (kind, value)
        projection = (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code,
            tuple(constant(v) for v in code.co_consts), code.co_names,
            code.co_varnames, code.co_freevars, code.co_cellvars, code.co_filename,
            code.co_name, code.co_qualname, code.co_firstlineno,
            code.co_linetable, code.co_exceptiontable)
        _memo[id(code)] = (code, projection)
        return projection

    @classmethod
    def target_source(cls, function, node, prepared=None, projection_memo=None):
        """Independent frozen source authority, then actual code identity binding.

        Reproduce pytest's AST transform before projecting the outer code; never
        equate a rewritten item with its plain compile or erase assertion sites.
        Preserve full module compilation context, including compiler globals.
        Neither the target nor any compiled module is executed.
        """
        import ast, dis, hashlib
        source_path = Path(__file__).parent / 'service' / 'test_recovery.py'
        filename = function.__code__.co_filename
        if Path(filename).resolve() != source_path.resolve():
            raise ValueError('B wrong target source')
        data = source_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != '916f1b75613da2eee32cbcb1b7321c51a17014a4d0c69bb2f0ff239cf7082387':
            raise ValueError('B changed target source')
        key = cls.fact_key(function, node, data)
        if prepared is not None:
            if (prepared.get('key') != key or prepared.get('config') is not node.config or
                    prepared.get('execute_code') is not function.__code__):
                raise ValueError('B stale prepared authority')
            if cls.code_projection(function.__code__, projection_memo) != prepared['outer_projection']:
                raise ValueError('B changed collected target code')
            return prepared['target']
        tree = ast.parse(data, filename=filename)
        targets = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == cls.name]
        if len(targets) != 1: raise ValueError('B ambiguous target source')
        target = targets[0]
        plain = compile(data, filename, 'exec', dont_inherit=True)
        original = cls.nested(plain, cls.name)
        # AST parent roles distinguish the two same-named lambda code objects.
        target_nodes = tuple(ast.walk(target))
        clock_call = [n for n in target_nodes if isinstance(n, ast.Call) and
            ast.unparse(n.func) == 'SimpleNamespace']
        operation_call = [n for n in target_nodes if isinstance(n, ast.Call) and
            ast.unparse(n.func) == 'recovery._run_bounded_maintenance']
        if len(clock_call) != 1 or len(operation_call) != 1:
            raise ValueError('B ambiguous lambda parent')
        clocks = [k.value for k in clock_call[0].keywords if k.arg == 'monotonic']
        operations = operation_call[0].args[2:]
        roles = {}
        instructions = tuple(dis.get_instructions(original))
        for role, nodes in (('clock', clocks), ('operation', operations)):
            if len(nodes) != 1 or not isinstance(nodes[0], ast.Lambda):
                raise ValueError('B missing lambda role')
            n = nodes[0]
            span = (n.lineno, n.end_lineno, n.col_offset, n.end_col_offset)
            codes = [i.argval for i in instructions if
                i.opname == 'LOAD_CONST' and isinstance(i.argval, type(original)) and
                i.argval.co_name == '<lambda>' and tuple(i.positions) == span]
            if len(codes) != 1: raise ValueError('B ambiguous lambda code')
            roles[role] = codes[0]
        mode = node.config.getoption('assertmode')
        if mode == 'rewrite':
            from _pytest.assertion.rewrite import rewrite_asserts
            rewritten = ast.parse(data, filename=filename)
            rewrite_asserts(rewritten, data, filename, node.config)
            expected = cls.nested(compile(rewritten, filename, 'exec', dont_inherit=True), cls.name)
        elif mode == 'plain':
            expected = original
        else:
            raise ValueError('B unknown assertion mode')
        if cls.code_projection(function.__code__) != cls.code_projection(expected):
            raise ValueError('B changed collected target code')
        return plain, original, roles, tree

    @classmethod
    def fact_key(cls, function, node, data):
        import hashlib
        config = node.config
        mode = config.getoption('assertmode')
        # This is the configuration read by AssertionRewriter. Identity also
        # prevents substituting another config with coincidentally equal values.
        hook = config.getini('enable_assertion_pass_hook') if mode == 'rewrite' else None
        return (data, hashlib.sha256(data).digest(), function.__code__.co_filename,
            str((Path(__file__).parent / 'service' / 'test_recovery.py').resolve()),
            id(config), mode, hook, function.__code__)

    @classmethod
    def prepare(cls, node, run, recovery):
        """Installation-owned facts; never executed code or a process cache."""
        import ast, dis
        facts = {}
        try:
            function = getattr(node, 'obj', None)
            target = cls.target_source(function, node)
            data = (Path(__file__).parent / 'service' / 'test_recovery.py').read_bytes()
            facts.update(key=cls.fact_key(function, node, data), config=node.config,
                execute_code=function.__code__, target=target,
                outer_projection=cls.code_projection(function.__code__))
            runner = run.__code__
            path = Path(runner.co_filename)
            data = path.read_bytes()
            tree = ast.parse(data)
            module = compile(data, runner.co_filename, 'exec')
            reap = cls.nested(runner, 'reap_owned')
            observed = cls.nested(function.__code__, 'observed')
            tables = {code: tuple(dis.get_instructions(code, show_caches=True))
                for code in (runner, reap, observed)}
            # Index each tree once, rather than walking/unparsing it per site.
            indexes = {}
            for source_tree in (tree, target[3]):
                indexes[id(source_tree)] = tuple((ast.unparse(n), n) for n in ast.walk(source_tree)
                    if isinstance(n, (ast.Call, ast.Raise)))
            analysis = (indexes, tables)
            sites = {}
            for name, code, expression, source_tree in (
                ('ready_sites', runner, 'select.select([read_fd], [], [], remaining)', tree),
                ('write_sites', runner, 'os.write(write_fd, memoryview(raw)[offset:offset + 65536])', tree),
                ('observed_sites', observed, 'write(fd, raw)', target[3]),
                ('exit_sites', runner, 'os._exit(74)', tree),
                ('reap_sites', runner, 'reap_owned()', tree),
                ('wait_first', reap, 'os.waitpid(pid, os.WNOHANG)', tree),
                ('wait_last', reap, 'os.waitpid(pid, 0 if killed else os.WNOHANG)', tree),
                ('kill_sites', reap, 'os.kill(pid, signal.SIGKILL)', tree)):
                sites[name] = cls.sites(code, expression, 'CALL', source_tree, analysis)
            sites['raise_child'], sites['raise_parent'], sites['raise_empty'] = cls.timeout_sites(runner, tree, tables[runner])
            kill_end = max(sites['kill_sites'])
            completion = next(i.offset for i in tables[reap] if i.offset > kill_end and i.opname != 'CACHE')
            kill_lines = frozenset(i.positions.lineno for i in tables[reap] if i.offset in sites['kill_sites'])
            projections = {}
            for code in (target[1], *target[2].values()): cls.code_projection(code, projections)
            facts.update(projections=projections, runner=runner, runner_path=path, runner_data=data,
                runner_module=module, frame_code=recovery._maintenance_result_frame.__code__,
                sites=sites, reap=reap, kill_completion=completion, kill_lines=kill_lines)
            return facts
        except BaseException:
            facts.clear()
            raise

    @staticmethod
    def sites(code, expression, opname, tree=None, analysis=None):
        import ast, dis
        tree = ast.parse(Path(code.co_filename).read_bytes()) if tree is None else tree
        instructions = (tuple(dis.get_instructions(code, show_caches=True)) if analysis is None
            else analysis[1][code])
        indexed = ((ast.unparse(n), n) for n in ast.walk(tree)
            if isinstance(n, (ast.Call, ast.Raise))) if analysis is None else analysis[0][id(tree)]
        end = max(i.positions.end_lineno or 0 for i in instructions)
        nodes = [n for text, n in indexed if text == expression and
            code.co_firstlineno <= n.lineno <= end]
        if len(nodes) != 1: raise ValueError('B ambiguous source site')
        node = nodes[0]
        offsets = set()
        for index, instruction in enumerate(instructions):
            p = instruction.positions
            if (instruction.opname.startswith(opname) and p.lineno == node.lineno and
                    p.end_lineno == node.end_lineno and p.col_offset == node.col_offset and
                    p.end_col_offset == node.end_col_offset):
                offsets.add(instruction.offset)
                for cache in instructions[index + 1:]:
                    if cache.opname != 'CACHE': break
                    offsets.add(cache.offset)
        if not offsets: raise ValueError('B unresolved instruction site')
        return frozenset(offsets)

    @staticmethod
    def identity(fd):
        import fcntl
        info = os.fstat(fd)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        return (info.st_dev, info.st_ino, info.st_mode, flags & os.O_ACCMODE), flags

    @staticmethod
    def timeout(exc):
        return (type(exc) is ValueError and type(exc.args) is tuple and len(exc.args) == 1 and
            type(exc.args[0]) is str and exc.args[0] == 'b2_maintenance_timeout')

    def __init__(self, node, run, operation, caller, recovery, writer, fork, wait,
            witness_fd, observer_fd, emit, originals, prepared=None):
        function = getattr(sys.modules.get('test_recovery'), self.name, None)
        self.eligible = (node.originalname == self.name or node.nodeid == self.nodeid or
            function is not None and getattr(node, 'obj', None) is function)
        self.valid = False
        self.phase = 'armed'
        self.child = self.owner = self.runner_id = self.raw_id = self.result_id = 0
        self.offset = self.raw_length = self.timeout_id = self.parent_id = self.framed_id = 0
        self.deadline = None
        self.pipe = self.read_pipe = None
        self.empty_select = self.kill_pending = self.killed = self.parent_checked = False
        self.emit = emit
        self.trace_callback = self.select_B
        self.runner = run.__code__
        self.operation = operation
        self.writer, self.fork, self.wait = writer, fork, wait
        self.witness_fd, self.observer_fd = witness_fd, observer_fd
        self.recovery = recovery
        if not self.eligible: return
        try:
            import ast, types, math, select
            self.owner = os.getpid()
            if not (callable(function) and getattr(node, 'obj', None) is function and
                    node.originalname == self.name and node.nodeid == self.nodeid and
                    getattr(getattr(node, 'callspec', None), 'params', {}) == {} and
                    caller.f_code is function.__code__ and function.__globals__ is caller.f_globals): return
            projections = {}
            source, target_code, roles, target_tree = self.target_source(function, node, prepared, projections)
            expected_projections = prepared['projections'] if prepared else {}
            model = function.__globals__.get('_t1_pipe_model')
            if not callable(model) or self.nested(source, '_t1_pipe_model') != model.__code__: return
            local = caller.f_locals
            self.observed = local['observed']; self.ready = local['ready']
            self.blocked = local['select_blocked']
            self.clock = recovery.time.monotonic
            self.real_clock = local['clock']; self.start = local['start']
            self.select0 = local['select0']
            self.time_module = local['time']
            self.sleep = self.time_module.sleep
            self.marker_id = id(local['marker'])
            self.session = recovery._maintenance_session
            self.limit = recovery._install_maintenance_limit
            self.model_fork = os.fork
            if not all(type(f) is types.FunctionType for f in
                    (self.observed, self.ready, self.blocked, self.clock, operation, self.model_fork, self.limit)): return
            for name, actual in (('observed', self.observed), ('ready', self.ready), ('select_blocked', self.blocked)):
                if (actual.__code__ is not self.nested(function.__code__, name) or
                        self.code_projection(actual.__code__, projections) != self.code_projection(self.nested(target_code, name), expected_projections) or
                        actual.__globals__ is not function.__globals__): return
            lambdas = [c for c in function.__code__.co_consts if isinstance(c, types.CodeType) and c.co_name == '<lambda>']
            if (len(lambdas) != 2 or not any(operation.__code__ is c for c in lambdas) or
                    not any(self.clock.__code__ is c for c in lambdas) or operation.__code__ is self.clock.__code__): return
            for role, actual in (('clock', self.clock), ('operation', operation)):
                if self.code_projection(actual.__code__, projections) != self.code_projection(roles[role], expected_projections): return
            closure_0 = self.closure(self.observed)
            closure_1 = self.closure(self.ready)
            closure_2 = self.closure(self.blocked)
            closure_3 = self.closure(self.clock)
            closure_4 = self.closure(self.session)
            closure_5 = self.closure(self.model_fork)
            closure_6 = self.closure(fork)
            closure_7 = self.closure(wait)
            if (set(closure_0) != {'write', 'marker'} or
                    closure_0.get('write') is not writer or
                    id(closure_0.get('marker')) != self.marker_id or local['write'] is not writer or
                    not isinstance(writer, types.BuiltinFunctionType) or writer is not _RAW_WRITE or
                    closure_1 != {'select_blocked': self.blocked} or
                    closure_2 != {'select0': self.select0, 'time': self.time_module} or
                    closure_3 != {'clock': self.real_clock, 'start': self.start} or
                    operation.__closure__ is not None or operation.__globals__ is not function.__globals__ or
                    self.clock.__globals__ is not function.__globals__ or
                    type(self.start) is not float or not math.isfinite(self.start) or
                    self.real_clock is not self.time_module.monotonic or
                    (self.real_clock, self.select0, self.sleep, os.kill) != originals or
                    not isinstance(self.sleep, types.BuiltinFunctionType) or
                    not isinstance(self.real_clock, types.BuiltinFunctionType) or
                    not isinstance(self.select0, types.BuiltinFunctionType) or
                    select.select is not self.ready or os.write is not self.observed): return
            if (self.limit.__code__ is not self.nested(model.__code__, '<lambda>') or
                    self.limit.__closure__ is not None or self.limit.__globals__ is not function.__globals__ or
                    self.model_fork.__globals__ is not function.__globals__ or
                    getattr(self.session, '__wrapped__', None) is None or
                    self.session.__wrapped__.__code__ is not self.nested(model.__code__, 'session') or
                    self.session.__code__ is not self.nested(contextmanager.__code__, 'helper') or
                    closure_4 != {'func': self.session.__wrapped__} or
                    self.session.__wrapped__.__globals__ is not function.__globals__ or
                    self.session.__wrapped__.__closure__ is not None or
                    self.model_fork.__code__ is not self.nested(model.__code__, 'tracked') or
                    closure_5.get('fork') is not fork or
                    not isinstance(closure_6.get('fork'), types.BuiltinFunctionType) or
                    not isinstance(closure_7.get('wait'), types.BuiltinFunctionType) or
                    closure_5.get('owned') is not local['owned'] or
                    os.waitpid is not wait): return
            self.owned_id = id(local['owned'])
            self.frame_result = recovery._maintenance_result_frame
            self.hook_functions = (self.observed, self.ready, self.blocked, self.clock, self.operation,
                self.limit, self.model_fork, self.session.__wrapped__, self.frame_result)
            self.hook_codes = tuple(f.__code__ for f in self.hook_functions)
            self.kill = os.kill
            if prepared is None:
                prepared = self.prepare(node, run, recovery)
            if (prepared.get('runner') is not self.runner or
                    prepared.get('frame_code') is not self.frame_result.__code__ or
                    prepared['runner_path'].read_bytes() != prepared['runner_data']): return
            runner_module = prepared['runner_module']
            if (self.nested(runner_module, '_run_bounded_maintenance') != self.runner or
                    self.nested(runner_module, '_maintenance_result_frame') != self.frame_result.__code__ or
                    run.__globals__ is not recovery.__dict__ or self.frame_result.__globals__ is not recovery.__dict__): return
            if not isinstance(self.kill, types.BuiltinFunctionType): return
            self.reap = prepared['reap']
            for name, value in prepared['sites'].items(): setattr(self, name, value)
            self.kill_completion = prepared['kill_completion']
            self.kill_lines = prepared['kill_lines']
            self.local_callbacks = ((self.runner, self.runner_callback), (self.reap, self.reap_callback),
                (self.observed.__code__, self.return_callback), (self.ready.__code__, self.return_callback),
                (self.blocked.__code__, self.return_callback), (self.operation.__code__, self.return_callback),
                (self.frame_result.__code__, self.return_callback))
            self.valid = True
        except BaseException:
            self.valid = False

    @staticmethod
    def timeout_sites(code, tree=None, instructions=None):
        import ast, dis
        tree = ast.parse(Path(code.co_filename).read_bytes()) if tree is None else tree
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.lineno == code.co_firstlineno)
        nodes = tuple(ast.walk(function))
        child = next(n for n in nodes if isinstance(n, ast.While) and ast.unparse(n.test) == 'offset < len(raw)')
        parent = next(n for n in nodes if isinstance(n, ast.While) and
            ast.unparse(n.test) == 'True' and any(isinstance(s, ast.Try) and
                any(isinstance(q, ast.Assign) and ast.unparse(q.value).startswith('os.read(') for q in ast.walk(s)) for s in n.body))
        statements = (child.body[1].body[0], parent.body[1].body[0], parent.body[2].body[0].body[0])
        instructions = tuple(dis.get_instructions(code)) if instructions is None else instructions
        result = []
        for statement in statements:
            if not isinstance(statement, ast.Raise) or ast.unparse(statement) != "raise ValueError('b2_maintenance_timeout')":
                raise ValueError('B timeout source shape')
            offsets = frozenset(i.offset for i in instructions if
                i.opname == 'RAISE_VARARGS' and i.positions.lineno == statement.lineno and
                i.positions.col_offset == statement.col_offset)
            if len(offsets) != 1: raise ValueError('B timeout instruction shape')
            result.append(offsets)
        return tuple(result)

    def bind_hooks(self, session, limit, framing):
        self.session_wrapper, self.limit_wrapper, self.framing_wrapper = session, limit, framing
        self.wrapper_codes = (session.__code__, session.__wrapped__.__code__, limit.__code__, framing.__code__)

    def hooks(self):
        if _B_GAP_ENABLED and _B_GAP_CHILD:
            try:
                if any(_B_SURVIVAL[44:52]): raise ValueError('duplicate gap store')
                _B_SURVIVAL[44:50] = (_B_GAP_CLOCK() // 1000).to_bytes(6, 'little')
                _B_SURVIVAL[50] = 0xA5
                _B_SURVIVAL[51] = 0x5A
            except BaseException:
                try:
                    _B_SURVIVAL[34] = 1
                    _B_SURVIVAL[35] = 254
                except BaseException:
                    pass
        import select
        if _B_GAP_ENABLED and _B_GAP_CHILD:
            try:
                if any(_B_SURVIVAL[60:62]): raise ValueError('duplicate gap store')
                _B_SURVIVAL[60] = 1
                _B_SURVIVAL[61] = 254
            except BaseException:
                try:
                    _B_SURVIVAL[34] = 1
                    _B_SURVIVAL[35] = 254
                except BaseException:
                    pass
        diagnostic = self.__dict__.get('_refusal')
        diagnostic = diagnostic if diagnostic and diagnostic['final_active'] else None
        if _B_GAP_ENABLED and _B_GAP_CHILD:
            try:
                if any(_B_SURVIVAL[62:64]): raise ValueError('duplicate gap store')
                _B_SURVIVAL[62] = 1
                _B_SURVIVAL[63] = 254
            except BaseException:
                try:
                    _B_SURVIVAL[34] = 1
                    _B_SURVIVAL[35] = 254
                except BaseException:
                    pass
        hook_gate = 0
        hook_returned = False
        try:
            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                try:
                    _B_SURVIVAL[15] = 0
                    _B_SURVIVAL[14] = 1
                    _B_SURVIVAL[15] = 255 - (1)
                    if _B_GAP_ENABLED and _B_GAP_CHILD:
                        try:
                            if any(_B_SURVIVAL[52:60]): raise ValueError('duplicate gap store')
                            _B_SURVIVAL[52:58] = (_B_GAP_CLOCK() // 1000).to_bytes(6, 'little')
                            _B_SURVIVAL[58] = 0xA5
                            _B_SURVIVAL[59] = 0x5A
                        except BaseException:
                            try:
                                _B_SURVIVAL[34] = 1
                                _B_SURVIVAL[35] = 254
                            except BaseException:
                                pass
                except BaseException:
                    try:
                        _B_SURVIVAL[34] = 1
                        _B_SURVIVAL[35] = 254
                    except BaseException:
                        pass
            closure_0 = self.closure(self.model_fork)
            closure_1 = self.closure(self.session_wrapper)
            closure_2 = self.closure(self.session_body)
            closure_3 = self.closure(self.limit_wrapper)
            closure_4 = self.closure(self.framing_wrapper)
            closure_5 = self.closure(self.observed)
            closure_6 = self.closure(self.ready)
            closure_7 = self.closure(self.blocked)
            closure_8 = self.closure(self.clock)
            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                try:
                    _B_SURVIVAL[15] = 0
                    _B_SURVIVAL[14] = 2
                    _B_SURVIVAL[15] = 255 - (2)
                except BaseException:
                    try:
                        _B_SURVIVAL[34] = 1
                        _B_SURVIVAL[35] = 254
                    except BaseException:
                        pass
            evaluated = 1
            if diagnostic: hook_gate = 1
            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                try:
                    _B_SURVIVAL[13] = 0
                    _B_SURVIVAL[12] = 1
                    _B_SURVIVAL[13] = 255 - (1)
                except BaseException:
                    try:
                        _B_SURVIVAL[34] = 1
                        _B_SURVIVAL[35] = 254
                    except BaseException:
                        pass
            answer = all((f.__code__ is code for f, code in zip(self.hook_functions, self.hook_codes)))
            if evaluated == 1 and answer:
                evaluated = 2
                if diagnostic: hook_gate = 2
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 2
                        _B_SURVIVAL[13] = 255 - (2)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = (self.session_wrapper.__code__, self.session_body.__code__, self.limit_wrapper.__code__, self.framing_wrapper.__code__) == self.wrapper_codes
            if evaluated == 2 and answer:
                evaluated = 3
                if diagnostic: hook_gate = 3
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 3
                        _B_SURVIVAL[13] = 255 - (3)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = os.fork is self.model_fork
            if evaluated == 3 and answer:
                evaluated = 4
                if diagnostic: hook_gate = 4
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 4
                        _B_SURVIVAL[13] = 255 - (4)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = os.waitpid is self.wait
            if evaluated == 4 and answer:
                evaluated = 5
                if diagnostic: hook_gate = 5
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 5
                        _B_SURVIVAL[13] = 255 - (5)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_0.get('fork') is self.fork
            if evaluated == 5 and answer:
                evaluated = 6
                if diagnostic: hook_gate = 6
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 6
                        _B_SURVIVAL[13] = 255 - (6)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = id(closure_0.get('owned')) == self.owned_id
            if evaluated == 6 and answer:
                evaluated = 7
                if diagnostic: hook_gate = 7
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 7
                        _B_SURVIVAL[13] = 255 - (7)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = os.write is self.observed
            if evaluated == 7 and answer:
                evaluated = 8
                if diagnostic: hook_gate = 8
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 8
                        _B_SURVIVAL[13] = 255 - (8)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = select.select is self.ready
            if evaluated == 8 and answer:
                evaluated = 9
                if diagnostic: hook_gate = 9
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 9
                        _B_SURVIVAL[13] = 255 - (9)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.recovery._maintenance_session is self.session_wrapper
            if evaluated == 9 and answer:
                evaluated = 10
                if diagnostic: hook_gate = 10
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 10
                        _B_SURVIVAL[13] = 255 - (10)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_1.get('func') is self.session_body
            if evaluated == 10 and answer:
                evaluated = 11
                if diagnostic: hook_gate = 11
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 11
                        _B_SURVIVAL[13] = 255 - (11)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_2.get('session') is self.session
            if evaluated == 11 and answer:
                evaluated = 12
                if diagnostic: hook_gate = 12
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 12
                        _B_SURVIVAL[13] = 255 - (12)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.recovery._install_maintenance_limit is self.limit_wrapper
            if evaluated == 12 and answer:
                evaluated = 13
                if diagnostic: hook_gate = 13
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 13
                        _B_SURVIVAL[13] = 255 - (13)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_3.get('function') is self.limit
            if evaluated == 13 and answer:
                evaluated = 14
                if diagnostic: hook_gate = 14
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 14
                        _B_SURVIVAL[13] = 255 - (14)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.recovery._maintenance_result_frame is self.framing_wrapper
            if evaluated == 14 and answer:
                evaluated = 15
                if diagnostic: hook_gate = 15
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 15
                        _B_SURVIVAL[13] = 255 - (15)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_4.get('function') is self.frame_result
            if evaluated == 15 and answer:
                evaluated = 16
                if diagnostic: hook_gate = 16
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 16
                        _B_SURVIVAL[13] = 255 - (16)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.recovery.time.monotonic is self.clock
            if evaluated == 16 and answer:
                evaluated = 17
                if diagnostic: hook_gate = 17
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 17
                        _B_SURVIVAL[13] = 255 - (17)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.time_module.monotonic is self.real_clock
            if evaluated == 17 and answer:
                evaluated = 18
                if diagnostic: hook_gate = 18
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 18
                        _B_SURVIVAL[13] = 255 - (18)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.time_module.sleep is self.sleep
            if evaluated == 18 and answer:
                evaluated = 19
                if diagnostic: hook_gate = 19
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 19
                        _B_SURVIVAL[13] = 255 - (19)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = set(closure_5) == {'write', 'marker'}
            if evaluated == 19 and answer:
                evaluated = 20
                if diagnostic: hook_gate = 20
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 20
                        _B_SURVIVAL[13] = 255 - (20)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_5.get('write') is self.writer
            if evaluated == 20 and answer:
                evaluated = 21
                if diagnostic: hook_gate = 21
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 21
                        _B_SURVIVAL[13] = 255 - (21)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = id(closure_5.get('marker')) == self.marker_id
            if evaluated == 21 and answer:
                evaluated = 22
                if diagnostic: hook_gate = 22
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 22
                        _B_SURVIVAL[13] = 255 - (22)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_6 == {'select_blocked': self.blocked}
            if evaluated == 22 and answer:
                evaluated = 23
                if diagnostic: hook_gate = 23
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 23
                        _B_SURVIVAL[13] = 255 - (23)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_7 == {'select0': self.select0, 'time': self.time_module}
            if evaluated == 23 and answer:
                evaluated = 24
                if diagnostic: hook_gate = 24
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[13] = 0
                        _B_SURVIVAL[12] = 24
                        _B_SURVIVAL[13] = 255 - (24)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = closure_8 == {'clock': self.real_clock, 'start': self.start}

            if answer: hook_gate = 0
            hook_returned = True
            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                try:
                    _B_SURVIVAL[15] = 0
                    _B_SURVIVAL[14] = 3 if hook_gate == 0 else 4
                    _B_SURVIVAL[15] = 255 - (3 if hook_gate == 0 else 4)
                except BaseException:
                    try:
                        _B_SURVIVAL[34] = 1
                        _B_SURVIVAL[35] = 254
                    except BaseException:
                        pass
            return answer
        finally:
            if diagnostic: diagnostic['hook_gate'] = hook_gate
            if not hook_returned:
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[15] = 0
                        _B_SURVIVAL[14] = 5
                        _B_SURVIVAL[15] = 255 - (5)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass

    def bound_raw(self, frame):
        local = frame.f_locals
        raw, offset = local.get('raw'), local.get('offset')
        return (id(frame) == self.runner_id and type(raw) is bytearray and id(raw) == self.raw_id and
            len(raw) == self.raw_length and type(offset) is int and offset == self.offset and
            local.get('write_fd') == self.fd and local.get('deadline') == self.deadline)

    def expired(self, frame):
        import math
        deadline = frame.f_locals.get('deadline')
        return (type(deadline) in (int, float) and math.isfinite(deadline) and deadline == self.deadline and
            self.recovery.time.monotonic is self.clock and self.clock() >= deadline)

    def eagain(self, exc, frame, tb):
        if not (type(exc) is BlockingIOError and exc.errno == errno.EAGAIN and tb is not None and
                tb.tb_frame is frame and tb.tb_lasti in self.write_sites and tb.tb_next is not None and
                tb.tb_next.tb_frame.f_code is self.observed.__code__ and
                tb.tb_next.tb_lasti in self.observed_sites and tb.tb_next.tb_next is None): return False
        local = frame.f_locals
        inner = tb.tb_next.tb_frame.f_locals
        raw, offset, fd = local.get('raw'), local.get('offset'), local.get('write_fd')
        view = inner.get('raw')
        sample = raw[offset:offset + 65536] if (type(raw) is bytearray and type(offset) is int and
            0 <= offset < len(raw) <= 1048576) else None
        identity, flags = self.identity(fd)
        if not (identity == self.pipe and stat.S_ISFIFO(identity[2]) and identity[3] == os.O_WRONLY and
                flags & os.O_NONBLOCK and identity[:2] != self.identity(self.witness_fd)[0][:2] and
                identity[:2] != self.identity(self.observer_fd)[0][:2] and
                os.write is self.observed and self.closure(self.observed).get('write') is self.writer and
                self.hooks() and type(fd) is int and inner.get('fd') == fd and type(raw) is bytearray and
                id(raw) == self.framed_id and
                0 < len(raw) <= 1048576 and type(offset) is int and 0 <= offset < len(raw) and
                type(view) is memoryview and view.obj is raw and view.ndim == 1 and view.c_contiguous and
                view.itemsize == 1 and view.format == 'B' and sample is not None and view.nbytes == len(sample) and
                view == sample and id(local.get('result')) == self.result_id and
                local.get('operation') is self.wrapper and self.closure(self.wrapper).get('operation') is self.operation): return False
        if self.phase == 'armed':
            self.raw_id, self.raw_length, self.offset, self.fd = id(raw), len(raw), offset, fd
            self.phase = 'backpressure'
        return self.phase == 'backpressure' and self.bound_raw(frame)

    def select_B(self, frame, event, arg):
        if not self.valid or event != 'call': return None
        for code, callback in self.local_callbacks:
            if frame.f_code is code:
                frame.f_trace_lines = code is self.runner or code is self.reap
                return callback(frame, event, arg)
        return None

    def runner_callback(self, frame, event, arg):
        self.trace_B(frame, event, arg)
        return self.local_callbacks[0][1] if self.valid else None

    def reap_callback(self, frame, event, arg):
        self.trace_B(frame, event, arg)
        return self.local_callbacks[1][1] if self.valid else None

    def return_callback(self, frame, event, arg):
        self.trace_B(frame, event, arg)
        return self.local_callbacks[2][1] if self.valid else None

    def trace_B(self, frame, event, arg):
        global _B_GAP_CHILD
        if not self.valid: return None
        code = frame.f_code
        token = b''
        try:
            if code is self.runner:
                local = frame.f_locals
                if event == 'call':
                    self.runner_id = id(frame)
                    if not self.hooks():
                        self.valid = False
                        diagnostic = self.__dict__.get('_refusal')
                        if diagnostic and not diagnostic['first_site']:
                            diagnostic['first_site'] = 1
                            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                try:
                                    _B_SURVIVAL[8] = 1
                                    _B_SURVIVAL[9] = 255 - (1)
                                except BaseException:
                                    try:
                                        _B_SURVIVAL[34] = 1
                                        _B_SURVIVAL[35] = 254
                                    except BaseException:
                                        pass
                if event == 'line':
                    if 'deadline' in local and self.deadline is None:
                        import math
                        value = local['deadline']
                        if type(value) not in (int, float) or not math.isfinite(value):
                            self.valid = False
                            diagnostic = self.__dict__.get('_refusal')
                            if diagnostic and not diagnostic['first_site']:
                                diagnostic['first_site'] = 2
                                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                    try:
                                        _B_SURVIVAL[8] = 2
                                        _B_SURVIVAL[9] = 255 - (2)
                                    except BaseException:
                                        try:
                                            _B_SURVIVAL[34] = 1
                                            _B_SURVIVAL[35] = 254
                                        except BaseException:
                                            pass
                        else: self.deadline = value
                    if local.get('identities') and self.pipe is None:
                        self.pipe = local['identities'].get(local.get('write_fd'))
                        self.read_pipe = local['identities'].get(local.get('read_fd'))
                        if (self.pipe is None or self.read_pipe is None or self.pipe[:3] != self.read_pipe[:3] or
                                not stat.S_ISFIFO(self.pipe[2]) or self.pipe[3] != os.O_WRONLY or
                                self.read_pipe[3] != os.O_RDONLY or
                                self.identity(local['read_fd'])[0] != self.read_pipe or
                                self.identity(local['write_fd'])[0] != self.pipe or
                                self.pipe[:2] == self.identity(self.witness_fd)[0][:2] or
                                self.pipe[:2] == self.identity(self.observer_fd)[0][:2]):
                                    self.valid = False
                                    diagnostic = self.__dict__.get('_refusal')
                                    if diagnostic and not diagnostic['first_site']:
                                        diagnostic['first_site'] = 3
                                        if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                            try:
                                                _B_SURVIVAL[8] = 3
                                                _B_SURVIVAL[9] = 255 - (3)
                                            except BaseException:
                                                try:
                                                    _B_SURVIVAL[34] = 1
                                                    _B_SURVIVAL[35] = 254
                                                except BaseException:
                                                    pass
                    if self.phase in ('backpressure', 'timeout') and not self.bound_raw(frame):
                        self.valid = False
                        diagnostic = self.__dict__.get('_refusal')
                        if diagnostic and not diagnostic['first_site']:
                            diagnostic['first_site'] = 4
                            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                try:
                                    _B_SURVIVAL[8] = 4
                                    _B_SURVIVAL[9] = 255 - (4)
                                except BaseException:
                                    try:
                                        _B_SURVIVAL[34] = 1
                                        _B_SURVIVAL[35] = 254
                                    except BaseException:
                                        pass
                if event == 'exception':
                    exc = arg[1]
                    if type(exc) is BlockingIOError and self.phase in ('armed', 'backpressure'):
                        first = self.phase == 'armed'
                        if not self.eagain(exc, frame, arg[2]):
                            self.valid = False
                            diagnostic = self.__dict__.get('_refusal')
                            if diagnostic and not diagnostic['first_site']:
                                diagnostic['first_site'] = 5
                                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                    try:
                                        _B_SURVIVAL[8] = 5
                                        _B_SURVIVAL[9] = 255 - (5)
                                    except BaseException:
                                        try:
                                            _B_SURVIVAL[34] = 1
                                            _B_SURVIVAL[35] = 254
                                        except BaseException:
                                            pass
                        elif first: token = b'B:e;'
                    elif self.timeout(exc):
                        tb = arg[2]  # Live trace-event chain; not yet attached to exc.
                        if local.get('pid') == 0:
                            diagnostic = self.__dict__.get('_refusal')
                            first_child_role = diagnostic and not diagnostic.get('child_role', False)
                            if diagnostic: diagnostic['child_role'] = True
                            if _B_GAP_ENABLED and first_child_role: _B_GAP_CHILD = True
                            if first_child_role and _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                try:
                                    _B_SURVIVAL[4] = 1
                                    _B_SURVIVAL[5] = 255 - (1)
                                except BaseException:
                                    try:
                                        _B_SURVIVAL[34] = 1
                                        _B_SURVIVAL[35] = 254
                                    except BaseException:
                                        pass
                            if not (self.phase == 'backpressure' and self.bound_raw(frame) and
                                    local.get('remaining', 1) <= 0 and self.expired(frame) and tb is not None and
                                    tb.tb_frame is frame and tb.tb_lasti in self.raise_child and tb.tb_next is None):
                                        self.valid = False
                                        diagnostic = self.__dict__.get('_refusal')
                                        if diagnostic and not diagnostic['first_site']:
                                            diagnostic['first_site'] = 6
                                            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                                try:
                                                    _B_SURVIVAL[8] = 6
                                                    _B_SURVIVAL[9] = 255 - (6)
                                                except BaseException:
                                                    try:
                                                        _B_SURVIVAL[34] = 1
                                                        _B_SURVIVAL[35] = 254
                                                    except BaseException:
                                                        pass
                            else: self.timeout_id, self.phase = id(exc), 'timeout'
                        elif not (tb is not None and tb.tb_frame is frame and tb.tb_next is None and
                                (tb.tb_lasti in self.raise_parent and local.get('remaining', 1) <= 0 or
                                 tb.tb_lasti in self.raise_empty and self.empty_select) and self.expired(frame)):
                            self.valid = False
                            diagnostic = self.__dict__.get('_refusal')
                            if diagnostic and not diagnostic['first_site']:
                                diagnostic['first_site'] = 7
                                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                    try:
                                        _B_SURVIVAL[8] = 7
                                        _B_SURVIVAL[9] = 255 - (7)
                                    except BaseException:
                                        try:
                                            _B_SURVIVAL[34] = 1
                                            _B_SURVIVAL[35] = 254
                                        except BaseException:
                                            pass
                        if _B_GAP_ENABLED and local.get('pid') != 0:
                            try:
                                _B_GAP_PARENT[3] += 1
                                if _B_GAP_PARENT[3] != 1: raise ValueError('duplicate parent timeout')
                                gap_time = _B_GAP_CLOCK() // 1000
                                if type(gap_time) is not int or not 0 <= gap_time < 1 << 48: raise ValueError('parent clock range')
                                _B_GAP_PARENT[1] = gap_time
                            except BaseException:
                                _B_GAP_PARENT[5] = 1
                    elif self.phase != 'armed' or local.get('pid') != 0:
                        self.valid = False
                        diagnostic = self.__dict__.get('_refusal')
                        if diagnostic and not diagnostic['first_site']:
                            diagnostic['first_site'] = 8
                            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                try:
                                    _B_SURVIVAL[8] = 8
                                    _B_SURVIVAL[9] = 255 - (8)
                                except BaseException:
                                    try:
                                        _B_SURVIVAL[34] = 1
                                        _B_SURVIVAL[35] = 254
                                    except BaseException:
                                        pass
            elif code is self.operation.__code__ and event == 'return': self.result_id = id(arg)
            elif code is self.frame_result.__code__ and event == 'return':
                self.framed_id = id(arg)
                if type(arg) is not bytearray or frame.f_locals.get('result') is None or id(frame.f_locals['result']) != self.result_id:
                    self.valid = False
                    diagnostic = self.__dict__.get('_refusal')
                    if diagnostic and not diagnostic['first_site']:
                        diagnostic['first_site'] = 9
                        if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                            try:
                                _B_SURVIVAL[8] = 9
                                _B_SURVIVAL[9] = 255 - (9)
                            except BaseException:
                                try:
                                    _B_SURVIVAL[34] = 1
                                    _B_SURVIVAL[35] = 254
                                except BaseException:
                                    pass
            elif code is self.observed.__code__ and event == 'return' and self.phase == 'backpressure' and arg is not None:
                self.valid = False  # Any post-certificate successful write/progress.
                diagnostic = self.__dict__.get('_refusal')
                if diagnostic and not diagnostic['first_site']:
                    diagnostic['first_site'] = 10
                    if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                        try:
                            _B_SURVIVAL[8] = 10
                            _B_SURVIVAL[9] = 255 - (10)
                        except BaseException:
                            try:
                                _B_SURVIVAL[34] = 1
                                _B_SURVIVAL[35] = 254
                            except BaseException:
                                pass
            elif code is self.blocked.__code__ and event == 'return' and frame.f_locals.get('r'):
                caller = frame.f_back
                self.empty_select = (caller is not None and caller.f_code is self.ready.__code__ and
                    caller.f_back is not None and caller.f_back.f_code is self.runner and
                    caller.f_back.f_lasti in self.ready_sites and self.hooks() and
                    frame.f_locals.get('r') == caller.f_locals.get('r') and
                    type(arg) is tuple and len(arg) == 3 and all(type(x) is list and not x for x in arg))
            elif code is self.reap:
                if event in ('call', 'line'):
                    # Line events arm only the kill statement, never the wait loop.
                    self.kill_pending = False
                    frame.f_trace_opcodes = frame.f_lineno in self.kill_lines
                if event in ('exception', 'return'):
                    self.kill_pending = False
                    frame.f_trace_opcodes = False
                if event == 'opcode':
                    if self.kill_pending and frame.f_lasti not in self.kill_sites:
                        if frame.f_lasti == self.kill_completion: self.killed = True
                        else:
                            self.valid = False
                            diagnostic = self.__dict__.get('_refusal')
                            if diagnostic and not diagnostic['first_site']:
                                diagnostic['first_site'] = 11
                                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                                    try:
                                        _B_SURVIVAL[8] = 11
                                        _B_SURVIVAL[9] = 255 - (11)
                                    except BaseException:
                                        try:
                                            _B_SURVIVAL[34] = 1
                                            _B_SURVIVAL[35] = 254
                                        except BaseException:
                                            pass
                        self.kill_pending = False
                        frame.f_trace_opcodes = False
                    if frame.f_lasti in self.kill_sites:
                        parent = frame.f_back
                        self.kill_pending = (os.kill is self.kill and frame.f_locals.get('pid') == self.child and
                            self.child > 0 and frame.f_globals.get('os') is os and
                            frame.f_locals.get('signal') is signal and signal.SIGKILL == 9 and parent is not None and parent.f_code is self.runner and
                            parent.f_lasti in self.reap_sites and parent.f_locals.get('pid') == self.child)
        except BaseException:
            self.valid = False
            diagnostic = self.__dict__.get('_refusal')
            if diagnostic and not diagnostic['first_site']:
                diagnostic['first_site'] = 12
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[8] = 12
                        _B_SURVIVAL[9] = 255 - (12)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
        finally:
            # In particular no event arg/frame/view/exception is captured by IO.
            frame = arg = local = exc = tb = inner = raw = view = caller = parent = None
        if token and self.valid:
            try: self.emit(token)
            except BaseException:
                self.valid = False
                diagnostic = self.__dict__.get('_refusal')
                if diagnostic and not diagnostic['first_site']:
                    diagnostic['first_site'] = 13
                    if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                        try:
                            _B_SURVIVAL[8] = 13
                            _B_SURVIVAL[9] = 255 - (13)
                        except BaseException:
                            try:
                                _B_SURVIVAL[34] = 1
                                _B_SURVIVAL[35] = 254
                            except BaseException:
                                pass
        return self.trace_callback if self.valid else None

    def live_write(self):
        identity, flags = self.identity(self.fd)
        return identity == self.pipe and flags & os.O_NONBLOCK

    def final_child(self, caller):
        diagnostic = self.__dict__.get('_refusal')
        gate = 0
        if diagnostic: diagnostic['final_active'] = True
        if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
            try:
                _B_SURVIVAL[6] = 1
                _B_SURVIVAL[7] = 255 - (1)
            except BaseException:
                try:
                    _B_SURVIVAL[34] = 1
                    _B_SURVIVAL[35] = 254
                except BaseException:
                    pass
        exc = tb = None
        try:
            exc = sys.exception()
            tb = exc.__traceback__ if exc is not None else None
            evaluated = 1
            if diagnostic: gate = 1
            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                try:
                    _B_SURVIVAL[11] = 0
                    _B_SURVIVAL[10] = 1
                    _B_SURVIVAL[11] = 255 - (1)
                except BaseException:
                    try:
                        _B_SURVIVAL[34] = 1
                        _B_SURVIVAL[35] = 254
                    except BaseException:
                        pass
            answer = self.valid
            if evaluated == 1 and answer:
                evaluated = 2
                if diagnostic: gate = 2
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 2
                        _B_SURVIVAL[11] = 255 - (2)
                        if _B_GAP_ENABLED and _B_GAP_CHILD:
                            try:
                                if any(_B_SURVIVAL[36:44]): raise ValueError('duplicate gap store')
                                _B_SURVIVAL[36:42] = (_B_GAP_CLOCK() // 1000).to_bytes(6, 'little')
                                _B_SURVIVAL[42] = 0xA5
                                _B_SURVIVAL[43] = 0x5A
                            except BaseException:
                                try:
                                    _B_SURVIVAL[34] = 1
                                    _B_SURVIVAL[35] = 254
                                except BaseException:
                                    pass
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.hooks()
            if evaluated == 2 and answer:
                evaluated = 3
                if diagnostic: gate = 3
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 3
                        _B_SURVIVAL[11] = 255 - (3)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = sys.gettrace() is self.trace_callback
            if evaluated == 3 and answer:
                evaluated = 4
                if diagnostic: gate = 4
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 4
                        _B_SURVIVAL[11] = 255 - (4)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.phase == 'timeout'
            if evaluated == 4 and answer:
                evaluated = 5
                if diagnostic: gate = 5
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 5
                        _B_SURVIVAL[11] = 255 - (5)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = id(exc) == self.timeout_id
            if evaluated == 5 and answer:
                evaluated = 6
                if diagnostic: gate = 6
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 6
                        _B_SURVIVAL[11] = 255 - (6)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.timeout(exc)
            if evaluated == 6 and answer:
                evaluated = 7
                if diagnostic: gate = 7
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 7
                        _B_SURVIVAL[11] = 255 - (7)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.bound_raw(caller)
            if evaluated == 7 and answer:
                evaluated = 8
                if diagnostic: gate = 8
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 8
                        _B_SURVIVAL[11] = 255 - (8)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.live_write()
            if evaluated == 8 and answer:
                evaluated = 9
                if diagnostic: gate = 9
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 9
                        _B_SURVIVAL[11] = 255 - (9)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = caller.f_code is self.runner
            if evaluated == 9 and answer:
                evaluated = 10
                if diagnostic: gate = 10
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 10
                        _B_SURVIVAL[11] = 255 - (10)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = caller.f_lasti in self.exit_sites
            if evaluated == 10 and answer:
                evaluated = 11
                if diagnostic: gate = 11
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 11
                        _B_SURVIVAL[11] = 255 - (11)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = tb is not None
            if evaluated == 11 and answer:
                evaluated = 12
                if diagnostic: gate = 12
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 12
                        _B_SURVIVAL[11] = 255 - (12)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = tb.tb_frame is caller
            if evaluated == 12 and answer:
                evaluated = 13
                if diagnostic: gate = 13
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 13
                        _B_SURVIVAL[11] = 255 - (13)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = tb.tb_lasti in self.raise_child
            if evaluated == 13 and answer:
                evaluated = 14
                if diagnostic: gate = 14
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 14
                        _B_SURVIVAL[11] = 255 - (14)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = tb.tb_next is None
            if evaluated == 14 and answer:
                evaluated = 15
                if diagnostic: gate = 15
                if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                    try:
                        _B_SURVIVAL[11] = 0
                        _B_SURVIVAL[10] = 15
                        _B_SURVIVAL[11] = 255 - (15)
                    except BaseException:
                        try:
                            _B_SURVIVAL[34] = 1
                            _B_SURVIVAL[35] = 254
                        except BaseException:
                            pass
                answer = self.expired(caller)
            if diagnostic:
                diagnostic['gate'] = gate if not answer else 0
                diagnostic['kind'] = 'FALSE' if not answer else 'UNREACHED'
            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                try:
                    _B_SURVIVAL[16] = 3 if answer is True else 1 if answer is False else 0
                    _B_SURVIVAL[17] = 2 if answer is True else 1 if answer is False else 0
                    _B_SURVIVAL[18] = diagnostic['gate']
                    _B_SURVIVAL[19] = diagnostic['hook_gate'] if diagnostic['gate'] == 2 else 0
                    _B_SURVIVAL[20] = 2 if self.valid is True else 1 if self.valid is False else 3
                    _B_SURVIVAL[21] = {'armed': 1, 'backpressure': 2, 'timeout': 3}.get(self.phase, 4)
                    _B_SURVIVAL[22] = 0xA5
                    _B_SURVIVAL[23] = 0x5A
                except BaseException:
                    try:
                        _B_SURVIVAL[34] = 1
                        _B_SURVIVAL[35] = 254
                    except BaseException:
                        pass
            return answer
        except BaseException:
            if diagnostic:
                diagnostic['gate'] = gate
                diagnostic['kind'] = 'CAUGHT_EXCEPTION'
            self.valid = False
            if _B_SURVIVAL is not None and diagnostic and diagnostic.get('child_role', False):
                try:
                    _B_SURVIVAL[16] = 2
                    _B_SURVIVAL[17] = 1
                    _B_SURVIVAL[18] = gate
                    _B_SURVIVAL[19] = diagnostic['hook_gate'] if gate == 2 else 0
                    _B_SURVIVAL[20] = 2 if self.valid is True else 1 if self.valid is False else 3
                    _B_SURVIVAL[21] = {'armed': 1, 'backpressure': 2, 'timeout': 3}.get(self.phase, 4)
                    _B_SURVIVAL[22] = 0xA5
                    _B_SURVIVAL[23] = 0x5A
                except BaseException:
                    try:
                        _B_SURVIVAL[34] = 1
                        _B_SURVIVAL[35] = 254
                    except BaseException:
                        pass
            return False
        finally:
            if diagnostic: diagnostic['final_active'] = False
            exc = tb = caller = None

    def terminal(self, result, caller, enrolled, child_bytes):
        exc = tb = parent = None
        try:
            parent = caller.f_back
            local = parent.f_locals if parent is not None else {}
            exc = local.get('original'); tb = exc.__traceback__ if exc is not None else None
            if not (self.valid and self.hooks() and sys.gettrace() is self.trace_callback and caller.f_code is self.reap and
                    caller.f_lasti in self.wait_first | self.wait_last and parent is not None and
                    parent.f_code is self.runner and id(parent) == self.runner_id and parent.f_lasti in self.reap_sites and
                    type(result) is tuple and len(result) == 2 and type(result[0]) is int and type(result[1]) is int and
                    result[0] == self.child == caller.f_locals.get('pid') == local.get('pid') and self.child in enrolled and
                    os.getpid() == self.owner and exc is sys.exception() and self.timeout(exc) and
                    type(local.get('raw')) is bytearray and len(local['raw']) == 0 and self.expired(parent) and
                    tb is not None and tb.tb_frame is parent and tb.tb_next is None and
                    (tb.tb_lasti in self.raise_parent and local.get('remaining', 1) <= 0 or
                     tb.tb_lasti in self.raise_empty and self.empty_select) and
                    child_bytes in (b'B:e;', b'B:e;B:f;')): return b''
            status = result[1]
            if status == 74 << 8 and child_bytes == b'B:e;B:f;': suffix = b'B:t;'
            elif status == signal.SIGKILL and self.killed and caller.f_lasti in self.wait_last and caller.f_locals.get('killed') is True:
                suffix = b'B:t;B:k;'
            else: return b''
            self.parent_id, self.parent_checked = id(exc), True
            return suffix
        except BaseException:
            self.valid = False
            return b''
        finally: exc = tb = parent = caller = local = None

    def escaping(self, exc, wrapper):
        tb = exc.__traceback__
        return (self.valid and sys.gettrace() is self.trace_callback and self.hooks() and
            self.parent_checked and id(exc) == self.parent_id and self.timeout(exc) and
            tb is not None and tb.tb_frame.f_code is wrapper and tb.tb_next is not None and
            tb.tb_next.tb_frame.f_code is self.runner and id(tb.tb_next.tb_frame) == self.runner_id and
            tb.tb_next.tb_lasti in self.raise_parent | self.raise_empty and tb.tb_next.tb_next is None)

    @staticmethod
    def transcript(witness, status):
        return (status == 74 << 8 and witness == b'B:a;B:e;B:f;B:t;' or
            status == signal.SIGKILL and witness in (b'B:a;B:e;B:t;B:k;', b'B:a;B:e;B:f;B:t;B:k;'))

    def clear(self):
        # Break callback/self and installed-hook references even after failed setup.
        self.__dict__.clear()


def install(request, monkeypatch, guard):
    from service import recovery
    run, fork, wait = recovery._run_bounded_maintenance, os.fork, os.waitpid
    import select, time
    originals_B = (time.monotonic, select.select, time.sleep, os.kill)
    original_locked_P = recovery._prepare_abandon_locked
    continuation_P = recovery._MaintenanceSession.continue_kernel.__code__
    prepare_P = [c for c in recovery._prepare_abandon.__code__.co_consts
        if isinstance(c, type(run.__code__)) and c.co_name == '<lambda>']
    rehearsal0, transaction0 = recovery._rehearse_preparation, recovery._prefix_transaction
    pipe, read, write, close, exit0 = os.pipe, os.read, os.write, os.close, os._exit
    state = dict(active=False, read=None, write=None, witness=b'', enrolling=False)
    diagnostic = None
    # This private sink belongs to one diagnostic node, not every service test.
    # Generic maintenance ownership and terminal-receipt checks stay active.
    if request.node.nodeid == BlockedPipeWitness_B.nodeid:
        from one_node_observer import Observer
        diagnostic = Observer(request, run.__code__)

    prepared_B = {}
    def clear_prepared_B():
        facts = prepared_B.pop('facts', None)
        if facts is not None: facts.clear()
    request.addfinalizer(clear_prepared_B)
    # Register before preparation/mutation. Fixture undo cannot resurrect facts.
    monkeypatch.setitem(prepared_B, 'facts', None)
    function_B = getattr(request.node, 'obj', None)
    if (request.node.originalname == BlockedPipeWitness_B.name or
            request.node.nodeid == BlockedPipeWitness_B.nodeid or
            function_B is not None and function_B is getattr(sys.modules.get('test_recovery'), BlockedPipeWitness_B.name, None)):
        try: prepared_B['facts'] = BlockedPipeWitness_B.prepare(request.node, run, recovery)
        except BaseException: clear_prepared_B()

    def tracked_fork():
        enrolling = state['active'] and state['enrolling']
        state['enrolling'] = False
        pid = fork()
        if enrolling and pid:
            if state.get('armed_B'):
                state['blocked_B'].child = pid
            guard.owned.add(pid)
            guard.event('enroll', pid=pid)
        return pid

    def tracked_wait(pid, flags):
        result = wait(pid, flags)
        if result[0] in guard.owned:
            if state.get('armed_B'):
                blocked = state['blocked_B']
                try: child_bytes = read(state['read'], 257)
                except BlockingIOError: child_bytes = b''
                combined = state['witness'] + child_bytes
                if len(combined) > 256:
                    blocked.valid = False
                    state['witness'] = b'B:overflow;'
                else:
                    suffix = blocked.terminal(result, sys._getframe(1), guard.owned, child_bytes)
                    if len(combined) + len(suffix) > 256:
                        blocked.valid = False
                        state['witness'] = b'B:overflow;'
                    else: state['witness'] = combined + suffix
            else:
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
            guard.observe(result, witness=state['witness'], armed_P=state.get('armed_P', False),
                armed_B=state.get('armed_B', False),
                expected_signal=signal.SIGKILL if cleanup_phase else _expected_cleanup_signal(request.node, cleanup), expected_exit=expected_exit)
            if diagnostic is not None:
                diagnostic.sample(2, state['witness'], result[0], result[1], cleanup, len(guard.failures), len(guard.owned))
        return result

    def guarded_run(roots, scope, operation, **kwargs):
        if guard.failures:
            pytest.exit(guard.failures[0], returncode=2)
        replay_P = (request.node.originalname == 'test_t1_durable_completion_lost_result_no_false_success' or
            request.node.nodeid == 'tests/service/test_recovery.py::test_t1_durable_completion_lost_result_no_false_success') and DurableLostResultWitness_P.replay_P(
                operation, recovery._prepare_abandon_locked, original_locked_P, sys._getframe(1))
        r, w = pipe()
        blocked_B = None
        trace_P = sys.gettrace()
        lost_P = None
        try:
            os.set_blocking(r, False)
            state.update(active=True, read=r, write=w, witness=b'', enrolling=True)
            sent = set()
            lost_P = DurableLostResultWitness_P(request.node, recovery._prepare_abandon_locked,
                run.__code__, continuation_P, prepare_P[0] if len(prepare_P) == 1 else None, w,
                getattr(operation, '__code__', None), replay=replay_P)
            trace_P = sys.gettrace()
            blocked_B = None
            state['armed_B'] = False
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
                    count = write(w, data)
                    if data.startswith(b'B:') and count != len(data):
                        raise RuntimeError('incomplete B witness write')
            lost_P.emit_P = mark
            # Arm designation before risk even when validation has already failed.
            # Only the positively bound checked replay is excluded above.
            state['armed_P'] = lost_P.eligible
            if lost_P.eligible: state['witness'] = lost_P.armed
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
                if state.get('armed_B'):
                    if diagnostic is not None:
                        diagnostic.sample(1, state['witness'])
                    final_B = code == 74 and blocked_B.final_child(sys._getframe(1))
                    if final_B:
                        try: mark(b'B:f;')
                        except BaseException: blocked_B.valid = False
                    exit0(code)
                    return
                if code == 74:
                    params = getattr(getattr(request.node, 'callspec', None), 'params', {})
                    exc = sys.exception()
                    mode = params.get('mode')
                    if lost_P.eligible:
                        final_P = lost_P.at_exit_P(exc, sys._getframe(1))
                        if final_P: mark(final_P)
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
                if code == 74:
                    if diagnostic is not None:
                        diagnostic.sample(1, state['witness'])
                exit0(code)
            blocked_B = BlockedPipeWitness_B(request.node, run, operation, sys._getframe(1),
                recovery, write, tracked_fork, tracked_wait, w, diagnostic.fd if diagnostic is not None else None, mark, originals_B,
                prepared=prepared_B.get('facts') or False)
            if blocked_B.eligible and not prepared_B.get('facts'): blocked_B.valid = False
            state['blocked_B'] = blocked_B
            state['armed_B'] = blocked_B.eligible
            if blocked_B.eligible:
                state['witness'] = b'B:a;'
                if lost_P.eligible: blocked_B.valid = False
                blocked_B.wrapper = observed_operation
            with monkeypatch.context() as patch:
                if modeled.fail_code(completion) is not None:
                    patch.setattr(recovery, '_write_core_completion', observed_completion)
                patch.setattr(recovery, '_maintenance_session', observed_session)
                patch.setattr(recovery, '_install_maintenance_limit', semantic_phase(install))
                patch.setattr(recovery, '_maintenance_result_frame', semantic_phase(frame_result))
                patch.setattr(os, '_exit', observed_exit)
                patch.setattr(os, 'open', journal_open)
                # Bind the prescribed operation wrapper as well as the actual
                # production lambda; retain its identity for the seam check.
                if blocked_B.eligible:
                    blocked_B.session_body = observed_session.__wrapped__
                    blocked_B.bind_hooks(observed_session, recovery._install_maintenance_limit,
                        recovery._maintenance_result_frame)
                    sys.settrace(blocked_B.trace_callback)
                elif lost_P.eligible:
                    lost_P.prepare_code = observed_operation.__code__
                    original_operation_P = operation
                    if getattr(original_operation_P, '__code__', None) not in prepare_P:
                        lost_P.valid = False
                    sys.settrace(lost_P.trace_P)
                try:
                    return run(roots, scope, observed_operation, **kwargs)
                except BaseException as exc:
                    if blocked_B.eligible and not blocked_B.escaping(exc, guarded_run.__code__):
                        guard.fail('incomplete or invalid escaping B parent contract')
                    raise
        finally:
            try:
                if diagnostic is not None:
                    diagnostic.sample(3, state['witness'], failures=len(guard.failures), owned=len(guard.owned))
            finally:
                if state.get('armed_B') or lost_P is not None and lost_P.eligible: sys.settrace(trace_P)
                if blocked_B is not None: blocked_B.clear()
                clear_prepared_B()
                state.pop('blocked_B', None)
                state['armed_B'] = False
                if lost_P is not None: lost_P.clear_P()
                state['active'] = False
                try: close(r)
                finally: close(w)
            if diagnostic is not None:
                diagnostic.sample(4, state['witness'], failures=len(guard.failures), owned=len(guard.owned))
            if guard.owned:
                guard.fail('missing terminal maintenance receipt')
            if guard.directory is not None and guard.failures:
                raise SystemExit(86)
    try:
        monkeypatch.setattr(os, 'fork', tracked_fork)
        monkeypatch.setattr(os, 'waitpid', tracked_wait)
        monkeypatch.setattr(recovery, '_run_bounded_maintenance', guarded_run)
    except BaseException:
        clear_prepared_B()
        raise
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
    finalizers = []
    install(SimpleNamespace(node=node, addfinalizer=finalizers.append), patch, guard)
    _CURRENT, _ROOT = guard, directory / 'nested'
    _ROOT.mkdir(mode=0o700)
    identity = dict(pid=os.getpid(), node=nodeid, token=token)
    atomic(directory / 'ack.json', identity)
    owner = os.getpid()
    def finish():
        if os.getpid() != owner: return
        if guard.owned: guard.fail('orphan owned maintenance PIDs')
        for finalize in reversed(finalizers):
            finalize()
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
