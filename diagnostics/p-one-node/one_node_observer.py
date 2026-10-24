"""Private observational sink. Never interprets records as acceptance."""
import os
import stat
import sys

NODE = 'tests/service/test_recovery.py::test_t1_real_blocked_pipe_backpressure_deadline'
DIRECTORY = '/tmp/pab-p-one-node-observer'


class Observer:
    def __init__(self, request, run_code):
        if request.node.nodeid != NODE:
            raise RuntimeError('STOP observer node')
        self.owner = os.getpid()
        self.run_code = run_code
        self.write = os.write
        self.close_fd = os.close
        self.fd = None
        request.addfinalizer(self.close)
        self.rows = 0
        self.bytes = 0
        os.mkdir(DIRECTORY, 0o700)  # Existing evidence is a STOP, never reused.
        info = os.lstat(DIRECTORY)
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
            raise RuntimeError('STOP observer directory')
        self.fd = os.open(DIRECTORY+'/records', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_APPEND, 0o600)
        info = os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1:
            self.close_fd(self.fd)
            self.fd = None
            raise RuntimeError('STOP observer file')

    def sample(self, stage, witness, child=0, status=0, cleanup=False, failures=0, owned=0):
        """Only transient exception/frame references; never pass them to writer.

        Stages: 1 child original catch-all; 2 existing terminal wait;
        3 guarded-run finally entry; 4 existing cleanup completed.
        Maximum 8 rows/process, 512 bytes/row; one child + owner <=8192.
        """
        exc = frame = args = original = runner = caller_code = None
        try:
            if self.fd is None or self.rows >= 8:
                return
            exc = sys.exception()
            frame = sys._getframe(1)
            caller_code = frame.f_code
            caller_offset = frame.f_lasti
            runner = frame
            for _ in range(12):
                if runner is None or runner.f_code is self.run_code:
                    break
                runner = runner.f_back
            runner_bound = runner is not None and runner.f_code is self.run_code
            # Parent cleanup's original is the runner's actual local, not a
            # predicted timeout or an exception supplied by an attestor.
            original = runner.f_locals.get('original') if runner_bound else None
            args = BaseException.args.__get__(exc) if isinstance(exc, BaseException) else ()
            shape = type(args) is tuple
            count = len(args) if shape else 0
            safe_one = shape and count == 1 and type(args[0]) is str
            number = OSError.errno.__get__(exc) if isinstance(exc, OSError) else None
            errno_int = type(number) is int and -(2**31) <= number < 2**31
            # All payload values are exact builtin ints/bools. No str/repr of
            # exceptions, classes, args, paths, rows, locals or tracebacks.
            values = [1, 1, stage, self.owner, os.getpid(), os.getppid(), child, status,
                cleanup, failures, owned, len(witness), witness == b'',
                exc is not None, id(exc) if exc is not None else 0,
                id(type(exc)), type(exc) is ValueError, type(exc) is BlockingIOError,
                type(exc) is BrokenPipeError, type(exc) is OSError,
                errno_int, number if errno_int else 0, number is None,
                shape, count, safe_one,
                safe_one and args[0] == 'b2_maintenance_timeout',
                safe_one and args[0] == 'b2_maintenance_child_failed',
                id(caller_code), caller_code.co_firstlineno, caller_offset,
                runner_bound, id(runner.f_code) if runner_bound else 0,
                runner.f_lasti if runner_bound else -1,
                original is not None, id(original) if original is not None else 0,
                id(type(original)), type(original) is ValueError,
                original is exc]
            values += [id(type(args[i])) if shape and i < count else 0 for i in range(8)]
            # Release references BEFORE serialization/IO. IDs are process-local
            # observations only and never portable type/name authority.
            del exc, frame, args, original, runner, caller_code
            raw = (','.join(str(int(v)) for v in values)+'\n').encode('ascii')
            if len(raw) > 512 or self.bytes + len(raw) > 8192:
                return
            self.rows += 1
            self.bytes += len(raw)
            self.write(self.fd, raw)  # one attempt; no retry/wait/clock
        except BaseException:
            # Missing/truncated observation is STOP, never test acceptance.
            pass
        finally:
            exc = frame = args = original = runner = caller_code = None

    def close(self):
        fd, self.fd = self.fd, None
        self.run_code = None
        if fd is not None:
            try: self.close_fd(fd)
            except BaseException: pass
