"""In-process RSS observations for one owned PID; no model or subprocess APIs.

Darwin ABI: Apple's sys/resource.h rusage_info_v0 + libproc proc_pid_rusage.
Linux: proc_pid_stat(5), starttime field 22 and resident pages field 24.
"""
import ctypes
import errno
import os
import sys
import threading
import time


class ProcessGone(ProcessLookupError):
    pass


class ProcessReused(RuntimeError):
    pass


class RUsageV0(ctypes.Structure):
    _fields_ = [('uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        'user_time', 'system_time', 'pkg_idle_wkups', 'interrupt_wkups', 'pageins',
        'wired_size', 'resident_size', 'phys_footprint', 'start_abstime', 'exit_abstime')]


class ProcessRSS:
    def __init__(self, pid):
        if type(pid) is not int or not 0 < pid <= 2**31-1:
            raise ValueError('positive_target_pid_required')
        self.pid, self.identity = pid, None
        if sys.platform == 'darwin':
            self.method = 'proc_pid_rusage_v0'
            self.library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
            self.query = self.library.proc_pid_rusage
            self.query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self.query.restype = ctypes.c_int
            if ctypes.sizeof(RUsageV0) != 96:
                raise RuntimeError('unsupported_rusage_v0_ABI')
        elif sys.platform.startswith('linux'):
            self.method = 'proc_pid_stat'
            self.page_size = os.sysconf('SC_PAGE_SIZE')
        else:
            raise RuntimeError('unsupported_RSS_platform')

    def _darwin(self):
        value = RUsageV0()
        ctypes.set_errno(0)
        result = self.query(self.pid, 0, ctypes.byref(value))
        if result != 0:
            code = ctypes.get_errno()
            if code in (errno.ESRCH, errno.ENOENT):
                raise ProcessGone(code, 'RSS target exited', self.pid)
            raise OSError(code, 'proc_pid_rusage failed', self.pid)
        if not value.start_abstime:
            raise ValueError('missing_process_start_identity')
        if value.exit_abstime:
            raise ProcessGone(errno.ESRCH, 'RSS target exited', self.pid)
        return value.start_abstime, value.resident_size

    def _linux(self):
        try:
            with open(f'/proc/{self.pid}/stat', 'rb') as stream:
                raw = stream.read(65537)
        except (FileNotFoundError, ProcessLookupError) as exc:
            raise ProcessGone(errno.ESRCH, 'RSS target exited', self.pid) from exc
        if len(raw) > 65536:
            raise ValueError('proc_stat_limit')
        # comm may contain spaces or closing parentheses. Numeric fields follow
        # its final closing parenthesis, with field 3 (state) at index zero.
        boundary = raw.rfind(b') ')
        if boundary < 0 or int(raw.split(b' ', 1)[0]) != self.pid:
            raise ValueError('proc_stat_target_or_format')
        fields = raw[boundary+2:].split()
        if len(fields) < 22:
            raise ValueError('proc_stat_truncated')
        if fields[0] in (b'Z', b'X', b'x'):
            raise ProcessGone(errno.ESRCH, 'RSS target exited', self.pid)
        start, pages = int(fields[19]), int(fields[21])
        if start <= 0 or pages < 0:
            raise ValueError('proc_stat_values')
        return start, pages * self.page_size

    def sample(self):
        identity, rss = self._darwin() if self.method == 'proc_pid_rusage_v0' else self._linux()
        if self.identity is not None and identity != self.identity:
            raise ProcessReused('RSS target PID start identity changed')
        self.identity = identity
        return dict(pid=self.pid, identity=identity, rss_bytes=rss,
                    observed_at=time.monotonic(), method=self.method)


class Sampler:
    """One non-replacing daemon reader; controller never waits for a sample.

    ctypes.CDLL releases the GIL during OS queries. A stalled query cannot hold
    the controller's elapsed deadline or child kill/wait path. It is never retried
    by spawning another reader; stop/join status is reported honestly.
    """
    def __init__(self, pid, *, reader_factory=ProcessRSS):
        self.pid, self.reader_factory = pid, reader_factory
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.last = self.error = None
        self.peak, self.count = None, 0
        self.thread = threading.Thread(target=self._run, name=f'RSS-{pid}', daemon=True)

    def _run(self):
        try:
            reader = self.reader_factory(self.pid)
            while not self.stop_event.is_set():
                value = reader.sample()
                if self.stop_event.is_set(): break  # Never publish after owner cleanup begins.
                if value['pid'] != self.pid:
                    raise ValueError('RSS sample target PID mismatch')
                with self.lock:
                    self.last = value
                    self.peak = max(self.peak or 0, value['rss_bytes'])
                    self.count += 1
                if self.stop_event.wait(.1): break
        except Exception as exc:
            with self.lock:
                self.error = dict(type=type(exc).__name__, message=str(exc)[:1000],
                    errno=getattr(exc, 'errno', None), observed_at=time.monotonic())

    def snapshot(self):
        with self.lock:
            return dict(last=None if self.last is None else dict(self.last),
                        error=None if self.error is None else dict(self.error),
                        peak=self.peak, count=self.count)

    def start(self):
        self.thread.start()

    def stop(self):
        self.request_stop()
        self.thread.join(timeout=.1)
        return not self.thread.is_alive()

    def request_stop(self):
        self.stop_event.set()
