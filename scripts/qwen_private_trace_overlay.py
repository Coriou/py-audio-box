"""Generate an operator-only source overlay for bounded private child stderr tails."""
import argparse
import pathlib
import textwrap

RESIDENT = '''            proc = subprocess.Popen(['python','-m','worker.qwen_resident',str(child.fileno()),str(gate_read)],'''
CHILD = '''                    proc = subprocess.Popen(['python', '-m', 'worker.runtime_child', str(child.fileno())] +'''
REDIRECT = 'stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)'
HELPER = '''\
from contextlib import contextmanager

_TRACE_LIMIT = 8192
_TRACE_RESERVE = 32768  # two tails, states, allocation overhead; fixed 64 KiB tmpfs
_trace_lock = threading.Lock()
_trace_owners = {}

def _trace_write(fd, data):
    offset = 0
    while offset < len(data):
        count = os.write(fd, data[offset:])
        if count <= 0:
            raise OSError('private_trace_write_no_progress')
        offset += count

def _trace_state(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        _trace_write(fd, (value + '\\n').encode('ascii'))
    finally:
        os.close(fd)

@contextmanager
def private_child_stderr(role):
    if role not in ('resident', 'attempt'):
        raise ValueError('private_trace_role')
    base = '/tmp/qwen-private-' + role
    with _trace_lock:
        owner = _trace_owners.get(role)
        if owner is not None and owner.is_alive():
            raise RuntimeError('private_trace_previous_reader_alive')
        # Account for both roles even when the other has already allocated files.
        st = os.statvfs('/tmp')
        if st.f_bavail * st.f_frsize < _TRACE_RESERVE or st.f_favail < 4:
            raise RuntimeError('private_trace_tmpfs_budget_unavailable')
        sink = None
        read_fd = write_fd = None
        thread = None
        try:
            sink = os.open(base + '.tail', os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            os.fchmod(sink, 0o600)
            _trace_state(base + '.state', 'open bytes=0 truncated=0')
            read_fd, write_fd = os.pipe()
            reader_fd = read_fd
            reader_sink = sink
            launch_failed = threading.Event()
            def drain():
                tail = bytearray()
                total = 0
                failed = False
                try:
                    while True:
                        chunk = os.read(reader_fd, 4096)
                        if not chunk:
                            break
                        total += len(chunk)
                        tail.extend(chunk)
                        if len(tail) > _TRACE_LIMIT:
                            del tail[:-_TRACE_LIMIT]
                        if not failed:
                            try:
                                os.lseek(reader_sink, 0, os.SEEK_SET)
                                _trace_write(reader_sink, tail)
                                os.ftruncate(reader_sink, len(tail))
                                _trace_state(base + '.state', 'open bytes=%d truncated=%d' % (total, total > _TRACE_LIMIT))
                            except OSError:
                                failed = True
                    _trace_state(base + '.state', 'error launch' if launch_failed.is_set() else ('error sink' if failed else 'complete bytes=%d truncated=%d' % (total, total > _TRACE_LIMIT)))
                except BaseException:
                    try: _trace_state(base + '.state', 'error reader')
                    except OSError: pass
                finally:
                    os.close(reader_fd)
                    os.close(reader_sink)
            thread = threading.Thread(target=drain, name='private-' + role, daemon=True)
            thread.start()
            _trace_owners[role] = thread
            read_fd = sink = None
        except BaseException:
            if thread is None or not thread.is_alive():
                try: _trace_state(base + '.state', 'error setup')
                except OSError: pass
            for fd in (read_fd, write_fd, sink):
                if fd is not None: os.close(fd)
            raise
    try:
        yield write_fd
    except BaseException:
        launch_failed.set()
        try: _trace_state(base + '.state', 'error launch')
        except OSError: pass
        raise
    finally:
        os.close(write_fd)
        thread.join(timeout=0.2)
''' 


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=pathlib.Path, required=True)
    p.add_argument('--image', required=True, help='verified local image tag in the remote Docker daemon')
    p.add_argument('--output', type=pathlib.Path, required=True)
    a = p.parse_args()
    if not a.image or any(c.isspace() for c in a.image) or a.image.startswith('-'):
        p.error('explicit local image tag required')
    source = a.source.read_text()
    if source.count(RESIDENT) != 1 or source.count(CHILD) != 1:
        p.error('launch anchors differ; inspect source before overlay')
    # Replace only the two selected launch statements and their stderr keywords.
    for anchor, role in ((RESIDENT, 'resident'), (CHILD, 'attempt')):
        start = source.index(anchor)
        end = source.index(REDIRECT, start) + len(REDIRECT)
        original = source[start:end]
        if original.count(REDIRECT) != 1:
            p.error('ambiguous stderr anchor')
        indent = original[:len(original) - len(original.lstrip())]
        changed = original.replace(REDIRECT, 'stdout=subprocess.DEVNULL, stderr=trace_write)')
        source = source[:start] + indent + 'with private_child_stderr(\'' + role + '\') as trace_write:\n' + '\n'.join('    ' + line for line in changed.splitlines()) + source[end:]
    if source.count('import threading\n') != 1:
        p.error('threading import differs')
    source = source.replace('import threading\n', 'import threading\n' + HELPER + '\n', 1)
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    (a.output / 'runtime_supervisor.py').write_text(source)
    (a.output / 'Dockerfile').write_text(textwrap.dedent(f'''\
        FROM {a.image}
        COPY --chown=0:10001 runtime_supervisor.py /opt/speech/worker/runtime_supervisor.py
        '''))
    print(a.output)


if __name__ == '__main__':
    main()
