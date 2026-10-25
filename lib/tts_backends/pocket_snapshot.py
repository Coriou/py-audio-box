"""Anonymous Pocket loader inputs, reclaimed by the kernel on worker exit."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

# Only aliases issued by a live context are accepted as internal references.
_active = set()


def is_snapshot(path):
    return str(path) in _active


@contextmanager
def snapshot(data):
    # Never create a directory entry in the scanned staging namespace. TemporaryFile
    # is anonymous before yielding, including platforms without Linux O_TMPFILE.
    with tempfile.TemporaryFile(mode='w+b') as stream:
        stream.write(data)
        stream.flush()
        stream.seek(0)
        alias = Path('/proc/self/fd' if os.path.isdir('/proc/self/fd') else '/dev/fd') / str(stream.fileno())
        _active.add(str(alias))
        try:
            yield alias
        finally:
            _active.remove(str(alias))
