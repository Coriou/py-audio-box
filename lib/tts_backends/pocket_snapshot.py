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


@contextmanager
def managed_directory(path):
    """Trusted Pocket attempt namespace, traversable by Core and one engine UID."""
    import stat
    directory=Path(path).absolute()
    uid=os.getuid()
    if (type(uid) is not int or not 20000<=uid<20064 or directory.name!='conditioning' or
            '..' in directory.parts or os.geteuid()!=uid or os.getgid()!=uid or os.getegid()!=uid or os.getgroups()):
        raise ValueError('managed_snapshot_namespace')
    flags=getattr(os,'O_PATH',os.O_RDONLY)|os.O_DIRECTORY|os.O_NOFOLLOW
    fd=os.open('/',flags)
    parent=None
    try:
        for part in directory.parts[1:-1]:
            child=os.open(part,flags,dir_fd=fd)
            os.close(fd);fd=child
        parent=fd;fd=None
        stage=os.fstat(parent)
        child=os.open('conditioning',flags,dir_fd=parent)
        fd=child;bound=os.fstat(child)
        def identity(info):return info.st_dev,info.st_ino,info.st_uid,info.st_gid,info.st_mode
        def check(info):
            if not stat.S_ISDIR(info.st_mode) or (info.st_uid,info.st_gid,stat.S_IMODE(info.st_mode))!=(10001,uid,0o770):
                raise ValueError('managed_snapshot_namespace')
        check(stage);check(bound)
        if stage.st_dev!=bound.st_dev:raise ValueError('managed_snapshot_namespace')
        def stable():
            if (identity(os.fstat(parent))!=identity(stage) or identity(os.stat(directory.parent,follow_symlinks=False))!=identity(stage) or
                    identity(os.fstat(child))!=identity(bound) or identity(os.stat('conditioning',dir_fd=parent,follow_symlinks=False))!=identity(bound)):
                raise ValueError('managed_snapshot_namespace_changed')
        stable()
        try:yield directory
        finally:stable()
    finally:
        if fd is not None:os.close(fd)
        if parent is not None:os.close(parent)
