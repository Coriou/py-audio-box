"""Private Linux initializer mount view. No host mount or general path interface.

Only the fresh exec child calls LinuxMounts. Tests inject namespace operations;
those tests establish pathname semantics, never kernel permission qualification.
"""
import ctypes
import os
from pathlib import Path
import stat
import sys

BASE = Path('/mnt')
CONTROL = BASE / 'pab-install/control'
STAGING = BASE / 'pab-install/staging'


class LinuxMounts:
    def __init__(self):
        if sys.platform != 'linux' or os.getuid() != 0 or os.geteuid() != 0:
            raise ValueError('linux_root_initializer_required')
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.unshare.argtypes = [ctypes.c_int]
        self.libc.unshare.restype = ctypes.c_int
        self.libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]
        self.libc.mount.restype = ctypes.c_int

    def _check(self, result):
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    def unshare(self):
        self._check(self.libc.unshare(0x00020000))  # CLONE_NEWNS only

    def mount(self, source, target, kind, flags, data=None):
        def b(value): return None if value is None else os.fsencode(value)
        self._check(self.libc.mount(b(source), b(target), b(kind), flags, b(data)))

    def mkdir(self, path):
        os.mkdir(path, 0o755)

    def stat(self, path):
        return os.stat(path, follow_symlinks=False)


def prepare(public, control_fd, staging_fd, *, ops=None):
    """Build fixed canonical paths before UID drop; failure kills this helper.

    No chroot, recursive bind, caller mount argument, host mkdir, or unmount.
    The namespace and its mounts disappear when this sole helper exits.
    """
    from service import install_policy as ip
    ip.installation(public)
    descriptors = (control_fd, staging_fd)
    if any(type(fd) is not int or fd < 3 for fd in descriptors) or len(set(descriptors)) != 2:
        raise ValueError('closed_initializer_descriptors')
    identities = []
    for role, fd in zip(('control', 'staging'), descriptors):
        info = os.fstat(fd)
        ip.check(info, ip.SPLIT, role, directory=True)
        record = public['roles'][role]
        identity = (info.st_dev, info.st_ino)
        if identity != (record['device'], record['inode']):
            raise ValueError('installed_root_changed')
        identities.append(identity)
    if identities[0] == identities[1]:
        raise ValueError('installed_root_alias')
    ops = LinuxMounts() if ops is None else ops
    # Refuse missing/symlink/non-root-owned mountpoint. No creation on host.
    info = ops.stat(BASE)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError('unsafe_initializer_mountpoint')
    ops.unshare()
    ops.mount(None, '/', None, 0x4000 | 0x40000)  # MS_REC | MS_PRIVATE
    # Private bounded tmpfs hides /mnt only, leaving interpreter/modules intact.
    ops.mount('tmpfs', BASE, 'tmpfs', 2 | 4, 'mode=0755,size=64k,nr_inodes=16')
    ops.mkdir(BASE / 'pab-install')
    for fd, target, identity in zip(descriptors, (CONTROL, STAGING), identities):
        ops.mkdir(target)
        ops.mount('/proc/self/fd/' + str(fd), target, None, 4096)  # MS_BIND
        info = ops.stat(target)
        if (info.st_dev, info.st_ino) != identity or not stat.S_ISDIR(info.st_mode):
            raise ValueError('initializer_bind_identity_changed')
    return CONTROL, STAGING
