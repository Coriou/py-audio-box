"""Post-reap Pocket/Qwen housekeeping: one literal empty-directory removal only.

Runs as the stage owner (10001), with the allocated child's primary group.
The dead exclusive child and trusted control owner provide writer exclusion.
There is deliberately no replay, recursive traversal, or repair interface.
"""
import os
import stat
import sys


class ConditioningRefused(ValueError):
    pass


def _identity(s):
    return s.st_dev, s.st_ino, s.st_uid, s.st_gid, s.st_mode


def _directory(s, uid, gid, mode, device=None):
    if (not stat.S_ISDIR(s.st_mode) or s.st_uid != uid or s.st_gid != gid or
            stat.S_IMODE(s.st_mode) != mode or (device is not None and s.st_dev != device)):
        raise ConditioningRefused('conditioning_identity')


def remove_empty(stage, child_uid, *, policy=None, managed_namespace=False):
    """Validate fixed /staging and remove only conditioning, if kernel-empty."""
    if type(managed_namespace) is not bool:raise ConditioningRefused('conditioning_policy')
    if (not isinstance(stage, str) or not stage or stage in ('.', '..') or
            '/' in stage or '\\' in stage or '\0' in stage or
            type(child_uid) is not int or not 20000 <= child_uid < 20000+64):
        raise ConditioningRefused('conditioning_allocation')
    from service import install_policy as ip
    if policy is not None and policy is not ip.POCKET_SPLIT and policy is not ip.QWEN_SPLIT:
        raise ConditioningRefused('conditioning_policy')
    flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    # Linux O_PATH permits traversal of mode1733 without directory-read access.
    root = os.open('/staging', os.O_PATH | flags)
    try:
        root_stat = os.fstat(root)
        _directory(root_stat, *(ip.expectation(policy, 'staging', directory=True) if policy in (ip.POCKET_SPLIT, ip.QWEN_SPLIT) else (0, 0, 0o1733)))
        def root_bound():
            if _identity(os.stat('/staging', follow_symlinks=False)) != _identity(root_stat):
                raise ConditioningRefused('conditioning_root_changed')
        root_bound()
        fd = os.open(stage, os.O_RDONLY | flags, dir_fd=root)
        try:
            bound = os.fstat(fd)
            _directory(bound, 10001, child_uid, 0o770, root_stat.st_dev)
            def stage_bound():
                root_bound()
                if (_identity(os.fstat(fd)) != _identity(bound) or
                        _identity(os.stat(stage, dir_fd=root, follow_symlinks=False)) != _identity(bound)):
                    raise ConditioningRefused('conditioning_stage_changed')
            stage_bound()
            try:
                child = os.stat('conditioning', dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                stage_bound()
                return
            _directory(child, 10001 if managed_namespace else child_uid, child_uid, 0o770 if managed_namespace else 0o700, bound.st_dev)
            stage_bound()
            if _identity(os.stat('conditioning', dir_fd=fd, follow_symlinks=False)) != _identity(child):
                raise ConditioningRefused('conditioning_changed')
            os.rmdir('conditioning', dir_fd=fd)  # Kernel, atomic emptiness check.
            os.fsync(fd)
            stage_bound()
        finally:
            os.close(fd)
    finally:
        os.close(root)


def main(*, _managed_namespace=False):
    if len(sys.argv) not in (3, 4):
        raise ConditioningRefused('conditioning_arguments')
    uid = int(sys.argv[2])
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid(), os.getgroups()) != (10001, 10001, uid, uid, []):
        raise ConditioningRefused('conditioning_credentials')
    from service import install_policy as ip
    policy = None
    if len(sys.argv) == 4:
        if sys.argv[3] not in (ip.POCKET_V2_POLICY, ip.QWEN_POLICY):
            raise ConditioningRefused('conditioning_policy')
        policy = ip.QWEN_SPLIT if sys.argv[3] == ip.QWEN_POLICY else ip.POCKET_SPLIT
    remove_empty(sys.argv[1], uid, policy=policy, **(dict(managed_namespace=True) if _managed_namespace else {}))


if __name__ == '__main__':
    main()


def pocket_main():
    return main(_managed_namespace=True)
