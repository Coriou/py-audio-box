"""Bounded local Qwen snapshot repair before immutable model publication.

Only a private provisioning stage is written. Published bundles and shared
Docker model authority are never modified by acquisition.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from service import installed_qwen_daemon as qd
from service import provisioning as p

REPAIR_SECONDS = 1200
TRANSFER_SECONDS = 300


def _remaining(end):
    left = end - time.monotonic()
    if left <= 0:
        raise TimeoutError('qwen_snapshot_repair_deadline')
    return left


def _source(snapshot, cache_root, name):
    """Accept normal files and HF blob symlinks only inside an explicit cache."""
    candidate = snapshot / name
    try:
        # Resolve before testing existence: exists() is false for a dangling link,
        # including one whose target escapes the selected cache.
        real = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError('qwen_snapshot_source_invalid_link') from exc
    if not real.is_relative_to(cache_root):
        raise ValueError('qwen_snapshot_source_outside_cache')
    try:
        source_stat = real.lstat()
    except FileNotFoundError:
        # An absent pinned file or a link to an absent blob inside the cache is
        # repairable. No directory or blob is created in the source cache.
        source_links = (path for path in (candidate, *candidate.parents)
                        if path == snapshot or path.is_relative_to(snapshot))
        if any(path.is_symlink() for path in source_links) and not real.parent.is_dir():
            raise ValueError('qwen_snapshot_source_invalid_path')
        return None
    except OSError as exc:
        raise ValueError('qwen_snapshot_source_invalid_path') from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError('qwen_snapshot_regular_source_required')
    fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        os.close(fd)
        raise ValueError('qwen_snapshot_regular_source_required')
    return fd, before, real


def _copy_verified(snapshot, cache_root, entry, target, end):
    opened = _source(snapshot, cache_root, entry['path'])
    if opened is None:
        return False
    fd, before, real = opened
    try:
        if before.st_size != entry['bytes']:
            return False
        digest = hashlib.sha256()
        with os.fdopen(fd, 'rb', closefd=False) as source, target.open('xb') as sink:
            while block := source.read(1024 * 1024):
                _remaining(end)
                digest.update(block)
                sink.write(block)
            sink.flush()
            os.fsync(sink.fileno())
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('qwen_snapshot_source_changed')
        if real != (snapshot / entry['path']).resolve(strict=True):
            raise ValueError('qwen_snapshot_source_changed')
        current = os.stat(real, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns):
            raise ValueError('qwen_snapshot_source_changed')
        if digest.hexdigest() != entry['sha256']:
            target.unlink()
            return False
        return True
    finally:
        os.close(fd)


def repair(root, descriptor, snapshot, *, cache_root=None, offline=False,
           downloader=p.download, seconds=REPAIR_SECONDS):
    """Reuse verified local bytes; fetch only absent/bad bytes with one bounded retry.

    The existing provisioner owns locking, full verification, sealing and the
    active pointer. A failed repair leaves staged bytes for explicit resumption.
    """
    p._budget(seconds)
    expected = p.validate_expected(descriptor)
    name = 'clone' if expected['checkpoint'] == qd.CHECKPOINT else 'Ryan'
    qd.service_descriptor(expected, name)
    if type(offline) is not bool:
        raise ValueError('qwen_snapshot_offline_flag')
    snapshot = Path(snapshot).absolute()
    cache_root = Path(cache_root).absolute() if cache_root is not None else snapshot
    if not snapshot.is_relative_to(cache_root):
        raise ValueError('qwen_snapshot_cache_root')
    if any(path.is_symlink() for path in (cache_root, *cache_root.parents)):
        raise ValueError('qwen_snapshot_cache_symlink')
    end = time.monotonic() + seconds
    # Validate every pinned source path before any private acquisition can run.
    for entry in expected['files']:
        opened = _source(snapshot, cache_root, entry['path'])
        if opened is not None:
            os.close(opened[0])
    by_url = {entry['origin']: entry for entry in expected['files']}
    def acquire(url, target):
        entry = by_url[url]
        if _copy_verified(snapshot, cache_root, entry, target, end):
            return
        if target.exists():
            target.unlink()
        if offline:
            raise FileNotFoundError('qwen_snapshot_missing_or_corrupt_offline:' + entry['path'])
        for attempt in range(2):
            _remaining(end)
            try:
                if downloader is p.download:
                    downloader(url, target, entry['bytes'], timeout=min(TRANSFER_SECONDS, _remaining(end)),
                               allowed_origins=expected['allowed_origins'],
                               qwen_hf_assets=set(expected['allowed_origins']) == p.QWEN_HF_ORIGINS)
                else:
                    downloader(url, target)
                _remaining(end)
                if target.stat().st_size != entry['bytes'] or p._sha256_until(target, lambda: _remaining(end)) != entry['sha256']:
                    raise ValueError('qwen_snapshot_download_pin_mismatch:' + entry['path'])
                _remaining(end)
                return
            except (p.TransientDownloadError, ConnectionError) as exc:
                if target.exists():
                    target.unlink()
                if attempt:
                    raise
                _remaining(end)
            except BaseException:
                if target.exists():
                    target.unlink()
                raise
    return p.provision(root, expected, runtime_versions=expected['versions'],
                       downloader=acquire, lock_timeout=min(30, _remaining(end)),
                       transfer_timeout=min(TRANSFER_SECONDS, _remaining(end)), deadline=end)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Repair a pinned Qwen snapshot in private staging; no Docker or model publication.')
    parser.add_argument('--model', choices=('clone', 'Ryan'), required=True)
    parser.add_argument('--root', required=True, help='Private managed provisioning root')
    parser.add_argument('--snapshot', required=True, help='Existing local snapshot directory')
    parser.add_argument('--cache-root', help='Explicit HF cache root containing snapshot and blob links')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    name = 'qwen-0.6b-base.expected.json' if args.model == 'clone' else 'qwen-0.6b-customvoice-cpu.expected.json'
    descriptor = json.loads((Path(__file__).resolve().parents[1] / 'requirements' / 'tts' / name).read_text())
    bundle = repair(args.root, descriptor, args.snapshot, cache_root=args.cache_root, offline=args.offline)
    print(bundle)


if __name__ == '__main__':
    main()
