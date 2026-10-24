"""Explicit operator provisioning. No discovery, inference or qualification here.

The managed root must be private operator-owned storage. Downloader callbacks
receive a fresh file, never an active bundle path; partial files are discarded on
resume, while complete verified files are reused.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request
import multiprocessing
import stat
import time
import math
from lib.tts_backends.assets import verified_bundle, sha256


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def manifest_bytes(value):
    """The one managed wire serialization; identity() remains the object bundle ID."""
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def manifest_sha256(value):
    return hashlib.sha256(manifest_bytes(value)).hexdigest()


def validate_expected(expected):
    if not isinstance(expected, dict):
        raise ValueError('Expected manifest object required')
    value = copy.deepcopy(expected)
    if set(value) != {'schema_version', 'backend', 'language', 'checkpoint', 'revision', 'license', 'access', 'versions', 'files'}:
        raise ValueError('Exact expected manifest fields required')
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or value['backend'] != 'kokoro' or value['language'] not in ('fr', 'en'):
        raise ValueError('Managed provisioning currently supports Kokoro only')
    for key in ('checkpoint', 'revision', 'license', 'access'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError('Expected identity/license/access required')
    versions = value['versions']
    if not isinstance(versions, dict) or set(versions) != {'kokoro-onnx', 'onnxruntime', 'espeakng-loader', 'phonemizer'} or not all(isinstance(v, str) and v for v in versions.values()):
        raise ValueError('Exact runtime versions required')
    if not isinstance(value['files'], list) or len(value['files']) != 2:
        raise ValueError('Exact Kokoro assets required')
    names = []
    for entry in value['files']:
        if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes', 'origin', 'revision'}:
            raise ValueError('Exact expected file fields required')
        names.append(entry['path'])
        if type(entry['bytes']) is not int or entry['bytes'] <= 0 or not isinstance(entry['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Operator expected size and hash required')
        if not isinstance(entry['origin'], str) or not entry['origin'].startswith('https://') or not isinstance(entry['revision'], str) or not entry['revision']:
            raise ValueError('Reviewed origin and revision required')
    if any(not isinstance(n, str) for n in names) or sorted(names) != ['model.onnx', 'voices.bin']:
        raise ValueError('Unsafe or unexpected asset paths')
    if len(manifest_bytes(value)) > 65536:
        raise ValueError('Manifest exceeds engine byte limit')
    return value


def _download_stream(url, target, expected_bytes):
    # At most expected bytes reach disk; one extra byte detects overflow.
    with urllib.request.urlopen(url, timeout=10) as source, target.open('xb') as sink:
        remaining = expected_bytes
        while True:
            block = source.read(min(1024 * 1024, remaining + 1))
            if len(block) > remaining:
                raise ValueError('Download exceeds expected size')
            if not block:
                if remaining:
                    raise ValueError('Download shorter than expected size')
                break
            sink.write(block)
            remaining -= len(block)
        sink.flush()
        os.fsync(sink.fileno())


def _transfer_child(url, target, size):
    try:
        _download_stream(url, target, size)
    except BaseException:
        raise SystemExit(1)


def _budget(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Finite positive wait budget required')


def download(url, target, expected_bytes=None, *, timeout=300, cancelled=lambda: False):
    """Managed transfer has a wall deadline including connect/read, and cancellation.

    A disposable POSIX child owns blocking network I/O. Deadline/cancel terminates
    then kills/reaps it; no background transfer retains the target or global lock.
    Legacy callers have a finite 4 GiB cap, without an exact length requirement.
    """
    _budget(timeout)
    if expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes <= 0):
        raise ValueError('Positive expected byte count required')
    if expected_bytes is None:
        # Legacy acquisition retains EOF semantics under a finite cap.
        expected_bytes = 4 * 1024**3
        worker = _legacy_transfer_child
    else:
        worker = _transfer_child
    process = multiprocessing.get_context('fork').Process(target=worker, args=(url, target, expected_bytes))
    process.start()
    deadline = time.monotonic() + timeout
    try:
        while process.is_alive():
            if cancelled():
                raise InterruptedError('Transfer cancelled')
            if time.monotonic() >= deadline:
                raise TimeoutError('Whole transfer deadline exceeded')
            process.join(min(.05, max(0, deadline - time.monotonic())))
        if process.exitcode != 0:
            raise ValueError('Bounded download failed')
    finally:
        if process.is_alive():
            process.terminate()
            process.join(.2)
        if process.is_alive():
            process.kill()
        process.join()
        process.close()


def _legacy_transfer_child(url, target, cap):
    try:
        with urllib.request.urlopen(url, timeout=10) as source, target.open('xb') as sink:
            remaining = cap
            while block := source.read(min(1024 * 1024, remaining + 1)):
                if len(block) > remaining:
                    raise ValueError('Legacy download cap exceeded')
                sink.write(block)
                remaining -= len(block)
    except BaseException:
        raise SystemExit(1)


def _directory(path):
    if path.is_symlink():
        raise ValueError('Symlink storage refused')
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Private owned directory required')


def _file(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
        raise ValueError('Owned single-link regular file required')


def _seal(bundle):
    for path in bundle.iterdir():
        _file(path)
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        path.chmod(0o444)
    bundle.chmod(0o555)
    if bundle.stat().st_mode & 0o222 or any(p.stat().st_mode & 0o222 for p in bundle.iterdir()):
        raise ValueError('Incomplete bundle sealing')


def provision(root, expected, *, runtime_versions, downloader=download, lock_timeout=30, transfer_timeout=300, cancelled=lambda: False):
    """Stage, verify, seal, activate; exceptions preserve the old active pointer.

    No qualification is created. Runtime versions are supplied by the trusted
    operator seam, not public request JSON. Consumers freeze returned bundle path.
    """
    _budget(lock_timeout)
    _budget(transfer_timeout)
    expected = validate_expected(expected)
    if runtime_versions != expected['versions']:
        raise ValueError('Runtime identity mismatch')
    root = Path(root).absolute()
    # Refuse symlink ancestors as well as leaf entries.
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Symlink storage refused')
    _directory(root)
    for name in ('staging', 'bundles'):
        _directory(root / name)
    key = identity(expected)
    stage = root / 'staging' / key
    bundle = root / 'bundles' / key
    # Serialize all mutations, including active pointer replacement.
    lock = root / '.provision.lock'
    if lock.exists() or lock.is_symlink():
        _file(lock)
    fd = os.open(lock, os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_WRONLY, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Unsafe lock file')
        deadline = time.monotonic() + lock_timeout
        while True:
            if cancelled():
                raise InterruptedError('Lock wait cancelled')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Lock wait deadline exceeded')
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
        if bundle.exists() or bundle.is_symlink():
            if bundle.is_symlink() or not bundle.is_dir() or bundle.stat().st_uid != os.getuid():
                raise ValueError('Unsafe bundle directory')
            for path in bundle.iterdir():
                _file(path)
            _, manifest = verified_bundle(bundle, 'kokoro')
            if {p.name for p in bundle.iterdir()} != {'manifest.json', 'model.onnx', 'voices.bin'}:
                raise ValueError('Unexpected immutable bundle entry')
            if manifest != expected or (bundle / 'manifest.json').read_bytes() != manifest_bytes(expected):
                raise ValueError('Immutable bundle identity mismatch')
        else:
            _directory(stage)
            allowed = {'model.onnx', 'voices.bin', 'manifest.json', 'model.onnx.part', 'voices.bin.part'}
            if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in stage.iterdir()):
                raise ValueError('Unsafe staging entry')
            for path in stage.iterdir():
                _file(path)
            for entry in expected['files']:
                target = stage / entry['path']
                partial = stage / (entry['path'] + '.part')
                if partial.exists():
                    partial.unlink()
                if target.exists():
                    if target.stat().st_size == entry['bytes'] and sha256(target) == entry['sha256']:
                        continue
                    target.unlink()
                # Injected callables are trusted and responsible for their own bounds.
                if downloader is download:
                    downloader(entry['origin'], partial, entry['bytes'], timeout=transfer_timeout, cancelled=cancelled)
                else:
                    downloader(entry['origin'], partial)
                _file(partial)
                if partial.is_symlink() or not partial.is_file() or partial.stat().st_size != entry['bytes'] or sha256(partial) != entry['sha256']:
                    raise ValueError('Downloaded asset differs from expected pins')
                os.replace(partial, target)
            manifest_path = stage / 'manifest.json'
            if manifest_path.exists():
                manifest_path.unlink()
            manifest_path.write_bytes(manifest_bytes(expected))
            verified_bundle(stage, 'kokoro')
            for path in stage.iterdir():
                _file(path)
                path.chmod(0o444)
            os.rename(stage, bundle)
        _seal(bundle)
        pointer = root / 'active.json'
        if pointer.is_symlink():
            raise ValueError('Symlink active pointer refused')
        temporary = root / '.active.next'
        if temporary.is_symlink():
            raise ValueError('Symlink activation temporary refused')
        if pointer.exists():
            _file(pointer)
        if temporary.exists():
            _file(temporary)
            temporary.unlink()
        with temporary.open('x') as stream:
            json.dump({'bundle': key, 'manifest_sha256': manifest_sha256(expected)}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pointer)
        return bundle
    finally:
        os.close(fd)



def readiness_matches(receipt, *, expected, runtime_versions, profile):
    """A real qualification is required; any bound identity change invalidates it."""
    return (receipt.get('qualification') == 'actual' and
            receipt.get('manifest_sha256') == manifest_sha256(validate_expected(expected)) and
            receipt.get('runtime_versions') == runtime_versions == expected['versions'] and
            receipt.get('profile_sha256') == identity(profile) and
            profile.get('backend') == expected['backend'] and
            profile.get('checkpoint') == expected['checkpoint'])
