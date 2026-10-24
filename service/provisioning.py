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
from lib.tts_backends.assets import verified_bundle, sha256


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_expected(expected):
    value = copy.deepcopy(expected)
    if set(value) != {'schema_version', 'backend', 'language', 'checkpoint', 'license', 'access', 'versions', 'files'}:
        raise ValueError('Exact expected manifest fields required')
    if value['schema_version'] != 1 or value['backend'] != 'kokoro' or value['language'] not in ('fr', 'en'):
        raise ValueError('Managed provisioning currently supports Kokoro only')
    for key in ('checkpoint', 'license', 'access'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError('Expected identity/license/access required')
    versions = value['versions']
    if not isinstance(versions, dict) or set(versions) != {'kokoro-onnx', 'onnxruntime', 'espeakng-loader', 'phonemizer'} or not all(isinstance(v, str) and v for v in versions.values()):
        raise ValueError('Exact runtime versions required')
    if not isinstance(value['files'], list) or len(value['files']) != 2:
        raise ValueError('Exact Kokoro assets required')
    names = []
    for entry in value['files']:
        if set(entry) != {'path', 'sha256', 'bytes', 'origin', 'revision'}:
            raise ValueError('Exact expected file fields required')
        names.append(entry['path'])
        if type(entry['bytes']) is not int or entry['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Operator expected size and hash required')
        if not isinstance(entry['origin'], str) or not entry['origin'].startswith('https://') or not isinstance(entry['revision'], str) or not entry['revision']:
            raise ValueError('Reviewed origin and revision required')
    if sorted(names) != ['model.onnx', 'voices.bin']:
        raise ValueError('Unsafe or unexpected asset paths')
    return value


def download(url, target):
    with urllib.request.urlopen(url, timeout=120) as source, target.open('xb') as sink:
        while block := source.read(1024 * 1024):
            sink.write(block)
        sink.flush()
        os.fsync(sink.fileno())


def _directory(path):
    if path.is_symlink():
        raise ValueError('Symlink storage refused')
    path.mkdir(exist_ok=True)
    if not path.is_dir():
        raise ValueError('Directory required')


def provision(root, expected, *, runtime_versions, downloader=download):
    """Stage, verify, seal, activate; exceptions preserve the old active pointer.

    No qualification is created. Runtime versions are supplied by the trusted
    operator seam, not public request JSON. Consumers freeze returned bundle path.
    """
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
    fd = os.open(lock, os.O_CREAT | os.O_NOFOLLOW | os.O_WRONLY, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        if bundle.exists() or bundle.is_symlink():
            _, manifest = verified_bundle(bundle, 'kokoro')
            if {p.name for p in bundle.iterdir()} != {'manifest.json', 'model.onnx', 'voices.bin'}:
                raise ValueError('Unexpected immutable bundle entry')
            if manifest != expected:
                raise ValueError('Immutable bundle identity mismatch')
        else:
            _directory(stage)
            allowed = {'model.onnx', 'voices.bin', 'manifest.json', 'model.onnx.part', 'voices.bin.part'}
            if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in stage.iterdir()):
                raise ValueError('Unsafe staging entry')
            for entry in expected['files']:
                target = stage / entry['path']
                partial = stage / (entry['path'] + '.part')
                if partial.exists():
                    partial.unlink()
                if target.exists():
                    if target.stat().st_size == entry['bytes'] and sha256(target) == entry['sha256']:
                        continue
                    target.unlink()
                downloader(entry['origin'], partial)
                if partial.is_symlink() or not partial.is_file() or partial.stat().st_size != entry['bytes'] or sha256(partial) != entry['sha256']:
                    raise ValueError('Downloaded asset differs from expected pins')
                os.replace(partial, target)
            manifest_path = stage / 'manifest.json'
            if manifest_path.exists():
                manifest_path.unlink()
            manifest_path.write_text(json.dumps(expected, sort_keys=True) + '\n')
            verified_bundle(stage, 'kokoro')
            for path in stage.iterdir():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
                path.chmod(0o444)
            os.rename(stage, bundle)
            bundle.chmod(0o555)
        pointer = root / 'active.json'
        if pointer.is_symlink():
            raise ValueError('Symlink active pointer refused')
        temporary = root / '.active.next'
        if temporary.is_symlink():
            raise ValueError('Symlink activation temporary refused')
        if temporary.exists():
            temporary.unlink()
        with temporary.open('x') as stream:
            json.dump({'bundle': key, 'manifest_sha256': key}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pointer)
        return bundle
    finally:
        os.close(fd)



def readiness_matches(receipt, *, expected, runtime_versions, profile):
    """A real qualification is required; any bound identity change invalidates it."""
    return (receipt.get('qualification') == 'actual' and
            receipt.get('manifest_sha256') == identity(validate_expected(expected)) and
            receipt.get('runtime_versions') == runtime_versions == expected['versions'] and
            receipt.get('profile_sha256') == identity(profile) and
            profile.get('backend') == expected['backend'] and
            profile.get('checkpoint') == expected['checkpoint'])
