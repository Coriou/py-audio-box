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
from urllib.parse import urlsplit
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
    return hashlib.sha256(manifest_bytes(bundle_manifest(value))).hexdigest()


def validate_expected(expected):
    if not isinstance(expected, dict):
        raise ValueError('Expected manifest object required')
    value = copy.deepcopy(expected)
    pocket = value.get('backend') == 'pocket-tts'
    qwen = value.get('backend') == 'qwen'
    extra = {'allowed_origins', 'execution'} if qwen else {'config', 'allowed_origins'} if pocket else set()
    if set(value) != extra | {'schema_version', 'backend', 'language', 'checkpoint', 'revision', 'license', 'access', 'versions', 'files'}:
        raise ValueError('Exact expected manifest fields required')
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or value['backend'] not in ('kokoro', 'pocket-tts', 'qwen') or value['language'] not in (('English', 'French') if qwen else ('fr', 'en')):
        raise ValueError('Unsupported managed backend or language')
    for key in ('checkpoint', 'revision', 'license', 'access'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError('Expected identity/license/access required')
    versions = value['versions']
    if not isinstance(versions, dict) or set(versions) != ({'qwen-tts', 'torch', 'transformers', 'tokenizers', 'safetensors'} if qwen else {'pocket-tts', 'torch', 'tokenizers'} if pocket else {'kokoro-onnx', 'onnxruntime', 'espeakng-loader', 'phonemizer'}) or not all(isinstance(v, str) and v for v in versions.values()):
        raise ValueError('Exact runtime versions required')
    if not isinstance(value['files'], list) or not (4 <= len(value['files']) <= 256 if qwen else 2 <= len(value['files']) <= 128 if pocket else len(value['files']) == 2):
        raise ValueError('Exact backend assets required')
    names = []
    for entry in value['files']:
        if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes', 'origin', 'revision'}:
            raise ValueError('Exact expected file fields required')
        names.append(entry['path'])
        if type(entry['bytes']) is not int or entry['bytes'] <= 0 or not isinstance(entry['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Operator expected size and hash required')
        if not isinstance(entry['origin'], str) or not entry['origin'].startswith('https://') or not isinstance(entry['revision'], str) or not entry['revision']:
            raise ValueError('Reviewed origin and revision required')
    if any(not isinstance(n, str) for n in names):
        raise ValueError('Unsafe asset path')
    if qwen:
        _qwen_descriptor(value)
    elif pocket:
        _pocket_config(value)
    elif sorted(names) != ['model.onnx', 'voices.bin']:
        raise ValueError('Unsafe or unexpected asset paths')
    if len(manifest_bytes(value)) > 65536:
        raise ValueError('Manifest exceeds engine byte limit')
    return value


def _qwen_descriptor(value):
    from worker.engines.qwen import validate_declarations
    validate_declarations(value['execution'], value['language'])
    if any(l not in ('English', 'French') for l in value['execution']['languages']):
        raise ValueError('Explicit managed Qwen languages required')
    if len(value['checkpoint']) > 256 or not re.fullmatch('[0-9a-f]{40}', value['revision']):
        raise ValueError('Immutable Qwen revision required')
    origins = value['allowed_origins']
    if type(origins) is not list or not 1 <= len(origins) <= 256 or any(type(o) is not str for o in origins) or len(set(origins)) != len(origins):
        raise ValueError('Explicit HTTPS authorities required')
    def authority(url):
        p = urlsplit(url)
        if p.scheme != 'https' or not p.hostname or p.username or p.password or p.query or p.fragment:
            raise ValueError('Unsafe Qwen origin')
        return 'https://' + p.netloc
    if any(authority(o) != o for o in origins): raise ValueError('HTTPS authorities required')
    names = set(); urls = set(); total = 0
    for entry in value['files']:
        name = entry['path']
        if len(name) > 512 or any(not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}', part) for part in name.split('/')) or Path(name).suffix not in ('.json','.safetensors','.txt','.model') or name in names:
            raise ValueError('Unsafe Qwen snapshot path')
        if (name.endswith('.json') and entry['bytes'] > 16*1024**2) or entry['bytes'] > 8*1024**3 or authority(entry['origin']) not in origins or entry['origin'] in urls or entry['revision'] != value['revision']:
            raise ValueError('Qwen asset identity or bound')
        names.add(name); urls.add(entry['origin']); total += entry['bytes']
    from lib.tts_backends.assets import qwen_tokenizer_layout
    qwen_tokenizer_layout(names, value['checkpoint'])
    if total > 16*1024**3 or not {'config.json','generation_config.json','speech_tokenizer/config.json','speech_tokenizer/preprocessor_config.json'} <= names or not any(n.startswith('speech_tokenizer/') and n.endswith('.safetensors') for n in names) or not any('/' not in n and n.endswith('.safetensors') for n in names):
        raise ValueError('Incomplete bounded Qwen snapshot')
    if any(any(n.startswith(other + '/') for other in names) for n in names): raise ValueError('File/directory conflict')


def _verified_managed(root, expected):
    if expected['backend'] == 'qwen':
        from worker.engines.qwen import verify_bundle
        return verify_bundle(root, manifest_sha256(expected), expected['checkpoint'], expected['revision'], expected['language'])
    return verified_bundle(root, expected['backend'])


def _pocket_config(value):
    """Compile explicit operator config; JSON is a safe YAML subset.

    Origins are exact HTTPS authorities, without credentials or URL parameters.
    No package defaults, hub lookup, model discovery or language inference occurs.
    """
    origins = value['allowed_origins']
    if type(origins) is not list or not origins or len(origins) > 128 or not all(isinstance(o, str) for o in origins) or len(set(origins)) != len(origins):
        raise ValueError('Explicit allowed HTTPS origins required')
    def authority(url):
        if not isinstance(url, str):
            raise ValueError('HTTPS origin required')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Unsafe HTTPS origin')
        return 'https://' + parsed.netloc
    if any(authority(o) != o for o in origins):
        raise ValueError('Allowed origins must be HTTPS authorities')
    if len(value['checkpoint']) > 256 or not re.fullmatch('[0-9a-f]{40}', value['revision']):
        raise ValueError('Immutable Pocket revision required')
    names = set()
    remote = {}
    total = 0
    for entry in value['files']:
        name = entry['path']
        if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}', name) or Path(name).suffix not in ('.safetensors', '.yaml', '.json') or name in names:
            raise ValueError('Unsafe or duplicate Pocket asset')
        names.add(name)
        if authority(entry['origin']) not in origins or entry['origin'] in remote or entry['revision'] != value['revision']:
            raise ValueError('Pocket origin or revision mismatch')
        remote[entry['origin']] = name
        total += entry['bytes']
        if entry['bytes'] > 2 * 1024**3 or total > 4 * 1024**3:
            raise ValueError('Pocket asset byte limit')
    if 'config.yaml' not in names or not any(n.endswith('.safetensors') for n in names):
        raise ValueError('Pocket config and safetensors required')
    config = value['config']
    if type(config) is not dict or not config or not isinstance(config.get('weights_path'), str):
        raise ValueError('Explicit cloning weights config required')
    def localize(item, key='', depth=0):
        if depth > 32:
            raise ValueError('Config nesting limit')
        if key.endswith('_path'):
            if item is None and key != 'weights_path':
                return None
            if not isinstance(item, str):
                raise ValueError('Config path must be a string')
            name = remote.get(item, item)
            if name not in names or name == 'config.yaml':
                raise ValueError('Config path outside authorized assets')
            if key == 'weights_path' and not name.endswith('.safetensors'):
                raise ValueError('Cloning requires safetensors')
            return name
        if type(item) is dict:
            if not all(isinstance(k, str) for k in item):
                raise ValueError('Config keys must be strings')
            return {k: localize(v, k, depth+1) for k, v in item.items()}
        if type(item) is list:
            return [localize(v, depth=depth+1) for v in item]
        if isinstance(item, str) and (':' in item or '/' in item or '\\' in item):
            raise ValueError('Unbound config origin or path')
        if item is not None and type(item) not in (str, int, float, bool):
            raise ValueError('Invalid config value')
        if type(item) is float and not math.isfinite(item):
            raise ValueError('Invalid config number')
        return item
    if config.get('language', value['language']) != value['language'] or 'weights_path_without_voice_cloning' in config:
        raise ValueError('Inconsistent language or non-cloning config')
    raw = manifest_bytes(localize(config))
    entry = next(e for e in value['files'] if e['path'] == 'config.yaml')
    if len(raw) > 65536 or len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('Generated config differs from operator pins')
    return raw


def bundle_manifest(expected):
    """Descriptor-only policy/config never enters the runtime manifest."""
    return {k: v for k, v in expected.items() if k not in ('config', 'allowed_origins')}


class _AllowedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, origins, public_github_assets=False):
        self.origins = origins
        self.public_github_assets=public_github_assets

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        signed_asset=self.public_github_assets and parsed.hostname=='release-assets.githubusercontent.com'
        if parsed.scheme != 'https' or parsed.username or parsed.password or (parsed.query and not signed_asset) or parsed.fragment or 'https://' + parsed.netloc not in self.origins:
            raise ValueError('Redirect outside allowed HTTPS origins')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_stream(url, target, expected_bytes, allowed_origins=None, public_github_assets=False):
    # At most expected bytes reach disk; one extra byte detects overflow.
    if allowed_origins is None:open_url=urllib.request.urlopen
    else:
        handlers=[_AllowedRedirects(allowed_origins,True) if public_github_assets else _AllowedRedirects(allowed_origins)]
        if public_github_assets:handlers.insert(0,urllib.request.ProxyHandler({}))
        open_url=urllib.request.build_opener(*handlers).open
    with open_url(url, timeout=10) as source, target.open('xb') as sink:
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


def _transfer_child(url, target, size, allowed_origins=None, public_github_assets=False):
    try:
        if public_github_assets:_download_stream(url,target,size,allowed_origins,True)
        else:_download_stream(url,target,size,allowed_origins)
    except BaseException:
        raise SystemExit(1)


def _budget(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Finite positive wait budget required')


def download(url, target, expected_bytes=None, *, timeout=300, cancelled=lambda: False, allowed_origins=None, public_github_assets=False):
    """Managed transfer has a wall deadline including connect/read, and cancellation.

    A disposable POSIX child owns blocking network I/O. Deadline/cancel terminates
    then kills/reaps it; no background transfer retains the target or global lock.
    Legacy callers have a finite 4 GiB cap, without an exact length requirement.
    """
    _budget(timeout)
    if type(public_github_assets) is not bool:raise ValueError('Explicit public GitHub transfer policy required')
    if public_github_assets:
        initial=urlsplit(url)
        if (set(allowed_origins or ())!={'https://github.com','https://release-assets.githubusercontent.com'} or
                expected_bytes is None or initial.scheme!='https' or initial.netloc!='github.com' or initial.query or initial.fragment or initial.username or initial.password):
            raise ValueError('Closed public GitHub asset origins required')
    if expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes <= 0):
        raise ValueError('Positive expected byte count required')
    if expected_bytes is None:
        # Legacy acquisition retains EOF semantics under a finite cap.
        expected_bytes = 4 * 1024**3
        worker = _legacy_transfer_child
    else:
        worker = _transfer_child
    args = (url, target, expected_bytes) if worker is _legacy_transfer_child else (url, target, expected_bytes, allowed_origins)
    if public_github_assets:args=(*args,True)
    process = multiprocessing.get_context('fork').Process(target=worker, args=args)
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
    for path in bundle.rglob('*'):
        if path.is_dir(): continue
        _file(path)
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        path.chmod(0o444)
    for path in sorted(bundle.rglob('*'), reverse=True):
        if path.is_dir(): path.chmod(0o555)
    bundle.chmod(0o555)
    if bundle.stat().st_mode & 0o222 or any(p.stat().st_mode & 0o222 for p in bundle.rglob('*')):
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
    manifest_expected = bundle_manifest(expected)
    backend = expected['backend']
    config_bytes = _pocket_config(expected) if backend == 'pocket-tts' else None
    names = {e['path'] for e in expected['files']}
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
            for path in bundle.rglob('*'):
                if path.is_symlink(): raise ValueError('Symlink bundle entry')
                if not path.is_dir(): _file(path)
            _, manifest = _verified_managed(bundle, expected)
            if {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()} != names | {'manifest.json'}:
                raise ValueError('Unexpected immutable bundle entry')
            if manifest != manifest_expected or (bundle / 'manifest.json').read_bytes() != manifest_bytes(manifest_expected):
                raise ValueError('Immutable bundle identity mismatch')
        else:
            _directory(stage)
            allowed = names | {n + '.part' for n in names} | {'manifest.json'}
            directories = {str(p) for n in names for p in Path(n).parents if str(p) != '.'}
            if any(p.is_symlink() or (str(p.relative_to(stage)) not in (directories if p.is_dir() else allowed)) for p in stage.rglob('*')):
                raise ValueError('Unsafe staging entry')
            for path in stage.rglob('*'):
                if not path.is_dir(): _file(path)
            for entry in expected['files']:
                target = stage / entry['path']
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                partial = stage / (entry['path'] + '.part')
                if partial.exists():
                    partial.unlink()
                if target.exists():
                    if target.stat().st_size == entry['bytes'] and sha256(target) == entry['sha256']:
                        continue
                    target.unlink()
                # Injected callables are trusted and responsible for their own bounds.
                if entry['path'] == 'config.yaml' and config_bytes is not None:
                    partial.write_bytes(config_bytes)
                elif downloader is download:
                    downloader(entry['origin'], partial, entry['bytes'], timeout=transfer_timeout, cancelled=cancelled, **({'allowed_origins': expected['allowed_origins']} if 'allowed_origins' in expected else {}))
                else:
                    downloader(entry['origin'], partial)
                _file(partial)
                if partial.is_symlink() or not partial.is_file() or partial.stat().st_size != entry['bytes'] or sha256(partial) != entry['sha256']:
                    raise ValueError('Downloaded asset differs from expected pins')
                os.replace(partial, target)
            manifest_path = stage / 'manifest.json'
            if manifest_path.exists():
                manifest_path.unlink()
            manifest_path.write_bytes(manifest_bytes(manifest_expected))
            _verified_managed(stage, expected)
            for path in stage.rglob('*'):
                if path.is_dir(): continue
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
