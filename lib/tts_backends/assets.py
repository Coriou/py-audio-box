"""Local, hash-verified bundles; provisioning is a separate network step."""
import hashlib
import json
from pathlib import Path
from .contract import BackendError


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_bundle(directory, backend):
    if backend == 'qwen':
        # Deliberate backend contract: upstream names, no global fr/en widening.
        from worker.engines.qwen import verify_bundle
        directory = Path(directory).absolute()
        from worker.engines.pocket import _read
        raw = _read(directory / 'manifest.json', 65536)
        manifest = json.loads(raw)
        return verify_bundle(directory, hashlib.sha256(raw).hexdigest(), manifest['checkpoint'], manifest['revision'], manifest['language'])
    directory = Path(directory).absolute()
    root = directory.resolve()
    try:
        if directory.is_symlink() or (root / 'manifest.json').is_symlink():
            raise ValueError('Symlink bundle refused')
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest['backend'] != backend or (type(manifest['schema_version']) is not int or manifest['schema_version'] != 1):
            raise ValueError('Wrong backend or bundle schema')
        if not manifest['files']:
            raise ValueError('Empty asset bundle')
        if not isinstance(manifest.get('checkpoint'), str) or manifest.get('language') not in ('fr', 'en'):
            raise ValueError('Missing checkpoint or supported language')
        names = set()
        for entry in manifest['files']:
            name = entry['path']
            if not isinstance(name, str) or not name or Path(name).is_absolute() or any(part in ('.', '..') for part in name.split('/')) or name in names:
                raise ValueError('Unsafe or duplicate asset path')
            names.add(name)
            candidate = root / name
            if any(p.is_symlink() for p in (candidate, *list(candidate.parents)[:len(Path(name).parts)-1])):
                raise ValueError('Symlink asset refused')
            path = (root / entry['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing or unsafe asset path')
            if 'bytes' in entry and (type(entry['bytes']) is not int or path.stat().st_size != entry['bytes']):
                raise ValueError('Asset size mismatch')
            if sha256(path) != entry['sha256']:
                raise ValueError('Asset checksum mismatch')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BackendError('invalid_assets', 'Provision a complete, verified local bundle', reason=str(exc)) from exc
    return root, manifest
