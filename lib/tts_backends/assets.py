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
    root = Path(directory).resolve()
    try:
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest['backend'] != backend or manifest['schema_version'] != 1:
            raise ValueError('Wrong backend or bundle schema')
        if not manifest['files']:
            raise ValueError('Empty asset bundle')
        if not isinstance(manifest.get('checkpoint'), str) or manifest.get('language') not in ('fr', 'en'):
            raise ValueError('Missing checkpoint or supported language')
        for entry in manifest['files']:
            path = (root / entry['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing or unsafe asset path')
            if sha256(path) != entry['sha256']:
                raise ValueError('Asset checksum mismatch')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BackendError('invalid_assets', 'Provision a complete, verified local bundle', reason=str(exc)) from exc
    return root, manifest
