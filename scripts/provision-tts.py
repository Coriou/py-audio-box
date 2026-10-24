#!/usr/bin/env python3
"""Network-authorized preparation; never run as part of synthesis."""
import argparse
import importlib.metadata
import importlib.resources
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from service.provisioning import provision as managed_provision, download
from tts_backends.assets import sha256, verified_bundle


def provision(backend, dest, language):
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Choose an empty bundle directory; existing assets are never overwritten')
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    def record(path, origin, revision=None):
        files.append({'path': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                      'origin': origin, 'revision': revision})
    if backend == 'kokoro':
        release = 'model-files-v1.0'
        for remote, local in [('kokoro-v1.0.onnx', 'model.onnx'), ('voices-v1.0.bin', 'voices.bin')]:
            url = f'https://github.com/thewh1teagle/kokoro-onnx/releases/download/{release}/{remote}'
            path = dest / local
            download(url, path)
            pins = json.loads((Path(__file__).resolve().parents[1] / 'requirements' / 'tts-assets.json').read_text())
            if sha256(path) != pins['kokoro'][local]:
                raise ValueError('Upstream Kokoro asset differs from the reviewed pin; do not promote this bundle')
            record(path, url, release)
        checkpoint = 'Kokoro-82M-v1.0-onnx-fp32'
        license_name = 'Apache-2.0 weights; MIT runtime; verify selected voice provenance'
    else:
        import yaml
        from huggingface_hub import hf_hub_download
        name = {'fr': 'french', 'en': 'english'}[language]
        source = importlib.resources.files('pocket_tts') / 'config' / f'{name}.yaml'
        config = yaml.safe_load(source.read_text())
        config.pop('weights_path_without_voice_cloning', None)
        def localize(value):
            if isinstance(value, dict):
                return {key: localize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [localize(item) for item in value]
            if isinstance(value, str) and value.startswith('hf://'):
                match = re.fullmatch(r'hf://([^/]+/[^/]+)/(.+)@([0-9a-f]{40})', value)
                if match is None:
                    raise ValueError('Pocket assets must have an immutable upstream revision')
                repo, filename, revision = match.groups()
                cached = hf_hub_download(repo, filename, revision=revision)
                path = dest / f'{len(files):02d}-{Path(filename).name}'
                import shutil
                shutil.copyfile(cached, path)
                record(path, value, revision)
                # Config paths are resolved from the config directory by Pocket.
                return path.name
            return value
        config = localize(config)
        path = dest / 'config.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        record(path, f'pocket-tts:{importlib.metadata.version("pocket-tts")}/{name}.yaml')
        checkpoint = f'pocket-tts-{name}-' + files[0]['revision']
        license_name = 'CC-BY-4.0 weights; MIT code; access conditions must be accepted by the owner'
    manifest = {'schema_version': 1, 'backend': backend, 'language': language,
                'checkpoint': checkpoint, 'license': license_name, 'files': files,
                'versions': {p: importlib.metadata.version(p) for p in
                             (['kokoro-onnx', 'onnxruntime', 'espeakng-loader', 'phonemizer'] if backend == 'kokoro' else ['pocket-tts', 'torch', 'tokenizers'])}}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verified_bundle(dest, backend)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['kokoro', 'pocket-tts'], required=True)
    parser.add_argument('--language', choices=['fr', 'en'], required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--expected-manifest', type=Path, help='Operator-reviewed expected Kokoro manifest; --dest becomes managed storage root')
    args = parser.parse_args()
    try:
        if args.expected_manifest:
            expected = json.loads(args.expected_manifest.read_text())
            if expected['backend'] != args.backend or expected['language'] != args.language:
                raise ValueError('Expected manifest differs from CLI selection')
            versions = {p: importlib.metadata.version(p) for p in expected['versions']}
            print(json.dumps({'bundle': str(managed_provision(args.dest, expected, runtime_versions=versions))}))
        else:
            print(json.dumps(provision(args.backend, args.dest, args.language)))
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': type(exc).__name__, 'message': str(exc)}), file=sys.stderr)
        sys.exit(1)
