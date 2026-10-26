#!/usr/bin/env python3
"""Native dependency imports and an optional, isolated tensor probe; never loads weights."""
import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

EXPECTED = {'qwen-tts': '0.1.1', 'torch': '2.10.0', 'torchaudio': '2.10.0',
            'transformers': '4.57.3', 'tokenizers': '0.22.2'}
MAX_REPORT_BYTES = 10 * 1024 * 1024


def validate_platform():
    if (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
        raise RuntimeError('requires native Darwin arm64')
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('requires CPython 3.11')
    if platform.python_implementation() != 'CPython':
        raise RuntimeError('requires CPython')


def validate_versions(versions):
    for name, expected in EXPECTED.items():
        if versions.get(name) != expected:
            raise RuntimeError(f'{name}: expected {expected}, got {versions.get(name)}')


def write_report(path, report):
    # Only a caller-selected JSON file, never a directory/glob of workspace data.
    path = Path(path)
    if path.suffix != '.json' or path.is_symlink():
        raise ValueError('report must be a non-symlink .json path')
    data = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode()
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError('report exceeds 10 MiB')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(data)


def deps_check():
    validate_platform()
    versions = {name: metadata.version(name) for name in EXPECTED}
    validate_versions(versions)
    for name in ('torch', 'torchaudio', 'transformers', 'tokenizers', 'qwen_tts'):
        importlib.import_module(name)
    import torch
    from qwen_tts import Qwen3TTSModel
    if not callable(Qwen3TTSModel.from_pretrained):
        raise RuntimeError('Qwen loader missing')
    # CPU tensor operation, no model construction or weight acquisition.
    value = torch.tensor([1., 2.], device='cpu') + 1
    if value.device.type != 'cpu' or value.tolist() != [2., 3.]:
        raise RuntimeError('CPU tensor validation failed')
    return {'platform': platform.system(), 'machine': platform.machine(),
            'python': platform.python_version(), 'versions': versions,
            'mps_built': torch.backends.mps.is_built(),
            'mps_available': torch.backends.mps.is_available(),
            'cpu_tensor': 'passed', 'models_loaded': False}


def mps_tensor_probe():
    validate_platform()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('MPS fallback must be disabled before importing torch')
    if os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is not None:
        raise RuntimeError('use the conservative default MPS memory watermark')
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no CPU fallback')
    value = torch.tensor([1., 2.], device='mps')
    result = value * 2 + 1
    torch.mps.synchronize()
    if result.device.type != 'mps' or result.cpu().tolist() != [3., 5.]:
        raise RuntimeError('MPS device/value validation failed')
    return {'mps_tensor': 'passed', 'device': str(result.device),
            'models_loaded': False, 'metal_synthesis_qualified': False}


def bounded_probe(report):
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0')
    env.pop('PYTORCH_MPS_HIGH_WATERMARK_RATIO', None)
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                              '_mps-child', '--report', str(report)],
                             env=env, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        code = child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            child.wait()  # Retain ownership until the exact child is reaped.
        raise RuntimeError('MPS probe exceeded 30 seconds; child killed and reaped')
    if code != 0:
        raise RuntimeError(f'MPS probe child failed (exit {code}); see JSON report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('deps-check', 'mps-probe', '_mps-child'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == 'mps-probe':
            validate_platform()
            bounded_probe(args.report)
            return
        report = deps_check() if args.mode == 'deps-check' else mps_tensor_probe()
        write_report(args.report, dict(status='passed', mode=args.mode, **report))
    except Exception as exc:
        write_report(args.report, {'status': 'failed', 'mode': args.mode,
                                   'error': str(exc)[:2000]})
        raise


if __name__ == '__main__':
    main()
