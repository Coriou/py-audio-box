#!/usr/bin/env python3
"""Native dependency imports and an optional, isolated tensor probe; never loads weights."""
import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import asyncio
import socket
import tempfile
import time
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


def sox_version(sox):
    banner = subprocess.check_output([sox, '--version'], timeout=10, text=True).strip()
    if re.search(r'\bSoX v?\d+(?:\.\d+)+', banner):
        return banner
    # Some Homebrew builds print only "SoX v". Record the installed package,
    # not a claimed pin: brew install sox deliberately follows Homebrew.
    brew = shutil.which('brew')
    if brew:
        try:
            package = subprocess.check_output([brew, 'list', '--versions', 'sox'],
                                              timeout=10, text=True).strip()
        except (subprocess.SubprocessError, OSError):
            package = ''
        if re.fullmatch(r'sox \d[\w. +_-]*', package):
            return 'Homebrew package: ' + package
    return 'unknown (SoX banner contains no version)'


def deps_check():
    validate_platform()
    sox = shutil.which('sox')
    if not sox:
        raise RuntimeError('SoX executable missing; provision brew install sox')
    installed_sox_version = sox_version(sox)
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
            'sox_executable': sox, 'sox_version': installed_sox_version,
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



def require_client_dependencies():
    # Deliberately fail on absence; never substitute modules or skip a smoke.
    for name in ('fastapi', 'mcp', 'uvicorn', 'jsonschema', 'httpx'):
        importlib.import_module(name)


def owned_run(argv, *, env, cwd, timeout=300):
    child = subprocess.Popen(argv, env=env, cwd=cwd, start_new_session=True)
    try:
        code = child.wait(timeout=timeout)
        if code:
            raise RuntimeError(f'{argv[:3]} failed (exit {code}); see step log')
    finally:
        reap(child)


def reap(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    else:
        child.wait()


def client_smoke():
    # The caller owns this fresh result path across child failure/timeout.
    path = os.environ.get('APPLE_CLIENT_RESULT')
    progress = {'fixture_calls': 'failed', 'protocol': 'not_run'}
    def persist():
        if path:
            write_report(path, progress)
    persist()
    try:
        _client_smoke(progress, persist)
        return progress
    finally:
        persist()


def _client_smoke(progress, persist):
    require_client_dependencies()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import httpx
    root = Path(__file__).resolve().parents[1]
    # No inherited credentials, proxy settings, upload grants or model caches.
    with tempfile.TemporaryDirectory(prefix='apple-client-') as private:
        env = {k: v for k, v in os.environ.items()
               if k in ('PATH', 'SYSTEMROOT', 'TMPDIR')}
        env.update(HOME=private, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen(16)
            env['SPEECH_URL'] = f'http://127.0.0.1:{listener.getsockname()[1]}'
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                      '_fixture', '--fd', str(listener.fileno())],
                                     cwd=private, env=env, pass_fds=(listener.fileno(),),
                                     start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            with httpx.Client(timeout=1, trust_env=False) as http:
                while True:
                    if child.poll() is not None:
                        raise RuntimeError('fixture exited before readiness')
                    try:
                        response = http.get(env['SPEECH_URL'] + '/v1/capabilities')
                        response.raise_for_status(); expected = response.json(); break
                    except httpx.HTTPError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError('fixture readiness exceeded 15 seconds')
                        time.sleep(.1)
            cli = subprocess.run([sys.executable, '-m', 'clients.cli', 'discover'],
                                 env=env, cwd=private, timeout=15,
                                 check=True, capture_output=True, text=True)
            if json.loads(cli.stdout) != expected:
                raise RuntimeError('CLI fixture response mismatch')
            progress['fixture_calls'] = 'passed'
            progress['protocol'] = 'failed'
            persist()
            async def protocol():
                params = StdioServerParameters(command=sys.executable,
                    args=['-m', 'clients.mcp'], env=env, cwd=private)
                async with stdio_client(params) as streams:
                    async with ClientSession(*streams) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        if 'speech_discover' not in {tool.name for tool in tools.tools}:
                            raise RuntimeError('discover tool absent')
                        result = await session.call_tool('speech_discover', {})
                        if result.isError or result.structuredContent != expected:
                            raise RuntimeError('MCP fixture response mismatch')
            async def bounded_protocol():
                await asyncio.wait_for(protocol(), timeout=30)
            asyncio.run(bounded_protocol())
            progress['protocol'] = 'passed'
            persist()
        finally:
            reap(child)


def clients_check(path):
    root = Path(__file__).resolve().parents[1]
    report = dict(status='failed', commit=os.environ.get('GITHUB_SHA', 'unknown'),
        runner_image=os.environ.get('ImageOS', 'unknown'),
        runner_version=os.environ.get('ImageVersion', 'unknown'),
        os=platform.platform(), arch=platform.machine(), python=platform.python_version(),
        cli_install='not_run', mcp_install='not_run', cli_help='not_run',
        voice_clients='not_run', protocol='not_run', fixture_calls='not_run',
        runtime_qualified=False, versions={})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    current = None
    try:
        validate_platform()
        for label, directory, lock in (
                ('cli_install', '.venv-client', 'requirements/client.lock'),
                ('mcp_install', '.venv-service', 'service/requirements.lock')):
            current = label
            target = root / directory
            if target.exists():
                raise RuntimeError(f'{directory} must be fresh')
            owned_run([sys.executable, '-m', 'venv', str(target)], env=env, cwd=root, timeout=60)
            python = str(target / 'bin/python')
            owned_run([python, '-m', 'pip', 'install', '--only-binary=:all:',
                       '--require-hashes', '--index-url', 'https://pypi.org/simple',
                       '-r', lock], env=env, cwd=root, timeout=300)
            owned_run([python, '-m', 'pip', 'check'], env=env, cwd=root, timeout=30)
            report[label] = 'passed'
            if label == 'cli_install':
                current = 'cli_help'
                owned_run([python, '-m', 'clients.cli', '--help'], env=env, cwd=root, timeout=15)
                report[current] = 'passed'
        current = None
        raw = subprocess.check_output([python, '-c',
            "import json; from importlib.metadata import version; print(json.dumps({n:version(n) for n in ('mcp','fastapi','httpx','jsonschema','uvicorn')}))"],
            env=env, cwd=root, timeout=15, text=True)
        report['versions'] = json.loads(raw)
        current = 'voice_clients'
        owned_run([python, '-m', 'unittest', 'tests.clients.test_voice_clients', '-v'], env=env, cwd=root, timeout=120)
        report[current] = 'passed'
        current = None
        with tempfile.TemporaryDirectory(prefix='apple-client-result-') as private:
            result_path = Path(private) / 'stages.json'
            smoke_env = dict(env, APPLE_CLIENT_RESULT=str(result_path))
            try:
                owned_run([python, '-m', 'unittest',
                           'tests.test_ci_apple_silicon.MacClientSmoke', '-v'],
                          env=smoke_env, cwd=root, timeout=60)
            finally:
                if result_path.exists():
                    stages = json.loads(result_path.read_text())
                    for stage in ('fixture_calls', 'protocol'):
                        if stages.get(stage) not in ('passed', 'failed', 'not_run'):
                            raise RuntimeError(f'invalid smoke stage: {stage}')
                        report[stage] = stages[stage]
            if any(report[stage] != 'passed' for stage in ('fixture_calls', 'protocol')):
                raise RuntimeError('smoke did not report both stages passed')
        report['status'] = 'passed'
    except Exception as exc:
        if current:
            report[current] = 'failed'
        report['error'] = str(exc)[:2000]
        raise
    finally:
        write_report(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('deps-check', 'mps-probe', '_mps-child', 'clients-check', '_fixture'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--fd', type=int)
    args = parser.parse_args()
    if args.mode == '_fixture':
        import uvicorn
        from service.api import app
        listener = socket.socket(fileno=args.fd)
        uvicorn.Server(uvicorn.Config(app, log_level='warning')).run(sockets=[listener])
        return
    if args.report is None:
        parser.error('--report is required')
    if args.mode == 'clients-check':
        clients_check(args.report)
        return
    try:
        if args.mode == 'mps-probe':
            validate_platform()
            bounded_probe(args.report)
            return
        report = deps_check() if args.mode == 'deps-check' else mps_tensor_probe()
        write_report(args.report, dict(status='passed', mode=args.mode, **report))
    except Exception as exc:
        write_report(args.report, {'status': 'failed', 'mode': args.mode,
                                   'platform': platform.system(), 'machine': platform.machine(),
                                   'python': platform.python_version(),
                                   'error': str(exc)[:2000]})
        raise


if __name__ == '__main__':
    main()
