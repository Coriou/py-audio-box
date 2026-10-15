import os
from pathlib import Path
import shutil
import subprocess


def test_wrapper_network_and_credentials_policy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / 'run-cpu', tmp_path / 'run-cpu')
    docker = tmp_path / 'docker'
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'])
    env.pop('HF_TOKEN', None)
    for action in ('speak', 'benchmark', 'export'):
        result = subprocess.run(['bash', str(tmp_path / 'run-cpu'), 'kokoro', action, '--help'], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        args = result.stdout.splitlines()
        assert args[args.index('--network') + 1] == 'none'
        assert 'HF_TOKEN' not in args
        assert any(arg.endswith('cache:/cache:ro') for arg in args)
    result = subprocess.run(['bash', str(tmp_path / 'run-cpu'), 'pocket-tts', 'provision', '--help'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--network' not in result.stdout.splitlines()
    assert 'HF_TOKEN' not in result.stdout.splitlines()
