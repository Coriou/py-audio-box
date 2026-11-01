import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('token', [None, 'synthetic-wrapper-token-r512b'])
def test_wrapper_network_and_credentials_policy(tmp_path, token):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / 'run-cpu', tmp_path / 'run-cpu')
    docker_bin = root / 'tests' / 'fixtures' / 'ci' / 'docker-bin'
    env = dict(os.environ, PATH=str(docker_bin) + ':' + os.environ['PATH'])
    env.pop('HF_TOKEN', None)
    if token is not None:
        env['HF_TOKEN'] = token
    for action in ('speak', 'benchmark', 'export'):
        result = subprocess.run(['bash', str(tmp_path / 'run-cpu'), 'kokoro', action, '--help'], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        args = result.stdout.splitlines()
        assert args[args.index('--network') + 1] == 'none'
        assert 'HF_TOKEN' not in args
        assert not any('HF_TOKEN' in arg for arg in args)
        assert token is None or token not in result.stdout
        assert any(arg.endswith('cache:/cache:ro') for arg in args)
    result = subprocess.run(['bash', str(tmp_path / 'run-cpu'), 'pocket-tts', 'provision', '--help'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert '--network' not in args
    assert any(arg.endswith('cache:/cache:rw') for arg in args)
    if token is None:
        assert 'HF_TOKEN' not in args
    else:
        assert args.count('HF_TOKEN') == 1
        assert args[args.index('HF_TOKEN') - 1] == '-e'
        assert [arg for arg in args if 'HF_TOKEN' in arg] == ['HF_TOKEN']
        assert token not in result.stdout
