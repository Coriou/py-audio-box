"""Model-free cache retrieval across fresh interpreters using the real loader."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
HF_KEYS = ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE')


def worker_probe(base, mode):
    """Executed in a fresh interpreter, with acquisition/model loading injected."""
    import importlib.abc
    import importlib.util
    import types
    from lib.tts_backends.qwen import _worker_impl

    base = Path(base)
    before = dict(os.environ)
    calls = []
    roots = []
    class Loader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'lib.tts':
                return importlib.util.spec_from_loader(fullname, self)
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            root = Path(os.environ['HOME']).parent
            roots.append(root)
            for key in ('HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'TMP', 'TEMP',
                        'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_HOME'):
                assert Path(os.environ[key]).is_relative_to(root), key
            assert tempfile.gettempdir() == os.environ['TMPDIR']
            assert os.environ['USER'] == os.environ['LOGNAME'] == 'qwen-worker'
            if mode == 'service':
                for key in HF_KEYS:
                    assert Path(os.environ[key]).is_relative_to(root), key
                assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
            cache = Path(os.environ['HF_HUB_CACHE'])
            def snapshot_download(name, revision, local_files_only):
                assert mode != 'service', 'service attempted model acquisition'
                target = cache / 'models--Qwen--Qwen3-TTS-12Hz-0.6B-Base' / 'snapshots' / 'pinned'
                marker = target / 'config.json'
                if not marker.exists():
                    assert mode == 'cold', 'warm/offline worker attempted a download'
                    target.mkdir(parents=True)
                    marker.write_text('{}')
                    with (base / 'downloads').open('a') as stream:
                        stream.write('download\n')
                calls.append(dict(cache=str(cache), offline=local_files_only))
                return str(target)
            def from_pretrained(source, **kwargs):
                assert Path(source).is_dir()
                assert kwargs['local_files_only'] == (mode in ('offline', 'service'))
                if mode == 'service':
                    assert source == str(base / 'immutable')
                    assert kwargs['device_map'] == 'cpu'
                calls.append(dict(source=source, **kwargs))
                return types.SimpleNamespace()
            sys.modules['huggingface_hub'] = types.SimpleNamespace(snapshot_download=snapshot_download)
            tree = ast.parse((ROOT / 'lib/tts.py').read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_load_tts_model_local')
            module.__dict__.update(Path=Path, torch=Mock(), _DTYPE_MAP={'float32': 'fp32'},
                get_device=lambda: 'cpu', best_dtype=lambda *_: 'fp32',
                _qwen_model_class=lambda: types.SimpleNamespace(from_pretrained=from_pretrained))
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'loader', 'exec'), module.__dict__)
    class Connection:
        def __init__(self):
            self.messages = []
            self.closed = False
        def send(self, message):
            self.messages.append(message)
        def recv(self):
            raise EOFError
        def close(self):
            self.closed = True
    sys.meta_path.insert(0, Loader())
    settings = dict(model_name='Qwen/Qwen3-TTS-12Hz-0.6B-Base', num_threads=2,
                    dtype_str='float32', revision='pinned', offline=mode in ('offline', 'service'))
    if mode == 'service':
        settings.update(model_name=str(base / 'immutable'), device='cpu', attention='sdpa')
    else:
        settings['legacy_hf_cache'] = True
    connection = Connection()
    _worker_impl(connection, settings)
    assert connection.closed
    assert connection.messages[0] == ('ready', {'revision': 'pinned'}), connection.messages
    assert connection.messages[1] == ('error', ('EOFError', '')), connection.messages
    assert all(not root.exists() for root in roots)
    # Offline flags are process-scoped in production; all path overrides restore.
    for key in (*HF_KEYS, 'HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'USER', 'LOGNAME'):
        assert os.environ.get(key) == before.get(key), key
    (base / ('result-' + mode)).write_text(json.dumps(calls))


class LegacyCacheTests(unittest.TestCase):
    def probe(self, base, mode, environment):
        result = subprocess.run([sys.executable, '-B', '-c',
            'from tests.test_qwen_legacy_cache import worker_probe; '
            'import sys; worker_probe(sys.argv[1], sys.argv[2])', str(base), mode],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads((base / ('result-' + mode)).read_text())

    def test_persistent_cache_across_separate_workers(self):
        for policy in ('hf_home', 'hub', 'old_hub', 'xdg', 'home'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                env = dict(os.environ)
                for key in (*HF_KEYS, 'XDG_CACHE_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
                    env.pop(key, None)
                env['HOME'] = str(base / 'home')
                expected = base / 'home' / '.cache' / 'huggingface' / 'hub'
                if policy == 'hf_home':
                    env['HF_HOME'] = str(base / 'hf')
                    expected = base / 'hf' / 'hub'
                elif policy in ('hub', 'old_hub'):
                    env['HF_HOME'] = str(base / 'hf')
                    env['HUGGINGFACE_HUB_CACHE'] = str(base / 'old-hub')
                    expected = base / 'old-hub'
                    if policy == 'hub':
                        env['HF_HUB_CACHE'] = str(base / 'hub')
                        expected = base / 'hub'
                elif policy == 'xdg':
                    env['XDG_CACHE_HOME'] = str(base / 'xdg')
                    expected = base / 'xdg' / 'huggingface' / 'hub'
                for mode in ('cold', 'warm', 'offline'):
                    calls = self.probe(base, mode, env)
                    self.assertEqual(calls[0], dict(cache=str(expected), offline=mode == 'offline'))
                    self.assertTrue(Path(calls[1]['source']).is_dir())
                self.assertEqual((base / 'downloads').read_text(), 'download\n')

    def test_service_local_offline_cache_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'immutable').mkdir()
            (base / 'immutable' / 'config.json').write_text('{}')
            env = dict(os.environ)
            for key in (*HF_KEYS, 'HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'USER', 'LOGNAME'):
                env[key] = str(base / 'inherited-must-not-be-created')
            calls = self.probe(base, 'service', env)
            self.assertEqual(len(calls), 1)
            self.assertFalse((base / 'inherited-must-not-be-created').exists())
            self.assertFalse((base / 'downloads').exists())

    def test_only_legacy_loader_opts_in(self):
        tree = ast.parse((ROOT / 'lib/tts.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_tts_model')
        env = dict(QwenBackend=Mock(), os=Mock(), env_flag_enabled=Mock(return_value=False))
        env['os'].getenv.return_value = None
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'loader', 'exec'), env)
        env['load_tts_model']('model', 2)
        self.assertTrue(env['QwenBackend'].call_args.kwargs['legacy_hf_cache'])
        env['load_tts_model']('local', 2, revision='r', offline=True, device='cpu', attention='sdpa')
        self.assertFalse(env['QwenBackend'].call_args.kwargs['legacy_hf_cache'])


if __name__ == '__main__':
    unittest.main()
