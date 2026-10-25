"""Source/inert checks: never import the ML loader or start a backend."""
import ast
import json
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch
import sys

ROOT = Path(__file__).resolve().parents[2]


class LegacyControlsTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse((ROOT / 'lib/tts.py').read_text())

    def function(self, name, env):
        node = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'inert-loader', 'exec'), env)
        return env[name]

    def test_recorded_constants_and_environment_locations(self):
        ledger = json.loads((ROOT / 'contracts/v1/legacy-controls.json').read_text())
        entry = next(e for e in ledger['shared_sources'] if e['source'] == 'lib/tts.py')
        constants = {n.targets[0].id: ast.unparse(n.value) for n in self.tree.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        for name, expression in entry['constants'].items():
            self.assertEqual(constants[name], expression, name)
        actual = {(n.lineno, ast.unparse(n)) for n in ast.walk(self.tree)
                  if isinstance(n, ast.Call) and ast.unparse(n.func) == 'os.getenv'}
        self.assertEqual(actual, {(e['line'], e['expression']) for e in entry['environment_controls']})

    def test_legacy_forwarding_defaults_and_environment(self):
        for environment, revision, offline in [({}, None, False),
                ({'TTS_MODEL_REVISION': 'env-rev', 'TTS_OFFLINE': True}, 'env-rev', True),
                ({'HF_HUB_OFFLINE': True}, None, True)]:
            with self.subTest(environment=environment):
                backend = Mock()
                env = dict(QwenBackend=backend, os=types.SimpleNamespace(getenv=environment.get),
                           env_flag_enabled=lambda name, default: environment.get(name, default))
                load = self.function('load_tts_model', env)
                load('Qwen/legacy', 2)
                backend.assert_called_once_with(model_name='Qwen/legacy', num_threads=2,
                    dtype_str='auto', revision=revision, offline=offline)
                backend.reset_mock()
                load('Qwen/legacy', 3, 'float32', revision='explicit', offline=False)
                backend.assert_called_once_with(model_name='Qwen/legacy', num_threads=3,
                    dtype_str='float32', revision='explicit', offline=False)

    def test_legacy_automatic_device_attention_and_snapshot_forwarding(self):
        for device, flash, expected in [('cpu', False, 'sdpa'), ('mps', False, 'sdpa'),
                ('cuda', False, 'sdpa'), ('cuda', True, 'flash_attention_2')]:
            with self.subTest(device=device, flash=flash):
                torch = Mock()
                upstream = Mock(return_value=types.SimpleNamespace())
                snapshot = Mock(return_value='/inert/snapshots/resolved-revision')
                env = dict(Path=Path, torch=torch, get_device=Mock(return_value=device),
                           best_dtype=Mock(return_value='fake-dtype'), _has_flash_attn=Mock(return_value=flash))
                load = self.function('_load_tts_model_local', env)
                with patch.dict(sys.modules, {
                        'qwen_tts': types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=upstream)),
                        'huggingface_hub': types.SimpleNamespace(snapshot_download=snapshot)}):
                    model = load('Qwen/legacy', 2, revision='pinned', offline=True)
                env['get_device'].assert_called_once_with()
                env['best_dtype'].assert_called_once_with(device, 'auto')
                torch.set_num_threads.assert_called_once_with(2)
                snapshot.assert_called_once_with('Qwen/legacy', revision='pinned', local_files_only=True)
                upstream.assert_called_once_with('/inert/snapshots/resolved-revision',
                    device_map='cuda:0' if device == 'cuda' else device, dtype='fake-dtype',
                    attn_implementation=expected, local_files_only=True)
                self.assertEqual(model._audio_box_revision, 'resolved-revision')
                self.assertEqual(model._audio_box_checkpoint, 'Qwen/legacy')


if __name__ == '__main__':
    unittest.main()
