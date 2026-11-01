"""Source/inert checks: never import the ML loader or start a backend."""
import ast
import hashlib
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

    def test_all_recorded_whole_source_hashes(self):
        # Same complete source sets as the aggregate contract test, without
        # unrelated schema/pytest dependencies. Keep that original test intact.
        ledger = json.loads((ROOT / 'contracts/v1/legacy-controls.json').read_text())
        for entry in ledger['commands'] + ledger['shared_sources'] + ledger['launchers']:
            with self.subTest(source=entry['source']):
                self.assertEqual(hashlib.sha256((ROOT / entry['source']).read_bytes()).hexdigest(),
                                 entry['sha256'])

    def test_added_voice_clone_controls_are_recorded_exactly(self):
        ledger = json.loads((ROOT / 'contracts/v1/legacy-controls.json').read_text())
        entry = next(e for e in ledger['commands'] if e['source'] == 'apps/voice-clone/voice-clone.py')
        tree = ast.parse((ROOT / entry['source']).read_text())
        parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'build_parser')
        calls = []
        for statement in parser.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 \
                    and isinstance(statement.targets[0], ast.Name) \
                    and statement.targets[0].id in ('repair', 'alias'):
                calls.append(statement.value)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) \
                    and isinstance(statement.value.func, ast.Attribute) \
                    and isinstance(statement.value.func.value, ast.Name) \
                    and statement.value.func.value.id in ('repair', 'alias'):
                calls.append(statement.value)
        actual = [dict(line=n.lineno, kind=n.func.attr,
                       arguments=[ast.unparse(arg) for arg in n.args],
                       keywords={kw.arg: ast.unparse(kw.value) for kw in n.keywords}) for n in calls]
        start = next(i for i, control in enumerate(entry['controls'])
                     if control['kind'] == 'add_parser'
                     and control['arguments'] == ["'repair-registry'"])
        recorded = entry['controls'][start:start + 9]
        self.assertEqual(len(actual), 9)
        self.assertEqual(actual, recorded)

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
                    dtype_str='auto', revision=revision, offline=offline, legacy_hf_cache=True)
                backend.reset_mock()
                load('Qwen/legacy', 3, 'float32', revision='explicit', offline=False)
                backend.assert_called_once_with(model_name='Qwen/legacy', num_threads=3,
                    dtype_str='float32', revision='explicit', offline=False, legacy_hf_cache=True)
                backend.reset_mock()
                load('Qwen/legacy', 4, device='cpu', attention='sdpa')
                backend.assert_called_once_with(model_name='Qwen/legacy', num_threads=4,
                    dtype_str='auto', revision=revision, offline=offline,
                    legacy_hf_cache=False, device='cpu', attention='sdpa')

    def test_legacy_automatic_device_attention_and_snapshot_forwarding(self):
        for device, flash, expected in [('cpu', False, 'sdpa'), ('mps', False, 'sdpa'),
                ('cuda', False, 'sdpa'), ('cuda', True, 'flash_attention_2')]:
            with self.subTest(device=device, flash=flash):
                torch = Mock()
                upstream = Mock(return_value=types.SimpleNamespace())
                snapshot = Mock(return_value='/inert/snapshots/resolved-revision')
                env = dict(Path=Path, torch=torch, get_device=Mock(return_value=device),
                           best_dtype=Mock(return_value='fake-dtype'), _has_flash_attn=Mock(return_value=flash))
                self.function('_qwen_model_class', env)
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
