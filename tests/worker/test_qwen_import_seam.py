"""Real worker/complete-loader import regression; no model or process is started."""
import contextlib
import importlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def isolated(root, legacy=False):
    prefixes = ('lib', 'tts', 'tts_backends', 'worker')
    saved = {n: m for n, m in sys.modules.items() if n.split('.')[0] in prefixes}
    for n in saved:
        del sys.modules[n]
    path = sys.path[:]
    sys.path[:] = [str(root/'lib' if legacy else root)] + [p for p in path
        if p and not Path(p).resolve().is_relative_to(ROOT)
        and not Path(p).resolve().is_relative_to(root)]
    try:
        yield
    finally:
        for n in tuple(sys.modules):
            if n.split('.')[0] in prefixes:
                del sys.modules[n]
        sys.modules.update(saved)
        sys.path[:] = path


class ImportSeamTests(unittest.TestCase):
    def exercise(self, root, legacy=False):
        with isolated(root, legacy):
            class NoML(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'torch', 'qwen_tts', 'huggingface_hub', 'numpy'}:
                        raise AssertionError('helper imported ML: '+fullname)
            self.assertTrue(all(n not in sys.modules for n in ('torch', 'qwen_tts', 'huggingface_hub', 'numpy')))
            finder = NoML()
            sys.meta_path.insert(0, finder)
            try:
                backend = importlib.import_module('tts_backends.qwen' if legacy else 'lib.tts_backends.qwen')
                if not legacy:
                    importlib.import_module('worker.engines.qwen')
                self.assertNotIn('tts' if legacy else 'lib.tts', sys.modules)
            finally:
                sys.meta_path.remove(finder)
            before_path = sys.path[:]
            events = []
            torch = types.ModuleType('torch')
            torch.dtype = str
            torch.Tensor = type('Tensor', (), {})
            torch.float32, torch.float16, torch.bfloat16 = 'fp32', 'fp16', 'bf16'
            torch.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
            torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
            torch.set_num_threads = lambda n: events.append(('threads', n))
            model = types.SimpleNamespace(
                generate_custom_voice=lambda **kw: events.append(('preset', kw)) or 'preset-result',
                generate_voice_design=lambda **kw: events.append(('design', kw)) or 'design-result',
                create_voice_clone_prompt=lambda **kw: events.append(('prompt', kw)) or 'private-prompt',
                generate_voice_clone=lambda **kw: events.append(('clone', kw)) or 'clone-result')
            factory = Mock(side_effect=lambda *a, **kw: events.append(('load', a, kw)) or model)
            upstream = types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=factory))
            hub = types.SimpleNamespace(snapshot_download=Mock(side_effect=AssertionError('download')))
            numpy = types.SimpleNamespace(ndarray=type('ndarray', (), {}))
            class Connection:
                def __init__(self):
                    self.messages = []; self.closed = False
                    self.requests = iter([
                        ('generate_custom_voice', (), {'text': 'hello', 'speaker': 'Ryan'}),
                        ('generate_voice_design', (), {'text': 'hello', 'instruct': 'warm'}),
                        ('_service_clone', (), dict(text='hello', ref_audio='inert.wav', ref_text='source', x_vector_only_mode=False))])
                def send(self, message): self.messages.append(message)
                def recv(self):
                    try: return next(self.requests)
                    except StopIteration: raise EOFError
                def close(self): self.closed = True
            with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules,
                    dict(torch=torch, numpy=numpy, qwen_tts=upstream, huggingface_hub=hub)), \
                    patch('platform.system', return_value='Darwin'), patch('platform.machine', return_value='arm64'), \
                    patch.dict('os.environ'):
                loader_name = 'tts' if legacy else 'lib.tts'
                for device, dtype, attention in [('cpu', 'float32', 'eager'), ('cuda:1', 'float16', 'flash_attention_2'), ('mps', 'float16', 'sdpa')]:
                    with self.subTest(device=device):
                        connection = Connection()
                        backend._worker_impl(connection, dict(model_name=tmp, num_threads=2,
                            dtype_str=dtype, revision='exact-revision', offline=True, device=device, attention=attention))
                        self.assertEqual(connection.messages[:4], [('ready', {'revision': 'exact-revision'}),
                            ('ok', 'preset-result'), ('ok', 'design-result'), ('ok', 'clone-result')])
                        self.assertTrue(connection.closed)
                        self.assertEqual(factory.call_args.args, (tmp,))
                        self.assertEqual(factory.call_args.kwargs, dict(device_map=device, dtype=getattr(torch, dtype),
                            attn_implementation=attention, local_files_only=True))
                loader = sys.modules[loader_name]
                self.assertIs(loader.QwenBackend, backend.QwenBackend)
                for module in (loader, backend):
                    self.assertTrue(Path(module.__file__).resolve().is_relative_to(root.resolve()))
                owner = object.__new__(backend.QwenBackend)
                owner._audio_box_checkpoint = 'checkpoint'
                owner._audio_box_revision = 'revision'
                owner.invoke = Mock(return_value=['inert-prompt'])
                bound = owner.create_voice_clone_prompt('inert.wav')
                self.assertIsInstance(bound, loader.ClonePrompt)
                self.assertEqual((bound.checkpoint, bound.revision), ('checkpoint', 'revision'))
                self.assertEqual(events[0], ('threads', 2))
                self.assertEqual([e[0] for e in events[2:6]], ['preset', 'design', 'prompt', 'clone'])
                self.assertEqual(events[5][1]['voice_clone_prompt'], 'private-prompt')
                # Full loader refuses unavailable explicit CUDA, never auto-selects CPU.
                torch.cuda.is_available = lambda: False
                with self.assertRaisesRegex(ValueError, 'CUDA device unavailable'):
                    loader._load_tts_model_local(tmp, 2, 'float16', 'r', True, 'cuda:1', 'sdpa')
                # Dependency failures remain the original error, without import retries.
                factory.side_effect = ImportError('upstream dependency sentinel')
                connection = Connection()
                backend._worker_impl(connection, dict(model_name=tmp, num_threads=2,
                    dtype_str='float32', revision='r', offline=True, device='cpu', attention='eager'))
                self.assertEqual(connection.messages, [('error', ('ImportError', 'upstream dependency sentinel'))])
                hub.snapshot_download.assert_not_called()
            self.assertEqual(sys.path, before_path)

    def test_root_only_real_worker_and_loader(self):
        self.exercise(ROOT)

    def test_legacy_lib_path_real_worker_and_loader(self):
        self.exercise(ROOT, legacy=True)

    def test_extracted_archive_real_worker_and_loader(self):
        with isolated(ROOT):
            from service.package_install import archive
            with tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp)/'headless.zip'
                archive(ROOT, destination)
                manifest = json.loads((ROOT/'headless-dist-manifest.json').read_text())
                with zipfile.ZipFile(destination) as z:
                    self.assertEqual(set(z.namelist()), {'speech-box/'+n for n in manifest['files']} | {'speech-box/speech-box'})
                    z.extractall(Path(tmp)/'extracted')
                package = Path(tmp)/'extracted/speech-box'
                for name in ('lib/tts.py', 'lib/tts_backends/qwen.py', 'worker/engines/qwen.py'):
                    self.assertEqual((package/name).read_bytes(), (ROOT/name).read_bytes())
                self.exercise(package)


if __name__ == '__main__':
    unittest.main()
