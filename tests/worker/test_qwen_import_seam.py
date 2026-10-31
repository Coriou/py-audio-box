"""Real worker/complete-loader import regression; no model or process is started."""
import contextlib
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import subprocess
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
    def test_failed_reap_storage_survives_owner_gc(self):
        import gc
        import multiprocessing as mp
        import shutil
        import time
        import weakref
        from lib.tts_backends.qwen import QwenBackend
        owner = QwenBackend(model_name='Qwen3-TTS-12Hz-0.6B-Base')
        root = Path(tempfile.mkdtemp(prefix='qwen-child-', dir='/tmp'))
        owner._runtime_directory = root
        process = mp.get_context('spawn').Process(target=time.sleep, args=(60,))
        process.start()
        owner._process = process
        ref = weakref.ref(owner)
        try:
            # Actual child/status; only the parent's reap operation fails.
            def failed_join(*args):
                raise OSError('injected reap failure')
            with patch.object(process, 'join', side_effect=failed_join):
                with self.assertRaises(OSError):
                    owner.close()
                self.assertTrue(owner._failed_reap)
                del owner
                gc.collect()
                self.assertIsNone(ref())
                self.assertTrue(root.is_dir())
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            self.assertIs(type(process.exitcode), int)
        finally:
            if process.is_alive():
                process.kill()
                process.join(5)
            if type(process.exitcode) is int:
                process.close()
                shutil.rmtree(root)

    def test_confirmed_reap_cleans_storage_and_close_is_idempotent(self):
        import multiprocessing as mp
        import time
        from lib.tts_backends.qwen import QwenBackend
        owner = QwenBackend(model_name='Qwen3-TTS-12Hz-0.6B-Base')
        root = Path(tempfile.mkdtemp(prefix='qwen-child-', dir='/tmp'))
        owner._runtime_directory = root
        process = mp.get_context('spawn').Process(target=time.sleep, args=(60,))
        process.start()
        owner._process = process
        owner.close()
        self.assertFalse(root.exists())
        self.assertIsNone(owner._process)
        self.assertIsNone(owner._runtime_directory)
        owner.close()

    def test_start_raises_after_spawn_without_published_pid(self):
        import multiprocessing as mp
        import time
        from lib.tts_backends.qwen import QwenBackend
        real = mp.get_context('spawn').Process(target=time.sleep, args=(60,))

        class AmbiguousProcess:
            pid = None
            exitcode = None
            verified = False

            def start(self):
                real.start()
                raise OSError('post-spawn failure before pid publication')

            def is_alive(self):
                return False  # Public metadata cannot establish absence.

            def join(self, timeout):
                if not self.verified:
                    raise AssertionError('no published process handle')
                real.join(timeout)
                self.exitcode = real.exitcode

            def close(self):
                real.close()

        stub = AmbiguousProcess()
        context = Mock()
        context.Pipe.return_value = (Mock(), Mock())
        context.Process.return_value = stub
        owner = QwenBackend(model_name='Qwen3-TTS-12Hz-0.6B-Base')
        try:
            with patch('lib.tts_backends.qwen.mp.get_context', return_value=context):
                with self.assertRaisesRegex(AssertionError, 'no published process handle'):
                    owner.invoke('unused')
            root = owner._runtime_directory
            self.assertTrue(real.is_alive())
            self.assertTrue(root.is_dir())
            self.assertIs(owner._process, stub)
            self.assertTrue(owner._failed_reap)
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                owner.invoke('unused')
            owner.__del__()
            self.assertTrue(root.is_dir())
            self.assertIs(owner._process, stub)
            real.terminate()
            real.join(5)
            self.assertIs(type(real.exitcode), int)
            stub.verified = True
            owner.close()  # Genuine reap now available through the existing API.
            self.assertFalse(root.exists())
            owner.close()
        finally:
            if not real._closed and real.pid is not None:
                if real.is_alive():
                    real.kill()
                real.join(5)
                if type(real.exitcode) is int:
                    stub.verified = True
                    owner.close()

    def test_missing_passwd_environment_before_loader_import(self):
        code = r"""
import os, sys, pwd, getpass, tempfile, importlib.abc, importlib.util
from pathlib import Path
from unittest.mock import patch
from lib.tts_backends.qwen import _worker_impl
keys = ('HOME','XDG_CACHE_HOME','TMPDIR','TMP','TEMP','TORCHINDUCTOR_CACHE_DIR',
        'TRITON_CACHE_DIR','HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE',
        'TRANSFORMERS_CACHE','TORCH_HOME','USER','LOGNAME')
for key in keys: os.environ[key] = '/host-secret/poison'
# Simulate the exact numeric-UID failure, even on a machine with passwd entries.
def missing(*args): raise KeyError('getpwuid(): uid not found: 20000')
pwd.getpwuid = missing
with patch.dict(os.environ):
    for key in ('USER','LOGNAME','LNAME','USERNAME'): os.environ.pop(key, None)
    try: getpass.getuser()
    except KeyError: pass
    else: raise AssertionError('passwd simulation failed')
roots = []
class Loader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'lib.tts': return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec): return None
    def exec_module(self, module):
        root = Path(os.environ['HOME']).parent
        assert root.parent == Path('/tmp') and root.name.startswith('qwen-child-')
        assert root.stat().st_mode & 0o777 == 0o700
        roots.append(root)
        for key in keys[:-2]:
            path = Path(os.environ[key])
            assert path.is_relative_to(root) and path.is_dir(), key
        assert tempfile.gettempdir() == os.environ['TMPDIR']
        assert getpass.getuser() == 'qwen-worker'
        assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
        # Corroborated Inductor cache selection: explicit cache avoids getuser.
        cache = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
        assert cache or getpass.getuser()
        def load(**settings): raise ImportError('inert upstream stop')
        module._load_tts_model_local = load
class Connection:
    def __init__(self): self.messages=[]; self.closed=False
    def send(self, message): self.messages.append(message)
    def close(self): self.closed=True
sys.meta_path.insert(0, Loader())
for _ in range(2):
    sys.modules.pop('lib.tts', None)
    connection=Connection()
    _worker_impl(connection, {'offline':True})
    assert connection.messages == [('error', ('ImportError','inert upstream stop'))]
    assert connection.closed and not roots[-1].exists()
    assert all(os.environ[key] == '/host-secret/poison' for key in keys)
assert roots[0] != roots[1]
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_private_wav_bytes_become_spawn_safe_waveform_rate(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'from tests.worker.test_qwen_import_seam import ImportSeamTests; '
            'ImportSeamTests().check_private_waveform()'], cwd=ROOT,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def check_private_waveform(self):
        import io, pickle, wave
        import numpy as np
        from lib.tts_backends.qwen import QwenBackend
        raw=io.BytesIO()
        with wave.open(raw,'wb') as w:
            w.setparams((1,2,24000,0,'NONE','none'));w.writeframes(b'\x00\x80\x00\x40')
        backend=object.__new__(QwenBackend)
        backend.generate_audio=Mock(return_value='injected')
        self.assertEqual(backend.synthesise_clone(dict(text='audition'),raw.getvalue(),None,True,1800),'injected')
        method,params,timeout=backend.generate_audio.call_args.args
        self.assertEqual(method,'_service_clone')
        waveform,rate=pickle.loads(pickle.dumps(params['ref_audio']))
        self.assertEqual(rate,24000);self.assertEqual(waveform.dtype,np.dtype('float32'))
        self.assertEqual(waveform.shape,(2,))
        np.testing.assert_array_equal(waveform,np.array([-1.,.5],dtype='float32'))
        self.assertIsNone(params['ref_text']);self.assertTrue(params['x_vector_only_mode'])
        self.assertGreater(timeout,0);self.assertLessEqual(timeout,1800)

    def exercise_native(self, root):
        # Enter the full loader, not an AST extraction or a fake loader.
        with isolated(root), tempfile.TemporaryDirectory() as tmp:
            events = []
            torch = types.ModuleType('torch')
            torch.dtype = str
            torch.float32, torch.float16, torch.bfloat16 = 'fp32', 'fp16', 'bf16'
            torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
            torch.set_num_threads = lambda n: events.append(('threads', n))
            prompt = object()
            def preset(**kw): events.append(('preset', kw)); return [[0.1, -0.1]], 24000
            def create(**kw): events.append(('prompt', kw)); return prompt
            def clone(**kw):
                self.assertIs(kw['voice_clone_prompt'], prompt)
                events.append(('clone', kw)); return [[0.2]], 24000
            model = types.SimpleNamespace(generate_custom_voice=preset,
                create_voice_clone_prompt=create, generate_voice_clone=clone,
                device=types.SimpleNamespace(type='mps'),
                model=types.SimpleNamespace(dtype='torch.float32'))
            def load(*args, **kw):
                self.assertEqual(events[0], ('threads', 2))
                self.assertEqual(__import__('os').environ['TOKENIZERS_PARALLELISM'], 'false')
                events.append(('load', args, kw)); return model
            factory = Mock(side_effect=load)
            hub = types.SimpleNamespace(snapshot_download=Mock(side_effect=AssertionError('download')))
            # Keep this import-isolation fixture entirely synthetic. Loading a
            # real NumPy extension inside patch.dict(sys.modules) then removing
            # it makes a second CPython 3.14 import invalid.
            number, complex_type = object(), object()
            numpy = types.SimpleNamespace(ndarray=type('ndarray', (), {}),
                asarray=lambda value:types.SimpleNamespace(ndim=1, dtype=float, size=len(value)),
                number=number, complexfloating=complex_type,
                issubdtype=lambda dtype, family:family is number,
                isfinite=lambda value:types.SimpleNamespace(all=lambda:True))
            with patch.dict(sys.modules, dict(torch=torch, numpy=numpy,
                    qwen_tts=types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=factory)),
                    huggingface_hub=hub)), patch.dict('os.environ'), \
                    patch('platform.system', return_value='Darwin'), patch('platform.machine', return_value='arm64'), \
                    patch('multiprocessing.get_context', side_effect=AssertionError('nested context')), \
                    patch('multiprocessing.process.BaseProcess.start', side_effect=AssertionError('nested process')):
                native = importlib.import_module('lib.tts_backends.qwen_native')
                backend = native.NativeQwenBackend(model_name=tmp, num_threads=2,
                    dtype_str='float32', revision='exact-revision', offline=True, device='mps', attention='sdpa')
                params = dict(text='hello', language='English', temperature=.9, top_p=1.,
                              top_k=50, repetition_penalty=1.05, max_new_tokens=2)
                result = backend.synthesise_preset(params, 'Ryan', 'warm', 25)
                contract = importlib.import_module('lib.tts_backends.contract')
                self.assertIs(type(result), contract.AudioResult)
                self.assertEqual(result.metadata, dict(backend='qwen', checkpoint=tmp,
                    revision='exact-revision', offline=True))
                backend.synthesise_clone(params, tmp+'/ref.wav', 'source', False, 25)
                self.assertEqual([e[0] for e in events], ['threads','load','preset','prompt','clone'])
                self.assertEqual(factory.call_args.kwargs, dict(device_map='mps', dtype='fp32',
                    attn_implementation='sdpa', local_files_only=True))
                self.assertFalse(hasattr(backend, 'invoke'))
                self.assertFalse(hasattr(backend, 'synthesise_design'))
                with self.assertRaises(ValueError): backend.synthesise_preset(dict(params, method='exec'), 'Ryan', None, 25)
                with self.assertRaises(ValueError): backend.synthesise_preset(params, 'Ryan', None, 31)
                backend.close()
                with self.assertRaises(ValueError): backend.synthesise_preset(params, 'Ryan', None, 25)
                hub.snapshot_download.assert_not_called()

    def test_native_root_complete_loader(self):
        self.exercise_native(ROOT)

    def test_native_extracted_complete_loader(self):
        with isolated(ROOT):
            from service.package_install import archive
            with tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp)/'headless.zip'
                archive(ROOT, destination)
                with zipfile.ZipFile(destination) as z:
                    z.extractall(Path(tmp)/'extracted')
                self.exercise_native(Path(tmp)/'extracted/speech-box')

    def check_cold_imports(self, root, legacy=False):
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

    def exercise(self, root, legacy=False):
        # A fresh interpreter proves cold imports even after other tests loaded
        # NumPy, without unloading/reloading its native extension in this process.
        result = subprocess.run([sys.executable, '-B', '-c',
            'from pathlib import Path; '
            'from tests.worker.test_qwen_import_seam import ImportSeamTests; '
            'ImportSeamTests().check_cold_imports(Path(__import__("sys").argv[1]), '
            '__import__("sys").argv[2] == "True")', str(root), str(legacy)],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with isolated(root, legacy):
            backend = importlib.import_module('tts_backends.qwen' if legacy else 'lib.tts_backends.qwen')
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
            def load(*a, **kw):
                if kw['device_map'] == 'cpu':
                    self.assertEqual(os.environ['RAYON_NUM_THREADS'], '2')
                events.append(('load', a, kw))
                return model
            factory = Mock(side_effect=load)
            upstream = types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=factory))
            hub = types.SimpleNamespace(snapshot_download=Mock(side_effect=AssertionError('download')))
            numpy = types.SimpleNamespace(ndarray=type('ndarray', (), {}))
            class Connection:
                def __init__(self):
                    self.messages = []; self.closed = False
                    self.requests = iter([
                        ('generate_custom_voice', (), {'text': 'hello', 'speaker': 'Ryan'}),
                        ('generate_voice_design', (), {'text': 'hello', 'instruct': 'warm'}),
                        ('_service_clone', (), dict(text='hello', ref_audio='inert.wav', ref_text='source', x_vector_only_mode=False)),
                        ('_service_clone', (), dict(text='audition', ref_audio=([.25,-.25],24000), ref_text=None, x_vector_only_mode=True))])
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
                        self.assertEqual(connection.messages[:5], [('ready', {'revision': 'exact-revision'}),
                            ('ok', 'preset-result'), ('ok', 'design-result'), ('ok', 'clone-result'), ('ok', 'clone-result')])
                        prompts=[event[1] for event in events if event[0]=='prompt']
                        self.assertEqual(prompts[-1],dict(ref_audio=([.25,-.25],24000),ref_text=None,x_vector_only_mode=True))
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
                for invalid in (0, 65, True, '2'):
                    connection = Connection()
                    backend._worker_impl(connection, dict(model_name=tmp, num_threads=invalid,
                        dtype_str='float32', revision='r', offline=True, device='cpu', attention='sdpa'))
                    self.assertEqual(connection.messages, [('error', ('ValueError', 'bounded_cpu_threads_required'))])
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
