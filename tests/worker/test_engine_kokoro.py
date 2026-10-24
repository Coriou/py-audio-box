"""Injected backend/source evidence only; no actual inference or containment."""
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from worker.engines.execution import Execution
from worker.engines.kokoro import execute
from worker.transport import Refused


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        files = []
        for name in ('model.onnx', 'voices.bin'):
            (self.root/name).write_bytes(b'injected fixture')
            files.append(dict(path=name, sha256=hashlib.sha256(b'injected fixture').hexdigest()))
        raw = json.dumps(dict(schema_version=1, backend='kokoro', checkpoint='test', revision='rev1', language='fr', files=files)).encode()
        (self.root/'manifest.json').write_bytes(raw)
        self.spec = Execution('bundle1', hashlib.sha256(raw).hexdigest(), 'test', 'rev1', 'ff_siwis', 'fr', 2, 1024, 100, 244, 1024*1024)
        self.calls = []
        outer = self
        class Backend:
            def __init__(self, root, **kwargs): outer.calls.append((root, kwargs))
            def synthesise(self, text, **kwargs):
                outer.calls.append((text, kwargs))
                return outer.audio
            def close(self): outer.calls.append('closed')
        self.factory = Backend
        from lib.tts_backends.contract import AudioResult
        self.audio = AudioResult([0., .5, -.5], 24000)

    def run_engine(self, spec=None, **kwargs):
        return execute(spec or self.spec, '  Bonjour\n', {'bundle1': self.root}, backend_factory=self.factory, **kwargs)

    def test_exact_arguments_and_detached_identity(self):
        result = self.run_engine()
        self.assertEqual(self.calls, [(self.root.resolve(), dict(language='fr', threads=2)), ('  Bonjour\n', dict(preset='ff_siwis')), 'closed'])
        self.assertEqual((result['frames'], result['size_bytes']), (3, 50))
        self.assertEqual(result['data'][:4], b'RIFF')
        result['identity']['preset'] = 'changed'
        self.assertEqual(self.spec.preset, 'ff_siwis')
        self.assertEqual(Execution.from_wire(self.spec.identity()), self.spec)

    def test_closed_descriptor_and_unsupported_controls(self):
        for field in ('reference', 'instruct', 'seed', 'path'):
            with self.assertRaises(Refused): Execution.from_wire(dict(self.spec.identity(), **{field: None}))
        for changes in (dict(device='cuda'), dict(threads=True), dict(language='en'), dict(max_frames=0), dict(bundle_id='/tmp/model')):
            with self.assertRaises(Refused): dataclasses.replace(self.spec, **changes)

    def test_limits_close_backend(self):
        for changes in (dict(max_frames=2), dict(max_output_bytes=48)):
            with self.assertRaises(Refused): self.run_engine(dataclasses.replace(self.spec, **changes))
            self.assertEqual(self.calls[-1], 'closed')
        with self.assertRaises(Refused): self.run_engine(dataclasses.replace(self.spec, max_text_bytes=1))

    def test_mutated_invalid_audio(self):
        import numpy as np
        for data, rate in (([float('nan')], 24000), ([1.1], 24000), ([0.], 48000), ([], 24000), ([[0., 0.]], 24000)):
            self.audio.samples = np.asarray(data)
            self.audio.sample_rate = rate
            with self.assertRaises(Refused): self.run_engine()
            self.assertEqual(self.calls[-1], 'closed')

    def test_bundle_binding_before_backend(self):
        for changes in (dict(bundle_id='missing'), dict(manifest_sha256='0'*64), dict(revision='other'), dict(checkpoint='other')):
            with self.assertRaises(Refused): self.run_engine(dataclasses.replace(self.spec, **changes))
        self.assertEqual(self.calls, [])
        (self.root/'model.onnx').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.run_engine()
        self.assertEqual(self.calls, [])

    def test_imports_are_lazy(self):
        code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('service', 'torch', 'onnxruntime', 'kokoro_onnx', 'numpy', 'requests', 'urllib'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import worker.engines.execution
import worker.engines.kokoro
'''
        subprocess.run([sys.executable, '-c', code], check=True)


if __name__ == '__main__': unittest.main()
