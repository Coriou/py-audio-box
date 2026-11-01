"""Small injected source tests: no model, network or worker lifecycle."""
import dataclasses
import hashlib
import io
import json
import os
import struct
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch
from worker.engines.pocket import execute
from worker.engines.pocket_execution import PocketExecution
from worker.transport import Refused
from lib.tts_backends.contract import AudioResult


class PocketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.bundle = self.root/'bundle'; self.bundle.mkdir()
        self.namespace = self.root/'attempt'; self.namespace.mkdir(mode=0o700)
        entries = []
        for name in ('config.yaml', 'model.safetensors'):
            data = b'inert fixture'
            (self.bundle/name).write_bytes(data)
            entries.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
        raw = json.dumps(dict(schema_version=1, backend='pocket-tts', checkpoint='test', revision='rev1', language='fr', files=entries)).encode()
        (self.bundle/'manifest.json').write_bytes(raw)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000); wav.writeframes(b'\0\0'*4)
        self.reference = buffer.getvalue()
        self.ref = self.root/'reference.wav'; self.ref.write_bytes(self.reference)
        self.spec = PocketExecution('bundle', hashlib.sha256(raw).hexdigest(), 'test', 'rev1', 'fr', 2, 1024, 100, 244, 1024*1024, 'reference', hashlib.sha256(self.reference).hexdigest(), len(self.reference), 4, 24000, 24000)
        self.audio = AudioResult([0., .5, -.5], 24000)
        self.calls = []; self.fail_close = False; self.fail_call = False
        outer = self
        class Backend:
            def __init__(self, root, **kwargs): outer.calls.append((root, kwargs))
            def synthesise(self, text, **kwargs):
                snapshot = kwargs['reference']
                outer.calls.append((text, snapshot.read_bytes(), snapshot.parent != outer.namespace, snapshot.stat().st_mode & 0o777))
                # Original mutation cannot change the already captured reference.
                outer.ref.write_bytes(b'changed after validation')
                if outer.fail_call: raise RuntimeError('call failed')
                return outer.audio
            def close(self):
                outer.calls.append('closed')
                if outer.fail_close: raise RuntimeError('close failed')
        self.factory = Backend

    def run_engine(self, spec=None, **kwargs):
        return execute(spec or self.spec, '  Bonjour\n', {'bundle': self.bundle}, {'reference': self.ref}, reference_temp_dir=self.namespace, backend_factory=self.factory, **kwargs)

    def test_exact_snapshot_arguments_and_identity(self):
        result = self.run_engine()
        self.assertEqual(self.calls, [(self.bundle, dict(language='fr', threads=2, reference_temp_dir=self.namespace)), ('  Bonjour\n', self.reference, True, 0o600), 'closed'])
        self.assertEqual((result['frames'], result['size_bytes']), (3, 50))
        self.assertEqual(hashlib.sha256(result['data']).hexdigest(), result['sha256'])
        with wave.open(io.BytesIO(result['data'])) as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (24000, 1, 2))
        result['identity']['language'] = 'en'
        self.assertEqual(self.spec.language, 'fr')
        self.assertEqual(PocketExecution.from_wire(self.spec.identity()), self.spec)
        self.assertEqual(list(self.namespace.iterdir()), [])

    def test_closed_descriptor_and_controls(self):
        for field in ('path', 'preset', 'instruct', 'seed', 'transcript'):
            with self.assertRaises(Refused): PocketExecution.from_wire(dict(self.spec.identity(), **{field: None}))
        for change in (dict(device='cuda'), dict(threads=True), dict(reference_id='../ref'), dict(sample_rate=True), dict(max_frames=0)):
            with self.assertRaises(Refused): dataclasses.replace(self.spec, **change)
        for kwargs in (dict(preset=''), dict(instruct='')):
            with self.assertRaises(Refused): self.run_engine(**kwargs)
        self.assertEqual(self.calls, [])

    def test_bindings_before_backend(self):
        for change in (dict(bundle_id='foreign'), dict(reference_id='foreign'), dict(manifest_sha256='0'*64), dict(reference_sha256='0'*64), dict(revision='other'), dict(checkpoint='other'), dict(language='en'), dict(reference_bytes=46), dict(reference_rate=22050), dict(reference_frames=5), dict(max_text_bytes=1)):
            with self.assertRaises((Refused, ValueError)): self.run_engine(dataclasses.replace(self.spec, **change))
        self.assertEqual(self.calls, [])

    def test_files_missing_tampered_symlink_link_permissions_owner(self):
        for path in (self.ref, self.bundle/'manifest.json', self.bundle/'model.safetensors'):
            original = path.read_bytes()
            for mode in ('missing', 'tamper', 'symlink', 'hardlink', 'permissions'):
                path.unlink()
                if mode == 'tamper': path.write_bytes(b'x')
                elif mode == 'symlink': path.symlink_to(self.root/'target')
                elif mode == 'hardlink':
                    target = self.root/'target'; target.write_bytes(original); os.link(target, path)
                elif mode == 'permissions': path.write_bytes(original); path.chmod(0o666)
                with self.assertRaises((Refused, OSError, ValueError)): self.run_engine()
                if path.exists() or path.is_symlink(): path.unlink()
                target = self.root/'target'
                if target.exists(): target.unlink()
                path.write_bytes(original)
        with patch('worker.engines.pocket.os.getuid', return_value=os.getuid()+1):
            with self.assertRaises(Refused): self.run_engine()
        self.assertEqual(self.calls, [])

    def test_oversize_and_reference_format(self):
        self.ref.write_bytes(self.reference+b'x')
        with self.assertRaises(Refused): self.run_engine()
        self.ref.write_bytes(b'not wave')
        spec = dataclasses.replace(self.spec, reference_bytes=8, reference_sha256=hashlib.sha256(b'not wave').hexdigest())
        with self.assertRaises(Refused): self.run_engine(spec)
        self.assertEqual(self.calls, [])

    def test_output_validation_and_close(self):
        import numpy as np
        for samples, rate, channels in (([float('nan')], 24000, 1), ([float('inf')], 24000, 1), ([1.1], 24000, 1), ([], 24000, 1), ([[0., 0.]], 24000, 2), ([0.], 22050, 1), ([0.], True, 1), ([0.], 24000., 1), ([0j], 24000, 1), (['x'], 24000, 1), ([0.]*101, 24000, 1)):
            self.ref.write_bytes(self.reference)
            self.audio.samples=np.asarray(samples); self.audio.sample_rate=rate; self.audio.channels=channels
            with self.assertRaises(Refused): self.run_engine()
            self.assertEqual(self.calls[-1], 'closed')
            self.assertEqual(list(self.namespace.iterdir()), [])
        self.ref.write_bytes(self.reference); self.audio=object()
        with self.assertRaises(Refused): self.run_engine()

    def reference_spec(self, raw):
        self.ref.write_bytes(raw)
        return dataclasses.replace(self.spec, reference_bytes=len(raw), reference_sha256=hashlib.sha256(raw).hexdigest())

    def test_r241_header_mutations_before_factory(self):
        for offset, fmt, value in ((34, '<H', 12), (32, '<H', 4), (40, '<I', 9), (4, '<I', 4096)):
            with self.subTest(offset=offset):
                raw = bytearray(self.reference)
                struct.pack_into(fmt, raw, offset, value)
                with self.assertRaisesRegex(Refused, 'reference_format'):
                    self.run_engine(self.reference_spec(bytes(raw)))
                self.assertEqual(self.calls, [])

    @staticmethod
    def riff(*chunks):
        body = b'WAVE' + b''.join(chunks)
        return b'RIFF' + struct.pack('<I', len(body)) + body

    @staticmethod
    def chunk(kind, data):
        return kind + struct.pack('<I', len(data)) + data + b'\0'*(len(data) & 1)

    def test_valid_ancillary_chunks_and_pcm_extension(self):
        fmt = self.reference[20:36]
        for fields in (fmt, fmt+b'\0\0'):
            with self.subTest(fmt_size=len(fields)):
                raw = self.riff(self.chunk(b'JUNK', b'odd'), self.chunk(b'fmt ', fields), self.chunk(b'LIST', b'info'), self.chunk(b'data', b'\0\0'*4), self.chunk(b'JUNK', b'x'))
                result = self.run_engine(self.reference_spec(raw))
                self.assertEqual(result['frames'], 3)
                self.assertEqual(self.calls[-2][1], raw)

    def test_invalid_riff_chunks_before_factory(self):
        fmt = self.chunk(b'fmt ', self.reference[20:36])
        data = self.chunk(b'data', b'\0\0'*4)
        cases = {
            'truncated_header': b'RIFF',
            'truncated_chunk_header': self.riff(fmt, data, b'JUNK'),
            'truncated_chunk_payload': self.riff(fmt, data, b'JUNK'+struct.pack('<I', 4)+b'x'),
            'missing_odd_padding': self.riff(fmt, data, b'JUNK'+struct.pack('<I', 1)+b'x'),
            'odd_data': self.riff(fmt, self.chunk(b'data', b'\0'*7)),
            'duplicate_fmt': self.riff(fmt, fmt, data),
            'duplicate_data': self.riff(fmt, data, data),
            'data_before_fmt': self.riff(data, fmt),
            'missing_fmt': self.riff(data),
            'missing_data': self.riff(fmt),
            'trailing_bytes': self.reference+b'x',
            'chunk_count': self.riff(fmt, data, self.chunk(b'JUNK', b'')*4095),
        }
        for tag in (3, 0xfffe):
            fields = bytearray(self.reference[20:36]); struct.pack_into('<H', fields, 0, tag)
            cases['format_'+str(tag)] = self.riff(self.chunk(b'fmt ', bytes(fields)), data)
        fields = bytearray(self.reference[20:36]); struct.pack_into('<I', fields, 8, 123)
        cases['byte_rate'] = self.riff(self.chunk(b'fmt ', bytes(fields)), data)
        cases['nonzero_extension'] = self.riff(self.chunk(b'fmt ', self.reference[20:36]+b'\x01\0'), data)
        for label, raw in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(Refused, 'reference_format'):
                    self.run_engine(self.reference_spec(raw))
                self.assertEqual(self.calls, [])

    def test_conditioning_never_adds_entries_to_resource_scan(self):
        original_factory = self.factory
        paths = []
        class Backend(original_factory):
            def synthesise(inner, *args, **kwargs):
                paths.append(kwargs['reference'])
                self.assertEqual(list(self.namespace.iterdir()), [])
                self.assertEqual(paths[-1].stat().st_size, len(self.reference))
                return super().synthesise(*args, **kwargs)
        self.factory = Backend
        self.run_engine()
        self.assertEqual(list(self.namespace.iterdir()), [])
        self.assertFalse(paths[0].exists())

    def test_lifecycle_failures_retain_namespace(self):
        sentinel = self.namespace/'caller'; sentinel.write_bytes(b'caller')
        for field in ('fail_call', 'fail_close'):
            self.ref.write_bytes(self.reference); setattr(self, field, True)
            with self.assertRaises(RuntimeError): self.run_engine()
            setattr(self, field, False)
            self.assertEqual(self.calls[-1], 'closed')
            self.assertEqual(list(self.namespace.iterdir()), [sentinel])
        def failed(*args, **kwargs): raise RuntimeError('constructor failed')
        self.ref.write_bytes(self.reference); self.factory=failed
        with self.assertRaises(RuntimeError): self.run_engine()
        self.assertEqual(list(self.namespace.iterdir()), [sentinel])


    def test_real_backend_with_inert_runtime_and_legacy_defaults(self):
        import sys
        import types
        import numpy as np
        from lib.tts_backends.pocket_tts import PocketBackend
        # Valid local config; update the trusted raw-manifest binding.
        config = b'weights_path: model.safetensors\n'
        (self.bundle/'config.yaml').write_bytes(config)
        manifest = json.loads((self.bundle/'manifest.json').read_bytes())
        manifest['files'][0].update(sha256=hashlib.sha256(config).hexdigest(), bytes=len(config))
        raw = json.dumps(manifest).encode()
        (self.bundle/'manifest.json').write_bytes(raw)
        spec = dataclasses.replace(self.spec, manifest_sha256=hashlib.sha256(raw).hexdigest())
        snapshots = []
        class Tensor:
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.array([0., .1])
        outer = self
        class Model:
            sample_rate = 24000
            @classmethod
            def load_model(cls, config):
                # Stub only: no upstream loader/weights are executed.
                assert Path(config).suffix in ('.yaml', '.yml')
                assert Path(config) == outer.bundle / 'config.yaml'
                assert Path.cwd() == outer.bundle
                assert Path('model.safetensors').read_bytes() == (outer.bundle / 'model.safetensors').read_bytes()
                assert list(outer.namespace.iterdir()) == []
                if outer.fail_load:
                    raise RuntimeError('loader failed')
                return cls()
            def get_state_for_audio_prompt(self, path):
                snapshot = Path(path)
                snapshots.append(snapshot)
                assert snapshot.read_bytes() == outer.reference
                return 'state'
            def generate_audio(self, state, text, copy_state):
                assert state == 'state' and copy_state
                return Tensor()
        threads = []; seeds = []
        outer.fail_load = False
        original_cwd = Path.cwd()
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(set_num_threads=threads.append, manual_seed=seeds.append), 'pocket_tts': types.SimpleNamespace(TTSModel=Model), 'yaml': types.SimpleNamespace(safe_load=lambda text: {'weights_path': 'model.safetensors'}, safe_dump=lambda value: repr(value))}):
            self.factory = PocketBackend
            result = self.run_engine(spec)
            self.assertEqual(Path.cwd(), original_cwd)
            self.assertEqual(result['frames'], 2)
            self.assertEqual(list(self.namespace.iterdir()), [])
            self.assertFalse(snapshots[-1].exists())
            # Historical direct-call namespace default, seed and cache remain.
            legacy = PocketBackend(self.bundle)
            try:
                from lib.tts_backends.contract import BackendError
                with tempfile.TemporaryFile() as foreign:
                    foreign_alias = Path('/proc/self/fd' if Path('/proc/self/fd').is_dir() else '/dev/fd') / str(foreign.fileno())
                    with self.assertRaises(BackendError):
                        legacy.synthesise('Unregistered descriptor', reference=foreign_alias)
                first = legacy.synthesise('Bonjour', reference=self.ref, seed=7)
                second = legacy.synthesise('Encore', reference=self.ref)
                self.assertEqual(first.metadata['reference_sha256'], spec.reference_sha256)
                self.assertTrue(second.metadata['reference_state_reused'])
                self.assertIsNone(legacy.reference_temp_dir)
            finally:
                legacy.close()
            outer.fail_load = True
            with self.assertRaisesRegex(RuntimeError, 'loader failed'):
                PocketBackend(self.bundle)
            self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual(threads, [2, 2, 2]); self.assertEqual(seeds, [7])
        self.assertEqual(list(self.namespace.iterdir()), [])

    def test_manifest_structure_and_limits(self):
        original = json.loads((self.bundle/'manifest.json').read_bytes())
        import copy
        for mutation in ('pickle', 'duplicate', 'escape', 'missing_config', 'oversize', 'bad_digest'):
            manifest = copy.deepcopy(original)
            if mutation == 'pickle': manifest['files'][1]['path'] = 'model.pkl'
            elif mutation == 'duplicate': manifest['files'].append(manifest['files'][0])
            elif mutation == 'escape': manifest['files'][1]['path'] = '../model.safetensors'
            elif mutation == 'missing_config': manifest['files'] = manifest['files'][1:]
            elif mutation == 'oversize': manifest['files'][1]['bytes'] = 2**40
            else: manifest['files'][1]['sha256'] = '0'*64
            raw = json.dumps(manifest).encode()
            (self.bundle/'manifest.json').write_bytes(raw)
            spec = dataclasses.replace(self.spec, manifest_sha256=hashlib.sha256(raw).hexdigest())
            with self.assertRaises((Refused, OSError)): self.run_engine(spec)
        self.assertEqual(self.calls, [])

    def test_output_byte_limit_and_namespace(self):
        with self.assertRaises(Refused): self.run_engine(dataclasses.replace(self.spec, max_output_bytes=48))
        self.assertEqual(self.calls[-1], 'closed')
        self.ref.write_bytes(self.reference)
        self.namespace.chmod(0o755)
        before = len(self.calls)
        with self.assertRaises(Refused): self.run_engine()
        self.assertEqual(len(self.calls), before)

    def test_imports_lazy(self):
        import subprocess
        import sys
        source = """
import builtins
import pathlib
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'pocket_tts', 'numpy', 'service', 'requests') or (name.split('.')[0] == 'urllib' and name != 'urllib.parse'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import worker.engines.pocket
import worker.engines.pocket_execution
"""
        subprocess.run([sys.executable, '-c', source], check=True)


if __name__ == '__main__': unittest.main()
