"""Unittest only: private temporary files, injected downloader, no ML/network."""
import sys
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from service.provisioning import provision, identity, manifest_sha256, readiness_matches, validate_expected
from lib.tts_backends.assets import verified_bundle


class ProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp' if sys.platform == 'darwin' else '/tmp')
        self.addCleanup(self.cleanup)
        self.root = Path(self.temp.name) / 'models'
        self.data = {'model.onnx': b'model', 'voices.bin': b'voices'}
        self.expected = dict(schema_version=1, backend='kokoro', language='en', checkpoint='fixture-kokoro', revision='fixture-model-revision',
                             license='fixture only', access='fixture only; no real qualification',
                             versions={p: 'fixture' for p in ('kokoro-onnx', 'onnxruntime', 'espeakng-loader', 'phonemizer')},
                             files=[dict(path=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                                         origin='https://fixture.invalid/' + n, revision='fixture') for n, b in self.data.items()])
        self.calls = []

    def cleanup(self):
        for p in Path(self.temp.name).rglob('*'):
            if not p.is_symlink():
                p.chmod(0o755 if p.is_dir() else 0o644)
        self.temp.cleanup()

    def downloader(self, url, target):
        self.calls.append(target.name)
        target.write_bytes(self.data[url.rsplit('/', 1)[1]])

    def run_provision(self, expected=None, downloader=None):
        return provision(self.root, expected or self.expected, runtime_versions=self.expected['versions'],
                         downloader=downloader or self.downloader)

    def test_exact_manifest_and_idempotent_immutable_path(self):
        bundle = self.run_provision()
        self.assertEqual(verified_bundle(bundle, 'kokoro')[1], self.expected)
        self.assertEqual(json.loads((self.root / 'active.json').read_text()),
                         {'bundle': identity(self.expected), 'manifest_sha256': manifest_sha256(self.expected)})
        before = {p.name: p.read_bytes() for p in bundle.iterdir()}
        self.assertEqual(self.run_provision(), bundle)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in bundle.iterdir()})

    def test_interruption_resume_reuses_verified_complete_files(self):
        def interrupted(url, target):
            if target.name == 'voices.bin.part':
                target.write_bytes(b'partial')
                raise KeyboardInterrupt()
            self.downloader(url, target)
        with self.assertRaises(KeyboardInterrupt):
            self.run_provision(downloader=interrupted)
        self.assertFalse((self.root / 'active.json').exists())
        self.run_provision()
        self.assertEqual(self.calls.count('model.onnx.part'), 1)

    def test_disk_failure_and_corruption_preserve_old_active(self):
        old = self.run_provision()
        pointer = (self.root / 'active.json').read_bytes()
        changed = copy.deepcopy(self.expected)
        changed['checkpoint'] = 'second-fixture'
        def disk_full(url, target):
            target.write_bytes(b'x')
            raise OSError(28, 'fixture disk full')
        with self.assertRaises(OSError):
            self.run_provision(changed, disk_full)
        with self.assertRaises(ValueError):
            self.run_provision(changed, lambda u, p: p.write_bytes(b'bad'))
        self.assertEqual((self.root / 'active.json').read_bytes(), pointer)
        self.assertEqual(verified_bundle(old, 'kokoro')[1], self.expected)

    def test_activation_failure_preserves_old(self):
        old = self.run_provision()
        pointer = (self.root / 'active.json').read_bytes()
        changed = copy.deepcopy(self.expected)
        changed['checkpoint'] = 'second'
        import os
        replace = os.replace
        def fail(source, target):
            if Path(target).name == 'active.json':
                raise OSError(28, 'fixture activation failure')
            return replace(source, target)
        with patch('service.provisioning.os.replace', side_effect=fail):
            with self.assertRaises(OSError):
                self.run_provision(changed)
        self.assertEqual((self.root / 'active.json').read_bytes(), pointer)
        self.assertTrue(old.is_dir())
        self.run_provision(changed)

    def test_unsafe_expected_and_runtime_refuse_before_download(self):
        for name in ('../escape', '/escape', 'voices.bin', 'sub/model.onnx'):
            bad = copy.deepcopy(self.expected)
            bad['files'][0]['path'] = name
            with self.assertRaises(ValueError):
                self.run_provision(bad)
        bad = copy.deepcopy(self.expected)
        del bad['files'][0]['bytes']
        with self.assertRaises(ValueError):
            self.run_provision(bad)
        with self.assertRaises(ValueError):
            provision(self.root, self.expected, runtime_versions={}, downloader=self.downloader)
        self.assertEqual(self.calls, [])

    def test_symlink_and_unknown_staging_refuse(self):
        self.root.mkdir(mode=0o700)
        stage = self.root / 'staging'
        stage.mkdir(mode=0o700)
        attempt = stage / identity(self.expected)
        attempt.mkdir(mode=0o700)
        outside = Path(self.temp.name) / 'outside'
        outside.write_bytes(b'untouched')
        (attempt / 'model.onnx').symlink_to(outside)
        with self.assertRaises(ValueError):
            self.run_provision()
        self.assertEqual(outside.read_bytes(), b'untouched')
        (attempt / 'model.onnx').unlink()
        (attempt / 'unexpected').write_bytes(b'x')
        with self.assertRaises(ValueError):
            self.run_provision()

    def test_corrupt_complete_stage_is_reacquired_and_bundle_tamper_refuses(self):
        def stop(url, target):
            if target.name == 'voices.bin.part':
                raise InterruptedError('fixture')
            self.downloader(url, target)
        with self.assertRaises(InterruptedError):
            self.run_provision(downloader=stop)
        stage = self.root / 'staging' / identity(self.expected)
        (stage / 'model.onnx').write_bytes(b'corrupt')
        bundle = self.run_provision()
        self.assertEqual(self.calls.count('model.onnx.part'), 2)
        pointer = (self.root / 'active.json').read_bytes()
        (bundle / 'model.onnx').chmod(0o644)
        (bundle / 'model.onnx').write_bytes(b'tamper')
        with self.assertRaises(Exception):
            self.run_provision()
        self.assertEqual((self.root / 'active.json').read_bytes(), pointer)

    def test_readiness_invalidates_each_binding_and_synthetic(self):
        profile = {'backend': 'kokoro', 'checkpoint': self.expected['checkpoint'], 'device': 'cpu'}
        receipt = dict(qualification='actual', manifest_sha256=manifest_sha256(self.expected),
                       runtime_versions=self.expected['versions'], profile_sha256=identity(profile))
        def ready(e=self.expected, v=self.expected['versions'], p=profile):
            return readiness_matches(receipt, expected=e, runtime_versions=v, profile=p)
        self.assertTrue(ready())
        receipt['qualification'] = 'synthetic'
        self.assertFalse(ready())
        receipt['qualification'] = 'actual'
        self.assertFalse(ready(v={}))
        self.assertFalse(ready(p={**profile, 'device': 'cuda'}))
        self.assertFalse(ready(e={**self.expected, 'checkpoint': 'changed'}))

    def test_legacy_manifest_compatible_size_and_symlink_checked(self):
        bundle = self.run_provision()
        # Legacy schema remains readable, including Pocket manifests without bytes.
        legacy = Path(self.temp.name) / 'legacy'
        legacy.mkdir()
        (legacy / 'config.yaml').write_bytes(b'config')
        manifest = dict(schema_version=1, backend='pocket-tts', language='fr', checkpoint='pocket-fixture',
                        files=[dict(path='config.yaml', sha256=hashlib.sha256(b'config').hexdigest())])
        (legacy / 'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(verified_bundle(legacy, 'pocket-tts')[1], manifest)
        (legacy / 'config.yaml').unlink()
        (legacy / 'config.yaml').symlink_to(bundle / 'model.onnx')
        with self.assertRaises(Exception):
            verified_bundle(legacy, 'pocket-tts')


class RecoveryAndBoundsTests(unittest.TestCase):
    setUp = ProvisioningTests.setUp
    cleanup = ProvisioningTests.cleanup
    downloader = ProvisioningTests.downloader
    run_provision = ProvisioningTests.run_provision

    def test_final_directory_seal_failure_retry(self):
        chmod = Path.chmod
        def fail(path, mode, **kwargs):
            if path.parent.name == 'bundles' and mode == 0o555:
                raise OSError('inert final sealing failure')
            return chmod(path, mode, **kwargs)
        with patch.object(Path, 'chmod', fail):
            with self.assertRaises(OSError): self.run_provision()
        self.assertFalse((self.root / 'active.json').exists())
        bundle = self.run_provision()
        self.assertFalse(bundle.stat().st_mode & 0o222)
        self.assertTrue(all(not p.stat().st_mode & 0o222 for p in bundle.iterdir()))

    def test_partial_file_sealing_resumes_without_redownload(self):
        chmod = Path.chmod
        def fail(path, mode, **kwargs):
            if path.name == 'voices.bin' and mode == 0o444:
                raise OSError('inert partial seal')
            return chmod(path, mode, **kwargs)
        with patch.object(Path, 'chmod', fail):
            with self.assertRaises(OSError): self.run_provision()
        self.run_provision()
        self.assertEqual(len(self.calls), 2)

    def test_stream_expected_size_and_overflow_probe(self):
        import io
        from service.provisioning import _download_stream
        for payload, success in ((b'12345', True), (b'1234', False), (b'x'*100, False)):
            target = Path(self.temp.name) / 'transfer'
            source = io.BytesIO(payload)
            with patch('service.provisioning.urllib.request.urlopen', return_value=source):
                if success: _download_stream('inert', target, 5)
                else:
                    with self.assertRaises(ValueError): _download_stream('inert', target, 5)
            self.assertLessEqual(target.stat().st_size, 5)
            target.unlink()

    def test_real_transfer_deadline_and_cancellation_reap(self):
        import time
        import multiprocessing
        from service.provisioning import download
        def stalled(*args, **kwargs):
            time.sleep(10)
        before = {p.pid for p in multiprocessing.active_children()}
        with patch('service.provisioning.urllib.request.urlopen', stalled):
            for kwargs, error in ((dict(timeout=.03), TimeoutError), (dict(cancelled=lambda: True), InterruptedError)):
                with self.assertRaises(error):
                    download('inert', Path(self.temp.name) / 'unused', 5, **kwargs)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_managed_carries_expected_bytes(self):
        from service import provisioning as owner
        sizes = []
        def fake(url, target, size, **kwargs):
            sizes.append(size)
            self.downloader(url, target)
        # Default callable binding is explicit here, preserving the trusted seam.
        with patch.object(owner, 'download', fake):
            provision(self.root, self.expected, runtime_versions=self.expected['versions'], downloader=fake)
        self.assertEqual(sizes, [5, 6])

    def test_lock_failure_closes_descriptor_and_wait_is_finite(self):
        import os
        from service import provisioning as owner
        opened = []
        original = os.open
        def capture(*args):
            fd = original(*args); opened.append(fd); return fd
        for error in (OSError('inert flock failure'), BlockingIOError()):
            with patch.object(owner.os, 'open', capture), patch.object(owner.fcntl, 'flock', side_effect=error):
                with self.assertRaises((OSError, TimeoutError)):
                    provision(self.root, self.expected, runtime_versions=self.expected['versions'], lock_timeout=.01, downloader=self.downloader)
            with self.assertRaises(OSError): os.fstat(opened[-1])
        self.assertEqual(self.calls, [])

    def test_private_root_fifo_and_hardlink_admission(self):
        import os
        self.root.mkdir(mode=0o755)
        # Keep the intentionally non-private mode independent of the process umask.
        self.root.chmod(0o755)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o755)
        with self.assertRaises(ValueError): self.run_provision()
        self.root.chmod(0o700)
        lock = self.root / '.provision.lock'
        os.mkfifo(lock)
        with self.assertRaises(ValueError): self.run_provision()
        lock.unlink()
        def stop(url, target): raise InterruptedError()
        with self.assertRaises(InterruptedError): self.run_provision(downloader=stop)
        stage = self.root / 'staging' / identity(self.expected)
        outside = Path(self.temp.name) / 'outside'
        outside.write_bytes(b'model')
        os.link(outside, stage / 'model.onnx')
        mode = outside.stat().st_mode
        with self.assertRaises(ValueError): self.run_provision()
        self.assertEqual(outside.stat().st_mode, mode)

    def test_schema_bool_and_missing_model_revision_refuse(self):
        for change in (dict(schema_version=True), dict(revision='')):
            with self.assertRaises(ValueError): self.run_provision({**self.expected, **change})

    def test_actual_managed_downloader_with_inert_urlopen(self):
        import io
        def source(url, **kwargs):
            return io.BytesIO(self.data[url.rsplit('/', 1)[1]])
        with patch('service.provisioning.urllib.request.urlopen', source):
            bundle = provision(self.root, self.expected, runtime_versions=self.expected['versions'])
        self.assertEqual(verified_bundle(bundle, 'kokoro')[1], self.expected)

    def test_lock_cancellation_before_acquisition(self):
        with self.assertRaises(InterruptedError):
            provision(self.root, self.expected, runtime_versions=self.expected['versions'],
                      downloader=self.downloader, cancelled=lambda: True)
        self.assertEqual(self.calls, [])

    def test_manifest_engine_read_limit(self):
        bad = {**self.expected, 'revision': 'x' * 65536}
        with self.assertRaises(ValueError): self.run_provision(bad)


class PocketProvisioningTests(unittest.TestCase):
    cleanup = ProvisioningTests.cleanup
    downloader = ProvisioningTests.downloader
    run_provision = ProvisioningTests.run_provision
    def setUp(self):
        ProvisioningTests.setUp(self)
        from service.provisioning import manifest_bytes
        self.data = {'model.safetensors': b'inert weights', 'tokenizer.json': b'{}'}
        self.expected.update(backend='pocket-tts', revision='a'*40,
                             versions={p: 'fixture' for p in ('pocket-tts', 'torch', 'tokenizers')},
                             allowed_origins=['https://fixture.invalid'],
                             config={'weights_path': 'https://fixture.invalid/model.safetensors',
                                     'tokenizer': {'tokenizer_path': 'tokenizer.json'}})
        config = manifest_bytes({'weights_path': 'model.safetensors', 'tokenizer': {'tokenizer_path': 'tokenizer.json'}})
        self.expected['files'] = [dict(path=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                                      origin='https://fixture.invalid/'+n, revision='a'*40)
                                  for n, b in {**self.data, 'config.yaml': config}.items()]

    # Kokoro-specific inherited tests remain in their original class.
    def test_generated_bundle_consumed_by_real_adapter_and_worker_verifier(self):
        import dataclasses
        import io
        import sys
        import types
        import wave
        from service.provisioning import bundle_manifest
        from worker.engines.pocket import verify_inputs
        from worker.engines.pocket_execution import PocketExecution
        from worker.transport import Refused
        from lib.tts_backends.pocket_tts import PocketBackend
        for language in ('fr', 'en'):
            self.expected['language'] = language
            bundle = self.run_provision()
            self.assertEqual(verified_bundle(bundle, 'pocket-tts')[1], bundle_manifest(self.expected))
            self.assertEqual(hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(), manifest_sha256(self.expected))
            self.assertFalse(bundle.stat().st_mode & 0o222)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000); wav.writeframes(b'\0\0'*4)
            raw = buffer.getvalue()
            ref = Path(self.temp.name)/'reference.wav'; ref.write_bytes(raw)
            spec = PocketExecution('bundle', manifest_sha256(self.expected), self.expected['checkpoint'], self.expected['revision'], language,
                                   2, 1024, 100, 244, 1024*1024, 'reference', hashlib.sha256(raw).hexdigest(), len(raw), 4, 24000, 24000)
            self.assertEqual(verify_inputs(spec, {'bundle': bundle}, {'reference': ref}), raw)
            for change in (dict(language='en' if language == 'fr' else 'fr'), dict(revision='b'*40), dict(checkpoint='other'), dict(manifest_sha256='0'*64)):
                with self.assertRaises(Refused):
                    verify_inputs(dataclasses.replace(spec, **change), {'bundle': bundle}, {'reference': ref})
            seen = []; loader_directories = []
            previous_directory = Path.cwd()
            class Model:
                @classmethod
                def load_model(cls, config):
                    compiled = json.loads(Path(config).read_text())
                    seen.append(compiled)
                    loader_directories.append(Path.cwd())
                    return cls()
            with patch.dict(sys.modules, {'torch': types.SimpleNamespace(set_num_threads=lambda n: None),
                                          'pocket_tts': types.SimpleNamespace(TTSModel=Model),
                                          'yaml': types.SimpleNamespace(safe_load=json.loads, safe_dump=lambda v, f: json.dump(v, f))}):
                backend = PocketBackend(bundle, language=language)
                backend.close()
            self.assertEqual(seen[0]['weights_path'], 'model.safetensors')
            self.assertEqual(seen[0]['tokenizer']['tokenizer_path'], 'tokenizer.json')
            self.assertEqual(loader_directories, [bundle.resolve()])
            self.assertEqual(Path.cwd(), previous_directory)
            self.assertTrue((loader_directories[0]/seen[0]['weights_path']).samefile(bundle/'model.safetensors'))
            self.assertTrue((loader_directories[0]/seen[0]['tokenizer']['tokenizer_path']).samefile(bundle/'tokenizer.json'))
            calls = len(self.calls)
            self.assertEqual(self.run_provision(), bundle)
            self.assertEqual(len(self.calls), calls)

    def test_closed_pocket_descriptor_negatives(self):
        mutations = [dict(backend='unknown'), dict(language='de'), dict(revision='main'), dict(extra=True),
                     dict(versions={'pocket-tts': 'fixture'}), dict(allowed_origins=['https://else.invalid']),
                     dict(config={'weights_path': '../model.safetensors'}),
                     dict(config={'weights_path': 'model.safetensors', 'language': 'fr'}),
                     dict(config={'weights_path': 'model.safetensors', 'other': 'hf://repo/model'}),
                     dict(config={'weights_path': 'model.safetensors', 'secret': 'https://user:password@host/model'})]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_provision({**self.expected, **change})
        for field, value in (('path', '../escape'), ('path', 'model.pkl'), ('origin', 'https://user:secret@fixture.invalid/x'),
                             ('revision', 'b'*40), ('bytes', 2*1024**3+1)):
            bad = copy.deepcopy(self.expected); bad['files'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.run_provision(bad)
        bad = copy.deepcopy(self.expected); bad['files'].append(bad['files'][0])
        with self.assertRaises(ValueError): self.run_provision(bad)
        self.assertEqual(self.calls, [])

    def test_pocket_seal_retry_and_bad_transfer(self):
        RecoveryAndBoundsTests.test_final_directory_seal_failure_retry(self)
        pointer = (self.root/'active.json').read_bytes()
        changed = {**self.expected, 'checkpoint': 'bad-transfer'}
        with self.assertRaises(ValueError):
            self.run_provision(changed, lambda url, target: target.write_bytes(b'x'*100))
        self.assertEqual((self.root/'active.json').read_bytes(), pointer)

    def test_pocket_transport_allowed_redirects_and_exact_caps(self):
        import io
        import urllib.request
        from service.provisioning import _AllowedRedirects, _download_stream
        handler = _AllowedRedirects(['https://fixture.invalid'])
        request = urllib.request.Request('https://fixture.invalid/model')
        for url in ('http://fixture.invalid/x', 'https://other.invalid/x', 'https://user:secret@fixture.invalid/x', 'https://fixture.invalid/x?token=secret'):
            with self.assertRaises(ValueError): handler.redirect_request(request, None, 302, '', {}, url)
        self.assertEqual(handler.redirect_request(request, None, 302, '', {}, 'https://fixture.invalid/next').full_url, 'https://fixture.invalid/next')
        for data, succeeds in ((b'12345', True), (b'1234', False), (b'123456', False)):
            target = Path(self.temp.name)/'bounded'
            opener = type('Opener', (), {'open': lambda self, *a, **k: io.BytesIO(data)})()
            with patch('service.provisioning.urllib.request.build_opener', return_value=opener):
                if succeeds: _download_stream(request.full_url, target, 5, ['https://fixture.invalid'])
                else:
                    with self.assertRaises(ValueError): _download_stream(request.full_url, target, 5, ['https://fixture.invalid'])
            self.assertLessEqual(target.stat().st_size, 5)
            target.unlink()

    def test_pocket_resume_and_preserve_active(self):
        old = self.run_provision(); pointer = (self.root/'active.json').read_bytes()
        changed = {**self.expected, 'checkpoint': 'second'}
        def stop(url, target):
            if target.name == 'tokenizer.json.part':
                target.write_bytes(b'partial'); raise InterruptedError()
            self.downloader(url, target)
        with self.assertRaises(InterruptedError): self.run_provision(changed, stop)
        self.assertEqual((self.root/'active.json').read_bytes(), pointer)
        calls = self.calls.count('model.safetensors.part')
        self.run_provision(changed)
        self.assertEqual(self.calls.count('model.safetensors.part'), calls)
        self.assertTrue(old.exists())

if __name__ == '__main__':
    unittest.main()
