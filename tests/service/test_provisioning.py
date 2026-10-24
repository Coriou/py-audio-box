"""Unittest only: private temporary files, injected downloader, no ML/network."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from service.provisioning import provision, identity, readiness_matches, validate_expected
from lib.tts_backends.assets import verified_bundle


class ProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.cleanup)
        self.root = Path(self.temp.name) / 'models'
        self.data = {'model.onnx': b'model', 'voices.bin': b'voices'}
        self.expected = dict(schema_version=1, backend='kokoro', language='en', checkpoint='fixture-kokoro',
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
                         {'bundle': identity(self.expected), 'manifest_sha256': identity(self.expected)})
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
        self.root.mkdir()
        stage = self.root / 'staging'
        stage.mkdir()
        attempt = stage / identity(self.expected)
        attempt.mkdir()
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
        receipt = dict(qualification='actual', manifest_sha256=identity(self.expected),
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


if __name__ == '__main__':
    unittest.main()
