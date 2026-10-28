"""No-weight native prerequisite failures, immutable pins and archive closure."""
from importlib import metadata
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from lib.tts_backends import kokoro_native as n
from lib.tts_backends.contract import BackendError

ROOT = Path(__file__).resolve().parents[1]


class NativeKokoroTests(unittest.TestCase):
    def test_native_platform_errors_are_actionable(self):
        with patch.object(n.platform, 'system', return_value='Darwin'), patch.object(n.platform, 'machine', return_value='x86_64'):
            with self.assertRaisesRegex(BackendError, 'Rosetta') as e: n.validate_host()
        self.assertEqual(e.exception.code, 'native_kokoro_platform')

    def test_missing_or_wrong_dependency_retains_cause(self):
        with patch.object(n.metadata, 'version', side_effect=metadata.PackageNotFoundError('kokoro-onnx')):
            with self.assertRaisesRegex(BackendError, 'require-hashes') as e: n.validate_versions()
        self.assertIsInstance(e.exception.__cause__, metadata.PackageNotFoundError)
        with patch.object(n.metadata, 'version', return_value='wrong'):
            with self.assertRaisesRegex(BackendError, 'pinned 0.6.1'): n.validate_versions()

    def test_explicit_bundled_espeak_no_system_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); library=root/'libespeak-ng.dylib'; library.touch()
            data=root/'espeak-ng-data'; data.mkdir(); (data/'en_dict').touch()
            loader=SimpleNamespace(__file__=str(root/'__init__.py'), get_library_path=lambda:str(library), get_data_path=lambda:str(data))
            with patch.object(n.ctypes, 'CDLL', side_effect=OSError('wrong architecture')):
                with self.assertRaisesRegex(BackendError, 'arm64') as e: n.espeak_paths(loader)
            self.assertEqual(str(e.exception.__cause__), 'wrong architecture')
            with patch.object(n.ctypes, 'CDLL') as load:
                self.assertEqual(n.espeak_paths(loader),(library,data)); load.assert_called_once_with(str(library))
            (data/'en_dict').unlink()
            with self.assertRaisesRegex(BackendError,'incomplete'): n.espeak_paths(loader)

    def test_pins_and_archive_inputs_without_widening_admission(self):
        pins=dict(re.findall(r'^([a-z0-9-]+)==([^\s]+)',(ROOT/'requirements/kokoro.lock').read_text(),re.M))
        for name,version in n.EXPECTED.items(): self.assertEqual(pins[name],version)
        from service import native_package as package
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'native.zip'; package.archive(archive); package.verify_archive(archive)
            with zipfile.ZipFile(archive) as z:
                for name in ('lib/tts_backends/kokoro_native.py','docs/agent-interface/NATIVE-KOKORO.md'):
                    self.assertEqual(z.read('native-worker/'+name),(ROOT/name).read_bytes())
        self.assertEqual(package.VARIANT,'native-qwen-v1')
        workflow=(ROOT/'.github/workflows/apple-kokoro.yml').read_text()
        self.assertIn('--require-hashes --only-binary=:all:',workflow)
        self.assertIn('-r requirements/kokoro.lock',workflow)
        self.assertNotIn('qwen_speech',workflow)
        self.assertNotIn('brew ',workflow)
        image=(ROOT/'Dockerfile.tests').read_text()
        self.assertIn('COPY .github/workflows/apple-kokoro.yml /app/.github/workflows/apple-kokoro.yml',image)
        self.assertIn('docs/agent-interface/NATIVE-KOKORO.md',image)


if __name__ == '__main__': unittest.main()
