"""Dirty-source archive closure only; no installation or runtime qualification."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from service import native_package, package_install

ROOT = Path(__file__).resolve().parents[2]


class SpokenPackageTests(unittest.TestCase):
    def test_both_archives_retain_spoken_runtime_and_documentation(self):
        required = {'lib/spoken_text.py', 'docs/TTS_SPOKEN_TEXT.md'}
        for manifest, prefix, build in (
            ('headless-dist-manifest.json', 'speech-box/', lambda p: package_install.archive(ROOT, p)),
            ('native-dist-manifest.json', 'native-worker/', native_package.archive),
        ):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as tmp:
                files = json.loads((ROOT / manifest).read_text())['files']
                self.assertTrue(required <= set(files))
                self.assertEqual(len(files), len(set(files)))
                self.assertLess(sum((ROOT / name).stat().st_size for name in files), 16 * 1024 * 1024)
                destination = Path(tmp) / 'dirty-source-test-only.zip'
                build(destination)
                with zipfile.ZipFile(destination) as archive:
                    for name in required:
                        self.assertEqual(archive.read(prefix + name), (ROOT / name).read_bytes())
                    self.assertEqual(
                        {name for name in archive.namelist() if name.endswith('.py') and '/lib/' in name},
                        {prefix + name for name in files if name.startswith('lib/') and name.endswith('.py')},
                    )
