"""Inert distribution boundary checks: real archives, no Docker or model access."""
import fnmatch
import hashlib
import json
import re
import shlex
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

from service import native_package, package_install

ROOT = Path(__file__).resolve().parents[1]
LEGAL = ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'docs/SOURCE-LICENSE.md')
GNU_SHA256 = '0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0'
IMAGES = ('Dockerfile', 'Dockerfile.cpu', 'Dockerfile.cuda', 'Dockerfile.cuda-base',
          'Dockerfile.watcher', 'Dockerfile.tests', 'service/Dockerfile.release')


class SourceDistributionTests(unittest.TestCase):
    def test_canonical_license_and_source_election(self):
        data = (ROOT / 'LICENSE').read_bytes()
        self.assertEqual(len(data), 34523)
        self.assertEqual(hashlib.sha256(data).hexdigest(), GNU_SHA256)
        meta = tomllib.loads((ROOT / 'pyproject.toml').read_text())['tool']
        self.assertEqual(meta['poetry']['license'], 'AGPL-3.0-only')
        self.assertFalse(meta['poetry']['package-mode'])
        self.assertEqual(meta['speech-box-headless']['license'], 'AGPL-3.0-only')
        self.assertEqual(meta['speech-box-headless']['license-files'], list(LEGAL[:2]))
        package = json.loads((ROOT / 'studio/package.json').read_text())
        lock = json.loads((ROOT / 'studio/package-lock.json').read_text())
        self.assertTrue(package['private'])
        self.assertEqual(package['license'], 'AGPL-3.0-only')
        self.assertEqual(lock['packages']['']['license'], package['license'])

    def test_real_archives_extract_exact_declared_bytes(self):
        for manifest, prefix, build in (
            ('headless-dist-manifest.json', 'speech-box', lambda p: package_install.archive(ROOT, p)),
            ('native-dist-manifest.json', 'native-worker', native_package.archive),
        ):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as tmp:
                files = json.loads((ROOT / manifest).read_text())['files']
                self.assertEqual(len(files), len(set(files)))
                self.assertTrue(set(LEGAL) <= set(files))
                first, second = Path(tmp) / 'one.zip', Path(tmp) / 'two.zip'
                self.assertEqual(build(first), build(second))
                if prefix == 'native-worker':
                    native_package.verify_archive(first)
                launcher = 'speech-box' if prefix == 'speech-box' else 'native-worker.py'
                with zipfile.ZipFile(first) as archive:
                    names = archive.namelist()
                    self.assertEqual(len(names), len(set(names)))
                    self.assertEqual(set(names), {prefix + '/' + n for n in files + [launcher]})
                    for name in files:
                        self.assertFalse(Path(name).is_absolute())
                        self.assertNotIn('..', Path(name).parts)
                        self.assertFalse(any(p in Path(name).parts for p in
                                             ('.env', '.git', 'cache', '__pycache__')))
                    archive.extractall(Path(tmp) / 'extracted')
                for name in files:
                    self.assertEqual((Path(tmp) / 'extracted' / prefix / name).read_bytes(),
                                     (ROOT / name).read_bytes(), name)

    def test_every_image_stage_carries_legal_files_and_scoped_label(self):
        for name in IMAGES:
            text = (ROOT / name).read_text()
            stages = re.split(r'^FROM .+\n', text, flags=re.M)[1:]
            self.assertTrue(stages, name)
            for index, stage in enumerate(stages):
                with self.subTest(image=name, stage=index):
                    self.assertIn('LABEL org.opencontainers.image.licenses="AGPL-3.0-only"', stage)
                    self.assertIn('project source', stage)
                    dest = '/opt/speech/' if name.startswith('service/') else '/app/'
                    self.assertIn('COPY LICENSE THIRD_PARTY_NOTICES.md ' + dest, stage)
            ignore = ROOT / (name + '.dockerignore')
            rules = (ignore if ignore.exists() else ROOT / '.dockerignore').read_text().splitlines()
            for legal in LEGAL[:2]:
                included = True
                for rule in rules:
                    rule = rule.strip()
                    if not rule or rule.startswith('#'):
                        continue
                    pattern = rule.lstrip('!')
                    if fnmatch.fnmatchcase(legal, pattern):
                        included = rule.startswith('!')
                self.assertTrue(included, (name, legal))
        rules = (ROOT / 'service/Dockerfile.release.dockerignore').read_text().splitlines()
        self.assertEqual(rules[0], '**')
        self.assertEqual([r for r in rules if r in ('!LICENSE', '!THIRD_PARTY_NOTICES.md')],
                         ['!LICENSE', '!THIRD_PARTY_NOTICES.md'])

    def test_test_image_copies_static_check_inputs(self):
        # Expand actual COPY sources to verify the tests can run in the narrow image.
        copied = set()
        for line in (ROOT / 'Dockerfile.tests').read_text().splitlines():
            if line.startswith('COPY '):
                parts = shlex.split(line)[1:]
                for source in parts[:-1]:
                    path = ROOT / source
                    self.assertTrue(path.exists(), source)
                    if path.is_dir():
                        copied.update(str(p.relative_to(ROOT)) for p in path.rglob('*') if p.is_file())
                    else:
                        copied.add(source)
        self.assertTrue(set(IMAGES + LEGAL + ('studio/package.json', 'studio/package-lock.json',
                                             'service/Dockerfile.release.dockerignore', '.dockerignore')) <= copied)


if __name__ == '__main__':
    unittest.main()
