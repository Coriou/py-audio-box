"""Injected r378 contract inputs only; no installation or qualification proof."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from jsonschema.exceptions import ValidationError

from service import installed_export as ex, install_policy as ip
from service.admission import DEFAULT_BUDGET
from tests.service.test_engine_admission import configuration
from tests.service import test_provisioning


class InstalledExportContractTests(unittest.TestCase):
    def setUp(self):
        fixture = test_provisioning.ProvisioningTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        context, profiles = configuration(fixture.expected)
        spec = context['execution']['descriptor']
        bid = spec['bundle_id']
        export = json.loads(Path('contracts/fixtures/v1/transcript-audio_export.json').read_text())
        converter = export['context']['execution']['converter']
        converter['worker_image_digest'] = 'sha256:' + '2' * 64
        self.config = dict(version=2, ownership_policy=ex.POLICY,
            profiles=profiles, budget=copy.deepcopy(DEFAULT_BUDGET),
            contexts={'alice': context},
            bundles={bid: dict(manifest_sha256=spec['manifest_sha256'],
                checkpoint=spec['checkpoint'], revision=spec['revision'],
                path='/models/bundles/' + bid)},
            images=dict(control='sha256:' + '1' * 64, worker='sha256:' + '2' * 64),
            mounts=dict(control='/control', staging='/staging', receipts='/receipts',
                        ipc='/ipc', models='/models'),
            audio_export=dict(converter=converter, profile=export['profiles'][0], principals=['alice']))

    def test_closed_config_detached_and_unqualified_preserved(self):
        checked = ex.configuration(self.config)
        self.assertEqual(checked, self.config)
        self.assertIsNot(checked, self.config)
        self.assertEqual(checked['audio_export']['profile'], self.config['audio_export']['profile'])
        self.assertEqual(len(ex.admission_profiles(checked)), 2)
        checked['audio_export']['principals'].clear()
        self.assertEqual(self.config['audio_export']['principals'], ['alice'])

    def test_legacy_parser_still_rejects_new_family(self):
        with self.assertRaises(ValueError):
            ip.configuration(self.config)
        legacy = {k: copy.deepcopy(v) for k, v in self.config.items() if k != 'audio_export'}
        legacy.update(version=1, ownership_policy=ip.POLICY)
        self.assertEqual(ip.configuration(legacy), legacy)
        with self.assertRaises(ValueError):
            ex.configuration(legacy)

    def test_strict_versions_and_family(self):
        for version in (True, 2.0, 1, '2', None):
            with self.subTest(version=version), self.assertRaises(ValueError):
                ex.configuration(dict(self.config, version=version))
        with self.assertRaises(ValueError):
            ex.configuration(dict(self.config, ownership_policy=ip.POLICY))

    def test_unknown_missing_and_oversized_records(self):
        for field in self.config:
            c = copy.deepcopy(self.config)
            del c[field]
            with self.subTest(field=field), self.assertRaises((ValueError, ValidationError)):
                ex.configuration(c)
        c = copy.deepcopy(self.config)
        c['audio_export']['extra'] = True
        with self.assertRaises(ValueError):
            ex.configuration(c)
        c['audio_export'].pop('extra')
        c['audio_export']['converter']['reported_build'] = 'x' * 48000
        with self.assertRaisesRegex(ValueError, 'installed_configuration_budget'):
            ex.configuration(c)

    def test_principals_canonical_nonempty_unique_configured(self):
        for principals in ([], ['alice', 'alice'], ['bob'], ['alice', 1], 'alice'):
            c = copy.deepcopy(self.config)
            c['audio_export']['principals'] = principals
            with self.subTest(principals=principals), self.assertRaises(ValueError):
                ex.configuration(c)

    def test_converter_pin_required_and_image_bound(self):
        for field in self.config['audio_export']['converter']:
            c = copy.deepcopy(self.config)
            del c['audio_export']['converter'][field]
            with self.subTest(field=field), self.assertRaises((ValueError, ValidationError)):
                ex.configuration(c)
        c = copy.deepcopy(self.config)
        c['images']['worker'] = 'sha256:' + '3' * 64
        with self.assertRaisesRegex(ValueError, 'converter_image_mismatch'):
            ex.configuration(c)

    def test_export_profile_requires_cpu_networkless_empty_compatibility(self):
        for key, value in (('device', 'cuda'), ('topology', 'native'), ('id', 'kokoro-cpu')):
            c = copy.deepcopy(self.config)
            c['audio_export']['profile'][key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, ValidationError)):
                ex.configuration(c)

    def test_seal_exact_canonical_all_fields_and_version(self):
        seal = ex.export_seal(self.config)
        self.assertEqual(ex.validate_seal(seal, self.config), seal)
        for key in seal:
            changed = dict(seal)
            changed[key] = 1 if key == 'publication_format' else 'f' * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                ex.validate_seal(changed, self.config)
        for version in (True, 2.0):
            with self.assertRaises(ValueError):
                ex.validate_seal(dict(seal, publication_format=version), self.config)
        with self.assertRaises(ValueError):
            ex.validate_seal(dict(seal, extra=True), self.config)
        changed = copy.deepcopy(self.config)
        changed['audio_export']['profile']['qualification_receipt'] = 'INJECTED changed assertion'
        with self.assertRaises(ValueError):
            ex.validate_seal(seal, changed)

    def local_converter(self, root):
        exe = root / 'converter'
        exe.write_bytes(b'INERT never executed')
        exe.chmod(0o555)
        lib = root / 'runtime'
        lib.write_bytes(b'INERT runtime')
        lib.chmod(0o444)
        c = copy.deepcopy(self.config)
        c['audio_export']['converter'].update(executable=str(exe),
            executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), runtime_closure=[str(lib)])
        return c, exe, lib

    def test_local_immutable_closure_and_hash_no_execution(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            c, exe, lib = self.local_converter(Path(tmp))
            fd = ex.verify_local_converter(c, owner_uid=os.getuid())
            try:
                self.assertEqual(os.read(fd, 100), b'INERT never executed')
            finally:
                os.close(fd)
            for path, mode in ((exe, 0o755), (lib, 0o644)):
                path.chmod(mode)
                with self.assertRaises(ValueError):
                    ex.verify_local_converter(c, owner_uid=os.getuid())
                path.chmod(0o555 if path == exe else 0o444)
            exe.chmod(0o755)
            exe.write_bytes(b'changed')
            exe.chmod(0o555)
            with self.assertRaises(ValueError):
                ex.verify_local_converter(c, owner_uid=os.getuid())

    def test_local_alias_and_symlink_closure_refuse(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            c, exe, lib = self.local_converter(Path(tmp))
            alias = Path(tmp) / 'alias'
            alias.symlink_to(lib)
            c['audio_export']['converter']['runtime_closure'] = [str(alias)]
            with self.assertRaises((ValueError, OSError)):
                ex.verify_local_converter(c, owner_uid=os.getuid())
            c['audio_export']['converter']['runtime_closure'] = [str(exe)]
            with self.assertRaises(ValueError):
                ex.verify_local_converter(c, owner_uid=os.getuid())
