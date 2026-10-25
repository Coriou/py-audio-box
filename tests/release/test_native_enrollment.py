"""Tiny durable SOURCE bindings; fake TLS/ML/processes, no installed authority."""
import copy
import os
import unittest
from unittest.mock import patch

from service import native_package as n
from service.admission import DEFAULT_BUDGET
from service.execution_context import NativeQwenContexts
from tests.service.test_native_runtime import NativeFixture
from worker.engines.qwen_execution import inventory_wire
from worker.transport import Refused, canonical


class NativeEnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.f = NativeFixture(self)
        self.root = self.f.f.fixture.root
        self.root.chmod(0o700)
        self.owner = self.root/'owner'; self.owner.mkdir(mode=0o700)
        for name in ('ca', 'cert', 'key'): self.write(name, b'INERT CERTIFICATE')
        self.config = dict(version=1, variant=n.VARIANT, native=self.f.native,
            host='127.0.0.1', port=4443, peer_sha256=self.f.pin,
            cafile=str(self.root/'ca'), certfile=str(self.root/'cert'), keyfile=str(self.root/'key'),
            spool=str(self.f.spool), staging=str(self.f.stage))
        context = self.f.snapshot[1]
        self.templates = dict(version=1, variant=n.VARIANT, profiles=self.f.snapshot[3],
            budget=DEFAULT_BUDGET, inventory=inventory_wire(self.f.f.inventory),
            registrations=[dict(principal='alice',profile='qwen-mps',voice_version='vv0',context=context)])
        self.write('control', canonical(self.config)); self.write('templates', canonical(self.templates))
        self.record = self.root/'enrollment'
        self.kw = dict(configuration_file=str(self.root/'control'), templates_file=str(self.root/'templates'),
                       owner_root=str(self.owner), connector=self.f.connect)
        tls = patch.object(n.p, 'tls_context', return_value=None); tls.start(); self.addCleanup(tls.stop)

    def write(self, name, raw):
        path = self.root/name; path.write_bytes(raw); path.chmod(0o600); return path

    def enroll(self): return n.enroll_native(self.record, **self.kw)
    def reopen(self): return n.native_startup_inputs(self.record, **self.kw)

    def test_durable_reopen_actual_context_and_runtime(self):
        saved = self.enroll()
        self.assertEqual(n.document(self.record), saved)
        result = self.reopen()
        self.assertIsInstance(result['production_contexts'], NativeQwenContexts)
        request = self.templates['registrations'][0]['context']['plan']['request']
        self.assertEqual(result['production_contexts'].resolve('alice','speech_generate',request)['execution']['native'], self.f.native)
        self.assertEqual(result['runtime'].native, self.f.native)
        self.assertEqual(self.record.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError): self.enroll()

    def test_missing_malformed_and_partial_record_refuse_without_repair(self):
        with self.assertRaises(FileNotFoundError): self.reopen()
        for raw in (b'{', b'{}', b'{"version":1,"version":1}'):
            self.write('enrollment', raw)
            with self.assertRaises((ValueError, Refused)): self.reopen()
            self.assertEqual(self.record.read_bytes(), raw)

    def record_type_substitutions(self, saved):
        for value in (True, 1.0):
            changed = copy.deepcopy(saved); changed['version'] = value
            yield 'version', value, changed
        for field in ('device', 'inode', 'uid'):
            changed = copy.deepcopy(saved)
            changed['owner'][field] = float(saved['owner'][field])
            # Establish that this specifically exercises Python's equality gap.
            self.assertEqual(changed, saved)
            yield field, changed['owner'][field], changed

    def test_record_type_substitutions_refuse_before_tls(self):
        saved = self.enroll()
        for field, value, changed in self.record_type_substitutions(saved):
            with self.subTest(field=field, value=value):
                raw = canonical(changed); self.write('enrollment', raw)
                count = len(self.f.requests)
                with self.assertRaisesRegex(Refused, 'native_enrollment_binding'): self.reopen()
                self.assertEqual(len(self.f.requests), count)
                self.assertEqual(self.record.read_bytes(), raw)

    def test_record_type_substitutions_during_hello_refuse(self):
        saved = self.enroll()
        for field, value, changed in self.record_type_substitutions(saved):
            with self.subTest(field=field, value=value):
                self.write('enrollment', canonical(saved))
                raw = canonical(changed)
                def mutate(q, response):
                    if q['op'] == 'hello': self.write('enrollment', raw)
                    return response
                self.f.mutate = mutate
                with self.assertRaisesRegex(Refused, 'native_enrollment_changed'): self.reopen()
                self.assertEqual(self.record.read_bytes(), raw)

    def test_changed_trust_configuration_templates_and_owner_refuse_before_tls(self):
        self.enroll()
        for name in ('key', 'control', 'templates'):
            original = (self.root/name).read_bytes()
            if name == 'key': raw = b'CHANGED'
            else:
                value = copy.deepcopy(self.config if name == 'control' else self.templates)
                if name == 'control': value['port'] += 1
                else: value['budget'] = dict(DEFAULT_BUDGET, queue=1)
                raw = canonical(value)
            self.write(name, raw)
            count = len(self.f.requests)
            with self.assertRaisesRegex(Refused, 'native_enrollment_binding'): self.reopen()
            self.assertEqual(len(self.f.requests), count)
            self.write(name, original)
        old = self.root/'old-owner'; self.owner.rename(old); self.owner.mkdir(mode=0o700)
        with self.assertRaisesRegex(Refused, 'native_enrollment_binding'): self.reopen()

    def test_boot_and_quarantine_refuse_without_learning(self):
        self.enroll(); saved = self.record.read_bytes()
        for field, value in (('boot','f'*32), ('quarantined',True)):
            self.f.mutate = lambda q, v: dict(v, **{field:value}) if q['op']=='hello' else v
            with self.assertRaises(Refused): self.reopen()
            self.assertEqual(self.record.read_bytes(), saved)

    def test_changed_native_template_and_duplicate_registration_refuse(self):
        for mutation in ('native', 'duplicate'):
            value = copy.deepcopy(self.templates)
            if mutation == 'native': value['registrations'][0]['context']['execution']['native']['boot']='e'*32
            else: value['registrations'] *= 2
            self.write('templates', canonical(value))
            with self.assertRaises(Refused): self.enroll()
            self.assertFalse(self.record.exists())

    def test_symlink_hardlink_and_permissive_record_refuse(self):
        self.enroll(); saved = self.record.read_bytes()
        self.record.chmod(0o644)
        with self.assertRaisesRegex(Refused,'native_package_file'): self.reopen()
        self.record.chmod(0o600)
        os.link(self.record,self.root/'linked')
        with self.assertRaisesRegex(Refused,'native_package_file'): self.reopen()
        (self.root/'linked').unlink()
        self.record.rename(self.root/'retained'); self.record.symlink_to(self.root/'retained')
        with self.assertRaises(OSError): self.reopen()
        self.assertEqual((self.root/'retained').read_bytes(),saved)

    def test_mutation_during_hello_refuses_enrollment_and_reopen(self):
        original = (self.root/'key').read_bytes()
        def mutate(q, value):
            if q['op'] == 'hello': self.write('key', b'CHANGED DURING HELLO')
            return value
        self.f.mutate = mutate
        with self.assertRaisesRegex(Refused, 'native_enrollment_changed'): self.enroll()
        self.assertFalse(self.record.exists())
        self.write('key', original); self.f.mutate = lambda q, v: v
        self.enroll(); saved = self.record.read_bytes()
        self.f.mutate = mutate
        with self.assertRaisesRegex(Refused, 'native_enrollment_changed'): self.reopen()
        self.assertEqual(self.record.read_bytes(), saved)

    def test_uncertain_retained_import_refuses_reopen(self):
        self.enroll()
        # Unknown spool evidence closes actual NativeRuntime admission on reopen.
        path = self.f.spool/'unrecognized-retained-evidence'
        path.write_bytes(b'INERT'); path.chmod(0o600)
        with self.assertRaises(Refused): self.reopen()
        self.assertEqual(path.read_bytes(), b'INERT')
