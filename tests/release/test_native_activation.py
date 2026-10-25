"""First-start SOURCE tests only; inherited fixture helpers use fake TLS."""
import threading
import time
import unittest
from unittest.mock import patch

from service import native_package as n
from service.core import Core
from service.dispatcher import Dispatcher
from service.native_activation import NativeActivation
from tests.release.test_native_enrollment import NativeEnrollmentTests
from worker.transport import Refused


class NativeActivationTests(unittest.TestCase):
    setUp = NativeEnrollmentTests.setUp
    write = NativeEnrollmentTests.write
    enroll = NativeEnrollmentTests.enroll

    def activation(self):
        return NativeActivation(self.record, **self.kw)

    def start(self, activation=None):
        return Dispatcher(self.owner, staging=self.f.stage,
                          installed_activation=activation or self.activation())

    def test_owner_thread_first_start_reconcile_close_and_clean_restart(self):
        self.enroll()
        constructed = []
        original = n.enrolled_runtime
        def capture(*args, **kw):
            result = original(*args, **kw)
            constructed.append((threading.get_ident(), result.owner))
            return result
        with patch.object(n, 'enrolled_runtime', side_effect=capture):
            owner = self.start()
        try:
            self.assertEqual(constructed, [(owner.thread.ident, owner.thread.ident)])
            self.assertNotEqual(owner.thread.ident, threading.get_ident())
            deadline = time.monotonic()+2
            while not owner.call('status')['reconciled']:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            status = owner.call('status')
            self.assertFalse(status['quarantined'])
            self.assertTrue(owner.installed_checked)
            self.assertTrue(owner.enrolled)
        finally:
            owner.close()
        self.assertFalse(owner.thread.is_alive())
        reopened = self.start()
        reopened.close()
        # Refused startup releases the same Core lifetime exclusion.
        core = Core(self.owner); core.close()

    def test_retained_namespace_refuses_before_tls(self):
        self.enroll()
        (self.owner/'unrecognized').write_bytes(b'evidence')
        count = len(self.f.requests)
        with self.assertRaisesRegex(Refused, 'native_retained_history_unsupported'): self.start()
        self.assertEqual(len(self.f.requests), count)
        self.assertFalse((self.owner/'core.sqlite').exists())

    def test_retained_transport_refuses(self):
        self.enroll()
        (self.f.stage/'evidence').write_bytes(b'retained')
        with self.assertRaisesRegex(Refused, 'native_retained_work_unsupported'): self.start()
        self.assertFalse((self.owner/'core.sqlite').exists())

    def test_gap_competitor_initialization_refuses_after_acquisition(self):
        self.enroll()
        activation = self.activation()
        original = activation.precheck
        def race(dispatcher):
            original(dispatcher)
            other = Core(self.owner); other.close()
        activation.precheck = race
        with self.assertRaisesRegex(Refused, 'native_retained_history_unsupported'): self.start(activation)
        core = Core(self.owner); core.close()

    def test_gap_trust_change_refuses_and_closes_core(self):
        self.enroll()
        activation = self.activation()
        original = activation.precheck
        def change(dispatcher):
            original(dispatcher)
            self.write('key', b'CHANGED INERT KEY')
        activation.precheck = change
        with self.assertRaisesRegex(Refused, 'native_enrollment_changed'): self.start(activation)
        core = Core(self.owner); core.close()

    def test_changed_boot_refuses_no_core_and_no_learning(self):
        self.enroll()
        saved = self.record.read_bytes()
        def change(q, response):
            if q['op'] == 'hello': response['boot'] = 'e'*32
            return response
        self.f.mutate = change
        with self.assertRaises(Refused): self.start()
        self.assertEqual(self.record.read_bytes(), saved)
        self.assertFalse((self.owner/'core.sqlite').exists())

    def test_activation_reuse_refuses(self):
        self.enroll()
        activation = self.activation()
        owner = self.start(activation)
        try:
            with self.assertRaisesRegex(Refused, 'native_activation_reuse'): self.start(activation)
            from service.entrypoint import _INSTALLED_LIFECYCLES
            self.assertIs(_INSTALLED_LIFECYCLES[id(activation)]['owner'], owner)
        finally:
            owner.close()
        with self.assertRaisesRegex(Refused, 'native_activation_reuse'): self.start(activation)

    def test_failed_close_retains_owner_obligation_without_retry(self):
        from service.entrypoint import _INSTALLED_LIFECYCLES
        self.enroll()
        activation = self.activation()
        owner = self.start(activation)
        original = Core.close
        calls = []
        def uncertain(core):
            calls.append(core)
            original(core)
            raise OSError('injected close uncertainty')
        with patch.object(Core, 'close', uncertain):
            with self.assertRaisesRegex(RuntimeError, 'control_cleanup_failed'): owner.close()
            with self.assertRaisesRegex(RuntimeError, 'control_cleanup_failed'): owner.close()
        self.assertEqual(len(calls), 1)
        self.assertIs(_INSTALLED_LIFECYCLES[id(activation)]['owner'], owner)
        self.assertFalse(_INSTALLED_LIFECYCLES[id(activation)]['closed'])
        # Test-owned obligation only, with real fixture Core already closed above.
        _INSTALLED_LIFECYCLES.pop(id(activation))
