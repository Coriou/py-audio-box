"""The internal refusal log exposes bounded codes without exception contents."""
import unittest
from unittest.mock import patch

from service.catalogue import Failure
from service.domain import SyntheticContexts, production_call
from worker.transport import Refused


class RefusalDiagnosticTests(unittest.TestCase):
    def test_known_code_and_untrusted_exception_text(self):
        class Dispatcher:
            def __init__(self, error): self.error = error
            def call(self, *args, **kwargs): raise Refused(self.error)

        for reason, expected in (('hardware_identity', 'hardware_identity'),
                                 ('execution_reference_mismatch', 'execution_reference_mismatch'),
                                 ('preparation_reference_limit', 'preparation_reference_limit'),
                                 ('explicit_transcript_required', 'explicit_transcript_required'),
                                 ('resident_cpu_execution_required', 'resident_cpu_execution_required'),
                                 ('resident_model_identity', 'resident_model_identity'),
                                 ('reference_identity', 'reference_identity'),
                                 ('private/path?credential=secret', 'unclassified')):
            with self.subTest(reason=expected), patch('service.domain.validate'), \
                    self.assertLogs('service.domain', level='WARNING') as logs:
                with self.assertRaises(Failure) as caught:
                    production_call(Dispatcher(reason), 'alice', 'voice_prepare', {}, 'key')
            self.assertEqual(caught.exception.body['code'], 'not_ready')
            self.assertIn('reason=' + expected, logs.output[0])
            self.assertNotIn('secret', logs.output[0])
            self.assertNotIn('private/path', logs.output[0])
            self.assertIn('type=ValueError source=unknown line=0', logs.output[0])

    def test_nested_exceptions_are_bounded_and_public_error_is_stable(self):
        secret = '/private/model/path?credential=never-log-me'
        cases = ((ValueError(secret), 'ValueError', 'unclassified'),
                 (KeyError(secret), 'KeyError', 'unclassified'),
                 (KeyError('hardware_identity'), 'KeyError', 'hardware_identity'),
                 (NotImplementedError(), 'NotImplementedError', 'unclassified'))

        for error, kind, reason in cases:
            class Dispatcher:
                def call(self, *args, **kwargs):
                    def nested(): raise error
                    nested()

            with self.subTest(kind=kind, reason=reason), patch('service.domain.validate'), \
                    self.assertLogs('service.domain', level='WARNING') as logs:
                with self.assertRaises(Failure) as caught:
                    production_call(Dispatcher(), 'alice', 'voice_prepare', {}, 'key')
            log = logs.output[0]
            self.assertIn('reason=' + reason + ' type=' + kind + ' source=unknown line=0', log)
            self.assertLess(len(log), 220)
            self.assertNotIn(secret, log)
            self.assertEqual(caught.exception.body['code'], 'not_ready')
            self.assertNotIn(secret, str(caught.exception.body))

    def test_repository_source_location_is_reported(self):
        class Dispatcher:
            def call(self, *args, **kwargs):
                def nested(): SyntheticContexts({('alice', 'voice_prepare'): {}})
                nested()

        with patch('service.domain.validate'), self.assertLogs('service.domain', level='WARNING') as logs:
            with self.assertRaises(Failure):
                production_call(Dispatcher(), 'alice', 'voice_prepare', {}, 'key')
        self.assertRegex(logs.output[0], r'reason=unclassified type=KeyError source=service/domain\.py line=[1-9][0-9]*$')

    def test_quoted_key_keeps_public_mapping(self):
        for key, public in (('candidate_missing', 'not_found'),
                            ('invalid_request', 'invalid_request'),
                            ('queue_full', 'resource_limit')):
            class Dispatcher:
                def call(self, *args, **kwargs): raise KeyError(key)

            with self.subTest(key=key), patch('service.domain.validate'), \
                    self.assertLogs('service.domain', level='WARNING') as logs:
                with self.assertRaises(Failure) as caught:
                    production_call(Dispatcher(), 'alice', 'voice_prepare', {}, 'key')
            self.assertEqual(caught.exception.body['code'], 'not_ready')
            self.assertIn('reason=' + key + ' type=KeyError', logs.output[0])

            class ValueDispatcher:
                def call(self, *args, **kwargs): raise ValueError(key)

            with patch('service.domain.validate'), self.assertLogs('service.domain', level='WARNING'):
                with self.assertRaises(Failure) as value_caught:
                    production_call(ValueDispatcher(), 'alice', 'voice_prepare', {}, 'key')
            self.assertEqual(value_caught.exception.body['code'], public)

    def test_worker_origin_does_not_report_service_caller(self):
        from service.qwen_local_lane import context_allowed

        class Dispatcher:
            def call(self, *args, **kwargs): return context_allowed({}, {})

        with patch('service.domain.validate'), self.assertLogs('service.domain', level='WARNING') as logs:
            with self.assertRaises(Failure) as caught:
                production_call(Dispatcher(), 'alice', 'voice_prepare', {}, 'key')
        self.assertEqual(caught.exception.body['code'], 'not_ready')
        self.assertRegex(logs.output[0], r'type=KeyError source=worker/runtime_protocol\.py line=[1-9][0-9]*$')
