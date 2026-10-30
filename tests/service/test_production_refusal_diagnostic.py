"""The internal refusal log exposes bounded codes without exception contents."""
import unittest
from unittest.mock import patch

from service.catalogue import Failure
from service.domain import production_call
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
