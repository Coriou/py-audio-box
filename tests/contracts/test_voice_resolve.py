"""Public voice name read contract and generated request parity."""
import json
from pathlib import Path
import unittest
from jsonschema import Draft202012Validator
from contracts.validate import SCHEMA, fixture_request, validate
from service.catalogue import OPERATIONS, VOICE_READS

class VoiceResolveContractTests(unittest.TestCase):
    def test_schema_and_generated_operation_requests(self):
        Draft202012Validator.check_schema(SCHEMA)
        requests = json.loads(Path('contracts/fixtures/v1/operation-requests.json').read_text())
        self.assertEqual(set(requests),set(OPERATIONS))
        for name, operation in OPERATIONS.items():
            validate(operation['request'],fixture_request(operation['request'],requests[name]))

    def test_resolve_metadata(self):
        self.assertIn('voice_resolve',VOICE_READS)
        operation = OPERATIONS['voice_resolve']
        self.assertEqual(operation['http'],dict(method='GET',route='/v1/voice-names/{name}',input_location='query_and_path'))
        self.assertEqual(operation['cli'],'voice resolve')
        self.assertEqual(operation['mcp'],'speech_voice_resolve')
        self.assertEqual(operation['scope'],'read')
        self.assertEqual(operation['response'],'voice')
        self.assertFalse(operation['idempotency'])
        self.assertIn('voice_archived',operation['errors'])
        self.assertEqual(SCHEMA['$defs']['voice_resolve_request']['properties']['name'],
                         SCHEMA['$defs']['voice_get_request']['properties']['voice_id'])
