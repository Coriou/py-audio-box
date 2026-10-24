"""Pure proposal tests; never invokes a Linux sandbox runtime."""
import json
from pathlib import Path
import unittest
from contracts import validate as oracle
from worker.synthetic import candidates
from worker import isolated_synthetic

class CandidateTests(unittest.TestCase):
    def test_exact_prepare_metadata(self):
        fixture = json.loads((Path(__file__).resolve().parents[2]/'contracts/fixtures/v1/transcript-voice_prepare.json').read_text())
        message = fixture['message']; asset = fixture['verified_manifest']['outputs'][0]
        result = candidates(message,asset)[0]
        oracle.validate('voice_candidate',result)
        self.assertEqual(result['version']['configuration'], message['plan']['resolved']['reuse_configuration'])
        self.assertEqual(result['version']['settings'], message['plan']['resolved']['reuse_settings'])
        self.assertEqual(result['audition_asset_id'],asset['id'])
        result['version']['settings']['seed'] = 9
        self.assertEqual(message['plan']['resolved']['reuse_settings']['seed'],1)
        self.assertTrue(callable(isolated_synthetic.run))

if __name__ == '__main__': unittest.main()
