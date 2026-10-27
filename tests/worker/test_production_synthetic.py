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


class ReferenceFixtureTests(unittest.TestCase):
    def test_distinct_exact_segment_and_transcript(self):
        import io
        import wave
        from worker.synthetic import fixture_outputs, fixture_manifest
        fixture=json.loads((Path(__file__).resolve().parents[2]/'contracts/fixtures/v1/transcript-voice_prepare-roles.json').read_text())
        message=fixture['message']
        outputs,files=fixture_outputs(message)
        manifest=fixture_manifest(message,outputs)
        candidate=manifest['candidates'][0]
        oracle.reference_roles(message['plan'],candidate,{a['id']:a for a in outputs})
        self.assertNotEqual(outputs[0]['sha256'],outputs[1]['sha256'])
        self.assertEqual(candidate['version']['reference_asset_ids'],[outputs[0]['id']])
        self.assertEqual(candidate['reference_roles']['reference']['original'],message['plan']['assets'][0])
        self.assertEqual(candidate['version']['reference_text'],message['plan']['request']['reference_text'])
        self.assertEqual([name for name,_ in files],['reference.wav','audition.wav'])
        for asset,(_,data) in zip(outputs,files):
            with wave.open(io.BytesIO(data),'rb') as wav:
                self.assertEqual(wav.getnframes(),asset['frames'])
                self.assertEqual(wav.getframerate(),48000)
        message['plan']['request'].update(in_tick=7,out_tick=108)
        outputs,_=fixture_outputs(message)
        self.assertEqual(outputs[0]['frames'],101)
        oracle.reference_roles(message['plan'],fixture_manifest(message,outputs)['candidates'][0],{a['id']:a for a in outputs})

    def test_isolated_worker_writes_only_the_planned_pair(self):
        import copy
        import os
        import tempfile
        import time
        from unittest.mock import patch
        fixture=json.loads((Path(__file__).resolve().parents[2]/'contracts/fixtures/v1/transcript-voice_prepare-roles.json').read_text())
        message=copy.deepcopy(fixture['message'])
        message.update(kind='execute',stage='loading',result_asset_ids=[])
        config=dict(job=fixture['job'],context=fixture['context'],allowed_assets=fixture['authorized_assets'],profiles=fixture['profiles'],secret='a'*64,control_uid=os.getuid(),gid=os.getgid())
        for profile in config['profiles']:
            profile.update(readiness='ready',support='qualified',installation='installed',verification='measured',qualification_receipt='inert synthetic test assertion')
        sent=[]
        with tempfile.TemporaryDirectory() as temporary:
            stage=Path(temporary)/fixture['job']['attempts'][-1]['staging_id'];stage.mkdir()
            with patch.object(isolated_synthetic,'authenticate'),patch.object(isolated_synthetic,'receive',return_value=message),patch.object(isolated_synthetic,'send',side_effect=lambda connection,value,cutoff:sent.append(value)):
                isolated_synthetic.run(None,config,temporary,time.monotonic()+5)
            self.assertEqual(sorted(p.name for p in stage.iterdir()),['audition.wav','reference.wav'])
            manifest=sent[0]['manifest']
            oracle.reference_roles(message['plan'],manifest['candidates'][0],{a['id']:a for a in manifest['outputs']})

if __name__ == '__main__': unittest.main()
