"""Owning-thread real Core/files/SQLite; unittest only, no conftests."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
from contracts import validate as oracle
from service.core import Core
from service.ingest import ingest_local
from service.production import Production

FIXTURES = Path(__file__).resolve().parents[2]/'contracts/fixtures/v1'

class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.core = Core(Path(self.temp.name)/'data', historical_local=True)
        self.domain = Production(self.core)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(48000)
            wav.writeframes(b'\0'*9600)
        self.asset = ingest_local(self.core, 'p', buffer.getvalue())
        self.context = json.loads((FIXTURES/'transcript-voice_prepare.json').read_text())['context']
        self.context['plan']['request']['asset_id'] = self.asset['id']
        self.context['plan']['resolved']['reuse_configuration']['reference_asset_id'] = 'audition-owned'

    def tearDown(self):
        if self.core.active: self.core.finish(cancel=True)
        self.core.close(); self.temp.cleanup()

    def prepared(self):
        job = self.domain.prepare('p', 'prepare', self.context)
        self.core.start(); result = self.core.finish()
        self.assertEqual(result['state'], 'succeeded')
        return job

    def test_lifecycle_restart_and_frozen_generation(self):
        job = self.prepared()
        candidate = self.domain.results('p', job['id'])[0]
        saved = self.domain.save('p','save',job['id'],candidate['id'],'voice',expected_revision=0)
        self.assertEqual(saved, self.domain.save('p','save',job['id'],candidate['id'],'voice',expected_revision=0))
        root = self.core.root; self.core.close()
        self.core = Core(root,historical_local=True); self.domain = Production(self.core)
        self.assertEqual(self.domain.prepare('p','prepare',self.context)['id'],job['id'])
        context = copy.deepcopy(self.context); plan = context['plan']; r = plan['resolved']; version = saved['version']
        plan['operation'] = 'speech_generate'
        template = json.loads((FIXTURES/'transcript-speech_generate.json').read_text())['context']['plan']['request']
        plan['request'] = dict(template,voice_version_id=version['id'],text=' Exact\n text ',identity=version['identity'])
        r.update(kind='speech',settings=version['settings'],settings_sha256=version['settings_sha256'],reuse_configuration=None,reuse_settings=None)
        r['profile']['compatibility'] = [dict(operation='speech_generate',mode='qwen_clone',identity=version['identity'])]
        generated = self.domain.generate('p','generate',context)
        context['plan']['request']['text'] = 'mutated'
        self.assertEqual(self.core._load(generated['id'])[1]['plan']['request']['text'],' Exact\n text ')
        self.core.start(); self.assertEqual(self.core.finish()['state'],'succeeded')

    def test_preset_legacy_and_invalid_settings(self):
        context = json.loads((FIXTURES/'transcript-speech_generate.json').read_text())['context']
        self.core.submit_synthetic('p','legacy',context)
        self.core.start(); self.assertEqual(self.core.finish()['state'],'succeeded')
        changed = copy.deepcopy(self.context)
        changed['plan']['resolved']['settings_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'settings_mismatch'):
            self.domain.prepare('p','invalid',changed)
        with self.assertRaisesRegex(ValueError,'forbidden'):
            self.domain.prepare('other','foreign',self.context)

    def test_stale_publication_and_candidate_contract(self):
        job = self.prepared()
        candidates = self.domain.results('p',job['id'])
        oracle.validate('voice_candidate', candidates[0])
        raw = self.core.db.execute('SELECT artifact FROM publications WHERE job_id=?',(job['id'],)).fetchone()[0]
        artifact = json.loads(raw); artifact['manifest_sha256'] = '0'*64
        with self.core.transaction():
            self.core.db.execute('UPDATE publications SET artifact=? WHERE job_id=?',(json.dumps(artifact),job['id']))
        with self.assertRaisesRegex(ValueError,'stale_candidate'):
            self.domain.results('p',job['id'])

    def test_selection_ownership_revision_and_missing_media(self):
        queued = self.domain.prepare('p','prepare',self.context)
        with self.assertRaisesRegex(ValueError,'unpublished'): self.domain.results('p',queued['id'])
        self.core.start(); self.core.finish()
        candidate = self.domain.results('p',queued['id'])[0]
        with self.assertRaisesRegex(ValueError,'forbidden'): self.domain.results('other',queued['id'])
        with self.assertRaisesRegex(ValueError,'candidate_missing'): self.domain.save('p','s',queued['id'],'missing','voice',expected_revision=0)
        self.domain.save('p','s',queued['id'],candidate['id'],'voice',expected_revision=0)
        with self.assertRaisesRegex(ValueError,'revision_conflict'): self.domain.save('p','s2',queued['id'],candidate['id'],'voice2',expected_revision=1)
        changed = copy.deepcopy(self.context); changed['plan']['request']['reference_text'] = 'other'
        with self.assertRaisesRegex(ValueError,'idempotency_conflict'): self.domain.prepare('p','prepare',changed)
        (self.core.root/'assets'/candidate['audition_asset_id']).unlink()
        with self.assertRaises((ValueError,OSError)): self.domain.results('p',queued['id'])

if __name__ == '__main__': unittest.main()
