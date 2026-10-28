"""Pinned native Base input/lock closure. No weights or model execution."""
import hashlib
import json
from pathlib import Path
import re
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]


class NativeCloneSourceTests(unittest.TestCase):
    def test_base_metadata_preserves_exact_assets_and_matches_native_lock(self):
        original=json.loads((ROOT/'requirements/tts/qwen-0.6b-base.expected.json').read_bytes())
        lock=(ROOT/'requirements/qwen-macos-arm64.lock').read_text()
        pins=dict(re.findall(r'^([a-z0-9-]+)==([^\s]+)',lock,re.M))
        for language,code in (('English','en'),('French','fr')):
            descriptor=json.loads((ROOT/f'requirements/tts/qwen-0.6b-base-macos-{code}.expected.json').read_bytes())
            self.assertEqual(descriptor['files'],original['files'])
            self.assertEqual(descriptor['revision'],original['revision'])
            self.assertEqual(descriptor['checkpoint'],'Qwen/Qwen3-TTS-12Hz-0.6B-Base')
            self.assertEqual(descriptor['language'],language)
            self.assertEqual(descriptor['execution']['devices'],[dict(device='mps',dtype='float32',attention='sdpa')])
            self.assertEqual(descriptor['execution']['presets'],[])
            self.assertEqual(descriptor['license'],'Apache-2.0')
            for name,version in descriptor['versions'].items():
                if name=='qwen-tts':self.assertEqual(version,'0.1.1')
                else:self.assertEqual(version,pins[name])

    def test_pinned_upstream_base_icl_and_prompt_reuse_signatures(self):
        path=ROOT/'requirements/upstream/qwen_tts-0.1.1-py3-none-any.whl'
        # This is the already locked upstream source, not a network fetch.
        with zipfile.ZipFile(path) as source:
            code=source.read('qwen_tts/inference/qwen3_tts_model.py').decode()
        self.assertIn('def create_voice_clone_prompt(',code)
        self.assertIn('self.model.tts_model_type != "base"',code)
        self.assertIn('voice_clone_prompt: Optional[',code)
        self.assertIn('ref_text is required',code)


if __name__=='__main__':unittest.main()
