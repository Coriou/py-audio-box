"""Pure admission checks; these do not qualify Pocket or invoke Docker/ML."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('cpu_pocket_recipe', ROOT/'scripts/cpu-pocket.py')
RECIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECIPE)


def pcm(rate=48000, frames=480):
    raw=io.BytesIO()
    with wave.open(raw,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(b'\0\0'*frames)
    return raw.getvalue()


class PocketRecipeAdmission(unittest.TestCase):
    def test_failure_diagnostic_retains_location_without_private_values(self):
        def private_failure():
            credential = 'private-test-credential'
            raise ValueError(credential)
        try:
            private_failure()
        except ValueError as exc:
            diagnostic = RECIPE.failure_diagnostic(exc)
        self.assertRegex(diagnostic, r'^ValueError:private_failure:[0-9]+$')
        self.assertNotIn('private-test-credential', diagnostic)
        self.assertEqual(RECIPE.failure_diagnostic(KeyboardInterrupt()), 'KeyboardInterrupt')

    def test_reference_ticks_use_48k_clock_at_both_allowed_rates(self):
        expected=json.loads((ROOT/'requirements/tts/pocket-fr.expected.json').read_bytes())
        for rate in (24000,48000):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'voice.wav';path.write_bytes(pcm(rate,rate*8))
                raw,ref=RECIPE.reference(path)
                ctx,profile=RECIPE.context(expected,ref)
                self.assertEqual(ctx['execution']['descriptor']['out_tick'],384000)
                self.assertEqual(ctx['execution']['descriptor']['reference_sha256'],hashlib.sha256(raw).hexdigest())
                self.assertEqual(profile['verification'],'synthetic')
                self.assertEqual(profile['support'],'unqualified')
                self.assertEqual(profile['qualification_receipt'],'')
                self.assertEqual(ctx['plan']['request']['audition_text'],'Bonjour.')
                self.assertEqual(profile['limits']['disk_bytes'],8*1024**2)
                self.assertGreaterEqual(profile['limits']['disk_bytes'],2*ctx['execution']['descriptor']['max_output_bytes']+6*profile['limits']['upload_bytes'])
                self.assertEqual(RECIPE.POCKET_INSTALL_DISK_BYTES,1024**3)

    def test_reference_rejects_duration_over_eight_seconds_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'voice.wav';path.write_bytes(pcm(24000,24000*8+1))
            with self.assertRaises(ValueError):RECIPE.reference(path)
            path.write_bytes(pcm())
            link=Path(tmp)/'link.wav';link.symlink_to(path)
            with self.assertRaises(OSError):RECIPE.reference(link)

    def test_hash_copy_refuses_mismatched_or_mutable_namespace_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.write_bytes(b'abc')
            with self.assertRaises(ValueError):RECIPE.bounded_copy(source,Path(tmp)/'wrong',3,'0'*64)
            link=Path(tmp)/'link';link.symlink_to(source)
            with self.assertRaises(OSError):RECIPE.bounded_copy(link,Path(tmp)/'linked',3,hashlib.sha256(b'abc').hexdigest())
