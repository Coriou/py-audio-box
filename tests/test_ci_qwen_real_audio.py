"""Inert media validation for the manual gate; no Docker or model operation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave

SOURCE = Path(__file__).resolve().parents[1] / 'scripts' / 'ci-qwen-real-audio.py'
spec = importlib.util.spec_from_file_location('ci_qwen_real_audio', SOURCE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class RealAudioMediaTests(unittest.TestCase):
    def test_real_nonzero_pcm16_has_stable_hash_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'real.wav'
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(24000)
                out.writeframes((500).to_bytes(2, 'little', signed=True) * 24000)
            result = gate.wav(path)
            self.assertEqual(result['duration_seconds'], 1)
            self.assertEqual(result['sha256'], gate.sha(path))
            self.assertGreater(result['rms'], 0)

    def test_silent_media_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'silent.wav'
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(24000)
                out.writeframes(b'\0\0' * 24000)
            with self.assertRaisesRegex(RuntimeError, 'silent_audio'):
                gate.wav(path)
