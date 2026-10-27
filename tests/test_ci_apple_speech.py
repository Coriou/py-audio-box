"""Finite native CI inputs and real descriptor construction without ML imports."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

SPEC=importlib.util.spec_from_file_location('apple_speech',Path(__file__).resolve().parents[1]/'scripts/ci-apple-speech.py')
ci=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class AppleSpeechTests(unittest.TestCase):
    def test_native_inference_uses_typed_closed_binding_and_validates_audio(self):
        from worker import native_child
        from worker.engines.qwen_execution import QwenExecution
        for valid in (True,False):
            with self.subTest(valid=valid),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'manifest.json').write_text('{}')
                buf=io.BytesIO()
                with wave.open(buf,'wb') as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000)
                    wav.writeframes((b'\x01\x00' if valid else b'\x00\x00')*240)
                with patch.object(native_child,'install_spawn_guard'),patch.object(native_child,'infer',return_value={'data':buf.getvalue()}) as run:
                    if valid:
                        ci.infer(root,root,'English')
                        spec=run.call_args.args[0]
                        self.assertIs(type(spec),QwenExecution)
                        self.assertEqual((spec.device,spec.dtype,spec.timeout_s),('mps','float32',30))
                        report=json.loads((root/'English.json').read_text())
                        self.assertFalse(report['installed_qualified'])
                        self.assertFalse(report['ending_verified'])
                    else:
                        with self.assertRaisesRegex(ValueError,'silent_wav'):ci.infer(root,root,'English')

    def test_acquisition_rejects_corrupt_public_file(self):
        expected=json.loads(ci.DESCRIPTOR.read_text())
        with tempfile.TemporaryDirectory() as tmp,patch.object(ci.shutil,'disk_usage',return_value=type('Disk',(),{'free':20*1024**3})()),patch.object(ci.urllib.request,'urlopen',return_value=io.BytesIO(b'bad')):
            with self.assertRaisesRegex(ValueError,'truncated'):ci.acquire(Path(tmp),expected)
