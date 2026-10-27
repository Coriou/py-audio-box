"""Finite native CI inputs and real descriptor construction without ML imports."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import wave

SPEC=importlib.util.spec_from_file_location('apple_speech',Path(__file__).resolve().parents[1]/'scripts/ci-apple-speech.py')
ci=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class AppleSpeechTests(unittest.TestCase):
    def test_download_redirect_policy(self):
        import urllib.request
        handler=ci.PublicRedirects()
        request=urllib.request.Request('https://huggingface.co/model')
        for url in ('http://huggingface.co/a','https://evil.invalid/a',
                    'https://huggingface.co.evil.invalid/a','https://user@huggingface.co/a',
                    'https://huggingface.co:444/a'):
            with self.subTest(url=url),self.assertRaises(ValueError):
                handler.redirect_request(request,None,302,'found',{},url)
        target='https://cas-bridge.xethub.hf.co/object?Signature=public-download'
        self.assertEqual(handler.redirect_request(request,None,302,'found',{},target).full_url,target)

    def test_late_zero_exit_is_failure_and_reaped(self):
        child=Mock(returncode=0);child.poll.return_value=0
        selector=Mock();selector.__enter__=Mock(return_value=selector);selector.__exit__=Mock(return_value=False)
        selector.select.return_value=[(type('Key',(),{'fd':10,'fileobj':child.stdout})(),1)]
        with tempfile.TemporaryDirectory() as tmp,patch.object(ci.subprocess,'Popen',return_value=child),patch.object(ci.selectors,'DefaultSelector',return_value=selector),patch.object(ci.os,'read',return_value=b''),patch.object(ci.time,'monotonic',side_effect=[0,29,29,31,31]):
            result={}
            with self.assertRaisesRegex(TimeoutError,'30_seconds'):
                ci.owned_child(['unused'],{},Path(tmp)/'child.log',result)
            self.assertTrue(result['child_reaped'])
            child.wait.assert_called_once_with(timeout=5)

    def test_rapid_oversized_log_is_bounded_and_child_reaped(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ci,'MAX_LOG',32):
            path=Path(tmp)/'child.log';result={}
            with self.assertRaisesRegex(ValueError,'child_log_limit'):
                ci.owned_child([ci.sys.executable,'-c','print("x"*100000)'],dict(ci.os.environ),path,result)
            self.assertLessEqual(path.stat().st_size,32)
            self.assertTrue(result['child_reaped'])

    def test_descriptor_versions_match_native_lock(self):
        import re
        lock=(ci.ROOT/'requirements/qwen-macos-arm64.lock').read_text()
        pins=dict(re.findall(r'^([a-z0-9-]+)==([^\s]+)',lock,re.M))
        for name,version in json.loads(ci.DESCRIPTOR.read_text())['versions'].items():
            self.assertEqual(pins[name],version,name)

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
        with tempfile.TemporaryDirectory() as tmp,patch.object(ci.shutil,'disk_usage',return_value=type('Disk',(),{'free':20*1024**3})()),patch.object(ci,'public_open',return_value=io.BytesIO(b'bad')):
            with self.assertRaisesRegex(ValueError,'truncated'):ci.acquire(Path(tmp),expected)
