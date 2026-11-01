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
    def test_separate_cold_load_generation_and_total_bounds(self):
        self.assertEqual(ci.phase_remaining({'stage':'cold_load','stage_started':0},60),60)
        self.assertEqual(ci.phase_remaining({'stage':'generation','stage_started':60},75),15)
        for progress,now in (({'stage':'cold_load','stage_started':0},121),
                ({'stage':'generation','stage_started':0},31),
                ({'stage':'encoding','stage_started':31,'generation_seconds':31},31),
                ({'stage':'complete','stage_started':16,'encoding_seconds':16},16),
                ({'stage':'complete','stage_started':181,'total_seconds':181},181)):
            with self.subTest(progress=progress),self.assertRaises(TimeoutError):
                ci.phase_remaining(progress,now)

    def test_precision_selector_is_closed_and_manifest_is_owned_copy(self):
        expected=json.loads(ci.DESCRIPTOR.read_text());original=json.dumps(expected,sort_keys=True)
        for mode,precision in ((False,'float16'),(True,'bfloat16'),(True,True),(True,'auto')):
            with self.subTest(mode=mode,precision=precision),self.assertRaises(ValueError):ci.validate_precision(mode,precision)
        ci.validate_precision(True,'float16');ci.validate_precision(False,'float32')
        selected=ci.diagnostic_manifest(expected,'English','float16')
        self.assertEqual(selected['execution']['devices'],[dict(device='mps',dtype='float16',attention='sdpa')])
        self.assertEqual(json.dumps(expected,sort_keys=True),original)
        self.assertEqual(selected['files'],expected['files'])

    def test_explicit_diagnostic_phase_caps_and_English_only_descriptor(self):
        selected=ci.budgets(True)
        self.assertEqual((selected['generation_seconds'],selected['total_seconds']),(120,270))
        self.assertEqual(ci.phase_remaining(dict(stage='generation',stage_started=0),119,diagnostic_en=True),1)
        with self.assertRaisesRegex(TimeoutError,'native_phase_timeout_generation'):
            ci.phase_remaining(dict(stage='generation',stage_started=0),120,diagnostic_en=True)
        with self.assertRaisesRegex(TimeoutError,'total_seconds'):
            ci.phase_remaining(dict(stage='complete',stage_started=271,total_seconds=271),271,diagnostic_en=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifest.json').write_text('{}')
            buf=io.BytesIO()
            with wave.open(buf,'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\x01\x00'*240)
            from worker import native_child
            with patch.object(native_child,'install_spawn_guard'),patch.object(native_child,'infer',return_value=dict(data=buf.getvalue())) as run:
                ci.infer(root,root,'English',diagnostic_en=True)
                self.assertEqual(run.call_args.args[0].timeout_s,120)
                self.assertEqual(run.call_args.kwargs['diagnostic_request_seconds'],120)
                report=json.loads((root/'English.json').read_text())
                self.assertEqual(report['effective_budgets'],selected)
                with self.assertRaisesRegex(ValueError,'English_only'):ci.infer(root,root,'French',diagnostic_en=True)

    def test_parent_threads_mode_total_and_not_requested_French_without_weights(self):
        expected=json.loads(ci.DESCRIPTOR.read_text())
        with tempfile.TemporaryDirectory() as tmp,patch('platform.system',return_value='Darwin'),patch('platform.machine',return_value='arm64'),patch.object(ci.subprocess,'check_output',return_value=b'7516192768'),patch.object(ci.importlib.metadata,'version',side_effect=expected['versions'].__getitem__),patch.object(ci,'acquire',side_effect=lambda root,expected:(root/'manifest.json').write_text('{}')),patch.object(ci,'owned_child') as child:
            output=Path(tmp)/'out'
            ci.run(output,diagnostic_en=True)
            child.assert_called_once()
            self.assertIn('--diagnostic-en-completeness',child.call_args.args[0])
            self.assertEqual(child.call_args.kwargs['seconds'],270)
            self.assertTrue(child.call_args.kwargs['diagnostic_en'])
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['requested_languages'],['English'])
            self.assertEqual(report['French'],'not_requested')
            self.assertTrue(report['private_model_tree_removed'])

    def test_owned_temp_alias_is_canonical_before_product_no_follow_walk(self):
        from worker.engines.pocket import _open
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);owned=root/'owned';owned.mkdir(mode=0o700)
            alias=root/'alias';alias.symlink_to(owned,target_is_directory=True)
            bundle=ci.new_bundle(alias)
            self.assertEqual(bundle.parent,owned.resolve())
            fd=_open(bundle,True,owner_uid=ci.os.getuid());ci.os.close(fd)
            # Product guards still reject the alias; they were not relaxed.
            with self.assertRaises(OSError):_open(alias/'bundle',True,owner_uid=ci.os.getuid())

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
        self.assertEqual(ci.download_url('https://us.aws.cdn.hf.co/object?Signature=public'),
                         'https://us.aws.cdn.hf.co/object?Signature=public')

    def test_late_zero_exit_is_failure_and_reaped(self):
        child=Mock(returncode=0);child.poll.return_value=0
        selector=Mock();selector.__enter__=Mock(return_value=selector);selector.__exit__=Mock(return_value=False)
        selector.select.return_value=[(type('Key',(),{'fd':10,'fileobj':child.stdout})(),1)]
        sampler=Mock();sampler.stop.return_value=True
        sampler.snapshot.return_value=dict(last=None,peak=None,count=0,error=None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(ci,'Sampler',return_value=sampler),patch.object(ci.subprocess,'Popen',return_value=child),patch.object(ci.selectors,'DefaultSelector',return_value=selector),patch.object(ci.os,'read',return_value=b''),patch.object(ci.time,'monotonic',side_effect=[0,29,29,31,31]):
            result={}
            with self.assertRaisesRegex(TimeoutError,'30_seconds'):
                ci.owned_child(['unused'],{},Path(tmp)/'child.log',result)
            self.assertTrue(result['child_reaped'])
            child.wait.assert_called_once_with(timeout=5)

    def test_diagnostic_owner_total_270_rejects_late_zero_exit(self):
        child=Mock();child.pid=999999;child.stdout=Mock();child.returncode=0;child.poll.return_value=0
        selector=Mock();selector.__enter__=Mock(return_value=selector);selector.__exit__=Mock(return_value=False)
        selector.select.return_value=[(type('Key',(),{'fd':1,'fileobj':child.stdout})(),1)]
        sampler=Mock();sampler.stop.return_value=True
        sampler.snapshot.return_value=dict(last=None,peak=None,count=0,error=None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(ci,'Sampler',return_value=sampler),patch.object(ci.subprocess,'Popen',return_value=child),patch.object(ci.selectors,'DefaultSelector',return_value=selector),patch.object(ci.os,'read',return_value=b''),patch.object(ci.time,'monotonic',side_effect=[0,269,269,271,271]):
            report={}
            with self.assertRaisesRegex(TimeoutError,'270_seconds'):
                ci.owned_child(['unused'],{},Path(tmp)/'child.log',report,seconds=270,diagnostic_en=True)
            self.assertEqual(report['owner_total_seconds'],270)
            self.assertEqual(report['stage_exceeded'],'owner_total')
            self.assertTrue(report['child_reaped'])

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
        from packaging.utils import parse_wheel_filename
        wheel=next((ci.ROOT/'requirements/wheels').glob('qwen_tts-*.whl'))
        name,version,_,_=parse_wheel_filename(wheel.name)
        self.assertIn('./requirements/wheels/'+wheel.name,lock)
        self.assertIn(ci.hashlib.sha256(wheel.read_bytes()).hexdigest(),lock)
        pins[str(name)]=str(version)
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
                        self.assertEqual(report['rtf_scope'],'generation')
                        self.assertIn('cold_total_rtf',report)
                    else:
                        with self.assertRaisesRegex(ValueError,'silent_wav'):ci.infer(root,root,'English')

    def test_acquisition_rejects_corrupt_public_file(self):
        expected=json.loads(ci.DESCRIPTOR.read_text())
        with tempfile.TemporaryDirectory() as tmp,patch.object(ci.shutil,'disk_usage',return_value=type('Disk',(),{'free':20*1024**3})()),patch.object(ci,'public_open',return_value=io.BytesIO(b'bad')):
            with self.assertRaisesRegex(ValueError,'truncated'):ci.acquire(Path(tmp),expected)
