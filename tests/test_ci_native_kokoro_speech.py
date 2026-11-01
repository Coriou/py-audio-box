"""No model downloads: pinned EN identity, closed phases, WAV validation and opt-in."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kokoro_test',ROOT/'scripts/ci-native-kokoro-speech.py')
k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)


class KokoroSpeechTests(unittest.TestCase):
    def test_exact_en_derivation_and_size_origin_hash_closure(self):
        source=json.loads((ROOT/'requirements/tts/kokoro-fr.expected.json').read_bytes())
        en=k.expected();source['language']='en';self.assertEqual(en,source)
        self.assertEqual(sum(f['bytes'] for f in en['files']),353746785)
        for entry in en['files']:
            self.assertTrue(entry['origin'].startswith('https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'))
        with patch.object(k.DESCRIPTOR.__class__,'read_bytes',return_value=b'changed'):
            with self.assertRaisesRegex(ValueError,'descriptor_identity'):k.expected()
        self.assertEqual(k.PRESET,'af_heart');self.assertEqual(k.TEXT,'Hello. The test is complete.')

    def test_redirect_hosts_fail_closed(self):
        for url in ('http://github.com/a','https://github.com.evil/a','https://user@github.com/a','https://github.com:444/a','https://github.com/a#fragment','https://huggingface.co/a'):
            with self.subTest(url=url),self.assertRaises(ValueError):k.public_url(url)
        for url in ('https://github.com/a','https://release-assets.githubusercontent.com/a?public-signed=1'):
            self.assertEqual(k.public_url(url),url)

    def test_owner_phase_caps_do_not_change_qwen_defaults_or_renew(self):
        owner=k.owner_module()
        self.assertEqual(owner.budgets()['generation_seconds'],30)
        self.assertEqual(owner.budgets(True)['generation_seconds'],120)
        limits=owner.budgets(phase_profile='kokoro')
        self.assertEqual((limits['cold_load_seconds'],limits['generation_seconds'],limits['total_seconds']),(60,60,180))
        for stage in ('cold_load','generation'):
            self.assertEqual(owner.phase_remaining(dict(stage=stage,stage_started=10),69,phase_profile='kokoro'),1)
            with self.assertRaises(TimeoutError):owner.phase_remaining(dict(stage=stage,stage_started=10),70,phase_profile='kokoro')
        with self.assertRaises(TimeoutError):owner.phase_remaining(dict(stage='encoding',stage_started=69,generation_seconds=61),70,phase_profile='kokoro')
        with self.assertRaises(ValueError):owner.budgets(phase_profile='arbitrary')

    def test_wav_nonempty_finite_pcm_identity_and_silence_rejection(self):
        def wav(values):
            stream=io.BytesIO()
            with wave.open(stream,'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(struct.pack('<'+'h'*len(values),*values))
            return stream.getvalue()
        raw=wav([0,120,-120,0]);r=k.wav_evidence(raw)
        self.assertEqual(r['sha256'],hashlib.sha256(raw).hexdigest());self.assertEqual(r['frames'],4)
        self.assertEqual(r['duration_seconds'],4/24000);self.assertEqual(r['semantic_ending'],'unverified')
        for raw in (wav([]),wav([0,0]),raw[:-1]):
            with self.assertRaises(ValueError):k.wav_evidence(raw)

    def test_download_exact_size_hash_and_first_failure_no_retry(self):
        from types import SimpleNamespace
        descriptor=k.expected()
        descriptor['files']=[dict(path='model.onnx',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest(),origin=descriptor['files'][0]['origin'])]
        for payload,message in ((b'ab','truncated'),(b'abcd','size_or_hash'),(b'xyz','size_or_hash')):
            with self.subTest(payload=payload),tempfile.TemporaryDirectory() as tmp:
                opener=SimpleNamespace(open=unittest.mock.Mock(return_value=io.BytesIO(payload)))
                with patch.object(k,'expected',return_value=descriptor),patch.object(k.urllib.request,'build_opener',return_value=opener),patch.object(k.shutil,'disk_usage',return_value=SimpleNamespace(free=2*1024**3)):
                    with self.assertRaisesRegex(ValueError,message):k.acquire(Path(tmp))
                opener.open.assert_called_once()
                self.assertFalse((Path(tmp)/'manifest.json').exists())

    def test_actual_engine_encoder_with_injected_backend_and_missing_voice(self):
        # Tiny inert bundle + fake backend: verifies harness/engine forwarding,
        # encoding and failure contracts only, never native inference evidence.
        from types import SimpleNamespace
        import numpy as np
        from lib.tts_backends.contract import AudioResult
        from lib.tts_backends import kokoro
        from service.provisioning import manifest_bytes
        calls=[]; voices={'af_heart'}
        class Fake:
            def __init__(self,root,**kwargs):
                calls.append(kwargs); self._providers=('CPUExecutionProvider',);self._voices=voices
                self.model=SimpleNamespace(create=lambda text,speed=1.0:None,
                    sess=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='style',type='tensor(float)')]))
            def synthesise(self,text,**kwargs):
                calls.append((text,kwargs));return AudioResult(np.array([0,.1,-.1],dtype=np.float32),24000)
            def close(self):calls.append('closed')
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle';bundle.mkdir();output=Path(tmp)/'out';output.mkdir()
            descriptor=k.expected(); descriptor['files']=[]
            for name in ('model.onnx','voices.bin'):
                (bundle/name).write_bytes(b'inert');descriptor['files'].append(dict(path=name,bytes=5,sha256=hashlib.sha256(b'inert').hexdigest()))
            (bundle/'manifest.json').write_bytes(manifest_bytes(descriptor))
            with patch('worker.native_child.install_spawn_guard'),patch.object(k,'expected',return_value=descriptor),patch.object(kokoro,'KokoroBackend',Fake):
                k.infer(bundle,output)
                self.assertEqual(calls,[{'language':'en','threads':2},(k.TEXT,{'preset':'af_heart'}),'closed'])
                receipt=json.loads((output/'English.json').read_text());self.assertTrue(receipt['progress']['backend_closed'])
                self.assertEqual(receipt['execution']['device'],'cpu')
                voices.clear()
                with self.assertRaisesRegex(ValueError,'af_heart_absent'):k.infer(bundle,output)
                self.assertEqual(len([c for c in calls if isinstance(c,tuple)]),1)

    def test_one_total_deadline_not_renewed_between_children_and_late_zero_fails(self):
        from types import SimpleNamespace
        from lib.tts_backends import kokoro_native
        stream=io.BytesIO()
        with wave.open(stream,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(struct.pack('<hh',100,-100))
        audio=stream.getvalue()
        for ending,fail_acquire in ((170,False),(180,False),(170,True)):
            with self.subTest(ending=ending,fail_acquire=fail_acquire),tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/'out'; clock=[0];calls=[]
                def owned(argv,env,log,observation,**kwargs):
                    calls.append(kwargs);observation.update(child_reaped=True,child_exit=0)
                    if argv[argv.index('--child')+1]=='acquire':
                        if fail_acquire:raise RuntimeError('first_download_failure')
                        clock[0]=100
                    else:
                        self.assertEqual(kwargs['phase_profile'],'kokoro');clock[0]=ending
                        (output/'English.wav').write_bytes(audio)
                        k.save(output/'English.json',dict(sha256=hashlib.sha256(audio).hexdigest()))
                with patch.object(kokoro_native,'validate_host'),patch.object(kokoro_native,'validate_versions'),patch.object(k,'owner_module',return_value=SimpleNamespace(owned_child=owned)),patch.object(k.time,'monotonic',side_effect=lambda:clock[0]):
                    if fail_acquire:
                        with self.assertRaisesRegex(RuntimeError,'first_download_failure'):k.run(output)
                    elif ending==180:
                        with self.assertRaisesRegex(TimeoutError,'total_180'):k.run(output)
                    else:k.run(output)
                self.assertEqual(calls[0]['seconds'],120)
                if not fail_acquire:self.assertEqual(calls[1]['seconds'],80)
                self.assertEqual(len(calls),1 if fail_acquire else 2)
                report=json.loads((output/'report.json').read_text())
                self.assertEqual(report['status'],'passed' if ending==170 and not fail_acquire else 'failed')
                self.assertTrue(report['private_model_tree_removed'])

    def test_descriptor_both_archives_and_manual_only_trusted_ref(self):
        from service import native_package
        self.assertIn(str(k.DESCRIPTOR.relative_to(ROOT)),native_package.members())
        self.assertIn(str(k.DESCRIPTOR.relative_to(ROOT)),json.loads((ROOT/'headless-dist-manifest.json').read_bytes())['files'])
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'native.zip';native_package.archive(archive);native_package.verify_archive(archive)
            with zipfile.ZipFile(archive) as z:self.assertEqual(z.read('native-worker/requirements/tts/kokoro-en-macos.expected.json'),k.DESCRIPTOR.read_bytes())
        workflow=(ROOT/'.github/workflows/apple-kokoro.yml').read_text()
        self.assertIn("github.event_name == 'workflow_dispatch' && inputs.en_backend",workflow)
        self.assertIn("github.ref == 'refs/heads/feat/apple-silicon-native'",workflow)
        self.assertIn('default: false',workflow)
        self.assertIn('timeout-minutes: 12',workflow)
        self.assertNotIn('actions/cache',workflow)


if __name__=='__main__':unittest.main()
