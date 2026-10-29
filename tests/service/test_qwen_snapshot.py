"""Inert local bytes through real provisioning; no network or model runtime."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import unittest
import urllib.request
import fcntl
import multiprocessing
import time
from unittest.mock import patch

from service import installed_qwen_daemon as qd
from service.provisioning import manifest_bytes, manifest_sha256, _AllowedRedirects, QWEN_HF_ORIGINS, download
from service.qwen_snapshot import repair
from worker.engines.qwen import verify_bundle


class QwenSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp' if sys.platform == 'darwin' else '/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / 'snapshot'
        self.snapshot.mkdir()
        self.data = {
            'config.json': manifest_bytes(dict(model_type='qwen3_tts', tts_model_type='base',
                tts_model_size='0b6', tokenizer_type='qwen3_tts_tokenizer_12hz',
                architectures=['Qwen3TTSForConditionalGeneration'])),
            'generation_config.json': b'{}', 'tokenizer_config.json': b'{}',
            'vocab.json': b'{"hello":1}', 'merges.txt': b'#version: 0.2\nh e\n',
            'model.safetensors': b'inert weights',
            'speech_tokenizer/config.json': b'{}',
            'speech_tokenizer/preprocessor_config.json': b'{}',
            'speech_tokenizer/model.safetensors': b'inert codec',
        }
        self.expected = dict(schema_version=1, backend='qwen', language='English',
            checkpoint=qd.CHECKPOINT, revision='a'*40, license='fixture', access='fixture',
            versions={k:'fixture' for k in ('qwen-tts','torch','transformers','tokenizers','safetensors')},
            allowed_origins=['https://fixture.invalid'],
            execution=dict(sample_rate=24000,languages=['English','French'],presets=[],
                devices=[dict(device='cpu',dtype='float32',attention='sdpa')]),
            files=[dict(path=n,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                        origin='https://fixture.invalid/'+n,revision='a'*40)
                   for n,raw in self.data.items()])

    def test_reviewed_cpu_descriptors_and_exact_service_selection(self):
        root = Path(__file__).resolve().parents[2] / 'requirements' / 'tts'
        for name, filename in [('clone','qwen-0.6b-base.expected.json'),
                               ('Ryan','qwen-0.6b-customvoice-cpu.expected.json')]:
            value = json.loads((root / filename).read_text())
            self.assertEqual(qd.service_descriptor(value,name),value)
            bad = json.loads(json.dumps(value)); bad['files'][0]['sha256']='0'*64
            with self.assertRaises(ValueError): qd.service_descriptor(bad,name)
        with self.assertRaises(ValueError): qd.service_model('Bruce')
        self.assertEqual(qd.resident_transition(None,'clone',0),'load')
        self.assertEqual(qd.resident_transition('clone','clone',1),'reuse')
        self.assertEqual(qd.resident_transition('clone','Ryan',0),'unload_then_load')
        with self.assertRaisesRegex(ValueError,'qwen_service_model_busy'):
            qd.resident_transition('clone','Ryan',1)

    def test_partial_offline_refuses_then_reuses_and_repairs_before_activation(self):
        # One valid local byte, one corrupt local byte and the remaining absent.
        (self.snapshot/'config.json').write_bytes(self.data['config.json'])
        (self.snapshot/'generation_config.json').write_bytes(b'bad')
        with patch.object(qd,'service_descriptor',return_value=self.expected):
            with self.assertRaisesRegex(FileNotFoundError,'qwen_snapshot_missing_or_corrupt_offline:generation_config.json'):
                repair(self.root/'managed',self.expected,self.snapshot,offline=True)
            self.assertFalse((self.root/'managed'/'active.json').exists())
            calls=[]
            def fetch(url,target):
                name=url.removeprefix('https://fixture.invalid/')
                calls.append(name)
                target.write_bytes(self.data[name])
            bundle=repair(self.root/'managed',self.expected,self.snapshot,downloader=fetch)
        self.assertEqual((bundle/'config.json').read_bytes(),self.data['config.json'])
        self.assertNotIn('config.json',calls)
        self.assertIn('generation_config.json',calls)
        self.assertEqual(verify_bundle(bundle,manifest_sha256(self.expected),qd.CHECKPOINT,'a'*40,'English')[0],bundle)
        self.assertTrue((self.root/'managed'/'active.json').exists())
        with patch.object(qd,'service_descriptor',return_value=self.expected):
            with patch.dict(qd.SERVICE_MODELS,{'Ryan':(qd.CHECKPOINT,'a'*40,manifest_sha256(self.expected)),
                                               'clone':(qd.CHECKPOINT,'a'*40,manifest_sha256(self.expected))}):
                status=qd.verified_service_models({'clone':self.expected,'Ryan':self.expected},
                                                   {'clone':bundle,'Ryan':self.root/'missing'})
        self.assertEqual(status['models']['clone'],{'source_verified':True})
        self.assertFalse(status['models']['Ryan']['source_verified'])
        self.assertFalse(status['runtime_qualified'])

    def test_one_retry_and_outside_cache_refusal(self):
        outside=self.root/'outside'; outside.write_bytes(self.data['config.json'])
        (self.snapshot/'config.json').symlink_to(outside)
        with patch.object(qd,'service_descriptor',return_value=self.expected):
            with self.assertRaisesRegex(ValueError,'qwen_snapshot_source_outside_cache'):
                repair(self.root/'managed',self.expected,self.snapshot,offline=True)
        (self.snapshot/'config.json').unlink()
        attempts=[]
        def fetch(url,target):
            name=url.removeprefix('https://fixture.invalid/')
            attempts.append(name)
            if name=='config.json' and attempts.count(name)==1:
                target.write_bytes(b'partial')
                raise ConnectionError('transport reset')
            else: target.write_bytes(self.data[name])
        with patch.object(qd,'service_descriptor',return_value=self.expected):
            repair(self.root/'managed',self.expected,self.snapshot,downloader=fetch)
        self.assertEqual(attempts.count('config.json'),2)

    def test_hf_blob_link_within_explicit_cache_is_reused(self):
        cache=self.root/'cache'; blobs=cache/'blobs'; snapshot=cache/'snapshots'/'revision'
        blobs.mkdir(parents=True); snapshot.mkdir(parents=True)
        (blobs/'config').write_bytes(self.data['config.json'])
        (snapshot/'config.json').symlink_to('../../blobs/config')
        calls=[]
        def fetch(url,target):
            name=url.removeprefix('https://fixture.invalid/')
            calls.append(name); target.write_bytes(self.data[name])
        with patch.object(qd,'service_descriptor',return_value=self.expected):
            bundle=repair(self.root/'managed',self.expected,snapshot,cache_root=cache,downloader=fetch)
        self.assertNotIn('config.json',calls)
        self.assertEqual((bundle/'config.json').read_bytes(),self.data['config.json'])

    def test_signed_hf_redirect_chain_is_closed_to_reviewed_destination(self):
        origins=sorted(QWEN_HF_ORIGINS)
        handler=_AllowedRedirects(origins,qwen_hf_assets=True)
        request=urllib.request.Request('https://huggingface.co/Qwen/model/resolve/revision/model.safetensors')
        signed='https://us.aws.cdn.hf.co/opaque/path?Expires=123&Signature=inert&Key-Pair-Id=key'
        redirected=handler.redirect_request(request,None,302,'Found',{},signed)
        self.assertEqual(redirected.full_url,signed)
        self.assertEqual(handler.redirect_request(redirected,None,302,'Found',{},signed).full_url,signed)
        for bad in ('http://us.aws.cdn.hf.co/path?Signature=x',
                    'https://other.invalid/path?Signature=x',
                    'https://user:secret@us.aws.cdn.hf.co/path?Signature=x',
                    'https://huggingface.co/path?Signature=x',
                    'https://us.aws.cdn.hf.co.evil.invalid/path?Signature=x',
                    'https://us.aws.cdn.hf.co/path?Signature=x#fragment'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                handler.redirect_request(request,None,302,'Found',{},bad)
        with self.assertRaises(ValueError):
            _AllowedRedirects(origins).redirect_request(request,None,302,'Found',{},signed)
        credentialed=urllib.request.Request(request.full_url,headers={'Authorization':'Bearer private'})
        with self.assertRaises(ValueError):
            handler.redirect_request(credentialed,None,302,'Found',{},signed)
        with patch('service.provisioning.multiprocessing.get_context') as process:
            with self.assertRaises(ValueError):
                download(request.full_url,self.root/'never',10,allowed_origins=origins+['https://evil.invalid'],qwen_hf_assets=True)
            process.assert_not_called()

    def test_corrupt_download_is_not_retried(self):
        attempts=[]
        def corrupt(url,target):
            attempts.append(url)
            target.write_bytes(b'corrupt')
        with patch.object(qd,'service_descriptor',return_value=self.expected):
            with self.assertRaisesRegex(ValueError,'qwen_snapshot_download_pin_mismatch:config.json'):
                repair(self.root/'managed',self.expected,self.snapshot,downloader=corrupt)
        self.assertEqual(len(attempts),1)
        self.assertFalse((self.root/'managed'/'active.json').exists())

    def test_repair_deadline_bounds_real_lock_wait(self):
        managed=self.root/'managed'
        managed.mkdir(mode=0o700)
        lock=managed/'.provision.lock'
        lock.touch(mode=0o600)
        ready=multiprocessing.get_context('fork').Pipe(duplex=False)
        def hold(path,send):
            with open(path,'w') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX)
                send.send(True)
                time.sleep(.3)
        process=multiprocessing.get_context('fork').Process(target=hold,args=(lock,ready[1]))
        process.start()
        try:
            self.assertTrue(ready[0].recv())
            start=time.monotonic()
            with patch.object(qd,'service_descriptor',return_value=self.expected):
                with self.assertRaises(TimeoutError):
                    repair(managed,self.expected,self.snapshot,seconds=.02,offline=True)
            self.assertLess(time.monotonic()-start,.2)
            self.assertFalse((managed/'active.json').exists())
        finally:
            process.join(1)
            if process.is_alive(): process.terminate(); process.join()

    def test_deadline_after_full_verify_never_activates(self):
        from service import provisioning as p
        managed=self.root/'managed'
        tick=[100.0]
        real_verify=p._verified_managed
        def late_verify(*args):
            result=real_verify(*args)
            tick[0]=102.0
            return result
        def fetch(url,target):
            target.write_bytes(self.data[url.removeprefix('https://fixture.invalid/')])
        with patch.object(qd,'service_descriptor',return_value=self.expected), \
                patch('service.qwen_snapshot.time.monotonic',side_effect=lambda:tick[0]), \
                patch.object(p,'_verified_managed',side_effect=late_verify):
            with self.assertRaises(TimeoutError):
                repair(managed,self.expected,self.snapshot,downloader=fetch,seconds=1)
        self.assertFalse((managed/'active.json').exists())

    def test_deadline_after_sealing_never_activates(self):
        from service import provisioning as p
        managed=self.root/'managed'
        tick=[100.0]
        real_seal=p._seal
        def late_seal(*args):
            real_seal(*args)
            tick[0]=102.0
        def fetch(url,target):
            target.write_bytes(self.data[url.removeprefix('https://fixture.invalid/')])
        with patch.object(qd,'service_descriptor',return_value=self.expected), \
                patch('service.qwen_snapshot.time.monotonic',side_effect=lambda:tick[0]), \
                patch.object(p,'_seal',side_effect=late_seal):
            with self.assertRaises(TimeoutError):
                repair(managed,self.expected,self.snapshot,downloader=fetch,seconds=1)
        self.assertFalse((managed/'active.json').exists())


if __name__=='__main__': unittest.main()
