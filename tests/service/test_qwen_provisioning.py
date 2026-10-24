"""Inert managed snapshots; no ML, download, hardware probe or pytest fixtures."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from tests.service import test_provisioning as existing
from service.provisioning import provision, manifest_bytes, manifest_sha256, identity, bundle_manifest
from worker.engines.qwen import verify_bundle, verify_inputs, execute
from worker.engines.qwen_execution import QwenExecution
from worker.hardware import Host, Device, DeviceFamily, HardwareInventory
from worker.transport import Refused
from lib.tts_backends.contract import AudioResult
from lib.tts_backends.assets import verified_bundle


class QwenProvisioningTests(unittest.TestCase):
    cleanup = existing.ProvisioningTests.cleanup
    run_provision = existing.ProvisioningTests.run_provision

    def setUp(self):
        existing.ProvisioningTests.setUp(self)
        self.data = {'config.json': manifest_bytes(dict(model_type='qwen3_tts', tts_model_type='custom_voice', tts_model_size='1b7', talker_config=dict(lang_id=dict(english=1,french=2),spk_id=dict(ryan=0)))),
                     'generation_config.json': b'{}', 'tokenizer_config.json': b'{}', 'tokenizer.json': b'{}',
                     'model.safetensors': b'inert weights', 'speech_tokenizer/config.json': b'{}',
                     'speech_tokenizer/preprocessor_config.json': b'{}', 'speech_tokenizer/model.safetensors': b'inert codec'}
        self.expected.update(backend='qwen', language='English', revision='a'*40,
            allowed_origins=['https://fixture.invalid'],
            versions={p:'fixture' for p in ('qwen-tts','torch','transformers','tokenizers','safetensors')},
            execution=dict(sample_rate=24000,languages=['English','French'],presets=['Ryan'],devices=[dict(device='cpu',dtype='float32',attention='sdpa')]))
        self.repin()
        # Explicit trusted TEST inventory; provisioning never receives one.
        self.inventory = HardwareInventory(Host('linux','x86_64','native'),(Device(DeviceFamily.CPU,0,True,True,'injected-test-only'),))

    def repin(self):
        self.expected['files'] = [dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),origin='https://fixture.invalid/'+n,revision='a'*40) for n,b in self.data.items()]

    def downloader(self,url,target):
        self.calls.append(str(target))
        target.write_bytes(self.data[url.removeprefix('https://fixture.invalid/')])

    def spec(self):
        return QwenExecution('bundle',manifest_sha256(self.expected),self.expected['checkpoint'],'a'*40,'custom_voice',self.expected['language'],'voice','v1',2,100,100,244,4096,24000,preset='Ryan')

    def test_provision_shared_verifier_actual_inputs_and_injected_execution(self):
        for language in ('English','French'):
            self.expected['language'] = language
            bundle = self.run_provision(); e = self.spec()
            self.assertEqual(verify_bundle(bundle,e.manifest_sha256,e.checkpoint,e.revision,e.language)[1],bundle_manifest(self.expected))
            self.assertEqual(verified_bundle(bundle,'qwen')[0],bundle)
            self.assertEqual(verify_inputs(e,{'bundle':bundle},{},self.inventory),(bundle,None))
            backend = Mock(); backend.synthesise_preset.return_value = AudioResult([0.1],24000,metadata=dict(checkpoint=str(bundle),revision='a'*40))
            factory = Mock(return_value=backend)
            result = execute(e,'bonjour',{'bundle':bundle},{},self.inventory,backend_factory=factory)
            self.assertEqual(result['frames'],1)
            self.assertEqual(backend.synthesise_preset.call_args.args[0]['language'],language)
            backend.close.assert_called_once()
            for path, raw in self.data.items(): self.assertEqual((bundle/path).read_bytes(),raw)
            self.assertTrue(all(not p.stat().st_mode & 0o222 for p in bundle.rglob('*')))
            calls = len(self.calls); self.assertEqual(self.run_provision(),bundle); self.assertEqual(len(self.calls),calls)
            with self.assertRaises(Refused): verify_inputs(e,{'bundle':bundle},{},None)
            with self.assertRaises(Refused): verify_inputs(replace(e,language='en'),{'bundle':bundle},{},self.inventory)
            self.assertNotEqual(identity(self.expected),e.manifest_sha256)
            self.assertEqual(hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),e.manifest_sha256)

    def test_resume_failure_and_old_active_preserved(self):
        old = self.run_provision(); pointer = (self.root/'active.json').read_bytes()
        self.expected['checkpoint']='second'
        def interrupted(url,target):
            if target.name == 'tokenizer.json.part': target.write_bytes(b'x'); raise InterruptedError()
            self.downloader(url,target)
        with self.assertRaises(InterruptedError): self.run_provision(downloader=interrupted)
        self.assertEqual((self.root/'active.json').read_bytes(),pointer)
        count = sum(p.endswith('/config.json.part') for p in self.calls)
        self.run_provision()
        self.assertEqual(sum(p.endswith('/config.json.part') for p in self.calls),count+1) # nested config had not yet transferred
        self.assertTrue(old.exists())

    def test_seal_failure_retry(self):
        old = self.run_provision(); pointer = (self.root/'active.json').read_bytes()
        self.expected['checkpoint']='second'
        with patch('service.provisioning._seal',side_effect=OSError('inert seal failure')):
            with self.assertRaises(OSError): self.run_provision()
        self.assertEqual((self.root/'active.json').read_bytes(),pointer)
        self.run_provision(); self.assertTrue(old.exists())

    def test_declarations_paths_versions_and_bounds_before_transfer(self):
        for changes in (dict(language='en'),dict(revision='main'),dict(extra=True),dict(allowed_origins=['http://fixture.invalid']),dict(versions={'torch':'fixture'}),dict(execution={**self.expected['execution'],'devices':[dict(device='cpu',dtype='float16',attention='sdpa')]})):
            with self.subTest(changes=changes), self.assertRaises((ValueError,Refused)): self.run_provision({**self.expected,**changes})
        for field,value in (('path','../escape.json'),('path','model.pkl'),('origin','https://other.invalid/x'),('origin','https://fixture.invalid/x?token=x'),('bytes',8*1024**3+1),('revision','b'*40)):
            bad=copy.deepcopy(self.expected); bad['files'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.run_provision(bad)
        with self.assertRaises(ValueError): provision(self.root,self.expected,runtime_versions={})
        self.assertEqual(self.calls,[])

    def test_hashed_config_refusals_preserve_active(self):
        self.run_provision(); pointer=(self.root/'active.json').read_bytes()
        original=self.data['config.json']
        for config in (dict(model_type='other'),dict(model_type='qwen3_tts',tts_model_type='base',tts_model_size='1b7'),
                       dict(model_type='qwen3_tts',tts_model_type='custom_voice',tts_model_size='1b7',auto_map={}),
                       dict(model_type='qwen3_tts',tts_model_type='custom_voice',tts_model_size='1b7',weights_path='https://else.invalid/x'),
                       dict(model_type='qwen3_tts',tts_model_type='custom_voice',tts_model_size='1b7',sample_rate=16000),
                       dict(model_type='qwen3_tts',tts_model_type='custom_voice',tts_model_size='1b7',talker_config=dict(lang_id={'english':1},spk_id={'other':0}))):
            self.data['config.json']=manifest_bytes(config); self.repin()
            with self.subTest(config=config),self.assertRaises((ValueError,Refused)): self.run_provision()
            self.assertEqual((self.root/'active.json').read_bytes(),pointer)
        self.data['config.json']=original

    def test_bad_transfer_and_runtime_identity_preserve_active(self):
        self.run_provision(); pointer=(self.root/'active.json').read_bytes()
        self.expected['checkpoint']='bad'
        with self.assertRaises(ValueError):
            self.run_provision(downloader=lambda url,target: target.write_bytes(b'bad'))
        self.assertEqual((self.root/'active.json').read_bytes(),pointer)
        with self.assertRaises(ValueError): provision(self.root,self.expected,runtime_versions={})
        self.assertEqual((self.root/'active.json').read_bytes(),pointer)

    def test_policy_identity_and_independent_config_pin(self):
        bundle=self.run_provision(); digest=manifest_sha256(self.expected)
        self.expected['allowed_origins'].append('https://reviewed.invalid')
        other=self.run_provision()
        self.assertNotEqual(bundle,other)
        self.assertEqual(manifest_sha256(self.expected),digest)
        self.assertEqual((other/'config.json').read_bytes(),self.data['config.json'])
        self.expected['checkpoint']='pin-failure'
        self.data['config.json']=b'{}' # expected config pins deliberately unchanged
        pointer=(self.root/'active.json').read_bytes()
        with self.assertRaises(ValueError): self.run_provision()
        self.assertEqual((self.root/'active.json').read_bytes(),pointer)

    def test_origin_redirect_stream_caps(self):
        existing.PocketProvisioningTests.test_pocket_transport_allowed_redirects_and_exact_caps(self)

if __name__ == '__main__': unittest.main()
