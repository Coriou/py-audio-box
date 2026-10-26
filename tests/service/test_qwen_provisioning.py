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

    def base_fixture(self):
        self.expected['checkpoint']='Qwen/Qwen3-TTS-12Hz-0.6B-Base'
        self.expected['execution']['presets']=[]
        del self.data['tokenizer.json']
        self.data.update({'vocab.json':b'{"hello":1}', 'merges.txt':b'#version: 0.2\nh e\n',
            'config.json':manifest_bytes(dict(model_type='qwen3_tts',tts_model_type='base',
                tts_model_size='0b6',tokenizer_type='qwen3_tts_tokenizer_12hz',
                architectures=['Qwen3TTSForConditionalGeneration']))})
        self.repin()

    def test_base_descriptor_layout_is_explicit(self):
        from service.provisioning import validate_expected
        self.base_fixture()
        self.assertEqual(validate_expected(self.expected),self.expected)
        for name in ('vocab.json','merges.txt','tokenizer_config.json'):
            bad=copy.deepcopy(self.expected)
            bad['files']=[f for f in bad['files'] if f['path'] != name]
            with self.subTest(name=name),self.assertRaises(ValueError): validate_expected(bad)
        bad=copy.deepcopy(self.expected); bad['checkpoint']='other-model'
        with self.assertRaises(ValueError): validate_expected(bad)

    def test_base_shared_bundle_config_and_hash_fences(self):
        self.base_fixture()
        bundle=self.run_provision()
        digest=manifest_sha256(self.expected)
        self.assertEqual(verify_bundle(bundle,digest,self.expected['checkpoint'],'a'*40,'English')[0],bundle)
        # No whole-weight read is needed by either provisioning or execution.
        with patch.object(Path,'read_bytes',side_effect=AssertionError('whole file read')):
            verify_bundle(bundle,digest,self.expected['checkpoint'],'a'*40,'English')
        path=bundle/'speech_tokenizer/model.safetensors'
        path.chmod(0o600); path.write_bytes(b'wrong codec!')
        with self.assertRaises(Refused): verify_bundle(bundle,digest,self.expected['checkpoint'],'a'*40,'English')

    def test_base_hashed_config_refusals(self):
        self.base_fixture()
        original=json.loads(self.data['config.json'])
        for changes in (dict(tts_model_size=True),dict(tts_model_type='custom_voice'),
                        dict(tokenizer_type='other'),dict(architectures=True),
                        dict(sample_rate=True),dict(auto_map={}),
                        dict(asset_path='https://outside.invalid/model')):
            self.data['config.json']=manifest_bytes({**original,**changes}); self.repin()
            with self.subTest(changes=changes),self.assertRaises((ValueError,Refused)):
                self.run_provision()

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

    def path_configs(self, value):
        for key in ('tokenizer_file', 'asset_path'):
            for wrap in (lambda field: field, lambda field: {'nested': field},
                         lambda field: {'nested': [{'deeper': [field]}]}):
                yield key, wrap({key: value})

    def test_path_field_refusals_preserve_active_pointer_and_all_old_bytes(self):
        old = self.run_provision()
        pointer = (self.root/'active.json').read_bytes()
        old_bytes = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob('*') if p.is_file()}
        for value in ({'location': '/outside/tokenizer.json'}, ['tokenizer.json'], 17, True, False, None, '', ' ',
                      '/outside/tokenizer.json', '../tokenizer.json', 'missing.json', 'https://outside.invalid/tokenizer.json'):
            for key, config in self.path_configs(value):
                with self.subTest(value=value, key=key, config=config):
                    self.data['tokenizer_config.json'] = manifest_bytes(config)
                    self.repin()
                    with self.assertRaises(Refused): self.run_provision()
                    self.assertEqual((self.root/'active.json').read_bytes(), pointer)
                    self.assertEqual({str(p.relative_to(old)): p.read_bytes() for p in old.rglob('*') if p.is_file()}, old_bytes)

    def test_path_field_authorized_strings_preserve_downloaded_bytes_and_pins(self):
        for name, target in (('tokenizer_config.json', 'tokenizer.json'),
                             ('speech_tokenizer/preprocessor_config.json', 'model.safetensors')):
            for key, config in self.path_configs(target):
                with self.subTest(name=name, key=key, config=config):
                    self.data[name] = manifest_bytes(config)
                    self.repin()
                    expected = copy.deepcopy(self.expected)
                    bundle = self.run_provision()
                    self.assertEqual(self.expected, expected)
                    for path, raw in self.data.items(): self.assertEqual((bundle/path).read_bytes(), raw)
                    self.assertEqual(verify_inputs(self.spec(), {'bundle':bundle}, {}, self.inventory), (bundle,None))

    def test_bad_path_fields_refused_by_shared_inputs_before_backend_construction(self):
        # Independently materialize a correctly repinned but malformed local bundle:
        # provisioning must not be the only gate protecting execution.
        bundle = Path(self.temp.name)/'local-input'
        for value in ({'location': '/outside/tokenizer.json'}, ['tokenizer.json'], 17, True, False, None, '', ' '):
            for key, config in self.path_configs(value):
                with self.subTest(value=value, key=key, config=config):
                    self.data['tokenizer_config.json'] = manifest_bytes(config)
                    self.repin()
                    for name, raw in self.data.items():
                        path = bundle/name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(raw)
                    (bundle/'manifest.json').write_bytes(manifest_bytes(bundle_manifest(self.expected)))
                    e = self.spec()
                    with self.assertRaises(Refused): verify_inputs(e, {'bundle':bundle}, {}, self.inventory)
                    factory = Mock()
                    with self.assertRaises(Refused): execute(e, 'bonjour', {'bundle':bundle}, {}, self.inventory, backend_factory=factory)
                    factory.assert_not_called()

if __name__ == '__main__': unittest.main()
