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


# Exact 4494-byte pinned config; no model tensors or ML imports.
PINNED_BASE_CONFIG = b'{\n  "architectures": [\n    "Qwen3TTSForConditionalGeneration"\n  ],\n  "assistant_token_id": 77091,\n  "im_end_token_id": 151645,\n  "im_start_token_id": 151644,\n  "tts_bos_token_id": 151672,\n  "tts_eos_token_id": 151673,\n  "tts_pad_token_id": 151671,\n  "model_type": "qwen3_tts",\n  "tokenizer_type": "qwen3_tts_tokenizer_12hz",\n  "tts_model_size": "0b6",\n  "tts_model_type": "base",\n  "speaker_encoder_config": {\n    "enc_dim": 1024,\n    "sample_rate": 24000\n  },\n  "talker_config": {\n    "attention_bias": false,\n    "attention_dropout": 0,\n    "code_predictor_config": {\n      "_name_or_path": "",\n      "add_cross_attention": false,\n      "architectures": null,\n      "attention_bias": false,\n      "attention_dropout": 0,\n      "bad_words_ids": null,\n      "begin_suppress_tokens": null,\n      "bos_token_id": null,\n      "chunk_size_feed_forward": 0,\n      "cross_attention_hidden_size": null,\n      "decoder_start_token_id": null,\n      "diversity_penalty": 0.0,\n      "do_sample": false,\n      "early_stopping": false,\n      "encoder_no_repeat_ngram_size": 0,\n      "eos_token_id": null,\n      "exponential_decay_length_penalty": null,\n      "finetuning_task": null,\n      "forced_bos_token_id": null,\n      "forced_eos_token_id": null,\n      "head_dim": 128,\n      "hidden_act": "silu",\n      "hidden_size": 1024,\n      "id2label": {\n        "0": "LABEL_0",\n        "1": "LABEL_1"\n      },\n      "initializer_range": 0.02,\n      "intermediate_size": 3072,\n      "is_decoder": false,\n      "is_encoder_decoder": false,\n      "label2id": {\n        "LABEL_0": 0,\n        "LABEL_1": 1\n      },\n      "layer_types": [\n        "full_attention",\n        "full_attention",\n        "full_attention",\n        "full_attention",\n        "full_attention"\n      ],\n      "length_penalty": 1.0,\n      "max_length": 20,\n      "max_position_embeddings": 65536,\n      "max_window_layers": 28,\n      "min_length": 0,\n      "model_type": "qwen3_tts_talker_code_predictor",\n      "no_repeat_ngram_size": 0,\n      "num_attention_heads": 16,\n      "num_beam_groups": 1,\n      "num_beams": 1,\n      "num_code_groups": 16,\n      "num_hidden_layers": 5,\n      "num_key_value_heads": 8,\n      "num_return_sequences": 1,\n      "output_attentions": false,\n      "output_hidden_states": false,\n      "output_scores": false,\n      "pad_token_id": null,\n      "prefix": null,\n      "problem_type": null,\n      "pruned_heads": {},\n      "remove_invalid_values": false,\n      "repetition_penalty": 1.0,\n      "return_dict": true,\n      "return_dict_in_generate": false,\n      "rms_norm_eps": 1e-06,\n      "rope_scaling": null,\n      "rope_theta": 1000000,\n      "sep_token_id": null,\n      "sliding_window": null,\n      "suppress_tokens": null,\n      "task_specific_params": null,\n      "temperature": 1.0,\n      "tf_legacy_loss": false,\n      "tie_encoder_decoder": false,\n      "tie_word_embeddings": false,\n      "tokenizer_class": null,\n      "top_k": 50,\n      "top_p": 1.0,\n      "dtype": null,\n      "torchscript": false,\n      "typical_p": 1.0,\n      "use_bfloat16": false,\n      "use_cache": true,\n      "use_sliding_window": false,\n      "vocab_size": 2048\n    },\n    "codec_bos_id": 2149,\n    "codec_eos_token_id": 2150,\n    "codec_think_id": 2154,\n    "codec_language_id": {\n        "chinese": 2055,\n        "english": 2050,\n        "german": 2053,\n        "italian": 2070,\n        "portuguese": 2071,\n        "spanish": 2054,\n        "japanese": 2058,\n        "korean": 2064,\n        "french": 2061,\n        "russian": 2069\n    },\n    "codec_nothink_id": 2155,\n    "codec_pad_id": 2148,\n    "codec_think_bos_id": 2156,\n    "codec_think_eos_id": 2157,\n    "spk_id": {\n    },\n    "spk_is_dialect": {\n    },\n    "head_dim": 128,\n    "hidden_act": "silu",\n    "hidden_size": 1024,\n    "initializer_range": 0.02,\n    "intermediate_size": 3072,\n    "max_position_embeddings": 32768,\n    "model_type": "qwen3_tts_talker",\n    "num_attention_heads": 16,\n    "num_code_groups": 16,\n    "num_hidden_layers": 28,\n    "num_key_value_heads": 8,\n    "position_id_per_seconds": 13,\n    "rms_norm_eps": 1e-06,\n    "rope_scaling": {\n      "interleaved": true,\n      "mrope_section": [\n        24,\n        20,\n        20\n      ],\n      "rope_type": "default",\n      "type": "default"\n    },\n    "rope_theta": 1000000,\n    "sliding_window": null,\n    "text_hidden_size": 2048,\n    "text_vocab_size": 151936,\n    "use_cache": true,\n    "use_sliding_window": false,\n    "vocab_size": 3072\n  },\n  "transformers_version": "4.57.3"\n}\n'


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

    def test_pinned_config_metadata_and_vocabulary_shared_validator(self):
        self.base_fixture()
        self.assertEqual(hashlib.sha256(PINNED_BASE_CONFIG).hexdigest(),
                         '2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011')
        self.data['config.json'] = PINNED_BASE_CONFIG
        self.data['vocab.json'] = manifest_bytes({'_file':2458, '_path':2638,
                                                'auto_map':5, 'https://token':6})
        self.repin()
        bundle = self.run_provision()
        self.assertEqual((bundle/'config.json').read_bytes(), PINNED_BASE_CONFIG)
        self.assertEqual(verify_bundle(bundle, manifest_sha256(self.expected),
                         self.expected['checkpoint'], 'a'*40, 'English')[0], bundle)
        original = json.loads(PINNED_BASE_CONFIG)
        def metadata_paths(node, path=()):
            found = []
            for key, value in node.items():
                if key == '_name_or_path': found.append(path + (key,))
                if type(value) is dict: found.extend(metadata_paths(value, path + (key,)))
            return found
        self.assertEqual(metadata_paths(original),
                         [('talker_config','code_predictor_config','_name_or_path')])
        for path in ((), ('talker_config',), ('speaker_encoder_config',),
                     ('talker_config','code_predictor_config')):
            for value in ('', None, True, 1, {}, [], 'model.safetensors',
                          'missing.json', '/absolute', '../escape', 'https://remote.invalid/x'):
                config = copy.deepcopy(original)
                node = config
                for key in path: node = node[key]
                node['_name_or_path'] = value
                self.data['config.json'] = manifest_bytes(config); self.repin()
                with self.subTest(path=path, value=value):
                    if value == '' and type(value) is str: self.run_provision()
                    else:
                        with self.assertRaises(Refused): self.run_provision()
        for changes in ({'unknown': {'_name_or_path':''}},
                        {'unknown': [{'_name_or_path':''}]}, {'auto_map':{}},
                        {'trust_remote_code':True}, {'weights_path':''},
                        {'weights_file':'missing.json'}):
            self.data['config.json'] = manifest_bytes({**original, **changes}); self.repin()
            with self.subTest(changes=changes), self.assertRaises(Refused): self.run_provision()
        self.data['config.json'] = PINNED_BASE_CONFIG
        for value in (True, None, {}, [], '', 'model.safetensors', 'https://remote.invalid'):
            self.data['vocab.json'] = manifest_bytes({'_path':value}); self.repin()
            with self.subTest(vocab=value), self.assertRaises(Refused): self.run_provision()

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

    def test_customvoice_bpe_is_checkpoint_and_config_specific(self):
        self.base_fixture()
        self.expected['checkpoint']='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
        self.expected['execution']['presets']=['Ryan']
        config=json.loads(self.data['config.json'])
        config.update(tts_model_type='custom_voice',talker_config={'spk_id':{'ryan':0}})
        self.data['config.json']=manifest_bytes(config);self.repin()
        bundle=self.run_provision()
        verify_bundle(bundle,manifest_sha256(self.expected),self.expected['checkpoint'],'a'*40,'English')
        for changes in ({'tts_model_type':'base'},{'tts_model_size':'1b7'},
                        {'unknown':{'_name_or_path':''}},{'auto_map':{}}):
            self.data['config.json']=manifest_bytes({**config,**changes});self.repin()
            with self.subTest(changes=changes),self.assertRaises((ValueError,Refused)):
                self.run_provision()

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
