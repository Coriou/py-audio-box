"""No-weight numerical-gate receipt and resource-scope contracts."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location('numerics',Path(__file__).resolve().parents[1]/'scripts/ci-native-numerics.py')
ci=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ci)


class NumericGateTests(unittest.TestCase):
    def test_tiny_configuration_keeps_codec_structure_but_bounds_parameters(self):
        config=ci.tiny_config()
        self.assertEqual((config['vocab_size'],config['num_code_groups']),(3072,16))
        self.assertEqual((config['hidden_size'],config['num_hidden_layers'],config['text_vocab_size']),(64,1,64))
        self.assertEqual(config['code_predictor_config']['vocab_size'],2048)
        self.assertEqual(sum(config['rope_scaling']['mrope_section'])*2,config['head_dim'])

    def test_receipt_cannot_promote_wrong_device_dtype_or_full_model_claim(self):
        case=dict(status='passed',selected_device='mps:0',sampling_dtype='torch.float32',steps=3,activation_checks=3,
            raw_logit_modules={'Qwen3TTSTalkerForConditionalGeneration':3,'Qwen3TTSTalkerCodePredictorModelForConditionalGeneration':30})
        report=dict(status='passed',full_model_numerics_qualified=False,pretrained_weights=False,speech_decoder_tested=False,
            nonfinite_sentinel_rejected=True,cases=[dict(case,dtype='float32'),dict(case,dtype='float16')])
        ci.validate_report(report)
        for key,value in (('status','failed'),('pretrained_weights',True),('full_model_numerics_qualified',True),('nonfinite_sentinel_rejected',False)):
            bad=copy.deepcopy(report);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ci.validate_report(bad)
        for key,value in (('selected_device','cpu'),('sampling_dtype','torch.float16'),('steps',0),('raw_logit_modules',{})):
            bad=copy.deepcopy(report);bad['cases'][1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ci.validate_report(bad)
