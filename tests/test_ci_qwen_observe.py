"""No weights: observation must preserve calls, outputs and exceptions."""
import types
import unittest
from scripts.ci_qwen_observe import observe
from worker.native_budgets import validate_job_cutoff, startup_cutoff, request_cutoff
from worker.transport import Refused


class ObservationTests(unittest.TestCase):
    def test_calls_outputs_bounds_restore_and_failure(self):
        hooks={}
        class Handle:
            def __init__(self,key):self.key=key
            def remove(self):hooks.pop(self.key)
        class Talker:
            def register_forward_pre_hook(self,fn):hooks['pre']=fn;return Handle('pre')
            def register_forward_hook(self,fn):hooks['post']=fn;return Handle('post')
        calls=[]; result=object(); stamp=[1.]
        def decode(*args,**kwargs):
            calls.append((args,kwargs))
            if kwargs.get('fail'):raise ValueError('original_decoder_cause')
            return result
        tokenizer=types.SimpleNamespace(decode=decode)
        model=types.SimpleNamespace(model=types.SimpleNamespace(talker=Talker(),speech_tokenizer=tokenizer))
        updates=[]
        def emit(**kw):
            if not updates:
                updates.append(kw)
                raise OSError('injected observation write failure')
            updates.append(kw)
        close=observe(model,emit,clock=lambda:stamp[0])
        for _ in range(1000):hooks['pre'](None,());hooks['post'](None,(),result)
        self.assertEqual(len(updates),1)
        self.assertEqual(updates[0]['talker_forward_started'],1)
        self.assertIs(tokenizer.decode(result,test=1),result)
        self.assertEqual(calls[-1],((result,),dict(test=1)))
        with self.assertRaisesRegex(ValueError,'original_decoder_cause'):tokenizer.decode(fail=True)
        close();self.assertIs(tokenizer.decode,decode);self.assertEqual(hooks,{})
        self.assertEqual(updates[-1]['talker_forward_completed'],1000)
        self.assertTrue(updates[-1]['decode_failed'])
        self.assertEqual(updates[-1]['observation_error']['message'],'injected observation write failure')

    def test_budget_caps_nonfinite_and_original_deadline(self):
        self.assertEqual(startup_cutoff(250,100),220)
        self.assertEqual(startup_cutoff(110,100),110)
        self.assertEqual(request_cutoff(188,163,25),188)
        for cutoff in (True,float('nan'),float('inf'),100,251):
            with self.subTest(cutoff=cutoff),self.assertRaises(Refused):validate_job_cutoff(cutoff,100)
        for cutoff,seconds in ((194,30),(188,True),(188,31),(163,25)):
            with self.subTest(cutoff=cutoff,seconds=seconds),self.assertRaises(Refused):request_cutoff(cutoff,163,seconds)
