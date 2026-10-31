"""No model: RSS loss, provisional exits and native CPU receipt coherence."""
import copy
import errno
import unittest
from worker.native_cpu_resources import assess,validate
from worker.engines.native_kokoro import MEMORY_BYTES
from worker.transport import Refused


def state(error=None,peak=100):
    return dict(count=2,peak=peak,last=dict(pid=123,identity=4,rss_bytes=100,method='proc_pid_rusage_v0',observed_at=.1),
        error=None if error is None else dict(type=error,message='fixture',errno=errno.ESRCH if error=='ProcessGone' else None,observed_at=.2))


class CPUResourceTests(unittest.TestCase):
    def test_real_sampler_pending_shape_has_no_bound_and_no_extra_grace(self):
        pending=dict(count=0,peak=None,last=None,error=None)
        first=assess(pending,started=10,now=10.1)
        self.assertIsNone(first['failure']);self.assertFalse(first['rss_bound_proven'])
        self.assertEqual(first['samples'],0)
        self.assertEqual(assess(pending,started=10,now=10.5)['failure'],'native_cpu_rss_lost')
    def test_gone_is_provisional_and_resolves_only_within_same_freshness(self):
        first=assess(state('ProcessGone'),started=0,now=.3)
        self.assertEqual(first['exit_resolution'],'provisional');self.assertIsNone(first['failure'])
        done=assess(state('ProcessGone'),started=0,now=.9,exit_observed_at=.4)
        self.assertEqual(done['exit_resolution'],'confirmed_within_freshness');self.assertFalse(done['observation_lost'])
        done['thread_stopped']=True
        validate(done,processes=[dict(pid=123,start_identity='4')],timely=True)
        for ending in (.6,.8):
            late=assess(state('ProcessGone'),started=0,now=ending,exit_observed_at=ending)
            self.assertEqual(late['failure'],'native_cpu_rss_lost');self.assertTrue(late['observation_lost'])
            late['thread_stopped']=True
            with self.assertRaises(Refused):validate(late,processes=[dict(pid=123,start_identity='4')],timely=True)

    def test_live_staleness_reader_failure_identity_change_and_memory_limit(self):
        for value,now,code in ((state(),.7,'native_cpu_rss_lost'),(state('RuntimeError'),.3,'native_cpu_rss_lost'),
                (state('ProcessReused'),.3,'native_cpu_identity_changed'),(state(peak=MEMORY_BYTES+1),.3,'native_cpu_memory_limit')):
            result=assess(value,started=0,now=now)
            self.assertEqual(result['failure'],code);self.assertFalse(result['rss_bound_proven'])
        no_sample=assess(dict(count=0,peak=0,last=None,error=None),started=0,now=.5)
        self.assertTrue(no_sample['observation_lost'])

    def test_receipt_requires_target_sample_and_thread_stop_for_success(self):
        result=assess(state(),started=0,now=.3,exit_observed_at=.3);result['thread_stopped']=True
        validate(result,processes=[dict(pid=123,start_identity='4')],timely=True)
        for change in (dict(thread_stopped=False),dict(samples=0),dict(observation_lost=True),dict(failure='native_cpu_rss_lost'),dict(age_at_observed_exit_seconds=.5)):
            with self.subTest(change=change),self.assertRaises(Refused):validate(dict(result,**change),processes=[dict(pid=123,start_identity='4')],timely=True)
        wrong=copy.deepcopy(result);wrong['last_valid_sample']['pid']=456
        with self.assertRaises(Refused):validate(wrong,processes=[dict(pid=123,start_identity='4')],timely=False)
        empty=assess(dict(count=0,peak=0,last=None,error=None),started=0,now=0);empty['thread_stopped']=True
        validate(empty,processes=[],timely=False)
        with self.assertRaises(Refused):validate(empty,processes=[],timely=True)


if __name__=='__main__':unittest.main()
