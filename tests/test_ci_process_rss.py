"""Real no-weight PID/RSS/pressure/death checks plus injected reader failures."""
import ctypes
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from scripts import ci_process_rss as rss

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('rss_owner', ROOT/'scripts/ci-apple-speech.py')
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class ProcessRSSTests(unittest.TestCase):
    def test_darwin_abi_bytes_target_identity_exit_and_errno(self):
        values=dict(start=123456,rss=987654321,exit=0,result=0,errno=0)
        calls=[]
        def query(pid,flavor,buffer):
            calls.append((pid,flavor))
            value=ctypes.cast(buffer,ctypes.POINTER(rss.RUsageV0)).contents
            value.start_abstime=values['start'];value.resident_size=values['rss'];value.exit_abstime=values['exit']
            ctypes.set_errno(values['errno'])
            return values['result']
        with patch.object(rss.sys,'platform','darwin'),patch.object(rss.ctypes,'CDLL',return_value=types.SimpleNamespace(proc_pid_rusage=query)):
            reader=rss.ProcessRSS(4321)
        self.assertEqual(ctypes.sizeof(rss.RUsageV0),96)
        sample=reader.sample()
        self.assertEqual(calls,[(4321,0)])
        self.assertEqual((sample['pid'],sample['identity'],sample['rss_bytes']),(4321,123456,987654321))
        values['start']+=1
        with self.assertRaises(rss.ProcessReused):reader.sample()
        values.update(start=123456,exit=77)
        with self.assertRaises(rss.ProcessGone):reader.sample()
        values.update(exit=0,result=-1,errno=errno.EACCES)
        with self.assertRaises(PermissionError):reader.sample()
        values['errno']=errno.ESRCH
        with self.assertRaises(rss.ProcessGone):reader.sample()

    def test_linux_pages_and_comm_parsing_without_darwin_symbol(self):
        # Fields 3..24; comm intentionally includes closing parentheses/spaces.
        fields=[b'S']+[b'0']*21;fields[19]=b'9001';fields[21]=b'17'
        raw=b'42 (some ) odd) name)) '+b' '.join(fields)
        with patch.object(rss.sys,'platform','linux'),patch.object(rss.ctypes,'CDLL',side_effect=AssertionError('Darwin symbol on Linux')):
            reader=rss.ProcessRSS(42)
            with patch('builtins.open',return_value=io.BytesIO(raw)):
                sample=reader.sample()
        self.assertEqual(sample['rss_bytes'],17*reader.page_size)
        self.assertEqual((sample['pid'],sample['identity']),(42,9001))
        for bad in (b'41 (wrong) '+b' '.join(fields),b'42 (truncated) S 0'):
            with patch('builtins.open',return_value=io.BytesIO(bad)),self.assertRaises(ValueError):reader.sample()
        with patch('builtins.open',side_effect=FileNotFoundError),self.assertRaises(rss.ProcessGone):reader.sample()
        with patch('builtins.open',side_effect=PermissionError),self.assertRaises(PermissionError):reader.sample()

    def test_invalid_target_pid(self):
        for pid in (0,-1,True,1.5,2**31):
            with self.subTest(pid=pid),self.assertRaises(ValueError):rss.ProcessRSS(pid)

    def test_real_target_growth_bytes_and_exit(self):
        source="import sys;print('ready');sys.stdin.readline();b=bytearray(32*1024**2);b[::4096]=b'x'*(len(b)//4096);print('allocated');sys.stdin.readline()"
        child=subprocess.Popen([sys.executable,'-u','-c',source],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        try:
            self.assertTrue(select.select([child.stdout],[],[],5)[0]);self.assertEqual(child.stdout.readline(),b'ready\n')
            reader=rss.ProcessRSS(child.pid);before=reader.sample()
            child.stdin.write(b'go\n');child.stdin.flush()
            self.assertTrue(select.select([child.stdout],[],[],5)[0]);self.assertEqual(child.stdout.readline(),b'allocated\n')
            after=reader.sample()
            self.assertEqual((after['pid'],after['identity']),(child.pid,before['identity']))
            self.assertGreater(after['rss_bytes']-before['rss_bytes'],16*1024**2)
        finally:
            child.kill();child.communicate(timeout=5)
        with self.assertRaises((rss.ProcessGone,rss.ProcessReused)):reader.sample()


class OwnedRSSRegressionTests(unittest.TestCase):
    def run_child(self,source,observation,*,seconds=3):
        return ci.owned_child([sys.executable,'-I','-u','-c',source],dict(os.environ),self.path,observation,seconds=seconds)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'child.log'

    def test_real_pressure_trips_guard_without_ps_and_records_signal(self):
        observation={}
        source="import time;b=bytearray(64*1024**2);b[::4096]=b'x'*(len(b)//4096);print('resident');time.sleep(5)"
        with patch.object(ci,'MAX_RSS',32*1024**2),patch.object(ci.subprocess,'check_output',side_effect=AssertionError('No ps subprocess')):
            with self.assertRaisesRegex(MemoryError,'RSS_limit'):self.run_child(source,observation)
        self.assertGreater(observation['sampled_peak_rss_bytes'],32*1024**2)
        self.assertEqual(observation['last_valid_rss_sample']['pid'],observation['child_pid'])
        self.assertTrue(observation['child_reaped']);self.assertTrue(observation['sampler_thread_stopped'])
        self.assertFalse(observation['rss_bound_proven'])
        self.assertEqual(observation['termination']['signals'][0]['signal'],'SIGKILL')
        self.assertTrue(observation['termination']['signals'][0]['accepted'])
        self.assertEqual(observation['observed_exit']['signal'],'SIGKILL')

    def test_reader_failure_preserves_last_sample_and_fails_closed(self):
        class FailsAfterSample:
            def __init__(self,pid):self.real=rss.ProcessRSS(pid);self.calls=0
            def sample(self):
                self.calls+=1
                if self.calls>1:raise PermissionError(errno.EACCES,'injected reader failure')
                return self.real.sample()
        observation={}
        with patch.object(ci,'Sampler',side_effect=lambda pid:rss.Sampler(pid,reader_factory=FailsAfterSample)):
            with self.assertRaisesRegex(ci.SamplerLost,'PermissionError'):self.run_child('import time;time.sleep(5)',observation)
        self.assertTrue(observation['rss_observation_lost']);self.assertEqual(observation['rss_guard_status'],'lost')
        self.assertEqual(observation['sampler_error']['type'],'PermissionError')
        self.assertEqual(observation['rss_samples'],1)
        self.assertIsNotNone(observation['last_valid_rss_age_seconds'])
        self.assertTrue(observation['child_reaped']);self.assertFalse(observation['rss_bound_proven'])

    def slow_sampler(self,*,first_sample):
        gate=threading.Event();samplers=[]
        class Slow:
            def __init__(self,pid):self.real=rss.ProcessRSS(pid);self.calls=0
            def sample(self):
                self.calls+=1
                if not first_sample or self.calls>1:gate.wait(5)
                return self.real.sample()
        def factory(pid):
            value=rss.Sampler(pid,reader_factory=Slow);samplers.append(value);return value
        self.addCleanup(lambda:(gate.set(),[s.thread.join(timeout=1) for s in samplers]))
        return factory,samplers

    def test_slow_reader_cannot_hold_elapsed_deadline_or_child_cleanup(self):
        factory,samplers=self.slow_sampler(first_sample=False);observation={};started=time.monotonic()
        with patch.object(ci,'Sampler',side_effect=factory):
            with self.assertRaises(TimeoutError):self.run_child("print('before slow sample');import time;time.sleep(5)",observation,seconds=.15)
        self.assertLess(time.monotonic()-started,1.5)
        self.assertEqual(len(samplers),1);self.assertFalse(observation['sampler_thread_stopped'])
        self.assertTrue(observation['child_reaped']);self.assertTrue(observation['rss_observation_lost'])
        self.assertIsNone(observation['sampled_peak_rss_bytes'])
        self.assertIn(b'before slow sample',self.path.read_bytes())

    def test_stale_reader_does_not_replace_thread_or_reuse_safe_peak(self):
        factory,samplers=self.slow_sampler(first_sample=True);observation={}
        with patch.object(ci,'Sampler',side_effect=factory):
            with self.assertRaisesRegex(ci.SamplerLost,'stale'):self.run_child('import time;time.sleep(5)',observation)
        self.assertEqual(len(samplers),1);self.assertEqual(observation['rss_samples'],1)
        self.assertGreater(observation['last_valid_rss_age_seconds'],ci.RSS_STALE_SECONDS)
        self.assertTrue(observation['rss_observation_lost']);self.assertFalse(observation['rss_bound_proven'])
        self.assertEqual(observation['sampler_error']['type'],'StaleSample')

    def test_process_death_has_observed_exit_without_requested_signal(self):
        observation={}
        with self.assertRaisesRegex(RuntimeError,'native_child_failed'):
            self.run_child("import os,signal,time;print('before signal');time.sleep(.2);os.kill(os.getpid(),signal.SIGTERM)",observation)
        self.assertEqual(observation['observed_exit']['signal'],'SIGTERM')
        self.assertEqual(observation['termination'],dict(requested_reason=None,signals=[]))
        self.assertTrue(observation['child_reaped']);self.assertIn(b'before signal',self.path.read_bytes())

    def test_bounded_phase_progress_and_invalid_durations(self):
        self.path.write_bytes(b' '*65537)
        with self.assertRaisesRegex(ValueError,'progress_limit'):ci.read_progress(self.path)
        for value in (True,float('nan'),float('inf'),-1):
            with self.subTest(value=value),self.assertRaises(ValueError):
                ci.phase_remaining(dict(stage='cold_load',stage_started=0,cold_load_seconds=value),1)
