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
        self.assertNotIn('effective_budgets',observation)  # No phase monitor was supplied.
        self.assertEqual(observation['owner_total_seconds'],3)
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


class ExitRaceTests(unittest.TestCase):
    """Deterministic controller transitions; no delays or real signals."""
    def exercise(self, polls, *, error='ProcessGone', seconds=1, cancel=False,
                 probe_errno=None, signal_errno=None, eof_after=1, last=True):
        from contextlib import ExitStack
        from unittest.mock import Mock
        clock=types.SimpleNamespace(now=0.)
        class Child:
            pid=999999
            returncode=None
            def __init__(self):self.polls=list(polls);self.waited=False;self.stdout=Mock()
            def poll(self):
                if self.returncode is not None:return self.returncode
                value=self.polls.pop(0) if self.polls else None
                if isinstance(value,tuple):value,clock.now=value
                if value is not None:self.returncode=value
                return value
            def wait(self,timeout):
                self.waited=True
                if self.returncode is None:self.returncode=0
                return self.returncode
        child=Child()
        class Selector:
            calls=0
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def register(self,*args):pass
            def unregister(self,*args):pass
            def select(self,timeout):
                self.calls+=1;clock.now+=.1
                if cancel:raise KeyboardInterrupt('cancelled')
                return [(types.SimpleNamespace(fd=1,fileobj=child.stdout),1)] if self.calls==eof_after else []
        selector=Selector()
        sample=dict(pid=child.pid,identity=77,rss_bytes=1024,observed_at=0.,method='fixture') if last else None
        state=dict(last=sample,count=int(last),peak=1024 if last else None,
            error=dict(type=error,errno=errno.ESRCH if error=='ProcessGone' else None,message='injected '+error,observed_at=0.))
        sampler=Mock();sampler.snapshot.return_value=state;sampler.stop.return_value=True
        def signal_group(pid,number):
            child.returncode=0 if signal_errno else -number
            if signal_errno:raise PermissionError(signal_errno,'injected signal failure')
        signal_mock=Mock(side_effect=signal_group)
        probe_mock=Mock(side_effect=OSError(probe_errno,'injected existence probe') if probe_errno else None)
        observation={};failure=None
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            stack.enter_context(patch.object(ci.time,'monotonic',side_effect=lambda:clock.now))
            stack.enter_context(patch.object(ci,'Sampler',return_value=sampler))
            stack.enter_context(patch.object(ci.subprocess,'Popen',return_value=child))
            stack.enter_context(patch.object(ci.selectors,'DefaultSelector',return_value=selector))
            stack.enter_context(patch.object(ci.os,'read',return_value=b''))
            stack.enter_context(patch.object(ci.os,'kill',probe_mock))
            stack.enter_context(patch.object(ci.os,'killpg',signal_mock))
            try:ci.owned_child(['fixture'],{},Path(tmp)/'log',observation,seconds=seconds)
            except BaseException as exc:failure=exc
        self.assertTrue(child.waited);self.assertTrue(observation['child_reaped'])
        return observation,failure,signal_mock,probe_mock,clock.now

    def test_gone_then_poll_none_to_zero_resolves_without_signal_or_loss(self):
        r,f,signal_mock,probe,_=self.exercise([None,None,0])
        self.assertIsNone(f);signal_mock.assert_not_called();probe.assert_not_called()
        self.assertTrue(r['sampler_target_exit_confirmed']);self.assertFalse(r['rss_observation_lost'])
        self.assertEqual(r['sampler_exit_resolution'],'confirmed_within_freshness')
        self.assertEqual(r['termination'],dict(requested_reason=None,signals=[]))
        self.assertEqual(r['sampler_error']['type'],'ProcessGone')

    def test_gone_then_nonzero_is_child_failure_not_sampling_loss(self):
        r,f,signals,_,_=self.exercise([None,None,7])
        self.assertIs(type(f),RuntimeError);self.assertIn('native_child_failed',str(f))
        self.assertEqual(r['observed_exit']['returncode'],7)
        self.assertFalse(r['rss_observation_lost']);signals.assert_not_called()

    def test_gone_live_past_original_freshness_fails_closed_and_reaps(self):
        r,f,signals,probe,elapsed=self.exercise([None]*20)
        self.assertIsInstance(f,ci.SamplerLost);self.assertGreaterEqual(elapsed,.5)
        self.assertTrue(r['rss_observation_lost']);self.assertFalse(r['rss_bound_proven'])
        self.assertEqual(r['sampler_exit_resolution'],'confirmed_after_observation_lost')
        probe.assert_called_once_with(999999,0);signals.assert_called_once()
        self.assertTrue(r['termination']['signals'][0]['accepted'])

    def test_identity_change_fails_immediately_and_never_signals_replacement(self):
        r,f,signals,probe,elapsed=self.exercise([None],error='ProcessReused')
        self.assertIsInstance(f,ci.SamplerLost);self.assertEqual(elapsed,0)
        signals.assert_not_called();probe.assert_not_called()
        self.assertTrue(r['ownership_uncertain']);self.assertTrue(r['rss_observation_lost'])
        self.assertEqual(r['termination']['signal_skipped_reason'],'process_identity_changed')

    def test_late_zero_still_fails_owner_deadline(self):
        r,f,signals,_,_=self.exercise([None,(0,.2)],seconds=.15)
        self.assertIsInstance(f,TimeoutError);signals.assert_not_called()
        self.assertEqual(r['observed_exit']['returncode'],0)
        self.assertEqual(r['sampler_exit_resolution'],'confirmed_after_owner_deadline')
        self.assertFalse(r['rss_observation_lost'])  # Owner timeout is distinct from fresh RSS.

    def test_exit_confirmed_in_time_does_not_expire_during_eof_drain(self):
        r,f,signals,_,_=self.exercise([None,0],eof_after=7)
        self.assertIsNone(f);signals.assert_not_called()
        self.assertFalse(r['rss_observation_lost']);self.assertEqual(r['sampler_exit_confirmed_at'],.1)

    def test_cancellation_kills_owned_live_child_without_waiting_for_grace(self):
        r,f,signals,_,elapsed=self.exercise([None]*10,cancel=True)
        self.assertIsInstance(f,KeyboardInterrupt);self.assertLess(elapsed,.5)
        signals.assert_called_once();self.assertEqual(r['observed_exit']['signal'],'SIGKILL')
        self.assertEqual(r['sampler_exit_resolution'],'confirmed_during_cleanup')

    def test_gone_on_cancel_skips_signal_and_retains_probe_eperm(self):
        r,f,signals,_,_=self.exercise([None]*10,cancel=True,probe_errno=errno.EPERM)
        self.assertIsInstance(f,KeyboardInterrupt);signals.assert_not_called()
        self.assertEqual(r['termination']['signals'],[])
        self.assertEqual(r['termination']['existence_probe_error']['errno'],errno.EPERM)
        self.assertEqual(r['observed_exit']['returncode'],0)

    def test_requested_but_denied_signal_is_not_observed_exit_cause(self):
        r,f,signals,_,_=self.exercise([None]*10,cancel=True,signal_errno=errno.EPERM)
        self.assertIsInstance(f,KeyboardInterrupt);signals.assert_called_once()
        self.assertFalse(r['termination']['signals'][0]['accepted'])
        self.assertEqual(r['termination']['signals'][0]['error']['errno'],errno.EPERM)
        self.assertIsNone(r['observed_exit']['signal'])

    def test_no_sample_cannot_be_promoted_by_zero_exit(self):
        r,f,signals,_,_=self.exercise([0],last=False)
        self.assertIsInstance(f,ci.SamplerLost);signals.assert_not_called()
        self.assertTrue(r['rss_observation_lost']);self.assertIsNone(r['sampled_peak_rss_bytes'])


class RealExitStressTests(unittest.TestCase):
    def test_bounded_real_child_exit_batch(self):
        """One sequential 24-exit batch, no model/import/network or retries."""
        started=time.monotonic();cutoff=started+20
        report=dict(status='failed',cases=[],batch_limit_seconds=20,
            child_limit_seconds=2,rss_limit_bytes=128*1024**2,log_limit_bytes=4096)
        observation={}
        try:
            with tempfile.TemporaryDirectory() as tmp,patch.object(ci,'MAX_RSS',report['rss_limit_bytes']),patch.object(ci,'MAX_LOG',report['log_limit_bytes']):
                for index in range(24):
                    budget=min(2,cutoff-time.monotonic())
                    if budget<=0:raise TimeoutError('real_exit_stress_20_seconds')
                    expected=(0,7,-15)[index%3]
                    ending='os.kill(os.getpid(),signal.SIGTERM)' if expected<0 else 'raise SystemExit('+str(expected)+')'
                    source="import os,signal,time;b=bytearray(8*1024**2);b[::4096]=b'x'*(len(b)//4096);print('exit-case-"+str(index)+"');time.sleep("+str((.11,.135,.16,.19)[index%4])+");"+ending
                    observation={};failure=None
                    try:
                        ci.owned_child([sys.executable,'-I','-u','-c',source],dict(os.environ),Path(tmp)/f'{index}.log',observation,seconds=budget)
                    except RuntimeError as exc:failure=exc
                    if expected==0:self.assertIsNone(failure)
                    else:
                        self.assertIs(type(failure),RuntimeError)
                        self.assertIn('native_child_failed',str(failure))
                    self.assertEqual(observation['observed_exit']['returncode'],expected)
                    self.assertTrue(observation['child_reaped']);self.assertTrue(observation['sampler_thread_stopped'])
                    self.assertFalse(observation['rss_observation_lost'])
                    self.assertGreater(observation['rss_samples'],0)
                    self.assertEqual(observation['termination']['signals'],[])
                    self.assertLessEqual(observation['sampled_peak_rss_bytes'],report['rss_limit_bytes'])
                    with self.assertRaises(ChildProcessError):os.waitpid(observation['child_pid'],os.WNOHANG)
                    report['cases'].append(dict(index=index,expected=expected,observed=observation['child_exit'],
                        seconds=observation['child_seconds'],samples=observation['rss_samples'],peak=observation['sampled_peak_rss_bytes'],
                        exit_resolution=observation.get('sampler_exit_resolution'),sampler_error=(observation.get('sampler_error') or {}).get('type'),
                        signals_requested=len(observation['termination']['signals']),reaped=observation['child_reaped']))
            self.assertLess(time.monotonic(),cutoff)
            report['status']='passed'
        except BaseException as exc:
            report.update(error=dict(type=type(exc).__name__,message=str(exc)[:1000]),failed_observation=observation)
            raise
        finally:
            report['seconds']=time.monotonic()-started
            path=os.environ.get('APPLE_EXIT_STRESS_REPORT')
            if path:
                target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
