"""Installed journey controller contracts only; no model downloads or synthesis."""
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import wave

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kokoro_installed_controller',ROOT/'scripts/ci-native-kokoro-installed.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class InstalledControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.root.chmod(0o700)

    def test_real_slow_owned_child_deadline_then_term_and_confirmed_reap(self):
        records=[];child=c.Child('cli',[sys.executable,'-u','-c','import time; time.sleep(30)'],self.root,time.monotonic()+40,records)
        try:
            end=time.monotonic()+1
            while child.record['identity'] is None and time.monotonic()<end:child.observe();time.sleep(.02)
            with self.assertRaisesRegex(c.Failure,'child_deadline'):child.wait(.15)
        finally:child.close(force_allowed=True)
        self.assertTrue(records[0]['reaped']);self.assertTrue(records[0]['sampler_stopped'])
        self.assertEqual([x['signal'] for x in records[0]['signals']],['SIGTERM'])
        self.assertNotIn(child.handle.pid,c._LIVE)

    def test_cli_stdout_json_is_separate_from_bounded_stderr_warning(self):
        records=[]
        code='import json,sys; sys.stderr.write("inert warning\\n"); print(json.dumps({"value":"café"},ensure_ascii=False))'
        child=c.Child('cli',[sys.executable,'-u','-c',code],self.root,time.monotonic()+40,records)
        try:self.assertEqual(child.wait(10),0)
        finally:child.close(force_allowed=True)
        self.assertEqual(json.loads(child.path.read_bytes()),dict(value='café'))
        self.assertEqual(child.stderr_path.read_text(),'inert warning\n')

    def test_separate_streams_keep_original_combined_log_bound_after_exit(self):
        records=[];code='import sys; print("{}"); sys.stderr.write("x"*(1024**2))'
        child=c.Child('cli',[sys.executable,'-u','-c',code],self.root,time.monotonic()+40,records)
        try:
            with self.assertRaisesRegex(c.Failure,'log_limit'):child.wait(10)
        finally:
            if not records[0]['reaped']:child.close(force_allowed=True)
        self.assertTrue(records[0]['reaped']);self.assertNotIn(child.handle.pid,c._LIVE)

    def test_canonical_non_ascii_matches_product_and_checks_loaded_source_closure(self):
        from worker.transport import canonical
        value=dict(path='/private/café/voix',text='été 🔈')
        self.assertEqual(c.canonical(value),canonical(value));self.assertIn('café'.encode(),c.canonical(value))
        counts=c.installed_modules(ROOT);self.assertGreater(sum(counts.values()),1)
        with patch.dict(sys.modules,{'worker.inert_foreign':SimpleNamespace(__file__='/foreign/worker.py')}),self.assertRaisesRegex(c.Failure,'installed_source_closure'):
            c.installed_modules(ROOT)

    def test_unknown_nested_custody_never_forces_kill_or_drops_live_handle(self):
        child=c.Child.__new__(c.Child);child.role='service';child.handle=SimpleNamespace(pid=123,poll=lambda:None)
        child.record=dict(cleanup_unknown=False);child.send=Mock();child.wait=Mock(side_effect=c.Failure('fixture deadline'))
        c._LIVE[123]=child
        try:
            with self.assertRaisesRegex(c.Failure,'nested_custody_unknown'):child.close(force_allowed=False)
            self.assertEqual(child.send.call_args_list,[unittest.mock.call(signal.SIGTERM)])
            self.assertTrue(child.record['cleanup_unknown']);self.assertIs(c._LIVE[123],child)
        finally:c._LIVE.pop(123)

    def test_deadline_is_original_total_and_artifacts_redact_private_material(self):
        with patch.object(c.time,'monotonic',return_value=580),self.assertRaises(c.Failure):c.budget(600)
        token='a'*96
        value=c.public(dict(pid=123,doctor=dict(token=token,credential='never publish',private_key='never publish'),
            error='Authorization: Bearer '+token,log='-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----'))
        raw=json.dumps(value)
        self.assertNotIn(token,raw);self.assertNotIn('never publish',raw);self.assertNotIn('secret',raw)
        self.assertEqual(value['pid'],123)

    def test_published_files_exclude_private_tree_and_scrub_nested_setup_logs(self):
        private=self.root/'.private';installation=private/'installation';installation.mkdir(parents=True,mode=0o700)
        (installation/'tls').mkdir();token='b'*96
        (installation/'credential.json').write_text(json.dumps(dict(token=token)))
        (installation/'tls/client.key').write_text('PRIVATE_KEY_FILE_SENTINEL')
        (installation/'operator-config.txt').write_text('PRIVATE_CONFIG_SENTINEL')
        (private/'setup-0.log').write_text('Authorization: Bearer '+token+'\nnormal progress\n')
        (installation/'finish-setup.log').write_text('failure '+token+'\n-----BEGIN PRIVATE KEY-----\nINERT_KEY_BODY\n-----END PRIVATE KEY-----')
        c.publish(self.root,dict(status='failed',credential=token))
        exported=[self.root/'report.json',*(self.root/'safe-logs').iterdir()]
        raw=''.join(p.read_text() for p in exported)
        for secret in (token,'PRIVATE_KEY_FILE_SENTINEL','PRIVATE_CONFIG_SENTINEL','INERT_KEY_BODY'):self.assertNotIn(secret,raw)
        self.assertIn('normal progress',raw)
        self.assertEqual({p.name for p in (self.root/'safe-logs').iterdir()},{'setup-0.log','setup-finish-setup.log'})

    def test_late_zero_exit_does_not_turn_expired_work_into_success(self):
        child=c.Child.__new__(c.Child);child.deadline=600;child.observe=Mock(return_value=0);child.reap=Mock()
        with patch.object(c.time,'monotonic',return_value=580),self.assertRaises(c.Failure):child.wait(10)
        child.reap.assert_not_called()  # cleanup still separately observes/waits it

    def test_identity_change_fails_before_any_signal(self):
        child=c.Child.__new__(c.Child);child.handle=SimpleNamespace(pid=123,poll=lambda:None)
        child.record=dict(identity=1);child.sampler=SimpleNamespace(snapshot=lambda:dict(last=dict(pid=123,identity=2,observed_at=time.monotonic()),peak=1,count=1,error=None))
        with patch.object(c.os,'killpg') as signal_group,self.assertRaisesRegex(c.Failure,'identity_changed'):child.observe()
        signal_group.assert_not_called()

    def test_live_unknown_custody_retains_private_tree(self):
        private=self.root/'.private';private.mkdir(mode=0o700);installation=private/'installation';installation.mkdir(mode=0o700)
        deadline=time.monotonic()+60
        state=dict(phase='setup_complete',deadline=deadline,started=time.monotonic(),root=str(installation),source_commit='a'*40,
            controller_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),owners=[],setup_cleanup_confirmed=True)
        c.save(private/'controller-state.json',state)
        child=SimpleNamespace(role='service',close=Mock(side_effect=c.Failure('fixture custody unknown')))
        c._LIVE[123]=child
        try:
            with patch.dict(os.environ,dict(GITHUB_REF='refs/heads/feat/apple-silicon-native',GITHUB_SHA='a'*40)),patch.object(c,'Journey') as journey,patch.object(c,'product_flow'),patch.object(c.shutil,'rmtree') as remove:
                journey.return_value.family_stopped.return_value=True;journey.return_value.evidence.return_value=[]
                result=c.run(SimpleNamespace(output=self.root,installed_phase=True,deadline=deadline))
                self.assertEqual(result,1);remove.assert_not_called()
            report=json.loads((self.root/'report.json').read_bytes())
            self.assertFalse(report['private_tree_removed']);self.assertEqual(report['live_owned_pids'],[123])
            self.assertTrue(installation.exists())
        finally:c._LIVE.pop(123,None)

    def test_original_four_job_limit_precedes_any_extra_write(self):
        journey=c.Journey.__new__(c.Journey);journey.state=dict(submitted_jobs=4);journey.http=Mock()
        with self.assertRaisesRegex(c.Failure,'four_job_limit'):journey.submit('no-extra',{})
        journey.http.assert_not_called()

    def test_blocking_http_does_not_block_custody_checks_or_spawn_replacement(self):
        journey=c.Journey.__new__(c.Journey);journey.deadline=time.monotonic()+40;journey.http_thread=None;journey.root=self.root
        entered=threading.Event();release=threading.Event()
        @contextmanager
        def credentials(root):
            with patch.dict(os.environ,dict(SPEECH_CREDENTIAL='inert-source-fixture')):yield
        class HTTP:
            def __init__(self,**kw):pass
            def call(self,*a,**kw):entered.set();release.wait(2);return {}
        journey.client_environment=credentials;journey.HTTP=HTTP
        observe=Mock(side_effect=[None,c.Failure('fixture sampler lost')]);c._LIVE[123]=SimpleNamespace(observe=observe)
        try:
            with self.assertRaisesRegex(c.Failure,'sampler lost'):journey.http('inert')
            self.assertTrue(entered.is_set());self.assertTrue(journey.http_thread.is_alive())
            c._LIVE.pop(123)
            original=journey.http_thread
            with self.assertRaisesRegex(c.Failure,'prior_http_custody_unknown'):journey.http('inert')
            self.assertIs(journey.http_thread,original)
        finally:
            c._LIVE.pop(123,None);release.set();journey.close_http()

    def test_fixed_role_order_has_one_session_and_stops_first_failure(self):
        events=[]
        class Fake:
            root='/inert';state={}
            def cli(self,*args):events.append(args[0])
            def model_identity(self):events.append('model');return {}
            def start_service(self):events.append('start')
            def first_public(self):events.append('first')
            def cancellations(self):events.append('cancel')
            def stop_mcp(self):events.append('mcp-close')
            def stop_service(self):events.append('stop')
            def restarted(self):events.append('restart-public')
        c.product_flow(Fake())
        self.assertEqual(events,['doctor','model','qualification-session','start','first','cancel','mcp-close','stop','start','restart-public','stop'])
        events.clear()
        with patch.object(Fake,'first_public',side_effect=c.Failure('first cause')):
            with self.assertRaisesRegex(c.Failure,'first cause'):c.product_flow(Fake())
        self.assertEqual(events,['doctor','model','qualification-session','start'])

    def test_terminal_failure_before_cancel_is_not_reported_as_completion_win(self):
        for state,reason in (('succeeded','cancel_completion_won'),('failed','job_failed_before_cancel'),('cancelled','job_cancelled_before_request')):
            with self.subTest(state=state):
                journey=c.Journey.__new__(c.Journey);journey.state={};journey.request={};journey.deadline=time.monotonic()+60
                journey.submit=Mock(return_value=dict(job_id='inert'))
                journey.http=Mock(side_effect=[dict(status=dict(phase='idle')),dict(state=state)])
                journey.cli=Mock();journey.mcp=Mock()
                with self.assertRaisesRegex(c.Failure,reason):journey.cancellations()
                self.assertEqual(journey.state['pre_cancel_terminal_job']['state'],state)
                self.assertEqual(journey.submit.call_count,1);journey.cli.assert_not_called();journey.mcp.tool.assert_not_called()

    def test_post_cancel_labels_and_warm_http_mcp_transport_order(self):
        cancelled=dict(id='running',state='cancelled',attempts=[dict(reaped=True)],artifact_ids=[])
        queued=dict(id='queued',state='cancelled',stage='never_started',attempts=[],artifact_ids=[])
        cases=[(dict(cancelled,state='succeeded'),queued,'cancel_completion_won'),
            (cancelled,dict(queued,state='succeeded',attempts=[dict(reaped=True)]),'queued_cancel_lost_job_started'),
            (cancelled,dict(queued,attempts=[dict(reaped=True)]),'queued_cancel_lost_job_started'),
            (dict(cancelled,state='failed'),queued,'running_job_failed_not_cancelled'),(cancelled,queued,None)]
        for a,b,reason in cases:
            with self.subTest(reason=reason):
                journey=c.Journey.__new__(c.Journey);journey.state={};journey.request={};journey.deadline=time.monotonic()+60
                journey.submit=Mock(side_effect=[dict(job_id='running'),dict(job_id='queued')])
                journey.http=Mock(side_effect=[dict(status=dict(phase='observing')),dict(state='running'),{}])
                journey.mcp=Mock();journey.cli=Mock();journey.wait_job=Mock(side_effect=[a,b])
                journey.evidence=Mock(return_value=[dict(job_id='running',receipt=dict(reaped=True,timely=False,processes=[{}],termination=dict(reason='cancelled')))])
                if reason:
                    with self.assertRaisesRegex(c.Failure,reason):journey.cancellations()
                else:journey.cancellations()
                self.assertEqual(journey.state['cancel_jobs'],[a,b]);self.assertEqual(journey.submit.call_count,2)
                self.assertEqual(journey.http.call_args_list[-1],unittest.mock.call('job_cancel',dict(job_id='queued'),key='installed-cancel-queued'))
                journey.mcp.tool.assert_called_once_with('speech_job_cancel',dict(job_id='running',idempotency_key='installed-cancel-running'))
                journey.cli.assert_not_called()

    def test_bounded_ranges_whole_hash_and_nonempty_playable_wav(self):
        output=io.BytesIO()
        with wave.open(output,'wb') as wav:wav.setparams((1,2,24000,0,'NONE','none'));wav.writeframes(b'\x01\0'*4800)
        raw=output.getvalue();asset=dict(id='inert',sha256=hashlib.sha256(raw).hexdigest());calls=[]
        def transfer(p,*,cutoff):
            calls.append(p);start,end=p['start'],p['end_exclusive']
            return dict(start=start,end_exclusive=end,total_bytes=len(raw),content_sha256=asset['sha256'],content_base64=base64.b64encode(raw[start:end]).decode())
        self.assertEqual(c.download(transfer,asset,time.monotonic()+60,mcp=True,known_size=len(raw)),raw)
        self.assertLessEqual(len(calls),43);self.assertTrue(all(p['end_exclusive']-p['start']<=3072 for p in calls))
        self.assertEqual(c.wav_info(raw)['sha256'],asset['sha256'])
        with self.assertRaisesRegex(c.Failure,'size_limit'):c.download(transfer,asset,time.monotonic()+60,mcp=True,known_size=128*1024+1)

    def test_range_rounds_share_one_cumulative_budget(self):
        now=[100.];calls=[];raw=b'x'*12000;asset=dict(id='inert',sha256=hashlib.sha256(raw).hexdigest())
        def transfer(p,*,cutoff):
            calls.append(p);now[0]+=6
            self.assertEqual(cutoff,120.)
            return dict(start=p['start'],end_exclusive=p['end_exclusive'],total_bytes=len(raw),content_sha256=asset['sha256'],content_base64=base64.b64encode(raw[p['start']:p['end_exclusive']]).decode())
        with patch.object(c.time,'monotonic',side_effect=lambda:now[0]),self.assertRaisesRegex(c.Failure,'download_total_deadline'):
            c.download(transfer,asset,1000,mcp=True,known_size=len(raw))
        self.assertEqual(len(calls),4)

    def test_mcp_ranges_can_use_more_than_ten_seconds_without_renewing_twenty(self):
        now=[100.];raw=b'x'*(12*3072);asset=dict(id='inert',sha256=hashlib.sha256(raw).hexdigest());cutoffs=[]
        def transfer(p,*,cutoff):
            cutoffs.append(cutoff);now[0]+=1
            return dict(start=p['start'],end_exclusive=p['end_exclusive'],total_bytes=len(raw),content_sha256=asset['sha256'],content_base64=base64.b64encode(raw[p['start']:p['end_exclusive']]).decode())
        with patch.object(c.time,'monotonic',side_effect=lambda:now[0]):
            self.assertEqual(c.download(transfer,asset,1000,mcp=True,known_size=len(raw)),raw)
        self.assertEqual(now[0],112.);self.assertEqual(cutoffs,[120.]*12)

    def test_mcp_roundtrip_honors_shared_cutoff_even_with_buffered_late_reply(self):
        client=c.MCP.__new__(c.MCP);client.deadline=1000;client.seq=0;client.child=SimpleNamespace(observe=Mock())
        client.write=Mock();client.buffer=c.canonical(dict(jsonrpc='2.0',id=1,result={}))+b'\n'
        with patch.object(c.time,'monotonic',return_value=119.):self.assertEqual(client.call('ping',{},cutoff=120.),{})
        self.assertEqual(client.write.call_args.kwargs,dict(cutoff=120.))
        client.seq=0;client.buffer=c.canonical(dict(jsonrpc='2.0',id=1,result={}))+b'\n'
        with patch.object(c.time,'monotonic',return_value=120.),self.assertRaisesRegex(c.Failure,'mcp_roundtrip_deadline'):
            client.call('ping',{},cutoff=120.)

    def test_real_mcp_stdio_protocol_without_http_or_weights(self):
        records=[];child=c.Child('mcp',[sys.executable,'-B','-m','clients.mcp'],self.root,time.monotonic()+60,records,pipes=True)
        client=None
        try:
            client=c.MCP(child,child.deadline)
            self.assertEqual(client.call('ping',{}),{})
        finally:
            if client:client.selector.close()
            child.close(force_allowed=True)
        self.assertEqual(records[0]['exit'],0);self.assertTrue(records[0]['reaped'])

    def test_trusted_manual_workflow_and_archive_roles_are_closed(self):
        workflow=(ROOT/'.github/workflows/apple-kokoro.yml').read_text()
        job=workflow.split('  installed-journey:',1)[1]
        for value in ("github.event_name == 'workflow_dispatch'",'inputs.installed_journey','!inputs.en_backend',
                "refs/heads/feat/apple-silicon-native","refs/heads/main",'timeout-minutes: 12','--run-installed-journey'):
            self.assertIn(value,job)
        artifacts=job.split('path: |',1)[1].split('retention-days:',1)[0]
        self.assertNotIn('.private',artifacts);self.assertNotIn('installation',artifacts)
        self.assertIn('safe-logs/*.log',artifacts)
        from service.native_cpu_package import members
        self.assertNotIn('scripts/ci-native-kokoro-installed.py',members())  # CI controller is not product archive input
        image=(ROOT/'Dockerfile.tests').read_text();self.assertIn('COPY scripts /app/scripts',image)


if __name__=='__main__':unittest.main()
