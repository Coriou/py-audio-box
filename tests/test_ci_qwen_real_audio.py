"""Inert media validation for the manual gate; no Docker or model operation."""
import importlib.util
import json
from pathlib import Path
from unittest import mock
import sys
import tempfile
import unittest
import wave

SOURCE = Path(__file__).resolve().parents[1] / 'scripts' / 'ci-qwen-real-audio.py'
spec = importlib.util.spec_from_file_location('ci_qwen_real_audio', SOURCE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class RealAudioMediaTests(unittest.TestCase):
    def test_real_nonzero_pcm16_has_stable_hash_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'real.wav'
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(24000)
                out.writeframes((500).to_bytes(2, 'little', signed=True) * 24000)
            result = gate.wav(path)
            self.assertEqual(result['duration_seconds'], 1)
            self.assertEqual(result['sha256'], gate.sha(path))
            self.assertGreater(result['rms'], 0)

    def test_silent_media_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'silent.wav'
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(24000)
                out.writeframes(b'\0\0' * 24000)
            with self.assertRaisesRegex(RuntimeError, 'silent_audio'):
                gate.wav(path)


class RealAudioControlTests(unittest.TestCase):
    SHA = 'a' * 40

    def test_25_gib_admission_retains_observations_without_actions(self):
        report = {'steps': []}
        meminfo = 'MemTotal: 16384000 kB\nMemAvailable: 2621440 kB\n'
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(gate.platform, 'system', return_value='Linux'), \
             mock.patch.object(gate.platform, 'machine', return_value='x86_64'), \
             mock.patch.object(gate.os, 'geteuid', return_value=0), \
             mock.patch.object(gate.shutil, 'disk_usage', return_value=mock.Mock(free=30*gate.GIB)), \
             mock.patch.object(Path, 'is_file', return_value=True), \
             mock.patch.object(Path, 'read_text', return_value=meminfo), \
             mock.patch.object(gate, 'json_command') as docker:
            with mock.patch.object(gate.sys, 'version_info', (3, 11, 0)):
                with self.assertRaisesRegex(RuntimeError, 'requires_12_gib_available_ram'):
                    gate.check_host(Path(tmp), report)
        self.assertEqual(report['host']['available_ram_bytes'], int(2.5*gate.GIB))
        self.assertEqual(report['host']['required_available_ram_bytes'], 12*gate.GIB)
        self.assertEqual(report['host']['free_disk_bytes'], 30*gate.GIB)
        self.assertEqual(report['steps'], [])
        docker.assert_not_called()

    def test_25_gib_main_receipt_is_preflight_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = root/'work'; work.mkdir()
            report_path = root/'report.json'
            argv = ['gate', '--reviewed-sha', self.SHA, '--work', str(work),
                    '--reference-wav', str(root/'reference.wav'),
                    '--reference-text-file', str(root/'reference.txt'),
                    '--cache-root', str(root), '--report', str(report_path),
                    '--reference-kind', 'human-recorded', '--reference-authorized']
            with mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(gate, 'verify_source'), \
                 mock.patch.object(gate.platform, 'system', return_value='Linux'), \
                 mock.patch.object(gate.platform, 'machine', return_value='x86_64'), \
                 mock.patch.object(gate.os, 'geteuid', return_value=0), \
                 mock.patch.object(gate.shutil, 'disk_usage', return_value=mock.Mock(free=30*gate.GIB)), \
                 mock.patch.object(Path, 'is_file', return_value=True), \
                 mock.patch.object(Path, 'read_text', return_value='MemAvailable: 2621440 kB\n'), \
                 mock.patch.object(gate, 'json_command') as docker, \
                 mock.patch.object(gate, 'command') as acquisition:
                with mock.patch.object(gate.sys, 'version_info', (3, 11, 0)):
                    self.assertEqual(gate.main(), 1)
            receipt = json.loads(report_path.read_text())
            self.assertEqual(receipt['failure'], 'requires_12_gib_available_ram')
            self.assertEqual(receipt['host']['available_ram_bytes'], int(2.5*gate.GIB))
            self.assertEqual(receipt['host']['required_available_ram_bytes'], 12*gate.GIB)
            self.assertEqual(receipt['host']['free_disk_bytes'], 30*gate.GIB)
            self.assertEqual(receipt['stage'], 'preflight')
            self.assertEqual(receipt['steps'], [])
            self.assertEqual(receipt['command_log'], [])
            self.assertEqual(receipt['evidence'], 'preflight_only')
            self.assertEqual(receipt['real_voice_acceptance'], 'not_tested')
            docker.assert_not_called(); acquisition.assert_not_called()

    def test_python_313_is_reported_and_refused_before_docker(self):
        report = {}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(gate.platform, 'system', return_value='Linux'), \
             mock.patch.object(gate.platform, 'machine', return_value='x86_64'), \
             mock.patch.object(gate.os, 'geteuid', return_value=0), \
             mock.patch.object(gate.platform, 'python_version', return_value='3.13.0'), \
             mock.patch.object(gate.shutil, 'disk_usage', return_value=mock.Mock(free=30*gate.GIB)), \
             mock.patch.object(Path, 'read_text', return_value='MemAvailable: 16777216 kB\n'), \
             mock.patch.object(gate, 'json_command') as docker:
            with mock.patch.object(gate.sys, 'version_info', (3, 13, 0)):
                with self.assertRaisesRegex(RuntimeError, 'requires_python_311'):
                    gate.check_host(Path(tmp), report)
        self.assertEqual(report['host']['python_version'], '3.13.0')
        docker.assert_not_called()

    def test_sufficient_budget_reaches_read_only_docker_checks(self):
        report = {}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(gate.platform, 'system', return_value='Linux'), \
             mock.patch.object(gate.platform, 'machine', return_value='x86_64'), \
             mock.patch.object(gate.os, 'geteuid', return_value=0), \
             mock.patch.object(gate.shutil, 'disk_usage', return_value=mock.Mock(free=30*gate.GIB)), \
             mock.patch.object(Path, 'is_file', return_value=True), \
             mock.patch.object(Path, 'read_text', return_value='MemAvailable: 12582912 kB\n'), \
             mock.patch.dict(gate.os.environ, {'DOCKER_HOST': 'unix:///var/run/docker.sock', 'DOCKER_CONTEXT': 'default'}), \
             mock.patch.object(gate, 'json_command', side_effect=[({'Version':'27.5.1'}, 0),
                 ({'CgroupDriver':'cgroupfs','CgroupVersion':'2','Architecture':'amd64','ID':'local'}, 0)]) as docker:
            with mock.patch.object(gate.sys, 'version_info', (3, 11, 0)):
                result = gate.check_host(Path(tmp), report)
        self.assertEqual(result['available_ram_bytes'], 12*gate.GIB)
        self.assertEqual(result['docker_version'], '27.5.1')
        self.assertEqual(docker.call_count, 2)

    def test_unknown_and_synthetic_reference_cannot_claim_audio(self):
        for kind in ('unknown', 'synthetic', 'human-recorded'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                work = root/'work'; work.mkdir()
                report_path = root/'report.json'
                argv = ['gate', '--reviewed-sha', self.SHA, '--work', str(work),
                        '--reference-wav', str(root/'reference.wav'),
                        '--reference-text-file', str(root/'reference.txt'),
                        '--cache-root', str(root), '--report', str(report_path),
                        '--reference-kind', kind]
                with mock.patch.object(sys, 'argv', argv), \
                     mock.patch.object(gate, 'verify_source'), \
                     mock.patch.object(gate, 'check_host', return_value={}), \
                     mock.patch.object(gate, 'command') as acquisition:
                    self.assertEqual(gate.main(), 1)
                receipt = json.loads(report_path.read_text())
                self.assertEqual(receipt['failure'], 'requires_authorized_human_recorded_reference')
                self.assertEqual(receipt['stage'], 'preflight')
                self.assertEqual(receipt['evidence'], 'preflight_only')
                self.assertEqual(receipt['qualification'], 'not_tested')
                self.assertEqual(receipt['real_voice_acceptance'], 'not_tested')
                self.assertEqual(receipt['steps'], [])
                self.assertEqual(receipt['command_log'], [])
                acquisition.assert_not_called()

    def test_source_admission_requires_exact_clean_checkout(self):
        def result(stdout):
            return mock.Mock(stdout=stdout)
        with mock.patch.object(gate.subprocess, 'run', side_effect=[result(self.SHA + '\n'), result('')]) as run:
            gate.verify_source(self.SHA)
            self.assertEqual(run.call_count, 2)
        with mock.patch.object(gate.subprocess, 'run', return_value=result('b' * 40 + '\n')):
            with self.assertRaisesRegex(RuntimeError, 'reviewed_sha_mismatch'):
                gate.verify_source(self.SHA)
        with mock.patch.object(gate.subprocess, 'run', side_effect=[result(self.SHA + '\n'), result(' M scripts/ci-qwen-real-audio.py\n')]):
            with self.assertRaisesRegex(RuntimeError, 'source_checkout_not_clean'):
                gate.verify_source(self.SHA)
        with mock.patch.object(gate.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'exact_reviewed_sha_required'):
                gate.verify_source('main')
            run.assert_not_called()

    def test_absent_snapshot_does_not_create_cache_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'cache'
            cache.mkdir(mode=0o500)
            descriptor = {'checkpoint': 'Qwen/Qwen3-TTS-12Hz-0.6B-Base', 'revision': 'a' * 40}
            snapshot = gate.snapshot_input(cache, descriptor)
            self.assertEqual(snapshot, cache/'Qwen3-TTS-12Hz-0.6B-Base'/'snapshots'/descriptor['revision'])
            self.assertFalse(snapshot.exists())
            self.assertEqual(list(cache.iterdir()), [])

    def test_command_pins_local_docker_transport(self):
        with mock.patch.dict(gate.os.environ, {'DOCKER_HOST': 'tcp://remote:2375',
                                               'DOCKER_CONTEXT': 'remote'}):
            script = 'import os;print(os.environ["DOCKER_HOST"]);print(os.environ["DOCKER_CONTEXT"])'
            output, _ = gate.command([sys.executable, '-c', script])
            self.assertEqual(output.splitlines(), ['unix:///var/run/docker.sock', 'default'])

    def test_failed_command_retains_bounded_private_cause_without_public_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate.COMMAND_LOG.clear()
            gate.COMMAND_LOG_DIR = Path(tmp)/'logs'
            secret = 'credential-secret reference-transcript /private/user/audio.wav'
            script = 'import sys;sys.stdout.write(sys.argv[1]*20000);sys.stderr.write("first cause\\n"+sys.argv[1]+"\\nlast cause\\n");sys.exit(7)'
            try:
                with self.assertRaisesRegex(RuntimeError, 'command_failed:python:7'):
                    gate.command([sys.executable, '-c', script, secret], phase='build_control', capture=False)
                receipt = gate.COMMAND_LOG[-1]
                public = json.dumps(receipt)
                self.assertNotIn(secret, public)
                self.assertNotIn('reference-transcript', public)
                self.assertEqual(receipt['exit_code'], 7)
                self.assertEqual(receipt['phase'], 'build_control')
                self.assertGreater(receipt['stdout_omitted_bytes'], 0)
                private = gate.COMMAND_LOG_DIR/(receipt['private_log_id']+'.stderr.log')
                self.assertIn(secret, private.read_text())
                self.assertIn('last cause', private.read_text())
                self.assertLessEqual(private.stat().st_size, 2*gate.LOG_EDGE_BYTES+64)
                self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            finally:
                gate.COMMAND_LOG_DIR = None

    def test_timeout_retains_first_cause_and_absolute_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate.COMMAND_LOG.clear()
            gate.COMMAND_LOG_DIR = Path(tmp)/'logs'
            try:
                script = 'import sys,time;sys.stderr.write("early failure cause\\n"+"x"*1000000);sys.stderr.flush();time.sleep(30)'
                with self.assertRaisesRegex(RuntimeError, 'command_timeout:python'):
                    gate.command([sys.executable, '-c', script], timeout=0.5, phase='public_up')
                receipt = gate.COMMAND_LOG[-1]
                self.assertEqual(receipt['outcome'], 'timeout')
                self.assertLess(receipt['seconds'], 3)
                private = gate.COMMAND_LOG_DIR/(receipt['private_log_id']+'.stderr.log')
                self.assertIn('early failure cause', private.read_text())
                self.assertGreater(receipt['stderr_omitted_bytes'], 0)
                self.assertLessEqual(private.stat().st_size, 2*gate.LOG_EDGE_BYTES+64)
            finally:
                gate.COMMAND_LOG_DIR = None

    def test_structured_cause_keeps_code_not_message(self):
        stdout, stderr = gate.BoundedOutput(), gate.BoundedOutput()
        stdout.add(b'{"phase":"generation_submit","code":"invalid_request","message":"credential-secret"}\n')
        self.assertEqual(gate.structured_cause(stdout, stderr),
                         {'phase':'generation_submit','code':'invalid_request'})
        stdout = gate.BoundedOutput()
        stdout.add(b'{"phase":"credential-secret","code":"reference-transcript","message":"/private/user/audio.wav"}\n')
        self.assertIsNone(gate.structured_cause(stdout, stderr))
        stderr.add(b'speech-box docker refused; daemon_helper_refusal:up:operation_failed:daemon_phase_timeout:ValueError:run:1:host=private\n')
        self.assertEqual(gate.structured_cause(stdout, stderr),
                         {'code':'daemon_helper_refusal','helper_phase':'up',
                          'helper_stage':'operation_failed','detail_code':'daemon_phase_timeout'})

    def test_repeats_require_new_jobs_and_expected_voice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []
            def fake_client(_py, _extracted, _client_root, *args, **_kwargs):
                calls.append(args)
                if args[:2] == ('voice','resolve'):
                    return {'version_ids':['version-1']}, 0.1
                output = Path(args[args.index('--output')+1])
                with wave.open(str(output), 'wb') as out:
                    out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
                    out.writeframes((500).to_bytes(2, 'little', signed=True)*24000)
                return {'state':'succeeded','job_id':'same-job','voice_version_id':'version-1'}, 0.1
            report = {'steps':[]}
            with mock.patch.object(gate, 'call_client', side_effect=fake_client):
                with self.assertRaisesRegex(RuntimeError, 'speech_job_not_fresh_ci-reference'):
                    gate.run_speech_repeats(report, 'python', root, root, root)
            speak = [a for a in calls if a[0] == 'speak']
            self.assertEqual(len(speak), 2)
            self.assertNotEqual(speak[0][speak[0].index('--text')+1],
                                speak[1][speak[1].index('--text')+1])

    def test_four_fresh_speech_submissions_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submissions = []
            def fake_client(_py, _extracted, _client_root, *args, **_kwargs):
                if args[:2] == ('voice', 'resolve'):
                    return {'version_ids':['version-'+args[2]]}, 0.1
                voice = args[args.index('--voice')+1]
                spoken = args[args.index('--text')+1]
                submissions.append((voice, spoken))
                output = Path(args[args.index('--output')+1])
                with wave.open(str(output), 'wb') as out:
                    out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
                    out.writeframes((500).to_bytes(2, 'little', signed=True)*24000)
                return {'state':'succeeded','job_id':'job-'+str(len(submissions)),
                        'voice_version_id':'version-'+voice}, 0.1
            report = {'steps':[]}
            with mock.patch.object(gate, 'call_client', side_effect=fake_client):
                gate.run_speech_repeats(report, 'python', root, root, root)
            self.assertEqual(len(report['steps']), 4)
            self.assertEqual(len({s['job_id'] for s in report['steps']}), 4)
            self.assertEqual(len({s['text_sha256'] for s in report['steps']}), 4)
            self.assertTrue(all(s['audio']['rms'] > 0 for s in report['steps']))

    def test_primary_failure_survives_stop_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            handle = Path(tmp) / 'handle.json'
            handle.write_text('{}')
            report = {'status': 'FAILED', 'failure': 'real_speech_failed_Ryan'}
            with mock.patch.object(gate, 'json_command', side_effect=RuntimeError('command_timeout:python')):
                gate.stop_owned(report, handle, Path(tmp), 'python')
            self.assertEqual(report['failure'], 'real_speech_failed_Ryan')
            self.assertEqual(report['stop']['failure'], 'command_timeout:python')
            self.assertEqual(report['status'], 'FAILED')

    def test_failed_stop_turns_prior_success_into_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            handle = Path(tmp) / 'handle.json'
            handle.write_text('{}')
            report = {'status': 'PASSED'}
            with mock.patch.object(gate, 'json_command', return_value=({'phase': 'retained'}, 1)):
                gate.stop_owned(report, handle, Path(tmp), 'python')
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual(report['failure'], 'owned_stop_failed')
            self.assertEqual(report['stop']['failure'], 'owned_stop_receipt_missing')

    def test_main_refuses_unreviewed_source_before_host_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / 'work'
            work.mkdir()
            report = root / 'report.json'
            argv = ['gate', '--reviewed-sha', 'main', '--work', str(work),
                    '--reference-wav', str(root/'missing.wav'),
                    '--reference-text-file', str(root/'missing.txt'),
                    '--cache-root', str(root), '--report', str(report)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(gate, 'check_host') as host:
                self.assertEqual(gate.main(), 1)
                host.assert_not_called()
            receipt = json.loads(report.read_text())
            self.assertEqual(receipt['failure'], 'exact_reviewed_sha_required')
            self.assertEqual(receipt['status'], 'FAILED')
