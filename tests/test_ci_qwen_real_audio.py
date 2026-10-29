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
            with mock.patch.object(gate.subprocess, 'run', return_value=mock.Mock(returncode=0, stdout='ok')) as run:
                self.assertEqual(gate.command(['docker', 'version'])[0], 'ok')
            env = run.call_args.kwargs['env']
            self.assertEqual(env['DOCKER_HOST'], 'unix:///var/run/docker.sock')
            self.assertEqual(env['DOCKER_CONTEXT'], 'default')

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
