"""Source/finite-child checks only; no Docker, ML or installed qualification."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('cpu_public_recipe', ROOT/'scripts/cpu-kokoro-export.py')
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


class FiniteCommands(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.commands = recipe.Commands(self.output, 'unused-test-context')

    def command(self, source):
        return [sys.executable, '-I', '-B', '-c', source]

    def assert_reaped(self):
        pid = json.loads((self.output/'01-command.pid.json').read_bytes())['pid']
        receipt = json.loads((self.output/'01-command.wait.json').read_bytes())
        self.assertEqual(receipt['pid'], pid)
        self.assertIs(type(receipt['returncode']), int)
        self.assertIs(receipt['reaped'], True)
        # Independently corroborate that this process is no longer waitable by
        # the parent. EOF/attach completion alone would not satisfy this check.
        with self.assertRaises(ChildProcessError): os.waitpid(pid, os.WNOHANG)
        self.assertFalse((self.output/'01-command.retained.json').exists())
        return receipt

    def test_success_retains_complete_log_and_actual_wait(self):
        result = self.commands.run(self.command("import sys;print('out');print('err',file=sys.stderr)"))
        self.assertEqual(set(result.splitlines()), {'out','err'})
        self.assertEqual((self.output/'01-command.log').read_text(), result)
        self.assertEqual(self.assert_reaped()['returncode'], 0)

    def test_nonzero_is_not_success_and_still_reaped(self):
        with self.assertRaisesRegex(ValueError, 'command_failed'):
            self.commands.run(self.command("import sys;print('failed',flush=True);sys.exit(7)"))
        self.assertEqual(self.assert_reaped()['returncode'], 7)
        self.assertIn('failed', (self.output/'01-command.log').read_text())

    @unittest.skipUnless(hasattr(os, 'killpg'), 'POSIX process groups required')
    def test_deadline_terminates_exact_group_then_records_wait(self):
        actual_killpg = os.killpg
        with patch.object(recipe.os, 'killpg', wraps=actual_killpg) as terminate:
            with self.assertRaisesRegex(TimeoutError, 'command_timeout'):
                self.commands.run(self.command('import time;time.sleep(60)'), timeout=.25)
        receipt = self.assert_reaped()
        self.assertEqual(receipt['returncode'], -signal.SIGTERM)
        terminate.assert_called_once_with(receipt['pid'], signal.SIGTERM)

    @unittest.skipUnless(hasattr(os, 'killpg'), 'POSIX process groups required')
    def test_output_flood_refused_without_unbounded_retained_log(self):
        with self.assertRaisesRegex(ValueError, 'command_output_limit'):
            self.commands.run(self.command("import os,time;os.write(1,b'x'*131072);time.sleep(60)"), maximum=1024, timeout=5)
        self.assertLessEqual((self.output/'01-command.log').stat().st_size, 1024)
        self.assert_reaped()

    def test_pending_wait_never_releases_owner_before_later_actual_exit(self):
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        stream = os.fdopen(read_fd, 'rb', buffering=0)
        process = Mock(pid=7654321, stdout=stream)
        process.poll.return_value = None
        expired = subprocess.TimeoutExpired(['finite-test-child'], 15)
        def actual_exit(*args, **kwargs):
            self.assertTrue((self.output/'01-command.retained.json').exists())
            self.assertFalse((self.output/'01-command.wait.json').exists())
            return 7
        steps = iter([expired, expired, KeyboardInterrupt(), actual_exit])
        def wait(*args, **kwargs):
            step = next(steps)
            if isinstance(step, BaseException): raise step
            return step(*args, **kwargs)
        process.wait.side_effect = wait
        with patch.object(recipe.subprocess, 'Popen', return_value=process), patch.object(recipe.os, 'killpg') as terminate, patch.object(recipe.sys, 'stderr'):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.commands.run(['finite-test-child'])
        self.assertEqual(process.wait.call_count, 4)
        terminate.assert_called_once_with(process.pid, signal.SIGTERM)
        receipt = json.loads((self.output/'01-command.wait.json').read_bytes())
        self.assertEqual(receipt, dict(pid=process.pid, returncode=7, reaped=True))
        self.assertTrue(stream.closed)

    def test_docker_override_environment_is_removed(self):
        with patch.dict(os.environ, {'DOCKER_HOST':'ssh://foreign', 'DOCKER_CONTEXT':'foreign', 'BUILDX_BUILDER':'foreign', 'BUILDKIT_HOST':'tcp://foreign'}):
            commands = recipe.Commands(self.output, 'explicit')
        self.assertFalse(any(key.startswith('DOCKER_') for key in commands.env))
        self.assertNotIn('BUILDX_BUILDER', commands.env)
        self.assertNotIn('BUILDKIT_HOST', commands.env)

    def test_pid_receipt_failure_still_terminates_and_reaps_child(self):
        real_save = recipe.save
        def save(path, value):
            if path.name.endswith('.pid.json'): raise OSError('disk_full')
            return real_save(path, value)
        with patch.object(recipe, 'save', side_effect=save):
            with self.assertRaisesRegex(OSError, 'disk_full'):
                self.commands.run(self.command('import time;time.sleep(60)'))
        receipt = json.loads((self.output/'01-command.wait.json').read_bytes())
        self.assertTrue(receipt['reaped'])
        with self.assertRaises(ChildProcessError): os.waitpid(receipt['pid'], os.WNOHANG)

    def test_attach_json_does_not_override_nonzero_container_wait(self):
        identity, image = 'a'*64, 'sha256:'+'b'*64
        inspected = dict(Image=image, State=dict(Running=False, OOMKilled=False, ExitCode=1))
        with patch.object(self.commands, 'docker', side_effect=[identity, '{"looks":"successful"}', '1', json.dumps([inspected])]):
            with self.assertRaisesRegex(ValueError, 'terminal_failure'):
                self.commands.container('probe', image, [], ['probe'])
        self.assertEqual(json.loads((self.output/'probe.id.json').read_bytes())['id'], identity)
        self.assertEqual(json.loads((self.output/'probe.inspect.json').read_bytes()), inspected)

    def test_zero_exit_with_oom_is_refused(self):
        identity, image = 'a'*64, 'sha256:'+'b'*64
        inspected = dict(Image=image, State=dict(Running=False, OOMKilled=True, ExitCode=0))
        with patch.object(self.commands, 'docker', side_effect=[identity, '{}', '0', json.dumps([inspected])]):
            with self.assertRaisesRegex(ValueError, 'terminal_failure'):
                self.commands.container('probe', image, [], ['probe'])


class PrivateSourceCapability(unittest.TestCase):
    def test_readonly_capability_exists_before_owner_transfer(self):
        import fcntl
        probe_spec = importlib.util.spec_from_file_location('cpu_public_probe', ROOT/'scripts/cpu-kokoro-export-probe.py')
        probe = importlib.util.module_from_spec(probe_spec)
        probe_spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'source.wav'
            opened = []
            actual_open = os.open
            def retain_open(*args, **kwargs):
                fd = actual_open(*args, **kwargs)
                opened.append(fd)
                return fd
            def owner_transfer(target, uid, gid):
                self.assertEqual((uid,gid), (10001,10001))
                self.assertEqual(target.stat().st_mode & 0o777, 0o600)
                # At the permission boundary, an actual readonly descriptor
                # must already exist. No fixture media or ML is involved.
                self.assertEqual(len(opened), 1)
                self.assertEqual(fcntl.fcntl(opened[0], fcntl.F_GETFL) & os.O_ACCMODE, os.O_RDONLY)
                target.chmod(0)
            with patch.object(probe.os, 'open', side_effect=retain_open), patch.object(probe.os, 'chown', side_effect=owner_transfer):
                fd = probe.private_source(path, b'bounded-source-bytes')
            try:
                self.assertEqual(os.read(fd, 100), b'bounded-source-bytes')
                with self.assertRaises(OSError): os.write(fd, b'mutation')
            finally:
                os.close(fd)
                path.chmod(0o600)


class FrozenContext(unittest.TestCase):
    def setUp(self):
        expected = json.loads((ROOT/'requirements/tts/kokoro-fr.expected.json').read_bytes())
        self.context, self.speech, self.export = recipe.context(expected)

    def test_frozen_candidate_passes_existing_oracles_without_qualification(self):
        from contracts import validate as oracle
        from worker.runtime_protocol import execution
        oracle.validate('worker_plan', self.context['plan'])
        oracle.frozen_plan(self.context['plan'], self.context['bundles'])
        spec = execution(self.context)
        self.assertEqual(spec.device, 'cpu')
        self.assertEqual(spec.preset, 'ff_siwis')
        for profile in (self.speech, self.export):
            oracle.validate('profile', profile)
            self.assertEqual(profile['support'], 'unqualified')
            self.assertEqual(profile['readiness'], 'not_ready')
            self.assertEqual(profile['qualification_receipt'], '')
        self.assertEqual(self.context['plan']['resolved']['profile'], oracle.profile_snapshot(self.speech))

    def test_descriptor_cannot_be_rebound_to_another_manifest(self):
        from worker.runtime_protocol import execution
        mutated = copy.deepcopy(self.context)
        mutated['execution']['descriptor']['manifest_sha256'] = '0'*64
        with self.assertRaises(ValueError): execution(mutated)

    def test_execution_limits_cannot_silently_differ_from_frozen_profile(self):
        from worker.runtime_protocol import execution
        mutated = copy.deepcopy(self.context)
        mutated['execution']['descriptor']['threads'] += 1
        with self.assertRaises(ValueError): execution(mutated)

    def test_request_settings_hash_is_checked_by_canonical_oracle(self):
        from contracts import validate as oracle
        mutated = copy.deepcopy(self.context)
        mutated['plan']['resolved']['settings_sha256'] = '0'*64
        with self.assertRaises(ValueError): oracle.frozen_plan(mutated['plan'], mutated['bundles'])


if __name__ == '__main__': unittest.main()
