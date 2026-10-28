"""Real bounded setup child and source archive contracts; never installs/models."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from service import native_cpu_bootstrap as b
from service import native_cpu_package as package


class CPUBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.root.chmod(0o700)
        for name in ('bootstrap-home','tmp'):(self.root/name).mkdir(mode=0o700)

    def test_real_owned_setup_child_success_deadline_and_reap(self):
        b.bounded([sys.executable,'-c','print("source-fixture")'],self.root,'success',5)
        self.assertEqual((self.root/'success.log').read_text().strip(),'source-fixture')
        self.assertFalse(b._OWNED)
        with self.assertRaises(TimeoutError):b.bounded([sys.executable,'-c','import time; time.sleep(30)'],self.root,'deadline',.15)
        self.assertFalse(b._OWNED)

    def test_completed_setup_child_still_obeys_log_limit(self):
        with self.assertRaisesRegex(ValueError,'log limit'):
            b.bounded([sys.executable,'-c','import sys; sys.stdout.write("x"*(1024**2+1))'],self.root,'oversize',5)
        self.assertFalse(b._OWNED)

    def test_create_only_setup_and_platform_errors_never_launch_pip(self):
        with patch.object(b,'host'),patch.object(b,'bounded') as command:
            with self.assertRaisesRegex(ValueError,'exists'):b.setup(self.root,download_model=True)
            command.assert_not_called()
        with patch.object(b.platform,'machine',return_value='x86_64'),self.assertRaisesRegex(ValueError,'Rosetta'):b.host()
        with patch.object(b,'setup') as setup,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):b.main(['setup','--root',str(self.root/'new')])
            setup.assert_not_called()

    def test_bootstrap_environment_does_not_inherit_credentials_or_proxies(self):
        with patch.dict('os.environ',dict(HF_TOKEN='inert-fixture',HTTPS_PROXY='inert-fixture',SPEECH_CREDENTIAL='inert-fixture')):
            env=b.environment(self.root)
        self.assertNotIn('HF_TOKEN',env);self.assertNotIn('HTTPS_PROXY',env);self.assertNotIn('SPEECH_CREDENTIAL',env)
        self.assertEqual(env['PIP_CONFIG_FILE'],b.os.devnull)

    def test_preflight_cli_report_is_not_an_error_exit_payload(self):
        from service import native_cpu_setup as setup
        with patch.object(setup,'preflight',return_value={'status':'passed'}):
            self.assertEqual(setup.main(['preflight','--output',str(self.root/'not-created')]),0)

    def test_finish_setup_cannot_acquire_before_actual_guarded_preflight(self):
        import hashlib
        from service import native_cpu_setup as setup
        from worker.transport import canonical
        names=('requirements/kokoro.lock','requirements/client.lock','service/requirements.lock')
        marker=dict(version=1,root=str(self.root),source=str(setup.ROOT),download_model=True,port=8788,
            locks={name:hashlib.sha256((setup.ROOT/name).read_bytes()).hexdigest() for name in names})
        (self.root/'bootstrap.json').write_bytes(canonical(marker));(self.root/'bootstrap.json').chmod(0o600)
        with patch.object(setup.sys,'prefix',str(self.root/'venv')),patch.object(setup,'preflight',side_effect=RuntimeError('guarded preflight fixture failure')),patch('service.provisioning.provision') as provision:
            with self.assertRaisesRegex(RuntimeError,'guarded preflight fixture failure'):setup.finish_setup(self.root)
            provision.assert_not_called()
        self.assertFalse((self.root/'models').exists());self.assertFalse((self.root/'install.json').exists())

    def test_separate_cpu_archive_is_closed_deterministic_and_stdlib_help_works(self):
        first=self.root/'first.zip';second=self.root/'second.zip'
        self.assertEqual(package.archive(first),package.archive(second));package.verify_archive(first)
        with zipfile.ZipFile(first) as source:
            self.assertFalse(any(n.endswith('.whl') or 'credential.json' in n or '.env' in n for n in source.namelist()))
            manifest=json.loads(source.read('native-worker/native-cpu-dist-manifest.json'))
            self.assertEqual((manifest['variant'],manifest['qualification']),('native-kokoro-cpu-v1','unqualified'))
            for name in ('requirements/kokoro.lock','requirements/client.lock','service/requirements.lock','worker/native_cpu_child.py'):
                self.assertIn('native-worker/'+name,source.namelist())
            source.extractall(self.root/'extracted')  # already exact verified test-owned archive
        result=subprocess.run([sys.executable,'-I',str(self.root/'extracted/native-worker/native-worker.py'),'--help'],capture_output=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        self.assertIn(b'qualification-session',result.stdout)


if __name__=='__main__':unittest.main()
