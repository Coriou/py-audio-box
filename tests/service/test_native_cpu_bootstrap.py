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

    def test_real_setup_interrupt_unwinds_its_separate_phase_group(self):
        import time
        inner="from pathlib import Path; import os,time; Path("+repr(str(self.root/'inner.pid'))+").write_text(str(os.getpid())); time.sleep(5)"
        code="from pathlib import Path; from service import native_cpu_bootstrap as b; import sys; root=Path("+repr(str(self.root))+ "); b._main=lambda args:b.bounded([sys.executable,'-c',"+repr(inner)+"],root,'fixture-phase',30); b.main(['setup','--root',str(root),'--download-model'])"
        process=subprocess.Popen([sys.executable,'-B','-c',code],cwd=package.common.ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        try:
            end=time.monotonic()+3
            while not (self.root/'inner.pid').exists() and time.monotonic()<end:time.sleep(.02)
            self.assertTrue((self.root/'inner.pid').exists())
            process.terminate();process.wait(timeout=5)
            receipt=json.loads((self.root/('bootstrap-cleanup-'+str(process.pid)+'.json')).read_bytes())
            self.assertEqual(receipt['active_owned_pids'],[])
            self.assertEqual(receipt['phases'][0]['pid'],int((self.root/'inner.pid').read_text()))
            self.assertTrue(receipt['phases'][0]['reaped'])
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=1)

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

    def test_preflight_separates_control_imports_from_permanent_backend_guard(self):
        from service import native_cpu_setup as setup
        from worker.native_child import SpawnRefused
        order=[]
        def backend():
            self.assertEqual(order,['control','guard']);order.append('backend');return {}
        with patch.object(setup,'_control_imports',side_effect=lambda:order.append('control')),patch('worker.native_child.install_spawn_guard',side_effect=lambda:order.append('guard')),patch('lib.tts_backends.kokoro_native.preflight',side_effect=backend),patch('worker.native_cpu_qualification.boot_identity'),patch.object(setup,'create_tls'),patch('worker.native_cpu_protocol.tls_context'),patch('subprocess.Popen',side_effect=SpawnRefused('fixture refusal')):
            setup._preflight_child(self.root/'guard-report.json')
        report=json.loads((self.root/'guard-report.json').read_bytes())
        self.assertEqual(order,['control','guard','backend'])
        self.assertTrue(report['backend_imports_guarded']);self.assertFalse(report['control_imports_guarded'])
        self.assertTrue(report['explicit_spawn_refused']);self.assertFalse(report['weights_loaded'])

    def test_real_control_import_then_unchanged_guard_in_fresh_interpreter(self):
        code='''
from service.native_cpu_setup import _control_imports
_control_imports()
import subprocess
cached_popen=subprocess.Popen
from worker.native_child import install_spawn_guard,SpawnRefused
install_spawn_guard()
for spawn in (subprocess.Popen,cached_popen):
    try:spawn(['never-execute-source-regression'])
    except SpawnRefused:pass
    else:raise AssertionError('permanent guard lost')
print('real-control-import-guard-refusal')
'''
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=package.common.ROOT,capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        self.assertEqual(result.stdout.strip(),b'real-control-import-guard-refusal')

    def test_model_entry_imports_exclude_control_clients(self):
        code="import sys; import worker.native_cpu_child; assert not {'mcp','clients.mcp','service.api'}.intersection(sys.modules)"
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=package.common.ROOT,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr.decode())

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

    def test_qwen_archive_excludes_cpu_installer_and_keeps_actual_shared_imports(self):
        archive=self.root/'qwen.zip';package.common.archive(archive)
        with zipfile.ZipFile(archive) as source:
            names=set(source.namelist())
            self.assertFalse(any(n.endswith('.lock') for n in names))
            for name in ('native-cpu-dist-manifest.json','scripts/native-kokoro.py','service/native_cpu_bootstrap.py',
                    'service/native_cpu_setup.py','service/native_cpu_install.py','service/native_cpu_package.py','docs/agent-interface/NATIVE-KOKORO.md'):
                self.assertNotIn('native-worker/'+name,names)
            source.extractall(self.root/'qwen-extracted')
        extracted=self.root/'qwen-extracted/native-worker'
        code='import sys; sys.path.insert(0,sys.argv[1]); import service.dispatcher; import worker.native_child; assert service.dispatcher.__file__.startswith(sys.argv[1]); assert "service.native_cpu_setup" not in sys.modules'
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(extracted)],capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr.decode())


if __name__=='__main__':unittest.main()
