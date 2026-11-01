"""Smoke the extracted client and resident entrypoint without starting a service."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from service.package_install import archive


ROOT = Path(__file__).resolve().parents[2]
ADDED = ('clients/config.py', 'clients/workflow.py', 'worker/qwen_resident.py',
         'docs/SERVICE-VOICE-CLIENT.md')


class HeadlessClientResidentArchiveTests(unittest.TestCase):
    def test_extracted_client_help_and_resident_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            path = temporary / 'speech-box.zip'
            archive(ROOT, path)
            with zipfile.ZipFile(path) as bundle:
                for name in ADDED:
                    self.assertEqual(bundle.read('speech-box/' + name), (ROOT / name).read_bytes())
                bundle.extractall(temporary / 'extracted')
            package = temporary / 'extracted' / 'speech-box'
            self.assertIn('Named voice service client',
                          (package / 'docs/SERVICE-VOICE-CLIENT.md').read_text())

            # -I excludes the checkout and PYTHONPATH; assert every local import
            # resolves to extracted bytes before entering the CLI argument parser.
            imports = """
import importlib, pathlib, sys
package = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(package))
for name in ('clients.config', 'clients.workflow', 'clients.cli',
             'worker.qwen_resident', 'worker.runtime_supervisor'):
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).is_relative_to(package), name
print('extracted client and resident imports')
"""
            result = subprocess.run([sys.executable, '-I', '-B', '-c', imports, str(package)],
                                    cwd=temporary, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'extracted client and resident imports')

            isolated_environment = {'PATH': os.defpath, 'PYTHONPATH': str(package)}
            help_result = subprocess.run([sys.executable, '-B', '-m', 'clients.cli', '--help'],
                                         cwd=temporary, env=isolated_environment,
                                         capture_output=True, text=True, timeout=10)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn('enroll', help_result.stdout)
            self.assertIn('speak', help_result.stdout)

            # A refused supervisor gate proves __main__ reached its real entrypoint
            # while remaining inert: it stops before opening a control socket or model.
            gate_read, gate_write = os.pipe()
            try:
                os.write(gate_write, b'0')
                os.close(gate_write)
                resident = subprocess.run([sys.executable, '-B', '-m',
                                           'worker.qwen_resident', '-1', str(gate_read)],
                                          cwd=temporary, env=isolated_environment,
                                          pass_fds=(gate_read,), capture_output=True,
                                          text=True, timeout=10)
            finally:
                os.close(gate_read)
            self.assertNotEqual(resident.returncode, 0)
            self.assertIn('resident_cgroup_assignment_required', resident.stderr)


if __name__ == '__main__':
    unittest.main()
