"""Source/fake-probe evidence only."""
import subprocess
import sys
import unittest
from unittest.mock import patch
from worker.hardware import DeviceFamily as F, Host, discover, host_metadata, probe


class HardwareTests(unittest.TestCase):
    def test_stdlib_host_source(self):
        with patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='aarch64'):
            self.assertEqual(host_metadata('docker'), Host('linux', 'aarch64', 'docker'))

    def test_discovery_never_probes(self):
        with patch('worker.hardware.probe', side_effect=AssertionError('implicit probe')):
            self.assertEqual(discover(Host('linux', 'aarch64', 'docker')).devices, ())

    def test_explicit_probe_fail_closed(self):
        inv = discover(Host('linux', 'x86_64', 'native'))
        calls = []
        def provider(family, index):
            calls.append((family, index))
            return True
        self.assertFalse(probe(inv, F.CUDA, 0).available)
        self.assertTrue(probe(inv, F.CUDA, 0, provider).available)
        self.assertEqual(calls, [(F.CUDA, 0)])
        self.assertFalse(probe(inv, F.CUDA, 0, lambda *_: False).available)
        self.assertEqual(probe(inv, F.CUDA, 0, lambda *_: 'yes').reason, 'probe_invalid')
        def broken(*_):
            raise RuntimeError('private detail')
        self.assertEqual(probe(inv, F.CUDA, 0, broken).reason, 'probe_failed')
        self.assertEqual(inv.devices, ())
        self.assertFalse(probe(inv, F.MPS, 0, provider).available)
        self.assertEqual(len(calls), 1)

    def test_fresh_import_and_discovery_forbid_ml_and_network(self):
        code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'transformers', 'qwen_tts', 'onnxruntime', 'socket', 'lib'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from service.profiles import SelectionPolicy
from worker.hardware import discover, host_metadata
assert discover(host_metadata()).devices == ()
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
