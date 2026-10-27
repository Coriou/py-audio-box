"""Stdlib-only checks; never import host ML dependencies."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('apple_ci', Path(__file__).resolve().parents[1] / 'scripts/ci-apple-silicon.py')
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class AppleCITests(unittest.TestCase):
    def test_exact_dependency_versions(self):
        ci.validate_versions(dict(ci.EXPECTED))
        for version in ('2.10.0+cpu', '2.9.0', None):
            with self.assertRaises(RuntimeError):
                ci.validate_versions(dict(ci.EXPECTED, torch=version))

    def test_platform_and_python(self):
        with patch.object(ci.platform, 'system', return_value='Darwin'), patch.object(ci.platform, 'machine', return_value='arm64'), patch.object(ci.sys, 'version_info', (3, 11)), patch.object(ci.platform, 'python_implementation', return_value='CPython'):
            ci.validate_platform()
            with patch.object(ci.platform, 'machine', return_value='x86_64'):
                with self.assertRaises(RuntimeError): ci.validate_platform()
            with patch.object(ci.sys, 'version_info', (3, 12)):
                with self.assertRaises(RuntimeError): ci.validate_platform()

    def test_bounded_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'report.json'
            ci.write_report(path, {'ok': True})
            self.assertLess(path.stat().st_size, ci.MAX_REPORT_BYTES)
            with patch.object(ci, 'MAX_REPORT_BYTES', 10):
                with self.assertRaises(ValueError): ci.write_report(path, {'long': 'x'*20})
            with self.assertRaises(ValueError): ci.write_report(Path(tmp)/'report.txt', {})
            link = Path(tmp)/'link.json'
            link.symlink_to(path)
            with self.assertRaises(ValueError): ci.write_report(link, {})

    def test_probe_deadline_and_reap(self):
        child = Mock(pid=123)
        child.wait.side_effect = [subprocess.TimeoutExpired('probe', 30), -9]
        with patch.object(ci.subprocess, 'Popen', return_value=child) as spawn, patch.object(ci.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'reaped'): ci.bounded_probe('report.json')
        kill.assert_called_once_with(123, ci.signal.SIGKILL)
        self.assertEqual(child.wait.call_count, 2)
        self.assertEqual(spawn.call_args.kwargs['env']['PYTORCH_ENABLE_MPS_FALLBACK'], '0')
        self.assertNotIn('PYTORCH_MPS_HIGH_WATERMARK_RATIO', spawn.call_args.kwargs['env'])


if __name__ == '__main__': unittest.main()
