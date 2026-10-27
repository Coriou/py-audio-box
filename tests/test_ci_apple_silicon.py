"""Stdlib-only checks; never import host ML dependencies."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch
from contextlib import asynccontextmanager, ExitStack
import json
import os
import types

SPEC = importlib.util.spec_from_file_location('apple_ci', Path(__file__).resolve().parents[1] / 'scripts/ci-apple-silicon.py')
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class AppleCITests(unittest.TestCase):
    def test_native_12hz_dependency_check_does_not_require_sox(self):
        class Tensor:
            device=types.SimpleNamespace(type='cpu')
            def __add__(self, other): return self
            def tolist(self): return [2.,3.]
        torch=types.SimpleNamespace(tensor=lambda *a,**kw:Tensor(),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(
                is_built=lambda:True,is_available=lambda:True)))
        qwen=types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=lambda:None))
        with patch.object(ci,'validate_platform'),patch.object(ci.shutil,'which',return_value=None), \
                patch.object(ci,'sox_version') as version,patch.object(ci.metadata,'version',side_effect=ci.EXPECTED.get), \
                patch.object(ci.importlib,'import_module'),patch.object(ci,'host_resources',return_value={}), \
                patch.dict(ci.sys.modules,{'torch':torch,'qwen_tts':qwen}):
            report=ci.deps_check()
        version.assert_not_called()
        self.assertIsNone(report['sox_executable'])
        self.assertIsNone(report['sox_version'])
        self.assertFalse(report['sox_required_for_native_12hz'])
        self.assertEqual(report['cpu_tensor'],'passed')

    def test_sox_version_banner_and_homebrew_fallback(self):
        with patch.object(ci.subprocess, 'check_output', return_value='sox: SoX v14.4.2\n') as run:
            self.assertEqual(ci.sox_version('/bin/sox'), 'sox: SoX v14.4.2')
            run.assert_called_once_with(['/bin/sox', '--version'], timeout=10, text=True)
        with patch.object(ci.shutil, 'which', return_value='/bin/brew'), patch.object(
                ci.subprocess, 'check_output', side_effect=['sox: SoX v', 'sox 14.4.2_6\n']) as run:
            self.assertEqual(ci.sox_version('/bin/sox'), 'Homebrew package: sox 14.4.2_6')
            self.assertEqual(run.call_args.args[0], ['/bin/brew', 'list', '--versions', 'sox'])
        for failure in ('', subprocess.CalledProcessError(1, 'brew'), subprocess.TimeoutExpired('brew', 10)):
            with self.subTest(failure=failure), patch.object(ci.shutil, 'which', return_value='/bin/brew'), patch.object(
                    ci.subprocess, 'check_output', side_effect=['sox: SoX v', failure]):
                self.assertTrue(ci.sox_version('/bin/sox').startswith('unknown'))

    def test_missing_client_dependency_fails(self):
        with patch.object(ci.importlib, 'import_module', side_effect=ModuleNotFoundError('fastapi')):
            with self.assertRaisesRegex(ModuleNotFoundError, 'fastapi'):
                ci.client_smoke()

    def test_owned_child_timeout_terminates_kills_and_reaps(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('child', 1),
                                 subprocess.TimeoutExpired('child', 5), -9]
        with patch.object(ci.subprocess, 'Popen', return_value=child), patch.object(ci.os, 'killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                ci.owned_run(['child'], env={}, cwd='.', timeout=1)
        child.terminate.assert_called_once_with()
        kill.assert_called_once_with(123, ci.signal.SIGKILL)
        self.assertEqual(child.wait.call_count, 3)

    def test_failed_install_report_retains_not_run(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'clients.json'
            with patch.object(ci, 'validate_platform'), patch.object(ci, 'owned_run', side_effect=RuntimeError('install failed')):
                with self.assertRaisesRegex(RuntimeError, 'install failed'):
                    ci.clients_check(path)
            report = json.loads(path.read_text())
            self.assertEqual(report['cli_install'], 'failed')
            for stage in ('mcp_install', 'protocol', 'fixture_calls'):
                self.assertEqual(report[stage], 'not_run')
            self.assertFalse(report['runtime_qualified'])

    def test_real_smoke_stage_failure_reporting(self):
        # Inject at actual CLI/MCP boundaries, run the real smoke control flow
        # and parent merger. No installation or functional transport evidence.
        for failure in ('startup', 'cli', 'cli_timeout', 'mcp', 'mcp_timeout', None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                path = Path(tmp) / 'clients.json'
                expected = {'operations': []}
                child = Mock(pid=123)
                child.poll.return_value = 1 if failure == 'startup' else None
                response = Mock()
                response.json.return_value = expected
                http = Mock()
                http.__enter__ = Mock(return_value=http)
                http.__exit__ = Mock(return_value=False)
                http.get.return_value = response
                session = Mock()
                session.initialize = AsyncMock(side_effect=(
                    RuntimeError('MCP injected') if failure == 'mcp' else
                    TimeoutError('MCP timeout') if failure == 'mcp_timeout' else None))
                session.list_tools = AsyncMock(return_value=types.SimpleNamespace(
                    tools=[types.SimpleNamespace(name='speech_discover')]))
                session.call_tool = AsyncMock(return_value=types.SimpleNamespace(
                    isError=False, structuredContent=expected))
                @asynccontextmanager
                async def context(*args, **kwargs):
                    yield session
                @asynccontextmanager
                async def streams(*args, **kwargs):
                    yield (None, None)
                modules = {'mcp': types.SimpleNamespace(ClientSession=context,
                    StdioServerParameters=Mock()),
                    'mcp.client': types.ModuleType('mcp.client'),
                    'mcp.client.stdio': types.SimpleNamespace(stdio_client=streams),
                    'httpx': types.SimpleNamespace(Client=Mock(return_value=http), HTTPError=OSError)}
                stack.enter_context(patch.dict(ci.sys.modules, modules))
                stack.enter_context(patch.object(ci, 'require_client_dependencies'))
                stack.enter_context(patch.object(ci, 'validate_platform'))
                stack.enter_context(patch.object(ci.platform, 'platform', return_value='test-host'))
                stack.enter_context(patch.object(ci.subprocess, 'Popen', return_value=child))
                reap = stack.enter_context(patch.object(ci, 'reap'))
                cli = stack.enter_context(patch.object(ci.subprocess, 'run'))
                cli.return_value.stdout = json.dumps({} if failure == 'cli' else expected)
                if failure == 'cli_timeout':
                    cli.side_effect = subprocess.TimeoutExpired('cli', 15)
                stack.enter_context(patch.object(ci.subprocess, 'check_output', return_value='{}'))
                def run(argv, *, env, cwd, timeout):
                    if 'tests.test_ci_apple_silicon.MacClientSmoke' in argv:
                        self.assertEqual(timeout, 60)
                        with patch.dict(os.environ, env):
                            ci.client_smoke()
                stack.enter_context(patch.object(ci, 'owned_run', side_effect=run))
                if failure:
                    with self.assertRaises(Exception):
                        ci.clients_check(path)
                else:
                    ci.clients_check(path)
                report = json.loads(path.read_text())
                fixture_failed = failure in ('startup', 'cli', 'cli_timeout')
                self.assertEqual(report['fixture_calls'], 'failed' if fixture_failed else 'passed')
                self.assertEqual(report['protocol'], 'not_run' if fixture_failed else
                                 'failed' if failure else 'passed')
                self.assertEqual(report['status'], 'failed' if failure else 'passed')
                for stage in ('cli_install', 'mcp_install', 'cli_help', 'voice_clients'):
                    self.assertEqual(report[stage], 'passed')  # simulated only
                if fixture_failed:
                    session.initialize.assert_not_called()
                reap.assert_called_once_with(child)
                self.assertFalse(report['runtime_qualified'])

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


class MacClientSmoke(unittest.TestCase):
    """Explicit hosted selection; importing this module requires only stdlib."""
    @unittest.skipUnless(__import__('os').environ.get('APPLE_CLIENT_SMOKE') == '1',
                         'selected only by the Mac client job')
    def test_real_cli_and_mcp_stdio(self):
        self.assertEqual(ci.client_smoke(),
                         {'protocol': 'passed', 'fixture_calls': 'passed'})


if __name__ == '__main__': unittest.main()
