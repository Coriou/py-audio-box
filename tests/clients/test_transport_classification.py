"""Stopped loopback service and raw socket error classification at client boundaries."""
import ast
import asyncio
import contextlib
import errno
import io
import json
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from clients.cli import main
from clients.http import HTTP, network_os_error
from clients.config import InstallationFailure
from service.catalogue import Failure, problem


ROOT = Path(__file__).resolve().parents[2]


class TransportClassificationTests(unittest.TestCase):
    def test_unused_loopback_socket_cli_enroll_and_speak(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        endpoint = f'http://127.0.0.1:{port}'
        with tempfile.TemporaryDirectory() as directory:
            cases = (
                ('enroll', '--name', 'sample', '--asset-id', 'confirmed-asset',
                 '--reference-text', 'reference', '--audition-text', 'audition', '--wait-seconds', '2'),
                ('speak', '--voice', 'sample', '--text', 'hello', '--output',
                 str(Path(directory) / 'speech.wav'), '--wait-seconds', '2'),
            )
            for command in cases:
                with self.subTest(command=command[0]):
                    run = subprocess.run(
                        [sys.executable, '-m', 'clients.cli', '--url', endpoint,
                         '--credential', 'a' * 32, *command], cwd=ROOT,
                        capture_output=True, text=True, timeout=5)
                    self.assertEqual(run.returncode, 2, run.stderr)
                    body = json.loads(run.stdout)
                    self.assertEqual((body['state'], body['code']), ('read_unavailable', 'not_ready'))
                    self.assertNotIn('field', body)
                    self.assertFalse((Path(directory) / 'speech.wav').exists())

    def test_raw_socket_error_and_local_file_error_remain_distinct(self):
        with patch('clients.http._body', side_effect=OSError(errno.EHOSTUNREACH, 'unreachable')):
            with self.assertRaises(ConnectionError):
                HTTP('http://127.0.0.1:1').get('/v1/status')
            with self.assertRaisesRegex(ConnectionError, 'outcome unknown'):
                HTTP('http://127.0.0.1:1').post('/v1/operations/voice_prepare', {}, 'key')
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory).resolve() / 'reference.wav'
                source.write_bytes(b'RIFF' + b'\0' * 64)
                with self.assertRaises(Failure) as caught:
                    HTTP('http://127.0.0.1:1').upload_file(source)
                self.assertEqual(caught.exception.body['code'], 'not_ready')
                self.assertIn('unknown', caught.exception.body['message'])
        self.assertTrue(network_os_error(OSError(errno.ENETUNREACH, 'unreachable')))
        self.assertFalse(network_os_error(FileNotFoundError(errno.ENOENT, 'missing')))
        with patch('clients.cli.HTTP.get', side_effect=OSError(errno.ENETUNREACH, 'unreachable')):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(['--url', 'http://127.0.0.1:1', 'status'])
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(output.getvalue()),
                             {'transport_error': 'Speech HTTP transport unavailable.'})
        with patch('clients.cli.HTTP.get', side_effect=FileNotFoundError(errno.ENOENT, 'missing')):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(['--url', 'http://127.0.0.1:1', 'status'])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())['field'], 'output')

    def test_mcp_mapping_without_optional_sdk(self):
        # Compile only the handler so this parity check works in the existing client venv.
        source = ROOT / 'clients' / 'mcp.py'
        tree = ast.parse(source.read_text())
        handler = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == 'call_tool')
        handler.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[]))
        namespace = {
            'Failure': Failure, 'ConfigurationFailure': ValueError,
            'InstallationFailure': InstallationFailure,
            'ValidationError': TypeError, 'ContractError': TypeError,
            'HTTP': HTTP, 'network_os_error': network_os_error,
            'problem': problem, 'asyncio': asyncio,
            'result': lambda body, error=False: (body, error),
        }
        exec(compile(module, str(source), 'exec'), namespace)
        for error, code in ((OSError(errno.EHOSTUNREACH, 'unreachable'), 'not_ready'),
                            (FileNotFoundError(errno.ENOENT, 'missing'), 'invalid_request')):
            with self.subTest(error=type(error)), patch('clients.http.HTTP.get', side_effect=error):
                body, is_error = asyncio.run(namespace['call_tool']('speech_status', {}))
            self.assertTrue(is_error)
            self.assertEqual(body['code'], code)
