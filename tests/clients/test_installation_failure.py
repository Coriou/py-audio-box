"""Client discovery remains safe and reports the local failing boundary."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from clients.cli import main
from clients.config import InstallationFailure, installation


class InstallationFailureTests(unittest.TestCase):
    def test_root_caller_rejects_user_owned_ancestor_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / 'qwen-svc' / 'root'
            root.mkdir(parents=True)
            (root / 'client.json').write_text('{}')
            (root / 'control.key').write_text('{}')
            actual_lstat = Path.lstat
            service_ancestor = Path(directory).resolve() / 'qwen-svc'
            seen = []

            def recorded_layout(path):
                seen.append(path)
                info = actual_lstat(path)
                if path in root.parents or path == root:
                    values = list(info)
                    values[stat.ST_UID] = 1000 if path == service_ancestor else 0
                    values[stat.ST_MODE] = stat.S_IFDIR | (0o755 if path == service_ancestor else 0o700)
                    return os.stat_result(values)
                return info

            with patch('clients.config.os.getuid', return_value=0), \
                    patch('clients.config.Path.lstat', autospec=True, side_effect=recorded_layout), \
                    patch('clients.cli.HTTP', side_effect=AssertionError('HTTP must not start')):
                with self.assertRaises(InstallationFailure) as caught:
                    installation(root)
                self.assertEqual(caught.exception.body['field'], 'installation')
                self.assertIn('ancestor owner', caught.exception.body['message'])
                self.assertEqual(seen[-1], service_ancestor)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(['--installation', str(root), 'discover']), 2)
                body = json.loads(output.getvalue())
                self.assertEqual(body['field'], 'installation')
                self.assertIn('ancestor owner', body['message'])
                self.assertNotIn(str(root), output.getvalue())

    def test_invalid_client_config_reports_config_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / 'root'
            root.mkdir(mode=0o700)
            (root / 'client.json').write_text('{}')
            (root / 'control.key').write_text('{}')
            for name in ('client.json', 'control.key'):
                (root / name).chmod(0o600)
            with self.assertRaises(InstallationFailure) as caught:
                installation(root)
            self.assertEqual(caught.exception.body['field'], 'config')
