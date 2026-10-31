import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from service.qwen_local_config import create, discover


class QwenLocalConfigTests(unittest.TestCase):
    def test_create_only_auth_and_discovery(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp' if os.uname().sysname == 'Darwin' else '/tmp') as temp:
            root=Path(temp)/'installation'
            created=create(root,port=18521)
            self.assertEqual(created,root)
            config=discover(root)
            self.assertEqual(config['endpoint'],'http://127.0.0.1:18521')
            self.assertEqual(config['credential_file'],str(root/'control.key'))
            credential=json.loads((root/'control.key').read_text())
            self.assertEqual(len(credential['credentials'][0]['token']),64)
            self.assertEqual((root/'control.key').stat().st_mode & 0o777,0o600)
            self.assertEqual((root/'client.json').stat().st_mode & 0o777,0o600)
            with self.assertRaises(FileExistsError): create(root)
            with patch.dict(os.environ,{'SPEECH_BOX_QWEN_HOME':str(root)}):
                self.assertEqual(discover(),config)
            (root/'client.json').chmod(0o644)
            with self.assertRaises(ValueError): discover(root)


if __name__=='__main__': unittest.main()
