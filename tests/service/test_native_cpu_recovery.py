"""Real tiny provision transactions under a mocked pending-install identity; no weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from service import native_cpu_setup as setup
from service.provisioning import identity
from worker.engines.native_kokoro import JOB_SECONDS,REQUEST_SECONDS


class CPUSetupRecoveryTests(unittest.TestCase):
    def test_explicit_same_directory_retry_reuses_only_verified_completed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();root.chmod(0o700)
            data={'model.onnx':b'inert-source-model','voices.bin':b'inert-source-voices'}
            expected=dict(schema_version=1,backend='kokoro',language='en',checkpoint='fixture-kokoro',revision='fixture',
                license='fixture only',access='no qualification',versions={p:'fixture' for p in ('kokoro-onnx','onnxruntime','espeakng-loader','phonemizer')},
                files=[dict(path=n,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),origin='https://github.com/fixture/'+n,revision='fixture') for n,raw in data.items()])
            bundle=root/'models/bundles'/identity(expected);record=dict(inert_source_fixture=True)
            calls=[];fail=True
            def download(url,target,size,**kw):
                nonlocal fail
                name=url.rsplit('/',1)[-1];calls.append(name)
                self.assertTrue(0<kw['timeout']<=600);self.assertTrue(kw['public_github_assets'])
                if name=='voices.bin' and fail:
                    fail=False;target.write_bytes(b'partial');raise TimeoutError('inert acquisition failure')
                target.write_bytes(data[name])
            with (patch.object(setup,'read_installation',return_value=(root,record,expected,bundle)) as pending,
                    patch.object(setup.sys,'prefix',str(root/'venv')),patch.object(setup,'preflight') as preflight,
                    patch('service.provisioning.download',side_effect=download)):
                with self.assertRaisesRegex(TimeoutError,'acquisition failure'):
                    setup._complete_setup(root,download_seconds=120,resume=False)
                self.assertFalse((root/'install.json').exists())
                stage=root/'models/staging'/identity(expected)
                self.assertEqual((stage/'model.onnx').read_bytes(),data['model.onnx'])
                setup._complete_setup(root,download_seconds=600,resume=True)
                self.assertEqual(calls,['model.onnx','voices.bin','voices.bin'])
                self.assertEqual((bundle/'model.onnx').read_bytes(),data['model.onnx'])
                self.assertEqual(json.loads((root/'install.json').read_bytes()),record)
                self.assertEqual(preflight.call_count,1)
                self.assertTrue(all(call.kwargs=={'_pending':True} for call in pending.call_args_list))
            self.assertEqual((JOB_SECONDS,REQUEST_SECONDS),(60,30))
            # Release only this inert test's readonly fixture tree.
            for path in root.rglob('*'):
                if path.is_dir():path.chmod(0o700)
                else:path.chmod(0o600)


if __name__=='__main__':unittest.main()
