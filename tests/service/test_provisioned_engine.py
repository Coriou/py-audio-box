"""Provisioned bytes/pointer to reviewed engine; inert backend only."""
import dataclasses
import hashlib
import json
import unittest
from tests.service import test_provisioning as fixtures
from worker.engines.execution import Execution
from worker.engines.kokoro import execute
from worker.transport import Refused


class ProvisionedEngineTests(unittest.TestCase):
    setUp = fixtures.ProvisioningTests.setUp
    cleanup = fixtures.ProvisioningTests.cleanup
    downloader = fixtures.ProvisioningTests.downloader
    run_provision = fixtures.ProvisioningTests.run_provision

    def test_output_manifest_and_pointer_execute(self):
        bundle = self.run_provision()
        pointer = json.loads((self.root / 'active.json').read_bytes())
        manifest = json.loads((bundle / 'manifest.json').read_bytes())
        self.assertEqual(pointer['manifest_sha256'], hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest())
        spec = Execution(pointer['bundle'], pointer['manifest_sha256'], manifest['checkpoint'], manifest['revision'],
                         'af_heart', manifest['language'], 1, 1024, 100, 244, 1024*1024)
        calls = []
        class Backend:
            def __init__(self, root, **kwargs): calls.append((root, kwargs))
            def synthesise(self, text, **kwargs):
                calls.append((text, kwargs))
                from lib.tts_backends.contract import AudioResult
                return AudioResult([0., .25], 24000)
            def close(self): calls.append('closed')
        mapping = {pointer['bundle']: bundle}
        result = execute(spec, 'Hello', mapping, backend_factory=Backend)
        self.assertEqual(result['frames'], 2)
        self.assertEqual(calls, [(bundle, dict(language='en', threads=1)), ('Hello', dict(preset='af_heart')), 'closed'])
        for changes in (dict(revision='wrong'), dict(manifest_sha256='0'*64)):
            with self.assertRaises(Refused):
                execute(dataclasses.replace(spec, **changes), 'Hello', mapping, backend_factory=Backend)
        self.assertEqual(len(calls), 3)


if __name__ == '__main__': unittest.main()
