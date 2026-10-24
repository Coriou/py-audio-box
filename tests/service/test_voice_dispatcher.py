"""Real owner-loop reads on test-owned temporary Core/media only."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from service.dispatcher import Dispatcher
from service.core import Core
from service.voice_store import VoiceStore

class VoiceDispatcherFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pab-h2-voice-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'data'
        self.version = json.loads(Path('contracts/fixtures/v1/voice-catalogue.json').read_text())['register']['version']
        core = Core(self.root)
        try:
            self.version = core.voices.register('alice', 'seed', self.version, expected_revision=0)['version']
        finally:
            core.close()
        self.dispatcher = Dispatcher(self.root, staging=Path(self.tmp.name)/'staging')
        self.addCleanup(lambda: self.dispatcher.close())

class VoiceDispatcherTests(VoiceDispatcherFixture, unittest.TestCase):
    def test_explicit_reads_owner_thread_detached_and_principal(self):
        request_thread = threading.get_ident()
        observed = []
        original = VoiceStore.get
        def get(store, *args, **kwargs):
            observed.append(threading.get_ident())
            return original(store, *args, **kwargs)
        with patch.object(VoiceStore, 'get', get):
            voice = self.dispatcher.call('voice_get', 'alice', self.version['voice_id'])
            self.assertEqual(self.dispatcher.call('voice_list', 'alice')[0]['voice'], voice)
            self.assertEqual(self.dispatcher.call('voice_resolve', 'alice', voice['id']), voice)
        self.assertTrue(observed)
        self.assertEqual(set(observed), {self.dispatcher.thread.ident})
        self.assertNotEqual(observed[0], request_thread)
        voice['aliases'].append('detached')
        self.assertNotIn('detached', self.dispatcher.call('voice_get','alice',voice['id'])['aliases'])
        self.assertEqual(self.dispatcher.call('voice_version','alice',self.version['id']), self.version)
        with self.assertRaisesRegex(ValueError, 'not_found'):
            self.dispatcher.call('voice_get','bob',voice['id'])
        for command in ('voices.register', 'voice_register', 'callback'):
            with self.assertRaisesRegex(ValueError, 'unknown_control_command'):
                self.dispatcher.call(command)
