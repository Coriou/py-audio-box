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

    def test_compact_owner_commands_avoid_full_get(self):
        observed=[]
        original=VoiceStore.summaries
        def summaries(store,*args,**kwargs):
            observed.append(threading.get_ident())
            return original(store,*args,**kwargs)
        with patch.object(VoiceStore,'summaries',summaries), patch.object(VoiceStore,'get',side_effect=AssertionError('full get')):
            rows=self.dispatcher.call('voice_summaries','alice')
            self.assertEqual(rows[0]['voice']['id'],self.version['voice_id'])
            versions=self.dispatcher.call('voice_version_summaries','alice',self.version['voice_id'])
            self.assertEqual(versions[0]['id'],self.version['id'])
        self.assertEqual(observed,[self.dispatcher.thread.ident])
        with self.assertRaisesRegex(ValueError,'invalid_request'):
            self.dispatcher.call('asset_ingest','alice',object())

class VoiceResolvePublicTests(VoiceDispatcherFixture, unittest.TestCase):
    def test_public_resolve_names_errors_and_budget(self):
        from service.domain import voice_read
        from service.catalogue import Failure
        voice_id = self.version['voice_id']
        self.dispatcher.close()
        core = Core(self.root)
        try:
            core.voices.aliases('alice','alias',voice_id,['default-reading'],expected_revision=1)
        finally:
            core.close()
        self.dispatcher = Dispatcher(self.root,staging=Path(self.tmp.name)/'staging')
        def resolve(name, principal='alice'):
            return voice_read(self.dispatcher,principal,'voice_resolve',dict(name=name))
        expected = self.dispatcher.call('voice_get','alice',voice_id)
        for name in (voice_id,'default-reading'):
            self.assertEqual(resolve(name),expected)
            with self.assertRaises(Failure) as error: resolve(name,'bob')
            self.assertEqual(error.exception.body['code'],'not_found')
        for payload in ({},dict(name='bad!'),dict(name=voice_id,principal='bob')):
            with self.assertRaises(Failure) as error:
                voice_read(self.dispatcher,'alice','voice_resolve',payload)
            self.assertEqual(error.exception.body['code'],'invalid_request')
        with self.assertRaises(Failure) as error: resolve('unknown')
        self.assertEqual(error.exception.body['code'],'not_found')
        with patch('service.catalogue.VOICE_RESPONSE_BYTES',1):
            with self.assertRaises(Failure) as error: resolve(voice_id)
            self.assertEqual(error.exception.body['code'],'resource_limit')
        from service.domain import production_call
        archived = production_call(self.dispatcher,'alice','voice_archive',
                                   dict(voice_id=voice_id,expected_revision=2),'archive')
        self.assertEqual((archived['revision'],archived['aliases']),(3,[]))
        for name in (voice_id,'default-reading'):
            with self.assertRaises(Failure) as error: resolve(name)
            self.assertEqual(error.exception.body['code'],'voice_archived' if name == voice_id else 'not_found')

        self.dispatcher.close()
        core = Core(self.root)
        try:
            replacement = dict(self.version, id='replacement-version', voice_id='replacement')
            core.voices.register('alice','replacement',replacement,expected_revision=0)
        finally:
            core.close()
        self.dispatcher = Dispatcher(self.root,staging=Path(self.tmp.name)/'staging')
        assigned = production_call(self.dispatcher,'alice','voice_aliases',
                                  dict(voice_id='replacement',aliases=['default-reading'],expected_revision=1),'reuse')
        self.assertEqual(resolve('default-reading'),assigned)
        production_call(self.dispatcher,'alice','voice_archive',
                        dict(voice_id=voice_id,expected_revision=2),'archive')
        self.assertEqual(resolve('default-reading')['id'],'replacement')
        with self.assertRaises(Failure) as error: resolve('default-reading','bob')
        self.assertEqual(error.exception.body['code'],'not_found')

    def test_cli_resolve_parse(self):
        import contextlib
        import io
        from clients.cli import main
        with patch('clients.cli.HTTP') as transport, contextlib.redirect_stdout(io.StringIO()):
            transport.return_value.call.return_value = {}
            self.assertEqual(main(['voice','resolve','default-reading']),0)
            transport.return_value.call.assert_called_once_with('voice_resolve',dict(name='default-reading'))
