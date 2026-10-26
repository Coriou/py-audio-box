import importlib.util
import json
import io
import os
import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

spec = importlib.util.spec_from_file_location('audio_judge', Path(__file__).parents[1] / 'benchmarks/tts/audio_judge.py')
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)


class AudioJudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.wav = self.root / 'engine-secret.wav'
        with wave.open(str(self.wav), 'wb') as w:
            w.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            w.writeframes(b'\0\0' * 8000)
        self.row = {'status': 'ok', 'requested_text': 'SECRET EXPECTED 1250',
                    'audio': {'path': self.wav.name, 'sha256': j.digest(self.wav.read_bytes())}}
        self.matrix = self.root / 'matrix.json'
        self.matrix.write_text(json.dumps({'schema_version': 1, 'samples': [self.row, self.row]}))
        self.state = self.root / 'private'
        self.snapshot = {'endpoint': {'supports_multiple_audio_references': True}, 'rates_usd_per_token': {'audio': '0.000001', 'prompt': '0.000001', 'completion': '0.000002'}}

    def plan(self, **kw):
        return j.plan(self.matrix, self.root, self.snapshot, 'google-ai-studio', **kw)

    def response(self, *args):
        return {'id': 'req-test', 'provider': 'Google AI Studio', 'model': j.MODEL,
                'usage': {'cost': 0.001, 'prompt_tokens': 32, 'completion_tokens': 20},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                    'heard_transcript': 'hello', 'naturalness': 'medium', 'intelligibility': 'high',
                    'defects': [{'description': 'unsure', 'time_seconds': None}], 'uncertainty': 'synthetic'})}}]}

    def test_dry_run_blind_cost_and_reference(self):
        p = j.run(self.plan(), self.state, key_loader=lambda: self.fail('key read'))
        self.assertEqual(p['sample_count'], 2)
        self.assertEqual(p['new_request_count'], 1)
        self.assertEqual(p['requests'][0]['audio_tokens_upper'], 32)
        self.assertNotIn('SECRET EXPECTED', json.dumps(p))
        self.assertEqual(self.plan(reference=self.wav.name)['requests'][0]['audio_tokens_upper'], 64)
        self.assertNotEqual(p['requests'][0]['cache_key'], self.plan(reference=self.wav.name)['requests'][0]['cache_key'])

    def test_execute_cache_and_full_reservation(self):
        calls = []
        def send(*args):
            ledger = json.loads((self.state / 'ledger.json').read_text())
            self.assertTrue(all(e['reserved_usd'] for e in ledger['entries'].values()))
            body = args[1]
            self.assertNotIn('engine-secret', json.dumps(body))
            self.assertNotIn('SECRET EXPECTED', json.dumps(body))
            calls.append(args)
            return self.response()
        j.run(self.plan(), self.state, True, '.5', lambda: 'secret', send)
        j.run(self.plan(), self.state, True, '.5', lambda: self.fail('cached key read'), send)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads((self.state / 'ledger.json').read_text())['entries'][calls and self.plan()['requests'][0]['cache_key']]['status'], 'complete')

    def test_uncertain_no_retry_redaction(self):
        def timeout(*args):
            raise TimeoutError('secret-key-provider-body')
        with self.assertRaises(ValueError):
            j.run(self.plan(), self.state, True, '.5', lambda: 'secret-key', timeout)
        ledger = (self.state / 'ledger.json').read_text()
        self.assertNotIn('secret-key', ledger)
        self.assertIn('uncertain', ledger)
        with self.assertRaises(ValueError):
            j.run(self.plan(), self.state, True, '.5', lambda: self.fail('retry'))

    def test_run_rejects_invalid_timeout_before_any_side_effect(self):
        for timeout in (61, 120, float('nan'), float('inf'), True, False, 0, -1, '60'):
            for execute in (False, True):
                with self.subTest(timeout=timeout, execute=execute):
                    with patch.object(j, 'locked_state') as lock, patch.object(j, 'atomic') as write:
                        with self.assertRaises(ValueError):
                            j.run(self.plan(), self.state, execute, '.5',
                                  lambda: self.fail('key read'),
                                  lambda *a: self.fail('transport called'), timeout=timeout)
                        lock.assert_not_called()
                        write.assert_not_called()
                    self.assertFalse(self.state.exists())

    def test_run_accepts_transport_deadline_bounds(self):
        for timeout in (.1, 60):
            state = self.root / ('valid-' + str(timeout))
            calls = []
            def send(*args):
                calls.append(args[-1])
                return self.response()
            j.run(self.plan(), state, True, '.5', lambda: 'secret', send, timeout=timeout)
            self.assertEqual(calls, [timeout])

    def test_caps(self):
        for cap in (None, '2', 'NaN', '0', '.000001'):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                j.run(self.plan(), self.state, True, cap, lambda: self.fail('credential before cap'))

    def test_invalid_response_keeps_usage(self):
        response = self.response()
        response['choices'][0]['message']['content'] = '{}'
        with self.assertRaises(ValueError):
            j.run(self.plan(), self.state, True, '.5', lambda: 'secret', lambda *a: response)
        e = next(iter(json.loads((self.state / 'ledger.json').read_text())['entries'].values()))
        self.assertEqual(e['request_id'], 'req-test')
        self.assertEqual(e['usage']['cost'], .001)
        self.assertEqual(e['status'], 'uncertain')

    def test_validation_paths_and_hashes(self):
        link = self.root / 'link.wav'
        link.symlink_to(self.wav)
        for path in (link, '../outside.wav'):
            with self.assertRaises(ValueError):
                j.audio(path, self.root)
        self.row['audio']['sha256'] = 'bad'
        self.matrix.write_text(json.dumps({'schema_version': 1, 'samples': [self.row]}))
        with self.assertRaises(ValueError):
            self.plan()
        self.wav.write_bytes(b'not a wave' * 10)
        with self.assertRaises((ValueError, wave.Error)):
            j.audio(self.wav, self.root)

    def test_lock_exclusion(self):
        with j.locked_state(self.state):
            with self.assertRaises(BlockingIOError):
                with j.locked_state(self.state):
                    self.fail('parallel lock')

    def test_metadata_refuses_unknown_or_unsupported(self):
        endpoint = {'name': 'Google AI Studio | ' + j.CANONICAL_MODEL,
                    'max_completion_tokens': 65536, 'tag': 'test', 'status': 0, 'service_tier': None,
                    'supported_parameters': ['response_format', 'structured_outputs', 'max_tokens', 'reasoning'],
                    'pricing': {'prompt': '.000001', 'completion': '.000002', 'audio': '.000003'}}
        data = {'data': {'id': j.MODEL, 'architecture': {'input_modalities': ['audio']}, 'endpoints': [endpoint]}}
        catalog = {'data': [{'id': j.MODEL, 'canonical_slug': j.CANONICAL_MODEL,
                            'reasoning': {'supported_efforts': ['low'], 'mandatory': True}}]}
        transport = lambda url: catalog if url.endswith('/models') else data
        self.assertEqual(j.metadata('test', transport)['rates_usd_per_token']['audio'], '0.000003')
        del endpoint['pricing']['audio']
        with self.assertRaises(ValueError):
            j.metadata('test', transport)
        endpoint['supported_parameters'] = []
        with self.assertRaises(ValueError):
            j.metadata('test', transport)

    def test_cache_identity_and_invalid_times(self):
        first = self.plan()['requests'][0]['cache_key']
        self.assertNotEqual(first, self.plan(output_cap=1024)['requests'][0]['cache_key'])
        value = json.loads(self.response()['choices'][0]['message']['content'])
        for timestamp in (True, -1, 2, float('nan')):
            value['defects'][0]['time_seconds'] = timestamp
            with self.assertRaises(ValueError):
                j.validate_result(value, 1)

    def test_all_unique_requests_reserved_before_submit(self):
        second = self.root / 'second.wav'
        with wave.open(str(second), 'wb') as w:
            w.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            w.writeframes(b'\1\0' * 8000)
        row = {'status': 'ok', 'audio': {'path': second.name, 'sha256': j.digest(second.read_bytes())}}
        self.matrix.write_text(json.dumps({'schema_version': 1, 'samples': [self.row, row]}))
        def send(*args):
            entries = json.loads((self.state / 'ledger.json').read_text())['entries']
            self.assertEqual(len(entries), 2)
            self.assertTrue(all('reserved_usd' in e for e in entries.values()))
            return self.response()
        j.run(self.plan(), self.state, True, '.5', lambda: 'secret', send)

    def test_bounded_growth_and_replacement(self):
        original_read = os.read
        calls = []
        def grow(fd, count):
            calls.append(count)
            if len(calls) == 1:
                with self.matrix.open('ab') as stream:
                    stream.write(b'x' * 1000)
            return original_read(fd, count)
        with patch.object(j.os, 'read', side_effect=grow):
            with self.assertRaises(ValueError):
                j.bounded_input(self.matrix, self.root, 512)
        self.assertEqual(sum(calls), 513)
        def replace(fd, count):
            self.wav.unlink()
            self.wav.symlink_to(self.matrix)
            return original_read(fd, count)
        with patch.object(j.os, 'read', side_effect=replace):
            with self.assertRaises(ValueError):
                j.audio(self.wav, self.root)

    def test_open_symlink_swap_and_matrix_symlink(self):
        real_open = os.open
        def swap(path, flags, *args, **kwargs):
            if path == self.wav.name:
                self.wav.unlink()
                self.wav.symlink_to(self.matrix)
            return real_open(path, flags, *args, **kwargs)
        with patch.object(j.os, 'open', side_effect=swap):
            with self.assertRaises(ValueError):
                j.audio(self.wav, self.root)
        self.matrix.unlink()
        self.matrix.symlink_to(self.wav)
        with self.assertRaises(ValueError):
            self.plan()

    def test_metadata_free_transmission_exact_frames(self):
        pcm = bytes(range(256)) * 62 + b'\x17\x81' * 64
        with wave.open(str(self.wav), 'wb') as w:
            w.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            w.writeframes(pcm)
        original = self.wav.read_bytes() + b'LIST' + (20).to_bytes(4, 'little') + b'PRIVATE BEN FINDINGS!'
        self.wav.write_bytes(original)
        self.row['audio']['sha256'] = j.digest(original)
        self.matrix.write_text(json.dumps({'schema_version': 1, 'samples': [self.row]}))
        meta, transmitted = j.audio(self.wav, self.root)
        self.assertEqual(meta['original_sha256'], j.digest(original))
        self.assertNotEqual(meta['original_sha256'], meta['transmitted_sha256'])
        with wave.open(io.BytesIO(transmitted), 'rb') as w:
            self.assertEqual(w.getparams()[:3], (1, 2, 8000))
            self.assertEqual(w.readframes(w.getnframes()), pcm)
        def send(*args):
            data = base64.b64decode(args[1]['messages'][0]['content'][1]['input_audio']['data'])
            self.assertEqual(data, transmitted)
            self.assertNotIn(b'PRIVATE', data)
            return self.response()
        j.run(self.plan(), self.state, True, '.25', lambda: 'secret', send)
        self.assertEqual(self.wav.read_bytes(), original)

    def test_replicate_selection_full_ceiling_and_reference_refusal(self):
        ordinary = self.plan()
        repeated = self.plan(replicate_id='blind-repeat-2', select={1})
        self.assertEqual(repeated['sample_count'], 1)
        self.assertNotEqual(ordinary['requests'][0]['cache_key'], repeated['requests'][0]['cache_key'])
        self.assertEqual(ordinary['requests'][0]['completion_tokens_reserved_including_thinking'], 65536)
        self.assertEqual(ordinary['requests'][0]['reserved_usd'], '0.25')
        self.assertEqual(ordinary['requests'][0]['identity']['request_schema']['reasoning'], {'effort': 'low'})
        self.snapshot['endpoint']['supports_multiple_audio_references'] = False
        with self.assertRaises(ValueError):
            self.plan(reference=self.wav.name)

    def test_whole_plan_refused_without_key_or_submission(self):
        manifest = self.plan()
        manifest['requests'] = [dict(manifest['requests'][0], cache_key=str(i)) for i in range(5)]
        with self.assertRaises(ValueError):
            j.run(manifest, self.state, True, '1', lambda: self.fail('key read'),
                  lambda *args: self.fail('submission'))
        self.assertFalse((self.state / 'ledger.json').exists())

    def test_literal_key_parser_no_execution_or_export(self):
        f = self.root / 'local.env'
        f.write_text("OPENROUTER_API_KEY='literal-key'\nOTHER=ignored\n")
        with patch.dict(j.os.environ, {}, clear=True):
            self.assertEqual(j.load_key(f), 'literal-key')
            self.assertNotIn('OPENROUTER_API_KEY', j.os.environ)
            f.write_text('OPENROUTER_API_KEY=$(echo unsafe)')
            with self.assertRaises(ValueError):
                j.load_key(f)

    def test_http_whole_deadline_slow_drip_and_signal_restore(self):
        import http.server
        import threading
        import time
        import signal
        class Drip(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Length', '100')
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b' ')
                        self.wfile.flush()
                        time.sleep(.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = http.server.HTTPServer(('127.0.0.1', 0), Drip)
        worker = threading.Thread(target=server.handle_request)
        worker.start()
        handler = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                j.http('http://127.0.0.1:' + str(server.server_port), timeout=.15)
            self.assertLess(time.monotonic() - started, .8)
            self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        finally:
            worker.join(3)
            server.server_close()
        self.assertFalse(worker.is_alive())

    def test_http_refuses_existing_alarm_and_nonmain_thread(self):
        import signal
        import threading
        old = signal.getsignal(signal.SIGALRM)
        custom = lambda *a: None
        signal.signal(signal.SIGALRM, custom)
        try:
            with self.assertRaises(ValueError):
                j.http('http://127.0.0.1:1', timeout=.1)
            self.assertIs(signal.getsignal(signal.SIGALRM), custom)
        finally:
            signal.signal(signal.SIGALRM, old)
        errors = []
        def call():
            try:
                j.http('http://127.0.0.1:1', timeout=.1)
            except ValueError:
                errors.append(True)
        thread = threading.Thread(target=call)
        thread.start()
        thread.join(1)
        self.assertEqual(errors, [True])


if __name__ == '__main__':
    unittest.main()
