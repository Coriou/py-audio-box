import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

spec = importlib.util.spec_from_file_location('stt', Path(__file__).resolve().parents[1] / 'benchmarks/tts/stt_check.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class STTTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.wav = self.root / 'a.wav'
        with wave.open(str(self.wav), 'wb') as w:
            w.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            w.writeframes(b'\0\0' * 8000)
        self.matrix = self.root / 'matrix.json'
        self.row = {'status': 'ok', 'language': 'en', 'requested_text': 'Hello world.',
                    'audio': {'path': 'a.wav', 'sha256': s.h.audio(self.wav, self.root)[0]['sha256']}}
        self.write_rows([self.row, self.row])
        self.snapshot = s.metadata(lambda url: {'data': [{'id': s.MODEL, 'canonical_slug': s.MODEL,
            'architecture': {'modality': 'audio->transcription'},
            'pricing': {'prompt': '0.0000075', 'completion': '0'}}]})
        self.state = self.root / 'state'
        self.calls = []

    def write_rows(self, rows):
        self.matrix.write_text(json.dumps({'schema_version': 1, 'samples': rows}))

    def plan(self, **kw):
        return s.plan(self.matrix, self.root, self.snapshot, **kw)

    def http(self, url, body, key, timeout):
        self.calls.append((url, body, key, timeout))
        return {'text': 'Hello world.', 'usage': {'cost': 0.0000075, 'seconds': 1}}

    def ledger(self):
        return json.loads((self.state / 'ledger.json').read_text())

    def execute(self, manifest=None, **kw):
        return s.run(manifest or self.plan(), self.state, execute=True, max_cost='1',
                     key_loader=lambda: 'fixture-secret', transport=self.http, **kw)

    def test_numeric_boundaries_do_not_clear_partial_or_signed_values(self):
        for lang, number, amount in [('en', 'Value 3 today', 'Pay 20 euros'),
                                     ('fr', 'Valeur 3 aujourd’hui', 'Payer 20 euros')]:
            for source, literal, kind, replacements in [
                (number, '3', 'number', ['-3', '−3', '+3', '3.5', '3,5', '13', '30', '1,3', '1 3', '€3', '3€']),
                (amount, '20 euros', 'amount', ['-20 euros', '−20 euros', '+20 euros', '120 euros', '1,20 euros', '1 20 euros'])]:
                for changed in replacements:
                    with self.subTest(lang=lang, kind=kind, changed=changed):
                        result = s.compare(source, source.replace(literal, changed), lang,
                                           [{'kind': kind, 'expected': literal}])
                        self.assertEqual(result['status'], 'REVIEW')
                        self.assertEqual(result['critical_checks'][0]['status'], 'review')
                self.assertEqual(s.compare(source, source, lang,
                    [{'kind': kind, 'expected': literal}])['status'], 'CLEAR')

    def test_signed_difference_survives_generic_normalization(self):
        for lang in ('en', 'fr'):
            for sign in ('-', '−', '+'):
                for expected, heard in [('Value 3 today', f'Value {sign}3 today'),
                                        (f'Value {sign}3 today', 'Value 3 today')]:
                    with self.subTest(lang=lang, expected=expected, heard=heard):
                        result = s.compare(expected, heard, lang, [])
                        self.assertEqual(result['status'], 'REVIEW')
                        self.assertIn('signed_numeric_difference', result['alerts'])
            text = 'Identifier A-2047 today'
            self.assertEqual(s.compare(text, text, lang,
                [{'kind': 'identifier', 'expected': 'A-2047'}])['status'], 'CLEAR')
            self.assertEqual(s.signed_numbers(text), [])
        for signed in ('-3', '−3', '+3', '-20 euros'):
            kind = 'amount' if 'euros' in signed else 'number'
            self.assertEqual(s.compare(signed, signed, 'en',
                [{'kind': kind, 'expected': signed}])['critical_checks'][0]['status'], 'review')

    def test_plan_deduplicates_exact_audio(self):
        m = self.plan()
        self.assertEqual((m['sample_count'], m['unique_stt_count']), (2, 1))
        self.assertEqual(m['estimated_usd'], '0.0000075')
        self.assertEqual(set(m['planned_critical_sidecar']), {'0', '1'})

    def test_dry_run_never_loads_secret_or_submits(self):
        def forbidden(*a):
            self.fail('dry run touched credential/paid transport')
        s.run(self.plan(), self.state, key_loader=forbidden, transport=forbidden)
        self.assertFalse((self.state / 'ledger.json').exists())
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.state / 'manifest.json').stat().st_mode & 0o777, 0o600)

    def test_request_is_audio_only_no_expected_engine_feedback(self):
        self.execute()
        url, body, key, timeout = self.calls[0]
        self.assertEqual(url, s.h.API + '/audio/transcriptions')
        self.assertEqual(set(body), {'model', 'language', 'input_audio', 'response_format'})
        self.assertNotIn('Hello', json.dumps(body))
        self.assertEqual(body['model'], s.MODEL)
        entry = next(iter(self.ledger()['entries'].values()))
        self.assertEqual(entry['transcript']['provider'], 'unknown')
        self.assertIsNone(entry['transcript']['timestamps'])

    def test_expected_change_only_recompares(self):
        self.execute()
        self.row['requested_text'] = 'Hello everyone.'
        self.write_rows([self.row])
        m = self.execute()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(m['new_stt_count'], 0)
        self.assertEqual(len(self.ledger()['comparisons']), 2)

    def test_language_version_params_audio_invalidate(self):
        original = self.plan()['requests'][0]['cache_key']
        self.row['language'] = 'fr'
        self.write_rows([self.row])
        self.assertNotEqual(original, self.plan()['requests'][0]['cache_key'])
        identity = self.plan()['requests'][0]['identity']
        for field in ('original_sha256', 'transmitted_sha256', 'model_version', 'parameters'):
            changed = copy.deepcopy(identity)
            changed[field] = 'changed'
            self.assertNotEqual(s.h.digest(s.h.canonical(identity)), s.h.digest(s.h.canonical(changed)))

    def test_timeout_retains_reservation_blocks_retry(self):
        def fail(*a):
            raise TimeoutError('fixture-secret must not leak')
        with self.assertRaises(ValueError):
            s.run(self.plan(), self.state, True, '1', lambda: 'fixture-secret', fail)
        entry = next(iter(self.ledger()['entries'].values()))
        self.assertEqual(entry['status'], 'uncertain')
        self.assertEqual(entry['reserved_usd'], '0.01')
        self.assertNotIn('fixture-secret', (self.state / 'ledger.json').read_text())
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_serial_failure_retains_unsubmitted_reservations(self):
        second = copy.deepcopy(self.row)
        second['language'] = 'fr'
        self.write_rows([self.row, second])
        calls = []
        def fail(*args):
            calls.append(args)
            raise TimeoutError()
        with self.assertRaises(ValueError):
            s.run(self.plan(), self.state, True, '1', lambda: 'secret', fail)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sorted(e['status'] for e in self.ledger()['entries'].values()),
                         ['reserved', 'uncertain'])

    def test_resolved_model_mismatch_stops(self):
        with self.assertRaises(ValueError):
            s.run(self.plan(), self.state, True, '1', lambda: 'secret',
                  lambda *args: {'text': 'Hello', 'model': 'different/model',
                                 'provider': 'fixture', 'usage': {'cost': 0, 'seconds': 1}})
        entry = next(iter(self.ledger()['entries'].values()))
        self.assertEqual(entry['status'], 'uncertain')
        self.assertEqual(entry['transcript']['resolved_model'], 'different/model')

    def test_transmitted_wav_metadata_free_exact_pcm(self):
        import base64
        import io
        self.execute()
        clean = base64.b64decode(self.calls[0][1]['input_audio']['data'])
        with wave.open(io.BytesIO(clean), 'rb') as wav:
            self.assertEqual(wav.readframes(wav.getnframes()), b'\0\0' * 8000)
        self.assertEqual(len(clean), 16044)

    def test_unknown_charge_keeps_transcript_before_comparison(self):
        with self.assertRaises(ValueError):
            s.run(self.plan(), self.state, True, '1', lambda: 'key',
                  lambda *a: {'text': 'Hello world.'})
        ledger = self.ledger()
        e = next(iter(ledger['entries'].values()))
        self.assertEqual(e['transcript']['text'], 'Hello world.')
        self.assertEqual(e['status'], 'uncertain')
        self.assertTrue(ledger['comparisons'])

    def test_transcript_is_durable_before_compare(self):
        original = s.compare
        def check(*args):
            self.assertEqual(next(iter(self.ledger()['entries'].values()))['transcript']['text'], 'Hello world.')
            return original(*args)
        with patch.object(s, 'compare', check):
            self.execute()

    def test_no_key_in_server_transcript_persisted(self):
        with self.assertRaises(ValueError):
            s.run(self.plan(), self.state, True, '1', lambda: 'fixture-secret',
                  lambda *a: {'text': 'fixture-secret', 'usage': {'cost': 0}})
        self.assertNotIn('fixture-secret', (self.state / 'ledger.json').read_text())

    def test_import_no_network(self):
        m = self.plan()
        s.run(m, self.state, imported={m['requests'][0]['cache_key']: {'text': 'Hello world.'}})
        self.assertEqual(next(iter(self.ledger()['entries'].values()))['status'], 'imported')
        self.assertTrue(self.ledger()['comparisons'])
        self.assertEqual(self.calls, [])

    def test_budget_and_timeout_rejected_before_http(self):
        for cap in ('0', '1.01', 'NaN', '0.001'):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                s.run(self.plan(), self.state, True, cap, lambda: self.fail('key read'))
        with self.assertRaises(ValueError):
            self.execute(timeout=61)
        self.assertEqual(self.calls, [])

    def test_matrix_hash_and_symlink_rejected(self):
        self.row['audio']['sha256'] = 'bad'
        self.write_rows([self.row])
        with self.assertRaises(ValueError):
            self.plan()
        self.matrix.unlink()
        self.matrix.symlink_to(self.wav)
        with self.assertRaises(ValueError):
            self.plan()

    def test_audio_change_after_plan_never_submitted(self):
        m = self.plan()
        with self.wav.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaises(ValueError):
            self.execute(m)
        self.assertEqual(self.calls, [])

    def test_metadata_rejects_unknown_pricing(self):
        for rate in ('NaN', '-1', '0'):
            model = copy.deepcopy(self.snapshot['model'])
            model['pricing']['prompt'] = rate
            with self.assertRaises(ValueError):
                s.metadata(lambda *a: {'data': [model]})

    def test_money_numeric_minor_units(self):
        self.assertEqual(s.money('1250.50€', 'en'), ('EUR', 125050))
        self.assertEqual(s.money('125050€', 'en'), ('EUR', 12505000))
        self.assertEqual(s.money('1 250,50 euros', 'fr'), ('EUR', 125050))
        self.assertEqual(s.money('USD 1,250.50', 'en'), ('USD', 125050))

    def test_ambiguous_dollar_currency(self):
        self.assertIsNone(s.money('$1250.50', 'en'))
        self.assertIsNone(s.money('twenty dollars', 'en'))

    def test_money_spoken_limited(self):
        self.assertEqual(s.money('twenty euros and fifty cents', 'en'), ('EUR', 2050))
        self.assertEqual(s.money('vingt euros et cinquante centimes', 'fr'), ('EUR', 2050))
        self.assertIsNone(s.money('twelve hundred euros', 'en'))

    def test_ambiguous_locale_is_unknown(self):
        for value, lang in [('1,250 euros', 'fr'), ('1.250 euros', 'en'), ('1.250,50€', 'fr'), ('€1€', 'en')]:
            with self.subTest(value=value):
                self.assertIsNone(s.money(value, lang))

    def test_money_mismatch_review(self):
        r = s.compare('Pay 1250.50€ now.', 'Pay 125050€ now.', 'en',
                      [{'kind': 'amount', 'expected': '1250.50€'}])
        self.assertEqual(r['status'], 'REVIEW')
        self.assertIn('critical_amount', r['alerts'])

    def test_identifiers_and_version_separate(self):
        r = s.compare('Use A-2047 version B.', 'Use A-2047 version D.', 'en',
            [{'kind': 'identifier', 'expected': 'A-2047'}, {'kind': 'version', 'expected': 'B'}])
        self.assertEqual([c['status'] for c in r['critical_checks']], ['no_anomaly_detected', 'review'])

    def test_identifier_prefix_never_equivalent(self):
        r = s.compare('Use A-2047.', 'Use A-20470.', 'en',
                      [{'kind': 'identifier', 'expected': 'A-2047'}])
        self.assertEqual(r['critical_checks'][0]['status'], 'review')

    def test_number_locale_annotations(self):
        r = s.compare('Valeur 1 250,50.', 'Valeur 1250,50.', 'fr',
                      [{'kind': 'number', 'expected': '1 250,50', 'heard_forms': ['1250,50']}])
        self.assertEqual(r['critical_checks'][0]['status'], 'no_anomaly_detected')
        self.assertEqual(r['status'], 'REVIEW')  # Text disagreement still needs a listener.

    def test_unsupported_amount_even_identical_needs_review(self):
        r = s.compare('twelve hundred euros', 'twelve hundred euros', 'en',
                      [{'kind': 'amount', 'expected': 'twelve hundred euros'}])
        self.assertEqual(r['status'], 'REVIEW')

    def test_contraction_negation(self):
        r = s.compare("We don't send it today.", 'We send it today.', 'en', [])
        self.assertIn('possible_negation_loss', r['alerts'])

    def test_negation_loss_low_wer_review(self):
        r = s.compare('We will not send the final project report to the office today.',
                      'We will send the final project report to the office today.', 'en', [])
        self.assertIn('possible_negation_loss', r['alerts'])
        self.assertEqual(r['status'], 'REVIEW')

    def test_french_negation(self):
        r = s.compare('Ne pas envoyer le rapport.', 'Envoyer le rapport.', 'fr', [])
        self.assertIn('possible_negation_loss', r['alerts'])

    def test_tail_omission(self):
        r = s.compare('The report is ready for tomorrow morning.', 'The report is ready.', 'en', [])
        self.assertIn('possible_tail_omission_or_change', r['alerts'])

    def test_proper_name_not_pronunciation_proof(self):
        r = s.compare('Ask Coriou.', 'Ask Corio.', 'en', [{'kind': 'proper_name', 'expected': 'Coriou'}])
        self.assertEqual(r['status'], 'REVIEW')
        self.assertIn('not pronunciation proof', r['proper_name_policy'])
        self.assertNotIn('intelligibility', r)

    def test_empty_uncertain(self):
        for text, uncertain in [('', False), ('Hello world.', True), ('[unclear]', False)]:
            self.assertEqual(s.compare('Hello world.', text, 'en', [], uncertain)['status'], 'REVIEW')

    def test_nfc_raw_normalized_metrics_clear_is_not_acceptance(self):
        r = s.compare('Été!', 'E\u0301TE\u0301.', 'fr', [])
        self.assertEqual(r['normalized']['wer']['rate'], 0)
        self.assertGreater(r['raw']['wer']['rate'], 0)
        self.assertEqual(r['status'], 'CLEAR')
        self.assertIn('not acceptance', r['meaning'])

    def test_missing_or_unsupported_annotations_review(self):
        self.assertEqual(s.compare('Hi', 'Hi', 'en')['status'], 'REVIEW')
        self.assertEqual(s.compare('Hi', 'Hi', 'en', [{'kind': 'semantic', 'expected': 'Hi'}])['status'], 'REVIEW')

    def test_billing_impossible_seconds_stops_serial_requests(self):
        other = copy.deepcopy(self.row)
        other['language'] = 'fr'
        self.write_rows([self.row, other])
        calls = []
        def send(*args):
            calls.append(1)
            return {'text': 'Hello', 'id': 'req-1', 'provider': 'fixture',
                    'usage': {'seconds': 999999, 'cost': .0000075, 'tokens': 2}}
        with self.assertRaises(ValueError):
            s.run(self.plan(), self.state, True, '1', lambda: 'secret', send)
        self.assertEqual(len(calls), 1)
        e = next(e for e in self.ledger()['entries'].values() if e['status'] == 'uncertain')
        self.assertEqual(e['usage']['seconds'], 999999)
        self.assertEqual(e['request_id'], 'req-1')

    def test_billing_metadata_retained_rounding_review(self):
        response = {'text': 'Hello', 'id': 'req-2', 'provider': 'fixture', 'model': s.MODEL,
                    'usage': {'seconds': 1.01, 'cost': .0000076, 'extra': {'tokens': 3}}}
        s.run(self.plan(), self.state, True, '1', lambda: 'secret', lambda *a: response)
        e = next(iter(self.ledger()['entries'].values()))
        self.assertEqual(e['usage'], response['usage'])
        self.assertEqual(e['billing_check']['status'], 'review')
        self.assertEqual(e['actual_usd'], '0.0000076')

    def test_billing_invalid_seconds_and_cost(self):
        for seconds, cost in [(0, .0000075), (-1, .0000075), ('NaN', .0000075),
                              (True, .0000075), (1, .011), (1, -1), (601, .0001), (1, 0), (1, 'NaN'), (1, None)]:
            with self.subTest(seconds=seconds, cost=cost), tempfile.TemporaryDirectory() as state:
                with self.assertRaises(ValueError):
                    s.run(self.plan(), Path(state), True, '1', lambda: 'secret',
                          lambda *a: {'text': 'Hi', 'usage': {'seconds': seconds, 'cost': cost}})

    def test_price_variation_continues_and_reuses_once(self):
        other = copy.deepcopy(self.row)
        other['language'] = 'fr'
        self.write_rows([self.row, other])
        calls = []
        def send(*args):
            calls.append(1)
            on_disk = self.ledger()['entries']
            self.assertEqual(sum(e['status'] == 'sending' for e in on_disk.values()), 1)
            return {'text': 'Hello', 'usage': {'seconds': 1, 'cost': .00006}}
        m = s.run(self.plan(), self.state, True, '.02', lambda: 'secret', send)
        self.assertEqual(len(calls), 2)
        self.assertEqual(m['cumulative_reserved_usd'], '0.02')
        self.assertTrue(all(e['status'] == 'price_review' for e in self.ledger()['entries'].values()))
        s.run(self.plan(), self.state, True, '.02', lambda: self.fail('key reload'), send)
        self.assertEqual(len(calls), 2)

    def test_hash_backup_reconcile_resume_retains_budget(self):
        other = copy.deepcopy(self.row)
        other['language'] = 'fr'
        self.write_rows([self.row, other])
        def stop(*args):
            raise TimeoutError()
        with self.assertRaises(ValueError):
            s.run(self.plan(), self.state, True, '.02', lambda: 'secret', stop)
        ledger = self.ledger()
        k = next(k for k, e in ledger['entries'].items() if e['status'] == 'uncertain')
        ledger['entries'][k].update(transcript=s.transcript_record({'text': 'Hello'}),
                                  usage={'seconds': 1, 'cost': .00006})
        s.h.atomic(self.state / 'ledger.json', ledger)
        raw = (self.state / 'ledger.json').read_bytes()
        manifest_raw = (self.state / 'manifest.json').read_bytes()
        with self.assertRaises(ValueError):
            s.reconcile_retained(self.state, 'bad', k)
        digest = s.h.digest(raw)
        s.reconcile_retained(self.state, digest, k)
        backup = self.state / ('backup-' + digest)
        self.assertEqual((backup / 'ledger.json').read_bytes(), raw)
        self.assertEqual((backup / 'manifest.json').read_bytes(), manifest_raw)
        self.assertEqual((backup / 'ledger.json').stat().st_mode & 0o777, 0o400)
        with self.assertRaises(ValueError):
            s.reconcile_retained(self.state, digest, k)
        m = self.execute()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(m['cumulative_reserved_usd'], '0.02')
        self.assertEqual(self.ledger()['entries'][k]['status'], 'price_review')

    def test_duration_review_continues_without_hiding_price_review(self):
        m = self.plan()
        s.run(m, self.state, True, '.01', lambda: 'secret',
              lambda *a: {'text': 'Hello', 'usage': {'seconds': 10, 'cost': .0003}})
        e = next(iter(self.ledger()['entries'].values()))
        self.assertEqual(e['status'], 'usage_review')
        self.assertTrue(e['usage_review'])
        self.assertTrue(e['price_review'])
        self.assertEqual(e['usage']['seconds'], 10)
        self.assertEqual(e['cost_state'], 'cost_unverified_invoice')
        s.run(m, self.state, True, '.01', lambda: self.fail('key reload'))

    def test_duration_review_reconciliation_preserves_prior_audit(self):
        m = self.plan()
        self.execute(m)
        ledger = self.ledger()
        k = m['requests'][0]['cache_key']
        ledger['entries'][k].update(status='uncertain', usage={'seconds': 10, 'cost': .0003})
        ledger['reconciliation_audit'] = {'historical': True}
        s.h.atomic(self.state / 'ledger.json', ledger)
        raw = (self.state / 'ledger.json').read_bytes()
        s.reconcile_retained(self.state, s.h.digest(raw), k)
        after = self.ledger()
        self.assertEqual(after['entries'][k]['status'], 'usage_review')
        self.assertEqual(after['entries'][k]['reserved_usd'], '0.01')
        self.assertEqual(after['prior_reconciliation_audits'], [{'historical': True}])
        self.assertFalse(after['reconciliation_audit']['invoice_verified'])

    def test_generic_reserved_and_sending_never_resumed(self):
        for status in ('reserved', 'sending'):
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                m = self.plan()
                s.h.atomic(state / 'ledger.json', {'entries': {
                    m['requests'][0]['cache_key']: {'status': status, 'reserved_usd': '.01'}},
                    'comparisons': {}})
                with self.assertRaises(ValueError):
                    s.run(m, state, True, '1', lambda: self.fail('key read'), self.http)
        self.assertEqual(self.calls, [])

    def test_annotations_validated_in_plan_and_coverage(self):
        for a in ([{'kind': 'tail', 'expected': 'absent'}], [{'kind': 'tail', 'expected': ''}], {}):
            with self.assertRaises(ValueError):
                self.plan(annotations={'0': a})
        m = self.plan(annotations={'0': [{'kind': 'tail', 'expected': 'world.'}], '1': []})
        self.assertEqual(m['annotation_coverage'], {'assessed_items': 1, 'missing_items': 1, 'checks': 1})

    def test_env_file_cli_dry_run_never_reads_key(self):
        with patch.object(s, 'metadata', return_value=self.snapshot), patch.object(s.h, 'load_key',
                side_effect=AssertionError('key read')):
            self.assertEqual(s.main(['--matrix', str(self.matrix), '--input-root', str(self.root),
                '--state', str(self.state), '--env-file', str(self.root / 'unread.env')]), 0)

    def test_env_file_process_environment_precedence(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'process-key'}):
            self.assertEqual(s.h.load_key(self.root / 'does-not-exist'), 'process-key')

    def test_tail_case_punctuation_quotes_and_apostrophes(self):
        expected = "The report is ready; Let's go!"
        r = s.compare(expected, 'THE REPORT IS READY. “LET’S GO.”', 'en',
                      [{'kind': 'tail', 'expected': "Let's go!"}])
        self.assertEqual(r['status'], 'CLEAR')
        self.assertGreater(r['raw']['cer']['errors'], 0)

    def test_clause_tail_does_not_clear_other_content_loss(self):
        r = s.compare('Pay 20 euros, before Friday.', 'Pay 50 euros before Friday', 'en',
                      [{'kind': 'tail', 'expected': 'Pay 20 euros, before Friday.'},
                       {'kind': 'amount', 'expected': '20 euros'}])
        self.assertEqual(r['critical_checks'][0]['status'], 'no_anomaly_detected')
        self.assertIn('critical_amount', r['alerts'])
        r = s.compare('Ready before Friday morning.', 'Ready before Friday.', 'en',
                      [{'kind': 'tail', 'expected': 'before Friday morning.'}])
        self.assertIn('critical_tail', r['alerts'])

    def test_finite_orthography_only_with_safe_checks(self):
        for annotations, status in [([{'kind': 'tail', 'expected': 'neighbourhood.'}], 'CLEAR'),
                                    ([], 'REVIEW'), (None, 'REVIEW')]:
            r = s.compare('Visit the neighbourhood.', 'Visit the neighborhood.', 'en', annotations)
            self.assertEqual(r['status'], status)
            self.assertIn('orthography_only', r['notes'])
            self.assertGreater(r['normalized']['wer']['errors'], 0)
        self.assertEqual(s.compare('Visit neighborhoods.', 'Visit neighborhood.', 'en', [])['status'], 'REVIEW')

    def test_finite_numeric_checks_keep_text_disagreement(self):
        for lang, word in [('en', 'three'), ('fr', 'trois')]:
            for expected, heard in [('3', word), (word, '3')]:
                r = s.compare(expected, heard, lang, [{'kind': 'number', 'expected': expected}])
                self.assertEqual(r['critical_checks'][0]['status'], 'no_anomaly_detected')
                self.assertEqual(r['status'], 'REVIEW')
        r = s.compare('3', 'thirty', 'en', [{'kind': 'number', 'expected': '3'}])
        self.assertIn('critical_number', r['alerts'])

    def test_amount_million_units_and_garbled_forms_not_cleared(self):
        for heard in ['1,250,000 euros', '1,250.50 cents', 'INE de 160-150 euros',
                      'INDE 100, 150, 50 euros']:
            r = s.compare('1,250.50 euros', heard, 'en',
                          [{'kind': 'amount', 'expected': '1,250.50 euros'}])
            self.assertIn('critical_amount', r['alerts'])
        r = s.compare('Costs decrease from 200 to 180 euros.',
                      'Costs decrease from $200 to $1,000. 180 euros.', 'en',
                      [{'kind': 'tail', 'expected': 'Costs decrease from 200 to 180 euros.'}])
        self.assertEqual(r['status'], 'REVIEW')

    def test_ambiguous_identifier_and_name_variants(self):
        for lang in ('en', 'fr'):
            r = s.compare('A-2047', 'à 2047', lang, [{'kind': 'identifier', 'expected': 'A-2047'}])
            self.assertIn('critical_identifier', r['alerts'])
        for heard in ['énéal', 'Eneal', 'ÉNEAL']:
            r = s.compare('Énéal', heard, 'fr', [{'kind': 'proper_name', 'expected': 'Énéal'}])
            self.assertIn('ASR_name_review', r['alerts'])
        self.assertIn('ASR_name_review', s.compare('La SAS arrive.', 'La SASS arrive.', 'fr', [])['alerts'])

    def test_negation_changes_and_unsupported_forms_still_review(self):
        for expected, heard, lang in [('Ne pas partir.', 'Ne partir.', 'fr'),
                                      ('No extra charge.', 'Extra charge.', 'en')]:
            r = s.compare(expected, heard, lang, [{'kind': 'tail', 'expected': expected}])
            self.assertIn('possible_negation_loss', r['alerts'])
        with self.assertRaises(ValueError):
            s.compare('Bonjour.', 'bonjour', 'fr', [{'kind': 'tail', 'expected': 'bonjour'}])

    def test_comparer_version_reuses_entries_and_preserves_billing(self):
        m = self.plan()
        self.execute(m)
        before = self.ledger()
        with patch.object(s, 'COMPARER', 'fixture-next'):
            s.run(m, self.state, transport=lambda *a: self.fail('HTTP'),
                  key_loader=lambda: self.fail('key read'))
        after = self.ledger()
        self.assertEqual(before['entries'], after['entries'])
        self.assertEqual(len(after['comparisons']), 2)
        self.assertEqual(len(self.calls), 1)

    def test_partial_comparisons_checkpoint_before_later_failure(self):
        m = self.plan()
        ledger = {'entries': {m['items'][0]['cache_key']: {
            'transcript': s.transcript_record({'text': 'Hello world.'})}}, 'comparisons': {}}
        m['items'][1]['expected'] = 'Different text'
        m['items'][1]['expected_sha256'] = s.h.digest(b'Different text')
        checkpoints = []
        original = s.compare
        def fail_second(expected, *args):
            if expected == 'Different text':
                raise ValueError('fixture comparison failure')
            return original(expected, *args)
        with patch.object(s, 'compare', fail_second), self.assertRaises(ValueError):
            s.update_comparisons(ledger, m, lambda: checkpoints.append(copy.deepcopy(ledger)))
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(len(checkpoints[0]['comparisons']), 1)
        s.update_comparisons(ledger, m)
        self.assertEqual(len(ledger['comparisons']), 2)
        s.update_comparisons(ledger, m)
        self.assertEqual(len(ledger['comparisons']), 2)


if __name__ == '__main__':
    unittest.main()
