#!/usr/bin/env python3
"""Finite, private STT feedback. No network submission without --execute."""
import argparse
import base64
from decimal import Decimal
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
import wave

HERE = Path(__file__).resolve().parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

h = module('stt_private_helpers', HERE / 'audio_judge.py')
fidelity = module('stt_fidelity', HERE.parent.parent / 'lib/fidelity.py')
MODEL = 'openai/whisper-large-v3'
CATALOG = h.API + '/models?output_modalities=transcription'
STATE_ROOT = Path.home() / '.local/state/py-audio-box'
STATE = STATE_ROOT / 'stt-evaluation-r411'
MATRIX = STATE_ROOT / 'voice-comparison-r409/matrix-r7.json'
COMPARER = 'r412-finite-v3'
ORTHOGRAPHY = {'en': {'neighbourhood': 'neighborhood'}, 'fr': {}}


def metadata(transport=h.http):
    models = transport(CATALOG)['data']
    matches = [m for m in models if m.get('id') == MODEL]
    if len(matches) != 1:
        raise ValueError('selected transcription model unavailable')
    m = matches[0]
    if (m.get('canonical_slug') != MODEL or
            m.get('architecture', {}).get('modality') != 'audio->transcription'):
        raise ValueError('model identity/modality changed')
    # Official transcription tutorial defines Whisper catalog pricing as USD/second.
    rate = Decimal(m['pricing']['prompt'])
    if not rate.is_finite() or rate <= 0 or Decimal(m['pricing']['completion']) != 0:
        raise ValueError('unknown duration pricing')
    return {'source': CATALOG, 'fetched_at': time.time(), 'model': m,
            'usd_per_second': str(rate), 'unit_source':
            'https://openrouter.ai/blog/tutorials/transcription-on-openrouter/',
            'provider_pin_supported': False, 'weights_revision': 'not exposed by catalog'}


def read_json(path, root, limit=4 * 1024 * 1024):
    return json.loads(h.bounded_input(path, root, limit)[1])


def plan(matrix=MATRIX, root=STATE_ROOT / 'voice-comparison-r409', snapshot=None,
         annotations=None):
    if snapshot is None:
        raise ValueError('verified pricing snapshot required')
    matrix_path, data = h.bounded_input(matrix, root, 4 * 1024 * 1024)
    rows = json.loads(data)
    if rows.get('schema_version') != 1 or not isinstance(rows.get('samples'), list):
        raise ValueError('invalid matrix')
    sidecar = annotations or {}
    if not isinstance(sidecar, dict) or any(k not in {str(i) for i in range(len(rows['samples']))} for k in sidecar):
        raise ValueError('invalid annotation indices')
    requests, items = [], []
    for index, row in enumerate(rows['samples']):
        if row.get('status') != 'ok':
            raise ValueError('finite matrix requires successful audio in every slot')
        lang, expected = row['language'], row['requested_text']
        if lang not in ('en', 'fr') or not isinstance(expected, str):
            raise ValueError('only explicit EN/FR text supported')
        audio, transmitted = h.audio(row['audio']['path'], root)
        if audio['original_sha256'] != row['audio']['sha256']:
            raise ValueError('matrix audio hash mismatch')
        identity = {'original_sha256': audio['original_sha256'],
                    'transmitted_sha256': audio['transmitted_sha256'],
                    'model': MODEL, 'model_version': snapshot['model']['canonical_slug'],
                    'language': lang, 'parameters': {'format': 'wav', 'response_format': 'json'}}
        key = h.digest(h.canonical(identity))
        with wave.open(io.BytesIO(transmitted), 'rb') as wav:
            estimate = (Decimal(wav.getnframes()) * Decimal(snapshot['usd_per_second'])
                        / Decimal(wav.getframerate()))
        requests.append({'cache_key': key, 'identity': identity, 'audio': audio,
                         'estimated_usd': str(estimate),
                         'reserved_usd': str(max(Decimal('0.01'), estimate))})
        critical = sidecar.get(str(index), row.get('critical_literals'))
        validate_annotations(expected, critical)
        items.append({'matrix_index': index, 'cache_key': key, 'expected': expected,
                      'expected_sha256': h.digest(expected.encode()), 'language': lang,
                      'critical_annotations': critical})
    if not items or len(items) > 48:
        raise ValueError('finite slice requires 1..48 items')
    unique = {r['cache_key']: r for r in requests}
    return {'version': 1, 'matrix_sha256': h.digest(data), 'matrix_path': str(matrix_path),
            'annotations_sha256': h.digest(h.canonical({str(i['matrix_index']): i['critical_annotations'] for i in items})),
            'pricing_snapshot': snapshot, 'sample_count': len(items),
            'unique_stt_count': len(unique), 'requests': list(unique.values()), 'items': items,
            'estimated_usd': str(sum((Decimal(r['estimated_usd']) for r in unique.values()), Decimal(0))),
            'reservation_policy': 'USD 0.01 minimum per unique request; local stop control, not server billing guarantee',
            'annotation_coverage': {'assessed_items': sum(bool(i['critical_annotations']) for i in items),
                                    'missing_items': sum(not i['critical_annotations'] for i in items),
                                    'checks': sum(len(i['critical_annotations'] or []) for i in items)},
            'planned_critical_sidecar': {str(i['matrix_index']): [] for i in items
                                        if i['critical_annotations'] is None}}


# Deliberately finite spoken forms, not a general semantic number interpreter.
SPOKEN = {'en': {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                 'ten': 10, 'twenty': 20, 'fifty': 50},
          'fr': {'zéro': 0, 'un': 1, 'deux': 2, 'trois': 3, 'quatre': 4,
                 'cinq': 5, 'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9,
                 'dix': 10, 'vingt': 20, 'cinquante': 50}}


def money(text, language):
    """One full amount literal -> (currency, integer minor units), else unknown."""
    text = unicodedata.normalize('NFC', text).casefold().strip()
    currencies = {'€': 'EUR', 'eur': 'EUR', 'euro': 'EUR', 'euros': 'EUR',
                  '$': 'USD', 'usd': 'USD', 'dollar': 'USD', 'dollars': 'USD'}
    # Digits: locale grouping must be valid; no guessing separators.
    token = r'(?:€|eur|euros?|\$|usd|dollars?)'
    match = re.fullmatch(rf'({token})?\s*([\d., \u00a0\u202f]+)\s*({token})?', text)
    if match and bool(match[1]) != bool(match[3]):
        currency_token = match[1] or match[3]
        if currency_token in ('$', 'dollar', 'dollars'):
            return None  # USD/CAD/AUD cannot be inferred from language alone.
        currency = currencies[currency_token]
        number = match[2].strip()
        if language == 'en':
            pattern = r'(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{1,2})?'
            group, decimal = ',', '.'
        elif language == 'fr':
            pattern = r'(?:\d+|\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?:,\d{1,2})?'
            group, decimal = ' ', ','
            number = number.replace('\u00a0', ' ').replace('\u202f', ' ')
        else:
            return None
        if not re.fullmatch(pattern, number):
            return None
        value = Decimal(number.replace(group, '').replace(decimal, '.')) * 100
        return currency, int(value)
    # Small exact units + optional cents. Unsupported phrases stay unknown.
    match = re.fullmatch(rf'(\w+)\s+({token})(?:\s+(?:and |et )?(\w+)\s+(?:cents?|centimes?))?', text)
    if match and match[2] not in ('$', 'dollar', 'dollars') and match[1] in SPOKEN.get(language, {}):
        cents = SPOKEN[language].get(match[3], None) if match[3] else 0
        if cents is not None:
            return currencies[match[2]], 100 * SPOKEN[language][match[1]] + cents
    return None


def surface_present(literal, text):
    return re.search(r'(?<!\w)' + re.escape(literal) + r'(?!\w)', text) is not None


def numeric_surface_present(literal, text):
    """Keep signs, grouping, decimal separators and currency with the number."""
    for match in re.finditer(r'(?<!\w)' + re.escape(literal) + r'(?!\w)', text):
        before, after = text[:match.start()], text[match.end():]
        if re.search(r'[+−\-€$]\s*$', before):
            continue
        if re.search(r'\d[., \u00a0\u202f]*$', before):
            continue
        if re.match(r'[., \u00a0\u202f]*\d', after):
            continue
        if re.match(r'[€$]', after):
            continue
        yield match


def signed_numbers(text):
    # A hyphen within an identifier (A-2047) is not a numeric sign.
    return re.findall(r'(?<![\w])(?:[€$]\s*)?[+−-]\s*(?:[€$]\s*)?'
                      r'\d+(?:[., \u00a0\u202f]\d+)*',
                      unicodedata.normalize('NFC', text))


def tokens(text):
    return fidelity.normalize_text(text).split()


def token_present(literal, text):
    needle, haystack = tokens(literal), tokens(text)
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                               for i in range(len(haystack) - len(needle) + 1))


def tail_tokens(literal):
    # Last meaningful clause; commas inside numeric literals are not boundaries.
    clauses = re.split(r'[;.!?]|(?<!\d),(?!\d)', literal)
    return tokens(next((c for c in reversed(clauses) if tokens(c)), ''))


def validate_annotations(expected, annotations):
    if annotations is None:
        return
    if not isinstance(annotations, list) or len(annotations) > 100:
        raise ValueError('invalid critical annotations')
    for a in annotations:
        if (not isinstance(a, dict) or not isinstance(a.get('kind'), str)
                or not isinstance(a.get('expected'), str) or not a['expected']
                or a['expected'] not in expected
                or not isinstance(a.get('heard_forms', []), list)
                or any(not isinstance(v, str) or not v or len(v) > 1000
                       for v in a.get('heard_forms', []))):
            raise ValueError('critical annotation must reference exact expected text')


def billing_record(response, request, snapshot, key):
    # Retain the complete bounded billing object, never arbitrary response payloads.
    audit = {name: response.get(name) for name in ('id', 'provider', 'model', 'usage')}
    def secret_in(value):
        if isinstance(value, str):
            return bool(key and key in value)
        if isinstance(value, dict):
            return any(secret_in(k) or secret_in(v) for k, v in value.items())
        if isinstance(value, list):
            return any(secret_in(v) for v in value)
        return False
    if secret_in(audit):
        raise ValueError('credential in billing metadata')
    encoded = json.dumps(audit, allow_nan=False)
    if len(encoded) > 16384 or (key and key in encoded):
        raise ValueError('unsafe billing metadata')
    for field in ('id', 'provider', 'model'):
        if audit[field] is not None and (not isinstance(audit[field], str) or len(audit[field]) > 200):
            raise ValueError('invalid billing identity')
    usage = audit['usage']
    return {'request_id': audit['id'], 'provider': audit['provider'],
            'resolved_model': audit['model'], 'usage': usage}


def reconcile_billing(entry, request, snapshot):
    usage = entry['usage']
    if not isinstance(usage, dict):
        raise ValueError('missing billing usage')
    for field in ('seconds', 'cost'):
        value = usage.get(field)
        if type(value) not in (int, float, str) or len(str(value)) > 100:
            raise ValueError('invalid billing number')
        number = Decimal(str(value))
        if not number.is_finite() or not 0 <= number <= Decimal('1000000'):
            raise ValueError('invalid billing number')
    seconds = Decimal(str(entry['usage']['seconds']))
    cost = Decimal(str(entry['usage']['cost']))
    duration = Decimal(str(request['audio']['duration_seconds']))
    rate = Decimal(snapshot['usd_per_second'])
    # Local anomaly tolerances, NOT a claim about provider rounding/granularity.
    duration_tolerance = max(Decimal('0.25'), duration * Decimal('0.05'))
    expected_cost = seconds * rate
    cost_tolerance = max(Decimal('0.000001'), expected_cost * Decimal('0.10'))
    entry['billing_check'] = {'status': 'review', 'rounding': 'unknown',
        'clip_seconds': str(duration), 'catalog_usd_per_second': str(rate),
        'usage_catalog_estimated_usd': str(expected_cost),
        'duration_tolerance_seconds': str(duration_tolerance),
        'cost_tolerance_usd': str(cost_tolerance)}
    entry['actual_usd'] = str(cost)
    entry['cost_state'] = 'cost_unverified_invoice'
    entry['price_review'] = abs(cost - expected_cost) > cost_tolerance
    entry['usage_review'] = abs(seconds - duration) > duration_tolerance
    if (not 0 < seconds <= 600 or cost <= 0
            or cost > Decimal(entry['reserved_usd'])):
        raise ValueError('inconsistent billing usage')
    entry['actual_usd'] = str(cost)


def compare(expected, transcript, language, annotations=None, uncertainty=False):
    if not isinstance(transcript, str) or len(transcript) > 100000:
        raise ValueError('invalid transcript')
    normalize = fidelity.normalize_text
    ref, hyp = normalize(expected), normalize(transcript)
    raw_ref, raw_hyp = (unicodedata.normalize('NFC', s) for s in (expected, transcript))
    alerts = []
    notes = []
    orthographic_equal = ([ORTHOGRAPHY[language].get(t, t) for t in ref.split()] ==
                          [ORTHOGRAPHY[language].get(t, t) for t in hyp.split()])
    if ref != hyp:
        alerts.append('asr_mismatch')
    if signed_numbers(expected) != signed_numbers(transcript):
        alerts.append('signed_numeric_difference')
    if not hyp:
        alerts.append('empty_transcript')
    if uncertainty or re.search(r'\[(?:unclear|inaudible|uncertain)\]|\?\?', transcript, re.I):
        alerts.append('uncertain_transcript')
    if annotations is None:
        alerts.append('critical_annotations_required')
        annotations = []
    validate_annotations(expected, annotations)
    checks = []
    for a in annotations:
        kind, literal = a.get('kind'), a.get('expected')
        if not isinstance(literal, str) or literal not in expected:
            raise ValueError('critical annotation must reference exact expected text')
        result = 'review'
        if kind in ('amount', 'number'):
            parse = (lambda value: money(value, language)) if kind == 'amount' else (
                lambda value: money(value + ' EUR', language))
            target = parse(literal)
            # Surface variants must be explicit, anchored strings; never search guessed semantics.
            candidates = [literal] + a.get('heard_forms', [])
            if kind == 'number' and target is not None:
                candidates += [word for word, value in SPOKEN[language].items()
                               if ('EUR', value * 100) == target]
                candidates += [str(target[1] // 100)] if target[1] % 100 == 0 else []
            found = [s for s in candidates if isinstance(s, str) and any(numeric_surface_present(s, transcript))]
            if target is not None and found and all(parse(s) == target for s in found):
                result = 'no_anomaly_detected'
        elif kind == 'tail':
            suffix = tail_tokens(literal)
            if suffix and ([ORTHOGRAPHY[language].get(t, t) for t in tokens(transcript)[-len(suffix):]] ==
                           [ORTHOGRAPHY[language].get(t, t) for t in suffix]):
                result = 'no_anomaly_detected'
        elif kind == 'negation':
            if token_present(literal, transcript):
                result = 'no_anomaly_detected'
        elif kind in ('identifier', 'version', 'proper_name'):
            if surface_present(literal, transcript):
                result = 'no_anomaly_detected'
            elif kind == 'proper_name':
                alerts.append('ASR_name_review')
        checks.append({'kind': kind, 'expected': literal, 'status': result,
                       'time_seconds': None})
        if result == 'review':
            alerts.append('critical_' + str(kind))
    # Generic conservative loss checks supplement, never replace annotations.
    negatives = {'en': {"not", "no", "never", "don't", "cannot"},
                 'fr': {'ne', 'pas', 'jamais', 'aucun', 'sans'}}[language]
    if any((' ' + normalize(w) + ' ') in (' ' + ref + ' ') and
           (' ' + normalize(w) + ' ') not in (' ' + hyp + ' ') for w in negatives):
        alerts.append('possible_negation_loss')
    tail = ref.split()[-3:]
    if tail and hyp.split()[-len(tail):] != tail:
        alerts.append('possible_tail_omission_or_change')
    # A finite spelling variant is informational only with assessed, safe checks.
    if ref != hyp and orthographic_equal:
        notes.append('orthography_only')
        if annotations and all(c['status'] == 'no_anomaly_detected' for c in checks):
            alerts = [a for a in alerts if a not in
                      ('asr_mismatch', 'possible_tail_omission_or_change')]
    # Acronym spelling remains name review, never pronunciation evidence.
    if re.search(r'\bSAS\b', expected) and re.search(r'\bSASS\b', transcript):
        alerts.append('ASR_name_review')
    return {'status': 'REVIEW' if alerts else 'CLEAR', 'notes': notes, 'alerts': sorted(set(alerts)),
            'meaning': 'no anomaly detected is not acceptance; ASR disagreement is not synthesis failure',
            'proper_name_policy': 'spelling disagreement is not pronunciation proof',
            'raw': {'wer': fidelity.sequence_errors(raw_ref.split(), raw_hyp.split()),
                    'cer': fidelity.sequence_errors(raw_ref, raw_hyp)},
            'normalized': {'wer': fidelity.sequence_errors(ref.split(), hyp.split()),
                           'cer': fidelity.sequence_errors(ref, hyp)},
            'normalization': 'NFC raw; NFC casefold word tokens normalized; no semantic rewriting',
            'critical_checks': checks, 'time_seconds': None}


def transcript_record(response, key=None):
    text = response.get('text')
    if not isinstance(text, str) or len(text) > 100000 or (key and key in text):
        raise ValueError('invalid or credential-containing transcript')
    # Do not retain arbitrary server payloads or exception messages.
    return {'text': text, 'transcript_sha256': h.digest(text.encode()),
            'uncertainty': response.get('uncertainty') is True,
            'timestamps': None, 'provider': 'unknown', 'resolved_model': 'unknown'}


def returned_status(entry):
    return ('usage_review' if entry['usage_review'] else
            'price_review' if entry['price_review'] else 'complete')


def reconcile_retained(state, expected_hash, cache_key):
    """Explicit hash-pinned reconciliation of one retained known response."""
    with h.locked_state(state) as state:
        path = state / 'ledger.json'
        raw = h.bounded_input(path, state, 4 * 1024 * 1024)[1]
        if h.digest(raw) != expected_hash:
            raise ValueError('ledger hash precondition failed')
        ledger = json.loads(raw)
        manifest_raw = h.bounded_input(state / 'manifest.json', state, 4 * 1024 * 1024)[1]
        manifest = json.loads(manifest_raw)
        entry = ledger['entries'][cache_key]
        if entry['status'] != 'uncertain':
            raise ValueError('expected uncertain response')
        t = entry['transcript']
        if transcript_record({'text': t['text']})['transcript_sha256'] != t['transcript_sha256']:
            raise ValueError('transcript hash mismatch')
        request = next(r for r in manifest['requests'] if r['cache_key'] == cache_key)
        candidate = dict(entry)
        reconcile_billing(candidate, request, manifest['pricing_snapshot'])
        if t.get('resolved_model') not in ('unknown', MODEL) or entry.get('resolved_model') not in (None, MODEL):
            raise ValueError('unexpected resolved model')
        if not (candidate['price_review'] or candidate['usage_review']):
            raise ValueError('expected billing discrepancy')
        backup = state / ('backup-' + expected_hash)
        backup.mkdir(mode=0o700, exist_ok=False)
        hashes = {}
        for name, data in [('ledger.json', raw), ('manifest.json', manifest_raw)]:
            target = backup / name
            with target.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            target.chmod(0o400)
            hashes[name] = h.digest(data)
        h.atomic(backup / 'hashes.json', hashes)
        (backup / 'hashes.json').chmod(0o400)
        backup.chmod(0o500)
        entry.update(candidate, status=returned_status(candidate))
        # Preserve original error and billing metadata as historical evidence.
        unsent = [k for k, e in ledger['entries'].items() if e['status'] == 'reserved']
        for k in unsent:
            ledger['entries'][k]['submission_state'] = 'never_sent_audited'
        if ledger.get('reconciliation_audit'):
            ledger.setdefault('prior_reconciliation_audits', []).append(ledger['reconciliation_audit'])
        ledger['reconciliation_audit'] = {
            'ledger_sha256': expected_hash, 'backup_hashes': hashes,
            'cache_key': cache_key, 'never_sent_keys': unsent,
            'old_status': 'uncertain', 'new_status': entry['status'],
            'reported_cost_usd': candidate['actual_usd'],
            'usage_sha256': h.digest(h.canonical(entry['usage'])),
            'transcript_sha256': t['transcript_sha256'],
            'operator_uid': os.getuid(),
            'reason': 'Known response, bounded positive usage and cost within retained reserve; duration and price discrepancies remain unverified',
            'precondition': 'Operator attests legacy sequential loop stopped before these reserved entries were sent',
            'reservation_policy': 'all reservations retained', 'invoice_verified': False,
            'timestamp': time.time()}
        h.atomic(path, ledger)


def update_comparisons(ledger, manifest, checkpoint=None):
    for item in manifest['items']:
        e = ledger['entries'].get(item['cache_key'], {})
        t = e.get('transcript')
        if not t:
            continue
        if h.digest(t['text'].encode()) != t['transcript_sha256']:
            raise ValueError('retained transcript hash mismatch')
        identity = {'transcript_sha256': t['transcript_sha256'],
                    'expected_sha256': item['expected_sha256'], 'comparer_version': COMPARER,
                    'language': item['language'], 'annotations': item['critical_annotations'],
                    'uncertainty': t['uncertainty']}
        ck = h.digest(h.canonical(identity))
        ledger['comparisons'][ck] = {'identity': identity, 'result': compare(
            item['expected'], t['text'], item['language'], item['critical_annotations'], t['uncertainty'])}
        item['comparison_key'] = ck
        if checkpoint is not None:
            checkpoint()


def run(manifest, state=STATE, execute=False, max_cost=None, key_loader=None,
        transport=h.http, timeout=60, imported=None):
    if type(timeout) not in (int, float) or not 0 < timeout <= 60:
        raise ValueError('timeout must be within 0..60 seconds')
    with h.locked_state(state) as state:
        path = state / 'ledger.json'
        ledger = read_json(path, state) if path.exists() else {'entries': {}, 'comparisons': {}}
        entries = ledger['entries']
        unique = {r['cache_key']: r for r in manifest['requests']}
        if imported is not None:
            if execute:
                raise ValueError('import and execute are separate operations')
            for k, response in imported.items():
                if k not in unique or k in entries:
                    raise ValueError('import requires a planned, unused cache key')
                entries[k] = {'status': 'imported', 'reserved_usd': '0',
                              'transcript': transcript_record(response), 'source': 'local_import_unverified'}
            h.atomic(path, ledger)  # Transcript durable before any comparison.
        pending = {k: r for k, r in unique.items() if k not in entries}
        reserved = sum((Decimal(e['reserved_usd']) for e in entries.values()), Decimal(0))
        new = sum((Decimal(r['reserved_usd']) for r in pending.values()), Decimal(0))
        manifest.update(new_stt_count=len(pending), new_reserved_usd=str(new),
                        cumulative_reserved_usd=str(reserved + new))
        h.atomic(state / 'manifest.json', manifest)
        if execute:
            cap = Decimal(str(max_cost)) if max_cost is not None else Decimal('NaN')
            if not cap.is_finite() or not 0 < cap <= 1 or reserved + new > cap:
                raise ValueError('execute requires sufficient cumulative budget <= USD 1')
            if any(e['status'] not in ('complete', 'imported', 'price_review', 'usage_review')
                   and not (e['status'] == 'reserved' and e.get('submission_state') == 'never_sent_audited')
                   for e in entries.values()):
                raise ValueError('unresolved reservation: billing resolution required, no retry')
            resumable = {k: unique[k] for k, e in entries.items()
                         if k in unique and e['status'] == 'reserved'
                         and e.get('submission_state') == 'never_sent_audited'}
            pending = {**resumable, **pending}
            key = (key_loader or (lambda: h.load_key(None)))() if pending else None
            # Reserve all new requests before submission; crash cannot free a reservation.
            for k, r in pending.items():
                if k not in entries:
                    entries[k] = {'status': 'reserved', 'reserved_usd': r['reserved_usd'],
                                  'submission_state': 'never_sent_audited'}
            h.atomic(path, ledger)
            for k, r in pending.items():
                e = entries[k]
                try:
                    current, data = h.audio(r['audio']['path'], Path(r['audio']['path']).parent)
                    if current != r['audio']:
                        raise ValueError('audio changed after plan')
                    e['status'] = 'sending'
                    e['submission_state'] = 'submitted'
                    h.atomic(path, ledger)
                    body = {'model': MODEL, 'language': r['identity']['language'],
                            'response_format': 'json',
                            'input_audio': {'format': 'wav', 'data': base64.b64encode(data).decode()}}
                    response = transport(h.API + '/audio/transcriptions', body, key, timeout)
                    e['transcript'] = transcript_record(response, key)
                    h.atomic(path, ledger)  # Persist before cost validation AND expected-text access.
                    e.update(billing_record(response, r, manifest['pricing_snapshot'], key))
                    h.atomic(path, ledger)
                    for field in ('provider', 'model'):
                        value = response.get(field)
                        if isinstance(value, str) and len(value) < 200 and key not in value:
                            e['transcript']['provider' if field == 'provider' else 'resolved_model'] = value
                    if e['transcript']['resolved_model'] not in ('unknown', MODEL):
                        raise ValueError('unexpected resolved model')
                    reconcile_billing(e, r, manifest['pricing_snapshot'])
                    e.update(status=returned_status(e))
                    h.atomic(path, ledger)
                except Exception:
                    e.update(status='uncertain', error='Submission or validation unresolved; no automatic retry.')
                    update_comparisons(ledger, manifest, lambda: h.atomic(path, ledger))
                    h.atomic(path, ledger)
                    h.atomic(state / 'manifest.json', manifest)
                    raise ValueError('STT stopped; reservation retained; resolve provider billing manually') from None
        update_comparisons(ledger, manifest, lambda: h.atomic(path, ledger))
        if entries:
            h.atomic(path, ledger)
        h.atomic(state / 'manifest.json', manifest)
        return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reconcile-ledger-sha256')
    p.add_argument('--reconcile-cache-key')
    p.add_argument('--matrix', type=Path, default=MATRIX)
    p.add_argument('--input-root', type=Path, default=MATRIX.parent)
    p.add_argument('--state', type=Path, default=STATE)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--import-transcripts', type=Path, help='JSON object cache_key -> {text, uncertainty?}')
    p.add_argument('--env-file', type=Path, help='literal key file, read only during execute')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-cost-usd')
    p.add_argument('--timeout', type=float, default=60)
    args = p.parse_args(argv)
    try:
        if args.reconcile_ledger_sha256:
            if args.execute or not args.reconcile_cache_key:
                raise ValueError('reconciliation must be separate and key-specific')
            reconcile_retained(args.state, args.reconcile_ledger_sha256, args.reconcile_cache_key)
            print('Retained response reconciled; reservations preserved; invoice unverified.')
            return 0
        annotations = read_json(args.annotations, args.input_root) if args.annotations else None
        imported = read_json(args.import_transcripts, args.input_root) if args.import_transcripts else None
        manifest = plan(args.matrix, args.input_root, metadata(), annotations)
        result = run(manifest, args.state, args.execute, args.max_cost_usd,
                     key_loader=lambda: h.load_key(args.env_file),
                     timeout=args.timeout, imported=imported)
        print(json.dumps({k: result[k] for k in ('sample_count', 'unique_stt_count', 'new_stt_count',
                                                'estimated_usd', 'cumulative_reserved_usd')}))
    except Exception:
        print('STT check stopped. Inspect private manifest/ledger; no automatic retry.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
