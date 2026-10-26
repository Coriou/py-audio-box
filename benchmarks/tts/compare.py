#!/usr/bin/env python3
"""Build a local performance summary and blind listening kit from cold TTS runs."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import random
import shutil
import statistics


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def stats(values):
    values = [v for v in values if finite(v)]
    return dict(count=len(values), median=statistics.median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


def no_symlinks(path):
    """Check the supplied spelling before resolution, including every ancestor."""
    path = Path(path)
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'symlinks are forbidden: {current}')
    return absolute


def local_audio(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('audio must be a relative path without parent traversal')
    path = no_symlinks(root / name)
    if not path.is_file() or path.suffix.lower() != '.wav':
        raise ValueError('missing regular WAV file')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('audio escapes run directory')
    return path


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args(argv)
    try:
        no_symlinks(args.out)
        for path in args.run:
            no_symlinks(path / 'results.json')
        if args.reference:
            no_symlinks(args.reference)
    except ValueError as exc:
        parser.error(str(exc))
    runs = [(path.resolve(), json.loads((path / 'results.json').read_text())) for path in args.run]
    hashes = [data.get('corpus_sha256') for _, data in runs]
    if not hashes or any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes) or len(set(hashes)) != 1:
        parser.error('runs must contain the same valid corpus_sha256')
    backends, scripts, issues = {}, {}, []
    for root, data in runs:
        for source in data['attempts']:
            record = dict(source)
            backend, item, repeat = record.get('backend'), record.get('id'), record.get('repeat')
            if not isinstance(backend, str) or not backend or not isinstance(item, str) or not item or not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
                parser.error('invalid backend, item id or repeat')
            text = record.get('text')
            critical = record.get('critical_literals', [])
            if not isinstance(text, str) or not text or not isinstance(critical, list) or any(not isinstance(v, str) for v in critical):
                parser.error('invalid text or critical literals')
            identity = (text, tuple(critical))
            if item in scripts and scripts[item] != identity:
                parser.error('text or critical literals differ for the same item id')
            scripts[item] = identity
            key = (item, repeat)
            attempts = backends.setdefault(backend, {})
            if key in attempts:
                parser.error('duplicate backend/item/repeat')
            record['_root'] = root
            attempts[key] = record
    if len(backends) < 2:
        parser.error('at least two backends are required')
    coverage = set().union(*(set(a) for a in backends.values()))
    if not coverage:
        parser.error('no attempts')
    summary = dict(schema_version=1, corpus_sha256=hashes[0], human_listening='UNMEASURED',
        overall_winner=None, engine_promotion=False,
        interpretation='Performance observations only. Fair narration comparison of the same scripts; preset narration and reference-conditioned speech do not establish a comparative cloning result. Audio quality, exact spoken text and speaker fidelity require human listening.',
        cold_policy='Each attempt is a fresh process; filesystem cache state is uncontrolled. Supplied run policy and environments are retained as evidence, not independently verified.',
        rss_scope='Sampled owned process-group RSS sum; shared pages may be counted multiple times; sampled lower bound, not container or host peak.',
        environments=[dict(run=str(root), environment=data.get('environment'), cold_warm_policy=data.get('cold_warm_policy')) for root, data in runs],
        coverage_match=True, coverage_basis='Union of supplied item/repeat keys; jointly omitted attempts cannot be detected without an external plan.', issues=issues, backends={})
    playable = []
    for backend, attempts in sorted(backends.items()):
        missing = sorted(coverage - set(attempts))
        if missing:
            summary['coverage_match'] = False
        rows, metric_values = [], {k: [] for k in ('cold_wall_sec', 'audio_duration_sec', 'cold_wall_rtf', 'sampled_group_rss_peak_bytes')}
        timing_values = {}
        for key, record in sorted(attempts.items()):
            row = {k: v for k, v in record.items() if k != '_root'}
            errors = []
            if record.get('status') != 'ok':
                errors.append(str(record.get('error', 'attempt not successful')))
            for field in ('wall_sec', 'duration_sec'):
                if not finite(record.get(field)) or record[field] <= 0:
                    errors.append('missing or invalid ' + field)
            rss = record.get('sampled_group_rss_peak_bytes')
            if rss is not None and not finite(rss):
                errors.append('invalid sampled_group_rss_peak_bytes')
            try:
                audio = local_audio(record['_root'], record.get('audio'))
                row['wav_sha256'] = digest(audio)
            except (ValueError, OSError) as exc:
                errors.append(str(exc))
                audio = None
            row['missing_metrics'] = [field for field in ('wall_sec', 'duration_sec', 'sampled_group_rss_peak_bytes') if record.get(field) is None]
            row['comparison_errors'] = errors
            rows.append(row)
            if errors:
                issues.append(dict(backend=backend, id=key[0], repeat=key[1], errors=errors))
            if record.get('status') == 'ok' and audio is not None:
                playable.append((backend, record, audio, row['wav_sha256']))
            if record.get('status') == 'ok' and finite(record.get('wall_sec')) and finite(record.get('duration_sec')) and record['duration_sec'] > 0:
                metric_values['cold_wall_sec'].append(record['wall_sec'])
                metric_values['audio_duration_sec'].append(record['duration_sec'])
                metric_values['cold_wall_rtf'].append(record['wall_sec'] / record['duration_sec'])
                if finite(rss):
                    metric_values['sampled_group_rss_peak_bytes'].append(rss)
            metadata = record.get('metadata') or {}
            for scope, timing in [('timings', metadata.get('timings') or {}), ('backend_metadata', metadata.get('metadata') or {})]:
                for name, value in timing.items():
                    if ('seconds' in name or name.endswith('_sec')) and finite(value):
                        timing_values.setdefault(scope + '.' + name, []).append(value)
        summary['backends'][backend] = dict(attempts=rows, missing=[dict(id=i, repeat=r) for i, r in missing],
            metrics={k: stats(v) for k, v in metric_values.items()}, metric_population='Successful attempts with finite source measurements; counts expose missing metrics. RSS availability is reported per attempt.', backend_specific_timings={k: stats(v) for k, v in timing_values.items()},
            timing_interpretation='Backend-defined timing boundaries; conditioning, generation and model initialization may overlap or differ and are not interchangeable across engines.')
    reference = None
    if args.reference:
        reference = local_audio(args.reference.parent.resolve(), args.reference.name)
    no_symlinks(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    random.SystemRandom().shuffle(playable)
    cards, mapping = [], []
    with (args.out / 'rating-worksheet.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['neutral_id', 'naturalness_1_to_5', 'exact_text_errors', 'critical_values_errors', 'speaker_fidelity_1_to_5_or_NA_after_unblind', 'notes'])
        for number, (backend, record, audio, sha) in enumerate(playable, 1):
            alias = f'sample{number:03d}'
            filename = alias + '.wav'
            shutil.copyfile(audio, args.out / filename)
            mapping.append(dict(neutral_id=alias, backend=backend, id=record['id'], repeat=record['repeat'], source_audio=str(audio), audio=filename, sha256=sha))
            writer.writerow([alias, '', '', '', '', ''])
            critical = ', '.join(record.get('critical_literals', []))
            cards.append(f'<section><h2>{alias}</h2><p>{html.escape(record["text"])}</p><p>Critical values: {html.escape(critical)}</p><audio controls preload="none" src="{filename}"></audio></section>')
    reference_html = ''
    if reference:
        shutil.copyfile(reference, args.out / 'reference.wav')
        summary['reference'] = dict(audio='reference.wav', sha256=digest(reference), meaning='Listening reference; shared speaker identity across engines is not assumed.')
        reference_html = '<h2>Listening reference</h2><p>This reference does not imply that every sample targets the same speaker identity.</p><audio controls preload="none" src="reference.wav"></audio>'
    write_json(args.out / 'blind-map.json', mapping)
    write_json(args.out / 'summary.json', summary)
    (args.out / 'listen.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Blind listening</title><style>body{max-width:850px;margin:3rem auto;padding:1rem;font:18px sans-serif}section{padding:1rem 0;border-top:1px solid #ccc}audio{width:100%}</style><h1>Blind listening</h1><p>Human listening is UNMEASURED. Rate naturalness from 1 to 5 and record exact-text and critical-value errors in rating-worksheet.csv. Leave speaker fidelity unrated until you choose to unblind; N/A may apply to preset narration. Do not open blind-map.json or summary.json until ratings are complete. No quality winner is inferred.</p>' + reference_html + ''.join(cards) + '</html>')
    print(args.out / 'summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
