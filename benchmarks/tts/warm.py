#!/usr/bin/env python3
"""Same-input Kokoro cold-process versus resident-batch CPU benchmark."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# The historical cold harness stays intact; reuse its corpus/environment contract.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import environment, load_corpus

ROOT = Path(__file__).resolve().parents[2]


def group_rss_bytes(group):
    """Sample Linux RSS of this owned group; shared mappings can count twice."""
    if not Path('/proc').exists():
        return None
    total = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) != group:
                continue
            line = next(line for line in (entry / 'status').read_text().splitlines()
                        if line.startswith('VmRSS:'))
            total += int(line.split()[1]) * 1024
        except (OSError, StopIteration, ValueError):
            continue
    return total


def execute(command, timeout, stdout, stderr):
    started = time.monotonic()
    samples, peak = 0, None
    with Path(stdout).open('w') as out, Path(stderr).open('w') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            while child.poll() is None:
                rss = group_rss_bytes(child.pid)
                if rss is not None:
                    peak = max(peak or 0, rss)
                    samples += 1
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    child.wait(timeout=min(0.05, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            for sig, wait in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    pass
            child.wait(timeout=5)
    return dict(exit_code=child.returncode, timed_out=timed_out,
                wall_sec=time.monotonic()-started, sampled_group_rss_peak_bytes=peak,
                rss_samples=samples, rss_scope='Linux owned process group RSS sum; shared pages may be counted more than once; sampled lower bound, not container/host peak')


def inspect_audio(path):
    import numpy as np
    import soundfile as sf
    samples, rate = sf.read(path, dtype='float64', always_2d=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Output audio is empty or nonfinite')
    return dict(wav_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                decoded_pcm_sha256=hashlib.sha256(samples.astype('<f8').tobytes()).hexdigest(),
                frames=len(samples), sample_rate=rate, channels=samples.shape[1],
                duration_sec=len(samples)/rate, finite=True)


def compare_audio(cold, warm):
    import numpy as np
    import soundfile as sf
    a, ar = sf.read(cold, dtype='float64', always_2d=True)
    b, br = sf.read(warm, dtype='float64', always_2d=True)
    same_shape = ar == br and a.shape == b.shape
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return dict(same_rate_shape=same_shape, both_finite=finite,
                pcm_exact_equal=bool(same_shape and finite and np.array_equal(a, b)),
                pcm_max_abs_difference=float(np.max(np.abs(a-b))) if same_shape and finite and a.size else None,
                meaning='Numerical reproducibility only; human listening and exact spoken-text acceptance are separate')


def inspect_attempt(item, output, process, *, mode, threads, repeat):
    record = dict(id=item['id'], language=item['language'], text=item['text'],
                  mode=mode, threads=threads, repeat=repeat, audio=output.name)
    try:
        if mode == 'cold' and (process['timed_out'] or process['exit_code'] != 0):
            raise ValueError('Cold process timed out or exited unsuccessfully')
        payload = json.loads(output.with_suffix('.meta.json').read_text())
        if payload.get('status') != 'ok' or payload.get('text_original') != item['text']:
            raise ValueError('Missing successful metadata or original script mismatch')
        record.update(inspect_audio(output))
        record.update(status='ok', metadata=payload,
                      timings=payload.get('timings'), rtf_process=process['wall_sec']/record['duration_sec'] if mode == 'cold' else None)
    except Exception as exc:
        record.update(status='error', error=str(exc), process_timed_out=process.get('timed_out'))
    return record


def save_report(out, report):
    temporary = out / '.results.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, out / 'results.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='one pinned Kokoro FP32 bundle directory')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('corpus.json'))
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--preset-fr', default='ff_siwis')
    parser.add_argument('--preset-en', default='af_heart')
    parser.add_argument('--timeout-sec', type=float, default=600)
    parser.add_argument('--image-digest', required=True, help='actual image identity supplied by operator')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--app', type=Path, default=ROOT / 'apps/voice-cpu/voice-cpu.py')
    args = parser.parse_args(argv)
    # Batch resolves relative job outputs against the jobs-file directory.
    # Store absolute output paths so a relative --out is not applied twice.
    args.out = args.out.resolve()
    if args.repeat < 1 or not math.isfinite(args.timeout_sec) or args.timeout_sec <= 0 or any(t < 1 for t in args.threads) or len(set(args.threads)) != len(args.threads):
        parser.error('repeat, timeout, and distinct thread counts must be positive')
    corpus = load_corpus(args.corpus)
    if set(args.ids) - {i['id'] for i in corpus}:
        parser.error('unknown corpus IDs')
    items = [i for i in corpus if i['id'] in args.ids]
    manifest_path = args.assets / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('backend') != 'kokoro' or 'fp32' not in str(manifest.get('checkpoint', '')).lower():
        parser.error('a pinned Kokoro FP32 asset manifest is required')
    # Record FP32 as an operator-controlled comparison constraint, never infer it from speed.
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, environment=environment(args.image_digest), backend='kokoro',
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        app_sha256=hashlib.sha256(args.app.read_bytes()).hexdigest(),
        asset_manifest=manifest, asset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        settings=dict(threads=args.threads, repeat=args.repeat, preset_fr=args.preset_fr,
                      preset_en=args.preset_en, per_job_timeout_sec=args.timeout_sec),
        policy='sequential; cold first per group then resident warm batch; identical scripts/presets/assets; OS page cache uncontrolled; no acoustic acceptance',
        sessions=[], attempts=[], comparisons=[], groups=[])
    save_report(args.out, report)
    for threads in args.threads:
        for language in ('fr', 'en'):
            selected = [i for i in items if i['language'] == language]
            if not selected:
                continue
            preset = args.preset_fr if language == 'fr' else args.preset_en
            for repeat in range(1, args.repeat+1):
                group = f'{language}-t{threads}-r{repeat}'
                base = [sys.executable, str(args.app)]
                common = ['--backend', 'kokoro', '--assets', str(args.assets), '--language', language,
                          '--preset', preset, '--threads', str(threads)]
                cold_records = []
                cold_wall = 0
                for index, item in enumerate(selected):
                    output = args.out / f'{group}-{index:03d}-cold.wav'
                    label = output.stem
                    result = execute(base+['speak']+common+['--text', item['text'], '--output', str(output),
                        '--json', '--timeout', str(args.timeout_sec)], args.timeout_sec+10,
                        args.out / f'{label}.stdout.json', args.out / f'{label}.stderr.log')
                    cold_wall += result['wall_sec']
                    result.update(mode='cold', group=group, id=item['id'])
                    report['sessions'].append(result)
                    attempt = inspect_attempt(item, output, result, mode='cold', threads=threads, repeat=repeat)
                    cold_records.append(attempt)
                    report['attempts'].append(attempt)
                    save_report(args.out, report)
                jobs = [dict(id=item['id'], text=item['text'], output=str(args.out / f'{group}-{index:03d}-warm.wav'),
                             timeout=args.timeout_sec) for index, item in enumerate(selected)]
                jobs_path = args.out / f'{group}.jobs.json'
                jobs_path.write_text(json.dumps(jobs, indent=2)+'\n')
                result = execute(base+['batch']+common+['--jobs', str(jobs_path), '--report', str(args.out / f'{group}-batch.report.json'), '--json', '--continue-on-error'],
                    args.timeout_sec*(len(jobs)+1)+10, args.out / f'{group}-warm.stdout.json', args.out / f'{group}-warm.stderr.log')
                result.update(mode='warm_batch', group=group)
                try:
                    result['metadata'] = json.loads((args.out / f'{group}-warm.stdout.json').read_text())
                except (OSError, ValueError) as exc:
                    result['metadata_error'] = str(exc)
                report['sessions'].append(result)
                warm_records = []
                for item, job, cold in zip(selected, jobs, cold_records):
                    warm = inspect_attempt(item, Path(job['output']), result, mode='warm', threads=threads, repeat=repeat)
                    warm_records.append(warm)
                    report['attempts'].append(warm)
                    comparison = dict(id=item['id'], group=group, cold_status=cold['status'], warm_status=warm['status'])
                    if cold['status'] == warm['status'] == 'ok':
                        comparison.update(compare_audio(args.out / cold['audio'], Path(job['output'])))
                    report['comparisons'].append(comparison)
                result['wall_sec_per_requested_job'] = result['wall_sec']/len(jobs)
                successful = all(a['status'] == 'ok' for a in cold_records+warm_records) and result['exit_code'] == 0 and not result['timed_out']
                report['groups'].append(dict(group=group, threads=threads, language=language, requested_jobs=len(jobs),
                    cold_process_wall_sum_sec=cold_wall, warm_batch_process_wall_sec=result['wall_sec'],
                    process_wall_speedup=cold_wall/result['wall_sec'] if successful and result['wall_sec'] > 0 else None))
                save_report(args.out, report)
                print(f'{group}: {sum(a["status"] == "ok" for a in report["attempts"])} successful attempts so far', flush=True)
    return 0 if all(a['status'] == 'ok' for a in report['attempts']) and all(s['exit_code'] == 0 and not s['timed_out'] for s in report['sessions']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
