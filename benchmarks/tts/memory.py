#!/usr/bin/env python3
"""Sequential resident CLI batches with sampled Linux owned-group memory."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from warm import inspect_audio, save_report
from run import environment

ROOT = Path(__file__).resolve().parents[2]


def group_memory(group, proc=Path('/proc'), getpgid=os.getpgid):
    """Reject partial group totals. A process disappearing during a read is missing."""
    totals = dict(rss=0, pss=0, uss=0)
    members = 0
    try:
        entries = list(proc.iterdir())
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                member = getpgid(int(entry.name)) == group
            except ProcessLookupError:
                continue
            except OSError:
                return None
            if not member:
                continue
            members += 1
            fields = {}
            for line in (entry / 'smaps_rollup').read_text().splitlines():
                bits = line.split()
                if bits and bits[0] in ('Rss:', 'Pss:', 'Private_Clean:', 'Private_Dirty:'):
                    if len(bits) != 3 or bits[2] != 'kB':
                        return None
                    fields[bits[0][:-1]] = int(bits[1]) * 1024
            if len(fields) != 4:
                return None
            totals['rss'] += fields['Rss']
            totals['pss'] += fields['Pss']
            totals['uss'] += fields['Private_Clean'] + fields['Private_Dirty']
    except (OSError, ValueError):
        return None
    return totals if members else None


def execute(command, timeout, stdout, stderr):
    started = time.monotonic()
    peaks = dict(rss=None, pss=None, uss=None)
    samples = missing = 0
    with Path(stdout).open('w') as out, Path(stderr).open('w') as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            while child.poll() is None:
                reading = group_memory(child.pid)
                if reading is None:
                    missing += 1
                else:
                    samples += 1
                    for key, value in reading.items():
                        peaks[key] = max(peaks[key] or 0, value)
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
                wall_sec=time.monotonic()-started, memory_samples=samples,
                memory_missing_samples=missing, memory_complete=bool(samples and not missing),
                observed_complete_group_peak_bytes=peaks if samples else dict.fromkeys(peaks),
                memory_scope='Linux owned process group smaps_rollup sum; PSS proportional shared pages, USS private clean plus dirty; RSS may double count shared pages; observed maxima of sequential non-atomic group sums, not guaranteed simultaneous peaks or lower bounds; not container or host peaks; missing intervals may conceal higher peaks; partial snapshots discarded')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--scripts', type=Path, required=True)
    p.add_argument('--language', choices=('fr', 'en'), required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--image-digest', required=True)
    p.add_argument('--timeout-sec', type=float, default=300)
    p.add_argument('--app', type=Path, default=ROOT / 'apps/voice-cpu/voice-cpu.py')
    a = p.parse_args(argv)
    if a.threads < 1 or a.repeat < 1 or not math.isfinite(a.timeout_sec) or a.timeout_sec <= 0:
        p.error('threads, repeat, and timeout must be positive')
    items = json.loads(a.scripts.read_text())['items']
    if not isinstance(items, list) or any(not isinstance(i, dict) or not isinstance(i.get('id'), str) or not isinstance(i.get('text'), str) or not i['text'].strip() or i.get('language') not in ('fr', 'en') for i in items):
        p.error('scripts require id, language and nonempty text')
    if len({i['id'] for i in items}) != len(items):
        p.error('script IDs must be unique')
    selected = [i for i in items if i['language'] == a.language]
    if not selected:
        p.error('no scripts for selected language')
    manifest_path = a.assets / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, environment=environment(a.image_digest),
                  allocator_environment={name: os.environ.get(name) for name in ('MALLOC_MMAP_THRESHOLD_', 'MALLOC_ARENA_MAX', 'MALLOC_TRIM_THRESHOLD_')},
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('warm.py', 'run.py')},
                  app_sha256=hashlib.sha256(a.app.read_bytes()).hexdigest(),
                  scripts_sha256=hashlib.sha256(a.scripts.read_bytes()).hexdigest(),
                  asset_manifest=manifest, asset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  settings=dict(language=a.language, threads=a.threads, repeat=a.repeat, timeout_sec=a.timeout_sec),
                  policy='sequential fresh CLI batch process per repeat; supplied script order and exact text; OS page cache uncontrolled; no acoustic acceptance',
                  sessions=[], attempts=[])
    save_report(a.out, report)
    for repeat in range(1, a.repeat+1):
        label = f'{a.language}-t{a.threads}-r{repeat}'
        jobs = [dict(id=i['id'], text=i['text'], output=str(a.out / f'{label}-{n:03d}.wav'), timeout=a.timeout_sec) for n, i in enumerate(selected)]
        jobs_path = a.out / f'{label}.jobs.json'
        jobs_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False)+'\n')
        command = [sys.executable, str(a.app), 'batch', '--backend', 'kokoro', '--assets', str(a.assets.resolve()), '--language', a.language, '--preset', 'ff_siwis' if a.language == 'fr' else 'af_heart', '--threads', str(a.threads), '--jobs', str(jobs_path), '--report', str(a.out / f'{label}.batch.report.json'), '--json', '--continue-on-error']
        try:
            result = execute(command, a.timeout_sec, a.out / f'{label}.stdout.json', a.out / f'{label}.stderr.log')
        except Exception as exc:
            result = dict(exit_code=None, timed_out=False, error=str(exc))
        result.update(repeat=repeat, group=label)
        report['sessions'].append(result)
        save_report(a.out, report)
        for item, job in zip(selected, jobs):
            output = Path(job['output'])
            record = dict(id=item['id'], text=item['text'], language=item['language'], repeat=repeat, audio=output.name)
            try:
                metadata = json.loads(output.with_suffix('.meta.json').read_text())
                if metadata.get('status') != 'ok' or metadata.get('text_original') != item['text']:
                    raise ValueError('Missing successful metadata or original script mismatch')
                record.update(metadata=metadata)
                record.update(inspect_audio(output))
                record['status'] = 'ok'
            except Exception as exc:
                record.update(status='error', error=str(exc))
            report['attempts'].append(record)
            save_report(a.out, report)
    return 0 if all(x['status'] == 'ok' for x in report['attempts']) and all(x['exit_code'] == 0 and not x['timed_out'] for x in report['sessions']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
