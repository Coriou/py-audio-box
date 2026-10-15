#!/usr/bin/env python3
"""Cold-process CPU benchmark. No inference downloads; all attempts are recorded."""
import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def environment(image_digest=None):
    packages = {}
    for name in ('numpy', 'torch', 'torchaudio', 'soundfile', 'kokoro', 'onnxruntime', 'pocket-tts', 'qwen-tts', 'faster-whisper'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpuinfo = read_optional('/proc/cpuinfo') or ''
    model = next((line.split(':', 1)[1].strip() for line in cpuinfo.splitlines()
                  if line.startswith('model name')), None)
    return dict(platform=platform.platform(), machine=platform.machine(),
                python=sys.version.split()[0], logical_cpus=os.cpu_count(), cpu_model=model,
                cgroup_cpu_max=read_optional('/sys/fs/cgroup/cpu.max'),
                cgroup_memory_max=read_optional('/sys/fs/cgroup/memory.max'),
                cgroup_cpuset=read_optional('/sys/fs/cgroup/cpuset.cpus.effective'),
                meminfo=read_optional('/proc/meminfo'), packages=packages,
                image_digest=image_digest, image_digest_source='operator' if image_digest else 'not_supplied')


def execute(command, timeout_sec, stdout_path, stderr_path):
    """Bounded owned process group: terminate + kill + reap after timeout/error.

    File-backed output prevents large model logs filling host memory. No shell is used.
    """
    start = time.monotonic()
    with open(stdout_path, 'w') as out, open(stderr_path, 'w') as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Reap descendants even if their parent exits first; this is our own group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if timed_out:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    return dict(exit_code=process.returncode, timed_out=timed_out,
                wall_sec=round(time.monotonic() - start, 6))


def load_corpus(path):
    payload = json.loads(Path(path).read_text())
    ids = set()
    for item in payload['items']:
        if item['id'] in ids or item['language'] not in ('fr', 'en') or not item['text'].strip():
            raise ValueError('invalid corpus item')
        ids.add(item['id'])
        for literal in item['critical_literals']:
            if literal not in item['text']:
                raise ValueError(f"critical literal absent from {item['id']}: {literal}")
    return payload['items']


def write_reports(out, records, env, corpus_hash, blind_seed):
    (out / 'results.json').write_text(json.dumps(dict(schema_version=1, environment=env,
        corpus_sha256=corpus_hash, attempts=records, cold_warm_policy='each attempt is a new cold process; filesystem cache state uncontrolled'), indent=2)+'\n')
    fields = ['id', 'backend', 'repeat', 'status', 'exit_code', 'timed_out', 'wall_sec',
              'duration_sec', 'rtf_wall', 'sample_rate', 'channels', 'seed', 'temperature',
              'cache_state', 'audio', 'error']
    with (out / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    successful = [r for r in records if r['status'] == 'ok']
    random.Random(blind_seed).shuffle(successful)
    mapping, cards = [], []
    for i, record in enumerate(successful, 1):
        alias = f'Sample {i:03d}'
        item = dict(alias=alias, id=record['id'], backend=record['backend'], audio=record['audio'])
        mapping.append(item)
        # Neutral filenames prevent engine names being exposed by the audio control.
        blind_audio = out / 'blind' / f'{i:03d}.wav'
        blind_audio.parent.mkdir(exist_ok=True)
        import shutil
        shutil.copyfile(out / record['audio'], blind_audio)
        cards.append(f'<section><h2>{alias}</h2><p>{html.escape(record["text"])}</p>'
                     f'<audio controls preload="none" src="blind/{i:03d}.wav"></audio>'
                     '<p>Record naturalness (1–5), voice consistency (1–5), exact text, names/quantities, '
                     'pauses and artifacts in your review sheet.</p></section>')
    (out / 'blind-map.json').write_text(json.dumps(mapping, indent=2)+'\n')
    (out / 'listen.html').write_text('<!doctype html><meta charset="utf-8"><title>Blind TTS listening</title>'
        '<style>body{max-width:850px;margin:3rem auto;font:18px sans-serif}section{padding:1rem;border-top:1px solid #ccc}audio{width:100%}</style>'
        '<h1>Blind TTS listening</h1><p>Human ratings are pending. Do not open blind-map.json before rating. '
        'The visible scripts are approved reference text; ASR alone cannot establish exact fidelity.</p>'+''.join(cards))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', action='append', choices=['kokoro', 'pocket-tts'], required=True)
    parser.add_argument('--assets', type=Path, required=True, help='preprovisioned root with backend subdirectories')
    parser.add_argument('--preset', help='explicit single-preset override for bounded single-language runs')
    parser.add_argument('--preset-fr', default='ff_siwis')
    parser.add_argument('--preset-en', default='af_heart')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('corpus.json'))
    parser.add_argument('--ids', nargs='+', help='bounded pilot item IDs')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--timeout-sec', type=float, default=600)
    parser.add_argument('--image-digest')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--app', type=Path, default=ROOT / 'apps/voice-cpu/voice-cpu.py')
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.timeout_sec <= 0 or args.threads < 1:
        parser.error('repeat, threads and timeout must be positive')
    if 'pocket-tts' in args.backend and not args.reference:
        parser.error('pocket-tts requires an approved local --reference WAV')
    items = load_corpus(args.corpus)
    if args.ids:
        known = {item['id'] for item in items}
        if set(args.ids) - known:
            parser.error('unknown corpus IDs')
        items = [item for item in items if item['id'] in args.ids]
    args.out.mkdir(parents=True, exist_ok=False)
    env = environment(args.image_digest)
    corpus_hash = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    records = []
    for item in items:
        for backend in args.backend:
            for repeat in range(args.repeat):
                attempt = f"{item['id']}-{backend}-{repeat+1:02d}"
                wav = args.out / f'{attempt}.wav'
                stdout, stderr = args.out / f'{attempt}.stdout.json', args.out / f'{attempt}.stderr.log'
                command = [sys.executable, str(args.app), 'speak', '--backend', backend,
                    '--assets', str(args.assets / backend), '--language', item['language'],
                    '--text', item['text'], '--output', str(wav), '--json',
                    '--threads', str(args.threads)]
                preset = args.preset or (args.preset_fr if item['language'] == 'fr' else args.preset_en)
                command += ['--preset', preset] if backend == 'kokoro' else [
                    '--reference', str(args.reference), '--seed', str(args.seed + repeat)]
                record = dict(id=item['id'], text=item['text'], critical_literals=item['critical_literals'],
                    backend=backend, preset=preset if backend == 'kokoro' else None, repeat=repeat+1, seed=None if backend == 'kokoro' else args.seed+repeat, temperature=None,
                    cache_state='cold_process_filesystem_cache_uncontrolled', audio=wav.name)
                try:
                    record.update(execute(command, args.timeout_sec, stdout, stderr))
                    if record['timed_out']:
                        raise RuntimeError('process timeout; owned process group terminated and reaped')
                    if record['exit_code'] != 0:
                        raise RuntimeError(f"CLI exit {record['exit_code']}; inspect local stderr log")
                    payload = json.loads(stdout.read_text())
                    import soundfile as sf
                    info = sf.info(wav)
                    if info.frames <= 0:
                        raise ValueError('empty output audio')
                    record.update(status='ok', metadata=payload, duration_sec=info.duration,
                        sample_rate=info.samplerate, channels=info.channels,
                        rtf_wall=record['wall_sec']/info.duration)
                except Exception as exc:
                    record.update(status='error', error=str(exc))
                records.append(record)
                write_reports(args.out, records, env, corpus_hash, args.seed)
                print(f'{attempt}: {record["status"]}', flush=True)
    return 0 if all(r['status'] == 'ok' for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
