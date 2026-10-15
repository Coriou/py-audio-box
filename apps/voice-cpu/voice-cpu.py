#!/usr/bin/env python3
"""Explicit optional CPU engines, isolated from the Qwen compatibility CLI."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import multiprocessing as mp
import tempfile
import hashlib
import math
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from tts_backends import get_backend, BackendError


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    speak = commands.add_parser('speak')
    speak.add_argument('--backend', choices=['kokoro', 'pocket-tts'], required=True)
    speak.add_argument('--assets', type=Path, required=True)
    speak.add_argument('--language', choices=['fr', 'en'], required=True)
    voice = speak.add_mutually_exclusive_group(required=True)
    voice.add_argument('--preset')
    voice.add_argument('--reference', type=Path)
    text = speak.add_mutually_exclusive_group(required=True)
    text.add_argument('--text')
    text.add_argument('--text-file', type=Path)
    speak.add_argument('--output', type=Path, required=True)
    speak.add_argument('--seed', type=int)
    speak.add_argument('--threads', type=int, default=2)
    speak.add_argument('--json', action='store_true')
    speak.add_argument('--timeout', type=float, default=600, help='Wall-clock limit including model load (seconds)')
    batch = commands.add_parser('batch', help='Sequential homogeneous jobs sharing one supervised model')
    batch.add_argument('--backend', choices=['kokoro', 'pocket-tts'], required=True)
    batch.add_argument('--assets', type=Path, required=True)
    batch.add_argument('--language', choices=['fr', 'en'], required=True)
    voice = batch.add_mutually_exclusive_group(required=True)
    voice.add_argument('--preset')
    voice.add_argument('--reference', type=Path)
    batch.add_argument('--jobs', type=Path, required=True, help='JSON list of id/text/output jobs')
    batch.add_argument('--report', type=Path, required=True, help='Fresh JSON result path')
    batch.add_argument('--seed', type=int)
    batch.add_argument('--threads', type=int, default=2)
    batch.add_argument('--timeout', type=float, default=600, help='Each job deadline including first model load')
    batch.add_argument('--continue-on-error', action='store_true')
    batch.add_argument('--json', action='store_true')
    return parser


def publish_take(output, result, payload):
    """Stage both artifacts and publish exclusively; never truncate another take."""
    import soundfile as sf
    output = Path(output)
    metadata = output.with_suffix('.meta.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_fd, audio_temp = tempfile.mkstemp(prefix='.take-', suffix='.wav', dir=output.parent)
    meta_fd, meta_temp = tempfile.mkstemp(prefix='.take-', suffix='.json', dir=output.parent)
    os.close(audio_fd)
    os.close(meta_fd)
    audio_linked = False
    try:
        sf.write(audio_temp, result.samples, result.sample_rate, subtype='PCM_24')
        Path(meta_temp).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
        os.link(audio_temp, output)
        audio_linked = True
        os.link(meta_temp, metadata)
    except BaseException:
        if audio_linked and output.exists() and os.path.samefile(audio_temp, output):
            output.unlink()
        raise
    finally:
        Path(audio_temp).unlink(missing_ok=True)
        Path(meta_temp).unlink(missing_ok=True)


def _make_backend(args):
    options = dict(assets=args.assets, language=args.language, threads=args.threads)
    if args.backend == 'pocket-tts':
        options['reference_temp_dir'] = getattr(args, '_reference_temp_dir', None)
    return get_backend(args.backend)(**options)


def _private_reference_directory():
    path = tempfile.mkdtemp(prefix='pab-pocket-worker-')
    try:
        os.chmod(path, 0o700)
    except BaseException:
        shutil.rmtree(path)  # No worker exists yet.
        raise
    return path


def _remove_reference_directory(path):
    if path is not None:
        shutil.rmtree(path)


def _runtime_versions(backend):
    versions = {}
    packages = ['numpy', 'soundfile'] + (['kokoro-onnx', 'onnxruntime', 'phonemizer', 'espeakng-loader'] if backend == 'kokoro' else ['pocket-tts', 'torch', 'tokenizers'])
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _speak(args):
    backend = None
    try:
        if args.threads < 1:
            raise BackendError('invalid_threads', 'Thread count must be positive')
        text = args.text if args.text is not None else args.text_file.read_text()
        if not text.strip():
            raise BackendError('empty_text', 'Text must contain speech')
        if args.output.exists() or args.output.with_suffix('.meta.json').exists():
            raise BackendError('output_exists', 'Choose a fresh output path to preserve prior takes')
        if args.backend == 'kokoro' and (args.reference or args.seed is not None):
            raise BackendError('unsupported_parameter', 'Kokoro requires a preset and does not implement seed')
        if args.backend == 'pocket-tts' and not args.reference:
            raise BackendError('invalid_reference', 'Pocket requires an authorized WAV reference')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        os.environ['ORT_NUM_THREADS'] = str(args.threads)
        started = time.monotonic()
        # Upstream engines can print progress to stdout; reserve it for results.
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            backend = _make_backend(args)
            result = backend.synthesise(text, preset=args.preset, reference=args.reference, seed=args.seed)
        elapsed = time.monotonic() - started
        duration = len(result.samples) / result.sample_rate
        runtime_versions = _runtime_versions(args.backend)
        payload = {'status': 'ok', 'output': str(args.output), 'sample_rate': result.sample_rate,
                   'channels': result.channels, 'duration_sec': duration,
                   'metadata': result.metadata, 'timings': {'total_seconds': elapsed, 'rtf': elapsed / duration},
                   'text_original': text, 'language': args.language, 'threads': args.threads,
                   'runtime_versions': runtime_versions,
                   'asset_manifest': backend.manifest, 'offline_requested': True,
                   'network_isolation_verified': False}
        publish_take(args.output, result, payload)
        print(json.dumps(payload, ensure_ascii=False) if args.json else str(args.output))
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, BackendError) else {'code': type(exc).__name__, 'message': str(exc)}
        print(json.dumps({'status': 'error', 'error': error}), file=sys.stderr)
        return 1
    finally:
        if backend is not None:
            backend.close()
    return 0


def _worker(args):
    raise SystemExit(_speak(args))


def _error(exc):
    return exc.as_dict() if isinstance(exc, BackendError) else {'code': type(exc).__name__, 'message': str(exc)}


def validate_batch(args):
    """Validate the whole plan before starting a model or publishing a take."""
    if args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise BackendError('invalid_limits', 'Threads and finite per-job timeout must be positive')
    if args.backend == 'kokoro' and (args.reference or args.seed is not None):
        raise BackendError('unsupported_parameter', 'Kokoro requires a preset and does not implement seed')
    if args.backend == 'pocket-tts' and not args.reference:
        raise BackendError('invalid_reference', 'Pocket requires an authorized WAV reference')
    if args.jobs.stat().st_size > 16 * 1024 * 1024:
        raise BackendError('jobs_too_large', 'Jobs JSON must not exceed 16 MiB')
    raw = json.loads(args.jobs.read_text(encoding='utf-8'))
    if not isinstance(raw, list) or not 1 <= len(raw) <= 1000:
        raise BackendError('invalid_jobs', 'Jobs must be a JSON list containing 1 to 1000 jobs')
    ids, artifacts, jobs = set(), set(), []
    for item in raw:
        if not isinstance(item, dict) or not {'id', 'text', 'output'} <= set(item) or set(item) - {'id', 'text', 'output', 'timeout'}:
            raise BackendError('invalid_job', 'Each job requires id, text, output and optional timeout')
        if not isinstance(item['id'], str) or not item['id'].strip() or item['id'] in ids:
            raise BackendError('invalid_job_id', 'Job IDs must be unique nonempty strings')
        if not isinstance(item['text'], str) or not item['text'].strip():
            raise BackendError('empty_text', 'Every job text must contain speech')
        if not isinstance(item['output'], str) or not item['output']:
            raise BackendError('invalid_output', 'Every job needs a WAV output path')
        timeout = item.get('timeout', args.timeout)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise BackendError('invalid_timeout', 'Each job timeout must be positive and finite')
        output = Path(item['output'])
        if not output.is_absolute():
            output = args.jobs.resolve().parent / output
        if os.path.lexists(output) or os.path.lexists(output.with_suffix('.meta.json')):
            raise BackendError('output_exists', 'Batch output paths must be fresh, including symlinks')
        output = output.resolve()
        if output.suffix.lower() != '.wav':
            raise BackendError('invalid_output', 'Batch outputs must end in .wav')
        pair = {output, output.with_suffix('.meta.json')}
        if artifacts & pair or any(os.path.lexists(path) for path in pair):
            raise BackendError('output_exists', 'Batch outputs and sidecars must be distinct fresh paths')
        if any(path.is_dir() for path in pair):
            raise BackendError('invalid_output', 'Output artifact cannot be a directory')
        artifacts.update(pair)
        ids.add(item['id'])
        jobs.append({'id': item['id'], 'text': item['text'], 'output': str(output), 'timeout': timeout})
    report = args.report.resolve()
    if report in artifacts or os.path.lexists(args.report) or os.path.lexists(report):
        raise BackendError('report_exists', 'Choose a fresh report path distinct from all takes')
    source = {'preset': args.preset, 'reference': None, 'reference_sha256': None}
    if args.reference:
        source.update(reference=str(args.reference.resolve()),
                      reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    return jobs, source


def _batch_worker(connection, args, source):
    """One backend instance for one homogeneous batch, never a general worker pool."""
    backend = None
    runtime_versions = None
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        os.environ['ORT_NUM_THREADS'] = str(args.threads)
        import contextlib
        while True:
            job = connection.recv()
            if job is None:
                break
            started = time.monotonic()
            cold = backend is None
            load_seconds = 0.0
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    if args.reference and hashlib.sha256(args.reference.read_bytes()).hexdigest() != source['reference_sha256']:
                        raise BackendError('reference_changed', 'Reference bytes changed during the batch')
                    if cold:
                        load_started = time.monotonic()
                        backend = _make_backend(args)
                        load_seconds = time.monotonic() - load_started
                        runtime_versions = _runtime_versions(args.backend)
                    if args.reference and hashlib.sha256(args.reference.read_bytes()).hexdigest() != source['reference_sha256']:
                        raise BackendError('reference_changed', 'Reference bytes changed during backend initialization')
                    synth_started = time.monotonic()
                    result = backend.synthesise(job['text'], preset=args.preset, reference=args.reference, seed=args.seed)
                    synth_seconds = time.monotonic() - synth_started
                elapsed = time.monotonic() - started
                duration = len(result.samples) / result.sample_rate
                metadata = dict(result.metadata)
                if args.backend == 'pocket-tts' and metadata.get('reference_sha256') != source['reference_sha256']:
                    raise BackendError('reference_identity_mismatch', 'Conditioned reference identity differs from the approved batch source')
                # Never repeat a cached backend's initial load value on warm jobs.
                metadata['load_seconds'] = load_seconds
                metadata['model_reused'] = not cold
                payload = dict(status='ok', id=job['id'], output=job['output'],
                    text_original=job['text'], language=args.language, threads=args.threads,
                    seed=args.seed, source_identity=source, backend=args.backend,
                    sample_rate=result.sample_rate, channels=result.channels, duration_sec=duration,
                    metadata=metadata, runtime_versions=runtime_versions, asset_manifest=backend.manifest, offline_requested=True,
                    network_isolation_verified=False,
                    timings=dict(cache_state='cold_model' if cold else 'warm_model',
                        backend_initialization_seconds=load_seconds, load_seconds=load_seconds, generation_seconds=synth_seconds,
                        total_seconds=elapsed, rtf=elapsed/duration))
                import soundfile as sf
                export_started = time.monotonic()
                sf.write(job['_audio_stage'], result.samples, result.sample_rate, subtype='PCM_24')
                payload['timings']['export_seconds'] = time.monotonic() - export_started
                payload['timings']['total_seconds'] = time.monotonic() - started
                payload['timings']['rtf'] = payload['timings']['total_seconds'] / duration
                Path(job['_meta_stage']).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
                connection.send(payload)
            except Exception as exc:
                connection.send(dict(status='error', id=job['id'], output=job['output'],
                    text_original=job['text'], error=_error(exc),
                    timings=dict(cache_state='cold_model' if cold else 'warm_model',
                        total_seconds=time.monotonic()-started)))
                # An inference error discards state; parent supervises shutdown/restart.
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if backend is not None:
            backend.close()
        connection.close()


def _stop_worker(process, connection, graceful=False, reference_temp_dir=None):
    if process is None:
        if connection is not None:
            connection.close()
        _remove_reference_directory(reference_temp_dir)
        return
    if process.pid is None:
        process.close()
        if connection is not None:
            connection.close()
        _remove_reference_directory(reference_temp_dir)
        return
    if graceful and process.is_alive():
        try:
            connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        process.join(2)
    if process.is_alive():
        process.terminate()
        process.join(2)
    if process.is_alive():
        process.kill()
        process.join(2)
    if process.is_alive():
        raise RuntimeError('CPU worker could not be reaped; no following job is safe')
    process.join()
    process.close()
    if connection is not None:
        connection.close()
    _remove_reference_directory(reference_temp_dir)


def _publish_staged(output, audio_stage, meta_stage):
    """Only the supervising parent publishes final paths, after a timely result."""
    linked = False
    try:
        os.link(audio_stage, output)
        linked = True
        os.link(meta_stage, output.with_suffix('.meta.json'))
    except BaseException:
        if linked and output.exists() and os.path.samefile(audio_stage, output):
            output.unlink()
        raise


def _publish_report(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix='.batch-report-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, path)
    finally:
        Path(staging).unlink(missing_ok=True)


def run_batch(args):
    batch_started = time.monotonic()
    jobs, source = validate_batch(args)
    records, process, connection = [], None, None
    generation, worker_successes = 0, 0
    staging = []
    reference_temp_dir = None
    try:
        for index, job in enumerate(jobs):
            started = time.monotonic()
            deadline = started + job['timeout']
            staging = []
            try:
                output = Path(job['output'])
                output.parent.mkdir(parents=True, exist_ok=True)
                for suffix in ('.wav', '.json'):
                    fd, path = tempfile.mkstemp(prefix='.batch-take-', suffix=suffix, dir=output.parent)
                    staging.append(path)
                    os.close(fd)
                job = dict(job, _audio_stage=staging[0], _meta_stage=staging[1])
                if process is None:
                    if args.backend == 'pocket-tts':
                        reference_temp_dir = _private_reference_directory()
                        args._reference_temp_dir = reference_temp_dir
                    connection, child = mp.get_context('spawn').Pipe()
                    try:
                        process = mp.get_context('spawn').Process(target=_batch_worker, args=(child, args, source))
                        process.start()
                    finally:
                        child.close()
                    generation += 1
                    worker_successes = 0
                connection.send(job)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Per-job deadline exceeded, including worker startup/model load')
                    if connection.poll(min(0.1, remaining)):
                        record = connection.recv()
                        if time.monotonic() > deadline:
                            raise TimeoutError('Per-job deadline expired before result receipt')
                        record['worker_generation'] = generation
                        record['worker_pid'] = process.pid if process is not None else None
                        if record['status'] == 'ok':
                            record['parent_seconds_before_publication'] = time.monotonic() - started
                            Path(staging[1]).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
                            _publish_staged(output, staging[0], staging[1])
                        break
                    if not process.is_alive():
                        raise RuntimeError(f'CPU worker exited {process.exitcode} without a result')
            except Exception as exc:
                record = dict(status='error', id=job['id'], output=job['output'],
                    text_original=job['text'], timings=dict(cache_state='cold_model' if worker_successes == 0 else 'warm_model'), error={'code': 'timeout' if isinstance(exc, TimeoutError) else type(exc).__name__, 'message': str(exc)})
            record['worker_generation'] = generation
            record['worker_pid'] = process.pid if process is not None else None
            record['job_wall_seconds'] = time.monotonic() - started
            records.append(record)
            if record['status'] == 'ok':
                worker_successes += 1
            if record['status'] != 'ok':
                _stop_worker(process, connection, reference_temp_dir=reference_temp_dir)
                process, connection, reference_temp_dir = None, None, None
                for path in staging:
                    Path(path).unlink(missing_ok=True)
                if not args.continue_on_error:
                    records.extend(dict(status='skipped', id=rest['id'], output=rest['output'],
                        text_original=rest['text'], reason='prior_job_failed') for rest in jobs[index+1:])
                    break
            for path in staging:
                Path(path).unlink(missing_ok=True)
    finally:
        try:
            _stop_worker(process, connection, graceful=True, reference_temp_dir=reference_temp_dir)
        finally:
            for path in staging:
                Path(path).unlink(missing_ok=True)
    payload = dict(schema_version=1, command='batch', backend=args.backend,
        language=args.language, source_identity=source, attempts=records,
        batch_wall_seconds=time.monotonic()-batch_started,
        status='ok' if all(r['status'] == 'ok' for r in records) else 'error')
    _publish_report(args.report, payload)
    print(json.dumps(payload, ensure_ascii=False) if args.json else str(args.report))
    return 0 if payload['status'] == 'ok' else 1


def main():
    args = build_parser().parse_args()
    if args.command == 'batch':
        try:
            return run_batch(args)
        except Exception as exc:
            print(json.dumps({'status': 'error', 'error': _error(exc)}), file=sys.stderr)
            return 1
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        print(json.dumps({'status': 'error', 'error': {'code': 'invalid_timeout', 'message': 'Timeout must be positive'}}), file=sys.stderr)
        return 1
    process, reference_temp_dir = None, None
    try:
        if args.backend == 'pocket-tts':
            reference_temp_dir = _private_reference_directory()
            args._reference_temp_dir = reference_temp_dir
        process = mp.get_context('spawn').Process(target=_worker, args=(args,))
        process.start()
        process.join(args.timeout)
        if process.is_alive():
            _stop_worker(process, None, reference_temp_dir=reference_temp_dir)
            process, reference_temp_dir = None, None
            print(json.dumps({'status': 'error', 'error': {'code': 'timeout', 'message': 'CPU inference exceeded deadline; worker terminated'}}), file=sys.stderr)
            return 1
        return process.exitcode or 0
    finally:
        # If reap fails, _stop_worker raises before deleting reference snapshots.
        _stop_worker(process, None, reference_temp_dir=reference_temp_dir)



if __name__ == '__main__':
    raise SystemExit(main())
