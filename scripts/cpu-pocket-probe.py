#!/usr/bin/env python3
"""Contained actual Pocket CPU measurement. Parent corroborates exit and image identity."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
import traceback
import wave

INPUTS = Path('/inputs')
ASSETS = INPUTS
PLAN = INPUTS
BUNDLE = Path('/tmp/bundle')
REFERENCE = Path('/tmp/reference.wav')
ATTEMPT = Path('/tmp/attempt')
CONDITIONING = ATTEMPT / 'conditioning'
MODEL_OWNER = 10001
CHILD_UID = 20000
AS_BYTES = 3 * 1024**3
FILE_BYTES = 1024**2
SECONDS = 180
VERSIONS = {'pocket-tts': '3.3.0', 'torch': '2.10.0+cpu', 'tokenizers': '0.23.3'}



def small_json(path, cap=65536):
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ValueError('input_json_budget')
    return json.loads(raw)


def copy_exact(source, destination, size, digest):
    if not 0 < size <= 220 * 1024**2:
        raise ValueError('model_file_budget')
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError('model_source_size')
        out = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            count = 0
            hashed = hashlib.sha256()
            while count < size:
                block = os.read(fd, min(1024**2, size - count))
                if not block:
                    raise ValueError('short_model_input')
                count += len(block)
                hashed.update(block)
                view = memoryview(block)
                while view:
                    n = os.write(out, view)
                    if n <= 0:
                        raise OSError('short_model_write')
                    view = view[n:]
            if os.read(fd, 1) or hashed.hexdigest() != digest:
                raise ValueError('model_hash_or_size')
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('model_input_changed')
            os.fchmod(out, 0o444)
            os.fchown(out, MODEL_OWNER, MODEL_OWNER)
        finally:
            os.close(out)
    finally:
        os.close(fd)


def stage_reference(entry):
    fd = os.open(INPUTS / 'reference.wav', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != entry['bytes'] or not 44 < info.st_size <= FILE_BYTES:
            raise ValueError('reference_input_size')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(FILE_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('reference_input_identity')
    fd = os.open(REFERENCE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    held = None
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw); stream.flush()
        held = os.open(REFERENCE, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        os.fchmod(fd, 0o600)
        os.fchown(fd, MODEL_OWNER, MODEL_OWNER)
    except BaseException:
        if held is not None: os.close(held)
        raise
    finally:
        os.close(fd)
    return held


def dropped_status():
    fields = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if os.getresuid() != (CHILD_UID,) * 3 or os.getresgid() != (CHILD_UID,) * 3 or os.getgroups():
        raise ValueError('permanent_child_identity')
    if fields['NoNewPrivs'].strip() != '1' or any(int(fields[k].strip(), 16) for k in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb')):
        raise ValueError('child_retained_capabilities')
    return {k: fields[k].strip() for k in ('Uid', 'Gid', 'Groups', 'CapInh', 'CapPrm', 'CapEff', 'CapAmb', 'NoNewPrivs')}


def expired(*unused):
    raise TimeoutError('engine_deadline')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='Bonjour.')
    parser.add_argument('--worker-image', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    stage = 'root_setup'
    reference_fd = None
    report = {'status': 'failed', 'evidence': 'single actual engine run only; no installed/API/clone release qualification', 'planned_parent_image': args.worker_image, 'image_identity_corroborated_by_script': False}
    try:
        if sys.platform != 'linux' or os.getresuid() != (0, 0, 0):
            raise ValueError('explicit_linux_root_setup_required')
        os.umask(0o077)
        inputs = small_json(PLAN / 'probe-input.json')
        entry = inputs['reference']
        descriptor = dict(inputs['context']['execution']['descriptor'])
        manifest_path = PLAN / 'manifest.json'
        with manifest_path.open('rb') as manifest_stream:
            manifest_raw = manifest_stream.read(65537)
        if len(manifest_raw) > 65536 or hashlib.sha256(manifest_raw).hexdigest() != descriptor['manifest_sha256']:
            raise ValueError('runtime_manifest_identity')
        manifest = json.loads(manifest_raw)
        if {e['path'] for e in manifest['files']} != {'model.safetensors', 'tokenizer.json', 'config.yaml'} or len(manifest['files']) != 3:
            raise ValueError('exact_model_namespace')
        BUNDLE.mkdir(mode=0o700)
        for item in manifest['files']:
            copy_exact(ASSETS / item['path'], BUNDLE / item['path'], item['bytes'], item['sha256'])
        copy_exact(manifest_path, BUNDLE / 'manifest.json', len(manifest_raw), descriptor['manifest_sha256'])
        os.chmod(BUNDLE, 0o555)
        os.chown(BUNDLE, MODEL_OWNER, MODEL_OWNER)
        reference_fd = stage_reference(entry)
        ATTEMPT.mkdir(mode=0o700)
        CONDITIONING.mkdir(mode=0o700)
        os.chmod(CONDITIONING, 0o770)
        os.chown(CONDITIONING, MODEL_OWNER, CHILD_UID)
        os.chmod(ATTEMPT, 0o770)
        os.chown(ATTEMPT, MODEL_OWNER, CHILD_UID)
        if descriptor['reference_sha256'] != entry['sha256']: raise ValueError('reference_binding')
        if descriptor['memory_bytes'] != AS_BYTES or descriptor['threads'] != 2 or descriptor['max_output_bytes'] > FILE_BYTES or len(args.text.encode('utf-8')) > descriptor['max_text_bytes']:
            raise ValueError('candidate_limits_mismatch')
        # All privileged file access and model copies finish before this boundary.
        os.setgroups([])
        os.setresgid(CHILD_UID, CHILD_UID, CHILD_UID)
        os.setresuid(CHILD_UID, CHILD_UID, CHILD_UID)
        identity = dropped_status()
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '2'
        os.environ.update(TORCH_DEVICE='cpu', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME='/tmp/huggingface', TMPDIR='/tmp')
        if os.environ.get('TORCHINDUCTOR_CACHE_DIR') is not None:
            raise ValueError('automatic_cache_test_requires_unset_initial_cache')
        resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (48, 48))
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(SECONDS)
        stage = 'limited_import_and_execute'
        print('Pocket prequal: UID20000, no groups/caps, AS3GiB FSIZE1MiB NPROC48, deadline180s; ML import begins now.', file=sys.stderr, flush=True)
        # No Torch, NumPy, Pocket or project import occurs before limits above.
        sys.path.insert(0, '/opt/speech')
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        if versions != VERSIONS:
            raise ValueError('runtime_version_mismatch')
        with contextlib.redirect_stdout(sys.stderr):
            from worker.engines.pocket_prepare import PocketPreparation, execute
            spec = PocketPreparation(**descriptor)
            result = execute(spec, args.text, {spec.bundle_id: str(BUNDLE)}, {spec.reference_id: str(REFERENCE)}, reference_temp_dir=CONDITIONING, managed_owner_uid=MODEL_OWNER, reference_fd=reference_fd, managed_namespace=True)
        stage = 'measure_output'
        entries = sorted(p.name for p in CONDITIONING.iterdir())
        if entries:
            raise ValueError('conditioning_not_empty_after_engine')
        if 'TORCHINDUCTOR_CACHE_DIR' in os.environ:
            raise ValueError('cache_environment_not_restored')
        report['conditioning_entries'] = sorted(x.name for x in CONDITIONING.iterdir())[:64]
        if result['reference']['sha256'] != entry['sha256'] or result['reference']['frames'] != entry['frames']:
            raise ValueError('prepared_reference_changed')
        audition = result['audition']
        data = audition['data']
        with wave.open(io.BytesIO(data), 'rb') as wav:
            rate, channels, width, frames = wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()
            if (rate, channels, width, wav.getcomptype()) != (24000, 1, 2, 'NONE') or not 0 < frames <= spec.max_frames:
                raise ValueError('measured_output_format')
        if len(data) != audition['size_bytes'] or len(data) > spec.max_output_bytes or hashlib.sha256(data).hexdigest() != audition['sha256']:
            raise ValueError('measured_output_identity')
        signal.alarm(0)
        report.update(status='engine_succeeded', managed_namespace=True, conditioning_entries=entries, automatic_cache_environment_restored=True, runtime_versions=versions, context_sha256=hashlib.sha256(json.dumps(inputs['context'],sort_keys=True,separators=(',', ':'),ensure_ascii=False).encode()).hexdigest(), execution=spec.identity(), text_sha256=hashlib.sha256(args.text.encode('utf-8')).hexdigest(), child_identity=identity, reference_sha256=entry['sha256'], audition=dict(sha256=audition['sha256'], bytes=len(data), frames=frames, sample_rate=rate, channels=channels, media_format='wav', media_subtype='PCM_16'), limits=dict(address_space_bytes=AS_BYTES, file_bytes=FILE_BYTES, nproc=48, threads=2, deadline_seconds=SECONDS), resource_observation=dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), qualification_receipt_created=False, installed_profile_changed=False, reaping_evidence='parent must corroborate actual container/process termination; this JSON is emitted before process exit')
    except Exception as exc:
        signal.alarm(0)
        # Frames and text are bounded; no locals, model state or token values.
        trace = traceback.TracebackException.from_exception(exc, limit=12, capture_locals=False)
        formatted = ''.join(trace.format())
        sys.stderr.write(formatted[:16384])
        if len(formatted) > 16384:
            sys.stderr.write('\n[traceback truncated at 16384 characters]\n')
        sys.stderr.flush()
        report.update(stage=stage, exception_type=type(exc).__name__, error=str(exc)[:600], child_uid=os.getuid(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    finally:
        if reference_fd is not None:
            os.close(reference_fd)
    report['elapsed_seconds'] = round(time.monotonic() - started, 6)
    print(json.dumps(report, sort_keys=True, separators=(',', ':')), flush=True)
    return 0 if report['status'] == 'engine_succeeded' else 2


if __name__ == '__main__':
    raise SystemExit(main())
