#!/usr/bin/env python3
"""Opt-in finite native inference prerequisite, not installed qualification."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from urllib.parse import urlsplit
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ci_process_rss import Sampler
DESCRIPTOR = ROOT / 'requirements/tts/qwen-0.6b-customvoice-macos.expected.json'
MAX_RSS = 6 * 1024**3
MAX_LOG = 1024**2
RSS_STALE_SECONDS = .5
DOWNLOAD_HOSTS = frozenset(('huggingface.co', 'cas-bridge.xethub.hf.co',
    'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.hf.co', 'cdn-lfs-eu-1.hf.co',
    'us.aws.cdn.hf.co'))


def download_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in DOWNLOAD_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.fragment):
        raise ValueError('download_outside_reviewed_https_hosts')
    return url


class PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, download_url(newurl))


def public_open(url):
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),
        PublicRedirects()).open(download_url(url), timeout=20)


def budgets(diagnostic_en=False):
    if type(diagnostic_en) is not bool: raise ValueError('diagnostic_mode_required')
    return dict(verification_seconds=30, cold_load_seconds=120,
        generation_seconds=120 if diagnostic_en else 30, encoding_seconds=15,
        total_seconds=270 if diagnostic_en else 180)


def phase_remaining(progress, now, *, diagnostic_en=False):
    limits = budgets(diagnostic_en)
    for key, cap in limits.items():
        value = progress.get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid_phase_duration_' + key)
        if value > cap:
            raise TimeoutError('native_completed_phase_timeout_' + key)
    stage = progress['stage']
    start = progress['stage_started']
    if type(start) not in (int, float) or not math.isfinite(start) or start > now:
        raise ValueError('invalid_phase_start')
    key = {'input_verification': 'verification_seconds', 'cold_load': 'cold_load_seconds',
           'generation': 'generation_seconds', 'encoding': 'encoding_seconds', 'complete': 'encoding_seconds'}[stage]
    limit = limits[key]
    remaining = limit - (now - start)
    if remaining <= 0:
        raise TimeoutError('native_phase_timeout_' + stage)
    return remaining


class SamplerLost(RuntimeError):
    pass


def read_progress(path):
    with path.open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('phase_progress_limit')
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError('phase_progress_object_required')
    return value


def owned_child(argv, env, log_path, observation, *, seconds=30, progress_path=None, diagnostic_en=False):
    """Independent owner deadline, one asynchronous RSS reader, bounded output."""
    effective_budgets = budgets(diagnostic_en)
    observation.update(effective_budgets=effective_budgets, owner_total_seconds=seconds)
    started = time.monotonic()
    cutoff = started + seconds
    total = 0
    failure = None
    sampler = None
    sampler_started = False
    rss_lost = False
    termination = dict(requested_reason=None, signals=[])
    child = subprocess.Popen(argv, env=dict(env, PYTHONUNBUFFERED='1'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
    observation.update(child_pid=child.pid, rss_limit_bytes=MAX_RSS,
        rss_bound_proven=False, termination=termination)
    try:
        sampler = Sampler(child.pid)
        sampler.start()
        sampler_started = True
        with selectors.DefaultSelector() as selector, log_path.open('xb') as log:
            selector.register(child.stdout, selectors.EVENT_READ)
            eof = False
            while True:
                now = time.monotonic()
                remaining = cutoff - now
                if remaining <= 0:
                    raise TimeoutError(f'native_child_{seconds}_seconds')
                if progress_path is not None and progress_path.exists():
                    # Darwin CPython 3.11 uses a system-wide monotonic clock.
                    progress = read_progress(progress_path)
                    observation['progress'] = progress
                    remaining = min(remaining, phase_remaining(progress, time.monotonic(), diagnostic_en=diagnostic_en))
                state = sampler.snapshot()
                # Do not reap/release the PID before the reader binds its first
                # start identity (or reports exit/failure). A zombie cannot reuse
                # its PID; late samples are discarded after request_stop below.
                code = child.poll() if state['count'] or state['error'] else child.returncode
                if state['error']:
                    observation['sampler_error'] = state['error']
                    if state['error']['type'] != 'ProcessGone' or child.poll() is None:
                        rss_lost = True
                        raise SamplerLost('RSS reader failed: ' + state['error']['type'] + ': ' + state['error']['message'])
                    observation['sampler_target_exit_confirmed'] = True
                if code is None:
                    last = state['last']
                    age = now - (last['observed_at'] if last else started)
                    if age > RSS_STALE_SECONDS:
                        rss_lost = True
                        observation['sampler_error'] = dict(type='StaleSample',
                            message='no fresh RSS observation within 0.5 seconds', age_seconds=age)
                        raise SamplerLost('RSS observation stale')
                    if state['peak'] is not None and state['peak'] > MAX_RSS:
                        raise MemoryError('native_child_6_GiB_RSS_limit')
                if eof and code is not None:
                    if code: raise RuntimeError('native_child_failed_see_language_log')
                    break
                for key, _ in selector.select(timeout=min(.05, max(0, cutoff-time.monotonic()))):
                    block = os.read(key.fd, 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        eof = True
                        continue
                    keep = min(len(block), MAX_LOG-total)
                    log.write(block[:keep])
                    total += keep
                    if keep < len(block): raise ValueError('child_log_limit')
            if time.monotonic() > cutoff:
                raise TimeoutError(f'native_child_{seconds}_seconds')
    except BaseException as exc:
        failure = exc
        observation['owner_failure'] = dict(type=type(exc).__name__, message=str(exc)[:1000])
        if isinstance(exc, TimeoutError):
            message = str(exc)
            if message.startswith('native_completed_phase_timeout_'):
                observation['stage_exceeded'] = message.removeprefix('native_completed_phase_timeout_').removesuffix('_seconds')
                observation['deadline_kind'] = 'completed_phase'
            elif message.startswith('native_phase_timeout_'):
                observation['stage_exceeded'] = message.removeprefix('native_phase_timeout_')
                observation['deadline_kind'] = 'active_phase'
            else:
                observation['stage_exceeded'] = 'owner_total'
                observation['deadline_kind'] = 'owner_total'
        raise
    finally:
        cleanup_error = None
        if sampler_started:
            sampler.request_stop()  # Stop accepting samples before any final reap.
        if child.poll() is None:
            termination['requested_reason'] = type(failure).__name__ + ': ' + str(failure)[:1000] if failure else 'owner_cleanup'
            request = dict(signal='SIGKILL', number=int(signal.SIGKILL), process_group=child.pid,
                           requested_at=time.monotonic(), accepted=False)
            termination['signals'].append(request)
            try:
                os.killpg(child.pid, signal.SIGKILL)
                request['accepted'] = True
            except OSError as exc:
                request['error'] = dict(type=type(exc).__name__, errno=exc.errno, message=str(exc)[:1000])
        try:
            child.wait(timeout=5)
            observation['child_reaped'] = True
        except BaseException as exc:
            cleanup_error = exc
            observation['child_reaped'] = False
            observation['cleanup_error'] = dict(type=type(exc).__name__, message=str(exc)[:1000])
        finally:
            child.stdout.close()
            observation.update(child_exit=child.returncode,
                child_seconds=time.monotonic()-started, log_bytes=total)
            stopped = sampler.stop() if sampler_started else None
            state = sampler.snapshot() if sampler_started else dict(last=None, peak=None, count=0, error=None)
            late_sampler_error = None
            if state['error']:
                observation.setdefault('sampler_error', state['error'])
                if state['error']['type'] == 'ProcessGone' and child.returncode is not None:
                    observation['sampler_target_exit_confirmed'] = True
                else:
                    rss_lost = True
                    late_sampler_error = SamplerLost('RSS reader failed during cleanup')
            if stopped is False:
                rss_lost = True
                late_sampler_error = SamplerLost('RSS reader thread did not stop; no replacement created')
            last = state['last']
            if last is None:
                rss_lost = True
                late_sampler_error = SamplerLost('no valid RSS observation obtained')
            observation.update(sampler_thread_stopped=stopped, rss_observation_lost=rss_lost,
                rss_samples=state['count'], last_valid_rss_sample=last,
                last_valid_rss_age_seconds=None if last is None else time.monotonic()-last['observed_at'],
                sampled_peak_rss_bytes=state['peak'],
                rss_guard_status='lost' if rss_lost else 'sampled' if last else 'unobserved')
            observation['observed_exit'] = dict(returncode=child.returncode,
                signal=signal.Signals(-child.returncode).name if type(child.returncode) is int and child.returncode < 0 else None)
        if cleanup_error is not None and failure is None:
            raise cleanup_error
        if late_sampler_error is not None and failure is None:
            raise late_sampler_error


def save(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if len(raw) > 65536:
        raise ValueError('report_limit')
    path.write_bytes(raw)


def new_bundle(private):
    # Only our fresh TemporaryDirectory is canonicalized. The product's model
    # and runtime path validators continue refusing aliases supplied by callers.
    bundle = Path(private).resolve(strict=True) / 'bundle'
    bundle.mkdir(mode=0o700)
    return bundle


def acquire(root, expected):
    """Cold, hash-checked public files only; no hub cache, token or retry."""
    from service.provisioning import validate_expected
    validate_expected(expected)
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise ValueError('requires_10_GiB_free_disk_before_download')
    cutoff = time.monotonic() + 180
    for entry in expected['files']:
        target = root / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        remaining = entry['bytes']
        with public_open(entry['origin']) as source, target.open('xb') as dst:
            while remaining:
                if time.monotonic() >= cutoff:
                    raise TimeoutError('download_180_seconds')
                block = source.read(min(1024**2, remaining))
                if not block:
                    raise ValueError('download_truncated')
                dst.write(block)
                digest.update(block)
                remaining -= len(block)
            if source.read(1) or digest.hexdigest() != entry['sha256']:
                raise ValueError('download_size_or_hash')
        target.chmod(0o400)
    from service.provisioning import manifest_bytes
    (root / 'manifest.json').write_bytes(manifest_bytes(expected))
    (root / 'manifest.json').chmod(0o400)
    for directory in sorted((p for p in root.rglob('*') if p.is_dir()), reverse=True):
        directory.chmod(0o500)
    root.chmod(0o500)


def infer(bundle, output, language, *, diagnostic_en=False):
    limits = budgets(diagnostic_en)
    if diagnostic_en and language != 'English': raise ValueError('diagnostic_English_only')
    from worker import native_child
    from worker.engines.qwen_execution import QwenExecution
    from worker.hardware import Host, Device, DeviceFamily, HardwareInventory
    from lib.tts_backends import qwen_native
    progress_path = output / (language + '-progress.json')
    progress = {}
    def stage(name, **values):
        progress.update(stage=name, stage_started=time.monotonic(), **values)
        temporary = progress_path.with_suffix('.tmp')
        save(temporary, progress)
        temporary.replace(progress_path)
    original_backend = qwen_native.NativeQwenBackend
    class MeasuredBackend(original_backend):
        # Observational wrapper only: actual product loader/backend still run.
        def __init__(self, **kwargs):
            stage('cold_load', verification_seconds=time.monotonic()-started)
            load_started = time.monotonic()
            try:
                super().__init__(**kwargs)
            except BaseException:
                stage('cold_load', cold_load_seconds=time.monotonic()-load_started,
                      cold_load_failed=True)
                raise
            from scripts.ci_qwen_observe import observe
            def observed(**values):
                progress.update(values)
                temporary = progress_path.with_suffix('.tmp')
                save(temporary, progress)
                temporary.replace(progress_path)
            self._stop_observation = observe(self._model, observed)
            import torch
            stage('cold_load', cold_load_seconds=time.monotonic()-load_started,
                selected_device=str(self._model.device),
                mps_allocated_after_load=torch.mps.current_allocated_memory(),
                mps_driver_after_load=torch.mps.driver_allocated_memory(),
                mps_recommended_max_bytes=torch.mps.recommended_max_memory())
        def close(self):
            try:
                if hasattr(self, '_stop_observation'): self._stop_observation()
            finally:
                super().close()
        def synthesise_preset(self, *args, **kwargs):
            stage('generation')
            generation_started = time.monotonic()
            try:
                value = super().synthesise_preset(*args, **kwargs)
            except BaseException:
                stage('generation', generation_seconds=time.monotonic()-generation_started,
                      generation_failed=True)
                raise
            stage('encoding', generation_seconds=time.monotonic()-generation_started)
            return value
    expected = json.loads(DESCRIPTOR.read_text())
    raw = (bundle / 'manifest.json').read_bytes()
    text = {'English': 'Hello. The test is complete.',
            'French': 'Bonjour. Le test est terminé.'}[language]
    spec = QwenExecution(bundle_id='native-ci', manifest_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint=expected['checkpoint'], revision=expected['revision'], task='custom_voice',
        language=language, voice_id='native-ci', voice_revision='v1', threads=2,
        max_text_bytes=256, max_frames=24000*20, max_output_bytes=960044,
        memory_bytes=MAX_RSS, sample_rate=24000, max_new_tokens=256, timeout_s=limits['generation_seconds'],
        device='mps', dtype='float32', attention='sdpa', preset='Ryan')
    inventory = HardwareInventory(Host('darwin', 'arm64', 'native'),
        (Device(DeviceFamily.MPS, 0, True, True, datetime.now(timezone.utc).isoformat()),))
    native_child.install_spawn_guard()
    started = time.monotonic()
    stage('input_verification', effective_budgets=limits, execution_request_seconds=spec.timeout_s, diagnostic_only=diagnostic_en, options=dict(text=text, max_new_tokens=256, dtype='float32', attention='sdpa', threads=2, speaker='Ryan', temperature=spec.temperature, top_p=spec.top_p, top_k=spec.top_k, repetition_penalty=spec.repetition_penalty))
    qwen_native.NativeQwenBackend = MeasuredBackend
    try:
        result = native_child.infer(spec, text, {'native-ci': bundle}, {}, inventory,
                                    managed_owner_uid=os.getuid(),
                                    **({'diagnostic_request_seconds': limits['generation_seconds']} if diagnostic_en else {}))
    finally:
        qwen_native.NativeQwenBackend = original_backend
    data = result['data']
    with wave.open(io.BytesIO(data), 'rb') as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        if (wav.getnchannels(), wav.getsampwidth(), rate) != (1, 2, 24000) or frames <= 0:
            raise ValueError('invalid_wav')
        pcm = wav.readframes(frames)
        if not any(pcm):
            raise ValueError('silent_wav')
    seconds = time.monotonic() - started
    encoding_seconds = time.monotonic()-progress['stage_started'] if progress['stage'] == 'encoding' else 0
    stage('complete', total_seconds=seconds, encoding_seconds=encoding_seconds)
    (output / (language + '.wav')).write_bytes(data)
    save(output / (language + '.json'), dict(status='passed', language=language,
        text=text, requested_device='mps', selected_device=progress.get('selected_device'),
        phases=progress, fallback_enabled=False, seconds=seconds,
        duration_seconds=frames/rate,
        rtf=progress.get('generation_seconds', 0)/(frames/rate), rtf_scope='generation',
        cold_total_rtf=seconds/(frames/rate),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
        ending_verified=False, installed_qualified=False, diagnostic_only=diagnostic_en, effective_budgets=limits))


def run(output, *, diagnostic_en=False):
    limits = budgets(diagnostic_en)
    import platform
    if (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
        raise ValueError('native_Darwin_arm64_required')
    expected = json.loads(DESCRIPTOR.read_text())
    output.mkdir(exist_ok=False, parents=True)
    report = dict(status='failed', commit=os.environ.get('GITHUB_SHA'),
        runner_image=os.environ.get('ImageOS'), runner_version=os.environ.get('ImageVersion'),
        model=expected['checkpoint'], revision=expected['revision'],
        descriptor_sha256=hashlib.sha256(DESCRIPTOR.read_bytes()).hexdigest(),
        python=sys.version, physical_memory_bytes=int(subprocess.check_output(
            ['/usr/sbin/sysctl', '-n', 'hw.memsize'], timeout=5)),
        effective_budgets=limits, diagnostic_only=diagnostic_en,
        purpose='completeness_measurement_not_optimization' if diagnostic_en else 'native_prerequisite',
        requested_languages=['English'] if diagnostic_en else ['English','French'],
        runtime='not_run', download='not_run', English='not_run', French='not_requested' if diagnostic_en else 'not_run', children={},
        installed_qualified=False, cpu_comparison='not_run_native_owner_requires_mps',
        versions={n: importlib.metadata.version(n) for n in expected['versions']})
    stage = 'runtime'
    try:
        if report['versions'] != expected['versions']:
            raise ValueError('runtime_version_mismatch')
        report['runtime'] = 'passed'
        stage = 'download'
        with tempfile.TemporaryDirectory(prefix='apple-speech-') as private:
            bundle = new_bundle(private)
            try:
                acquire(bundle, expected)
                report['download'] = 'passed'
                for language in report['requested_languages']:
                    stage = language
                    from service.provisioning import manifest_bytes
                    bundle.chmod(0o700)
                    manifest = bundle / 'manifest.json'
                    manifest.chmod(0o600)
                    manifest.write_bytes(manifest_bytes(dict(expected, language=language)))
                    manifest.chmod(0o400)
                    bundle.chmod(0o500)
                    env = {k: v for k, v in os.environ.items() if k in (
                        'PATH', 'HOME', 'TMPDIR', 'SYSTEMROOT')}
                    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                        HF_HUB_DISABLE_TELEMETRY='1', PYTORCH_ENABLE_MPS_FALLBACK='0',
                        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
                    observation = report['children'][language] = {}
                    argv = [sys.executable, '-u', str(Path(__file__).resolve()),
                        '--child', str(bundle), '--language', language, '--output', str(output)]
                    if diagnostic_en: argv.append('--diagnostic-en-completeness')
                    owned_child(argv,
                        env, output / (language + '.log'), observation,
                        seconds=limits['total_seconds'], progress_path=output / (language + '-progress.json'), diagnostic_en=diagnostic_en)
                    report[language] = 'passed'
            finally:
                for p in bundle.rglob('*'):
                    if p.is_dir(): p.chmod(0o700)
                bundle.chmod(0o700)
        report['status'] = 'passed'
    except Exception as exc:
        report[stage] = 'failed'
        report['error'] = str(exc)[:1000]
        raise
    finally:
        report['private_model_tree_removed'] = not Path(private).exists() if 'private' in locals() else True
        save(output / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--child', type=Path)
    parser.add_argument('--language', choices=('English', 'French'))
    parser.add_argument('--diagnostic-en-completeness', action='store_true', help='EN only: 120s generation/270s child completeness measurement, not installed qualification')
    args = parser.parse_args()
    if args.child:
        infer(args.child, args.output, args.language, diagnostic_en=args.diagnostic_en_completeness)
    else:
        run(args.output, diagnostic_en=args.diagnostic_en_completeness)
