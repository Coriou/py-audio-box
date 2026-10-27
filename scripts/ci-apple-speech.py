#!/usr/bin/env python3
"""Opt-in finite native inference prerequisite, not installed qualification."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DESCRIPTOR = ROOT / 'requirements/tts/qwen-0.6b-customvoice-macos.expected.json'
MAX_RSS = 6 * 1024**3


def save(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if len(raw) > 65536:
        raise ValueError('report_limit')
    path.write_bytes(raw)


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
        with urllib.request.urlopen(entry['origin'], timeout=20) as source, target.open('xb') as dst:
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


def infer(bundle, output, language):
    from worker import native_child
    from worker.engines.qwen_execution import QwenExecution
    from worker.hardware import Host, Device, DeviceFamily, HardwareInventory
    expected = json.loads(DESCRIPTOR.read_text())
    raw = (bundle / 'manifest.json').read_bytes()
    text = {'English': 'Hello. The test is complete.',
            'French': 'Bonjour. Le test est terminé.'}[language]
    spec = QwenExecution(bundle_id='native-ci', manifest_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint=expected['checkpoint'], revision=expected['revision'], task='custom_voice',
        language=language, voice_id='native-ci', voice_revision='v1', threads=2,
        max_text_bytes=256, max_frames=24000*20, max_output_bytes=960044,
        memory_bytes=MAX_RSS, sample_rate=24000, max_new_tokens=256, timeout_s=30,
        device='mps', dtype='float32', attention='sdpa', preset='Ryan')
    inventory = HardwareInventory(Host('darwin', 'arm64', 'native'),
        (Device(DeviceFamily.MPS, 0, True, True, datetime.now(timezone.utc).isoformat()),))
    native_child.install_spawn_guard()
    started = time.monotonic()
    result = native_child.infer(spec, text, {'native-ci': bundle}, {}, inventory,
                                managed_owner_uid=os.getuid())
    data = result['data']
    with wave.open(io.BytesIO(data), 'rb') as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        if (wav.getnchannels(), wav.getsampwidth(), rate) != (1, 2, 24000) or frames <= 0:
            raise ValueError('invalid_wav')
        pcm = wav.readframes(frames)
        if not any(pcm):
            raise ValueError('silent_wav')
    seconds = time.monotonic() - started
    (output / (language + '.wav')).write_bytes(data)
    save(output / (language + '.json'), dict(status='passed', language=language,
        text=text, device='mps', fallback_enabled=False, seconds=seconds,
        duration_seconds=frames/rate, rtf=seconds/(frames/rate),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
        ending_verified=False, installed_qualified=False))


def run(output):
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
        download='not_run', English='not_run', French='not_run',
        installed_qualified=False, cpu_comparison='not_run_native_owner_requires_mps',
        versions={n: importlib.metadata.version(n) for n in expected['versions']})
    stage = 'download'
    try:
        if report['versions'] != expected['versions']:
            raise ValueError('runtime_version_mismatch')
        with tempfile.TemporaryDirectory(prefix='apple-speech-') as private:
            bundle = Path(private) / 'bundle'
            bundle.mkdir(mode=0o700)
            try:
                acquire(bundle, expected)
                report['download'] = 'passed'
                for language in ('English', 'French'):
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
                    with (output / (language + '.log')).open('wb') as log:
                        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                            '--child', str(bundle), '--language', language, '--output', str(output)],
                            env=env, stdout=log, stderr=log, start_new_session=True)
                        cutoff = time.monotonic() + 30
                        started = time.monotonic()
                        peak_rss = 0
                        try:
                            while child.poll() is None:
                                if time.monotonic() >= cutoff:
                                    raise TimeoutError('native_child_30_seconds')
                                try:
                                    rss = subprocess.check_output(['/bin/ps', '-o', 'rss=', '-p', str(child.pid)],
                                        timeout=2, text=True).strip()
                                except subprocess.CalledProcessError:
                                    if child.poll() is not None: break
                                    raise
                                peak_rss = max(peak_rss, int(rss or '0')*1024)
                                if peak_rss > MAX_RSS:
                                    raise MemoryError('native_child_6_GiB_RSS_limit')
                                if log.tell() > 1024**2:
                                    raise ValueError('child_log_limit')
                                time.sleep(.25)
                            if child.wait() != 0:
                                raise RuntimeError('native_child_failed_see_child_log')
                        finally:
                            if child.poll() is None:
                                os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=5)
                            report['child_reaped'] = True
                            report['child_exit'] = child.returncode
                            report['child_seconds'] = time.monotonic() - started
                            report['sampled_peak_rss_bytes'] = peak_rss
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
    args = parser.parse_args()
    if args.child:
        infer(args.child, args.output, args.language)
    else:
        run(args.output)
