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
DESCRIPTOR = ROOT / 'requirements/tts/qwen-0.6b-customvoice-macos.expected.json'
MAX_RSS = 6 * 1024**3
MAX_LOG = 1024**2
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


def owned_child(argv, env, log_path, observation):
    """Drain a bounded pipe; check deadlines even after a zero exit."""
    started = time.monotonic()
    cutoff = started + 30
    peak_rss = total = 0
    child = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
    try:
        with selectors.DefaultSelector() as selector, log_path.open('xb') as log:
            selector.register(child.stdout, selectors.EVENT_READ)
            eof = False
            while True:
                remaining = cutoff - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('native_child_30_seconds')
                code = child.poll()
                if eof and code is not None:
                    if code: raise RuntimeError('native_child_failed_see_language_log')
                    break
                if code is None:
                    try:
                        rss = subprocess.check_output(['/bin/ps', '-o', 'rss=', '-p', str(child.pid)],
                            timeout=min(.5, remaining), text=True).strip()
                    except subprocess.CalledProcessError:
                        if child.poll() is None: raise
                        rss = ''
                    peak_rss = max(peak_rss, int(rss or '0')*1024)
                    if peak_rss > MAX_RSS:
                        raise MemoryError('native_child_6_GiB_RSS_limit')
                for key, _ in selector.select(timeout=min(.1, max(0, cutoff-time.monotonic()))):
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
                raise TimeoutError('native_child_30_seconds')
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        child.stdout.close()
        observation.update(child_reaped=True, child_exit=child.returncode,
            child_seconds=time.monotonic()-started, sampled_peak_rss_bytes=peak_rss,
            log_bytes=total)


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
        runtime='not_run', download='not_run', English='not_run', French='not_run', children={},
        installed_qualified=False, cpu_comparison='not_run_native_owner_requires_mps',
        versions={n: importlib.metadata.version(n) for n in expected['versions']})
    stage = 'runtime'
    try:
        if report['versions'] != expected['versions']:
            raise ValueError('runtime_version_mismatch')
        report['runtime'] = 'passed'
        stage = 'download'
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
                    observation = report['children'][language] = {}
                    owned_child([sys.executable, str(Path(__file__).resolve()),
                        '--child', str(bundle), '--language', language, '--output', str(output)],
                        env, output / (language + '.log'), observation)
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
