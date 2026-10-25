#!/usr/bin/env python3
"""Private finite image probe. No Core, credentials, publication or setup API."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

sys.path.insert(0, '/opt/speech')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def exclusive(path, raw):
    with open(path, 'xb') as stream:
        stream.write(raw)


def private_source(path, raw):
    """Open the readonly capability before removing setup's pathname access."""
    exclusive(path, raw)
    path.chmod(0o600)
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    try:
        os.chown(path, 10001, 10001)
        return fd
    except BaseException:
        os.close(fd)
        raise


def versions(expected):
    measured = {name: importlib.metadata.version(name) for name in expected['versions']}
    if measured != expected['versions']:
        raise ValueError('runtime_versions_mismatch')
    return measured


def drop(uid):
    # Permanent UID transition clears effective/permitted capabilities.
    os.setgroups([])
    os.setresgid(uid, uid, uid)
    os.setresuid(uid, uid, uid)
    if os.getgroups() or os.getresuid() != (uid,)*3 or os.getresgid() != (uid,)*3:
        raise ValueError('identity_drop')
    status = Path('/proc/self/status').read_text()
    for name in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb'):
        if int(re.search(r'^'+name+r':\s*([0-9a-f]+)', status, re.M)[1], 16):
            raise ValueError('capabilities_retained')


def child(uid, directory, action, seconds):
    directory.mkdir(mode=0o770)
    directory.chmod(0o770)
    os.chown(directory, uid, uid)
    pid = os.fork()
    if pid == 0:
        try:
            drop(uid)
            value = action(directory)
            exclusive(directory/'result.json', encoded(value))
            os._exit(0)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
    deadline = time.monotonic()+seconds
    while True:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited:
            if not os.WIFEXITED(status) or os.WEXITSTATUS(status):
                raise ValueError('probe_child_failed')
            os.chown(directory, 0, 10001)
            return json.loads((directory/'result.json').read_bytes())
        if time.monotonic() >= deadline:
            os.kill(pid, 15)
            # Exact child remains owned until actual wait. Container deadline is
            # the outer bound if a child cannot respond; no success is emitted.
            os.waitpid(pid, 0)
            raise TimeoutError('probe_child_timeout')
        time.sleep(.05)


def provision(args, expected):
    from service.provisioning import provision as acquire
    root = Path('/models')
    os.chown(root, 10001, 10001)
    measured = versions(expected)
    drop(10001)
    kwargs = {}
    if args.cache:
        files = {entry['origin']: entry for entry in expected['files']}
        def local(url, target, expected_bytes=None, **unused):
            entry = files[url]
            source = Path('/cache')/entry['path']
            if source.is_symlink() or not source.is_file():
                raise ValueError('cache_file_required')
            with source.open('rb') as src, Path(target).open('xb') as dst:
                remaining = entry['bytes']
                while remaining:
                    chunk = src.read(min(65536, remaining))
                    if not chunk: raise ValueError('cache_size')
                    dst.write(chunk); remaining -= len(chunk)
                if src.read(1): raise ValueError('cache_size')
        kwargs['downloader'] = local
    bundle = acquire(root, expected, runtime_versions=measured, **kwargs)
    # Provisioning owns private staging; executors traverse only sealed bundles.
    (root/'bundles').chmod(0o755)
    root.chmod(0o755)
    print(encoded(dict(bundle_id=bundle.name, versions=measured)).decode(), flush=True)


def probe(args, expected):
    from contracts import validate as oracle
    from worker.audio_export import registration, verify_converter, execute as export, pcm
    from worker.runtime_protocol import execution
    from worker.runtime_child import contain_kokoro
    payload = json.loads(Path('/probe-input.json').read_bytes())
    context = payload['context']
    spec = execution(context)
    measured_versions = versions(expected)
    root = Path('/models/bundles')/spec.bundle_id
    for path in (root, *root.rglob('*')):
        info = path.lstat()
        if path.is_symlink() or info.st_uid != 10001 or info.st_mode & 0o222:
            raise ValueError('sealed_model_required')
    version = subprocess.check_output(['/usr/bin/ffmpeg', '-version'], timeout=10, text=True).splitlines()
    closure_output = subprocess.check_output(['/usr/bin/ldd', '/usr/bin/ffmpeg'], timeout=10, text=True)
    if len(closure_output) > 65536 or 'not found' in closure_output:
        raise ValueError('converter_closure')
    closure = sorted({str(Path(p).resolve(strict=True)) for p in re.findall(r'(/[^\s()]+)', closure_output)})
    if not 0 < len(closure) <= 256:
        raise ValueError('converter_closure')
    converter = registration(dict(version='audio-converter-v1', worker_image_digest=args.image,
        executable='/usr/bin/ffmpeg', executable_sha256=hashlib.sha256(Path('/usr/bin/ffmpeg').read_bytes()).hexdigest(),
        reported_version=version[0], reported_build='\n'.join(version[1:]), runtime_closure=closure))
    if converter['executable_sha256'] != payload['ffmpeg_sha256']:
        raise ValueError('converter_pin')
    os.close(verify_converter(converter, args.image))
    work = Path('/tmp/probe')
    work.mkdir(mode=0o755)
    work.chmod(0o755)
    def generation(directory):
        contain_kokoro(context)  # Production limits BEFORE any ML import.
        from worker.engines.kokoro import execute
        result = execute(spec, context['plan']['request']['text'], context['execution']['bundles'])
        raw = result.pop('data')
        exclusive(directory/'speech.wav', raw)
        return dict(result, media=pcm(raw), uid=os.getuid(), groups=os.getgroups())
    generated = child(20000, work/'engine', generation, 45)
    raw = (work/'engine'/'speech.wav').read_bytes()
    source = work/'source.wav'
    # Setup has no DAC override after transferring the private source owner.
    source_fd = private_source(source, raw)
    media = pcm(raw)
    asset = dict(media, id='probe-source', provenance=dict(kind='generated', source_sha256=media['sha256'], source_id='kokoro-ff_siwis', missing_fields=[]))
    request = dict(asset_id=asset['id'], rate=48000, channels=2, loudness=None)
    profile = payload['export_profile']
    plan = dict(schema_version='1.3.0', operation='audio_export', request=request, project=None, voice_versions=[], assets=[asset], resolved=dict(kind='audio_export', identity=dict(kind='audio_export', converter=converter, profile_id=profile['id']), profile=oracle.profile_snapshot(profile), settings=request, settings_sha256=oracle.canonical_hash(request)))
    export_context = dict(plan=plan, bundles={}, execution=dict(converter=converter))
    def conversion(directory):
        export_spec = execution(export_context)
        resource.setrlimit(resource.RLIMIT_AS, (export_spec.memory_bytes,)*2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (export_spec.max_output_bytes,)*2)
        resource.setrlimit(resource.RLIMIT_NPROC, (48, 48))
        output, measured = export(export_context, source_fd, directory, image_digest=args.image, converter_uid=0)
        return dict(media=measured, bytes=len(output), uid=os.getuid(), groups=os.getgroups())
    try:
        converted = child(20001, work/'converter', conversion, 35)
    finally:
        os.close(source_fd)
    print(encoded(dict(kind='actual-cpu-kokoro-export-probe-v1', image=args.image,
        versions=measured_versions, context_sha256=oracle.canonical_hash(context),
        manifest_sha256=spec.manifest_sha256, descriptor=spec.identity(),
        converter=converter, generation=generated, export=converted)).decode(), flush=True)


def main():
    def terminated(signum, frame):
        raise SystemExit('probe_terminated')
    signal.signal(signal.SIGTERM, terminated)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['provision', 'probe'])
    parser.add_argument('--image')
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()
    expected = json.loads(Path('/expected.json').read_bytes())
    if args.mode == 'provision': provision(args, expected)
    else: probe(args, expected)


if __name__ == '__main__': main()
