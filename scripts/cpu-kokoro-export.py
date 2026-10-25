#!/usr/bin/env python3
"""One create-only, local Docker CPU Kokoro/export-v2 worked recipe.

Builds, provisions, measures, installs and performs a canonical HTTP journey.
Failures retain all output, container IDs, volumes and handles. No repair path.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def save(path, value):
    raw = value if isinstance(value, bytes) else encoded(value)
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''): digest.update(chunk)
    return digest.hexdigest()


class Commands:
    def __init__(self, output, context):
        self.output, self.context, self.index = output, context, 0
        self.env = {k:v for k,v in os.environ.items() if not k.startswith('DOCKER_') and k not in ('BUILDX_BUILDER', 'BUILDKIT_HOST')}

    def run(self, command, *, timeout=60, maximum=16*1024**2):
        self.index += 1
        prefix = self.output/('%02d-command'%self.index)
        save(prefix.with_suffix('.json'), dict(argv=command, timeout=timeout))
        # Read concurrently and enforce a byte ceiling; neither PIPE capacity nor
        # an unbounded communicate() may turn build output into host memory use.
        process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        selector = None
        try:
            save(prefix.with_suffix('.pid.json'), dict(pid=process.pid))
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            selector.register(process.stderr, selectors.EVENT_READ)
            deadline, total, chunks = time.monotonic()+timeout, 0, []
            with prefix.with_suffix('.log').open('xb') as log:
                while selector.get_map():
                    if time.monotonic() >= deadline: raise TimeoutError('command_timeout')
                    for key, _ in selector.select(.2):
                        block = os.read(key.fd, 65536)
                        if not block:
                            selector.unregister(key.fileobj); continue
                        total += len(block)
                        if total > maximum: raise ValueError('command_output_limit')
                        log.write(block)
                        if key.fileobj is process.stdout: chunks.append(block)
                code = process.wait(timeout=max(.1, deadline-time.monotonic()))
                if code: raise ValueError('command_failed_see_retained_log')
                save(prefix.with_suffix('.wait.json'), dict(pid=process.pid, returncode=code, reaped=True))
                return b''.join(chunks).decode('utf-8')
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # A concurrent exit still requires the exact wait below.
            # Preserve this Popen owner until real reap. A pending receipt is
            # evidence of unfinished work, never permission for another command.
            pending = False
            while True:
                try:
                    code = process.wait(timeout=15)
                    break
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    if not pending:
                        pending = True
                        try:
                            save(prefix.with_suffix('.retained.json'), dict(pid=process.pid, reap_pending=True))
                            print('Waiting for exact child '+str(process.pid)+' to exit after TERM; retained evidence may require manual intervention.', file=sys.stderr, flush=True)
                        except (OSError, KeyboardInterrupt):
                            pass  # Receipt failure cannot release the live owner.
            if not prefix.with_suffix('.wait.json').exists():
                save(prefix.with_suffix('.wait.json'), dict(pid=process.pid, returncode=code, reaped=True))
            raise
        finally:
            if selector is not None: selector.close()
            process.stdout.close()
            process.stderr.close()

    def docker(self, args, **kwargs):
        return self.run(['docker', '--host', self.endpoint, *args], **kwargs)

    def container(self, name, image, mounts, arguments, *, network='none', timeout=90):
        command = ['create', '--name', name, '--platform', 'linux/amd64', '--pull', 'never',
            '--network', network, '--read-only', '--user', '0:10001', '--cap-drop', 'ALL',
            '--cap-add', 'CHOWN', '--cap-add', 'SETUID', '--cap-add', 'SETGID', '--cap-add', 'KILL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '128', '--memory', '2g',
            '--memory-swap', '2g', '--cpus', '2', '--tmpfs', '/tmp:rw,nosuid,nodev,size=536870912,mode=1777']
        for source, target, readonly in mounts:
            command += ['--mount', 'type=bind,src='+str(source)+',dst='+target+(',readonly' if readonly else '')]
        command += ['--entrypoint', 'python', image, '-I', '-B', '/recipe.py', *arguments]
        identity = self.docker(command).strip()
        if not re.fullmatch('[0-9a-f]{64}', identity): raise ValueError('container_id')
        save(self.output/(name+'.id.json'), dict(id=identity, image=image))
        try:
            output = self.docker(['start', '--attach', identity], timeout=timeout, maximum=1024**2)
        except BaseException:
            before = None
            try:
                before = json.loads(self.docker(['inspect', identity]))[0]
                save(self.output/(name+'.failure-inspect.json'), before)
            except Exception:
                pass  # Retained exact ID still permits TERM and independent wait.
            if before is None or before['State']['Running']:
                try:
                    self.docker(['kill', '--signal', 'TERM', identity], timeout=20)
                except Exception:
                    pass  # An exit racing TERM must still reach wait/inspection.
            try:
                self.docker(['wait', identity], timeout=20)
            finally:
                save(self.output/(name+'.inspect.json'), json.loads(self.docker(['inspect', identity]))[0])
            raise
        waited = self.docker(['wait', identity], timeout=20).strip()
        inspected = json.loads(self.docker(['inspect', identity]))[0]
        save(self.output/(name+'.inspect.json'), inspected)
        state = inspected['State']
        if waited != '0' or state['Running'] or state['OOMKilled'] or state['ExitCode'] != 0 or inspected['Image'] != image:
            raise ValueError('probe_container_terminal_failure')
        return json.loads(output)


def profile(identity, *, export=False):
    limits = dict(deadline_seconds=25 if export else 30, decoded_ticks=28800000 if export else 480000,
        disk_bytes=134217728 if export else 67108864, expensive_slots=1, queue=32,
        ram_bytes=2147483648, resident_models=1, text_chars=10000 if export else 128,
        threads=2, upload_bytes=37748736 if export else 1048576, variants=1, vram_bytes=0)
    return dict(id='audio-export-cpu' if export else 'kokoro-cpu', device='cpu', topology='networkless_container',
        support='unqualified', installation='installed', configuration='valid', readiness='not_ready',
        verification='synthetic', checked_at='', qualification_receipt='', limits=limits,
        config_revision='public-cpu-kokoro-export-v1', next_action='Run the contained actual probe',
        compatibility=[] if export else [dict(identity=identity, mode='kokoro_preset', operation='speech_generate')])


def context(expected):
    from contracts import validate as oracle
    from service.provisioning import identity as bundle_id, manifest_sha256
    from worker.engines.execution import Execution
    from worker.runtime_protocol import execution
    key, manifest = bundle_id(expected), manifest_sha256(expected)
    identity = dict(engine='kokoro', model=expected['checkpoint'], bundle_sha256=manifest, device='cpu', precision='float32')
    spec = Execution(bundle_id=key, manifest_sha256=manifest, checkpoint=expected['checkpoint'], revision=expected['revision'], preset='ff_siwis', language='fr', threads=2, max_text_bytes=512, max_frames=240000, max_output_bytes=480044, memory_bytes=2147483648)
    speech_profile, export_profile = profile(identity), profile(identity, export=True)
    settings = dict(kind='kokoro_preset', language='fr')
    settings_hash = oracle.canonical_hash(settings)
    version = dict(id='vv0', voice_id='voice.0', identity=identity, origin='preset',
        configuration=dict(mode='kokoro_preset', preset='ff_siwis', language='fr'),
        design_lineage=None, reference_asset_ids=[], reference_text='', settings=settings,
        settings_sha256=settings_hash, provenance=dict(kind='https', source_id=expected['files'][0]['origin'], source_sha256=expected['files'][0]['sha256'], missing_fields=[]))
    request = dict(voice_version_id='vv0', text='Bonjour, ceci est un essai réel de notre boîte audio.', identity=identity, fallback='forbid', variants=1, reuse='never')
    plan = dict(schema_version='1.3.0', operation='speech_generate', request=request, project=None, voice_versions=[version], assets=[], resolved=dict(kind='speech', identity=dict(effective=identity, requested=identity, fallback='forbid', reason='', profile_id=speech_profile['id']), operator_fallback='forbid', profile=oracle.profile_snapshot(speech_profile), reuse_configuration=None, reuse_identity=None, reuse_settings=None, settings=settings, settings_sha256=settings_hash))
    value = dict(plan=plan, bundles=[dict(bundle_sha256=manifest, engine='kokoro', instructions=False, languages=['fr'], mode='kokoro_preset', model=expected['checkpoint'], selectors=[dict(languages=['fr'], value='ff_siwis')])], execution=dict(descriptor=spec.identity(), bundles={key:'/models/bundles/'+key}))
    oracle.validate('worker_plan', plan)
    oracle.frozen_plan(plan, value['bundles'])
    execution(value)
    return value, speech_profile, export_profile


def journey(output, port, token):
    from clients.http import HTTP
    http = HTTP('http://127.0.0.1:'+str(port), credential=token)
    status = http.get('/skeleton')
    save(output/'status-before.json', status)
    if not status['activation_ready'] or not status['ready_for_speech']: raise ValueError('service_not_ready')
    registered = http.call('voice_preset_register', dict(preset='ff_siwis', voice_id='cpu-siwis', expected_revision=0), key='cpu-preset')
    save(output/'preset.json', registered)
    version = registered['version']
    def job(operation, request, key):
        receipt = http.call(operation, request, key=key)
        if http.call(operation, request, key=key) != receipt: raise ValueError('idempotency_receipt_changed')
        save(output/(key+'-receipt.json'), receipt)
        deadline = time.monotonic()+90
        while time.monotonic() < deadline:
            value = http.call('job_get', dict(job_id=receipt['job_id']))
            if value['state'] in ('succeeded', 'failed', 'cancelled'): break
            time.sleep(.5)
        else: raise TimeoutError('bounded_job_wait')
        save(output/(key+'-job.json'), value)
        if value['state'] != 'succeeded': raise ValueError('job_failed')
        artifact = http.call('artifact_get', dict(artifact_id=value['artifact_ids'][0]))
        asset = http.call('asset_get', dict(asset_id=artifact['asset_ids'][0]))
        save(output/(key+'-asset.json'), asset)
        return asset
    generated = job('speech_generate', dict(voice_version_id=version['id'], text='Bonjour, la synthèse vocale fonctionne sur le processeur.', identity=version['identity'], fallback='forbid', variants=1, reuse='never'), 'cpu-generation')
    exported = job('audio_export', dict(asset_id=generated['id'], rate=48000, channels=2, loudness=None), 'cpu-export')
    first = http.call('asset_content', dict(asset_id=exported['id'], start=0, end_exclusive=1))
    total = first['total_bytes']
    if not 0 < total <= 36*1024**2: raise ValueError('download_budget')
    digest = hashlib.sha256()
    with (output/'export.wav').open('xb') as stream:
        for start in range(0, total, 65536):
            end = min(total, start+65536)
            chunk = http.call('asset_content', dict(asset_id=exported['id'], start=start, end_exclusive=end))
            raw = base64.b64decode(chunk['content_base64'], validate=True)
            if (chunk['start'],chunk['end_exclusive'],chunk['total_bytes'],chunk['content_sha256']) != (start,end,total,exported['sha256']) or len(raw) != end-start:
                raise ValueError('download_metadata')
            stream.write(raw); digest.update(raw)
    if digest.hexdigest() != exported['sha256']: raise ValueError('download_hash')
    save(output/'journey.json', dict(kind='actual-installed-cpu-http-journey-v1', generated=generated['id'], exported=exported['id'], bytes=total, sha256=digest.hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--principal', default='operator')
    parser.add_argument('--model-cache', type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3,11): raise ValueError('python_3_11_required')
    if not 1024 <= args.port <= 65535 or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', args.principal): raise ValueError('port_or_principal')
    output = args.output.absolute()
    if any(p.is_symlink() for p in (output, *output.parents)): raise ValueError('symlink_output')
    os.umask(0o077)
    output.mkdir(mode=0o700, parents=False)
    commands = Commands(output, args.context)
    expected = json.loads((ROOT/'requirements/tts/kokoro-fr.expected.json').read_bytes())
    pins = json.loads((ROOT/'requirements/tts/kokoro-export.build.json').read_bytes())
    save(output/'source.json', dict(root=str(ROOT), expected_sha256=digest_file(ROOT/'requirements/tts/kokoro-fr.expected.json'), build=pins))
    # The existing client resolver rejects remote/TLS/non-Unix contexts and
    # freezes the actual endpoint. All builds/probes use that same explicit one.
    from service.daemon_client import Docker
    docker = Docker(args.context)
    save(output/'daemon.json', docker.frozen)
    commands.endpoint = docker.endpoint
    images = {}
    for target in ('control','worker-export'):
        iidfile = output/(target+'.iid')
        build = ['build', '--platform', pins['platform'], '--pull=false', '--target', target,
            '--file', pins['dockerfile'], '--iidfile', str(iidfile)]
        if target == 'worker-export':
            for key,value in sorted(pins['build_args'].items()): build += ['--build-arg', key+'='+value]
        commands.docker([*build, '.'], timeout=1800, maximum=32*1024**2)
        image = iidfile.read_text().strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image): raise ValueError('finished_image_id')
        inspected = json.loads(commands.docker(['image','inspect',image]))[0]
        save(output/(target+'-image.json'), inspected)
        if inspected['Id'] != image or inspected['Architecture'] != 'amd64' or inspected['Os'] != 'linux': raise ValueError('image_platform')
        images['control' if target == 'control' else 'worker'] = image
    save(output/'images.json', images)
    models = output/'models'; models.mkdir(mode=0o755)
    mounts = [(ROOT/'scripts/cpu-kokoro-export-probe.py','/recipe.py',True), (ROOT/'requirements/tts/kokoro-fr.expected.json','/expected.json',True)]
    provision_arguments = ['provision']
    provision_mounts = [*mounts, (models,'/models',False)]
    if args.model_cache:
        cache = args.model_cache.absolute()
        if cache.is_symlink() or not cache.is_dir(): raise ValueError('model_cache_directory')
        provision_mounts += [(cache,'/cache',True)]; provision_arguments += ['--cache']
    acquired = commands.container('cpu-acquire-'+secrets.token_hex(6), images['worker'], provision_mounts, provision_arguments, network='none' if args.model_cache else 'bridge', timeout=700)
    ctx, speech, export = context(expected)
    key = ctx['execution']['descriptor']['bundle_id']
    if acquired['bundle_id'] != key or acquired['versions'] != expected['versions']: raise ValueError('provision_identity')
    save(output/'probe-input.json', dict(context=ctx, export_profile=export, ffmpeg_sha256=pins['build_args']['EXPORT_FFMPEG_SHA256']))
    # The bind-mounted, nonsecret input must be readable without DAC override.
    # Its containing output directory remains private to the host operator.
    (output/'probe-input.json').chmod(0o644)
    measured = commands.container('cpu-probe-'+secrets.token_hex(6), images['worker'], [*mounts, (models,'/models',True), (output/'probe-input.json','/probe-input.json',True)], ['probe','--image',images['worker']], timeout=120)
    from contracts import validate as oracle
    if measured['image'] != images['worker'] or measured['context_sha256'] != oracle.canonical_hash(ctx) or measured['versions'] != expected['versions'] or measured['manifest_sha256'] != ctx['execution']['descriptor']['manifest_sha256'] or measured['descriptor'] != ctx['execution']['descriptor']:
        raise ValueError('measurement_binding')
    if measured['kind'] != 'actual-cpu-kokoro-export-probe-v1' or measured['generation']['uid'] != 20000 or measured['export']['uid'] != 20001 or measured['generation']['groups'] or measured['export']['groups'] or measured['export']['media']['media_subtype'] != 'PCM_24':
        raise ValueError('actual_probe_required')
    save(output/'probe.json', measured)
    receipt = 'cpu-probe:'+hashlib.sha256(encoded(measured)).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    for candidate in (speech, export): candidate.update(support='qualified', verification='measured', readiness='ready', checked_at=now, qualification_receipt=receipt, next_action='')
    bundle = models/'bundles'/key
    manifest = ctx['execution']['descriptor']['manifest_sha256']
    config = dict(version=2, ownership_policy='docker-kokoro-export-split-v2', profiles=[speech],
        # Retained speech/export reservations share this quota across jobs.
        budget=dict(disk_bytes=1073741824, free_bytes=16777216, pids=128, queue=32, ram_bytes=2147483648, resident_models=1, slots=1, vram_bytes=0),
        contexts={args.principal:ctx}, bundles={key:dict(checkpoint=expected['checkpoint'], manifest_sha256=manifest, path='/models/bundles/'+key, revision=expected['revision'])},
        images=images, mounts=dict(control='/control',ipc='/ipc',models='/models',receipts='/receipts',staging='/staging'),
        audio_export=dict(converter=measured['converter'], profile=export, principals=[args.principal]))
    from service.installed_export import configuration, export_seal
    config = configuration(config)
    save(output/'expected-seal.json', export_seal(config))
    save(output/'install.json', config)
    token = secrets.token_hex(32)
    save(output/'credential.json', dict(version=1,credentials=[dict(token=token,principal=args.principal,scopes=['read','production-write'])]))
    entries = []
    for path in [bundle/'manifest.json', *[bundle/e['path'] for e in expected['files']]]:
        entries.append(dict(role='models',path='bundles/'+key+'/'+path.name,source=str(path),bytes=path.stat().st_size,sha256=digest_file(path)))
    for role,name,destination in [('config','install.json','install.json'),('credential','credential.json','control.key')]:
        path = output/name
        entry = dict(role=role,path=destination,source=str(path),bytes=path.stat().st_size)
        if role != 'credential': entry['sha256'] = digest_file(path)
        entries.append(entry)
    from service.daemon_client import manifest as input_manifest
    request = dict(configuration=config,helper_image=images['control'],inputs=input_manifest(entries),cpu=dict(control=1000000000,worker=2000000000),tmpfs={role:dict(bytes=134217728,inodes=4096) for role in ('control','worker')},control_port=args.port)
    save(output/'request.json', request)
    handle = output/'handle'
    entrypoint = "import sys;sys.path.insert(0,"+repr(str(ROOT))+");from service.daemon_client import export_main;export_main()"
    def lifecycle(command):
        arguments = [sys.executable,'-I','-B','-c',entrypoint,command,'--handle',str(handle)]
        if command == 'install': arguments += ['--context',args.context,'--request',str(output/'request.json')]
        result = commands.run(arguments, timeout=300, maximum=1024**2)
        save(output/(command+'-result.json'), json.loads(result))
    docker.end = time.monotonic()+120
    docker.check()
    lifecycle('install'); lifecycle('up'); lifecycle('status')
    journey(output, args.port, token)
    lifecycle('stop')
    # Retain stop proof separately: starting again must use the existing handle.
    result = commands.run([sys.executable,'-I','-B','-c',entrypoint,'up','--handle',str(handle)], timeout=300, maximum=1024**2)
    save(output/'restart-result.json', json.loads(result))
    restarted = __import__('clients.http',fromlist=['HTTP']).HTTP('http://127.0.0.1:'+str(args.port),credential=token).get('/skeleton')
    save(output/'status-after-restart.json', restarted)
    if not restarted['activation_ready'] or not restarted['ready_for_speech']:
        raise ValueError('restart_not_ready')
    print('CPU Kokoro/export-v2 running on http://127.0.0.1:'+str(args.port))
    print('Private credential and retained evidence: '+str(output))


if __name__ == '__main__':
    try: main()
    except (Exception, KeyboardInterrupt):
        raise SystemExit('CPU recipe refused; retain output, handles, containers and volumes for review.') from None
