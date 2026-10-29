#!/usr/bin/env python3
"""Manual, one-shot installed Qwen audio gate on a dedicated Linux runner.

This entrypoint deliberately has no dry-run or synthetic-success mode. It writes
only a public, redacted JSON receipt; private work and failed Docker handles stay
on the runner for operator review.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
import uuid
import wave
import zipfile

GIB = 1024 ** 3
COMMAND_LOG = []
COMMAND_LOG_DIR = None
LOG_EDGE_BYTES = 65536
PUBLIC_ERROR_CODES = frozenset(('revision_conflict','idempotency_conflict','unsupported_control',
    'not_ready','resource_limit','invalid_media','invalid_request','not_found','forbidden',
    'stale_attempt','cancel_pending','missing_media','stale_take','clipping',
    'migration_collision','alias_conflict','version_conflict','voice_archived',
    'publication_format_migration_required','transport_error'))
PUBLIC_ERROR_PHASES = frozenset(('voice_resolve','preset_status','preset_register',
    'voice_version_get','generation_submit','job_get','artifact_get','download',
    'upload','enroll','speak'))
PUBLIC_HELPER_CODES = frozenset(('daemon_closed_record','daemon_container_mount_binding',
    'daemon_container_security_binding','daemon_export_exact_mount_options',
    'daemon_helper_begin','daemon_helper_capability_identity','daemon_helper_deadline',
    'daemon_helper_distinct_roots','daemon_helper_incarnation_changed',
    'daemon_helper_linux_root','daemon_helper_mount_alias','daemon_helper_mountinfo',
    'daemon_helper_phase','daemon_helper_root_changed','daemon_no_copyup_mount_required',
    'daemon_phase_eof_retained','daemon_phase_replay','daemon_phase_timeout',
    'daemon_phase_write_timeout','daemon_requested_mount_binding',
    'daemon_runtime_wait_timeout_retained','daemon_snapshot_binding',
    'daemon_terminal_eof_timeout','daemon_trailing_protocol',
    'qwen_service_cgroupfs_required'))
SAFE_FLAGS = {'--format','--output','--require-hashes','--only-binary=:all:', '-r', '-m',
              '--model','--root','--snapshot','--cache-root','--pull=true','--target','-f','-t',
              '--clone-bundle','--ryan-bundle','--context','--control-image','--worker-image',
              '--helper-image','--custody','--reference-wav','--reference-text','--port','--handle',
              '--installation','--name','--wav','--audition-text','--wait-seconds','--voice',
              '--text','--idempotency-key','--request','--limit'}
PUBLIC_TOKENS = {'docker','build','image','inspect','version','info','archive','qwen-snapshot',
                 'up','docker-qwen','stop','status','enroll','speak','voice','resolve','version',
                 'get','speech','generate','job','cancel','pip','install','venv','clients.cli',
                 'clone','Ryan','control','worker-qwen','default'}
BASE = Path(__file__).resolve().parents[1]


def fail(reason):
    raise RuntimeError(reason)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def wav(path):
    with wave.open(str(path), 'rb') as f:
        channels, width, rate, frames = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        if f.getcomptype() != 'NONE' or channels != 1 or width != 2 or rate <= 0 or frames <= 0:
            fail('invalid_mono_pcm16_wav')
        raw = f.readframes(frames)
        if len(raw) != frames * 2 or f.readframes(1):
            fail('truncated_wav')
    samples = [int.from_bytes(raw[i:i+2], 'little', signed=True) for i in range(0, len(raw), 2)]
    peak = max(abs(x) for x in samples)
    rms = math.sqrt(sum(x*x for x in samples) / len(samples))
    if peak == 0 or rms < 1:
        fail('silent_audio')
    return dict(sha256=sha(path), bytes=Path(path).stat().st_size, sample_rate=rate,
                frames=frames, duration_seconds=frames/rate, peak=peak, rms=rms)


class BoundedOutput:
    """Drain a pipe continuously while retaining exact first and last bytes."""
    def __init__(self):
        self.first = bytearray()
        self.last = bytearray()
        self.total = 0

    def add(self, data):
        self.total += len(data)
        take = max(0, LOG_EDGE_BYTES-len(self.first))
        self.first.extend(data[:take])
        self.last.extend(data)
        if len(self.last) > LOG_EDGE_BYTES:
            del self.last[:-LOG_EDGE_BYTES]

    def retained(self):
        if self.total <= LOG_EDGE_BYTES:
            return bytes(self.first)
        if self.total <= 2*LOG_EDGE_BYTES:
            return bytes(self.first) + bytes(self.last)[-(self.total-LOG_EDGE_BYTES):]
        return bytes(self.first) + (b'\n[omitted %d bytes]\n' % (self.total-2*LOG_EDGE_BYTES)) + bytes(self.last)


def structured_cause(stdout, stderr):
    """Expose only code-shaped fields, never arbitrary helper messages."""
    for line in reversed(stdout.last.decode('utf-8', 'replace').splitlines()):
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            cause = {}
            if isinstance(value.get('code'), str) and value['code'] in PUBLIC_ERROR_CODES:
                cause['code'] = value['code']
            if isinstance(value.get('phase'), str) and value['phase'] in PUBLIC_ERROR_PHASES:
                cause['phase'] = value['phase']
            if cause:
                return cause
    diagnostic = stderr.last.decode('utf-8', 'replace')
    match = re.search(r'\bdaemon_helper_refusal:(install|init|up|status|stop):'
                      r'(operation_failed|prior_failure|final_refusal):([a-z][a-z0-9_]{1,79}):', diagnostic)
    if match and match.group(3) in PUBLIC_HELPER_CODES:
        return dict(code='daemon_helper_refusal', helper_phase=match.group(1),
                    helper_stage=match.group(2), detail_code=match.group(3))
    if 'qwen_service_cgroupfs_required:' in diagnostic:
        return dict(code='qwen_service_cgroupfs_required')
    return None


def command(args, *, timeout=600, cwd=None, capture=True, phase=None):
    start = time.monotonic()
    safe = [Path(args[0]).name] + [x if x in SAFE_FLAGS or x in PUBLIC_TOKENS else '<private-or-pin>'
                                  for x in args[1:]]
    streams = {'stdout': BoundedOutput(), 'stderr': BoundedOutput()}
    proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True,
                            env={**os.environ, 'HF_HUB_DISABLE_TELEMETRY': '1',
                                 'DOCKER_HOST': 'unix:///var/run/docker.sock', 'DOCKER_CONTEXT': 'default'})
    selector = selectors.DefaultSelector()
    for name, pipe in (('stdout', proc.stdout), ('stderr', proc.stderr)):
        selector.register(pipe, selectors.EVENT_READ, name)
    deadline = start + timeout
    expired = False
    try:
        while selector.get_map() or proc.poll() is None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                expired = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            for key, _ in selector.select(min(remaining, 0.25)):
                data = os.read(key.fd, 65536)
                if data:
                    streams[key.data].add(data)
                else:
                    selector.unregister(key.fileobj)
    finally:
        selector.close()
        for pipe in (proc.stdout, proc.stderr):
            pipe.close()
        if expired:
            proc.wait(timeout=5)
        else:
            proc.wait()
    seconds = round(time.monotonic()-start, 3)
    receipt = dict(argv=safe, phase=phase or Path(args[0]).name, seconds=seconds,
                   outcome='timeout' if expired else 'completed',
                   exit_code=None if expired else proc.returncode)
    if COMMAND_LOG_DIR is not None:
        COMMAND_LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        identifier = 'command-%04d' % (len(COMMAND_LOG)+1)
        receipt['private_log_id'] = identifier
        for name, output in streams.items():
            path = COMMAND_LOG_DIR/(identifier+'.'+name+'.log')
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as log:
                log.write(output.retained())
            receipt[name+'_bytes'] = output.total
            receipt[name+'_omitted_bytes'] = max(0, output.total-2*LOG_EDGE_BYTES)
    cause = structured_cause(streams['stdout'], streams['stderr'])
    if cause and (expired or proc.returncode):
        receipt['cause'] = cause
    COMMAND_LOG.append(receipt)
    if expired:
        fail('command_timeout:' + Path(args[0]).name)
    if proc.returncode:
        fail('command_failed:' + Path(args[0]).name + ':' + str(proc.returncode))
    if capture:
        if streams['stdout'].total > 2*LOG_EDGE_BYTES:
            fail('command_output_exceeds_capture_budget:' + Path(args[0]).name)
        return streams['stdout'].retained().decode('utf-8').strip(), seconds
    return '', seconds

def json_command(args, **kw):
    raw, duration = command(args, **kw)
    return json.loads(raw.splitlines()[-1]), duration


def verify_source(reviewed_sha):
    """Bind the archive and this harness to one clean, reviewed commit."""
    if not re.fullmatch(r'[0-9a-f]{40}', reviewed_sha):
        fail('exact_reviewed_sha_required')
    # The harness runs through sudo; trust only its own exact checkout path.
    git = ['git', '-c', 'safe.directory=' + str(BASE), '-C', str(BASE)]
    head = subprocess.run([*git, 'rev-parse', 'HEAD'],
                          text=True, capture_output=True, timeout=15, check=True).stdout.strip()
    if head != reviewed_sha:
        fail('reviewed_sha_mismatch')
    dirty = subprocess.run([*git, 'status', '--porcelain=v1',
                            '--untracked-files=all'], text=True, capture_output=True,
                           timeout=15, check=True).stdout
    if dirty:
        fail('source_checkout_not_clean')


def check_host(work, report=None):
    # These are observations only. Keep them even when admission refuses before
    # Docker inspection, model acquisition, or any installed command.
    disk = shutil.disk_usage(work).free
    memory = None
    if Path('/proc/meminfo').is_file():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                memory = int(line.split()[1]) * 1024
                break
    host = dict(platform_system=platform.system(), platform_machine=platform.machine(),
                python_version=platform.python_version(), python_executable=sys.executable,
                docker_cli_available=shutil.which('docker') is not None,
                docker_socket_present=Path('/var/run/docker.sock').exists(),
                available_ram_bytes=memory, required_available_ram_bytes=12*GIB,
                free_disk_bytes=disk, required_free_disk_bytes=22*GIB,
                memory_policy='6_gib_control_plus_6_gib_worker_budget_conservative_admission')
    if report is not None:
        report['host'] = host
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or os.geteuid() != 0:
        fail('requires_linux_amd64_root')
    if sys.version_info[:2] != (3, 11):
        fail('requires_python_311')
    if not Path('/sys/fs/cgroup/cgroup.controllers').is_file():
        fail('requires_cgroup_v2')
    if disk < 22*GIB:
        fail('requires_22_gib_free_disk')
    if memory is None or memory < 12*GIB:
        fail('requires_12_gib_available_ram')
    if (os.environ.get('DOCKER_HOST', 'unix:///var/run/docker.sock') != 'unix:///var/run/docker.sock'
            or os.environ.get('DOCKER_CONTEXT', 'default') != 'default'):
        fail('requires_local_docker_socket')
    version, _ = json_command(['docker','version','--format','{{json .Server}}'], timeout=15)
    if int(version['Version'].split('.')[0]) != 27:
        fail('requires_docker_27')
    info, _ = json_command(['docker','info','--format','{{json .}}'], timeout=15)
    if info.get('CgroupDriver') != 'cgroupfs' or info.get('CgroupVersion') != '2' or info.get('Architecture') not in ('x86_64','amd64'):
        fail('requires_local_rootful_cgroupfs_v2_amd64')
    if info.get('SecurityOptions') and any('rootless' in str(s) for s in info['SecurityOptions']):
        fail('requires_rootful_docker')
    host.update(docker_version=version['Version'], docker_id=info.get('ID'), cgroup_driver='cgroupfs')
    return host


def extract_archive(archive, dest):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if not names or any(not n.startswith('speech-box/') or '..' in Path(n).parts for n in names):
            fail('archive_members_invalid')
        z.extractall(dest)
    return dest/'speech-box'


def snapshot_input(cache_root, descriptor):
    """A missing snapshot means private staging acquisition, never a cache write."""
    return cache_root/descriptor['checkpoint'].split('/')[-1]/'snapshots'/descriptor['revision']


def stop_owned(report, handle, extracted, py):
    if not (handle and handle.is_file() and extracted):
        return
    try:
        stopped, sec = json_command([py or sys.executable, str(extracted/'speech-box'),
                                     'docker-qwen', 'stop', '--handle', str(handle)],
                                    cwd=extracted, timeout=180)
        if stopped.get('phase') != 'stopped':
            fail('owned_stop_receipt_missing')
        report['stop'] = dict(seconds=sec, completed=True)
    except Exception as exc:
        report['stop'] = dict(failure=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__)
        report['status'] = 'FAILED'
        report.setdefault('failure', 'owned_stop_failed')


def call_client(python, extracted, root, *args, timeout=610):
    uid=os.environ.get('SUDO_UID')
    prefix=['sudo','-u','#'+uid] if uid and int(uid)>0 else []
    return json_command([*prefix,python,'-m','clients.cli','--installation',str(root),*map(str,args)], cwd=extracted, timeout=timeout)


def run_speech_repeats(report, py, extracted, root, work):
    seen_jobs = set()
    for voice in ('ci-reference', 'Ryan'):
        expected_version = None
        for repeat in (1, 2):
            output = work/(voice+'-'+str(repeat)+'.wav')
            spoken_text = 'The real audio gate is speaking now. Pass %d for %s.' % (repeat, voice)
            result, sec = call_client(py, extracted, root, 'speak', '--voice', voice,
                                      '--text', spoken_text, '--output', output, '--wait-seconds', '600')
            if result.get('state') != 'succeeded' or not output.is_file():
                fail('real_speech_failed_'+voice)
            job_id = result.get('job_id')
            version_id = result.get('voice_version_id')
            if not isinstance(job_id, str) or not job_id or job_id in seen_jobs:
                fail('speech_job_not_fresh_'+voice)
            seen_jobs.add(job_id)
            if expected_version is None:
                selected, _ = call_client(py, extracted, root, 'voice', 'resolve', voice, timeout=30)
                versions = selected.get('version_ids')
                if not isinstance(versions, list) or not versions:
                    fail('speech_voice_identity_missing_'+voice)
                expected_version = versions[-1]
            if version_id != expected_version:
                fail('speech_voice_identity_mismatch_'+voice)
            audio = wav(output)
            report['steps'].append(dict(name='speak', voice=voice, repeat=repeat, seconds=sec,
                                        job_id=job_id, voice_version_id=version_id,
                                        text_sha256=hashlib.sha256(spoken_text.encode()).hexdigest(), audio=audio))
            report['evidence'] = 'real_installed_audio_only'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--reference-wav', type=Path, required=True)
    p.add_argument('--reference-text-file', type=Path, required=True)
    p.add_argument('--reference-kind', choices=('unknown','human-recorded','synthetic'),
                   default='unknown', help='Operator-declared provenance; unknown is refused')
    p.add_argument('--reference-authorized', action='store_true',
                   help='Operator attests authorization to use this reference for voice cloning')
    p.add_argument('--cache-root', type=Path, required=True, help='Private pinned HF snapshots, cache may initially be empty')
    p.add_argument('--reviewed-sha', required=True, help='Exact reviewed source commit, not a moving branch')
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    report = dict(status='FAILED', stage='preflight', evidence='preflight_only',
                  source_sha=args.reviewed_sha if re.fullmatch(r'[0-9a-f]{40}', args.reviewed_sha) else '',
                  steps=[], qualification='not_tested', real_voice_acceptance='not_tested',
                  reference_provenance=dict(kind=args.reference_kind,
                                            operator_authorized=args.reference_authorized),
                  command_log=COMMAND_LOG)
    handle = None
    extracted = None
    py = None
    start = time.monotonic()
    global COMMAND_LOG_DIR
    COMMAND_LOG.clear()
    COMMAND_LOG_DIR = None
    try:
        verify_source(args.reviewed_sha)
        work = args.work.resolve(strict=True)
        if any(work.iterdir()): fail('work_must_be_empty')
        COMMAND_LOG_DIR = work/'command-logs'
        host = check_host(work, report)
        report['host'] = host
        if args.reference_kind != 'human-recorded' or not args.reference_authorized:
            fail('requires_authorized_human_recorded_reference')
        if (not args.reference_wav.is_absolute() or not args.reference_wav.is_file() or
                not args.reference_text_file.is_absolute() or not args.reference_text_file.is_file()):
            fail('authorized_reference_and_exact_transcript_required')
        reference_text=args.reference_text_file.read_text().strip()
        if not reference_text or len(reference_text)>4096: fail('exact_transcript_required')
        if not args.cache_root.is_absolute() or not args.cache_root.is_dir():
            fail('private_cache_root_required')
        if args.reference_wav.stat().st_size > 1048576: fail('reference_exceeds_service_budget')
        report['reference'] = wav(args.reference_wav)
        report['stage'] = 'installed_setup'
        archive = work/'speech-box.zip'
        _, sec = command([sys.executable, str(BASE/'scripts/headless-install.py'),'archive','--output',str(archive)], timeout=60, capture=False)
        report['steps'].append(dict(name='archive',seconds=sec,sha256=sha(archive)))
        extracted = extract_archive(archive, work/'extract')
        # The runner environment is provisioned from exact package locks, never from the checkout.
        _, sec = command([sys.executable,'-m','venv',str(work/'venv')],timeout=60,capture=False)
        py = str(work/'venv/bin/python')
        report['steps'].append(dict(name='isolated_python',seconds=sec))
        _, sec = command([py,'-m','pip','install','--require-hashes','--only-binary=:all:',
                          '-r',str(extracted/'service/requirements.lock'),'-r',str(extracted/'requirements/client.lock')], timeout=900,capture=False)
        report['steps'].append(dict(name='locked_client_install',seconds=sec))
        bundles = {}
        for name, model in (('clone','clone'),('Ryan','Ryan')):
            descriptor = json.loads((extracted/'requirements/tts'/('qwen-0.6b-base.expected.json' if name=='clone' else 'qwen-0.6b-customvoice-cpu.expected.json')).read_text())
            # The cache snapshot is read-only input; repair downloads absent pinned bytes into private staging.
            snapshot = snapshot_input(args.cache_root, descriptor)
            out, sec = command([py,str(extracted/'speech-box'),'qwen-snapshot','--model',model,'--root',str(work/('stage-'+name)),
                                '--snapshot',str(snapshot),'--cache-root',str(args.cache_root)], cwd=extracted, timeout=1300)
            bundles[name] = Path(out.splitlines()[-1])
            if not bundles[name].is_dir(): fail('bundle_not_published')
            report['steps'].append(dict(name='bundle_'+name,seconds=sec,manifest_sha256=sha(bundles[name]/'manifest.json'),
                                        checkpoint=descriptor['checkpoint'],revision=descriptor['revision']))
        images = {}
        for role, target in (('control','control'),('worker','worker-qwen')):
            tag='pab-qwen-'+uuid.uuid4().hex
            _, sec = command(['docker','build','--pull=true','--target',target,'-f',str(extracted/'service/Dockerfile.release'),
                              '-t',tag,str(extracted)],timeout=1800,capture=False,phase='build_'+target)
            image,_ = command(['docker','image','inspect',tag,'--format','{{.Id}}'],timeout=15)
            if not image.startswith('sha256:'): fail('image_identity_missing')
            images[role]=image
            report['steps'].append(dict(name='build_'+target,seconds=sec,image_id=image))
        token=uuid.uuid4().hex
        custody=Path('/sys/fs/cgroup/speech-box-qwen')/token
        handle=work/'handle.json'
        root=work/'client-root'
        cmd=[py,str(extracted/'speech-box'),'up','--clone-bundle',str(bundles['clone']),
             '--ryan-bundle',str(bundles['Ryan']),'--context','default','--control-image',images['control'],
             '--worker-image',images['worker'],'--helper-image',images['control'],'--custody',str(custody),
             '--reference-wav',str(args.reference_wav),'--reference-text',reference_text,
             '--root',str(root),'--port','18800','--handle',str(handle)]
        up, sec = json_command(cmd,cwd=extracted,timeout=900,phase='public_up')
        if up.get('phase') != 'up' or up.get('qualification') != 'unqualified': fail('unqualified_up_receipt_missing')
        if not handle.is_file(): fail('retained_handle_missing_after_up')
        report['qualification'] = 'unqualified'
        report['steps'].append(dict(name='public_up',seconds=sec,handle_created=True,custody=str(custody)))
        status, sec=call_client(py,extracted,root,'status',timeout=30)
        report['steps'].append(dict(name='authenticated_status',seconds=sec,available=bool(status.get('configured_capabilities'))))
        if not status.get('configured_capabilities'): fail('service_not_configured')
        profile=status['configured_capabilities'].get('profile',{})
        if profile.get('support')!='unqualified' or profile.get('qualification_receipt'):
            fail('local_lane_misrepresented_as_qualified')
        enrollment, sec=call_client(py,extracted,root,'enroll','--name','ci-reference','--wav',args.reference_wav,
                                    '--reference-text',reference_text,'--audition-text','A short test audition.',
                                    '--wait-seconds','600')
        if enrollment.get('state')!='succeeded': fail('real_enrollment_failed')
        report['steps'].append(dict(name='enroll',seconds=sec,state='succeeded',job_id=enrollment.get('job_id')))
        run_speech_repeats(report, py, extracted, root, work)
        # After acquisition the worker's installed policy denies network access.
        # Recheck the exact source bundle manifests and invoke no acquisition command.
        for name in ('clone','Ryan'):
            expected=next(e['manifest_sha256'] for e in report['steps'] if e.get('name')=='bundle_'+name)
            if sha(bundles[name]/'manifest.json') != expected: fail('bundle_manifest_changed')
        offline, sec=call_client(py,extracted,root,'speak','--voice','Ryan','--text','Offline service remains available.',
                                 '--output',work/'Ryan-offline.wav','--wait-seconds','600')
        if offline.get('state')!='succeeded': fail('offline_speech_failed')
        offline_output = work/'Ryan-offline.wav'
        if not offline_output.is_file(): fail('offline_audio_missing')
        report['steps'].append(dict(name='offline_service',seconds=sec,audio=wav(offline_output)))
        # Submit a fresh, deliberately long generation and cancel the confirmed job.
        selected, _=call_client(py,extracted,root,'voice','resolve','ci-reference',timeout=30)
        version_id=selected['version_ids'][-1]
        version,_=call_client(py,extracted,root,'voice','version','get',version_id,timeout=30)
        payload=dict(voice_version_id=version_id,text=('Please read this complete sentence clearly. '*15),
                     identity=version['identity'],fallback='forbid',variants=1,reuse='never')
        key='ci-cancel-'+uuid.uuid4().hex
        receipt, sec=call_client(py,extracted,root,'speech','generate','--request',json.dumps(payload),
                                  '--idempotency-key',key,timeout=30)
        job_id=receipt.get('job_id')
        if not job_id: fail('cancel_job_receipt_missing')
        wait_until=time.monotonic()+60
        while True:
            job,_=call_client(py,extracted,root,'job','get',job_id,timeout=30)
            if job.get('state')=='running': break
            if job.get('state')!='queued' or time.monotonic()>=wait_until:
                fail('cancel_job_not_running')
            time.sleep(0.5)
        cancelled, sec2=call_client(py,extracted,root,'job','cancel',job_id,'--idempotency-key',
                                    'ci-cancel-request-'+uuid.uuid4().hex,timeout=30)
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            job,_=call_client(py,extracted,root,'job','get',job_id,timeout=30)
            if job.get('state') in ('cancelled','failed','succeeded'): break
            time.sleep(1)
        if job.get('state')!='cancelled': fail('cancel_terminal_not_cancelled')
        if not job.get('attempts') or not all(a.get('reaped') is True for a in job['attempts']):
            fail('cancel_reap_receipt_missing')
        report['steps'].append(dict(name='cancel_reap',submit_seconds=sec,cancel_seconds=sec2,
                                    job_id=job_id,state='cancelled',reap_observed=True))
        report['status']='PASSED'
        report['real_voice_acceptance']='technical_checks_only_listening_not_assessed'
        report['stage']='complete'
    except Exception as exc:
        report['status']='FAILED'
        report['failure'] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        if report['failure'] == 'requires_12_gib_available_ram':
            report['next_step'] = 'Use a dedicated Linux runner with at least 12 GiB MemAvailable; no VM or balloon change is performed by this gate.'
        elif report['failure'] == 'requires_python_311':
            report['next_step'] = 'Invoke this gate with an absolute Python 3.11 executable.'
    finally:
        stop_owned(report, handle, extracted, py)
        report['elapsed_seconds']=round(time.monotonic()-start,3)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return 0 if report['status']=='PASSED' else 1


if __name__=='__main__':
    sys.exit(main())
