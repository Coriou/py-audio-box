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
import shutil
import subprocess
import sys
import time
import uuid
import wave
import zipfile

GIB = 1024 ** 3
COMMAND_LOG = []
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


def command(args, *, timeout=600, cwd=None, capture=True):
    start = time.monotonic()
    safe = [Path(args[0]).name] + [x if x in SAFE_FLAGS or x in PUBLIC_TOKENS else '<private-or-pin>'
                                  for x in args[1:]]
    try:
        cp = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                            stderr=subprocess.PIPE if capture else subprocess.DEVNULL, timeout=timeout,
                            env={**os.environ, 'HF_HUB_DISABLE_TELEMETRY': '1'})
    except subprocess.TimeoutExpired:
        COMMAND_LOG.append(dict(argv=safe, seconds=round(time.monotonic()-start,3), outcome='timeout'))
        fail('command_timeout:' + Path(args[0]).name)
    COMMAND_LOG.append(dict(argv=safe, seconds=round(time.monotonic()-start,3), exit_code=cp.returncode))
    # Raw stdout/stderr can contain paths, media or keys; only redacted command receipts are retained.
    if cp.returncode:
        fail('command_failed:' + Path(args[0]).name + ':' + str(cp.returncode))
    return cp.stdout.strip() if capture else '', round(time.monotonic()-start, 3)

def json_command(args, **kw):
    raw, duration = command(args, **kw)
    return json.loads(raw.splitlines()[-1]), duration


def check_host(work):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or os.geteuid() != 0:
        fail('requires_linux_amd64_root')
    if sys.version_info[:2] != (3, 11):
        fail('requires_python_311')
    if not Path('/sys/fs/cgroup/cgroup.controllers').is_file():
        fail('requires_cgroup_v2')
    if shutil.disk_usage(work).free < 22*GIB:
        fail('requires_22_gib_free_disk')
    memory = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    if memory < 12*GIB:
        fail('requires_12_gib_available_ram')
    version, _ = json_command(['docker','version','--format','{{json .Server}}'], timeout=15)
    if int(version['Version'].split('.')[0]) != 27:
        fail('requires_docker_27')
    info, _ = json_command(['docker','info','--format','{{json .}}'], timeout=15)
    if info.get('CgroupDriver') != 'cgroupfs' or info.get('CgroupVersion') != '2' or info.get('Architecture') not in ('x86_64','amd64'):
        fail('requires_local_rootful_cgroupfs_v2_amd64')
    if info.get('SecurityOptions') and any('rootless' in str(s) for s in info['SecurityOptions']):
        fail('requires_rootful_docker')
    if os.environ.get('DOCKER_HOST','unix:///var/run/docker.sock') != 'unix:///var/run/docker.sock':
        fail('requires_local_docker_socket')
    return dict(docker_version=version['Version'], docker_id=info.get('ID'), cgroup_driver='cgroupfs',
                available_ram_bytes=memory, free_disk_bytes=shutil.disk_usage(work).free)


def extract_archive(archive, dest):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if not names or any(not n.startswith('speech-box/') or '..' in Path(n).parts for n in names):
            fail('archive_members_invalid')
        z.extractall(dest)
    return dest/'speech-box'


def call_client(python, extracted, root, *args, timeout=610):
    uid=os.environ.get('SUDO_UID')
    prefix=['sudo','-u','#'+uid] if uid and int(uid)>0 else []
    return json_command([*prefix,python,'-m','clients.cli','--installation',str(root),*map(str,args)], cwd=extracted, timeout=timeout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--reference-wav', type=Path, required=True)
    p.add_argument('--reference-text-file', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True, help='Private pinned HF snapshots, cache may initially be empty')
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    report = dict(status='FAILED', evidence='real_installed_audio_only', source_sha=os.environ.get('GITHUB_SHA',''),
                  steps=[], qualification='unqualified', command_log=COMMAND_LOG)
    handle = None
    extracted = None
    py = None
    start = time.monotonic()
    try:
        if len(report['source_sha']) != 40 or any(c not in '0123456789abcdef' for c in report['source_sha']):
            fail('exact_github_sha_required')
        work = args.work.resolve(strict=True)
        if any(work.iterdir()): fail('work_must_be_empty')
        host = check_host(work)
        report['host'] = host
        if (not args.reference_wav.is_absolute() or not args.reference_wav.is_file() or
                not args.reference_text_file.is_absolute() or not args.reference_text_file.is_file()):
            fail('authorized_reference_and_exact_transcript_required')
        reference_text=args.reference_text_file.read_text().strip()
        if not reference_text or len(reference_text)>4096: fail('exact_transcript_required')
        if not args.cache_root.is_absolute() or not args.cache_root.is_dir():
            fail('private_cache_root_required')
        if args.reference_wav.stat().st_size > 1048576: fail('reference_exceeds_service_budget')
        report['reference'] = wav(args.reference_wav)
        archive = work/'speech-box.zip'
        _, sec = command([sys.executable, str(BASE/'scripts/headless-install.py'),'archive','--output',str(archive)], timeout=60)
        report['steps'].append(dict(name='archive',seconds=sec,sha256=sha(archive)))
        extracted = extract_archive(archive, work/'extract')
        # The runner environment is provisioned from exact package locks, never from the checkout.
        _, sec = command([sys.executable,'-m','venv',str(work/'venv')],timeout=60)
        py = str(work/'venv/bin/python')
        report['steps'].append(dict(name='isolated_python',seconds=sec))
        _, sec = command([py,'-m','pip','install','--require-hashes','--only-binary=:all:',
                          '-r',str(extracted/'service/requirements.lock'),'-r',str(extracted/'requirements/client.lock')], timeout=900)
        report['steps'].append(dict(name='locked_client_install',seconds=sec))
        bundles = {}
        for name, model in (('clone','clone'),('Ryan','Ryan')):
            descriptor = json.loads((extracted/'requirements/tts'/('qwen-0.6b-base.expected.json' if name=='clone' else 'qwen-0.6b-customvoice-cpu.expected.json')).read_text())
            # The cache snapshot is read-only input; repair downloads absent pinned bytes into private staging.
            snapshot = args.cache_root/descriptor['checkpoint'].split('/')[-1]/'snapshots'/descriptor['revision']
            snapshot.mkdir(parents=True, exist_ok=True)
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
                              '-t',tag,str(extracted)],timeout=1800,capture=False)
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
        up, sec = json_command(cmd,cwd=extracted,timeout=900)
        if up.get('phase') != 'up' or up.get('qualification') != 'unqualified': fail('unqualified_up_receipt_missing')
        if not handle.is_file(): fail('retained_handle_missing_after_up')
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
        for voice in ('ci-reference','Ryan'):
            for repeat in (1,2):
                output=work/(voice+'-'+str(repeat)+'.wav')
                result, sec=call_client(py,extracted,root,'speak','--voice',voice,'--text','The real audio gate is speaking now.',
                                        '--output',output,'--wait-seconds','600')
                if result.get('state')!='succeeded' or not output.is_file(): fail('real_speech_failed_'+voice)
                audio=wav(output)
                report['steps'].append(dict(name='speak',voice=voice,repeat=repeat,seconds=sec,job_id=result.get('job_id'),audio=audio))
        # After acquisition the worker's installed policy denies network access.
        # Recheck the exact source bundle manifests and invoke no acquisition command.
        for name in ('clone','Ryan'):
            expected=next(e['manifest_sha256'] for e in report['steps'] if e.get('name')=='bundle_'+name)
            if sha(bundles[name]/'manifest.json') != expected: fail('bundle_manifest_changed')
        offline, sec=call_client(py,extracted,root,'speak','--voice','Ryan','--text','Offline service remains available.',
                                 '--output',work/'Ryan-offline.wav','--wait-seconds','600')
        if offline.get('state')!='succeeded': fail('offline_speech_failed')
        report['steps'].append(dict(name='offline_service',seconds=sec,audio=wav(work/'Ryan-offline.wav')))
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
    except Exception as exc:
        report['status']='FAILED'
        report['failure'] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
    finally:
        if handle and handle.is_file() and extracted:
            try:
                stopped, sec=json_command([py or sys.executable,str(extracted/'speech-box'),'docker-qwen','stop','--handle',str(handle)],cwd=extracted,timeout=180)
                report['stop']=dict(seconds=sec,completed=True)
            except Exception as exc:
                report['stop']=dict(failure=type(exc).__name__)
                report['status']='FAILED'
                report['failure']='owned_stop_failed'
        report['elapsed_seconds']=round(time.monotonic()-start,3)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return 0 if report['status']=='PASSED' else 1


if __name__=='__main__':
    sys.exit(main())
