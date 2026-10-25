#!/usr/bin/env python3
"""Create-only headless local Docker Pocket CPU worked recipe."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import time
import wave

# Preparation reserves two output copies plus six reference/segment copies.
POCKET_JOB_DISK_BYTES = 8 * 1024**2
POCKET_INSTALL_DISK_BYTES = 1024**3
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location('cpu_recipe_commands', ROOT/'scripts/cpu-kokoro-export.py')
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
Commands, save, encoded, digest_file = _shared.Commands, _shared.save, _shared.encoded, _shared.digest_file


def bounded_copy(source, destination, size, digest):
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size or not 0 < size <= 220*1024**2:
            raise ValueError('input_size_or_type')
        h = hashlib.sha256()
        with destination.open('xb') as target:
            remaining = size
            while remaining:
                block = os.read(fd, min(65536, remaining))
                if not block: raise ValueError('input_short')
                remaining -= len(block); h.update(block); target.write(block)
            if os.read(fd, 1) or h.hexdigest() != digest: raise ValueError('input_digest')
        after = os.fstat(fd)
        fields = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        if any(getattr(before,k) != getattr(after,k) for k in fields): raise ValueError('input_changed')
        destination.chmod(0o444)
    finally:
        os.close(fd)


def reference(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 44 < info.st_size <= 1048576: raise ValueError('reference_budget')
        with os.fdopen(fd, 'rb', closefd=False) as stream: raw = stream.read(1048577)
    finally: os.close(fd)
    if len(raw) != info.st_size: raise ValueError('reference_changed')
    with wave.open(io.BytesIO(raw), 'rb') as w:
        rate, frames = w.getframerate(), w.getnframes()
        if (w.getnchannels(),w.getsampwidth(),w.getcomptype()) != (1,2,'NONE') or rate not in (24000,48000) or not 0 < frames <= rate*8:
            raise ValueError('reference_pcm16_mono_24k_or_48k_max8s')
        pcm=w.readframes(frames)
        if len(pcm) != frames*2: raise ValueError('reference_truncated')
    canonical=io.BytesIO()
    with wave.open(canonical,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm)
    if canonical.getvalue()!=raw: raise ValueError('canonical_pcm16_wav_required')
    return raw, dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),frames=frames,sample_rate=rate)


def context(expected, ref):
    from contracts import validate as oracle
    from service.provisioning import identity as bundle_id, manifest_sha256
    from worker.engines.pocket_prepare import PocketPreparation
    from worker.runtime_protocol import execution
    bid, manifest = bundle_id(expected), manifest_sha256(expected)
    identity = dict(engine='pocket-tts',model=expected['checkpoint'],bundle_sha256=manifest,device='cpu',precision='float32')
    profile = dict(id='pocket-cpu',device='cpu',topology='networkless_container',support='unqualified',installation='installed',configuration='valid',readiness='not_ready',verification='synthetic',checked_at='',qualification_receipt='',config_revision='public-cpu-pocket-v1',next_action='Run actual contained probe',compatibility=[dict(identity=identity,mode=mode,operation=op) for mode,op in [('prepare','voice_prepare'),('pocket_clone','speech_generate')]],limits=dict(deadline_seconds=180,decoded_ticks=720000,disk_bytes=POCKET_JOB_DISK_BYTES,expensive_slots=1,queue=4,ram_bytes=3221225472,resident_models=1,text_chars=128,threads=2,upload_bytes=1048576,variants=1,vram_bytes=0))
    spec = PocketPreparation(bundle_id=bid,manifest_sha256=manifest,checkpoint=expected['checkpoint'],revision=expected['revision'],language='fr',threads=2,max_text_bytes=512,max_frames=360000,max_output_bytes=720044,memory_bytes=3221225472,reference_id='template-reference',reference_sha256=ref['sha256'],reference_bytes=ref['bytes'],reference_frames=ref['frames'],reference_rate=ref['sample_rate'],sample_rate=24000,in_tick=0,out_tick=ref['frames']*48000//ref['sample_rate'])
    asset = dict(id=spec.reference_id,sha256=ref['sha256'],frames=ref['frames'],sample_rate=ref['sample_rate'],channels=1,media_format='wav',media_subtype='PCM_16',provenance=dict(kind='upload',source_id=spec.reference_id,source_sha256=ref['sha256'],missing_fields=[]))
    settings = dict(conditioning='pocket',kind='prepare',output_format='wav_pcm16')
    plan = dict(schema_version='1.3.0',operation='voice_prepare',project=None,assets=[asset],voice_versions=[],reference_roles=dict(version='reference-roles-v1',reference_kind='prepared_reference',reference_asset_id='pending',audition_asset_id='pending-audition',audition_text='Bonjour.'),request=dict(asset_id=spec.reference_id,in_tick=0,out_tick=spec.out_tick,reference_text='',audition_text='Bonjour.',identity=identity),resolved=dict(kind='prepare',identity=dict(effective=identity,requested=identity,fallback='forbid',reason='',profile_id=profile['id']),operator_fallback='forbid',profile=oracle.profile_snapshot(profile),reuse_configuration=dict(language='fr',mode='pocket_clone',reference_asset_id='pending'),reuse_identity=None,reuse_settings=dict(kind='pocket_clone',language='fr',seed=None),settings=settings,settings_sha256=oracle.canonical_hash(settings)))
    bundles=[dict(bundle_sha256=manifest,engine='pocket-tts',instructions=False,languages=['fr'],mode='pocket_clone',model=expected['checkpoint'],selectors=[])]
    ctx=dict(plan=plan,bundles=bundles,execution=dict(descriptor=spec.identity(),bundles={bid:'/models/bundles/'+bid},references={spec.reference_id:'/references/assets/'+spec.reference_id}))
    oracle.validate('worker_plan',plan); oracle.frozen_plan(plan,bundles); execution(ctx)
    return ctx, profile


def journey(output, port, token, ctx):
    from clients.http import HTTP
    http=HTTP('http://127.0.0.1:'+str(port),credential=token)
    def status():
        s=http.get('/skeleton'); core=s.get('status',{})
        if not s.get('activation_ready') or not core.get('persistent_activation') or any(core.get(k) for k in ('resource_fault','persistence_fault','integrity_fault','quarantined')): raise ValueError('service_activation_fault')
        return s
    def submit(op,payload,key):
        status(); save(output/(key+'-request.json'),payload)
        result=http.call(op,payload,key=key); save(output/(key+'-receipt.json'),result)
        if http.call(op,payload,key=key)!=result: raise ValueError('idempotency_replay')
        return result
    def complete(key,receipt):
        deadline=time.monotonic()+210
        while time.monotonic()<deadline:
            job=http.call('job_get',dict(job_id=receipt['job_id']))
            if job['state'] in ('succeeded','failed','cancelled'):
                save(output/(key+'-job.json'),job)
                if job['state']!='succeeded' or not job['attempts'] or any(a.get('reaped') is not True for a in job['attempts']): raise ValueError('terminal_job_or_reap')
                while time.monotonic()<deadline:
                    s=status()
                    if s['status']['phase']=='idle' and s['status']['runtime_id'] is None: return job
                    time.sleep(.5)
                break
            status(); time.sleep(.5)
        raise TimeoutError('job_or_ack_deadline')
    def download(asset,name):
        first=http.call('asset_content',dict(asset_id=asset['id'],start=0,end_exclusive=1)); total=first['total_bytes']
        if type(total) is not int or not 0<total<=1048576: raise ValueError('download_budget')
        h=hashlib.sha256()
        with (output/name).open('xb') as stream:
            for start in range(0,total,65536):
                end=min(total,start+65536); chunk=http.call('asset_content',dict(asset_id=asset['id'],start=start,end_exclusive=end)); raw=base64.b64decode(chunk['content_base64'],validate=True)
                if (chunk['start'],chunk['end_exclusive'],chunk['total_bytes'],chunk['content_sha256'])!=(start,end,total,asset['sha256']) or len(raw)!=end-start: raise ValueError('download_binding')
                stream.write(raw);h.update(raw)
        if h.hexdigest()!=asset['sha256']: raise ValueError('download_digest')
        save(output/(name+'.json'),asset)
    save(output/'status-before.json',status()); save(output/'discover.json',http.call('discover',{}))
    asset=http.upload_file(output/'reference.wav'); save(output/'upload.json',asset)
    if asset['sha256']!=ctx['plan']['assets'][0]['sha256']: raise ValueError('upload_digest')
    request=dict(ctx['plan']['request']);request['asset_id']=asset['id']
    receipt=submit('voice_prepare',request,'prepare'); complete('prepare',receipt)
    result=http.call('voice_results',dict(job_id=receipt['job_id']));save(output/'voice-results.json',result)
    if len(result['candidates'])!=1: raise ValueError('one_candidate')
    candidate=result['candidates'][0];roles=candidate['reference_roles'];prepared=roles['reference']['asset']
    if prepared['sha256']!=asset['sha256']: raise ValueError('prepared_reference_digest')
    download(prepared,'prepared.wav');download(roles['audition']['asset'],'audition.wav')
    saved=submit('voice_save',dict(job_id=receipt['job_id'],candidate_id=candidate['id'],voice_id='cpu-pocket-voice',expected_revision=0),'save');version=saved['version']
    if version['reference_asset_ids']!=[prepared['id']] or version['identity']!=request['identity']: raise ValueError('saved_version_binding')
    generated=submit('speech_generate',dict(voice_version_id=version['id'],text='Bonjour.',identity=version['identity'],fallback='forbid',variants=1,reuse='never'),'generate')
    job=complete('generate',generated)
    if len(job['artifact_ids'])!=1: raise ValueError('one_artifact')
    artifact=http.call('artifact_get',dict(artifact_id=job['artifact_ids'][0]))
    if len(artifact['asset_ids'])!=1: raise ValueError('one_asset')
    download(http.call('asset_get',dict(asset_id=artifact['asset_ids'][0])),'generated.wav')
    save(output/'journey.json',dict(kind='actual-installed-pocket-cpu-http-v1',voice_version=version['id'],listening_acceptance='unmeasured'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--port',required=True,type=int);p.add_argument('--principal',default='operator');p.add_argument('--model-cache',required=True,type=Path);p.add_argument('--reference',required=True,type=Path)
    a=p.parse_args()
    if sys.version_info[:2]!=(3,11): raise ValueError('python_3_11_required')
    if not 1024<=a.port<=65535 or not re.fullmatch('[a-zA-Z0-9_-]{1,128}',a.principal): raise ValueError('port_or_principal')
    output=a.output.absolute()
    if any(x.is_symlink() for x in (output,*output.parents)): raise ValueError('symlink_output')
    os.umask(0o077);output.mkdir(mode=0o700,parents=False)
    raw,ref=reference(a.reference);save(output/'reference.wav',raw);(output/'reference.wav').chmod(0o444)
    expected=json.loads((ROOT/'requirements/tts/pocket-fr.expected.json').read_bytes())
    ctx,profile=context(expected,ref)
    from service.provisioning import _pocket_config,bundle_manifest,manifest_bytes
    config_raw=_pocket_config(expected)
    if isinstance(config_raw,str): config_raw=config_raw.encode()
    bid=ctx['execution']['descriptor']['bundle_id'];models=output/'models';bundle=models/'bundles'/bid;bundle.mkdir(parents=True,mode=0o755)
    for entry in expected['files']:
        if entry['path']=='config.yaml':
            if len(config_raw)!=entry['bytes'] or hashlib.sha256(config_raw).hexdigest()!=entry['sha256']: raise ValueError('localized_config_identity')
            save(bundle/'config.yaml',config_raw);(bundle/'config.yaml').chmod(0o444)
        else: bounded_copy(a.model_cache/entry['path'],bundle/entry['path'],entry['bytes'],entry['sha256'])
    save(bundle/'manifest.json',manifest_bytes(bundle_manifest(expected)));(bundle/'manifest.json').chmod(0o444)
    save(output/'probe-input.json',dict(reference=ref,context=ctx));(output/'probe-input.json').chmod(0o444)
    commands=Commands(output,a.context)
    revision=commands.run(['git','rev-parse','HEAD']).strip()
    if not re.fullmatch('[0-9a-f]{40}',revision): raise ValueError('source_revision')
    save(output/'source.json',dict(root=str(ROOT),revision=revision,files={name:digest_file(ROOT/name) for name in ('scripts/cpu-pocket.py','scripts/cpu-pocket-probe.py','scripts/cpu-kokoro-export.py','requirements/tts/pocket-fr.expected.json','requirements/pocket-installed.lock','service/Dockerfile.release')}))
    from service.daemon_client import Docker,manifest
    docker=Docker(a.context);commands.endpoint=docker.endpoint;save(output/'daemon.json',docker.frozen)
    images={}
    for target,role in [('control','control'),('worker-pocket','worker')]:
        iid=output/(target+'.iid')
        commands.docker(['build','--platform','linux/amd64','--pull=false','--target',target,'--file','service/Dockerfile.release','--iidfile',str(iid),'.'],timeout=1800,maximum=32*1024**2)
        image=iid.read_text().strip()
        if not re.fullmatch('sha256:[0-9a-f]{64}',image): raise ValueError('finished_image_id')
        inspected=json.loads(commands.docker(['image','inspect',image]))[0];save(output/(target+'-image.json'),inspected)
        if (inspected['Id'],inspected['Architecture'],inspected['Os'])!=(image,'amd64','linux'): raise ValueError('image_platform')
        images[role]=image
    save(output/'images.json',images)
    mounts=[(ROOT/'scripts/cpu-pocket-probe.py','/recipe.py',True),(output/'probe-input.json','/inputs/probe-input.json',True),(output/'reference.wav','/inputs/reference.wav',True)]
    mounts += [(path,'/inputs/'+path.name,True) for path in sorted(bundle.iterdir())]
    measured=commands.container('cpu-pocket-probe-'+secrets.token_hex(6),images['worker'],mounts,['--worker-image',images['worker']],timeout=210)
    from contracts import validate as oracle
    if measured.get('status')!='engine_succeeded' or measured.get('planned_parent_image')!=images['worker'] or measured.get('runtime_versions')!=expected['versions'] or measured.get('execution')!=ctx['execution']['descriptor'] or measured.get('context_sha256')!=oracle.canonical_hash(ctx) or measured.get('conditioning_entries')!=[] or measured.get('automatic_cache_environment_restored') is not True or measured.get('managed_namespace') is not True: raise ValueError('actual_measurement_binding')
    if measured.get('limits')!=dict(address_space_bytes=3221225472,deadline_seconds=180,file_bytes=1048576,nproc=48,threads=2): raise ValueError('measured_limits')
    child=measured.get('child_identity',{})
    if child.get('Uid')!='20000\t20000\t20000\t20000' or child.get('Groups')!='' or child.get('NoNewPrivs')!='1' or any(int(child.get(k,'1'),16) for k in ('CapInh','CapPrm','CapEff','CapAmb')): raise ValueError('measured_child_identity')
    save(output/'probe.json',measured)
    profile.update(support='qualified',verification='measured',readiness='ready',checked_at=datetime.now(timezone.utc).isoformat(),qualification_receipt='actual-pocket-engine:'+hashlib.sha256(encoded(measured)).hexdigest(),next_action='')
    ctx['plan']['resolved']['profile']=oracle.profile_snapshot(profile)
    config=dict(version=1,ownership_policy='docker-pocket-split-v2',profiles=[profile],budget=dict(disk_bytes=POCKET_INSTALL_DISK_BYTES,free_bytes=67108864,pids=96,queue=4,ram_bytes=4294967296,resident_models=1,slots=1,vram_bytes=0),contexts={a.principal:ctx},bundles={bid:dict(checkpoint=expected['checkpoint'],manifest_sha256=ctx['execution']['descriptor']['manifest_sha256'],path='/models/bundles/'+bid,revision=expected['revision'])},images=images,mounts=dict(control='/control',ipc='/ipc',models='/models',receipts='/receipts',staging='/staging'))
    from service.installed_pocket_daemon import configuration
    config=configuration(config);save(output/'install.json',config)
    token=secrets.token_hex(32);save(output/'credential.json',dict(version=1,credentials=[dict(token=token,principal=a.principal,scopes=['read','production-write'])]))
    entries=[dict(role='models',path='bundles/'+bid+'/'+path.name,source=str(path),bytes=path.stat().st_size,sha256=digest_file(path)) for path in sorted(bundle.iterdir())]
    for role,name,destination in [('config','install.json','install.json'),('credential','credential.json','control.key')]:
        path=output/name;entry=dict(role=role,path=destination,source=str(path),bytes=path.stat().st_size)
        if role!='credential': entry['sha256']=digest_file(path)
        entries.append(entry)
    save(output/'request.json',dict(configuration=config,helper_image=images['control'],inputs=manifest(entries),cpu=dict(control=1000000000,worker=2000000000),tmpfs={role:dict(bytes=134217728,inodes=4096) for role in ('control','worker')},control_port=a.port))
    handle=output/'handle';entrypoint="import sys;sys.path.insert(0,"+repr(str(ROOT))+");from service.daemon_client import pocket_main;pocket_main()"
    def lifecycle(command,label=None):
        argv=[sys.executable,'-I','-B','-c',entrypoint,command,'--handle',str(handle)]
        if command=='install': argv+=['--context',a.context,'--request',str(output/'request.json')]
        result=json.loads(commands.run(argv,timeout=300,maximum=1024**2));save(output/((label or command)+'-result.json'),result);return result
    docker.end=time.monotonic()+120;docker.check()
    lifecycle('install');lifecycle('up');before=lifecycle('status')
    journey(output,a.port,token,ctx)
    lifecycle('stop');lifecycle('up','restart');after=lifecycle('status','restart-status')
    if before.get('containers')!=after.get('containers') or not before.get('containers'): raise ValueError('restart_container_identity_changed')
    # The finite client validates the retained binding and exact daemon-owned IDs.
    from clients.http import HTTP
    status=HTTP('http://127.0.0.1:'+str(a.port),credential=token).get('/skeleton');save(output/'status-after-restart.json',status)
    if not status['activation_ready'] or not status['ready_for_speech']: raise ValueError('restart_not_ready')
    print('Pocket CPU running on http://127.0.0.1:'+str(a.port))
    print('Private credential, handle and retained evidence: '+str(output))


if __name__=='__main__':
    try: main()
    except (Exception,KeyboardInterrupt):
        raise SystemExit('Pocket recipe refused; retain output, handles, containers and volumes for review.') from None
