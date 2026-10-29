"""Create a resident Qwen installation from bounded operator inputs.

The seed reference supplies truthful template metadata only. Each enrolled
voice replaces it with its own owned upload and exact transcript.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import platform
import pwd

from contracts import validate as oracle
from service import installed_qwen_daemon as qd
from service import qwen_local_config as local
from service.daemon_client import exclusive, manifest, read
from service.qwen_service_config import build
from service import qwen_local_lane as lane
from worker.engines.qwen_execution import QwenExecution


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8000


def _measurement(path):
    """Read operator evidence as a bounded, unchanged regular file."""
    from service.daemon_client import file_identity, parse
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 65536:
            raise ValueError('setup_measured_profile_file_required')
        raw = stream.read(65537)
        if (len(raw) != before.st_size or file_identity(os.fstat(stream.fileno())) != file_identity(before)
                or file_identity(os.stat(path, follow_symlinks=False)) != file_identity(before)):
            raise ValueError('setup_measured_profile_changed')
    evidence = parse(raw)
    if type(evidence) is not dict:
        raise ValueError('setup_measured_profile_file_required')
    return evidence


def _image_ids(context, names, *, docker=None):
    """Resolve each explicitly named local image, never select from a listing."""
    from service.daemon_client import Docker
    docker = docker or Docker(context, family=qd.FAMILY)
    images = {}
    for role in ('control', 'worker', 'helper'):
        name = names[role]
        if not name or name.startswith('-') or any(c.isspace() for c in name):
            raise ValueError('setup_explicit_local_image_reference_required')
        identity = docker.call(['image', 'inspect', name, '--format', '{{.Id}}'])
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', identity):
            raise ValueError('setup_local_image_id_required')
        if name.startswith('sha256:') and name != identity:
            raise ValueError('setup_image_id_changed')
        images[role] = identity
    docker.check()
    return images


def flag_request(args, *, docker=None):
    """Translate genuine operator inputs into the existing closed request."""
    images = _image_ids(args.context, dict(control=args.control_image, worker=args.worker_image,
                                           helper=args.helper_image), docker=docker)
    evidence = _measurement(args.measurement) if args.measurement else None
    if evidence is None and not getattr(args,'local_unqualified',False):
        raise ValueError('setup_measured_profile_file_required')
    if evidence is not None and getattr(args,'local_unqualified',False):
        raise ValueError('setup_local_lane_receipt_forbidden')
    threads = args.threads
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ValueError('setup_threads_required')
    return dict(principal=args.principal,
                bundles=dict(clone=args.clone_bundle, Ryan=args.ryan_bundle),
                docker_context=args.context, images=images,
                root=str(local.root_path(args.root)), port=args.port, custody=args.custody,
                reference=dict(wav=args.reference_wav, text=args.reference_text),
                measurement=evidence, ram_bytes=args.ram_bytes, threads=threads,
                pids=max(64, threads+48), cpu=dict(control=10**9, worker=threads*10**9),
                tmpfs={role:dict(bytes=65536, inodes=64) for role in ('control','worker')})


def _source(name, path, verifier):
    from service.provisioning import bundle_manifest, manifest_bytes
    descriptor = json.loads((ROOT/'requirements'/'tts'/(
        'qwen-0.6b-base.expected.json' if name == 'clone' else
        'qwen-0.6b-customvoice-cpu.expected.json')).read_text())
    qd.service_descriptor(descriptor, name)
    root = Path(path)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('setup_verified_bundle_absolute_directory_required')
    digest = qd.service_model(name)[2]
    verifier(root, digest, descriptor['checkpoint'], descriptor['revision'], descriptor['language'])
    expected = manifest_bytes(bundle_manifest(descriptor))
    if hashlib.sha256(expected).hexdigest() != digest or (root/'manifest.json').read_bytes() != expected:
        raise ValueError('setup_bundle_manifest_pin_changed')
    entries = [dict(role='models',path='bundles/'+digest+'/'+item['path'],
                    source=str(root/item['path']),bytes=item['bytes'],sha256=item['sha256'])
               for item in descriptor['files']]
    entries.append(dict(role='models',path='bundles/'+digest+'/manifest.json',
                        source=str(root/'manifest.json'),bytes=len(expected),sha256=digest))
    return descriptor, digest, entries


def _profile(evidence, identities, ram, threads):
    if evidence is None:
        profile = copy.deepcopy(json.loads((ROOT/'contracts/fixtures/v1/capabilities.json').read_text())['profiles'][0])
        profile.update(id='qwen-cpu',device='cpu',support='unqualified',installation='installed',
                       configuration='valid',readiness='not_ready',verification='synthetic',
                       checked_at=datetime.now().astimezone().isoformat(),qualification_receipt='',
                       config_revision=lane.REVISION,
                       next_action='Local execution is unqualified; enroll a reference or register Ryan.')
        profile['limits'].update(threads=threads,ram_bytes=lane.MEMORY_BYTES,
                                 disk_bytes=lane.DISK_BYTES,deadline_seconds=lane.DEADLINE_SECONDS,
                                 vram_bytes=0,expensive_slots=1,resident_models=1)
        profile['compatibility'] = [dict(operation=op,mode=mode,identity=identities[name])
            for op,mode,name in (('voice_prepare','prepare','clone'),
                                 ('speech_generate','qwen_clone','clone'),
                                 ('speech_generate','qwen_preset','Ryan'))]
        oracle.validate('profile',profile)
        return profile
    required = {'receipt','checked_at','memory_bytes','disk_bytes','deadline_seconds'}
    if type(evidence) is not dict or set(evidence) != required:
        raise ValueError('setup_explicit_profile_measurement_required')
    if (type(evidence['receipt']) is not str or not evidence['receipt'].strip() or
            'synthetic' in evidence['receipt'].lower() or
            'not-a-qualification' in evidence['receipt'].lower() or
            'estimate' in evidence['receipt'].lower() or
            any(type(evidence[k]) is not int or evidence[k] < 1 for k in required-{'receipt','checked_at'})):
        raise ValueError('setup_real_profile_measurement_required')
    try:
        stamp=datetime.fromisoformat(evidence['checked_at'])
        if stamp.tzinfo is None: raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('setup_measured_checked_at_required') from None
    profile = copy.deepcopy(json.loads((ROOT/'contracts/fixtures/v1/capabilities.json').read_text())['profiles'][0])
    profile.update(id='qwen-cpu',device='cpu',support='qualified',installation='installed',
                   configuration='valid',readiness='ready',verification='measured',
                   checked_at=evidence['checked_at'],
                   qualification_receipt=evidence['receipt'],
                   next_action='Enroll a reference WAV with its exact transcript or register Ryan.')
    profile['limits'].update(threads=threads,ram_bytes=ram,disk_bytes=evidence['disk_bytes'],
                             deadline_seconds=evidence['deadline_seconds'],vram_bytes=0,
                             expensive_slots=1,resident_models=1)
    profile['compatibility'] = [dict(operation=op,mode=mode,identity=identities[name])
        for op,mode,name in (('voice_prepare','prepare','clone'),
                             ('speech_generate','qwen_clone','clone'),
                             ('speech_generate','qwen_preset','Ryan'))]
    oracle.validate('profile',profile)
    return profile


def _context(name, descriptor, digest, profile, evidence, reference, threads, principal):
    template = copy.deepcopy(json.loads((ROOT/'contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context'])
    mode = 'qwen_clone' if name == 'clone' else 'qwen_preset'
    identity = next(c['identity'] for c in profile['compatibility'] if c['mode'] == mode)
    settings = dict(kind=mode,language=descriptor['language'],seed=None,temperature=.9,
                    top_p=1.,top_k=50,repetition_penalty=1.05,max_new_tokens=2048,instruction=None)
    spec = QwenExecution(digest,digest,descriptor['checkpoint'],descriptor['revision'],
        'clone' if name == 'clone' else 'custom_voice',descriptor['language'],
        'template-clone' if name == 'clone' else 'template-Ryan','vv0',threads,
        1024,24000*60,
        11520044,evidence['memory_bytes'] if evidence else lane.MEMORY_BYTES,24000,
        preset='Ryan' if name == 'Ryan' else None,
        reference_id=reference['id'] if name == 'clone' else None,
        reference_sha256=reference['sha256'] if name == 'clone' else None,
        reference_bytes=reference['bytes'] if name == 'clone' else 0,
        reference_frames=reference['frames'] if name == 'clone' else 0,
        reference_rate=reference['rate'] if name == 'clone' else 0,
        reference_text=reference['text'] if name == 'clone' else None,
        timeout_s=evidence['deadline_seconds'] if evidence else lane.DEADLINE_SECONDS)
    plan = template['plan']; version = plan['voice_versions'][0]
    plan['request'].update(voice_version_id=version['id'],identity=identity,
                           text='Template admission text.',fallback='forbid',variants=1,reuse='never')
    version.update(origin='clone' if name == 'clone' else 'preset',identity=identity,
                   voice_id=spec.voice_id,
                   settings=settings,settings_sha256=oracle.canonical_hash(settings),
                   configuration=(dict(mode=mode,reference_asset_id=reference['id'],
                       reference_text=reference['text'],conditioning='icl') if name == 'clone' else
                       dict(mode=mode,speaker='Ryan')),
                   reference_asset_ids=[reference['id']] if name == 'clone' else [],
                   reference_text=reference['text'] if name == 'clone' else '')
    version['provenance']=(dict(kind='synthetic',source_sha256=reference['sha256'],
                               source_id=reference['id'],missing_fields=[]) if name == 'clone' else
                           dict(kind='synthetic',source_sha256=digest,source_id='Ryan',missing_fields=[]))
    if name == 'clone': version['reference_asset_sha256']={reference['id']:reference['sha256']}
    else: version.pop('reference_asset_sha256',None)
    plan['assets'] = ([dict(id=reference['id'],principal_id=principal,
        sha256=reference['sha256'],frames=reference['frames'],sample_rate=reference['rate'],
        channels=1,media_format='wav',media_subtype='PCM_16',
        provenance=copy.deepcopy(version['provenance']))] if name == 'clone' else [])
    plan['resolved'].update(identity=dict(requested=identity,effective=identity,fallback='forbid',
                        reason='',profile_id='qwen-cpu'),profile=oracle.profile_snapshot(profile),
                        settings=settings,settings_sha256=oracle.canonical_hash(settings))
    template['bundles']=[dict(instructions=True,engine='qwen',model=identity['model'],mode=mode,
        bundle_sha256=digest,languages=descriptor['execution']['languages'],
        selectors=[] if name == 'clone' else [dict(value='Ryan',languages=descriptor['execution']['languages'])])]
    template['execution']=dict(descriptor=spec.identity(),bundles={digest:'/models/bundles/'+digest},
        references=({reference['id']:dict(path='/references/assets/'+reference['id'],
                      voice_id=spec.voice_id,voice_revision=spec.voice_revision,
                      checkpoint=spec.checkpoint,revision=spec.revision,text=reference['text'])}
                    if name == 'clone' else {}),
        hardware=dict(host=dict(system='linux',architecture='x86_64',deployment='docker'),
                      devices=[dict(family='cpu',index=0,supported=True,available=True,
                                    probed_at=evidence['checked_at'] if evidence else profile['checked_at'])]))
    oracle.frozen_plan(plan,template['bundles'])
    return template


def generate(request, *, verifier=None):
    """Read and verify pinned local sources; produce the exact installer request."""
    from worker.engines.qwen import verify_bundle
    from service import install_policy as ip
    required={'principal','bundles','docker_context','images','root','port','custody',
              'reference','measurement','ram_bytes','pids','threads','cpu','tmpfs'}
    if type(request) is not dict or set(request)!=required:
        raise ValueError('setup_closed_request_required')
    if type(request['bundles']) is not dict or set(request['bundles'])!={'clone','Ryan'}:
        raise ValueError('setup_two_verified_bundle_paths_required')
    if type(request['images']) is not dict or set(request['images'])!={'control','worker','helper'}:
        raise ValueError('setup_three_pinned_loaded_images_required')
    if any(type(image) is not str or not re.fullmatch(r'sha256:[0-9a-f]{64}',image)
           for image in request['images'].values()):
        raise ValueError('setup_pinned_image_digests_required')
    if any(type(request[k]) is not int or request[k]<1 for k in ('ram_bytes','pids','threads','port')):
        raise ValueError('setup_positive_resource_inputs_required')
    if (request['port']>65535 or request['threads']>64 or request['pids']<48 or
            type(request['docker_context']) is not str or not request['docker_context'] or
            type(request['custody']) is not str or not
            re.fullmatch(r'/sys/fs/cgroup/speech-box-qwen/[0-9a-f]{32}',request['custody'])):
        raise ValueError('setup_explicit_docker_context_and_dedicated_custody_required')
    if (type(request['cpu']) is not dict or set(request['cpu'])!={'control','worker'} or
            any(type(v) is not int or not 1<=v<=1024*10**9 for v in request['cpu'].values()) or
            type(request['tmpfs']) is not dict or set(request['tmpfs'])!={'control','worker'} or
            any(type(v) is not dict or set(v)!={'bytes','inodes'} or
                type(v['bytes']) is not int or not 65536<=v['bytes']<=1024**3 or
                type(v['inodes']) is not int or not 16<=v['inodes']<=65536
                for v in request['tmpfs'].values())):
        raise ValueError('setup_explicit_cpu_tmpfs_budgets_required')
    selected=request['reference']
    if (type(selected) is not dict or set(selected)!={'wav','text'} or
            type(selected['wav']) is not str or not Path(selected['wav']).is_absolute() or
            type(selected['text']) is not str or not selected['text'].strip()):
        raise ValueError('setup_real_template_reference_required')
    from service.ingest import _audio
    raw=read(selected['wav'],1048576)
    frames, rate, channels, subtype=_audio(io.BytesIO(raw),len(raw))
    if channels != 1 or subtype != 'PCM_16':
        raise ValueError('setup_mono_pcm16_reference_required')
    digest=hashlib.sha256(raw).hexdigest()
    reference=dict(id='setup-'+digest[:32],sha256=digest,bytes=len(raw),frames=frames,
                   rate=rate,text=selected['text'])
    sources={name:_source(name,request['bundles'][name],verifier or verify_bundle)
             for name in ('clone','Ryan')}
    identities={name:dict(engine='qwen',model=qd.PUBLIC_MODEL if name=='clone' else qd.PUBLIC_CUSTOM_MODEL,
                          bundle_sha256=sources[name][1],device='cpu',precision='float32')
                for name in sources}
    if request['ram_bytes'] < (request['measurement']['memory_bytes'] if request['measurement'] else lane.MEMORY_BYTES):
        raise ValueError('setup_ram_below_measured_model_requirement')
    if request['measurement'] is None and request['ram_bytes'] > 12*1024**3:
        raise ValueError('setup_local_ram_budget_exceeded')
    profile=_profile(request['measurement'],identities,request['measurement']['memory_bytes'] if request['measurement'] else lane.MEMORY_BYTES,request['threads'])
    contexts={name:_context(name,*sources[name][:2],profile,request['measurement'],reference,request['threads'],request['principal'])
              for name in sources}
    budget=dict(slots=1,resident_models=1,ram_bytes=request['ram_bytes'],vram_bytes=0,
                pids=request['pids'],queue=32,disk_bytes=request['measurement']['disk_bytes'] if request['measurement'] else lane.DISK_BYTES,free_bytes=0)
    config=build(request['principal'],contexts['clone'],contexts['Ryan'],profile,budget,
                 {k:request['images'][k] for k in ('control','worker')})
    entries=[item for name in ('clone','Ryan') for item in sources[name][2]]
    result=dict(configuration=config,helper_image=request['images']['helper'],inputs=entries,
                cpu=request['cpu'],tmpfs=request['tmpfs'],control_port=request['port'],
                custody=request['custody'])
    # The existing installer owns final Docker admission. Check its complete
    # input shape before any local document is created.
    manifest(entries+[dict(role='config',path='install.json',source='/pending',bytes=1,sha256='0'*64),
                      dict(role='credential',path='control.key',source='/pending',bytes=1)],
             family=qd.FAMILY,service=True)
    ip.configuration(config)
    local.root_path(request['root'])
    return result


def preflight_host_cgroup(request):
    """Require the actual authorized Linux host's pre-provisioned subtree."""
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or os.geteuid() != 0:
        raise ValueError('setup_authorized_linux_host_root_required')
    try:
        custody=Path(request['custody'])
        if not custody.is_dir() or custody.is_symlink():
            raise ValueError('setup_dedicated_cgroup_host_preflight_required')
        info=custody.stat()
        if (info.st_uid,info.st_gid,stat.S_IMODE(info.st_mode)) != (0,0,0o700):
            raise ValueError('setup_dedicated_cgroup_host_preflight_required')
        expected={'memory.max':request['ram_bytes'],'memory.swap.max':0,
                  'pids.max':request['pids']}
        for name,maximum in expected.items():
            raw=(custody/name).read_text().strip()
            if not raw.isdecimal() or int(raw)>maximum:
                raise ValueError('setup_dedicated_cgroup_host_preflight_required')
        cpu=(custody/'cpu.max').read_text().split()
        if len(cpu)!=2 or not all(part.isdecimal() for part in cpu) or int(cpu[0])>request['cpu']['worker']*int(cpu[1])//10**9:
            raise ValueError('setup_dedicated_cgroup_host_preflight_required')
        if not {'cpu','memory','pids'} <= set((custody/'cgroup.subtree_control').read_text().split()):
            raise ValueError('setup_dedicated_cgroup_host_preflight_required')
        if not (custody/'cgroup.kill').exists() or not (custody/'cgroup.procs').exists():
            raise ValueError('setup_dedicated_cgroup_host_preflight_required')
    except OSError as exc:
        raise ValueError('setup_dedicated_cgroup_host_preflight_required') from exc


def preflight_docker(request, *, docker=None):
    """Read-only host checks before minting credentials or creating a handle."""
    from service.daemon_client import Docker
    preflight_host_cgroup(request)
    try:
        docker = docker or Docker(request['docker_context'],family=qd.FAMILY)
        if not str(getattr(docker,'endpoint','')).startswith('unix://'):
            raise ValueError('setup_local_host_docker_socket_required')
        if docker.call(['info','--format','{{.Name}}']) != platform.node():
            raise ValueError('setup_docker_daemon_host_mismatch')
        if docker.call(['info','--format','{{.CgroupDriver}}']) != 'cgroupfs':
            raise ValueError('setup_cgroupfs_driver_required')
        for role in ('control','worker','helper'):
            image=request['images'][role]
            if docker.call(['image','inspect',image,'--format','{{.Id}}']) != image:
                raise ValueError('setup_pinned_loaded_'+role+'_image_required')
        docker.check()
    except (OSError, TimeoutError) as exc:
        raise ValueError('setup_local_rootful_linux_amd64_docker_required') from exc


def main(argv=None):
    p=argparse.ArgumentParser(description='Verify two local Qwen bundles and install/start the authenticated resident service.')
    p.add_argument('--request',help='Advanced closed operator request JSON')
    p.add_argument('--handle',required=True,help='New retained Docker installation handle')
    p.add_argument('--prepare-only',action='store_true',help='Write validated local inputs without Docker mutation')
    p.add_argument('--local-unqualified',action='store_true',help='Explicit bounded local execution; no qualification receipt')
    for flag in ('clone-bundle','ryan-bundle','context','control-image','worker-image',
                 'helper-image','custody','reference-wav','reference-text','measurement'):
        p.add_argument('--'+flag)
    p.add_argument('--ram-bytes',type=int)
    p.add_argument('--threads',type=int)
    p.add_argument('--principal',default='local')
    p.add_argument('--root',help='New private root; defaults to the local Qwen service root')
    p.add_argument('--port',type=int,default=DEFAULT_PORT)
    a=p.parse_args(argv)
    if not a.request and a.root is None and os.geteuid() == 0 and os.environ.get('SUDO_UID'):
        a.root=str(Path(pwd.getpwuid(int(os.environ['SUDO_UID'])).pw_dir)/'.local/state/speech-box/qwen-service')
    if a.local_unqualified and not a.request:
        if a.ram_bytes is None: a.ram_bytes=6*1024**3
        if a.threads is None: a.threads=2
        if a.custody is None:
            a.custody='/sys/fs/cgroup/speech-box-qwen/'+hashlib.sha256(os.fsencode(a.handle)).hexdigest()[:32]
    flags=('clone_bundle','ryan_bundle','context','control_image','worker_image',
           'helper_image','custody','reference_wav','reference_text',
           'ram_bytes','threads')
    if a.request:
        if any(getattr(a,key) is not None for key in flags) or a.root is not None or a.principal != 'local' or a.port != DEFAULT_PORT:
            p.error('--request cannot be combined with setup flags')
        request=json.loads(read(a.request,65536))
    else:
        missing=[key.replace('_','-') for key in flags if getattr(a,key) is None]
        if a.measurement is None and not a.local_unqualified: missing.append('measurement')
        if missing: p.error('required setup flags: '+', '.join('--'+key for key in missing))
        request=flag_request(a)
    if a.request and a.local_unqualified and request.get('measurement') is not None:
        p.error('local unqualified lane requires null measurement')
    if request.get('measurement') is None and not a.local_unqualified:
        p.error('null measurement requires --local-unqualified')
    result=generate(request)
    if not a.prepare_only:
        preflight_docker(request)
    root=local.root_path(request['root'])
    root=local.create(root,port=request['port'],principal=request['principal'])
    config_path=root/'install.json'; exclusive(config_path,result['configuration'])
    result['inputs'] += [dict(role='config',path='install.json',source=str(config_path),
                              bytes=config_path.stat().st_size,sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()),
                         dict(role='credential',path='control.key',source=str(root/'control.key'),
                              bytes=(root/'control.key').stat().st_size)]
    manifest(result['inputs'],family=qd.FAMILY,service=True)
    installer=root/'install-request.json'; exclusive(installer,result)
    if os.geteuid() == 0 and os.environ.get('SUDO_UID'):
        uid=int(os.environ['SUDO_UID']); gid=int(os.environ['SUDO_GID'])
        if uid <= 0 or uid != pwd.getpwuid(uid).pw_uid:
            raise ValueError('setup_sudo_client_identity_required')
        for path in (root/'control.key',root/'client.json',root):
            os.chown(path,uid,gid,follow_symlinks=False)
    client_command=['python3','-m','clients.cli']
    if root != local.DEFAULT_ROOT:
        client_command += ['--installation',str(root)]
    if a.prepare_only:
        print(json.dumps(dict(phase='prepared',qualification='unqualified' if request['measurement'] is None else 'operator-declared',request=str(installer),client=str(root/'client.json'),
                              client_command=client_command)))
        return
    from service.daemon_client import qwen_main
    qwen_main(['install','--context',request['docker_context'],'--request',str(installer),'--handle',a.handle])
    qwen_main(['up','--handle',a.handle])
    print(json.dumps(dict(phase='up',qualification='unqualified' if request['measurement'] is None else 'operator-declared',handle=a.handle,request=str(installer),client=str(root/'client.json'),
                          client_command=client_command)))


if __name__=='__main__': main()
