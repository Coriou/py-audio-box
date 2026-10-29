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
from pathlib import Path
import re

from contracts import validate as oracle
from service import installed_qwen_daemon as qd
from service import qwen_local_config as local
from service.daemon_client import exclusive, manifest, read
from service.qwen_service_config import build
from worker.engines.qwen_execution import QwenExecution


ROOT = Path(__file__).resolve().parents[1]


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
    required = {'receipt','checked_at','memory_bytes','disk_bytes','deadline_seconds'}
    if type(evidence) is not dict or set(evidence) != required:
        raise ValueError('setup_explicit_profile_measurement_required')
    if (type(evidence['receipt']) is not str or not evidence['receipt'].strip() or
            'synthetic' in evidence['receipt'].lower() or
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
        11520044,evidence['memory_bytes'],24000,
        preset='Ryan' if name == 'Ryan' else None,
        reference_id=reference['id'] if name == 'clone' else None,
        reference_sha256=reference['sha256'] if name == 'clone' else None,
        reference_bytes=reference['bytes'] if name == 'clone' else 0,
        reference_frames=reference['frames'] if name == 'clone' else 0,
        reference_rate=reference['rate'] if name == 'clone' else 0,
        reference_text=reference['text'] if name == 'clone' else None,
        timeout_s=evidence['deadline_seconds'])
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
                                    probed_at=evidence['checked_at'])]))
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
    if request['ram_bytes'] < request['measurement']['memory_bytes']:
        raise ValueError('setup_ram_below_measured_model_requirement')
    profile=_profile(request['measurement'],identities,request['measurement']['memory_bytes'],request['threads'])
    contexts={name:_context(name,*sources[name][:2],profile,request['measurement'],reference,request['threads'],request['principal'])
              for name in sources}
    budget=dict(slots=1,resident_models=1,ram_bytes=request['ram_bytes'],vram_bytes=0,
                pids=request['pids'],queue=32,disk_bytes=request['measurement']['disk_bytes'],free_bytes=0)
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


def preflight_docker(request, *, docker=None):
    """Read-only host checks before minting credentials or creating a handle."""
    from service.daemon_client import Docker
    try:
        docker = docker or Docker(request['docker_context'],family=qd.FAMILY)
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
    p.add_argument('--request',required=True,help='Small operator input JSON; see QWEN-RESIDENT-SERVICE.md')
    p.add_argument('--handle',required=True,help='New retained Docker installation handle')
    p.add_argument('--prepare-only',action='store_true',help='Write validated local inputs without Docker mutation')
    a=p.parse_args(argv)
    request=json.loads(read(a.request,65536))
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
    if a.prepare_only:
        print(json.dumps(dict(phase='prepared',request=str(installer),client=str(root/'client.json'))))
        return
    from service.daemon_client import qwen_main
    qwen_main(['install','--context',request['docker_context'],'--request',str(installer),'--handle',a.handle])
    qwen_main(['up','--handle',a.handle])
    print(json.dumps(dict(phase='up',handle=a.handle,client=str(root/'client.json'))))


if __name__=='__main__': main()
