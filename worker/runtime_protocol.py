"""Closed private lifecycle-v1. Canonical execution/result remains 1.3.0."""
import hashlib
import hmac
from worker.transport import Session, Refused, authenticate, receive, send

VERSION = 'lifecycle-v1'
OPS = frozenset(('hello', 'prepare', 'start', 'observe', 'stop', 'collect', 'ack'))


def session(secret, uid, gid):
    key = hmac.new(bytes.fromhex(secret), b'private-lifecycle-v1/auth', hashlib.sha256).hexdigest()
    return Session(key, hashlib.sha256(b'private-lifecycle-v1').hexdigest(), uid, gid)


def request(value):
    if (not isinstance(value, dict) or set(value) != {'version', 'op', 'body'} or
            value['version'] != VERSION or value['op'] not in OPS or not isinstance(value['body'], dict)):
        raise Refused('closed_lifecycle_frame_required')
    return value['op'], value['body']


def execution_session(secret, binding, uid, gid):
    key = hmac.new(bytes.fromhex(secret), ('execution-allocation-v1/'+binding).encode(), hashlib.sha256).hexdigest()
    return Session(key, binding, uid, gid)


def execution(context):
    """Validate a private engine snapshot against the complete frozen plan."""
    from worker.engines.execution import Execution
    value = context.get('execution')
    plan = context['plan']; profile = plan['resolved']['profile']
    if value is None:
        if profile['id'] in ('kokoro-cpu', 'pocket-cpu', 'qwen-cpu', 'qwen-cuda', 'qwen-mps'): raise Refused('execution_required')
        return None
    if set(context) != {'plan', 'bundles', 'execution'}: raise Refused('closed_execution_context')
    if type(value) is not dict or set(value) not in ({'descriptor', 'bundles'}, {'descriptor', 'bundles', 'references'}, {'descriptor','bundles','references','hardware'}, {'descriptor','bundles','references','hardware','native'}):
        raise Refused('closed_execution_context')
    if type(value['descriptor']) is not dict: raise Refused('closed_execution_required')
    if value['descriptor'].get('engine') == 'qwen':
        return qwen_execution(context)
    if value['descriptor'].get('engine') == 'pocket-tts':
        return pocket_preparation(context) if value['descriptor'].get('phase') == 'prepare' else pocket_execution(context)
    if set(value) != {'descriptor', 'bundles'}: raise Refused('closed_execution_context')
    spec = Execution.from_wire(value['descriptor'])
    if value['bundles'] != {spec.bundle_id: '/models/bundles/'+spec.bundle_id}:
        raise Refused('owned_mount_mapping')
    identity = plan['resolved']['identity']['effective']
    versions = plan['voice_versions']; request = plan['request']
    if (profile['id'] != 'kokoro-cpu' or plan['operation'] != 'speech_generate' or
            request['fallback'] != 'forbid' or request['identity'] != identity or
            request['variants'] != 1 or request['reuse'] != 'never' or
            len(versions) != 1 or plan['assets'] or
            identity != dict(engine='kokoro', model=spec.checkpoint,
                bundle_sha256=spec.manifest_sha256, device='cpu', precision='float32')):
        raise Refused('execution_plan_mismatch')
    version = versions[0]
    if (request['voice_version_id'] != version['id'] or version['identity'] != identity or
            version['origin'] != 'preset' or version['reference_asset_ids'] or
            version['configuration'] != dict(mode='kokoro_preset', preset=spec.preset, language=spec.language) or
            plan['resolved']['settings'] != dict(kind='kokoro_preset', language=spec.language)):
        raise Refused('execution_preset_mismatch')
    limits = profile['limits']
    if (spec.threads != limits['threads'] or spec.memory_bytes != limits['ram_bytes'] or
            spec.max_frames*2 > limits['decoded_ticks'] or
            spec.max_output_bytes > limits['upload_bytes'] or
            spec.max_output_bytes*2 > limits['disk_bytes'] or
            len(request['text'].encode('utf-8')) > spec.max_text_bytes):
        raise Refused('execution_budget_mismatch')
    return spec


def generated_source(context):
    spec = execution(context)
    if spec is None: raise Refused('execution_required')
    from worker.transport import canonical
    return spec.engine+'-execution-sha256:'+hashlib.sha256(canonical(spec.identity())).hexdigest()


def dispatch_mode(spec):
    return {'kokoro': 'kokoro', 'pocket-tts': 'pocket', 'qwen': 'qwen'}[spec.engine]


def pocket_execution(context):
    from worker.engines.pocket_execution import PocketExecution
    value = context['execution']; plan = context['plan']
    if set(value) != {'descriptor', 'bundles', 'references'}:
        raise Refused('closed_execution_context')
    spec = PocketExecution.from_wire(value['descriptor'])
    if (value['bundles'] != {spec.bundle_id: '/models/bundles/'+spec.bundle_id} or
            value['references'] != {spec.reference_id: '/references/assets/'+spec.reference_id}):
        raise Refused('owned_mount_mapping')
    identity = dict(engine=spec.engine, model=spec.checkpoint, bundle_sha256=spec.manifest_sha256,
                    device='cpu', precision='float32')
    request = plan['request']; versions = plan['voice_versions']; assets = plan['assets']
    if (plan['operation'] != 'speech_generate' or plan['resolved']['profile']['id'] != 'pocket-cpu' or
            request['identity'] != identity or plan['resolved']['identity']['effective'] != identity or
            request['fallback'] != 'forbid' or request['variants'] != 1 or request['reuse'] != 'never' or
            len(versions) != 1 or len(assets) != 1):
        raise Refused('execution_plan_mismatch')
    version = versions[0]; asset = assets[0]
    settings = dict(kind='pocket_clone', language=spec.language, seed=None)
    if (request['voice_version_id'] != version['id'] or version['identity'] != identity or
            version['origin'] != 'clone' or version['reference_asset_ids'] != [spec.reference_id] or
            version.get('reference_asset_sha256') != {spec.reference_id: spec.reference_sha256} or
            version['configuration'] != dict(mode='pocket_clone', language=spec.language, reference_asset_id=spec.reference_id) or
            version['reference_text'] != '' or version['settings'] != settings or
            plan['resolved']['settings'] != settings or
            any(asset[k] != v for k,v in dict(id=spec.reference_id, sha256=spec.reference_sha256,
                frames=spec.reference_frames, sample_rate=spec.reference_rate, channels=1,
                media_format='wav', media_subtype='PCM_16').items())):
        raise Refused('execution_reference_mismatch')
    limits = plan['resolved']['profile']['limits']
    if (spec.threads != limits['threads'] or spec.memory_bytes != limits['ram_bytes'] or
            spec.max_frames*48000 > limits['decoded_ticks']*spec.sample_rate or
            spec.max_output_bytes > limits['upload_bytes'] or
            2*(spec.max_output_bytes+spec.reference_bytes) > limits['disk_bytes'] or
            len(request['text'].encode('utf-8')) > spec.max_text_bytes):
        raise Refused('execution_budget_mismatch')
    return spec


def pocket_preparation(context):
    from worker.engines.pocket_prepare import PocketPreparation
    value = context['execution']; plan = context['plan']
    spec = PocketPreparation.from_wire(value['descriptor'])
    if (set(value) != {'descriptor', 'bundles', 'references'} or
            value['bundles'] != {spec.bundle_id: '/models/bundles/'+spec.bundle_id} or
            value['references'] != {spec.reference_id: '/references/assets/'+spec.reference_id}):
        raise Refused('owned_mount_mapping')
    identity = dict(engine=spec.engine, model=spec.checkpoint, bundle_sha256=spec.manifest_sha256,
                    device='cpu', precision='float32')
    request = plan['request']; resolved = plan['resolved']; roles = plan.get('reference_roles', {})
    if request.get('reference_text') != '': raise Refused('unsupported_parameter')
    if (plan['operation'] != 'voice_prepare' or resolved['profile']['id'] != 'pocket-cpu' or
            request['identity'] != identity or resolved['identity']['effective'] != identity or
            resolved['identity']['fallback'] != 'forbid' or plan['voice_versions'] or
            len(plan['assets']) != 1 or request['asset_id'] != spec.reference_id or
            (request['in_tick'], request['out_tick']) != (spec.in_tick, spec.out_tick) or
            resolved['settings'] != dict(kind='prepare', conditioning='pocket', output_format='wav_pcm16') or
            resolved['reuse_settings'] != dict(kind='pocket_clone', language=spec.language, seed=None) or
            resolved['reuse_configuration'] != dict(mode='pocket_clone', language=spec.language,
                reference_asset_id=roles.get('reference_asset_id')) or
            roles.get('version') != 'reference-roles-v1' or roles.get('reference_kind') != 'prepared_reference' or
            not request.get('audition_text', '').strip() or roles.get('audition_text') != request['audition_text'] or
            len(roles['audition_text'].encode('utf-8')) > spec.max_text_bytes):
        raise Refused('execution_plan_mismatch')
    asset = plan['assets'][0]
    if any(asset[k] != v for k,v in dict(id=spec.reference_id, sha256=spec.reference_sha256,
            frames=spec.reference_frames, sample_rate=spec.reference_rate, channels=1,
            media_format='wav', media_subtype='PCM_16').items()):
        raise Refused('execution_reference_mismatch')
    limits = resolved['profile']['limits']
    if max(spec.reference_bytes, spec.max_output_bytes) > 1048576:
        raise Refused('preparation_publication_limit')
    prepared_bytes = 44 + 2*(spec.out_tick-spec.in_tick)*spec.reference_rate//48000
    if (spec.threads != limits['threads'] or spec.memory_bytes != limits['ram_bytes'] or
            spec.max_frames*48000 > limits['decoded_ticks']*spec.sample_rate or
            spec.out_tick-spec.in_tick > limits['decoded_ticks'] or
            max(spec.max_output_bytes, spec.reference_bytes, prepared_bytes) > limits['upload_bytes'] or
            2*spec.max_output_bytes + 4*spec.reference_bytes + 2*prepared_bytes > limits['disk_bytes'] or
            len(roles['audition_text']) > limits['text_chars']):
        raise Refused('execution_budget_mismatch')
    return spec


def qwen_execution(context):
    lane = "native" if context["plan"]["resolved"]["profile"]["id"] == "qwen-mps" else "linux"
    return qwen_relation(context, lane=lane)


def qwen_relation(context, *, lane):
    from worker.engines.qwen_execution import QwenExecution, inventory_from_wire
    value=context['execution']; plan=context['plan']; resolved=plan['resolved']
    if lane not in ('linux', 'native'): raise Refused('closed_qwen_lane')
    keys = {'descriptor','bundles','references','hardware'}
    if set(value) != keys and not (lane == 'native' and set(value) == keys | {'native'}): raise Refused('closed_execution_context')
    spec=QwenExecution.from_wire(value['descriptor'])
    inventory_from_wire(value['hardware'])
    if spec.task not in ('custom_voice','clone'): raise Refused('qwen_generation_only')
    if lane == 'native':
        if (spec.device,spec.dtype,spec.attention) != ('mps','float32','sdpa') or resolved['profile']['topology'] != 'packaged_native': raise Refused('native_topology')
    elif spec.device == 'mps': raise Refused('qwen_generation_only')
    if value['bundles'] != {spec.bundle_id:'/models/bundles/'+spec.bundle_id}: raise Refused('owned_mount_mapping')
    identity=dict(engine='qwen',model=spec.checkpoint,bundle_sha256=spec.manifest_sha256,device='metal' if lane == 'native' else ('cuda' if spec.device.startswith('cuda:') else 'cpu'),precision=spec.dtype)
    request=plan['request']; versions=plan['voice_versions']
    profile_id='qwen-mps' if lane == 'native' else ('qwen-cuda' if spec.device.startswith('cuda:') else 'qwen-cpu')
    if (plan['operation'] != 'speech_generate' or resolved['profile']['id'] != profile_id or
        request['identity'] != identity or resolved['identity']['effective'] != identity or
        request['fallback'] != 'forbid' or request['variants'] != 1 or request['reuse'] != 'never' or len(versions) != 1): raise Refused('execution_plan_mismatch')
    v=versions[0]; mode='qwen_clone' if spec.task == 'clone' else 'qwen_preset'
    settings=dict(kind=mode,language=spec.language,seed=None,instruction=spec.instruct,
                  **{k:getattr(spec,k) for k in ('temperature','top_p','top_k','repetition_penalty','max_new_tokens')})
    if (v['id'] != request['voice_version_id'] or v['id'] != spec.voice_revision or v['voice_id'] != spec.voice_id or
        v['identity'] != identity or v['settings'] != settings or resolved['settings'] != settings): raise Refused('execution_voice_mismatch')
    if spec.task == 'custom_voice':
        if (v['origin'] != 'preset' or v['reference_asset_ids'] or plan['assets'] or value['references'] or
            v['configuration'] != dict(mode=mode,speaker=spec.preset)): raise Refused('execution_preset_mismatch')
    else:
        expected=dict(path='/references/assets/'+spec.reference_id,voice_id=spec.voice_id,
                      voice_revision=spec.voice_revision,checkpoint=spec.checkpoint,revision=spec.revision,text=spec.reference_text)
        if (value['references'] != {spec.reference_id:expected} or v['origin'] != 'clone' or
            v['reference_asset_ids'] != [spec.reference_id] or v.get('reference_asset_sha256') != {spec.reference_id:spec.reference_sha256} or
            v['reference_text'] != spec.reference_text or v['configuration'] != dict(mode=mode,reference_asset_id=spec.reference_id,
                reference_text=spec.reference_text,conditioning='x_vector' if spec.x_vector_only else 'icl') or len(plan['assets']) != 1): raise Refused('execution_reference_mismatch')
        a=plan['assets'][0]
        if any(a[k] != v for k,v in dict(id=spec.reference_id,sha256=spec.reference_sha256,frames=spec.reference_frames,
                sample_rate=spec.reference_rate,channels=1,media_format='wav',media_subtype='PCM_16').items()): raise Refused('execution_reference_mismatch')
    limits=resolved['profile']['limits']
    if (spec.threads != limits['threads'] or spec.memory_bytes != limits['ram_bytes'] or
        spec.max_frames*48000 > limits['decoded_ticks']*spec.sample_rate or spec.max_output_bytes > limits['upload_bytes'] or
        2*(spec.max_output_bytes+spec.reference_bytes) > limits['disk_bytes'] or
        len(request['text'].encode()) > spec.max_text_bytes): raise Refused('execution_budget_mismatch')
    return spec
