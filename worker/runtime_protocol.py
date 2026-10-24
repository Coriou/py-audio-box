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
    """Validate a private Kokoro snapshot against the complete frozen plan."""
    from worker.engines.execution import Execution
    value = context.get('execution')
    plan = context['plan']; profile = plan['resolved']['profile']
    if value is None:
        if profile['id'] == 'kokoro-cpu': raise Refused('execution_required')
        return None
    if set(context) != {'plan', 'bundles', 'execution'}: raise Refused('closed_execution_context')
    if type(value) is not dict or set(value) != {'descriptor', 'bundles'}:
        raise Refused('closed_execution_context')
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
    return 'kokoro-execution-sha256:'+hashlib.sha256(canonical(spec.identity())).hexdigest()
