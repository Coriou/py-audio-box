"""Private finite Qwen CPU daemon transport; publication and history remain v1.

The selection token is trusted executable authority, never retained metadata.
"""
import copy
from service import install_policy as ip

FAMILY = object()
LAUNCH = 'installed-qwen-cpu-daemon-volumes-v1'
EVIDENCE = 'daemon-volumes-inspection-v1'
POLICY = 'docker-qwen-cpu-split-v1'
SCOPE = 'owned-qwen-cpu-v1'
PUBLIC_MODEL = 'Qwen3-TTS-12Hz-0.6B-Base'
PUBLIC_CUSTOM_MODEL = 'Qwen3-TTS-12Hz-0.6B-CustomVoice'
CHECKPOINT = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
REVISION = '5d83992436eae1d760afd27aff78a71d676296fc'
SERVICE_POLICY = 'qwen-resident-cpu-v1'
CUSTOM_CHECKPOINT = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
CUSTOM_REVISION = '85e237c12c027371202489a0ec509ded67b5e4b5'
CUSTOM_MANIFEST_SHA256 = 'dc1b15de72cbc49aeb4d2fa558e0f08e2e784a257a7b48243f6f5bfb8aeeb89a'


def public_model(spec):
    """Only the exact reviewed CPU checkpoints receive public model names."""
    if (getattr(spec,'task',None) == 'clone' and getattr(spec,'device',None) == 'cpu' and
            (spec.checkpoint,spec.revision) == (CHECKPOINT,REVISION)):
        return PUBLIC_MODEL
    if (getattr(spec,'task',None) == 'custom_voice' and getattr(spec,'preset',None) == 'Ryan' and
            (spec.checkpoint,spec.revision) == (CUSTOM_CHECKPOINT,CUSTOM_REVISION)):
        return PUBLIC_CUSTOM_MODEL
    return spec.checkpoint


def configuration(value):
    value = ip.configuration(value)
    if value['ownership_policy'] != POLICY:
        raise ValueError('qwen_cpu_daemon_configuration_required')
    return value


def binding(value):
    from service import daemon_volumes as dv
    if type(value) is not dict or value.get('policy') != LAUNCH:
        raise ValueError('qwen_daemon_launch_required')
    projected = copy.deepcopy(value)
    projected['policy'] = dv.POLICY
    dv._binding(projected, shared='model_storage' in projected, custody='custody' in projected)
    return copy.deepcopy(value)


def binding_configuration(value, config):
    from service import daemon_volumes as dv
    binding(value); configuration(config)
    if dv.digest(config) != value['configuration_digest']:
        raise ValueError('daemon_configuration_binding')
    if ('custody' in value) != ip.qwen_service(config):
        raise ValueError('qwen_service_delegated_cgroup_required')
    if 'model_storage' in value:
        if ip.qwen_service(config):
            raise ValueError('qwen_service_two_bundle_publication_required')
        key = value['model_storage']['manifest_sha256']
        if set(config['bundles']) != {key} or config['bundles'][key]['manifest_sha256'] != key:
            raise ValueError('shared_configuration_binding')
        for context in config['contexts'].values():
            spec = context['execution']['descriptor']
            if spec['bundle_id'] != key or spec['manifest_sha256'] != key:
                raise ValueError('shared_configuration_binding')
    return value


def installation(value):
    value = ip.installation(value)
    configuration(value['configuration'])
    binding_configuration(value['launch'], value['configuration'])
    if value['deployment_id'] != value['launch']['installation']:
        raise ValueError('qwen_daemon_installation_binding')
    return value



def validate_spec(spec, *, service=False):
    from worker.runtime_protocol import managed_qwen_cpu
    if service:
        from worker.runtime_protocol import resident_qwen_model
        resident_qwen_model(spec)
        if ((spec.dtype, spec.attention) != ('float32', 'sdpa') or
                (spec.task == 'clone' and (spec.x_vector_only or spec.reference_text is None))):
            raise ValueError('installed_qwen_cpu_execution_required')
    elif (not managed_qwen_cpu(spec) or spec.reference_text is not None or spec.manifest_sha256 != MANIFEST_SHA256 or
          (spec.checkpoint, spec.revision, spec.dtype, spec.attention) !=
          (CHECKPOINT, REVISION, 'float32', 'sdpa')):
        raise ValueError('installed_qwen_cpu_execution_required')


MANIFEST_SHA256 = 'd2d42c4a726518f747bb094f5b5180746a4a965020f9abc5ee739ee3d1c76fa8'
SERVICE_MODELS = {
    'clone': (CHECKPOINT, REVISION, MANIFEST_SHA256),
    'Ryan': (CUSTOM_CHECKPOINT, CUSTOM_REVISION, CUSTOM_MANIFEST_SHA256),
}


def service_model(name):
    """Opt-in resident worker selection. A worker must unload before switching.

    This is a closed identity seam, not a claim that a model is installed or loaded.
    """
    if type(name) is not str or name not in SERVICE_MODELS:
        raise ValueError('qwen_service_model_unavailable')
    return SERVICE_MODELS[name]


def service_descriptor(value, name):
    """Bind the service's two reviewed CPU descriptors to their exact manifests."""
    from service.provisioning import validate_expected, manifest_sha256
    expected = validate_expected(value)
    checkpoint, revision, digest = service_model(name)
    if (expected['checkpoint'], expected['revision'], manifest_sha256(expected)) != (checkpoint, revision, digest):
        raise ValueError('qwen_service_descriptor_required')
    if expected['execution']['devices'] != [dict(device='cpu', dtype='float32', attention='sdpa')]:
        raise ValueError('qwen_service_cpu_descriptor_required')
    if (name == 'clone' and expected['execution']['presets'] or
            name == 'Ryan' and expected['execution']['presets'] != ['Ryan']):
        raise ValueError('qwen_service_presets_required')
    return expected


def resident_transition(current, requested, active_jobs):
    """Decision only; worker owns actual close/reap/load and must serialize it.

    A switch cannot run while any job still owns the resident checkpoint.
    There is no second resident slot and no silent fallback to another model.
    """
    service_model(requested)
    if current is not None:
        service_model(current)
    if type(active_jobs) is not int or active_jobs < 0:
        raise ValueError('qwen_service_active_jobs')
    if current == requested:
        return 'reuse'
    if active_jobs:
        raise ValueError('qwen_service_model_busy')
    return 'load' if current is None else 'unload_then_load'


def verified_service_models(descriptors, bundles):
    """Read-only source status; never calls inference or claims runtime readiness."""
    from pathlib import Path
    from worker.engines.qwen import verify_bundle
    if type(descriptors) is not dict or type(bundles) is not dict or set(descriptors) != set(SERVICE_MODELS) or set(bundles) != set(SERVICE_MODELS):
        raise ValueError('qwen_service_exact_models_required')
    result = {}
    for name in SERVICE_MODELS:
        expected = service_descriptor(descriptors[name], name)
        checkpoint, revision, digest = service_model(name)
        path = Path(bundles[name])
        try:
            verify_bundle(path, digest, checkpoint, revision, expected['language'])
        except (OSError, ValueError) as exc:
            result[name] = dict(source_verified=False, reason=type(exc).__name__)
        else:
            result[name] = dict(source_verified=True)
    return dict(policy=SERVICE_POLICY, resident_limit=1, models=result,
                runtime_qualified=False)


def manifest(value, *, service=False):
    import hashlib
    from service.provisioning import manifest_bytes
    allowed=(MANIFEST_SHA256,CUSTOM_MANIFEST_SHA256) if service else (MANIFEST_SHA256,)
    if hashlib.sha256(manifest_bytes(value)).hexdigest() not in allowed:
        raise ValueError('installed_qwen_exact_manifest_required')
    return value

# 2,516,100,892 pinned bytes reached only 973,078,528 bytes in the
# observed 100-second import. 600 seconds also covers full descriptor rehash
# at the 4 GiB admission ceiling, without chunk renewal. Transfer alone
# would require 6.83 MiB/s at that ceiling; final rehash shares the same cap.
IMPORT_SECONDS = 600


def import_deadline(expires):
    """Convert the sender's original UTC expiry once to a local monotonic cap.

    Docker's host and helper monotonic clocks have different epochs. UTC is
    used only at this handshake; subsequent progress cannot renew either cap.
    """
    import math
    import time
    if type(expires) not in (int, float) or not math.isfinite(expires):
        raise ValueError('qwen_import_deadline')
    remaining = expires - time.time()
    if not 0 < remaining <= IMPORT_SECONDS:
        raise ValueError('qwen_import_deadline')
    return time.monotonic() + remaining
