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
CHECKPOINT = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
REVISION = '5d83992436eae1d760afd27aff78a71d676296fc'


def public_model(spec):
    """One exact CPU binding; never normalize arbitrary checkpoints."""
    from worker.runtime_protocol import managed_qwen_cpu
    if managed_qwen_cpu(spec) and (spec.checkpoint, spec.revision) == (CHECKPOINT, REVISION):
        return PUBLIC_MODEL
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
    dv._binding(projected, shared='model_storage' in projected)
    return copy.deepcopy(value)


def binding_configuration(value, config):
    from service import daemon_volumes as dv
    binding(value); configuration(config)
    if dv.digest(config) != value['configuration_digest']:
        raise ValueError('daemon_configuration_binding')
    if 'model_storage' in value:
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



def validate_spec(spec):
    from worker.runtime_protocol import managed_qwen_cpu
    if (not managed_qwen_cpu(spec) or spec.reference_text is not None or spec.manifest_sha256 != MANIFEST_SHA256 or
            (spec.checkpoint, spec.revision, spec.dtype, spec.attention) !=
            (CHECKPOINT, REVISION, 'float32', 'sdpa')):
        raise ValueError('installed_qwen_cpu_execution_required')


MANIFEST_SHA256 = 'd2d42c4a726518f747bb094f5b5180746a4a965020f9abc5ee739ee3d1c76fa8'


def manifest(value):
    import hashlib
    from service.provisioning import manifest_bytes
    if hashlib.sha256(manifest_bytes(value)).hexdigest() != MANIFEST_SHA256:
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
