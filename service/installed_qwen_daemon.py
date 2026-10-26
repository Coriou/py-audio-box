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
    dv.binding(projected)
    return copy.deepcopy(value)


def installation(value):
    value = ip.installation(value)
    configuration(value['configuration'])
    binding(value['launch'])
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
