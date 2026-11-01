"""Private finite Pocket CPU daemon transport; publication and history remain v1.

The selection token is trusted executable authority, never retained metadata.
"""
import copy
from service import install_policy as ip

FAMILY = object()
LAUNCH = 'installed-pocket-daemon-volumes-v2'
EVIDENCE = 'daemon-volumes-inspection-v1'
POLICY = ip.POCKET_V2_POLICY
SCOPE = ip.POCKET_V2_SCOPE


def configuration(value):
    value = ip.configuration(value)
    if value['ownership_policy'] != POLICY:
        raise ValueError('pocket_daemon_v2_configuration_required')
    return value


def binding(value):
    from service import daemon_volumes as dv
    if type(value) is not dict or value.get('policy') != LAUNCH:
        raise ValueError('pocket_daemon_launch_required')
    projected = copy.deepcopy(value)
    projected['policy'] = dv.POLICY
    dv.binding(projected)
    return copy.deepcopy(value)


def installation(value):
    value = ip.installation(value)
    configuration(value['configuration'])
    binding(value['launch'])
    if value['deployment_id'] != value['launch']['installation']:
        raise ValueError('pocket_daemon_installation_binding')
    return value


def require(family):
    from service import installed_qwen_daemon as qd
    if family is None or family is FAMILY or family is qd.FAMILY: return
    from service import installed_export as ex
    if family is not ex.FAMILY:
        raise ValueError('explicit_installed_family_required')


def launch(family):
    from service import daemon_volumes as dv
    if family is None: return dv.POLICY
    if family is FAMILY: return LAUNCH
    from service import installed_qwen_daemon as qd
    if family is qd.FAMILY: return qd.LAUNCH
    from service import installed_export as ex
    require(family)
    return ex.LAUNCH


def is_export(family):
    from service import installed_qwen_daemon as qd
    if family is None or family is FAMILY or family is qd.FAMILY: return False
    from service import installed_export as ex
    if family is not ex.FAMILY: raise ValueError('explicit_installed_family_required')
    return True
