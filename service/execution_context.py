"""Operator-only immutable local snapshots. No discovery/download or readiness inference."""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from worker.runtime_protocol import execution
from worker.transport import Refused


def verify_local(context, local_bundles, *, policy=None):
    if policy is not None:
        from service import install_policy as ip
        ip.require(policy)
    owner = os.getuid() if policy is None else 10001
    spec = execution(context)
    if spec is None: return
    if set(local_bundles) != {spec.bundle_id}: raise Refused('owned_bundle_required')
    root = Path(local_bundles[spec.bundle_id])
    # Trusted operator storage only; no hostile concurrent mutation claim.
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        raise Refused('owned_bundle_required')
    for path in (root, *root.iterdir()):
        info = path.lstat()
        if (info.st_uid != owner or info.st_mode & 0o222 or
                not (stat.S_ISDIR(info.st_mode) if path == root else stat.S_ISREG(info.st_mode) and info.st_nlink == 1)):
            raise Refused('sealed_bundle_required')
    with (root/'manifest.json').open('rb') as stream:
        raw = stream.read(65537)
    if len(raw)>65536 or hashlib.sha256(raw).hexdigest()!=spec.manifest_sha256:
        raise Refused('bundle_identity')
    from lib.tts_backends.assets import verified_bundle
    _, manifest = verified_bundle(root, 'kokoro')
    if (manifest.get('checkpoint'), manifest.get('revision'), manifest.get('language')) != (spec.checkpoint, spec.revision, spec.language):
        raise Refused('bundle_identity')


class KokoroContexts:
    """Explicit principal template + sealed provisioned bundle; no public paths.

    RuntimeSupervisor separately requires the identical owned mount mapping.
    Installed bind-mount/read-only topology still requires actual qualification.
    """
    def __init__(self, contexts, local_bundles):
        self._contexts = {}
        for principal, context in contexts.items():
            snapshot = copy.deepcopy(context)
            verify_local(snapshot, local_bundles)
            if execution(snapshot) is None: raise Refused('execution_required')
            self._contexts[principal] = json.dumps(snapshot, allow_nan=False)

    def resolve(self, principal, operation, request):
        if operation != 'speech_generate' or principal not in self._contexts:
            raise Refused('not_ready')
        context = json.loads(self._contexts[principal])
        context['plan']['request'] = copy.deepcopy(request)
        execution(context)
        return context


def installed_configuration(path, *, policy):
    from service import install_policy as ip
    from service.auth import private_input
    from service.recovery import host_json
    ip.require(policy)
    return ip.configuration(host_json(private_input(path, mode=0o440, limit=48000)))


def installed_mapping(configuration, *, policy):
    from service import install_policy as ip
    ip.require(policy)
    config = ip.configuration(configuration)
    mapping = {key: value['path'] for key, value in config['bundles'].items()}
    if any(context['execution']['bundles'] != mapping for context in config['contexts'].values()):
        raise Refused('unregistered_mount_mapping')
    # Verify every hash and immutable tree; no discovery or provisioning.
    for context in config['contexts'].values(): verify_local(context, mapping, policy=policy)
    return mapping
