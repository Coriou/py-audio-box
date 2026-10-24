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
            spec = execution(snapshot)
            if spec is None or spec.engine != 'kokoro': raise Refused('kokoro_execution_required')
            self._contexts[principal] = json.dumps(snapshot, allow_nan=False)

    def resolve(self, principal, operation, request):
        if operation != 'speech_generate' or principal not in self._contexts:
            raise Refused('not_ready')
        context = json.loads(self._contexts[principal])
        context['plan']['request'] = copy.deepcopy(request)
        execution(context)
        return context

    def presets(self, principal):
        if principal not in self._contexts: return []
        context = json.loads(self._contexts[principal])
        spec = execution(context)
        return [dict(preset=spec.preset, engine=spec.engine, model=spec.checkpoint,
                     language=spec.language, bundle_sha256=spec.manifest_sha256,
                     supports=['voice_preset_register', 'speech_generate'], clone_prepare=False)]

    def preset_version(self, principal, preset, voice_id):
        from contracts import validate as oracle
        from service.voice_store import _id
        _id(voice_id)
        if principal not in self._contexts: raise Refused('not_ready')
        context = json.loads(self._contexts[principal])
        spec = execution(context)
        if spec.preset != preset: raise Refused('unsupported_control')
        version = copy.deepcopy(context['plan']['voice_versions'][0])
        version['voice_id'] = voice_id
        version['id'] = 'preset-' + oracle.canonical_hash([principal, voice_id, version, spec.identity()])
        return version

    def resolve_registered(self, principal, request, version):
        from contracts import validate as oracle
        if principal not in self._contexts: raise Refused('not_ready')
        template = json.loads(self._contexts[principal])
        spec = execution(template)
        expected = self.preset_version(principal, spec.preset, version['voice_id'])
        expected['reference_asset_sha256'] = {}
        if version != expected: raise Refused('unsupported_control')
        template['plan']['voice_versions'] = [copy.deepcopy(version)]
        template['plan']['request'] = copy.deepcopy(request)
        oracle.validate('speech_generate_request', request)
        execution(template)
        return template


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


class PocketContexts:
    """Trusted owner constructs principal preparation/generation templates and mappings.

    No public JSON path or descriptor input. Production reloads the immutable
    principal-owned catalogue record and binds the managed bytes before admission.
    This does not activate an installed worker or infer qualification.
    """
    def __init__(self, contexts, local_bundles, local_references):
        from worker.engines.pocket import verify_inputs
        from worker.engines.pocket_execution import PocketExecution
        from worker.engines.pocket_prepare import PocketPreparation
        self._contexts = {}
        for principal, context in contexts.items():
            snapshot = copy.deepcopy(context)
            spec = execution(snapshot)
            if type(spec) not in (PocketExecution, PocketPreparation): raise Refused('pocket_execution_required')
            if set(local_bundles) != {spec.bundle_id} or set(local_references) != {spec.reference_id}:
                raise Refused('owned_mapping_required')
            verify_inputs(spec, local_bundles, local_references)
            self._contexts[principal] = json.dumps(snapshot, allow_nan=False)

    def resolve(self, principal, operation, request):
        if principal not in self._contexts:
            raise Refused('not_ready')
        context = json.loads(self._contexts[principal])
        if operation != context['plan']['operation']: raise Refused('not_ready')
        from contracts import validate as oracle
        oracle.validate(operation+'_request', request)
        if operation == 'voice_prepare' and request['reference_text'] != '': raise Refused('unsupported_parameter')
        if operation == 'speech_generate' and request.get('voice_version_id') != context['plan']['voice_versions'][0]['id']:
            raise Refused('saved_version_required')
        context['plan']['request'] = copy.deepcopy(request)
        execution(context)
        return context


class QwenContexts:
    """Operator-owned exact principal/version/profile and sealed local mappings.

    Linux generation only. No installer activation or caller-selected descriptor.
    Hardware provenance expires at resolution/admission, never in history readers.
    """
    def __init__(self, contexts, local_bundles, local_references, inventory, profiles):
        from worker.engines.qwen_execution import inventory_wire, current_hardware
        from contracts import validate as oracle
        self._contexts = {}
        for principal, context in contexts.items():
            snapshot = copy.deepcopy(context)
            spec = execution(snapshot)
            if spec is None or spec.engine != 'qwen': raise Refused('qwen_execution_required')
            if snapshot['execution']['hardware'] != inventory_wire(inventory): raise Refused('hardware_identity')
            current_hardware(spec, inventory)
            oracle.admission(dict(operation='speech_generate',input_sha256=oracle.plan_hash(snapshot['plan'])),snapshot,profiles)
            verify_qwen_local(snapshot, local_bundles, local_references, inventory)
            self._contexts[principal] = json.dumps(snapshot, allow_nan=False)
        self._inventory = copy.deepcopy(inventory)
        self._bundles = copy.deepcopy(local_bundles)
        self._references = copy.deepcopy(local_references)

    def resolve(self, principal, operation, request):
        from contracts import validate as oracle
        from worker.engines.qwen_execution import current_hardware
        if operation != 'speech_generate' or principal not in self._contexts: raise Refused('not_ready')
        oracle.validate('speech_generate_request',request)
        context = json.loads(self._contexts[principal])
        if request['voice_version_id'] != context['plan']['voice_versions'][0]['id']: raise Refused('saved_version_required')
        context['plan']['request'] = copy.deepcopy(request)
        spec = execution(context)
        current_hardware(spec,self._inventory)
        verify_qwen_local(context,self._bundles,self._references,self._inventory)
        return context


def verify_qwen_local(context, bundles, references, inventory, *, owner_uid=None):
    """Same full verifier for control and independent Linux supervisor."""
    spec = execution(context)
    if set(bundles) != {spec.bundle_id} or set(references) != set(context['execution']['references']): raise Refused('owned_mapping_required')
    from worker.engines.qwen import verify_service_inputs
    verify_service_inputs(spec,bundles,references,inventory,owner_uid=owner_uid)
