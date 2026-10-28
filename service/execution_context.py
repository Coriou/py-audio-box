"""Operator-only immutable local snapshots. No discovery/download or readiness inference."""
from lib.spoken_text import freeze
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
    if spec.engine == 'qwen':
        from worker.engines.qwen import verify_bundle
        if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
            raise Refused('owned_bundle_required')
        for directory, dirs, files in os.walk(root,followlinks=False):
            for path in (Path(directory), *(Path(directory)/n for n in dirs+files)):
                info=path.lstat()
                if info.st_uid != owner or info.st_mode & 0o222 or path.is_symlink():
                    raise Refused('sealed_bundle_required')
        _, manifest=verify_bundle(root,spec.manifest_sha256,spec.checkpoint,spec.revision,spec.language,owner_uid=owner,execution=spec)
        if policy is not None:
            from service import installed_qwen_daemon as qd
            if policy is not ip.QWEN_SPLIT: raise Refused('explicit_installed_family_required')
            qd.manifest(manifest)
        return
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
    if spec.engine == 'pocket-tts':
        from worker.engines.pocket import verify_bundle
        verify_bundle(spec, local_bundles, owner_uid=owner)
    _, manifest = verified_bundle(root, spec.engine)
    if spec.engine == 'pocket-tts' and policy is not None:
        from service.pocket_install import manifest as installed_manifest
        installed_manifest(manifest)
    if (manifest.get('checkpoint'), manifest.get('revision'), manifest.get('language')) != (spec.checkpoint, spec.revision, spec.language):
        raise Refused('bundle_identity')


def kokoro_preset_version(context,principal,preset,voice_id):
    """Pure catalogue identity shared by the container and native CPU factories."""
    from contracts import validate as oracle
    from service.voice_store import _id
    _id(voice_id)
    spec=execution(context)
    if spec.engine!='kokoro' or spec.preset!=preset:raise Refused('unsupported_control')
    version=copy.deepcopy(context['plan']['voice_versions'][0])
    version['voice_id']=voice_id
    version['id']='preset-'+oracle.canonical_hash([principal,voice_id,version,spec.identity()])
    return version


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
        freeze(context['plan'])
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
        if principal not in self._contexts: raise Refused('not_ready')
        return kokoro_preset_version(json.loads(self._contexts[principal]),principal,preset,voice_id)

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
        freeze(template['plan'])
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
    if policy is not ip.configuration_policy(config):
        raise ValueError('installed_configuration_policy_binding')
    mapping = {key: value['path'] for key, value in config['bundles'].items()}
    if any(context['execution']['bundles'] != mapping for context in config['contexts'].values()):
        raise Refused('unregistered_mount_mapping')
    # Verify every hash and immutable tree; no discovery or provisioning.
    for context in config['contexts'].values(): verify_local(context, mapping, policy=policy)
    return mapping


class PocketContexts:
    """One trusted bundle/profile template per principal, reusable across voices.

    No public JSON path or descriptor input. The Core owner resolves uploads and
    immutable versions at command time; Production rechecks before admission.
    This does not activate an installed worker or infer qualification.
    """
    def __init__(self, contexts, local_bundles, local_references=None, *, installed=False):
        from worker.engines.pocket import verify_inputs, verify_bundle
        from worker.engines.pocket_execution import PocketExecution
        from worker.engines.pocket_prepare import PocketPreparation
        self._contexts = {}
        self._installed = installed
        self._bundles = copy.deepcopy(local_bundles)
        for principal, context in contexts.items():
            snapshot = copy.deepcopy(context)
            spec = execution(snapshot)
            if type(spec) not in (PocketExecution, PocketPreparation): raise Refused('pocket_execution_required')
            if set(local_bundles) != {spec.bundle_id}: raise Refused('owned_mapping_required')
            if local_references is None:
                verify_bundle(spec, local_bundles)
            else:
                if set(local_references) != {spec.reference_id}: raise Refused('owned_mapping_required')
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
        freeze(context['plan'])
        execution(context)
        return context


    def capabilities(self, principal):
        if principal not in self._contexts: return None
        context = json.loads(self._contexts[principal])
        return dict(engine='pocket-tts', profile=context['plan']['resolved']['profile'],
                    operations=['voice_prepare', 'voice_results', 'voice_save', 'speech_generate', 'job_get', 'job_cancel', 'artifact_get'],
                    installed=self._installed, next_action='asset_upload → voice_prepare with audition_text → voice_results → voice_save → speech_generate')

    def resolve_owned(self, core, principal, operation, request):
        """Called only by the Core owner, using managed metadata and owned versions.

        The configured template supplies bundle/profile/budgets, never the new
        reference identity. Supervisor independently verifies its read-only root.
        """
        from contracts import validate as oracle
        from worker.engines.pocket_execution import PocketExecution
        from worker.engines.pocket_prepare import PocketPreparation
        if principal not in self._contexts or operation not in ('voice_prepare', 'speech_generate'):
            raise Refused('not_ready')
        oracle.validate(operation+'_request', request)
        context = json.loads(self._contexts[principal])
        template = execution(context)
        plan = context['plan']; resolved = plan['resolved']
        descriptor = {k: v for k, v in template.identity().items()
                      if k not in ('phase', 'transform', 'in_tick', 'out_tick')}
        plan.update(operation=operation, request=copy.deepcopy(request))
        plan.pop('reference_roles', None)
        if operation == 'voice_prepare':
            if request['reference_text'] != '': raise Refused('unsupported_parameter')
            if not request.get('audition_text') or not request['audition_text'].strip():
                raise Refused('explicit_audition_required')
            reference_id = request['asset_id']
            plan['voice_versions'] = []
            settings = dict(kind='prepare', conditioning='pocket', output_format='wav_pcm16')
            resolved.update(kind='prepare', reuse_settings=dict(kind='pocket_clone', language=template.language, seed=None),
                            reuse_configuration=dict(mode='pocket_clone', language=template.language, reference_asset_id='pending'))
            plan['reference_roles'] = dict(version='reference-roles-v1', reference_kind='prepared_reference',
                reference_asset_id='pending', audition_asset_id='pending-audition', audition_text=request['audition_text'])
        else:
            version = core.voices.version(principal, request['voice_version_id'])
            if len(version['reference_asset_ids']) != 1: raise Refused('execution_reference_mismatch')
            reference_id = version['reference_asset_ids'][0]
            plan['voice_versions'] = [version]
            settings = dict(kind='pocket_clone', language=template.language, seed=None)
            resolved.update(kind='speech', reuse_settings=None, reuse_configuration=None)
        core.voices._asset(principal, reference_id)
        asset = core.asset(reference_id)
        path = core.root/'assets'/reference_id
        descriptor.update(reference_id=reference_id, reference_sha256=asset['sha256'],
                          reference_bytes=path.stat().st_size, reference_frames=asset['frames'], reference_rate=asset['sample_rate'])
        spec = (PocketPreparation(**descriptor, in_tick=request['in_tick'], out_tick=request['out_tick'])
                if operation == 'voice_prepare' else PocketExecution(**descriptor))
        plan['assets'] = [asset]
        resolved.update(settings=settings, settings_sha256=oracle.canonical_hash(settings))
        context['execution'].update(descriptor=spec.identity(), references={reference_id:'/references/assets/'+reference_id})
        freeze(plan)
        execution(context)
        # Full sealed bundle and reference byte/PCM verification, before admission.
        from worker.engines.pocket import verify_inputs
        verify_inputs(spec, self._bundles, {reference_id:path})
        return context


class QwenContexts:
    """Operator-owned exact principal/version/profile and sealed local mappings.

    Linux generation only. No installer activation or caller-selected descriptor.
    Hardware provenance expires at resolution/admission, never in history readers.
    """
    def __init__(self, contexts, local_bundles, local_references, inventory, profiles, *, installed_family=None):
        from worker.engines.qwen_execution import inventory_wire, current_hardware
        from contracts import validate as oracle
        from service import installed_qwen_daemon as qd
        if installed_family is not None and installed_family is not qd.FAMILY: raise Refused('explicit_installed_family_required')
        self.installed_family = installed_family
        self._contexts = {}
        for principal, context in contexts.items():
            snapshot = copy.deepcopy(context)
            spec = execution(snapshot)
            if spec is None or spec.engine != 'qwen': raise Refused('qwen_execution_required')
            if snapshot['execution']['hardware'] != inventory_wire(inventory): raise Refused('hardware_identity')
            self._hardware(spec, inventory)
            oracle.admission(dict(operation='speech_generate',input_sha256=oracle.plan_hash(snapshot['plan'])),snapshot,profiles)
            if installed_family is None:
                verify_qwen_local(snapshot, local_bundles, {k:local_references[k] for k in snapshot['execution']['references']}, inventory)
            else:
                from worker.engines.qwen import verify_bundle
                verify_bundle(local_bundles[spec.bundle_id],spec.manifest_sha256,spec.checkpoint,spec.revision,spec.language,owner_uid=10001,execution=spec)
            self._contexts[principal] = json.dumps(snapshot, allow_nan=False)
        self._inventory = copy.deepcopy(inventory)
        self._bundles = copy.deepcopy(local_bundles)
        self._references = copy.deepcopy(local_references)

    def _hardware(self, spec, inventory):
        from worker.engines.qwen_execution import current_hardware, installed_cpu_hardware
        if self.installed_family is None: return current_hardware(spec, inventory)
        return installed_cpu_hardware(spec, inventory, family=self.installed_family)

    def selection_metadata(self):
        """Configured snapshots only; no media/bundle verification or resolution.

        This describes possible executions, not readiness of an owned request.
        """
        from contracts import validate as oracle
        records = []
        for raw in self._contexts.values():
            context = json.loads(raw)
            spec = execution(context)  # Pure canonical decoding; no input verifier.
            records.append(dict(profile=context['plan']['resolved']['profile'],
                execution_sha256=oracle.canonical_hash(context['execution']),
                engine=spec.engine, device=spec.device,
                memory_bytes=spec.memory_bytes, threads=spec.threads,
                hardware=context['execution']['hardware'],
                native=context['execution'].get('native')))
        return tuple(records)

    def resolve(self, principal, operation, request):
        from contracts import validate as oracle
        from worker.engines.qwen_execution import current_hardware
        if operation != 'speech_generate' or principal not in self._contexts: raise Refused('not_ready')
        oracle.validate('speech_generate_request',request)
        context = json.loads(self._contexts[principal])
        if request['voice_version_id'] != context['plan']['voice_versions'][0]['id']: raise Refused('saved_version_required')
        context['plan']['request'] = copy.deepcopy(request)
        freeze(context['plan'])
        spec = execution(context)
        self._hardware(spec,self._inventory)
        verify_qwen_local(context,self._bundles,self._references,self._inventory)
        return context

    def resolve_owned(self, core, principal, operation, request):
        from worker.runtime_protocol import managed_qwen_cpu
        from worker.engines.qwen_execution import current_hardware
        if principal not in self._contexts: raise Refused('not_ready')
        template=json.loads(self._contexts[principal])
        if not managed_qwen_cpu(execution(template)):
            return self.resolve(principal, operation, request)
        if operation not in ('voice_prepare','speech_generate'): raise Refused('not_ready')
        version=None
        if operation == 'speech_generate':
            version=core.voices.version(principal,request['voice_version_id'])
            if len(version['reference_asset_ids']) != 1: raise Refused('execution_reference_mismatch')
            reference=version['reference_asset_ids'][0]
        else:
            reference=request['asset_id']
        core.voices._asset(principal,reference)
        asset=core.asset(reference)
        context=cpu_clone_context(template,principal,request,asset,version=version)
        spec=execution(context)
        self._hardware(spec,self._inventory)
        references=copy.deepcopy(context['execution']['references'])
        references[reference]['path']=core.root/'assets'/reference
        verify_qwen_local(context,self._bundles,references,self._inventory,owner_uid=10001 if self.installed_family is not None else None)
        return context

    def capabilities(self, principal):
        from worker.runtime_protocol import managed_qwen_cpu
        if principal not in self._contexts: return None
        context=json.loads(self._contexts[principal])
        clone=managed_qwen_cpu(execution(context))
        return dict(engine='qwen',profile=context['plan']['resolved']['profile'],installed=self.installed_family is not None,
            operations=(['voice_prepare','voice_results','voice_save'] if clone else [])+
                ['speech_generate','job_get','job_cancel','artifact_get'],
            next_action='asset_upload → voice_prepare → voice_results → voice_save → speech_generate')


def cpu_clone_context(template, principal, request, asset, *, version=None):
    """Owner/supervisor shared reconstruction; the template alone owns execution."""
    from contracts import validate as oracle
    from worker.runtime_protocol import managed_qwen_cpu, QWEN_CPU_SECONDS
    from worker.engines.qwen_prepare import QwenPreparation
    from worker.engines.qwen_execution import QwenExecution
    from worker.transport import canonical
    spec = execution(template)
    if not managed_qwen_cpu(spec) or spec.reference_text is not None:
        raise Refused('cpu_xvector_required')
    operation = 'voice_prepare' if version is None else 'speech_generate'
    oracle.validate(operation+'_request', request)
    context = copy.deepcopy(template); plan = context['plan']; resolved = plan['resolved']
    identity = resolved['identity']['effective']
    if request['identity'] != identity: raise Refused('identity_substitution')
    if (asset['channels'], asset['media_format'], asset['media_subtype']) != (1,'wav','PCM_16'):
        raise Refused('reference_format')
    descriptor = spec.identity()
    descriptor.pop('phase', None)
    descriptor.update(reference_id=asset['id'], reference_sha256=asset['sha256'],
        reference_frames=asset['frames'], reference_rate=asset['sample_rate'],
        reference_bytes=44+2*asset['frames'], timeout_s=QWEN_CPU_SECONDS)
    if descriptor['reference_bytes'] > 1048576 or asset['frames'] > asset['sample_rate']*60:
        raise Refused('preparation_reference_limit')
    settings = copy.deepcopy(resolved['settings'])
    cfg = dict(mode='qwen_clone',reference_asset_id=asset['id'],reference_text='',conditioning='x_vector')
    if version is None:
        if request['reference_text'] != '': raise Refused('unsupported_parameter')
        if not request.get('audition_text','').strip(): raise Refused('explicit_audition_required')
        if request['in_tick'] != 0 or request['out_tick']*asset['sample_rate'] != asset['frames']*48000:
            raise Refused('unsupported_segment')
        descriptor.update(voice_id='preparation',voice_revision='prepare-'+oracle.canonical_hash([principal,request]))
        plan.update(operation=operation,request=copy.deepcopy(request),voice_versions=[],assets=[copy.deepcopy(asset)])
        plan['reference_roles'] = dict(version='reference-roles-v1',reference_kind='prepared_reference',
            reference_asset_id='pending',audition_asset_id='pending-audition',audition_text=request['audition_text'])
        cfg['reference_asset_id']='pending'
        selected=dict(kind='prepare',conditioning='x_vector',output_format='wav_pcm16')
        resolved.update(kind='prepare',settings=selected,settings_sha256=oracle.canonical_hash(selected),
            reuse_settings=settings,reuse_configuration=cfg)
        derived=QwenPreparation(**descriptor)
    else:
        expected=dict(identity=identity,settings=settings,settings_sha256=oracle.canonical_hash(settings),
            origin='clone',provenance=asset['provenance'],reference_asset_ids=[asset['id']],
            reference_asset_sha256={asset['id']:asset['sha256']},reference_text='',configuration=cfg,design_lineage=None)
        if any(canonical(version.get(k)) != canonical(v) for k,v in expected.items()): raise Refused('saved_version_required')
        descriptor.update(voice_id=version['voice_id'],voice_revision=version['id'])
        derived=QwenExecution(**descriptor)
        plan.update(operation=operation,request=copy.deepcopy(request),voice_versions=[copy.deepcopy(version)],assets=[copy.deepcopy(asset)])
        plan.pop('reference_roles',None)
        resolved.update(kind='speech',reuse_settings=None,reuse_configuration=None)
    context['execution'].update(descriptor=derived.identity(),references={asset['id']:dict(
        path='/references/assets/'+asset['id'],voice_id=derived.voice_id,voice_revision=derived.voice_revision,
        checkpoint=derived.checkpoint,revision=derived.revision,text=None)})
    freeze(plan)
    execution(context)
    return context


def verify_qwen_local(context, bundles, references, inventory, *, owner_uid=None):
    """Same full verifier for control and independent Linux supervisor."""
    spec = execution(context)
    if set(bundles) != {spec.bundle_id} or set(references) != set(context['execution']['references']): raise Refused('owned_mapping_required')
    from worker.engines.qwen import verify_service_inputs
    verify_service_inputs(spec,bundles,references,inventory,owner_uid=owner_uid)


def installed_contexts(config, mapping):
    """One explicitly selected installed engine; reference-free Pocket factory."""
    from service import install_policy as ip
    config = ip.configuration(config)
    if ip.qwen(config):
        from service import installed_qwen_daemon as qd
        from worker.engines.qwen_execution import inventory_from_wire
        inventories=[c['execution']['hardware'] for c in config['contexts'].values()]
        if any(i != inventories[0] for i in inventories): raise Refused('hardware_identity')
        references={}
        for c in config['contexts'].values(): references.update(c['execution']['references'])
        return QwenContexts(config['contexts'], mapping, references, inventory_from_wire(inventories[0]), config['profiles'], installed_family=qd.FAMILY)
    if ip.pocket(config):
        return PocketContexts(config['contexts'], mapping, installed=True)
    return KokoroContexts(config['contexts'], mapping)


class NativeQwenContexts:
    """Trusted, bounded principal/profile/version metadata registrations.

    The control cannot inspect Darwin model paths. The enrolled native supervisor
    independently verifies its sealed bundle and exact allocation registration.
    No public request can add an entry or supply hardware/configuration metadata.
    """
    def __init__(self, contexts, inventory, profiles, native):
        from worker.native_protocol import native_binding
        from worker.engines.qwen_execution import static_native_hardware, inventory_wire
        from contracts import validate as oracle
        if type(contexts) is not dict or not 1 <= len(contexts) <= 64: raise Refused('native_registration')
        self._availability = None
        self._contexts = {}
        self._inventory = copy.deepcopy(inventory)
        self._native = copy.deepcopy(native)
        for key, context in contexts.items():
            if type(key) is not tuple or len(key) != 3: raise Refused('native_registration')
            principal, profile, version = key
            snapshot = copy.deepcopy(context)
            if native_binding(snapshot) != native: raise Refused('native_registration')
            spec = execution(snapshot)
            if (profile,version) != ('qwen-mps',spec.voice_revision): raise Refused('native_registration')
            if snapshot['execution']['hardware'] != inventory_wire(inventory): raise Refused('hardware_identity')
            static_native_hardware(spec, inventory)
            oracle.frozen_plan(snapshot['plan'],snapshot['bundles'])
            oracle.admission(dict(operation='speech_generate',input_sha256=oracle.plan_hash(snapshot['plan'])),snapshot,profiles)
            self._contexts[key] = json.dumps(snapshot,allow_nan=False)

    def bind_availability(self, runtime):
        if runtime.native != self._native: raise Refused('native_registration')
        self._availability = runtime.check_availability

    def _current(self, context):
        if self._availability is None: raise Refused('native_availability_required')
        self._availability(context)

    def authorize_origins(self, original, current, mapping):
        from worker.native_protocol import origin_template_pairs
        pairs = origin_template_pairs(original, current, mapping)
        from worker.transport import canonical
        configured = [dict(principal=key[0],profile=key[1],voice_version=key[2],context=json.loads(raw))
                      for key, raw in self._contexts.items()]
        if sorted(canonical(b) for _, b in pairs) != sorted(canonical(c) for c in configured):
            raise Refused('native_origin')
        self._origins = {canonical(b['context']): a['context'] for a, b in pairs}

    def _origin(self, context):
        from worker.transport import canonical
        return copy.deepcopy(getattr(self, '_origins', {}).get(canonical(context), context))

    def _clone_templates(self, principal):
        return [json.loads(raw) for key, raw in self._contexts.items()
                if key[0] == principal and execution(json.loads(raw)).task == 'clone'
                and not execution(json.loads(raw)).x_vector_only
                and any(c == dict(operation='voice_prepare',mode='prepare',
                                  identity=json.loads(raw)['plan']['resolved']['identity']['effective'])
                        for c in json.loads(raw)['plan']['resolved']['profile']['compatibility'])]

    def resolve_owned(self, core, principal, operation, request):
        from contracts import validate as oracle
        from worker.native_protocol import owned_clone_context
        from worker.engines.pocket import _read
        oracle.validate(operation+'_request', request)
        version = None
        if operation == 'speech_generate':
            version = core.voices.version(principal, request['voice_version_id'])
            if version['origin'] != 'clone' or (principal,'qwen-mps',version['id']) in self._contexts:
                return self.resolve_registered(principal,request,version)
            if len(version['reference_asset_ids']) != 1: raise Refused('execution_reference_mismatch')
            reference = version['reference_asset_ids'][0]
        elif operation == 'voice_prepare': reference = request['asset_id']
        else: raise Refused('not_ready')
        templates = self._clone_templates(principal)
        if len(templates) != 1: raise Refused('not_ready')
        core.voices._asset(principal,reference)
        asset = core.asset(reference)
        context = owned_clone_context(self._origin(templates[0]), principal, request, asset, version=version)
        from worker.native_protocol import origin_execution_context
        context = origin_execution_context(context, templates[0])
        spec = execution(context)
        self._current(context)
        raw = _read(core.root/'assets'/reference,spec.reference_bytes,spec.reference_sha256)
        from worker.engines.qwen_prepare import QwenPreparation, selected_reference
        if type(spec) is QwenPreparation: selected_reference(spec,raw)
        else:
            from worker.engines.pocket import _reference_pcm16
            if len(raw) != spec.reference_bytes: raise Refused('reference_size')
            _reference_pcm16(raw,spec.reference_rate,spec.reference_frames)
        return context

    def capabilities(self, principal):
        contexts = [json.loads(raw) for key, raw in self._contexts.items() if key[0] == principal]
        if not contexts: return None
        return dict(engine='qwen', profile=contexts[0]['plan']['resolved']['profile'],
                    operations=(['voice_preset_register'] if self.presets(principal) else []) + (['voice_prepare', 'voice_results', 'voice_save'] if len(self._clone_templates(principal)) == 1 else []) + ['speech_generate', 'job_get', 'job_cancel', 'artifact_get'],
                    installed=True, next_action=('asset_upload (complete canonical mono PCM16 WAV, at most 1 MiB/60 seconds) → voice_prepare (in_tick=0, out_tick=frames*48000/sample_rate, exact reference_text, separate audition_text) → voice_results → voice_save → speech_generate' if len(self._clone_templates(principal)) == 1 else 'Read configured_presets → voice_preset_register → speech_generate'))

    def presets(self, principal):
        result = []
        for key, raw in self._contexts.items():
            if key[0] != principal: continue
            spec = execution(json.loads(raw))
            if spec.task != 'custom_voice': continue
            # A selector must identify one exact trusted template, never first-match.
            if sum(execution(json.loads(v)).preset == spec.preset for k, v in self._contexts.items() if k[0] == principal) != 1:
                continue
            result.append(dict(preset=spec.preset, engine='qwen', model=spec.checkpoint,
                language=spec.language, bundle_sha256=spec.manifest_sha256,
                supports=['voice_preset_register', 'speech_generate'], clone_prepare=False))
        return result

    def _preset_context(self, principal, preset, voice_id):
        from worker.native_protocol import owned_preset_context
        matches = [json.loads(raw) for key, raw in self._contexts.items()
                   if key[0] == principal and execution(json.loads(raw)).task == 'custom_voice'
                   and execution(json.loads(raw)).preset == preset]
        if len(matches) != 1: raise Refused('unsupported_control')
        self._current(matches[0])
        from worker.native_protocol import origin_execution_context
        return origin_execution_context(owned_preset_context(self._origin(matches[0]), principal, preset, voice_id), matches[0])

    def preset_version(self, principal, preset, voice_id):
        return self._preset_context(principal, preset, voice_id)['plan']['voice_versions'][0]

    def resolve_registered(self, principal, request, version):
        from contracts import validate as oracle
        oracle.validate('speech_generate_request',request)
        key = (principal,'qwen-mps',request['voice_version_id'])
        if key in self._contexts:
            context = json.loads(self._contexts[key])
        else:
            context = self._preset_context(principal, version['configuration'].get('speaker'), version['voice_id'])
        from worker.transport import canonical
        if canonical(context['plan']['voice_versions']) != canonical([version]): raise Refused('saved_version_required')
        context['plan']['request'] = copy.deepcopy(request)
        freeze(context['plan'])
        self._current(context)
        return context

    def resolve(self, principal, operation, request):
        if operation != 'speech_generate': raise Refused('not_ready')
        key=(principal,'qwen-mps',request.get('voice_version_id'))
        if key not in self._contexts: raise Refused('not_ready')
        version=json.loads(self._contexts[key])['plan']['voice_versions'][0]
        return self.resolve_registered(principal,request,version)
