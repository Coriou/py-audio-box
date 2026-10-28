"""Owner-only finite context composition for ONE existing runtime.

Candidates are trusted resolver objects, never public descriptors. Linux and native
registries cannot be combined. Selection returns the exact context consumed by
Production/Admission; no scheduler, runtime multiplexing, or provisioning lives here.
"""
import copy
from dataclasses import dataclass
from datetime import datetime, timezone

from contracts import validate as oracle
from service.profiles import (ExecutionProfile, Resources, VoiceWorkRequest,
                              SelectionPolicy, Performance, select)
from worker.hardware import DeviceFamily, Device, HardwareInventory, Host
from worker.runtime_protocol import execution


class SelectionUnavailable(ValueError):
    def __init__(self, reasons):
        super().__init__('not_ready')
        self.reasons = tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class Measurement:
    """Operator receipt bound to the complete executable context, not a device name."""
    execution_sha256: str
    performance: Performance

    def __post_init__(self):
        if (not isinstance(self.performance, Performance) or len(self.execution_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.execution_sha256)):
            raise ValueError('bound_measurement_required')


class ContextComposition:
    """At most 16 resolvers sharing the very same authenticated runtime/mounts.

    Native supports only its enrolled qwen-mps registry. Installed Linux keeps its
    existing single-engine policy. An explicitly configured Linux Qwen supervisor
    may select among registered device indices; this does not install Qwen.
    """
    def __init__(self, candidates, *, runtime, inventory, measurements=None,
                 policy=SelectionPolicy(), max_age_seconds=300,
                 concurrency=1):
        if (not isinstance(inventory, HardwareInventory) or not 1 <= len(candidates) <= 16 or len(dict(candidates)) != len(candidates)
                or type(max_age_seconds) is not int or not 0 < max_age_seconds <= 3600
                or type(concurrency) is not int or concurrency < 1
                or any(type(name) is not str or not 1 <= len(name) <= 128 for name, _ in candidates)):
            raise ValueError('finite_context_composition_required')
        self.candidates = tuple(candidates)
        self.runtime = runtime
        self.inventory = copy.deepcopy(inventory)
        self.measurements = copy.deepcopy(measurements or {})
        self.policy = policy
        self.max_age_seconds = max_age_seconds
        self.concurrency = concurrency

    def bind_availability(self, runtime):
        if runtime is not self.runtime: raise SelectionUnavailable(['runtime_binding_changed'])
        for _, provider in self.candidates:
            bind = getattr(provider, 'bind_availability', None)
            if bind is not None: bind(runtime)

    def capabilities(self, principal):
        provider = self.candidates[0][1] if len(self.candidates) == 1 else None
        describe = getattr(provider, 'capabilities', None)
        return describe(principal) if describe else None

    def presets(self, principal):
        provider = self.candidates[0][1] if len(self.candidates) == 1 else None
        describe = getattr(provider, 'presets', None)
        return describe(principal) if describe else []

    def preset_version(self, principal, preset, voice_id):
        provider = self.candidates[0][1] if len(self.candidates) == 1 else None
        resolve = getattr(provider, 'preset_version', None)
        if resolve is None: raise SelectionUnavailable(['preset_composition_unsupported'])
        return resolve(principal, preset, voice_id)

    def resolve_selected(self, dispatcher, principal, operation, request):
        if dispatcher.runtime is not self.runtime:
            raise SelectionUnavailable(['runtime_binding_changed'])
        if dispatcher.installed_checked and len(self.candidates) != 1:
            raise SelectionUnavailable(['installed_composition_unsupported'])
        resolved, refused = [], []
        for name, provider in self.candidates:
            try:
                context = resolve_owned(provider, dispatcher.core, principal, operation, request)
                resolved.append((name, context))
            except ValueError:
                refused.append('candidate_identity_or_readiness_refused')
        if not resolved: raise SelectionUnavailable(refused or ['no_profiles'])
        return choose(dispatcher, resolved, request, inventory=self.inventory,
                      measurements=self.measurements, policy=self.policy,
                      max_age_seconds=self.max_age_seconds, concurrency=self.concurrency)


def resolve_owned(provider, core, principal, operation, request):
    if hasattr(provider, 'resolve_owned'):
        return provider.resolve_owned(core, principal, operation, request)
    if operation == 'speech_generate' and hasattr(provider, 'resolve_registered'):
        version = core.voices.version(principal, request['voice_version_id'])
        return provider.resolve_registered(principal, request, version)
    return provider.resolve(principal, operation, request)


def fresh_inventory(inventory, max_age_seconds):
    now = datetime.now(timezone.utc)
    devices = []
    for d in inventory.devices:
        try:
            stamp = datetime.fromisoformat(d.probed_at)
            age = (now - stamp).total_seconds()
            fresh = 0 <= age <= max_age_seconds
        except (TypeError, ValueError, OverflowError):
            fresh = False
        devices.append(Device(d.family, d.index, d.supported,
                              d.available and fresh, d.probed_at if fresh else None))
    return HardwareInventory(inventory.host, tuple(devices))


def choose(dispatcher, contexts, request, *, inventory=None, measurements=None,
           policy=SelectionPolicy(allow_unmeasured=True), max_age_seconds=300, concurrency=1):
    """Resolve before admission, return ONE complete immutable execution snapshot."""
    records, snapshots = [], {}
    blocked = []
    for name, context in contexts:
        context = copy.deepcopy(context)
        plan = context['plan']; resolved = plan['resolved']; snapshot = resolved['profile']
        matches = [p for p in dispatcher.profiles if p['id'] == snapshot['id']]
        if len(matches) != 1 or oracle.profile_snapshot(matches[0]) != snapshot:
            blocked.append('profile_configuration_changed'); continue
        p = matches[0]
        if request.get('profile_override') not in (None, p['id']):
            blocked.append('override_unavailable'); continue
        try:
            spec = execution(context)
        except ValueError as exc:
            blocked.append(str(exc) if str(exc) in ('profile_override_unavailable', 'device_override_unavailable') else 'execution_context_refused')
            continue
        # Synthetic contexts remain a separate, explicit fixture path.
        if spec is None and p['id'] != 'cpu-synthetic':
            blocked.append('execution_required'); continue
        family = DeviceFamily.MPS if spec and spec.device == 'mps' else (
            DeviceFamily.CUDA if spec and spec.device.startswith('cuda:') else DeviceFamily.CPU)
        index = int(spec.device.split(':')[1]) if family == DeviceFamily.CUDA else 0
        native = hasattr(dispatcher.runtime, 'native')
        from worker.engines.native_kokoro import NativeKokoroExecution
        native_cpu=type(spec) is NativeKokoroExecution
        if native_cpu:
            from service.native_cpu_runtime import NativeCPURuntime
            if type(dispatcher.runtime) is not NativeCPURuntime or context['execution'].get('native')!=dispatcher.runtime.native:
                blocked.append('native_binding_changed');continue
        if spec and family == DeviceFamily.MPS and context['execution'].get('native') != getattr(dispatcher.runtime, 'native', None):
            blocked.append('native_binding_changed'); continue
        if spec and ((family == DeviceFamily.MPS or native_cpu) != native):
            blocked.append('runtime_topology_incompatible'); continue
        if inventory is None:
            if spec and (spec.engine == 'qwen' or native_cpu):
                from worker.engines.qwen_execution import inventory_from_wire
                inv = inventory_from_wire(context['execution']['hardware'])
            else:
                # Live CPU policy metadata for the existing networkless Linux worker;
                # no claim about the physical host architecture or measured performance.
                inv = HardwareInventory(Host('linux', 'unspecified', 'docker'), (
                    Device(DeviceFamily.CPU, 0, True, True, datetime.now(timezone.utc).isoformat()),))
        else: inv = inventory
        if native and (inv.host.system != 'darwin' or inv.host.deployment != 'native'):
            blocked.append('runtime_topology_incompatible'); continue
        if not native and inv.host.system != 'linux':
            blocked.append('runtime_topology_incompatible'); continue
        ident = resolved['identity']['effective']
        if request['identity'] != ident:
            blocked.append('identity_mismatch'); continue
        if native_cpu:
            from worker.native_cpu_qualification import permitted
            if permitted(context,p):
                if len(contexts)!=1:raise SelectionUnavailable(['qualification_single_candidate_required'])
                return context  # Explicit bounded test authority; never measured/ready.
        conditioning = oracle.canonical_hash(plan['voice_versions'] if plan['voice_versions'] else
                                             resolved.get('reuse_configuration'))
        measurement = (measurements or {}).get(name)
        performance = None
        if measurement is not None:
            if measurement.execution_sha256 != oracle.canonical_hash(context.get('execution')):
                blocked.append('measurement_identity_changed'); continue
            performance = measurement.performance
        qualified = (p['support'] == 'qualified' and p['verification'] == 'measured'
                     and bool(p['qualification_receipt']) and p['configuration'] == 'valid'
                     and p['readiness'] == 'ready')
        if spec and 'synthetic' in p['qualification_receipt'].lower(): qualified = False
        resources = Resources(spec.memory_bytes if spec else 192 * 1024**2,
                              spec.threads if spec else p['limits']['threads'], p['limits']['expensive_slots'])
        record = ExecutionProfile(name, ident['engine'], ident['model'],
            spec.revision if spec else 'synthetic', conditioning, family, index,
            ident['precision'], spec.attention if spec and hasattr(spec, 'attention') else 'eager',
            frozenset([plan['operation']]), frozenset([resolved['settings'].get('language', 'und')]),
            resources, p['installation'] == 'installed', True, True,
            p['checked_at'] if qualified else None, p['qualification_receipt'] if qualified else None, performance)
        if family == DeviceFamily.CUDA and p['limits']['vram_bytes'] > dispatcher.admission.budget['vram_bytes']:
            blocked.append('memory_insufficient'); continue
        if spec and inventory is not None and spec.engine == 'qwen':
            # Do not select with inventory different from the independently checked descriptor.
            from worker.engines.qwen_execution import inventory_wire
            if context['execution']['hardware'] != inventory_wire(inventory):
                blocked.append('hardware_identity_changed'); continue
        if native:
            try: inv = dispatcher.runtime.check_availability(context)
            except ValueError:
                blocked.append('stale_or_unavailable_device'); continue
        records.append(record); snapshots[name] = context
    if not records: raise SelectionUnavailable(blocked or ['no_profiles'])
    first = records[0]
    override = request.get('device_override')
    device = (DeviceFamily(override['family']), override['index']) if override else None
    work = VoiceWorkRequest(first.engine, first.checkpoint, first.revision, first.conditioning,
        first.features, next(iter(first.languages)), dispatcher.admission.budget['ram_bytes'],
        concurrency=concurrency, device_override=device)
    result = select(tuple(records), work, inv if native else fresh_inventory(inv, max_age_seconds), policy)
    if result.profile is None:
        raise SelectionUnavailable([result.reason.value, *blocked,
            *(r.value for status in result.diagnostics for r in status.reasons)])
    return snapshots[result.profile.profile_id]


def metadata_status(dispatcher, composition, name, p, facts):
    """All witnesses belong to ONE configured template; never resolve owned inputs."""
    from worker.engines.qwen_execution import inventory_wire
    reasons = []
    family = (DeviceFamily.MPS if facts['device'] == 'mps' else
              DeviceFamily.CUDA if facts['device'].startswith('cuda:') else DeviceFamily.CPU)
    index = int(facts['device'].split(':')[1]) if family == DeviceFamily.CUDA else 0
    native = hasattr(dispatcher.runtime, 'native')
    if family == DeviceFamily.MPS and facts['native'] != getattr(dispatcher.runtime, 'native', None):
        reasons.append('native_binding_changed')
    inv = composition.inventory
    if ((family == DeviceFamily.MPS) != native or
            (native and (inv.host.system != 'darwin' or inv.host.deployment != 'native')) or
            (not native and inv.host.system != 'linux')):
        reasons.append('runtime_topology_incompatible')
        return reasons
    if facts['engine'] != 'qwen':
        return ['measurement_binding_verification_required']
    if facts['hardware'] != inventory_wire(inv):
        reasons.append('hardware_identity_changed')
    measurement = composition.measurements.get(name)
    performance = None
    if measurement is not None:
        if measurement.execution_sha256 != facts['execution_sha256']:
            reasons.append('measurement_required')
        else:
            performance = measurement.performance
    if 'synthetic' in p['qualification_receipt'].lower():
        reasons.append('profile_unqualified')
    if (facts['memory_bytes'] != p['limits']['ram_bytes'] or
            facts['threads'] != p['limits']['threads']):
        reasons.append('execution_context_refused')
    if family == DeviceFamily.CUDA and p['limits']['vram_bytes'] > dispatcher.admission.budget['vram_bytes']:
        reasons.append('memory_insufficient')
    # Reuse execution's pure device/topology/resource/concurrency and policy gates.
    # Identity placeholders deliberately express no request-specific voice promise.
    record = ExecutionProfile(name, 'metadata', 'metadata', 'metadata', 'metadata',
        family, index, 'metadata', 'metadata', frozenset(), frozenset({'und'}),
        Resources(facts['memory_bytes'], facts['threads'], p['limits']['expensive_slots']),
        p['installation'] == 'installed', True, True, p['checked_at'],
        p['qualification_receipt'], performance)
    work = VoiceWorkRequest('metadata', 'metadata', 'metadata', 'metadata',
        frozenset(), 'und', dispatcher.admission.budget['ram_bytes'],
        concurrency=composition.concurrency)
    observed = dispatcher.runtime.cached_availability() if native else fresh_inventory(inv, min(composition.max_age_seconds, 300))
    if observed is None: return reasons + ['stale_or_unavailable_device']
    result = select((record,), work, observed, composition.policy)
    if result.profile is None:
        reasons.extend(r.value for status in result.diagnostics for r in status.reasons)
        reasons.append(result.reason.value)
    return reasons


def discovery(dispatcher):
    """Metadata only: never resolve media, verify bundles, import ML or provision."""
    composition = dispatcher.production_contexts
    if not dispatcher.installed_checked and not isinstance(composition, ContextComposition):
        return []
    inventory = (composition.inventory if isinstance(composition, ContextComposition) else
                 getattr(composition, '_inventory', None))
    age = composition.max_age_seconds if isinstance(composition, ContextComposition) else 300
    installed_cpu=False
    if getattr(composition,'installed_family',None) is not None:
        from service import installed_qwen_daemon as qd
        installed_cpu=composition.installed_family is qd.FAMILY
    if installed_cpu:
        try:
            for raw in composition._contexts.values():
                import json
                dispatcher.runtime.check_availability(json.loads(raw))
        except (ValueError, RuntimeError, OSError):
            inventory = HardwareInventory(inventory.host, ())
    elif hasattr(dispatcher.runtime, 'cached_availability'):
        inventory = dispatcher.runtime.cached_availability()
        if inventory is None and getattr(composition, '_inventory', None) is not None:
            inventory = HardwareInventory(composition._inventory.host, ())
    else:
        inventory = fresh_inventory(inventory, age) if inventory is not None else None
    profiles = copy.deepcopy(dispatcher.profiles)
    for p in profiles:
        reasons = []
        for key, expected, reason in (('support', 'qualified', 'profile_unqualified'),
                ('installation', 'installed', 'backend_missing'),
                ('verification', 'measured', 'measurement_required'),
                ('configuration', 'valid', 'configuration_invalid'),
                ('readiness', 'ready', 'profile_not_ready')):
            if p[key] != expected: reasons.append(reason)
        if not p['qualification_receipt']: reasons.append('profile_unqualified')
        if (not dispatcher.reconciled or dispatcher.closing or dispatcher.quarantined
                or dispatcher.orphaned or dispatcher.resource_fault or dispatcher.persistence_fault
                or dispatcher.integrity_fault or not dispatcher._activation_open()
                or not getattr(dispatcher.runtime, 'admission_open', True)):
            reasons.append('runtime_not_ready')
        if isinstance(composition, ContextComposition):
            if composition.runtime is not dispatcher.runtime: reasons.append('runtime_binding_changed')
            if composition.concurrency > p['limits']['expensive_slots']: reasons.append('concurrency_exceeded')
            if dispatcher.installed_checked and len(composition.candidates) != 1:
                reasons.append('installed_composition_unsupported')
            candidate_ready = False
            unknown = False
            failures = []
            for name, provider in composition.candidates:
                describe = getattr(provider, 'selection_metadata', None)
                if describe is None:
                    unknown = True
                    continue
                try:
                    metadata = describe()
                    for facts in metadata:
                        if facts['profile'] != oracle.profile_snapshot(p):
                            continue
                        candidate_reasons = metadata_status(dispatcher, composition, name, p, facts)
                        if not candidate_reasons:
                            candidate_ready = True
                        failures.extend(candidate_reasons)
                except (ValueError, TypeError, KeyError, AttributeError):
                    unknown = True
            if not candidate_ready:
                reasons.extend(failures or ['measurement_required'])
                if unknown:
                    reasons.append('measurement_binding_verification_required')
        family = DeviceFamily.MPS if p['device'] == 'metal' else DeviceFamily(p['device'])
        if inventory is not None and not any(d.family == family and d.available and d.supported
                                             for d in inventory.devices):
            reasons.append('stale_or_unavailable_device')
        if p['limits']['ram_bytes'] > dispatcher.admission.budget['ram_bytes'] or p['limits']['vram_bytes'] > dispatcher.admission.budget['vram_bytes']:
            reasons.append('memory_insufficient')
        if reasons:
            p['readiness'] = 'not_ready'
            p['next_action'] = 'Operator: refresh/qualify exact installed profile: ' + ', '.join(sorted(set(reasons)))
    return profiles
