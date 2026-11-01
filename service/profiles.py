"""Pure H4 selection seam; trusted qualification records, not a public JSON contract.

No default profiles. production_selection binds owner-resolved executable contexts
to this policy before admission. Persisted records require operator-owned freshness/invalidation before use. Selection never provisions or updates readiness.
"""
from dataclasses import dataclass
from enum import Enum
import math
from worker.hardware import DeviceFamily, HardwareInventory, topology_allows


class Reason(str, Enum):
    SELECTED_MEASURED = 'selected_measured'
    SELECTED_UNMEASURED = 'selected_unmeasured_policy'
    SELECTED_OVERRIDE = 'selected_override'
    EXPLICIT_FALLBACK = 'explicit_identity_compatible_fallback'
    NO_PROFILES = 'no_profiles'
    OVERRIDE_UNAVAILABLE = 'override_unavailable'
    IDENTITY = 'identity_mismatch'
    FEATURES = 'features_incompatible'
    LANGUAGE = 'language_incompatible'
    EXECUTION = 'execution_incompatible'
    BACKEND = 'backend_missing'
    PROVISION = 'bundle_not_verified'
    QUALIFICATION = 'profile_unqualified'
    DEVICE = 'device_unavailable'
    TOPOLOGY = 'topology_incompatible'
    MEMORY = 'memory_insufficient'
    CONCURRENCY = 'concurrency_exceeded'
    MEASUREMENT = 'measurement_required'


@dataclass(frozen=True)
class Resources:
    memory_bytes: int
    threads: int
    concurrency: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.memory_bytes, self.threads, self.concurrency)):
            raise ValueError('finite positive resource budgets required')


@dataclass(frozen=True)
class Performance:
    """Comparable qualification workload key and observed seconds; no invented rank."""
    workload: str
    cold_seconds: float
    warm_seconds: float
    evidence: str

    def __post_init__(self):
        if not self.workload or not self.evidence or any(
                isinstance(v, bool) or not math.isfinite(v) or v <= 0
                for v in (self.cold_seconds, self.warm_seconds)):
            raise ValueError('invalid performance evidence')


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    engine: str
    checkpoint: str
    revision: str
    conditioning: str  # exact format/version or immutable conditioning identity
    family: DeviceFamily
    device_index: int
    dtype: str
    attention: str
    features: frozenset[str]
    languages: frozenset[str]
    resources: Resources
    installed: bool = False
    provisioned: bool = False
    verified: bool = False
    qualified_at: str | None = None
    qualification_evidence: str | None = None
    performance: Performance | None = None

    def __post_init__(self):
        if not all((self.profile_id, self.engine, self.checkpoint, self.revision,
                    self.conditioning, self.dtype, self.attention)):
            raise ValueError('exact profile identity required')
        if not isinstance(self.family, DeviceFamily) or type(self.device_index) is not int or self.device_index < 0:
            raise ValueError('invalid device')


@dataclass(frozen=True)
class VoiceWorkRequest:
    engine: str
    checkpoint: str
    revision: str
    conditioning: str
    features: frozenset[str]
    language: str
    memory_available_bytes: int
    concurrency: int = 1
    dtype: str | None = None
    attention: str | None = None
    profile_override: str | None = None
    device_override: tuple[DeviceFamily, int] | None = None

    def __post_init__(self):
        if not all((self.engine, self.checkpoint, self.revision, self.conditioning, self.language)):
            raise ValueError('exact request identity required')
        if type(self.memory_available_bytes) is not int or self.memory_available_bytes <= 0 or type(self.concurrency) is not int or self.concurrency <= 0:
            raise ValueError('invalid request budget')
        if self.device_override is not None:
            override = self.device_override
            if (not isinstance(override, tuple) or len(override) != 2 or
                    not isinstance(override[0], DeviceFamily) or
                    type(override[1]) is not int or override[1] < 0):
                raise ValueError('invalid device override')


@dataclass(frozen=True)
class SelectionPolicy:
    objective: str = 'warm'
    workload: str = ''
    allow_unmeasured: bool = False
    preferred_device: DeviceFamily | None = None
    fallback_devices: tuple[DeviceFamily, ...] = ()

    def __post_init__(self):
        if self.objective not in ('cold', 'warm'):
            raise ValueError('invalid objective')
        if self.fallback_devices and self.preferred_device is None:
            raise ValueError('fallback requires preferred device')


@dataclass(frozen=True)
class ProfileStatus:
    profile_id: str
    reasons: tuple[Reason, ...]

    @property
    def ready(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class ResolvedSelection:
    status: str
    profile: ExecutionProfile | None
    reason: Reason
    diagnostics: tuple[ProfileStatus, ...]
    # profile freezes engine/checkpoint/revision/device index/dtype/attention/resources
    # within the selected complete production context; no public descriptor authority.


def profile_status(p: ExecutionProfile, r: VoiceWorkRequest,
                   inventory: HardwareInventory) -> ProfileStatus:
    reasons = []
    if (p.engine, p.checkpoint, p.revision, p.conditioning) != (r.engine, r.checkpoint, r.revision, r.conditioning):
        reasons.append(Reason.IDENTITY)
    if not r.features <= p.features:
        reasons.append(Reason.FEATURES)
    if r.language not in p.languages:
        reasons.append(Reason.LANGUAGE)
    if (r.dtype and r.dtype != p.dtype) or (r.attention and r.attention != p.attention):
        reasons.append(Reason.EXECUTION)
    if not p.installed:
        reasons.append(Reason.BACKEND)
    if not (p.provisioned and p.verified):
        reasons.append(Reason.PROVISION)
    if not (p.qualified_at and p.qualification_evidence):
        reasons.append(Reason.QUALIFICATION)
    if not topology_allows(inventory.host, p.family):
        reasons.append(Reason.TOPOLOGY)
    if not any(d.family == p.family and d.index == p.device_index and
               d.supported and d.available and d.probed_at for d in inventory.devices):
        reasons.append(Reason.DEVICE)
    if p.resources.memory_bytes > r.memory_available_bytes:
        reasons.append(Reason.MEMORY)
    if r.concurrency > p.resources.concurrency:
        reasons.append(Reason.CONCURRENCY)
    return ProfileStatus(p.profile_id, tuple(reasons))


def select(profiles: tuple[ExecutionProfile, ...], request: VoiceWorkRequest,
           inventory: HardwareInventory, policy: SelectionPolicy = SelectionPolicy()) -> ResolvedSelection:
    if len({p.profile_id for p in profiles}) != len(profiles):
        raise ValueError('duplicate profile IDs')
    if len(profiles) > 256:
        raise ValueError('profile limit exceeded')
    profiles = tuple(sorted(profiles, key=lambda p: p.profile_id))
    diagnostics = tuple(profile_status(p, request, inventory) for p in profiles)
    candidates = [p for p, s in zip(profiles, diagnostics) if s.ready]
    override = request.profile_override is not None or request.device_override is not None
    if request.profile_override is not None:
        candidates = [p for p in candidates if p.profile_id == request.profile_override]
    if request.device_override is not None:
        candidates = [p for p in candidates if (p.family, p.device_index) == request.device_override]
    fallback = False
    if not override and policy.preferred_device is not None:
        preferred = [p for p in candidates if p.family == policy.preferred_device]
        if preferred:
            candidates = preferred
        else:
            candidates = [p for p in candidates if p.family in policy.fallback_devices]
            fallback = bool(candidates)
    measured = [p for p in candidates if p.performance is not None and
                policy.workload and p.performance.workload == policy.workload]
    if measured:
        candidates = sorted(measured, key=lambda p: (
            getattr(p.performance, policy.objective + '_seconds'), p.profile_id))
        reason = Reason.SELECTED_MEASURED
    elif candidates and policy.allow_unmeasured:
        reason = Reason.SELECTED_UNMEASURED
    else:
        reason = Reason.OVERRIDE_UNAVAILABLE if override else (
            Reason.MEASUREMENT if candidates else Reason.NO_PROFILES)
        return ResolvedSelection('not_ready', None, reason, diagnostics)
    return ResolvedSelection('ready', candidates[0],
                             Reason.SELECTED_OVERRIDE if override else (
                                 Reason.EXPLICIT_FALLBACK if fallback else reason), diagnostics)
