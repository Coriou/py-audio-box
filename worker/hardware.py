"""Cheap host metadata and explicit, injected availability probes. No ML imports."""
from dataclasses import dataclass
from enum import Enum
import platform
from typing import Callable


class DeviceFamily(str, Enum):
    CPU = 'cpu'
    CUDA = 'cuda'
    MPS = 'mps'


@dataclass(frozen=True)
class Host:
    system: str
    architecture: str
    deployment: str  # operator-declared: native, docker, or unknown

    def __post_init__(self):
        if self.deployment not in ('native', 'docker', 'unknown'):
            raise ValueError('invalid deployment')


def host_metadata(deployment: str = 'unknown') -> Host:
    """OS is the current process OS, never an inferred Docker host OS."""
    return Host(platform.system().lower(), platform.machine().lower(), deployment)


@dataclass(frozen=True)
class Device:
    family: DeviceFamily
    index: int
    supported: bool
    available: bool
    probed_at: str | None

    def __post_init__(self):
        if not isinstance(self.family, DeviceFamily) or type(self.index) is not int or self.index < 0:
            raise ValueError('invalid device')


@dataclass(frozen=True)
class HardwareInventory:
    host: Host
    devices: tuple[Device, ...] = ()

    def __post_init__(self):
        keys = [(d.family, d.index) for d in self.devices]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate device')


def topology_allows(host: Host, family: DeviceFamily) -> bool:
    return family != DeviceFamily.MPS or (
        host.system == 'darwin' and host.architecture in ('arm64', 'aarch64')
        and host.deployment == 'native')


def discover(host: Host, persisted_devices: tuple[Device, ...] = ()) -> HardwareInventory:
    """Trusted persisted probe records only; no probing, filesystem scanning or loading."""
    return HardwareInventory(host, tuple(persisted_devices))


@dataclass(frozen=True)
class ProbeResult:
    available: bool
    reason: str


def probe(inventory: HardwareInventory, family: DeviceFamily, index: int,
          provider: Callable[[DeviceFamily, int], bool] | None = None) -> ProbeResult:
    """Operator-only hook. Provider owns any lazy runtime import; never called by discovery."""
    if not topology_allows(inventory.host, family):
        return ProbeResult(False, 'topology_incompatible')
    if provider is None:
        return ProbeResult(False, 'probe_unavailable')
    try:
        result = provider(family, index)
        if type(result) is not bool:
            return ProbeResult(False, 'probe_invalid')
        return ProbeResult(result, 'probe_available' if result else 'probe_unavailable')
    except Exception:
        return ProbeResult(False, 'probe_failed')
