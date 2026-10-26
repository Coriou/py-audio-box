"""Closed trusted Qwen binding; no public path or serialized prompt authority."""
from dataclasses import dataclass, fields
import re
from worker.transport import Refused


@dataclass(frozen=True)
class QwenExecution:
    bundle_id: str
    manifest_sha256: str
    checkpoint: str
    revision: str
    task: str
    language: str
    voice_id: str
    voice_revision: str
    threads: int
    max_text_bytes: int
    max_frames: int
    max_output_bytes: int
    memory_bytes: int
    sample_rate: int
    device: str = 'cpu'
    dtype: str = 'float32'
    attention: str = 'sdpa'
    preset: str | None = None
    instruct: str | None = None
    reference_id: str | None = None
    reference_sha256: str | None = None
    reference_bytes: int = 0
    reference_frames: int = 0
    reference_rate: int = 0
    reference_text: str | None = None
    x_vector_only: bool = False
    seed: int | None = None
    timeout_s: int = 300
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    engine: str = 'qwen'

    def __post_init__(self):
        for key in ('bundle_id', 'voice_id', 'voice_revision', 'checkpoint', 'revision', 'language'):
            if type(getattr(self, key)) is not str or not 0 < len(getattr(self, key)) <= 256:
                raise Refused('execution_identity')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', self.bundle_id): raise Refused('bundle_id')
        if type(self.manifest_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', self.manifest_sha256): raise Refused('manifest_digest')
        if self.engine != 'qwen' or self.task not in ('clone', 'custom_voice', 'voice_design'): raise Refused('execution_task')
        if type(self.device) is not str or not re.fullmatch(r'cpu|mps|cuda:(0|[1-9][0-9]{0,2})', self.device): raise Refused('execution_device')
        if self.dtype not in ('float32', 'float16', 'bfloat16') or self.attention not in ('eager', 'sdpa', 'flash_attention_2'): raise Refused('execution_precision')
        if (self.device == 'cpu' and self.dtype != 'float32') or (self.device == 'mps' and self.dtype == 'bfloat16') or (not self.device.startswith('cuda:') and self.attention == 'flash_attention_2'): raise Refused('execution_precision')
        for key, ceiling in (('threads',64), ('max_text_bytes',65536), ('max_frames',192000*300), ('max_output_bytes',115200044), ('memory_bytes',64*1024**3), ('sample_rate',192000), ('timeout_s',3600)):
            if type(getattr(self,key)) is not int or not 0 < getattr(self,key) <= ceiling: raise Refused('execution_limit')
        if self.max_frames > self.sample_rate*300 or self.max_output_bytes < 46 or self.memory_bytes < self.max_output_bytes: raise Refused('execution_limit')
        import math
        for key, low, high in (('temperature',0,2), ('top_p',0,1), ('repetition_penalty',0,10)):
            v = getattr(self,key)
            if type(v) not in (int,float) or not math.isfinite(v) or not low <= v <= high or (key == 'repetition_penalty' and v == 0): raise Refused('execution_sampling')
        for key, low, high in (('top_k',0,1000), ('max_new_tokens',1,32768)):
            if type(getattr(self,key)) is not int or not low <= getattr(self,key) <= high: raise Refused('execution_sampling')
        # Seed support is not established by the local capability contract.
        if self.seed is not None: raise Refused('unsupported_seed')
        for value in (self.preset, self.instruct, self.reference_text):
            if value is not None and (type(value) is not str or not value.strip() or len(value.encode()) > 65536): raise Refused('execution_control')
        if type(self.x_vector_only) is not bool: raise Refused('execution_control')
        if self.task == 'clone':
            if self.device == 'cpu' and self.x_vector_only and self.reference_text is not None:
                raise Refused('reference_conditioning')
            if self.preset is not None or self.instruct is not None: raise Refused('unsupported_control')
            if type(self.reference_id) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}', self.reference_id): raise Refused('reference_identity')
            if type(self.reference_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', self.reference_sha256): raise Refused('reference_identity')
            for value, cap in ((self.reference_bytes, 24000000), (self.reference_frames,11520000), (self.reference_rate,192000)):
                if type(value) is not int or not 0 < value <= cap: raise Refused('reference_limit')
            if self.reference_frames > self.reference_rate*60 or (not self.x_vector_only and self.reference_text is None): raise Refused('reference_conditioning')
        elif any((self.reference_id is not None, self.reference_sha256 is not None, self.reference_bytes != 0, self.reference_frames != 0, self.reference_rate != 0, self.reference_text is not None, self.x_vector_only)):
            raise Refused('unsupported_reference')
        if self.task == 'custom_voice' and self.preset is None: raise Refused('preset_required')
        if self.task == 'voice_design' and (self.preset is not None or self.instruct is None): raise Refused('design_required')

    @classmethod
    def from_wire(cls, value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}: raise Refused('closed_execution_required')
        return cls(**value)

    def identity(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def inventory_wire(inventory):
    from worker.hardware import HardwareInventory
    if type(inventory) is not HardwareInventory: raise Refused('hardware_required')
    return dict(host=dict(system=inventory.host.system, architecture=inventory.host.architecture,
                         deployment=inventory.host.deployment),
                devices=[dict(family=d.family.value,index=d.index,supported=d.supported,
                              available=d.available,probed_at=d.probed_at) for d in inventory.devices])


def inventory_from_wire(value):
    from worker.hardware import HardwareInventory, Host, Device, DeviceFamily
    from datetime import datetime
    try:
        if type(value) is not dict or set(value) != {'host','devices'}: raise ValueError()
        if set(value['host']) != {'system','architecture','deployment'} or type(value['devices']) is not list or len(value['devices']) > 64: raise ValueError()
        if any(type(v) is not str or not 0 < len(v) <= 128 for v in value['host'].values()): raise ValueError()
        devices=[]
        for d in value['devices']:
            if set(d) != {'family','index','supported','available','probed_at'} or type(d['supported']) is not bool or type(d['available']) is not bool: raise ValueError()
            stamp=d['probed_at']
            if stamp is not None and (type(stamp) is not str or len(stamp) > 128 or datetime.fromisoformat(stamp).tzinfo is None): raise ValueError()
            devices.append(Device(DeviceFamily(d['family']),d['index'],d['supported'],d['available'],d['probed_at']))
        return HardwareInventory(Host(**value['host']),tuple(devices))
    except (ValueError,TypeError,KeyError): raise Refused('closed_hardware_required') from None


def current_hardware(spec, inventory):
    """Service Linux lane only. Historical readers deliberately do not expire."""
    from datetime import datetime, timezone
    from worker.hardware import DeviceFamily
    inventory_wire(inventory)
    if inventory.host.system != 'linux' or spec.device == 'mps': raise Refused('native_transport_required')
    family=DeviceFamily.CUDA if spec.device.startswith('cuda:') else DeviceFamily.CPU
    index=int(spec.device.split(':')[1]) if family == DeviceFamily.CUDA else 0
    try:
        d=next(d for d in inventory.devices if d.family == family and d.index == index)
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(d.probed_at)).total_seconds()
        if not d.supported or not d.available or not 0 <= age <= 300: raise ValueError()
    except (StopIteration,ValueError,TypeError): raise Refused('stale_or_unavailable_device') from None


def static_native_hardware(spec, inventory):
    """Validate immutable native identity, independently of observation age."""
    from worker.hardware import DeviceFamily, topology_allows
    inventory_from_wire(inventory_wire(inventory))
    from worker.engines.qwen_prepare import QwenPreparation
    if type(spec) is QwenPreparation: spec = spec.generation()
    if (type(spec) is not QwenExecution
            or (spec.device, spec.dtype, spec.attention) != ('mps', 'float32', 'sdpa')
            or spec.task not in ('custom_voice', 'clone')
            or not topology_allows(inventory.host, DeviceFamily.MPS)):
        raise Refused('native_execution_required')
    try:
        device = next(d for d in inventory.devices if d.family == DeviceFamily.MPS and d.index == 0)
        if device.supported is not True: raise ValueError()
    except (StopIteration, ValueError):
        raise Refused('stale_or_unavailable_device') from None


def current_native_hardware(spec, inventory, *, now=None):
    """Current observation only; immutable enrollment uses static validation."""
    from datetime import datetime, timezone
    from worker.hardware import DeviceFamily
    static_native_hardware(spec, inventory)
    instant = datetime.now(timezone.utc) if now is None else now
    try:
        device = next(d for d in inventory.devices if d.family == DeviceFamily.MPS and d.index == 0)
        age = (instant - datetime.fromisoformat(device.probed_at)).total_seconds()
        if device.supported is not True or device.available is not True or not 0 <= age <= 300:
            raise ValueError()
    except (StopIteration, ValueError, TypeError):
        raise Refused('stale_or_unavailable_device') from None
