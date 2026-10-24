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
        # Seed support is not established by the local capability contract.
        if self.seed is not None: raise Refused('unsupported_seed')
        for value in (self.preset, self.instruct, self.reference_text):
            if value is not None and (type(value) is not str or not value.strip() or len(value.encode()) > 65536): raise Refused('execution_control')
        if type(self.x_vector_only) is not bool: raise Refused('execution_control')
        if self.task == 'clone':
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
