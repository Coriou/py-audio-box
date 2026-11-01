"""Closed, detached Kokoro execution values for the integration owner.

These values confer no authority. The caller must supply the frozen attempt's
managed bundle mapping, never a path taken from a public request. Resource values
must be reserved/enforced by the supervisor before calling an engine.
"""
from dataclasses import dataclass, fields
import re
from worker.transport import Refused


@dataclass(frozen=True)
class Execution:
    bundle_id: str
    manifest_sha256: str
    checkpoint: str
    revision: str
    preset: str
    language: str
    threads: int
    max_text_bytes: int
    max_frames: int
    max_output_bytes: int
    memory_bytes: int
    device: str = 'cpu'
    engine: str = 'kokoro'
    sample_rate: int = 24000
    channels: int = 1
    media_format: str = 'wav'
    media_subtype: str = 'PCM_16'

    def __post_init__(self):
        for key in ('bundle_id', 'checkpoint', 'revision', 'preset'):
            value = getattr(self, key)
            if not isinstance(value, str) or not value or len(value) > 256:
                raise Refused('execution_identity')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', self.bundle_id):
            raise Refused('bundle_id')
        if not isinstance(self.manifest_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.manifest_sha256):
            raise Refused('manifest_digest')
        if (self.engine, self.device, self.sample_rate, self.channels, self.media_format, self.media_subtype) != ('kokoro', 'cpu', 24000, 1, 'wav', 'PCM_16') or type(self.sample_rate) is not int or type(self.channels) is not int:
            raise Refused('execution_format')
        if self.language not in ('fr', 'en') or not self.preset.startswith('f' if self.language == 'fr' else 'a'):
            raise Refused('preset_language')
        for key, ceiling in [('threads', 64), ('max_text_bytes', 65536), ('max_frames', 24000*300), ('max_output_bytes', 14400044), ('memory_bytes', 64*1024**3)]:
            value = getattr(self, key)
            if type(value) is not int or not 0 < value <= ceiling:
                raise Refused('execution_limit')
        if self.max_output_bytes < 46 or self.memory_bytes < self.max_output_bytes:
            raise Refused('execution_limit')

    @classmethod
    def from_wire(cls, value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise Refused('closed_execution_required')
        return cls(**value)

    def identity(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}
