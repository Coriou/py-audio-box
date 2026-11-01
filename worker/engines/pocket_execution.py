"""Closed integration-owned Pocket CPU generation binding; no path authority."""
from dataclasses import dataclass, fields
import re
from worker.transport import Refused


@dataclass(frozen=True)
class PocketExecution:
    bundle_id: str
    manifest_sha256: str
    checkpoint: str
    revision: str
    language: str
    threads: int
    max_text_bytes: int
    max_frames: int
    max_output_bytes: int
    memory_bytes: int
    reference_id: str
    reference_sha256: str
    reference_bytes: int
    reference_frames: int
    reference_rate: int
    sample_rate: int
    engine: str = 'pocket-tts'
    device: str = 'cpu'
    channels: int = 1
    media_format: str = 'wav'
    media_subtype: str = 'PCM_16'

    def __post_init__(self):
        for key in ('bundle_id', 'reference_id'):
            if type(getattr(self, key)) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}', getattr(self, key)):
                raise Refused('execution_identity')
        for key in ('checkpoint', 'revision'):
            if type(getattr(self, key)) is not str or not 0 < len(getattr(self, key)) <= 256:
                raise Refused('execution_identity')
        for key in ('manifest_sha256', 'reference_sha256'):
            if type(getattr(self, key)) is not str or not re.fullmatch('[0-9a-f]{64}', getattr(self, key)):
                raise Refused('execution_digest')
        if self.language not in ('fr', 'en') or (self.engine, self.device, self.channels, self.media_format, self.media_subtype) != ('pocket-tts', 'cpu', 1, 'wav', 'PCM_16') or type(self.channels) is not int:
            raise Refused('execution_format')
        for key, ceiling in (('threads', 64), ('max_text_bytes', 65536), ('max_frames', 192000*300), ('max_output_bytes', 115200044), ('memory_bytes', 64*1024**3), ('reference_bytes', 192000*60*2+65536), ('reference_frames', 192000*60), ('reference_rate', 192000), ('sample_rate', 192000)):
            if type(getattr(self, key)) is not int or not 0 < getattr(self, key) <= ceiling:
                raise Refused('execution_limit')
        if self.reference_frames > self.reference_rate*60 or self.max_frames > self.sample_rate*300 or self.max_output_bytes < 46 or self.memory_bytes < self.max_output_bytes:
            raise Refused('execution_limit')

    @classmethod
    def from_wire(cls, value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise Refused('closed_execution_required')
        return cls(**value)

    def identity(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}
