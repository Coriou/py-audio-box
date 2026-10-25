"""Typed native transcript clone preparation: whole canonical mono PCM16 only."""
from dataclasses import dataclass, fields
from .qwen_execution import QwenExecution
from worker.transport import Refused


@dataclass(frozen=True)
class QwenPreparation(QwenExecution):
    phase: str = 'prepare'

    def __post_init__(self):
        super().__post_init__()
        if self.phase != 'prepare' or self.task != 'clone' or self.x_vector_only:
            raise Refused('typed_preparation_required')
        if self.reference_bytes != 44 + 2*self.reference_frames or self.reference_bytes > 1048576:
            raise Refused('preparation_reference_limit')

    def generation(self):
        return QwenExecution(**{f.name: getattr(self, f.name) for f in fields(QwenExecution)})


def selected_reference(spec, raw):
    import hashlib
    from worker.engines.pocket import _reference_pcm16
    if type(spec) is not QwenPreparation: raise Refused('typed_preparation_required')
    if len(raw) != spec.reference_bytes or hashlib.sha256(raw).hexdigest() != spec.reference_sha256:
        raise Refused('reference_digest')
    _reference_pcm16(raw, spec.reference_rate, spec.reference_frames)
    # This finite mode preserves the complete canonical WAV, including exact PCM.
    import io, wave
    with wave.open(io.BytesIO(raw), 'rb') as source:
        pcm = source.readframes(spec.reference_frames)
    out = io.BytesIO()
    with wave.open(out, 'wb') as target:
        target.setnchannels(1); target.setsampwidth(2); target.setframerate(spec.reference_rate); target.writeframes(pcm)
    if out.getvalue() != raw: raise Refused('canonical_reference_required')
    return raw, spec.reference_frames


def output_assets(context, audition):
    from worker.engines.pocket_prepare import outputs
    from worker.runtime_protocol import execution
    spec = execution(context)
    result = dict(reference=dict(sha256=spec.reference_sha256, frames=spec.reference_frames), audition=audition)
    return outputs(context, result)
