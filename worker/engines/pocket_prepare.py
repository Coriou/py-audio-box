"""Exact PCM16 mono segment extraction and explicit Pocket spoken audition.

No denoise, normalization, resampling, transcript interpretation or inferred text.
Only frame-aligned 48kHz ticks are supported. Managed input and namespace policies
are the reviewed Pocket adapter's; caller owns cleanup after confirmed reap.
"""
from dataclasses import dataclass
import hashlib
import io
import wave
from .pocket_execution import PocketExecution
from worker.transport import Refused


@dataclass(frozen=True)
class PocketPreparation(PocketExecution):
    """reference_* binds the complete original; sample_rate binds the audition.

    Reference output retains the original rate. Both ticks are exact frame
    boundaries; phase/transform are closed private authority, not public controls.
    """
    in_tick: int = 0
    out_tick: int = 1
    transform: str = 'pcm16-mono-exact-segment-v1'
    phase: str = 'prepare'

    def __post_init__(self):
        super().__post_init__()
        if (self.phase != 'prepare' or self.transform != 'pcm16-mono-exact-segment-v1' or
                type(self.in_tick) is not int or type(self.out_tick) is not int or
                not 0 <= self.in_tick < self.out_tick or
                self.in_tick*self.reference_rate % 48000 or self.out_tick*self.reference_rate % 48000 or
                self.out_tick*self.reference_rate > self.reference_frames*48000):
            raise Refused('unsupported_segment')


def selected_reference(e, raw):
    """Deterministic canonical WAV, preserving every selected original PCM byte."""
    from .pocket import _reference_pcm16
    if type(e) is not PocketPreparation: raise Refused('typed_preparation_required')
    if len(raw) != e.reference_bytes or hashlib.sha256(raw).hexdigest() != e.reference_sha256:
        raise Refused('reference_digest')
    _reference_pcm16(raw, e.reference_rate, e.reference_frames)
    start = e.in_tick*e.reference_rate//48000
    frames = (e.out_tick-e.in_tick)*e.reference_rate//48000
    with wave.open(io.BytesIO(raw), 'rb') as source:
        source.setpos(start)
        pcm = source.readframes(frames)
    if len(pcm) != frames*2: raise Refused('reference_format')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as target:
        target.setnchannels(1); target.setsampwidth(2); target.setframerate(e.reference_rate)
        target.writeframes(pcm)
    return buffer.getvalue(), frames


def execute(e, text, bundles, references, *, reference_temp_dir, backend_factory=None, managed_owner_uid=None, reference_fd=None, managed_namespace=False):
    from .pocket import verify_inputs, _synthesise
    if type(e) is not PocketPreparation: raise Refused('typed_preparation_required')
    if type(text) is not str or not text or len(text.encode('utf-8')) > e.max_text_bytes:
        raise Refused('text_limit')
    raw = verify_inputs(e, bundles, references, owner_uid=managed_owner_uid, reference_fd=reference_fd)
    prepared, frames = selected_reference(e, raw)
    audition = _synthesise(e, text, bundles[e.bundle_id], prepared, reference_temp_dir, backend_factory, managed_namespace=managed_namespace)
    return dict(reference=dict(data=prepared, frames=frames, size_bytes=len(prepared),
                               sha256=hashlib.sha256(prepared).hexdigest()), audition=audition)


def outputs(context, result):
    """Closed role metadata; the Core publisher independently measures both files."""
    from worker.runtime_protocol import execution, generated_source
    e = execution(context); plan = context['plan']; roles = plan['reference_roles']
    original = plan['assets'][0]
    assets = []
    for role, rate, provenance in (
            ('reference', e.reference_rate, dict(kind=original['provenance']['kind'],
                source_sha256=original['sha256'], source_id=original['id'], missing_fields=[])),
            ('audition', e.sample_rate, dict(kind='generated', source_sha256=result['audition']['sha256'],
                source_id=generated_source(context), missing_fields=[]))):
        audio = result[role]
        assets.append(dict(id=roles[role+'_asset_id'], sha256=audio['sha256'], frames=audio['frames'],
            sample_rate=rate, channels=1, media_format='wav', media_subtype='PCM_16', provenance=provenance))
    return assets
