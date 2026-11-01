"""One bounded local Kokoro call. No publication or service authority."""
import hashlib
import io
from numbers import Integral
from pathlib import Path
import wave
from worker.transport import Refused
from .execution import Execution


def execute(execution, text, bundles, *, backend_factory=None):
    """Return measured WAV bytes and detached identity; publish via existing protocol.

    bundles is an integration-owned mapping of authorized IDs to immutable local
    bundle directories. memory_bytes is a supervisor admission/OS limit, not a
    Python allocator guarantee. The legacy backend materializes one bounded-text
    inference result; it is closed after every call, including failures.
    """
    if type(execution) is not Execution:
        raise Refused('typed_execution_required')
    if type(text) is not str or not text or len(text.encode('utf-8')) > execution.max_text_bytes:
        raise Refused('text_limit')
    if execution.bundle_id not in bundles:
        raise Refused('unauthorized_bundle')
    root = Path(bundles[execution.bundle_id])
    with (root / 'manifest.json').open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != execution.manifest_sha256:
        raise Refused('bundle_identity')
    # Verification precedes even an injected backend; no runtime import here.
    from lib.tts_backends.assets import verified_bundle
    root, manifest = verified_bundle(root, 'kokoro')
    if (manifest.get('checkpoint'), manifest.get('revision'), manifest.get('language')) != (execution.checkpoint, execution.revision, execution.language):
        raise Refused('bundle_identity')
    if backend_factory is None:
        from lib.tts_backends.kokoro import KokoroBackend
        backend_factory = KokoroBackend
    backend = backend_factory(root, language=execution.language, threads=execution.threads)
    try:
        result = backend.synthesise(text, preset=execution.preset)
        from lib.tts_backends.contract import AudioResult
        if type(result) is not AudioResult:
            raise Refused('audio_result_required')
        import numpy as np
        data = np.asarray(result.samples)
        frames = data.shape[0] if data.ndim in (1, 2) else 0
        size = 44 + frames*2
        if not 0 < frames <= execution.max_frames or size > execution.max_output_bytes:
            raise Refused('output_limit')
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (result.sample_rate, result.channels)) or (result.sample_rate, result.channels) != (execution.sample_rate, 1) or data.shape not in ((frames,), (frames, 1)):
            raise Refused('audio_format')
        # AudioResult is mutable: revalidate after receiving it, in bounded chunks.
        if not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.complexfloating):
            raise Refused('invalid_audio')
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(execution.sample_rate)
            for start in range(0, frames, 4096):
                block = data[start:start+4096].reshape(-1).astype('float64')
                if not np.isfinite(block).all() or (np.abs(block) > 1).any():
                    raise Refused('invalid_audio')
                wav.writeframesraw((block * 32767).astype('<i2').tobytes())
        output = buffer.getvalue()
        if len(output) != size:
            raise Refused('output_size')
        return dict(data=output, frames=frames, size_bytes=len(output), sha256=hashlib.sha256(output).hexdigest(), identity=execution.identity())
    finally:
        backend.close()
