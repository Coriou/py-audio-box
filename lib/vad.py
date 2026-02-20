"""
lib/vad.py — shared Silero VAD helpers.

Provides:
  load_silero   — load the snakers4/silero-vad model (optionally cached hub dir + device)
  run_silero    — run Silero on a WAV and return merged (start_sec, end_sec) segments

Both voice-clone and voice-split use the same core logic; they differ only in
their merge/min-length thresholds and the higher-level functions built on top.
Pass explicit ``merge_gap`` / ``min_len`` to override the defaults per caller.
"""

from pathlib import Path
from typing import Optional

import torch
import torchaudio

# Default merge/filter thresholds.
# voice-clone uses tighter values (cleaner ref clips, less speech tolerance).
# voice-split uses looser values (long-form scan of mixed content).
# Use keyword arguments at the call site to override.
_DEFAULT_MERGE_GAP = 0.20  # merge segments closer than this (seconds)
_DEFAULT_MIN_LEN   = 0.30  # drop segments shorter than this (seconds)


def load_silero(
    hub_dir: Optional[Path] = None,
    device: Optional[str] = None,
):
    """
    Load the ``snakers4/silero-vad`` model.

    Parameters
    ----------
    hub_dir:
        If given, sets ``torch.hub`` cache directory before loading.
        Pass ``cache / "torch" / "hub"`` to persist weights across container restarts.
    device:
        If given, the model is moved to this device.
        Omit (or pass ``None``) when the model will be used on CPU only
        or when the caller manages device placement manually.

    Returns ``(model, get_speech_timestamps)`` — the standard Silero two-tuple.
    """
    if hub_dir is not None:
        torch.hub.set_dir(str(hub_dir))

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    (get_speech_timestamps, *_) = utils

    if device is not None:
        model = model.to(device)

    return model, get_speech_timestamps


def run_silero(
    wav_path: Path,
    model,
    get_speech_timestamps,
    merge_gap: float = _DEFAULT_MERGE_GAP,
    min_len: float = _DEFAULT_MIN_LEN,
    device: Optional[str] = None,
) -> list[tuple[float, float]]:
    """
    Run Silero VAD on *wav_path* and return merged speech segments.

    Parameters
    ----------
    wav_path:
        Path to any audio file loadable by torchaudio.
    model:
        Silero VAD model (from ``load_silero``).
    get_speech_timestamps:
        Silero utility function (from ``load_silero``).
    merge_gap:
        Merge neighbouring segments separated by less than this many seconds.
        voice-clone default 0.20 s; voice-split uses 0.35 s.
    min_len:
        Drop segments shorter than this many seconds.
        voice-clone default 0.30 s; voice-split uses 0.60 s.
    device:
        Move input tensor to this device before inference.
        Pass ``None`` (default) to leave the tensor on CPU.

    Returns a list of ``(start_sec, end_sec)`` tuples, sorted and merged.
    """
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    TARGET_SR = 16_000
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR

    tensor = waveform.squeeze(0)
    if device is not None:
        tensor = tensor.to(device)

    timestamps = get_speech_timestamps(tensor, model, sampling_rate=sr)

    segs: list[tuple[float, float]] = []
    for ts in timestamps:
        s = ts["start"] / sr
        e = ts["end"]   / sr
        if e - s < min_len:
            continue
        if segs and s - segs[-1][1] <= merge_gap:
            segs[-1] = (segs[-1][0], max(segs[-1][1], e))
        else:
            segs.append((s, e))
    return segs
