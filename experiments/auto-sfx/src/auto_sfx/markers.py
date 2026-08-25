"""Synthetic marker-click audio for the auto-sfx POC.

Turns a list of ContactEvents into a stereo click track: every observable
foot contact becomes a short percussive click placed at the event time.
Left-foot events are panned hard-left and right-foot events hard-right,
and each side gets its own pitch so the sides stay distinguishable on mono
playback too.

Audio synthesis is stdlib `wave` + numpy only — no scipy, no ffmpeg.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import LEFT, RIGHT, ContactEvent

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 48_000
CLICK_DURATION_S = 0.07          # one click, including decay tail
CLICK_LEFT_HZ = 1_500.0          # left foot  -> lower thunk
CLICK_RIGHT_HZ = 2_300.0         # right foot -> brighter tick
SIDE_PAN = {LEFT: (1.0, 0.30), RIGHT: (0.30, 1.0)}   # (L gain, R gain)
KIND_GAIN = {"initial_contact": 1.0, "toe_off": 0.65}

# Phase-2 T15: ground-truth marks get a SECOND, distinct voice so phase
# error is audible against prediction clicks: much lower pitch, longer
# decay, centred in the stereo image (predictions are hard-panned).
GT_CLICK_HZ = 620.0
GT_CLICK_DURATION_S = 0.13
GT_CLICK_PAN = (0.72, 0.72)
GT_CLICK_GAIN = 0.85


def side_freq(side: str) -> float:
    return CLICK_LEFT_HZ if side == LEFT else CLICK_RIGHT_HZ


# ---------------------------------------------------------------------------
# Click synthesis
# ---------------------------------------------------------------------------


def click_waveform(
    freq: float,
    sr: int = DEFAULT_SAMPLE_RATE,
    duration_s: float = CLICK_DURATION_S,
    seed: int | None = None,
) -> np.ndarray:
    """One percussive click as a mono float array with peak amplitude 1.

    Damped sinusoid body + a low-passed noise transient; both share an
    exponential decay so the click reads as a single strike.
    """

    n = max(2, int(round(sr * duration_s)))
    t = np.arange(n, dtype=np.float64) / sr
    tau = max(duration_s / 6.0, 1e-4)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(9) / 9.0                       # crude low-pass, numpy only
    noise = np.convolve(noise, kernel, mode="same")

    body = np.sin(2.0 * np.pi * freq * t) + 0.45 * np.sin(2.0 * np.pi * freq * 2.71 * t)
    thump = np.sin(2.0 * np.pi * (freq / 4.0) * t)
    env_body = np.exp(-t / tau)
    env_noise = np.exp(-t / (tau * 0.45))

    sig = 0.62 * body * env_body + 0.22 * thump * env_body + 0.30 * noise * env_noise

    peak = float(np.max(np.abs(sig)))
    if peak > 0:
        sig /= peak
    fade = min(n // 4, int(sr * 0.005))             # de-click the tail to zero
    if fade > 1:
        sig[-fade:] *= np.linspace(1.0, 0.0, fade)
    return sig


# ---------------------------------------------------------------------------
# Track assembly
# ---------------------------------------------------------------------------


def default_duration_s(events: list[ContactEvent], tail_s: float = 0.75) -> float:
    """Track length covering the last event plus a decay tail."""

    latest = max((e.time_s for e in events), default=0.0)
    return max(latest + tail_s, 1.0)


def render_click_track(
    events: list[ContactEvent],
    duration_s: float,
    sr: int = DEFAULT_SAMPLE_RATE,
    gain: float = 0.8,
    gt_marks: Iterable[float] | None = None,
) -> np.ndarray:
    """Mix events into an interleaved-ready (n, 2) int16 stereo buffer.

    Non-observable events are silent by definition; events outside
    [0, duration_s) are dropped. Overlapping clicks add linearly and the
    whole mix is peak-normalised down into [-1, 1] before quantisation.

    ``gt_marks`` are ground-truth preferred instants (seconds, Phase-2 T15);
    each is rendered with the distinct GT voice (see GT_CLICK_* constants)
    so phase error against prediction clicks is audible by ear.
    """

    n = max(1, int(round(duration_s * sr)))
    mix = np.zeros((n, 2), dtype=np.float64)

    for e in events or []:
        if not getattr(e, "observable", True):
            continue
        if e.time_s < 0.0 or e.time_s >= duration_s:
            continue
        seg = click_waveform(side_freq(e.foot), sr=sr)
        i0 = int(round(e.time_s * sr))
        i1 = min(n, i0 + len(seg))
        if i1 <= i0:
            continue
        pan_l, pan_r = SIDE_PAN.get(e.foot, SIDE_PAN[LEFT])
        amp = float(np.clip(getattr(e, "score", 0.8), 0.35, 1.0)) * gain
        amp *= KIND_GAIN.get(getattr(e, "kind", "initial_contact"), 1.0)
        mix[i0:i1, 0] += seg[: i1 - i0] * pan_l * amp
        mix[i0:i1, 1] += seg[: i1 - i0] * pan_r * amp

    for t_s in gt_marks or []:
        if t_s < 0.0 or t_s >= duration_s:
            continue
        seg = click_waveform(
            GT_CLICK_HZ, sr=sr, duration_s=GT_CLICK_DURATION_S, seed=0
        )
        i0 = int(round(t_s * sr))
        i1 = min(n, i0 + len(seg))
        if i1 <= i0:
            continue
        pan_l, pan_r = GT_CLICK_PAN
        amp = GT_CLICK_GAIN * gain / 0.8     # neutral at the default gain
        mix[i0:i1, 0] += seg[: i1 - i0] * pan_l * amp
        mix[i0:i1, 1] += seg[: i1 - i0] * pan_r * amp

    peak = float(np.max(np.abs(mix), initial=0.0))
    if peak > 1.0:
        mix /= peak
    return np.clip(np.round(mix * 32767.0), -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# WAV output
# ---------------------------------------------------------------------------


def write_wav_stereo_int16(path: str | Path, stereo: np.ndarray, sr: int) -> str:
    """Write an (n, 2) int16 array as a 16-bit PCM WAV via stdlib `wave`."""

    arr = np.asarray(stereo)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (n, 2) stereo array, got shape {arr.shape}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.ascontiguousarray(arr, dtype="<i2").reshape(-1)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)                  # 16-bit PCM
        wf.setframerate(int(sr))
        wf.writeframes(interleaved.tobytes())
    return str(out)


def synthesize_marker_wav(
    events: list[ContactEvent],
    duration_s: float,
    out_path: str | Path,
    *,
    sr: int = DEFAULT_SAMPLE_RATE,
    gain: float = 0.8,
    gt_marks: Iterable[float] | None = None,
) -> str:
    """Full pipeline: ContactEvents (+ optional GT marks) -> marker WAV."""

    stereo = render_click_track(
        events, duration_s=duration_s, sr=sr, gain=gain, gt_marks=gt_marks
    )
    return write_wav_stereo_int16(out_path, stereo, sr)
