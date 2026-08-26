#!/usr/bin/env python3
"""Stage 1 — timing oracle: extract sound events from Mirelo's raw soundtrack.

For each bench clip, the untouched Mirelo mp4 from the prior fal-sfx-bench run
is treated as the timing oracle (its samples are otherwise disposable). Audio
is analysed with a SuperFlux-style onset detector; every detected event gets

- ``time_s``      clip-relative onset time (frame resolution ≈ 5.8 ms)
- ``duration_s``  time until segment energy decays 21 dB under its peak
- ``intensity_db`` RMS level (dBFS) over a short post-onset window
- ``centroid_hz`` spectral centroid of the attack, for adaptation heuristics

Dependencies: numpy + scipy only (librosa/numba have no macOS x86_64 wheels).
Outputs per clip: ``oracle/<clip>.wav`` (analysis copy) and
``oracle/<clip>.events.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.io import wavfile

from common import (
    BENCH_RUN_DIR,
    atomic_json,
    extract_mono_wav,
    model_raw_source,
    ORACLE_MODELS,
    sha256,
    utc_now,
)

ANALYSIS_SR = 22_050
HOP = 128                      # ≈ 5.8 ms frame hop
N_FFT = 1024
MEL_BANDS = 80
FRAME_S = HOP / ANALYSIS_SR

#: Peak-picker geometry (frames) and prominence policy.
PEAK_PRE_MAX = 6                # ~35 ms
PEAK_POST_MAX = 6
PEAK_PRE_AVG = 17               # ~100 ms
PEAK_POST_AVG = 17
PEAK_WAIT_S = 0.11              # refractory: one event per physical impact
ENVELOPE_SMOOTH_S = 0.03        # merge sub-impact jitter into single peaks
#: Prominence above local envelope average (normalized units).
DELTA_NORM_MIN = 0.10
#: Amplitude-domain gate: an audible event must jump over its local past.
#: Mirelo mixes steps into an ambience bed, so jumps are small (p90 ≈ 4-6 dB).
ENERGY_JUMP_DB = 3.0
JUMP_PRE_S = 0.08               # reference level: mean RMS of the preceding…
JUMP_RISE_S = 0.06              # …versus the first 60 ms after the onset

DECAY_DB = 21.0                 # event ends 21 dB under its peak energy
MAX_DURATION_S = 1.5            # cap for sustained sources (surf)
INTENSITY_WINDOW_S = 0.12
ATTACK_CENTROID_WINDOW_S = 0.10


# ---------------------------------------------------------------------------
# Minimal spectral toolkit (STFT → mel → log) replacing librosa.
# ---------------------------------------------------------------------------

def hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequencies, dtype=np.float64) / 700.0)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mels, dtype=np.float64) / 2595.0) - 1.0)


def mel_band_centers(n_mels: int, fmax: float) -> np.ndarray:
    return mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(fmax), n_mels))


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Triangular mel filters, shape (n_mels, n_fft//2 + 1)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_points = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), n_mels + 2))
    widths = np.diff(mel_points)
    ramps = mel_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    return np.maximum(0.0, np.minimum(lower, upper))


def stft_power(audio: np.ndarray) -> np.ndarray:
    """Magnitude-squared STFT, shape (n_freqs, n_frames)."""
    window = np.hanning(N_FFT)
    frame_count = max(1, 1 + (audio.size - N_FFT) // HOP)
    indices = np.arange(N_FFT)[None, :] + HOP * np.arange(frame_count)[:, None]
    framed = audio[indices] * window
    spectrum = np.fft.rfft(framed, axis=1)
    return np.abs(spectrum).T ** 2


def power_to_db(power: np.ndarray, floor_db: float = 60.0) -> np.ndarray:
    reference = max(float(power.max()), 1e-12)
    floor = reference * 10.0 ** (-floor_db / 10.0)
    return 10.0 * np.log10(np.maximum(power, floor))


def superflux_envelope(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SuperFlux novelty envelope plus its log-mel spectrogram.

    Maximum-filter across frequency (3 bins), first-order difference with lag 1,
    half-wave rectified and summed over bands (Böck's SuperFlux, minus jitter
    compensation, acceptable at video frame rates).
    """
    power = stft_power(audio)
    mel_basis = mel_filterbank(ANALYSIS_SR, N_FFT, MEL_BANDS)
    mel_power = mel_basis @ power
    logmel = power_to_db(mel_power)
    smoothed = ndimage.maximum_filter1d(logmel, size=3, axis=0)
    difference = np.diff(smoothed, axis=1, prepend=smoothed[:, :1])
    envelope = np.maximum(difference, 0.0).sum(axis=0)
    return envelope.astype(np.float64), logmel


def detect_onsets(envelope: np.ndarray, audio: np.ndarray,
                  relax_gate_before_s: float = 0.0) -> tuple[list[int], float]:
    """Peak picking with texture-adaptive prominence and an energy-jump gate.

    The novelty envelope is smoothed (one physical impact → one peak), scaled
    by its 99.5th percentile, and peaks must stand above the local average
    plus ``DELTA_NORM_MIN``. Candidates must additionally jump ≥
    ``ENERGY_JUMP_DB`` over their local past in the amplitude domain — this
    rejects continuous ambience swells that flux alone cannot separate from
    impacts.

    ``relax_gate_before_s`` skips the jump gate for onsets this close to the
    start of the file (one-shot samples often begin AT the transient, so no
    honest "local past" exists).

    Returns (onset frames, effective delta used) for the audit record.
    """
    if envelope.size == 0:
        return [], 0.0
    smoothed = ndimage.uniform_filter1d(
        envelope, size=max(1, int(round(ENVELOPE_SMOOTH_S / FRAME_S))))
    ceiling = float(np.percentile(smoothed, 99.5))
    if ceiling <= 0:
        return [], 0.0
    normalized = smoothed / ceiling
    delta = DELTA_NORM_MIN
    wait_frames = max(1, int(round(PEAK_WAIT_S / FRAME_S)))
    pre_samples = int(JUMP_PRE_S * ANALYSIS_SR)
    rise_samples = int(JUMP_RISE_S * ANALYSIS_SR)
    total_samples = audio.size

    def audible_jump(frame: int) -> bool:
        onset_sample = frame * HOP
        pre = segment_rms_db(audio, max(0, onset_sample - pre_samples), onset_sample)
        post = segment_rms_db(
            audio, onset_sample, min(total_samples, onset_sample + rise_samples))
        return (post - pre) >= ENERGY_JUMP_DB

    peaks: list[int] = []
    for index in range(len(normalized)):
        value = normalized[index]
        if value <= 0:
            continue
        window = normalized[max(0, index - PEAK_PRE_MAX):index + PEAK_POST_MAX + 1]
        if value < window.max():
            continue
        avg_window = normalized[max(0, index - PEAK_PRE_AVG):index + PEAK_POST_AVG + 1]
        if value < float(avg_window.mean()) + delta:
            continue
        if index * FRAME_S >= relax_gate_before_s and not audible_jump(index):
            continue
        if peaks and index - peaks[-1] < wait_frames:
            if value > normalized[peaks[-1]]:
                peaks[-1] = index
            continue
        peaks.append(index)
    return peaks, round(delta, 4)


# ---------------------------------------------------------------------------
# Per-event measurement.
# ---------------------------------------------------------------------------

def segment_rms_db(audio: np.ndarray, start: int, stop: int) -> float:
    window = audio[start:stop]
    if window.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(window))))
    return max(-120.0, 20.0 * np.log10(max(rms, 1e-9)))


def measure_event(audio: np.ndarray, onset_frame: int, logmel: np.ndarray) -> dict[str, Any]:
    onset_sample = onset_frame * HOP
    total_samples = audio.size
    peak_db = segment_rms_db(
        audio, onset_sample,
        min(total_samples, onset_sample + int(INTENSITY_WINDOW_S * ANALYSIS_SR)),
    )
    floor_linear = 10.0 ** ((peak_db - DECAY_DB) / 20.0)
    limit = min(total_samples - 1, onset_sample + int(MAX_DURATION_S * ANALYSIS_SR))
    position = onset_sample
    while position < limit:
        window_end = min(total_samples, position + HOP)
        if segment_rms_db(audio, position, window_end) < 20.0 * np.log10(max(floor_linear, 1e-9)):
            break
        position += HOP
    duration_s = (position - onset_sample) / ANALYSIS_SR

    attack_end = min(logmel.shape[1], onset_frame + max(2, int(ATTACK_CENTROID_WINDOW_S / FRAME_S)))
    band_power = 10.0 ** (logmel[:, onset_frame:attack_end] / 10.0)
    weights = band_power.sum(axis=1)
    centers = mel_band_centers(MEL_BANDS, ANALYSIS_SR / 2.0)
    centroid_hz = float(np.sum(centers * weights) / max(float(np.sum(weights)), 1e-12))

    salience = float(np.sum(10.0 ** (logmel[:, onset_frame] / 10.0)))
    return {
        "time_s": round(onset_frame * FRAME_S, 4),
        "duration_s": round(duration_s, 4),
        "intensity_db": round(peak_db, 2),
        "salience": round(salience, 8),
        "centroid_hz": round(centroid_hz, 1),
    }


def analyse_clip(clip: str, model: str, oracle_dir: Path, bench_run_dir: Path) -> dict[str, Any]:
    source = model_raw_source(bench_run_dir, model, clip)
    wav_path = oracle_dir / f"{clip}.{model}.wav"
    extract_mono_wav(source, wav_path, ANALYSIS_SR)
    sample_rate, audio = wavfile.read(wav_path)
    assert sample_rate == ANALYSIS_SR, f"expected {ANALYSIS_SR} Hz, got {sample_rate}"
    audio = audio.astype(np.float64)

    envelope, logmel = superflux_envelope(audio)
    onset_frames, effective_delta = detect_onsets(envelope, audio)
    events = []
    for index, frame in enumerate(onset_frames, start=1):
        events.append({"id": f"ev_{index:03d}", **measure_event(audio, int(frame), logmel)})

    record = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "clip": clip,
        "model": model,
        "source": {
            "path": str(source),
            "sha256": sha256(source),
        },
        "audio": {"path": wav_path.name, "sample_rate": ANALYSIS_SR},
        "detector": {
            "name": "superflux-logmel",
            "sr": ANALYSIS_SR, "hop": HOP, "n_fft": N_FFT, "n_mels": MEL_BANDS,
            "delta_norm_min": DELTA_NORM_MIN,
            "effective_delta_norm": effective_delta,
            "wait_s": PEAK_WAIT_S, "smooth_s": ENVELOPE_SMOOTH_S,
            "energy_jump_db": ENERGY_JUMP_DB,
            "decay_db": DECAY_DB, "max_duration_s": MAX_DURATION_S,
        },
        "event_count": len(events),
        "events": events,
    }
    atomic_json(oracle_dir / f"{clip}.{model}.events.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="pipeline run directory (oracle/ written inside)")
    parser.add_argument("--bench-dir", type=Path, default=BENCH_RUN_DIR)
    parser.add_argument("--clips", default="walk1,walk2,man,ocean-long,ocean-short")
    parser.add_argument("--models", default=",".join(ORACLE_MODELS),
                        help="benchmark models whose raw outputs to analyse")
    parser.add_argument("--force", action="store_true", help="re-analyse existing clips")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips = [part.strip() for part in args.clips.split(",") if part.strip()]
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    oracle_dir = args.out_dir / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        for clip in clips:
            target = oracle_dir / f"{clip}.{model}.events.json"
            if target.exists() and not args.force:
                record = json.loads(target.read_text())
                reused = True
            else:
                record = analyse_clip(clip, model, oracle_dir, args.bench_dir)
                reused = False
            print(f"{model}/{clip}: {record['event_count']} events"
                  f"{' (reused)' if reused else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
