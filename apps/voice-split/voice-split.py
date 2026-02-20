#!/usr/bin/env python3
"""
voice-split.py — Download a YouTube video and produce voice-optimised WAV clips.

Smart pipeline (avoids running Demucs on the full file):
  1. Download audio            (cached per video)
  2. Fast VAD on raw audio     (Silero, ~10s even for 1hr — cached)
  3. Select candidate windows  (speech-densest N chunks of --window-len seconds)
  4. Extract each window       (ffmpeg, cheap)
  5. Demucs on each window     (cached per window — only runs once)
  6. Precise VAD on vocals     (cached per window)
  7. Pick & export clips

Output structure (clips are always scoped so multiple voices never collide):
    --voice-name given  →  <out>/<slug>/clip_NN_from_Xs.wav
                           <out>/<slug>/clip_ref_from_Xs.wav
    standalone (no --voice-name)  →  <out>/<source-id>/clip_NN_from_Xs.wav

Usage:
    python apps/voice-split/voice-split.py --url "https://www.youtube.com/watch?v=..." --voice-name my-voice --out /work
    # → clips land in /work/my-voice/
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio
import yt_dlp

# ── constants ─────────────────────────────────────────────────────────────────

VOICE_AF = (
    "highpass=f=80,lowpass=f=8000,"
    "afftdn=nf=-25,"
    "compand=attacks=0:decays=0.25:points=-90/-90|-35/-18|-10/-8|0/-6"
)

# ── yt-dlp helpers ────────────────────────────────────────────────────────────

YT_DLP_BASE_OPTS: dict = {
    "quiet": False,
    "no_warnings": False,
}


def _ydl_opts(extra: dict | None = None, cookies: str | None = None) -> dict:
    opts = {**YT_DLP_BASE_OPTS}
    if cookies:
        opts["cookiefile"] = cookies
    if extra:
        opts.update(extra)
    return opts


def get_video_id(url: str, cookies: str | None = None) -> str:
    opts = _ydl_opts({"skip_download": True}, cookies=cookies)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["id"]


def download_audio(url: str, dl_cache: Path, video_id: str, cookies: str | None = None) -> Path:
    """Download best audio as .m4a; skip if already cached."""
    target = dl_cache / f"{video_id}.m4a"
    if target.exists():
        print(f"  [cache hit] audio → {target}")
        return target

    # yt-dlp may produce a different extension (e.g. .webm) on some videos;
    # accept any cached file for this video_id to avoid a wasted re-download.
    dl_cache.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dl_cache.glob(f"{video_id}.*") if p.suffix != ".part"]
    if existing:
        cached = existing[0]
        print(f"  [cache hit] audio → {cached}")
        return cached
    opts = _ydl_opts(
        {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                    "preferredquality": "0",
                }
            ],
            "outtmpl": str(dl_cache / f"{video_id}.%(ext)s"),
            "no_playlist": True,
        },
        cookies=cookies,
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    if not target.exists():
        candidates = list(dl_cache.glob(f"{video_id}.*"))
        if candidates:
            target = candidates[0]
        else:
            raise FileNotFoundError(f"No downloaded audio found for {video_id} in {dl_cache}")
    return target


# ── audio helpers ─────────────────────────────────────────────────────────────

def get_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def parse_timestamp(s: str) -> float:
    """
    Parse a timestamp string to seconds.

    Accepted formats:
      - Raw seconds:          "90"  or "90.5"
      - Minutes:seconds:      "1:30"  or "1:30.5"
      - Hours:minutes:seconds "1:02:30"
    """
    parts = s.strip().split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    import argparse as _ap
    raise _ap.ArgumentTypeError(
        f"Cannot parse timestamp {s!r}. "
        "Use seconds (90 / 90.5), MM:SS (1:30), or HH:MM:SS (1:02:30)."
    )


def trim_audio(audio: Path, start_sec: float, end_sec: float, dest: Path) -> Path:
    """
    Trim audio to [start_sec, end_sec] via ffmpeg stream copy (no re-encode).
    Result is cached — re-runs are instant.
    """
    if dest.exists():
        print(f"  [cache hit] trimmed audio \u2192 {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dur = end_sec - start_sec
    print(f"  trimming {start_sec:.1f}s \u2013 {end_sec:.1f}s  ({dur:.1f}s)  \u2192 {dest.name}")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(audio),
            "-t", f"{dur:.3f}",
            "-c", "copy",
            str(dest),
        ],
        check=True,
    )
    return dest


def extract_chunk(audio: Path, start: float, length: float, dest: Path) -> None:
    """Extract [start, start+length] from audio, decode to 44.1kHz stereo WAV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio),
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(dest),
        ],
        check=True,
    )


# ── device helper ──────────────────────────────────────────────────────────────

def _get_device() -> str:
    """
    Select the best available compute device.

    Resolution order:
      1. ``TORCH_DEVICE`` env var  — explicit override.
                                     Set automatically by docker-compose.gpu.yml.
      2. CUDA                      — when a GPU is present and torch was built with
                                     CUDA support (i.e. the CUDA image variant).
      3. CPU                       — universal fallback.

    Silero VAD and its input tensors are moved to this device.
    Demucs runs as a subprocess and detects CUDA independently.
    """
    override = os.getenv("TORCH_DEVICE", "").strip()
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Silero VAD ──────────────────────────────────────────────────────────────

def _load_silero():
    device = _get_device()
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    model = model.to(device)
    (get_speech_timestamps, *_) = utils
    return model, get_speech_timestamps


def _run_silero(wav_path: Path, model, get_speech_timestamps) -> list[list[float]]:
    """Return merged speech segments [[start_sec, end_sec], ...] for wav_path."""
    device   = _get_device()
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sr = 16_000
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    timestamps = get_speech_timestamps(waveform.squeeze(0).to(device), model, sampling_rate=sr)

    MERGE_GAP = 0.35   # merge segments closer than this (seconds)
    MIN_LEN   = 0.6    # drop segments shorter than this (seconds)

    segs: list[list[float]] = []
    for ts in timestamps:
        s = ts["start"] / sr
        e = ts["end"]   / sr
        if e - s < MIN_LEN:
            continue
        if segs and s - segs[-1][1] <= MERGE_GAP:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    return segs


def raw_vad_scan(audio: Path, cache: Path, video_id: str,
                 max_seconds: float | None, model, get_ts) -> list[list[float]]:
    """
    Run Silero VAD on the raw (unseparated) audio.
    Fast (~10s for 1hr) and sufficient to find speech-dense regions.
    Result is cached as JSON.
    """
    suffix = "" if max_seconds is None else f"_first{int(max_seconds)}s"
    seg_path = cache / f"{video_id}{suffix}_raw_segments.json"

    if seg_path.exists():
        with open(seg_path) as f:
            segs = json.load(f)
        print(f"  [cache hit] {len(segs)} raw VAD segments → {seg_path}")
        return segs

    # Convert to 16-kHz mono WAV for Silero
    tmp_wav = cache / f"{video_id}{suffix}_raw_16k.wav"
    if not tmp_wav.exists():
        print("  converting raw audio to 16kHz mono for VAD scan...")
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio),
            "-ar", "16000", "-ac", "1",
        ]
        if max_seconds is not None:
            ffmpeg_cmd += ["-t", str(max_seconds)]
        ffmpeg_cmd.append(str(tmp_wav))
        subprocess.run(ffmpeg_cmd, check=True)

    print("  running Silero VAD on raw audio...")
    segs = _run_silero(tmp_wav, model, get_ts)

    # Clean up the temp wav (large); the JSON is our real cache
    tmp_wav.unlink(missing_ok=True)

    cache.mkdir(parents=True, exist_ok=True)
    with open(seg_path, "w") as f:
        json.dump(segs, f)
    print(f"  saved {len(segs)} raw segments → {seg_path}")
    return segs


def select_windows(
    raw_segs: list[list[float]],
    n: int,
    window_len: float,
    audio_duration: float,
) -> list[tuple[float, float]]:
    """
    Choose n non-overlapping windows of window_len seconds that maximise
    speech coverage according to raw_segs.

    Slides with stride=window_len/4, scores each position by total speech
    duration inside the window, then greedily picks top-n non-overlapping.
    """
    # If the audio is shorter than the requested window, clamp so we always
    # get at least one candidate window covering the full audio.
    window_len = min(window_len, audio_duration)
    stride = max(1.0, window_len / 4)
    candidates: list[tuple[float, float, float]] = []  # (coverage, start, end)

    t = 0.0
    while t + window_len <= audio_duration + 1e-6:
        win_end = min(t + window_len, audio_duration)
        coverage = sum(
            min(e, win_end) - max(s, t)
            for s, e in raw_segs
            if e > t and s < win_end
        )
        candidates.append((coverage, t, win_end))
        t += stride

    if not candidates:
        return []

    candidates.sort(reverse=True)

    selected: list[tuple[float, float]] = []
    for _cov, ws, we in candidates:
        if any(ws < pe and we > ps for ps, pe in selected):
            continue
        selected.append((ws, we))
        if len(selected) == n:
            break

    selected.sort()
    return selected


# ── Demucs ────────────────────────────────────────────────────────────────────

def separate_vocals_chunk(chunk: Path, sep_cache: Path) -> Path:
    """
    Run Demucs htdemucs two-stems on a single short chunk.
    Cached under sep_cache/htdemucs/<chunk_stem>/vocals.wav.
    """
    vocals = sep_cache / "htdemucs" / chunk.stem / "vocals.wav"
    if vocals.exists():
        print(f"    [cache hit] vocals → {vocals.parent.name}")
        return vocals

    sep_cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", "htdemucs",
        "--two-stems", "vocals",
        "--out", str(sep_cache),
        str(chunk),
    ]
    print(f"    demucs running on {chunk.name} ...")
    subprocess.run(cmd, check=True)

    if not vocals.exists():
        raise FileNotFoundError(f"Demucs did not produce {vocals}")
    return vocals


# ── VAD on vocal chunks ───────────────────────────────────────────────────────

def vad_on_vocals(vocals: Path, cache: Path, model, get_ts) -> list[list[float]]:
    """Precise VAD on a separated vocals chunk. Cached per chunk."""
    seg_path = cache / f"{vocals.parent.name}_vocal_segments.json"
    if seg_path.exists():
        with open(seg_path) as f:
            segs = json.load(f)
        print(f"    [cache hit] {len(segs)} vocal segments")
        return segs

    segs = _run_silero(vocals, model, get_ts)

    cache.mkdir(parents=True, exist_ok=True)
    with open(seg_path, "w") as f:
        json.dump(segs, f)
    return segs


# ── clip selection & export ───────────────────────────────────────────────────

def pick_clips(
    pool: list[tuple[Path, float, float, float]],  # (vocals_path, file_dur, seg_start, seg_end)
    n: int,
    target_len: float,
) -> list[tuple[Path, float, float]]:              # (vocals_path, clip_start, clip_len)
    """
    Anchor clip start within a speech segment, then extend forward by target_len
    regardless of speech/silence boundaries.  Clips are always exactly target_len
    seconds long (or shorter only if the file itself is too short).
    """
    # Build candidate anchor ranges: [seg_start, latest_start] where latest_start
    # still leaves target_len seconds before end-of-file.
    anchors: list[tuple[Path, float, float, float]] = []  # (vocals, seg_s, seg_e_clamped, clip_len)
    weights: list[float] = []
    for vocals, file_dur, s, e in pool:
        latest = file_dur - target_len
        if latest >= s:
            # Full target_len fits; anchor anywhere in [s, min(e, latest)]
            anchor_end = min(e, latest)
            anchors.append((vocals, s, anchor_end, target_len))
            weights.append(anchor_end - s)

    if not anchors:
        # Fallback: file shorter than target_len — take as much as possible
        for vocals, file_dur, s, e in pool:
            clip_len = file_dur - s
            if clip_len > 0.5:
                anchors.append((vocals, s, e, clip_len))
                weights.append(e - s)

    total = sum(weights)
    w = [v / total for v in weights] if total > 0 else None

    plans: list[tuple[Path, float, float]] = []
    for _ in range(n):
        vocals, seg_s, seg_e, clip_len = random.choices(anchors, weights=w, k=1)[0]
        start = random.uniform(seg_s, seg_e)
        plans.append((vocals, start, clip_len))
    return plans


def export_clips(plans: list[tuple[Path, float, float]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for idx, (vocals, start, length) in enumerate(plans, 1):
        dest = out / f"clip_{idx:02d}_from_{start:.1f}s.wav"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(vocals),
                "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
                "-vn", "-af", VOICE_AF,
                "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
                str(dest),
            ],
            check=True,
        )
        print(f"  wrote {dest}")


def _interactive_pick_clip(out: Path) -> Path | None:
    """
    Present a numbered menu of all clip_*.wav files in *out* and return
    the user's chosen Path.  Returns None if the user accepts the default
    (the auto-selected clip_ref_* file) or if the prompt is skipped.

    Intended for interactive runs where the user wants to listen to all
    clips before deciding which segment to hand to voice-clone.
    """
    clips = sorted(out.glob("clip_*.wav"))
    if not clips:
        print("  (no clip_*.wav files found — skipping interactive selection)", file=sys.stderr)
        return None

    # Mark the auto-selected registry clip so the user knows the default.
    ref_clips = [p for p in clips if p.name.startswith("clip_ref_")]
    default = ref_clips[0] if ref_clips else clips[0]

    line = "─" * 60
    print(f"\n{line}")
    print("  Interactive clip selection")
    print(f"{line}")
    print(f"  Clips are in: {out}")
    print()
    for i, p in enumerate(clips):
        dur = get_duration(p)
        marker = "  ← auto-selected (best score)" if p == default else ""
        print(f"  [{i}] {p.name}  ({dur:.1f}s){marker}")
    print()
    print("  Listen to each clip, then enter the number of the one you")
    print("  want to use as the reference for voice cloning.")
    default_idx = clips.index(default)
    while True:
        try:
            raw = input(f"  Your choice [default {default_idx}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  (no tty / interrupted — keeping auto-selected clip)")
            return None
        if raw == "":
            print(f"  ✓ keeping auto-selected: {default.name}")
            return None
        try:
            idx = int(raw)
        except ValueError:
            print(f"  Please enter a number between 0 and {len(clips) - 1}.")
            continue
        if not 0 <= idx < len(clips):
            print(f"  Please enter a number between 0 and {len(clips) - 1}.")
            continue
        chosen = clips[idx]
        if chosen == default:
            print(f"  ✓ keeping auto-selected: {chosen.name}")
            return None
        print(f"  ✓ selected: {chosen.name}")
        return chosen


def export_registry_clip(
    pool: list[tuple[Path, float, float, float]],
    target_len: float,
    out: Path,
) -> Path | None:
    """
    Export a single deterministic best-quality clip for voice registry use.

    Picks the pool segment with the longest contiguous speech run (most likely
    to have sustained, solo speech) and anchors the clip at the segment start
    so that voice-clone\'s VAD+scoring gets maximum clean speech to work with.

    Returns the exported path (``out/clip_ref_from_<t>s.wav``) or None.
    """
    if not pool:
        return None

    # Rank by speech segment duration — longer = more sustained solo speech
    ranked = sorted(pool, key=lambda x: x[3] - x[2], reverse=True)

    for vocals, file_dur, s, e in ranked:
        clip_len = min(target_len, file_dur - s)
        if clip_len < 5.0:  # need at least 5s for voice-clone to work well
            continue
        dest = out / f"clip_ref_from_{s:.1f}s.wav"
        if not dest.exists():
            out.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(vocals),
                    "-ss", f"{s:.3f}", "-t", f"{clip_len:.3f}",
                    "-vn", "-af", VOICE_AF,
                    "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
                    str(dest),
                ],
                check=True,
            )
            print(f"  wrote registry clip → {dest.name}")
        else:
            print(f"  [cache hit] registry clip → {dest.name}")
        return dest

    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract clean, voice-only WAV clips from a YouTube video or a local audio file. "
            "Use --url for any yt-dlp source, or --audio for a local file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── audio source (exactly one required) ──────────────────────────────────
    src_grp = ap.add_mutually_exclusive_group(required=True)
    src_grp.add_argument(
        "--url",
        metavar="URL",
        help="YouTube (or any yt-dlp-supported) URL to extract audio from",
    )
    src_grp.add_argument(
        "--audio",
        metavar="PATH",
        help=(
            "Path to a local audio file (WAV, MP3, M4A, FLAC, \u2026). "
            "Skips the download step. "
            "Mount the file into the container with -v /host/path:/container/path."
        ),
    )
    # ── optional timestamp trim ───────────────────────────────────────────────
    ap.add_argument(
        "--start",
        default=None, metavar="TIMESTAMP", type=parse_timestamp,
        help=(
            "Trim start time in the source audio. "
            "Accepts: seconds (90 / 1:30.5), MM:SS (1:30), HH:MM:SS (1:02:30). "
            "Only audio from this point onwards is processed."
        ),
    )
    ap.add_argument(
        "--end",
        default=None, metavar="TIMESTAMP", type=parse_timestamp,
        help=(
            "Trim end time in the source audio. Same formats as --start. "
            "Only audio up to this point is processed."
        ),
    )
    ap.add_argument("--out",    default="/work",
        help="Root output directory. Clips are written to <out>/<voice-name>/ (or <out>/<source-id>/ when --voice-name is omitted) so multiple voices never collide.")
    ap.add_argument("--clips",  type=int,   default=10,  help="Number of clips to produce")
    ap.add_argument("--length", type=float, default=30,  help="Target clip length in seconds")
    ap.add_argument(
        "--window-len",
        type=float, default=None, metavar="SECS",
        help="Demucs chunk size in seconds (default: 4× --length, e.g. 120s for 30s clips)",
    )
    ap.add_argument(
        "--candidates",
        type=int, default=None, metavar="N",
        help="Number of candidate windows to run Demucs on (default: 2× --clips)",
    )
    ap.add_argument("--cache",  default="/cache",        help="Persistent cache directory")
    ap.add_argument("--seed",   type=int,   default=None, help="Random seed for reproducibility")
    ap.add_argument(
        "--max-scan-seconds",
        type=float, default=None, metavar="SECS",
        help="Limit raw VAD scan to first N seconds (useful for quick tests)",
    )
    ap.add_argument(
        "--cookies",
        default=None, metavar="FILE",
        help="Netscape cookies.txt for YouTube downloads (mount into container)",
    )
    ap.add_argument(
        "--voice-name",
        default=None, metavar="SLUG",
        help=(
            "Register best extracted clip as a named voice in /cache/voices/<slug>/. "
            "Slug must be lowercase with hyphens only (e.g. 'david-attenborough')."
        ),
    )
    ap.add_argument(
        "--interactive", "-i",
        action="store_true",
        help=(
            "After exporting clips, show a numbered menu so you can listen to each"
            " one and pick which clip to register as the voice reference."
            " Requires --voice-name."
        ),
    )
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    window_len   = args.window_len  or args.length * 4   # e.g. 120s windows for 30s clips
    n_candidates = args.candidates  or args.clips * 2    # e.g. 20 windows for 10 clips

    cache = Path(args.cache)
    out   = Path(args.out)

    # Persist Torch Hub weights across container restarts
    torch_hub_dir = cache / "torch" / "hub"
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub_dir))

    # ── 1. Resolve audio source ────────────────────────────────────────────────
    if args.audio:
        # Local file — skip download entirely.
        audio = Path(args.audio)
        if not audio.exists():
            print(f"ERROR: Audio file not found: {audio}", file=sys.stderr)
            sys.exit(1)
        source_id   = audio.stem
        source_meta: dict = {"type": "file", "path": str(audio.resolve())}
        print(f"==> Using local audio file...")
        print(f"    {audio.name}")
    else:
        print("==> Resolving video ID...")
        video_id = get_video_id(args.url, cookies=args.cookies)
        print(f"    {video_id}")

        print("==> Downloading audio...")
        audio = download_audio(args.url, cache / "downloads", video_id, cookies=args.cookies)
        source_id   = video_id
        source_meta = {"type": "youtube", "url": args.url, "video_id": video_id}

    # ── 1b. Optional timestamp trim ────────────────────────────────────────────
    raw_duration = get_duration(audio)
    t_start = args.start if args.start is not None else 0.0
    t_end   = args.end   if args.end   is not None else raw_duration

    if t_start != 0.0 or t_end != raw_duration:
        if t_start < 0:
            print(f"ERROR: --start {t_start:.1f}s is negative.", file=sys.stderr)
            sys.exit(1)
        if t_end > raw_duration + 0.5:
            print(
                f"ERROR: --end {t_end:.1f}s exceeds audio duration {raw_duration:.1f}s.",
                file=sys.stderr,
            )
            sys.exit(1)
        if t_start >= t_end:
            print(
                f"ERROR: --start ({t_start:.1f}s) must be less than --end ({t_end:.1f}s).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"==> Trimming audio to {t_start:.1f}s \u2013 {t_end:.1f}s...")
        trimmed_id   = f"{source_id}_{int(t_start)}_{int(t_end)}"
        trimmed_path = cache / "downloads" / f"{trimmed_id}{audio.suffix}"
        audio     = trim_audio(audio, t_start, t_end, trimmed_path)
        source_id = trimmed_id
        source_meta = {**source_meta, "start_sec": t_start, "end_sec": t_end}

    duration = get_duration(audio)
    scan_end  = min(duration, args.max_scan_seconds) if args.max_scan_seconds else duration
    print(f"    duration: {duration/60:.1f} min  |  scanning first {scan_end/60:.1f} min")

    # ── 2. Load Silero once for both raw scan and vocal VAD ────────────────────
    print("==> Loading Silero VAD model...")
    vad_model, get_speech_ts = _load_silero()

    # ── 3. Fast VAD on raw audio ───────────────────────────────────────────────
    print("==> Scanning raw audio for speech regions...")
    raw_segs = raw_vad_scan(audio, cache, source_id, args.max_scan_seconds, vad_model, get_speech_ts)
    if not raw_segs:
        print("ERROR: No speech detected in raw audio.", file=sys.stderr)
        sys.exit(1)
    raw_speech_total = sum(e - s for s, e in raw_segs)
    print(f"    {len(raw_segs)} raw segments, {raw_speech_total/60:.1f} min speech "
          f"in {scan_end/60:.1f} min audio")

    # ── 4. Select speech-densest windows ──────────────────────────────────────
    print(f"==> Selecting {n_candidates} candidate windows ({window_len:.0f}s each)...")
    windows = select_windows(raw_segs, n_candidates, window_len, scan_end)
    if not windows:
        print("ERROR: Could not find candidate windows.", file=sys.stderr)
        sys.exit(1)
    demucs_total = sum(e - s for s, e in windows)
    print(f"    {len(windows)} windows → Demucs will process "
          f"{demucs_total/60:.1f} min  (vs {scan_end/60:.1f} min full file)")

    # ── 5 & 6. Per-window: extract → Demucs → vocal VAD (all cached) ──────────
    chunks_dir = cache / "chunks"
    sep_cache  = cache / "separated"
    pool: list[tuple[Path, float, float, float]] = []  # (vocals_path, file_dur, seg_start, seg_end)

    for i, (ws, we) in enumerate(windows, 1):
        chunk_id  = f"{source_id}_{ws:.0f}_{we:.0f}"
        chunk_wav = chunks_dir / f"{chunk_id}.wav"

        print(f"\n  window {i}/{len(windows)}: {ws/60:.1f}–{we/60:.1f} min")

        if chunk_wav.exists():
            print(f"    [cache hit] chunk")
        else:
            print(f"    extracting chunk with ffmpeg...")
            extract_chunk(audio, ws, we - ws, chunk_wav)

        vocals    = separate_vocals_chunk(chunk_wav, sep_cache)
        file_dur  = get_duration(vocals)
        segs      = vad_on_vocals(vocals, cache, vad_model, get_speech_ts)
        print(f"    {len(segs)} vocal speech segments")

        for s, e in segs:
            pool.append((vocals, file_dur, s, e))

    print()
    if not pool:
        print("ERROR: No speech segments found in any candidate window.", file=sys.stderr)
        sys.exit(1)

    total_pool = sum(e - s for _, _, s, e in pool)
    print(f"==> Pool: {len(pool)} vocal segments, {total_pool/60:.1f} min total clean speech")

    # ── 7. Pick and export ─────────────────────────────────────────────────────
    print(f"==> Picking {args.clips} clips (≤ {args.length}s each)...")
    plans = pick_clips(pool, args.clips, args.length)

    # Scope output directory so clips from different voices / sources never
    # collide inside the same --out root.
    #   --voice-name given  →  <out>/<slug>/
    #   standalone run       →  <out>/<source_id>/
    # voice-clone already does the same scoping (out_dir / voice_name), so
    # both steps land in the same per-voice subdirectory.
    if args.voice_name:
        out = out / args.voice_name.strip().lower()
    else:
        out = out / source_id

    print(f"==> Exporting to {out} ...")
    export_clips(plans, out)

    print(f"\nDone — {len(plans)} clips in {out}")
    print(f"Cache:  {cache}  (re-runs are fully cached)")

    # ── optional: register as named voice ────────────────────────────────────────
    if args.voice_name:
        import sys as _sys
        _lib = str(Path(__file__).resolve().parent.parent.parent / "lib")
        if _lib not in _sys.path:
            _sys.path.insert(0, _lib)
        from voices import VoiceRegistry, validate_slug  # type: ignore

        try:
            slug = validate_slug(args.voice_name)
        except ValueError as exc:
            print(f"  WARNING: {exc} — skipping voice registration.", file=sys.stderr)
            slug = None

        if slug:
            # Export a single deterministic best-quality clip for the registry.
            # Picks the pool segment with the longest contiguous speech run so
            # voice-clone's VAD+scoring gets the best possible input.
            print(f"\n==> Selecting best registry clip...")
            src_clip = export_registry_clip(pool, args.length, out)

            if not src_clip:
                # All pool segments were too short (< 5s); do NOT create a
                # partial registry entry that would confuse voice-clone later.
                print(
                    f"  WARNING: voice '{slug}' was NOT registered — no pool segment "
                    f"was long enough (need ≥ 5s of clean solo speech).\n"
                    f"  Suggestions:\n"
                    f"    • try a video with longer continuous speech\n"
                    f"    • increase --window-len or widen --max-scan-seconds\n"
                    f"    • lower --length so the 5s minimum is easier to satisfy",
                    file=sys.stderr,
                )
            else:
                # Optional: let the user listen to all clips and pick a better ref.
                if args.interactive:
                    chosen = _interactive_pick_clip(out)
                    if chosen is not None:
                        src_clip = chosen

                reg = VoiceRegistry(cache)
                reg.create(
                    slug,
                    display_name=slug,
                    source={**source_meta, "n_clips": len(plans)},
                )
                reg.set_source_clip(slug, src_clip)
                print(f"  [registry] voice '{slug}' created")
                print(f"  source clip: {src_clip.name} → {cache}/voices/{slug}/source_clip.wav")
                print(f"  Next steps:")
                print(f"    ./run voice-clone synth --voice {slug} --text 'Hello'")
                print(f"    ./run voice-synth speak --voice {slug} --text 'Hello'")
                print(f"    ./run voice-synth list-voices  # see all registered voices")


if __name__ == "__main__":
    main()
