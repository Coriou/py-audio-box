#!/usr/bin/env python3
"""
voice-register.py — One-shot pipeline: download → split → clone → register.

Chains voice-split and voice-clone synth so that a voice is fully ready for
fast synthesis (./run voice-synth speak) after a single command.

Pipeline steps:
  1. voice-split  — download audio, run Demucs, extract best clip, register to /cache/voices/<slug>/
  2. voice-clone  — normalise ref, VAD-select best segment, transcribe, build clone prompt

Usage:
    ./run voice-register \\
        --url "https://www.youtube.com/watch?v=XXXX" \\
        --voice-name david-attenborough \\
        --text "Nature is the greatest artist."

    # Re-run is safe — all heavy steps are cached; only missing stages are re-run.
    # Use --force to recompute everything from scratch.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n", flush=True)


def _run_step(label: str, script: str, args: list[str]) -> None:
    """Run a pipeline step as a subprocess; abort the whole pipeline on failure."""
    script_path = Path(__file__).resolve().parent.parent / script
    cmd = [sys.executable, str(script_path), *args]

    _banner(label)
    t0 = time.perf_counter()

    result = subprocess.run(cmd)

    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(
            f"\nERROR: step '{label}' failed (exit {result.returncode}).",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    print(f"\n  ✓  {label} done  ({elapsed:.1f}s)", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "One-shot voice registration: download → Demucs split → "
            "clone-prompt build → synthesis test."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required ──────────────────────────────────────────────────────────────
    ap.add_argument(
        "--url",
        required=True,
        help="YouTube (or any yt-dlp) URL to extract the voice from",
    )
    ap.add_argument(
        "--voice-name",
        required=True,
        metavar="SLUG",
        help=(
            "Named voice slug to create/update in /cache/voices/<slug>/. "
            "Lowercase letters, digits, hyphens only (e.g. 'david-attenborough')."
        ),
    )
    ap.add_argument(
        "--text",
        required=True,
        help=(
            "Text to synthesise as a quality check once the voice is ready. "
            "A short sentence works best (e.g. 'Hello, welcome to the show.')."
        ),
    )

    # ── voice-split passthrough ───────────────────────────────────────────────
    ap.add_argument(
        "--clips",
        type=int, default=3,
        help=(
            "Number of random voice clips to export alongside the registry clip. "
            "These land in --out for preview; use 0 to skip random clips "
            "(the deterministic registry clip is always exported)."
        ),
    )
    ap.add_argument(
        "--length",
        type=float, default=30,
        help="Target clip length in seconds (passed to voice-split)",
    )
    ap.add_argument(
        "--cookies",
        default=None, metavar="FILE",
        help="Netscape cookies.txt path for authenticated YouTube downloads",
    )
    ap.add_argument(
        "--max-scan-seconds",
        type=float, default=None, metavar="SECS",
        help="Limit raw VAD scan to the first N seconds (useful for quick tests)",
    )

    # ── voice-clone passthrough ───────────────────────────────────────────────
    ap.add_argument(
        "--language",
        default="Auto",
        help=(
            "Synthesis language for the test synthesis. 'Auto' detects from "
            "--text then from the ref audio. Set explicitly if detection is off "
            "(e.g. 'English', 'French')."
        ),
    )
    ap.add_argument(
        "--tone",
        default=None, metavar="NAME",
        help=(
            "Tone label for the extracted prompt (e.g. 'neutral', 'warm'). "
            "Stored in voice.json['tones'] so voice-synth speak --tone NAME "
            "can select it later."
        ),
    )
    ap.add_argument(
        "--ref-language",
        default="Auto",
        help="Language of the reference audio for whisper transcription",
    )
    ap.add_argument(
        "--whisper-model",
        default="small",
        help="faster-whisper model size for ref transcription ('tiny', 'small', 'medium', …)",
    )
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Qwen3-TTS model repo ID or local path for the clone-synth step",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16", choices=["bfloat16", "float32", "float16"],
        help="Model weight dtype (bfloat16 is ~2x faster on Apple Silicon / AVX-512-BF16)",
    )
    ap.add_argument(
        "--seed",
        type=int, default=None,
        help="Random seed for reproducible synthesis",
    )
    ap.add_argument(
        "--force-bad-ref",
        action="store_true",
        help="Bypass the transcript quality gate (proceed even on low-confidence ref)",
    )

    # ── common ────────────────────────────────────────────────────────────────
    ap.add_argument("--out",   default="/work",  help="Output directory for WAV clips")
    ap.add_argument("--cache", default="/cache", help="Persistent cache directory")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore all cached results and recompute every stage from scratch",
    )
    ap.add_argument(
        "--skip-synth",
        action="store_true",
        help=(
            "Stop after building the clone prompt; do not run the test synthesis. "
            "The voice is ready to use immediately. Useful to save time when you "
            "plan to synthesise with voice-synth speak separately."
        ),
    )

    args = ap.parse_args()

    t_total = time.perf_counter()

    # ── Step 1: voice-split ───────────────────────────────────────────────────
    split_args: list[str] = [
        "--url",        args.url,
        "--voice-name", args.voice_name,
        "--clips",      str(args.clips),
        "--length",     str(args.length),
        "--out",        args.out,
        "--cache",      args.cache,
    ]
    if args.cookies:
        split_args += ["--cookies", args.cookies]
    if args.max_scan_seconds is not None:
        split_args += ["--max-scan-seconds", str(args.max_scan_seconds)]

    _run_step("STEP 1/2 — voice-split (download + Demucs + extract + register)", "voice-split/voice-split.py", split_args)

    # ── Step 2: voice-clone synth ─────────────────────────────────────────────
    clone_args: list[str] = [
        "synth",
        "--voice",  args.voice_name,
        "--text",   args.text,
        "--out",    args.out,
        "--cache",  args.cache,
        "--whisper-model", args.whisper_model,
        "--model",  args.model,
        "--dtype",  args.dtype,
    ]
    if args.language != "Auto":
        clone_args += ["--language", args.language]
    if args.ref_language != "Auto":
        clone_args += ["--ref-language", args.ref_language]
    if args.tone:
        clone_args += ["--tone", args.tone]
    if args.seed is not None:
        clone_args += ["--seed", str(args.seed)]
    if args.force:
        clone_args.append("--force")
    if args.force_bad_ref:
        clone_args.append("--force-bad-ref")

    step2_label = "STEP 2/2 — voice-clone synth (ref processing + prompt build + synthesis)"
    if args.skip_synth:
        # Replace --text with a minimal placeholder; voice-clone still requires
        # --text for synth but we want to confirm the voice is ready, so hand
        # off "." and let the output be discarded.  A cleaner approach is to
        # invoke prepare-ref only and skip synthesis entirely.
        clone_args_no_synth: list[str] = [
            "prepare-ref",
            "--voice",  args.voice_name,
            "--cache",  args.cache,
            "--whisper-model", args.whisper_model,
        ]
        if args.ref_language != "Auto":
            clone_args_no_synth += ["--ref-language", args.ref_language]
        if args.force:
            clone_args_no_synth.append("--force")
        if args.force_bad_ref:
            clone_args_no_synth.append("--force-bad-ref")
        step2_label = "STEP 2/2 — voice-clone prepare-ref (ref processing only; --skip-synth)"
        clone_args = clone_args_no_synth

    _run_step(step2_label, "voice-clone/voice-clone.py", clone_args)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_sec = time.perf_counter() - t_total
    _banner(f"✓  Voice '{args.voice_name}' is ready to use  ({total_sec:.0f}s total)")
    print(f"  Synthesise now:")
    print(f"    ./run voice-synth speak \\")
    print(f"        --voice {args.voice_name} \\")
    print(f"        --text \"Your text here.\"")
    if args.tone:
        print(f"    # or with tone:")
        print(f"    ./run voice-synth speak --voice {args.voice_name} --tone {args.tone} --text \"...\"")
    print(f"\n  Inspect all voices:")
    print(f"    ./run voice-synth list-voices")
    print()


if __name__ == "__main__":
    main()
