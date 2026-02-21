#!/usr/bin/env python3
"""
voice-register.py — One-shot pipeline: source audio → split → clone → register.

Chains voice-split and voice-clone synth so that a voice is fully ready for
fast synthesis (./run voice-synth speak) after a single command.

Pipeline steps:
  1. voice-split  — obtain audio (download or local file), run Demucs, extract
                    best clip, register to /cache/voices/<slug>/
  2. voice-clone  — normalise ref, VAD-select best segment, transcribe, build clone prompt

Usage — YouTube source:
    ./run voice-register \\
        --url "https://www.youtube.com/watch?v=XXXX" \\
        --voice-name david-attenborough \\
        --text "Nature is the greatest artist."

    # Target a specific segment of a long video
    ./run voice-register \\
        --url "https://www.youtube.com/watch?v=XXXX" \\
        --start 1:23 --end 3:45 \\
        --voice-name david-attenborough \\
        --text "Nature is the greatest artist."

Usage — local audio file:
    ./run voice-register \\
        --audio /work/my-recording.wav \\
        --voice-name my-voice \\
        --text "Hello, this is a test."

    # Trim to a specific segment
    ./run voice-register \\
        --audio /work/interview.mp3 \\
        --start 0:45 --end 2:10 \\
        --voice-name interviewee \\
        --text "Hello, this is a test."

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
            "One-shot voice registration: source audio → Demucs split → "
            "clone-prompt build → synthesis test."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── audio source (exactly one required) ──────────────────────────────────
    src_grp = ap.add_mutually_exclusive_group(required=True)
    src_grp.add_argument(
        "--url",
        metavar="URL",
        help="YouTube (or any yt-dlp-supported) URL to extract the voice from",
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
        default=None, metavar="TIMESTAMP",
        help=(
            "Trim start time in the source audio. "
            "Accepts: seconds (90 / 1:30.5), MM:SS (1:30), HH:MM:SS (1:02:30). "
            "Only audio from this point onwards is processed."
        ),
    )
    ap.add_argument(
        "--end",
        default=None, metavar="TIMESTAMP",
        help=(
            "Trim end time in the source audio. Same formats as --start. "
            "Only audio up to this point is processed."
        ),
    )
    # ── required identity ────────────────────────────────────────────────────
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
        "--ref-text",
        default=None, metavar="TEXT",
        help=(
            "Exact transcript of the reference audio clip. "
            "When supplied, the final Whisper transcript+quality gate step is skipped "
            "and this text is used for prompt build."
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
        default="auto", choices=["auto", "bfloat16", "float32", "float16"],
        help=(
            "Model weight dtype passed to voice-clone.  auto (default): float32 on CPU; "
            "bfloat16 on CUDA Ampere+; float16 on older CUDA GPUs (Maxwell / Pascal / Volta / Turing)."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int, default=None,
        help="Random seed for reproducible synthesis",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int, default=None, dest="max_new_tokens",
        help=(
            "Hard cap on generated audio tokens (passed to voice-clone synth). "
            "Defaults to an estimate based on --text length."
        ),
    )
    ap.add_argument(
        "--timeout",
        type=float, default=None,
        help=(
            "Wall-clock timeout in seconds for the synthesis step "
            "(passed to voice-clone synth). "
            "Defaults to 4× the CPU ETA estimate. Pass 0 to disable."
        ),
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
    ap.add_argument(
        "--interactive", "-i",
        action="store_true",
        help=(
            "Enable interactive selection at both pipeline steps:\n"
            "  1. voice-split: pick the best extracted clip before registering.\n"
            "  2. voice-clone: pick the best VAD sub-segment before building the prompt.\n"
            "Run from a real terminal (not a batch script) so you can listen to the\n"
            "candidate files and type a number at each prompt."
        ),
    )

    args = ap.parse_args()

    t_total = time.perf_counter()

    # ── Step 1: voice-split ───────────────────────────────────────────────────
    split_args: list[str] = [
        "--voice-name", args.voice_name,
        "--clips",      str(args.clips),
        "--length",     str(args.length),
        "--out",        args.out,
        "--cache",      args.cache,
    ]
    if args.url:
        split_args += ["--url", args.url]
    else:
        split_args += ["--audio", args.audio]
    if args.start is not None:
        split_args += ["--start", args.start]
    if args.end is not None:
        split_args += ["--end", args.end]
    if args.cookies:
        split_args += ["--cookies", args.cookies]
    if args.max_scan_seconds is not None:
        split_args += ["--max-scan-seconds", str(args.max_scan_seconds)]
    if args.interactive:
        split_args.append("--interactive")

    step1_source = "download + Demucs" if args.url else "Demucs"
    _run_step(
        f"STEP 1/2 — voice-split ({step1_source} + extract + register)",
        "voice-split/voice-split.py",
        split_args,
    )

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
    if args.ref_text:
        clone_args += ["--ref-text", args.ref_text]
    if args.language != "Auto":
        clone_args += ["--language", args.language]
    if args.ref_language != "Auto":
        clone_args += ["--ref-language", args.ref_language]
    if args.tone:
        clone_args += ["--tone", args.tone]
    if args.seed is not None:
        clone_args += ["--seed", str(args.seed)]
    if args.max_new_tokens is not None:
        clone_args += ["--max-new-tokens", str(args.max_new_tokens)]
    if args.timeout is not None:
        clone_args += ["--timeout", str(args.timeout)]
    if args.force:
        clone_args.append("--force")
    if args.force_bad_ref:
        clone_args.append("--force-bad-ref")
    if args.interactive:
        clone_args.append("--interactive")

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
        if args.ref_text:
            clone_args_no_synth += ["--ref-text", args.ref_text]
        if args.ref_language != "Auto":
            clone_args_no_synth += ["--ref-language", args.ref_language]
        if args.force:
            clone_args_no_synth.append("--force")
        if args.force_bad_ref:
            clone_args_no_synth.append("--force-bad-ref")
        if args.interactive:
            clone_args_no_synth.append("--interactive")
        step2_label = "STEP 2/2 — voice-clone prepare-ref (ref processing only; --skip-synth)"
        clone_args = clone_args_no_synth

    _run_step(step2_label, "voice-clone/voice-clone.py", clone_args)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_sec = time.perf_counter() - t_total
    _banner(f"✓  Voice '{args.voice_name}' is ready to use  ({total_sec:.0f}s total)")
    print("  Synthesise now:")
    print("    ./run voice-synth speak \\")
    print(f"        --voice {args.voice_name} \\")
    print("        --text \"Your text here.\"")
    if args.tone:
        print("    # or with tone:")
        print(f"    ./run voice-synth speak --voice {args.voice_name} --tone {args.tone} --text \"...\"")
    print("\n  Inspect all voices:")
    print("    ./run voice-synth list-voices")
    print()


if __name__ == "__main__":
    main()
