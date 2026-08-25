#!/usr/bin/env python3
"""Build the fal-sfx-bench manifest.json from prepared inputs.

Reads `inputs-index.json` produced by prepare_inputs.py and emits a manifest
with 30 cases (6 models x 5 clips) named "<model_key>__<clip_key>". Each case
carries the canonical clip prompt, the model-specific request payload (minus
`video_url`, which the runner attaches at submit time), and an estimated cost.

Costs come from a local price table that mirrors the spec registry; it is
duplicated by design here (models.py owns the authoritative copy) and must be
reconciled against it. Estimates are advisory regardless — reconcile.py is the
source of truth once the Fal usage CSV lands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SEED = 20260825

MODEL_KEYS = [
    "mirelo16",
    "kling-v2a",
    "sonilo11",
    "hunyuan",
    "cassette-v2s",
    "cassette-t2s",
]

CLIP_KEYS = ["walk1", "walk2", "man", "ocean-long", "ocean-short"]

# Canonical prompts, identical across models; per-model key mapping happens in
# build_arguments. Strings pinned by the spec / prior hunyuan-eval manifest.
PROMPT_WALK1 = (
    "A man hiking on a paved mountain road, synchronized shoe footsteps at "
    "every visible foot contact, subtle mountain wind and natural outdoor "
    "ambience, no speech, no music"
)
PROMPT_WALK2 = (
    "A barefoot man walking softly on artificial grass outdoors, synchronized "
    "soft footfalls at every visible foot contact, gentle outdoor ambience, "
    "no speech, no music"
)
PROMPT_MAN = (
    "A man walking on a paved mountain overlook, synchronized shoe footsteps "
    "at every visible foot contact, subtle mountain wind and natural outdoor "
    "ambience, no speech, no music"
)
PROMPT_OCEAN = (
    "Natural ocean waves surging and crashing onto a rocky shore, detailed "
    "foamy surf and water movement, no speech, no music"
)

CLIP_PROMPTS = {
    "walk1": PROMPT_WALK1,
    "walk2": PROMPT_WALK2,
    "man": PROMPT_MAN,
    "ocean-long": PROMPT_OCEAN,
    "ocean-short": PROMPT_OCEAN,
}

HUNYUAN_NEGATIVE_PROMPT = "speech, voices, music, noisy, harsh"

# Duplicated by design from the spec's model registry (models.py is
# authoritative); reconcile via models.py later.
PRICE_TABLE: dict[str, dict[str, Any]] = {
    "mirelo16": {"unit": "per_second", "per_second": 0.01, "per_sample_multiplier": True},
    "kling-v2a": {"unit": "flat_per_request", "flat": 0.035},
    "sonilo11": {"unit": "per_second_of_input", "per_second": 0.009},
    "hunyuan": {"unit": "per_second_of_input", "per_second": 0.01},
    "cassette-v2s": {"unit": "per_minute_of_input", "per_minute": 0.2},
    "cassette-t2s": {"unit": "flat_per_request", "flat": 0.01},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_arguments(
    model_key: str,
    prompt: str,
    input_duration_s: float,
    seed: int,
) -> dict[str, Any]:
    """Final request payload minus video_url (runner attaches it at submit)."""
    if model_key == "mirelo16":
        # duration schema 1-60; >10 uses sliding window. num_samples default
        # is 2 — MUST set 1.
        duration = min(math.ceil(input_duration_s), 60)
        return {
            "text_prompt": prompt,
            "duration": duration,
            "num_samples": 1,
            "seed": seed,
        }
    if model_key == "kling-v2a":
        # background_music_prompt schema default is "intense car race" — MUST
        # override explicitly.
        return {
            "sound_effect_prompt": prompt[:200],
            "background_music_prompt": "",
            "asmr_mode": False,
        }
    if model_key == "sonilo11":
        return {"prompt": prompt, "audio_format": "wav"}
    if model_key == "hunyuan":
        return {
            "text_prompt": prompt,
            "negative_prompt": HUNYUAN_NEGATIVE_PROMPT,
            "guidance_scale": 4.5,
            "num_inference_steps": 50,
            "seed": seed,
        }
    if model_key == "cassette-v2s":
        # Schema has NO prompt field; Gemma3 auto-analyzes the video.
        return {}
    if model_key == "cassette-t2s":
        # Text-only: duration schema 1-30.
        duration = max(1, min(30, math.ceil(input_duration_s)))
        return {"prompt": prompt, "duration": duration}
    raise KeyError(f"unknown model key: {model_key}")


def estimate_cost_usd(
    model_key: str,
    arguments: dict[str, Any],
    input_duration_s: float,
) -> float:
    """Advisory estimate from the local price table (see module docstring)."""
    price = PRICE_TABLE[model_key]
    if price["unit"] == "flat_per_request":
        return float(price["flat"])
    if price["unit"] == "per_second":
        # mirelo16 bills per second of requested output x samples.
        seconds = float(arguments.get("duration") or input_duration_s)
        samples = arguments.get("num_samples", 1) if price["per_sample_multiplier"] else 1
        return price["per_second"] * seconds * samples
    if price["unit"] == "per_second_of_input":
        return price["per_second"] * input_duration_s
    if price["unit"] == "per_minute_of_input":
        return price["per_minute"] / 60.0 * input_duration_s
    raise KeyError(f"unknown price unit for {model_key}: {price['unit']}")


def build_case(
    model_key: str,
    entry: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    clip_key = entry["key"]
    duration_s = entry["duration_s"]
    if duration_s is None:
        raise ValueError(f"inputs-index.json has no duration for clip {clip_key!r}")
    prompt = CLIP_PROMPTS[clip_key]
    arguments = build_arguments(model_key, prompt, duration_s, seed)
    return {
        "name": f"{model_key}__{clip_key}",
        "model": model_key,
        "input": str(Path(entry["output_path"])),
        "input_sha256": entry["sha256"],
        "input_duration_s": duration_s,
        "prompt": prompt,
        "arguments": arguments,
        "estimated_cost_usd": round(estimate_cost_usd(model_key, arguments, duration_s), 6),
    }


def load_inputs_index(inputs_dir: Path) -> list[dict[str, Any]]:
    index_path = inputs_dir / "inputs-index.json"
    if not index_path.exists():
        print(f"missing {index_path}; run prepare_inputs.py first", file=sys.stderr)
        raise SystemExit(2)
    with index_path.open() as source:
        index = json.load(source)
    entries = {entry["key"]: entry for entry in index["clips"]}
    missing = [key for key in CLIP_KEYS if key not in entries]
    if missing:
        print(
            "inputs-index.json is missing clip(s): " + ", ".join(missing),
            file=sys.stderr,
        )
        raise SystemExit(2)
    for key in CLIP_KEYS:
        if not Path(entries[key]["output_path"]).exists():
            print(f"indexed input missing on disk: {entries[key]['output_path']}", file=sys.stderr)
            raise SystemExit(2)
    return [entries[key] for key in CLIP_KEYS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit manifest.json (30 cases) for fal-sfx-bench."
    )
    parser.add_argument("--run-dir", required=True, help="run directory (created if absent)")
    parser.add_argument("--inputs-dir", default=None, help="directory holding inputs-index.json (default: <run-dir>/inputs)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = (
        Path(args.inputs_dir).expanduser().resolve()
        if args.inputs_dir is not None
        else run_dir / "inputs"
    )

    entries = load_inputs_index(inputs_dir)
    cases = [build_case(model_key, entry, args.seed) for model_key in MODEL_KEYS for entry in entries]
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "seed": args.seed,
        "cases": cases,
    }
    manifest_path = run_dir / "manifest.json"
    atomic_json(manifest_path, manifest)

    total_cost = sum(case["estimated_cost_usd"] for case in cases)
    for model_key in MODEL_KEYS:
        model_cases = [case for case in cases if case["model"] == model_key]
        model_total = sum(case["estimated_cost_usd"] for case in model_cases)
        print(f"[{model_key}] {len(model_cases)} case(s), est ${model_total:.4f}")
    print(f"wrote {manifest_path} ({len(cases)} cases)")
    print(f"total estimated cost: ${total_cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
