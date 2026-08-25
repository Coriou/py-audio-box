#!/usr/bin/env python3
"""Model registry for fal-sfx-bench.

Every benchmarked fal.ai endpoint is described by a frozen ModelSpec: how to
build its request payload from a manifest case, how to read every output
artifact back out of the raw response, and the published pricing used for
advisory cost estimates. Registry contents were verified against the fal.ai
model pages and OpenAPI schemas on 2026-08-25 (schema_checked_at); the notes
field records per-model quirks (prompt key names, schema defaults that must be
overridden, caps).

No network calls happen at import time; FAL_KEY is never touched here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_SEED = 20260825
SCHEMA_CHECKED_AT = "2026-08-25"
FAL_MODELS_BASE_URL = "https://fal.ai/models/"

#: Kling's sound_effect_prompt is hard-capped at 200 characters by the schema.
KLING_PROMPT_CHARACTERS = 200

#: One output file returned by a model:
#: {"kind": "audio"|"video", "url": str, "content_type": str|None, "file_name": str|None}
OutputArtifact = dict[str, Any]


@dataclass(frozen=True)
class PriceSpec:
    """Published pricing for one endpoint.

    Estimates are advisory; reconcile.py against the Fal usage CSV is the
    source of truth once billing data lands.
    """

    unit: str  # "per_second_of_output" | "flat_per_request"
    per_second: float | None
    flat: float | None
    per_sample_multiplier: bool
    published_text: str
    source_url: str


@dataclass(frozen=True)
class ModelSpec:
    """Everything the runner needs to drive one fal.ai endpoint."""

    key: str  # short id used in case names, e.g. "mirelo16"
    endpoint_id: str  # e.g. "mirelo-ai/sfx1.6/video-to-video"
    label: str  # human name, e.g. "Mirelo SFX 1.6"
    vendor: str
    build_arguments: Callable[[dict[str, Any]], dict[str, Any]]
    extract_outputs: Callable[[dict[str, Any]], list[OutputArtifact]]
    price: PriceSpec
    docs_urls: list[str]
    schema_checked_at: str
    notes: str
    #: Key under which the uploaded input video URL is injected at submit time,
    #: or None for text-only endpoints that take no media input.
    media_key: str | None = "video_url"


def _artifact(
    kind: str,
    url: str,
    content_type: str | None = None,
    file_name: str | None = None,
) -> OutputArtifact:
    return {"kind": kind, "url": url, "content_type": content_type, "file_name": file_name}


def _entry_artifact(kind: str, entry: Any) -> OutputArtifact | None:
    """Normalize one response entry ({url, content_type, file_name, ...})."""
    if not isinstance(entry, dict):
        return None
    url = entry.get("url")
    if not url:
        return None
    return _artifact(kind, str(url), entry.get("content_type"), entry.get("file_name"))


def _dedupe(artifacts: list[OutputArtifact]) -> list[OutputArtifact]:
    """Drop repeated URLs (e.g. sonilo's audio mirroring audios[0])."""
    seen: set[str] = set()
    unique: list[OutputArtifact] = []
    for artifact in artifacts:
        if artifact["url"] in seen:
            continue
        seen.add(artifact["url"])
        unique.append(artifact)
    return unique


def _single_output_extractor(result_key: str, kind: str) -> Callable[[dict[str, Any]], list[OutputArtifact]]:
    """Extractor for responses shaped {result_key: {url, ...}}."""

    def extract_outputs(result: dict[str, Any]) -> list[OutputArtifact]:
        artifact = _entry_artifact(kind, result.get(result_key))
        return [artifact] if artifact else []

    return extract_outputs


def _seed(case: dict[str, Any]) -> int:
    return int(case.get("seed", DEFAULT_SEED))


def _clamped_ceil_seconds(value: Any, low: int, high: int) -> int:
    return max(low, min(high, int(math.ceil(float(value)))))


# ---------------------------------------------------------------------------
# Per-model argument builders (final payload minus the uploaded media URL).
# ---------------------------------------------------------------------------


def _mirelo_build_arguments(case: dict[str, Any]) -> dict[str, Any]:
    return {
        # NOTE: text_prompt, not prompt.
        "text_prompt": case["prompt"],
        # Schema 1-60 s, default 10; >10 s inputs use sliding-window analysis.
        "duration": _clamped_ceil_seconds(case["input_duration_s"], 1, 60),
        # Schema default is 2 samples — MUST be forced to 1.
        "num_samples": 1,
        "seed": _seed(case),
    }


def _kling_build_arguments(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "sound_effect_prompt": str(case["prompt"])[:KLING_PROMPT_CHARACTERS],
        # Schema default is "intense car race" — MUST override explicitly.
        "background_music_prompt": "",
        "asmr_mode": False,
    }


def _sonilo_build_arguments(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": case["prompt"],
        "audio_format": "wav",
    }


def _hunyuan_build_arguments(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "text_prompt": case["prompt"],
        "negative_prompt": "speech, voices, music, noisy, harsh",
        "guidance_scale": 4.5,
        "num_inference_steps": 50,
        "seed": _seed(case),
    }


def _cassette_v2s_build_arguments(case: dict[str, Any]) -> dict[str, Any]:
    # Schema has NO prompt field; Gemma3 auto-analyzes the input video.
    return {}


def _cassette_t2s_build_arguments(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": case["prompt"],
        # Schema 1-30 s, integer seconds.
        "duration": _clamped_ceil_seconds(case["input_duration_s"], 1, 30),
    }


# ---------------------------------------------------------------------------
# Per-model output extractors (must return EVERY artifact the model returned).
# ---------------------------------------------------------------------------


def _mirelo_extract_outputs(result: dict[str, Any]) -> list[OutputArtifact]:
    entries = result.get("video") or []
    if isinstance(entries, dict):
        entries = [entries]
    artifacts = [_entry_artifact("video", entry) for entry in entries]
    return _dedupe([artifact for artifact in artifacts if artifact])


def _kling_extract_outputs(result: dict[str, Any]) -> list[OutputArtifact]:
    artifacts = [
        _entry_artifact("video", result.get("video")),
        _entry_artifact("audio", result.get("audio")),
    ]
    return _dedupe([artifact for artifact in artifacts if artifact])


def _sonilo_extract_outputs(result: dict[str, Any]) -> list[OutputArtifact]:
    audios = result.get("audios") or []
    if isinstance(audios, dict):
        audios = [audios]
    artifacts = [_entry_artifact("audio", result.get("audio"))]
    artifacts.extend(_entry_artifact("audio", entry) for entry in audios)
    return _dedupe([artifact for artifact in artifacts if artifact])


_extract_single_video = _single_output_extractor("video", "video")
_extract_audio_file = _single_output_extractor("audio_file", "audio")


def estimate_cost(
    spec: ModelSpec | PriceSpec,
    arguments: dict[str, Any] | None,
    output_duration_s: float | None,
) -> float:
    """Advisory cost estimate in USD for one request."""
    price: PriceSpec = getattr(spec, "price", spec)
    if price.unit == "flat_per_request":
        return round(price.flat or 0.0, 6)
    cost = (price.per_second or 0.0) * (float(output_duration_s or 0.0))
    if price.per_sample_multiplier and arguments:
        samples = arguments.get("num_samples")
        if isinstance(samples, (int, float)) and samples > 0:
            cost *= samples
    return round(cost, 6)


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(
            f"unknown model key {key!r}; known keys: {', '.join(sorted(MODELS))}"
        ) from None


MODELS: dict[str, ModelSpec] = {
    "mirelo16": ModelSpec(
        key="mirelo16",
        endpoint_id="mirelo-ai/sfx1.6/video-to-video",
        label="Mirelo SFX 1.6",
        vendor="mirelo-ai",
        build_arguments=_mirelo_build_arguments,
        extract_outputs=_mirelo_extract_outputs,
        price=PriceSpec(
            unit="per_second_of_output",
            per_second=0.01,
            flat=None,
            per_sample_multiplier=True,
            published_text="$0.01 per second x duration x num_samples",
            source_url=FAL_MODELS_BASE_URL + "mirelo-ai/sfx1.6/video-to-video",
        ),
        docs_urls=[FAL_MODELS_BASE_URL + "mirelo-ai/sfx1.6/video-to-video"],
        schema_checked_at=SCHEMA_CHECKED_AT,
        notes=(
            "Prompt key is text_prompt (not prompt). duration=ceil(input "
            "seconds), schema 1-60 (default 10; inputs >10 s use sliding "
            "window). num_samples schema default is 2 — forced to 1. Returns "
            '{"video": [entry, ...]} with the soundtrack baked into the mp4. '
            "Billing verified 2026-08-25 export: billed seconds = ceil per "
            "request (a 10.01 s clip is billed as 11 s)."
        ),
    ),
    "kling-v2a": ModelSpec(
        key="kling-v2a",
        endpoint_id="fal-ai/kling-video/video-to-audio",
        label="Kling Video-to-Audio",
        vendor="fal-ai",
        build_arguments=_kling_build_arguments,
        extract_outputs=_kling_extract_outputs,
        price=PriceSpec(
            unit="flat_per_request",
            per_second=None,
            flat=0.035,
            per_sample_multiplier=False,
            published_text="$0.035 per video",
            source_url=FAL_MODELS_BASE_URL + "fal-ai/kling-video/video-to-audio",
        ),
        docs_urls=[FAL_MODELS_BASE_URL + "fal-ai/kling-video/video-to-audio"],
        schema_checked_at=SCHEMA_CHECKED_AT,
        notes=(
            "Prompt key is sound_effect_prompt, truncated to 200 chars. "
            'background_music_prompt schema default is "intense car race" — '
            "overridden to empty string; asmr_mode=false. Input "
            "constraints: .mp4/.mov only, <=100 MB, duration 3.0-20.0 s. "
            'Returns {"video": dubbed mp4, "audio": mp3}; flat price per '
            "video. Billing verified 2026-08-25 export: exact flat per "
            "video, no extras."
        ),
    ),
    "sonilo11": ModelSpec(
        key="sonilo11",
        endpoint_id="sonilo/v1.1/video-to-sound-effects",
        label="Sonilo v1.1",
        vendor="sonilo",
        build_arguments=_sonilo_build_arguments,
        extract_outputs=_sonilo_extract_outputs,
        price=PriceSpec(
            unit="per_second_of_output",
            per_second=0.009,
            flat=None,
            per_sample_multiplier=False,
            published_text="$0.009 per second of output and per sample",
            source_url=FAL_MODELS_BASE_URL + "sonilo/v1.1/video-to-sound-effects",
        ),
        docs_urls=[FAL_MODELS_BASE_URL + "sonilo/v1.1/video-to-sound-effects"],
        schema_checked_at=SCHEMA_CHECKED_AT,
        notes=(
            "Prompt key is prompt; audio_format forced to wav. Returns "
            '{"audio": entry, "audios": [entries]} (audio files only); '
            "duplicate URLs are deduplicated on extraction. Billing verified "
            "2026-08-25 export: output seconds are ceiled per request (a "
            "9.4 s clip is billed as 10 s)."
        ),
    ),
    "hunyuan": ModelSpec(
        key="hunyuan",
        endpoint_id="fal-ai/hunyuan-video-foley",
        label="HunyuanVideo-Foley (baseline)",
        vendor="fal-ai",
        build_arguments=_hunyuan_build_arguments,
        extract_outputs=_extract_single_video,
        price=PriceSpec(
            unit="per_second_of_output",
            per_second=0.01,
            flat=None,
            per_sample_multiplier=False,
            published_text="$0.1 per 10 seconds",
            source_url=FAL_MODELS_BASE_URL + "fal-ai/hunyuan-video-foley",
        ),
        docs_urls=[FAL_MODELS_BASE_URL + "fal-ai/hunyuan-video-foley"],
        schema_checked_at=SCHEMA_CHECKED_AT,
        notes=(
            "Baseline carried over from fal-hunyuan-foley-eval. Takes the "
            "input as video_url plus text_prompt; negative_prompt/guidance_"
            "scale/num_inference_steps pinned to speech-free defaults "
            '(4.5 / 50 steps / negative "speech, voices, music, noisy, '
            'harsh"). Returns {"video": muxed mp4}. Billing verified '
            "2026-08-25 export: fractional units (seconds/10), no rounding."
        ),
    ),
    "cassette-v2s": ModelSpec(
        key="cassette-v2s",
        endpoint_id="cassetteai/video-sound-effects-generator",
        label="CassetteAI Video SFX",
        vendor="cassetteai",
        build_arguments=_cassette_v2s_build_arguments,
        extract_outputs=_extract_single_video,
        price=PriceSpec(
            unit="per_second_of_output",
            per_second=0.2 / 60,
            flat=None,
            per_sample_multiplier=False,
            published_text="$0.2 per minute of video processed",
            source_url=FAL_MODELS_BASE_URL + "cassetteai/video-sound-effects-generator",
        ),
        docs_urls=[FAL_MODELS_BASE_URL + "cassetteai/video-sound-effects-generator"],
        schema_checked_at=SCHEMA_CHECKED_AT,
        notes=(
            "Schema has NO prompt field — video_url is the only argument; "
            "Gemma3 auto-analyzes the video and overlays generated SFX. "
            'Returns {"video": mp4 with baked-in SFX track}. Billing '
            "verified 2026-08-25 export: video time is ceiled to whole "
            "minutes per request — a 10 s clip costs $0.20 (6.3x the naive "
            "per-second estimate), the priciest option at short clip lengths."
        ),
    ),
    "cassette-t2s": ModelSpec(
        key="cassette-t2s",
        endpoint_id="cassetteai/sound-effects-generator",
        label="CassetteAI Text-SFX (no video cond.)",
        vendor="cassetteai",
        build_arguments=_cassette_t2s_build_arguments,
        extract_outputs=_extract_audio_file,
        price=PriceSpec(
            unit="flat_per_request",
            per_second=None,
            flat=0.01,
            per_sample_multiplier=False,
            published_text="$0.01 per generation",
            source_url=FAL_MODELS_BASE_URL + "cassetteai/sound-effects-generator",
        ),
        docs_urls=[FAL_MODELS_BASE_URL + "cassetteai/sound-effects-generator"],
        schema_checked_at=SCHEMA_CHECKED_AT,
        notes=(
            "Text-only control: takes prompt plus integer duration (schema "
            "1-30), NO video conditioning — the manifest input exists for "
            "duration and the review mux only (media_key=None). Returns "
            '{"audio_file": wav}. Billing verified 2026-08-25 export: exact '
            "flat per generation."
        ),
        media_key=None,
    ),
}
