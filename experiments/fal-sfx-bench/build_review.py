#!/usr/bin/env python3
"""Build the loudness-normalized review set and self-contained review page.

For every completed case record: extract the model output audio (from the
primary artifact — video-with-audio when the model returned one, else the
audio file), apply two-pass ``ffmpeg loudnorm I=-16:TP=-1.5:LRA=11`` to
48 kHz stereo AAC, then mux the normalized track onto the ORIGINAL benchmark
input video stream with ``-c:v copy`` so every model is reviewed against
identical frames. Writes ``review/<case>.mp4``, ``review/<case>.loudness.json``
and a self-contained ``review/index.html`` (inline CSS, relative sources,
one section per clip with every model side by side).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET_I = -16.0
TARGET_TP = -1.5
TARGET_LRA = 11.0
AUDIO_RATE = 48_000
AUDIO_CHANNELS = 2
AUDIO_BITRATE = "192k"

_LOUDNESS_JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_ffmpeg(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    command = ["ffmpeg", "-hide_banner", "-nostdin", *arguments]
    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if process.returncode != 0:
        tail = "\n".join(process.stderr.strip().splitlines()[-12:])
        raise RuntimeError(
            f"ffmpeg failed ({process.returncode}): {' '.join(command)}\n{tail}"
        )
    return process


def resolve_run_layout(run_dir: Path) -> tuple[str, Path, Path]:
    """Return ``(run_id, records_dir, review_dir)`` for a run directory."""
    run_id = run_dir.name
    records_dir = run_dir / "runs" / run_id / "records"
    if records_dir.is_dir():
        return run_id, records_dir, records_dir.parent / "review"
    nested = sorted((run_dir / "runs").glob("*/records"))
    if nested:
        found = nested[0].parent
        print(
            f"build_review: directory name does not match run id, "
            f"using run '{found.name}'",
            flush=True,
        )
        return found.name, nested[0], found / "review"
    raise FileNotFoundError(f"no records directory under {run_dir}")


def load_record(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"build_review: skipping unreadable record {path.name}: {exc}",
              file=sys.stderr, flush=True)
        return None


def is_completed(record: dict[str, Any]) -> bool:
    return not record.get("error") and record.get("status") != "failed"


def parse_case_name(name: str) -> tuple[str, str]:
    """Split ``<model>__<clip>`` case names on the first separator."""
    model, separator, clip = name.partition("__")
    if not separator:
        return "", name
    return model, clip


def primary_artifact(record: dict[str, Any]) -> tuple[Path | None, str | None]:
    """Pick the review artifact: video-with-audio when present, else audio."""
    outputs = record.get("outputs") or []
    primary = record.get("primary_output")

    def kind_for(local_path: str) -> str | None:
        for artifact in outputs:
            if artifact.get("local_path") == local_path:
                return artifact.get("kind")
        return None

    if primary:
        return Path(primary), kind_for(primary)
    for kind in ("video", "audio"):
        for artifact in outputs:
            if artifact.get("kind") == kind and artifact.get("local_path"):
                return Path(artifact["local_path"]), kind
    return None, None


def coerce_metric(value: Any) -> float | None:
    """Parse loudnorm readings; ``-inf``/``nan`` style strings become None."""
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def measure_loudness(source: Path) -> dict[str, Any]:
    """loudnorm pass 1: print_format=json measurement of the source audio."""
    process = run_ffmpeg([
        "-i", str(source),
        "-vn",
        "-af",
        f"loudnorm=I={TARGET_I:g}:TP={TARGET_TP:g}:LRA={TARGET_LRA:g}"
        ":print_format=json",
        "-f", "null", "-",
    ])
    matches = _LOUDNESS_JSON_RE.findall(process.stderr)
    if not matches:
        raise RuntimeError(f"loudnorm produced no measurements for {source}")
    return json.loads(matches[-1])


def normalize_audio(source: Path, destination: Path,
                    measured: dict[str, Any]) -> bool:
    """loudnorm pass 2 into 48 kHz stereo AAC. Returns True for linear mode."""
    chain = (f"loudnorm=I={TARGET_I:g}:TP={TARGET_TP:g}:LRA={TARGET_LRA:g}")
    linear_values = {
        "measured_I": coerce_metric(measured.get("input_i")),
        "measured_TP": coerce_metric(measured.get("input_tp")),
        "measured_LRA": coerce_metric(measured.get("input_lra")),
        "measured_thresh": coerce_metric(measured.get("input_thresh")),
        "offset": coerce_metric(measured.get("target_offset")),
    }
    linear = all(value is not None for value in linear_values.values())
    if linear:
        chain += ":linear=true" + "".join(
            f":{key}={value:.2f}" for key, value in linear_values.items()
        )
    else:
        print(
            f"build_review: non-finite pass-1 measurements for {source.name}; "
            "falling back to dynamic normalization",
            flush=True,
        )
    run_ffmpeg([
        "-y",
        "-i", str(source),
        "-vn",
        "-af", chain,
        "-ac", str(AUDIO_CHANNELS),
        "-ar", str(AUDIO_RATE),
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-f", "mp4",
        str(destination),
    ])
    return linear


def mux_review(original_video: Path, normalized_audio: Path,
               destination: Path) -> None:
    """Overlay the normalized track on the original input frames (lossless).

    Streams are kept at full length on purpose: a model returning audio
    shorter/longer than the input shows up as a truncated/frozen tail in the
    review player instead of being silently cut.
    """
    run_ffmpeg([
        "-y",
        "-i", str(original_video),
        "-i", str(normalized_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(destination),
    ])


def build_case(case: str, record: dict[str, Any], review_dir: Path,
               force: bool) -> str:
    """Produce review/<case>.mp4 + loudness.json. Returns built|skipped."""
    review_mp4 = review_dir / f"{case}.mp4"
    loudness_path = review_dir / f"{case}.loudness.json"
    if not force and review_mp4.is_file() and loudness_path.is_file():
        print(f"{case}: review artifacts exist, skipping (use --force)",
              flush=True)
        return "skipped"

    source, kind = primary_artifact(record)
    if source is None:
        raise RuntimeError("record has no usable output artifact")
    if not source.is_file():
        raise RuntimeError(f"primary artifact missing on disk: {source}")
    original = record.get("input")
    if not original or not Path(original).is_file():
        raise RuntimeError(f"original benchmark input missing: {original}")

    print(f"{case}: loudnorm pass 1 on {kind or 'media'} "
          f"{source.name}", flush=True)
    measured = measure_loudness(source)

    print(f"{case}: loudnorm pass 2 -> 48 kHz stereo AAC", flush=True)
    normalized_part = review_dir / f"{case}.norm.part"
    linear = normalize_audio(source, normalized_part, measured)

    print(f"{case}: muxing normalized audio onto original input frames",
          flush=True)
    review_part = review_dir / (review_mp4.name + ".part")
    try:
        mux_review(Path(original), normalized_part, review_part)
        review_part.replace(review_mp4)
    finally:
        normalized_part.unlink(missing_ok=True)
        review_part.unlink(missing_ok=True)

    out_metrics: dict[str, Any] = {
        "out_i": coerce_metric(measured.get("output_i")),
        "out_tp": coerce_metric(measured.get("output_tp")),
        "out_lra": coerce_metric(measured.get("output_lra")),
        "out_source": "predicted",
    }
    try:
        verified = measure_loudness(review_mp4)
        out_metrics = {
            "out_i": coerce_metric(verified.get("input_i")),
            "out_tp": coerce_metric(verified.get("input_tp")),
            "out_lra": coerce_metric(verified.get("input_lra")),
            "out_source": "measured",
        }
    except RuntimeError as exc:
        print(f"{case}: post-mux loudness verification failed ({exc}); "
              "storing predicted values", file=sys.stderr, flush=True)

    payload = {
        "case": case,
        "model": record.get("model"),
        "clip": parse_case_name(case)[1],
        "generated_at": utc_now(),
        "source": {"path": str(source), "kind": kind},
        "original_input": str(original),
        "targets": {"I": TARGET_I, "TP": TARGET_TP, "LRA": TARGET_LRA},
        "input_i": coerce_metric(measured.get("input_i")),
        "input_tp": coerce_metric(measured.get("input_tp")),
        "input_lra": coerce_metric(measured.get("input_lra")),
        "input_thresh": coerce_metric(measured.get("input_thresh")),
        "target_offset": coerce_metric(measured.get("target_offset")),
        "linear_normalization": linear,
        **out_metrics,
    }
    atomic_json(loudness_path, payload)
    return "built"


_CSS = """
:root { color-scheme: dark; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0 auto; padding: 1.5rem; max-width: 1440px;
       background: #101014; color: #e8e8ea; }
h1 { font-size: 1.35rem; margin: .2rem 0 .3rem; }
h2 { border-bottom: 1px solid #2a2a32; padding-bottom: .35rem;
     margin-top: 2.5rem; font-size: 1.1rem; }
.meta { color: #9a9aa5; font-size: .85rem; margin-top: 0; }
.row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: stretch; }
.card { flex: 1 1 300px; max-width: 430px; background: #18181f;
        border: 1px solid #26262e; border-radius: 10px; padding: .75rem; }
.card h3 { margin: .1rem 0 .5rem; font-size: .95rem; color: #cfcfe0; }
video { width: 100%; border-radius: 6px; background: #000; min-height: 120px; }
table.stats { width: 100%; border-collapse: collapse; margin-top: .5rem;
              font-size: .78rem; }
table.stats th { text-align: left; color: #8b8b98; font-weight: 500;
                 padding: .18rem .4rem .18rem 0; white-space: nowrap;
                 vertical-align: top; }
table.stats td { padding: .18rem 0; font-variant-numeric: tabular-nums; }
.chip { display: inline-block; padding: .3rem .65rem; border-radius: 999px;
        font-size: .75rem; max-width: 100%; overflow-wrap: anywhere; }
.chip-error { background: #46161c; color: #ff9ba3; white-space: pre-wrap; }
footer { margin-top: 3rem; color: #77777f; font-size: .75rem; }
"""


def seconds_text(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.2f} s"
    return "—"


def money_text(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"${float(value):.4f}"
    return "—"


def loudness_delta_text(out_i: Any, input_i: Any) -> str:
    if isinstance(out_i, (int, float)) and isinstance(input_i, (int, float)):
        return f"{float(out_i) - float(input_i):+.2f} LUFS"
    return "—"


def collect_entry(case: str, record: dict[str, Any],
                  review_dir: Path) -> dict[str, Any]:
    loudness: dict[str, Any] = {}
    loudness_path = review_dir / f"{case}.loudness.json"
    if loudness_path.is_file():
        try:
            loudness = json.loads(loudness_path.read_text())
        except (OSError, ValueError):
            loudness = {}
    model_key, clip = parse_case_name(case)
    return {
        "case": case,
        "model": record.get("model") or model_key,
        "clip": clip,
        "file": f"{case}.mp4",
        "total_wall_s": record.get("total_wall_s"),
        "queue_to_processing_s": record.get("queue_to_processing_s"),
        "output_duration_s": record.get("output_duration_s"),
        "estimated_cost_usd": record.get("estimated_cost_usd"),
        "input_i": loudness.get("input_i"),
        "out_i": loudness.get("out_i"),
    }


def render_card(entry: dict[str, Any]) -> str:
    rows: list[str] = []

    def add(label: str, value: str) -> None:
        rows.append(
            f"<tr><th>{html.escape(label)}</th>"
            f"<td>{html.escape(value)}</td></tr>"
        )

    add("model", str(entry["model"]))
    add("case", entry["case"])
    add("status", "ok")
    add("wall", seconds_text(entry.get("total_wall_s")))
    add("queue → processing", seconds_text(entry.get("queue_to_processing_s")))
    add("output duration", seconds_text(entry.get("output_duration_s")))
    add("est cost", money_text(entry.get("estimated_cost_usd")))
    add("loudness Δ", loudness_delta_text(entry.get("out_i"), entry.get("input_i")))
    source = html.escape(str(entry["file"]), quote=True)
    return (
        '<div class="card">'
        f'<h3>{html.escape(str(entry["model"]))}</h3>'
        f'<video controls preload="metadata" src="{source}"></video>'
        f'<table class="stats">{"".join(rows)}</table>'
        "</div>"
    )


def render_error_card(model: str, message: str) -> str:
    return (
        '<div class="card">'
        f"<h3>{html.escape(model)}</h3>"
        f'<span class="chip chip-error">{html.escape(message)}</span>'
        "</div>"
    )


def render_index(review_dir: Path, run_id: str, entries: list[dict[str, Any]],
                 failures: dict[str, str]) -> None:
    clips: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in entries:
        clips.setdefault(entry["clip"], {})[entry["model"]] = entry

    failed_by_clip: dict[str, list[tuple[str, str]]] = {}
    for case, message in sorted(failures.items()):
        model, clip = parse_case_name(case)
        failed_by_clip.setdefault(clip, []).append((model or "?", message))
    ordered_clips = sorted(set(clips) | set(failed_by_clip))

    title = f"SFX bench review — {run_id}"
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        '<p class="meta">'
        f"generated {html.escape(utc_now())} · "
        f"{len(ordered_clips)} clips · "
        f"{len(entries)} clips built · {len(failures)} problem cases · "
        f"target {TARGET_I:g} LUFS / {TARGET_TP:g} dBTP / LRA {TARGET_LRA:g}"
        "</p>",
    ]
    for clip in ordered_clips:
        lines.append(f'<section id="{html.escape(clip, quote=True)}">')
        lines.append(f"<h2>{html.escape(clip)}</h2>")
        lines.append('<div class="row">')
        for _, entry in sorted(clips.get(clip, {}).items()):
            lines.append(render_card(entry))
        for model, message in failed_by_clip.get(clip, []):
            lines.append(render_error_card(model, message))
        lines.append("</div>")
        lines.append("</section>")
    lines.append("<footer>Normalized audio (two-pass loudnorm) muxed onto the "
                 "original benchmark input frames with -c:v copy.</footer>")
    lines.append("</body>")
    lines.append("</html>")
    (review_dir / "index.html").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="work/sfx-bench/<run_id> directory")
    parser.add_argument("--cases", default="",
                        help="substring filter applied to case names")
    parser.add_argument("--force", action="store_true",
                        help="rebuild review artifacts even if they exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"build_review: run directory not found: {run_dir}",
              file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print("build_review: ffmpeg not found on PATH", file=sys.stderr)
        return 1
    try:
        run_id, records_dir, review_dir = resolve_run_layout(run_dir)
    except FileNotFoundError as exc:
        print(f"build_review: {exc}", file=sys.stderr)
        return 1
    review_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, Any]] = {}
    for path in sorted(records_dir.glob("*.json")):
        record = load_record(path)
        if record is None:
            continue
        name = str(record.get("name") or path.stem)
        if args.cases and args.cases not in name:
            continue
        records[name] = record
    if not records:
        print("build_review: no case records matched", file=sys.stderr)
        return 1

    completed = {n: r for n, r in records.items() if is_completed(r)}
    failed = {n: r for n, r in records.items() if not is_completed(r)}

    built = 0
    skipped = 0
    failures: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for case, record in sorted(completed.items()):
        try:
            outcome = build_case(case, record, review_dir, args.force)
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f"{case}: review build FAILED: {exc}",
                  file=sys.stderr, flush=True)
            failures[case] = str(exc)
            continue
        if outcome == "built":
            built += 1
        else:
            skipped += 1
        entries.append(collect_entry(case, record, review_dir))
    for case, record in sorted(failed.items()):
        failures[case] = str(record.get("error") or record.get("status")
                             or "failed")

    render_index(review_dir, run_id, entries, failures)
    print(
        f"build_review: built {built}, reused {skipped}, "
        f"problem cases {len(failures)}; index at "
        f"{review_dir / 'index.html'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
