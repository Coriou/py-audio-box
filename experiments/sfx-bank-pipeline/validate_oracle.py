#!/usr/bin/env python3
"""Stage 2 — validate V2A timing oracles against auto-sfx contacts.

Races every benchmarked model's raw output as the timing oracle: greedy
one-to-one matching between detected onsets and foot-contact ground truth
(clip-relative), absolute-error statistics, and a per-clip winner selection
(most matched contacts first, then lowest median error). The POC gate is
median |error| ≤ 50 ms for footstep clips (owner-confirmed 2026-08-26).

Outputs ``validation/timing.json`` and a human-readable ``timing.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from common import (
    BENCH_RUN_DIR,
    ORACLE_MODELS,
    REPO_ROOT,
    atomic_json,
    load_contacts,
    utc_now,
)
#: Matching tolerance; beyond this an event is "missed", not "late".
MATCH_TOLERANCE_MS = 250.0
GATE_MEDIAN_MS = 50.0


def match_events(
    contacts: list[float], detections: list[float]
) -> tuple[list[tuple[int, int, float]], set[int]]:
    """Greedy nearest matching under tolerance.

    Returns list of (contact_index, detection_index, error_ms) plus the set of
    unmatched detection indices.
    """
    pairs: list[tuple[int, int, float]] = []
    used: set[int] = set()
    for contact_index, contact_time in enumerate(contacts):
        best_index: int | None = None
        best_error = MATCH_TOLERANCE_MS
        for detection_index, detection_time in enumerate(detections):
            if detection_index in used:
                continue
            error = abs(detection_time - contact_time) * 1000.0
            if error < best_error:
                best_error = error
                best_index = detection_index
        if best_index is not None:
            used.add(best_index)
            error_ms = (detections[best_index] - contact_time) * 1000.0
            pairs.append((contact_index, best_index, round(error_ms, 2)))
    return pairs, used


def score_pairing(pairs: list[tuple[int, int, float]], contact_count: int,
                  detection_count: int) -> dict[str, Any]:
    errors = [abs(error) for _, _, error in pairs]
    signed = [error for _, _, error in pairs]
    return {
        "contacts": contact_count,
        "detections": detection_count,
        "matched": len(pairs),
        "hit_rate": round(len(pairs) / contact_count, 3) if contact_count else None,
        "median_abs_error_ms": round(statistics.median(errors), 1) if errors else None,
        "mean_abs_error_ms": round(statistics.fmean(errors), 1) if errors else None,
        "p90_abs_error_ms": round(sorted(errors)[int(0.9 * (len(errors) - 1))], 1) if errors else None,
        "max_abs_error_ms": round(max(errors), 1) if errors else None,
        "mean_signed_error_ms": round(statistics.fmean(signed), 1) if signed else None,
        "false_positives": max(0, detection_count - len(pairs)),
    }


def better_winner(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when b beats a as oracle candidate for a clip."""
    key_a = (a["matched"], -(a["median_abs_error_ms"] or 10**9))
    key_b = (b["matched"], -(b["median_abs_error_ms"] or 10**9))
    return key_b > key_a


def validate_run(run_dir: Path, auto_sfx_runs: Path) -> dict[str, Any]:
    oracle_dir = run_dir / "oracle"
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "tolerance_ms": MATCH_TOLERANCE_MS,
        "gate_median_ms": GATE_MEDIAN_MS,
        "oracle_models": list(ORACLE_MODELS),
        "clips": {},
    }

    for clip in ("walk1", "walk2", "man", "ocean-long", "ocean-short"):
        contacts_payload = load_contacts(clip, auto_sfx_runs)
        contact_times = [entry["time_s"] for entry in contacts_payload]
        clip_entry: dict[str, Any] = {"models": {}}
        if not contact_times:
            clip_entry["note"] = "no foot-contact ground truth for this clip"
            report["clips"][clip] = clip_entry
            continue

        for model in ORACLE_MODELS:
            events_path = oracle_dir / f"{clip}.{model}.events.json"
            if not events_path.exists():
                continue
            payload = json.loads(events_path.read_text())
            detections = [float(event["time_s"]) for event in payload["events"]]
            pairs, used = match_events(contact_times, detections)
            stats = score_pairing(pairs, len(contact_times), len(detections))
            stats["gate_pass"] = (
                stats["median_abs_error_ms"] is not None
                and stats["matched"] == stats["contacts"]
                and stats["median_abs_error_ms"] <= GATE_MEDIAN_MS
            )
            clip_entry["models"][model] = stats

        candidates = [m for m in clip_entry["models"]
                      if clip_entry["models"][m]["matched"] > 0]
        if candidates:
            best = candidates[0]
            for model in candidates[1:]:
                if better_winner(clip_entry["models"][best], clip_entry["models"][model]):
                    best = model
            clip_entry["best_model"] = best
        report["clips"][clip] = clip_entry

    winners = [entry["models"][entry["best_model"]]
               for entry in report["clips"].values()
               if entry.get("best_model")]
    medians = [w["median_abs_error_ms"] for w in winners if w["median_abs_error_ms"] is not None]
    all_matched = all(w["matched"] == w["contacts"] for w in winners) if winners else False
    pooled_median = round(statistics.median(medians), 1) if medians else None
    report["overall"] = {
        "footstep_clips": len(winners),
        "winning_models": {clip: entry.get("best_model")
                           for clip, entry in report["clips"].items()
                           if entry.get("best_model")},
        "pooled_median_abs_error_ms": pooled_median,
        "all_contacts_matched": all_matched,
        "gate_pass": pooled_median is not None and pooled_median <= GATE_MEDIAN_MS,
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    def ms(value: Any) -> str:
        return "—" if value is None else f"{value} ms"

    lines = [
        "# Oracle timing validation — V2A outputs vs auto-sfx contacts",
        "",
        f"Generated {report['generated_at']} · match tolerance "
        f"{report['tolerance_ms']:.0f} ms · gate: ALL contacts matched AND median ≤ "
        f"{report['gate_median_ms']:.0f} ms",
        "",
        "| clip | oracle | contacts | det | matched | hit | median abs | p90 | max | mean signed | FP | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for clip, entry in report["clips"].items():
        if entry.get("note"):
            lines.append(f"| {clip} | — | — | — | — | — | — | — | — | — | — | n/a ({entry['note']}) |")
            continue
        best = entry.get("best_model")
        for model, stats in sorted(entry["models"].items()):
            marker = " **← best**" if model == best else ""
            lines.append(
                f"| {clip}{marker} | {model} | {stats['contacts']} | {stats['detections']} "
                f"| {stats['matched']} | {stats['hit_rate']:.0%} | {ms(stats['median_abs_error_ms'])} "
                f"| {ms(stats['p90_abs_error_ms'])} | {ms(stats['max_abs_error_ms'])} "
                f"| {ms(stats['mean_signed_error_ms'])} | {stats['false_positives']} "
                f"| {'PASS' if stats['gate_pass'] else 'FAIL'} |"
            )
    overall = report["overall"]
    pooled = overall["pooled_median_abs_error_ms"]
    lines += [
        "",
        f"Winners: `{overall['winning_models'] or '—'}`",
        "",
        f"Pooled median over {overall['footstep_clips']} footstep clips: "
        + ("—" if pooled is None else f"**{pooled} ms**")
        + f" · all contacts matched: {overall['all_contacts_matched']}"
        + f" → gate {'PASS' if overall['gate_pass'] else 'FAIL'}.",
        "",
        "FP = oracle onsets with no contact within tolerance (ambience/cloth/rustle",
        "transients in the generated track). Detection params: see oracle/*.events.json.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, default=BENCH_RUN_DIR)
    parser.add_argument("--auto-sfx-runs", type=Path,
                        default=REPO_ROOT / "work" / "auto-sfx" / "runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_run(args.run_dir, args.auto_sfx_runs)
    out_dir = args.run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(out_dir / "timing.json", report)
    md_path = out_dir / "timing.md"
    md_path.write_text(render_markdown(report))
    print(md_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
