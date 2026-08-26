#!/usr/bin/env python3
"""Aggregate non-overlapping Metrics-v2 shards into one acceptance table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    docs = [json.loads(p.read_text("utf-8")) for p in args.metrics]
    ex_keys = ("n_true", "n_pred", "matched", "false_positives", "misses", "duplicates")
    ex = {k: sum(int(d["existence"].get(k) or 0) for d in docs) for k in ex_keys}
    ex["precision"] = ex["matched"] / ex["n_pred"] if ex["n_pred"] else None
    ex["recall"] = ex["matched"] / ex["n_true"] if ex["n_true"] else None
    pairs = [p for d in docs for p in d["timing"].get("diagnostic_pairs", [])]
    abs_frames = [abs(float(p["err_frames"])) for p in pairs]
    signed_frames = [float(p["err_frames"]) for p in pairs]
    worse = sum(v > 3.0 for v in abs_frames)
    side_flips = sum(int(d["side"].get("side_flips") or 0) for d in docs)
    # Metrics v2 defines flips among strict known-side matches.  Uncertain-side
    # contacts participate in existence but never enter this denominator.
    known_matched = sum(
        int(d["side"].get("strict_known_side_matches") or 0) for d in docs
    )
    simultaneous = sum(
        int(d["sequence"].get("impossible_simultaneous_ic") or 0) for d in docs
    )
    result = {
        "schema_version": 1,
        "label": args.label,
        "n_shards": len(docs),
        "sources": [str(p) for p in args.metrics],
        "existence": ex,
        "timing_frames": {
            "n": len(abs_frames),
            "bias": sum(signed_frames) / len(signed_frames) if signed_frames else None,
            "median_abs": percentile(abs_frames, 0.5),
            "p90_abs": percentile(abs_frames, 0.9),
            "p95_abs": percentile(abs_frames, 0.95),
            "within_1_frac": (
                sum(v <= 1.0 for v in abs_frames) / len(abs_frames)
                if abs_frames else None
            ),
            "worse_than_3": worse,
            "worse_than_3_frac": worse / len(abs_frames) if abs_frames else None,
        },
        "side": {
            "strict_known_side_matches": known_matched,
            "side_flips": side_flips,
            "attribution_accuracy": (
                1.0 - side_flips / known_matched if known_matched else None
            ),
        },
        "sequence": {"impossible_simultaneous_ic": simultaneous},
        "acceptance": {
            "precision_ge_0_90": ex["precision"] is not None and ex["precision"] >= 0.90,
            "recall_ge_0_90": ex["recall"] is not None and ex["recall"] >= 0.90,
            "median_abs_le_1f": bool(abs_frames) and percentile(abs_frames, 0.5) <= 1.0,
            "p90_abs_le_3f": bool(abs_frames) and percentile(abs_frames, 0.9) <= 3.0,
            "worse_than_3_le_5pct": bool(abs_frames) and worse / len(abs_frames) <= 0.05,
            "side_accuracy_ge_0_95": (
                known_matched > 0 and 1.0 - side_flips / known_matched >= 0.95
            ),
            "no_simultaneous_lr": simultaneous == 0,
        },
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, "utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
