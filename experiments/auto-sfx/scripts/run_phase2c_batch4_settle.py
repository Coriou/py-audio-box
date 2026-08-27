#!/usr/bin/env python3
"""Opened-development post-apex-settle mechanism diagnostic.

Loads cached pose traces only. Scores agent-draft Phase 2B development shards
with Metrics v2. Never opens validation/challenge/holdout shards, never runs
pose inference, and never registers a global timing offset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from auto_sfx.metrics import evaluate_v2, load_gt_any
from auto_sfx.types import PoseTrace, VideoMeta


SCHEMA = "auto-sfx-phase2c-b4-settle-diagnostic-v1"
METRICS_SCHEMA = "auto-sfx-phase2c-b4-public-metrics-v1"
EVIDENCE_BOUNDARY = (
    "All Phase 2B shards are opened diagnostics with agent-draft truth. No "
    "challenge/locked corpus exists, and no contact accuracy is claimed. "
    "Development-only comparison diagnoses mechanism bias; it is not the "
    "operator-gold gate."
)
OFFSET_KEYS = frozenset({"global_offset", "timing_offset", "frame_offset"})
FORBIDDEN_FOLDS = frozenset({"validation", "challenge", "holdout"})
CLIP_BY_TRACE = {"walk1": "walk1.mp4", "walk2": "walk2.webm"}
STEM_BY_CLIP = {"walk1.mp4": "walk1", "walk2.webm": "walk2"}
DEVELOPMENT_GT_NAME = re.compile(r"^(walk[12])\.development\.\d+\.gt\.json$")
EXCLUDED_GT_FOLD = re.compile(r"\.(validation|challenge|holdout)\.")
VARIANT_ORDER = (
    "baseline",
    "speed_control",
    "candidate",
)
REQUIRED_VARIANTS = {
    "baseline": {"kind": "contact-state-v1"},
    "speed_control": {
        "kind": "contact-settle-v1",
        "settle_signal": "speed",
    },
    "candidate": {
        "kind": "contact-settle-v1",
        "settle_signal": "downward_vy",
    },
}
REQUIRED_DECISION_GATES = {
    "recall_slack": 0.02,
    "offset_like_mode_fraction": 0.8,
    "accept_requires": [
        "candidate_n_pred_eq_baseline",
        "candidate_signed_median_closer_to_zero_than_baseline",
        "candidate_signed_median_greater_than_baseline",
        "not_offset_like",
        "candidate_diagnostic_median_abs_le_baseline",
        "candidate_strict_recall_ge_baseline_minus_slack",
        "candidate_simultaneous_lr_eq_0",
        "wall_and_rss_within_budget",
    ],
    "reject_if_any": [
        "offset_like",
        "signed_bias_not_improved",
        "recall_drop_gt_slack",
        "existence_not_preserved",
        "tests_fail",
        "global_offset_key_exists",
    ],
    "inconclusive_if": "mixed_and_not_offset_like",
}
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "evidence_boundary",
    "fold",
    "maximum_candidate_wall_seconds",
    "maximum_candidate_peak_rss_mib",
    "candidate_config",
    "baseline_config",
    "variants",
    "decision_gates",
}
# Frozen from the Batch 4 contract: SETTLE_CONFIG plus contact-state-v1
# passthrough keys. Unknown keys fail closed so config tests collect before
# Worker A's modules exist. geometry_mode is interval-only and rejected.
CANDIDATE_KEYS = {
    "min_step_interval_s",
    "max_step_interval_s",
    "min_landmark_visibility",
    "shot_boundaries",
    "actor_too_small_px",
    "camera_motion_std_px",
    "track_policy",
    "track_jump_speed_bs_s",
    "track_swap_margin_bs",
    "track_max_interp_gap",
    "track_max_swap_run_frames",
    "smoothing",
    "local_window_s",
    "min_grounded_frames",
    "grounded_score",
    "support_height_frac",
    "phase_min_velocity_bs_s",
    "phase_peak_radius_frames",
    "min_same_foot_cycle_s",
    "max_forward_frames",
    "settle_vy_bs_s",
    "settle_speed_bs_s",
    "settle_signal",
}
BASELINE_KEYS = {
    "min_step_interval_s",
    "max_step_interval_s",
    "min_landmark_visibility",
    "shot_boundaries",
    "actor_too_small_px",
    "camera_motion_std_px",
    "track_policy",
    "track_jump_speed_bs_s",
    "track_swap_margin_bs",
    "track_max_interp_gap",
    "track_max_swap_run_frames",
    "smoothing",
    "local_window_s",
    "min_grounded_frames",
    "grounded_score",
    "support_height_frac",
    "phase_min_velocity_bs_s",
    "phase_peak_radius_frames",
    "min_same_foot_cycle_s",
}
SETTLE_SIGNALS = frozenset({"downward_vy", "speed"})
VARIANT_SUMMARY_KEYS = (
    "n_true",
    "n_pred",
    "n_matched",
    "strict_precision",
    "strict_recall",
    "clear_recall",
    "diagnostic_n",
    "diagnostic_median_abs_frames",
    "diagnostic_p90_abs_frames",
    "fraction_gt3_frames",
    "signed_median_frames",
    "known_side_matches",
    "known_side_attribution_accuracy",
    "simultaneous_lr",
    "wall_seconds",
)
REQUIRED_PUBLIC_SCALARS = (
    "recommended_decision",
    "offset_like",
    "existence_preserved",
    "candidate_n_pred",
    "baseline_n_pred",
    "candidate_strict_recall",
    "baseline_strict_recall",
    "candidate_signed_median_frames",
    "baseline_signed_median_frames",
)
PUBLIC_LEAK_KEYS = frozenset({
    "argv",
    "fold_ranges",
    "preferred_s",
    "preferred_pts_us",
    "earliest_s",
    "latest_s",
    "earliest_pts_us",
    "latest_pts_us",
    "evaluation_ranges_s",
    "path",
    "paths",
    "gt_dir",
    "walk1_pose",
    "walk2_pose",
    "output_dir",
    "split_membership",
})


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(document: Any) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from None


def _load_trace(path: Path) -> PoseTrace:
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = json.loads(next(handle))
            rows = [json.loads(line) for line in handle]
    except (OSError, StopIteration, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pose trace {path}: {exc}") from None
    if header.get("format") != "auto-sfx-pose-trace" or header.get("format_version") != 1:
        raise ValueError(f"unsupported pose trace format in {path}")
    trace = PoseTrace.from_rows(
        header["backend"], header["mode"], VideoMeta(**header["meta"]), rows
    )
    if len(trace.frames) != trace.meta.n_frames:
        raise ValueError("pose trace frame count disagrees with metadata")
    return trace


def _contains_offset_key(document: Any) -> str | None:
    if isinstance(document, dict):
        found = sorted(OFFSET_KEYS & set(document))
        if found:
            return found[0]
        for value in document.values():
            nested = _contains_offset_key(value)
            if nested is not None:
                return nested
    elif isinstance(document, (list, tuple)):
        for value in document:
            nested = _contains_offset_key(value)
            if nested is not None:
                return nested
    return None


def _validate_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        missing = sorted(CONFIG_KEYS - set(raw)) if isinstance(raw, dict) else []
        extra = sorted(set(raw) - CONFIG_KEYS) if isinstance(raw, dict) else []
        raise ValueError(f"config keys are not exact; missing={missing}, extra={extra}")
    forbidden = _contains_offset_key(raw)
    if forbidden is not None:
        raise ValueError(f"config contains forbidden offset key: {forbidden}")
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise ValueError("config schema_version must be 1")
    if raw["fold"] != "development":
        raise ValueError("only the opened development fold is permitted")
    if raw["evidence_boundary"] != EVIDENCE_BOUNDARY:
        raise ValueError("evidence_boundary must be the frozen Batch 2 sentence")
    if not isinstance(raw["experiment_id"], str) or not raw["experiment_id"]:
        raise ValueError("experiment_id must be a non-empty string")
    for key in ("maximum_candidate_wall_seconds", "maximum_candidate_peak_rss_mib"):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key, allowed in (
        ("candidate_config", CANDIDATE_KEYS),
        ("baseline_config", BASELINE_KEYS),
    ):
        value = raw[key]
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError(f"{key} contains unknown keys")
    signal = raw["candidate_config"].get("settle_signal")
    if signal is not None and signal not in SETTLE_SIGNALS:
        raise ValueError("unsupported candidate settle_signal")
    if raw["variants"] != REQUIRED_VARIANTS:
        raise ValueError("variants are not the frozen registered set")
    if raw["decision_gates"] != REQUIRED_DECISION_GATES:
        raise ValueError("decision_gates are not the frozen registered set")
    return raw


def _atomic_json(path: Path, document: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _peak_rss_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if platform.system() == "Linux" else 1024.0 * 1024.0
    return round(raw / divisor, 6)


def _source_pts_failures(trace: PoseTrace, records: Sequence[dict[str, Any]]) -> int:
    expected = {frame.frame_index: frame.source_pts_us for frame in trace.frames}
    pts_to_frame = {frame.source_pts_us: frame.frame_index for frame in trace.frames}
    failures = 0
    for record in records:
        frame = record.get("frame")
        pts = record.get("source_pts_us")
        if frame is not None:
            if frame not in expected or pts != expected[frame]:
                failures += 1
        elif pts not in pts_to_frame:
            failures += 1
    return failures


def _fold_ranges(split: Any, clip_key: str, fold: str) -> list[tuple[int, int]]:
    try:
        raw_ranges = split["clips"][clip_key][fold]
        ranges = [(int(item["start_frame"]), int(item["end_frame"])) for item in raw_ranges]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"split has no valid {clip_key}/{fold} ranges") from exc
    if not ranges or any(start < 0 or end <= start for start, end in ranges):
        raise ValueError("split ranges must be non-empty half-open frame intervals")
    overlaps = any(right[0] < left[1] for left, right in zip(ranges, ranges[1:]))
    if ranges != sorted(ranges) or overlaps:
        raise ValueError("split ranges must be sorted and disjoint")
    return ranges


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _seconds_range_to_frames(start_s: float, end_s: float, fps: float) -> tuple[int, int]:
    start = int(round(float(start_s) * fps))
    end_inclusive = int(round(float(end_s) * fps))
    return start, end_inclusive + 1


def _preferred_frame(record: dict[str, Any], fps: float) -> int:
    if record.get("preferred_pts_us") is not None:
        return int(round(float(record["preferred_pts_us"]) / 1e6 * fps))
    return int(round(float(record["preferred_s"]) * fps))


def _in_eval_range(frame: int, ranges: Sequence[tuple[int, int]], guard: int) -> bool:
    return any(start + guard <= frame < end - guard for start, end in ranges)


def _development_gt_paths(gt_dir: Path) -> list[Path]:
    if not gt_dir.is_dir():
        raise ValueError(f"gt-dir is not a directory: {gt_dir}")
    paths: list[Path] = []
    for path in sorted(gt_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        # Never open validation/challenge/holdout shards, even as JSON.
        if EXCLUDED_GT_FOLD.search(name):
            continue
        if DEVELOPMENT_GT_NAME.fullmatch(name):
            paths.append(path)
    if not paths:
        raise ValueError("no walk1/walk2 development GT shards found")
    return paths


def _assert_development_range(
    clip: str,
    range_s: Sequence[float] | None,
    contacts: Sequence[dict[str, Any]],
    split: dict[str, Any],
    fps: float,
) -> None:
    validation = _fold_ranges(split, clip, "validation")
    if range_s is not None:
        loaded = _seconds_range_to_frames(range_s[0], range_s[1], fps)
        for other in validation:
            if _ranges_overlap(loaded, other):
                raise ValueError(
                    f"loaded GT clip-range for {clip} overlaps a validation range"
                )
    for record in contacts:
        frame = _preferred_frame(record, fps)
        if any(start <= frame < end for start, end in validation):
            raise ValueError(
                f"loaded GT contact for {clip} falls in a validation range"
            )


def _load_development_gt(
    gt_dir: Path, split: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    if split.get("schema_version") != 1:
        raise ValueError("opened Phase 2B split is not schema version 1")
    guard = split.get("boundary_guard_frames")
    if guard != 2 or isinstance(guard, bool):
        raise ValueError("split boundary_guard_frames must be 2")
    fps = float(split.get("fps_display") or 0)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("split fps_display must be a positive number")
    paths = _development_gt_paths(gt_dir)
    by_clip: dict[str, list[dict[str, Any]]] = {clip: [] for clip in STEM_BY_CLIP}
    for path in paths:
        match = DEVELOPMENT_GT_NAME.fullmatch(path.name)
        assert match is not None
        stem = match.group(1)
        expected_clip = CLIP_BY_TRACE[stem]
        raw = _read_json(path)
        file_clip = expected_clip
        file_range = None
        if isinstance(raw, dict):
            file_clip = raw.get("clip") or expected_clip
            file_fps = raw.get("fps_display")
            if file_fps is not None and abs(float(file_fps) - fps) > 1e-9:
                raise ValueError(f"GT fps_display disagrees with split in {path.name}")
            if file_clip != expected_clip:
                raise ValueError(f"GT clip {file_clip!r} does not match {path.name}")
            range_s = raw.get("range_s")
            if isinstance(range_s, dict) and "start" in range_s and "end" in range_s:
                file_range = (float(range_s["start"]), float(range_s["end"]))
        if file_clip not in STEM_BY_CLIP:
            raise ValueError(f"unsupported GT clip {file_clip!r}")
        records = load_gt_any(str(path))
        development = _fold_ranges(split, file_clip, "development")
        unique_ranges = sorted({
            tuple(record["evaluation_range_s"])
            for record in records
            if record.get("evaluation_range_s") is not None
        })
        if file_range is not None:
            unique_ranges = sorted(set(unique_ranges) | {file_range})
        if unique_ranges:
            for range_s in unique_ranges:
                _assert_development_range(file_clip, range_s, records, split, fps)
        else:
            _assert_development_range(file_clip, None, records, split, fps)
        kept: list[dict[str, Any]] = []
        for record in records:
            frame = _preferred_frame(record, fps)
            if not _in_eval_range(frame, development, int(guard)):
                continue
            kept.append(record)
        by_clip[file_clip].extend(kept)
    return by_clip, paths


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        payload = dict(event)
    elif hasattr(event, "to_dict"):
        payload = dict(event.to_dict())
    else:
        payload = {
            "id": getattr(event, "id", None),
            "time_s": getattr(event, "time_s", None),
            "foot": getattr(event, "foot", None),
            "source_pts_us": getattr(event, "source_pts_us", None),
            "evidence": getattr(event, "evidence", {}) or {},
            "method": getattr(event, "method", None),
            "detector": getattr(event, "detector", None),
        }
    evidence = payload.get("evidence") or {}
    frame = evidence.get("settle_frame")
    if frame is None:
        frame = evidence.get("apex_frame")
    if frame is None:
        frame = evidence.get("support_frame")
    if frame is None:
        frame = evidence.get("collision_frame")
    if frame is None:
        frame = evidence.get("heel_onset_frame")
    return {
        "id": payload.get("id"),
        "time_s": payload.get("time_s"),
        "foot": payload.get("foot"),
        "source_pts_us": payload.get("source_pts_us"),
        "frame": frame,
        "method": payload.get("method"),
        "detector": payload.get("detector"),
        "evidence": {
            key: evidence.get(key)
            for key in (
                "apex_frame",
                "settle_frame",
                "shift_frames",
                "settle_signal",
                "reason",
                "collision_frame",
                "support_frame",
                "heel_onset_frame",
            )
            if key in evidence
        },
    }


def _bind_event_frames(
    records: list[dict[str, Any]], trace: PoseTrace
) -> list[dict[str, Any]]:
    pts_to_frame = {frame.source_pts_us: frame.frame_index for frame in trace.frames}
    bound = []
    for record in records:
        row = dict(record)
        pts = row.get("source_pts_us")
        if pts in pts_to_frame:
            row["frame"] = pts_to_frame[pts]
        bound.append(row)
    return bound


def _offset_like(shifts: Sequence[int], fraction: float = 0.8) -> bool:
    if not shifts:
        return False
    counts = Counter(int(shift) for shift in shifts)
    mode, count = counts.most_common(1)[0]
    return mode != 0 and (count / len(shifts)) >= float(fraction)


def _existence_preserved(candidate_n_pred: Any, baseline_n_pred: Any) -> bool:
    if isinstance(candidate_n_pred, bool) or isinstance(baseline_n_pred, bool):
        return False
    try:
        return int(candidate_n_pred) == int(baseline_n_pred)
    except (TypeError, ValueError):
        return False


def _matched_frame_shifts(
    candidate_events: Sequence[dict[str, Any]],
    baseline_events: Sequence[dict[str, Any]],
    horizon_s: float = 0.5,
) -> list[int]:
    candidates: list[tuple[float, int, int]] = []
    for i, left in enumerate(candidate_events):
        for j, right in enumerate(baseline_events):
            try:
                dt = abs(float(left["time_s"]) - float(right["time_s"]))
            except (KeyError, TypeError, ValueError):
                continue
            if dt <= horizon_s:
                candidates.append((dt, i, j))
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    shifts: list[int] = []
    for _, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        cand_frame = candidate_events[i].get("frame")
        base_frame = baseline_events[j].get("frame")
        if cand_frame is None or base_frame is None:
            continue
        if isinstance(cand_frame, bool) or isinstance(base_frame, bool):
            continue
        used_left.add(i)
        used_right.add(j)
        shifts.append(int(cand_frame) - int(base_frame))
    return shifts


def _summarize_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ex_keys = ("n_true", "n_pred", "matched", "false_positives", "misses", "duplicates")
    existence = {
        key: sum(int((report.get("existence") or {}).get(key) or 0) for report in reports)
        for key in ex_keys
    }
    precision = (
        existence["matched"] / existence["n_pred"] if existence["n_pred"] else None
    )
    recall = existence["matched"] / existence["n_true"] if existence["n_true"] else None
    pairs = [
        pair
        for report in reports
        for pair in (report.get("timing") or {}).get("diagnostic_pairs", [])
    ]
    signed_frames = [
        float(pair["err_frames"])
        for pair in pairs
        if pair.get("err_frames") is not None
    ]
    abs_frames = [abs(value) for value in signed_frames]
    worse = sum(value > 3.0 for value in abs_frames)
    known_matched = sum(
        int((report.get("side") or {}).get("strict_known_side_matches") or 0)
        for report in reports
    )
    side_flips = sum(
        int((report.get("side") or {}).get("side_flips") or 0) for report in reports
    )
    clear_true = sum(
        int((view.get("visibility_counts") or {}).get("clear") or 0)
        for report in reports
        for view in (report.get("views") or {}).values()
    )
    clear_matched = sum(
        1
        for pair in pairs
        if pair.get("strictly_matched") and pair.get("visibility") == "clear"
    )
    simultaneous = sum(
        int((report.get("sequence") or {}).get("impossible_simultaneous_ic") or 0)
        for report in reports
    )
    return {
        "n_true": existence["n_true"],
        "n_pred": existence["n_pred"],
        "n_matched": existence["matched"],
        "strict_precision": _round(precision),
        "strict_recall": _round(recall),
        "clear_recall": _round(clear_matched / clear_true if clear_true else None),
        "diagnostic_n": len(signed_frames),
        "diagnostic_median_abs_frames": _round(_percentile(abs_frames, 0.5)),
        "diagnostic_p90_abs_frames": _round(_percentile(abs_frames, 0.9)),
        "fraction_gt3_frames": _round(worse / len(abs_frames) if abs_frames else None),
        "signed_median_frames": _round(_percentile(signed_frames, 0.5)),
        "known_side_matches": known_matched,
        "known_side_attribution_accuracy": _round(
            (1.0 - side_flips / known_matched) if known_matched else None
        ),
        "simultaneous_lr": simultaneous,
    }


def _closer_to_zero(candidate: float | None, baseline: float | None) -> bool:
    if candidate is None or baseline is None:
        return False
    return abs(float(candidate)) < abs(float(baseline))


def _recommended_decision(
    predicates: dict[str, bool], gates: dict[str, Any]
) -> str:
    if predicates.get("offset_like"):
        return "reject"
    for name in gates["reject_if_any"]:
        if predicates.get(name):
            return "reject"
    if all(predicates.get(name) for name in gates["accept_requires"]):
        return "accept"
    return "inconclusive"


def _predicates(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    offset_like: bool,
    tests_fail: bool,
    global_offset_key_exists: bool,
    candidate_wall: float,
    peak_rss: float,
    gates: dict[str, Any],
    max_wall: float,
    max_rss: float,
) -> dict[str, bool]:
    slack = float(gates["recall_slack"])
    cand_signed = candidate.get("signed_median_frames")
    base_signed = baseline.get("signed_median_frames")
    closer = _closer_to_zero(cand_signed, base_signed)
    greater = (
        cand_signed is not None
        and base_signed is not None
        and float(cand_signed) > float(base_signed)
    )
    cand_abs = candidate.get("diagnostic_median_abs_frames")
    base_abs = baseline.get("diagnostic_median_abs_frames")
    abs_ok = (
        cand_abs is not None
        and base_abs is not None
        and float(cand_abs) <= float(base_abs)
    )
    cand_recall = candidate.get("strict_recall")
    base_recall = baseline.get("strict_recall")
    recall_ok = (
        cand_recall is not None
        and base_recall is not None
        and float(cand_recall) >= float(base_recall) - slack
    )
    simultaneous_ok = int(candidate.get("simultaneous_lr") or 0) == 0
    budget_ok = candidate_wall <= float(max_wall) and peak_rss <= float(max_rss)
    existence_ok = _existence_preserved(
        candidate.get("n_pred"), baseline.get("n_pred")
    )
    return {
        "candidate_n_pred_eq_baseline": existence_ok,
        "candidate_signed_median_closer_to_zero_than_baseline": closer,
        "candidate_signed_median_greater_than_baseline": greater,
        "not_offset_like": not offset_like,
        "candidate_diagnostic_median_abs_le_baseline": abs_ok,
        "candidate_strict_recall_ge_baseline_minus_slack": recall_ok,
        "candidate_simultaneous_lr_eq_0": simultaneous_ok,
        "wall_and_rss_within_budget": budget_ok,
        "offset_like": offset_like,
        "signed_bias_not_improved": not (closer and greater),
        "recall_drop_gt_slack": not recall_ok,
        "existence_not_preserved": not existence_ok,
        "tests_fail": tests_fail,
        "global_offset_key_exists": global_offset_key_exists,
    }


def _variant_public_scalars(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scalars: dict[str, Any] = {}
    for name in VARIANT_ORDER:
        for key in VARIANT_SUMMARY_KEYS:
            scalars[f"{name}_{key}"] = summaries[name][key]
    return scalars


def _assert_public_metrics(metrics: dict[str, Any]) -> None:
    for key in REQUIRED_PUBLIC_SCALARS:
        if key not in metrics:
            raise ValueError(f"public metrics missing scalar {key}")
        if isinstance(metrics[key], (dict, list)):
            raise ValueError(f"public metric must be a JSON scalar: {key}")
    stack = [metrics]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            leaked = sorted(PUBLIC_LEAK_KEYS & set(current))
            if leaked:
                raise ValueError(f"public metrics contain forbidden keys: {leaked}")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _run_variant(
    trace: PoseTrace,
    spec: dict[str, Any],
    config: dict[str, Any],
    detect_contacts,
    detect_settle_contacts,
):
    if spec["kind"] == "contact-state-v1":
        return detect_contacts(
            trace, "contact-state-v1", config=config["baseline_config"] or None
        )
    if spec["kind"] != "contact-settle-v1":
        raise ValueError(f"unsupported variant kind: {spec['kind']}")
    overlay = dict(config["candidate_config"])
    overlay["settle_signal"] = spec["settle_signal"]
    if overlay["settle_signal"] not in SETTLE_SIGNALS:
        raise ValueError("unsupported settle_signal")
    if _contains_offset_key(overlay) is not None:
        raise ValueError("settle config contains a forbidden offset key")
    return detect_settle_contacts(trace, config=overlay)


def _require_image_id() -> str:
    image_id = os.environ.get("AUTO_SFX_IMAGE_ID")
    if (
        image_id is None
        or not image_id.startswith("sha256:")
        or len(image_id) != 71
        or any(character not in "0123456789abcdef" for character in image_id[7:])
    ):
        raise ValueError("AUTO_SFX_IMAGE_ID must be the pinned lowercase SHA-256 image ID")
    return image_id


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    image_id = _require_image_id()
    paths = {
        "walk1": Path(args.walk1_pose).resolve(),
        "walk2": Path(args.walk2_pose).resolve(),
        "split": Path(args.split).resolve(),
        "config": Path(args.config).resolve(),
    }
    gt_dir = Path(args.gt_dir).resolve()
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} input is not a file: {path}")
    config = _validate_config(_read_json(paths["config"]))
    split = _read_json(paths["split"])
    if not isinstance(split, dict) or split.get("schema_version") != 1:
        raise ValueError("opened Phase 2B split is not schema version 1")
    if config["fold"] in FORBIDDEN_FOLDS:
        raise ValueError("only the opened development fold is permitted")
    gt_by_clip, gt_paths = _load_development_gt(gt_dir, split)
    missing = [
        clip for clip in CLIP_BY_TRACE.values() if not gt_by_clip.get(clip)
    ]
    if missing:
        raise ValueError(f"no decision-evidence GT contacts for {missing}")
    fps = float(split["fps_display"])

    from auto_sfx.contacts import detect_contacts
    from auto_sfx.state_settle import detect_settle_contacts

    traces: dict[str, PoseTrace] = {}
    variant_events: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {} for name in VARIANT_ORDER
    }
    variant_wall = {name: 0.0 for name in VARIANT_ORDER}
    source_pts_failures = 0
    deterministic_failures = 0

    for trace_name in ("walk1", "walk2"):
        trace = _load_trace(paths[trace_name])
        traces[trace_name] = trace
        for variant_name in VARIANT_ORDER:
            spec = config["variants"][variant_name]
            started = time.perf_counter()
            result = _run_variant(
                trace, spec, config, detect_contacts, detect_settle_contacts
            )
            elapsed = time.perf_counter() - started
            variant_wall[variant_name] += elapsed
            events = _bind_event_frames(
                [_event_dict(event) for event in result.events], trace
            )
            source_pts_failures += _source_pts_failures(trace, events)
            if variant_name == "candidate":
                repeated = _run_variant(
                    trace, spec, config, detect_contacts, detect_settle_contacts
                )
                first_hash = _hash_json(events)
                repeated_hash = _hash_json(
                    _bind_event_frames(
                        [_event_dict(event) for event in repeated.events], trace
                    )
                )
                deterministic_failures += int(first_hash != repeated_hash)
            variant_events[variant_name][trace_name] = events
        del trace
        gc.collect()

    summaries: dict[str, dict[str, Any]] = {}
    for variant_name in VARIANT_ORDER:
        reports = []
        spec = config["variants"][variant_name]
        method = spec["kind"]
        for trace_name in ("walk1", "walk2"):
            reports.append(
                evaluate_v2(
                    variant_events[variant_name][trace_name],
                    gt_by_clip[CLIP_BY_TRACE[trace_name]],
                    fps,
                    backend=traces[trace_name].backend,
                    method=method,
                )
            )
        summaries[variant_name] = {
            **_summarize_reports(reports),
            "wall_seconds": round(variant_wall[variant_name], 6),
        }

    shifts: list[int] = []
    for trace_name in ("walk1", "walk2"):
        shifts.extend(
            _matched_frame_shifts(
                variant_events["candidate"][trace_name],
                variant_events["baseline"][trace_name],
            )
        )
    fraction = float(config["decision_gates"]["offset_like_mode_fraction"])
    offset_like = _offset_like(shifts, fraction)
    mode = Counter(shifts).most_common(1)[0][0] if shifts else None
    mode_fraction = (
        (Counter(shifts).most_common(1)[0][1] / len(shifts)) if shifts else None
    )
    peak_rss = _peak_rss_mib()
    candidate_wall = round(variant_wall["candidate"], 6)
    tests_fail = source_pts_failures > 0 or deterministic_failures > 0
    predicates = _predicates(
        candidate=summaries["candidate"],
        baseline=summaries["baseline"],
        offset_like=offset_like,
        tests_fail=tests_fail,
        global_offset_key_exists=_contains_offset_key(config) is not None,
        candidate_wall=candidate_wall,
        peak_rss=peak_rss,
        gates=config["decision_gates"],
        max_wall=float(config["maximum_candidate_wall_seconds"]),
        max_rss=float(config["maximum_candidate_peak_rss_mib"]),
    )
    recommended = _recommended_decision(predicates, config["decision_gates"])
    existence_ok = bool(predicates["candidate_n_pred_eq_baseline"])
    nested_variants = {
        name: {key: summaries[name][key] for key in VARIANT_SUMMARY_KEYS}
        for name in VARIANT_ORDER
    }
    metrics = {
        "schema": METRICS_SCHEMA,
        "experiment_id": config["experiment_id"],
        "fold": "development",
        "accuracy_claim": False,
        "operator_truth_used": False,
        "pose_inference_run": False,
        "validation_shards_loaded": 0,
        "recommended_decision": recommended,
        "offset_like": offset_like,
        "offset_like_mode": mode,
        "offset_like_mode_fraction": _round(mode_fraction),
        "offset_like_n": len(shifts),
        "existence_preserved": existence_ok,
        "candidate_wall_seconds": candidate_wall,
        "baseline_wall_seconds": round(variant_wall["baseline"], 6),
        "speed_control_wall_seconds": round(variant_wall["speed_control"], 6),
        "candidate_peak_rss_mib": peak_rss,
        "source_pts_membership_failures": source_pts_failures,
        "deterministic_failures": deterministic_failures,
        **_variant_public_scalars(summaries),
    }
    _assert_public_metrics(metrics)
    artifact = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "evidence_boundary": config["evidence_boundary"],
        "accuracy_claim": False,
        "operator_truth_used": False,
        "pose_inference_run": False,
        "validation_shards_loaded": 0,
        "fold": "development",
        "inputs": {
            name: {"sha256": _hash_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(paths.items())
        },
        "gt_development_shards": [
            {"name": path.name, "sha256": _hash_file(path), "bytes": path.stat().st_size}
            for path in gt_paths
        ],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "docker_image_id": image_id,
        },
        "variants": nested_variants,
        "gates": predicates,
        "summary": {
            **metrics,
            "decision_note": (
                "Opened development mechanism diagnostics only. Agent-draft GT "
                "is used for signed-error diagnosis; no contact accuracy is claimed."
            ),
        },
        "variant_event_sha256": {
            variant_name: {
                trace_name: _hash_json(rows)
                for trace_name, rows in traces_rows.items()
            }
            for variant_name, traces_rows in variant_events.items()
        },
        "trace_frames": {
            name: len(trace.frames) for name, trace in traces.items()
        },
    }
    return artifact, metrics


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--walk1-pose", required=True, dest="walk1_pose")
    parser.add_argument("--walk2-pose", required=True, dest="walk2_pose")
    parser.add_argument("--split", required=True)
    parser.add_argument("--gt-dir", required=True, dest="gt_dir")
    parser.add_argument("--output-dir", required=True, dest="output_dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    artifact_path = output_dir / "artifact.json"
    metrics_path = output_dir / "metrics.json"
    if artifact_path == metrics_path:
        raise SystemExit("artifact and metrics paths must differ")
    if artifact_path.exists() or metrics_path.exists():
        raise SystemExit("refusing to overwrite an existing output")
    try:
        artifact, metrics = run(args)
        _atomic_json(artifact_path, artifact)
        _atomic_json(metrics_path, metrics)
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
