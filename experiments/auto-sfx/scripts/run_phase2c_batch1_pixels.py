#!/usr/bin/env python3
"""Run the opened-data Phase 2C sparse-pixel mechanism diagnostic.

This driver does not run pose inference and does not consume ground-truth
contacts. It binds an immutable source video to genuine ffprobe presentation
PTS, an existing tracked pose trace, existing coarse pose events, the opened
Phase 2B split, and the cached rejected dense-ROI output. Results compare
mechanical coverage, shifts, abstentions, timing, and resources only; they are
not accuracy evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from auto_sfx.foot_contact_session import probe_source_timestamps
from auto_sfx.sparse_roi_refiner import (
    analyze_video_candidate_window,
    sparse_roi_config,
)
from auto_sfx.types import Point, PoseTrace, VideoMeta


SCHEMA = "auto-sfx-phase2c-b1-sparse-diagnostic-v1"
METRICS_SCHEMA = "auto-sfx-phase2c-b1-public-metrics-v1"
EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "evidence_boundary",
    "clip_key",
    "fold",
    "require_cfr",
    "window_half_frames",
    "minimum_suggestion_coverage",
    "maximum_candidate_wall_seconds",
    "maximum_candidate_peak_rss_mib",
    "sparse_config",
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("pose trace frame count disagrees with its metadata")
    return trace


def _validate_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != EXPECTED_CONFIG_KEYS:
        missing = sorted(EXPECTED_CONFIG_KEYS - set(raw or {})) if isinstance(raw, dict) else []
        extra = sorted(set(raw or {}) - EXPECTED_CONFIG_KEYS) if isinstance(raw, dict) else []
        raise ValueError(f"config keys are not exact; missing={missing}, extra={extra}")
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise ValueError("config schema_version must be 1")
    if raw["fold"] != "development":
        raise ValueError("only the opened development fold is permitted")
    if raw["require_cfr"] is not True:
        raise ValueError("this OpenCV driver is restricted to a verified CFR source")
    half = raw["window_half_frames"]
    if not isinstance(half, int) or isinstance(half, bool) or not 2 <= half <= 10:
        raise ValueError("window_half_frames must be an integer in [2, 10]")
    for key in (
        "minimum_suggestion_coverage",
        "maximum_candidate_wall_seconds",
        "maximum_candidate_peak_rss_mib",
    ):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if float(raw["minimum_suggestion_coverage"]) > 1:
        raise ValueError("minimum_suggestion_coverage must be at most 1")
    if not isinstance(raw["sparse_config"], dict):
        raise ValueError("sparse_config must be an object")
    # Validate overrides against the prototype's fail-closed config surface.
    effective = sparse_roi_config()
    unknown = sorted(set(raw["sparse_config"]) - set(effective))
    if unknown:
        raise ValueError(f"unknown sparse config keys: {unknown}")
    effective.update(raw["sparse_config"])
    return {**raw, "effective_sparse_config": effective}


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


def _fold_ranges(split: Any, clip_key: str, fold: str) -> list[tuple[int, int]]:
    try:
        raw_ranges = split["clips"][clip_key][fold]
        ranges = [(int(item["start_frame"]), int(item["end_frame"])) for item in raw_ranges]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"split has no valid {clip_key}/{fold} ranges") from exc
    if not ranges or any(start < 0 or end <= start for start, end in ranges):
        raise ValueError("split ranges must be non-empty half-open frame intervals")
    overlaps = any(
        right[0] < left[1] for left, right in zip(ranges, ranges[1:])
    )
    if ranges != sorted(ranges) or overlaps:
        raise ValueError("split ranges must be sorted and disjoint")
    return ranges


def _inside_ranges(frame_index: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= frame_index < end for start, end in ranges)


def _finite_visible(point: Point | None, minimum: float = 0.35) -> bool:
    return bool(
        point is not None
        and point.visibility >= minimum
        and all(math.isfinite(value) for value in (point.x, point.y, point.z))
    )


def _bound_actor_box(frame, width: int, height: int) -> list[float] | None:
    points = [point for point in frame.landmarks if _finite_visible(point)]
    if len(points) < 8:
        return None
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span_x, span_y = x1 - x0, y1 - y0
    if span_x <= 1 or span_y <= 1:
        return None
    padding = 0.08 * max(span_x, span_y)
    return [
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(float(width - 1), x1 + padding),
        min(float(height - 1), y1 + padding),
    ]


def _median(values: Sequence[float | int]) -> float | None:
    return round(float(np.median(values)), 6) if values else None


def _peak_rss_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; Darwin reports bytes.
    divisor = 1024.0 if platform.system() == "Linux" else 1024.0 * 1024.0
    return round(raw / divisor, 6)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    import cv2

    thread_budget = int(os.environ.get("AUTO_SFX_THREADS", "1"))
    cv2.setNumThreads(max(1, thread_budget))
    image_id = os.environ.get("AUTO_SFX_IMAGE_ID")
    if (
        image_id is None
        or not image_id.startswith("sha256:")
        or len(image_id) != 71
        or any(character not in "0123456789abcdef" for character in image_id[7:])
    ):
        raise ValueError("AUTO_SFX_IMAGE_ID must be the pinned lowercase SHA-256 image ID")
    paths = {
        "video": Path(args.video).resolve(),
        "tracked_pose": Path(args.tracked_pose).resolve(),
        "pose_events": Path(args.pose_events).resolve(),
        "dense_events": Path(args.dense_events).resolve(),
        "split": Path(args.split).resolve(),
        "config": Path(args.config).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} input is not a file: {path}")
    config = _validate_config(_read_json(paths["config"]))
    split = _read_json(paths["split"])
    trace = _load_trace(paths["tracked_pose"])
    if config["require_cfr"] and (not trace.meta.cfr or trace.meta.vfr):
        raise ValueError("pixel driver refuses a non-CFR trace")
    nominal_fps, probed_pts = probe_source_timestamps(paths["video"])
    trace_pts = [int(frame.source_pts_us) for frame in trace.frames]
    if probed_pts != trace_pts:
        raise ValueError("ffprobe presentation PTS do not exactly match the pose trace")
    if abs(nominal_fps - trace.meta.fps_stream) > 1e-6:
        raise ValueError("ffprobe nominal FPS does not match pose-trace metadata")

    pose_document = _read_json(paths["pose_events"])
    dense_document = _read_json(paths["dense_events"])
    pose_events = pose_document.get("events") if isinstance(pose_document, dict) else None
    dense_events = dense_document.get("events") if isinstance(dense_document, dict) else None
    if not isinstance(pose_events, list) or not isinstance(dense_events, list):
        raise ValueError("event inputs must be event-document objects")
    ranges = _fold_ranges(split, config["clip_key"], config["fold"])
    pts_to_frame = {frame.source_pts_us: frame.frame_index for frame in trace.frames}
    if len(pts_to_frame) != len(trace.frames):
        raise ValueError("trace source PTS are not unique")
    dense_by_raw_id = {
        event.get("evidence", {}).get("raw_pose_event_id"): event
        for event in dense_events
        if isinstance(event, dict)
    }

    half = int(config["window_half_frames"])
    candidates = []
    for event in pose_events:
        if not isinstance(event, dict) or event.get("kind") != "initial_contact":
            continue
        coarse = pts_to_frame.get(event.get("source_pts_us"))
        if coarse is None or not _inside_ranges(coarse, ranges):
            continue
        start, end = coarse - half, coarse + half
        if start < 0 or end >= len(trace.frames):
            continue
        # A candidate may not reach across an opened fold boundary.
        if not any(range_start <= start and end < range_end for range_start, range_end in ranges):
            continue
        candidates.append((event, coarse, start, end))

    started = time.perf_counter()
    rows = []
    source_pts_membership_failures = 0
    for event, coarse, start, end in candidates:
        window = trace.frames[start : end + 1]
        foot = event.get("foot")
        if foot not in {"left", "right"}:
            result = {
                "status": "abstain",
                "reason": "uncertain_anatomical_side",
                "suggested_frame": None,
                "suggested_pts_us": None,
            }
        else:
            heel_xy = []
            toe_xy = []
            actor_boxes = []
            for frame in window:
                observation = (frame.feet or {}).get(foot)
                heel = observation.heel if observation is not None else None
                toe = observation.toe if observation is not None else None
                heel_xy.append([heel.x, heel.y] if _finite_visible(heel) else None)
                toe_xy.append([toe.x, toe.y] if _finite_visible(toe) else None)
                actor_boxes.append(_bound_actor_box(frame, trace.meta.width, trace.meta.height))
            mapping = [(frame.frame_index, frame.source_pts_us) for frame in window]
            result = analyze_video_candidate_window(
                paths["video"],
                start,
                end,
                [item[1] for item in mapping],
                heel_xy,
                toe_xy,
                actor_boxes=actor_boxes,
                verified_frame_pts=mapping,
                config=config["sparse_config"],
            )
        suggested_frame = result.get("suggested_frame")
        suggested_pts = result.get("suggested_pts_us")
        if suggested_frame is not None:
            if not (0 <= suggested_frame < len(trace.frames)):
                source_pts_membership_failures += 1
            elif trace.frames[suggested_frame].source_pts_us != suggested_pts:
                source_pts_membership_failures += 1
        dense = dense_by_raw_id.get(event.get("id"))
        dense_frame = (
            dense.get("evidence", {}).get("roi_refined_frame")
            if isinstance(dense, dict)
            else None
        )
        rows.append({
            "raw_pose_event_id": event.get("id"),
            "foot": foot,
            "coarse_frame": coarse,
            "coarse_source_pts_us": event.get("source_pts_us"),
            "dense_frame": dense_frame,
            "dense_shift_frames": None if dense_frame is None else dense_frame - coarse,
            "sparse_status": result.get("status"),
            "sparse_reason": result.get("reason"),
            "sparse_frame": suggested_frame,
            "sparse_pts_us": suggested_pts,
            "sparse_shift_frames": (
                None if suggested_frame is None else suggested_frame - coarse
            ),
            "sparse_score": result.get("score"),
            "sparse_result": result,
        })
    candidate_wall_seconds = round(time.perf_counter() - started, 6)
    candidate_peak_rss_mib = _peak_rss_mib()

    suggested = [row for row in rows if row["sparse_status"] == "suggested"]
    reasons = Counter(row["sparse_reason"] for row in rows if row["sparse_status"] != "suggested")
    coverage = len(suggested) / len(rows) if rows else 0.0
    sparse_shifts = [row["sparse_shift_frames"] for row in suggested]
    dense_shifts = [
        row["dense_shift_frames"]
        for row in rows
        if row["dense_shift_frames"] is not None
    ]
    dense_sparse_shifts = [
        row["sparse_frame"] - row["dense_frame"]
        for row in suggested if row["dense_frame"] is not None
    ]
    gates = {
        "source_pts_membership": source_pts_membership_failures == 0,
        "minimum_suggestion_coverage": coverage
        >= float(config["minimum_suggestion_coverage"]),
        "candidate_wall": candidate_wall_seconds
        <= float(config["maximum_candidate_wall_seconds"]),
        "candidate_peak_rss": candidate_peak_rss_mib
        <= float(config["maximum_candidate_peak_rss_mib"]),
        "explicit_disposition": len(rows) == len(suggested) + sum(reasons.values()),
    }
    metrics = {
        "schema": METRICS_SCHEMA,
        "source_pts_membership_failures": source_pts_membership_failures,
        "candidate_wall_seconds": candidate_wall_seconds,
        "candidate_peak_rss_mib": candidate_peak_rss_mib,
        "candidate_count": len(rows),
        "suggested_count": len(suggested),
        "suggestion_coverage": round(coverage, 6),
        "mechanical_gate_failures": sum(not passed for passed in gates.values()),
    }
    artifact = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "evidence_boundary": config["evidence_boundary"],
        "accuracy_claim": False,
        "operator_truth_used": False,
        "detector_or_pose_inference_run": False,
        "inputs": {
            name: {"sha256": _hash_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(paths.items())
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "docker_image_id": image_id,
            "opencv_version": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "effective_sparse_config": config["effective_sparse_config"],
        },
        "source_binding": {
            "ffprobe_pts_match_trace": True,
            "frames": len(trace.frames),
            "nominal_fps": nominal_fps,
            "cfr": trace.meta.cfr,
            "vfr": trace.meta.vfr,
            "fold_ranges": [list(item) for item in ranges],
        },
        "summary": {
            **metrics,
            "abstention_reasons": dict(sorted(reasons.items(), key=lambda item: str(item[0]))),
            "median_sparse_shift_frames": _median(sparse_shifts),
            "sparse_shift_range_frames": (
                [min(sparse_shifts), max(sparse_shifts)] if sparse_shifts else None
            ),
            "median_dense_shift_frames": _median(dense_shifts),
            "median_sparse_minus_dense_frames": _median(dense_sparse_shifts),
            "gates": gates,
            "mechanism_decision": (
                "mechanically_passed_opened_diagnostic"
                if all(gates.values())
                else "rejected_or_inconclusive_opened_diagnostic"
            ),
            "decision_note": (
                "Mechanical opened-data coverage/resource result only; proximity to "
                "agent-draft marks is deliberately not measured."
            ),
        },
        "candidates": rows,
    }
    return artifact, metrics


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracked-pose", required=True)
    parser.add_argument("--pose-events", required=True)
    parser.add_argument("--dense-events", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--metrics", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    artifact_path = Path(args.artifact).resolve()
    metrics_path = Path(args.metrics).resolve()
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
