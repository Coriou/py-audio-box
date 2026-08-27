#!/usr/bin/env python3
"""Compare opened full-leg identity mechanics without running pose inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from auto_sfx.leg_tracking import track_leg_chains
from auto_sfx.tracking import track_feet
from auto_sfx.types import PoseTrace, VideoMeta


SCHEMA = "auto-sfx-phase2c-b1-identity-diagnostic-v1"
METRICS_SCHEMA = "auto-sfx-phase2c-b1-public-metrics-v1"
CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "evidence_boundary",
    "known_opened_exchange",
    "maximum_candidate_wall_seconds",
    "maximum_candidate_peak_rss_mib",
    "candidate_config",
    "baseline_config",
}
KNOWN_KEYS = {
    "trace_name",
    "frame",
    "required_candidate_state",
    "required_neighbour_state",
}
CANDIDATE_KEYS = {
    "shot_boundary_frames",
    "min_visibility",
    "min_valid_joints",
    "max_missing_gap_frames",
    "switch_penalty",
    "seed_swap_penalty",
    "missing_penalty",
    "max_prediction_step_bs",
    "max_reassignment_run_frames",
    "attribution_margin_threshold",
}
BASELINE_KEYS = {
    "policy",
    "min_visibility",
    "jump_speed_bs_s",
    "swap_margin_bs",
    "max_interp_gap",
    "max_swap_run_frames",
}


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


def _validate_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        missing = sorted(CONFIG_KEYS - set(raw)) if isinstance(raw, dict) else []
        extra = sorted(set(raw) - CONFIG_KEYS) if isinstance(raw, dict) else []
        raise ValueError(f"config keys are not exact; missing={missing}, extra={extra}")
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise ValueError("config schema_version must be 1")
    known = raw["known_opened_exchange"]
    if not isinstance(known, dict) or set(known) != KNOWN_KEYS:
        raise ValueError("known_opened_exchange keys are not exact")
    if (
        not isinstance(known["frame"], int)
        or isinstance(known["frame"], bool)
        or known["frame"] < 1
    ):
        raise ValueError("known opened frame must be a positive integer")
    if known["required_candidate_state"] not in {
        "direct",
        "swapped",
        "distal_swapped",
        "proximal_swapped",
    }:
        raise ValueError("unsupported required candidate state")
    if known["required_neighbour_state"] != "direct":
        raise ValueError("known exchange neighbours must remain direct")
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
    if raw["baseline_config"].get("policy", "interpolate") not in {
        "raw",
        "quarantine",
        "swap",
        "interpolate",
    }:
        raise ValueError("unsupported baseline tracking policy")
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
    failures = 0
    for record in records:
        frame = record.get("frame")
        if frame not in expected or record.get("source_pts_us") != expected[frame]:
            failures += 1
    return failures


def _extract_candidate(
    trace: PoseTrace,
    config: dict[str, Any],
    known_frame: int | None,
) -> tuple[dict[str, Any], float, str]:
    started = time.perf_counter()
    result = track_leg_chains(trace, **config)
    elapsed = time.perf_counter() - started
    records_hash = _hash_json(result.records)
    by_frame = {row["frame"]: row for row in result.records}
    neighborhood = []
    if known_frame is not None:
        neighborhood = [
            by_frame[frame]
            for frame in range(known_frame - 2, known_frame + 3)
            if frame in by_frame
        ]
    component_changed = [
        row["frame"]
        for row in result.records
        if row.get("assignment_state") not in {None, "direct"}
    ]
    unsafe = [
        row["frame"]
        for row in result.records
        if row.get("whole_chain_status") != "coherent_observed"
    ]
    summary = {
        "summary": result.summary,
        "records_sha256": records_hash,
        "source_pts_membership_failures": _source_pts_failures(trace, result.records),
        "changed_component_frames": component_changed,
        "unsafe_whole_chain_frames": unsafe,
        "known_neighborhood": neighborhood,
    }
    return summary, elapsed, records_hash


def _extract_baseline(
    trace: PoseTrace,
    config: dict[str, Any],
    known_frame: int | None,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = track_feet(trace, **config)
    elapsed = time.perf_counter() - started
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in result.records:
        by_frame.setdefault(row.get("frame", -1), []).append(row)
    return {
        "summary": result.summary,
        "records_sha256": _hash_json(result.records),
        "known_frame_records": by_frame.get(known_frame, []) if known_frame is not None else [],
    }, elapsed


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    image_id = os.environ.get("AUTO_SFX_IMAGE_ID")
    if (
        image_id is None
        or not image_id.startswith("sha256:")
        or len(image_id) != 71
        or any(character not in "0123456789abcdef" for character in image_id[7:])
    ):
        raise ValueError("AUTO_SFX_IMAGE_ID must be the pinned lowercase SHA-256 image ID")
    paths = {
        "walk1": Path(args.walk1_trace).resolve(),
        "walk2": Path(args.walk2_trace).resolve(),
        "split": Path(args.split).resolve(),
        "config": Path(args.config).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} input is not a file: {path}")
    config = _validate_config(_read_json(paths["config"]))
    # The split is provenance only; every inherited shard is explicitly opened.
    split = _read_json(paths["split"])
    if not isinstance(split, dict) or split.get("schema_version") != 1:
        raise ValueError("opened Phase 2B split is not schema version 1")

    known = config["known_opened_exchange"]
    results: dict[str, Any] = {}
    candidate_wall = 0.0
    baseline_wall = 0.0
    deterministic_failures = 0
    source_pts_failures = 0
    for name in ("walk1", "walk2"):
        trace = _load_trace(paths[name])
        known_frame = known["frame"] if name == known["trace_name"] else None
        baseline, baseline_elapsed = _extract_baseline(
            trace, config["baseline_config"], known_frame
        )
        candidate, candidate_elapsed, first_hash = _extract_candidate(
            trace, config["candidate_config"], known_frame
        )
        repeated = track_leg_chains(trace, **config["candidate_config"])
        repeated_hash = _hash_json(repeated.records)
        deterministic_failures += int(first_hash != repeated_hash)
        source_pts_failures += candidate["source_pts_membership_failures"]
        baseline_wall += baseline_elapsed
        candidate_wall += candidate_elapsed
        results[name] = {
            "frames": len(trace.frames),
            "backend": trace.backend,
            "mode": trace.mode,
            "baseline": baseline,
            "candidate": candidate,
            "baseline_wall_seconds": round(baseline_elapsed, 6),
            "candidate_wall_seconds": round(candidate_elapsed, 6),
            "repeat_records_sha256": repeated_hash,
        }
        del repeated, trace
        gc.collect()

    known_rows = results[known["trace_name"]]["candidate"]["known_neighborhood"]
    known_by_frame = {row["frame"]: row for row in known_rows}
    target = int(known["frame"])
    known_exchange_failures = int(
        known_by_frame.get(target, {}).get("assignment_state")
        != known["required_candidate_state"]
    )
    neighbour_regression_failures = sum(
        known_by_frame.get(frame, {}).get("assignment_state")
        != known["required_neighbour_state"]
        for frame in (target - 1, target + 1)
    )
    unsafe_frames = sum(
        result["candidate"]["summary"]["unsafe_whole_chain_frames"]
        for result in results.values()
    )
    total_frames = sum(result["frames"] for result in results.values())
    whole_chain_rejections = sum(
        result["candidate"]["summary"]["whole_chain_claim"]["status"] == "rejected"
        for result in results.values()
    )
    peak_rss = _peak_rss_mib()
    candidate_wall = round(candidate_wall, 6)
    baseline_wall = round(baseline_wall, 6)
    gates = {
        "known_exchange": known_exchange_failures == 0,
        "known_neighbours": neighbour_regression_failures == 0,
        "source_pts_membership": source_pts_failures == 0,
        "deterministic_repeat": deterministic_failures == 0,
        "candidate_wall": candidate_wall
        <= float(config["maximum_candidate_wall_seconds"]),
        "candidate_peak_rss": peak_rss
        <= float(config["maximum_candidate_peak_rss_mib"]),
        "general_whole_chain_claim": whole_chain_rejections == 0 and unsafe_frames == 0,
    }
    metrics = {
        "schema": METRICS_SCHEMA,
        "known_exchange_failures": known_exchange_failures,
        "neighbour_regression_failures": neighbour_regression_failures,
        "source_pts_membership_failures": source_pts_failures,
        "deterministic_failures": deterministic_failures,
        "candidate_wall_seconds": candidate_wall,
        "baseline_wall_seconds": baseline_wall,
        "candidate_peak_rss_mib": peak_rss,
        "total_frames": total_frames,
        "unsafe_whole_chain_frames": unsafe_frames,
        "whole_chain_rejections": whole_chain_rejections,
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
        },
        "summary": {
            **metrics,
            "gates": gates,
            "unsafe_whole_chain_fraction": round(unsafe_frames / total_frames, 6),
            "mechanism_decision": (
                "mechanically_passed_opened_diagnostic"
                if all(gates.values())
                else "rejected_opened_mechanism"
            ),
            "decision_note": (
                "Opened mechanism diagnostics only. The known frame was used for repair, "
                "and no identity or anatomical accuracy is claimed."
            ),
        },
        "traces": results,
    }
    return artifact, metrics


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk1-trace", required=True)
    parser.add_argument("--walk2-trace", required=True)
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
