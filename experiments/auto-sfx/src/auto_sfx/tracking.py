"""Continuity-aware two-foot tracking for offline contact detection.

Pose backends expose anatomical ``left``/``right`` labels independently at
every frame.  Those labels are observations, not temporal track identities:
MediaPipe can exchange the complete leg labels for a frame while continuing
to report high landmark confidence.  This module builds two temporal tracks
first and keeps the initial anatomical attribution as separate metadata.

The tracker is deliberately small and deterministic.  It evaluates both
possible two-foot assignments against constant-velocity predictions, applies
a swap only when it wins by a scale-normalised margin, and treats motion that
is implausible under *both* assignments as a discontinuity.  Short unresolved
gaps may be interpolated offline; longer gaps remain missing so downstream
detectors must abstain.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .types import FootObservation, Point, PoseTrace, SIDES


@dataclass
class TrackingResult:
    trace: PoseTrace
    records: list[dict]
    summary: dict
    attributions: dict[str, "TemporalTrackAttribution"]


@dataclass(frozen=True)
class TemporalTrackAttribution:
    """Anatomical claim kept separate from a temporal trajectory identity."""

    track_id: str
    anatomical_side: str
    source: str
    confidence: float | None = None
    confidence_note: str = (
        "pose landmark confidence is not calibrated anatomical-side confidence"
    )


def _points(obs: FootObservation | None) -> list[Point]:
    if obs is None:
        return []
    return [p for p in (obs.heel, obs.ankle, obs.toe) if p is not None]


def _centre(obs: FootObservation | None, min_visibility: float) -> np.ndarray | None:
    pts = [p for p in _points(obs) if p.visibility >= min_visibility]
    if len(pts) < 2:
        return None
    return np.asarray([
        float(np.mean([p.x for p in pts])),
        float(np.mean([p.y for p in pts])),
    ])


def _body_scale(trace: PoseTrace, min_visibility: float) -> float:
    vals: list[float] = []
    for fr in trace.frames:
        if fr.lost or fr.hips is None:
            continue
        for side in SIDES:
            c = _centre((fr.feet or {}).get(side), min_visibility)
            if c is not None:
                vals.append(math.hypot(c[0] - fr.hips.x, c[1] - fr.hips.y))
    return float(np.median(vals)) if vals else 1.0


def _interp_point(a: Point | None, b: Point | None, u: float) -> Point | None:
    if a is None or b is None:
        return None
    return Point(
        x=(1.0 - u) * a.x + u * b.x,
        y=(1.0 - u) * a.y + u * b.y,
        z=(1.0 - u) * a.z + u * b.z,
        # Interpolated confidence must never exceed either observation.
        visibility=min(a.visibility, b.visibility),
    )


def _interp_obs(a: FootObservation | None, b: FootObservation | None,
                u: float) -> FootObservation | None:
    if a is None or b is None:
        return None
    return FootObservation(
        heel=_interp_point(a.heel, b.heel, u),
        ankle=_interp_point(a.ankle, b.ankle, u),
        toe=_interp_point(a.toe, b.toe, u),
    )


def _predict(history: list[tuple[float, np.ndarray]], t: float) -> np.ndarray | None:
    if not history:
        return None
    if len(history) == 1:
        return history[-1][1]
    t0, p0 = history[-2]
    t1, p1 = history[-1]
    dt = t1 - t0
    if dt <= 0 or t - t1 > 0.15:
        return p1
    velocity = (p1 - p0) / dt
    # A bounded prediction helps assignment without extrapolating a pose
    # glitch into several future frames.
    delta = velocity * (t - t1)
    mag = float(np.linalg.norm(delta))
    if mag > 80.0:
        delta *= 80.0 / mag
    return p1 + delta


def _assignment_cost(pred: dict[str, np.ndarray | None],
                     obs: dict[str, np.ndarray | None],
                     assignment: dict[str, str], scale: float) -> tuple[float, float]:
    costs: list[float] = []
    max_jump = 0.0
    for track_side in SIDES:
        p = pred[track_side]
        q = obs[assignment[track_side]]
        if p is None or q is None:
            costs.append(0.35)  # missing-data penalty in body-scale units
            continue
        jump = float(np.linalg.norm(q - p)) / max(scale, 1e-9)
        costs.append(jump)
        max_jump = max(max_jump, jump)
    return float(sum(costs)), max_jump


def _fill_short_gaps(frames, records: list[dict], max_gap: int) -> int:
    if max_gap <= 0:
        return 0
    filled = 0
    for side in SIDES:
        i = 0
        while i < len(frames):
            if (frames[i].feet or {}).get(side) is not None:
                i += 1
                continue
            start = i
            while i < len(frames) and (frames[i].feet or {}).get(side) is None:
                i += 1
            end = i - 1
            gap = end - start + 1
            if (gap > max_gap or start == 0 or i >= len(frames)
                    or frames[start - 1].lost or frames[i].lost):
                continue
            left = frames[start - 1].feet.get(side)
            right = frames[i].feet.get(side)
            if left is None or right is None:
                continue
            for off, j in enumerate(range(start, i), start=1):
                obs = _interp_obs(left, right, off / (gap + 1.0))
                if obs is None:
                    break
                frames[j].feet[side] = obs
                records.append({
                    "kind": "tracking", "frame": frames[j].frame_index,
                    "t_s": round(frames[j].t_s, 6), "side": side,
                    "decision": "interpolate", "gap_frames": gap,
                })
                filled += 1
    return filled


def track_feet(
    trace: PoseTrace,
    *,
    policy: str = "interpolate",
    min_visibility: float = 0.35,
    jump_speed_bs_s: float = 14.0,
    swap_margin_bs: float = 0.08,
    max_interp_gap: int = 1,
    max_swap_run_frames: int = 2,
) -> TrackingResult:
    """Return a trace whose anatomical slots follow temporal foot tracks.

    ``policy`` controls the ablation:

    - ``raw`` records discontinuities but preserves backend labels;
    - ``quarantine`` preserves labels and removes implausible observations;
    - ``swap`` enables continuity assignment and leaves unresolved gaps empty;
    - ``interpolate`` enables swaps and fills isolated one-frame gaps.
    """

    if policy not in {"raw", "quarantine", "swap", "interpolate"}:
        raise ValueError(
            "track_policy must be raw, quarantine, swap, or interpolate"
        )
    out = copy.deepcopy(trace)
    records: list[dict] = []
    scale = _body_scale(trace, min_visibility)
    histories: dict[str, list[tuple[float, np.ndarray]]] = {s: [] for s in SIDES}
    swap_count = quarantine_count = high_conf_outliers = 0
    previous_t: float | None = None

    for fr in out.frames:
        raw = dict(fr.feet or {})
        raw.setdefault("left", None)
        raw.setdefault("right", None)
        centres = {s: _centre(raw[s], min_visibility) for s in SIDES}
        if fr.lost or all(v is None for v in centres.values()):
            histories = {s: [] for s in SIDES}
            previous_t = fr.t_s
            continue

        pred = {s: _predict(histories[s], fr.t_s) for s in SIDES}
        direct = {"left": "left", "right": "right"}
        crossed = {"left": "right", "right": "left"}
        direct_cost, direct_jump = _assignment_cost(pred, centres, direct, scale)
        swap_cost, swap_jump = _assignment_cost(pred, centres, crossed, scale)
        dt = fr.t_s - previous_t if previous_t is not None else 0.0
        jump_limit = jump_speed_bs_s * max(dt, 1.0 / max(trace.meta.fps_stream, 1.0))

        have_prediction = any(pred[s] is not None for s in SIDES)
        raw_implausible = (
            have_prediction
            and direct_jump > jump_limit
        )
        crossed_plausible = swap_jump <= jump_limit
        assignment = direct
        # A cheaper crossed assignment is not sufficient: physical feet
        # routinely cross in image space.  Correct a backend label exchange
        # only when its anatomical assignment violates the motion bound and
        # the crossed assignment restores plausible continuity.
        if policy in {"swap", "interpolate"} and (
            raw_implausible and crossed_plausible
            and swap_cost + swap_margin_bs < direct_cost
        ):
            assignment = crossed

        chosen_cost, chosen_jump = (
            (swap_cost, swap_jump) if assignment is crossed
            else (direct_cost, direct_jump)
        )
        both_implausible = have_prediction and (
            direct_jump > jump_limit and swap_jump > jump_limit
        )

        if assignment is crossed and not both_implausible:
            swap_count += 1
            confs = [p.visibility for obs in raw.values() for p in _points(obs)]
            if confs and float(np.mean(confs)) >= 0.8 and raw_implausible:
                high_conf_outliers += 1
            records.append({
                "kind": "tracking", "frame": fr.frame_index,
                "t_s": round(fr.t_s, 6), "decision": "swap",
                "temporal_tracks": {
                    "track_0": {"anatomical_side": "left", "observed_label": "right"},
                    "track_1": {"anatomical_side": "right", "observed_label": "left"},
                },
                "direct_cost_bs": round(direct_cost, 4),
                "swap_cost_bs": round(swap_cost, 4),
                "direct_max_jump_bs": round(direct_jump, 4),
                "swap_max_jump_bs": round(swap_jump, 4),
            })

        quarantine = both_implausible or (
            policy == "quarantine" and raw_implausible
        )
        if quarantine and policy != "raw":
            quarantine_count += 1
            confs = [p.visibility for obs in raw.values() for p in _points(obs)]
            high_conf = bool(confs and float(np.mean(confs)) >= 0.8)
            high_conf_outliers += int(high_conf)
            fr.feet = {s: None for s in SIDES}
            records.append({
                "kind": "tracking", "frame": fr.frame_index,
                "t_s": round(fr.t_s, 6), "decision": "quarantine",
                "reason": "implausible_under_both_assignments",
                "high_confidence": high_conf,
                "direct_cost_bs": round(direct_cost, 4),
                "swap_cost_bs": round(swap_cost, 4),
                "max_jump_bs": round(chosen_jump, 4),
                "jump_limit_bs": round(jump_limit, 4),
            })
            histories = {s: [] for s in SIDES}
            previous_t = fr.t_s
            continue

        if policy == "raw":
            assignment = direct
        fr.feet = {s: raw[assignment[s]] for s in SIDES}
        for side in SIDES:
            c = _centre(fr.feet.get(side), min_visibility)
            if c is not None:
                histories[side].append((fr.t_s, c))
                histories[side] = histories[side][-2:]
        previous_t = fr.t_s

    # Long crossed-assignment episodes are not safely attributable from
    # continuity alone (especially in stateless image-mode traces).  Preserve
    # the observation in records but abstain instead of silently relabelling
    # anatomy across a whole gait phase.
    swap_frames = sorted(
        r["frame"] for r in records if r.get("decision") == "swap"
    )
    long_swap_frames: list[int] = []
    for start in range(len(swap_frames)):
        if start and swap_frames[start] == swap_frames[start - 1] + 1:
            continue
        end = start
        while (end + 1 < len(swap_frames)
               and swap_frames[end + 1] == swap_frames[end] + 1):
            end += 1
        group = swap_frames[start:end + 1]
        if len(group) > max_swap_run_frames:
            long_swap_frames.extend(group)
    by_index = {fr.frame_index: fr for fr in out.frames}
    for frame_index in long_swap_frames:
        fr = by_index[frame_index]
        fr.feet = {s: None for s in SIDES}
        records.append({
            "kind": "tracking", "frame": frame_index,
            "t_s": round(fr.t_s, 6), "decision": "quarantine_swap_run",
            "reason": "crossed_assignment_exceeds_safe_duration",
            "max_swap_run_frames": max_swap_run_frames,
        })
        quarantine_count += 1

    interpolated = 0
    if policy == "interpolate":
        interpolated = _fill_short_gaps(out.frames, records, max_interp_gap)
    records.sort(key=lambda r: (r.get("frame", -1), r.get("side", "")))
    summary = {
        "policy": policy,
        "scale_px": round(scale, 4),
        "proposed_swap_frames": swap_count,
        "swap_frames": swap_count - len(long_swap_frames),
        "accepted_swap_frames": swap_count - len(long_swap_frames),
        "quarantined_frames": quarantine_count,
        "interpolated_foot_frames": interpolated,
        "abstained_long_swap_frames": len(long_swap_frames),
        "high_confidence_outliers": high_conf_outliers,
        "jump_speed_bs_s": float(jump_speed_bs_s),
        "swap_margin_bs": float(swap_margin_bs),
        "max_swap_run_frames": int(max_swap_run_frames),
        "anatomical_attribution": (
            "track_0/track_1 inherit left/right anatomical attribution at "
            "each continuous-segment seed; observed backend labels may be "
            "reassigned only across an implausible one-frame discontinuity"
        ),
        "temporal_tracks": {
            "track_0": {"anatomical_side": "left", "seed_source": "backend_label"},
            "track_1": {"anatomical_side": "right", "seed_source": "backend_label"},
        },
    }
    attributions = {
        "track_0": TemporalTrackAttribution(
            "track_0", "left", "backend label at continuous-segment seed"
        ),
        "track_1": TemporalTrackAttribution(
            "track_1", "right", "backend label at continuous-segment seed"
        ),
    }
    return TrackingResult(
        trace=out, records=records, summary=summary,
        attributions=attributions,
    )
