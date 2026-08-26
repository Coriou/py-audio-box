"""Versioned contact-state detector used by the Phase-2B experiments.

Unlike ``hybrid-v0``, this detector does not timestamp the beginning of an
arbitrarily long low-speed run.  It estimates per-frame swing/uncertain/
grounded evidence from separate heel and toe trajectories and retrospectively
emits the change point into a sustained grounded episode.  All statistics are
bounded, centred windows; no historical maximum survives for the shot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tracking import track_feet
from .types import (
    DOMINANT_ACTOR_ID,
    EVENT_FAMILY_FOOT_CONTACT,
    ContactEvent,
    DiagReason,
    SIDES,
    s_to_pts_us,
)


STATE_METHOD = "contact-state-v1"


STATE_CONFIG = {
    "min_step_interval_s": 0.25,
    "max_step_interval_s": 2.0,
    "min_landmark_visibility": 0.5,
    "shot_boundaries": [],
    "actor_too_small_px": 24.0,
    "camera_motion_std_px": 12.0,
    "track_policy": "interpolate",
    "track_jump_speed_bs_s": 14.0,
    "track_swap_margin_bs": 0.08,
    "track_max_interp_gap": 1,
    "track_max_swap_run_frames": 2,
    "smoothing": "raw",
    "local_window_s": 1.2,
    "min_grounded_frames": 3,
    "grounded_score": 0.56,
    "support_height_frac": 0.09,
    "phase_min_velocity_bs_s": 0.08,
    "phase_peak_radius_frames": 3,
    "min_same_foot_cycle_s": 0.45,
}


def default_state_config() -> dict:
    return dict(STATE_CONFIG)


def validate_state_config(config: dict | None) -> dict:
    cfg = default_state_config()
    if config:
        unknown = sorted(set(config) - set(cfg))
        if unknown:
            raise ValueError(f"unknown contact-config keys: {unknown}")
        cfg.update(config)
    for key in (
        "min_step_interval_s", "max_step_interval_s",
        "min_landmark_visibility", "actor_too_small_px",
        "camera_motion_std_px", "track_jump_speed_bs_s",
        "track_swap_margin_bs", "local_window_s", "grounded_score",
        "support_height_frac", "phase_min_velocity_bs_s",
        "min_same_foot_cycle_s",
    ):
        cfg[key] = float(cfg[key])
    cfg["track_max_interp_gap"] = max(0, int(cfg["track_max_interp_gap"]))
    cfg["track_max_swap_run_frames"] = max(
        1, int(cfg["track_max_swap_run_frames"])
    )
    cfg["min_grounded_frames"] = max(2, int(cfg["min_grounded_frames"]))
    cfg["phase_peak_radius_frames"] = max(
        2, int(cfg["phase_peak_radius_frames"])
    )
    cfg["shot_boundaries"] = sorted(float(v) for v in cfg["shot_boundaries"])
    if cfg["smoothing"] not in {"raw", "light", "zero_phase"}:
        raise ValueError("smoothing must be raw, light, or zero_phase")
    return cfg


@dataclass
class _Candidate:
    side: str
    idx: int
    earliest_idx: int
    latest_idx: int
    score: float
    evidence: dict


def _finite_point(point, min_vis: float) -> bool:
    return bool(
        point is not None
        and point.visibility >= min_vis
        and np.isfinite(point.x)
        and np.isfinite(point.y)
    )


def _extract_points(trace, min_vis: float) -> dict:
    n = len(trace.frames)
    data = {
        side: {
            name: {
                "x": np.full(n, np.nan),
                "y": np.full(n, np.nan),
                "conf": np.full(n, np.nan),
            }
            for name in ("heel", "toe")
        }
        for side in SIDES
    }
    for i, fr in enumerate(trace.frames):
        if fr.lost:
            continue
        for side in SIDES:
            obs = (fr.feet or {}).get(side)
            if obs is None:
                continue
            for name in ("heel", "toe"):
                point = getattr(obs, name)
                if point is None:
                    continue
                data[side][name]["conf"][i] = float(point.visibility)
                if _finite_point(point, min_vis):
                    data[side][name]["x"][i] = float(point.x)
                    data[side][name]["y"][i] = float(point.y)
    return data


def _light_filter(x: np.ndarray, y: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Replace isolated spatial impulses only; ordinary gait is untouched."""

    ox, oy = x.copy(), y.copy()
    replaced = 0
    for i in range(1, len(x) - 1):
        if not all(np.isfinite(v) for v in (x[i - 1], y[i - 1], x[i], y[i],
                                             x[i + 1], y[i + 1])):
            continue
        neighbour_gap = math.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1])
        mx = 0.5 * (x[i - 1] + x[i + 1])
        my = 0.5 * (y[i - 1] + y[i + 1])
        impulse = math.hypot(x[i] - mx, y[i] - my)
        if neighbour_gap <= 0.10 * scale and impulse > 0.14 * scale:
            ox[i], oy[i] = mx, my
            replaced += 1
    return ox, oy, replaced


def _zero_phase_filter(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0])
    out = []
    for arr in (x, y):
        dst = arr.copy()
        for i in range(len(arr)):
            lo, hi = max(0, i - 2), min(len(arr), i + 3)
            vals = arr[lo:hi]
            w = weights[lo - i + 2:hi - i + 2]
            good = np.isfinite(vals)
            if good.sum() >= 2:
                dst[i] = float(np.sum(vals[good] * w[good]) / np.sum(w[good]))
        out.append(dst)
    return out[0], out[1]


def _smooth(data: dict, mode: str, scale: float) -> int:
    replaced = 0
    for side in SIDES:
        for name in ("heel", "toe"):
            arr = data[side][name]
            if mode == "light":
                arr["x"], arr["y"], n = _light_filter(arr["x"], arr["y"], scale)
                replaced += n
            elif mode == "zero_phase":
                arr["x"], arr["y"] = _zero_phase_filter(arr["x"], arr["y"])
    return replaced


def _rolling_quantile(values: np.ndarray, half: int, q: float,
                      fallback: float) -> np.ndarray:
    out = np.full(len(values), fallback, dtype=float)
    for i in range(len(values)):
        win = values[max(0, i - half):min(len(values), i + half + 1)]
        win = win[np.isfinite(win)]
        if win.size:
            out[i] = float(np.quantile(win, q))
    return out


def _kinematics(x: np.ndarray, y: np.ndarray, times: np.ndarray,
                scale: float) -> dict:
    n = len(x)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    acc = np.full(n, np.nan)
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 0 or not all(np.isfinite(v) for v in (x[i - 1], y[i - 1], x[i], y[i])):
            continue
        vx[i] = (x[i] - x[i - 1]) / dt / scale
        vy[i] = (y[i] - y[i - 1]) / dt / scale
        speed[i] = math.hypot(vx[i], vy[i])
    for i in range(2, n):
        dt = times[i] - times[i - 1]
        if dt > 0 and np.isfinite(speed[i]) and np.isfinite(speed[i - 1]):
            acc[i] = (speed[i] - speed[i - 1]) / dt
    return {"vx": vx, "vy": vy, "speed": speed, "acc": acc}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(mask) and mask[i + 1]:
            i += 1
        out.append((start, i))
        i += 1
    return out


def _point_onset(stable: np.ndarray, lo: int, hi: int, persistence: int) -> int | None:
    for i in range(lo, hi + 1):
        end = min(len(stable), i + persistence + 2)
        if stable[i] and int(np.sum(stable[i:end])) >= persistence:
            return i
    return None


def _r(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 5)


def _state_candidates(data: dict, times: np.ndarray, seg: np.ndarray,
                      side: str, scale: float, cfg: dict,
                      records: list[dict],
                      invalid_contact: np.ndarray) -> tuple[list[_Candidate], list[dict]]:
    n = len(times)
    med_dt = float(np.median(np.diff(times[seg]))) if len(seg) > 1 else 1 / 30.0
    half = max(2, int(round(0.5 * cfg["local_window_s"] / max(med_dt, 1e-6))))
    support_tol = cfg["support_height_frac"]
    point = {}
    for name in ("heel", "toe"):
        arr = data[side][name]
        kin = _kinematics(arr["x"], arr["y"], times, scale)
        local_swing = _rolling_quantile(kin["speed"], half, 0.75, 0.8)
        # A bounded local threshold cannot be poisoned by an old spike.
        speed_thr = np.clip(0.30 * local_swing, 0.16, 0.85)
        floor_y = _rolling_quantile(arr["y"], half, 0.90, np.nan)
        height = (floor_y - arr["y"]) / scale
        speed_score = np.clip(1.0 - kin["speed"] / np.maximum(speed_thr, 1e-6), 0.0, 1.0)
        support_score = np.clip(1.0 - height / support_tol, 0.0, 1.0)
        score = 0.58 * speed_score + 0.42 * support_score
        stable = (
            np.isfinite(arr["x"]) & np.isfinite(arr["y"])
            & np.isfinite(kin["speed"])
            & (score >= cfg["grounded_score"])
            & (height <= 1.15 * support_tol)
        )
        point[name] = {
            **arr, **kin, "local_swing": local_swing,
            "speed_thr": speed_thr, "height": height,
            "score": score, "stable": stable,
        }

    valid = np.isfinite(point["heel"]["x"]) & np.isfinite(point["toe"]["x"])
    contact_valid = valid & ~invalid_contact
    max_score = np.maximum(point["heel"]["score"], point["toe"]["score"])
    min_score = np.minimum(point["heel"]["score"], point["toe"]["score"])
    base_grounded = contact_valid & (max_score >= cfg["grounded_score"]) & (min_score >= 0.22)
    height_max = np.maximum(point["heel"]["height"], point["toe"]["height"])
    speed_ratio = np.maximum(
        point["heel"]["speed"] / np.maximum(point["heel"]["speed_thr"], 1e-6),
        point["toe"]["speed"] / np.maximum(point["toe"]["speed_thr"], 1e-6),
    )
    # Speed alone is not swing: planted-pose jitter can be fast.  A foot must
    # also be visibly above its bounded local support level.
    strong_swing = contact_valid & (height_max > 0.06) & (speed_ratio > 1.20)

    seg_mask = np.zeros(n, dtype=bool)
    seg_mask[seg] = True
    base_grounded &= seg_mask
    strong_swing &= seg_mask
    state = np.full(n, "UNCERTAIN", dtype=object)

    # Offline hysteresis: sustained grounded evidence opens stance, and
    # sustained elevated+moving evidence closes it.  Isolated jitter within a
    # stance therefore cannot manufacture another initial contact.
    mode = "UNKNOWN"
    missing_run = 0
    for pos, i in enumerate(seg):
        if not contact_valid[i]:
            state[i] = "UNCERTAIN"
            missing_run += 1
            if missing_run > 1:
                mode = "UNKNOWN"
            continue
        missing_run = 0
        future = seg[pos:min(len(seg), pos + 5)]
        swing_confirmed = int(np.sum(strong_swing[future[:3]])) >= 2
        ground_confirmed = (
            not strong_swing[i]
            and int(np.sum(base_grounded[future])) >= cfg["min_grounded_frames"]
        )
        if mode == "GROUNDED":
            if swing_confirmed:
                mode = "SWING"
                state[i] = "SWING"
            else:
                state[i] = "GROUNDED"
        elif ground_confirmed:
            mode = "GROUNDED"
            state[i] = "GROUNDED"
        elif swing_confirmed or mode == "SWING":
            mode = "SWING"
            state[i] = "SWING"

    grounded = state == "GROUNDED"
    swing = state == "SWING"

    min_frames = cfg["min_grounded_frames"]
    candidates: list[_Candidate] = []
    toe_offs: list[dict] = []
    for start, end in _runs(grounded[seg]):
        start = int(seg[0]) + start
        end = int(seg[0]) + end
        if end - start + 1 < min_frames:
            state[start:end + 1] = "UNCERTAIN"
            continue
        prior_lo = max(int(seg[0]), start - max(3, int(round(0.8 / med_dt))))
        prior_hi = max(prior_lo, start - 1)
        if int(np.sum(swing[prior_lo:prior_hi + 1])) < 2:
            # Seeded stance: its contact precedes observable evidence.
            continue
        search_lo = max(prior_lo, start - 3)
        search_hi = min(end, start + 2)
        onsets = {
            name: _point_onset(
                point[name]["stable"] & contact_valid,
                search_lo, search_hi, min_frames,
            )
            for name in ("heel", "toe")
        }
        onset_vals = [v for v in onsets.values() if v is not None]
        if not onset_vals:
            onset_vals = [start]
        earliest = min(onset_vals)
        latest = max(onset_vals)

        def onset_strength(name: str, idx: int | None) -> float:
            if idx is None:
                return -1.0
            p = point[name]
            pre = p["speed"][max(prior_lo, idx - 3):idx]
            post = p["speed"][idx:min(end + 1, idx + 3)]
            pre = pre[np.isfinite(pre)]
            post = post[np.isfinite(post)]
            drop = 0.0
            if pre.size and post.size:
                drop = max(0.0, float(np.max(pre) - np.median(post)))
                drop /= max(float(p["local_swing"][idx]), 0.2)
            vy_pre = p["vy"][max(prior_lo, idx - 2):idx]
            down = float(np.max(vy_pre[np.isfinite(vy_pre)])) if np.isfinite(vy_pre).any() else 0.0
            return float(p["score"][idx]) + 0.30 * min(drop, 1.0) + 0.15 * min(max(down, 0.0), 1.0)

        strengths = {name: onset_strength(name, idx) for name, idx in onsets.items()}
        strike = max(strengths, key=strengths.get)
        preferred = onsets[strike] if onsets[strike] is not None else earliest
        # A heel/toe disagreement is real uncertainty, not something to average away.
        earliest = min(earliest, preferred)
        latest = max(latest, preferred)
        score = min(1.0, max(0.0, strengths[strike] / 1.45))
        candidates.append(_Candidate(
            side=side, idx=preferred, earliest_idx=earliest,
            latest_idx=latest, score=score,
            evidence={
                "state_change": "swing_or_uncertain_to_sustained_grounded",
                "strike_region": strike,
                "heel_onset_frame": onsets["heel"],
                "toe_onset_frame": onsets["toe"],
                "heel_speed_bs_s": _r(point["heel"]["speed"][preferred]),
                "toe_speed_bs_s": _r(point["toe"]["speed"][preferred]),
                "heel_speed_threshold_bs_s": _r(point["heel"]["speed_thr"][preferred]),
                "toe_speed_threshold_bs_s": _r(point["toe"]["speed_thr"][preferred]),
                "heel_support_height_bs": _r(point["heel"]["height"][preferred]),
                "toe_support_height_bs": _r(point["toe"]["height"][preferred]),
                "heel_grounded_score": _r(point["heel"]["score"][preferred]),
                "toe_grounded_score": _r(point["toe"]["score"][preferred]),
                "grounded_run_frames": end - start + 1,
                "grounded_run_s": round(times[end] - times[start] + med_dt, 5),
            },
        ))
        if end + 1 < len(state) and swing[end + 1]:
            toe_offs.append({
                "kind": "toe_off_internal", "side": side,
                "frame": end + 1, "t_s": round(float(times[end + 1]), 6),
                "grounded_start_frame": start,
            })

    for i in seg:
        prev = "UNKNOWN" if i == seg[0] else str(state[i - 1])
        records.append({
            "kind": "frame", "method": STATE_METHOD, "side": side,
            "frame": int(i), "t_s": round(float(times[i]), 6),
            "pts_us": int(round(float(times[i]) * 1_000_000)),
            "state_before": prev, "state_after": str(state[i]),
            "speed_bs_s": _r(max(point["heel"]["speed"][i], point["toe"]["speed"][i])),
            "heel_speed_bs_s": _r(point["heel"]["speed"][i]),
            "toe_speed_bs_s": _r(point["toe"]["speed"][i]),
            "heel_vx_bs_s": _r(point["heel"]["vx"][i]),
            "heel_vy_bs_s": _r(point["heel"]["vy"][i]),
            "toe_vx_bs_s": _r(point["toe"]["vx"][i]),
            "toe_vy_bs_s": _r(point["toe"]["vy"][i]),
            "heel_acc_bs_s2": _r(point["heel"]["acc"][i]),
            "toe_acc_bs_s2": _r(point["toe"]["acc"][i]),
            "heel_support_height_bs": _r(point["heel"]["height"][i]),
            "toe_support_height_bs": _r(point["toe"]["height"][i]),
            "heel_grounded_score": _r(point["heel"]["score"][i]),
            "toe_grounded_score": _r(point["toe"]["score"][i]),
            "heel_threshold_bs_s": _r(point["heel"]["speed_thr"][i]),
            "toe_threshold_bs_s": _r(point["toe"]["speed_thr"][i]),
        })
    records.extend(toe_offs)
    return candidates, toe_offs


def _fill_short_gaps(values: np.ndarray, max_gap: int = 1) -> np.ndarray:
    """Fill only bounded gaps for derivatives; never bridge pose-loss runs."""

    out = values.copy()
    good = np.flatnonzero(np.isfinite(out))
    for left, right in zip(good, good[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= max_gap:
            out[left:right + 1] = np.linspace(out[left], out[right], gap + 2)
    return out


def _relative_phase_candidates(
    data: dict,
    body_x: np.ndarray,
    times: np.ndarray,
    seg: np.ndarray,
    side: str,
    scale: float,
    cfg: dict,
    records: list[dict],
    invalid_contact: np.ndarray,
) -> tuple[list[_Candidate], list[dict]]:
    """Infer contact from forward swing -> camera-relative stance reversal.

    A planted foot is not stationary in a camera-following shot.  Absolute
    image speed therefore cannot define grounded state.  The bounded signal
    here is each heel/toe's horizontal position relative to the pelvis.  A
    defensible initial contact needs (1) sustained forward motion before a
    local advance apex, (2) sustained rearward motion after it, (3) low heel
    or toe support height, and (4) several subsequent supported frames.

    Heel and toe reversals stay separate.  Their disagreement becomes the
    emitted uncertainty window; heel is preferred for a heel-first landing.
    """

    n = len(times)
    if len(seg) < 9:
        return [], []
    med_dt = float(np.median(np.diff(times[seg])))
    med_dt = max(med_dt, 1e-6)
    half = max(2, int(round(0.5 * cfg["local_window_s"] / med_dt)))
    radius = int(cfg["phase_peak_radius_frames"])
    min_v = float(cfg["phase_min_velocity_bs_s"])
    support_tol = float(cfg["support_height_frac"])
    min_grounded = int(cfg["min_grounded_frames"])

    point: dict[str, dict] = {}
    for name in ("heel", "toe"):
        arr = data[side][name]
        kin = _kinematics(arr["x"], arr["y"], times, scale)
        rel_x = arr["x"] - body_x
        rel_filled = _fill_short_gaps(rel_x)
        rel_vx = np.full(n, np.nan)
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if dt > 0 and np.isfinite(rel_filled[i - 1:i + 1]).all():
                rel_vx[i] = (rel_filled[i] - rel_filled[i - 1]) / dt / scale
        floor_y = _rolling_quantile(arr["y"], half, 0.92, np.nan)
        height = (floor_y - arr["y"]) / scale
        local_swing = _rolling_quantile(kin["speed"], half, 0.75, 0.8)
        speed_thr = np.clip(0.30 * local_swing, 0.16, 0.85)
        speed_score = np.clip(
            1.0 - kin["speed"] / np.maximum(speed_thr, 1e-6), 0.0, 1.0
        )
        support_score = np.clip(1.0 - height / support_tol, 0.0, 1.0)
        point[name] = {
            **arr,
            **kin,
            "rel_x": rel_x,
            "rel_filled": rel_filled,
            "rel_vx": rel_vx,
            "height": height,
            "local_swing": local_swing,
            "speed_thr": speed_thr,
            "score": 0.45 * speed_score + 0.55 * support_score,
        }

    valid = (
        np.isfinite(point["heel"]["rel_x"])
        & np.isfinite(point["toe"]["rel_x"])
        & ~invalid_contact
    )
    seg_start, seg_end = int(seg[0]), int(seg[-1])
    seg_mask = np.zeros(n, dtype=bool)
    seg_mask[seg] = True
    valid &= seg_mask

    def reversal_rows(name: str) -> list[dict]:
        p = point[name]
        rows: list[dict] = []
        advance_floor = _rolling_quantile(
            p["rel_filled"], half, 0.65, 0.0
        )
        lo = max(seg_start + radius + 3, 6)
        hi = min(seg_end - radius - 5, n - 7)
        for i in range(lo, hi + 1):
            if not valid[i]:
                continue
            # Both velocity lobes and the persistence interval are evidence
            # for the change point.  A valid apex immediately before pose
            # loss is therefore an abstention, not a contact.
            if not valid[i - 5:i + 6].all():
                continue
            window = p["rel_filled"][i - radius:i + radius + 1]
            if not np.isfinite(window).all() or p["rel_filled"][i] < np.max(window):
                continue
            # A new landing is the locally forward foot's apex.  Use a
            # bounded local percentile rather than image x=0: a tracker may
            # legitimately seed both feet behind the pelvis origin.
            if p["rel_filled"][i] < advance_floor[i]:
                continue
            pre = p["rel_vx"][i - 5:i]
            post = p["rel_vx"][i + 1:i + 6]
            pre = pre[np.isfinite(pre)]
            post = post[np.isfinite(post)]
            if pre.size < 3 or post.size < 3:
                continue
            pre_v, post_v = float(np.median(pre)), float(np.median(post))
            if pre_v <= min_v or post_v >= -min_v:
                continue
            support = (
                np.isfinite(p["height"])
                & (p["height"] <= 1.15 * support_tol)
            )
            support_hi = min(n, i + max(5, min_grounded + 2))
            if int(np.sum(support[i:support_hi] & valid[i:support_hi])) < min_grounded:
                continue
            if invalid_contact[max(seg_start, i - radius):min(n, i + radius + 1)].any():
                continue
            prominence = (
                p["rel_filled"][i]
                - min(
                    float(np.min(p["rel_filled"][i - 6:i])),
                    float(np.min(p["rel_filled"][i + 1:i + 7])),
                )
            ) / scale
            rows.append({
                "frame": i,
                "pre_v": pre_v,
                "post_v": post_v,
                "prominence": prominence,
                "strength": pre_v - post_v + max(0.0, prominence),
                "advance": float(p["rel_filled"][i] / scale),
            })

        # Collapse jitter peaks within a bounded fraction of one same-foot
        # cycle.  This removes no evidence: the strongest local reversal and
        # all rejected neighbours are retained in the trace record.
        observed_gaps = np.diff([r["frame"] for r in rows])
        positive_gaps = observed_gaps[observed_gaps > 0]
        local_cycle_floor = (
            int(round(0.55 * float(np.median(positive_gaps))))
            if positive_gaps.size >= 3 else 0
        )
        min_gap = max(
            2,
            int(round(cfg["min_same_foot_cycle_s"] / med_dt)),
            local_cycle_floor,
        )
        kept: list[dict] = []
        for row in rows:
            if kept and row["frame"] - kept[-1]["frame"] < min_gap:
                winner = max((kept[-1], row), key=lambda r: r["advance"])
                loser = row if winner is kept[-1] else kept[-1]
                records.append({
                    "kind": "phase_reversal_rejected",
                    "method": STATE_METHOD,
                    "side": side,
                    "point": name,
                    "frame": int(loser["frame"]),
                    "decision": "bounded_same_foot_dedup",
                    "kept_frame": int(winner["frame"]),
                })
                kept[-1] = winner
            else:
                kept.append(row)
        return kept

    heel_rows = reversal_rows("heel")
    toe_rows = reversal_rows("toe")
    candidates: list[_Candidate] = []
    toe_offs: list[dict] = []
    state = np.full(n, "UNCERTAIN", dtype=object)

    # Every heel reversal opens grounded state.  The internal toe-off is the
    # first later toe elevation plus forward relative motion, bounded before
    # the next same-foot contact.
    for number, hrow in enumerate(heel_rows):
        hidx = int(hrow["frame"])
        nearby_toe = [r for r in toe_rows if abs(int(r["frame"]) - hidx) <= radius + 2]
        tidx = (
            int(max(nearby_toe, key=lambda r: r["strength"])["frame"])
            if nearby_toe else None
        )
        onset_frames = [hidx] + ([tidx] if tidx is not None else [])
        earliest, latest = min(onset_frames), max(onset_frames)
        next_contact = (
            int(heel_rows[number + 1]["frame"])
            if number + 1 < len(heel_rows) else seg_end + 1
        )
        toe_off = None
        search_start = min(next_contact, hidx + min_grounded)
        for j in range(search_start, next_contact):
            if not valid[j]:
                continue
            toe = point["toe"]
            elevated = np.isfinite(toe["height"][j]) and toe["height"][j] > 0.035
            forward = np.isfinite(toe["rel_vx"][j]) and toe["rel_vx"][j] > min_v
            if elevated and forward:
                toe_off = j
                break
        grounded_end = (toe_off if toe_off is not None else next_contact) - 1
        if grounded_end >= hidx:
            state[hidx:grounded_end + 1] = "GROUNDED"
        if toe_off is not None:
            state[toe_off:next_contact] = "SWING"
            toe_offs.append({
                "kind": "toe_off_internal",
                "side": side,
                "frame": int(toe_off),
                "t_s": round(float(times[toe_off]), 6),
                "grounded_start_frame": hidx,
            })

        heel = point["heel"]
        toe = point["toe"]
        support = max(
            float(heel["score"][hidx]) if np.isfinite(heel["score"][hidx]) else 0.0,
            float(toe["score"][hidx]) if np.isfinite(toe["score"][hidx]) else 0.0,
        )
        reversal_strength = min(
            1.0, max(0.0, (hrow["pre_v"] - hrow["post_v"]) / 2.5)
        )
        score = min(1.0, 0.55 * support + 0.45 * reversal_strength)
        candidates.append(_Candidate(
            side=side,
            idx=hidx,
            earliest_idx=earliest,
            latest_idx=latest,
            score=score,
            evidence={
                "state_change": "body_relative_forward_swing_to_sustained_grounded",
                "strike_region": "heel",
                "heel_onset_frame": hidx,
                "toe_onset_frame": tidx,
                "heel_relative_vx_pre_bs_s": _r(hrow["pre_v"]),
                "heel_relative_vx_post_bs_s": _r(hrow["post_v"]),
                "heel_reversal_prominence_bs": _r(hrow["prominence"]),
                "heel_speed_bs_s": _r(heel["speed"][hidx]),
                "toe_speed_bs_s": _r(toe["speed"][hidx]),
                "heel_speed_threshold_bs_s": _r(heel["speed_thr"][hidx]),
                "toe_speed_threshold_bs_s": _r(toe["speed_thr"][hidx]),
                "heel_support_height_bs": _r(heel["height"][hidx]),
                "toe_support_height_bs": _r(toe["height"][hidx]),
                "heel_grounded_score": _r(heel["score"][hidx]),
                "toe_grounded_score": _r(toe["score"][hidx]),
                "grounded_run_frames": max(0, grounded_end - hidx + 1),
                "grounded_run_s": round(
                    max(0, grounded_end - hidx + 1) * med_dt, 5
                ),
            },
        ))

    for i in seg:
        prev = "UNKNOWN" if i == seg_start else str(state[i - 1])
        heel, toe = point["heel"], point["toe"]
        records.append({
            "kind": "frame",
            "method": STATE_METHOD,
            "side": side,
            "frame": int(i),
            "t_s": round(float(times[i]), 6),
            "pts_us": int(round(float(times[i]) * 1_000_000)),
            "state_before": prev,
            "state_after": str(state[i]),
            "speed_bs_s": _r(max(heel["speed"][i], toe["speed"][i])),
            "heel_speed_bs_s": _r(heel["speed"][i]),
            "toe_speed_bs_s": _r(toe["speed"][i]),
            "heel_vx_bs_s": _r(heel["vx"][i]),
            "heel_vy_bs_s": _r(heel["vy"][i]),
            "toe_vx_bs_s": _r(toe["vx"][i]),
            "toe_vy_bs_s": _r(toe["vy"][i]),
            "heel_relative_vx_bs_s": _r(heel["rel_vx"][i]),
            "toe_relative_vx_bs_s": _r(toe["rel_vx"][i]),
            "heel_acc_bs_s2": _r(heel["acc"][i]),
            "toe_acc_bs_s2": _r(toe["acc"][i]),
            "heel_support_height_bs": _r(heel["height"][i]),
            "toe_support_height_bs": _r(toe["height"][i]),
            "heel_grounded_score": _r(heel["score"][i]),
            "toe_grounded_score": _r(toe["score"][i]),
            "heel_threshold_bs_s": _r(heel["speed_thr"][i]),
            "toe_threshold_bs_s": _r(toe["speed_thr"][i]),
        })
    records.extend(toe_offs)
    return candidates, toe_offs


def detect_state_contacts(trace, config: dict | None = None,
                          record_trace: bool = False):
    # Lazy import avoids a module cycle while retaining the established result
    # container and diagnostic helpers.
    from .contacts import (
        ContactMethodResult, _body_scale, _camera_motion_diagnostics,
        _extract, _in_too_small_span, _observability_diagnostics, _segments,
        _shot_boundary_diagnostics, _sort_diags,
    )

    cfg = validate_state_config(config)
    tracking = track_feet(
        trace,
        policy=cfg["track_policy"],
        min_visibility=min(0.35, cfg["min_landmark_visibility"]),
        jump_speed_bs_s=cfg["track_jump_speed_bs_s"],
        swap_margin_bs=cfg["track_swap_margin_bs"],
        max_interp_gap=cfg["track_max_interp_gap"],
        max_swap_run_frames=cfg["track_max_swap_run_frames"],
    )
    corrected = tracking.trace
    records: list[dict] = []
    header = {
        "kind": "header", "method": STATE_METHOD, "config": dict(cfg),
        "backend": trace.backend, "tracking": tracking.summary,
    }
    if not corrected.frames:
        return ContactMethodResult(
            [], [], [header] if record_trace else None,
            header if record_trace else None, corrected,
        )

    fd = _extract(corrected, cfg["min_landmark_visibility"])
    scale = _body_scale(fd, cfg["min_landmark_visibility"])
    header["scale_px"] = None if scale is None else round(float(scale), 6)
    diags = _shot_boundary_diagnostics(fd, cfg)
    diags.extend(_observability_diagnostics(fd, cfg, scale))
    diags.extend(_camera_motion_diagnostics(fd, cfg))
    for row in tracking.records:
        if row.get("decision") in {"quarantine", "quarantine_swap_run"}:
            dt = 1.0 / max(corrected.meta.fps_stream, 1.0)
            diags.append({
                "code": DiagReason.LEFT_RIGHT_DISCONTINUITY,
                "t0_s": row["t_s"], "t1_s": round(row["t_s"] + dt, 6),
                "detail": (
                    "contact-state-v1 abstained: implausible foot-track jump "
                    f"({row.get('max_jump_bs')} body-scale/frame)"
                ),
            })
    if scale is None or not np.isfinite(scale) or scale <= 0:
        diags.append({
            "code": DiagReason.FEET_OUT_OF_FRAME,
            "t0_s": corrected.frames[0].t_s,
            "t1_s": corrected.frames[-1].t_s,
            "detail": "no body-scale reference measurable; no contacts inferred",
        })
        trace_rows = [header, *tracking.records] if record_trace else None
        return ContactMethodResult(
            [], _sort_diags(diags), trace_rows,
            header if record_trace else None, corrected,
        )

    data = _extract_points(corrected, cfg["min_landmark_visibility"])
    filtered_impulses = _smooth(data, cfg["smoothing"], scale)
    body_x = np.asarray([
        (
            float(fr.hips.x)
            if fr.hips is not None and not fr.lost
            and np.isfinite(fr.hips.x)
            else np.nan
        )
        for fr in corrected.frames
    ], dtype=float)
    header["smoothing"] = {
        "mode": cfg["smoothing"], "filtered_impulses": filtered_impulses,
        "phase": "centred/offline; zero nominal group delay",
    }
    times = np.asarray([fr.t_s for fr in corrected.frames], dtype=float)
    invalid_contact = np.zeros(len(corrected.frames), dtype=bool)
    invalid_frames = {
        int(r["frame"]) for r in tracking.records
        if r.get("decision") in {
            "quarantine", "quarantine_swap_run", "interpolate"
        }
    }
    for i in invalid_frames:
        if 0 <= i < len(invalid_contact):
            invalid_contact[i] = True
    header["contact_abstention_frames"] = sorted(invalid_frames)
    candidates: list[_Candidate] = []
    for seg_index, seg in enumerate(_segments(fd, cfg)):
        segment_record = {
            "kind": "segment", "index": seg_index,
            "start_frame": int(seg[0]), "end_frame": int(seg[-1]),
        }
        records.append(segment_record)
        side_counts = {}
        for side in SIDES:
            got, _ = _relative_phase_candidates(
                data, body_x, times, seg, side, scale, cfg, records,
                invalid_contact,
            )
            side_counts[side] = len(got)
            candidates.extend(got)
        segment_record["phase_candidates"] = side_counts
        duration = float(times[int(seg[-1])] - times[int(seg[0])])
        if duration >= cfg["min_step_interval_s"] and not any(side_counts.values()):
            diags.append({
                "code": DiagReason.AMBIGUOUS_SUPPORT_PHASE,
                "t0_s": round(float(times[int(seg[0])]), 6),
                "t1_s": round(float(times[int(seg[-1])]), 6),
                "detail": (
                    "contact-state-v1 abstained: no defensible body-relative "
                    "swing-to-supported-stance reversal in this segment"
                ),
            })
        elif duration >= cfg["max_step_interval_s"]:
            for side, count in side_counts.items():
                if count == 0 and max(side_counts.values()) >= 2:
                    diags.append({
                        "code": DiagReason.AMBIGUOUS_SUPPORT_PHASE,
                        "t0_s": round(float(times[int(seg[0])]), 6),
                        "t1_s": round(float(times[int(seg[-1])]), 6),
                        "detail": (
                            f"contact-state-v1 abstained for {side} track: "
                            "phase evidence was not observable"
                        ),
                    })

    # Keep raw physical evidence.  We diagnose sequence contradictions here;
    # no contact is fabricated or silently removed by a cadence rule.
    candidates.sort(key=lambda c: (times[c.idx], c.side))
    for a, b in zip(candidates, candidates[1:]):
        dt = float(times[b.idx] - times[a.idx])
        if a.side != b.side and abs(dt) <= 1.0 / max(corrected.meta.fps_stream, 1.0):
            diags.append({
                "code": DiagReason.LEFT_RIGHT_DISCONTINUITY,
                "t0_s": round(float(times[a.idx]), 6),
                "t1_s": round(float(times[b.idx]), 6),
                "detail": "raw contact-state evidence implies simultaneous L/R contact",
            })
        if dt < cfg["min_step_interval_s"] or dt > cfg["max_step_interval_s"]:
            diags.append({
                "code": DiagReason.CADENCE_VIOLATION,
                "t0_s": round(float(times[a.idx]), 6),
                "t1_s": round(float(times[b.idx]), 6),
                "detail": f"raw {a.side}->{b.side} interval {dt:.3f}s outside bounds",
            })

    events: list[ContactEvent] = []
    for rank, cand in enumerate(candidates, start=1):
        if _in_too_small_span(fd, cand, cfg):
            continue
        idx = cand.idx
        evidence = dict(cand.evidence)
        evidence.update({
            "tracking_policy": cfg["track_policy"],
            "smoothing": cfg["smoothing"],
            "raw_layer": True,
            "anatomical_attribution": "continuity_track_seeded_from_backend_label",
            "anatomical_attribution_confidence": None,
            "anatomical_attribution_note": (
                "pose confidence is not calibrated anatomical-side confidence"
            ),
        })
        events.append(ContactEvent(
            id=f"evt_{rank:03d}",
            family=EVENT_FAMILY_FOOT_CONTACT,
            actor_id=DOMINANT_ACTOR_ID,
            foot=cand.side,
            source_pts_us=s_to_pts_us(times[idx]),
            time_s=round(float(times[idx]), 6),
            earliest_s=round(float(times[cand.earliest_idx]), 6),
            latest_s=round(float(times[cand.latest_idx]), 6),
            score=round(float(cand.score), 3),
            observable=True,
            detector=f"{trace.backend}-contact-state-v1",
            method=STATE_METHOD,
            evidence=evidence,
            kind="initial_contact",
        ))

    trace_rows = [header, *tracking.records, *records] if record_trace else None
    return ContactMethodResult(
        events=events,
        diagnostics=_sort_diags(diags),
        frame_trace=trace_rows,
        trace_header=header if record_trace else None,
        diagnostic_trace=corrected,
    )
