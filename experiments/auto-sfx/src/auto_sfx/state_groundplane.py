"""Opt-in contact-groundplane-v1 detector (Phase 2C batch 2).

Shot-local support geometry plus a bounded-duration semi-Markov state
path (SWING → COLLISION → SUPPORT) timestamps initial contact at the
collision, not at a body-relative forward apex.  Called by the eval
driver; not registered on detect_contacts.
"""

from __future__ import annotations

import numpy as np

from .state_contacts import (
    STATE_METHOD,
    _Candidate,
    _extract_points,
    _fill_short_gaps,
    _kinematics,
    _r,
    _relative_phase_candidates,
    _rolling_quantile,
    _smooth,
)
from .support_geometry import estimate_shot_support
from .tracking import track_feet
from .types import (
    DOMINANT_ACTOR_ID,
    EVENT_FAMILY_FOOT_CONTACT,
    ContactEvent,
    DiagReason,
    SIDES,
    s_to_pts_us,
)


GROUNDPLANE_METHOD = "contact-groundplane-v1"

SWING = "SWING"
COLLISION = "COLLISION"
SUPPORT = "SUPPORT"
UNCERTAIN = "UNCERTAIN"
_STATES = (SWING, COLLISION, SUPPORT, UNCERTAIN)
_S, _C, _G, _U = range(4)
_NEG = -1e12

_FORBIDDEN_OFFSET_KEYS = ("global_offset", "timing_offset", "frame_offset")
_GEOMETRY_MODES = ("shot_local", "rolling_quantile_control")
_STATE_MODES = ("semi_markov", "phase_apex_control")

GROUNDPLANE_CONFIG = {
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
    "geometry_mode": "shot_local",
    "support_height_frac": 0.09,
    "min_support_points": 8,
    "max_support_residual_rms_bs": 0.08,
    "state_mode": "semi_markov",
    "min_grounded_frames": 3,
    "min_collision_frames": 1,
    "max_collision_frames": 4,
    "min_same_foot_cycle_s": 0.45,
    "phase_min_velocity_bs_s": 0.08,
    "phase_peak_radius_frames": 3,
    "local_window_s": 1.2,
}


def default_groundplane_config() -> dict:
    return dict(GROUNDPLANE_CONFIG)


def validate_groundplane_config(config: dict | None) -> dict:
    cfg = default_groundplane_config()
    if config:
        banned = [k for k in _FORBIDDEN_OFFSET_KEYS if k in config]
        if banned:
            raise ValueError(f"forbidden timing-offset keys: {banned}")
        unknown = sorted(set(config) - set(cfg))
        if unknown:
            raise ValueError(f"unknown contact-config keys: {unknown}")
        cfg.update(config)
    for key in (
        "min_step_interval_s", "max_step_interval_s",
        "min_landmark_visibility", "actor_too_small_px",
        "camera_motion_std_px", "track_jump_speed_bs_s",
        "track_swap_margin_bs", "local_window_s",
        "support_height_frac", "phase_min_velocity_bs_s",
        "min_same_foot_cycle_s", "max_support_residual_rms_bs",
    ):
        cfg[key] = float(cfg[key])
    cfg["track_max_interp_gap"] = max(0, int(cfg["track_max_interp_gap"]))
    cfg["track_max_swap_run_frames"] = max(
        1, int(cfg["track_max_swap_run_frames"])
    )
    cfg["min_grounded_frames"] = max(2, int(cfg["min_grounded_frames"]))
    cfg["min_collision_frames"] = max(1, int(cfg["min_collision_frames"]))
    cfg["max_collision_frames"] = max(
        cfg["min_collision_frames"], int(cfg["max_collision_frames"])
    )
    cfg["min_support_points"] = max(2, int(cfg["min_support_points"]))
    cfg["phase_peak_radius_frames"] = max(
        2, int(cfg["phase_peak_radius_frames"])
    )
    cfg["shot_boundaries"] = sorted(float(v) for v in cfg["shot_boundaries"])
    if cfg["smoothing"] not in {"raw", "light", "zero_phase"}:
        raise ValueError("smoothing must be raw, light, or zero_phase")
    if cfg["geometry_mode"] not in _GEOMETRY_MODES:
        raise ValueError(
            f"geometry_mode must be one of {_GEOMETRY_MODES}"
        )
    if cfg["state_mode"] not in _STATE_MODES:
        raise ValueError(f"state_mode must be one of {_STATE_MODES}")
    return cfg


def _copy_point_data(data: dict) -> dict:
    return {
        side: {
            name: {k: np.copy(v) for k, v in data[side][name].items()}
            for name in data[side]
        }
        for side in data
    }


def _encode_shot_local_y(data: dict, geom, scale: float, floor_row: float = 1000.0) -> dict:
    """Map shot-local height onto image-y so phase-apex can reuse rolling-y."""

    out = _copy_point_data(data)
    for side in SIDES:
        for name in ("heel", "toe"):
            arr = out[side][name]
            for i in range(len(arr["y"])):
                if not (np.isfinite(arr["x"][i]) and np.isfinite(arr["y"][i])):
                    continue
                h = geom.height_bs(float(arr["x"][i]), float(arr["y"][i]), scale)
                if not np.isfinite(h):
                    arr["y"][i] = np.nan
                else:
                    arr["y"][i] = floor_row - h * scale
    return out


def _remap_phase_records(records: list[dict], start: int) -> None:
    for rec in records[start:]:
        if rec.get("method") == STATE_METHOD:
            rec["method"] = GROUNDPLANE_METHOD
        for key in ("state_before", "state_after"):
            if rec.get(key) == "GROUNDED":
                rec[key] = SUPPORT


def _heights_for_point(
    arr: dict,
    scale: float,
    geom,
    geometry_mode: str,
    half: int,
) -> np.ndarray:
    n = len(arr["x"])
    if geometry_mode == "rolling_quantile_control":
        floor_y = _rolling_quantile(arr["y"], half, 0.92, np.nan)
        return (floor_y - arr["y"]) / scale
    height = np.full(n, np.nan)
    if geom is None or geom.kind != "line":
        return height
    for i in range(n):
        height[i] = geom.height_bs(float(arr["x"][i]), float(arr["y"][i]), scale)
    return height


def _point_bundle(
    data: dict,
    body_x: np.ndarray,
    times: np.ndarray,
    side: str,
    scale: float,
    geom,
    geometry_mode: str,
    half: int,
) -> dict:
    n = len(times)
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
        height = _heights_for_point(arr, scale, geom, geometry_mode, half)
        local_swing = _rolling_quantile(kin["speed"], half, 0.75, 0.8)
        speed_thr = np.clip(0.30 * local_swing, 0.16, 0.85)
        point[name] = {
            **arr,
            **kin,
            "rel_x": rel_x,
            "rel_filled": rel_filled,
            "rel_vx": rel_vx,
            "height": height,
            "local_swing": local_swing,
            "speed_thr": speed_thr,
        }
    return point


def _emit_frame(i: int, point: dict, cfg: dict, usable: bool) -> np.ndarray:
    """Emission scores [SWING, COLLISION, SUPPORT, UNCERTAIN]; higher is better."""

    out = np.full(4, _NEG)
    if not usable:
        out[_U] = 0.0
        return out

    support_tol = float(cfg["support_height_frac"])
    min_v = float(cfg["phase_min_velocity_bs_s"])
    band = 1.15 * support_tol
    heel, toe = point["heel"], point["toe"]
    heights = [h for h in (heel["height"][i], toe["height"][i]) if np.isfinite(h)]
    if not heights:
        out[_U] = 0.0
        return out
    h_lo = float(min(heights))
    vy_vals = [v for v in (heel["vy"][i], toe["vy"][i]) if np.isfinite(v)]
    vy_down = float(max(vy_vals)) if vy_vals else 0.0
    rel_vals = [v for v in (heel["rel_vx"][i], toe["rel_vx"][i]) if np.isfinite(v)]
    rel_fwd = float(max(rel_vals)) if rel_vals else 0.0
    sp_vals = [v for v in (heel["speed"][i], toe["speed"][i]) if np.isfinite(v)]
    speed = float(max(sp_vals)) if sp_vals else 0.0

    clearly_air = h_lo > band
    near_plane = h_lo <= band
    planted = h_lo <= support_tol

    swing = np.clip(h_lo / max(support_tol, 1e-6), 0.0, 4.0)
    if clearly_air:
        swing += 2.0
    if rel_fwd > min_v:
        swing += 1.2
    if speed > 0.35:
        swing += 0.4
    if planted and not clearly_air:
        swing -= 1.5

    collision = 0.0
    if near_plane:
        collision += 1.2
    if planted:
        collision += 0.6
    collision += float(np.clip(vy_down / 0.4, 0.0, 2.0))
    if 0.0 <= h_lo <= band:
        collision += 1.0
    if clearly_air:
        collision -= 3.0
    settled = (
        planted and speed < 0.22 and vy_down < 0.08
        and h_lo <= 0.5 * support_tol
    )
    if settled:
        collision -= 1.8

    support = 0.0
    if planted:
        support += 2.2
    if h_lo <= 0.5 * support_tol:
        support += 1.0
    if speed < 0.28:
        support += 0.8
    if rel_fwd <= min_v:
        support += 0.6
    if clearly_air:
        support -= 4.0
    if vy_down > 0.5:
        support -= 0.8
    if rel_fwd > 3.0 * min_v:
        support -= 1.0

    out[_S] = float(swing)
    out[_C] = float(collision)
    out[_G] = float(support)
    out[_U] = 0.15
    return out


def _can_enter(ps: int, dur: int, ns: int, min_c: int, min_g: int) -> bool:
    if ns == _U:
        return True
    if ps == _S and ns == _C:
        return True
    if ps == _C and ns == _G and dur >= min_c:
        return True
    if ps == _G and ns == _S and dur >= min_g:
        return True
    if ps == _U and ns == _S:
        return True
    if ps == _U and dur == 1 and ns in (_C, _G):
        return True
    return False


def _viterbi(emit: np.ndarray, cfg: dict) -> np.ndarray:
    n = int(emit.shape[0])
    states = np.full(n, UNCERTAIN, dtype=object)
    if n == 0:
        return states
    min_c = int(cfg["min_collision_frames"])
    max_c = int(cfg["max_collision_frames"])
    min_g = int(cfg["min_grounded_frames"])
    max_d = (n, max_c, n, n)
    dp = np.full((n, 4, n), _NEG)
    bp_s = np.full((n, 4, n), -1, dtype=np.int16)
    bp_d = np.full((n, 4, n), -1, dtype=np.int16)
    dp[0, :, 0] = emit[0]

    # For each prev state, durations that may leave into each next state.
    leave = [[None] * 4 for _ in range(4)]
    for ps in range(4):
        durs = np.arange(1, max_d[ps] + 1)
        for ns in range(4):
            ok = np.array([_can_enter(ps, int(d), ns, min_c, min_g) for d in durs])
            leave[ps][ns] = np.flatnonzero(ok)

    for t in range(1, n):
        prev = dp[t - 1]
        for ns in range(4):
            e = emit[t, ns]
            best = _NEG
            best_ps = -1
            best_pd = -1
            for ps in range(4):
                idx = leave[ps][ns]
                if idx.size == 0:
                    continue
                vals = prev[ps, idx]
                k = int(np.argmax(vals))
                val = float(vals[k])
                if val > best:
                    best = val
                    best_ps = ps
                    best_pd = int(idx[k])
            if best > _NEG / 2:
                dp[t, ns, 0] = best + e
                bp_s[t, ns, 0] = best_ps
                bp_d[t, ns, 0] = best_pd
            stay_max = max_d[ns]
            if stay_max >= 2:
                src = prev[ns, :stay_max - 1]
                good = src > _NEG / 2
                if np.any(good):
                    dst = np.flatnonzero(good) + 1
                    src_i = dst - 1
                    dp[t, ns, dst] = src[src_i] + e
                    bp_s[t, ns, dst] = ns
                    bp_d[t, ns, dst] = src_i.astype(np.int16)

    last = dp[n - 1]
    end_s, end_d = 3, 0
    best = _NEG
    for s in range(4):
        sl = last[s, :max_d[s]]
        d = int(np.argmax(sl))
        val = float(sl[d])
        if val > best:
            best = val
            end_s, end_d = s, d
    path_s = np.empty(n, dtype=np.int8)
    s, d = end_s, end_d
    for t in range(n - 1, -1, -1):
        path_s[t] = s
        if t == 0:
            break
        ps, pd = int(bp_s[t, s, d]), int(bp_d[t, s, d])
        if ps < 0:
            s, d = _U, 0
        else:
            s, d = ps, pd
    for t in range(n):
        states[t] = _STATES[int(path_s[t])]
    return states


def _first_band(
    height: np.ndarray,
    valid: np.ndarray,
    lo: int,
    hi: int,
    band: float,
) -> int | None:
    lo = max(0, lo)
    hi = min(len(height) - 1, hi)
    for i in range(lo, hi + 1):
        if valid[i] and np.isfinite(height[i]) and height[i] <= band:
            return i
    return None


def _nearest_apex(
    rel_x: np.ndarray,
    collision_idx: int,
    radius: int,
) -> int | None:
    n = len(rel_x)
    lo = max(0, collision_idx - radius)
    hi = min(n - 1, collision_idx + radius)
    best = None
    best_val = -np.inf
    for i in range(lo, hi + 1):
        if not np.isfinite(rel_x[i]):
            continue
        wlo = max(0, i - radius)
        whi = min(n, i + radius + 1)
        window = rel_x[wlo:whi]
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            continue
        if rel_x[i] + 1e-12 >= float(np.max(finite)):
            if rel_x[i] > best_val:
                best_val = float(rel_x[i])
                best = i
    return best


def _semi_markov_candidates(
    point: dict,
    times: np.ndarray,
    seg: np.ndarray,
    side: str,
    cfg: dict,
    records: list[dict],
    invalid_contact: np.ndarray,
    geom_kind: str,
) -> list[_Candidate]:
    n = len(times)
    seg = np.asarray(seg, dtype=int)
    if seg.size == 0:
        return []
    valid = (
        np.isfinite(point["heel"]["x"])
        & np.isfinite(point["heel"]["y"])
        & np.isfinite(point["toe"]["x"])
        & np.isfinite(point["toe"]["y"])
        & np.isfinite(point["heel"]["height"])
        & np.isfinite(point["toe"]["height"])
        & ~invalid_contact
    )
    seg_mask = np.zeros(n, dtype=bool)
    seg_mask[seg] = True
    valid &= seg_mask

    emit = np.full((n, 4), _NEG)
    for i in range(n):
        if not seg_mask[i]:
            emit[i, _U] = 0.0
            continue
        emit[i] = _emit_frame(i, point, cfg, bool(valid[i]))

    state = np.full(n, UNCERTAIN, dtype=object)
    local = _viterbi(emit[int(seg[0]):int(seg[-1]) + 1], cfg)
    state[int(seg[0]):int(seg[-1]) + 1] = local

    # Invalid frames cannot be collision or support evidence.
    for i in np.flatnonzero(seg_mask & ~valid):
        if state[i] in (COLLISION, SUPPORT):
            state[i] = UNCERTAIN

    min_g = int(cfg["min_grounded_frames"])
    min_c = int(cfg["min_collision_frames"])
    band = 1.15 * float(cfg["support_height_frac"])
    radius = int(cfg["phase_peak_radius_frames"])
    candidates: list[_Candidate] = []
    i = int(seg[0])
    seg_end = int(seg[-1])
    while i <= seg_end:
        if state[i] != COLLISION or not valid[i]:
            i += 1
            continue
        c0 = i
        while i <= seg_end and state[i] == COLLISION:
            i += 1
        c1 = i - 1
        if (c1 - c0 + 1) < min_c or not valid[c0:c1 + 1].all():
            continue
        j = i
        if j <= seg_end and state[j] == UNCERTAIN:
            if j + 1 <= seg_end and state[j + 1] == UNCERTAIN:
                continue
            j += 1
        if j > seg_end or state[j] != SUPPORT:
            continue
        s0 = j
        while j <= seg_end and state[j] == SUPPORT:
            j += 1
        s1 = j - 1
        if (s1 - s0 + 1) < min_g or not valid[s0:s1 + 1].all():
            continue
        k = c0 - 1
        if k >= int(seg[0]) and state[k] == UNCERTAIN:
            k -= 1
        if k < int(seg[0]) or state[k] != SWING:
            continue

        search_lo = max(int(seg[0]), c0 - 2)
        search_hi = s0
        heel_c = _first_band(point["heel"]["height"], valid, search_lo, search_hi, band)
        toe_c = _first_band(point["toe"]["height"], valid, search_lo, search_hi, band)
        entries = [v for v in (heel_c, toe_c) if v is not None]
        earliest_idx = min(entries) if entries else c0
        latest_idx = s0
        if heel_c is not None and toe_c is not None:
            pick = heel_c if heel_c <= toe_c else toe_c
        elif heel_c is not None:
            pick = heel_c
        elif toe_c is not None:
            pick = toe_c
        else:
            pick = c0
        pick = min(max(int(pick), int(earliest_idx)), int(latest_idx))

        apex = _nearest_apex(point["heel"]["rel_filled"], c0, radius)
        strike = "heel" if pick == heel_c or toe_c is None else "toe"
        heel, toe = point["heel"], point["toe"]
        support_conf = 0.0
        for name in ("heel", "toe"):
            h = point[name]["height"][s0]
            if np.isfinite(h):
                support_conf = max(
                    support_conf,
                    float(np.clip(1.0 - h / max(cfg["support_height_frac"], 1e-6), 0.0, 1.0)),
                )
        score = min(1.0, max(0.15, 0.55 * support_conf + 0.45))
        candidates.append(_Candidate(
            side=side,
            idx=pick,
            earliest_idx=int(earliest_idx),
            latest_idx=int(latest_idx),
            score=score,
            evidence={
                "state_change": "swing_to_collision_to_support",
                "strike_region": strike,
                "apex_frame": None if apex is None else int(apex),
                "collision_frame": int(c0),
                "support_frame": int(s0),
                "geometry_kind": geom_kind,
                "heel_onset_frame": heel_c,
                "toe_onset_frame": toe_c,
                "heel_speed_bs_s": _r(heel["speed"][pick]),
                "toe_speed_bs_s": _r(toe["speed"][pick]),
                "heel_support_height_bs": _r(heel["height"][pick]),
                "toe_support_height_bs": _r(toe["height"][pick]),
                "heel_relative_vx_bs_s": _r(heel["rel_vx"][pick]),
                "toe_relative_vx_bs_s": _r(toe["rel_vx"][pick]),
                "grounded_run_frames": int(s1 - s0 + 1),
            },
        ))
        i = j

    for i in seg:
        prev = "UNKNOWN" if i == int(seg[0]) else str(state[i - 1])
        heel, toe = point["heel"], point["toe"]
        records.append({
            "kind": "frame",
            "method": GROUNDPLANE_METHOD,
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
            "heel_support_height_bs": _r(heel["height"][i]),
            "toe_support_height_bs": _r(toe["height"][i]),
            "geometry_kind": geom_kind,
        })
    return candidates


def _collect_shot_observations(
    data: dict,
    times: np.ndarray,
    seg: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    speeds: list[float] = []
    ts: list[float] = []
    idx: list[int] = []
    for side in SIDES:
        for name in ("heel", "toe"):
            arr = data[side][name]
            kin = _kinematics(arr["x"], arr["y"], times, scale)
            for i in seg:
                i = int(i)
                if not (np.isfinite(arr["x"][i]) and np.isfinite(arr["y"][i])
                        and np.isfinite(kin["speed"][i])):
                    continue
                xs.append(float(arr["x"][i]))
                ys.append(float(arr["y"][i]))
                speeds.append(float(kin["speed"][i]))
                ts.append(float(times[i]))
                idx.append(i)
    if not xs:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty, np.asarray([], dtype=int)
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(speeds, dtype=float),
        np.asarray(ts, dtype=float),
        np.asarray(idx, dtype=int),
    )


def _phase_candidates(
    data: dict,
    body_x: np.ndarray,
    times: np.ndarray,
    seg: np.ndarray,
    side: str,
    scale: float,
    cfg: dict,
    records: list[dict],
    invalid_contact: np.ndarray,
    geom,
    geom_kind: str,
) -> list[_Candidate]:
    start = len(records)
    phase_data = data
    if cfg["geometry_mode"] == "shot_local":
        if geom is None or geom.kind != "line":
            return []
        phase_data = _encode_shot_local_y(data, geom, scale)
    got, _ = _relative_phase_candidates(
        phase_data, body_x, times, seg, side, scale, cfg, records,
        invalid_contact,
    )
    _remap_phase_records(records, start)
    for cand in got:
        ev = dict(cand.evidence)
        ev.setdefault("apex_frame", int(cand.idx))
        ev.setdefault("collision_frame", int(cand.idx))
        ev.setdefault("support_frame", int(cand.idx))
        ev["geometry_kind"] = geom_kind
        cand.evidence = ev
    return got


def detect_groundplane_contacts(trace, config: dict | None = None,
                                record_trace: bool = False):
    """Return contacts.ContactMethodResult. Do not go through detect_contacts.

    method on events: GROUNDPLANE_METHOD
    detector: f"{trace.backend}-contact-groundplane-v1"
    frame states: SWING | COLLISION | SUPPORT | UNCERTAIN only
    interpolated/quarantined frames are invalid as COLLISION or SUPPORT
    """

    from .contacts import (
        ContactMethodResult, _body_scale, _camera_motion_diagnostics,
        _extract, _in_too_small_span, _observability_diagnostics, _segments,
        _shot_boundary_diagnostics, _sort_diags,
    )

    cfg = validate_groundplane_config(config)
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
        "kind": "header",
        "method": GROUNDPLANE_METHOD,
        "config": dict(cfg),
        "backend": trace.backend,
        "tracking": tracking.summary,
        "geometry_mode": cfg["geometry_mode"],
        "state_mode": cfg["state_mode"],
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
                    "contact-groundplane-v1 abstained: implausible foot-track jump "
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
            "quarantine", "quarantine_swap_run", "interpolate",
        }
    }
    for i in invalid_frames:
        if 0 <= i < len(invalid_contact):
            invalid_contact[i] = True
    header["contact_abstention_frames"] = sorted(invalid_frames)
    candidates: list[_Candidate] = []
    geometry_rows: list[dict] = []

    for seg_index, seg in enumerate(_segments(fd, cfg)):
        med_dt = (
            float(np.median(np.diff(times[seg]))) if len(seg) > 1 else 1 / 30.0
        )
        med_dt = max(med_dt, 1e-6)
        half = max(2, int(round(0.5 * cfg["local_window_s"] / med_dt)))
        geom = None
        if cfg["geometry_mode"] == "shot_local":
            xs, ys, speeds, ts, idx = _collect_shot_observations(
                data, times, seg, scale,
            )
            if idx.size:
                geom = estimate_shot_support(
                    xs, ys, speeds, ts, idx, scale,
                    min_points=int(cfg["min_support_points"]),
                    max_residual_rms_bs=float(cfg["max_support_residual_rms_bs"]),
                )
            else:
                geom = estimate_shot_support(
                    np.asarray([np.nan]), np.asarray([np.nan]),
                    np.asarray([np.nan]), np.asarray([np.nan]),
                    np.asarray([int(seg[0])], dtype=int),
                    scale,
                    min_points=int(cfg["min_support_points"]),
                    max_residual_rms_bs=float(cfg["max_support_residual_rms_bs"]),
                )
            geom_kind = geom.kind
            geometry_rows.append({
                "kind": "support_geometry",
                "index": seg_index,
                "geometry_kind": geom.kind,
                "reason": geom.reason,
                "residual_rms_bs": (
                    None if not np.isfinite(geom.residual_rms_bs)
                    else round(float(geom.residual_rms_bs), 6)
                ),
                "inlier_count": geom.inlier_count,
                "shot_start_frame": geom.shot_start_frame,
                "shot_end_frame": geom.shot_end_frame,
            })
            if geom.kind == "abstain":
                diags.append({
                    "code": DiagReason.AMBIGUOUS_SUPPORT_PHASE,
                    "t0_s": round(float(times[int(seg[0])]), 6),
                    "t1_s": round(float(times[int(seg[-1])]), 6),
                    "detail": (
                        "contact-groundplane-v1 abstained: support geometry "
                        f"kind=abstain ({geom.reason})"
                    ),
                })
        else:
            geom_kind = "rolling_quantile"

        segment_record = {
            "kind": "segment", "index": seg_index,
            "start_frame": int(seg[0]), "end_frame": int(seg[-1]),
            "geometry_kind": geom_kind,
        }
        records.append(segment_record)
        side_counts = {}
        for side in SIDES:
            if cfg["state_mode"] == "phase_apex_control":
                got = _phase_candidates(
                    data, body_x, times, seg, side, scale, cfg, records,
                    invalid_contact, geom, geom_kind,
                )
            else:
                point = _point_bundle(
                    data, body_x, times, side, scale, geom,
                    cfg["geometry_mode"], half,
                )
                got = _semi_markov_candidates(
                    point, times, seg, side, cfg, records,
                    invalid_contact, geom_kind,
                )
            side_counts[side] = len(got)
            candidates.extend(got)
        segment_record["phase_candidates"] = side_counts
        duration = float(times[int(seg[-1])] - times[int(seg[0])])
        if duration >= cfg["min_step_interval_s"] and not any(side_counts.values()):
            if not (
                cfg["geometry_mode"] == "shot_local"
                and geom is not None
                and geom.kind == "abstain"
            ):
                diags.append({
                    "code": DiagReason.AMBIGUOUS_SUPPORT_PHASE,
                    "t0_s": round(float(times[int(seg[0])]), 6),
                    "t1_s": round(float(times[int(seg[-1])]), 6),
                    "detail": (
                        "contact-groundplane-v1 abstained: no defensible "
                        "swing-to-collision-to-support path in this segment"
                    ),
                })

    header["support_geometry"] = geometry_rows
    candidates.sort(key=lambda c: (times[c.idx], c.side))
    for a, b in zip(candidates, candidates[1:]):
        dt = float(times[b.idx] - times[a.idx])
        if a.side != b.side and abs(dt) <= 1.0 / max(corrected.meta.fps_stream, 1.0):
            diags.append({
                "code": DiagReason.LEFT_RIGHT_DISCONTINUITY,
                "t0_s": round(float(times[a.idx]), 6),
                "t1_s": round(float(times[b.idx]), 6),
                "detail": (
                    "raw contact-groundplane evidence implies simultaneous L/R contact"
                ),
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
            detector=f"{trace.backend}-contact-groundplane-v1",
            method=GROUNDPLANE_METHOD,
            evidence=evidence,
            kind="initial_contact",
        ))

    trace_rows = (
        [header, *tracking.records, *records] if record_trace else None
    )
    return ContactMethodResult(
        events=events,
        diagnostics=_sort_diags(diags),
        frame_trace=trace_rows,
        trace_header=header if record_trace else None,
        diagnostic_trace=corrected,
    )
