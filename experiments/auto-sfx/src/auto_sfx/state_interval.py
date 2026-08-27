"""Opt-in contact-interval-v1 detector (Phase 2C batch 3).

Wraps contact-state-v1 existence.  Preferred time moves to the first
shot-local (or rolling-quantile) heel-or-toe support crossing inside a
bounded forward window.  Events are never added or dropped.  Called by
the eval driver; not registered on detect_contacts.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .state_contacts import (
    STATE_CONFIG,
    _extract_points,
    _kinematics,
    _rolling_quantile,
    detect_state_contacts,
)
from .support_geometry import estimate_shot_support
from .types import SIDES, s_to_pts_us


INTERVAL_METHOD = "contact-interval-v1"

_FORBIDDEN_OFFSET_KEYS = ("global_offset", "timing_offset", "frame_offset")
_GEOMETRY_MODES = ("shot_local", "rolling_quantile_control")

INTERVAL_CONFIG = {
    **STATE_CONFIG,
    "geometry_mode": "shot_local",
    "max_forward_frames": 4,
    "min_support_points": 8,
    "max_support_residual_rms_bs": 0.08,
}


def default_interval_config() -> dict:
    return dict(INTERVAL_CONFIG)


def validate_interval_config(config: dict | None) -> dict:
    cfg = default_interval_config()
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
        "track_swap_margin_bs", "local_window_s", "grounded_score",
        "support_height_frac", "phase_min_velocity_bs_s",
        "min_same_foot_cycle_s", "max_support_residual_rms_bs",
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
    cfg["min_support_points"] = max(2, int(cfg["min_support_points"]))
    cfg["max_forward_frames"] = max(0, int(cfg["max_forward_frames"]))
    cfg["shot_boundaries"] = sorted(float(v) for v in cfg["shot_boundaries"])
    if cfg["smoothing"] not in {"raw", "light", "zero_phase"}:
        raise ValueError("smoothing must be raw, light, or zero_phase")
    if cfg["geometry_mode"] not in _GEOMETRY_MODES:
        raise ValueError(
            f"geometry_mode must be one of {_GEOMETRY_MODES}"
        )
    return cfg


def _state_layer_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in STATE_CONFIG}


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


def _point_heights(
    arr: dict,
    scale: float,
    geom,
    geometry_mode: str,
    half: int,
) -> np.ndarray:
    n = len(arr["x"])
    if geometry_mode == "rolling_quantile_control":
        if not np.isfinite(scale) or scale <= 0:
            return np.full(n, np.nan)
        floor_y = _rolling_quantile(arr["y"], half, 0.92, np.nan)
        return (floor_y - arr["y"]) / scale
    height = np.full(n, np.nan)
    if geom is None or geom.kind != "line":
        return height
    for i in range(n):
        height[i] = geom.height_bs(float(arr["x"][i]), float(arr["y"][i]), scale)
    return height


def _event_frame_index(event, frames, times: np.ndarray) -> int | None:
    pts = int(event.source_pts_us)
    for i, fr in enumerate(frames):
        if int(fr.source_pts_us) == pts:
            return i
    if times.size == 0:
        return None
    return int(np.argmin(np.abs(times - float(event.time_s))))


def _at_support(heel_h: np.ndarray, toe_h: np.ndarray, idx: int,
                thresh: float) -> bool:
    hh, th = heel_h[idx], toe_h[idx]
    return bool(
        (np.isfinite(hh) and hh <= thresh)
        or (np.isfinite(th) and th <= thresh)
    )


def _first_support_crossing(
    heel_h: np.ndarray,
    toe_h: np.ndarray,
    invalid: np.ndarray,
    preferred_idx: int,
    max_forward: int,
    thresh: float,
) -> int | None:
    n = len(heel_h)
    lo = preferred_idx + 1
    hi = min(n - 1, preferred_idx + max_forward)
    if max_forward <= 0 or lo > hi:
        return None
    # Already in-band at the apex: the crossing is not forward of preferred.
    if 0 <= preferred_idx < n and _at_support(heel_h, toe_h, preferred_idx, thresh):
        return None
    for i in range(lo, hi + 1):
        if invalid[i]:
            continue
        if _at_support(heel_h, toe_h, i, thresh):
            return i
    return None


def _window_missing_height(
    heel_h: np.ndarray,
    toe_h: np.ndarray,
    invalid: np.ndarray,
    preferred_idx: int,
    max_forward: int,
) -> bool:
    n = len(heel_h)
    lo = preferred_idx + 1
    hi = min(n - 1, preferred_idx + max_forward)
    if max_forward <= 0 or lo > hi:
        return False
    saw_valid = False
    for i in range(lo, hi + 1):
        if invalid[i]:
            continue
        saw_valid = True
        if np.isfinite(heel_h[i]) or np.isfinite(toe_h[i]):
            return False
    return saw_valid


def detect_interval_contacts(trace, config: dict | None = None,
                             record_trace: bool = False):
    """Return contacts.ContactMethodResult. Do not go through detect_contacts.

    Existence is contact-state-v1.  Timing is the first support crossing
    in (preferred_idx, preferred_idx + max_forward_frames], else the apex.
    Interpolated/quarantined frames cannot be the crossing.
    """

    from .contacts import (
        ContactMethodResult, _body_scale, _extract, _segments,
    )

    cfg = validate_interval_config(config)
    base = detect_state_contacts(
        trace, config=_state_layer_config(cfg), record_trace=True,
    )
    corrected = base.diagnostic_trace
    state_header = base.trace_header or {}
    invalid_frames = {
        int(v) for v in state_header.get("contact_abstention_frames", [])
    }
    geometry_rows: list[dict] = []
    interval_rows: list[dict] = []

    header = {
        "kind": "header",
        "method": INTERVAL_METHOD,
        "config": dict(cfg),
        "backend": trace.backend,
        "geometry_mode": cfg["geometry_mode"],
        "max_forward_frames": int(cfg["max_forward_frames"]),
        "contact_abstention_frames": sorted(invalid_frames),
        "tracking": state_header.get("tracking"),
        "scale_px": state_header.get("scale_px"),
    }

    if corrected is None or not corrected.frames or not base.events:
        header["support_geometry"] = geometry_rows
        events = [
            replace(
                ev,
                method=INTERVAL_METHOD,
                detector=f"{trace.backend}-{INTERVAL_METHOD}",
                evidence={
                    **dict(ev.evidence),
                    "apex_frame": None,
                    "support_frame": None,
                    "shift_frames": 0,
                    "geometry_kind": None,
                    "reason": "no_events" if not base.events else "empty_trace",
                },
            )
            for ev in base.events
        ]
        tracking_rows = [
            row for row in (base.frame_trace or [])
            if row.get("kind") == "tracking"
        ]
        trace_rows = (
            [header, *tracking_rows, *interval_rows] if record_trace else None
        )
        return ContactMethodResult(
            events=events,
            diagnostics=list(base.diagnostics),
            frame_trace=trace_rows,
            trace_header=header if record_trace else None,
            diagnostic_trace=corrected,
        )

    fd = _extract(corrected, cfg["min_landmark_visibility"])
    scale = _body_scale(fd, cfg["min_landmark_visibility"])
    header["scale_px"] = None if scale is None else round(float(scale), 6)
    times = np.asarray([fr.t_s for fr in corrected.frames], dtype=float)
    n = len(corrected.frames)
    invalid = np.zeros(n, dtype=bool)
    for i in invalid_frames:
        if 0 <= i < n:
            invalid[i] = True

    data = _extract_points(corrected, cfg["min_landmark_visibility"])
    segments = _segments(fd, cfg)
    shot_of = np.full(n, -1, dtype=int)
    shot_geom: list[dict] = []

    for seg_index, seg in enumerate(segments):
        shot_of[seg] = seg_index
        med_dt = (
            float(np.median(np.diff(times[seg]))) if len(seg) > 1 else 1 / 30.0
        )
        med_dt = max(med_dt, 1e-6)
        half = max(2, int(round(0.5 * cfg["local_window_s"] / med_dt)))
        geom = None
        reason = None
        if cfg["geometry_mode"] == "shot_local":
            if scale is None or not np.isfinite(scale) or scale <= 0:
                geom_kind = "abstain"
                reason = "invalid_scale"
            else:
                xs, ys, speeds, ts, idx = _collect_shot_observations(
                    data, times, seg, scale,
                )
                if idx.size:
                    geom = estimate_shot_support(
                        xs, ys, speeds, ts, idx, scale,
                        min_points=int(cfg["min_support_points"]),
                        max_residual_rms_bs=float(
                            cfg["max_support_residual_rms_bs"]
                        ),
                    )
                else:
                    geom = estimate_shot_support(
                        np.asarray([np.nan]), np.asarray([np.nan]),
                        np.asarray([np.nan]), np.asarray([np.nan]),
                        np.asarray([int(seg[0])], dtype=int),
                        scale if scale is not None else float("nan"),
                        min_points=int(cfg["min_support_points"]),
                        max_residual_rms_bs=float(
                            cfg["max_support_residual_rms_bs"]
                        ),
                    )
                geom_kind = geom.kind
                reason = geom.reason
            geometry_rows.append({
                "kind": "support_geometry",
                "index": seg_index,
                "geometry_kind": geom_kind,
                "reason": reason,
                "residual_rms_bs": (
                    None if geom is None or not np.isfinite(geom.residual_rms_bs)
                    else round(float(geom.residual_rms_bs), 6)
                ),
                "inlier_count": 0 if geom is None else geom.inlier_count,
                "shot_start_frame": int(seg[0]),
                "shot_end_frame": int(seg[-1]),
            })
        else:
            geom_kind = "rolling_quantile"
            geometry_rows.append({
                "kind": "support_geometry",
                "index": seg_index,
                "geometry_kind": geom_kind,
                "reason": None,
                "shot_start_frame": int(seg[0]),
                "shot_end_frame": int(seg[-1]),
            })
        heights = {}
        usable_scale = (
            float(scale)
            if scale is not None and np.isfinite(scale) and scale > 0
            else float("nan")
        )
        for side in SIDES:
            heights[side] = {
                name: _point_heights(
                    data[side][name], usable_scale, geom,
                    cfg["geometry_mode"], half,
                )
                for name in ("heel", "toe")
            }
        shot_geom.append({
            "geom_kind": geom_kind,
            "reason": reason,
            "heights": heights,
        })

    header["support_geometry"] = geometry_rows
    thresh = float(cfg["support_height_frac"])
    max_forward = int(cfg["max_forward_frames"])
    events = []
    for ev in base.events:
        apex_idx = _event_frame_index(ev, corrected.frames, times)
        evidence = dict(ev.evidence)
        if apex_idx is None:
            evidence.update({
                "apex_frame": None,
                "support_frame": None,
                "shift_frames": 0,
                "geometry_kind": None,
                "reason": "missing_apex_frame",
            })
            events.append(replace(
                ev,
                method=INTERVAL_METHOD,
                detector=f"{trace.backend}-{INTERVAL_METHOD}",
                evidence=evidence,
            ))
            continue

        shot_i = int(shot_of[apex_idx])
        if shot_i < 0:
            geom_kind, reason = None, "event_outside_shot"
            heights = None
        else:
            slot = shot_geom[shot_i]
            geom_kind, reason = slot["geom_kind"], slot["reason"]
            heights = slot["heights"].get(ev.foot)

        shifted = ev
        shift_frames = 0
        support_idx = None
        keep_reason = reason
        if heights is None:
            keep_reason = keep_reason or "missing_height"
        elif geom_kind == "abstain":
            keep_reason = reason or "geometry_abstain"
        else:
            heel_h = heights["heel"]
            toe_h = heights["toe"]
            if _window_missing_height(
                heel_h, toe_h, invalid, apex_idx, max_forward,
            ):
                keep_reason = "missing_height"
            else:
                support_idx = _first_support_crossing(
                    heel_h, toe_h, invalid, apex_idx, max_forward, thresh,
                )
                if support_idx is None:
                    keep_reason = keep_reason or "no_support_crossing"
                else:
                    new_t = round(float(times[support_idx]), 6)
                    shift_frames = int(support_idx - apex_idx)
                    keep_reason = None
                    shifted = replace(
                        ev,
                        source_pts_us=s_to_pts_us(times[support_idx]),
                        time_s=new_t,
                        earliest_s=ev.earliest_s,
                        latest_s=max(float(ev.latest_s), new_t),
                    )

        evidence.update({
            "apex_frame": int(apex_idx),
            "support_frame": None if support_idx is None else int(support_idx),
            "shift_frames": int(shift_frames),
            "geometry_kind": geom_kind,
            "reason": keep_reason,
        })
        events.append(replace(
            shifted,
            method=INTERVAL_METHOD,
            detector=f"{trace.backend}-{INTERVAL_METHOD}",
            evidence=evidence,
        ))
        interval_rows.append({
            "kind": "interval_shift",
            "method": INTERVAL_METHOD,
            "id": ev.id,
            "foot": ev.foot,
            "apex_frame": int(apex_idx),
            "support_frame": None if support_idx is None else int(support_idx),
            "shift_frames": int(shift_frames),
            "geometry_kind": geom_kind,
            "reason": keep_reason,
            "t_s": events[-1].time_s,
        })

    tracking_rows = [
        row for row in (base.frame_trace or [])
        if row.get("kind") == "tracking"
    ]
    trace_rows = (
        [header, *tracking_rows, *geometry_rows, *interval_rows]
        if record_trace else None
    )
    return ContactMethodResult(
        events=events,
        diagnostics=list(base.diagnostics),
        frame_trace=trace_rows,
        trace_header=header if record_trace else None,
        diagnostic_trace=corrected,
    )
