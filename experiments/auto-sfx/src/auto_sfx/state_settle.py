"""Opt-in contact-settle-v1 detector (Phase 2C batch 4).

Wraps contact-state-v1 existence.  Preferred time moves to the first
post-apex frame where downward heel/toe vy (or isotropic speed) settles
inside a bounded forward window.  Height does not participate.  Events
are never added or dropped.  Called by the eval driver; not registered
on detect_contacts.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .state_contacts import (
    STATE_CONFIG,
    _extract_points,
    _kinematics,
    detect_state_contacts,
)
from .types import SIDES, s_to_pts_us


SETTLE_METHOD = "contact-settle-v1"

_FORBIDDEN_OFFSET_KEYS = ("global_offset", "timing_offset", "frame_offset")
_SETTLE_SIGNALS = ("downward_vy", "speed")

SETTLE_CONFIG = {
    **STATE_CONFIG,
    "max_forward_frames": 4,
    "settle_vy_bs_s": 0.08,
    "settle_speed_bs_s": 0.16,
    "settle_signal": "downward_vy",
}


def default_settle_config() -> dict:
    return dict(SETTLE_CONFIG)


def validate_settle_config(config: dict | None) -> dict:
    cfg = default_settle_config()
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
        "min_same_foot_cycle_s", "settle_vy_bs_s", "settle_speed_bs_s",
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
    cfg["max_forward_frames"] = max(0, int(cfg["max_forward_frames"]))
    cfg["shot_boundaries"] = sorted(float(v) for v in cfg["shot_boundaries"])
    if cfg["smoothing"] not in {"raw", "light", "zero_phase"}:
        raise ValueError("smoothing must be raw, light, or zero_phase")
    if cfg["settle_signal"] not in _SETTLE_SIGNALS:
        raise ValueError(
            f"settle_signal must be one of {_SETTLE_SIGNALS}"
        )
    return cfg


def _state_layer_config(cfg: dict) -> dict:
    return {k: cfg[k] for k in STATE_CONFIG}


def _event_frame_index(event, frames, times: np.ndarray) -> int | None:
    pts = int(event.source_pts_us)
    for i, fr in enumerate(frames):
        if int(fr.source_pts_us) == pts:
            return i
    if times.size == 0:
        return None
    return int(np.argmin(np.abs(times - float(event.time_s))))


def _max_finite_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        vals = []
        if np.isfinite(a[i]):
            vals.append(float(a[i]))
        if np.isfinite(b[i]):
            vals.append(float(b[i]))
        if vals:
            out[i] = max(vals)
    return out


def _side_signal(data: dict, times: np.ndarray, scale: float,
                 settle_signal: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    key = "speed" if settle_signal == "speed" else "vy"
    for side in SIDES:
        heel = _kinematics(
            data[side]["heel"]["x"], data[side]["heel"]["y"], times, scale,
        )
        toe = _kinematics(
            data[side]["toe"]["x"], data[side]["toe"]["y"], times, scale,
        )
        out[side] = _max_finite_pair(heel[key], toe[key])
    return out


def _frame_settled(signal: np.ndarray, invalid: np.ndarray, idx: int,
                   bound: float) -> bool:
    if idx < 0 or idx >= len(signal) or invalid[idx]:
        return False
    val = signal[idx]
    return bool(np.isfinite(val) and val <= bound)


def _first_settle(
    signal: np.ndarray,
    invalid: np.ndarray,
    apex_idx: int,
    max_forward: int,
    bound: float,
) -> tuple[int | None, str | None]:
    n = len(signal)
    if _frame_settled(signal, invalid, apex_idx, bound):
        return None, "already_settled"
    lo = apex_idx + 1
    hi = min(n - 1, apex_idx + max_forward)
    if max_forward <= 0 or lo > hi:
        if (
            0 <= apex_idx < n
            and not invalid[apex_idx]
            and not np.isfinite(signal[apex_idx])
        ):
            return None, "missing_signal"
        return None, "no_settle"
    saw_usable = False
    saw_finite = False
    for i in range(lo, hi + 1):
        if invalid[i]:
            continue
        saw_usable = True
        if not np.isfinite(signal[i]):
            continue
        saw_finite = True
        if signal[i] <= bound:
            return i, None
    if saw_usable and not saw_finite:
        return None, "missing_signal"
    return None, "no_settle"


def _stamp(ev, evidence: dict, backend: str):
    return replace(
        ev,
        method=SETTLE_METHOD,
        detector=f"{backend}-{SETTLE_METHOD}",
        evidence=evidence,
    )


def detect_settle_contacts(trace, config: dict | None = None,
                           record_trace: bool = False):
    """Return contacts.ContactMethodResult. Do not go through detect_contacts.

    Existence is contact-state-v1.  Timing is the first settled sample in
    (preferred_idx, preferred_idx + max_forward_frames], else the apex.
    Interpolated/quarantined frames cannot be settle_frame.  Height is not
    an accept/reject gate.
    """

    from .contacts import (
        ContactMethodResult, _body_scale, _extract,
    )

    cfg = validate_settle_config(config)
    base = detect_state_contacts(
        trace, config=_state_layer_config(cfg), record_trace=True,
    )
    corrected = base.diagnostic_trace
    state_header = base.trace_header or {}
    invalid_frames = {
        int(v) for v in state_header.get("contact_abstention_frames", [])
    }
    settle_rows: list[dict] = []
    header = {
        "kind": "header",
        "method": SETTLE_METHOD,
        "config": dict(cfg),
        "backend": trace.backend,
        "settle_signal": cfg["settle_signal"],
        "max_forward_frames": int(cfg["max_forward_frames"]),
        "contact_abstention_frames": sorted(invalid_frames),
        "tracking": state_header.get("tracking"),
        "scale_px": state_header.get("scale_px"),
    }

    if corrected is None or not corrected.frames or not base.events:
        events = [
            _stamp(
                ev,
                {
                    **dict(ev.evidence),
                    "apex_frame": None,
                    "settle_frame": None,
                    "shift_frames": 0,
                    "settle_signal": cfg["settle_signal"],
                    "reason": "no_events" if not base.events else "empty_trace",
                },
                trace.backend,
            )
            for ev in base.events
        ]
        tracking_rows = [
            row for row in (base.frame_trace or [])
            if row.get("kind") == "tracking"
        ]
        trace_rows = (
            [header, *tracking_rows, *settle_rows] if record_trace else None
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

    bound = (
        float(cfg["settle_speed_bs_s"])
        if cfg["settle_signal"] == "speed"
        else float(cfg["settle_vy_bs_s"])
    )
    max_forward = int(cfg["max_forward_frames"])
    signals: dict[str, np.ndarray] | None = None
    if scale is not None and np.isfinite(scale) and scale > 0:
        data = _extract_points(corrected, cfg["min_landmark_visibility"])
        signals = _side_signal(data, times, float(scale), cfg["settle_signal"])

    events = []
    for ev in base.events:
        apex_idx = _event_frame_index(ev, corrected.frames, times)
        evidence = dict(ev.evidence)
        if apex_idx is None:
            evidence.update({
                "apex_frame": None,
                "settle_frame": None,
                "shift_frames": 0,
                "settle_signal": cfg["settle_signal"],
                "reason": "event_unindexed",
            })
            events.append(_stamp(ev, evidence, trace.backend))
            continue

        shifted = ev
        shift_frames = 0
        settle_idx = None
        if signals is None or ev.foot not in signals:
            keep_reason = "missing_signal"
        else:
            settle_idx, keep_reason = _first_settle(
                signals[ev.foot], invalid, apex_idx, max_forward, bound,
            )
            if settle_idx is not None:
                new_t = round(float(times[settle_idx]), 6)
                shift_frames = int(settle_idx - apex_idx)
                keep_reason = None
                shifted = replace(
                    ev,
                    source_pts_us=s_to_pts_us(times[settle_idx]),
                    time_s=new_t,
                    earliest_s=ev.earliest_s,
                    latest_s=max(float(ev.latest_s), new_t),
                )

        evidence.update({
            "apex_frame": int(apex_idx),
            "settle_frame": None if settle_idx is None else int(settle_idx),
            "shift_frames": int(shift_frames),
            "settle_signal": cfg["settle_signal"],
            "reason": keep_reason,
        })
        events.append(_stamp(shifted, evidence, trace.backend))
        settle_rows.append({
            "kind": "settle_shift",
            "method": SETTLE_METHOD,
            "id": ev.id,
            "foot": ev.foot,
            "apex_frame": int(apex_idx),
            "settle_frame": None if settle_idx is None else int(settle_idx),
            "shift_frames": int(shift_frames),
            "settle_signal": cfg["settle_signal"],
            "reason": keep_reason,
            "t_s": events[-1].time_s,
        })

    tracking_rows = [
        row for row in (base.frame_trace or [])
        if row.get("kind") == "tracking"
    ]
    trace_rows = (
        [header, *tracking_rows, *settle_rows] if record_trace else None
    )
    return ContactMethodResult(
        events=events,
        diagnostics=list(base.diagnostics),
        frame_trace=trace_rows,
        trace_header=header if record_trace else None,
        diagnostic_trace=corrected,
    )
