"""Foot-contact / gait detection engine (POC steps 4-6).

Three inspectable detectors over a :class:`auto_sfx.types.PoseTrace`:

``velocity``
    O'Connor-style adaptation.  Per-side foot speed is the mean
    displacement of the usable heel/ankle/toe landmarks between consecutive
    frames, divided by the frame dt and by a body scale (median hip->ankle
    distance), giving units of body-scale/s -- no magic pixel constants.
    Speed drives an offline per-foot state machine::

        UNKNOWN -> SWING -> CONTACT_CANDIDATE -> STANCE -> EXIT -> SWING

    A swing-peak speed falling below ``speed_drop_frac * recent_peak``
    opens a candidate at the first low-speed frame; ``confirm_frames``
    persistent low-speed frames confirm stance.  Two guards reject the
    mid-swing apex dip (where a sinusoidal swing foot is momentarily slow):
    the low run must last at least ``_MIN_STANCE_FRAC * min_step_interval``
    seconds, and the foot must not have been ascending into the run.
    Runs opened straight out of EXIT (the foot was observed airborne a
    moment before) are confirmed on persistence alone -- but only when the
    preceding stance was machine-confirmed, never merely seeded.
    Unconfirmed candidates are dropped silently -- missing a questionable
    step is preferred over inventing one.

``pelvis``
    Zeni-style adaptation.  The anterior/posterior ankle-relative-to-hips
    offset is projected on the dominant walk direction (principal axis of
    the hip trajectory, signed so the MEAN forward step is positive -- the
    median is biased by discretely sampled sway whose per-step velocity
    dwarfs drift).  Candidates are local maxima of the offset during
    deceleration.  The raw offset of a small-step walker is dominated by
    step-frequency pelvis sway, so the offset is smoothed with a centred,
    zero-phase moving average spanning exactly one sway period (the FIRST
    autocorrelation local maximum, i.e. the fundamental -- never a
    subharmonic); that nulls the sway band without shifting peak timing,
    while the per-foot strike content survives at ~0.64 gain.  Frontal-view
    degenerate geometry (hip-trajectory variance ratio near 1, or hips that
    barely translate) yields NO candidates plus one
    ``AMBIGUOUS_SUPPORT_PHASE`` interval spanning the segment.

``hybrid``
    The velocity state machine as primary.  A vertical envelope -- foot
    height above the running per-foot minima band, deliberately NOT called
    a ground plane -- and the heel/toe height relationship refine the score
    and record ``evidence["strike"]`` (``"heel"``, ``"forefoot"`` or
    ``"ambiguous"``).  Both the velocity drop AND envelope support are
    required before a candidate confirms.

Policies (docs/auto-sft-poc.md, diagnostics-first-class):

- Offline analysis MAY use future context, but every event reports the
  earliest frame whose evidence supports it (``earliest_s``); ``latest_s``
  adds one local frame duration, bounding uncertainty by one source frame.
- Observability gates: spans with feet missing / sub-threshold visibility
  (``FEET_OUT_OF_FRAME`` / ``HEEL_LOW_CONFIDENCE`` / ``TOE_LOW_CONFIDENCE``
  picked by which landmarks fail), lost pose (``POSE_LOST``), hip-ankle
  pixel scale below ``actor_too_small_px`` (``ACTOR_TOO_SMALL``) and
  degenerate pelvis geometry produce diagnostics and NEVER events.
  ``CAMERA_MOTION_HIGH`` (high-pass-filtered hip stddev) is measured but
  non-fatal.
- ``observable=False`` marks events kept despite sub-threshold heel/toe
  visibility at the contact frame (the ankle and remaining landmarks
  carried the measurement).  This is the only way an event may be emitted
  from partially confident landmarks; everything weaker abstains.
- Boundary contacts are not invented: a strike whose transition lies
  before the first frame, or whose stance cannot be observed for
  ``confirm_frames`` (and the minimum stance duration) inside the trace,
  is abstained from by design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # package-relative when imported as auto_sfx.contacts
    from .types import (
        CONTACT_METHODS,
        DOMINANT_ACTOR_ID,
        EVENT_FAMILY_FOOT_CONTACT,
        SIDES,
        ContactEvent,
        DiagReason,
        PoseTrace,
        s_to_pts_us,
    )
except ImportError:  # pragma: no cover - direct module import
    from types import (  # type: ignore[no-redef]
        CONTACT_METHODS,
        DOMINANT_ACTOR_ID,
        EVENT_FAMILY_FOOT_CONTACT,
        SIDES,
        ContactEvent,
        DiagReason,
        PoseTrace,
        s_to_pts_us,
    )

__all__ = ["ContactMethodResult", "default_config", "detect_contacts"]

# Report floats are rounded to this many decimal places.
_ROUND = 6

# Floor for the tracked swing peak (body-scale/s) so thresholds stay sane
# when a foot barely moves.
_MIN_PEAK_BS_S = 0.4

# A confirmed low-speed run must last at least this fraction of
# min_step_interval_s.  The sinusoidal apex dip of a swing foot lasts
# ~23% of the swing (about 11% of a step interval) -- comfortably below.
_MIN_STANCE_FRAC = 0.45

# If the foot rose more than this fraction of body scale within the frame
# preceding a low-speed run, the run is a mid-swing apex dip, not a plant.
_ASCENT_FRAC = 0.05

# Hybrid envelope: foot height above the running minima band must not
# exceed this fraction of body scale for a contact to be supported.
_ENV_TOL_FRAC = 0.15

# Hybrid strike labelling: heel/toe height differences below this fraction
# of body scale are recorded as "ambiguous".
_STRIKE_EPS_FRAC = 0.015

# Pelvis degeneracy thresholds: hip trajectory eigenvalue ratio above the
# first, or axis span below the second fraction of body scale, refuse to
# invent events.
_PELVIS_RATIO_MAX = 0.5
_PELVIS_SPAN_FRAC = 0.3

# Pelvis peak prominence floor, as a fraction of body scale.
_PELVIS_PROM_FRAC = 0.008

# High-pass window (seconds) for the camera-motion proxy.
_CAMERA_HP_WINDOW_S = 0.5

# Pelvis peak prominence floor, as a fraction of body scale.  Deliberately
# small: after sway nulling the residual ripple is far below the gait
# triangle's ~1.3% peak prominence, while genuine noise spikes fail the
# decline confirmation regardless.
_PELVIS_PROM_FRAC = 0.005
_PELVIS_ACF_MIN = 0.25

# State-machine states.
_UNKNOWN, _SWING, _CAND, _STANCE, _EXIT = (
    "UNKNOWN", "SWING", "CANDIDATE", "STANCE", "EXIT",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "min_step_interval_s": 0.25,
    "max_step_interval_s": 2.0,
    "confirm_frames": 2,
    "speed_drop_frac": 0.35,
    "min_landmark_visibility": 0.5,
    "shot_boundaries": [],
    "actor_too_small_px": 24.0,
    "camera_motion_std_px": 12.0,
}


def default_config(method: str) -> dict:
    """Return the default detector configuration for ``method``."""

    if method not in CONTACT_METHODS:
        raise ValueError(
            f"unknown contact method {method!r}; expected one of {CONTACT_METHODS}"
        )
    return dict(_BASE_CONFIG)


def _validated_config(method: str, config: dict | None) -> dict:
    """Merge user config over defaults; unknown keys are a programming error."""

    cfg = default_config(method)
    if config:
        unknown = sorted(set(config) - set(cfg))
        if unknown:
            raise ValueError(f"unknown contact-config keys: {unknown}")
        cfg.update(config)
    cfg["min_step_interval_s"] = float(cfg["min_step_interval_s"])
    cfg["max_step_interval_s"] = float(cfg["max_step_interval_s"])
    cfg["confirm_frames"] = max(1, int(cfg["confirm_frames"]))
    cfg["speed_drop_frac"] = float(cfg["speed_drop_frac"])
    cfg["min_landmark_visibility"] = float(cfg["min_landmark_visibility"])
    cfg["actor_too_small_px"] = float(cfg["actor_too_small_px"])
    cfg["camera_motion_std_px"] = float(cfg["camera_motion_std_px"])
    cfg["shot_boundaries"] = sorted(float(b) for b in cfg["shot_boundaries"])
    return cfg


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ContactMethodResult:
    """Outcome of one detector pass.

    ``events`` is sorted by ``time_s`` ascending.  Each diagnostics entry is
    ``{"code": DiagReason.*, "t0_s": float|None, "t1_s": float|None,
    "detail": str}`` sorted by ``(t0_s, t1_s, code)``.
    """

    events: list[ContactEvent]
    diagnostics: list[dict]


@dataclass
class _Cand:
    """Internal pre-event contact candidate."""

    side: str
    idx: int            # global frame index of the earliest supporting frame
    t: float
    score: float
    observable: bool
    evidence: dict


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _extract(trace: PoseTrace, min_vis: float) -> dict:
    """Flatten a trace into per-frame numpy arrays (NaN where unavailable).

    The foot reference position is the mean of the landmarks that are both
    present and at/above ``min_vis``; at least two are required so the
    reference is stable.
    """

    n = len(trace.frames)
    times = np.full(n, np.nan)
    dt = np.full(n, np.nan)
    lost = np.zeros(n, dtype=bool)
    hips_x = np.full(n, np.nan)
    hips_y = np.full(n, np.nan)
    hips_ok = np.zeros(n, dtype=bool)
    feet = {side: {
        "rx": np.full(n, np.nan),
        "ry": np.full(n, np.nan),
        "usable": np.zeros(n, dtype=bool),
        "heel_vis": np.full(n, np.nan),
        "ankle_vis": np.full(n, np.nan),
        "toe_vis": np.full(n, np.nan),
        "heel_y": np.full(n, np.nan),
        "toe_y": np.full(n, np.nan),
    } for side in SIDES}

    prev_t = None
    for i, fr in enumerate(trace.frames):
        times[i] = fr.t_s
        if prev_t is not None and fr.t_s > prev_t:
            dt[i] = fr.t_s - prev_t
        prev_t = fr.t_s
        lost[i] = bool(fr.lost)
        if fr.hips is not None and not fr.lost:
            hips_x[i] = fr.hips.x
            hips_y[i] = fr.hips.y
            hips_ok[i] = True
        for side in SIDES:
            obs = (fr.feet or {}).get(side)
            if obs is None:
                continue
            arr = feet[side]
            marks = (("heel_vis", "heel_y", obs.heel),
                     ("ankle_vis", None, obs.ankle),
                     ("toe_vis", "toe_y", obs.toe))
            n_usable = 0
            sx = sy = 0.0
            for vis_key, y_key, pt in marks:
                if pt is None:
                    continue
                arr[vis_key][i] = pt.visibility
                if y_key is not None:
                    arr[y_key][i] = pt.y
                if pt.visibility >= min_vis:
                    sx += pt.x
                    sy += pt.y
                    n_usable += 1
            if n_usable >= 2:
                arr["rx"][i] = sx / n_usable
                arr["ry"][i] = sy / n_usable
                arr["usable"][i] = not fr.lost

    return {
        "n": n,
        "times": times,
        "dt": dt,
        "lost": lost,
        "hips_x": hips_x,
        "hips_y": hips_y,
        "hips_ok": hips_ok,
        "feet": feet,
    }


def _finite(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


def _median_dt(fd: dict) -> float:
    vals = fd["dt"][_finite(fd["dt"])]
    return float(np.median(vals)) if vals.size else 0.0


def _local_dt(fd: dict, idx: int) -> float:
    """Frame duration at ``idx``: median of its neighbouring dt samples."""

    vals = [v for v in (fd["dt"][idx],
                        fd["dt"][idx + 1] if idx + 1 < fd["n"] else np.nan)
            if np.isfinite(v)]
    if vals:
        return float(np.median(vals))
    return _median_dt(fd)


# ---------------------------------------------------------------------------
# Body scale and observability diagnostics
# ---------------------------------------------------------------------------


def _body_scale(fd: dict, min_vis: float) -> float | None:
    """Median hip->ankle distance over frames where heel, ankle and toe all
    have visibility >= min_vis (the contract's body-scale definition)."""

    dists = []
    keys = ("heel_vis", "ankle_vis", "toe_vis")
    for side in SIDES:
        arr = fd["feet"][side]
        for i in range(fd["n"]):
            if not fd["hips_ok"][i]:
                continue
            if not all(np.isfinite(arr[k][i]) and arr[k][i] >= min_vis for k in keys):
                continue
            d = math.hypot(fd["hips_x"][i] - arr["rx"][i],
                           fd["hips_y"][i] - arr["ry"][i])
            if d > 0:
                dists.append(d)
    if not dists:
        return None
    return float(np.median(dists))


def _frame_scale(fd: dict, side: str) -> np.ndarray:
    """Hip->foot-reference pixel distance per frame (NaN where unknown)."""

    arr = fd["feet"][side]
    out = np.hypot(fd["hips_x"] - arr["rx"], fd["hips_y"] - arr["ry"])
    out[~(fd["hips_ok"] & arr["usable"])] = np.nan
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal True runs of ``mask`` as inclusive (first, last) indices."""

    out = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _diag(code: str, t0: float | None, t1: float | None, detail: str) -> dict:
    return {
        "code": code,
        "t0_s": None if t0 is None else round(float(t0), _ROUND),
        "t1_s": None if t1 is None else round(float(t1), _ROUND),
        "detail": detail,
    }


def _sort_diags(diags: list[dict]) -> list[dict]:
    diags.sort(key=lambda d: (d["t0_s"] if d["t0_s"] is not None else float("-inf"),
                              d["t1_s"] if d["t1_s"] is not None else float("-inf"),
                              d["code"]))
    return diags


def _observability_diagnostics(fd: dict, cfg: dict, scale: float | None) -> list[dict]:
    """Emit DiagReason intervals for every observability problem found."""

    diags: list[dict] = []
    times = fd["times"]
    n = fd["n"]
    min_vis = cfg["min_landmark_visibility"]

    for (a, b) in _runs(fd["lost"]):
        diags.append(
            _diag(DiagReason.POSE_LOST, times[a], times[b] + _local_dt(fd, b),
                  f"pose lost, frames {a}-{b}")
        )

    for side in SIDES:
        arr = fd["feet"][side]
        # Feet effectively out of frame: observation absent, or fewer than
        # two landmarks at/above the visibility floor (the reference point
        # needs a stable multi-landmark base).
        n_marks = np.zeros(n, dtype=int)
        for key in ("heel_vis", "ankle_vis", "toe_vis"):
            n_marks += (_finite(arr[key]) & (arr[key] >= min_vis)).astype(int)
        bad = (~fd["lost"]) & (n_marks < 2)
        for (a, b) in _runs(bad):
            diags.append(
                _diag(DiagReason.FEET_OUT_OF_FRAME, times[a], times[b] + _local_dt(fd, b),
                      f"{side} foot unobservable (absent or sub-threshold), frames {a}-{b}")
            )
        for key, code in (("heel", DiagReason.HEEL_LOW_CONFIDENCE),
                          ("toe", DiagReason.TOE_LOW_CONFIDENCE)):
            fail = ((~fd["lost"]) & _finite(arr[f"{key}_vis"])
                    & (arr[f"{key}_vis"] < min_vis) & ~bad)
            for (a, b) in _runs(fail):
                diags.append(
                    _diag(code, times[a], times[b] + _local_dt(fd, b),
                          f"{key} visibility < {min_vis:.2f} on {side} foot, frames {a}-{b}")
                )
    if scale is not None:
        for side in SIDES:
            fscale = _frame_scale(fd, side)
            for (a, b) in _runs(_finite(fscale) & (fscale < cfg["actor_too_small_px"])):
                diags.append(
                    _diag(DiagReason.ACTOR_TOO_SMALL, times[a], times[b] + _local_dt(fd, b),
                          f"{side} hip-ankle scale {fscale[a]:.1f}px < "
                          f"{cfg['actor_too_small_px']:.0f}px, frames {a}-{b}")
                )

    diags.extend(_camera_motion_diagnostics(fd, cfg))
    return diags


def _centered_ma(values: np.ndarray, half: int) -> np.ndarray:
    """Centred, zero-phase moving average with edge replication."""

    if half <= 0:
        return values.copy()
    pad = np.pad(values, half, mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(pad, kernel, mode="valid")


def _camera_motion_diagnostics(fd: dict, cfg: dict) -> list[dict]:
    """Non-fatal CAMERA_MOTION_HIGH when filtered hip jitter exceeds budget."""

    ok = fd["hips_ok"]
    if ok.sum() < 8:
        return []
    med = _median_dt(fd)
    if med <= 0:
        return []
    idx = np.arange(fd["n"])
    series = []
    for key in ("hips_x", "hips_y"):
        arr = fd[key].copy()
        good = _finite(arr)
        if good.sum() < 2:
            return []
        arr[~good] = np.interp(idx[~good], idx[good], arr[good])
        series.append(arr)
    half = max(1, int(round((_CAMERA_HP_WINDOW_S / 2.0) / med)))
    acc = 0.0
    for arr in series:
        hp = arr - _centered_ma(arr, half)
        acc += float(np.mean(hp[ok] ** 2))
    std = math.sqrt(acc)
    if std <= cfg["camera_motion_std_px"]:
        return []
    valid = np.flatnonzero(ok)
    return [
        _diag(DiagReason.CAMERA_MOTION_HIGH, fd["times"][valid[0]], fd["times"][valid[-1]],
              f"high-pass hip stddev {std:.2f}px > {cfg['camera_motion_std_px']:.0f}px (non-fatal)")
    ]


def _shot_boundary_diagnostics(fd: dict, cfg: dict) -> list[dict]:
    med = _median_dt(fd)
    diags = []
    t0, t1 = float(fd["times"][0]), float(fd["times"][-1])
    for b in cfg["shot_boundaries"]:
        if t0 <= b <= t1:
            diags.append(
                _diag(DiagReason.SHOT_BOUNDARY, b, b + med, f"shot boundary at {b:.3f}s")
            )
    return diags


# ---------------------------------------------------------------------------
# Segments (shot-boundary splits; per-foot machines reset at each cut)
# ---------------------------------------------------------------------------


def _segments(fd: dict, cfg: dict) -> list[np.ndarray]:
    n = fd["n"]
    cuts = {0, n}
    for b in cfg["shot_boundaries"]:
        i = int(np.searchsorted(fd["times"], b - 1e-9))  # first frame with t >= b
        if 0 < i < n:
            cuts.add(i)
    edges = sorted(cuts)
    return [np.arange(edges[k], edges[k + 1])
            for k in range(len(edges) - 1) if edges[k + 1] > edges[k]]


# ---------------------------------------------------------------------------
# Vertical envelope (hybrid only)
# ---------------------------------------------------------------------------


def _env_heights(fd: dict, seg: np.ndarray, side: str) -> np.ndarray:
    """Height of the foot reference above the running support band.

    Image y grows downward, so the support band is the running MAXIMUM of
    the reference y within a +/-0.5 s centred window (the floor-most the
    foot gets locally); height = band_y - foot_y, zero while planted.
    A running envelope -- explicitly NOT a reconstructed ground plane.
    """

    ys = fd["feet"][side]["ry"][seg]
    heights = np.full(len(seg), np.nan)
    med = _median_dt(fd)
    half = max(1, int(round(0.5 / med))) if med > 0 else 3
    for p in range(len(seg)):
        if not np.isfinite(ys[p]):
            continue
        window = ys[max(0, p - half):p + half + 1]
        window = window[_finite(window)]
        if window.size:
            heights[p] = float(np.max(window)) - ys[p]
    return heights


# ---------------------------------------------------------------------------
# Velocity / hybrid state machine
# ---------------------------------------------------------------------------


def _visibility_at(arr: dict, gi: int, key: str) -> float | None:
    v = arr[key][gi]
    return round(float(v), 3) if np.isfinite(v) else None


def _velocity_candidates(
    fd: dict,
    seg: np.ndarray,
    side: str,
    cfg: dict,
    scale: float,
    method: str,
) -> list[_Cand]:
    """Run the per-foot offline state machine over one segment."""

    arr = fd["feet"][side]
    times = fd["times"]
    m = len(seg)
    rx = arr["rx"][seg]
    ry = arr["ry"][seg]
    usable = arr["usable"][seg]

    # Pairwise speed stamped at the later frame of each pair (body-scale/s);
    # dt comes from the actual timestamps -- never from an fps assumption.
    spd = np.full(m, np.nan)
    for p in range(1, m):
        if usable[p] and usable[p - 1]:
            dt = times[seg[p]] - times[seg[p - 1]]
            if dt > 0:
                moved = math.hypot(rx[p] - rx[p - 1], ry[p] - ry[p - 1])
                spd[p] = moved / dt / scale

    heights = _env_heights(fd, seg, side) if method == "hybrid" else None
    env_tol = _ENV_TOL_FRAC * scale
    min_stance_s = _MIN_STANCE_FRAC * cfg["min_step_interval_s"]
    confirm = cfg["confirm_frames"]

    def thr_now() -> float:
        return cfg["speed_drop_frac"] * max(peak_holder[0], _MIN_PEAK_BS_S)

    cands: list[_Cand] = []
    state = _UNKNOWN
    peak_holder = [0.0]
    run_start = -1        # segment position where the current low run began
    relaxed_run = False   # run opened out of a machine-confirmed stance's EXIT

    def close_run(p_end: int, p_break: int | None = None) -> None:
        """Evaluate the finished low-speed run; confirm a contact or abstain.

        ``p_break`` is the segment position that ended the run (the first
        high-speed or invalid sample); observed stationarity spans from the
        run's first supporting frame up to that frame.
        """
        nonlocal state, run_start, relaxed_run
        p0, was_relaxed = run_start, relaxed_run
        run_start = -1
        relaxed_run = False
        # Conservative reseed on any rejection: UNKNOWN never emits, so a
        # rejected run can only cost us a miss, never a wrong-time event.
        state = _UNKNOWN

        n_frames = p_end - p0 + 1
        if n_frames < confirm:
            return
        gi = int(seg[p0])
        # Observed stationarity spans from the first frame of the opening
        # (already-planted) pair to the frame where motion resumed -- the
        # pair stamps alone would undercount by one dt.
        supp = int(seg[p0 - 1]) if p0 >= 1 else gi
        if p_break is not None and p_break < m:
            dur = float(times[seg[p_break]]) - float(times[supp])
        else:  # closed by end-of-data: count the last frame's exposure
            dur = float(times[seg[p_end]]) - float(times[supp]) + _local_dt(fd, gi)
        if not was_relaxed:
            if dur < min_stance_s:
                return
            # Ascent veto: rising into the slowdown means swing apex, not a
            # plant.  Needs the last pre-run sample; without it, abstain.
            if p0 < 1 or not np.isfinite(ry[p0 - 1]):
                return
            if (ry[p0 - 1] - ry[p0]) > _ASCENT_FRAC * scale:
                return
        h = None
        if heights is not None:
            window = heights[p0:p_end + 1]
            window = window[_finite(window)]
            if not window.size:
                return
            h = float(np.min(window))
            if h > env_tol:
                return

        thr = cfg["speed_drop_frac"] * max(peak_holder[0], _MIN_PEAK_BS_S)
        s0 = float(spd[p0])
        vel = min(1.0, max(0.0, 1.0 - s0 / max(thr, 1e-9)))
        persist = min(1.0, n_frames / (2.0 * confirm))
        vis_vals = [v for v in (arr["heel_vis"][gi], arr["ankle_vis"][gi],
                                arr["toe_vis"][gi]) if np.isfinite(v)]
        vis = min(1.0, max(0.0, float(np.mean(vis_vals)))) if vis_vals else 0.0
        if method == "hybrid":
            env = min(1.0, max(0.0, 1.0 - (h if h is not None else env_tol) / env_tol))
            score = round(0.40 * vel + 0.20 * persist + 0.25 * env + 0.15 * vis, 3)
        else:
            score = round(0.55 * vel + 0.30 * persist + 0.15 * vis, 3)

        evidence = {
            "speed_bs_s": round(s0, 4),
            "swing_peak_bs_s": round(max(peak_holder[0], _MIN_PEAK_BS_S), 4),
            "threshold_bs_s": round(thr, 4),
            "low_run_frames": int(n_frames),
            "low_run_s": round(dur, 4),
            "heel_vis": _visibility_at(arr, gi, "heel_vis"),
            "ankle_vis": _visibility_at(arr, gi, "ankle_vis"),
            "toe_vis": _visibility_at(arr, gi, "toe_vis"),
        }
        if method == "hybrid":
            evidence["height_above_band_px"] = round(h, 3) if h is not None else None
            hy, ty = arr["heel_y"][gi], arr["toe_y"][gi]
            if np.isfinite(hy) and np.isfinite(ty):
                diff = hy - ty  # image y grows downward: heel lower => heel-first
                eps = _STRIKE_EPS_FRAC * scale
                evidence["strike"] = ("heel" if diff > eps
                                      else "forefoot" if diff < -eps else "ambiguous")
            else:
                evidence["strike"] = "ambiguous"

        observable = bool(
            np.isfinite(arr["heel_vis"][gi])
            and arr["heel_vis"][gi] >= cfg["min_landmark_visibility"]
            and np.isfinite(arr["toe_vis"][gi])
            and arr["toe_vis"][gi] >= cfg["min_landmark_visibility"]
        )
        cands.append(_Cand(side=side, idx=gi, t=float(times[gi]),
                           score=min(1.0, max(0.0, score)),
                           observable=observable, evidence=evidence))
        stance_confirmed[0] = True
        state = _STANCE

    stance_confirmed = [False]  # True once a contact was actually confirmed
    prev_valid = -1
    for p in range(m):
        if not np.isfinite(spd[p]):
            if state == _CAND and prev_valid >= 0:
                close_run(prev_valid, p)
            continue
        sp = float(spd[p])
        if state == _UNKNOWN:
            if sp > thr_now():
                state = _SWING
                peak_holder[0] = sp
            else:
                # Seeded stance: the transition happened before the segment;
                # recorded so a following EXIT cannot relax confirmation.
                stance_confirmed[0] = False
                state = _STANCE
        elif state == _SWING:
            if sp >= thr_now():
                peak_holder[0] = max(peak_holder[0], sp)
            else:
                state = _CAND
                run_start = p
                relaxed_run = False
        elif state == _CAND:
            if sp >= thr_now():
                close_run(p - 1, p)
                peak_holder[0] = max(peak_holder[0], sp)
        elif state == _STANCE:
            if sp >= thr_now():
                state = _EXIT
        elif state == _EXIT:
            if sp < thr_now():
                # Re-plant right after an airborne moment.  Relaxation is
                # only safe when the preceding stance was machine-confirmed,
                # never when it was merely seeded at the segment start.
                state = _CAND
                run_start = p
                relaxed_run = stance_confirmed[0]
            elif sp > 0.7 * max(peak_holder[0], _MIN_PEAK_BS_S):
                state = _SWING
                peak_holder[0] = sp
        prev_valid = p

    if state == _CAND and prev_valid >= 0:
        close_run(prev_valid)
    return cands


# ---------------------------------------------------------------------------
# Pelvis (Zeni-style) method
# ---------------------------------------------------------------------------


def _pca_axis(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Principal axis of 2-D points plus the minor/major variance ratio."""

    centred = pts - pts.mean(axis=0)
    cov = (centred.T @ centred) / max(1, len(centred))
    evals, evecs = np.linalg.eigh(cov)
    major = int(np.argmax(evals))
    axis = evecs[:, major]
    minor = 1 - major
    ratio = float(evals[minor] / evals[major]) if evals[major] > 1e-12 else 0.0
    return axis, ratio


def _fundamental_period(vals: np.ndarray, dt_med: float,
                        lo_s: float, hi_s: float) -> float | None:
    """Fundamental oscillation period: FIRST autocorrelation local maximum.

    Taking the global maximum would lock onto subharmonics (any multiple of
    the true period scores ~1.0 on a periodic signal), which would place
    the smoothing null on the gait band itself.
    """

    v = vals - vals.mean()
    if len(v) < 12 or dt_med <= 0:
        return None
    ac = np.correlate(v, v, mode="full")[len(v) - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    for k in range(1, len(ac)):
        lag = k * dt_med
        if lag > hi_s:
            break
        if lag < lo_s:
            continue
        rising = ac[k] >= ac[k - 1]
        falling = (k + 1 >= len(ac)) or (ac[k] >= ac[k + 1])
        if rising and falling and ac[k] >= _PELVIS_ACF_MIN:
            return float(lag)
    return None


def _pelvis_candidates(
    fd: dict,
    seg: np.ndarray,
    cfg: dict,
    scale: float,
    diags: list[dict],
) -> list[_Cand]:
    """Zeni-style candidates per segment; abstains on degenerate geometry."""

    times = fd["times"]
    med = _median_dt(fd)
    cands: list[_Cand] = []
    hip_sel = seg[fd["hips_ok"][seg]]
    if len(hip_sel) < 8 or med <= 0:
        return cands

    pts = np.column_stack([fd["hips_x"][hip_sel], fd["hips_y"][hip_sel]])
    axis, ratio = _pca_axis(pts)
    proj = pts @ axis
    prog = np.diff(proj)
    # Net direction must use the MEAN step, not the median: pelvis sway
    # velocity dwarfs drift per-sample, and the median of a discretely
    # sampled oscillation is biased off zero, which can invert the axis.
    # Sway is near-zero-mean over whole periods, so the mean isolates drift.
    if prog.size and float(np.mean(prog)) < 0:
        axis = -axis
    span = float((pts @ axis).max() - (pts @ axis).min())

    if ratio > _PELVIS_RATIO_MAX or span < _PELVIS_SPAN_FRAC * scale:
        diags.append(
            _diag(DiagReason.AMBIGUOUS_SUPPORT_PHASE,
                  float(times[seg[0]]), float(times[seg[-1]]),
                  f"degenerate hip geometry: variance ratio {ratio:.3f}, "
                  f"axis span {span:.1f}px (scale {scale:.1f}px)")
        )
        return cands

    confirm = cfg["confirm_frames"]
    min_step = cfg["min_step_interval_s"]
    max_step = cfg["max_step_interval_s"]

    for side in SIDES:
        arr = fd["feet"][side]
        sel = seg[(fd["hips_ok"][seg]) & arr["usable"][seg]]
        if len(sel) < 12:
            continue
        # Anterior/posterior ankle-relative-to-hips offset along the axis.
        f_raw = ((arr["rx"][sel] - fd["hips_x"][sel]) * axis[0]
                 + (arr["ry"][sel] - fd["hips_y"][sel]) * axis[1])

        seg_dur = float(times[sel[-1]] - times[sel[0]])
        period = _fundamental_period(
            f_raw, med,
            lo_s=0.9 * min_step,
            hi_s=min(1.2 * max_step, max(seg_dur / 2.0, 2 * min_step)),
        )
        # The dominant offset oscillation IS the step-frequency pelvis sway;
        # the per-foot strike content sits at half that rate, so a centred
        # MA spanning exactly one sway period nulls the sway band (odd tap
        # count keeps zero phase; an even period drops one tap, residual
        # ~sin(pi/k), tiny) while the strike band survives with gain ~0.64.
        # Without a confident period estimate, fall back to min_step.
        window = period if period else min_step
        window = min(max(window, min_step), max_step)
        k = max(3, int(round(window / med)))
        half = max(1, (k - 1) // 2)
        sm = _centered_ma(f_raw, half)

        # Local maxima = deceleration turning points of the offset.
        peaks = [p for p in range(1, len(sm) - 1)
                 if sm[p] >= sm[p - 1] and sm[p] >= sm[p + 1]
                 and (sm[p] > sm[p - 1] or sm[p] > sm[p + 1])]
        if not peaks:
            continue
        # Minimum-time-separation greedy selection, tallest first.
        kept: list[int] = []
        for p in sorted(peaks, key=lambda q: (-sm[q], q)):
            t_p = float(times[sel[p]])
            if all(abs(t_p - float(times[sel[q]])) >= min_step for q in kept):
                kept.append(p)
        kept.sort()

        prom_floor = _PELVIS_PROM_FRAC * scale
        for j, p in enumerate(kept):
            left_lo = kept[j - 1] if j > 0 else 0
            right_hi = kept[j + 1] if j + 1 < len(kept) else len(sm)
            left_valley = float(np.min(sm[left_lo:p + 1]))
            right_valley = float(np.min(sm[p:right_hi]))
            prom = float(sm[p]) - max(left_valley, right_valley)
            if prom < prom_floor:
                continue
            # Confirmation: the offset must keep declining (body progressing
            # over a planted foot) for confirm_frames further samples; a
            # trace ending too soon for that abstains.
            if p + confirm >= len(sm):
                continue
            if not all(sm[p + k] < sm[p] for k in range(1, confirm + 1)):
                continue

            gi = int(sel[p])
            vis_vals = [v for v in (arr["heel_vis"][gi], arr["ankle_vis"][gi],
                                    arr["toe_vis"][gi]) if np.isfinite(v)]
            vis = min(1.0, max(0.0, float(np.mean(vis_vals)))) if vis_vals else 0.0
            score = round(min(1.0, max(0.0,
                        0.7 * min(1.0, prom / (0.04 * scale)) + 0.3 * vis)), 3)
            observable = bool(
                np.isfinite(arr["heel_vis"][gi])
                and arr["heel_vis"][gi] >= cfg["min_landmark_visibility"]
                and np.isfinite(arr["toe_vis"][gi])
                and arr["toe_vis"][gi] >= cfg["min_landmark_visibility"]
            )
            cands.append(
                _Cand(side=side, idx=gi, t=float(times[gi]), score=score,
                      observable=observable,
                      evidence={
                          "rel_offset_px": round(float(f_raw[p]), 3),
                          "prominence_px": round(prom, 3),
                          "smooth_window_s": round(window, 4),
                          "walk_dir": [round(float(axis[0]), 4),
                                       round(float(axis[1]), 4)],
                          "hips_variance_ratio": round(ratio, 4),
                          "heel_vis": _visibility_at(arr, gi, "heel_vis"),
                          "ankle_vis": _visibility_at(arr, gi, "ankle_vis"),
                          "toe_vis": _visibility_at(arr, gi, "toe_vis"),
                      })
            )
    return cands


# ---------------------------------------------------------------------------
# Post-processing: duplicates, cadence, alternation
# ---------------------------------------------------------------------------


def _suppress_duplicates(cands: list[_Cand], min_step: float) -> list[_Cand]:
    """Same-foot candidates within min_step collapse to the higher score."""

    kept: list[_Cand] = []
    for c in sorted(cands, key=lambda x: (x.t, x.side)):
        if kept and kept[-1].side == c.side and c.t - kept[-1].t < min_step:
            if c.score > kept[-1].score:
                kept[-1] = c
            continue
        kept.append(c)
    kept.sort(key=lambda x: (x.t, x.side))
    return kept


def _cadence_and_alternation(events: list[_Cand], cfg: dict, diags: list[dict]) -> None:
    """Scan the ordered per-segment event list for cadence/alternation faults."""

    if len(events) < 2:
        return
    pairs = list(zip(events, events[1:]))
    gaps = [b.t - a.t for a, b in pairs]
    mn, mx = cfg["min_step_interval_s"], cfg["max_step_interval_s"]
    for (a, b), gap in zip(pairs, gaps):
        if gap < mn or gap > mx:
            diags.append(
                _diag(DiagReason.CADENCE_VIOLATION, a.t, b.t,
                      f"{a.side}->{b.side} inter-contact interval {gap:.3f}s "
                      f"outside [{mn:.2f},{mx:.2f}]s")
            )
    median_gap = float(np.median(gaps))
    for (a, b), gap in zip(pairs, gaps):
        if a.side == b.side and gap < 0.5 * median_gap:
            diags.append(
                _diag(DiagReason.LEFT_RIGHT_DISCONTINUITY, a.t, b.t,
                      f"two {a.side}-foot events {gap:.3f}s apart without the "
                      f"opposite foot (< 0.5 * median {median_gap:.3f}s)")
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def detect_contacts(
    trace: PoseTrace, method: str = "hybrid", *, config: dict | None = None
) -> ContactMethodResult:
    """Detect individual foot contacts in ``trace`` with the given method.

    See the module docstring for the per-method algorithms and the
    abstention/observability policies.  Unknown methods and unknown config
    keys raise ``ValueError``.  ``trace`` is treated as read-only.
    """

    if method not in CONTACT_METHODS:
        raise ValueError(
            f"unknown contact method {method!r}; expected one of {CONTACT_METHODS}"
        )
    cfg = _validated_config(method, config)
    diags: list[dict] = []

    if not trace.frames:
        return ContactMethodResult(events=[], diagnostics=diags)

    fd = _extract(trace, cfg["min_landmark_visibility"])
    diags.extend(_shot_boundary_diagnostics(fd, cfg))

    scale = _body_scale(fd, cfg["min_landmark_visibility"])
    diags.extend(_observability_diagnostics(fd, cfg, scale))
    if scale is None or not np.isfinite(scale) or scale <= 0:
        # No body-scale reference anywhere in the trace: emit nothing.
        diags.append(
            _diag(DiagReason.FEET_OUT_OF_FRAME, float(fd["times"][0]),
                  float(fd["times"][-1]),
                  "no body-scale reference measurable; no contacts inferred")
        )
        return ContactMethodResult(events=[], diagnostics=_sort_diags(diags))

    seg_bounds = _segments(fd, cfg)
    all_cands: list[_Cand] = []
    for seg in seg_bounds:
        if method in ("velocity", "hybrid"):
            for side in SIDES:
                all_cands.extend(_velocity_candidates(fd, seg, side, cfg, scale, method))
        else:  # pelvis
            all_cands.extend(_pelvis_candidates(fd, seg, cfg, scale, diags))

    final_cands: list[_Cand] = []
    for seg in seg_bounds:
        members = [c for c in all_cands if seg[0] <= c.idx <= seg[-1]]
        # Actor-too-small frames invalidate any candidate inside them.
        members = [
            c for c in members
            if not _in_too_small_span(fd, c, cfg)
        ]
        members = _suppress_duplicates(members, cfg["min_step_interval_s"])
        _cadence_and_alternation(members, cfg, diags)
        final_cands.extend(members)

    final_cands.sort(key=lambda c: (c.t, c.side))
    detector = f"{trace.backend}-contact-{method}-v0"
    events: list[ContactEvent] = []
    for rank, c in enumerate(final_cands, start=1):
        events.append(
            ContactEvent(
                id=f"evt_{rank:03d}",
                family=EVENT_FAMILY_FOOT_CONTACT,
                actor_id=DOMINANT_ACTOR_ID,
                foot=c.side,
                source_pts_us=s_to_pts_us(c.t),
                time_s=round(c.t, _ROUND),
                earliest_s=round(c.t, _ROUND),
                latest_s=round(c.t + _local_dt(fd, c.idx), _ROUND),
                score=c.score,
                observable=c.observable,
                detector=detector,
                method=method,
                evidence=c.evidence,
                kind="initial_contact",
            )
        )

    return ContactMethodResult(events=events, diagnostics=_sort_diags(diags))


def _in_too_small_span(fd: dict, cand: _Cand, cfg: dict) -> bool:
    """True when the candidate's own frame measures below the actor minimum."""

    fscale = _frame_scale(fd, cand.side)[cand.idx]
    return bool(np.isfinite(fscale) and fscale < cfg["actor_too_small_px"])
