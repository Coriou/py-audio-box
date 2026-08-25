"""Per-event forensic windows over cached pose traces (Phase-2 T2).

For every predicted event and every missed GT contact, emit one machine-
readable JSON window (schema docs/auto-sfx-phase2-truth-repair.md §5.2) and
one PNG plot, plus one summary PNG stacking consecutive gait cycles. Consumes
ONLY run artifacts (pose.jsonl, events.json, contact-trace.jsonl) and the GT
file — never re-runs pose inference.

Exact vs reconstructed (honesty notes, by design):
- When ``contact-trace.jsonl`` exists (written by ``analyze`` with tracing
  on), per-frame machine states, transitions, thresholds, support-band
  heights and reference speeds are read EXACTLY from it; every window is
  stamped ``win["trace"] = {"source": "contact-trace.jsonl", "exact": true}``
  and frames the trace does not cover stay ``None`` (never interpolated).
- Without a trace file nothing is invented: states fall back to the coarse
  ``"PRED_WINDOW"``/``"-"`` approximation derived from predicted-event
  windows, ``transitions`` stays empty, thresholds are method defaults;
  the window then carries ``win["trace"] = {"source": "none",
  "exact": false}`` and a note saying plainly that these are NOT the
  detector's internal machine.
- Representation choice (this module's documented contract): exact-mode
  per-frame thresholds live under ``series["thr_bs_s"][side]`` (aligned
  element-for-element with ``series["t_s"]``), while the scalar active
  config lives in ``series["thresholds"]``;
  ``series["support_band_score"]`` remains max-over-sides with
  ``series["support_band_score_by_side"]`` splitting it per side and
  ``series["support_band_source"]`` naming provenance.
- ``conf_smoothed`` mirrors ``conf_raw``: this module applies no smoothing;
  any temporal filtering lives in the pose backend (T8 measures it).
- GT overlay and missed-GT selection use the shared one-to-one diagnostic
  matcher (:func:`auto_sfx.association.associate_diagnostic`, horizon 0.5 s)
  over ALL events x GT once per run; side is never a blocking criterion —
  disagreement is reported as ``side_ok: false``, not dropped. This module
  keeps no private matcher of its own.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

try:  # package-relative when imported as auto_sfx.forensics
    from .association import DIAGNOSTIC_HORIZON_S, associate_diagnostic
    from .cache import TRACE_FORMAT
    from .contacts import _env_heights, default_config
    from .metrics import load_gt_any
    from .types import SIDES, PoseTrace, VideoMeta, s_to_pts_us
except ImportError:  # pragma: no cover - direct module import
    from association import (  # type: ignore[no-redef]
        DIAGNOSTIC_HORIZON_S, associate_diagnostic,
    )
    from cache import TRACE_FORMAT  # type: ignore[no-redef]
    from contacts import _env_heights, default_config  # type: ignore
    from metrics import load_gt_any  # type: ignore[no-redef]
    from types import (  # type: ignore
        SIDES, PoseTrace, VideoMeta, s_to_pts_us,
    )

import numpy as np


SCHEMA_VERSION = 1
#: Diagnostic horizon for GT overlay / missed-GT selection. Aliased to the
#: shared matcher's ceiling so every Phase-2 artifact links identically.
GT_MATCH_HORIZON_S = DIAGNOSTIC_HORIZON_S

FORENSICS_SCHEMA_NOTE = (
    "no contact-trace.jsonl in run dir: states are APPROXIMATED from "
    "predicted-event windows ('PRED_WINDOW'), transitions cannot be "
    "recovered, thresholds are method defaults -- none of these are the "
    "detector's internal machine"
)
TRACE_EXACT_NOTE = (
    "states/transitions/thresholds/support-band/reference speeds are exact "
    "per-frame values read from contact-trace.jsonl; frames the trace does "
    "not cover stay null"
)

# Ribbon colors for exact machine states in plot_window.
_STATE_COLORS = {
    "STANCE": "#2ca02c",     # green
    "CANDIDATE": "#ffbf00",  # amber
    "SWING": "#1f77b4",      # blue
    "EXIT": "#9467bd",       # purple
    "UNKNOWN": "#7f7f7f",    # gray
}


# ---------------------------------------------------------------------------
# Run-artifact loading
# ---------------------------------------------------------------------------


def load_run_trace(run_dir: str | Path) -> tuple[PoseTrace, dict, dict]:
    """(trace, events_document, diagnostics_document) from one run dir.

    pose.jsonl mirrors the PoseCache on-disk format under a fixed name, so
    it is parsed directly (the cache loader keys entries by filename).
    """

    run_dir = Path(run_dir)
    events_doc = json.loads((run_dir / "events.json").read_text("utf-8"))
    diag_path = run_dir / "diagnostics.json"
    diag_doc = (
        json.loads(diag_path.read_text("utf-8")) if diag_path.is_file() else {}
    )
    lines = (run_dir / "pose.jsonl").read_text("utf-8").splitlines()
    if not lines:
        raise ValueError(f"{run_dir}/pose.jsonl: empty file")
    header = json.loads(lines[0])
    if header.get("format") != TRACE_FORMAT:
        raise ValueError(
            f"{run_dir}/pose.jsonl: unexpected format {header.get('format')!r}"
        )
    meta = VideoMeta(**header["meta"])
    rows = [json.loads(line) for line in lines[1:]]
    trace = PoseTrace.from_rows(header["backend"], header["mode"], meta, rows)
    return trace, events_doc, diag_doc


def load_contact_trace(run_dir: str | Path) -> tuple[list[dict] | None, dict | None]:
    """(rows, header) from ``<run_dir>/contact-trace.jsonl`` or (None, None).

    ``rows`` excludes the leading header line; ``header`` is that line's
    dict (``kind == "header"``). Absent/empty file -> ``(None, None)`` so
    callers can say honestly that no machine internals were persisted.
    """

    path = Path(run_dir) / "contact-trace.jsonl"
    if not path.is_file():
        return None, None
    rows = [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None, None
    header = rows[0] if rows[0].get("kind") == "header" else None
    data = rows[1:] if header is not None else rows
    return data, header


class ContactTraceIndex:
    """Frame-keyed view over contact-trace.jsonl (exact machine internals).

    Built once per run, passed to every :func:`build_window` call. Frame
    keys are the detector's global frame indices -- the same positional
    indexing as :class:`auto_sfx.types.PoseTrace.frames`, because the
    writer recorded ``seg[p]`` positions of the very trace it consumed.
    Frames without a velocity/hybrid row (pelvis-method rows, gaps,
    out-of-segment frames) are simply absent from the maps; consumers
    must surface them as ``None``, never interpolate.
    """

    def __init__(self, rows: list[dict], header: dict | None):
        self.rows = rows
        self.header = header
        cfg = (header or {}).get("config") or {}
        self.config: dict = dict(cfg)
        scale = (header or {}).get("scale_px")
        try:
            self.scale_px: float | None = (
                float(scale) if scale and float(scale) > 0 else None
            )
        except (TypeError, ValueError):
            self.scale_px = None
        #: side -> {global frame index -> exact machine state}
        self.state: dict[str, dict[int, str]] = {s: {} for s in SIDES}
        #: side -> {global frame index -> threshold actually used}
        self.thr_bs_s: dict[str, dict[int, float]] = {s: {} for s in SIDES}
        #: side -> {global frame index -> reference foot speed (body-scale/s)}
        self.spd_bs_s: dict[str, dict[int, float]] = {s: {} for s in SIDES}
        #: side -> {global frame index -> support band height in px}
        self.band_h_px: dict[str, dict[int, float]] = {s: {} for s in SIDES}
        #: real state transitions, sorted by frame: {"side","frame","pts_us","from","to"}
        self.transitions: list[dict] = []
        for r in rows:
            if r.get("kind") != "frame":
                continue
            side = r.get("side")
            if side not in self.state:
                continue
            fi = r.get("frame")
            if not isinstance(fi, int):
                continue
            st = r.get("state_after", r.get("state_before"))
            if st is not None:
                self.state[side][fi] = str(st)
                tr = r.get("transition")
                if isinstance(tr, dict) and "from" in tr and "to" in tr:
                    self.transitions.append({
                        "side": side, "frame": fi,
                        "pts_us": r.get("pts_us"),
                        "from": tr["from"], "to": tr["to"],
                    })
            for key, sink in (("thr_bs_s", self.thr_bs_s),
                              ("spd_bs_s", self.spd_bs_s),
                              ("band_h_px", self.band_h_px)):
                val = r.get(key)
                if isinstance(val, (int, float)) and math.isfinite(val):
                    sink[side][fi] = float(val)
        self.transitions.sort(key=lambda t: (t["frame"], t["side"]))

    @classmethod
    def from_rows(
        cls, rows: list[dict] | None, header: dict | None,
    ) -> "ContactTraceIndex | None":
        """Index from :func:`load_contact_trace` output; None when absent."""

        if rows is None:
            return None
        return cls(rows, header)

    def has_states(self) -> bool:
        """True when at least one velocity/hybrid machine state was traced."""

        return any(self.state[s] for s in SIDES)


def _pca_axis(points: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0)
    cov = (centred.T @ centred) / max(1, len(centred))
    evals, evecs = np.linalg.eigh(cov)
    return evecs[:, int(np.argmax(evals))]


def _pairwise_speed(vals: np.ndarray, times: np.ndarray, scale: float):
    out = np.full(len(vals), np.nan)
    for i in range(1, len(vals)):
        if np.isfinite(vals[i]) and np.isfinite(vals[i - 1]):
            dt = times[i] - times[i - 1]
            if dt > 0:
                out[i] = abs(vals[i] - vals[i - 1]) / dt / scale
    return out


def _frame_body_scale(trace: PoseTrace) -> np.ndarray:
    """Per-frame mean hip->usable-foot-reference distance (NaN if absent)."""

    n = len(trace.frames)
    out = np.full(n, np.nan)
    for i, fr in enumerate(trace.frames):
        if fr.hips is None or fr.lost:
            continue
        dists = []
        for side in SIDES:
            obs = (fr.feet or {}).get(side)
            if obs is None:
                continue
            marks = [p for p in (obs.heel, obs.ankle, obs.toe)
                     if p is not None and p.visibility >= 0.5]
            if len(marks) >= 2:
                rx = sum(p.x for p in marks) / len(marks)
                ry = sum(p.y for p in marks) / len(marks)
                dists.append(math.hypot(fr.hips.x - rx, fr.hips.y - ry))
        if dists:
            out[i] = float(np.mean(dists))
    return out


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def build_window(
    trace: PoseTrace,
    events: list[dict],
    gt_records: list[dict] | None,
    center_pts_us: int,
    *,
    kind: str,
    half_s: float = 0.65,
    method: str = "hybrid",
    trace_index: ContactTraceIndex | None = None,
    gt_record: dict | None = None,
    assoc_pair: dict | None = None,
) -> dict:
    """One §5.2 forensic window around ``center_pts_us``.

    ``trace_index`` (exact machine internals) is supplied by
    :func:`build_forensics`; when omitted the window falls back to the
    honest approximation described in the module docstring. ``gt_record``
    / ``assoc_pair`` carry the shared diagnostic matcher's decision; when
    both are None and GT is given, this window re-runs the SHARED matcher
    scoped to its own centre event (no private matching logic here).
    """

    frames = trace.frames
    times = np.array([fr.t_s for fr in frames], dtype=float)
    pts_us = np.array([fr.source_pts_us for fr in frames], dtype=np.int64)
    scale_global = _frame_body_scale(trace)
    med_scale = float(np.nanmedian(scale_global)) if np.isfinite(
        scale_global).any() else 1.0

    t_center = center_pts_us / 1_000_000.0
    lo, hi = t_center - half_s, t_center + half_s
    idx = np.flatnonzero((times >= lo) & (times <= hi))
    if idx.size == 0:
        raise ValueError(f"no frames inside window [{lo:.3f}, {hi:.3f}]")

    # Walk axis over usable hips (same PCA family as the pelvis method).
    hip_ok = np.array(
        [fr.hips is not None and not fr.lost for fr in frames], dtype=bool
    )
    if hip_ok.sum() >= 8:
        pts = np.array(
            [(fr.hips.x, fr.hips.y) for fr, ok in zip(frames, hip_ok) if ok]
        )
        axis = _pca_axis(pts)
    else:
        axis = np.array([1.0, 0.0])

    exact = trace_index is not None
    band_exact = bool(
        exact and trace_index.scale_px is not None
        and any(trace_index.band_h_px[s] for s in SIDES)
    )
    if exact:
        thresholds = {
            "speed_drop_frac": trace_index.config.get("speed_drop_frac"),
            "min_peak_floor_bs_s": 0.4,
            "note": "exact active config from contact-trace.jsonl",
        }
    else:
        thresholds = {
            "speed_drop_frac": default_config(method)["speed_drop_frac"],
            "min_peak_floor_bs_s": 0.4,
            "note": "defaults; per-run --config overrides are not persisted",
        }
    series = {
        "t_s": [round(float(times[i]), 6) for i in idx],
        "pts_us": [int(pts_us[i]) for i in idx],
        "heel_xy": {}, "ankle_xy": {}, "toe_xy": {}, "foot_center_xy": {},
        "pelvis_xy": [], "body_scale_px": [],
        "conf_raw": {"heel": {}, "toe": {}},
        "conf_smoothed": {"heel": {}, "toe": {}},
        "vel": {"heel": {}, "toe": {}, "foot_center": {}},
        "accel": {"foot_center": {}},
        "pelvis_rel_long": [], "pelvis_rel_vert": [],
        "support_band_score": [],
        "state": {},
        "thresholds": thresholds,
    }
    if exact:
        # Exact per-frame thresholds, aligned element-for-element with t_s
        # (None wherever the trace has no row). Documented representation.
        series["thr_bs_s"] = {s: [] for s in SIDES}
    for side in SIDES:
        for key, attr in (("heel_xy", "heel"), ("ankle_xy", "ankle"),
                          ("toe_xy", "toe")):
            series[key][side] = [
                [round(fr.feet[side].__getattribute__(attr).x, 3),
                 round(fr.feet[side].__getattribute__(attr).y, 3)]
                if fr.feet.get(side) is not None
                and fr.feet[side].__getattribute__(attr) is not None else None
                for fr in (frames[i] for i in idx)
            ]
        # foot reference centre: mean of usable landmarks (>=2), like _extract
        centre = []
        for i in idx:
            fr = frames[i]
            obs = (fr.feet or {}).get(side)
            if obs is None:
                centre.append(None)
                continue
            marks = [p for p in (obs.heel, obs.ankle, obs.toe)
                     if p is not None and p.visibility >= 0.5]
            centre.append(
                [round(sum(p.x for p in marks) / len(marks), 3),
                 round(sum(p.y for p in marks) / len(marks), 3)]
                if len(marks) >= 2 else None
            )
        series["foot_center_xy"][side] = centre
        if exact:
            smap = trace_index.state[side]
            series["state"][side] = [smap.get(int(i)) for i in idx]
        else:
            series["state"][side] = [
                "PRED_WINDOW" if any(
                    e.get("foot") == side
                    and float(e.get("earliest_s", e["time_s"]))
                    <= times[i] <= float(e.get("latest_s", e["time_s"]))
                    for e in events
                ) else "-"
                for i in idx
            ]

    if exact:
        for side in SIDES:
            tmap = trace_index.thr_bs_s[side]
            series["thr_bs_s"][side] = [tmap.get(int(i)) for i in idx]

    seg = np.asarray(idx, dtype=int)
    for side in SIDES:
        # foot-centre trajectory -> speed (body-scale/s) + acceleration
        cx = np.full(len(seg), np.nan)
        cy = np.full(len(seg), np.nan)
        for k, i in enumerate(seg):
            fr = frames[i]
            obs = (fr.feet or {}).get(side)
            if obs is None:
                continue
            marks = [p for p in (obs.heel, obs.ankle, obs.toe)
                     if p is not None and p.visibility >= 0.5]
            if len(marks) >= 2:
                cx[k] = sum(p.x for p in marks) / len(marks)
                cy[k] = sum(p.y for p in marks) / len(marks)
        speed = np.full(len(seg), np.nan)
        for k in range(1, len(seg)):
            i0, i1 = seg[k - 1], seg[k]
            if np.isfinite(cx[k]) and np.isfinite(cx[k - 1]):
                dt = times[i1] - times[i0]
                if dt > 0:
                    speed[k] = math.hypot(
                        cx[k] - cx[k - 1], cy[k] - cy[k - 1]) / dt / med_scale
        series["vel"]["foot_center"][side] = [
            round(float(v), 4) if np.isfinite(v) else None for v in speed
        ]
        accel = np.full(len(seg), np.nan)
        for k in range(2, len(seg)):
            dt = times[seg[k]] - times[seg[k - 1]]
            if np.isfinite(speed[k]) and np.isfinite(speed[k - 1]) and dt > 0:
                accel[k] = (speed[k] - speed[k - 1]) / dt
        series["accel"]["foot_center"][side] = [
            round(float(v), 4) if np.isfinite(v) else None for v in accel
        ]
    # heel/toe speeds from y-only displacement (the vertical component is
    # the gait-discriminative one in most views)
    for side in SIDES:
        for key in ("heel", "toe"):
            vals = np.array([
                (fr.feet[side].__getattribute__(key).y
                 if fr.feet.get(side) is not None
                 and fr.feet[side].__getattribute__(key) is not None else np.nan)
                for fr in (frames[i] for i in seg)
            ], dtype=float)
            v = _pairwise_speed(vals, times[seg], med_scale)
            series["vel"][key][side] = [
                round(float(x), 4) if np.isfinite(x) else None for x in v
            ]
    series["pelvis_xy"] = [
        [round(fr.hips.x, 3), round(fr.hips.y, 3)]
        if fr.hips is not None and not fr.lost else None
        for fr in (frames[i] for i in idx)
    ]
    series["body_scale_px"] = [
        round(float(scale_global[i]), 3) if np.isfinite(scale_global[i]) else None
        for i in idx
    ]
    series["pelvis_rel_long"] = [
        round(float((fr.hips.x - _foot_ref_x(fr)) * axis[0]
                    + (fr.hips.y - _foot_ref_y(fr)) * axis[1]), 3)
        if fr.hips is not None and not fr.lost else None
        for fr in (frames[i] for i in idx)
    ]
    series["pelvis_rel_vert"] = [
        round(float(fr.hips.y - _foot_ref_y(fr)), 3)
        if fr.hips is not None and not fr.lost else None
        for fr in (frames[i] for i in idx)
    ]

    # support band: envelope height / scale per side. With a trace this is
    # the detector's own band_h_px over the header scale; without, a pose
    # reconstruction (marked as such -- never presented as machine truth).
    if band_exact:
        sc = trace_index.scale_px
        by_side = {
            s: [
                round(trace_index.band_h_px[s][int(i)] / sc, 4)
                if int(i) in trace_index.band_h_px[s] else None
                for i in idx
            ]
            for s in SIDES
        }
        series["support_band_source"] = (
            "contact-trace.jsonl#band_h_px/scale_px"
        )
    else:
        n_all = len(frames)
        fake_fd = _mini_fd(trace)
        by_side = {}
        for side in SIDES:
            env = _env_heights(fake_fd, np.arange(n_all), side)
            by_side[side] = [
                round(float(env[int(i)] / med_scale), 4)
                if np.isfinite(env[int(i)]) else None
                for i in idx
            ]
        series["support_band_source"] = (
            "pose reconstruction (_env_heights / median body scale)"
        )
    series["support_band_by_side"] = by_side
    series["support_band_score"] = [
        max((v for v in (by_side[SIDES[0]][k], by_side[SIDES[1]][k])
             if v is not None), default=None)
        for k in range(len(idx))
    ]
    if exact:
        # Detector's own per-frame reference speeds (body-scale/s).
        series["vel"]["reference"] = {
            s: [trace_index.spd_bs_s[s].get(int(i)) for i in idx]
            for s in SIDES
        }

    # Real state transitions inside the window (exact trace only).
    transitions: list[dict] = []
    if exact:
        f_lo, f_hi = int(idx[0]), int(idx[-1])
        transitions = [
            t for t in trace_index.transitions
            if f_lo <= t["frame"] <= f_hi
        ]

    # GT overlay via the SHARED diagnostic matcher (side-agnostic; side
    # agreement is reported, never used as a blocking filter).
    matching = [e for e in events
                if int(e.get("source_pts_us", -1)) == int(center_pts_us)]
    center_side = matching[0].get("foot") if matching else None
    pair = assoc_pair
    rec = gt_record
    if kind == "predicted":
        if rec is None and pair is None and gt_records:
            center_event = {
                "id": matching[0].get("id") if matching else None,
                "time_s": t_center,
                "foot": center_side,
            }
            res = associate_diagnostic(
                [center_event], gt_records, horizon_s=GT_MATCH_HORIZON_S
            )
            if res.pairs:
                pair = res.pairs[0]
                rec = gt_records[pair["gt_idx"]]
    elif kind == "missed_gt" and rec is None and gt_records:
        # The window centre was derived from a GT preferred time.
        for r in gt_records:
            if abs(s_to_pts_us(r["preferred_s"]) - int(center_pts_us)) <= 1:
                rec = r
                break

    gt_overlay = None
    if rec is not None:
        gt_overlay = {
            "gt_id": rec.get("gt_id"),
            "foot": rec["foot"],
            "earliest_pts_us": s_to_pts_us(
                rec["preferred_s"] if rec.get("earliest_s") is None
                else rec["earliest_s"]),
            "preferred_pts_us": s_to_pts_us(rec["preferred_s"]),
            "latest_pts_us": s_to_pts_us(
                rec["preferred_s"] if rec.get("latest_s") is None
                else rec["latest_s"]),
            "visibility": rec.get("visibility"),
            "viewpoint": rec.get("viewpoint"),
        }
        if pair is not None:
            gt_overlay.update({
                "assoc_err_s": pair.get("err_s"),
                "assoc_err_frames": pair.get("err_frames"),
                "assoc_inside_gt_window": pair.get("inside_gt_window"),
                "assoc_side_ok": pair.get("side_ok"),
            })

    notes = []
    if exact:
        notes.append(TRACE_EXACT_NOTE)
        if not trace_index.has_states():
            notes.append(
                "trace present but has no velocity/hybrid frame rows "
                "(pelvis method?): machine states unavailable"
            )
    else:
        notes.append(FORENSICS_SCHEMA_NOTE)

    return _nan_to_none({
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "center_pts_us": int(center_pts_us),
        "window_s": [round(lo, 4), round(hi, 4)],
        "backend": trace.backend,
        "method": method,
        "trace": (
            {"source": "contact-trace.jsonl", "exact": True}
            if exact else {"source": "none", "exact": False}
        ),
        "series": series,
        "transitions": transitions,
        "notes": notes,
        "gt": gt_overlay,
    })


def _foot_ref_x(fr):
    obs_l = (fr.feet or {}).get("left")
    obs_r = (fr.feet or {}).get("right")
    xs = []
    for obs in (obs_l, obs_r):
        if obs is None:
            continue
        marks = [p for p in (obs.heel, obs.ankle, obs.toe)
                 if p is not None and p.visibility >= 0.5]
        if len(marks) >= 2:
            xs.append(sum(p.x for p in marks) / len(marks))
    return float(np.mean(xs)) if xs else np.nan


def _foot_ref_y(fr):
    obs_l = (fr.feet or {}).get("left")
    obs_r = (fr.feet or {}).get("right")
    ys = []
    for obs in (obs_l, obs_r):
        if obs is None:
            continue
        marks = [p for p in (obs.heel, obs.ankle, obs.toe)
                 if p is not None and p.visibility >= 0.5]
        if len(marks) >= 2:
            ys.append(sum(p.y for p in marks) / len(marks))
    return float(np.mean(ys)) if ys else np.nan


def _mini_fd(trace: PoseTrace) -> dict:
    """Rebuild the minimal fd dict ``_env_heights`` needs (rx/ry per side)."""

    n = len(trace.frames)
    frames = trace.frames
    times = np.array([fr.t_s for fr in frames], dtype=float)
    dt = np.full(n, np.nan)
    for i in range(1, n):
        if times[i] > times[i - 1]:
            dt[i] = times[i] - times[i - 1]
    feet = {}
    for side in SIDES:
        rx = np.full(n, np.nan)
        ry = np.full(n, np.nan)
        for i, fr in enumerate(trace.frames):
            obs = (fr.feet or {}).get(side)
            if obs is None:
                continue
            marks = [p for p in (obs.heel, obs.ankle, obs.toe)
                     if p is not None and p.visibility >= 0.5]
            if len(marks) >= 2:
                rx[i] = sum(p.x for p in marks) / len(marks)
                ry[i] = sum(p.y for p in marks) / len(marks)
        feet[side] = {"rx": rx, "ry": ry}
    return {"n": n, "times": times, "dt": dt, "feet": feet}


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


def select_windows(events: list[dict], gt_records: list[dict] | None,
                   event_id: str | None, horizon_s: float = GT_MATCH_HORIZON_S,
                   assoc_result=None):
    """Which (kind, center_pts_us, label) windows to emit.

    Every predicted event gets a window; every GT contact left unmatched
    by the shared one-to-one DIAGNOSTIC association additionally gets a
    ``missed_gt`` window. One-to-one means a prediction sitting between
    two GT contacts pairs with at most one of them -- the other is
    missed -- unlike the old any-prediction-in-horizon loop which hid it.
    Pass ``assoc_result`` (the :func:`associate_diagnostic` result) to
    reuse the run-level pass; without it one is computed here.
    """

    out = []
    preds = [e for e in events if event_id in (None, e.get("id"))]
    if event_id is not None and not preds:
        raise ValueError(f"event id {event_id!r} not found in events.json")
    for e in preds:
        out.append(("predicted", int(e["source_pts_us"]), e.get("id", "?")))
    if gt_records and event_id is None:
        res = assoc_result if assoc_result is not None else associate_diagnostic(
            events, gt_records, horizon_s=horizon_s)
        for u in res.unmatched_gts:
            r = gt_records[u["gt_idx"]]
            out.append((
                "missed_gt",
                s_to_pts_us(r["preferred_s"]),
                f"gt_{int(round(r['preferred_s'] * 1000))}ms",
            ))
    return out


# ---------------------------------------------------------------------------
# Plots (matplotlib, Agg backend)
# ---------------------------------------------------------------------------


def plot_window(win: dict, out_path: str | Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    s = win["series"]
    t = np.array([x if x is not None else np.nan for x in s["t_s"]])

    def arr(pairs):
        return np.array([
            [np.nan, np.nan] if p is None else p for p in pairs
        ], dtype=float)

    def _runs(values):
        """[[start, count, state]] over consecutive equal non-null states."""
        runs = []
        for k, v in enumerate(values):
            if v is None:
                continue
            if runs and runs[-1][2] == v and runs[-1][0] + runs[-1][1] == k:
                runs[-1][1] += 1
            else:
                runs.append([k, 1, v])
        return runs

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    colors = {"left": "tab:green", "right": "tab:orange"}
    for side in SIDES:
        c = colors[side]
        heel = arr(s["heel_xy"][side])
        centre = arr(s["foot_center_xy"][side])
        axes[0].plot(t, heel[:, 0], color=c, alpha=0.4, label=f"{side} heel x")
        axes[0].plot(t, centre[:, 0], color=c, label=f"{side} centre x")
        axes[1].plot(t, heel[:, 1], color=c, alpha=0.4, label=f"{side} heel y")
        axes[1].plot(t, centre[:, 1], color=c, label=f"{side} centre y")
        v = [x if x is not None else np.nan for x in s["vel"]["foot_center"][side]]
        axes[2].plot(t, v, color=c, label=f"{side} centre speed")

    # Threshold floor uses the window's OWN config value (exact when a
    # trace backs it, method default otherwise); 0.4 = min_peak_floor_bs_s.
    drop = (s.get("thresholds") or {}).get("speed_drop_frac")
    if isinstance(drop, (int, float)):
        axes[2].axhline(float(drop) * 0.4, color="gray", ls=":", lw=1,
                        label=f"thr floor ({drop}*0.4)")

    if win.get("gt"):
        g = win["gt"]
        for ax in axes:
            ax.axvspan(g["earliest_pts_us"] / 1e6, g["latest_pts_us"] / 1e6,
                       color="yellow", alpha=0.18)
            ax.axvline(g["preferred_pts_us"] / 1e6, color="gold", lw=2)
    pred_t = None
    if win["kind"] == "predicted":
        pred_t = win["center_pts_us"] / 1e6
        for ax in axes:
            ax.axvline(pred_t, color="red", lw=1.5)

    # Exact machine-state ribbon along the bottom of the speed axis
    # (only when the window is backed by contact-trace.jsonl).
    state_patches = []
    if win.get("trace", {}).get("exact"):
        trans = blended_transform_factory(
            axes[2].transData, axes[2].transAxes)
        strips = {"left": (0.02, 0.05), "right": (0.095, 0.05)}
        n = len(s["t_s"])
        seen = []
        for side in SIDES:
            y0, h = strips[side]
            states_side = s.get("state", {}).get(side) or []
            for start, count, st in _runs(states_side):
                i0, i1 = start, start + count - 1
                left = float(s["t_s"][i0])
                right = float(s["t_s"][i1])
                if i1 + 1 < n:
                    right += (float(s["t_s"][i1 + 1]) - right) / 2
                if i0 > 0:
                    left -= (left - float(s["t_s"][i0 - 1])) / 2
                col = _STATE_COLORS.get(str(st), "#dddddd")
                axes[2].broken_barh(
                    [(left, max(right - left, 1e-6))], (y0, h),
                    transform=trans, color=col, alpha=0.85, zorder=0.9)
                name = str(st)
                if name not in seen:
                    seen.append(name)
                    state_patches.append(Patch(color=col, label=name))

    axes[0].set_ylabel("x (px)")
    axes[1].set_ylabel("y (px)")
    axes[2].set_ylabel("speed (body-scale/s)")
    axes[2].set_xlabel("t (s)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    if state_patches:
        handles, _labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles=handles + state_patches, fontsize=7,
                       loc="best")
    title = (
        f"{win['kind']} @ {win['center_pts_us']}us  backend={win['backend']} "
        f"method={win['method']}"
    )
    if win.get("trace", {}).get("exact"):
        title += "  [exact trace]"
    if win.get("gt"):
        g = win["gt"]
        title += f"  GT({g['foot']}) pref={g['preferred_pts_us']}us"
    if pred_t is not None and win.get("gt"):
        diff = round((pred_t - win["gt"]["preferred_pts_us"] / 1e6) * 30)
        title += f"  err={diff:+d}f@30fps"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


def plot_summary(trace: PoseTrace, gt_records: list[dict],
                 out_path: str | Path, cycles: int = 4) -> str:
    """Stack >=3 consecutive gait cycles of foot-centre speed + GT marks."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not gt_records:
        raise ValueError("summary plot needs GT records")
    frames = trace.frames
    times = np.array([fr.t_s for fr in frames])
    t0 = gt_records[0]["preferred_s"]
    span = (gt_records[min(cycles, len(gt_records) - 1)]["preferred_s"]
            - t0) + 1.0
    mask = (times >= max(0.0, t0 - 0.5)) & (times <= t0 + span)
    idx = np.flatnonzero(mask)
    if idx.size < 4:
        raise ValueError("not enough frames inside the summary span")

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"left": "tab:green", "right": "tab:orange"}
    for side in SIDES:
        speed = []
        for i in idx:
            fr = frames[i]
            obs = (fr.feet or {}).get(side)
            marks = ([p for p in (obs.heel, obs.ankle, obs.toe)
                      if p is not None and p.visibility >= 0.5]
                     if obs else [])
            speed.append(
                math.hypot(
                    sum(p.x for p in marks) / len(marks) - _px(frames, i - 1, side, "x"),
                    sum(p.y for p in marks) / len(marks) - _py(frames, i - 1, side),
                ) / max(times[i] - times[i - 1], 1e-3)
                if len(marks) >= 2 and i > 0 else np.nan
            )
        ax.plot(times[idx], speed, color=colors[side], label=f"{side} speed")
    for j, r in enumerate(gt_records[:cycles + 1]):
        ax.axvline(r["preferred_s"], color="gold",
                   lw=2 if r["foot"] == "left" else 1.2,
                   ls="-" if r["foot"] == "left" else "--")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("foot-centre speed (px/s, unnormalized)")
    ax.set_title(f"{cycles} consecutive cycles from first GT contact")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


def _px(frames, i, side, _axis):
    if i < 0:
        return np.nan
    obs = (frames[i].feet or {}).get(side)
    marks = ([p for p in (obs.heel, obs.ankle, obs.toe)
              if p is not None and p.visibility >= 0.5] if obs else [])
    if len(marks) < 2:
        return np.nan
    return sum(p.x for p in marks) / len(marks)


def _py(frames, i, side):
    if i < 0:
        return np.nan
    obs = (frames[i].feet or {}).get(side)
    marks = ([p for p in (obs.heel, obs.ankle, obs.toe)
              if p is not None and p.visibility >= 0.5] if obs else [])
    if len(marks) < 2:
        return np.nan
    return sum(p.y for p in marks) / len(marks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_forensics(
    run_dir: str | Path,
    gt_path: str | Path | None,
    out_dir: str | Path | None = None,
    *,
    event_id: str | None = None,
    half_s: float = 0.65,
) -> Path:
    """Emit forensics/<id>.json + .png (+ summary.png + index.json)."""

    trace, events_doc, _diag = load_run_trace(run_dir)
    events = events_doc.get("events") or []
    method = events_doc.get("contact_method", "hybrid")
    gt_records = load_gt_any(str(gt_path)) if gt_path else None
    out_dir = Path(out_dir) if out_dir else Path(run_dir) / "forensics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persisted detector internals (exact) or None -- never reconstructed.
    trace_rows, trace_header = load_contact_trace(run_dir)
    tindex = ContactTraceIndex.from_rows(trace_rows, trace_header)

    # ONE shared diagnostic association over ALL events x GT; every window
    # receives its pair (or its unmatched-GT record) from this pass.
    fps = float(trace.meta.fps_stream or 0.0)
    assoc = (
        associate_diagnostic(
            events, gt_records, horizon_s=GT_MATCH_HORIZON_S, fps=fps or None)
        if gt_records else None
    )

    windows = select_windows(events, gt_records, event_id, assoc_result=assoc)
    index = []
    for kind, center_us, label in windows:
        gt_rec, pair = None, None
        if assoc is not None:
            if kind == "predicted":
                pred_idx = next(
                    (i for i, e in enumerate(events)
                     if int(e.get("source_pts_us", -1)) == int(center_us)),
                    None,
                )
                if pred_idx is not None:
                    pair = assoc.partner_of_pred(pred_idx)
                    if pair is not None:
                        gt_rec = gt_records[pair["gt_idx"]]
            elif kind == "missed_gt":
                u = next(
                    (u for u in assoc.unmatched_gts
                     if s_to_pts_us(u["preferred_s"]) == int(center_us)),
                    None,
                )
                if u is not None:
                    gt_rec = gt_records[u["gt_idx"]]
        win = build_window(
            trace, events, gt_records, center_us, kind=kind,
            half_s=half_s, method=method, trace_index=tindex,
            gt_record=gt_rec, assoc_pair=pair,
        )
        safe = str(label).replace("/", "_")
        json_path = out_dir / f"{safe}.json"
        json_path.write_text(json.dumps(win, indent=2) + "\n", "utf-8")
        png_path = plot_window(win, out_dir / f"{safe}.png")
        entry = {
            "label": label, "kind": kind, "center_pts_us": center_us,
            "json": json_path.name, "png": Path(png_path).name,
            "gt": win.get("gt"),
        }
        g = win.get("gt") or {}
        if g.get("gt_id"):
            entry["gt_id"] = g["gt_id"]
        if g.get("assoc_err_frames") is not None:
            entry["assoc_err_frames"] = g["assoc_err_frames"]
        index.append(entry)
    if gt_records and event_id is None:
        try:
            summary = plot_summary(trace, gt_records, out_dir / "summary.png")
            index.append({"label": "summary", "png": Path(summary).name})
        except ValueError as exc:
            index.append({"label": "summary", "error": str(exc)})
    (out_dir / "index.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION,
                    "run_dir": str(run_dir),
                    "windows": index}, indent=2) + "\n", "utf-8"
    )
    return out_dir
