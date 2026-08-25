"""Per-event forensic windows over cached pose traces (Phase-2 T2).

For every predicted event and every missed GT contact, emit one machine-
readable JSON window (schema docs/auto-sfx-phase2-truth-repair.md §5.2) and
one PNG plot, plus one summary PNG stacking consecutive gait cycles. Consumes
ONLY run artifacts (pose.jsonl, events.json) and the GT file — never
re-runs pose inference.

Honesty notes, by design:
- ``conf_smoothed`` mirrors ``conf_raw``: this module applies no smoothing;
  any temporal filtering lives in the pose backend (T8 measures it).
- ``state`` is derived from predicted-event windows (PRED_WINDOW / "-"),
  NOT from the detector's internal machine, whose per-frame states are not
  persisted; ``transitions`` is therefore empty in v1 of this tool.
- ``thresholds`` carries the method's DEFAULT config values; per-run
  ``--config`` overrides are not persisted in run artifacts today.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

try:  # package-relative when imported as auto_sfx.forensics
    from .cache import TRACE_FORMAT
    from .contacts import _env_heights, default_config
    from .metrics import load_gt_any
    from .types import SIDES, PoseTrace, VideoMeta, s_to_pts_us
except ImportError:  # pragma: no cover - direct module import
    from cache import TRACE_FORMAT  # type: ignore[no-redef]
    from contacts import _env_heights, default_config  # type: ignore
    from metrics import load_gt_any  # type: ignore[no-redef]
    from types import (  # type: ignore
        SIDES, PoseTrace, VideoMeta, s_to_pts_us,
    )

SCHEMA_VERSION = 1
GT_MATCH_HORIZON_S = 0.5
FORENSICS_SCHEMA_NOTE = (
    "state derived from predicted-event windows; transitions not persisted "
    "by the detector (see module docstring)"
)


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
) -> dict:
    """One §5.2 forensic window around ``center_pts_us``."""

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
        "thresholds": {
            "speed_drop_frac": default_config(method)["speed_drop_frac"],
            "min_peak_floor_bs_s": 0.4,
            "note": "defaults; per-run --config overrides are not persisted",
        },
    }
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
        series["state"][side] = [
            "PRED_WINDOW" if any(
                e.get("foot") == side
                and float(e.get("earliest_s", e["time_s"]))
                <= times[i] <= float(e.get("latest_s", e["time_s"]))
                for e in events
            ) else "-"
            for i in idx
        ]

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

    # support band: envelope height / scale per side, max over sides
    band = {}
    n_all = len(frames)
    fake_fd = _mini_fd(trace)
    for side in SIDES:
        env = _env_heights(fake_fd, np.arange(n_all), side)
        band[side] = env
    series["support_band_score"] = [
        round(float(max(
            (band[s][i] / med_scale if np.isfinite(band[s][i]) else np.nan)
            for s in SIDES
        )), 4) if any(np.isfinite(band[s][i]) for s in SIDES) else None
        for i in idx
    ]

    # GT overlay: nearest same-foot record within horizon
    gt_overlay = None
    center_side = None
    if kind == "predicted":
        matching = [e for e in events
                    if int(e.get("source_pts_us", -1)) == int(center_pts_us)]
        if matching:
            center_side = matching[0].get("foot")
    if gt_records:
        best = None
        for r in gt_records:
            if center_side and r["foot"] not in (center_side, "uncertain"):
                continue
            d = abs(r["preferred_s"] - t_center)
            if d <= GT_MATCH_HORIZON_S and (best is None or d < best[0]):
                best = (d, r)
        if best is not None:
            r = best[1]
            gt_overlay = {
                "earliest_pts_us": s_to_pts_us(r["earliest_s"]),
                "preferred_pts_us": s_to_pts_us(r["preferred_s"]),
                "latest_pts_us": s_to_pts_us(r["latest_s"]),
                "foot": r["foot"],
            }

    return _nan_to_none({
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "center_pts_us": int(center_pts_us),
        "window_s": [round(lo, 4), round(hi, 4)],
        "backend": trace.backend,
        "method": method,
        "series": series,
        "transitions": [],
        "notes": [FORENSICS_SCHEMA_NOTE],
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
                   event_id: str | None, horizon_s: float = GT_MATCH_HORIZON_S):
    """Which (kind, center_pts_us, label) windows to emit.

    Every predicted event gets a window; every GT contact with no same-foot
    prediction within ``horizon_s`` additionally gets a ``missed_gt`` window.
    """

    out = []
    preds = [e for e in events if event_id in (None, e.get("id"))]
    if event_id is not None and not preds:
        raise ValueError(f"event id {event_id!r} not found in events.json")
    for e in preds:
        out.append(("predicted", int(e["source_pts_us"]), e.get("id", "?")))
    if gt_records and event_id is None:
        for r in gt_records:
            hit = any(
                (e.get("foot") == r["foot"] or r["foot"] == "uncertain")
                and abs(float(e["time_s"]) - r["preferred_s"]) <= horizon_s
                for e in events
            )
            if not hit:
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

    s = win["series"]
    t = np.array([x if x is not None else np.nan for x in s["t_s"]])

    def arr(pairs):
        return np.array([
            [np.nan, np.nan] if p is None else p for p in pairs
        ], dtype=float)

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
    axes[2].axhline(0.35 * 0.4, color="gray", ls=":", lw=1,
                    label="thr floor (0.35*0.4)")

    if win.get("gt"):
        g = win["gt"]
        for ax in axes:
            ax.axvspan(g["earliest_pts_us"] / 1e6, g["latest_pts_us"] / 1e6,
                       color="yellow", alpha=0.18)
            ax.axvline(g["preferred_pts_us"] / 1e6, color="gold", lw=2)
    for e_kind, style in (("predicted", "r--"),):
        pass
    pred_t = None
    if win["kind"] == "predicted":
        pred_t = win["center_pts_us"] / 1e6
        for ax in axes:
            ax.axvline(pred_t, color="red", lw=1.5)

    axes[0].set_ylabel("x (px)")
    axes[1].set_ylabel("y (px)")
    axes[2].set_ylabel("speed (body-scale/s)")
    axes[2].set_xlabel("t (s)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    title = (
        f"{win['kind']} @ {win['center_pts_us']}us  backend={win['backend']} "
        f"method={win['method']}"
    )
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

    windows = select_windows(events, gt_records, event_id)
    index = []
    for kind, center_us, label in windows:
        win = build_window(
            trace, events, gt_records, center_us, kind=kind,
            half_s=half_s, method=method,
        )
        safe = str(label).replace("/", "_")
        json_path = out_dir / f"{safe}.json"
        json_path.write_text(json.dumps(win, indent=2) + "\n", "utf-8")
        png_path = plot_window(win, out_dir / f"{safe}.png")
        index.append({
            "label": label, "kind": kind, "center_pts_us": center_us,
            "json": json_path.name, "png": Path(png_path).name,
            "gt": win.get("gt"),
        })
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
