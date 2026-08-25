"""T3 -- honest per-event error classification (Phase 2).

Produces the directive's table (docs/auto-sfx-phase2-truth-repair.md
§4-T3, extended per CTO update #2)::

    event | GT | prediction | error frames/ms | error class |
    backend | method | viewpoint | visibility | notes

CLASSIFICATION POLICY -- mechanically defensible only.  Classes are
assigned ONLY where arithmetic on times, sides and persisted run
artifacts forces the conclusion:

``NONE``
    Diagnostic partner exists and the prediction falls inside the GT's
    defensible [earliest, latest] interval (sides agree or are not
    judgeable).  Not an error.
``WRONG_SIDE``
    The impact moment matched an opposite-foot mark INSIDE its window:
    good timing, wrong anatomical label (T7: not a timing failure).
``EARLY_PHASE / LATE_STANCE / TOE_OFF_CONFUSED_AS_CONTACT``
    Timing is off.  When the diagnostic partner is same-foot, the class
    comes from the documented midpoint phase map of the own-foot
    timeline; when the partner is opposite-foot, the prediction is
    re-anchored to its OWN foot's nearest mark and judged by signed
    frame error + stride bands (a badly timed own-foot attempt is a
    phase error, not an attribution error).
``DUPLICATE_CONTACT``
    Unpaired prediction whose nearest GT contact is already claimed by
    another prediction inside the diagnostic horizon.
``SIMULTANEOUS_LR``
    Two predictions with known opposite feet within one displayed frame.
``MISSED_CONTACT``
    GT contact left unpaired by the diagnostic association.
``POSE_LOSS``
    Missed contact whose whole window lies inside a recorded POSE_LOST
    diagnostics interval (evidence: run diagnostics.json).
``UNKNOWN``
    Everything else -- always with the evidence that failed to force a
    class.

The judgement-requiring classes (POSE_JITTER, LANDMARK_SWAP,
CAMERA_MOTION, VIEW_GEOMETRY, CROPPING) are NEVER auto-assigned.
"""

from __future__ import annotations

import csv
import json

try:  # package-relative when imported as auto_sfx.classify
    from .association import (
        DIAGNOSTIC_HORIZON_S,
        associate_diagnostic,
        norm_side,
        pred_time,
    )
    from .metrics import (
        _as_predictions,
        _tolerance_s,
        classify_phase_for_foot,
    )
except ImportError:  # pragma: no cover - direct module import
    from association import (  # type: ignore[no-redef]
        DIAGNOSTIC_HORIZON_S,
        associate_diagnostic,
        norm_side,
        pred_time,
    )
    from metrics import (  # type: ignore[no-redef]
        _as_predictions,
        _tolerance_s,
        classify_phase_for_foot,
    )

SCHEMA_VERSION = 1

# Classes this module may assign autonomously.
MECHANICAL_CLASSES = frozenset({
    "NONE", "WRONG_SIDE", "DUPLICATE_CONTACT", "SIMULTANEOUS_LR",
    "MISSED_CONTACT", "POSE_LOSS", "EARLY_PHASE", "LATE_STANCE",
    "TOE_OFF_CONFUSED_AS_CONTACT", "UNKNOWN",
})

# Never auto-assigned; human-judgement classes from the frozen vocabulary.
HUMAN_ONLY_CLASSES = frozenset({
    "POSE_JITTER", "LANDMARK_SWAP", "CAMERA_MOTION",
    "VIEW_GEOMETRY", "CROPPING",
})

CSV_COLUMNS = [
    "event_id", "gt_id", "prediction_time_s", "gt_preferred_s",
    "error_frames", "error_ms", "error_class",
    "backend", "method", "viewpoint", "visibility", "notes",
]


def load_pose_lost_intervals(diagnostics_doc: dict | None) -> list[tuple]:
    """[(t0_s, t1_s)] POSE_LOST intervals from a diagnostics document."""

    out = []
    for d in (diagnostics_doc or {}).get("diagnostics") or []:
        if d.get("code") == "POSE_LOST":
            t0, t1 = d.get("t0_s"), d.get("t1_s")
            if t0 is not None and t1 is not None:
                out.append((float(t0), float(t1)))
    return out


def _in_pose_lost(t0: float, t1: float, lost: list[tuple]) -> bool:
    return any(lo <= t0 and t1 <= hi for lo, hi in lost)


def _phase_class_from_map(phase_cls: str, err_f):
    """Map a same-foot midpoint-phase word onto the vocabulary."""

    if phase_cls == "early_swing":
        return "EARLY_PHASE"
    if phase_cls in ("mid_stance", "terminal_stance"):
        return "LATE_STANCE"
    if phase_cls == "toe_off_neighborhood":
        return "TOE_OFF_CONFUSED_AS_CONTACT"
    if phase_cls == "IC_neighborhood":
        return "LATE_STANCE" if (err_f or 0) > 0 else "EARLY_PHASE"
    return None  # UNKNOWN_* -- not mechanically decidable


def _own_foot_phase_class(t_s, foot, gt_records, fps, median_stride=0.6):
    """Classify against the OWN foot's nearest mark.

    Used when side-agnostic association latched onto the opposite foot:
    anchor = temporally nearest SAME-foot GT contact; class follows from
    the signed frame error plus the documented stride bands (u measured
    from that anchor over the median same-foot interval).  Returns
    ``(class_or_None, detail_note)``.
    """

    candidates = [r for r in gt_records if r.get("foot") == foot]
    if not candidates:
        return None, "no same-side marks", None
    anchor_rec = min(candidates, key=lambda r: abs(r["preferred_s"] - t_s))
    anchor = anchor_rec["preferred_s"]
    own_err_s = t_s - anchor
    err_f = int(round(own_err_s * float(fps))) if fps else None
    ordered = sorted(r["preferred_s"] for r in candidates)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    stride = (sorted(gaps)[len(gaps) // 2] if gaps else median_stride)
    u = own_err_s / max(stride, 1e-9)
    detail = f"own-foot {anchor:.3f}s"
    if anchor_rec["earliest_s"] <= t_s <= anchor_rec["latest_s"]:
        return "NONE", detail + " (inside own-foot window)", own_err_s
    if err_f is not None and err_f < 0:
        # Fired before its own foot's contact -- early by that much.
        return "EARLY_PHASE", f"{detail} {err_f:+d}f", own_err_s
    if u < 0.80:
        return "LATE_STANCE", f"{detail} u={u:.2f}", own_err_s
    if u < 0.92:
        return "TOE_OFF_CONFUSED_AS_CONTACT", f"{detail} u={u:.2f}", own_err_s
    return "EARLY_PHASE", f"{detail} u={u:.2f} (next cycle)", own_err_s


def classify_run(events_doc: dict, gt_records: list[dict],
                 diagnostics_doc: dict | None = None,
                 diagnostic_horizon_s: float = DIAGNOSTIC_HORIZON_S,
                 tol_ms: float | None = None) -> dict:
    """Build the full classification report (see module docstring)."""

    video = events_doc.get("video") or {}
    fps = video.get("fps_stream") or None
    backend = events_doc.get("backend")
    method = events_doc.get("contact_method")

    events = events_doc.get("events") or []
    preds = _as_predictions(events)

    diag = associate_diagnostic(
        events, gt_records, horizon_s=diagnostic_horizon_s, fps=fps
    )
    tol_s = _tolerance_s(fps, tol_ms)

    # Simultaneous opposite-foot prediction pairs (one displayed frame).
    simultan: dict[int, tuple[int, float]] = {}
    ordered = sorted(range(len(preds)), key=lambda i: (pred_time(preds[i]), i))
    for a, b in zip(ordered, ordered[1:]):
        fa = norm_side(preds[a].get("foot"))
        fb = norm_side(preds[b].get("foot"))
        dt = abs(pred_time(preds[b]) - pred_time(preds[a]))
        if fa and fb and fa != fb and fps and dt * fps <= 1.0:
            simultan[a] = (b, dt)
            simultan[b] = (a, dt)

    partner_of_pred = {p["pred_idx"]: p for p in diag.pairs}
    rows: list[dict] = []
    claimed_gts: set[int] = set()

    # -- predictions with partners -------------------------------------------
    for pi, pred in enumerate(preds):
        pair = partner_of_pred.get(pi)
        notes: list[str] = []
        err_override = None
        if pi in simultan:
            other, dt = simultan[pi]
            notes.append(
                f"simultaneous L/R with event {preds[other].get('id')} "
                f"(dt={dt * 1000:.0f} ms)")
        if pair is None:
            continue  # handled in the unpaired pass below
        gt_rec = gt_records[pair["gt_idx"]]
        claimed_gts.add(pair["gt_idx"])
        err_f = pair["err_frames"]

        if pi in simultan:
            cls = "SIMULTANEOUS_LR"
        elif pair["side_ok"] is False:
            if pair["inside_gt_window"]:
                cls = "WRONG_SIDE"
                notes.append(
                    f"side-agnostic timing ok ({err_f:+d}f); "
                    f"attribution wrong")
            else:
                mapped, detail, own_err_s = _own_foot_phase_class(
                    pred_time(pred),
                    norm_side(pred.get("foot")) or "uncertain",
                    gt_records, fps)
                if mapped is None:
                    cls = "WRONG_SIDE"
                    notes.append(
                        f"paired opposite-foot {gt_rec['gt_id']} "
                        f"({err_f:+d}f); {detail}")
                elif mapped == "NONE":
                    cls = "NONE"
                    err_override = own_err_s
                    notes.append(f"own-foot window hit: {detail}")
                else:
                    cls = mapped
                    err_override = own_err_s
                    notes.append(
                        f"side-flipped pairing to {gt_rec['gt_id']}; "
                        f"classified vs own-foot timeline: {detail}")
        elif pair["inside_gt_window"]:
            cls = "NONE"
        else:
            phase_cls, n_same = classify_phase_for_foot(
                pred_time(pred), norm_side(pred.get("foot")) or "uncertain",
                gt_records)
            mapped = _phase_class_from_map(phase_cls, err_f)
            if mapped is None:
                cls = "UNKNOWN"
                notes.append(f"phase_map={phase_cls} same_foot_gt={n_same}")
            else:
                cls = mapped

        rows.append(_row(pred, gt_rec, pair, cls, backend, method, notes,
                         err_override=err_override, fps=fps))

    # -- unpaired predictions --------------------------------------------------
    for u in diag.unmatched_preds:
        pi = u["pred_idx"]
        pred = preds[pi]
        notes: list[str] = []
        if pi in simultan:
            other, dt_s = simultan[pi]
            notes.append(
                f"simultaneous L/R with event {preds[other].get('id')} "
                f"(dt={dt_s * 1000:.0f} ms)")
        best_d, best_i = None, None
        for gi, r in enumerate(gt_records):
            d = abs(r["preferred_s"] - pred_time(pred))
            if best_d is None or d < best_d:
                best_d, best_i = d, gi
        if pi in simultan:
            cls = "SIMULTANEOUS_LR"
        elif (best_i is not None and best_i in claimed_gts
                and best_d <= diagnostic_horizon_s):
            cls = "DUPLICATE_CONTACT"
            winner = next(
                p["pred_id"] for p in diag.pairs if p["gt_idx"] == best_i)
            notes.append(
                f"nearest gt {gt_records[best_i]['gt_id']} claimed by "
                f"{winner} ({best_d * 1000:.0f} ms away)")
        else:
            cls = "UNKNOWN"
            if best_i is not None:
                notes.append(
                    f"no GT within {diagnostic_horizon_s:.2f}s; nearest "
                    f"{gt_records[best_i]['gt_id']} at {best_d:.3f}s")
            else:
                notes.append("no GT records at all")
        rows.append(_row(pred, None, None, cls, backend, method, notes))

    # -- unpaired GT contacts ---------------------------------------------------
    lost = load_pose_lost_intervals(diagnostics_doc)
    for u in diag.unmatched_gts:
        gt_rec = gt_records[u["gt_idx"]]
        w0, w1 = gt_rec["earliest_s"], gt_rec["latest_s"]
        notes = []
        if lost and _in_pose_lost(w0, w1, lost):
            cls = "POSE_LOSS"
            notes.append("window fully inside a POSE_LOST diagnostics span")
        else:
            cls = "MISSED_CONTACT"
            if lost and any(max(lo, w0) <= min(hi, w1) for lo, hi in lost):
                notes.append("partial overlap with POSE_LOST span")
        rows.append({
            "event_id": None,
            "gt_id": gt_rec["gt_id"],
            "prediction_time_s": None,
            "gt_preferred_s": round(gt_rec["preferred_s"], 6),
            "error_frames": None,
            "error_ms": None,
            "error_class": cls,
            "backend": backend,
            "method": method,
            "viewpoint": gt_rec.get("viewpoint"),
            "visibility": gt_rec.get("visibility"),
            "notes": "; ".join(notes),
        })

    rows.sort(key=lambda r: (
        r["prediction_time_s"] if r["prediction_time_s"] is not None
        else r["gt_preferred_s"],
        r["event_id"] or "",
    ))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["error_class"]] = counts.get(r["error_class"], 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "identification": {
            "backend": backend,
            "method": method,
            "displayed_fps": fps,
            "ground_truth_n": len(gt_records),
            "predictions_n": len(preds),
            "diagnostic_horizon_s": diagnostic_horizon_s,
            "strict_tolerance_s": round(tol_s, 6),
        },
        "policy": {
            "auto_assigned": sorted(MECHANICAL_CLASSES),
            "human_only": sorted(HUMAN_ONLY_CLASSES),
        },
        "class_counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def _row(pred, gt_rec, pair, cls, backend, method, notes,
         err_override: float | None = None, fps=None) -> dict:
    if err_override is not None:
        err_frames = (
            int(round(err_override * float(fps))) if fps else None)
        err_ms = round(err_override * 1000.0, 1)
    elif pair:
        err_frames, err_ms = pair["err_frames"], round(pair["err_s"] * 1000.0, 1)
    else:
        err_frames, err_ms = None, None
    return {
        "event_id": pred.get("id"),
        "gt_id": (gt_rec or {}).get("gt_id"),
        "prediction_time_s": round(pred_time(pred), 6),
        "gt_preferred_s": (
            round(gt_rec["preferred_s"], 6) if gt_rec else None),
        "error_frames": err_frames,
        "error_ms": err_ms,
        "error_class": cls,
        "backend": backend,
        "method": method,
        "viewpoint": (gt_rec or {}).get("viewpoint"),
        "visibility": (gt_rec or {}).get("visibility"),
        "notes": "; ".join(notes),
    }


def write_outputs(report: dict, json_path, csv_path=None) -> None:
    """classify.json (+ optional classify.csv)."""

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for r in report["rows"]:
                writer.writerow({k: r.get(k) for k in CSV_COLUMNS})
