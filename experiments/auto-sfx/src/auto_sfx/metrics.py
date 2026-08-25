"""Evaluation metrics: detector events vs manual foot-contact annotations.

Annotation format (JSON list, one object per manually marked contact)::

    [{"time_s": 1.432, "foot": "left"}, {"time_s": 2.101, "foot": "right"}]

Only ``time_s`` (seconds on the source presentation clock) and ``foot``
(``"left" | "right"``) participate in evaluation; extra keys are tolerated
and ignored. Predictions are ``events.json`` entries (dicts) or
``types.ContactEvent`` objects; matching likewise uses only their
``time_s`` / ``foot`` / ``id``.

Matching rule (docs/auto-sfx-research.md §12.7): greedy nearest-first
one-to-one association over all candidate pairs with ``|dt| <= tol``,
where ``tol`` defaults to ``max(66 ms, 1.5 / displayed_fps)`` and may be
overridden with ``tol_ms`` -- implemented once in ``association`` and
shared with preview, forensics and T3. A prediction pairing with an
already-matched annotation within tolerance counts as a *duplicate*,
not a fresh false positive. Foot side never blocks a match; side
mismatches are reported separately as ``wrong_side``.
"""

from __future__ import annotations

import json
import math

try:  # package-relative when imported as auto_sfx.metrics
    from .association import (
        DIAGNOSTIC_HORIZON_S,
        associate_diagnostic,
        associate_strict,
    )
    from .types import SIDES
except ImportError:  # pragma: no cover - direct module import
    from association import (  # type: ignore[no-redef]
        DIAGNOSTIC_HORIZON_S,
        associate_diagnostic,
        associate_strict,
    )
    from types import SIDES  # type: ignore[no-redef]

_VALID_SIDES = frozenset(SIDES)

# Report floats are rounded to this many decimal places.
_ROUND = 6


# ---------------------------------------------------------------------------
# Access helpers
# ---------------------------------------------------------------------------


def _norm_side(value):
    """Normalise a side label to 'left'/'right', or None when absent/invalid."""
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in _VALID_SIDES:
            return stripped
    return None


def _get_field(record, name):
    """Read ``name`` from a dict or attribute-style event record."""
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _percentile(sorted_values, q):
    """Linear-interpolation percentile of an already-sorted sequence."""
    if not sorted_values:
        return None
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


def _parse_json_array(text):
    """Parse a top-level JSON array, returning [([int_line, value], ...)].

    Incremental scan so every element keeps the 1-based line it starts on;
    plain ``json.loads`` would collapse that information.
    """
    decoder = json.JSONDecoder()
    n = len(text)
    items = []

    def skip_ws(i):
        while i < n and text[i] in " \t\r\n":
            i += 1
        return i

    def line_at(i):
        return text.count("\n", 0, i) + 1

    i = skip_ws(0)
    if i >= n or text[i] != "[":
        raise ValueError(f"expected a JSON array of annotations at line {line_at(i)}")
    i = skip_ws(i + 1)
    if i < n and text[i] == "]":
        i = skip_ws(i + 1)
        if i < n:
            raise ValueError(f"unexpected trailing content at line {line_at(i)}")
        return items

    while True:
        if i >= n:
            raise ValueError("unterminated annotation array (missing ']')")
        try:
            value, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON near line {line_at(i)}: {exc.msg}") from None
        items.append((line_at(i), value))
        i = skip_ws(end)
        if i < n and text[i] == ",":
            i = skip_ws(i + 1)
            continue
        if i < n and text[i] == "]":
            i = skip_ws(i + 1)
            if i < n:
                raise ValueError(f"unexpected trailing content at line {line_at(i)}")
            return items
        raise ValueError(f"expected ',' or ']' at line {line_at(i)}")


def _normalise_annotation(obj):
    """Validate one annotation record; return (record, problem_message)."""
    if not isinstance(obj, dict):
        return None, "record must be a JSON object"

    out = dict(obj)

    if "time_s" not in obj:
        return None, "'time_s' missing"
    raw_t = obj["time_s"]
    if isinstance(raw_t, bool):
        return None, f"'time_s' not numeric: {raw_t!r}"
    try:
        t_s = float(raw_t)
    except (TypeError, ValueError):
        return None, f"'time_s' not numeric: {raw_t!r}"
    if not math.isfinite(t_s):
        return None, f"'time_s' not finite: {raw_t!r}"
    out["time_s"] = t_s

    foot = _norm_side(obj.get("foot")) or (
        "uncertain" if isinstance(obj.get("foot"), str)
        and obj.get("foot").strip().lower() == "uncertain" else None
    )
    if foot is None:
        return None, (
            f"'foot' must be 'left', 'right' or 'uncertain': "
            f"{obj.get('foot')!r}"
        )
    out["foot"] = foot

    return out, None


def load_annotations(path):
    """Load and validate a manual annotation file.

    Returns a list of ``{"time_s": float, "foot": "left"|"right", ...}``
    records (extra source keys preserved). Raises ``ValueError`` naming the
    offending 1-based file line(s) for structural or per-record problems.
    """
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()

    try:
        raw_items = _parse_json_array(text)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None

    records = []
    problems = []
    for lineno, obj in raw_items:
        record, problem = _normalise_annotation(obj)
        if problem is None:
            records.append(record)
        else:
            problems.append(f"line {lineno}: {problem}")

    if problems:
        listing = "\n  ".join(problems)
        raise ValueError(f"{path}: invalid annotation record(s):\n  {listing}")
    return records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _tolerance_s(displayed_fps, tol_ms):
    if tol_ms is not None:
        return float(tol_ms) / 1000.0
    if displayed_fps is not None and displayed_fps > 0:
        return max(0.066, 1.5 / float(displayed_fps))
    return 0.066


def _as_predictions(events):
    """Coerce events.json entries / ContactEvent objects to uniform dicts."""
    preds = []
    for idx, event in enumerate(events):
        raw_id = _get_field(event, "id")
        preds.append(
            {
                "id": str(raw_id) if raw_id is not None else f"event_{idx}",
                "time_s": float(_get_field(event, "time_s")),
                "foot": _norm_side(_get_field(event, "foot")),
            }
        )
    return preds


def _as_annotations(records):
    anns = []
    for idx, record in enumerate(records):
        ann, problem = _normalise_annotation(record)
        if problem is not None:
            raise ValueError(f"invalid annotation at index {idx}: {problem}")
        anns.append(ann)
    return anns


def evaluate(events, annotations, displayed_fps, tol_ms=None):
    """Compare predicted contact events against manual annotations.

    Greedy one-to-one matching: all (prediction, annotation) pairs within
    ``tol`` seconds are visited in increasing ``|dt|`` order; a pair is
    accepted when both sides are still free. A prediction whose best
    remaining pairing lands on an already-matched annotation is counted as a
    duplicate rather than a plain false positive.

    ``median_err_s`` / ``p90_err_s`` are computed over absolute matched
    errors (signed errors remain available per-pair in ``matched_pairs``).
    ``within_1_frame_frac`` is the fraction of matched pairs with
    ``|err| * displayed_fps <= 1``. ``fp_per_minute`` divides false positives
    by the annotation timespan ``(max - min) / 60``; it is ``None`` when the
    span cannot be established. All floats in the report are rounded to
    6 decimal places; empty inputs yield zero counts and ``None`` ratios.
    """
    tol = _tolerance_s(displayed_fps, tol_ms)
    preds = _as_predictions(events)
    anns = _as_annotations(annotations)

    # Shared one-to-one matcher (association.associate_strict) -- the
    # identical greedy nearest-first algorithm, also used by preview,
    # forensics and T3 so every artifact links consistently.
    strict = associate_strict(preds, anns, tol, fps=displayed_fps)

    matched_pairs = []
    wrong_side_pairs = []
    abs_errors = []
    for pair in strict.pairs:
        pi, ai = pair["pred_idx"], pair["gt_idx"]
        err = pair["err_s"]
        foot_a = anns[ai]["foot"]
        foot_p = preds[pi]["foot"]
        foot_ok = foot_a is not None and foot_p is not None and foot_a == foot_p
        entry = {
            "pred_id": preds[pi]["id"],
            "ann_time": round(anns[ai]["time_s"], _ROUND),
            "pred_time": round(preds[pi]["time_s"], _ROUND),
            "err_s": round(err, _ROUND),
            "foot_ok": foot_ok,
        }
        matched_pairs.append(entry)
        abs_errors.append(abs(err))
        if foot_a is None or foot_p is None or foot_ok:
            continue
        wrong_side_pairs.append(
            {
                "pred_id": preds[pi]["id"],
                "ann_time": entry["ann_time"],
                "pred_time": entry["pred_time"],
                "err_s": entry["err_s"],
                "ann_foot": foot_a,
                "pred_foot": foot_p,
            }
        )

    matched = len(matched_pairs)
    n_true = len(anns)
    n_pred = len(preds)
    duplicates = sum(
        1 for u in strict.unmatched_preds if u["reason"] == "dup"
    )
    false_positives = n_pred - matched - duplicates
    misses = n_true - matched

    median_err = _percentile(sorted(abs_errors), 0.5)
    p90_err = _percentile(sorted(abs_errors), 0.9)

    if matched and displayed_fps is not None and displayed_fps > 0:
        within_1_frame = sum(1 for e in abs_errors if e * float(displayed_fps) <= 1.0)
        within_1_frame_frac = round(within_1_frame / matched, _ROUND)
    else:
        within_1_frame_frac = None

    ann_times = [a["time_s"] for a in anns]
    span = max(ann_times) - min(ann_times) if ann_times else 0.0
    if span > 0:
        fp_per_minute = round(false_positives / (span / 60.0), _ROUND)
    else:
        fp_per_minute = None

    unmatched_preds = [
        {
            "id": p["pred_id"],
            "time_s": round(p["time_s"], _ROUND),
            "foot": p["foot"],
        }
        for p in strict.unmatched_preds
    ]
    unmatched_anns = [
        {
            "time_s": round(a["preferred_s"], _ROUND),
            "foot": a["foot"],
        }
        for a in strict.unmatched_gts
    ]

    return {
        "tol_s": round(tol, _ROUND),
        "n_true": n_true,
        "n_pred": n_pred,
        "matched": matched,
        "duplicates": duplicates,
        "false_positives": false_positives,
        "misses": misses,
        "wrong_side": len(wrong_side_pairs),
        "wrong_side_pairs": wrong_side_pairs,
        "median_err_s": None if median_err is None else round(median_err, _ROUND),
        "p90_err_s": None if p90_err is None else round(p90_err, _ROUND),
        "within_1_frame_frac": within_1_frame_frac,
        "fp_per_minute": fp_per_minute,
        "matched_pairs": matched_pairs,
        "unmatched_preds": unmatched_preds,
        "unmatched_anns": unmatched_anns,
    }



# ---------------------------------------------------------------------------
# Metrics v2 (Phase-2 T14; docs/auto-sfx-phase2-truth-repair.md §5.4)
# Repaired per CTO update #2: one shared association (strict for
# existence, diagnostic for timing/phase), GT windows honoured,
# uncertain-foot safe, per-foot phase timelines, view-stratified report.
# ---------------------------------------------------------------------------

# Report sections that identify what produced the numbers (directive:
# "each report must identify backend, method, and viewpoint").


def load_gt_any(path):
    """Load ground truth as uniform records, accepting v2 and legacy v1.

    v2 (``auto_sfx.gt`` schema) is canonical: preferred/earliest/latest
    windows AND annotation metadata (visibility, viewpoint, notes,
    source PTS) survive -- downstream reports stratify by viewpoint and
    quote visibility, so dropping them silently is a defect. Legacy v1
    arrays (``[{"time_s", "foot"}]``) degrade to point annotations
    (earliest = latest = preferred; metadata None).

    Every record gets a deterministic ``gt_id`` (``"<file-stem>#NNN"``)
    used as the stable join key across metrics, forensics, preview and
    T3 rows.
    """

    import json as _json
    from pathlib import Path as _Path

    from .gt import load_ground_truth

    text = _Path(path).read_text(encoding="utf-8-sig")
    head = text.lstrip()[:1]
    stem = _Path(path).stem
    if head == "[":
        records = []
        for i, raw in enumerate(_json.loads(text)):
            t = float(raw["time_s"])
            pts = int(round(t * 1_000_000))
            records.append({
                "gt_id": f"{stem}#{i:03d}",
                "foot": raw.get("foot") or "uncertain",
                "preferred_s": t, "earliest_s": t, "latest_s": t,
                "preferred_pts_us": pts,
                "earliest_pts_us": pts, "latest_pts_us": pts,
                "visibility": None, "viewpoint": None,
                "notes": "legacy v1 record (no window/metadata)",
                "evaluation_range_s": None,
            })
        return records
    gt = load_ground_truth(path)
    return [
        {
            "gt_id": f"{stem}#{i:03d}",
            "foot": c.foot,
            "preferred_s": c.preferred_s,
            "earliest_s": c.earliest_s,
            "latest_s": c.latest_s,
            "preferred_pts_us": c.preferred_pts_us,
            "earliest_pts_us": c.earliest_pts_us,
            "latest_pts_us": c.latest_pts_us,
            "visibility": getattr(c, "visibility", None),
            "viewpoint": getattr(c, "viewpoint", None),
            "notes": getattr(c, "notes", "") or "",
            "evaluation_range_s": [gt.range_start_s, gt.range_end_s],
        }
        for i, c in enumerate(gt.contacts)
    ]


_PHASE_CLASSES = (
    "early_swing", "IC_neighborhood", "mid_stance",
    "terminal_stance", "toe_off_neighborhood",
)


def classify_phase(t_s, gt_times, median_interval):
    """Phase of ``t_s`` against ONE timeline (documented model).

    Approximation per directive T14: midpoints between consecutive GT
    contacts bound each stride segment; within a segment,
    u = (t - prev IC) / (next IC - prev IC) maps onto conventional
    stance/swing sub-phases. The split points are reporting conventions,
    not biomechanical measurements. The ``u < 0.10`` band is the IC
    neighborhood.
    """

    if not gt_times:
        return "UNKNOWN_NO_GT"
    import bisect

    i = bisect.bisect_left(gt_times, t_s)
    if i < len(gt_times) and gt_times[i] == t_s:
        # Exactly on a mark: the IC neighbourhood, never a fallback-band
        # guess (matters when only one same-foot mark exists).
        return "IC_neighborhood"
    prev_t = gt_times[i - 1] if i > 0 else None
    next_t = gt_times[i] if i < len(gt_times) else None
    if prev_t is not None and next_t is not None:
        length = next_t - prev_t
        u = (t_s - prev_t) / length if length > 0 else 0.0
    elif next_t is not None:            # before the first GT contact
        length = median_interval or 0.6
        u = 1.0 - (next_t - t_s) / length
        u = min(max(u, 0.0), 1.0)
    elif prev_t is not None:            # after the last GT contact
        length = median_interval or 0.6
        u = (t_s - prev_t) / length
        u = min(max(u, 0.0), 1.0)
    else:
        return "IC_neighborhood"
    if u < 0.10:
        return "IC_neighborhood"
    if u < 0.50:
        return "mid_stance"
    if u < 0.80:
        return "terminal_stance"
    if u < 0.92:
        return "toe_off_neighborhood"
    return "early_swing"


def classify_phase_for_foot(t_s, foot, gt_records, median_interval=None):
    """Phase of a prediction against its OWN FOOT's GT timeline.

    Directive repair item: alternating contacts from both feet must never
    be treated as one stride -- a left prediction is phased only against
    left GT contacts (right against right). Predictions whose foot is
    unknown/uncertain and GT-only-uncertain cases fall back to the
    combined timeline ONLY when no same-side marks exist, and the result
    is reported under an explicit ``UNKNOWN_*`` class so it can never be
    confused with a real same-foot phase measurement.

    Returns ``(phase_class, n_same_foot_gt)``.
    """

    side = foot if foot in ("left", "right") else None
    times = sorted(
        r["preferred_s"] for r in gt_records
        if r.get("foot") == side
    ) if side else []
    if not times:
        return (
            "UNKNOWN_NO_SAME_FOOT_GT" if side else "UNKNOWN_FOOT_AND_NO_GT"
        ), 0
    if len(times) >= 2:
        gaps = [b - a for a, b in zip(times, times[1:])]
        median_interval = _percentile(sorted(gaps), 0.5)
    cls = classify_phase(t_s, times, median_interval)
    if side is None:
        cls = "UNKNOWN_FOOT_" + cls
    return cls, len(times)


def evaluate_v2(events, gt_records, displayed_fps, tol_ms=None,
                resources=None, min_step_s=0.25, max_step_s=2.0,
                backend=None, method=None,
                diagnostic_horizon_s=DIAGNOSTIC_HORIZON_S):
    """Phase-2 metrics report (§5.4): existence/side/timing/phase/views.

    Association semantics (CTO update #2):

    - EXISTENCE uses the strict matcher only: precision/recall/
      duplicates/misses are unchanged from v1 semantics. A prediction
      four frames late at 30 fps is still a miss + false positive here.
    - TIMING/PHASE use the wider DIAGNOSTIC matcher (monotone, strictly
      one-to-one, local horizon ``diagnostic_horizon_s``, default 0.5 s):
      wide errors appear with signed frame errors and increment
      ``catastrophic_gt3frame`` instead of vanishing.
    - ``uncertain``-foot GT participates in existence/timing but is
      excluded from anatomical attribution denominators.

    ``gt_records`` come from :func:`load_gt_any` (metadata preserved).
    v1 fields survive verbatim nested under ``existence``.
    ``backend``/``method`` identify the producing run in the report.
    """

    fps = float(displayed_fps) if displayed_fps else None

    # A v2 GT file explicitly declares the portion that was exhaustively
    # annotated.  Predictions outside it are neither false positives nor
    # evidence of correctness.  Earlier Metrics v2 ignored this contract and
    # therefore made subset precision depend on unannotated full-clip output.
    ranges = sorted({
        tuple(float(v) for v in r["evaluation_range_s"])
        for r in gt_records if r.get("evaluation_range_s") is not None
    })
    input_event_count = len(events)
    if ranges:
        events = [
            e for e in events
            if any(lo <= float(
                e.get("time_s", 0.0) if isinstance(e, dict) else e.time_s
            ) <= hi for lo, hi in ranges)
        ]

    # -- existence (strict) -------------------------------------------------
    anns = [
        {"time_s": r["preferred_s"], "foot": r["foot"]}
        for r in gt_records
    ]
    existence = evaluate(events, anns, fps, tol_ms=tol_ms)
    n_matched = existence["matched"]
    precision = (
        round(n_matched / existence["n_pred"], _ROUND)
        if existence["n_pred"] else None
    )
    recall = (
        round(n_matched / existence["n_true"], _ROUND)
        if existence["n_true"] else None
    )
    existence = {
        **existence,
        "precision": precision,
        "recall": recall,
    }

    # -- side (attribution excludes uncertain-foot GT) ----------------------
    known = [r for r in gt_records if r["foot"] in ("left", "right")]
    known_anns = [
        {"time_s": r["preferred_s"], "foot": r["foot"]} for r in known
    ]
    side_eval = evaluate(events, known_anns, fps, tol_ms=tol_ms)
    attribution_accuracy = (
        round(1.0 - side_eval["wrong_side"] / side_eval["matched"], _ROUND)
        if side_eval["matched"] else None
    )
    side = {
        "side_agnostic_matched": n_matched,
        "attribution_accuracy": attribution_accuracy,
        "side_flips": side_eval["wrong_side"],
        "side_flip_pairs": side_eval["wrong_side_pairs"],
        "gt_known_foot": len(known),
        "gt_uncertain_foot": len(gt_records) - len(known),
    }

    # -- timing / phase diagnosis (diagnostic association) ------------------
    diag = associate_diagnostic(
        events, gt_records, horizon_s=diagnostic_horizon_s, fps=fps
    )
    tol = _tolerance_s(fps, tol_ms)
    strict_pairs = associate_strict(
        events, [
            {"time_s": r["preferred_s"], "foot": r["foot"]}
            for r in gt_records
        ], tol, fps=fps
    ).pairs
    strict_pair_keys = {
        (p["pred_idx"], p["gt_idx"]) for p in strict_pairs
    }
    errs = sorted(p["err_s"] for p in diag.pairs)
    abs_errs = sorted(abs(e) for e in errs)
    timing = {
        "association": "diagnostic",
        "horizon_s": diagnostic_horizon_s,
        "one_to_one_monotonic": True,
        "n": len(errs),
        "signed_errors_s": [round(e, _ROUND) for e in errs],
        "bias_s": (
            round(sum(errs) / len(errs), _ROUND) if errs else None
        ),
        "median_abs_err_s": (
            round(_percentile(abs_errs, 0.5), _ROUND) if abs_errs else None
        ),
        "p90_abs_err_s": (
            round(_percentile(abs_errs, 0.9), _ROUND) if abs_errs else None
        ),
        "p95_abs_err_s": (
            round(_percentile(abs_errs, 0.95), _ROUND) if abs_errs else None
        ),
    }
    if diag.pairs and fps:
        frames = [abs(e) * fps for e in errs]
        timing["within_1_frame_frac"] = round(
            sum(1 for f in frames if f <= 1.0) / len(frames), _ROUND)
        timing["within_2_frame_frac"] = round(
            sum(1 for f in frames if f <= 2.0) / len(frames), _ROUND)
        timing["catastrophic_gt3frame"] = sum(1 for f in frames if f > 3.0)
    else:
        timing["within_1_frame_frac"] = existence["within_1_frame_frac"]
        timing["within_2_frame_frac"] = None
        timing["catastrophic_gt3frame"] = 0
    timing["outside_defensible_window_count"] = sum(
        1 for p in diag.pairs if not p["inside_gt_window"]
    )
    timing["diagnostic_pairs"] = [
        {
            **p,
            "strictly_matched":
                (p["pred_idx"], p["gt_idx"]) in strict_pair_keys,
            "visibility": gt_records[p["gt_idx"]].get("visibility"),
            "viewpoint": gt_records[p["gt_idx"]].get("viewpoint"),
            "gt_foot": gt_records[p["gt_idx"]].get("foot"),
        }
        for p in diag.pairs
    ]
    timing["unmatched_predictions"] = list(diag.unmatched_preds)
    timing["missed_gt_contacts"] = list(diag.unmatched_gts)

    # -- phase (per predicted foot, same-foot GT timelines) -----------------
    preds = _as_predictions(events)
    gt_times_all = sorted(r["preferred_s"] for r in gt_records)
    gaps_gt = [b - a for a, b in zip(gt_times_all, gt_times_all[1:])]
    median_interval = (
        _percentile(sorted(gaps_gt), 0.5) if gaps_gt else None
    )
    phase_counts: dict[str, int] = {}
    phase_rows = []
    for pi, p in enumerate(preds):
        partner = diag.partner_of_pred(pi)
        cls, n_same = classify_phase_for_foot(
            p["time_s"], p["foot"], gt_records, median_interval
        )
        phase_counts[cls] = phase_counts.get(cls, 0) + 1
        row = {
            "pred_id": p["id"],
            "time_s": round(p["time_s"], _ROUND),
            "foot": p["foot"],
            "phase_class": cls,
            "same_foot_gt_count": n_same,
            "assoc_gt_id": partner["gt_id"] if partner else None,
            "assoc_kind": (
                "strict" if (partner and partner.get("strictly_matched"))
                else "diagnostic" if partner
                else None
            ),
            "err_frames": partner["err_frames"] if partner else None,
            "inside_gt_window": (
                partner["inside_gt_window"] if partner else None
            ),
            "side_ok": partner["side_ok"] if partner else None,
        }
        phase_rows.append(row)
    phase = {
        "model": "per-predicted-foot midpoint segment map over SAME-foot "
                 "GT contacts (u<0.10 IC / <0.50 mid / <0.80 terminal / "
                 "<0.92 toe_off / else early_swing); documented "
                 "approximation, not biomechanics; predictions without "
                 "same-side marks get UNKNOWN_* classes",
        "counts": phase_counts,
        "per_prediction": phase_rows,
    }

    # -- sequence ------------------------------------------------------------
    times = sorted(p["time_s"] for p in preds)
    gaps = [b - a for a, b in zip(times, times[1:])]
    same_side_consecutive = 0
    simultaneous = 0
    cadence_outliers = 0
    ordered = sorted(preds, key=lambda p: p["time_s"])
    for a, b in zip(ordered, ordered[1:]):
        if a["foot"] and b["foot"] and a["foot"] == b["foot"]:
            same_side_consecutive += 1
        if a["foot"] and b["foot"] and a["foot"] != b["foot"] \
                and fps and (b["time_s"] - a["time_s"]) * fps <= 1.0:
            simultaneous += 1
    for gap in gaps:
        if gap < min_step_s or gap > max_step_s:
            cadence_outliers += 1
    n_g = len(ordered)
    sequence = {
        "same_side_consecutive": same_side_consecutive,
        "same_side_consecutive_rate": (
            round(same_side_consecutive / (n_g - 1), _ROUND) if n_g > 1 else None
        ),
        "impossible_simultaneous_ic": simultaneous,
        "cadence_outliers": cadence_outliers,
        "cadence_outlier_rate": (
            round(cadence_outliers / len(gaps), _ROUND) if gaps else None
        ),
    }

    # -- view-stratified results --------------------------------------------
    views = _view_stratify(events, gt_records, fps, tol_ms,
                           diagnostic_horizon_s)

    return {
        "schema_version": 2,
        "identification": {
            "backend": backend,
            "method": method,
            "displayed_fps": fps,
            "viewpoints": sorted({
                str(r.get("viewpoint")) for r in gt_records
            }),
            "evaluation_ranges_s": [list(r) for r in ranges],
            "input_event_count": input_event_count,
            "evaluated_event_count": len(events),
        },
        "existence": existence,
        "side": side,
        "timing": timing,
        "phase": phase,
        "sequence": sequence,
        "views": views,
        "resources": dict(resources or {}),
    }


def _view_stratify(events, gt_records, fps, tol_ms, horizon_s) -> dict:
    """Existence + timing broken down by GT viewpoint stratum.

    Each GT contact carries an operator-assigned ``viewpoint``; pairs are
    attributed to the stratum of their GT contact, so every prediction
    appears in at most one stratum (global associations computed once,
    then filtered -- no double counting across strata). False positives
    have no GT contact and therefore live only in the top-level report;
    this is stated here rather than hidden.
    """
    strata: dict[str, list[dict]] = {}
    for r in gt_records:
        vp = r.get("viewpoint") or "unspecified"
        strata.setdefault(vp, []).append(r)
    out = {}
    for vp in sorted(strata):
        subset = strata[vp]
        sub_events_diag = associate_diagnostic(
            events, subset, horizon_s=horizon_s, fps=fps
        )
        sub_anns = [{"time_s": r["preferred_s"], "foot": r["foot"]}
                    for r in subset]
        ex = evaluate(events, sub_anns, fps, tol_ms=tol_ms)
        errs = sorted(p["err_s"] for p in sub_events_diag.pairs)
        abs_errs = sorted(abs(e) for e in errs)
        out[vp] = {
            "n_gt": len(subset),
            "matched": ex["matched"],
            "misses": ex["misses"],
            "precision": (
                round(ex["matched"] / ex["n_pred"], _ROUND)
                if ex["n_pred"] else None
            ),
            "recall": (
                round(ex["matched"] / ex["n_true"], _ROUND)
                if ex["n_true"] else None
            ),
            "timing_n": len(errs),
            "median_abs_err_s": (
                round(_percentile(abs_errs, 0.5), _ROUND)
                if abs_errs else None
            ),
            "bias_s": (
                round(sum(errs) / len(errs), _ROUND) if errs else None
            ),
            "catastrophic_gt3frame": (
                sum(1 for e in errs if fps and abs(e) * fps > 3.0)
                if errs else 0
            ),
            "visibility_counts": _counts(
                r.get("visibility") or "unspecified" for r in subset
            ),
        }
    return out


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
