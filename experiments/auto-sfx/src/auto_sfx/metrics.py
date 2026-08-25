"""Evaluation metrics: detector events vs manual foot-contact annotations.

Annotation format (JSON list, one object per manually marked contact)::

    [{"time_s": 1.432, "foot": "left"}, {"time_s": 2.101, "foot": "right"}]

Only ``time_s`` (seconds on the source presentation clock) and ``foot``
(``"left" | "right"``) participate in evaluation; extra keys are tolerated
and ignored. Predictions are ``events.json`` entries (dicts) or
``types.ContactEvent`` objects; matching likewise uses only their
``time_s`` / ``foot`` / ``id``.

Matching rule (POC simplification of docs/auto-sfx-research.md §12.7):
greedy nearest-first one-to-one association over all candidate pairs with
``|dt| <= tol``, where ``tol`` defaults to ``max(66 ms, 1.5 / displayed_fps)``
and may be overridden with ``tol_ms``. A prediction pairing with an
already-matched annotation within tolerance counts as a *duplicate*, not a
fresh false positive. Foot side never blocks a match; side mismatches are
reported separately as ``wrong_side``.
"""

from __future__ import annotations

import json
import math

try:  # package-relative when imported as auto_sfx.metrics
    from .types import SIDES
except ImportError:  # pragma: no cover - direct module import
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

    foot = _norm_side(obj.get("foot"))
    if foot is None:
        return None, f"'foot' must be 'left' or 'right': {obj.get('foot')!r}"
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

    candidates = []
    for pi, pred in enumerate(preds):
        for ai, ann in enumerate(anns):
            dt = abs(pred["time_s"] - ann["time_s"])
            if dt <= tol:
                candidates.append((dt, pi, ai))
    candidates.sort()

    pred_match = [None] * len(preds)  # ann index, "dup", or None
    ann_match = [None] * len(anns)  # pred index or None
    duplicate_preds = set()
    for dt, pi, ai in candidates:
        if pred_match[pi] is not None:
            continue
        if ann_match[ai] is not None:
            pred_match[pi] = "dup"
            duplicate_preds.add(pi)
            continue
        pred_match[pi] = ai
        ann_match[ai] = pi

    matched_pairs = []
    wrong_side_pairs = []
    abs_errors = []
    for ai, pi in enumerate(ann_match):
        if pi is None:
            continue
        err = preds[pi]["time_s"] - anns[ai]["time_s"]
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
    duplicates = len(duplicate_preds)
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
            "id": p["id"],
            "time_s": round(p["time_s"], _ROUND),
            "foot": p["foot"],
        }
        for pi, p in enumerate(preds)
        if pred_match[pi] is None or pred_match[pi] == "dup"
    ]
    unmatched_anns = [
        {
            "time_s": round(a["time_s"], _ROUND),
            "foot": a["foot"],
        }
        for ai, a in enumerate(anns)
        if ann_match[ai] is None
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
# ---------------------------------------------------------------------------


def load_gt_any(path):
    """Load ground truth as uniform records, accepting v2 and legacy v1.

    v2 (``auto_sfx.gt`` schema) is canonical: preferred/earliest/latest
    windows survive. Legacy v1 arrays (``[{"time_s", "foot"}]``) degrade to
    point annotations (earliest = latest = preferred).
    """

    import json as _json
    from pathlib import Path as _Path

    from .gt import load_ground_truth

    text = _Path(path).read_text(encoding="utf-8-sig")
    head = text.lstrip()[:1]
    if head == "[":
        records, _problems = [], []
        for raw in _json.loads(text):
            t = float(raw["time_s"])
            records.append({
                "foot": raw.get("foot"),
                "preferred_s": t, "earliest_s": t, "latest_s": t,
            })
        return records
    gt = load_ground_truth(path)
    return [
        {
            "foot": c.foot,
            "preferred_s": c.preferred_s,
            "earliest_s": c.earliest_s,
            "latest_s": c.latest_s,
        }
        for c in gt.contacts
    ]


_PHASE_CLASSES = (
    "early_swing", "IC_neighborhood", "mid_stance",
    "terminal_stance", "toe_off_neighborhood",
)


def classify_phase(t_s, gt_times, median_interval):
    """Phase of ``t_s`` against the GT contact timeline (documented model).

    Approximation per directive T14: midpoints between consecutive GT
    contacts bound each stride segment; within a segment, u = (t - prev IC)
    / (next IC - prev IC) maps onto conventional stance/swing sub-phases.
    The split points are reporting conventions, not biomechanical
    measurements. The ``u < 0.10`` band is the IC neighborhood.
    """

    if not gt_times:
        return "UNKNOWN_NO_GT"
    import bisect

    i = bisect.bisect_left(gt_times, t_s)
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


def evaluate_v2(events, gt_records, displayed_fps, tol_ms=None,
                resources=None, min_step_s=0.25, max_step_s=2.0):
    """Phase-2 metrics report (§5.4): existence/side/timing/phase/sequence.

    ``gt_records`` are uniform records from :func:`load_gt_any`. v1 fields
    survive verbatim nested under ``existence`` for back-compat.
    """

    fps = float(displayed_fps) if displayed_fps else None
    anns = [
        {"time_s": r["preferred_s"], "foot": r["foot"] or None}
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

    errs = sorted(p["err_s"] for p in existence["matched_pairs"])
    abs_errs = sorted(abs(e) for e in errs)
    timing = {
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
    if errs and fps:
        frames = [abs(e) * fps for e in errs]
        timing["within_1_frame_frac"] = round(
            sum(1 for f in frames if f <= 1.0) / len(frames), _ROUND)
        timing["within_2_frame_frac"] = round(
            sum(1 for f in frames if f <= 2.0) / len(frames), _ROUND)
        timing["catastrophic_gt3frame"] = sum(1 for f in frames if f > 3.0)
    else:
        timing["within_1_frame_frac"] = existence["within_1_frame_frac"]
        timing["within_2_frame_frac"] = None
        timing["catastrophic_gt3frame"] = None

    preds = _as_predictions(events)
    gt_times = sorted(r["preferred_s"] for r in gt_records)
    gaps_gt = [b - a for a, b in zip(gt_times, gt_times[1:])]
    median_interval = (
        _percentile(sorted(gaps_gt), 0.5) if gaps_gt else None
    )
    phase_counts = {c: 0 for c in _PHASE_CLASSES}
    phase_rows = []
    for p in preds:
        cls = classify_phase(p["time_s"], gt_times, median_interval)
        if cls in phase_counts:
            phase_counts[cls] += 1
        phase_rows.append({
            "pred_id": p["id"], "time_s": round(p["time_s"], _ROUND),
        })
    phase = {
        "model": "midpoint segment map (u<0.10 IC / <0.50 mid / <0.80 "
                 "terminal / <0.92 toe_off / else early_swing); "
                 "documented approximation, not biomechanics",
        "counts": phase_counts,
        "per_prediction": phase_rows,
    }

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

    return {
        "schema_version": 2,
        "existence": existence,
        "side": side,
        "timing": timing,
        "phase": phase,
        "sequence": sequence,
        "resources": dict(resources or {}),
    }
