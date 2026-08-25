"""Shared one-to-one GT<->prediction association (Phase-2 repair).

One implementation, four consumers: ``metrics``, the T15 preview
(``render``), ``forensics``, and T3 classification.

Two deliberately separate concepts (docs/auto-sfx-phase2-truth-repair.md;
CTO update #2 directive):

STRICT -- "did the detector get this contact right?"
    Greedy nearest-first one-to-one matching within the correctness
    tolerance (default ``max(66 ms, 1.5 / displayed_fps)``, about two
    displayed frames). Drives precision/recall/duplicates/misses ONLY.
    A prediction outside tolerance is an existence failure no matter how
    small its absolute error; that is the point -- existence must not be
    relaxed just because the wider band would flatter it.

DIAGNOSTIC -- "what did the detector actually do near this contact?"
    Wider local horizon (default 0.5 s, the directive ceiling), still
    strictly one-to-one in BOTH directions and order-preserving: if
    prediction A precedes prediction B in time, their partners are
    non-decreasing in time (a monotone alignment, computed by dynamic
    programming that maximises the number of pairs, tie-broken by
    smaller total |dt|). Drives signed timing distributions, phase
    diagnosis, forensic links, preview labels and T3 rows. It NEVER
    changes existence counts: a prediction that only matches under the
    diagnostic horizon remains a strict miss/FP while still contributing
    its signed error (e.g. ``+4f``) and incrementing
    ``catastrophic_gt3frame``.

Side is never a blocking criterion in either matcher: timing is measured
side-agnostically (directive T7 -- good timing with a flipped side is a
side failure, not a timing failure); side agreement is reported per pair
as ``side_ok`` (``None`` when either side is unknown/uncertain).

GT records are uniform dicts (see ``metrics.load_gt_any``)::

    {"gt_id", "foot", "preferred_s", "earliest_s", "latest_s",
     ...optional metadata...}

Predictions are dicts or objects exposing ``id`` / ``time_s`` /
``foot`` (events.json entries or ContactEvent objects).
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:  # package-relative when imported as auto_sfx.association
    from .types import SIDES
except ImportError:  # pragma: no cover - direct module import
    from types import SIDES  # type: ignore[no-redef]

#: Widest defensible diagnostic pairing horizon (seconds). The directive
#: caps diagnostic association at 0.5 s; forensics/T15 use the same value
#: so every artifact links consistently.
DIAGNOSTIC_HORIZON_S = 0.5

_ROUND = 6


# ---------------------------------------------------------------------------
# Uniform access helpers
# ---------------------------------------------------------------------------


def norm_side(value):
    """'left'/'right' normalised, else None (missing, uncertain, invalid)."""
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in SIDES:
            return stripped
    return None


def get_field(record, name):
    """Read ``name`` from a dict or attribute-style record."""
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def pred_time(pred) -> float:
    return float(get_field(pred, "time_s"))


def pred_id(pred, index: int) -> str:
    raw = get_field(pred, "id")
    return str(raw) if raw is not None else f"event_{index}"


@dataclass(frozen=True)
class AssocInput:
    """Index-aligned view over one side of an association problem."""

    times: list[float]
    ids: list[str]
    sides: list[str | None]
    starts: list[float]   # GT earliest (== preferred for predictions)
    ends: list[float]     # GT latest   (== preferred for predictions)

    @classmethod
    def of_predictions(cls, preds) -> "AssocInput":
        rows = list(preds)
        return cls(
            times=[pred_time(p) for p in rows],
            ids=[pred_id(p, i) for i, p in enumerate(rows)],
            sides=[norm_side(get_field(p, "foot")) for p in rows],
            starts=[pred_time(p) for p in rows],
            ends=[pred_time(p) for p in rows],
        )

    @classmethod
    def of_ground_truth(cls, gt_records) -> "AssocInput":
        """Accept v2 records (preferred_s + windows) and legacy v1
        annotation dicts (``time_s``, no window -- point annotation)."""
        rows = list(gt_records)

        def pref(r):
            return float(r["preferred_s"] if "preferred_s" in r
                         else r["time_s"])

        return cls(
            times=[pref(r) for r in rows],
            ids=[str(r.get("gt_id", f"gt_{i}")) for i, r in enumerate(rows)],
            sides=[
                s if s in ("left", "right", "uncertain")
                else norm_side(r.get("foot"))
                for s in (r.get("foot") for r in rows)
            ],
            starts=[
                float(r["earliest_s"]) if "earliest_s" in r else pref(r)
                for r in rows
            ],
            ends=[
                float(r["latest_s"]) if "latest_s" in r else pref(r)
                for r in rows
            ],
        )

@dataclass
class AssocResult:
    """Outcome of one association pass (JSON-serialisable)."""

    #: {"pred_idx", "gt_idx", "err_s", "err_frames", "inside_gt_window",
    #:  "side_ok"} -- sorted by gt time then pred time.
    pairs: list[dict] = field(default_factory=list)
    #: Predictions with no partner ("dup" for strict-only duplicates).
    unmatched_preds: list[dict] = field(default_factory=list)
    #: Ground-truth contacts with no partner.
    unmatched_gts: list[dict] = field(default_factory=list)

    def partner_of_pred(self, pred_idx: int) -> dict | None:
        for p in self.pairs:
            if p["pred_idx"] == pred_idx:
                return p
        return None

    def partner_of_gt(self, gt_idx: int) -> dict | None:
        for p in self.pairs:
            if p["gt_idx"] == gt_idx:
                return p
        return None

def _pair_record(a: AssocInput, ai: int, b: AssocInput, bi: int,
                 fps: float | None) -> dict:
    """One association row; ``a`` is always the predictions side."""
    err = a.times[ai] - b.times[bi]
    inside = b.starts[bi] <= a.times[ai] <= b.ends[bi]
    side_a, side_b = a.sides[ai], b.sides[bi]
    if side_a in ("left", "right") and side_b in ("left", "right"):
        side_ok = side_a == side_b
    else:
        side_ok = None  # unknown/uncertain on either side: not judgeable
    return {
        "pred_idx": ai,
        "gt_idx": bi,
        "pred_id": a.ids[ai],
        "gt_id": b.ids[bi],
        "err_s": round(err, _ROUND),
        "err_frames": (
            int(round(err * float(fps))) if fps and fps > 0 else None
        ),
        "inside_gt_window": bool(inside),
        "side_ok": side_ok,
    }


# ---------------------------------------------------------------------------
# STRICT association (existence correctness)
# ---------------------------------------------------------------------------


def associate_strict(preds, gts, tol_s: float,
                     fps: float | None = None) -> AssocResult:
    """Greedy nearest-first one-to-one matching within ``tol_s``.

    Semantics identical to the v1 matcher in ``metrics.evaluate``: all
    candidate pairs within tolerance visited in increasing ``|dt|`` order
    (ties broken by prediction then GT index); a pair is accepted when
    both sides are still free; a free prediction whose remaining best
    pairing lands on an already-taken GT is a *duplicate*, recorded in
    ``unmatched_preds`` with ``reason="dup"``.
    """
    a = AssocInput.of_predictions(preds)
    b = AssocInput.of_ground_truth(gts)

    candidates = []
    for pi, pt in enumerate(a.times):
        for gi, gt_t in enumerate(b.times):
            dt = abs(pt - gt_t)
            if dt <= tol_s:
                candidates.append((dt, pi, gi))
    candidates.sort()

    pred_match: list[int | None] = [None] * len(a.times)
    gt_match: list[int | None] = [None] * len(b.times)
    duplicate_preds: set[int] = set()
    for _dt, pi, gi in candidates:
        if pred_match[pi] is not None:
            continue
        if gt_match[gi] is not None:
            duplicate_preds.add(pi)
            continue
        pred_match[pi] = gi
        gt_match[gi] = pi

    result = AssocResult()
    for gi, pi in enumerate(gt_match):
        if pi is None:
            continue
        result.pairs.append(
            _pair_record(a, pi, b, gi, fps))
    result.pairs.sort(key=lambda r: (b.times[r["gt_idx"]],
                                     a.times[r["pred_idx"]]))
    taken = {p["pred_idx"] for p in result.pairs}
    for pi in range(len(a.times)):
        if pi in taken:
            continue
        row = {
            "pred_idx": pi, "pred_id": a.ids[pi],
            "foot": a.sides[pi], "time_s": a.times[pi],
            "reason": "dup" if pi in duplicate_preds else "unmatched",
        }
        result.unmatched_preds.append(row)
    for gi in range(len(b.times)):
        if gt_match[gi] is None:
            result.unmatched_gts.append({
                "gt_idx": gi, "gt_id": b.ids[gi],
                "foot": b.sides[gi], "preferred_s": b.times[gi],
            })
    return result


# ---------------------------------------------------------------------------
# DIAGNOSTIC association (wide, monotone, one-to-one)
# ---------------------------------------------------------------------------


def associate_diagnostic(preds, gts, horizon_s: float = DIAGNOSTIC_HORIZON_S,
                         fps: float | None = None) -> AssocResult:
    """Wider diagnostic pairing: local horizon, monotone, one-to-one.

    Both lists are sorted by time and aligned with dynamic programming:
    ``f(i, j)`` = best outcome aligning predictions[i:] against gts[j:]
    where each step either advances one list (skip) or pairs the current
    heads when ``|dt| <= horizon_s``. Monotonicity and one-to-one-ness are
    structural: indices only move forward, each element pairs at most
    once. Among maximum-cardinality alignments the smallest total |dt|
    wins (deterministic tie-break).

    A prediction may therefore match AT MOST one GT contact even when it
    sits between two of them, and vice versa. Pairs beyond the strict
    tolerance still carry signed ``err_s`` / ``err_frames`` so wide
    errors (the 2-6-frame wrong-phase band) reach the timing report and
    ``catastrophic_gt3frame`` instead of vanishing.
    """
    a = AssocInput.of_predictions(preds)
    b = AssocInput.of_ground_truth(gts)

    order_p = sorted(range(len(a.times)), key=lambda i: (a.times[i], i))
    order_g = sorted(range(len(b.times)), key=lambda i: (b.times[i], i))
    pts = [a.times[i] for i in order_p]
    gts_t = [b.times[i] for i in order_g]

    n, m = len(pts), len(gts_t)
    # dp[i][j] = (pairs_achievable_from_i_j, minus_total_abs_dt),
    # compared lexicographically (maximise count, then minimise total
    # |dt|). Boundary cells (one list exhausted) hold (0, 0.0).
    zero = (0, 0.0)
    dp = [[zero] * (m + 1) for _ in range(n + 1)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = max(dp[i + 1][j], dp[i][j + 1])
            choice = False
            dt = abs(pts[i] - gts_t[j])
            if dt <= horizon_s:
                sub = dp[i + 1][j + 1]
                cand = (sub[0] + 1, sub[1] - dt)
                if cand > best:
                    best, choice = cand, True
            dp[i][j] = best
            take[i][j] = choice

    result = AssocResult()
    i = j = 0
    paired_p: set[int] = set()
    paired_g: set[int] = set()
    while i < n and j < m:
        if take[i][j]:
            pi, gi = order_p[i], order_g[j]
            result.pairs.append(
                _pair_record(a, pi, b, gi, fps))
            paired_p.add(pi)
            paired_g.add(gi)
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    result.pairs.sort(key=lambda r: (b.times[r["gt_idx"]],
                                     a.times[r["pred_idx"]]))
    for pi in range(len(a.times)):
        if pi not in paired_p:
            result.unmatched_preds.append({
                "pred_idx": pi, "pred_id": a.ids[pi],
                "foot": a.sides[pi], "time_s": a.times[pi],
                "reason": "unmatched",
            })
    for gi in range(len(b.times)):
        if gi not in paired_g:
            result.unmatched_gts.append({
                "gt_idx": gi, "gt_id": b.ids[gi],
                "foot": b.sides[gi], "preferred_s": b.times[gi],
            })
    return result
