"""Focused tests for the shared association module (CTO update #2).

Covers the directive repair items: ±4-frame errors stay visible to the
diagnostic matcher while strict existence rejects them; one-to-one-ness
when one prediction sits near two GT contacts; monotonicity; side flips
independent of timing; uncertain foot; horizon cap.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.association import (  # noqa: E402
    DIAGNOSTIC_HORIZON_S,
    associate_diagnostic,
    associate_strict,
)


def gt(t, foot="left", ident=None, window=1 / 30):
    return {
        "gt_id": ident or f"g@{t}",
        "foot": foot,
        "preferred_s": t,
        "earliest_s": t - window,
        "latest_s": t + window,
    }


def pred(t, foot="left", ident=None):
    return {"id": ident or f"p@{t}", "time_s": t, "foot": foot}


class TestStrict:
    def test_four_frames_late_at_30fps_is_rejected(self):
        # +4 frames = 133 ms > tol (max(66 ms, 50 ms) = 66 ms).
        res = associate_strict([pred(10.0 + 4 / 30)], [gt(10.0)],
                               tol_s=0.066, fps=30)
        assert res.pairs == []
        assert res.unmatched_preds[0]["reason"] == "unmatched"
        assert len(res.unmatched_gts) == 1

    def test_duplicate_pred_lands_on_taken_gt(self):
        res = associate_strict(
            [pred(5.0, ident="first"), pred(5.02, ident="second")],
            [gt(5.0)], tol_s=0.066, fps=30)
        assert [p["pred_id"] for p in res.pairs] == ["first"]
        assert res.unmatched_preds[0]["pred_id"] == "second"
        assert res.unmatched_preds[0]["reason"] == "dup"

    def test_side_flip_pairs_but_flags_side_ok_false(self):
        res = associate_strict([pred(2.0, foot="right")], [gt(2.0, "left")],
                               tol_s=0.066, fps=30)
        assert len(res.pairs) == 1
        assert res.pairs[0]["side_ok"] is False


class TestDiagnostic:
    def test_horizon_default_is_half_second(self):
        assert DIAGNOSTIC_HORIZON_S == 0.5

    def test_four_frame_error_visible_with_signed_frames(self):
        res = associate_diagnostic([pred(10.0 + 4 / 30)], [gt(10.0)],
                                   fps=30)
        pair = res.pairs[0]
        assert pair["err_frames"] == 4
        assert pair["err_s"] == 0.133333
        # Outside the defensible GT window [+/-1 frame].
        assert pair["inside_gt_window"] is False

    def test_early_negative_frames(self):
        res = associate_diagnostic([pred(10.0 - 5 / 30)], [gt(10.0)],
                                   fps=30)
        assert res.pairs[0]["err_frames"] == -5

    def test_one_prediction_between_two_gts_matches_exactly_one(self):
        gts = [gt(10.0, "left", "a"), gt(10.15, "right", "b")]
        res = associate_diagnostic([pred(10.05, "left", "p1")], gts)
        assert len(res.pairs) == 1
        assert len(res.unmatched_gts) == 1

    def test_two_predictions_near_one_gt_match_at_most_one(self):
        res = associate_diagnostic(
            [pred(5.0, ident="d1"), pred(5.05, ident="d2")],
            [gt(5.0, ident="x")])
        assert len(res.pairs) == 1
        assert len(res.unmatched_preds) == 1

    def test_monotonic_order_preserved(self):
        gts = [gt(1.0, "left", "a"), gt(1.30, "right", "b")]
        preds = [pred(0.98, "left", "q1"), pred(1.28, "right", "q2")]
        res = associate_diagnostic(preds, gts)
        assert [(p["gt_id"], p["pred_id"]) for p in res.pairs] == [
            ("a", "q1"), ("b", "q2")
        ]

    def test_no_crossing_even_when_input_order_is_scrambled(self):
        # p2 is earlier in time than p1 despite its later list position;
        # a nearest-first greedy without order constraints could cross,
        # the monotone alignment must not.
        gts = [gt(1.00, "left", "a"), gt(1.20, "right", "b")]
        preds = [pred(1.16, "left", "p1"), pred(1.04, "right", "p2")]
        res = associate_diagnostic(preds, gts, horizon_s=0.5)
        assert len(res.pairs) == 2
        # In prediction-TIME order, partner GT times must be
        # non-decreasing (the definition of a non-crossing matching).
        times = {i: p["time_s"] for i, p in enumerate(preds)}
        ordered = sorted(res.pairs, key=lambda r: times[r["pred_idx"]])
        gt_times = [gts[r["gt_idx"]]["preferred_s"] for r in ordered]
        assert gt_times == sorted(gt_times)

    def test_side_flips_still_pair_and_report_side_ok_false(self):
        gts = [gt(2.0, "left", "L"), gt(2.6, "right", "R")]
        preds = [pred(2.01, "right", "f1"), pred(2.61, "left", "f2")]
        res = associate_diagnostic(preds, gts, fps=30)
        assert all(p["side_ok"] is False for p in res.pairs)
        assert res.unmatched_gts == []

    def test_uncertain_foot_pairs_with_side_ok_none(self):
        res = associate_diagnostic(
            [pred(3.01, "right")],
            [gt(3.0, "uncertain", "u", window=0.02)], fps=30)
        assert len(res.pairs) == 1
        assert res.pairs[0]["side_ok"] is None

    def test_beyond_horizon_not_paired(self):
        res = associate_diagnostic([pred(0.6)], [gt(0.0)])
        assert res.pairs == []
        assert len(res.unmatched_preds) == 1 and len(res.unmatched_gts) == 1

    def test_unpaired_events_listed(self):
        res = associate_diagnostic(
            [pred(100.0, ident="far")], [gt(1.0, ident="lone")])
        assert res.unmatched_preds[0]["pred_id"] == "far"
        assert res.unmatched_gts[0]["gt_id"] == "lone"
