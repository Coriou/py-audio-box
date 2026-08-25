"""Tests for the T3 mechanical error classifier (auto_sfx.classify)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.classify import (  # noqa: E402
    HUMAN_ONLY_CLASSES,
    MECHANICAL_CLASSES,
    classify_run,
)
from auto_sfx.metrics import load_gt_any  # noqa: E402

FPS = 30


def gt_rec(t, foot="left", ident=None, window=1 / 30, viewpoint="sagittal",
           visibility="clear"):
    return {
        "gt_id": ident or f"g@{t}",
        "foot": foot,
        "preferred_s": t,
        "earliest_s": t - window,
        "latest_s": t + window,
        "visibility": visibility,
        "viewpoint": viewpoint,
        "notes": "",
    }


def events_doc(events, backend="mediapipe-lite", method="hybrid"):
    return {
        "video": {"fps_stream": float(FPS)},
        "backend": backend,
        "contact_method": method,
        "events": events,
    }


def ev(t, foot="left", ident=None):
    return {"id": ident or f"e@{t}", "time_s": t, "foot": foot}


def classes_of(report):
    return {r["event_id"] or r["gt_id"]: r["error_class"]
            for r in report["rows"]}


class TestMechanicalClasses:
    def test_correct_prediction_is_none(self):
        gts = [gt_rec(10.0)]
        report = classify_run(
            events_doc([ev(10.0)]), gts)
        assert classes_of(report)["e@10.0"] == "NONE"

    def test_wrong_side_with_good_timing(self):
        gts = [gt_rec(2.0, "left")]
        report = classify_run(
            events_doc([ev(2.0, foot="right")]), gts)
        assert classes_of(report)["e@2.0"] == "WRONG_SIDE"
        row = report["rows"][0]
        assert row["error_frames"] == 0  # timing itself fine
        assert "attribution wrong" in row["notes"]

    def test_duplicate_contact(self):
        gts = [gt_rec(5.0)]
        events = [ev(5.0, ident="first"), ev(5.04, ident="second")]
        report = classify_run(events_doc(events), gts)
        cls = classes_of(report)
        assert cls["first"] == "NONE"
        assert cls["second"] == "DUPLICATE_CONTACT"

    def test_simultaneous_lr(self):
        gts = [gt_rec(1.0, "left"), gt_rec(1.6, "right")]
        events = [ev(1.0, foot="left"),
                  ev(1.0 + 0.5 / FPS, foot="right", ident="twin")]
        report = classify_run(events_doc(events), gts)
        cls = classes_of(report)
        # The right pred sits between L/R GTs; simultaneity outranks.
        assert cls.get("e@1.0") in ("NONE", "SIMULTANEOUS_LR")
        assert cls["twin"] == "SIMULTANEOUS_LR"
        assert any("simultaneous" in r["notes"] for r in report["rows"])

    def test_missed_contact(self):
        gts = [gt_rec(9.0)]
        report = classify_run(events_doc([]), gts)
        assert classes_of(report)["g@9.0"] == "MISSED_CONTACT"

    def test_early_phase(self):
        # Alternating stride 0.6 s per foot; pred 5 frames before the
        # window on the same foot -> early_swing band -> EARLY_PHASE.
        gts = [
            gt_rec(10.0, "left", "L1"),
            gt_rec(10.3, "right", "R1"),
            gt_rec(10.6, "left", "L2"),
            gt_rec(10.9, "right", "R2"),
        ]
        report = classify_run(
            events_doc([ev(10.6 - 5 / FPS, foot="left")]),
            gts)
        row = next(r for r in report["rows"] if r["event_id"])
        assert row["error_class"] == "EARLY_PHASE"
        assert row["error_frames"] == -5

    def test_late_stance(self):
        # Pred half a same-foot stride late: mid-stance of that foot.
        gts = [
            gt_rec(20.0, "left", "L1"),
            gt_rec(20.3, "right", "R1"),
            gt_rec(20.6, "left", "L2"),
            gt_rec(20.9, "right", "R2"),
        ]
        report = classify_run(
            events_doc([ev(20.6 + 0.15, foot="left")]),
            gts)
        row = next(r for r in report["rows"] if r["event_id"])
        assert row["error_class"] == "LATE_STANCE"

    def test_toe_off_confused_as_contact(self):
        gts = [
            gt_rec(30.0, "left", "L1"),
            gt_rec(30.3, "right", "R1"),
            gt_rec(30.6, "left", "L2"),
            gt_rec(30.9, "right", "R2"),
        ]
        # u ~0.86 into the left stride -> toe_off_neighborhood.
        report = classify_run(
            events_doc([ev(30.6 + 0.517, foot="left")]),
            gts)
        row = next(r for r in report["rows"] if r["event_id"])
        assert row["error_class"] == "TOE_OFF_CONFUSED_AS_CONTACT"


class TestEvidenceAndSafety:
    def test_unknown_far_prediction_carries_evidence(self):
        gts = [gt_rec(1.0)]
        report = classify_run(
            events_doc([ev(50.0)]), gts)
        row = next(r for r in report["rows"] if r["event_id"] == "e@50.0")
        assert row["error_class"] == "UNKNOWN"
        assert "nearest" in row["notes"]

    def test_pose_loss_requires_full_window_overlap(self):
        gts = [gt_rec(4.0)]
        diag_doc = {"diagnostics": [
            {"code": "POSE_LOST", "t0_s": 3.9, "t1_s": 4.1,
             "detail": ""},
        ]}
        report = classify_run(
            events_doc([]), gts, diag_doc)
        assert classes_of(report)["g@4.0"] == "POSE_LOSS"

    def test_missed_without_pose_loss_stays_missed(self):
        gts = [gt_rec(4.0)]
        diag_doc = {"diagnostics": [
            {"code": "POSE_LOST", "t0_s": 7.0, "t1_s": 8.0, "detail": ""},
        ]}
        report = classify_run(
            events_doc([]), gts, diag_doc)
        assert classes_of(report)["g@4.0"] == "MISSED_CONTACT"

    def test_human_only_classes_never_assigned(self):
        gts = [
            gt_rec(1.0, "left"), gt_rec(1.3, "right"),
            gt_rec(1.6, "left"), gt_rec(50.0, "right"),
        ]
        events = [
            ev(1.0, foot="right"),                    # flip
            ev(1.001, foot="left", ident="dup-ish"),  # simultaneous
            ev(1.6, foot="left"),                     # ok
            ev(49.99, foot="right", ident="near-ok"),
        ]
        report = classify_run(events_doc(events), gts)
        assigned = {r["error_class"] for r in report["rows"]}
        assert assigned <= MECHANICAL_CLASSES
        assert assigned.isdisjoint(HUMAN_ONLY_CLASSES)

    def test_uncertain_foot_not_judged_on_side(self):
        gts = [gt_rec(2.0, foot="uncertain")]
        report = classify_run(
            events_doc([ev(2.0, foot="right")]), gts)
        assert classes_of(report)["e@2.0"] == "NONE"

    def test_table_columns_complete(self):
        gts = [gt_rec(1.0, viewpoint="frontal", visibility="partial")]
        report = classify_run(events_doc([ev(1.0)]), gts)
        row = report["rows"][0]
        for col in ("event_id", "gt_id", "prediction_time_s",
                    "error_frames", "error_ms", "error_class",
                    "backend", "method", "viewpoint", "visibility",
                    "notes"):
            assert col in row
        assert row["viewpoint"] == "frontal"
        assert row["visibility"] == "partial"
        ident = report["identification"]
        assert ident["backend"] == "mediapipe-lite"
        assert ident["method"] == "hybrid"


class TestCliIntegration:
    def test_classify_cli_writes_outputs(self, tmp_path):
        from auto_sfx.cli import main as cli_main

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "events.json").write_text(json.dumps(
            events_doc([ev(1.0), ev(9.9, ident="wild")])), encoding="utf-8")
        (run_dir / "diagnostics.json").write_text(json.dumps(
            {"diagnostics": []}), encoding="utf-8")
        gt_path = tmp_path / "clip.gt.json"
        gt_path.write_text(json.dumps({
            "schema_version": 2, "clip": "clip.mp4",
            "range_s": {"start": 0.0, "end": 12.0},
            "annotator": "op", "fps_display": 30.0,
            "contacts": [_v2_contact(1.0, "left"),
                         _v2_contact(10.0, "right")],
        }), encoding="utf-8")

        rc = cli_main(["classify", str(run_dir), str(gt_path)])
        assert rc == 0
        out = json.loads(
            (run_dir / "classify.json").read_text(encoding="utf-8"))
        assert out["identification"]["ground_truth_n"] == 2
        csv_rows = (run_dir / "classify.csv").read_text().splitlines()
        assert csv_rows[0].startswith("event_id,gt_id")


def _v2_contact(t, foot):
    pts = int(round(t * 1_000_000))
    win = int(round((1 / 30) * 1_000_000))
    return {
        "foot": foot,
        "earliest_pts_us": pts - win,
        "preferred_pts_us": pts,
        "latest_pts_us": pts + win,
        "visibility": "clear",
        "viewpoint": "sagittal",
        "notes": "",
    }
