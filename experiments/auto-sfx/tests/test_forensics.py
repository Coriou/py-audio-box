"""Tests for Phase-2 T2 forensic windows (structure + selection logic).

Pure-synthetic: reuses the fake trace builders from test_render_markers;
no video files, no matplotlib (plot functions are exercised in-container
by the forensics smoke, not here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.forensics import (  # noqa: E402
    GT_MATCH_HORIZON_S,
    build_window,
    select_windows,
)
from auto_sfx.gt import GTContact  # noqa: E402
from auto_sfx.types import PoseTrace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_render_markers import make_event, make_meta, make_pf, make_trace  # noqa: E402


def _gt(i: int, foot: str) -> dict:
    base = i * 1_000_000
    return {
        "foot": foot,
        "preferred_s": base / 1e6 + 0.033333,
        "earliest_s": base / 1e6,
        "latest_s": base / 1e6 + 0.066667,
    }


def _events_doc_events(trace: PoseTrace):
    return [
        {
            "id": "evt_001",
            "time_s": trace.frames[2].t_s,
            "source_pts_us": trace.frames[2].source_pts_us,
            "earliest_s": trace.frames[2].t_s,
            "latest_s": trace.frames[2].t_s + 0.033,
            "foot": "left",
        },
    ]


class TestSelectWindows:
    def test_predicted_and_missed_gt_split(self):
        trace = make_trace(n=12, fps=12.0)
        events = _events_doc_events(trace)
        gt = [_gt(2, "left"), _gt(8, "right")]   # 2.033s and 8.033s
        # evt_001 at ~0.167s matches nothing; gt@2.033 has no pred either.
        windows = select_windows(events, gt, None)
        kinds = [k for k, _c, _l in windows]
        assert kinds.count("predicted") == 1
        assert kinds.count("missed_gt") == 2

    def test_matched_gt_gets_no_extra_window(self):
        trace = make_trace(n=12, fps=12.0)
        events = _events_doc_events(trace)
        near = {
            "foot": "left",
            "preferred_s": events[0]["time_s"] + 0.01,
            "earliest_s": events[0]["time_s"],
            "latest_s": events[0]["time_s"] + 0.02,
        }
        windows = select_windows(events, [near], None)
        assert [k for k, _c, _l in windows] == ["predicted"]

    def test_event_filter_selects_one(self):
        trace = make_trace(n=12, fps=12.0)
        events = _events_doc_events(trace)
        windows = select_windows(events, None, "evt_001")
        assert len(windows) == 1 and windows[0][0] == "predicted"
        with pytest.raises(ValueError):
            select_windows(events, None, "evt_999")


class TestBuildWindow:
    def test_schema_shape_and_gt_overlay(self):
        trace = make_trace(n=24, fps=12.0)
        events = _events_doc_events(trace)
        gt = [{
            "foot": "left",
            "preferred_s": events[0]["time_s"] + 1.0 / 30.0,
            "earliest_s": events[0]["time_s"],
            "latest_s": events[0]["time_s"] + 2.0 / 30.0,
        }]
        win = build_window(
            trace, events, gt, events[0]["source_pts_us"],
            kind="predicted", half_s=0.5, method="hybrid",
        )
        assert win["schema_version"] == 1
        assert win["kind"] == "predicted"
        assert win["center_pts_us"] == events[0]["source_pts_us"]
        assert win["gt"] is not None and win["gt"]["foot"] == "left"
        s = win["series"]
        for key in ("t_s", "pts_us", "heel_xy", "foot_center_xy", "pelvis_xy",
                    "body_scale_px", "conf_raw", "vel", "accel",
                    "support_band_score", "state", "thresholds"):
            assert key in s, key
        assert win["transitions"] == []
        assert len(s["t_s"]) == len(s["pts_us"]) > 0
        # conf_smoothed mirrors conf_raw by construction (no smoothing here)
        assert s["conf_smoothed"] == s["conf_raw"]

    def test_no_json_nans(self):
        trace = make_trace(n=24, fps=12.0)
        events = _events_doc_events(trace)
        win = build_window(
            trace, events, None, events[0]["source_pts_us"],
            kind="predicted", half_s=0.4,
        )
        text = json.dumps(win)          # NaN would serialize as NaN (invalid)
        assert "NaN" not in text and "Infinity" not in text
