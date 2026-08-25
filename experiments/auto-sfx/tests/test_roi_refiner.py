"""Behavioral checks for camera-compensated foot-pixel refinement."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_sfx import roi_refiner  # noqa: E402
from auto_sfx.types import (  # noqa: E402
    ContactEvent, DOMINANT_ACTOR_ID, EVENT_FAMILY_FOOT_CONTACT,
)
from test_contacts import build_walk, uniform_times  # noqa: E402


def _event(trace, frame=5):
    fr = trace.frames[frame]
    return ContactEvent(
        id="raw_001", family=EVENT_FAMILY_FOOT_CONTACT,
        actor_id=DOMINANT_ACTOR_ID, foot="left",
        source_pts_us=fr.source_pts_us, time_s=fr.t_s,
        earliest_s=fr.t_s, latest_s=fr.t_s,
        score=0.7, observable=True, detector="pose-state",
        method="contact-state-v1", evidence={},
    )


def test_common_camera_flow_is_not_contact_motion():
    # Identical common flow in foot and ground patches must subtract to zero.
    prev = np.zeros((80, 100), dtype=np.uint8)
    curr = prev.copy()
    flow = np.zeros((80, 100, 2), dtype=np.float32)
    flow[..., 0] = 4.0
    flow[..., 1] = -2.0
    got = roi_refiner._region_evidence(
        prev, curr, flow, (50.0, 40.0), 50.0,
        np.asarray([4.0, -2.0]), roi_refiner.roi_config(),
    )
    assert got is not None
    assert got["relative_flow_px"] == 0.0
    assert got["static_pixel_fraction"] == 1.0


def test_refiner_selects_relative_deceleration_change_point(monkeypatch):
    trace = build_walk(uniform_times(0.5))
    raw = _event(trace, frame=5)
    dummy = [np.zeros((20, 20, 3), np.uint8) for _ in trace.frames]
    monkeypatch.setattr(
        roi_refiner, "_decode_frames", lambda _p, _needed=None: dummy
    )

    def fake_pair(_prev, _curr, _trace, frame, _side, _scale, _cfg):
        speed = 4.0 if frame == 6 else 0.1 if frame == 7 else 1.0
        static = 0.9 if frame == 7 else 0.1
        region = {
            "relative_flow_px": speed,
            "static_pixel_fraction": static,
            "edge_persistence": 0.8 if frame == 7 else 0.1,
            "edge_pixels": 20,
            "ground_flow_xy": [5.0, 1.0],
        }
        return {
            "abstain": False, "global_motion_px": 5.1,
            "global_response": 0.8, "heel": region, "toe": region,
        }

    monkeypatch.setattr(roi_refiner, "_pair_metrics", fake_pair)
    events, diagnostics, summary = roi_refiner.refine_contacts(
        "unused.mp4", trace, [raw], {"roi_half_window_frames": 3}
    )
    assert diagnostics == []
    assert len(events) == 1
    assert events[0].evidence["raw_pose_frame"] == 5
    assert events[0].evidence["roi_refined_frame"] == 6
    assert events[0].evidence["roi_shift_frames"] == 1
    assert events[0].evidence["camera_motion_is_contact_evidence"] is False
    assert summary["raw_pose_events"] == 1
    assert summary["refined_events"] == 1


def test_refiner_abstains_when_motion_compensation_is_unavailable(monkeypatch):
    trace = build_walk(uniform_times(0.5))
    raw = _event(trace, frame=5)
    dummy = [np.zeros((20, 20, 3), np.uint8) for _ in trace.frames]
    monkeypatch.setattr(
        roi_refiner, "_decode_frames", lambda _p, _needed=None: dummy
    )
    monkeypatch.setattr(
        roi_refiner, "_pair_metrics",
        lambda *_a, **_k: {
            "abstain": True, "global_motion_px": 80.0,
            "global_response": 0.0,
        },
    )
    events, diagnostics, summary = roi_refiner.refine_contacts(
        "unused.mp4", trace, [raw]
    )
    assert events == []
    assert diagnostics and "abstained" in diagnostics[0]["detail"]
    assert summary["abstained_events"] == 1
