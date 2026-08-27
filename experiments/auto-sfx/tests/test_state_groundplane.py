"""Synthetic tests for contact-groundplane-v1 (no real clips, no GT files)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_sfx.state_groundplane import (  # noqa: E402
    GROUNDPLANE_METHOD,
    default_groundplane_config,
    detect_groundplane_contacts,
    validate_groundplane_config,
)
from auto_sfx.types import (  # noqa: E402
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
)
from test_contacts import make_meta, vfr_times  # noqa: E402


APEX_FRAME = 40
LAND_FRAME = 43
N_FRAMES = 60
GROUND_Y = 200.0
AIR_Y = 160.0
HIP_Y = 100.0
ALLOWED_STATES = {"SWING", "COLLISION", "SUPPORT", "UNCERTAIN"}


def _pt(x: float, y: float, vis: float = 0.95) -> Point:
    return Point(x=float(x), y=float(y), z=0.0, visibility=float(vis))


def _foot(ankle_x: float, ankle_y: float) -> FootObservation:
    return FootObservation(
        heel=_pt(ankle_x - 4.0, ankle_y),
        ankle=_pt(ankle_x, ankle_y),
        toe=_pt(ankle_x + 8.0, ankle_y),
    )


def _left_xy(i: int) -> tuple[float, float]:
    hip_x = 100.0 + 1.2 * i
    if i <= APEX_FRAME:
        rel = 20.0 - 2.0 * (APEX_FRAME - i)
        x = hip_x + rel
    else:
        x = (100.0 + 1.2 * APEX_FRAME) + 20.0
    if i <= APEX_FRAME:
        y = AIR_Y
    elif i == APEX_FRAME + 1:
        y = 180.0
    elif i == APEX_FRAME + 2:
        y = 192.0
    else:
        y = GROUND_Y
    return x, y


def build_early_apex_trace(times: list[float] | None = None) -> PoseTrace:
    if times is None:
        times = [k / 30.0 for k in range(N_FRAMES)]
    n = len(times)
    frames = []
    for i, t in enumerate(times):
        hip_x = 100.0 + 1.2 * i
        lx, ly = _left_xy(i)
        frames.append(PersonFrame(
            t_s=float(t),
            source_pts_us=int(round(float(t) * 1_000_000)),
            frame_index=i,
            person_id=0,
            score=0.9,
            feet={
                "left": _foot(lx, ly),
                "right": _foot(70.0, GROUND_Y),
            },
            hips=_pt(hip_x, HIP_Y, 0.99),
            shoulders_mid=None,
            landmarks=[],
            lost=False,
        ))
    meta = make_meta(times, vfr=len(times) != N_FRAMES or times[-1] != (n - 1) / 30.0)
    meta.n_frames = n
    return PoseTrace(
        backend="mediapipe-lite", mode="video", meta=meta, frames=frames,
    )


def _left_events(result):
    return [e for e in result.events if e.foot == "left"]


def _event_payload(result):
    return [e.to_dict() for e in result.events]


def test_early_apex_full_candidate_ic_after_apex():
    trace = build_early_apex_trace()
    apex_time = float(trace.frames[APEX_FRAME].t_s)
    full = detect_groundplane_contacts(
        trace,
        {"geometry_mode": "shot_local", "state_mode": "semi_markov"},
        record_trace=True,
    )
    left = _left_events(full)
    assert left, "full candidate should emit a left initial contact"
    for ev in left:
        assert ev.method == GROUNDPLANE_METHOD
        assert ev.detector.endswith("contact-groundplane-v1")
        assert ev.time_s > apex_time
        assert ev.earliest_s <= ev.time_s <= ev.latest_s
        assert ev.evidence.get("raw_layer") is True
        apex = ev.evidence.get("apex_frame")
        collision = ev.evidence.get("collision_frame")
        support = ev.evidence.get("support_frame")
        assert collision is not None
        assert support is not None
        assert ev.evidence.get("geometry_kind") in {"line", "rolling_quantile"}
        assert ev.evidence.get("state_change")
        if apex is not None:
            assert ev.time_s > float(trace.frames[int(apex)].t_s)
            assert int(collision) > int(apex)
        assert "global_offset" not in ev.evidence
        assert "timing_offset" not in ev.evidence
        assert "frame_offset" not in ev.evidence

    control = detect_groundplane_contacts(
        trace,
        {
            "geometry_mode": "rolling_quantile_control",
            "state_mode": "phase_apex_control",
        },
        record_trace=True,
    )
    ctrl_left = _left_events(control)
    if ctrl_left:
        nearest = min(ctrl_left, key=lambda e: abs(e.time_s - apex_time))
        assert abs(nearest.time_s - apex_time) <= 2.0 / 30.0 + 1e-9


def test_interpolated_frame_cannot_be_initial_contact():
    trace = build_early_apex_trace()
    frame = LAND_FRAME
    for side in ("left", "right"):
        obs = trace.frames[frame].feet[side]
        for point in (obs.heel, obs.ankle, obs.toe):
            point.x += 600.0
            point.y -= 500.0
    result = detect_groundplane_contacts(trace, record_trace=True)
    pts = trace.frames[frame].source_pts_us
    assert frame in result.trace_header["contact_abstention_frames"]
    assert not any(e.source_pts_us == pts for e in result.events)
    states = [
        row.get("state_after")
        for row in result.frame_trace
        if row.get("kind") == "frame" and row.get("frame") == frame
    ]
    assert states
    assert set(states) <= {"UNCERTAIN"}


def test_vfr_event_pts_match_trace_frames():
    times = vfr_times(2.0)
    times = times[:N_FRAMES] if len(times) >= N_FRAMES else times
    while len(times) < N_FRAMES:
        times.append(times[-1] + 1.0 / 30.0)
    times = times[:N_FRAMES]
    trace = build_early_apex_trace(times)
    result = detect_groundplane_contacts(trace)
    frame_pts = {fr.source_pts_us for fr in trace.frames}
    t_by_pts = {fr.source_pts_us: fr.t_s for fr in trace.frames}
    assert result.events, "VFR fixture should still emit a contact"
    for ev in result.events:
        assert ev.source_pts_us in frame_pts
        assert abs(ev.time_s - t_by_pts[ev.source_pts_us]) < 1e-6


def test_validate_rejects_global_offset():
    with pytest.raises(ValueError):
        validate_groundplane_config({"global_offset": 2})
    with pytest.raises(ValueError):
        validate_groundplane_config({"timing_offset": 1})
    with pytest.raises(ValueError):
        validate_groundplane_config({"frame_offset": 3})
    cfg = default_groundplane_config()
    assert "global_offset" not in cfg
    assert "timing_offset" not in cfg
    assert "frame_offset" not in cfg
    assert cfg["geometry_mode"] == "shot_local"
    assert cfg["state_mode"] == "semi_markov"


def test_frame_states_are_physical_vocabulary():
    trace = build_early_apex_trace()
    result = detect_groundplane_contacts(trace, record_trace=True)
    states = {
        row.get("state_after")
        for row in result.frame_trace
        if row.get("kind") == "frame"
    }
    states.discard(None)
    assert states <= ALLOWED_STATES
    assert "CANDIDATE" not in states
    assert "GROUNDED" not in states
    assert "UNKNOWN" not in {
        row.get("state_after")
        for row in result.frame_trace
        if row.get("kind") == "frame"
    }


def test_deterministic_events_and_pts():
    trace = build_early_apex_trace()
    a = detect_groundplane_contacts(trace)
    b = detect_groundplane_contacts(trace)
    assert [(e.foot, e.source_pts_us, e.time_s, e.earliest_s, e.latest_s)
            for e in a.events] == [
        (e.foot, e.source_pts_us, e.time_s, e.earliest_s, e.latest_s)
        for e in b.events
    ]
    assert _event_payload(a) == _event_payload(b)


def test_tracing_does_not_change_events():
    trace = build_early_apex_trace()
    off = detect_groundplane_contacts(trace, record_trace=False)
    on = detect_groundplane_contacts(trace, record_trace=True)
    assert _event_payload(on) == _event_payload(off)
    assert on.diagnostics == off.diagnostics
    assert off.frame_trace is None and off.trace_header is None
    assert on.frame_trace and on.trace_header
    assert on.trace_header["method"] == GROUNDPLANE_METHOD
