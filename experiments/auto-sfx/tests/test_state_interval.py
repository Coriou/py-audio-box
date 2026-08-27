"""Synthetic tests for contact-interval-v1 (no real clips, no GT files)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_sfx.contacts import detect_contacts  # noqa: E402
from auto_sfx.state_contacts import detect_state_contacts  # noqa: E402
from auto_sfx.state_interval import (  # noqa: E402
    INTERVAL_METHOD,
    default_interval_config,
    detect_interval_contacts,
    validate_interval_config,
)
from auto_sfx.types import (  # noqa: E402
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
)
from test_contacts import build_walk, make_meta, uniform_times  # noqa: E402


APEX_FRAME = 40
N_FRAMES = 60
GROUND_Y = 200.0
AIR_Y = 160.0
HIP_Y = 100.0


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


def _event_payload(result):
    return [e.to_dict() for e in result.events]


def _identity(result):
    return [(e.foot, e.id) for e in result.events]


def test_early_apex_shifts_past_apex_same_event_count():
    trace = build_early_apex_trace()
    apex_time = float(trace.frames[APEX_FRAME].t_s)
    baseline = detect_state_contacts(trace)
    got = detect_interval_contacts(
        trace, {"geometry_mode": "shot_local"}, record_trace=True,
    )
    assert len(got.events) == len(baseline.events)
    assert _identity(got) == _identity(baseline)
    assert got.events, "interval should keep contact-state-v1 existence"
    left = [e for e in got.events if e.foot == "left"]
    assert left
    for ev in left:
        assert ev.method == INTERVAL_METHOD
        assert ev.detector.endswith(INTERVAL_METHOD)
        assert ev.time_s > apex_time
        assert ev.earliest_s <= ev.time_s <= ev.latest_s
        assert ev.evidence.get("apex_frame") == APEX_FRAME
        assert ev.evidence.get("support_frame") is not None
        assert int(ev.evidence["support_frame"]) > APEX_FRAME
        assert int(ev.evidence["shift_frames"]) == (
            int(ev.evidence["support_frame"]) - APEX_FRAME
        )
        assert ev.evidence.get("geometry_kind") in {"line", "rolling_quantile"}
        assert "global_offset" not in ev.evidence
        assert "timing_offset" not in ev.evidence
        assert "frame_offset" not in ev.evidence
    base_left = [e for e in baseline.events if e.foot == "left"]
    assert base_left
    assert abs(base_left[0].time_s - round(apex_time, 6)) <= 1e-9


def test_existence_identity_on_walk():
    trace = build_walk(uniform_times(3.0))
    baseline = detect_contacts(trace, "contact-state-v1")
    got = detect_interval_contacts(trace)
    assert len(got.events) == len(baseline.events)
    assert [e.foot for e in got.events] == [e.foot for e in baseline.events]
    assert [e.id for e in got.events] == [e.id for e in baseline.events]


def test_max_forward_frames_zero_leaves_times_unchanged():
    trace = build_early_apex_trace()
    baseline = detect_state_contacts(trace)
    got = detect_interval_contacts(trace, {"max_forward_frames": 0})
    assert len(got.events) == len(baseline.events)
    for ev, base in zip(got.events, baseline.events):
        assert ev.foot == base.foot
        assert ev.time_s == base.time_s
        assert ev.earliest_s == base.earliest_s
        assert ev.latest_s == base.latest_s
        assert ev.source_pts_us == base.source_pts_us
        assert ev.evidence.get("shift_frames") == 0
        assert ev.evidence.get("apex_frame") is not None
        assert ev.evidence.get("support_frame") is None


def test_geometry_abstain_or_missing_height_keeps_original_time():
    trace = build_early_apex_trace()
    baseline = detect_state_contacts(trace)
    abstain = detect_interval_contacts(
        trace, {"geometry_mode": "shot_local", "min_support_points": 10_000},
        record_trace=True,
    )
    assert len(abstain.events) == len(baseline.events)
    for ev, base in zip(abstain.events, baseline.events):
        assert ev.time_s == base.time_s
        assert ev.source_pts_us == base.source_pts_us
        assert ev.evidence.get("shift_frames") == 0
        assert ev.evidence.get("geometry_kind") == "abstain"
        assert ev.evidence.get("reason")
        assert ev.evidence.get("support_frame") is None

    # Frame 41 is still above the support band; a 1-frame window has no
    # finite in-band height, so times stay at the original apex.
    no_cross = detect_interval_contacts(trace, {"max_forward_frames": 1})
    assert len(no_cross.events) == len(baseline.events)
    for ev, base in zip(no_cross.events, baseline.events):
        assert ev.time_s == base.time_s
        assert ev.source_pts_us == base.source_pts_us
        assert ev.evidence.get("shift_frames") == 0
        assert ev.evidence.get("support_frame") is None
        assert ev.evidence.get("reason") == "no_support_crossing"


def test_validate_rejects_global_offset():
    with pytest.raises(ValueError):
        validate_interval_config({"global_offset": 2})
    with pytest.raises(ValueError):
        validate_interval_config({"timing_offset": 1})
    with pytest.raises(ValueError):
        validate_interval_config({"frame_offset": 3})
    cfg = default_interval_config()
    assert "global_offset" not in cfg
    assert "timing_offset" not in cfg
    assert "frame_offset" not in cfg
    assert cfg["geometry_mode"] == "shot_local"
    assert cfg["max_forward_frames"] == 4
    assert cfg["support_height_frac"] == 0.09


def test_interpolated_frame_cannot_be_the_support_crossing():
    trace = build_walk(uniform_times(3.0))
    frame = 50
    pts = trace.frames[frame].source_pts_us
    for side in ("left", "right"):
        obs = trace.frames[frame].feet[side]
        for point in (obs.heel, obs.ankle, obs.toe):
            point.x += 600.0
            point.y -= 500.0
    result = detect_interval_contacts(trace, record_trace=True)
    abstained = set(result.trace_header["contact_abstention_frames"])
    assert frame in abstained
    assert result.events, "other contacts should survive a one-frame quarantine"
    for ev in result.events:
        assert ev.evidence.get("support_frame") != frame
        assert ev.source_pts_us != pts
        support = ev.evidence.get("support_frame")
        if support is not None:
            assert int(support) not in abstained

    early = build_early_apex_trace()
    cross = APEX_FRAME + 2
    for side in ("left", "right"):
        obs = early.frames[cross].feet[side]
        for point in (obs.heel, obs.ankle, obs.toe):
            point.x += 600.0
            point.y -= 500.0
    jumped = detect_interval_contacts(early, record_trace=True)
    assert cross in jumped.trace_header["contact_abstention_frames"]
    for ev in jumped.events:
        assert ev.evidence.get("support_frame") != cross
        if ev.evidence.get("support_frame") is not None:
            assert int(ev.evidence["support_frame"]) > cross


def test_deterministic_and_tracing_neutral():
    trace = build_early_apex_trace()
    a = detect_interval_contacts(trace)
    b = detect_interval_contacts(trace)
    assert [(e.foot, e.source_pts_us, e.time_s, e.earliest_s, e.latest_s)
            for e in a.events] == [
        (e.foot, e.source_pts_us, e.time_s, e.earliest_s, e.latest_s)
        for e in b.events
    ]
    assert _event_payload(a) == _event_payload(b)
    off = detect_interval_contacts(trace, record_trace=False)
    on = detect_interval_contacts(trace, record_trace=True)
    assert _event_payload(on) == _event_payload(off)
    assert on.diagnostics == off.diagnostics
    assert off.frame_trace is None and off.trace_header is None
    assert on.frame_trace and on.trace_header
    assert on.trace_header["method"] == INTERVAL_METHOD
    states = {
        row.get("state_after")
        for row in on.frame_trace
        if row.get("kind") == "frame"
    }
    assert "CANDIDATE" not in states
