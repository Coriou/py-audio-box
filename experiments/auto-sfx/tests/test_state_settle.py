"""Synthetic tests for contact-settle-v1 (no real clips, no GT files)."""

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
from auto_sfx.state_settle import (  # noqa: E402
    SETTLE_METHOD,
    default_settle_config,
    detect_settle_contacts,
    validate_settle_config,
)
from auto_sfx.types import (  # noqa: E402
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
)
from test_contacts import (  # noqa: E402
    build_walk,
    make_meta,
    uniform_times,
    vfr_times,
)


APEX_FRAME = 40
SETTLE_FRAME = 42
N_FRAMES = 60
GROUND_Y = 200.0
HIP_Y = 100.0
# Near-ground at the apex (in-band vs support_height_frac=0.09 ≈ 9 px at
# hip-ground scale ~100) with remaining +y motion that stops at frame 42.
PRE_APEX_Y = 191.0
APEX_Y = 195.0
PLANT_Y = 199.0


def _pt(x: float, y: float, vis: float = 0.95) -> Point:
    return Point(x=float(x), y=float(y), z=0.0, visibility=float(vis))


def _foot(ankle_x: float, ankle_y: float) -> FootObservation:
    return FootObservation(
        heel=_pt(ankle_x - 4.0, ankle_y),
        ankle=_pt(ankle_x, ankle_y),
        toe=_pt(ankle_x + 8.0, ankle_y),
    )


def _left_x(i: int) -> float:
    hip_x = 100.0 + 1.2 * i
    if i <= APEX_FRAME:
        rel = 20.0 - 2.0 * (APEX_FRAME - i)
        return hip_x + rel
    return (100.0 + 1.2 * APEX_FRAME) + 20.0


def _left_y(i: int, *, planted: bool) -> float:
    if planted:
        return GROUND_Y
    if i < APEX_FRAME:
        return PRE_APEX_Y
    if i == APEX_FRAME:
        return APEX_Y
    return PLANT_Y


def build_inband_trace(
    times: list[float] | None = None, *, planted: bool = False,
) -> PoseTrace:
    if times is None:
        times = [k / 30.0 for k in range(N_FRAMES)]
    n = len(times)
    frames = []
    for i, t in enumerate(times):
        hip_x = 100.0 + 1.2 * i
        lx, ly = _left_x(i), _left_y(i, planted=planted)
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


def test_height_in_band_late_settle_shifts_two_frames():
    trace = build_inband_trace()
    apex_time = float(trace.frames[APEX_FRAME].t_s)
    settle_time = round(float(trace.frames[SETTLE_FRAME].t_s), 6)
    baseline = detect_state_contacts(trace)
    got = detect_settle_contacts(trace, record_trace=True)
    assert len(got.events) == len(baseline.events)
    assert _identity(got) == _identity(baseline)
    assert got.events, "settle should keep contact-state-v1 existence"
    left = [e for e in got.events if e.foot == "left"]
    base_left = [e for e in baseline.events if e.foot == "left"]
    assert left and base_left
    for ev, base in zip(left, base_left):
        assert ev.method == SETTLE_METHOD
        assert ev.detector.endswith(SETTLE_METHOD)
        assert abs(base.time_s - round(apex_time, 6)) <= 1e-9
        heel_h = base.evidence.get("heel_support_height_bs")
        toe_h = base.evidence.get("toe_support_height_bs")
        assert heel_h is not None and heel_h <= 0.09
        assert toe_h is not None and toe_h <= 0.09
        assert ev.time_s > apex_time
        assert ev.time_s == settle_time
        assert ev.earliest_s == base.earliest_s
        assert ev.latest_s == max(float(base.latest_s), ev.time_s)
        assert ev.earliest_s <= ev.time_s <= ev.latest_s
        assert ev.evidence.get("apex_frame") == APEX_FRAME
        assert ev.evidence.get("settle_frame") == SETTLE_FRAME
        assert ev.evidence.get("shift_frames") == 2
        assert ev.evidence.get("settle_signal") == "downward_vy"
        assert ev.evidence.get("reason") is None
        assert "global_offset" not in ev.evidence
        assert "timing_offset" not in ev.evidence
        assert "frame_offset" not in ev.evidence


def test_existence_identity_on_walk():
    trace = build_walk(uniform_times(3.0))
    baseline = detect_contacts(trace, "contact-state-v1")
    got = detect_settle_contacts(trace)
    assert len(got.events) == len(baseline.events)
    assert [e.foot for e in got.events] == [e.foot for e in baseline.events]
    assert [e.id for e in got.events] == [e.id for e in baseline.events]


def test_max_forward_frames_zero_leaves_times_unchanged():
    trace = build_inband_trace()
    baseline = detect_state_contacts(trace)
    got = detect_settle_contacts(trace, {"max_forward_frames": 0})
    assert len(got.events) == len(baseline.events)
    for ev, base in zip(got.events, baseline.events):
        assert ev.foot == base.foot
        assert ev.time_s == base.time_s
        assert ev.earliest_s == base.earliest_s
        assert ev.latest_s == base.latest_s
        assert ev.source_pts_us == base.source_pts_us
        assert ev.evidence.get("shift_frames") == 0
        assert ev.evidence.get("apex_frame") is not None
        assert ev.evidence.get("settle_frame") is None


def test_already_settled_at_apex_does_not_shift():
    trace = build_inband_trace(planted=True)
    baseline = detect_state_contacts(trace)
    got = detect_settle_contacts(trace, record_trace=True)
    assert len(got.events) == len(baseline.events)
    assert got.events, "planted in-band apex should still exist"
    for ev, base in zip(got.events, baseline.events):
        assert ev.time_s == base.time_s
        assert ev.source_pts_us == base.source_pts_us
        assert ev.earliest_s == base.earliest_s
        assert ev.latest_s == base.latest_s
        assert ev.evidence.get("shift_frames") == 0
        assert ev.evidence.get("settle_frame") is None
        assert ev.evidence.get("reason") == "already_settled"
        assert ev.evidence.get("apex_frame") is not None
        # Later planted frames are also settled; they must not pull time forward.
        assert ev.evidence.get("settle_frame") != (
            None if ev.evidence.get("apex_frame") is None
            else int(ev.evidence["apex_frame"]) + 1
        )


def test_validate_rejects_offset_keys():
    with pytest.raises(ValueError):
        validate_settle_config({"global_offset": 2})
    with pytest.raises(ValueError):
        validate_settle_config({"timing_offset": 1})
    with pytest.raises(ValueError):
        validate_settle_config({"frame_offset": 3})
    with pytest.raises(ValueError):
        validate_settle_config({"geometry_mode": "shot_local"})
    cfg = default_settle_config()
    assert "global_offset" not in cfg
    assert "timing_offset" not in cfg
    assert "frame_offset" not in cfg
    assert "geometry_mode" not in cfg
    assert "min_support_points" not in cfg
    assert "max_support_residual_rms_bs" not in cfg
    assert cfg["max_forward_frames"] == 4
    assert cfg["settle_vy_bs_s"] == 0.08
    assert cfg["settle_speed_bs_s"] == 0.16
    assert cfg["settle_signal"] == "downward_vy"
    assert cfg["support_height_frac"] == 0.09
    assert cfg["phase_min_velocity_bs_s"] == 0.08


def test_interpolated_frame_cannot_be_settle_frame():
    trace = build_walk(uniform_times(3.0))
    frame = 50
    pts = trace.frames[frame].source_pts_us
    for side in ("left", "right"):
        obs = trace.frames[frame].feet[side]
        for point in (obs.heel, obs.ankle, obs.toe):
            point.x += 600.0
            point.y -= 500.0
    result = detect_settle_contacts(trace, record_trace=True)
    abstained = set(result.trace_header["contact_abstention_frames"])
    assert frame in abstained
    assert result.events, "other contacts should survive a one-frame quarantine"
    for ev in result.events:
        assert ev.evidence.get("settle_frame") != frame
        assert ev.source_pts_us != pts
        settle = ev.evidence.get("settle_frame")
        if settle is not None:
            assert int(settle) not in abstained

    late = build_inband_trace()
    cross = SETTLE_FRAME
    for side in ("left", "right"):
        obs = late.frames[cross].feet[side]
        for point in (obs.heel, obs.ankle, obs.toe):
            point.x += 600.0
            point.y -= 500.0
    jumped = detect_settle_contacts(late, record_trace=True)
    assert cross in jumped.trace_header["contact_abstention_frames"]
    for ev in jumped.events:
        assert ev.evidence.get("settle_frame") != cross
        if ev.evidence.get("settle_frame") is not None:
            assert int(ev.evidence["settle_frame"]) > cross


def test_deterministic_and_tracing_neutral():
    trace = build_inband_trace()
    a = detect_settle_contacts(trace)
    b = detect_settle_contacts(trace)
    assert [(e.foot, e.source_pts_us, e.time_s, e.earliest_s, e.latest_s)
            for e in a.events] == [
        (e.foot, e.source_pts_us, e.time_s, e.earliest_s, e.latest_s)
        for e in b.events
    ]
    assert _event_payload(a) == _event_payload(b)
    off = detect_settle_contacts(trace, record_trace=False)
    on = detect_settle_contacts(trace, record_trace=True)
    assert _event_payload(on) == _event_payload(off)
    assert on.diagnostics == off.diagnostics
    assert off.frame_trace is None and off.trace_header is None
    assert on.frame_trace and on.trace_header
    assert on.trace_header["method"] == SETTLE_METHOD
    states = {
        row.get("state_after")
        for row in on.frame_trace
        if row.get("kind") == "frame"
    }
    assert "CANDIDATE" not in states


def test_vfr_time_s_is_existing_source_frame_timestamp():
    times = vfr_times(2.0)
    times = times[:N_FRAMES] if len(times) >= N_FRAMES else times
    while len(times) < N_FRAMES:
        times.append(times[-1] + 1.0 / 30.0)
    times = times[:N_FRAMES]
    trace = build_inband_trace(times)
    result = detect_settle_contacts(trace)
    frame_pts = {fr.source_pts_us for fr in trace.frames}
    t_by_pts = {fr.source_pts_us: fr.t_s for fr in trace.frames}
    assert result.events, "VFR fixture should still emit a contact"
    for ev in result.events:
        assert ev.source_pts_us in frame_pts
        assert abs(ev.time_s - t_by_pts[ev.source_pts_us]) < 1e-6
        assert any(abs(ev.time_s - fr.t_s) < 1e-6 for fr in trace.frames)
