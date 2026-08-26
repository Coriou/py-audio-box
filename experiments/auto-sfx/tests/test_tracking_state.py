"""Phase-2B regressions for identity continuity and contact-state timing."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_sfx.contacts import detect_contacts  # noqa: E402
from auto_sfx.tracking import track_feet  # noqa: E402
from auto_sfx.types import DiagReason  # noqa: E402

# Reuse the established analytic walker; importing a test helper keeps the
# regression's gait geometry identical to the hybrid-v0 tests.
from test_contacts import build_walk, uniform_times  # noqa: E402


def _centre(obs):
    pts = [obs.heel, obs.ankle, obs.toe]
    return np.asarray([
        np.mean([p.x for p in pts]), np.mean([p.y for p in pts])
    ])


def test_one_frame_left_right_swap_is_corrected_without_confidence_drop():
    original = build_walk(uniform_times(3.0))
    damaged = copy.deepcopy(original)
    frame = 50
    damaged.frames[frame].feet["left"], damaged.frames[frame].feet["right"] = (
        damaged.frames[frame].feet["right"], damaged.frames[frame].feet["left"]
    )
    assert damaged.frames[frame].feet["left"].heel.visibility == 0.95

    # This small synthetic walker has a smaller inter-foot separation than
    # real walk1; lower the physical bound so its injected exchange has the
    # same "implausible direct / plausible crossed" relationship.
    result = track_feet(damaged, policy="swap", jump_speed_bs_s=8.0)

    for side in ("left", "right"):
        got = _centre(result.trace.frames[frame].feet[side])
        want = _centre(original.frames[frame].feet[side])
        assert np.linalg.norm(got - want) < 1e-9
    rows = [r for r in result.records if r.get("frame") == frame]
    assert [r["decision"] for r in rows] == ["swap"]
    assert result.summary["swap_frames"] == 1
    assert result.summary["high_confidence_outliers"] == 1


def test_implausible_one_frame_jump_is_quarantined_or_interpolated():
    original = build_walk(uniform_times(3.0))
    damaged = copy.deepcopy(original)
    frame = 50
    for side in ("left", "right"):
        obs = damaged.frames[frame].feet[side]
        for point in (obs.heel, obs.ankle, obs.toe):
            point.x += 500.0
            point.y -= 400.0

    quarantine = track_feet(damaged, policy="quarantine")
    assert quarantine.trace.frames[frame].feet == {"left": None, "right": None}
    assert quarantine.summary["quarantined_frames"] == 1
    assert quarantine.summary["high_confidence_outliers"] == 1

    interpolated = track_feet(damaged, policy="interpolate", max_interp_gap=1)
    assert interpolated.summary["quarantined_frames"] == 1
    assert interpolated.summary["interpolated_foot_frames"] == 2
    for side in ("left", "right"):
        got = _centre(interpolated.trace.frames[frame].feet[side])
        before = _centre(original.frames[frame - 1].feet[side])
        after = _centre(original.frames[frame + 1].feet[side])
        assert np.linalg.norm(got - 0.5 * (before + after)) < 1e-9


def test_contact_state_v1_survives_swap_and_has_no_candidate_state():
    trace = build_walk(uniform_times(5.4))
    frame = 50
    trace.frames[frame].feet["left"], trace.frames[frame].feet["right"] = (
        trace.frames[frame].feet["right"], trace.frames[frame].feet["left"]
    )
    result = detect_contacts(
        trace, "contact-state-v1",
        config={"track_jump_speed_bs_s": 8.0},
        record_trace=True,
    )

    assert result.events
    assert all(e.method == "contact-state-v1" for e in result.events)
    assert all(e.detector.endswith("contact-state-v1") for e in result.events)
    assert all(e.earliest_s <= e.time_s <= e.latest_s for e in result.events)
    states = {
        row.get("state_after") for row in result.frame_trace
        if row.get("kind") == "frame"
    }
    assert states <= {"SWING", "UNCERTAIN", "GROUNDED"}
    assert "CANDIDATE" not in states
    assert result.trace_header["tracking"]["swap_frames"] == 1


def test_legitimate_fast_crossings_do_not_latch_temporal_identity():
    trace = build_walk(uniform_times(5.4))
    result = track_feet(trace, policy="swap")
    assert result.summary["swap_frames"] == 0
    assert result.summary["abstained_long_swap_frames"] == 0


def test_contact_state_threshold_is_bounded_after_velocity_spike():
    trace = build_walk(uniform_times(5.4))
    frame = 47
    for side in ("left", "right"):
        obs = trace.frames[frame].feet[side]
        for point in (obs.heel, obs.ankle, obs.toe):
            point.x += 500.0
    result = detect_contacts(
        trace, "contact-state-v1",
        config={"track_policy": "raw"},
        record_trace=True,
    )
    rows = [
        r for r in result.frame_trace
        if r.get("kind") == "frame" and r.get("frame", 0) > frame + 20
    ]
    thresholds = [
        r[k] for r in rows for k in ("heel_threshold_bs_s", "toe_threshold_bs_s")
        if r.get(k) is not None
    ]
    assert thresholds
    assert max(thresholds) <= 0.85
    # The bounded local statistic recovers and still sees later contacts.
    assert any(e.time_s > 3.0 for e in result.events)


def test_interpolated_discontinuity_frame_cannot_be_contact_evidence():
    trace = build_walk(uniform_times(3.0))
    frame = 19  # analytic left contact neighbourhood
    for side in ("left", "right"):
        obs = trace.frames[frame].feet[side]
        for point in (obs.heel, obs.ankle, obs.toe):
            point.x += 600.0
            point.y -= 500.0
    result = detect_contacts(trace, "contact-state-v1", record_trace=True)

    assert frame in result.trace_header["contact_abstention_frames"]
    assert not any(e.source_pts_us == trace.frames[frame].source_pts_us
                   for e in result.events)
    assert any(r.get("frame") == frame and r.get("decision") == "interpolate"
               for r in result.frame_trace)


def test_contact_state_is_invariant_to_smooth_tracking_camera_translation():
    original = build_walk(uniform_times(5.4))
    translated = copy.deepcopy(original)
    for frame in translated.frames:
        dx = 6.0 * frame.frame_index
        frame.hips.x += dx
        for side in ("left", "right"):
            obs = frame.feet[side]
            for point in (obs.heel, obs.ankle, obs.toe):
                point.x += dx

    baseline = detect_contacts(original, "contact-state-v1")
    moving_camera = detect_contacts(translated, "contact-state-v1")

    assert [(e.foot, e.source_pts_us) for e in moving_camera.events] == [
        (e.foot, e.source_pts_us) for e in baseline.events
    ]


def test_unobservable_phase_is_an_explicit_abstention():
    trace = build_walk(uniform_times(2.0))
    seed = copy.deepcopy(trace.frames[0])
    for frame in trace.frames:
        frame.hips = copy.deepcopy(seed.hips)
        frame.feet = copy.deepcopy(seed.feet)

    result = detect_contacts(trace, "contact-state-v1", record_trace=True)

    assert result.events == []
    assert any(
        d["code"] == DiagReason.AMBIGUOUS_SUPPORT_PHASE
        and "abstained" in d["detail"]
        for d in result.diagnostics
    )
