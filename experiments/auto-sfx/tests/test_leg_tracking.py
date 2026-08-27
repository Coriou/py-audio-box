"""Behavioral fixtures for the opt-in full-leg-chain identity prototype."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.leg_tracking import (  # noqa: E402
    MEDIAPIPE_LEG_INDICES,
    track_leg_chains,
)
from auto_sfx.types import (  # noqa: E402
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
    VideoMeta,
)


def _meta(n: int, fps: float = 30.0) -> VideoMeta:
    return VideoMeta(
        path="synthetic-leg-tracking.mp4",
        codec="fixture",
        width=100,
        height=100,
        fps_stream=fps,
        n_frames=n,
        duration_s=n / fps,
        cfr=True,
        vfr=False,
        time_base=1 / 1_000_000,
        audio_present=False,
    )


def _physical_chain(hip_x: float, foot_x: float, z: float) -> dict[str, Point]:
    return {
        "hip": Point(hip_x, 25.0, z, 0.95),
        "knee": Point(0.55 * hip_x + 0.45 * foot_x, 52.0, z * 0.75, 0.94),
        "ankle": Point(foot_x, 79.0, z * 0.50, 0.93),
        "heel": Point(foot_x - 1.5, 84.0, z * 0.48, 0.92),
        "toe": Point(foot_x + 3.0, 87.0, z * 0.46, 0.91),
    }


def _frame(
    index: int,
    left_foot_x: float,
    right_foot_x: float,
    *,
    swapped_labels: bool = False,
    distal_swapped_labels: bool = False,
    depth: float = 18.0,
    mirror_image: bool = False,
    lost: bool = False,
    hidden_sides: tuple[str, ...] = (),
    hidden_joints: tuple[tuple[str, str], ...] = (),
    x_jump: float = 0.0,
) -> PersonFrame:
    if lost:
        return PersonFrame(
            t_s=index / 30.0,
            source_pts_us=round(index / 30.0 * 1_000_000),
            frame_index=index,
            person_id=0,
            score=0.0,
            feet={"left": None, "right": None},
            hips=None,
            shoulders_mid=None,
            landmarks=[],
            lost=True,
        )

    left_hip_x, right_hip_x = ((62.0, 38.0) if mirror_image else (38.0, 62.0))
    physical = {
        "left": _physical_chain(left_hip_x + x_jump, left_foot_x + x_jump, -depth),
        "right": _physical_chain(right_hip_x + x_jump, right_foot_x + x_jump, depth),
    }
    labelled = {"left": {}, "right": {}}
    for observed_side, other_side in (("left", "right"), ("right", "left")):
        for joint in ("hip", "knee", "ankle", "heel", "toe"):
            exchange = swapped_labels or (
                distal_swapped_labels and joint in {"ankle", "heel", "toe"}
            )
            labelled[observed_side][joint] = physical[
                other_side if exchange else observed_side
            ][joint]
    landmarks = [Point(50.0, 50.0, 0.0, 0.01) for _ in range(33)]
    # Shoulders are not part of temporal leg assignment.  Their unsigned depth
    # separation only controls attribution confidence.
    landmarks[11] = Point(left_hip_x + x_jump, 12.0, -depth, 0.95)
    landmarks[12] = Point(right_hip_x + x_jump, 12.0, depth, 0.95)
    for observed_side in ("left", "right"):
        for joint, landmark_index in MEDIAPIPE_LEG_INDICES[observed_side].items():
            point = labelled[observed_side][joint]
            if observed_side in hidden_sides or (observed_side, joint) in hidden_joints:
                point = Point(point.x, point.y, point.z, 0.05)
            landmarks[landmark_index] = point

    def foot(side: str) -> FootObservation:
        indices = MEDIAPIPE_LEG_INDICES[side]
        return FootObservation(
            heel=landmarks[indices["heel"]],
            ankle=landmarks[indices["ankle"]],
            toe=landmarks[indices["toe"]],
        )

    left_hip = landmarks[23]
    right_hip = landmarks[24]
    return PersonFrame(
        t_s=index / 30.0,
        source_pts_us=round(index / 30.0 * 1_000_000),
        frame_index=index,
        person_id=0,
        score=0.93,
        feet={"left": foot("left"), "right": foot("right")},
        hips=Point(
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2,
            min(left_hip.visibility, right_hip.visibility),
        ),
        shoulders_mid=Point(50.0 + x_jump, 12.0, 0.0, 0.95),
        landmarks=landmarks,
        lost=False,
    )


def _trace(frames: list[PersonFrame], backend: str = "mediapipe-lite") -> PoseTrace:
    return PoseTrace(backend=backend, mode="video", meta=_meta(len(frames)), frames=frames)


def _linear_trace(n: int = 9, swaps: set[int] | None = None, **frame_kwargs) -> PoseTrace:
    swaps = swaps or set()
    frames = [
        _frame(
            index,
            left_foot_x=35.0 + 0.8 * index,
            right_foot_x=65.0 - 0.7 * index,
            swapped_labels=index in swaps,
            **frame_kwargs,
        )
        for index in range(n)
    ]
    return _trace(frames)


def _states(result) -> list[str | None]:
    return [frame.assignment_state for frame in result.frames]


def test_uses_full_mediapipe_chain_and_preserves_partial_missingness():
    trace = _trace([
        _frame(0, 35.0, 65.0, hidden_joints=(("left", "toe"),)),
        _frame(1, 36.0, 64.0, hidden_joints=(("left", "toe"),)),
    ])
    result = track_leg_chains(trace)
    left = result.frames[0].tracks["track_0"]
    assert left.chain is not None
    assert left.chain.valid_joints == ("hip", "knee", "ankle", "heel")
    assert left.chain.toe is None
    assert left.observed is True and left.interpolated is False
    assert left.contact_evidence_eligible is False
    assert result.frames[0].tracks["track_1"].contact_evidence_eligible is True


def test_offline_viterbi_repairs_one_frame_complete_leg_label_swap():
    trace = _linear_trace(swaps={4})
    original = copy.deepcopy(trace.to_rows())
    result = track_leg_chains(trace)
    assert _states(result) == [
        "direct", "direct", "direct", "direct", "swapped",
        "direct", "direct", "direct", "direct",
    ]
    corrected = result.frames[4]
    assert corrected.tracks["track_0"].source_label == "right"
    assert corrected.tracks["track_0"].chain.heel.x == pytest.approx(35.0 + 0.8 * 4 - 1.5)
    assert result.summary["backend_label_reassigned_frames"] == 1
    assert result.summary["assignment_state_transitions"] == 2
    assert trace.to_rows() == original  # the opt-in audit never mutates PoseTrace


def test_short_swap_run_is_corrected_without_latching_after_backend_recovers():
    result = track_leg_chains(_linear_trace(n=10, swaps={4, 5}))
    assert _states(result)[3:7] == ["direct", "swapped", "swapped", "direct"]
    assert _states(result)[-1] == "direct"
    assert result.summary["backend_label_reassigned_frames"] == 2
    attr = result.attributions["segment_000"]
    assert (attr.track_0_side, attr.track_1_side) == ("left", "right")


def test_distal_only_exchange_is_represented_as_incoherent_not_averaged_or_eligible():
    frames = [
        _frame(index, 35.0 + index, 65.0 - index,
               distal_swapped_labels=index == 4)
        for index in range(9)
    ]
    result = track_leg_chains(_trace(frames))
    assert result.frames[4].assignment_state == "distal_swapped"
    left = result.frames[4].tracks["track_0"]
    assert left.source_label == "mixed"
    assert left.chain.hip.x == pytest.approx(38.0)
    assert left.chain.heel.x == pytest.approx(35.0 + 4 - 1.5)
    assert left.contact_evidence_eligible is False
    assert result.records[4]["tracks"]["track_0"]["source_labels_by_joint"] == {
        "hip": "left", "knee": "left",
        "ankle": "right", "heel": "right", "toe": "right",
    }
    assert result.summary["distal_reassigned_frames"] == 1
    assert result.summary["proximal_reassigned_frames"] == 0
    assert result.summary["incoherent_partial_chain_frames"] == 1
    assert result.records[4]["whole_chain_status"] == "incoherent_mixed"
    assert result.summary["whole_chain_claim"]["status"] == "rejected"


def test_long_crossed_episode_is_quarantined_instead_of_latching():
    result = track_leg_chains(_linear_trace(n=10, swaps={3, 4, 5, 6}))
    assert _states(result)[3:7] == ["quarantined"] * 4
    assert result.summary["backend_label_reassigned_frames"] == 0
    assert result.summary["quarantined_proximal_frames"] == 4
    assert result.summary["quarantined_distal_frames"] == 4
    for frame in result.frames[3:7]:
        assert all(not track.contact_evidence_eligible for track in frame.tracks.values())
    assert all(result.records[index]["whole_chain_status"] == "component_quarantined"
               for index in range(3, 7))


def test_real_physical_crossing_does_not_trigger_or_latch_label_swap():
    # Feet cross in image x while hip/knee/full-chain identity remains smooth.
    left = [30, 35, 41, 48, 55, 62, 69]
    right = [70, 64, 57, 50, 43, 36, 29]
    trace = _trace([
        _frame(index, left[index], right[index]) for index in range(len(left))
    ])
    result = track_leg_chains(trace)
    assert _states(result) == ["direct"] * len(left)
    heels = [frame.tracks["track_0"].chain.heel.x for frame in result.frames]
    assert heels == pytest.approx([value - 1.5 for value in left])
    assert result.summary["backend_label_reassigned_frames"] == 0


@pytest.mark.parametrize("mirror_image", [False, True])
def test_frontal_rear_image_inversion_is_uncertain_not_hard_image_x(mirror_image):
    result = track_leg_chains(_linear_trace(depth=0.0, mirror_image=mirror_image))
    attr = result.attributions["segment_000"]
    assert attr.orientation_strength == 0.0
    assert attr.posterior_margin == 0.0
    assert (attr.track_0_side, attr.track_1_side) == ("uncertain", "uncertain")
    assert all(
        (frame.tracks["track_0"].anatomical_side,
         frame.tracks["track_1"].anatomical_side) == ("uncertain", "uncertain")
        for frame in result.frames
    )


def test_explicit_cut_resets_local_track_identity_before_large_reentry_jump():
    frames = [
        _frame(index, 34 + index, 66 - index) for index in range(3)
    ] + [
        _frame(index, 15 + index, 85 - index, x_jump=120.0) for index in range(3, 6)
    ]
    result = track_leg_chains(_trace(frames), shot_boundary_frames=[3])
    assert result.summary["segments"] == 2
    assert result.frames[2].segment_id == "segment_000"
    assert result.frames[3].segment_id == "segment_001"
    assert result.frames[3].reset_reason == "shot_boundary"
    assert result.frames[2].tracks["track_0"].track_id == "segment_000.track_0"
    assert result.frames[3].tracks["track_0"].track_id == "segment_001.track_0"
    assert result.frames[3].assignment_state == "direct"


def test_crop_and_multi_frame_occlusion_are_explicit_without_interpolation():
    frames = [
        _frame(0, 35, 65),
        _frame(1, 36, 64),
        _frame(2, 37, 63, lost=True),
        _frame(3, 38, 62, lost=True),
        _frame(4, 39, 61),
        _frame(5, 40, 60, hidden_sides=("right",)),
        _frame(6, 41, 59),
    ]
    result = track_leg_chains(_trace(frames), max_missing_gap_frames=3)
    assert result.summary["segments"] == 1
    assert result.summary["missing_frames"] == 2
    assert result.summary["partial_pair_frames"] == 1
    assert result.summary["interpolated_frames"] == 0
    for index in (2, 3):
        frame = result.frames[index]
        assert frame.segment_id == "segment_000"
        assert frame.assignment_state is None
        assert all(not track.observed and not track.interpolated
                   and not track.contact_evidence_eligible
                   for track in frame.tracks.values())
    cropped = result.frames[5]
    assert cropped.observation_status == "partial_pair"
    assert sum(track.observed for track in cropped.tracks.values()) == 1
    assert all(not track.contact_evidence_eligible
               for track in cropped.tracks.values() if not track.observed)


def test_long_occlusion_resets_on_reentry_instead_of_claiming_continuity():
    frames = [_frame(0, 35, 65), _frame(1, 36, 64)]
    frames += [_frame(index, 0, 0, lost=True) for index in range(2, 6)]
    frames += [_frame(6, 45, 55), _frame(7, 46, 54)]
    result = track_leg_chains(_trace(frames), max_missing_gap_frames=2)
    assert result.summary["segments"] == 2
    assert result.frames[6].reset_reason == "missing_gap"
    assert result.frames[6].tracks["track_0"].track_id == "segment_001.track_0"
    assert all(result.frames[index].segment_id is None for index in range(2, 6))


def test_joint_attribution_never_emits_same_side_under_label_drift():
    # A sustained but bounded label-drift episode weakens the joint posterior.
    result = track_leg_chains(_linear_trace(n=10, swaps={3, 4, 5, 6}))
    for attribution in result.attributions.values():
        pair = (attribution.track_0_side, attribution.track_1_side)
        assert pair in {
            ("left", "right"),
            ("right", "left"),
            ("uncertain", "uncertain"),
        }
        assert pair not in {("left", "left"), ("right", "right")}


def test_unsupported_backend_and_invalid_landmarks_fail_gracefully():
    trace = _linear_trace(n=3)
    trace.backend = "rtmpose-lightweight"
    result = track_leg_chains(trace)
    assert result.summary["backend_supported"] is False
    assert result.summary["segments"] == 0
    assert all(frame.observation_status == "unsupported_backend" for frame in result.frames)
    assert all(not track.contact_evidence_eligible
               for frame in result.frames for track in frame.tracks.values())


def test_records_and_summary_distinguish_mapping_changes_from_measured_id_errors():
    result = track_leg_chains(
        _linear_trace(swaps={4}),
        shot_boundary_frames=[],
        min_visibility=0.4,
        min_valid_joints=4,
        max_missing_gap_frames=2,
        switch_penalty=0.04,
        seed_swap_penalty=0.11,
        missing_penalty=0.21,
        attribution_margin_threshold=0.3,
    )
    assert len(result.records) == len(result.frames)
    row = result.records[4]
    assert row["assignment_state"] == "swapped"
    assert row["source_pts_us"] == result.frames[4].source_pts_us
    assert row["tracks"]["track_0"]["source_label"] == "right"
    assert row["tracks"]["track_0"]["interpolated"] is False
    assert "not measured physical ID errors" in result.summary["identity_note"]
    assert result.summary["uncertain_attribution_segments"] == 0
    assert result.summary["confident_attribution_segments"] == 1
    assert result.summary["same_side_attribution_segments"] == 0
    assert "uncalibrated" in result.summary["anatomy_note"]
    assert result.summary["effective_config"] == {
        "shot_boundary_frames": [],
        "min_visibility": 0.4,
        "min_valid_joints": 4,
        "max_missing_gap_frames": 2,
        "switch_penalty": 0.04,
        "seed_swap_penalty": 0.11,
        "missing_penalty": 0.21,
        "max_prediction_step_bs": 0.35,
        "max_reassignment_run_frames": 2,
        "attribution_margin_threshold": 0.3,
    }
    assert result.summary["segment_boundaries"] == [{
        "segment_id": "segment_000",
        "reset_reason": "trace_start",
        "first_frame": 0,
        "last_frame": 8,
        "first_source_pts_us": 0,
        "last_source_pts_us": 266_667,
        "valid_observation_frames": 9,
    }]


def test_output_is_deterministic_and_configuration_is_validated():
    trace = _linear_trace(swaps={3, 4})
    first = track_leg_chains(trace)
    second = track_leg_chains(trace)
    assert first.records == second.records
    assert first.summary == second.summary
    assert first.attributions == second.attributions
    with pytest.raises(ValueError, match="min_visibility"):
        track_leg_chains(trace, min_visibility=math.nan)
    with pytest.raises(ValueError, match="shot_boundary_frames"):
        track_leg_chains(trace, shot_boundary_frames=[True])
