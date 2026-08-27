"""Synthetic behavioral tests for the opt-in sparse ROI candidate analyzer."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_sfx import sparse_roi_refiner as sparse  # noqa: E402


HEIGHT = 96
WIDTH = 128
_BACKGROUND = cv2.GaussianBlur(
    np.random.default_rng(101).integers(0, 256, (HEIGHT, WIDTH), dtype=np.uint8),
    (3, 3),
    0,
)
_HEEL_TEXTURE = np.random.default_rng(201).integers(0, 256, (13, 13), dtype=np.uint8)
_TOE_TEXTURE = np.random.default_rng(202).integers(0, 256, (13, 13), dtype=np.uint8)


def _render(background: np.ndarray, heel: tuple[float, float], toe: tuple[float, float]):
    frame = background.copy()
    for (x, y), texture in ((heel, _HEEL_TEXTURE), (toe, _TOE_TEXTURE)):
        cx, cy = int(round(x)), int(round(y))
        frame[cy - 6 : cy + 7, cx - 6 : cx + 7] = texture
    return frame


def _actor_box(heel: tuple[float, float], toe: tuple[float, float]):
    return (
        min(heel[0], toe[0]) - 19,
        min(heel[1], toe[1]) - 18,
        max(heel[0], toe[0]) + 19,
        max(heel[1], toe[1]) + 11,
    )


def _sequence(
    y_positions,
    *,
    pts_us=None,
    transforms: list[np.ndarray] | None = None,
    backgrounds: list[np.ndarray] | None = None,
):
    count = len(y_positions)
    pts = list(pts_us or [index * 40_000 for index in range(count)])
    transforms = transforms or [np.asarray([[1, 0, 0], [0, 1, 0]], np.float32)] * count
    backgrounds = backgrounds or [_BACKGROUND] * count
    frames, heels, toes, boxes = [], [], [], []
    for y, matrix, background in zip(y_positions, transforms, backgrounds, strict=True):
        world_heel = (46.0, float(y))
        world_toe = (60.0, float(y))
        world = _render(background, world_heel, world_toe)
        frame = cv2.warpAffine(world, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)

        def transform(point):
            projected = matrix @ np.asarray([point[0], point[1], 1.0])
            return float(projected[0]), float(projected[1])

        heel = transform(world_heel)
        toe = transform(world_toe)
        raw_box = _actor_box(world_heel, world_toe)
        corners = [
            transform((raw_box[0], raw_box[1])),
            transform((raw_box[2], raw_box[1])),
            transform((raw_box[0], raw_box[3])),
            transform((raw_box[2], raw_box[3])),
        ]
        box = (
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )
        frames.append(frame)
        heels.append(heel)
        toes.append(toe)
        boxes.append(box)
    return frames, pts, heels, toes, boxes


def _analyze(sequence, **kwargs):
    frames, pts, heels, toes, boxes = sequence
    return sparse.analyze_candidate_window(
        frames, pts, heels, toes, actor_boxes=boxes, **kwargs
    )


def test_static_planted_foot_is_not_a_contact_change():
    result = _analyze(_sequence([30, 30, 30, 30, 30, 30]))
    assert result["status"] == "abstain"
    assert result["reason"] == "no_contact_change"
    assert result["score"] == 0.0
    assert result["suggested_frame"] is None
    assert all(pair["staticness"] == 1.0 for pair in result["pair_evidence"])


def test_downward_moving_foot_that_settles_suggests_first_static_frame():
    result = _analyze(_sequence([22, 25, 28, 28, 28, 28]))
    assert result["status"] == "suggested"
    assert result["method"] == "sparse-stabilized-roi-v1"
    assert result["suggested_frame"] == 3
    assert result["suggested_pts_us"] == 120_000
    assert result["suggested_interval_pts_us"] == [100_000, 140_000]
    assert result["score"] >= 0.58
    assert result["camera_motion_is_contact_evidence"] is False
    assert result["pair_evidence"][1]["relative_vertical_velocity_px_s"] > 18
    assert abs(result["pair_evidence"][2]["relative_vertical_velocity_px_s"]) < 1


@pytest.mark.parametrize("motion", ["pan", "affine"])
def test_camera_motion_alone_never_scores_contact(motion):
    transforms = []
    for index in range(6):
        if motion == "pan":
            matrix = np.asarray([[1, 0, 1.4 * index], [0, 1, 0.6 * index]], np.float32)
        else:
            angle = np.deg2rad(0.45 * index)
            cosine, sine = np.cos(angle), np.sin(angle)
            matrix = np.asarray(
                [
                    [cosine + 0.002 * index, -sine + 0.001 * index, 1.1 * index],
                    [sine, cosine, 0.45 * index],
                ],
                np.float32,
            )
        transforms.append(matrix)
    result = _analyze(_sequence([30] * 6, transforms=transforms))
    assert result["status"] == "abstain"
    assert result["reason"] == "no_contact_change"
    assert result["score"] == 0.0
    assert result["camera_motion_is_contact_evidence"] is False
    speeds = [pair["relative_speed_px_s"] for pair in result["pair_evidence"]]
    assert float(np.median(speeds)) < 3.0


def test_low_texture_fails_closed():
    flat = np.full((HEIGHT, WIDTH), 127, dtype=np.uint8)
    sequence = _sequence([30] * 6, backgrounds=[flat] * 6)
    frames, pts, heels, toes, boxes = sequence
    frames = [np.full_like(frame, 127) for frame in frames]
    result = sparse.analyze_candidate_window(
        frames, pts, heels, toes, actor_boxes=boxes
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "weak_background_compensation"
    assert result["suggested_pts_us"] is None


def test_textured_background_with_textureless_foot_fails_closed():
    frames, pts, heels, toes, boxes = _sequence([30] * 6)
    textureless = []
    for frame, heel, toe in zip(frames, heels, toes, strict=True):
        altered = frame.copy()
        cv2.circle(altered, tuple(map(round, heel)), 10, 127, -1)
        cv2.circle(altered, tuple(map(round, toe)), 10, 127, -1)
        textureless.append(altered)
    result = sparse.analyze_candidate_window(
        textureless, pts, heels, toes, actor_boxes=boxes
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "weak_foot_tracks"


def test_missing_or_narrow_actor_exclusion_fails_before_tracking():
    frames, pts, heels, toes, boxes = _sequence([30] * 6)
    missing = sparse.analyze_candidate_window(frames, pts, heels, toes)
    assert missing["reason"] == "missing_actor_exclusion"
    narrow_boxes = [(44, 27, 62, 33)] * 6
    narrow = sparse.analyze_candidate_window(
        frames, pts, heels, toes, actor_boxes=narrow_boxes
    )
    assert narrow["status"] == "abstain"
    assert narrow["reason"] == "geometry_failure"
    assert narrow["pair_evidence"] == []


def test_crop_or_ground_roi_invalidity_fails_closed():
    frames, pts, _, _, _ = _sequence([30] * 6)
    heels = [(46, 90)] * 6
    toes = [(60, 90)] * 6
    boxes = [(25, 60, 80, 95)] * 6
    result = sparse.analyze_candidate_window(
        frames, pts, heels, toes, actor_boxes=boxes
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "geometry_failure"


def test_edit_cut_counterfactual_cannot_produce_a_suggestion():
    other = cv2.GaussianBlur(
        np.random.default_rng(999).integers(0, 256, (HEIGHT, WIDTH), dtype=np.uint8),
        (3, 3),
        0,
    )
    backgrounds = [_BACKGROUND, _BACKGROUND, _BACKGROUND, other, other, other]
    result = _analyze(_sequence([26, 28, 30, 30, 30, 30], backgrounds=backgrounds))
    assert result["status"] == "abstain"
    assert result["reason"] in {"edit_cut", "weak_background_compensation"}
    assert result["score"] == 0.0
    assert result["camera_motion_is_contact_evidence"] is False


def test_vfr_pts_drive_velocity_and_reported_uncertainty_interval():
    pts = [0, 30_000, 80_000, 115_000, 170_000, 205_000]
    result = _analyze(_sequence([20, 23, 26, 26, 26, 26], pts_us=pts))
    assert result["status"] == "suggested"
    assert result["suggested_frame"] == 3
    assert result["suggested_pts_us"] == 115_000
    assert result["suggested_interval_pts_us"] == [97_500, 142_500]
    assert result["pair_evidence"][0]["dt_s"] == 0.03
    assert result["pair_evidence"][1]["dt_s"] == 0.05


def test_non_monotonic_pts_abstain_before_pixel_tracking(monkeypatch):
    frames, _, heels, toes, boxes = _sequence([30] * 6)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pixel tracking must not run")

    monkeypatch.setattr(sparse, "_pair_evidence", forbidden)
    result = sparse.analyze_candidate_window(
        frames, [0, 40_000, 80_000, 80_000, 120_000, 160_000], heels, toes, actor_boxes=boxes
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "non_monotonic_timestamps"


def test_supplied_window_is_consumed_once_and_result_is_deterministic():
    frames, pts, heels, toes, boxes = _sequence([22, 25, 28, 28, 28, 28])

    class OneShotFrames:
        def __init__(self, values):
            self.values = values
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("window consumed more than once")
            return iter(self.values)

    supplied = OneShotFrames(frames)
    first = sparse.analyze_candidate_window(
        supplied, pts, heels, toes, actor_boxes=boxes
    )
    second = sparse.analyze_candidate_window(
        frames, pts, heels, toes, actor_boxes=boxes
    )
    assert supplied.iterations == 1
    assert first == second


def test_oversized_iterable_stops_at_bounded_lookahead():
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    class EndlessFrames:
        def __init__(self):
            self.yielded = 0

        def __iter__(self):
            while True:
                self.yielded += 1
                yield frame

    supplied = EndlessFrames()
    result = sparse.analyze_candidate_window(supplied, [], [], [], actor_boxes=[])
    assert result["status"] == "abstain"
    assert result["reason"] == "window_budget_exceeded"
    assert supplied.yielded == sparse.sparse_roi_config()["max_window_frames"] + 1


def test_video_wrapper_requires_verified_frame_pts_before_decode(monkeypatch):
    frames, pts, heels, toes, boxes = _sequence([22, 25, 28, 28, 28, 28])
    calls = []

    def decode_once(_path, mapping, **_kwargs):
        calls.append(mapping)
        return frames

    monkeypatch.setattr(sparse, "decode_verified_candidate_window", decode_once)
    missing = sparse.analyze_video_candidate_window(
        "unused.mp4", 10, 15, pts, heels, toes, actor_boxes=boxes
    )
    assert missing["reason"] == "unverified_video_timestamps"
    assert calls == []
    wrong = [(10 + index, value + (1 if index == 2 else 0)) for index, value in enumerate(pts)]
    mismatched = sparse.analyze_video_candidate_window(
        "unused.mp4",
        10,
        15,
        pts,
        heels,
        toes,
        actor_boxes=boxes,
        verified_frame_pts=wrong,
    )
    assert mismatched["reason"] == "unverified_video_timestamps"
    assert calls == []
    verified = list(zip(range(10, 16), pts))
    result = sparse.analyze_video_candidate_window(
        "unused.mp4",
        10,
        15,
        pts,
        heels,
        toes,
        actor_boxes=boxes,
        verified_frame_pts=verified,
    )
    assert calls == [verified]
    assert result["status"] == "suggested"
    assert result["suggested_frame"] == 13


def test_sparse_module_has_no_dense_flow_call():
    source = inspect.getsource(sparse)
    forbidden = "calcOpticalFlow" + "Farneback"
    assert forbidden not in source
    assert "calcOpticalFlowPyrLK" in source


def test_verified_decoder_mapping_validation_fails_before_pyav_import():
    with pytest.raises(ValueError, match="contiguous and monotonic"):
        sparse.decode_verified_candidate_window(
            "unused.mp4", [(10, 0), (12, 40_000)]
        )
