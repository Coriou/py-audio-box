"""Driver-level contracts for the opened sparse-pixel diagnostic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_sfx.sparse_roi_refiner import sparse_roi_config
from auto_sfx.types import PersonFrame, Point


SCRIPT = ROOT / "scripts" / "run_phase2c_batch1_pixels.py"
SPEC = importlib.util.spec_from_file_location("phase2c_batch1_pixels", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def _config(**changes):
    document = {
        "schema_version": 1,
        "experiment_id": "phase2c-b1-sparse-paired-walk1-opened-v1-a1",
        "evidence_boundary": "opened only",
        "clip_key": "walk1.mp4",
        "fold": "development",
        "require_cfr": True,
        "window_half_frames": 5,
        "minimum_suggestion_coverage": 0.25,
        "maximum_candidate_wall_seconds": 35.57,
        "maximum_candidate_peak_rss_mib": 320.0,
        "sparse_config": {},
    }
    document.update(changes)
    return document


def test_tracked_config_file_is_exact_and_fail_closed():
    path = ROOT / "configs" / "phase2c-b1-sparse-paired-walk1-v1.json"
    validated = driver._validate_config(json.loads(path.read_text()))
    assert validated["fold"] == "development"
    assert validated["require_cfr"] is True
    assert validated["effective_sparse_config"] == sparse_roi_config()
    with pytest.raises(ValueError, match="config keys are not exact"):
        driver._validate_config({**_config(), "prediction_threshold": 0.9})
    with pytest.raises(ValueError, match="opened development"):
        driver._validate_config(_config(fold="validation"))
    with pytest.raises(ValueError, match="restricted to a verified CFR"):
        driver._validate_config(_config(require_cfr=False))


def test_split_ranges_are_half_open_sorted_and_disjoint():
    split = {"clips": {"walk1.mp4": {"development": [
        {"start_frame": 0, "end_frame": 10},
        {"start_frame": 20, "end_frame": 30},
    ]}}}
    ranges = driver._fold_ranges(split, "walk1.mp4", "development")
    assert ranges == [(0, 10), (20, 30)]
    assert driver._inside_ranges(0, ranges)
    assert driver._inside_ranges(9, ranges)
    assert not driver._inside_ranges(10, ranges)
    with pytest.raises(ValueError, match="sorted and disjoint"):
        driver._fold_ranges(
            {"clips": {"walk1.mp4": {"development": [
                {"start_frame": 4, "end_frame": 9},
                {"start_frame": 8, "end_frame": 12},
            ]}}},
            "walk1.mp4",
            "development",
        )


def test_actor_box_requires_visible_skeleton_and_is_bounded():
    landmarks = [Point(10 + index, 20 + index, 0, 0.9) for index in range(10)]
    frame = PersonFrame(
        t_s=0.0,
        source_pts_us=0,
        frame_index=0,
        person_id=0,
        score=0.9,
        feet={},
        hips=None,
        shoulders_mid=None,
        landmarks=landmarks,
    )
    box = driver._bound_actor_box(frame, 100, 80)
    assert box is not None
    assert 0 <= box[0] < min(point.x for point in landmarks)
    assert max(point.y for point in landmarks) < box[3] < 80
    frame.landmarks = landmarks[:7]
    assert driver._bound_actor_box(frame, 100, 80) is None


def test_atomic_outputs_refuse_overwrite(tmp_path):
    path = tmp_path / "result.json"
    driver._atomic_json(path, {"b": 2, "a": 1})
    assert path.read_text() == '{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        driver._atomic_json(path, {"a": 3})
