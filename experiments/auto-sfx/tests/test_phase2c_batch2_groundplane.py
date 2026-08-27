"""Driver-level contracts for the opened ground-plane mechanism diagnostic."""

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
SCRIPT = ROOT / "scripts" / "run_phase2c_batch2_groundplane.py"
SPEC = importlib.util.spec_from_file_location("phase2c_batch2_groundplane", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

SPLIT = json.loads((ROOT / "gt" / "phase2b-split.json").read_text(encoding="utf-8"))


def _config(**changes):
    document = {
        "schema_version": 1,
        "experiment_id": "phase2c-b2-state-groundplane-opened-dev-v1-a1",
        "evidence_boundary": driver.EVIDENCE_BOUNDARY,
        "fold": "development",
        "maximum_candidate_wall_seconds": 8.0,
        "maximum_candidate_peak_rss_mib": 256.0,
        "candidate_config": {
            "geometry_mode": "shot_local",
            "state_mode": "semi_markov",
        },
        "baseline_config": {},
        "variants": json.loads(json.dumps(driver.REQUIRED_VARIANTS)),
        "decision_gates": json.loads(json.dumps(driver.REQUIRED_DECISION_GATES)),
    }
    document.update(changes)
    return document


def _gt_document(clip: str = "walk1.mp4", start: float = 0.06666666666666667,
                 end: float = 8.8, frame: int = 31, fps: float = 30.0):
    pts = int(round(frame / fps * 1_000_000))
    return {
        "schema_version": 2,
        "clip": clip,
        "range_s": {"start": start, "end": end},
        "annotator": "unit-test fixture; not operator truth",
        "fps_display": fps,
        "contacts": [
            {
                "foot": "left",
                "earliest_pts_us": pts - 33_333,
                "preferred_pts_us": pts,
                "latest_pts_us": pts + 33_333,
                "visibility": "clear",
                "viewpoint": "sagittal",
                "notes": "synthetic",
            }
        ],
    }


def _accept_predicates(**changes):
    predicates = {name: True for name in driver.REQUIRED_DECISION_GATES["accept_requires"]}
    predicates.update({
        "offset_like": False,
        "signed_bias_not_improved": False,
        "recall_drop_gt_slack": False,
        "tests_fail": False,
        "global_offset_key_exists": False,
    })
    predicates.update(changes)
    return predicates


def test_tracked_config_keys_are_exact_and_fail_closed():
    path = ROOT / "configs" / "phase2c-b2-state-groundplane-opened-dev-v1.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["experiment_id"] == "phase2c-b2-state-groundplane-opened-dev-v1-a1"
    assert loaded["evidence_boundary"] == driver.EVIDENCE_BOUNDARY
    assert driver._validate_config(loaded) == loaded
    with pytest.raises(ValueError, match="config keys are not exact"):
        driver._validate_config({**_config(), "accuracy_gate": 0.9})
    with pytest.raises(ValueError, match="unknown keys"):
        driver._validate_config(_config(candidate_config={"not_a_real_key": 1}))


def test_fold_other_than_development_raises():
    for fold in ("validation", "challenge", "holdout", "mixed"):
        with pytest.raises(ValueError, match="opened development"):
            driver._validate_config(_config(fold=fold))


def test_global_offset_in_candidate_config_raises():
    with pytest.raises(ValueError, match="global_offset"):
        driver._validate_config(_config(candidate_config={"global_offset": 2}))
    with pytest.raises(ValueError, match="timing_offset"):
        driver._validate_config(
            _config(candidate_config={"geometry_mode": "shot_local", "timing_offset": 1})
        )
    with pytest.raises(ValueError, match="frame_offset"):
        driver._validate_config(_config(baseline_config={"frame_offset": 3}))


def test_validation_shard_paths_are_not_loaded(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "walk1.development.01.gt.json").write_text(
        json.dumps(_gt_document()) + "\n", encoding="utf-8"
    )
    (gt_dir / "walk2.development.01.gt.json").write_text(
        json.dumps(_gt_document(
            clip="walk2.webm", start=0.06666666666666667, end=1.9333333333333333, frame=12
        )) + "\n",
        encoding="utf-8",
    )
    (gt_dir / "walk1.validation.01.gt.json").write_text("NOT JSON", encoding="utf-8")
    (gt_dir / "walk2.validation.99.gt.json").write_text("{", encoding="utf-8")
    (gt_dir / "walk1.challenge.01.gt.json").write_text("NOT JSON", encoding="utf-8")
    (gt_dir / "walk1.holdout.01.gt.json").write_text("NOT JSON", encoding="utf-8")
    paths = driver._development_gt_paths(gt_dir)
    assert [path.name for path in paths] == [
        "walk1.development.01.gt.json",
        "walk2.development.01.gt.json",
    ]
    by_clip, loaded = driver._load_development_gt(gt_dir, SPLIT)
    assert [path.name for path in loaded] == [
        "walk1.development.01.gt.json",
        "walk2.development.01.gt.json",
    ]
    assert all("validation" not in path.name for path in loaded)
    assert by_clip["walk1.mp4"] and by_clip["walk2.webm"]


def test_loaded_validation_range_is_refused(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "walk1.development.01.gt.json").write_text(
        json.dumps(_gt_document(start=8.9, end=17.7, frame=300)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation range"):
        driver._load_development_gt(gt_dir, SPLIT)


def test_offset_like_helper_constant_versus_mixed():
    assert driver._offset_like([2, 2, 2, 2, 2]) is True
    assert driver._offset_like([2] * 8 + [3, 4]) is True
    assert driver._offset_like([2] * 7 + [3, 4, 5]) is False
    assert driver._offset_like([1, -1, 2, 0]) is False
    assert driver._offset_like([0, 0, 0, 0]) is False
    assert driver._offset_like([]) is False


def test_recommended_decision_rejects_offset_like():
    gates = driver.REQUIRED_DECISION_GATES
    assert driver._recommended_decision(
        _accept_predicates(offset_like=True, not_offset_like=False), gates
    ) == "reject"
    assert driver._recommended_decision(_accept_predicates(), gates) == "accept"
    mixed = _accept_predicates(
        candidate_diagnostic_median_abs_le_baseline=False,
        wall_and_rss_within_budget=False,
    )
    assert driver._recommended_decision(mixed, gates) == "inconclusive"


def test_atomic_groundplane_outputs_refuse_overwrite(tmp_path):
    path = tmp_path / "result.json"
    driver._atomic_json(path, {"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        driver._atomic_json(path, {"a": 3})


def test_source_pts_membership_and_json_hash_are_deterministic():
    class Frame:
        def __init__(self, frame_index, source_pts_us):
            self.frame_index = frame_index
            self.source_pts_us = source_pts_us

    class Trace:
        frames = [Frame(0, -1000), Frame(1, 40_000)]

    records = [
        {"frame": 0, "source_pts_us": -1000},
        {"frame": 1, "source_pts_us": 40_000},
    ]
    assert driver._source_pts_failures(Trace(), records) == 0
    records[1]["source_pts_us"] = 39_999
    assert driver._source_pts_failures(Trace(), records) == 1
    assert driver._hash_json({"b": 2, "a": 1}) == driver._hash_json({"a": 1, "b": 2})


def test_boundary_guard_drops_censored_contacts(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    kept = _gt_document(start=0.0, end=8.866666666666667, frame=31)
    censored = {
        "foot": "right",
        "earliest_pts_us": 0,
        "preferred_pts_us": 0,
        "latest_pts_us": 33_333,
        "visibility": "clear",
        "viewpoint": "sagittal",
        "notes": "on the two-frame guard",
    }
    kept["contacts"] = [censored, *kept["contacts"]]
    (gt_dir / "walk1.development.01.gt.json").write_text(
        json.dumps(kept) + "\n", encoding="utf-8"
    )
    (gt_dir / "walk2.development.01.gt.json").write_text(
        json.dumps(_gt_document(
            clip="walk2.webm", start=0.06666666666666667, end=1.9333333333333333, frame=12
        )) + "\n",
        encoding="utf-8",
    )
    by_clip, _ = driver._load_development_gt(gt_dir, SPLIT)
    frames = [driver._preferred_frame(row, 30.0) for row in by_clip["walk1.mp4"]]
    assert 31 in frames
    assert 0 not in frames
