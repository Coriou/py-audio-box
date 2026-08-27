"""Driver-level contracts for the opened full-leg identity diagnostic."""

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
SCRIPT = ROOT / "scripts" / "run_phase2c_batch1_identity.py"
SPEC = importlib.util.spec_from_file_location("phase2c_batch1_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def _config(**changes):
    document = {
        "schema_version": 1,
        "experiment_id": "phase2c-b1-identity-paired-opened-v2-a1",
        "evidence_boundary": "opened only",
        "known_opened_exchange": {
            "trace_name": "walk1",
            "frame": 364,
            "required_candidate_state": "distal_swapped",
            "required_neighbour_state": "direct",
        },
        "maximum_candidate_wall_seconds": 5.0,
        "maximum_candidate_peak_rss_mib": 256.0,
        "candidate_config": {},
        "baseline_config": {"policy": "interpolate"},
    }
    document.update(changes)
    return document


def test_tracked_identity_config_is_exact_and_opened_only():
    path = ROOT / "configs" / "phase2c-b1-identity-paired-opened-v2.json"
    assert driver._validate_config(json.loads(path.read_text())) == json.loads(
        path.read_text()
    )
    with pytest.raises(ValueError, match="config keys are not exact"):
        driver._validate_config({**_config(), "accuracy_gate": 0.9})
    with pytest.raises(ValueError, match="candidate_config contains unknown"):
        driver._validate_config(_config(candidate_config={"global_offset": 2}))


def test_source_pts_membership_is_exact_and_deterministic():
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


def test_atomic_identity_outputs_refuse_overwrite(tmp_path):
    path = tmp_path / "result.json"
    driver._atomic_json(path, {"b": 2, "a": 1})
    assert path.read_text() == '{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        driver._atomic_json(path, {"a": 3})
