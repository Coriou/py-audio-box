"""Operator foot-contact schema and timestamp-mapping tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_EXPERIMENT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.foot_contact_truth import (  # noqa: E402
    ANNOTATION_KIND,
    PTS_MAP_KIND,
    TOOL,
    TOOL_VERSION,
    annotation_json,
    load_annotation,
    media_time_s_for_frame,
    nearest_display_frame,
    pts_map_sha256,
    validate_annotation,
    validate_pts_map,
    write_annotation,
)
from auto_sfx.gt import load_ground_truth  # noqa: E402


def _pts_map() -> dict:
    return {
        "schema_version": 1,
        "kind": PTS_MAP_KIND,
        "clip": "vfr-walk",
        "source_file": "work/operator-sources/vfr-walk.mp4",
        "source_sha256": "a" * 64,
        "timestamp_source": "ffprobe.best_effort_timestamp_time",
        # Nonzero/negative start and deliberately nonuniform VFR intervals.
        "source_pts_us": [-250_000, -210_000, -140_000, -100_000],
    }


def _contact(**over) -> dict:
    document = {
        "contact_id": "c0001",
        "foot": "left",
        "earliest_pts_us": -210_000,
        "preferred_pts_us": -140_000,
        "latest_pts_us": -100_000,
        "display_frames": {"earliest": 1, "preferred": 2, "latest": 3},
        "visibility": "clear",
        "viewpoint": "diagonal",
        "actor_id": "actor-1",
        "observability": "observable",
        "cropping": "none",
        "audible_onset": {"source_pts_us": -250_000, "displayed_frame": 0},
        "notes": "heel edge first settles at preferred frame",
    }
    document.update(over)
    return document


def _annotation(**over) -> dict:
    pts_map = _pts_map()
    document = {
        "schema_version": 2,
        "kind": ANNOTATION_KIND,
        "clip": "vfr-walk",
        "range_s": {"start": -0.25, "end": -0.1},
        "fps_display": 30.0,
        "annotator": "operator-ben",
        "pass_id": "pass-a",
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "session_id": "challenge-01",
        "source": {
            "file": pts_map["source_file"],
            "sha256": pts_map["source_sha256"],
            "pts_map_sha256": pts_map_sha256(pts_map),
            "frame_count": len(pts_map["source_pts_us"]),
        },
        "contacts": [_contact()],
    }
    document.update(over)
    return document


def test_nonuniform_nonzero_source_pts_stays_canonical_during_browser_seek_mapping():
    pts_map = _pts_map()
    assert media_time_s_for_frame(pts_map, 0) == 0.0
    assert media_time_s_for_frame(pts_map, 2) == pytest.approx(0.11)
    assert nearest_display_frame(pts_map, 0.11) == 2
    assert nearest_display_frame(pts_map, 0.075) == 1
    # Browser media time is normalized; exported source PTS is not relabelled.
    assert pts_map["source_pts_us"][2] == -140_000


def test_validates_full_operator_contract_and_separate_audible_onset():
    canonical = validate_annotation(_annotation(), pts_map=_pts_map())
    contact = canonical["contacts"][0]
    assert contact["preferred_pts_us"] == -140_000
    assert contact["display_frames"]["preferred"] == 2
    assert contact["audible_onset"] == {
        "source_pts_us": -250_000,
        "displayed_frame": 0,
    }
    assert contact["audible_onset"]["source_pts_us"] != contact["preferred_pts_us"]


def test_optional_audible_onset_can_be_absent():
    document = _annotation(contacts=[_contact(audible_onset=None)])
    assert validate_annotation(document, pts_map=_pts_map())["contacts"][0][
        "audible_onset"
    ] is None


def test_rejects_frame_pts_relabelling_on_vfr_map():
    contact = _contact(preferred_pts_us=-183_333)
    with pytest.raises(ValueError, match="does not match displayed frame 2"):
        validate_annotation(_annotation(contacts=[contact]), pts_map=_pts_map())


def test_rejects_declared_pts_map_digest_that_is_not_actual_payload():
    document = _annotation()
    document["source"]["pts_map_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="canonical PTS-map payload"):
        validate_annotation(document, pts_map=_pts_map())


@pytest.mark.parametrize(
    "changes, fragment",
    [
        (
            {"visibility": "clear", "observability": "unobservable"},
            "clear visibility requires observable",
        ),
        (
            {"cropping": "complete", "visibility": "partial"},
            "complete cropping requires occluded visibility",
        ),
    ],
)
def test_rejects_contradictory_observability(changes, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_annotation(
            _annotation(contacts=[_contact(**changes)]), pts_map=_pts_map()
        )


def test_rejects_unknown_prediction_field():
    contact = _contact()
    contact["prediction_score"] = 0.99
    with pytest.raises(ValueError, match="unknown fields are forbidden"):
        validate_annotation(_annotation(contacts=[contact]), pts_map=_pts_map())


def test_deterministic_lossless_round_trip(tmp_path):
    original = _annotation()
    first = tmp_path / "first.gt.json"
    second = tmp_path / "second.gt.json"
    write_annotation(first, original, pts_map=_pts_map())
    loaded = load_annotation(first, pts_map=_pts_map())
    write_annotation(second, loaded, pts_map=_pts_map())
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8") == annotation_json(original, _pts_map())
    assert json.loads(first.read_text()) == loaded


def test_additive_export_remains_readable_by_existing_schema_v2_loader(tmp_path):
    path = tmp_path / "compatible.gt.json"
    path.write_text(annotation_json(_annotation(), _pts_map()), encoding="utf-8")
    existing = load_ground_truth(path)
    assert existing.annotator == "operator-ben"
    assert existing.contacts[0].foot == "left"
    assert existing.contacts[0].preferred_pts_us == -140_000


def test_pts_map_requires_strictly_increasing_genuine_timestamps():
    document = _pts_map()
    document["source_pts_us"] = [0, 40_000, 40_000]
    with pytest.raises(ValueError, match="strictly greater"):
        validate_pts_map(document)
