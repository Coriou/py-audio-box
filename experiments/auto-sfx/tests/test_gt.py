"""Tests for the schema-v2 ground-truth loader (Phase-2 T1 tooling).

The loader is the gate in front of every Phase-2 measurement, so its
validation contract is tested exhaustively: vocabulary enforcement,
PTS integer types, window sanity, sort order, and non-fatal warnings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.gt import (  # noqa: E402
    FOOT_VOCABULARY,
    SCHEMA_VERSION,
    VISIBILITY_VOCABULARY,
    VIEWPOINT_VOCABULARY,
    load_ground_truth,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _contact(i: int, foot: str = "left", **over) -> dict:
    base = i * 1_000_000
    doc = {
        "foot": foot,
        "earliest_pts_us": base,
        "preferred_pts_us": base + 33_333,
        "latest_pts_us": base + 66_666,
        "visibility": "clear",
        "viewpoint": "sagittal",
        "notes": "",
    }
    doc.update(over)
    return doc


def _doc(contacts=None, **over) -> dict:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "clip": "walk1.mp4",
        "range_s": {"start": 10.0, "end": 30.0},
        "annotator": "operator",
        "fps_display": 30.0,
        "contacts": contacts if contacts is not None else [_contact(0)],
    }
    doc.update(over)
    return doc


def _write(tmp_path: Path, payload, name="sample.gt.json") -> str:
    p = tmp_path / name
    p.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    return str(p)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_loads_full_document(self, tmp_path):
        gt = load_ground_truth(_write(tmp_path, _doc()))
        assert gt.clip == "walk1.mp4"
        assert gt.annotator == "operator"
        assert gt.range_start_s == 10.0 and gt.range_end_s == 30.0
        assert gt.fps_display == 30.0
        assert len(gt.contacts) == 1
        c = gt.contacts[0]
        assert c.foot == "left"
        assert c.preferred_pts_us == 33_333
        # PTS integers are canonical; seconds derived.
        assert c.preferred_s == pytest.approx(0.033333)
        assert c.earliest_s <= c.preferred_s <= c.latest_s
        assert gt.source_path.endswith("sample.gt.json")

    def test_multiple_contacts_and_summary(self, tmp_path):
        contacts = [
            _contact(10, foot="left"),
            _contact(11, foot="right"),
            _contact(12, foot="uncertain"),
        ]
        gt = load_ground_truth(
            _write(tmp_path, _doc(contacts=contacts, range_s={"start": 9.0,
                                                             "end": 15.0}))
        )
        s = gt.summary()
        assert s["n_contacts"] == 3
        assert s["by_foot"] == {"left": 1, "right": 1, "uncertain": 1}
        assert s["warnings"] == []

    def test_vocabularies_are_the_frozen_phase2_sets(self):
        assert set(FOOT_VOCABULARY) == {"left", "right", "uncertain"}
        assert set(VISIBILITY_VOCABULARY) == {"clear", "partial", "occluded"}
        assert set(VIEWPOINT_VOCABULARY) == {
            "sagittal", "diagonal", "frontal", "rear", "uncertain",
        }


# ---------------------------------------------------------------------------
# Hard validation failures
# ---------------------------------------------------------------------------


class TestRejections:
    def test_rejects_v1_array_with_pointer_to_metrics_loader(self, tmp_path):
        with pytest.raises(ValueError, match="schema_version"):
            load_ground_truth(
                _write(tmp_path, [{"time_s": 1.0, "foot": "left"}])
            )

    def test_rejects_wrong_schema_version(self, tmp_path):
        with pytest.raises(ValueError, match="schema_version must be 2"):
            load_ground_truth(_write(tmp_path, _doc(schema_version=1)))

    @pytest.mark.parametrize("field", ["clip", "annotator"])
    def test_rejects_missing_provenance_strings(self, tmp_path, field):
        doc = _doc()
        doc[field] = ""
        with pytest.raises(ValueError, match=field):
            load_ground_truth(_write(tmp_path, doc))

    def test_rejects_bad_foot_vocabulary(self, tmp_path):
        with pytest.raises(ValueError, match="foot"):
            load_ground_truth(
                _write(tmp_path, _doc(contacts=[_contact(0, foot="Left")]))
            )

    @pytest.mark.parametrize("bad", [10_000_000.5, True, "10000000", None])
    def test_rejects_noninteger_pts(self, tmp_path, bad):
        with pytest.raises(ValueError, match="integer"):
            load_ground_truth(
                _write(tmp_path, _doc(contacts=[
                    _contact(0, earliest_pts_us=bad),
                ]))
            )

    def test_rejects_window_inversion(self, tmp_path):
        with pytest.raises(ValueError, match="window sanity"):
            load_ground_truth(_write(tmp_path, _doc(contacts=[
                _contact(0, preferred_pts_us=5, latest_pts_us=3),
            ])))

    def test_rejects_unsorted_contacts(self, tmp_path):
        with pytest.raises(ValueError, match="sort order"):
            load_ground_truth(_write(tmp_path, _doc(contacts=[
                _contact(5),   # later
                _contact(4),   # earlier -> violation
            ])))

    def test_rejects_unknown_visibility(self, tmp_path):
        with pytest.raises(ValueError, match="visibility"):
            load_ground_truth(_write(tmp_path, _doc(contacts=[
                _contact(0, visibility="blurry"),
            ])))

    def test_rejects_unknown_viewpoint(self, tmp_path):
        with pytest.raises(ValueError, match="viewpoint"):
            load_ground_truth(_write(tmp_path, _doc(contacts=[
                _contact(0, viewpoint="topdown"),
            ])))

    def test_rejects_inverted_range(self, tmp_path):
        with pytest.raises(ValueError, match="start < end"):
            load_ground_truth(
                _write(tmp_path, _doc(range_s={"start": 20.0, "end": 10.0}))
            )

    def test_rejects_nonpositive_fps(self, tmp_path):
        with pytest.raises(ValueError, match="fps_display"):
            load_ground_truth(_write(tmp_path, _doc(fps_display=0)))

    def test_rejects_invalid_json(self, tmp_path):
        with pytest.raises(ValueError, match="not valid JSON"):
            load_ground_truth(_write(tmp_path, "{not json"))

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="cannot read"):
            load_ground_truth(str(tmp_path / "absent.gt.json"))


# ---------------------------------------------------------------------------
# Non-fatal warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_out_of_range_mark_warns_but_loads(self, tmp_path):
        doc = _doc(range_s={"start": 10.0, "end": 11.0})
        doc["contacts"] = [
            _contact(10),   # ~10.03 s -> inside the range
            _contact(25),   # ~25.03 s -> outside the range
        ]
        gt = load_ground_truth(_write(tmp_path, doc))
        assert len(gt.contacts) == 2
        assert any("outside range_s" in w for w in gt.warnings)

    def test_duplicate_preferred_warns_as_simultaneous_lr(self, tmp_path):
        base = 12 * 1_000_000
        a = _contact(12, foot="left")
        b = _contact(12, foot="right")
        b["preferred_pts_us"] = a["preferred_pts_us"]
        b["earliest_pts_us"] = base
        b["latest_pts_us"] = base + 66_666
        gt = load_ground_truth(_write(tmp_path, _doc(contacts=[a, b])))
        assert any("simultaneous-LR" in w for w in gt.warnings)

    def test_empty_contacts_warns(self, tmp_path):
        gt = load_ground_truth(_write(tmp_path, _doc(contacts=[])))
        assert gt.contacts == []
        assert any("no contacts" in w for w in gt.warnings)
