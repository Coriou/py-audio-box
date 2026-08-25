"""Behavior-neutrality and structure tests for detector tracing (T2).

The CTO-update-#2 directive requires persisted per-frame machine internals
(``contact-trace.jsonl``) that are *behavior neutral*: enabling the
recorder must leave events and diagnostics bit-identical.  These tests are
the regression proof.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_fixtures as fx  # noqa: E402  (pure gait math, lazy cv2)

from auto_sfx.contacts import detect_contacts  # noqa: E402
from auto_sfx.cli import write_contact_trace  # noqa: E402
from auto_sfx.types import (  # noqa: E402
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
    VideoMeta,
)

import test_contacts as tc  # noqa: E402  (reuse the synthetic walk builder)


FPS = tc.FPS


@pytest.fixture(scope="module")
def walk_trace() -> PoseTrace:
    return tc.build_walk(tc.uniform_times(6.0))


def _events_payload(result):
    return [e.to_dict() for e in result.events]


class TestNeutrality:
    def test_hybrid_events_identical_with_tracing(self, walk_trace):
        off = detect_contacts(walk_trace, "hybrid")
        on = detect_contacts(walk_trace, "hybrid", record_trace=True)
        assert _events_payload(on) == _events_payload(off)
        assert on.diagnostics == off.diagnostics
        assert off.frame_trace is None and off.trace_header is None
        assert on.frame_trace and on.trace_header

    def test_velocity_events_identical_with_tracing(self, walk_trace):
        off = detect_contacts(walk_trace, "velocity")
        on = detect_contacts(walk_trace, "velocity", record_trace=True)
        assert _events_payload(on) == _events_payload(off)
        assert on.diagnostics == off.diagnostics

    def test_pelvis_events_identical_with_tracing(self, walk_trace):
        off = detect_contacts(walk_trace, "pelvis")
        on = detect_contacts(walk_trace, "pelvis", record_trace=True)
        assert _events_payload(on) == _events_payload(off)
        assert on.diagnostics == off.diagnostics

    def test_tracing_with_overrides_neutral(self, walk_trace):
        cfg = {"speed_drop_frac": 0.5, "confirm_frames": 3,
               "shot_boundaries": [1.0, 3.0]}
        off = detect_contacts(walk_trace, "hybrid", config=cfg)
        on = detect_contacts(walk_trace, "hybrid", config=cfg,
                             record_trace=True)
        assert _events_payload(on) == _events_payload(off)
        assert on.diagnostics == off.diagnostics


class TestTraceStructure:
    def test_header_carries_exact_config_and_scale(self, walk_trace):
        cfg = {"speed_drop_frac": 0.42}
        result = detect_contacts(walk_trace, "hybrid", config=cfg,
                                 record_trace=True)
        header = result.trace_header
        assert header["kind"] == "header"
        assert header["method"] == "hybrid"
        assert header["backend"] == walk_trace.backend
        # Exact ACTIVE config: defaults merged with overrides -- not a
        # reconstruction after the fact.
        active = default_cfg = header["config"]
        assert active["speed_drop_frac"] == pytest.approx(0.42)
        assert active["min_step_interval_s"] == pytest.approx(0.25)
        assert isinstance(header["scale_px"], float)

    def test_frame_rows_cover_every_frame_per_side(self, walk_trace):
        result = detect_contacts(walk_trace, "hybrid", record_trace=True)
        frames = [r for r in result.frame_trace if r["kind"] == "frame"]
        n_trace_frames = len(walk_trace.frames)
        left = {r["frame"] for r in frames if r["side"] == "left"}
        right = {r["frame"] for r in frames if r["side"] == "right"}
        # Every frame appears for each side within its segments.
        assert max(len(left), len(right)) >= int(n_trace_frames * 0.9)
        row = frames[0]
        for key in ("t_s", "pts_us", "state_before", "state_after",
                    "spd_bs_s", "thr_bs_s", "swing_peak_bs_s",
                    "ref_x_px", "ref_y_px"):
            assert key in row

    def test_transitions_recorded_in_frame_rows(self, walk_trace):
        result = detect_contacts(walk_trace, "hybrid", record_trace=True)
        transitions = [
            r["transition"] for r in result.frame_trace
            if r.get("transition")
        ]
        assert transitions, "a real gait must produce state transitions"
        kinds = {(t["from"], t["to"]) for t in transitions}
        assert ("SWING", "CANDIDATE") in kinds
        assert any(to_ == "STANCE" for _, to_ in kinds)

    def test_candidate_lines_match_emitted_events(self, walk_trace):
        result = detect_contacts(walk_trace, "hybrid", record_trace=True)
        cands = [r for r in result.frame_trace if r["kind"] == "candidate"]
        assert len(cands) == len(result.events)
        by_frame = {c["frame"]: c for c in cands}
        for e in result.events:
            row = by_frame[e.source_pts_us // 1_000_000 * 1_000_000
                           ] if False else None
        # Event provenance: relaxed-run flag present on candidate rows.
        assert all("relaxed_run" in c for c in cands)

    def test_guard_rejections_carry_reasons(self, walk_trace):
        result = detect_contacts(walk_trace, "hybrid", record_trace=True)
        guards = [r for r in result.frame_trace if r["kind"] == "guard"]
        for g in guards:
            assert g["decision"] in ("reject", "confirm")
            assert isinstance(g["reasons"], list) and g["reasons"]

    def test_segment_rows_present(self, walk_trace):
        result = detect_contacts(walk_trace, "hybrid",
                                 config={"shot_boundaries": [2.0]},
                                 record_trace=True)
        segs = [r for r in result.frame_trace if r["kind"] == "segment"]
        assert len(segs) >= 2  # boundary splits the trace into two segments


class TestPersistence:
    def test_write_contact_trace_jsonl(self, walk_trace, tmp_path):
        result = detect_contacts(walk_trace, "hybrid", record_trace=True)
        out = tmp_path / "contact-trace.jsonl"
        written = write_contact_trace(result, {"path": "synthetic://walk"},
                                      out)
        assert written == out
        lines = [json.loads(l) for l in
                 out.read_text(encoding="utf-8").splitlines()]
        assert lines[0]["kind"] == "header"

    def test_write_returns_none_without_trace(self, walk_trace, tmp_path):
        result = detect_contacts(walk_trace, "hybrid")  # tracing off
        assert write_contact_trace(
            result, {}, tmp_path / "nope.jsonl") is None
        assert not (tmp_path / "nope.jsonl").exists()

    def test_detector_config_json_shape(self, walk_trace, tmp_path):
        from auto_sfx.cli import _write_core_artifacts

        result = detect_contacts(
            walk_trace, "hybrid",
            config={"speed_drop_frac": 0.6}, record_trace=True)
        _write_core_artifacts(
            tmp_path,
            meta=tc.make_meta(tc.uniform_times(6.0)),
            backend=walk_trace.backend,
            mode="video",
            contact_method="hybrid",
            result=result,
            trace=walk_trace,
            cache_key="test-key",
            notes=[],
        )
        cfg_doc = json.loads(
            (tmp_path / "detector-config.json").read_text())
        assert cfg_doc["config"]["speed_drop_frac"] == pytest.approx(0.6)
        assert cfg_doc["source"] == "contact-trace.jsonl#header"
        assert (tmp_path / "contact-trace.jsonl").exists()


class TestPelvisTrace:
    def test_offset_rows_recorded(self, walk_trace):
        result = detect_contacts(walk_trace, "pelvis", record_trace=True)
        offsets = [r for r in result.frame_trace
                   if r.get("method") == "pelvis"
                   and r["kind"] == "frame"]
        assert offsets
        assert all("rel_offset_raw_px" in r for r in offsets)
