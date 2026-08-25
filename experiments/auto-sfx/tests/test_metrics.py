"""Tests for auto_sfx.metrics (plus neighbours/extraction helpers).

Pure stdlib + pytest. The ffmpeg integration test skips itself when the
binary is unavailable.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx import annotation_helper as ah  # noqa: E402
from auto_sfx import metrics  # noqa: E402


def ann(t, foot="left"):
    return {"time_s": t, "foot": foot}


def ev(t, foot="left", ident=None):
    return {"id": ident or f"e_{t}", "time_s": t, "foot": foot}


# ---------------------------------------------------------------------------
# Matching behaviour
# ---------------------------------------------------------------------------


class TestMatching:
    def test_exact_match(self):
        report = metrics.evaluate([ev(1.5)], [ann(1.5)], 30)
        assert report["matched"] == 1
        assert report["misses"] == 0
        assert report["false_positives"] == 0
        assert report["median_err_s"] == 0.0
        assert report["matched_pairs"] == [
            {
                "pred_id": "e_1.5",
                "ann_time": 1.5,
                "pred_time": 1.5,
                "err_s": 0.0,
                "foot_ok": True,
            }
        ]

    def test_edge_of_tolerance_accepted(self):
        # 0.75 - 0.5 = 0.25 is exact in binary floating point, so this hits
        # the inclusive boundary precisely.
        report = metrics.evaluate([ev(0.75)], [ann(0.5)], 30, tol_ms=250)
        assert report["tol_s"] == 0.25
        assert report["matched"] == 1
        assert report["matched_pairs"][0]["err_s"] == 0.25

    def test_just_outside_tolerance_rejected(self):
        report = metrics.evaluate([ev(0.75)], [ann(0.5)], 30, tol_ms=249)
        assert report["matched"] == 0
        assert report["misses"] == 1
        assert report["false_positives"] == 1
        assert report["unmatched_anns"] == [{"time_s": 0.5, "foot": "left"}]
        assert report["unmatched_preds"][0]["id"] == "e_0.75"

    def test_duplicate_second_prediction_counts(self):
        events = [ev(1.0, ident="first"), ev(1.02, ident="second")]
        report = metrics.evaluate(events, [ann(1.0)], 30, tol_ms=50)
        assert report["n_pred"] == 2
        assert report["n_true"] == 1
        assert report["matched"] == 1
        assert report["duplicates"] == 1
        # A duplicate is not a plain false positive...
        assert report["false_positives"] == 0
        assert report["misses"] == 0
        # ...but it did not get a one-to-one match either.
        assert [u["id"] for u in report["unmatched_preds"]] == ["second"]
        assert report["matched_pairs"][0]["pred_id"] == "first"

    def test_wrong_side_matched_but_flagged(self):
        report = metrics.evaluate([ev(2.0, foot="right")], [ann(2.0, foot="left")], 30)
        assert report["matched"] == 1
        assert report["wrong_side"] == 1
        assert report["wrong_side_pairs"] == [
            {
                "pred_id": "e_2.0",
                "ann_time": 2.0,
                "pred_time": 2.0,
                "err_s": 0.0,
                "ann_foot": "left",
                "pred_foot": "right",
            }
        ]
        assert report["matched_pairs"][0]["foot_ok"] is False

    def test_miss_and_false_positive(self):
        # Pred far outside tolerance of every annotation: one miss, one FP.
        report = metrics.evaluate([ev(5.0)], [ann(1.0)], 30, tol_ms=500)
        assert report["matched"] == 0
        assert report["duplicates"] == 0
        assert report["misses"] == 1
        assert report["false_positives"] == 1

    def test_accounting_invariants_on_mixed_scene(self):
        events = [
            ev(10.00, foot="left"),   # matches a0
            ev(10.04, foot="left"),   # duplicate of a0
            ev(20.09, foot="right"),  # wrong-side match of a1 (dt 0.09 < tol)
            ev(31.00, foot="left"),   # plain false positive
            ev(99.99, foot="left"),   # plain false positive
        ]
        anns = [ann(10.0), ann(20.0, foot="left"), ann(29.98)]
        report = metrics.evaluate(events, anns, 30, tol_ms=100)
        assert report["matched"] == 2          # a0, a1
        assert report["duplicates"] == 1       # second pred near a0
        assert report["false_positives"] == 2  # 31.00, 99.99
        assert report["wrong_side"] == 1       # right-foot pred on left-foot a1
        assert report["matched"] + report["misses"] == report["n_true"]
        assert (
            report["matched"] + report["duplicates"] + report["false_positives"]
            == report["n_pred"]
        )


class TestToleranceDefaults:
    def test_default_tol_uses_fps_when_frame_gap_dominates(self):
        # 1.5 / 10 fps = 150 ms > 66 ms floor.
        report = metrics.evaluate([], [], 10)
        assert report["tol_s"] == 0.15

    def test_default_tol_floor_at_66ms(self):
        # 1.5 / 60 fps = 25 ms -> clamped to 66 ms.
        report = metrics.evaluate([], [], 60)
        assert report["tol_s"] == 0.066

    def test_nonpositive_fps_falls_back_to_floor(self):
        report = metrics.evaluate([], [], 0)
        assert report["tol_s"] == 0.066


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


class TestAggregates:
    @staticmethod
    def _spread_errors():
        """Ten pairs whose absolute errors are 0.01 .. 0.10 s."""
        anns = [ann(i * 10.0) for i in range(10)]
        events = [ev(i * 10.0 + 0.01 * (i + 1)) for i in range(10)]
        return events, anns

    def test_median_and_p90_on_known_set(self):
        events, anns = self._spread_errors()
        report = metrics.evaluate(events, anns, 30, tol_ms=2000)
        # Linear interpolation over sorted errors [0.01 .. 0.10]:
        #   median: pos 4.5  -> 0.055
        #   p90:    pos 8.1  -> 0.091
        assert report["median_err_s"] == pytest.approx(0.055, abs=1e-9)
        assert report["p90_err_s"] == pytest.approx(0.091, abs=1e-9)

    def test_within_one_frame_frac(self):
        # 50 fps -> one frame = 20 ms; errors 4 ms (in) and 30 ms (out).
        events = [ev(100.004), ev(200.03)]
        anns = [ann(100.0), ann(200.0)]
        report = metrics.evaluate(events, anns, 50, tol_ms=100)
        assert report["within_1_frame_frac"] == 0.5

    def test_fp_per_minute_over_annotation_span(self):
        events = [ev(5.0), ev(65.0)]  # two FPs inside a 60 s span
        anns = [ann(10.0), ann(70.0)]
        report = metrics.evaluate(events, anns, 30, tol_ms=500)
        assert report["false_positives"] == 2
        assert report["fp_per_minute"] == 2.0

    def test_fp_per_minute_none_without_span(self):
        assert metrics.evaluate([ev(1.0)], [ann(1.0)], 30)["fp_per_minute"] is None
        assert metrics.evaluate([], [], 30)["fp_per_minute"] is None
        # A single annotation has zero span.
        assert metrics.evaluate([ev(9.0)], [ann(1.0)], 30)["fp_per_minute"] is None


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestInputs:
    def test_empty_inputs_are_safe(self):
        report = metrics.evaluate([], [], 30)
        assert report["n_true"] == 0
        assert report["n_pred"] == 0
        assert report["matched"] == 0
        assert report["duplicates"] == 0
        assert report["false_positives"] == 0
        assert report["misses"] == 0
        assert report["wrong_side"] == 0
        assert report["median_err_s"] is None
        assert report["p90_err_s"] is None
        assert report["within_1_frame_frac"] is None
        assert report["matched_pairs"] == []
        assert report["unmatched_preds"] == []
        assert report["unmatched_anns"] == []

    def test_object_events_via_getattr(self):
        event = types.SimpleNamespace(id="obj-1", time_s=3.0, foot="right")
        report = metrics.evaluate([event], [ann(3.0, foot="right")], 30)
        assert report["matched"] == 1
        assert report["matched_pairs"][0]["pred_id"] == "obj-1"
        assert report["matched_pairs"][0]["foot_ok"] is True

    def test_missing_event_id_gets_synthetic_name(self):
        report = metrics.evaluate([{"time_s": 1.0, "foot": "left"}], [ann(1.0)], 30)
        assert report["matched_pairs"][0]["pred_id"] == "event_0"

    def test_extra_annotation_keys_tolerated_in_evaluate(self):
        # The "note" key must not invalidate the record. Greedy one-to-one
        # matching: the pred pairs with the first ann (candidates tie at dt=0
        # and (pi, ai) order puts the first ann first); the second ann then
        # has no free pred left, so it is a plain miss -- no duplicate is
        # recorded because duplicates require a FREE pred hitting a taken ann.
        report = metrics.evaluate(
            [ev(1.0)],
            [ann(1.0), {"time_s": 1.0, "foot": "left", "note": "blurry"}],
            30,
        )
        assert report["n_true"] == 2
        assert report["matched"] == 1
        assert report["duplicates"] == 0
        assert report["misses"] == 1
        assert report["false_positives"] == 0


# ---------------------------------------------------------------------------
# load_annotations
# ---------------------------------------------------------------------------


class TestLoadAnnotations:
    def test_loads_and_normalises(self, tmp_path):
        path = tmp_path / "ann.json"
        path.write_text(
            "[\n"
            "  {\"time_s\": 1.432, \"foot\": \"left\", \"certainty\": \"high\"},\n"
            "  {\"time_s\": \"2.5\", \"foot\": \"RIGHT\"}\n"
            "]",
            encoding="utf-8",
        )
        records = metrics.load_annotations(path)
        assert records == [
            {"time_s": 1.432, "foot": "left", "certainty": "high"},
            {"time_s": 2.5, "foot": "right"},
        ]
        assert all(isinstance(r["time_s"], float) for r in records)

    def test_empty_array_is_valid(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        assert metrics.load_annotations(path) == []

    def test_bad_records_reported_with_line_numbers(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(
            "[\n"                                             # line 1
            "  {\"time_s\": 1.0, \"foot\": \"left\"},\n"      # line 2 (good)
            "  {\"time_s\": 2.0, \"foot\": \"up\"},\n"        # line 3 (bad foot)
            "  {\"time_s\": \"abc\", \"foot\": \"right\"},\n"  # line 4 (bad time)
            "  {\"foot\": \"left\"}\n"                        # line 5 (missing time)
            "]",
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as excinfo:
            metrics.load_annotations(path)
        message = str(excinfo.value)
        assert "line 3" in message
        assert "line 4" in message
        assert "line 5" in message
        assert "\n  line 2:" not in message  # the good record is never listed

    def test_non_array_top_level_rejected(self, tmp_path):
        path = tmp_path / "obj.json"
        path.write_text("{\"time_s\": 1.0}", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            metrics.load_annotations(path)

    def test_malformed_json_reports_line(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("[\n  {\"time_s\": 1.0, },\n]", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            metrics.load_annotations(path)

    def test_non_finite_time_rejected(self, tmp_path):
        path = tmp_path / "nan.json"
        # NaN is valid JSON for Python's parser but useless as a timestamp.
        path.write_text("[{\"time_s\": NaN, \"foot\": \"left\"}]", encoding="utf-8")
        with pytest.raises(ValueError, match="finite"):
            metrics.load_annotations(path)


# ---------------------------------------------------------------------------
# annotation_helper.neighbors
# ---------------------------------------------------------------------------


class TestNeighbors:
    FRAMES = [round(i * 0.1, 10) for i in range(11)]  # 0.0 .. 1.0 @ 10 fps

    def test_k_before_and_after_nearest(self):
        result = ah.neighbors(self.FRAMES, 0.52, k=2)
        assert [idx for idx, _ in result] == [3, 4, 5, 6, 7]
        assert result[2] == (5, 0.5)

    def test_window_clipped_at_boundaries(self):
        result = ah.neighbors(self.FRAMES, 0.02, k=3)
        assert [idx for idx, _ in result] == [0, 1, 2, 3]

    def test_exact_hit_is_centre(self):
        result = ah.neighbors(self.FRAMES, 0.7, k=1)
        assert [idx for idx, _ in result] == [6, 7, 8]
        assert result[1] == (7, pytest.approx(0.7))

    def test_zero_k_returns_only_centre(self):
        assert ah.neighbors(self.FRAMES, 0.44, k=0) == [(4, 0.4)]

    def test_empty_input(self):
        assert ah.neighbors([], 1.0, k=3) == []


# ---------------------------------------------------------------------------
# annotation_helper.extract_frames (integration, needs ffmpeg)
# ---------------------------------------------------------------------------

_ffmpeg_ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _ffmpeg_ok, reason="ffmpeg/ffprobe not installed")
class TestExtractFrames:
    @pytest.fixture(scope="class")
    def tiny_video(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("video") / "tiny.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1:r=10",
                "-pix_fmt", "yuv420p", "-y", str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    def test_extracts_neighbourhood_sorted(self, tmp_path, tiny_video):
        paths = ah.extract_frames(tiny_video, 0.5, k=2, out_dir=tmp_path)
        # 10 fps source, 2k+1 frames around t=0.5.
        assert len(paths) == 5
        assert paths == sorted(paths)
        pattern = re.compile(r"^frame_(\d{4})_t(\d+)\.png$")
        stamps = []
        for p in paths:
            match = pattern.match(Path(p).name)
            assert match, f"unexpected name {Path(p).name}"
            assert Path(p).stat().st_size > 0
            stamps.append(int(match.group(2)))
        # Microsecond PTS names land on the 100 ms grid around 0.5 s.
        expected = [int(round(t * 1e6)) for t in (0.3, 0.4, 0.5, 0.6, 0.7)]
        assert stamps == expected

    @pytest.fixture(scope="class")
    def audio_only_video(self, tmp_path_factory):
        # Valid container, zero video streams: v:0 probing must find nothing.
        out = tmp_path_factory.mktemp("video") / "audio_only.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=440:d=1",
                "-c:a", "aac", "-y", str(out),
            ],
            check=True, capture_output=True,
        )
        return out

    def test_missing_frames_raises_usefully(self, tmp_path, audio_only_video):
        with pytest.raises(RuntimeError, match="no frames probed"):
            ah.extract_frames(audio_only_video, 0.5, k=1, out_dir=tmp_path)


# ---------------------------------------------------------------------------
# print_table formatting
# ---------------------------------------------------------------------------


class TestPrintTable:
    def test_alignment_and_rounding(self, capsys):
        ah.print_table([
            {"time_s": 1.432123456, "foot": "left"},
            {"time_s": 2.0, "foot": "right", "note": "ok"},
        ])
        lines = capsys.readouterr().out.rstrip("\n").splitlines()
        assert len(lines) == 4  # header, separator, two rows
        header = lines[0]
        assert header.split() == ["time_s", "foot", "note"]  # first-seen order
        assert set(lines[1]) == {"-", " "}  # separator under every column
        # Floats render rounded to 6 dp; missing cells render blank-padded.
        assert "1.432123" in lines[2]
        assert "left" in lines[2]
        assert "2.000000" in lines[3]
        assert "ok" in lines[3]

    def test_empty_rows(self, capsys):
        ah.print_table([])
        assert capsys.readouterr().out.strip() == "(no rows)"


# ---------------------------------------------------------------------------
# Sanity: documented annotation format round-trips end to end
# ---------------------------------------------------------------------------


def test_documented_format_example(tmp_path):
    payload = [{"time_s": 1.432, "foot": "left"}]
    path = tmp_path / "example.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = metrics.load_annotations(path)
    assert math.isclose(loaded[0]["time_s"], 1.432)
    report = metrics.evaluate(
        [{
            "id": "c1",
            "family": "locomotion.foot_contact",
            "time_s": 1.435,
            "foot": "left",
        }],
        loaded,
        30,
    )
    assert report["matched"] == 1
    assert abs(report["matched_pairs"][0]["err_s"]) <= 0.066
