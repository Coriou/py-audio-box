"""Tests for the auto-sfx CLI surface (cli.py / __main__.py).

Covers the artifact writers' round-trips, run-id sanitisation, --config
parsing, Contract B's PTS-injectable backend signatures, an end-to-end
``analyze`` happy path with a fake pose backend injected via monkeypatch,
the ``compare``/``metrics`` commands against fabricated runs, and the
annotate command's ground-truth validation.

The happy-path test renders real videos, so it needs ``ffmpeg``/``ffprobe``
on PATH (present in the POC container); it skips elsewhere. Everything else
is hermetic stdlib-only.
"""

from __future__ import annotations

import contextlib
import csv
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx import cli  # noqa: E402
from auto_sfx.cache import TRACE_FORMAT, TRACE_FORMAT_VERSION, PoseCache  # noqa: E402
from auto_sfx.contacts import ContactMethodResult  # noqa: E402
from auto_sfx.pose import mediapipe_backend, rtmpose_backend  # noqa: E402
from auto_sfx.types import (  # noqa: E402
    LEFT,
    RIGHT,
    ContactEvent,
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
    VideoMeta,
)

_ffmpeg_ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_meta(path="v.mp4", n_frames=6, fps=10.0):
    return VideoMeta(
        path=path,
        codec="h264",
        width=64,
        height=64,
        fps_stream=fps,
        n_frames=n_frames,
        duration_s=n_frames / fps,
        cfr=True,
        vfr=False,
        time_base=None,
        audio_present=False,
    )


def make_frame(i, *, lost=False):
    heel = Point(10 + i, 40)
    return PersonFrame(
        t_s=i / 10.0,
        source_pts_us=i * 100_000,
        frame_index=i,
        person_id=0,
        score=0.0 if lost else 0.9,
        feet={
            LEFT: None
            if lost
            else FootObservation(heel=heel, ankle=Point(10 + i, 42), toe=Point(13 + i, 44)),
            RIGHT: None
            if lost
            else FootObservation(heel=Point(30 - i, 40), ankle=Point(30 - i, 42), toe=Point(33 - i, 44)),
        },
        hips=None if lost else Point(20, 30),
        shoulders_mid=None if lost else Point(20, 12),
        landmarks=[],  # renderer skips skeleton drawing for empty landmark lists
        lost=lost,
    )


def make_event(t, foot, ident=None):
    return ContactEvent(
        id=ident or f"evt_{int(round(t * 1000)):04d}",
        family="locomotion.foot_contact",
        actor_id="person_01",
        foot=foot,
        source_pts_us=int(round(t * 1_000_000)),
        time_s=t,
        earliest_s=t - 0.02,
        latest_s=t + 0.02,
        score=0.87,
        observable=True,
        detector="test-contact",
        method="hybrid",
        evidence={"heel_confidence": 0.94},
    )


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch):
    """Redirect the pose cache away from /cache for every test that runs."""
    monkeypatch.setenv("AUTO_SFX_CACHE_DIR", str(tmp_path / "pose-cache"))
    return tmp_path / "pose-cache"


# ---------------------------------------------------------------------------
# Run-id sanitisation / --config parsing (pure)
# ---------------------------------------------------------------------------


class TestSanitizeRunId:
    def test_safe_characters_pass_through(self):
        assert cli.sanitize_run_id("plain-id_1.2") == "plain-id_1.2"

    def test_path_and_shell_hostile_characters_replaced(self):
        assert cli.sanitize_run_id("../evil:name") == "evil_name"
        assert cli.sanitize_run_id("my_run/id v2") == "my_run_id_v2"
        assert "/" not in cli.sanitize_run_id("a/b//c")

    def test_edge_noise_stripped_and_empty_falls_back(self):
        assert cli.sanitize_run_id("///") == "run"
        assert cli.sanitize_run_id("  ..x..  ") == "x"

    def test_length_capped(self):
        assert len(cli.sanitize_run_id("x" * 500)) <= 120

    def test_default_shape_is_stem_backend_contact_stamp(self):
        import re

        rid = cli.default_run_id("/work/auto-sfx/walk.mp4", "mediapipe-lite", "hybrid")
        assert re.fullmatch(r"walk-mediapipe-lite-hybrid-\d{8}T\d{6}Z", rid)


class TestConfigParsing:
    def test_numeric_values_parsed(self):
        assert cli.parse_config_value("3") == 3
        assert isinstance(cli.parse_config_value("3"), int)
        assert cli.parse_config_value("-0.5") == -0.5
        assert cli.parse_config_value("1e3") == 1000.0

    def test_non_numeric_passthrough(self):
        assert cli.parse_config_value("hybrid") == "hybrid"

    def test_pairs(self):
        got = cli.parse_config_pairs(["a=1", "b=2.5", "c=x"])
        assert got == {"a": 1, "b": 2.5, "c": "x"}

    def test_missing_separator_rejected(self):
        with pytest.raises(ValueError, match="KEY=VAL"):
            cli.parse_config_pairs(["oops"])


# ---------------------------------------------------------------------------
# Artifact writers round-trip
# ---------------------------------------------------------------------------


class TestEventsWritersRoundTrip:
    def test_csv_fixed_columns_and_values(self, tmp_path):
        ev = make_event(3.483, LEFT, "evt_001")
        ev2 = make_event(4.083, RIGHT, "evt_002")
        ev2.observable = False  # exercise the false branch of the CSV writer
        path = cli.write_events_csv([ev, ev2], tmp_path / "events.csv")

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            header = reader.fieldnames
        assert tuple(header) == cli.EVENTS_CSV_COLUMNS

        first = rows[0]
        assert first["id"] == "evt_001"
        assert first["family"] == "locomotion.foot_contact"
        assert first["actor_id"] == "person_01"
        assert first["foot"] == "left"
        assert first["kind"] == "initial_contact"
        assert int(first["source_pts_us"]) == 3483000
        assert float(first["time_s"]) == pytest.approx(3.483)
        assert float(first["earliest_s"]) == pytest.approx(ev.earliest_s)
        assert float(first["latest_s"]) == pytest.approx(ev.latest_s)
        assert float(first["score"]) == pytest.approx(0.87)
        assert first["observable"] == "true"
        assert json.loads(first["evidence"]) == {"heel_confidence": 0.94}
        assert rows[1]["observable"] == "false"

    def test_events_json_document_schema(self, tmp_path):
        meta = make_meta()
        doc = cli.events_document(meta, "mediapipe-lite", "video", "hybrid",
                                  [make_event(0.5, LEFT)])
        assert doc["schema_version"] == 1
        assert set(doc) == {
            "schema_version", "video", "backend", "mode", "contact_method", "events",
        }
        assert doc["video"] == meta.to_dict()

        text = (tmp_path / "e.json")
        cli.write_json(text, doc)
        assert json.loads(text.read_text(encoding="utf-8"))["events"][0] == \
            make_event(0.5, LEFT).to_dict()

    def test_pose_jsonl_mirrors_cache_format(self, tmp_path):
        meta = make_meta(n_frames=3)
        trace = PoseTrace(
            backend="mediapipe-lite", mode="video", meta=meta,
            frames=[make_frame(i) for i in range(3)],
        )
        path = cli.write_pose_jsonl(trace, "cafe1234", tmp_path / "pose.jsonl")

        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        assert header["format"] == TRACE_FORMAT
        assert header["format_version"] == TRACE_FORMAT_VERSION
        assert header["key"] == "cafe1234"
        assert len(lines) == 4  # header + one row per frame

        # Byte-compatible with the PoseCache loader under its own naming.
        (tmp_path / "cafe1234.jsonl").write_bytes(path.read_bytes())
        loaded = PoseCache(root=str(tmp_path)).load_trace("cafe1234")
        assert loaded is not None and len(loaded.frames) == 3
        assert loaded.frames[1].feet[LEFT].heel.y == 40.0


# ---------------------------------------------------------------------------
# Contract B: PTS-injectable backend signatures (import-level only)
# ---------------------------------------------------------------------------


class TestContractBSignatures:
    @pytest.mark.parametrize("module", [mediapipe_backend, rtmpose_backend])
    def test_run_pose_accepts_optional_frames_iterable(self, module):
        sig = inspect.signature(module.run_pose)
        assert "frames" in sig.parameters
        assert sig.parameters["frames"].default is None

    def test_cli_dispatch_exposes_the_same_seam(self):
        sig = inspect.signature(cli.run_pose)
        assert sig.parameters["frames"].default is None


# ---------------------------------------------------------------------------
# analyze happy path with a fake pose backend (real decode/render plumbing)
# ---------------------------------------------------------------------------


def _make_tiny_video(tmp_path: Path) -> Path:
    """A real 64x64 0.4s CFR mp4 so ingest/probe/render run genuinely."""
    video = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.4:r=10",
            "-pix_fmt", "yuv420p", "-y", str(video),
        ],
        check=True, capture_output=True,
    )
    return video


@pytest.mark.skipif(not _ffmpeg_ok, reason="ffmpeg/ffprobe not installed")
class TestAnalyzeHappyPath:
    def test_end_to_end_artifacts_with_fake_backend(
        self, tmp_path, monkeypatch, isolated_cache, capsys
    ):
        video = _make_tiny_video(tmp_path)
        meta = make_meta(str(video), n_frames=4, fps=10.0)

        fake_trace = PoseTrace(
            backend="mediapipe-lite", mode="video", meta=meta,
            frames=[make_frame(i) for i in range(4)],
        )

        seen = {}

        def fake_run_pose(video_path, *, backend, mode="video", frames=None):
            seen["backend"] = backend
            seen["mode"] = mode
            consumed = list(frames) if frames is not None else None
            seen["frames_kw_used"] = frames is not None
            seen["n_consumed"] = len(consumed or [])
            return fake_trace

        def fake_detect(trace, method="hybrid", config=None):
            seen["method"] = method
            seen["config"] = config
            return ContactMethodResult(
                events=[make_event(0.1, LEFT), make_event(0.25, RIGHT)],
                diagnostics=[{"code": "shot_boundary", "t0_s": 0.0, "t1_s": 0.4,
                              "detail": "synthetic"}],
            )

        # The fake pose backend replaces ONLY inference/contact logic;
        # probing, shot detection, decoding, rendering and caching are real.
        monkeypatch.setattr(cli, "run_pose", fake_run_pose)
        monkeypatch.setattr(cli, "detect_contacts", fake_detect)
        monkeypatch.setattr(cli, "resolve_model_sha256", lambda backend: "f" * 64)

        out_dir = tmp_path / "run"
        rc = cli.main([
            "analyze", str(video),
            "--pose", "mediapipe-lite", "--contact", "hybrid",
            "--config", "velocity_weight=0.5",
            "--out", str(out_dir), "--run-id", "unit/run id!",
        ])
        assert rc == 0

        names = {p.name for p in out_dir.iterdir()}
        for want in ("events.json", "events.csv", "diagnostics.json", "pose.jsonl",
                     "resource-report.json", "annotated.mp4", "marker-preview.mp4"):
            assert want in names, names

        # The backend was driven through the ffprobe-PTS frame adapter.
        assert seen["frames_kw_used"] is True
        assert seen["n_consumed"] == 4
        assert seen["backend"] == "mediapipe-lite" and seen["mode"] == "video"
        assert seen["method"] == "hybrid"
        assert seen["config"]["velocity_weight"] == 0.5          # --config passthrough
        assert seen["config"]["shot_boundaries"] == []

        doc = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
        assert doc["schema_version"] == 1
        assert doc["backend"] == "mediapipe-lite"
        assert doc["contact_method"] == "hybrid"
        # Probed metadata (honest decoded-timeline values) stamped onto trace.
        assert doc["video"]["duration_s"] == pytest.approx(0.4)
        assert [e["id"] for e in doc["events"]] == ["evt_0100", "evt_0250"]

        rows = list(csv.DictReader(open(out_dir / "events.csv", newline="",
                                        encoding="utf-8")))
        assert [r["id"] for r in rows] == ["evt_0100", "evt_0250"]

        diag = json.loads((out_dir / "diagnostics.json").read_text(encoding="utf-8"))
        assert diag["diagnostics"][0]["code"] == "shot_boundary"
        assert any("velocity_weight" in n for n in diag["notes"])

        report = json.loads((out_dir / "resource-report.json").read_text(encoding="utf-8"))
        assert report["cache_hit"] is False
        stage_names = {s["name"] for s in report["stages"]}
        assert {"probe", "shots", "pose_extract", "contacts", "render"} <= stage_names
        assert report["total_s"] > 0
        assert isinstance(report["threads_env"], dict)

        # Both videos rendered non-empty; click track written alongside.
        assert (out_dir / "annotated.mp4").stat().st_size > 0
        assert (out_dir / "marker-preview.mp4").stat().st_size > 0
        assert (out_dir / "markers.wav").stat().st_size > 0

        printed = capsys.readouterr().out
        assert "== auto-sfx analyze ==" in printed
        assert "events      2 (left=1 right=1) observable=2" in printed
        assert "cache=MISS" in printed

        # Second identical invocation hits the pose cache: no re-extraction.
        out2 = tmp_path / "run2"
        rc = cli.main(["analyze", str(video), "--out", str(out2)])
        assert rc == 0
        assert seen["n_consumed"] == 4  # unchanged: backend did NOT run again
        report2 = json.loads((out2 / "resource-report.json").read_text(encoding="utf-8"))
        assert report2["cache_hit"] is True
        assert "cache=HIT" in capsys.readouterr().out


@pytest.mark.skipif(not _ffmpeg_ok, reason="ffmpeg/ffprobe not installed")
class TestAnalyzeWithGroundTruth:
    """Phase-2 T15 wiring: analyze --gt threads GT into wav + strip + docs."""

    def _stub_pipeline(self, monkeypatch, video, meta):
        fake_trace = PoseTrace(
            backend="mediapipe-lite", mode="video", meta=meta,
            frames=[make_frame(i) for i in range(4)],
        )
        monkeypatch.setattr(
            cli, "run_pose", lambda *a, **k: fake_trace
        )
        monkeypatch.setattr(
            cli, "detect_contacts",
            lambda trace, method="hybrid", config=None:
                ContactMethodResult(
                    events=[make_event(0.1, LEFT), make_event(0.25, RIGHT)],
                    diagnostics=[],
                ),
        )
        monkeypatch.setattr(cli, "resolve_model_sha256", lambda b: "f" * 64)

    def _gt_file(self, tmp_path, video_stem="tiny"):
        gt = {
            "schema_version": 2,
            "clip": f"{video_stem}.mp4",
            "range_s": {"start": 0.0, "end": 0.4},
            "annotator": "operator",
            "fps_display": 10.0,
            "contacts": [
                {
                    "foot": "left",
                    "earliest_pts_us": 90_000,
                    "preferred_pts_us": 100_000,
                    "latest_pts_us": 130_000,
                    "visibility": "clear",
                    "viewpoint": "sagittal",
                    "notes": "",
                },
            ],
        }
        p = tmp_path / f"{video_stem}.gt.json"
        p.write_text(json.dumps(gt), encoding="utf-8")
        return str(p)

    def test_gt_threads_through_all_artifacts(
        self, tmp_path, monkeypatch, isolated_cache, capsys
    ):
        video = _make_tiny_video(tmp_path)
        meta = make_meta(str(video), n_frames=4, fps=10.0)
        self._stub_pipeline(monkeypatch, video, meta)
        out_dir = tmp_path / "run"
        rc = cli.main([
            "analyze", str(video), "--out", str(out_dir),
            "--gt", self._gt_file(tmp_path),
        ])
        assert rc == 0

        doc = json.loads((out_dir / "events.json").read_text(encoding="utf-8"))
        assert doc["ground_truth"]["n_contacts"] == 1
        assert doc["ground_truth"]["clip"] == "tiny.mp4"

        # The marker WAV covers the GT mark and is longer than the event tail.
        import wave as _wave
        with _wave.open(str(out_dir / "markers.wav"), "rb") as wf:
            assert wf.getnframes() / wf.getframerate() >= 0.13 + 0.100

        printed = capsys.readouterr().out
        assert "ground-truth 1 contacts (left=1)" in printed

    def test_invalid_gt_fails_fast_with_exit_2(
        self, tmp_path, monkeypatch, isolated_cache, capsys
    ):
        video = _make_tiny_video(tmp_path)
        meta = make_meta(str(video), n_frames=4, fps=10.0)
        self._stub_pipeline(monkeypatch, video, meta)
        bad = tmp_path / "bad.gt.json"
        bad.write_text(json.dumps({"schema_version": 2, "clip": "tiny.mp4"}),
                       encoding="utf-8")
        out_dir = tmp_path / "run"
        rc = cli.main(["analyze", str(video), "--out", str(out_dir),
                       "--gt", str(bad)])
        assert rc == 2
        err = capsys.readouterr().err
        assert "annotator" in err          # loader names the missing field
        assert not out_dir.exists() or not (out_dir / "events.json").exists()


# ---------------------------------------------------------------------------
# Empty-trace honesty: POSE_LOST interval + empty events, exit 0
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ffmpeg_ok, reason="ffmpeg/ffprobe not installed")
class TestAnalyzeEmptyTrace:
    def test_all_lost_trace_writes_honest_diagnostics(
        self, tmp_path, monkeypatch, isolated_cache, capsys
    ):
        video = _make_tiny_video(tmp_path)
        meta = make_meta(str(video), n_frames=4, fps=10.0)
        lost_trace = PoseTrace(
            backend="mediapipe-lite", mode="video", meta=meta,
            frames=[make_frame(i, lost=True) for i in range(4)],
        )

        monkeypatch.setattr(cli, "run_pose", lambda *a, **k: lost_trace)
        # Contacts abstains silently -> the CLI must append its own interval.
        monkeypatch.setattr(
            cli, "detect_contacts",
            lambda trace, method="hybrid", config=None: ContactMethodResult([], []),
        )
        monkeypatch.setattr(cli, "resolve_model_sha256", lambda backend: "f" * 64)
        monkeypatch.setattr(cli, "render_annotated", self._fake_render)
        monkeypatch.setattr(cli, "synthesize_marker_wav",
                            lambda events, dur, out, **kw: str(out))

        out_dir = tmp_path / "lost"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["analyze", str(video), "--no-video", "--out", str(out_dir)])
        assert rc == 0

        diag = json.loads((out_dir / "diagnostics.json").read_text(encoding="utf-8"))
        codes = [d["code"] for d in diag["diagnostics"]]
        assert codes.count("pose_lost") == 1
        interval = diag["diagnostics"][0]
        assert interval["t0_s"] == 0.0
        assert interval["t1_s"] == pytest.approx(0.3)  # last decoded frame PTS
        assert json.loads((out_dir / "events.json").read_text())["events"] == []
        assert "WARNING" in buf.getvalue()

    @staticmethod
    def _fake_render(trace, events, frames_iter, out_path, **kw):
        from types import SimpleNamespace

        Path(out_path).write_bytes(b"\x00\x00\x00\x00")
        return SimpleNamespace(path=str(out_path))


# ---------------------------------------------------------------------------
# compare: one shared extraction, per-method dirs, compare-summary.json
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ffmpeg_ok, reason="ffmpeg/ffprobe not installed")
class TestCompareCommand:
    def test_methods_share_cached_trace(
        self, tmp_path, monkeypatch, isolated_cache, capsys
    ):
        video = _make_tiny_video(tmp_path)
        meta = make_meta(str(video), n_frames=4, fps=10.0)
        trace = PoseTrace(backend="mediapipe-lite", mode="video", meta=meta,
                          frames=[make_frame(i) for i in range(4)])

        calls = []

        def fake_detect(tr, method="hybrid", config=None):
            calls.append(method)
            events = ([make_event(0.1, LEFT)] if method != "pelvis"
                      else [make_event(0.1, LEFT), make_event(0.35, RIGHT)])
            return ContactMethodResult(events, [])

        monkeypatch.setattr(cli, "run_pose", lambda *a, **k: trace)
        monkeypatch.setattr(cli, "detect_contacts", fake_detect)
        monkeypatch.setattr(cli, "resolve_model_sha256", lambda b: "f" * 64)

        out_dir = tmp_path / "cmp"
        rc = cli.main(["compare", str(video), "--contacts", "velocity", "pelvis",
                       "--no-video", "--out", str(out_dir)])
        assert rc == 0
        assert calls == ["velocity", "pelvis"]

        summary = json.loads((out_dir / "compare-summary.json").read_text())
        assert set(summary["methods"]) == {"velocity", "pelvis"}
        assert summary["methods"]["velocity"]["n_events"] == 1
        assert summary["methods"]["pelvis"]["n_events"] == 2
        assert summary["methods"]["pelvis"]["n_left"] == 1
        assert summary["methods"]["pelvis"]["n_right"] == 1
        assert summary["methods"]["pelvis"]["mean_step_interval_s"] == \
            pytest.approx(0.25)
        assert "mean_score" in summary["methods"]["velocity"]

        mdir = out_dir / "compare" / "velocity"
        for want in ("events.json", "events.csv", "diagnostics.json", "pose.jsonl",
                     "resource-report.json"):
            assert (mdir / want).is_file(), want
        rep = json.loads((mdir / "resource-report.json").read_text())
        assert {"contacts"} <= {s["name"] for s in rep["stages"]}
        # Both methods report the SAME shared extraction honestly.
        rep2 = json.loads((out_dir / "compare" / "pelvis" /
                           "resource-report.json").read_text())
        assert rep["cache_hit"] == rep2["cache_hit"]

        printed = capsys.readouterr().out
        assert "velocity" in printed and "pelvis" in printed


# ---------------------------------------------------------------------------
# metrics command: dup/miss/wrong-side numbers match metrics.evaluate semantics
# ---------------------------------------------------------------------------


class TestMetricsCommand:
    @staticmethod
    def _fabricate_run(tmp_path: Path) -> tuple[Path, Path]:
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        meta = make_meta("walk.mp4", n_frames=250, fps=25.0)

        def pred(i, t, foot):
            event = make_event(t, foot, f"p{i}")
            return event.to_dict()

        cli.write_json(run_dir / "events.json", {
            "schema_version": 1,
            "video": meta.to_dict(),
            "backend": "mediapipe-lite",
            "mode": "video",
            "contact_method": "hybrid",
            "events": [
                pred(1, 1.005, "left"),   # matches gt 1.0 L
                pred(2, 1.02, "left"),    # duplicate of the same gt contact
                pred(3, 3.005, "right"),  # matched against 3.0 L -> wrong side
                pred(4, 7.77, "left"),    # far from anything -> false positive
            ],
        })
        cli.write_json(run_dir / "resource-report.json", {
            "stages": [{"name": "startup", "duration_s": 1.5}],
            "total_s": 12.5,
            "peak_rss_bytes": 800 * 1024 * 1024,
            "rss_samples": [],
            "sample_interval_s": 0.05,
            "platform": "linux",
            "cache_hit": True,
        })

        gt = tmp_path / "gt.json"
        gt.write_text(json.dumps([
            {"time_s": 1.0, "foot": "left"},
            {"time_s": 2.0, "foot": "right"},
            {"time_s": 3.0, "foot": "left"},
        ]) + "\n")
        return run_dir, gt

    def test_counts_and_resource_merge(self, tmp_path, capsys):
        run_dir, gt = self._fabricate_run(tmp_path)
        out_path = tmp_path / "custom-metrics.json"

        rc = cli.main(["metrics", str(run_dir), str(gt),
                       "--tolerance-s", "0.034", "--out", str(out_path)])
        assert rc == 0
        doc = json.loads(out_path.read_text(encoding="utf-8"))

        # Greedy matching semantics: 1.005 matches 1.0; 1.02 becomes a
        # duplicate; 3.005 R matches 3.0 L (wrong side); 2.0 missed; 7.77 fp.
        assert (doc["n_true"], doc["n_pred"], doc["matched"]) == (3, 4, 2)
        assert doc["duplicates"] == 1
        assert doc["false_positives"] == 1
        assert doc["misses"] == 1
        assert doc["wrong_side"] == 1
        assert doc["within_1_frame_frac"] == pytest.approx(1.0)  # 0.005s*25fps
        assert doc["median_err_s"] == pytest.approx(0.005)
        assert doc["tol_s"] == pytest.approx(0.034)

        # Numbers agree with a direct evaluate() call on identical inputs.
        from auto_sfx.metrics import evaluate

        events_doc = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
        direct = evaluate(events_doc["events"],
                          [{"time_s": 1.0, "foot": "left"},
                           {"time_s": 2.0, "foot": "right"},
                           {"time_s": 3.0, "foot": "left"}],
                          25.0, tol_ms=34.0)
        for key in ("matched", "duplicates", "false_positives", "misses",
                    "wrong_side", "median_err_s"):
            assert doc[key] == direct[key]

        # Resource-report values merged when present.
        res = doc["resources"]
        assert res["processing_wall_time_s"] == pytest.approx(12.5)
        assert res["peak_rss_mb"] == pytest.approx(800.0)
        assert res["rtf"] == pytest.approx(1.25)     # 12.5s wall / 10s clip
        assert res["startup_s"] == pytest.approx(1.5)
        assert res["cache_hit"] is True

        table = capsys.readouterr().out
        assert "duplicates" in table and "wrong_side" in table

    def test_default_out_lands_next_to_events_json(self, tmp_path):
        run_dir, gt = self._fabricate_run(tmp_path)
        rc = cli.main(["metrics", str(run_dir), str(gt)])
        assert rc == 0
        assert (run_dir / "metrics.json").is_file()

    def test_missing_inputs_exit_two(self, tmp_path):
        run_dir, gt = self._fabricate_run(tmp_path)
        assert cli.main(["metrics", str(tmp_path / "absent"), str(gt)]) == 2
        assert cli.main(["metrics", str(run_dir), str(tmp_path / "absent.json")]) == 2


# ---------------------------------------------------------------------------
# annotate command: GT validation and session output
# ---------------------------------------------------------------------------


class TestAnnotateValidation:
    @pytest.fixture()
    def stubbed_probe(self, monkeypatch, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 16)
        meta = make_meta(str(video), n_frames=30, fps=10.0)
        monkeypatch.setattr(cli, "probe_video", lambda path: meta)
        monkeypatch.setattr(cli, "collect_frame_times",
                            lambda path: [i / 10.0 for i in range(30)])
        extracted = []

        def fake_extract(video_path, t_s, k=3, out_dir=None, half_window=0.5):
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for j in range(2 * k + 1):
                (out_dir / f"frame_{j:04d}.png").write_bytes(b"png")
            extracted.append((float(t_s), sorted(out_dir.glob("*.png"))))
            return sorted(out_dir.glob("*.png"))

        monkeypatch.setattr(cli, "extract_frames", fake_extract)
        return video, extracted

    def test_at_extracts_adjacent_frames(self, stubbed_probe, tmp_path):
        video, extracted = stubbed_probe
        out = tmp_path / "ann"
        rc = cli.main(["annotate", str(video), "--at", "1.25",
                       "--frames", "5", "--out", str(out)])
        assert rc == 0
        assert extracted == [(1.25, [out / f"frame_{j:04d}.png" for j in range(5)])]
        assert len(list(out.glob("*.png"))) == 5

    def test_gt_sessions_sorted_and_extracted_per_entry(self, stubbed_probe, tmp_path):
        video, extracted = stubbed_probe
        gt = tmp_path / "gt.json"
        gt.write_text(json.dumps([{"time_s": 0.5, "foot": "left"},
                                  {"time_s": 2.0, "foot": "right"}]))
        out = tmp_path / "gtout"
        rc = cli.main(["annotate", str(video), "--gt", str(gt),
                       "--frames", "3", "--out", str(out)])
        assert rc == 0
        assert [t for t, _ in extracted] == [0.5, 2.0]
        assert sorted(p.name for p in out.iterdir()) == [
            "000_left_t500000", "001_right_t2000000",
        ]

    @pytest.mark.parametrize("payload,fragment", [
        ([{"time_s": 2.0, "foot": "left"}, {"time_s": 0.5, "foot": "right"}],
         "sorted"),
        ([{"time_s": 9.9, "foot": "left"}], "outside video range"),
        ([{"time_s": 1.0, "foot": "middle"}], "foot"),
        ([{"time_nothing": 1}], "time_s"),
    ])
    def test_invalid_gt_exits_two(self, stubbed_probe, tmp_path, capsys,
                                  payload, fragment):
        video, _ = stubbed_probe
        gt = tmp_path / "bad.json"
        gt.write_text(json.dumps(payload))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli.main(["annotate", str(video), "--gt", str(gt),
                           "--out", str(tmp_path / "x")])
        assert rc == 2
        assert fragment in err.getvalue()

    def test_time_outside_video_exits_two_for_at(self, stubbed_probe, tmp_path):
        video, _ = stubbed_probe
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli.main(["annotate", str(video), "--at", "42.0",
                           "--out", str(tmp_path / "y")])
        assert rc == 2
        assert "outside video range" in err.getvalue()

    def test_add_subparsers_wires_five_handlers(self):
        parser = cli.build_parser()
        samples = {
            "analyze": ["v.mp4"],
            "compare": ["v.mp4"],
            "metrics": ["run_dir", "gt.json"],
            "annotate": ["v.mp4", "--at", "1.0"],
            "selftest": [],
        }
        for command, extra in samples.items():
            args = parser.parse_args([command] + extra)
            assert callable(args.handler), command


class TestSelftestCommand:

    def test_runs_cfr_then_vfr_and_prints_raw_metrics(
        self, tmp_path, monkeypatch, capsys
    ):
        meta = make_meta(n_frames=30, fps=10.0)
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()
        (fixtures / "walk_synthetic.mp4").write_bytes(b"f")
        (fixtures / "walk_synthetic_contacts.json").write_text(json.dumps(
            [{"time_s": 0.5, "foot": "left"}]))
        (fixtures / "vfr_walk.mp4").write_bytes(b"v")

        calls = []

        def fake_pipeline(video, *, backend, mode, contact, out_dir,
                          run_id=None, config_pairs=(), no_video=False):
            calls.append({"video": str(video), "backend": backend,
                          "contact": contact, "no_video": no_video})
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            cli.write_json(out / "events.json", {
                "schema_version": 1, "video": meta.to_dict(),
                "backend": backend, "mode": mode, "contact_method": contact,
                "events": [{"id": "e1", "family": "locomotion.foot_contact",
                            "actor_id": "person_01", "foot": "left",
                            "source_pts_us": 500000, "time_s": 0.5,
                            "earliest_s": 0.48, "latest_s": 0.52, "score": 0.9,
                            "observable": True, "detector": "d", "method": "m",
                            "evidence": {}}],
            })
            from types import SimpleNamespace

            return SimpleNamespace(run_id=out.name, out_dir=out, meta=meta,
                                   events=[], diagnostics_counts={},
                                   cache_hit=False,
                                   report={"stages": [{"name": "probe",
                                                       "duration_s": 0.01}],
                                           "total_s": 0.6},
                                   pose_lost=False)

        monkeypatch.setattr(cli, "FIXTURES_DIR", fixtures)
        monkeypatch.setattr(cli, "DEFAULT_RUNS_DIR", str(tmp_path / "runs"))
        monkeypatch.setattr(cli, "analyze_pipeline", fake_pipeline)

        rc = cli.main(["selftest"])
        assert rc == 0
        assert len(calls) == 2
        assert calls[0]["video"].endswith("walk_synthetic.mp4")
        assert calls[0]["no_video"] is False           # videos ON for CFR leg
        assert calls[0]["backend"] == "mediapipe-lite"
        assert calls[1]["video"].endswith("vfr_walk.mp4")
        assert calls[1]["no_video"] is True

        out = capsys.readouterr().out
        assert "raw metrics vs walk_synthetic_contacts.json" in out
        assert "vfr events  1" in out                  # parsed + counted
        assert "timings (CFR vs VFR):" in out
        assert "no gates applied" in out

    def test_missing_fixtures_without_script_fail_honestly(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cli, "FIXTURES_DIR", tmp_path / "absent")
        monkeypatch.setattr(cli, "_fixture_script_candidates", lambda: [])
        with pytest.raises(RuntimeError, match="make_fixtures.py"):
            cli.main(["selftest"])

    def test_fixture_generation_via_subprocess(self, tmp_path, monkeypatch, capsys):
        meta = make_meta(n_frames=30, fps=10.0)
        fixtures = tmp_path / "genme"
        fixtures.mkdir()

        script = tmp_path / "make_fixtures.py"
        script.write_text(
            "import json\n"
            "from pathlib import Path\n"
            f"fx = Path({str(fixtures)!r})\n"
            "(fx / 'walk_synthetic.mp4').write_bytes(b'generated')\n"
            "(fx / 'walk_synthetic_contacts.json').write_text("
            "json.dumps([{'time_s': 0.5, 'foot': 'left'}]) + '\\n')\n"
            "(fx / 'vfr_walk.mp4').write_bytes(b'v')\n"
        )

        captured = {}

        def fake_pipeline(video, *, backend, mode, contact, out_dir,
                          run_id=None, config_pairs=(), no_video=False):
            captured.setdefault("videos", []).append(Path(video).name)
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            cli.write_json(out / "events.json", {
                "schema_version": 1, "video": meta.to_dict(),
                "backend": backend, "mode": mode, "contact_method": contact,
                "events": [],
            })
            from types import SimpleNamespace

            return SimpleNamespace(run_id=out.name, out_dir=out, meta=meta,
                                   events=[], diagnostics_counts={},
                                   cache_hit=False,
                                   report={"stages": [], "total_s": 0.1},
                                   pose_lost=False)

        monkeypatch.setattr(cli, "FIXTURES_DIR", fixtures)
        monkeypatch.setattr(cli, "DEFAULT_RUNS_DIR", str(tmp_path / "runs"))
        monkeypatch.setattr(cli, "analyze_pipeline", fake_pipeline)
        monkeypatch.setenv("AUTO_SFX_MAKE_FIXTURES", str(script))

        rc = cli.main(["selftest"])
        assert rc == 0
        assert (fixtures / "walk_synthetic.mp4").read_bytes() == b"generated"
        assert "[selftest] generating fixtures" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI plumbing details
# ---------------------------------------------------------------------------


class TestCliPlumbing:
    def test_missing_video_exits_two_with_stderr(self, capsys):
        rc = cli.main(["analyze", "/nonexistent/nope.mp4"])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_bare_invocation_prints_help_and_exits_two(self, capsys):
        rc = cli.main([])
        assert rc == 2
        assert "usage" in capsys.readouterr().err

    def test_add_subparsers_wires_five_handlers(self):
        parser = cli.build_parser()
        for command, extra in (
            ("analyze", ["v"]),
            ("compare", ["v"]),
            ("metrics", ["run_dir", "gt.json"]),   # two required positionals
            ("annotate", ["v", "--at", "0.5"]),    # --at/--gt required
            ("selftest", []),
        ):
            args = parser.parse_args([command] + extra)
            assert callable(args.handler)

    def test_analyze_defaults_match_contract(self):
        parser = cli.build_parser()
        args = parser.parse_args(["analyze", "v.mp4"])
        assert args.pose == "mediapipe-lite"
        assert args.mode == "video"
        assert args.contact == "hybrid"
        assert args.no_video is False
        assert args.config == []
