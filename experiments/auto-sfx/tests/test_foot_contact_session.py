"""Foot-contact session builder and prediction-free UI contract tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_EXPERIMENT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.foot_contact_session import (  # noqa: E402
    build_session,
    load_clip_manifest,
    probe_source_timestamps,
    session_http_url,
)
from auto_sfx.foot_contact_truth import load_pts_map, sha256_file  # noqa: E402


def _probe_result(timestamps, fps="30000/1001"):
    document = {
        "streams": [{"avg_frame_rate": fps, "r_frame_rate": fps}],
        "frames": [
            {"best_effort_timestamp_time": timestamp} for timestamp in timestamps
        ],
    }

    def runner(command, **kwargs):
        assert any("frame=best_effort_timestamp_time" in argument for argument in command)
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(document), stderr="")

    return runner


def test_ffprobe_nonuniform_nonzero_pts_are_not_synthesized_from_fps(tmp_path):
    clip = tmp_path / "vfr.mp4"
    clip.write_bytes(b"fixture")
    fps, pts = probe_source_timestamps(
        clip, runner=_probe_result(["10.000000", "10.040000", "10.110000"])
    )
    assert fps == pytest.approx(30000 / 1001)
    assert pts == [10_000_000, 10_040_000, 10_110_000]
    assert [b - a for a, b in zip(pts, pts[1:])] == [40_000, 70_000]


def test_ffprobe_negative_first_pts_is_preserved(tmp_path):
    clip = tmp_path / "negative.mp4"
    clip.write_bytes(b"fixture")
    _, pts = probe_source_timestamps(
        clip, runner=_probe_result(["-0.125000", "-0.085000", "-0.015000"])
    )
    assert pts == [-125_000, -85_000, -15_000]


def test_ffprobe_fails_closed_when_frame_pts_is_missing(tmp_path):
    clip = tmp_path / "missing.mp4"
    clip.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="refusing to synthesize source PTS from FPS"):
        probe_source_timestamps(
            clip, runner=_probe_result(["0.000000", None, "0.080000"])
        )


def test_builder_emits_hashed_pts_map_and_separate_pass_directory(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "work" / "sources" / "walk.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-video-bytes")
    out = repo / "work" / "annotation" / "pass-a"
    session = build_session(
        entries=[{"path": "work/sources/walk.mp4", "actor_ids": ["lead", "extra"]}],
        out_dir=out,
        repo_root=repo,
        session_id="challenge-01",
        pass_id="pass-a",
        order_seed="seed-a",
        probe=lambda path: (25.0, [5_000_000, 5_040_000, 5_110_000]),
    )
    clip = session["clips"][0]
    pts_path = repo / clip["pts_map_file"]
    pts_map = load_pts_map(pts_path)
    assert pts_map["source_pts_us"] == [5_000_000, 5_040_000, 5_110_000]
    assert clip["source"]["sha256"] == sha256_file(source)
    assert clip["source"]["pts_map_sha256"] == sha256_file(pts_path)
    assert clip["range_s"] == {"start": 5.0, "end": 5.11}
    assert session_http_url(out.resolve(), repo) == (
        "http://127.0.0.1:8765/experiments/auto-sfx/foot_contact_annotator.html"
        "?session=work/annotation/pass-a/session.json"
    )
    with pytest.raises(ValueError, match="separate directory for every annotation pass"):
        build_session(
            entries=[{"path": "work/sources/walk.mp4"}],
            out_dir=out,
            repo_root=repo,
            session_id="challenge-01",
            pass_id="pass-b",
            order_seed="seed-b",
            probe=lambda path: (25.0, [0, 40_000]),
        )


def test_manifest_rejects_prediction_or_overlay_fields(tmp_path):
    path = tmp_path / "clips.json"
    path.write_text(
        json.dumps({"clips": [{"path": "source.mp4", "prediction_file": "events.json"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden/unknown fields: prediction_file"):
        load_clip_manifest(path)


def test_ui_uses_source_pts_for_exports_and_zero_based_media_time_only_for_seek():
    path = _EXPERIMENT_ROOT / "foot_contact_annotator.html"
    if not path.is_file():
        pytest.skip("Docker test target mounts src/tests only, not the top-level HTML artifact")
    html = path.read_text(encoding="utf-8")
    assert "return (sourcePts[frame] - sourcePts[0]) / 1e6;" in html
    assert "earliest_pts_us: sourcePts[earliest]" in html
    assert "preferred_pts_us: sourcePts[preferred]" in html
    assert "location.protocol === \"file:\"" in html
    assert "const requestedToken = ++selectionToken;" in html
    assert "if (requestedToken !== selectionToken) return;" in html
    assert "if (identityChanged && selectedClipId)" in html
    assert "contacts = [];\n    selectClip(index);" in html
    # The only network reads are the declared session and its PTS map.
    assert html.count("fetch(") == 2
