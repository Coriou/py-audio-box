"""Tests for auto_sfx.ingest: probing, PTS mapping, frame streaming.

cv2/numpy are needed only to *build* clips; each cv2-using test skips
gracefully when they are absent, so the suite stays runnable on hosts
without them (the container runs the full thing).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from auto_sfx.ingest import FrameReader, collect_frame_times, probe_video

VFR_FIXTURE = Path("/work/auto-sfx/fixtures/vfr_walk.mp4")
VFR_EXPECTED = Path("/work/auto-sfx/fixtures/vfr_walk_expected_pts.json")

WIDTH, HEIGHT, N_FRAMES, FPS = 64, 48, 12, 30.0


def _write_cfr_clip(
    path: Path,
    n_frames: int = N_FRAMES,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: float = FPS,
) -> Path:
    """Tiny CFR clip: flat frames whose gray level shifts every frame."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened(), "VideoWriter refused to open"
    for i in range(n_frames):
        level = (17 * i) % 255
        writer.write(np.full((height, width, 3), level, dtype=np.uint8))
    writer.release()
    return path


# ---------------------------------------------------------------------------
# probe_video
# ---------------------------------------------------------------------------


def test_probe_video_reports_sane_metadata(tmp_path):
    clip = _write_cfr_clip(tmp_path / "cfr.mp4")

    meta = probe_video(clip)

    assert meta.path == str(clip)
    assert (meta.width, meta.height) == (WIDTH, HEIGHT)
    assert meta.codec == "mpeg4"  # ffprobe codec_name for MPEG-4 part 2
    assert meta.fps_stream == pytest.approx(FPS, abs=0.01)
    # n_frames comes from the decoded PTS timeline, not container claims.
    assert meta.n_frames == N_FRAMES
    assert meta.cfr is True and meta.vfr is False
    assert meta.duration_s == pytest.approx(N_FRAMES / FPS, rel=1e-3)
    assert meta.time_base is not None and meta.time_base > 0
    assert meta.audio_present is False
    assert meta.to_dict()["width"] == WIDTH


def test_probe_video_accepts_str_path(tmp_path):
    clip = _write_cfr_clip(tmp_path / "cfr.mp4")
    meta = probe_video(str(clip))
    assert meta.n_frames == N_FRAMES


def test_probe_video_missing_file_raises():
    with pytest.raises(RuntimeError):
        probe_video("/nonexistent/nope.mp4")


# ---------------------------------------------------------------------------
# collect_frame_times
# ---------------------------------------------------------------------------


def test_collect_frame_times_monotonic_cfr(tmp_path):
    clip = _write_cfr_clip(tmp_path / "cfr.mp4")

    times = collect_frame_times(clip)

    assert len(times) == N_FRAMES
    assert all(b >= a for a, b in zip(times, times[1:])), "PTS not monotonic"
    deltas = [b - a for a, b in zip(times, times[1:])]
    assert deltas, "expected inter-frame deltas"
    for d in deltas:
        assert d == pytest.approx(1.0 / FPS, rel=0.10)


def test_cfr_verdict_matches_tolerance_constant():
    from auto_sfx.ingest import _summarise_decoded_times

    cfr_times = [i / FPS for i in range(90)]
    n, duration, cfr, vfr = _summarise_decoded_times(cfr_times)
    assert (n, cfr, vfr) == (90, True, False)
    assert duration == pytest.approx(3.0, rel=1e-9)

    vfr_times: list[float] = []
    acc = 0.0
    for i in range(360):
        vfr_times.append(acc)
        acc += 1.0 / 60.0 if i % 2 == 0 else 1.0 / 20.0
    n, duration, cfr, vfr = _summarise_decoded_times(vfr_times)
    assert (n, cfr, vfr) == (360, False, True)

    # Degenerate timelines cannot demonstrate a constant rate -> vfr branch.
    assert _summarise_decoded_times([]) == (0, 0.0, False, True)
    assert _summarise_decoded_times([5.0]) == (1, 0.0, False, True)


# ---------------------------------------------------------------------------
# FrameReader
# ---------------------------------------------------------------------------


def test_frame_reader_yields_all_frames_with_ts(tmp_path):
    clip = _write_cfr_clip(tmp_path / "cfr.mp4")
    times = collect_frame_times(clip)

    with FrameReader(clip) as reader:
        items = list(reader)
        assert reader.width == WIDTH and reader.height == HEIGHT
        assert reader.times == times

    assert len(items) == N_FRAMES
    assert [idx for _, _, idx in items] == list(range(N_FRAMES))
    assert [t for _, t, _ in items] == pytest.approx(times)
    np = pytest.importorskip("numpy")
    for frame, _, _ in items:
        assert frame.shape == (HEIGHT, WIDTH, 3)
        assert frame.dtype == np.uint8


def test_frame_reader_start_index_and_max_frames(tmp_path):
    clip = _write_cfr_clip(tmp_path / "cfr.mp4")

    with FrameReader(clip, start_index=2, max_frames=3) as reader:
        items = list(reader)
    assert [idx for _, _, idx in items] == [2, 3, 4]

    with FrameReader(clip, max_frames=5) as reader:
        idxs = [idx for _, _, idx in reader]
    assert idxs == [0, 1, 2, 3, 4]


def test_frame_reader_exhaustion_is_stable(tmp_path):
    clip = _write_cfr_clip(tmp_path / "cfr.mp4")
    reader = FrameReader(clip, max_frames=1)
    with reader:
        first = next(iter(reader))
        assert first[2] == 0
        assert list(reader) == []  # exhausted readers keep raising StopIteration


# ---------------------------------------------------------------------------
# VFR fixture fidelity (requires generated fixtures inside the container)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not VFR_FIXTURE.exists() or not VFR_EXPECTED.exists(),
    reason="run scripts/make_fixtures.py to generate /work/auto-sfx/fixtures",
)
def test_vfr_fixture_decode_fidelity():
    meta = probe_video(VFR_FIXTURE)
    assert meta.vfr is True
    assert meta.cfr is False

    expected = json.loads(VFR_EXPECTED.read_text())
    times = collect_frame_times(VFR_FIXTURE)
    assert len(times) == len(expected), (
        f"decoded {len(times)} frames, expected {len(expected)}"
    )

    diffs = [abs(a - e) for a, e in zip(times, expected)]
    assert statistics.median(diffs) <= 1e-3, "median PTS error exceeds 1 ms"
    assert max(diffs) <= 15e-3, "max PTS error exceeds 15 ms"


