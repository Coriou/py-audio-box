"""Unit tests: pose trace content-hash cache + MediaPipe backend pure helpers.

Runs without mediapipe/cv2/numpy installed -- those imports are lazy by
contract, and only the pure conversion/cache layers are exercised here.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_sfx import cache as pose_cache_mod  # noqa: E402
from auto_sfx.cache import (  # noqa: E402
    PoseCache,
    compute_pose_cache_key,
    hash_file,
)
from auto_sfx.pose import mediapipe_backend as mpb  # noqa: E402
from auto_sfx.types import POSE_CACHE_PREPROCESS_VERSION, PoseTrace, VideoMeta  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_meta(tmp_path: Path, name: str = "clip.mp4") -> VideoMeta:
    return VideoMeta(
        path=str(tmp_path / name),
        codec="avc1",
        width=1280,
        height=720,
        fps_stream=30.0,
        n_frames=3,
        duration_s=0.1,
        cfr=True,
        vfr=False,
        time_base=None,
        audio_present=False,
    )


def make_trace(meta: VideoMeta) -> PoseTrace:
    """Trace with one good frame (right-foot observation absent) + one lost frame."""
    from auto_sfx.types import FootObservation, Point

    p = lambda x, y, z, v: Point(x=x, y=y, z=z, visibility=v)  # noqa: E731
    from auto_sfx.types import PersonFrame
    good = PersonFrame(
        t_s=0.0,
        source_pts_us=0,
        frame_index=0,
        person_id=0,
        score=0.9,
        feet={
            "left": FootObservation(
                heel=p(10.5, 700.25, 1.0, 0.95),
                ankle=p(11.0, 690.0, 1.0, 0.97),
                toe=p(20.0, 702.0, 1.0, 0.90),
            ),
            "right": None,
        },
        hips=p(100.0, 400.0, 0.5, 0.99),
        shoulders_mid=p(100.0, 200.0, 0.5, 0.98),
        landmarks=[p(float(i), float(i) * 2, 0.1 * i, 0.8) for i in range(33)],
        lost=False,
    )
    partial = PersonFrame(
        t_s=1 / 30.0,
        source_pts_us=33333,
        frame_index=1,
        person_id=0,
        score=0.4,
        feet={"left": None, "right": None},
        hips=None,
        shoulders_mid=None,
        landmarks=[],
        lost=True,
    )
    return PoseTrace(backend="mediapipe-lite", mode="video", meta=meta, frames=[good, partial])


def fake_landmark(x: float, y: float, z: float, visibility: float):
    return types.SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def fake_landmarks(n: int = 33, vis: float = 0.75):
    lms = [fake_landmark(0.01 * i, 0.02 * i, -0.03 * i, vis) for i in range(n)]
    # Give the tracked joints distinctive values.
    lms[23] = fake_landmark(0.40, 0.60, 0.05, 0.91)  # left hip
    lms[24] = fake_landmark(0.60, 0.60, 0.05, 0.89)  # right hip
    lms[27] = fake_landmark(0.42, 0.95, 0.02, 0.80)  # left ankle
    lms[29] = fake_landmark(0.41, 0.97, 0.01, 0.70)  # left heel
    lms[31] = fake_landmark(0.45, 0.99, 0.00, 0.60)  # left toe
    lms[28] = fake_landmark(0.52, 0.95, 0.02, 0.78)  # right ankle
    lms[30] = fake_landmark(0.29, 0.97, 0.01, 0.68)  # right heel
    lms[32] = fake_landmark(0.55, 0.99, 0.00, 0.58)  # right toe
    return lms


@pytest.fixture()
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "pose-cache"


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------


def test_hash_file_is_content_addressed(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    c = tmp_path / "c.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")   # same bytes, different path
    c.write_bytes(b"goodbye world")
    assert hash_file(a) == hash_file(b)
    assert hash_file(a) != hash_file(c)


def test_key_deterministic_and_path_independent(tmp_path: Path) -> None:
    va = tmp_path / "one.mp4"
    vb = tmp_path / "two.mp4"
    va.write_bytes(b"videobytes")
    vb.write_bytes(b"videobytes")
    k1 = compute_pose_cache_key(va, "mediapipe-lite", "video", "msha")
    k2 = compute_pose_cache_key(vb, "mediapipe-lite", "video", "msha")
    assert k1 == k2
    assert len(k1) == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "mediapipe-full"},
        {"mode": "image"},
        {"model_sha256": "different"},
    ],
)
def test_key_sensitive_to_backend_mode_and_model(tmp_path: Path, kwargs: dict) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"videobytes")
    base = compute_pose_cache_key(video, "mediapipe-lite", "video", "msha")
    changed = compute_pose_cache_key(
        video,
        kwargs.get("backend", "mediapipe-lite"),
        kwargs.get("mode", "video"),
        kwargs.get("model_sha256", "msha"),
    )
    assert base != changed


def test_key_changes_when_video_content_changes(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"take-1")
    k1 = compute_pose_cache_key(video, "mediapipe-lite", "video", "msha")
    video.write_bytes(b"take-2")
    k2 = compute_pose_cache_key(video, "mediapipe-lite", "video", "msha")
    assert k1 != k2


def test_preprocess_version_bump_invalidates_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"stable-bytes")
    before = compute_pose_cache_key(video, "mediapipe-lite", "video", "msha")
    monkeypatch.setattr(pose_cache_mod, "POSE_CACHE_PREPROCESS_VERSION", POSE_CACHE_PREPROCESS_VERSION + 1)
    after = compute_pose_cache_key(video, "mediapipe-lite", "video", "msha")
    assert before != after


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------



def test_roundtrip_preserves_rows_and_metadata(cache_root: Path) -> None:
    pc = PoseCache(root=cache_root)
    key = "k" * 64
    trace = make_trace(make_meta(cache_root.parent))
    out = pc.save_trace(trace, key)
    assert out.exists()

    loaded = pc.load_trace(key)
    assert loaded is not None
    assert loaded.backend == trace.backend
    assert loaded.mode == trace.mode
    assert loaded.meta.to_dict() == trace.meta.to_dict()
    assert loaded.to_rows() == trace.to_rows()


def test_load_miss_returns_none(cache_root: Path) -> None:
    assert PoseCache(root=cache_root).load_trace("f" * 64) is None


def test_load_corrupt_entry_returns_none(cache_root: Path) -> None:
    pc = PoseCache(root=cache_root)
    key = "c" * 64
    entry = pc.path_for(key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("this is not json\n{\n", encoding="utf-8")
    assert pc.load_trace(key) is None


def test_empty_entry_returns_none(cache_root: Path) -> None:
    pc = PoseCache(root=cache_root)
    key = "e" * 64
    entry = pc.path_for(key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("", encoding="utf-8")
    assert pc.load_trace(key) is None


def test_stale_entry_with_mismatched_key_is_rejected(cache_root: Path) -> None:
    pc = PoseCache(root=cache_root)
    trace = make_trace(make_meta(cache_root.parent))
    pc.save_trace(trace, "a" * 64)
    # Copy the entry under a different name: header key no longer matches.
    src = pc.path_for("a" * 64).read_text(encoding="utf-8")
    dst = pc.path_for("b" * 64)
    dst.write_text(src, encoding="utf-8")
    assert pc.load_trace("b" * 64) is None
    assert pc.load_trace("a" * 64) is not None


def test_save_is_atomic_no_tmp_leftovers(cache_root: Path) -> None:
    pc = PoseCache(root=cache_root)
    pc.save_trace(make_trace(make_meta(cache_root.parent)), "d" * 64)
    leftovers = [p for p in pc.root.iterdir() if ".tmp-" in p.name]
    assert leftovers == []


def test_end_to_end_key_agreement(tmp_path: Path, cache_root: Path) -> None:
    """The exact wiring the CLI uses: key from content, save, load back."""
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"real-video-content")
    pc = PoseCache(root=cache_root)
    key = compute_pose_cache_key(video, "mediapipe-full", "image", "sha-full")

    assert pc.load_trace(key) is None  # miss
    trace = make_trace(make_meta(tmp_path))
    trace.backend = "mediapipe-full"
    trace.mode = "image"
    pc.save_trace(trace, key)

    loaded = pc.load_trace(key)
    assert loaded is not None
    assert loaded.backend == "mediapipe-full"
    assert loaded.to_rows() == trace.to_rows()

    # Any input change must miss.
    other = compute_pose_cache_key(video, "mediapipe-lite", "image", "sha-full")
    assert pc.load_trace(other) is None



# ---------------------------------------------------------------------------
# MediaPipe backend pure helpers (no mediapipe import required)
# ---------------------------------------------------------------------------


def test_importing_backend_module_does_not_import_heavy_deps() -> None:
    """Importing the backend modules in a fresh interpreter stays light.

    A subprocess is required: sibling test files legitimately pull cv2 into
    THIS interpreter (the renderer uses it), which would make a bare
    ``sys.modules`` assertion order-dependent.
    """
    code = (
        "import sys, importlib"
        "; importlib.import_module('auto_sfx.pose.mediapipe_backend')"
        "; importlib.import_module('auto_sfx.pose.rtmpose_backend')"
        "; heavy = {'mediapipe', 'cv2', 'onnxruntime', 'rtmlib'}"
        "; loaded = heavy & set(sys.modules)"
        "; assert not loaded, f'heavy deps imported: {loaded}'"
    )
    import os
    import subprocess
    env = dict(os.environ, PYTHONPATH="/app/src")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr


def test_landmarks_to_person_frame_mapping_and_scaling() -> None:
    frame = mpb.landmarks_to_person_frame(
        fake_landmarks(),
        t_s=2 / 30.0,
        source_pts_us=66667,
        frame_index=2,
        width=1920,
        height=1080,
    )
    assert not frame.lost
    assert frame.person_id == 0
    assert frame.t_s == pytest.approx(2 / 30.0)
    assert frame.source_pts_us == 66667

    # Full skeleton preserved in pixel space.
    assert len(frame.landmarks) == 33
    nose = frame.landmarks[0]
    assert nose.x == pytest.approx(0.0)
    assert nose.y == pytest.approx(0.0)

    left = frame.feet["left"]
    assert left is not None
    assert left.heel.x == pytest.approx(0.41 * 1920)
    assert left.ankle.y == pytest.approx(0.95 * 1080)
    assert left.toe.visibility == pytest.approx(0.60)
    assert frame.feet["right"] is not None
    assert frame.feet["right"].heel.x == pytest.approx(0.29 * 1920)  # index 30 default row

    # Midpoints of hips/shoulders.
    assert frame.hips is not None
    assert frame.hips.x == pytest.approx((0.40 + 0.60) / 2 * 1920)
    assert frame.hips.y == pytest.approx(0.60 * 1080)
    assert frame.shoulders_mid is not None
    # lm[11]=(0.11, 0.22), lm[12]=(0.12, 0.24) -> mid y = 0.23 of frame height.
    assert frame.shoulders_mid.x == pytest.approx(0.115 * 1920)
    assert frame.shoulders_mid.y == pytest.approx(0.23 * 1080)

    # Score is mean landmark visibility.
    assert frame.score == pytest.approx(sum(lm.visibility for lm in fake_landmarks()) / 33)


def test_lost_frame_when_no_detection() -> None:
    frame = mpb.landmarks_to_person_frame(
        None, t_s=0.0, source_pts_us=0, frame_index=0, width=640, height=360
    )
    assert frame.lost
    assert frame.feet == {"left": None, "right": None}
    assert frame.hips is None
    assert frame.landmarks == []
    assert frame.score == 0.0


def test_monotonic_timestamp_ms() -> None:
    # Microsecond path floors to ms.
    assert mpb.monotonic_timestamp_ms(1_999_999, None, -1) == 1999
    # Equal timestamps are bumped to stay strictly increasing.
    assert mpb.monotonic_timestamp_ms(1_000_000, None, 1000) == 1001
    # t_s fallback: 1.0 s -> 1000 ms, bumped past the previous 1001.
    assert mpb.monotonic_timestamp_ms(None, 1.0, 1001) == 1002
    assert mpb.monotonic_timestamp_ms(None, 1.0, 1500) == 1501
    # Missing both time fields is an error.
    with pytest.raises(ValueError):
        mpb.monotonic_timestamp_ms(None, None, 0)


def test_model_catalogue_and_paths() -> None:
    assert set(mpb.MODEL_URLS) >= {"lite", "full"}
    for variant, url in mpb.MODEL_URLS.items():
        assert url.startswith("https://storage.googleapis.com/mediapipe-models/")
        assert url.endswith(f"float16/latest/pose_landmarker_{variant}.task")
    with pytest.raises(ValueError):
        mpb.model_filename("gigantic")


def test_ensure_models_reports_lite_and_full(tmp_path: Path) -> None:
    """Without network access ensure_models still reports existing files.

    We pre-seed both model files so no download occurs; sha256 must match the
    seeded bytes.
    """
    import hashlib

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for variant in ("lite", "full"):
        dest = models_dir / f"pose_landmarker_{variant}.task"
        payload = f"fake-{variant}-weights".encode()
        dest.write_bytes(payload)
        payloads[variant] = hashlib.sha256(payload).hexdigest()

    result = mpb.ensure_models(models_dir=models_dir)
    assert set(result) == {"mediapipe-lite", "mediapipe-full"}
    for backend_name, variant in (
        ("mediapipe-lite", "lite"),
        ("mediapipe-full", "full"),
    ):
        entry = result[backend_name]
        assert Path(entry["path"]) == models_dir / f"pose_landmarker_{variant}.task"
        assert entry["sha256"] == payloads[variant]
        assert entry["url"] == mpb.MODEL_URLS[variant]


def test_run_pose_validates_arguments() -> None:
    with pytest.raises(ValueError):
        mpb.run_pose("whatever.mp4", backend="rtmpose-lightweight")
    with pytest.raises(ValueError):
        mpb.run_pose("whatever.mp4", mode="live")


def test_cache_header_shape_on_disk(cache_root: Path) -> None:
    pc = PoseCache(root=cache_root)
    key = "9" * 64
    pc.save_trace(make_trace(make_meta(cache_root.parent)), key)
    lines = pc.path_for(key).read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["format"] == "auto-sfx-pose-trace"
    assert header["key"] == key
    assert header["backend"] == "mediapipe-lite"
    rows = [json.loads(line) for line in lines[1:]]
    assert len(rows) == 2
