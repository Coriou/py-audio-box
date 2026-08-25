"""MediaPipe Tasks Pose Landmarker backend for the auto-sfx POC.

Public surface (CLI/selftest wiring contract):

* :func:`run_pose` -- decode a video and produce a :class:`PoseTrace`.
* :func:`ensure_models` -- pre-fetch the ``.task`` models; returns
  ``{backend_name: {"path", "sha256", "url"}}``.

Typical wiring::

    models = ensure_models()
    key = compute_pose_cache_key(video, "mediapipe-lite", mode,
                                 models["mediapipe-lite"]["sha256"])
    trace = cache.load_trace(key) or run_pose(video, backend="mediapipe-lite",
                                              mode=mode)
    cache.save_trace(trace, key)

Heavy dependencies (mediapipe, numpy, cv2) are imported lazily inside
functions; importing this module is always cheap.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ..types import (
    LEFT,
    RIGHT,
    RUNNING_MODES,
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
    VideoMeta,
)

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

_MODELS_BASE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
)

# Google serves stable content under the ``float16/latest`` bucket slot. If
# upstream ever refreshes ``latest``, the recorded sha256 changes and the pose
# cache key changes with it -- stale traces cannot survive silently.
MODEL_URLS: dict[str, str] = {
    "lite": f"{_MODELS_BASE_URL}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": f"{_MODELS_BASE_URL}/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": f"{_MODELS_BASE_URL}/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

_BACKEND_VARIANTS = {
    "mediapipe-lite": "lite",
    "mediapipe-full": "full",
}

DEFAULT_CONFIDENCE = 0.5
_DEFAULT_TRACKING_CONFIDENCE = 0.5
_DEFAULT_PRESENCE_CONFIDENCE = 0.5

_DOWNLOAD_CHUNK = 1 << 20


def resolve_models_dir(models_dir: str | os.PathLike | None = None) -> Path:
    """Explicit argument > ``AUTO_SFX_MODELS_DIR`` > image default mount."""
    if models_dir is not None:
        return Path(models_dir)
    return Path(os.environ.get("AUTO_SFX_MODELS_DIR", "/cache/auto-sfx/models"))


def model_filename(variant: str) -> str:
    if variant not in MODEL_URLS:
        raise ValueError(
            f"unknown MediaPipe pose model variant {variant!r}; "
            f"expected one of {tuple(MODEL_URLS)}"
        )
    return f"pose_landmarker_{variant}.task"


def model_path(variant: str, models_dir: str | os.PathLike | None = None) -> Path:
    """Where the ``variant`` model lives locally (may not exist yet)."""
    return resolve_models_dir(models_dir) / model_filename(variant)


def ensure_model(
    variant: str,
    models_dir: str | os.PathLike | None = None,
    force: bool = False,
) -> Path:
    """Return the local ``.task`` file, downloading it on first use."""
    dest = model_path(variant, models_dir)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".part-{os.getpid()}")
    try:
        with urllib.request.urlopen(MODEL_URLS[variant]) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_models(models_dir: str | os.PathLike = "/cache/auto-sfx/models") -> dict[str, dict[str, Any]]:
    """Download any missing Lite/Full models before inference.

    Returns ``{"mediapipe-lite": {...}, "mediapipe-full": {...}}`` where each
    value carries the local ``path``, its ``sha256`` (the pose cache key
    ingredient), and the source ``url``.
    """
    out: dict[str, dict[str, Any]] = {}
    for backend_name, variant in _BACKEND_VARIANTS.items():
        path = ensure_model(variant, models_dir)
        out[backend_name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "url": MODEL_URLS[variant],
        }
    return out


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose, 33 landmarks, blaze_pose topology)
# ---------------------------------------------------------------------------

_LM_LEFT_SHOULDER = 11
_LM_RIGHT_SHOULDER = 12
_LM_LEFT_HIP = 23
_LM_RIGHT_HIP = 24
_LM_LEFT_ANKLE = 27
_LM_RIGHT_ANKLE = 28
_LM_LEFT_HEEL = 29
_LM_RIGHT_HEEL = 30
_LM_LEFT_TOE = 31
_LM_RIGHT_TOE = 32


# ---------------------------------------------------------------------------
# Pure conversion helpers (no mediapipe import needed -- unit-testable)
# ---------------------------------------------------------------------------


def midpoint(a: Point, b: Point) -> Point:
    return Point(
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        z=(a.z + b.z) / 2.0,
        visibility=min(a.visibility, b.visibility),
    )


def normalized_to_point(lm: Any, width: int, height: int) -> Point:
    """Normalised landmark -> pixel-space :class:`Point` (z scaled by width)."""
    return Point(
        x=float(lm.x) * width,
        y=float(lm.y) * height,
        z=float(lm.z) * width,
        visibility=float(getattr(lm, "visibility", 0.0)),
    )


def landmarks_to_person_frame(
    landmarks: Any | None,
    *,
    t_s: float,
    source_pts_us: int,
    frame_index: int,
    width: int,
    height: int,
) -> PersonFrame:
    """One frame of normalised MediaPipe landmarks -> :class:`PersonFrame`.

    ``landmarks`` is a sequence of duck-typed objects exposing ``x``, ``y``,
    ``z`` (and optionally ``visibility``) in normalised [0, 1] coordinates.
    Empty/None yields a ``lost`` frame.
    """
    if not landmarks:
        return PersonFrame(
            t_s=t_s,
            source_pts_us=source_pts_us,
            frame_index=frame_index,
            person_id=0,
            score=0.0,
            feet={LEFT: None, RIGHT: None},
            hips=None,
            shoulders_mid=None,
            landmarks=[],
            lost=True,
        )

    pts = [normalized_to_point(lm, width, height) for lm in landmarks]
    score = sum(p.visibility for p in pts) / len(pts)
    feet = {
        LEFT: FootObservation(
            heel=pts[_LM_LEFT_HEEL], ankle=pts[_LM_LEFT_ANKLE], toe=pts[_LM_LEFT_TOE]
        ),
        RIGHT: FootObservation(
            heel=pts[_LM_RIGHT_HEEL], ankle=pts[_LM_RIGHT_ANKLE], toe=pts[_LM_RIGHT_TOE]
        ),
    }
    return PersonFrame(
        t_s=t_s,
        source_pts_us=source_pts_us,
        frame_index=frame_index,
        person_id=0,
        score=score,
        feet=feet,
        hips=midpoint(pts[_LM_LEFT_HIP], pts[_LM_RIGHT_HIP]),
        shoulders_mid=midpoint(pts[_LM_LEFT_SHOULDER], pts[_LM_RIGHT_SHOULDER]),
        landmarks=pts,
        lost=False,
    )


def monotonic_timestamp_ms(source_pts_us: int | None, t_s: float | None, last_ms: int) -> int:
    """Next strictly increasing millisecond timestamp for ``detect_for_video``."""
    if source_pts_us is not None:
        ms = source_pts_us // 1000
    elif t_s is not None:
        ms = int(round(t_s * 1000.0))
    else:
        raise ValueError("frame needs source_pts_us or t_s")
    if ms <= last_ms:
        ms = last_ms + 1
    return ms


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _fourcc_to_str(fourcc: int) -> str:
    raw = int(fourcc) & 0xFFFFFFFF
    return "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip()


def frames_from_video(
    video_path: str | os.PathLike, fps: float | None = None
) -> Iterator[dict[str, Any]]:
    """Yield ``{"bgr", "t_s", "source_pts_us", "frame_index"}`` frame dicts.

    Timestamps derive deterministically as ``frame_index / fps`` (POC assumes
    constant frame rate). Callers holding a more accurate PTS map should build
    their own iterable instead and drive the landmarker directly.
    """
    import cv2  # lazy: keeps module import light

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {video_path}")
    if fps is None or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"no usable frame rate for {video_path}; pass fps=")
    try:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t_s = index / fps
            yield {
                "bgr": bgr,
                "t_s": t_s,
                "source_pts_us": int(round(t_s * 1_000_000)),
                "frame_index": index,
            }
            index += 1
    finally:
        cap.release()


def _probe_video(video_path: str | os.PathLike):
    """Return ``(fps, width, height, fourcc_int, reported_frame_count)``."""
    import cv2  # lazy

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps or fps <= 0:
        raise RuntimeError(f"no usable frame rate for {video_path}")
    return float(fps), width, height, fourcc, reported


# ---------------------------------------------------------------------------
# Public inference entry point
# ---------------------------------------------------------------------------

ProgressCB = Callable[[int, int | None], None]


def run_pose(
    video_path: str | os.PathLike,
    *,
    backend: str = "mediapipe-lite",
    mode: str = "video",
    min_detection_confidence: float = DEFAULT_CONFIDENCE,
    progress_cb: ProgressCB | None = None,
    frames: Iterable[dict[str, Any]] | None = None,
) -> PoseTrace:
    """Run MediaPipe Pose Landmarker over a whole video file.

    ``backend``   -- ``"mediapipe-lite"`` or ``"mediapipe-full"``.
    ``mode``      -- ``"video"`` (temporal tracking) or ``"image"``
                     (stateless per-frame; the POC compares both for contact-
                     timing phase shift caused by smoothing).
    ``progress_cb(done_frames, total_estimate)`` fires after each frame;
                     ``total_estimate`` may be ``None`` when the container
                     reports no frame count.
    ``frames``    -- optional iterable of ``{"bgr", "t_s", "source_pts_us",
                     "frame_index"}`` dicts (the same shape
                     :func:`frames_from_video` yields). When given, frames
                     are consumed from it and the internal OpenCV decode is
                     skipped entirely -- the CLI uses this to feed
                     ffprobe-accurate PTS so VFR sources keep their true
                     timing. Default ``None`` keeps the legacy index/fps
                     decode behaviour for existing callers.
    """
    if backend not in _BACKEND_VARIANTS:
        raise ValueError(
            f"unknown pose backend {backend!r}; expected one of "
            f"{tuple(_BACKEND_VARIANTS)}"
        )
    if mode not in RUNNING_MODES:
        raise ValueError(f"mode must be one of {RUNNING_MODES}, got {mode!r}")

    variant = _BACKEND_VARIANTS[backend]
    model_asset = ensure_model(variant)
    fps, width, height, fourcc, reported_n = _probe_video(video_path)

    # Lazy heavy imports (module top level stays mediapipe-free).
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mp_vision

    running_mode = (
        mp_vision.RunningMode.VIDEO if mode == "video" else mp_vision.RunningMode.IMAGE
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_asset)),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=float(min_detection_confidence),
        min_pose_presence_confidence=_DEFAULT_PRESENCE_CONFIDENCE,
        min_tracking_confidence=_DEFAULT_TRACKING_CONFIDENCE,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    total_estimate: int | None = reported_n if reported_n > 0 else None
    person_frames: list[PersonFrame] = []
    last_ms = -1
    try:
        import numpy as np

        # Frame source: a caller-supplied PTS-accurate iterable when given
        # (Contract B), otherwise the internal index/fps OpenCV decode.
        frame_iter = (
            frames if frames is not None else frames_from_video(video_path, fps=fps)
        )
        for fr in frame_iter:
            bgr = fr["bgr"]

            # BGR -> contiguous RGB for mp.Image.
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            if mode == "video":
                last_ms = monotonic_timestamp_ms(
                    fr["source_pts_us"], fr["t_s"], last_ms
                )
                result = landmarker.detect_for_video(mp_image, last_ms)
            else:
                result = landmarker.detect(mp_image)

            detections = getattr(result, "pose_landmarks", None) or []
            first = detections[0] if detections else None
            person_frames.append(
                landmarks_to_person_frame(
                    first,
                    t_s=fr["t_s"],
                    source_pts_us=fr["source_pts_us"],
                    frame_index=fr["frame_index"],
                    width=width,
                    height=height,
                )
            )
            if progress_cb is not None:
                progress_cb(fr["frame_index"] + 1, total_estimate)
    finally:
        landmarker.close()

    meta = VideoMeta(
        path=str(video_path),
        codec=_fourcc_to_str(fourcc),
        width=width,
        height=height,
        fps_stream=fps,
        n_frames=len(person_frames),
        duration_s=len(person_frames) / fps,
        cfr=True,          # timestamps are derived as index/fps by construction
        vfr=False,         # unknown from OpenCV; POC assumption
        time_base=None,    # container time base requires ffprobe (ingest stage)
        audio_present=False,
    )
    return PoseTrace(backend=backend, mode=mode, meta=meta, frames=person_frames)


__all__ = [
    "MODEL_URLS",
    "ensure_model",
    "ensure_models",
    "frames_from_video",
    "landmarks_to_person_frame",
    "midpoint",
    "model_filename",
    "model_path",
    "monotonic_timestamp_ms",
    "normalized_to_point",
    "resolve_models_dir",
    "run_pose",
]
