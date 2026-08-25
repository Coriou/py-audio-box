"""RTMPose/Halpe26 comparison backend (rtmlib + ONNX Runtime, CPU).

Wraps ``rtmlib.BodyWithFeet`` (RTMPose-S/M/X checkpoints trained on the
body7/Halpe26 setup) as an alternative pose backend for the auto-sfx POC,
emitting :class:`auto_sfx.types.PersonFrame` traces whose coordinates live in
decoded-frame pixel units (rtmlib already rescales SimCC output back into the
original image space; see citation below).  Halpe26 is a 2-D layout, so every
landmark keeps ``z = 0.0`` and carries the model confidence as ``visibility``.

Heavy dependencies (numpy, opencv, onnxruntime, rtmlib) are imported lazily
inside methods so that importing this module stays cheap and dependency-free;
the pure keypoint-to-trace mapping lives in module-level functions that only
duck-type sequences and are unit-testable without any ML stack installed.

License
-------
The RTMPose/Halpe26 ONNX checkpoints are research outputs (OpenMMLab /
HumanArt training sets) and are NOT cleared for production use.
``auto_sfx.types.RTMPOSE_LICENSE_NOTICE`` is printed verbatim once per
``run_pose`` call (at model-load time).

Verified primary sources (checked 2026-08-25)
---------------------------------------------
Keypoint index table (26 Halpe26 keypoints):

* Tau-J/rtmlib @ main (git tree ``03a1693e59e4f7cd84582c0fb30459b3bf18ad42``),
  ``rtmlib/visualization/skeleton/halpe26.py`` -- canonical index->name table
  used to render skeletons.  Byte-identical in the PyPI sdist of
  ``rtmlib==0.0.16`` (latest release, satisfies our ``>=0.0.5,<0.1`` pin),
  re-fetched and diffed during this verification.
* Chain of custody: ``BodyWithFeet`` documents the "Halpe26 keypoint format"
  and emits 26 keypoints (``rtmlib/tools/solution/body_with_feet.py``);
  ``draw_skeleton`` (``rtmlib/visualization/draw.py``) selects the
  ``halpe26`` skeleton definition whenever ``keypoints.shape[1] == 26`` --
  i.e. the output order IS the halpe26 table below.
* Upstream dataset definition agrees entry-for-entry:
  open-mmlab/mmpose @ main, ``configs/_base_/datasets/halpe26.py``
  (PaStaNet / Halpe-FullBody, CVPR 2020).

Default lightweight model URLs (and balanced/performance alternatives) are
copied verbatim from ``BodyWithFeet.MODE`` in
``rtmlib/tools/solution/body_with_feet.py`` (identical on main and in the
0.0.16 sdist).  The ``.zip`` entries contain ``*-end2end.onnx``;
``rtmlib/tools/file.py::download_checkpoint`` extracts them into
``$TORCH_HOME|~/.cache/rtmlib/hub/checkpoints/<zip-stem>.onnx`` and falls
back to the HuggingFace mirror ``https://huggingface.co/Tau-J/RTMPose`` when
``download.openmmlab.com/mmpose/v1/projects/`` is unreachable.
"""

from __future__ import annotations

import hashlib

from typing import Iterable

from auto_sfx.types import (
    LEFT,
    RIGHT,
    RUNNING_MODES,
    RTMPOSE_LICENSE_NOTICE,
    SIDES,
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
    VideoMeta,
)

__all__ = [
    "BACKEND_NAME",
    "BODY_WITH_FEET_MODELS",
    "DEFAULT_KPT_THRESHOLD",
    "DEFAULT_MODE",
    "HALPE26_KEYPOINTS",
    "RTMPoseBackend",
    "make_lost_frame",
    "person_frame_from_keypoints",
    "run_pose",
    "select_dominant_person",
]

# ---------------------------------------------------------------------------
# Verified Halpe26 keypoint table (see module docstring for provenance)
# ---------------------------------------------------------------------------

BACKEND_NAME = "rtmpose-lightweight"  # one of types.POSE_BACKENDS

#: Index -> semantic name, exactly as in rtmlib's halpe26 skeleton table.
HALPE26_KEYPOINTS = (
    "nose",             #  0
    "left_eye",         #  1
    "right_eye",        #  2
    "left_ear",         #  3
    "right_ear",        #  4
    "left_shoulder",    #  5
    "right_shoulder",   #  6
    "left_elbow",       #  7
    "right_elbow",      #  8
    "left_wrist",       #  9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
    "head",             # 17
    "neck",             # 18
    "hip",              # 19
    "left_big_toe",     # 20
    "right_big_toe",    # 21
    "left_small_toe",   # 22
    "right_small_toe",  # 23
    "left_heel",        # 24
    "right_heel",       # 25
)

# Semantic indices consumed by the POC mapping.  FootObservation.toe maps to
# the big toe (types.py: "toe # foot index / big toe"); the small toes are
# part of the verified table but unused downstream in POC v0.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
_LEFT_ANKLE, _RIGHT_ANKLE = 15, 16
_LEFT_BIG_TOE, _RIGHT_BIG_TOE = 20, 21
_LEFT_HEEL, _RIGHT_HEEL = 24, 25
_FOOT_INDICES = {
    LEFT: (_LEFT_HEEL, _LEFT_ANKLE, _LEFT_BIG_TOE),
    RIGHT: (_RIGHT_HEEL, _RIGHT_ANKLE, _RIGHT_BIG_TOE),
}

#: Minimum per-keypoint confidence for a point to participate in the derived
#: observations (feet / hips / mid-shoulders).  Inclusive (score == thr passes).
DEFAULT_KPT_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Verified BodyWithFeet checkpoint table (rtmlib BodyWithFeet.MODE, verbatim)
# ---------------------------------------------------------------------------

_ONNX_SDK = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"

#: mode -> checkpoint URLs + network input sizes, byte-for-byte the values in
#: rtmlib's ``BodyWithFeet.MODE`` (see module docstring).  Input sizes follow
#: rtmlib's own convention: ``(width, height)``.
BODY_WITH_FEET_MODELS = {
    "lightweight": {
        "det": _ONNX_SDK + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        "det_input_size": (416, 416),
        "pose": _ONNX_SDK +
        "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
        "pose_input_size": (192, 256),
    },
    # rtmlib's own constructor default; kept for parity even though this
    # backend defaults to "lightweight".
    "balanced": {
        "det": _ONNX_SDK + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "det_input_size": (640, 640),
        "pose": _ONNX_SDK +
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        "pose_input_size": (192, 256),
    },
    "performance": {
        "det": _ONNX_SDK + "yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        "det_input_size": (640, 640),
        "pose": _ONNX_SDK +
        "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        "pose_input_size": (288, 384),
    },
}

DEFAULT_MODE = "lightweight"


def _log_license_notice() -> None:
    """Print the research-only license notice verbatim (once per run)."""
    print(RTMPOSE_LICENSE_NOTICE, flush=True)


# ---------------------------------------------------------------------------
# Pure mapping helpers (no numpy / rtmlib required)
# ---------------------------------------------------------------------------


def make_lost_frame(*, t_s: float, pts_us: int, frame_index: int) -> PersonFrame:
    """Build the sentinel frame used when nobody was detected."""
    return PersonFrame(
        t_s=t_s,
        source_pts_us=pts_us,
        frame_index=frame_index,
        person_id=0,
        score=0.0,
        feet={s: None for s in SIDES},
        hips=None,
        shoulders_mid=None,
        landmarks=[],
        lost=True,
    )


def select_dominant_person(scores) -> int | None:
    """Return the row index with the highest mean keypoint confidence.

    ``scores`` is an (N, K) sequence of sequences; returns None when empty.
    """
    best_idx = None
    best_mean = -1.0
    for idx, row in enumerate(scores):
        mean = sum(float(s) for s in row) / len(row)
        if mean > best_mean:
            best_mean = mean
            best_idx = idx
    return best_idx


def _as_point(xy, score: float) -> Point:
    return Point(float(xy[0]), float(xy[1]), 0.0, float(score))


def person_frame_from_keypoints(
    kpts,
    scores,
    *,
    t_s: float,
    pts_us: int,
    frame_index: int,
    kpt_threshold: float = DEFAULT_KPT_THRESHOLD,
) -> PersonFrame:
    """Map one rtmlib result into a :class:`auto_sfx.types.PersonFrame`.

    Accepts either batched output (kpts (N, 26, 2), scores (N, 26) -- what
    ``BodyWithFeet.__call__`` returns) or a single instance (kpts (26, 2),
    scores (26,)).  Coordinates must already be in decoded-frame pixels
    (rtmlib's RTMPose postprocess guarantees that).  Only sequences are
    touched, so numpy arrays pass straight through unconverted.
    """
    if len(kpts) == 0:
        return make_lost_frame(t_s=t_s, pts_us=pts_us, frame_index=frame_index)

    # Normalise single-instance (26, 2) input to the batched view.
    if not hasattr(kpts[0], "__len__"):  # pragma: no cover - defensive
        raise ValueError("kpts must be (N, 26, 2) or (26, 2)")
    if kpts[0] is not None and isinstance(kpts[0][0], (int, float)):
        kpts, scores = [kpts], [scores]

    person = select_dominant_person(scores)
    if person is None:
        return make_lost_frame(t_s=t_s, pts_us=pts_us, frame_index=frame_index)

    row_kpts = kpts[person]
    row_scores = scores[person]
    conf = [float(s) for s in row_scores]

    def confident(idx: int) -> bool:
        return conf[idx] >= kpt_threshold

    def midpoint(a: int, b: int) -> Point | None:
        if not (confident(a) and confident(b)):
            return None
        return Point(
            (float(row_kpts[a][0]) + float(row_kpts[b][0])) / 2.0,
            (float(row_kpts[a][1]) + float(row_kpts[b][1])) / 2.0,
            0.0,
            min(conf[a], conf[b]),
        )

    feet: dict = {}
    for side in SIDES:
        heel_i, ankle_i, toe_i = _FOOT_INDICES[side]
        if confident(heel_i) and confident(ankle_i) and confident(toe_i):
            feet[side] = FootObservation(
                heel=_as_point(row_kpts[heel_i], conf[heel_i]),
                ankle=_as_point(row_kpts[ankle_i], conf[ankle_i]),
                toe=_as_point(row_kpts[toe_i], conf[toe_i]),
            )
        else:
            feet[side] = None

    landmarks = [
        _as_point(row_kpts[i], conf[i]) for i in range(len(row_kpts))
    ]

    return PersonFrame(
        t_s=t_s,
        source_pts_us=pts_us,
        frame_index=frame_index,
        person_id=0,  # dominant actor slot in POC v0
        score=sum(conf) / len(conf),
        feet=feet,
        hips=midpoint(LEFT_HIP, RIGHT_HIP),
        shoulders_mid=midpoint(LEFT_SHOULDER, RIGHT_SHOULDER),
        landmarks=landmarks,
        lost=False,
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class RTMPoseBackend:
    """rtmlib ``BodyWithFeet`` (RTMPose/Halpe26) on ONNX Runtime CPU.

    Model weights are downloaded once by rtmlib into its standard hub cache
    (``$TORCH_HOME`` -> ``/cache/torch/hub/checkpoints`` inside the toolbox
    container), so repeated runs stay offline.
    """

    name = BACKEND_NAME

    def __init__(
        self,
        *,
        mode: str = DEFAULT_MODE,
        backend: str = "onnxruntime",
        device: str = "cpu",
        det: str | None = None,
        det_input_size: tuple | None = None,
        pose: str | None = None,
        pose_input_size: tuple | None = None,
        kpt_threshold: float = DEFAULT_KPT_THRESHOLD,
    ):
        if mode not in BODY_WITH_FEET_MODELS:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of "
                f"{sorted(BODY_WITH_FEET_MODELS)}"
            )
        self.mode = mode
        self.backend = backend
        self.device = device
        self.det_override = det
        self.det_input_size_override = det_input_size
        self.pose_override = pose
        self.pose_input_size_override = pose_input_size
        self.kpt_threshold = kpt_threshold
        self._solution = None
        #: url/path/sha256 evidence for diagnostics.json; filled at load time.
        self.trace_meta: dict = {}

    # -- lifecycle ----------------------------------------------------------

    def ensure_loaded(self) -> "RTMPoseBackend":
        """Lazily import rtmlib and build the detector + pose pipeline."""
        if self._solution is not None:
            return self

        _log_license_notice()

        from rtmlib import BodyWithFeet  # lazy heavy import

        cfg = BODY_WITH_FEET_MODELS[self.mode]
        det = self.det_override or cfg["det"]
        det_input_size = self.det_input_size_override or cfg["det_input_size"]
        pose = self.pose_override or cfg["pose"]
        pose_input_size = (
            self.pose_input_size_override or cfg["pose_input_size"]
        )

        self._solution = BodyWithFeet(
            det=det,
            det_input_size=det_input_size,
            pose=pose,
            pose_input_size=pose_input_size,
            mode=self.mode,
            to_openpose=False,  # keep native Halpe26 layout
            backend=self.backend,
            device=self.device,
        )
        self._collect_trace_meta(det_url=det, pose_url=pose)
        return self

    def _collect_trace_meta(self, *, det_url: str, pose_url: str) -> None:
        """Record resolved model paths + sha256 as run evidence."""

        def describe(url: str, tool) -> dict:
            # rtmlib's BaseTool replaces a URL with the downloaded local ONNX
            # path and keeps it in ``tool.onnx_model``.
            path = tool.onnx_model
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            return {
                "url": url,
                "path": str(path),
                "sha256": digest.hexdigest(),
                "input_size": list(tool.model_input_size),
            }

        self.trace_meta = {
            "backend": BACKEND_NAME,
            "rtmlib_mode": self.mode,
            "runtime": f"{self.backend}/{self.device}",
            "det_model": describe(det_url, self._solution.det_model),
            "pose_model": describe(pose_url, self._solution.pose_model),
        }

    def infer_frame(self, frame_bgr, *, t_s: float, pts_us: int,
                    frame_index: int) -> PersonFrame:
        """Run detection + pose on one BGR frame (decoded pixels in/out)."""
        self.ensure_loaded()
        kpts, scores = self._solution(frame_bgr)
        return person_frame_from_keypoints(
            kpts,
            scores,
            t_s=t_s,
            pts_us=pts_us,
            frame_index=frame_index,
            kpt_threshold=self.kpt_threshold,
        )

    def close(self) -> None:
        self._solution = None

    def __enter__(self) -> "RTMPoseBackend":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Whole-video entry point (CLI wiring contract)
# ---------------------------------------------------------------------------


def _fourcc_to_str(value: int) -> str:
    chars = []
    for shift in (0, 8, 16, 24):
        ch = chr((value >> shift) & 0xFF)
        chars.append(ch if ch.isalnum() else "")
    return "".join(chars)


def _decode_video(video_path: str, backend: RTMPoseBackend,
                  progress_cb,
                  frames: Iterable[dict] | None = None) -> tuple[VideoMeta, list]:
    import cv2  # lazy heavy import

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    est_total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    total_estimate = est_total if est_total > 0 else None  # cb contract

    if fps <= 0.0:
        cap.release()
        raise RuntimeError(
            f"{video_path}: stream reports fps={fps}; cannot derive timestamps"
        )

    def _decoded_frames():
        pos = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                return
            t_s = pos / fps
            yield {
                "bgr": frame_bgr,
                "t_s": t_s,
                "source_pts_us": int(round(t_s * 1_000_000)),
                "frame_index": pos,
            }
            pos += 1

    out_frames = []
    index = 0
    # Frame source: caller-supplied PTS-accurate dicts when given (Contract
    # B), otherwise OpenCV index/fps decoding; inference downstream is
    # identical either way.
    for item in frames if frames is not None else _decoded_frames():
        out_frames.append(
            backend.infer_frame(
                item["bgr"],
                t_s=item["t_s"],
                pts_us=item["source_pts_us"],
                frame_index=item["frame_index"],
            )
        )
        index += 1
        if progress_cb is not None:
            progress_cb(index, total_estimate)
    cap.release()

    # Timestamps are derived deterministically as t_s = frame_index / fps,
    # so the trace is constant-frame-rate *by construction*; cv2 exposes no
    # PTS to detect real VFR (vfr stays False), and audio probing belongs to
    # the ingest stage.
    meta = VideoMeta(
        path=str(video_path),
        codec=fourcc,
        width=width,
        height=height,
        fps_stream=fps,
        n_frames=len(out_frames),  # actually decoded frame count
        duration_s=len(out_frames) / fps,
        cfr=True,
        vfr=False,
        time_base=None,
        audio_present=False,
    )
    return meta, out_frames


def run_pose(video_path: str, *, mode: str = "video",
             progress_cb=None,
             frames: Iterable[dict] | None = None) -> PoseTrace:
    """Pose a whole video (or a single still, ``mode="image"``).

    ``progress_cb(done, total)`` fires once per decoded frame after its
    inference completes: ``done`` is the 1-based count of processed frames,
    ``total`` OpenCV's CAP_PROP_FRAME_COUNT estimate or None when unknown.
    Returns a :class:`auto_sfx.types.PoseTrace` with
    ``backend="rtmpose-lightweight"``
    and attaches a ``trace_meta`` dict (model URLs, local paths, sha256
    digests, runtime info) for diagnostics.json evidence.  The research-only
    license notice is printed verbatim once per call.

    ``frames`` -- optional iterable of ``{"bgr", "t_s", "source_pts_us",
    "frame_index"}`` dicts; when given, inference consumes it instead of the
    internal OpenCV decode (:func:`_decode_video` swaps only the frame
    source), letting callers feed ffprobe-accurate PTS for VFR sources.
    Default ``None`` preserves previous behaviour for existing callers.
    """
    if mode not in RUNNING_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected {RUNNING_MODES}")

    backend = RTMPoseBackend()
    backend.ensure_loaded()

    if mode == "video":
        meta, frames = _decode_video(video_path, backend, progress_cb, frames)
    else:
        import cv2  # lazy heavy import

        frame_bgr = cv2.imread(str(video_path))
        if frame_bgr is None:
            raise ValueError(f"cannot read image: {video_path}")
        height, width = frame_bgr.shape[:2]
        frames = [
            backend.infer_frame(
                frame_bgr, t_s=0.0, pts_us=0, frame_index=0)
        ]
        meta = VideoMeta(
            path=str(video_path),
            codec="image",
            width=width,
            height=height,
            fps_stream=0.0,
            n_frames=1,
            duration_s=0.0,
            cfr=False,
            vfr=False,
            time_base=None,
            audio_present=False,
        )

    backend.close()
    trace = PoseTrace(backend=BACKEND_NAME, mode=mode, meta=meta, frames=frames)
    trace.trace_meta = dict(backend.trace_meta)  # attached evidence dict
    return trace
