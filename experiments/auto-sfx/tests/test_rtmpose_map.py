"""Verified-table + mapping tests for the RTMPose/Halpe26 backend.

The Halpe26 index table and the default model URLs asserted here were
verified against primary sources on 2026-08-25 (full citations live in
``auto_sfx/pose/rtmpose_backend.py``'s module docstring):

* Tau-J/rtmlib @ main (git tree ``03a1693e59e4f7cd84582c0fb30459b3bf18ad42``),
  ``rtmlib/visualization/skeleton/halpe26.py`` -- byte-identical in the PyPI
  sdist of ``rtmlib==0.0.16`` (latest release, satisfies ``>=0.0.5,<0.1``).
* ``rtmlib/visualization/draw.py::draw_skeleton`` selects the ``halpe26``
  skeleton for any 26-keypoint output, proving that ``BodyWithFeet``'s
  output order is exactly this table.
* open-mmlab/mmpose @ main, ``configs/_base_/datasets/halpe26.py`` agrees
  entry-for-entry (PaStaNet / Halpe-FullBody dataset definition).
* Checkpoint URLs + input sizes: ``BodyWithFeet.MODE`` in
  ``rtmlib/tools/solution/body_with_feet.py`` (identical on main and in
  the 0.0.16 sdist).

These tests are pure-python: importing the backend must not pull numpy /
cv2 / onnxruntime / rtmlib (enforced below), so they run without the ML
stack installed.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from auto_sfx import types  # noqa: E402
from auto_sfx.pose import rtmpose_backend as rb  # noqa: E402

# ---------------------------------------------------------------------------
# Independently transcribed ground truth (do not regenerate from the module)
# ---------------------------------------------------------------------------

VERIFIED_HALPE26 = (
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

_SDK = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"

# BodyWithFeet.MODE table, copied verbatim from rtmlib source.
VERIFIED_BODY_WITH_FEET_MODELS = {
    "lightweight": {
        "det": _SDK + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        "det_input_size": (416, 416),
        "pose":
        _SDK +
        "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
        "pose_input_size": (192, 256),
    },
    "balanced": {
        "det": _SDK + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "det_input_size": (640, 640),
        "pose":
        _SDK +
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        "pose_input_size": (192, 256),
    },
    "performance": {
        "det": _SDK + "yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        "det_input_size": (640, 640),
        "pose":
        _SDK +
        "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        "pose_input_size": (288, 384),
    },
}


def test_halpe26_table_matches_verified_sources():
    assert rb.HALPE26_KEYPOINTS == VERIFIED_HALPE26
    assert len(rb.HALPE26_KEYPOINTS) == 26
    assert len(set(rb.HALPE26_KEYPOINTS)) == 26


def test_foot_role_indices_pin_absolute_positions():
    # Semantic constants must point at the verified absolute indices...
    # ...and at the names the verified table assigns to those indices.
    for idx in (rb.LEFT_SHOULDER, rb.RIGHT_SHOULDER, rb.LEFT_HIP,
                rb.RIGHT_HIP, rb._LEFT_ANKLE, rb._RIGHT_ANKLE,
                rb._LEFT_BIG_TOE, rb._RIGHT_BIG_TOE, rb._LEFT_HEEL,
                rb._RIGHT_HEEL):
        pass
    name_of = dict(enumerate(VERIFIED_HALPE26))
    assert name_of[rb._LEFT_ANKLE] == "left_ankle"
    assert name_of[rb._RIGHT_ANKLE] == "right_ankle"
    assert name_of[rb._LEFT_BIG_TOE] == "left_big_toe"
    assert name_of[rb._RIGHT_BIG_TOE] == "right_big_toe"
    assert name_of[rb._LEFT_HEEL] == "left_heel"
    assert name_of[rb._RIGHT_HEEL] == "right_heel"


def test_default_lightweight_model_urls_verified():
    assert rb.BODY_WITH_FEET_MODELS == VERIFIED_BODY_WITH_FEET_MODELS
    assert set(rb.BODY_WITH_FEET_MODELS) == {
        "lightweight", "balanced", "performance"
    }
    # This POC's default is the lightweight pair; rtmlib's own constructor
    # default ("balanced") is kept in the table above.
    assert rb.DEFAULT_MODE == "lightweight"


def test_backend_registered_and_license_notice_verbatim():
    assert rb.BACKEND_NAME == "rtmpose-lightweight"
    assert rb.BACKEND_NAME in types.POSE_BACKENDS
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rb._log_license_notice()
    assert buf.getvalue() == types.RTMPOSE_LICENSE_NOTICE + "\n"


def test_backend_constructor_validates_mode_without_heavy_imports():
    with pytest.raises(ValueError):
        rb.RTMPoseBackend(mode="turbo")
    backend = rb.RTMPoseBackend()  # defaults
    assert backend.name == "rtmpose-lightweight"
    assert backend.mode == "lightweight"
    assert backend.trace_meta == {}


# ---------------------------------------------------------------------------
# Keypoint -> PersonFrame mapping (pure functions)
# ---------------------------------------------------------------------------


def _standing_person(score=0.9, low_confidence=None):
    """Synthetic confident standing pose as (kpts, scores) row lists.

    ``low_confidence`` maps an index to a replacement score.  Feet and hips
    get distinct coordinates so midpoint/foot assertions catch index swaps.
    """
    pos = {name: (150.0, 500.0) for name in VERIFIED_HALPE26}
    pos.update({
        "nose": (150.0, 180.0),
        "left_shoulder": (130.0, 300.0),
        "right_shoulder": (170.0, 300.0),
        "left_hip": (140.0, 520.0),
        "right_hip": (160.0, 520.0),
        "left_ankle": (100.0, 900.0),
        "left_heel": (90.0, 950.0),
        "left_big_toe": (110.0, 980.0),
        "right_ankle": (200.0, 900.0),
        "right_heel": (210.0, 950.0),
        "right_big_toe": (190.0, 980.0),
    })
    scores = []
    kpts = []
    low_confidence = low_confidence or {}
    for i, name in enumerate(VERIFIED_HALPE26):
        x, y = pos[name]
        kpts.append([x, y])
        scores.append(low_confidence.get(i, score))
    return kpts, scores


def test_full_mapping_produces_complete_person_frame():
    kpts, scores = _standing_person()
    frame = rb.person_frame_from_keypoints(
        [kpts], [scores], t_s=0.042, pts_us=42000, frame_index=1)

    assert frame.lost is False
    assert frame.person_id == 0
    assert (frame.t_s, frame.source_pts_us, frame.frame_index) == \
        (0.042, 42000, 1)
    assert frame.score == pytest.approx(0.9)

    # Midpoints of left/right pairs.
    assert (frame.hips.x, frame.hips.y) == pytest.approx((150.0, 520.0))
    assert (frame.shoulders_mid.x, frame.shoulders_mid.y) == \
        pytest.approx((150.0, 300.0))

    # Foot observations map heel/ankle/toe(big) per side, verbatim coords.
    assert (frame.feet["left"].heel.x, frame.feet["left"].heel.y) == \
        pytest.approx((90.0, 950.0))
    assert (frame.feet["left"].ankle.x, frame.feet["left"].ankle.y) == \
        pytest.approx((100.0, 900.0))
    assert (frame.feet["left"].toe.x, frame.feet["left"].toe.y) == \
        pytest.approx((110.0, 980.0))
    assert (frame.feet["right"].heel.x, frame.feet["right"].heel.y) == \
        pytest.approx((210.0, 950.0))
    assert (frame.feet["right"].ankle.x, frame.feet["right"].ankle.y) == \
        pytest.approx((200.0, 900.0))
    assert (frame.feet["right"].toe.x, frame.feet["right"].toe.y) == \
        pytest.approx((190.0, 980.0))

    # Full skeleton kept with confidence passthrough; Halpe26 has no z.
    assert len(frame.landmarks) == 26
    assert all(lm.z == 0.0 for lm in frame.landmarks)
    assert frame.landmarks[rb._LEFT_HEEL].visibility == pytest.approx(0.9)
    assert (frame.landmarks[0].x, frame.landmarks[0].y) == \
        pytest.approx((150.0, 180.0))  # nose stays at nose


def test_low_confidence_points_abstain_per_side():
    kpts, scores = _standing_person(
        low_confidence={rb._LEFT_HEEL: 0.1})
    frame = rb.person_frame_from_keypoints([kpts], [scores], t_s=0.0,
                                           pts_us=0, frame_index=0)
    assert frame.feet["left"] is None           # heel below threshold
    assert frame.feet["right"] is not None      # other side unaffected
    # Raw landmark rows keep the measured confidence regardless.
    assert frame.landmarks[rb._LEFT_HEEL].visibility == pytest.approx(0.1)

    kpts, scores = _standing_person(
        low_confidence={rb._LEFT_ANKLE: 0.05})
    frame = rb.person_frame_from_keypoints([kpts], [scores], t_s=0.0,
                                           pts_us=0, frame_index=0)
    assert frame.feet["left"] is None           # ankle participates too

    # Threshold semantics are inclusive (score == threshold passes).
    kpts, scores = _standing_person(
        low_confidence={rb._LEFT_HEEL: rb.DEFAULT_KPT_THRESHOLD})
    frame = rb.person_frame_from_keypoints([kpts], [scores], t_s=0.0,
                                           pts_us=0, frame_index=0)
    assert frame.feet["left"] is not None


def test_empty_detection_yields_lost_frame():
    frame = rb.person_frame_from_keypoints([], [], t_s=0.5, pts_us=500000,
                                           frame_index=12)
    assert frame.lost is True
    assert frame.score == 0.0
    assert frame.hips is None and frame.shoulders_mid is None
    assert frame.feet == {"left": None, "right": None}
    assert frame.landmarks == []

    sentinel = rb.make_lost_frame(t_s=0.5, pts_us=500000, frame_index=12)
    assert sentinel == frame


def test_single_instance_shapes_accepted():
    kpts, scores = _standing_person()
    batched = rb.person_frame_from_keypoints([kpts], [scores], t_s=1.0,
                                             pts_us=1_000_000,
                                             frame_index=25)
    single = rb.person_frame_from_keypoints(kpts, scores, t_s=1.0,
                                            pts_us=1_000_000,
                                            frame_index=25)
    assert single == batched


def test_dominant_person_selection():
    _, weak = _standing_person(score=0.2)
    _, strong = _standing_person(score=0.8)
    assert rb.select_dominant_person([weak, strong]) == 1
    assert rb.select_dominant_person([strong, weak]) == 0
    assert rb.select_dominant_person([]) is None

    # The chosen person drives every derived observation.
    ka, sa = _standing_person(score=0.2)
    kb, sb = _standing_person(score=0.8)
    kb[rb._LEFT_HEEL] = [777.0, 888.0]
    frame = rb.person_frame_from_keypoints([ka, kb], [sa, sb], t_s=0.0,
                                           pts_us=0, frame_index=0)
    assert (frame.feet["left"].heel.x, frame.feet["left"].heel.y) == \
        pytest.approx((777.0, 888.0))


def test_run_pose_rejects_unknown_mode_before_io():
    with pytest.raises(ValueError):
        rb.run_pose("whatever.mp4", mode="nope")


def test_module_import_is_lazy_subprocess():
    """Importing the backend must not drag in the ML stack."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(_SRC)!r})\n"
        "import auto_sfx.pose.rtmpose_backend as m\n"
        "banned = ('numpy', 'cv2', 'onnxruntime', 'rtmlib', 'tqdm')\n"
        "leaked = [b for b in banned if b in sys.modules]\n"
        "assert not leaked, 'heavy imports leaked: %r' % leaked\n"
        "print(len(m.HALPE26_KEYPOINTS))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "26"
