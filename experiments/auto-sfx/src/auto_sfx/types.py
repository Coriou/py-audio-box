"""Shared vocabulary for the auto-sfx POC.

This module is the single source of truth for cross-module data shapes.
Backend-specific landmark index tables live in their backend modules, NOT
here — this file only defines semantic roles (heel / ankle / toe, hips,
shoulders).

Coordinate convention
---------------------
All pixel-space coordinates are floats in *decoded frame pixel* units
(x in [0, width), y in [0, height)), already scaled from any normalized
model output. Visibility/confidence values are floats in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEFT = "left"
RIGHT = "right"
SIDES = (LEFT, RIGHT)

EVENT_FAMILY_FOOT_CONTACT = "locomotion.foot_contact"
DOMINANT_ACTOR_ID = "person_01"

CONTACT_METHODS = ("velocity", "pelvis", "hybrid")
POSE_BACKENDS = (
    "mediapipe-lite",
    "mediapipe-full",
    "rtmpose-lightweight",
)
RUNNING_MODES = ("video", "image")

# Printed verbatim whenever a non-production-cleared model is used.
RTMPOSE_LICENSE_NOTICE = "RESEARCH POC ONLY - PRODUCTION LICENSE NOT CLEARED"

# Cache format version; bump to invalidate every cached pose trace.
POSE_CACHE_PREPROCESS_VERSION = 1


class DiagReason:
    """Registry of abstention/failure reasons (diagnostics.json)."""

    FEET_OUT_OF_FRAME = "feet_out_of_frame"
    HEEL_LOW_CONFIDENCE = "heel_low_confidence"
    TOE_LOW_CONFIDENCE = "toe_low_confidence"
    POSE_LOST = "pose_lost"
    ACTOR_TOO_SMALL = "actor_too_small"
    CAMERA_MOTION_HIGH = "camera_motion_high"
    AMBIGUOUS_SUPPORT_PHASE = "ambiguous_support_phase"
    LEFT_RIGHT_DISCONTINUITY = "left_right_discontinuity"
    SHOT_BOUNDARY = "shot_boundary"
    CADENCE_VIOLATION = "cadence_violation"

    ALL = (
        FEET_OUT_OF_FRAME,
        HEEL_LOW_CONFIDENCE,
        TOE_LOW_CONFIDENCE,
        POSE_LOST,
        ACTOR_TOO_SMALL,
        CAMERA_MOTION_HIGH,
        AMBIGUOUS_SUPPORT_PHASE,
        LEFT_RIGHT_DISCONTINUITY,
        SHOT_BOUNDARY,
        CADENCE_VIOLATION,
    )


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------


@dataclass
class VideoMeta:
    path: str
    codec: str
    width: int
    height: int
    fps_stream: float          # container-reported average frame rate
    n_frames: int              # actually decoded frame count
    duration_s: float          # last PTS - first PTS (+ last frame duration)
    cfr: bool                  # constant frame rate within tolerance
    vfr: bool                  # variable frame rate detected
    time_base: float | None    # stream time base in seconds, if known
    audio_present: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Pose traces
# ---------------------------------------------------------------------------


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


def point_dist(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


@dataclass
class FootObservation:
    heel: Point | None
    ankle: Point | None
    toe: Point | None          # foot index / big toe


@dataclass
class PersonFrame:
    """One person's pose at one decoded frame."""

    t_s: float                 # source-accurate PTS in seconds
    source_pts_us: int         # same instant as integer microseconds
    frame_index: int           # decode order, 0-based
    person_id: int             # dominant actor slot (always 0 in POC v0)
    score: float               # overall pose confidence for this frame
    feet: dict                 # side -> FootObservation | None
    hips: Point | None         # midpoint of left/right hip
    shoulders_mid: Point | None
    landmarks: list[Point] = field(default_factory=list)  # full skeleton, px
    lost: bool = False         # no usable person detected this frame


@dataclass
class PoseTrace:
    backend: str               # one of POSE_BACKENDS
    mode: str                  # "video" | "image"
    meta: VideoMeta
    frames: list[PersonFrame]

    def to_rows(self) -> list[dict]:
        """Compact row dicts for jsonl caching."""

        def pt(p: Point | None) -> list[float] | None:
            if p is None:
                return None
            return [p.x, p.y, p.z, p.visibility]

        rows = []
        for f in self.frames:
            rows.append(
                {
                    "t_s": f.t_s,
                    "pts_us": f.source_pts_us,
                    "i": f.frame_index,
                    "score": f.score,
                    "lost": f.lost,
                    "hips": pt(f.hips),
                    "sho": pt(f.shoulders_mid),
                    "feet": {
                        s: None if f.feet.get(s) is None
                        else {
                            "heel": pt(f.feet[s].heel),
                            "ankle": pt(f.feet[s].ankle),
                            "toe": pt(f.feet[s].toe),
                        }
                        for s in SIDES
                    },
                    "lm": [pt(p) for p in f.landmarks],
                }
            )
        return rows

    @classmethod
    def from_rows(
        cls, backend: str, mode: str, meta: VideoMeta, rows: list[dict]
    ) -> "PoseTrace":
        def mkpt(v):
            if v is None:
                return None
            return Point(v[0], v[1], v[2], v[3])

        frames = []
        for r in rows:
            feet = {}
            for s in SIDES:
                raw = (r.get("feet") or {}).get(s)
                if raw is None:
                    feet[s] = None
                else:
                    feet[s] = FootObservation(
                        heel=mkpt(raw.get("heel")),
                        ankle=mkpt(raw.get("ankle")),
                        toe=mkpt(raw.get("toe")),
                    )
            frames.append(
                PersonFrame(
                    t_s=r["t_s"],
                    source_pts_us=r["pts_us"],
                    frame_index=r["i"],
                    person_id=0,
                    score=r["score"],
                    feet=feet,
                    hips=mkpt(r.get("hips")),
                    shoulders_mid=mkpt(r.get("sho")),
                    landmarks=[mkpt(v) for v in (r.get("lm") or [])],
                    lost=r.get("lost", False),
                )
            )
        return cls(backend=backend, mode=mode, meta=meta, frames=frames)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class ContactEvent:
    id: str
    family: str                       # EVENT_FAMILY_FOOT_CONTACT
    actor_id: str                     # DOMINANT_ACTOR_ID
    foot: str                         # LEFT | RIGHT
    source_pts_us: int                # exact source PTS, microseconds
    time_s: float
    earliest_s: float
    latest_s: float
    score: float
    observable: bool
    detector: str                     # e.g. "mediapipe-full-contact-hybrid"
    method: str                       # velocity | pelvis | hybrid
    evidence: dict                    # free-form per-method measurements
    kind: str = "initial_contact"     # initial_contact | toe_off

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Small shared helpers used by several modules
# ---------------------------------------------------------------------------


def pts_us_to_s(pts_us: int) -> float:
    return pts_us / 1_000_000.0


def s_to_pts_us(t_s: float) -> int:
    return int(round(t_s * 1_000_000))
