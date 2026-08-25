#!/usr/bin/env python3
"""Synthetic fixtures for the auto-sfx POC (run inside the container).

    python scripts/make_fixtures.py

Writes to /work/auto-sfx/fixtures/:

* walk_synthetic.mp4            12 s lateral stick-figure walker, 720x480@30,
                                deterministic gait; heel strikes alternate
                                left/right at t = k*T/2 (+T/4 offset for the
                                right foot), stride period T = 1.2 s.
* walk_synthetic_contacts.json  ground-truth foot-contact events, emitted
                                straight from that formula.
* vfr_walk.mp4                  the same frames re-encoded with alternating
                                frame durations 1/60 s / 1/20 s via the concat
                                demuxer (VFR decode-fidelity fixture).
* vfr_walk_expected_pts.json    cumulative intended PTS per frame, for
                                decode-fidelity checks.
* frontal_walk.mp4              8 s camera-facing walker (feet move mostly
                                vertically, minimal pelvis translation); used
                                later to prove pelvis-method honest abstention.

Fully deterministic: no RNG anywhere. cv2/numpy are imported lazily inside
the rendering functions so the gait math itself is importable and testable
on a bare python.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path("/work/auto-sfx/fixtures")

WIDTH, HEIGHT, FPS = 720, 480, 30.0

# --- gait model -----------------------------------------------------------
# Heel strikes alternate L/R every STRIDE_T/2 seconds: the left foot strikes
# at t = k*(STRIDE_T/2), the right at t = k*(STRIDE_T/2) + STRIDE_T/4. Each
# foot therefore repeats with cycle CYCLE = STRIDE_T/2: first half is stance
# (the contact point stays pinned at a fixed pixel position), second half is
# swing (smooth arc to the next contact position with sinusoidal lift).
STRIDE_T = 1.2
CYCLE = STRIDE_T / 2.0
RIGHT_OFFSET = STRIDE_T / 4.0

GROUND_Y = 430            # ground line (pixel row)
HIP_Y = 300               # nominal pelvis centre row
TORSO_H = 150             # torso length in px
HEAD_R = 22
THIGH_L, SHIN_L = 72, 62
ANKLE_UP = 10             # ankle sits this many px above the ground line
SWING_LIFT = 34           # peak swing-foot lift in px
STEP_DRIFT = 14.0         # forward drift per strike; 20 strikes stay in frame
BASE_X = 110.0            # x of the first (left-foot) contact
BODY_SWAY = 3.5           # fore-aft / vertical bob amplitude in px
ARM_SWING = 20.0

BG_COLOR = (72, 74, 80)        # mid-gray backdrop: enough contrast without
FG_COLOR = (238, 238, 240)     # crushing to pure black, which BlazePose's
FAR_COLOR = (168, 172, 180)    # person detector handles poorly
FLOOR_COLOR = (56, 58, 64)
LIMB_THICKNESS = 16            # capsule limbs: pose detectors need limb MASS
TORSO_THICKNESS = 26
FOOT_THICKNESS = 12


# ---------------------------------------------------------------------------
# Gait math (pure python -- no cv2 needed)
# ---------------------------------------------------------------------------


def contact_events(duration_s: float) -> list[dict]:
    """Ground-truth contacts, straight from the strike formula.

    Left heel strikes at t = k*CYCLE, right at t = k*CYCLE + RIGHT_OFFSET,
    for every strike instant within [0, duration_s].
    """
    events: list[tuple[float, str]] = []
    k = 0
    while k * CYCLE <= duration_s:
        events.append((round(k * CYCLE, 6), "left"))
        k += 1
    k = 0
    while k * CYCLE + RIGHT_OFFSET <= duration_s:
        events.append((round(k * CYCLE + RIGHT_OFFSET, 6), "right"))
        k += 1
    events.sort()
    return [{"time_s": t, "foot": side} for t, side in events]


def _gait_phase(t: float, offset: float) -> tuple[float, int]:
    """Return (phase in [0, 1), strike index j) for one foot.

    phase < 0.5 -> stance (pinned contact), phase >= 0.5 -> swing.
    """
    tau = t - offset
    j = math.floor(tau / CYCLE + 1e-9)
    ph = (tau - j * CYCLE) / CYCLE
    return ph, int(j)


def _contact_x(strike_index: int, side_bit: int) -> float:
    """Pinned x of a given contact; global strikes advance rightward."""
    return BASE_X + (2 * strike_index + side_bit) * STEP_DRIFT


def _lateral_ankle(t: float, side: str) -> tuple[float, float]:
    """Ankle position of one leg of the lateral (side-view) walker."""
    side_bit = 0 if side == "left" else 1
    offset = 0.0 if side == "left" else RIGHT_OFFSET
    ph, j = _gait_phase(t, offset)
    x_hit = _contact_x(j, side_bit)
    if ph < 0.5:
        # stance: contact point pinned at its fixed pixel position
        return x_hit, GROUND_Y - ANKLE_UP
    # swing: smooth forward travel to the next contact + sinusoidal lift
    u = (ph - 0.5) * 2.0
    x_next = _contact_x(j + 1, side_bit)
    return (
        x_hit + (x_next - x_hit) * u,
        GROUND_Y - ANKLE_UP - SWING_LIFT * math.sin(math.pi * u),
    )


def _frontal_ankle(
    t: float, side: str, centre_x: float
) -> tuple[float, float]:
    """Ankle position of one leg of the frontal (camera-facing) walker.

    Feet move mostly vertically; only a few pixels of inward shuffle so the
    pelvis-method has as little horizontal foot motion as possible to cheat
    with.
    """
    side_sign = -1.0 if side == "left" else 1.0
    offset = 0.0 if side == "left" else RIGHT_OFFSET
    ph, _j = _gait_phase(t, offset)
    x0 = centre_x + side_sign * 16.0
    if ph < 0.5:
        return x0, GROUND_Y - ANKLE_UP
    u = (ph - 0.5) * 2.0
    x = x0 - side_sign * 3.0 * math.sin(math.pi * u)
    return x, GROUND_Y - ANKLE_UP - SWING_LIFT * math.sin(math.pi * u)


def _lateral_pelvis(t: float) -> tuple[float, float]:
    """Pelvis centre: slow left->right drift plus small step-frequency sway."""
    drift = 2.0 * STEP_DRIFT * (t / CYCLE)  # two strikes per cycle
    x = (
        BASE_X
        + 8.0
        + drift
        + BODY_SWAY * math.sin(2.0 * math.pi * t / (CYCLE / 2.0))
    )
    y = HIP_Y - BODY_SWAY * math.sin(2.0 * math.pi * t / (CYCLE / 2.0))
    return x, y


def _frontal_pelvis(t: float) -> tuple[float, float]:
    """Frontal pelvis: minimal horizontal motion, slight vertical bob."""
    x = WIDTH / 2.0 + 2.0 * math.sin(2.0 * math.pi * t / STRIDE_T)
    y = HIP_Y + BODY_SWAY * math.sin(2.0 * math.pi * t / (2.0 * CYCLE))
    return x, y


def _knee_joint(
    hip: tuple[float, float],
    ankle: tuple[float, float],
    l1: float = THIGH_L,
    l2: float = SHIN_L,
    dir_sign: float = 1.0,
) -> tuple[float, float]:
    """Two-bone IK knee between hip and ankle.

    Picks the bend candidate toward ``dir_sign * x``: knees point forward
    (+x) for the lateral walker and bow outward (away from the centreline)
    for the frontal one.
    """
    ax, ay = hip
    bx, by = ankle
    dx, dy = bx - ax, by - ay
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        dx, dy, dist = 0.0, 1.0, 1.0
    ux, uy = dx / dist, dy / dist
    reach = min(max(dist, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    a = (l1 * l1 - l2 * l2 + reach * reach) / (2.0 * reach)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    mx, my = ax + ux * a, ay + uy * a
    cands = ((mx - uy * h, my + ux * h), (mx + uy * h, my - ux * h))
    return max(cands, key=lambda k: dir_sign * k[0])


# ---------------------------------------------------------------------------
# Rendering (cv2/numpy imported lazily)
# ---------------------------------------------------------------------------


def _pt(p: tuple[float, float]) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def _capsule(img, p0, p1, color, thickness: int) -> None:
    """Thick line with round joint caps so limbs read as body mass."""
    import cv2

    a, b = _pt(p0), _pt(p1)
    cv2.line(img, a, b, color, thickness)
    r = thickness // 2
    cv2.circle(img, a, r, color, -1)
    cv2.circle(img, b, r, color, -1)


def _limb(img, p0, p1, p2, color) -> None:
    _capsule(img, p0, p1, color, LIMB_THICKNESS)
    _capsule(img, p1, p2, color, LIMB_THICKNESS)


def _foot(img, ankle, color, forward: float) -> None:
    toe = (ankle[0] + forward, min(ankle[1] + ANKLE_UP, GROUND_Y))
    _capsule(img, ankle, toe, color, FOOT_THICKNESS)


def _upper_body(img, pelvis, color=FG_COLOR):
    """Filled torso + head; return the neck point."""
    import cv2

    neck = (pelvis[0], pelvis[1] - TORSO_H + 6)
    _capsule(img, pelvis, neck, color, TORSO_THICKNESS)
    head_c = (neck[0], neck[1] - HEAD_R - 8)
    cv2.circle(img, _pt(head_c), HEAD_R, color, -1)
    return neck


def _scene_bg(img) -> None:
    """Backdrop + floor line, drawn before the walker."""
    import cv2
    import numpy as np

    img[:] = BG_COLOR
    cv2.line(img, (0, GROUND_Y + 4), (img.shape[1] - 1, GROUND_Y + 4),
             FLOOR_COLOR, 3)
    _ = np


def _draw_lateral(img, t: float) -> None:
    """One frame of the side-view walker, facing/creeping right."""
    import cv2

    _scene_bg(img)
    pelvis = _lateral_pelvis(t)
    neck = _upper_body(img, pelvis)

    # arms, counter-phase to the same-side leg
    shoulder = (neck[0], neck[1] + 4)
    for side, color in (("right", FAR_COLOR), ("left", FG_COLOR)):
        off = 0.0 if side == "left" else RIGHT_OFFSET
        swing = math.sin(2.0 * math.pi * (t - off) / CYCLE + math.pi)
        elbow = (shoulder[0] + 0.45 * ARM_SWING * swing, shoulder[1] + 30)
        hand = (shoulder[0] + ARM_SWING * swing, shoulder[1] + 58)
        _limb(img, shoulder, elbow, hand, color)

    # legs: far (right) dimmer, near (left) bright
    for side, color in (("right", FAR_COLOR), ("left", FG_COLOR)):
        hx = pelvis[0] - 3 if side == "right" else pelvis[0] + 3
        hip = (hx, pelvis[1] + 4)
        ankle = _lateral_ankle(t, side)
        knee = _knee_joint(hip, ankle, dir_sign=1.0)  # knees point forward
        _limb(img, hip, knee, ankle, color)
        _foot(img, ankle, color, forward=13.0)
    _ = cv2  # keep the lazy import meaningful even on empty-frame callers


def _draw_frontal(img, t: float) -> None:
    """One frame of the camera-facing walker."""
    import cv2

    _scene_bg(img)
    pelvis = _frontal_pelvis(t)
    neck = _upper_body(img, pelvis)

    shoulder = (neck[0], neck[1] + 4)
    for side, color in (("right", FAR_COLOR), ("left", FG_COLOR)):
        off = 0.0 if side == "left" else RIGHT_OFFSET
        swing = math.sin(2.0 * math.pi * (t - off) / CYCLE + math.pi)
        sx = 10.0 if side == "left" else -10.0
        elbow = (shoulder[0] + sx + 3.0 * swing, shoulder[1] + 30)
        hand = (shoulder[0] + sx * 1.7 + 5.0 * swing, shoulder[1] + 56)
        _limb(img, shoulder, elbow, hand, color)

    for side, color in (("right", FAR_COLOR), ("left", FG_COLOR)):
        side_sign = -1.0 if side == "left" else 1.0
        hip = (pelvis[0] + side_sign * 9.0, pelvis[1] + 4)
        ankle = _frontal_ankle(t, side, pelvis[0])
        knee = _knee_joint(hip, ankle, dir_sign=side_sign)  # bow outward
        _limb(img, hip, knee, ankle, color)
        _foot(img, ankle, color, forward=0.0)


def _render_walker(out_path: Path, duration_s: float, draw_fn) -> Path:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")
    n_frames = int(round(duration_s * FPS))
    try:
        for i in range(n_frames):
            img = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)
            draw_fn(img, i / FPS)
            writer.write(img)
    finally:
        writer.release()
    return out_path


# ---------------------------------------------------------------------------
# VFR re-encode
# ---------------------------------------------------------------------------


def _dump_frames(source: Path, frames_dir: Path) -> int:
    """Re-decode `source` into numbered PNGs; return the frame count."""
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"could not reopen {source} for frame dump")
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frames_dir / f"f{count:05d}.png"), frame)
            count += 1
    finally:
        cap.release()
    return count



def render_vfr_walk(dest: Path, source: Path) -> list[float]:
    """Re-encode `source` with alternating frame durations 1/60 s / 1/20 s.

    Frames are decoded with OpenCV and written with PyAV so every frame's
    PTS is assigned exactly (time base 1/90000): frame N displays at
    ``floor(N/2)/15 + mod(N,2)/60`` seconds (pair period 1/60 + 1/20 =
    1/15 s; the odd member of each pair sits 1/60 s after it). Two
    ffmpeg-CLI approaches were tried first and both silently produced flat
    CFR on this Debian 5.1.9 build: an ffconcat list with per-entry
    ``duration`` directives, and a closed-form ``setpts`` retime (this
    build evaluates ``N`` as constant zero). Assigning ``frame.pts``
    directly removes the toolchain from the equation. Returns the intended
    PTS per frame in seconds.
    """
    from fractions import Fraction

    import av
    import cv2
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"could not reopen {source} for VFR re-encode")
    bgr_frames: list = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bgr_frames.append(frame)
    finally:
        cap.release()
    if not bgr_frames:
        raise RuntimeError(f"no frames decoded from {source}")

    n = len(bgr_frames)
    durations = [1.0 / 60.0 if i % 2 == 0 else 1.0 / 20.0 for i in range(n)]
    times: list[float] = []
    acc = 0.0
    for d in durations:
        times.append(round(acc, 9))
        acc += d

    tb = Fraction(1, 90000)
    height, width = bgr_frames[0].shape[:2]
    with av.open(str(dest), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.time_base = tb
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        stream.options = {"crf": "18"}
        for bgr, t_s in zip(bgr_frames, times):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = int(round(t_s * 90000))
            frame.time_base = tb
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return times


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    walk = _render_walker(FIXTURE_DIR / "walk_synthetic.mp4", 12.0, _draw_lateral)
    contacts_path = FIXTURE_DIR / "walk_synthetic_contacts.json"
    contacts_path.write_text(json.dumps(contact_events(12.0), indent=2) + "\n")

    vfr = FIXTURE_DIR / "vfr_walk.mp4"
    expected_pts = render_vfr_walk(vfr, walk)
    expected_path = FIXTURE_DIR / "vfr_walk_expected_pts.json"
    expected_path.write_text(json.dumps(expected_pts) + "\n")

    frontal = _render_walker(FIXTURE_DIR / "frontal_walk.mp4", 8.0, _draw_frontal)

    for p in (walk, contacts_path, vfr, expected_path, frontal):
        print(str(p.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
