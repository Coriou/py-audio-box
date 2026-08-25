"""Unit tests: foot-contact / gait detection engine (auto_sfx.contacts).

Synthetic PoseTrace objects are built programmatically from the same gait
phase math the fixture renderer uses -- ``scripts/make_fixtures.py`` is
imported for its pure functions (no cv2 at import time): stance pinned at
the contact pixel, swing a sinusoidal arc, hip->ankle scale ~95 px.
Ground truth: left strikes at t = k*0.6 s, right at t = k*0.6 + 0.3 s.

Boundary policy (mirrors the engine's documented abstention rule): strikes
whose transition lies before the trace start (t=0 left) or whose stance
cannot be observed for the confirmation horizon inside the trace are
excluded from expectations -- abstaining there is correct behaviour, not a
miss.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_fixtures as fx  # noqa: E402  (pure gait math, lazy cv2)

from auto_sfx import contacts as contacts_mod  # noqa: E402
from auto_sfx.contacts import (  # noqa: E402
    ContactMethodResult,
    default_config,
    detect_contacts,
)
from auto_sfx.types import (  # noqa: E402
    DiagReason,
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
    VideoMeta,
)

FPS = 30.0
DT = 1.0 / FPS
TOL_FRAME = DT + 1e-9          # +/- 1 frame
LEFT, RIGHT = "left", "right"

# Heel/toe geometry relative to the ankle (image px; y grows downward).
# Heel sits 2 px below the toe line => heel-first strikes for the hybrid
# evidence["strike"] label.
HEEL_OFF = (-4.0, 2.0)
TOE_OFF = (7.0, 0.0)


# ---------------------------------------------------------------------------
# Synthetic trace builders
# ---------------------------------------------------------------------------


def make_meta(times: list[float], vfr: bool = False) -> VideoMeta:
    last_dt = times[-1] - times[-2] if len(times) > 1 else DT
    return VideoMeta(
        path="synthetic://walk",
        codec="synth",
        width=fx.WIDTH,
        height=fx.HEIGHT,
        fps_stream=30.0,
        n_frames=len(times),
        duration_s=times[-1] + last_dt,
        cfr=not vfr,
        vfr=vfr,
        time_base=None,
        audio_present=False,
    )


def _pt(x: float, y: float, vis: float = 0.95) -> Point:
    return Point(x=float(x), y=float(y), z=0.0, visibility=float(vis))


def _ankle(view: str, t: float, side: str) -> tuple[float, float]:
    if view == "lateral":
        return fx._lateral_ankle(t, side)
    return fx._frontal_ankle(t, side, fx.WIDTH / 2.0)


def build_walk(
    times: list[float],
    view: str = "lateral",
    backend: str = "mediapipe-lite",
    feet_missing=None,   # callable(t, side) -> bool
    heel_vis=None,       # callable(t) -> float  (heel visibility override)
    lost=None,           # callable(t) -> bool
    dy_hooks=None,       # dict side -> callable(t) -> dy px (ankle y offset)
) -> PoseTrace:
    feet_missing = feet_missing or (lambda t, side: False)
    heel_vis = heel_vis or (lambda t: 0.95)
    lost = lost or (lambda t: False)
    dy_hooks = dy_hooks or {}

    def pelvis(t: float) -> Point:
        px, py = (fx._lateral_pelvis(t) if view == "lateral"
                  else fx._frontal_pelvis(t))
        return _pt(px, py, 0.99)

    frames = []
    for i, t in enumerate(times):
        is_lost = bool(lost(t))
        feet = {}
        for side in (LEFT, RIGHT):
            if is_lost or feet_missing(t, side):
                feet[side] = None
                continue
            ax, ay = _ankle(view, t, side)
            ay += float(dy_hooks.get(side, lambda _: 0.0)(t))
            hv = float(heel_vis(t))
            feet[side] = FootObservation(
                heel=_pt(ax + HEEL_OFF[0], ay + HEEL_OFF[1], hv),
                ankle=_pt(ax, ay, 0.95),
                toe=_pt(ax + TOE_OFF[0], ay + TOE_OFF[1], 0.95),
            )
        frames.append(PersonFrame(
            t_s=t,
            source_pts_us=int(round(t * 1_000_000)),
            frame_index=i,
            person_id=0,
            score=0.4 if is_lost else 0.9,
            feet=feet,
            hips=None if is_lost else pelvis(t),
            shoulders_mid=None,
            landmarks=[],
            lost=is_lost,
        ))
    return PoseTrace(backend=backend, mode="video", meta=make_meta(times), frames=frames)


def uniform_times(duration_s: float) -> list[float]:
    n = int(math.floor(duration_s * FPS + 1e-9))
    return [k / FPS for k in range(n + 1)]


def vfr_times(duration_s: float) -> list[float]:
    """Alternating 1/60 s / 1/20 s frame durations (mean 1/30 s)."""

    times, t = [0.0], 0.0
    k = 0
    while t + (1 / 60.0 if k % 2 == 0 else 1 / 20.0) <= duration_s + 1e-9:
        t += 1 / 60.0 if k % 2 == 0 else 1 / 20.0
        times.append(t)
        k += 1
    return times


def analytic_strikes(t_last: float) -> list[tuple[float, str]]:
    """Formula ground truth, minus boundary contacts the engine must skip.

    Lower cut: a strike is detectable only if its swing onset is inside the
    trace (t >= CYCLE/2).  Upper cut: its stance must remain observable for
    the confirmation horizon (min stance duration 0.45 * 0.25 s ~= 0.113 s;
    0.15 s is a conservative margin).
    """

    lo = fx.CYCLE / 2.0 - 1e-9
    hi = t_last - 0.15 + 1e-9
    return [(e["time_s"], e["foot"]) for e in fx.contact_events(t_last)
            if lo <= e["time_s"] <= hi]


def assert_matches(events, expected: list[tuple[float, str]], tol: float) -> None:
    """Bijective nearest-match assertion, per foot, with no extras."""

    assert len(events) == len(expected), (
        f"event count {len(events)} != expected {len(expected)}: "
        f"{[(round(e.time_s, 3), e.foot) for e in events]}"
    )
    for side in (LEFT, RIGHT):
        got = sorted(e.time_s for e in events if e.foot == side)
        want = sorted(t for t, f in expected if f == side)
        assert len(got) == len(want), f"{side}: {got} vs {want}"
        pool = list(got)
        for t in want:
            best = min(pool, key=lambda g: abs(g - t)) if pool else None
            assert best is not None and abs(best - t) <= tol, (
                f"{side}: no detection within {tol:.4f}s of {t} "
                f"(have {[round(g, 3) for g in pool]})"
            )
            pool.remove(best)


def assert_diag_schema(result: ContactMethodResult) -> None:
    for d in result.diagnostics:
        assert set(d) == {"code", "t0_s", "t1_s", "detail"}
        assert d["code"] in DiagReason.ALL
        if d["t0_s"] is not None and d["t1_s"] is not None:
            assert d["t0_s"] <= d["t1_s"] + 1e-9


# ---------------------------------------------------------------------------
# 1. velocity + hybrid recover every analytic strike
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["velocity", "hybrid"])
def test_velocity_and_hybrid_detect_all_analytic_strikes(method):
    times = uniform_times(8.4)
    trace = build_walk(times)
    result = detect_contacts(trace, method)

    assert isinstance(result, ContactMethodResult)
    assert_diag_schema(result)
    assert_matches(result.events, analytic_strikes(8.4), TOL_FRAME)

    # Event metadata contract.
    for rank, ev in enumerate(result.events, start=1):
        assert ev.id == f"evt_{rank:03d}"
        assert ev.family == "locomotion.foot_contact"
        assert ev.actor_id == "person_01"
        assert ev.kind == "initial_contact"
        assert ev.method == method
        assert ev.detector == f"mediapipe-lite-contact-{method}-v0"
        assert ev.source_pts_us == int(round(ev.time_s * 1_000_000))
        assert 0.0 <= ev.score <= 1.0
        assert ev.earliest_s <= ev.time_s < ev.latest_s
        # Engine rounds times to 6 decimals: allow rounding slack on top of
        # one source-frame uncertainty.
        assert ev.latest_s - ev.earliest_s <= DT + 2e-6
        assert ev.observable is True
        assert "heel_vis" in ev.evidence and "toe_vis" in ev.evidence

    # Sorted by time ascending.
    ts = [ev.time_s for ev in result.events]
    assert ts == sorted(ts)

    if method == "hybrid":
        # Flat synthetic foot with heel 2 px below toe -> heel-first labels.
        assert all(ev.evidence.get("strike") == "heel" for ev in result.events)
        assert all("height_above_band_px" in ev.evidence for ev in result.events)


# ---------------------------------------------------------------------------
# 2. pelvis method on the lateral walker
# ---------------------------------------------------------------------------


def test_pelvis_detects_lateral_walk_strikes():
    times = uniform_times(8.4)
    trace = build_walk(times)
    result = detect_contacts(trace, "pelvis")

    assert_diag_schema(result)
    # Zero-phase smoothing preserves peak timing to ~1 frame; allow 2.5.
    # The FINAL strike (t=8.1) is excluded by the engine's documented
    # boundary-abstention policy: the post-contact forward-decline
    # confirmation window extends past the trace end, so the candidate
    # cannot be confirmed and is dropped rather than invented.
    assert_matches(result.events, analytic_strikes(8.1), 2.5 * DT)
    for ev in result.events:
        assert "rel_offset_px" in ev.evidence
        assert "walk_dir" in ev.evidence
    # The walker drifts +x: the walk axis must point forward.
    axis = result.events[0].evidence["walk_dir"]
    assert axis[0] > 0.9


# ---------------------------------------------------------------------------
# 3. observability spans: abstention diagnostics, no invented events
# ---------------------------------------------------------------------------


EPS = 1e-9


def test_observability_spans_abstain_and_flag():
    def missing(t, side):
        return 1.5 - EPS <= t <= 2.5 + EPS

    def heel_low(t):
        return 2.6 - EPS <= t <= 3.2 + EPS

    def heel_vis_fn(t):
        return 0.2 if (2.6 - EPS <= t <= 3.2 + EPS) else 0.95

    def lost(t):
        return 3.4 - EPS <= t <= 3.5 + EPS

    trace = build_walk(uniform_times(4.2), feet_missing=missing,
                       heel_vis=heel_vis_fn, lost=lost)
    result = detect_contacts(trace, "hybrid")
    assert_diag_schema(result)

    diags = result.diagnostics
    feet_diags = [d for d in diags if d["code"] == DiagReason.FEET_OUT_OF_FRAME]
    heel_diags = [d for d in diags if d["code"] == DiagReason.HEEL_LOW_CONFIDENCE]
    lost_diags = [d for d in diags if d["code"] == DiagReason.POSE_LOST]
    assert any(d["t0_s"] <= 1.5 + EPS and d["t1_s"] >= 2.5 - EPS for d in feet_diags)
    assert any(d["t0_s"] <= 2.6 + EPS and d["t1_s"] >= 3.2 - EPS for d in heel_diags)
    assert any(d["t0_s"] <= 3.4 + EPS and d["t1_s"] >= 3.5 - EPS for d in lost_diags)

    # Nothing inside any inadequate span.
    for ev in result.events:
        t = ev.time_s
        assert not (1.5 - EPS <= t <= 2.5 + EPS), ev
        assert not (3.4 - EPS <= t <= 3.5 + EPS), ev

    # Expected survivors: analytic strikes minus those whose contact frame
    # falls inside the missing span ({1.5R, 1.8L, 2.1R, 2.4L}) and 3.3R
    # whose stance is truncated by the lost span below the minimum stance
    # duration.  Strikes contacting inside the low-heel span stay but carry
    # observable=False.
    expected = [
        (0.6, LEFT), (1.2, LEFT), (3.0, LEFT), (3.6, LEFT),
        (0.3, RIGHT), (0.9, RIGHT), (2.7, RIGHT), (3.9, RIGHT),
    ]
    assert_matches(result.events, expected, TOL_FRAME)

    # Strikes contacting inside the low-heel span stay, marked unobservable.
    weak = sorted((ev.foot, round(ev.time_s, 3)) for ev in result.events
                  if not ev.observable)
    assert weak == [(LEFT, 3.033), (RIGHT, 2.733)]


# ---------------------------------------------------------------------------
# 4. frontal/degenerate geometry: pelvis abstains
# ---------------------------------------------------------------------------


def test_frontal_degenerate_pelvis_abstains():
    trace = build_walk(uniform_times(3.0), view="frontal")
    result = detect_contacts(trace, "pelvis")

    assert result.events == []
    amb = [d for d in result.diagnostics
           if d["code"] == DiagReason.AMBIGUOUS_SUPPORT_PHASE]
    assert len(amb) == 1
    assert amb[0]["t0_s"] <= 0.0 + EPS
    assert amb[1 - 1]["t1_s"] >= 3.0 - 2 * DT

    # Contrast: the velocity machine still sees the (physically real)
    # in-place plants -- only the pelvis geometry is uninformative.
    velocity = detect_contacts(trace, "velocity")
    assert len(velocity.events) > 0


# ---------------------------------------------------------------------------
# 5. irregular (VFR-like) timestamps shift detections accordingly
# ---------------------------------------------------------------------------


def test_irregular_timestamps_shift_detections():
    times = vfr_times(5.4)
    assert abs((times[-1]) - 5.4) < 0.1
    trace = build_walk(times, backend="rtmpose-lightweight")
    result = detect_contacts(trace, "velocity")

    assert_diag_schema(result)
    # Tolerance: one worst-case local frame (1/20 s), well under half a step.
    assert_matches(result.events, analytic_strikes(times[-1]), 0.05 + 1e-9)
    assert all(ev.detector.startswith("rtmpose-lightweight-contact-")
               for ev in result.events)


# ---------------------------------------------------------------------------
# 6a. long gap => cadence violation diagnostic, events survive on both sides
# ---------------------------------------------------------------------------


def test_long_gap_raises_cadence_violation():
    def missing(t, side):
        return 2.0 - EPS <= t <= 4.5 + EPS

    trace = build_walk(uniform_times(6.0), feet_missing=missing)
    result = detect_contacts(trace, "velocity")
    assert_diag_schema(result)

    violations = [d for d in result.diagnostics
                  if d["code"] == DiagReason.CADENCE_VIOLATION]
    assert violations, "expected a cadence violation across the gap"
    assert any(d["t0_s"] < 2.0 and d["t1_s"] > 4.49
               and (d["t1_s"] - d["t0_s"]) > 2.0 for d in violations)

    # Nothing invented inside the gap; both sides still detected overall.
    for ev in result.events:
        assert not (2.05 <= ev.time_s <= 4.45), ev
    assert {ev.foot for ev in result.events} == {LEFT, RIGHT}
    # Events on both sides of the gap exist.
    assert any(ev.time_s < 2.0 for ev in result.events)
    assert any(ev.time_s > 4.5 for ev in result.events)


# ---------------------------------------------------------------------------
# 6b. near-double plant (bounce) suppressed to a single event
# ---------------------------------------------------------------------------

_BOUNCE_POINTS = [(0.70, 0.0), (0.733, -4.0), (0.767, -8.0), (0.833, -8.0),
                  (0.867, -4.0), (0.900, 0.0)]


def _bounce_dy(t: float) -> float:
    """Piecewise-linear hop of the planted left foot inside [0.70, 0.90]."""

    if t <= _BOUNCE_POINTS[0][0] or t >= _BOUNCE_POINTS[-1][0]:
        return 0.0
    for (t0, y0), (t1, y1) in zip(_BOUNCE_POINTS, _BOUNCE_POINTS[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0)
            return y0 + (y1 - y0) * u
    return 0.0


def test_near_double_plant_suppressed():
    trace = build_walk(uniform_times(3.0), dy_hooks={LEFT: _bounce_dy})
    result = detect_contacts(trace, "velocity")
    assert_diag_schema(result)

    left = [ev for ev in result.events if ev.foot == LEFT]
    in_window = [ev for ev in left if 0.5 <= ev.time_s <= 1.1]
    assert len(in_window) == 1, (
        f"double plant not suppressed: {[round(e.time_s, 3) for e in left]}"
    )
    # The hop merges into the normal rhythm -- no net extra steps.
    assert len(left) == 4
    # Same-foot events never repeat within one minimum step interval.
    for side in (LEFT, RIGHT):
        ts = sorted(ev.time_s for ev in result.events if ev.foot == side)
        assert all(b - a >= 0.25 - 1e-9 for a, b in zip(ts, ts[1:]))


# ---------------------------------------------------------------------------
# 7. unknown method / config keys rejected
# ---------------------------------------------------------------------------


def test_unknown_method_rejected():
    trace = build_walk(uniform_times(1.0))
    with pytest.raises(ValueError):
        detect_contacts(trace, "voodoo")
    with pytest.raises(ValueError):
        default_config("accelerometer")


def test_unknown_config_key_rejected():
    trace = build_walk(uniform_times(1.0))
    with pytest.raises(ValueError):
        detect_contacts(trace, "velocity", config={"nonsense": 1})


def test_default_config_keys():
    cfg = default_config("hybrid")
    for key, value in (
        ("min_step_interval_s", 0.25), ("max_step_interval_s", 2.0),
        ("confirm_frames", 2), ("speed_drop_frac", 0.35),
        ("min_landmark_visibility", 0.5), ("shot_boundaries", []),
        ("actor_too_small_px", 24), ("camera_motion_std_px", 12.0),
    ):
        assert key in cfg


# ---------------------------------------------------------------------------
# 8. shot boundaries reset machines and emit diagnostics
# ---------------------------------------------------------------------------


def test_shot_boundary_resets_and_diagnoses():
    trace = build_walk(uniform_times(4.2))
    result = detect_contacts(trace, "velocity",
                             config={"shot_boundaries": [2.0]})
    assert_diag_schema(result)
    # Every analytic strike is still detected: a strike whose stance merely
    # TAILS into the cut (1.8L) has its touchdown and 0.2 s of observable
    # stance inside segment 1; only contacts planted BEFORE the cut with no
    # observable transition are abstained from.
    assert_matches(result.events, analytic_strikes(4.2), TOL_FRAME)


# ---------------------------------------------------------------------------
# 9. degenerate traces abstain cleanly
# ---------------------------------------------------------------------------


def test_degenerate_traces_abstain_cleanly():
    blank = build_walk([0.0, DT, 2 * DT], lost=lambda t: True)
    result = detect_contacts(blank, "hybrid")
    assert result.events == []
    assert result.diagnostics
    assert_diag_schema(result)

    empty = PoseTrace(backend="mediapipe-lite", mode="video",
                      meta=make_meta([0.0, DT]), frames=[])
    result = detect_contacts(empty, "velocity")
    assert result.events == []
    assert result.diagnostics == []


# ---------------------------------------------------------------------------
# 10. determinism: identical input, identical output
# ---------------------------------------------------------------------------


def test_detect_is_deterministic():
    trace = build_walk(uniform_times(3.0))
    a = detect_contacts(trace, "hybrid")
    b = detect_contacts(trace, "hybrid")
    assert [e.to_dict() for e in a.events] == [e.to_dict() for e in b.events]
    assert a.diagnostics == b.diagnostics
