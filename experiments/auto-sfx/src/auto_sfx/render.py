"""Annotated diagnostic video renderer for the auto-sfx POC.

Consumes decoded frames from an external iterator (decoupled from ingest),
overlays pose/contact diagnostics per frame, writes a diagnostic MP4 and
optionally muxes a marker-click WAV (see markers.py) into it via ffmpeg.

Per-frame overlays
------------------
- skeleton edges between landmark points (generic COCO-17 table by default;
  callers may pass backend-specific edge pairs)
- per-foot state visuals: contact ring / swing ring / low-confidence X / lost
- overlay panels: run HUD (top-left), per-side foot chips (top-right),
  legend + license notice (bottom-left)
- contact banners along the bottom whenever the current time falls inside an
  event's [earliest_s, latest_s] window
- scrolling heel-speed graph (bottom-right), one coloured series per side

`import cv2` happens lazily inside functions so importing this module never
pays the cv2 load cost.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import (
    LEFT,
    RIGHT,
    SIDES,
    ContactEvent,
    PersonFrame,
    Point,
    PoseTrace,
    point_dist,
)

# Phase-2 repair: the truth strip pairs predictions against GT through the
# shared one-to-one diagnostic association (never a private nearest-match),
# and predicted phase chips come from persisted detector state.
try:  # package-relative when imported as auto_sfx.render
    from .association import associate_diagnostic
except ImportError:  # pragma: no cover - direct module import
    from association import associate_diagnostic  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Visual constants (BGR)
# ---------------------------------------------------------------------------

COL_LEFT = (80, 220, 80)          # green
COL_RIGHT = (40, 130, 255)        # orange
COL_TEXT = (240, 240, 240)
COL_DIM = (170, 170, 170)
COL_PANEL_BG = (28, 28, 28)
COL_LOWCONF = (60, 190, 255)      # yellow
COL_GRAPH_BG = (24, 24, 24)
COL_OK = (120, 220, 120)          # timing within +-1 frame
COL_WARN = (60, 190, 255)         # within +-3 frames
COL_BAD = (80, 80, 240)           # worse than +-3 frames / unmatched
SIDE_COLORS = {LEFT: COL_LEFT, RIGHT: COL_RIGHT}

EDGE_VISIBILITY_THRESHOLD = 0.3
LOWCONF_VISIBILITY = 0.5

# Phase-2 T15 GT-vs-prediction timeline strip. GT stance intervals use the
# documented approximation: from a contact's preferred instant to the next
# same-foot preferred instant (mirrors the metrics-v2 phase map).
TRUTH_WINDOW_S = 6.0              # seconds of history shown in the strip
TRUTH_ROW_H = 13                  # pixels per timeline row
TRUTH_MATCH_HORIZON_S = 0.5       # pair pred<->GT only within this window

# Generic COCO-17 connectivity: nose0 eyes/ears1-4 shoulders5/6 elbows7/8
# wrists9/10 hips11/12 knees13/14 ankles15/16. Backends with other index
# tables pass their own `edges` list of (i, j) pairs.
SKELETON_EDGES_COCO17 = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _lazy_cv2():
    import cv2  # noqa: PLC0415 — deliberate lazy import

    return cv2


# ---------------------------------------------------------------------------
# Renderer state (scrolling-graph history)
# ---------------------------------------------------------------------------


@dataclass
class RenderState:
    """Mutable per-run scratch space: heel-speed history for the graph."""

    window_s: float = 2.0
    graph_wh: tuple[int, int] = (320, 96)
    hist: dict = field(default_factory=lambda: {s: deque(maxlen=4000) for s in SIDES})
    last_heel: dict = field(default_factory=dict)      # side -> (t_s, Point)
    speed_ema: dict = field(default_factory=lambda: {s: None for s in SIDES})


def update_state(state: RenderState, pf: PersonFrame | None, t_s: float) -> None:
    """Fold one frame's heel positions into the scrolling-graph history."""

    if pf is None:
        return
    for side in SIDES:
        obs = (pf.feet or {}).get(side)
        heel = obs.heel if obs is not None else None
        prev = state.last_heel.get(side)
        if heel is not None and prev is not None and t_s > prev[0]:
            v = point_dist(heel, prev[1]) / (t_s - prev[0])
            ema = state.speed_ema[side]
            v_smooth = v if ema is None else 0.6 * v + 0.4 * ema
            state.speed_ema[side] = v_smooth
            state.hist[side].append((t_s, v_smooth))
        # Keep the stored history itself windowed (bounded memory); drawing
        # filters again defensively but must not be the only pruning point.
        cutoff = t_s - state.window_s - 1e-9
        while state.hist[side] and state.hist[side][0][0] < cutoff:
            state.hist[side].popleft()
        if heel is not None:
            state.last_heel[side] = (t_s, heel)


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def active_events(events: list[ContactEvent], t_s: float) -> list[ContactEvent]:
    """Events whose uncertainty window [earliest_s, latest_s] contains t_s."""

    return [e for e in events or [] if e.earliest_s <= t_s <= e.latest_s]


def classify_foot(obs, side_events: list[ContactEvent]) -> str:
    """Visual state word for one foot: contact | swing | lowconf | lost."""

    if obs is None or obs.heel is None:
        return "lost"
    if any(e.observable for e in side_events):
        return "contact"
    vis = [p.visibility for p in (obs.heel, obs.toe) if p is not None]
    if vis and min(vis) < LOWCONF_VISIBILITY:
        return "lowconf"
    return "swing"


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _alpha_rect(img, pt1, pt2, color, alpha: float) -> None:
    x1, y1 = max(pt1[0], 0), max(pt1[1], 0)
    x2, y2 = min(pt2[0], img.shape[1]), min(pt2[1], img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    solid = np.empty_like(roi)
    solid[:] = color
    cv2 = _lazy_cv2()
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, solid, alpha, 0)


def put_text(img, text, org, *, scale=0.42, color=COL_TEXT, thickness=1) -> None:
    cv2 = _lazy_cv2()
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_panel(img, lines: list[tuple[str, tuple]], x: int, y: int, *,
               alpha: float = 0.45, pad: int = 6, leading: int = 16,
               scale: float = 0.42) -> None:
    """Translucent text panel anchored at its top-left corner."""

    if not lines:
        return
    cv2 = _lazy_cv2()
    widths = []
    for text, _c in lines:
        (tw, th), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        widths.append(tw)
    w = (max(widths) if widths else 0) + 2 * pad
    h = pad * 2 + leading * (len(lines) - 1) + int(scale * 22)
    x = max(0, min(x, img.shape[1] - w))
    y = max(0, min(y, img.shape[0] - h))
    _alpha_rect(img, (x, y), (x + w, y + h), COL_PANEL_BG, alpha)
    ty = y + pad + int(scale * 18)
    for (text, color), dx in zip(lines, widths):
        put_text(img, text, (x + pad, ty), scale=scale, color=color)
        ty += leading


def draw_skeleton(img, landmarks: list[Point], edges=SKELETON_EDGES_COCO17, *,
                  vis_threshold: float = EDGE_VISIBILITY_THRESHOLD) -> None:
    """Edges + joint dots for a full landmark list (pixel-space Points)."""

    cv2 = _lazy_cv2()
    for i, j in edges or ():
        if i >= len(landmarks) or j >= len(landmarks):
            continue
        a, b = landmarks[i], landmarks[j]
        if a is None or b is None:
            continue
        if a.visibility < vis_threshold or b.visibility < vis_threshold:
            continue
        pa = (int(a.x), int(a.y))
        pb = (int(b.x), int(b.y))
        cv2.line(img, pa, pb, (210, 210, 210), 2, cv2.LINE_AA)
    for p in landmarks:
        if p is None or p.visibility < vis_threshold:
            continue
        cv2.circle(img, (int(p.x), int(p.y)), 3, (90, 200, 255), -1, cv2.LINE_AA)


def draw_feet(img, pf: PersonFrame | None, side_events: dict,
              t_s: float) -> dict:
    """Per-side state visuals around the heel; returns {side: state_word}."""

    cv2 = _lazy_cv2()
    states = {}
    for side in SIDES:
        obs = (pf.feet or {}).get(side) if pf is not None else None
        evs = side_events.get(side, [])
        st = classify_foot(obs, evs)
        states[side] = st
        color = SIDE_COLORS[side]
        # secondary markers: ankle dot + toe dot
        if obs is not None:
            for p, r in ((obs.ankle, 2), (obs.toe, 2)):
                if p is not None and p.visibility >= LOWCONF_VISIBILITY:
                    cv2.circle(img, (int(p.x), int(p.y)), r, color, -1, cv2.LINE_AA)
        if obs is None or obs.heel is None:
            continue
        hx, hy = int(obs.heel.x), int(obs.heel.y)
        if st == "contact":
            cv2.circle(img, (hx, hy), 9, color, -1, cv2.LINE_AA)
            ev = next((e for e in evs if e.observable), None)
            if ev is not None and ev.latest_s > ev.earliest_s:
                prog = min(max((t_s - ev.earliest_s) /
                               (ev.latest_s - ev.earliest_s), 0.0), 1.0)
                cv2.circle(img, (hx, hy), 9 + int(prog * 8), color, 2, cv2.LINE_AA)
            put_text(img, f"{side[0].upper()} CONTACT", (hx + 12, hy - 8),
                     scale=0.38, color=color)
        elif st == "swing":
            cv2.circle(img, (hx, hy), 7, color, 2, cv2.LINE_AA)
        elif st == "lowconf":
            d = 7
            cv2.line(img, (hx - d, hy - d), (hx + d, hy + d), COL_LOWCONF, 2)
            cv2.line(img, (hx - d, hy + d), (hx + d, hy - d), COL_LOWCONF, 2)
        # "lost": nothing drawn on-body; HUD chip reports it
    return states


# ---------------------------------------------------------------------------
# Overlay panels / banners / graph
# ---------------------------------------------------------------------------


def draw_hud_panels(img, pf: PersonFrame | None, t_s: float, states: dict,
                    *, backend: str = "", mode: str = "",
                    n_events: int = 0, n_observable: int = 0) -> None:
    """HUD top-left, per-side foot chips top-right, legend bottom-left."""

    hud = ["auto-sfx diag", f"backend: {backend or '?'}" + (f"  ({mode})" if mode else "")]
    if pf is None:
        hud.append("no pose row for this frame")
    else:
        hud.append(f"frame {pf.frame_index}  t={t_s:.3f}s")
        hud.append(f"pose score {pf.score:.2f}")
        if pf.lost:
            hud.append("POSE LOST")
    hud.append(f"events {n_observable}/{n_events} observable")
    lines = [(hud[0], COL_TEXT)] + [(s, COL_DIM) for s in hud[1:]]

    chips = []
    for side in SIDES:
        st = states.get(side, "lost")
        chips.append((f"{side[0].upper()} {st}", SIDE_COLORS[side]))
    draw_panel(img, lines, 8, 8)
    draw_panel(img, chips, img.shape[1] - 150, 8)

    legend = [("L left foot", COL_LEFT), ("R right foot", COL_RIGHT)]
    if backend.startswith("rtmpose"):
        legend.append((RTMPOSE_LICENSE_NOTICE, COL_LOWCONF))
    draw_panel(img, legend, 8, img.shape[0] - (16 * len(legend) + 20))


def draw_contact_banners(img, events: list[ContactEvent], t_s: float) -> None:
    """Bottom strip banner per event whose window contains t_s (max 2 rows)."""

    cv2 = _lazy_cv2()
    h_img = img.shape[0]
    for k, e in enumerate(events[:2]):
        row = k % 2
        bh = 22
        y1 = h_img - bh - row * (bh + 4)
        color = SIDE_COLORS.get(e.foot, COL_DIM)
        alpha = 0.55 if e.observable else 0.25
        text = (f"{e.foot.upper()} {e.kind} t={e.time_s:.2f}s "
                f"score={e.score:.2f} [{e.method}]")
        if not e.observable:
            text += " (unobservable)"
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        if e.foot == RIGHT:
            x1 = max(4, img.shape[1] - tw - 20)
        else:
            x1 = 8
        _alpha_rect(img, (x1, y1), (x1 + tw + 12, y1 + bh), (10, 10, 10), alpha)
        cv2.rectangle(img, (x1, y1), (x1 + tw + 12, y1 + bh), color, 1)
        put_text(img, text, (x1 + 6, y1 + 15), scale=0.42,
                 color=color if e.observable else COL_DIM)


def draw_scrolling_graph(img, state: RenderState, t_now: float) -> None:
    """Scrolling heel-speed series per side over the last `window_s`."""

    cv2 = _lazy_cv2()
    gw, gh = state.graph_wh
    H, W = img.shape[:2]
    x1, y1 = W - gw - 10, H - gh - 56          # clears banner strip rows
    x2, y2 = x1 + gw, y1 + gh
    _alpha_rect(img, (x1, y1), (x2, y2), COL_GRAPH_BG, 0.55)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 70), 1)
    put_text(img, "heel speed (px/s)", (x1 + 4, y1 + 13), scale=0.36, color=COL_DIM)

    t_lo, t_hi = t_now - state.window_s, t_now
    for side in SIDES:
        pts = [(t, v) for t, v in state.hist[side] if t_lo <= t <= t_hi]
        if len(pts) < 2:
            continue
        vmax = max(max(v for _t, v in pts), 1.0) * 1.15

        def to_px(t: float, v: float) -> tuple[int, int]:
            fx = (t - t_lo) / max(state.window_s, 1e-6)
            fy = 1.0 - min(v / vmax, 1.0)
            return (int(x1 + fx * gw), int(y1 + gh - fy * (gh - 18)))

        pix = np.array([to_px(t, v) for t, v in pts], dtype=np.int32)
        pix[:, 0] = np.clip(pix[:, 0], x1 + 1, x2 - 1)
        pix[:, 1] = np.clip(pix[:, 1], y1 + 16, y2 - 2)
        cv2.polylines(img, [pix.reshape(-1, 1, 2)], False, SIDE_COLORS[side],
                      2, cv2.LINE_AA)
        last = tuple(pix[-1])
        cv2.circle(img, last, 3, SIDE_COLORS[side], -1, cv2.LINE_AA)


def gt_state_at(gt_contacts: list, foot: str, t_s: float) -> str:
    """GT phase word for one foot at t_s (documented approximation).

    Phase map mirrors metrics-v2: midpoints between consecutive same-foot
    contacts split stance from swing (stance occupies the first half of a
    stride interval). Before the first mark and after the last one the
    state is unknown ("--") — never invented.
    """

    times = sorted(c.preferred_s for c in gt_contacts if c.foot == foot)
    if not times or t_s < times[0]:
        return "--"
    for i, tc in enumerate(times):
        if i + 1 >= len(times):
            break
        nxt = times[i + 1]
        mid = (tc + nxt) / 2.0
        if t_s < mid:
            return "STANCE"
        if t_s < nxt:
            return "SWING"
    return "--"                          # after the last mark: unknown


def nearest_gt_diff_frames(
    gt_contacts: list, foot: str, t_s: float, fps_display: float
) -> int | None:
    """Signed frame error of a prediction vs its diagnostic GT partner.

    Thin wrapper over the shared one-to-one diagnostic association
    (auto_sfx.association): side-agnostic, monotone, at most one partner.
    ``foot`` filters to same-foot partners only (kept for the strip's
    per-row labelling); returns None when no partner within the horizon.
    """

    records = [
        {
            "gt_id": str(i),
            "foot": c.foot,
            "preferred_s": c.preferred_s,
            "earliest_s": c.earliest_s,
            "latest_s": c.latest_s,
        }
        for i, c in enumerate(gt_contacts)
    ]
    res = associate_diagnostic(
        [{"id": "probe", "time_s": t_s, "foot": foot}],
        records,
        horizon_s=TRUTH_MATCH_HORIZON_S,
        fps=fps_display,
    )
    for p in res.pairs:
        if records[p["gt_idx"]]["foot"] == foot:
            return p["err_frames"]
    return None


def strip_pair_labels(gt_contacts: list, events: list[ContactEvent],
                      fps_display: float) -> tuple[dict, set]:
    """Shared-association pairing for the truth strip (computed once).

    Returns ``(labels, matched_gt_ids)``:
    ``labels`` maps ``(pred_foot, pred_time)`` -> signed frame error vs
    the prediction's diagnostic partner; ``matched_gt_ids`` is the set of
    GT indices (position in gt_contacts) that have a partner -- used to
    fill vs hollow the TRUE triangles.
    """

    records = [
        {
            "gt_id": str(i),
            "foot": c.foot,
            "preferred_s": c.preferred_s,
            "earliest_s": c.earliest_s,
            "latest_s": c.latest_s,
        }
        for i, c in enumerate(gt_contacts)
    ]
    res = associate_diagnostic(
        events or [], records,
        horizon_s=TRUTH_MATCH_HORIZON_S,
        fps=fps_display,
    )
    labels: dict = {}
    matched: set = set()
    for p in res.pairs:
        pred = (events or [])[p["pred_idx"]]
        labels[(getattr(pred, "foot", None), pred.time_s)] = p["err_frames"]
        matched.add(p["gt_idx"])
    return labels, matched


def states_to_timeline(rows) -> dict:
    """Per-side step timelines [(t_s, state)] from contact-trace rows.

    Rows without machine-state fields (e.g. pelvis method traces) are
    skipped; ``None`` input yields an empty dict.
    """

    timeline: dict = {}
    for r in rows or []:
        if not isinstance(r, dict) or "state_after" not in r:
            continue
        timeline.setdefault(r.get("side"), []).append(
            (float(r["t_s"]), r["state_after"]))
    for side in timeline:
        timeline[side].sort(key=lambda x: x[0])
    return timeline


def state_at(timeline: dict, side: str, t_s: float,
             max_age_s: float = 1.0) -> str:
    """Machine state word for ``side`` at ``t_s`` ("--" when unknown).

    Rightmost sample at/before ``t_s`` within ``max_age_s``; anything
    else is honestly unknown rather than extrapolated.
    """

    series = timeline.get(side) or []
    best = None
    for ts, st in series:
        if ts <= t_s + 1e-9 and (best is None or ts >= best[0]):
            best = (ts, st)
    if best is None or t_s - best[0] > max_age_s:
        return "--"
    return best[1]


def _pred_chip_word(timeline: dict, side: str, t_now: float) -> str:
    """Chip word from persisted detector state (never event windows)."""

    st = state_at(timeline, side, t_now)
    return {"STANCE": "ST", "CANDIDATE": "ST",
            "SWING": "SW", "EXIT": "SW"}.get(st, "--")




def draw_truth_strip(img, gt_contacts: list, all_events: list[ContactEvent],
                     t_now: float, *, fps_display: float = 30.0,
                     pred_states: dict | None = None) -> None:
    """GT-vs-prediction timeline strip per foot (Phase-2 T15, repaired).

    Four rows — TRUE/PRED × LEFT/RIGHT over the last TRUTH_WINDOW_S.
    Pairing runs through the shared diagnostic association:
    GT marks are filled when their partner exists (hollow red otherwise)
    and predictions carry the signed frame error of THEIR pair.
    The predicted phase chip comes from ``pred_states`` -- the persisted
    detector state timeline (contact-trace.jsonl); without it the chip is
    honestly "--", never derived from event windows.
    Skipped on frames too small to be readable.
    """

    cv2 = _lazy_cv2()
    H, W = img.shape[:2]
    label_w = 52
    header_h = 14
    rows = 4
    panel_h = header_h + rows * TRUTH_ROW_H + 6
    y2 = H - 130                       # clear legend / graph / banners below
    y1 = y2 - panel_h
    if y1 < 60 or W < 320:
        return
    x1, x2 = 8, W - 8
    _alpha_rect(img, (x1, y1), (x2, y2), COL_GRAPH_BG, 0.6)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 70), 1)
    put_text(img, "truth vs prediction", (x1 + 4, y1 + 11),
             scale=0.34, color=COL_DIM)

    t_lo = t_now - TRUTH_WINDOW_S

    def tx(t_s: float) -> int:
        fx = (t_s - t_lo) / TRUTH_WINDOW_S
        return int(x1 + label_w + fx * (x2 - x1 - label_w - 30))

    # Current-phase chips. The PREDICTED word comes ONLY from the
    # persisted machine-state timeline; without a trace it is "--".
    chips = []
    for side in SIDES:
        chips.append(
            f"{side[0].upper()} {gt_state_at(gt_contacts, side, t_now)}(GT)/"
            + _pred_chip_word(pred_states or {}, side, t_now) + "(PRED)"
        )
    put_text(img, "  ".join(chips), (x1 + 150, y1 + 11),
             scale=0.34, color=COL_TEXT)

    labels, matched_gts = strip_pair_labels(
        gt_contacts, all_events, fps_display)
    for row, (side, truth) in enumerate(
        [(LEFT, True), (LEFT, False), (RIGHT, True), (RIGHT, False)]
    ):
        ry = y1 + header_h + row * TRUTH_ROW_H + TRUTH_ROW_H - 3
        color = SIDE_COLORS[side]
        put_text(
            img,
            ("TRUE" if truth else "PRED") + f" {side[0].upper()}",
            (x1 + 4, ry), scale=0.32,
            color=color if truth else COL_DIM,
        )
        cv2.line(img, (x1 + label_w, ry + 2), (x2 - 26, ry + 2),
                 (60, 60, 60), 1)

        if truth:
            for gi, c in enumerate(gt_contacts):
                if c.foot != side or not (t_lo <= c.preferred_s <= t_now):
                    continue
                mt = c.preferred_s
                x = max(x1 + label_w + 1, min(tx(mt), x2 - 27))
                matched = gi in matched_gts
                tri = np.array(
                    [[x, ry - 5], [x - 4, ry + 1], [x + 4, ry + 1]], np.int32
                )
                if matched:
                    cv2.fillPoly(img, [tri], color)
                else:
                    cv2.polylines(img, [tri.reshape(-1, 1, 2)], True,
                                  COL_BAD, 1, cv2.LINE_AA)
        else:
            for e in all_events or []:
                if e.foot != side or not (t_lo <= e.time_s <= t_now):
                    continue
                x = max(x1 + label_w + 1, min(tx(e.time_s), x2 - 27))
                diff = labels.get((e.foot, e.time_s))
                if diff is None and fps_display:
                    # Side-flipped pairings: label with the signed error
                    # of the prediction's actual diagnostic partner.
                    res = associate_diagnostic(
                        [e],
                        [{"gt_id": str(i), "foot": c.foot,
                          "preferred_s": c.preferred_s,
                          "earliest_s": c.earliest_s,
                          "latest_s": c.latest_s}
                         for i, c in enumerate(gt_contacts)],
                        horizon_s=TRUTH_MATCH_HORIZON_S, fps=fps_display)
                    if res.pairs:
                        diff = res.pairs[0]["err_frames"]
                cv2.drawMarker(img, (x, ry + 1), color,
                               cv2.MARKER_TRIANGLE_UP, 7,
                               1 if e.kind != "initial_contact" else 2)
                if diff is not None:
                    dcol = (
                        COL_OK if abs(diff) <= 1
                        else COL_WARN if abs(diff) <= 3
                        else COL_BAD
                    )
                    put_text(img, f"{diff:+d}f", (x + 5, ry + 3),
                             scale=0.30, color=dcol)


def annotate_frame(
    img,
    pf: PersonFrame | None,
    t_s: float,
    events: list[ContactEvent],
    state: RenderState | None = None,
    *,
    backend: str = "",
    mode: str = "",
    edges=SKELETON_EDGES_COCO17,
    all_events: list[ContactEvent] | None = None,
    gt_contacts: list | None = None,
    fps_display: float = 30.0,
    pred_states: dict | None = None,
) -> dict:
    """Annotate one BGR frame in place; returns {side: state_word}.

    `events` are the events active at t_s (see active_events); `all_events`
    (defaults to `events`) is the full sorted list, needed by the Phase-2
    truth strip to place past/future markers in its scroll window.
    With ``gt_contacts`` supplied, draws the GT-vs-prediction timeline
    strip; without it, v1 behaviour is unchanged.
    """

    if state is None:
        state = RenderState()
    update_state(state, pf, t_s)
    side_events = {s: [e for e in events if e.foot == s] for s in SIDES}

    if pf is not None and pf.landmarks:
        draw_skeleton(img, pf.landmarks, edges)
    states = draw_feet(img, pf, side_events, t_s)
    draw_hud_panels(
        img, pf, t_s, states,
        backend=backend, mode=mode, n_events=len(events),
        n_observable=sum(1 for e in events if e.observable),
    )
    draw_contact_banners(img, events, t_s)
    draw_scrolling_graph(img, state, t_s)
    if gt_contacts:
        draw_truth_strip(
            img, gt_contacts, all_events if all_events is not None else events,
            t_s, fps_display=fps_display, pred_states=pred_states,
        )
    return states


# ---------------------------------------------------------------------------
# Video writing + audio muxing
# ---------------------------------------------------------------------------


@dataclass
class RenderResult:
    path: str
    n_frames: int
    width: int
    height: int
    fps: float
    duration_s: float
    audio_muxed: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def mux_audio(video_path: str | Path, wav_path: str | Path,
              out_path: str | Path) -> str:
    """Copy video stream + AAC-encode the click track into `out_path`."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path), "-i", str(wav_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(out),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {proc.stderr.strip()[:500]}")
    return str(out)


def _open_writer(cv2, path: Path, fps: float, width: int, height: int):
    for fourcc in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                 float(fps), (int(width), int(height)))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        "OpenCV VideoWriter could not open an MP4 output (mp4v/avc1 missing "
        "in this build)"
    )


def _coerce_frame_item(item):
    """Normalise producer shapes to (frame_index|None, t_s_hint|None, frame).

    Accepted: bare HxWx3 ndarray | (frame_index:int, ndarray) |
    (ndarray, t_s:float, frame_index:int) — the latter is FrameReader's
    native yield, passed through without glue.
    """

    if isinstance(item, np.ndarray):
        return None, None, item
    if isinstance(item, (tuple, list)):
        if len(item) == 2 and isinstance(item[0], (int, np.integer)):
            return int(item[0]), None, item[1]
        if len(item) == 3 and getattr(item[0], "ndim", None) == 3 \
                and isinstance(item[2], (int, np.integer)):
            return int(item[2]), float(item[1]), item[0]
    return None, None, item


def render_annotated(
    trace: PoseTrace,
    events: list[ContactEvent],
    frames_iter,
    out_path: str | Path,
    *,
    fps: float | None = None,
    click_wav: str | Path | None = None,
    mux: bool = True,
    edges=SKELETON_EDGES_COCO17,
    scroll_window_s: float = 2.0,
    gt_contacts: list | None = None,
    fps_display: float | None = None,
    pred_states: dict | None = None,
) -> RenderResult:
    """Render annotated diagnostic video; optionally mux marker-click audio.

    frames_iter yields decoded BGR frames in decode order — bare ndarrays,
    (frame_index, ndarray) tuples, or FrameReader's native
    (ndarray, t_s, frame_index) triples; rows are matched via
    PersonFrame.frame_index. This renderer never opens the source video.
    """

    cv2 = _lazy_cv2()
    rows = {pf.frame_index: pf for pf in trace.frames}
    events_sorted = sorted(events or [], key=lambda e: e.time_s)
    state = RenderState(window_s=scroll_window_s)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    fps_eff = float(fps) if fps else None
    if fps_eff is None and trace.meta.fps_stream:
        fps_eff = float(trace.meta.fps_stream) or None
    fps_eff = fps_eff or 30.0

    n = 0
    size = None
    with tempfile.TemporaryDirectory(prefix="autosfx-render-") as td:
        silent = Path(td) / "silent.mp4"
        for i, item in enumerate(frames_iter):
            idx, ts_hint, frame = _coerce_frame_item(item)
            if frame is None or getattr(frame, "ndim", 0) != 3:
                raise ValueError(f"bad frame payload at position {i}")
            if writer is None:
                h, w = frame.shape[:2]
                size = (w, h)
                writer = _open_writer(cv2, silent, fps_eff, w, h)
            pf = rows.get(idx if idx is not None else i)
            t_s = pf.t_s if pf is not None else (
                ts_hint if ts_hint is not None else i / fps_eff)
            active = active_events(events_sorted, t_s)
            canvas = np.ascontiguousarray(frame)
            annotate_frame(canvas, pf, t_s, active, state,
                           backend=trace.backend, mode=trace.mode, edges=edges,
                           all_events=events_sorted, gt_contacts=gt_contacts,
                           fps_display=fps_display or fps_eff,
                           pred_states=pred_states)
            writer.write(canvas)
            n += 1
        if writer is None:
            raise ValueError("frames_iter yielded no frames")
        writer.release()

        duration_s = n / fps_eff
        audio_muxed = False
        if click_wav is not None and mux and have_ffmpeg():
            mux_audio(silent, click_wav, out)
            audio_muxed = True
        else:
            shutil.move(str(silent), str(out))

    return RenderResult(
        path=str(out), n_frames=n,
        width=size[0] if size else 0, height=size[1] if size else 0,
        fps=fps_eff, duration_s=duration_s, audio_muxed=audio_muxed,
    )
