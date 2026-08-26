"""Small OpenCV foot-pixel timing refiner for Phase-2B.

Corrected pose supplies only foot identity, a spatial ROI, and a coarse event
window.  Timing evidence comes from pixels: foot motion relative to adjacent
ground, deceleration, static-pixel proportion, and edge persistence.  Common
camera translation is measured and removed; high/ambiguous camera motion
causes abstention rather than becoming contact evidence.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .types import ContactEvent, DiagReason


ROI_CONFIG = {
    "roi_half_window_frames": 7,
    "roi_max_global_motion_px": 24.0,
    "roi_min_phase_response": 0.02,
    "roi_static_flow_px": 0.75,
    "roi_min_edge_pixels": 8,
    "roi_min_change_score": 0.34,
}


def roi_config() -> dict:
    return dict(ROI_CONFIG)


@dataclass
class _PixelEvidence:
    frame: int
    score: float
    global_motion_px: float
    global_response: float
    heel: dict | None
    toe: dict | None


def _point(trace, frame: int, side: str, name: str):
    if not (0 <= frame < len(trace.frames)):
        return None
    obs = (trace.frames[frame].feet or {}).get(side)
    p = getattr(obs, name) if obs is not None else None
    if p is None or p.visibility < 0.35:
        return None
    return float(p.x), float(p.y)


def _scale(trace) -> float:
    vals = []
    for fr in trace.frames:
        if fr.hips is None or fr.lost:
            continue
        for side in ("left", "right"):
            obs = (fr.feet or {}).get(side)
            if obs is None or obs.ankle is None:
                continue
            vals.append(math.hypot(
                obs.ankle.x - fr.hips.x, obs.ankle.y - fr.hips.y
            ))
    return float(np.median(vals)) if vals else 80.0


def _gray_small(frame, max_width: int = 640):
    import cv2

    if isinstance(frame, tuple) and len(frame) == 2:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    factor = min(1.0, max_width / max(1, gray.shape[1]))
    if factor < 1.0:
        gray = cv2.resize(gray, None, fx=factor, fy=factor,
                          interpolation=cv2.INTER_AREA)
    return gray, factor


def _global_translation(prev: np.ndarray, curr: np.ndarray) -> tuple[np.ndarray, float]:
    import cv2

    # Hanning suppresses border/cut artefacts. phaseCorrelate returns the
    # translation from prev to curr and a response strength.
    win = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(
        prev.astype(np.float32), curr.astype(np.float32), win
    )
    return np.asarray(shift, dtype=float), float(response)


def _rect(cx: float, cy: float, rx: int, ry: int,
          width: int, height: int) -> tuple[slice, slice] | None:
    x0, x1 = max(0, int(round(cx)) - rx), min(width, int(round(cx)) + rx + 1)
    y0, y1 = max(0, int(round(cy)) - ry), min(height, int(round(cy)) + ry + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    return slice(y0, y1), slice(x0, x1)


def _region_evidence(prev: np.ndarray, curr: np.ndarray,
                     flow: np.ndarray, point_xy, scale_px: float,
                     global_shift: np.ndarray, cfg: dict) -> dict | None:
    import cv2

    h, w = curr.shape
    cx, cy = point_xy
    rx = max(5, int(round(0.10 * scale_px)))
    ry = max(4, int(round(0.07 * scale_px)))
    foot = _rect(cx, cy, rx, ry, w, h)
    # Nearby support patch: wider and immediately below the landmark ROI.
    ground = _rect(cx, cy + 1.7 * ry, int(1.5 * rx), max(3, ry // 2), w, h)
    if foot is None or ground is None:
        return None
    fy, fx = foot
    gy, gx = ground
    foot_flow_plane = flow[fy, fx]
    ground_flow = flow[gy, gx].reshape(-1, 2)
    if foot_flow_plane.size < 40 or len(ground_flow) < 20:
        return None
    # Ground median is a local camera/support estimate. Fall back toward the
    # global translation when the ground patch is texture-poor/noisy.
    ground_med = np.median(ground_flow, axis=0)
    if not np.all(np.isfinite(ground_med)):
        ground_med = global_shift
    # Dense-flow medians over the rectangular ROI are dominated by static
    # background pixels around a narrow shoe.  Restrict timing evidence to a
    # central ellipse and textured foot-edge neighbourhood, then use a robust
    # upper quantile so a moving shoe cannot be averaged away by turf pixels.
    prev_roi = prev[fy, fx]
    curr_roi = curr[fy, fx]
    edges_prev = cv2.Canny(prev_roi, 60, 140)
    edges_curr = cv2.Canny(curr_roi, 60, 140)
    edge_band = cv2.dilate(
        edges_prev | edges_curr, np.ones((3, 3), np.uint8), iterations=1
    ) > 0
    rh, rw = curr_roi.shape
    yy, xx = np.ogrid[:rh, :rw]
    ellipse = (
        ((xx - (rw - 1) / 2) / max(1.0, 0.48 * rw)) ** 2
        + ((yy - (rh - 1) / 2) / max(1.0, 0.48 * rh)) ** 2
    ) <= 1.0
    foot_mask = ellipse & edge_band
    if int(np.count_nonzero(foot_mask)) < 12:
        foot_mask = ellipse
    foot_flow = foot_flow_plane[foot_mask].reshape(-1, 2)
    rel = foot_flow - ground_med
    rel_mag = np.linalg.norm(rel, axis=1)
    moving_order = np.argsort(rel_mag)
    moving_count = max(12, int(math.ceil(0.25 * len(rel_mag))))
    moving_subset = moving_order[-moving_count:]
    rel_xy = np.median(rel[moving_subset], axis=0)
    static_frac = float(np.mean(rel_mag <= cfg["roi_static_flow_px"]))
    # The completed development ablation used q75.  A stricter q90 variant
    # amplified isolated texture noise and exceeded the local CPU budget
    # before completing, so keep the last fully measured prototype here.
    rel_speed = float(np.quantile(rel_mag, 0.75))
    edge_count = int(np.count_nonzero(edges_curr))
    persistence = None
    if edge_count >= cfg["roi_min_edge_pixels"]:
        matrix = np.asarray([
            [1.0, 0.0, float(ground_med[0])],
            [0.0, 1.0, float(ground_med[1])],
        ], dtype=np.float32)
        warped = cv2.warpAffine(
            edges_prev, matrix, (edges_prev.shape[1], edges_prev.shape[0])
        )
        a, b = (warped > 0) & foot_mask, (edges_curr > 0) & foot_mask
        union = int(np.count_nonzero(a | b))
        persistence = float(np.count_nonzero(a & b) / union) if union else 0.0
    return {
        "relative_flow_px": rel_speed,
        "relative_flow_xy": [round(float(v), 5) for v in rel_xy],
        "static_pixel_fraction": static_frac,
        "edge_persistence": persistence,
        "edge_pixels": edge_count,
        "flow_support_pixels": int(len(rel_mag)),
        "ground_flow_xy": [round(float(v), 4) for v in ground_med],
    }


def _pair_metrics(prev_frame, curr_frame, trace, frame_index: int,
                  side: str, scale: float, cfg: dict) -> dict | None:
    import cv2

    prev, factor = _gray_small(prev_frame)
    curr, factor2 = _gray_small(curr_frame)
    if prev.shape != curr.shape or abs(factor - factor2) > 1e-9:
        return None
    shift, response = _global_translation(prev, curr)
    global_mag = float(np.linalg.norm(shift))
    if (not np.isfinite(global_mag)
            or global_mag > cfg["roi_max_global_motion_px"] * factor
            or response < cfg["roi_min_phase_response"]):
        return {
            "abstain": True,
            "global_motion_px": global_mag / max(factor, 1e-9),
            "global_response": response,
        }
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    out = {
        "abstain": False,
        "global_motion_px": global_mag / max(factor, 1e-9),
        "global_response": response,
    }
    for name in ("heel", "toe"):
        xy = _point(trace, frame_index, side, name)
        if xy is None:
            out[name] = None
            continue
        xy_small = (xy[0] * factor, xy[1] * factor)
        out[name] = _region_evidence(
            prev, curr, flow, xy_small, scale * factor, shift, cfg
        )
    return out


def _decode_frames(video_path: str, needed_indices: set[int] | None = None) -> dict:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for ROI refinement: {video_path}")
    frames = {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if needed_indices is None or index in needed_indices:
            # Store one downsampled gray plane only. Retaining every decoded
            # 1080p BGR frame cost 6.5 GB RSS in the rejected first smoke.
            frames[index] = _gray_small(frame)
        index += 1
        if needed_indices and index > max(needed_indices):
            break
    cap.release()
    return frames


def _event_frame(trace, event: ContactEvent) -> int:
    return min(
        range(len(trace.frames)),
        key=lambda i: abs(trace.frames[i].source_pts_us - event.source_pts_us),
    )


def refine_contacts(video_path: str, trace, events: list[ContactEvent],
                    config: dict | None = None) -> tuple[list[ContactEvent], list[dict], dict]:
    """Return pixel-refined events, abstention diagnostics, and audit summary."""

    cfg = roi_config()
    if config:
        unknown = sorted(set(config) - set(cfg))
        if unknown:
            raise ValueError(f"unknown ROI config keys: {unknown}")
        cfg.update(config)
    for key in ("roi_half_window_frames", "roi_min_edge_pixels"):
        cfg[key] = int(cfg[key])
    for key in (
        "roi_max_global_motion_px", "roi_min_phase_response",
        "roi_static_flow_px", "roi_min_change_score",
    ):
        cfg[key] = float(cfg[key])

    needed: set[int] = set()
    for event in events:
        coarse = _event_frame(trace, event)
        lo = max(1, coarse - int(cfg["roi_half_window_frames"]) - 1)
        hi = min(len(trace.frames) - 1,
                 coarse + int(cfg["roi_half_window_frames"]) + 2)
        needed.update(range(lo, hi + 1))
    frames = _decode_frames(video_path, needed)
    if isinstance(frames, list):  # test/back-compat convenience
        frames = dict(enumerate(frames))
    n = len(trace.frames)
    scale = _scale(trace)
    pair_cache: dict[tuple[int, str], dict | None] = {}
    refined: list[ContactEvent] = []
    diagnostics: list[dict] = []
    abstained = 0
    shifts: list[int] = []

    def pair(i: int, side: str):
        key = (i, side)
        if key not in pair_cache:
            pair_cache[key] = (
                _pair_metrics(frames[i - 1], frames[i], trace, i, side,
                              scale, cfg)
                if 1 <= i < n and i - 1 in frames and i in frames else None
            )
        return pair_cache[key]

    for event in events:
        coarse = _event_frame(trace, event)
        lo = max(2, coarse - cfg["roi_half_window_frames"])
        hi = min(n - 2, coarse + cfg["roi_half_window_frames"])
        evidence_rows: list[_PixelEvidence] = []
        for i in range(lo, hi + 1):
            pairs = {
                j: pair(j, event.foot)
                for j in range(max(1, i - 3), min(n, i + 4))
            }
            if any(
                not row or row.get("abstain")
                for row in pairs.values()
            ):
                continue
            region_scores = []
            per_region = {}
            for name in ("heel", "toe"):
                pre = [
                    pairs[j].get(name) for j in range(i - 2, i + 1)
                    if j in pairs and pairs[j].get(name)
                ]
                post = [
                    pairs[j].get(name) for j in range(i + 1, i + 4)
                    if j in pairs and pairs[j].get(name)
                ]
                if len(pre) < 2 or len(post) < 2:
                    per_region[name] = None
                    continue
                pre_speed = float(np.quantile(
                    [r["relative_flow_px"] for r in pre], 0.75
                ))
                post_speed = float(np.median(
                    [r["relative_flow_px"] for r in post]
                ))
                decel = max(0.0, pre_speed - post_speed)
                # Camera-compensated foot motion must visibly change from a
                # moving pre-contact run into a persistent locally static run.
                # A high static score by itself merely describes stance and
                # cannot choose an arbitrary frame from the window.
                decel_score = decel / max(pre_speed, 0.25)
                static_fraction = float(np.median(
                    [r["static_pixel_fraction"] for r in post]
                ))
                persistence_vals = [
                    r["edge_persistence"] for r in post
                    if r["edge_persistence"] is not None
                ]
                persistence = (
                    float(np.median(persistence_vals))
                    if persistence_vals else 0.0
                )
                pre_down = float(np.quantile([
                    r.get("relative_flow_xy", [0.0, 0.0])[1]
                    for r in pre
                ], 0.75))
                descending_pairs = sum(
                    r.get("relative_flow_xy", [0.0, 0.0])[1] > 0.08
                    for r in pre
                )
                moving = (
                    pre_speed >= max(0.40, 1.30 * post_speed)
                    and decel >= 0.18
                    and descending_pairs >= 1
                )
                post_vertical = float(np.median([
                    abs(r.get("relative_flow_xy", [0.0, 0.0])[1])
                    for r in post
                ]))
                vertical_stop = max(0.0, pre_down - post_vertical)
                vertical_score = vertical_stop / (vertical_stop + 0.5)
                score = (
                    0.42 * min(1.0, decel_score)
                    + 0.28 * static_fraction
                    + 0.18 * persistence
                    + 0.12 * vertical_score
                )
                if not moving:
                    score *= 0.35
                per_region[name] = {
                    **post[0],
                    "pre_relative_flow_px": round(pre_speed, 5),
                    "post_relative_flow_px": round(post_speed, 5),
                    "deceleration_px": round(decel, 5),
                    "vertical_deceleration_px": round(vertical_stop, 5),
                    "post_static_pixel_fraction": round(static_fraction, 5),
                    "post_edge_persistence": round(persistence, 5),
                    "moving_to_static": moving,
                    "contact_score": round(score, 5),
                }
                region_scores.append(score)
            if not region_scores:
                continue
            evidence_rows.append(_PixelEvidence(
                frame=i, score=max(region_scores),
                global_motion_px=max(
                    row["global_motion_px"] for row in pairs.values()
                ),
                global_response=min(
                    row["global_response"] for row in pairs.values()
                ),
                heel=per_region.get("heel"), toe=per_region.get("toe"),
            ))
        if not evidence_rows:
            abstained += 1
            diagnostics.append({
                "code": DiagReason.AMBIGUOUS_SUPPORT_PHASE,
                "t0_s": event.earliest_s, "t1_s": event.latest_s,
                "detail": "contact-roi-v1 abstained: no camera-compensated textured foot ROI",
            })
            continue
        evidence_rows.sort(key=lambda r: (-r.score, r.frame))
        peak = evidence_rows[0].score
        # Very weak evidence is an abstention, never a pose-timestamp fallback.
        if peak < cfg["roi_min_change_score"]:
            abstained += 1
            diagnostics.append({
                "code": DiagReason.AMBIGUOUS_SUPPORT_PHASE,
                "t0_s": event.earliest_s, "t1_s": event.latest_s,
                "detail": f"contact-roi-v1 abstained: weak moving-to-static score {peak:.3f}",
            })
            continue
        # Prefer the first defensible physical change point among essentially
        # tied peaks.  The old prototype picked window boundaries because a
        # late stance frame could have a slightly cleaner edge map.
        tied = [r for r in evidence_rows if r.score >= peak - 0.025]
        best = min(tied, key=lambda r: r.frame)
        by_frame = {r.frame: r for r in evidence_rows}
        near = {best.frame}
        for direction in (-1, 1):
            j = best.frame + direction
            while j in by_frame and by_frame[j].score >= 0.90 * peak:
                near.add(j)
                j += direction
        out = copy.deepcopy(event)
        out.time_s = round(trace.frames[best.frame].t_s, 6)
        out.source_pts_us = trace.frames[best.frame].source_pts_us
        out.earliest_s = round(trace.frames[min(near)].t_s, 6)
        out.latest_s = round(trace.frames[max(near)].t_s, 6)
        out.detector = f"{trace.backend}-contact-roi-v1"
        out.method = "contact-roi-v1"
        out.score = round(min(1.0, 0.45 * event.score + 0.55 * best.score), 3)
        out.evidence = {
            **event.evidence,
            "raw_pose_event_id": event.id,
            "raw_pose_frame": coarse,
            "roi_refined_frame": best.frame,
            "roi_shift_frames": best.frame - coarse,
            "roi_score": round(best.score, 5),
            "global_motion_px": round(best.global_motion_px, 5),
            "global_phase_response": round(best.global_response, 5),
            "heel_pixel_evidence": best.heel,
            "toe_pixel_evidence": best.toe,
            "camera_motion_is_contact_evidence": False,
        }
        refined.append(out)
        shifts.append(best.frame - coarse)

    refined.sort(key=lambda e: (e.time_s, e.foot))
    for rank, event in enumerate(refined, start=1):
        event.id = f"evt_{rank:03d}"
    return refined, diagnostics, {
        "config": cfg,
        "retained_candidate_frames": len(frames),
        "raw_pose_events": len(events),
        "refined_events": len(refined),
        "abstained_events": abstained,
        "shift_frames": shifts,
        "paper_constraint": (
            "design-inspired only; Kobayashi/Nakazawa 2025 controlled fixed "
            "low-camera results are not transferred to this footage"
        ),
    }
