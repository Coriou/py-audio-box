"""Shot-local support geometry: one constant line per shot, never rolling.

Image y grows downward.  Signed height is positive in the air (smaller y).
The fit is a robust line through slow, floor-most landmarks; near-vertical
lines and large residuals abstain instead of inventing a wall or a plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


DEFAULT_SUPPORT_GEOMETRY_CONFIG = {
    "min_points": 8,
    "max_residual_rms_bs": 0.08,
    "min_speed_quantile": 0.40,
    "ground_y_quantile": 0.50,
}

_FORBIDDEN_OFFSET_KEYS = ("global_offset", "timing_offset", "frame_offset")
_VERTICAL_DEG = 15.0
_RANSAC_ITERS = 400
_RANSAC_SEED = 0


def validate_support_geometry_config(config: dict | None) -> dict:
    """Return defaults merged with ``config``. Unknown keys raise ValueError."""

    cfg = dict(DEFAULT_SUPPORT_GEOMETRY_CONFIG)
    if config:
        banned = [k for k in _FORBIDDEN_OFFSET_KEYS if k in config]
        if banned:
            raise ValueError(f"forbidden timing-offset keys: {banned}")
        unknown = sorted(set(config) - set(cfg))
        if unknown:
            raise ValueError(f"unknown contact-config keys: {unknown}")
        cfg.update(config)
    cfg["min_points"] = max(2, int(cfg["min_points"]))
    cfg["max_residual_rms_bs"] = float(cfg["max_residual_rms_bs"])
    cfg["min_speed_quantile"] = float(cfg["min_speed_quantile"])
    cfg["ground_y_quantile"] = float(cfg["ground_y_quantile"])
    return cfg


@dataclass(frozen=True)
class SupportGeometry:
    kind: str                    # "line" | "abstain"
    a: float                     # line a*x + b*y + c = 0
    b: float
    c: float
    residual_rms_bs: float       # inlier RMS in body-scale; nan if abstain
    inlier_count: int
    reason: str | None           # required when kind == "abstain"
    shot_start_frame: int
    shot_end_frame: int          # inclusive

    def height_bs(self, x: float, y: float, scale: float) -> float:
        """Signed height in body-scale. Air (image-up / smaller y) is > 0.

        Returns nan when kind == "abstain" or scale is not finite/positive.
        """

        if self.kind == "abstain":
            return float("nan")
        if not np.isfinite(scale) or scale <= 0:
            return float("nan")
        if not np.isfinite(x) or not np.isfinite(y):
            return float("nan")
        norm = math.hypot(self.a, self.b)
        if norm < 1e-12:
            return float("nan")
        return (self.a * x + self.b * y + self.c) / (norm * scale)


def _shot_span(shot_indices: np.ndarray) -> tuple[int, int]:
    if shot_indices.size == 0:
        return 0, 0
    return int(np.min(shot_indices)), int(np.max(shot_indices))


def _abstain(reason: str, start: int, end: int) -> SupportGeometry:
    return SupportGeometry(
        kind="abstain",
        a=float("nan"),
        b=float("nan"),
        c=float("nan"),
        residual_rms_bs=float("nan"),
        inlier_count=0,
        reason=reason,
        shot_start_frame=start,
        shot_end_frame=end,
    )


def _orient_air_positive(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Flip the normal so a point with smaller y (air) has positive signed value."""

    norm = math.hypot(a, b)
    if norm < 1e-12:
        return a, b, c
    a, b, c = a / norm, b / norm, c / norm
    # Sample a unit step toward image-up from the origin of the line's offset.
    # For ax+by+c, decreasing y by 1 changes the value by -b; we want that > 0.
    if -b < 0:
        a, b, c = -a, -b, -c
    return a, b, c


def _angle_from_vertical_deg(a: float, b: float) -> float:
    """Angle between the line and the image-vertical axis, in degrees."""

    norm = math.hypot(a, b)
    if norm < 1e-12:
        return 90.0
    # Line direction (-b, a); vertical is (0, 1).
    cos_phi = min(1.0, abs(a) / norm)
    return math.degrees(math.acos(cos_phi))


def _pca_line(px: np.ndarray, py: np.ndarray) -> tuple[float, float, float] | None:
    pts = np.column_stack((px, py))
    mean = pts.mean(axis=0)
    centered = pts - mean
    if not np.isfinite(centered).all():
        return None
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1]
    a, b = float(normal[0]), float(normal[1])
    c = float(-a * mean[0] - b * mean[1])
    return _orient_air_positive(a, b, c)


def _signed_abs_dist(a: float, b: float, c: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    norm = math.hypot(a, b)
    if norm < 1e-12:
        return np.full(x.shape, np.inf)
    return np.abs(a * x + b * y + c) / norm


def _line_from_pair(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float] | None:
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    if math.hypot(a, b) < 1e-12:
        return None
    return a, b, c


def _ransac_line(
    px: np.ndarray,
    py: np.ndarray,
    inlier_px: float,
) -> tuple[tuple[float, float, float], np.ndarray] | None:
    n = int(px.size)
    if n < 2:
        return None
    rng = np.random.default_rng(_RANSAC_SEED)
    if n <= 16:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = [tuple(int(k) for k in rng.choice(n, 2, replace=False))
                 for _ in range(_RANSAC_ITERS)]

    best_mask = None
    best_count = -1
    best_rms = np.inf
    min_sep = 1e-6
    for i, j in pairs:
        if math.hypot(px[i] - px[j], py[i] - py[j]) < min_sep:
            continue
        fitted = _line_from_pair(px[i], py[i], px[j], py[j])
        if fitted is None:
            continue
        a, b, c = fitted
        dist = _signed_abs_dist(a, b, c, px, py)
        mask = dist <= inlier_px
        count = int(mask.sum())
        if count < 2:
            continue
        rms = float(np.sqrt(np.mean(dist[mask] ** 2)))
        if count > best_count or (count == best_count and rms < best_rms):
            best_count = count
            best_rms = rms
            best_mask = mask

    if best_mask is None or best_count < 2:
        return None
    refit = _pca_line(px[best_mask], py[best_mask])
    if refit is None:
        return None
    a, b, c = refit
    dist = _signed_abs_dist(a, b, c, px, py)
    mask = dist <= inlier_px
    if int(mask.sum()) < 2:
        mask = best_mask
    refit2 = _pca_line(px[mask], py[mask])
    if refit2 is None:
        return (a, b, c), mask
    return refit2, mask


def estimate_shot_support(
    xs: np.ndarray,
    ys: np.ndarray,
    speeds_bs: np.ndarray,
    times: np.ndarray,
    shot_indices: np.ndarray,
    body_scale_px: float,
    *,
    min_points: int = 8,
    max_residual_rms_bs: float = 0.08,
    min_speed_quantile: float = 0.40,
    ground_y_quantile: float = 0.50,
) -> SupportGeometry:
    """One constant support line for the shot. Never a rolling floor.

    Seed observations: finite (x, y, speed) inside shot_indices whose speed
    is at or below the shot's min_speed_quantile AND whose y is at or above
    the shot's ground_y_quantile (lower in the image). Fit a robust line
    (RANSAC or IRLS). Orient the normal so smaller y (air) has positive
    height. Abstain when seeds < min_points, the line is within 15 degrees
    of vertical, residual RMS exceeds max_residual_rms_bs, or body_scale
    is invalid.
    """

    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    speeds_bs = np.asarray(speeds_bs, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    shot_indices = np.asarray(shot_indices, dtype=int).reshape(-1)
    if not (xs.size == ys.size == speeds_bs.size == times.size == shot_indices.size):
        raise ValueError("estimate_shot_support arrays must have the same length")

    start, end = _shot_span(shot_indices)
    min_points = max(2, int(min_points))
    max_residual_rms_bs = float(max_residual_rms_bs)

    if not np.isfinite(body_scale_px) or body_scale_px <= 0:
        return _abstain("invalid_scale", start, end)

    good = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(speeds_bs)
    if int(good.sum()) < min_points:
        return _abstain("too_few_seeds", start, end)

    gx, gy, gs = xs[good], ys[good], speeds_bs[good]
    speed_cut = float(np.quantile(gs, min_speed_quantile))
    y_cut = float(np.quantile(gy, ground_y_quantile))
    seed = (gs <= speed_cut) & (gy >= y_cut)
    if int(seed.sum()) < min_points:
        return _abstain("too_few_seeds", start, end)

    px, py = gx[seed], gy[seed]
    span = math.hypot(float(px.max() - px.min()), float(py.max() - py.min()))
    if span < max(1e-6, 1e-4 * body_scale_px):
        return _abstain("too_few_seeds", start, end)

    inlier_px = max(1.0, 1.5 * max_residual_rms_bs * body_scale_px)
    fitted = _ransac_line(px, py, inlier_px)
    if fitted is None:
        return _abstain("too_few_seeds", start, end)
    (a, b, c), mask = fitted
    a, b, c = _orient_air_positive(a, b, c)
    if _angle_from_vertical_deg(a, b) < _VERTICAL_DEG:
        return _abstain("near_vertical", start, end)

    dist = _signed_abs_dist(a, b, c, px, py)
    inliers = dist <= inlier_px
    inlier_count = int(inliers.sum())
    if inlier_count < min_points:
        return _abstain("too_few_seeds", start, end)
    rms_px = float(np.sqrt(np.mean(dist[inliers] ** 2)))
    rms_bs = rms_px / body_scale_px
    if not np.isfinite(rms_bs) or rms_bs > max_residual_rms_bs:
        return _abstain("residual_too_large", start, end)

    return SupportGeometry(
        kind="line",
        a=float(a),
        b=float(b),
        c=float(c),
        residual_rms_bs=float(rms_bs),
        inlier_count=inlier_count,
        reason=None,
        shot_start_frame=start,
        shot_end_frame=end,
    )
