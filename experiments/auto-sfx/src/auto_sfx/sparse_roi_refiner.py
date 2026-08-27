"""Opt-in sparse, stabilized heel/toe candidate-window analysis.

This research helper is deliberately separate from ``contact-roi-v1``.  It
uses bounded sparse corner tracks, robust background affine compensation, and
pose-guided heel/toe/ground tracks.  Camera motion is removed before scoring
and never counts as contact evidence.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


METHOD = "sparse-stabilized-roi-v1"
HARD_MAX_WINDOW_FRAMES = 61
HARD_MAX_WIDTH = 1280
HARD_MAX_BACKGROUND_FEATURES = 512
HARD_MAX_REGION_FEATURES = 128
HARD_MAX_SEEK_PREROLL_FRAMES = 300

SPARSE_ROI_CONFIG: dict[str, float | int] = {
    "max_window_frames": 21,
    "max_width": 640,
    "max_background_features": 160,
    "max_region_features": 32,
    "min_background_tracks": 18,
    "min_landmark_tracks": 2,
    "min_ground_tracks": 5,
    "feature_quality": 0.012,
    "feature_min_distance_px": 5.0,
    "lk_window_px": 9,
    "lk_max_level": 2,
    "lk_fb_max_error_px": 1.25,
    "affine_reprojection_px": 1.75,
    "affine_min_inlier_ratio": 0.58,
    "affine_min_coverage": 0.035,
    "affine_max_scale_delta": 0.16,
    "affine_max_shear": 0.18,
    "affine_max_rotation_deg": 14.0,
    "cut_max_warped_mae": 34.0,
    "cut_histogram_distance": 0.34,
    "foot_radius_scale": 0.58,
    "foot_radius_min_px": 7.0,
    "foot_radius_max_px": 18.0,
    "ground_gap_scale": 0.65,
    "pose_assignment_gate_px": 2.25,
    "actor_box_foot_margin_scale": 0.72,
    "actor_box_min_height_scale": 2.5,
    "min_pre_vertical_speed_px_s": 18.0,
    "max_post_speed_px_s": 18.0,
    "min_vertical_deceleration_px_s2": 180.0,
    "min_contact_score": 0.58,
}


def sparse_roi_config() -> dict[str, float | int]:
    return dict(SPARSE_ROI_CONFIG)


def _config(overrides: dict[str, Any] | None) -> dict[str, float | int]:
    cfg = sparse_roi_config()
    if overrides:
        unknown = sorted(set(overrides) - set(cfg))
        if unknown:
            raise ValueError(f"unknown sparse ROI config keys: {unknown}")
        cfg.update(overrides)
    integer_keys = {
        "max_window_frames",
        "max_width",
        "max_background_features",
        "max_region_features",
        "min_background_tracks",
        "min_landmark_tracks",
        "min_ground_tracks",
        "lk_window_px",
        "lk_max_level",
    }
    for key in integer_keys:
        cfg[key] = int(cfg[key])
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in set(cfg) - integer_keys:
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")
    hard_limits = {
        "max_window_frames": HARD_MAX_WINDOW_FRAMES,
        "max_width": HARD_MAX_WIDTH,
        "max_background_features": HARD_MAX_BACKGROUND_FEATURES,
        "max_region_features": HARD_MAX_REGION_FEATURES,
        "lk_max_level": 4,
    }
    for key, limit in hard_limits.items():
        if cfg[key] > limit:
            raise ValueError(f"{key} exceeds hard CPU bound {limit}")
    return cfg


def _abstain(
    reason: str,
    detail: str,
    *,
    frame_count: int,
    pair_evidence: list[dict[str, Any]] | None = None,
    scale: float = 1.0,
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "status": "abstain",
        "reason": reason,
        "detail": detail,
        "frame_count": frame_count,
        "processing_scale": round(scale, 8),
        "pair_evidence": pair_evidence or [],
        "camera_motion_is_contact_evidence": False,
        "suggested_frame": None,
        "suggested_pts_us": None,
        "suggested_interval_pts_us": None,
        "score": 0.0,
    }


def _as_gray(frame: np.ndarray, factor: float) -> np.ndarray:
    import cv2

    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if array.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        array = cv2.cvtColor(array, code)
    elif array.ndim != 2:
        raise ValueError("frames must be gray, BGR, or BGRA arrays")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if factor < 1.0:
        array = cv2.resize(array, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(array)


def _prepare_frames(
    frames: Iterable[np.ndarray], max_width: int, max_frames: int
) -> tuple[list[np.ndarray], float, bool]:
    # Materialize the supplied window once.  Generators and decoded frames are
    # never revisited or decoded per candidate pair.
    supplied = list(itertools.islice(iter(frames), max_frames + 1))
    if len(supplied) > max_frames:
        return [], 1.0, True
    if not supplied:
        return [], 1.0, False
    first = np.asarray(supplied[0])
    if first.ndim not in (2, 3) or first.shape[0] < 2 or first.shape[1] < 2:
        raise ValueError("invalid frame geometry")
    source_shape = first.shape[:2]
    factor = min(1.0, max_width / source_shape[1])
    prepared: list[np.ndarray] = []
    for frame in supplied:
        if np.asarray(frame).shape[:2] != source_shape:
            raise ValueError("frame geometry changed inside candidate window")
        prepared.append(_as_gray(frame, factor))
    return prepared, factor, False


def _points(
    values: Sequence[Sequence[float] | None], count: int, factor: float
) -> list[tuple[float, float] | None]:
    if len(values) != count:
        raise ValueError("trajectory length must equal frame count")
    result: list[tuple[float, float] | None] = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        if len(value) != 2:
            raise ValueError("trajectory points must contain x and y")
        x, y = float(value[0]) * factor, float(value[1]) * factor
        result.append((x, y) if math.isfinite(x) and math.isfinite(y) else None)
    return result


def _radius(heel: tuple[float, float], toe: tuple[float, float], cfg: dict) -> float:
    distance = math.hypot(heel[0] - toe[0], heel[1] - toe[1])
    return float(
        np.clip(
            distance * cfg["foot_radius_scale"],
            cfg["foot_radius_min_px"],
            cfg["foot_radius_max_px"],
        )
    )


def _boxes(
    values: Sequence[Sequence[float] | None] | None,
    heels: list[tuple[float, float] | None],
    toes: list[tuple[float, float] | None],
    factor: float,
    width: int,
    height: int,
    cfg: dict,
) -> list[tuple[float, float, float, float]] | None:
    if values is None or len(values) != len(heels):
        return None
    result = []
    for index, (heel, toe) in enumerate(zip(heels, toes, strict=True)):
        if heel is None or toe is None:
            return None
        radius = _radius(heel, toe, cfg)
        raw = values[index]
        if raw is None or len(raw) != 4:
            return None
        box = tuple(float(value) * factor for value in raw)
        x0, y0, x1, y1 = box
        if not all(math.isfinite(value) for value in box) or x1 <= x0 or y1 <= y0:
            return None
        clipped = (max(0.0, x0), max(0.0, y0), min(width - 1.0, x1), min(height - 1.0, y1))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        margin = cfg["actor_box_foot_margin_scale"] * radius
        foot_x0 = min(heel[0], toe[0]) - margin
        foot_x1 = max(heel[0], toe[0]) + margin
        foot_y0 = min(heel[1], toe[1]) - margin
        foot_y1 = max(heel[1], toe[1]) + margin
        if not (
            clipped[0] <= foot_x0
            and clipped[1] <= foot_y0
            and clipped[2] >= foot_x1
            and clipped[3] >= foot_y1
        ):
            return None
        if clipped[3] - clipped[1] < cfg["actor_box_min_height_scale"] * radius:
            return None
        result.append(clipped)
    return result


def _regions_valid(
    heels: list[tuple[float, float] | None],
    toes: list[tuple[float, float] | None],
    width: int,
    height: int,
    cfg: dict,
) -> bool:
    for heel, toe in zip(heels, toes, strict=True):
        if heel is None or toe is None:
            return False
        radius = _radius(heel, toe, cfg)
        for x, y in (heel, toe):
            if x - radius < 1 or x + radius >= width - 1:
                return False
            if y - radius < 1 or y + radius >= height - 1:
                return False
        ground_top = max(heel[1], toe[1]) + cfg["ground_gap_scale"] * radius
        ground_bottom = ground_top + 1.35 * radius
        if ground_bottom >= height - 1:
            return False
    return True


def _background_mask(
    shape: tuple[int, int],
    box: tuple[float, float, float, float],
    heel: tuple[float, float],
    toe: tuple[float, float],
    radius: float,
) -> np.ndarray:
    import cv2

    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    border = max(5, int(round(radius)))
    mask[:border] = 0
    mask[-border:] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    x0, y0, x1, y1 = box
    pad = int(round(0.5 * radius))
    cv2.rectangle(
        mask,
        (max(0, int(math.floor(x0)) - pad), max(0, int(math.floor(y0)) - pad)),
        (min(width - 1, int(math.ceil(x1)) + pad), min(height - 1, int(math.ceil(y1)) + pad)),
        0,
        thickness=-1,
    )
    for point in (heel, toe):
        cv2.circle(mask, tuple(int(round(value)) for value in point), int(1.4 * radius), 0, -1)
    return mask


def _ellipse_mask(
    shape: tuple[int, int], center: tuple[float, float], rx: float, ry: float
) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        tuple(int(round(value)) for value in center),
        (max(2, int(round(rx))), max(2, int(round(ry)))),
        0,
        0,
        360,
        255,
        -1,
    )
    return mask


def _ground_mask(
    shape: tuple[int, int],
    heel: tuple[float, float],
    toe: tuple[float, float],
    radius: float,
    cfg: dict,
) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    x0 = int(round(min(heel[0], toe[0]) - 1.15 * radius))
    x1 = int(round(max(heel[0], toe[0]) + 1.15 * radius))
    y0 = int(round(max(heel[1], toe[1]) + cfg["ground_gap_scale"] * radius))
    y1 = int(round(y0 + 1.25 * radius))
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def _features(gray: np.ndarray, mask: np.ndarray, maximum: int, cfg: dict) -> np.ndarray:
    import cv2

    found = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=maximum,
        qualityLevel=cfg["feature_quality"],
        minDistance=cfg["feature_min_distance_px"],
        mask=mask,
        blockSize=3,
        useHarrisDetector=False,
    )
    if found is None:
        return np.empty((0, 2), dtype=np.float32)
    points = found.reshape(-1, 2)
    order = np.lexsort((points[:, 0], points[:, 1]))
    return points[order].astype(np.float32)


def _inside_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=bool)
    xy = np.rint(points).astype(int)
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < mask.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < mask.shape[0])
    )
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[xy[valid, 1], xy[valid, 0]] > 0
    return inside


def _track(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    cfg: dict,
    current_mask: np.ndarray | None = None,
    initial_displacement: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    import cv2

    if not len(points):
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=np.float32), 0
    source = points.reshape(-1, 1, 2)
    window = (cfg["lk_window_px"], cfg["lk_window_px"])
    forward_seed = None
    forward_flags = 0
    if initial_displacement is not None:
        forward_seed = (points + initial_displacement).reshape(-1, 1, 2).astype(np.float32)
        forward_flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        source,
        forward_seed,
        winSize=window,
        maxLevel=cfg["lk_max_level"],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
        minEigThreshold=1e-4,
        flags=forward_flags,
    )
    if forward is None or status_forward is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=np.float32), len(points)
    backward_seed = source.copy() if initial_displacement is not None else None
    backward_flags = cv2.OPTFLOW_USE_INITIAL_FLOW if backward_seed is not None else 0
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        forward,
        backward_seed,
        winSize=window,
        maxLevel=cfg["lk_max_level"],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
        minEigThreshold=1e-4,
        flags=backward_flags,
    )
    if backward is None or status_backward is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=np.float32), len(points)
    destination = forward.reshape(-1, 2)
    reverse = backward.reshape(-1, 2)
    fb_error = np.linalg.norm(reverse - points, axis=1)
    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & np.all(np.isfinite(destination), axis=1)
        & np.isfinite(fb_error)
        & (fb_error <= cfg["lk_fb_max_error_px"])
    )
    if current_mask is not None:
        valid &= _inside_mask(destination, current_mask)
    return points[valid], destination[valid], fb_error[valid], len(points)


def _solve_affine(source: np.ndarray, destination: np.ndarray) -> np.ndarray | None:
    design = np.column_stack((source, np.ones(len(source), dtype=float)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, destination, rcond=None)
    return coefficients.T if rank == 3 and np.all(np.isfinite(coefficients)) else None


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _robust_affine(
    source: np.ndarray, destination: np.ndarray, shape: tuple[int, int], cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    count = len(source)
    if count < cfg["min_background_tracks"]:
        return None
    anchor_count = min(12, count)
    anchor_indices = np.unique(np.linspace(0, count - 1, anchor_count, dtype=int))
    threshold = cfg["affine_reprojection_px"]
    best: tuple[int, float, tuple[int, int, int], np.ndarray, np.ndarray] | None = None
    for triple in itertools.combinations(anchor_indices.tolist(), 3):
        selected = np.asarray(triple)
        triangle = source[selected]
        first_edge = triangle[1] - triangle[0]
        second_edge = triangle[2] - triangle[0]
        area = abs(float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]))
        if area < 8.0:
            continue
        matrix = _solve_affine(source[selected], destination[selected])
        if matrix is None:
            continue
        errors = np.linalg.norm(destination - _apply_affine(matrix, source), axis=1)
        inliers = errors <= threshold
        inlier_count = int(np.count_nonzero(inliers))
        median = float(np.median(errors[inliers])) if inlier_count else math.inf
        candidate = (inlier_count, -median, triple, matrix, inliers)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return None
    inliers = best[4]
    matrix = _solve_affine(source[inliers], destination[inliers])
    if matrix is None:
        return None
    errors = np.linalg.norm(destination - _apply_affine(matrix, source), axis=1)
    inliers = errors <= threshold
    if int(np.count_nonzero(inliers)) < cfg["min_background_tracks"]:
        return None
    matrix = _solve_affine(source[inliers], destination[inliers])
    if matrix is None:
        return None
    errors = np.linalg.norm(destination - _apply_affine(matrix, source), axis=1)
    height, width = shape
    span = np.ptp(source[inliers], axis=0)
    coverage = float((span[0] * span[1]) / max(1.0, width * height))
    if coverage < cfg["affine_min_coverage"]:
        return None
    return matrix, inliers, errors


def _affine_geometry_ok(matrix: np.ndarray, cfg: dict) -> bool:
    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    if not math.isfinite(determinant) or determinant <= 0:
        return False
    singular = np.linalg.svd(linear, compute_uv=False)
    if np.max(np.abs(singular - 1.0)) > cfg["affine_max_scale_delta"]:
        return False
    columns = linear[:, 0], linear[:, 1]
    column_product = np.linalg.norm(columns[0]) * np.linalg.norm(columns[1])
    shear = abs(float(np.dot(*columns))) / max(1e-9, column_product)
    rotation = abs(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
    return shear <= cfg["affine_max_shear"] and rotation <= cfg["affine_max_rotation_deg"]


def _histogram_distance(previous: np.ndarray, current: np.ndarray) -> float:
    old = np.histogram(previous, bins=32, range=(0, 256))[0].astype(float)
    new = np.histogram(current, bins=32, range=(0, 256))[0].astype(float)
    old /= max(1.0, old.sum())
    new /= max(1.0, new.sum())
    return float(0.5 * np.abs(old - new).sum())


def _background_affine(
    previous: np.ndarray,
    current: np.ndarray,
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    cfg: dict,
) -> tuple[dict[str, Any] | None, str | None]:
    import cv2

    points = _features(previous, previous_mask, cfg["max_background_features"], cfg)
    if len(points) < cfg["min_background_tracks"]:
        return None, "weak_background_compensation"
    source, destination, fb_error, initial_count = _track(
        previous, current, points, cfg, current_mask
    )
    histogram_distance = _histogram_distance(previous, current)
    if len(source) < cfg["min_background_tracks"]:
        reason = (
            "edit_cut"
            if histogram_distance >= cfg["cut_histogram_distance"]
            else "weak_background_compensation"
        )
        return None, reason
    fitted = _robust_affine(source, destination, previous.shape, cfg)
    if fitted is None:
        reason = (
            "edit_cut"
            if histogram_distance >= cfg["cut_histogram_distance"]
            else "weak_background_compensation"
        )
        return None, reason
    matrix, inliers, errors = fitted
    if not _affine_geometry_ok(matrix, cfg):
        return None, "geometry_failure"
    inlier_ratio = float(np.count_nonzero(inliers) / max(1, initial_count))
    if inlier_ratio < cfg["affine_min_inlier_ratio"]:
        return None, "weak_background_compensation"
    warped = cv2.warpAffine(previous, matrix, (previous.shape[1], previous.shape[0]))
    valid = cv2.warpAffine(
        np.full(previous.shape, 255, np.uint8), matrix, (previous.shape[1], previous.shape[0])
    )
    compare = (valid > 0) & (current_mask > 0)
    if int(np.count_nonzero(compare)) < cfg["min_background_tracks"] * 8:
        return None, "weak_background_compensation"
    warped_mae = float(
        np.median(np.abs(current[compare].astype(np.int16) - warped[compare].astype(np.int16)))
    )
    if warped_mae > cfg["cut_max_warped_mae"]:
        return None, "edit_cut"
    evidence = {
        "affine": [[round(float(value), 7) for value in row] for row in matrix],
        "initial_features": int(initial_count),
        "tracked_features": int(len(source)),
        "inlier_features": int(np.count_nonzero(inliers)),
        "inlier_ratio": round(inlier_ratio, 6),
        "median_reprojection_error_px": round(float(np.median(errors[inliers])), 6),
        "median_forward_backward_error_px": round(float(np.median(fb_error)), 6),
        "warped_background_mae": round(warped_mae, 6),
        "histogram_distance": round(histogram_distance, 6),
    }
    return {"matrix": matrix, "evidence": evidence}, None


def _region_track(
    previous: np.ndarray,
    current: np.ndarray,
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    affine: np.ndarray,
    minimum: int,
    cfg: dict,
    expected_displacement: np.ndarray | None = None,
    expected_residual: np.ndarray | None = None,
) -> dict[str, Any] | None:
    points = _features(previous, previous_mask, cfg["max_region_features"], cfg)
    source, destination, fb_error, initial_count = _track(
        previous,
        current,
        points,
        cfg,
        current_mask,
        initial_displacement=expected_displacement,
    )
    residuals = destination - _apply_affine(affine, source)
    if expected_residual is not None and len(source):
        assigned = np.linalg.norm(residuals - expected_residual, axis=1) <= cfg[
            "pose_assignment_gate_px"
        ]
        source = source[assigned]
        destination = destination[assigned]
        fb_error = fb_error[assigned]
        residuals = residuals[assigned]
    if len(source) < minimum:
        return None
    residual = np.median(residuals, axis=0)
    spread = np.linalg.norm(residuals - residual, axis=1)
    support_quality = min(1.0, len(source) / max(float(minimum * 2), 1.0))
    fb_quality = max(0.0, 1.0 - float(np.median(fb_error)) / cfg["lk_fb_max_error_px"])
    quality = support_quality * fb_quality
    return {
        "initial_features": int(initial_count),
        "tracked_features": int(len(source)),
        "residual_xy_px": [round(float(value), 6) for value in residual],
        "residual_spread_px": round(float(np.median(spread)), 6),
        "median_forward_backward_error_px": round(float(np.median(fb_error)), 6),
        "quality": round(float(quality), 6),
        "_residual": residual,
    }


def _pair_evidence(
    previous: np.ndarray,
    current: np.ndarray,
    previous_heel: tuple[float, float],
    current_heel: tuple[float, float],
    previous_toe: tuple[float, float],
    current_toe: tuple[float, float],
    previous_box: tuple[float, float, float, float],
    current_box: tuple[float, float, float, float],
    dt_s: float,
    from_frame: int,
    to_frame: int,
    cfg: dict,
) -> tuple[dict[str, Any] | None, str | None]:
    radius = _radius(previous_heel, previous_toe, cfg)
    current_radius = _radius(current_heel, current_toe, cfg)
    background_previous = _background_mask(
        previous.shape, previous_box, previous_heel, previous_toe, radius
    )
    background_current = _background_mask(
        current.shape, current_box, current_heel, current_toe, current_radius
    )
    background, reason = _background_affine(
        previous, current, background_previous, background_current, cfg
    )
    if background is None:
        return None, reason
    affine = background["matrix"]
    heel_displacement = np.asarray(current_heel) - np.asarray(previous_heel)
    toe_displacement = np.asarray(current_toe) - np.asarray(previous_toe)
    heel_residual = np.asarray(current_heel) - _apply_affine(
        affine, np.asarray([previous_heel], dtype=float)
    )[0]
    toe_residual = np.asarray(current_toe) - _apply_affine(
        affine, np.asarray([previous_toe], dtype=float)
    )[0]
    region_specs = {
        "heel": (
            _ellipse_mask(previous.shape, previous_heel, radius, 0.82 * radius),
            _ellipse_mask(current.shape, current_heel, current_radius, 0.82 * current_radius),
            cfg["min_landmark_tracks"],
        ),
        "toe": (
            _ellipse_mask(previous.shape, previous_toe, radius, 0.82 * radius),
            _ellipse_mask(current.shape, current_toe, current_radius, 0.82 * current_radius),
            cfg["min_landmark_tracks"],
        ),
        "ground": (
            _ground_mask(previous.shape, previous_heel, previous_toe, radius, cfg),
            _ground_mask(current.shape, current_heel, current_toe, current_radius, cfg),
            cfg["min_ground_tracks"],
        ),
    }
    regions = {
        "heel": _region_track(
            previous,
            current,
            *region_specs["heel"][:2],
            affine,
            region_specs["heel"][2],
            cfg,
            expected_displacement=heel_displacement,
            expected_residual=heel_residual,
        ),
        "toe": _region_track(
            previous,
            current,
            *region_specs["toe"][:2],
            affine,
            region_specs["toe"][2],
            cfg,
            expected_displacement=toe_displacement,
            expected_residual=toe_residual,
        ),
        "ground": _region_track(
            previous,
            current,
            *region_specs["ground"][:2],
            affine,
            region_specs["ground"][2],
            cfg,
        ),
    }
    if regions["ground"] is None:
        return None, "weak_ground_tracks"
    if regions["heel"] is None or regions["toe"] is None:
        return None, "weak_foot_tracks"
    foot_residual = np.median(
        np.vstack((regions["heel"]["_residual"], regions["toe"]["_residual"])), axis=0
    )
    ground_residual = regions["ground"]["_residual"]
    relative = foot_residual - ground_residual
    relative_speed = float(np.linalg.norm(relative) / dt_s)
    vertical_speed = float(relative[1] / dt_s)
    staticness = max(0.0, 1.0 - relative_speed / cfg["max_post_speed_px_s"])
    qualities = [regions[name]["quality"] for name in ("heel", "toe", "ground")]
    for region in regions.values():
        region.pop("_residual", None)
    return {
        "from_frame": from_frame,
        "to_frame": to_frame,
        "dt_s": round(dt_s, 8),
        "background": background["evidence"],
        "heel": regions["heel"],
        "toe": regions["toe"],
        "ground": regions["ground"],
        "foot_vs_ground_residual_xy_px": [round(float(value), 6) for value in relative],
        "foot_vs_ground_shear_px": round(float(np.linalg.norm(relative)), 6),
        "relative_vertical_velocity_px_s": round(vertical_speed, 6),
        "relative_speed_px_s": round(relative_speed, 6),
        "staticness": round(staticness, 6),
        "track_quality": round(float(np.mean(qualities)), 6),
        "pose_guidance_role": "track_assignment_only_not_contact_evidence",
        "camera_motion_is_contact_evidence": False,
    }, None


def analyze_candidate_window(
    frames: Iterable[np.ndarray],
    source_pts_us: Sequence[int],
    heel_xy: Sequence[Sequence[float] | None],
    toe_xy: Sequence[Sequence[float] | None],
    *,
    actor_boxes: Sequence[Sequence[float] | None] | None = None,
    frame_indices: Sequence[int] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze one bounded pose-guided window and suggest a contact or abstain."""

    cfg = _config(config)
    try:
        prepared, factor, budget_exceeded = _prepare_frames(
            frames, cfg["max_width"], cfg["max_window_frames"]
        )
    except ValueError as exc:
        return _abstain("geometry_failure", str(exc), frame_count=0)
    if budget_exceeded:
        return _abstain(
            "window_budget_exceeded",
            f"window exceeds the {cfg['max_window_frames']}-frame CPU limit",
            frame_count=cfg["max_window_frames"] + 1,
        )
    count = len(prepared)
    if count < 4:
        return _abstain("geometry_failure", "at least four frames are required", frame_count=count)
    if len(source_pts_us) != count:
        return _abstain("geometry_failure", "PTS length must equal frame count", frame_count=count)
    pts = [int(value) for value in source_pts_us]
    if any(current <= previous for previous, current in zip(pts, pts[1:])):
        return _abstain(
            "non_monotonic_timestamps",
            "source PTS must be strictly increasing",
            frame_count=count,
            scale=factor,
        )
    indices = (
        list(range(count))
        if frame_indices is None
        else [int(value) for value in frame_indices]
    )
    if len(indices) != count or any(b <= a for a, b in zip(indices, indices[1:])):
        return _abstain(
            "geometry_failure",
            "frame indices must be strictly increasing and match the window",
            frame_count=count,
            scale=factor,
        )
    try:
        heels = _points(heel_xy, count, factor)
        toes = _points(toe_xy, count, factor)
    except (TypeError, ValueError) as exc:
        return _abstain("geometry_failure", str(exc), frame_count=count, scale=factor)
    if actor_boxes is None:
        return _abstain(
            "missing_actor_exclusion",
            "verified actor boxes are required to exclude moving body features",
            frame_count=count,
            scale=factor,
        )
    height, width = prepared[0].shape
    boxes = _boxes(actor_boxes, heels, toes, factor, width, height, cfg)
    if boxes is None or not _regions_valid(heels, toes, width, height, cfg):
        return _abstain(
            "geometry_failure",
            "heel/toe, actor exclusion, or adjacent-ground ROI is invalid",
            frame_count=count,
            scale=factor,
        )

    pairs: list[dict[str, Any]] = []
    for position in range(1, count):
        dt_s = (pts[position] - pts[position - 1]) / 1_000_000.0
        evidence, reason = _pair_evidence(
            prepared[position - 1],
            prepared[position],
            heels[position - 1],
            heels[position],
            toes[position - 1],
            toes[position],
            boxes[position - 1],
            boxes[position],
            dt_s,
            indices[position - 1],
            indices[position],
            cfg,
        )
        if evidence is None:
            details = {
                "edit_cut": "appearance/track discontinuity is consistent with an edit cut",
                "geometry_failure": "background affine geometry is implausible",
                "weak_background_compensation": "too few reliable distributed background tracks",
                "weak_ground_tracks": "adjacent ground lacks reliable sparse tracks",
                "weak_foot_tracks": "heel or toe region lacks reliable sparse texture/tracks",
            }
            return _abstain(
                reason or "weak_background_compensation",
                details.get(reason, "sparse tracking failed closed"),
                frame_count=count,
                pair_evidence=pairs,
                scale=factor,
            )
        pairs.append(evidence)

    candidates: list[dict[str, Any]] = []
    for pair_index in range(1, len(pairs) - 1):
        before, current, after = pairs[pair_index - 1 : pair_index + 2]
        pre_vertical = before["relative_vertical_velocity_px_s"]
        current_vertical = current["relative_vertical_velocity_px_s"]
        post_speed = max(current["relative_speed_px_s"], after["relative_speed_px_s"])
        transition_dt = 0.5 * (before["dt_s"] + current["dt_s"])
        deceleration = max(0.0, (pre_vertical - current_vertical) / transition_dt)
        transition = max(0.0, pre_vertical - max(0.0, current_vertical)) / max(
            pre_vertical, cfg["min_pre_vertical_speed_px_s"]
        )
        staticness = min(current["staticness"], after["staticness"])
        quality = min(current["track_quality"], after["track_quality"])
        deceleration_score = min(1.0, deceleration / cfg["min_vertical_deceleration_px_s2"])
        stop_score = max(0.0, 1.0 - post_speed / cfg["max_post_speed_px_s"])
        score = 0.34 * deceleration_score + 0.26 * transition + 0.22 * stop_score + 0.18 * quality
        valid = (
            pre_vertical >= cfg["min_pre_vertical_speed_px_s"]
            and abs(current_vertical) <= cfg["max_post_speed_px_s"]
            and post_speed <= cfg["max_post_speed_px_s"]
            and deceleration >= cfg["min_vertical_deceleration_px_s2"]
        )
        candidates.append(
            {
                "pair_index": pair_index,
                "frame": current["to_frame"],
                "window_position": pair_index + 1,
                "pre_vertical_velocity_px_s": round(pre_vertical, 6),
                "post_vertical_velocity_px_s": round(current_vertical, 6),
                "post_relative_speed_px_s": round(post_speed, 6),
                "vertical_deceleration_px_s2": round(deceleration, 6),
                "staticness": round(staticness, 6),
                "track_quality": round(quality, 6),
                "score": round(score if valid else 0.0, 6),
                "valid": valid,
            }
        )
    valid_candidates = [item for item in candidates if item["valid"]]
    if not valid_candidates:
        return _abstain(
            "no_contact_change",
            "no descending-to-static relative foot change point",
            frame_count=count,
            pair_evidence=pairs,
            scale=factor,
        )
    best = min(valid_candidates, key=lambda item: (-item["score"], item["frame"]))
    if best["score"] < cfg["min_contact_score"]:
        return _abstain(
            "weak_contact_change",
            f"best sparse moving-to-static score {best['score']:.3f} is below threshold",
            frame_count=count,
            pair_evidence=pairs,
            scale=factor,
        )
    position = best["window_position"]
    interval = [int(round((pts[position - 1] + pts[position]) / 2))]
    interval.append(int(round((pts[position] + pts[position + 1]) / 2)))
    return {
        "method": METHOD,
        "status": "suggested",
        "reason": "sparse_relative_deceleration",
        "detail": (
            "pose-guided foot tracks decelerate and remain static relative to adjacent ground"
        ),
        "frame_count": count,
        "processing_scale": round(factor, 8),
        "pair_evidence": pairs,
        "candidate_evidence": candidates,
        "camera_motion_is_contact_evidence": False,
        "suggested_frame": best["frame"],
        "suggested_pts_us": pts[position],
        "suggested_interval_pts_us": interval,
        "score": best["score"],
    }


def decode_candidate_window(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    *,
    max_frames: int = int(SPARSE_ROI_CONFIG["max_window_frames"]),
) -> list[np.ndarray]:
    """Decode an inclusive bounded window exactly once."""

    import cv2

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("invalid inclusive decode window")
    frame_count = end_frame - start_frame + 1
    if max_frames < 1 or max_frames > HARD_MAX_WINDOW_FRAMES:
        raise ValueError("invalid decode CPU bound")
    if frame_count > max_frames:
        raise ValueError(f"decode window exceeds {max_frames}-frame CPU limit")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open candidate video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        reported_start = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if math.isfinite(reported_start) and abs(reported_start - start_frame) > 0.5:
            raise RuntimeError(
                f"decoder did not seek to verified frame {start_frame}: {reported_start}"
            )
        for expected_index in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"decoder stopped before verified frame {expected_index}")
            reported_next = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if math.isfinite(reported_next) and abs(reported_next - (expected_index + 1)) > 0.5:
                raise RuntimeError(
                    "decoder frame identity diverged from verified mapping at "
                    f"{expected_index}: next={reported_next}"
                )
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _av_pts_us(frame: Any) -> int:
    if frame.pts is None or frame.time_base is None:
        raise RuntimeError("decoded video frame has no source PTS")
    time_base = frame.time_base
    value = (
        Decimal(int(frame.pts))
        * Decimal(int(time_base.numerator))
        / Decimal(int(time_base.denominator))
        * Decimal(1_000_000)
    )
    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def decode_verified_candidate_window(
    video_path: str | Path,
    verified_frame_pts: Sequence[Sequence[int]],
    *,
    max_frames: int = int(SPARSE_ROI_CONFIG["max_window_frames"]),
) -> list[np.ndarray]:
    """Decode a bounded presentation window and verify every actual source PTS.

    Frame indices are provenance carried by the caller. Pixel identity is
    established independently from PyAV's decoded frame PTS; no frame/fps or
    OpenCV seek position is reinterpreted as source time.
    """

    try:
        mapping = [(int(item[0]), int(item[1])) for item in verified_frame_pts]
    except (IndexError, TypeError, ValueError):
        raise ValueError("verified frame/PTS mapping is invalid") from None
    if not mapping or len(mapping) > max_frames or max_frames > HARD_MAX_WINDOW_FRAMES:
        raise ValueError("verified decode window violates the bounded frame budget")
    if any(
        next_frame != frame_index + 1 or next_pts <= pts
        for (frame_index, pts), (next_frame, next_pts) in zip(mapping, mapping[1:])
    ):
        raise ValueError("verified frame/PTS mapping must be contiguous and monotonic")

    try:
        import av
    except ImportError:
        raise RuntimeError("PyAV is required for verified source-PTS video decoding") from None

    expected_pts = [item[1] for item in mapping]
    frames: list[np.ndarray] = []
    decoded = 0
    try:
        with av.open(str(video_path), mode="r") as container:
            if not container.streams.video:
                raise RuntimeError("verified decoder found no video stream")
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise RuntimeError("verified decoder video stream has no time base")
            target = Fraction(expected_pts[0], 1_000_000)
            seek_offset = int(target / stream.time_base)
            container.seek(seek_offset, stream=stream, backward=True, any_frame=False)
            position = 0
            for frame in container.decode(stream):
                decoded += 1
                if decoded > len(mapping) + HARD_MAX_SEEK_PREROLL_FRAMES:
                    raise RuntimeError("verified decoder exceeded bounded keyframe preroll")
                pts_us = _av_pts_us(frame)
                if pts_us < expected_pts[0]:
                    continue
                if position >= len(expected_pts):
                    break
                if pts_us != expected_pts[position]:
                    raise RuntimeError(
                        "decoded source PTS diverged from verified mapping at "
                        f"frame {mapping[position][0]}: {pts_us} != {expected_pts[position]}"
                    )
                frames.append(frame.to_ndarray(format="bgr24"))
                position += 1
                if position == len(expected_pts):
                    break
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"verified video decode failed: {exc}") from None
    if len(frames) != len(mapping):
        raise RuntimeError(
            f"verified decoder returned {len(frames)}/{len(mapping)} expected frames"
        )
    return frames


def analyze_video_candidate_window(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    source_pts_us: Sequence[int],
    heel_xy: Sequence[Sequence[float] | None],
    toe_xy: Sequence[Sequence[float] | None],
    *,
    verified_frame_pts: Sequence[Sequence[int]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Analyze video only with an explicit verified frame-index/PTS contract."""

    expected_count = end_frame - start_frame + 1
    if start_frame < 0 or expected_count < 1:
        return _abstain("geometry_failure", "invalid inclusive video window", frame_count=0)
    try:
        wrapper_cfg = _config(kwargs.get("config"))
    except ValueError as exc:
        return _abstain("geometry_failure", str(exc), frame_count=expected_count)
    if expected_count > wrapper_cfg["max_window_frames"]:
        return _abstain(
            "window_budget_exceeded",
            f"video window exceeds the {wrapper_cfg['max_window_frames']}-frame CPU limit",
            frame_count=expected_count,
        )
    try:
        expected_mapping = list(
            zip(range(start_frame, end_frame + 1), map(int, source_pts_us))
        )
        supplied_mapping = (
            None
            if verified_frame_pts is None
            else [(int(item[0]), int(item[1])) for item in verified_frame_pts]
        )
    except (IndexError, TypeError, ValueError):
        expected_mapping = []
        supplied_mapping = None
    if (
        len(source_pts_us) != expected_count
        or supplied_mapping is None
        or supplied_mapping != expected_mapping
    ):
        return _abstain(
            "unverified_video_timestamps",
            "video analysis requires an exact verified frame-index to source-PTS mapping",
            frame_count=expected_count,
        )
    if "frame_indices" in kwargs:
        return _abstain(
            "unverified_video_timestamps",
            "video frame indices come only from the verified mapping contract",
            frame_count=expected_count,
        )
    try:
        frames = decode_verified_candidate_window(
            video_path,
            supplied_mapping,
            max_frames=wrapper_cfg["max_window_frames"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return _abstain(
            "decode_identity_failure",
            str(exc),
            frame_count=expected_count,
        )
    return analyze_candidate_window(
        frames,
        source_pts_us,
        heel_xy,
        toe_xy,
        frame_indices=[item[0] for item in supplied_mapping],
        **kwargs,
    )
