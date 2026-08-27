"""Synthetic tests for shot-local support geometry (no real clips)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_sfx.state_contacts import _rolling_quantile  # noqa: E402
from auto_sfx.support_geometry import (  # noqa: E402
    DEFAULT_SUPPORT_GEOMETRY_CONFIG,
    estimate_shot_support,
    validate_support_geometry_config,
)


def _pack(xs, ys, speeds, scale, start=0):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    n = xs.size
    times = np.arange(n, dtype=float) / 30.0
    idx = np.arange(start, start + n, dtype=int)
    return estimate_shot_support(xs, ys, speeds, times, idx, scale)


def test_horizontal_ground_planted_near_zero_swing_positive():
    x_plant = np.linspace(20.0, 220.0, 25)
    y_plant = np.full(25, 180.0)
    speed_plant = np.full(25, 0.02)
    x_swing = np.linspace(40.0, 180.0, 15)
    y_swing = np.full(15, 120.0)
    speed_swing = np.full(15, 1.4)
    xs = np.concatenate([x_plant, x_swing])
    ys = np.concatenate([y_plant, y_swing])
    speeds = np.concatenate([speed_plant, speed_swing])
    scale = 80.0
    geom = _pack(xs, ys, speeds, scale)
    assert geom.kind == "line"
    assert geom.reason is None
    planted_h = [geom.height_bs(x, 180.0, scale) for x in x_plant]
    swing_h = [geom.height_bs(x, 120.0, scale) for x in x_swing]
    assert all(np.isfinite(h) for h in planted_h + swing_h)
    assert max(abs(h) for h in planted_h) < 0.03
    assert min(swing_h) > 0.4
    assert geom.height_bs(100.0, 90.0, scale) > min(swing_h)


def test_tilted_camera_plants_near_zero_unlike_rolling_y():
    x_plant = np.linspace(40.0, 240.0, 21)
    y_plant = 0.35 * x_plant + 90.0
    speed_plant = np.full(x_plant.size, 0.03)
    x_swing = x_plant[::2]
    y_swing = 0.35 * x_swing + 90.0 - 45.0
    speed_swing = np.full(x_swing.size, 1.1)
    xs = np.concatenate([x_plant, x_swing])
    ys = np.concatenate([y_plant, y_swing])
    speeds = np.concatenate([speed_plant, speed_swing])
    scale = 90.0
    geom = _pack(xs, ys, speeds, scale)
    assert geom.kind == "line"
    plant_h = np.array([geom.height_bs(x, y, scale) for x, y in zip(x_plant, y_plant)])
    assert np.max(np.abs(plant_h)) < 0.04
    # A rolling image-y floor treats the uphill (smaller-y) plants as airborne
    # relative to later downhill plants.
    series_y = np.concatenate([y_plant, y_swing])
    floor = _rolling_quantile(series_y, half=8, q=0.92, fallback=np.nan)
    rolling_h = (floor[: x_plant.size] - y_plant) / scale
    assert rolling_h[0] - rolling_h[-1] > 0.15


def test_vertical_too_few_huge_residual_abstain():
    scale = 60.0
    vertical = _pack(
        np.full(16, 100.0),
        np.linspace(80.0, 200.0, 16),
        np.full(16, 0.02),
        scale,
    )
    assert vertical.kind == "abstain"
    assert vertical.reason == "near_vertical"

    few = _pack(
        np.array([10.0, 20.0, 30.0]),
        np.array([200.0, 201.0, 199.0]),
        np.array([0.01, 0.01, 0.02]),
        scale,
    )
    assert few.kind == "abstain"
    assert few.reason == "too_few_seeds"

    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    cloud = _pack(
        80.0 + 55.0 * np.cos(angles),
        80.0 + 55.0 * np.sin(angles),
        np.full(16, 0.02),
        scale,
    )
    assert cloud.kind == "abstain"

    bad_scale = _pack(
        np.linspace(0.0, 100.0, 12),
        np.full(12, 50.0),
        np.full(12, 0.01),
        0.0,
    )
    assert bad_scale.kind == "abstain"
    assert bad_scale.reason == "invalid_scale"


def test_validate_support_geometry_unknown_and_offset_keys():
    with pytest.raises(ValueError):
        validate_support_geometry_config({"global_offset": 2})
    with pytest.raises(ValueError):
        validate_support_geometry_config({"not_a_real_key": 1})
    cfg = validate_support_geometry_config({"min_points": 10})
    assert cfg["min_points"] == 10
    assert cfg["max_residual_rms_bs"] == DEFAULT_SUPPORT_GEOMETRY_CONFIG["max_residual_rms_bs"]
