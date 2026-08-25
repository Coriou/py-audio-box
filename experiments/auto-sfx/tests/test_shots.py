"""Shot-boundary detection tests: a single hard cut must be found once."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from auto_sfx.shots import detect_shot_boundaries

FPS = 30.0


def _write_hard_cut_clip(
    path: Path,
    n_frames: int = 24,
    width: int = 64,
    height: int = 48,
    fps: float = FPS,
) -> Path:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened(), "VideoWriter refused to open"
    half = n_frames // 2
    for i in range(n_frames):
        level = 0 if i < half else 255  # black -> white hard cut midway
        writer.write(np.full((height, width, 3), level, dtype=np.uint8))
    writer.release()
    return path


def test_single_boundary_at_hard_cut(tmp_path):
    clip = _write_hard_cut_clip(tmp_path / "cut.mp4")
    cut_t = 12.0 / FPS  # first frame of the white shot

    bounds = detect_shot_boundaries(clip)

    assert len(bounds) == 1, f"expected exactly one boundary, got {bounds}"
    assert abs(bounds[0] - cut_t) <= 2.0 / FPS
