"""Lightweight annotation helpers for the auto-sfx POC.

Small, dependency-free utilities used while manually marking foot-contact
events: finding the frame neighbourhood around a timestamp, pulling those
exact frames out of a video with ffmpeg/ffprobe, and printing aligned text
tables of annotation sessions. Pure stdlib; the extraction helpers shell
out to the ``ffmpeg`` / ``ffprobe`` binaries when available.
"""

from __future__ import annotations

import bisect
import math
import shutil
import subprocess
from pathlib import Path

# Half-width of the probe/extract window around a timestamp, in seconds.
DEFAULT_HALF_WINDOW_S = 0.5

# Backwards nudge applied to precise seeks so rounding at 6 decimal places
# can never skip past the target frame.
SEEK_EPSILON_S = 0.002


def neighbors(frame_times, t_s, k=3):
    """Return ``[(frame_index, t_s)]`` for up to ``k`` frames before and after.

    ``frame_times`` must be ascending (decoded frame order). The centre is
    the nearest frame to ``t_s`` (lowest index wins ties); the result is the
    contiguous index window clipped to the sequence bounds, in ascending
    order. Returns ``[]`` for empty input.
    """
    if not frame_times:
        return []
    k = max(0, int(k))
    n = len(frame_times)

    pos = bisect.bisect_left(frame_times, t_s)
    cand = [i for i in (pos - 1, pos) if 0 <= i < n]
    centre = min(cand, key=lambda i: (abs(frame_times[i] - t_s), i))

    lo = max(0, centre - k)
    hi = min(n, centre + k + 1)
    return [(i, frame_times[i]) for i in range(lo, hi)]


def _probe_frame_times(video_path, start_s, end_s):
    """Frame presentation times within [start_s, end_s], via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        "-read_intervals", f"{start_s:.6f}%{end_s:.6f}",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {proc.stderr.strip()}")
    times = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line == "N/A":
            continue
        t = float(line)
        if math.isfinite(t):
            times.append(t)
    return times


def extract_frames(video_path, t_s, k=3, out_dir=None, half_window=DEFAULT_HALF_WINDOW_S):
    """Extract the ``2k + 1`` frames nearest ``t_s`` as PNG files.

    Frames are probed over ``[t_s - half_window, t_s + half_window]``, the
    ``k`` nearest on each side of ``t_s`` selected (see :func:`neighbors`),
    and each pulled with a precise ffmpeg seek. Files are named
    ``frame_<idx>_t<us>.png`` where ``<idx>`` is the 0-based position within
    the probed window and ``<us>`` is the frame's exact microsecond PTS;
    names therefore sort chronologically.

    Returns the list of written paths, sorted. Raises ``RuntimeError`` when
    ffprobe/ffmpeg fail or the window contains no decodable frames.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    video = Path(video_path)
    if out_dir is None:
        out_dir = Path(f"frames_t{int(round(t_s * 1_000_000))}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = max(0.0, float(t_s) - float(half_window))
    end = float(t_s) + float(half_window)
    times = _probe_frame_times(video, start, end)
    if not times:
        raise RuntimeError(
            f"no frames probed in [{start:.3f}, {end:.3f}] of {video}; "
            "try a larger half_window"
        )

    picked = neighbors(times, t_s, k)
    paths = []
    for idx, tau in picked:
        dest = out_dir / f"frame_{idx:04d}_t{int(round(tau * 1_000_000))}.png"
        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, tau - SEEK_EPSILON_S):.6f}",
            "-i", str(video),
            "-frames:v", "1",
            "-update", "1",
            "-y", str(dest),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not dest.exists():
            raise RuntimeError(f"ffmpeg failed extracting t={tau:.6f}: {proc.stderr.strip()}")
        paths.append(dest)
    return sorted(paths)


def print_table(rows):
    """Print a list of dicts as plain aligned text to stdout.

    Columns appear in first-seen key order across all rows; missing values
    render as blanks and floats as 6-decimal rounded strings. No-op message
    for empty input.
    """
    rows = list(rows)
    if not rows:
        print("(no rows)")
        return

    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    widths = {
        col: max([len(col)] + [len(fmt(row.get(col))) for row in rows])
        for col in columns
    }

    def render(cells):
        return "  ".join(str(cell).ljust(widths[col]) for cell, col in zip(cells, columns))

    print(render(columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print(render(fmt(row.get(col)) for col in columns))
