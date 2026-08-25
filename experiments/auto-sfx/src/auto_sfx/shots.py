"""Shot-boundary detection for the auto-sfx POC.

Cuts a long video into shots before pose tracking. Boundaries are reported
as source PTS seconds -- the same clock as
:func:`auto_sfx.ingest.collect_frame_times` -- so shot ranges map directly
onto decoded frame indices.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

FFMPEG = "ffmpeg"

# showinfo log lines look like:
#   [Parsed_showinfo_1 @ 0x55..] n:   0 pts:12345 pts_time:0.411500 ...
_SHOWINFO_PTS_TIME = re.compile(r"pts_time:(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

__all__ = ["detect_shot_boundaries"]


def detect_shot_boundaries(path: str | Path, threshold: float = 0.3) -> list[float]:
    """Return sorted unique PTS times (seconds) of hard shot boundaries.

    Runs ``ffmpeg -vf select='gt(scene,T)',showinfo -f null -`` and parses
    ``pts_time:`` out of the showinfo log lines on stderr (ffmpeg writes its
    filter log there; stdout belongs to the null muxer). ``threshold`` is
    ffmpeg's scene score cutoff in [0, 1].

    Raises RuntimeError when ffmpeg fails outright and nothing could be
    parsed; a nonzero exit with parsed boundaries degrades to a warning.
    """
    path = str(path)
    vf = f"select='gt(scene,{threshold})',showinfo"
    cmd = [
        FFMPEG,
        "-nostdin",
        "-i",
        path,
        "-vf",
        vf,
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    times: set[float] = set()
    for line in proc.stderr.splitlines():
        if "[Parsed_showinfo" not in line:
            continue
        m = _SHOWINFO_PTS_TIME.search(line)
        if m:
            times.add(float(m.group(1)))

    if proc.returncode != 0:
        if not times:
            raise RuntimeError(
                f"ffmpeg scene detection failed for {path}: "
                f"{(proc.stderr or '').strip()[-500:]}"
            )
        logger.warning(
            "detect_shot_boundaries(%s): ffmpeg exited %d but %d boundaries "
            "were parsed; returning what was recovered",
            path,
            proc.returncode,
            len(times),
        )

    result = sorted(times)
    logger.debug(
        "detect_shot_boundaries(%s, threshold=%s): %d boundary(ies) %s",
        path,
        threshold,
        len(result),
        result,
    )
    return result
