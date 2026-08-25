"""Media ingest for the auto-sfx POC.

Owns the video -> (metadata, per-frame timestamps, raw frames) surface that
every later stage consumes:

* :func:`probe_video` -- container/stream metadata via ``ffprobe``; frame
  count, duration and the CFR/VFR verdict come from the *actually decoded*
  PTS timeline, never from container claims. ``nb_read_packets`` /
  ``nb_frames`` are deliberately ignored: they are frequently absent or
  wrong, and the honest number is the decodable one.
* :func:`collect_frame_times` -- the authoritative frame -> seconds mapping
  used across the pipeline.
* :class:`FrameReader` -- streamed BGR frames paired 1:1 with those
  timestamps.

numpy is imported lazily inside functions so pure-python unit tests can
import this module without it installed.
"""

from __future__ import annotations

import json
import logging
import statistics
import subprocess
from pathlib import Path

from auto_sfx.types import VideoMeta

logger = logging.getLogger(__name__)

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

#: Relative spread of consecutive frame deltas tolerated for CFR:
#: (max(delta) - min(delta)) / median(delta) must not exceed this.
CFR_SPREAD_TOLERANCE = 0.02

__all__ = [
    "FFMPEG",
    "FFPROBE",
    "CFR_SPREAD_TOLERANCE",
    "FrameReader",
    "collect_frame_times",
    "probe_video",
]


# ---------------------------------------------------------------------------
# ffprobe plumbing
# ---------------------------------------------------------------------------


def _run_ffprobe(args: list[str]) -> dict:
    """Run ``ffprobe`` with JSON output and return the parsed document."""
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", *args]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (args={args!r}): {(proc.stderr or '').strip()}"
        )
    return json.loads(proc.stdout or "{}")


def _parse_ratio(value: str | None, default: float | None = None) -> float | None:
    """Parse an ffprobe rational string like ``30000/1001`` (or ``N/A``)."""
    if value is None:
        return default
    value = str(value).strip()
    if not value or value == "0/0":
        return default
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            den_f = float(den)
        except ValueError:
            return default
        if den_f == 0.0:
            return default
        try:
            return float(num) / den_f
        except ValueError:
            return default
    try:
        return float(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# PTS timeline
# ---------------------------------------------------------------------------


def collect_frame_times(path: str | Path) -> list[float]:
    """Per-frame source presentation timestamps in seconds (video stream 0).

    This shells out to ``ffprobe`` for a second full decode pass instead of
    reusing OpenCV's per-frame clock. The double pass is deliberate:
    OpenCV's PTS (``CAP_PROP_POS_MSEC`` and friends) is unreliable for VFR
    material on several backends, and an honest frame->seconds mapping is
    the foundation everything downstream stands on. The extra decode cost is
    real but paid once per video; its price is measured separately by the
    caller, never hidden here.
    """
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe frame scan failed for {path}: {(proc.stderr or '').strip()}"
        )
    times: list[float] = []
    for raw_line in proc.stdout.splitlines():
        # ffprobe's csv writer can emit trailing separators / blank rows when
        # a frame lacks the requested key; never let that poison the map.
        line = raw_line.strip().rstrip(",").strip()
        if not line or line.upper() == "N/A":
            continue
        try:
            times.append(float(line))
        except ValueError:
            logger.warning(
                "collect_frame_times(%s): unparsable timestamp line %r",
                path,
                line,
            )
    return times


def _summarise_decoded_times(
    times: list[float],
) -> tuple[int, float, bool, bool]:
    """Decide (n_frames, duration_s, cfr, vfr) from a decoded PTS timeline.

    duration is ``last PTS - first PTS`` plus one median frame delta for the
    final frame's display time. Degenerate timelines (0 or 1 frames) cannot
    demonstrate a constant rate, so per the decision rule they fall through
    to ``vfr``.
    """
    n = len(times)
    if n == 0:
        return 0, 0.0, False, True
    deltas = [b - a for a, b in zip(times, times[1:])]
    if not deltas:
        return n, 0.0, False, True
    median_d = statistics.median(deltas)
    if median_d > 0.0:
        spread = (max(deltas) - min(deltas)) / median_d
        cfr = spread <= CFR_SPREAD_TOLERANCE
    else:
        cfr = False
    duration = (times[-1] - times[0]) + (median_d if median_d > 0.0 else 0.0)
    return n, duration, cfr, not cfr


def probe_video(path: str | Path) -> VideoMeta:
    """Probe a video file and return its :class:`VideoMeta`.

    Stream facts (codec, dimensions, r_frame_rate, time_base, audio
    presence) come from ``ffprobe -show_streams -show_format``; n_frames,
    duration and the CFR/VFR verdict come from the actually decoded PTS
    timeline (:func:`collect_frame_times`) -- container frame-count claims
    such as ``nb_read_packets`` are intentionally not trusted.
    """
    path = str(path)
    info = _run_ffprobe(["-show_streams", "-show_format", path])
    streams = info.get("streams") or []
    video = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    if video is None:
        raise RuntimeError(f"no video stream found in {path}")

    width = int(video["width"])
    height = int(video["height"])
    fps_stream = _parse_ratio(video.get("r_frame_rate"), default=0.0) or 0.0
    codec = str(video.get("codec_name", ""))
    time_base = _parse_ratio(video.get("time_base"), default=None)
    audio_present = any(s.get("codec_type") == "audio" for s in streams)

    n_frames, duration_s, cfr, vfr = _summarise_decoded_times(
        collect_frame_times(path)
    )

    return VideoMeta(
        path=path,
        codec=codec,
        width=width,
        height=height,
        fps_stream=float(fps_stream),
        n_frames=n_frames,
        duration_s=duration_s,
        cfr=cfr,
        vfr=vfr,
        time_base=time_base,
        audio_present=audio_present,
    )


# ---------------------------------------------------------------------------
# Frame streaming
# ---------------------------------------------------------------------------


def _decode_bgr(buf: bytes, width: int, height: int):
    """Decode one rawvideo bgr24 buffer into a WRITABLE HxWx3 uint8 array.

    numpy is imported lazily (see module docstring). ``np.frombuffer``
    yields a read-only view over the pipe buffer; the single copy here
    makes every downstream consumer (pose backends, renderer overlays)
    safe to mutate or annotate frames in place. The copy cost (~1 MiB per
    720p frame) is negligible against pose inference.
    """
    import numpy as np

    frame = np.frombuffer(buf, dtype=np.uint8)
    if frame.size != width * height * 3:  # defensive; callers pre-check
        raise RuntimeError(
            f"short rawvideo frame: got {frame.size} bytes, "
            f"expected {width * height * 3}"
        )
    return frame.reshape(height, width, 3).copy()


class FrameReader:
    """Stream decoded BGR frames paired with source PTS.

    Usage::

        with FrameReader(path) as reader:
            for frame_bgr, t_s, frame_index in reader:
                ...

    Frames come from ``ffmpeg -nostdin -i <path> -f rawvideo -pix_fmt
    bgr24 -`` piped to stdout; timestamps come from
    :func:`collect_frame_times` (a separate ffprobe pass -- deliberate, see
    its docstring). Frames and timestamps are paired positionally, like
    ``zip(..., strict=False)``: iteration stops at whichever ends first and
    a count mismatch is logged once via ``logging``.

    ``frame_index`` counts absolutely decoded frames (0-based from the start
    of the file), so ``start_index`` skips leading frames (they are decoded
    and discarded to keep index/PTS alignment) and ``max_frames`` caps how
    many frames are yielded. Iteration is single-pass; a reader raises
    ``StopIteration`` for good once exhausted. ``close()`` terminates the
    ffmpeg child process; it is also invoked on garbage collection.
    """

    def __init__(
        self,
        path: str | Path,
        start_index: int = 0,
        max_frames: int | None = None,
    ) -> None:
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")
        if max_frames is not None and max_frames < 0:
            raise ValueError(f"max_frames must be >= 0, got {max_frames}")
        self.path = str(path)
        self.start_index = int(start_index)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.width: int | None = None
        self.height: int | None = None
        self.times: list[float] = []
        self._frame_bytes = 0
        self._proc: subprocess.Popen | None = None
        self._raw_read = 0
        self._yielded = 0
        self._finished = False
        self._mismatch_warned = False

    # -- lifecycle ----------------------------------------------------------

    def _open(self) -> None:
        if self._proc is not None:
            return
        info = _run_ffprobe(
            [
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                self.path,
            ]
        )
        streams = info.get("streams") or []
        if not streams:
            raise RuntimeError(f"no video stream found in {self.path}")
        self.width = int(streams[0]["width"])
        self.height = int(streams[0]["height"])
        self._frame_bytes = self.width * self.height * 3
        self.times = collect_frame_times(self.path)
        cmd = [
            FFMPEG,
            "-nostdin",
            "-i",
            self.path,
            # Passthrough: never let ffmpeg re-time VFR input to CFR for the
            # rawvideo pipe -- duplicated frames would desync the positional
            # pairing with collect_frame_times().
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def close(self) -> None:
        """Terminate the ffmpeg child and release the pipe. Idempotent."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

    def __enter__(self) -> "FrameReader":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- iteration ----------------------------------------------------------

    def __iter__(self) -> "FrameReader":
        return self

    def _warn_count_mismatch(self, msg: str) -> None:
        if self._mismatch_warned:
            return
        self._mismatch_warned = True
        logger.warning("FrameReader(%s): %s", self.path, msg)

    def __next__(self):
        if self._finished:
            raise StopIteration
        if self._proc is None:
            self._open()
        if self.max_frames is not None and self._yielded >= self.max_frames:
            self._finished = True
            self.close()
            raise StopIteration
        while True:
            buf = self._proc.stdout.read(self._frame_bytes)
            idx = self._raw_read
            self._raw_read += 1
            if buf is None or len(buf) < self._frame_bytes:
                # EOF: compare fully decoded frames against the PTS list.
                if idx != len(self.times):
                    self._warn_count_mismatch(
                        f"decoded {idx} raw frames but collected "
                        f"{len(self.times)} PTS values"
                    )
                self._finished = True
                self.close()
                raise StopIteration
            if idx >= len(self.times):
                # More raw frames than timestamps: cannot pair further.
                self._warn_count_mismatch(
                    f"decoded more than {len(self.times)} raw frames but "
                    f"only {len(self.times)} PTS values were collected"
                )
                self._finished = True
                self.close()
                raise StopIteration
            if idx < self.start_index:
                continue  # discard pre-start frames; consume their PTS slot
            frame = _decode_bgr(buf, self.width, self.height)
            self._yielded += 1
            return frame, self.times[idx], idx
