"""Stage/resource measurement for the auto-sfx POC.

Wall-clock stage timings plus a background RSS sampler, rolled into one
JSON-safe ResourceReport for diagnostics output.

RSS notes
---------
- Linux: instantaneous resident set from /proc/self/statm.
- Other platforms (e.g. macOS dev hosts): no cheap instantaneous RSS exists,
  so the sampler records the getrusage peak water-mark instead — values are
  correct bytes but monotonic non-decreasing. peak_rss_bytes is always the
  true process peak either way.
"""

from __future__ import annotations

import os
import resource
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# Raw sampling
# ---------------------------------------------------------------------------


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 4096


def current_rss_bytes() -> int:
    """Best-effort resident/peak RSS in bytes (never None)."""

    try:                                            # Linux: instant resident
        with open("/proc/self/statm", "rb") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * _page_size()
    except (OSError, IndexError, ValueError):
        pass
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru) * (1 if sys.platform == "darwin" else 1024)


def peak_rss_bytes() -> int:
    """OS-reported peak RSS for this process, normalised to bytes."""

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru) * (1 if sys.platform == "darwin" else 1024)


# ---------------------------------------------------------------------------
# Sampler thread
# ---------------------------------------------------------------------------


class RssSampler:
    """Daemon thread appending (t_rel_s, rss_bytes) samples until stopped."""

    def __init__(self, interval_s: float = 0.05):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = interval_s
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "RssSampler":
        if self._thread is not None:
            return self
        t0 = time.monotonic()

        def _loop() -> None:
            while not self._stop.is_set():
                self.samples.append(
                    (time.monotonic() - t0, current_rss_bytes())
                )
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=_loop, name="auto-sfx-rss", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> list[tuple[float, int]]:
        """Stop sampling; returns the collected timeline."""

        if self._thread is None:
            return self.samples
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 10))
        self._thread = None
        return self.samples


# ---------------------------------------------------------------------------
# Report shapes
# ---------------------------------------------------------------------------

@dataclass
class StageTiming:
    name: str
    duration_s: float

    def to_dict(self) -> dict:
        return {"name": self.name, "duration_s": round(self.duration_s, 6)}


@dataclass
class ResourceReport:
    stages: list[StageTiming] = field(default_factory=list)
    total_s: float = 0.0
    peak_rss_bytes: int | None = None
    rss_samples: list[tuple[float, int]] = field(default_factory=list)
    sample_interval_s: float = 0.0
    platform: str = sys.platform

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def to_dict(self) -> dict:
        """JSON-safe shape consumed by diagnostics.json."""

        return {
            "stages": [s.to_dict() for s in self.stages],
            "total_s": round(self.total_s, 6),
            "peak_rss_bytes": self.peak_rss_bytes,
            "rss_samples": [[round(t, 4), int(b)] for t, b in self.rss_samples],
            "sample_interval_s": self.sample_interval_s,
            "platform": self.platform,
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ResourceTracker:
    """Times named stages and samples RSS; report() finalises.

    with ResourceTracker() as rt:
        with rt.stage("ingest"):
            ...
        with rt.stage("pose"):
            ...
        rep = rt.report()
    """

    def __init__(self, *, sample_interval_s: float = 0.05):
        self.sample_interval_s = sample_interval_s
        self._stages: list[StageTiming] = []
        self._sampler: RssSampler | None = None
        self._start_wall = time.perf_counter()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "ResourceTracker":
        if self._sampler is None:
            self._sampler = RssSampler(self.sample_interval_s).start()
        return self

    def stop(self) -> None:
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler = None

    def __enter__(self) -> "ResourceTracker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # -- measurement --------------------------------------------------------

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Record wall duration of the wrapped block (errors still recorded)."""

        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._stages.append(StageTiming(name=name, duration_s=time.perf_counter() - t0))

    # -- results ------------------------------------------------------------

    def report(self) -> ResourceReport:
        sampler, self._sampler = self._sampler, None
        if sampler is not None:
            sampler.stop()
        samples = list(sampler.samples) if sampler else []
        sampled_peak = max((b for _, b in samples), default=0)
        return ResourceReport(
            stages=list(self._stages),
            total_s=time.perf_counter() - self._start_wall,
            peak_rss_bytes=max(peak_rss_bytes(), sampled_peak) or None,
            rss_samples=samples,
            sample_interval_s=self.sample_interval_s,
        )
