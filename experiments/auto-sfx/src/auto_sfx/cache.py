"""Content-hash cache for pose traces (auto-sfx POC).

Cache key material -- per the POC plan, a pose trace may be reused only when
ALL of the following are identical:

* video content hash          (sha256 of the file bytes)
* video stream selector       (POC constant: first video stream ``"v:0"``)
* pose backend/model identity (backend name + model file sha256)
* pose configuration          (running mode)
* preprocessing version       (:data:`POSE_CACHE_PREPROCESS_VERSION`)

Public surface::

    key = compute_pose_cache_key(video_path, backend, mode, model_sha256)
    cache = PoseCache()                # $AUTO_SFX_CACHE_DIR or /cache/auto-sfx/pose
    trace = cache.load_trace(key)      # PoseTrace | None
    if trace is None:
        trace = run_pose(...)
        cache.save_trace(trace, key)

On-disk format: one JSONL file per key. Line 1 is a header dict; every
following line is one :meth:`PoseTrace.to_rows` row. Writes are atomic
(tmp + rename). A corrupt/stale entry is treated as a miss, never an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .types import (
    POSE_CACHE_PREPROCESS_VERSION,
    PoseTrace,
    VideoMeta,
)

TRACE_FORMAT = "auto-sfx-pose-trace"
TRACE_FORMAT_VERSION = 1

# POC assumes a single dominant video stream; the slot stays in the key so a
# future multi-stream world only changes this constant plus callers.
DEFAULT_STREAM_TAG = "v:0"

_DOWNLOAD_CHUNK = 1 << 20


def hash_file(path: str | os.PathLike) -> str:
    """Streaming sha256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_key_material(
    *,
    video_path: str | os.PathLike,
    backend: str,
    mode: str,
    model_sha256: str | None,
    stream: str = DEFAULT_STREAM_TAG,
) -> dict[str, Any]:
    return {
        "video_sha256": hash_file(video_path),
        "stream": stream,
        "backend": backend,
        "mode": mode,
        "model_sha256": model_sha256,
        # Read at call time so a POSE_CACHE_PREPROCESS_VERSION bump takes
        # effect even for callers that imported this module earlier.
        "preprocess_version": POSE_CACHE_PREPROCESS_VERSION,
        "key_schema": TRACE_FORMAT_VERSION,
    }


def compute_pose_cache_key(
    video_path: str | os.PathLike,
    backend: str,
    mode: str,
    model_sha256: str | None,
) -> str:
    """Deterministic sha256 hex digest over the full key material.

    ``model_sha256`` comes from :func:`auto_sfx.pose.mediapipe_backend.ensure_models`.
    """
    encoded = json.dumps(
        _canonical_key_material(
            video_path=video_path, backend=backend, mode=mode, model_sha256=model_sha256
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PoseCache:
    """Filesystem cache of :class:`PoseTrace` objects keyed by content hash."""

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        self.root = Path(
            root
            if root is not None
            else os.environ.get("AUTO_SFX_CACHE_DIR", "/cache/auto-sfx/pose")
        )

    # -- layout -------------------------------------------------------------

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.jsonl"

    # -- operations ----------------------------------------------------------

    def load_trace(self, key: str) -> PoseTrace | None:
        """Return the cached trace for ``key``, or None on miss/corruption."""
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            if not lines:
                raise ValueError("empty cache entry")
            header = json.loads(lines[0])
            rows = [json.loads(line) for line in lines[1:]]
            if header.get("format") != TRACE_FORMAT:
                raise ValueError(f"unexpected format {header.get('format')!r}")
            if header.get("format_version") != TRACE_FORMAT_VERSION:
                raise ValueError("incompatible format version")
            if header.get("key") != key:
                raise ValueError("key mismatch (stale or moved entry)")
            meta = VideoMeta(**header["meta"])
            return PoseTrace.from_rows(header["backend"], header["mode"], meta, rows)
        except Exception as exc:  # cache must never break the pipeline
            print(f"[pose-cache] ignoring unusable entry {path.name}: {exc}", file=sys.stderr)
            return None

    def save_trace(self, trace: PoseTrace, key: str) -> Path:
        """Atomically write ``trace`` under ``key``; returns the entry path."""
        dest = self.path_for(key)
        self.root.mkdir(parents=True, exist_ok=True)
        header = {
            "format": TRACE_FORMAT,
            "format_version": TRACE_FORMAT_VERSION,
            "key": key,
            "backend": trace.backend,
            "mode": trace.mode,
            "meta": trace.meta.to_dict(),
        }
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(header, sort_keys=True) + "\n")
                for row in trace.to_rows():
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()
        return dest


__all__ = [
    "DEFAULT_STREAM_TAG",
    "PoseCache",
    "TRACE_FORMAT",
    "TRACE_FORMAT_VERSION",
    "compute_pose_cache_key",
    "hash_file",
]
