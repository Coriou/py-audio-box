"""Build prediction-free foot-contact annotation sessions from source clips."""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from .foot_contact_truth import (
    PTS_MAP_KIND,
    PTS_MAP_SCHEMA_VERSION,
    TOOL,
    TOOL_VERSION,
    sha256_file,
    validate_pts_map,
)

SESSION_KIND = "auto-sfx-foot-contact-session"
SESSION_SCHEMA_VERSION = 1
ANNOTATOR_PATH = "experiments/auto-sfx/foot_contact_annotator.html"

_ENTRY_KEYS = {"path", "file", "clip_id", "actor_ids"}
_CLIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _atomic_json(path: Path, document: Any) -> None:
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _nominal_fps(stream: dict[str, Any]) -> float:
    for field in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(field)
        try:
            value = float(Fraction(raw))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    raise ValueError("ffprobe returned no positive nominal video frame rate")


def _timestamp_us(raw: Any, frame_index: int) -> int:
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            f"ffprobe frame {frame_index} has no best_effort_timestamp_time; "
            "refusing to synthesize source PTS from FPS"
        )
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise ValueError(
            f"ffprobe frame {frame_index} has invalid source timestamp {raw!r}"
        ) from None
    if not value.is_finite():
        raise ValueError(
            f"ffprobe frame {frame_index} has non-finite source timestamp {raw!r}"
        )
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_HALF_UP))


def probe_source_timestamps(
    path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[float, list[int]]:
    """Return nominal FPS and genuine per-frame source PTS from ffprobe.

    The timestamps are probed in presentation order using
    ``best_effort_timestamp_time``.  Missing, duplicated, or non-monotonic
    timestamps fail closed; this function never falls back to ``frame / fps``.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = runner(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise ValueError("ffprobe is required to build a foot-contact session") from None
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise ValueError(f"ffprobe failed for {path}: {detail or exc}") from None
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid JSON for {path}: {exc}") from None

    streams = document.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"ffprobe found no video stream in {path}")
    fps = _nominal_fps(streams[0])
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError(f"ffprobe returned fewer than two video frames for {path}")
    pts = [
        _timestamp_us(frame.get("best_effort_timestamp_time"), index)
        if isinstance(frame, dict)
        else _timestamp_us(None, index)
        for index, frame in enumerate(frames)
    ]
    for index, (previous, current) in enumerate(zip(pts, pts[1:]), start=1):
        if current <= previous:
            raise ValueError(
                f"ffprobe source PTS is not strictly increasing at displayed frame "
                f"{index}: {current} <= {previous}"
            )
    return fps, pts


def _manifest_entries(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        entries = document
    elif isinstance(document, dict) and set(document) == {"clips"}:
        entries = document["clips"]
    else:
        raise ValueError("clip manifest must be a list or an object containing only 'clips'")
    if not isinstance(entries, list) or not entries:
        raise ValueError("clip manifest must contain at least one clip")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"clip manifest entry {index} must be an object")
        unknown = sorted(set(item) - _ENTRY_KEYS)
        if unknown:
            raise ValueError(
                f"clip manifest entry {index} has forbidden/unknown fields: "
                f"{', '.join(unknown)}"
            )
        refs = [item.get("path"), item.get("file")]
        refs = [ref for ref in refs if ref is not None]
        if len(refs) != 1 or not isinstance(refs[0], str) or not refs[0]:
            raise ValueError(f"clip manifest entry {index} needs exactly one string path/file")
        actor_ids = item.get("actor_ids", ["actor-1"])
        if (
            not isinstance(actor_ids, list)
            or not actor_ids
            or any(not isinstance(actor, str) or not actor.strip() for actor in actor_ids)
            or len(set(actor_ids)) != len(actor_ids)
        ):
            raise ValueError(
                f"clip manifest entry {index} actor_ids must be unique non-empty strings"
            )
        normalized.append(
            {
                "path": refs[0],
                "clip_id": item.get("clip_id"),
                "actor_ids": list(actor_ids),
            }
        )
    return normalized


def load_clip_manifest(path: str | Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"cannot read clip manifest {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid clip manifest JSON {path}: {exc}") from None
    return _manifest_entries(document)


def session_http_url(
    out_dir: str | Path,
    repo_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> str:
    """Return an HTTP URL whose session query is always repo-root-relative."""

    root = Path(repo_root).resolve()
    try:
        relative = Path(out_dir).resolve().relative_to(root)
    except ValueError:
        raise ValueError("out_dir must be inside the repo root for HTTP serving") from None
    return (
        f"http://{host}:{port}/{ANNOTATOR_PATH}"
        f"?session={relative.as_posix()}/session.json"
    )


def build_session(
    *,
    entries: list[dict[str, Any]],
    out_dir: str | Path,
    repo_root: str | Path,
    session_id: str,
    pass_id: str,
    order_seed: str,
    probe: Callable[[Path], tuple[float, list[int]]] = probe_source_timestamps,
) -> dict[str, Any]:
    """Build one deterministic, pass-isolated annotation session."""

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(pass_id, str) or not pass_id.strip():
        raise ValueError("pass_id must be a non-empty string")
    if not isinstance(order_seed, str) or not order_seed.strip():
        raise ValueError("order_seed must be a non-empty string")
    entries = _manifest_entries(entries)
    root = Path(repo_root).resolve()
    destination = Path(out_dir).resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        raise ValueError("out_dir must be inside the repo root for HTTP serving") from None
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "out_dir must be empty; use a separate directory for every annotation pass"
        )
    destination.mkdir(parents=True, exist_ok=True)
    pts_dir = destination / "pts"
    annotations_dir = destination / "annotations"
    pts_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)

    shuffled = list(entries)
    random.Random(order_seed).shuffle(shuffled)
    clips: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(shuffled):
        source = (root / entry["path"]).resolve()
        try:
            source_relative = source.relative_to(root)
        except ValueError:
            raise ValueError(f"clip path escapes repo root: {entry['path']}") from None
        if not source.is_file():
            raise ValueError(f"clip does not exist: {source_relative}")
        clip_id = entry.get("clip_id") or source.stem
        if not isinstance(clip_id, str) or not _CLIP_ID.fullmatch(clip_id):
            raise ValueError(
                f"clip {index} clip_id must match {_CLIP_ID.pattern!r}, got {clip_id!r}"
            )
        if clip_id in seen_ids:
            raise ValueError(f"duplicate clip_id {clip_id!r}")
        seen_ids.add(clip_id)

        fps, source_pts_us = probe(source)
        source_digest = sha256_file(source)
        pts_document = validate_pts_map(
            {
                "schema_version": PTS_MAP_SCHEMA_VERSION,
                "kind": PTS_MAP_KIND,
                "clip": clip_id,
                "source_file": source_relative.as_posix(),
                "source_sha256": source_digest,
                "timestamp_source": "ffprobe.best_effort_timestamp_time",
                "source_pts_us": source_pts_us,
            }
        )
        pts_path = pts_dir / f"{clip_id}.source-pts.json"
        _atomic_json(pts_path, pts_document)
        pts_relative = pts_path.relative_to(root)
        clips.append(
            {
                "clip_id": clip_id,
                "file": source_relative.as_posix(),
                "nominal_fps": fps,
                "range_s": {
                    "start": source_pts_us[0] / 1_000_000,
                    "end": source_pts_us[-1] / 1_000_000,
                },
                "actor_ids": entry["actor_ids"],
                "pts_map_file": pts_relative.as_posix(),
                "source": {
                    "file": source_relative.as_posix(),
                    "sha256": source_digest,
                    "pts_map_sha256": sha256_file(pts_path),
                    "frame_count": len(source_pts_us),
                },
            }
        )

    session = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "kind": SESSION_KIND,
        "session_id": session_id,
        "pass_id": pass_id,
        "order_seed": order_seed,
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "annotator_path": ANNOTATOR_PATH,
        "clips": clips,
    }
    _atomic_json(destination / "session.json", session)
    return session
