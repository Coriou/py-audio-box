"""Strict, prediction-free operator truth for visible foot contact.

The operator format deliberately remains ``schema_version: 2`` so existing
``auto_sfx.gt.load_ground_truth`` consumers can read its canonical contact
windows.  This module validates the richer operator-only contract: genuine
source PTS mapped to displayed frames, anatomical and observability metadata,
independent-pass provenance, and an optional audible onset which never
replaces the visible physical-contact window.

Unknown fields are rejected.  This keeps detector output, scores, overlays,
and other prediction-derived data out of the marking contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
PTS_MAP_SCHEMA_VERSION = 1
ANNOTATION_KIND = "auto-sfx-foot-contact-operator-gold"
PTS_MAP_KIND = "auto-sfx-source-pts-map"
TOOL = "foot_contact_annotator"
TOOL_VERSION = "1.0.0"

FOOT_VOCABULARY = ("left", "right", "uncertain")
VISIBILITY_VOCABULARY = ("clear", "partial", "occluded")
VIEWPOINT_VOCABULARY = (
    "sagittal",
    "diagonal",
    "frontal",
    "rear",
    "uncertain",
)
OBSERVABILITY_VOCABULARY = ("observable", "partial", "unobservable")
CROPPING_VOCABULARY = ("none", "partial", "complete", "uncertain")

_TOP_LEVEL_KEYS = (
    "schema_version",
    "kind",
    "clip",
    "range_s",
    "fps_display",
    "annotator",
    "pass_id",
    "tool",
    "tool_version",
    "session_id",
    "source",
    "contacts",
)
_SOURCE_KEYS = ("file", "sha256", "pts_map_sha256", "frame_count")
_CONTACT_KEYS = (
    "contact_id",
    "foot",
    "earliest_pts_us",
    "preferred_pts_us",
    "latest_pts_us",
    "display_frames",
    "visibility",
    "viewpoint",
    "actor_id",
    "observability",
    "cropping",
    "audible_onset",
    "notes",
)
_DISPLAY_FRAME_KEYS = ("earliest", "preferred", "latest")
_AUDIBLE_ONSET_KEYS = ("source_pts_us", "displayed_frame")
_PTS_MAP_KEYS = (
    "schema_version",
    "kind",
    "clip",
    "source_file",
    "source_sha256",
    "timestamp_source",
    "source_pts_us",
)


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 hex digest without mutating *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(where: str, message: str) -> None:
    raise ValueError(f"{where}: {message}")


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "must be an object")
    return value


def _require_exact_keys(
    value: dict[str, Any], expected: tuple[str, ...], where: str
) -> None:
    missing = [key for key in expected if key not in value]
    unknown = [key for key in value if key not in expected]
    if missing:
        _fail(where, f"missing required fields: {', '.join(missing)}")
    if unknown:
        _fail(where, f"unknown fields are forbidden: {', '.join(unknown)}")


def _require_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(where, "must be a non-empty string")
    return value


def _require_int(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(where, "must be an integer")
    return value


def _require_number(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(where, "must be a number")
    return float(value)


def _require_enum(value: Any, vocabulary: tuple[str, ...], where: str) -> str:
    if value not in vocabulary:
        _fail(where, f"must be one of {list(vocabulary)}, got {value!r}")
    return value


def validate_pts_map(document: Any, where: str = "PTS map") -> dict[str, Any]:
    """Validate and canonicalize one per-frame source timestamp map.

    ``source_pts_us`` must be strictly increasing.  It is never synthesized
    from nominal FPS; the session builder records ffprobe's
    ``best_effort_timestamp_time`` for every displayed frame.
    """

    raw = _require_object(document, where)
    _require_exact_keys(raw, _PTS_MAP_KEYS, where)
    if raw["schema_version"] != PTS_MAP_SCHEMA_VERSION or isinstance(
        raw["schema_version"], bool
    ):
        _fail(where, f"schema_version must be {PTS_MAP_SCHEMA_VERSION}")
    if raw["kind"] != PTS_MAP_KIND:
        _fail(where, f"kind must be {PTS_MAP_KIND!r}")
    clip = _require_string(raw["clip"], f"{where}.clip")
    source_file = _require_string(raw["source_file"], f"{where}.source_file")
    source_sha256 = _require_string(raw["source_sha256"], f"{where}.source_sha256")
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        _fail(f"{where}.source_sha256", "must be a lowercase SHA-256 hex digest")
    if raw["timestamp_source"] != "ffprobe.best_effort_timestamp_time":
        _fail(
            f"{where}.timestamp_source",
            "must be 'ffprobe.best_effort_timestamp_time'",
        )
    values = raw["source_pts_us"]
    if not isinstance(values, list) or not values:
        _fail(f"{where}.source_pts_us", "must be a non-empty array")
    pts = [_require_int(value, f"{where}.source_pts_us[{i}]") for i, value in enumerate(values)]
    for index, (previous, current) in enumerate(zip(pts, pts[1:]), start=1):
        if current <= previous:
            _fail(
                f"{where}.source_pts_us[{index}]",
                f"must be strictly greater than previous PTS {previous}",
            )
    return {
        "schema_version": PTS_MAP_SCHEMA_VERSION,
        "kind": PTS_MAP_KIND,
        "clip": clip,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "timestamp_source": "ffprobe.best_effort_timestamp_time",
        "source_pts_us": pts,
    }


def load_pts_map(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"{path}: cannot read PTS map: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid PTS-map JSON: {exc}") from None
    return validate_pts_map(document, str(path))


def pts_map_json(document: Any) -> str:
    """Return the one canonical byte representation used for map hashing."""

    return json.dumps(validate_pts_map(document), indent=2, ensure_ascii=False) + "\n"


def pts_map_sha256(document: Any) -> str:
    return hashlib.sha256(pts_map_json(document).encode("utf-8")).hexdigest()


def media_time_s_for_frame(pts_map: dict[str, Any], displayed_frame: int) -> float:
    """Map a displayed frame to browser media time without relabelling PTS.

    Browsers expose a zero-based media timeline even when a stream's first
    source PTS is nonzero.  Annotation exports retain the absolute source PTS;
    only seeking subtracts the first PTS.
    """

    canonical = validate_pts_map(pts_map)
    frame = _require_int(displayed_frame, "displayed_frame")
    values = canonical["source_pts_us"]
    if frame < 0 or frame >= len(values):
        _fail("displayed_frame", f"{frame} is outside PTS map with {len(values)} frames")
    return (values[frame] - values[0]) / 1_000_000


def nearest_display_frame(pts_map: dict[str, Any], media_time_s: float) -> int:
    """Return the nearest mapped frame for a browser's zero-based media time."""

    canonical = validate_pts_map(pts_map)
    media_time = _require_number(media_time_s, "media_time_s")
    values = canonical["source_pts_us"]
    target = values[0] + round(media_time * 1_000_000)
    low, high = 0, len(values) - 1
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    if low == 0:
        return 0
    before = low - 1
    return before if target - values[before] <= values[low] - target else low


def _validate_frame_mapping(
    *,
    frame: Any,
    pts: Any,
    pts_values: list[int] | None,
    where: str,
) -> tuple[int, int]:
    displayed_frame = _require_int(frame, f"{where}.displayed_frame")
    source_pts_us = _require_int(pts, f"{where}.source_pts_us")
    if displayed_frame < 0:
        _fail(f"{where}.displayed_frame", "must be non-negative")
    if pts_values is not None:
        if displayed_frame >= len(pts_values):
            _fail(
                f"{where}.displayed_frame",
                f"{displayed_frame} is outside PTS map with {len(pts_values)} frames",
            )
        expected = pts_values[displayed_frame]
        if source_pts_us != expected:
            _fail(
                f"{where}.source_pts_us",
                f"{source_pts_us} does not match displayed frame {displayed_frame} "
                f"source PTS {expected}",
            )
    return displayed_frame, source_pts_us


def validate_annotation(
    document: Any,
    pts_map: dict[str, Any] | None = None,
    where: str = "annotation",
) -> dict[str, Any]:
    """Validate and canonicalize a foot-contact operator export.

    Pass the clip's PTS map for the strongest check.  Every displayed frame is
    then verified against its genuine source timestamp, including nonuniform
    VFR intervals.  The returned object has a stable key order suitable for
    deterministic serialization.
    """

    raw = _require_object(document, where)
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, where)
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(raw["schema_version"], bool):
        _fail(where, f"schema_version must be {SCHEMA_VERSION}")
    if raw["kind"] != ANNOTATION_KIND:
        _fail(where, f"kind must be {ANNOTATION_KIND!r}")

    clip = _require_string(raw["clip"], f"{where}.clip")
    annotator = _require_string(raw["annotator"], f"{where}.annotator")
    pass_id = _require_string(raw["pass_id"], f"{where}.pass_id")
    session_id = _require_string(raw["session_id"], f"{where}.session_id")
    if raw["tool"] != TOOL:
        _fail(f"{where}.tool", f"must be {TOOL!r}")
    if raw["tool_version"] != TOOL_VERSION:
        _fail(f"{where}.tool_version", f"must be {TOOL_VERSION!r}")

    range_s = _require_object(raw["range_s"], f"{where}.range_s")
    _require_exact_keys(range_s, ("start", "end"), f"{where}.range_s")
    range_start = _require_number(range_s["start"], f"{where}.range_s.start")
    range_end = _require_number(range_s["end"], f"{where}.range_s.end")
    if range_end <= range_start:
        _fail(f"{where}.range_s", "must satisfy start < end")
    fps_display = _require_number(raw["fps_display"], f"{where}.fps_display")
    if fps_display <= 0:
        _fail(f"{where}.fps_display", "must be positive")

    source = _require_object(raw["source"], f"{where}.source")
    _require_exact_keys(source, _SOURCE_KEYS, f"{where}.source")
    source_file = _require_string(source["file"], f"{where}.source.file")
    source_sha256 = _require_string(source["sha256"], f"{where}.source.sha256")
    source_pts_map_sha256 = _require_string(
        source["pts_map_sha256"], f"{where}.source.pts_map_sha256"
    )
    for name, digest in (
        ("sha256", source_sha256),
        ("pts_map_sha256", source_pts_map_sha256),
    ):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            _fail(f"{where}.source.{name}", "must be a lowercase SHA-256 hex digest")
    frame_count = _require_int(source["frame_count"], f"{where}.source.frame_count")
    if frame_count <= 0:
        _fail(f"{where}.source.frame_count", "must be positive")

    validated_map = validate_pts_map(pts_map, "PTS map") if pts_map is not None else None
    pts_values = validated_map["source_pts_us"] if validated_map is not None else None
    if validated_map is not None:
        if validated_map["clip"] != clip:
            _fail(where, "clip does not match PTS map")
        if validated_map["source_file"] != source_file:
            _fail(where, "source.file does not match PTS map")
        if validated_map["source_sha256"] != source_sha256:
            _fail(where, "source.sha256 does not match PTS map")
        if frame_count != len(pts_values):
            _fail(where, "source.frame_count does not match PTS map")
        actual_pts_digest = pts_map_sha256(validated_map)
        if source_pts_map_sha256 != actual_pts_digest:
            _fail(
                where,
                "source.pts_map_sha256 does not match the canonical PTS-map payload",
            )

    contacts_raw = raw["contacts"]
    if not isinstance(contacts_raw, list):
        _fail(f"{where}.contacts", "must be an array")
    contacts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_preferred: int | None = None
    for index, item in enumerate(contacts_raw):
        contact_where = f"{where}.contacts[{index}]"
        contact = _require_object(item, contact_where)
        _require_exact_keys(contact, _CONTACT_KEYS, contact_where)
        contact_id = _require_string(contact["contact_id"], f"{contact_where}.contact_id")
        if contact_id in seen_ids:
            _fail(f"{contact_where}.contact_id", f"duplicate {contact_id!r}")
        seen_ids.add(contact_id)
        foot = _require_enum(contact["foot"], FOOT_VOCABULARY, f"{contact_where}.foot")

        display = _require_object(contact["display_frames"], f"{contact_where}.display_frames")
        _require_exact_keys(display, _DISPLAY_FRAME_KEYS, f"{contact_where}.display_frames")
        mapped: dict[str, int] = {}
        mapped_pts: dict[str, int] = {}
        for point in _DISPLAY_FRAME_KEYS:
            frame, source_pts = _validate_frame_mapping(
                frame=display[point],
                pts=contact[f"{point}_pts_us"],
                pts_values=pts_values,
                where=f"{contact_where}.{point}",
            )
            mapped[point] = frame
            mapped_pts[point] = source_pts
        if not (
            mapped["earliest"] <= mapped["preferred"] <= mapped["latest"]
            and mapped_pts["earliest"] <= mapped_pts["preferred"] <= mapped_pts["latest"]
        ):
            _fail(
                contact_where,
                "physical-contact window must satisfy earliest <= preferred <= latest",
            )
        if previous_preferred is not None and mapped_pts["preferred"] < previous_preferred:
            _fail(contact_where, "contacts must be sorted by preferred source PTS")
        previous_preferred = mapped_pts["preferred"]

        audible_raw = contact["audible_onset"]
        audible: dict[str, int] | None
        if audible_raw is None:
            audible = None
        else:
            audible_object = _require_object(audible_raw, f"{contact_where}.audible_onset")
            _require_exact_keys(
                audible_object, _AUDIBLE_ONSET_KEYS, f"{contact_where}.audible_onset"
            )
            audible_frame, audible_pts = _validate_frame_mapping(
                frame=audible_object["displayed_frame"],
                pts=audible_object["source_pts_us"],
                pts_values=pts_values,
                where=f"{contact_where}.audible_onset",
            )
            audible = {"source_pts_us": audible_pts, "displayed_frame": audible_frame}

        notes = contact["notes"]
        if not isinstance(notes, str):
            _fail(f"{contact_where}.notes", "must be a string")
        visibility = _require_enum(
            contact["visibility"], VISIBILITY_VOCABULARY, f"{contact_where}.visibility"
        )
        observability = _require_enum(
            contact["observability"],
            OBSERVABILITY_VOCABULARY,
            f"{contact_where}.observability",
        )
        cropping = _require_enum(
            contact["cropping"], CROPPING_VOCABULARY, f"{contact_where}.cropping"
        )
        if visibility == "clear" and observability != "observable":
            _fail(contact_where, "clear visibility requires observable contact")
        if cropping == "complete" and (
            visibility != "occluded" or observability != "unobservable"
        ):
            _fail(
                contact_where,
                "complete cropping requires occluded visibility and unobservable contact",
            )

        contacts.append(
            {
                "contact_id": contact_id,
                "foot": foot,
                "earliest_pts_us": mapped_pts["earliest"],
                "preferred_pts_us": mapped_pts["preferred"],
                "latest_pts_us": mapped_pts["latest"],
                "display_frames": mapped,
                "visibility": visibility,
                "viewpoint": _require_enum(
                    contact["viewpoint"], VIEWPOINT_VOCABULARY, f"{contact_where}.viewpoint"
                ),
                "actor_id": _require_string(contact["actor_id"], f"{contact_where}.actor_id"),
                "observability": observability,
                "cropping": cropping,
                "audible_onset": audible,
                "notes": notes,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ANNOTATION_KIND,
        "clip": clip,
        "range_s": {"start": range_start, "end": range_end},
        "fps_display": fps_display,
        "annotator": annotator,
        "pass_id": pass_id,
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "session_id": session_id,
        "source": {
            "file": source_file,
            "sha256": source_sha256,
            "pts_map_sha256": source_pts_map_sha256,
            "frame_count": frame_count,
        },
        "contacts": contacts,
    }


def annotation_json(document: Any, pts_map: dict[str, Any] | None = None) -> str:
    """Return deterministic, UTF-8-safe JSON with one trailing newline."""

    canonical = validate_annotation(document, pts_map=pts_map)
    return json.dumps(canonical, indent=2, ensure_ascii=False) + "\n"


def load_annotation(
    path: str | Path, pts_map: dict[str, Any] | None = None
) -> dict[str, Any]:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"{path}: cannot read annotation: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid annotation JSON: {exc}") from None
    return validate_annotation(document, pts_map=pts_map, where=str(path))


def write_annotation(
    path: str | Path,
    document: Any,
    pts_map: dict[str, Any] | None = None,
) -> None:
    """Validate and atomically write canonical annotation JSON."""

    path = Path(path)
    payload = annotation_json(document, pts_map=pts_map)
    path.parent.mkdir(parents=True, exist_ok=True)
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
