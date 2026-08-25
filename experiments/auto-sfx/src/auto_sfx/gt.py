"""Ground-truth contact annotations, schema v2.

Canonical format for Phase-2 ground truth (docs/auto-sfx-phase2-truth-repair.md
§5.1): one JSON object per clip with PTS-microsecond integer windows per
contact. PTS integers are canonical; seconds are derived. The reference MUST
be independent of the detector — this module only validates structure, it
never consults detector output.

Legacy v1 arrays (``[{"time_s": .., "foot": ..}]``) remain readable by
``metrics.load_annotations`` for back-compat, but they carry no windows and
are NOT accepted here: v2 is canonical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2

FOOT_VOCABULARY = ("left", "right", "uncertain")
VISIBILITY_VOCABULARY = ("clear", "partial", "occluded")
VIEWPOINT_VOCABULARY = ("sagittal", "diagonal", "frontal", "rear", "uncertain")


@dataclass(frozen=True)
class GTContact:
    """One annotated initial contact with a defensible frame window."""

    foot: str
    earliest_pts_us: int
    preferred_pts_us: int
    latest_pts_us: int
    visibility: str = "clear"
    viewpoint: str = "uncertain"
    notes: str = ""

    @property
    def earliest_s(self) -> float:
        return self.earliest_pts_us / 1_000_000.0

    @property
    def preferred_s(self) -> float:
        return self.preferred_pts_us / 1_000_000.0

    @property
    def latest_s(self) -> float:
        return self.latest_pts_us / 1_000_000.0


@dataclass
class GroundTruth:
    """Validated contents of one ``<clip-stem>.gt.json`` file."""

    clip: str
    range_start_s: float
    range_end_s: float
    annotator: str
    fps_display: float
    contacts: list[GTContact] = field(default_factory=list)
    source_path: str = ""
    #: Non-fatal observations (out-of-range marks, duplicate timestamps).
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for c in self.contacts:
            counts[c.foot] = counts.get(c.foot, 0) + 1
        return {
            "clip": self.clip,
            "range_s": [self.range_start_s, self.range_end_s],
            "annotator": self.annotator,
            "fps_display": self.fps_display,
            "n_contacts": len(self.contacts),
            "by_foot": counts,
            "warnings": list(self.warnings),
        }


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fail(path, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _parse_contact(path, raw, index: int) -> GTContact:
    where = f"contact {index}"
    if not isinstance(raw, dict):
        _fail(path, f"{where} is not an object")
    foot = raw.get("foot")
    if foot not in FOOT_VOCABULARY:
        _fail(
            path,
            f"{where} foot must be one of {list(FOOT_VOCABULARY)}, got {foot!r}",
        )
    pts = {}
    for name in ("earliest_pts_us", "preferred_pts_us", "latest_pts_us"):
        value = raw.get(name)
        if not _is_int(value):
            _fail(path, f"{where} {name} must be an integer (microseconds)")
        pts[name] = value
    e, p, l = (
        pts["earliest_pts_us"],
        pts["preferred_pts_us"],
        pts["latest_pts_us"],
    )
    if not e <= p <= l:
        _fail(path, f"{where} window sanity violated: {e} <= {p} <= {l} required")

    visibility = raw.get("visibility", "clear")
    if visibility not in VISIBILITY_VOCABULARY:
        _fail(
            path,
            f"{where} visibility must be one of "
            f"{list(VISIBILITY_VOCABULARY)}, got {visibility!r}",
        )
    viewpoint = raw.get("viewpoint", "uncertain")
    if viewpoint not in VIEWPOINT_VOCABULARY:
        _fail(
            path,
            f"{where} viewpoint must be one of "
            f"{list(VIEWPOINT_VOCABULARY)}, got {viewpoint!r}",
        )
    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        _fail(path, f"{where} notes must be a string")
    return GTContact(
        foot=foot,
        earliest_pts_us=e,
        preferred_pts_us=p,
        latest_pts_us=l,
        visibility=visibility,
        viewpoint=viewpoint,
        notes=notes,
    )


def load_ground_truth(path: str | Path) -> GroundTruth:
    """Load and validate one schema-v2 ground-truth JSON file.

    Raises ``ValueError`` with a precise, operator-fixable message on any
    structural violation (schema version, vocabularies, integer PTS types,
    window sanity, sort order). Non-fatal issues (marks outside the declared
    range, duplicated preferred timestamps) land in ``GroundTruth.warnings``.
    """

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"{path}: cannot read file: {exc}") from None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from None

    if not isinstance(doc, dict):
        _fail(path, "expected a JSON object (schema_version 2); bare arrays are "
                    "the legacy v1 shape readable by metrics, not by this loader")
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION or isinstance(version, bool):
        _fail(
            path,
            f"schema_version must be {SCHEMA_VERSION}, got {version!r}",
        )

    clip = doc.get("clip")
    if not isinstance(clip, str) or not clip:
        _fail(path, "clip must be a non-empty string")
    annotator = doc.get("annotator")
    if not isinstance(annotator, str) or not annotator:
        _fail(path, "annotator must be a non-empty string")

    range_s = doc.get("range_s")
    if not isinstance(range_s, dict) or set(range_s) < {"start", "end"}:
        _fail(path, 'range_s must be an object with "start" and "end"')
    start, end = range_s.get("start"), range_s.get("end")
    if not (_is_num(start) and _is_num(end)):
        _fail(path, "range_s.start/end must be numbers (seconds)")
    start, end = float(start), float(end)
    if end <= start:
        _fail(path, f"range_s must satisfy start < end, got [{start}, {end}]")

    fps_display = doc.get("fps_display")
    if not _is_num(fps_display) or fps_display <= 0:
        _fail(path, "fps_display must be a positive number")

    contacts_raw = doc.get("contacts")
    if not isinstance(contacts_raw, list):
        _fail(path, "contacts must be an array")

    contacts = [_parse_contact(path, raw, i) for i, raw in enumerate(contacts_raw)]

    warnings: list[str] = []
    for i in range(len(contacts) - 1):
        a, b = contacts[i], contacts[i + 1]
        if b.preferred_pts_us < a.preferred_pts_us:
            _fail(
                path,
                f"contact {i + 1} preferred_pts_us ({b.preferred_pts_us}) "
                f"breaks sort order after {a.preferred_pts_us}; sort contacts "
                "by preferred_pts_us",
            )
        elif b.preferred_pts_us == a.preferred_pts_us:
            warnings.append(
                f"contacts {i}/{i + 1} share preferred_pts_us "
                f"{b.preferred_pts_us} (possible simultaneous-LR mark)"
            )
    for i, c in enumerate(contacts):
        if not (start * 1e6 - 0.5 <= c.preferred_pts_us <= end * 1e6 + 0.5):
            warnings.append(
                f"contact {i} preferred_pts_us {c.preferred_pts_us} lies "
                f"outside range_s [{start}, {end}] s"
            )
    if not contacts:
        warnings.append("no contacts annotated yet")

    return GroundTruth(
        clip=clip,
        range_start_s=start,
        range_end_s=end,
        annotator=annotator,
        fps_display=float(fps_display),
        contacts=contacts,
        source_path=str(path),
        warnings=warnings,
    )
