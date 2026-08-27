"""Freeze, compare, and adjudicate prediction-blind foot-contact passes.

The annotation UI deliberately knows nothing about detector output.  This
module keeps that boundary after marking: it validates every source/PTS hash,
hash-locks complete pass exports, aligns only operator contacts, and records
adjudication decisions in an append-only hash chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .foot_contact_session import SESSION_KIND, SESSION_SCHEMA_VERSION
from .foot_contact_truth import (
    ANNOTATION_KIND,
    TOOL,
    TOOL_VERSION,
    annotation_json,
    load_annotation,
    load_pts_map,
    sha256_file,
    validate_annotation,
)

FINALIZATION_KIND = "auto-sfx-foot-contact-pass-finalization"
COMPARISON_KIND = "auto-sfx-foot-contact-pass-comparison"
DECISION_KIND = "auto-sfx-foot-contact-adjudication-decision"
RECEIPT_KIND = "auto-sfx-foot-contact-adjudication-receipt"
SCHEMA_VERSION = 1

FINALIZATION_FILE = "pass-finalization.json"
COMPARISON_FILE = "comparison.json"
ADJUDICATION_LOG = "adjudication.jsonl"

_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_SESSION_KEYS = {
    "schema_version", "kind", "session_id", "pass_id", "order_seed", "tool",
    "tool_version", "annotator_path", "clips",
}
_SESSION_CLIP_KEYS = {
    "clip_id", "file", "nominal_fps", "range_s", "actor_ids", "pts_map_file",
    "source",
}
_SESSION_SOURCE_KEYS = {"file", "sha256", "pts_map_sha256", "frame_count"}
_CONTACT_COMPARE_FIELDS = (
    "foot", "earliest_pts_us", "preferred_pts_us", "latest_pts_us",
    "display_frames", "visibility", "viewpoint", "actor_id", "observability",
    "cropping", "audible_onset",
)


def _canonical_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(value, indent=2, ensure_ascii=False)
    else:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ValueError(f"{where}: missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{where}: unknown fields are forbidden: {', '.join(unknown)}")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: must be a non-empty string")
    return value


def _utc(value: Any, where: str) -> str:
    result = _string(value, where)
    if not _UTC_RE.fullmatch(result):
        raise ValueError(f"{where}: must be UTC as YYYY-MM-DDTHH:MM:SSZ")
    return result


def _digest(value: Any, where: str) -> str:
    result = _string(value, where)
    if not _SHA_RE.fullmatch(result):
        raise ValueError(f"{where}: must be a lowercase SHA-256 digest")
    return result


def _inside(path: str | Path, root: Path, where: str) -> Path:
    result = Path(path).resolve()
    try:
        result.relative_to(root)
    except ValueError:
        raise ValueError(f"{where}: must be inside repository root {root}") from None
    return result


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _read_json(path: Path, where: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"{where}: cannot read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: invalid JSON in {path}: {exc}") from None


def _write_new(path: Path, payload: bytes) -> None:
    """Durably create *path* without ever replacing an existing artifact."""

    if path.exists():
        raise ValueError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise ValueError(f"refusing to overwrite existing artifact: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_session(document: Any, where: str) -> dict[str, Any]:
    session = _strict_keys(document, _SESSION_KEYS, where)
    if session["schema_version"] != SESSION_SCHEMA_VERSION:
        raise ValueError(f"{where}: schema_version must be {SESSION_SCHEMA_VERSION}")
    if session["kind"] != SESSION_KIND:
        raise ValueError(f"{where}: kind must be {SESSION_KIND!r}")
    for key in ("session_id", "pass_id", "order_seed", "annotator_path"):
        _string(session[key], f"{where}.{key}")
    if session["tool"] != TOOL or session["tool_version"] != TOOL_VERSION:
        raise ValueError(f"{where}: tool identity/version does not match the annotator")
    clips = session["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"{where}.clips: must be a non-empty array")
    seen: set[str] = set()
    for index, raw in enumerate(clips):
        clip_where = f"{where}.clips[{index}]"
        clip = _strict_keys(raw, _SESSION_CLIP_KEYS, clip_where)
        clip_id = _string(clip["clip_id"], f"{clip_where}.clip_id")
        if not _ID_RE.fullmatch(clip_id) or clip_id in seen:
            raise ValueError(f"{clip_where}.clip_id: invalid or duplicate {clip_id!r}")
        seen.add(clip_id)
        for key in ("file", "pts_map_file"):
            _string(clip[key], f"{clip_where}.{key}")
        fps = clip["nominal_fps"]
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            raise ValueError(f"{clip_where}.nominal_fps: must be positive")
        actors = clip["actor_ids"]
        if (not isinstance(actors, list) or not actors or
                any(not isinstance(item, str) or not item.strip() for item in actors) or
                len(set(actors)) != len(actors)):
            raise ValueError(f"{clip_where}.actor_ids: must be unique non-empty strings")
        source = _strict_keys(clip["source"], _SESSION_SOURCE_KEYS, f"{clip_where}.source")
        if source["file"] != clip["file"]:
            raise ValueError(f"{clip_where}: file must match source.file")
        _digest(source["sha256"], f"{clip_where}.source.sha256")
        _digest(source["pts_map_sha256"], f"{clip_where}.source.pts_map_sha256")
        if (not isinstance(source["frame_count"], int) or
                isinstance(source["frame_count"], bool) or source["frame_count"] <= 0):
            raise ValueError(f"{clip_where}.source.frame_count: must be positive")
        bounds = _strict_keys(clip["range_s"], {"start", "end"}, f"{clip_where}.range_s")
        if any(not isinstance(bounds[key], (int, float)) or isinstance(bounds[key], bool)
               for key in ("start", "end")) or bounds["end"] <= bounds["start"]:
            raise ValueError(f"{clip_where}.range_s: must satisfy numeric start < end")
    return session


def _inspect_pass(
    pass_dir: str | Path, repo_root: str | Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    root = Path(repo_root).resolve()
    directory = _inside(pass_dir, root, "pass_dir")
    session_path = directory / "session.json"
    if not session_path.is_file():
        raise ValueError(f"pass has no session.json: {directory}")
    session = _validate_session(_read_json(session_path, "session"), str(session_path))

    annotation_dir = directory / "annotations"
    if not annotation_dir.is_dir():
        raise ValueError(f"pass has no annotations directory: {annotation_dir}")
    nonhidden = sorted(path for path in annotation_dir.iterdir() if not path.name.startswith("."))
    invalid = [
        path.name
        for path in nonhidden
        if not path.is_file() or not path.name.endswith(".gt.json")
    ]
    if invalid:
        raise ValueError(
            f"annotations directory contains unsupported entries: {', '.join(invalid)}"
        )

    annotations: dict[str, dict[str, Any]] = {}
    annotation_paths: dict[str, Path] = {}
    for annotation_path in nonhidden:
        raw = _read_json(annotation_path, "annotation")
        clip_id = raw.get("clip") if isinstance(raw, dict) else None
        if not isinstance(clip_id, str):
            raise ValueError(f"{annotation_path}: annotation has no string clip ID")
        if clip_id in annotations:
            raise ValueError(f"duplicate annotation export for clip {clip_id!r}")
        annotations[clip_id] = raw
        annotation_paths[clip_id] = annotation_path

    session_clips = {clip["clip_id"]: clip for clip in session["clips"]}
    missing = sorted(set(session_clips) - set(annotations))
    extra = sorted(set(annotations) - set(session_clips))
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing clips {missing}")
        if extra:
            detail.append(f"unexpected clips {extra}")
        raise ValueError("pass is incomplete: " + "; ".join(detail))

    clip_rows: list[dict[str, Any]] = []
    canonical_annotations: dict[str, dict[str, Any]] = {}
    pts_maps: dict[str, dict[str, Any]] = {}
    annotators: set[str] = set()
    for clip_id in sorted(session_clips):
        clip = session_clips[clip_id]
        pts_path = _inside(root / clip["pts_map_file"], root, "pts_map_file")
        source_path = _inside(root / clip["source"]["file"], root, "source.file")
        if not pts_path.is_file() or not source_path.is_file():
            raise ValueError(f"{clip_id}: source or PTS-map file is missing")
        if sha256_file(pts_path) != clip["source"]["pts_map_sha256"]:
            raise ValueError(f"{clip_id}: PTS-map bytes changed after session creation")
        if sha256_file(source_path) != clip["source"]["sha256"]:
            raise ValueError(f"{clip_id}: source bytes changed after session creation")
        pts_map = load_pts_map(pts_path)
        annotation_path = annotation_paths[clip_id]
        annotation = load_annotation(annotation_path, pts_map=pts_map)
        if annotation_path.read_bytes() != annotation_json(annotation, pts_map).encode("utf-8"):
            raise ValueError(f"{annotation_path}: export is valid but not canonical bytes")
        if annotation["session_id"] != session["session_id"]:
            raise ValueError(f"{annotation_path}: session_id does not match session.json")
        if annotation["pass_id"] != session["pass_id"]:
            raise ValueError(f"{annotation_path}: pass_id does not match session.json")
        if annotation["source"] != clip["source"]:
            raise ValueError(f"{annotation_path}: source identity does not match session.json")
        if annotation["range_s"] != clip["range_s"]:
            raise ValueError(f"{annotation_path}: range_s does not match session.json")
        if annotation["fps_display"] != float(clip["nominal_fps"]):
            raise ValueError(f"{annotation_path}: fps_display does not match session.json")
        unknown_actors = sorted(
            {contact["actor_id"] for contact in annotation["contacts"]} - set(clip["actor_ids"])
        )
        if unknown_actors:
            raise ValueError(f"{annotation_path}: unknown actor IDs {unknown_actors}")
        annotators.add(annotation["annotator"])
        canonical_annotations[clip_id] = annotation
        pts_maps[clip_id] = pts_map
        clip_rows.append(
            {
                "clip_id": clip_id,
                "annotation_file": _relative(annotation_path, root),
                "annotation_sha256": sha256_file(annotation_path),
                "pts_map_file": _relative(pts_path, root),
                "pts_map_sha256": sha256_file(pts_path),
                "source_file": _relative(source_path, root),
                "source_sha256": sha256_file(source_path),
                "frame_count": clip["source"]["frame_count"],
                "contact_count": len(annotation["contacts"]),
            }
        )
    if len(annotators) != 1:
        raise ValueError("one pass must use one stable annotator ID across every clip")

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "kind": FINALIZATION_KIND,
        "session_id": session["session_id"],
        "pass_id": session["pass_id"],
        "order_seed_sha256": _sha_bytes(session["order_seed"].encode("utf-8")),
        "annotator": next(iter(annotators)),
        "session_file": _relative(session_path, root),
        "session_sha256": sha256_file(session_path),
        "clip_count": len(clip_rows),
        "contact_count": sum(row["contact_count"] for row in clip_rows),
        "clips": clip_rows,
    }
    return snapshot, canonical_annotations, pts_maps


def finalize_pass(
    pass_dir: str | Path,
    *,
    repo_root: str | Path,
    finalized_at_utc: str,
) -> dict[str, Any]:
    """Validate a complete pass and create its no-overwrite hash lock."""

    timestamp = _utc(finalized_at_utc, "finalized_at_utc")
    directory = _inside(pass_dir, Path(repo_root).resolve(), "pass_dir")
    snapshot, _, _ = _inspect_pass(directory, repo_root)
    document = dict(snapshot)
    document["finalized_at_utc"] = timestamp
    # Keep a stable, intentional field order in the human-readable artifact.
    document = {
        "schema_version": document["schema_version"],
        "kind": document["kind"],
        "finalized_at_utc": document["finalized_at_utc"],
        **{key: value for key, value in document.items()
           if key not in {"schema_version", "kind", "finalized_at_utc"}},
    }
    _write_new(directory / FINALIZATION_FILE, _canonical_bytes(document))
    return document


def verify_finalized_pass(
    pass_dir: str | Path, *, repo_root: str | Path
) -> dict[str, Any]:
    """Recompute every pass commitment and fail on any post-finalization drift."""

    root = Path(repo_root).resolve()
    directory = _inside(pass_dir, root, "pass_dir")
    path = directory / FINALIZATION_FILE
    document = _strict_keys(
        _read_json(path, "pass finalization"),
        {
            "schema_version", "kind", "finalized_at_utc", "session_id", "pass_id",
            "order_seed_sha256", "annotator", "session_file", "session_sha256",
            "clip_count", "contact_count", "clips",
        },
        str(path),
    )
    if document["schema_version"] != SCHEMA_VERSION or document["kind"] != FINALIZATION_KIND:
        raise ValueError(f"{path}: unsupported finalization schema/kind")
    _utc(document["finalized_at_utc"], f"{path}.finalized_at_utc")
    snapshot, _, _ = _inspect_pass(directory, root)
    expected = dict(document)
    expected.pop("finalized_at_utc")
    if expected != snapshot:
        raise ValueError(f"{path}: finalized pass no longer matches its committed bytes")
    if path.read_bytes() != _canonical_bytes(document):
        raise ValueError(f"{path}: finalization is not canonical JSON")
    return document


def _pair_contacts(
    pass_a: list[dict[str, Any]],
    pass_b: list[dict[str, Any]],
    max_distance: int,
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """Deterministic actor-scoped monotonic proximity alignment."""

    rows: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    actors = sorted(
        {contact["actor_id"] for contact in pass_a}
        | {contact["actor_id"] for contact in pass_b}
    )
    for actor in actors:
        left = [contact for contact in pass_a if contact["actor_id"] == actor]
        right = [contact for contact in pass_b if contact["actor_id"] == actor]
        i = j = 0
        while i < len(left) and j < len(right):
            a_frame = left[i]["display_frames"]["preferred"]
            b_frame = right[j]["display_frames"]["preferred"]
            if abs(a_frame - b_frame) <= max_distance:
                rows.append((left[i], right[j]))
                i += 1
                j += 1
            elif a_frame < b_frame:
                rows.append((left[i], None))
                i += 1
            else:
                rows.append((None, right[j]))
                j += 1
        rows.extend((contact, None) for contact in left[i:])
        rows.extend((None, contact) for contact in right[j:])
    return sorted(
        rows,
        key=lambda pair: (
            min(
                contact["display_frames"]["preferred"]
                for contact in pair if contact is not None
            ),
            next(contact["actor_id"] for contact in pair if contact is not None),
            0 if pair[0] is not None else 1,
        ),
    )


def _differences(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    return [field for field in _CONTACT_COMPARE_FIELDS if a[field] != b[field]]


def _build_comparison_document(
    pass_a_dir: str | Path,
    pass_b_dir: str | Path,
    *,
    repo_root: str | Path,
    compared_at_utc: str,
    max_pair_distance_frames: int = 6,
) -> dict[str, Any]:
    """Recompute the canonical comparison directly from two frozen passes."""

    timestamp = _utc(compared_at_utc, "compared_at_utc")
    if (not isinstance(max_pair_distance_frames, int) or
            isinstance(max_pair_distance_frames, bool) or max_pair_distance_frames < 0):
        raise ValueError("max_pair_distance_frames must be a non-negative integer")
    root = Path(repo_root).resolve()
    a_dir = _inside(pass_a_dir, root, "pass_a_dir")
    b_dir = _inside(pass_b_dir, root, "pass_b_dir")
    if a_dir == b_dir:
        raise ValueError("pass A and pass B must be different directories")
    final_a = verify_finalized_pass(a_dir, repo_root=root)
    final_b = verify_finalized_pass(b_dir, repo_root=root)
    snap_a, annotations_a, _ = _inspect_pass(a_dir, root)
    snap_b, annotations_b, _ = _inspect_pass(b_dir, root)
    session_a = _validate_session(_read_json(a_dir / "session.json", "session A"), "session A")
    session_b = _validate_session(_read_json(b_dir / "session.json", "session B"), "session B")
    if final_a["session_id"] != final_b["session_id"]:
        raise ValueError("passes belong to different session IDs")
    if final_a["pass_id"] == final_b["pass_id"]:
        raise ValueError("passes must have distinct pass IDs")
    if final_a["clip_count"] > 1 and final_a["order_seed_sha256"] == final_b["order_seed_sha256"]:
        raise ValueError("multi-clip passes must use distinct recorded order seeds")
    order_a = [clip["clip_id"] for clip in session_a["clips"]]
    order_b = [clip["clip_id"] for clip in session_b["clips"]]
    if len(order_a) > 1 and order_a == order_b:
        raise ValueError("multi-clip passes must have different realized clip orders")
    clips_a = {row["clip_id"]: row for row in snap_a["clips"]}
    clips_b = {row["clip_id"]: row for row in snap_b["clips"]}
    if set(clips_a) != set(clips_b):
        raise ValueError("passes do not contain the same clip IDs")
    session_clip_a = {clip["clip_id"]: clip for clip in session_a["clips"]}
    session_clip_b = {clip["clip_id"]: clip for clip in session_b["clips"]}

    clip_documents: list[dict[str, Any]] = []
    totals = {
        "contacts_a": 0,
        "contacts_b": 0,
        "aligned_pairs": 0,
        "exact_agreements": 0,
        "disagreements": 0,
        "pass_a_only": 0,
        "pass_b_only": 0,
    }
    for clip_id in sorted(clips_a):
        identity_fields = (
            "source_file", "source_sha256", "pts_map_sha256", "frame_count"
        )
        if any(clips_a[clip_id][key] != clips_b[clip_id][key] for key in identity_fields):
            raise ValueError(f"{clip_id}: passes do not bind the same source/PTS identity")
        if session_clip_a[clip_id]["actor_ids"] != session_clip_b[clip_id]["actor_ids"]:
            raise ValueError(f"{clip_id}: passes declare different actor IDs")
        annotation_a = annotations_a[clip_id]
        annotation_b = annotations_b[clip_id]
        totals["contacts_a"] += len(annotation_a["contacts"])
        totals["contacts_b"] += len(annotation_b["contacts"])
        aligned = _pair_contacts(
            annotation_a["contacts"], annotation_b["contacts"], max_pair_distance_frames
        )
        rows: list[dict[str, Any]] = []
        for index, (contact_a, contact_b) in enumerate(aligned, start=1):
            differences: list[str]
            if contact_a is None:
                kind = "pass-b-only"
                differences = ["existence"]
                totals["pass_b_only"] += 1
            elif contact_b is None:
                kind = "pass-a-only"
                differences = ["existence"]
                totals["pass_a_only"] += 1
            else:
                totals["aligned_pairs"] += 1
                differences = _differences(contact_a, contact_b)
                if differences:
                    kind = "disagreement"
                    totals["disagreements"] += 1
                else:
                    kind = "agreement"
                    totals["exact_agreements"] += 1
            rows.append(
                {
                    "row_id": f"{clip_id}.r{index:04d}",
                    "kind": kind,
                    "differences": differences,
                    "pass_a": contact_a,
                    "pass_b": contact_b,
                }
            )
        base = dict(annotation_a)
        base["contacts"] = []
        clip_documents.append(
            {
                "clip_id": clip_id,
                "actor_ids": session_clip_a[clip_id]["actor_ids"],
                "pts_map_file": clips_a[clip_id]["pts_map_file"],
                "source_sha256": clips_a[clip_id]["source_sha256"],
                "annotation_base": base,
                "rows": rows,
            }
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": COMPARISON_KIND,
        "compared_at_utc": timestamp,
        "session_id": final_a["session_id"],
        "pass_a": {
            "pass_id": final_a["pass_id"],
            "annotator": final_a["annotator"],
            "finalization_file": _relative(a_dir / FINALIZATION_FILE, root),
            "finalization_sha256": sha256_file(a_dir / FINALIZATION_FILE),
        },
        "pass_b": {
            "pass_id": final_b["pass_id"],
            "annotator": final_b["annotator"],
            "finalization_file": _relative(b_dir / FINALIZATION_FILE, root),
            "finalization_sha256": sha256_file(b_dir / FINALIZATION_FILE),
        },
        "policy": {
            "pairing": "actor-scoped-monotonic-nearest-v1",
            "max_pair_distance_frames": max_pair_distance_frames,
            "prediction_fields_loaded": False,
        },
        "summary": {
            "clip_count": len(clip_documents),
            **totals,
            "rows_requiring_adjudication": (
                totals["disagreements"] + totals["pass_a_only"] + totals["pass_b_only"]
            ),
        },
        "clips": clip_documents,
    }
    return document


def compare_passes(
    pass_a_dir: str | Path,
    pass_b_dir: str | Path,
    *,
    out_dir: str | Path,
    repo_root: str | Path,
    compared_at_utc: str,
    max_pair_distance_frames: int = 6,
) -> dict[str, Any]:
    """Verify two frozen passes and create a prediction-free comparison package."""

    root = Path(repo_root).resolve()
    document = _build_comparison_document(
        pass_a_dir,
        pass_b_dir,
        repo_root=root,
        compared_at_utc=compared_at_utc,
        max_pair_distance_frames=max_pair_distance_frames,
    )
    output = _inside(out_dir, root, "out_dir")
    if output.exists() and any(output.iterdir()):
        raise ValueError("out_dir must be empty; comparison artifacts are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    _write_new(output / COMPARISON_FILE, _canonical_bytes(document))
    _write_new(output / ADJUDICATION_LOG, b"")
    return document


def _load_comparison(package_dir: Path) -> dict[str, Any]:
    path = package_dir / COMPARISON_FILE
    document = _read_json(path, "comparison")
    expected = {
        "schema_version", "kind", "compared_at_utc", "session_id", "pass_a",
        "pass_b", "policy", "summary", "clips",
    }
    _strict_keys(document, expected, str(path))
    if document["schema_version"] != SCHEMA_VERSION or document["kind"] != COMPARISON_KIND:
        raise ValueError(f"{path}: unsupported comparison schema/kind")
    _utc(document["compared_at_utc"], f"{path}.compared_at_utc")
    _string(document["session_id"], f"{path}.session_id")
    for label in ("pass_a", "pass_b"):
        side = _strict_keys(
            document[label],
            {"pass_id", "annotator", "finalization_file", "finalization_sha256"},
            f"{path}.{label}",
        )
        for key in ("pass_id", "annotator", "finalization_file"):
            _string(side[key], f"{path}.{label}.{key}")
        _digest(side["finalization_sha256"], f"{path}.{label}.finalization_sha256")
    policy = _strict_keys(
        document["policy"],
        {"pairing", "max_pair_distance_frames", "prediction_fields_loaded"},
        f"{path}.policy",
    )
    if policy["pairing"] != "actor-scoped-monotonic-nearest-v1":
        raise ValueError(f"{path}.policy: unsupported pairing policy")
    distance = policy["max_pair_distance_frames"]
    if not isinstance(distance, int) or isinstance(distance, bool) or distance < 0:
        raise ValueError(f"{path}.policy.max_pair_distance_frames: must be non-negative")
    if policy["prediction_fields_loaded"] is not False:
        raise ValueError(f"{path}.policy: prediction_fields_loaded must be false")
    summary_keys = {
        "clip_count", "contacts_a", "contacts_b", "aligned_pairs",
        "exact_agreements", "disagreements", "pass_a_only", "pass_b_only",
        "rows_requiring_adjudication",
    }
    summary = _strict_keys(document["summary"], summary_keys, f"{path}.summary")
    if any(not isinstance(summary[key], int) or isinstance(summary[key], bool) or
           summary[key] < 0 for key in summary_keys):
        raise ValueError(f"{path}.summary: every count must be a non-negative integer")
    clips = document["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"{path}.clips: must be a non-empty array")
    for clip_index, clip in enumerate(clips):
        clip_where = f"{path}.clips[{clip_index}]"
        _strict_keys(
            clip,
            {"clip_id", "actor_ids", "pts_map_file", "source_sha256",
             "annotation_base", "rows"},
            clip_where,
        )
        _string(clip["clip_id"], f"{clip_where}.clip_id")
        _string(clip["pts_map_file"], f"{clip_where}.pts_map_file")
        _digest(clip["source_sha256"], f"{clip_where}.source_sha256")
        if not isinstance(clip["rows"], list):
            raise ValueError(f"{clip_where}.rows: must be an array")
        for row_index, row in enumerate(clip["rows"]):
            row_where = f"{clip_where}.rows[{row_index}]"
            _strict_keys(
                row,
                {"row_id", "kind", "differences", "pass_a", "pass_b"},
                row_where,
            )
            _string(row["row_id"], f"{row_where}.row_id")
            if row["kind"] not in {
                "agreement", "disagreement", "pass-a-only", "pass-b-only"
            }:
                raise ValueError(f"{row_where}.kind: unsupported row kind")
            if (not isinstance(row["differences"], list) or
                    any(not isinstance(item, str) for item in row["differences"])):
                raise ValueError(f"{row_where}.differences: must be a string array")
    if path.read_bytes() != _canonical_bytes(document):
        raise ValueError(f"{path}: comparison is not canonical JSON")
    return document


def _comparison_rows(
    comparison: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    rows: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for clip in comparison["clips"]:
        for row in clip["rows"]:
            row_id = row["row_id"]
            if row_id in rows:
                raise ValueError(f"comparison has duplicate row ID {row_id!r}")
            rows[row_id] = (clip, row)
    return rows


def _validate_decision_payload(
    payload: Any,
    *,
    rows: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    repo_root: Path,
) -> dict[str, Any]:
    decision = _strict_keys(
        payload,
        {
            "decision_id", "row_id", "supersedes", "decided_at_utc", "adjudicator",
            "resolution", "contacts", "notes",
        },
        "decision",
    )
    for key in ("decision_id", "row_id", "adjudicator"):
        _string(decision[key], f"decision.{key}")
    _utc(decision["decided_at_utc"], "decision.decided_at_utc")
    if decision["supersedes"] is not None:
        _string(decision["supersedes"], "decision.supersedes")
    if decision["resolution"] not in {"resolved", "unresolved"}:
        raise ValueError("decision.resolution must be 'resolved' or 'unresolved'")
    if not isinstance(decision["notes"], str):
        raise ValueError("decision.notes must be a string")
    contacts = decision["contacts"]
    if not isinstance(contacts, list):
        raise ValueError("decision.contacts must be an array")
    if decision["resolution"] == "unresolved" and contacts:
        raise ValueError("an unresolved decision cannot assert adjudicated contacts")
    if decision["row_id"] not in rows:
        raise ValueError(f"unknown comparison row {decision['row_id']!r}")
    clip, row = rows[decision["row_id"]]
    if row["kind"] == "agreement":
        raise ValueError("exact-agreement rows do not require an adjudication decision")
    pts_map = load_pts_map(repo_root / clip["pts_map_file"])
    if decision["resolution"] == "resolved":
        candidate = dict(clip["annotation_base"])
        candidate["annotator"] = decision["adjudicator"]
        candidate["pass_id"] = "adjudication"
        candidate["contacts"] = sorted(
            contacts,
            key=lambda contact: (
                contact.get("preferred_pts_us", 0) if isinstance(contact, dict) else 0,
                contact.get("contact_id", "") if isinstance(contact, dict) else "",
            ),
        )
        canonical = validate_annotation(candidate, pts_map=pts_map, where="decision.contacts")
        contacts = canonical["contacts"]
        unknown_actors = sorted(
            {contact["actor_id"] for contact in contacts} - set(clip["actor_ids"])
        )
        if unknown_actors:
            raise ValueError(f"decision.contacts: unknown actor IDs {unknown_actors}")
    return {
        "decision_id": decision["decision_id"],
        "row_id": decision["row_id"],
        "supersedes": decision["supersedes"],
        "decided_at_utc": decision["decided_at_utc"],
        "adjudicator": decision["adjudicator"],
        "resolution": decision["resolution"],
        "contacts": contacts,
        "notes": decision["notes"],
    }


def _read_log(
    package: Path,
    comparison: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str | None]:
    path = package / ADJUDICATION_LOG
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise ValueError(f"cannot read adjudication log {path}: {exc}") from None
    rows = _comparison_rows(comparison)
    records: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any]] = {}
    decision_ids: set[str] = set()
    previous_hash: str | None = None
    comparison_hash = sha256_file(package / COMPARISON_FILE)
    expected_record_keys = {
        "schema_version", "kind", "sequence", "previous_record_sha256",
        "comparison_sha256", "decision",
    }
    for index, line in enumerate(lines, start=1):
        if not line.endswith(b"\n"):
            raise ValueError(f"adjudication log line {index} has no terminating newline")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"adjudication log line {index} is invalid JSON: {exc}") from None
        _strict_keys(record, expected_record_keys, f"adjudication log line {index}")
        if line != _canonical_bytes(record, pretty=False):
            raise ValueError(f"adjudication log line {index} is not canonical JSONL")
        if record["schema_version"] != SCHEMA_VERSION or record["kind"] != DECISION_KIND:
            raise ValueError(f"adjudication log line {index} has unsupported schema/kind")
        if record["sequence"] != index:
            raise ValueError(f"adjudication log line {index} has wrong sequence")
        if record["previous_record_sha256"] != previous_hash:
            raise ValueError(f"adjudication log line {index} breaks the hash chain")
        if record["comparison_sha256"] != comparison_hash:
            raise ValueError(f"adjudication log line {index} binds another comparison")
        decision = _validate_decision_payload(record["decision"], rows=rows, repo_root=repo_root)
        decision_id = decision["decision_id"]
        if decision_id in decision_ids:
            raise ValueError(f"duplicate adjudication decision ID {decision_id!r}")
        current = active.get(decision["row_id"])
        expected_supersedes = current["decision_id"] if current else None
        if decision["supersedes"] != expected_supersedes:
            raise ValueError(
                f"decision {decision_id!r} must supersede {expected_supersedes!r}"
            )
        decision_ids.add(decision_id)
        active[decision["row_id"]] = decision
        records.append(record)
        previous_hash = _sha_bytes(line)
    return records, active, previous_hash


def _combined_clip_contacts(
    clip: dict[str, Any], active: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for row in clip["rows"]:
        if row["kind"] == "agreement":
            contacts.append(row["pass_a"])
        else:
            contacts.extend(active[row["row_id"]]["contacts"])
    return sorted(
        contacts, key=lambda item: (item["preferred_pts_us"], item["contact_id"])
    )


def verify_adjudication_package(
    package_dir: str | Path, *, repo_root: str | Path
) -> dict[str, Any]:
    """Verify source passes, comparison commitments, and adjudication hash chain."""

    root = Path(repo_root).resolve()
    package = _inside(package_dir, root, "package_dir")
    comparison = _load_comparison(package)
    pass_directories: list[Path] = []
    for label in ("pass_a", "pass_b"):
        side = comparison[label]
        finalization_path = _inside(root / side["finalization_file"], root, f"{label}.finalization")
        if sha256_file(finalization_path) != side["finalization_sha256"]:
            raise ValueError(f"{label} finalization bytes no longer match comparison")
        verify_finalized_pass(finalization_path.parent, repo_root=root)
        pass_directories.append(finalization_path.parent)
    expected_comparison = _build_comparison_document(
        pass_directories[0],
        pass_directories[1],
        repo_root=root,
        compared_at_utc=comparison["compared_at_utc"],
        max_pair_distance_frames=comparison["policy"]["max_pair_distance_frames"],
    )
    if comparison != expected_comparison:
        raise ValueError("comparison no longer matches the two finalized source-only passes")
    records, active, head = _read_log(package, comparison, root)
    required = comparison["summary"]["rows_requiring_adjudication"]
    resolved = sum(decision["resolution"] == "resolved" for decision in active.values())
    ready = len(active) == required and resolved == required
    if ready:
        for clip in comparison["clips"]:
            candidate = dict(clip["annotation_base"])
            candidate["annotator"] = "adjudication-package-verifier"
            candidate["pass_id"] = "adjudication"
            candidate["contacts"] = _combined_clip_contacts(clip, active)
            validate_annotation(
                candidate,
                pts_map=load_pts_map(root / clip["pts_map_file"]),
                where=f"adjudication package {clip['clip_id']}",
            )
    return {
        "session_id": comparison["session_id"],
        "comparison_sha256": sha256_file(package / COMPARISON_FILE),
        "record_count": len(records),
        "active_decision_count": len(active),
        "required_decision_count": required,
        "resolved_decision_count": resolved,
        "adjudication_head_sha256": head,
        "ready_to_seal": ready,
    }


def append_adjudication_decision(
    package_dir: str | Path,
    decision: Any,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Append one validated decision; revisions must explicitly supersede the active row."""

    root = Path(repo_root).resolve()
    package = _inside(package_dir, root, "package_dir")
    verify_adjudication_package(package, repo_root=root)
    comparison = _load_comparison(package)
    records, active, previous_hash = _read_log(package, comparison, root)
    canonical = _validate_decision_payload(
        decision, rows=_comparison_rows(comparison), repo_root=root
    )
    current = active.get(canonical["row_id"])
    expected_supersedes = current["decision_id"] if current else None
    if canonical["supersedes"] != expected_supersedes:
        raise ValueError(
            f"decision must supersede {expected_supersedes!r} for row {canonical['row_id']}"
        )
    if any(record["decision"]["decision_id"] == canonical["decision_id"] for record in records):
        raise ValueError(f"duplicate adjudication decision ID {canonical['decision_id']!r}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": DECISION_KIND,
        "sequence": len(records) + 1,
        "previous_record_sha256": previous_hash,
        "comparison_sha256": sha256_file(package / COMPARISON_FILE),
        "decision": canonical,
    }
    payload = _canonical_bytes(record, pretty=False)
    log_path = package / ADJUDICATION_LOG
    descriptor = os.open(log_path, os.O_WRONLY | os.O_APPEND)
    try:
        with os.fdopen(descriptor, "ab", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        # fdopen owns the descriptor after successful construction.
        pass
    return record


def seal_adjudication(
    package_dir: str | Path,
    *,
    out_dir: str | Path,
    repo_root: str | Path,
    sealed_at_utc: str,
    adjudicator: str,
    pass_id: str = "adjudicated-v1",
) -> dict[str, Any]:
    """Materialize canonical operator-gold annotations after every row is resolved."""

    timestamp = _utc(sealed_at_utc, "sealed_at_utc")
    operator = _string(adjudicator, "adjudicator")
    final_pass_id = _string(pass_id, "pass_id")
    root = Path(repo_root).resolve()
    package = _inside(package_dir, root, "package_dir")
    summary = verify_adjudication_package(package, repo_root=root)
    if not summary["ready_to_seal"]:
        raise ValueError(
            "adjudication is incomplete: every non-agreement row needs an active resolved decision"
        )
    comparison = _load_comparison(package)
    _, active, _ = _read_log(package, comparison, root)
    output = _inside(out_dir, root, "out_dir")
    if output.exists() and any(output.iterdir()):
        raise ValueError("out_dir must be empty; adjudicated truth is never overwritten")
    output.mkdir(parents=True, exist_ok=True)

    prepared: list[tuple[Path, bytes, str, int, int]] = []
    for clip in comparison["clips"]:
        contacts = _combined_clip_contacts(clip, active)
        document = dict(clip["annotation_base"])
        document["annotator"] = operator
        document["pass_id"] = final_pass_id
        document["contacts"] = contacts
        pts_map = load_pts_map(root / clip["pts_map_file"])
        path = output / f"{clip['clip_id']}.{final_pass_id}.foot-contact.gt.json"
        payload = annotation_json(document, pts_map=pts_map).encode("utf-8")
        prepared.append(
            (
                path,
                payload,
                clip["clip_id"],
                len(contacts),
                sum(contact["visibility"] == "clear" for contact in contacts),
            )
        )
    annotations: list[dict[str, Any]] = []
    for path, payload, clip_id, contact_count, clear_contact_count in prepared:
        _write_new(path, payload)
        annotations.append(
            {
                "clip_id": clip_id,
                "annotation_file": _relative(path, root),
                "annotation_sha256": sha256_file(path),
                "contact_count": contact_count,
                "clear_contact_count": clear_contact_count,
            }
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "sealed_at_utc": timestamp,
        "session_id": comparison["session_id"],
        "adjudicator": operator,
        "pass_id": final_pass_id,
        "source_pass_ids": [comparison["pass_a"]["pass_id"], comparison["pass_b"]["pass_id"]],
        "comparison_sha256": summary["comparison_sha256"],
        "adjudication_log_sha256": sha256_file(package / ADJUDICATION_LOG),
        "adjudication_head_sha256": summary["adjudication_head_sha256"],
        "decision_count": summary["active_decision_count"],
        "annotations": annotations,
    }
    _write_new(output / "adjudication-receipt.json", _canonical_bytes(receipt))
    return receipt
