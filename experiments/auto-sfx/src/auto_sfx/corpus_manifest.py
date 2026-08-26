"""Rights-aware source manifests and blind, source-group split reservations.

This module deliberately knows nothing about detector predictions.  It derives
leakage components from operator-supplied source/actor/session keys, verifies
media bytes, and makes a keyed deterministic reservation.  Locked membership
is only written to an explicit operator-private mapping; ordinary engineering
output contains development/challenge members and opaque holdout commitments.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SOURCE_SCHEMA = "auto-sfx-source-manifest/v1"
ENGINEERING_SCHEMA = "auto-sfx-engineering-corpus/v1"
PRIVATE_SPLIT_SCHEMA = "auto-sfx-private-split/v1"
RECEIPT_SCHEMA = "auto-sfx-split-receipt/v1"

REQUIRED_USE = "auto-sfx-local-research-evaluation"
VIEWPOINTS = frozenset({"sagittal", "diagonal", "frontal", "rear", "uncertain"})
REQUIRED_VIEWPOINTS = frozenset({"sagittal", "diagonal", "frontal", "rear"})
REQUIRED_LEAKAGE_PREFIXES = frozenset({"source", "actor", "session", "camera"})
ROLES = ("development", "challenge", "locked_holdout")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_LEAKAGE_KEY_RE = re.compile(r"^([a-z][a-z0-9_-]*):([^:\s].*)$")


class ManifestError(ValueError):
    """An operator-fixable manifest or reservation integrity error."""


@dataclass(frozen=True)
class SourceGroup:
    """One transitive leakage component."""

    group_id: str
    clip_ids: tuple[str, ...]
    leakage_keys: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedManifest:
    """Validated source manifest and its derived, deterministic inventory."""

    document: dict[str, Any]
    source_path: Path
    manifest_sha256: str
    groups: tuple[SourceGroup, ...]
    gate: dict[str, Any]
    media_hashes_verified: bool


def _fail(where: str | Path, message: str) -> None:
    raise ManifestError(f"{where}: {message}")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(raw: dict[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(where, f"{key} must be a non-empty string")
    return value


def _reject_unknown_keys(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        _fail(where, f"unknown fields are forbidden: {unknown}")


def _require_id(raw: dict[str, Any], key: str, where: str) -> str:
    value = _require_string(raw, key, where)
    if not _ID_RE.fullmatch(value):
        _fail(where, f"{key} must match {_ID_RE.pattern!r}")
    return value


def _require_utc(raw: dict[str, Any], key: str, where: str) -> str:
    value = _require_string(raw, key, where)
    if not _UTC_RE.fullmatch(value):
        _fail(where, f"{key} must be UTC as YYYY-MM-DDTHH:MM:SSZ")
    return value


def _require_nonnegative_int(raw: dict[str, Any], key: str, where: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(where, f"{key} must be a non-negative integer")
    return value


def _validate_rights(rights: Any, where: str) -> None:
    if not isinstance(rights, dict):
        _fail(where, "rights must be an object")
    _reject_unknown_keys(
        rights,
        {"status", "basis", "restrictions", "confirmed_by", "confirmed_at_utc",
         "allowed_uses", "evidence"},
        f"{where}.rights",
    )
    if rights.get("status") != "verified":
        _fail(where, "rights.status must be 'verified'; unknown/denied media is ineligible")
    _require_string(rights, "basis", where)
    _require_string(rights, "restrictions", where)
    _require_string(rights, "confirmed_by", where)
    _require_utc(rights, "confirmed_at_utc", where)
    uses = rights.get("allowed_uses")
    if (not isinstance(uses, list) or not uses or
            any(not isinstance(item, str) or not item for item in uses)):
        _fail(where, "rights.allowed_uses must be a non-empty string array")
    if REQUIRED_USE not in uses:
        _fail(where, f"rights.allowed_uses must explicitly include {REQUIRED_USE!r}")
    evidence = rights.get("evidence")
    if not isinstance(evidence, dict):
        _fail(where, "rights.evidence must be an object")
    _reject_unknown_keys(evidence, {"reference", "sha256"}, f"{where}.rights.evidence")
    _require_string(evidence, "reference", where)
    digest = _require_string(evidence, "sha256", where)
    if not _SHA256_RE.fullmatch(digest):
        _fail(where, "rights.evidence.sha256 must be 64 lowercase hex characters")


def _validate_provenance(provenance: Any, where: str) -> None:
    if not isinstance(provenance, dict):
        _fail(where, "provenance must be an object")
    _reject_unknown_keys(
        provenance,
        {"source_reference", "creator_or_owner", "acquired_by", "acquired_at_utc", "notes"},
        f"{where}.provenance",
    )
    for key in ("source_reference", "creator_or_owner", "acquired_by"):
        _require_string(provenance, key, where)
    _require_utc(provenance, "acquired_at_utc", where)
    if "notes" in provenance and not isinstance(provenance["notes"], str):
        _fail(where, "provenance.notes must be a string when supplied")


def _validate_gold_evidence(gold: Any, where: str) -> None:
    if gold is None:
        return
    if not isinstance(gold, dict):
        _fail(where, "gold_evidence must be null or an object")
    _reject_unknown_keys(
        gold,
        {"provenance", "annotation_sha256", "clear_contact_count",
         "annotated_duration_us", "hard_negative_duration_us", "pass_ids"},
        f"{where}.gold_evidence",
    )
    if gold.get("provenance") != "operator-gold":
        _fail(where, "gold_evidence.provenance must be 'operator-gold'")
    digest = _require_string(gold, "annotation_sha256", where)
    if not _SHA256_RE.fullmatch(digest):
        _fail(where, "gold_evidence.annotation_sha256 must be 64 lowercase hex characters")
    _require_nonnegative_int(gold, "clear_contact_count", where)
    annotated = _require_nonnegative_int(gold, "annotated_duration_us", where)
    hard_negative = _require_nonnegative_int(gold, "hard_negative_duration_us", where)
    if annotated <= 0:
        _fail(where, "gold_evidence.annotated_duration_us must be positive")
    if hard_negative > annotated:
        _fail(where, "hard-negative duration cannot exceed annotated duration")
    pass_ids = gold.get("pass_ids")
    if (not isinstance(pass_ids, list) or len(pass_ids) < 2 or
            any(not isinstance(item, str) or not item for item in pass_ids) or
            len(set(pass_ids)) != len(pass_ids)):
        _fail(
            where,
            "gold_evidence.pass_ids must contain at least two unique pass IDs",
        )


def _validate_source(source: Any, index: int, manifest_dir: Path,
                     verify_media: bool) -> None:
    where = f"source[{index}]"
    if not isinstance(source, dict):
        _fail(where, "must be an object")
    _reject_unknown_keys(
        source,
        {"clip_id", "media", "rights", "provenance", "leakage_keys",
         "viewpoints", "actor_ids", "camera_ids", "gold_evidence"},
        where,
    )
    _require_id(source, "clip_id", where)

    media = source.get("media")
    if not isinstance(media, dict):
        _fail(where, "media must be an object")
    _reject_unknown_keys(media, {"path", "sha256", "bytes", "duration_us"}, f"{where}.media")
    media_path = _require_string(media, "path", where)
    relative_media_path = PurePosixPath(media_path)
    if (relative_media_path.is_absolute() or ".." in relative_media_path.parts or
            "." in relative_media_path.parts or "\\" in media_path or
            relative_media_path.as_posix() != media_path):
        _fail(where, "media.path must be a normalized relative POSIX path without traversal")
    digest = _require_string(media, "sha256", where)
    if not _SHA256_RE.fullmatch(digest):
        _fail(where, "media.sha256 must be 64 lowercase hex characters")
    declared_bytes = _require_nonnegative_int(media, "bytes", where)
    duration_us = _require_nonnegative_int(media, "duration_us", where)
    if duration_us <= 0:
        _fail(where, "media.duration_us must be positive")
    if verify_media:
        path = manifest_dir / media_path
        if not path.is_file():
            _fail(where, f"media file is missing or not a regular file: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != declared_bytes:
            _fail(where, f"media.bytes is {declared_bytes}, actual size is {actual_bytes}")
        actual_digest = sha256_file(path)
        if actual_digest != digest:
            _fail(where, f"media.sha256 mismatch: expected {digest}, got {actual_digest}")

    _validate_rights(source.get("rights"), where)
    _validate_provenance(source.get("provenance"), where)

    keys = source.get("leakage_keys")
    if (not isinstance(keys, list) or not keys or
            any(not isinstance(item, str) for item in keys)):
        _fail(where, "leakage_keys must be a non-empty string array")
    if keys != sorted(set(keys)):
        _fail(where, "leakage_keys must be sorted and unique")
    prefixes: set[str] = set()
    for key in keys:
        match = _LEAKAGE_KEY_RE.fullmatch(key)
        if not match:
            _fail(where, f"invalid leakage key {key!r}; expected namespace:value")
        prefixes.add(match.group(1))
    missing = sorted(REQUIRED_LEAKAGE_PREFIXES - prefixes)
    if missing:
        _fail(where, f"leakage_keys missing required namespaces: {missing}")

    viewpoints = source.get("viewpoints")
    if (not isinstance(viewpoints, list) or not viewpoints or
            any(item not in VIEWPOINTS for item in viewpoints)):
        _fail(where, f"viewpoints must be a non-empty array drawn from {sorted(VIEWPOINTS)}")
    if viewpoints != sorted(set(viewpoints)):
        _fail(where, "viewpoints must be sorted and unique")
    actors = source.get("actor_ids")
    if (not isinstance(actors, list) or not actors or
            any(not isinstance(item, str) or not item for item in actors) or
            actors != sorted(set(actors))):
        _fail(where, "actor_ids must be a non-empty sorted unique string array")
    cameras = source.get("camera_ids")
    if (not isinstance(cameras, list) or not cameras or
            any(not isinstance(item, str) or not item for item in cameras) or
            cameras != sorted(set(cameras))):
        _fail(where, "camera_ids must be a non-empty sorted unique string array")
    for namespace, identifiers in (("actor", actors), ("camera", cameras)):
        expected = {f"{namespace}:{identifier}" for identifier in identifiers}
        declared = {key for key in keys if key.startswith(f"{namespace}:")}
        if expected != declared:
            _fail(
                where,
                f"{namespace}_ids must exactly match {namespace}: leakage keys; "
                f"expected {sorted(expected)}, got {sorted(declared)}",
            )
    _validate_gold_evidence(source.get("gold_evidence"), where)


def _derive_groups(sources: list[dict[str, Any]]) -> tuple[SourceGroup, ...]:
    """Derive transitive groups: sharing any leakage key joins two clips."""

    parents = list(range(len(sources)))

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parents[right_root] = left_root

    owners: dict[str, int] = {}
    for index, source in enumerate(sources):
        for key in source["leakage_keys"]:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    components: dict[int, list[int]] = {}
    for index in range(len(sources)):
        components.setdefault(find(index), []).append(index)

    groups: list[SourceGroup] = []
    for members in components.values():
        clip_ids = tuple(sorted(sources[index]["clip_id"] for index in members))
        keys = tuple(sorted({key for index in members
                             for key in sources[index]["leakage_keys"]}))
        identity = {"clip_ids": clip_ids, "leakage_keys": keys}
        group_id = "sg-" + _sha256_bytes(_canonical_bytes(identity))[:20]
        groups.append(SourceGroup(group_id, clip_ids, keys))
    return tuple(sorted(groups, key=lambda group: group.group_id))


def _criterion(status: str, observed: Any, minimum: Any) -> dict[str, Any]:
    return {"status": status, "observed": observed, "minimum": minimum}


def _build_gate(sources: list[dict[str, Any]], groups: tuple[SourceGroup, ...]) -> dict[str, Any]:
    criteria: dict[str, Any] = {}
    criteria["source_groups"] = _criterion(
        "pass" if len(groups) >= 12 else "fail", len(groups), 12,
    )

    observed_views = sorted({view for source in sources for view in source["viewpoints"]
                             if view != "uncertain"})
    criteria["multiple_viewpoints"] = {
        "status": "pass" if len(observed_views) >= 2 else "fail",
        "observed": observed_views,
        "minimum_distinct": 2,
    }
    missing_views = sorted(REQUIRED_VIEWPOINTS - set(observed_views))
    criteria["required_viewpoint_coverage"] = {
        "status": "pass" if not missing_views else "fail",
        "observed": observed_views,
        "required": sorted(REQUIRED_VIEWPOINTS),
        "missing": missing_views,
    }

    observed_actors = sorted({actor for source in sources for actor in source["actor_ids"]})
    criteria["multiple_actors"] = {
        "status": "pass" if len(observed_actors) >= 2 else "fail",
        "observed_count": len(observed_actors),
        "minimum_distinct": 2,
    }
    observed_cameras = sorted({camera for source in sources for camera in source["camera_ids"]})
    criteria["multiple_cameras"] = {
        "status": "pass" if len(observed_cameras) >= 2 else "fail",
        "observed_count": len(observed_cameras),
        "minimum_distinct": 2,
    }

    evidence = [source.get("gold_evidence") for source in sources]
    if evidence and all(item is not None for item in evidence):
        clear_contacts = sum(item["clear_contact_count"] for item in evidence)
        annotated_duration = sum(item["annotated_duration_us"] for item in evidence)
        hard_negative_duration = sum(item["hard_negative_duration_us"] for item in evidence)
        ratio = hard_negative_duration / annotated_duration if annotated_duration else 0.0
        criteria["clear_contacts"] = _criterion(
            "pass" if clear_contacts >= 200 else "fail", clear_contacts, 200,
        )
        criteria["hard_negative_fraction"] = _criterion(
            "pass" if ratio >= 0.25 else "fail", round(ratio, 9), 0.25,
        )
    else:
        known = sum(item is not None for item in evidence)
        criteria["clear_contacts"] = {
            "status": "pending", "observed": None, "minimum": 200,
            "gold_sources_present": known, "source_count": len(sources),
        }
        criteria["hard_negative_fraction"] = {
            "status": "pending", "observed": None, "minimum": 0.25,
            "gold_sources_present": known, "source_count": len(sources),
        }

    statuses = {criterion["status"] for criterion in criteria.values()}
    overall = "fail" if "fail" in statuses else ("pending" if "pending" in statuses else "pass")
    return {
        "status": overall,
        "claim": "operator-gold challenge corpus readiness",
        "criteria": criteria,
    }


def load_source_manifest(path: str | Path, *, verify_media: bool = True,
                         require_ready: bool = False) -> ValidatedManifest:
    """Load and validate a rights/source manifest without consulting predictions."""

    source_path = Path(path).resolve()
    try:
        text = source_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ManifestError(f"{source_path}: cannot read manifest: {exc}") from None
    try:
        document = json.loads(text, object_pairs_hook=_no_duplicate_object)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{source_path}: invalid JSON: {exc}") from None
    if not isinstance(document, dict):
        _fail(source_path, "manifest must be a JSON object")
    _reject_unknown_keys(
        document,
        {"schema_version", "manifest_status", "corpus_id", "created_at_utc",
         "prepared_by", "sources"},
        str(source_path),
    )
    if document.get("schema_version") != SOURCE_SCHEMA:
        _fail(source_path, f"schema_version must be {SOURCE_SCHEMA!r}")
    status = document.get("manifest_status")
    if status not in {"template-empty", "ready"}:
        _fail(source_path, "manifest_status must be 'template-empty' or 'ready'")
    sources = document.get("sources")
    if not isinstance(sources, list):
        _fail(source_path, "sources must be an array")

    if status == "template-empty":
        if sources:
            _fail(source_path, "template-empty manifest must have no sources")
        if require_ready:
            _fail(source_path, "template-empty is not a real corpus and cannot be reserved")
        corpus_id = document.get("corpus_id")
        if corpus_id is not None:
            _fail(source_path, "template-empty corpus_id must be null")
        if document.get("created_at_utc") is not None or document.get("prepared_by") is not None:
            _fail(source_path, "template-empty created_at_utc and prepared_by must be null")
        gate = {
            "status": "pending", "claim": "operator-gold challenge corpus readiness",
            "criteria": {}, "reason": "no corpus supplied",
        }
        return ValidatedManifest(
            document, source_path, _sha256_bytes(_canonical_bytes(document)), (), gate,
            False,
        )

    _require_id(document, "corpus_id", str(source_path))
    _require_utc(document, "created_at_utc", str(source_path))
    _require_string(document, "prepared_by", str(source_path))
    if not sources:
        _fail(source_path, "ready manifest must contain at least one source")
    for index, source in enumerate(sources):
        _validate_source(source, index, source_path.parent, verify_media)

    clip_ids = [source["clip_id"] for source in sources]
    if len(clip_ids) != len(set(clip_ids)):
        _fail(source_path, "clip_id values must be unique")
    content_hashes = [source["media"]["sha256"] for source in sources]
    if len(content_hashes) != len(set(content_hashes)):
        _fail(source_path, "duplicate media content hashes are forbidden")
    if clip_ids != sorted(clip_ids):
        _fail(source_path, "sources must be sorted by clip_id")

    groups = _derive_groups(sources)
    gate = _build_gate(sources, groups)
    return ValidatedManifest(
        document, source_path, _sha256_bytes(_canonical_bytes(document)), groups, gate,
        verify_media,
    )


def _hmac_hex(secret: bytes, domain: str, value: Any) -> str:
    message = domain.encode("ascii") + b"\0" + _canonical_bytes(value)
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _fraction_count(group_count: int, fraction: float, label: str) -> int:
    if not isinstance(fraction, (float, int)) or isinstance(fraction, bool):
        raise ManifestError(f"{label} fraction must be numeric")
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ManifestError(f"{label} fraction must be between zero and one")
    return max(1, int(math.floor(group_count * fraction + 0.5)))


def build_reservation(
    manifest: ValidatedManifest,
    secret: bytes,
    *,
    reserved_at_utc: str,
    challenge_fraction: float = 0.25,
    locked_holdout_fraction: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build engineering, private, and public documents deterministically."""

    if manifest.document.get("manifest_status") != "ready":
        raise ManifestError("template-empty is not a real corpus and cannot be reserved")
    if not manifest.media_hashes_verified:
        raise ManifestError("reservation requires media byte counts and SHA-256 to be verified")
    if len(secret) < 32:
        raise ManifestError("reservation secret must contain at least 32 bytes")
    if not _UTC_RE.fullmatch(reserved_at_utc):
        raise ManifestError("reserved_at_utc must be UTC as YYYY-MM-DDTHH:MM:SSZ")
    groups = manifest.groups
    challenge_count = _fraction_count(len(groups), challenge_fraction, "challenge")
    holdout_count = _fraction_count(len(groups), locked_holdout_fraction, "locked holdout")
    if challenge_count + holdout_count >= len(groups):
        raise ManifestError("fractions leave no development source group")

    corpus_id = manifest.document["corpus_id"]
    ranked = sorted(
        groups,
        key=lambda group: (
            _hmac_hex(secret, "auto-sfx-group-rank-v1", [corpus_id, group.group_id]),
            group.group_id,
        ),
    )
    role_by_group: dict[str, str] = {}
    for index, group in enumerate(ranked):
        if index < holdout_count:
            role = "locked_holdout"
        elif index < holdout_count + challenge_count:
            role = "challenge"
        else:
            role = "development"
        role_by_group[group.group_id] = role

    group_by_clip = {clip_id: group for group in groups for clip_id in group.clip_ids}
    private_groups = []
    for group in groups:
        hashes = sorted(
            source["media"]["sha256"] for source in manifest.document["sources"]
            if source["clip_id"] in group.clip_ids
        )
        private_groups.append({
            "source_group_id": group.group_id,
            "role": role_by_group[group.group_id],
            "clip_ids": list(group.clip_ids),
            "content_sha256": hashes,
        })

    reservation_id = "rs-" + _hmac_hex(
        secret, "auto-sfx-reservation-id-v1",
        [corpus_id, manifest.manifest_sha256, challenge_fraction, locked_holdout_fraction],
    )[:24]
    assignment_commitment = _hmac_hex(
        secret, "auto-sfx-assignment-v1", private_groups,
    )
    locked_groups = [group for group in private_groups if group["role"] == "locked_holdout"]
    locked_commitment = _hmac_hex(
        secret, "auto-sfx-locked-membership-v1", locked_groups,
    )

    engineering_sources = []
    for source in manifest.document["sources"]:
        group = group_by_clip[source["clip_id"]]
        role = role_by_group[group.group_id]
        if role == "locked_holdout":
            continue
        entry = dict(source)
        entry["source_group_id"] = group.group_id
        entry["reserved_role"] = role
        engineering_sources.append(entry)

    engineering = {
        "schema_version": ENGINEERING_SCHEMA,
        "corpus_id": corpus_id,
        "reservation_id": reservation_id,
        "source_manifest_sha256": manifest.manifest_sha256,
        "warning": "Contains development/challenge only. Locked holdout membership is operator-private.",
        "sources": engineering_sources,
    }
    private = {
        "schema_version": PRIVATE_SPLIT_SCHEMA,
        "corpus_id": corpus_id,
        "reservation_id": reservation_id,
        "reserved_at_utc": reserved_at_utc,
        "source_manifest_sha256": manifest.manifest_sha256,
        "algorithm": "HMAC-SHA256 group rank over transitive leakage components, v1",
        "fractions": {
            "challenge": challenge_fraction,
            "locked_holdout": locked_holdout_fraction,
        },
        "assignment_commitment_hmac_sha256": assignment_commitment,
        "groups": private_groups,
    }

    role_group_counts = {
        role: sum(item["role"] == role for item in private_groups) for role in ROLES
    }
    role_clip_counts = {
        role: sum(len(item["clip_ids"]) for item in private_groups if item["role"] == role)
        for role in ROLES
    }
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "corpus_id": corpus_id,
        "reservation_id": reservation_id,
        "reserved_at_utc": reserved_at_utc,
        "source_manifest_sha256": manifest.manifest_sha256,
        "algorithm": "HMAC-SHA256 group rank over transitive leakage components, v1",
        "counts": {
            "development": {
                "source_groups": role_group_counts["development"],
                "clips": role_clip_counts["development"],
            },
            "challenge": {
                "source_groups": role_group_counts["challenge"],
                "clips": role_clip_counts["challenge"],
            },
            "locked_holdout": {
                "source_groups": role_group_counts["locked_holdout"],
                "clips": role_clip_counts["locked_holdout"],
            },
        },
        "assignment_commitment_hmac_sha256": assignment_commitment,
        "locked_membership_commitment_hmac_sha256": locked_commitment,
        "leakage_audit": {
            "status": "clean",
            "grouping": "transitive closure over every declared leakage key",
            "source_group_overlap": 0,
            "content_hash_overlap": 0,
        },
        "corpus_gate": manifest.gate,
        "locked_holdout_policy": (
            "No locked clip ID, source-group ID, path, or content hash is present in this receipt "
            "or the engineering manifest."
        ),
    }
    return engineering, private, receipt


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"),
                           object_pairs_hook=_no_duplicate_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{path}: cannot read valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: expected a JSON object")
    return value


def audit_reservation(
    manifest: ValidatedManifest,
    secret: bytes,
    engineering: dict[str, Any],
    private: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild a reservation and prove grouping, bytes, and blind boundary."""

    if private.get("schema_version") != PRIVATE_SPLIT_SCHEMA:
        raise ManifestError("operator mapping has an unexpected schema_version")
    fractions = private.get("fractions")
    if not isinstance(fractions, dict):
        raise ManifestError("operator mapping is missing fractions")
    expected_engineering, expected_private, expected_receipt = build_reservation(
        manifest,
        secret,
        reserved_at_utc=private.get("reserved_at_utc", ""),
        challenge_fraction=fractions.get("challenge"),
        locked_holdout_fraction=fractions.get("locked_holdout"),
    )
    if engineering != expected_engineering:
        raise ManifestError("engineering manifest differs from deterministic reconstruction")
    if private != expected_private:
        raise ManifestError("operator mapping differs from deterministic reconstruction")
    if receipt != expected_receipt:
        raise ManifestError("public receipt differs from deterministic reconstruction")

    locked_ids: set[str] = set()
    locked_groups: set[str] = set()
    locked_hashes: set[str] = set()
    for group in private["groups"]:
        if group["role"] == "locked_holdout":
            locked_groups.add(group["source_group_id"])
            locked_ids.update(group["clip_ids"])
            locked_hashes.update(group["content_sha256"])
    ordinary_text = json.dumps([engineering, receipt], sort_keys=True)
    leaked = sorted(value for value in locked_ids | locked_groups | locked_hashes
                    if value in ordinary_text)
    if leaked:
        raise ManifestError("locked membership appears in ordinary engineering output")
    return {
        "status": "clean",
        "reservation_id": receipt["reservation_id"],
        "source_groups": len(manifest.groups),
        "clips": len(manifest.document["sources"]),
        "locked_membership_exposed": False,
        "media_hashes_verified": True,
    }


def load_reservation_documents(engineering_path: str | Path,
                               private_path: str | Path,
                               receipt_path: str | Path) -> tuple[dict[str, Any], ...]:
    return tuple(_read_json(Path(path)) for path in
                 (engineering_path, private_path, receipt_path))


def _atomic_write_new(path: Path, document: dict[str, Any], *, private: bool) -> None:
    if path.exists():
        raise ManifestError(f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        mode = 0o600 if private else 0o644
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_bytes(document))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def write_reservation_documents(
    engineering_path: str | Path,
    private_path: str | Path,
    receipt_path: str | Path,
    documents: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    """Write new reservation outputs without overwriting any frozen artifact."""

    engineering_path = Path(engineering_path).resolve()
    private_path = Path(private_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    if len({engineering_path, private_path, receipt_path}) != 3:
        raise ManifestError("engineering, private, and receipt outputs must be distinct paths")
    for path in (engineering_path, private_path, receipt_path):
        if path.exists():
            raise ManifestError(f"refusing to overwrite frozen output: {path}")
    engineering, private, receipt = documents
    _atomic_write_new(engineering_path, engineering, private=False)
    _atomic_write_new(private_path, private, private=True)
    _atomic_write_new(receipt_path, receipt, private=False)


def load_secret(path: str | Path) -> bytes:
    try:
        secret = Path(path).read_bytes().strip()
    except OSError as exc:
        raise ManifestError(f"cannot read reservation secret: {exc}") from None
    if len(secret) < 32:
        raise ManifestError("reservation secret must contain at least 32 bytes")
    return secret


def enforce_operator_private_path(
    path: str | Path,
    *,
    protected_roots: Iterable[str | Path],
    label: str,
    require_existing: bool,
) -> Path:
    """Fail closed if operator material is in a shared workspace or open directory."""

    resolved = Path(path).resolve()
    for root in protected_roots:
        root_path = Path(root).resolve()
        if resolved == root_path or root_path in resolved.parents:
            raise ManifestError(f"{label} must be outside shared workspace {root_path}")
    if require_existing and not resolved.is_file():
        raise ManifestError(f"{label} does not exist as a regular file: {resolved}")
    parent = resolved.parent
    if not parent.is_dir():
        raise ManifestError(f"{label} parent directory must already exist: {parent}")
    parent_mode = stat.S_IMODE(parent.stat().st_mode)
    if parent_mode & 0o077:
        raise ManifestError(
            f"{label} parent directory must deny group/other access (mode 0700 or stricter): "
            f"{parent} is {parent_mode:04o}"
        )
    return resolved


def safe_validation_summary(manifest: ValidatedManifest) -> dict[str, Any]:
    """Return aggregate validation output without source membership."""

    return {
        "status": "valid",
        "manifest_status": manifest.document["manifest_status"],
        "corpus_id": manifest.document.get("corpus_id"),
        "source_manifest_sha256": manifest.manifest_sha256,
        "clips": len(manifest.document["sources"]),
        "source_groups": len(manifest.groups),
        "corpus_gate": manifest.gate,
        "media_hashes_verified": manifest.media_hashes_verified,
    }
