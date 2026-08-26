"""Corpus integrity, leakage grouping, and blind reservation tests."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.corpus_manifest import (  # noqa: E402
    ManifestError,
    audit_reservation,
    build_reservation,
    enforce_operator_private_path,
    load_source_manifest,
    safe_validation_summary,
    write_reservation_documents,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source(tmp_path: Path, index: int, *, with_gold: bool = True) -> dict:
    content = f"synthetic-media-fixture-{index}".encode()
    media = tmp_path / f"clip-{index:02d}.bin"
    media.write_bytes(content)
    actor = f"actor-{index:02d}"
    camera = f"camera-{index:02d}"
    keys = sorted([
        f"actor:{actor}",
        f"camera:{camera}",
        f"session:session-{index:02d}",
        f"source:origin-{index:02d}",
    ])
    gold = None
    if with_gold:
        gold = {
            "provenance": "operator-gold",
            "annotation_sha256": _sha(f"annotation-{index}".encode()),
            "clear_contact_count": 17,
            "annotated_duration_us": 1_000_000,
            "hard_negative_duration_us": 250_000,
            "pass_ids": ["operator-pass-a", "operator-pass-b"],
        }
    return {
        "clip_id": f"clip-{index:02d}",
        "media": {
            "path": media.name,
            "sha256": _sha(content),
            "bytes": len(content),
            "duration_us": 1_000_000,
        },
        "rights": {
            "status": "verified",
            "basis": "synthetic test fixture",
            "allowed_uses": ["auto-sfx-local-research-evaluation"],
            "restrictions": "test fixture only",
            "evidence": {
                "reference": f"fixture-rights-{index}",
                "sha256": _sha(f"rights-{index}".encode()),
            },
            "confirmed_by": "pytest",
            "confirmed_at_utc": "2026-08-26T10:00:00Z",
        },
        "provenance": {
            "source_reference": f"generated-test-fixture-{index}",
            "creator_or_owner": "pytest",
            "acquired_by": "pytest",
            "acquired_at_utc": "2026-08-26T10:00:00Z",
        },
        "leakage_keys": keys,
        "viewpoints": [["sagittal", "diagonal", "frontal", "rear"][index % 4]],
        "actor_ids": [actor],
        "camera_ids": [camera],
        "gold_evidence": gold,
    }


def _write_manifest(tmp_path: Path, *, count: int = 12,
                    with_gold: bool = True,
                    mutate=None) -> Path:
    sources = [_source(tmp_path, index, with_gold=with_gold) for index in range(count)]
    if mutate:
        mutate(sources)
    document = {
        "schema_version": "auto-sfx-source-manifest/v1",
        "manifest_status": "ready",
        "corpus_id": "operator-challenge-001",
        "created_at_utc": "2026-08-26T10:00:00Z",
        "prepared_by": "pytest-operator",
        "sources": sources,
    }
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _load(tmp_path: Path, **kwargs):
    return load_source_manifest(_write_manifest(tmp_path, **kwargs))


def test_checked_in_empty_template_is_valid_but_not_reservable():
    template = Path(__file__).resolve().parents[1] / "corpus" / "source-manifest.empty.json"
    manifest = load_source_manifest(template)
    assert safe_validation_summary(manifest)["manifest_status"] == "template-empty"
    assert safe_validation_summary(manifest)["media_hashes_verified"] is False
    assert manifest.gate["status"] == "pending"
    assert manifest.groups == ()
    with pytest.raises(ManifestError, match="cannot be reserved"):
        load_source_manifest(template, require_ready=True)


def test_full_gate_passes_only_from_supplied_operator_evidence(tmp_path):
    manifest = _load(tmp_path)
    gate = manifest.gate
    assert gate["status"] == "pass"
    criteria = gate["criteria"]
    assert criteria["source_groups"]["observed"] == 12
    assert criteria["clear_contacts"]["observed"] == 204
    assert criteria["hard_negative_fraction"]["observed"] == 0.25
    assert criteria["required_viewpoint_coverage"]["missing"] == []
    assert criteria["multiple_actors"]["observed_count"] == 12
    assert criteria["multiple_cameras"]["observed_count"] == 12


def test_supplied_gold_below_contact_and_hard_negative_thresholds_fails(tmp_path):
    def below_gate(sources):
        for source in sources:
            source["gold_evidence"]["clear_contact_count"] = 16
            source["gold_evidence"]["hard_negative_duration_us"] = 249_999

    manifest = _load(tmp_path, mutate=below_gate)
    criteria = manifest.gate["criteria"]
    assert manifest.gate["status"] == "fail"
    assert criteria["clear_contacts"] == {
        "status": "fail", "observed": 192, "minimum": 200,
    }
    assert criteria["hard_negative_fraction"]["status"] == "fail"
    assert criteria["hard_negative_fraction"]["observed"] < 0.25


def test_annotation_dependent_gates_remain_pending_when_gold_is_absent(tmp_path):
    manifest = _load(tmp_path, with_gold=False)
    assert manifest.gate["status"] == "pending"
    assert manifest.gate["criteria"]["clear_contacts"]["status"] == "pending"
    assert manifest.gate["criteria"]["hard_negative_fraction"]["observed"] is None


def test_schema_only_manifest_cannot_be_reserved(tmp_path):
    path = _write_manifest(tmp_path)
    manifest = load_source_manifest(path, verify_media=False)
    assert safe_validation_summary(manifest)["media_hashes_verified"] is False
    with pytest.raises(ManifestError, match="requires media byte counts"):
        build_reservation(
            manifest, b"reservation-secret-is-at-least-32-bytes",
            reserved_at_utc="2026-08-26T11:00:00Z",
        )


def test_transitive_room_and_camera_leakage_never_crosses_folds(tmp_path):
    def join(sources):
        for index in (0, 1):
            sources[index]["camera_ids"] = ["shared-camera"]
            keys = [key for key in sources[index]["leakage_keys"]
                    if not key.startswith("camera:")]
            sources[index]["leakage_keys"] = sorted(
                keys + ["camera:shared-camera", "room:shared-room"]
            )

    manifest = _load(tmp_path, mutate=join)
    assert len(manifest.groups) == 11
    shared = next(group for group in manifest.groups if "room:shared-room" in group.leakage_keys)
    assert shared.clip_ids == ("clip-00", "clip-01")
    _, private, _ = build_reservation(
        manifest, b"reservation-secret-is-at-least-32-bytes",
        reserved_at_utc="2026-08-26T11:00:00Z",
    )
    mapping = {clip: group["role"] for group in private["groups"]
               for clip in group["clip_ids"]}
    assert mapping["clip-00"] == mapping["clip-01"]


def test_reservation_is_deterministic_and_blinds_locked_membership(tmp_path):
    manifest = _load(tmp_path)
    kwargs = {
        "reserved_at_utc": "2026-08-26T11:00:00Z",
        "challenge_fraction": 0.25,
        "locked_holdout_fraction": 0.25,
    }
    secret = b"reservation-secret-is-at-least-32-bytes"
    first = build_reservation(manifest, secret, **kwargs)
    second = build_reservation(manifest, secret, **kwargs)
    assert first == second
    engineering, private, receipt = first
    assert receipt["counts"]["locked_holdout"]["source_groups"] == 3
    assert all(source["reserved_role"] != "locked_holdout"
               for source in engineering["sources"])

    locked_values: set[str] = set()
    for group in private["groups"]:
        if group["role"] == "locked_holdout":
            locked_values.add(group["source_group_id"])
            locked_values.update(group["clip_ids"])
            locked_values.update(group["content_sha256"])
    ordinary = json.dumps([engineering, receipt], sort_keys=True)
    assert locked_values
    assert all(value not in ordinary for value in locked_values)
    assert "groups" not in receipt

    changed = build_reservation(manifest, b"a-different-secret-that-is-32-bytes!", **kwargs)
    assert changed[2]["reservation_id"] != receipt["reservation_id"]


def test_operator_audit_detects_tampering_without_public_membership(tmp_path):
    manifest = _load(tmp_path)
    secret = b"reservation-secret-is-at-least-32-bytes"
    documents = build_reservation(
        manifest, secret, reserved_at_utc="2026-08-26T11:00:00Z",
    )
    result = audit_reservation(manifest, secret, *documents)
    assert result["status"] == "clean"
    assert result["locked_membership_exposed"] is False

    engineering, private, receipt = copy.deepcopy(documents)
    engineering["reservation_id"] += "-tampered"
    with pytest.raises(ManifestError, match="deterministic reconstruction"):
        audit_reservation(manifest, secret, engineering, private, receipt)


def test_outputs_are_new_only_and_private_mapping_is_mode_0600(tmp_path):
    manifest = _load(tmp_path)
    documents = build_reservation(
        manifest, b"reservation-secret-is-at-least-32-bytes",
        reserved_at_utc="2026-08-26T11:00:00Z",
    )
    engineering = tmp_path / "release" / "engineering.json"
    private = tmp_path / "private" / "split.json"
    receipt = tmp_path / "release" / "receipt.json"
    write_reservation_documents(engineering, private, receipt, documents)
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert stat.S_IMODE(engineering.stat().st_mode) == 0o644
    with pytest.raises(ManifestError, match="refusing to overwrite"):
        write_reservation_documents(engineering, private, receipt, documents)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda sources: sources[0].update({"prediction_trace": "forbidden"}),
         "unknown fields"),
        (lambda sources: sources[0]["rights"].update({"status": "unknown"}),
         "status must be 'verified'"),
        (lambda sources: sources[0]["rights"].update({"allowed_uses": ["view-only"]}),
         "must explicitly include"),
        (lambda sources: sources[0]["media"].update({"sha256": "0" * 64}),
         "sha256 mismatch"),
        (lambda sources: sources[0]["media"].update({"path": "../private/clip.bin"}),
         "normalized relative POSIX path"),
        (lambda sources: sources[0]["gold_evidence"].update({"pass_ids": ["only-pass"]}),
         "at least two unique pass IDs"),
    ],
)
def test_manifest_fails_closed_on_untrusted_or_unverified_fields(
    tmp_path, mutation, message,
):
    path = _write_manifest(tmp_path, mutate=lambda sources: mutation(sources))
    with pytest.raises(ManifestError, match=message):
        load_source_manifest(path)


def test_actor_and_camera_ids_must_match_leakage_keys(tmp_path):
    def mutate(sources):
        sources[0]["camera_ids"] = ["different-camera"]

    with pytest.raises(ManifestError, match="camera_ids must exactly match"):
        load_source_manifest(_write_manifest(tmp_path, mutate=mutate))


def test_private_boundary_rejects_workspace_and_open_directory(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    candidate = workspace / "secret"
    with pytest.raises(ManifestError, match="outside shared workspace"):
        enforce_operator_private_path(
            candidate, protected_roots=[workspace], label="secret", require_existing=False,
        )

    open_dir = tmp_path / "open"
    open_dir.mkdir(mode=0o755)
    with pytest.raises(ManifestError, match="deny group/other access"):
        enforce_operator_private_path(
            open_dir / "mapping.json", protected_roots=[workspace],
            label="mapping", require_existing=False,
        )


def test_schema_files_are_valid_json_and_fail_closed():
    schema_dir = Path(__file__).resolve().parents[1] / "corpus" / "schemas"
    source_schema = json.loads((schema_dir / "source-manifest-v1.schema.json").read_text())
    receipt_schema = json.loads((schema_dir / "split-receipt-v1.schema.json").read_text())
    engineering_schema = json.loads(
        (schema_dir / "engineering-corpus-v1.schema.json").read_text()
    )
    private_schema = json.loads((schema_dir / "private-split-v1.schema.json").read_text())
    assert source_schema["additionalProperties"] is False
    assert source_schema["$defs"]["source"]["additionalProperties"] is False
    assert receipt_schema["additionalProperties"] is False
    assert engineering_schema["additionalProperties"] is False
    assert private_schema["additionalProperties"] is False
