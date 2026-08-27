"""Two-pass finalization, disagreement, and append-only adjudication tests."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_EXPERIMENT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.foot_contact_reconcile import (  # noqa: E402
    ADJUDICATION_LOG,
    append_adjudication_decision,
    compare_passes,
    finalize_pass,
    seal_adjudication,
    verify_adjudication_package,
    verify_finalized_pass,
)
from auto_sfx.foot_contact_session import build_session  # noqa: E402
from auto_sfx.foot_contact_truth import (  # noqa: E402
    ANNOTATION_KIND,
    TOOL,
    TOOL_VERSION,
    load_annotation,
    load_pts_map,
    write_annotation,
)

PTS = [0, 40_000, 90_000, 125_000, 180_000, 220_000, 280_000, 315_000]
UTC = "2026-08-27T12:00:00Z"


def _contact(
    contact_id: str = "c0001",
    *,
    preferred: int = 2,
    foot: str = "left",
    visibility: str = "clear",
    notes: str = "source-only observation",
) -> dict:
    return {
        "contact_id": contact_id,
        "foot": foot,
        "earliest_pts_us": PTS[max(0, preferred - 1)],
        "preferred_pts_us": PTS[preferred],
        "latest_pts_us": PTS[min(len(PTS) - 1, preferred + 1)],
        "display_frames": {
            "earliest": max(0, preferred - 1),
            "preferred": preferred,
            "latest": min(len(PTS) - 1, preferred + 1),
        },
        "visibility": visibility,
        "viewpoint": "diagonal",
        "actor_id": "lead",
        "observability": "observable" if visibility == "clear" else "partial",
        "cropping": "none",
        "audible_onset": None,
        "notes": notes,
    }


def _build_pass(
    repo: Path,
    pass_id: str,
    annotator: str,
    contacts_by_clip: dict[str, list[dict]],
    *,
    source_suffix: str = "",
    order_seed: str | None = None,
) -> Path:
    entries = []
    for clip_id in sorted(contacts_by_clip):
        source = repo / "work" / "sources" / f"{clip_id}{source_suffix}.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            source.write_bytes(f"fixture-{clip_id}{source_suffix}".encode())
        entries.append(
            {
                "path": source.relative_to(repo).as_posix(),
                "clip_id": clip_id,
                "actor_ids": ["lead"],
            }
        )
    directory = repo / "work" / "gold" / pass_id
    session = build_session(
        entries=entries,
        out_dir=directory,
        repo_root=repo,
        session_id="challenge-01",
        pass_id=pass_id,
        order_seed=order_seed or f"seed-{pass_id}",
        probe=lambda path: (25.0, PTS),
    )
    for clip in session["clips"]:
        pts_map = load_pts_map(repo / clip["pts_map_file"])
        annotation = {
            "schema_version": 2,
            "kind": ANNOTATION_KIND,
            "clip": clip["clip_id"],
            "range_s": clip["range_s"],
            "fps_display": clip["nominal_fps"],
            "annotator": annotator,
            "pass_id": pass_id,
            "tool": TOOL,
            "tool_version": TOOL_VERSION,
            "session_id": session["session_id"],
            "source": clip["source"],
            "contacts": contacts_by_clip[clip["clip_id"]],
        }
        write_annotation(
            directory / "annotations" / f"{clip['clip_id']}.{annotator}.{pass_id}.gt.json",
            annotation,
            pts_map=pts_map,
        )
    return directory


def _finalize_pair(
    repo: Path,
    contacts_a: dict[str, list[dict]],
    contacts_b: dict[str, list[dict]],
) -> tuple[Path, Path]:
    pass_a = _build_pass(repo, "pass-a", "operator-a", contacts_a)
    pass_b = _build_pass(repo, "pass-b", "operator-b", contacts_b)
    finalize_pass(pass_a, repo_root=repo, finalized_at_utc=UTC)
    finalize_pass(pass_b, repo_root=repo, finalized_at_utc=UTC)
    return pass_a, pass_b


def _decision(
    row_id: str,
    *,
    decision_id: str = "decision-1",
    resolution: str = "resolved",
    contacts: list[dict] | None = None,
    supersedes: str | None = None,
) -> dict:
    return {
        "decision_id": decision_id,
        "row_id": row_id,
        "supersedes": supersedes,
        "decided_at_utc": UTC,
        "adjudicator": "operator-c",
        "resolution": resolution,
        "contacts": contacts or [],
        "notes": "reviewed both source-only passes",
    }


def test_finalize_requires_complete_canonical_pass_and_detects_later_drift(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    directory = _build_pass(repo, "pass-a", "operator-a", {"walk": [_contact()]})
    document = finalize_pass(directory, repo_root=repo, finalized_at_utc=UTC)
    assert document["clip_count"] == 1
    assert document["contact_count"] == 1
    assert verify_finalized_pass(directory, repo_root=repo) == document
    with pytest.raises(ValueError, match="overwrite"):
        finalize_pass(directory, repo_root=repo, finalized_at_utc=UTC)

    annotation_path = next((directory / "annotations").glob("*.gt.json"))
    pts_map = load_pts_map(directory / "pts" / "walk.source-pts.json")
    annotation = load_annotation(annotation_path, pts_map=pts_map)
    annotation["contacts"][0]["notes"] = "changed after finalization"
    write_annotation(annotation_path, annotation, pts_map=pts_map)
    with pytest.raises(ValueError, match="no longer matches"):
        verify_finalized_pass(directory, repo_root=repo)


def test_finalize_rejects_missing_clip_export_and_unknown_actor(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    directory = _build_pass(
        repo,
        "pass-a",
        "operator-a",
        {"walk-a": [_contact()], "walk-b": []},
    )
    next((directory / "annotations").glob("walk-b*.gt.json")).unlink()
    with pytest.raises(ValueError, match="missing clips"):
        finalize_pass(directory, repo_root=repo, finalized_at_utc=UTC)

    directory = _build_pass(
        repo,
        "pass-c",
        "operator-a",
        {"walk-c": [_contact()]},
    )
    path = next((directory / "annotations").glob("*.gt.json"))
    pts_map = load_pts_map(directory / "pts" / "walk-c.source-pts.json")
    annotation = load_annotation(path, pts_map=pts_map)
    annotation["contacts"][0]["actor_id"] = "not-declared"
    write_annotation(path, annotation, pts_map=pts_map)
    with pytest.raises(ValueError, match="unknown actor IDs"):
        finalize_pass(directory, repo_root=repo, finalized_at_utc=UTC)


def test_compare_aligns_vfr_contacts_and_surfaces_timing_and_side_disagreement(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    pass_a, pass_b = _finalize_pair(
        repo,
        {"walk": [_contact(preferred=2, foot="left")]},
        {"walk": [_contact(preferred=3, foot="right")]},
    )
    package = repo / "work" / "gold" / "comparison"
    comparison = compare_passes(
        pass_a,
        pass_b,
        out_dir=package,
        repo_root=repo,
        compared_at_utc=UTC,
        max_pair_distance_frames=2,
    )
    assert comparison["policy"]["prediction_fields_loaded"] is False
    assert comparison["summary"]["rows_requiring_adjudication"] == 1
    row = comparison["clips"][0]["rows"][0]
    assert row["kind"] == "disagreement"
    assert {"foot", "preferred_pts_us", "display_frames"} <= set(row["differences"])
    assert (package / ADJUDICATION_LOG).read_bytes() == b""


def test_compare_requires_distinct_order_seeds_for_multiple_clips(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    contacts = {"walk-a": [_contact()], "walk-b": []}
    pass_a = _build_pass(repo, "pass-a", "operator-a", contacts, order_seed="same")
    pass_b = _build_pass(repo, "pass-b", "operator-b", contacts, order_seed="same")
    finalize_pass(pass_a, repo_root=repo, finalized_at_utc=UTC)
    finalize_pass(pass_b, repo_root=repo, finalized_at_utc=UTC)
    with pytest.raises(ValueError, match="distinct recorded order seeds"):
        compare_passes(
            pass_a,
            pass_b,
            out_dir=repo / "work" / "comparison",
            repo_root=repo,
            compared_at_utc=UTC,
        )


def test_compare_rejects_distinct_seeds_that_realize_the_same_clip_order(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    clips = ["walk-a", "walk-b", "walk-c"]
    seen: dict[tuple[str, ...], str] = {}
    pair: tuple[str, str] | None = None
    for index in range(100):
        seed = f"seed-{index}"
        order = list(clips)
        random.Random(seed).shuffle(order)
        previous = seen.get(tuple(order))
        if previous is not None:
            pair = previous, seed
            break
        seen[tuple(order)] = seed
    assert pair is not None
    contacts = {clip_id: [] for clip_id in clips}
    pass_a = _build_pass(repo, "pass-a", "operator-a", contacts, order_seed=pair[0])
    pass_b = _build_pass(repo, "pass-b", "operator-b", contacts, order_seed=pair[1])
    finalize_pass(pass_a, repo_root=repo, finalized_at_utc=UTC)
    finalize_pass(pass_b, repo_root=repo, finalized_at_utc=UTC)
    with pytest.raises(ValueError, match="different realized clip orders"):
        compare_passes(
            pass_a,
            pass_b,
            out_dir=repo / "work" / "comparison",
            repo_root=repo,
            compared_at_utc=UTC,
        )


def test_package_verification_recomputes_comparison_from_frozen_passes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    pass_a, pass_b = _finalize_pair(
        repo,
        {"walk": [_contact()]},
        {"walk": [_contact()]},
    )
    package = repo / "work" / "gold" / "comparison"
    compare_passes(
        pass_a, pass_b, out_dir=package, repo_root=repo, compared_at_utc=UTC
    )
    path = package / "comparison.json"
    document = json.loads(path.read_text())
    document["summary"]["contacts_a"] = 999
    path.write_text(json.dumps(document, indent=2) + "\n")
    with pytest.raises(ValueError, match="no longer matches"):
        verify_adjudication_package(package, repo_root=repo)


def test_append_revision_is_hash_chained_and_seal_materializes_resolved_window(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    pass_a, pass_b = _finalize_pair(
        repo,
        {"walk": [_contact(preferred=2, foot="left")]},
        {"walk": [_contact(preferred=3, foot="right")]},
    )
    package = repo / "work" / "gold" / "comparison"
    comparison = compare_passes(
        pass_a, pass_b, out_dir=package, repo_root=repo, compared_at_utc=UTC
    )
    row = comparison["clips"][0]["rows"][0]
    append_adjudication_decision(
        package,
        _decision(row["row_id"], resolution="unresolved"),
        repo_root=repo,
    )
    assert verify_adjudication_package(package, repo_root=repo)["ready_to_seal"] is False

    chosen = dict(row["pass_b"])
    chosen["contact_id"] = "gold-0001"
    append_adjudication_decision(
        package,
        _decision(
            row["row_id"],
            decision_id="decision-2",
            contacts=[chosen],
            supersedes="decision-1",
        ),
        repo_root=repo,
    )
    status = verify_adjudication_package(package, repo_root=repo)
    assert status["record_count"] == 2
    assert status["active_decision_count"] == 1
    assert status["ready_to_seal"] is True

    out = repo / "work" / "gold" / "sealed"
    receipt = seal_adjudication(
        package,
        out_dir=out,
        repo_root=repo,
        sealed_at_utc=UTC,
        adjudicator="operator-c",
    )
    assert receipt["source_pass_ids"] == ["pass-a", "pass-b"]
    pts_map = load_pts_map(pass_a / "pts" / "walk.source-pts.json")
    gold = load_annotation(next(out.glob("*.gt.json")), pts_map=pts_map)
    assert gold["pass_id"] == "adjudicated-v1"
    assert gold["contacts"][0]["contact_id"] == "gold-0001"
    assert gold["contacts"][0]["display_frames"]["preferred"] == 3


def test_log_tampering_breaks_chain(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    pass_a, pass_b = _finalize_pair(
        repo,
        {"walk": [_contact(preferred=2)]},
        {"walk": []},
    )
    package = repo / "work" / "gold" / "comparison"
    comparison = compare_passes(
        pass_a, pass_b, out_dir=package, repo_root=repo, compared_at_utc=UTC
    )
    row_id = comparison["clips"][0]["rows"][0]["row_id"]
    append_adjudication_decision(
        package, _decision(row_id, resolution="unresolved"), repo_root=repo
    )
    append_adjudication_decision(
        package,
        _decision(
            row_id,
            decision_id="decision-2",
            contacts=[_contact("gold-1")],
            supersedes="decision-1",
        ),
        repo_root=repo,
    )
    log = package / ADJUDICATION_LOG
    lines = log.read_text().splitlines()
    first = json.loads(lines[0])
    first["decision"]["notes"] = "tampered"
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="breaks the hash chain"):
        verify_adjudication_package(package, repo_root=repo)


def test_exact_agreements_can_seal_without_adjudication_rows(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    # Notes and pass-local contact IDs may differ without changing the physical mark.
    pass_a, pass_b = _finalize_pair(
        repo,
        {"walk": [_contact("a-1", notes="pass A note")]},
        {"walk": [_contact("b-9", notes="pass B note")]},
    )
    package = repo / "work" / "gold" / "comparison"
    comparison = compare_passes(
        pass_a, pass_b, out_dir=package, repo_root=repo, compared_at_utc=UTC
    )
    assert comparison["summary"]["exact_agreements"] == 1
    assert comparison["summary"]["rows_requiring_adjudication"] == 0
    assert verify_adjudication_package(package, repo_root=repo)["ready_to_seal"] is True
    receipt = seal_adjudication(
        package,
        out_dir=repo / "work" / "gold" / "sealed",
        repo_root=repo,
        sealed_at_utc=UTC,
        adjudicator="operator-c",
    )
    assert receipt["decision_count"] == 0
