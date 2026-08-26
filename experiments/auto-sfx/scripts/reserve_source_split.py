#!/usr/bin/env python3
"""Validate an Auto-SFX source corpus and reserve blind source-level folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from auto_sfx.corpus_manifest import (
    ManifestError,
    audit_reservation,
    build_reservation,
    enforce_operator_private_path,
    load_reservation_documents,
    load_secret,
    load_source_manifest,
    safe_validation_summary,
    write_reservation_documents,
)


def _protected_roots() -> tuple[Path, ...]:
    script = Path(__file__).resolve()
    roots = [script.parents[1]]  # experiments/auto-sfx, including container /app.
    for candidate in script.parents:
        if (candidate / ".git").exists():
            roots.append(candidate)
            break
    return tuple(roots)


def _json_stdout(document: dict) -> None:
    print(json.dumps(document, sort_keys=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rights/provenance validation and deterministic source-group reservation. "
            "This tool never reads detector output."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate media, rights, grouping, and corpus readiness gates",
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument(
        "--skip-media-hash", action="store_true",
        help="schema-only inspection; output does not claim media hashes were verified",
    )

    reserve = subparsers.add_parser(
        "reserve", help="write a new engineering manifest, private mapping, and public receipt",
    )
    reserve.add_argument("manifest", type=Path)
    reserve.add_argument("--secret-file", type=Path, required=True)
    reserve.add_argument("--reserved-at-utc", required=True,
                         help="explicit audit timestamp, YYYY-MM-DDTHH:MM:SSZ")
    reserve.add_argument("--engineering-output", type=Path, required=True)
    reserve.add_argument("--private-output", type=Path, required=True)
    reserve.add_argument("--receipt-output", type=Path, required=True)
    reserve.add_argument("--challenge-fraction", type=float, default=0.25)
    reserve.add_argument("--locked-holdout-fraction", type=float, default=0.25)

    audit = subparsers.add_parser(
        "audit", help="operator-only deterministic reconstruction and blind-boundary audit",
    )
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--secret-file", type=Path, required=True)
    audit.add_argument("--engineering", type=Path, required=True)
    audit.add_argument("--private", type=Path, required=True)
    audit.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            manifest = load_source_manifest(
                args.manifest, verify_media=not args.skip_media_hash,
            )
            result = safe_validation_summary(manifest)
            _json_stdout(result)
            return 0

        protected_roots = _protected_roots()
        private_manifest_path = enforce_operator_private_path(
            args.manifest,
            protected_roots=protected_roots,
            label="full source manifest",
            require_existing=True,
        )
        secret_path = enforce_operator_private_path(
            args.secret_file,
            protected_roots=protected_roots,
            label="reservation secret",
            require_existing=True,
        )
        manifest = load_source_manifest(private_manifest_path, require_ready=True)
        secret = load_secret(secret_path)
        if args.command == "reserve":
            private_output = enforce_operator_private_path(
                args.private_output,
                protected_roots=protected_roots,
                label="operator-only split mapping",
                require_existing=False,
            )
            documents = build_reservation(
                manifest,
                secret,
                reserved_at_utc=args.reserved_at_utc,
                challenge_fraction=args.challenge_fraction,
                locked_holdout_fraction=args.locked_holdout_fraction,
            )
            write_reservation_documents(
                args.engineering_output, private_output, args.receipt_output, documents,
            )
            receipt = documents[2]
            _json_stdout({
                "status": "reserved",
                "reservation_id": receipt["reservation_id"],
                "counts": receipt["counts"],
                "corpus_gate": receipt["corpus_gate"],
                "private_output_mode": "0600",
                "locked_membership_exposed": False,
            })
            return 0

        private_mapping = enforce_operator_private_path(
            args.private,
            protected_roots=protected_roots,
            label="operator-only split mapping",
            require_existing=True,
        )
        engineering, private, receipt = load_reservation_documents(
            args.engineering, private_mapping, args.receipt,
        )
        _json_stdout(audit_reservation(
            manifest, secret, engineering, private, receipt,
        ))
        return 0
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
