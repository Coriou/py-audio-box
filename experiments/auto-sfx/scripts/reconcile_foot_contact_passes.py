#!/usr/bin/env python3
"""Freeze, compare, append decisions to, and seal foot-contact operator passes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_EXPERIMENT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.foot_contact_reconcile import (  # noqa: E402
    append_adjudication_decision,
    compare_passes,
    finalize_pass,
    seal_adjudication,
    verify_adjudication_package,
    verify_finalized_pass,
)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(),
        help="repository root (default: current working directory)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    finalize = commands.add_parser("finalize", help="hash-lock one complete pass")
    finalize.add_argument("pass_dir", type=Path)
    finalize.add_argument("--finalized-at-utc", required=True)

    verify = commands.add_parser("verify", help="verify a finalized pass has not drifted")
    verify.add_argument("pass_dir", type=Path)

    compare = commands.add_parser(
        "compare", help="create a prediction-free two-pass disagreement package"
    )
    compare.add_argument("--pass-a", type=Path, required=True)
    compare.add_argument("--pass-b", type=Path, required=True)
    compare.add_argument("--out-dir", type=Path, required=True)
    compare.add_argument("--compared-at-utc", required=True)
    compare.add_argument("--max-pair-distance-frames", type=int, default=6)

    append = commands.add_parser(
        "append", help="append one resolved/unresolved adjudication decision JSON"
    )
    append.add_argument("--package", type=Path, required=True)
    append.add_argument("--decision", type=Path, required=True)

    verify_package = commands.add_parser(
        "verify-package", help="verify pass locks, comparison, and decision hash chain"
    )
    verify_package.add_argument("--package", type=Path, required=True)

    seal = commands.add_parser(
        "seal", help="materialize canonical adjudicated annotations after all rows resolve"
    )
    seal.add_argument("--package", type=Path, required=True)
    seal.add_argument("--out-dir", type=Path, required=True)
    seal.add_argument("--sealed-at-utc", required=True)
    seal.add_argument("--adjudicator", required=True)
    seal.add_argument("--pass-id", default="adjudicated-v1")

    args = parser.parse_args()
    root = args.repo_root.resolve()
    try:
        if args.command == "finalize":
            result = finalize_pass(
                args.pass_dir, repo_root=root, finalized_at_utc=args.finalized_at_utc
            )
        elif args.command == "verify":
            result = verify_finalized_pass(args.pass_dir, repo_root=root)
        elif args.command == "compare":
            result = compare_passes(
                args.pass_a,
                args.pass_b,
                out_dir=args.out_dir,
                repo_root=root,
                compared_at_utc=args.compared_at_utc,
                max_pair_distance_frames=args.max_pair_distance_frames,
            )
        elif args.command == "append":
            try:
                decision = json.loads(args.decision.read_text(encoding="utf-8-sig"))
            except OSError as exc:
                raise ValueError(f"cannot read decision {args.decision}: {exc}") from None
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid decision JSON {args.decision}: {exc}") from None
            result = append_adjudication_decision(
                args.package, decision, repo_root=root
            )
        elif args.command == "verify-package":
            result = verify_adjudication_package(args.package, repo_root=root)
        else:
            result = seal_adjudication(
                args.package,
                out_dir=args.out_dir,
                repo_root=root,
                sealed_at_utc=args.sealed_at_utc,
                adjudicator=args.adjudicator,
                pass_id=args.pass_id,
            )
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
