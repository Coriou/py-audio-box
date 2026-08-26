#!/usr/bin/env python3
"""Build a prediction-free, source-PTS foot-contact annotation session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_EXPERIMENT_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx.foot_contact_session import (  # noqa: E402
    build_session,
    load_clip_manifest,
    session_http_url,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips", type=Path, required=True, help="strict source-only clip manifest"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--pass-id", required=True)
    parser.add_argument(
        "--order-seed",
        required=True,
        help="recorded seed used to randomize clip order for this pass",
    )
    args = parser.parse_args()
    repo_root = Path.cwd().resolve()
    try:
        session = build_session(
            entries=load_clip_manifest(args.clips),
            out_dir=args.out_dir,
            repo_root=repo_root,
            session_id=args.session_id,
            pass_id=args.pass_id,
            order_seed=args.order_seed,
        )
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"session ready: {args.out_dir}/session.json "
        f"({len(session['clips'])} clips, pass {args.pass_id})"
    )
    print(f"open from repo-root HTTP server: {session_http_url(args.out_dir, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
