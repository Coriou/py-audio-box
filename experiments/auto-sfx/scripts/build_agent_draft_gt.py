#!/usr/bin/env python3
"""Convert a reviewed raw-frame adjudication CSV into schema-v2 draft GT.

This is deliberately mechanical: contact decisions live in the CSV, and this
tool only converts displayed-frame ordinals to source-time fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--n-frames", required=True, type=int)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    contacts = []
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            preferred = int(row["preferred_frame"])
            earliest = int(row["earliest_frame"])
            latest = int(row["latest_frame"])
            if not 0 <= earliest <= preferred <= latest < args.n_frames:
                raise SystemExit(f"invalid frame window: {row}")
            foot = row.get("foot") or row.get("anatomical_side") or "uncertain"
            visibility = row.get("visibility") or "partial"
            if visibility not in {"clear", "partial", "occluded"}:
                visibility = (
                    "clear" if row.get("confidence") == "high" else "partial"
                )
            raw_viewpoint = row.get("viewpoint") or "uncertain"
            viewpoint = raw_viewpoint
            if viewpoint not in {
                "sagittal", "diagonal", "frontal", "rear", "uncertain"
            }:
                # Schema v2 has no overhead enum.  Preserve the raw label in
                # notes and use the honest non-inferential bucket rather than
                # pretending the view was sagittal.
                viewpoint = "uncertain"
            notes = row["notes"]
            if viewpoint != raw_viewpoint:
                notes = f"{notes}; raw-viewpoint-label={raw_viewpoint}"
            contacts.append({
                "foot": foot,
                "earliest_pts_us": int(round(earliest / args.fps * 1e6)),
                "preferred_pts_us": int(round(preferred / args.fps * 1e6)),
                "latest_pts_us": int(round(latest / args.fps * 1e6)),
                "visibility": visibility,
                "viewpoint": viewpoint,
                "notes": notes,
            })
    contacts.sort(key=lambda c: c["preferred_pts_us"])
    document = {
        "schema_version": 2,
        "clip": args.clip,
        "range_s": {"start": 0.0, "end": (args.n_frames - 1) / args.fps},
        "annotator": args.annotator,
        "fps_display": args.fps,
        "contacts": contacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", "utf-8")
    print(f"wrote {args.out}: {len(contacts)} agent-draft contacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
