#!/usr/bin/env python3
"""Mechanically shard adjudicated v2 GT by the pre-frozen Phase-2B split.

This script never creates or adjusts a contact mark.  It copies contacts whose
preferred displayed frame falls in each frozen half-open range so Metrics v2
can evaluate only exhaustively annotated pixels and reports can aggregate the
same immutable shards for every ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_json", type=Path)
    parser.add_argument("gt_json", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text("utf-8"))
    guard = max(0, int(split.get("boundary_guard_frames", 0)))
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for gt_path in args.gt_json:
        gt = json.loads(gt_path.read_text("utf-8"))
        clip = gt["clip"]
        spec = split["clips"].get(clip)
        if spec is None:
            raise SystemExit(f"{clip}: absent from {args.split_json}")
        fps = float(gt["fps_display"])
        for fold in ("development", "validation"):
            for number, rng in enumerate(spec[fold], start=1):
                start, end = int(rng["start_frame"]), int(rng["end_frame"])
                eval_start = start + guard
                eval_end = end - guard
                if eval_end <= eval_start:
                    raise SystemExit(
                        f"{clip} {fold} shard {number}: guard removes range"
                    )
                contacts = [
                    c for c in gt["contacts"]
                    if eval_start <= int(round(
                        c["preferred_pts_us"] / 1e6 * fps
                    )) < eval_end
                ]
                doc = {
                    "schema_version": 2,
                    "clip": clip,
                    "range_s": {
                        "start": eval_start / fps,
                        "end": (eval_end - 1) / fps,
                    },
                    "annotator": gt["annotator"] + f"; frozen-{fold}-shard",
                    "fps_display": fps,
                    "contacts": contacts,
                }
                stem = Path(clip).stem
                path = args.out / f"{stem}.{fold}.{number:02d}.gt.json"
                path.write_text(json.dumps(doc, indent=2) + "\n", "utf-8")
                index.append({
                    "clip": clip, "fold": fold, "shard": number,
                    "start_frame": start, "end_frame": end,
                    "evaluation_start_frame": eval_start,
                    "evaluation_end_frame": eval_end,
                    "boundary_guard_frames": guard,
                    "n_contacts": len(contacts), "path": str(path),
                })
    (args.out / "index.json").write_text(
        json.dumps({
            "schema_version": 1,
            "split": str(args.split_json),
            "shards": index,
        }, indent=2) + "\n", "utf-8"
    )
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
