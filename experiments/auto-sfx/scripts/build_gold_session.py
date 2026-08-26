#!/usr/bin/env python3
"""Scaffold a human-gold annotation session for gold_annotator.html.

Phase-2 Q2 prerequisite (docs/sfx-bank-foley-phase2-eval-brief.md §3.2).
Emits <out-dir>/session.json + README.md. Clip files stay where they are; the
session records repo-root-relative paths (served via `python3 -m http.server`
from the repo root).

Known corruption handling: walk2.mp4 is corrupt — when both X.webm and X.mp4
exist, .webm wins and the choice is recorded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BENCH_INPUTS = Path("work/sfx-bench/20260825-165347/inputs")
DEMO_CLIPS = ["walk1", "man", "ocean-short"]


def ffprobe_meta(path: Path, fps_default: float) -> tuple[float, float | None]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return fps_default, None
    fps = fps_default
    dur: float | None = None
    for st in data.get("streams", []):
        if st.get("codec_type") == "video":
            r = st.get("r_frame_rate") or st.get("avg_frame_rate")
            if r and "/" in r:
                num, den = r.split("/")
                if float(den):
                    fps = round(float(num) / float(den), 4)
            break
    fmt = data.get("format", {})
    if fmt.get("duration"):
        dur = float(fmt["duration"])
    return fps, dur


def resolve_clip(stem_or_path: str) -> tuple[Path | None, str]:
    """Resolve to an existing file; prefer .webm sibling over possibly-corrupt .mp4."""
    p = Path(stem_or_path)
    if p.is_file():
        if p.suffix == ".mp4" and p.with_suffix(".webm").is_file():
            return p.with_suffix(".webm"), f"chose {p.with_suffix('.webm').name} over {p.name} (both exist)"
        return p, ""
    for ext in (".webm", ".mp4"):
        cand = BENCH_INPUTS / f"{stem_or_path}{ext}"
        if cand.is_file():
            note = ""
            if ext == ".mp4" and cand.with_suffix(".webm").is_file():
                continue  # webm listed first anyway
            return cand, note
    return None, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path,
                        help="manifest JSON list of {path|stem, family_hint?}")
    parser.add_argument("--demo", action="store_true",
                        help="build a demo session over bench clips walk1/man/ocean-short")
    parser.add_argument("--fps-default", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.demo:
        entries = [{"stem": s} for s in DEMO_CLIPS]
    elif args.clips:
        raw = json.loads(args.clips.read_text())
        entries = raw if isinstance(raw, list) else raw.get("clips", [])
    else:
        parser.error("need --clips MANIFEST or --demo")

    session_dir = args.out_dir.resolve()
    annotations = session_dir / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)

    clips_out = []
    notes: list[str] = []
    for e in entries:
        ref = e.get("path") or e.get("file") or e.get("stem")
        path, note = resolve_clip(str(ref))
        if path is None or not path.is_file():
            print(f"warning: clip not found, skipping: {ref}")
            continue
        if note:
            notes.append(f"{path.name}: {note}")
        rel = path.resolve().relative_to(Path.cwd().resolve()) if path.is_absolute() else path
        fps, dur = ffprobe_meta(path, args.fps_default)
        clip_id = path.stem
        clips_out.append({
            "clip_id": clip_id,
            "file": str(rel),
            "fps": fps,
            "duration_s": dur,
            "family_hint": e.get("family_hint"),
        })

    session = {
        "schema_version": 1,
        "kind": "human-gold-session",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": "experiments/auto-sfx/gold_annotator.html",
        "notes": notes,
        "clips": clips_out,
    }
    (session_dir / "session.json").write_text(json.dumps(session, indent=2))

    readme = f"""# Human-gold annotation session ({len(clips_out)} clips)

Serve from the REPO ROOT so both this page and the video files resolve:

    cd <repo-root> && python3 -m http.server 8765

Open: http://localhost:8765/experiments/auto-sfx/gold_annotator.html?session={args.out_dir}/session.json

Steps: enter your name → pick a clip → step frames (`,`/`.` ±1 frame, arrow keys,
space play/pause, `J`/`K`/`L` speeds) → Add event → capture start/contact/end
frames with buttons → save. Every annotator downloads their own GT JSON into
this directory (`annotations/<clip>.<annotator>.gt.json`). Disagreements are
kept — never merge by hand.

Frames of reference per clip are recorded in session.json (fps probed via ffprobe).
"""
    (session_dir / "README.md").write_text(readme)
    for n in notes:
        print(f"note: {n}")
    print(f"session ready: {session_dir}/session.json — {len(clips_out)} clips")
    return 0 if clips_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
