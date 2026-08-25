#!/usr/bin/env python3
"""Prepare standardized silent h264 inputs for the fal-sfx-bench clip set.

Builds the five benchmark inputs described in the shared contracts spec into a
run directory (`--out-dir`, typically `work/sfx-bench/<run_id>/inputs`) and
records an auditable `inputs-index.json` next to them: source and output
sha256, the exact ffmpeg argv used, trim windows, and full ffprobe JSON.

Every output is h264 yuv420p, faststart, and silent (no audio track), so the
exact same bytes can be submitted to every model. Byte-identical re-runs are
not guaranteed (encoder timestamps), but the recorded argv fully specifies
each build. Existing outputs are reused unless `--force` is given.

Source notes:
- walk2 uses `clips/walk2.webm`. A stale, corrupt sibling .mp4 sits next to it
  in that directory; it is damaged and deliberately never used as a source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(relative: str) -> Path:
    return REPO_ROOT / relative

# Encode settings shared by every re-encoded clip (see spec: h264 yuv420p,
# crf 20 preset slow, silent, faststart).
ENCODE_ARGS = [
    "-c:v",
    "libx264",
    "-crf",
    "20",
    "-preset",
    "slow",
    "-pix_fmt",
    "yuv420p",
    "-an",
    "-movflags",
    "+faststart",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:stream=index,codec_type,codec_name,"
        "width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class ClipSpec:
    """One benchmark input to build."""

    key: str
    source: str  # relative to repo root
    trim_s: tuple[float, float] | None  # inclusive window (start, end)
    video_filter: str | None  # -vf value, or None to keep source geometry
    operation: str = "ffmpeg"  # "ffmpeg" | "verbatim_copy"


CLIPS: list[ClipSpec] = [
    ClipSpec(
        key="walk1",
        source="work/auto-sfx/clips/walk1.mp4",
        trim_s=(10.0, 19.4),
        video_filter="scale=-2:480",
    ),
    ClipSpec(
        key="walk2",
        # NOTE: the sibling .mp4 of this source is corrupt; use the webm.
        source="work/auto-sfx/clips/walk2.webm",
        trim_s=(18.0, 27.4),
        video_filter="scale=480:-2",  # portrait 720x1280 -> short side 480
    ),
    ClipSpec(
        key="man",
        source="work/auto-sfx/clips/man-walk-no-audio.mp4",
        trim_s=None,  # native 854x480, already silent: plain h264 re-encode
        video_filter=None,
    ),
    ClipSpec(
        key="ocean-long",
        source="work/auto-sfx/clips/3571264-hd_1920_1080_30fps.mp4",
        trim_s=(10.0, 20.0),
        video_filter=None,  # keep 1280x720; -an strips the source aac track
    ),
    ClipSpec(
        key="ocean-short",
        source="work/auto-sfx/fal-hunyuan-eval-20260825/inputs/ocean-9.4s.mp4",
        trim_s=None,
        video_filter=None,
        operation="verbatim_copy",  # prior eval's proven input, fresh sha
    ),
]


def build_ffmpeg_argv(spec: ClipSpec, destination: Path) -> list[str]:
    """Fully specified ffmpeg invocation for one clip."""
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if spec.trim_s is not None:
        start_s, end_s = spec.trim_s
        argv += ["-ss", f"{start_s:.1f}", "-i", str(spec.source), "-t", f"{end_s - start_s:.1f}"]
    else:
        argv += ["-i", str(spec.source)]
    if spec.video_filter is not None:
        argv += ["-vf", spec.video_filter]
    argv += ENCODE_ARGS
    argv.append(str(destination))
    return argv


def output_duration_s(probe: dict[str, Any]) -> float | None:
    duration = probe.get("format", {}).get("duration")
    return None if duration is None else float(duration)


def prepare_clip(
    spec: ClipSpec,
    out_dir: Path,
    force: bool,
) -> dict[str, Any]:
    """Encode (or copy) one clip and return its inputs-index.json entry."""
    source_path = resolve_repo_path(spec.source)
    output_path = out_dir / f"{spec.key}.mp4"
    existed = output_path.exists()

    if spec.operation == "verbatim_copy":
        command_used: list[str] | None = None
        if force or not existed:
            shutil.copyfile(source_path, output_path)
            verb = "rebuilt" if existed else "built"
            print(f"[{spec.key}] {verb}: copied {source_path} -> {output_path}")
        else:
            print(f"[{spec.key}] exists, skipping copy (use --force to rebuild)")
    else:
        command_used = build_ffmpeg_argv(spec, output_path)
        if force or not existed:
            subprocess.run(command_used, check=True)
            verb = "re-encoded" if existed else "encoded"
            print(f"[{spec.key}] {verb}: {output_path}")
        else:
            print(f"[{spec.key}] exists, skipping encode (use --force to rebuild)")

    if not output_path.exists():
        raise RuntimeError(f"[{spec.key}] expected output was not created: {output_path}")

    probe = ffprobe(output_path)
    duration = output_duration_s(probe)
    trim_record: list[float] | None = (
        None if spec.trim_s is None else [spec.trim_s[0], spec.trim_s[1]]
    )
    return {
        "key": spec.key,
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "trim": trim_record,
        "ffmpeg_command": command_used,
        "operation": spec.operation,
        "output_path": str(output_path),
        "sha256": sha256(output_path),
        "probe": probe,
        "duration_s": duration,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standardized silent h264 inputs for fal-sfx-bench."
    )
    parser.add_argument("--out-dir", required=True, help="directory for built clips + inputs-index.json")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated clip keys to build (default: all of %s)" % ",".join(c.key for c in CLIPS),
    )
    parser.add_argument("--force", action="store_true", help="re-encode even if the output already exists")
    return parser.parse_args()


def select_clips(only: str | None) -> list[ClipSpec]:
    if only is None:
        return CLIPS
    wanted = [key.strip() for key in only.split(",") if key.strip()]
    known = {clip.key for clip in CLIPS}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise SystemExit(f"unknown clip key(s): {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    selected = [clip for clip in CLIPS if clip.key in set(wanted)]
    # Preserve canonical ordering even when --only reorders keys.
    return selected


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        str(resolve_repo_path(spec.source))
        for spec in select_clips(args.only)
        if not resolve_repo_path(spec.source).exists()
    ]
    if missing:
        print("missing source clip(s):", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    entries = [prepare_clip(spec, out_dir, args.force) for spec in select_clips(args.only)]
    index = {
        "schema_version": 1,
        "created_at": utc_now(),
        "out_dir": str(out_dir),
        "clips": entries,
    }
    atomic_json(out_dir / "inputs-index.json", index)

    for entry in entries:
        duration = entry["duration_s"]
        shown = "unknown" if duration is None else f"{duration:.2f}s"
        print(f"[{entry['key']}] sha256={entry['sha256']} duration={shown}")
    print(f"wrote {out_dir / 'inputs-index.json'} ({len(entries)} clip(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
