"""Command-line entry point for the auto-sfx POC.

Owns the ``python -m auto_sfx`` surface (the Docker ENTRYPOINT) that wires
the pipeline together::

    ingest/probe -> shot boundaries -> cached pose extraction ->
    contact detection -> artifacts -> annotated diagnostic videos

Subcommands
-----------
``analyze``   one video, one backend/method; writes the full artifact set
              (events.json/csv, diagnostics.json, pose.jsonl,
              resource-report.json, annotated.mp4, marker-preview.mp4).
``compare``   several contact methods over ONE cached pose extraction;
              per-method artifact directories plus compare-summary.json.
``metrics``   score a run directory's events.json against manual ground
              truth using auto_sfx.metrics; merges resource-report values.
``annotate``  frame/PTS neighbourhood printing + PNG extraction around a
              timestamp (or every entry of a ground-truth file).
``selftest``  end-to-end run over the synthetic fixtures with raw metric
              output and no pass/fail gates.

Timing is measured per stage (probe, shots, pose_extract, contacts, render)
with :class:`auto_sfx.resources.ResourceTracker`; ``total`` comes from the
tracker itself. Pose extraction goes through the content-hash cache keyed by
video bytes + backend/model + mode + preprocessing version, so contact
algorithms can be iterated without rerunning inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from auto_sfx.annotation_helper import extract_frames, neighbors, print_table
from auto_sfx.cache import (
    TRACE_FORMAT,
    TRACE_FORMAT_VERSION,
    PoseCache,
    compute_pose_cache_key,
    hash_file,
)
from auto_sfx.contacts import default_config, detect_contacts
from auto_sfx.gt import GroundTruth, load_ground_truth
from auto_sfx.ingest import FrameReader, collect_frame_times, probe_video
from auto_sfx.markers import default_duration_s, synthesize_marker_wav
from auto_sfx.metrics import evaluate, load_annotations
from auto_sfx.pose import mediapipe_backend, rtmpose_backend
from auto_sfx.render import render_annotated
from auto_sfx.resources import ResourceTracker
from auto_sfx.shots import detect_shot_boundaries
from auto_sfx.types import (
    CONTACT_METHODS,
    LEFT,
    POSE_BACKENDS,
    RIGHT,
    RUNNING_MODES,
    RTMPOSE_LICENSE_NOTICE,
    SIDES,
    ContactEvent,
    DiagReason,
    PoseTrace,
    VideoMeta,
    s_to_pts_us,
)

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = "/work/auto-sfx/runs"
DEFAULT_ANNOTATIONS_DIR = "/work/auto-sfx/annotations"
FIXTURES_DIR = Path(
    os.environ.get("AUTO_SFX_FIXTURES_DIR", "/work/auto-sfx/fixtures")
)

EVENTS_SCHEMA_VERSION = 1
COMPARE_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1

#: Fixed column order for events.csv (evidence cell is a JSON string).
EVENTS_CSV_COLUMNS = (
    "id",
    "family",
    "actor_id",
    "foot",
    "kind",
    "source_pts_us",
    "time_s",
    "earliest_s",
    "latest_s",
    "score",
    "observable",
    "detector",
    "evidence",
)

#: Thread-hygiene environment knobs worth recording alongside timings.
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ORT_NUM_THREADS",
    "AUTO_SFX_THREADS",
)

# rtmpose-lightweight has no pre-known model sha256: rtmlib downloads its
# ONNX checkpoints itself at first load.  Cache-key stability matters more
# than the digest value here, so the slot hashes the backend name plus this
# fixed constant instead.  Bump the constant if the rtmlib model selection
# ever changes (that must invalidate cached traces).
_RTMPOSE_MODEL_SHA_SEED = "rtmlib-default"


# ---------------------------------------------------------------------------
# Pose backend dispatch
# ---------------------------------------------------------------------------


def run_pose(
    video_path: str,
    *,
    backend: str,
    mode: str = "video",
    frames: Iterable[dict] | None = None,
) -> PoseTrace:
    """Dispatch pose extraction to the selected backend.

    Kept module-level precisely so tests can monkeypatch a fake backend
    here.  Both real backends accept the PTS-accurate ``frames`` iterable
    produced by :func:`_frame_dicts` (Contract B); supplying it replaces
    only their internal decode source.
    """
    if backend.startswith("mediapipe"):
        return mediapipe_backend.run_pose(
            video_path, backend=backend, mode=mode, frames=frames
        )
    if backend == rtmpose_backend.BACKEND_NAME:
        return rtmpose_backend.run_pose(video_path, mode=mode, frames=frames)
    raise ValueError(
        f"unknown pose backend {backend!r}; expected one of {POSE_BACKENDS}"
    )


def resolve_model_sha256(backend: str) -> str:
    """sha256 identifying the model behind ``backend`` for the cache key."""
    if backend.startswith("mediapipe"):
        # mediapipe-lite/full map onto the lite/full .task variants.
        variant = backend.removeprefix("mediapipe-")
        return hash_file(mediapipe_backend.ensure_model(variant))
    if backend == rtmpose_backend.BACKEND_NAME:
        seed = f"{backend}:{_RTMPOSE_MODEL_SHA_SEED}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()
    raise ValueError(
        f"unknown pose backend {backend!r}; expected one of {POSE_BACKENDS}"
    )


def _frame_dicts(reader: FrameReader) -> Iterator[dict]:
    """Adapt FrameReader ``(bgr, t_s, idx)`` triples to backend frame dicts.

    Timestamps come from ingest.collect_frame_times, paired positionally
    inside the reader, so variable-frame-rate sources keep their true source
    PTS instead of the index/fps approximation the backends' internal
    decoders would apply.  This adaptation is what makes VFR timing correct.
    """
    for bgr, t_s, idx in reader:
        yield {
            "bgr": bgr,
            "t_s": t_s,
            "source_pts_us": s_to_pts_us(t_s),
            "frame_index": idx,
        }


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

_RUN_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]+")
_RUN_ID_MAX_LEN = 120


def sanitize_run_id(raw: str) -> str:
    """Reduce ``raw`` to a filesystem-safe single path component."""

    cleaned = _RUN_ID_UNSAFE.sub("_", str(raw).strip())
    cleaned = cleaned.strip("._")
    return cleaned[:_RUN_ID_MAX_LEN] or "run"


def default_run_id(video: str, backend: str, contact: str) -> str:
    """``<stem>-<backend>-<contact>-<UTC timestamp>`` run identifier."""

    stem = Path(video).stem
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return sanitize_run_id(f"{stem}-{backend}-{contact}-{stamp}")


def parse_config_value(raw: str) -> Any:
    """--config value parsing: int first, then float, else the raw string."""

    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_config_pairs(pairs: Iterable[str]) -> dict[str, Any]:
    """Parse repeated ``KEY=VAL`` strings into a config dict."""

    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--config expects KEY=VAL, got {pair!r}")
        out[key.strip()] = parse_config_value(value)
    return out


def diagnostics_counts(diagnostics: Iterable[dict]) -> dict[str, int]:
    """Count diagnostic codes; ordering is stable (sorted by code)."""

    counts = Counter(d.get("code", "?") for d in diagnostics or [])
    return dict(sorted(counts.items()))


def _csv_cell(value: Any) -> Any:
    """JSON-encode the cells that do not survive CSV as plain strings."""

    if isinstance(value, (bool, dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_json(path: str | Path, payload: Any) -> Path:
    """Pretty JSON with trailing newline; parents created on demand."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def events_document(
    meta: VideoMeta,
    backend: str,
    mode: str,
    contact_method: str,
    events: list[ContactEvent],
    ground_truth: GroundTruth | None = None,
) -> dict:
    """Top-level events.json shape (schema kept deliberately small)."""

    doc = {
        "schema_version": EVENTS_SCHEMA_VERSION,
        "video": meta.to_dict(),
        "backend": backend,
        "mode": mode,
        "contact_method": contact_method,
        "events": [e.to_dict() for e in events],
    }
    if ground_truth is not None:
        # Provenance only; the GT file itself stays the single source of
        # truth and is never copied into the run artifacts.
        doc["ground_truth"] = {
            "path": ground_truth.source_path,
            "clip": ground_truth.clip,
            "n_contacts": len(ground_truth.contacts),
        }
    return doc


def write_events_csv(events: Iterable[Any], path: str | Path) -> Path:
    """Fixed-column events.csv; ``evidence`` serialised as a JSON string."""

    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(EVENTS_CSV_COLUMNS)
        for event in events:
            row = event.to_dict() if hasattr(event, "to_dict") else dict(event)
            writer.writerow([_csv_cell(row.get(col)) for col in EVENTS_CSV_COLUMNS])
    return path


def write_pose_jsonl(trace: PoseTrace, key: str, path: str | Path) -> Path:
    """Mirror the PoseCache on-disk format: header line + one row per frame."""

    header = {
        "format": TRACE_FORMAT,
        "format_version": TRACE_FORMAT_VERSION,
        "key": key,
        "backend": trace.backend,
        "mode": trace.mode,
        "meta": trace.meta.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True) + "\n")
        for row in trace.to_rows():
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def finalize_report(tracker: ResourceTracker, *, cache_hit: bool) -> dict:
    """tracker.report() dict plus cache flag and thread env snapshot."""

    rep = tracker.report().to_dict()
    rep["cache_hit"] = bool(cache_hit)
    rep["threads_env"] = {
        k: os.environ[k] for k in _THREAD_ENV_KEYS if k in os.environ
    }
    return rep


def ensure_pose_lost_diagnostic(result: Any, trace: PoseTrace, meta: VideoMeta) -> bool:
    """Append an honest POSE_LOST interval when NO frame had a usable pose.

    Returns True when the CLI added the entry (contacts may already have
    emitted finer-grained pose_lost intervals of its own).
    """

    if any(not f.lost for f in trace.frames):
        return False
    if any(d.get("code") == DiagReason.POSE_LOST for d in result.diagnostics):
        return False
    t0 = trace.frames[0].t_s if trace.frames else 0.0
    t1 = trace.frames[-1].t_s if trace.frames else float(meta.duration_s or 0.0)
    result.diagnostics.append(
        {
            "code": DiagReason.POSE_LOST,
            "t0_s": round(float(t0), 6),
            "t1_s": round(float(t1), 6),
            "detail": (
                "no usable pose in any of "
                f"{len(trace.frames)} decoded frame(s); event set left empty"
            ),
        }
    )
    return True


# ---------------------------------------------------------------------------
# Shared pipeline pieces
# ---------------------------------------------------------------------------


@dataclass
class TraceSource:
    """Result of a (possibly cached) pose extraction."""

    trace: PoseTrace
    cache_key: str
    cache_hit: bool
    model_sha256: str


def extract_trace_cached(
    video: str,
    backend: str,
    mode: str,
    tracker: ResourceTracker,
    meta: VideoMeta | None = None,
) -> TraceSource:
    """Pose extraction through the content-hash cache (shared by commands).

    On a miss the backend receives frames adapted from
    :class:`auto_sfx.ingest.FrameReader`, whose timestamps come straight
    from ffprobe -- the VFR-correct clock.  The probed
    :class:`auto_sfx.types.VideoMeta` (honest duration/CFR verdict from the
    decoded timeline) replaces the decoder-approximate metadata before the
    trace is cached, so hits and misses see identical shapes.
    """

    model_sha = resolve_model_sha256(backend)
    key = compute_pose_cache_key(video, backend, mode, model_sha)
    cache = PoseCache()  # $AUTO_SFX_CACHE_DIR or /cache/auto-sfx/pose
    with tracker.stage("pose_extract"):
        trace = cache.load_trace(key)
        if trace is not None:
            logger.info("pose cache HIT %s", key[:12])
            return TraceSource(trace=trace, cache_key=key, cache_hit=True,
                               model_sha256=model_sha)
        logger.info("pose cache MISS %s; extracting (%s/%s)", key[:12], backend, mode)
        with FrameReader(video) as reader:
            trace = run_pose(
                video, backend=backend, mode=mode, frames=_frame_dicts(reader)
            )
        if meta is not None:
            trace.meta = meta
        cache.save_trace(trace, key)
    return TraceSource(trace=trace, cache_key=key, cache_hit=False,
                       model_sha256=model_sha)


def build_contact_config(
    method: str, shot_boundaries: list[float], overrides: dict[str, Any]
) -> dict:
    """default_config(method) + shot boundaries + --config passthrough."""

    config = default_config(method)
    config["shot_boundaries"] = list(shot_boundaries)
    config.update(overrides)
    return config


@dataclass
class AnalyzeOutcome:
    """Everything the human summary (and selftest) needs afterwards."""

    run_id: str
    out_dir: Path
    meta: VideoMeta
    events: list[ContactEvent]
    diagnostics_counts: dict[str, int] = field(default_factory=dict)
    cache_hit: bool = False
    report: dict = field(default_factory=dict)
    pose_lost: bool = False
    gt_summary: dict | None = None


def _write_core_artifacts(
    out_dir: Path,
    *,
    meta: VideoMeta,
    backend: str,
    mode: str,
    contact_method: str,
    result: Any,
    trace: PoseTrace,
    cache_key: str,
    notes: list[str],
    ground_truth: GroundTruth | None = None,
) -> None:
    """events/diagnostics/pose artifacts; written BEFORE video rendering."""

    write_json(
        out_dir / "events.json",
        events_document(
            meta, backend, mode, contact_method, result.events,
            ground_truth=ground_truth,
        ),
    )
    write_events_csv(result.events, out_dir / "events.csv")
    write_json(
        out_dir / "diagnostics.json",
        {
            "diagnostics": list(result.diagnostics),
            "video": meta.to_dict(),
            "notes": notes,
        },
    )
    write_pose_jsonl(trace, cache_key, out_dir / "pose.jsonl")



def _render_videos(
    video: str,
    trace: PoseTrace,
    events: list[ContactEvent],
    out_dir: Path,
    meta: VideoMeta,
    *,
    preview: bool = True,
    ground_truth: GroundTruth | None = None,
) -> None:
    """annotated.mp4 (+ marker-preview.mp4) with click markers muxed in.

    marker-preview repeats the render without skeleton clutter (banners,
    foot chips and the speed graph only) so predicted contacts can be
    checked by ear/eye quickly; ``render_annotated`` muxes the click WAV
    itself.

    With ground truth supplied (Phase-2 T15): the click WAV gains a second,
    distinct voice at every GT preferred instant, and annotated.mp4 gains a
    GT-vs-prediction timeline strip so phase error is visible immediately.
    """

    fps_hint = float(meta.fps_stream) if meta.fps_stream and meta.fps_stream > 0 else None
    # Clicks outside the track length would be dropped by the mixer; cover
    # the whole clip plus decay tail even when metadata duration is short.
    track_len = max(float(meta.duration_s or 0.0), default_duration_s(events))
    gt_times = (
        [c.preferred_s for c in ground_truth.contacts] if ground_truth else []
    )
    if gt_times:
        track_len = max(track_len, max(gt_times) + 0.75)
    wav = synthesize_marker_wav(
        events, track_len, out_dir / "markers.wav", gt_marks=gt_times
    )
    with FrameReader(video) as reader:
        render_annotated(
            trace,
            events,
            reader,  # FrameReader yields (bgr, t_s, idx): render's native shape
            out_dir / "annotated.mp4",
            fps=fps_hint,
            click_wav=wav,
            gt_contacts=list(ground_truth.contacts) if ground_truth else None,
            fps_display=fps_hint,
        )
    if preview:
        with FrameReader(video) as reader:
            render_annotated(
                trace,
                events,
                reader,  # same (bgr, t_s, idx) triples as above
                out_dir / "marker-preview.mp4",
                fps=fps_hint,
                click_wav=wav,
                edges=[],  # banner/chip/graph-only preview
                scroll_window_s=1.2,
            )


def analyze_pipeline(
    video: str,
    *,
    backend: str,
    mode: str,
    contact: str,
    out_dir: str | Path,
    run_id: str | None = None,
    config_pairs: Iterable[str] = (),
    no_video: bool = False,
    gt_path: str | Path | None = None,
) -> AnalyzeOutcome:
    """Full analyze pipeline; shared verbatim by the CLI and selftest.

    ``gt_path`` (Phase-2 T15/T1) optionally attaches a schema-v2 ground-truth
    file: GT clicks enter the marker WAV, annotated.mp4 gains the truth
    strip, and events.json records GT provenance. Raises ``ValueError`` on
    an invalid GT file before any expensive stage runs.
    """

    video = str(video)
    run_id = run_id or default_run_id(video, backend, contact)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    if backend == rtmpose_backend.BACKEND_NAME:
        notes.append(RTMPOSE_LICENSE_NOTICE)

    ground_truth: GroundTruth | None = None
    if gt_path is not None:
        ground_truth = load_ground_truth(gt_path)
        if Path(ground_truth.clip).stem != Path(video).stem:
            notes.append(
                f"ground-truth clip {ground_truth.clip!r} does not match "
                f"video stem {Path(video).stem!r}"
            )
        notes.extend(f"ground-truth warning: {w}" for w in ground_truth.warnings)
    with ResourceTracker() as tracker:
        with tracker.stage("probe"):
            meta = probe_video(video)

        with tracker.stage("shots"):
            try:
                shot_boundaries = detect_shot_boundaries(video)
            except RuntimeError as exc:
                # Shot boundaries only feed contact state resets; degrade
                # honestly instead of losing the run to an auxiliary stage.
                logger.warning(
                    "shot detection failed (%s); continuing without boundaries",
                    exc,
                )
                shot_boundaries = []
                notes.append(f"shot_boundary_detection_failed: {exc}")

        source = extract_trace_cached(video, backend, mode, tracker, meta=meta)
        trace = source.trace

        with tracker.stage("contacts"):
            overrides = parse_config_pairs(config_pairs)
            if overrides:
                notes.append(
                    "config overrides: "
                    + ", ".join(f"{k}={v!r}" for k, v in sorted(overrides.items()))
                )
            config = build_contact_config(contact, shot_boundaries, overrides)
            result = detect_contacts(trace, contact, config=config)

        pose_lost = ensure_pose_lost_diagnostic(result, trace, meta)
        if pose_lost:
            notes.append(
                "pose lost across the entire clip: wrote empty event set "
                "with a POSE_LOST interval in diagnostics.json"
            )

        # Durable JSON/CSV artifacts land before video encoding so a render
        # failure can never lose them.
        interim_samples = tracker.report().to_dict().get("rss_samples") or []
        _write_core_artifacts(
            out_dir,
            meta=meta,
            backend=backend,
            mode=mode,
            contact_method=contact,
            result=result,
            trace=trace,
            cache_key=source.cache_key,
            ground_truth=ground_truth,
            notes=notes,
        )

        if not no_video:
            with tracker.stage("render"):
                _render_videos(
                    video, trace, result.events, out_dir, meta,
                    ground_truth=ground_truth,
                )

        report = finalize_report(tracker, cache_hit=source.cache_hit)
        if not report.get("rss_samples") and interim_samples:
            # report() detaches the sampler; keep the interim sample series.
            report["rss_samples"] = interim_samples
    write_json(out_dir / "resource-report.json", report)

    return AnalyzeOutcome(
        run_id=run_id,
        out_dir=out_dir,
        meta=meta,
        events=list(result.events),
        diagnostics_counts=diagnostics_counts(result.diagnostics),
        cache_hit=source.cache_hit,
        report=report,
        pose_lost=pose_lost,
        gt_summary=ground_truth.summary() if ground_truth else None,
    )


# ---------------------------------------------------------------------------
# Human-readable summaries
# ---------------------------------------------------------------------------


def _fmt_stage_line(report: dict) -> str:
    names = ("probe", "shots", "pose_extract", "contacts", "render")
    stages = {s["name"]: s["duration_s"] for s in report.get("stages", [])}
    parts = [f"{n}={stages[n]:.2f}s" for n in names if n in stages]
    parts.append(f"total={float(report.get('total_s') or 0.0):.2f}s")
    return " ".join(parts)


def print_analyze_summary(outcome: AnalyzeOutcome) -> None:
    """Human summary block: counts, diagnostics, wall times, RTF, RSS."""

    meta = outcome.meta
    events = outcome.events
    n_left = sum(1 for e in events if e.foot == LEFT)
    n_right = sum(1 for e in events if e.foot == RIGHT)
    n_obs = sum(1 for e in events if e.observable)
    wall = float(outcome.report.get("total_s") or 0.0)
    dur = float(meta.duration_s or 0.0)
    rtf = f"{wall / dur:.2f}x" if dur > 0 and wall > 0 else "-"
    eff_fps = f"{meta.n_frames / wall:.1f}" if wall > 0 else "-"
    peak_mb = (outcome.report.get("peak_rss_bytes") or 0) / (1024 * 1024)
    diag = (
        ", ".join(f"{k}={v}" for k, v in outcome.diagnostics_counts.items())
        or "-"
    )

    print("== auto-sfx analyze ==")
    print(f"video       {meta.path}")
    rate = f"{meta.fps_stream:.3f}" if meta.fps_stream else "?"
    verdict = "CFR" if meta.cfr else ("VFR" if meta.vfr else "?")
    print(
        f"            {meta.width}x{meta.height} @ {rate}fps {verdict}  "
        f"{meta.n_frames} frames  {dur:.3f}s"
    )
    print(
        f"run         {outcome.run_id}  cache={'HIT' if outcome.cache_hit else 'MISS'}"
    )
    print(
        f"events      {len(events)} (left={n_left} right={n_right}) "
        f"observable={n_obs}"
    )
    print(f"diagnostics {diag}")
    print(f"timings     {_fmt_stage_line(outcome.report)}")
    print(f"throughput  RTF={rtf}  effective_fps={eff_fps}  peak_rss={peak_mb:.1f}MB")
    if outcome.pose_lost or outcome.diagnostics_counts.get(DiagReason.POSE_LOST):
        print(
            "WARNING     pose lost (entire clip or stretches of it); results "
            "are an honest abstention set (see diagnostics.json)"
        )
    if getattr(outcome, "gt_summary", None):
        gs = outcome.gt_summary
        by_foot = " ".join(f"{k}={v}" for k, v in sorted(gs["by_foot"].items()))
        print(
            f"ground-truth {gs['n_contacts']} contacts"
            + (f" ({by_foot})" if by_foot else "")
            + f" range={gs['range_s'][0]:.1f}-{gs['range_s'][1]:.1f}s"
        )
        for w in gs["warnings"]:
            print(f"WARNING     {w}")
    print(f"output      {outcome.out_dir}")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _require_video(video: str) -> bool:
    """Exit-code 2 gate with a clear stderr line for missing inputs."""

    if not Path(video).is_file():
        print(f"auto-sfx: video not found: {video}", file=sys.stderr)
        return False
    return True


def cmd_analyze(args: argparse.Namespace) -> int:
    if not _require_video(args.video):
        return 2
    run_id = (
        sanitize_run_id(args.run_id)
        if args.run_id
        else default_run_id(args.video, args.pose, args.contact)
    )
    out_dir = Path(args.out) if args.out else Path(DEFAULT_RUNS_DIR) / run_id
    try:
        outcome = analyze_pipeline(
            args.video,
            backend=args.pose,
            mode=args.mode,
            contact=args.contact,
            out_dir=out_dir,
            run_id=run_id,
            config_pairs=args.config,
            no_video=args.no_video,
            gt_path=args.gt,
        )
    except ValueError as exc:
        # Invalid --gt file: fail before any expensive stage, with the
        # loader's operator-fixable message.
        print(f"auto-sfx: {exc}", file=sys.stderr)
        return 2
    print_analyze_summary(outcome)
    return 0


def summarize_method(events: list[ContactEvent], diagnostics: list[dict]) -> dict:
    """Per-method aggregates for compare-summary.json."""

    times = sorted(e.time_s for e in events)
    gaps = [b - a for a, b in zip(times, times[1:])]
    mean_score = (
        sum(e.score for e in events) / len(events) if events else None
    )
    return {
        "n_events": len(events),
        "n_left": sum(1 for e in events if e.foot == LEFT),
        "n_right": sum(1 for e in events if e.foot == RIGHT),
        "mean_score": None if mean_score is None else round(mean_score, 6),
        # Mean gap between consecutive detected contacts (any foot) -- a
        # rough cadence read for quick cross-method comparison.
        "mean_step_interval_s": (
            round(sum(gaps) / len(gaps), 6) if gaps else None
        ),
        "diagnostics_counts": diagnostics_counts(diagnostics),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    if not _require_video(args.video):
        return 2
    methods = list(args.contacts)
    run_id = (
        sanitize_run_id(args.run_id)
        if args.run_id
        else default_run_id(args.video, args.pose, "compare")
    )
    out_dir = Path(args.out) if args.out else Path(DEFAULT_RUNS_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = out_dir / "compare"
    notes: list[str] = []

    # One probe/shots/extraction shared by every method.
    with ResourceTracker() as master:
        with master.stage("probe"):
            meta = probe_video(args.video)
        with master.stage("shots"):
            try:
                shot_boundaries = detect_shot_boundaries(args.video)
            except RuntimeError as exc:
                logger.warning(
                    "shot detection failed (%s); continuing without boundaries",
                    exc,
                )
                shot_boundaries = []
                notes.append(f"shot_boundary_detection_failed: {exc}")
        source = extract_trace_cached(
            args.video, args.pose, args.mode, master, meta=meta
        )

    summary: dict[str, dict] = {}
    for method in methods:
        method_dir = compare_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        sub = ResourceTracker()
        with sub:
            with sub.stage("contacts"):
                config = build_contact_config(method, shot_boundaries, {})
                result = detect_contacts(source.trace, method, config=config)
            ensure_pose_lost_diagnostic(result, source.trace, meta)
            _write_core_artifacts(
                method_dir,
                meta=meta,
                backend=args.pose,
                mode=args.mode,
                contact_method=method,
                result=result,
                trace=source.trace,
                cache_key=source.cache_key,
                notes=list(notes),
            )
            interim_samples = sub.report().to_dict().get("rss_samples") or []
            if not args.no_video:
                with sub.stage("render"):
                    _render_videos(
                        args.video,
                        source.trace,
                        result.events,
                        method_dir,
                        meta,
                        preview=False,  # annotated.mp4 only per method
                    )
            rep = finalize_report(sub, cache_hit=source.cache_hit)
            if not rep.get("rss_samples") and interim_samples:
                rep["rss_samples"] = interim_samples
        write_json(method_dir / "resource-report.json", rep)
        summary[method] = summarize_method(result.events, result.diagnostics)

    write_json(
        out_dir / "compare-summary.json",
        {
            "schema_version": COMPARE_SCHEMA_VERSION,
            "video": meta.to_dict(),
            "backend": args.pose,
            "mode": args.mode,
            "methods": summary,
        },
    )

    print("== auto-sfx compare ==")
    print(f"video       {meta.path}  backend={args.pose} mode={args.mode}")
    print(f"pose cache  {'HIT' if source.cache_hit else 'MISS'} (shared)")
    rows = []
    for method, agg in summary.items():
        diag = ",".join(f"{k}={v}" for k, v in agg["diagnostics_counts"].items())
        rows.append(
            {
                "method": method,
                "n_events": agg["n_events"],
                "left": agg["n_left"],
                "right": agg["n_right"],
                "mean_score": agg["mean_score"],
                "mean_step_interval_s": agg["mean_step_interval_s"],
                "diagnostics": diag or "-",
            }
        )
    print_table(rows)
    print(f"output      {out_dir}")
    return 0


def _merge_resources(run_dir: Path) -> tuple[dict, float]:
    """Pull processing wall/peak RSS/startup out of resource-report.json."""

    resources: dict[str, Any] = {
        "processing_wall_time_s": None,
        "peak_rss_bytes": None,
        "startup_s": None,
        "cache_hit": None,
    }
    rp_path = run_dir / "resource-report.json"
    if not rp_path.is_file():
        return resources, 0.0
    try:
        rp = json.loads(rp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", rp_path, exc)
        return resources, 0.0
    wall = rp.get("total_s")
    peak = rp.get("peak_rss_bytes")
    startup = next(
        (
            s.get("duration_s")
            for s in rp.get("stages", [])
            if s.get("name") == "startup"
        ),
        None,
    )
    resources.update(
        {
            "processing_wall_time_s": wall,
            "peak_rss_bytes": peak,
            "startup_s": startup,
            "cache_hit": rp.get("cache_hit"),
        }
    )
    return resources, float(wall or 0.0)


def _gt_is_v2(gt_path: str | Path) -> bool:
    """True when the GT file starts as a v2 object (not a legacy array)."""

    return Path(gt_path).read_text(encoding="utf-8-sig").lstrip()[:1] == "{"


def evaluate_run_document(
    events_doc: dict, gt_path: str | Path, tolerance_s: float | None = None
) -> dict:
    """evaluate()/evaluate_v2() over an events.json document + resources.

    Legacy v1 GT arrays produce the v1 flat report (back-compat); schema-v2
    GT files produce the Phase-2 report shape (§5.4) with v1 fields nested
    under ``existence``.
    """

    video = events_doc.get("video") or {}
    displayed_fps = video.get("fps_stream") or None
    run_dir = Path(str(events_doc.get("_run_dir") or "."))
    resources, _wall = _merge_resources(run_dir)
    duration = float(video.get("duration_s") or 0.0)
    wall = float(resources.get("processing_wall_time_s") or 0.0)
    resources["rtf"] = (round(wall / duration, 6) if duration > 0 and wall > 0 else None)
    peak = resources.get("peak_rss_bytes")
    resources["peak_rss_mb"] = (
        round(float(peak) / (1024 * 1024), 3) if peak else None
    )

    if _gt_is_v2(gt_path):
        from auto_sfx.metrics import evaluate_v2, load_gt_any

        report = evaluate_v2(
            events_doc.get("events") or [],
            load_gt_any(str(gt_path)),
            displayed_fps,
            tol_ms=None if tolerance_s is None else float(tolerance_s) * 1000.0,
            resources=resources,
        )
        return report

    annotations = load_annotations(str(gt_path))
    report = evaluate(
        events_doc.get("events") or [],
        annotations,
        displayed_fps,
        tol_ms=None if tolerance_s is None else float(tolerance_s) * 1000.0,
    )
    report["resources"] = resources
    return report



_METRIC_ROWS = (
    ("true_events", "n_true"),
    ("predicted_events", "n_pred"),
    ("matched", "matched"),
    ("false_positives", "false_positives"),
    ("misses", "misses"),
    ("duplicates", "duplicates"),
    ("wrong_side", "wrong_side"),
    ("median_error_s", "median_err_s"),
    ("p90_error_s", "p90_err_s"),
    ("within_1_displayed_frame_frac", "within_1_frame_frac"),
)


def cmd_metrics(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    events_path = run_dir / "events.json"
    if not events_path.is_file():
        print(
            f"auto-sfx: {events_path} not found (expected a run directory "
            "produced by analyze)",
            file=sys.stderr,
        )
        return 2
    if not Path(args.gt_json).is_file():
        print(f"auto-sfx: ground truth file not found: {args.gt_json}",
              file=sys.stderr)
        return 2

    doc = json.loads(events_path.read_text(encoding="utf-8"))
    doc["_run_dir"] = str(run_dir)  # internal: where resource-report lives
    report = evaluate_run_document(doc, args.gt_json, tolerance_s=args.tolerance_s)

    out_path = Path(args.out) if args.out else run_dir / "metrics.json"
    write_json(
        out_path,
        {
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "ground_truth": str(args.gt_json),
            "tolerance_s": args.tolerance_s,
            **report,
        },
    )

    resources = report.get("resources", {})
    if "existence" in report:
        ex = report["existence"]
        flat = {
            "n_true": ex.get("n_true"),
            "n_pred": ex.get("n_pred"),
            "matched": ex.get("matched"),
            "false_positives": ex.get("false_positives"),
            "misses": ex.get("misses"),
            "duplicates": ex.get("duplicates"),
            "wrong_side": report["side"].get("side_flips"),
            "median_err_s": report["timing"].get("median_abs_err_s"),
            "p90_err_s": report["timing"].get("p90_abs_err_s"),
            "within_1_frame_frac": report["timing"].get("within_1_frame_frac"),
            "precision": ex.get("precision"),
            "recall": ex.get("recall"),
            "attribution_accuracy": report["side"].get("attribution_accuracy"),
            "bias_s": report["timing"].get("bias_s"),
            "catastrophic_gt3frame": report["timing"].get(
                "catastrophic_gt3frame"),
            "impossible_simultaneous_ic": report["sequence"].get(
                "impossible_simultaneous_ic"),
        }
        rows = [{"metric": k, "value": v} for k, v in flat.items()]
    else:
        rows = [
            {"metric": label, "value": report.get(key)}
            for label, key in _METRIC_ROWS
        ]
    rows += [
        {
            "metric": "processing_wall_time_s",
            "value": resources.get("processing_wall_time_s"),
        },
        {"metric": "rtf", "value": resources.get("rtf")},
        {"metric": "peak_rss_mb", "value": resources.get("peak_rss_mb")},
        {"metric": "startup_s", "value": resources.get("startup_s")},
    ]
    print(f"== metrics: {run_dir} vs {args.gt_json} ==")
    print_table(rows)
    print(f"written     {out_path}")
    return 0


def cmd_forensics(args: argparse.Namespace) -> int:
    if not Path(args.run_dir).is_dir():
        print(f"auto-sfx: run directory not found: {args.run_dir}",
              file=sys.stderr)
        return 2
    if args.gt_json and not Path(args.gt_json).is_file():
        print(f"auto-sfx: ground truth file not found: {args.gt_json}",
              file=sys.stderr)
        return 2
    from auto_sfx.forensics import build_forensics

    try:
        out_dir = build_forensics(
            args.run_dir,
            args.gt_json,
            args.out,
            event_id=args.event,
            half_s=args.half_s,
        )
    except ValueError as exc:
        print(f"auto-sfx: {exc}", file=sys.stderr)
        return 2
    print(f"== forensics: {args.run_dir} ==")
    print(f"windows     see {out_dir}/index.json")
    print(f"written     {out_dir}")
    return 0


def load_gt_entries(path: str | Path) -> list[dict]:
    """Load and validate a ground-truth array for the annotate command."""

    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from None
    parsed: list[dict] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {i} is not an object")
        t = entry.get("time_s")
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            raise ValueError(f"{path}: entry {i} has no numeric time_s")
        foot = entry.get("foot")
        if foot not in SIDES:
            raise ValueError(
                f"{path}: entry {i} foot must be one of {list(SIDES)}"
            )
        parsed.append({"time_s": float(t), "foot": foot})
    for a, b in zip(parsed, parsed[1:]):
        if b["time_s"] < a["time_s"]:
            raise ValueError(
                f"{path}: ground-truth entries must be sorted by time_s "
                f"({a['time_s']} followed by {b['time_s']})"
            )
    return parsed


def cmd_annotate(args: argparse.Namespace) -> int:
    if not _require_video(args.video):
        return 2
    k = max(0, (args.frames - 1) // 2)

    meta = probe_video(args.video)
    duration = float(meta.duration_s or 0.0)
    out_root = (
        Path(args.out)
        if args.out
        else Path(DEFAULT_ANNOTATIONS_DIR) / Path(args.video).stem
    )

    if args.at is not None:
        sessions = [("at", {"time_s": float(args.at), "foot": ""})]
    else:
        try:
            sessions = [
                (f"{i:03d}", entry)
                for i, entry in enumerate(load_gt_entries(args.gt))
            ]
        except (OSError, ValueError) as exc:
            print(f"auto-sfx: {exc}", file=sys.stderr)
            return 2

    frame_times = collect_frame_times(args.video)
    if frame_times:
        lo, hi = frame_times[0], frame_times[-1]
    else:
        lo, hi = 0.0, duration
    for name, entry in sessions:
        t = entry["time_s"]
        if t < lo - 1e-6 or t > hi + 1e-6:
            print(
                f"auto-sfx: annotation time {t:.6f}s outside video range "
                f"[{lo:.6f}, {hi:.6f}]",
                file=sys.stderr,
            )
            return 2

        window = neighbors(frame_times, t, k) if frame_times else []
        header = f"-- {name} t={t:.6f}s"
        if entry["foot"]:
            header += f" foot={entry['foot']}"
        print(header)
        print_table(
            [{"frame_index": idx, "t_s": ts} for idx, ts in window] or
            [{"note": "no frames probed near this timestamp"}]
        )
        session_dir = (
            out_root
            if args.at is not None
            else out_root / f"{name}_{entry['foot']}_t{int(round(t * 1_000_000))}"
        )
        paths = extract_frames(args.video, t, k=k, out_dir=session_dir)
        for p in paths:
            print(f"extracted   {p}")
    return 0


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


def _fixture_script_candidates() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("AUTO_SFX_MAKE_FIXTURES")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()  # <root>/src/auto_sfx/cli.py
    candidates += [
        here.parents[2] / "scripts" / "make_fixtures.py",  # repo checkout
        here.parents[1] / "scripts" / "make_fixtures.py",  # flat layout
        Path.cwd() / "scripts" / "make_fixtures.py",
        Path("/app/scripts/make_fixtures.py"),  # container image layout
    ]
    return candidates


def _ensure_fixtures() -> Path:
    """Make sure the synthetic fixtures exist, generating them if possible."""

    walk = FIXTURES_DIR / "walk_synthetic.mp4"
    if walk.is_file():
        return walk
    script = next((c for c in _fixture_script_candidates() if c.is_file()), None)
    if script is None:
        raise RuntimeError(
            f"{walk} is missing and scripts/make_fixtures.py could not be "
            "located; generate the fixtures first (python "
            "experiments/auto-sfx/scripts/make_fixtures.py from the "
            "repository root, or set AUTO_SFX_MAKE_FIXTURES)"
        )
    print(f"[selftest] generating fixtures via {script} ...")
    proc = subprocess.run([sys.executable, str(script)], check=False)
    if proc.returncode != 0 or not walk.is_file():
        raise RuntimeError(
            f"fixture generation failed (rc={proc.returncode}, "
            f"{walk} still missing)"
        )
    return walk


def _timing_rows(label: str, report: dict) -> list[dict]:
    rows = [
        {"run": label, "stage": s["name"], "duration_s": s.get("duration_s")}
        for s in report.get("stages", [])
    ]
    rows.append(
        {"run": label, "stage": "total", "duration_s": report.get("total_s")}
    )
    return rows


def cmd_selftest(args: argparse.Namespace) -> int:
    walk = _ensure_fixtures()
    gt_path = FIXTURES_DIR / "walk_synthetic_contacts.json"
    vfr_path = FIXTURES_DIR / "vfr_walk.mp4"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"== selftest: CFR synthetic walk ({args.pose}/{args.contact}, videos ON) ==")
    cfr_out = Path(DEFAULT_RUNS_DIR) / f"selftest-cfr-{stamp}"
    cfr = analyze_pipeline(
        str(walk),
        backend=args.pose,
        mode="video",
        contact=args.contact,
        out_dir=cfr_out,
        run_id=cfr_out.name,
        no_video=False,
    )
    print_analyze_summary(cfr)

    print("\nraw metrics vs walk_synthetic_contacts.json (no gates):")
    doc = json.loads((cfr.out_dir / "events.json").read_text(encoding="utf-8"))
    doc["_run_dir"] = str(cfr.out_dir)
    cfr_metrics = evaluate_run_document(doc, gt_path)
    rows = [
        {"metric": label, "value": cfr_metrics.get(key)}
        for label, key in _METRIC_ROWS
    ]
    res = cfr_metrics.get("resources", {})
    rows += [
        {"metric": "processing_wall_time_s", "value": res.get("processing_wall_time_s")},
        {"metric": "rtf", "value": res.get("rtf")},
        {"metric": "peak_rss_mb", "value": res.get("peak_rss_mb")},
        {"metric": "startup_s", "value": res.get("startup_s")},
    ]
    print_table(rows)

    print(f"\n== selftest: VFR synthetic walk (videos OFF) ==")
    vfr_out = Path(DEFAULT_RUNS_DIR) / f"selftest-vfr-{stamp}"
    vfr = analyze_pipeline(
        str(vfr_path),
        backend=args.pose,
        mode="video",
        contact=args.contact,
        out_dir=vfr_out,
        run_id=vfr_out.name,
        no_video=True,
    )
    vfr_doc = json.loads((vfr.out_dir / "events.json").read_text(encoding="utf-8"))
    assert isinstance(vfr_doc.get("events"), list), "vfr events.json must parse"
    print(f"vfr events  {len(vfr_doc['events'])}")
    print_analyze_summary(vfr)

    print("\ntimings (CFR vs VFR):")
    print_table(_timing_rows("cfr", cfr.report) + _timing_rows("vfr", vfr.report))
    print("\nselftest complete: raw values above, no gates applied.")
    return 0


# ---------------------------------------------------------------------------
# Parser / entry point
# ---------------------------------------------------------------------------


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    """Attach the five POC subcommands (kept separate for testability)."""

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser(
        "analyze",
        help="detect foot contacts in one video; write the full artifact set",
    )
    p.add_argument("video", help="input video path")
    p.add_argument("--pose", choices=list(POSE_BACKENDS), default="mediapipe-lite")
    p.add_argument("--mode", choices=list(RUNNING_MODES), default="video")
    p.add_argument("--contact", choices=list(CONTACT_METHODS), default="hybrid")
    p.add_argument(
        "--out",
        default=None,
        help=f"output directory (default {DEFAULT_RUNS_DIR}/<run-id>)",
    )
    p.add_argument("--run-id", default=None, help="override the generated run id")
    p.add_argument(
        "--no-video", action="store_true", help="skip annotated.mp4/marker-preview.mp4"
    )
    p.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="passthrough merged into the contact method's default config "
        "(numeric values parsed; repeatable)",
    )
    p.add_argument(
        "--gt",
        default=None,
        metavar="GT_JSON",
        help="schema-v2 ground-truth file (docs/auto-sfx-phase2-truth-repair.md "
        "§5.1): adds GT clicks to the marker WAV and a GT-vs-prediction "
        "timeline strip to annotated.mp4",
    )
    p.set_defaults(handler=cmd_analyze)

    p = sub.add_parser(
        "compare",
        help="run several contact methods over one shared pose extraction",
    )
    p.add_argument("video", help="input video path")
    p.add_argument("--pose", choices=list(POSE_BACKENDS), default="mediapipe-lite")
    p.add_argument("--mode", choices=list(RUNNING_MODES), default="video")
    p.add_argument(
        "--contacts",
        nargs="+",
        choices=list(CONTACT_METHODS),
        default=list(CONTACT_METHODS),
        help="contact methods to compare (default: all)",
    )
    p.add_argument(
        "--out", default=None, help=f"output directory (default {DEFAULT_RUNS_DIR}/<run-id>)"
    )
    p.add_argument("--run-id", default=None, help="override the generated run id")
    p.add_argument("--no-video", action="store_true", help="skip per-method annotated.mp4")
    p.set_defaults(handler=cmd_compare)

    p = sub.add_parser(
        "metrics",
        help="score a run directory's events.json against manual ground truth",
    )
    p.add_argument("run_dir", help="run directory containing events.json")
    p.add_argument("gt_json", help='ground truth: [{"time_s": .., "foot": ..}]')
    p.add_argument(
        "--tolerance-s",
        type=float,
        default=0.034,
        help="match tolerance in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="where to write metrics.json (default: next to events.json)",
    )
    p.set_defaults(handler=cmd_metrics)

    p = sub.add_parser(
        "annotate",
        help="print frame/PTS neighbourhoods and extract PNGs for hand-marking",
    )
    p.add_argument("video", help="input video path")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--at", type=float, metavar="SECONDS", help="timestamp to inspect")
    group.add_argument(
        "--gt", metavar="FILE", help="ground-truth JSON; inspect every entry"
    )
    p.add_argument(
        "--frames",
        type=int,
        default=5,
        help="number of adjacent frames to show/extract (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        default=None,
        help=f"output directory (default {DEFAULT_ANNOTATIONS_DIR}/<video stem>)",
    )
    p.set_defaults(handler=cmd_annotate)

    p = sub.add_parser(
        "forensics",
        help="per-event diagnostic windows (JSON+PNG) from a run's cached "
        "pose trace (Phase-2 T2)",
    )
    p.add_argument(
        "run_dir", help="run directory containing events.json + pose.jsonl"
    )
    p.add_argument(
        "gt_json",
        nargs="?",
        default=None,
        help="ground-truth JSON (v2 canonical, v1 readable); adds GT overlays "
        "and missed-GT windows",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--event", metavar="ID", help="only this event id (e.g. evt_007)"
    )
    group.add_argument(
        "--all", action="store_true", help="all windows (default)"
    )
    p.add_argument(
        "--out",
        default=None,
        help="output directory (default: <run-dir>/forensics)",
    )
    p.add_argument(
        "--half-s",
        type=float,
        default=0.65,
        help="half window length in seconds (default: %(default)s)",
    )
    p.set_defaults(handler=cmd_forensics)

    p = sub.add_parser(
        "selftest",
        help="end-to-end run over the synthetic fixtures; raw values, no gates",
    )
    p.add_argument("--pose", choices=list(POSE_BACKENDS), default="mediapipe-lite")
    p.add_argument("--contact", choices=list(CONTACT_METHODS), default="hybrid")
    p.set_defaults(handler=cmd_selftest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-sfx",
        description="auto-sfx POC: video -> physical event detection -> timeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    add_subparsers(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("auto-sfx: interrupted", file=sys.stderr)
        return 130
