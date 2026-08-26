#!/usr/bin/env python3
"""Timing bake-off scorer — vision detectors vs HUMAN gold (CTO §7 metrics).

Phase-2 Q2 tool (docs/sfx-bank-foley-phase2-eval-brief.md §3.3). Scores impulse
contact predictions against human frame-level gold annotations. No audio
generation, no network.

Metrics per clip and aggregate:
  - recall @ ±33 / ±50 / ±100 ms (detection within tolerance of a gold contact)
  - false positives per minute (unmatched detections / clip duration)
  - missed gold events count
  - median + p95 onset error over matched pairs (ms) — never median alone
  - secondary: % detections landing inside the human acceptable_frames window

Detector input is sniffed from three shapes:
  a) µs-PTS contact JSONs: {"contacts": [{"earliest_pts_us":…,"preferred_pts_us":…}, …]}
     (auto-sfx draft-GT / runs/events.json style)
  b) {"events": [{"time_s": …}, …]}                  (POC oracle/validation style)
  c) bare numeric arrays                             (seconds)

Gold input: human-gold-timing JSONs produced by experiments/auto-sfx/
gold_annotator.html (contract in docs/sfx-bank-foley-phase2-eval-brief.md):
  {"schema_version":1,"kind":"human-gold-timing","clip":str,"fps":float,
   "annotator":str,"events":[{"event_id","temporal_type","family","class",
   "start_frame","contact_frame","end_frame","acceptable_frames","confidence",
   "sequence", "notes"}]}

Self-test (--selftest) asserts exact known-answer metrics; --dryrun scores
auto-sfx preferred contacts against its own earliest..latest windows and is
labelled NOT human gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

GOLD_KIND = "human-gold-timing"
SELFTEST_DURATION_S = 6.0


# ---------------------------------------------------------------- input parsing

def _sniff_detected(path: Path) -> tuple[str, list[float]]:
    """Return (clip_hint, detection_seconds) from flexible detector JSON."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        if all(isinstance(x, (int, float)) for x in data):
            return path.stem, [float(x) for x in data]
        raise SystemExit(f"{path}: bare array must contain numbers")
    if isinstance(data, dict):
        contacts = data.get("contacts")
        if isinstance(contacts, list):
            times = []
            for c in contacts:
                pref = c.get("preferred_pts_us")
                if pref is None:
                    raise SystemExit(f"{path}: contact entry missing preferred_pts_us")
                times.append(pref / 1e6)
            clip = data.get("clip") or path.stem.replace(".agent-draft.gt", "")
            return str(clip), times
        events = data.get("events")
        if isinstance(events, list):
            times = []
            for ev in events:
                t = ev.get("time_s", ev.get("time"))
                if t is None:
                    raise SystemExit(f"{path}: event entry missing time_s")
                times.append(float(t))
            clip = data.get("clip") or path.stem
            return str(clip), times
    raise SystemExit(f"{path}: unrecognized detector format")


def _load_gold(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if data.get("kind") != GOLD_KIND:
        raise SystemExit(f"{path}: expected kind={GOLD_KIND!r}")
    fps = float(data["fps"])
    out = []
    for ev in data["events"]:
        cf = ev.get("contact_frame")
        window = ev.get("acceptable_frames")
        win = [int(window[0]), int(window[1])] if window else None
        out.append({
            "event_id": ev["event_id"],
            "family": ev.get("family", ""),
            "temporal_type": ev.get("temporal_type", ""),
            # impulse contacts define timing; other types scored only when contact given
            "t_s": None if cf is None else (int(cf) + 0.5) / fps,
            "window_s": ([w / fps for w in win] if win else None),
        })
    return out


def load_gold_paths(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Multiple annotator files may target one clip → merge by clip id."""
    merged: dict[str, list[dict[str, Any]]] = {}
    for p in paths:
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            if f.name.endswith(".tmp"):
                continue
            data = json.loads(f.read_text())
            if data.get("kind") != GOLD_KIND:
                continue  # silently skip unrelated JSONs in dirs
            clip = data["clip"]
            events = _load_gold(f)
            entry = merged.setdefault(clip, [])
            entry.extend(events)
    return merged


def load_detected_paths(paths: list[Path]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for p in paths:
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            try:
                clip, times = _sniff_detected(f)
            except SystemExit as exc:
                if p.is_dir():
                    continue  # tolerate unrelated files inside dirs
                raise exc
            merged.setdefault(clip, []).extend(times)
    return {k: sorted(v) for k, v in merged.items()}


# ---------------------------------------------------------------- matching

def match_pairs(gold_times: list[float], dets: list[float], tol_ms: int
                ) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Greedy nearest mutual-best matching under tolerance. Returns
    ([(gold_idx, det_idx, err_ms)], unmatched_gold, unmatched_det)."""
    tol = tol_ms / 1000.0
    cand = sorted(
        ((abs(t - g), gi, di) for gi, g in enumerate(gold_times)
         for di, t in enumerate(dets) if abs(t - g) <= tol),
    )
    used_g: set[int] = set()
    used_d: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for _, gi, di in cand:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        pairs.append((gi, di, abs(dets[di] - gold_times[gi]) * 1000.0))
    return pairs, set(range(len(gold_times))) - used_g, set(range(len(dets))) - used_d


def pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 1) if denominator else None


def score_clip(gold_events: list[dict[str, Any]], dets: list[float],
               duration_s: float | None, tolerances_ms: list[int]) -> dict[str, Any]:
    impulses = [g for g in gold_events if g["t_s"] is not None]
    gold_times = [g["t_s"] for g in impulses]
    row: dict[str, Any] = {
        "gold_impulses": len(gold_times),
        "detections": len(dets),
    }
    if duration_s:
        row["duration_s"] = round(duration_s, 2)
    recalls: dict[str, float | None] = {}
    errors_all: list[float] = []
    for tol in tolerances_ms:
        pairs, ug, ud = match_pairs(gold_times, dets, tol)
        row[f"missed@±{tol}"] = len(ug)
        row[f"fp@±{tol}"] = len(ud)
        if duration_s:
            row[f"fp_per_min@±{tol}"] = round(len(ud) * 60.0 / duration_s, 2) if duration_s > 0 else None
        recalls[f"{tol}"] = pct(len(pairs), len(gold_times))
        if not errors_all and pairs:
            # error stats come from the TIGHTEST tolerance (list is sorted ascending)
            errors_all = [p[2] for p in pairs]
    row["recall"] = recalls
    if errors_all:
        es = sorted(errors_all)
        med = es[len(es) // 2] if len(es) % 2 else (es[len(es) // 2 - 1] + es[len(es) // 2]) / 2
        p95 = es[min(len(es) - 1, int(round(0.95 * (len(es) - 1))))]
        row["median_err_ms"] = round(med, 1)
        row["p95_err_ms"] = round(p95, 1)
    else:
        row["median_err_ms"] = None
        row["p95_err_ms"] = None
    # secondary: acceptable-window coverage
    n_window, n_in = 0, 0
    for g in impulses:
        if g["window_s"] and dets:
            lo, hi = g["window_s"]
            hits = [d for d in dets if lo <= d <= hi]
            n_window += 1
            if hits:
                n_in += 1
    row["acceptable_window_coverage"] = pct(n_in, n_window)
    return row


# ---------------------------------------------------------------- selftest / dryrun

def run_selftest(tolerances_ms: list[int]) -> bool:
    gold = [
        {"t_s": 1.0, "family": "steps", "window_s": None},
        {"t_s": 2.0, "family": "steps", "window_s": None},
        {"t_s": 3.0, "family": "impacts", "window_s": None},
    ]
    dets = [1.020, 2.080, 3.000, 3.900]
    ok = True
    check = score_clip(gold, dets, SELFTEST_DURATION_S, tolerances_ms)

    def expect(key: str, value: Any) -> None:
        nonlocal ok
        actual = key.split(".")[-1] and _dig(check, key)
        good = actual == value or (
            isinstance(value, float) and isinstance(actual, (int, float))
            and abs(actual - value) < 1e-9)
        if not good:
            print(f"  FAIL {key}: expected {value}, got {actual}")
            ok = False
        else:
            print(f"  ok   {key} = {actual}")

    expect("gold_impulses", 3)
    expect("recall.33", 66.7)          # hits: 1.020, 3.000
    expect("recall.50", 66.7)          # 2.080 is 80 ms out
    expect("recall.100", 100.0)        # 2.080 within ±100
    expect("missed@±33", 1)
    expect("missed@±100", 0)
    expect("fp_per_min@±33", 20.0)     # unmatched @33: 2.080, 3.900 over 6 s
    expect("fp_per_min@±100", 10.0)    # only 3.900 unmatched
    expect("median_err_ms", 10.0)      # matched errors [0, 20] at ±33
    expect("p95_err_ms", 20.0)
    print(json.dumps(check, indent=2))

    pairs33, _, ud33 = match_pairs([g["t_s"] for g in gold], dets, min(tolerances_ms))
    assert len(pairs33) == 2 and ud33 == {1, 3}, (pairs33, ud33)
    pairs100, ug100, ud100 = match_pairs([g["t_s"] for g in gold], dets, max(tolerances_ms))
    assert len(pairs100) == 3 and not ug100 and ud100 == {3}, (pairs100, ug100, ud100)
    boundary = match_pairs([10.0], [10.033], 33)[0]
    assert len(boundary) == 1 and abs(boundary[0][2] - 33.0) < 1e-6, "inclusive boundary failed"
    return ok


def _dig(row: dict[str, Any], dotted: str) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def dryrun_paths(repo: Path) -> tuple[list[Path], list[Path]]:
    gt_dir = repo / "experiments" / "auto-sfx" / "gt"
    gold = sorted(gt_dir.glob("*.agent-draft.gt.json"))
    return gold, gold


# ---------------------------------------------------------------- report

def render_markdown(per_clip: dict[str, dict[str, Any]], tolerances_ms: list[int],
                    title: str, note: str | None) -> str:
    lines = [f"# Timing bake-off — {title}", ""]
    if note:
        lines += [f"> **{note}**", ""]
    cols = ["gold_impulses", "detections"]
    for t in tolerances_ms:
        cols.append(f"recall@±{t}")
    cols += ["fp_per_min@±%d" % tolerances_ms[-1], "median_err_ms", "p95_err_ms",
             "acceptable_window_coverage"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines += [header, sep]

    out_rows = []
    for clip, row in per_clip.items():
        if clip == "aggregate":
            continue
        cells = []
        for col in cols:
            if col.startswith("recall@±"):
                v: Any = row["recall"].get(col.split("±")[1]) if "recall" in row else None
            else:
                v = row.get(col)
            cells.append(str(v) if v is not None else "—")
        out_rows.append("| " + clip + " | " + " | ".join(cells) + " |")
    lines += out_rows
    if "aggregate" in per_clip:
        agg = per_clip["aggregate"]
        lines.append(sep)
        cells = []
        for col in cols:
            if col.startswith("recall@±"):
                v: Any = agg["recall"].get(col.split("±")[1])
            else:
                v = agg.get(col)
            cells.append(str(v) if v is not None else "—")
        lines.append("| **aggregate** | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", nargs="*", type=Path, default=[])
    parser.add_argument("--detected", nargs="*", type=Path, default=[])
    parser.add_argument("--tolerances", nargs="*", type=int, default=[33, 50, 100])
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--duration-json", type=Path,
                        help="optional {'<clip>': duration_s, ...} map for FP/min normalization")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dryrun", action="store_true",
                        help="score auto-sfx preferred vs earliest..latest pseudo-gold "
                             "(NOT human gold; smoke test only)")
    args = parser.parse_args()
    args.tolerances = sorted(set(args.tolerances))

    if args.selftest:
        print("selftest:")
        passed = run_selftest(args.tolerances)
        print("PASS" if passed else "FAIL")
        return 0 if passed else 1

    durations: dict[str, float] = {}
    gold_by_clip: dict[str, list[dict[str, Any]]]
    det_by_clip: dict[str, list[float]]
    if args.dryrun:
        # Synthesize pseudo-gold from the draft GT's own µs windows:
        # gold contact = window midpoint, acceptable window = [earliest, latest],
        # detector = preferred_pts_us. Internal consistency smoke test ONLY.
        gt_files = sorted((Path(__file__).resolve().parents[1].parent
                           / "experiments/auto-sfx/gt").glob("walk*.agent-draft.gt.json"))
        gold_by_clip, det_by_clip = {}, {}
        for p in gt_files:
            data = json.loads(p.read_text())
            clip = data["clip"]
            rng_s = data.get("range_s") or {}
            if rng_s.get("end"):
                durations[clip] = float(rng_s["end"])
            gevs: list[dict[str, Any]] = []
            dets: list[float] = []
            for c in data.get("contacts", []):
                lo = c["earliest_pts_us"] / 1e6
                hi = c["latest_pts_us"] / 1e6
                gevs.append({"event_id": "pseudo", "family": "pseudo", "temporal_type": "impulse",
                             "t_s": (lo + hi) / 2.0, "window_s": [lo, hi]})
                dets.append(c["preferred_pts_us"] / 1e6)
            gold_by_clip[clip] = gevs
            det_by_clip[clip] = sorted(dets)
    else:
        if not args.gold or not args.detected:
            parser.error("need --gold and --detected (or --selftest/--dryrun)")
        gold_by_clip = load_gold_paths(list(args.gold))
        det_by_clip = load_detected_paths(list(args.detected))
        if args.duration_json:
            durations = {k: float(v) for k, v in json.loads(args.duration_json.read_text()).items()}

    all_clips = sorted(set(gold_by_clip) & set(det_by_clip))
    skipped = sorted(set(det_by_clip) - set(gold_by_clip))
    if skipped:
        print(f"note: detector clips without gold skipped: {skipped}")

    per_clip: dict[str, dict[str, Any]] = {}
    bucket: dict[int, list[list[tuple[int, int, float]]]] = {t: [] for t in args.tolerances}
    tot_gold = tot_det = 0
    fp_counts = {t: 0 for t in args.tolerances}
    miss_counts = {t: 0 for t in args.tolerances}
    errors_master: list[float] = []

    note = ("DRY RUN — auto-sfx internal consistency check (preferred vs own windows); "
            "NOT human ground truth." if args.dryrun else None)
    title = ("dry-run: auto-sfx self-consistency" if args.dryrun
             else ", ".join(str(p.name) for p in args.gold)[:120])

    for clip in all_clips:
        gevs, dets = gold_by_clip[clip], det_by_clip[clip]
        row = score_clip(gevs, dets, durations.get(clip), args.tolerances)
        per_clip[clip] = row
        tot_gold += row["gold_impulses"]
        tot_det += row["detections"]
        dur = durations.get(clip, 0.0)
        gold_times = [g["t_s"] for g in gevs if g["t_s"] is not None]
        for t in args.tolerances:
            pairs, ug, ud = match_pairs(gold_times, dets, t)
            miss_counts[t] += len(ug)
            fp_counts[t] += len(ud)
            for (_, _, e) in pairs:
                errors_master.append(e)
        if dur:
            per_clip.setdefault("_durations", {})[clip] = dur  # type: ignore[assignment]
    per_clip.pop("_durations", None)

    agg_row: dict[str, Any] = {
        "gold_impulses": tot_gold,
        "detections": tot_det,
        "recall": {},
        "median_err_ms": None,
        "p95_err_ms": None,
    }
    total_dur = sum(durations.values())
    for t in args.tolerances:
        agg_row["recall"][str(t)] = pct(tot_gold - miss_counts[t], tot_gold)
        agg_row[f"missed@±{t}"] = miss_counts[t]
        agg_row[f"fp@±{t}"] = fp_counts[t]
        agg_row[f"fp_per_min@±{t}"] = (
            round(fp_counts[t] * 60.0 / total_dur, 2) if total_dur else None)
    if errors_master:
        es = sorted(errors_master)
        agg_row["median_err_ms"] = round(es[len(es) // 2], 1) if len(es) % 2 else round(
            (es[len(es) // 2 - 1] + es[len(es) // 2]) / 2, 1)
        agg_row["p95_err_ms"] = round(es[int(round(0.95 * (len(es) - 1)))], 1)
    per_clip["aggregate"] = agg_row

    md = render_markdown(per_clip, args.tolerances, title, note)
    print(md)

    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "timing_bakeoff.md").write_text(md)
        serial = {k: v for k, v in per_clip.items()}
        (args.report_dir / "timing_bakeoff.json").write_text(json.dumps(serial, indent=2))
        print(f"wrote {args.report_dir}/timing_bakeoff.{{md,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
