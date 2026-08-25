#!/usr/bin/env python3
"""Summarize auto-sfx analyze/compare runs into a markdown stats table.

Usage (inside the container):
    python /scripts/summarize_runs.py RUN [RUN ...]

Each RUN is a directory name under /work/auto-sfx/runs/ (analyze layout:
events.json + pose.jsonl + diagnostics.json at top level, or compare layout:
compare/<method>/... plus compare-summary.json).

Reports, without ground truth:
* pose coverage        fraction of decoded frames where a person was detected
* events / left / right, mean score
* step-interval stats  median, p90, fraction inside the plausible [0.3, 1.2] s band
* diagnostics histogram (top codes)
* cross-method agreement for compare runs: fraction of the smaller method's
  events that have a counterpart of the same foot within 67 ms in the other
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path("/work/auto-sfx/runs")
TOL_S = 0.067
PLAUSIBLE = (0.3, 1.2)


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _rows(run_dir: Path):
    return [json.loads(l) for l in open(run_dir / "pose.jsonl")][1:]


def _intervals(times):
    ds = sorted(b - a for a, b in zip(sorted(times), sorted(times)[1:]))
    if not ds:
        return None
    n = len(ds)
    med = ds[n // 2]
    p90 = ds[min(n - 1, int(0.9 * n))]
    plausible = sum(1 for d in ds if PLAUSIBLE[0] <= d <= PLAUSIBLE[1]) / n
    return med, p90, plausible


def _v2_metrics(run: Path):
    """Flat (P, R, bias_s, med_abs, cat3) from a v2 metrics.json, or None."""

    p = run / "metrics.json"
    if not p.is_file():
        return None
    m = _load(p)
    if "existence" not in m:
        return None
    t = m.get("timing") or {}
    return (
        m["existence"].get("precision"),
        m["existence"].get("recall"),
        t.get("bias_s"),
        t.get("median_abs_err_s"),
        t.get("catastrophic_gt3frame"),
    )


def summarize_analyze(run: Path) -> str:
    doc = _load(run / "events.json")
    events = doc.get("events") or []
    video = doc.get("video") or {}
    times = [e["time_s"] for e in events]
    n_l = sum(1 for e in events if e.get("foot") == "left")
    n_r = sum(1 for e in events if e.get("foot") == "right")
    n_obs = sum(1 for e in events if e.get("observable", True))
    cov = "n/a"
    try:
        rows = _rows(run)
        usable = sum(1 for r in rows if not r.get("lost", True))
        cov = f"{usable}/{len(rows)}"
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    score = (
        sum(e.get("score", 0.0) for e in events) / len(events)
        if events else 0.0
    )
    med, p90, plausible = _intervals(times)
    iv_s = (
        f"{med:.2f}s / {p90:.2f}s / {plausible:.0%}"
        if med is not None else "—"
    )
    diag_path = run / "diagnostics.json"
    if diag_path.exists():
        counts = {}
        for d in _load(diag_path).get("diagnostics", []):
            counts[d["code"]] = counts.get(d["code"], 0) + 1
        diags = [f"{k}={v}" for k, v in sorted(counts.items(),
                                               key=lambda kv: -kv[1])[:3]]
    top = ", ".join(diags)
    line = (
        f"| {run.name} | {doc.get('backend', '?')}/{doc.get('mode', '?')} "
        f"| {cov} | {len(events)} (obs {n_obs}) | {n_l}/{n_r} "
        f"| {score:.2f} |"
    ) + (f" | {iv_s} | {top or '—'} |")
    v2 = _v2_metrics(run)
    if v2 is not None:
        p, r, bias, medabs, cat3 = v2
        fmt = lambda x: "—" if x is None else x
        line += (
            f" || P={fmt(p)} R={fmt(r)} bias={fmt(bias)}s"
            f" med|e|={fmt(medabs)}s cat>3f={fmt(cat3)}"
        )
    return line

def _events_of(run: Path, method: str):
    p = run / "compare" / method / "events.json"
    if not p.exists():
        return []
    return _load(p)["events"]


def _match_fraction(a, b):
    if not a:
        return "—"
    hits = sum(
        1 for e in a
        if any(o["foot"] == e["foot"] and abs(o["time_s"] - e["time_s"]) <= TOL_S
               for o in b))
    return f"{hits / len(a) * 100:.0f}%"


def summarize_compare(run: Path) -> list[str]:
    methods = ["velocity", "pelvis", "hybrid"]
    evs = {m: _events_of(run, m) for m in methods}
    lines = []
    for m in methods:
        r = run / "compare" / m
        if not (r / "events.json").exists():
            continue
        lines.append(summarize_analyze(r))
    agree = " / ".join(
        f"v∩p {_match_fraction(evs['velocity'], evs['pelvis'])}"
        f" v∩h {_match_fraction(evs['velocity'], evs['hybrid'])}"
        f" p∩h {_match_fraction(evs['pelvis'], evs['hybrid'])}")
    lines.append(f"| {run.name} *agreement* | | | | | | | {agree} |")
    return lines


def main(argv):
    print("| run | backend/mode | coverage | events | L/R | mean score | "
          "interval med/p90/in-band | top diagnostics |")
    print("|---|---|---|---|---|---|---|---|")
    for name in argv:
        run = RUNS / name
        if (run / "compare-summary.json").exists():
            for line in summarize_compare(run):
                print(line)
        elif (run / "events.json").exists():
            print(summarize_analyze(run))
        else:
            print(f"| {name} | MISSING | | | | | | |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
