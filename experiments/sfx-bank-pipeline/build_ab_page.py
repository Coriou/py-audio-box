#!/usr/bin/env python3
"""Stage 6a — per-clip A/B review page (Mirelo original vs bank render).

Same review conventions as ``experiments/fal-sfx-bench/build_review.py``
(identical frames, two-pass −16 LUFS audio) plus a per-event decision table:
label, retrieval top-k, strategy tier, DSP ops, and timing error against the
auto-sfx contacts. Output: ``review/index.html`` (self-contained).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parents[1] / "fal-sfx-bench"
sys.path.insert(0, str(BENCH_DIR))
from build_review import _CSS  # noqa: E402 — identical styling as the benchmark


def money(value: Any) -> str:
    return "—" if value is None else f"${float(value):.4f}"


def seconds(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f} s"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def tier_histogram(events: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for event in events:
        counts[event["tier"]] = counts.get(event["tier"], 0) + 1
    return ", ".join(f"{tier}×{count}" for tier, count in sorted(counts.items())) or "—"


def event_rows(plan: dict[str, Any], errors_by_event: dict[str, float]) -> str:
    if not plan["events"]:
        return "<p><em>No discrete events.</em></p>"
    rows = [
        "<table><tr><th>event</th><th>t (s)</th><th>label</th>"
        "<th>top candidate</th><th>score</th><th>tier</th><th>ops</th>"
        "<th>gap (s)</th><th>gain</th><th>vs contact</th></tr>"
    ]
    for event in plan["events"]:
        candidate = event.get("candidate") or {}
        error = errors_by_event.get(event["id"])
        error_text = "—" if error is None else f"{error:+.1f} ms"
        rows.append(
            f"<tr><td>{event['id']}</td>"
            f"<td>{event['time_s']:.3f}</td>"
            f"<td>{event['label']} ({event['label_score']:.2f})</td>"
            f"<td>{candidate.get('rel_path', '—')}</td>"
            f"<td>{candidate.get('score', '—')}"
            f"{' (a' + format(candidate.get('audio_cos'), '+.2f') + ' / t' + format(candidate.get('text_cos'), '+.2f') + ')' if candidate else ''}</td>"
            f"<td><strong>{event['tier']}</strong></td>"
            f"<td>{', '.join(event.get('ops', [])) or '—'}</td>"
            f"<td>{seconds(event.get('available_gap_s'))}</td>"
            f"<td>{(event.get('gain_db') if event.get('gain_db') is not None else '—')} dB</td>"
            f"<td>{error_text}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def render_clip_section(clip: str, plan: dict[str, Any], loudness: dict[str, Any],
                        timing_entry: dict[str, Any],
                        clip_cost_usd: float | None) -> str:
    best_model = plan["oracle_model"]
    source = plan["event_source"]
    stats = timing_entry.get("models", {}).get(best_model)
    gate = "n/a"
    if stats:
        gate = (
            f"matched {stats['matched']}/{stats['contacts']} · median "
            f"{stats['median_abs_error_ms']} ms · {'PASS' if stats['gate_pass'] else 'FAIL'}"
        )
    out_i = loudness.get("output_i")
    mode = "linear" if loudness.get("linear") else "dynamic"
    cost_text = money(clip_cost_usd) + (" (fal)" if clip_cost_usd else " ($0 fal)")

    return f"""
<section>
  <h2>{clip} <span class="muted">· events from {source} · oracle {best_model}</span></h2>
  <p class="meta">oracle-vs-contacts: {gate} · tiers: {tier_histogram(plan['events'])}
     · fal cost this clip: {cost_text} · bank mix loudnorm: {mode}, output {out_i} LUFS</p>
  <div class="pair">
    <figure>
      <figcaption>A — Mirelo 1.6 original (untouched)</figcaption>
      <video controls preload="none" src="{clip}.mirelo16.mp4"></video>
    </figure>
    <figure>
      <figcaption>B — SFX-bank render ({len(plan['events'])} placed events)</figcaption>
      <video controls preload="none" src="{clip}.bank.mp4"></video>
    </figure>
  </div>
  {event_rows(plan, {})}
</section>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    review_dir = args.run_dir / "review"
    timing = load_json(args.run_dir / "validation" / "timing.json") or {"clips": {}}

    # Cost per clip from the ledger (SA3 generations are the only fal spend).
    costs: dict[str, float] = {}
    ledger_path = args.run_dir / "runs" / "ledger.jsonl"
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            entry = json.loads(line)
            case = entry.get("case", "")
            if entry.get("event") == "submitted" and "__" in case:
                clip = case.split("__")[1]
                costs[clip] = costs.get(clip, 0.0) + float(entry.get("estimated_cost_usd") or 0.0)

    sections = []
    for plan_path in sorted((args.run_dir / "plan").glob("*.placement.json")):
        clip = plan_path.stem.replace(".placement", "")
        bank_video = review_dir / f"{clip}.bank.mp4"
        mirelo_video = review_dir / f"{clip}.mirelo16.mp4"
        if not bank_video.exists() or not mirelo_video.exists():
            continue
        plan = json.loads(plan_path.read_text())
        loudness = load_json(review_dir / f"{clip}.bank.loudness.json") or {}
        sections.append(render_clip_section(
            clip, plan, loudness,
            timing.get("clips", {}).get(clip, {}),
            costs.get(clip),
        ))

    total_cost = sum(costs.values())
    header = f"""
<header>
  <h1>SFX-bank foley pipeline — A/B review</h1>
  <p>A: Mirelo 1.6 raw soundtrack (benchmark run 20260825-165347), untouched except
     the shared loudnorm pass. B: retrieval + placement render from
     work/sfx-bank/template using vision/V2A hybrid event timing.
     Total fal spend this run: <strong>{money(total_cost)}</strong>.</p>
</header>
"""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>sfx-bank-pipeline A/B</title>
<style>{_CSS}</style></head>
<body>{header}{''.join(sections)}
<footer><p class="muted">Generated by experiments/sfx-bank-pipeline/build_ab_page.py</p></footer>
</body></html>
"""
    out_path = review_dir / "index.html"
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(sections)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
