#!/usr/bin/env python3
"""Stage 6b — mechanical run summary (timing, tiers, cost).

Human judgment (the verdict) is written separately in ``report/verdict.md``;
this script only aggregates what the artifacts already say.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    timing = load(args.run_dir / "validation" / "timing.json") or {}
    ledger_lines: list[dict[str, Any]] = []
    ledger_path = args.run_dir / "runs" / "ledger.jsonl"
    if ledger_path.exists():
        ledger_lines = [json.loads(line) for line in ledger_path.read_text().splitlines()]

    submissions = [entry for entry in ledger_lines if entry.get("event") == "submitted"]
    total_cost = sum(float(entry.get("estimated_cost_usd") or 0.0) for entry in submissions)

    lines = [
        "# sfx-bank-pipeline — run summary",
        "",
        f"Run dir: `{args.run_dir}`",
        "",
        "## Oracle timing gate (≤50 ms median, all contacts matched)",
        "",
    ]
    overall = timing.get("overall", {})
    lines.append(
        f"- winners: `{overall.get('winning_models', {})}`"
    )
    pooled = overall.get("pooled_median_abs_error_ms")
    lines.append(
        f"- pooled median over {overall.get('footstep_clips', 0)} footstep clips: "
        f"{pooled if pooled is not None else '—'} ms → "
        f"gate {'**PASS**' if overall.get('gate_pass') else '**FAIL**'}"
    )
    lines.append("")

    lines.append("## Strategy tiers + fal spend")
    lines.append("")
    lines.append("| clip | source | tiers | fal est |")
    lines.append("|---|---|---|---:|")
    tier_totals: Counter[str] = Counter()
    clip_costs: dict[str, float] = {}
    for entry in submissions:
        case = entry.get("case", "")
        if "__" in case:
            clip_costs[case.split("__")[1]] = (
                clip_costs.get(case.split("__")[1], 0.0)
                + float(entry.get("estimated_cost_usd") or 0.0))
    for plan_path in sorted((args.run_dir / "plan").glob("*.placement.json")):
        plan = json.loads(plan_path.read_text())
        counts = Counter(event["tier"] for event in plan["events"])
        tier_totals.update(counts)
        histogram = ", ".join(f"{t}×{c}" for t, c in sorted(counts.items())) or "—"
        lines.append(
            f"| {plan['clip']} | {plan['event_source']} | {histogram} "
            f"| ${clip_costs.get(plan['clip'], 0.0):.4f} |"
        )
    lines.append("")
    lines.append(f"Tier totals: {dict(sorted(tier_totals.items()))}")
    lines.append("")
    lines.append(f"Ledgered fal requests: {len(submissions)} · estimated total **${total_cost:.4f}**")
    lines.append("")
    lines.append("See `review/index.html` for the A/B players and `validation/timing.md`")
    lines.append("for the full oracle comparison matrix.")
    lines.append("")

    out_dir = args.run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.md"
    out_path.write_text("\n".join(lines))
    print(out_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
