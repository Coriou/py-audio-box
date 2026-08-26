#!/usr/bin/env python3
"""Stage 7 — reconcile the run ledger against a Fal dashboard CSV export.

The Fal export is an aggregated rollup per endpoint (+ credential) with no
request ids, so — exactly like ``fal-sfx-bench/reconcile.py`` — comparison
happens per endpoint. CSV rows are filtered to this repo's FAL_KEY credential
when the export carries one; anything unmatched is reported as foreign /
pre-existing spend instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys

BENCH_DIR = Path(__file__).resolve().parents[1] / "fal-sfx-bench"
sys.path.insert(0, str(BENCH_DIR))
from reconcile import (  # noqa: E402  (bench parsing helpers, reused verbatim)
    AGGREGATE_COST_CANDIDATES,
    parse_amount,
    redact_label,
)

from common import utc_now  # noqa: E402

ENDPOINT_CANDIDATES = ("app", "endpoint", "endpoint_id", "model")
CREDENTIAL_CANDIDATES = ("credential", "credentials", "key_name")


def load_fal_key() -> str | None:
    import os

    key = os.environ.get("FAL_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("FAL_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def pick_column(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", header.lower()): header for header in headers}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def ledger_totals(ledger_path: Path) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    if not ledger_path.exists():
        return totals
    for line in ledger_path.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("event") != "submitted":
            continue
        bucket = totals.setdefault(entry.get("endpoint_id", "unknown"),
                                   {"est": 0.0, "requests": 0})
        bucket["est"] += float(entry.get("estimated_cost_usd") or 0.0)
        bucket["requests"] += 1
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True,
                        help="Fal dashboard usage export (aggregated)")
    args = parser.parse_args()

    headers, rows = read_csv_rows(args.csv)
    endpoint_column = pick_column(headers, ENDPOINT_CANDIDATES)
    cost_column = None
    for candidate in AGGREGATE_COST_CANDIDATES:
        cost_column = pick_column(headers, (candidate,))
        if cost_column:
            break
    if not endpoint_column or not cost_column:
        raise SystemExit(f"could not identify endpoint/cost columns in {headers}")

    credential_column = pick_column(headers, CREDENTIAL_CANDIDATES)
    fal_key = load_fal_key()
    key_prefix = fal_key[:8] if fal_key else None

    billed: dict[str, float] = {}
    foreign: list[tuple[str, float, str]] = []
    for row in rows:
        label = row.get(endpoint_column, "") or ""
        amount = parse_amount(row.get(cost_column)) or 0.0
        credential = (row.get(credential_column, "") if credential_column else "") or ""
        if key_prefix and credential and key_prefix not in credential:
            foreign.append((label, amount, redact_label(credential)))
            continue
        billed[label] = billed.get(label, 0.0) + amount

    estimates = ledger_totals(args.run_dir / "runs" / "ledger.jsonl")

    matched: list[tuple[str, dict[str, float], float]] = []
    used_keys = set()
    for endpoint, bucket in estimates.items():
        hit = next((name for name in billed if name and
                    (endpoint.split("/")[-1] in name or name in endpoint
                     or endpoint.rstrip("/").endswith(name))), None)
        if hit is not None:
            matched.append((hit, bucket, billed.pop(hit)))
            used_keys.add(hit)

    lines = [
        "# sfx-bank-pipeline — reconciliation",
        "",
        f"Generated {utc_now()} · CSV `{args.csv}` · ledger "
        f"`runs/ledger.jsonl` · credential `{key_prefix}…`",
        "",
        "| endpoint | requests | estimated | billed | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    total_est = total_billed = 0.0
    for name, bucket, billed_amount in sorted(matched):
        delta = billed_amount - bucket["est"]
        total_est += bucket["est"]
        total_billed += billed_amount
        lines.append(
            f"| {name} | {bucket['requests']} | ${bucket['est']:.4f} "
            f"| ${billed_amount:.4f} | {delta:+.4f} |"
        )
    other_billed = sum(amount for _, amount in billed.items())
    lines.append(f"| *(other endpoints on this credential)* | — | — "
                 f"| ${other_billed:.4f} | — |")
    lines.append(f"| **total** | **{sum(b['requests'] for _, b in estimates.items())}** "
                 f"| **${total_est:.4f}** | **${total_billed + other_billed:.4f}** | — |")
    if foreign:
        lines += [
            "",
            "## Foreign-credential rows excluded",
            "",
            "| endpoint | amount | credential |",
            "|---|---:|---|",
        ]
        for label, amount, credential in foreign:
            lines.append(f"| {label} | ${amount:.4f} | {credential} |")
    budget_note = (
        "**Within the $5 POC budget.**"
        if (total_est <= 5.0 and total_billed <= 5.0)
        else f"**Over budget check:** estimated ${total_est:.4f}."
    )
    lines += ["", budget_note, ""]

    out_dir = args.run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reconciliation.md"
    out_path.write_text("\n".join(lines))
    print(out_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
