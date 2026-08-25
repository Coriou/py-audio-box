#!/usr/bin/env python3
"""Summarize a fal-sfx-bench run into markdown, CSV and pricing JSON.

Reads ``runs/<run_id>/records/*.json`` plus ``ledger.jsonl`` and writes
``report/summary.md``, ``report/summary.csv`` (long format: one row per case)
and ``report/pricing.json``. Price specs come from the sibling ``models.py``
registry; when it cannot be imported, embedded defaults matching the shared
spec are used instead (flagged in every output).
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

PRICE_FIELDS = ("unit", "per_second", "flat", "per_sample_multiplier",
                "published_text", "source_url")

# Fallback price specs (verified 2026-08-25, see local://sfx-bench-spec.md).
EMBEDDED_PRICING: dict[str, dict[str, Any]] = {
    "mirelo16": {
        "label": "Mirelo SFX 1.6",
        "endpoint_id": "mirelo-ai/sfx1.6/video-to-video",
        "vendor": "mirelo-ai",
        "schema_checked_at": "2026-08-25",
        "price": {
            "unit": "per_second_of_output", "per_second": 0.01, "flat": None,
            "per_sample_multiplier": True,
            "published_text": "$0.01 per second",
            "source_url": "https://fal.ai/models/mirelo-ai/sfx1.6/video-to-video",
        },
    },
    "kling-v2a": {
        "label": "Kling Video-to-Audio",
        "endpoint_id": "fal-ai/kling-video/video-to-audio",
        "vendor": "kling",
        "schema_checked_at": "2026-08-25",
        "price": {
            "unit": "flat_per_request", "per_second": None, "flat": 0.035,
            "per_sample_multiplier": False,
            "published_text": "$0.035 per video",
            "source_url": "https://fal.ai/models/fal-ai/kling-video/video-to-audio",
        },
    },
    "sonilo11": {
        "label": "Sonilo v1.1",
        "endpoint_id": "sonilo/v1.1/video-to-sound-effects",
        "vendor": "sonilo",
        "schema_checked_at": "2026-08-25",
        "price": {
            "unit": "per_second_of_output", "per_second": 0.009, "flat": None,
            "per_sample_multiplier": False,
            "published_text": "$0.009 per second of output and per sample",
            "source_url": "https://fal.ai/models/sonilo/v1.1/video-to-sound-effects",
        },
    },
    "hunyuan": {
        "label": "HunyuanVideo-Foley (baseline)",
        "endpoint_id": "fal-ai/hunyuan-video-foley",
        "vendor": "tencent",
        "schema_checked_at": "2026-08-25",
        "price": {
            "unit": "per_second_of_output", "per_second": 0.01, "flat": None,
            "per_sample_multiplier": False,
            "published_text": "$0.1 per 10 seconds",
            "source_url": "https://fal.ai/models/fal-ai/hunyuan-video-foley",
        },
    },
    "cassette-v2s": {
        "label": "CassetteAI Video SFX",
        "endpoint_id": "cassetteai/video-sound-effects-generator",
        "vendor": "cassetteai",
        "schema_checked_at": "2026-08-25",
        "price": {
            "unit": "per_second_of_output", "per_second": 0.2 / 60,
            "flat": None, "per_sample_multiplier": False,
            "published_text": "$0.2 per minutes of video processed",
            "source_url": "https://fal.ai/models/cassetteai/video-sound-effects-generator",
        },
    },
    "cassette-t2s": {
        "label": "CassetteAI Text-SFX (no video cond.)",
        "endpoint_id": "cassetteai/sound-effects-generator",
        "vendor": "cassetteai",
        "schema_checked_at": "2026-08-25",
        "price": {
            "unit": "flat_per_request", "per_second": None, "flat": 0.01,
            "per_sample_multiplier": False,
            "published_text": "$0.01 per generations",
            "source_url": "https://fal.ai/models/cassetteai/sound-effects-generator",
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_run_layout(run_dir: Path) -> tuple[str, Path]:
    """Return ``(run_id, records_dir)`` for a run directory."""
    run_id = run_dir.name
    records_dir = run_dir / "runs" / run_id / "records"
    if records_dir.is_dir():
        return run_id, records_dir
    nested = sorted((run_dir / "runs").glob("*/records"))
    if nested:
        found = nested[0].parent
        print(f"report: directory name does not match run id, "
              f"using run '{found.name}'", flush=True)
        return found.name, nested[0]
    raise FileNotFoundError(f"no records directory under {run_dir}")


def is_failed(record: dict[str, Any]) -> bool:
    return bool(record.get("error")) or record.get("status") == "failed"


def parse_case_name(name: str) -> tuple[str, str]:
    model, separator, clip = name.partition("__")
    if not separator:
        return "", name
    return model, clip


def load_records(records_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"report: skipping unreadable record {path.name}: {exc}",
                  file=sys.stderr, flush=True)
            continue
        record.setdefault("name", path.stem)
        records.append(record)
    return records


def load_ledger_counts(ledger_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not ledger_path.is_file():
        return counts
    for line_number, line in enumerate(
            ledger_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            print(f"report: skipping malformed ledger line {line_number}",
                  file=sys.stderr, flush=True)
            continue
        event = str(entry.get("event") or "unknown")
        counts[event] = counts.get(event, 0) + 1
    return counts


def load_pricing() -> tuple[dict[str, dict[str, Any]], str]:
    """Registry price specs from sibling models.py, else embedded defaults."""
    try:
        import models  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - any import failure falls back
        print(f"report: models.py unavailable "
              f"({exc.__class__.__name__}: {exc}); using embedded spec "
              "defaults", flush=True)
        return copy.deepcopy(EMBEDDED_PRICING), "embedded_defaults"
    registry = getattr(models, "MODELS", None)
    if not isinstance(registry, dict) or not registry:
        print("report: models.MODELS missing or empty; using embedded "
              "spec defaults", flush=True)
        return copy.deepcopy(EMBEDDED_PRICING), "embedded_defaults"
    specs: dict[str, dict[str, Any]] = {}
    for key, spec in registry.items():
        price = getattr(spec, "price", None)
        if dataclasses.is_dataclass(price) and not isinstance(price, type):
            price_map: dict[str, Any] = dataclasses.asdict(price)
        elif isinstance(price, dict):
            price_map = dict(price)
        else:
            price_map = {field: getattr(price, field, None)
                         for field in PRICE_FIELDS}
        entry = {field: price_map.get(field) for field in PRICE_FIELDS}
        entry.update({
            "label": getattr(spec, "label", key),
            "endpoint_id": getattr(spec, "endpoint_id", ""),
            "vendor": getattr(spec, "vendor", ""),
            "schema_checked_at": getattr(spec, "schema_checked_at", ""),
            "docs_urls": list(getattr(spec, "docs_urls", []) or []),
        })
        specs[key] = entry
    return specs, "models.py"


def fmt_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or
                                         math.isinf(value)):
            return ""
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def build_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        name = str(record.get("name") or "")
        model_key, clip = parse_case_name(name)
        rows.append({
            "case": name,
            "model": str(record.get("model") or model_key),
            "clip": clip,
            "status": "failed" if is_failed(record) else "ok",
            "total_wall_s": record.get("total_wall_s"),
            "queue_to_processing_s": record.get("queue_to_processing_s"),
            "processing_observed_s": record.get("processing_observed_s"),
            "output_duration_s": record.get("output_duration_s"),
            "estimated_cost_usd": record.get("estimated_cost_usd"),
            "endpoint_id": str(record.get("endpoint_id") or ""),
            "error": record.get("error"),
        })
    return rows


CSV_COLUMNS = ("case", "model", "clip", "status", "total_wall_s",
               "queue_to_processing_s", "processing_observed_s",
               "output_duration_s", "estimated_cost_usd")


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([fmt_number(row[column]) for column in CSV_COLUMNS])


def aggregate_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = buckets.setdefault(row["model"], {
            "model": row["model"],
            "endpoint_id": row.get("endpoint_id", ""),
            "ok": 0,
            "failed": 0,
            "walls": [],
            "estimated": 0.0,
        })
        if row["status"] == "ok":
            bucket["ok"] += 1
        else:
            bucket["failed"] += 1
        wall = row.get("total_wall_s")
        if row["status"] == "ok" and isinstance(wall, (int, float)) \
                and not isinstance(wall, bool):
            bucket["walls"].append(float(wall))
        estimated = row.get("estimated_cost_usd")
        if isinstance(estimated, (int, float)) and not isinstance(estimated,
                                                                  bool):
            bucket["estimated"] += float(estimated)
    aggregates = []
    for key in sorted(buckets):
        bucket = buckets[key]
        bucket["mean_wall_s"] = (statistics.fmean(bucket["walls"])
                                 if bucket["walls"] else None)
        bucket["median_wall_s"] = (statistics.median(bucket["walls"])
                                   if bucket["walls"] else None)
        aggregates.append(bucket)
    return aggregates


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def seconds_or_dash(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.2f} s"
    return "—"


def render_markdown(run_id: str, run_dir: Path, rows: list[dict[str, Any]],
                    aggregates: list[dict[str, Any]], ledger_counts: dict[str, int],
                    pricing: dict[str, dict[str, Any]], pricing_source: str,
                    manifest: dict[str, Any] | None) -> str:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] == "failed"]
    total_estimated = math.fsum(row["estimated_cost_usd"] for row in rows
                                if isinstance(row["estimated_cost_usd"],
                                              (int, float)))
    total_wall = math.fsum(row["total_wall_s"] for row in rows
                           if isinstance(row["total_wall_s"], (int, float)))
    total_output = math.fsum(row["output_duration_s"] for row in rows
                             if isinstance(row["output_duration_s"],
                                           (int, float)))

    lines = [
        f"# fal-sfx-bench report — {md_cell(run_id)}",
        "",
        f"Generated: {utc_now()}",
        f"Run dir: `{md_cell(str(run_dir))}`",
    ]
    if manifest is not None:
        lines.append(
            f"Manifest: `{md_cell(str(run_dir / 'manifest.json'))}` "
            f"({md_cell(len(manifest.get('cases', [])))} cases, "
            f"seed {md_cell(manifest.get('seed', '—'))})"
        )
    lines += [
        "",
        "## Run totals",
        "",
        "| metric | value |",
        "|---|---:",
        f"| cases | {len(rows)} |",
        f"| succeeded | {len(ok_rows)} |",
        f"| failed | {len(failed_rows)} |",
        f"| estimated spend | {usd(total_estimated)} |",
        f"| total wall time | {total_wall:.2f} s |",
        f"| total output audio | {total_output:.2f} s |",
    ]
    if ledger_counts:
        summary = ", ".join(f"{count} {event}"
                            for event, count in sorted(ledger_counts.items()))
        lines.append(f"| ledger events | {md_cell(summary)} |")
    lines.append("")

    lines += [
        "## Per-model aggregates",
        "",
        "| model | endpoint | ok | failed | mean wall | median wall | est $ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for bucket in aggregates:
        lines.append(
            f"| {md_cell(bucket['model'])} "
            f"| {md_cell(bucket['endpoint_id'])} "
            f"| {bucket['ok']} | {bucket['failed']} "
            f"| {seconds_or_dash(bucket['mean_wall_s'])} "
            f"| {seconds_or_dash(bucket['median_wall_s'])} "
            f"| {usd(bucket['estimated'])} |"
        )
    lines.append("")

    lines += [
        "## Pricing",
        "",
        f"Price specs source: `{pricing_source}`.",
        "",
        "| model | unit | per second | flat | published | source | schema checked |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for key in sorted(pricing):
        spec = pricing[key]
        # models.py-sourced entries flatten price fields; embedded defaults
        # nest them under "price" — accept both shapes.
        price = spec.get("price") or spec
        per_second = price.get("per_second")
        flat = price.get("flat")
        lines.append(
            f"| {md_cell(key)} | {md_cell(price.get('unit', '—'))} "
            f"| {usd(per_second) + '/s' if per_second is not None else '—'} "
            f"| {usd(flat) if flat is not None else '—'} "
            f"| {md_cell(price.get('published_text', '—'))} "
            f"| {md_cell(price.get('source_url', '—'))} "
            f"| {md_cell(spec.get('schema_checked_at', '—'))} |"
        )
    lines.append("")

    if failed_rows:
        lines += ["## Failures", ""]
        for row in failed_rows:
            lines.append(
                f"- **{md_cell(row['case'])}** ({md_cell(row['model'])}): "
                f"{md_cell(row.get('error') or 'unknown error')}"
            )
        lines.append("")
    else:
        lines += ["## Failures", "", "None.", ""]

    lines += [
        "## Case data",
        "",
        "Long-format per-case data lives in `summary.csv` "
        "(case, model, clip, status, timings, estimated cost).",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="work/sfx-bench/<run_id> directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"report: run directory not found: {run_dir}", file=sys.stderr)
        return 1
    try:
        run_id, records_dir = resolve_run_layout(run_dir)
    except FileNotFoundError as exc:
        print(f"report: {exc}", file=sys.stderr)
        return 1

    records = load_records(records_dir)
    if not records:
        print("report: no case records found", file=sys.stderr)
        return 1

    rows = build_rows(records)
    aggregates = aggregate_models(rows)
    ledger_counts = load_ledger_counts(records_dir.parent / "ledger.jsonl")
    pricing, pricing_source = load_pricing()
    manifest: dict[str, Any] | None = None
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            print(f"report: ignoring unreadable manifest: {exc}",
                  file=sys.stderr, flush=True)

    report_dir = records_dir.parent / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(rows, report_dir / "summary.csv")
    atomic_json(report_dir / "pricing.json", {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source": pricing_source,
        "models": pricing,
    })
    markdown = render_markdown(run_id, run_dir, rows, aggregates,
                               ledger_counts, pricing, pricing_source,
                               manifest)
    (report_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(f"report: wrote {report_dir / 'summary.md'}, "
          f"{report_dir / 'summary.csv'}, {report_dir / 'pricing.json'} "
          f"({len(rows)} cases, pricing source: {pricing_source})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
