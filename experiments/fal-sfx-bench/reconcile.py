#!/usr/bin/env python3
"""Reconcile fal-sfx-bench ledger request ids against a Fal billing CSV.

Joins ``runs/<run_id>/ledger.jsonl`` request ids to billed rows in a Fal
usage export (auto-detecting request-id/model/cost/currency columns),
then writes ``report/reconciliation.md`` with per-case billed-vs-estimated
deltas, per-model totals, unmatched CSV rows, ledger requests missing from
the CSV, and grand totals. Ambiguous column mappings degrade gracefully
unless ``--strict`` aborts.

Aggregated exports (daily per-endpoint rollups with no usable request-id
column) take a separate path automatically: rows are filtered to the
FAL_KEY credential, grouped by endpoint/app id, and compared against
ledger estimates summed per model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COST_CANDIDATES: tuple[str, ...] = ("cost", "amount", "price", "total",
                                    "usd", "gross_amount", "net_amount")
AGGREGATE_COST_CANDIDATES: tuple[str, ...] = (("gross_amount", "net_amount")
                                              + COST_CANDIDATES)

FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "request_id": ("request_id", "request id", "requestid"),
    "model": ("model", "endpoint", "endpoint_id", "app_id", "app"),
    "cost": COST_CANDIDATES,
    "currency": ("currency",),
}

REQUIRED_FIELDS = ("request_id", "cost")
AGGREGATE_REQUIRED_FIELDS = ("cost", "model")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


def detect_columns(
    headers: list[str],
    cost_candidates: tuple[str, ...] | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map logical fields to CSV headers; returns (mapping, ambiguous)."""
    normalized = {header: normalize_header(header) for header in headers}
    mapping: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    claimed: set[str] = set()

    for field, candidates in FIELD_CANDIDATES.items():
        if field == "cost" and cost_candidates is not None:
            candidates = cost_candidates
        ranked: list[tuple[int, str]] = []
        # Exact normalized matches first, in candidate priority order.
        for rank, candidate in enumerate(candidates):
            target = normalize_header(candidate)
            for header in headers:
                if header in claimed:
                    continue
                if normalized[header] == target:
                    ranked.append((rank, header))
        if not ranked:
            # Substring fallback (skips short tokens like "app" to avoid noise).
            for rank, candidate in enumerate(candidates):
                target = normalize_header(candidate)
                if len(target) < 4:
                    continue
                for header in headers:
                    if header in claimed:
                        continue
                    if target in normalized[header]:
                        ranked.append((rank + 100, header))
        if not ranked:
            continue
        best_rank = min(rank for rank, _ in ranked)
        best = sorted({header for rank, header in ranked
                       if rank == best_rank})
        if len(best) == 1:
            mapping[field] = best[0]
            claimed.add(best[0])
        else:
            ambiguous[field] = best
    return mapping, ambiguous


def parse_amount(raw: Any) -> float | None:
    """Parse "$0.035", "0.035", "1,234.56", "(1.25)" style cost cells."""
    if raw is None:
        return None
    text = str(raw).strip().replace("\u00a0", " ")
    if not text:
        return None
    for token in ("$", "€", "£", "¥", ",", " ", "USD", "usd", "Usd"):
        text = text.replace(token, "")
    negative = text.startswith("-")
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    text = text.lstrip("+-").strip()
    if not text or any(character not in "0123456789." for character in text):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def find_header(headers: list[str], *names: str) -> str | None:
    """Return the first header whose normalized form equals a name."""
    normalized = {normalize_header(header): header for header in headers}
    for name in names:
        header = normalized.get(normalize_header(name))
        if header is not None:
            return header
    return None


def request_column_all_empty(csv_path: Path, header: str) -> bool:
    """True when the request-id column exists but no data row fills it."""
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get(header) or "").strip():
                return False
    return True


def load_fal_key() -> str | None:
    """FAL_KEY from the environment, else a naive repo-root .env parse."""
    key = os.environ.get("FAL_KEY")
    if key:
        return key.strip()
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() != "FAL_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            return value
    return None


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")


def redact_label(label: str) -> str:
    """Mask everything after '@' in email-looking substrings.

    Applied before any auth_method_label reaches the markdown report;
    ``user@…`` keeps the local part but never the domain.
    """
    return EMAIL_PATTERN.sub(lambda match: match.group(0).split("@")[0]
                             + "@…", label)


def fmt_number(value: float | None) -> str:
    """Compact fixed-point rendering ("5", "4.765273"); em dash when None."""
    if value is None:
        return "—"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def resolve_run_layout(run_dir: Path) -> tuple[str, Path]:
    """Return ``(run_id, records_dir)`` for a run directory."""
    run_id = run_dir.name
    records_dir = run_dir / "runs" / run_id / "records"
    if records_dir.is_dir():
        return run_id, records_dir
    nested = sorted((run_dir / "runs").glob("*/records"))
    if nested:
        found = nested[0].parent
        print(f"reconcile: directory name does not match run id, "
              f"using run '{found.name}'", flush=True)
        return found.name, nested[0]
    raise FileNotFoundError(f"no records directory under {run_dir}")


def load_ledger(path: Path) -> dict[str, dict[str, Any]]:
    """Group ledger lines by request id."""
    requests: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return requests
    for line_number, line in enumerate(path.read_text().splitlines(),
                                       start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            print(f"reconcile: skipping malformed ledger line {line_number}",
                  file=sys.stderr, flush=True)
            continue
        request_id = str(entry.get("request_id") or "").strip()
        if not request_id:
            continue
        bucket = requests.setdefault(request_id, {
            "request_id": request_id,
            "case": None,
            "model": None,
            "endpoint_id": None,
            "estimated_cost_usd": None,
            "events": [],
        })
        for field in ("case", "model", "endpoint_id"):
            if entry.get(field) is not None:
                bucket[field] = entry[field]
        estimated = entry.get("estimated_cost_usd")
        if isinstance(estimated, (int, float)) and not isinstance(estimated,
                                                                  bool):
            bucket["estimated_cost_usd"] = float(estimated)
        bucket["events"].append({
            "ts": entry.get("ts"),
            "event": entry.get("event"),
        })
    return requests


def is_failed_record(record: dict[str, Any]) -> bool:
    return bool(record.get("error")) or record.get("status") == "failed"


def load_records(records_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(records_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"reconcile: skipping unreadable record {path.name}: {exc}",
                  file=sys.stderr, flush=True)
            continue
        name = str(record.get("name") or path.stem)
        records[name] = record
    return records


def ledger_status(events: list[dict[str, Any]]) -> str:
    names = {str(event.get("event") or "") for event in events}
    if "completed" in names:
        return "completed"
    if "failed" in names:
        return "failed"
    if "submitted" in names:
        return "submitted"
    return "unknown"


def read_billing_rows(csv_path: Path, mapping: dict[str, str]) -> list[
        dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, raw_row in enumerate(reader, start=2):
            request_raw = ""
            if "request_id" in mapping:
                request_raw = str(raw_row.get(mapping["request_id"]) or "")
            model_raw = ""
            if "model" in mapping:
                model_raw = str(raw_row.get(mapping["model"]) or "")
            cost_raw: Any = None
            if "cost" in mapping:
                cost_raw = raw_row.get(mapping["cost"])
            currency_raw = ""
            if "currency" in mapping:
                currency_raw = str(raw_row.get(mapping["currency"]) or "")
            rows.append({
                "row_number": row_number,
                "request_raw": request_raw.strip(),
                "request_key": request_raw.strip(),
                "model": model_raw.strip(),
                "cost": parse_amount(cost_raw),
                "cost_raw": "" if cost_raw is None else str(cost_raw).strip(),
                "currency": currency_raw.strip(),
            })
    return rows


def read_aggregate_rows(csv_path: Path, app_header: str | None,
                        cost_header: str | None,
                        fallback_header: str | None,
                        aux: dict[str, str | None]) -> list[
        dict[str, Any]]:
    """Read aggregated rows; labels are redacted before they are stored."""

    def cell(raw_row: dict[str, Any], header: str | None) -> Any:
        return raw_row.get(header) if header is not None else None

    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, raw_row in enumerate(reader, start=2):
            cost = parse_amount(cell(raw_row, cost_header))
            if cost is None:
                cost = parse_amount(cell(raw_row, fallback_header))
            label = str(cell(raw_row, aux["auth_method_label"]) or "").strip()
            rows.append({
                "row_number": row_number,
                "app": str(cell(raw_row, app_header) or "").strip(),
                "cost": cost,
                "quantity": parse_amount(cell(raw_row, aux["quantity"])),
                "unit": str(cell(raw_row, aux["unit"]) or "").strip(),
                "unit_price": parse_amount(cell(raw_row, aux["unit_price"])),
                "auth_method": str(
                    cell(raw_row, aux["auth_method"]) or "").strip(),
                "label": redact_label(label),
            })
    return rows


def ledger_endpoint_totals(path: Path) -> dict[str, dict[str, Any]]:
    """Sum per-endpoint ledger estimates; the first estimate per request wins.

    Ledger lines arrive chronologically per request, so keeping the first
    ``estimated_cost_usd`` preserves the submission-time advisory price
    instead of the output-duration-corrected value on the completed event.
    """
    estimates: dict[str, float] = {}
    request_endpoint: dict[str, str] = {}
    endpoint_models: dict[str, str] = {}
    if not path.is_file():
        return {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        request_id = str(entry.get("request_id") or "")
        endpoint = str(entry.get("endpoint_id") or "")
        if request_id and endpoint:
            request_endpoint.setdefault(request_id, endpoint)
        estimated = entry.get("estimated_cost_usd")
        numeric = isinstance(estimated, (int, float)) and not isinstance(
            estimated, bool)
        if request_id and numeric:
            estimates.setdefault(request_id, float(estimated))
        if endpoint and endpoint not in endpoint_models:
            endpoint_models[endpoint] = str(entry.get("model") or "")
    totals: dict[str, dict[str, Any]] = {}
    for request_id, endpoint in request_endpoint.items():
        bucket = totals.setdefault(endpoint, {
            "endpoint": endpoint,
            "model": endpoint_models.get(endpoint) or "",
            "requests": 0,
            "estimated": 0.0,
        })
        bucket["requests"] += 1
        bucket["estimated"] += estimates.get(request_id, 0.0)
    return totals


def match_endpoint(app_id: str,
                   endpoints: dict[str, dict[str, Any]]) -> str | None:
    """Match a CSV app id to a ledger endpoint id.

    Exact first, then a one-directional suffix fallback: the ledger
    endpoint id may carry an owner prefix that the exported app id omits.
    """
    if not app_id:
        return None
    if app_id in endpoints:
        return app_id
    for endpoint in sorted(endpoints):
        if endpoint.endswith(app_id):
            return endpoint
    return None


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def usd(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def delta_usd(value: float | None) -> str:
    return "—" if value is None else f"{value:+.4f} USD"


def describe_mapping(mapping: dict[str, str], ambiguous: dict[str, list[str]],
                     headers: list[str],
                     aggregate_mode: bool = False) -> list[str]:
    lines = ["| field | header | status |", "|---|---|---|"]
    for field in FIELD_CANDIDATES:
        if field in mapping:
            status = "detected"
            header = mapping[field]
        elif field in ambiguous:
            status = "ambiguous"
            header = ", ".join(md_cell(item) for item in ambiguous[field])
        else:
            if aggregate_mode:
                status = ("**missing**" if field in AGGREGATE_REQUIRED_FIELDS
                          else "not detected")
            else:
                status = ("not required" if field not in REQUIRED_FIELDS
                          else "**missing**")
            header = "—"
        lines.append(f"| {field} | {md_cell(header)} | {status} |")
    lines.append("")
    lines.append("Headers found: " + ", ".join(
        "`" + md_cell(header) + "`" for header in headers))
    return lines


def render_reconciliation(run_id: str, csv_path: Path,
                          mapping: dict[str, str],
                          ambiguous: dict[str, list[str]],
                          headers: list[str],
                          case_results: list[dict[str, Any]],
                          model_totals: list[dict[str, Any]],
                          unmatched_rows: list[dict[str, Any]],
                          missing_requests: list[dict[str, Any]],
                          no_request_cases: list[str],
                          totals: dict[str, Any]) -> str:
    lines = [
        f"# Billing reconciliation — {md_cell(run_id)}",
        "",
        f"Generated: {utc_now()}",
        f"Billing CSV: `{md_cell(str(csv_path))}` "
        f"({totals['csv_row_count']} data rows)",
        "",
        "## Column mapping",
        "",
    ]
    lines += describe_mapping(mapping, ambiguous, headers)
    if ambiguous:
        lines.append("")
        lines.append("WARNING: ambiguous column mapping — best guesses used; "
                     "rerun with --strict to abort instead.")

    lines += [
        "",
        "## Grand totals",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| billed total (entire CSV) | {usd(totals['csv_billed_total'])} |",
        f"| billed rows matched to ledger | {totals['matched_billed']:.4f} "
        f"({totals['matched_row_count']} rows) |",
        f"| estimated total (all ledger requests) | "
        f"{usd(totals['estimated_total'])} |",
        f"| estimated total (billed requests) | "
        f"{usd(totals['matched_estimated'])} |",
        f"| delta (matched billed − billed-request estimates) | "
        f"{delta_usd(totals['grand_delta'])} |",
        f"| CSV rows with unparseable cost | "
        f"{totals['unparseable_cost_rows']} |",
    ]

    lines += [
        "",
        "## Per-model totals",
        "",
        "| model | requests | billed | estimated | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in model_totals:
        lines.append(
            f"| {md_cell(bucket['model'])} | {bucket['requests']} "
            f"| {usd(bucket['billed'])} | {usd(bucket['estimated'])} "
            f"| {delta_usd(bucket['delta'])} |"
        )

    lines += [
        "",
        "## Per-case detail",
        "",
        "| case | model | request id | status | estimated | billed | delta |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    if case_results:
        for result in case_results:
            lines.append(
                f"| {md_cell(result['case'])} | {md_cell(result['model'])} "
                f"| {md_cell(result['request_id'])} "
                f"| {md_cell(result['status'])} "
                f"| {usd(result['estimated'])} | {usd(result['billed'])} "
                f"| {delta_usd(result['delta'])} |"
            )
    else:
        lines.append("| — | — | — | no ledger requests | — | — | — |")

    lines += [
        "",
        f"## CSV rows not matched to ledger ({len(unmatched_rows)})",
        "",
    ]
    if unmatched_rows:
        lines += ["| CSV row | request id | model | cost |", "|---:|---|---|---:|"]
        for row in unmatched_rows:
            lines.append(
                f"| {row['row_number']} | {md_cell(row['request_raw'])} "
                f"| {md_cell(row['model'])} "
                f"| {usd(row['cost']) if row['cost'] is not None else md_cell(row['cost_raw'])} |"
            )
    else:
        lines.append("None — every billed row maps to a ledger request.")

    lines += [
        "",
        f"## Ledger requests missing from CSV ({len(missing_requests)})",
        "",
    ]
    if missing_requests:
        for request in missing_requests:
            lines.append(
                f"- `{md_cell(request['request_id'])}` "
                f"({md_cell(request['case'] or '?')}, "
                f"{md_cell(request['model'] or '?')})"
            )
    else:
        lines.append("None — every ledger request appears in the CSV.")

    if no_request_cases:
        lines += [
            "",
            f"## Records without a ledger request id ({len(no_request_cases)})",
            "",
        ]
        lines += [f"- {md_cell(case)}" for case in no_request_cases]

    currencies = totals.get("currencies") or []
    lines += ["", "## Notes", ""]
    if "currency" in mapping:
        note = ("detected currency values: "
                + ", ".join(md_cell(code) for code in currencies))
        if currencies and all(code.upper() == "USD" for code in currencies):
            note += " — all USD."
        else:
            note += " — WARNING: non-USD amounts are summed as-is."
        lines.append(f"- {note}")
    else:
        lines.append("- Currency column not detected; all amounts assumed USD.")
    if "model" not in mapping:
        lines.append("- Model column not detected; per-model totals fall back "
                     "to ledger/record model keys.")
    lines.append("- Estimates are advisory; billed amounts are the source of "
                 "truth.")
    return "\n".join(lines) + "\n"


def render_aggregate_reconciliation(run_id: str, csv_path: Path,
                                    mapping: dict[str, str],
                                    ambiguous: dict[str, list[str]],
                                    headers: list[str],
                                    structure_rows: list[dict[str, Any]],
                                    model_totals: list[dict[str, Any]],
                                    excluded_rows: list[dict[str, Any]],
                                    totals: dict[str, Any],
                                    notes: list[str]) -> str:
    lines = [
        f"# Billing reconciliation — {md_cell(run_id)}",
        "",
        f"Generated: {utc_now()}",
        f"Billing CSV: `{md_cell(str(csv_path))}` "
        f"({totals['csv_row_count']} data rows)",
        "",
        "> **aggregated billing export** — request-level join impossible; "
        "per-endpoint totals.",
        "",
        "## Column mapping",
        "",
    ]
    lines += describe_mapping(mapping, ambiguous, headers, aggregate_mode=True)
    if ambiguous:
        lines.append("")
        lines.append("WARNING: ambiguous column mapping — best guesses used; "
                     "rerun with --strict to abort instead.")

    lines += [
        "",
        "## Billing structure",
        "",
        "| endpoint (app id) | quantity | unit | unit price | billed |",
        "|---|---:|---|---:|---:|",
    ]
    for row in structure_rows:
        lines.append(
            f"| {md_cell(row['app'])} | {row['quantity']} "
            f"| {md_cell(row['unit'])} | {row['unit_price']} "
            f"| {row['billed']} |"
        )

    lines += [
        "",
        "## Per-model billed vs estimated",
        "",
        "| model | billed | estimated | delta | delta % |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in model_totals:
        lines.append(
            f"| {md_cell(bucket['model'])} | {bucket['billed']} "
            f"| {bucket['estimated']} | {bucket['delta']} "
            f"| {bucket['delta_pct']} |"
        )

    lines += [
        "",
        f"## Rows from other credentials ({len(excluded_rows)})",
        "",
    ]
    if excluded_rows:
        lines += [
            "| CSV row | app id | billed | credential label |",
            "|---:|---|---:|---|",
        ]
        for row in excluded_rows:
            lines.append(
                f"| {row['row_number']} | {md_cell(row['app'])} "
                f"| {row['billed']} | {md_cell(row['label'])} |"
            )
    else:
        lines.append("None — every row was billed to this run's credential.")

    lines += [
        "",
        "## Grand totals",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| billed total, kept rows ({totals['kept_row_count']}) | "
        f"{usd(totals['kept_billed'])} |",
        f"| billed total, excluded rows "
        f"({totals['excluded_row_count']}) | "
        f"{usd(totals['excluded_billed'])} |",
        f"| estimated total (matched endpoints) | "
        f"{usd(totals['estimated_total'])} |",
        f"| delta (billed kept − estimated) | "
        f"{delta_usd(totals['grand_delta'])} |",
    ]

    lines += ["", "## Notes", ""]
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def run_aggregate(args: argparse.Namespace, run_id: str, records_dir: Path,
                  csv_path: Path, headers: list[str]) -> int:
    """Reconcile an aggregated (per-endpoint rollup) Fal billing export."""
    mapping, ambiguous = detect_columns(
        headers, cost_candidates=AGGREGATE_COST_CANDIDATES)
    if args.strict and "cost" not in mapping and "model" not in mapping:
        print("reconcile: aggregated export has neither a cost nor a model "
              "column; headers found: " + ", ".join(headers),
              file=sys.stderr)
        print("reconcile: aborting (--strict)", file=sys.stderr)
        return 1

    fallback_header = find_header(headers, "amount")
    if fallback_header == mapping.get("cost"):
        fallback_header = None
    aux = {
        "quantity": find_header(headers, "quantity"),
        "unit": find_header(headers, "unit"),
        "unit_price": find_header(headers, "unit_price"),
        "auth_method": find_header(headers, "auth_method"),
        "auth_method_label": find_header(headers, "auth_method_label"),
    }
    rows = read_aggregate_rows(csv_path, mapping.get("model"),
                               mapping.get("cost"), fallback_header, aux)

    warnings: list[str] = []
    fal_key = load_fal_key()
    key_id = fal_key.partition(":")[0].strip() if fal_key else ""
    filter_active = bool(key_id) and aux["auth_method"] is not None
    if not fal_key:
        warning = ("FAL_KEY unavailable (env and repo-root .env) — no "
                   "credential filter applied; every row is treated as ours.")
        print(f"reconcile: WARNING {warning}", file=sys.stderr)
        warnings.append(warning)
    elif aux["auth_method"] is None:
        warnings.append("No auth_method column detected — rows cannot be "
                        "filtered by credential; every row is treated as "
                        "ours.")

    groups: dict[str, dict[str, Any]] = {}
    excluded_rows: list[dict[str, Any]] = []
    kept_billed = excluded_billed = 0.0
    unparseable_kept = 0
    kept_row_count = 0
    for row in rows:
        if filter_active and key_id not in row["auth_method"]:
            excluded_rows.append(row)
            if row["cost"] is not None:
                excluded_billed += row["cost"]
            continue
        kept_row_count += 1
        bucket = groups.setdefault(row["app"], {
            "app": row["app"], "billed": 0.0, "billed_any": False,
            "quantity": None, "units": [], "unit_prices": []})
        if row["cost"] is not None:
            bucket["billed"] += row["cost"]
            bucket["billed_any"] = True
            kept_billed += row["cost"]
        else:
            unparseable_kept += 1
        if row["quantity"] is not None:
            bucket["quantity"] = (bucket["quantity"] or 0.0) + row["quantity"]
        if row["unit"] and row["unit"] not in bucket["units"]:
            bucket["units"].append(row["unit"])
        price_seen = row["unit_price"] not in bucket["unit_prices"]
        if row["unit_price"] is not None and price_seen:
            bucket["unit_prices"].append(row["unit_price"])

    if filter_active and rows and kept_row_count == 0:
        warning = ("credential filter matched no rows — the FAL_KEY id does "
                   "not appear in any auth_method value")
        print(f"reconcile: WARNING {warning}", file=sys.stderr)
        warnings.append(f"WARNING: {warning}")

    endpoint_totals = ledger_endpoint_totals(records_dir.parent
                                             / "ledger.jsonl")

    structure_rows: list[dict[str, Any]] = []
    model_buckets: dict[str, dict[str, Any]] = {}
    for app in sorted(groups):
        group = groups[app]
        structure_rows.append({
            "app": app or "(no app id)",
            "quantity": fmt_number(group["quantity"]),
            "unit": ", ".join(group["units"]) or "—",
            "unit_price": (", ".join("$" + fmt_number(price)
                                     for price in group["unit_prices"])
                           or "—"),
            "billed": (usd(group["billed"]) if group["billed_any"]
                       else "unparseable"),
        })
        endpoint = match_endpoint(app, endpoint_totals)
        info = endpoint_totals.get(endpoint) if endpoint else None
        model = info["model"] or info["endpoint"] if info else "(unmatched)"
        bucket = model_buckets.setdefault(model, {
            "model": model, "billed": 0.0, "billed_any": False,
            "estimated": 0.0, "estimated_any": False})
        if group["billed_any"]:
            bucket["billed"] += group["billed"]
            bucket["billed_any"] = True
        if info:
            bucket["estimated"] += info["estimated"]
            bucket["estimated_any"] = True

    model_totals: list[dict[str, Any]] = []
    ordered_models = sorted(model_buckets,
                            key=lambda name: (name == "(unmatched)", name))
    for model in ordered_models:
        bucket = model_buckets[model]
        delta = bucket["billed"] - bucket["estimated"]
        delta_pct = (delta / bucket["estimated"] * 100.0
                     if bucket["estimated"] else None)
        model_totals.append({
            "model": model,
            "billed": usd(bucket["billed"]) if bucket["billed_any"] else "—",
            "estimated": (usd(bucket["estimated"])
                          if bucket["estimated_any"] else "—"),
            "delta": (delta_usd(delta)
                      if bucket["billed_any"] and bucket["estimated_any"]
                      else "—"),
            "delta_pct": (f"{delta_pct:+.2f}%" if delta_pct is not None
                          else "—"),
        })

    excluded_display = [{
        "row_number": row["row_number"],
        "app": row["app"] or "(no app id)",
        "billed": (usd(row["cost"]) if row["cost"] is not None
                   else "unparseable"),
        "label": row["label"] or "—",
    } for row in excluded_rows]

    estimated_total = sum(bucket["estimated"]
                          for bucket in model_buckets.values())
    totals = {
        "csv_row_count": len(rows),
        "kept_row_count": kept_row_count,
        "kept_billed": kept_billed,
        "excluded_row_count": len(excluded_rows),
        "excluded_billed": excluded_billed,
        "unparseable_kept": unparseable_kept,
        "estimated_total": estimated_total,
        "grand_delta": kept_billed - estimated_total,
    }

    notes: list[str] = []
    if filter_active:
        notes.append("Credential filter: kept rows whose auth_method contains"
                     f" key id `{key_id[:8]}…`; other credentials appear"
                     " under 'Rows from other credentials'.")
    notes.extend(warnings)
    notes.append("Estimated amounts sum submission-time ledger entries per "
                 "matched endpoint; they are advisory — billed amounts are "
                 "the source of truth.")
    if unparseable_kept:
        notes.append(f"{unparseable_kept} kept rows had no parseable billed "
                     "amount.")

    report_path = args.report
    if report_path is None:
        report_path = records_dir.parent / "report" / "reconciliation.md"
    else:
        report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = render_aggregate_reconciliation(
        run_id, csv_path, mapping, ambiguous, headers, structure_rows,
        model_totals, excluded_display, totals, notes)
    report_path.write_text(markdown, encoding="utf-8")
    print(
        f"reconcile: wrote {report_path} — aggregated export: "
        f"{kept_row_count} kept rows ({usd(kept_billed)}), "
        f"{len(excluded_rows)} other-credential rows "
        f"({usd(excluded_billed)}), estimated {usd(estimated_total)}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="work/sfx-bench/<run_id> directory")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Fal billing/usage CSV export")
    parser.add_argument("--report", type=Path, default=None,
                        help="output markdown path "
                             "(default: <run-dir>/runs/<run_id>/report/"
                             "reconciliation.md)")
    parser.add_argument("--strict", action="store_true",
                        help="abort when column mapping is ambiguous or "
                             "required columns are missing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"reconcile: run directory not found: {run_dir}",
              file=sys.stderr)
        return 1
    if not csv_path.is_file():
        print(f"reconcile: billing CSV not found: {csv_path}",
              file=sys.stderr)
        return 1
    try:
        run_id, records_dir = resolve_run_layout(run_dir)
    except FileNotFoundError as exc:
        print(f"reconcile: {exc}", file=sys.stderr)
        return 1

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = [header for header in (reader.fieldnames or [])
                   if header is not None]
    mapping, ambiguous = detect_columns(headers)

    # Aggregated Fal exports roll up to one row per endpoint/credential
    # without request ids; reconcile per endpoint instead of per request.
    if "model" in mapping and (
            "request_id" not in mapping
            or request_column_all_empty(csv_path, mapping["request_id"])):
        return run_aggregate(args, run_id, records_dir, csv_path, headers)

    usable = all(field in mapping for field in REQUIRED_FIELDS)
    if ambiguous or not usable:
        print("reconcile: column mapping "
              + ("ambiguous: " + ", ".join(sorted(ambiguous))
                 if ambiguous else "incomplete: missing "
                 + ", ".join(field for field in REQUIRED_FIELDS
                             if field not in mapping)),
              file=sys.stderr)
        print("reconcile: headers found: " + ", ".join(headers),
              file=sys.stderr)
        guess = dict(mapping)
        for field, options in ambiguous.items():
            guess[field] = f"{options[0]} (guess)"
        print("reconcile: best-guess mapping: " + json.dumps(guess),
              file=sys.stderr)
        if args.strict:
            print("reconcile: aborting (--strict)", file=sys.stderr)
            return 1

    requests = load_ledger(records_dir.parent / "ledger.jsonl")
    records = load_records(records_dir)
    rows = read_billing_rows(csv_path, mapping)

    billed_by_request: dict[str, dict[str, Any]] = {}
    unmatched_rows: list[dict[str, Any]] = []
    csv_billed_total = 0.0
    unparseable_cost_rows = 0
    currencies: set[str] = set()
    matched_row_count = 0
    for row in rows:
        if row["cost"] is not None:
            csv_billed_total += row["cost"]
        else:
            unparseable_cost_rows += 1
        if row["currency"]:
            currencies.add(row["currency"])
        key = row["request_key"]
        if not key:
            unmatched_rows.append(row)
            continue
        bucket = billed_by_request.setdefault(key, {
            "cost": 0.0, "parsed": False, "count": 0})
        bucket["count"] += 1
        if row["cost"] is not None:
            bucket["cost"] += row["cost"]
            bucket["parsed"] = True
        if key not in requests:
            unmatched_rows.append(row)
        else:
            matched_row_count += 1

    case_results: list[dict[str, Any]] = []
    model_buckets: dict[str, dict[str, Any]] = {}
    matched_billed_total = 0.0
    estimated_total = 0.0
    matched_estimated_total = 0.0
    missing_requests: list[dict[str, Any]] = []
    no_request_cases: list[str] = []

    for request_id in sorted(requests):
        request = requests[request_id]
        case = str(request.get("case") or "")
        record = records.get(case, {})
        model = str(request.get("model") or record.get("model") or "unknown")
        record_estimated = record.get("estimated_cost_usd")
        if isinstance(record_estimated, (int, float)) and not isinstance(
                record_estimated, bool):
            estimated: float | None = float(record_estimated)
        else:
            estimated = request.get("estimated_cost_usd")
        billed_bucket = billed_by_request.pop(request_id, None)
        if billed_bucket is None:
            missing_requests.append(request)
            billed: float | None = None
        else:
            billed = billed_bucket["cost"] if billed_bucket["parsed"] else None
            if billed is not None:
                matched_billed_total += billed
        if estimated is not None:
            estimated_total += estimated
            if billed_bucket is not None:
                matched_estimated_total += estimated
        delta = (billed - estimated
                 if billed is not None and estimated is not None else None)
        case_results.append({
            "case": case or request_id,
            "model": model,
            "request_id": request_id,
            "status": (ledger_status(request["events"]) if not record
                       else ("failed" if is_failed_record(record)
                             else "ok")),
            "estimated": estimated,
            "billed": billed,
            "delta": delta,
        })
        bucket = model_buckets.setdefault(model, {
            "model": model, "requests": 0, "billed": 0.0,
            "billed_any": False, "estimated": 0.0})
        bucket["requests"] += 1
        if billed is not None:
            bucket["billed"] += billed
            bucket["billed_any"] = True
        if estimated is not None:
            bucket["estimated"] += estimated

    for name, record in sorted(records.items()):
        if not record.get("request_id"):
            no_request_cases.append(name)

    model_totals = []
    for model in sorted(model_buckets):
        bucket = model_buckets[model]
        model_totals.append({
            "model": model,
            "requests": bucket["requests"],
            "billed": bucket["billed"] if bucket["billed_any"] else None,
            "estimated": (bucket["estimated"] if bucket["estimated"] else None),
            "delta": ((bucket["billed"] - bucket["estimated"])
                      if bucket["billed_any"] else None),
        })

    totals = {
        "csv_row_count": len(rows),
        "csv_billed_total": csv_billed_total,
        "matched_row_count": matched_row_count,
        "matched_billed": matched_billed_total,
        "estimated_total": estimated_total,
        "matched_estimated": matched_estimated_total,
        "grand_delta": matched_billed_total - matched_estimated_total,
        "unparseable_cost_rows": unparseable_cost_rows,
        "currencies": sorted(currencies),
    }

    report_path = args.report
    if report_path is None:
        report_path = records_dir.parent / "report" / "reconciliation.md"
    else:
        report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = render_reconciliation(
        run_id, csv_path, mapping, ambiguous, headers, case_results,
        model_totals, unmatched_rows, missing_requests, no_request_cases,
        totals)
    report_path.write_text(markdown, encoding="utf-8")
    print(
        f"reconcile: wrote {report_path} — "
        f"{len(case_results)} ledger requests, {len(unmatched_rows)} "
        f"unmatched CSV rows, {len(missing_requests)} missing from CSV",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
