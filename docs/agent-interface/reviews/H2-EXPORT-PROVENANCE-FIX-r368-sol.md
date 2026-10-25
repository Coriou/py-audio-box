# r368 — F1 public export provenance corroboration candidate

2026-10-25. **F1 implementation candidate; fresh independent review pending.**
Parent adoption and release remain **NO-GO**; legacy **CLOSED**. This report
claims bounded source/inert regression evidence only, not independent acceptance.

## Authority and exact pins

Own isolated worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-312bb29e`,
branch `t3/export-provenance-fix`. Initially clean at exact authorized base
`96a1b249c1b6a505e1d32c500adca10355115458`; fast-forwarded only this worktree to
candidate `23a682e7eb16e26fc11d121237870d0c577ca324`, whose implementation is
`fb1e8fe5473db70676aa25f794b9df33c7b3f214`.

F1 source/correction commit: `d6f66076fb2c8a5400a1c44e393d8e989e29df56`.
This report is a subsequent documentation-only commit. Read FULL root r367 and
r365 from `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/`
(they are absent from the candidate/source worktree), full candidate r366,
current publication validation, persistence, Core, reconciliation, owner/public
adapters and focused test contracts. No applicable ancestor/nested AGENTS found.
No original/root writes, push, parent adoption or delegation occurred.

## Narrow correction

`service/persistence.py:public_artifact` reuses `publication_record` /
`publication_intent`, existing canonical JSON identity and plan hashing. It
corroborates the independently served SQL artifact row against its retained
publication artifact, manifest hash, publication intent, producing job and frozen
context. Producer-shaped expected artifact binds row/artifact ID, succeeded job
and artifact list, operation, input hash, latest attempt/generation/owner/staging,
output asset IDs, project revision and resolved identity. Export provenance is
reconstructed from frozen source ID/hash, request settings and converter identity;
the manifest must match it canonically too. Prepared two-file intent manifests
also retain canonical comparison. Numeric representation divergence refuses.

`Core.artifact` performs that corroboration before existing asset integrity reads.
Public divergence refuses with HTTP 503 (`not_ready` through existing error
mapping), including after reopen. Retained-history reconciliation calls the same
helper for every public artifact row and records `public_artifact_mismatch`,
blocking/latching integrity through the existing diagnostics. It does not repair
any trusted/public copy or fabricate success. SQL mutations described here are
private inert test fixtures, not a claim that public callers can execute SQL.

No new store, scheduler, migration or public operation. Existing publication,
frozen-input, ownership, idempotency, ACK/reap and capacity mechanisms remain.
Installed availability is unchanged: existing installed families have no enrolled
export converter. The one-line r366 attribution correction names the
**ORCHESTRATOR**, not the user; historical test/failure receipts are unchanged.

Changed source/correction files: `service/core.py`, `service/persistence.py`,
`service/reconciliation.py`, `tests/service/test_audio_export.py`, and
`docs/agent-interface/reviews/H2-MANAGED-AUDIO-EXPORT-r366-sol.md`.

## Verification

All Python execution used CPython **3.14.7**, exact interpreter
`/private/tmp/pab-0c-venv/bin/python -I -B`. Both temporary runners assert its
version and install the audit hook **before application/test imports**, refusing
`subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect` with
`FORBIDDEN_REAL_OPERATION:<event>`. Inspected top-level/service conftests;
neither was activated. No broad pytest collection.

Focused export run: **17 tests PASS**, 9.541 s, log
`/tmp/pab-r368-focused.log`. Final coherent regression after the final source/test
changes: **69 tests PASS**, 22.228 s, `/tmp/pab-r368-final.log`:

```text
tests/worker/test_audio_export.py
tests/service/test_audio_export.py
tests/service/test_production_api.py
tests/clients/test_production_clients.py
tests/service/test_production_admission.py
tests/service/test_production_reconciliation.py
tests/worker/test_production_synthetic.py
tests/worker/test_engine_lifecycle.py
tests/service/test_engine_admission.py
tests/contracts/test_reference_roles.py
tests/contracts/test_legacy_controls.py
```

Runner `/tmp/pab-r368-check.py` inserts this exact root and each selected module's
parent into sys.path, uses importlib explicit file paths and unittest module
loading, and exits nonzero on any failure. Invocation is the exact interpreter,
runner path, then the module paths above in that order. No reference_publication
module or real child-dependent method was selected; all seven remain **BLOCKED**.
Existing speech/prepare compatibility is covered by the selected in-process
production API and injected engine lifecycle/contract regressions.

Nine new export tests: eight independently mutate ONLY `artifacts.metadata`
(converter reported build, settings rate, source hash, removed export provenance,
artifact ID, producing job ID, output asset IDs, integral-float generation).
Each first proves clean HTTP 200, then asserts HTTP 503 before close, explicit
`public_artifact_mismatch` reconciliation block, unchanged jobs/publications/intent
rows and unchanged tampered row, then latched integrity and HTTP 503 after reopen.
The ninth proves export → export produces two healthy retained artifacts and
healthy reconciliation/reopen with no converter registration. These are tiny
SQLite/ASGI/PCM fixtures with injected process/reap observations, not conversion
or containment qualification. All new tests passed on their first focused run;
final regression also passed. No test failure was hidden or relabelled.

Nine functions directly called from `tests/contracts/test_contracts.py`: all PASS:
`test_schema_and_all_fixtures`,
`test_exact_text_ids_defaults_and_source_control_coverage`,
`test_revision_idempotency`, `test_lifecycle_cancel_fence_publication`,
`test_device_policy_and_readiness`, `test_tick_rounding_and_immutability`,
`test_migration_and_worker_boundaries`,
`test_operation_requests_jobs_artifacts_and_error_mapping`,
`test_binary_fixture_boundary`.

`/tmp/pab-r368-contracts.py` uses the same guarded interpreter and explicit root.
It then runs the deterministic generator twice in memory, intercepting wave writes,
Path.write_text/read_bytes and mkdir. Both complete virtual directory maps equal
disk and each other, with identical generated name sets: **20 retained files,
19 generated, voice-catalogue.json preserved, zero fixture writes**. Disk map was
also compared afterward. Contract/generator log: `/tmp/pab-r368-contracts.log`.
`git diff --check` passed. No product fixtures/contracts were changed.

## Risks, limits and parent-owned gates

The common reader now refuses historical public artifacts whose retained producer
copies are absent or contradictory; there is no fallback or automatic repair.
This finite single-copy corroboration is not an adversarial-database authenticity
claim against coordinated replacement of every retained copy. Fresh Sol review,
then Astra gate, remain parent-owned; no self-adoption or self-review acceptance.

Actual converter pin, image corroboration, sealed registration, installed-family
enrollment and real conversion/loudness/process descendant qualification remain
**UNRUN**. Actual Linux/amd64, this Intel Mac Docker Desktop and OrbStack install
remain **UNRUN**. CUDA and native Metal are mandatory and **unqualified/UNRUN**.
No Docker/build/pull/load, network/listener, application child, FFmpeg, ML/hardware,
HELD/STT, operational cleanup, install/download or heavy 6 GiB waiver occurred.
Inert fixture temporary-directory teardown is ordinary test lifetime only.
Release **NO-GO**, legacy **CLOSED**.
