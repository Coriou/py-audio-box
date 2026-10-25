# r369 — independent Sol review of bounded export F1 correction

2026-10-25. **SOURCE GO for the bounded F1 correction; no blocking finding.**
The r367 single-public-copy provenance defect is closed by this candidate under
source inspection and the authorized inert checks. This is independent review,
not parent adoption. **Release NO-GO; legacy CLOSED.** Fresh parent-owned Astra
review remains required before adoption.

## Authority, pins and retained scope

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-312bb29e` verified as the
following consecutive chain:

- HEAD/report-only `381064e9772871e180a0c178b064b4972a6d27cf`.
- F1 correction `d6f66076fb2c8a5400a1c44e393d8e989e29df56`.
- Prior report `23a682e7eb16e26fc11d121237870d0c577ca324`.
- Export implementation `fb1e8fe5473db70676aa25f794b9df33c7b3f214`.
- Base `96a1b249c1b6a505e1d32c500adca10355115458`.

Candidate status was empty before and after checks; `git diff --check 96a1b24
HEAD` passed. HEAD changes only r368. Correction changes only Core, persistence,
reconciliation, export tests and r366 attribution. Checked filesystem ancestors
through both root and candidate paths and nested candidate/root docs AGENTS:
none applies. Root contains pre-existing untracked STT evidence and r365/r367;
only filenames were inventoried for unrelated evidence, and all were untouched.

Read full root r367/r365 and candidate r366/r368, AUDIO-EXPORT, the correction
diff, publication producer/intent/barrier/commit, canonical transport/hash and
current schema/operation/validation consumers, public owner/API/results readers,
retained reconciliation and relevant speech/prepare/export fixtures/tests.
Historical planning statements are not substituted for current source behavior.
The corrected r366 explicitly attributes converter authority to ORCHESTRATOR.

Sole persistent write: this report in the root checkout, mode 0600. No source
writes, runner/log files, commits, merges, delegation or self-adoption. Existing
inspected temporary r368 runners were executed directly with exact candidate
paths; fresh output is in this review's tool receipts, not a new log file.

## F1 closure and regression assessment

`service/persistence.py:329-372` corroborates the independent public row against
publication artifact, manifest hash, retained publication intent and frozen job
context. The expected artifact is reconstructed with producing job ID, latest
attempt/generation/owner/staging, input plan hash, output IDs, operation, project
revision and resolved identity. Export source ID/hash, request settings and
converter are reconstructed from the frozen plan, and compared to the manifest.
The artifact, publication copy and intent artifact all must equal that producer
shape by canonical JSON bytes. The prepared intent's complete manifest also gets
canonical comparison (`:368-369`). Integral-float substitutions cannot hide
behind Python equality on these copies.

`service/core.py:765-779` invokes this shared corroboration before reading output
assets. A schema-valid public converter-build mutation therefore cannot be
returned merely because the audio still hashes correctly. Owner checks remain
in `service/dispatcher.py:293-296`; results/save consumers also traverse the Core
artifact reader (`service/production.py:70`).

`service/reconciliation.py:553-557` invokes the same helper for every retained
artifact row and converts divergence to `public_artifact_mismatch`. The new
negative tests verify explicit reconciliation block, unchanged trusted rows,
unchanged tampered public row, latched integrity and refusal after reopen. There
is no repair, fallback provenance or new storage authority.

The original defect is directly covered by
`tests/service/test_audio_export.py:142-143`: change only the public converter
reported build, retaining publication/context/intent. The helper at `:111-140`
asserts initial HTTP 200, immediate HTTP 503 after mutation, reconciliation
quarantine and HTTP 503 after reopen. Seven other single-copy cases cover rate,
source hash, missing provenance, artifact ID, job ID, output IDs and numeric
representation. Clean export-to-export retained history remains healthy.

Common-reader compatibility was checked against the actual producer:
`service/core.py:629-632` creates the same artifact shape and `:643-646` retains
literal single-output intent or the explicit two-file prepared intent. Commit
at `:714-729` records succeeded state and a singleton artifact list. The added
reader does not reinterpret a legacy single-output intent as the prepared pair.
Existing single-output speech reads and retained prepare/save/reopen regressions
passed. A separate two-file prepare probe additionally exercised public read,
canonical intent checks, healthy reconciliation and public read after reopen.
Missing or contradictory historical producer copies deliberately refuse; this
is a compatibility boundary, not a migration or automatic repair claim. No
pre-P1 installation enrollment is asserted.

The review is bounded to independently changed retained copies and the affected
common reader. It does not claim authenticity against coordinated replacement
of every database copy, nor expand that into another adversarial framework.

## Fresh execution receipts

All application/test Python execution used **CPython 3.14.7** at
`/private/tmp/pab-0c-venv/bin/python -I -B`. Inspected runners assert that version,
insert the exact candidate root/module directories, and install an audit hook
before application/test imports refusing `subprocess.Popen`, `os.system`,
`socket.bind` and `socket.connect` with `FORBIDDEN_REAL_OPERATION:<event>`.
Read top-level and service conftests; neither was activated. No broad pytest.
The uvicorn/listener fixture was not used.

**Exactly one 69-test allowlist run: PASS, 23.937 seconds**, using the inspected
`/tmp/pab-r368-check.py` followed by these explicit paths, in order:

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

Fresh tool receipts end `Ran 69 tests in 23.937s / OK`. Expected entrypoint
refusal messages and fake CPU/MPS/CUDA backend labels are not real installation
or hardware execution. All eight public-copy mutation cases and the clean
export-to-export case passed. This coherent run is independent of r368's
historical 22.228-second run.

**Nine contract functions PASS**, directly invoked by the inspected
`/tmp/pab-r368-contracts.py` from `tests/contracts/test_contracts.py`:
`test_schema_and_all_fixtures`,
`test_exact_text_ids_defaults_and_source_control_coverage`,
`test_revision_idempotency`, `test_lifecycle_cancel_fence_publication`,
`test_device_policy_and_readiness`, `test_tick_rounding_and_immutability`,
`test_migration_and_worker_boundaries`,
`test_operation_requests_jobs_artifacts_and_error_mapping`,
`test_binary_fixture_boundary`.

The same process intercepted generator wave writes, Path.write_text/read_bytes
and mkdir. **Two in-memory generations PASS: 20 retained files, 19 generated,
identical virtual byte maps/name sets, voice-catalogue.json retained, zero fixture
writes**; physical fixture map also matched afterward. No subprocess determinism
test was run.

Additional guarded inline inert probe receipts:

```text
TWO_FILE_PREPARE public HTTP200, canonical producer, retained reconciliation healthy
intent_manifest_numeric REFUSED
intent_artifact_numeric REFUSED
missing_intent REFUSED
TWO_FILE_PREPARE corrected ASGI binding: before/reopen HTTP200, identical artifact, healthy integrity
```

The three refusal probes changed detached in-memory snapshots only. They tested
the prepared intent's integral-float manifest generation, intent artifact
generation, and absent intent. Every refusal was `public_artifact_mismatch`.
The public prepare used the existing injected synthetic runtime with tiny PCM,
local SQLite and ASGI, never a real application child or converter.

Harness failure retained honestly: the first extra probe's final HTTP assertion
failed because its ASGI app still referenced the closed Dispatcher after replacing
that Dispatcher. Its preceding public read/reconciliation/refusal assertions had
passed. A separate corrected probe explicitly rebuilt `create_voice_app` around
the reopened Dispatcher and passed with identical before/after artifacts. This
was a review harness binding error; the 69-test allowlist was not rerun or tuned.

All seven real-child methods of `test_reference_publication.py` remain
**BLOCKED / NOT RUN**, not passes. Their historical audit failures are not erased.
The extra inert probe does not qualify crash/fsync/real-child interruption or
process containment behavior.

## Parent-owned gates and disposition

**SOURCE GO** is restricted to this F1 correction atop the pinned export source
candidate. No blocker was found in canonical producer corroboration, literal
single-output intent handling, two-file prepare compatibility or the exercised
common-reader consumers. Parent must obtain the fresh independent Astra gate;
this review neither merges nor adopts the candidate.

Actual immutable converter pin/acquisition, trusted image corroboration, sealed
registration, installed-family enrollment and real short FFmpeg conversion,
loudness and descendant-reap qualification remain **UNRUN**. Linux/amd64,
this Intel Mac Docker Desktop and OrbStack actual install remain **UNRUN**.
CUDA and native Metal are mandatory and **unqualified/UNRUN**. HTTP/CLI/MCP
source parity does not establish installed runtime parity or full product
completion. Existing installed converter availability remains closed.

No actual Docker/build/pull/load, network/listener, application child, FFmpeg,
ML/hardware, HELD/STT content access, operational cleanup, install or download
occurred. Ordinary disposable inert fixture teardown only. Disk observed
5,252,420 KiB available, below the unchanged 6 GiB heavy gate; no waiver.
**Release NO-GO; legacy CLOSED.**
