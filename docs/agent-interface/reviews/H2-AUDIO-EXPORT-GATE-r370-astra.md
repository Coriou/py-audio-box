# r370 — Astra overall source gate: managed audio export

2026-10-25. **SOURCE NO-GO for adoption of the entire bounded vertical.**
One high-priority functional integration defect remains: admitted PCM24 exports
larger than 1 MiB cannot enter durable publication and instead latch a storage
fault. **r367 F1 is closed at source level** by the reviewed correction; that
closure does not close this separate size-contract break. Parent must not merge
this exact candidate under an overall GO. **Release NO-GO; legacy CLOSED.**

## Authority and verified pins

Source-readonly review of `/Users/ben/.txcode/worktrees/py-audio-box/t3-312bb29e`.
Verified consecutive ancestry:

- HEAD/report-only: `381064e9772871e180a0c178b064b4972a6d27cf`.
- F1 correction: `d6f66076fb2c8a5400a1c44e393d8e989e29df56`.
- Prior report-only: `23a682e7eb16e26fc11d121237870d0c577ca324`.
- Export implementation: `fb1e8fe5473db70676aa25f794b9df33c7b3f214`.
- Base: `96a1b249c1b6a505e1d32c500adca10355115458`.

Root checkout HEAD is the stated base. Candidate status was empty; base-to-HEAD
`git diff --check` passed. HEAD changes only r368; the prior report commit changes
only r366. Correction changes Core, persistence, reconciliation, export tests and
r366 attribution. Applicable ancestor AGENTS checks and nested candidate/root-docs
inventory found none. A lightweight memory registry search returned no relevant
export-review entries; conclusions use the inspected candidate and supplied reports.

Read FULL root r365/r367/r369 and candidate r366/r368/AUDIO-EXPORT, the source diff,
changed schema/operation/fixture/coverage/distribution contracts, and relevant
Core, admission, persistence, reconciliation, worker protocol/child/supervisor,
owner/public HTTP/CLI/MCP and prepare/results/save consumers. r366 now correctly
attributes converter authority to the **ORCHESTRATOR**.

Sole write: this root-checkout report, mode **0600**. No application imports,
tests, generators, code edits, commits, merge, delegation, Docker, network/listener,
application child, FFmpeg, ML/hardware, HELD/STT content access or cleanup. Shell
reads and git inspection only. No new runtime reproduction is claimed below.

## F2 — high: admitted exports exceed the unchanged durable publication ceiling

Locations in the pinned candidate:

- `worker/audio_export.py:100-104,124-127`: compute resampled frame/output budgets,
  allowing up to 36 MiB subject to the selected profile and 60-second source limit.
- `service/core.py:646-650`: derive the actual output byte length and validate the
  publication intent before inserting the durable publication/work rows.
- **`service/persistence.py:307-309`**: every intent entry still requires
  `1 <= byte_length <= 1048576`, otherwise `publication_work_mismatch`.
- **`service/persistence.py:39`**: the SQL CHECK independently retains that same
  1 MiB ceiling. **`:66-68`**: the durable file verifier also refuses larger files
  with `publication_length_limit`.
- `service/persistence.py:89-103` and `service/dispatcher.py:621-630`: the intent
  exception is a `PublicationQuarantine`, latching publication storage failure,
  dispatcher persistence fault and quarantine, rather than normal media refusal.

Concrete source-derived repro, **NOT EXECUTED in this lane**:

1. Reuse the existing inert `AudioExportTests` setup and injected runtime in
   `tests/service/test_audio_export.py`, including its qualified fixture profile
   and explicit budget. Upload `audio(rate=24000, channels=1, width=2,
   frames=96000)` through the existing ASGI upload helper: a four-second source,
   192,044 bytes with the test helper's classic WAV header.
2. Submit `{asset_id: uploaded_id, rate: 48000, channels: 2, loudness: null}` with
   a fresh idempotency key. The source and requested output fit the existing
   fixture profile and export admission bounds; no converter change is needed.
3. The existing injected runtime computes 192,000 output frames and writes valid
   PCM24 stereo: **1,152,044 bytes**, exceeding **1,048,576**. Its output remains
   below the export descriptor's **1,156,102-byte** computed ceiling. Core's
   independent PCM/header/hash/duration policy therefore permits these bytes.
4. `_publish` builds the intent with `byte_length=1152044` and calls
   `publication_intent`; the fixed 1 MiB check necessarily raises
   `PublicationQuarantine('publication_work_mismatch')` before durable insertion.
   The dispatcher takes its storage-fault/quarantine branch, preventing success
   and ordinary completion/ACK/capacity release in that session. No public master
   is committed. This is a legitimate request, not database tampering.

This breaks a common short delivery export, not merely a theoretical maximum.
The tests' default source is only 2,400 frames at 24 kHz (`:18-22,73-76`); their
48 kHz stereo result is 28,844 bytes, so the existing success/parity/provenance
checks do not cross the old publication ceiling. r369's small two-file prepare
probe likewise cannot establish the new export size envelope.

Required finite correction: reconcile export admission/output policy with durable
intent validation, SQL format and file verification across fresh AND existing
supported roots. Preserve bounded verification, old speech/prepare policies,
retained intents and publication fences. Changing only the Python limit or fresh
CREATE TABLE leaves existing roots' SQL CHECK incompatible; no silent old-root
restamping or generic migration authority is implied. Retain an explicit supported
format/compatibility decision. Add a focused inert >1 MiB export publication,
public read/download, reopen/replay/reconciliation case and relevant ceiling
refusal coverage. Recheck the affected common consumers; no broad unchanged test
campaign or new framework is requested. A documented reduction of scope would
require a changed product brief and truthful pre-admission limits, not merely
relabeling these valid requests as storage failures.

## Rest of the bounded source assessment

F1 correction: `service/persistence.py:329-372` reconstructs the expected public
artifact from the frozen producing plan, job, last attempt and retained manifest,
then canonically compares the public row, publication and intent copies. Export
source/hash/settings/converter are bound to frozen producer context. Prepared
intent manifests receive canonical comparison too. `service/core.py:765-779`
uses this before serving artifact assets; `service/reconciliation.py:553-557`
uses the same helper and blocks divergent public rows without repair. This closes
the original single-public-copy build mutation and avoids Python numeric-equality
acceptance in the compared retained copies. No coordinated-all-copies threat model
is needed to assess this correction.

The expected shape matches the current producer at `service/core.py:629-632` and
its literal single-output / explicit prepared-pair intent at `:646-650`.
`Production.results` reads through `Core.artifact` and `save` consumes those
results (`service/production.py:62-80`). No additional common-reader regression
was identified by this source inspection. Missing or contradictory historical
producer evidence intentionally refuses; this is not an auto-repair or pre-P1
migration promise. F2 is also enforced by that shared intent reader, so merely
bypassing the initial publishing check would not resolve retained-history reads.

The public request and frozen plan are closed, with owned managed input and
explicit settings/converter identity. HTTP is canonical; CLI `audio export` and
MCP `speech_audio_export` use the existing typed operation transport. The manifest
schema's discriminated speech/export branches preserve reference traversal;
`source_asset_id` is explicitly walked. Existing verified range download uses
actual total bytes. Both distribution manifests include the new modules/fixture.
Idempotent replay precedes readiness/source resolution, retry retains the frozen
context, and submission/reservations remain in existing Core transactions.

The worker independently binds converter registration to principal/profile and
trusted supervisor image identity (`worker/runtime_supervisor.py:335-357`),
passes an owned source descriptor, and uses closed FFmpeg argv and private stage
output (`worker/audio_export.py:162-203`). Export output verification is distinct
from existing speech/prepare mono PCM16 policy. The existing supervisor drains
children before durable reap receipts, and dispatcher publication/ACK/release
still depend on those receipts. No second scheduler, caller-selected executable,
public path, fake voice identity or synthetic export success path was introduced.
Actual converter and descendant behavior remains unqualified.

Installed gating is adequate for the declared **source integration seam**:
`service/dispatcher.py:323-330,348` refuses new installed-family export and omits
its availability; `service/api.py:164,215` filters capabilities and reports
`unavailable_converter`. The installed supervisor entrypoint supplies no converter
registration (`worker/runtime_supervisor.py:564`). Static MCP tool declarations
explicitly say availability depends on configured service; they do not enroll a
converter. Fixture zero hashes are explicitly INERT/unqualified, not production
pins. The coverage ledger and AUDIO-EXPORT document do not claim installed
completion. This truthful gate does not excuse the admitted source-path F2 defect.

## Evidence limits and finite next product step

No tests ran in r370. Prior report evidence remains attributed, not re-executed:
r367 records 60 PASS and nine contract functions but SOURCE NO-GO for F1; r368
records the correction and nine added export cases. r369 records one fresh
**69 PASS / 23.937 s**, nine contract functions, deterministic in-memory generation
(**20 retained / 19 generated / zero fixture writes**), a healthy two-file prepare
public read/reopen and three canonical intent refusals. Its corrected ASGI-binding
probe follows a documented harness failure. Those receipts support F1 closure;
they do not cover the newly identified >1 MiB boundary. All **seven real-child
publication tests remain BLOCKED**, not passes.

First resolve F2 and obtain a fresh bounded source gate; parent alone owns merge
and adoption after GO. After adoption, prioritize one usable installed CPU journey:
select and corroborate the real immutable converter image/executable/runtime
closure, seal registration and narrowly enroll the installed family; then qualify
managed upload or generated speech -> PCM24 master -> verified public download,
including >1 MiB, reopen/replay, cancellation and descendant reap. Tie this to
clean-clone installation and management with the full existing public controls:
profiles/discovery, immutable voice prepare/results/save and generation,
jobs/list/events/wait/cancel/retry and artifact/asset retrieval over HTTP/CLI/MCP.
Finish that concrete installed workflow before more abstract plumbing. No UI,
public STT, stitch or acquisition expansion is part of this step.

Real converter pin/image corroboration, sealed enrollment, conversion/loudness and
descendant qualification are **UNRUN**. Actual Linux/amd64, this Intel Mac Docker
Desktop and OrbStack are **UNRUN**. CUDA and native Metal remain mandatory and
**unqualified/UNRUN**; no GPU or Apple Silicon hardware authority was supplied.
Last supplied disk observation is **5,252,420 KiB**, approximately 5 GiB, below the
unchanged **6 GiB heavy DISK gate**. It was not refreshed here; no RAM measurement
or waiver is inferred. These are parent-owned operational gates, not authority to
execute them in this review. **Overall SOURCE NO-GO pending F2; release NO-GO;
legacy CLOSED.**
