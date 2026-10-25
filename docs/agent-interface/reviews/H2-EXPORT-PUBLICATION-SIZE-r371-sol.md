# r371 — export publication size: pending Astra format design

2026-10-25. **BLOCKED on parent-owned Astra targeted design / explicit format
choice. F2 remains SOURCE NO-GO. No implementation or durable format change.**
Release NO-GO; legacy CLOSED. This is an uncommitted implementation handoff,
not a fix acceptance or completed product report.

## Authority and retained work

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-b413d79c`, branch
`t3/fix-export-publication-size`. Verified initially clean at exact base
`96a1b249c1b6a505e1d32c500adca10355115458`, verified source worktree
`/Users/ben/.txcode/worktrees/py-audio-box/t3-312bb29e` at
`381064e9772871e180a0c178b064b4972a6d27cf`, verified ancestry and fast-forwarded
ONLY this worktree to that commit. No subsequent commits. No root/source checkout
writes, push, merge adoption, delegation or self-adoption. No applicable AGENTS
found in candidate or relevant ancestor/root paths. Lightweight memory registry
search provided no relevant export/format decision; findings use current source.

Read root r365 brief, r367 review, r369 review, r370 gate and candidate r366/r368;
inspected publication producer/readers, persistence/DDL, reconciliation, recovery,
installation, native history, admission and export descriptors/test fixtures.

Parent response to the format question: **Request Astra design first. Do not
implement/migrate/rebuild durable format yet.** Parent is commissioning Astra.
This instruction supersedes proceeding with the proposed v2 choice. No durable
migration code, initializer, DDL, validator or product file was changed. No real
root migration is authorized by any future source design alone.

Pending uncommitted files:

- `tests/service/test_export_publication_size.py`: two architecture-independent
  descriptor tests and one prepared public success regression.
- This report. No other worktree edits.

## Actual inert F2 reproduction

Executed `/private/tmp/pab-0c-venv/bin/python -I -B /tmp/pab-r371-repro.py`.
Runner asserts CPython 3.14.7 and installs the audit hook BEFORE application/test
imports, refusing `subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect`
with `FORBIDDEN_REAL_OPERATION:<event>`. Exact own root and tests/service paths
are inserted explicitly. Inspected tests/conftest.py and tests/service/conftest.py;
neither activated, no broad pytest. The listener/uvicorn fixture was not used.

Reused `AudioExportTests` and its injected runtime. ASGI upload of mono PCM16,
24 kHz / 96,000 frames succeeded. Submitted 48 kHz stereo PCM24 export. Runtime
fabricated valid 192,000-frame output: **1,152,044 bytes**; descriptor allows
**1,156,102**. Waited for persistence fault, without changing any limit.
Raw receipt `/tmp/pab-r371-repro.log`:

```text
F2_REPRO output_bytes=1152044
reconciled=True quarantined=True phase=receipt error=PublicationQuarantine
persistence_fault=True resource_fault=False integrity_fault=False
job_state=running
reservation.state=executing
```

The injected PID1-style receipt and PCM establish source control flow only, not
conversion or descendant containment. Fixture teardown closes the owner and
ordinary disposable temporary resources; no operational cleanup was performed.
No publication succeeded. This confirms r370's previously source-derived repro.

## Current exact formats and consumers for Astra

Locations below refer to unchanged candidate 381064e9.

- `service/core.py:275-294`: PRAGMA user_version accepts only 0/1; 0 is fresh-only;
  initializes Core tables and stamps user_version=1. `:295-296` calls private P1
  initialization and makes pre-P1 roots query-only. Core revision is distinct
  from the private publication revision and enrollment revision.
- `service/persistence.py:13-41`: P1 marker exactly `(1,'synthetic-core')`;
  partial markers refuse. Existing validation checks column names, not full P1
  DDL identity. Fresh publication_work has this exact declaration:

```sql
CREATE TABLE publication_work(
          job_id TEXT PRIMARY KEY REFERENCES publications(job_id),
          binding TEXT NOT NULL, destination TEXT NOT NULL UNIQUE,
          temporary TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,
          byte_length INTEGER NOT NULL CHECK(byte_length BETWEEN 1 AND 1048576))
```

- `service/persistence.py:66-84`: durable verifier independently enforces 1 MiB,
  no-follow regular owned inode, link count/size/hash/fsync; reads bounded chunks
  but accumulates the complete file. `:259-314` intent validation independently
  caps every entry at 1 MiB; literal single-output versus explicit two-file
  prepared-reference-adoption-v1 binding is retained. `:317-326` retained record
  reader; `:329-372` F1 canonical public-artifact corroboration must remain closed.
- `service/core.py:637-653`: publication validates intent before inserting rows;
  `:674-710` barrier re-decodes intent and verifies final/alias inodes with the
  common verifier. Both links=2 and links=1 paths must receive the correct cap.
  `:765-779` public artifact uses the F1 producer corroboration.
- `service/reconciliation.py:28-58,118-136`: closed tables/columns adapter,
  schema fingerprint, exact catalogue DDL check; `:553-557` public artifact
  corroboration; `:647-651,679-685` retained intents/reference expansion.
  No general publication v2 validator exists. Existing P1 check is insufficient
  to authorize accepting arbitrary widened/invalid DDL.
- `service/recovery.py:781`: snapshot marker requires P1 revision 1 plus resource
  protocol `(1,2)` and user_version=1. `:2874-2899` canonical installed v1 DDL
  includes the exact old publication_work SQL above. `:2901-2945`
  _installed_detached_rows compares complete normalized token identity, including
  indexes/triggers, against an independently constructed model and replays row
  constraints in private memory. `:2977-2979` intent document validation;
  `:3000-3002` detached installed history requires revision 1. Do not replace
  this exact schema validation with column-only or permissive CHECK inspection.
- `service/native_activation.py:124-152`: native completed_history derives two
  schema models from _INSTALLED_V1_TABLE_DDL, invokes the detached row validator,
  and independently requires P1 revision 1. `:277` native historical output reads
  cap at 1 MiB. Native conversion availability remains closed; native speech
  caps must not be globally enlarged to accommodate container export.
- `service/persistent_init.py:81` create-only owned-synthetic initialization;
  `:150-185` resume/private validation and Core construction, no migration route.
  `:317` split initialization, `:473-502` split private-copy fresh validation
  explicitly fixes P1 revision 1 and user_version=1. Fresh v2 cannot simply be
  emitted by Core while leaving this expected fresh-format marker unchanged.
- `service/persistence.py:191-227`: optional B2 is separately versioned, exact
  DDL, constructors validate but never install. `service/recovery.py:1765-1784`
  plan_abandon installs B2 under offline exclusion only after predecessor
  snapshot/consumer checks. This is a precedent, not general migration authority.
- `service/voice_store.py:51-65`: token schema identity includes implicit indexes
  and extra triggers/indexes for owned catalogue objects; reusable comparison
  principle, not authority to change that catalogue format.
- `worker/audio_export.py:20-21,100-104,124-127`: source 24 MiB, output 36 MiB;
  bounded descriptor, 60-second source, profile/disk/frames checks. Existing
  maximum admitted duration/rate yields below 36 MiB; no 36 MiB allocation needed
  to test scalar durable ceilings. `:137-142` independently verifies output.
- `service/admission.py:44-76,128-146`: selected execution admission and output
  growth/PCM check already use the export descriptor. `:26` native-only 1 MiB
  accounting is a separate speech boundary, not an export budget.
- `service/dispatcher.py:319-330`: duplicate lookup first; new installed-family
  export refused. A v1 gate must refuse before job/reservation allocation rather
  than admit and storage-quarantine; replay of retained keys must remain first.
  `:621-630` storage errors latch fault/quarantine and retain capacity.
- `clients/cli.py:98-121`: verified content download uses actual total bytes,
  64 KiB ranges, verifies range metadata and whole SHA before exclusive publish.
- Existing version assumptions in tests: tests/service/test_persistence.py:54,359;
  detached invalid-DDL coverage in tests/service/test_installed_restart.py:348.
  All seven real-child test_reference_publication methods remain BLOCKED/UNRUN.

## Locks and migration authority gap

`service/core.py:90-155,230-256` process/thread ownership registry plus nonblocking
control.lock excludes live owner construction; constructor startup_guard runs
before DB initialization and retained native activation can inspect before P1.
A migration must not bypass these ownership/native-renewal fences by opening SQL
under an unrelated ad hoc flock.

`service/recovery.py:117-160` _locked_roots holds control/core/control.lock and
receipts/lane.lock, verifies private ownership and nonblocking exclusion; split
policy uses fixed-role locked_file and revalidates root identities.
`offline:163-215` validates a source-preserving private copy before opening original
SQLite writable, uses trusted_schema=OFF, FULL and bounded SQL progress.
BUT `policy is not None, write=True` explicitly raises
`installed_original_writer_required`; no privileged split-root migration route
is supplied. `:218-234` legacy writer scope requires owned-synthetic in both init
receipt and DB enrollment. Unenrolled standalone Core has no such receipt.

`service/install_child.py:98-167` dropped permanent UID/GID 10001 registration
writer accepts a verified inherited lock and exact prestate; explicitly **No DDL**,
Core constructor, receipt access or generic SQL. Reusing it for migration would
require a separately reviewed finite authority change, not a call-site tweak.
Existing B2 maintenance machinery is not authorization to broaden this writer.

## Proposed design — NOT APPROVED OR IMPLEMENTED

Question submitted before durable changes: retain Core user_version=1; recognize
named P1 v1 and v2 formats with exact corresponding DDL; fresh roots use P1 v2;
explicit offline opt-in v1→v2 transaction rebuilds only publication_work with
CHECK BETWEEN 1 AND 37748736 and changes P1 marker to revision=2. Preserve all
retained rows and identities byte-for-byte, including pending intents; check
complete predecessor schema/constraints/foreign keys, bounded rows/bytes and
retained intents before mutation. Unknown/mixed/partial schemas refuse with no
restamping. No downgrade. Old binaries reject revision 2 at unsupported_p1_format.
No implicit migration on open or installer resume.

Speech/prepare retain respective 1 MiB policies; only a canonically corroborated
audio_export producer/context/manifest/intent receives the bounded export limit.
A public artifact audio_export field alone is not cap authority. Verifier budget
must come from validated intent/operation, not caller-supplied unconstrained cap.
Retained-history and detached consumers need the selected explicit format too.
F1 canonical comparisons, ACK/reap, reservations and publication fences remain.

Astra decision needed: which explicit migration entrypoint/authority safely covers
standalone roots and owned-synthetic versus split installed/native roots, while
preserving existing writer refusal and exact full-format validation? The initial
question's phrase 'existing offline locks' is insufficient for split writes.
Also decide fresh installed/native format selection and historical v1/v2 schema
model selection; changing a global reference tuple would invalidate old history.
This report does not claim those compatibility gaps solved or reduce the product
ceiling to 1 MiB. No existing root is migrated by this lane.

## Prepared tests and receipts

New file tests/service/test_export_publication_size.py:

- ExportSizeDescriptorTests: admitted 1,152,044-byte PCM24 fixture fits the exact
  descriptor and passes owner-side verify_output; 36 MiB constant and >60-second
  descriptor refusal without allocating near-ceiling media.
- ExportPublicationSizeTests: prepared future regression for ASGI large export,
  succeeded/idle publication, public artifact/format/provenance, actual CLI range
  download verified against hash and 1,152,044-byte length, unchanged source,
  no executing reservation/persistence fault/quarantine, healthy retained-history
  reconciliation, reopen without converter and same-key replay/public read.
  **NOT RUN**: intentionally requires the pending F2 correction, not a pass.

Runner /tmp/pab-r371-boundaries.py explicitly loads only
ExportSizeDescriptorTests, same interpreter/audit prefix. First log
/tmp/pab-r371-boundaries.log: **1 PASS / 1 FAIL** (0.010 s): the test incorrectly
assumed fixture source rate 24 kHz for its 60-second negative case; the contract
fixture actually uses a different rate. Corrected the test to derive its duration
boundary from source.sample_rate, no source/limit change. Corrected log
/tmp/pab-r371-boundaries-corrected.log: **2 PASS / 0.011 s**. Subsequent import-only
cleanup avoids unittest collecting the imported existing fixture class; no change
to executed assertions. No claim of final regression or format compatibility.

Final 69-test allowlist plus new format/boundary tests and nine explicit contract
functions have **NOT RUN**: no final implementation exists. Avoid repeating broad
unchanged tests before the approved format correction. Still required after that
correction: old v1 retained speech/prepare/public intents preservation; exact old
DDL acceptance; widened-under-v1/mixed marker/unknown DDL/index/trigger refusal;
36 MiB scalar ceiling and >ceiling refusal without large allocation; old policy
caps; migration atomicity/failure retention and lock exclusion; fresh and existing
supported-root large public journey; version/downgrade refusal. Final test scope
must reflect the actual approved compatibility design.

No Docker/build/pull/load/network/listener/app child/FFmpeg/ML/hardware/HELD/STT,
installation/acquisition or heavy resource waiver. Inert WAV only ~1.1 MiB output.
Actual converter pin/enrollment/real conversion UNRUN; Linux, this Intel Mac
Docker Desktop and OrbStack UNRUN; CUDA/Metal mandatory and unqualified.
Fresh independent Sol then Astra source gates remain parent-owned. **F2 SOURCE
NO-GO pending design and implementation; release NO-GO; legacy CLOSED.**
