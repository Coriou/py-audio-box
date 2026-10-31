# r373 — local publication-v2 implementation candidate

2026-10-25. **IMPLEMENTATION CANDIDATE; fresh independent Sol then Astra gates
remain parent-owned. No adoption or release acceptance. Release NO-GO; legacy
CLOSED.** The inert 1,152,044-byte export now publishes, downloads with verified
ranges/hash, reopens and replays on both fresh and explicitly migrated disposable
local roots. This is source evidence, not real conversion or installed completion.

## Authority, pins and retained work

Sole authorized writable checkout:
`/Users/ben/.txcode/worktrees/py-audio-box/t3-b413d79c`, branch
`t3/fix-export-publication-size`. Initial HEAD exactly
`381064e9772871e180a0c178b064b4972a6d27cf`; initial status contained exactly the
untracked r371 report and `tests/service/test_export_publication_size.py`.
Neither was reset or discarded. Source candidate commit:
**`052abeac6a352e5c6a844098f58342195f09bd3f`**. This report is a subsequent
report-only commit. Root HEAD was read-verified at
`96a1b249c1b6a505e1d32c500adca10355115458`; original source candidate
`t3-312bb29e` remained at `381064e9772871e180a0c178b064b4972a6d27cf`.
No root/original-candidate writes, merge, push, delegation or self-adoption.

Read FULL root r372 before edits, full retained r371, root r370/r369/r367/r365,
and candidate r366/r368. No applicable ancestor/nested AGENTS found. The memory
registry supplied the existing numeric canonical-identity caution only; decisions
and evidence here use current source and the normative r372 contract. No unresolved
format/authority question remained after reading that contract; no extra writer
or operational authority was inferred.

The r371 historical report is committed verbatim, retaining its then-pending,
then-uncommitted statements and attributed reproduction rather than rewriting
history as a success. SHA-256:
`ae3d558ac83ec7ec2f3797e5d35f6c63261488b6d1b95105d716d55de1daccce`.
Its prepared test file was completed/extended for this implementation. r371's
failed running/executing root was never opened, migrated, cleaned or repaired.

## Exact formats and admission

- publication-v1: SQLite user_version **1**, sole integer P1 revision **1** /
  `synthetic-core`, exact CHECK **1..1,048,576**.
- publication-v2: user_version **2**, sole integer P1 revision **2** / same scope,
  exact CHECK **1..37,748,736** (36 MiB).
- Cross-pairs, missing/partial v2 markers, wrong DDL, duplicate/unknown revisions,
  added publication indexes/triggers and detached float/bool revisions refuse.
  SQL inspection budgets are checked before retained schema/row materialization.
- Fresh unenrolled local Core defaults to v2. Fresh Core/P1 DDL and both markers
  commit together. A v2 root missing its catalogue refuses incomplete initialization.
  Existing exact v1 stays v1 with no constructor migration; v1 pre-P1 stays query-only.
- Installed/native trusted creation explicitly selects v1. Existing restricted
  constructors inspect a source-preserving private copy and refuse v2 before
  invoking their Core writer/native retained activation. No enrollment format,
  resource protocol, native/runtime revision or registration writer was changed.

Every new audio_export, including small output and new retry, requires v2.
`Core.submit_execution` gates immediately after duplicate lookup and before native
preallocation, resource demand/preflight/observation, transaction, job/event,
settings or reservation. Dispatcher preserves installed/unavailable-converter
precedence and replay before readiness/resolution. HTTP explicitly maps
`publication_format_migration_required` to **409**, with a safe stable message;
CLI/MCP retain canonical HTTP transport. Discovery omits unavailable export;
local v1 operation detail reports the migration restriction when converter
configuration otherwise permits export. Existing retained keys replay unchanged.

## Publication and consumer closure

Only an authenticated frozen audio_export operation/job/context/execution gets
the v2 ceiling, further bounded by its exact descriptor/profile. An artifact's
provenance tag alone cannot authorize a larger read. Speech and each prepare file
remain <=1 MiB. Literal single-output and prepared-reference-adoption-v1 binding
shapes are unchanged. Existing F1 canonical producer/publication/manifest/intent
corroboration remains, including numeric representation identity. Hash verification
streams 64 KiB chunks and retains no-follow, owner, inode, link-count, size and
fsync checks and intent -> files -> public commit ordering.

r372 consumer obligations, with implementation locations:

- `service/persistence.py`: immutable v1 declaration, distinct v2 declaration,
  exact marker+DDL detached selection and coherent live `store_format`;
  `publication_limit`, `publication_intent`, `publication_record`, `public_artifact`
  and `verify_file` carry trusted format and corroborated producer context.
- `service/core.py`: fresh coherent pair / existing format open; duplicate-first
  new-export gate; `_publish`, `_publication_barrier`, both links=2 checks and
  final links=1 check; public artifact and direct export asset reads. Barrier
  loads job/context before selecting the ceiling. Direct export reads find their
  bounded retained publication join, rather than trusting the public asset tag.
- `service/reconciliation.py`: `_collect` validates the live pair/DDL; all three
  retained public-artifact/work expansion sites pass its validated format.
  Offline callers that bypass the mutating constructor retain supported setup.
- `service/recovery.py`: `_INSTALLED_V1_TABLE_DDL` unchanged; a separate exact v2
  model is selected by marker+DDL in the shared detached row validator, including
  constraint/affinity replay. Detached documents receive the selected format.
  No historical user_version field was invented in source envelopes.
- Actual installed snapshot opens still require their coherent v1 pair. B2 archive
  snapshots retain their own historical user_version=1/schema model; no archive,
  certificate, abandon writer or B2 authority changed. Migration refuses B2 scope.
- `service/native_activation.py` keeps both explicit v1 model constructions and
  its v1 gate. Native runtime's 1 MiB reads/accounting remain unchanged. Its shared
  detached validator continues to accept the exact v1 model. No native-v2 support.
- `service/persistent_init.py` explicitly requests v1 at both trusted Core creation
  sites; resumed owned initialization checks v1 before Core; split fresh validation
  still demands its exact v1 model/pair. Installer resume is never migration.
- `host_install`, `daemon_helper`, `package_install` retain their existing validated
  initialization/activation paths; no v2 installed writer bypass was added.
  `install_child.registration_writer` remains unchanged and registration-only.
- Dispatcher new submission/retry, capability/detail, Core admission and the public
  errors/operations/schema enum are aligned. Both distribution manifests include
  the one narrow private migration module. No new store/scheduler/framework/API.

Only validated unenrolled local v2 reconciliation separates **160 MiB media-name
accounting** from **16 MiB SQL/JSON scalar accounting**, retaining 2048 entries and
its existing <=1-second scan deadline. Each hardlink name is charged. The intended
single-attempt arithmetic is 24 + 3*36 + 16 = 148 MiB. v1/installed/native retain
the existing combined accounting and constructor budget bounds; recovery, worker
speech, native and installation caps were not generally increased. Accumulated
history or deadline exhaustion still refuses. This is no latency qualification.

## Fixed private migration and finite exits

`python -m service.publication_format migrate-v1-to-v2 --root ABSOLUTE_CORE_ROOT --expect-v1`
implements only the explicit fixed transition. No target selector, force, repair,
SQL/policy/ownership override or public voice/API/MCP route. **It was not run
against an actual root.** Tests invoke its private function on disposable fixtures.

The supported shape is same-UID, existing private local root/DB/control lock with
no enrollment, native/renewal/installed/B1/B2/receipt/maintenance/unknown namespace.
It refuses dispatch settings/history/commands even when terminal/acked, active or
unreaped jobs, pending intent/aliases, nonterminal reservations and unhealthy
monitor/integrity. Eligible retained direct-Core speech and two-file prepare
history are preserved. Resource demands, when present, are corroborated against
terminal frozen producers without observations, hardware refresh, release or
reconstruction. No deletion of history makes a root eligible.

The existing process maintenance lease excludes live and uncertain owners.
Maintenance owns its SQLite/control/retained-owner-lock resources directly under
that lease; it does not construct or reserve Core recursively. Literal ancestor,
root, DB, lock and sidecar owner/mode/no-follow/inode checks are repeated; missing
files are not created or permissions repaired. Retained owner locks must be
available and remain held through the transition. A bounded WAL-aware private copy
is validated before opening original mode=rw. Original SQLite uses trusted_schema
OFF, FK ON, FULL, existing WAL, timeout=0, finite progress deadlines and no checkpoint
prerequisite, VACUUM, backup replacement or journal-mode change.

Only complete canonical Core/P1/catalogue and optional inventory/resource/dispatch
initializer groups are accepted. Full sqlite_master identity includes constraints,
implicit indexes and immutable catalogue triggers; retained SQL is never executed.
Inspection is bounded to 2048 rows / 16 MiB scalars. Integrity/FK checks, typed row
replay, retained documents, canonical copies and filesystem inventory precede DDL.

One BEGIN IMMEDIATE transaction revalidates, renames only publication_work to the
fixed absent transition name, creates canonical v2, copies explicit rowid plus all
six raw typed columns in rowid order, drops old work, updates exactly the sole
integer P1 marker and user_version in that transaction. No executescript spans the
transition. Post-validation compares the complete expected model and every typed
row/rowid, including all other tables and sqlite_sequence. COMMIT occurs once,
followed by namespace sync and pair/identity verification under exclusion.

Finite results:

- 0 / migrated: exact eligible v1 committed coherently; logical rows/media retained.
- 0 / already-v2: eligible v2 validated without schema/data mutation.
- 2 / refused with reason: scope/busy/dispatch/pending/unknown/budget condition.
- 2 / invalid-prestate with reason: mixed/unknown format, malformed schema/rows/joins.
- 3 / failed-rolled-back: pre-COMMIT failure with confirmed rollback; logical v1 retained.
- 4 / outcome-unknown: COMMIT may have happened, verification/sync/rollback/close
  uncertain; explicit reinspection required. No retry/downgrade or success claim.

Unconfirmed SQLite/private-copy/lock close retains the maintenance ownership record
and resources, fencing later Core/maintenance admission until process restart.
The controlled close-fault test separately proves its disposable handles closed
before releasing that test record; this is not a production cleanup/repair API.
Rollback promises logical preservation, not byte-identical SQLite pages/sidecars.

## Guarded execution and exact receipts

All Python execution used **CPython 3.14.7**:
`/private/tmp/pab-0c-venv/bin/python -I -B`. Runners assert the version and install
an audit hook before application/test imports blocking `subprocess.Popen`,
`os.system`, `socket.bind`, `socket.connect` with `FORBIDDEN_REAL_OPERATION:<event>`.
Root and each test-module parent are explicit own-worktree paths. Read both
conftests; neither activated. No broad pytest or listener/uvicorn fixture.

Final common coherent run: **97 PASS / 34.794 s**, raw log
`/tmp/pab-r373-final-bounded.log`, runner `/tmp/pab-r373-check.py`:

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
tests/service/test_export_publication_size.py
tests/service/test_publication_format.py
```

That is the existing **69**, **4** export-size cases and **24** format matrix
methods (including finite subcase matrices). After the final private refusal-label
alignment only, affected focused selection of the last two paths: **28 PASS /
7.830 s**, `/tmp/pab-r373-exits-final.log`. No common-path source changed afterward.

Coverage includes fresh and migrated 1,152,044-byte ASGI publication, actual CLI
range/hash download, unchanged source, succeeded/idle job, no executing capacity,
no quarantine/persistence fault, reconciliation/reopen/replay without converter;
v1 new export and retry HTTP409 with unchanged reservations, retained-key replay,
converter precedence, exact v1 speech and two-file prepare reads/migration,
raw row/rowid/sequence preservation, pending/running and dispatch-history refusal,
exact/mixed models, detached affinity/FK/numeric refusal, private scope/UID/names,
process/control/owner lock exclusion, SQL failure at each transition boundary,
post-COMMIT lost reply/reinspection and uncertain-close ownership retention.
Policy scalars exercise 1 MiB, +1, 36 MiB, +1 without ceiling media allocation;
speech/prepare stay bounded. Scalar accounting/SQL preflight and 160 MiB media
accounting are exercised without large media allocation. Final inert output is
~1.1 MiB. Installed/native compatibility evidence is exact models and source
creation/refusal paths, **not executed installation/enrollment/runtime qualification**.

Nine contract functions PASS, final log `/tmp/pab-r373-contracts-bounded.log`,
runner `/tmp/pab-r373-contracts.py`: schema/all fixtures; exact IDs/defaults/coverage;
revision/idempotency; lifecycle/cancel/publication; device/readiness; ticks/
immutability; migration/worker boundaries; operations/jobs/artifacts/errors;
binary fixture boundary. Two in-memory generations PASS: **20 retained / 19
generated / zero fixture writes**, equal complete byte maps/name sets with
voice-catalogue retained. `git diff --check` passed.

Failures and intermediate evidence remain truthful: first combined selection
`/tmp/pab-r373-final-first.log` ran 89 tests, reporting **11 failed subcases**,
all migration fixtures using macOS `/var` symlink paths (`Not a directory: 'var'`).
The no-follow refusal was correct; disposable test roots were changed to physical
paths, with no weaker product check. Corrected matrix: 17 PASS / 2.378 s. Later
focused 21 PASS / 6.572 s and 22 PASS / 2.906 s preceded additional coverage.
An earlier 97 PASS / 32.875 s preceded the missing-v2-marker/bounded-read hardening;
the 34.794-second run above includes that hardening. No failed run is relabelled.
All seven real-child reference-publication methods remain **BLOCKED / UNRUN**.
Fake CPU/MPS/CUDA labels and reap receipts are inert source observations only.

## Old binaries and parent-owned gates

Pinned old Core source explicitly accepts only user_version 0/1 and old persistence
only P1=1. Old recovery snapshot and installed/native models reject the v2 pair/DDL.
These are source proofs (`git show` of 381064e9), not an executed old-binary test.
Old Core may set WAL/FULL or touch namespace metadata before refusing; no claim of
zero filesystem effects. Crucially the unchanged old registration writer does
**not** check user_version/P1 revision. Therefore **not all old binaries universally
refuse v2**. Installed/native v2 creation/migration/enrollment remains closed; local
v2 lacks the enrollment needed by that writer. No actual old binary was launched.

No actual root conversion/migration, converter pin/image corroboration,
registration/enrollment, real FFmpeg conversion/loudness or descendant containment
qualification ran. Linux/amd64, this Intel Mac Docker Desktop and OrbStack:
**UNRUN**. CUDA and native Metal: **mandatory, unqualified/UNRUN**. No Docker,
build/pull/load, network/listener, application child, FFmpeg, ML/hardware, HELD/STT,
operational cleanup, install/acquisition/download or heavy 6 GiB waiver. Ordinary
disposable inert-fixture teardown only. No new resource measurement/waiver inferred.
Headless HTTP/CLI/MCP only; no UI or public migration API.

Fresh independent Sol review then Astra gate, parent adoption, actual converter
and installed-family enrollment and CPU/CUDA/Metal qualification remain parent-owned.
This candidate does **not** make the source vertical installed-complete.
**Implementation handed off for independent gate; release NO-GO; legacy CLOSED.**
