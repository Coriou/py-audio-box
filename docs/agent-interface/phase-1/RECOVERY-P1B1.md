# P1B1 — create-only enrollment and offline bounded recovery planning

Implementation candidate, awaiting separate Sol-low review. No self-GO, integration acceptance, installed qualification or activation claim.

Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-e310284d`, branch `t3/p1b1-enrollment-recovery`, baseline `75bc71c64eff6a131599ba2d680d20a978ef0762`. Baseline was clean. Parent-supplied predecessor: integrated A2 `1602331`, source equal to reviewed `6723528`, 39 focused passed in 8.63s and 279 passed / 18 skipped full review. Those are predecessor receipts, not reruns claimed here. Read P1 implementation decision including new-installation rules, death authority, B1 allowlist, interfaces and complete exit matrix; PRODUCT, implementation start/protocol, CONTRACTS 1.3.0, A1/A2 reports/reviews and R1 authority; inspected actual Core/persistence/reconciliation/Dispatcher/Admission/ResourceStore/DispatchStore and Journal/supervisor seams. No applicable AGENTS.md found.

## Exact scope and records

Only these eight authorized paths change:

- `service/persistent_init.py` (new): create-only four-role synthetic initialization, private initialization record, explicit matching completion and predecessor initialization.
- `service/recovery.py` (new): bounded offline inspection, registration, immutable plan persistence/verification and unconditional unavailable apply; read-only startup enrollment validation.
- `scripts/p1-operator.py` (new): fixed read-only Docker host projection and authenticated supervisor registration; structured local recovery calls.
- `service/persistence.py`: exact B1 schema initializer/column definitions.
- `service/reconciliation.py`: parent-approved B1 adapter only: exact columns, existing bounded traversal/fingerprint, closed record validation and typed references. No A2 epoch change.
- `service/dispatcher.py`: explicit startup/configuration gate and closed persistent activation only.
- `tests/service/test_recovery.py` (new): disposable-root E1 regression/fault tests.
- This report.

No worker/Core/resource implementation, public contracts, dependencies, packaging, legacy, ledger or existing reviews change. Public contract remains **1.3.0**, Core `user_version=1`, resource protocol **2**, private P1 revision **1**.

Additive tables in existing Core SQLite:

```text
p1_installation(id, revision, scope, record, digest)
p1_registrations(id, record, digest)
p1_recovery_plans(id, record, digest)
```

Registration and plans are insert-only through shipped functions. Plan IDs are random immutable IDs; digests use canonical UTF-8 JSON/SHA256. There is no B2 resolution/death-proof/epoch table. A2 recognizes these tables, rejects unknown columns/fields/references/digests, and follows explicit job/dispatch/publication references into canonical documents. Scalar metadata, scope strings and prose are not asset references. No cleanup eligibility or new authority is produced.

## Initialization and startup interfaces

`initialize(roots, scope, profiles, budget, complete=None, fault=...)` accepts exactly distinct owned `control`, `staging`, `receipts`, `ipc` directories. No ancestor nesting, role aliases or symlink volume roots. All must be empty for a new installation, including absence of old DBs, journal, keys, markers and unexplained files. Core lives at `control/core`, preserving A2's exact Core namespace.

The exclusive 0600 receipts-side `p1-init.json` is first. It binds random deployment identity, exact daemon/context and named-volume identities (name/driver/creation identity per role), profile/budget digest, private revision, synthetic scope and local role-directory identity. Its private seed fixes key/journal bytes for matching completion; seed/key material is never emitted. Exclusive per-file creation intents precede exclusive files. File fsync and directory/parent barriers precede subsequent steps. Missing files with an existing creation intent quarantine; keys/journal are never reminted over a missing previously attempted file. Partial differing bytes remain untouched. Core/R1/A2/B1 initialize coherently, then the final enrollment marker is exclusive and parent-synced.

Default rerun refuses every fragment. Explicit completion requires the exact initialization-record digest and exact scope/configuration/role identities; it can finish matching pre-Core files or a fully coherent empty database stage. Ambiguous partial Core schema/authority quarantines; no schema reconstruction, old-root stamping or migration. Already enrolled installations refuse reinitialization even when keys/journal/markers are missing.

```text
python -m service.persistent_init new --request PRIVATE_REQUEST
python -m service.persistent_init complete --request PRIVATE_REQUEST --initialization-digest EXACT_DIGEST
```

Initialization request has exactly `roots`, `scope`, `profiles`, `budget`. Scope has exactly `daemon`, `context`, `volumes`; volume roles each have exactly `name`, `driver`, `created`. One canonical `cpu-synthetic` profile and existing finite-budget validation remain mandatory.

Host registration checks authenticated supervisor hello against the requested journal lane/boot and observes immutable image IDs, full container IDs, started incarnation, exact volume roles, private PID namespace and restart-no topology. All containers are enumerated, including stopped containers; every consumer mounting any exact named volume must be registered. Unknown/missing consumers or another daemon/context/volume identity refuse. Docker output is narrowly projected; no environment, credentials or unrelated raw Docker receipt is returned. Registration retains old incarnations. It holds Core ownership; the worker may hold its journal lock during authenticated registration, so registration instead repeats a bounded journal-digest stability check. Offline inspect/plan require both normal Core and journal locks and never remove a lock. Host inspect/plan additionally refuse running/restarting consumers or nonzero runtime PIDs; these observations still supply no death authority.

```text
python scripts/p1-operator.py register --request PRIVATE_HOST_REQUEST
python -m service.dispatcher --persistent-registration EXACT_REGISTRATION_ID --persistent-config PRIVATE_CONFIG
```

Host request has exactly `roots`, `scope`, `registration`; config has exactly `profiles`, `budget`. Normal runtime imports no host Docker client and receives no socket/daemon access. Startup validates enrollment/configuration digest and authenticated current lane/boot before constructing Core. Persistent work remains activation-closed in B1, including submit/replace/cancel and the execution loop when an enrolled root is opened without persistent options. Existing un-enrolled diagnostic Dispatcher mode remains supported. C1 must qualify the actual installed initialization/start topology and container association; this source work does not claim it.

## Inspect, plan and refusal interfaces

```text
python -m service.recovery inspect --request PRIVATE_REQUEST
python -m service.recovery plan --request PRIVATE_REQUEST
python -m service.recovery apply --request PRIVATE_REQUEST
python scripts/p1-operator.py inspect --request PRIVATE_HOST_REQUEST
python scripts/p1-operator.py plan --request PRIVATE_HOST_REQUEST
python scripts/p1-operator.py apply --request PRIVATE_HOST_REQUEST
```

Packaged inspect request has exactly `roots`, `scope`; plan additionally requires `observation`. Host commands obtain that observation through fixed Docker operations, not supplied executable input. No shell, arbitrary argv proxy, stop/wait/remove/reset command or service socket mount is implemented. Apply raises `P1B2_apply_unavailable` before reading its request or performing any action, including with a policy/digest supplied programmatically.

`inspect`, `plan`, `verify_plan`, `register` are trusted offline Python seams. Inspection uses a read-only SQLite transaction including committed WAL, query-only mode, existing exclusive lock files, exact supported schema, aggregate row/byte checks before materialization and a SQLite progress deadline. Limits: 2,048 rows/entries, 16 MiB plus bounded journal/private-file reads, one-second filesystem inventory and two-second SQL deadline; refusal on incompleteness. It returns exact affected IDs; job/attempt/generation/deadline, dispatch/binding/runtime/receipt digest, reservation/demand/monitor/fault, publication/inventory, journal/lane and version diagnostics. It does not open or complete R1/A2 observation epochs, checkpoint/delete WAL, repair metadata or publish output.

Plans bind deployment/revision, logical source-state digest, structured host-observation digest, exact jobs/unacked dispatches/publications, finite count/byte bound, closed abandonment policy name, created/expiry (maximum 300 seconds), and `apply_available=false`. Proposal bytes themselves are excluded from source authority, including their duplicate reference labels; otherwise creating a plan would invalidate itself. Every actual source row, diagnostics, schema, filesystem inventory and journal identity remains bound. No wildcard or caller deletion path exists. A changed source, digest, scope, registration, container incarnation/state, missing key/journal, unknown consumer or occupied lock refuses verification. Unknown reference kinds/schema or foreign inventory prevent plan creation; read-only inspection still reports recognized diagnostic defects. A host observation digest is not death proof or authentication by itself. B1 cannot authorize a resource reset, release, ACK, epoch transition, budget change or cleanup.

## Evidence and remaining gates

Tests use real temporary directories, exclusive files, SQLite/WAL and flock plus deterministic syscall failure hooks. Docker projection tests are explicitly **E1 mocks**; no actual host inspection, container campaign or installed C1/E2 qualification was run. Temporary synthetic pytest data only; no private audio/models/network work. No `scripts/run-e2e-test.sh`, cloud/STT/Voicebox, push/PR/deploy or external checkout/subagent.

Development findings corrected: missing-key refusal initially surfaced OSError rather than the test's narrower ValueError expectation; private-file identity comparison initially included read-induced atime changes; plan source fingerprints were adjusted to exclude the proposal's own duplicate reference labels. Failures remain development evidence, not final acceptance.

Existing isolated `/tmp/pab-0c-venv`, no installs; all commands use bytecode/cache disabled:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_recovery.py
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_recovery.py tests/service/test_persistence.py tests/service/test_dispatcher.py tests/service/test_admission.py tests/service/test_core.py
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service tests/release
```

Development affected run: **273 passed in 99.34s**, `/tmp/p1b1-affected.log`; source was still being refined, so not the frozen final gate. First complete run: **324 passed, 18 skipped in 109.11s**, `/tmp/p1b1-final-full.log`, superseded by the enrolled-default/live-consumer gates and their regressions. Focused run after those two gates: **46 passed in 2.68s**, `/tmp/p1b1-final-focused.log`; final full run also includes the additional unknown-inventory plan-refusal test. One short focused development invocation overlapped the tail of the first affected run; it has no acceptance claim. Final full validation runs alone, with only owned disposable synthetic roots.

Final full result on the final source/test tree: **326 passed, 18 skipped in 107.41s, exit 0**, `/tmp/p1b1-final-full2.log`. This includes all **47 B1 tests** plus the full prior service/release suite. `git diff --check` passed. No installed skip is counted as qualification. Source/test files were not edited during this final run; only this report was updated afterward. The local commit carrying this report is the review candidate; exact SHA is supplied in the handoff. A1's global Docker inventory drift remains unresolved and is not waived here. B2 apply/death proof/archive/reset, B3 cleanup, C1 real packaging/install/start, C2 installed storage/fault matrix, global Astra and public/unattended activation remain pending. No physical power-loss/ENOSPC, daemon/VM/controller durability or acoustic/engine qualification is claimed.
