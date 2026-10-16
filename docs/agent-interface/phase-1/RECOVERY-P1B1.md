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

The exclusive 0600 receipts-side `p1-init.json` is first. It binds random deployment identity, exact daemon/context and named-volume identities (name/driver/creation identity per role), profile/budget digest, private revision, synthetic scope and local role-directory identity. Its private seed fixes key/journal bytes for matching completion; seed/key material is never emitted. Exclusive per-file creation intents precede exclusive files. A synced receipts-side `p1-core-intent` precedes any Core namespace/database attempt; existing intent with missing/empty/partial Core quarantines before the Core constructor can create authority. File fsync and directory/parent barriers precede subsequent steps. Missing files with an existing creation intent quarantine; keys/journal are never reminted over a missing previously attempted file. Partial differing bytes remain untouched. Core/R1/A2/B1 initialize coherently, then the final enrollment marker is exclusive and parent-synced.

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

`inspect`, `plan`, `verify_plan`, `register` are trusted offline Python seams. Inspection takes the existing exclusive lock files and opens a read-only, query-only SQLite transaction on a private WAL-aware copy. Original SQLite files are never opened by SQLite on this path. Exact supported schema, aggregate row/byte checks before materialization and a SQLite progress deadline remain enforced. Limits: 2,048 rows/entries, 16 MiB plus bounded journal/private-file reads, one-second filesystem inventory and two-second SQL deadline; refusal on incompleteness. It returns exact affected IDs; job/attempt/generation/deadline, dispatch/binding/runtime/receipt digest, reservation/demand/monitor/fault, publication/inventory, journal/lane and version diagnostics. It does not open or complete R1/A2 observation epochs, checkpoint/delete WAL, repair metadata or publish output.

Plans bind deployment/revision, logical source-state digest, structured host-observation digest, exact jobs/unacked dispatches/publications, finite count/byte bound, closed abandonment policy name, created/expiry (maximum 300 seconds), and `apply_available=false`. Only the current proposal's own row and exact duplicate reference label are excluded when reconstructing its source digest. Every preexisting plan row and reference obligation remains bound. A later appended proposal invalidates older plans; it does not disappear from their source projection. All other source rows, diagnostics, schema, filesystem inventory and journal identity remain bound. No wildcard or caller deletion path exists. A changed source, digest, scope, registration, container incarnation/state, missing key/journal, unknown consumer or occupied lock refuses verification. Unknown reference kinds/schema or foreign inventory prevent plan creation; read-only inspection still reports recognized diagnostic defects. A host observation digest is not death proof or authentication by itself. B1 cannot authorize a resource reset, release, ACK, epoch transition, budget change or cleanup.

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


## Round-1 NO-GO corrections — awaiting fresh independent review

Correction worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-94e6816a`, branch `t3/94e6816a`, clean root baseline `ca613d8`. Candidate `b2af46fe7b60f6fe30624e549a21201827ef3fce` was cherry-picked as `b172e06` without conflict; root review/protocol documentation remains intact. Only initializer, recovery, B1 tests and this handoff changed in the correction. No other checkout was edited. Final commit is the commit carrying this section (exact SHA in final handoff).

- **F1 response:** exclusive, file/directory-synced Core intent is committed before Core namespace creation. A crash at `core_intent`, lost Core after `database`, or an empty replacement refuses matching completion without a database or enrollment marker. Retained Core namespace/schema/version/health/enrollment is checked through a private read under normal locks before invoking the writing Core constructor. Partial schemas remain untouched. Existing matching completion tests still cover all seven genuinely completed/not-yet-attempted boundaries. No guessed migration or authority reconstruction was added.
- **F2 response:** source projection retains all preexisting plans, canonical rows and typed references; verification removes only its own ID/label. Prior-plan deletion and valid digest-recomputed content mutation now refuse. Later appends explicitly stale older plans. Immediate nonempty verification and job/publication/dispatch reference retention remain tested. No reconciliation authority or cleanup rule changed.
- **F3 response:** read-only offline operations use a unique process-owned 0700 temporary directory, 0600 copies of DB/WAL/SHM, and SQLite only on that private namespace. Missing source SHM is permitted and rebuilt privately. The source names, file identities and full bytes are checked before/after copy and after use; any detected source change/error refuses. Copying is bounded to 16 MiB aggregate physical SQLite bytes, 2,048 top-level entries and a two-second observation deadline (in addition to existing SQL/inventory budgets). This is deliberately conservative: a small logical DB with excessive WAL/free pages can refuse. Private copy cleanup occurs on success/errors; original checkpoint/unlink/repair never occurs. The normal offline Core/journal locks exclude cooperative writers. Startup enrollment's pre-Core read also uses the stable private copy; it grants no writer authority. No protection against a malicious host changing and restoring bytes is claimed.

New E1 regressions: `test_core_creation_intent_never_remints` (lost/empty/intent), `test_plan_binds_other_plan_evidence` (delete/content/append), `test_crash_wal_inspection_preserves_source` (four SHM/directory-mode combinations), and two private snapshot source-change/open-error cleanup tests. Crash WAL tests run a real child that commits `checking` and calls `os._exit(0)`; inspection must retain that committed state while every original regular-file hash and name remains unchanged. Read-only directory mode is 0500; this is local filesystem/process evidence, not an installed mount qualification. The old `logical()` test helper now explicitly closes its SQLite connection so garbage collection cannot independently checkpoint the source during inspection.

RED command (before source corrections):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k 'core_creation_intent or binds_other_plan or crash_wal' -q
```

**10 failed, 47 deselected**, `/tmp/p1b1-corrections-red.log`. The intent case initially fails because the new fault boundary does not exist; lost/empty completion, all plan-history probes and WAL preservation probes expose the reported behavior. An initial edit command used unavailable `python` and made no changes; subsequent commands use the existing interpreter. Intermediate focused runs exposed an unclosed test connection and preserved missing-database exception compatibility; final results below supersede those development runs.

Frozen-source GREEN command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -q
```

**59 passed in 3.33 seconds**, `/tmp/p1b1-corrections-focused-final.log`. Full service/release result follows below. No source/test edits during that final full run. Public contracts 1.3.0, Core user_version 1, R1 protocol 2 and private P1 revision 1 remain unchanged. Apply remains unavailable and persistent activation closed; diagnostic mode remains covered.

E1 only. A1 unexplained whole-Docker-inventory EXIT 1 remains unqualified, with no external cause or waiver inferred. B2/B3/C1/C2, integration verification, global Astra and public/unattended acceptance remain pending. No Docker campaign, models/audio/network/paid operation, E2E runner, subagent, push, PR, deployment or prune was performed. Ready for fresh independent review; no self-GO.

Final frozen-source service/release command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**338 passed, 18 skipped in 108.56 seconds, exit 0**, `/tmp/p1b1-corrections-full.log`. This includes all affected Core/Dispatcher/Admission/persistence/reconciliation tests and the 59 B1 cases. Only the handoff was edited during/after this run; source/tests remained frozen. `git diff --check` passed. The complete baseline-to-candidate path set remains exactly the authorized eight; the correction delta itself is four of them. Independent review and parent integration are the next gates.
