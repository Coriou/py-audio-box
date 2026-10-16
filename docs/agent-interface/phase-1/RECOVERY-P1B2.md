# P1B2 implementation handoff

## B2.1 — closed records, startup/publication fences, self-binding plans

Implementation candidate for separate Sol-low review, followed by root integration verification. No self-GO. B2.2/B2.3, B3, C1/C2, final Astra and public/unattended operation remain pending.

Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-15922103`; branch `t3/closed-records-publication-fences`; clean baseline `8ede631c0f1c5428edf58d581f3e1700b5e4432c`. Implementation runtime gpt-6.1-sol / low. The commit carrying this report is the candidate; exact SHA accompanies the handoff. Read the authoritative full P1B2 boundary, PRODUCT, implementation start/protocol, relevant canonical shared/Phase 1/evidence rules, original P1 design/B2 invariants/exit matrix, R1 authority, deployed lifecycle, both B1 reviews and B1 handoff/corrections. No applicable AGENTS.md found. Memory search yielded no B2-specific authority; no prior runtime result is reused as current evidence.

Exactly ten authorized paths change: six production files (`service/persistence.py`, `service/recovery.py`, `service/reconciliation.py`, `service/dispatcher.py`, `worker/runtime_journal.py`, `worker/runtime_supervisor.py`), the three allowed test files, and this report. Core, initializer, resource/admission/dispatch stores, host script, contracts/fixtures, dependencies, packaging, legacy, ledger and prior reviews are unchanged. Public contract 1.3.0, Core user_version 1, R1 protocol 2 and private P1 revision 1 remain unchanged.

### Frozen schema and record interfaces

Optional B2 extension in the existing database, installed as one transaction only by explicit offline `plan_abandon`. `B2_COLUMNS` is separate from B1/ordinary required columns; normal Core construction validates but never installs or completes it. Exact DDL, PK/FK/uniqueness and `(id,revision)=(1,1)` are validated. Unknown/partial extensions refuse. Immutable application interfaces only insert plans; there is no production accepted-intent/step/resolution writer in this lot.

| Table | Exact column order |
| --- | --- |
| p1_b2_format | id revision |
| p1_abandon_plans | id record digest |
| p1_recoveries | id plan_id record digest |
| p1_recovery_steps | recovery_id step record digest |
| dispatch_resolutions | dispatch_seq recovery_id record digest |
| p1_retired_publications | job_id recovery_id record digest |
| p1_integrity_resolutions | object_id recovery_id record digest |

Closed canonical JSON keys are executable constants/validators in recovery.py:

- Plan: `id deployment_id revision policy source_digest observation_digest proof_digest affected prior_epoch new_epoch old_lane new_lane journal_state journal_digest budget_before budget_after diagnostic_decisions count bytes created expires`.
- Accepted recovery: `id deployment_id plan_id plan_digest source_digest observation_digest proof_digest proof_kind prior_epoch new_epoch old_lane new_lane affected before_digest monitor_digest faults_digest highwater budget_before budget_after actions diagnostic_decisions archive_id expected created`.
- Step: `recovery_id step input_digest output_digest lane epoch`. Fixed order: `pending`, `evidence`, `journal`, `core-complete`, `receipt-complete`; expected descriptors have exactly `step input_digest output_digest`, with contiguous digest chain. Actual step rows bind the expected descriptor and selected lane/epoch.
- Dispatch resolution: `dispatch_seq recovery_id job_id binding original_digest decision`, decision `abandoned`.
- Retirement: `job_id recovery_id attempt_id binding publication_digest work_digest decision`, decision `retired`. Binding hashes the canonical publication-work artifact; publication/work digests bind complete original SQL tuples. Original rows remain untouched.
- Diagnostic resolution: `object_id recovery_id reason content_digest decision availability`; diagnostic decision omits only recovery_id. Decision `accept-degraded`; availability `unavailable` or `retained`.
- Per-job action: `job_id action before_digest`; action `retain-terminal`, `fail`, `cancel-requested`, `cancel-never-started`. Exact action coverage matches affected jobs.

Typed affected sets contain exactly `jobs dispatches publications reservations`; duplicates and missing targets refuse. UUID identities are exact lowercase 32-hex; digest fields exact 64-hex. Validators reject unknown keys/enums, booleans as integers, nonfinite times, noncanonical records and malformed reference/history chains. Plan TTL is at most 300 seconds; new lane/epoch are chosen once at plan creation. Budget records use the existing finite closed budget validator. This lot permits only the enrolled configuration, no budget mutation. Proof digest/provenance fields bind a proposal/history; neither a supplied digest nor E1-mock is apply authority.

Before JSON materialization, complete logical table counts/bytes are bounded to 2,048 / 16 MiB. SQL has finite two-second progress deadlines; inventory retains its one-second deadline and 2,048/16 MiB limits. B2 snapshots additionally charge combined SQL/inventory/journal populations conservatively against the aggregate ceilings. Journal remains at most 64 records / 1 MiB; maintenance documents at most 64 KiB each. No history eviction or larger limits.

### Offline interfaces and source projection

`plan_abandon(roots, scope, observation, *, proof_digest, profiles, budget, now=None)` acquires the existing maintenance locks, validates coherent enrolled B1 and exact configuration/consumer coverage before installing B2 schema. It then records a distinct B2 proposal. B1 plans remain unavailable, readable and bound as prior history. No B1 plan can become a B2 plan.

`verify_abandon_plan(roots, db, scope, observation, plan_id, expected_digest, *, proof_digest, now=None)` is a pure read under caller-owned maintenance locks. It acquires no lock or SQLite transaction, including when the caller already holds a transaction. It checks expiry, exact proof/observation/source/affected set. Accepted plans refuse this unapplied-plan verifier; same-plan resume belongs to B2.2, without an expiry bypass here.

The source projection includes schema, all logical rows/history, inventory, journal and enrollment marker. Only the current B2 proposal's row and exact duplicate typed-reference label are excluded. Prior B1/B2 plans, accepted records, steps/resolutions and all other reference obligations remain included. Deletion, valid content mutation or later append stales an older proposal. Immediate nonempty verification covers both queued reservations and real terminal publication/media references.

Reconciliation changes only schema, closed record/reference traversal and retained namespace recognition. Resolutions do not suppress diagnostics or dispatches in this lot. Control archives have exact derived flat names under Core: `p1-recovery-<32hex archive_id>.sqlite`, `.manifest.json`, `.certificate.json`. Only an accepted record can name these; owned single-link private regular files are bounded and inventoried with content digest and typed recovery reference. Manifest fields are `revision recovery_id source_digest snapshot_digest journal_state journal_digest`; certificate fields match the completion binding below. Unknown names/documents refuse. No archive/rotation/transition kernel is implemented.

### Startup and publication order

`persistence.check_recovery_open(db)` validates the optional closed extension and refuses every accepted intent without a valid ordered Core completion. `initialize` calls it before returning usable Core, including historical-local opt-in. This is a mutation-authority fence after SQLite open/PRAGMAs, not byte-preserving inspection. B1 private WAL-aware copies remain the read path.

`check_publication_allowed(db, job_id)` runs in the existing `storage_session` wrapper before BOTH `_publish` and `_commit_publication`, including the latter's terminal fast return. It checks pending/partial state and permanent exact retirement before invoking the method body. No Core source edit, callback or bypass flag. Unenrolled historical behavior remains covered.

Dispatcher first takes a bounded source-preserving private pre-read for every enrolled root, including the default constructor. It obtains authenticated supervisor hello through the existing bounded runtime adapter before mutable Core/Admission/A2/R1 startup. Default enrolled startup requires explicit registration and therefore cannot begin mutable epochs by omitting the option; this deliberately strengthens the old B1 default diagnostic test. Explicit startup validates registration/configuration/completion, constructs Core (repeating DB checks under its normal control lock), verifies deployment identity and repeats the exact hello before Admission. Changed/unavailable hello refuses. Activation stays closed after completed startup. No receipts mount or new mount is introduced in control.

Journal takes the existing lane lock and checks protected maintenance BEFORE journal load, supervisor boot generation/watchdog or socket listening. Failure closes the lock and never remints a journal. Required marker `p1-b2-required.json` has exactly `revision deployment_id`; persistent fence `p1-maintenance.json` has exactly `revision deployment_id recovery_id plan_id plan_digest proof_digest old_lane new_lane step certificate`. Fence steps: `pending`, `evidence`, `journal`, `core-complete`, `complete`; certificate must be null until complete. Marker/fence are owned single-link 0600 managed files.

Completion binding has exactly `deployment_id recovery_id plan_digest proof_digest lane epoch certificate_digest`; the digest is the canonical Core-complete step digest, avoiding self-hashing. A replacement journal carries this same optional `maintenance` binding in its payload; removal of required marker/fence cannot make that journal look fresh. Completed worker hello exposes this binding through the existing authenticated channel. Worker requires certificate/new journal lane agreement; control requires exact Core/hello agreement. Old unmarked diagnostic workers retain their journal behavior.

Protocol for the later writer is frozen: under both locks/death proof, required marker and receipt pending intent precede Core pending, before any journal/resource authority change. Evidence barriers precede replacement. Core completion precedes receipt completion. This lot adds readers/validators only: synthetic tests install pending/complete records directly. Every programmatic/CLI/host apply entrypoint remains unconditionally unavailable; no production abandonment function is callable.

| Core side | Receipt/worker side | Required result |
| --- | --- | --- |
| no accepted intent | pending | worker refuses; enrolled Dispatcher cannot hello/start |
| pending | absent/pending/complete or unavailable | direct Core refuses; Dispatcher refuses before mutable startup |
| complete | absent/pending/malformed | worker or exact hello binding refuses; no control startup |
| complete | matching complete/new journal | worker may hello; explicit registered control may inspect, activation remains closed |
| complete | different lane/certificate/incarnation | exact hello/registration or repeated hello refuses |
| complete + retired publication | any | both decorated publication methods permanently refuse, including terminal fast return |

### Verification and limits

E1 local temporary synthetic roots, real SQLite/WAL/flock/files and deterministic mocks only. No heavy Docker or installed C1/C2 campaign. Tests include two restarts of pending combinations, direct historical Core refusal, both decorated retirement paths before body/filesystem mutation with unchanged terminal SQL/media, optional schema rollback, partial/unknown/oversize/duplicate/nonfinite/reference negatives, nonempty immediate self-verification, history delete/content/append staleness, worker preboot/prelisten refusal, completed authenticated hello/recheck and unchanged B1 initialization/no-remint/WAL/un-enrolled regressions.

Initial RED (before implementation):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b2 -q
```

2 failed / 59 deselected: missing plan_abandon and missing partial-schema constructor refusal (`/tmp/p1b21-red.log`). Development recovery run exposed the shared-column-map B1 completion regression, a fixture connection left open until GC (source-copy change refusal), and a fixture serialization refusal for NaN before DB injection. Fixed in owned files, no initializer expansion. Development results are not final gates. One ancestor-instruction shell lookup used zsh's special `path` variable and could not find tail/sed in that subprocess; rerun with a task-specific variable succeeded, with no source effect.

Focused B2 GREEN before final identity/canonical hardening: 36 passed / 59 deselected in 5.76s (`/tmp/p1b21-green.log`). Earlier affected trio: 148 passed in 42.48s (`/tmp/p1b21-affected-dev.log`), superseded by the frozen checks below.

Frozen affected/full results are recorded below after completion. No source/test edits during these final runs; only this handoff is edited.

A1 whole-Docker-inventory drift EXIT 1 remains unexplained and unqualified, with no cause or waiver inferred. Installed/platform skips are not passes. B2.2 transition/resolution kernel and same-plan resume, B2.3 trusted host proof/apply, B3 cleanup, C1/C2 installed proof, root integration/independent review and final Astra remain pending. No physical power-loss/ENOSPC/controller/VM, real engine/audio, hardware or public/unattended acceptance claim. No installs, paid/remote scripts, run-e2e-test.sh, models/private audio/network/cloud/STT/Voicebox, subagents, push/PR/deploy, prune or unrelated cleanup. Unrelated research/worktrees preserved.

Frozen affected command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py tests/service/test_persistence.py tests/service/test_admission.py tests/service/test_core.py -q
```

**344 passed in 106.46 seconds, exit 0**, `/tmp/p1b21-affected-final.log`. This is the frozen source/test tree; all 59 predecessor B1 tests remain present, with its enrolled-default test strengthened to require pre-Core registration rather than constructing mutable diagnostic authority.

Frozen full preservation command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**379 passed, 18 skipped in 115.31 seconds, exit 0**, `/tmp/p1b21-full-final.log`. This includes the full B1/R1/W2/A1/A2 and release preservation suite. Skips remain skips. Source/tests were frozen throughout both final runs; only this report changed afterward. `git diff --check` passed. Explicit protected-path diff check was empty for Core, initializer, resource/admission/dispatch stores, host script and contracts. Candidate is ready for separate independent review, not self-approved; parent owns review launch, integration and subsequent sublots.
