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

## B2.1 round-1 NO-GO correction — fresh independent review required

2026-10-16, gpt-6.1-sol / low implementation. Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-361ad50c`, branch `t3/b21-recovery-evidence-correction`. Started clean at root documentation baseline `d5b9bb9`; inspected before cherry-picking the single candidate `ccf3c07c93f504056a538d2f318bdbaa35bb4dd3` as `ae54dcc`. Root reviews/docs are preserved. No original/root checkout was edited. Correction source/test commits are `141810adeea4b81013f214d82928277a4cf1f50e`, `9dda03193c6786fc0e80b6ca5146a39e097cae31` and final frozen source `aae8919cca644b6197c2fb60267ace292a9dfe9d`. Final report-only commit follows these; its exact SHA accompanies the handoff.

Read the complete P1B21-r1-sol findings/reproducers, P1B2-BOUNDARY-ASTRA architecture/records/plan binding/brief/exits, original P1 decision, PRODUCT, start/protocol, R1 authority, B1 round-2 review and original/corrected B1 and candidate B2 handoffs. Read relevant canonical shared/Phase 1/evidence guidance and inspected actual callers. No applicable ancestor/path AGENTS.md was found. A lightweight memory registry search provided no B2 authority or acceptance evidence; current source and fresh checks are the evidence here.

Correction delta is limited to `service/recovery.py`, `service/persistence.py`, `service/reconciliation.py` (one archive-root argument in the existing adapter), `worker/runtime_journal.py` (maintenance startup readers only), `tests/service/test_recovery.py`, and this appended section. Candidate changes to Dispatcher/supervisor and the other allowed tests remain intact. No Core, initializer, resource/admission/dispatch store, host script, contracts, dependencies, packaging, ledger or review edit. Public 1.3.0, Core 1, R1 protocol 2 and private P1 revision 1 remain unchanged.

### F1 response: structural allocation provenance before DDL and on verification

`_coherent_b2` now consumes direct allocation joins, rather than a partial list of diagnostic reason strings. The bounded snapshot independently checks job/reservation identity coverage, queued versus attempted allocation, executing charge with an actual unacknowledged dispatch and attempt, terminal retained state, frozen Authority/context/generation/deadline/attempt identity, demand configuration/charges, receipt/outcome/reservation consistency and publication owner. This also avoids the reconciliation reason-overwrite issue where `unbound_dispatch` could hide an earlier owner mismatch. Unknown schema/reference kinds and foreign namespaces still refuse. Unbound prepares and UNKNOWN runtime identities with coherent allocation provenance remain supported planning populations; they do not become death proof.

The one-field queued-to-executing reservation negative runs before first installation and with an already installed plan. Both planning refusals preserve the complete logical/schema population; the installed case separately verifies refusal with no plan mutation. Existing queued positive and new real-allocation unbound/UNKNOWN positives immediately self-verify. Optional schema remains all-or-none and installed only by explicit offline planning, after coherent enrolled B1 validation. Normal constructors remain validate-only.

### F2 response: bounded actual documents, transitive hashes and retained archives

The accepted record adds closed `before` and `proof` documents. `before` has exactly `source prior_certificate journal`. `source` is the exact previously fingerprinted `schema data inventory journal marker` projection: every supported table with finite scalar canonical rows, exact bounded inventory/owner/reference records and all previous B1/B2 plans/accepted records/steps/resolutions. The current proposal alone remains self-excluded. `journal` has exactly `state bytes_hex`, retaining bounded original journal bytes (not arbitrary nested data or a patch language); valid journal identity/digest/lane and typed allocation snapshots are checked. `prior_certificate` is null only for the deployment's first epoch; later histories require the actual previous completion binding. `proof` has exactly `kind observation prior_epoch old_lane`; its actual closed observation and retained registrations undergo scope/incarnation/coverage/stopped-state checks. This is a record/provenance binding, not authenticated real host proof. E1-mock cannot reach apply.

Readers recompute `source_digest`, `before_digest`, `monitor_digest`, `faults_digest`, proof/observation digests, original per-job action digests and expected output hashes from these documents. Action choice is checked against actual frozen job state/attempts. Every expected step has exactly `step input_digest output_digest output`; every step receipt adds `recovery_id lane epoch`. The output digest hashes **the step name, input digest and closed output document**, so the certificate transitively binds the entire chain rather than independent matching labels. The first input is the actual before-image digest.

Fixed output schemas are executable in recovery.py:

- `pending`: exact recovery/plan/proof/before binding.
- `evidence`: archive identity, actual snapshot/manifest hashes, old journal status/digest.
- `journal`: exact old/new lanes and the bounded empty new-journal document.
- `core-complete`: exact before/prior/new epoch, actions/diagnostic decisions, finite monotonic high-water/configuration, healthy new monitor and finite canonical output rows for jobs/reservations/monitor/faults/accounting. Validators preserve terminal job rows, identities, original demand/charges and required unsuccessful nonterminal outcomes. These are output-record contracts; no transition writer or new scan kernel exists in this lot.
- `receipt-complete`: exact recovery/lane/epoch and preceding Core output digest.

At and after recorded evidence creation, Core-side readers require the actual derived control snapshot and manifest; Core completion additionally requires the actual immutable certificate. Manifest binds before/source/proof, snapshot bytes and retained original journal document. Snapshot bytes are hashed, then queried only on a private 0700/0600 read-only immutable copy. Exact schema/version, all canonical logical rows/history and the exact current proposal are compared to the frozen before-image. Nothing is restored over Core. Original WAL-preserving inspection remains in place. Certificates must equal the recomputed actual Core step binding, including every output; absent/changed archives cannot be replaced by caller-authored digest strings. Earlier archives/history are checked again when a later recovery completes.

Receipt-side startup additionally requires `p1-recovery-<recovery_id>.certificate.json` with exactly `certificate steps` (the four actual Core-side step documents), and `p1-recovery-<recovery_id>.journal` containing original journal bytes or explicit empty missing-journal evidence. Both are fixed derived managed names. Under the existing lane lock, the worker checks protected file identity/permissions/bounds, the transitive step chain, fence/certificate/lane/epoch/plan/proof agreement and actual original journal bytes before boot/listen. No service import or control mount is added to the worker. Core validates control archives; the worker validates its own receipt archives; authenticated hello and the repeated control check still bind both volumes.

Original 2,048-entry/16 MiB aggregate ceilings remain; embedded histories/output rows and control archive populations consume them rather than enlarging them. SQL/filesystem/private-copy deadlines remain finite. Receipt certificate documents retain the existing 64 KiB limit and old journals retain 64 records/1 MiB; larger evidence refuses conservatively. Every actual apply entrypoint is still unconditionally unavailable. Persistent activation remains closed. There is no production accepted-intent/step/archive writer, mutation kernel, resolution consumption, resume, host authentication or host apply transport here.

### Probes, development failures and final frozen-source validation

All commands use the existing interpreter, no installs, `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`, with disposable synthetic roots. Logs below are local `/tmp` evidence, not repository artifacts.

Initial combined RED: `tests/service/test_recovery.py -k 'unexplained_execution or grounded_evidence' -q`, `/tmp/p1b21-correction-red.log`: **9 failed, 1 passed, 95 deselected in 5.29s**. Seven F2 cases exposed accepted arbitrary before/monitor/fault/action hashes or absent/changed archives/certificate. The output-label case was already rejected by the old step-chain check. The two initial F1 failures stopped at an incorrectly selected not-ready profile, so those two are not valid F1 defect evidence.

Corrected F1 RED ran against the original cherry-picked source (the three corrected source files were saved/restored in this worktree with a `try/finally`, no other checkout touched): `tests/service/test_recovery.py -k unexplained_execution -q`, `/tmp/p1b21-correction-f1-red.log`: **2 failed, 126 deselected in 1.12s**, both `DID NOT RAISE ValueError` at the intended planning-refusal assertion. This reproduces the real reported hole with a ready queued job, before and after installation.

Development runs are superseded by the frozen checks below. They exposed test snapshot creation inside a source-preserving namespace read, fixed by creating bytes privately before installing the test archive; an attempted in-memory SQLite deserialize of a WAL-mode backup refused with `unable to open database file`, replaced by a private immutable read-only file copy. Development B2 runs then passed 46 and 69 cases. A provisional affected trio passed **182 in 64.76s** before the final accounting/history checks. Final focused probes before adding the second-history regression passed **47, 88 deselected in 23.62s**; second-history/rehashed-output/worker probes passed **16, 122 deselected in 4.41s**. These are development evidence, not frozen-source acceptance. A first prospective final trio passed 192 in 67.17s but was superseded by finite-accounting/retained-state hardening; its last few seconds overlapped the next trio (192 in 67.49s). The then-frozen affected run passed 387 in 132.84s. A final read-through found action selection consulting a nonexistent cancellation Boolean instead of `state=='cancel_requested'`; both receipts are therefore superseded by the final source checks below. Source was not edited during that affected run: it completed before the cancellation correction. The added real allocated/cancel-requested pending-intent fixture accepts the correct action and refuses a digest-recomputed `fail` action. Test-only expected output documents describe its future unsuccessful outcome; no actual recovery transition runs. Cancellation/history/rehashed probes passed 6, 133 deselected in 3.62s (`/tmp/p1b21-correction-actions.log`).

New negatives independently alter prestate monitor/fault/high-water/job/history/original journal, proof observation/kind/prior certificate, every step output, archive presence/bytes/manifest fields and certificate fields, with valid canonical SQL row-digest recomputation. Further probes rehash the entire expected/receipt chain and rewrite archive manifests/certificates after changing a snapshot's actual logical rows or a terminal output row: grounding still refuses. Completion, private pre-Core startup and historical Core readers are exercised separately. Receipt archive absence/journal/chain/certificate changes refuse twice before boot/listen. Two coherent completed histories retain prior authority and archives; removing old evidence/history refuses.

Frozen final command receipts follow below. Source/tests are unchanged during those runs; only this handoff is edited.

E1 local source/SQLite/filesystem/process and mocked host projections only. The 18 installed/platform skips are not qualification. A1 whole-Docker-inventory EXIT 1 remains unexplained and unqualified, with no cause or waiver inferred. B2.2/B2.3, B3, C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending. No physical power-loss/ENOSPC/controller/VM, real engine/audio/hardware or clean-host claim. No subagents, heavy Docker, paid scripts, E2E runner, network/models/private audio/cloud/STT/Voicebox, push/PR/deploy, prune or user cleanup. Unrelated research/worktrees preserved. No self-GO: both findings require fresh independent review of the final candidate.

Final frozen source/test commit: `aae8919cca644b6197c2fb60267ace292a9dfe9d`.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**193 passed in 67.79s, exit 0**, `/tmp/p1b21-correction-focused-final3.log`. This final source includes all corrections and the cancellation regression. The subsequent affected/full receipts below are sequential runs on the same unchanged source/tests.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py tests/service/test_persistence.py tests/service/test_admission.py tests/service/test_core.py -q
```

**388 passed in 132.31s, exit 0**, `/tmp/p1b21-correction-affected-final2.log`. Final source/tests remained frozen throughout, with only this handoff edited. The previous 387-test run is superseded by this final 388-test result.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**423 passed, 18 skipped in 139.26s, exit 0**, `/tmp/p1b21-correction-full-final.log`. This full preservation run followed the focused and affected runs sequentially on unchanged final source/tests. Skips remain unqualified; no aggregate passing count waives either original review finding or supplies independent acceptance. Final `git diff --check` passed. The complete baseline-to-candidate path set is exactly the ten allowed paths; protected Core/initializer/resource/admission/dispatch-store/host/contracts/reviews/ledger comparison against root baseline is empty. Only this report is committed after the frozen source/test commit. Fresh independent Sol-low review and parent integration verification are required before proceeding; no self-GO.

## B2.1 round-2 targeted correction — awaiting independent review

2026-10-16, gpt-6.1-sol / low implementation. Only this fresh worktree was changed: `/Users/ben/.txcode/worktrees/py-audio-box/t3-b05a2844`, branch `t3/recovery-job-context-validation`, clean root baseline `361deab2a9269800925cf9f7095fde033f98c680`. Inspected the ordered candidate log `d5b9bb9..d4290c6`; imported the complete chain oldest first: `ae54dcc` → `141810a` → `9dda031` → `aae8919` → `d4290c6`, yielding local `0d8f77d` → `75d4178` → `6f12aae` → `b7d5e07` → `ce0e262`. Root documentation/reviews/ledger are preserved against the actual fresh-root baseline. Original candidate and root checkouts were not edited.

Read both complete P1B21 reviews, the candidate handoff/responses, full B2 boundary architecture/records/brief/exits, original P1 invariants and exit matrix, PRODUCT, start/protocol, canonical shared/Phase 1/evidence guidance, R1 fault-authority and B1 context. No applicable AGENTS.md was found. The quick memory registry search yielded no B2 authority or current acceptance evidence.

### Finding responses and frozen transition

F1: preserve the imported structural allocation joins unchanged. Existing queued, coherent real allocated/unbound and old-lane UNKNOWN positives and missing-allocation negatives remain in the focused preservation suite. No reason blacklist or allocation relaxation was introduced.

F2: `service/recovery.py::_validate_core_output` now validates both before/output jobs with the canonical closed job oracle, validates the exact `plan bundles` context and its canonical stored-plan/input binding, and checks active-attempt generation ordering/prior reap rules. Validation errors become bounded recovery refusals. The context and every SQL job-row column outside JSON remain exactly equal to the before-image.

For each nonterminal action, construct a deep copy of the validated frozen job and compare the entire output against it. The only permitted deltas are `state` to `failed` for `fail`, otherwise `cancelled`, and the last attempt's `reaped` to true if an attempt exists. Stage, artifact IDs, input, principal/idempotency identity, deadline, generation, attempt count/order/IDs/ownership/staging, all prior attempts and all other fields remain unchanged. Terminal rows remain byte-for-byte equal. This freezes one conservative output contract; it does not add a transition writer. The separately validated retained operator proof supports reap; no PID-1 receipt, timeliness, ACK or dispatch outcome is fabricated. Action selection still uses actual `state == 'cancel_requested'`. Queued never-started cancellation retains an empty attempt list.

The synthetic completed-evidence fixture now describes proven reap in that exact allowed output; actual Core jobs/dispatches are not transitioned. New real Admission/DispatchStore allocated fail and cancel-requested positives exercise all three readers. Separate fully rehashed negatives cover erased attempts, attempt ID/generation/owner/staging changes, unknown job and nested attempt fields, stage/artifact/input/principal changes, changed context, unreaped current attempt, wrong terminal outcome, generation and deadline. Each fixture rehashes the entire expected descriptor chain, receipt-complete Core binding, accepted/step SQL row digests, and replacement certificate. Completion, source-preserving pre-Core startup and historical Core readers independently refuse each negative.

Correction delta after import is only recovery validator, recovery tests and this appended response. No schema/record-key/table/version delta. The complete fresh-root-to-candidate path set remains the original ten allowed paths. No Core, initializer, resource/admission/dispatch store, host script, contracts, dependencies, packaging, ledger or reviews edit. Optional atomic bounded records, exact plan self-exclusion/history, private WAL copies, no-remint initialization, pre-body publication retirement, cross-volume authenticated hello and enrolled activation closure remain intact. Every apply remains unconditionally unavailable; no B2.2/B2.3 implementation.

### Actual verification and limits

All checks use existing `/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, no installs and disposable local synthetic roots.

RED before production correction: `tests/service/test_recovery.py -k rehashed_nonterminal -q`, **22 failed, 2 passed, 139 deselected in 14.27s**, exit 1, `/tmp/p1b21-r2-red.log`. All 22 failures reach the intended missing-refusal assertion; the two unchanged allocated fail/cancel witnesses pass. This includes the exact separate erased-attempt and unknown-field reproductions. Initial GREEN on the same 24 cases: **24 passed, 139 deselected in 15.07s**, exit 0, `/tmp/p1b21-r2-green.log`. Five further single-field variants per cancellation state were then added before freezing. No other development test failures occurred in this correction; prior-round failures/receipts above remain attributed and superseded as stated there.

Frozen source/test commit: `0d976dac304db7eccdfb60d2078defda07026fdf`. Final focused/full receipts follow; only this report is edited during/after them.

E1 local source/SQLite/filesystem/process and mocked host projections only. Installed/platform skips are not qualifications. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver. B2.2/B2.3, B3, C1/C2, parent integration, fresh independent review, final Astra and public/unattended acceptance remain pending. No physical power-loss/ENOSPC/controller/VM, real audio/engine/hardware or clean-host claim. No subagents, Docker/network/models/audio/paid/E2E runner, installs, push/PR/deploy, prune or user cleanup. Unrelated research/worktrees preserved. Ready for fresh independent review after the final checks below; no self-GO.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**227 passed in 91.19s, exit 0**, `/tmp/p1b21-r2-focused-final.log`. Includes the 34 new real-allocated/rehashed reader cases and all imported preservation cases. Source/tests remained frozen. The full service/release run follows sequentially; no redundant separate affected-full run is needed.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**457 passed, 18 skipped in 163.44s, exit 0**, `/tmp/p1b21-r2-full-final.log`. Sequential after the focused suite, on unchanged source/tests `0d976dac304db7eccdfb60d2078defda07026fdf`. The 18 skips remain unqualified. Final `git diff --check` passes. Exact protected-path comparison against fresh root `361deab` is empty for Core, initializer, resource/admission/dispatch stores, host script, contracts, reviews and ledger; complete changed-path set is exactly the original ten. No source/test/worker changes after the frozen commit. This report-only commit follows it. Parent owns fresh independent review and integration verification; neither these passing checks nor this response supplies self-acceptance.

## B2.2a — private accepted-intent and evidence prefix (awaiting independent review)

2026-10-16, implementation gpt-6.1-sol / low. Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-fa982d42`, branch `t3/b22a-accepted-intent-archive`, clean starting HEAD `c8d3fa0f53f6fcead11467ab96ca81c59fb09276`. That baseline contains root's bounded B2.1 integration and B2.2a narrowing. Read actual source, full P1B2-BOUNDARY-ASTRA (including sections 2/3/4 and B2.2), the complete existing handoff/corrections and P1B21-r3-sol, and orchestration protocol. No applicable AGENTS.md was found. Memory registry search supplied no B2-specific authority. Predecessor review/test receipts remain predecessor evidence.

Frozen source/test commit: `b999c129c8cea52d637d270c977995cf11dca32b`. Only `service/recovery.py` and `tests/service/test_recovery.py` change there; this appended report is the only subsequent path. No frozen schema, closed record keys, versions, ordinary B1 verifier, startup reader or activation contract changes. No persistence/worker change was needed. Core/initializer/supervisor/resource/admission/dispatch stores, host scripts, contracts, packaging, dependencies, legacy, ledger and reviews remain unchanged. No other checkout was edited.

### Private interface and state sequence

`_pending_through_evidence(roots, scope, intent, observation, *, now=None, snapshot_bytes=None, fault=...)` is an internal production helper, unreachable from every apply/CLI/host entrypoint. It accepts the existing closed accepted-recovery record, including immutable before/source, structured proof provenance and all five fixed expected output documents. It accepts no death Boolean, arbitrary path, SQL, action callback or new activation toggle. `fault` is the deterministic internal boundary callback; it supplies no mutation operation. Both normal ownership locks are acquired nonblocking by the existing `offline(..., write=True)` for the entire invocation. Supplied E1-mock documents exercise these validators only; this is not host authentication or B2.3 transport.

Before acceptance, the helper checks exact plan/digest, proof/observation, initial TTL, deployment/volume roles/keys/schema, every logical table/history, media/inventory identity, all retained registration owner coverage, coherent allocation authority and exact affected set. Only the current proposal row/label, current accepted row, its two allowed step rows/labels, and its exact snapshot/manifest inventory entries are accounted for separately on resume. No table-wide history, inventory, resolution or journal exclusion exists. A different plan, old-history mutation, extra namespace entry, changed proof or media refuses.

Sequence: durable required marker; immutable receipt-side full accepted document at `p1-recovery-<recovery_id>.intent.json`; durable pending fence; Core accepted intent transaction; `pending` step transaction; fixed archive creation intents and artifacts; verified `evidence` step transaction. The fence stays `pending`. Return is explicitly `status='pending-through-evidence'` with the existing recovery/lane/epoch identities. No normal constructor is used to bypass the pending fence. No journal rotation/fresh lane, job/attempt/event transition, resource reset, resolution, retirement, certificate or receipt completion occurs.

Receipt-only resume uses the same complete accepted document/identity and original prestate after TTL; it can recapture the still-unchanged original database before the Core intent exists. With a Core intent already committed but no snapshot artifact created, callers may supply the exact retained original `snapshot_bytes`; its expected hash and complete logical contents are validated. The helper never substitutes a backup of the changed pending database for that original snapshot. Once the snapshot exists, ordinary same-intent resume needs no supplied bytes. Missing snapshot bytes without an existing artifact stop pending; they do not mint another snapshot identity. This private input requirement is explicit for later transport/kernel integration, not a claim that a trusted restart bundle exists today.

Initial evidence capture uses the existing bounded private DB/WAL/SHM copy, SQLite backup with page/time bounds, exact expected physical SHA-256, and B2.1's grounded archive oracle. A prospective private candidate containing both prefix step rows is validated, including retained prior archives and aggregate ceilings, before receipt acceptance. Archive snapshot includes all prior history and the exact current proposal, with no current accepted-intent mutations. No snapshot is ever restored over Core.

Control names remain exactly B2.1's `p1-recovery-<archive_id>.sqlite` and `.manifest.json`; manifest contains the original journal bytes/status. Receipt-side original bytes additionally use `p1-recovery-<recovery_id>.journal`. Three closed creation descriptors use that recovery prefix with `.snapshot.intent.json`, `.manifest.intent.json`, `.original-journal.intent.json`; each binds revision, recovery ID, fixed kind and expected SHA-256. There is no generic path writer exposed through the interface. Files use exclusive no-follow 0600 creation, file fsync and parent barriers. On resume, expected existing bytes are verified and re-barriered; an absent step receipt is then recorded without overwriting/replacing the artifact. A recorded creation intent with a missing/partial/wrong artifact quarantines. Lost accepted receipt/fence, unrecorded artifacts, premature completion certificates and unknown receipt/control entries refuse. Earlier completion archives/certificates remain intact when a new pending fence replaces a verified completed fence; an interrupted replacement temporary is quarantined.

The private source observation mode preserves `missing` versus bounded `unreadable` journal status and actual bytes, returns no clean journal-record observation for either, and requires coherent Core authority plus all retained registration/consumer coverage. Ordinary B1 inspect/plan still requires its valid journal. The prefix does not infer clean receipts or fabricate ACKs from absent/corrupt bytes.

### Tests and fault matrix

New tests exercise the production helper against real temporary SQLite/WAL/files/flock. The existing B2.1 input builder gained a prepare-only branch: it constructs immutable supplied documents but writes no accepted intent, step or archive in this mode. It is not an alternate transition writer.

- Receipt marker/accepted document/fence: create, complete write, file fsync, parent barrier; partial documents quarantine, durable exact documents resume. Receipt-only acceptance resumes after expiry with the same identity and one Core intent.
- Core accepted intent and both step receipts: BEGIN, write, COMMIT and selected parent boundaries; rollback/committed ambiguity is inspected only on a new invocation/connection. No live COMMIT retry.
- Snapshot/manifest/original-journal: persisted creation-intent boundary, create, write, file fsync and parent boundary. Actual injected file-fsync and parent-sync exceptions also retain the original artifact inode/bytes on verified resume. Missing artifact after intent and deleted creation descriptor refuse rather than remint.
- Two final pending restarts refuse direct historical Core and Journal startup. Same accepted expired replay keeps one intent, one archive identity, and only the two prefix steps. Both locks busy refuse. Never-accepted expired plans refuse before marker creation.
- Changed logical state, old history, later/changed plan, schema, media, proof/provenance, affected set, journal, archive, extra entries and lost fence refuse. Missing/corrupt journal exact-coverage positives retain the original bytes/status; omitted owner/live observation negatives refuse.
- Real allocated cancel-requested work retains jobs, dispatch/receipt/outcome/ACK, reservations, demands/charges/high-water, monitor/faults and events. Real terminal publication/media/history remain unchanged. A later prefix preserves prior completed history/archives; old-history/old-archive damage refuses.

Baseline RED was run after authoring the tests, temporarily substituting only `git show HEAD:service/recovery.py` in this same worktree with guaranteed `finally` restoration: `tests/service/test_recovery.py -k b22_prefix_pending_and_exact_expired_resume -q`, **1 failed, 227 deselected in 0.56s**, exit 1 (`/tmp/b22-red.log`), at missing production `_pending_through_evidence`. No other checkout/source was substituted. Development first run: 34 passed/1 failed because the test incorrectly expected a fully written existing snapshot to quarantine; corrected to required verify-and-record behavior. Later development runs similarly corrected complete accepted-write/fence-create expectations, an empty affected-set no-op negative, and a nonexistent allocation fixture helper. Expanded 55-case run passed. Prior-history positive then exposed requiring a receipt-complete SQL row where the accepted B2.1 completion fixture has the validated Core completion plus receipt certificate; fixed to validate that existing completion/fence chain. The six affected history/barrier/media tests passed. Receipt-only no-new-identity trio passed; prospective aggregate-validation quartet passed. These development receipts are superseded by the frozen runs below.

All commands use the existing `/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, pytest `-p no:cacheprovider`, no installs. Final source/tests remain frozen during focused/full checks; only this report changes. Final receipts follow below.

E1 local synthetic source/SQLite/filesystem/process and mocked host projections only. No self-GO: parent owns fresh independent review and root integration. Remaining B2.2 tranches must implement journal/resource/job/resolution/Core completion and receipt completion; B2.3 must supply actual trusted transport/authentication. B3, C1/C2 installed operation, final Astra and public/unattended acceptance remain pending. A1 whole-Docker-inventory drift remains unexplained and unqualified; no cause or waiver inferred. Installed/platform skips remain unqualified. No physical power-loss/ENOSPC/daemon/VM/controller, actual-host, engine/audio/hardware or clean-host claim. No subagents, Docker, network/models/private audio, paid/E2E runner, installs, push/PR/deploy, prune or user cleanup; unrelated research/worktrees preserved.

Frozen focused command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py -q
```

**252 passed in 70.55s, exit 0**, `/tmp/b22-focused-final.log`, source/tests `b999c129c8cea52d637d270c977995cf11dca32b`. This includes 65 new prefix cases and existing B1/B2.1/packaged lifecycle preservation. The full service/release run follows sequentially on unchanged source/tests; no redundant affected-full campaign.

Frozen full preservation command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**522 passed, 18 skipped in 177.95s, exit 0**, `/tmp/b22-full-final.log`. Sequential after the focused run, on unchanged source/tests `b999c129c8cea52d637d270c977995cf11dca32b`. Skips remain unqualified. Final `git diff --check` passes. `git diff 5959d8fb9b577c5e6b4cf5f30276f712a4ee4fa1 c8d3fa0f53f6fcead11467ab96ca81c59fb09276 -- service worker tests/service` is empty: actual starting production/tests match the independently reviewed B2.1 predecessor. Complete baseline-to-candidate changes are exactly recovery.py, recovery tests and this report. The final report-only commit follows the frozen source/test commit; exact candidate SHA is returned with the handoff. No independent review or integration GO is asserted.

## B2.2a round-1 NO-GO targeted correction — awaiting fresh independent review

2026-10-16, gpt-6.1-sol / low implementation. Only worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-c72faa32` changed; current branch `t3/b22a-durable-resume`. Clean root HEAD was `177cdd7`. Inspected actual `c8d3fa0..903b44367bbbd07724b17d24627d693dfc1c308f` chain: exactly source `b999c129` then report `903b4436`, imported oldest first as `ad91496` then `76b8fd7`. Root reviews/docs remain preserved. Read full P1B22a-r1-sol, boundary architecture/resume/B2.2, this complete handoff/corrections and P1B21-r3/protocol. No applicable AGENTS.md found. Memory registry search returned no relevant B2 guidance; no memory-derived acceptance used.

Production correction commit: `ba120cb81c967a4fd33f969ca0e5c65960cc83bd`; final frozen source/test commit is recorded below after the explicit caller-state-discard test strengthening. Correction changes only `service/recovery.py`, `tests/service/test_recovery.py`, and this appended response. No namespace adapter dependency or broader allowlist amendment was needed. All apply entrypoints remain unconditionally unavailable; enrolled activation remains closed, with no new flags.

### F1 response: owned durable restart input

The earlier paragraph describing caller-retained RAM snapshot bytes as a later transport requirement is **superseded and was insufficient for B2.2a acceptance**. The old exception matrix's caller-RAM replay is not process-restart evidence. Normal restart now requires no caller `snapshot_bytes`; the optional private input remains validated and cannot repair missing/corrupt staging.

Under both existing locks, after exact prestate/proof/initial-expiry checks and prospective grounded snapshot validation, order is: marker; full accepted receipt document; pending fence; fixed staging creation descriptor; exact original snapshot staging bytes; file and receipt-parent barriers; Core accepted-intent commit; pending step commit; existing final snapshot/manifest/original-journal creation descriptors and artifacts; evidence step commit. Receipt-first pending ordering remains intact. Snapshot is evidence only, never restored over Core or reconstructed from the mutated pending database.

New receipt names are only `p1-recovery-<accepted recovery_id>.bundle.intent.json` and `.snapshot.bundle`. Descriptor bytes are exactly canonical `{revision,recovery_id,intent_digest,source_digest,snapshot_digest}`; accepted-intent digest transitively binds archive ID, plan, proof, source history, lane/epoch and original physical snapshot digest. The bundle is exact original bounded SQLite snapshot bytes. Existing protected no-follow/nonblocking readers check owner, regular-file identity, single link, bounded stable bytes; `_prefix_exact` additionally requires private permissions. Exclusive 0600 creation, file fsync and parent fsync precede Core acceptance. Both new files consume the existing bounded receipt namespace; no enlarged limits or general path writer.

On resume descriptor and bundle are verified/re-barriered, SHA-256 and grounded logical snapshot/source are checked again, then those exact bytes feed the final archive. A recorded staging creation intent with missing/partial/wrong bundle quarantines without recapture or overwrite. An accepted Core row without staging also quarantines even if caller supplies correct bytes. Existing final-artifact creation-intent missing/partial/wrong quarantine remains unchanged.

Actual `os.fork` child exits via `os._exit(73)` at both Core-intent commit and pending-step commit. Parent discards intent/observation/snapshot caller values, reloads only the owned persisted accepted document, and resumes twice after TTL with no snapshot argument. Assertions retain one accepted recovery, two prefix steps, one final archive and identical recovery/lane/epoch. Core and worker refuse pending across both restarts.

### F2 response: every completed predecessor receipt on every invocation

Before any new prefix mutation, iterate all validated predecessor accepted records. Derive the first four exact expected step documents, immutable Core certificate, and fixed receipt names. Compare actual bounded protected certificate bytes against canonical `{certificate,steps}` and original journal bytes against the accepted before-image. Missing or changed earlier journal/certificate refuses even after the current pending fence is durable. No history removal, certificate rewrite or corruption repair. Existing control-archive/source/history grounding remains.

Regression fixtures now retain two completed predecessors and start a third pending recovery; separate changed/missing oldest journal and certificate cases refuse on two invocations, while both unchanged variants resume coherently.

### RED/GREEN and boundary evidence

Existing interpreter only; every pytest command uses `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider`, no installs. Disposable test roots only.

- Real RED on imported candidate, `tests/service/test_recovery.py -k b22_correction -q`: **6 failed, 2 passed, 238 deselected in 6.27s**, `/tmp/b22-correction-red.log`. Both actual child-death restarts failed at missing snapshot; all four changed/missing predecessor receipt variants reached missing-refusal assertions. Unchanged positives passed.
- Initial same-reproducer GREEN: **8 passed, 238 deselected in 5.33s**, `/tmp/b22-correction-green.log`.
- Expanded prefix/correction matrix: **89 passed, 173 deselected in 23.20s**, `/tmp/b22-correction-dev.log`.
- Strengthened two-predecessor history and actual snapshot/bundle file-fsync/parent-sync failures: **10 passed, 254 deselected in 7.08s**, `/tmp/b22-correction-history-barrier.log`.

New staging fault points cover creation descriptor create/write/fsync/parent/recorded, bundle create/write/fsync/parent. Incomplete descriptor or absent/empty bundle quarantines before Core acceptance; complete exact outputs resume preserving inode/bytes. Separate post-acceptance missing/changed bundle, changed descriptor, public mode, symlink, extra hard link and oversized bundle refuse twice even with supplied original caller bytes. Existing archive creation fault matrix and no-remint behavior remain. No development failures after production correction; frozen focused/full receipts follow.

E1 local source/SQLite/filesystem/process and mocked host evidence only. Skips are not qualifications. A1 whole-Docker-inventory drift remains unexplained and unqualified; no inferred cause or waiver. Remaining B2.2 tranches/B2.3, B3, C1/C2, root integration, final Astra and public/unattended acceptance remain pending. No actual-host transport, physical power-loss/ENOSPC/daemon/VM/controller, engine/audio/hardware or installed qualification. No subagents, Docker/network/models/audio/paid/E2E/cloud/STT/Voicebox, installs, push/PR/deploy, prune or user cleanup; unrelated research/worktrees preserved. Ready for fresh independent review after the frozen receipts; no self-GO.

First focused run on `ba120cb`: **278 passed in 85.66s**, `/tmp/b22-correction-focused-final.log`. Superseded for final frozen-source attribution by the following run: the child-death test was strengthened to explicitly delete fixture snapshot bytes before fork (rather than merely never supplying them) and directly compare final archive bytes to staging. Production code did not change. This justified the repeated focused check; the full service/release campaign runs once on the final source/tests.

Final frozen source/tests: `9d273aad1d50be38ecb66ec6cd4dbad06e1b1c84` (production `ba120cb`, explicit discard/direct byte-equality test strengthening `9d273aa`). Only this report changes during final verification.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py -q
```

**278 passed in 82.78s**, exit 0, `/tmp/b22-correction-focused-final2.log`. Includes 26 added/expanded correction cases over the imported 252 focused tests. The single full service/release run follows sequentially on this unchanged source/test commit.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**548 passed, 18 skipped in 188.28s**, exit 0, `/tmp/b22-correction-full-final.log`. Sequential after final focused verification on unchanged `9d273aad1d50be38ecb66ec6cd4dbad06e1b1c84`. The 18 skips remain unqualified. No redundant full campaign. Final `git diff --check` passes; complete fresh-root baseline `177cdd7` to candidate path set is exactly recovery.py, recovery tests and this report. Protected root reviews/ledger, Core/initializer/supervisor/reconciliation and all other paths are preserved. This report-only commit follows frozen source/tests; exact candidate SHA accompanies handoff. Fresh independent review and root integration verification remain required; no self-GO.

## B2.2b — private durable journal rotation (awaiting independent review)

Implementation gpt-6.1-sol / low, 2026-10-16 environment date. Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-7d5b1e82`, branch `t3/private-journal-rotation`. Actual clean starting HEAD `a6451ccb8172e99213b9d6841613b0ab050f89d1`, containing integrated `ab0c7da`. `git diff 2f7d070 ab0c7da -- service/recovery.py worker/runtime_journal.py tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py` is empty: those baseline sources/tests match the exact reviewed B2.2a candidate. Predecessor R2 reports 548 passed/18 skipped; root ledger reports 24 targeted passed/240 deselected in 11.31s. Those are inherited receipts, not fresh runs of this tranche. Read boundary architecture/closed records/cross-volume authority/resume/B2.2 exits, complete handoff and corrections, both B2.2a reviews, B2.1-r3 and orchestration/R1 authority context. No applicable AGENTS.md found. Memory supplied only general exact-checkout/evidence-tier guidance, no B2 acceptance or authority.

Frozen production/tests commit: `8864b6fa50d6418a2a468cb040a50eb47e90c4e0`. Changes only recovery.py, runtime_journal.py, recovery tests, and this appended report. No additional dependency/amendment needed. No Core, initializer, supervisor, host, reconciliation, resource/admission/dispatch store, schema/contracts/dependencies/packaging, ledger or review changes. All public/programmatic/host apply entrypoints stay unconditionally unavailable. No activation toggle.

Private interface: `_pending_through_journal(roots, scope, intent, observation, *, now=None, fault=...)`. It reuses the accepted closed record and its exact third expected output; it accepts no arbitrary output/path/writer/death Boolean. Existing `_pending_through_evidence` remains its original bounded stop; both delegate to the same private locked validator/sequence. `fault` names deterministic boundaries only and is not supplied any write operation. E1-mock proof inputs exercise internal validators, not actual-host authentication.

Ordering under BOTH normal nonblocking Core/lane locks on EVERY invocation:

1. Validate accepted record/plan/proof, death/consumer coverage, exact source prestate, schema/media/history/affected set and all predecessor control and receipt evidence. Initial expiry refuses; an already accepted identical intent can resume after TTL. Retained predecessor journal/certificate bytes are checked even after the new fence supersedes their fence.
2. Preserve B2.2a receipt marker/full accepted document/pending fence, original snapshot bundle durable before Core acceptance, pending receipt, grounded snapshot/manifest and exact original journal receipt archive, and verified evidence receipt. Original journal absence/corruption retains exact status/bytes; no clean receipt is inferred.
3. Before replacing journal, exclusively create/fsync/parent-sync fixed `p1-recovery-<accepted recovery_id>.rotation.intent.json`. Its canonical bytes bind revision, recovery ID, full accepted-intent digest, exact expected journal step including input/output digests, original journal before-image/status, and expected new envelope SHA-256. The finite receipt budget includes this descriptor and replacement temporary before creation. No lane/epoch is generated here.
4. Explicit worker `_maintenance_replace` uses the same factored `_durable_save` as ordinary `Journal.save`: write fixed `journal.tmp`, flush, file fsync, replace fixed `journal.json`, parent-directory fsync. Maintenance uses exclusive temporary creation and pending-fence/new-lane binding. It never constructs Journal or bypasses its startup fence. Ordinary Journal save/load behavior retains its existing path and startup checks.
5. Verify exact canonical new envelope, protected owner/private regular single-link file and bytes; repeat file/parent barriers. Persist the exact third `journal` step via BEGIN/write/COMMIT with existing FULL SQLite semantics; validate the step chain/archives again. Return `pending-through-journal`. Marker and fence remain pending with no completion certificate.

Resume accepts only the precise expected current new journal when the exact rotation descriptor exists. It verifies/re-barriers and records a missing step receipt without replacement or a new lane. A recorded attempt with absent, old, partial or wrong output quarantines; an unfinished temporary remains an unexpected namespace entry and is never removed/retried automatically. Before/after projection accounts for this one exact verified current journal by replaying its recorded original bytes/status into the original source comparison; every other journal/history/SQL/inventory/media obligation stays bound. Current recovery rows are independently validated against only the allowed contiguous prefix steps; no history table is blanket ignored. Accepted history retains old/new lane mapping and old runtime evidence. No Core/job/resource/resolution/epoch completion or receipt completion is performed.

Tests use actual private production helpers with real local SQLite/WAL/files/flock. Deterministic faults cover journal-intent create/write/fsync/parent/recorded, journal write/file-fsync/replace/dirsync, and journal step BEGIN/write/COMMIT. Pre-output attempts quarantine twice; exact post-replace output resumes twice retaining inode/bytes and one SQL step. Actual os.fsync and os.replace denials separately exercise file/replace/directory boundaries. Actual fork child calls os._exit(73) immediately after replace before dirsync/step receipt; parent discards intent/observation/snapshot variables, launches a fresh interpreter with only owned root paths/scope/recovery identity, reloads the durable accepted document and resumes TWICE after TTL. Both normal constructors refuse pending before both restarts; one accepted recovery, one new lane, three steps and one archive remain.

Separate changes to current journal/lane/hash, symlink/hardlink, missing output, receipt original-journal archive, control manifest, media, history, proof or extra entry refuse twice. Missing/corrupt old journal coherent positives run the rotation helper and retain exact original archive status/bytes; omitted/live coverage negatives refuse. Coherent UNKNOWN dispatch positives retain original jobs/dispatches/ACK/accounting/monitor/fault/event rows; omitted/live/unknown coverage negatives refuse. Both locks and unaccepted expiry refuse. Extended prior-receipt matrix retains two completed predecessors and verifies changed/missing oldest journal/certificate refusal after rotation, plus unchanged twice-resume positives. Real allocated cancel-requested work and terminal media/history preservation run through both evidence and journal helpers.

RED: `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b22b -q`: 1 failed/264 deselected in 1.37s at absent production helper (`/tmp/b22b-red.log`). Initial GREEN 1 passed/264 deselected in 1.86s. Expanded first development run: 23 passed/5 failed in 7.20s, all five failures from missing test import `os`, corrected without production changes. Subsequent development receipts: 40 passed/258 deselected in 10.08s (`/tmp/b22b-dev2.log`); expanded allocated/media/predecessor matrix 44 passed/262 deselected in 24.16s (`/tmp/b22b-dev3.log`); UNKNOWN/expiry/lock additions 5 passed/306 deselected in 3.36s (`/tmp/b22b-dev4.log`). Frozen focused/full receipts follow. All commands use the existing interpreter, bytecode/cache disabled, no installs.

E1 local source/SQLite/filesystem/process and mocked host inputs only. No self-GO; fresh independent Sol-low review and parent integration are required. Remaining atomic Core/resource/resolution/receipt completion, B2.3 trusted host transport, B3, C1/C2 installed proof, final Astra and public/unattended acceptance remain pending. A1 whole-Docker-inventory drift remains unexplained/unqualified, with no cause or waiver. Skips do not qualify installed operation. No physical power-loss/ENOSPC/daemon/VM/controller, actual-host, audio/engine/hardware or clean-host claim. No subagents, Docker/network/models/private audio/paid/E2E runner, push/PR/deploy, prune or user cleanup. Unrelated research/worktrees preserved.

Frozen focused verification on `8864b6fa50d6418a2a468cb040a50eb47e90c4e0`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py -q
```

**325 passed in 105.95s, exit 0**, `/tmp/b22b-focused-final.log`. Includes 47 added/expanded cases over the 278-test predecessor focused suite. The full service/release run follows sequentially on unchanged source/tests. Only this report is edited during final checks.

Frozen full preservation verification, sequential after focused verification on the same `8864b6fa50d6418a2a468cb040a50eb47e90c4e0`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**595 passed, 18 skipped in 212.40s, exit 0**, `/tmp/b22b-full-final.log`. The 18 skips remain unqualified. No source/tests changed between or during frozen focused/full runs; no redundant full campaign. Final `git diff --check` passes. Baseline-to-candidate changed paths are exactly service/recovery.py, worker/runtime_journal.py, tests/service/test_recovery.py and this report. All protected paths remain unchanged. This final report-only commit follows the frozen source/test commit; exact candidate SHA is returned in the handoff. Fresh independent review is required, with no self-GO.

## B2.2c — private atomic Core completion (awaiting independent review)

2026-10-16, implementation gpt-6.1-sol / low. Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-bb9261f6`, branch `t3/b22c-atomic-core-recovery`. Actual clean baseline was `e52df2fe28aaafbff381fe3d650d92b7aa6b08ae`, not the supplied integration ancestor `fdd0e5a`. `git diff fdd0e5a e52df2f -- service tests` is empty. Thus the actual baseline source/tests retain the reviewed journal prefix; the additional commit records its acceptance and this tranche's scope. Inherited 595-pass/18-skip review and 33-pass/12.59s root receipts are historical evidence, not fresh execution here. Fresh baseline command, existing interpreter with bytecode/cache disabled, `tests/service/test_recovery.py -k b22b -q`: **33 passed, 278 deselected in 12.19s**, `/tmp/b22c-baseline.log`.

Read the full boundary architecture, records, authority/reconciliation amendment, resume rules and B2.2 exits; the full recovery handoff including corrections/tranches; P1B22b-r1, P1B22a-r2, P1B21-r3, orchestration protocol, R1 fault-authority and original P1 design/B2/exit matrix. No applicable AGENTS.md found. Memory supplied general exact-checkout/evidence-tier guidance only; no B2 authority or acceptance came from memory.

Initial production/tests commit: `1d32289e5e55fdd441b7214828a917f6f78651e1`, superseded by the uncertified-window correction below. Exactly seven source/test paths change there: recovery, persistence, resource_store, dispatch_store, dispatcher, reconciliation, and recovery tests. This appended handoff is the eighth and only subsequent path. No additional dependency/amendment was needed. No Core, initializer, supervisor, host script, worker, admission, contracts, dependencies, packaging, ledger, review or unrelated file change. Private/public versions and exact SQL schemas/columns are unchanged. All apply entrypoints remain unconditionally unavailable, enrolled activation remains closed, and no activation toggle exists.

### Interfaces, authority and ordering

`_pending_through_core(roots, scope, intent, observation, *, now=None, fault=...)` extends the existing private production prefix under BOTH existing nonblocking locks. It accepts the same validated closed accepted intent, exact supplied observation/provenance, all-owner death coverage, fixed lane/epoch, original canonical before-image and five expected step descriptions. It adds no death Boolean, PID-1 receipt, timeliness, ACK or arbitrary action/SQL/path input. E1-mock is internal validator evidence only, never actual-host authentication. Initial TTL and exact accepted-after-TTL resume rules remain.

The reviewed evidence and journal prefix remains durable first: owned snapshot bundle and creation descriptor; Core accepted intent/pending receipt; grounded snapshot/manifest/original journal archives; evidence receipt; fixed durable journal rotation and journal receipt. Core completion adds:

1. Validate complete original source/history/media/typed references and exact affected set. Missing/corrupt journals retain the original status/bytes with full death coverage. UNKNOWN/unbound dispatches remain original evidence. Unknown schema/references, reservation origins and foreign namespace cannot be acknowledged away. Require explicit exact decisions for degraded media/orphans; arbitrary historical authority defects cannot be erased.
2. Private `_recovery_scan(db, (core_root, staging_root), intent)` requires durable accepted pending authority, no active SQL transaction and no Core completion. It never constructs Core/ResourceStore, borrows an ordinary token, or invokes `observe`. Bounded entry/deadline/type/owner/namespace/actual-byte measurement, original attempt/output bounds and free reserve precede authority changes. Compute `max(old highwater, actual)` and include all original cumulative disk charges. The actual result must equal the accepted closed Core output. Quota/free-reserve/output or changed-output refusal stays pending. No cleanup credit or automatic budget increase; budget changes are explicitly unsupported in this tranche, including reader validation. Enrollment stays immutable.
3. One `BEGIN IMMEDIATE`/write/`COMMIT` updates only the exact permitted unsuccessful job JSON delta, retained reservations, monotonic accounting, healthy new resource monitor, cleared active resource faults, and healthy A2 integrity state with cleared inherited token. The accepted before/source and grounded snapshot retain the old monitor, fault reasons, demands/highwater, A2 state/token and complete earlier history. The same transaction appends exact dispatch resolutions, publication retirements, diagnostic resolutions, and the `core-complete` step binding the distinct new epoch and verified output rows. No events, original dispatch/commands/receipt/outcome/ACK, attempts outside the prescribed latest reap bit, terminal JSON, artifacts, idempotency rows or original publication/work records change. The full prospective DB/history/archive population and exact future certificate bytes are budget-validated before COMMIT.
4. After durable Core completion, exclusively create the fixed receipt-side `p1-recovery-<recovery_id>.core-certificate.intent.json`, then the existing fixed control `p1-recovery-<archive_id>.certificate.json`. Descriptor canonical keys are exactly `revision recovery_id intent_digest sha256`. Descriptor and certificate use protected 0600 exclusive creation, complete writes, file fsync and parent barriers; they consume existing budgets. The immutable certificate is the existing closed certificate bound to the fourth exact SQL step. No receipt completion file/step/fence is produced: `p1-maintenance.json` remains PENDING, with the new empty journal lane retained.

Transaction exceptions stop the session; no live retry assumes rollback. Best-effort rollback preserves the original error if rollback itself fails; owning `offline` closes the connection and releases locks. Resume uses a newly opened connection. A committed Core result without certificate/creation intent undergoes exact after-image verification, original-projection reconstruction and unaffected-history/media comparison before certificate creation. Only the current declared Core rows, integrity transition, exact current resolution rows, named steps/references and current archive entries are projected to their original before-images; all unaffected SQL/history/media and all prior receipt/control archives stay bound. There is no blanket history/table omission and no requirement that the old source digest survive its own expected mutations.

If certificate creation was attempted, absent/partial/wrong output quarantines under the existing no-remint policy. An exact output is verified and re-barriered without overwrite, then returned. Once certified, replay validates the immutable chain, complete exact resolution sets, all prior archives and current owned accepted/bundle/creation/rotation evidence without a new inventory/resource scan, lane, epoch, events, resolutions or outcome writes. Later ordinary rows/current media do not replay an old recovery; current startup eligibility remains a separate obligation. This private return explicitly says `core-complete-receipt-pending`; it never grants startup or activation. Scope/proof/history/evidence changes refuse. Both persistent startup modes still refuse because pending worker maintenance supplies no hello. Direct historical Core may open after its certificate, but both publication wrappers permanently refuse exact retired publications.

### Resolution semantics and exact amendment

Completed history now requires the entire finite resolution set derived from the accepted original tuples; a missing/deleted resolution cannot expose an old dispatch. Pending resolution rows have no filtering effect. `DispatchStore.current()` excludes only validated completed dispatch resolutions. Allocation and Dispatcher replacement reject jobs covered by completed operator actions, including jobs without publications. Explicit new retry jobs retain ordinary lineage.

A2 consumes only completed recovery identities and unchanged original dispatch tuples, exact expected recovered job/reservation rows, and the known old absent-receipt/outcome/unbound mismatches. New owner/binding/receipt/publication defects remain unresolved. Historical dispatch diagnostics are acknowledged only when their original identity, reason and content match the accepted before-history and exact completed dispatch resolution. Degraded diagnostic decisions bind the original object ID, classification/namespace/name, identity/content and owner tuple, reason and availability. `diagnostic_digest` hashes that complete defect document excluding only its separately retained typed-reference list: adding a recovery audit reference is not a change to the defect's bytes/tuple. Unknown-reference/schema or arbitrary owner defects cannot be accepted as media degradation. Raw diagnostics, reference traversal and unavailable reports remain. Strict asset/artifact retrieval remains unchanged. Ordinary R1/A2 inherited-checking rules are unchanged; only the explicit proof-bound recovery transaction creates the new authority epoch.

### Deterministic E1 verification and fault matrix

Tests call actual production helpers against temporary SQLite/WAL/files/locks, with no alternate recovery writer. Existing complete-record fixtures now include the required resolution set; they remain record-oracle fixtures, while the new B2.2c tests execute the production writer. Existing damaged-journal cases additionally exercise the Core continuation. No sleeps or accept-either outcomes were added.

- Real queued, allocated failure/cancel-requested, UNKNOWN/unbound and terminal publication positives; exact canonical job delta, frozen events/attempt history/artifacts/idempotency, original dispatch/commands/receipt/outcome/ACK, demand/disk charges and media preservation; permanent old valid publication retirement before either decorated body.
- Inherited R1 checking after actual denied fault persistence and immediate `os._exit(73)`: two ordinary writable restarts cannot heal; explicit recovery archives old checking and creates a distinct epoch. Separate inherited A2 checking/token and known resource-fault histories are archived before active settlement.
- Faults at fresh scan, Core SQL BEGIN/write/COMMIT/parent, certificate intent create/write/fsync/parent/recorded, and certificate create/write/fsync/parent. Precommit faults roll back jobs/resources/resolutions/step together and retain pending. Postcommit faults retain the complete same epoch. Recorded missing/partial outputs refuse twice; exact outputs resume twice without replacement.
- Actual SQLite authorizer denial of the resource-monitor UPDATE rolls back all Core authority changes. Actual certificate file/parent fsync denials leave complete output for verify/re-barrier resume. Physical disk/VM/controller/power-loss failure is not claimed.
- Actual fork death at Core transaction write and post-COMMIT, then a fresh Python interpreter supplied ONLY owned root paths/scope/recovery ID. It reloads the durable accepted document, resumes twice after TTL, and asserts exactly one recovery, epoch and resolution set/four Core steps. No caller intent/observation/snapshot RAM supplies restart authority.
- Both persistent startup modes refuse twice on Core-complete/receipt-pending; worker cannot hello. Completed replay has forbidden inventory/resource scan functions and remains a no-op. Missing/changed immutable evidence/history/resolutions/proof/scope refuses. Current new media activity remains separate from immutable completion.
- Exact old unavailable-media and dispatch diagnostics are accounted; changed old content/reason, new orphan or unknown references remain blocked. Terminal success history and strict unavailable retrieval persist. Quota/free-reserve and changed expected high-water refusals leave three prefix steps and no partial healthy Core completion.

Every command uses `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider`, existing environment only. Real initial RED, `tests/service/test_recovery.py -k b22c -q`: **1 failed, 311 deselected in 1.91s**, absent production continuation (`/tmp/b22c-red.log`). First GREEN: **1 passed, 311 deselected in 0.74s**, `/tmp/b22c-green1.log`.

Development receipts (superseded by frozen verification): B2.2c matrix **13 passed/311 deselected, 12.47s** (`dev2`), **19/311, 17.16s** (`dev3`); expanded B2.1 output/retirement/history preservation **58/272, 41.22s** (`dev4`); enlarged Core matrix **33/311, 29.83s** (`dev5`); exact diagnostic/terminal/no-scan subset **7/337, 8.23s** (`dev6`); Core/damaged-journal matrix **66/299, 45.54s** (`dev7`); strengthened actual fault-death/startup subset **3/362, 2.06s** (`dev8`); exact dispatch diagnostic test **1/365, 1.83s** (`dev9`); Core matrix **49/317, 44.45s** (`dev-final`); final completed/projection/crash subset **21/345, 22.16s** (`dev10`). Log paths are `/tmp/b22c-<name>.log`. No earlier development test failures after the initial missing-helper RED; the later uncertified-window RED and correction are recorded below. These evolving-source runs do not replace the final frozen suite below.

E1 local source/SQLite/filesystem/process and mocked host inputs only. Skips are not qualifications. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no cause or waiver. Receipt finalization is the next tranche; B2.3 trusted transport, B3, C1/C2 installed proof, parent integration, independent Sol-low review, final Astra and public/unattended acceptance remain pending. No actual-host authentication, physical power-loss/ENOSPC/daemon/VM/controller, engine/audio/hardware or clean-host qualification. No subagents, Docker/network/models/audio/paid/E2E runner, installs, push/PR/deploy, prune or user cleanup; unrelated research/worktrees preserved. No self-GO.

Initial focused verification (superseded after the correction below) on unchanged `1d32289e5e55fdd441b7214828a917f6f78651e1`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**498 passed in 225.43s, exit 0**, `/tmp/b22c-focused-final.log`. Includes 55 added/expanded recovery cases over the 311-test recovery baseline. Only this report changed while running. The initial full run was subsequently interrupted for the correction below; it supplies no complete preservation result.

### Pre-handoff correction: committed but uncertified evidence window

A final read-through found that the post-COMMIT/no-certificate branch validated the grounded control snapshot but returned before revalidating the current owned staging bundle/descriptor and receipt-side original-journal archive. Thus a changed owned restart input could receive a first Core certificate, even though the next completed replay would reject it. This was a concrete implementation defect, not an independent-review finding. No GO is inferred from the earlier passing tests.

Stopped only this task's ongoing full-suite process (exact PID 47230, parent 10481, matching `python -m pytest ... tests/service tests/release -q`) with SIGTERM before any source/test edit. Exit **143**, `/tmp/b22c-full-final.log`, partial progress through 53% with no displayed failures; this interrupted run is **not** a full verification receipt. The initial 498-pass focused result above remains accurate for `1d32289` but is superseded for final-candidate attribution. This concrete correction is the reason for repeating focused/full checks.

Added four actual post-COMMIT/fresh-invocation regressions: missing bundle, changed bundle, changed descriptor, and changed original-journal archive. Before correction, `tests/service/test_recovery.py -k uncertified_core -q`: **4 failed, 366 deselected in 5.90s**, `/tmp/b22c-uncertified-red.log`; all reached the missing-refusal assertion. Correction adds only the same existing `_completed_prefix_receipts` verification before certificate creation on both the immediate post-COMMIT and uncertified resume paths. No new schema/interface/authority or record delta. Missing/changed inputs now refuse twice before a certificate appears.

GREEN, `tests/service/test_recovery.py -k 'uncertified_core or b22c_atomic_faults or b22c_process' -q`: **15 passed, 355 deselected in 19.37s**, `/tmp/b22c-uncertified-green.log`. Final frozen source/tests commit is **`766dc11d2ee736ddf70d2b755bc0c577163bf95a`**, following initial implementation `1d32289`. The two-call correction and four cases are the complete source/test delta between them. Only this report changes during final verification. Final recovery population is 370 tests, 59 added/expanded over the actual 311-test baseline. Also verified `git diff 5290e50 e52df2f -- service worker tests/service` is empty: the actual starting production/worker/tests match the exact reviewed journal-prefix source.

Final frozen focused verification on `766dc11d2ee736ddf70d2b755bc0c577163bf95a`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**502 passed in 226.88s, exit 0**, `/tmp/b22c-focused-final2.log`. The final full service/release run follows sequentially on unchanged source/tests. Only this handoff is edited during verification.

### Full-suite fixture-lifetime correction and final attribution

The complete run on `766dc11` finished **1 failed, 653 passed, 18 skipped in 268.93s**, exit 1, `/tmp/b22c-full-final2.log`. Failure: `test_b22_prefix_damaged_journal_preserves_status_and_coverage[exact-missing-core]`, `snapshot_source_changed` on exiting the source-preserving private database context. This is a failed preservation receipt, not a pass or waiver.

The newly added Core parameter setup used `with sqlite3.connect(...) as db`, which commits but leaves its connection open. That same function later binds `db` from `offline(...)` after the private source capture has begun; closing the displaced source handle can checkpoint/unlink the source WAL/SHM inside the protected-copy window. A separate deterministic disposable SQLite/WAL probe explicitly demonstrated: (1) transaction-context exit leaves the connection usable and WAL present; (2) closing that source connection inside `private_database` produces the exact `snapshot_source_changed` refusal; (3) closing before entering the copy succeeds. Probe exit 0; no production validation was relaxed. This explains the new fixture handle lifetime only, **not** the unrelated historical A1 Docker-inventory drift, which remains unexplained/unqualified without cause or waiver.

Correction is only explicit `closing(sqlite3.connect(...))` around that new fixture setup, preserving its transaction commit and closing before any snapshot. No production change. Targeted `tests/service/test_recovery.py -k damaged_journal -q`: **18 passed, 352 deselected in 3.70s**, `/tmp/b22c-fixture-close-green.log`. Final frozen source/test commit is now **`d1c732d9ead927e7705e559a775df75ceaf7bbf9`**; production remains exactly `766dc11d2ee736ddf70d2b755bc0c577163bf95a`. Test population remains unchanged. The 502-pass `final2` focused result is superseded for final test attribution by the following corrected-fixture run. This observed failure/test correction is the reason for repeating the final focused/full campaign; no redundant unchanged-candidate broad repeat is claimed. Only this handoff changes during it.

Final corrected-fixture focused verification on unchanged `d1c732d9ead927e7705e559a775df75ceaf7bbf9` (production `766dc11`):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**502 passed in 230.61s, exit 0**, `/tmp/b22c-focused-final3.log`. The full service/release campaign follows sequentially on this unchanged source/test commit. Only this handoff changes during verification.

Final frozen full preservation verification, sequential after the corrected-fixture focused suite on the same unchanged `d1c732d9ead927e7705e559a775df75ceaf7bbf9` (production `766dc11`):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**654 passed, 18 skipped in 267.54s, exit 0**, `/tmp/b22c-full-final3.log`. All 18 skips remain unqualified. No source/tests changed during or between the final focused/full runs. Earlier interrupted/failed runs and their concrete corrections remain recorded above; they are not counted as passing receipts. No further broad repeat is needed.

Final `git diff --check` passes. The complete actual-baseline-to-candidate path set is exactly eight allowed files: `service/recovery.py`, `service/persistence.py`, `service/resource_store.py`, `service/dispatch_store.py`, `service/dispatcher.py`, `service/reconciliation.py`, `tests/service/test_recovery.py`, and this handoff. Protected paths and unrelated research/worktrees are preserved. Local sequence: implementation `1d32289` → uncertified-window correction `766dc11` → fixture source-handle closure `d1c732d` → final report-only commit. Exact final candidate SHA accompanies the handoff, avoiding a self-referential commit hash in this file. Ready for fresh independent Sol-low review and parent integration verification; **no self-GO**. Receipt completion remains the next tranche; trusted B2.3 transport, B3, C1/C2, final Astra and public/unattended acceptance remain pending.

## B2.2c round-1 F1 correction — awaiting fresh independent review

2026-10-16, gpt-6.1-sol / low implementation. Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-0f37e098`, branch `t3/b22c-recovery-context-fix`, clean root baseline `ac92b4175df53c03ea97672a894af17733dd4ac4`. Verified the complete `e52df2f..c5173055bce21101621f25fb4d868495a7b7e2c3` chain and imported all four commits oldest first: `1d32289e5e55fdd441b7214828a917f6f78651e1` → `766dc11d2ee736ddf70d2b755bc0c577163bf95a` → `d1c732d9ead927e7705e559a775df75ceaf7bbf9` → `c5173055bce21101621f25fb4d868495a7b7e2c3`, locally `b888bba` → `cf03131` → `c8ea7a2` → `4e53399`. Root documentation/reviews are preserved; no original/root checkout edits.

Read the full P1B22c-r1-sol F1 reproducer/minimal fix/other probes, boundary architecture/records/resume/resolution amendment/B2.2, this handoff including all corrections, orchestration protocol, P1B22b-r1 and R1 fault-authority context. No applicable AGENTS.md found. Memory supplied only general exact-checkout/evidence-tier guidance, not acceptance evidence.

F1 response: reconciliation accumulates independent dispatch join failures rather than overwriting them. A completed resolution consumes only the three permitted old unbound/missing receipt/outcome reasons, and only when no strict current defect exists. It compares the entire current recovered SQL job row (all columns, including complete context), and the complete reservation, to the accepted canonical Core output. Original Authority/binding/current attempt/generation/deadline/context/configuration/receipt/publication checks remain. Every detected reason stays in the summary; the stable inventory diagnostic prefers a strict defect. No text-field blacklist, completed-replay mutation, or additional current-state restriction in immutable replay was added. Pending resolutions have no effect. Raw history, typed references, availability and unknown-schema/origin refusals remain.

Actual correction delta after import is only `service/reconciliation.py`, `tests/service/test_recovery.py`, and this appended response. Frozen source/test commit: `45f8cd7778b5d423c7536eb059e5d70dbed45e71`. Public 1.3.0, Core user_version 1, R1 protocol 2, private P1 revision 1 and exact B2 schemas/record keys are unchanged. No Core/initializer/supervisor/host/contracts/dependencies/packaging/ledger/reviews edits. Both locks/death provenance, owned bundles/prior archives, atomic SQL/certificate ordering, monotonic charges/high-water, distinct recovery epoch and ordinary R1/A2 checking refusal remain as imported. Budget changes remain unsupported. All apply entrypoints remain unavailable and enrolled activation closed; receipt fence remains PENDING.

Regression uses real Admission/DispatchStore allocation and the production `_pending_through_core` writer, then independently varies one current field and opens two fresh Core sessions. Exact F1 changes only `context.plan.request.text`; related cases change context bundles, SQL principal, input hash, attempt owner or generation. The unchanged control must be clean twice; every mutation must block twice while original ACK/receipt/outcome remain unchanged. Existing old-diagnostic/no-scan/pending tests are retained.

Commands use the existing interpreter, bytecode/cache disabled, no installs:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k current_joins_two_startups -q
```

RED on imported candidate: **3 failed, 4 passed, 370 deselected in 11.37s**, exit 1, `/tmp/b22c-f1-red.log`. Exact text-only F1, bundle and SQL principal changes were falsely clean on both startups. Input/owner/generation already blocked. One edit command failed with `zsh: command not found: python` before changing source; its following nine-case run therefore remained RED (**3 failed, 6 passed, 368 deselected in 13.03s**). Reissued the edit with the specified interpreter; no validator relaxation.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k 'current_joins_two_startups or dispatch_diagnostic_history or completed_no_scan' -q
```

GREEN: **9 passed, 368 deselected in 12.91s**, exit 0, `/tmp/b22c-f1-green.log`. Final frozen focused/full receipts follow; no source/tests change during them.

E1 local SQLite/filesystem/process and mocked host documents only; E1-mock does not authenticate actual host authority. Skips remain unqualified. A1 Docker-inventory drift remains unexplained/unqualified, with no inferred cause or waiver. Receipt completion, B2.3 trusted host transport, B3, C1/C2, parent integration, fresh independent review, final Astra and public/unattended acceptance remain pending. No physical power-loss/ENOSPC/daemon/VM/controller, installed lifecycle, engine/audio/hardware or clean-host qualification. No subagents, Docker/network/model/audio/paid/E2E runner, push/PR/deploy, prune or user cleanup; unrelated research/worktrees preserved. No self-GO.

Frozen focused verification on unchanged `45f8cd7778b5d423c7536eb059e5d70dbed45e71`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**509 passed in 235.40s**, exit 0, `/tmp/b22c-f1-focused-final.log`. Full service/release follows sequentially on unchanged source/tests; only this report is edited.

Frozen full preservation verification, sequential after focused on the same unchanged source/tests:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**661 passed, 18 skipped in 272.07s**, exit 0, `/tmp/b22c-f1-full-final.log`. All skips remain unqualified. No source/test changes during/between final runs and no redundant broad campaign. `git diff --check` passes. Protected comparison against actual root baseline `ac92b41` is empty for Core/initializer/worker/contracts/reviews/ledger. Complete imported-plus-correction path set remains the original eight authorized paths. Correction-only delta from `4e53399` is exactly reconciliation, recovery tests and this handoff. Final report-only commit follows frozen `45f8cd7`; its exact SHA accompanies handoff. Fresh independent review of that exact final candidate and parent integration verification are required; no self-GO.

## B2.2d — final private receipt completion (awaiting independent review)

2026-10-16, implementation gpt-6.1-sol / low. Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-11ad9433`, branch `t3/b22d-private-receipt-completion`. Actual clean baseline `50ee0b0eeed91cd1ef3836da4a27457a19180692` contains root acceptance/scope documentation after supplied `c6f2629`; `git diff c6f2629 50ee0b0 -- service worker tests` is empty. Reviewed predecessor `3f411135` and its R2 results remain historical evidence. Frozen source/tests commit: `1ceec7726f799c80d586ac9ed5e5f835e8374b78`. Only this appended report changes afterward. No self-GO; fresh independent Sol-low review and parent integration verification are required, followed by final Astra at the assigned integrated boundary.

Read full P1B2-BOUNDARY-ASTRA architecture, records, resume, authority, completion handshake and B2.2; full preceding recovery handoff/contracts/tranches/corrections; P1B22c-r1/r2, P1B22b-r1, orchestration protocol and R1 fault-authority context. Inspected actual production readers/writers, enrollment/Dispatcher startup and activation closure. No applicable AGENTS.md found. Memory provided only general exact-checkout and evidence-tier guidance, no B2 authority/current acceptance.

Changes are exactly `service/recovery.py`, `tests/service/test_recovery.py`, and this appended handoff. Existing persistence/Journal/supervisor readers already consume the frozen completion documents; no changes to those files were necessary. No Core, initializer, supervisor, Dispatcher, host, reconciliation, resource/admission/dispatch stores, contracts, dependencies, packaging, ledger or review changes. Public 1.3.0, Core 1, R1 protocol 2, private P1 revision 1, exact SQL schema and all closed record keys remain unchanged. All public/programmatic/CLI apply entrypoints remain unconditionally unavailable; enrolled activation remains closed. No activation toggle, engine, retry or B2.3 transport is added.

### Private interface and exact durable sequence

`_complete_receipts(roots, scope, intent, observation, *, now=None, fault=...)` continues the existing production prefix through final receipt completion under BOTH existing nonblocking control/lane ownership locks for the whole invocation. It accepts only the existing validated immutable accepted intent/proof/observation; no arbitrary path, SQL, death Boolean or caller mutation callback. `fault` is solely the existing internal deterministic boundary hook. E1-mock proof documents exercise validators, never authenticate actual host death.

Every invocation validates accepted plan/proof/scope/all-retained-owner death coverage and immutable Core history/control archives, current owned intent/bundle/creation/rotation/original-journal evidence. Previously completed receipt archives remain exact. Histories with the new fifth step additionally require all retained production bundle/rotation/creation inputs and all three final creation descriptors. No whole history/journal/resolution omission or new lane/epoch generation is introduced. Initial acceptance TTL and accepted exact resume after TTL remain distinct. Original B2.2c strict current authority/context joins remain unchanged.

The reviewed Core transaction, durable fourth step and actual protected immutable Core certificate precede all finalization. The fourth-step certificate remains the exact existing certificate; fifth-step completion does not replace or rehash it.

1. Preflight the finite receipt namespace/budget and every existing final creation attempt/output, before creating any later artifact. Unknown entries and leftover temporaries refuse. Files verified by `_prefix_exact` must now have exactly owner-private 0600 permissions as well as the existing no-follow, owner, single-link, regular-file, bounds, exact-byte and repeated file/parent barrier checks.
2. Exclusively create, write, file-fsync and parent-sync fixed `p1-recovery-<recovery_id>.receipt-certificate.intent.json`; then exclusively create/write/fsync/parent-sync `p1-recovery-<recovery_id>.certificate.json`. The document is exactly `certificate steps`: the existing Core certificate plus all four exact validated Core step documents, each including its actual closed output. It retains the 64 KiB limit. Verify it against the durable Core binding and original journal with the production worker archive validator.
3. Record/fsync/parent-sync `.completion-journal.intent.json`, binding the accepted-intent digest and SHA-256 of the exact new journal envelope. Exclusively write/fsync/parent-sync fixed `p1-maintenance.journal.tmp`, replace `journal.json`, fsync receipts directory and verify/re-barrier the actual target. The journal changes only by adding the exact `maintenance` certificate binding to the already selected empty lane; lane/version/records stay unchanged. The fence remains pending, so worker startup still refuses.
4. Record/fsync/parent-sync `.completion-fence.intent.json`, then exclusively write/fsync/parent-sync fixed `p1-maintenance.complete.tmp`. Replace `p1-maintenance.json` only after the verified Core and receipt certificate and bound journal; fsync receipts directory and verify/re-barrier the actual complete fence. Fence deployment/recovery/plan/proof/old/new lane and certificate/epoch match exactly.
5. Only after those receipt barriers, append the exact accepted fifth `receipt-complete` step in SQLite BEGIN IMMEDIATE/write/COMMIT, validate the complete chain and sync the Core parent. Its closed output remains `recovery_id lane epoch core_output_digest`; its input/output chain extends the fourth step. It writes no jobs, attempts, events, dispatch/ACK/receipt/outcome, reservations/demand/charges, accounting/high-water, monitor, faults, resource epoch or resolution. Return `status='complete'` with the existing recovery/lane/epoch/certificate.

All final creation descriptors have exactly `revision recovery_id intent_digest sha256`, use fixed derived names, and bind canonical deterministic bytes. No caller-selected namespace is exposed. The original 2,048-entry/16 MiB population ceiling, bounded reads/deadlines, journal limits and 64 KiB certificate limit remain. Prospective finalization charges the additional certificate, journal/fence temporaries and descriptor allowance before creation; no archive is evicted.

### Crash states, replay and startup

| Durable state at interruption | Subsequent same-intent behavior |
| --- | --- |
| Core fourth step/certificate, no final creation attempt | Verify immutable chain and continue finalization. Receipt pending blocks worker/control hello. |
| Recorded creation attempt with missing/partial/wrong certificate or replacement output | Quarantine twice; never recreate, overwrite, remove temporary or mint another identity. |
| Exact receipt certificate, journal still unbound, no journal attempt | Verify/re-barrier certificate and continue the fixed journal/fence sequence. |
| Exact replaced bound journal, fence pending, no fence attempt | Verify/re-barrier journal; worker still refuses; continue fence sequence. |
| Exact complete fence after replacement, fifth SQL row absent/rolled back | Verify/re-barrier all exact outputs and record only the fifth row. No replacement is repeated. |
| Fifth step committed | Immutable completed no-op; no inventory/resource scan, old mutations, new events/resolutions/lane/epoch or overwrite. Later ordinary activity/registration does not replay this recovery. |

A crash immediately before replacement leaves an attempted incomplete output/temporary and quarantines. A crash immediately after replacement resumes its exact target, including a missing directory barrier, by re-barriering first. SQL failure/ambiguous COMMIT stops the session; a fresh invocation opens a new connection and inspects actual state. There is no live retry assuming rollback and no cross-volume atomicity claim.

The complete fence is the worker's startup authority; receipt and Core step certificates remain separate volume evidence. After a crash in the complete-fence/fifth-row window, worker may boot only with the matching bound journal/certificate, while supported maintenance resume still requires the accepted stopped-owner proof and both locks. Restarting consumers changes the supported freeze assumptions; no new proof is substituted. This ordering does not claim the worker can read the Core SQL row. Current startup eligibility is separate from immutable completed replay. An old completed replay validates retained evidence even after later ordinary Core activity or a later recovery; it never restores the old current journal/fence.

Production completion is used in both two-restart half-state negatives: Core-complete/receipt-pending refuses worker and both control paths; deliberately unsupported Core-pending/receipt-complete refuses Core/control. A real Journal after full completion supplies the supervisor's production hello operation. Old registration and omitted registration refuse. Fresh explicit registration permits inspect-only Dispatcher startup, with persistent/resource activation false, submit/replace/cancel refused, zero launches and hello-only runtime calls. The registration document in this local test is E1 synthetic provenance, not authenticated B2.3 actual-host evidence.

### Tests, failures and evidence

74 new cases use production helpers, real disposable SQLite/WAL/files/flock and process death; input builders construct accepted documents, not an alternate transition writer. The new fault matrix covers all three creation-intent create/write/file-fsync/parent/recorded boundaries; receipt certificate create/write/file-fsync/parent; bound-journal and complete-fence temporary create/write/fsync/parent/before-replace/after-replace/directory-sync; fifth SQL BEGIN/write/COMMIT/parent. Each resumes or quarantines twice after TTL as dictated by actual durable state, retaining all unrelated logical rows.

Four fork children call actual `os._exit(73)` before and after fence replacement and before and after fifth-step COMMIT. A fresh interpreter receives only owned root paths/scope/recovery ID/expected lane/epoch, reloads the protected accepted document and attempts resume TWICE after TTL. Pre-replace attempted output quarantines; exact replaced/committed outputs finish with one accepted recovery, five steps and one original resolution set/lane/epoch. No inherited intent/observation/snapshot RAM authorizes restart.

Actual os.fsync denials target the newly written receipt certificate and its parent, actual os.replace denials target each completion replacement, and SQLite authorizer denials target fifth-step INSERT and COMMIT. Exact durable outputs resume; attempted unreplaced outputs quarantine. Wrong/missing/malformed/mode/symlink/hardlink receipt evidence, wrong Core certificate, bundle/original archive/prior history, lane/epoch/proof, fence missing/malformed/mode and extra namespace refuse twice with unchanged evidence bytes/inodes. Two production completed histories retain predecessor bundles/rotation and final creation descriptors; independent damage to each refuses. Later real ordinary retry submission/allocation followed by two no-scan completed replays preserves the complete current logical state.

Existing interpreter only, no installs, bytecode/cache disabled. RED against baseline recovery source (saved/restored in this worktree with `finally`): `tests/service/test_recovery.py -k b22d_complete_production_writer -q`, **1 failed, 377 deselected in 3.06s**, exit 1, `/tmp/b22d-red.log`, missing `_complete_receipts`. Initial positive GREEN: **1 passed, 377 deselected in 4.50s**.

Development failures are not passing receipts: first expanded run **37 failed, 5 passed, 377 deselected in 22.97s** (`/tmp/b22d-dev.log`), second **59 failed, 5 passed, 377 deselected in 31.45s** (`dev2`). The new tests had reused an evidence-only zero-high-water input builder, so the existing real Core scan correctly refused `b2_scan_output_changed` before the intended finalization boundary. Replaced that setup with the coherent production Core fixture's preaccepted bounded high-water. Also corrected the new logical-dump comparison before it could validate fault results. No production scan/accounting validator was relaxed. Subsequent development: **64 passed, 377 deselected in 57.74s** (`dev3`); startup/later-activity and actual-denial subset **10 passed, 435 deselected in 10.97s** (`dev4`); retained predecessor/denial subset **12 passed, 439 deselected in 18.92s** (`dev5`). Final source adds up-front validation of every existing final creation output before any later write; frozen checks below supersede development receipts.

Final focused/full commands and results are recorded below. Source/tests stay frozen at `1ceec7726f799c80d586ac9ed5e5f835e8374b78` throughout; only this handoff changes.

E1 local synthetic proof documents/source/SQLite/filesystem/process evidence only. E1-mock is not actual-host proof; installed/platform skips are not qualification. No physical power-loss/ENOSPC/daemon/VM/controller, installed lifecycle, engine/audio/hardware or clean-host claim. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver. B2.3 trusted transport, B3, C1/C2, parent integration, independent review, final Astra and public/unattended acceptance remain pending. No subagents, Docker/network/models/audio/paid/E2E runner, push/PR/deploy, prune or user cleanup; unrelated research/worktrees preserved.

Frozen focused command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

Frozen full preservation command (sequential after focused, same unchanged source/tests):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

Focused result: **583 passed in 319.35 seconds, exit 0**, `/tmp/b22d-focused-final.log`. All 74 new receipt cases and predecessor recovery/startup/authority tests passed on frozen `1ceec7726f799c80d586ac9ed5e5f835e8374b78`. Full preservation starts sequentially afterward; it is not a redundant unchanged-candidate campaign.

Full preservation result: **735 passed, 18 skipped in 359.07 seconds, exit 0**, `/tmp/b22d-full-final.log`. Sequential after the 583-pass focused run, on the same unchanged source/tests `1ceec7726f799c80d586ac9ed5e5f835e8374b78`. The 18 skips remain unqualified. No source/test changes during or between frozen checks and no redundant broad campaign.

Final `git diff --check` passes. Actual baseline production/worker/tests match reviewed `3f411135` exactly. Complete baseline-to-candidate changes are only recovery.py, recovery tests and this appended handoff. Protected Core/persistence/initializer/Dispatcher/supervisor/worker/resource/admission/dispatch-store/reconciliation/host/contracts/dependencies/packaging/reviews/ledger comparison is empty. The report-only commit follows frozen source/tests; its exact candidate SHA accompanies handoff. Candidate is ready for fresh independent Sol-low review, not self-approved. Parent owns integration verification and final Astra scheduling; B2.3/B3/C1/C2/public activation remain pending.

## B2.2d round-1 F1/F2 corrections — awaiting fresh independent review

2026-10-16, implementation gpt-6.1-sol / low. Only worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-ec6f294d`, branch `t3/b22d-recovery-fixes`, was edited. Actual clean root baseline: `2bced6e` (root's NO-GO documentation, after `ce86feb` and `50ee0b0`). Inspected `git log --reverse 50ee0b0..fccb0391db4d45d52f56dfbdc33a63a2aa6cb6a0`: exactly frozen source/tests `1ceec7726f799c80d586ac9ed5e5f835e8374b78`, then report `fccb0391db4d45d52f56dfbdc33a63a2aa6cb6a0`. Cherry-picked ALL oldest first as `1a0f2fc` then `c6ffb06`. `git range-diff 50ee0b0..fccb0391 2bced6e..c6ffb06` reports both patches equal; direct source/test and report comparisons are empty. Root documentation/reviews are preserved. No other checkout was edited.

Read full P1B22d-r1-sol including exact reproducer, minimal fixes, ordering/process/startup probes and evidence limits; full P1B2-BOUNDARY-ASTRA architecture/records/resume/authority/B2.2; full preceding RECOVERY-P1B2 contracts, tranches and correction responses; orchestration protocol, P1B22c-r1/r2 and R1 fault-authority/B1 WAL/no-remint context. No applicable AGENTS.md found. Memory supplied only general exact-checkout/evidence-tier guidance; no historical acceptance is used as current evidence.

Frozen correction source/tests: **`b3e4fec940b5cfb8e1afe70bee7c76fccada9f96`**. Correction delta from imported `c6ffb06` is only `service/recovery.py`, `tests/service/test_recovery.py`, and this appended response. The complete fresh-root-to-candidate path set is the same three paths. No persistence or worker reader change was necessary. No Core/initializer/Dispatcher/supervisor/host/reconciliation/resources/contracts/dependencies/packaging/ledger/reviews edits. Versions, schemas, exact closed record keys, journal lanes/epochs/events/jobs/resources/ACK behavior and public/private apply interfaces remain unchanged. Every apply entrypoint remains unavailable; enrolled activation remains closed, without flags or autolaunch.

### F1 response

The certified Core branch now distinguishes the actual fifth SQL receipt from an unfinished receipt continuation. Before remaining receipt writes or fifth insertion, an unfinished `_complete_receipts` compares the entire current bounded retained registration table, in canonical row order, to the immutable accepted before-image. The existing full database validator bounds the population before this read. New/changed registration identities, boot/lane or container/start incarnations refuse with `b2_prefix_registrations_changed`; validated unchanged registrations must also satisfy the original accepted observation's complete consumer/death coverage. No new proof, observation, recovery identity or ordinary scan is substituted. Both locks remain held; free locks are not death evidence.

Regression stops the real production writer at `completion-fence-dirsync`, before the fifth row. A real Journal and production supervisor `operation('hello', {})` expose the completed fence/new lane; production `register` retains a new lane/boot, first with the same container set, then separately with a different worker container/full ID and started incarnation. Both subsequent stale-proof resumes refuse; fifth count remains zero, SQL logical state and every retained file byte (including SQLite) remain unchanged. Forbidden production file/SQL writer hooks additionally detect any attempted mutation before refusal. The unchanged stopped-owner control completes and replays successfully. These documents retain E1-mock provenance and are not actual-host proof. Existing fresh-registration/inspect-only Dispatcher and later ordinary-activity completed no-op tests remain intact: a genuine fifth-completed replay does not impose current registration eligibility.

### F2 response

Completed replay retains the old completed predecessor certificate/journal obligations and full production evidence checks for every fifth-completed history. A later unfinished history is validated according to its actual durable steps and recorded attempts, instead of requiring its future receipt certificate. The existing immutable owned-input validator now has an internal stage-aware mode: accepted document/installation/keys/bundle are retained; evidence requires its exact snapshot/manifest/original-journal creation records and outputs; journal requires the exact rotation descriptor; certified Core completion requires the exact Core certificate descriptor and certificate. Existing control archive validation still grounds all canonical source/history and snapshot rows. No table/history exclusion or special recovery-ID rule is added.

For the latest pending successor, verify its declared current journal and pending fence; exact attempted rotation or final receipt replacements are validated against their deterministic outputs without repeating a write. A recorded final creation attempt still requires its exact output, correct order and durable Core stage. Missing/changed evidence cannot be acknowledged away. No descriptor, certificate, lane or epoch is reminted. Completed predecessors and their production bundle/rotation/final creation evidence retain their existing checks.

A-completed/B-pending regression executes the production B writer separately through evidence, journal and certified Core-complete stages. A returns its original complete result twice at each stage with snapshot/inventory/resource scans and file/SQL writers forbidden; all SQL and file bytes stay unchanged. Independent damage to B's bundle, original journal archive, stage creation descriptor, current journal or fence must refuse twice without mutations. Two further positives cover B's exact post-journal-replace/pre-step and complete-fence/pre-fifth outputs. These are retained-stage validation, not permission to finish B or to perform ordinary resource scans.

### Reproduction and verification

All pytest commands use existing `/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, and `-p no:cacheprovider`; no installs. Only disposable synthetic roots are used.

First extracted the exact report's embedded Python reproducer to `/tmp/b22d-exact-reproducer.py` and executed it unchanged against the imported candidate:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /tmp/pab-0c-venv/bin/python /tmp/b22d-exact-reproducer.py
```

Exit 0, printing `restarted-before-fifth reproduced twice` and `later-pending reproduced twice`. This diagnostic asserts the incorrect behavior, not the desired correction.

Required-correctness RED, before production edits:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b22d_correction -q
```

**5 failed, 10 passed, 451 deselected in 28.73s**, exit 1, `/tmp/b22d-correction-red.log`. Both new registration variants reached the forbidden fifth-step SQL writer (AssertionError), proving continuation attempted a mutation before current-registration refusal. All three unmodified pending successor stages raised FileNotFoundError for B's future receipt certificate. The unchanged stopped control and existing-refusal damage cases passed; those damage passes alone do not establish stage-aware validation.

Initial GREEN, same 15 cases/command: **15 passed, 451 deselected in 28.72s**, exit 0, `/tmp/b22d-correction-green.log`. Expanded byte-preservation/current-journal/fence and retained predecessor/startup check:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k 'b22d_correction or b22d_prior_completed or b22d_real_journal' -q
```

**28 passed, 444 deselected in 53.43s**, exit 0, `/tmp/b22d-correction-green2.log`. Final exact attempted-output positives, `tests/service/test_recovery.py -k pending_exact_attempt -q`: **2 passed, 472 deselected in 5.06s**, exit 0, `/tmp/b22d-correction-attempts.log`. No development failures after the intended RED. The frozen source adds 23 regression cases; final focused/full results follow below. Only this response changes during those sequential checks.

Core certificate -> four-step receipt certificate -> bound journal -> durable complete fence -> fifth SQL receipt ordering remains unchanged. No cross-volume atomicity is claimed. Existing real-process-death/owned-state resume, both locks, exact canonical grounded histories/all prior archives, fixed names/0600/no-follow/single-link/bounds/file-and-parent barriers/no-remint quarantine, R1/B1 WAL/no-remint and B2.2c strict resolution/context checking are preserved by unchanged writers and the full preservation suite.

E1 local source/SQLite/filesystem/process and mocked host documents only; E1-mock is not actual-host authentication. All installed/platform skips remain unqualified. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver. No physical power-loss/ENOSPC/daemon/VM/controller, installed lifecycle, engine/audio/hardware or clean-host qualification. B2.3 trusted host transport, B3, C1/C2, parent integration verification, fresh independent Sol-low review, final Astra and public/unattended acceptance remain pending. No subagents, Docker/network/model/audio/paid/E2E runner, push/PR/deploy, prune or user cleanup; research and other worktrees preserved. Ready for fresh independent review after final receipts, **no self-GO**.

Frozen focused command on unchanged `b3e4fec940b5cfb8e1afe70bee7c76fccada9f96`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

**606 passed in 359.07 seconds, exit 0**, `/tmp/b22d-correction-focused-final.log`. Includes all 23 correction cases and the inherited receipt ordering/fault/process-death/startup/authority cases. Full service/release follows sequentially on the same unchanged source/tests; only this handoff changes.

Frozen full preservation command, sequential after focused on unchanged `b3e4fec940b5cfb8e1afe70bee7c76fccada9f96`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**758 passed, 18 skipped in 398.44 seconds, exit 0**, `/tmp/b22d-correction-full-final.log`. All 18 skips remain unqualified. This single full campaign followed the 606-pass focused suite sequentially on the same frozen source/tests. No source/test changes occurred during or between final checks; no redundant broad repeat was run.

Final `git diff --check` passes. Actual root baseline is `2bced6ec2cec67e9953f07cba1bea989af06f6b3`. Complete root-to-candidate changed paths and correction-only paths are exactly recovery.py, recovery tests and this appended handoff. Protected root reviews/ledger, Core/initializer/persistence/Dispatcher/supervisor/worker/host/reconciliation/resource/admission/dispatch stores/contracts/dependencies/packaging remain unchanged. Imported chain patch equivalence was verified by range-diff and direct comparisons. No source/test delta exists after frozen `b3e4fec940b5cfb8e1afe70bee7c76fccada9f96`; the final report-only commit follows it, and its exact candidate SHA accompanies the handoff. Ready for fresh independent review and parent integration verification; **no self-GO**, no B2.3/B3/C1/C2/final Astra or public/unattended acceptance.

## B2.2d round-2 F2/F3 correction — awaiting fresh independent review

2026-10-16, implementation gpt-6.1-sol / low. Only `/Users/ben/.txcode/worktrees/py-audio-box/t3-42ef127a`, branch `t3/b22d-recovery-resume-fixes`, was edited. Actual clean fresh-root baseline `3692f17` lacked the candidate implementation. Inspected the complete `2bced6e..6d84699a9fa79ba1ef601e75bf1c06e0695bc7e9` log and imported ALL four commits oldest first: `1a0f2fc` → `c6ffb06` → `b3e4fec` → `6d84699`, locally `da0561a` → `0954f61` → `b2270f8` → `899c25c`. Imports were clean; root documentation/reviews are preserved. No other checkout was edited.

Read the full R2 report including exact F2/F3 reproducers/minimal fixes/prior F1 closure; full R1 report; full P1B2-BOUNDARY-ASTRA architecture, records, resume and B2.2; this full handoff including original contracts, tranches and every correction; orchestration protocol, P1B22c-r1/r2, P1B22b-r1 and R1/B1 authority/WAL/no-remint context. No applicable ancestor/scoped AGENTS.md found. Lightweight memory registry search returned no relevant B2 guidance; no historical acceptance is current evidence.

Frozen correction source/tests: **`8dbfdca1da5b79ce1714984594a06e86bce3ead4`**. Correction-only delta from imported `899c25c` is recovery.py, recovery tests and this appended response. Complete fresh-root-to-candidate path set is the same three paths. No persistence/Journal change was needed; no Core/initializer/Dispatcher/supervisor/host/reconciliation/resources/contracts/dependencies/packaging/ledger/review changes. No schema, record-key, version, lane/epoch, job/resource/event/ACK or activation-interface delta.

### F1 preserved; F2 response

R1 F1 closure remains unchanged: actual fifth completion selects immutable replay; unfinished receipt continuation compares the entire current bounded registration population with the accepted before-image and checks original all-owner death coverage BEFORE any remaining receipt write/fifth insertion. Both locks span the operation. No substituted proof or current-startup requirement is imposed on a genuinely completed no-op. Existing production hello/register stale-proof and later ordinary-activity tests remain.

The private database/archive validator now accepts an explicit retained-receipts root from the owning locked continuation. Only the latest unfinished history can reserve its exact future Core certificate when BOTH its creation attempt and output are absent. It still validates and grounds its snapshot/manifest, complete canonical before-history and SQL chain; every completed predecessor retains its actual certificate requirement. This applies to the initial validation and subsequent immutable replay/final-return validations, avoiding a later repeat of the same false refusal. Default startup/database readers retain their actual-certificate requirement. The existing requested-current `_uncertified` prospective transaction seam is unchanged.

The retained-stage helper independently recognizes committed Core with no creation attempt/output as a legitimate prefix. Once an attempt exists, exact descriptor and output are mandatory; missing/partial/wrong output quarantines. An output without its descriptor also refuses. It never creates B's certificate while replaying completed A and never exempts a whole history/table. Existing exact source/archives/bundle/original-journal/rotation and prior completed-history obligations remain.

### F3 response: closed finite prerequisite matrix

Retained attempts are checked against the following fixed production order, in addition to exact bytes and the existing contiguous SQL chain. No general workflow framework or requirement for future receipts/certificates is introduced.

| Retained durable item | Required predecessor evidence | Own missing SQL receipt permitted? |
| --- | --- | --- |
| Accepted Core intent / original bundle | Existing protected accepted document, marker/fence and exact grounded staging bundle/descriptor | Pending receipt may still be absent |
| Snapshot creation attempt/output | Durable pending SQL receipt | Evidence receipt may be absent |
| Manifest creation attempt/output | Pending receipt and exact snapshot attempt/output | Evidence receipt may be absent |
| Original-journal creation attempt/output | Pending receipt and exact snapshot/manifest attempts/outputs | Evidence receipt may be absent |
| Evidence SQL receipt | All three exact creation attempts/outputs | Not applicable |
| Rotation attempt/exact replacement | Durable evidence SQL receipt and its predecessor chain | Journal receipt may be absent |
| Core SQL completion | Durable journal SQL receipt and its predecessor chain (existing SQL validator) | Core certificate attempt/output may still be absent |
| Core certificate attempt/output | Durable Core fourth SQL receipt and exact rotation/evidence inputs | No future receipt certificate required |
| Receipt-certificate attempt/output | Durable Core receipt and exact Core certificate attempt/output | Fifth receipt may be absent |
| Completion-journal attempt/output | Exact receipt-certificate attempt/output | Fifth receipt may be absent |
| Completion-fence attempt/output | Exact completion-journal attempt/output | Fifth receipt may be absent |
| Fifth SQL receipt | Existing complete certificate/bound-journal/fence production sequence | Completed immutable replay |

The exact F3 one-row deletion now refuses before any scan/write: after actual `journal-replace` and before journal SQL receipt, deleting only B's evidence row cannot be mistaken for interrupted unrecorded journal output. Earlier evidence-order checks also reject a missing pending receipt or missing preceding creation descriptor. Final attempts retain their ordered chain and additionally require the Core certificate's own persisted creation attempt/output. Existing SQL validation already rejects missing journal/Core predecessor rows when later SQL steps exist. Missing predecessors are corruption; a valid output missing its own receipt remains accepted.

### RED, GREEN and actual interruption evidence

Existing interpreter only; no installs. Every pytest command uses `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider`; disposable owned synthetic roots only.

RED on imported candidate before production edits:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b22d_r2 -q
```

**4 failed, 1 passed, 474 deselected in 13.75s**, exit 1, `/tmp/b22d-r2-red.log`. Both actual `core-complete-commit` and `core-complete-parent` faults fired and falsely required B's absent certificate; exact `journal-replace`/single evidence-row deletion and snapshot-parent/single pending-row deletion falsely accepted corruption. Unchanged post-journal-replace control passed. Every hook explicitly asserts it fired. Initial same-five GREEN: **5 passed, 474 deselected in 13.98s**, exit 0, `/tmp/b22d-r2-green.log`.

Expanded same-command matrix: **25 passed, 474 deselected in 50.86s**, exit 0, `/tmp/b22d-r2-matrix.log`. Positives cover Core intent before pending receipt, pending COMMIT, exact snapshot/manifest/original-journal outputs before evidence receipt, evidence COMMIT, exact replaced journal before its SQL receipt, journal COMMIT, Core COMMIT and parent before certificate creation, certified Core, receipt certificate, bound-journal replace and complete-fence replace before fifth. Negatives delete exactly one SQL row/preceding attempt or remove/partially replace/wrongly replace attempted Core certificate output; the recorded-but-not-created Core certificate boundary quarantines. No development failures followed the intended RED.

Actual process test command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b22d_r2_completed_replay_crash -q
```

**2 passed, 499 deselected in 9.46s**, exit 0, `/tmp/b22d-r2-process.log`. Production B `_pending_through_core` exits via actual `os._exit(73)` at COMMIT and parent separately; exit status proves each hook fired. Parent discards A/B intent/observation and launches a fresh interpreter with only owned roots/scope/A recovery ID. It reloads actual accepted documents, confirms no B certificate attempt exists, then replays A twice after TTL. No inherited snapshot/intent RAM supplies authority. Logical capture occurs before physical-byte baseline because opening crash-left SQLite may recover/checkpoint WAL; byte preservation is claimed only after that explicit capture, not across initial crash-state opening.

All 27 new cases replay A twice with snapshot/Reconciliation collection/recovery scan and production file/SQL writers forbidden; A returns its original result in valid windows or refuses corrupt retained inputs. Entire logical SQL state and every retained regular-file byte remain equal to their baselines on each replay, including all B state. Existing attempted-missing negatives, certified-B positives, F1 stale-proof cases and prior completed-history corruption tests remain in final focused/full verification. The expanded matrix and process subset briefly overlapped on separate disposable local roots; final focused/full suites are sequential on frozen source/tests.

Core certificate → exact four-step receipt certificate → bound journal → durable complete fence → fifth SQL receipt ordering remains unchanged. Both locks, original all-owner proof, closed bounded records, exact 0600/no-follow/owner/single-link reads, file/parent barriers, creation-before-attempt/no-remint quarantine, all prior archives/staging/rotation evidence and B2.2c strict context/resolution and R1/B1 WAL/uncertainty behavior remain. All apply is unconditionally unavailable; enrolled activation stays closed without flags/autolaunch.

E1 local source/SQLite/filesystem/process and mocked host documents only; E1-mock is not actual-host proof. Installed/platform skips remain unqualified. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver. No physical power-loss/ENOSPC/daemon/VM/controller durability, installed lifecycle, engine/audio/hardware or clean-host qualification. B2.3 host authentication is later work, not a prerequisite added to this correction. B2.3/B3/C1/C2, parent integration, fresh independent review, final Astra and public/unattended acceptance remain pending. No subagents, Docker/network/models/audio/paid/E2E runner, installs, push/PR/deploy, prune or user cleanup; research/other worktrees preserved. **No self-GO.**

Final frozen focused/full receipts follow; only this report is edited during verification.

Frozen focused command on unchanged `8dbfdca1da5b79ce1714984594a06e86bce3ead4`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

**633 passed in 418.05 seconds, exit 0**, `/tmp/b22d-r2-focused-final.log`. Includes all 27 new R2 cases plus inherited F1, attempted-output quarantine, prior-history, ordering/process/startup/authority cases. Full service/release follows sequentially on this same frozen source/tests; only this handoff changes.

Frozen full preservation command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**785 passed, 18 skipped in 459.32 seconds, exit 0**, `/tmp/b22d-r2-full-final.log`. Sequential after the 633-pass focused run on unchanged frozen `8dbfdca1da5b79ce1714984594a06e86bce3ead4`. All 18 skips remain unqualified. No source/tests changed during or between the frozen checks, and no redundant broad campaign was run.

Final `git diff --check` passes. Actual fresh-root baseline is `3692f17428b5ad58d0fdaddb0eba9e541a4347e2`. `git range-diff 2bced6e..6d84699 3692f17..899c25c` reports all four imported patches equal. Complete baseline-to-candidate and correction-only path sets are exactly recovery.py, recovery tests and this handoff. Protected-path comparison excluding these three paths is empty; root reviews/ledger and all other source/worker/contracts/dependencies/packaging remain preserved. Frozen-to-final delta is report-only. Final report-only commit follows the frozen implementation; its exact candidate SHA accompanies the handoff. Fresh independent review of that exact candidate and parent integration verification are required. No self-GO or B2.3/B3/C1/C2/final Astra/public/unattended acceptance is asserted.

## B2.2d round-3 F2 receipt-only successor correction — awaiting independent review

2026-10-17, gpt-6.1-sol / low implementation. Only `/Users/ben/.txcode/worktrees/py-audio-box/t3-108eec6e`, branch `t3/receipt-only-successor`, changed. Actual clean root baseline `657ff8f` lacked the implementation. Inspected `3692f17..3eddd2662ebbbb08a728fce6e23d22e3784e18af`; imported ALL six commits oldest first: `da0561a`, `0954f61`, `b2270f8`, `899c25c`, `8dbfdca`, `3eddd26`, locally `544c19a`, `3d9481a`, `7d0df05`, `53e6f2e`, `0f9f674`, `dacf8fd`. Range-diff reports all six patches equal. Root docs/reviews are preserved; no other checkout was edited.

Read the FULL R3 exact reproducer/minimal fix/closures, R1/R2 reviews, FULL boundary architecture/records/resume/B2.2 and FULL recovery handoff including original contracts, tranches and corrections, orchestration protocol and retained R1 fault-authority/B1 WAL/no-remint context. No applicable AGENTS.md found. The lightweight memory registry search returned no relevant guidance; no historical acceptance is current evidence.

Correction-only and complete fresh-root delta remain exactly recovery.py, recovery tests and this appended response. No persistence/Journal change was needed. No Core/initializer/Dispatcher/supervisor/host/reconciliation/resources/contracts/dependencies/packaging/ledger/review edit. No versions, schemas, record keys, lanes/epochs/jobs/resources/events/resolutions/ACK or public interfaces change. All apply remains unconditionally unavailable; enrolled activation remains closed without flags/autolaunch.

### F2 response and entire pre-Core finite sequence

Before SQL-derived receipt namespace validation, immutable replay now discovers at most ONE new receipt-only accepted document. Discovery has the existing finite file/byte/deadline bounds and protected no-follow/owner/regular/single-link reads; matching a filename selects a candidate, never exempts it. Multiple candidates explicitly refuse. The full closed canonical accepted record must bind the EXISTING SQL plan, exact latest fifth-completed predecessor certificate, all-owner accepted proof/scope, installation/volume/key identities, complete canonical SQL before-image and schema. Only its own plan row is self-excluded; all prior plans/registrations/steps/resolutions remain compared. Embedded histories consume the existing population bound. No current media/inventory/resource scan or fresh proof is used.

The stage-owned marker, accepted document, fence, journal, bundle descriptor and bundle require exact 0600 protected evidence. Only the accepted filename and actually reached exact bundle descriptor/output names can be added to the namespace allowlist. The old completed fence is legitimate immediately after full accepted-document persistence; the exact pending successor fence is legitimate after replacement. Bundle creation requires that pending fence. Its descriptor precedes its output, binds the complete accepted record/source/expected snapshot, and requires exact actual bundle bytes. The existing grounded archive oracle checks a private copy against all canonical source rows and the actual plan. No future SQL accepted row, snapshot/manifest archive or certificate is required; premature control archives or receipt outputs refuse. No repair/remint/finish-B operation occurs.

| Actual pre-Core boundary | A replay result / retained obligation |
| --- | --- |
| Required marker, before accepted-file attempt | A remains immutable; successor acceptance has not occurred. A completion already owns the exact marker, so B does not recreate it. |
| Accepted create, before complete write | Empty/partial accepted document quarantines. |
| Accepted complete write / file fsync / parent fsync | Validate full accepted record; previous complete fence and original journal remain exact; no bundle is demanded. |
| Pending-fence temporary create / complete write / fsync / parent | Quarantine the retained replacement temporary, including complete bytes, consistently with existing B resume; no automatic replacement/cleanup. |
| Pending-fence replace + parent barrier | Exact successor pending fence, accepted record and original journal; future bundle not demanded. |
| Bundle-descriptor create / write / fsync / parent / recorded | Incomplete descriptor or recorded missing output quarantines; descriptor alone is not success. |
| Bundle create, before complete output | Empty/partial attempted bundle quarantines. |
| Bundle complete write / fsync / parent | Exact descriptor + grounded complete bundle + pending fence accepted; no future Core SQL row or archive demanded. |
| Core accepted SQL and later stages | Existing retained SQL/creation prerequisite validator remains in force, including legitimate own-output/pre-own-receipt windows and committed-Core/no-certificate-attempt. |

After validation, a genuinely fifth-completed A validates its original immutable Core/receipt/bundle/rotation/final descriptors and returns its existing result directly. It does not re-enter the finalization writer or infer the live fence from the latest SQL history when a validated receipt-only successor owns it. The requested A's Core-certificate creation descriptor remains explicitly required. Unfinished A still uses the existing whole-registration/original-death-proof check BEFORE remaining writes (F1 preserved). Existing cross-stage SQL/creation prerequisites, missing predecessor refusals and attempted-missing/partial/wrong output quarantine remain (F3 preserved).

Core certificate → exact four-step receipt certificate → bound journal → complete fence → fifth SQL receipt remains unchanged. Both locks span each invocation. All prior grounded archives/staging/rotation/closed history, strict canonical job/context/B2.2c joins and R1/B1 WAL/inherited-uncertainty/no-remint semantics remain.

### Actual RED/GREEN and process evidence

Every pytest command uses the existing interpreter, no installs:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b22d_r3 -q
```

Before production correction, exact R3 correctness regressions at accepted-parent, fence-replace-parent and bundle-parent were **3 failed, 501 deselected in 9.59s**, exit 1, `/tmp/b22d-r3-red.log`. Each asserted its production hook fired; each failed with the reported namespace refusal. Same three GREEN: **3 passed, 501 deselected in 11.78s**, exit 0, `/tmp/b22d-r3-green.log`.

Expanded entire pre-Core boundary matrix, same command on evolving tests: **18 passed, 501 deselected in 38.22s**, exit 0, `/tmp/b22d-r3-matrix.log`. Every case replays A twice with snapshot/Reconciliation collection/resource scan/file and SQL writers forbidden, checking complete logical SQL and every retained file byte. All seven complete-output windows restore hooks and resume B after TTL with its same identity and no supplied snapshot bytes, then replay A successfully. Recorded incomplete outputs refuse twice without mutation.

Tamper/process command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k 'b22d_r3_receipt_only_tamper or b22d_r3_receipt_only_crash' -q
```

**33 passed, 519 deselected in 56.40s**, exit 0, `/tmp/b22d-r3-tamper-process.log`. Separate negatives cover accepted presence/canonical/schema/mode/link/filename, plan, prior binding, scope/proof, descriptor, bundle, fence/marker/current journal, unknown extras, premature outputs/attempts, old Core descriptor, later plan and multiple candidates. No new development failure followed the intended RED. These are development receipts; the population bound and marker-before-acceptance test were finalized before the frozen campaign below.

Actual child `os._exit(73)` at bundle-parent is followed by a FRESH interpreter receiving ONLY owned roots/scope/A ID. It reloads A's accepted document, replays A twice after TTL with all scans/writers forbidden and preserves logical/file state. After restoring hooks it discovers B from the owned accepted receipt, resumes B after TTL with no supplied snapshot/caller documents, and replays A. Parent discards A/B intent/observation and fixture snapshot variables; no inherited RAM authorizes the fresh interpreter. Logical capture precedes physical-byte baseline because initial opening of crash-left SQLite may recover/checkpoint WAL; no preservation claim is made across that first open.

Final combined targeted and frozen focused/full receipts follow. Only this report changes during the final frozen checks. No self-GO; parent owns fresh independent review of the exact resulting candidate and integration verification.

E1 local source/SQLite/filesystem/process and mocked host inputs only; E1-mock is NOT actual-host proof. Skips are not qualification. Physical power-loss/ENOSPC/daemon/VM/controller durability, installed lifecycle, engine/audio/hardware and clean-host operation remain unqualified. A1 whole-Docker-inventory drift remains unexplained/unqualified with no inferred cause or waiver. B2.3 host authentication is later work, not a prerequisite imposed here. B2.3/B3/C1/C2, parent integration, fresh independent review, final Astra and public/unattended acceptance remain pending. No subagents, Docker/network/model/audio/paid/E2E runner, installs, push/PR/deploy, prune or user cleanup; research/other worktrees preserved.

Combined final targeted command, after all source/test changes:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k 'b22d_r3 or b22d_r2 or b22d_correction or b22d_prior_completed' -q
```

**108 passed, 445 deselected in 191.04s**, exit 0, `/tmp/b22d-r3-targeted.log`. Includes all 52 new R3 cases, prior F1 registration controls/refusals, F2 SQL pending/journal/Core uncertified windows and F3 missing-predecessor/attempted-output quarantine. Final frozen source/tests commit: **`8d6ccd5308aa6eee5413f203ab9c3296260d3dd0`**. No source/tests change afterward; subsequent commit is report-only. Development subsets used disposable independent roots; final focused/full campaigns run sequentially on this frozen commit.

Frozen focused command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

Frozen full preservation command, sequential afterward on unchanged source/tests:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

Frozen focused result: **685 passed in 506.08s (8:26)**, exit 0, `/tmp/b22d-r3-focused-final.log`, on unchanged `8d6ccd5308aa6eee5413f203ab9c3296260d3dd0`. All 52 new R3 cases and inherited receipt/process/ordering/startup/F1/F3/authority cases pass. Full service/release starts sequentially afterward on the same source/tests; only this report changes.

Frozen full preservation result: **837 passed, 18 skipped in 551.14s (9:11)**, exit 0, `/tmp/b22d-r3-full-final.log`. Sequential after the 685-pass focused suite on the same unchanged `8d6ccd5308aa6eee5413f203ab9c3296260d3dd0`; all 18 skips remain unqualified. No source/test changes during or between final suites. No redundant broad campaign and no observed final-suite failure.

Final `git diff --check` passes. Source/test-to-frozen comparison is empty; frozen-to-final is report-only. Complete actual-root `657ff8f` to candidate path set is exactly service/recovery.py, tests/service/test_recovery.py and this handoff. Comparison excluding those three paths is empty: root reviews/ledger and every other source/worker/contracts/dependency/packaging path are preserved. The six imported patches have equal range-diff identities. Final report-only commit follows `8d6ccd5`; its exact SHA accompanies the handoff. Fresh independent review of that exact final candidate and parent integration verification remain required. No self-GO; B2.3/B3/C1/C2/final Astra/public/unattended acceptance remain pending.

## B2.2d round-4 F4 retained Core creation descriptor correction — awaiting independent review

2026-10-17, gpt-6.1-sol / low implementation. Only worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-dec21b4c`, branch `t3/fix-b22d-f4-history-evidence`, is edited. Actual clean fresh-root baseline `2efa63d` lacked the implementation. Inspected the complete `657ff8f..e3d58452de53de114feb71b4bbc7c7c5d3821305` log and imported ALL eight commits oldest first: `544c19a`, `3d9481a`, `7d0df05`, `53e6f2e`, `0f9f674`, `dacf8fd`, `8d6ccd5`, `e3d5845`, locally `1b34b57`, `2a867c6`, `fe1e9c5`, `588cf13`, `2c7de87`, `4c1fae9`, `16fc113`, `f56854b`. Range-diff reports every imported patch equal. Root docs/reviews are preserved; no other checkout was edited.

Read full R4 including its exact reproducer/minimal fix and prior closures, R1/R2/R3, full boundary architecture/records/resume/B2.2, and full RECOVERY-P1B2 contracts/tranches/correction responses; orchestration protocol and R1 fault-authority/B1 round-2 WAL/no-remint context. No applicable ancestor/scoped AGENTS.md found. Memory supplied only general exact-checkout/evidence-tier guidance; no historical acceptance is current evidence.

### F4 response and finite completed creation obligations

The common `_verify_finished_receipt_evidence` now derives the actual canonical Core certificate from each validated history's fourth step, requires the actual derived control certificate bytes, and requires its exact canonical `.core-certificate.intent.json` using the same accepted-intent digest and actual certificate SHA-256. The protected exact reader enforces 0600, owner/regular/single-link/no-follow/bounded bytes. This validator already runs for EVERY fifth-completed history, including a completed successor when requesting an earlier recovery. The old requested-only descriptor guard is removed. There is no special recovery-ID condition, wildcard allowance or skipped history.

Audited the entire finite production creation sequence against the shared completed-history calls:

| Completed creation obligation | Common retained validation |
| --- | --- |
| Accepted full intent / required marker / installation keys and volume binding | `_completed_prefix_receipts` exact owned inputs |
| Original snapshot bundle descriptor and actual grounded bundle | `_completed_prefix_receipts`, plus existing database/archive grounding |
| Snapshot, manifest, original-journal descriptors and exact outputs | `_completed_prefix_receipts` for all three; database validates actual control archive/source |
| Rotation descriptor | `_completed_prefix_receipts` canonical original/new-journal binding |
| Core certificate descriptor and actual Core certificate | `_verify_finished_receipt_evidence`, added here |
| Receipt certificate descriptor and actual four-step receipt certificate | `_verify_finished_receipt_evidence` |
| Completion-journal and completion-fence descriptors | `_verify_finished_receipt_evidence`, canonical historical outputs; current outputs can legitimately belong to later activity/history |
| Fifth SQL receipt and prior SQL chain/resolution sets | Existing bounded database/chain validator |

The latest unfinished Core-COMMIT/no-attempt exception and stage-aware SQL/receipt-only pending validators are unchanged. Missing/partial/wrong attempted outputs still quarantine; no future output is demanded. F1 whole-registration/original stopped-owner coverage before unfinished writes remains; F2 valid pending SQL/receipt-only prefixes and F3 preceding SQL/creation prerequisites remain. No current-startup eligibility restriction, scan, repair/remint, output rewrite or broad redesign is added.

The actual production delta is four added validation lines and deletion of the requested-only check. Core certificate → four-step receipt certificate → bound journal → complete fence → fifth SQL receipt ordering and both locks are unchanged. Closed bounds/records, original proof, file/parent barriers, no-remint, immutable completed replay, ordinary-activity independence, strict context/resolution and R1 inherited uncertainty/B1 private WAL preservation remain. Public/programmatic/CLI apply remains unconditionally unavailable and enrolled activation closed without flags/autolaunch. No B2.3 host authentication is needed for this correction.

### Actual reproduction and verification

Existing interpreter only; no installs, bytecode/cache disabled, disposable local roots. Extracted the exact R4 embedded Python probe in memory and ran it unchanged via `/tmp/pab-0c-venv/bin/python -c` with `PYTHONDONTWRITEBYTECODE=1`: exit 0; missing/wrong/mode each printed `incorrectly accepted twice`, `/tmp/b22d-r4-exact-red.log`. This is defect reproduction, not correctness evidence. After correction the identical setup with only the acceptance assertion replaced by mandatory refusal exits 0, each variant `correctly refused twice`, `/tmp/b22d-r4-exact-green.log`.

Added eight production two-history regression cases: missing/wrong/mode damage to ONLY the other completed history's Core descriptor, in BOTH predecessor and successor directions, plus unchanged controls in both directions. Actual Core certificate, receipt certificate and all five SQL steps are explicitly verified intact before replay. Two after-TTL replays forbid snapshot, Reconciliation collection, resource scan and production file/SQL writers. Each refusal/control preserves all logical SQL, every retained regular-file byte and file modes. No fixture acceptance digest is rewritten.

Required-correctness RED before production edits:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b22d_r4 -q
```

**6 failed, 2 passed, 553 deselected in 22.48s**, exit 1, `/tmp/b22d-r4-red.log`. All six negatives reached `DID NOT RAISE`; both unchanged controls passed. No setup failure is counted as defect evidence.

Combined meaningful GREEN command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k 'b22d_r4 or b22d_r3 or b22d_r2 or b22d_correction or b22d_prior_completed' -q
```

Final targeted and frozen focused/full results follow. Source/tests remain unchanged during final verification; only this report changes. Complete imported-plus-correction and correction-only path sets are recovery.py, recovery tests and this report. No persistence/Journal changes are needed; no Core/init/Dispatcher/supervisor/host/reconciliation/resource/contracts/dependencies/packaging/ledger/reviews changes.

E1 local source/SQLite/filesystem/process and mocked host inputs only. E1-mock is not actual-host proof. Skips are not qualification. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver. Physical power-loss/ENOSPC/daemon/VM/controller durability, installed lifecycle, engine/audio/hardware and clean-host operation remain unqualified. B2.3/B3/C1/C2, parent integration, fresh independent review, final Astra and public/unattended acceptance remain pending. No subagents, Docker/network/model/audio/paid/E2E runner, installs, push/PR/deploy, prune or user cleanup; research/other worktrees preserved. No self-GO.

Combined targeted GREEN: **116 passed, 445 deselected in 213.80s**, exit 0, `/tmp/b22d-r4-targeted-green.log`. Includes all eight F4 cases and all inherited F1/F2/F3 targeted controls/refusals, receipt-only and SQL pending/no-attempt positives, attempted-output quarantine and retained prerequisites. No development failures occurred after the intended RED. Final frozen source/test commit: **`0d1a89440d4fa17764d484c8cf31b1b13ea0453a`**. Only this handoff changes afterward.

Frozen focused command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

Frozen full preservation command, sequential after focused on the same unchanged source/tests:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

Frozen focused result: **693 passed in 536.57s (8:56)**, exit 0, `/tmp/b22d-r4-focused-final.log`, on unchanged `0d1a89440d4fa17764d484c8cf31b1b13ea0453a`. Includes all eight new F4 cases plus inherited receipt/process/ordering/startup/F1/F2/F3/authority cases. Full service/release starts sequentially afterward on the same source/tests; only this report changes.

Frozen full preservation result: **845 passed, 18 skipped in 571.85s (9:31)**, exit 0, `/tmp/b22d-r4-full-final.log`. Sequential after the 693-pass focused suite on the same unchanged `0d1a89440d4fa17764d484c8cf31b1b13ea0453a`; all 18 skips remain unqualified. No source/test changes during or between final suites, no final-suite failure and no redundant broad campaign.

Final `git diff --check` passes. Complete actual-root `2efa63d` to candidate path set is exactly service/recovery.py, tests/service/test_recovery.py and this handoff. Protected-path comparison excluding those three paths is empty; root docs/reviews/ledger and every other source/worker/contracts/dependency/packaging path are preserved. All eight imported patches are range-diff equal. Source/test-to-frozen comparison is empty; frozen-to-final is report-only. The report-only commit follows frozen `0d1a894`; its exact candidate SHA accompanies the handoff. Fresh independent review of that exact candidate and parent integration verification remain required. No self-GO; B2.3/B3/C1/C2/final Astra/public/unattended acceptance remain pending.

## B2.3a — closed host authority producer/verifier (awaiting independent review)

2026-10-17, implementation `gpt-6.1-sol`, low. Only worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-c1fea5bd`, branch `t3/closed-host-authority-verifier`, is edited. Actual clean baseline **`38b622142844b7d5d9cc011295eba5a1420bae96`** includes integrated `11b3184`, the kernel gate and this assignment. `git diff 5bfaaa65 38b6221 -- service worker scripts tests` is empty: baseline production/tests match the exact independently reviewed private kernel. R5's 693 focused / 845 passed, 18 skipped and root's 8 F4 in 22.37s remain attributed predecessor evidence. Astra's gate is SOURCE ONLY, bounded private architecture, with no actual-host qualification.

Read full P1B2-KERNEL-ASTRA, original P1B2-BOUNDARY-ASTRA, PRODUCT, implementation start/protocol, the full preceding frozen RECOVERY-P1B2 contracts/corrections, P1B22d-r5 and latest ledger. No applicable AGENTS.md was found. A lightweight memory registry search supplied no B2-specific authority. No subagents/reviewer were launched: parent owns independent Sol-low review, integration verification and final Astra. This is implementation evidence, **not self-GO**.

Initial source/tests: **`4390c0354162b79ae545ec71357ccbb3ea4d5d27`**, superseded by the clock-boundary correction below. Exactly three source/test paths change there: `scripts/p1-operator.py`, `service/recovery.py`, `tests/service/test_recovery.py`. This appended handoff is the fourth and only later path. Recovery's entire prior 1,929-line kernel is unchanged; the host verifier is appended separately. No preparation, apply wiring, kernel mutation, packaging, dependencies, schemas/contracts, ledger, reviews or other paths change. Public 1.3.0, Core 1, resource protocol 2, private P1 revision 1, new-enrolled-synthetic-only scope and activation closure remain. **Every apply remains unconditionally `P1B2_apply_unavailable`.** Host helpers never call a private continuation. The new tests forbid all four private kernel entrypoints while exercising the host producer/reader.

### Host interfaces and finite sequence

The host CLI retains `register`, B1 `inspect`/unavailable `plan`, and unavailable `apply`, adding separate `preflight` and explicitly selected `freeze` commands. Their closed outer request is `{roots, scope, freeze}`; `freeze` has exactly `{deployment_id, scope, registration_digest, purpose, original_invocation}`. There is no caller-selected container list, subprocess argv, executable, proof path, force option, environmental authorization or E1 authorization option. `purpose` is `{kind, plan_id, plan_digest}`: explicit `pre-plan` with both plan fields null, or `initial`/`resume` with exact 32-hex plan ID / 64-hex digest. These fields bind content; authoritative plan/intent preparation is B2.3b, not supplied by this lot.

`host_state(roots, scope)` reads installation and ALL retained registrations from a bounded stable private WAL copy while live services may hold BOTH locks. It never opens original SQLite with SQLite, takes over a live owner's lock, runs reconciliation or creates a healthy epoch. Source instability refuses. Protected installation/marker and volume-role inode identities are checked; registration IDs/digests/closed records are checked before use. The latest retained lane and bounded recorded Core epoch are advisory proof bindings, not a kernel readiness decision. A pending SQL or receipt fence requires an explicit original-capsule resume. Host coverage does not require a healthy/readable current journal. B2.3b/c still own the full locked source, affected-set, intent and accepted-history verification; this inventory is not an alternative source projection.

`host_preflight(observer, state, request)` is read-only. `host_registration_union` unions every retained registration and rejects duplicates, conflicting starts/images for one full ID and ambiguous worker boot/lane reuse. `host_inventory` enumerates `docker --context <fixed> container ls --all --quiet --no-trunc`, inspects EVERY full ID, and includes consumers of ANY exact deployment volume, running or stopped. It also detects bind/path/inode aliases to that storage. Missing registered consumers, unrelated consumers, incomplete/changed enumeration, extra mounts and volume aliases refuse before stop. Driver `local` with no volume options, distinct actual mountpoint device/inode pairs, exact names/driver/creation/source, fixed role destinations and exact mount modes are required. Daemon ID and Linux type are checked through the explicitly selected local context at both ends of each enumeration.

Registration now brackets the existing authenticated `SupervisorRuntime(...).call('hello', {})` with complete host inventory and repeated start/PID/process observations. It accounts for retained stopped owners as well as the selected live owners, and requires exactly one live worker associated with this IPC. The bounded local Linux `/proc/<validated State.Pid>/stat`, PID namespace, cgroup-v2 membership and `/proc/<pid>/root/ipc` device/inode establish the process/execution-tree/IPC association. Root process identity is repeated; every bounded enumerated cgroup task must remain in the private PID namespace and exact container cgroup tree. Exact supported cgroups are `/docker/<full ID>` and `/system.slice/docker-<full ID>.scope`; unknown/delegated/shared namespaces, added capabilities, privileged containers and host cgroup namespace refuse. No cgroup-parent guess, label, hash, caller UID assertion or self-reported hello hash substitutes for these OS observations.

`register_host` writes a protected exclusive registration provenance document, then invokes the existing B1 registration writer. A crash/duplicate/partial file is retained and refuses rather than overwritten. Its schema is `{revision, kind, registration_digest, ipc_identity, hello, owners}`; kind is `local-linux-proc-v1`, hello exactly `{lane, boot}`, each owner exactly `{id, image, started, pid, start_ticks, pid_namespace, cgroup}`. This is a host transport record outside Core, not a change to the frozen registration/kernel schema.

`freeze_host` performs preflight and repeats full topology before any stop. It stops/waits each control writer BEFORE stopping any worker, using only `container stop --time 2 <full ID>` and `container wait <full ID>`. Successful wait COMMAND execution is mandatory for every retained owner; container exit code 137 is deliberately a positive. Then all owners must be non-running/non-restarting/PID 0, restart `no`, unchanged full ID/image/start, with a meaningful finish strictly after start (nanosecond order retained). Already-stopped owners must retain their exact finish. Full all-consumer/daemon/volume inspection repeats, retained registration authority is re-read, then full host inspection repeats immediately before capsule publication. Failure retains containers and evidence; there is no remove/prune/recreate/restart/start or cleanup in `finally`.

The fixed subprocess layer uses `shell=False`, DEVNULL stdin, private temporary output files, a five-second per-process limit (including stop/wait), 1 MiB stdout, 64 KiB stderr and a 120-second collection deadline. Both streams are checked after process exit as well as during polling. Timeout kills/reaps only that Docker client; it never cleans up a container. Diagnostics exclude captured daemon output. Population is at most 128 registration records, 128 unique owners and 128 globally enumerated containers, within the original per-registration 16-owner bound. Provenance reads have aggregate/time bounds; prospective complete capsule size, repeated topology/proof, bounded finish strings and wait records are conservatively budgeted BEFORE stop. Capsules and protected files are at most 1 MiB, validation has a two-second deadline, timestamps are finite, and issuance TTL is at most 300 seconds. Larger populations/topologies refuse, without enlarging kernel bounds.

### Capsule, origin and resume boundary

The exact closed capsule keys are:

```text
revision kind invocation_id original_invocation deployment_id scope
registrations registration_digest kernel_proof kernel_proof_digest purpose
issued expires collection_seconds topology provenance waits
```

Revision is integer 1; kind is `p1-closed-host-authority-v1`. `registration_digest` binds the canonical digest-sorted COMPLETE retained registration set. `kernel_proof` remains exactly **`{kind, observation, prior_epoch, old_lane}`** and has the original host-admin teardown kind. Its digest and exact plan/purpose are bound without rewriting the proof or a plan. `topology` is `{scope, volumes, consumers}`. Each volume is `{role, name, driver, created, source, device, inode}`; each consumer is `{id, image, started, finished, running, restarting, pid, restart, pidmode, privileged, capadd, cgroupns, mounts}`; each mount is `{type, name, source, destination, rw}`. `provenance` has one protected registration document per retained registration. `waits` has one `{id, command_status, container_status}` per owner, command status exactly integer zero. Closed nested fields, exact types, finite numbers, integer bounds, duplicate JSON keys, duplicate/omitted identities and aggregate byte/population/time bounds are checked. Hashes bind content, **not host origin**. `validate_host_capsule` explicitly validates content only and grants no operation.

Host evidence lives only below the fixed `/var/lib/py-audio-box/p1-host-authority`: host-admin-owned 0700, non-writable/non-symlink ancestors, exclusive no-follow 0600 regular single-link bounded files, file fsync and directory barriers. Provenance names are derived from registration digests. Each explicit freeze attempt exclusively creates `invocations/<new 32hex invocation ID>/capsule.json`. There is no overwrite/remint/delete-after-read path.

`recovery.read_host_capsule()` reads ONLY `/run/p1-host-authority/capsule.json`. It verifies ACTUAL `/proc/self/mountinfo`, a separate RO mount rooted at the exact fixed host invocation path, no writable/ancestor/nested proof alias, fixed role mounts, and actual role directory device/inode plus mount-root/device agreement with the frozen volume topology. It checks the fixed mapped owner, directory 0700, file 0600/no-follow/regular/single-link/stable identity/bounded bytes and repeated kernel topology. Missing trusted launch mount refuses: no ordinary-service file, environment, mock provenance label, actual-host label/hash or local CLI fallback authorizes it. It returns a validated retained capsule only, and still cannot call apply.

The host admin and pinned one-shot launcher are trusted. No claim is made against a malicious host administrator. Host-side ownership is the actual invoking host-admin UID; the proposed maintenance mapped UID is explicitly **container UID 0**, independently checked inside the maintenance namespace. These are separate namespace obligations, not an equality between a macOS UID and container UID. C1 must qualify a deliberate compatible NEW-installation root/user mapping, image/entrypoint/user/mount tuple and `roots_checked` model. No existing data is chowned. The current package supplies no qualified proof mount, so the shipped reader refuses there.

Explicit unfinished resume loads the selected protected ORIGINAL invocation capsule even after its issuance expires, validates its original contents at original issuance, and rechecks the original registration union, daemon, volumes, owner images/starts/stopped states and exact finished timestamps. It does NOT stop again, but executes a fresh successful wait for every owner and repeats final topology. Fresh capsule has a new invocation ID/time and original-invocation reference; original kernel proof/digest stay byte-for-byte/content-identical. A previously plan-bound original also retains its exact plan ID/digest. A same-container restart followed by another stop changes start/finish and refuses BEFORE any stop/wait or evidence publication. Current rotated lane/epoch does not substitute for the accepted original proof. Reading is repeatable and retains evidence; one-shot means one bounded explicit attempt, not consuming a file. Historical completed kernel inspection remains unchanged and never invokes these stop/wait helpers.

### E1 negative and preservation evidence

Only deterministic mocked Docker and explicit fake OS/transport boundaries are exercised. Procfs/cgroupfs tests use synthetic local trees; capsule transport tests fake only mount/open/UID namespace boundaries and use real protected temporary evidence files. There is no exported mock-authorization control. Tests assert exact Docker argv, control-before-worker order, successful nonzero-container-status wait, no unauthorized stop, no kernel invocation, unchanged logical SQL and retained resources. The full retained-union positive includes old STOPPED owners; the omission negative has a complete protected old provenance file, so it reaches actual missing-consumer validation rather than merely missing fixture setup.

| Fault group | Observed required behavior |
| --- | --- |
| Wrong context/daemon/volume/image/start; removed/old omitted owner; unknown stopped/live owner; duplicate or ambiguous incarnation/boot | Refuse before stop/wait and capsule creation |
| Nonprivate PID/cgroup namespace, privileged/cap-add, changed PID, foreign/guessed cgroup, child escape/namespace, PID reuse, IPC alias, unsupported remote/Desktop mode | Refuse before registration mutation or freeze stop |
| Restart policy; mount mode/source/extra mount; underlying volume alias; duplicate or changing enumeration | Refuse before stop |
| Stop/wait failure/timeout/bad output; empty finish; post-stop restart; final topology/registration drift; global deadline | Retain containers/evidence, publish no capsule, no authority mutation or cleanup |
| Unknown nested fields, wrong types, nonfinite/exponent overflow, duplicates/omissions, forged label/hash, E1, expiry, oversize/time budget | Closed content rejection; digest never authenticates origin |
| Absent/writable/ordinary-service proof path; alias/nested mount; wrong device/root/inode/owner/mode/link; file/mount replacement | Trusted reader refuses; evidence retained |
| Final stdout/stderr overflow, process timeout, global timeout, spawn failure | Bounded fixed client, sanitized error, client-only kill/reap |
| Exact unchanged complete mocked host/capsule/RO transport, repeated read and original-proof resume | Produces/verifies capsule; ALL apply still unavailable |

Fresh baseline command, on original `38b6221` source temporarily saved/restored in THIS worktree with `finally`, original F4 tests unchanged:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b22d_r4 -q
```

**8 passed, 652 deselected in 18.49s**, exit 0, `/tmp/b23-baseline.log` (new B2.3a tests were already present in the evolving collection). This is a fresh actual source-baseline receipt, distinct from root's 22.37s receipt.

Baseline RED command during the same guaranteed source restoration:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k 'b23_closed_authority_entrypoints or (b23_fixed_subprocess and (stderr or global-timeout))' -q
```

**3 failed, 668 deselected in 0.94s**, exit 1, `/tmp/b23-red.log`: missing producer entrypoint and both final-stderr/global-timeout missing-refusal assertions. This is missing-feature/real bounded-subprocess RED evidence, not a claim that all later negative fixtures ran against an absent producer.

Development failures are retained honestly: `/tmp/b23-dev.log` **38 failed / 35 passed**, 19.08s, exposed an overstrict ISO timestamp regex; `/tmp/b23-dev2.log` **67 failed / 6 passed**, 18.41s, exposed a missing `digest` import. The initial process/registration subset had 2 failures / 14 passes from that same import. `/tmp/b23-dev3.log` and `dev4` each **2 failed / 87 passed**, 6.56s / 6.70s: the injected inspect fault also intercepted context-inspect JSON, corrected to container-inspect only. `/tmp/b23-dev5.log` **81 setup errors / 8 passed**, 15.12s: the new retained-only reader incorrectly assumed installation's three-column shape instead of its frozen five columns; corrected without schema change. These are failed evolving-source receipts, not passing negatives. Subsequent 92-, 110-, 114- and 115-case development runs passed; the final focused/full receipts below supersede them. A one-line syntax error during an intermediate edit was caught by import and corrected before tests; no external operation ran.

Initial targeted command before the clock-boundary correction:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b23 -q
```

**116 passed, 561 deselected in 7.14s**, exit 0, `/tmp/b23-target-frozen.log`, source/tests subsequently frozen as `4390c0354162b79ae545ec71357ccbb3ea4d5d27`. No installs, bytecode/cache disabled, existing interpreter only.

Final preservation commands run sequentially on the corrected frozen source/tests identified below; only this handoff changes:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

Final focused/full outcomes follow after completion. No redundant unchanged-source broad campaign is planned.

### Remaining installation and acceptance limits

E1 mocked host evidence is **NOT actual-host proof**. No actual Docker command, network/audio/model work, paid/E2E runner, install, push/PR/deploy, prune, cleanup or other worktree/research mutation ran. Supported source envelope is narrow local Linux, fixed local Unix Docker socket, exact cgroup-v2 paths, local option-free volumes, no added capabilities/privileged/shared PID topology. Docker Desktop, remote/VM, alternate/delegated cgroups, larger populations and unprovable namespace associations refuse this provenance mode. The actual C1 image, host-admin identity, mapped UID 0/root ownership model, mount topology and fixed launcher remain UNQUALIFIED. C1 must account for the exact launcher-created maintenance consumer/image/command/user/mount tuple; no wildcard exception is implemented here, so an already-running unregistered maintenance consumer also refuses enumeration. Ordinary services must receive neither proof mount nor Docker socket. No packaging improvisation is supplied.

Production preparation, the conservative unchanged-budget/existing-high-water footprint proof, general low-high-water preparation and budget increases remain outside B2.3a. B2.3b/c, B3, C1/C2, fresh independent Sol-low review, parent integration and final global Astra remain pending. All 18 inherited installed/platform skips remain unqualified if skipped again. A1 global Docker-inventory drift remains **unexplained and unqualified; no cause or waiver** is inferred. Physical ENOSPC/power-loss/daemon/VM/controller durability, actual installed lifecycle/host teardown, clean-host operation, engines/audio/hardware and public/unattended use remain unqualified. No B2/P1/global acceptance is claimed.

### Pre-handoff correction: trusted-reader clock is not caller input

A final source read found a concrete boundary defect in the initial `read_host_capsule(*, now=None)`: a programmatic caller in the trusted mount context could pass the original issuance time to accept an expired invocation. Although apply remained unavailable, leaving that override on the future trusted transport boundary was wrong. Content-only validation still needs explicit time for retained expired ORIGINAL evidence; the authenticated reader does not.

Before editing source, identified and SIGTERM'd ONLY this task's exact focused pytest child PID 76123, parent 76122, running the four named B2 suites. Wrapper reported **focused exit -15**; `/tmp/b23-focused-frozen.log` retained partial progress with no displayed failure. Full service/release had NOT started. This interrupted run is not a complete receipt. No unrelated process was stopped.

Regression command `tests/service/test_recovery.py -k b23_expired_trusted -q`, existing interpreter/bytecode/cache settings above, on initial production: **1 failed, 677 deselected in 1.32s**, exit 1, `/tmp/b23-clock-red.log`. With a fully valid fake RO transport and OS clock after expiry, the ordinary call refused correctly but supplying the original clock reached **DID NOT RAISE**. The exact correction removes the reader's `now` parameter and makes its content check use the OS clock. No production force/test clock/authorization parameter remains. Tests fake the OS boundary explicitly; retained content-only historical validation remains separate.

Corrected targeted command `tests/service/test_recovery.py -k b23 -q`: **117 passed, 561 deselected in 7.25s**, exit 0, `/tmp/b23-target-corrected.log`. Final frozen source/tests are now **`eb6e0c69205d49e39123ef3597cb569a747052d9`**, after initial `4390c03`. Only recovery and the expiry regression change in that correction; all four-path ownership and unavailable apply/kernel boundaries remain. The final four-B2 focused and service/release commands above restart sequentially on this corrected frozen commit, logs `/tmp/b23-focused-corrected-frozen.log` and `/tmp/b23-full-corrected-frozen.log`. Only this handoff changes during those checks. This concrete reproduced correction justifies restarting the incomplete preservation campaign.

Corrected frozen focused result: **810 passed in 539.49s (8:59)**, exit 0, `/tmp/b23-focused-corrected-frozen.log`, on unchanged `eb6e0c69205d49e39123ef3597cb569a747052d9`. Includes all 117 new B2.3a cases and the complete inherited recovery/packaged lifecycle/Admission/Dispatcher preservation population. Full service/release starts sequentially afterward on exactly the same source/tests; only this handoff changes. No focused failure occurred on this corrected frozen candidate.

Corrected frozen FULL preservation result: **962 passed, 18 skipped in 577.81s (9:37)**, exit 0, `/tmp/b23-full-corrected-frozen.log`. Sequential after the 810-pass focused suite on the SAME unchanged `eb6e0c69205d49e39123ef3597cb569a747052d9`. All 18 skips remain unqualified. Neither completed corrected frozen suite had a failure; no source/tests changed during or between them. No further broad campaign ran.

Final `git diff --check` passes. Source/test comparison to frozen `eb6e0c6` is empty. The entire **131,362-byte original recovery kernel prefix** from actual baseline `38b6221` remains byte-for-byte unchanged. Protected-path comparison excluding the exact four authorized paths is empty. Complete baseline-to-candidate path set is exactly the host script, recovery module, recovery tests and this appended handoff. No other worktree, root reviews/ledger, packaging/dependencies/contracts or research were changed. Final local sequence is initial implementation `4390c03` → reproduced clock-boundary correction `eb6e0c6` → report-only commit; that last exact candidate SHA accompanies the handoff. Working-tree/source identity is checked again after committing this report.

Ready for parent-owned fresh independent Sol-low review of the exact candidate and root integration verification, **not self-approved**. B2.3b/c, B3, C1/C2, final global Astra, actual-host/installed qualification and public/unattended acceptance remain pending; A1 inventory drift remains unexplained/unqualified without cause or waiver. All apply remains unconditionally unavailable.

## B2.3a round-1 F1/F2 topology correction — awaiting fresh independent review

2026-10-17, implementation gpt-6.1-sol / low. Only `/Users/ben/.txcode/worktrees/py-audio-box/t3-c77b484f`, branch `t3/b23a-bind-mount-boundaries`, changed. Actual clean fresh-root baseline `aa72485` lacked the candidate. Inspected the complete `38b6221..05afbaec0c3d61f60bb2f430d9108da30f285d0c` chain and imported ALL three implementation/handoff commits oldest first: `4390c03`, `eb6e0c6`, `05afbaec`, locally `eb3843f`, `017dbb8`, `533aa67`. Range-diff reports all three patches equal. Root docs/reviews are preserved; no other checkout was edited.

Read the full R1 report including its exact reproducer, minimal fixes and source boundaries; full P1B2-KERNEL-ASTRA trusted boundary/B2.3a exits; original P1B2-BOUNDARY-ASTRA decision; full RECOVERY-P1B2 original contracts, tranches, corrections, B2.3a response and caller-clock correction; orchestration protocol and R1 fault-authority context. No applicable ancestor/scoped AGENTS.md was found. A lightweight memory registry search returned no relevant B2 guidance; no historical result supplies current acceptance.

Frozen source/tests: `37d2457` (full SHA recorded in final handoff). Correction source/test paths are only host operator, recovery and recovery tests. This appended report is the fourth authorized path. No kernel/preparation/apply wiring, packaging/dependencies/contracts/ledger/reviews or other path changes. The original 131,362-byte recovery prefix is byte-for-byte unchanged, SHA-256 `e4206e1a719dfa969fe73962cae8e67f98e6c1d7064dd0193c22e5cfa5ed6e64`. Every apply remains unconditionally unavailable and enrolled activation closed.

### F1 response: conservative unsupported storage, before any stop

No new host topology resolver is claimed. `_mount_touches` retains exact deployment-volume name, canonical lexical ancestor/descendant and root device/inode matches as positive consumer detections. These positives do NOT prove that any other source is disjoint. For EVERY remaining bind/volume source it now raises `unsupported_host_storage_disjointness`, including same-filesystem/different-inode sources and different-device sources. Missing/stat-unavailable sources also refuse. Thus an ordinary Linux ancestor bind alias with a descendant inode cannot be silently excluded. Different device alone is deliberately not a disjointness proof: nested mounts would require a reviewed complete filesystem-topology algorithm. Unrelated foreign bind/volume storage is unsupported in this envelope; there is no guessed same-filesystem disjoint case, allow-label/hash or wildcard trusted consumer.

The rule runs in EVERY existing all-container inventory, including stopped containers: preflight; complete repeat immediately before any stop; final stopped inspection; repeated final inspection; and inspection after re-reading retained registration authority immediately before publication. The existing 128-container, 16-mount/container, fixed subprocess output/per-process/global bounds and repeated daemon/volume/enumeration checks remain. This correction adds no OS traversal, mount/proc operation, unbounded observation or dependency. It sacrifices availability for unproved foreign storage rather than asserting a topology proof. Exact known consumers still proceed to the unchanged strict incarnation/role/mode validator; direct-root aliases and unknown exact-name consumers still refuse.

Sixteen F1 regressions cover LIVE/STOPPED exact R1 `/alias/staging/_data/retained-subdir` (device 7/inode 999), same-filesystem foreign storage, different-device unproved storage, and direct-root alias. Each appears either in the first inventory or ONLY in the repeated inventory immediately before stop. Required result: refusal, ZERO stop/wait, no invocation/capsule, unchanged logical SQL and all three retained containers with unknown running state unchanged. The inherited exact-name negatives and unchanged complete-owner/retained-registration positives remain.

### F2 response: every observation rejects all fixed-role submounts

`_host_authority_mount_observation` wraps the actual bounded kernel mount reader and rejects EVERY mount target below `/control`, `/staging`, `/receipts` or `/ipc`, including managed-file mounts. There is no submount exception in this lot. All THREE reader observations use this wrapper: initial topology, after protected capsule read, and after content/role identity validation. The separate fixed RO proof mount checks, proof alias/nesting refusal, role root presence/source/device/inode checks and full repeated mount-list equality remain intact.

Twenty-four F2 cases add one foreign writable mount at a time, with unchanged base role identities and capsule bytes, at a directory and at the exact R1 destinations `/control/core`, `/receipts/journal.json`, `/staging/foreign`, `/ipc/lifecycle.sock`. Each variant is present from observation 1 or first appears at observation 2 or 3. All require `host_role_nested_mount`, exactly that observation reached, and unchanged capsule/logical SQL. The unchanged fake RO transport is established first in every case.

### Actual RED/GREEN and frozen verification

All pytest commands use the existing `/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`; no installs. Disposable synthetic roots only. Exact report probe extracted without changing its Python content to `/tmp/b23a-r1-exact.py`: first invocation failed importing `service` because file execution lacked repository PYTHONPATH. This setup failure is not defect evidence. Reissued with `PYTHONPATH=.` and bytecode disabled: exit 0, `/tmp/b23a-r1-exact-red2.log`, printing LIVE/STOPPED alias omitted/capsule published, all four nested roles accepted, and OS-expiry/no-caller-clock controls passed. These assertions demonstrate incorrect acceptance, not correctness.

Before production edits:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b23a_r1 -q
```

**14 failed, 18 passed, 678 deselected in 9.61s**, exit 1, `/tmp/b23a-r1-correctness-red.log`. Six non-root-alias F1 negatives and eight stable first-observation F2 negatives reached missing-refusal assertions. Direct-root aliases already refused; mount drift on later observations already refused via equality. No setup failure counted as defect evidence.

Initial combined B2.3 GREEN after production edits: **149 passed, 561 deselected in 14.22s**, `/tmp/b23a-r1-target-green.log`. Then strengthened only tests to inject F1 storage on the second inventory and require the specific F2 refusal on all observations. Final targeted command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k b23 -q
```

**157 passed, 561 deselected in 14.66s**, exit 0, `/tmp/b23a-r1-target-final.log`. Includes all 40 new cases and all 117 inherited host tests (clock, retained union, unique authenticated hello/proc binding, resume, subprocess bounds, stop/wait order and failures, fixed RO transport). No development failure followed the intended RED. Source/tests then frozen at `37d2457`; only this report changes during final sequential suites.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

Frozen results follow below after completion. No redundant broad campaign is authorized by these receipts.

### Preservation and qualification limits

All retained-registration union, all-container including stopped enumeration, exact full IDs/image/start/finish/private PID/cgroup/hello binding/restart-no/volume roles/modes remain. Fixed shell-false bounded Docker operations, control-before-worker and successful wait, meaningful finish, repeated daemon/consumer inspections and failure without cleanup remain. Trusted fixed RO mount, OS clock, protected owner/0600/no-follow/single-link/bounded stable files and no force/environment/E1 bypass remain. Hashes bind content, not origin. No kernel, preparation, apply or packaging expansion occurred.

E1 mocks only: fake Docker/OS boundaries are NOT actual-host authentication. No actual Docker/proc/mount operation, network/model/audio/paid/E2E runner, install, push/PR/deploy, prune or user cleanup; research and other worktrees preserved. Local Linux remains the selected supported provenance envelope; Desktop/remote refusal remains. Actual C1 host-admin/mapped UID/image/entrypoint/mount/launcher proof remains pending. All installed/platform skips remain unqualified. A1 global inventory drift remains unexplained and unqualified, with no cause or waiver. B2.3b/c, B3, C1/C2, parent integration, fresh independent review, final global Astra and public/unattended acceptance remain pending. No self-GO.

Frozen focused result: **850 passed in 540.10s (9:00), exit 0**, `/tmp/b23a-r1-focused-frozen.log`, on unchanged `37d2457c73de6712534d237017336b37a9d6531e`. Frozen full result, sequential afterward: **1002 passed, 18 skipped in 578.07s (9:38), exit 0**, `/tmp/b23a-r1-full-frozen.log`, on the SAME source/tests. All 18 skips remain unqualified. No source/tests changed during or between these suites; only this handoff changed. No final-suite failure or redundant broad campaign occurred.

Final `git diff --check` passes. Source/tests-to-frozen comparison is empty; frozen-to-final is report-only. Complete actual-root `aa72485` to candidate path set is exactly the four authorized paths. Protected comparison excluding these paths is empty. Imported three-commit chain is range-diff equal. The original 131,362-byte kernel prefix remains byte-for-byte unchanged. Final local sequence: imported `eb3843f` → `017dbb8` → `533aa67`, correction/frozen source `37d2457c73de6712534d237017336b37a9d6531e`, then report-only commit. Exact candidate SHA accompanies the handoff. Fresh independent review of that final candidate and parent integration verification remain required; no self-GO or actual-host/installed/public/unattended acceptance.

## Lot K — bounded maintenance process and exact snapshot handoff (awaiting independent review)

2026-10-17, implementation `gpt-6.1-sol` / low. Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-3e867443`, branch `t3/astra-bounded-snapshot-handoff`; clean actual baseline **`8ef0b0c91006f2e2c9c4ff63185ec77e9e440c84`**. Frozen source/tests **`4c2066189848b3f254fe0e6aa54b043344a7c65e`**. Root's recorded accounting decision and independently reviewed B2.3a are present at baseline. No B2.3b candidate was imported. Inspected `59c4a7373f899f1397277ba4a5a290390c3201bf` measurement material read-only, only after reading the actual baseline scan; no cherry-pick, preparation builder or always-refusing candidate adoption. No other worktree/research mutation.

Read full P1B23B-ACCOUNTING-ASTRA, P1B2-KERNEL-ASTRA, P1B2-BOUNDARY-ASTRA and this original handoff/contracts/corrections, P1B23a-r2, orchestration protocol and R1 fault-authority context. No applicable AGENTS.md was found. Memory supplied only general checkout/evidence-tier guidance, no current B2 authority. This is an implementation handoff, **not self-GO or B2.3b completion**. Parent owns independent Sol-low review of this exact candidate, including the retained measurement factor, root integration verification and final Astra. No subagents were launched.

### Process, platform and authority boundaries

`_run_bounded_maintenance(roots, scope, operation)` is an INTERNAL one-shot fork seam, unreachable from CLI/apply. `operation` is internal Python code for the future P owner, never a serialized request, executable, generic subprocess service or capability option. It returns a bounded 64 KiB JSON result, with a 30-second parent deadline; failed/signalled child, timeout or invalid/oversize result refuses. Only this task's own child can be killed/reaped on that deadline. There is no success result on SIGXFSZ/EFBIG/SQLite failure. The child exits after one operation. Normal Core, B1 inspection and the invoking host shell never receive the child's resource limit.

Before any SQLite connection in that child, `_install_maintenance_limit` selects deliberate `SIGXFSZ=SIG_IGN` (so extending writes report errors), sets BOTH soft and hard `RLIMIT_FSIZE` to the internal fixed **4,194,304** and verifies exact readback. The process token checks PID, the private child-entry lifetime, Python thread count, bounded Linux `/proc/self/task` count and exact limit on session entry/capture/continuation. Unsupported platform, setup/readback failure and a fresh unfinished invocation without the same cap refuse. No caller/environment/plan limit, capability-enable flag, restore, auto-raise or retry with larger limits exists. Internal objects are plumbing, not authorization; arbitrary Python execution inside this module is not the threat model.

Lock order is **control.lock → lane.lock**, both nonblocking, held through host/size checks, original connection open, capture, acceptance/continuation and connection close/checkpoint. `_locked_roots` is shared with the unchanged ordinary offline interface; the new session never reacquires its own locks. The Linux model admits only fixed `/control`, `/staging`, `/receipts`, `/ipc` role roots on declared local **ext4**, no nested role submount, overlay/network/FUSE/unknown filesystem. The existing fixed trusted RO capsule reader remains a separate mandatory check BEFORE original writable open. Under the locks a private WAL-aware copy checks existing WAL mode, absence of triggers and exact full retained-registration set against that capsule. No hash, process token or supplied E1 string replaces trusted transport/full freeze.

Prewriter and repeated file checks require original DB/WAL/SHM to be owned regular single-link files each <= L; unknown `core.sqlite*` sidecars refuse. No larger file is truncated. Static measured evidence is not excluded merely because it exceeds L; copying unsupported retained evidence can refuse. The original private writer requires existing WAL (no journal-mode conversion), FULL, `trusted_schema=OFF`, `temp_store=MEMORY`, `mmap_size=0`, disabled automatic checkpoint and disabled extension loading. Its authorizer rejects ATTACH/DETACH, extension loading, trigger/virtual-table creation and unsafe pragma changes. Reads of schema pragmas remain available to the frozen validators. No VACUUM, manual WAL deletion, preallocation, persistent page limit, helper child writer or original restoration is added. All original write/checkpoint/backup-close operations remain in the limited one-shot process.

This E1 source model assumes standard SQLite Unix VFS and ordinary Linux regular-file write/pwrite/truncate/ftruncate semantics, no mmap EOF extension, external writer, child-helper escape, extension, unknown trigger or privileged limit-raising launcher. C1/C2 must still identify and prove the actual kernel/build/VFS/filesystem/mounts/UID/capabilities/entrypoint and trusted freeze. The source does not claim installed enforcement, defense against a malicious host administrator, physical disk allocation bounds or guaranteed completion. Hitting L may leave accepted recovery pending/quarantined. No auto-raise, SQL retry or healthy-half-state repair follows.

### Exact measurement, namespace and bound

Retained only the candidate's narrow measurement extraction, re-audited against actual baseline source: an executable byte comparison confirms (1) the **entire original bounded scan loop** is identical, (2) the suffix from job/staging eligibility through max/high-water, every demand charge, quota/free-reserve and exact accepted-output equality is identical, and (3) the complete ordinary `ResourceStore` class is identical. `_measure_recovery_roots` measures `st_size`, including descendant directories and sparse logical lengths, excluding each root directory's own size. Independent review of this retained factor is still mandatory.

`_maintenance_footprint` exposes the selected fixed conservative calculation, without accounting writes:

```text
L = 4_194_304
snapshot <= 1_048_576
canonical control manifest <= 65_536
later control certificate <= 65_536
M = unchanged bounded measured st_size across control/core + staging
U_all_early = M + 3*L + 1_048_576 + 2*65_536
require U_all_early <= retained H
require H + ALL retained demand.disk_bytes <= unchanged enrolled B
require current actual free reserve; original scan repeats the real check
```

Finite integer checks and a conservative five-entry reservation retain the existing 2,048 scanner ceiling: up to two SQLite sidecars, two flat archives before scan and one later flat certificate. Existing SQL/evidence/receipt aggregate 2,048 / 16 MiB limits, journal limits and 64 KiB receipt certificate limit remain. Capture checks actual retained H/charges and this formula before its backup. P still must call the available early guard before its schema/plan writes; K writes no plan or preparation output.

The only measured mutable files are `control/core/core.sqlite`, `core.sqlite-wal`, `core.sqlite-shm`. New measured files are flat `p1-recovery-<archive_id>.sqlite` and `.manifest.json` BEFORE scan; `.certificate.json` is AFTER scan/Core COMMIT. Existing media, static files, prior archives and all staging entries remain included in M. Receipt accepted documents, bundle, creation descriptors, journal and replacements remain outside those two measured roots, with their original independent limits. No counted descendant directory gets a new entry; staging is untouched. **Counted directory delta = zero**, conditional on this fixed namespace/freeze, because control/core's own directory size is excluded. Tests trace actual kernel file destinations, verify no descendant/staging mutation and separately show exact flat archive growth by actual file lengths. No scanner exclusion, budget increase, H inflation, charge credit, synthetic padding or disk fill is used.

### Acyclic snapshot ownership and shared kernel

`_MaintenanceSession.capture(plan_id, plan_digest)` requires the actual immutable current plan row/digest, no transaction/current accepted recovery, and one capture per session. It backs up the ACTUAL original committed database through a protected WAL-aware copy, AFTER the existing plan and BEFORE accepted intent/step writes, at most 1 MiB. `_PreparedSnapshot` retains exact bytes/hash/plan identity and its owning live lock-held session. The hash is available for P's later acyclic intent construction. No P builder, plan INSERT/UPDATE or digest rewrite is supplied by K.

`session.continue_kernel(intent, prepared=owned)` permits only that exact same-session, once-consumed object, matching plan/hash and unchanged trusted capsule proof. Plain bytes, another object/session and repeated consumption cannot select this route. Actual hash and complete snapshot/schema/rows/current plan are checked by the original archive oracle; current source/history/old archives are checked by the shared original kernel. Manifest/control/receipt certificate and accepted-document limits are checked before acceptance. Receipt acceptance/fence and **owned exact durable bundle precede Core accepted SQL**, unchanged. No snapshot is restored over Core.

The fixed kernel body is factored to `_pending_prefix_locked` for the owning session; existing supplied-document semantic helpers retain their INTERNAL unbounded-fixture route and unchanged validation/order/no-remint history semantics. The real defect is corrected: supplied frozen bytes no longer trigger another backup merely because acceptance is new. Both supplied fixture bytes and owned bytes pass the SAME prospective actual archive/SQL validators. Only absent bytes on the old semantic-fixture path may recapture. The new bounded session NEVER selects that branch: initial invocation requires its owned capture; fresh continuation loads only exact owned accepted document and <=1 MiB bundle. A public four-field preparation result alone is not restart authority.

Any kernel exception, including ambiguous COMMIT, burns the session; no second live SQL attempt. Fresh invocation reinstalls/rechecks the cap and reads retained state. Missing attempted bundle/artifact remains quarantine; no remint. Existing F1 registration, F2 prefix-window, F3 prerequisite and F4 all-completed-history checks remain in the shared kernel. Full production preparation/capacity rehearsal and final closed apply/resume wiring remain P/B2.3c work; no alternative SQL recovery writer is introduced.

### Actual RED/GREEN and frozen preservation

All commands use the existing interpreter, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`; no installs. RED BEFORE source implementation: `tests/service/test_recovery.py -k lotk -q`, **1 failed, 718 deselected in 1.33s**, `/tmp/lotk-red.log`. The coherent original snapshot was supplied, and the actual old prefix reached the forbidden recapture assertion. This is behavioral RED, not an absent-name test. First handoff/predecessor GREEN: **2 passed, 717 deselected in 1.34s**, `/tmp/lotk-green1.log`.

Development failure retained: `/tmp/lotk-dev.log`, **3 failed, 13 passed, 1 skipped in 11.61s**. A new connection authorizer misclassified `PRAGMA table_info(table)` as a mutating pragma, blocking the unchanged schema validator. A two-case diagnostic reproduction confirmed `sqlite3.DatabaseError: not authorized`; the temporary diagnostic traceback was removed. Corrected by allowing the finite read-only schema pragmas, without relaxing unsafe mode/mmap/page-limit changes. `/tmp/lotk-dev2.log`: **16 passed, 1 skipped in 5.04s**. Expanded development: `/tmp/lotk-dev3.log` **18 passed, 1 skipped in 5.25s**; `/tmp/lotk-dev4.log` **24 passed, 1 skipped in 9.51s**. These evolving-source results are not frozen preservation receipts.

Final targeted command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py -k lotk -q
```

**33 passed, 1 skipped, 796 deselected in 11.05s**, `/tmp/lotk-target-final.log`, on source/tests frozen immediately afterward as `4c2066189848b3f254fe0e6aa54b043344a7c65e`. Tests include same-lock actual snapshot hash/rows through all five kernel steps and exact durable bundle/archive bytes; explicit process/OS model; setup/readback/platform/host/both-lock/oversize/unknown-sidecar refusal with a separate parent-visible original-open trap; caller-limit/environment/capability refusals; byte/plan/source/old-archive tampering; namespace/formula/charges/equality preservation; connection escape refusals; unfinished fresh readback failure; failure before bundle and before/after Core COMMIT with no partial healthy authority and burned session.

Two actual child deaths at bundle-parent and Core-intent COMMIT are followed by a FRESH interpreter receiving ONLY roots/scope/recovery ID. It loads owned accepted state and completes/replays twice through the bounded seam, retaining one accepted recovery/five steps without caller snapshot RAM. Host/Linux OS boundaries in these macOS runs are explicitly modeled, not installed evidence. The real Linux-only child test covers successful boundary write/pwrite/truncate to L, refused L+1 extension with maximum length <=L, and actual SQLite DB/WAL/SHM growth/failure/checkpoint/close. It is **SKIPPED on this macOS host: actual Linux cap evidence PENDING**, never counted as proof. The root agent/pytest/shell limit is never lowered.

Frozen final commands, sequential, with source/tests unchanged and only this appended report edited:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

Final focused/full results follow after completion. No redundant unchanged-source campaign is planned. Only the exact five assigned paths change; no Core/persistence/admission/dispatcher/worker/host/packaging/dependency/contract/ledger/review edit. No Docker/network/audio/model/paid runner/E2E script/push/PR/deploy/prune/user cleanup ran. Research and other worktrees remain preserved.

K is a candidate storage/snapshot foundation only. Independent review (including retained factor), parent integration and final Astra remain pending. **P, B2.3b/c, B3, C1/C2, actual-host/installed Linux enforcement, physical ENOSPC/power-loss/daemon/VM/controller durability, clean-host, hardware/audio and public/unattended acceptance remain pending. A1 global Docker-inventory drift remains unexplained and unqualified, with no cause or waiver.** Every apply entrypoint stays unconditionally unavailable; enrolled activation stays closed.

### Pre-handoff correction: explicitly select standard Unix VFS

Source review during the first frozen focused run found `_maintenance_session`'s original `sqlite3.connect(...?mode=rw)` implicitly selecting the default VFS. The supported model requires standard Unix, so relying on a potentially alternate default was unnecessarily weaker than this lot's contract. Before editing any source/test, verified ONLY this task's pytest PID 78498, wrapper 78497, exact four-suite command and cwd via `lsof`. SIGTERM to that pytest yielded **focused exit -15 after 179.14s**, partial progress without a displayed failure, `/tmp/lotk-focused-frozen.log`. Full service/release had NOT started. This interrupted result is NOT a preservation pass.

Added an original-writable-open regression checking explicit VFS selection. RED on `4c20661`, `tests/service/test_recovery.py -k lotk_original_connection_pins -q`: **1 failed, 750 deselected in 1.70s**, `/tmp/lotk-vfs-red.log`. The writer-open interceptor rejected the original default-VFS URI, causing the child failure before the successful-session assertion. Correction pins the bounded original writer, protected prewriter copy reader and actual snapshot backup source/destination to fixed **`vfs=unix`** URIs. No caller VFS selector or fallback is accepted; unavailable Unix VFS fails SQLite open. Standard registered Unix VFS/build remains a C1 qualification assumption, not defense against a malicious administrator replacing SQLite code.

Corrected final targeted command is the same two-file `-k lotk` command above: **34 passed, 1 skipped, 796 deselected in 11.40s**, `/tmp/lotk-target-corrected.log`. Final frozen source/tests are now **`5d650eae1cdf3a4990f5cf8ba88167f2a8820954`**, after initial `4c2066189848b3f254fe0e6aa54b043344a7c65e`. All initial frozen-source references/results above are superseded for final-candidate attribution by this corrected commit. Only this appended report changes during the corrected sequential four-B2/full-service-release campaign, logs `/tmp/lotk-focused-corrected-frozen.log` and `/tmp/lotk-full-corrected-frozen.log`. This concrete correction justifies restarting the incomplete campaign; no redundant unchanged-source campaign is claimed. Linux cap evidence remains PENDING, every apply remains unavailable, and no P builder is introduced.

Corrected frozen focused result: **884 passed, 1 skipped in 554.40s (9:14), exit 0**, `/tmp/lotk-focused-corrected-frozen.log`, on unchanged `5d650eae1cdf3a4990f5cf8ba88167f2a8820954`. Includes the full inherited B1/B2/R1/W2/host/startup/history population and 35 added K cases (34 pass, actual Linux cap test skipped). All F1–F4 preservation cases pass. Full service/release began sequentially afterward on exactly the same frozen source/tests. Only this report changes.

Baseline identity cross-check: `git diff ee650cc7920d1e237c21c6b795cb4cc6ac4573e5 8ef0b0c91006f2e2c9c4ff63185ec77e9e440c84 -- service worker scripts tests` is empty. Thus actual baseline code is exactly independently reviewed B2.3a, with no unreviewed B2.3b source imported. Source/test comparison to corrected frozen commit remains empty; `git diff --check` passes. Full result follows after completion; neither this focused pass nor this report supplies independent acceptance or actual Linux enforcement evidence.

Corrected frozen FULL preservation result: **1,036 passed, 19 skipped in 590.66s (9:50), exit 0**, `/tmp/lotk-full-corrected-frozen.log`. Sequential after the 884-pass focused suite on the SAME unchanged `5d650eae1cdf3a4990f5cf8ba88167f2a8820954`. The 19 skips are the 18 inherited Linux peer-credential cases plus the new actual Linux RLIMIT_FSIZE test. Every skipped mechanism remains unqualified; macOS E1 models are not actual Linux cap/host proof. No corrected frozen suite failed, and no source/tests changed during or between them.

Final `git diff --check` passes. Complete baseline-to-candidate path set is EXACTLY `service/recovery.py`, `service/resource_store.py`, `tests/service/test_recovery.py`, `tests/service/test_admission.py`, and this append-only handoff. Protected-path comparison excluding those five paths is empty. Source/tests-to-frozen comparison is empty. Ordinary ResourceStore, the factored original measurement loop and scan arithmetic/eligibility/equality remain byte-identical to baseline as described above. No Core/persistence/admission/dispatcher/worker/host/packaging/dependencies/contracts/ledger/reviews or other worktree/research edits. Local sequence: initial source/tests `4c2066189848b3f254fe0e6aa54b043344a7c65e` → explicit VFS correction/final frozen source `5d650eae1cdf3a4990f5cf8ba88167f2a8820954` → report-only commit, whose exact final SHA accompanies the handoff.

Ready for parent-owned independent Sol-low review of the exact candidate, INCLUDING any retained measurement factor and OS assumptions; **NO SELF-GO**. Root integration verification and final Astra remain required. K is NOT B2.3b completion. Production P builder/positive, B2.3c, B3, C1/C2, actual Linux cap/host/installed qualification and final global/public/unattended acceptance remain pending. A1 inventory drift remains unexplained/unqualified without cause or waiver. Apply remains unconditionally unavailable and enrolled activation closed.

## Lot K round-1 F1 private connection correction — awaiting independent review

2026-10-17, implementation `gpt-6.1-sol` / low. Only worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-38eac3cc`, branch `t3/bounded-writer-vfs-hardening`, changed. Clean actual root baseline `88463f8` lacked K. Inspected the complete `8ef0b0c..cc3d790baead83d79117e372461668bfbda34076` chain; imported ALL commits oldest first: `4c2066189848b3f254fe0e6aa54b043344a7c65e` → `5d650eae1cdf3a4990f5cf8ba88167f2a8820954` → `cc3d790baead83d79117e372461668bfbda34076`, locally `c6f5140` → `a884e1f` → `0d3441e`. Range-diff reports all three patches equal. Root documents/reviews and unrelated research/worktrees are preserved. No applicable AGENTS.md found. The lightweight memory registry search returned no relevant K guidance; no memory-derived acceptance is used.

Read the full P1B23K-r1-sol exact first-INSERT reproducer, minimal fix and assessments, accounting arbitration K/P model/formula/snapshot handoff, kernel gate and original boundary, full recovery contracts/tranches/corrections, orchestration protocol and R1 fault-authority context. This response implements F1 only; no self-GO. Frozen source/tests: **`6518c9488715e52faa609d2965fec87f262845d6`**. Correction-only delta is recovery.py and recovery tests, plus this appended report. Complete imported-plus-correction scope remains exactly the five K paths. ResourceStore and admission tests are unchanged from imported K; no ordinary ResourceStore, measurement factor, formula or scan eligibility/arithmetic/equality change.

### F1 response and reached connection inventory

`_run_bounded_maintenance` → fixed child limit → `_maintenance_session` (both locks, trusted capsule and prewriter checks) → `capture` → `continue_kernel` → `_pending_prefix_locked` remains the supported internal route. Relevant SQLite opens are:

| Open / actual use | Fixed URI and configuration | Lifetime |
| --- | --- | --- |
| Prewriter protected WAL-copy reader | `mode=ro&vfs=unix`; shared maintenance hardening before WAL/registration queries | Existing explicit `closing` |
| Original writer | `mode=rw&vfs=unix`; existing WAL/FULL, no auto-checkpoint, shared hardening | Existing session `finally` close under both locks |
| Capture protected WAL-copy source | `mode=ro&vfs=unix`; shared hardening before backup | Existing explicit `closing` |
| Capture backup destination | `mode=rwc&vfs=unix`; shared hardening before backup | Existing explicit `closing` |
| Prospective private SQL writer (F1) | **`mode=rw&vfs=unix`**, shared hardening before FIRST INSERT | **Explicit `closing`**, including validation/commit exceptions |
| Every immutable private archive reader | **`mode=ro&immutable=1&vfs=unix`**, shared hardening before yield/query | Existing explicit `finally` close, now including configuration failure |

Shared hardening disables extension loading, sets trusted_schema OFF, temp_store MEMORY and mmap zero, rejects triggers, and installs the same escape-denying authorizer. ATTACH/DETACH, unsafe mmap/mode/page-limit changes and SQL extension loading refuse at actual use. The sole additional safe pragma allowance is `query_only=ON`: archive validation already sets that restrictive mode after opening; OFF remains denied. Immutable readers retain read-only/immutable semantics; the prospective writer remains writable for the prescribed validation transaction. No ordinary B1 offline/host-state/required-registration connection code changes. `_archive_database` is the existing B2 archive oracle, not a B1 inspection replacement.

Regression intercepts every private writable and reader open reached by a coherent SAME-LOCK owned-snapshot five-step completion, beginning before session/prewriter entry. Checks run at actual schema/registration reads, backup and FIRST production `INSERT INTO p1_recoveries`, never prematurely in the constructor. It checks fixed URI, trusted/temp/mmap configuration and escape refusal; every intercepted handle must be explicitly closed and unusable afterward. The original writer pin/escape tests remain. The existing source-preserving WAL-copy capture checks remain active; no source WAL deletion/checkpoint exemption is added. The owned hash/bundle oracle, no recapture, process/session burn, consumption and fresh-interpreter no-RAM death/resume population remain.

### Actual commands and failures

Existing interpreter only, no installs. All pytest commands below use `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider`. Disposable local roots only; parent/pytest/shell limits never lowered.

Before production edits, extracted the exact report Python unchanged to `/tmp/lotk-f1-exact.py` and ran `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /tmp/pab-0c-venv/bin/python /tmp/lotk-f1-exact.py`: exit 0, printed `[{'at_first_insert': True, 'attach_allowed': True, 'mmap': [1048576]}]`. This reproduces incorrect acceptance, not correctness. Required-correctness regression command `tests/service/test_recovery.py -k lotk_f1 -q`: **1 failed, 751 deselected in 7.11s**, exit 1, `/tmp/lotk-f1-red.log`; the bounded child failed the actual-use audit on the unconfigured private writer. No absent-name assertion or constructor-time escape probe supplies this RED.

Initial development combined K run after hardening: **10 failed, 25 passed, 1 skipped, 796 deselected in 11.27s**, `/tmp/lotk-f1-green.log`. The new authorizer on archive readers denied the existing `PRAGMA query_only=ON`; corrected by the restrictive ON-only allowance above. No unsafe pragma allowance was added. Next development combined K run: **35 passed, 1 skipped, 796 deselected in 17.60s**, `/tmp/lotk-f1-green2.log`. Strengthened inventory interception to include prewriter and backup source/destination, with actual-use reader/backup checks and explicit-close assertions: **1 passed, 751 deselected in 1.82s**, `/tmp/lotk-f1-inventory.log`. Final escape assertions precede configuration queries, and the FIRST INSERT repeats them independently of earlier schema reads.

Final targeted command `tests/service/test_recovery.py tests/service/test_admission.py -k lotk -q -rs`: **35 passed, 1 skipped, 796 deselected in 11.91s**, exit 0, `/tmp/lotk-f1-target-final.log`. Frozen source/tests committed immediately afterward as `6518c9488715e52faa609d2965fec87f262845d6`. Only this report changes during sequential frozen preservation:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q -rs
```

Frozen focused/full results follow after completion. No redundant unchanged-source campaign.

The fixed 4 MiB hard/soft child limit before original write and every unfinished resume, no caller/environment bypass or raise/restore, both locks and separate OS-clock/trusted-host origin remain. Original DB/WAL/SHM <=L and unknown-sidecar prewriter refusals never truncate. Snapshot <=1 MiB, manifest/certificate <=64 KiB, aggregate ceilings and `M+3L+1MiB+2*64KiB<=OLD H`, unchanged charges/budget remain. Capacity/semantic rehearsal and production P builder remain future work. All apply remains unconditionally unavailable; enrolled activation stays closed without flags.

E1 source/local SQLite/filesystem/process with explicit OS/host models only. macOS is NOT actual Linux/ext4/Unix-VFS cap enforcement or actual-host proof. The real Linux cap test remains skipped/unqualified; no Docker/Linux workaround. C1/C2 must qualify actual kernel/build/VFS/mount/UID/capability/launcher/freeze assumptions. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no cause or waiver. P/B2.3b production-generated positive, B2.3c, B3, C1/C2, fresh independent review, parent integration, final Astra and public/unattended acceptance remain pending. Physical ENOSPC/power-loss/daemon/VM/controller durability, installed/clean-host and hardware/audio/engines remain unqualified. No subagents, Docker/network/other-user-proc/mount/model/audio/paid/E2E runner, installs, push/PR/deploy/prune/user cleanup; research and other worktrees preserved. No self-GO.

Additional first-INSERT correctness probe uses the report's coherent setup, replaces its allowed-escape assertions with required `DatabaseError` refusal, and accepts either old Path or corrected URI for interception. Baseline production source was loaded from `git show 0d3441e:service/recovery.py` **in memory in a separate disposable interpreter**, with explicitly modeled OS/limit/host boundaries and direct owned session to expose the assertion (no real process limit change). It reached **`Failed: DID NOT RAISE DatabaseError` at FIRST INSERT**, exit 1, `/tmp/lotk-f1-first-insert-red2.log`. Identical corrected escape assertions through the actual bounded child on frozen current source completed all five steps and printed `[{'at_first_insert': True, 'escapes_blocked': True}]`, exit 0, `/tmp/lotk-f1-first-insert-green2.log`. No checkout source was substituted during preservation. An initial adapted probe accidentally reused its incoming `sql` variable as the escape-loop variable, then re-executed denied load_extension instead of INSERT; it failed with child_failed / direct `not authorized to use function: load_extension`. Corrected the probe variable only; those initial adapted failures are not production defect evidence. The original exact report probe and repository regression receipts above remain separately attributed.

An executable byte comparison against actual root `88463f8` independently confirms the complete ordinary ResourceStore prefix, entire bounded measurement loop, and scan suffix from job/staging eligibility through all charges/high-water/quota/free-reserve/exact equality are byte-identical. Initial comparison script mistakenly selected an earlier ordinary `return total` as its endpoint and asserted; correcting the endpoint search to start at the factored loop passes. No production/test change followed either diagnostic correction.

Frozen focused result: **885 passed, 1 skipped in 555.75s (9:15), exit 0**, `/tmp/lotk-f1-focused-frozen.log`, on unchanged `6518c9488715e52faa609d2965fec87f262845d6`. Full service/release starts sequentially afterward on the SAME unchanged source/tests. The Linux cap skip remains unqualified. No source/test change during or between suites; only this handoff changes.

Frozen FULL service/release result: **1,037 passed, 19 skipped in 604.72s (10:04), exit 0**, `/tmp/lotk-f1-full-frozen.log`. Sequential after the 885-pass focused run on the SAME unchanged `6518c9488715e52faa609d2965fec87f262845d6`. Skips: one actual Linux RLIMIT_FSIZE case and 18 inherited Linux peer-credential cases (10 authenticated exchange, 1 forged control proof, 1 replayed hello, 6 wrong execute). All remain unqualified. No frozen focused/full failure or source/test change during/between suites; no redundant broad campaign.

Final `git diff --check` passes. Source/tests-to-frozen comparison is empty. Actual root `88463f8` to candidate path set is exactly recovery.py, resource_store.py, recovery tests, admission tests and this append-only report. Correction from imported `0d3441e` changes only recovery.py, recovery tests and this report. Root reviews/ledger and all protected paths are preserved. All three imported patches are range-diff equal; retained ResourceStore factor is independently byte-compared above. Local sequence: `c6f5140` → `a884e1f` → `0d3441e` → correction/frozen `6518c9488715e52faa609d2965fec87f262845d6` → report-only commit, whose exact candidate SHA accompanies the handoff. Fresh independent review of this exact candidate and parent integration verification remain required; no self-GO. All pending qualifications and activation closures above remain.

## Lot P — production B2.3b preparation under reviewed K (implementation candidate)

2026-10-17, `gpt-6.1-sol` / low implementation, no subagents or self-GO. Only worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-c5b26ca8`, branch `t3/b23b-production-preparation`, changed. Clean actual baseline **`9b45eac59f177e9166689749dd14cc23efb4a58d`** contains integrated K **`35d0cab`**. Both `git diff 6456ff42 9b45eac -- service worker scripts tests` and `git diff 35d0cab 9b45eac -- service worker scripts tests` are empty. Thus source/tests equal exact R2-reviewed K `6456ff42ed8d588713d5c8d5530c86f70b9a174a`. R2's 1,037 passed/19 skipped and root's 35 passed/1 skipped in 11.87s are **attributed baseline receipts**, not freshly rerun baseline results here.

Read the complete P1B23B-ACCOUNTING-ASTRA (ten dependencies, formula, K and exact P brief/exits), kernel and boundary Astra, full RECOVERY-P1B2 original contracts/tranches/corrections/K connection inventory, K R1/R2, orchestration protocol and R1 fault-authority context. No applicable ancestor/scoped AGENTS.md was found. Lightweight memory-registry search returned no relevant P/K guidance. No `59c4a7` candidate import, fixture intent import/call, new dependency or authority file. Actual changed paths are a SUBSET of the five allowed paths: **recovery.py, recovery tests, and this append-only report**. ResourceStore and admission tests need no new edit: retained K measurement/calculation is reused byte-for-byte. Every other source/worker/host/contract/packaging/ledger/review path is preserved.

Initial source/tests were frozen as **`9ae8e7cebd764c9ff81be2d7849bc9ec2a634420`**. The physical-rehearsal correction below supersedes that freeze; final frozen source/tests and sequential preservation receipts are recorded at the end. The report-only handoff commit follows them. No independent acceptance is claimed.

### Complete production call chain and ten-step ordering

`_prepare_abandon(roots, scope, *, profiles, budget, diagnostic_decisions)` is the closed PRIVATE production request. No actions, before/expected rows, snapshot/proof bytes, clock, high-water, limit, capability selector or budget-after input exists. It freezes request configuration/decisions as canonical JSON and calls K's dedicated `_run_bounded_maintenance` with internal preparation code and a protected prewriter validation callback. This callback is internal plumbing, not a serialized request or activation flag. Public/programmatic/CLI apply remains unconditionally `P1B2_apply_unavailable`; enrolled activation remains closed.

1. K installs/verifies fixed soft AND hard 4 MiB limits in its one-shot child before SQLite opens, verifies process identity/single-thread assumptions and declared local Linux/ext4 model, then takes **control.lock → lane.lock**. The actual fixed-mount `read_host_capsule()` authenticates the pre-plan capsule with the OS clock. Its existing final kernel proof/hash exists BEFORE any plan. Exact deployment/scope/retained registration union, original lane/prior epoch and stopped owner coverage are required. Preparation re-reads the same trusted capsule before its original-state work and after rehearsal: changed content/transport or OS expiry refuses, never substitutes another proof.
2. Under both retained locks, the existing source-preserving private WAL-aware reader runs `_preparation_prestate` BEFORE original writable open. It checks schema/reference/allocation provenance, enrollment/key/volume identities, configuration, full history/media, journal observation, exact diagnostics, early footprint/quota/entries and real reserve. A canonical complete source projection is saved in the private owning session. After K opens/hardens the original Unix-VFS WAL/FULL writer, preparation repeats validation and requires exact equality to that prewriter projection before DDL. No lock reacquisition or nested transaction occurs.
3. `_JournalObservation` is a narrowly typed internal observation: `valid`, `missing`, or bounded `unreadable`, with exact original bytes/absence, fixed known lane and digest. It re-reads protected owned bytes during source validation. Valid journal records still undergo the full existing allocation/runtime/registration checks. Missing/unreadable observations return no clean journal-record authority and require coherent allocation plus complete host coverage. With completed history, the retained receipt fence/certificate is independently grounded against actual Core completion; damaged current bytes are not required to pretend to carry a clean maintenance field. Ordinary B1 snapshot/inspect/plan retains its strict journal path. No half-built accepted intent is supplied to bypass source validation.
4. `_validate_diagnostic_eligibility` is a pure factor shared with `_complete_core`. Original Core eligibility checks remain, plus exact current required-decision coverage: object/reason/content/availability, no missing or extra decision. Historical defects retain their exact content obligation. Unknown schema/reference kind, foreign namespace and incoherent owner/reservation origin cannot be acknowledged away. Views, user indexes and triggers refuse before original writes as well as in B2's full validator.
5. Optional complete B2 DDL uses the existing DDL in one original transaction only after predecessor checks. The production source projection is then computed from the ACTUAL post-DDL original, including every prior plan/history/reference and inventory. Plan/new lane/new epoch and recovery/archive IDs are each chosen once. The final proof-bound plan is INSERTed once with its final digest; no plan UPDATE/digest rewriting exists. `verify_abandon_plan` immediately self-verifies under the same locks using the typed journal mode, excluding ONLY the current exact proposal row and typed label. Later appends/deletion/content changes still stale it.
6. `session.capture` then backs up the ACTUAL committed original AFTER the plan, BEFORE current accepted intent/steps. The <=1 MiB snapshot contains the actual current plan and all prior rows, no current intent. K retains exact bytes/hash in the private owning `_PreparedSnapshot`, consumed at most once under that SAME session. The before/source projection excludes only the current plan; the unchanged actual archive oracle adds that exact plan back when checking actual snapshot tables. No shadow snapshot supplies the evidence hash and no later backup remints it.
7. Canonical before/proof/actions/output rows come from owned source. Terminal job tuples/context are unchanged. A nonterminal changes only prescribed failed/cancelled state and the latest attempt's proven reap bit. Original dispatch/commands/receipts/outcomes/ACKs/events/attempt history remain. Reservations become retained while demand/charges remain identical. Monitor/fault/integrity settlement and resolution SQL are the existing closed kernel contract; accepted accounting is exactly `[[1,H]]` with H unchanged. No production preparation high-water/budget UPDATE exists.
8. Pending/evidence (ACTUAL snapshot hash)/new-journal/Core outputs are built first. The manifest is constructed directly from its existing NONRECURSIVE source/before/proof/snapshot/journal fields, then hashed. No zero/placeholder digest or fixed point is used. Descriptor hashes are generated in order from before_digest; fifth output binds the fourth output digest, and the certificate remains bound to the fourth step. Intent digest is computed only from the final immutable description. No fixture builder supplies expected documents.
9. `validate_b2_record`, actual `_validate_archives` on protected copies, full database/history/aggregate validators, snapshot/manifest/control/receipt certificate bounds, fixed receipt namespace capacity and K's unchanged 64 KiB result-transport ceiling must pass before return/acceptance. The conservative receipt preflight reserves 18 entries and 18*64 KiB for the finite descriptors/replacement temporaries in addition to exact new documents and bundle, within original 2,048/16 MiB limits. Larger nested histories may therefore refuse even below the file-length limit; nothing evicts history or raises a ceiling.
10. `_rehearse_preparation` uses a disposable protected copy of the actual original DB/WAL/SHM physical prestate, actual retained archives, and the exact frozen evidence snapshot/manifest. All its SQLite connections explicitly select `vfs=unix`, WAL/FULL, shared extension/ATTACH/mmap/schema escape hardening and explicit close. It runs the concrete prescribed transactions through shared `_insert_recovery`, `_insert_recovery_step`, `_write_core_completion`, the unchanged `_recovery_scan`, and full semantic/history/archive validators through the fifth SQL receipt. This is a bounded capacity/semantic rehearsal under the same process cap, not another recovery writer, physical-growth proof or future-I/O guarantee. Original authority is untouched. Final trusted capsule and original self-verification repeat before returning.

The returned public-to-this-private-layer fields are exactly `{plan, plan_digest, intent, intent_digest}`. Internally `_Preparation` additionally retains its private session/snapshot references for later SAME-LOCK acceptance. The standalone child returns only those four documents and closes; it does not accept a receipt, fence, bundle or Core intent. Its returned documents alone are not durable restart authority. Tests instrument the internal owning seam to consume the prepared snapshot through K; no apply wiring or automatic receipt acceptance was added. A later initial invocation can only consume a recapture whose frozen hash and complete plan/source revalidate, otherwise it refuses. Accepted resume reads owned accepted document/bundle, never supplied expected documents or caller snapshot RAM.

Any late validation/capacity/transport refusal can leave the immutable unaccepted plan and optional schema installed. They are NEVER deleted or rewritten on failure. Early low-H/quota/free/configuration/unsupported-envelope/provenance/schema negatives precede original writer/plan/schema creation. No refusal grants healthy authority, adjusts charges, increases H/budget or restarts owners.

### Finite footprint and actual nonempty observations

ResourceStore, `_measure_recovery_roots`, `_maintenance_footprint` and the entire ordinary/recovery scan eligibility/arithmetic/equality remain unchanged from K. Reuse is exactly:

```text
L = 4,194,304; S <= 1,048,576; J,K <= 65,536
M = all existing measured st_size in control/core + staging
U_all_early = M + 3*L + 1,048,576 + 2*65,536
require U_all_early <= retained H; H + every retained demand.disk_bytes <= B
require real free reserve and five-entry headroom; repeat actual scan/reserve
expected highwater = H; actual scan requires max(actual,H) == H
```

Existing media, old archives and staging stay in M; no deletion credit or SQLite exclusion. Only the three capped original SQLite files grow, plus the flat exact snapshot/manifest before scan and flat certificate afterward. No counted descendant directory gains an entry and staging is untouched; counted directory delta is zero because the existing scanner excludes each root directory's own size. Receipts/bundle/descriptors/journal/replacement temporaries are outside these two measured roots and keep their independent limits. This is a bound on file **length/st_size**, not allocated blocks, host filesystem metadata or total physical disk consumption.

Nine decisive nonempty runs use a real Admission queued/allocation/cancel-requested population, retained H declared BEFORE preparation, unchanged B, the ACTUAL production trusted reader and production builder, all semantic/archive validators, then all five existing K kernel steps under the owning session. OS/mount/Linux-limit enforcement is explicitly E1-modeled, never installed qualification. No `b2_intent`, supplied expected documents, digest UPDATE or production H inflation occurs.

The corrected targeted log contains exact observations. For all nine: **H=20,971,520; C=2,097,152; B=67,108,864; K=437 bytes**, so H+C=23,068,672. Early measured M/U are before preparation; the later guard remeasures actual original files without changing H.

| Work / journal | Early M | Early U_all | Actual S | Actual J | Actual original scan | Measured after fifth/certificate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| queued / valid | 188,608 | 13,951,168 | 245,760 | 859 | 612,227 | 674,464 |
| allocated / valid | 192,704 | 13,955,264 | 249,856 | 859 | 628,659 | 699,136 |
| cancel-requested / valid | 196,800 | 13,959,360 | 249,856 | 859 | 632,755 | 703,232 |
| queued / missing | 188,608 | 13,951,168 | 245,760 | 551 | 611,919 | 674,156 |
| allocated / missing | 192,704 | 13,955,264 | 249,856 | 551 | 628,351 | 698,828 |
| cancel-requested / missing | 196,800 | 13,959,360 | 249,856 | 551 | 632,447 | 702,924 |
| queued / unreadable | 188,608 | 13,951,168 | 245,760 | 605 | 611,973 | 674,210 |
| allocated / unreadable | 192,704 | 13,955,264 | 249,856 | 605 | 628,405 | 698,882 |
| cancel-requested / unreadable | 196,800 | 13,959,360 | 249,856 | 605 | 632,501 | 702,978 |

Each asserts every exact output row, unchanged unaffected tables/charges/budget, one accepted recovery, one lane/epoch, exactly five steps and byte-identical snapshot/bundle/archive hash. The later conservative guard also passes: M=299,688/303,784/307,880 and U=14,062,248/14,066,344/14,070,440 respectively. The early fixed bound remains the source-derived invariant, not a fitted estimate from these successful observations.

### Fault, history, restart and unsupported-population matrix

- Early original-writable-open traps cover low H, H+C>B, real reserve failure, wrong enrolled configuration, executing-without-allocation origin, foreign namespace, unknown table/index/view, journal symlink/oversize, live/omitted owner, wrong purpose and unknown/oversize SQLite sidecar. Authority/schema/plans/receipt bytes remain unchanged; K retains unsupported platform/process/readback/filesystem/VFS/busy-lock checks.
- Known snapshot/manifest/control certificate/history/private capacity failures refuse before receipt acceptance; late tests retain one immutable unaccepted plan, zero accepted intent/steps and checking authority. Manifest uses a REAL 40,000-byte damaged journal to exceed its 64 KiB canonical bound. SQLite capacity and prospective-history failures are explicitly modeled E1 injections, not physical ENOSPC.
- Exact degraded-media decisions yield a production-generated valid intent and complete kernel. Missing/extra/content/reason/availability variants refuse. Actual terminal success/context is retained through two fresh startups, unavailable asset retrieval still fails, and exact accepted degradation does not relatch. Preparation alone leaves old R1/A2 checking unresolved on two ordinary restarts.
- Source/media/new-defect/journal/plan/snapshot tampering and later plan append refuse before new acceptance. Every prior B1 plan remains in before/source; deletion, fully rehashed content change or later append stales the proposal. Rehashing erased expected attempts through the entire expected-output chain still fails the canonical semantic oracle.
- Small two-completed-history positives run the production builder/kernel with valid, missing and unreadable later journals, full old/new owner registration/provenance union and all retained archives. Old actual archive or creation descriptor damage refuses before original writes. A second history embedding the larger nonempty queued first history exceeds K's unchanged 64 KiB returned-document cap: it honestly refuses BEFORE second acceptance, leaving its second unaccepted plan; the nine decisive first nonempty populations remain successful. No limit is increased to support it.
- Trace regression proves the final capsule proof digest precedes the SINGLE original plan INSERT and forbids original plan UPDATE or preparation H UPDATE. Actual snapshot contains that current plan, zero current accepted intents/steps; source excludes only its exact row/label. Actual-use auditing reaches every new private rehearsal/reader/backup connection, verifies Unix VFS/settings/escape refusal at first INSERT/read/backup, and proves explicit close.
- Production-GENERATED intent faults cover accepted-parent, bundle descriptor/output, Core-intent write/commit, pending/evidence/journal receipts, Core write/commit, complete fence and fifth COMMIT. Whole Core output is either original or atomically complete; no partial healthy authority. K burns the session after continuation and no live retry occurs.
- Two ACTUAL `os._exit(73)` deaths at **bundle-parent and Core-intent COMMIT**, using the production-generated intent, are followed by a FRESH interpreter receiving only roots/scope. It discovers owned accepted document/bundle and completes/replays TWICE with identical immutable bytes, one recovery and five steps. No fixture intent/caller expected document/snapshot RAM supplies authority. Fresh resume host/OS transport is explicitly modeled, separate from real owned durable authority.
- Actual preaccept preparation exit leaves only the unaccepted plan/schema and no marker/fence/intent/bundle. A fresh interpreter cannot treat that returned description as accepted restart authority; a physically changed recapture refuses the frozen hash without reminting. All apply remains unavailable even after successful preparation.

Unsupported populations include low retained H, overquota/free-reserve, unqualified Linux/ext4/Unix VFS/process/privilege/launcher model, missing trusted mount/full owner freeze, sidecars >L or unknown sidecars, document/snapshot/aggregate/history/64 KiB result ceilings, incoherent origin/unknown namespace/schema/reference, unresolved or changed diagnostics, media/source/history drift and private capacity failure. Future I/O/cap hits may still leave an already accepted recovery pending/quarantined. No progress guarantee, actual Linux enforcement or host-physical capacity claim follows from private rehearsal.

### Actual development failures, correction and frozen verification

All pytest commands use ONLY the existing `/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, disposable synthetic roots, no installs.

RED against baseline `9b45eac` production loaded **in memory in a separate interpreter**, new coherent queued fixture and actual expected-success preparation call: **1 failed, 760 deselected in 1.34s**, `/tmp/lotp-baseline-red2.log`, missing `_prepare_abandon` at that call. It is not an assertion that a function name is absent, a refused result accepted as green, or evidence that old `59c4a7` was imported. A prior RED failed earlier while fetching the missing locked helper (6.23s); the actual-call RED supersedes it. Checkout source was never substituted.

First evolving P matrix: **37 passed, 752 deselected in 24.42s**, `/tmp/lotp-development.log`. K/degraded preservation: **39 passed, 1 skipped, 829 deselected in 16.65s**, `/tmp/lotp-k-preservation.log`. Added degraded/preaccept/fault matrix first failed at test-only nonexistent `Core.asset_bytes`; corrected to actual strict `Core.asset`, then **19 passed, 789 deselected in 17.86s**, `/tmp/lotp-development3.log`. Prior B1 history/closed-shape subset: **14 passed, 808 deselected in 4.28s**, `/tmp/lotp-development4.log`.

Two-history development failures are retained: first a test provenance cgroup still contained the old container ID; fixed the fixture's exact association. The next run exposed the real prewriter archive-root bug: `check_recovery_open(db)` derived a private-copy directory lacking retained archives. Passing the existing explicit actual archive-root argument fixes it, with no persistence change. The next run completed its internal kernel but exceeded K's 64 KiB child transport result ceiling. Added the explicit returned-document preflight BEFORE any snapshot consumption/acceptance, preserved that ceiling and qualified the small-history positive separately from the larger-history refusal. Small-history matrix then **3 passed** (2.67s), expanded valid/missing/unreadable history matrix **9 passed** (7.01s). Rehearsal connection actual-use audit **1 passed, 831 deselected in 2.37s**, `/tmp/lotp-connections.log`.

Preparation-alone first failed because the test expected Admission's constructor to run a check; actual authority refusal is `Admission(...).check()`. Corrected the test call, not ResourceStore: **1 passed, 835 deselected in 2.27s**, `/tmp/lotp-alone2.log`. Final schema/index/view/capsule/alone subset: **19 passed, 821 deselected in 6.16s**, `/tmp/lotp-last-target.log`.

Initial combined P/K targeted receipt: **118 passed, 1 skipped, 796 deselected in 69.21s**, `/tmp/lotp-target-final.log` (before the last alone/schema/capsule additions). Source/tests were subsequently frozen as `9ae8e7c`. During its first focused run, source audit identified the physical capacity issue below. Verified task-owned pytest PID **71107**, parent **71104**, exact four-suite command AND this checkout cwd with `lsof`/`ps`, then SIGTERM'd ONLY it before editing source/tests. Wrapper receipt: **focused exit -15 after 57.97s**, partial progress with no displayed failure, `/tmp/lotp-focused-initial-interrupted.log` and receipt JSON. Full service/release had NOT started. This is not a preservation pass.

The initial rehearsal used the normalized immutable backup as its starting writable database. That preserved logical rows but erased the actual original WAL length, weakening predictable-capacity detection. Added a required-correctness regression capturing actual original DB/WAL/SHM at rehearsal entry and checking them at the FIRST private production INSERT. RED on `9ae8e7c`: **1 failed, 840 deselected in 2.15s**, `/tmp/lotp-wal-capacity-red.log`. Correction starts rehearsal from the existing protected source-preserving physical DB/WAL/SHM copy, retains original WAL bytes/length before writes and keeps the separately frozen snapshot/hash as evidence. It does not checkpoint/truncate original, remint evidence, change SQL/accounting, raise caps or restore a snapshot over authority. This concrete source correction justifies the corrected frozen campaign.

Final corrected targeted, source freeze, sequential focused/full results and final scope checks follow below. No independent GO is implied by any passing receipt.

### Qualification and unchanged closures

All evidence is E1 source/local SQLite/filesystem/process with explicit OS/host transport models. The real Linux RLIMIT_FSIZE case and 18 inherited Linux peer-credential cases remain skipped/unqualified on Darwin; **19 skips remain expected**, never installation evidence. C1/C2 must qualify actual kernel/build/standard Unix VFS/mmap/truncate behavior, ext4 mounts, UID/capabilities/hard-limit launcher and trusted host freeze. No root agent/pytest/host shell limit is lowered. No Docker, network, audio/model, paid or run-e2e-test operation, push/PR/deploy, prune or user cleanup ran; research and other worktrees are preserved.

Independent Sol-low review of the EXACT final candidate, root integration verification and final Astra remain required; **NO SELF-GO**. B2.3c, B3, C1/C2, installed/actual-host/Linux-cap evidence, physical ENOSPC/power-loss/daemon/VM/controller durability, clean-host, engines/audio/hardware and final global/public/unattended acceptance remain pending. **A1 global Docker-inventory drift remains unexplained and unqualified, with no cause or waiver.** All apply remains unconditionally unavailable and enrolled activation remains closed.

Corrected final combined targeted command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py -k 'lotp or lotk' -q -s
```

**124 passed, 1 skipped, 796 deselected in 72.16s**, exit 0, `/tmp/lotp-target-corrected-final.log`. Includes all **89 new P cases** plus 35 passing K cases and the skipped actual-Linux cap case. All nine numerical observations in the table were reobtained on the corrected source. The physical-prestate regression passes through the actual child/production builder, observing original DB/WAL bytes and SHM length at the FIRST private production INSERT. Final corrected frozen source/tests: **`bb6bd05154176ce7f2bd45f5d93a809d734b8a55`**, after initial `9ae8e7c`. Only this report changes during corrected frozen preservation.

A separate source-frozen production startup probe `/tmp/lotp-production-startup-probe.py`, run with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /tmp/pab-0c-venv/bin/python`, uses real queued/allocated setup, actual production preparation and its generated intent/snapshot. At actual Core-certificate-parent it interrupts with receipt pending: direct Core opens but worker Journal refuses TWICE. A separately fully production-completed population is deliberately changed to the unsupported opposite state by removing only fourth/fifth SQL receipts while retaining real receipt completion: Core refuses while Journal reads its valid receipt-side authority TWICE. No fixture intent/certificate or supplied expected rows. Final probe **exit 0**, `/tmp/lotp-production-startup-probe3.log`, both direction receipts printed. Initial probe attempts had fixture-only mistakes: unresolved macOS `/var` versus `/private/var` path equality, then nonexistent `Journal.close` instead of its actual `journal.lock.close`; neither is a production defect or passing receipt. Probe work briefly overlapped focused preservation on separate disposable roots; source/tests stayed frozen.

Corrected frozen preservation commands execute sequentially, guarded by exact HEAD and empty source/test diff before each:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q -rs
```

Results follow after completion; logs `/tmp/lotp-focused-corrected-frozen.log`, `/tmp/lotp-full-corrected-frozen.log`, with command/SHA/exit/wall-time JSON receipts alongside. No further unchanged-source broad campaign is planned.

Corrected frozen focused result: **974 passed, 1 skipped in 642.04s (10:42), exit 0**, `/tmp/lotp-focused-corrected-frozen.log`, on unchanged **`bb6bd05154176ce7f2bd45f5d93a809d734b8a55`**. Wrapper wall time 642.51s. Includes all 89 new P cases and inherited K/B1/B2/R1/F2/W2/host/startup/history preservation. Sole skip is actual Linux RLIMIT_FSIZE, explicitly PENDING. Full service/release starts sequentially afterward on exactly the same frozen source/tests; only this report changes.

Executable final scope audit confirms ResourceStore and admission tests byte-identical to `9b45eac`; original recovery tests and this report remain exact prefixes; public apply's function source is byte-identical; protected-path diff excluding the three actual changed paths is empty. `git diff --check` passes. These scope checks do not supply independent review or installed qualification.

Corrected frozen FULL service/release result: **1,126 passed, 19 skipped in 671.67s (11:11), exit 0**, `/tmp/lotp-full-corrected-frozen.log`, wrapper wall time 672.43s. Sequential after the 974-pass focused suite on the SAME unchanged **`bb6bd05154176ce7f2bd45f5d93a809d734b8a55`**. Skips remain exactly one actual Linux RLIMIT_FSIZE case plus 18 inherited Linux peer-credential cases (10 authenticated exchanges, 1 forged control proof, 1 replayed hello, 6 wrong execute). All remain unqualified. Neither corrected frozen suite failed; no source/tests changed during or between them, and no redundant broad repeat followed.

Final `git diff --check` passes. Source/tests-to-frozen comparison is empty. The complete baseline-to-candidate path set is EXACTLY `service/recovery.py`, `tests/service/test_recovery.py`, and this append-only report, within the assigned five-path allowlist. ResourceStore/measurement/calculation/scan equality and admission tests are byte-identical to baseline; original recovery tests/report are retained as exact prefixes; public apply is unchanged; every protected path is unchanged. Local sequence: initial implementation/source freeze `9ae8e7cebd764c9ff81be2d7849bc9ec2a634420` → actual-WAL rehearsal correction/final frozen source `bb6bd05154176ce7f2bd45f5d93a809d734b8a55` → report-only handoff commit, whose exact SHA accompanies the final handoff. Working tree is checked after that commit. No other checkout/research/user data is changed.

Ready for parent-owned independent Sol-low review of the exact candidate and root integration verification; **NO SELF-GO**. The nonempty production preparation/intent/kernel call chain is demonstrated under E1 K assumptions, not installed Linux/host or overall B2/P1 acceptance. B2.3c, B3, C1/C2, actual Linux/host qualification, final Astra and public/unattended acceptance remain pending. A1 inventory drift remains unexplained/unqualified without cause or waiver. Every apply remains unconditionally unavailable and enrolled activation stays closed.

## K pre-bundle exact-recapture correction — implementation candidate, no self-GO

2026-10-17, `gpt-6.1-sol` / low, implementation sub-agent. Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-ef74844b`, branch `t3/k-prebundle-correction`. Clean actual baseline **`3454c4c3522dc5052028964fd3dc60a41357a571`** contains integrated P `4210c8e`; `git diff 4210c8e 3454c4c -- service worker scripts tests` is empty. Frozen source/tests **`2a5e5e06b86aa275cc58e851f9d25632f445eab8`**. Only recovery.py, append-only recovery regressions and this append-only report change. No applicable ancestor/scoped AGENTS.md was found. Memory registry search supplied no relevant K/P/C authority or acceptance evidence.

Read the authoritative FULL P1B23C-ACCEPTANCE-ASTRA predicates/interfaces/F1–F4/matrix/ready brief/C stack rules, ACCOUNTING-ASTRA, KERNEL-ASTRA, BOUNDARY-ASTRA, K R1/R2, P R1, orchestration protocol and RECOVERY-P1B2 contracts/correction history/K/P handoff. C's dependency/full-matrix report from line 1349 was inspected READONLY at `/Users/ben/.txcode/worktrees/py-audio-box/t3-4a8727d4`, exact HEAD **`6cfe3973f41db92679531e38660d88cf72ce27eb`**, containing WIP source `1fc069f`. No C source was imported, edited or integrated. Its blocked regression remains C evidence; this candidate reproduces the seam through existing P/K entrypoints only.

### Closed call chain and invariants

Initial P preparation and four-field result are unchanged. `_prepare_abandon_locked` commits the final plan, captures actual original committed DB/WAL-aware evidence once through the owning session, and provides its same-session `_PreparedSnapshot`. Initial consumption is still exact-object/session/plan/hash/once-only. No snapshot, digest, plan, proof, ID, epoch or lane is reminted.

Fresh continuation uses `_run_bounded_maintenance` → fixed process envelope → `_maintenance_session` under control then lane lock → `session.continue_kernel` → `_pending_prefix_locked(..., _session=session)`. The optional argument is internal nonserialized lifetime plumbing. The kernel requires exact `_MaintenanceSession` type, live verified bounded process, the identical DB and roots objects and the same scope. Legacy supplied-document helpers omit it; the bounded path explicitly refuses rather than falling into their larger-limit capture helper.

`continue_kernel` checks owned canonical acceptance but no longer eagerly reads a bundle. Snapshot selection occurs INSIDE `_pending_prefix_locked`, after common full plan/before/source/history/namespace validation, before any new fence/descriptor/Core output. `_no_attempt_prefix` then requires:

- No current accepted/step/resolution/retirement/diagnostic SQL, no transaction, every earlier recovery fifth-completed, and no current control archive.
- BOTH bundle entries absent using `lstat` (`_entry_present`), not target-following `exists`; unsafe/dangling/inaccessible entries do not count as absence. The already bounded complete receipt namespace rejects unknown names and temporaries; every current later-stage spelling is independently rejected by the classifier, even though ordinary prefix validation recognizes that spelling.
- Exact protected canonical accepted bytes and required marker. On first history, null prior certificate and absent or exact pending fence; with history, actual latest fifth certificate and exact preceding complete or current pending fence. An absent predecessor fence refuses.
- Exact original journal bytes/status, including exact absence. The small shared `_original_journal_exact` also corrects receipt-only successor replay for a declared missing journal; it does not demand a future journal or reinterpret absence as clean data.

The common validators still ground the full current schema/logical before-image, ONLY-current-plan exclusion, source digest, references, affected set, inventory/media, enrollment/keys/volume identities and every prior archive/descriptor. Fresh OS-clock trusted capsule/original proof/registration union/freeze remain existing K prerequisites; C's future pinned host check is not fabricated here. Accepted TTL expiry is allowed; transport freshness is separate.

Only after classification does the owning session reuse `capture`'s unchanged plan/order/file/retained-H/all-charges/budget/footprint guards and `_capture_original_snapshot(..., MAINTENANCE_SNAPSHOT_LIMIT)` once. The actual protected original DB/WAL copy, Unix VFS, all connection hardening/closes, time/population bounds, 1 MiB snapshot ceiling and 4 MiB soft AND hard child envelope are unchanged. Parent/pytest/shell limits are never lowered.

SHA-256 MUST equal the frozen `expected[1].output.snapshot_digest` before any new protocol output. Classified no-attempt resumes explicitly execute the existing actual snapshot-plan/source archive oracle AND prospective prefix SQL/history validation through shared SQL writers, with complete prior archives and aggregate bounds. Hash equality alone never skips these validators. Exact internal bytes then follow the original fence → bundle descriptor → bundle → Core acceptance sequence. A descriptor/output or any downstream attempt/SQL precludes recapture; missing/partial/wrong attempted evidence remains retained quarantine. No repair/deletion/cleanup path exists. The entire continuation, including early validation/classification/capture errors, burns the session; no live retry.

New deterministic fault points are `recapture-before`, `recapture-after`, `recapture-hash`, `recapture-oracle` and `recapture-verified`. They bracket actual bounded capture, exact-hash comparison, actual semantic oracle and prospective validation before new protocol outputs. Actual process-death regressions record hook execution outside authority roots, discard caller RAM and exec fresh interpreters with owned roots/scope/optional recovery identity only.

### Correction and preservation matrix

The 237 appended `kpre` cases exercise production P-generated acceptance, not fixture expected documents or C wiring. Existing tests remain an exact byte prefix.

| Retained stage / condition | Candidate regressions and required result |
| --- | --- |
| Actual accepted write/fsync/parent; first fence absent | Fresh bounded exact capture, fixed hash/plan/proof/identity, complete twice |
| Actual first pending-fence write/fsync/parent | Same exact no-attempt continuation; no new identity |
| Small completed predecessor; accepted write/fsync/parent or pending replacement completed | Completed A replay twice with capture/source/resource scans and new file/SQL writers forbidden; independent B resume completes twice |
| All ten admitted first/prior-history windows × valid/missing/bounded-unreadable journal | Positive matrix preserves original status/bytes; accepted-after-TTL covered on first-history rows |
| Each of those windows × each journal state × source drift / unavailable fresh capsule / busy control lock / busy lane lock | Two fresh refusals, no capture and no new receipt/protocol bytes |
| Pending replacement temporary at create/write/fsync/parent | Both A replay and B resume quarantine twice; no finish or cleanup |
| Actual death before/after capture, after hash, after archive oracle, after prospective validation | Hook asserted; no new persistent protocol artifact; next fresh attempt completes/replays with fixed original inputs |
| Bundle descriptor create/write/fsync/parent/recorded; empty output create | Recorded attempt quarantines, never recaptures |
| Exact bundle write/fsync/parent; Core BEGIN/write/COMMIT/parent; pending/evidence/journal/Core/fifth COMMIT | Exact owned bundle only, no capture; continue/replay twice |
| Bundle descriptor AND output deleted after actual current SQL or completion | Refuse twice, no recapture or remint |
| Each later descriptor; unrecorded bundle/output; dangling descriptor/bundle; mode/hardlink/oversize/extra/temp; accepted/marker/fence/plan/source/schema/media/journal/proof/key drift; multiple candidates; premature control archive | Two fresh refusals, no new receipt/protocol bytes and no caller repair |
| Old bundle/archive/Core creation/final creation descriptor damage; missing predecessor fence | Refuse twice before capture; all completed history checks retained |
| Physically different, logically equal SQLite header backup | Frozen hash mismatch refuses twice before new outputs; frozen hash is never reassigned |
| Capture/copy/read I/O, actual snapshot ceiling, modeled VFS/readback/cap/host/budget failure; exact-hash oracle refusal; prospective SQL refusal | No new receipt/protocol output, session burned whenever created; fresh invocation only |
| Actual capture backup/read/first private INSERT | Explicit Unix URI, modeled installed cap, hardening/escape authorizer at actual use, one backup/one prospective insert, explicit closes audited |
| F1–F4, initial nine P positives, startup half-states, R1/A2 inherited uncertainty, all older histories | Existing K/P/B2 targeted and frozen four-B2/full-service-release suites below; no history omission or future-output waiver |

F1 registration coverage before unfinished final writes and separate completed no-op remain unchanged. F2 completed replay stays observational and does not capture/scan/finish receipt-only B. F3 predecessor creation/SQL prerequisites remain unchanged; a missing OWN receipt differs from a missing predecessor. F4 still requires EVERY completed history's bundle/evidence/rotation/Core certificate AND creation descriptor plus all final descriptors. No requested-ID-only exclusion was added.

### Actual development receipts and frozen verification

All commands use the existing `/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, disposable synthetic roots and explicit E1 OS/host models; no installs.

Initial baseline RED with actual production P deaths at accepted-parent and fence-parent: **2 failed, 841 deselected in 4.22s**, `/tmp/kpre-red.log`. Stronger exact-baseline RED rerun loads `git show 3454c4c:service/recovery.py` IN MEMORY in both the test interpreter and exec'd fresh resume interpreter, without replacing checkout files: **2 failed, 1076 deselected in 5.28s**, `/tmp/kpre-baseline-hook-red.log`. Both death hooks are explicitly asserted, canonical accepted/marker and zero current Core rows/bundle attempts checked; the expected successful resume fails with the actual missing `.snapshot.bundle` traceback. This is behavioral RED, not a fixture/setup failure. It briefly overlaps final targeted tests on separate disposable roots; source/tests stay frozen.

First correction GREEN: **2 passed, 841 deselected in 5.00s**, `/tmp/kpre-green2.log`. One earlier edit command used unavailable `python` and made no source change; its subsequent rerun remained baseline RED, not a corrected-source result.

Evolving matrix `/tmp/kpre-dev-matrix.log`: **2 failed, 54 passed, 841 deselected in 90.55s**. A hardlink was rejected by the test's protected setup reader before entering the assertion; the accepted-tamper fixture varied unbound creation metadata instead of a plan-bound field. Corrected only the setup observation and single-field bound-source tamper. Initial history harness **17 failed, 897 deselected in 27.26s**, `/tmp/kpre-history-dev.log`, because the fresh script still asserted one retained accepted file. Corrected target selection and current registration-union setup; **19 passed, 895 deselected in 35.78s**, `/tmp/kpre-history-dev2.log`. An overly broad edit briefly touched the old P test script; restored the ENTIRE baseline test prefix byte-for-byte before freezing.

Attempt matrix: **22 passed, 1 failed, 914 deselected in 40.22s**, `/tmp/kpre-attempt-dev.log`; sole failure was the new audit omitting its prewriter registration SELECT observation. Added that actual-use hook; capture audit plus all five capture-death boundaries **6 passed, 933 deselected in 13.55s**, `/tmp/kpre-audit-dev2.log`. Extra closed-input/lock matrix **10 passed, 939 deselected in 14.79s**, `/tmp/kpre-extra-dev.log`. Pending replacement matrix **4 passed, 954 deselected in 7.73s**, `/tmp/kpre-temporary-dev.log`.

Failure-matrix editing first produced an unterminated test string (collection exit 2, 0.25s), `/tmp/kpre-failures-dev2.log`; corrected before execution. Next **8 passed, 2 failed, 944 deselected in 16.06s**, `/tmp/kpre-failures-dev3.log`: harness did not classify SQLite's OperationalError as an expected refusal and the oracle injection fired at the initial empty-history validator. Corrected the harness exception type and restricted the exact-hash oracle injection to actual snapshot evidence. Same two corrected cases **2 passed, 956 deselected in 4.51s**, `/tmp/kpre-failures-dev4.log`. These evolving-source receipts are not frozen preservation passes.

Frozen sequential commands, guarded by empty diff against `2a5e5e06b86aa275cc58e851f9d25632f445eab8` across service/worker/scripts/tests before EACH command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py -k 'kpre or lotp or lotk or b22d' -q -rs
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q -rs
```

Results append below after completion. Logs `/tmp/kpre-{target,focused,full}-frozen.log`; adjacent JSON receipts record exact command, frozen SHA, exit and wall time. No source/test edit during these campaigns; only this report changes. No redundant unchanged-source broad campaign is planned.

### Qualification, review and C stack handoff

All current evidence is source/local E1 SQLite/filesystem/process with explicit Linux/host transport models. Actual Linux 4 MiB enforcement/ext4/Unix-VFS/SHM behavior and actual pinned host authentication remain UNQUALIFIED; the inherited 19 platform skips remain skips, not qualifications. Parent/pytest/shell limits remain unchanged. No Docker/network/other-user-proc/mount/model/audio/paid/E2E runner/install/push/PR/deploy/prune/user cleanup or other-worktree edit. Research and C WIP are preserved.

All apply paths remain unconditionally unavailable, enrolled activation closed. This correction does not implement C, B3, C1/C2, public/unattended operation or freeze release. C's real pinned launcher/host reader remains mandatory future work, with its production refusal retained in the unintegrated WIP; no Boolean/hash/environment bypass is supplied. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no cause or waiver. Physical ENOSPC/power-loss/daemon/VM/controller durability, installed/clean-host operation and engines/audio/hardware remain unqualified.

Fresh independent Sol-low review of the EXACT candidate, parent-owned root integration/source verification and focused integrated restart rerun, then final Astra remain required; **NO SELF-GO**. No subagents were launched, as explicitly instructed. After reviewed K integration, C must start from that fresh integrated SHA and replay its own stack in order in an isolated copy, preserving original `6cfe3973` and its RED. C must rerun its original accepted-parent regression through the shipped boundary and complete every pending C matrix row before its own final campaigns/review. This candidate grants no C integration or host acceptance.

### Frozen targeted preservation failure and narrow exception compatibility correction

Initial frozen targeted run on `2a5e5e06b86aa275cc58e851f9d25632f445eab8`: **544 passed, 1 failed, 1 skipped, 612 deselected in 714.15s**, exit 1, `/tmp/kpre-target-frozen.log`; wrapper 714.62s. Sole failure: existing `test_lotk_io_failure_burns_session_no_partial_healthy_authority[bundle-intent-recorded]`. The bounded session formerly read its missing bundle eagerly and raised FileNotFoundError. Deferred selection reached the kernel's existing explicit ValueError quarantine instead. Both refused safely, but preserving the existing bounded seam's exception behavior is required. No test expectation was changed or waived. ALL 237 new correction cases and inherited b22d history cases passed in that run. The queued four-B2/full suites did NOT start after exit 1.

Correction commit and FINAL frozen source/tests: **`eb2ef00a1c494b1e2737377a9e2429e9c4f56f0d`**. Its only delta is recovery.py: after exact descriptor validation, the bounded route lets its protected <=1 MiB read raise the original FileNotFoundError for absent output; legacy supplied-document helpers retain their explicit ValueError. The descriptor still forbids no-attempt classification/recapture, and no fence/descriptor/Core write is added. Session failure still burns. This changes no durable record, name, order, cap, source/history validation or allowed state.

Corrected frozen sequence uses the same interpreter/environment and source-diff guard. Targeted command is now `tests/service/test_recovery.py tests/service/test_admission.py -k 'kpre or lotp or lotk' -q -rs`, covering every new case and all K/P cases; the unchanged inherited b22d cases also run in BOTH final broad suites. Four-B2 and full-service/release commands are exactly those above. Logs and JSON receipts are `/tmp/kpre-{target,focused,full}-corrected-frozen.{log,json}`. This actual observed preservation failure and source correction justify the corrected frozen campaign; no unchanged-source broad repeat occurred. Only this report changes during it.

### Pre-capture historical identity predicate correction and final freeze

During the corrected targeted run, source audit identified a missing early predicate: the existing shared SQL oracle rejects reuse of ANY retained lane/epoch identity, but `_no_attempt_prefix` had not checked older/cross-kind identities before backup. Canonical plan validation rejects new==immediate-prior within each field; that does not reject a new epoch equal to an older lane (or converse). This would still refuse before protocol writes, but capture itself must follow ALL no-attempt predicates. No claim of the earlier run waives that ordering requirement.

Before editing, identified ONLY this task's pytest **PID 54114**, parent runner **54106**, via the exact `/tmp/kpre-target-corrected-frozen.log` open handle, verified exact pytest command AND this checkout cwd, then sent SIGINT to pytest. Runner collected **exit 2 after 250.47s**, test receipt **234 passed, 1 skipped, 796 deselected in 250.02s**, KeyboardInterrupt, `/tmp/kpre-target-corrected-frozen.log`. No displayed test failure. No broad suite had started. Confirmed the task log had no remaining open handles before source edits. This interrupted run is NOT a preservation pass. No unrelated process was inspected/stopped.

Added two independent fully rehashed closed plan/accepted-record negatives for new epoch colliding with retained lane and new lane colliding with retained epoch. They preserve the original snapshot hash and request no repair. Required-correctness RED on `eb2ef00`: **2 failed, 1078 deselected in 8.14s**, `/tmp/kpre-identity-red.log`; both explicitly reached `AssertionError: identity collision captured`. Correction adds the SAME retained lane/epoch-set check already used by receipt-only discovery, now in the no-attempt classifier BEFORE `session.capture`. Recovery/archive identity collision is already rejected through existing current-record comparison and required retained control archives/full namespace; no new identity or schema is generated.

Strengthened the twelve production two-history replay/resume positives to execute after accepted TTL as well; capture-failure tests now explicitly compare complete logical SQL before/after, in addition to all receipt bytes and burned-session checks. Targeted identity/TTL/logical-SQL matrix: **24 passed, 1056 deselected in 42.83s**, `/tmp/kpre-identity-ttl-sql-green.log`. All source changes remain confined to the selected predicate. The final appended correction population is **239 `kpre` cases** (237 earlier plus two identity negatives); original tests remain an exact byte prefix.

**FINAL frozen source/tests are `a21e4b3a51bd381d4214f073cea35b031062c7f1`**, superseding both earlier freezes. The final sequence is the corrected K/P command (`-k 'kpre or lotp or lotk'`), then all four B2 suites, then full service/release, sequential with exact frozen-source guards before each. Commands otherwise remain identical. Final logs/receipts: `/tmp/kpre-{target,focused,full}-final-frozen.{log,json}`. Only this append-only report changes during them. The concrete RED/required predicate correction justifies this final campaign; there was no completed broad campaign to repeat.

Protocol qualification: `accepted-parent` precedes the NEW pending fence. On first history the exact required marker without a fence blocks worker boot; with a completed predecessor, the old complete fence can still authorize worker boot until replacement. The authenticated host freeze therefore remains mandatory across this whole window and through host-observed fifth completion. No worker-side pre-fence guarantee, automatic restart, freeze-release authority or cross-volume atomicity is added.

FINAL frozen targeted receipt on unchanged **`a21e4b3a51bd381d4214f073cea35b031062c7f1`**: **363 passed, 1 skipped, 796 deselected in 439.10s (7:19)**, exit 0, `/tmp/kpre-target-final-frozen.log`; wrapper 439.55s. Includes all 239 correction cases, all 89 P cases and 35 passing K cases. The actual Linux RLIMIT_FSIZE row remains skipped/unqualified. Four-B2 preservation started sequentially afterward on the SAME frozen source/tests; full service/release follows it. No source/tests change during or between these runs; only this report changes.

FINAL frozen four-B2 preservation: **1,213 passed, 1 skipped in 992.37s (16:32)**, exit 0, `/tmp/kpre-focused-final-frozen.log`; wrapper 992.85s. Sequential after targeted verification on the SAME unchanged **`a21e4b3a51bd381d4214f073cea35b031062c7f1`**. This includes all inherited F1–F4/history/startup/R1/A2 and nine production P positives, plus every appended correction case. Sole skip remains actual Linux RLIMIT_FSIZE, UNQUALIFIED. Full service/release started sequentially afterward, with the source/test guard passing again. No source/test edit or failed final frozen targeted/focused test; only this report changes.

FINAL frozen FULL service/release: **1,365 passed, 19 skipped in 1,039.03s (17:19)**, exit 0, `/tmp/kpre-full-final-frozen.log`; wrapper 1,039.84s. Sequential after the 1,213-pass four-B2 suite on the SAME unchanged **`a21e4b3a51bd381d4214f073cea35b031062c7f1`**. Skips are exactly one actual Linux RLIMIT_FSIZE case plus 18 inherited Linux peer-credential cases (10 authenticated exchanges, one forged-control-proof, one replayed-hello, six wrong-execute). ALL remain UNQUALIFIED. No final frozen targeted/focused/full test failed. No source/tests changed during or between them, and no redundant broad campaign followed.

Final executable scope audit passes: baseline-to-candidate paths are EXACTLY recovery.py, recovery tests and this report; original tests and report remain exact byte prefixes; source/tests equal final frozen `a21e4b3`; every protected path is unchanged; capture and its existing guards, all process/connection setup, P preparation/rehearsal, F3/F4 retained validators and public apply are byte-identical function segments to baseline. `git diff --check` passes. C remains at exact **`6cfe3973f41db92679531e38660d88cf72ce27eb`**. Before the report commit, status contains ONLY this report. No C import/edit/integration or other-worktree/research/user-data mutation.

Local recoverable sequence from baseline `3454c4c3522dc5052028964fd3dc60a41357a571`: initial correction/source tests **`2a5e5e06b86aa275cc58e851f9d25632f445eab8`** → missing-output exception compatibility **`eb2ef00a1c494b1e2737377a9e2429e9c4f56f0d`** → historical-identity predicate/final frozen source tests **`a21e4b3a51bd381d4214f073cea35b031062c7f1`** → report-only handoff commit, whose exact SHA accompanies the final response. No amend/reset/squash/prune or push. Final call-chain anchors are `_MaintenanceSession.continue_kernel`, `_pending_prefix_locked` and `_no_attempt_prefix`; `_original_journal_exact` is the sole shared journal-status factor. Current source locations are recovery.py:341, :1763, :1672 and :1711 respectively.

Ready for parent-owned fresh independent Sol-low review of the exact final candidate, root integration/source equality and meaningful integrated restart verification, then final Astra. **NO SELF-GO.** B2.3c/C WIP, B3, C1/C2, actual Linux/host/pinned-launch qualification and global/public/unattended acceptance remain pending. A1 inventory drift remains unexplained/unqualified, with no cause or waiver. All apply remains unavailable and enrolled activation closed; no automatic restart/freeze release is granted.

## K pre-fifth F1 correction — resumed implementation handoff (2026-10-22)

Implementation candidate only; NO SELF-ACCEPTANCE. Parent owns the separate Sol-low adversarial review, integration, the original four C RED reruns on a fresh C replay, and final Astra. This section supersedes no prior review and grants no C/host/release/activation qualification.

### Provenance, ownership and preservation

Fresh worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-85d4bb33`, branch `t3/fix-k-prefifth-validation`, clean entry baseline `7916f17ebe4608833269d958cd1c2b1430d78400`. Integrated K/P/prebundle executable source equals reviewed `a5bdf7ba`; the newer parent Apple Silicon documentation is preserved. No applicable ancestor or scoped AGENTS.md was found. Read canonical protocol/ledger/plan, the full decisive P1B23C-PREFIFTH-r1-sol correction/exit matrix, KERNEL-ASTRA, C-ACCEPTANCE-ASTRA and K-PREBUNDLE-r1-sol. Implementation is Sol `gpt-6.1-sol`, low. No delegation or independent review was performed by this implementer.

Audited and replayed ONLY the two recovered partial commits, oldest first: `5351324452ae582461b575ba459681e588acc2b6` became `799eab2`; `7d4370b5ac3ca22ae5234a12f8eb142ad48c62b1` became `08b7e20`. Their prior tests/validity were not accepted on trust. Additional source/tests `f5b4f171d6b5fd4866e3603f7da374dd0cb6dfd2`; final test coverage `44e00b94d4f40ce7a058e3f84c26d571f7feb1f2` is the FINAL FROZEN SOURCE/TEST SHA. The final report-only candidate SHA accompanies the response. All final campaigns below use that frozen source without source/test edits during or between them.

Exact writable delta from baseline: `service/recovery.py`, append-only `tests/service/test_recovery.py`, and this append-only handoff. The original test/report bytes remain exact prefixes. Existing definitions changed: ONLY `_pending_prefix_locked`; new definition: ONLY `_unfinished_source`. AST function-source comparison verifies every other existing function/class is byte-identical, including snapshot/output oracles, retained history/stage validators, process/cap/connection/session/capture code and P preparation/rehearsal. No Core/persistence/Journal/resource/reconciliation/schema/contracts/dependencies/packaging/C/host-script/Dispatcher/supervisor/ledger/review change. No C or old-base implementation import into the candidate.

Preserved worktrees, verified clean and unchanged: `t3-38916671` at `00c50e0880488700342af1c225a860e4f984e47f`; `k-prefifth-validation` at `7d4370b5ac3ca22ae5234a12f8eb142ad48c62b1`; C `t3-adbf212d` at `58e3eb9f40a8db5ca162fe8f64922efb13f224c4`; original C `t3-4a8727d4` at `6cfe3973f41db92679531e38660d88cf72ce27eb`. No reset/clean/prune/amend/squash, broad add, research removal, push/PR/deploy, install, Docker, external network, model download/run, real-audio/listening/paid script or legacy E2E runner. Disposable test roots, harness and receipts under `/tmp` only.

### Owning invariant and call chain

Fresh production-P acceptance uses the unchanged `_prepare_abandon` → bounded child/session → `_prepare_abandon_locked` → once-owned prepared snapshot → `continue_kernel` → `_pending_prefix_locked`. Restart execs a fresh interpreter with roots/scope/recovery identity and explicit E1 OS transport models; accepted intent/bundle remain protected owned files, never caller snapshot/expected-row RAM. `_kpre_attempt` asserts actual `os._exit(73)` hook execution. No new authorization Boolean, production bypass, proof/hash repair, recapture, epoch or plan is introduced.

Source anchors: `_unfinished_source` at recovery.py:1763; `_pending_prefix_locked` at :1782; certified validation at :1828; explicit fourth-only read at :1830; shared uncertified validation at :1902.

For any certified Core recovery still lacking fifth, `through_receipt` continuation now calls the shared read-only `_unfinished_source` BEFORE `_core_certificate`, `_finish_receipts`, any descriptor/artifact replacement or fifth transaction. Fourth without a certificate follows the existing uncertified path through the SAME helper before its certificate creation. The helper first rejects retained finalization temporaries and registration drift, then invokes `_retained_pending_receipts` at the actual SQL stages. It validates actual attempted Core/receipt certificates and completion-bound journal/fence targets BEFORE substituting the accepted original journal for semantic reconstruction. A legitimately bound journal before fifth is accepted; missing/partial/wrong attempted output quarantines, never remints.

The shared snapshot call remains exactly `exclude_abandon=plan['id'], transaction=False, _prefix=intent, _journal_after=journal_after, _core_after=core_after`. The existing `_verify_core_rows` checks ALL jobs/reservations/monitor/fault/accounting outputs, exact healthy p1 integrity and all three exact resolution sets BEFORE projection. Existing snapshot/coherence/consumer/affected checks and complete `evidence_projection == intent.before.source` comparison preserve all unaffected SQL/schema/population/history/older plans, typed references, media/inventory, installation/keys/volume identities and aggregate bounds. Only this recovery's prescribed output rows/labels and current plan are projected away. No heal, reservation deletion, charge reset, ordinary resource scan or budget change.

Audited `7d4370b` fourth-only compatibility: the no-current-reconstruction exception is now an explicit READ-ONLY return. Both existing Core-certificate creation descriptor AND output must verify using the existing protected validators; no creation helper is called. Observational here means no file-content/namespace or SQL mutation: the unchanged `_completed_prefix_receipts`/`_prefix_exact` validators still fsync/re-barrier exact existing files, as on baseline historical replay. Those checks cannot create a missing descriptor/output or mint authority. Missing either refuses before any writer. If both are absent, the ordinary unfinished continuation must pass full current validation before creating them. This exception cannot finalize, insert fifth, reobserve, scan, capture or repair.

A genuine fifth-completed history skips `_unfinished_source`. Existing immutable receipts and every completed history's creation descriptors remain mandatory in both history directions. `_receipt_only_successor`, F2 actual-prefix validation, F3 predecessor-vs-own-receipt distinction and F4 all-history validation are unchanged. Completed A replay neither resumes B nor demands B's future outputs. Explicit unfinished B receives full validation before writes. Both locks, fixed Unix VFS, process limits, connection hardening, session burn, accepted TTL semantics and frozen identities remain unchanged.

### Behavioral matrix and baseline RED

164 appended correction cases use actual production P-generated positives, not fixture intents/expected rows:

- Two actual-death monitor RED setups: nonempty running allocation, death at `completion-fence-dirsync`, fourth/certificates/bound journal/complete fence present, fifth absent. Change ONLY monitor healthy→fault. The existing exact oracle rejects `b2_core_output_changed`; two fresh direct-K refusals forbid protocol writers and preserve logical SQL, receipt namespace/bytes and fault. No C copy/routing is needed.
- 12 reached fourth/finalization durable prefixes × unchanged/monitor drift: fourth COMMIT/parent, Core-certificate attempt/output, receipt-certificate attempt/output, completion-journal attempt/replace/dirsync and completion-fence attempt/replace/dirsync. Valid outputs finish once and replay; intent-only attempts with missing/old targets quarantine. Accepted-after-TTL positives retain original plan/intent/proof/hash.
- 16 source/output drift kinds × certified/final-fence prefixes: each output family, healthy integrity, each resolution family, unaffected SQL sequence, typed references, event history, media, key, installation and volume. Writers forbidden, two fresh refusals, unchanged retained state. Existing targeted B2/K/P tests additionally cover schema/old plans/history, original consumers, exact affected sets, no-attempt/hash and caps/connections.
- 23 actual production successor prefixes: completed A remains observational at B's actual receipt-only/SQL/finalization stage, then explicit B completes or quarantines as prescribed, then both history directions replay. All nine retained creation-descriptor kinds × missing/wrong/mode × older/newer direction (54 cases), no future evidence/scan/capture/write.
- Four attempted finalization output kinds × missing/partial/wrong/descriptor/temporary (20 cases), strict pre-writer quarantine.
- Completed current-state mutation replay plus actual valid worker hello/new registration/resource-monitor/highwater change replay; fourth-only observational query plus missing descriptor/output/both refusal; production complete-fence/pre-fifth same-container hello, new registration and different-container refusal. The latter retain the ORIGINAL capsule/proof: the owning session refuses changed registration coverage before opening the original writable DB, rather than forging a new proof to reach K.

Baseline diagnostic loaded root `7916f17` recovery source ONLY in the disposable fresh exec interpreter; it was never imported into candidate code or copied from C. Current production P creates the identical retained pre-fifth prefix. With writer traps removed solely in this baseline diagnostic, both required-refusal tests fail; independent read-only SQL inspection confirms monitor `fault` AND all five steps. Actual death trace and pre-resume oracle rejection are asserted. `/tmp/kfifth-baseline-red.log`: **2 expected RED failures**, exit 1, 5.36s; `/tmp/kfifth-baseline-red-proof.json` records the two distinct disposable roots/states. The `current` pytest symlink is excluded from the two-root count. This is reproduced wrong-fifth behavior, not a setup failure or acceptance.

### Environment and honest execution history

The specified `/tmp/pab-0c-venv/bin/python` is ABSENT. Its attempted command `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k kfifth -q -rs` exited 127 (`/tmp/kfifth-initial.log`). The same pytest arguments with `python3` (3.14) then failed collection for missing jsonschema (`/tmp/kfifth-initial-system.log`). No environment was installed/rebuilt and no dependency/repository file changed. Thus the EXACT PRESCRIBED-INTERPRETER RECEIPT REMAINS UNMET; the alternative environment below is explicitly different, not a silently substituted pinned receipt. Parent must restore/requalify that environment and rerun the required commands before treating the prescribed-environment gate as met.

Existing alternative interpreter: `/Users/ben/.hermes/hermes-agent/venv/bin/python`, CPython 3.11.16; existing pytest 9.0.2 from `/usr/local/lib/python3.14/site-packages`; jsonschema 4.26.0, httpx 0.28.1, FastAPI 0.133.1, uvicorn 0.41.0, PyYAML 6.0.3, mcp 2.0.0 from the interpreter's own environment. `/tmp/kfifth-environment.json` records versions/origins/limits. `/tmp/kfifth-python-bridge/sitecustomize.py` only appends the existing pytest package directory after the interpreter's own packages, so fresh exec children can import pytest without replacing native dependencies. It has no recovery/authorization monkeypatch. `/tmp/kfifth-run.py` inserts cwd, appends that same existing directory, calls `pytest.main` with the exact args, and writes adjacent JSON command/HEAD/status/exit/duration/limit receipts. This is test harness plumbing, not an install or production path.

Preserved diagnostic/correction history:

- Initial replay-only correction selection: **157 passed, 1080 deselected**, 353.61s, exit 0, `/tmp/kfifth-initial-compatible.log` (before explicit query/added coverage freeze).
- First `/tmp` runner collection failed because cwd was absent from its import path, `/tmp/kfifth-additional.log`; fixed runner only. Added owner test setup then had **4 failures, 5 passes**, `/tmp/kfifth-additional-r2.log`: registration correctly refused pending fence/old lane. Restricted to the real complete-fence window and used the new lane; **7 passed**, 14.03s, `/tmp/kfifth-additional-r3.log`.
- First valid-hello assertion used an extra `['certificate']` on an API already returning the certificate: **1 setup assertion failure**, `/tmp/kfifth-valid-hello.log`; corrected assertion, **1 passed**, 3.64s, `/tmp/kfifth-valid-hello-r2.log`.
- Earlier target deliberately interrupted after **179 passed, 408 deselected**, 91.80s, exit 2, `/tmp/kfifth-final-target.log`, to add valid-hello coverage before final freeze.
- First frozen target interrupted after **369 passed, 2 failures, 408 deselected**, 321.89s, exit 2, `/tmp/kfifth-frozen-target.log`. Both failures were fresh exec children unable to import pytest. Test-only bridge fixed propagation; the two exact tests then **2 passed**, 5.39s, `/tmp/kfifth-fresh-bridge.log`. No source/test change for this environment repair. No broad suite ran while these failures were unresolved.

- Corrected-import-path frozen target `/tmp/kfifth-frozen-target-r2.log`: **913 passed, 1 skipped, 1 failed**, 408 deselected, 1122.92s, exit 1. The single failure is unchanged `tests/service/test_admission.py::test_lotk_measurement_counts_descendants_not_root_and_flat_growth`: Python 3.11 `Path.is_symlink()` calls the test-monkeypatched `Path.lstat()` on a root for identity checking, triggering its no-root-measurement trap before measurement. `service/resource_store.py` and `test_admission.py` are byte-identical to baseline; neither is in the correction allowlist. No source/test change or waiver was made. `/tmp/kfifth-pathlib-probe.py` executes the exact current measurement function in isolation with that same guard: existing Python 3.14.7 passes (measured=expected=105), existing Python 3.11.16 trips the root-lstat trap; receipts `/tmp/kfifth-pathlib-314.json` and `-311.json`. This function-only diagnostic does NOT qualify the full 3.14 or prescribed environment. `-x` left one selected admission formula test unrun; it then **1 passed**, 0.74s, `/tmp/kfifth-target-tail.log`. All appended correction cases had passed before this inherited assertion. The target gate remains failed/unmet.

- The exact full service/release command then failed **collection**, exit 2, 3.75s, `/tmp/kfifth-frozen-full.log`: `tests/release/test_compactness.py` imports unavailable `tiktoken`. No tests executed in that command. Bounded local inventory found pipx tiktoken 0.5.2 and cached UV tiktoken 0.14.0 with a CPython-3.14-only native extension, incompatible with the current 3.11 interpreter; no local o200k_base cache was found. None was installed, substituted or downloaded. The full gate remains BLOCKED. A separate explicitly partial service/available-release command below ignores ONLY that uncollectable module to exercise the remaining tests; this is coverage of available tests, not a waiver or full-suite pass.

### Final frozen campaigns

All commands below ran sequentially on exact frozen `44e00b94d4f40ce7a058e3f84c26d571f7feb1f2`, clean entry, with no intervening source/test change. Common executable prefix:

```sh
PYTHONPATH=/tmp/kfifth-python-bridge PYTHONDONTWRITEBYTECODE=1 \
  /Users/ben/.hermes/hermes-agent/venv/bin/python /tmp/kfifth-run.py RECEIPT_NAME \
  -p no:cacheprovider
```

Each named receipt has `/tmp/NAME.log` plus `/tmp/NAME.json` with exact args, source SHA, entry status, executable/version, exit, elapsed time and inherited RLIMIT before/after. Append these exact arguments after the common prefix (replacing RECEIPT_NAME with the listed name):

- `kfifth-frozen-target-r2`: `tests/service/test_recovery.py tests/service/test_admission.py -k 'kfifth or kpre or lotk or lotp or b22' -q -rs -x` — **913 passed, 1 skipped, 1 failed**, 408 deselected, 1122.92s, exit 1. Only the inherited root-lstat assertion failed; `kfifth-target-tail`: `tests/service/test_admission.py -k 'lotk and not measurement_counts_descendants' -q -rs` — **1 passed**, 79 deselected, 0.74s, exit 0, covers the sole selected test left by `-x`.
- `kfifth-frozen-four-b2`: `tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs` — **1376 passed, 1 skipped, 1 failed**, 1245.13s, exit 1. Same single inherited root-lstat assertion, no additional failure.
- `kfifth-frozen-full`: `tests/service tests/release -q -rs` — **1 collection error**, 3.75s, exit 2, missing tiktoken; zero tests executed. This required full-suite gate is BLOCKED.
- `kfifth-frozen-available`: `tests/service tests/release --ignore=tests/release/test_compactness.py -q -rs` — **1526 passed, 19 skipped, 2 failed**, 1213.60s, exit 1. Explicitly PARTIAL coverage; one compactness test remains unexecuted. The second failure is `test_http_cli_mcp_parity_and_errors`: unchanged `clients/mcp.py:27` fails at `@server.list_tools()` because installed MCP 2.0.0 `Server` has no `list_tools`; the client reports connection closed during initialize. Repository release lock requires MCP **1.30.0**, pytest **9.1.1**, tiktoken **0.14.0**. These differ from the alternative environment; no client/SDK/test repair or dependency installation was made. `clients/mcp.py`, adapter tests, measurement source and admission tests remain byte-identical to baseline. Both failures remain open validation gates, with no cause/acceptance waiver.

All **164 appended correction cases passed** in the final targeted, four-B2 and available-test campaigns. The full gate is not green. The 19 partial-campaign skips are exactly one actual Linux RLIMIT_FSIZE row and 18 Linux peer-credential rows (10+1+1+6); every skipped mechanism remains UNQUALIFIED. Actual inherited agent-child/pytest RLIMIT_FSIZE is unchanged at `(9223372036854775807, 9223372036854775807)` before/after each final runner. No source/tests changed after `44e00b9`; no further broad campaign was repeated after these final results.

Sanitized evidence index: `/tmp/kfifth-source-audit.json`, `/tmp/kfifth-preserved-worktrees.json`, `/tmp/kfifth-environment.json`, `/tmp/kfifth-inherited-limit.json`, baseline RED log/proof, and each named log/JSON above. Source SHA-256: recovery `8c37d444276f95a0ddb5bdfbaa4983b2e2b5b8bc75b69441dfffa5df75a16597`; tests `d4db87959423ca5b36b9da6335efe9ea866a883755fc59d0c232932bd2e0f5ef`.


### Bounded verdict and remaining gates

**Reviewable owning K source/E1 candidate; required validation exit NO-GO / UNMET. No self-acceptance.** The concrete wrong-fifth defect is reproduced on baseline and all appended correction behaviors pass under the explicitly alternative E1 environment. The prescribed interpreter is absent, the inherited Python-3.11 assertion persists, the required full command cannot collect without its tokenizer, and the available broader run exposes the MCP-major-version mismatch. These are disclosed unmet gates, not accepted skips or permission to broaden this implementation lane. Parent must restore/requalify the prescribed locked environment and run the required frozen commands before marking that gate met, then obtain separate Sol-low adversarial review, root integration/source equality and meaningful integrated restart proof, fresh C replay with all four original REDs/full C matrix, and final Astra. This source/report is supplied for that process; it is not integrated, independently approved or production qualified.


All evidence is bounded local E1 source/process/SQLite/filesystem behavior with explicit Linux/host transport models. No actual Linux 4 MiB soft/hard enforcement, ext4/Unix-VFS/SHM/peer-credential, pinned-launch/host-authentication, installed/clean-host, physical ENOSPC/power-loss/daemon/VM/controller, hardware/browser/listening or acoustic qualification. Skipped mechanisms are UNQUALIFIED, never passed. No agent/shell/pytest RLIMIT was lowered; disposable child limits are modeled and actual inherited parent limits are recorded before/after. Fixed process/connection guards remain source-identical.

C1's pinned host reader remains mandatory production refusal in the untouched C WIP. No bypass or functionality claim is added. Freeze remains mandatory through host-observed fifth plus matching outputs; no automatic restart/freeze-release authority follows from historical replay. All apply remains unavailable and enrolled activation closed. A1 whole-Docker-inventory drift remains unexplained and unqualified with no inferred cause/waiver. C, B3, C1/C2, global Astra, actual hardware/browser/listening and public/unattended acceptance remain pending. Apple Silicon is preserved as documentation-only qualification scope; no implementation lane expansion.

## Owning successor coherence correction — blocked transport boundary / narrow Astra handoff (2026-10-22)

**BLOCKED WIP / NO-GO for integration or completion. No self-acceptance.** The minimal coherence correction makes the independent nonempty completed-A oracle coherent in two actual production setups. The required production B positive then reaches a second, unchanged finite boundary: its complete preparation result is larger than 65,536 bytes. No ceiling, identity, source projection or history was changed to bypass that refusal. Return this precise conflict to parent/Astra before extending implementation scope.

### Provenance and exact scope

Implementer gpt-6.1-sol / low. Fresh root-based worktree `/Users/ben/.txcode/worktrees/py-audio-box/owning-successor-coherence`, branch `fix/owning-successor-coherence`, baseline `682a179e78f7530071271789cd4c42e764a6a9ca`. Source/tests frozen in **`a11c4208b1db7220d9c401940b649f5693a0f6e9`**. Final candidate is the subsequent append-report-only commit. This is a recoverable diagnostic candidate, not a validated correction candidate.

Exact writable allowlist remains `service/recovery.py`, append-only `tests/service/test_recovery.py`, append-only this RECOVERY-P1B2 handoff. AST comparison changes only `snapshot` and `_allocation_coherent`, adds only `_recovered_dispatches`; no definition removed. Baseline tests/report remain exact byte prefixes. No Reconciliation, persistence, Core, Journal, resource_store, schemas/contracts, dependencies, packaging, C source, scripts, dispatcher, supervisor, ledger or review changes. No C source/commit was imported. `git diff --check` passes.

Read canonical orchestration/implementation plan/ledger, the entire P1B23-SUCCESSOR-r1-sol original finding, correction brief and exit matrix, KERNEL-ASTRA, C-ACCEPTANCE-ASTRA, C-PREFIFTH, K-PREFIFTH source and locked-environment reports, and C's handoff from 1717 onward. Inspected the owning snapshot/coherence, history/archive/resolution validators, Reconciliation's strict recovered-row consumption, production preparation/rehearsal and original P/K/F1–F4 test harnesses. No applicable AGENTS.md was found. Memory supplied only exact-checkout/evidence-boundary guidance, not acceptance evidence.

READONLY historical preservation checks: C `t3-1e1cddba` remains clean at exact `a27f78b5e2e09a98940b9c543474337b511ce616`; `t3-adbf212d` and `t3-4a8727d4` remain clean. No root/research/historical worktree was reset, cleaned, pruned or changed. No delegation, push, PR, deploy, Docker/network/model/audio/install/paid E2E or remote runner occurred. No parent/shell/pytest resource limit was lowered.

### Bounded source change, not acceptance

`snapshot` retains the structured result of its existing `validate_b2_database` call and supplies it to allocation coherence. This does not add/remove schema/population/projection fields or bypass the existing canonical intent/source, actual archive/schema/table, original dispatch digest, all-history and complete expected-resolution-set checks.

The small internal helper considers only Core-completed histories present in the actual/projected population. It follows validated history order and uses the latest completed resolution for each retained original dispatch. Eligibility additionally requires exact original dispatch/source bytes and digest, the entire current SQL job row (including context/principal/bundles) and reservation row equal to that completed history's accepted output. Current tuple eligibility does not require the whole current database to equal an old A output. Projected unfinished validation cannot borrow its own completion.

Only the existing reservation/outcome and missing terminal receipt mismatch branches can consume that eligibility. Ordinary authority/generation/deadline/attempt, reservation coverage/category/demand/configuration, publication and real-receipt checks remain in place. A malformed real receipt still takes the ordinary validation branch. Raw matching resolution rows are never supplied by the snapshot caller as authority. No fake PID-1 receipt/outcome/ACK, evidence deletion, reservation release, resource reset, high-water/budget/proof/hash/identity change or new command was added.

This design still requires the full independent adversarial matrix. The successful coherence assertion alone is not proof of strict negative preservation or composition through B completion.

### Actual production RED and newly reached blocker

Appended `test_successor_nonempty_production_composition[0,1]` uses `_lotp_population(...,'running')` followed by existing `_kpre_predecessor`: actual production P preparation and same-session owning K five-step A completion, then actual new owner registration and host capsule producer covering both generations. It imports no C source and supplies no fixture intent/expected rows or replacement snapshot/hash. The declared retained high-water is the existing P fixture prestate before preparation.

Baseline RED, before any source repair: two independent setups pass `_verify_core_rows(A)`, empty inventory reasons, exactly three recovery-evidence entries, original dispatch `(receipt=None,outcome=None,ACK=0)`, retained reservation; both fail the desired `allocation_coherent` positive. This reproduces the independently reviewed owning defect without relying on C's boundary.

With the correction, those same assertions including desired coherence pass twice. B proceeds through production preparation prestate and immutable plan construction, semantic intent validation and actual frozen snapshot construction. It then raises **`b2_maintenance_result_budget`** in `_prepare_abandon_locked` at its unchanged result-size check, before `_rehearse_preparation` and before B acceptance/continuation. `_run_bounded_maintenance` correctly wraps the failed child as `b2_maintenance_child_failed`; test-only exception diagnostics retain the exact internal cause and scalar sizes, not authority bytes.

Measured current-result sizes from the actual local `result` at that exception:

| Independent setup | Encoded intent bytes | Encoded plan bytes | Entire `encode({result: result})` bytes | Ceiling |
| --- | ---: | ---: | ---: | ---: |
| 0 | 92,875 | 1,256 | 94,325 | 65,536 |
| 1 | 92,872 | 1,256 | 94,322 | 65,536 |

The two digest strings each encode to 66 bytes. Differences are actual generated scalar/time encodings, not a claimed growth model. Each failure independently verifies A's exact Core output, original dispatch unchanged, **one accepted recovery / five steps / two plans**. No B accepted receipt, B recovery or B epoch was created. The committed unaccepted B plan remains immutable; it is not deleted or retried on the same root. The desired test remains RED, without skip/xfail/expected-refusal substitution.

### Narrow decision needed before continuing

The requested exit requires a nonempty production A→B positive, B rehearsal and completion with retained histories. Existing production preparation explicitly refuses nested results above 65,536 bytes **before a same-session consumer can use its owned snapshot**; the outer child transport independently has the same finite ceiling. The actual required B result is about 94 KiB. This is separate from the corrected allocation predicate and from the resource budget/high-water contract.

Please settle a bounded owning follow-up for that result-transport/history-capacity conflict, or explicitly retain this population as unsupported and revise the required positive. This implementer does not choose either policy. Increasing the ceiling, compacting/removing retained source/history, changing the four-field preparation interface, skipping the check only for a same-session consumer or altering the result transport would exceed this assignment's **minimal coherence/validated-context factor only** scope. No such workaround is present. No new architectural waiver is needed for the coherence design itself; the newly measured transport conflict is the reason for this narrow escalation.

### Commands and receipts

All pytest commands ran from the new worktree with this exact prefix (no bridge/PYTHONPATH/dependency changes):

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py
```

Actual interpreter verified CPython 3.14.7, MCP 1.30.0, pytest 9.1.1, tiktoken 0.14.0. Inherited RLIMIT_FSIZE `(9223372036854775807,9223372036854775807)`. Existing test child-only OS/limit substitutes remain E1 models, not Linux enforcement proof.

- `/tmp/pab-successor-owning-red.log`, suffix `-k successor_nonempty_production_composition -q -rs`: **2 failed, 1244 deselected, 5.73s**, exit 1. Baseline executable source; desired allocation coherence assertion is the exact failure after the independent A oracles pass.
- `/tmp/pab-successor-owning-green1.log`, same suffix: **2 failed, 1244 deselected, 6.73s**, exit 1. Coherence passes; child exception first exposes the next boundary, initially without internal cause diagnostics.
- `/tmp/pab-successor-owning-green2.log`, same suffix: **2 failed, 1244 deselected, 6.95s**, exit 1. Test-only traceback diagnostics identify the unchanged result-budget check. No source weakening between these two attempts.
- `/tmp/pab-successor-owning-budget.log`, same suffix: **2 failed, 1244 deselected, 6.79s**, exit 1. Final source/tests; exact sizes above and retained A/SQL assertions pass. Each attempt uses fresh pytest disposable installations.
- `/tmp/pab-successor-owning-preservation.log`, suffix `-k 'lotp_production_nonempty_complete or kfifth_fourth_query or kfifth_actual_death_monitor or kpre_actual_death_owned_fresh_resume' -q -rs`: **17 passed, 1229 deselected, 27.53s**, exit 0. Final source/tests; focused existing nonempty P, original accepted restart, pre-fifth monitor refusal and protected fourth-query preservation only. No skips in this selection.

No final broad campaign was started after the concrete required-positive blocker. Previous 1529/19 locked-root results remain attributed prerequisites, not this candidate's results.

### Mandatory exit-matrix accounting

1. **PARTIAL/RED:** actual nonempty A completed twice; baseline exact oracle/inventory PASS and coherence RED; corrected coherence GREEN; actual new registration/capsule covers both generations. Direct production B refuses at the next finite result boundary. B verify/rehearse/complete, repeated same-dispatch composition, optional C and both-history preservation therefore **NOT QUALIFIED**. Actual C boundary rerun remains parent-after-integration work; no C import occurred.
2. **PENDING:** complete fresh strict-refusal matrix (pending/not-complete, wrong/missing/extra resolution sets; all dispatch/job/context/reservation/category/demand/configuration/receipt/outcome/ACK, new fault/integrity/source/history/archive/reference/media defects; valid terminal receipt and original missing-receipt controls). No full negative-matrix GO inferred from source inspection or existing passes.
3. **PARTIAL:** selected existing accepted-parent, monitor pre-fifth refusal and protected fourth observational query pass unchanged. Complete reached-fourth/certificate/finalization/owner/TTL/quarantine/frozen-hash matrix **NOT RERUN** on this candidate.
4. **PENDING:** all nonempty A/B actual SQL/receipt prefixes, all creation descriptor corruptions both directions, explicit unfinished B full validation, legitimate completed-current activity and no-future-B/no-scan/no-capture observation. Existing tests/source were not weakened; their full current preservation is not claimed.
5. **BLOCKED / NOT RUN:** final frozen correction/KP/F1–F4 target, four B2 suites and full service/release, independent Sol-low review, parent integration/equality, fresh C stack/full qualification and final Astra. The 17 selected passes do not replace these gates.

### Evidence identities and limits

Frozen recovery SHA-256 `3dd97a21e43e6d812f2b578373a8d5f478deaf9a426754eab1b4ccbffe0eb903`; tests `c98dfccad9a610ac3f391956252a23a2c81e7cf62492db0e9b9cf666b6dea233`. Final budget log `9337fe64bbc98d47265e6af9e8f1c6004f6230e1f9fa92c3f433d32a69322774`; preservation log `155872742ca3335110ca2fb02e545e7df5b4150db830f3d7a64b58d114290327`. Temporary logs/disposable roots are retained and are not durable installed receipts.

All 19 inherited Linux cap/peer-credential skips remain UNQUALIFIED; no new skip waiver. Actual Linux 4 MiB/ext4/Unix-VFS/SHM, pinned C1 host/authentication/provenance/launcher, C1/C2 installed/clean-host, B3/global Astra, hardware/browser/listening/acoustic, physical ENOSPC/power-loss/daemon/VM/controller durability and public activation remain pending. C1's mandatory pinned-context refusal is untouched. A1 whole-Docker-inventory drift remains unexplained/unqualified with no inferred cause or waiver. Apple Silicon remains documentation-only qualification scope. No full Phase 1 claim.

**Return blocked WIP for narrow parent/Astra decision, not self-accepted implementation.** Keep the exact source/test commit and this append-only report recoverable; do not integrate as a complete correction or stack C over it until the new boundary and full required matrix are resolved.

## T1 bounded transport continuation — recoverable WIP, NOT complete (2026-10-22)

**NO-GO for complete T1, integration, T2 stacking or self-acceptance.** Actual production nonempty A→B→C now completes in two independent installations. The finite continuation completes D, then E refuses at the unchanged 1 MiB snapshot limit (`b2_prefix_snapshot_budget`); completed A/B/C/D replay remains observational. This is progress, not the seven-exit acceptance claim. A concrete inherited-test/allowlist conflict remains: the three original `test_lotp_completed_history_preparation[result-budget-{valid,missing,unreadable}]` cases explicitly require B below the selected 1 MiB envelope to refuse at the superseded 64 KiB policy. They now fail `DID NOT RAISE ValueError` at the original line 4147. The assignment requires APPEND-only tests and no weakened original tests. Those exact original bytes/assertions remain untouched; no override, skip, xfail, fixture-dependent production refusal or collection manipulation was added. A parent scope resolution is needed to update that obsolete expectation faithfully before a fully green target is possible. This is a test-contract conflict, not an additional production capacity blocker or a request to raise a limit.

### Identity, replay, scope and preservation

- Implementer: gpt-6.1-sol / low; no delegation or independent review performed here.
- ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-f0f283c4`, branch `t3/bounded-recovery-transport`; clean starting baseline **`e7c5e2fec680c52a1fd9c57218b5a2130765a857`**. Baseline executable diff against reviewed integrated K `6130d1fecce3d978671ed1bfd509693d3aab3c97` is empty over service/worker/scripts/tests/clients/contracts.
- Inspected oldest-first `682a179..6cf6236`: only source/tests `a11c4208b1db7220d9c401940b649f5693a0f6e9`, then report `6cf6236bd42750cd82a9f680fac18d7f523bbe51`. Replayed ONLY these two as **`348fe17`**, **`82db44f`**, without conflict. Root reports/ledger and reviewed K remain intact. No base/C import.
- Frozen continuation source/tests **`8256f55ee7609315ad487e5ec7342cdc8d99a571`**. Subsequent handoff commit changes this document only. Final candidate is that report-only commit, not an accepted implementation.
- Exactly three baseline-delta paths: `service/recovery.py`, appended `tests/service/test_recovery.py`, appended this document. Original tests and handoff are exact byte prefixes, verified against the starting baseline. No Core/persistence/reconciliation/resource_store/schema/admission/worker/dispatcher/host script/C/packaging/dependency/contract/ledger/review write. No enabled apply.
- Read the entire TRANSPORT-ASTRA authoritative amendment, independent COHERENCE-r1 and original SUCCESSOR-r1, KERNEL-ASTRA, C-ACCEPTANCE-ASTRA, ACCOUNTING-ASTRA, canonical PRODUCT/IMPLEMENTATION-PLAN/ORCHESTRATION-PROTOCOL/EXECUTION-LEDGER, and owning handoff from 1555 through EOF; inspected actual preparation/session/codec/transport/rehearsal/coherence/history code and existing test harness. No applicable AGENTS.md found in checkout or ancestors. Memory supplied only exact-scope/evidence-tier guidance, not acceptance evidence.
- READONLY historical identity/status checks: owning-successor-coherence remains clean at **`6cf6236bd42750cd82a9f680fac18d7f523bbe51`**; t3-1e1cddba clean at **`a27f78b5e2e09a98940b9c543474337b511ce616`**; t3-adbf212d clean at **`58e3eb9f40a8db5ca162fe8f64922efb13f224c4`**; t3-4a8727d4 clean at **`6cfe3973f41db92679531e38660d88cf72ce27eb`**. No research/worktree cleanup/reset/prune or other-checkout write.

### Implemented bounded source delta

Distinct internal `MAINTENANCE_RESULT_LIMIT = 1_048_576`, inclusive entire canonical UTF-8 `{"result":result}`. Four preparation fields/full documents remain unchanged. One incremental canonical chunk encoder uses the existing finite/sorted/compact/ensure_ascii=False settings; each potentially large JSON string chunk is UTF-8 encoded in at most 16,384-character slices (at most 64 KiB each). It stops before accumulating above T. Preparation discards its temporary frame before rehearsal; no serialization is stored on `_Preparation`. The reader holds at most T+1 accumulated bytes and compares canonical re-encoding incrementally without a second accumulated frame. This is a finite transport-buffer policy, NOT total RSS≤T or RLIMIT_AS qualification.

Strict decoder rejects duplicate nested keys, nonfinite values, invalid UTF-8/surrogates, wrong outer shape, trailing/second frames and noncanonical bytes. Preparation validates exact four-field shape/types, canonical plan/intent semantics, both digests, plan identity, full shared plan-intent fields, creation time and journal bindings before the preparation check and again in the parent production entrypoint. Codec JSON supplies no owned snapshot/restart authority. The existing private K operation test seam still returns its prior test-only JSON values; it was not converted into preparation authority.

The common absolute 30-second deadline now starts before fork and covers writing, bounded reading, EOF and child termination. Reads are `min(65536,T+1-used)`. Child writes use nonblocking bounded memoryviews with partial-write progress and EINTR handling; EPIPE/zero write/error exits 74. Parent parses only after EOF and successful normal exit. Post-EOF wait uses WNOHANG/remaining-deadline bounded waits; error/expiry kills and reaps the exact owned child. Fork failure closes both pipe descriptors; read/write/wait EINTR do not reset deadline. No retry of an operation, service restart or rollback inference.

Retained coherence WIP still consumes only validated completed history, exact original dispatch, whole current job/context and reservation, selecting the latest completed resolution. No resource formula, H, charges, budget, 1 MiB snapshot, 64 KiB document/certificate, 4 MiB file, 2048-entry or 16 MiB aggregate limit changed. No C proposal/stdout change belongs to this WIP; T2 remains parent-owned after reviewed T1 integration.

### Actual production witness and finite capacity

Appended `test_t1_nonempty_third_production_generation[0,1]` starts real running allocation, seeds retained H only as declared pre-preparation fixture state, and uses actual production preparation/rehearsal and same-session K completion. Every generation registers a distinct actual owner/capsule union through existing production host producer paths under explicitly modeled E1 OS/host transport. Capture counting requires exactly one original capture per generation and unchanged bundle/control snapshot SHA-256. Actual Core oracle, all completed descriptor/receipt validators and resolution sets run; original dispatch remains receipt=null/outcome=null/ACK=0, reservation retained, H/charges/budget unchanged. No fixture intent/expected-row/hash injection, empty replacement or preparation-time H update.

Representative measured run (first independent setup in `/tmp/pab-t1-positive-negative.log`; second setup also completed):

| Generation | Entire frame | Intent | Plan | Original snapshot | Manifest | Receipt certificate | Receipt bytes/count | M / U_all | Scan / actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| A | 36,617 | 35,170 | 1,253 | 249,856 | 859 | 8,735 | 299,412 / 26 | 303,784 / 14,066,344 | 628,659 / 699,136 |
| B | 94,319 | 92,871 | 1,254 | 294,912 | 1,763 | 8,746 | 700,083 / 39 | 587,296 / 14,349,856 | 1,032,291 / 1,082,168 |
| C | 222,912 | 221,462 | 1,256 | 405,504 | 1,763 | 8,746 | 1,339,937 / 52 | 995,000 / 14,757,560 | 1,665,947 / 1,711,704 |

Every row H=20,971,520, retained charges=2,097,152, unchanged enrolled disk budget=67,108,864/free reserve=16,777,216. `max(actual,H)==H`. Observed DB/WAL/SHM maxima at actual continuation hooks: A 192,512/222,512/32,768; B 294,912/206,032/32,768; C 405,504/317,272/32,768. These are sampled observations, not actual Linux enforcement qualification or total heap measurements. Scalar/time encodings cause small byte differences across runs.

Finite witness is predeclared at most three further preparations D/E/F, each using the same 30-second process deadline, stopping on first refusal. Both installations complete D and refuse E at `b2_prefix_snapshot_budget`; no F attempt. E leaves its immutable unaccepted plan retained. Logical SQL excluding that one plan and retained protocol/archive bytes stay unchanged; physical current SQLite DB/WAL/SHM churn is explicitly excluded from byte immutability. Fresh interpreter replay twice for each A/B/C/D uses the current retained registration union and traps snapshot/source reconstruction/scan/continuation writers. All remain complete without granting current freeze release. No history deletion, credits or larger cap.

### Seven-exit accounting — no aggregate GO

1. **Substantial witness demonstrated, not full acceptance:** two independent nonempty production A/B/C, 3 histories/15 steps and repeated original dispatch resolutions; exact captured/archive bytes, Core/receipt validators and metrics above. Independent adversarial verification and the full all-prefix interaction campaign remain pending.
2. **PARTIAL:** exact 65,536/65,537/T−1/T/T+1 UTF-8 framing, multibyte and escaped nested strings, invalid semantics/shape/digests/bindings and duplicate/nonfinite/noncanonical/trailing/invalid encoding tests. Framing strings are explicitly not semantically valid preparation history. Actual budget-before-rehearsal/acceptance writer traps cover standalone and same-consumer paths. Full document-boundary/capacity cross product remains pending.
3. **PARTIAL:** 18 actual fork/pipe cases plus fork-failure descriptor check, EOF/partial/extra/second frame, fragmented/short/EINTR/EPIPE/zero writes, read EINTR, nonzero/signalled child, post-EOF alive, never EOF, slow/backpressure/error paths and exact-T-alive deadline. Deadline is accelerated consistently in these small process tests; parent RLIMIT_FSIZE unchanged and every owned child reaped. Not a complete independent transport fault matrix; explicit waitpid-EINTR/real blocked-backpressure and every exact-size successful pipe combination remain pending.
4. **PARTIAL:** new isolated encoder-budget failures prove late immutable unaccepted plan, session burn, zero recovery/steps/resolutions and no rehearsal/continuation; selected existing P preparation preservation passes. Full actual snapshot/document/file/aggregate/entry/free-reserve equality/above/below matrix on both consumers remains pending. No equality claimed from patched size alone.
5. **DEMONSTRATED finite local capacity witness:** real E snapshot refusal after A/B/C/D, immutable unaccepted plan/no new accepted authority and twice-fresh observational replay of every completed history. This does not promise unlimited history or D elsewhere.
6. **PARTIAL:** 28 appended corruption populations, two fresh refusals each with preparation/protocol writer traps and unchanged logical SQL/protocol bytes: missing/wrong/extra/pending resolution; original authority/binding/runtime/phase/receipt/outcome/ACK; principal/context/bundles/state/attempt/generation/deadline/reap; reservation category/configuration/demand/state; history/archive/descriptor/media/typed reference. Genuine terminal receipt control versus unresolved terminal missing/malformed receipt, new fault/integrity variants, every independent historical-set combination and full negative composition remain pending.
7. **PENDING complete campaign:** selected existing accepted restart, protected fourth query and pre-fifth monitor cases pass; prior reviewed source preserved. Full nonempty B/C actual-prefix F1/F2/F3/F4, every history descriptor both directions, all current-after-image/unaffected-source before each remaining write, TTL/both-lock/session-burn/Unix-VFS campaign has not been rerun in full on this candidate. Do not borrow predecessor K's 1529/19 result.

Because full target is not green and all seven exits are not closed, **no four-B2 or full service/release campaign, independent review, integration/equality acceptance, T2 or final Astra was performed**. The original 3 RED expectations are not waived. Parent must resolve their obsolete policy expectation within an explicit test-edit scope; then complete the pending rows and frozen sequential campaigns before independent review/integration.

### Commands, failures and receipts

All pytest runs use precisely this locked prefix, from this worktree, no environment/dependency install/change:

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py
```

Verified CPython3.14.7/pytest9.1.1/MCP1.30.0/tiktoken0.14.0. Parent RLIMIT_FSIZE before/after remains `(9223372036854775807,9223372036854775807)`; only existing disposable child models apply. No Linux cap pass inferred.

- `/tmp/pab-t1-b-first.log`, `-k successor_nonempty_production_composition -q -rs`: 2 passed/1244 deselected, 13.89s, exit0. B transport contradiction removed, real B completes.
- `/tmp/pab-t1-c-first.log`: 2 failed/1246 deselected, 14.93s, exit1, test setup reused original container IDs and correctly hit ambiguous-incarnation refusal. Fixed only appended test identities; original checks unchanged.
- `/tmp/pab-t1-c-second.log`, `-k t1_nonempty_third -q -rs`: 2 passed/1246 deselected, 23.02s, exit0, actual C completed before metrics/capacity extension.
- `/tmp/pab-t1-capacity.log`: 2 failed, 21.19s, test incorrectly indexed `logical()` SQL dump as dict and omitted `_prewrite` on direct test seam. `/tmp/pab-t1-capacity2.log`: 2 failed, 23.32s, overbroad test trap blocked rehearsal. Corrected test to use production prewrite and declared finite D/E/F continuation. `/tmp/pab-t1-capacity3.log`: 2 failed, 38.81s, overstrong physical DB-byte assertion; `/tmp/pab-t1-capacity4.log`: 2 failed, 39.07s, exclusion missed retained-bytes role suffix. `/tmp/pab-t1-capacity5.log`: 2 failed, 40.85s, fresh replay host model omitted the later current registration union. These are retained test-development failures, not counted as passing capacity proof.
- `/tmp/pab-t1-codec-pipe.log`, `-k 't1_exact_utf8 or t1_strict_closed or t1_actual_pipe or t1_fork_failure' -q -rs`: 47 passed/1248 deselected, 2.42s, exit0.
- `/tmp/pab-t1-positive-negative.log`, `-k 't1_nonempty_third or t1_successor_strict' -q -rs -s`: 29 passed/1 failed/1293 deselected, 101.83s, exit1. Both complete production/capacity/replay witnesses pass; sole failure is appended deadline mutation treating an ISO string as numeric. Fixed to a changed ISO timestamp, with no production change.
- `/tmp/pab-t1-semantics-legacy.log`, selection `t1_preparation or (t1_successor_strict and job-deadline) or (lotp_completed_history_preparation and result-budget)`: 8 passed/5 failed/1319 deselected, 20.90s, exit1. Three inherited obsolete-limit assertions plus two new overbroad preparation transaction traps. Corrected the new trap to permit only schema/plan transactions, preserving all acceptance traps. Deadline/semantic negatives pass.
- `/tmp/pab-t1-budget-corrected.log`, `-k t1_preparation_budget -q -rs`: 2 passed/1330 deselected, 2.66s, exit0.

Frozen-source bounded diagnostic receipt is appended below once its process ends. It is deliberately not described as the full correction/K/P/F1–F4 target. Logs and disposable roots remain under /tmp and pytest temporary directories, retained for review.

Frozen SHA-256: recovery `d54e776b114710ac0fd8ac6781b5ef11e9457d48c269ccabc98637cb7b304dcc`; tests `e17cf47b65d02087411520e1397acd81b6f98f5c028e8751788b4fb128e45a94`. Source/test guards and `git diff --check` pass; final handoff commit must not change those bytes.

All 19 Linux cap/peer-credential mechanisms remain unqualified. Actual Linux4MiB/ext4/Unix-VFS/SHM, pinned C1 host/provenance/launcher and mandatory refusal, B3/C1/C2 installed/clean-host, global Astra, hardware/browser/acoustic and physical durability remain pending. A1 global Docker inventory drift remains unexplained/unqualified with no inferred cause/waiver. Apple Silicon remains documentation-only qualification scope. No Docker/network/model/audio/paid E2E/remote script, dependency installation, parent limit lowering, push/PR/deploy, service restart or activation. Host freeze and public/enrolled closure remain unchanged.

### Frozen bounded diagnostic completion

On exact frozen source/tests `8256f55ee7609315ad487e5ec7342cdc8d99a571`, `/tmp/pab-t1-frozen-bounded.log`, suffix:

```sh
-k 't1_ or successor_nonempty_production_composition or lotp_production_nonempty_complete or kfifth_fourth_query or kfifth_actual_death_monitor or kpre_actual_death_owned_fresh_resume or (lotp_completed_history_preparation and result-budget)' -q -rs
```

**105 passed, 3 failed, 1224 deselected in 160.44s, pytest exit1; no skips/xfails.** All 86 appended T1 cases pass, both retained successor composition cases pass, and 17 selected existing K/P preservation cases pass. The only three failures are the original obsolete `result-budget` expectations described above, all at original test line4147 (`DID NOT RAISE ValueError`). No assertion was weakened or marked expected-failure to obtain green. Full correction/K/P/F1–F4 target, four-B2 and service/release remain NOT RUN; no aggregate GO.

Post-run SHA-256 guards still equal the frozen values above; only appended handoff bytes differ from the source/test freeze. Starting-baseline delta remains exactly the three authorized paths and original test/report prefixes remain intact. **Return this recoverable WIP and exact RED gate to parent; complete T1 is NOT delivered or accepted.**

## Complete T1 owning continuation — frozen local candidate for independent review (2026-10-23)

**Implementation candidate; no self-GO, integration, T2 or global acceptance.** This section supersedes the previous T1 WIP's pending local rows only to the extent of the fresh evidence below. The original WIP and its RED receipts remain historical evidence. Separate Sol-low review must inspect the entire final delta; parent owns integration/equality proof, T2 and final Astra.

### Identity and preservation

- ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-99d25777`, branch `t3/complete-t1-transport`; fresh root **`7ae446ccb81473cc82ae50ae5ea8305f1729eb4c`**, retaining the parent's explicit test-scope amendment. Original WIP baseline **`e7c5e2fec680c52a1fd9c57218b5a2130765a857`**; original candidate **`d03bc47472913880d32fd2e8b1b738806ade01a0`**, source **`8256f55ee7609315ad487e5ec7342cdc8d99a571`**, stays untouched in `t3-f0f283c4`.
- Replayed only `e7c5e2f..d03bc47` oldest first: `348fe17`, `82db44f`, `8256f55`, `d03bc47` became `b2617b8`, `d3b146b`, `2373883`, `b3acb8e`. No root-base/C import. The fresh root's executable baseline equals reviewed K; root reviews, canonical documents and ledger remain unchanged.
- First source/test freeze **`8788ecd7cfcd46822b0c117c13175079da501b75`** exposed an unchanged preservation-test conflict with mutable owned proposal reads. Corrected source/test candidate **`b91d51dc2bda5e7f013a6bd99426548e831a983d`** is the immutable basis of the three final campaigns. The final handoff commit is report-only; exact final HEAD is returned separately.
- Writable baseline delta is exactly recovery.py, recovery tests and this appended handoff. All old recovery-test bytes outside the specifically amended result-budget function remain exact; inside it only the obsolete result-budget branch and its stale cap comment changed. All other original branch semantics remain. All seven local T1 rows are accounted for below, with actual and isolated boundary evidence distinguished. Handoff is append-only. No Core/persistence/reconciliation/resource_store/schema/admission/worker/dispatcher/supervisor/C/script/packaging/dependency/contract/ledger/review changes.
- Recovery SHA-256 **`6da71f0870043e068c47464b81a2a40522a6246d2adbbe0bae8b088f5c0f72a1`**; tests **`06ceb9499582ba67d5b2634f72fb8e9029936b09471aa380153f8077b771e945`**. `/tmp/pab-t1-frozen-identity.json` records exact cwd, SHA, hashes, interpreter/packages and parent limit. Final source guards and historical-worktree inventory: `/tmp/pab-t1-final-guards.json` confirms clean frozen HEAD, exact source/test hashes, normal parent FSIZE, exact three-path scope, unchanged old-test/handoff prefixes except the approved branch, no removed definition and reviewed-K executable baseline equality. AST changes only `_run_bounded_maintenance`, `snapshot`, `_allocation_coherent`, `_preparation_prestate`, `_Preparation`, `_prepare_abandon`, `_prepare_abandon_locked`; adds only `_maintenance_result_chunks`, `_maintenance_result_frame`, `_validate_preparation_result`, `_decode_maintenance_result` and `_recovered_dispatches`. Historical READONLY checks in `/tmp/pab-t1-historical-inventory.json` all match exact clean pins:

| Worktree | Exact clean HEAD |
| --- | --- |
| `t3-f0f283c4` | `d03bc47472913880d32fd2e8b1b738806ade01a0` |
| `owning-successor-coherence` | `6cf6236bd42750cd82a9f680fac18d7f523bbe51` |
| `t3-1e1cddba` | `a27f78b5e2e09a98940b9c543474337b511ce616` |
| `t3-adbf212d` | `58e3eb9f40a8db5ca162fe8f64922efb13f224c4` |
| `t3-4a8727d4` | `6cfe3973f41db92679531e38660d88cf72ce27eb` |
| `t3-85d4bb33` | `6130d1fecce3d978671ed1bfd509693d3aab3c97` |

Read the entire authoritative TRANSPORT-ASTRA, COHERENCE-r1-sol and original SUCCESSOR-r1-sol, canonical orchestration/implementation plan and current ledger, and the entire newest candidate T1 handoff. Prior memory supplied scope/evidence-tier guidance only, never test or acceptance evidence. No delegation or self-review was performed.

### Actual source corrections and the original-test amendment

The replayed fixed **T=1,048,576**, entire canonical UTF-8 `{"result":four_fields}`, remains independent of snapshot/document/file/history limits. Incremental encoding uses at most 16,384-character strict UTF-8 slices; parent reads at most 65,536 bytes and one overflow byte. One accumulated encoded frame per endpoint, no persistent serialization on `_Preparation`, no chunk list/compression/reference protocol. Decoder rejects duplicate/nonfinite/invalid/noncanonical/trailing/wrong-shape data; the production P entrypoint validates full preparation semantics/digests/bindings at both ends. Existing source parsing/object allocations remain separately bounded; this is not total RSS <= T or RLIMIT_AS qualification.

The same absolute 30-second deadline covers operation, read, EOF and normal child exit; WNOHANG replaces post-EOF blocking wait, and failure/expiry closes descriptors and kills/reaps the exact owned child. EINTR does not reset the deadline; short/zero/error writes, EPIPE, abnormal exit, partial/extra frames and backpressure cannot return success. No retry, rollback inference or service restart. An actual completed recovery followed by lost result-pipe write returns no success, while a fresh completed lookup remains observational.

Durable writer markers exposed three gaps that exception-only child-failure assertions could conceal. Narrow fixes require supported current integrity identity/revision, the actual fifth receipt before a new initial successor proposal, and retention of the recovered job's bound append-only event rows when consuming its latest completed resolution. Added events remain permitted; whole current DB equality with old A is not imposed. Original dispatch, whole job/context/reservation and latest completed selection remain exact. New resource faults are intentionally repairable by a fresh explicit plan; the strict new-fault negative injects after C acceptance and proves full current-source refusal before remaining writes, without combining unrelated corruption to manufacture refusal.

The first frozen target found unchanged `test_lotp_frozen_output_and_source_oracles[expected]`: it mutates a candidate intent for a refused continuation, then returns the preparation. The strict parent correctly rejected the mutated proposal. `_Preparation` now returns defensive copies for plan/intent reads, preserving its original validated proposal while the invalid continuation still refuses and burns its session. The parent validator remains strict; an explicitly malformed raw result still refuses. No old assertion was inverted or amended for this finding.

The **only** preexisting test amendment is the parent's approved `test_lotp_completed_history_preparation[result-budget-{valid,missing,unreadable}]` branch. It still expects refusal and retains two immutable plans / one recovery / five steps. Actual B is semantically validated and below T; a narrowly isolated encoder supplies a canonical UTF-8 framing fixture of **T+1** bytes, never caller padding in production history. Rehearsal/receipt/continuation traps, unchanged logical SQL except the one new plan, unchanged protocol bytes, and session-burn/no-retry checks are now explicit. The original historical bytes remain in d03bc47. Below/equal boundary fixtures and actual production B/C positives accompany this refusal. No skip/xfail/deletion/collection override.

### Explicit seven-exit and pending-row accounting

| Exit | Fresh exercised rows and limits |
| --- | --- |
| 1 — actual nonempty composition | Two independent real Core/Admission/DispatchStore running populations; production P preparation/rehearsal and same-session K A/B/C, fifteen steps and three repeated-dispatch resolutions each. Distinct actual registration/capsule-producer generations under explicit E1 OS/host models. Exact Core/archives/descriptors, original dispatch/null receipt/null outcome/ACK=0, retained reservation, H/charges/budget and once-captured original snapshot bytes checked. Numeric observations below. No fixture intent/output/hash substitution. |
| 2 — canonical codec | 65,536/65,537/T-1/T/T+1 whole UTF-8 frames, multibyte and escaped retained-string fixtures; noncanonical/duplicate/unknown/nonfinite/encoding/trailing/shape/digest/binding negatives. Framing strings explicitly fail preparation semantics. Both consumers retain full real preparation validation before isolated encoder-budget checks; T-1/T positives and T+1 preaccept refusal. |
| 3 — actual pipe/process | EOF empty/mid/exact, extra byte/second frame, fragments/short/zero writes, read/write/waitpid EINTR, injected EPIPE, signalled/nonzero children, post-EOF alive/never EOF/slow writer, real kernel-full backpressure, exact-size successful pipes through T, exact-T alive deadline, receiver T+1 refusal before decoder, fork-failure descriptor closure and exact-child reap. Durable-complete/lost-result control plus twice-fresh observational lookup. Parent FSIZE unchanged; accelerated deadline fixtures are E1 control-flow tests, not Linux enforcement. |
| 4 — capacity and no authority | Both standalone and same-session routes. Real SQLite backups at 255/256/257 4KiB pages bracket the 1MiB cap; equality is a valid DB, while byte +/-1 is a file-reader guard fixture (valid SQLite sizes are page multiples). Real closed manifests at 65,535 and 65,537 bytes: the fixed 557 bytes plus hex journal make exact 65,536 unattainable for that shape. Actual one-job versus sixteen-job populations bracket receipt-certificate capacity; fixed closed core certificates cannot be padded into a semantic 64KiB positive. Physical known-oversize/copy and real private SQLite-full rehearsal refusals; file-reader/stat guards at 4MiB +/-1; modeled free-reserve equality/below/above with all other production prerequisites valid. Actual namespace-file guards at 2,048 entries and 16MiB +/-1, and an explicitly isolated real namespace measurement at the real rehearsal reserve guard (count+18 / bytes+snapshot+documents+18*65536) on both consumers. These namespace/reserve measurements are not fabricated accepted history or claims of a valid live population at every equality. Actual production capacity refusals and A/B/C/D/E witness are separate. Early refusals create no plan; late refusals retain one immutable unaccepted plan/schema, burn session, and create no recovery/step/resolution/fence/bundle/healthy epoch. Physical SQLite header/WAL/SHM churn is excluded from byte immutability, never from logical SQL checks. |
| 5 — finite history | Predeclared at most D/E/F attempts, each under the same finite process deadline. Both installations complete D and first refuse E at unchanged `b2_prefix_snapshot_budget`; no F attempt, larger cap, deletion or credit. Exactly five plans, four recoveries, twenty steps and four resolutions after E: one unaccepted plan, no E authority. A/B/C/D completed lookups remain observational twice fresh with scan/capture/continuation writers trapped. No unlimited-history or universal-D promise. |
| 6 — strict negatives | Original 28 A-based cases retained; 28 additional actual A/B repeated-resolution populations use durable writer markers, two fresh production refusals and unchanged logical SQL/protocol bytes. Missing/wrong/extra/pending sets; original dispatch authority/binding/runtime/phase/receipt/outcome/ACK; whole job principal/context/bundles/state/attempt/generation/deadline/reap; reservation category/configuration/demand/state; history/archive/descriptor/media/typed-reference. Added incomplete fifth, unsupported integrity revision, malformed real receipt with old resolution, accepted-C new fault, and genuine terminal receipt versus unresolved missing/malformed controls. No blanket terminal exemption or old-resolution fallback. |
| 7 — F1/F2/F3/F4 and preservation | Actual nonempty B and C at every reached receipt-only/SQL/fourth/certificate/finalization/final-fence prefix, with asserted death hooks and fresh interpreter replay. Every A/B/C creation descriptor, missing/wrong/mode, in both directions (including self checks), actual protected files and twice-fresh refusals before scans/writers. C after-image and unaffected-source mutations at five reached fourth/finalization stages refuse before any remaining writer. Genuine A/B/C completed lookups tolerate later ordinary current activity, with no successor continuation, future-output demand, capture, scan, old-state restoration or current freeze release. The complete current K/P/b22/kpre/kfifth target freshly covers original owner union/incarnations, stage prerequisites, accepted TTL, exact no-attempt hash/oracle recapture, no recapture after attempts, session burn, Unix-VFS and both locks. No predecessor aggregate is borrowed. |

### Fresh frozen production measurements

`/tmp/pab-t1-target-r2-metrics.json`, source b91d51d, copies both actual target-run witnesses and independently checks their post-E SQL counts, original dispatch tuple fields, retained reservation, H and charges. These are actual local synthetic production paths with host/OS modeling, not installed-host/worker authentication evidence.

| Setup / generation | Frame | Intent | Plan | Snapshot | Manifest | Receipt certificate | Receipt bytes / count | M / U_all | Scan / actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 / A | 36,617 | 35,170 | 1,253 | 249,856 | 859 | 8,735 | 299,412 / 26 | 303,784 / 14,066,344 | 628,659 / 699,136 |
| 1 / B | 94,322 | 92,872 | 1,256 | 294,912 | 1,763 | 8,746 | 700,084 / 39 | 587,296 / 14,349,856 | 1,032,291 / 1,082,168 |
| 1 / C | 222,912 | 221,464 | 1,254 | 405,504 | 1,763 | 8,746 | 1,339,940 / 52 | 995,000 / 14,757,560 | 1,665,947 / 1,711,704 |
| 2 / A | 36,617 | 35,170 | 1,253 | 249,856 | 859 | 8,735 | 299,412 / 26 | 303,784 / 14,066,344 | 628,659 / 699,136 |
| 2 / B | 94,319 | 92,871 | 1,254 | 294,912 | 1,763 | 8,746 | 700,083 / 39 | 587,296 / 14,349,856 | 1,032,291 / 1,082,168 |
| 2 / C | 222,909 | 221,461 | 1,254 | 405,504 | 1,763 | 8,746 | 1,339,936 / 52 | 995,000 / 14,757,560 | 1,665,947 / 1,711,704 |

For all six A/B/C rows: H=20,971,520; retained charges=2,097,152; unchanged enrolled disk budget=67,108,864/free reserve=16,777,216; `max(actual,H)==H`. `U_all=M+13,762,560` is unchanged `M+3*4,194,304+1,048,576+2*65,536`. Observed DB/WAL/SHM maxima during actual continuation hooks, A/B/C respectively: 192,512/222,512/32,768; 294,912/206,032/32,768; 405,504/317,272/32,768 in both setups. These are sampled file observations, not continuous peaks, total heap/RSS or actual Linux cap/ext4/Unix-VFS/SHM qualification. Scalar time encodings explain small byte variation; no fixed recursive growth promise is inferred.

### Commands, failures and final receipts

All pytest commands use exactly this locked prefix from this worktree, wrapped by `/usr/bin/time -p` for final runs:

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider
```

CPython3.14.7, pytest9.1.1, MCP1.30.0, tiktoken0.14.0. No installation/environment replacement. Actual parent FSIZE before/after remains `(9223372036854775807,9223372036854775807)`; no parent limit lowering.

All three completed sequentially on b91d51d with exit0 and clean exact source/hash/limit guards before/after. No final-campaign failure or xfail. Full skip reasons: one actual Linux RLIMIT_FSIZE case and eighteen Linux peer-credential cases (10+1+1+6); every one remains unqualified.

| Stage / log | Exact pytest suffix | Result | pytest / wrapper seconds | Exit |
| --- | --- | --- | --- | --- |
| `/tmp/pab-t1-final-target-r2.log` | `tests/service/test_recovery.py tests/service/test_admission.py -k 'kfifth or kpre or lotk or lotp or b22 or successor or t1_' -q -rs` | 1,204 passed / 1 skipped / 408 deselected | 2678.90 / 2679.83 | 0 |
| `/tmp/pab-t1-final-four-b2.log` | `tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs` | 1,666 passed / 1 skipped | 2894.26 / 2895.30 | 0 |
| `/tmp/pab-t1-final-full.log` | `tests/service tests/release -q -rs` | 1,818 passed / 19 skipped | 2888.25 / 2889.47 | 0 |

Receipt SHA-256:

| File | SHA-256 |
| --- | --- |
| `/tmp/pab-t1-final-target-r2.log` | `60f17ead8d613d3a533f82b8b620d3566d90972ab8ce83e29444d63e0bda741f` |
| `/tmp/pab-t1-final-four-b2.log` | `7fe1fd4e1a5aab0e0456aef68845342d65fa18106cd1b2b5138d915a6a47cafa` |
| `/tmp/pab-t1-final-full.log` | `9961565f73dd60bc0ba7dd66edef9168f1b6120047568ac8e5a03c9adadc4552` |
| `/tmp/pab-t1-target-r2-metrics.json` | `e22e56509c2a58c9f160076fef4ba2a28e47350e002447177b23b3c6f6a5e0d7` |
| `/tmp/pab-t1-final-guards.json` | `ba951875c855fcbc2991254507406d8793bbbc8da5aed91ea9bd4b3ca00df823` |
| `/tmp/pab-t1-historical-inventory.json` | `6e34b731755f6a19d1928a545eb6db376ce2a24391cb9899cdc26cbd6fa5d5d5` |

Three-path diff against the fresh root before this final report:

```text
 docs/agent-interface/phase-1/RECOVERY-P1B2.md |  170 ++++
 service/recovery.py                           |  208 ++++-
 tests/service/test_recovery.py                | 1123 ++++++++++++++++++++++++-
 3 files changed, 1475 insertions(+), 26 deletions(-)
```

New development receipts are supporting diagnostics, not substitutes for those final frozen campaigns. `/tmp/pab-t1-complete-boundaries.log`:18 passed12.37s; capacity development6 failed/33 passed17.72s because the freelist fixture overshot its target, corrected real snapshot6 passed12.00s; history probes22 passed182.80s. Durable strict markers exposed3 failures/29 passes162.10s; focused guard correction17 passed46.29s. Certificate population4 passed9.32s; reserve/legacy15 passed20.47s; full new-history development83 passed1130.20s (source changed during development, not frozen evidence); other new matrix207 passed357.91s; final transport/legacy seams6 passed9.92s. The new-fault early-refusal development assumption was corrected to the actual postaccept drift invariant, not enforced by an unrelated corruption or a new blanket fault policy.

The first frozen target on8788ecd was interrupted for inspection after **1 failed,520 passed,1 skipped,408 deselected in479.53s, exit2**; remaining selected tests were not run in that attempt. Failure was the unchanged mutable-owned-proposal test described above. `/tmp/pab-t1-owned-proposal-correction.log`:19 passed77.03s, including the unchanged failure, strict malformed-result negative, actual successor compositions and both finite production witnesses. The entire target was then restarted on b91d51d, followed sequentially by four-B2 and full. No failing test was skipped or weakened.

Named logs, source identities and copied metrics remain in /tmp; ordinary pytest/temp-context retention is unchanged, with no manual evidence/worktree/research cleanup. Final source/test guards, `git diff --check`, scope and historical-worktree receipts accompany the final handoff. No post-validation executable change.

### Remaining authority and qualification gates

All **19 Linux cap/peer-credential skips remain UNQUALIFIED**, not passed. Actual Linux4MiB/ext4/Unix-VFS/SHM, pinned C1 launcher/host provenance/authentication and its mandatory production refusal, C1/C2 installed/clean-host, B3/global Astra, physical ENOSPC/power-loss/daemon/VM/controller durability, hardware/browser/listening/acoustic and public activation remain pending. A1 whole-inventory drift remains unexplained/unqualified; no cause or waiver inferred. Apple Silicon remains documentation-only qualification scope. Host freeze persists through host-observed fifth and matching outputs; no automatic service restart/freeze release follows from history lookup.

No Docker, external network, model/hardware/audio campaign, paid E2E/remote script, install, limit lowering, push/PR/deploy, root-base/C import, other-worktree edits, reset/clean/prune/add-A or delegation. Apply/public/enrolled activation stays closed. **Return this clean candidate for separate Sol-low adversarial review of the entire final delta; no self-acceptance.**

## T1 reviewer transport F1 correction — frozen implementation handoff

2026-10-23 · implementation owner `gpt-6.1-sol`, low; no delegation. **F1 response implemented and frozen checks GREEN; no self-GO, integration, T2 or Phase 1 acceptance. A fresh separate Sol-low review must inspect the entire final T1 delta, followed by parent integration/equality checks and final integrated Astra.** This section supersedes no original requirement or qualification gate.

### Identity, replay and exact scope

- Fresh ABS `/Users/ben/.txcode/worktrees/py-audio-box/t1-close-f1-20261023`, branch `fix/t1-close-f1-20261023`, root base `eb1e5938c9c521f8f840cb8bcbd5328e86decdf6`. Root's reviewed-K executable prefix is unchanged (empty diff against `6130d1fecce3d978671ed1bfd509693d3aab3c97` over service/worker/scripts/tests/clients/contracts).
- Original T1 baseline `7ae446ccb81473cc82ae50ae5ea8305f1729eb4c`, source/tests `b91d51dc2bda5e7f013a6bd99426548e831a983d`, candidate `8294c946311701766e6acef2d733944d9aad548a`. Original ABS `t3-99d25777` remains clean at that candidate. Initial root checkout `t3-752f6771` remains clean at `eb1e5938c9c521f8f840cb8bcbd5328e86decdf6`.
- Inspected `git log --reverse 7ae446c..8294c94`; replayed ONLY its seven T1 commits, oldest first: `b2617b8→af984b9`, `d3b146b→a4cad3d`, `2373883→207882e`, `b3acb8e→d35e60b`, `8788ecd→5340a8a`, `b91d51d→173d14a`, `8294c94→2c1537d`. No conflicts, root-base/C imports, review or ledger replacement.
- Correction source/test commits: `084e67b6697b3ed610097e3cf3f58fd13c99a2c7`, then final frozen `35b2a28816fb41ec9a5c690812f144dad4abb883`. This handoff is the subsequent report-only commit; exact candidate SHA is supplied with the final handoff reply.
- Correction against replayed candidate `2c1537d`: recovery **+38/-13**, tests **+96/-0**. AST comparison against original candidate changes ONLY `_run_bounded_maintenance`, adds/removes no top-level definition. Two nested helpers factor only descriptor identity/close ownership. Existing candidate tests are an exact byte prefix; the explicitly approved legacy result-budget amendment is unchanged. Handoff is an exact candidate prefix plus this append.
- Root-base-to-final scope remains exactly `service/recovery.py`, `tests/service/test_recovery.py`, and this document. Root reviews, canonical ORCHESTRATION and ledger retain exact root-base bytes. `git diff --check` passes. No applicable ancestor/scoped AGENTS.md found.
- Final executable SHA-256: recovery `7da04c775428354543e431d2cd45c40e16fee9ff90f4d74805a8dc15fe21ab82`; tests `09d603555433148d2a081097091b851150688c5b24cf8dca3c6dfa28b3c81f68`. These bytes were checked before every final campaign and again after all campaigns.

Read the entire independent `P1B23-T1-r1-sol.md` including probes/source assessment, authoritative `P1B23-SUCCESSOR-TRANSPORT-ASTRA.md` including all seven exits, original SUCCESSOR and COHERENCE reviews, canonical ORCHESTRATION and ledger. No original brief, negative matrix, resource bound or pending gate is waived.

### F1 response and ownership audit

The parent's initial write-end close and result allocation now sit inside its ownership `try/finally`. Read-end cleanup has an independent nested `finally` for exact owned-child SIGKILL/waitpid reaping, so a close exception cannot bypass reaping. Fork-failure closes are independent through `try/finally`. The child's initial read-end close is inside the existing sanitized `except BaseException: os._exit(74)` boundary, before limit/session/operation entry.

`close_owned` releases its local fd slot before attempting close, including InterruptedError/KeyboardInterrupt/BaseException paths. An errored close is not blindly repeated: it checks whether the number still identifies the original pipe end using device, inode, mode AND `F_GETFL & O_ACCMODE`. Only a verified still-open original end gets a finishing close; EBADF and a reused file/opposite pipe end do not. Access mode matters because opposite pipe ends can share inode identity. A first development helper omitted access mode; the final source fixes that concrete reuse counterexample. This is descriptor cleanup only, never a retry of the maintenance operation.

Close errors still propagate as refusal; child errors are sanitized nonzero exit. No exception is converted into accepted result authority. A cleanup error can supersede another failure but cannot return success. The original single absolute 30-second read/EOF/normal-exit deadline, fixed T=1,048,576, bounded codec, normal exit checks and parent limits remain unchanged. There is no rollback inference, recapture, auto-repeat or service restart after lost transport, including possible durable completion before a transport error. The remaining codec/coherence/K/P/history/capacity implementation is byte-preserved outside this function.

Appended `test_t1_close_ownership_cleanup`: **64 cases** = two fresh repetitions × InterruptedError/KeyboardInterrupt × fifteen before/after/reused-file close-site combinations plus one opposite-pipe-end reuse combination. Sites cover parent initial close before reading, read-close during failure cleanup, each fork-failure close, child initial read close. Actual pipe/fork/fds, exact child reap, under-five-second cleanup, parent FSIZE equality, and a durable child-operation marker are observed. Child initial-close failures prove operation not entered and exit 74 already reaped. Reused sentinels remain open for their harness owner; all operation-owned fds are closed or demonstrably released. Harness `finally` closes its exact descriptors and kills/reaps only its own children even on RED.

The opposite-end case uses actual `dup2` and access modes with shared-inode identity explicitly modeled on macOS. All injected close failures are control-flow tests, **not claims of real kernel EINTR or Linux enforcement**. The existing single-threaded platform restriction remains. Source audit confirms BaseException closes cannot skip the independent obligations, while child BaseException remains sanitized refusal.

### RED, development failures, and superseded evidence

Common locked prefix for every pytest command below, cwd this ABS:

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS \
  PYTHONDONTWRITEBYTECODE=1 \
  TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache \
  /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider
```

Verified CPython 3.14.7, pytest 9.1.1, MCP 1.30.0, tiktoken 0.14.0. No install, substitution or hermesbridge/PYTHONPATH. Parent/pytest FSIZE was never lowered; observed before/after `(9223372036854775807,9223372036854775807)`.

Original reviewer source `/tmp/test_pab_t1_reviewer.py` remains SHA-256 `9c7f675658c60e087069ac17bf620d6722b7a8c00819cc285bb99cb9fd38d240`. `/tmp/test_pab_t1_f1_replay.py` changes only its imported candidate-test absolute path to this worktree (SHA-256 `7733969de62123640d3e8c4e1802931efe28d4f26704f20de3f09d4eecb9ba45`). Exact suffix `/tmp/test_pab_t1_f1_replay.py -k reviewer_parent_close_interruption -q -rs -s`, before repair: `/tmp/pab-f1-red.log`, **2 failed/1537 deselected, 0.45s, exit 1**; both record waitpid `(0,0)` and read fd open. Probe-owned cleanup ran.

Development receipts are retained and not substituted for frozen proof:

| Log under `/tmp/` | Actual result / correction |
| --- | --- |
| `pab-f1-focused.log` | Initial 60-case development run interrupted, shell exit 143; four failure indicators, no final summary. The cleanup-read fixture slept before producing a frame, preventing the injected read until the real deadline; changed that operation to return immediately. Its exact owned child had exited before terminating the remaining runner; no unrelated process stopped. |
| `pab-f1-focused-r2.log` | 60 passed/1533 deselected, 21.92s, exit 0, intermediate helper. Reuse fixture explicitly dup2s to the intended number when another free number is allocated. |
| `pab-f1-focused-final.log` | 79 passed/1514 deselected, 2.84s, exit 0, initial helper plus old actual-pipe/fork checks. |
| `pab-f1-reviewer-green.log` | 3 passed/1596 deselected, 30.44s, exit 0, intermediate helper; exact reviewer two repeats plus real-clock probe. |
| `pab-f1-pipe-reuse-red.log` | 4 failed/1609 deselected, 1.97s, exit 1; faulty probe assertion examined already-closed original read fd. Not a valid source RED. |
| `pab-f1-pipe-reuse-r2.log` | 4 passed/1593 deselected, 13.55s, exit 0; corrected assertion, ordinary macOS distinct-inode observation, not shared-inode proof. |
| `pab-f1-pipe-reuse-r3-red.log` | 4 failed/1593 deselected, 16.34s, exit 1; shared-inode fixture read its identity before initialization. Not source RED. |
| `pab-f1-focused-r4-final.log` | 4 failed/79 passed/1514 deselected, 4.78s, exit 1; same fixture initialization error. |
| `pab-f1-focused-r5-final.log` | 60 failed/23 passed/1514 deselected, 27.13s, exit 1; original fstat capture was scoped only to one parameter branch; moved capture before all branches. |
| `pab-f1-pipe-identity-old-red.log` | Corrected shared-inode/opposite-end probe against the exact `_run_bounded_maintenance` AST loaded from `084e67b` in a separate temporary process: **4 failed/1593 deselected, 0.58s, exit 1**, reused descriptor already closed. `/tmp/test_pab_f1_pipe_identity_old.py -k 'close_ownership_cleanup and pipe-reuse' -q -rs`. No repository rollback/edit for this probe. |
| `pab-f1-focused-r6-final.log` | **83 passed/1514 deselected, 2.79s, exit 0**, final source/tests; suffix `tests/service/test_recovery.py -k 'close_ownership_cleanup or actual_pipe_transport or fork_failure_closes_descriptors' -q -rs`. |

The initial freeze `084e67b` completed target **1264 passed/1 skipped/408 deselected, 2667.27s, exit 0**, log `/tmp/pab-f1-frozen-target.log`. Its four-B2 run was deliberately interrupted for the concrete shared-inode/opposite-end source gap: **1482 passed/1 skipped, 2620.07s, exit 2**, `/tmp/pab-f1-frozen-four-b2.log`. Full was not started on that freeze. `/tmp/pab-f1-campaign-receipts.json` and driver log retain those receipts. The access-mode correction is the specific new reason for repeating campaigns; neither run approves final bytes.

### Final frozen sequential receipts

Frozen source/tests `35b2a28816fb41ec9a5c690812f144dad4abb883`; all exits 0. `/tmp/pab-f1-campaigns-r2.py` executes the above prefix sequentially, records exact argv, wrapper time, log hash and source guards in `/tmp/pab-f1-final-campaign-receipts.json`. No executable/test edits during or after these runs.

| Final log under `/tmp/` | Exact pytest suffix | Actual result / pytest seconds |
| --- | --- | --- |
| `pab-f1-final-reviewer-frozen.log` | `/tmp/test_pab_t1_f1_replay.py -k 'reviewer_parent_close_interruption or reviewer_real_clock' -q -rs -s` | **3 passed/1600 deselected, 30.44s**. Both close-interruption cases: already reaped/read fd closed. Real-clock EOF-with-live-child refuses at **30.003946s**, then exact child reaped and FSIZE unchanged. |
| `pab-f1-final-target.log` | `tests/service/test_recovery.py tests/service/test_admission.py -k 'kfifth or kpre or lotk or lotp or b22 or successor or t1_' -q -rs` | **1268 passed/1 skipped/408 deselected, 2667.55s** |
| `pab-f1-final-four-b2.log` | `tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs` | **1730 passed/1 skipped, 2856.62s** |
| `pab-f1-final-full.log` | `tests/service tests/release -q -rs` | **1882 passed/19 skipped, 2931.00s** |

Fresh campaign coverage retains the full T1 seven-exit matrix: actual nonempty A/B/C, finite D/E capacity refusal, strict joins/alias defense, closed codec and transport, K/P/history/no-attempt and all-current-source preservation. The original review's 308 passes and prior 1818/19 full result remain attributed historical evidence, not these new results. Six actual two-setup witness roots from final target/four/full (pytest-77/78/79) are copied in `/tmp/pab-f1-final-production-witnesses.json`, including generation sizes/bounds/observed maxima and first refusal E at unchanged `b2_prefix_snapshot_budget`. For final full setups 0/1, A/B/C frames are **36617/94319/222909** and **36611/94316/222903** bytes. These are the passing tests' actual recorded metrics, not an additional independent population audit or universal capacity guarantee.

Final SHA-256 receipts:

- reviewer log `b1036a0cded9f38c927830c16d290630a0fe69bad2fd2312da37b1486505eb12`;
- target log `e0095e2b5114f4918d4186fc3cda5f18888455cb96e3115a34883de447410f88`;
- four-B2 log `18db94bd0dfa26448431ecf15803f5613cd100c924a67b5f43e36cab77c30532`;
- full log `9b2a71681535178eed0fc587575393302cf8b5976028a36f9669eaff5dc9f810`;
- campaign receipts JSON `48f733809868f229dd708216db2470d26b51f5d5560654886872ccb9d19ff9f1`;
- source guards JSON `/tmp/pab-f1-final-source-guards.json` `a51799e7898c4187bc653c2e381e8ddc5b094ac872ab69ddb1b203410ddb1d1a`;
- production witnesses JSON `12233a233093a58fa4908d4d26b2897704c846304a1d177a50e02b8db09171af`.

### Preservation, limits and next owner

Final source guards verify clean unchanged historical pins: `t3-f0f283c4=d03bc47472913880d32fd2e8b1b738806ade01a0`, `owning-successor-coherence=6cf6236bd42750cd82a9f680fac18d7f523bbe51`, `t3-1e1cddba=a27f78b5e2e09a98940b9c543474337b511ce616`, `t3-adbf212d=58e3eb9f40a8db5ca162fe8f64922efb13f224c4`, `t3-4a8727d4=6cfe3973f41db92679531e38660d88cf72ce27eb`, plus original T1 and root pins above. No old WIP/research/root docs were edited. Task-owned runners finished; regressions/probes explicitly release their own descriptors and reap their own children. Temporary evidence is retained; no reset/clean/prune or evidence deletion.

All **19 Linux mechanisms remain UNQUALIFIED**, including actual Linux 4 MiB/ext4/Unix-VFS/SHM and peer credentials. C1 host pin/provenance/authentication and mandatory `b2_pinned_launcher_context_unavailable` production refusal remain. Installed C1/C2/clean-host, B3, physical durability, hardware/browser/listening/acoustic and final global Astra remain pending. A1 Docker-inventory drift remains unexplained and unqualified; no inferred cause or waiver. Apple Silicon remains documentation-only qualification scope. Host freeze/public activation boundaries remain unchanged; no false freeze release, rollback inference or automatic restart.

No delegation, Docker/network/model/audio/install/paid E2E, C/script/worker/resource/schema/dependency changes, push/PR/deploy, other-worktree edits or parent-limit lowering. T2 is not implemented here. Next: separate Sol-low independent review of the **whole final delta**, then parent acceptance/integration/equality/focused checks, fresh T2 and final Astra. **No self-GO.**


## T1 round-2 F2 kill/reap interruption correction — frozen implementation handoff

2026-10-23 · implementation owner gpt-6.1-sol / low, no delegation. **Correction implemented with fresh local GREEN receipts; NO self-GO, integration, T2, host or Phase 1 acceptance. Parent must arrange fresh round-3 separate Sol-low adversarial review of the entire final T1 candidate before integration, then source equality/focused integrated proof, fresh T2 and final integrated Astra.** All original seven T1 exits and qualification limits remain binding.

### Identity, replay and exact scope

Fresh ABS `/Users/ben/.txcode/worktrees/py-audio-box/t1-f2-cleanup-20261023`, branch `t1-f2-cleanup-20261023`, root base `eb1e5938c9c521f8f840cb8bcbd5328e86decdf6`. Original T1 baseline remains `7ae446ccb81473cc82ae50ae5ea8305f1729eb4c`. Preserved prior candidate `bdf9a6920964acd72ef108b2bf7516796f37e4be`, executable freeze `35b2a28816fb41ec9a5c690812f144dad4abb883`, remains clean at `t1-close-f1-20261023`.

Inspected root-base..candidate log, then replayed ONLY its ten T1 commits oldest first, no conflicts: `af984b9→878a278`, `a4cad3d→a30e0ec`, `207882e→8d44504`, `d35e60b→db43813`, `5340a8a→0484e4d`, `173d14a→00e8883`, `2c1537d→b836968`, `084e67b→f4ef343`, `35b2a28→1c44a3f`, `bdf9a69→5ccf5a9`. Replayed tree equals exact bdf9a692 before correction. No root-base/C imports or replacement of reviews/ledger/research. Root-base executable paths still equal reviewed K `6130d1fecce3d978671ed1bfd509693d3aab3c97`.

Correction source/test commits: `6f703be88489d9cccc0a86c6354f5f03184a5679`, then final freeze **`61d391c1ccf33aad1b37d7a7502692c7363c63d5`**. This append is the subsequent report-only commit; exact final candidate SHA is supplied in the final reply. No executable/test edits during or after the final campaigns.

Against bdf9a692: recovery +67/-8, tests +171/-0 before this report. AST comparison changes ONLY `_run_bounded_maintenance`, adding no top-level definition and removing none. Existing tests are an exact byte prefix, including all 64 F1 close cases and the originally approved legacy result-budget branch. This document is append-only. Root-base-to-final writable scope remains exactly `service/recovery.py`, `tests/service/test_recovery.py`, `docs/agent-interface/phase-1/RECOVERY-P1B2.md`; all other paths, including canonical docs/reviews/ledger, remain root-base bytes. No applicable ancestor/scoped AGENTS.md found; diff-check passes.

Read entire R2 and original R1 T1 reviews, exact F2/probes/source assessments and original F1, authoritative TRANSPORT-ASTRA/all seven exits, original SUCCESSOR and COHERENCE reviews, canonical orchestration/implementation plan and pertinent ledger chronology, and complete prior T1/fix handoffs. Memory supplied only checkout/evidence-boundary guidance, not acceptance evidence.

Final executable SHA-256: recovery `5540c380fa6a44e5a18ca9ffc81c77d7cb8b0a31eba3136eebea6d298e2c55d0`; tests `f026142e5024675e8e577f3add8729f4d9fc9da5b726f71ee000f2ec16e9d969`.

### F2 response and boundary audit

The cleanup factor retains the local owned PID until successful exact-child wait or ECHILD resolves the obligation. Before EVERY cleanup signal, exact `waitpid(pid,WNOHANG)` checks ownership: a reaped child clears PID; ECHILD clears it without signalling; an interrupted probe is resolved by another probe. An unreaped child/zombie retains its PID. A failed-before-kill interruption does not skip wait: independent nonblocking reap is attempted, then ownership is rechecked before another signal. A successful SIGKILL is followed by wait; an interruption after actual kill or before/after actual wait is resolved through the next ownership probe. No signal follows a lost successful reap. ESRCH alone is not a reap receipt: wait remains required. This is single-threaded child ownership, not an API for competing SIGCHLD reapers.

Cleanup retains its first interruption, completes the supported cleanup path, then propagates the invocation's original refusal/interrupt in preference to cleanup errors. Explicit invocation-local exception tracking prevents an already handled caller exception from suppressing a new cleanup refusal. A cleanup failure on an otherwise successful result cannot become success.

Normal WNOHANG EINTR-before-syscall remains retryable within the original absolute deadline. If a wait actually reaped and its result was lost to InterruptedError, subsequent ECHILD clears PID and propagates that original interruption: lost exit status cannot authorize success. Normal KeyboardInterrupt/BaseException also reaches independent descriptor/child cleanup, including after actual reap. No operation retry/restart, result/freeze-release success on refusal, or rollback inference is introduced.

The original absolute 30-second operation/read/EOF/normal-exit deadline is not reset or extended. Cleanup has a finite 64-round interruption guard rather than the former unbounded EINTR loop; blocking wait is used only after SIGKILL/ESRCH, never after a failed-before-kill. No global interrupt suppression. This does not claim successful cleanup against endless external interference or a hard wall-clock bound for an uninterruptible kernel syscall; exhaustion remains refusal, never authority. Actual Linux syscall/kill latency qualification is still pending.

F1 descriptor identity/access-mode and independent close/fork cleanup remain unchanged: initial parent close inside ownership try; child initial close sanitized before operation; independent read-close and child cleanup; no blind close of a released/reused fd. Audit covers cleanup ownership probes, kill, wait, normal WNOHANG lost results, ECHILD/ESRCH, original-exception precedence and caller exception context, alongside the existing close/fork/child boundaries.

Appended 53 cases: 48 = two repetitions × InterruptedError/KeyboardInterrupt/BaseException × before/after actual syscall × kill/cleanup blocking wait/cleanup ownership probe/normal wait; two ESRCH/ECHILD controls; three caller-handler cleanup-refusal controls. Actual fork/pipe/fds, exact already-reaped assertion before escape, actual unrelated sleeping child, no signal after an injected actual reap (PID-reuse countercase), fsync-completed operation marker before cleanup and exactly one operation entry, descriptor closure, elapsed bounds and unchanged FSIZE. Each new harness owns finally cleanup, probes before signalling, and releases only its own children/fds even on RED. These are fault-injected E1 control-flow tests, not real Linux EINTR qualification.

### RED/GREEN, actual development failures and commands

Common exact locked prefix, cwd this fresh ABS:

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS \
  PYTHONDONTWRITEBYTECODE=1 \
  TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache \
  /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider
```

Verified CPython 3.14.7, pytest 9.1.1, MCP 1.30.0, tiktoken 0.14.0. Parent FSIZE before/after and campaign receipts `(9223372036854775807,9223372036854775807)`; no lowering/install/hermesbridge/environment replacement.

- Exact reviewer `/tmp/test_pab_t1_r2_negative.py` replay changes ONLY the imported test path, saved as `/tmp/test_pab_t1_f2_replay.py`. Baseline suffix `... -k test_r2_cleanup_reap_interrupt -q -rs -s`: `/tmp/pab-f2-red.log`, **4 failed/1603 deselected, 0.58s, exit 1**. Both kill-before children live/unreaped; both wait-keyboard children killed but waitable. Harness cleanup ran. Initial correction: `/tmp/pab-f2-reviewer-green.log`, **4 passed/1603 deselected, 0.44s, exit 0**; final frozen replay below supersedes it.
- `/tmp/pab-f2-focused.log`: **23 failed/118 passed/1506 deselected, 25.83s, exit 1**. Fixture observed marker existence before buffered child contents flushed, so cleanup could kill before marker contents became visible. Corrected only new fixtures to flush/fsync marker, then publish separate ready marker. `/tmp/pab-f2-focused-r2.log`: **141 passed/1506 deselected, 3.97s, exit 0**.
- Source audit exposed first correction's `sys.exception()` caller-context defect. `/tmp/pab-f2-caller-context-red.log`: **3 failed/1647 deselected, 2.11s, exit 1**, new cleanup exception incorrectly suppressed. Fixed with invocation-local primary-exception tracking, preserving these exact tests. First campaign driver was interrupted before broad stages; `/tmp/pab-f2-campaign-driver.log` has KeyboardInterrupt and no completed receipt, `/tmp/pab-f2-final-reviewer-frozen.log` has no completed summary. Neither is counted as proof. The final freeze below is a new source reason, not a blind broad repeat.
- Final focused suffix `tests/service/test_recovery.py -k 'reap_interruption_ownership or cleanup_missing_process_receipts or cleanup_refuses_inside_caller_exception_handler or close_ownership_cleanup or actual_pipe_transport or successful_exact_size_pipe_waitpid_eintr or fork_failure_closes_descriptors' -q -rs`: `/tmp/pab-f2-focused-final.log`, **144 passed/1506 deselected, 4.03s, exit 0**, including every F1 close case and all new regressions.

### Final frozen sequential receipts

`/tmp/pab-f2-campaigns-frozen.py` checks clean status and exact executable hashes before/after each stage; `/tmp/pab-f2-frozen-campaign-receipts.json` records exact argv, wrapper seconds, exit, log SHA-256 and source/limit guards. All stages exit 0, no failures/xfails, executed sequentially on 61d391c:

- `/tmp/pab-f2-frozen-reviewer-frozen.log`: **7 passed, 1653 deselected in 30.50s**; wrapper 31.264s. Exact suffix: `/tmp/test_pab_t1_f2_replay.py -k test_r2_cleanup_reap_interrupt or reviewer_parent_close_interruption or reviewer_real_clock -q -rs -s`. SHA-256 `5ba7ac2957d88c34d50eb8b37760a52ee8e767dfd71f2b3ce55fe28310383448`.

- `/tmp/pab-f2-frozen-target.log`: **1321 passed, 1 skipped, 408 deselected in 2672.48s (0:44:32)**; wrapper 2673.467s. Exact suffix: `tests/service/test_recovery.py tests/service/test_admission.py -k kfifth or kpre or lotk or lotp or b22 or successor or t1_ -q -rs`. SHA-256 `ebe968dbfedfb1402b9fe59370cf740918b538ecbed030543482668fb5478d50`.

- `/tmp/pab-f2-frozen-four-b2.log`: **1783 passed, 1 skipped in 2844.96s (0:47:24)**; wrapper 2846.379s. Exact suffix: `tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs`. SHA-256 `e2eea2b53f6b077bfee4ec69dcb0dff6ec34d8c2d540a1638a81bccb9eb443fa`.

- `/tmp/pab-f2-frozen-full.log`: **1935 passed, 19 skipped in 2888.57s (0:48:08)**; wrapper 2890.036s. Exact suffix: `tests/service tests/release -q -rs`. SHA-256 `365c20fee196d9015fecfaed51af4e468476a37cda645fc3eb67984ca92b3031`.

Frozen reviewer replay includes original F1 twice (already reaped/read fd closed), all four R2 REDs now already reaped, and real-clock valid-frame/EOF-with-live-child refusal at **30.006412633s**, followed by exact reap and unchanged FSIZE. Original reviewer files are preserved. All 64 close cases are fresh in focused/target/four/full.

The target/four/full retain the whole seven-exit proof: actual production A/B/C and finite D/E, closed canonical whole 1MiB frames and bounded buffers, all EOF/exit/deadline cases, independent existing storage/population/capacity guards, strict exact completed-history/current tuple joins, and all owner/no-attempt/current-before-any-unfinished-write/F1–F4/history/session/locks/Unix-VFS preservation. Outside cleanup plumbing, original codec/coherence/K/P/capacity source is byte-preserved. No old test definition or approved legacy branch changed. Prior **1882 passed/19 skipped** remains attributed predecessor evidence, not this run.

Six current production witness roots (pytest-85/86/87, two per campaign) copied into `/tmp/pab-f2-frozen-production-witnesses.json`. Full-run setup 0/1 both record A/B/C frames **36,617 / 94,319 / 222,909** and snapshots **249,856 / 294,912 / 405,504**. Existing tests assert original snapshot/archive bytes, complete descriptors/Core oracle and fifteen steps/three histories before finite continuation. All six complete D then first refuse E at unchanged `b2_prefix_snapshot_budget`, with five plans/four recoveries/twenty steps/four resolutions, original receipt/outcome null/ACK=0, retained charges 2,097,152, H=20,971,520, disk budget 67,108,864/free reserve 16,777,216. Read-only post-run audit freshly checks those SQL counts and original receipt/outcome/ACK plus recorded limits. These are implementer executions/recorded observations, not independent review or continuous physical/RSS/Linux enforcement maxima.

Post-run audit development had two script assertions: pytest current symlinks double-counted roots (9 instead of 6), then complete budget mapping was incorrectly compared to a scalar. Corrected only temporary inventory/audit code; final `/tmp/pab-f2-witnesses.py` exits 0 with six unique roots. Notes `/tmp/pab-f2-witness-audit-development.txt`; no production/test changes or campaign repeat.

Additional receipt SHA-256:

- `/tmp/pab-f2-red.log`: `b5748d9500407899147bb42b0de7cfa6350f74f72866f547964b84544a872444`.

- `/tmp/test_pab_t1_f2_replay.py`: `1ab75403cfa2496d822b5325ae65aa2068b09bc251448fb46dda57853a62df4d`.

- `/tmp/pab-f2-focused-final.log`: `be9e5abf07e22fe09e796dfef782d5a4b3f42438e69aa98c6dfea7f8aa7eed50`.

- `/tmp/pab-f2-frozen-campaign-receipts.json`: `576c7b062b6c91c2777ea6160231d7d399961562ddfc04c1cdcacce7236961ac`.

- `/tmp/pab-f2-frozen-guards-before.json`: `4169b036ce9a47e492729290d4870ed12c0c6ca5cf63053097603660c832f955`.

- `/tmp/pab-f2-frozen-guards-after.json`: `4169b036ce9a47e492729290d4870ed12c0c6ca5cf63053097603660c832f955`.

- `/tmp/pab-f2-frozen-production-witnesses.json`: `072d69cc22e6eb35073eb130040833f70f549490c6c08166b15102ce5925bc9b`.

### Preservation, limits and parent handoff

Before/after guard JSONs match byte-for-byte. Read-only historical inventory confirms exact clean pins: prior F1 `bdf9a692`, original T1 `8294c946`, original WIP `d03bc474`, coherence `6cf6236b`, C `a27f78b5`, historical C `58e3eb9f`/`6cfe3973`, reviewed K `6130d1fe`, root-base checkout `eb1e5938`. No historical WIP/research/review/ledger mutation. Final campaign runner exited; new harnesses assert exact owned reaping and preserve unrelated children. Temporary logs/probes remain; ordinary pytest retention is unchanged, no manual evidence cleanup.

All **19 Linux skips remain UNQUALIFIED**: one actual RLIMIT_FSIZE and eighteen peer-credential mechanisms (10+1+1+6). Actual Linux 4MiB/ext4/Unix-VFS/SHM, C1 pinned launcher/provenance/authentication and mandatory production refusal, installed C1/C2/clean host, B3, physical durability, hardware/browser/listening/acoustic and global Astra remain pending. A1 whole-inventory drift remains unexplained/unqualified; no inferred cause/waiver. Apple Silicon is documentation-only qualification scope. Host freeze/public activation stays closed; no completion/replay/transport failure grants automatic restart, rollback inference or current freeze release.

No delegation, integration, Docker/network/model/audio/install/paid E2E, push/PR/deploy, reset/clean/prune/add-A, other-worktree mutation or resource/schema/dependency/C/host/worker change. **Return clean candidate for parent-owned fresh round-3 review of the entire final T1 delta; no self-acceptance.** T2 is not this assignment.


## 2026-10-23 — Family P source-only implementation handoff

Implementation prepared; independent source review and every runtime gate remain pending. No GO or runtime pass is claimed. Parent explicitly adopted both complete Astra contracts, with revised S superseding old S; this lane implements P only. Both entire reports were read. H/S/A/R are not implemented or redesigned here.

Provenance and freeze:

- Worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-1cebb715`, branch `t3/family-p-durable-completion-witness`.
- Root entry: `5ef14a18ba27e57dac6b74a30c086468060d4a07` (root T1 documents, not the later source candidate).
- Immutable source: `c224a4aaedf4c15765c8a19540a77f2f221a59c5`, ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-eb791eef`; historical executable freeze `65eef52fe7579b7d357719134a04c47fcb384b2e`. Historical worktree was read only and remained clean at the exact source HEAD on final inspection.
- Reconstructed baseline commit: `fbe4f111934ee49b280da0c7a324051cc214c82e`. All 426 committed non-document paths (exclude docs/ and *.md) were reconstructed by Git object bytes and explicitly compared equal before P edits. SHA256 of the ordered `sha256 + two spaces + path + newline` baseline manifest: `f83839103f046acb8b89af1f1eea6e56cc52d6f2fabd516f041f925ef7156ae9`.
- Frozen P executable/test commit: `d00f49f9a16f3f0eac11ba4c62622ec1278cc176`. Its delta from reconstructed baseline is exactly the three allowed test files. Root documents are unchanged through that freeze. The following commit is this append-only handoff, with no executable change.
- Final non-document diff against c224a4 names only tests/_maintenance_wait_guard.py, tests/service/test_recovery.py and tests/service/test_child_wait_guard.py. No production, dependency, conftest, aggregate, bootstrap, selector, receipt schema or process-management amendment belongs to P. Reconstruction is separately identified and must not be confused with new family delta.

Concrete P implementation:

`DurableLostResultWitness_P` and scoped install hooks capture the original production runner, original continuation, exact _prepare_abandon lambda, actual collected finish_then_lose and broken code objects. The parent's P armed flag and P:a; precede child execution. Missing collected identity or wrong installed code fails the armed invocation; arbitrary outer-name replay helpers do not arm. The actual production lambda and installed function are independently checked. The captured writer emits seam/final markers despite the child's os.write patch.

A scoped trace observes only the runner, continuation and collected finish frames. It binds the actual prepared snapshot/session/intent, original continuation's complete result with matching recovery_id, one SQL recovery with the exact intent ID and five real SQL steps, and the durable-completion sentinel before accepting the result-write seam. No extra SQLite connection is opened by the observer. Parent target retains every original assertion and adds snapshot consumption, ID shape and exact SQL identity assertions. The real _kfifth_fresh replay remains exit0 with byte/logical equality assertions intact.

The seam requires exact built-in BrokenPipeError, args ('modeled lost result pipe',), errno None; actual captured runner os.write CALL offset and collected broken RAISE_VARARGS offset; exact runner frame; enrolled result-pipe device/inode/mode/access identity distinct from witness pipe; bounded bytearray frame, integer offset and actual memoryview owner/slice and FD arguments. Real EPIPE, another pipe, reused FD, file or operation failure cannot satisfy that seam. At runner catch-all74, sys.exception() must be the retained actual object with unchanged attributes and the exact two ordered (actual frame, tb_lasti) pairs. Only the last captured _exit CALL site is final authority. There is no traceback, payload, SQL value or path serialization.

The armed parent veto runs before legacy handling, including for exit0, expected_exit74 and synthetic signal11 with otherwise valid tokens. Closed token parsing rejects missing, duplicate, truncated, reordered, unknown and overflowing evidence. One exception/frame route is held per invocation and cleared in finally; the previous trace hook is restored. Existing R class/predicates/tokens remain byte-exact. Existing kernel wait, enrollment, latch, single reaper, session stop and helper ownership machinery are unchanged.

Static token bound: P:a;/P:s;/P:f; are 4 bytes each, total12. Existing mutually possible semantic markers V (1), V:session; (10), V:operation; (12) add23, so this P invocation is at most35 bytes. Reserving the same 12-byte triples for both future H and S yields 12*3+23=59 <=256. R/J/T/C/D declarations have different exact nodes and cannot emit on this P invocation; legacy marker deduplication is unchanged. This is a static bound under the adopted short-token convention, not a claim that other family implementations have been reviewed. Explicit overlap/token-bound review is still required at combination.

Static verification performed, without candidate imports:

- Standard-library ast.parse and compile(source, filename, 'exec') on the three changed files; AST parse of the embedded fresh-helper source. No generated code was executed and no pyc was written.
- AST-delimited exact-byte comparison of every preexisting out-of-scope top-level definition and every existing top-level non-definition statement in order. Only guard observe/install and the named P target differ; R and all unrelated recovery bodies remain identical. Guard tests preserve their complete original file as a byte-exact prefix. Original assertion AST multiplicities are retained.
- git diff --check; source scope/name comparison to c224a4; exact historical HEAD/clean inspection. Static checking script was `/tmp/check_family_p.py`, output `/tmp/family-P-static.json` (ephemeral convenience files; complete durable results below).
- One static syntax typo was found and corrected before freeze. A hash command initially addressed absent root requirements.lock; corrected pin is service/requirements.lock below. Neither command imported candidate code or exercised a runtime.

Frozen hashes (SHA256):

| Object | Old c224a4 | New frozen P |
|---|---|---|
| tests/_maintenance_wait_guard.py | `4842c9c9881841c876f48dea801b3e09862723da1a6b281c0df3de2d4546eaee` | `c28448dfca0406157bd07a1f768e211791ca285569cc251d7f69d9ac828af877` |
| tests/service/test_recovery.py | `93297423794a1abcf496d898332129c0596e51d55ad50494c8a84cfe86c064ac` | `c236e4045a0c35836fc269fd5938c053ab3e97f1f2682f023470afe3a68dd211` |
| tests/service/test_child_wait_guard.py | `6424d288a7bea6af55a4fc258d2252dffd2837f9dff2403c940cb355761abc18` | `cffbcf2e9bd75412e30f53be221b2ac58dbabc94f5653324db9f5b6bbb2ba62c` |
| Exact P target definition, including source newline | `1bef1910986243e5fe75f468e1d616b35973fd30ac31da3dd3b9002d6386e474` | `11725409583de8133a93de2e8db0135250e76f5e0fe7048f97119c4c0c67393e` |

Unchanged pins (SHA256):

- service/recovery.py: `c314b2423b71c7d205b4558aee44d8c895e41394adbbb05a85458ef9bd7acf29`.
- service/requirements.lock: `4ad7f2d9e6d4f03b7abb22d4946524f4372b5277b7b80c186332720f490909ea`.
- tests/conftest.py: `d967baee72fa7fba3cb8974ac0406b2f44231a1c9d49fb7e7c088016f07cc7a3`.
- poetry.lock: `2885f7f62d5a4b4a483e001622ba1d0d5ec4110578971664ecced01fbdfeb553`.

Finite future P matrix — static enumeration only, not collected/executed counts:

| Selection | Static cases | Purpose |
|---|---:|---|
| Original test_t1_durable_completion_lost_result_no_false_success | 1 | Real durable completion/lost result, SQL1/steps5, real fresh observational replay |
| test_P_ordered_armed_transcript | 11 | normal; missing; no-seam; no-final; final-only; duplicate; truncated; overflow; wrong-order; exit0; synthetic signal11 |
| test_P_durable_lost_result_fresh_contract | 30 | normal real fixture/fresh helper plus the directed cases below |

Fresh cases: normal; wrong-class; subclass; wrong-args; real-EPIPE; wrong-code; wrong-node; wrong-collected; precompletion; durable-id; missing-sentinel; wrong-fd; reused-fd; witness-pipe; operation-failure; cleanup-value; cleanup-nonsemantic; cleanup-new-object; changed-route; changed-attributes; wrong-final-caller; bad-offset; bad-slice; frame-budget; pipe-mode; wrong-operation-code; missing-markers; signal11; foreign-owner; missing-done. The FD controls actually replace the enrolled numeric FD with a file, unrelated pipe or witness pipe. Cleanup controls raise/unwind at a test-owned final-observer cleanup boundary after seam capture, including ValueError with V marker, nonsemantic error and a fresh same-type object; they do not fabricate serialized receipts. Changed-route reraises the same object through extra actual frames. These controls and code paths have NOT run. Synthetic observer11 is explicitly synthetic and never native signal evidence. Foreign owner and missing-done exercise the existing fresh ownership channel; the fail latch and refusal of another launch are checked outside the target's pytest.raises.

Required next steps, not authorized/executed by this handoff: parent independent frozen-source review first; then separately authorized finite P runtime/actual-receipt review with the real positive and replay, directed controls and required preservation selection. Combine only frozen family commits after explicit overlap review (observe/install are intentional shared edit sites); then independent full combined source review/full grouped gate. No delegation, integration, pytest/collection, candidate import, Docker, SQLite open, native experiment, runtime lane or production action occurred here. No owned runtime resource was acquired. Historical clean worktrees, STT untracked artifacts and raw proof were preserved. No synthetic case is counted as an actual positive, no tests are called passing, and no implementation self-GO follows from static checks.

Mandatory C1 b2_pinned_launcher_context_unavailable, native Darwin causes/controls, A1 drift, all platform/19 mechanism and installed-durability obligations, T2/C1295, four-B2/full service/release, final global Astra, Apple-Silicon/browser/acoustic and activation remain pending. The revised S late genuine-completion capacity gap remains the other lane's explicit coverage gap; P does not resolve it or historical PID161/PID255 exceptions.

## 2026-10-23 — Family P SOURCEONLY correction round 2

**Implementation handback for independent source review; runtime remains NO-GO.** Read the entire P-WITNESS-SOURCE-r1-sol review (original brief/authority, F1–F3, complete control census, token argument and unresolved risks), original GROUPED-WITNESS-CAPACITY-ASTRA and S1-CAPACITY-REVISED-ASTRA contracts, the complete history-refusal audit/census and preceding P candidate handoff. The revised S contract supersedes old S only in its own lane. This round corrects P only; no H/S/R amendment, dependency or production change is authorized or made.

### Provenance and executable freeze

- Fresh ABS `/Users/ben/.txcode/worktrees/py-audio-box/p-source-r2-20261023`, branch `fix/p-source-r2-20261023`, reconstructed by `git worktree add` directly from committed candidate `478953d0cfa500d849c5222260a5fef05a8db2c8`. No uncommitted source was copied. Its inherited executable freeze is `d00f49f9a16f3f0eac11ba4c62622ec1278cc176`; historical source remains `c224a4aaedf4c15765c8a19540a77f2f221a59c5` / executable `65eef52fe7579b7d357719134a04c47fcb384b2e`.
- All426 committed non-document paths (excluding docs/ and *.md) are inherited by exact Git mode/blob from the candidate. SHA256 of sorted `mode + space + blob_oid + space + path + newline` baseline manifest: `d1e5a82a90b6fcbf3256de4d93874e97dca730f9ca06b99ec81ef9038ea94494`. This format differs from the first implementation's content-hash manifest; they are not interchangeable hashes.
- New executable/tests freeze: `8adc9c640ffbeb312bb6a23b226b0635722d3f94`. Exactly two paths differ from 478953d: tests/_maintenance_wait_guard.py and the appended P fresh control in tests/service/test_child_wait_guard.py. The target file remains byte-identical to 478953d. This following append is docs-only; no later executable edit is included.
- Historical ABS t3-1cebb715 remains clean at exact478953d; t3-eb791eef remains clean at exactc224a4. This lane did not write historical worktrees, retained proof, STT artifacts or other family trees. Entry checkout t3-9c8161e8 at f8612c3 was not used as the source candidate and was not modified.

### Finding responses and source locations at the executable freeze

**F1:** guard:277–310 separates designated invocation eligibility from validity. Exact node/name designation arms even when actual operation code, installed finish, collected object or node identity is invalid; these facts invalidate evidence instead of permitting legacy V. guard:315–339 positively excludes only the checked replay: installed original preparation function identity, actual module-owned operation identity/code constant and complete resume code equality to the unchanged `_KPRE_FRESH` source literal, compiled with the actual fresh bootstrap's one-line prefix. Missing/malformed source gives no exemption. This check is P-scoped and runs before pipe acquisition (guard:522–528); it executes no compiled helper code. Parent P:a/armed flag still precede the original runner (guard:549–550). Original seam operation validation remains intact (guard:670–673).

Control: test_child_wait_guard.py:670–744 changes the **actual outer callable** for operation-failure and new operation-value, rather than changing only witness.prepare_code after arming. operation-value raises early ValueError with real observed_operation/observed_session V phases; the exact expected transcript is P:a;V:operation;V:session;, raw74 rejected, no seam/final, no continuation and a latched next-launch refusal. Wrong installed code/node/collected identity also require armed-invalid evidence; the latter two retain full real durable prerequisites and fail their designated contract. The normal control requires the real target plus both real `_kfifth_fresh` replay subprocesses, each unarmed/empty phase/exit0 and matching ack/done.

**F2:** test_child_wait_guard.py:573–635 uses a test-owned post-seam unwind around the genuine original seam predicate. After the real BrokenPipeError has passed unchanged production seam validation, it emits the genuine seam token then raises through the trace callback into the **original runner catch-all**. ValueError, RuntimeError, a new BrokenPipeError object and the same object with an added actual route are distinct cases. test_child_wait_guard.py:636–663 audits the object/caller supplied by unchanged observed_exit, additionally checking `sys.exception() is exc`; cleanup cases do not substitute predicate arguments. Exact final replacement identity and changed-route inequality are asserted, final must be absent, raw74 rejected, and subsequent maintenance/fresh launch refused. Same-object changed route is expected to REJECT, never accept. Positive original exception/route predicates remain unchanged.

No manually forged V is emitted by cleanup controls. At this post-result-write seam the session and observed_operation have already exited, so their legacy ValueError observers are not on the unwind route. Actual cleanup ValueError therefore has P:a;P:s; with no V; the separately real operation-value case supplies V-producing armed rejection. A pre-result session cleanup fault is not counted as post-seam coverage. The deliberately predicate-only wrong-final-caller case remains labelled; changed-route supplies its separate real-route representative.

**F3:** test_child_wait_guard.py:518–525 and549–873 require bounded case-specific evidence, not any guard failure. Each helper has exactly two private test-only records, at most24 unique labels each and each published through the existing <=4096-byte atomic writer. Records contain only mode/closed phase labels; no payload, SQL data, exception serialization or paths. These records do not modify receipt schemas. Exact ordered parent/child label lists prove reached, actual mutation/readback, seam/final rejection or acceptance, reference clearing and hook restoration. A bootstrap/assertion error cannot meet those lists. Parent runner refusal events are explicitly excluded from the child's seam control (581–585).

FD controls (678–713) verify original enrolled device/inode/mode/access and original runner ownership map, distinct unchanged witness identity, actual dup2 replacement identity and dev/inode change, access mode, saved old identity, replacement type, and temporary-FD closure via actual EBADF. pipe-mode replaces the actual number with an O_RDONLY pipe end; it no longer changes only a saved expected tuple. No pipe is consumed and no wait/reaper/signal is added. Error factories preserve the actual collected broken code and prove the valid completed session/sentinel, original result arguments/frame/slice and FD prerequisites before mutation. The real-EPIPE case is explicitly a constructed errno-shaped exception, not an actual kernel EPIPE.

Predicate-only bad-offset/frame-budget/bad-slice/durable-id/wrong-operation-code first prove the unchanged seam would accept, restore its retained exception/route fields, then mutate the live fact, assert readback and demand predicate rejection. They are not described as full integration faults; actual outer-operation, precompletion, real FD, missing-sentinel, type/args and post-seam routes remain separate. bad-slice checks the actual traceback frame's argument memoryview and owner. Precompletion replaces the continuation before completion and requires its own entered/unused-prepared evidence; outer-operation failure replaces the actual outer callable and requires no continuation entry.

Exact terminal count/phase/status/acceptance and continuation-sentinel expectations are asserted at818–837. foreign-owner requires a complete same-child enroll/terminal pair with exact foreign owner and exact `foreign helper event owner` aggregation reason (838–844). missing-done requires the precise FileNotFoundError from the unchanged aggregator for this done.json and exactly the corresponding outer failure (845–852). Observer-only missing-markers and status11 require valid original tokens/raw74 before substitution and are labelled synthetic; no signal is generated. Parent hook/trace restoration and clear_P reference clearing are explicit (664–669,773–782). missing-done deliberately keeps its helper-local done suppressor through bootstrap atexit, as required for that aggregation negative; it is not an ambient parent hook.

The prior appended P normal control incorrectly expected only two aggregate waits and two local terminals. Source `_kfifth_fresh` launches two separate replay helpers; local events contain only the original P terminal, and nested directories contain the two replay terminals. This round corrects those two control-count assertions to exact1 local terminal /2 nested terminals /aggregate [raw18944,0,0], and adds ack/done/phase/status checks. This is correction of the unexecuted P control's census, not removal or weakening of an original recovery assertion. Every historical original assertion and all prior non-P guard controls remain intact.

### Static verification and source preservation

Only standard-library AST parsing, `compile(...,'exec')` without executing the resulting code or writing pyc, code-constant comparison, SHA256 and Git inspection were used. No pytest, collection, candidate import, SQLite open, Docker, runtime/native experiment, integration or delegation occurred. No owned runtime resource was acquired. Temporary static scripts/output are `/tmp/check_p_r2_source.py`, `/tmp/p-r2-static.json` and the prior `/tmp/check_family_p.py`; these are convenience evidence, not runtime receipts.

Comparisons against478953d preserve byte-exact all267 recovery definitions and the entire recovery file, all14 other guard-control definitions, all12 other guard top-level definitions/classes, and every top-level non-definition statement/assignment in each file. Against c224a4, the guard-control historical file remains an exact byte prefix, all266 unrelated recovery definitions/statements and every original assertion AST multiplicity remain; only the original P target has its already-frozen additive assertions. R's entire witness class, MaintenanceWaitGuard including observe, aggregate/bootstrap/fresh_run/metadata/receipt helpers and original guard __init__/event/fail remain byte-exact. Within install, the raw wait tuple, enrollment/single reaper/latch/next-launch machinery, observed_exit reading sys.exception itself, raw guard writer, finally trace restoration and clear_P behavior are unchanged. No process/schema/policy/cap/authorizer/SQL payload/dependency change was made. Static compilation includes all three P modules and the complete embedded helper. A compile-only comparison confirms canonical resume code matches under the one-line bootstrap prefix on the host compiler; it is not pinned-runtime execution evidence. `git diff --check` passed.

Frozen SHA256:

- guard: `46682286cb471ea758667da8994eb17957ef6fa4b81858e0c8c0f28929b9ccff`.
- guard controls: `bf4abef58b666878ecade2a0870c30554368043ec59ba4b4102b956ea06f55a1`.
- recovery tests: `c236e4045a0c35836fc269fd5938c053ab3e97f1f2682f023470afe3a68dd211`; P target definition `11725409583de8133a93de2e8db0135250e76f5e0fe7048f97119c4c0c67393e`.
- production recovery: `c314b2423b71c7d205b4558aee44d8c895e41394adbbb05a85458ef9bd7acf29`; service lock `4ad7f2d9e6d4f03b7abb22d4946524f4372b5277b7b80c186332720f490909ea`; conftest `d967baee72fa7fba3cb8974ac0406b2f44231a1c9d49fb7e7c088016f07cc7a3`; poetry lock `2885f7f62d5a4b4a483e001622ba1d0d5ec4110578971664ecced01fbdfeb553`. These unchanged pins also equal c224a4.

Token proof remains **P12 + deduplicated V23 =35 bytes maximum**. P tokens are exact4-byte P:a;/P:s;/P:f;. Strict parser/order/uniqueness rules are unchanged. P's exact node excludes J/R/T/C and declared D emission predicates, so no long unrelated token adds to this actual invocation. Existing conservative other-node envelopes remain R112/J57/backpressure125/other-T42/D153 <=256. Future H/S short triples give59 only conditional on their exact4-byte tokens, single-contract eligibility, preserved node exclusions and explicit combined-source overlap review. No unconditional future combined-source proof or H/S credit is claimed; the historical256-byte read/record bound is unchanged.

### Finite future selection — source enumeration only, not collection or authorization

| Selection | Static outer cases |
|---|---:|
| Original P target | 1 |
| test_P_ordered_armed_transcript (unchanged eleven defects) | 11 |
| test_P_durable_lost_result_fresh_contract (the prior30 modes plus operation-value) | 31 |
| All other existing test_child_wait_guard.py nodes, including R15/J9/backpressure8/ownership and receipt controls | 76 |
| test_lotp_late_refusal_retains_unaccepted_plan, all5 | 5 |
| test_lotp_early_refusals_precede_original_writer[journal-symlink] | 1 |
| test_t1_actual_pipe_transport, all18 modes | 18 |
| test_t1_real_blocked_pipe_backpressure_deadline; test_t1_actual_pipe_overflow_precedes_decoder | 2 |
| test_t1_close_ownership_cleanup: child-read all3 timings + parent-write all4 timings, each2 failure classes ×2 repeats | 28 |
| test_t1_reap_interruption_ownership: normal-wait/cleanup-wait ×2 phases ×3 error classes ×2 repeats | 24 |
| test_kpre_predecessor_damage_no_capture[old-core-descriptor] | 1 |
| test_lotk_actual_death_fresh_owned_resume, both points | 2 |
| test_lifetime_wait_guard_unit all6; real_semantic_refusal_and_unrelated_child; reporting_overrides_caught_negative | 8 |
| Total prospective standalone P/preservation outer items | 208 |

This is a finite source-selected list, not a collected count or passing test count. Embedded helpers/intentional rejected rows are additional bounded internal evidence, not extra outer PASS claims. The31-mode decorator is the exact finite fresh list; no mode or selected count may be adjusted after execution to manufacture green. Normal requires SQL1/5, exact durable intent/sentinel, original child_failed refusal, native result-write/frame/offset/memoryview/errnoNone identity and both real observational replay byte/logical equalities. Negative rows retain exact rejected status and fail-latch evidence. Parent must separately authorize the pinned Linux selection only after independent full source review, then retain and independently review actual raw owner/PID/node/status and launch/ack/done evidence before acceptance.

### Specific unresolved issues and handback boundary

F1–F3 have implementation responses, **not independent acceptance**. Mandatory next step is a new independent SOURCEONLY review of this exact freeze/full delta before any runtime. CPython3.14.8 trace callback exception propagation, writable frame-local proxy readback, bytecode/cache offsets, exact positive/final route and code equality under its real bootstrap remain unexecuted. Case-specific proof assertions are intentionally strict; unexpected phases/repeated events/fixture failure must STOP, not be recounted as the directed negative. The test-owned replacement deliberately adds a different route; no production cleanup defect is claimed fixed. No actual resource-cleanup, engine, library, installed or durability evidence was obtained here. General legacy V/J stale-primary concerns remain outside P; historical PID161/PID255 exceptions remain unproven.

Do not integrate or combine this commit without parent acceptance and explicit shared observe/install overlap resolution, independent combined-source review and the separately authorized grouped actual gate. H/S work and genuine late-completion engine-capacity coverage remain separate; P supplies none of that evidence. Four-B2 and conditionally full service/release remain later gates. Mandatory C1 `b2_pinned_launcher_context_unavailable`, native Darwin V1/V2/V3/fresh-helper/collection causes, A1 drift, all19 Linux mechanisms/ext4/Unix-VFS/WAL/FULL/SHM/peer-credential/installed and physical durability, Apple Silicon/browser/acoustic/activation, T2/full C1295 and final global Astra remain pending. Historical worktrees and STT proof are preserved. No runtime-positive or self-GO is claimed.
