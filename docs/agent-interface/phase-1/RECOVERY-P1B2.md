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
