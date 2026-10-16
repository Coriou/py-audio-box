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
