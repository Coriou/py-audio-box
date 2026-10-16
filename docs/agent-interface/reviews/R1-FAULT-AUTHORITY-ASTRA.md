# R1 fault authority — targeted Astra design decision

2026-10-16. Read-only functional durability review; only this report is authored.

**Candidate remains NO-GO:** `0aaa70780a979d501eae959147f01c0f7475ebf3`, inspected at `/Users/ben/.txcode/worktrees/py-audio-box/t3-f6dcdb48`. F1 remains open; F2's bounded closure is preserved. This is a design decision for the next correction, not implementation acceptance or final global Astra. R1 integration, P1/public/unattended persistent activation and full Phase 1A remain closed.

## Decision and unavoidable availability cost

Choose **a committed, short, per-resource-check uncertainty fence**, using the existing Core SQLite authority. Commit `checking` BEFORE any resource observation; attest `healthy` only after that same uninterrupted check has completed successfully and its accounting updates have committed. On restart, any inherited `checking` means unresolved resource authority. Never complete another process's check by taking a fresh healthy sample.

This deliberately sacrifices availability if control dies during an otherwise healthy check. The affected root remains quarantined without an R1 automatic recovery path. This is an explicit, narrow product-engineering tradeoff for the trusted self-hosted diagnostic slice. It does NOT make every restart, metadata read, asynchronous lifecycle call or entire execution permanently uncertain. It does mean that an arbitrary process death may coincide with a check. There is no honest guarantee that every no-resource-fault crash will recover automatically.

The old unconditional expectation “crash-start-sent always succeeds on restart, regardless of overlapping control scan” and F1's required guarantee cannot both hold with the evidence currently supplied. Preserve W2 ordinary reconnect when the durable monitor is clean; explicitly test and document the overlapping-scan exception. Do not label this exception unchanged W2 availability or silently weaken a failing test. If unconditional recovery even during an unattested scan is required, this small design is insufficient: a separate pre-established authoritative monitoring mechanism and its proof would be necessary. Do not implement an optimistic reset instead.

## Evidence and impossibility argument

Read root PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, canonical IMPLEMENTATION-PLAN resource/shared/CPU-GPU/Phase 1 gates and CONTRACTS 1.3.0; both complete root R1 reviews; candidate RESOURCE-R1 and RESPONSE-R1-r1; actual candidate Dispatcher, Admission, ResourceStore, Core, DispatchStore, SupervisorRuntime and relevant installed-test paths. Candidate HEAD matches the requested SHA. No tests or external probes were run for this design review. The 173 E1 tests and 36 installed steps are the independent round-2 review's reported results, not a fresh run here; they do not cover its proven counterexample.

Current source establishes:

- `Dispatcher._resource_check` sets memory latches before writing, but its first monitor UPDATE occurs AFTER the failing observation. `persistence_fault` can then suppress later witness attempts. Startup reconstructs solely from the still-healthy monitor and empty fault table. That is the F1 gap.
- `ResourceStore.__init__` calls `check_disk` before Dispatcher reconstructs the monitor. `check_disk` itself measures directories, updates high-water and reads statvfs. It is also called inside reservation/acquisition transactions. Fencing only the top-level periodic `Admission.check` would leave these paths uncovered.
- `Core.transaction` rolls back its transaction on exceptions. A `checking` UPDATE inside that same transaction is not a durable precondition: rollback can erase the very witness needed after a fault.
- `_terminal` validates runtime/binding/original deadline/reap before publication and release. `_service_stop` carries the F2 obligation across pending I/O. Neither a validated reap receipt nor a good result supplies missing control-scan history.
- RESPONSE-R1-r1 records an earlier per-scan attempt failing installed start-sent recovery. `SupervisorRuntime.call` injects that exit from the pool thread immediately after send; meanwhile the owner loop can be scanning. This is a real ambiguity, not evidence that writing only after a fault solves durability.

Consider two histories with identical committed database state and supervisor state: (A) a healthy control check interrupted before its completion is recorded; (B) a transient output violation observed, all subsequent fault writes denied, control killed, temporary pressure removed. Both can later present the same valid timely result, exact receipt and writable filesystem. A restart cannot distinguish them. Accepting A on those facts also accepts B. Retrying a memory latch after writability returns does not cover death before retry. A second post-fault file/database is not a proof unless its availability and ordering are independently guaranteed BEFORE the vulnerable observation.

A normal execution reservation proves charged capacity and ownership; it does not prove “no control fault observed.” Repurposing every executing reservation as uncertainty would block all active W2 reconnects. An allocation-time monitoring epoch can be sound, but without an independent completion attestation it remains open for the entire attempt. That sacrifices substantially more availability than the chosen short fence. A heartbeat, timer, later healthy scan or timely result cannot close either kind of uncertainty.

## Frozen protocol for the correction

### Authority and states

Keep one serialized Core writer and the existing root lock. Use the existing monitor states with these precise meanings:

- `healthy`: no incomplete resource-check epoch and no latched resource fault. It says nothing about continuous filesystem health between samples.
- `checking`: a committed check began, but no trustworthy completion is committed. On another process/session this is irrevocably unresolved in R1.
- `fault`: an observed violation and its bounded diagnostic were committed. This stays latched; a later healthy reading never clears it.

An in-process token identifies the currently executing check and is the sole permission to complete its `checking` row. It is NOT durable fault authority. Only the already-committed row provides crash safety. With one owner and no overlapping checks, a new general-purpose epoch log is unnecessary. Assert owner serialization and reject reentrant/unfenced checks. A persisted protocol revision distinguishes the new pre-observation meaning from the candidate's post-fault-only semantics; do not reinterpret a populated old root's `healthy` as proof it never lost a fault. Fresh roots initialize the new revision. Old/unversioned populated roots fail closed pending a separately designed migration/recovery policy. Never overwrite old faults or initialize a missing monitor as healthy on a populated root.

### Exact ordering

1. Recover the monitor and faults before scanning, accepting commands, reconciling cancellation releases or replaying terminal publication/ACK. Missing, invalid, old-protocol or unresolved authority blocks progress. Reading status, existing jobs and idempotent duplicate receipts remains possible if storage can be read.
2. At a legitimate new check, require durable `healthy`, no fault row, no local latch and no active DB transaction. Commit conditional `healthy → checking` and verify exactly one row changed. The commit must return successfully BEFORE calling the observation function. Never wrap begin and observation in one rollback-able transaction.
3. If begin fails, do not perform the scan or its dependent admission/start/publication/release/ACK. Quarantine the live process and service bounded stop as feasible. A failed begin is not evidence that a resource violation was observed. After crash, persisted `checking` blocks; a truly unchanged `healthy` can reconnect because no observation or dependent action was allowed after that failed begin. If the commit outcome is ambiguous, stop this session; restart reads the actual committed state. No new store is assumed writable.
4. Run the finite resource observation synchronously on the owner: managed/staging tree, sizes/output bounds, free reserve and publication WAV checks when applicable. Keep asynchronous runtime I/O outside the check's lifetime. The pool may still be pending; do not hold the fence for the duration of an RPC, wait or whole execution. Do not suppress monitoring during pending calls merely to pass crash tests.
5. On successful observation, commit high-water/accounting changes and `checking → healthy` together, conditional on the SAME live token and absence of any fault. Only then may dependent work continue. If this completion write fails or control dies, the precommitted `checking` survives. No success-path `finally` reset. No new check may close an old unresolved check.
6. On any observation exception, latch quarantine/resource fault/stop in memory before fallible writes. Try to commit bounded reason plus `checking → fault` atomically. Failed UPDATE, INSERT or commit leaves the prior committed `checking`; do not let rollback remove it. Once persistence fails, keep the unresolved state for this session and future restarts; no automatic later retry converts it into release authority. A successful known-fault record permits the established F2 failed/reaped terminal flow; an unresolved record does not.
7. Resume lifecycle observation/collection and necessary stop despite a latched fence. Skip new resource scans when authority is already unresolved rather than trapping the loop before stop service. A returned prepare is bound and stopped before start; failed/unbound prepare is not relaunched. One stop per runtime/session; UNKNOWN or failed stop is not reap. A later exact bound receipt can discharge the physical stop obligation, but cannot discharge resource uncertainty.

For unresolved authority: no fresh key, acquisition, replacement, publication or publication-intent recovery; no new ACK or execution-charge release, even after exact reap. Record a failed/cancelled job after validated reap if writable, preserving receipt and ownership evidence; do not fabricate a successful outcome. Already committed successes/artifacts remain readable and terminal states never reopen. Previously completed releases/ACKs are not undone. For a durably recorded ordinary `fault`, preserve existing failed/cancelled terminal plus exact-reap execution release/ACK, while admission/publication remain blocked and disk charges stay retained.

### Cover every resource-read entry point without destroying atomicity

Refactor measurement from admission arithmetic. A fenced preflight returns a private, single-owner-call snapshot of measured bytes/free reserve and completes its accounting attestation before the consuming transaction. `reserve_tx`/`acquire_tx` still calculate cumulative charges, FIFO, current capacity and replacement additions inside the original atomic job/settings/reservation or attempt/Authority transaction. They consume that immediate preflight; they must not perform a new unfenced filesystem read or accept client-provided/stale measurements. Replacement and ordinary submission use the same policy. Pure capacity/profile/input refusals are not automatically resource-fault observations: a request that does not fit must remain an ordinary atomic refusal, not permanently poison an otherwise healthy root.

Remove constructor-side unguarded measurement. Fence startup, periodic and pre-publication scans. Audit all `measure`, `check_disk`, `statvfs` and Admission.check callers; no accidental transactional bypass. Generic trusted allocator tests must use the same private preflight contract. Duplicate lookup remains ahead of preflight and mutable readiness. Core's historical non-Admission path stays unchanged. Existing publication verification/adoption remains the W2 protocol; this correction fences R1 resource checks, not every validation/read or filesystem mutation in the program. Do not claim a continuous disk quota or extend these attestations to unobserved periods.

Normal queued cancellation/release stays atomic. Under inherited unresolved authority, conservative refusal of its combined release is acceptable in this bounded slice; retain the job/reservation unchanged rather than partially cancelling. Running cancellation records intent where writable and does not erase the resource-stop obligation. Metadata reads never create a monitor epoch.

## Why the supervisor cannot clear this fence

The present supervisor supplies ownership, authenticated binding, launch/stop state, original deadline, timely completion and confirmed reap. Container limits bound RAM/PIDs; the fixed synthetic executor bounds its intended output. These are useful execution invariants. They do not attest host filesystem free reserve, other managed-root bytes, transient namespace/read errors, high-water accounting, or what the control actually observed before dying. Even a perfectly valid 9,644-byte final output does not refute an earlier 10,000-byte observation. No “resource healthy” receipt field exists here.

A future independent authority could close a longer execution epoch only with pre-established coverage of every relevant invariant, correct scope/binding, continuity through control death and its own failure policy. That would require new protocol and evidence. Merely adding a supervisor boolean or trusting its success receipt invents evidence and is excluded from this correction.

## Ready Sol-low implementation brief

Outcome: close F1 with the protocol above, preserving F2 and conditional W2 recovery. Base is the exact candidate SHA named above. This report makes no implementation edits, launches no agent and authorizes no integration or activation. Parent assigns the next bounded correction and separate independent review.

Exact proposed implementation allowlist:

- `service/resource_store.py`: monitor revision/state transitions, guarded observation/accounting and preflight snapshot consumption.
- `service/admission.py`: fenced checks and trusted preflight boundary.
- `service/dispatcher.py`: startup ordering, latches, guarded call sites and unresolved terminal/ACK behavior; preserve F2 scheduling.
- `service/dispatch_store.py`, `service/core.py`: only Admission-aware preflight/transaction plumbing needed to keep atomic reservation/acquisition and cancellation; preserve historical behavior.
- `tests/service/test_admission.py`, `tests/service/test_dispatcher.py`: deterministic new ordering/restart regressions and preserved lifecycle assertions.
- `tests/release/test_admission.py`, `tests/release/test_dispatcher.py`: installed crash/recovery coverage and explicit clean-versus-open crash barriers owned by the tests; no production health bypass.
- `docs/agent-interface/phase-1/RESOURCE-R1.md` and new `docs/agent-interface/phase-1/RESPONSE-R1-r2.md`: frozen semantics, availability exception, evidence and finding response.
- New `docs/agent-interface/phase-1/r1-evidence/authority-*.json` receipts only. The runner may add a closed `authority-` prefix in `tests/release/test_admission.py`; preserve earlier evidence.

No worker/protocol/schema/fixtures, packaging/dependencies, public clients, legacy apps, planning ledger or existing reviews in this correction allowlist. No new fallback journal, recovery command, cleanup policy, model/private audio/cloud/remote E2E, push/PR/deploy. Existing offline test entry points suffice; do not run `scripts/run-e2e-test.sh`. Do not broaden into P1's power-loss/fsync/operator recovery matrix.

### Required regression matrix

| Case | Mandatory result |
| --- | --- |
| Begin-fence write denied before observation | Observation count zero; no dependent action. Live refusal. Restart distinguishes unchanged healthy from a committed checking row; never pretends an observation occurred. |
| Original INSERT trigger after a real transient output violation | Memory stop/fault precedes writes; committed checking survives failed fault transaction; no new admission/publication/ACK/release after pressure disappears. |
| Fault witness UPDATE itself denied, immediate process death | First prove checking committed BEFORE the real 10,000-byte observation. Enable owner-connection query_only only after observation raises, deny fault transition, kill without close/finally/later loop/retry. Reopen same root writable with pressure removed. Still unresolved; exact valid timely result cannot publish or unlock admission/ACK/release. |
| Same denial, healthy interval before death | Live latch remains; healthy interval cannot attest old epoch. Restart has same refusal, including a second restart. |
| Healthy scan interrupted before observation, mid-observation, after return but before completion commit | All three inherit checking and quarantine. Explicit availability-negative tests, not evidence of an actual violation. |
| Healthy completion UPDATE/commit failure | Precommitted checking retained; rollback cannot restore healthy; no dependent publication or allocation. |
| Transaction rollback around reserve/acquire | Fence cannot be rolled back with job changes. No partial job/settings/reservation/attempt. Pure queue/RAM/disk-demand refusal does not become a false observed resource fault. |
| Startup, periodic, replacement, pre-publication and direct allocator paths | All resource observations covered, no constructor hole, stale snapshot rejected; old/missing monitor does not mint healthy authority. |
| Clean W2 crash-start-sent/reconnect | Explicit barrier establishes no open scan at death; one physical launch, same binding/generation/deadline, observation resumes, valid result publishes, ACK and release occur once. No fresh execution/replacement inferred. |
| Start-sent death overlapping open scan | Durable checking requires uncertainty even with healthy supervisor. Keep separately from clean reconnect; do not reset state to satisfy former unconditional recover assertion. |
| Other clean W2 boundaries | Preserve postspawn, collect, preverification, before/after adoption, after commit, durable outcome, before/after ACK, lost ACK and control-absent deadline outcomes. Clean means no unresolved scan, not a new healthy sample after restart. |
| Cancel paths | Queued normal cancel/release atomic; running cancel and completion race preserved when clean. Unresolved running cancel never permits success/ACK/release. Queued unresolved refusal has no partial state. |
| F2 matrix | Fault during pending prepare/start/observe, each with/without cancel; successful and failed/UNKNOWN stop; failed pending calls; repeated faults. Bind-then-stop before start, no lost obligation, no stop storm, no release on signal/ALIVE. |
| Known-fault versus unresolved receipt | Exact reap permits known-fault failed/cancelled outcome and execution release/ACK; unresolved permits recording failure/reap but retains execution and withholds ACK. Disk charges never shrink. |
| Read/dedup and already terminal state | Existing key/job/artifact retrieval works without opening epochs. No terminal rewrite; no duplicate publication or release; no ACK bypass through a stored outcome when authority is unresolved. |

Use owner-thread injection and barriers instead of sleeps as the decisive ordering evidence. Run immediate-death tests in disposable subprocess roots, disabling bytecode/cache writes to reviewed sources. `PRAGMA query_only` is an ordinary E1 deterministic write denial, not physical ENOSPC or power-loss proof. Add an installed E2 restart case with the same control injection and real supervisor stop/reap/receipt; label the policy fault E1 and lifecycle evidence E2. Record exact source/image identities and bounded owned-resource cleanup.

Retain the full existing suite and installed scenario coverage. Do not report “all previous recovery tests unchanged” if a formerly racy start-sent assertion is split: explain and evidence both outcomes. The clean-boundary assertion must still require successful W2 recovery. An overlap case must require refusal. Neither may accept either outcome without inspecting the persisted fence at the crash barrier.

## Closure conditions

F1 closes only when an independent reviewer verifies the pre-observation commit ordering in actual source, every R1 observation entry point, rollback separation, no inherited-fence clearing, and both first-write-denial/immediate-death and writable/pressure-cleared restart regressions. Verify no publication/ACK/release bypass through terminal or stored publication/outcome paths. F2's complete pending-I/O stop matrix must remain green. Clean-boundary W2 scenarios must still recover with one execution; scan-overlap quarantine must be explicitly documented and tested as the selected availability cost.

If implementation needs unconditional start-sent recovery during an unresolved scan, return NO-GO against this brief rather than moving the write after the observation again. If the availability cost is unacceptable for product release, the next work is a separately designed authoritative monitoring/recovery protocol. That is not grounds to waive today's F1 as P1. P1's full storage/power-loss/operator recovery and the later global Astra review remain separate, closed gates.
