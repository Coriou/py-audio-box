# R1 round 3 independent review

2026-10-16 · gpt-6.1-sol, low · read-only candidate review.

**Bounded GO for R1 at `5ab47170c4c467cb3d3bef6075a0ff6480e8d56f`**, branch `t3/r1-fault-authority-round2`, inspected at `/Users/ben/.txcode/worktrees/py-audio-box/t3-f53dbe4c`. **F1 is closed against the complete Astra authority decision; F2 closure is preserved.** No blocking finding or reproduction blocker remains in this bounded E1/E2 slice. This is candidate review approval, not integration verification, P1 acceptance, full Phase 1A, public/unattended persistent activation or final global Astra acceptance. Those gates remain closed.

## F1 — P1/high finding closed

Actual ordering is now conservative across process death. `service/resource_store.py:53` checks owner, transaction/reentrancy and healthy authority, commits conditional healthy→checking, and only then invokes observation at line 70. The same live token gates completion; high-water and healthy commit together at lines 86–98. Observation failure latches local failure and the Dispatcher callback before the separate fallible fault UPDATE/reason INSERT transaction at lines 71–84. Rollback cannot remove the prior committed checking. There is no success reset in finally, retry-to-healthy, or fresh-scan clearance of inherited checking.

Revision-2 authority initializes only a new empty resource root (`resource_store.py:16–48`); populated old/unversioned, missing and invalid authority fails closed. Constructor measurement is removed. `measure` and `check_disk` guard owner/token and reject observation inside a consuming transaction (`:122–147`), covering statvfs as well as traversal. Admission's output-size and publication WAV checks run within that token (`service/admission.py:54–79`). Core submission (`service/core.py:192–225`) and ordinary/replacement DispatchStore allocation (`service/dispatch_store.py:38–60`) complete private preflight before atomic mutation. The scoped snapshot requires the immediate next Core transaction serial (`resource_store.py:101–117`); stale reuse and transactional observation are refused. Reservation/acquisition arithmetic still reads current durable charges/FIFO/capacity in the job/attempt transaction. Pure demand refusal does not poison healthy authority.

Dispatcher reconstructs authority before cancellation-release reconciliation (`service/dispatcher.py:54–69`), skips new scans when latched but continues stop/lifecycle processing (`:147–164`), blocks stored-outcome ACK (`:214–216`), and gates publication, intent recovery and execution release after exact receipt validation (`:241–279`). Unresolved receipt can record failed/cancelled and reap, but cannot publish, ACK or release execution. Already committed success/artifact/release stays readable and terminal. Known durable fault retains the separate failed/cancelled exact-reap release/ACK path. No bypass was found through stored receipt, publication intent, terminal outcome or duplicate lookup.

The fresh full suite re-executed `test_query_only_after_real_violation_immediate_death_twice_restart` in both immediate and interval modes (`tests/service/test_admission.py:424–498`). This reaches the real Dispatcher owner loop. An independent SQLite reader first proves checking committed; the original Admission check observes actual 10,000-byte output and raises resource_output_bytes; only then owner query_only is enabled. SQL tracing requires the fault UPDATE itself to be attempted and denied. Immediate os._exit(29) occurs before close/finally/later loop/retry. Pressure-free writable restart occurs twice with a valid timely E1 result: failed job, no artifact, no new key/replacement, zero ACK, execution reservation retained and checking unchanged. The interval variant restores writable storage/removes pressure and requests checks while latched before death; it cannot attest the old epoch.

The installed authority-setup/authority-denied/authority-recover/recover sequence independently passed through actual packaged Dispatcher and supervisor. The test operator writes exactly 10,000 bytes into its owned staging volume; control observes them under committed checking, query_only denies the fault UPDATE, and the process exits immediately. Only injected bytes are removed before two writable restarts. Real supervisor stop/reap/receipt is preserved, no artifact/outcome ACK is minted, execution charge remains, one physical launch only. Policy/write denial is E1; real UDS/PID-1 lifecycle is E2. This is not physical ENOSPC or power-loss evidence.

## F2 — P2/medium closure preserved

`service/dispatcher.py:156–207` retains the stop obligation across pending prepare/start/observe, successful or failed replies and cancellation. Returned prepare is bound then stopped before start. An unsuccessful/UNKNOWN stop or ALIVE is not reap; exact runtime/binding/original-deadline/pid1-waitpid-echild receipt is required (`:241–246`). One stop per runtime/session avoids a storm. Unbound failed prepare retains ownership without another launch. The exact receipt discharges physical stop, never unresolved resource authority.

Fresh passing tests cover the complete pending-operation × cancel × stop-success/UNKNOWN matrix (`tests/service/test_admission.py:249–293`), failed pending calls (`:353–393`), repeated pressure (`:396–421`), and original INSERT failure with/without cancellation (`:296–336`). Installed resource-transient passed with one physical launch, one actual UDS stop, failed/reaped job, no artifacts, retained disk, continued quarantine and fresh-key refusal. No cancellation masks the cancel=false resource-stop cases.

## Entire Astra regression matrix disposition

Every row below was inspected in source and exercised in the fresh complete suite; installed cases are identified explicitly.

1. **Begin UPDATE/commit denied:** zero observation, no dependent allocation; live refusal. Unchanged healthy restart is distinguished from inherited checking. Trigger and deterministic transaction-failure tests pass. Additional independent Dispatcher probes below verify actual entry reach.
2. **Original INSERT trigger after real transient output:** checking survives rollback; no admission/publication/ACK/execution release after pressure disappears; both cancellation variants pass.
3. **Fault UPDATE denied, immediate death:** independent committed-fence read, actual 10,000-byte violation, traced denied UPDATE, immediate exit and two writable restarts pass in E1 and installed E2 lifecycle.
4. **Same denial with healthy interval:** restored writability and cleared pressure never close the epoch; two restarts retain refusal.
5. **Healthy interruption before/mid/after observation:** three disposable subprocess cases inherit checking and refuse. These are explicit availability negatives, not claimed violations.
6. **Healthy completion UPDATE/commit failure:** precommitted checking remains, accounting rollback does not restore healthy, dependent allocation is refused.
7. **Reserve/acquire rollback:** no partial job/events/settings/reservation/attempt/dispatch; attestation is separate. Stale snapshot fails; queue/RAM/disk-demand refusal remains ordinary capacity refusal.
8. **All entry points:** constructor has no measurement; startup/periodic/prepublication, submission, replacement and direct trusted allocator paths use fencing/preflight. measure/check_disk/statvfs and Admission callers were searched. Missing/old authority never mints healthy.
9. **Clean W2 start-sent:** owner barrier plus independent database read proves healthy at actual send/death. Mandatory successful recovery passed with one physical allocation/launch, original stored Authority/binding/generation/deadline, publication, ACK and release. No accept-either-outcome assertion.
10. **Start-sent overlapping scan:** separate checking barrier requires quarantine through two restarts despite supervisor health. Passed. Death during an otherwise healthy scan deliberately costs availability; no R1 recovery command or unconditional reconnect promise exists.
11. **Other clean W2 boundaries:** postspawn, collect, preverification, before/after adoption, after commit, durable outcome, before/after ACK, lost ACK, control-absent deadline and worker-restart UNKNOWN all passed their required outcomes. Clean crash hooks inspect healthy, rather than taking a fresh post-restart sample as evidence.
12. **Cancellation:** clean queued atomic cancel/release, running cancel/completion race pass; unresolved queued refusal leaves job/events/reservation unchanged. Unresolved running cancellation cannot publish/ACK/release.
13. **Full F2 matrix:** all pending prepare/start/observe combinations, cancel/no-cancel, stop success/UNKNOWN, failed pending calls and repeated-fault/no-storm assertions pass.
14. **Known fault versus unresolved receipt:** known fault can fail/cancel then release execution/ACK after exact reap; unresolved records failure/reap while retaining execution and withholding ACK. Durable disk charges do not shrink.
15. **Read/dedup/terminal/stored-outcome:** metadata/idempotent retrieval does not open epochs; already committed success/artifact is preserved. Explicit publication-intent and stored-terminal-outcome regressions prevent adoption/ACK bypass under unresolved authority.

Original R1 requirements remain coherent: one integrated allocator covers inference/decode/render/provision, finite independent slot/resident/RAM/VRAM/PID/queue/disk budgets, current trusted profile checks separate from frozen plans, FIFO, atomic submission/acquisition/cancel, retry/replacement disk accumulation, UNKNOWN retention and conservative restart. Only fixed synthetic inference has an executor. Generic stronger-budget tests qualify allocator arithmetic, not GPU/Metal/model capacity. Actual/high-water bytes plus cumulative retained charges are intentionally conservative; scans are sampled detection, not a continuous or race-free filesystem quota. One root is one operator-allocated deployment budget with separate control overhead, not automatic host-wide capacity discovery.

## Fresh verification, identity and preservation

Read PRODUCT, implementation start/protocol, canonical shared/resource/CPU-GPU/Phase 1/evidence gates, CONTRACTS 1.3.0, both full prior Sol reviews, the full Astra fault-authority protocol/matrix, candidate RESOURCE-R1 and responses r1/r2. Inspected actual ResourceStore/Admission/Core/DispatchStore/Dispatcher, Authority/runtime adapter, affected source/test diff, installed runner and Docker topology. The candidate was clean at the exact SHA; `git diff --check f6bd3f3042dedcc6a2cac535d50a85c9a0076f95..HEAD` passed. Worker/contracts/clients/packaging/reviews are unchanged from correction base 04deec4.

Inspected runner before execution; exported exact candidate with git archive to `/var/folders/gk/6rj578d95ls341449hb55pj80000gn/T/r1-r3-sol-zw7alzfh`. Runner evidence writes occurred only in that disposable export. Command:

```sh
R1_EVIDENCE_PREFIX=authority- R1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-resource-r1.sh
```

Log `/tmp/r1-r3-sol-run.log`. Exit **0: 200 E1 tests passed in 74.54 seconds; all 43 installed E2 steps passed**. Steps include setup/crash/recovery pairs, not 43 independent fault families. No skips or failures. Existing pinned dependency overlay and network-none builds used; no dependency installation.

Fresh receipt: export `docs/agent-interface/phase-1/r1-evidence/authority-run.json`, SHA256 `de0090351c7dc74b678cecdb122aea2a8f93f1953d85ea4392bbf88b650c2164`.

- Dependency: `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.
- Worker: `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b`.
- Control: `sha256:4cc00cca231ad920710b9e28aa36ea0e22838d87a14920e774a6b70774e7ec8a`.

Post-run independent rehash against candidate found zero mismatches across 96 control, 18 worker and four packaging entries; actual Docker image IDs were rechecked. Sole heavy-lane project `pab-r1-f8d162d9f7` used bounded network-none containers, random owned named volumes and no user-data mounts. Confirmed termination preceded owned removal. All 15 prior containers and 140 prior volumes preserved by inventory digests; zero owned containers/volumes remain. Images and temporary evidence retained; no prune.

Two additional independent ordinary E1 probes injected monitor UPDATE denial through actual Dispatcher construction/owner loop, using existing Runtime and no source modification. Begin denial: observation count 0, durable healthy, live persistence quarantine, no prepare/start/ACK/fresh admission. Completion denial: observation count 1, durable checking, same refusal/no lifecycle launch. Results `/tmp/r1-r3-entrypoints.json`, SHA256 `0c7732cdcae013fc815c1db9d1c5f4f0518523fe1720ac581bba5e871c272dc9`; disposable roots are recorded there. Threads/pools closed. These supplement, rather than replace, the actual Dispatcher crash/restart and installed tests.

Only this root report was written. Candidate source/evidence and root research are preserved. No fix, commit, ledger edit, worker/contract/dependency/legacy change, subagent, model/private audio, cloud/STT/Voicebox, external/security exploitation, remote E2E, prohibited scripts/run-e2e-test.sh, push/PR or deploy occurred.

P1 still owns the full partial-write/file+directory-fsync/SQLite/persistent-volume/orphan/missing-terminal-media/operator-recovery matrix. D1–D5, real adapters/residency/CPU-CUDA-Metal qualification, clients/browser/acoustic acceptance remain separate. Final global Astra must review the integrated candidate independently. This bounded GO closes today's F1/F2 objections without opening those gates.
