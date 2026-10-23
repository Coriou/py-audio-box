# P1B23 combined staged broad validation — 2026-10-23

**STOP / broad validation NO-GO for exact `bd29d757995c98b8f87b596be9cffe80074dc997`, executable freeze `30a4130d626ef46cd031ad469d2a9eb75dc0b266`.** Stage 1 passed 18/18. Stage 2 stopped at its first failure: 254 passed, 1 failed. Stage 3 was not started. No retry, fix, commit, delegation, integration or self-GO.

## Identity and authority

READONLY ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-70c69445`. Sole repository write is this report. Parent accepted the bounded combined review (27 independent passes; 388 separate attributed passes audited), and explicitly authorized original17(+B), exact four-B2, then full service/release only after each previous stage was green. Those prior passes are not executions of this report.

Read the full COMBINED-OWNERSHIP-GUARD-CAP-r1-sol, combined candidate RECOVERY-P1B2:2025–EOF, ORCHESTRATION-PROTOCOL, ledger chronology through staged authorization, CHILDWAIT-GUARD-r2-sol, SQLITE-CAP-CONTRACT-ASTRA, SQLITE-CAP-WITNESS-r2-sol, CORE-OWNERSHIP-r1-sol and LIFETIME-RETENTION-ASTRA. No applicable AGENTS.md found in checked ancestry/scoped paths. Memory registry supplied only evidence-tier guidance; current pins and executions determine this result.

Entry, before/after each executed stage and final HEAD/empty porcelain checks pass. Freeze-to-HEAD is solely `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. All191 retained manifest entries match host candidate before imports and after stages; all191 match inside `/combined` before runtime identity/product imports and after each stage. No T2/C imports or source changes. Host Python3 was static/hash/tar/Docker orchestration only; no Darwin pytest/product imports/native causal/GC/allocator work.

## Prioritized failure and pending evidence

**P1 — `tests/service/test_recovery.py::test_b22_correction_process_death_owned_restart[core-intent-commit]`, line1364.** Actual failure:

```text
result=recovery._pending_through_evidence(roots,scope,owned,owned['proof']['observation'],now=1000)
KeyError: 'proof'
1 failed, 254 passed in 55.29s
```

The preceding direct fork child exit73 assertion and both pending refusals were reached without reported failure; no native signal is established. This direct test fork is outside the maintenance seam and guarded fresh EXEC launcher: no raw maintenance wait/helper receipt was emitted for it. Do not turn the stdout exit assertion into an independently retained raw guard receipt.

The test uses `next(roots['receipts'].glob('p1-recovery-*.intent.json'))` at line1358. Read-only retained-tar inspection finds BOTH matching names in the failing fixture:

- `p1-recovery-99999999999999999999999999999999.bundle.intent.json`: keys intent_digest/recovery_id/revision/snapshot_digest/source_digest, no proof.
- `p1-recovery-99999999999999999999999999999999.intent.json`: full recovery record including proof.

The traceback's missing key is consistent with selection of the bundle descriptor by the broad glob. Retained files alone do not replay historical directory iteration order. This is a concrete ambiguous-selection diagnosis, not a repaired test or a proven production recovery defect/native cause. The failing original function is unchanged by the combined G/cap overlays relative to accepted O (the overlay starts later); no assertion was weakened or changed here. Parent must adjudicate the exact node/fixture contract and authorize any correction/verification. No retry or continuation is inferred.

The remaining four-B2 cases, the other three files after recovery in this invocation, and the entire full service/release stage remain unexecuted/pending. No count extrapolation.

## Exact executed commands and results

`PYTHONDONTWRITEBYTECODE=1`; no cache provider; normal unchanged conftest/guard. Exactly two pytest invocations, one per stage, no collection-only or diagnostic pytest invocation. Both stderr files empty. No skips/xfails reported in the executed prefix.

```sh
docker exec -w /combined pab-combined-broad-entries-20261023 python -m pytest -p no:cacheprovider tests/service/test_recovery.py::test_b22b_process_death_fresh_interpreter 'tests/service/test_recovery.py::test_b22c_process_death_fresh_interpreter_twice[core-complete-write]' 'tests/service/test_recovery.py::test_b22d_process_death_owned_fresh_resume_twice[completion-fence-before-replace]' 'tests/service/test_recovery.py::test_b22d_r2_completed_replay_crash_successor_fresh_interpreter[core-complete-commit]' tests/service/test_recovery.py::test_b22d_r3_receipt_only_crash_fresh_interpreter tests/service/test_recovery.py::test_lotk_actual_linux_child_file_and_sqlite_cap 'tests/service/test_recovery.py::test_lotk_actual_death_fresh_owned_resume[bundle-parent]' 'tests/service/test_recovery.py::test_lotp_production_intent_actual_death_fresh_resume[bundle-parent]' tests/service/test_recovery.py::test_lotp_single_immutable_plan_proof_first_and_preaccept_exit 'tests/service/test_recovery.py::test_kpre_actual_death_owned_fresh_resume[accepted-parent]' 'tests/service/test_recovery.py::test_kpre_recapture_actual_death_fresh_retry[recapture-before]' 'tests/service/test_recovery.py::test_kpre_capture_failures_burn_without_outputs[cap]' 'tests/service/test_recovery.py::test_kpre_no_attempt_both_busy_locks[control]' 'tests/service/test_recovery.py::test_kpre_each_no_attempt_window_rechecks[capsule-valid-False-accepted-write]' tests/service/test_recovery.py::test_kfifth_fourth_query_cannot_bypass_unfinished_finish 'tests/service/test_recovery.py::test_kfifth_production_owner_changes_before_writers[same-container-completion-fence-dirsync]' 'tests/service/test_recovery.py::test_kfifth_fourth_query_missing_certificate_cannot_write[descriptor]' tests/service/test_recovery.py::test_lotk_actual_linux_sqlite_checkpoint_cap -q -rs -x --basetemp=/tmp/broad-entries
```

Wrapper exit 0; elapsed 28.309981s; **18 passed in27.55s**, 0 failed/0 skipped.

```sh
docker exec -w /combined pab-combined-broad-fourb2-20261023 python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q -rs -x --basetemp=/tmp/broad-fourb2
```

Wrapper exit 1; elapsed 56.120091s; **254 passed,1 failed in55.29s**, 0 skipped in executed prefix; stopped by `-x`.

Stage3 intended scope was exactly `tests/service tests/release -q -rs -x -p no:cacheprovider` on the same image with a unique basetemp. It was NOT executed because stage2 was red.

## Original17 identity and actual launch coverage

The original entry-selection.json is used verbatim, preserving all17 exact node identities/parameters and assertions; only explicit capB is appended for18. Raw helper counts below bind actual node/PID/token, not a substituted static count:

- `tests/service/test_recovery.py::test_b22b_process_death_fresh_interpreter` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_b22c_process_death_fresh_interpreter_twice[core-complete-write]` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_b22d_process_death_owned_fresh_resume_twice[completion-fence-before-replace]` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_b22d_r2_completed_replay_crash_successor_fresh_interpreter[core-complete-commit]` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_b22d_r3_receipt_only_crash_fresh_interpreter` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_lotk_actual_linux_child_file_and_sqlite_cap` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_lotk_actual_death_fresh_owned_resume[bundle-parent]` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_lotp_production_intent_actual_death_fresh_resume[bundle-parent]` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_lotp_single_immutable_plan_proof_first_and_preaccept_exit` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_kpre_actual_death_owned_fresh_resume[accepted-parent]` — 1 actual helper triplet(s).
- `tests/service/test_recovery.py::test_kpre_recapture_actual_death_fresh_retry[recapture-before]` — 2 actual helper triplet(s).
- `tests/service/test_recovery.py::test_kpre_capture_failures_burn_without_outputs[cap]` — 2 actual helper triplet(s).
- `tests/service/test_recovery.py::test_kpre_no_attempt_both_busy_locks[control]` — 2 actual helper triplet(s).
- `tests/service/test_recovery.py::test_kpre_each_no_attempt_window_rechecks[capsule-valid-False-accepted-write]` — 2 actual helper triplet(s).
- `tests/service/test_recovery.py::test_kfifth_fourth_query_cannot_bypass_unfinished_finish` — 4 actual helper triplet(s).
- `tests/service/test_recovery.py::test_kfifth_production_owner_changes_before_writers[same-container-completion-fence-dirsync]` — 2 actual helper triplet(s).
- `tests/service/test_recovery.py::test_kfifth_fourth_query_missing_certificate_cannot_write[descriptor]` — 2 actual helper triplet(s).

Extra `test_lotk_actual_linux_sqlite_checkpoint_cap`: one actual helper triplet. Total27 triplets across18 nodes, including all17 original selected node identities; no nested helper directories/receipts in this selection. The original17 **static substitutions** are 16 test-function call sites plus `_kpre_fresh`. The latter is exercised by `test_kpre_actual_death_owned_fresh_resume[accepted-parent]` (source4606→helper4594–4599 and its bound raw triplet), and also by selected recapture flow. The other16 are single guarded call expressions in their respective selected functions; capA is the accepted replacement. This supports execution of each original guarded entry site under these representative parameters. It does not prove every parameter/branch, all fresh launchers in the suite, or individual call-location instrumentation: receipts record node/PID/token, and repeated launches from loops/helpers are not extra static sites. The newly approved compatibility EXEC is not in this17 selection; stage2 stopped before reaching it. Its prior27/388 focused coverage remains attributed to the combined review.

## Runtime, isolation, footprint and bounds

Existing exact image `sha256:ef1fbbb46f3913b1f1015df921d6354b698aaa93e5f5196b4bfffb997fc3deaa`; no build/pull/install. Fresh per-stage identity confirms CPython3.14.8, all42 locked installed package versions, SQLite3.40.1, source_id `2022-12-28 14:03:47 df5c253c0b3dd24916e4ec7cf77d3db5294cc9fd45ae7b9c5e82ad8197f3alt1`, Debian libsqlite3-0 `3.40.1-2+deb12u2 amd64`; linked libsqlite3.so.0.8.6 SHA256 `2e6eef9a727f081f0d453b4e5e6cbd8b9ef8b6f86cbf7681cbad444d3b0b55c8`. Compile options/extension/ldd/package and42 exact versions retained in each identity.stdout. Prior image lineage/distribution-source assessment remains attributed, not fresh exact binary-source authentication or Darwin equivalence.

Atomic mkdir `/tmp/pab-worker-w1-docker-lane.lock`, exact PID/token/pin owner. Host reserve >=4GiB checked initially, before/after each stage and finally; export required >=5GiB. Final free13,509,976,064 bytes. Separate owned containers per stage prevent cumulative tmpfs growth; no old basetemp/manual fixture deletion. Known focused archive footprint inspected; full-suite peak was unknown and never claimed safe from focused size. Actual retained stage bytes below show ample space for executed prefixes, not a full-suite peak bound.

Both containers: CPU2/memory2g/PID256/networknone/read-only/user10001:10001/capdropALL/no-new-privileges, private /tmp tmpfs1g/nosuid/nodev, bytecode off, fixed tokenizer cache, wait log `/tmp/waits.jsonl`. No host bind/socket/secret/port/model/GPU/audio/service activation. PID1 finite6000s with graceful SIGTERM handling (longer than75min plus export); pytest bounded4500s each and10800s aggregate. No cap increase or parent-limit change. Exact create/inspect/commands in evidence. No timeout/OOM occurred.

## Live raw evidence and audit

Before each test invocation, an owned0600 sentinel was binary-streamed using LIVE `docker exec NAME tar -C /tmp -cf - sentinel` and host-verified for uid10001/token/content. After each stage, INCLUDING FAILURE, entire mounted /tmp streamed to a host binary file using LIVE `docker exec NAME tar -C /tmp -cf - .`. Never docker cp. Every archived regular-file hash matched a separately collected live hash map before container removal. Safe relative names, no traversal/duplicate/hardlink/special type; uid10001; individual<=64MiB,total<=1GiB. Symlinks are retained metadata only; supplemental audit confirms internal existing /tmp targets; no extraction/SQLite open. JSON helper metadata <=4096/private mode, closed launch/identity/event schemas, positive PID/token/node binding, ack==done==launch.identity, event owner and enroll/terminal equality checked. No missing receipt, rejected event or fabricated repair.

- entries: 912 members / 643 regular files / 38901205 regular bytes; tar SHA256 `046a6a54fd398be1e501d0b3b7bc67ca5ffd18a6ae38f55a06c33971e0ac7618`.
- fourb2: 7262 members / 4676 regular files / 148262026 regular bytes; tar SHA256 `41dc241919fc61e57aa0daa9b49e1e28ab1bb89d4ac898ba4390140f24b5d616`.

Entries:36 unique actual(owner,PID) maintenance terminal records:16 exit0,12 raw18688/exit73 with declared D phase,8 raw18944/exit74 with reviewed semantic V phase. No native signal or crafted/synthetic wait records in this selection.27 complete actual launch/ack/done triplets; no nested records. Parent guard.directory=None: raw waits.jsonl is the actual parent receipt; no parent event files assumed/manufactured. Child event populations remain owner-bound. CapA/B actual helpers exited successfully with raw triplets and retained SQLite fixtures; pytest capture hides witness stdout on pass without `-s`, so this report does not independently assert printed operation/commit counts from this run. Accepted test assertions passed, and physical fixtures remain available for fresh review without opening originals.

Four-B2 prefix:0 guarded fresh-helper triplets and0 maintenance wait-log records before this early failure; no receipts expected from uninstrumented direct forks. This is a named evidence limit, not zero process activity. All disposable SQLite fixture bytes and both intent files retained. Synthetic/crafted guard negatives were not reached; no such controls counted as actual deaths.

`run.py`, full stdout/stderr, each argv/duration/exit receipt, create/inspect/prestop/stopped/stats, source checks, live-hashes/inventories, raw tar streams, final-audit.py/json and evidence-hashes.json retained in `/tmp/pab-combined-broad-20261023`. STOP.txt records the intentional red-stage stop. Driver process exit0 is orchestration completion, NOT pytest success; fourb2-pytest.json exit1 is authoritative. No second test invocation after failure.

## Cleanup, preservation and remaining gates

After live audit verification and post-stage source checks, exact-owned `docker stop --time10`, inspect ExitCode0/OOMKilledfalse/Runningfalse/Error empty, `docker rm NAME` without-v, exact-name absence verified for both containers. CPU/memory snapshots retained; failed stage posttest memory158.8MiB/2GiB, PID1 only. Lane marker equality checked before owner-only release; lane absent. Images/evidence retained; no foreign cleanup/prune/reset/clean. All1008 historical/STT reference hashes match. No historical worktree/source write. Final candidate remains clean at exact requested HEAD/freeze and source191 bytes; root STT untracked evidence preserved. Only this report is authored in repository.

No all19 Linux mechanism qualification: actual Linux capA direct OS exact-L/EFBIG and SQLite append plus capB explicit-checkpoint witnesses executed in stage1. Eighteen peer-credential cases and broader Linux/platform matrices were not reached in the stopped prefix; no skipped case is converted to qualified. Linux tmpfs/container is not ext4 installed-host qualification, durability or native Darwin evidence.

Parent fresh receipt review and integration decision remain required. Distinct T2/C, final global Astra, mandatory C1 `b2_pinned_launcher_context_unavailable`, B3/C1/C2/install/durability, native Darwin V1/V2/V3/fresh-helper/collection causes, A1 unexplained drift, Apple-Silicon hardware/browser/acoustic and host/public activation remain open. Preserve L4MiB, canonical result1MiB vs snapshot1MiB, docs64KiB, inspection2048/16MiB, H/charges `U_all_early<=H` and `H+C<=B`, both locks, strict complete source/history joins, accepted/no-attempt authority, once-only capture/consumption and finite64-round exact-child single-reaper cleanup/kernel-latency limits. No waiver, T2 import or integration follows from stage1 green.
