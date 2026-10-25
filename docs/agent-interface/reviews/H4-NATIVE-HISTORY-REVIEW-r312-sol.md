# Independent native retained restart review — r312 Sol

2026-10-25. **Bounded NO-GO for adoption of the r309+r311 SOURCE candidate.** The five exact r310 invalid acceptances are closed, but the new command-sequence validator still accepts command histories the actual Dispatcher cannot produce. One P2 finding below. The authorized final selection ran once: **58 PASS / 38.069s / exit 0**. Three additional live lifecycle tests passed. No self-adoption or candidate correction.

## Authority and exact pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-e6631dc9`, branch `t3/native-history-acceptance-fixes`. Verified clean at entry and after execution, at report HEAD `4b0712934d57a90b9f50ef221de77a07a9ee35bf`.

Verified contiguous chain:

- Base `2e0f82aec6a9aff2f82d1a27fdc2e14f6e5f3b35`.
- r309 initial cherry-pick `12dc00ff23b3a8025b21326f75fe53e9bfa35169`.
- r309 ordering cherry-pick `ab4bc72248f08c0b6880f1e99fe55238a3dbe259`.
- r309 report cherry-pick `01d76f03870459e87545b5f3f5dd0a95b4cff35a`.
- r311 final SOURCE/tests `243a2fb7bcbaf1036fc889029b9939aa4ba25c66`.
- Report HEAD `4b0712934d57a90b9f50ef221de77a07a9ee35bf`; only r311 report differs from final SOURCE.

SHA256 independently checked:

- service/native_activation.py `4a2c34bf4df4454bb1c6f4c15904487a8b8ac614b2caded280727cbbd76c022f`.
- service/admission.py `d72b8edafdf696d0bf07811da140744d302a363d92cd9aaf011ff2af1ad2d164`.
- service/core.py `13ef9dca789b82b68d3798af76fdfe3eed07c285b05654af28d11bef35a4a5ee`.
- service/native_package.py `879106b73fe379de93e5f069fc4c56648ac8cb70c2b3d5d396134eb37dbed152`.
- Unchanged lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

Read full r310 (including all five exact reproductions), r311 and r309; PRODUCT.md, NATIVE-PACKAGE.md and relevant normative r277. Inspected actual base-to-HEAD changes and Authority, Core startup/publication, Dispatcher/DispatchStore, admission/ResourceStore, native protocol/execution, detached validation, reconciliation and publication-intent consumers. Inspected both conftests and fixture seams before execution. No AGENTS.md discovered in candidate or checked ancestors. A bounded memory-registry search found no relevant entry; no memory-derived technical claim is used.

Sole lasting write is this root-checkout report. No candidate edits, commits, merge, push, delegation or operational cleanup.

## F1 — P2: incomplete/impossible Dispatcher command histories are accepted

`service/native_activation.py:161–167` requires prepare/start/collect/ack, permits an empty observe/stop interval, and accepts any number of consecutive returned collects. This checks a looser language than the actual producer.

In `service/dispatcher.py:443–453`, the only launch of the journalled collect operation follows consumption of an observe response whose state is STOPPED. Successful collect then calls `_terminal` (`457–459`), durably records the outcome, and the next journalled operation is ACK (`471–474`). It cannot directly collect after start, or perform two consecutive successful collect commands. Repeated observation before terminal, stop followed by further observation, and ACK replay need to be distinguished from those impossible sequences. Native RPC idempotence does not mean the Dispatcher writes arbitrary repeated RPCs to its command journal. The retained-reader's direct authenticated `_rpc('collect')` also does not append a Dispatcher command.

Two independent tiny fixture probes each created a clean succeeded native job, closed its owner, applied only the stated command mutation through Core, closed Core, and invoked ordinary NativeActivation/Dispatcher reopen:

1. Exact mutation `DELETE FROM dispatch_commands WHERE op='observe'`. Before: `prepare,start,observe,collect,ack`, all returned. After: `prepare,start,collect,ack`, all returned. **ACCEPTED: installed_checked=True, quarantined=False, extra ACKs=0.**
2. Insert one returned collect for that dispatch and place it immediately before ACK, preserving the other operation order. After: `prepare,start,observe,collect,collect,ack`, all returned. **ACCEPTED: installed_checked=True, quarantined=False, extra ACKs=0.**

Both were in one finite stdin invocation, session 29062, chunks a6cd79/091065, exit 0. Each used a separate NativeHistoryTests fixture and the actual completed_history path. The second mutation rebuilt only dispatch_commands with the listed ordered `(dispatch_id,op,state)` rows; no receipt, job, Authority, resource or publication evidence was changed. Fixture-owned teardown only.

This is a retained journal-integrity acceptance gap within the advertised command obligation/order proof. These probes do **not** show forged terminal receipt acceptance, duplicate inference, publication without reap, or negative resource accounting. The existing native receipt/publication proofs still run. Nevertheless, startup declares incomplete/inconsistent durable command evidence checked and clears import closure instead of refusing it.

Require command progression consistent with this finite producer: in particular a returned terminal observe immediately preceding collect, and no consecutive collect transition without the producer's intervening observe. For the current clean completed slice, one returned collect is sufficient; preserve any deliberately supported retry shape only with its actual producer transition. Add focused refusal regressions with unchanged SQLite/files, no ACK and uncleared import closure. Do not repair the journal. This does not demand a new runtime/recovery framework.

## Closure of the five r310 reproductions

- **r310 F1:** closed. Exact demand keys, exact integer dimensions excluding bool/float, 0..2**53-1 bounds and mandatory positivity precede acceptance. Canonical equality with shared native_demand corroborates the frozen execution configuration, descriptor RAM, one slot/PID, zero dedicated VRAM and frozen profile disk for both disk fields. All eight added demand refusal cases pass, including the exact combined -1 reproduction. No historical reserve/repair or freshness expiry is introduced.
- **r310 F2:** closed by explicit retry_of IS NULL; self-reference regression passes. One attempt is no longer used as a substitute for lineage validation.
- **r310 F3:** closed using the existing oracle.fingerprint producer over retained principal/operation and Authority-frozen request. The positive retained test replays the same production key/request after each of two reopens and retains one inference call total.
- **r310 F4:** the exact unknown-returned-operation reproduction and unknown state now refuse. Namespace closure is sound. The new sequencing portion has the residual F1 above; do not describe all command-history semantics as closed.
- **r310 F5:** closed for the inspected immutable joins. Canonical context/native identity, normalized frozen job, outcome and artifact metadata comparisons distinguish int/float substitutions. Only intended mutable completion fields are excluded/normalized; exact True reap and succeeded/completed are required. Four numeric-substitution refusal regressions pass. Authority snapshot, retained receipt/proposal/output and import marker joins retain byte/canonical comparisons.

The new refusal helper compares logical SQLite dumps and spool/staging/assets bytes after mutation versus after refused startup, and checks no new ACK. Where runtime construction occurs it checks pending import closure remains populated and import_completed empty. This is stronger than only asserting an exception. The original five invalid acceptances are independently covered by the once-run selection; no claim that passing old tests alone proves closure.

## Shared live Admission assessment

The r311 extraction preserves live authorization, current profile readiness/qualification checks, hardware freshness and native binding before invoking native_demand. CPU/CUDA/non-native return paths remain unchanged. The shared helper independently decodes the native execution relation and retains the native transfer disk floor; historical use deliberately avoids current-inventory expiry.

Added three existing direct unittest cases because the shared live producer changed:

- NativeLifecycleTests.test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job.
- NativeLifecycleTests.test_stale_future_native_registration_and_historical_decode.
- NativeLifecycleTests.test_cancel_retains_reservation_until_actual_owned_receipt.

**3 PASS / 3.469s.** These cover actual source admission/publication, next-job admission, one PID/zero VRAM, stale/future refusal and cancellation retaining reservation until the injected exact owned receipt. They use fake TLS/process/backend observations, not actual children or hardware.

Additional finite direct demand observations used the same fixture context: live demand equals pure native_demand at baseline; disk -1/0/1.0/True/2**53 and RAM 1.0/True refuse. Valid-size disk `2097152.0` is returned as float by the pure/live producer, but ResourceStore.reserve_tx rejects it as `finite_resource_demand` before SQL. This split validation is unchanged in effect from the former inline producer and does not admit a float reservation; the historical reader additionally rejects it explicitly. Descriptor numeric types remain exact through QwenExecution. No new live numeric acceptance regression was established.

Probe accounting: the first supplemental invocation ran the three tests successfully, then its ad hoc freshness check incorrectly asserted the entire historical demand stayed equal after changing the hardware timestamp. It stopped with AssertionError, exit 1 (session 96377, chunks 3ec570/990984). The configuration hash correctly includes that changed execution inventory. This was a reviewer-probe expectation error, not a product failure or a rerun of the final 58 selection. A separate corrected observation checked the hash against oracle.canonical_hash of each changed execution: historical decode succeeds, while live admission refuses -301-second and +1-second inventory as stale_or_unavailable_device. It also observed the valid-size float disk behavior above (chunk f1fd84, exit 0). A tiny uninitialized ResourceStore numeric-guard probe confirmed float disk refusal before any SQL (chunk fe4180, exit 0).

## Whole-slice assessment and limits

Core lifetime flock and the sole SQLite connection still precede the exact-type NativeActivation retained hook; schema/catalogue initialization follows it. Exact canonical DDL/index/trigger and detached SQL constraints run before retained interpretation. Missing-entire-catalogue regression passes without recreating the tables. The finite row/scalar budgets bound retained materialization, not arbitrary hostile database scan time. Dedicated account, immutable configured inputs and absence of hostile same-UID writers remain premises; repeated snapshots do not supply hostile-writer isolation.

Runtime construction and SQLite stay on the Dispatcher writer. Authenticated collect receives detached Authority, with no SQLite transaction spanning TLS. Current configured identity/boot/trust is bracketed rather than learned. Native terminal proof, retained/current receipt/proposal/output, actual asset hashes/staging bytes, publication oracle/intent, exact spool membership and import intent/completion remain mandatory. No new history write, repair, ACK, scheduler or readiness promotion was found in the reader. Import closure clears only after its implemented checks; F1 is an incompleteness in those checks.

Zero-job repeat reopen and succeeded single-attempt non-clone history replay pass. Failed/cancelled/retried/clone/crash histories remain outside this acceptance policy. recovery.py, reconciliation.py and native_runtime.py are unchanged relative to the requested base; explicit Linux native-history refusal remains, and the selected Linux/native substitution regression passes. No deferred public lifespan/topology/onboarding/hardware requirement is asserted as a current blocker.

## Once-final 58-test receipt

Candidate cwd; `/private/tmp/pab-0c-venv/bin/python -I -B -`; explicit absolute candidate and candidate/tests/service sys.path entries. Printed/asserted Python **3.14.7**, Clang 17.0.0, not target Python 3.11. defaultTestLoader.loadTestsFromNames; asserted count 58; TextTestRunner(verbosity=2,failfast=True); nonzero exit on failure. Direct unittest avoids pytest maintenance autouse and socket/uvicorn fixtures.

Exact selection:

- tests.release.test_native_history.NativeHistoryTests — all 34.
- tests.release.test_native_activation.NativeActivationTests — all 8.
- tests.release.test_native_enrollment.NativeEnrollmentTests — all 10.
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- tests.service.test_installed_activation.ActivationTests.test_preowner_refusal_never_constructs_core.
- Same ActivationTests.test_post_core_private_reader_under_lifetime_lock.
- Same ActivationTests.test_changed_post_acquire_hello_refuses_and_releases_core.
- Same ActivationTests.test_changed_post_acquire_second_hello_refuses.
- tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change.

**58 tests / 38.069s / OK / exit 0**, session 54283, chunks 34241c/5a0bfc/536c7b. Ran once, no final-selection retry. r311's retained initial helper AttributeError and later 58 PASS / 39.073s are historical receipts, not unexplained product failure and not counted as this review's execution.

`git diff --check base..HEAD` passed. Candidate remained clean at exact report HEAD after all execution. Observed available disk 2,948,152 KiB, below 6 GiB. Only authorized tiny inert TLS/filesystem/audio/ZIP/process observations occurred. No network/listener, Docker, install/download, actual child, ML, crash, HELD/STT or operational cleanup. Fixture-owned temporary teardown only. HEAVY ACTUAL headroom was not exercised or waived.

## Parent-owned gates

Correct F1 in a new pinned candidate with focused evidence-preserving refusal tests and obtain fresh independent review before source adoption. Do not self-adopt this report or substitute 58 PASS for the reproduced acceptance gap.

Authenticated installed public lifespan/credentials, topology/mounts, fresh owned voice onboarding and stopped boot/trust renewal remain separate work. No reviewed arm64 wheelhouse exists. Actual dependency/install/import/ABI, TLS/topology/denied peers, process/FD/death/resource, hardware/model/MPS, generate/cancel/restart, clone lifetime and audio/performance/release qualification remain parent-owned and unrun. **Native Metal remains UNAVAILABLE / UNQUALIFIED.** CPU and CUDA release rows remain mandatory. Legacy lib/tts digest RED is unchanged/unwaived; no digest regeneration/test. This SOURCE/inert review confers no operational, hardware or release acceptance.
