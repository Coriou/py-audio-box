# Independent native history SOURCE review — r314 Sol

2026-10-25. **Bounded NO-GO for adoption of the corrected r309+r311+r313 candidate.** r312 F1's command-sequence defect is closed. One remaining P2 exact immutable publication-intent join defect was independently reproduced below. The authorized final selection ran once: **67 PASS / 48.690s / exit 0**. Passing tests do not waive this acceptance gap. No self-adoption or candidate correction.

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-57da2e53`, branch `t3/native-history-sequence-fix`. Verified clean at entry and after all execution, exact HEAD `77dc43f3d6fbd7ac970629de61d8a1c7af26abf5`.

Verified contiguous parent chain, oldest first:

- Base `55ad442c9badb8eb02c749d6b4ea0a4146a2ca9d`.
- Inherited `1798e42341337eab4de38ea05b98d8726f816927`.
- Inherited `da4d09ad69af8e25390f89c5fffcd43e66751219`.
- Inherited `719823a1fdbf279e811341bad9d65b14fff3c732`.
- Inherited `f0e85fa68264df3f260cdeca0a370fc23b5d131f`.
- Inherited `b9e28ce7fedef1b51caa67157e4443afc8da9e56`.
- Fix SOURCE `f1c434eb5cc4e318bc6d9be09491a299b3208fab`.
- Report HEAD `77dc43f3d6fbd7ac970629de61d8a1c7af26abf5`; only r313 documentation differs from fix SOURCE.

Independently checked SHA256: service/native_activation.py `c3366709336b0a768cb682bd70b28b8b8066a7d446e7b4f39e48b841a1aa4003`; tests/release/test_native_history.py `4f3586cb0ce34d01a7c1c6cd65e41827fe929da5c35b5e60e24be10768ecc83c`; lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

Read r312/r313 and r310/r311/r309, PRODUCT.md, NATIVE-PACKAGE.md and relevant r277 contract. Inspected base-to-HEAD source changes, actual Dispatcher/DispatchStore producer, NativeActivation validator, Core exclusion/preinitialization, detached schema/document validators, Authority, native terminal/runtime/import accounting, Admission and publication-intent/reconciliation consumers and fixture seams. No applicable AGENTS.md found in candidate discovery or checked ancestors. A bounded memory registry search supplied no relevant native-history guidance; no memory-derived technical conclusion is used.

Sole lasting write is this root-checkout report. No candidate edits, commits, merge, push, delegation or operational cleanup.

## F1 — P2: publication intent still accepts a nonidentical immutable artifact

The new retained reader delegates publication-work validation to `_installed_detached_documents` and Reconciliation, both using `service.persistence.publication_record`. Its `publication_intent` comparison at `service/persistence.py:296–302` uses Python dictionary/scalar equality for the artifact in `publication_work.binding` versus the retained publication artifact and job. Python equates integer generation with integral float generation. The canonical artifact-metadata comparison at `service/native_activation.py:203` checks the artifacts table, not this publication-intent copy, so it does not close this join.

Independent exact reproduction, separate tiny completed-history fixture:

```python
t.prepare()
c = Core(t.root)
try:
    v = json.loads(c.db.execute('SELECT binding FROM publication_work').fetchone()[0])
    v['generation'] = float(v['generation'])
    c.db.execute('UPDATE publication_work SET binding=?', (json.dumps(v),))
finally:
    c.close()
d = t.reopen()
```

Only the publication-work binding document was changed. Job, frozen Authority, publication artifact, artifacts metadata, native receipt/proposal/output and files were untouched. Result: **ACCEPTED; installed_checked=True; quarantined=False; extra ACKs=0**. Probe session 18254, chunks b7de6e/70105b, exit 0. This is an invalid-acceptance reproduction, not a passing correctness assertion.

This is the primitive-identity defect class of r310 F5 at another immutable join used by the advertised exact publication-intent proof. r311's four exact reproductions remain closed, but their closure does not establish this additional join. The reader clears import closure and declares checked history despite publication intent no longer retaining the identical artifact representation.

Require canonical primitive identity between this intent artifact and the corroborating publication/job representation, without rewriting evidence. Add this exact focused refusal regression with unchanged logical SQL/files, no extra ACK and uncleared import closure. A narrow native-reader corroboration can close this without broadening shared Linux policy; any shared-helper change needs appropriate consumer review.

The reproduction does **not** establish forged audio, bypassed reap, duplicate inference, resource undercharge or publication without a valid native terminal receipt. Severity is P2 retained-evidence integrity within the stated SOURCE contract. It is not a hardware or deferred public-service requirement.

## r312 F1 closure and other reviewed obligations

`service/native_activation.py:161–173` now accepts exactly the stated per-dispatch operation language: `prepare,start,(observe|stop)*,observe,collect,ack+`, all returned. Prefix plus the closed middle and suffix enforce exactly one prepare/start/collect; a collect requires a directly preceding observe, and nothing but ACK follows collect. Unknown operations/states refuse globally; detached SQL foreign-key validation binds every command to an actual dispatch. No arbitrary RPC retry semantics are inferred.

Actual Dispatcher `_tick` launches collect only after consuming STOPPED observe; successful collect calls `_terminal`, records outcome, and proceeds to ACK. The new two exact r312 refusal cases and nearby stop-before-collect, observe-after-collect and missing-ACK cases pass. Real Dispatcher three-observe positive passes with the exact journal asserted; its positive history is not rewritten. The refusal helper compares post-mutation logical SQLite dump and spool/staging/assets bytes, checks no extra ACK, and checks pending import closure remains when a runtime was constructed.

Observe response states are not journalled. This review proves the supported operation progression and the separately retained/current native terminal evidence, **not historical STOPPED response reconstruction**. Stop-loop and ACK suffix support are source-language observations, not new actual lifecycle qualification.

Two additional independent boundary probes used separate NativeHistoryTests fixtures and the existing evidence-preserving refusal helper:

- Rewrite commands to returned `prepare,prepare,start,observe,collect,ack`: refused.
- Disable fixture connection foreign keys and insert returned observe with dispatch_id 999999: refused by ordinary reopen.

Both preserved logical SQL/files, added no ACK, and retained import closure. One finite stdin invocation, session 96569, chunks c11777/94d410, exit 0. No retry. These check duplicate-prefix and orphan-command joins beyond the six new tests.

The five original r310 exact reproductions remain closed: finite exact integer resource dimensions/canonical frozen native_demand; retry_of IS NULL; production fingerprint recomputation; closed command namespace and progression; canonical context/native/job/outcome/artifacts-metadata identity. The additional publication-intent join above remains material. Live Admission retains authorization/current qualification and hardware freshness before shared native_demand; historical decode does not expire frozen inventory or re-reserve charges.

Core's sole SQLite connection invokes exact-type NativeActivation under lifetime flock before schema/catalogue initialization. Canonical DDL/index/trigger identity and detached constraints precede retained interpretation; missing entire catalogue cannot be recreated into acceptance. Runtime construction and SQLite remain on the control writer; synchronous authenticated collect gets detached Authority and no SQLite transaction spans TLS.

Native terminal receipt validation, same configured boot/trust, retained/current receipt/proposal/output, actual assets/staging bytes, import intent/completion and exact spool membership remain mandatory. No new repair, ACK, history deletion, scheduler, readiness promotion or boot learning was found. recovery.py, reconciliation.py and native_runtime.py and lib/tts.py are unchanged against base; explicit Linux native refusal remains intact and its selected regression passes.

## Once-final 67-test receipt

Inspected tests/conftest.py and tests/service/conftest.py before execution. Direct unittest avoids pytest autouse maintenance and socket/uvicorn fixtures. Inspected injected TLS/owned process/backend seams and inert spawn guard. No actual child or listener was executed despite some historical test names.

Candidate cwd; `/private/tmp/pab-0c-venv/bin/python -I -B -`; explicit absolute candidate and candidate/tests/service sys.path entries. Printed/asserted Python **3.14.7**, Clang 17.0.0, not target Python 3.11. `defaultTestLoader.loadTestsFromNames`, count asserted 67, `TextTestRunner(verbosity=2, failfast=True)`, nonzero exit on failure. Exact selection:

- tests.release.test_native_history.NativeHistoryTests — all 40.
- tests.release.test_native_activation.NativeActivationTests — all 8.
- tests.release.test_native_enrollment.NativeEnrollmentTests — all 10.
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- tests.service.test_installed_activation.ActivationTests.test_preowner_refusal_never_constructs_core.
- Same ActivationTests.test_post_core_private_reader_under_lifetime_lock.
- Same ActivationTests.test_changed_post_acquire_hello_refuses_and_releases_core.
- Same ActivationTests.test_changed_post_acquire_second_hello_refuses.
- tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change.
- tests.service.test_native_lifecycle.NativeLifecycleTests.test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job.
- Same NativeLifecycleTests.test_stale_future_native_registration_and_historical_decode.
- Same NativeLifecycleTests.test_cancel_retains_reservation_until_actual_owned_receipt.

**67 tests / 48.690s / OK / exit 0**, session 78963, chunks ca5d41/277a85/04098b/98b23d. Executed once, no retry or broad testing. Supplemental probes were separate from this selection and use the same asserted Python and explicit paths. No failed invocation occurred; F1's accepted mutation is retained as a finding.

`git diff --check 55ad442..HEAD` passed. Candidate remained clean at exact report HEAD after execution. Observed free disk 2,912,988 KiB, below 6 GiB. Only authorized tiny inert filesystem/audio/model/ZIP metadata and fake TLS/process observations ran. No network/listener, Docker, install/download, actual child, ML, crash, HELD/STT or operational cleanup. Fixture-owned temporary teardown only. HEAVY ACTUAL headroom was not exercised or waived.

## Remaining parent-owned gates

Correct F1 in a pinned candidate with an evidence-preserving regression and obtain independent review before source adoption. This report does not self-adopt or merge anything.

Accepted-history scope stays finite: zero jobs or fewer than 64 clean succeeded single-attempt non-clone native jobs, same immutable identity/boot/trust, 2048 SQL rows/16 MiB retained scalar budget plus existing runtime limits. These budgets bound materialization, not hostile database scan time. Dedicated account/immutable inputs/no hostile same-UID writers remain premises; repeated snapshots do not provide hostile-writer isolation. Failed/cancelled/retried/clone/crash/unfinished histories remain unsupported.

Deferred authenticated public lifespan/credentials, topology/mounts, fresh owned onboarding and stopped boot/trust renewal are separate work, not current blockers asserted here. No reviewed arm64 wheelhouse exists. Actual dependency/install/import/ABI, TLS/topology/denied peers, process/FD/death/resource, hardware/model/MPS, generate/cancel/restart, clone lifetime, audio/performance and release qualification remain parent-owned and unrun. **Native Metal remains UNAVAILABLE / UNQUALIFIED.** CPU/CUDA release rows remain mandatory. Legacy lib/tts digest RED is unchanged/unwaived; no digest regeneration or legacy digest test. SOURCE/inert evidence grants no operational, hardware or release acceptance.
