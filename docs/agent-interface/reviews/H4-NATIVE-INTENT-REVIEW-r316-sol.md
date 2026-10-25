# Independent native history SOURCE review — r316 Sol

2026-10-25. **Bounded GO for parent consideration of the corrected r309/r311/r313/r315 SOURCE candidate.** No remaining material defect was established within the advertised finite native-history scope. r314 F1 is closed, including coordinated artifact-copy substitutions. Independent once-final selection: **71 PASS / 51.311s / exit 0**. One additional neighboring manifest-owner refusal probe passed. This review does not adopt, merge, qualify hardware or authorize release.

## Authority and exact pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-79171b09`, branch `t3/native-intent-canonical-identity`, clean at entry and after execution.

- Base `7799c5204d5ddbbb60a585413c9950388e69f91c`.
- Final SOURCE/tests `eaeaa42eb6acdaad1d40c3d996e17eb157d604fd`.
- Report HEAD `27fc299101dbbdbd464e1dae80d06fd3283520fa`; only the r315 report differs from final SOURCE.

Verified the contiguous parent chain and independently ran `git range-diff` between the original exclusive seven-commit range `55ad442c9badb8eb02c749d6b4ea0a4146a2ca9d..77dc43f3d6fbd7ac970629de61d8a1c7af26abf5` and `7799c5204d5ddbbb60a585413c9950388e69f91c..73d4d95a52313b3717c8da01ac3968fe36f9be9e`. All seven patches are identical (`=`), in order:

- `1798e42341337eab4de38ea05b98d8726f816927` → `5c700d13163b5b0c183814c25808c04ea00ddc92`.
- `da4d09ad69af8e25390f89c5fffcd43e66751219` → `35763ea2c2ed004f2a41391c286a63c61337f8df`.
- `719823a1fdbf279e811341bad9d65b14fff3c732` → `cd436255da146bd02bbf6a873f029f67cc7afcb8`.
- `f0e85fa68264df3f260cdeca0a370fc23b5d131f` → `395c709eddcca52859340ce3caa03fc06a0bc6d9`.
- `b9e28ce7fedef1b51caa67157e4443afc8da9e56` → `daaf78c4849b191dda6ece8597cc27ec5eb1fc2e`.
- `f1c434eb5cc4e318bc6d9be09491a299b3208fab` → `62065c8d430fa047aff3fb539d8ecb3432edfe67`.
- `77dc43f3d6fbd7ac970629de61d8a1c7af26abf5` → `73d4d95a52313b3717c8da01ac3968fe36f9be9e`.

Independently verified SHA256:

- service/native_activation.py: `4549ba923ef4667ec5321cc6a8e13815c8657f120255c23c41f18dd8035d0a5e`.
- tests/release/test_native_history.py: `17906dc923a2d6157ac5a69f2be0779dfbc418cbfb66aca7b028bd762ad90f43`.
- Unchanged lib/tts.py: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

Read full r314 F1/r315 and prior r312/r313/r310/r311/r309 reports, PRODUCT.md, NATIVE-PACKAGE.md, package metadata and relevant normative r277. Inspected the actual base-to-source changes and Core initialization/publication/barrier, Authority, Dispatcher/DispatchStore, native runtime import/RPC, Admission, detached schema/document validation, publication_intent/publication_record, publication oracle and Reconciliation consumers. No applicable AGENTS.md found in candidate discovery or checked candidate/root-report ancestors. Bounded memory-registry search returned no relevant native-history entry; no memory-derived technical conclusion is used.

Sole lasting write: this root-checkout report. No candidate edit, commit, merge, push, delegation or operational cleanup.

## r314 F1 closure and systematic join assessment

At service/native_activation.py:198–213, the native reader now checks three distinct relationships rather than relying on shared Python scalar equality:

1. Manifest owner tuple canonically equals the frozen Authority job/attempt owner tuple.
2. The entire retained artifact canonically equals Core._verified's producer representation: manifest artifact ID and ordered output IDs, frozen owner tuple, canonical manifest hash, frozen resolved identity, and integer project_revision=0.
3. Each publication_work intent artifact canonically equals that retained artifact; an optional versioned manifest copy canonically equals the publication manifest.

Core._verified at service/core.py:552 emits precisely that artifact representation. Revision zero is appropriate for the supported native preset/design producer. The reader still runs the shared closed document and publication validation; it does not admit prepared-reference pairs into the single-output non-clone slice. No shared persistence/oracle equality semantics were changed.

This closes the exact r314 intent-only generation float substitution, its revision neighbor, and generation/revision float substitutions coordinated across publication artifact, artifacts metadata and intent. All four new refusal regressions passed independently. The full-artifact reconstruction also covers nested resolved identity and every artifact field; matching the three copies alone is no longer sufficient. The explicit manifest-owner comparison prevents a coordinated manifest numeric substitution from inheriting the oracle's primitive-equality weakness.

Other joins were traced by their authoritative representation: canonical frozen context/native/job/outcome and artifact metadata; immutable Authority snapshot bytes and binding hash; canonical retained/current receipt/proposal/output; raw snapshot/import marker/output bytes; exact integer reservation dimensions and canonical demand; closed SQL identities and detached foreign keys. Remaining work destinations/temporary names/digests and IDs are strings, while work byte lengths receive explicit integer/bounds checks and actual-file validation. Current authenticated proposal corroboration includes the complete message and manifest. No additional material numeric-equality acceptance was established in this slice.

Independent supplemental probe: create one separate completed NativeHistoryTests fixture; change publications.manifest.generation to its float equivalent; recompute manifest_sha256 in all three artifact copies (publications.artifact, artifacts.metadata, publication_work.binding); use the ordinary retained refusal helper. **Refused**, logical SQLite dump and spool/staging/assets file bytes unchanged, no extra ACK, pending import closure retained and import_completed empty. This probes the neighboring manifest-owner boundary with internally coordinated artifact hashes. It is a refusal observation, not historical runtime qualification or an invalid-acceptance reproduction. Session 50507, chunks d479a6/3633e2, exit 0; no retry.

## Whole corrected slice

The original five r310 gaps remain closed: strict finite positive mandatory resource dimensions and canonical frozen native demand; retry_of IS NULL; production fingerprint recomputation; closed returned command namespace; canonical frozen immutable identity/outcome/metadata joins. Live Admission still performs authorization, qualification and current hardware freshness before the shared pure native_demand calculation. Historical decoding does not expire frozen inventory or re-reserve charges.

The r312 progression defect remains closed. The implemented language is exactly `prepare,start,(observe|stop)*,observe,collect,ack+`, all returned. Actual Dispatcher launches collect after consuming STOPPED observe, then records terminal outcome before ACK. The repeated-observation positive uses the actual Dispatcher producer. Observe response values are not journalled: this review does not invent historical STOPPED response reconstruction. Terminal authority instead remains the separately retained/current native receipt and proposal proof.

Core invokes the exact-type NativeActivation hook under its lifetime exclusion and sole SQLite connection before WAL/schema/catalogue initialization. Exact schema/index/trigger identity, materialization budgets and detached row constraints precede history interpretation. Missing catalogue cannot be repaired into acceptance by ordinary initialization. Runtime construction and SQLite stay on the control writer; collect receives detached Authority with no SQL transaction spanning TLS.

Same configured identity/boot/trust, healthy bracketed hello and exact launch count remain required. Native terminal proof, reservation, actual publication/staging bytes, exact spool membership, import intent/completion and current authenticated receipt/proposal/output remain mandatory. Import closure is discharged in memory only after the complete implemented history proof. No history repair, ACK, boot learning, readiness promotion or second scheduler was added.

service/recovery.py, reconciliation.py, native_runtime.py, persistence.py, contracts and lib/tts.py have no changes against this candidate's base. Explicit Linux native-history refusal remains intact; its selected regression passed. The use of shared detached validators does not invoke Linux history acceptance.

## Independent once-final receipt

Inspected tests/conftest.py and tests/service/conftest.py before execution. Direct unittest avoids pytest autouse maintenance and server fixtures. Inspected NativeFixture's in-memory TLS channel, injected supervisor Popen/handles/channels and native-child inert spawn guard/backend; no actual child or network connection ran. Existing test names containing “actual” identify source integration paths, not actual process/hardware qualification.

One final invocation from candidate cwd using `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit absolute candidate and candidate/tests/service sys.path entries. Printed/asserted **Python 3.14.7 / Clang 17.0.0**, not target Python 3.11. Used defaultTestLoader.loadTestsFromNames, asserted 71 cases, TextTestRunner(verbosity=2, failfast=True), nonzero on failure.

Exact r315 selection:

- tests.release.test_native_history.NativeHistoryTests — all 44.
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

**71 tests / 51.311s / OK / exit 0**, session 13065; chunks a167d9/e65719/600eaa/c4d6b3. Ran once, no failed invocation or retry. The supplemental probe above is separate and was not a repeated final selection.

`git diff --check 7799c52..HEAD` passed. Candidate remained clean at exact report HEAD after all execution. Available disk observed 2,885,596 KiB, below 6 GiB. Only authorized tiny inert TLS/filesystem/audio/model metadata/process observations occurred. No network/listener, Docker, install/download, actual child, ML, crash, HELD/STT or operational cleanup. Fixture-owned temporary teardown only. HEAVY ACTUAL admission was neither exercised nor waived.

## Remaining parent-owned gates

Parent adoption remains a separate decision; this independent review does not self-adopt. Its GO covers zero jobs or fewer than 64 clean succeeded single-attempt non-clone native jobs under the same immutable configured identity/boot/trust, with 2048 SQL rows/16 MiB retained scalar budget and existing runtime limits. These bound materialization, not hostile-database scan time. Dedicated account, immutable configured inputs and no hostile same-UID writers remain premises. Failed/cancelled/retried/clone/crash/unfinished histories remain unsupported.

Deferred authenticated public lifespan/credentials, topology/mounts, fresh onboarding and stopped boot/trust renewal are separate work, not current SOURCE blockers. No reviewed arm64 wheelhouse exists; **Native Metal remains UNAVAILABLE / UNQUALIFIED**. Actual dependency/install/import/ABI, TLS/topology/denied peers, process/FD/death/resource, hardware/model/MPS, generate/cancel/restart, clone lifetime, audio/performance and release qualification remain unrun and parent-owned. CPU/CUDA release rows remain mandatory. Legacy lib/tts digest RED is unchanged/unwaived; no digest regeneration or legacy digest test. SOURCE/inert evidence grants no operational, hardware or release acceptance.
