# Independent corrected bounded native enrollment review — r306 Sol

2026-10-25. **Bounded GO for parent consideration/adoption of r303+r305 SOURCE at the exact pins below. No remaining material finding in this finite enrollment/startup-preparation slice.** r304 F1/P3 is closed. Independent final selection: **58 PASS / 5.082s, exit 0**, once. This is not installed activation, full Lot B, Metal readiness or release acceptance. Parent adoption remains separate; no self-adoption occurred.

## Authority and verified pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-e31d5aee`, branch `t3/native-enrollment-exact-identity`. Verified clean at entry and after tests, exact HEAD `6492b82d8b13c5122efe1a7c743d6ab2c5d4a009`. Commit chain is base `bdd1259a8dc212e2aa73caeed9ddf252aae94884`, r303 SOURCE `80c6bef64d188e5c59d74862967274dfe8b79633`, r303 report `8e4b6caecb7b828e0e096dc70999920c0bd84c9f`, correction SOURCE `5ee444f503cb282fe93d2bf48318089b7cb67839`, then report HEAD. Base-to-HEAD diff changes only native_package.py, enrollment tests, NATIVE-PACKAGE.md and the two implementation reports. `git diff --check bdd1259..HEAD` passed.

Verified SHA256:

- `service/native_package.py`: `1f6a1e779319e0b74b3adf80c4eb7e3e0be5b3c57f6bc9c97d6baf067e3009d8`.
- `tests/release/test_native_enrollment.py`: `e3d881bd805b5bcc8e9ad00b118ae1e42f3b875f3eff5fe571299425b184a8d1`.
- Unchanged `lib/tts.py`: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known legacy digest RED remains unchanged/unwaived; no regeneration or legacy digest test.

Read PRODUCT.md, NATIVE-PACKAGE.md, normative r277 including B/C, r303, r304, r305 and r302; inspected the complete enrollment diff/tests, actual context/admission, private path/document readers, NativeRuntime construction/hello/import admission and protocol consumers. No applicable AGENTS.md found in candidate discovery or checked ancestors. Memory supplied scope/receipt conventions only, not technical findings. Sole lasting write is this root-checkout report. No candidate edits, commits, merge, push, delegation or operational cleanup.

## Findings and closure

**r304 F1/P3 closed.** At `service/native_package.py:534–543`, saved canonical bytes must equal reconstructed snapshot bytes before TLS. After hello, both reconstructed snapshot bytes and reread record bytes must match. Enrollment's own before/after comparison at `:515` also uses canonical bytes. JSON true, 1 and 1.0 consequently cannot exploit Python dictionary equality to stand in for exact record version or owner device/inode/UID. Duplicate-key, nonfinite and canonical-syntax refusal remains in the shared document reader. The two regression methods each execute five substitutions: version true/1.0 and float device/inode/UID. They establish the Python equality gap, refusal before any extra fake TLS request, and refusal when record substitution occurs during hello, without repairing changed evidence. Both passed independently.

**No further material issue found in the full bounded slice.** `_enrollment_snapshot:462–501` binds owner literal path/device/inode/UID, explicit configuration/templates paths, canonical document hashes, individual CA/client certificate/client key hashes and the exact five-field native tuple. Closed bounded registration rows reject duplicate principal/profile/version keys. Actual `NativeQwenContexts:287–306` validates native binding, execution/version/profile, hardware identity/freshness, frozen plan and profile admission; finite_budget rejects noninteger/nonfinite/out-of-range budget shapes. These checks validate trusted source registrations; they do not create owned catalogue versions or authenticate actual Darwin model bytes.

`enroll_native:504–524` validates snapshots around actual enrolled_runtime, refuses observed changes, creates an exclusive no-follow record with mode 0600, flushes/fsyncs it and fsyncs its parent. An existing destination refuses even for identical input. Failure after creation retains a blocking partial record; no overwrite/repair path exists. No-follow component walking, private immediate directories and stable owner/mode/single-link regular-file reads remain in use. Missing/malformed/changed/permissive/linked records refuse on reopen.

`enrolled_runtime:396–420` constructs actual NativeRuntime with explicit numeric endpoint, exact peer pin and client trust, calls hello, compares every configured native identity including boot, requires nonquarantined state and open import admission. NativeRuntime validates the closed native tuple and actual pinned frame exchange; uncertain retained import evidence cannot be turned into healthy enrollment. The finite tests independently cover exact reopen, changed trust/config/templates/owner, wrong boot/quarantine, duplicate/changed native registration, mutations during hello and retained import refusal. They do not replace real TLS or hardware qualification.

The dedicated-account, immutable operator-input and no-hostile-same-UID-writer premise is explicit. Trust is reopened by TLS path; repeated snapshots are not an atomic filesystem transaction, FD-pinned TLS loading or hostile-writer/ABA defense. Owner device/inode/UID identifies the selected directory incarnation, not an unforgeable process lifetime. Stopped deployment and reconciliation are operator preconditions, not a new lock claimed by these APIs. These limits are appropriate to this bounded preparation slice and do not constitute a request to implement deferred Lot B here.

No changed code constructs Dispatcher/Core, reads/blesses completed native history, widens Linux enrollment policy, publishes artifacts, releases reservations or transfers Core/SQLite callbacks into TLS. Existing owner-thread, publication and import-intent fences remain untouched. Returned runtime/contexts/profiles/budget are trusted private preparation inputs, not installed activation or readiness credentials. Fixture profiles bearing ready/qualified labels remain explicitly inert and supply no product readiness authority. Linux native-history refusal remains in force.

## Independent finite execution receipt

Inspected both conftests before execution: root maintenance autouse guard and service socket/uvicorn fixtures. Used direct unittest, avoiding pytest collection/fixture activation. Inspected selected fixture setup: tiny disposable files, metadata-only ZIPs/model fixtures, fake certificate/TLS channels, injected process/host observations; archive help executes in-process. No actual child or listener is required by this selection.

Exactly one final test invocation, candidate cwd:

```
/private/tmp/pab-0c-venv/bin/python -I -B -
```

The stdin script explicitly prepended candidate root and candidate `tests/service` absolute paths to sys.path, loaded the four names below with defaultTestLoader.loadTestsFromNames, asserted count 58 before running, and used TextTestRunner(verbosity=2, failfast=True), with nonzero exit on failure. Observed interpreter `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`, **not target Python 3.11**.

1. `tests.release.test_native_enrollment`: 10 PASS.
2. `tests.release.test_native_package`: 27 PASS.
3. `tests.service.test_installed_host_policy.InstalledHost`: 20 PASS.
4. `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`: 1 PASS.

Combined runner receipt: **Ran 58 tests in 5.082s; OK; exit 0** (tool chunk `cf1988`). No rerun, development execution or additional probe. Raw verbose output is retained in the conversation tool receipt. Candidate remained clean at the pinned HEAD afterward.

The parent's clarification authorizes these tiny inert fixtures below 6 GiB. No disk gate was imposed on this selection; the heavy ACTUAL-work gate remains unchanged. Only fixture-owned temporary teardown occurred. No network/listener, Docker, install/download, actual child, ML, crash, HELD or STT work, and no operational cleanup.

## Remaining parent-owned gates

Parent may separately adopt the exact reviewed SOURCE. Deferred Lot B is not a current bounded-slice blocker: owner-thread startup checks under Core exclusion, native installed identity/completed-history and publication/import-intent validation, entrypoint/lifespan/credential wiring, topology/mount policy, stopped boot/trust renewal and supported empty-catalogue onboarding remain distinct work. Preparation alone must not enable installed Dispatcher activation.

Actual Darwin interpreter/artifact acquisition and installed provenance/import/ABI, real authenticated TLS/reachability/firewall/denied peers, model mapping, process/FD/death/resource observations, Apple Silicon probe/load/generate/cancel/restart, clone conditioning lifetime and audio/performance remain unqualified. CPU and CUDA release rows remain mandatory separately. **Native Metal remains UNAVAILABLE / UNQUALIFIED; legacy lib/tts digest RED remains unwaived.** No source test receipt closes these gates.
