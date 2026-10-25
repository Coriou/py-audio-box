# Independent native public launch SOURCE review — r318 Sol

2026-10-25. **Bounded GO for parent consideration of r317 SOURCE.** No material defect established in the explicit native authenticated launcher/lifespan slice. Independent exact final selection, once: **106 PASS / 70.001s / exit 0**. One additional post-acquisition credential-change probe passed. This is no self-adoption, operational acceptance or release qualification. Native Metal remains **UNAVAILABLE / UNQUALIFIED**.

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-4bc474cd`, branch `t3/native-public-launch`, clean at entry and after execution:

- Base `87c4908730d4672dfccf3c5f7ddb7d2ee8db8591`; independently verified merge-base equals this exact base.
- SOURCE/tests `29d6628e87379e7aae2b42c5dc127808454ae084`.
- Resolved report HEAD `341c852e2cdbac8eeb25859e937fccb783c5168d`. SOURCE-to-HEAD changes only `H4-NATIVE-PUBLIC-LAUNCH-r317-sol.md`.

Independently verified SHA256: native_launch.py `77f1cb0a1fb25d4a3a3b4ecc68e2a51c6b55dff249aedd1bc3259e3f481ea08e`; entrypoint.py `743e7fba5d2e9509300c63ebce5d5091374cf7292b7a553521708f830a1b5671`; test_native_launch.py `4f97fa97bf93b5cae10078780549c27550ac0664c2bb39db0d756a15069b4122`; unchanged lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

Read PRODUCT.md, NATIVE-PACKAGE.md, normative r277, r316 and r317, actual base-to-SOURCE diff and report-only delta. Inspected entrypoint/thread-result/lifespan, auth, HTTP discovery/status/write authorization, CLI/MCP shared HTTP consumer, NativeQwenContexts, NativeActivation and owner factory, native private filesystem readers, archive producers/manifests and selected test seams. No applicable AGENTS.md found in candidate/repository discovery or checked ancestors. Bounded memory-registry search returned no relevant hits; no memory-derived technical conclusion.

Sole lasting write is this root-checkout report. Existing root untracked STT evidence was listed, not read or changed. No candidate edits, commit, merge, push, delegation or operational cleanup.

## Findings and source assessment

**No blocking finding.** The new command consumes `create_native_owner` with explicit paths; it introduces no alternative API, scheduler, Core, public trusted-configuration route, UI or STT. Native runtime/context construction remains on the Dispatcher writer thread. Existing NativeActivation precheck, Core lifetime exclusion, retained-history proof, postcheck and persistent activation precede public mounting.

The `_owned_app` extraction retains the Linux thread-owned Future, shielded acquisition, cleanup advancement and process-lifetime strong registry. Cancellation does not cancel the underlying acquisition or close operation. Eventual acquired owners are closed once; failed/unknown close stays retained without automatic retry. Linux activation/notification and command signal behavior continue through the existing implementation. Selected direct wrapper cancellation, repeated cancellation, callback/loop closure, preparation/close cancellation, failed-close and receipt-drain regressions passed.

The previous mount is removed before the new authenticated app is mounted on a second successful lifespan. The two-lifespan native test makes authenticated requests through both mounts and verifies both owners close. The development old-mount failure in r317 is consistent with the corrected source. The reported development cancellation guard was a test sequencing deadlock; the final test releases eventual acquisition after repeated cancellation and asserts the retained obligation rather than treating cancellation as close authority. No failed final run or retry occurred here.

Native credentials reuse the closed existing decoder and constant-time token resolver. Native no-follow readers enforce owner UID, 0600, regular/single-link stable file; the credential parent is checked as 0700. Principals must occur in templates; read is mandatory and the only additional scope is production-write. Existing write endpoints resolve that scope again; authentication supplies the principal to canonical ownership operations. Raw credential bytes and canonical template bytes are compared before acquisition and before mounting. Native enrollment independently binds templates/control/trust/owner before and after Core acquisition. These checks are appropriate under the explicit immutable operator-input/dedicated-account/no-hostile-same-UID-writer premise; they do not establish hostile-writer isolation or live rotation.

Native capabilities now describe registered principal metadata and Dispatcher substitutes the current configured profile. This prevents synthetic fixture profile fallback for valid registered native principals without changing inventory timestamps, profile qualification or support. Native `/skeleton` explicitly reports non-synthetic installed ownership, persistent activation separately, speech readiness false and UNAVAILABLE / UNQUALIFIED. No fresh-catalogue native generation journey is demonstrated by the new public read tests; that is a qualification/onboarding limit, not a blocker for this slice.

The CLI requires every input path plus explicit loopback host and port, rejects non-loopback and out-of-range ports, disables access logs and uses critical Uvicorn logging plus a fixed redacted launch failure. It reuses TERM/INT ordinary drain and requires a started server and closed/error-free lifecycle before success. Source archive allowlists add exactly the native launch module and script; deterministic/exact-membership, secret exclusion and extra-member refusal checks passed. Archive help in this selection is in-process runpy, not an actual child. The command does not validate Docker routing/mount/firewall topology.

## Independent execution receipt

Read tests/conftest.py and tests/service/conftest.py before execution. Used direct unittest, avoiding pytest maintenance/autouse and listener/subprocess fixtures. In-memory TLS, injected server/process/backend/channel observations and tiny filesystem/ZIP/ASGI fixtures only. Names containing “actual” identify source integration paths with injected process/backend seams, not actual child/hardware evidence.

Exactly one final invocation from candidate cwd: `/private/tmp/pab-0c-venv/bin/python -I -B -`; explicit absolute candidate and candidate/tests/service sys.path entries. Printed and asserted Python **3.14.7**, full runtime `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`, not Python 3.11. `defaultTestLoader.loadTestsFromNames`, asserted count 106, `TextTestRunner(verbosity=2, failfast=True)`, nonzero on failure.

Exact r317 selection reproduced:

- NativeLaunchTests — all 8; NativeHistoryTests — all 44; NativeActivationTests — all 8; NativeEnrollmentTests — all 10.
- tests.service.test_installed_activation.ActivationTests — all 22; PrivateInputs — all 3.
- NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- NativePackageTests.test_archive_exact_members_secrets_excluded_extracted_help; test_archive_deterministic_bytes; test_extra_archive_member_refuses.
- tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change; InstalledCommandTests — both; SignalPolicyTests — its one case.
- NativeLifecycleTests.test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job; test_stale_future_native_registration_and_historical_decode; test_cancel_retains_reservation_until_actual_owned_receipt.

**Ran 106 tests in 70.001s / OK / exit 0**, session 51211, initial chunk 4e0a5f, terminal chunk 8eff9c. No retry. Expected injected `close_unknown`/`injected_close_unknown` shielded-future diagnostics appeared; retained-owner/no-retry assertions passed. Expected argparse refusal usage also appeared.

Supplemental independent adversarial probe (separate from final selection): one NativeLaunchTests fixture, existing fake TLS connector, wrap the real create_native_owner, mutate credential bytes only after successful owner return, then enter lifespan. Required exact `native_launch_inputs_changed`, no `/` mount, closed lifecycle, exactly one owner and dead owner thread. **PASS / exit 0**, session 33208, chunks c8fd93/6ffe9b. This directly exercises the acquisition-to-mount input bracket and cleanup, without a listener or real child.

Candidate `git diff --check 87c4908..HEAD` passed; exact HEAD and clean status rechecked. Observed available disk 2,956,184 KiB, below 6 GiB. Only authorized tiny inert work and fixture-owned temporary teardown. No network/listener, Docker, install/download, actual child, ML, crash, HELD/STT or operational cleanup. The 6 GiB HEAVY ACTUAL gate was neither exercised nor waived.

## Remaining parent-owned gates

Parent adoption is separate. Existing finite history policy is unchanged: zero jobs or fewer than 64 clean succeeded single-attempt non-clone jobs under the same immutable configured identity/boot/trust, with existing row/scalar/import/receipt/publication bounds. Failed/cancelled/retried/clone/crash/unfinished retained histories remain unsupported. No renewal or history waiver follows from this GO.

Fresh native owned-voice onboarding, explicit Docker topology/mount/firewall/reachability validation and stopped/reconciled boot/trust renewal remain subsequent slices, not current SOURCE blockers. No reviewed arm64 wheelhouse or compatible installed dependency/interpreter ABI is supplied. Actual install/import/ABI, authenticated TLS/denied peers, process/FD/death/resource behavior, hardware/model/MPS, generate/cancel/restart, clone lifetime, audio/performance and release qualification remain unrun. CPU/CUDA remain mandatory release rows. Known legacy lib/tts digest RED remains unchanged/unwaived; no digest regeneration or legacy digest test. This source/inert GO grants no installed, hardware, acoustic or release readiness.
