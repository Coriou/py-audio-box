# Native preset authorization fix — r321 Sol

2026-10-25. Implementation candidate for fresh independent SOURCE review. F1's working-registration authorization bypass is removed; no self-adoption or installed/hardware/release GO. Metal remains **UNQUALIFIED**, with no reviewed arm64 wheelhouse.

## Authority and exact pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-bc47df4f`, branch `t3/native-preset-auth-fix`. Clean entry parent `3c640ae0cafe985e46dddd6e5af87e1b6250dd10` did not contain r319. Requested SOURCE `dee425feb460a44987560a59e9fadc2c36c16c2d` cherry-picked cleanly to `27006c9c23b0d1a750aaa640e5e0d640249cf24c`; requested report `716a106148d170cd8debdc3d594b48dbf7bc13fb` to `a507eecb78c2a3c09f8f42c2662dc0b2198aec0b`. Fix SOURCE/tests commit **`72fcdba3962abb3ff8ff2dc4f2b8adce1589e938`**. This report is a subsequent report-only commit.

Read full H4-NATIVE-ONBOARDING-REVIEW-r320-sol.md and r319 report, actual register_snapshot/registration/prepare, original-template/working-registration roles, NativeQwenContexts resolution and shared derivation, and fixture consumers. No applicable AGENTS.md found in worktree/checked ancestry discovery. Memory registry search had no relevant hit; inspected one-shot skill scope, which excludes ordinary unit-test work; no memory-derived technical conclusion. No root-checkout write, merge, push, delegation or self-adoption.

## Change and bounded adjacent joins

Both original exact-template allocations and owned preset derivations now authorize exclusively from `_preset_templates`, with canonical primitive identity of principal, request-independent context and profiles. Working registrations remain bounded allocation records, never template authority. Every owned repeat reconstructs from original enrollment. Request/allocation handling, Authority, admission, principal/profile/model/native bindings and existing 64-registration cap remain in place.

The exact r320 descriptor substitution (`top_p: 1.0` to `1`, otherwise unchanged) now refuses before legitimate registration, after it, and on repeated attempts for original and owned contexts. Canonical originals and owned repeats remain accepted. Tests assert Python equality and canonical inequality explicitly.

Inspected the same-class joins in this changed route. `prepare` binds canonical full snapshot SHA256 before lookup/replay; binding identity therefore already distinguishes primitive substitutions. Its registration consumer additionally now compares canonical snapshots instead of Python equality, preventing an existing binding from masking a primitive-different supplied snapshot. Native `resolve_registered` now compares the saved version against the trusted original/derived version canonically, closing the equivalent numeric representation gap for both resolution branches. Tests exercise both joins and valid repeats. Typed execution/hardware checks continue after snapshot authorization; trusted bundles/inventory/execution metadata are copied from original enrollment. No generic equality framework or unrelated Linux policy change.

NativeFixture explicitly binds native identity after supervisor construction; NativeLifecycleTests explicitly substitutes trusted principal/profile/catalogue data during fixture setup. Those fixture enrollment steps now synchronize the original-template store. This is test-only synchronization, not a production enrollment hook or added database seed. Existing lifecycle fixture seeding is unchanged. Public empty-catalogue onboarding still uses the existing CLI/HTTP/MCP registration and generation route, with authenticated scopes, activation, owned immutable versions, idempotency, artifact digest and same-root restart replay.

No public trusted configuration, self-enrollment, new API/scheduler, readiness promotion, voice fallback or Core schema/history/publication/import/owner fence change. Contracts, lib/tts.py and Linux engines are unchanged by this fix.

## Execution receipts

Read `tests/conftest.py` and `tests/service/conftest.py` before direct unittest. No pytest/autouse maintenance or listener fixture execution. Every invocation used `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit absolute worktree and worktree/tests/service sys.path; printed and asserted Python **3.14.7**, full runtime `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`. Not Python 3.11. `defaultTestLoader`, `TextTestRunner(verbosity=2, failfast=True)`, unsuccessful result maps to nonzero exit.

Initial 97-case attempt session 67965 (initial f32e09, terminal 7b3587): **9 passed, case 10 failed / 11.354s / exit 1**. All new tests and public journey passed; first history case timed out after prepare because its post-constructor fixture enrollment existed only in working registrations. This failure is not passing evidence. Corrected only explicit fixture enrollment synchronization as described above.

Final corrected relevant selection, once, session 61794 (initial 909812, terminal 7a60b2): **97 PASS / 57.262s / exit 0**. No source/test edits after this selection. Exact selection is the r320 92 cases plus three new onboarding methods and two narrow supervisor regressions:

- NativeOnboardingTests (6), InstalledPresetTests (3), NativeHistoryTests (44), NativeActivationTests (8), NativeLaunchTests (8), NativeEnrollmentTests (10).
- Full tests.worker.test_native_protocol, tests.clients.test_production_clients and tests.service.test_production_api.
- NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- NativePackageTests.test_archive_deterministic_bytes and test_extra_archive_member_refuses.
- NativeSupervisorTests.test_completed_oldboot_then_new_registered_work and test_prepare_replay_tuple_session_and_stop_ack_restart.

Separate narrow original future-allocation regression, once, chunk 533f91: NativeRuntimeTests.test_future_allocation_authorizes_from_template_without_control_hook, **1 PASS / 0.212s / exit 0**, same interpreter/path setup. It authorizes a fresh allocation from original enrollment and replays prepare. Expected injected close_unknown shielded-future traceback and argparse refusal usage appeared in the final selection with passing assertions. `git diff --check` passed.

SOURCE SHA256:

- worker/native_supervisor.py: `6858e310712805f47e0ac616d0a4358d26ac55980d86842fca8711463c302e6d`
- service/execution_context.py: `07879efa5cc31c97a24562e13bf922a2bd4f5e3e7fab5a28d1a3b30e8a22d838`
- tests/release/test_native_onboarding.py: `07054425771298ef1721d48ccf8d31cf3c96acb7106648c04cc5b313ddae24ca`

## Remaining limits and parent-owned gates

Only authorized tiny inert fake TLS/ASGI/filesystem/audio/ZIP/owned-process observations and fixture-owned teardown. Observed available disk 2,881,224 KiB, below 6 GiB; no heavy ACTUAL operation or gate waiver. No network/listener, Docker, install/download, actual child, ML, crash, HELD/STT or operational cleanup.

Fresh independent review and parent adoption remain required. Source tests establish no actual installed/hardware/acoustic authority. Same boot/trust, retained finite history and exact Authority/receipt/publication/import fences remain required. Actual arm64 ABI/wheelhouse/install, pinned TLS/denied peers, Docker topology, hardware/model/MPS generation/cancel/restart, process/resource behavior, clone lifetime, audio/performance and release acceptance remain unrun. CPU and CUDA remain mandatory qualification rows; Metal remains UNQUALIFIED. Managed clone onboarding and stopped/reconciled renewal remain deferred.

Legacy lib/tts.py digest RED is unchanged and unwaived: SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. No digest regeneration or legacy qualification rerun.
