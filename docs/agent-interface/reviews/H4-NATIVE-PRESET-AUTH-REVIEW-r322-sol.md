# Independent native preset authorization SOURCE review — r322 Sol

2026-10-25. **Bounded GO for parent adoption consideration of r319+r321 SOURCE. No blocking finding identified.** r320 F1 is closed in the reviewed source and independent execution. Exactly one combined authorized selection: **98 PASS / 59.638s / exit 0**. No self-adoption or installed, hardware, acoustic, operational or release qualification. Metal remains **UNQUALIFIED**, without a reviewed arm64 wheelhouse. Legacy lib/tts digest RED remains unchanged and unwaived.

## Authority, pins and writes

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-bc47df4f`, branch `t3/native-preset-auth-fix`. Clean entry and final HEAD `1c917bf91a0b1a968c04a73dabdf7ebd10f59a14`. Base and verified merge-base `3c640ae0cafe985e46dddd6e5af87e1b6250dd10`; r319 cherry-picks `27006c9c23b0d1a750aaa640e5e0d640249cf24c` and `a507eecb78c2a3c09f8f42c2662dc0b2198aec0b`; fix SOURCE `72fcdba3962abb3ff8ff2dc4f2b8adce1589e938`. Fix-to-HEAD changes only the r321 report. `git diff --check 3c640ae HEAD` passed. No applicable AGENTS.md found in candidate/root discovery and checked candidate ancestors.

Sole lasting write is this report in `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/`. No candidate edit, commit, merge, push, delegation or operational cleanup. Read full r320 exact reproduction, r321 and r319 reports, PRODUCT, NATIVE-PACKAGE and normative r277; inspected actual delta, native execution/template/supervisor/Authority consumers, Dispatcher, Production, CoreVoiceStore and enrollment/runtime/launch/onboarding fixtures. Memory registry search returned no relevant hit; no memory-derived technical conclusion.

SHA256 independently matched:

- worker/native_supervisor.py: `6858e310712805f47e0ac616d0a4358d26ac55980d86842fca8711463c302e6d`
- service/execution_context.py: `07879efa5cc31c97a24562e13bf922a2bd4f5e3e7fab5a28d1a3b30e8a22d838`
- tests/release/test_native_onboarding.py: `07054425771298ef1721d48ccf8d31cf3c96acb7106648c04cc5b313ddae24ca`
- lib/tts.py: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`

## F1 disposition and adversarial assessment

**r320 F1 closed.** Previously Python equality over all working rows could accept descriptor `top_p:1` after legitimate derived registration with `top_p:1.0`, although original-template canonical authorization refused it beforehand. `register_snapshot` now selects authority exclusively from `_preset_templates`, for both exact original contexts and owned derivations. The comparison canonically binds principal, context with only plan.request removed, and full profiles. Each owned attempt reconstructs from an original template; working rows supply bounded allocations only. Neither successful original nor owned registration inserts into the template store. Constructor deep copies isolate caller and working mutations; production has no subsequent assignment to `_preset_templates`.

The independent selection executes `test_original_and_owned_registration_require_canonical_primitives`: it asserts Python equality and canonical inequality for the exact top_p substitution, refuses before/after legitimate registration and on repeated attempts for both original and owned contexts, and accepts canonical repeats. Existing-binding snapshot and saved original/owned version regressions also passed. `registration` now compares canonical full snapshots; `resolve_registered` compares canonical saved versions on both original and derived routes. `prepare` hashes the canonical full snapshot before replay lookup, so Python equality in the journal replay condition cannot admit a primitive-different snapshot with a different binding. Existing working-row equality occurs only after original-template authorization and a full Authority binding; it does not restore F1 authority.

The shared helper copies the original context and changes catalogue voice/version identity, corresponding request version and execution voice identity only. Deterministic version ID hashes principal, target and full original template, binding settings, model/bundle, profile and native deployment/configuration/lane/boot/runtime identity. Derivation does not mutate enrollment. The original request contributes to stable enrolled-template identity, while per-generation requests replace plan.request after resolution; registration authorization excludes that request but Authority/admission validate the actual frozen plan and input hash. Thus differing authored requests can allocate from one enrollment without becoming new enrollment authority.

Tiny independent supplemental probe completed with 12 refusals: principal, profile config_revision, native configuration digest, model checkpoint, descriptor temperature, and coherent forged version/request/descriptor identity, each before and after legitimate owned registration. Original template contents remained unchanged. A synthetic 64-working-row capacity edge refused a new allocation without mutating templates. This cap probe uses inert duplicate rows to reach the branch, not 64 actual jobs or durable-history qualification. Some malformed descriptor changes refuse during typed decoding; this evidence does not claim every mutation reached canonical matching. Existing focused test separately covers valid coherent forged identity and trust/profile changes.

Original exact-template support remains, including future allocation without a control hook, completed-oldboot/new-work and prepare replay tests. Allocation count stays capped at 64, journal at 64 lifetime records/8 unacked, with no eviction or renewal. Original template count remains bounded at construction. Native models/inventory/bundles are retained from enrollment and checked through existing typed execution, hardware/admission and bundle consumers.

## Public route and fixture legitimacy

The canonical closed HTTP `voice_preset_register`, CLI `voice preset-register` and MCP `speech_voice_preset_register` remain the route. Authentication supplies principal; public input supplies selector, target and expected revision plus idempotency key. Dispatcher retains installed readiness gating; Production delegates registration to existing CoreVoiceStore transaction, ownership, CAS, request fingerprint/replay and immutable-version trigger logic. Generation reloads the owned version and resolves it against trusted originals before Core admission/submission. No public trusted descriptor/configuration, DB seed, fallback, readiness promotion, alternate scheduler or schema is added.

The passing public journey starts with an empty catalogue, registers via CLI with HTTP/MCP replay, generates via HTTP, verifies artifact/asset bytes and SHA256, and closes/reopens the same owner and transport root. Registration and generation replay preserve the same job receipt and exactly one fake backend call. Missing/wrong/read-only credentials, selector/idempotency/CAS/version conflicts and foreign/altered versions refuse. Discovery still separates activation from unavailable/unqualified speech. Ambiguous selectors and clone templates do not acquire preset permissions.

Fixture synchronization is legitimate explicit test enrollment: NativeFixture replaces native identity/working registrations after construction; lifecycle setup replaces principal/profile/context. The test-only updates now bind the original-template store to those same enrollment inputs. No production synchronization hook was introduced. The onboarding public journey itself performs no catalogue seed. Existing seeded lifecycle fixtures remain seeded tests. Fake TLS/process/backend seams exercise actual public lifespan/Core/native protocol/child-function/publication consumers, without establishing actual child or installed authority.

Delta leaves Linux Kokoro/Pocket implementations, contracts/fixtures, Core history/publication/import/owner fences and lib/tts.py unchanged. Independent selection covers Linux preset bootstrap/restart and native-lane refusal/exact Linux receipt preservation; it is not a full Linux release suite. Archive deterministic-byte and extra-member refusal tests passed; shared helper remains in already packaged native_protocol.

## Independent execution and failed probe history

Read tests/conftest.py and tests/service/conftest.py before direct unittest. Their pytest autouse maintenance and HTTP listener fixtures were not run. Exactly one combined 98-case selection, session **37576**, initial chunk **553ee7**, terminal **3a5efc**. Invocation `/private/tmp/pab-0c-venv/bin/python -I -B -`, cwd candidate, explicit absolute candidate and candidate/tests/service sys.path. Printed/asserted Python **3.14.7**, full runtime `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`. `defaultTestLoader.loadTestsFromNames`; count asserted 98; TextTestRunner verbosity=2/failfast=True; unsuccessful result mapped to nonzero exit. **98 PASS / 59.638s / exit 0**, no rerun.

Exact selection:

- tests.release.test_native_onboarding.NativeOnboardingTests (6)
- tests.service.test_installed_preset.InstalledPresetTests (3)
- tests.release.test_native_history.NativeHistoryTests (44)
- tests.release.test_native_activation.NativeActivationTests (8)
- tests.release.test_native_launch.NativeLaunchTests (8)
- tests.release.test_native_enrollment.NativeEnrollmentTests (10)
- full tests.worker.test_native_protocol, tests.clients.test_production_clients, tests.service.test_production_api
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt
- tests.release.test_native_package.NativePackageTests.test_archive_deterministic_bytes and test_extra_archive_member_refuses
- tests.worker.test_native_supervisor.NativeSupervisorTests.test_completed_oldboot_then_new_registered_work and test_prepare_replay_tuple_session_and_stop_ack_restart
- tests.service.test_native_runtime.NativeRuntimeTests.test_future_allocation_authorizes_from_template_without_control_hook

Expected injected close_unknown shielded-future traceback and argparse refusal usage appeared with passing assertions. r321's historical initial 97 attempt (9 passed then unsynchronized history fixture failed), corrected final 97 PASS/57.262s and separate future-allocation PASS are report evidence only; this review independently executed the corrected source above.

Supplemental probe history is separate from the one-shot suite. Session 44024 exited 1 before authorization because canonical serialization of the entire fixture registration map encountered a PosixPath bundle. Session 96605 exited 1 during construction of an incoherent forged version (`voice_version_missing` from plan_hash), after earlier mutations had refused; neither is a complete passing probe. Corrected session **39384**, initial chunk **776ff4**, terminal **82ae4a**, exit 0, uses deep-copy comparison for fixture objects, coherent forged identity and hash construction before production validation. Output: `forgeries refused before/after 12 original templates unchanged`; `64 working allocation cap refused`. Same isolated Python/path setup, fixture-owned teardown only. No candidate repair occurred.

## Remaining parent-owned gates

Parent owns adoption; this report performs none. Preserve same immutable boot/trust, zero-job or fewer-than-64 succeeded single-attempt non-clone clean history, exact schema/Authority/receipt/publication/import bounds and owner exclusion. No cleanup, history waiver or automatic renewal authority follows from this review.

Only tiny inert ASGI/fake TLS/filesystem/SQLite/ZIP/fake-process observations were performed. Available disk observed **2,881,052 KiB**, below 6 GiB; no heavy ACTUAL operation or disk-gate waiver. No network/listener, Docker, install/download, actual child/ML, crash, HELD, STT or operational cleanup.

Managed clone onboarding, topology/mount/firewall/reachability and stopped/reconciled renewal remain deferred slices, not current SOURCE blockers. Reviewed arm64 wheelhouse/ABI/install, real pinned TLS/denied peers, hardware/model/MPS generate/cancel/restart, process/death/resource behavior, clone lifetime, audio/performance and release acceptance remain unqualified. CPU and CUDA remain mandatory release rows. Metal stays UNQUALIFIED. Legacy lib/tts digest RED remains unchanged/unwaived; no regeneration or legacy qualification rerun.
