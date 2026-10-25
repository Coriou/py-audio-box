# Native owned preset onboarding SOURCE implementation — r319 Sol

2026-10-25. Implementation candidate for fresh independent Sol review; no self-adoption or release GO. Native Metal remains **UNAVAILABLE / UNQUALIFIED**.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-09593ec2`, branch `t3/native-qwen-onboarding`, clean entry base `4c8f8c4d1ec5a663ca69ec4d6809b505a63c3146` (integrated reviewed r317). SOURCE/tests/documentation commit `dee425feb460a44987560a59e9fadc2c36c16c2d`. This report is a subsequent report-only commit.

Read PRODUCT, NATIVE-PACKAGE, normative r277, r318 and r271; inspected native contexts, production, voice store, Dispatcher, HTTP/CLI/MCP, activation/history, native runtime/protocol/supervisor and relevant fixtures. No applicable AGENTS.md found in checked ancestors or worktree discovery. Bounded memory registry search returned no relevant hit; no memory-derived technical conclusion. No root checkout writes, merge, push or delegation.

## Implemented behavior and authority boundary

The existing closed `voice_preset_register` HTTP operation, CLI `voice preset-register` and MCP `speech_voice_preset_register` now work with explicitly configured native Qwen custom-voice presets. No schema, operation, database DDL, scheduler, publication or Linux runtime change. Requests still supply only preset, target voice ID and expected revision plus the existing idempotency key. The existing installed activation/readiness fence remains necessary; registration never promotes readiness.

NativeQwenContexts advertises principal-scoped, unambiguous explicit selectors. A shared pure derivation copies a trusted native custom-voice template and changes only catalogue voice/version identity, matching request version ID and execution voice identity. The deterministic immutable version ID binds principal, target and the entire trusted template, including native deployment/configuration/boot/runtime digest, model/bundle, profile and settings. Core VoiceStore supplies the existing transaction, CAS, principal ownership, idempotency and immutable-version rules. No catalogue is seeded by the new journey.

Generation reloads the owned version through the existing Dispatcher/Production path and requires exact equality with the trusted derivation before admission/freeze/submission. Existing exact registered versions continue through their original path. Wrong selectors, ambiguous selectors, foreign principals and modified versions refuse. No Kokoro relabeling, model/voice fallback, caller-supplied trusted configuration, self-enrollment or profile promotion.

The native supervisor separately reconstructs the owned context from its **original enrolled templates**, compares canonical template bytes including principal/context/profiles, then runs the existing admission/Authority/registration checks. Derived allocation registrations never become new preset templates. Existing finite registration limits remain. Its existing exact-template allocation route is retained. The shared helper lives in the already packaged native protocol module; archive membership is unchanged.

Native `/skeleton` now includes configured_presets; capability operations include registration only when explicit selectors exist. Speech readiness remains false with UNAVAILABLE / UNQUALIFIED status. CLI and MCP use their existing HTTP operation/schema, with no privileged second write route.

## Execution receipt

Read tests/conftest.py and tests/service/conftest.py first. Direct unittest only, no pytest/autouse maintenance guard. Every invocation used `/private/tmp/pab-0c-venv/bin/python -I -B -` with explicit absolute worktree sys.path (and tests/service for the final selection), asserted `sys.version_info[:3] == (3,14,7)`. Printed full runtime: `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`. Not Python 3.11.

Development journey: 1 PASS / 1.490s / exit 0, session 33243. After preserving original supervisor templates, the initial 92-case selection stopped after 2 cases (1 pass, 1 error, 9.447s, session 18594): the fixture updated working registrations after constructor but not original templates, producing generation timeout and close timeout. Corrected fixture enrollment; next affected/unrun selection stopped at adversarial test setup (journey passed, 1.630s, session 98511): an unsupported profile field failed schema validation before the intended assertion. Replaced it with valid config_revision. Next run stopped at malformed forged-version input before assertion (0.156s, session 26275); made request/descriptor/version coherent. Next run stopped at positive authorization (0.173s, session 64434), revealing the same post-constructor NativeFixture identity-binding issue in that standalone test. Fixture setUp now explicitly synchronizes the original enrolled template snapshot as well. These failed selections are not passing receipts. Product source was unchanged throughout those fixture corrections. No unchanged broad suite had run past these early failures.

Final corrected fixture selection: session 8885, initial chunk 35c699, exact 92 cases, TextTestRunner verbosity=2/failfast=True. **92 PASS / 55.915s / exit 0**, terminal chunk 3e06d6. No product/test edits after this corrected selection. Expected injected close_unknown shielded-future diagnostics and argparse refusal usage appeared; their assertions passed.

Selection (all methods of named classes/modules unless a method is specified):

- tests.release.test_native_onboarding.NativeOnboardingTests (3).
- tests.service.test_installed_preset.InstalledPresetTests (3).
- tests.release.test_native_history.NativeHistoryTests (44).
- tests.release.test_native_activation.NativeActivationTests (8).
- tests.release.test_native_launch.NativeLaunchTests (8).
- tests.release.test_native_enrollment.NativeEnrollmentTests (10).
- tests.worker.test_native_protocol; tests.clients.test_production_clients; tests.service.test_production_api.
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- tests.release.test_native_package.NativePackageTests.test_archive_deterministic_bytes; test_extra_archive_member_refuses.

New public journey starts with an empty catalogue under actual native installed lifespan/activation using fake TLS and backend/process handles. CLI registers; HTTP/MCP replay the same registration. It generates through canonical HTTP, waits for existing publication, reads job/artifact/asset/content and verifies SHA256. It closes/reopens the same Core/transport root under the same boot/trust, replays registration and the same generation job, and verifies one backend launch total. Missing/wrong/read-only credentials, wrong preset, registration key changes, CAS mismatch and immutable-version conflict refuse. Foreign principal and altered identity/version/speaker resolution refuse. Separate supervisor probes reject principal, native boot, preset, profile revision and coherently forged version substitutions; ambiguity and clone-as-preset refuse. Linux Kokoro public bootstrap/restart and existing closed HTTP/MCP schema tests are included.

`git diff --check` passed. Contracts/fixtures and lib/tts.py are unchanged against base. Current legacy lib/tts.py SHA256 remains `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; known legacy digest RED is unchanged/unwaived, not rerun or regenerated.

## Remaining limits and parent-owned gates

Only tiny in-process ASGI, fake TLS/process/backend, SQLite/filesystem and archive fixtures. Observed available disk 2,926,440 KiB, below 6 GiB; no heavy ACTUAL operation, network/listener, Docker, install/download, actual child, ML, crash, HELD/STT, operational cleanup or subdelegation. Fixture-owned temporary teardown only. Source tests and inert profile declarations establish no real installed/hardware/acoustic authority.

Finite clean retained history remains unchanged: zero jobs or fewer than 64 succeeded single-attempt non-clone jobs with existing exact schema/rows/Authority/receipt/import/publication/scalar bounds and the same configured boot/trust. No boot learning, renewal, history waiver or automatic cleanup. Real Docker topology/mount/firewall/reachability, pinned TLS/denied peer qualification, compatible arm64 interpreter/wheelhouse/ABI, actual model/MPS generation/cancel/restart, process/resource behavior, audio/performance and release acceptance remain unrun. CPU and CUDA qualification remain mandatory separate rows.

Clean managed clone onboarding remains separate: it needs a narrow canonical operation joining owned verified reference media/text to an immutable clone version, independent native reference authorization and lifetime handling, and a separately reviewed completed clone history policy. A preset cannot confer those permissions. No UI/STT or unbounded clone expansion was added. Parent owns fresh independent Sol review and adoption.
