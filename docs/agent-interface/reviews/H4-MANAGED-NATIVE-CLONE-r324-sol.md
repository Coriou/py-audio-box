# Managed native clone lane — r324 Sol bounded prerequisite

2026-10-25. **Lane 1 INCOMPLETE.** This candidate implements the clean completed-clone history prerequisite and the r323 Pocket discovery correction. It does not implement managed native clone preparation/registration/audition/save onboarding. No adoption, release GO or actual hardware qualification is claimed. Fresh independent Sol review remains parent-owned.

## Pins and scope

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-cfd9ac5f`, branch `t3/managed-native-clone`. Clean entry/base `e4f710bb55e34e1623d13637671c8302713f135d`. SOURCE/tests commit `7b120c28b42003b55c698d3e00f5144448f00783`; this report follows separately. No root writes, merge, push, subdelegation or self-adoption.

Read full authoritative HEADLESS-MILESTONE-r323-astra.md first, PRODUCT, NATIVE-PACKAGE, r322, and relevant r277/r319/r321 advice. Inspected actual Pocket owned-resolution/preparation/save, contracts, native contexts, Dispatcher/Production, upload/VoiceStore, native runtime/supervisor/child, activation/history and public/client fixtures. Repository discovery and checked worktree ancestors found no applicable AGENTS.md. Memory registry search returned no relevant hit; no memory-derived technical conclusion.

The requested whole lane spans a new typed preparation/result boundary, native independently authorized dynamic reference derivation, immutable save/reuse resolution and clean history. This finite slice implements the latter prerequisite without exposing a partially authorized public clone route. Existing preset derivation remains restricted to original custom-voice templates. No new typed contract, public operation, scheduler, store, model download, topology/selector, VoiceDesign, UI or STT was added.

## Implemented behavior

`completed_history` now admits an otherwise valid succeeded single-attempt exact enrolled clone. All existing finite schema/row/command/Authority/native boot-trust/reservation/receipt/publication/import/inventory checks still apply. It additionally reuses VoiceStore's read-only ownership check, compares current Core reference metadata canonically with the frozen asset, verifies bounded exact bytes/digest/mono PCM16 and requires the retained control input to equal those bytes. Only clones permit the fixed `input` spool member. Missing, foreign or altered references refuse before recovery or scheduling; no repair, ACK or durable write occurs on refusal.

This check runs before Core attaches its `voices` facade or initializes schema. A lightweight existing `VoiceStore(core)` performs the canonical read-only ownership check without constructing catalogue DDL. There is no new authority inferred from a derived allocation or from retained history. Future execution continues to require the existing original native enrollment/Authority path. Failed, cancelled, unknown, multi-attempt, changed-boot/trust and uncertain import histories remain refused. Caps remain unchanged; reference files are retained, with no reclamation/renewal policy added.

Inert clone fixtures explicitly enroll a base-model clone template and upload tiny PCM through `ingest_local` before seeding an immutable catalogue version. This is **seeded prerequisite evidence**, not the requested empty-catalogue managed journey. Resource protocol initialization precedes fixture upload, matching Dispatcher ordering. Tests perform same-root/same-boot replay twice with one fake backend launch, verify exact transcript and reference retention, refuse owned/detached tamper, missing reference and foreign ownership, and demonstrate reference bytes remain in Core/control/native storage during cancellation before confirmed reap and afterward. Existing native child-function/backend and independently verified byte-transfer consumers run; no actual child or ML runs.

`api.voice_detail` now bases prepare/results/save availability on the configured operation capability rather than installation alone. Installed Pocket discovery/detail agreement is asserted for all three operations; `voice_results` correctly reports `implemented_read`. Unconfigured preset registration remains unavailable and unqualified speech remains unqualified.

## Direct unittest receipts and corrections

Read tests/conftest.py and tests/service/conftest.py first. No pytest/autouse maintenance or listener fixture execution. Every observation used `/private/tmp/pab-0c-venv/bin/python -I -B -`, with explicit absolute worktree and worktree/tests/service sys.path. Asserted Python **3.14.7**; printed runtime `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`. TextTestRunner verbosity=2/failfast=True; failed results map to exit 1.

Receipts are separate; **there was no clean combined 84-test PASS**:

1. Session `82047`, initial `da8738`, terminal `6ef924`: planned 84; **44 existing NativeHistoryTests passed**, then first new clone case errored `ValueError: not_ready`; 45 run / 44.504s / exit 1. Fixture had uploaded into a populated Core before resource protocol initialization, correctly leaving admission invalid. Corrected fixture ordering only. Completed 44-case history block was not repeated.
2. Tiny read-only fixture startup diagnostic `599f29`, exit 0: confirmed `resource_fault=True`, `persistence_fault=True`, healthy reconciliation/integrity. No job submitted. Fixture teardown only.
3. Session `79344`, initial `71ed8b`, terminal `a3fe8f`: affected/unrun 40 selected; first clone generated successfully but reopen errored `AttributeError: 'Core' object has no attribute 'voices'`; 1 run / 1.313s / exit 1. Corrected production pre-schema ownership access to existing read-only VoiceStore as described above.
4. Session `43823`, initial `7fd9ae`, terminal `62462f`: corrected affected/unrun 40 selected; **12 passed** (five new clone history cases, clone cancellation lifetime, all six NativeOnboardingTests), then discovery assertion failed because it expected `implemented` for GET `voice_results`; 13 run / 10.107s / exit 1. Corrected test expectation only. These 12 passing cases were not rerun.
5. Final remaining affected selection **28 PASS / 15.773s / exit 0**, session `94554`, initial `4cba6d`, terminal `8da78b`. No source/test edits afterward. Full DiscoveryTests, test_pocket_preparation (13), InstalledPresetTests (3), test_production_clients (3), test_production_api (5), and NativeSupervisorTests methods `test_input_complete_replay_conflict_gap_and_hash`, `test_clone_conditioning_nonempty_blocks_terminal`, `test_input_wrong_complete_hash_quarantines`.

The five selected NativeCloneHistoryTests methods were `test_clone_keeps_exact_reference_and_transcript_across_replay`, `test_changed_owned_reference_refuses`, `test_changed_detached_reference_refuses`, `test_missing_owned_reference_refuses`, `test_foreign_owned_reference_refuses`. Its inherited history methods were not executed with clone fixtures in this selection. Existing history regressions passed before the final pre-schema VoiceStore correction; that correction is clone-branch-only. No broad unchanged repeat or legacy qualification run.

`git diff --check` passed. Exact SOURCE hashes:

- service/api.py: `4186ad45a072adce88c361c249b3ba3726bda06a2a3ec79218e3f4bcd00f99b1`
- service/native_activation.py: `73a59c0e63b86c8e47c4d7713cc55a493d65ec20240e1162c4c95a794ec8e806`
- tests/release/test_native_history.py: `e4431c0c57abc584312e988fd85ef285adf796c66c8c4b373a06a51b3f47f014`
- tests/service/test_native_lifecycle.py: `9a906b1cf00f5439f218f9f5357dbfeeaa4a2c68325c7f3198ce9c386be85688`
- tests/service/test_installed_pocket.py: `ae39b0d5a032123b8548bc1de478f68768d00cc7ce4e556be5f793af3b567b2b`

## Remaining lane exit and parent gates

Still missing: one supported provisioned clone capability resolved on the writer thread from owned upload and exact user transcript or explicit supported x-vector mode; typed Qwen preparation with separate explicit audition/reference provenance; independently reconstructed native reference authorization exclusively from original templates; public HTTP plus thin CLI/MCP prepare/results/save integration; immutable saved-clone reuse without per-voice enrollment editing; empty-catalogue upload-to-artifact-to-clean-replay demonstration. Clone-specific transcript/model mismatch, CAS/idempotency and reference-versus-audition provenance tests for that new route remain required. Pocket coverage and preset tests do not close these gaps. Do not mark r323 lane 1 complete or proceed to global Astra adoption on this report alone.

Only authorized tiny in-process ASGI/fake TLS/filesystem/SQLite/audio/fake-process observations and fixture-owned temporary teardown occurred. Available disk observed 2,838,744 KiB, below 6 GiB. No heavy ACTUAL, network/listener, Docker, install/download, actual child/ML, crash, HELD, STT or operational cleanup. Test fixtures inject free-space/hardware declarations; they do not establish real resource or MPS qualification.

Metal remains **UNQUALIFIED**, without a reviewed arm64 wheelhouse. CPU/CUDA remain mandatory actual qualification rows. Topology/mount/firewall, real pinned TLS, installed ABI/model/hardware, acoustic/performance, fault/restart/renewal and release gates remain parent-owned and unrun. Legacy lib/tts.py digest RED remains unchanged/unwaived: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; no metadata regeneration or reconciliation.
