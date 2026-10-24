# Installed Kokoro preset bootstrap — r271 Sol

Base: `b189463f137aa9f240a7bd18d8fe1f433af43d31`; its parent is requested source `16f16c6a3834de054e7d870f620cd29b8c55c502`, and its only delta is the r270 report. Clean at entry.

## Pre-regeneration fixture allowlist

Contract inspection before regeneration: add `voice_preset_register` with closed selector/target/revision request, reuse `voice_register_result`. `contracts.generate` derives operation requests from operations/schema; no existing transcript consumes the new request. The exact generated fixture allowlist is therefore:

- `contracts/fixtures/v1/operation-requests.json`

All other generated fixture filenames and bytes must remain identical.

## Implementation candidate

Source commit: `2a1f4fd304e5d82fc56396915e1791bad4f0cedd`. R270-1's source integration gap is addressed; adoption remains parent-owned and awaits the requested independent Sol review. No additional Astra gate is proposed. No release or actual installed-runtime GO follows.

Read full r270 normative NEXT, PRODUCT and r188. No applicable repository/ancestor AGENTS.md was found. Scope remained the named service, clients, contracts, tests and install documentation. The full install document is `docs/agent-interface/HEADLESS-INSTALL.md`, not a repository-root file.

`voice_preset_register` is a named authenticated HTTP mutation, CLI `voice preset-register`, MCP `speech_voice_preset_register`. Its closed request contains only `preset`, `voice_id`, `expected_revision`; the key is the existing Idempotency-Key header/MCP argument. It accepts no descriptor, filesystem path, principal, model metadata, readiness or qualification. The owner requires checked installed activation and the existing readiness predicate, resolves the principal's immutable verified Kokoro context and passes its version through existing `VoiceStore.register` and its transaction/CAS/replay logic. No DDL, initializer seeding, scheduler, recovery or frozen-job implementation changed.

Version identity deterministically binds principal, target, trusted version snapshot and exact execution identity, including verified manifest digest. The target is principal-local; this does not add arbitrary multivoice/model resolution. Identical key/request replays return the original result; changed target/revision with the same key conflicts; a different key cannot overwrite the same deterministic immutable version. Unsupported selectors and foreign/unconfigured principals refuse. Generation reloads the owned version, compares it with the configured deterministic version, then uses the existing Production/admission/freeze/submission path. An arbitrary imported identity cannot inherit authority just by matching model fields. Existing duplicate job lookup still returns the original frozen submission before resolving a new context.

Installed discovery advertises preset registration/generation, omits clone prepare/results/save from configured operations, and marks their route details unavailable for this profile. `/skeleton` exposes principal-scoped configured selectors, exact model/bundle identity, generation-only support and the next action. Profile qualification fields are preserved; registration does not promote them. Canonical operation/schema metadata is included unchanged by the existing archive manifest. CLI/MCP are standalone client source as at base; this lot does not change archive membership or claim a new client installation package.

## Validation and exact counts

All execution used `/private/tmp/pab-0c-venv/bin/python -I -B` with an explicit repository `sys.path` insertion and explicit unittest names. Read both conftests first. No pytest collection or maintenance guard was executed. The installed fixture uses disposable small files/SQLite and injected identity, privilege, host, hello, supervisor, backend output and reap observations. These are SOURCE/in-process adapter checks only.

- Broad explicit unittest run: **102 tests PASS**, 0 failures/errors, 83.280 seconds. Modules: installed preset (3), installed activation (26), installed restart (32), engine admission (5), production API (then 4), production clients (then 1), headless install (31).
- After the final installed-only registration fence and two additional assertions: **31 tests PASS**, 0 failures/errors, 18.880 seconds: installed preset (3), installed activation (26), new production API installed-activation/closed-request test (1), new MCP named closed schema/key test (1). This covers the changed source branch on the final source candidate; unchanged broader regressions were not needlessly repeated. The union is 104 distinct unittest methods, not 133 distinct methods.
- Final source-commit archive closure run: **5 tests PASS**, 0 failures/errors. `tests.release.test_release_package.ArchiveTests` checks deterministic archives, exact manifest/file bytes, extracted help/imports and existing isolated Pocket closure; it runs only inert Python help/import subprocesses, no worker workload or listener. Earlier pre-final archive run also passed the same five checks.
- Four existing contract checks called directly, without pytest: `test_schema_and_all_fixtures`, `test_determinism`, `test_operation_requests_jobs_artifacts_and_error_mapping`, `test_binary_fixture_boundary`: **4 PASS**.
- `test_exact_text_ids_defaults_and_source_control_coverage`: **FAIL**, an independently verified pre-existing base mismatch for `lib/tts.py`. Recorded metadata digest `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`; actual base and current digest `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Verified both bytes and metadata using `git show b189463f137aa9f240a7bd18d8fe1f433af43d31:...`. Neither legacy source nor its unrelated digest was changed. This check is not represented as passing.
- First regeneration changed exactly `contracts/fixtures/v1/operation-requests.json`. Second and final regeneration preserved every fixture filename and byte. Git's fixture delta contains exactly that allowlisted file. `git diff --check` passed.

The fresh-state regression starts with an empty catalogue and reproduces `not_found` for generation before bootstrap. It then exercises authenticated CLI registration, MCP/HTTP identical replay, public list/version/versions, generation, public job/artifact/asset/content and SHA256 verification. It registers another explicit catalogue target while checking the previous frozen authority remains unchanged. It closes the actual Dispatcher and reopens the exact same completed fixture root, retaining SQLite catalogue/job/dispatch/reap/ack history; public reads return the old version and job, and a new generation uses that retained version. This reuses the injected installed gate/hello and is not a container/host restart qualification. No `Core.voices.register` seeds this journey or the updated activation fixture.

Negatives cover absent/read-only credentials, foreign principal/version access, wrong preset, forbidden model/path/readiness fields, changed-key payload conflicts, fresh-key immutable version conflict, unavailable activation with empty catalogue retained, and mismatched model/preset/arbitrary imported version identity. Existing activation/cancellation/restart/admission/API/client regressions remain intact. The deliberate close-failure test emits its existing `injected_close_unknown` shielded-future diagnostic while passing its expected assertions; it is not an unexpected runtime failure.

Exploratory checks initially failed because the new test imported a TestCase into unittest discovery, the test bridge passed a keyword-only argument positionally, a setup variable shadowed the ownership callback, and the new commands were initially absent from Dispatcher's public command allowlist. Corrected those defects before the passing runs above. The first mixed discovery run reported 25 tests, 23 failures/2 errors; intermediate focused runs also exposed these wiring defects. These are not counted as passes or concealed as qualification. Tool-session outputs retain the execution history.

## Limits and parent handoff

No actual ML, model load, hardware probe, Docker, network, download, dependency installation, listener, process-crash test, historical cleanup, HELD operation or delegation occurred. Only small disposable fixture/archive files were used; the known roughly 3.4 GiB headroom remains below the 6 GiB actual-work gate, which was not lowered. No UI, STT, native Metal or simultaneous backend installation was added.

Parent owns independent Sol review before adoption and normal ledger bookkeeping. Separately unrun: compatible authorized Linux CPU host/headroom; exact release runtime/images and reviewed model/preset provenance; fresh package install and explicit managed provisioning/configuration; real offline register/generate/download; audible intelligibility and latency/memory; actual cancellation/reap, same-volume restart and fault behavior. Pocket cloning/dynamic resolution, CUDA installation/qualification and native Apple Silicon Metal remain later bounded work. The unrelated base legacy digest failure remains visible for parent disposition.
