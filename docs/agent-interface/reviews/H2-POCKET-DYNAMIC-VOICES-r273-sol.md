# Dynamic Pocket voices — r273 Sol

Base verified before edits: `fd82e2e9c750b51d15b5bd03e0e2ad2cd0335fd6`; clean worktree.

## Pre-generation fixture allowlist

Before invoking the generator, inspected schema, generator and transcript consumers. Adding optional bounded `audition_text` changes generated example requests and the two preparation transcripts which inherit that request (including their derived plan hashes). Exact allowed generated paths:

- `contracts/fixtures/v1/operation-requests.json`
- `contracts/fixtures/v1/transcript-voice_prepare.json`
- `contracts/fixtures/v1/transcript-voice_prepare-roles.json`

No other fixture filenames/bytes are authorized to change. Optional absence retains historical synthetic fixture compatibility; real Pocket requires explicit nonblank request text.

## Candidate and ownership

Implementation source commit: `40f7294549b693c1d10f08acab1ab82d355b98f7`, direct child of the requested base. Submitted for parent fresh independent Sol review; no self-adoption, installed GO or release GO.

Read full r270 (including Lot 2), r271 and r272 and current execution contexts, Production, Dispatcher, VoiceStore, Pocket prepare/engine, complete runtime protocol/supervisor mapping and affected API/client/generator/archive consumers. No applicable AGENTS.md in this checkout or checked ancestors. Bounded memory registry lookup supplied no relevant premise. No delegation. Both conftests were inspected before execution; direct unittest bypassed pytest collection, maintenance guard and listener fixture.

`PocketContexts` now supplies one reusable trusted bundle/profile/budget factory per principal. Its optional reference-free construction verifies the configured bundle with the existing full file/hash verifier, extracted without weakening the previous `verify_inputs` path. The private protocol-shaped template remains configuration, but its original operation, reference identity, segment and voice version are replaced from owner-resolved data on every new command. Explicit old reference mappings remain supported by the lower-level constructor/resolve seams.

Dispatcher calls `resolve_owned` synchronously on Core's sole owning thread after duplicate lookup and existing readiness fences. Preparation checks the selected principal-owned upload; generation reloads the requested principal-owned immutable version. Core metadata supplies exact asset identity/hash/rate/frames, and managed bytes supply bounded file size. No public path, descriptor, model metadata, execution context, arbitrary mount mapping or readiness claim is accepted. The complete unchanged Pocket protocol decoder then checks identity, immutable version configuration/settings/hash, profile, segment, mappings and budgets, and full bundle/reference verification precedes admission. Production independently reloads/checks managed data. No DDL, second registry/queue, recovery redesign or installed activation was added.

`audition_text` is an optional bounded field in the closed public preparation schema (1–10000 characters); real Pocket additionally requires nonblank exact request text and its configured byte/profile caps. Frozen role text and plan hash bind that exact string. Schema absence remains compatible with historical synthetic requests; no historical fixture fallback grants real Pocket authority. CLI accepts named `--audition-text` and refuses conflicting JSON/flag values. MCP inherits the named bounded field from the canonical closed schema and keeps its idempotency argument. Generated roles use request text when supplied. Existing reference PCM versus generated audition separation, independent measurements and pair publication remain intact.

Supervisor supports a trusted principal-to-managed-reference-root configuration. It constructs only `root / exact protocol-validated asset ID`, checks the owner/root without following symlinks, independently verifies complete bundle/reference bytes/hash/PCM, and checks exact canonical execution mapping before journaling. It repeats verification before launch. With no root registration, the old exact reference allowlist remains required. This reads no Core DB. The source fixture supplies a trusted managed assets root; an actual read-only mount, installed path policy and installed configuration route are explicitly NOT implemented or qualified here. Roots/mounts must be trusted, stable and provided by integration, never a public request. This change supplies no hostile same-UID concurrent filesystem guarantee.

Pocket discovery now names its configured engine/profile and prepare/results/save/generate support, retains jobs/cancel/artifact reads, excludes Kokoro preset registration, and exposes a concrete next action. Profile qualification comes from existing owner configuration, not a probe or registration side effect. The noninstalled Pocket status explicitly says installed=false and does not promote ready-for-speech. Installed Kokoro discovery/registration fences remain in place. The archive now includes the four client source files, with isolated extracted imports and named schema checks; this is source dependency closure using the test environment's dependencies, not an installed dependency/runtime qualification.

## Tests and exact evidence

Interpreter: `/private/tmp/pab-0c-venv/bin/python -I -B`, CPython **3.14.7**. Explicit absolute sys.path entries: this checkout and, for legacy bare test imports, its `tests/service` directory. No pytest invocation or collection.

Final broad explicit selection (38.261 s): **124 cases executed, 123 PASS, one test-fixture AttributeError**. The remaining fixture typo was corrected and the exact `tests.worker.test_engine_lifecycle.DispatcherEngineTests` rerun **1 PASS, 1.226 s, exit 0**. Thus all **124 distinct selected cases have passing final results**, across these commands; this is not a claim of one all-green broad invocation. Production source did not change after the broad run. Exact broad selection:

```
tests.service.test_pocket_preparation
tests.service.test_pocket_lifecycle
tests.worker.test_engine_pocket
tests.worker.test_runtime_conditioning
tests.service.test_engine_admission
tests.service.test_production
tests.service.test_production_admission
tests.service.test_production_reconciliation
tests.service.test_production_api
tests.clients.test_production_clients
tests.service.test_installed_preset
tests.service.test_qwen_lifecycle
tests.worker.test_engine_kokoro
tests.worker.test_engine_lifecycle
tests.release.test_release_package.ArchiveTests
```

The final Pocket public case starts with an empty catalogue on its separate public root and a reference-free factory configured before upload. It performs two authenticated HTTP uploads with distinct selected PCM bytes, two exact client audition strings, CLI prepare plus MCP/HTTP replay, public results/audition metadata, save/replay, alternating saved-version generation, artifact retrieval, actual Dispatcher close/reopen on the SAME completed root with the SAME factory, both retained-version generations and old prepare replay. No Core version seeding, context swap or post-upload/save reconfiguration occurs on that root. Its inherited lower-level preparation support uses a separate seeded fixture root; that is not the public root or evidence of public bootstrap. Every public runtime allocation also invokes the actual independent supervisor verifier against the managed root, with fixture owner UID injection; actual transport/backend/death observations remain injected.

Negatives exercise missing/read-only/foreign credentials, foreign upload/version, missing version/reference, altered bytes, caller mapping/path fields, blank/oversized audition text, profile incompatibility/qualification, replay payload conflicts, immutable save conflicts and stale CAS using a different prepared candidate against an existing voice. Frozen role/request mismatch is refused even before hashing; coherently changed request+role produces a distinct authority binding. Existing pair atomicity/tamper, failed audition, cancellation reservation retention until reap, literal-empty helper, completed restart and opaque nonempty UNKNOWN regressions remain exercised. Existing Qwen generation and configured Kokoro preset bootstrap pass.

The older Kokoro Dispatcher lifecycle fixture at base still seeded its legacy version ID, rejected by r271's configured-registration resolver. Only that test fixture was aligned to `KokoroContexts.preset_version`; production Kokoro behavior was not changed. The first alignment used a nonexistent fixture `self.spec`, causing the final broad run's one error; its corrected descriptor lookup passes the exact rerun. Other exploratory failures were test edits (CLI `save0` typo; assertions expecting no historical reservations, raw OSError instead of `missing_media`, a `voices` rather than `items` response key, and stale mismatch ordering after stricter frozen-role validation), two missing bare-test import paths, and an archive stdout expectation after expanding closure. Earlier intermediate suites were red and are not represented as passes. Final broad emitted existing SQLite ResourceWarnings and the expected injected `lifecycle supervisor unavailable; lane quarantined` diagnostic; neither was hidden or changed.

Direct existing contract functions: `test_schema_and_all_fixtures`, `test_operation_requests_jobs_artifacts_and_error_mapping`, `test_binary_fixture_boundary`: **3 PASS**. In-process complete generator comparison across all fixture filenames/bytes: **PASS**, repeated after implementation; exact git fixture delta equals the pre-recorded three-file allowlist. No unrelated fixture bytes changed. `git diff --check`: PASS.

**Known red check retained:** `test_exact_text_ids_defaults_and_source_control_coverage` FAILS. Enumerating all base legacy command/shared-source/launcher digests finds the sole mismatch `lib/tts.py`: recorded `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`, actual base/current `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Neither source nor unrelated metadata was changed. No all-green contract-pack claim.

## Limits and parent gates

Only small inert fixture/SQLite/archive files and isolated help/import subprocesses were used. Entry disk observation: 3,503,376 KiB available, about 3.34 GiB, below the unchanged 6 GiB actual-work gate. No actual ML/hardware, Docker, network/download, dependency installation, listener, process crash, operational cleanup, HELD operation or delegation.

Parent owns fresh independent Sol source review and adoption. Installed Pocket policy/provisioning/runtime dependencies/configuration/read-only mount enforcement remain separate work, as requested. Actual bundle/runtime/audio, cross-UID/kernel enforcement, measured budgets/latency, installed cancellation/reap and restart qualification remain UNRUN. Nonempty interrupted conditioning remains UNKNOWN and retained; this does not claim arbitrary cancellation-point recovery. CUDA and Apple Metal release rows remain open. The legacy digest failure still needs separate parent disposition.
