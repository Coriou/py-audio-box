# r329 Sol — independent adversarial production-selection SOURCE review

2026-10-25. **NO-GO for accepting r328 as completing Lane 2: one P2 truthful-discovery defect remains.** The selected-context execution wiring is substantively present and the exact final 34-test selection passes in one independent invocation. This is bounded SOURCE/inert evidence, not source adoption, installation acceptance, hardware qualification or release approval.

## Authority and verified pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-4a617bfa`, branch `t3/production-hardware-selection`, HEAD `e6690c83885239310b6a58e731bdcd53656e0d3d`. Base `69ef5c69f13e594f5d21d30bc29bde9f8a59b2a4` is the verified merge-base. Candidate `git status --porcelain=v1` was empty before and after execution. `git diff --check BASE HEAD` passed. No applicable AGENTS.md was found in candidate/root discovery or checked ancestors.

Read FULL r328, FULL r323 (including Lane 2), PRODUCT.md, the entire actual diff, selector, production selection, execution protocol, owner duplicate/admission path, Production, canonical validation/profile snapshots, resolver consumers, public API and MCP schema/HTTP path, generator and affected test/fixture consumers. Existing root STT untracked files were observed only through status and left untouched. The sole lasting write is this report. No candidate edits, commit, merge, push, delegation, network/listener, Docker, install/download, actual child launch, ML, crash, HELD access, STT work or operational cleanup occurred. Tiny test-fixture lifecycle teardown is part of the authorized inert tests, not operational cleanup. No heavy ACTUAL work or 6 GiB waiver.

Verified SHA256:

```
e87ce186706edabde559465f4216eb393d7577ac667d06cadf493fa1f6ef0755  service/production_selection.py
cd3010036de67d7168d30e035f7d8733541bbd0cf9eac16863983eec978d72f1  contracts/generate.py
2a12772220231d50d869e74a43e8db10bbf974594a99af16e0ad175214f83e20  contracts/v1/schema.json
```

## Finding — P2: unrelated measurement makes strict discovery advertise ready

`service/production_selection.py:233–235` accepts any measurement with the configured workload, without requiring its key to name a configured candidate or relating it to the profile being advertised. Both `/v1/capabilities` and `/skeleton/profiles/{profile_id}` use this result. The execution selector correctly looks up measurements by actual candidate name (`choose`, lines 163–169); therefore discovery and execution disagree before any request-specific voice/media question arises.

Independent inert reproduction using the existing `ProductionSelectionTests` fixture:

1. Instantiate the fixture and call `setUp()` with its real Dispatcher/Core/admission, two configured Qwen CUDA candidates, fresh inventory and strict `SelectionPolicy(workload='inert')`.
2. Replace only `t.composition.measurements` with `{'not-a-candidate': t.measurements['gpu0']}`. Neither configured candidate now has measurement evidence.
3. Authenticated `GET /skeleton/profiles/qwen-cuda` returns HTTP 200 and `readiness='ready'` (next action remains the fixture's generic inventory/qualification text).
4. Submit the unchanged valid payload to `POST /v1/operations/speech_generate` with key `review-unbound-measurement`. Result: HTTP 503, `code='not_ready'`, action `Operator must refresh/qualify the exact installed profile or correct the override: measurement_required`.
5. Owner jobs remain `[]`; no admission occurred. Fixture teardown completed, process exit 0.

This is a metadata readiness defect, not an execution-authority bypass. It matters because truthful, actionable discovery is an explicit Lane 2 exit and r328 claims missing-measurement discovery. An orphaned measurement must not turn every profile ready. Bind discovery's evidence check to configured candidate/profile metadata and the applicable policy, without resolving media or loading ML. If metadata cannot establish current bound evidence, report a conservative actionable not-ready/verification requirement. Add a regression for an unrelated measurement key (and profile-scoped evidence), preserving metadata-only discovery. The current test checks missing measurements only on submission, so all 34 passing tests do not cover this case.

## Bounded positive assessment

Dispatcher retains principal/operation/key fingerprint replay before resolution, readiness checks or ranking. New requests resolve on the existing owner thread, select before Production/Admission, and pass the chosen complete context onward. The selected descriptor/device index, bundles/references and profile snapshot reach the existing frozen plan and Authority; selection is not discarded. Core still owns persistence, admission, reservations, publication/import, cancellation and retained reference/reap behavior. The diff introduces no second scheduler or runtime router.

Owner current profile snapshot/configuration must match; installation, measured qualification and readiness remain required. Typed admission independently rejects synthetic qualification evidence. Measurement ranking binds the canonical hash of the whole `context['execution']` mapping and the policy workload. Changed execution invalidates its receipt; explicit unmeasured ordering remains a separate policy. RAM uses executable demand, CUDA VRAM uses existing profile/admission budgets, and configured concurrency is compared with the candidate profile's capacity. Existing reservation/queue limits still apply; these checks do not qualify increased throughput.

Composition constructor bounds 1..16 unique candidates and shares the identical runtime object. Installed activation refuses multiple resolvers. Private Linux Qwen multi-CUDA-index candidates retain exact model/bundle, voice/settings and runtime semantics; this is not a new installed Qwen package. Native binding/topology and hardware-wire identity are checked; native/Linux and cross-host routing are not introduced. Singleton resolver verification and the independent runtime authorization boundary remain in place.

Optional profile/device overrides are canonical prepare/generate fields, with a closed device object, family cpu/cuda/mps and integer index 0..255. Existing explicit identity remains the engine/model/device/precision authority; no alternative model field or silent voice/model fallback appears. Resolver/private execution validation enforces overrides, and CLI/MCP reuse canonical HTTP/schema validation. Discovery reads owner metadata without resolving owned media or importing ML; the finding above limits its readiness claim. The live CPU record is policy availability metadata for the existing Linux worker, with unspecified architecture, not persisted accelerator or performance qualification.

Pocket preparation/save/reuse and native preset/managed-clone public paths remain represented by passing regression journeys. No claim of exhaustive security or lifecycle clearance is made.

## Independent executable receipt

Inspected both pytest conftests before execution. The service conftest has a listener/subprocess fixture; it was bypassed by direct unittest. Inspected the selected test doubles: ASGI requests, fake TLS/connector/backend/child function and tiny temporary sealed files; no real service listener or child process was requested.

Exactly one coherent final selection invocation, from the candidate cwd:

```
/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/pab-r328-finite-tests.py
```

Read the runner before use: it inserts only the candidate root and explicit service/client test paths, prints interpreter/base/selection, and runs these names:

```
tests.service.test_production_selection
tests.service.test_profiles
tests.service.test_production_api.ProductionAPITests
tests.clients.test_production_clients.ProductionClientTests
tests.service.test_pocket_preparation.PocketPublicJourneyTests.test_http_cli_mcp_dispatcher_prepare_save_generate_and_reopen
tests.service.test_installed_pocket.InstalledPocketJourney.test_http_cli_mcp_dispatcher_prepare_save_generate_and_reopen
tests.service.test_installed_pocket.DiscoveryTests
tests.release.test_native_onboarding.NativeOnboardingTests.test_fresh_public_registration_generation_artifact_restart_replay
tests.release.test_native_managed_clone.ManagedCloneTests.test_public_upload_prepare_results_save_generate_reopen_replay
```

Observed interpreter **CPython 3.14.7**, Clang 17.0.0; not Python 3.11. Result **Ran 34 tests in 26.880s, OK, exit 0**. This is the fresh all-green combined receipt; r328 itself had 32 PASS plus two fixture failures followed by two corrected PASS, not an earlier all-green combined run. No retries or broad additional selections were run. Tool execution session 15024 contains the full named-test receipt; no extra receipt file was written.

The selection includes `SelectionContractTests.test_generator_filename_and_byte_determinism_and_existing_fixtures`: two generations compare exact filename-to-byte maps and corresponding committed fixtures. Generator change omits only the optional override fields from defaults; no fixture changes are in the diff. `test_all_generator_request_and_exchange_consumers` validates the schema, every operation-request fixture (with existing binary oracle decoding), and production transcript request/response consumers. Client tests cover canonical closed MCP schemas and HTTP forwarding; the new integration exercises overrides through actual CLI/MCP→ASGI→owner selection. These are source/inert checks, not public deployment acceptance.

One additional justified focused probe, described in the finding, used the same interpreter/flags and explicit paths through `python -c`; it exercised the uncovered measurement/discovery mismatch once. No production code or test files were changed.

## Parent-owned disposition and remaining exits

Resolve the P2 discovery objection, add focused regression evidence, and obtain review of the resulting exact source under parent authority. This report does not adopt or integrate the candidate. Parent then decides whether the bounded selection slice satisfies its commission and requests Astra's integrated milestone judgment.

The installed boundary remains one configured Linux Kokoro/Pocket CPU resolver or the separate enrolled native Qwen MPS registry. Private multi-index selection is meaningful execution wiring but does not fulfill full installed hardware auto-selection. Model auto/device auto, automatic voice migration, automatic acquisition, multi-engine installed deployment and cross-host/native-Linux composition remain unsupported. These are explicit unmet broader PRODUCT/Astra exits, not a demand to add an installer/topology rewrite as a defect in this bounded patch.

Performance receipts remain trusted operator inputs, not independently acquired measurements. Existing native history/lifetime/renewal/topology limits, FULL API/CLI/MCP gaps and actual CPU/CUDA/Metal installation, speech/clone, cancellation/reap/restart/fault, listening and latency/memory/throughput qualification remain parent-owned gates. **Metal remains UNQUALIFIED; no authorized arm64 wheelhouse. Release remains NO-GO.**

Legacy digest RED is unchanged and unwaived: current `lib/tts.py` hash independently remains `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`, versus r323's recorded legacy-control digest `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`. Neither loader nor legacy control is changed by this candidate; reconciliation remains separate authority.
