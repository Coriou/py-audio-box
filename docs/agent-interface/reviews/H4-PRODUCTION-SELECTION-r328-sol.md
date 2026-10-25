# r328 Sol — production selection implementation candidate

2026-10-25. Implemented in own worktree; **pending fresh independent Sol review, then Astra integrated milestone judgment**. This is source/inert evidence, not self-adoption, installation acceptance, hardware qualification or release approval.

## Authority and pins

Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-4a617bfa`, branch `t3/production-hardware-selection`. Clean starting HEAD `69ef5c69f13e594f5d21d30bc29bde9f8a59b2a4` (integrated managed clone journey). Read PRODUCT.md, full HEADLESS-MILESTONE-r323-astra.md (especially lane 2), selector/tests, Dispatcher/Production/admission, execution contexts/runtime protocol, Linux installation and native activation boundaries, and public HTTP/CLI/MCP adapters. Repository discovery found no AGENTS.md. Memory registry was consulted only for the existing Core ownership/frozen-plan fences; current source determined implementation.

No root checkout writes, merge, push, subdelegation, listener, network request, Docker, install/download, ML inference, actual child launch, crash experiment, HELD access, STT work or operational cleanup. Tests use tiny temporary sealed inert files and fabricated runtime/TLS/backend receipts. No heavy ACTUAL operation was attempted; the 6 GiB gate was not waived. Direct unittest bypasses the inspected pytest conftests (including their listener/process fixture).

## Executable selection

Dispatcher's existing duplicate lookup remains first. For a new prepare/generate request, owner-side resolution now feeds `production_selection.choose` and the existing pure selector **before** Production and Admission. The selected record indexes the complete context returned to Production; its descriptor, hardware index, bundles, references, immutable voice version and profile snapshot become the existing Core frozen execution plan. There is no discarded selection, parallel scheduler or runtime router.

The default existing resolver is a singleton candidate. Its sealed/provisioned verification stays in the existing trusted resolver and independent runtime boundary. Current owner profile authority must match the snapshot and be installed, configured, ready, qualified and measured, with a qualification receipt; typed execution still rejects synthetic qualification receipts. Selection does not promote installation or invent qualification. A qualified singleton without comparable latency evidence uses explicit unmeasured policy, not an invented speed ranking.

`ContextComposition` is a private operator constructor, not a public JSON/API contract. It admits at most 16 uniquely named resolver candidates bound to the **same runtime object** and injected inventory. Installed activation permits exactly one resolver, retaining the existing installed policies. The existing native registry can still contain its bounded enrolled versions/templates. Singleton composition forwards preset/capability methods. Multi-candidate preset registration is unavailable.

Within an explicitly configured existing Linux Qwen supervisor, compatible candidates can differ by CUDA device index while retaining the exact model/bundle, saved voice, dtype, settings and current profile configuration. The selected descriptor is dispatched through that same runtime. Independent existing supervisor registration/mount/authorization checks remain authoritative; composition cannot enroll a new runtime or acquire another model. Hardware-wire identity must match the resolver snapshot. Native contexts must match the runtime's native binding; Linux/native composition and cross-host routing refuse.

Measured ordering accepts only operator `Performance` evidence for the policy workload, wrapped in a receipt bound to the canonical hash of the entire execution mapping (including descriptor/hardware). Changed execution invalidates that measurement. The strict composition default requires measurement; explicit unmeasured policy remains available as policy ordering, never fastest-hardware evidence. Inventory records expire after the configured bounded age (default 300 seconds; at most 3600), including future/invalid timestamps. Local CPU availability is cheap live policy metadata for the existing Linux worker, with unspecified physical architecture; it is not an accelerator probe or a speed measurement.

Memory uses the selected executable RAM demand and the existing owner budget; CUDA VRAM is also bounded. Trusted composition concurrency demand is checked against each candidate's own profile capacity. There is no new CPU-sized global cap and no widening of the existing scheduler or reservation rules. Existing reservation/queue/disk/cancellation/reap fences still decide admission/execution.

## Public contract and discovery

Canonical `voice_prepare_request` and `speech_generate_request` add optional `profile_override` and a closed `device_override={family: cpu|cuda|mps,index: 0..255}`. Existing `identity` remains the explicit engine/model/bundle/device/precision contract; no second model selector is invented. Mismatched overrides refuse, including at private runtime decoding. No silent model, voice or CPU fallback is introduced. HTTP remains canonical; CLI request JSON and MCP schemas inherit these fields without alternate execution paths.

Authenticated capabilities and `/skeleton/profiles/{profile_id}` read owner metadata, never resolve candidate media or load/download ML. They reflect qualification, installation, inventory freshness, memory, configured concurrency, missing measurement and owner/runtime readiness. Failed selection returns compact actionable reasons to refresh/qualify the exact profile or correct the override. Discovery cannot promise that a particular owned voice/reference resolves; that remains request-time verification. Existing operation capability policy remains in place, including the reviewed installed Pocket detail agreement.

The fixture generator explicitly leaves optional selection overrides absent from default examples. No generated fixtures were rewritten. Two isolated generations verified identical filename sets and bytes, and equality with all corresponding committed fixtures. Indirect operation-request/transcript consumers and MCP closed-schema validation were included.

## Finite test receipt

Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B`, **CPython 3.14.7**, not 3.11. Explicit repository/service/client paths were inserted by the temporary unittest runner; no pytest invocation or conftest execution.

Initial command:

```
/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/pab-r328-finite-tests.py
```

The runner loaded these explicit names once:

- `tests.service.test_production_selection` (8 tests)
- `tests.service.test_profiles` (13 tests)
- `tests.service.test_production_api.ProductionAPITests` (5 tests)
- `tests.clients.test_production_clients.ProductionClientTests` (3 tests)
- `tests.service.test_pocket_preparation.PocketPublicJourneyTests.test_http_cli_mcp_dispatcher_prepare_save_generate_and_reopen`
- `tests.service.test_installed_pocket.InstalledPocketJourney.test_http_cli_mcp_dispatcher_prepare_save_generate_and_reopen`
- `tests.service.test_installed_pocket.DiscoveryTests`
- `tests.release.test_native_onboarding.NativeOnboardingTests.test_fresh_public_registration_generation_artifact_restart_replay`
- `tests.release.test_native_managed_clone.ManagedCloneTests.test_public_upload_prepare_results_save_generate_reopen_replay`

Initial result: 34 tests in 25.060 seconds, 32 PASS, one test fixture ERROR and one test assertion FAIL. The ASGI client double lacked `post`; the expected pre-registration voice fixture lacked Core's normalized metadata. Fixed only those tests: added `post` forwarding and compared the frozen voice to the actual owned catalogue version. **No production source changed after the initial run.**

Only those two corrected tests were rerun with the same interpreter/flags and explicit paths: 2 PASS in 1.785 seconds. Thus **34 distinct checks passed across the retained initial and targeted correction receipts**; this is not a claim of one all-green 34-test invocation. No broad unchanged repeat.

Raw retained receipts:

- `/private/tmp/pab-r328-sol-unittest-receipt.txt`, SHA256 `41e984a808269600c0dc5a08537e85ef281a93045866f4a35b1f2b4803d96314`.
- `/private/tmp/pab-r328-sol-corrected-tests.txt`, SHA256 `e322f51424d4538f17a2fa45c743536ce4c6a331479ec90aba60c98c89c063ff`.

New integration checks use real Dispatcher/Core/admission/Authority with actual Qwen context verification over sealed inert bundles, two injected CUDA indices and a fabricated hold/cancel/reap runtime. They prove measured candidate selection reaches the executable descriptor; exact voice/request retention; profile/device/model override refusal; stale/unqualified authority refusal; VRAM and concurrency bounds; missing/stale-bound measurement behavior; metadata-only discovery; canonical CLI and MCP selection; and same-root replay after ranking/candidate availability changes. No accelerator, containment or inference performance was measured. Existing actual public Pocket/native preset/managed clone fixture journeys passed unchanged apart from the discovery test double gaining the new owner metadata command.

`git diff --check` passed. Source SHA256 pins at test completion:

```
cd3010036de67d7168d30e035f7d8733541bbd0cf9eac16863983eec978d72f1  contracts/generate.py
2a12772220231d50d869e74a43e8db10bbf974594a99af16e0ad175214f83e20  contracts/v1/schema.json
76ee57d741a8444409c89c41a25e07a0f3d1b78862e65e35cd123356bc56a63a  service/api.py
e3438a04fae9f5d144770070edc495c616391e898199df4844c5e3ac16d3bf5e  service/dispatcher.py
8586d4cb1f6b01c66460f61a21be225c6827061c32aafcc90580166591fc981d  service/domain.py
f5a004e4b8a534d1259da9a8509ab21f0e7bb53278e10acf00abbb064b552477  service/profiles.py
e87ce186706edabde559465f4216eb393d7577ac667d06cadf493fa1f6ef0755  service/production_selection.py
4979848ed24f0b55aa1bb12c832d368ad9014f3f90b846c5f7b53be4ff67e51e  worker/runtime_protocol.py
97683dcff5997cad8a7acb3076141a7a5079d6d4d2f790a88852e3fdf642ca25  tests/service/test_installed_pocket.py
3766e24f0f4b5adbb2ffac50f9cb9cdc46a23812e3b624a1c6bd251d9f18c211  tests/service/test_production_selection.py
```

## Remaining limits and parent gates

The finite installed composition is still one configured Linux Kokoro/Pocket CPU engine or the distinct enrolled native Qwen MPS registry. This does not create a Linux Qwen installer, multi-engine installed deployment, native/Linux combined worker, cross-host selection or a universal runtime. The multi-index ranking integration uses existing private Linux Qwen contexts and supervisor semantics, not a newly supported installed Qwen package.

Public runtime execution still requires the existing exact request identity; `identity.device=auto`, `precision=auto`, automatic migration of a saved voice between CPU/CUDA/Metal and automatic model acquisition remain unsupported by these installed runtime contracts. The bounded automatic choice here is among compatible concrete executable candidates (including device indices), not an assertion that every desired hardware/model combination can execute. Explicit incompatible identities/overrides refuse. Model/voice provenance is never rewritten to manufacture alternatives.

Performance evidence is trusted operator metadata, not automatically acquired or independently attested here. Qualification revocation remains owner authority; inventory expiration is enforced at future selection, not by mutating historical plans. Live CPU policy availability is distinct from persisted accelerator probes. The existing scheduler remains bounded and serialized; this work does not qualify stronger-hardware throughput merely because profile resource capacity is larger. Generic private compositions do not establish new installed operation capabilities.

Metal remains **UNQUALIFIED**, with no authorized arm64 wheelhouse. CPU/CUDA/native Metal actual install, model execution, audible quality, latency/memory/throughput, crash/restart/renewal and release qualification remain their original gates. Native terminal-history/lifetime/topology limits remain. Legacy digest RED is **unchanged and unwaived**, in its separate lane; no legacy metadata or loader was modified.

Parent must obtain fresh independent Sol review of this exact source/report candidate, resolve objections, integrate under parent authority, then request Astra's integrated milestone review. No source adoption, release GO or waiver is asserted here.
