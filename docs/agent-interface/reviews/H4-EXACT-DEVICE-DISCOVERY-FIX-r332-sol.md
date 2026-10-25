# r332 Sol — exact candidate/current hardware discovery fix

2026-10-25. Implementation candidate for fresh independent review, then parent-owned Astra judgment. Final coherent selection: **45 PASS, 29.185 seconds, exit 0**. This is SOURCE/inert evidence only; no self-adoption, integration, installed/hardware qualification or release approval.

## Authority and exact pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-d4d10b25`, branch `t3/exact-device-discovery-fix`. Clean starting HEAD `c338229bedbce2ba6c758700c99f9f8713a5aacf`. Required ordered cherry-picks:

- `e254197cd55ae706385fb7912e6065774a2c2fce` → `edc88a3efb599920e769e794a30d11f92effb0d9`.
- `3f99abdd76d5f7036a7e3c7999dc379b05015f21` → `c8bad623a39ccbd5754376098b0fb5ee677d47a4`.

Read full root-provided r331 report, including its exact probe, full r330/r329/r328/r323 and PRODUCT.md. Inspected actual choose/discovery, pure profile selector, canonical execution decoding, Qwen metadata/resolution/current hardware, Dispatcher duplicate and admission paths, affected tests and conftests. No applicable AGENTS.md found in worktree discovery or checked ancestors. Memory supplied existing ownership/frozen-plan conventions only; current source and receipts establish findings.

Lasting writes are source/tests and this report in this worktree. No root checkout writes, merge, push, subdelegation, self-adoption, network/listener, Docker, install/download, actual child process, ML, crash experiment, HELD access, STT or operational cleanup. Authorized tiny ASGI/filesystem/ZIP/fake TLS/backend/process observations and fixture teardown only. No heavy ACTUAL operation; no 6 GiB waiver.

## Correction and gate audit

Qwen selection metadata now returns detached canonical profile, whole-execution hash, engine, exact device, executable memory/threads, hardware wire and native binding. It decodes configured serialized templates through the existing pure execution validator. It does not expose owned voice/media, bundle/reference mappings or full descriptors; it does not call resolution or input verifiers, read media, import ML or provision.

Discovery evaluates each configured template independently. At least one SAME candidate must satisfy all metadata-known predicates together. It never combines a measured stale candidate with another available device/candidate. Candidate failures are diagnostic only when no candidate succeeds. Unknown/malformed metadata conservatively requires actionable binding verification, including under explicit unmeasured policy.

Systematic discovery-versus-execution audit:

- Current canonical profile snapshot must match the candidate; global installed/configured/ready/measured/qualified authority and receipt remain required. Pure selection also checks qualification timestamp/evidence. Synthetic qualification receipts cannot establish typed readiness.
- Candidate-key measurement must bind that template's complete execution hash and the nonempty policy workload. A supplied stale receipt blocks even explicit unmeasured policy, matching choose. Missing comparable evidence can pass only the existing explicit allow_unmeasured policy; no performance claim is invented.
- Whole hardware wire must equal the current injected inventory, including host, indices and probe facts. Matching device index alone is insufficient. Exact family/index must also be supported, available and fresh; unrelated CUDA 1 cannot qualify CUDA 0. Qwen's existing 300-second current-hardware bound remains conservative even if composition's configured age is longer.
- Executable memory/threads must agree with canonical profile limits; executable RAM, CUDA VRAM and configured concurrency must fit existing owner/profile capacity. Discovery reuses the existing pure `select` predicate and policy for exact device/topology/RAM/concurrency/qualification/workload/preference gates, using request-neutral identity placeholders. It promises no particular voice/request compatibility.
- Composition must share the identical runtime; installed composition remains one resolver. Per-template native binding, native versus Linux runtime family, host topology and pure Metal topology rules are checked. Unknown providers are not resolved to fill gaps.
- Existing owner readiness/activation/fault and runtime admission gates remain. Request-specific identity/overrides, owned voice/media, sealed inputs, runtime registration/authorization, current reservations/queue/disk and full admission remain request-time authority; discovery does not promise them.

`choose`, owner resolution, duplicate-first replay, frozen plan and allocation paths are unchanged by this fix. No schema, generator, fixture, public Pocket/preset/clone or topology policy expansion. Existing installed Linux CPU versus enrolled native registry boundary and native lifetime/history/renewal limits remain.

## Regression and executable receipts

Inspected both conftests; direct unittest bypasses pytest autouse and the listener/subprocess fixture. Selected lifecycle code uses inert sealed files, in-process ASGI, fabricated TLS/connectors/process handles/backend and child-function execution, without an actual child launch.

Seven added tests cover:

- Exact r331 CUDA-0 unavailable/CUDA-1 available reproduction, discovery not-ready and new submission 503 `hardware_identity_changed`, jobs/reservations unchanged.
- Changed hardware identity with identical device indices, with the same refusal/no-allocation checks.
- Matching hardware wire/hash but exact CUDA 0 unavailable, proving the exact pure device predicate still refuses.
- Multi-candidate witness mixing: only stale-hardware gpu0 measured; current gpu1 unmeasured. Discovery and submission refuse. Adding gpu1's own bound measurement makes discovery ready and dispatches CUDA 1 despite the bad alternative.
- RAM/VRAM, concurrency, identical runtime and installed-composition gates. Installed-composition execution is checked directly before resolution because the fixture has no genuine installed activation; other negative cases use authenticated new submissions/no allocation.
- Runtime topology and conservative unknown metadata even under allow_unmeasured.
- Detached metadata mutation and minimal field set, preserving positive discovery.

The shared discovery helper patches Qwen resolution to raise and checks authenticated capabilities/detail agreement. Existing orphan/profile/hash/workload regressions and positive same-candidate execution remain.

Development invocation: explicit current-root/service/client paths, `/private/tmp/pab-0c-venv/bin/python -I -B -c ...`, loaded `ProductionSelectionTests`: **18 PASS in 6.635 seconds, exit 0**. No corrective rerun was needed. Receipt `/private/tmp/pab-r332-development.txt`, SHA256 `aa618a703b5bbd72716f8cf68ca493cd7163725113c116e46dfa81a6f66fc5c1`.

Exactly one final coherent invocation after final source/test edits:

```
/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/pab-r332-finite-tests.py
```

Observed CPython **3.14.7**, Clang 17.0.0, not 3.11. **45 tests in 29.185 seconds, OK, exit 0**. Prior 38 plus seven new tests. Explicit nine loaded names:

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

Final receipt `/private/tmp/pab-r332-final-receipt.txt`, SHA256 `9fba47517e71d5a86f3652a5ec1c7ed4fb7704adb7e03a029d10974435a9f9c4`.
Runner SHA256 `2c96025bf2a289d69566674518aa4a81e10af869e5518206f5391553b6dc8817`. Runner inserts this exact worktree and explicit service/client paths. Generator filename/byte determinism, committed fixture equality, request/exchange consumers, canonical clients, replay and public Pocket/native preset/managed-clone journeys passed. No production/test changes followed the final run. `git diff --check` passed.

Final source SHA256 pins:

```
9704e5f2ffe98525f0cea5f19282f5025b05beb7859ba5cfd4e31c77ff85b3b7  service/production_selection.py
e3ed424bcec6c42fed27c151d13240189ac39d8775b49515e59f0ccc7f73f9fe  service/execution_context.py
c67195d3eb5289fd0485b986ed45917850b01cc385b9565e75f71e99ef7111ab  tests/service/test_production_selection.py
cd3010036de67d7168d30e035f7d8733541bbd0cf9eac16863983eec978d72f1  contracts/generate.py
2a12772220231d50d869e74a43e8db10bbf974594a99af16e0ad175214f83e20  contracts/v1/schema.json
```

## Parent handoff and limits

Obtain fresh independent review of this exact committed candidate, resolve objections under parent authority, then Astra integrated milestone judgment. No review/adoption is performed by this implementation sub-agent.

This remains trusted operator metadata and SOURCE/inert selection evidence, not measured hardware performance, containment, installation, speech quality or release qualification. FULL API/CLI/MCP gaps and actual CPU/CUDA/Metal install, speech/clone, cancellation/reap/restart/fault, listening and measured latency/memory/throughput remain parent-owned gates. Metal remains **UNQUALIFIED**, no authorized arm64 wheelhouse. Release remains **NO-GO**.

Legacy digest RED remains unchanged/unwaived: current `lib/tts.py` SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528` versus recorded control `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`. No loader/control reconciliation or operational cleanup occurred.
