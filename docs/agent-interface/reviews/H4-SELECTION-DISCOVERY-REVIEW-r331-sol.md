# r331 Sol — independent adversarial r328+r330 SOURCE review

2026-10-25. **NO-GO for declaring the bounded Lane 2 truthful-discovery exit complete: one P2 exact-device/inventory readiness defect remains.** r329's orphan-measurement finding is closed by this source and independent regression execution. The authorized final selection ran exactly once: **38 PASS, 27.611 seconds, exit 0**. Execution continues to refuse the newly reproduced discovery mismatch before allocation. This is SOURCE/inert evidence only, with no source adoption or release/hardware approval.

## Authority, pins and scope

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-ec6c788d`, branch `t3/truthful-discovery`, HEAD `3f99abdd76d5f7036a7e3c7999dc379b05015f21`. Verified history: base `a9f858332aeaeb0eef3cc0d99090c71373bac567` → r328 cherry-pick `e254197cd55ae706385fb7912e6065774a2c2fce` → reviewed HEAD. Base ancestry verified; candidate status empty before and after tests/probe; `git diff --check a9f8583 HEAD` passed. No applicable AGENTS.md found in candidate/root discovery or checked ancestors.

Read full r329 including its exact orphan reproduction, r330, r328, r323/Lane 2 and PRODUCT.md. Inspected actual base-to-HEAD and r330 diffs, complete production selection, profile selector, Qwen serialized context construction/metadata/resolution, owner replay/dispatch, Production and canonical snapshot/admission consumers, public API, runtime override decoding, MCP schema/forwarding and affected generator/fixture/test consumers. Memory supplied one-shot evidence discipline only; findings and pins derive from this candidate.

Sole lasting write: this ROOT report. No candidate edits, commit, merge, push, delegation, network/listener, Docker, install/download, actual child launch, ML, crash experiment, HELD access, STT work or operational cleanup. Direct unittest bypassed inspected conftests, including their listener/subprocess fixture. Tiny temporary filesystem/ZIP, ASGI, fake TLS/process/backend and child-function observations were authorized; fixture teardown is not operational cleanup. No heavy ACTUAL operation or 6 GiB waiver occurred.

Verified SHA256:

```
15584b88b4dfbb1fa24e5d6d6c1fa746927813ec87073327e29fdae7277d29c9  service/production_selection.py
32496405c8921591403e1d034d501644dcb03a15a7f0f9e8f63903d735ad045d  service/execution_context.py
188a63a506eb6b7a18af7214a511494a89307846699f50be0455f1b4b9d29558  tests/service/test_production_selection.py
cd3010036de67d7168d30e035f7d8733541bbd0cf9eac16863983eec978d72f1  contracts/generate.py
2a12772220231d50d869e74a43e8db10bbf974594a99af16e0ad175214f83e20  contracts/v1/schema.json
dcbe261d0b825587479879b2c0b4f016afa154dacf3ba030812cb1239d770753  /private/tmp/pab-r330-finite-tests.py
```

## P2 — bound measurement plus unrelated available device still advertises ready

`service/production_selection.py:243–250` establishes measurement binding without retaining the execution's exact device or hardware identity. Lines 253–256 then accept **any available device in the profile's family**. Thus the measurement can describe CUDA 0 while only unrelated CUDA 1 is available. This is a metadata-known impossibility, before owned-media or request-specific resolution. `choose` correctly checks the complete hardware wire at lines 186–190 and the pure selector checks exact device index. Discovery therefore promises ready when no configured candidate can execute.

One independent focused inert probe used the existing real owner/Core/admission fixture, same interpreter/flags and explicit paths, without source edits. After `ProductionSelectionTests.setUp()`:

```python
from dataclasses import replace
from worker.hardware import HardwareInventory
# Retain only the configured, measured CUDA-0 resolver.
t.composition.candidates = (t.composition.candidates[0],)
t.composition.measurements = {'gpu0': t.measurements['gpu0']}
# Fresh inventory now marks CUDA 0 unavailable; unrelated CUDA 1 remains available.
t.composition.inventory = HardwareInventory(t.inventory.host, tuple(
    replace(d, available=False) if d.index == 0 else d
    for d in t.inventory.devices))
p = t.discovery_profile()
r = t.request('POST', '/v1/operations/speech_generate', payload=t.payload,
              key='r331-unavailable-bound-device')
```

`t.discovery_profile()` patches Qwen resolution to raise if discovery invokes it, and checks authenticated capability/detail agreement. Observed:

```
DISCOVERY ready Inventory host and qualify exact backend/model/precision before advertising support.
SUBMISSION 503
code: not_ready
action: Operator must refresh/qualify the exact installed profile or correct the override: hardware_identity_changed
JOBS [] RESERVATIONS []
```

Probe asserted these outcomes, completed fixture cleanup and exited 0 (tool session 77824). It ran once, separately from the final 38-test selection. The stale configured execution still has its originally valid receipt; the current inventory change makes it unusable. Discovery's opaque execution hash cannot determine that binding, and its family-wide availability predicate incorrectly fills the gap.

Correction should remain metadata-only: expose sufficient detached canonical execution metadata to join the eligible candidate/receipt to exact device, current hardware binding and supported topology, or conservatively require binding verification when it cannot establish them. Require at least one candidate that satisfies these conditions together; do not combine one candidate's measurement with another device's availability. Add a regression for a measured configured CUDA-0 candidate with CUDA 0 unavailable and unrelated CUDA 1 available, preserving execution refusal/no allocation and no resolver/media/ML calls. Do not weaken `choose`'s hardware-identity check to align with discovery.

This is a P2 truthful-discovery defect, not an execution-authority bypass. It was already latent in r328's family-only discovery; r330 fixes the narrower r329 objection but does not close this remaining original-scope readiness exit.

## What is closed and preserved

r329's exact orphan case now yields `measurement_required` in discovery and HTTP 503 on submission with no jobs/reservations. Discovery joins the configured candidate key to its own Measurement, the canonical profile snapshot to owner metadata, the whole serialized execution hash to its receipt, and the nonempty policy workload to Performance.workload. Other-profile evidence, stale/forged hashes and wrong workloads cannot establish strict measured readiness. Unknown provider metadata produces conservative `measurement_binding_verification_required`; known missing evidence produces `measurement_required`. Explicit `allow_unmeasured` remains separate policy availability, not measured performance authority. Trusted operator providers/receipts remain the trust boundary; a hash is not independent performance attestation.

`QwenContexts.selection_metadata()` deserializes previously configured templates and uses the same canonical profile/hash definitions as execution selection. It returns detached objects and does not resolve owned media, inspect bundles/references, load ML or provision. The r330 production diff adds only this metadata method and discovery's measurement join; execution resolution/ranking/admission paths are unchanged by r330. The exact-device gap above limits its readiness claim.

r328's owner-side path retains principal/operation/key fingerprint replay first, including same-root replay after ranking/candidates change. New requests resolve and choose before Production/Admission; the selected complete context reaches existing frozen execution/Authority. Core remains the sole owner of jobs, reservations, publication/import, cancellation and retained reference/reap fences. No second scheduler is introduced.

Composition is bounded to 1..16 uniquely named trusted resolver candidates and must use the identical runtime object. Installed selection refuses multiple resolvers. Private Qwen multi-CUDA-index selection retains concrete model/bundle/voice/settings semantics; native binding and Linux/native topology refusal remain. Installation/configuration/qualification/current snapshot, inventory expiration, RAM/executable demand, CUDA VRAM budget and configured concurrency checks remain execution gates, alongside existing admission limits. Larger profile capacity does not qualify throughput.

Canonical prepare/generate overrides remain closed profile/device fields; boolean indices, extra device keys and unsupported families refuse. Existing request identity remains engine/model/bundle/device/precision authority. CLI and MCP reuse canonical HTTP validation/dispatch; no caller-supplied trusted execution descriptor or silent voice/model fallback is introduced. Generator/default fixture bytes remain stable in the selected determinism and consumer checks.

## Independent final test receipt

Read `/private/tmp/pab-r330-finite-tests.py` before use. Its root is this exact candidate, not r328's old worktree; explicit root/service/client paths are inserted. Inspected both conftests and selected lifecycle doubles before execution. Exactly one final-selection invocation under the fresh independent authority:

```
/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/pab-r330-finite-tests.py
```

Observed **CPython 3.14.7**, Clang 17.0.0, not 3.11. Tool session **62157** retains the named-test raw stream. Result:

```
Ran 38 tests in 27.611s
OK
exit 0
```

Exact nine loaded names:

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

All four r330 regressions passed, including the corrected other-profile reason, no-allocation assertions and positive submission tail previously UNRUN. Generator filename/byte determinism, committed fixture equality, request/exchange consumers, canonical clients and Pocket/native preset/managed clone journeys passed. This fresh coherent receipt does not rewrite r330's historical 37 PASS/1 FAIL receipt. No rerun or broader selection occurred. No additional receipt/runner file was written by this review.

## Parent-owned disposition and remaining limits

Resolve the exact-device/current-hardware discovery objection and obtain review of the resulting source under parent authority before claiming the bounded truthful-discovery exit complete. Parent owns adoption/integration and Astra's integrated milestone judgment; this report performs neither.

The existing installed boundary remains one configured Linux Kokoro/Pocket CPU resolver or the separate enrolled native Qwen MPS registry. Private multi-index selection does not supply universal installed hardware auto-selection, a Linux Qwen installer, multi-engine installed deployment, cross-host routing, model/device auto, automatic voice migration or automatic acquisition. Preserve concrete identity and Linux/native separation rather than widening this fix into topology work.

Native terminal-history/lifetime/reboot/renewal/topology limits, remaining FULL API/CLI/MCP lanes, CPU/CUDA/Metal actual clean install, speech/clone, cancellation/reap/restart/fault safety, listening and measured latency/memory/throughput remain separate parent-owned exits. **Metal remains UNQUALIFIED; no authorized arm64 wheelhouse. Release remains NO-GO.**

Legacy digest RED independently remains unchanged/unwaived: current `lib/tts.py` SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`, versus `contracts/v1/legacy-controls.json:3605` value `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`. No loader/control reconciliation occurred.
