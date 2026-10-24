# H4A hardware profiles — r172 implementation handoff

2026-10-24. Bounded SOURCE + fake policy tests complete; independent Sol review pending. Hardware qualification NOT RUN; release NO-GO remains for all three required actual hardware rows.

## Provenance and scope

Fresh supplied checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-33672c8a`, branch `t3/hardware-profile-selection`, clean starting HEAD `6e982c828df7c9d4a25974fca5d198916995c44c`, matching root HEAD at inspection. No reset/replacement. Source commit `d1491b18403bb54bd75eda6ba4be89dcf8eef236` contains exactly four new source/test files. This report is a separate subsequent commit; only this report is copied to root.

Read canonical IMPLEMENTATION-PLAN hardware section/H4/parallel ownership, r158 actual inventory, lib/tts.py get_device/best_dtype, lazy backend factory and backend capability declarations, backend contract, worker/runtime_protocol.py lifecycle/session boundary. No applicable AGENTS.md found in supplied worktree. Memory registry H4/r158 search had no relevant hit; no memory-derived implementation fact used.

## Implementation

- `worker/hardware.py`: frozen host/device/inventory models, stdlib process platform/architecture metadata with explicit operator deployment topology (unknown by default), persisted probe metadata discovery only. MPS requires native Darwin arm64; Linux arm64 Docker and Docker on Mac cannot qualify as native Metal. CPU/CUDA/MPS capability and availability are distinct fields. No runtime import, model scan/hash/load, subprocess GPU tool, download or default probe.
- Explicit `probe` accepts a provider callable only when invoked by an operator. Provider may later own lazy backend availability imports. Missing/negative/invalid/raising provider fails closed with bounded reasons; errors do not leak provider text. Probe does not qualify devices/profiles or mutate inventory.
- `service/profiles.py`: typed execution profiles, exact voice work constraints, resource budgets, performance evidence/objective, policy, per-profile diagnostics and resolved selection. Eligibility requires installed backend, provisioned+verified bundle, qualification timestamp/evidence, supported+available persisted device with probe timestamp, compatible topology, exact engine/checkpoint/revision/conditioning, feature subset, explicit language, requested dtype/attention, finite memory and concurrency budgets. Qualification and persisted availability are trusted inputs, not established by this code.
- No profiles ship by default. At most 256 supplied profiles; duplicate IDs/devices rejected. Per-profile reasons use a finite enum and deterministic order. Ready result contains the selected frozen profile, including resolved engine/checkpoint/revision/device index/dtype/attention/resources/concurrency, with selection reason and bounded diagnostics. Not-ready returns no executable profile.
- Ranking compares only supplied cold/warm seconds for the policy's exact workload key; ties use profile ID. Measurements carry evidence references. Missing/comparison-incompatible measurements refuse by default; explicit allow_unmeasured policy permits deterministic profile-ID selection and labels it unmeasured. Qualification remains mandatory even under that policy. There is no inferred fastest hardware claim, CPU preference, synthetic memory ceiling, or MPS bf16 assumption.
- Explicit profile/device override filters eligible candidates and cannot fall back. Auto preference can use only explicitly allowed fallback families, still passing identical semantic constraints; selection discloses fallback. No fallback policy means no family substitution when a preferred family is unavailable.

## Verification

Authorized interpreter `/private/tmp/pab-0c-venv/bin/python`:

- `-m unittest tests.service.test_profiles -v`: 10 PASS.
- `-m unittest discover -s tests/worker -p test_hardware.py -v`: 4 PASS.
- `git diff --check`: PASS before source commit.

Coverage: constrained CPU/backend absent; compatible measured CUDA and larger independent concurrency; native MPS versus Docker CPU/unknown topology; qualification/provision/device probe gates; exact engine/checkpoint/revision/reference conditioning; feature/language/dtype/attention refusal; memory/concurrency; conflicting/unavailable overrides; identity-preserving explicit fallback; cold/warm ordering, stable unmeasured selection and mismatched workload refusal; finite budgets and bounded inventory. Fresh subprocess guarded imports prohibit torch/numpy/transformers/qwen/onnxruntime/lib/socket during import and discovery; fake provider calls prove probes are explicit and fail closed. These are CPU-only source tests, not engine or hardware execution.

Initial invocation passed the ten selection tests but failed to create/import the hardware test because tests/worker did not exist. Created that directory (no extra tracked initializer) and ran the isolated discovery command above successfully. No pytest guard/full suite was run.

## Parent-owned seam and remaining required work

These dataclasses are an internal trusted selection seam, not new public JSON contracts or mutations of the lifecycle/execution protocol. Parent must own eventual serialization, immutable voice/version/settings binding, execution/result identity fencing, API/dispatcher admission and explicit fallback disclosures. Conditioning is an exact opaque identity; the caller must freeze the authorized voice/version and all settings before constructing constraints. Workload evidence must bind equivalent text/voice/model/settings and resource scope; no measurement validation or benchmarking occurs here.

Trusted installation/qualification records require a parent-owned persistence, provenance, freshness/expiry and invalidation policy before production use; timestamps alone do not prove current availability. Caller supplies process deployment topology; unknown topology cannot authorize Metal. Dtype/attention are qualified profile metadata, not backend support inferred from device family. This selector is not wired into existing jobs, legacy defaults, engine adapters or admission and does not confer runtime readiness on them.

Required next implementation: real CPU/CUDA engine adapters and provisioning/packaging; packaged native Apple Silicon Metal worker with authenticated scoped transport, transfer/path mapping, child cancel/reap and publication fencing; model resident reuse/resource admission; then actual offline install/generate/cancel/restart, constrained CPU responsiveness and bounded higher concurrency, cold/warm/resource measurements and human listening on all three families. Apple Silicon native Metal remains mandatory, not replaced by arm64 Docker CPU. Actual host/runtime/model/access rows remain unqualified; none was probed here.

No Docker, provision/download, network, model/ML import, host ML install, GPU/Metal tool/benchmark, paid host, lane operation, delegation, core/API/dispatcher/contracts/backend/protocol/packaging edit or held-resource action. All HELD resources untouched. Parent integration and independent Sol review are next; this report does not self-approve release or hardware support.
