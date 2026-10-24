# Qwen headless import seam correction — r261

2026-10-24. Implementation candidate closes the concrete r260 P1 in injected source tests. Fresh independent parent-owned Sol review is required before adoption. No actual engine, hardware, acoustic, installation or release qualification.

## Pins and scope

Own checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-ca925942`, branch `t3/qwen-headless-import-fix`, clean entry `538ea57` (root `c6c443dadde339f47ca3255f2467111e83cfc187` plus review documentation). Exact requested source `4ec18019b133c6e0a9b8d164417a0ff29531f9d5` then report `c056daefc5b739f363fe5c828a313e393d2913d8` cherry-picked as `de5c84d` then `f2dfc20`. Fix source commit **`1a79358c04ecef5806c7c7617b74027173bf8af3`**. This report is a separate documentation commit. Historical candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-f6c979da` remains clean at `c056daefc5b739f363fe5c828a313e393d2913d8`; no writes there.

Read full r260, r259 and original r188 including assignment 4; inspected the real backend/complete loader, runtime tests, archive packaging and root/service conftests. No applicable AGENTS.md found. Bounded memory registry search had no relevant hit and supplied no premise. No delegation.

## Deliberate compatibility fix

`lib/tts_backends/qwen.py` now chooses relative `..tts` for package context `lib.tts_backends`, and legacy `tts` for the existing `tts_backends` context. Both the actual worker loader and clone-prompt binder use this rule. `lib/tts.py` selects relative backend/contract imports when loaded as a package and preserves absolute backend imports when loaded as legacy `tts`. No path mutation, exception-driven alternate import, dependency-error masking or model-import relocation was introduced. Backend/helper import remains lazy with respect to Torch/upstream/numpy; the complete loader module is still imported only when needed by the existing worker/binding path.

No descriptor, device/dtype/attention, manifest verifier, resource bound, output contract, lifecycle, provisioning or installation policy changed. Existing explicit CPU/cuda:N/native Darwin arm64 MPS refusal and forwarding rules and legacy automatic policy remain intact. No manifest change was necessary: both fixed modules and their package dependencies are already archived.

## Executed evidence

All Python execution used `/private/tmp/pab-0c-venv/bin/python -I -B`, explicit test file paths or directly loaded inspected functions. No pytest collection or conftest execution.

- `tests/worker/test_engine_qwen.py -v`: **11 PASS**, unchanged supplied descriptor/adapter/AST loader checks.
- `tests/worker/test_qwen_import_seam.py -v`: **3 PASS**. Complete real worker and complete real loader under repository-root-only visibility, legacy lib-path visibility, and an extracted archive with checkout paths removed. Each case exercises CPU float32/eager, cuda:1 float16/flash_attention_2, and mocked native Darwin arm64 MPS float16/sdpa. Exact local path/device/dtype/attention/offline, revision-ready message, thread-before-load order and preset/design/clone named routes are asserted. Clone prompt construction and consumption remain in the in-process fake owner. Real clone binding imports and module identity are checked. Helper import rejects heavy-module loading; loaded modules must originate from the selected checkout/archive. Unavailable CUDA refuses and upstream ImportError retains its original sentinel without import retry; download is forbidden; production imports leave sys.path unchanged. Archive names exactly equal manifest plus launcher and the relevant archived source bytes match.
- Direct inspected existing runtime functions: **8 PASS**: checkpoint matrix, lazy identity, invalid audio layout, noninteger/complex audio refusal, atexit ownership, local config capabilities, worker output release, legacy automatic environment device selection. Fake Torch; real numpy for existing audio tests. No model.
- Direct inspected existing diagnostics: **5 PASS**: send/poll/recv races and constructor/start failure, all with fake transport/process objects. Actual crash/kill/restart tests were inspected and excluded.
- Three selected existing ArchiveTests: **3 PASS**: deterministic extracted help, Pocket archive dependency closure, manifest exact files/legacy metadata. Only ordinary finite Python import/help subprocesses in those existing archive tests; no model child or actual spawn inference.
- `git diff --check`: PASS. Source commit left tracked/untracked status clean before this report.

Initial new-test run reached ready and all three named routes but failed its class-identity assertion because test `patch.dict(sys.modules)` restoration discarded the backend imported inside that context. Corrected test isolation to use an import blocker without discarding the backend. Subsequent three-case run and final expanded three-case run passed. This was a fixture defect, not an additional product change.

Disk observed about 3.5 GiB free, below the 6 GiB heavy-work gate; only small inert temporary fixtures/archive bytes used. No actual Torch/model/hardware, Docker, network, download, install, listener, process crash, operational cleanup or HELD action. Tests establish import/forwarding source behavior only, not actual spawned-runtime support.

## r259 compatibility clarification and parent gates

The historical r259 statement about existing verifier extra-field compatibility is narrow and must not imply existing Qwen provisioning. Existing `verified_bundle` accepts `fr`/`en`; the new Qwen operator manifest uses upstream language names such as `English` under a separate Qwen execution contract. Shared schema vocabulary and tolerated extra fields do not make that bundle already provisionable or activatable by the existing path. No provisioning implementation or language translation was added. Historical r259 is unchanged.

Parent must obtain fresh independent Sol review of this exact source before adoption. Integration still owns immutable voice/settings/profile authority, resource admission/enforcement, trusted sealed bundle/capability provisioning, readiness invalidation, attempt/result/publication fences, and confirmed ownership/reap of model-child/helper descendants before releasing capacity/reference namespace. Native transport/package ownership remains separate. Actual constrained CPU, CUDA and native Apple Silicon Metal install/generate/cancel/restart, resource/performance, listening and integrated release acceptance remain separately authorized, unrun gates.
