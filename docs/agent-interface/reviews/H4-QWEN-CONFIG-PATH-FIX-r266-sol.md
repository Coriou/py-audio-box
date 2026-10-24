# Qwen config path declaration correction — r266

2026-10-24. SOURCE correction candidate for the r265 P2. Fresh parent-owned independent review is required before adoption. No actual model, installed activation, readiness, hardware, audio or release qualification.

## Exact pins and scope

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-be31f699`, branch `t3/qwen-config-path-validation`, clean entry `4bf0e43d5d1e62ac5dbd947a90ec8ebbe86ea09b` (root `6b9f81a1a7c972aa26227ccbf975efa7c89839ea` plus r265 review documentation). Requested source `f0b0217cf8575f8ed0a1d94018538a529269dc8f` and report `46af39ca3737f2cd3b3c5e35f8a3baaf69b30d26` were cherry-picked in order as `eaec8dd8bbfa233de55094b6051af0a3f7b35e71` and `d855179c32c6c265384663d30483234cdfe6ab7f`. Correction source commit: **`72ab60c327a6bfa4636b8f2a854cc5e4dd4a6b3f`**. This report is a separate documentation commit.

Historical candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-008ea39f` was inspected read-only and remained clean at `46af39ca3737f2cd3b3c5e35f8a3baaf69b30d26` before and after implementation. No writes there.

Read full r265, r264 and r259–r262 context, current shared Qwen verifier/execution and provisioning fixtures. No applicable AGENTS.md found in this checkout or checked ancestors. Bounded memory registry search returned no relevant hit and supplied no premise. Inspected tests/conftest.py and tests/service/conftest.py; direct unittest avoided maintenance-guard collection and listener fixtures.

Correction changes only worker/engines/qwen.py and tests/service/test_qwen_provisioning.py. No provisioning framework, lifecycle, installer, hardware inventory, depth framework or unrelated hardening change.

## Contract and regression coverage

The JSON inspector now validates every key ending in `_file` or `_path` before descending into its value. Such declarations require a nonempty, non-whitespace string resolving relative to the containing JSON asset to an explicitly listed local asset, without absolute paths or parent traversal. Dictionaries, lists, integers, booleans, null and empty strings refuse. No null exception is established or allowed; no upstream need for one was demonstrated in this task. Values are neither coerced nor discarded. General nested dictionary/list inspection and remote-code/URL refusals remain.

Three new unittest methods exercise both suffixes at the root, inside a dictionary and inside nested lists/dictionaries. Provisioning negatives cover dictionary, list, integer, both booleans, null, empty/whitespace strings, absolute/parent/unlisted/remote strings: 72 subcases. Every refusal checks the existing active pointer and every old bundle file byte. Correctly repinned malformed local bundles independently exercise actual verify_inputs and execute: 48 subcases assert refusal and zero backend-factory calls. Positive listed strings cover both suffixes and all three nesting forms in both root and speech_tokenizer JSON assets: 12 subcases check successful provisioning/shared inputs, unchanged expected pins and verbatim downloaded bytes.

The existing pure verify_bundle remains shared by provisioning and verify_inputs. Exact operator pins, closed English/French declarations, sealed nested local graph, old-active preservation and execution hardware/reference authorization are retained. This correction manufactures no hardware evidence and rewrites no downloaded config bytes or hashes.

## Executed evidence

Every Python invocation used the existing isolated interpreter with explicit candidate paths, without pytest collection:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c "import sys,unittest; sys.path.insert(0,'/Users/ben/.txcode/worktrees/py-audio-box/t3-be31f699'); s=unittest.defaultTestLoader.loadTestsFromNames(['tests.service.test_qwen_provisioning','tests.service.test_provisioning','tests.worker.test_engine_qwen','tests.worker.test_engine_pocket','tests.worker.test_engine_kokoro','tests.service.test_provisioned_engine']); r=unittest.TextTestRunner().run(s); sys.exit(not r.wasSuccessful())"
/private/tmp/pab-0c-venv/bin/python -I -B /Users/ben/.txcode/worktrees/py-audio-box/t3-be31f699/tests/worker/test_qwen_import_seam.py -v
```

First functional run: **70 PASS (2.114 s)** = unchanged 67 plus three new methods. Separate fresh import-seam run: **3 PASS (0.232 s)**, root-only/legacy/extracted archive with injected ML/device behavior. Total 73 selected unittest cases; subcases are not additional test-method counts. git diff --check passed. Source commit left status clean before this report.

Free disk observed 3,549,560 KiB, below the 6 GiB heavy-work gate. Only tiny inert fixtures and existing small archive checks were used. No actual ML/model/hardware, Docker, external network/download, dependency installation, listener, process-crash campaign, operational cleanup, HELD action or delegation. Existing injected downloader checks and test-owned temporary teardown retain their bounded behavior.

## Parent handoff

Obtain new independent review of the exact correction before adoption; this implementation does not self-approve. Original provisioning source/report remain unmerged candidates. Existing r265 hardening notes are unchanged and were not expanded into separate blockers. Parent retains immutable authority/readiness invalidation, resource/lifecycle/publication fences and confirmed model/helper descendant reap. Actual reviewed snapshot pins/runtime closure, offline loader compatibility, English/French listening, installed journeys and constrained CPU/CUDA/native Metal performance/cancel/restart/release qualification remain separate unrun gates.
