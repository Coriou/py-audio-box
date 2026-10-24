# Independent corrected managed Qwen review — r267 Sol

2026-10-24. **Bounded SOURCE GO for parent adoption of this exact candidate. The r265 P2 is resolved; no new material source finding established.** Actual snapshot availability, offline loader/model compatibility, installed activation, hardware, acoustic and release qualification remain separate.

## Pins and authority

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-be31f699`, branch `t3/qwen-config-path-validation`, clean before and after checks at `178614286042f26bd542f512753d42efdaefe268`. Verified parent `4bf0e43d5d1e62ac5dbd947a90ec8ebbe86ea09b` → source cherry-pick `eaec8dd8bbfa233de55094b6051af0a3f7b35e71` → r264 documentation `d855179c32c6c265384663d30483234cdfe6ab7f` → correction `72ab60c327a6bfa4636b8f2a854cc5e4dd4a6b3f` → documentation HEAD above. The correction changes only worker/engines/qwen.py and tests/service/test_qwen_provisioning.py. The full parent-to-HEAD delta contains five source/test files and two reports.

Read full r264/r265/r266 and r259–r262, and the r188 original assignment context. Inspected the complete provisioning implementation, shared Qwen verifier and execution boundary, descriptor, assets dispatch, CLI delta and focused fixtures/import checks. Checked candidate/ancestor AGENTS.md locations and repository discovery; none found. Bounded memory-registry search returned no relevant hit and supplied no premise. Root/service conftests were inspected: direct unittest bypasses the maintenance collection guard and HTTP listener fixtures.

Sole intentional lasting write is this root report. No source edit, commit, delegation, actual ML/model/hardware, Docker, external network/download, install, listener, process-crash campaign, operational cleanup or HELD action. Only inspected tiny inert fixtures, injected backends/ML and existing mock-download children were exercised, with test-owned temporary teardown. Disk observed 3,551,552 KiB available, below the heavy-work gate; no heavy work was attempted.

## Findings

**Closed P2:** worker/engines/qwen.py validates every `_file`/`_path` key before dictionary/list traversal. The value must be a nonempty, non-whitespace string whose path, relative to its containing JSON asset, names a listed local asset. Absolute paths and parent traversal refuse. Dictionary/list/integer/boolean/null/empty values cannot erase the enclosing key, be coerced or receive an exception. General nested inspection and remote URL/code refusals remain. Config bytes and operator hash/size pins are unchanged by validation.

The three added methods substantiate the correction rather than merely testing the helper in isolation: 72 provisioning refusal subcases preserve the old active pointer and all old bundle file bytes; 48 independently repinned malformed local-bundle subcases refuse through actual verify_inputs and execute with no backend construction; 12 positive subcases cover both suffixes and nesting forms in root/subdirectory JSON, preserving expected pins and downloaded bytes verbatim. These are subcases within three methods, not 132 extra unittest cases.

No new semantic bypass or Kokoro/Pocket regression was established within the trusted-operator/private-storage premise:

- Managed Qwen descriptors remain closed and bounded, requiring explicit immutable operator revision, per-file size/hash/origin/revision, HTTPS authority policy, license/access metadata, exact runtime-version equality and declared execution capabilities. English/French is Qwen-specific; Kokoro/Pocket retain en/fr. No translation, discovery or generated Qwen configuration was added.
- Provisioning reuses the existing lock, finite lock/transfer budgets, cancellation, partial discard, complete-file rehash/resume, staging and atomic active-pointer replacement. Nested assets are verified and recursively sealed before activation; seal retry and failure preservation remain covered. Full descriptor identity includes origin policy; the active digest authenticates exact runtime-manifest bytes. Atomic replacement does not establish power-loss durability.
- Provisioning and verify_inputs call the same pure local verify_bundle. Full raw-manifest identity, owned no-follow files, complete size/hash checks, actual listed file set and hashed config/capability checks precede activation or backend construction. Hardware inventory/probe/topology and exact selected device/dtype/attention/rate/preset checks remain in execution. Clone retains exact authorized reference identity/transcript and complete PCM16 authentication; named routes, measured output limits and close remain intact.
- The managed JSON policy is selected by the runtime manifest's `versions` field. Legacy explicit operator bundles retain their earlier contract. Under the stated trusted operator/pinned manifest premise, this compatibility distinction is not a new activation bypass; provisioning always emits versions. This review does not confer public authority to construct trusted bundle mappings or strip manifest fields.
- Qwen dispatch is lazy and backend-specific. Fresh root-only, legacy and extracted-archive checks exercise the actual worker and complete loader with injected ML. Existing Pocket/Kokoro regressions pass. No module/distribution entry is added by this change.

The r265 limitations remain: no explicit JSON depth/node budget beyond byte caps/interpreter refusal, no independent empty-directory count cap, no hostile same-UID filesystem isolation or tensor/architecture semantic validation. No expanded hostile-operator framework or new blocker is inferred from these unchanged limits.

## Independently executed evidence

Executed the exact r266 functional command, then the import file in a separate fresh interpreter, each once:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c "import sys,unittest; sys.path.insert(0,'/Users/ben/.txcode/worktrees/py-audio-box/t3-be31f699'); s=unittest.defaultTestLoader.loadTestsFromNames(['tests.service.test_qwen_provisioning','tests.service.test_provisioning','tests.worker.test_engine_qwen','tests.worker.test_engine_pocket','tests.worker.test_engine_kokoro','tests.service.test_provisioned_engine']); r=unittest.TextTestRunner().run(s); sys.exit(not r.wasSuccessful())"
/private/tmp/pab-0c-venv/bin/python -I -B /Users/ben/.txcode/worktrees/py-audio-box/t3-be31f699/tests/worker/test_qwen_import_seam.py -v
```

**70 functional tests PASS (2.231 s); 3 fresh import-seam tests PASS (0.215 s).** No pytest collection. The import checks preserve their fresh NumPy precondition. Mock loader log lines naming CPU/CUDA/MPS are injected forwarding evidence only. Existing selected lazy-import tests launch finite import-only Python children; downloader tests patch network I/O before bounded inert transfer children. No inference child or real network request was exercised. `git diff --check 4bf0e43d5d1e62ac5dbd947a90ec8ebbe86ea09b HEAD` passed; candidate status remained clean and HEAD unchanged.

## Parent handoff

Parent may adopt this exact SOURCE candidate; this review does not adopt or activate it. Parent still owns immutable profile/settings/mapping authority and readiness invalidation, resource admission/enforcement, lifecycle/result/publication fences and confirmed model-child/helper descendant reap before releasing capacity or reference namespaces.

Supply reviewed real snapshot pins and complete compatible assets, license/access acceptance and runtime dependency closure separately. Actual offline loader compatibility, French/English support and listening/audio quality, installed provisioning/HTTP/MCP journeys, constrained CPU/CUDA/native Apple Silicon Metal resource/performance/cancellation/restart and confirmed descendant reap remain unrun qualification gates. Inert fixture hashes, injected hardware and source GO satisfy none of those gates.
