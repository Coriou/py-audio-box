# Independent managed Qwen provisioning review — r265 Sol

2026-10-24. **Bounded SOURCE NO-GO pending one P2 config-path validation correction.** The 67 functional checks and three separate fresh-import checks pass independently. No actual snapshot, loader/model, hardware, audio, installed activation or release qualification is established.

## Pins and review authority

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-008ea39f`, branch `t3/qwen-provisioning`, clean at entry and after checks. Verified base `6b9f81a1a7c972aa26227ccbf975efa7c89839ea` → source `f0b0217cf8575f8ed0a1d94018538a529269dc8f` → docs HEAD `46af39ca3737f2cd3b3c5e35f8a3baaf69b30d26`. Source-to-HEAD adds only r264. The actual source delta has exactly five files: service/provisioning.py, lib/tts_backends/assets.py, worker/engines/qwen.py, scripts/provision-tts.py and tests/service/test_qwen_provisioning.py.

Read full r264, r259–r262, r255–r256 and r188 including assignments 2/4; inspected the five-file delta, shared verifier, provisioning/downloader/sealing, Qwen descriptor/capability/loader seams, reference/file helpers, asset consumers, CLI, archive/import tests and selected regression source. No applicable candidate or checked ancestor AGENTS.md found. A bounded memory-registry lookup supplied no task-relevant premise. Root/service conftests were inspected before execution: direct unittest bypassed maintenance and listener fixtures.

Sole intentional lasting write is this root report. No source edit, commit, delegation, lifecycle/install activation, actual ML/hardware, Docker, external network/download, dependency installation, listener, process-crash campaign, operational cleanup or HELD action. Existing inspected mock-download tests use bounded inert children. Test-owned temporary teardown only. Free disk observed 3,597,460 KiB, below the 6 GiB heavy-work gate; only tiny fixtures and the existing small source archive were used.

## P2 — validate the type of a path declaration before descending into its value

`worker/engines/qwen.py:50–59` applies the `*_file`/`*_path` restriction only in the string branch. A dictionary loses the enclosing path key when its children are visited; numbers, booleans and null bypass the branch altogether. Consequently the shared verifier accepts and provisioning activates malformed explicit path declarations, including an absolute unlisted path hidden in an object. This contradicts the promised complete explicit local asset-reference validation. It is a validation-contract finding, not proof that an upstream loader follows this particular malformed object or an arbitrary-file-read exploit. Actual architecture/tensor validation remains outside this lot, but enforcing the verifier's own path-field rule is within it.

Reproduced independently with the candidate's existing tiny fixture, no loader or network:

```python
import sys
sys.path.insert(0, '/Users/ben/.txcode/worktrees/py-audio-box/t3-008ea39f')
from tests.service.test_qwen_provisioning import QwenProvisioningTests
from service.provisioning import manifest_bytes
for value in ({'location': '/outside/tokenizer.json'}, 17, ['tokenizer.json']):
    t = QwenProvisioningTests(); t.setUp()
    try:
        t.data['tokenizer_config.json'] = manifest_bytes({'tokenizer_file': value})
        t.repin()
        t.run_provision()
        assert (t.root / 'active.json').exists()
    finally:
        t.doCleanups()
```

All three activated. Controls with a direct `../outside.json` string, HTTPS string and `auto_map={}` refused. Reject unsupported value types at the path-bearing key before general dictionary/list recursion; define any intentionally permitted null/list semantics explicitly, and require every permitted path element to resolve to an authorized listed local asset. Preserve downloaded config bytes and their pins. Add negatives for dictionary/scalar/empty or null values and verify refusal preserves an existing active pointer. No production change was made by this review.

## Other assessed boundaries

- The extraction preserves raw-manifest digest/identity and complete streamed asset hash/size checks. Both provision and actual verify_inputs call the same pure verifier. verify_inputs still requires exact QwenExecution, trusted available/probed device inventory and topology, authorized bundle mapping, selected rate/preset/device tuple and checkpoint task/instruction permission. Clone still binds exact authorized voice/revision/checkpoint/revision/transcript and authenticated complete PCM16 bytes. Execution still validates before backend construction, uses named routes and exact offline device/dtype/attention forwarding, measures output and closes the backend. No fabricated inventory/reference was introduced to make provisioning pass.
- Managed Qwen declarations are closed, bounded and unique, with explicit English/French only. Existing Kokoro/Pocket fr/en contracts remain separate; no fallback or translation was added. Exact operator revision, per-file revision, expected hash/size, license/access and runtime-version equality are required. These declarations do not establish model capability truth or complete installed dependency closure.
- Nested staging uses the existing lock/budgets/downloader/resume path. Complete files are rehashed before reuse; partial files are discarded. Required root and speech_tokenizer assets are explicit. Nested files are verified before rename and sealed recursively before active pointer replacement; retries reseal existing complete bundles. Old active bytes remain preserved on the tested transfer/runtime/config/seal failures. Atomic replacement is not power-loss durability evidence. Trusted private storage, stable mappings and no hostile same-UID mutation remain premises.
- File count/per-file/aggregate/path-length limits bound the declared graph. JSON bytes are capped per asset; the recursive JSON inspector has no explicit application depth/node limit, relying on byte ceilings and interpreter recursion refusal. This is a hardening limitation, not a demonstrated activation bypass or separate blocker here. The verifier checks the actual file set and symlinks; empty directory count is not independently capped. Do not describe this as general hostile-filesystem traversal containment.
- Production downloads receive the explicit allowed HTTPS authorities; the existing redirect checks and exact stream caps remain. Trusted injected callbacks retain responsibility for their transfer bounds. Full descriptor identity includes origin policy, whereas active manifest_sha256 hashes exact runtime-manifest bytes with descriptor-only policy removed. Policy-only changes correctly create different storage directories with equal runtime bytes/digest.
- assets → Qwen verifier → Pocket file helpers is lazy; Qwen's backend capability import occurs inside verification, without importing the complete loader or ML. Root-only, legacy and extracted-archive regression checks pass. Existing Kokoro/Pocket consumers pass. No new module/distribution allowlist entry or circular-import failure was established.

## Independently executed checks

Every Python invocation used `/private/tmp/pab-0c-venv/bin/python -I -B` with explicit candidate paths. No pytest collection or combined stale-import run.

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c "import sys,unittest; sys.path.insert(0,'/Users/ben/.txcode/worktrees/py-audio-box/t3-008ea39f'); s=unittest.defaultTestLoader.loadTestsFromNames(['tests.service.test_qwen_provisioning','tests.service.test_provisioning','tests.worker.test_engine_qwen','tests.worker.test_engine_pocket','tests.worker.test_engine_kokoro','tests.service.test_provisioned_engine']); r=unittest.TextTestRunner().run(s); sys.exit(not r.wasSuccessful())"
/private/tmp/pab-0c-venv/bin/python -I -B /Users/ben/.txcode/worktrees/py-audio-box/t3-008ea39f/tests/worker/test_qwen_import_seam.py -v
```

Results: **67 PASS (0.869 s)** and **3 PASS (0.207 s)**. The separate fresh interpreter preserves the import-seam test's NumPy precondition; no assertion was weakened. These tests use inert safetensors text, fixture versions/revision/origin, fake backend results and injected ML/hardware, not actual models.

Additional tiny probe groups: the three accepted malformed path values above; three refusal controls; seven successful refusals for nested path escape, nested remote URL, custom pipelines, trust_remote_code, unlisted index shard, nested symlink and unlisted nested file. `git diff --check 6b9f81a f0b0217` passed. Candidate HEAD and clean status remained unchanged.

## Parent handoff

Correct P2 and obtain focused independent review before adopting this source candidate. Parent still owns adoption and immutable profile/settings authority, readiness invalidation, resource enforcement, lifecycle/publication fences and descendant ownership/reap before releasing capacity or reference namespaces. This lot adds no service lifecycle or installation activation.

Actual reviewed snapshot pins, license/access acceptance, full runtime dependency closure, offline loader compatibility, French/English capability and listening/audio quality, installed provisioning/HTTP/MCP journeys, constrained CPU/CUDA/native MPS performance/resource/cancel/restart and confirmed model/helper descendant reap remain unqualified. Those execution/install/release gates cannot be satisfied by these fixtures or this source review.
