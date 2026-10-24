# H4 Pocket CPU generation adapter — r240 Sol implementation

2026-10-24. Bounded source implementation submitted for independent review. **No self-adoption, actual inference, lifecycle integration, acoustic qualification or release approval.** Parent commissions the separate Sol low adversarial review.

## Pins and scope

Owning checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-5dcb9a62`, branch `t3/pocket-cpu-adapter`, entered clean at required integrated base **`2a374c998135a0b18000ac5187ac5e155522da8c`**. No fast-forward was needed. Source commit **`a023bba32100c24d06aba370e76b94d43206712f`** adds exactly:

- `worker/engines/pocket_execution.py`
- `worker/engines/pocket.py`
- `tests/worker/test_engine_pocket.py`

This report is committed separately afterward. No root-checkout write, delegation, service/core/persistence/recovery/contracts/installer/provisioning change, or legacy backend edit. Status was clean after source commit.

Read PRODUCT, r188 assignment 3, r238 and r239, existing execution/Kokoro adapters, PocketBackend, bundle verifier and audio contract. No applicable ancestor or descendant AGENTS.md was found. Inspected the root conftest; no worker conftest exists. A bounded memory registry search returned no matching Pocket premise and supplied no implementation assumption.

## Implemented boundary

`PocketExecution` is frozen and closed on wire decoding. It binds bundle ID/raw manifest SHA-256/checkpoint/revision/language, exact thread count, text/output/memory bounds, authorized reference ID/full hash/byte length/frame count/rate, and exact output rate/mono PCM16 WAV format. CPU and Pocket are fixed; only explicit French/English are accepted. No public path, seed, transcript, arbitrary model controls, preset or instruction enters the descriptor. Even explicitly supplied empty presets/instructions are refused by execute.

`execute` resolves IDs only through integration-owned bundle/reference mappings. It never resolves a public arbitrary filename. Paths must be canonical absolute paths with no symlink components; component traversal uses directory descriptors and O_NOFOLLOW. Final managed directories/files must be current-UID owned and not group/world writable; regular files must have one link. Bounded reads check size and metadata stability. The raw manifest is capped at 64 KiB and must match the exact descriptor digest and selected model/revision/language before backend construction. Its asset inventory is bounded to 128 unique safe relative paths, 2 GiB per file and 4 GiB aggregate; all bytes are hashed in 1 MiB chunks. Only safetensors/YAML/JSON assets are admitted, never pickle/pt/bin. Config must be present. Known optional existing inventory provenance fields origin/revision are allowed, without granting authority. The existing PocketBackend independently verifies the local bundle and local config paths when invoked.

Reference resolution checks the complete bounded file digest, exact byte count and mono PCM16 WAV header, rate, frame count and complete declared PCM payload before backend construction. References are capped at sixty seconds and 192 kHz. The authorized bytes are copied to a mode0600 temporary WAV inside the supplied existing private owned mode0700 attempt namespace. Backend construction receives exactly the selected local root, language, threads and namespace; synthesis receives exact authored text and that private snapshot only. Original mutation after capture cannot substitute conditioning bytes. PocketBackend takes its own identical snapshot inside that namespace. No Backend change was necessary, preserving direct legacy behavior.

Only our adapter snapshot is unlinked on normal unwind. Backend close runs after synthesis/output failures and success; close failure propagates and no successful result is returned. Constructor failure also unwinds the adapter snapshot, but partial constructor-owned runtime state cannot be closed through an instance that was never returned. Neither adapter nor backend deletes the caller's namespace or other caller files. Forced reap cannot run Python cleanup: the lifecycle owner must retain the namespace, prove descendant death/reap, and then perform separately authorized cleanup. This lot adds no such integration or cleanup authority.

Results must be exact AudioResult instances. Mutable audio is revalidated for integral exact bound rate, one channel, finite real normalized numeric samples, shape, positive frame count and output byte/frame limits. PCM16 WAV serialization uses the actual validated backend rate. The returned bytes have measured frames, byte length and SHA-256 plus a fresh detached complete descriptor identity and SHA-256 of the exact submitted UTF-8 text. ML imports occur only through backend construction after validation; importing the adapter/descriptor imports neither ML nor NumPy.

## Verification

All tests used `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.` and **`/private/tmp/pab-0c-venv/bin/python`**, reporting **Python 3.14.7**. No pytest collection, maintenance guard or listener fixture ran.

Final focused commands:

```
/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/worker -p test_engine_pocket.py -v
/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/worker -p test_engine_kokoro.py -v
```

- **11 Pocket methods PASS**, including mutation loops: exact factory/synthesis arguments, exact private snapshots/source mutation, detached identity and WAV measurement; closed descriptor and explicit unsupported controls; missing/foreign IDs and exact model/language/revision/reference binding; missing/tampered/symlink/hardlink/writable/wrong-owner files; oversize/reference-format refusal; unsafe/duplicate/escaped/missing-config/pickle inventory, size/hash mismatch; nonfinite, complex, nonnumeric, empty, multichannel, wrong-rate and oversized outputs; call/close/constructor failures and caller-file/namespace retention; lazy imports.
- Pocket compatibility method invokes the unchanged real PocketBackend class with inert `torch`, `pocket_tts` and YAML stubs. It exercises backend-private conditioning snapshots, generation result forwarding, default direct-call namespace, seed and state-cache behavior. No real Torch/YAML dependency, upstream loader, weights or model runs. YAML was absent in this interpreter, so the fixture deliberately supplies an inert parser/dumper rather than installing anything. This is not real config parsing or inference evidence.
- **7 existing Kokoro methods PASS** in the final focused run.
- An earlier broader pattern `test_engine_*.py` also selected **11 existing lifecycle methods**, for 27 methods at that point (then 9 Pocket + 7 Kokoro + 11 lifecycle), all PASS. This selection was broader than intended. Subsequent inspection confirms injected in-process transport/backend/runtime, temporary SQLite/staging fixtures and one dispatcher thread; supervisor serve is mocked. It did not invoke a real worker process, listener or crash. Final runs use literal individual module patterns. These lifecycle methods do not establish Pocket lifecycle integration.
- `ast.parse(..., feature_version=(3,11))` PASS for all three added Python files under Python 3.14.7. **Grammar compatibility only, not execution on Python 3.11.**
- Staged source `git diff --cached --check`: PASS. Source commit left the worktree clean.

Initial fixture-only failures were corrected: macOS TemporaryDirectory spelled through `/var` is rejected by component no-follow traversal, so fixtures use its canonical `/private/var` path; the compatibility fixture supplies an inert YAML stub because PyYAML is absent. Affected focused tests passed afterward. Unique final/earlier executed coverage is 29 methods (11 Pocket, 7 Kokoro, 11 existing injected lifecycle), with internal matrices beyond method counts.

No Docker, downloads, actual model loading/inference, network/listener, real process crash, historical HELD resource access or operational cleanup. Only small owned test fixtures and their ordinary temporary-fixture teardown ran, within the source-test authorization below the 6 GiB actual gate.

## Integration limits and handoff

The mappings confer authority only when the integration owner derives them from the owned immutable voice/reference and exact selected local bundle. This adapter does not authenticate a principal, own durable reference storage, enforce OS memory/time limits, reserve resources, or prevent a malicious same-UID writer changing supposedly immutable managed storage between verification and backend reopening. Canonical path storage must remain stable and immutable throughout the call. The backend's existing safe local config/safetensors loader remains responsible for semantic weight/config decoding; this lot verifies inventory bytes and reference WAV format without loading model files.

Existing legacy provisioned bundles without an explicit top-level revision are deliberately not silently upgraded or given a fallback revision. Integration must authorize a compatible exact manifest. Only the bounded safe asset extensions above are supported; any further upstream asset format needs explicit review rather than a permissive loader fallback.

This is generation from an already prepared authorized reference. It does not select ticks, decode/clean/resample source audio, consume a transcript, prepare a reusable reference, create an audition pair, publish an asset, or prove audible speech/clone similarity. Those transformations and roles remain independently owned and must retain r238/r239 semantics.

Parent still owns execution-context construction, attempt mapping/namespace allocation, supervisor admission and OS limits, child invocation, cancel/deadline/stop/reap/descendant tracking, post-reap namespace ownership, publication/provenance protocol, immutable saved-reference restart reuse, and real CPU listening/resource qualification. No service runtime route is activated by these new modules. CUDA/nativeMetal, Qwen, UI and STT are outside this lot. Independent adversarial source review precedes parent adoption; later actual qualification requires separate authority and sufficient disk admission.
