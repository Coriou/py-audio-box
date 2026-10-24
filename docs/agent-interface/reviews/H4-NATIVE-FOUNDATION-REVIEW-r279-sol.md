# Partial native foundation independent review — r279 Sol

2026-10-25. **Bounded GO for the implemented partial SOURCE foundation at r278. No material issue found within that subset. Full Lot A acceptance and native activation remain NO-GO.** This disposition neither accepts the r277 vertical slice nor authorizes installation, readiness, Metal speech, cancellation/restart qualification or release.

## Authority and exact candidate

Review-only authority; sole lasting write is this mode-0600 report in root. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-05c63b46` was clean at entry and after checks.

- Base: `326c4d28ecbb935b96cb3b50e35342b076a66ac4`.
- Source: `49dbb509d24040c0d446a5b66ea519962d4f9900`.
- Documentation HEAD: `f8f7a63ff2da8fcf681c914e3395452f5f4e11d9` (resolved from `f8f7a63`). Source-to-HEAD delta is solely r278's report.
- Root HEAD at report entry: `326c4d28ecbb935b96cb3b50e35342b076a66ac4`. Existing untracked STT evidence was listed only, not opened or changed.

Read full normative `H4-NATIVE-WORKER-CONTRACT-r277-sol.md` and full `H4-NATIVE-WORKER-SOURCE-r278-sol.md`, plus the actual eight-file base-to-source delta and relevant loader, engine, hardware, contract and test consumers. r277 SHA256 is `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`. No applicable AGENTS.md was found in checked ancestors/candidate discovery. Memory supplied only review/evidence-boundary conventions; explicit current authority permits the named inert tests.

## Findings and bounded assessment

No material correctness or authority-boundary issue found in the implemented subset. The following conclusions are confined to source inspection and injected fixtures:

1. `lib/tts_backends/qwen_native.py:16–102` is an in-process preset/clone owner. Module import does not import ML; constructor sets thread/offline/tokenizer environment before importing the complete existing loader. It fixes MPS/float32/sdpa, checks revision, uses package AudioResult identity and named methods, and creates/consumes clone prompts in that owner. There is no arbitrary invoke/design route or process API. Close drops its model reference; it supplies no timeout enforcement or process-death evidence. Scalar/text/token controls and output format/finite/frame/byte validation remain bounded through this backend and the existing encoder.
2. `worker/native_child.py:28–70` exposes a guarded inference function, not an executable child entry. It checks guard presence, fresh native hardware and 30-second/1-MiB descriptor ceilings before sealed input verification and the fixed backend. Python subprocess, multiprocessing start, fork/forkpty, posix_spawn, system and exec routes are patched to refuse, with audit checks too. This is accidental-spawn refusal for trusted immutable runtime code, not a hostile native-extension/direct-syscall sandbox. The inert test mocks audit-hook registration, directly checks the hook, and restores hooks; it does not install a permanent audit hook in the test owner.
3. `worker/native_protocol.py` implements separate codecs: closed request/body keys and operation names, bounded identity and integer checks (excluding bool), canonical bounded base64, 65536-byte framing, duplicate/nonfinite/noncanonical refusal, explicit numeric bind checks, TLS-1.3 context limits and exact peer-certificate SHA256 pin checks before framed I/O. Request validation is separate from generic send/receive. Prepare currently checks four-element snapshot shape and size only; Authority relations, registration, session freshness, allocation state and blob authorization still require future handlers. No handler/client exists, so these primitives are not a complete authenticated bridge.
4. Receipt validation binds the entire expected immutable/deployment tuple, rejects Linux reap-method substitution, requires receiver closure/reap booleans and bounded owned-process shape, and requires one zero-exit model for timely success. Empty processes are permitted for an unstarted stopped allocation with timely false, as r277 specifies. Empty data is not evidence of completed inference. This pure validator cannot prove actual waits or establish whether an allocation was unstarted; the missing supervisor must supply that authority. It remains disconnected from service receipt consumers.
5. `current_native_hardware` requires typed Qwen execution, preset/clone, exact MPS/float32/sdpa, Darwin arm64/aarch64/native, MPS index zero, supported/available true and timestamp age 0..300 seconds. Linux current_hardware and historical decoding are unchanged. This validates supplied inventory; it performs no fresh hardware probe or enrollment and is not wired into admission/start.
6. Archive membership includes all three new production modules. Installed Linux policy and activation remain unchanged. `lib/tts.py` is unchanged, SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; the historical broader legacy digest RED is not waived.

r278 describes these limits honestly. The absence of supervisor/runtime integration is an explicit partial-scope boundary here, not a newly discovered defect or a completed Lot A obligation.

## Independently executed checks

Inspected `tests/conftest.py` and `tests/service/conftest.py` before execution. The former installs a pytest maintenance guard; the latter can launch a real uvicorn listener. Neither was collected or executed. Used direct unittest selection with `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd, and explicit sys.path entries for the candidate root and candidate `tests/service`. Three separate interpreter invocations, no pytest discovery:

- `tests.worker.test_native_protocol` plus `tests.worker.test_native_qwen.NativeQwenTests`: **9 PASS**, 0.119 seconds reported by unittest.
- Fresh interpreter, `tests.worker.test_qwen_import_seam`: **5 PASS**, 0.445 seconds. Root/legacy/extracted archive and native complete-loader fixtures use injected upstream ML; native import fixtures inject NumPy too. This is loader/import identity evidence, not dependency/ABI compatibility.
- Targeted regressions: **44 PASS**, 16.148 seconds. Selected `tests.worker.test_engine_qwen`; `tests.service.test_qwen_lifecycle.QwenLifecycleTests`; `tests.service.test_pocket_lifecycle.PocketLifecycleTests`; `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`; `tests.service.test_production_reconciliation`; six `tests.worker.test_engine_kokoro.EngineTests` methods (`test_exact_arguments_and_detached_identity`, `test_closed_descriptor_and_unsupported_controls`, `test_limits_close_backend`, `test_mutated_invalid_audio`, `test_mutated_numeric_metadata_refuses`, `test_bundle_binding_before_backend`); and `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`. Explicit service path avoids r278's sibling-import error. Process-launching Kokoro lazy-import and release executable tests were excluded.

`git diff --check 326c4d28..HEAD`: PASS. No test failures or reruns. Tests use disposable tiny fixtures and their ordinary fixture teardown/archive extraction; no operational cleanup was performed. No actual ML, inference process, listener/network, Docker, acquisition/download, installation, hardware probe, HELD action, delegation, candidate edits or commit occurred. Printed device/load names are injected fixture output, not Metal observations.

## Parent-owned remaining gates

Lot A stays NO-GO until the coherent r277 vertical slice exists and receives fresh independent acceptance review: native supervisor/journal/durable launch fence/watchdog/exact handle wait/receiver closure; bounded transfer/spool/quota/deadline and replay/cancel/restart behavior; NativeRuntime and exact local output import; frozen native-lane decoding, trusted registration and admission; shared lane-selected live/detached receipt validation and existing Dispatcher/Core publication/reservation/ACK integration. No source activation is opened by this review.

Lot B package/enrollment/dependency closure and Lot C actual Apple Silicon probe/load/generate/cancel/restart remain separate. Clone conditioning at arbitrary cancellation remains unresolved. CPU Docker and CUDA qualification remain mandatory separate release rows. **GO means only that this partial foundation may be retained as source for completing those gates.**
