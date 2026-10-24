# Native worker source implementation — r278 Sol

2026-10-25. **PARTIAL / NO-GO for Lot A source acceptance.** This change does
not implement the commissioned coherent native vertical slice. Native installed
activation remains closed. No native installation, Metal inference, speech,
performance, containment, cancellation or restart qualification is claimed.

## Exact source basis and authority

- Worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-05c63b46`.
- Root: `e0c5831ecd31d5171647a6101e99285d46e794c3`.
- Entry HEAD / r277 documentation commit:
  `326c4d28ecbb935b96cb3b50e35342b076a66ac4`, whose parent is exactly that root.
- Normative r277 file read in full before edits:
  `docs/agent-interface/reviews/H4-NATIVE-WORKER-CONTRACT-r277-sol.md`;
  SHA256 `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`.
- Partial source commit: `49dbb509d24040c0d446a5b66ea519962d4f9900`.
- No applicable ancestor/repository AGENTS.md found. Both test conftests were
  inspected. Tests used direct unittest loading, never pytest collection.
- No delegation, listener, network, Docker, ML, acquisition, installation,
  native subprocess, actual UID/kernel/hardware testing, crash experiment,
  HELD action or operational cleanup was performed.
- Entry free space was 3,492,900 KiB (about 3.33 GiB), below the unchanged 6 GiB
  actual-work gate. Only small inert fixtures/archive copies were used; no
  actual-work admission or gate reduction was attempted.

## Implemented source, with narrow limits

`lib/tts_backends/qwen_native.py` adds an in-process owner that uses the complete
existing `_load_tts_model_local`, package `AudioResult`, explicit local
MPS/float32/sdpa/offline/revision settings, and thread/offline/tokenizer
environment before ML imports. It exposes only preset and clone synthesis.
Clone prompt construction and consumption stay in the same model owner; close
only drops the model reference. The legacy QwenBackend and lib/tts.py are
unchanged. This backend is not registered in installed discovery.

`worker/native_child.py` adds the real inference function using the existing
sealed input verification, loader and WAV encoder. It requires the native spawn
guard and a fresh native inventory. Its current bounded source subset refuses
descriptors with timeout above 30 seconds or max_output_bytes above 1 MiB.
It does not yet implement inherited-socket input/output, token authentication,
exec entry, receiver lifetime or owned-process terminal authority. The permanent
guard hooks ordinary Python subprocess, multiprocessing start, fork, spawn,
system and exec APIs, with audit hooks as an additional check. It is not a
malicious native-extension sandbox. There is no executable native-child CLI.

`worker/native_protocol.py` adds closed request/body validation, bounded
canonical JSON framing, duplicate/nonfinite refusal, canonical bounded base64,
numeric explicit bind validation, TLS 1.3 peer-pin checks, TLS context
construction and a native-only receipt shape/full expected-binding validator.
These are primitives, not a complete protocol implementation: response value
validation, session/registration enforcement and operation state belong to the
still-missing supervisor/client. No handler currently consumes these codecs.
The receipt validator is deliberately not wired into service consumers and
does not select its own lane from a received receipt. Credential ownership and
no-follow enrollment verification are not implemented by tls_context.

`worker/engines/qwen_execution.py` adds current_native_hardware for Darwin
arm64/aarch64/native, MPS index zero, supported/available and age 0..300 seconds.
Existing Linux current_hardware and historical inventory decoding are unchanged.
Resolution/admission/prepare/start integration is still missing.

`headless-dist-manifest.json` explicitly includes the three new production
modules above. Its Linux platform declaration/policy is unchanged; inclusion
does not constitute native packaging. Tests extend the actual archive/import
fixture and add two focused files, rather than an alternative package harness.

## Inert validation actually executed

Interpreter: `/private/tmp/pab-0c-venv/bin/python -I -B`, Python **3.14.7**.
The launcher symlink points to the existing managed interpreter; no environment
was installed or changed. Each invocation explicitly inserted the checkout
root for unittest loading. No discovery/full-suite invocation was used.

1. `tests.worker.test_native_protocol` and
   `tests.worker.test_native_qwen.NativeQwenTests`: **9 PASS**. The same two
   modules were rerun after removing accidental imported-TestCase collection:
   **9 PASS**. Fake TLS/sockets only. Tests enter the actual inference function,
   sealed tiny model metadata, complete loader with fake Torch/Qwen/hub, actual
   encoder and WAV bytes. They exercise exact loader forwarding, guard refusal
   for an injected upstream subprocess attempt, native freshness and preserved
   Linux MPS refusal. This is not Authority-to-Core execution.
2. `tests.worker.test_qwen_import_seam`: final **5 PASS**, including legacy
   root/lib-path import tests and native full-loader root/extracted-archive
   tests. An initial run had 4 PASS / 1 ERROR: the new fixture loaded real NumPy
   inside patch.dict(sys.modules), which removed it on exit and made the second
   CPython 3.14 C-extension import fail. The new fixture now injects fake NumPy
   too; the complete lib/tts loader remains exercised. Production code was not
   changed to conceal the fixture problem.
3. Targeted regression invocation: **41 PASS / 1 import ERROR**. Selected names:
   `tests.worker.test_engine_qwen`,
   `tests.service.test_qwen_lifecycle.QwenLifecycleTests`,
   `tests.service.test_pocket_lifecycle.PocketLifecycleTests`,
   `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`,
   `tests.service.test_production_reconciliation`, and six Kokoro EngineTests:
   `test_exact_arguments_and_detached_identity`,
   `test_closed_descriptor_and_unsupported_controls`,
   `test_limits_close_backend`, `test_mutated_invalid_audio`,
   `test_mutated_numeric_metadata_refuses`,
   `test_bundle_binding_before_backend`.
   The reconciliation module expects sibling `test_production_api` on its import
   path. Rerunning that module with both `.` and `tests/service` explicitly on
   sys.path gave **2 PASS**. The other 41 tests passed in the original run.
   The subprocess-based Kokoro lazy-import test and release executable tests
   were not selected.
4. `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`:
   **1 PASS**. Archive/extracted source-byte closure is also exercised in (2).
5. `git diff --check`: PASS before source commit.

These counts are bounded source results, not performance evidence. Existing
25-second Production / 30-second supervisor budgets were not changed. Printed
MPS loader names in injected tests are not hardware observations.

`lib/tts.py` is byte-unchanged against the root, SHA256
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
The historical legacy digest RED named in r277 is neither waived nor
regenerated; its broader check was not rerun here.

## Outstanding Lot A source work — all required before acceptance

There is no NativeSupervisor, closed journal, durable launch-intent fence,
independent watchdog, exact handle wait/receiver closure, process identity lock,
bounded transfer spool/quota, cancellation escalation, terminal persistence,
uncertain restart block, or live listener entry. `worker/native_supervisor.py`
has not been created. No empty supervisor stub stands in for these requirements.

There is no NativeRuntime RPC adapter, detached immutable transfer handoff,
private control spool, exact output import/fsync/PCM/hash check, trusted local
control import UID, or durable outcome/reservation/ACK integration.
`service/native_runtime.py` has not been created. No empty adapter is claimed
as implementation.

The common runtime_protocol/Qwen relation extraction, qwen-mps frozen-lane
decoding, NativeQwenContexts registration, native admission demand and
Production/DispatchStore handoff remain unimplemented. Dispatcher,
Reconciliation, recovery and other receipt consumers remain Linux-only. Their
Linux PID1/auth/UID/reap rules were not weakened. The required shared pure
finite receipt adapter and old-Linux-substitution negative tests across live
and detached consumers are still needed.

Acceptance 1 (Authority→native handlers→child→receipt→NativeRuntime→Dispatcher/
Core artifact→same-fixture restart/replay) is **NOT IMPLEMENTED**. Acceptance
2–3 have primitive coverage only, with no service-level allocation/transfer
proof. Acceptance 4–6 (replay, cancel, launch/wait/receiver/journal failures,
unknown restarts and quotas) are **NOT IMPLEMENTED**. Acceptance 7 has complete
loader/root/archive, private prompt and Python spawn-refusal coverage, but
not the actual inherited-channel child entry or qualified dependency closure.

The next implementation must complete these existing r277 obligations before a
parent fresh Sol acceptance review can yield GO. This report is not an
independent review and does not authorize activation. Lot B native packaging,
enrollment and dependency closure, then actual Lot C Apple Silicon Metal
probe/load/generate/cancel/restart remain required. Clone conditioning at
arbitrary cancellation points remains unresolved. CPU Docker and CUDA release
qualification remain separate. No UI/STT/public schema/DDL/queue/registry or
installed-policy change was made.
