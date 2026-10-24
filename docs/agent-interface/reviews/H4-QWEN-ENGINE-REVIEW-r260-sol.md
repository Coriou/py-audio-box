# Independent Qwen explicit engine review — r260

2026-10-24. **Bounded SOURCE NO-GO for adoption: one P1 startup integration defect.** The supplied 11 tests pass, but the new adapter cannot reach its loader under the repository-root-only import environment used by headless workers. No actual engine, hardware, acoustic, performance or release qualification is conveyed.

## Pins and scope

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-f6c979da`, branch `t3/qwen-explicit-device-adapter`, clean before and after inspection/checks. Verified exact ancestry: base `c6c443dadde339f47ca3255f2467111e83cfc187` → source `4ec18019b133c6e0a9b8d164417a0ff29531f9d5` → report HEAD `c056daefc5b739f363fe5c828a313e393d2913d8`. The source delta is exactly six files: `headless-dist-manifest.json`, `lib/tts.py`, `lib/tts_backends/qwen.py`, `tests/worker/test_engine_qwen.py`, `worker/engines/qwen.py`, `worker/engines/qwen_execution.py`. Report HEAD adds only r259.

Read the complete r259 handoff, r188 including assignment 4, r172/r177, the actual six-file delta, complete new adapter/descriptor/tests and existing Qwen backend, loader/device/dtype/clone bindings, asset verifier, checkpoint capabilities, hardware inventory, Pocket reference validation, relevant callers and release archive tests. No applicable AGENTS.md was found in candidate/report ancestry. A small memory-registry lookup located the prior r177 review; current candidate documents and source independently supplied all conclusions.

Sole lasting write is this root report. No source changes, commits, delegation, operational cleanup or HELD action.

## P1 — the new package import path cannot start the existing model worker

Affected new entry: `worker/engines/qwen.py:88-98`; failing inherited import: `lib/tts_backends/qwen.py:86`. The adapter selects `lib.tts_backends.qwen.QwenBackend`, which is importable with only the repository/package root on `sys.path`. Its worker then unconditionally does `from tts import _load_tts_model_local`. There is no root `tts.py`; it resides under `lib`. Neither this adapter nor the backend establishes the legacy `lib` import path. Spawn inherits the parent's import path, so it does not repair this. Existing CLI callers explicitly insert `lib`; the headless launchers insert only the package root (`service/Dockerfile.release:13,28`, `service/package_install.py:39`).

Reproduced in the authorized isolated interpreter, without spawning, ML import or model construction:

```python
import sys
sys.path.insert(0, '/Users/ben/.txcode/worktrees/py-audio-box/t3-f6c979da')
from lib.tts_backends.qwen import _worker_impl
class Connection:
    def send(self, message): print(repr(message))
    def close(self): pass
_worker_impl(Connection(), dict(model_name='/inert', num_threads=1,
    dtype_str='float32', revision='r', offline=True,
    device='cpu', attention='sdpa'))
assert 'torch' not in sys.modules
```

Observed exactly `('error', ('ModuleNotFoundError', "No module named 'tts'"))`. This occurs before model-path inspection or inference; a valid verified bundle cannot fix module resolution. The real backend would surface startup failure instead of producing audio on every device. The defect is exposed by the new service entry, although the inherited legacy worker import itself is unchanged.

Fix the package/legacy loader import seam deliberately while preserving legacy callers and lazy ML loading. Simply switching to `lib.tts` also needs its absolute `tts_backends` imports addressed. Add a regression that enters the actual backend worker/complete loader with only the package root visible, including an extracted-archive case. Inject upstream/Torch and transport in-process; actual hardware or process spawning is unnecessary for this source fix. The current AST tests replace the loader's module globals, and engine tests replace the entire backend, so both bypass this defect. Existing archive checks cover Pocket and package imports, not Qwen worker startup.

## Other reviewed behavior and bounded observations

The frozen descriptor rejects auto/unindexed device selection, unsupported precision/attention, seed, bool positive budgets, unsupported task controls and malformed clone bindings. Explicit loader calls require device, concrete dtype, attention, local directory, revision and offline mode together; partial None arguments refuse rather than enter automatic policy. Runtime CUDA index/availability and native Darwin arm64 MPS availability are rechecked. Legacy calls without explicit arguments retain automatic policy. The real `best_dtype` returns the selected concrete dtype unchanged. Thread forwarding and exact attention forwarding are present.

Bundle verification authenticates raw manifest bytes, identity, listed sizes and streamed hashes before backend construction. It rejects duplicate/unsafe listed paths, symlink components, unlisted files and unsupported suffixes, with per-file/total/count limits. Execution sample rate and selected language/preset/device tuple must appear in operator metadata. The parser checks selection membership rather than fully normalizing every unused capability entry; those entries remain trusted operator declarations. This review does not infer upstream capability truth, semantic tensor validity, or hostile-operator isolation from those hashes. `verified_bundle` itself accepts only `fr`/`en`, whereas r259's Qwen fixture uses `English`; the new verifier is a separate Qwen contract, so the handoff's compatibility wording must not be taken as proof that existing provisioning can activate it. Provisioning was explicitly deferred.

Checkpoint mode and instruction permission reuse the local hashed config's existing capability logic. The three service methods select named generation methods; clone prompt construction and consumption stay in the model owner over existing private IPC. No public pickle or caller-selected method was added. Reference metadata binds exact voice/revision/model/revision/transcript; complete authenticated bytes pass Pocket's RIFF walker (exact container length, fmt/data ordering and uniqueness, PCM16 mono/rate/alignment/frame count, chunk/padding bounds). The temporary reference snapshot requires an executing-owner 0700 namespace. Ordinary output revalidates exact model/revision metadata, rate/channels, finite real mono samples, amplitude, measured frames/bytes and PCM16 encoding. Close runs after construction on success/error and close failure prevents return. The actual Qwen constructor is lazy and has no spawned resource before inference; no new constructor-failure leak was established.

No lifecycle/Core/provisioning/installation activation changes are in the delta. Existing spawn/timeout/close/failed-reap behavior is retained, not newly qualified. No additional material current source finding was established within the authorized scope.

## Executed checks

All commands used `/private/tmp/pab-0c-venv/bin/python -I -B` and explicit candidate paths; only small temporary inert fixtures and injected ML were used. No pytest collection or maintenance/listener fixtures ran.

- Supplied `tests/worker/test_engine_qwen.py -v`: **11 PASS**.
- Direct unittest loading of `test_engine_pocket.py` plus three selected `ArchiveTests` (deterministic extracted help, Pocket archive closure, manifest files): **18 PASS**.
- Direct invocation of eight existing `test_tts_runtime_contract.py` cases with fake Torch and real candidate modules: checkpoint matrix, lazy identity, two audio-contract cases, atexit ownership, local config capabilities, worker output release, legacy auto-device environment: **8 PASS**.
- Direct invocation of existing fake-only Qwen diagnostics cases: send/poll/recv races and construct/start failures: **5 PASS**. Actual crash/kill/restart process tests were inspected and intentionally not run.
- Full `tts` module import with injected Torch/upstream, real loader and real `best_dtype`: explicit CPU eager/float32 forwarded; partial device/attention, None dtype and unavailable CUDA refused: **PASS**. This explicitly adds the legacy `lib` path to exercise loader behavior and therefore does not close P1.
- Actual `_worker_impl` with package-root-only path and inert connection: reproduced P1 above, before Torch import.
- `git diff --check c6c443d 4ec1801`: **PASS**. Candidate HEAD/clean status unchanged.

No actual ML, hardware probe, Docker, network, download, install, listener, process-crash operation, broad test campaign or resource cleanup was performed. Passing injected tests establish source behavior only.

## Parent handoff

Require P1 correction and independent focused review before source adoption. Parent integration still owns immutable voice/settings/profile authority, resource admission, truthful sealed bundle/probe records and invalidation, and result/publication fencing. The existing spawned model child and multiprocessing helper descendants must be included in startup/timeout/cancel/worker-death/restart ownership; capacity and reference namespace cannot be released before confirmed reap. Native transport/package ownership remains separate.

Actual constrained CPU, CUDA and native Apple Silicon Metal install/generate/cancel/restart, resource/performance measurements, audio listening and integrated release acceptance remain unrun, separately authorized gates. This source NO-GO requests a concrete import-seam fix, not a hardware campaign or lifecycle redesign.
