# H4 Kokoro engine interface — r189 Sol implementation

2026-10-24. Engine-owner source implementation complete; independent Sol review next. No self-adoption, real-engine activation, installed readiness, inference, acoustic or containment claim.

## Pins and scope

Own fresh assigned checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-6aeda740`, branch `t3/kokoro-engine-slice`, clean entry HEAD `cdbd3c786c5cc25a681391d2a5501489d4bd0b9d`. This exactly matches r188's source/entry pin. Root had advanced through commissioning documentation to `e29d073`; no source was copied from it. Read root `HEADLESS-MILESTONE-r188-astra.md` in full, backend contract/assets/Kokoro, child/protocol/supervisor, synthetic exchange and root test conftest. No AGENTS.md found. Memory registry search supplied no relevant premise.

Source commit: `85eb3d41ce3bea6642ef680c7aa277eb4d6bde09`.

Only four source files added: `worker/engines/__init__.py`, `execution.py`, `kokoro.py`, `tests/worker/test_engine_kokoro.py`. Existing backend and runtime child remain unchanged. No concrete legacy adapter defect was established. No service, protocol, supervisor, DB, provisioning or installation edits.

## Interface and parent handoff

`Execution` is a frozen typed value. `from_wire` requires every declared field and refuses extra fields, including clone/reference, instruction, seed and filesystem path. Its detached `identity()` returns a new scalar dictionary. Fields are bundle_id, manifest_sha256, checkpoint, revision, preset, language, threads, max_text_bytes, max_frames, max_output_bytes, memory_bytes, device, engine, sample_rate, channels, media_format, media_subtype. Only Kokoro/CPU/24000 Hz/mono/WAV/PCM_16 is accepted; language is fr or en, with matching preset prefix. Numeric budgets reject bool and have finite ceilings. These are interface ceilings, not qualified profile values.

`execute(execution, text, bundles, *, backend_factory=None)` accepts the typed descriptor and exact authored text. `bundles` MUST be an integration-owned authorized mapping of managed bundle IDs to immutable local directories. It must never be constructed from public paths, ambient discovery or untrusted request JSON. Descriptor and mapping convey no authority themselves. The parent confirmed ownership of subsequently binding both through Authority and the authenticated lifecycle snapshot. There is currently no model-bundle mapping in that snapshot; activation remains unwired, synthetic protocol/default unchanged. No alternate unauthenticated activation route was added.

Parent must select a trusted provisioned manifest digest, checkpoint and revision, freeze them with the owned preset/profile and job input, ensure preset registration, retain immutable bundle contents for the full attempt, authenticate the descriptor/mapping and reserve/enforce its thread/memory/output budgets before launch. Descriptor limits cannot be enlarged by public requests. The engine bounds manifest reading to 65,537 bytes, compares exact SHA256, reuses `verified_bundle` to verify local file hashes, then checks checkpoint/revision/language before constructing the backend. Existing manifests without an explicit matching revision refuse; provisioning/integration must supply one. Manifest digest equality alone does not establish trusted provenance. Existing bundle verifier and immutable mapping assumptions remain the authority boundary; this slice does not add hostile-filesystem race protection.

Backend import is lazy and confined to the worker execute call. It reuses `KokoroBackend` and exact AudioResult, passing resolved local directory, explicit language/thread count and unchanged text/preset. No seed/reference/instruction controls exist in this interface. Legacy backend already demands CPUExecutionProvider only. No downloads, fallback device selection, service/discovery imports or second inference stack were introduced.

The result is `{data, frames, size_bytes, sha256, identity}`. `data` is measured mono PCM16 WAV bytes. It refuses wrong rate/channel/shape, empty/oversized/nonfinite/out-of-range audio, and checks actual serialized size. Validation/conversion runs in 4096-frame chunks after checking frame/byte limits; backend closes on success and post-construction failure. Result metadata is not trusted or forwarded. Publication must remain the existing authenticated result/manifest protocol and staging/publication barrier; the parent must adapt these bytes and detached identity there, never let the engine write DB or bypass fences.

Finite behavior: authored-text/output sizes and conversion scratch are bounded; one backend is created and closed per call. Legacy inference itself materializes an AudioResult before frame checking. `memory_bytes` is an admission/OS-enforcement contract, not a Python allocator or inference-memory guarantee. Parent must enforce it through the supervised runtime, retain cancellation/reap capacity fences, and qualify actual profile memory. No claim of hard inference-memory enforcement is made by this callable alone.

## Validation

Used only `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/worker -p test_engine_kokoro.py -v`: 6 methods PASS. First run had one test-only macOS resolved-path expectation failure; corrected expectation to canonical resolved directory, then all six passed. `git diff --check` passed before source commit. Root conftest inspected; unittest avoids pytest/autouse maintenance hooks. Test subprocess uses the same interpreter.

Injected backend verifies exact authored text/preset/language/threads, detached identity/wire roundtrip, unknown/unsupported descriptor controls and CPU/limit restrictions, hash/checkpoint/revision/unknown-bundle refusal before backend creation, modified bundle bytes, frame/byte/text limits, mutated invalid audio, backend close, and fresh imports with service/ML/numpy/network-library imports forbidden. No real backend was constructed. No actual inference/download/ML install/network/Docker/listener/Linux sandbox on macOS/HELD operations, delegation or historical campaign occurred.

Remaining: independent Sol source review; parent-owned profile/admission/Authority/supervisor/protocol/publication integration and managed immutable bundle mapping; separate authorized actual audible check and resource/cancel/restart qualification. This report is committed separately and copied alone to root. Parent adoption remains pending.
