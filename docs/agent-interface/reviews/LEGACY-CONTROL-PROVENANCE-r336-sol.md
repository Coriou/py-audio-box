# Legacy control provenance reconciliation — r336 Sol

2026-10-25. Implementation candidate: the stale loader digest is explained by reviewed additive execution controls and import compatibility changes. No legacy loader/default regression established in this bounded SOURCE/inert assessment. **Digest RED closure remains NO-GO pending fresh independent Sol review; no self-adoption or release qualification.**

## Authority and exact pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-ee0932bc`, branch `t3/legacy-control-provenance`, clean entry HEAD `b9f2b6b53e5efb7c8919ecd4e97cffcb732fd5a9`. User authorized the small separate reconciliation from the final section of HEADLESS-INTEGRATED-MILESTONE-r334-astra.md. Read that report, actual ledger, fixture generator, compatibility test, loader, backend import/dispatch diff, legacy callers, r259 and independent r262. No applicable repository/checked ancestor AGENTS.md found. A memory registry search supplied no technical premise.

Only writes: `contracts/v1/legacy-controls.json`, new `tests/contracts/test_legacy_controls.py`, this report. No schema/generator/public route, loader, Redis or CLI source edits; no root writes, merge, push, delegation or adoption. No ML imports, application processes, network/listener, Docker, acquisition/install/download, crash, operational cleanup, HELD or STT action. Heavy gate remains unchanged; tiny in-memory inert tests only.

## Source chain and behavior

The recorded SHA256 `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd` exactly matches `git show 4fbd770cf0f15f7c465adb5470625a68b5ff2cf3:lib/tts.py` (initial contract pack) and `git show 0a4e53ad9237a5b19617344063411234e1a88888:lib/tts.py` (last loader revision before the two changes). This is an identifiable source baseline, not a guessed hash.

`git log 0a4e53ad..HEAD -- lib/tts.py` contains exactly:

1. `de5c84d0ae3a2a9d7f80cbe928f08fe37061ad12`: explicit local Qwen execution adapter. Appends optional `device=None, attention=None` to private local loader and keyword-only public loader controls. Explicit selection requires a concrete supported device, dtype, attention, offline local directory and revision, and refuses unavailable/incompatible hardware. Omitted controls retain get_device/best_dtype and automatic flash-attention/SDPA selection. Forwarder keeps the original five backend settings when both controls are omitted. Thread setup moves before upstream Qwen import; this is intentional initialization ordering, not a changed legacy control/default.
2. `1a79358c04ecef5806c7c7617b74027173bf8af3`: selects relative backend/contract imports for package imports and original absolute imports for legacy lib-path imports. Corresponding backend worker/binder selects the same import family. No sys.path rewriting or catch-and-retry import fallback. Package and legacy class families remain distinct, not promised interchangeable.

Current lib/tts.py is byte-identical to `1a79358` and has SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Full baseline-to-current loader diff contains only those changes. r262 independently reviewed the corrected chain and its package/legacy import seam; original source `4ec18019b133c6e0a9b8d164417a0ff29531f9d5` was cherry-picked as de5c84d, followed by the import correction. r262's historical mocked receipts are provenance context, not new execution here.

All recorded constants, generation profile defaults, token budgets, checkpoint identities, synthesis/provenance functions, get_device and best_dtype are unchanged by this loader diff. TORCH_DEVICE auto/override/fallback, dtype auto policy, TTS_MODEL_REVISION precedence and TTS_OFFLINE/HF_HUB_OFFLINE fallback remain. CUDA legacy `cuda` maps to `cuda:0`; automatic attention still chooses flash only on CUDA with availability, otherwise SDPA. Existing CLI call sites omit new execution controls. Backend's added `_service_clone` dispatch is a separate named service route; prior ordinary methods retain their getattr route. Redis source is unchanged across this baseline interval. Other historical CLI/launcher edits exist and are not claimed byte-unchanged since 0a4e53ad; their current recorded digests/control counts pass the exact check below. This reconciliation introduces no changes to them.

## Justified ledger changes

Changed only lib/tts.py digest and its four direct `os.getenv` source locations: 177→182 (generic flag), 206→211 (CustomVoice flag), 718→745 (model revision), 487→492 (device override). Expressions are identical. No constants or control values added/removed/relabelled. `coverage=unchanged_legacy` denotes retained legacy controls/default behavior, not byte identity to the historical revision; the new digest pins the reviewed current bytes. Optional explicit service controls do not become new legacy defaults.

`contracts/generate.py` generates synthetic fixtures/media and does not generate this ledger. It was inspected and not executed or modified. No blanket regeneration, waiver, expected-failure suppression or loader alteration occurred.

Candidate SHA256: ledger `716654429e5c77db6dd2ea0a6c5e62c75c811ae19682154168d57cd86783e947`; new test `b9a05f0ab8f4e514c36f91ea9112bdb98bfb07f24b19743b1b82b207a56b5a94`. Unchanged generator `cd3010036de67d7168d30e035f7d8733541bbd0cf9eac16863983eec978d72f1`; exact compatibility test file `f22e13f44cc49e619aee17e417dd4607df2943f0863a68047afb81bb3b38e816`.

## Fresh execution receipt

Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B`: verified **3.14.7**, Clang 17.0.0. Inspected root conftest (maintenance guard) and service conftest (listener/process fixture); no pytest collection/conftest executed.

- Direct `tests/contracts/test_legacy_controls.py -v`: **3 PASS**, first run, 0.040 seconds. Standard library only; AST-extracts two actual loader functions with fake backend/Torch/upstream/snapshot modules. Verifies every recorded constant and exact direct environment expression/location set; default/revision/offline forwarding; explicit existing revision/offline overrides; automatic CPU/MPS/CUDA device/attention and snapshot/revision forwarding. No actual loader module, Torch, Qwen, model, network or backend process imported/started. MPS is an inert supplied device result, not automatic MPS discovery or support qualification.
- Exact `test_exact_text_ids_defaults_and_source_control_coverage`: **1 PASS**, first invocation, 0.117 seconds. Executed via `runpy.run_path` with explicit worktree sys.path and `unittest.FunctionTestCase`, selecting only that inspected function from `tests/contracts/test_contracts.py`. Verifies all command/shared-source/launcher digests, command control counts and exact fixture text/IDs. Does not invoke the neighboring determinism subprocess/generator test.
- `git diff --check`: PASS.

The exact existing check alone counts command controls rather than comparing all expressions. The new test closes the relevant loader constant/environment/default assertions within this commission; it is not exhaustive CLI behavior, real dependency import, process lifecycle, hardware or acoustic testing. Independent Sol must review the exact committed candidate before parent considers digest RED closed. Installed CPU/CUDA/Metal, heavy work and broader release gates remain unchanged.
