# Independent legacy control reconciliation review — r337 Sol

2026-10-25. **Bounded SOURCE GO for parent adoption of r336. The legacy loader digest RED can close after adoption of this exact candidate. No material finding requires correction. Release remains NO-GO; no self-adoption occurred.**

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-ee0932bc`, branch `t3/legacy-control-provenance`, HEAD `f3fc55087a23da29c16642b528b2d60f266e4a2e`, merge-base with supplied base exactly `b9f2b6b53e5efb7c8919ecd4e97cffcb732fd5a9`. Candidate status was clean before and after inspection/execution; HEAD remained unchanged. No applicable AGENTS.md was found in candidate discovery or checked ancestors/root destination. A bounded memory registry search found no relevant entry and supplied no technical premise.

Read FULL r336, r334 including its final legacy commission, and prior independent r262; inspected the actual ledger/test diff, existing compatibility test, generator, loader and backend import/dispatch changes, environment/device/dtype functions and current legacy loader call sites. Sole lasting write is this root report. No candidate edits, commit/merge/push, delegation, adoption, operational cleanup or parallel job-control candidate access.

## Reconciliation and findings

The candidate diff from the supplied base contains exactly three paths: modified `contracts/v1/legacy-controls.json`, new `tests/contracts/test_legacy_controls.py`, and new r336 report. No loader, Redis, CLI, schema, generator or public route change is introduced by this candidate.

Independently hashed historical loader bytes: both `4fbd770cf0f15f7c465adb5470625a68b5ff2cf3` and `0a4e53ad9237a5b19617344063411234e1a88888` produce recorded SHA256 `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`; their loader diff is empty. Loader history from the latter to HEAD contains exactly `de5c84d0ae3a2a9d7f80cbe928f08fe37061ad12` and `1a79358c04ecef5806c7c7617b74027173bf8af3`. Inspected both actual loader/backend diffs. Loader diff from 1a79358 to current HEAD is empty. Current loader SHA256 is `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

The ledger update is justified: only that digest and four direct getenv locations change (177→182, 206→211, 718→745, 487→492). Expressions remain respectively `os.getenv(name)`, `os.getenv(QWEN3_CUSTOMVOICE_FLAG_ENV)`, `os.getenv('TTS_MODEL_REVISION')`, and `os.getenv('TORCH_DEVICE', '')`. Constants/control values and coverage labels are unchanged. The generator creates synthetic fixtures/media, not this ledger; it was inspected, never executed. No blind regeneration, waiver or suppression was found.

The complete historical loader diff preserves constants, profile presets/defaults, token budgets, checkpoint identities, generation/provenance helpers, environment parsing, get_device and best_dtype. Omitted new device/attention controls retain the original five backend settings. Revision environment precedence and TTS_OFFLINE/HF_HUB_OFFLINE fallback remain; explicit revision/offline override remains. Legacy automatic device policy and CUDA-unavailable CPU fallback remain; automatic attention is flash on CUDA when available, otherwise SDPA, and bare cuda still maps to cuda:0. New explicit controls are additive, require concrete supported configuration/offline directory/revision, and do not change legacy defaults. Thread setup now precedes upstream import: an intentional initialization-order change, not byte-for-byte behavioral identity in all failure scenarios.

Package-root imports select relative backend/contract imports, legacy lib-path imports retain absolute imports; worker and clone binder select the matching family. No sys.path rewrite or import retry appears. Families remain distinct module/class identities, not interchangeable DTOs. r262 independently reviewed the corrected seam; its historical execution receipt is context, not execution repeated here. The added private service-clone dispatch leaves ordinary legacy getattr routing intact. Current legacy CLI loader calls omit the new controls. Redis jobqueue source is absent from the baseline-to-current lib change list. Other historical CLI/lib changes are outside this narrow reconciliation; the exact current command/shared-source/launcher ledger hashes pass below, not a claim of historical whole-product byte identity.

## Independent single-run receipt

Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B`, verified **Python 3.14.7**, Clang 17.0.0; no Python 3.11 execution. Inspected `tests/conftest.py` maintenance guard and `tests/service/conftest.py` listener/subprocess fixture; neither executed. No pytest collection occurred.

One execution invocation loaded the inspected new file with `runpy.run_path` under a non-main name, selected its three unittest cases, then AST-extracted the exact existing function `test_exact_text_ids_defaults_and_source_control_coverage` from its unchanged source into explicit standard-library globals and its equivalent JSON fixture loader. Wrapped that unmodified function body in `unittest.FunctionTestCase`. No neighboring test, module-level pytest/jsonschema/contracts import, subprocess or generator ran.

**4 tests PASS, first and only run, 0.156 seconds:**

- `test_legacy_automatic_device_attention_and_snapshot_forwarding`
- `test_legacy_forwarding_defaults_and_environment`
- `test_recorded_constants_and_environment_locations`
- `test_exact_text_ids_defaults_and_source_control_coverage`

The first three execute AST-extracted real loader function bodies with fake Torch/backend/upstream/snapshot objects. They assert actual recorded constant expressions and exact direct getenv location/expression set, unchanged omitted-control forwarding and environment fallback/override, CPU/MPS/CUDA automatic attention/device mapping, thread call and snapshot/revision forwarding. The existing exact test verifies current command/shared-source/launcher hashes, command control counts and fixture text/IDs. `git diff --check b9f2b6b HEAD` also passed.

Verified SHA256: ledger `716654429e5c77db6dd2ea0a6c5e62c75c811ae19682154168d57cd86783e947`; new test `b9a05f0ab8f4e514c36f91ea9112bdb98bfb07f24b19743b1b82b207a56b5a94`; unchanged generator `cd3010036de67d7168d30e035f7d8733541bbd0cf9eac16863983eec978d72f1`; unchanged existing test file `f22e13f44cc49e619aee17e417dd4607df2943f0863a68047afb81bb3b38e816`.

## Meaningfulness, limits and parent gates

These tests provide useful regression assertions beyond merely accepting a new hash. Limits: recorded constants are compared against ledger expressions rather than an independent frozen historical oracle; annotated profile dictionaries and real environment boolean parsing/device/dtype resolution are not comprehensively executed by these three tests. Their preservation is supported by the inspected historical diff. Fake MPS is a supplied device result, not automatic MPS discovery. Exact command coverage counts controls rather than proving every CLI expression/behavior. AST extraction does not exercise real module import families or spawned owners; r262 remains prior evidence for that seam. Thread-call presence is tested, not full real dependency initialization ordering.

No actual Torch/Qwen/ML import, inference, process/listener, network, Docker, install/download, crash, HELD, STT or cleanup action occurred. Only tiny inert work was authorized/executed; the below-6-GiB heavy-work gate is unchanged and no live capacity claim is made.

Parent may adopt this exact reconciliation and then close the stale legacy-loader digest RED. Parent owns adoption/integration; this report does neither. The bounded compatibility conclusion supplies no actual dependency availability, installed CPU/CUDA/Metal, lifecycle/recovery/security, resource/performance, acoustic or release qualification. r334's broader product and release gates remain open. No review of the separate unmerged job-control candidate is implied.
