# Native import uncertainty independent review — r295 Sol

2026-10-25. **Bounded GO for the inspected F11 correction; NO-GO for complete corrected native Lot A SOURCE acceptance from this review's evidence.** No new material source defect was established. Additional actual Dispatcher fault scenarios close the demonstrated private/stage sync replay and visible-COMPLETE uncertainty paths. However, this reviewer lost the continuation handle/result of the authorized one-shot 96-method invocation. Its terminal outcome and completed count are UNKNOWN, not PASS. It was not repeated. Parent adoption remains separate; native B/C, actual Metal and release remain CLOSED / UNQUALIFIED.

## Authority and exact pins

Review only. Sole lasting write is this report in the root checkout. No source edit, commit, delegation, pytest collection/maintenance/listener, actual model subprocess/signal/socket, ML/hardware, Docker/network/download/install, crash experiment, operational cleanup or HELD action. Successful disposable fixtures performed their existing teardown. Root pre-existing untracked STT evidence was listed only, not opened or changed.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-f76594f2`, branch `t3/latch-import-fsync-failure`, clean at entry and after executions:

- Parent/root: `313991689993dd550fd948d50231357f41838e3e`.
- Integration: `b29384938e7a1ea1412af98644564716dd529dae` (original `434e26747736e303e116bdde0b413282755b1f30`).
- r286: `54850fa472ac020c58dd0e2a821ee92509cdaa4f` (original `e187bf2061ec61a458aafee82d68472ef9097086`).
- F6–F8: `6d239c25d20748b84d29f9c85fc87c6a6e9d56f4` (original `9ed05f744ac3342dacb114e301e33fe4cfbc254b`).
- r288: `61ffd2f60bbcb32980f4f708c6ed62de4190ed03` (original `5ba3c66ace2c5853101424c44c73be863c924854`).
- F9: `f1b4eb6c7c41d496f97daf756662fa35152187f6` (original `f368f31c1fdd7c158f31f38ad311bcbf942957a6`).
- r290: `a10dc1a6486fea59c9c7a29d1ca0d74ee98b485c` (original `1277b8fe791ad1747934c1d7fb5fdbbd08234b09`).
- F10: `15b6d723a5d60325035743986dc534e7d4a1a625` (original `2a8b8dacb1d82fc44b7b1cba1f0897c72465ee5f`).
- r292: `f4ebb15e28a574cfbbe18d3433f0c20eabcc4c92` (original `1bbc63b887d2c941947c93615a690ea6f7d9d60e`).
- Correction SOURCE: **`afa4aaa6aff9778963fddb6d9f46aecd50e328e5`**.
- Reviewed docs HEAD: **`479af9e1532b3ac15ecab8bb887119d4001a9e54`**; source-to-HEAD delta is r294 documentation only.

Read full r277, r285 and r286–r294, recovering combined-output truncations with focused reads. Inspected complete NativeRuntime and correction delta, Dispatcher handoff/startup/terminal ordering, actual focused fixtures, Authority, Core verification/publication/storage, ResourceStore scans, native prepare/terminal paths and changed live/history/Linux policy consumers. No AGENTS.md found in candidate/adjacent worktree discovery. A bounded memory-registry keyword search returned no relevant hit; no memory-derived technical authority. Both conftests were read before execution: direct unittest/direct functions did not collect the maintenance autouse guard or invoke HTTP listener fixtures.

Available storage observed **3,200,164 KiB**, below 6 GiB. Only tiny inert fixtures/archive contents with existing injected capacity ran. Production 6 GiB-plus-demand and RAM admission gates were unchanged. Printed CPU/CUDA/MPS labels are synthetic declarations.

## F11 correction and restore proof

Material locations on reviewed HEAD:

- `service/native_runtime.py:141–202`: strict retained-file accounting, marker decoding and initially closed reopen state.
- `service/native_runtime.py:212–236`: runtime uncertainty query and finite owner-startup Core-intent restoration.
- `service/native_runtime.py:383–400,448–477`: collect refusal latch, completed-only existing-output replay and failed-finalization closure.
- `service/native_runtime.py:479–529`: held-stage verification, link/unlink/sync and COMPLETE-before-handoff ordering.
- `service/dispatcher.py:104–106,410–418,433–443,494–565`: startup restore, owner finalization, STOPPED quarantine, terminal refusal, Core outcome/reservation sequence.
- `service/core.py:491–578,630–645`: independent media/oracle verification, intent insertion/adoption, then success-state commit.

The old r293 identity=None shortcut cannot promote an uncertain first import: existing output requires membership in the completed set, and owner failure adds the binding to the closed set and sets runtime.failed. Collect refuses the closed binding before wire I/O; stale PendingImport finalization and ACK refuse it; active mutation RPCs refuse failed/closed state. Observation and stop retain actual supervisor accounting. Dispatcher may continue observing truthful STOPPED and attempting collect, but STOPPED no longer clears quarantine, and the terminal guard precedes receipt consumption/publication/release. This preserves evidence and executing reservations rather than declaring failed storage a successful first import.

INTENT contains version `native-import-v1`, Authority binding, runtime ID and authenticated output tuple. It is written through exclusive no-follow creation, file fsync and parent-directory fsync before private import creation/link. COMPLETE is created only after no-replace link, owner alias retirement, private-directory and held-stage-FD sync, cancellation/cutoff and stage/output checks. Its own file/directory sync must return before completed-set insertion and TransportObservation return. There is no rename/replace/temporary-marker retirement. `_write` has a generic exact-existing-byte branch, but the two marker callers explicitly refuse existing marker names; no ordinary marker retry follows uncertainty.

On reopen even valid INTENT+COMPLETE starts closed. A visible COMPLETE does not assert that its last sync returned. INTENT-only never qualifies for restoration. Markerless output history also starts closed and needs the same Core proof. `_account` authenticates snapshot binding, bounded output hash/PCM/tuple, marker version/runtime/output consistency, exact COMPLETE canonical bytes, fixed names, file mode0600/owner/type/nlink1, directory0700 and finite64-record/64MiB declared accounting. Partial output or a retained import also sets failed; restoration does not clear that flag. Marker metadata fits the existing96000-byte allowance. No unknown-name, ENOENT, ownership, hardlink or quota waiver was introduced.

`recover_imports` is called on the Core owner at startup, before executor creation. It requires exact stored Authority bytes, bound runtime (also equal to the new-format marker runtime), stored native receipt validated against frozen configuration/lane/original boot/binding/deadline, timely=true, and a Core publications artifact matching job/generation/attempt/owner/staging. It reads Core-owned state; no RPC, received assertion or disk marker alone supplies proof. Live failed sessions have no reopened candidates for that binding and cannot repair themselves through the hook. Its source assumption is the trusted sole-writer/current-source history, not protection against arbitrary in-process code or same-UID database forgery.

The ordering premise is real and narrower than “job already succeeded.” Dispatcher finalizes the import before `_terminal`; `_terminal` validates/stores the receipt before invoking Core._publish. Core independently verifies media/hash/provenance/Authority before inserting publications and publication_work in one transaction (`core.py:566–570`), then syncs root, adopts files and only later commits succeeded state. SQLite uses WAL/synchronous=FULL (`:231–232`) and transaction COMMIT (`:288–305`). Thus a durable Core intent can exist while the job remains running, but in this source path it follows successful import handoff. It is not evidence that all subsequent adoption/state steps succeeded. Normal Core publication recovery still validates its work/manifest/artifact/media and may refuse. No crash/power-loss guarantee is inferred from these inert calls.

The additional positive below observed both orderings directly with production methods delegated unchanged. No fake receipt, manually inserted publication row, substituted oracle or fabricated durability proof was used. Legacy markerless restoration shares these exact Core joins by inspection; a distinct historical markerless fixture was not executed here.

## Additional deterministic observations

One fresh stdin interpreter ran **three separate actual Dispatcher/Core/SQLite/Admission/ResourceStore/NativeRuntime journeys**, with the existing fake TLS channel, ML backend and owned Popen handle. Each fixture used real Authority, codec, supervisor handlers, child inference function/encoder, receiver/wait, collect/finalize and publication code. Only exact sync calls were faulted once, delegating all others:

1. Exact binding private-directory `_sync`, after actual link/unlink.
2. `os.fsync` for the exact held stage inode, after actual link/unlink.
3. Exact private-directory `_sync` after actual creation/file sync of `import-complete` (visible COMPLETE, uncertain final directory sync).

For every case: wait for the latch; permit six further automatic observe calls; repeat explicit observe/stop and refused collect/start/ACK three times; compare retained private filename/byte maps unchanged; close control; read actual Core publications (empty); invoke the owner hook on the failed live adapter (still closed); construct a fresh adapter and reopen actual Dispatcher; permit three more automatic observations. All retained no artifact, executing reservation, no wire ACK and closed admission. Observed STOPPED remained truthful. The final-directory case extends the runtime-only visible-COMPLETE regression into the actual Dispatcher/reopen path.

Raw summary before successful fixture teardown:

```text
private: binding=760fe788345193e98a4837fe03774b189b7a4f44355ee5de45e733b647e69fb4
stage: binding=5bdcb31409cda0ce03553339db85f9f77738e018c904eaaf67d26022aa2a1dd1
complete-dir: binding=df9bb0a1cbf4f2272d933c290d83341a60b27100ac0aa659df114db13ec50519
each dispatch row (phase,receipt,outcome,acked) = ('unknown',None,None,0)
private/stage files = complete,import-intent,output,snapshot
complete-dir files = complete,import-complete,import-intent,output,snapshot
each: reservation=executing, artifact_ids=[], wire_ack=0, reopened_closed=true
R295 additional three Dispatcher fault scenarios PASS
```

Separate fresh interpreter: **one real Core-intent positive scenario PASS**. An observational wrapper delegated Core._publish unchanged after asserting completed import/COMPLETE and no prior publications row. The existing after_adoption fault point asserted real publications+publication_work rows and still-running job, then raised OSError. Control retained executing reservation/no artifact/no ACK. Fresh NativeRuntime started closed; actual startup hook restored from the existing Core intent; actual recovery produced one artifact, one total native launch, retained reservation and exactly one ACK after outcome. Output:

```text
PASS real Core intent restore
['handoff-before-Core-intent','durable-intent-before-success-state']
one artifact, one launch, reservation retained, ACK last
```

These four scenarios are additional inert observations, not additional unittest methods or crash experiments. Reproduction uses NativeLifecycleTests.setUp/open, patch worker.native_child.parent_exchange to f.exchange, exact private path/held-stage inode sync hooks, and bounded fixture waits; original methods/results remain authoritative. No owner RPC was added. All project interpreters and paths are specified below.

## Independent execution record and missing receipt

Every project/test interpreter was `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd, explicit:

```python
sys.path[:0] = [
 '/Users/ben/.txcode/worktrees/py-audio-box/t3-f76594f2',
 '/Users/ben/.txcode/worktrees/py-audio-box/t3-f76594f2/tests/service',
]
```

1. **96-method selection launched exactly once, failfast=True; final outcome/count UNKNOWN.** Selections: tests.service.test_native_runtime; tests.service.test_native_lifecycle; tests.worker.test_native_supervisor; tests.worker.test_native_protocol; tests.worker.test_native_qwen.NativeQwenTests; tests.worker.test_engine_qwen; tests.service.test_pocket_preparation.PocketResourceRaceTests. Used TextTestRunner(verbosity=1,failfast=True), with a result subclass printing first unexpected test/attributes and suppressing that test's tearDown/doCleanups. Reviewer mistakenly returned only exec_command.output from the orchestration call, discarding its continuation session handle. Captured output is only `..`; later process listing found no matching running test interpreter. That establishes neither completion nor success. No terminal traceback/result or retained-state path was recovered, and failure retention cannot be certified. No suite or failing case was rerun to replace this missing evidence. This is a reviewer evidence-capture defect, not an established candidate source failure.
2. **6 PASS /0.408s**, separate fresh interpreter, once: tests.worker.test_qwen_import_seam; tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change. unittest.main(module=None, explicit argv, verbosity=1, failfast=True). Fake upstream/root/legacy/extracted archive bytes only.
3. **25 PASS /16.747s**, once: tests.worker.test_engine_lifecycle.DispatcherEngineTests; tests.service.test_qwen_lifecycle.QwenLifecycleTests; tests.service.test_pocket_lifecycle.PocketLifecycleTests; tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests. Same failfast runner shape. Captured terminal `Ran 25 tests ... OK`.
4. **Five direct Dispatcher cases PASS**, once: serialized responsive cancel/unknown/retry; timely completion/lost ACK; late/stale proposal `(False,False)` and `(True,True)`; replacement only after confirmed terminal. Added explicit candidate/tests path; each separate TemporaryDirectory and pair.__wrapped__ generator closed in finally on success. No pytest collection or maintenance helper activation.
5. **Three additional fault scenarios plus one Core-intent positive PASS**, above, separate from method counts.

Confirmed fresh terminal receipts therefore cover **31 unittest methods plus five direct cases**, and four additional deterministic scenarios. Do not report127 fresh methods PASS. The required96 selection remains unconfirmed. r294's historical evidence remains **26 runtime passes from an interrupted invocation +70 remaining passes, then only three affected final branches**, with separate6/25/five-direct groups; it was not a single final96 all-green run. Its127 distinct-method claim is historical implementation evidence, not independently reproduced acceptance here.

`git diff --check 3139916..HEAD`: PASS. Candidate remains clean at exact reviewed HEAD. r277 SHA256 `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`; unchanged lib/tts.py SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known broader legacy digest RED remains unwaived, unregenerated and unexecuted.

## Preserved boundaries and parent gates

F9 final-mode stage creation remains owner-serialized before executor submission. F10 private alias retirement remains owner-only after executor completion and between resource observations; stable markers introduce no executor name retirement. ResourceStore and its conservative highwater/reservations/namespace checks are unchanged. NativeRuntime finalization performs bounded filesystem comparisons/syncs, no RPC/hash/PCM work; filesystem latency has no hard real-time bound.

F6 template authorization remains finite and fixed to pre-enrolled principal/version/context/profile/bundle/inventory; ordinary prepare extends exact allocation authority without public enrollment. F7 numeric endpoint/one absolute connector cutoff, TLS1.3/CERT_REQUIRED/pin and descriptor closure remain. F8 strict Linux receipt shape remains, with truthful synthetic zero-child fixtures. Shared receipts select by frozen lane, not received method. Linux runtime/install-policy paths continue refusing native activation. Core/publication/oracle/DDL and ResourceStore were not weakened by r294. Full source archive/import seams have fresh six-method evidence; the broad native regression result is missing as stated.

Frozen descriptor provenance, original allocation boot/deadline and independently validated local import UID remain separate from native process accounting. Native and common generated_source use the same descriptor identity. First publication checks UTC cutoff; existing Core intent recovery remains historical. Exact receipt and durable Core terminal outcome precede reservation transition; ACK remains last. These source observations do not establish exhaustive interleavings or actual platform behavior.

**Parent-owned acceptance gate:** recover an authoritative result for the original96-method execution, or commission a fresh bounded validation under new authority; do not treat this report as complete final-suite acceptance. No candidate source fix is prescribed from the evidence-capture gap. No parent adoption occurred here. Earlier r290 intermittent second-generation causes remain unproven and unwaived; these green focused observations do not assign their exact cause.

Lot B packaging/launcher/dependency closure, private credentials/configuration enrollment/rotation, bridge/firewall/numeric endpoint and supported owned-version onboarding remain closed. Actual TLS deployment and installed native activation remain unqualified. Lot C actual Apple Silicon Metal probe/load/generate/cancel/restart, kernel process/FD/resource/crash durability, acoustic/performance and release qualification remain closed with unchanged headroom. Actual Metal remains UNAVAILABLE / UNQUALIFIED. Nonempty clone conditioning and arbitrary cancellation-point readiness remain unresolved. CPU Docker and CUDA remain separate mandatory release rows.
