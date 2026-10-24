# Native private spool independent review — r293 Sol

2026-10-25. **NO-GO for complete corrected native Lot A SOURCE adoption.** F10's ordinary executor/private-name retirement race is closed in this owner/executor arrangement, and F9/F6–F8 remain closed in bounded SOURCE scope. One material finding remains: **F11 (P2), owner-finalization storage uncertainty is automatically retried and promoted to publication/ACK.** All four requested test groups pass independently; the additional single inert fault probe below establishes F11. Parent adoption remains pending. Native B/C, actual TLS, installed activation, Metal and release remain CLOSED / UNQUALIFIED.

## Authority and exact pins

Review-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-3309c0e6`, branch `t3/fix-private-import-race`, clean at entry and after execution. Sole lasting write is this report in the root checkout. No source edits, commit, delegation, pytest collection/maintenance/listener, actual socket/process/signal, ML/hardware, Docker/network/download/install, crash, operational cleanup or HELD action. Existing fixture teardown handled only its disposable resources. Root pre-existing untracked STT evidence was listed, not opened or modified.

Exact chain, independently resolved:

- Parent `cc5c740a55d7efa922ca515579d3440d25e770e2`.
- Integration `329d7cbd6c0a28ce6395d6c35ee7543ab3c006b8`; r286 `e021dc482e189b46324b64ef2755238b79dade89`.
- F6–F8 `d5699ddb1a13dc4bed2e36b344883dd8bcbd94fa`; r288 `4ee3a5c6a791f60109c73d203a9335db44b80f7c`.
- F9 `a1822ecc3d03568de072007e9d235036b1e600ef`; r290 `0d1c55ccd81d1499d7d8c4f94540e5992074e2c6`.
- F10 correction `2a8b8dacb1d82fc44b7b1cba1f0897c72465ee5f`.
- Reviewed docs HEAD **`1bbc63b887d2c941947c93615a690ea6f7d9d60e`**.

Read r277, r285 and r286–r292, recovering combined-output truncations with focused reads. Inspected complete NativeRuntime, Dispatcher, ResourceStore, native supervisor/protocol/child, changed integration seams and actual selected fixtures; followed Authority, admission/allocation, native context resolution, Core verification/publication and shared live/detached/install-policy receipt consumers. No applicable AGENTS.md found in candidate/root discovery and checked ancestors. Memory supplied general one-shot/evidence-boundary conventions only; current source and this explicit task supplied technical/execution authority. Both conftests were inspected before execution; unittest and direct allowlisted functions bypassed pytest maintenance hooks and HTTP process fixtures.

## F11 — P2: failed owner fsync is silently promoted through automatic collect replay

Material locations: `service/native_runtime.py:398–442`, especially link/unlink/sync at434–437 and existing-output branch425–426; replay at387–396. Dispatcher finalization/error handling is `service/dispatcher.py:410–418`; automatic observe/collect is432–442 and457–468; publication/reservation/ACK is494–568 and449–451.

`finalize_collect` links `import` into the held stage FD, removes the private alias, then syncs the private directory and stage FD. An OSError at either sync leaves an output file but no successful finalized handoff. The method has no exception latch in the runtime or record. Dispatcher records the command failed, sets phase unknown/quarantine and returns, but this is temporary: its next tick observes the same runtime. A truthful STOPPED observation clears quarantine and launches collect again. `_collect` sees the existing exact output and returns PendingImport with identity=None; finalization takes the existing-output branch, omits both directory syncs and active-import checks, and returns an eligible TransportObservation. Core then publishes, transitions the reservation and ACKs. No reopen, explicit recovery or operator authority is required.

This is different from legitimate replay of a previously completed import or an already durable Core publication intent: the first import never completed its required durability fences, and there was no Core intent at the failure. Byte equality/nlink and a valid native death receipt do not establish the missing control import completion. The implementation report's claim that finalization failure retains unknown/reservation without automatic recovery is therefore false for this path. A failure after unlink can also erase the distinction between an incomplete import and completed historical replay on reopen; no broader crash experiment was performed.

### One deterministic inert reproduction

One fresh interpreter used actual `NativeLifecycleTests` setup/open, actual seeded Core/SQLite/Admission/ResourceStore, actual Dispatcher/NativeRuntime/native supervisor/codecs/child function with the existing fake TLS/ML/owned handle. Only `_sync` was wrapped to raise once for the exact private binding directory on its owner thread, after `import` disappeared. All other calls delegated to the original; no scanner, receipt, validator, collect/finalization or publication return was replaced. The ordinary owner loop performed its own automatic retry. This was one fault scenario, not a retry of any failed suite.

Reproduction core, with the explicit paths/interpreter below:

```python
case = NativeLifecycleTests(); case.setUp(); f = case.f
real_sync = service.native_runtime._sync
# wrapper: if path == exact binding directory, current thread == runtime.owner,
# import is absent, and no prior injection: print first state and raise OSError.
# Otherwise call real_sync(path).
with patch('worker.native_child.parent_exchange', side_effect=f.exchange), \
     patch('service.native_runtime._sync', side_effect=wrapper):
    d = case.open()
    result = d.call('production_generate', 'alice',
                    case.context['plan']['request'], key='r293-finalize-fsync')
    # Existing bounded fixture wait for terminal job, then idle; no manual retry.
```

First state was printed before raising, preserving it in the tool transcript:

```text
OSError('r293 injected owner directory fsync uncertainty')
job=job-001d3d7d874d4464994417557bb286b9
attempt=attempt-09c99456591d4135a647a66c905b461a
stage=stage-8cf23aac1a354ec7b6721895078e5287
job.state=running; artifact_ids=[]; attempt.reaped=false
dispatch=(observing, receipt=null, outcome=null)
reservation=executing; disk_bytes=2097152; pids=1; vram_bytes=0
private files=['snapshot','complete','output']; runtime.failed=false
```

Actual eventual result, with exactly one injection:

```text
wire_ops=hello,prepare,start,observe,collect,get_output,observe,collect,ack
job.state=succeeded; attempt.reaped=true
artifact_ids=['artifact-e4581f4fe9c65b3f46a49be615494dd2']
reservation=retained; native_record.acked=true
phase=idle; quarantined=false; resource_activation=true
resource_fault=false; persistence_fault=false; runtime.failed=false
```

Full first/eventual job/status/demand values were emitted before normal disposable fixture teardown. The failed sync was not repeated on the second existing-output finalization. This proves automatic promotion after injected control-storage uncertainty, not actual crash durability or an invalid artifact's acceptance.

Required correction: latch failed owner import finalization so ordinary observe/collect cannot mint a successful handoff, release reservation or ACK it. Preserve safe observation/stop service and evidence; distinguish incomplete owner import from previously completed historical replay and existing Core intent recovery. Add an actual Dispatcher regression at post-link/unlink private/stage fsync uncertainty, with assertions across subsequent ticks and appropriately bounded reopen behavior. Do not weaken ResourceStore, byte/identity/receipt or publication fences to make it pass. Obtain fresh independent review before adoption.

## F10 closure and preserved positive source assessment

Executor `_collect:342–396` fully validates receipt/proposal/output and retains stable O_EXCL output/complete/import files; its return is PendingImport, not an eligible TransportObservation. Dispatcher attaches owner at startup/reopen (`:60`) and finalizes only after `future.done()` and before command completion/terminal handling (`:407–420`). The tick's scan completes first; terminal publication scans occur afterwards. Thus the sole resource owner cannot enumerate the private alias and retire it before its own lstat. No monitor lock, ENOENT waiver or Pocket exception widening was introduced; ResourceStore is unchanged.

The actual private third-root barrier test passed: real os.walk enumerates import, executor completes while the scan is paused, alias remains present and final output absent, scan finishes healthy, then owner finalization retires import and a subsequent real DB-backed scan remains healthy with no fault rows and nondecreasing highwater. The separate restrictive0077 stage barrier and required-root/removal/permission/symlink/substitution/nonempty conditioning negatives also pass. These are actual production-class inert interleavings, not replacement scanner results.

Writer audit: prepare directly creates stable snapshot/input names; collect directly creates stable output/complete/import names. The only NativeRuntime unlink/rename/replace/rmdir retirement is owner import.unlink at435. Growing files are conservatively charged by highwater plus full retained reservations; restart `_account` charges declared worst case, caps64 records/64MiB and blocks partial output, retained import, unknown/conflicting/type/owner/mode/link evidence. There is no other executor temporary-name retirement. F11 concerns the failed owner transition, not recurrence of the fixed ordinary executor race.

PendingImport is an internal mutable carrier containing immutable raw bytes/Authority plus mutable decoded receipt/proposal and identity tuple. It is neither a serialized token nor independently authenticated capability: construction/fields are spoofable by arbitrary in-process code. Current production has one trusted producer/future consumer, no public PendingImport deserialization or retained executor mutation, so that fact alone is not a demonstrated public trust bypass under r277's trusted-runtime/no-hostile-same-UID-writer premise. Owner thread check, private inode equality and no-replace link apply to new imports; existing output requires exact capped1MiB bytes. Dispatcher revalidates receipt against its own frozen Authority/runtime and Core independently checks result/manifest/media/provenance/UID/nlink. No native UID is accepted as local import UID. Owner finalization contains no RPC/download/hash/PCM; filesystem/fsync and bounded reads have no hard real-time storage latency guarantee.

F9 owner stage creation reaches final0770 via held FD and fsync before executor submission; collision/substitution/partial uncertainty remains conservative. F6 future prepare derives exact allocations from pre-enrolled principal/context/profile templates with independent typed execution/inventory/admission checks, finite retained capacity and no test register callback. Fresh first/next/same-root and public HTTP/CLI/MCP seeded SOURCE journeys pass. F7 numeric endpoint and one absolute TCP/TLS/framing cutoff, TLS1.3/CERT_REQUIRED/exact pin and owned descriptor closure remain. F8 strict Linux shape/count fixtures remain. These facts do not provide installed enrollment or actual TLS authority.

Frozen native receipt selection binds configuration/lane/original boot/runtime/binding/deadline; both live and detached readers use it. Linux adapter/install recovery still reject native activation rather than weakening Linux UID/PID1 policy. Native/common generated_source hash the same typed execution identity; Core independently checks provenance. First-publication UTC deadline and durable cancellation remain owner-fenced; a prior durable publication intent is replayed before the new first-publication deadline check. Normal validated receipt → resource verification → durable Core outcome → reservation transition → ACK ordering is preserved; F11 precedes that path and incorrectly re-enters it after uncertainty.

Supervisor F1–F5 terminal/watchdog/owned-handle/journal paths are unchanged by F10. Exact wait/receiver closure and validation precede the locked cancel/timely decision; durable receipt precedes control import. Independent signaling remains outside state-lock I/O; uncertain native persistence does not regain receipt authority. Their selected fake-handle/barrier regressions pass. Nonempty clone conditioning and arbitrary cancellation readiness remain unresolved.

## Independent executions

Every project/test interpreter was `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd, with:

```python
sys.path[:0] = [
 '/Users/ben/.txcode/worktrees/py-audio-box/t3-3309c0e6',
 '/Users/ben/.txcode/worktrees/py-audio-box/t3-3309c0e6/tests/service',
]
```

Each group ran once, using unittest.main(module=None, argv=['unittest', *selections], verbosity=0, failfast=True):

1. **89 PASS /13.766s**: tests.service.test_native_runtime; tests.service.test_native_lifecycle; tests.worker.test_native_supervisor; tests.worker.test_native_protocol; tests.worker.test_native_qwen.NativeQwenTests; tests.worker.test_engine_qwen; tests.service.test_pocket_preparation.PocketResourceRaceTests. An observational Dispatcher._fail wrapper printed caught exception/job/resource fault rows only for verification_failed and delegated unchanged to the original; no such trace occurred.
2. **6 PASS /0.403s**, separate fresh interpreter: tests.worker.test_qwen_import_seam; tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change. Fake upstream root/legacy/extracted archive evidence only; selected archive method has no subprocess.
3. **25 PASS /17.201s**: tests.worker.test_engine_lifecycle.DispatcherEngineTests; tests.service.test_qwen_lifecycle.QwenLifecycleTests; tests.service.test_pocket_lifecycle.PocketLifecycleTests; tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests. Existing SQLite ResourceWarnings remained visible.
4. **Five direct cases PASS**: test_serialized_responsive_cancel_unknown_and_retry; test_timely_completion_wins_cancel_and_lost_ack; test_late_or_stale_proposal_cannot_publish(False,False) and(True,True); test_replacement_only_confirmed_terminal_increments_generation. Each had a separate TemporaryDirectory and pair.__wrapped__ closed in finally; explicit candidate/tests path additionally supplied. Inspected allowlist, not other subprocess/maintenance tests in that module.

Total **120 distinct unittest methods PASS plus five direct cases PASS**. One additional deterministic F11 scenario is separate from these counts. No failing suite/invocation was rerun. Historical r290 second-job intermittent causes remain unproven and unwaived; this fresh green does not establish their exact cause or exhaustive concurrency safety. The injected F11 is an expected diagnostic finding, not recurrence of that historical intermittent.

Available storage:3,243,068KiB, below6GiB. Only tiny fixtures/archive bytes with existing injected capacity ran; production6GiB-plus-demand/RAM gates were unchanged. Printed device labels are synthetic. No actual runtime/hardware/acoustic qualification follows.

`git diff --check cc5c740..HEAD`:PASS. Candidate remains clean at exact reviewed HEAD. r277 SHA256 `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`; lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known broader legacy digest RED remains unwaived, unregenerated and unexecuted.

## Parent-owned gates

Correct F11 and obtain fresh independent review before complete SOURCE adoption. Retain F10/F9/F6–F8 bounded closures and historical intermittent limitations. No adoption occurred here.

Lot B native package/launcher/dependency closure, private credential/config enrollment/rotation, numeric endpoint/bridge/firewall topology, supported owned-version onboarding, actual TLS and installed activation remain closed. Lot C actual Apple Silicon probe/load/generate/cancel/restart, process/FD/resource/crash durability, acoustic/performance and release qualification remain closed with unchanged headroom. Metal remains UNAVAILABLE / UNQUALIFIED; CPU Docker and CUDA remain separate mandatory release rows.
