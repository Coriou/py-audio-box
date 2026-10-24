# Native terminal/cancel independent review — r285 Sol

2026-10-25. **Bounded GO for the corrected NativeSupervisor SOURCE stage.** No material finding remains in the reviewed F1–F5 paths. Parent adoption is separate. Full native Lot A, installation/readiness, actual Metal and release acceptance remain closed.

## Authority and exact candidate

Review-only; sole lasting write is this root report. No source edit, commit or delegation.

- Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-85b1b50f`, branch `t3/fix-native-terminal-cancel-r284`.
- Integrated parent `ab460e5087eabb008d8c9c1a0c81d3287e9a5c8d`.
- Supervisor `36e329d6a291189483a963d907defffea70edcc2`; report `9d8113dc5dff4b6b51316c052aa1569f3746e838`.
- F1–F4 correction `f91b8e44abcbf4ef45ab3c1be740cf10cf7b6bc4`; report `02e82ba56b27cfa2b8800c12fbae47cd3672d358`.
- F5 correction source `2bc93f5152eb942d54788867b18ed6f223c65e31`.
- Final docs HEAD **`4956e38ad9b1babaeae52c8ded9095304140ed58`**, clean at entry and after execution.
- Root HEAD `ab460e5087eabb008d8c9c1a0c81d3287e9a5c8d`. Existing untracked STT evidence was listed only, not read or modified.

Read full r277 and r280–r284, actual complete supervisor, protocol, child and their tests, selected engine/import/archive fixtures and correction delta. r277 SHA256 is `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`. No AGENTS.md found in candidate discovery/checked ancestors. Memory registry search returned no relevant hit and supplied no technical authority. Both conftests were inspected before execution: root pytest maintenance hooks and service listener fixtures were neither collected nor executed.

## Cancellation and terminal decision

F5 is corrected at `worker/native_supervisor.py:218–230, 458–484`. Result validation, output write/hash/file and directory fsync, exact wait and receiver join/FD closure precede the decision and occur outside `handle_lock`. The initial cancellation read at line 461 is only an optimization: cancellation is reconciled again at the actual decision under the same lock used by stop. A stop winning that lock first receives `{}`, latches cancellation and forces timely=false with null proposal/output. Thus a cancellation during validation or fsync cannot fall through the former check-to-persist gap.

The decision at lines 476–480 samples the original cutoff under that lock and freezes its boolean. A later stop receives the finite committing/terminal response without setting cancellation. Persistence happens outside the handle lock, while the active slot remains owned; retirement takes the handle lock only after persistence succeeds. A stop that captured an older immutable journal record can conservatively return committing after durability; it still reports the same decision and never falsely acknowledges cancellation. A stop missing the retired slot re-enters state serialization, refreshes the record and returns its durable terminal value. New allocation cannot replace the slot through this interval because terminal finalization holds the state lock.

Collect/observe/ACK serialize behind finalization and cannot consume an in-memory committing decision as a receipt. Replay start cannot launch again. Terminal persistence failure latches allocation uncertainty; subsequent ticks do not retry or mint evidence, stop refuses decision authority, collect is all-null, observe UNKNOWN and ACK/start refuse. Post-replace directory-sync uncertainty was independently probed below. A committing reply received before a later failure is explicitly an observation of a decision, not durability or stopped proof.

The protocol change is confined to `worker/native_protocol.py:240–249`: `{}` accepted cancellation, or exactly `{state: committing|terminal, timely: bool}`. Extra fields, invalid state and nonboolean timely refuse. The future NativeRuntime client must implement these variants; no existing complete client integration is claimed. Shutdown uses the same handle lock and does not overwrite a frozen decision; it retains active accounting until completion or UNKNOWN. No listener/shutdown execution was performed.

Cancellation during output staging may leave a private durable output file whose journal proposal/output are null. It stays in the bounded allocation spool, is not downloadable through get_output and is not recursively cleaned. This is safe retained evidence under the declared private-directory/trusted-writer premise, not a new artifact or publication authority.

## Original findings and synchronization

- **F1 remains closed:** storage failure does not gate exact-owned live-handle TERM/KILL. Historical ACK failure during active work and failure between TERM/KILL retain ownership and continue escalation. Identity mismatch refuses signals; observed exit prevents stale group signalling.
- **F2 remains closed:** signal_tick uses the minimal handle lock before tick tries the state lock nonblocking. Journal, hashing, validation, Popen, receiver join and exact wait are outside handle_lock. No state-lock acquisition under handle_lock was found. Blocked historical ACK tests cover cutoff and public stop. Poll/identity/signal themselves are trusted platform operations; this is no realtime scheduling claim.
- **F3 remains closed in process:** persist latches affected IDs uncertain, Journal.value advances only after successful directory fsync, prepared tick and terminal cannot repair quarantine, and affected receipt/output authority stays inaccessible. A frozen terminal decision does not bypass these fences. The in-memory latch is not crash/restart recovery authority.
- **F4 remains closed:** terminal get_output permits exact original-boot tuples after current deployment/config authentication, including after ACK, while digest/range/full-file hash checks and old-boot mutation refusal remain. The actual child-function/fake-ML/encoder/receiver test verifies downloaded bytes across journal reopen.

Receiver result/closed fields are consumed after receiver termination/join; exact wait is attempted once and must match observed exit. Handle registration, cancellation, decision and retirement are serialized; signal_tick's exit observation precedes finish_tick consumption. The unsynchronized early cancel read cannot determine final success because the authoritative read is locked. State records are replaced rather than edited in place. These observations and tests support this bounded correction; they are not an exhaustive language-runtime/platform race proof.

## Independently executed checks

All Python invocations used candidate cwd, `/private/tmp/pab-0c-venv/bin/python -I -B` and explicit `sys.path[:0]=['.','tests/service']`. No pytest invocation or collection.

Exact final focused invocation, once:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=[".","tests/service"]; unittest.main(module=None,argv=["unittest","tests.worker.test_native_supervisor","tests.worker.test_native_protocol","tests.worker.test_native_qwen.NativeQwenTests","tests.worker.test_engine_qwen"],verbosity=0)'
```

**58 PASS, 3.024 seconds.** Includes the six correction methods: barriers during validation, actual output fsync, terminal persistence, durable/preretirement, terminal persistence failure, and closed stop response negatives. Actual handlers use fake handles/channels/clocks and tiny valid encoded results. Assertions cover cancellation state, reply semantics, retained slot, receiver FD closure, wait count one, repeated stop/start/collect/ACK/download and launch count one. Existing F1–F4 regressions also pass.

Separate fresh interpreter on the same final HEAD:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=[".","tests/service"]; unittest.main(module=None,argv=["unittest","tests.worker.test_qwen_import_seam","tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change"],verbosity=1)'
```

**6 PASS, 0.385 seconds.** Root/extracted archive/legacy complete-loader seams use fake upstream; selected manifest check has no subprocess invocation.

One additional stdin interpreter ran **two observational inert scenarios**, separate from the 64 unittest passes:

1. Use NativeSupervisorTests setup, fixture.run_engine and native_child.proposal; patch parent_exchange to return the tiny valid result, start and join the fake receiver, then set fake exit=0. Wrap actual validate_proposal to move fake monotonic time to the exact original cutoff after validation. Tick yields timely=false, coherent terminal stop, wait count one, closed receiver FD and retired slot. This checks deadline sampling after predecision work.
2. Same setup, wrapping actual os.fsync and raising on the second directory fsync after start: the first is output-directory sync, the second terminal journal-directory sync after replacement. Tick retains the frozen decision and active handle, marks failed/uncertain, and yields UNKNOWN/null. Three further ticks plus stop/start/ACK/observe/collect produce no journal.save call and no renewed authority. Wait remains one and receiver FD closed. No restart/crash was attempted.

`git diff --check ab460e5 HEAD` on the candidate: PASS. Candidate clean after checks. `lib/tts.py` SHA256 remains `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; historical broader legacy digest RED is not waived. No failed suite or retry occurred.

Available storage observed 3,405,472 KiB, below 6 GiB. Only tiny disposable fixture/archive contents ran with injected capacity. Production 6 GiB plus declared demand admission remains unchanged. Existing fixture teardown touched only owned temporary fixtures. No actual process launch/signal, socket/listener, real ML, hardware, Docker, network, download/install, crash experiment, operational cleanup, HELD action, source edit, commit or delegation. CPU/CUDA/MPS printed labels are synthetic declarations.

## Parent-owned gates

This GO applies only to the reviewed corrected supervisor source stage. Parent decides adoption. Full Lot A still requires NativeRuntime/client including stop variants, detached immutable control transfers and trusted local import UID, native registration/admission/resolution, shared frozen-lane receipt consumers, Dispatcher/Core cancellation/publication-intent/reservation/ACK ordering, and the complete Authority-to-Core restart/replay slice. No source acceptance of that absent vertical slice is granted.

Native packaging/enrollment/dependency closure and actual Apple Silicon probe/load/generate/cancel/restart remain subsequent gates. Nonempty clone conditioning remains UNKNOWN. Actual kernel PID/FD identity, TLS deployment, durability/crash behavior, acoustic/performance and installed readiness are unqualified. Metal remains unavailable/unqualified; CPU Docker and CUDA remain mandatory separate release rows.
