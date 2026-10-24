# Corrected native supervisor independent review — r283 Sol

2026-10-25. **NO-GO for acceptance of this supervisor stage.** The four r281 findings are corrected in their reported paths, but a new P1 cancellation/finalization race remains in the implemented supervisor. Full native Lot A, activation/readiness, actual Metal and release acceptance remain closed. Parent adoption must wait for correction and fresh independent review.

## Authority and exact candidate

Review only; sole lasting write is this root report. No candidate/source edit, commit or delegation.

- Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-4d9ec80c`, branch `t3/fix-native-supervisor-races`.
- Parent `8ac77dd7e0e53daa66516cde4a9bff1761d79ead`.
- Supervisor cherry-pick `128a189f2597917677598e1f1a1316b6db02597f`.
- Historical implementation report `60dae5082a3b94cdbe995074c83c414b67dfb512`.
- Correction source `e9c17128f65d36d0e97befc3bf38e7a5227183f8`.
- Final docs HEAD **`30a7a09ce19029104d2db5feb5554860a8538c30`**, resolved and verified before execution. Candidate clean before and after checks.
- Root HEAD `8ac77dd7e0e53daa66516cde4a9bff1761d79ead`; pre-existing untracked STT evidence was listed only, not opened or changed.

Read full normative r277 and reports r280/r281/r282. r277 SHA256 verified `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`. Inspected the supervisor, correction delta, child result/receiver path, protocol, regression fixtures and selected import/archive execution. No AGENTS.md found in candidate discovery or checked ancestors/root. Memory registry keyword search returned no relevant entries and supplied no technical authority.

Both conftests were read before execution. The root installs a pytest maintenance guard; the service fixture can launch a real HTTP listener. Neither was collected or executed. All Python commands used `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd, and explicit `sys.path[:0]=['.','tests/service']`.

## New material finding

### F5 — P1: acknowledged cancellation during result finalization still produces a timely successful proposal

`worker/native_supervisor.py:218–226, 454–469`.

The fast stop handler sets `active['cancel']=True` under `handle_lock` and returns immediately, including while finalization owns the state lock. Finalization reads cancellation only once at line 456. It then validates the proposal, writes/fsyncs output, computes timely from the clock alone, persists a terminal receipt/proposal/output and retires the active slot. It never reconciles a stop accepted during those operations. The cancellation field is therefore shared across two synchronization domains without a terminal decision protocol.

Two deterministic inert barrier probes entered actual handlers with fake Popen/channel/clock and a valid tiny encoded result. Both reproduced:

```text
stop_reply={}
cancel_latched=True
timely=True
proposal/output exposed=True
failed=False
exact wait count=1
```

Probe A blocked `worker.native_child.validate_proposal` after finalization passed the cancellation check. Probe B blocked the actual output-file `os.fsync` at line 461. In each case, another thread called the authenticated exact-tuple stop handler and received success before the barrier was released. Finalization then completed with a timely receipt and downloadable output. The fake time remained before the original cutoff. These are successful adverse observations, not failed unittest cases. No process, socket or listener was created.

This concerns implemented cancellation behavior, independently of the deferred NativeRuntime/Core slice. It does not demonstrate an actual Core publication: Core is absent from these probes and must retain its own cancellation fence. It does demonstrate that an accepted native cancellation can leave success-eligible evidence, contrary to r277 acceptance 4 and the supervisor's own cancellation predicate.

Required correction: define and synchronize cancellation versus terminal commitment/retirement. Any stop accepted before the terminal decision must prevent a timely successful result; a stop arriving after an irrevocable terminal decision must observe that decision consistently. Keep journal/hash/fsync/wait work outside `handle_lock`, preserve watchdog independence, and do not solve this by holding that lock across durable I/O. A late unsynchronized cancellation reread alone leaves another check-to-persist window. Cover validation, output fsync and terminal persistence/retirement boundaries with deterministic barriers, including replay. Retain exact wait, receiver closure, durable terminal and uncertainty fences.

### Minimal reproduction recipe

Use `NativeSupervisorTests` setup and its existing cleanup only. Prepare; obtain a valid result through `fixture.run_engine()` and `native_child.proposal(snapshot,result,e)`; patch `native_child.parent_exchange` to return `(proposal,result['data'])`; dispatch start and join its fake receiver; set fake handle code to zero. Patch `native_child.validate_proposal` with a wrapper that sets an Event, waits on a second Event (bounded two seconds), then calls the original validator. Run `s.tick` in a thread. After the first Event, dispatch `q('stop')` on the main thread and inspect `s.active['cancel']`; release the barrier, join the tick thread, then dispatch collect. For the second scenario, wrap `worker.native_supervisor.os.fsync` instead, blocking its first call after start/receiver completion; call original fsync after release. Both scenarios completed without quarantine and exposed the result described above. No source test was added.

## Assessment of all four original findings

**F1 closed in the reported failure paths.** `signal_tick` no longer gates TERM/KILL on storage health. Historical ACK failure quarantines receipt/launch authority while retaining exact active ownership. Public exact-tuple stop reaches that handle before acquiring the state lock. The focused regressions exercise failure during a second active job and between TERM/KILL, including retained receiver uncertainty. Poll exit prevents later group signals; identity mismatch refuses signalling and quarantines. These are fake-handle source results, not Darwin PID-reuse qualification.

**F2 closed for live-child signalling independence.** `handle_lock` protects registration, poll/identity/signal selection and cancellation. No journal/hash/Popen/join/wait or state-lock acquisition occurs under it. State-to-handle acquisition has no reverse state-lock acquisition path. Tick signals first and tries the state lock nonblocking. The blocked historical ACK tests exercise both cutoff and public stop while save owns the state lock. Start reserves the active slot before Popen; stop latches there during blocked Popen, and registration preserves cancellation. Trusted poll/identity/signal calls themselves remain platform operations; no realtime guarantee is inferred. F5 is a separate terminal-decision defect introduced by the now-concurrent stop path.

**F3 closed for in-process persistence uncertainty.** `persist` latches the affected allocation in `uncertain` on failure, including after replace/directory-sync failure. `Journal.value` changes only after the directory fsync succeeds; the failed disk write can consequently differ from memory, but no later tick/stop/ACK/start repairs or terminalizes that uncertain allocation. Observe returns UNKNOWN; collect returns nulls; get_output refuses uncertain evidence. The focused pre/post-replace tests repeatedly exercise these handlers and assert no later save. Prepared tick and terminal also have global failure fences. Historical immutable valid records do not inherit the affected allocation's receipt authority. The uncertainty set is memory-only: no restart/crash recovery authority or durable uncertainty marker was added or qualified, and no restart experiment was performed here. Original evidence must remain parent-owned.

**F4 closed.** Exact original-boot terminal get_output is admitted after current deployment/configuration authentication, including after ACK. Digest/range/full-file hash remain checked; wrong boot/binding/configuration/digest refuse. The existing actual child-function/fake-ML/encoder/receiver fixture compares downloaded bytes before restart, after journal reopen and after ACK. Old-boot start still refuses; this exception does not permit old-boot mutations.

## Executed checks and accounting

1. Final focused invocation, once on exact final HEAD:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=[".","tests/service"]; unittest.main(module=None,argv=["unittest","tests.worker.test_native_supervisor","tests.worker.test_native_protocol","tests.worker.test_native_qwen.NativeQwenTests","tests.worker.test_engine_qwen"],verbosity=0)'
```

**52 PASS, 2.485 seconds.** Includes the eight correction methods, real handlers with fake handles/channels, thread barriers, pre/post-replace faults, final wait refinements and old-boot output extension.

2. Separate fresh interpreter on the same final HEAD:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=[".","tests/service"]; unittest.main(module=None,argv=["unittest","tests.worker.test_qwen_import_seam","tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change"],verbosity=1)'
```

**6 PASS, 0.398 seconds.** Fake upstream root/extracted archive/legacy imports, plus manifest membership/metadata. This independent run occurred after the final wait refinements, closing the implementation report's timing limitation.

3. Two stdin probe invocations, **three observational scenarios**, separate from the 58 unittest passes: validation-barrier cancellation; output-fsync-barrier cancellation; repeated wait-exception UNKNOWN. The first two reproduce F5. The third injected one wait exception, called tick three times and observed wait_calls=1, failed=True, retained active handle and all-null collect. No retry of a failed suite or source correction occurred.

4. `git diff --check 8ac77dd7 HEAD`: PASS. Candidate remained clean. `lib/tts.py` SHA256 remains `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; historical broader legacy digest RED is not waived.

Available storage observed 3,439,556 KiB, below 6 GiB. Execution used only tiny disposable fixtures/archive contents with injected admission capacity. Production's 6 GiB plus declared demand gate remains unchanged. Fixture teardown affected only each test's own disposable resources. No operational cleanup, HELD action, real inference child/process/signal, socket/listener, real ML, network, Docker, download/install, hardware or crash test occurred. Printed MPS/CPU/CUDA labels are synthetic fixture declarations.

## Bounded positive conclusions and remaining gates

Normal terminal still requires exact handle poll/wait, joined receiver and verified parent FD closure before durable receipt. Failed wait is attempted once and remains UNKNOWN. Result without exit and exit without result do not establish timely success. Launch-intent precedes Popen; factory uncertainty retains the allocation and prevents replay launch through quarantine. Records are replaced under the state lock; active registration/retirement and signal identity use the handle lock. Receiver result/closure is consumed after join. The material newly observed shared-state defect is the cancellation/terminal race above; this is not an exhaustive platform data-race or crash matrix.

No Linux receipt/proof, UID or ECHILD substitution was found in this correction. The trusted exact Popen handle and optional start identity remain the source premise; real kernel process/group identity and poll-to-signal behavior remain unqualified. NativeRuntime/client, immutable control-side transfers/local import UID, shared frozen-lane native receipt consumers, registration/admission/resolution, Dispatcher/Core cancellation/publication-intent/reservation/ACK ordering and the complete Authority-to-Core restart/replay slice remain parent-owned full Lot A gates. They are deferred scope, not additional findings.

Correct F5 and independently review the exact resulting source before parent adoption. Native packaging/enrollment/dependency closure and actual Apple Silicon probe/load/generate/cancel/restart remain later gates. Nonempty clone conditioning remains UNKNOWN. Actual Metal stays unavailable/unqualified; CPU Docker and CUDA remain mandatory separate qualification rows. This report grants no full Lot A, installation, hardware, acoustic, performance or release authority.
