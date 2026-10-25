# r361 Astra — finite installer ownership correction brief

2026-10-25. **GO for one bounded Sol source correction; NO-GO for adoption of unchanged e29af7892e6200fdaea5c1388c9e80bcf0fd227c.** A shared local-process ownership/collector primitive plus an enclosing exchange finalizer is sufficient. Route the existing notify subprocess through the same ownership rule. No new service, scheduler, recovery framework, public API, UI or product redesign is needed. Independent Sol review followed by Astra assessment and parent adoption remain separate steps.

## Basis and scope

Source-readonly advice against `/Users/ben/.txcode/worktrees/py-audio-box/t3-1e21c6da`; verified HEAD and ancestry: `e29af7892e6200fdaea5c1388c9e80bcf0fd227c` → `bc075ccc93510564f25c9788348c2ac0aa88df54` → `45ce583b1f5c61e14c0218911a76ea68a6e928d9`. Read normative r356, candidate r357/r359, root r358/r360, collector/client/helper/Wire and focused adapter fixtures; inspected initializer's synchronous child seam. Bounded applicable AGENTS checks found none. Memory registry search supplied no relevant premise. Prior test counts are report evidence only: r358 239 PASS and r360 249 PASS do not cover their demonstrated faults; no tests or probes were rerun here.

Sole write: this mode-0600 report. No application imports, source edits, tests, Docker/network/listener, actual child, ML/hardware, HELD/STT content access, operational cleanup, commit, merge or delegation. Existing unrelated files remain untouched.

## Findings to retain and narrow

R1 remains a concrete exclusion defect: `daemon_client.collect:109-121` can escape from sleep inside its wait-error handler, allowing helper/initializer contexts to unwind before child reap. Cleanup closes at 105/108 can similarly bypass the wait loop. Fix the entire local ownership interval, rather than wrapping only the reproduced sleep.

R2 remains a concrete local attach-process leak: `Docker.exchange:246-280` lacks guaranteed finalization after Popen; finish/send/receive/setup failures bypass wait, and success retains references/streams. This does **not** demonstrate helper lock release, a second Core, or false publication. It is the local Docker CLI process that must be reaped; helper completion is a separate observation.

The nearby `Docker.action('notify'):218-228` also starts a local process, uses communicate with a timeout, and never removes its retained reference. Include it in this same finite ownership correction, not as another architecture gate. All four new-lane Popen sites are therefore in scope: call, exchange, notify, helper child. Preserve strict host launch code.

F2's live PIPE payload caps/no disk spool and F3's exact terminal EOF rejection remain corrected at SOURCE. Keep those tests; do not reopen their original defects merely because lifecycle finalization is unfinished. Correct r359's “user explicitly resolved” attribution to **orchestrator-resolved exceptional parking**, preserving the historical report's provenance rather than inventing new user authorization.

## One ownership rule, four call sites

Prepare argv, input, limits and ordinary deadlines before spawning where practical. From successful Popen acquisition, establish one synchronous ownership scope immediately, covering reference registration, stream/selector setup, protocol work, decoding, timeout, cleanup and exit. A local variable is already ownership; list insertion must not be the first unprotected failure point. Popen failure with no acquired process is a normal pre-ownership error.

Use a small internal owner/finalization function or context manager in the existing module; collect and exchange share it. Its state need only distinguish owned/unreaped, sticky failure, verified reaped, and resources released. No persistent state store. The parent helper stays on the same stack, inside the existing root/lane/Core exclusions, while its child is unresolved. No background reaper or detached ownership transfer.

After the first failure: stop protocol/actions, discard retained output, close input, service or close output, and wait for that exact local process. Never clear intent, run a replacement, advance phase, publish success or force KILL. A later zero exit does not erase the failure. Keep at most one bounded failure record; repeated exceptions must not accumulate tracebacks, payloads or logs. Return/raise only after local reap and resource disposition are established. Remove the matching retained reference only then; an unconditional finally-remove is invalid if finalization can escape early.

Successful collection still requires exact intended output and zero exit. Exchange success additionally requires one valid complete frame, no pending/trailing bytes, actual EOF within its existing terminal budget, local attach zero exit, and the existing same-helper ID/image/mount/incarnation/daemon final checks. Close/release streams on success too. Final inspection/check failure occurs after attach finalization and must return failure without leaking it.

Notify should use the same bounded input/wait owner, supporting absent stdout or an explicitly capped PIPE. Preserve its closed input budget, fixed command and nonlogging behavior. No separate communicate-timeout ownership exception.

## Cleanup, cadence and completion semantics

The ownership guard must surround cleanup and cadence operations themselves, including selector exit/unregister, stdin/stdout close, clock/deadline evaluation and wait-error recovery. Catching BaseException around only p.wait is insufficient. Catch recoverable KeyboardInterrupt/SystemExit as sticky failure while owned; defer propagation until finalization. A close failure must not prevent the other stream disposition or the next reap attempt.

Track each resource independently. After close raises, inspect the owned object's closed state before retrying: a close can have completed before raising. Retry through that object when needed; do not blindly os.close a remembered numeric fd that may have been reused. Unresolved closure is not “released.” Keep the bounded owner and retry with cadence, including after reap if resource disposition remains unresolved; do not admit another invocation that accumulates resources. Once genuinely reaped, the child exclusion obligation itself is satisfied; retaining slightly longer for local finalization is acceptable and needs no new authority rule.

Use bounded wait/select slices, approximately the existing 0.1 seconds. Immediate wait errors or nonterminal returns must pass through a guarded cadence before another wait attempt. If that cadence is interrupted, resume the remaining monotonic interval inside the guard; do not place another unguarded sleep in an exception handler. Cleanup retries share this cadence. Continuous ready output may perform useful bounded reads; idle/error paths must not spin. No per-retry output or process creation.

The ordinary deadline bounds the success attempt, not ownership duration. After expiry, keep one child, bounded streams/buffers and held exclusions until real completion. This exceptional duration is explicitly unbounded by the orchestrator decision. Tests inject finite fault sequences ending in completion; they need not hang forever to prove parking.

For these synchronous adapters, trust the normal OS-backed Popen wait/poll ownership contract with no competing reaper, ignored-SIGCHLD auto-reap arrangement or caller-written returncode. An integer from the controlled production wait path denotes collected exit status; negative status is a genuine signaled exit but never success. An arbitrary integer/mock returncode, EOF, timeout, ECHILD under lost ownership, or helper container disappearance does not independently prove reap. Do not fabricate a returncode on errors. There is no need for a new /proc attestation or replacement waitpid subsystem. Fake wait transitions are source-control evidence only; actual OS behavior remains unqualified.

BaseException handling is not protection against SIGKILL, interpreter crash, os._exit, OOM death or host/VM loss. Repeated asynchronous exceptions at arbitrary bytecodes cannot be made absolutely atomic by Python try/finally. The acceptance target is explicit operational/setup/cleanup/wait/cadence fault paths while the owner executes and can make progress, not survival under unlimited adversarial interruption of every instruction or a permanently broken runtime. Do not add signal machinery or another review campaign for that impossible claim. On parent death, durable pending intent/refusal remains necessary; it is not a reap receipt or permission to activate.

## Attach identity is not helper identity

The client owns the local `docker start --attach --interactive <exact-id>` Popen and its pipes. Docker owns the retained helper container. Inside that container the helper owns its Core/hello/writer child and the volume exclusions. Closing attach stdin/stdout may disconnect or make writes fail; it neither proves nor requests helper completion. Reaping the attach CLI proves only local CLI completion. No TERM/KILL/rollback of the helper is a substitute.

On exchange failure, close input, discard/drain output in bounded chunks, or close the output reader if draining cannot proceed safely; then retain/wait the same attach process. If it remains attached to a parked helper, the client may also remain parked. Caller timeout is not helper completion. Preserve existing exact helper IDs, handle/authority receipts and incomplete state; do not synthesize terminal helper history from the CLI exit. During normal handshake, the attach process may coexist with one synchronous action CLI: “ONE child” limits helper child ownership and prevents replacement/accumulation, not all client protocol subprocesses globally. Finalize an action subprocess before resuming or unwinding its enclosing attach owner.

EOF and byte budgets remain separate: capped retained payload plus fixed read chunk, OS pipe and object overhead; no unbounded temporary output. After overflow/timeout, continue bounded discard through EOF/reap when usable. Reader closure is a failure fallback, not proof of EOF. EOF alone never proves process exit, and process exit alone never validates the terminal protocol. A producer/descendant retaining a pipe can keep failure ownership parked without authorizing success.

## Finite implementation and review exit

Confine changes to client/helper ownership seams, focused fixtures, and truthful new-lane documentation/report attribution. Keep existing Core, persistence, lock order, intent, ACK/reap/publication, activation and old-host validators unchanged. Add this finite regression matrix with inert pipes/fake processes and finite fault sequences:

1. Reproduce r360 R1 at actual helper init: wait error → cadence KeyboardInterrupt → later wait timeout/unknown → genuine modeled reap. All five roots remain contested at every pre-reap observation; only afterward can they be acquired. Intent remains, one spawn, no done/success or next phase. Cover the noninteger-wait cadence too.
2. Inject setup/selector teardown and stdin/stdout close faults, including close-then-raise and transient close-before-completion. Reap still occurs; other resources are serviced; errors remain sticky; no premature retained removal. Assert cadence on immediate-error retries using a controlled clock, not wall-clock performance tests.
3. Exercise exchange pending junk/second frame, separately read trailing byte, terminal EOF timeout, Wire construction, send/receive and action failures. Each must finalize its exact attach process and streams before error. Assert before fixture teardown, which previously hid leaks.
4. Exercise successful exchange and nonzero/signaled attach exits, final inspection failure, and attach-zero with helper still running/nonzero. Only the full existing success predicate passes; every completed local owner releases references/streams.
5. Exercise notify success, timeout and interruption using the common owner; when nested in exchange, both local owners finalize in stack order without any helper kill or replacement.
6. Retain current live overflow bounds, exact EOF, ordinary timeout and good-output tests for client/helper. Run the existing r360 coherent allowlist plus these additions once on the final candidate; report any failures and necessary focused correction runs honestly. Independent review checks these finite conditions and the four-site ownership coverage, not a new exhaustive product audit.

Once those source conditions pass and independent review agrees, R1/R2 can close at SOURCE. No requirement to prove infinite fault resistance, rerun unchanged broad suites repeatedly, or reopen unrelated hardware design. Parent owns adoption; this advice neither codes nor self-adopts the correction.

Actual Linux/amd64, THIS Intel Mac Desktop and OrbStack remain separate **UNRUN** qualification rows. Actual process/pipe/UID/capability enforcement, packaged runtime, audible Kokoro/artifact/restart behavior and measured resources remain unqualified. Preserve headless HTTP/CLI/MCP and cleanvoice product scope, CPU/GPU/native Metal requirements and no UI. CUDA/Metal remain unqualified; **release NO-GO; legacy CLOSED**. No operational or cleanup authorization follows from this report.
