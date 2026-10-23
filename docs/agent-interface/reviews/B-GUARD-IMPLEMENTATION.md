# B blocked-pipe guard implementation handoff

2026-10-24. SOURCE IMPLEMENTATION ONLY. No GO or test/runtime acceptance is claimed. Controls, independent Sol review after controls, and any actual execution remain parent-owned gates. Historical r42/r43 RED and historical PID229 UNKNOWN remain unchanged.

## Frozen source and scope

Checkout: `/Users/ben/.txcode/worktrees/py-audio-box/t3-a31af448`, branch `t3/implement-pipe-wait-guard`. Reconstructed from exact `ccf7ab3a7ec2f36c3460617351a634d2a1b9223c`. Immutable predecessor `t3-48db44e1` was not modified.

Executable guard commit: `4f04cdbc51bc13081f09bc3dd2205deb4e55fc99`.
Guard SHA256: `3ab2f33633be9bd9ed00123a6daa13ba79a227cbcd8d3229f00f78e999fcc7e3`.
This handoff is the only following documentation change; its commit is reported separately at handback rather than embedding a circular self-hash.

Controlling design read in full: root `P1B23-BLOCKED-PIPE-WITNESS-ASTRA-r44.md`; root r43 facts read in full. Also read guard, original singleton/model, runner/framing and complete observer source. Those root reviews reside in `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/`, outside the frozen candidate. No applicable ancestor/scoped AGENTS.md was found. No delegation was performed in this implementation lane.

Only authorized writes: `tests/_maintenance_wait_guard.py` and this handoff. Production, tests/model/assertions, controls, conftest, observer, entry, runtime, dependencies, manifests and the original P211 inventory remain unchanged. No control was authored.

## Capabilities and executing sites

`BlockedPipeWitness_B` independently designates the singleton before the owned fork, including invalid designations. Admission requires exact nodeid/name/no parameters, positively present callable `node.obj` identical to the actual module function, and that function's actual caller/code/globals. Missing obj never defaults. The actual collected code supplies observed/ready/select_blocked and the two lambda capabilities. Pytest assertion rewriting is respected: the collected target is not compared against a plain source compile. The assertion-free model and production runner/framing are compile-compared without execution.

Captured function objects, their code identities, closure contents, original builtin write/clock/select/sleep/kill, model limit/session/fork, original guard fork/wait closures and installed wrapper/original operation identities are checked. A same-code observed clone with a fake write closure cannot qualify. Wrapper restoration is scoped; no new product seam exists.

AST statement/call positions plus disassembly of the executing captured code resolve instruction sets, including CALL cache positions and duplicated finally call sites. No host numeric offset is an admission constant. Host CPython 3.14.7 compile/disassembly cross-checks, without execution:

| Source site | Host instruction offsets |
| --- | --- |
| runner child deadline raise 558 | 844 |
| runner write 562 | 1010 |
| target observed raw builtin call 5844 | 10 |
| runner catch-all exit74 569 | 2576 |
| runner parent deadline raise 624 | 1286 |
| runner empty-read timeout raise 627 | 1378 |
| runner parent read-select 626 | 1322 |
| runner reap call 661, duplicated finally paths | 2068, 3020 |
| reap initial wait 581 | 102 |
| reap kill 593 | 202 |
| reap cleanup wait 604 | 302 |

Child561, parent636 and child_failed648 never qualify. The executing interpreter must resolve its own sites successfully or B remains invalid.

## State and route proof

The invocation trace stores finite phases (`armed`, `backpressure`, `timeout`), primitive process/frame/raw/result/exception numeric identities, FD/dev/inode/mode/access facts, raw length/offset, finite original deadline, flags and captured code/callable capabilities. It stores no frame, traceback, exception, memoryview, operation result or session object. Trace arguments/locals are released before witness IO; no retry history, registry or payload serialization is added. Captured callable closures necessarily retain their original target-owned capabilities until invocation cleanup.

The ownership map is captured before fork, compared against both live pipe ends, and checked as the same FIFO with opposite exact read/write access. Result dev/inode must differ from witness pipe and observer regular file. At EAGAIN the live write descriptor must still match and be O_NONBLOCK/O_WRONLY. The sole accepted EAGAIN traceback is runner562 then exact observed builtin-write call, with exact builtin BlockingIOError/EAGAIN. It binds the original framing return's raw identity, operation result identity, bounded bytearray, exact int offset and exact contiguous one-dimensional byte memoryview owner/content/slice. The bound raw/offset/FD/deadline and hook identities are revalidated; post-certificate successful writes/progress, replacements and unexpected runner exceptions invalidate the certificate.

Child timeout requires one traceback entry at558, same bound raw/offset/FD, remaining<=0 and expiration against the unchanged accelerated clock/deadline. Only the numeric timeout identity is retained. Final exit independently reads `sys.exception()`, checks exact builtin ValueError/exact one-str args, same active numeric identity, exact complete one-entry traceback whose frame is the actual direct caller, catch-all569 instruction, live write pipe and installed trace. The unchanged immediate runner handler bridges identity without an additional retention root. Observation precedes final qualification; the final token is immediately followed by captured exit, with no diagnostic callback between them.

Parent terminal qualification occurs before guard.observe and requires exact reap581/604 and runner661 caller, original returned tuple/owned PID, original is active exact timeout, empty exact bytearray, original finite expired deadline and the complete single-runner parent624/627 traceback. Route624 requires nonpositive remaining;627 permits positive remaining but requires the exact ready→select_blocked empty-read return and clock expiration. Saved pre-close pipe identity is used; the closed parent read FD is not fstat'ed as proof. Propagation separately rechecks escaping original identity/type and the exact guard-call/runner traceback prefix; a cleanup replacement latches refusal.

For SIGKILL, opcode observation surrounds the actual captured builtin kill call at593 with exact owned PID, signal module/SIGKILL9 and runner/reap route. A normal following opcode records successful call return; an exception cancels pending kill evidence. ESRCH alone is not successful signal provenance. Raw9 additionally requires the terminal604 cleanup wait with killed=True. No new signal/wait/reaper/sleep is introduced.

## Token and shared-surface proof

Parent owns `B:a;` and independently appends t/k. Child pipe bytes must be exactly e or e/f. Accepted complete transcripts are:

- raw18944: `B:a;B:e;B:f;B:t;` (16 bytes).
- raw9: `B:a;B:e;B:t;B:k;` (16) or `B:a;B:e;B:f;B:t;B:k;` (20).

Raw74 always requires f, even after a kill attempt. Raw9 proves backpressure plus parent deadline cleanup; it does not prove execution of child558. Unknown/duplicate/partial/reordered/surplus/foreign ownership tokens reject. Armed B takes priority over legacy expected_exit/signal and P; a B/P overlap rejects. Invalid B cannot fall back to V/J/R/T. e/f emit once through the captured builtin writer; short/failed B writes invalidate or leave incomplete evidence. Conservative erroneous-path envelope is43 bytes including23 legacy semantic bytes; such mixtures reject. B alone uses one257-byte terminal lookahead and rejects combined evidence>256 before parsing; parent suffix additions are checked before concatenation. Non-B read/truncation behavior is preserved.

The invocation try/finally is registered before descriptor setup, witness construction and hook mutations. Monkeypatch context owns scoped patches; the exact previous trace is restored before clearing B/P state and closing both descriptors (the second close is attempted even if the first fails). B clears all callable/scalar state, including its bound-callback self-reference. No control finalizer is added that can resurrect a prior target closure.

Static comparison to the base confirmed byte-identical pre-existing top-level functions/classes except the explicitly changed shared MaintenanceWaitGuard.observe/install surfaces. This includes P class/replay/parser, R class/final predicates, backpressure_phase, cleanup-signal policy, aggregate/bootstrap/fresh_run and metadata helpers. Within install, journal_open, semantic_phase, observed_session, observed_operation and observed_completion are byte-identical after dedenting the common setup indentation change. Existing enrollment/reporting behavior remains, with additive B child binding/parser priority. Moving setup under the existing finally changes common indentation; it does not rewrite those predicates. mark adds a B-only short-write check. P's source-selected211 inventory and all pending/unrun labels remain untouched.

Unchanged SHA256 pins:

- service/recovery.py: `c314b2423b71c7d205b4558aee44d8c895e41394adbbb05a85458ef9bd7acf29`.
- tests/service/test_recovery.py: `c236e4045a0c35836fc269fd5938c053ab3e97f1f2682f023470afe3a68dd211`.
- tests/service/test_child_wait_guard.py: `e22095b26c7219fd093f335baeaad3d8c7271cdf2bc30cb8aeef74b3f77c27cd`.
- diagnostics/p-one-node/one_node_observer.py: `ee20c60ff9d5890ae6d9ceacf892443a4aea88dedf7a3952baf5bc98797de582`.

## Remaining gates and evidence limits

Completed only stdlib AST/source comparisons, compile/disassembly without exec, Git scope/hash checks and diff --check. No candidate/product imports, pytest/collection, Docker/runtime, DB or native campaign. Static success is not proof of tracing behavior, performance, restoration under injected faults, original assertion reachability or an actual accepted transcript.

Next lane must append the r44 directed controls without changing old definitions/assertions: real collected and fresh-helper positive binding; node/code/closure/model negatives with valid bootstrap prerequisites; real EAGAIN FD/alias/access provenance and labelled predicate-only view/budget mutations; child route/active exception/replacement/extra-frame/trace-disabled/writer failures; independent parent624/627 and owned kill/race controls; closed transcript/overflow/legacy/foreign ownership controls; restoration/setup failure/reference-release controls. Each negative must prove reached seam, mutation/readback, exact B refusal, latch and next-maintenance/helper refusal. Setup errors do not count as negatives.

Independent Sol review follows those controls. Any authorized actual gate must retain the real observer, unchanged original assertions, exact receipts and at least one unmodified actual74 complete child route. SIGKILL-only success leaves actual74 qualification pending. The singleton observer/entry is not a broad control runner; isolated inert-sink controls must be labelled and restored. Original P/J/R/transport/close/reap/shared reporting preservation remains pending under the already-declared subsets, with no broadened inventory or blind rerun. All native/platform/installed/durability/release claims remain unqualified.

## B1 capability-origin response — independent successor, 2026-10-24

SOURCE IMPLEMENTATION ONLY; no self-GO. This response supersedes the earlier statement that current collected constants alone supply capability authority. Root r47's B1 is accepted. Read original root Astra r44, root Sol r47 and the complete preceding candidate handoff independently. No applicable AGENTS.md exists in checkout ancestry or tracked/scoped inventory. Memory was used only for general evidence-boundary guidance, not technical authority.

Own fresh T3 checkout: `/Users/ben/.txcode/worktrees/py-audio-box/t3-eca7b725`, reconstructed by detaching at exact `be8a32ba1aca6073a9125e7e251b70565069d2d8`. Immutable `t3-a31af448` remains clean at that pin; its executable `4f04cdbc51bc13081f09bc3dd2205deb4e55fc99` and guard hash `3ab2f33633be9bd9ed00123a6daa13ba79a227cbcd8d3229f00f78e999fcc7e3` were preserved.

Successor executable commit: `920009f77521ea34fee8e277360708118110f391`.
Successor guard SHA256: `e6009824186b29585d597308806ae3e7925e161cc4043127550e6cb220a35829`.
The following documentation-only commit is reported separately at handback to avoid a circular hash. Only the guard and this appended response differ from the frozen predecessor. No control was authored or changed; the parallel control lane remains independent.

### Independent authority and closed outer proof

`target_source` reads the canonical guard-relative `tests/service/test_recovery.py`, requires the executing filename to resolve to that source and requires its exact frozen SHA256 `c236e4045a0c35836fc269fd5938c053ab3e97f1f2682f023470afe3a68dd211`. Thus a replacement function filename cannot nominate another file as authority. Missing, unreadable, wrong-digest or ambiguous source raises into the existing armed-invalid path.

The original module is compiled without execution and with `dont_inherit=True`. The original model comparison remains. All five assertion-free capabilities are independently compared to this compile: observed, ready, select_blocked, accelerated clock and result operation. Clock is selected as the `monotonic` keyword lambda of the unique `SimpleNamespace` call; result operation is the third positional argument lambda of the unique production-runner call. Each role must be exactly one Lambda and exactly one source-compiled LOAD_CONST code object at the complete AST span. Neither lambda is selected by same-name membership or line alone. Their complete code projections must match their distinct roles, then actual collected-constant identity is required. Named capabilities likewise require both source projection equality and actual collected nested-code identity. All previous globals, builtin, actual callable, closure and hook-stability bindings remain mandatory.

The outer target is also independently checked. Under pytest `rewrite` mode, the guard reproduces `rewrite_asserts` on a separate AST of the same frozen full module using the actual node configuration, then compiles that AST without execution. Under explicit `plain` mode it uses the original compile. Unknown/missing assertion configuration refuses; there is no plain fallback for a rewritten mismatch. Full module context is intentional: compile-only inspection on CPython 3.14.7 showed that isolated target compilation can change outer LOAD_GLOBAL/PUSH_NULL code generation and exception/line tables, even when its five nested codes match. The final implementation retains full compilation context. The original, unrewritten AST remains available for the existing instruction-site resolver.

The finite projection enumerates stored fields: argument counts, local count, stack size, flags, bytecode, typed recursive constants (including nested code), names, variable/free/cell names, filename, name/qualname, first line, line table and exception table. Tuple/frozenset constants recurse; scalar constants retain their exact type. Unsupported constants refuse. There is no open `dir()`/method comparison, omitted nested semantics, numerical-constant erasure, position-only acceptance or plain-whole-function comparison in rewrite mode. Existing source-position/disassembly site checks remain. A replacement outer body must match all these expected fields, including every instruction/operand/constant and exception range; preserving nested names, closures or coordinates cannot confer authority. An exactly identical closed representation supplies the same code semantics, rather than a different source authority.

### Compile-only mutation evidence and positive-preservation argument

Host CPython 3.14.7, stdlib AST/compile/disassembly/code-object inspection only. No compiled object was executed, and no guard, target, product or pytest was imported/executed for these checks.

- Normal source nested codes from independent full and selected-target plain compilation matched all projected fields: clock5836 (`clock`, `start`), select_blocked5837 (`select0`, `time`), observed5843 (`marker`, `write`), ready5849 (`select_blocked`), operation5854 (no freevars). Both lambda roles resolved to exactly one LOAD_CONST at their full source span, with constants `(100,)` and `('a', 900000)` respectively.
- `clock.replace(co_consts=(1000,))` preserved co_code, every co_position and co_freevars, reproducing r47's minimal mutation. Its typed recursive constant projection differs from frozen `(100,)`, so independent nested admission refuses. Replacing the corresponding outer constant also differs recursively at outer admission, before capture can become authority.
- An in-memory outer co_code replacement preserving every co_consts object still differed in the closed projection. This is a compile/object structural check, not a claim that invalid replacement bytecode was executed. More generally any outer instruction, operand, nested code, binding layout, exception range or position-table replacement changes an explicitly compared stored field.

The legitimate pytest positive is preserved by construction: the source bytes, full-module AST rewrite algorithm, configured assertion-pass-hook setting, filename and compile context are the same inputs used by pytest's `_rewrite_test` (inspected installed source, not run). Assertion rewriting does not rewrite the three assertion-free nested definitions or the two lambda expressions. Their plain-source semantic comparison is therefore compatible with the rewritten outer code. This is a static argument, not a collected positive or performance claim. Actual pytest code-generation equality, elapsed<2, tracing, assertion reachability and transcript qualification remain unexecuted gates.

### Shared-surface preservation and remaining gates

Independent AST comparison against `be8a32ba` found only `BlockedPipeWitness_B` changed among existing top-level definitions. Within B, only existing `__init__` changed; `code_projection` and `target_source` are the two additions. All other B methods, MaintenanceWaitGuard.observe, install and every P/J/R/shared helper remain AST-identical; the delta touches no source outside B. Existing designation-before-validity, no-fallback, state/route/FD/closure checks, parser/budget, tracing, parent/kill proof and restoration logic remain unchanged.

Git byte comparisons confirm every other tracked file remains at the predecessor. In particular the preceding handoff's recovery, recovery-test, control and observer SHA256 pins remain exact; tests/conftest.py remains `d967baee72fa7fba3cb8974ac0406b2f44231a1c9d49fb7e7c088016f07cc7a3`. Production, original target/model/assertions, control tests, observer/entry and manifests were not edited. In-memory guard compilation and `git diff --check` passed. Both this checkout and immutable predecessor were checked clean before the documentation append.

A fresh independent Sol source review is requested after freezing this response, with its task/result returned at handback. It must review B1 and unchanged-positive compatibility independently; this implementer does not approve its own change. Parallel directed controls still need valid bootstrap/caller/FD prerequisites, pre-capture same-function-object nested/outer mutation and readback, exact B refusal, latch, next maintenance and helper refusal. None was executed or authored here. All r44 directed/runtime gates, original P211 pending/unrun labels, historical r42/r43 RED, PID229 UNKNOWN, actual74 requirement and broader native/platform/installed/durability/release HOLD remain unchanged. No runtime, collection, Docker, DB, native campaign or cleanup was performed or authorized by this response.
