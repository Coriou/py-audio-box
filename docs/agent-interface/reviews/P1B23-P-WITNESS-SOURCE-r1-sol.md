# P1B23 family P — independent Sol-low SOURCEONLY r1

2026-10-23. **SOURCE NO-GO for runtime readiness** on `478953d0cfa500d849c5222260a5fef05a8db2c8`, frozen executable/tests `d00f49f9a16f3f0eac11ba4c62622ec1278cc176`. Three blocking source/control findings below. This is not a runtime rejection or acceptance: no test, collection, candidate import, SQLite open or Docker action occurred. No prior P-review findings were supplied; none are silently waived. H/S receive no credit.

## Authority and exact provenance

READONLY candidate ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-1cebb715`. Sole repository write is `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23-P-WITNESS-SOURCE-r1-sol.md`. No fixes, commits, integration, delegation, native work, process experiments or receipt mutation. Host Python was restricted to AST parsing, compile without execution/pyc, hashes and Git comparisons. No runtime resource was acquired. Existing STT untracked artifacts, other active family worktrees and historical proof were not changed.

Candidate porcelain was empty at entry. Freeze-to-HEAD is exactly 63 added handoff lines in RECOVERY-P1B2.md. Historical source is `c224a4aaedf4c15765c8a19540a77f2f221a59c5`, executable freeze `65eef52fe7579b7d357719134a04c47fcb384b2e`; historical ABS t3-eb791eef was independently inspected clean at that HEAD. Root report checkout observed HEAD `f8612c38c1d66722731897f8aa7bc7ab188c8ced`, with preexisting STT untracked paths; it is not the candidate. No applicable AGENTS.md was found in scoped directories/checked ancestry.

Reviewed appended P handoff, orchestration protocol and ledger chronology; original GROUPED-WITNESS-CAPACITY-ASTRA P/shared rules, S1-CAPACITY-REVISED-ASTRA shared boundary, HISTORY-REFUSAL-WITNESS-AUDIT P1/census and existing LOTP-REHEARSAL-WITNESS-r1-sol. Prior reports supply contracts and attributed historical facts only. Historical PID161/PID255 inner exceptions remain unproven. The original S diagnostic remains failed; the revised early-S contract and missing genuine late-completion engine-capacity dimension are separate from P.

## Blocking findings

### F1 — operation-code validation controls arming, permitting legacy fallback

Location: tests/_maintenance_wait_guard.py:286–290, 509–516, 634–641; tests/service/test_child_wait_guard.py:533.

`eligible` includes `operation_code is prepare_code`. `guarded_run` uses that same boolean both to arm the parent veto and to install observation. Therefore an invocation of the designated collected P test with the installed collected finish function, but a changed outer operation code, is **unarmed**. If that changed operation raises ValueError early, observed_operation/session emit legacy V phases and observe can accept exit74. No P completion, seam or final is required. This contradicts arm-before-risk/missing-identity-fails-closed and the explicit prohibition on rescuing an eligible P failure with V. The actual replay must be excluded by its real invocation/installed-operation identity, not by allowing every malformed operation to become legacy.

Static reproduction path: preserve exact P node, collected object and installed finish_then_lose; pass a different operation code which raises ValueError before calling it. Line289 makes eligible false; line515 sets armed_P false; observed_operation emits V:operation;, observed_session emits V:session;; observe's legacy74 branch accepts V. This is a reasoned source counterexample, not executed evidence.

Minimal correction: distinguish designated invocation arming from validation of its operation/code identity. Arm the intended preparation before faults; treat mismatched/missing operation identity as invalid armed evidence. Preserve actual `_kfifth_fresh` checked replay as unarmed exit0 using a positive invocation distinction. Add a control changing the **actual operation callable**, raising early ValueError, and require armed/rejected74, no seam/final, latch and next-launch refusal. The current wrong-operation-code control only assigns `self.prepare_code` inside an already armed seam wrapper; it cannot detect this counterexample.

### F2 — cleanup controls substitute predicate arguments inside final observer

Location: tests/service/test_child_wait_guard.py:537–554; tests/_maintenance_wait_guard.py:584–591.

cleanup-value/nonsemantic/new-object and changed-route raise and catch entirely inside the replacement `at_exit_P` method. They call original_exit with the caught object and supplied caller, then return to observed_exit. For the three replacement cases the actual runner's active exception remains the original BrokenPipeError when this helper returns; the original runner catch-all never receives the replacement. Thus these are predicate-level argument/unwind controls, not the required real replacement propagation to the production catch-all. The V marker is manually emitted inside this helper. The comments/handoff overstate this as actual cleanup-boundary coverage.

Changed-route really mutates the original traceback by raising it through the helper, but still calls the predicate from the helper; it does not establish the complete real outward route at final observation. No arbitrary route allowance is requested.

Minimal correction: induce a bounded test-owned post-seam unwind which escapes to the original runner's catch-all, with original observed_exit reading sys.exception() itself. Do not replace at_exit_P to supply its exception/caller. Cover ValueError with a real legacy marker, nonsemantic replacement, fresh same-type object, and same-object changed route. Assert seam present/final absent, actual runner74 and rejected terminal, identity of final replacement/route in bounded test evidence, restoration and next-launch latch. P's result-write occurs after session context exit, so a session-cleanup injection before result writing cannot substitute for a post-seam control. No production cleanup change is authorized.

### F3 — most fresh negatives do not establish their directed premise

Location: tests/service/test_child_wait_guard.py:523–535, 559–590, 625–644.

For 25 non-cleanup/route negatives, the common outer assertion is merely any guard failure/session stop. A generic bootstrap/fixture error or an unrelated unwitnessed74 can satisfy this. There is no per-case assertion that mutation occurred at the intended seam with all preceding bindings valid. The normal positive is valuable, but cannot prove each altered fixture reached its own predicate.

Specific gaps: bad-offset/frame-budget/bad-slice mutate live f_locals without checking readback or that the original seam rejected exactly that altered fact; pipe-mode mutates saved witness identity rather than actual pipe access mode; wrong-final-caller replaces a predicate argument. precompletion and operation-failure both replace continue_kernel with the same early function, so the latter does not separately test failure of the actual outer operation. The FD cases do perform real dup2 replacement of the runner-local number, but never assert old ownership, changed device/inode/access identity, success of replacement, temporary-FD closure or preserved witness identity. An exception during these mutations also yields a generic rejection and can count as the directed case.

Minimal correction: add bounded, case-specific reached/mutated/rejected evidence and assert expected terminal phase/status/count and absence of continuation. Verify actual FD old/new identities and owned cleanup around dup2; retain no additional reaper/wait. Separate integration controls from deliberately predicate-level controls and supply the required real-route representatives. Keep foreign-owner/missing-done as aggregation negatives with their exact expected failure reason. Do not count any generic missing-receipt rejection as proof of another predicate. Required later finite test execution must demonstrate these premises; untested source is not waived.

## Positive route and preservation assessment

The original P target at test_recovery.py:6430–6453 retains every original assertion AST (multiplicity included), original child_failed refusal, real continue_kernel with prepared=r._snapshot before os.write assignment, sentinel, exact one-recovery/five-step SQL counts and observational replay byte/logical equality. Added snapshot consumption and exact durable ID SQL corroboration are appropriate. No source-confirmed impossible positive route was established: do not interpret that as proof the trace/offset/context-manager route works on the pinned runtime.

The witness captures production runner/continuation/lambda code before fresh controls, resolves actual nested collected finish_then_lose/broken code objects, uses runner-local pipe enrollment rather than a hardcoded FD, and captures the raw writer before the test patches os.write. At continuation return it requires real completion, consumed prepared snapshot, same session/intent, SQL1/5 and matching returned recovery ID. Finish return and seam recheck sentinel ID. At seam it requires exact built-in BrokenPipeError, args ('modeled lost result pipe',), errno None, actual runner write and broken raise offsets, bounded bytearray/offset/memoryview slice and exact enrolled pipe identity distinct from witness. Final requires same object/type/args/errno, actual runner caller at final _exit call and the complete stored two-frame/tb_lasti route. These are substantive predicates; F1–F3 concern their mandatory arming/validation coverage, not a recommendation to weaken them.

At terminal observation armed P rejects exit0 and any signal including synthetic11, irrespective of expected_exit/expected_signal. Transcript parser rejects missing, duplicate, unknown, reordered family tokens and malformed/truncated data. Finally restores the previous trace and clears exception/frame/session-bearing references. There is no new global registry, wait, reaper or signal operation. Exact raw kernel result tuple, first-owned fork/PID binding, failure latch and stop/next-launch checks remain structurally preserved. The normal fresh fixture includes actual original target and real `_kfifth_fresh` replay; replay's operation is not the preparation lambda and therefore stays unarmed in current source. Preserve this positive behavior while correcting F1.

Independent Git **mode/blob** comparison against exact c224a4 shows only three non-document paths changed: guard, appended guard controls, P target file. All production/worker/dependency/runtime/launcher/conftest bytes are identical. An initial filesystem-byte comparison followed three existing .impeccable image symlinks and compared targets to Git link text; mode/blob comparison resolved those apparent differences, with no image/source change. No image asset was edited.

AST-delimited byte comparison: all266 unrelated recovery top-level definitions and all top-level statements/assignments are identical; only P target differs. Guard's 11 other preexisting top-level definitions, including entire R class, aggregate/bootstrap/fresh_run/receipt helpers, are byte-identical; only MaintenanceWaitGuard.observe and install differ among preexisting guard definitions. __init__/event/fail remain identical. Existing guard-control file is a byte-exact prefix. Nested wrappers outside P additions retain their original logic; no process/receipt schema or policy amendment is credited. Freeze-to-HEAD has no executable delta; git diff --check passed. Three changed modules and embedded P helper parse/compile statically; no candidate code was imported or executed.

Hashes independently recomputed:

- guard `c28448dfca0406157bd07a1f768e211791ca285569cc251d7f69d9ac828af877`
- recovery tests `c236e4045a0c35836fc269fd5938c053ab3e97f1f2682f023470afe3a68dd211`
- guard tests `cffbcf2e9bd75412e30f53be221b2ac58dbabc94f5653324db9f5b6bbb2ba62c`
- production recovery `c314b2423b71c7d205b4558aee44d8c895e41394adbbb05a85458ef9bd7acf29`
- service lock `4ad7f2d9e6d4f03b7abb22d4946524f4372b5277b7b80c186332720f490909ea`
- conftest `d967baee72fa7fba3cb8974ac0406b2f44231a1c9d49fb7e7c088016f07cc7a3`
- poetry lock `2885f7f62d5a4b4a483e001622ba1d0d5ec4110578971664ecced01fbdfeb553`

## Complete new control-source census

Static matrix is **1 original + 11 transcript + 30 fresh cases**, not collection or PASS. All control bodies were inspected, not just decorator lengths.

- Transcript11: normal; missing; no-seam; no-final; final-only; duplicate; truncated; overflow; wrong-order; exit0; signal11. These are synthetic observer/parser tests with exact tuple identity and latch assertions, not actual child receipts.
- Fresh normal: actual collected original test, durable SQL/sentinel, lost-result74 and checked replay0; requires exact phases and accepted receipts. No runtime proof exists yet.
- wrong-class/subclass/wrong-args/real-EPIPE: same nested broken CODE via FunctionType globals substitution; error factory returns altered exception, original broken raises it. Real-EPIPE is a constructed errno-shaped exception, not an actual kernel EPIPE. All need directed reached evidence under F3.
- wrong-code: replaces installed preparation with ValueError function; wrong-node mutates bootstrap node; wrong-collected changes its obj. These are invocation identity controls; wrong-node currently becomes legacy/unarmed and relies on the unwitnessed BrokenPipeError rejection, not proof of arm-before-risk.
- precompletion/operation-failure: both patch continuation to early BrokenPipeError; not distinct outer-operation coverage (F3).
- durable-id changes stored completed ID; missing-sentinel deletes actual sentinel during error construction. Both should prove actual completion preceded mutation and specific rejection.
- wrong-fd/reused-fd/witness-pipe: actual dup2 to /dev/null, unrelated pipe, or duplicate witness writer respectively; identities/cleanup not verified (F3).
- cleanup-value/cleanup-nonsemantic/cleanup-new-object/changed-route: helper-local raise/catch and predicate argument substitution (F2), not accepted as required real propagation controls.
- changed-attributes mutates actual exception args; wrong-final-caller substitutes caller argument. bad-offset/bad-slice/frame-budget mutate frame locals; pipe-mode mutates expected identity; wrong-operation-code mutates expected code after arming. These are predicate-level controls; they do not cover malformed invocation arming (F1).
- missing-markers substitutes V at observer; signal11 substitutes observer status only. No real signal is sent. foreign-owner rewrites event owner; missing-done suppresses done publication. Those four are observer/receipt negatives, requiring their own explicit failure classification, not actual fault evidence.

Fresh bootstrap installs guard before inner pytest; --noconftest avoids a second guard installation. Plugin patches delegate to already installed guarded_run, and the target later installs its own collected finish function. Captured continuation stays original for precompletion rejection. No fixture-order code-identity drift was established from source, but actual offset/trace behavior on CPython3.14.8 remains unexecuted.

## Token capacity and combination boundary

P exact tokens are 4+4+4=12 bytes. Its only possible current legacy semantic markers are V=1, V:session;=10 and V:operation;=12, each mark deduplicated: **35 bytes total**. J requires the distinct exact journal node/parameter; R fail_code requires its distinct exact collected rehearsal node; T/C require exact transport/limit/close nodes; D requires its actual declared test-frame binding. Thus these cannot add their long tokens to an actual P invocation. This uses predicates, not just family names.

For existing other node branches a conservative deduplicated envelope is R44+45+23=112; J34+23=57; backpressure at most one final T (max52), one C (max50), plus23=125; other T19+23=42; D at most2+128 plus23=153. These remain below256. P does not enlarge the existing read/record bound. Reserved future H/S short triples give12*3+23=59 only **conditional on those exact future token bytes, one-contract eligibility and preserved node exclusions**. Their source is outside this review, so the handoff's convention is not an unconditional combined-source proof. Combination must check actual token definitions, emission sites and shared observe/install resolution before credit; no H/S acceptance is implied. Oversize synthetic transcript rejects at parser level; current tracked_wait still truncates to its historical256-byte bound, making the production-emission bound essential.

## Unresolved limits and required handback

No runtime PASS, actual receipt, binary/library identity, Linux mechanism, cleanup-resource or native qualification was established here. Static compile does not prove bytecode call-offset sets or trace-event route timing on the pinned execution runtime. F1–F3 require a new frozen correction and independent source re-review before separately authorized finite execution. Do not repair in this review or proceed on the theory that later tests will probably expose the blockers.

Then retain original positive/replay, all directed controls with verified premises, required R/J/T/C/death73/V/reporting and ownership preservation selections; retain exact finite nodes/counts and actual owner/PID/node/raw wait/launch/ack/done evidence. Independent actual-receipt review and parent acceptance precede combining family commits; resolve shared observe/install edits explicitly and obtain full combined source review/grouped actual finite gate. Four-B2 and conditionally full service/release remain separate later gates. T2/C1295, final Astra, mandatory C1 `b2_pinned_launcher_context_unavailable`, B3/C1/C2/native Darwin V1/V2/V3/A1 inventory drift/all19 Linux mechanisms/ext4/Unix-VFS/WAL/FULL/SHM/peer-credential/installed and physical durability, Apple Silicon/browser/acoustic and activation remain pending. H/S active worktrees untouched; late genuine engine capacity remains unproven, not supplied by P/R/A/B.

Final handback: candidate clean at exact HEAD `478953d0cfa500d849c5222260a5fef05a8db2c8`; executable freeze `d00f49f9a16f3f0eac11ba4c62622ec1278cc176`; historical baseline clean at `c224a4aaedf4c15765c8a19540a77f2f221a59c5` / `65eef52fe7579b7d357719134a04c47fcb384b2e`. Sole write: this report. SOURCE NO-GO; no runtime acceptance.
