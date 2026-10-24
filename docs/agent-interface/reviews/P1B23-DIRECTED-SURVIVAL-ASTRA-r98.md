# P1B23 directed diagnostic survival — targeted Astra design r98

2026-10-24. **Bounded GO to SOURCE preparation of the finite integrated design below. No operational GO.** Implement survival and remove the selected directed callback JSON from the child timed path together; do not spend another actual attempt on additional late labels. This is a design disposition, not evidence that a successor works or will reach final-child before SIGKILL. No build, invocation, retry, cleanup, resource release or lane action is authorized here.

Parent acceptance of r97's bounded ACTUAL receipt/retention GO stands; diagnostic remains RED. Historical campaign remains **17 PASS / 017 FAIL / 82 UNRUN**, first historical cause UNKNOWN, nativeT2 UNKNOWN, PASS32 UNVERIFIED. All old/new resources and evidence remain held, including volume `pab-b-8e0d1637-3fa2-4bec-9c08-8e46c534d930` and the lane. Resource identities and private proof endpoints remain those in r96/r97; this review did not query their current runtime state.

## Read boundary and exact basis

Read FULL r97, r96, r81 and r86, recovering truncated combined output with focused reads. Inspected the candidate's guard construction/preparation, hook/trace/final/terminal/clear, tracked-wait/observed-exit/restoration, directed wrappers and publication/validation, full real Observer, original target and product child/reap paths. Read the exact private diagnostic validator dependency as text. No tests, helper programs, product imports, Docker, delegation, commits or lane actions. No applicable AGENTS.md was found. Memory registry search supplied no technical premise. Sole write: this root report; preexisting untracked material untouched.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-b285df32` is clean at documentation HEAD `4671cb115fa7c3b11f7d3e8e93e13a0b5d2b4835`; source `99adb8b718ae7a2479aaa350491993ef0abd60a9`. Source-to-HEAD diff is the r87 report only. Independently rehashed source:

- `tests/_maintenance_wait_guard.py`: `cb12d9a66f5a44f875ef9ffd3fc9197889f147a9d29466d4c7082ccb57e7f26e`.
- `tests/service/test_child_wait_guard.py`: `e7a310859cb94236dbf1219d0a255bb19fb4aa0a299d590c3256eda7c69a6f5b`.
- `diagnostics/p-one-node/one_node_observer.py`: `ee20c60ff9d5890ae6d9ceacf892443a4aea88dedf7a3952baf5bc98797de582`.
- `tests/service/test_recovery.py`: `c236e4045a0c35836fc269fd5938c053ab3e97f1f2682f023470afe3a68dd211`.
- `service/recovery.py`: `c314b2423b71c7d205b4558aee44d8c895e41394adbbb05a85458ef9bd7acf29`.

The 28 captures, 56 streams and 29 PAX paths are accepted r97 evidence, not independently re-audited here. The existing image `sha256:1d00af1660042b2e0d4de584cad9718aeebd4883c9b4ab47ad692403a4e98c35` certifies no future edits.

## Observations versus inference

Accepted retained evidence proves real EAGAIN, accepted EAGAIN and child558 timeout milestones. Observer stage1 proves the child reached `observed_exit` and executed the sample1 record write, before `final_child`. It does not prove sample1 returned or final entered. No child-refusal file or F token survives. Final may not have entered, may have been incomplete, or may have returned false/true without surviving publication; which occurred remains UNKNOWN.

Parent raw9/child E/suffix TK precede the directed positive assertion that interrupts return to tracked_wait/observe. ESCAPING_B/MISSING_RECEIPT, owned1/receipts0, and later parent trace site8 are consistent with that interruption. Site8 is not the first child refusal; a later parent invalid snapshot is not the original terminal's prerequisite state. Preserve this distinction.

Source establishes ordinary witness-dictionary fork copy-on-write and late child JSON publication. Neither survives child death as a parent-readable committed observation. The 100x target clock gives nominally about 0.30 real seconds for the 30-unit product deadline, with no child publication grace. Directed JSON temporary-file/write/close/replace work and Observer regular-file writes definitely occur in that timed path. Their contribution to this kill is unmeasured. IO competition is an engineering hypothesis, not measured timing causality. The original positive passed without these directed wrappers; this does not identify their failing predicate or guarantee that removing IO will restore PASS.

## Selected mechanism and ownership

Choose exactly one **64-byte anonymous MAP_SHARED mmap**, created and initialized in `_B_SEAM_PROGRAM` setup before nested pytest invokes the target, hence before its clock/fork. Use Unix `mmap.mmap(-1, 64, flags=mmap.MAP_SHARED | mmap.MAP_ANON, prot=mmap.PROT_READ | mmap.PROT_WRITE)`; resolve the platform's anonymous flag alias at setup if necessary. No backing pathname, shared-memory name, file descriptor, resource tracker, pipe, semaphore, lock, ctypes atomic, manager or platform abstraction. No `flush`/fsync/msync. The logical record is 64 bytes; OS page rounding is not a claim of a 64-byte physical allocation. Unsupported mapping/setup is a preparation failure, not a fallback to private memory.

Darwin can supply HOST synthetic evidence for this Unix mechanism. Linux ACTUAL evidence is still separate and later. Do not claim a Python memory-model guarantee, hardware atomicity, or universal kill consistency. The design relies on ordinary coherent Unix shared mappings and a terminated writer before reading. Redundant encodings and commit-last detect interrupted/inconsistent observations; they are not memory barriers or a proof against arbitrary store visibility/reordering. If such an unconditional proof becomes a requirement, this scope has a concrete conflict: Python byte stores plus no synchronization cannot supply it. Stop rather than add a synchronization framework.

Storage owner is the directed control, **outside the witness dictionary**. Add only private guard-module bindings, default `None`, temporarily installed/restored by this diagnostic fixture. The binding references the mmap, not a mutable recorder callback. Do not attach mmap/view/recorder/function/frame/exception to `_refusal`; it retains primitive fields only. A primitive child-role flag may be set at the existing runner trace point where its already-read `local['pid']` is zero. It starts false before fork; parent never becomes writer. No new getpid/FD/clock/identity probe is needed in guard instrumentation. Validate this placement against every earlier child rejection path; failure before a child role can be established remains unknown, not a fabricated child site. Existing control `evidence` already distinguishes process roles; preserve its existing role observation rather than add another probe.

Only child writes record payload after fork. Parent initialization is over before target begins; parent must never mirror its later site8 into the record. No guard callback reads the record to determine acceptance. First-site stickiness uses the existing primitive child-local slot, not parent-shared authority. Module global bindings do not replace `self.trace_callback`, its stored bound `select_B`, or any cached local callback. Keep their identities, return selection and installation unchanged.

Parent reads a single immutable bytes copy only after an original terminal wait has actually been delivered. Record that delivery at the existing terminal wrapper (which is called after `wait` returns), without an added wait or observe. Prefer copying/decoding/serializing after nested target/pytest completion in the existing handback finally. If that original delivery was never observed, emit WAIT_UNOBSERVED/RED without inspecting mutable child bytes; no compensating wait, signal or polling. Capture delivery independently of the later original-terminal return/positive assertion, so the already-reaped raw9 case remains readable even when the wrapper asserts.

After copying, restore the private guard binding and close the parent's mmap in guaranteed finalization, including setup/pytest/publication failures. No memoryview export should exist to prevent close. Child process exit owns its mapping teardown; do not add a child close callback. Existing original clear still empties the entire witness dictionary, including all primitive diagnostic references; existing hook restoration, weakrefs and collection assertions remain. No saved closure/global/snapshot may root the witness, operation, exception, traceback or caller frame. Parent bytes and decoded integers can outlive clear. Closing the parent's mapping is local diagnostic resource disposal, not permission to clean held operational resources.

## Concrete 64-byte layout and writer rules

Use a fixed version2 schema, exact byte offsets, zero-filled reserved bytes and no strings/pointers/PIDs in shared payload. The table is an implementer contract, not a new transport protocol:

| Bytes | Meaning |
|---|---|
| 0–3 | immutable setup signature/version, e.g. ASCII BDS followed by byte2 |
| 4–5 | CHILD_STARTED pair, value1 |
| 6–7 | FINAL_ENTRY pair, value1, written inside original final method before exception acquisition/gates |
| 8–9 | sticky first trace site pair, existing IDs1–13; zero pair means none recorded |
| 10–11 | last final gate-start pair, IDs1–15 |
| 12–13 | last nested hook gate-start pair, IDs1–24 |
| 14–15 | hook stage pair: 1 closure acquisition, 2 conjunct evaluation, 3 returned true, 4 returned false, 5 unwinding |
| 16–21 | one-shot result payload: kind, result, final gate, hook gate, valid, phase |
| 22–23 | one-shot result commit bytes, 0xA5 then 0x5A, last |
| 24–33 | five append-only milestone pairs in fixed sequence, enum1–5 |
| 34–35 | diagnostic error pair, finite nonzero enum; zero means no recorded error, not proof no error |
| 36–63 | reserved zero; reject any nonzero byte |

Pair encoding: absent is `(0,0)`; committed value v is `(v,255-v)`, restricted to that field's enum. For reused progress pairs, write complement byte0 first (invalidates old pair), then value, then complement last. A one-shot pair also writes value before complement. Do not reset sticky site, entry, milestones or result; no wrapping counters/ABA reuse. Any other pair is interrupted/invalid. A progress pair records the last **committed gate-start**, never a completed predicate or exact kill instruction. An older valid progress pair can survive an interrupted next update; do not claim no later gate was attempted. Entry has the same unavoidable execution-before-marker gap.

Result enums: kind1 FALSE, 2 CAUGHT_EXCEPTION, 3 TRUE; result1 false, 2 true; valid1 false, 2 true, 3 OTHER; phase1 ARMED, 2 BACKPRESSURE, 3 TIMEOUT, 4 OTHER. Zero payload is unrecorded. Preserve existing gate IDs/order: valid, hooks, trace identity, phase, exception identity, timeout shape, bound_raw, live_write, caller code, exit site, traceback present/frame/raise-site/sole-entry, expired. Hook IDs remain the existing24 conjunct order. FALSE must have gate1–15; CAUGHT_EXCEPTION permits gate0 for exception acquisition before conjuncts; TRUE uses gate0/hook0. Hook refusal detail applies to final gate2 only; distinguish closure-acquisition unwind/hook0 from a failed hook conjunct. A committed TRUE must agree with completed hook success and final progress15; no success claim from progress15 alone. Cross-check entry, role, result/kind/gates, milestones, error/reserved bytes, and monotonic legal stage combinations. Unknown or contradictory values stay RED.

Write result payload **inside original final-child**, from the actual existing evaluation, before return, with commit last. Commit a caught outcome only on the existing caught-exception route, after the original validity change; do not classify a diagnostic store failure as a predicate refusal. Isolate storage errors from original catch/return semantics with narrowly scoped inline protection and an error flag when possible; otherwise the missing/invalid commit is RED. Never suppress an original exception/assertion or use a diagnostic failure to set witness validity. This separation must be demonstrated in source/counter review, not assumed.

All gate-start stores are inline adjacent to the existing single operand evaluation. No Python recorder/helper frame while acquiring active exception, inspecting traceback/caller, or evaluating hooks. Builtin mmap byte assignment may be expressed inline; an implementation that cannot preserve operand order, short circuit and truth-consumption counts must stop. Do not duplicate `sys.exception`, hooks, expired, live_write, comparison or truth evaluations. Preserve the original exception scope and `finally` reference release. First-site stores occur at the existing rejecting branches only. Hook progress is enabled only during final evaluation. The existing `_refusal` primitive semantics can coexist; do not let a second instrumentation implementation become an alternate evaluator.

Parent classifications:

- Clean initialized header/role with absent entry and no later fields: **NOT_ENTERED_RECORDED** (display NOT_ENTERED only with this definition). This means no committed entry marker; it cannot prove the method never began before an interrupted marker store.
- Entry present without a valid complete result: **INCOMPLETE**; preserve validated gate/hook starts and milestones, but result is UNKNOWN. A torn entry likewise remains incomplete/unknown, not NOT_ENTERED.
- Valid committed FALSE or CAUGHT_EXCEPTION: the corresponding first observed source refusal, not its unobserved inner physical cause.
- Valid committed TRUE: original final result observed true; does not prove F was emitted, wrapper returned, exit74 happened or test passed.
- Bad header/role/encoding, contradictions, unknown wait, missing handback or publication error: UNKNOWN/RED.

NOT_ENTERED_RECORDED and INCOMPLETE are useful bounded observations but remain diagnostic RED. A coherent committed result may qualify result completeness separately from test status. No decoder can recover a predicate evaluated immediately before kill but not committed; no guarantee of final entry or publication is offered.

## Explicit integrated reduction of directed timed IO

This deliberately supersedes r81's instruction to preserve existing evidence IO for its first revision. It is an additional, finite source-preparation change justified by the now-proven survival weakness and known timed work, not a measured IO-causality finding. Enable only `original74-real-observer`; all other directed negative/positive/scheduling controls retain their current evidence, substitutions and assertions unchanged.

Replace this mode's child `evidence` JSON writes by five fixed ordered shared milestone slots: 1 real-eagain-baseline, 2 eagain-accepted, 3 actual-child558-timeout, 4 final-baseline, 5 final-accepted. Preserve exact call sites and original prerequisite assertions, duplicate prohibition and order validation. Unknown/duplicate/out-of-order calls must retain the existing assertion failure behavior and mark diagnostic error where possible; do not silently truncate. A partial/torn slot is not a milestone. Preserve all committed prefix entries if a later slot tears. Five slots suffice for the selected mode's exact source paths; keep other modes on the old bounded24 mechanism. Do not introduce a generic event stream.

Parent selected-mode milestones can stay in the existing bounded primitive list, with publication deferred until after target completion. Deferred JSON reconstructs only committed child milestones and actual parent calls, never inserts missing final-accepted or asserts their truth from a final TRUE. Retain the existing positive milestone assertions and source purpose; replace only their input construction for this mode. In particular failure before final-accepted must still fail existing result/positive checks, while the handback finally preserves its partial diagnostic. No fake child PID filenames: selected-mode child milestones can be embedded in the version2 child leaf and passed as the decoded list to the existing assertions; retain all other modes' numeric-file discovery unchanged. Parent publication failure remains visible/RED, with restoration still attempted.

Remove selected-mode late child `publish_diagnostic` and its unnecessary child callback snapshot/serialization. Guard-inline result commit supplies its data. Preserve the following `assert answer, 'actual74 final prerequisite absent'`, final-baseline and final-accepted calls. Defer selected-mode parent callback publication until handback where possible, keeping the actual assertion operands and pre-assert teardown snapshot in memory. Do not move original assertions, call observe early/in finally, discharge ownership, or manufacture a terminal receipt.

**Unchanged:** real Observer class/FD/sample1–4/47 fields and bounds, target actual-EAGAIN file and its read assertion, original target clock/deadlines/elapsed<2/RLIMIT/wait assertions, exact B grammar/256-byte cap, original suffix demand, every negative-control mutation/assertion, latch checks, hooks/restoration/lifetime checks. No SIGKILL-positive reinterpretation for original74. Do not use saved IO to widen a deadline, wait for a commit, or force predicate execution.

## Finite implementer allowlist and dependency

Successor only, never this frozen candidate or retained bundles. Executable tracked scope is exactly:

1. `tests/_maintenance_wait_guard.py`: private default-absent shared binding; primitive role/entry/site/gate/hook/result/error stores only at the named B instrumentation sites. No changes to preparation/projection, install, observed_exit, tracked_wait, terminal, observe, transcript, ownership or product behavior.
2. `tests/service/test_child_wait_guard.py`: selected-mode mmap setup/ownership/restoration, bounded milestone handling, wait-delivery flag at existing terminal wrapper, deferred serialization/decoding, version2 strict validators and focused inert checks. Preserve all original acceptance assertions.

Mechanical guard/control hashes in `diagnostics/b-controls/runner.py` and `diagnostics/p-one-node/source-manifest.json` are permitted provenance consequences only; no runner logic change. Bind unchanged target/product/observer hashes explicitly. Use existing report/provenance conventions; no new general protocol/framework.

One actual private dependency is known: `/private/tmp/pab-r94-bound-singleton-prep-20261024/diagnostic-host.py`, lines189–216, locates three diagnostic leaves and `diagnostic_complete` explicitly requires schema1 child FALSE/CAUGHT_EXCEPTION, final_called true/final_result false. Lines228 onward embed extracted schema1 callback validators. A fresh successor copy must narrowly adapt those leaf/validator portions to the exact version2 child layout and outcome cross-checks; the old frozen bytes remain untouched. Keep `child-refusal.json` as the existing bounded leaf name but explicitly define its version2 payload as child-survival observation, including true/unentered/incomplete; the filename is not evidence of refusal. Keep the other two leaf names,4096-byte caps and existing snapshot/overflow/phase safeguards. Do not silently make version1 accept version2 or label every retained record COMPLETE. WAIT_UNOBSERVED/torn/incomplete remain RED; committed true is distinct from refusal and from PASS. The parent/child version2 schema must be fixed by implementation and copied exactly to this validator, with rejection of mixed versions.

Reuse the existing lifecycle, PAX reader, capture, inventories and code/data snapshot binding. Only new source/validator digests and their mechanical private bindings/provenance are in scope; no broad lifecycle rewrite, envelope/cap change or cleanup/release path. Source guards/controls edits require a fresh source-pinned image later through the normal parent-authorized build/review sequence. Neither old image nor old approval authorizes new bytes. Source and inert preparation need no lane mutation.

## Bounded validation and exit conditions

No checks were executed in this design task. The implementer should provide exact diff/pins, enum-to-source mapping and finite SOURCE/HOST evidence covering:

- Strip only named diagnostic stores and compare all15 final/24 hook operands/order/catches, trace rejection sites, global/local callback bindings, original assertions, receipt/ownership and reference cleanup against the pinned source. Default-absent and every nonselected mode remain behaviorally unchanged.
- Actual extracted operand/wrapper counters for success, each reached false/caught point, skipped later gates, once-only truth/active-exception behavior and diagnostic-store errors. No cloned predicate as sole evidence; no real target/product import needed.
- Inert byte snapshots at every store boundary for entry, sticky first site, progress, result payload/commit and milestones; missing/inconsistent/mixed/unknown/overflow/OTHER and commit-after-true cases. No false completed result from interrupted bytes; no false first-child-site from parent fallout. Cross-field validation and legal-prefix preservation matter more than case counts.
- Allocation before target clock, child-only role, no read before original wait delivery, handback despite terminal/assert/pytest failures, restored private binding/closed mapping even if publication fails, no new witness roots. The existing real lifetime/cycle assertions remain required later, not credited by an inert surrogate.
- Source compatibility of every existing negative control and exact original assertion inputs, plus parent serializer/private validator agreement for false, caught, true, unentered, incomplete and unknown. Retained FAIL stays FAIL; no diagnostic completeness shortcut affects acceptance.

If separately commissioned for HOST preparation, a minimal normally exiting fork/mmap smoke check on Darwin may demonstrate inherited shared bytes with its own inert child and normal wait; it is not this target, not SIGKILL qualification, and adds no waits to the diagnostic. It must be reported HOST-only. Synthetic interrupted snapshots establish decoder behavior, not hardware visibility under kill. Linux actual mapping survival and execution timing remain unqualified until separately authorized evidence exists.

Finite sequence: **Sol implements this integrated SOURCE preparation; fresh Sol reviews exact successor and the private validator dependency; Astra targeted final only if that review leaves a concrete design/visibility/trace/lifetime objection or parent explicitly requires final disposition.** Close ordinary implementation defects in that finite review rather than commission general recovery audits or add successive label-only gates. Stop at source review if role placement, inline single evaluation, diagnostic-error isolation, reference release or validator binding cannot fit this allowlist; identify the exact conflict.

After source closure, the normal independently reviewed source-pinned build and any ONE017 actual invocation remain parent-owned, separately authorized stages. If later runtime shows NOT_ENTERED_RECORDED, INCOMPLETE, coherent true/false, unexpected PASS, publication failure or another RED, retain it and STOP. No automatic repeat, preceding17/original-positive rerun,018–099 continuation or B100. A useful partial survival observation does not establish historical cause or justify forcing a gate to run. **GO to bounded source preparation only; held resources and all operational authority unchanged.**
