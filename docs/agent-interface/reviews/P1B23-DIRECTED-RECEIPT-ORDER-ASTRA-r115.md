# P1B23 directed receipt ordering — targeted Astra SOURCE design r115

2026-10-24. **GO for a finite baseline-only design; NO-GO for a generally transparent relocation across all39 B-S modes.** Select exactly `original74-real-observer` and `collected-control` for the integrated shared-record/deferred-child-IO and deferred parent qualification path. Keep the other37 modes' mutations, prerequisites, assertions, evidence and ordering unchanged. This disposition is ready for a separately commissioned Sol implementation and exact-diff review; it grants no implementation, HOST execution, build, LOCKONLY, invocation or lane authority in this turn.

Parent-accepted r114 retention/prefix evidence carries without another actual audit: current **17 PASS / 018 FAIL / 81 UNRUN**, with017 excluded and its separately accepted PASS intact. Historical **17 PASS / 017 FAIL / 82 UNRUN** remains historical. First child cause/status/predicate at018 UNKNOWN; nativeT2 UNKNOWN.018 is a nonmutating positive using InertObserver, with EAGAIN and timeout milestones but no final-baseline, terminal-reached but no parent qualification. Its ordinary timed JSON path differs from017. The nominal approximately0.30 real-second deadline makes IO perturbation plausible, not measured causality. No campaign acceptance, causal resolution or resource release follows.

## Basis and source boundary

Read FULL r114 causal proposal, r113, r98 and r106, including focused reads after combined-output truncation. Read candidate seam wrappers, mode inventory, guard terminal/observe/tracked_wait/observed_exit/escaping/clear, selected hook/trace/final instrumentation and target/reap source. Only text, read-only Git and file hashes were used; no helper execution, tests, product imports, Docker, delegation, operations or commits. Sole write is this report. No applicable AGENTS.md was found in checked ancestors or candidate tests. Memory registry supplied only the general evidence-boundary convention, no technical premise.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-9f2be833` is clean at docs `8765e1884b2846bfaef348f533c88a10891ab900`; diff from source `5b8e00341f0edee4751ecb60fb89649aa79b82c3` is the r107 report only. Current SHA256:

- `tests/_maintenance_wait_guard.py`: `d3a011ffcc50e785f39abba54f07b51fe733b132425fff8dfc9a9848c92d254d`.
- `tests/service/test_child_wait_guard.py`: `6fc82dc7bff6e7b4fad55e42f7eb0af0d0b884cd1e068ce08eef3f703c01fbb8`.

Below, C means that control file and G means that guard file at those pins. Source anchors: C1034–1046 inventory; C1115–1395 schema/evidence/observer; C1423–1470 observe; C1487–1590 admission/EAGAIN; C1591–1700 trace/final; C1701–1730 terminal; C1730–1829 cleanup/teardown/handback/assertions. G47–90 observe; G840–1308 hooks; G1309–1611 selective trace; G1613–1900 final; G1901–1946 terminal/transcript/clear; G1988–2038 tracked wait; G2145–2153 observed exit; G2228–2249 escaping/cleanup. These are source proofs, not per-mode execution results.

## Why broad transparent relocation is false

G1989 obtains the original wait result. G2002 calls terminal before appending its suffix; G2034 is the sole natural observe call. G47 removes ownership and appends the receipt before transcript rejection. C1728 can raise between these operations. G's escaping and missing-receipt failures, and the target's reap exception handling, make the placement observable. C1759's universal armed-B suffix cannot repair this interruption.

Three concrete counterexamples rule out a general transparency claim:

1. **direct-exit74 has a source conflict already.** C trace calls installed `os._exit(74)` immediately after accepted EAGAIN. Installed observed_exit still calls final; C final explicitly demands rejection for direct-exit74, so no F is emitted. C observe expects `A/E`. Yet C terminal's exclusion set omits direct-exit74, so it demands raw74 plus child EF plus suffix T. G terminal cannot produce T from child E/status74. Thus the intended negative receipt conflicts with an earlier positive terminal assertion. Moving that assertion after observe changes missing/unconsumed receipt into consumed ARMED_B refusal; suppressing it would weaken the test. Leave unchanged and flag as a specific future control-contract defect, not expected negative PASS.
2. **schedule9-no-f and schedule9-f are scheduling positives**, not negative baselines. Terminal checks `w.killed`, actual caller local `killed`, exact child E/EF and TK. Re-evaluating after clear loses the witness/caller authority; retaining a frame violates lifetime constraints. Removing the exception edge changes reap trace/kill handling. They need their own contract if ever changed and are outside this relocation.
3. **final mutation and trace controls need immediate preconditions.** Predicate-F performs original final TRUE, mutation/readback, then original final FALSE inside the active exception; replacement/reraise/trace-disabled inspect that actual exception route. Writer-failure changes the captured emitter only after original final succeeds. A generic deferred evaluator cannot preserve these active-exception, hook and caller semantics. No final/trace/EAGAIN assertion moves in this design.

Even moving only terminal/observe assertions on ordinary negatives changes behavior on an unexpected status or witness: existing wrapper failure stops before discharge; moved failure permits guard rejection/discharge and changes exceptions reaching traced frames. Therefore exact all-path negative refusal semantics cannot coexist with a universal lossless wrapper under this scope. This does not prove each negative fails in practice. Preserve them, rather than instrument37 more modes.

## All39 source matrix

Notation: A/E/F/T/K mean exact `B:a;`/`B:e;`/`B:f;`/`B:t;`/`B:k;`; O means exact `B:overflow;`. `74` means raw18944, `9` raw9. R means exactly one actual terminal receipt, owned empty, wrapper returns original result identity; no manufactured receipt. P means positive: no guard failures, shouldstop false. N means intended negative: ARMED_B present, shouldstop equals last failure, both existing next-launch refusals, nested target assertions complete/result0, one receipt. ESCAPING_B may additionally occur through unchanged invalid parent witness; N does not require it absent. Missing receipt or unexpected wrapper/prerequisite failure is RED, never N credit. S means setup-failure special contract. Entries state required source outcomes conditional on prerequisites, not predictions of actual status. All use the exact collected target identity; all except017 keep InertObserver.

| Index / mode | Mutation point and mandatory precondition/readback | Required receipt/transcript; failure family / ordering decision |
|---|---|---|
|017 original74-real-observer|No mutation; actual admission, EAGAIN, timeout, original final TRUE; real Observer|R74 AEFT; P; selected baseline deferral, existing shared mechanism|
|018 collected-control|No mutation; same prerequisites; InertObserver distinct real FD|R74 AEFT; P; selected baseline deferral plus shared reuse|
|019 missing-obj|After valid init, reconstruct node without obj; assert absent; reconstruct eligible/invalid|R74 A; N; unchanged|
|020 wrong-obj|After valid init, foreign callable; identity differs; reconstruct eligible/invalid|R74 A; N; unchanged|
|021 wrong-node|After valid init, foreign nodeid with original callable; unequal nodeid; invalid reconstruction|R74 A; N; unchanged|
|022 parameter|After valid init, nonempty foreign callspec; invalid reconstruction|R74 A; N; unchanged|
|023 wrong-code|After valid init, replace runner argument with foreign lambda code; invalid reconstruction|R74 A; N; unchanged|
|024 closure|After valid init, same observed code with replaced write closure; exact clone/local readback; invalid reconstruction|R74 A; N; unchanged|
|025 operation|After valid init, foreign operation code; invalid reconstruction|R74 A; N; unchanged|
|026 wrong-errno|probe_e TRUE first; mutate exception errno EPIPE/readback; original eagain FALSE|R74 A; N; unchanged|
|027 subclass|probe_e TRUE; BlockingIOError subclass with original traceback/type readback; original eagain FALSE|R74 A; N; unchanged|
|028 missing-eagain|probe_e TRUE; call original eagain, withhold returned certificate as False; no fake kernel event|R74 A; N; unchanged|
|029 fd-other|probe_e TRUE; original FD/pipe/nonalias checks; dup2 /dev/null, distinct identity/access readback, temporary EBADF then replacement FD closed|R74 A; N; unchanged|
|030 fd-reuse|Same FD prerequisites; close original then dup2 new pipe writer; identity/access and close readbacks|R74 A; N; unchanged|
|031 fd-witness-alias|Same prerequisites; dup witness FD onto result FD; assert alias and access, close readbacks|R74 A; N; unchanged|
|032 fd-read-access|Same prerequisites; dup new pipe reader; assert O_RDONLY/distinct identity, close readbacks|R74 A; N; unchanged|
|033 raw-budget|probe_e TRUE; replace raw with1048577-byte bytearray, adjust framed_id and inner view; length/id/owner readbacks|R74 A; N; unchanged|
|034 offset|probe_e TRUE; write frame offset=-1/readback; original eagain FALSE|R74 A; N; unchanged|
|035 view-owner|probe_e TRUE; inner view on copied bytes, assert different owner; original eagain FALSE|R74 A; N; unchanged|
|036 view-slice|probe_e TRUE; inner view length1/readback; original eagain FALSE|R74 A; N; unchanged|
|037 premature|Original final TRUE first; deadline +=1000000, assert not expired; second original final FALSE|R74 AE; N; unchanged|
|038 route561|Original final TRUE; traceback at actual runner RAISE_VARARGS line561, assert site outside raise_child; final FALSE|R74 AE; N; unchanged|
|039 raise-offset|Original final TRUE; raise_child={-1}, actual tb offset excluded; final FALSE|R74 AE; N; unchanged|
|040 call-offset|Original final TRUE; exit_sites={-1}, actual caller offset excluded; final FALSE|R74 AE; N; unchanged|
|041 absent-route|Original final TRUE; exception traceback=None/readback; final FALSE|R74 AE; N; unchanged|
|042 extra-route|Original final TRUE; prepend traceback on caller, retain old tb as next/readback; final FALSE|R74 AE; N; unchanged|
|043 reordered-route|Original final TRUE; prepend wrapper frame, assert not caller; final FALSE|R74 AE; N; unchanged|
|044 wrong-args|Original final TRUE; exc.args=('foreign',)/readback; final FALSE|R74 AE; N; unchanged|
|045 replacement|After original trace valid timeout/id check, raise distinct same-args ValueError; final checks active replacement ID/trace route; final FALSE|R74 AE; N; unchanged|
|046 same-object-extra-frame|After original trace valid timeout, extra(original exception); final checks same ID and extra code in route; final FALSE|R74 AE; N; unchanged|
|047 trace-disabled|After original trace valid timeout, sys.settrace(None)/readback; final verifies original exception ID and no trace; final FALSE|R74 AE; N; unchanged|
|048 writer-failure|Original final TRUE; captured mark write must be _RAW_WRITE; replace with ENOSPC function demanding exact F; closure readback|R74 AE; N; unchanged|
|049 direct-exit74|Original trace accepted EAGAIN; installed exit74; final must reject; no F|Intended R74 AE/N conflicts with terminal EF/T requirement; RED conflict, unchanged|
|050 actual-overflow|Original final TRUE; emit257 bytes, then normal final-accepted/F; no fake overflow flag|R74 O; N; tracked_wait bypasses terminal when combined>256; unchanged|
|051 schedule9-no-f|After original trace accepted EAGAIN, sleep1; no final required; actual owned kill/caller readbacks|R9 AETK; P; unchanged|
|052 schedule9-f|Original final TRUE; replace captured exit0 with delay checking74, after actual F emission; closure readback|R9 AEFTK; P; unchanged|
|053 previous-trace|Install previous_trace before fixture captures trace0; normal baseline prerequisites; exact restoration|R74 AEFT; P; unchanged, not selected|
|054 setup-failure|bind_hooks sees installed exit, then raises exact ENOSPC; classified invocation failure and restoration|No receipt/owned/failures/stop; nested nonzero; S; unchanged|
|055 fixture-undo|Target monkeypatch replaces framing callable then undo; exact before identity restored before run|R74 AEFT; P; unchanged, not selected|

For all admission/EAGAIN/FD negatives final explicitly asserts original final FALSE. The first EAGAIN wrapper uses probe_e with witness-dictionary restoration, then the actual evaluation; later EAGAIN retries bypass mutation/evidence through `seen`. Neither this deliberate probe nor predicate-F's deliberate two evaluations may be reduced to a blanket once-only rule. “Once” below concerns the moved baseline qualification expressions and the original observe call only.

All N rows retain C1757–1765's actual armed-B suffix/latch checks and C1803–1829's outer result/receipt/milestone assertions, including real-eagain-baseline for non-admission modes and final-baseline/mutation markers for predicate-F. These checks do not guarantee the modes can complete under every schedule. Overflow specifically has no terminal-wrapper invocation and no wait-delivered marker through that wrapper, which is another reason not to apply the baseline decoder to every mode.

## Finite selected contract and schema

Use an explicit immutable mapping in C: `original74-real-observer -> ORIGINAL74_REAL_OBSERVER`, `collected-control -> COLLECTED_CONTROL`. No prefix matching, positive-set inference or catch-all mode. Unknown control rejected before target setup. Diagnostic/deferral enabled only for those two keys. Observer selection remains the existing exact comparison: only017 uses real Observer. All other37 modes retain ordinary evidence and existing wrapper branches.

Keep G byte-frozen, including default-absent `_B_SURVIVAL`, child-role placement, all15 final and24 hook operands, short circuit/catches, trace callbacks, terminal and receipt logic. C already supplies `_refusal` for diagnostic_enabled; extending the exact selector supplies it to018 without a guard identity change. No source incompatibility requiring guard edits was found. Role is only marked at the child timeout trace point; earlier milestones without role do not become a false claim of child final entry.

Reuse exactly the64-byte BDS2 mapping, complementary pairs, five milestones, commit-last result, error/reserved bytes and decoder from r98/current C. Five-milestone sequence applies only to these baselines. Preserve WAIT_UNOBSERVED/no-read, coherent TRUE versus test PASS, FALSE/CAUGHT, NOT_ENTERED_RECORDED, INCOMPLETE and UNKNOWN/RED. Never infer final-accepted from TRUE. No new channel/protocol or generic event stream.

Keep schema2 child payload and existing three leaf names/caps. Parameterize callback, decoder and outcome `control` by the finite identity; validator must receive the externally selected expected identity and demand equality across all leaves, parent record, node and private binding. A record's self-declared identity is insufficient. No mixed017/018 or old schema acceptance. The bound source/validator revision identifies the changed exact parent shape; old frozen validators remain untouched.

Add one fixed parent `qualification` object, also cross-checked in outcome, with exact keys:

- `policy`: literal `BASELINE_AFTER_OBSERVE`.
- `terminal`, `observe_status`, `observe_witness`, `observe_return`, `receipt_owned`: each exact boolean or `NOT_REACHED` (unexpected type/evaluation exception is RED, never true).
- `observe_calls`: bounded integer0/1, with duplicate a hard failure, not truncation.
- `assertions`: `PENDING`, `PASSED`, `FAILED`, `NOT_REACHED`; `first_failed`: one of the five keys or `NONE`.

Existing primitive status/child_bytes/suffix/armed and witness flags remain actual observations. No frames, witness objects, exceptions, tracebacks, kw dicts or saved evaluators in this object. Retain bounded exact immutable operand bytes locally (child E/EF etc., witness <=256, suffix) until evaluation, then only primitive classifications/booleans. For primitive status74/child EF/suffix T, reuse once-computed comparisons for diagnostics and compound qualification instead of recomputing them for assertion. No decoder may manufacture a passing qualification from a guard receipt.

`diagnostic=COMPLETE` continues to mean a coherent diagnostic, not PASS. A completed record of failed qualification can be COMPLETE with test FAIL. Missing evaluation/observation/publication or unknown state remains RED. A positive PASS additionally demands all five qualifications true, assertions PASSED, full original result/milestones, no failures/stop, receipt1/owned0 and existing phases/exit0. A fresh private validator must enforce this distinction explicitly. Complement NOT_APPLICABLE must not silently become017 qualification: later preparation must bind the selected018 identity if diagnostics are consumed.

## Exact sequence, assertion timing and exception contract

1. Before target clock/fork, allocate/init the same mapping, parent fixed slots and private binding. Leave all child final/EAGAIN/trace prerequisites and assertion locations unchanged.018's evidence now uses the existing fixed child slots and deferred parent list; it still uses InertObserver.
2. On the existing terminal wrapper entry, record actual wait delivery as now; call original terminal once with the original caller/enrolled/child bytes. Do not wrap it in a helper that changes the authoritative caller. If it raises, propagate unchanged: no deferred assertion machinery calls observe. Delivered shared bytes may still be read later under the existing rule; that is distinct from an observed guard receipt.
3. After original terminal returns, capture/evaluate baseline terminal expression once from actual operands, preserving left-to-right `and` short circuit for qualification. Record terminal-reached. Return the exact original suffix even if the qualification is false. Append actual74-parent-qualified only when that expression is true; no false milestone. Do not assert in this timed callback. Existing diagnostic comparisons previously evaluated extra fields eagerly; primitive diagnostic classifications are observations, not a second predicate evaluator.
4. Unchanged tracked_wait appends the original suffix and invokes the patched observe exactly once at G2034. At that wrapper's original status/witness check positions, capture/evaluate actual status and armed/witness expressions, preserving each expression's short circuit. Since the prior terminal/status assertion no longer raises, later checks can now be reached after an earlier false: this is an intentional baseline failure-path change, not identical cross-statement reachability. Original observe receives the unchanged result/kwargs once, at this same natural call site. No finally, polling, extra wait, observe retry, direct ownership edit or synthesized receipt.
5. After original observe returns, evaluate original identity and receipt/owned assertions once against that immediate state; do not consult mutable receipt state at later teardown. Return original result. Mark exact-terminal-readback only if the original observe checks all qualify. If original observe raises, propagate it unchanged, keep incomplete slots and FAIL; do not call it again even if it already discharged ownership. Do not retain its exception object. Arbitrary operand-evaluation exceptions likewise propagate, with incomplete diagnostic; losslessness is not a guarantee against arbitrary Python/storage failures.
6. Let the original target finish, including marker, elapsed<2, RLIMIT and nonowning reap assertions. Let guarded_run's existing escaping checks, trace restoration, witness clear, pipe close and missing-receipt logic run normally. **Deferred qualification assertions run in directed fixture teardown, after the existing guard-failure/latch-refusal block, before the registered `finish` finalizer.** Take the pre-assert guard snapshot as now. Use a narrow teardown `try/finally` to ensure the deferred check executes even if the existing latch block fails; both failures stay visible in Python exception chaining/retained phase failure. No suffix waiver. `finish` remains pytest-registered and must still restore hooks and check weakrefs even if deferred checks fail.
7. At that point assert the saved five results in original logical order: terminal, observe status, observe witness, observe return identity, receipt/owned. Preserve first-failure ordering; a false raises, later assertion statements remain unexecuted as normal. Their already captured operands do not become reevaluated predicates. Missing completion is FAIL, with original failure preserved, not an accepted negative. Record first_failed/assertion state before raising. The existing post-pytest result/no-failure/receipt/milestone assertions remain intact. Handback publication/restore-binding/close still runs in its existing finally, including teardown failure.

This places failure enforcement outside the traced product frames and after natural consumption. Extra callback lines and removed exception edges are real trace/timing changes. G select_B enables local traces only on its selected code identities; added baseline bookkeeping is not a new authorized target/caller. G's hook identities, all selected frame/offset checks and local callback selection stay unchanged. On baseline failure, removing the injected AssertionError deliberately avoids the old parent trace site8/escaping/missing-receipt fallout. No claim of transparent failure trace is made; the contract is **natural original receipt processing plus separately mandatory baseline qualification**. Raising just after observe inside tracked_wait would still perturb reap/trace and is not the chosen design.

A significant boundary is raw9/E/TK or raw9/EF/TK: G's existing grammar can accept that actual receipt. The baseline terminal/status qualification nevertheless remains false, teardown MUST fail, outer result MUST be nonzero. Do not change G.fail or manufacture an ARMED_B latch merely to label that positive failure; accepted guard receipt and failed baseline are separate axes. There is no next maintenance launch between saved qualification and teardown in this one-target baseline. Existing negative latch semantics remain byte-preserved; this finite contract does not authorize reuse of a failed baseline guard for further work.

## Failure schema, without suffix repairs

| Situation | Required handling |
|---|---|
|Selected R74/AEFT, all captures true|No guard failure/stop; deferred assertions pass; remaining original assertions still mandatory|
|Selected R9/AETK or AEFTK accepted by original guard|Receipt1/owned0 may be valid, no guard latch required; saved positive qualification false; teardown/outer FAIL, never positive status9|
|Selected delivered malformed/incomplete B transcript|Original observe discharges then emits ARMED_B; existing latch/refused-launch tests run; baseline remains FAIL even if target assertions passed|
|Selected original terminal throws or wait unobserved|No observe from recovery machinery introduced here; original exceptions and guard fallout untouched; unknown/incomplete qualification FAIL|
|Selected original observe throws|No retry; preserve actual partial ownership/receipt state and RED/FAIL; no invented readback|
|Unselected N with prescribed transcript|Existing ARMED_B suffix and last-failure latch, both refused launches, result0, receipt1/owned0 required; optional original ESCAPING_B retained|
|Unselected prerequisite/wrapper failure, especially049 conflict|RED; never accept MISSING_RECEIPT/ESCAPING_B as a replacement for required ARMED_B; leave existing assertions intact|
|054 setup failure|Existing exact ENOSPC/nonzero/no receipt/no ownership/no guard failure/restoration contract only|

## Allowlist, finite HOST proposal and exit conditions

Next preparation, if parent commissions it, may edit only `tests/service/test_child_wait_guard.py` for finite selector/schema, baseline wrapper captures/deferred assertions, evidence reuse and inert validation. `tests/_maintenance_wait_guard.py` stays byte-frozen. Mechanical changed control/hash references in `diagnostics/b-controls/runner.py` and `diagnostics/p-one-node/source-manifest.json` are allowed consequences; runner logic/selection/caps unchanged. A fresh private validator may receive the exact finite identity and qualification shape/bindings. Reuse existing PAX, snapshots, inventories and minimal binding process later; no metadata recopy campaign, generic orchestration or lifecycle rewrite. Product/target/Observer, original15/24 predicates, trace authority, FD/receipt/clear logic remain pinned.

No HOST work was run here. Finite proposed checks use extracted actual source expressions/wrappers with inert originals and primitive stand-ins, never product imports or target subprocesses:

| HOST group | Required evidence |
|---|---|
|Inventory/nonselected preservation|Exact39 names/order and original indices; residual AST/control-flow comparison for all37 nonselected branches, prerequisite/mutation expressions, assertion order and ordinary IO; G byte equality and target/product/Observer pins|
|Baseline successful sequence, both identities|Single terminal and observe calls, same result/suffix/kwargs, receipt discharge before deferred assertion; status/witness/identity/receipt comparisons evaluated once at original state; teardown order relative latch and finish|
|Failure operands|Each of five false outcomes separately; terminal short circuit and armed false skips witness comparison; capture actual old operands then mutate inert guard before teardown to prove no reevaluation; raw9 with accepted grammar still FAIL|
|Original exceptions|Terminal throws: observe count0; no delivered wait: no shared read; observe throws before/after inert discharge: one call/no retry and FAIL; exceptions/evaluation/storage/publication failure cannot produce PASS|
|Trace/caller and cleanup|Extracted selected trace dispatch/exception branches plus counters show no added selected code or authoritative helper frame; fail enforcement after guarded_run cleanup; finish runs on every teardown failure; no saved frame/witness/exception; original latch block still executes before deferred FAIL|
|Negative contracts|Finite source/counter checks of seven admission, six predicate-E, four FD, eight predicate-F, three route/trace, writer/direct/overflow, two scheduling, previous/setup/undo branches against matrix; retain049 as detected conflict, not green expectation; no actual per-mode run and no shared negative decoder|
|Schema/storage|Existing64-byte interrupted/commit/prefix/error/role matrix reused for018 identity; mixed identities/versions, unknown/duplicate/missing milestones rejected; child-only writes, no parent first-site substitution; validator/serializer agreement including diagnostic COMPLETE plus test FAIL|

Inert counters establish only wrapper/order/validation behavior; they cannot certify Linux timing, real caller bytecode admission, actual weakref lifetime, fork visibility under kill or remaining control PASS. Source review must specifically verify the changed baseline exception contract and that no guard/product assertion or mutation was deleted. Do not rerun all old instrumentation tests merely to rename identity; reuse established checks where unchanged and exercise the new finite branches.

Finite handoff: Sol integrated SOURCE preparation and commissioned inert HOST checks, then fresh Sol exact-diff/schema/counter review. Astra only for a concrete unresolved design objection or explicit parent requirement. Stop if a proposed change needs guard stores, negative-order changes, weakened predicates, a new frame at an authority seam, or a broader protocol; report that exact conflict.049 remains a disclosed pre-existing source conflict and prevents promising an all39/all100 pass; it does not block meaningful018 baseline validation or justify skipping049 later. No additional general recovery audit is required to implement this bounded design.

After parent acceptance of source/HOST evidence, fresh source-pinned build and any bounded actual stage require separate authority. A later failure must be retained and STOP, with no automatic continuation019–099 or retry. r114's accepted receipt carry remains untouched. **All resources, proofs, images, containers, volume and lane remain HELD; cleanup NONE. This report supplies a finite design, not operational permission.**
