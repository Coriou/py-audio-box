# P1B23 directed fix independent review r85 Sol

2026-10-24. **Bounded SOURCE GO for the exact r84 successor; R83-1 and R83-2 closed. No new actionable finding in this finite delta. Next gate is Astra targeted final disposition, not operations.** No helper, target, pytest, product import, Docker or build was executed. Historical first child predicate remains UNKNOWN; campaign remains 17 PASS / 017 FAIL / 82 UNRUN. R72 original-positive assessment remains separate. Native T2 UNKNOWN; PASS32 UNVERIFIED.

## Identity and review boundary

Read FULL root r83 original findings and positive assessment, r81 Astra, r82 implementation and r84 responses. Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-bb323d30`, clean at documentation HEAD `faf0ef25cb6c5cc6a292065bf130bfdbfe6e7e2f`; source `72ef098537ed8c6f0ae5f6c584ca57ecc3c7e241`; r82 documentation base `db5df551c04d4c05144516c96fe47accf65ba5ac`. Source-to-documentation delta adds only r84 report. No applicable AGENTS.md found in the checked ancestor paths and checkout. Memory registry supplied general evidence-boundary guidance only, no task-specific premise.

Independently verified SHA256:

- Guard `cb12d9a66f5a44f875ef9ffd3fc9197889f147a9d29466d4c7082ccb57e7f26e` (unchanged).
- Controls `e7a310859cb94236dbf1219d0a255bb19fb4aa0a299d590c3256eda7c69a6f5b`.
- Runner `9efe46ccd59f458eb51918cc48fe0ed96e9a46fc63fbef82e96420edb580c974`.
- Manifest `6e63a53063901844c53104153fe06d1928eb050d2053b91261000616636f7bd8`.
- Target `tests/service/test_recovery.py`: `c236e4045a0c35836fc269fd5938c053ab3e97f1f2682f023470afe3a68dd211` (unchanged).
- Product `service/recovery.py`: `c314b2423b71c7d205b4558aee44d8c895e41394adbbb05a85458ef9bd7acf29` (unchanged).

All 197 manifest mappings independently match current bytes. Exact all-file base-to-source diff is three files, 36 insertions/15 deletions: seam semantics, runner CONTROLS digest alone, manifest controls/runner hashes alone. No guard, target or product delta. `git diff --check` passed. Outer controls module AST is identical after replacing only the seam literal; unchanged nested seam functions are AST-identical. Changed existing functions are observe, enclosing directed and publish_outcome; snapshot_complete is the sole new function. This review retains r83's established base assessment rather than reopening that base as a new platform audit.

## R83-1 closed

Controls lines1288–1323 preserve the original status assertion's equality and truth consumption once, inline in the observe frame. Diagnostic status rejection publishes only after that captured false result and then asserts it. The non-diagnostic branch retains the literal original assertion. Expected-witness selection at1301–1307 remains after status and before armed/witness evaluation, with the same alternatives.

At1310 the diagnostic branch consumes armed truth once. Only its true branch evaluates witness equality and consumes that result's truth once at1312. False armed records false and leaves observe_witness at its initialized NOT_REACHED; it never indexes/compares witness. Captured primitive booleans drive publication/assertion without re-probing the operands. Comparison or truth exceptions escape directly: no wrapper catch, forced exception IO, fabricated return or finally discharge. Marker stores do not move evaluation to another frame. The non-diagnostic armed-and-witness assertion remains literal at1323.

Observe0 remains exactly once at1324, after both prerequisites. Independent AST comparison confirms the entire tail from that assignment through return1325–1332 is identical to r82: original-return identity assertion, receipt/ownership assertions, existing evidence and return. There is no early call, receipt creation, ownership removal or acceptance mutation in the change. Added captured assertions retain ordinary refusal behavior; this is single evaluation of original operands, not a claim that diagnostic bookkeeping has zero timing cost.

## R83-2 closed

Controls1187–1193 define a read-only closed completeness predicate for the actual fixed primitive snapshot shape. It rejects each of four explicit overflow flags, any failure outside ESCAPING_B/MISSING_RECEIPT/ARMED_B, any shouldstop outside FALSE/MATCH_FIRST/MATCH_LAST/MATCH_BOTH, and any receipt status other than exact integer9/18944 (including bool). At1220 it applies independently to final and pre-assert teardown snapshots. A clean final snapshot cannot mask teardown OTHER/overflow, and the converse also rejects. Missing snapshots reject.

The unchanged remaining condition still requires callback presence/validity, no diagnostic IO error, no callback OTHER, no duplicate/unknown phase, and complete recognized phases. Recognized failing phases with pytest exit1 can be COMPLETE: the observation's completeness does not turn pytest FAIL into PASS. No failure list, shouldstop, receipt, owned set or pytest result is modified by the validator. This is the requested fixed-shape validation, not a general external-input schema expansion.

## Preserved instrumentation scope

The guard is byte-identical to the reviewed r82 guard, preserving all15 final gates,24 hook gates, thirteen first trace sites, original predicate order/catches/cleanup, stored callback selection and first-site stickiness. Every unchanged nested seam function, including final/terminal/clear wrappers, is AST-identical. The original final/terminal calls, prerequisite assertions, default diagnostic absence, primitive-only snapshot copies and complete clear remain as previously reviewed. No new retained witness/frame/exception root appears in this delta.

The three fixed diagnostic leaves and existing atomic4096-byte bound, callback fixed schema/enums, NOT_REACHED defaults, bounded failure/status/count/phase handling, exact failure forms, late publication points, missing/OTHER/IO rejection, teardown suffix demand and original acceptance assertions remain unchanged except the two corrections above. No wait, probe, clock, signal, retry, trace installation, threshold change, suffix waiver or generic exit74 acceptance was added. Existing successful-path evidence IO remains; refusal IO can delay child exit and affect later parent scheduling. Unexpected instrumented PASS still lacks required refusal evidence and remains diagnostic RED/missing; it cannot resolve historical causality or authorize retry.

## Retained HOST evidence inspected, not rerun

Independently verified `/private/tmp/pab-r84-host-20261024/inventory.json` SHA256 `78c34a59f0f4ea549972c5f126caa7e44078203b1a6e366eb29fd6f39301d133`: exact five inventoried leaves plus self-excluded inventory, matching lengths/hashes/modes, private root0700 and inventoried leaves0600. Completion binds checks.py, four current source/provenance hashes and raw stdout/stderr lengths/hashes; completed=true, raw OS exit0, empty stderr. Raw stdout contains307 checks and197 mappings, each mapping independently reconciled to candidate bytes.

Read the complete checks.py as source without executing it. It compiles/extracts the actual observe wrapper rather than a rewritten behavioral substitute. Inert equality results have separate truth counters/exception points. Expected exact event lists cover success, false status/armed/witness, comparison exceptions at status/witness, and truth exceptions at status/armed/witness. They require zero witness comparison after false armed, one comparison/truth when reached, no later evaluations after exceptions, and observe0 only once after success. Refusal checks require one publication for false operands and NOT_REACHED witness after false armed. This is meaningful evidence for the directed correction.

Actual extracted snapshot_complete/publish_outcome receive fixed primitives. Each teardown-only overflow and OTHER failure/shouldstop/status is isolated with clean final, then each converse case is checked. Recognized call/teardown FAIL with exit1 remains COMPLETE; distinct snapshot retention and unchanged exit1 are asserted. No nested pytest or real target is invoked by those checks. Count307 is bounded HOST instrumentation evidence, not307 runtime invariants. Prior647 evidence belongs to r82's preserved base; it was neither rerun nor claimed as successor execution.

## Disposition and limits

No further source correction requested for this exact finite successor. Carry r83 original findings, r84 responses and this closure to Astra targeted final review. Any subsequent one-node Linux diagnostic still requires separate parent authorization and exact source/image/selection/envelope/capture/quiescence/retention binding. No build, operation, campaign continuation, new wait, retry, cleanup or lane release is authorized here. No native/platform/product/release credit follows from source or inert HOST evidence. Old objects, proof roots and lane remain held.

Only this root report was written. Candidate stayed read-only; no commit, delegation, Docker/build/pytest/product import or selected helper execution occurred. Independent inspection used Git reads and read-only stdlib AST/hash/JSON/filesystem operations.
