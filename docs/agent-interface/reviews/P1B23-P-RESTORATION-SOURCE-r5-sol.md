# P1B23 P restoration — independent Sol-low SOURCEONLY r5

2026-10-23. **SOURCE GO for readiness for a separately parent-authorized finite gate**, executable/tests freeze `3546b1dd35bd18342ed911747ba9dc6dfca35fef`, documentation HEAD `0f8c88c9f88d5d51f1d187eec34220d1903f7137`, against `c17af4fbd0ad88fee955c0d2455152e950e1bcc2`. No blocking source finding. **Runtime acceptance remains STOP / NO-GO.** No runtime was authorized or performed in this review. This is neither P acceptance nor integration authority; root production remains T1.

## Authority and provenance

READONLY candidate `/Users/ben/.txcode/worktrees/py-audio-box/p-restoration-r4-1-20261023`. Sole write is this named report in root `docs/agent-interface/reviews`; no root commit. Candidate porcelain was empty. Root HEAD observed `e70392d0e66d7b7d093aac2586897d78b7b87e38`, with preexisting untracked STT evidence retained. Checked ancestry and scoped recursive inventory found no applicable AGENTS.md. Read the orchestration protocol, relevant ledger chronology, original P/shared Astra requirements and revised S/shared boundary carried in r3, prior r1/r2/r3 findings and responses, R3-1 handoff, complete R4 runtime report and new R4-1 handoff. No prior objection is waived.

Only static text, AST, hash and Git inspection were used. No candidate import/execution, pytest/collection, SQLite open, Docker, native diagnostic, fix, delegation, push or integration. Host `python` was unavailable; the static AST/hash inspection used `python3` without importing candidate modules or writing pyc. No temporary evidence file or runtime resource was created. Memory-registry quick search supplied no relevant P review evidence; no memory-derived fact supports this verdict.

## R4-1 — CLOSED at source; actual closure pending

Locations: `tests/service/test_child_wait_guard.py:550–564,570–585,787–796`.

The two installed-function mutations now call `install_locked`, rather than putting `_prepare_abandon_locked` into the manual MonkeyPatch stack. The fixture captures the entry production object before target execution. Its local slot stores only the exact wrapper it installs. Restoration writes the captured production object only if the current attribute **is** that owned wrapper, then clears the slot. It never records or reinstalls the target's `finish_then_lose` closure as an undo value.

The three relevant states are coherent:

- Own wrapper remains installed: restore entry production identity.
- Target monkeypatch fixture already restored production: leave it intact and clear ownership.
- A distinct foreign replacement is installed: preserve it and clear ownership. The unchanged subsequent production-identity assertion fails, withholding `hooks-restored`; preserving foreign ownership does not certify clean restoration.

Thus the R4 reasoned resurrection path is removed: target fixture undo followed by plugin undo cannot reinstall a stale target closure, because the plugin undo stack no longer owns that attribute. Wrapper construction still copies the actual installed code/defaults/closure and replaces only test globals as before. Wrong-code still installs its actual replacement callable. No code/name equality substitutes for exact restoration authority; no global owner registry is introduced.

`request.addfinalizer(restore_P)` remains before the first collected assertion, binding or hook mutation. `patch.undo()` is still its first action; `restore_locked()` follows, before all identity checks and evidence publication. Exact node/nodeid/obj absence, trace, os.write/os._exit, P class hooks, continuation, preparation and runner assertions remain. Setup exceptions before yield and target failures after yield reach registered cleanup. Evidence publication failure occurs after restoration. Ordinary wrong-class still requires the exact child premise/terminal and a parent record containing `hooks-restored`; the prior missing record cannot pass.

The historical actual result stays **89 PASS / 1 FAIL / 120 NOT RUN** on the prior freeze. Its actual original P and fresh normal positives had P:a;P:s;P:f; and real replay0+0. Wrong-class reached durable completion and the directed mutation, rejected raw18944 with P:a only and latched next launch, but lacked hooks-restored. These are attributed R4 report facts, not newly audited live receipts or rerun results. The precise historical inner teardown traceback remains UNATTESTED; the source ownership explanation is not promoted to an observed traceback.

## Directed ordering and setup-failure control

Locations: `test_child_wait_guard.py:813–866,891–904,914–949`.

The sole added mode, `cleanup-ordering`, obtains the actual pytest `monkeypatch` fixture through request.getfixturevalue. Three inert function identities exercise own-wrapper restoration, foreign-owner preservation, and real target-fixture undo before the registered plugin finalizer. Each step asserts the current identity. The last scenario deliberately leaves the owner slot populated after target undo has restored production; later plugin finalization must leave production intact. Earlier scenarios empty the real fixture undo stack before proceeding.

This is a setup-only lifecycle representative, not execution of the durable target or proof of its historical teardown order. Its deliberate setup writer failure requires exact OSError(ENOSPC, 'P directed setup evidence failure'), no call phase, clean teardown, pytest TESTS_FAILED and helper exit87. Seven exact ordered labels must include the ownership premises, setup classification and hooks-restored. Outer checks demand no sentinel, receipts, owned children, guard failures, session stop, events or nested helper, plus matching launch/ack/done identity. Generic setup failure cannot supply that proof. Seven labels remain inside the existing24-label bound.

The original setup-evidence-failure mode retains its exact three-label requirement in its own branch. Both setup controls remain unexecuted on this freeze. Missing-done still directly installs its helper-local atomic suppression through bootstrap atexit; it is intentionally outside undo, and the finalizer excludes only that mode from atomic-identity checking. Its exact FileNotFoundError aggregation failure remains required. Foreign-owner receipt aggregation and foreign preparation-hook ownership are separate controls and are not conflated.

## Prior objections and preservation

F1 remains CLOSED at source: designation arms before risky execution independently of validity; missing/wrong collected identity and malformed actual operation remain armed-invalid, with veto before legacy V. The actual operation-value representative and positive canonical replay exclusion remain unchanged.

F2 remains CLOSED at source-design level: post-seam replacement escapes the callback toward the original runner catch-all; unchanged observed_exit reads sys.exception and its actual caller. Same-object changed route, new same-type object, ValueError and nonsemantic replacement remain distinct, with seam present/final absent. Predicate-only caller/frame probes retain their limited scope. No runtime propagation result is inferred here.

F3 and R2-1 remain CLOSED for the enumerated source gaps: exact collected request binding, deliberate absent binding, reached/mutated/rejected evidence, actual FD identity/access/readback/witness/temp-close checks and precise aggregation reasons remain. R3-1 remains CLOSED at source: finalizer registration precedes mutation and setup evidence failure cannot bypass registered cleanup. R4-1 is closed only at source by the ownership correction above. No generic rejection is credited as a directed pass.

Independent Git comparison shows exactly one executable path changed from c17af4f to3546b1d: the final appended P control in `tests/service/test_child_wait_guard.py`. Freeze→HEAD changes only appended RECOVERY-P1B2 documentation. Every other tracked path, including guard, recovery targets, production, dependencies, conftest, bootstrap, launcher, aggregation and schemas, is byte-identical to c17af4f. Historical c224a4→HEAD non-document scope remains exactly the original three P test paths. Text before the final P decorator is an exact prefix. No skip/xfail, assertion waiver, cap change or production cleanup modification appears.

AST parsing succeeded for all three P modules and the embedded helper. Original final-function outer assertion multisets retain multiplicity:34→35, zero removed; embedded92→104, zero removed. Old33 mode values/order are an exact prefix of new34; cleanup-ordering is the sole addition. `git diff --check c224a4a HEAD` passed. Static parsing is not candidate execution or runtime qualification.

Independently recomputed SHA256:

- guard controls: `e22095b26c7219fd093f335baeaad3d8c7271cdf2bc30cb8aeef74b3f77c27cd`.
- unchanged guard: `0f31e002ad780eb0cac694cb1a739f6807c6b7da89bc525a66a57c5fab436cd7`.
- unchanged recovery tests: `c236e4045a0c35836fc269fd5938c053ab3e97f1f2682f023470afe3a68dd211`.

## Finite selection and unresolved gates

Prospective **211 outer items, UNEXECUTED on this freeze**: originalP1 + other guard76 + transcript11 + fresh34 + R5 + J1 + transport18 + blocked/overflow2 + close28 + reap24 + predecessor1 + death2 + semantic/report8. Preserve the prior exact nodes/subsets/order: original P first, complete guard source order with cleanup-ordering after setup-evidence-failure, then the prior preservation sequence. Close subset remains child-read3/parent-write4 timings ×2 failure classes ×2 repeats; reap subset remains normal/cleanup wait ×2 phases ×3 errors ×2 repeats. No pytest collection or post-failure count adjustment occurred. Exact explicit argv still must be frozen by source enumeration before any separately authorized gate.

P12 plus deduplicated V23 remains35<=256 bytes; guard/emission/eligibility bytes are unchanged. All strict completion, durable SQL1/5/ID/sentinel, exact modeled BrokenPipeError args/errnoNone, enrolled result-FD, frame/slice and same-final-object/full ordered route requirements remain. No signal allowance, legacy rescue, extra reaper/wait or weakened restoration check was added.

SOURCE GO does not authorize runtime. Parent authorization, actual original/fresh positive and replay receipts, every directed and preservation result, exact owner/PID/node/raw waits/launch/ack/done, hook restoration, retained proof and owned cleanup require a new finite gate, independent actual-receipt review and parent acceptance. Prior positives do not qualify this changed freeze. Actual pytest ordering, setup/finalizer handling, CPython trace propagation, writable locals, bytecode offsets/final route and all previously unexecuted negatives remain obligations. First unexpected result or missing hooks-restored must STOP without retry or any-failure negative credit.

Shared overlap resolution, combined-source review/grouped actual gate, four-B2/full service/release, T2/C1295 and final global Astra remain pending. H/S receive no credit; genuine late-completion engine capacity remains UNPROVEN. Mandatory C1 `b2_pinned_launcher_context_unavailable`, native Darwin causes, A1/all19 Linux/ext4/VFS/WAL/FULL/SHM/peer-credential/installed/physical durability, Apple Silicon/browser/acoustic/activation remain pending. Historical PID161/PID255 and general V/J stale-primary remain unproven. Historical worktrees, raw proof and STT artifacts were not modified. No runtime hold or resource cleanup obligation was acquired.
