# P1B23 S witness — independent source review r1

2026-10-23. **SOURCE NO-GO. Runtime readiness NO-GO; no acceptance.** The frozen S implementation has a positive-path trace-budget defect and directed controls that reject unrelated instrumentation before exercising their named condition. Return to the parent for a bounded source correction and fresh independent review before any family runtime. No test or actual positive was executed in this review.

## Authority and provenance

READONLY candidate ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-87bb88ab`, exact HEAD `b7d8ea32310846375ce4b960b7b8923485269a78`, executable/test freeze `742cab3ea9dbee0daa0a2f69b39db8942f2d6beb`. Candidate porcelain was empty on entry. Freeze-to-HEAD is exactly 150 appended lines in `docs/agent-interface/phase-1/RECOVERY-P1B2.md`, with no executable change.

Reconstruction `048fcbb9ac78f3bc238d425f0ad6b3011a94fdb9` independently matches every tracked non-document baseline blob/mode from `c224a4aaedf4c15765c8a19540a77f2f221a59c5`. Historical executable pin is `65eef52fe7579b7d357719134a04c47fcb384b2e`. Reconstruction-to-freeze changes only `tests/_maintenance_wait_guard.py`, `tests/service/test_recovery.py`, and `tests/service/test_child_wait_guard.py`. No production, dependencies, worker, conftest, launcher, resource policy or receipt-schema delta exists relative to that reconstructed baseline. Root/document chronology is distinct from this executable provenance.

Read the entire appended S handoff, revised S1 contract (authoritative for S), grouped shared rules (old late-S premise superseded), history audit/census, diagnostic failure boundary, orchestration protocol and relevant ledger chronology. Inspected the complete three-file narrow diff, current guard/bootstrap/aggregation, original target and production runner/session/preparation/transaction/rehearsal/insert routes. No applicable AGENTS.md was found in checked ancestry/scoped paths. Memory supplied only exact-checkout and evidence-tier guidance, not candidate or historical acceptance evidence.

Only host static AST parsing/compilation, source/hash comparisons and Git/read commands were used. Compilation was of ASTs without execution; the embedded future helper body was separately AST-compiled without execution. No pytest/collection, candidate import, SQLite open, Docker, native experiment, delegation, repair, commit or integration. No runtime resources were acquired.

## Blocking findings

### F1 — trace budget counts transient context-manager propagation and vetoes the intended positive

`tests/_maintenance_wait_guard.py:481–484` records **every** code receiving the retained exception and sets `bad_S` on the eleventh distinct code. These are not just the final traceback entries declared at `:510–513`. `bad_S` is sticky and immediately prevents final attestation at `:494`.

Minimal source counterexample is the unchanged positive itself after a genuine INSERT FULL. Its final declared route has eight codes on False and nine on True: insert, intent lambda, transaction, rehearsal, prepare, optional consume, preparation operation lambda, observed_operation, runner. Unwind also throws this same exception into the generator bodies of production `_maintenance_session` (`service/recovery.py:421`), `_locked_roots` (`:131`), and guard `observed_session` (`tests/_maintenance_wait_guard.py:667–673`). The contextlib generator context-manager `__exit__` receives/catches it while calling `gen.throw` as well. Those temporary exception events occur even when contextlib subsequently restores the original exception traceback and those frames are absent from the final route. Thus even False exceeds ten distinct observed codes; True has another code. The final ordered traceback may be correct while the observer has already set `bad_S=True`.

The 16-entry `route_S` limit does not solve this: it limits a different, current traceback representation. Nor is the dictionary actually capped at ten: the code sets the veto and then still inserts the new entry. The handoff's claim that exact final propagation is source-ready is unsupported by this implementation.

Minimal correction: record only the finite declared final-route code identities in the map used for final frame/offset matching. Account separately, with an explicit finite bound, for required transient generator/contextlib observations if they are needed at all. Preserve exact final order, native inner offset and same-object checks; do not merely widen the count until a run passes or accept a subsequence. Re-review trace/profile/contextlib ordering before runtime. This is a source counterexample, not a claim of an executed CPython result.

### F2 — named policy/SQL controls break the hardener route first and can pass vacuously

`tests/service/test_child_wait_guard.py:545–579` installs an extra `hardener_S` around the target's `harden_private_S`. The target already installed its wrapper before `altered_S` runs. Consequently the actual call chain becomes rehearsal → control hardener_S → target harden_private_S → captured production hardener.

But target `harden_private_S` (`test_recovery.py:6155–6173`) only configures when its **direct** caller is rehearsal, and the witness (`_maintenance_wait_guard.py:416–432`) also requires that direct parent. With this control wrapper present, target setup is skipped, `rehearsal_frame_S`/configuration identity are never established, and no S seam can qualify. The control's `sys._getframe(1)` check at `test_child_wait_guard.py:555` still sees rehearsal, so some named operations happen, but they happen on a path already invalidated by the wrapper.

Concrete examples: `preprepared` performs `SELECT 1` **after** `result=harden(...)`, rather than preparing the intended workload before hardening. `AUTH` attempts a forbidden setting, but has already lost the required setup route. `policy-bypass` removes the authorizer on that same invalid route. BEGIN/COMMIT/rollback/other-SQL/error-code/type/forged-FULL/later-phase/completion-entry rows additionally replace the captured insert function, so they cannot independently challenge its error/phase predicates. All ordinary negative rows assert only generic outer `guard.failures` (`:644–647`); they do not prove that their named phase was reached or that preceding bindings remained valid. A baseline early rejection is sufficient to make them green.

Minimal correction: retain the positive direct hardener route and instrument each intended boundary without interposing that invalidating frame. Add bounded reachability/precondition assertions for each control: valid collected identity and setup, named defect actually performed at the intended phase, and the expected seam/final disposition. A preprepared-workload control must actually prepare the workload before policy installation; AUTH must attest actual native AUTH at its intended post-hardening probe. Isolate code/type/phase predicates rather than changing several prerequisites at once. Keep these controls explicitly synthetic where applicable and never count them as native positives.

### F3 — the promised per-predicate directed matrix is incomplete

The adopted contract requires a directed representative for each new predicate. The 41-mode list at `test_child_wait_guard.py:490–500` does not provide this simply by its size. Missing independent challenges include native connection type/proxy or factory substitution; exact attachment/main-DB identity; private directory unsafe mode/ownership; main/WAL/SHM nonregular, wrong owner or multiple links; frozen raw/hash/plan-digest/session snapshot ownership dimensions; and a forged code identity (the existing forged-collected row changes a function object, not a code object). Wrong SQLite subclass is also absent: `wrong-engine-type` supplies DatabaseError, not an OperationalError subclass. F2 further leaves the alleged phase/workload/policy controls unqualified.

Several advertised controls are only counter manipulation: missing/duplicate config set `configured_S` to 0/2, wrong-hardener sets `hardened_S=0`, and wrong-order sets `authorized_S=0` (`:593–599`). They test comparisons, not missing/duplicate native configuration, absence of hardening, or ordering. They are correctly labeled synthetic, but that label does not fulfill the directed setup/policy requirements. `wrong-private` substitutes a nonexistent path, not an unsafe owned-directory case. `wrong-inode` changes the saved comparison tuple, not the database path/inode binding. There is no independent native-configuration occurrence observer: `configured_S` counts entry to a Python wrapper, not the PRAGMA execution (`_maintenance_wait_guard.py:419`).

Minimal correction: supply finite representative controls for the actual independent binding dimensions, with evidence that the intended predicate is challenged and unrelated prerequisites hold. Bind configuration occurrence/order to actual native calls and distinguish permitted readbacks/hardener setup from workload. This is a test-owned correctness proof; it need not claim security against arbitrary hostile Python. Do not waive missing controls by acknowledging them as synthetic or pending runtime. No new production policy is requested.

## Preservation and positive-source assessment

Independent byte/AST comparisons confirm:

- Every out-of-scope top-level recovery-test span, including decorators and assignments, is byte-identical to c224a4aa. `_t1_assert_unaccepted` is untouched. Only the selected capacity target changes.
- All original guard imports/top-level statements and assignments remain byte-exact; one new token assignment is added. Out-of-scope guard definitions are AST-identical. `ModeledRehearsalCompletionWitness`, `aggregate` and `bootstrap` are unchanged; no R, receipt, helper ownership, reaper or process-management redesign is hidden in the delta.
- Existing child-wait tests remain the exact 25,015-byte prefix. The appended two tests add 11 transcript rows and 41 × 2 fresh-control rows by source declaration, without collection or execution.
- Original target has four Assert nodes and the candidate seven, the three additions being private setup/readback checks. Existing parent ValueError, trap absence, receipt/source/unaccepted checks and True burn marker remain, with the explicit failure-string rename. The other five failure values × both booleans retain their original branches and assertions. The production `_t1_assert_unaccepted(roots,1)` still requires one immutable plan, zero recoveries/steps/dispatch resolutions, highwater20MiB, checking and no receipts.

The target explicitly replaces the unreachable huge registration INSERT with same `PRAGMA max_page_count=1` once at its direct rehearsal hardener entry, before captured unchanged hardening once. Positive source retains native sqlite3.Connection and exact production INSERT/intent argument construction, without proxy, UDF, factory replacement or injected exception. Before/after readbacks are exact 4096/61/0/effective61 and WAL/FULL/mmap0/autocheckpoint0. Private/source identities, prepared snapshot bindings and policy installation are inspected; code entry of captured completion/scan vetoes S once rehearsal is bound. The first private transaction must fail without a completed intent, receipt or scan. False burn is observed at preparation unwind; True retains its original consumer assertion. These are intended source properties, not actual engine or route receipts, and F1–F3 prevent readiness approval.

Arm is emitted before runner risk for the designated new failure; wrong/missing boolean yields an armed invalid contract rather than truthiness fallback. Fresh transport adds only exact typed bool for S. The observer vetoes incomplete S before V/J/R/T/expected74, exit0 and signals. Actual runner catch-all code/call offset, final `sys.exception()` object/type/args/code/name, exact ordered frame/offset route and inner INSERT offset are required. No route normalization or primary-error repair is added. Finally restores parent hooks and clears invocation references. Legacy source behavior and general V/J stale-primary concerns retain their separate scope.

Target body hashes (excluding decorators), independently recomputed: old `52d8c527606d37b55d02e9a0566dcb9913cf7fb4e045995281ee5356797f6ff5`; new `2f08345c60beb43e266d6ccfa728165a46272ed605e88d8e32ef86e58563a1cc`.

Frozen SHA256, independently recomputed:

- guard: `a686ece82b1fce6bd7e25a10e98f0aebceb7a8aff949166c0e2e765afe45fd31`
- recovery tests: `8fa360486b61cdfd3c13a2b6bfdc7ea15d75bb56502f7dfe40c153f72aa80868`
- guard tests: `46631d7d3536331496b036d8ed54f2ee0ac6db18bbac416d99af8b5739dc4fca`

## Finite future matrix and token bound

No row below was run or collected. Cardinalities are source declarations only; the matrix must be corrected/frozen before runtime.

| Source rows | Review assessment |
|---|---|
| complete/missing all/missing seam/missing final/final-only/duplicate/truncated/overflow/legacy-only/exit0/synthetic11 (11) | Exact grammar/status observer controls; synthetic, not native mechanism proof. |
| positive × False/True (2) | Genuine native INSERT FULL and final route required; F1 blocks source readiness. |
| cleanup-ValueError/cleanup-RuntimeError/cleanup-new-object × both (6) | Real session generator replacement during unwind, not record-only simulation. Explicit seam-present/final-absent assertions retained. Still depend on repaired positive. |
| wrong-node/old-param/missing-bool/wrong-bool/forged-collected × both | Identity/transport controls; no truthiness conversion. Forged-code dimension remains missing. |
| missing-config/duplicate-config/wrong-hardener/wrong-order/wrong-readback/wrong-private/wrong-roots/wrong-session/wrong-plan/wrong-frozen/wrong-inode/burn-missing × both | Synthetic state/trace-boundary controls; several only overwrite observer counters, with gaps listed in F3. |
| later-phase/completion-entry/policy-bypass/preprepared/original-config/setup-ValueError/AUTH/BEGIN/COMMIT/rollback/other-SQL/wrong-engine-code/wrong-engine-name/wrong-engine-type/forged-FULL × both | F2 unrelated early route invalidation; named coverage not established. |
| attribute-replaced/route-replaced × both | Mutate final actual object/traceback; require repaired positive to establish meaningful rejection. |
| foreign-owner/missing-done/synthetic11 × both | Shared aggregation/status controls; synthetic11 sends no signal. Current generic failure assertions do not establish otherwise-complete S tokens. |

These exhaust the 41 declared modes; no mode count is treated as coverage by itself. Keep the two original S routes plus other ten capacity rows, R15 controls/R5 parameters, J, T/C, declared73, semantic V/report override/unrelated-child, old-core-descriptor successful fresh resume, all18 transport modes, all child-read/parent-write close subsets, normal-wait/cleanup-wait reap subsets as future preservation gates. H/P have no credit from this S tree and remain separate frozen-family work.

Current S emits fixed `Sa;`, `Ss;`, `Sf;`: 9 bytes, with only allowed operation/session V suffixes totaling at most22, so accepted S grammar is at most31 bytes. Static future reserve is A/H/P/S =36 bytes plus mutually exclusive legacy maxima: D130+V25=155; R≤90+V25=115; J≤36+V25=61; T55+C50+V25=130. Thus 36+max(155,115,61,130)=191≤256. No bound is enlarged and no private payload is serialized. This is a conservative declaration/exclusivity bound for proposed combined families, not proof that absent H/P implementations preserve that exclusivity; recheck overlap and actual final combined code before grouped execution. Missing/duplicate/truncated/overflow evidence cannot rescue an incomplete S via legacy acceptance.

## Historical evidence, handback and clean exit

Historical S1 diagnostic remains FAIL: genuine early core-intent FULL with `same_route=false`, False-only parent assertions and no qualifying S acceptance. It is not relabeled, and this review has not revalidated raw runtime proof. Genuine engine FULL at private completion after three receipts/scan remains **UNPROVEN**. Early S, physical A/B and modeled R do not substitute for it. Historical PID161/PID255 inner exceptions remain unproven. STT artifacts, historical worktrees and retained proof were not altered.

Parent fix → fresh independent source review → separately authorized family runtime/receipt review remain required. Then combine frozen H/P/S with explicit overlap review, complete frozen-diff review and the full grouped actual-positive/directed-negative gate. Four-B2, conditionally full service/release, T2/C1295 and final global Astra remain pending. Mandatory C1 `b2_pinned_launcher_context_unavailable`, B3/C1/C2, native Darwin causes/fresh-helper/collection, A1 drift, all19 Linux mechanisms/ext4/VFS/WAL/FULL/SHM/peer credentials/installed durability, Apple Silicon/hardware/browser/acoustic and activation remain pending. No source or runtime acceptance, integration or release authority is granted.

Candidate remains at the exact requested HEAD with clean porcelain; `git diff --check` passes. Sole review write is `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23-S-WITNESS-SOURCE-r1-sol.md`. Root preexisting STT untracked files are retained; root is not represented as globally clean. No other repository or worktree was written, and no resource cleanup was needed.
