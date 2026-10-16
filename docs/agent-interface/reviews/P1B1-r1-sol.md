# P1B1 independent adversarial review — round 1

2026-10-16 · gpt-6.1-sol / low. **NO-GO** for bounded B1 acceptance at `b2af46fe7b60f6fe30624e549a21201827ef3fce`, based on `75bc71c64eff6a131599ba2d680d20a978ef0762`. Three independently reproduced gaps remain. This is the first independent review; there are no inherited findings.

Candidate inspected read-only at `/Users/ben/.txcode/worktrees/py-audio-box/t3-e310284d`. HEAD matches the requested SHA and status is clean. The diff has exactly the eight authorized files, including the explicitly approved reconciliation adapter. No applicable AGENTS.md found. Public contracts 1.3.0, Core user_version 1, R1 protocol 2 and private P1 revision 1 remain distinct.

## F1 — high: completion recreates previously initialized missing Core authority

Location: `service/persistent_init.py:119–142`, especially `existing_core` at 129 and unconditional `Core(...)` at 130. Per-file creation intents protect keys/journal, but no durable attempt record protects Core creation. If Core disappears before final enrollment, matching explicit completion treats its absence as permission to create it again, including healthy R1/A2 authority.

Independent reproduction using real disposable roots:

1. Call `initialize(..., fault=...)` with a fault raising OSError at `database` (line 154). Core, resource protocol/monitor, integrity and enrollment tables have already committed; final receipt-side enrollment marker does not exist.
2. Read the retained `p1-init.json` and its digest. Simulate storage loss of only `control/core` with `shutil.rmtree` inside the disposable fixture. All four volume root inodes, keys, journal, initialization record and file intents remain unchanged.
3. Call `initialize(..., complete=digest(private_record))`.

Observed: completion succeeds, recreates the database/healthy authority and writes the final enrolled marker. Probe output: `MISSING_PREVIOUSLY_CREATED_CORE_RECREATED_HEALTHY_AND_ENROLLED`.

Consequence: the matching record cannot distinguish a never-attempted Core from a previously created but lost Core. Explicit completion can remint healthy authority after damaged initialization, violating the crash-boundary/no-healthy-remint invariant. The reproduction deliberately models loss in a temporary fixture; it makes no physical power-loss claim.

Minimal correction: persist and sync a Core-creation intent before attempting its namespace/database creation; if that intent exists but the corresponding exact coherent namespace is absent/empty/partial, quarantine rather than reconstruct. Preserve matching completion for genuinely not-yet-attempted Core creation. Add real-root tests for loss/empty replacement after the database boundary, plus crash between intent and Core creation; ensure no new healthy row or enrolled marker is produced by completion in ambiguous cases.

## F2 — medium: plan self-exclusion hides every other recovery plan

Location: `service/recovery.py:202–208`, consumed by `verify_plan` at 306–308. The source digest removes the entire `p1_recovery_plans` table and all its reference labels, rather than excluding only the proposal being verified. Validation of the remaining plan records does not detect a deleted record.

Independent reproduction with the existing temporary-root helpers:

```python
observation, first = enrolled_plan(installation)
second = recovery.plan(roots, scope, observation, now=101)
with sqlite3.connect(roots['control']/'core/core.sqlite') as db:
    db.execute('DELETE FROM p1_recovery_plans WHERE id=?',
               (first['plan']['id'],))
recovery.verify_plan(roots, scope, observation,
                     second['plan']['id'], second['digest'], now=102)
```

Observed: verification succeeds (`PRIOR_PLAN_DELETE_ACCEPTED`). This is an adversarial state-change probe, not a claim that a shipped deletion command exists. It demonstrates that a later plan does not bind all actual source records present when it was created. Prior proposal/reference evidence can disappear without stale-plan refusal. The approved exception is for self-exclusion, not blanket exclusion of other retained recovery evidence.

Minimal correction: define a stable source projection that binds preexisting plan records/reference obligations while excluding the current proposal's own serialization. Verification must reconstruct the same projection. Add prior-plan deletion and valid content mutation negatives, while retaining nonempty immediate self-verification and typed-reference tests. Explicitly define whether later appended proposals invalidate older plans; do not silently omit all history.

## F3 — medium: read-only inspection changes SQLite SHM bytes

Location: `service/recovery.py:86–91` and 226–227. SQLite `mode=ro`/`query_only` blocks SQL writes but still permits modification of the live WAL shared-memory index. It is insufficient for the requested file-preserving inspection boundary.

Independent reproduction: initialize real temporary roots; have a disposable child open the DB in WAL mode, commit `UPDATE resource_monitor SET state='checking'`, then call `os._exit(0)` without connection close. Hash regular files under `control/core` before and after `recovery.inspect(roots, scope)`. Inspection succeeds and sees committed WAL state, but `core.sqlite-shm` changes. In the observed run its length remains 32768 bytes and its SHA256 changes from `042a459ea68975596bb6ec7f51ed5b049570d39a3cbf674ce52e493afae5e822` to `a74fd50cb85e3230d9d76502dc0e5f9ec65e6de2a553ea4ac53481f266240be0`. DB/WAL hashes showed no change in this probe.

Consequence: the inspection path is logically read-only but does not preserve all source files as required. The supplied `logical()`/iterdump test cannot detect this. This is not evidence of lost SQL authority or WAL corruption; the narrower demonstrated defect is SHM mutation.

Minimal correction: inspect a bounded consistent private snapshot/copy including committed WAL, or another verified read-only strategy that preserves original DB/WAL/SHM bytes and namespace. Do not use immutable mode on the original DB if it would ignore committed WAL. Add actual crash-left-WAL tests asserting logical results plus original file hashes/presence before/after, including missing SHM and read-only source directories. Do not checkpoint/delete the original WAL/SHM to obtain a green assertion.

## Verification and preserved behavior

Read PRODUCT, implementation start/protocol, the full P1 Astra decision (B1 requirements, invariants, allowlist, exit matrix and ready brief), R1 fault-authority decision, A1 review, A2 handoff and relevant correction/final reviews, and candidate RECOVERY-P1B1. Inspected the actual eight-file diff, initializer, recovery/host CLI, dispatcher startup/command/tick paths, Core constructor/transactions, R1/A2 initialization and typed-reference traversal. The memory registry contained no B1-specific authority or verdict; none was used to approve behavior.

Positive source/test observations: create-only empty-root check; exact volume roles/configuration identity; exclusive per-key/journal intents; closed registration and plan records; fixed projected Docker reads and all-container enumeration; exact image/full ID/start scope; normal lock refusal; finite SQL/inventory bounds; unconditional apply refusal; retained typed job/dispatch/publication references; enrolled activation closed in command and execution paths. E1 mocks do not establish the real host/container association. No new cleanup/death/reset authority was found in the approved adapter.

Fresh command in the exact candidate, existing interpreter, no installs:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**326 passed, 18 skipped in 108.96 seconds, exit 0.** This includes the focused/affected B1, persistence, dispatcher, admission and Core tests. Additional inline disposable-root probes above are independent of the passing implementation assertions. One initial probe invocation had an incorrect `service.profiles` import and stopped before setup; corrected to the existing test helper, then the reported probes completed. `git diff --check 75bc71c b2af46f` passes.

Separate focused rerun after the full suite: same interpreter/options with `tests/service/test_recovery.py -q`: **47 passed in 2.55 seconds, exit 0**. Final candidate HEAD is still the exact requested SHA and `git status --porcelain` is empty.

## Evidence limits and write boundary

Local E1/source/SQLite/process evidence only. E1 host projection mocks are not installed qualification; 18 installed/platform skips are not passes. A1's whole Docker runner EXIT 1 due unexplained prior-inventory drift remains unresolved; no external cause is inferred or waiver given. B2 apply/death proof/authority reset, B3 cleanup, C1 installed topology/start, C2 installed fault matrix, final Astra and public/unattended activation remain pending. No physical power-loss/ENOSPC/controller/VM, real model/audio or hardware qualification claim.

Only this root review file was written. No source fixes, candidate edits, commit, subagents, push/deploy, Docker campaign, paid remote E2E, model/private-audio/network access or cleanup/prune. Existing root research/untracked files and other worktrees were preserved. Parent correction/integration verification and final Astra remain separate gates.
