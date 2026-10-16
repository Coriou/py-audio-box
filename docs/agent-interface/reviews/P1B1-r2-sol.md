# P1B1 independent adversarial review — round 2

2026-10-16 · gpt-6.1-sol / low. **GO for bounded B1 source/E1 acceptance only** at exact candidate `70251613902948c474d8030ab8f566815dc9e860`, correction source/test commit `34e68d58180a0429d5ca5dbe45cab57c344e7232`, imported precursor `b172e06`. F1, F2 and F3 from round 1 are closed within the evidence below. No new blocking finding. This does not approve integrated P1, installed topology, recovery apply, cleanup or activation. Parent integration verification and final Astra remain separate gates.

## Identity, scope and review basis

Inspected actual candidate read-only at `/Users/ben/.txcode/worktrees/py-audio-box/t3-94e6816a`. HEAD matched the requested SHA; candidate status was empty before and after testing. Relative to its root baseline `ca613d8`, the change set is exactly the eight authorized paths: initializer, recovery, host operator, persistence, Dispatcher, reconciliation adapter, B1 tests and handoff. Relative to `b172e06`, only initializer/recovery/tests/handoff change. Relative to `34e68d5`, final HEAD changes only the handoff. The older `75bc71c` comparison additionally includes parent ledger/review context; those are not implementation scope expansion.

Read the complete original P1B1-r1-sol findings and reproducers, complete candidate RECOVERY-P1B1 original report and correction response, original P1-IMPLEMENTATION-ASTRA including B1 invariants/allowlist/full exit matrix, PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, canonical shared/Phase 1/evidence guidance, R1 fault-authority decision and A2 handoff/final review context. Inspected actual initializer ordering, recovery snapshot/plan/verification/start gate, host Docker projection, Core constructor, Dispatcher gates and the exact persistence/reconciliation adapter delta. No applicable AGENTS.md found. Memory search supplied no B1-specific authority or acceptance evidence; the verdict uses current source/tests.

## F1 — high — closed: no ambiguous Core authority remint

`service/persistent_init.py:129–166` now records the deployment-bound Core intent through exclusive creation and file/directory/parent sync before the Core constructor at line 167. With an existing intent, absent/empty/incomplete namespace refuses before construction. Retained Core is privately read under the normal Core/journal locks; exact table/column presence and retained R1/A2/enrollment authority are checked before invoking the writer. Later existing-Core checks retain empty-installation and coherent-authority requirements. No healthy reset or guessed schema reconstruction was added.

Independent disposable-root reproduction repeated the original `database` fault followed by deletion of only `control/core`, then exact-record completion. It now returns `core_namespace_quarantine`; no database or enrolled marker appears. Empty replacement and interruption immediately after durable `core_intent` return the same refusal. Additional retained-Core probes dropped `resource_monitor`, or changed its row to `checking`: respectively `partial_database_initialization_quarantine` and `partial_authority_initialization_quarantine`. Every original regular Core file name/hash remained unchanged in these negative probes; no enrolled marker was created. Matching completion at genuinely not-yet-attempted/completed boundaries remains covered by the passing seven-boundary regression. Per-file attempted key/journal refusal and old populated-root preservation remain green.

The crash window after intent persistence deliberately costs availability: even a genuinely uncreated Core cannot be guessed safe once its attempt is recorded. That is the required fail-closed distinction, not a missing automatic retry. No minimal fix remains requested for F1.

## F2 — medium — closed: prior plans and references remain bound

`service/recovery.py:254–262` retains all source tables and excludes only the current plan ID plus its exact duplicate reference label. `verify_plan` passes that ID at line 358. Preexisting plan rows, digests and typed reference obligations remain in the source projection; later plan append explicitly stales older plans. Schema, other database rows, inventory and journal/marker evidence remain bound. Inspection of `service/reconciliation.py:184–223` found the approved closed B1 adapter and typed job/dispatch/publication traversal, without an epoch or cleanup-authority change.

Independently registered disposable roots, created first/second plans, immediately verified the second, deleted the first through a test-only SQLite mutation, and verified the second again: now `stale_plan`. The focused regressions also pass valid digest-recomputed prior-content mutation and later append negatives, nonempty immediate self-verification, and job/publication/dispatch asset-reference retention. No remaining blanket plan-history omission was found. No minimal fix remains requested for F2.

## F3 — medium — closed: inspection preserves source SQLite namespace/bytes

`service/recovery.py:72–113` copies DB and present WAL/SHM into a unique private temporary namespace; SQLite opens that copy, not the originals. The directory is 0700 and explicit file creation uses 0600. Source capture validates owned single-link regular files, identity/full bytes and namespace before/after copying and after use. Physical SQLite bytes are capped at 16 MiB, top-level names at 2,048, with a two-second observation deadline. Normal offline Core/journal locks remain mandatory. Missing SHM is reconstructed only privately. The copy is inspection material, never a second authority database or a recovery writer.

Independent real child-process probes committed `resource_monitor='checking'` into WAL then called `os._exit(0)`. With retained SHM and with SHM deliberately absent in the disposable fixture, inspection under a 0500 source directory returned the committed checking row and preserved every original regular-file name/hash. The four SHM/directory-mode combinations also pass in the focused suite. No immutable-original shortcut, original checkpoint, original WAL/SHM deletion or original SQLite open occurs on the read-only inspection path.

Independent injection denied creation of the private DB copy: inspection raised the original injected OSError, removed the private directory and released locks; subsequent inspection succeeded. Focused tests additionally cover detected source namespace change, private SQLite-open failure and cleanup. A source change refuses the result rather than returning a partial success. No minimal fix remains requested for F3.

This is conservative bounded local evidence under cooperative owner locks. It does not claim exclusion of a malicious administrator changing/restoring bytes, physical storage durability or installed read-only mount qualification. Read-induced atime is outside the preserved-byte/namespace assertion.

## Preserved boundaries and fresh verification

The host tool retains fixed narrowly projected read-only Docker operations and all-container enumeration, exact named volume identities/roles, daemon/context, immutable image/full container IDs and start observations. Unknown/missing/live consumers and mismatches refuse. Registration/start source binds authenticated lane/boot; its actual host/container association still needs C1 qualification. No generic shell/argv proxy or service Docker socket was added. Apply remains unconditionally `P1B2_apply_unavailable` before request processing. Enrolled submit/replace/cancel/execution remain closed, including diagnostic-default bypass protection; un-enrolled diagnostic behavior remains tested. Public contracts 1.3.0, Core user_version 1, R1 protocol 2 and private P1 revision 1 remain unchanged.

Fresh commands in the exact candidate using the existing interpreter, no installs:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**338 passed, 18 skipped in 108.87 seconds, exit 0.** Includes affected Core/Dispatcher/Admission/persistence/reconciliation and service/release preservation tests.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -q
```

**59 passed in 3.79 seconds, exit 0.** This short disposable-root focused run overlapped part of the full local suite; no Docker/model/heavy lane ran. The independent inline probes described above also exited 0. No source/test edits occurred during either run. `git diff --check ca613d8 HEAD` passed. Final candidate HEAD/status were rechecked unchanged/clean.

## Evidence limits and write boundary

Local E1 source/SQLite/filesystem/synthetic-process evidence only; Docker projection tests are mocks. The 18 installed/platform skips are not passes. A1 whole-Docker-inventory EXIT 1 remains unexplained and unqualified; no cause or waiver is inferred. B2 apply/death proof/reset, B3 cleanup, C1 actual packaging/topology/start, C2 installed fault/persistence matrix, parent integration, final Astra and public/unattended activation remain pending. No physical power-loss, ENOSPC, controller/VM durability, engine/audio, hardware or clean-host qualification claim.

Only this root report was written. No candidate/source fixes, commit, subagent, push/PR/deploy, heavy Docker campaign, paid operation, E2E runner, audio/model/network work or prune. Disposable synthetic test roots only; preexisting root research/untracked artifacts and other worktrees were preserved.
