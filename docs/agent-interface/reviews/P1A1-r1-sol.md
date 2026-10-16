# P1A1 independent functional / reliability review

2026-10-16 · gpt-6.1-sol / low · read-only candidate review.

**Bounded GO for A1 at `29fc22c484c4bf32f36cb6bfd6ce602d81554325`.** No blocking functional/storage finding found in the assigned publication slice. This approves the candidate for parent integration verification; it does not certify a green global runner, global resource preservation, installed P1 durability, full P1, activation or final global Astra acceptance. A2, B and C remain closed.

## Exact identity and inspection

Candidate inspected clean at `/Users/ben/.txcode/worktrees/py-audio-box/t3-e92f4233`, branch `t3/p1a1-publication-durability`; base `b77cdcdf35681943361e92a9501e808cf50be30d`. `a85df081e1adcfb9b003dc4a63b4a5d9212aff68` to candidate changes only the implementation report. Independently checked all five changed source/test files against `/tmp/p1a1-final-source-hashes.json`: five matches. `git diff --check b77cdcd HEAD` passes. No candidate modifications.

Read PRODUCT, implementation start/protocol, CONTRACTS 1.3.0, the full P1 implementation decision including private format, ordering, exit matrix and ready A1 brief, R1 fault authority/R1-r3, and the full PERSISTENCE-P1A1 handoff. Inspected actual complete source/test diff, Core publication/recovery/initialization/transactions, Dispatcher callers and terminal paths, persistence helpers, Admission publication preflight, and installed runner teardown. No applicable repository AGENTS.md found. Contract 1.3.0, Core user_version 1 and R1 protocol 2 remain distinct from private P1 revision 1.

## A1 requirement closure

| Requirement | Actual closure and evidence |
| --- | --- |
| Exact intent/work before creation | `service/core.py:379–385` generates the temporary identity and commits original intent plus binding/destination/temp/hash/length atomically before namespace sync or exclusive creation. Original manifest/message are retained. BEGIN/INSERT/COMMIT denial and applied-then-error tests check durable pair-or-neither and absence of temporary creation. |
| Complete writes and no replacement | `service/persistence.py:52–62` loops positive short writes, rejects invalid progress, flushes, verifies length and fsyncs. Core uses exclusive `xb`, then no-replace link (`core.py:391–393`). Failures retain evidence without success. |
| Recorded two-link completion only | `core.py:401–441` revalidates work binding through the oracle, validates managed destination and generated basename, reads both O_NOFOLLOW descriptors, requires owner/regular type/hash/length/exactly two links and same inode, rejects another published asset sharing that inode, then unlinks only the recorded alias. Third/wrong/symlink/unrecorded aliases remain quarantined across two restarts. No general GC. |
| Repeated barriers before recovery success | Recorded-alias unlink is followed by directory sync, then one-link destination verification/file fsync and directory sync. `_commit_publication` repeats that barrier before its reference/terminal transaction. Both actual subprocess crash boundaries and injected recovery file/directory failures pass. Readable bytes alone cannot commit recovery. |
| Parent/root/SQLite namespace | Durable mkdir syncs new directory and parent. Fresh database uses exclusive create/file fsync/root sync; constructor syncs root and parent after schema/private-format initialization. Intent commit additionally syncs Core namespace before temporary creation. Initialization fault tests refuse fragmented namespaces rather than reseeding them. |
| Fresh private format; old roots | `persistence.initialize` accepts only the named revision/scope/column shape and refuses partial format. Only missing/truly empty roots initialize. Populated old roots are not stamped; query_only fences direct downstream schema writers as well as Core transactions. Metadata/history/artifact/dedup remain readable; old pending publication does not infer an alias or acquire an old owner lock. This is A1 private synthetic Core format, not B1 deployment/volume/profile enrollment. |
| Incomplete/missing destination and terminal history | Missing destination is classified separately from other stat I/O failures. After exact reap, incomplete adoption can fail without synthesis replay or staging promotion; evidence remains. Already terminal publication returns without duplicate rows/events or history rewrite. Orphan/integrity inventory remains A2. |
| SQLite ambiguity / rollback | Publication decorators latch storage failures until reopen. Transaction rollback preserves the original exception, tries DB-API rollback if SQL rollback fails, and closes if both fail. Intent/terminal and real Dispatcher outcome/release applied-then-error tests reopen actual state and require one success/event/ACK/launch. |
| WAL/checkpoint | Real reader-pinned checkpoint BUSY retains committed WAL success and stable history across reopen. No WAL deletion or successful-checkpoint prerequisite exists. This qualifies BUSY handling, not physical checkpoint I/O corruption recovery or power-loss durability. |
| Storage versus invalid result | `_publish` maps worker-result ENOENT/ELOOP to deterministic verification rejection, while other I/O, SQLite and provenance uncertainty latch publication storage failure. Dispatcher catches storage separately from ValueError/AssertionError; it withholds outcome/release/ACK and requires reopen. Adopted namespace EIO is not treated as missing adoption. |
| R1/W2 preservation | Exact runtime/binding/original-deadline/PID-1 reap and R1 publication preflight still precede publication. Stored intent checks generation/attempt/owner/staging; unresolved authority still blocks release/ACK. New early storage latch applies after publication/outcome faults, where exact reap already occurred; it does not bypass live R1 pending-I/O stop obligations. Resource/worker/protocol implementations are unchanged. |

No severity/path/reproduction findings are open. The limitations below are evidence boundaries, not silently waived success criteria.

## Fresh tests and inherited installed evidence

Fresh command in the exact candidate worktree, existing isolated venv, no installs:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_persistence.py tests/service/test_core.py tests/service/test_dispatcher.py
```

**103 passed in 46.56 seconds, exit 0**, including all 47 new persistence checks. This is fresh E1/local process evidence, not a new installed campaign. Temporary synthetic pytest roots only; no source cache writes requested.

Audited retained final runner receipt, log, before snapshot, sanitized audit and hash reports under `/tmp`. The retained final run reports **247 tests / 125.64 seconds and 43 installed scenario steps passed**, but **whole runner exit 1**, with the raw final assertion `assert all(set(before[k])<=set(after[k]) for k in before)`. Receipt SHA256 independently recomputed as `b4c02b558967d601f7f23d9fcd08706485ae5dac2bcff6d61dc52ed437c34df8`. These are implementer-run results audited now, not fresh reviewer execution. The earlier 245-test run likewise exited 1 and is development evidence only.

Final recorded control image is `sha256:62b7f8ba77a5be34bf884c72c3e9254c1af08107dc8d2754d6dbdf1ed3cc7976`; actual image inspect independently confirms that immutable ID exists. Retained independent verification reports 98 control, 18 worker and four packaging entries with zero mismatches. Worker `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b`; dependency overlay `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`. I did not reuse/run the image or rebuild it; fresh tests used source verified above. Installed attribution remains the retained verified source/image campaign.

## Inventory limitation and bounded acceptance

Global inventory decreased **16→15 containers and 141→140 volumes**. The retained independent audit identifies one missing prior member in each set. This review does not attribute a cause, infer an external actor, or claim global preservation PASS.

Independently parsed all **74 removal commands** in the final receipt: explicit `docker rm` targets begin with the exact random project `pab-r1-5cb23297c5`; compose down commands name that same project. Inspected actual runner: controls return before removal, worker stop/wait precedes teardown, and inspect requires non-running containers before compose down. No force removal/prune path. Retained post-run audit reports zero owned containers/volumes; this is historical post-run evidence, not a new live inventory claim. The recorded owned removals do not target the missing prior resources; that does not establish why those resources disappeared.

A1's required targeted checks are green and its publication protocol is independently reviewable without claiming the global harness succeeded. Therefore the inventory gap does **not** block bounded A1 source acceptance, but **does block any claim of a green full runner/global-preservation gate**. No third heavy campaign was justified by a source change or a new failure hypothesis; the harness was not weakened. Parent integration verification remains required. Any later preservation acceptance needs an appropriately controlled fresh inventory and evidence, not reclassification of these failures.

A2 inventory/integrity/startup joins, B enrollment/operator authority/cleanup, C shipped persistent topology and installed P1 fault matrix remain unaccepted. ENOSPC is injected, not physical capacity proof. Process kills/fsync ordering do not certify daemon/VM/controller/power-loss durability. Public/unattended persistence, real engines, client/browser/acoustic/hardware gates and final global Astra remain closed.

Only this report in the root repository was written. No source fix, commit, ledger/dependency/worker/contract/legacy edit, subagent, Docker execution campaign, model/private audio, cloud/STT/Voicebox, remote E2E, prohibited run-e2e script, prune, push, PR or deployment. Existing root research and clean candidate preserved.
