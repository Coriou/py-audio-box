# P1A1 publication durability candidate

2026-10-16. Implementation handoff, awaiting independent Sol-low review; no self-approval, integration acceptance or activation claim.

## Identity and scope

- Assigned worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-e92f4233`, branch `t3/p1a1-publication-durability`; initially clean.
- Base: `b77cdcdf35681943361e92a9501e808cf50be30d`.
- Predecessor: exact `16950490919b18253f5d12e7ff4e9e31b4557f2d` is an ancestor; its `service/` diff to the base is empty. The parent supplied the integrated R1 receipt of 200 tests / 81.84 s and implementation equivalence to reviewed `5ab47170c4c467cb3d3bef6075a0ff6480e8d56f`. Those predecessor results are supplied evidence, not a fresh rerun by this implementer.
- Source candidate: `a85df081e1adcfb9b003dc4a63b4a5d9212aff68`. The subsequent report-only commit does not change tested source; final handoff SHA is in the implementation response.
- Exact delta: `service/core.py`, new `service/persistence.py`, `service/dispatcher.py` (storage-error classification/latch only), `tests/service/test_core.py`, new `tests/service/test_persistence.py`, this report. No allowlist expansion.
- Contract/fixtures/worker protocol stay 1.3.0; Core `user_version=1`; R1 resource protocol remains 2. No dependencies, packaging, workers, contracts, ledger, prior reviews or evidence files changed.

Read PRODUCT, implementation start and full orchestration protocol, canonical plan shared architecture/Phase 1A/evidence gates, CONTRACTS 1.3.0, full P1 Astra ordering/recovery/exit matrix/ready brief, R1 fault-authority decision and R1-r3 Sol report; inspected actual Core/Dispatcher and the existing release runner. No applicable repository AGENTS.md found. No subagents or additional Git checkout.

## Behavior and private schema

`p1_format(revision,scope)` contains exactly `(1,'synthetic-core')`. This is the A1 private Core format, **not** B1 deployment enrollment. `publication_work` is additive in the same SQLite DB:

| Column | Meaning |
| --- | --- |
| `job_id` | Primary key, FK to original `publications` intent |
| `binding` | Canonical original artifact, binding job/input/generation/attempt/owner/staging and immutable output identity |
| `destination` | Unique validated managed output ID |
| `temporary` | Unique control-generated `adoption-<32 hex>` basename |
| `sha256`, `byte_length` | Exact verified bytes; length constrained to 1..1048576 |

The intent and work row commit together before a Core-directory barrier for SQLite namespace persistence and exclusive temporary creation. Complete writes handle positive short writes; zero/invalid progress fails. Flush, exact file length and file fsync precede no-replace link. Live publication and recovery use the same adoption barrier: verify O_NOFOLLOW descriptors, expected owner/type/hash/length/link count, prove the recorded two names share one inode, reject any published metadata ID sharing it, unlink only the recorded redundant temporary, directory fsync, reverify one-link destination with file fsync, and directory fsync before references/success commit. Original manifest/message remain unchanged.

New directories sync themselves and their parent. The database name is exclusively created and file/parent synced before SQLite use; schema commits use WAL/FULL and the Core namespace/parent are synced after initialization/reopen. No WAL deletion, forced checkpoint or checkpoint prerequisite is introduced.

The existing Core constructor is the private synthetic-Core creation initializer. Only a missing or truly empty root can create the A1 format. A root containing an old DB, keys/marker/unexplained entries, or incomplete namespaces is never stamped/reseeded. Populated pre-P1 roots remain metadata-readable; Core is query-only (including downstream direct schema writers), transactions refuse mutation and old publication intents have no inferred alias provenance. Duplicate-key reads precede mutation/preflight, including old terminal roots. Existing tests remain meaningful by creating fresh disposable synthetic roots. The prior disk-full test now requires same-session quarantine followed by reopen before an incomplete adoption can fail. No populated-root compatibility migration was added. Full explicit deployment identity, role/profile/budget enrollment and matching partial-install completion belong to B1/C1 and remain unavailable.

A storage failure or ambiguous publication/outcome commit latches the session until reopen; Dispatcher stops dependent progress, retaining receipt/intent/reservation and withholding ACK. Missing/symlink worker results remain deterministic verification rejection, while adopted-namespace I/O errors retain pending work. Exact supervisor reap and R1 preflight/release rules retain their ordering. The Dispatcher storage latch is set only in the post-reap publication/outcome paths, so R1's live pending-I/O stop obligations remain intact.

Rollback failure cannot mask the original write/COMMIT error: SQL rollback is attempted, followed by bounded DB-API rollback; failure of both closes the handle. Reopen reads actual committed state. Already terminal jobs return without reinserting assets/events or changing attempts/history; asset/artifact retrieval still verifies availability separately.

## Fault/state table

| Boundary / state | Result and repeat behavior |
| --- | --- |
| Intent BEGIN/write/COMMIT denied, including COMMIT applied then exception | No temp creation. Atomic intent/work pair or neither after reopen; no live retry. Missing adoption fails only after existing reap authority; two restarts stable. |
| Positive short writes | Loop completes exact bytes, flush/fsync then ordinary adoption. |
| Zero/partial-error/flush/file-fsync failure | Intent/temp retained; no references/success. Reopen classifies incomplete adoption and fails without resynthesis/promotion; temp retained across second restart. |
| Actual subprocess death after link, before unlink | Two-link recorded alias verified and completed once; one-link destination and one terminal event after repeated restart. |
| Third link, different inode, symlink or unrecorded alias | Quarantine; every name retained; no success or deletion across two restarts. |
| Unlink then failed directory barrier / recovery file or directory fsync fails | No DB success; subsequent reopened recovery repeats barriers before commit. |
| Adopted destination stat I/O error | Quarantine, not missing-media failure; restored storage plus reopen recovers once. |
| Reference/terminal COMMIT denied or committed-then-error | Reopen distinguishes running from already succeeded; inserts exactly one asset/artifact/success event, preserving history over second restart. |
| Outcome/release COMMIT denied or committed-then-error | No ACK in failed session; reopened state determines pending/completed outcome and release; exact receipt retained, one ACK/launch after two restarts. |
| Checkpoint BUSY with reader pinning WAL | Committed success remains readable through reopen with WAL retained; no requirement for successful checkpoint and no history rewrite. This is actual SQLite BUSY, not physical I/O corruption. |
| Reopened recovery writes denied twice | Intent and nonterminal history retained, no reference rows; later writable reopen commits exactly once. |
| Root/subdir/database/format initialization barrier failure | No stamping over fragments; repeated opens refuse incomplete namespaces/schema. A truly empty directory has no minted identity/DB and may initialize; a committed format survives final namespace-barrier failure without reminting. |
| Old populated intent / old terminal history | No automatic enrollment or guessed temp deletion. Old pending intent quarantines without opening/creating owner locks; old terminal job/events/artifacts/dedup reads remain historical, fresh key mutation refuses. |

## Commands, results and evidence

Final-source E1 command (existing isolated venv, no installs):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_persistence.py
```

**47 passed / 21.47 s**, log `/tmp/p1a1-final-e1.log`.

Final-source Docker command, cwd `/tmp/p1a1-final-b62e_mwp` (exact source commit exported without another Git checkout):

```sh
R1_EVIDENCE_PREFIX=authority- R1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-resource-r1.sh
```

The unchanged harness builds network-none using the existing dependency overlay, verifies copied source, then runs `python -m pytest tests/service tests/release -q -p no:cacheprovider` in a read-only container with 2 CPU / 1 GiB / 128 PIDs / bounded temporary storage. **247 passed / 125.64 s, no skips/failures; all 43 installed R1/W2 scenario steps passed.** These include clean versus open-scan recovery, F2, deadlines/cancellation/lost ACK and authority-denial/restart. They are steps, not 43 new P1 fault families.

**The runner as a whole exited 1**, at its final unchanged assertion:

```text
assert all(set(before[k])<=set(after[k]) for k in before)
AssertionError
```

Global Docker inventory drifted during the run: the independent raw before/after snapshot has 16 -> 15 prior containers and 141 -> 140 prior volumes, with one missing prior ID in each set. No global-preservation PASS is claimed. All 74 recorded removal commands target only this run's unique project `pab-r1-5cb23297c5`; none targets a prior ID/name. The existing harness confirms terminated ownership before cleanup; no force removal/prune was used. Independent post-run queries find **zero owned containers and zero owned volumes**. Missing prior resources are outside that project; their external lifecycle was not controlled or repaired. Resource-preservation acceptance is therefore **unverified**, despite passing source/lifecycle tests and verified bounded owned cleanup. No third heavy rerun against this drifting inventory.

The first sequential Docker run, source `61aadfaad524a7918a2defff9b21c88b680dd971`, likewise passed 245 tests / 121.89 s and all 43 installed steps, then exited 1 at the same inventory assertion. Its 74 removals were independently checked to target only its unique owned project, and zero owned containers/volumes remain. It is retained as development evidence, not a green final gate. Both runs used the existing exclusive heavy-lane lock; no overlapping heavy lane was launched by this task.

Final identities:

- Dependency: `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.
- Worker: `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b`.
- Control: `sha256:62b7f8ba77a5be34bf884c72c3e9254c1af08107dc8d2754d6dbdf1ed3cc7976`.
- Independent post-build verification against the assigned worktree: **98 control + 18 worker + 4 packaging entries, zero mismatches**, both image IDs rechecked. `/tmp/p1a1-final-independent-hashes.json`.
- Full final runner log: `/tmp/p1a1-docker-final.log`.
- Final runner receipt (retained only in temporary export): `/tmp/p1a1-final-b62e_mwp/docs/agent-interface/phase-1/r1-evidence/authority-run.json`, SHA256 `b4c02b558967d601f7f23d9fcd08706485ae5dac2bcff6d61dc52ed437c34df8`.
- Independent before snapshot and sanitized audit: `/tmp/p1a1-final-docker-before.json`, `/tmp/p1a1-final-docker-audit.json`.
- Source-only aggregate diff from base to candidate: five files, **660 insertions / 25 deletions**; this sixth file is report-only. `git diff --check` passed.


Development failures were retained in `/tmp` logs. Initial targeted Core/Dispatcher: 56 passed / 30.02 s. Intermediate expanded target: 90 passed / 35.91 s. An intermediate Dispatcher terminal-COMMIT test exposed connection-local uncommitted success after rollback denial; bounded rollback fallback was added. The next full local run recorded 11 failures / 211 passed / 18 platform skips in 110.51 s, all failures caused by the new `FaultDB` test facade's boolean `rollback` attribute shadowing DB-API `rollback()`; the facade was corrected to `deny_rollback`. Focused corrected run: 101 passed / 57.89 s. Later namespace-stat I/O inspection prompted the final ENOENT-only correction/test. After adding the intent namespace barrier, one file-fsync fixture initially injected into the earlier directory barrier instead; its injection was narrowed to regular-file descriptors (1 failed / 46 passed before correction; focused 7 passed afterward). Old-root checks passed 2 tests. Only final-source results above qualify this candidate; intermediate results are not acceptance.

No new dependency installation, models/private audio, cloud/STT/Voicebox, legacy model suite, remote E2E, `scripts/run-e2e-test.sh`, push, PR, deploy, prune or unrelated-resource stop/removal. Docker work uses the existing pinned dependency overlay, network-none bounded synthetic containers, fresh owned volumes and no user-data mounts. Build contexts are disposable `/tmp` source exports, not extra Git worktrees/checkouts. Runner evidence stays under those `/tmp` exports; no evidence/ledger/review writes occurred in this worktree.

## Remaining gates and limitations

E1 covers deterministic storage faults and actual Core publication/process-restart ordering; the existing installed R1/W2 suite provides regression E2 lifecycle evidence only. Injected ENOSPC/query-only/syscall errors are not physical disk-full or power-loss evidence. Process/container kills do not establish VM/daemon/controller/power-loss durability. Checkpoint BUSY coverage does not certify corrupt storage recovery. One serialized owner and owned private local filesystem remain assumptions.

A2 still owns bounded inventory/integrity/status, orphan and missing-terminal-media reports, and full startup joins. B1/B2/B3 still own explicit deployment enrollment, offline proof-bound abandonment/new authority epoch and bounded cleanup. C1/C2 still own shipped persistent topology and the exact installed P1 named-volume fault gate. No inventory UI, recovery/reset/cleanup command, general GC, migration or public/unattended persistent activation is delivered. D1-D5, real-engine/hardware/client/browser/acoustic/global Astra gates remain open.

Next action: independent Sol-low adversarial review of the exact final source/diff and these limits; then parent integration/rerun before A2. This report makes no GO decision.
