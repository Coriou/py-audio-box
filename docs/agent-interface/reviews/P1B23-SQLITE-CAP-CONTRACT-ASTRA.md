# P1B23 SQLite physical-cap contract — Astra design decision

2026-10-23. **Select a narrow test correction, with mandatory positive WAL-write and explicit-checkpoint failure witnesses. An unchanged ordinary diagnostic is not necessary to select this design. Cap execution remains HOLD; no amendment, execution, combined integration or broad retry is authorized by this advisory alone.** Parent must adopt the concrete amendment before Sol-low implements it and a separate Sol-low adversarially reviews the exact candidate. This is design advice, not a candidate GO.

## Evidence and identity

Read the entire latest SQLITE-CAP-OFFLINE-20261023, CHILDWAIT-GUARD-r2-sol and CORE-OWNERSHIP-r1-sol reports, ORCHESTRATION-PROTOCOL, current ledger chronology through this advisory assignment, relevant RECOVERY-P1B2 Lot K model/caps and the original ACCOUNTING-ASTRA envelope/exit requirements. Inspected the actual failing test and maintenance limit, connection and session cleanup source. No applicable report-path ancestor/scoped AGENTS.md found.

Verified current pins and clean candidate porcelains:

- Guard `/Users/ben/.txcode/worktrees/py-audio-box/t3-207e07d8`: `f8822965c2e6ba970d36937e11225fb0fd0dc2b9`; frozen source/tests `3f010b3ae469b68f4ca0716c4019bb5f3e490b42` is review-attributed. Independently empty production/dependency diff against `4259415daaf5acecfe53ed9d44f9895207a00d1b`.
- Ownership `/Users/ben/.txcode/worktrees/py-audio-box/t3-0727f054`: `e5d4df14deb49013ed7cc9f3435444c23eae1b39`; frozen `5619270d5cdce0d305a17fd60be58f53c53f550a` is review-attributed.
- Report root HEAD `1f5778acc5d1df484cc869cf63274760e6fefee4`. Root T1 source integration does not include the later ownership/guard/C candidates. Preexisting STT artifacts preserved.

Guard frozen 61/200/85 and independent 61+42 passes, ownership frozen 93/128 and independent 12 plus actual SQLite/ownership probe remain attributed evidence, not executions here. Their bounded GOs do not combine into a GO.

Retained `frozen-17-entry.json/stdout` and unchanged test agree: 5 pass/1 fail, helper exit 1 at `assert failed`, after all 100 INSERT/COMMIT pairs and per-iteration size assertions. No native crash. Direct write/pwrite/truncate/ftruncate exact-L/EFBIG checks passed. Explicit rollback/checkpoint/close/final verification and the outer parent-limit assertion were not reached. Do not credit interpreter teardown as their execution.

Offline report establishes 125 manifest matches and checksum-valid current-generation 798 frames/44 commits, final 1615 pages = 6,615,040 logical bytes; 219 stale slots excluded. DB 4,194,304, WAL 4,190,072 and SHM 32,768 bytes individually satisfy L. Complete address coverage is stronger than a header claim, but is not semantic integrity, an atomic snapshot, a durability result or a historical syscall trace. I did not rerun its parser/hash campaign or open any archived SQLite file. Its observations are explicitly attributed.

## Contract finding and source basis

The original ACCOUNTING-ASTRA requirements concern physical file lengths and real SQLite creation/checkpoint/close within the child envelope; they do not require logical database size <=L or a commit error after a particular number of rows. RECOVERY-P1B2 retains `U_all_early = M + 3*L + 1_048_576 + 2*65_536 <= H`, then H plus all retained charges <=B. Three physically bounded SQLite files are budgeted. Neither `page_count*page_size` nor the sum of DB+WAL bytes is an interchangeable cap witness.

The test at guard `tests/service/test_recovery.py:3406–3450` added a stronger, unsupported premise: some `execute` or `commit` must raise within 100 inserts despite automatic checkpointing/reuse. Production `_maintenance_connection(original=True)` already sets `wal_autocheckpoint=0`, FULL and mmap_size=0; the test omits the first setting. `_install_maintenance_limit` sets both limits to 4,194,304 and ignores SIGXFSZ to expose ordinary failures. Session close and file inspection remain inside both locks and the limited process.

Primary upstream source inspected is SQLite **version-3.40.1**, matching the recorded Linux library version, not current SQLite HEAD:

- [main.c](https://raw.githubusercontent.com/sqlite/sqlite/version-3.40.1/src/main.c), `sqlite3WalDefaultHook`, lines 2345–2356: calls the checkpoint function but discards its return code and returns SQLITE_OK. `sqlite3_wal_checkpoint_v2` propagates its own checkpoint result. Therefore an automatic checkpoint error need not make a successful WAL commit raise. This alone defeats the test's universal exception premise.
- [os_unix.c](https://raw.githubusercontent.com/sqlite/sqlite/version-3.40.1/src/os_unix.c), `unixWrite` 3470–3547 and `unixTruncate` 3788–3831: incomplete writes are handled; a negative write with errno other than ENOSPC maps to SQLITE_IOERR_WRITE; failed ftruncate maps to SQLITE_IOERR_TRUNCATE. EFBIG is not generally SQLITE_FULL on this path. An arbitrary sqlite3.Error is too weak a positive witness.
- [wal.c](https://raw.githubusercontent.com/sqlite/sqlite/version-3.40.1/src/wal.c), `walCheckpoint` 2035–2223: writes selected frames to DB, stops on error, truncates/syncs on the successful path, and advances completed backfill only after success. `sqlite3WalClose` 2238 onward conditions WAL deletion on successful checkpoint. These source paths explain why a live logical DB can require WAL after failed backfill; close success alone is not a checkpoint-success oracle.
- [vdbe.c](https://raw.githubusercontent.com/sqlite/sqlite/version-3.40.1/src/vdbe.c), `OP_Checkpoint` 7620–7642: non-BUSY checkpoint errors abort the statement; BUSY becomes a result-row flag. An explicit checkpoint's nonzero busy field is not a physical-cap error witness.

The [auto-checkpoint API](https://www.sqlite.org/c3ref/wal_autocheckpoint.html) documents the default threshold and passive checkpoint mode; the [checkpoint API](https://www.sqlite.org/c3ref/wal_checkpoint_v2.html) distinguishes checkpoint completion, BUSY and errors. [WAL format](https://www.sqlite.org/fileformat2.html#walformat) supports generation/checksum/commit-based overlay interpretation. These sources support a contract correction, **not a finding that a particular historical checkpoint returned EFBIG**.

Fetched upstream source SHA256s, in the order main/os_unix/wal/vdbe above:

```text
95a002803f241415936ace56baabf959fbf401d52026698bf3c54c42d77e6af3
e2bf58f8b7abf4a1b9c13709a0630888d51dbc47877d0372967cb2ec9a7c3bd0
70f62d19e85047a23cdd94bb2ae353882e9f548ef5b79912df64d87f61ea2953
3684b9183e332034002387a76eec8695d37ee0dae26ee6538948a4921cf25731
```

Important provenance limit: release-tag source is not independently authenticated as the exact linked distribution build, including patches/compile options. Before the later authorized candidate run, record sqlite_version/source_id, compile options, linked-library/package identity and existing image identity; inspect relevant distribution deltas if they differ. Do not install another SQLite to force agreement. This gap prevents exact historical runtime attribution, but does not restore a missing API guarantee. Linux CPython 3.14.8/SQLite 3.40.1 versus Darwin 3.14.7/3.53.4 is not a cause finding.

## Selected amendment for parent adoption

Test-only; no production change selected. Keep the existing named test and all four direct OS boundary checks, helper timeout 30 seconds, exact hard/soft L and unchanged-parent-limit assertion. Make parent-limit verification run even when helper outcome fails, without masking that failure. Add a separately named checkpoint test and small test-local witness controls. Preserve the original failing evidence and record the old body hash in the implementation handoff.

### A. Mandatory production-aligned WAL append failure

In `test_lotk_actual_linux_child_file_and_sqlite_cap` retain the same table, BLOB payload 65,536 bytes, at most 100 INSERT/COMMIT pairs, real SQLite and fresh guarded interpreter. Explicitly select `vfs=unix` with URI mode and configure/read back WAL, FULL, mmap_size=0 and **wal_autocheckpoint=0** before the workload. The last setting matches existing production; it removes WAL reuse from the witness, not from the physical-cap proof. Do not change page limits, journal_size_limit, cache/spill policy, limits or payload/iteration ceiling.

Require at least one committed insert before failure. Distinguish execute from commit, count only returned successful commits, and capture `sqlite_errorcode`/`sqlite_errorname`. The required failure is **SQLITE_IOERR_WRITE (778)** during INSERT or COMMIT; no catch-all sqlite3.Error, SQLITE_FULL, BUSY, AUTH, syntax error or synthetic exception counts. With one connection, no checkpoints and an initially empty DB, 100 non-overwriting 64KiB blobs cannot fit a never-reset <=4MiB WAL. No exception within the bound is a hard test failure.

Verify each present DB/WAL/SHM length <=L after every operation, including the failing one, rollback, explicit checkpoint and close; verify the process token throughout. Inventory only the three expected SQLite names, require the DB, and fail unexpected sidecars rather than relying on an empty glob. Record page size, successful count, logical page count where available and all three physical sizes. Do not claim that Python's error code alone exposes the kernel errno: the SQLite cap interpretation combines source, controlled envelope, positive prefix and direct EFBIG controls.

After this expected append failure, perform exactly one rollback, assert no live transaction, then exactly one explicit `PRAGMA wal_checkpoint(TRUNCATE)` with fetched result `(0,0,0)` expected for the surviving sub-L committed prefix. Require an ordinary read of that prefix to match the successful commit count and payload lengths; this is a disposable test fixture, not a production live retry. Close once, verify the cap and file inventory, and report an explicit cleanup completion marker. Any cleanup error is a failure, not swallowed. Use finally to attempt close and final observations even if an earlier assertion fails, preserving primary and cleanup failures. No new SQL insert after failure.

This repairs the required positive error witness and makes previously unreachable rollback/checkpoint/close checks mandatory. It does not certify accepted recovery or general ambiguous-COMMIT handling.

### B. Separate mandatory explicit-checkpoint cap failure

Add `test_lotk_actual_linux_sqlite_checkpoint_cap` beside A. Separate guarded fresh interpreter and owned temporary directory; same L/30-second timeout, Unix VFS, WAL/FULL/mmap=0/auto-checkpoint=0. Pin **4096-byte pages only in this newly created test database**, before WAL creation, and verify the readback; no production page-size change.

Use the same one-BLOB table. Commit **48** 65,536-byte rows, assert logical bytes <L and all physical lengths <=L, and perform one successful TRUNCATE checkpoint `(0,0,0)`. This establishes a real populated main DB and a checkpoint-success control. Commit **20** additional identical rows without checkpointing; require all 68 rows readable, logical page_count*page_size >L, each physical file <=L and no active transaction. The second WAL is small; the DB+WAL overlay crosses L without an individual-file overrun. If these prerequisites fail, stop as fixture failure; do not adapt counts until green.

Now issue exactly one explicit TRUNCATE checkpoint and require **SQLITE_IOERR_WRITE (778)**, at the checkpoint operation, not at preparation or close. With the inspected standard Unix path and this page geometry, copying a page above DB page 1024 requires an extending write. Do not admit BUSY/result-row refusal, SQLITE_FULL, corruption or arbitrary IOERR; unexpected IOERR_TRUNCATE requires source review rather than broadening the accepted set. Require DB length exactly L after the expected write failure and every SQLite file <=L.

Rollback once (a no-op outside a transaction), close once in finally, verify child limits and final files, retain the WAL/SHM if SQLite retains them; never unlink/truncate them manually. No second checkpoint attempt after the expected failure. Parent verifies original limits unchanged and exact helper success plus the checkpoint/cleanup markers. These markers mean the test correctly observed a storage failure, not that a maintenance operation completed successfully. The successful seed checkpoint and failing final checkpoint are separate operations with different preconditions, not retries.

This fixture uses 68 inserts, each 64KiB, and three <=L files. It needs no resource-budget, journal or schema-policy weakening. It positively covers the checkpoint/error/close path that A alone would not cover. Do not make all 100 commits in the historical default-auto-checkpoint run a new required success: that also depends on implementation details. Preserve that historical run as evidence, not a new portable success guarantee.

### Controls and coverage requirements

- Real positive: small committed prefix and successful checkpoint; real required negatives: A's append failure and B's checkpoint failure; direct OS exact-L writes and L+1 EFBIG controls remain unchanged.
- Witness rejection controls: missing exception, wrong operation, wrong error code/name, BUSY checkpoint row, zero successful prefix, oversized/missing DB, unexpected sidecar, missing cleanup marker, changed child limits and nonzero/signal helper exit must fail. Small pure/test-local fabricated records may test rejection; they cannot replace either actual SQLite witness. Do not lower the real cap or fill the disk for a negative control.
- Cleanup assertion failures must remain visible outside expected-error handling. Preserve the independent child guard's signal/receipt/session-stop semantics; no new accepted exit code, catch-all or receipt exception.
- Preserve existing failure-at-acceptance/transaction/session-burn, fresh-resume, strict authority/history, nonempty preparation/continuation and snapshot positives. These resource fixtures do not replace those product witnesses. Actual file-size checks sample states; the no-peak-overrun argument remains the OS/VFS envelope, not a sampling claim.

## Implementation scope and exits

Parent's prospective cap lot may allow only `tests/service/test_recovery.py` (A, new B and adjacent small test-local controls) plus a bounded appended handoff in `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. No edits to service code, guard helper/conftest, dependencies, limits, schemas, archives or historical reports. If a shared helper becomes necessary, return a concrete scope request rather than silently edit the guard. Preserve all unrelated existing test bodies. No commits or implementation occurred in this advisory.

Parent adoption should state the exact amendment and source base. Sol-low then implements in a fresh owned candidate, records old/new body delta, upstream/runtime identity limits, manifests and bounded commands. First authorized runtime gate is only A/B and directed controls on the existing isolated Linux image with the established serialized resource lane and guarded fresh helpers. Stop on first unexpected result, signal, missing receipt or cleanup failure; preserve evidence. No repeated run, count tuning, alternate runtime, weaker assertion or broad campaign follows automatically. Require fresh independent Sol-low full-diff review and separately authorized bounded independent positive executions on the exact frozen candidate. The remaining current cap HOLD closes only on those actual receipts and parent acceptance, not this source argument.

No ordinary diagnostic is selected now. Exact historical checkpoint timing is unnecessary to repair the invalid premise. If the selected witnesses contradict inspected source, that is new evidence requiring a new bounded decision; it is not permission to enter a native-causal reducer or retry the original workload. No production checkpoint/error-handling change is justified here. In particular, do not turn close return into proof of completed checkpoint/durability or add a logical max_page_count policy. A demand to report every internally suppressed SQLite checkpoint failure would be a different production contract requiring its own design/review.

After cap review/acceptance, parent combines exact reviewed ownership and guard with this reviewed cap delta and the **sole already approved** fresh-exec adaptation of `test_b22c_denied_fault_update_inherited_checking_new_epoch`: same query_only/checking/transient-growth body, exit 73 and every parent persistence/resume assertion, bounded JSON. Preserve both original candidates. Resolve shared test edits explicitly, freeze the new tree, and obtain exact combined review plus guarded remaining-entry/four-B2/full validation before integration. Individual 61/93/etc. receipts do not qualify that combination. Then resume the distinct T2/C lane and its 1295 matrix/frozen preservation gates under parent scheduling, followed by final Astra; no self-integration.

All 19 Linux mechanisms as a complete set, actual ext4/VFS/SHM/peer credentials, C1 mandatory `b2_pinned_launcher_context_unavailable`, B3/C1/C2/install/durability, native Darwin V1/V2/V3/fresh-helper/collection causes, A1 drift and hardware/browser/acoustic/activation remain open. No Linux result clears Darwin or installed-host gates. Preserve 1MiB snapshot/result distinctions, 64KiB documents, 2048/16MiB inspection, H/charges, locks, once-only ownership, accepted/no-attempt authority and bounded exact-child cleanup.

## Advisory completion

Source/evidence reading and official upstream-source retrieval only. No tests, diagnostic workload, SQLite open, imports of candidate code, container launch, dependency installation, delegation, commits, integration, push or deployment. Sole repo write is this report. No retained evidence, worktree source or STT data changed. **Design unblock: adopt the two mandatory physical-cap witnesses; execution and integration remain gated.**
