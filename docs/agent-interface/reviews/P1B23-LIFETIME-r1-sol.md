# P1B23 lifetime round 1 — independent Sol-low review

2026-10-23. **NO-GO for the complete L1/L2/L3 correction contract on exact `4259415daaf5acecfe53ed9d44f9895207a00d1b`.** F1 below is a reproducible L2 timeout/no-overlap gap. Normal failed-Future ordering is corrected; that narrower property does not establish the required caller/next-startup ordering. No native-cause clearance, T2 GO, integration or platform acceptance.

## Identity and scope

Reviewed ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-5d6bc116`, baseline `0c71e1bc85ed7fe3293be6bffe671867eadc5e27`, frozen executable/tests `254ef56d22e725a99865d9dedfae8df419e16358`. HEAD resolves to the full candidate above. Porcelain empty before and after probes; frozen-to-candidate diff over service/worker/scripts/tests/clients/contracts empty. Frozen C was independently read-only checked clean at `1d741cb328228e67f99a096e9d6a9c1acd1f01b6`; no C imports or execution.

Read the entire RUNTIME-LIFETIME-ASTRA report, including L1–L5, ready brief, required matrix, unresolved cause and limits; T2-r1 V1/V2/V3; T1-r3 contracts; canonical orchestration protocol, ledger chronology and candidate RECOVERY-P1B2 line 2024 through EOF. Inspected full source delta and appended tests/guard. No applicable ancestor instructions found. AST comparison independently confirms only `enrollment_gate`, `startup_read`, `Core.__init__`, `Dispatcher._run` changed, with no added production definition. All three old test files remain exact byte prefixes. `git diff --check` passes. No C replay, production policy/schema/cap/dependency/architecture change.

## Blocking F1 — constructor timeout still permits startup reads during previous cleanup

Locations: `service/dispatcher.py:35`, `:53`, `:129–141`; appended barrier tests `tests/service/test_dispatcher.py:418` onward.

`initialized.result(timeout=5)` returns control via TimeoutError even when the failed-startup thread remains blocked in pool shutdown or Core close. There is no surviving caller-visible cleanup handle or startup-read exclusion at this boundary. The next constructor executes `startup_read` before Core acquires its control lock. Moving only `set_exception` therefore fixes nominal Future failure publication, but does not satisfy Astra's required “next startup cannot overlap that close” and bounded failure matrix. Existing tests release their barriers within two seconds and never exercise the constructor's actual timeout.

Independent actual harmless SQLite witness `/tmp/pab-lifetime-r1-probe.py`: use real Core and real Dispatcher constructor; make Admission raise a synthetic ValueError; pause Core.close on an Event before calling the original close; let the unchanged five-second wait expire; immediately construct another Dispatcher on the same root. Wrap actual startup_read only to record whether cleanup remains paused. Observed, exit 0:

```json
{"first":"TimeoutError","elapsed":5.002,"cleanup_still_blocked":true,"second":"control_busy","reads":[{"during_close":false},{"during_close":true}]}
```

The second Core refuses with control_busy, but its private-copy startup read already overlapped the old owner/close interval. This proves the contract gap, not actual concurrent SHM deletion or V3 causality. Finally releases the barrier, waits for real close, joins probe control-writer threads, then removes only its temporary root. No native crash, signals, GC or allocator probes.

Required bounded response: preserve the five-second wait and no-retry rule while providing an enforceable exclusion/ownership outcome for subsequent startup reads when cleanup remains incomplete, or return this precise conflicting scope requirement to the parent/Astra for a contract decision. Do not merely describe TimeoutError as permission to overlap, enlarge the wait, or add an unlimited join. Add an actual constructor-timeout barrier regression that attempts the next startup while cleanup is still held, proves exclusion before startup_read, then releases and verifies closure/primary-failure evidence. A narrower nominal-Future-only acceptance requires an explicit contract amendment; this reviewer does not grant one.

## Positive source assessment and independent evidence

L1 retains the transaction context inside explicit connection ownership; transaction exit precedes close and close precedes private context final check/removal. Success/early return, connect failure and BaseException paths are covered structurally. Primary failures survive close/private teardown failures with secondary notes/context. Read-only connection flags and source checks remain unchanged; no new transaction or original writable open. This preserves existing SQLite context commit/rollback semantics rather than replacing them with close-only behavior.

L3 encloses namespace, connection, all PRAGMAs/fetch/schema, persistence and final sync in BaseException cleanup. DB and control-lock closes are attempted independently, with the original exception retained. Removed special-branch closes avoid double cleanup. Ordinary Core.close is never called on an incomplete instance; successful ownership remains unchanged. A close that itself fails cannot be represented as proven resource release merely because its failure was noted.

L2 independently attempts both pool and Core cleanup and retains the startup primary through secondary cleanup failures. Normal failure publication follows both attempts. F1 concerns the other observable constructor exit, not a claim that nominal Future ordering is still wrong.

All executions used candidate cwd, with no environment install or parent-limit change. Locked versions independently checked: CPython 3.14.7, pytest 9.1.1, MCP 1.30.0, tiktoken 0.14.0; FSIZE `(9223372036854775807,9223372036854775807)`.

Pytest prefix:

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS \
 PYTHONDONTWRITEBYTECODE=1 \
 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache \
 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider
```

- `tests/service/test_core.py tests/service/test_dispatcher.py tests/service/test_recovery.py -k lifetime -q -rs`: **30 passed, 1725 deselected, 2.32s, exit 0**. Includes actual SQLite readers/resource ownership, both cleanup barriers, retained primary exceptions and synthetic guard controls; no native signal generated.
- `/tmp/test_pab_lifetime_r1.py -q -rs`: **9 passed, 0.42s, exit 0**. Eight independent actual SQLite constructor probes inject a custom BaseException at journal-mode, foreign-key PRAGMA, version fetch, schema script, persistence, query-only and final root/parent sync. Retained traceback, actual closed-DB query refusal and independent nonblocking control-lock reacquisition pass. Ninth probe extracts the baseline startup_read AST, witnesses its connection OPEN at private teardown, explicitly closes it, then witnesses candidate CLOSED at the same seam: deterministic baseline RED property versus candidate GREEN, with a harmless actual SQLite database.
- `env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache /tmp/pab-0c-venv/bin/python /tmp/pab-lifetime-r1-probe.py`: **exit 0**, F1 witness above. Same interpreter for source AST/prefix checks, exit 0.

These are 39 fresh passing pytest cases plus one successful defect witness, not a GO. The handoff's **85 Core/Dispatcher + 177 recovery passes / 116 waits are implementer-attributed**, not this review's executions. No broad campaigns were duplicated.

## Guard assessment and remaining gates

The guard records only enrolled owned terminal waits, returns original wait results, latches unexpected statuses outside broad raises, changes a caught-negative call report to failed and stops subsequent maintenance work. Intentional 73 requires declared source/qualified hook-site witness; 74 requires semantic or narrowly declared fault witness; unrelated PID and synthetic signal/status controls pass fresh. It does not itself reap/signal. Existing registration/hello/refusal code is unchanged by this candidate, and existing refusal tests remain preserved; C-specific Admission/A2/full exact-registration evidence is not freshly qualified here.

Fresh `python -c` maintenance waits do not load conftest. That explicit separate implementation-lane guard gap remains pending; no full guarded campaign GO. No native sequence or native root-cause diagnostic was run. V1 native failures, V2 invalid UTF-8 and V3 original SHM disappearance remain unresolved; lifetime greens are neither causal clearance nor T2 acceptance.

All 19 Linux mechanisms remain unqualified; actual Linux 4MiB/ext4/VFS/SHM/peer credentials, mandatory C1 `b2_pinned_launcher_context_unavailable`, B3/C1/C2/install/durability, final Astra, hardware/browser/acoustic/activation remain pending. A1 inventory drift remains unexplained. All T1 limits/strict history-source joins/H/charges and finite cleanup qualifications remain unchanged.

Sole repository write: this root report. No fixes, commits, delegation, other-worktree edits, environment installation, runtime/GC/fork/dependency changes, native crash generation, Docker/network/model/audio/paid E2E, parent-limit lowering, reset/clean/prune or integration. Owned temporary probes are retained in /tmp; probe threads and SQLite resources were released. Candidate and frozen C remained clean after execution; all WIPs/research preserved.
