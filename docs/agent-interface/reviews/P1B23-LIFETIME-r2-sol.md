# P1B23 lifetime round 2 — fresh independent Sol-low review

2026-10-23. **NO-GO for the complete lifetime contract on exact `c8e94cf42dbe7724e6a9b27b5cc08e0195c924aa`, frozen source/tests `fa63b46064abf2320ff6af2e9890cf02dff476d2`. Original R1 F1 actual-timeout path is CLOSED; a distinct constructor cleanup exclusion defect F2 remains.** No integration, T2, Phase 1, native-cause or campaign GO.

## Identity and scope

Reviewed ABS `/Users/ben/.txcode/worktrees/py-audio-box/lifetime-f1-final-20261023`. Exact HEAD and empty porcelain verified before and after all executions. Frozen-to-HEAD diff over service/worker/scripts/tests/clients/contracts empty; only appended RECOVERY-P1B2 follows the frozen pin. Authorized baseline `0c71e1bc85ed7fe3293be6bffe671867eadc5e27` executable paths independently equal reviewed T1 `7b9cafdc2073eba52149ba46fa9253706ace3015` and integrated `3d764b8e16b58979125103c4198fe1a61efd2812`.

Chronological scoped replay equality independently verified over all six executable/test path groups: `4e7703c == e5c9cc3`, `1823bd6 == 254ef56d`, `94421f2 == 4259415d`. Original `t3-5d6bc116` remains clean at `4259415daaf5acecfe53ed9d44f9895207a00d1b`; C `t3-48d615de` remains clean at `1d741cb328228e67f99a096e9d6a9c1acd1f01b6`. Neither imported nor executed. No applicable ancestor/scoped AGENTS.md found.

Read entire LIFETIME-r1, RUNTIME-LIFETIME-ASTRA (original scope/matrix/limits), T2-r1 V1/V2/V3, canonical orchestration protocol, relevant ledger chronology and T1-r3 invariants; candidate RECOVERY-P1B2 lifetime handoffs from line 2024 through EOF. Inspected entire executable/test delta and callers, including existing private/offline locking. Against baseline, production changes exactly Core.__init__, Dispatcher._run, enrollment_gate, startup_read, plus new _startup_read_lock. All three old owning test files remain exact byte prefixes of baseline and original frozen lifetime tests. Core and conftest are byte-identical to original frozen lifetime candidate. No old test deletion/skip/xfail or semantic rewrite; diff-check passes. Memory supplied scope/evidence-tier guidance only.

## Blocking F2 — constructor releases exclusion after an unsuccessful DB close

**Location: service/core.py:143–150, specifically the unconditional `(self.db, self.owner)` cleanup loop.** If construction fails after connect and `self.db.close()` raises before releasing the actual SQLite handle, the loop notes the failure but proceeds to close `self.owner`. The primary traceback retains the incomplete instance/connection. A subsequent Dispatcher can acquire `_startup_read_lock` and enter startup_read while that prior original connection is still open. This violates the requested “control ownership released ONLY after DB close” exclusion invariant. Ordinary Core.close at lines 153–156 correctly retains the owner when DB.close raises; constructor cleanup does not.

Independent harmless actual-SQLite reproducer `/tmp/pab-lifetime-r2-close-failure.py`:

1. Initialize and cleanly close an owned temporary Core root.
2. For one subsequent constructor, use a real sqlite3.Connection subclass whose close raises RuntimeError **before** invoking actual SQLite close; inject a primary ValueError at persistence.initialize.
3. Retain the primary exception/traceback. Verify the old connection still executes SELECT 1 on its owning thread.
4. Restore ordinary construction and replace startup_read with an entry witness that immediately refuses before copying/opening anything. A new real Dispatcher reaches that witness instead of control_busy.
5. Verify the retained connection still executes SELECT 1 after that reader entry. Finally explicitly close it on its owning thread using sqlite3.Connection.close; remove only the owned temporary root.

Final result, exit 0:

```json
{"primary_retained":true,"cleanup_note":["constructor cleanup failed: RuntimeError"],"next_startup_reader_while_previous_db_open":[true]}
```

This is an injected supported cleanup-failure counterexample, not a naturally observed SQLite close failure, native crash or V3 reproduction. Its first harness attempt incorrectly queried the main-thread SQLite object from the Dispatcher thread and failed with ProgrammingError (exit 1); corrected witness records entry only, with real liveness queries before and after on the owning thread. The corrected probe passed twice, including the final after-entry liveness assertion. No production exception was broadened.

Required bounded response: preserve primary-error diagnostics and deterministically attempt cleanup, while keeping enforceable exclusion when original SQLite release is unresolved. Do not release the sole exclusion merely because close was attempted; do not add blind retries, enlarged timeout or unlimited join. A regression must inject failure before actual DB close, retain the traceback, attempt same-process and separate-process startup, prove refusal before reader entry, and account explicitly for eventual resource/owner cleanup. If independent owner release on any cleanup failure conflicts with the required exclusion contract, return that precise conflict to the parent/Astra; this reviewer does not waive it. This is narrower than claiming arbitrary kernel cleanup can be bounded.

R1's positive L3 assessment correctly established encompassing cleanup attempts and primary-error preservation, and explicitly did not prove resource release after a raising close. It did not establish this stronger lock-until-release property. F2 therefore does not discard the prior positives or treat an attempted close as proven closure.

## F1 closure and preserved positives

The original paused ordinary Core.close / actual five-second timeout F1 is closed. Dispatcher acquires a validated existing control.lock with nonblocking exclusive flock before startup_read and enrollment_gate. Missing root/lock discovery does not create artifacts; missing lock with an existing database refuses. O_NOFOLLOW/O_NONBLOCK, regular-file/current-user/single-link checks precede the read-side critical section. Descriptor close is scoped; failed/nested acquisition closes its own FD and does not unlock the other owner.

Independent `/tmp/pab-lifetime-r2-independent.py` holds real Core.close across the unchanged constructor timeout, then tries a real second Dispatcher and a separate harmless interpreter. Result, exit 0:

```json
{"elapsed":5.003,"reads_during_close":[],"outcomes":["timeout","same_process_refused","separate_process_refused","db_closed","reacquired"],"primary_retained":true}
```

The original Admission ValueError retains traceback and a secondary post-real-close RuntimeError note. The probe releases its barrier, joins only recorded owned control threads, observes actual closed SQLite and owner FD, and reacquires Core. Nested read-side lock/Core attempts also refuse without releasing the outer owner. This is independent actual-timeout evidence, not borrowed acceptance.

Read-side exclusion ends before Core independently acquires its own lock: no double-acquire deadlock or ownership transfer. Active intervening owner makes Core refuse; absence-to-enrollment drift now refuses before Admission. Existing deployment/completion and repeated exact hello checks cover the supported enrolled gap; source checks remain intact rather than being replaced by the lock. Direct startup_read/enrollment_gate remain private APIs requiring external exclusion; production callers are Dispatcher under the new lock, with offline/maintenance retaining their existing lock scopes. No reentrant production path acquiring this same lock inside those readers was found.

L1 still closes private SQLite before private teardown while retaining transaction contexts, primary errors and fail-closed source checking. No ignored SHM, recapture or retry was introduced. L3 closes ordinary partial resources even with retained traceback; schema/PRAGMA/persistence/sync coverage and successful Core ownership remain preserved, subject to F2 on a close that fails before release. Normal failed-Future publication remains after both pool/Core cleanup attempts; actual timeout is distinct from cleanup completion.

Five additional independent temporary artifact/exception cases (directory, FIFO, symlink, hard link, valid regular lock with exception from yield) passed exact acquired-FD/close counting and fstat EBADF checks, exit 0. They use no SQLite or maintenance fork. Existing native SourceInstability/refusal behavior is unchanged by the delta.

## Fresh commands and results

Candidate cwd throughout. Independently verified CPython 3.14.7, pytest 9.1.1, MCP 1.30.0, tiktoken 0.14.0; FSIZE `(9223372036854775807,9223372036854775807)` unchanged. No install or runtime substitution.

Pytest prefix:

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS \
 PYTHONDONTWRITEBYTECODE=1 \
 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache \
 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider
```

- `tests/service/test_core.py tests/service/test_dispatcher.py tests/service/test_recovery.py -k 'lifetime or explicit_startup_rechecks_hello' -q -rs`: **40 passed, 1721 deselected, 14.23s, exit 0**. Includes original 30 lifetime cases, actual-timeout parameter cases, gap checks, primary/secondary cleanup and repeated hello controls.
- `/tmp/test_pab_lifetime_r2_replay.py -q -rs`: **9 passed, 0.48s, exit 0**. Original independent nine-case file adapted ONLY to the exact candidate import path: eight real SQLite retained-traceback constructor BaseException phases and baseline-reader OPEN versus candidate CLOSED before private teardown. Original probe unchanged. Thus all original 39 review positives are freshly represented within these 49 passes.
- `tests/service/test_recovery.py -k '(startup or enrollment) and not lifetime' -q -rs`: **18 passed, 1654 deselected, 14.91s, exit 0**. Additional old enrollment/startup preservation controls.
- Same locked interpreter with `env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache`, running `/tmp/pab-lifetime-r2-independent.py` and `/tmp/pab-lifetime-r2-close-failure.py`: final exits **0**, outputs above. Read-only provenance/AST/prefix/version checks and five FD cases exit 0; git diff --check exit 0.

**67 fresh pytest passes, no skips/xfails/failures in those selections; one independently proven exclusion defect remains.** Implementer 40-focused/89-CoreDispatcher/32-recovery claims remain attributed; the 89 full-file run was not duplicated. No broad campaign or parallel guard-completion workload was launched by this reviewer.

## Limits and handoff

The existing guard remains prospective, byte-identical to the original frozen guard. Fresh python-c maintenance coverage is the separate lane's unresolved gap, not a change in this candidate; the harmless lock-only interpreter here provides no maintenance-guard qualification. No full guarded campaign GO. V1 native failures, V2 invalid UTF-8 and V3 original SHM disappearance remain UNRESOLVED; neither F1 closure nor the 67 passes establishes native causality or repair.

All T1 caps, strict current-source/history joins, retained H/charges, once-only authority and completed observational behavior remain unchanged. Finite 64-round cleanup/single-reaper and unbounded kernel-latency qualifications remain explicit; they do not waive F2's enforceable exclusion problem. All 19 Linux cap/peer-credential mechanisms remain unqualified; actual Linux 4MiB/ext4/VFS/SHM, mandatory C1 b2_pinned_launcher_context_unavailable, B3/C1/C2/install/durability, T2, final Astra, hardware/browser/acoustic/activation remain pending. A1 inventory drift remains unexplained; Apple Silicon remains documentation-only qualification.

Sole repository write: this root report. No fixes, commits, delegation, other-worktree changes, guard repair, environment/GC/dependency/fork architecture changes, native crash generation, allocator diagnostics, Docker/network/model/audio/paid E2E, parent-limit lowering, integration/push/PR/deploy/reset/clean/prune. All probe-owned SQLite/Fds/threads/processes released; temporary scripts retained in /tmp. Candidate and original/C checkouts remain clean at exact pins after probes. Parent must resolve F2, then combine source and guard in a new candidate and obtain final independent review/qualification before integration.
