# P1B2.3 lot K — independent FIRST review

2026-10-17 · gpt-6.1-sol / low. **NO-GO for lot K at exact candidate `cc3d790baead83d79117e372461668bfbda34076`.** One medium blocking connection-envelope gap remains in the bounded route's private writable validation. This is a bounded source/E1 verdict, not a production P-builder, apply, installed-Linux or release verdict. No candidate repair was made.

## Identity and scope

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-3e867443` read-only. Entry HEAD is exact and candidate status is empty. Actual baseline is `8ef0b0c91006f2e2c9c4ff63185ec77e9e440c84`; corrected frozen source/tests are `5d650eae1cdf3a4990f5cf8ba88167f2a8820954`. Frozen-to-final changes only the handoff. Baseline-to-final changes exactly:

- `service/recovery.py`
- `service/resource_store.py`
- `tests/service/test_recovery.py`
- `tests/service/test_admission.py`
- `docs/agent-interface/phase-1/RECOVERY-P1B2.md`

Baseline code/tests equal reviewed B2.3a `ee650cc7920d1e237c21c6b795cb4cc6ac4573e5` across service/worker/scripts/tests. No `59c4a7` preparation implementation is approved or adopted by this review. No applicable ancestor/scoped AGENTS.md was found. Root's preexisting untracked research is preserved.

Read the full accounting arbitration, kernel Astra gate, original BOUNDARY, candidate RECOVERY-P1B2 contracts/tranches/corrections/lot K and explicit Unix-VFS correction, P1B23a-r2, P1B22d-r5, orchestration protocol and R1 fault-authority context. Inspected the actual process/session/connection/snapshot/kernel delta and callers, resource arithmetic, archive validation, locks, transaction failure paths and tests. Memory registry search returned no relevant lot-K authority; no memory-derived acceptance is used.

## F1 — medium — private writer bypasses K connection envelope

**Location:** `service/recovery.py:1849`, reached from `continue_kernel:381` through `_pending_prefix_locked:1832-1860`. Related hardening is `_maintenance_connection:249-276`; the regression at `tests/service/test_recovery.py:3522-3535` checks only `session.db`, and the VFS regression at `3604-3614` checks only the original writer.

The initial bounded owned-snapshot route executes `with sqlite3.connect(path) as candidate:` for its prospective archive/SQL validation. This is a writable private SQLite connection using the registered DEFAULT VFS. It does not call `_maintenance_connection`, so it has no explicit Unix VFS, fixed temp/mmap/trusted-schema configuration, extension-disable call or escape-denying authorizer. This is already reachable in K; it is not merely missing future P capacity-rehearsal implementation.

I independently completed a coherent owned-snapshot five-step invocation while intercepting precisely this existing private connection. A strengthened probe checks at its FIRST production `INSERT INTO p1_recoveries`, after any configuration opportunity: `ATTACH ':memory:' AS extra_db`, `DETACH extra_db` and `PRAGMA mmap_size=1048576` all succeeded; the latter returned `(1048576,)`. The same ATTACH/mmap operations first refused on the original session connection. The invocation then returned `complete`. This probe changed only disposable private connection settings, without changing an accepted document, plan digest, source row, snapshot hash or candidate file. Output: `[{'at_first_insert': True, 'attach_allowed': True, 'mmap': [1048576]}]`, exit 0. An earlier constructor-time diagnostic also succeeded, but the first-INSERT probe establishes the configuration gap at actual use.

Consequence: the claimed bounded maintenance call chain still opens a private writer outside the selected standard-Unix/fixed-pragma/no-ATTACH model. Its validation can run under different VFS/settings from the original writer. The pre-handoff correction pins original/prewriter/backup connections but misses this relevant writer. This violates the explicit K connection requirement and leaves a hole before P can reuse the foundation. **This does not demonstrate original-file growth above L, malicious-caller SQL execution, or a physical-allocation failure.** The callback and SQL remain internal trusted code; the blocker is the concrete incomplete connection invariant, not a hypothetical public SQL endpoint.

Minimal fix: give the bounded route's prospective private writer an explicit fixed `vfs=unix` URI and the maintenance connection configuration/authorizer, with an explicit close. Keep ordinary B1 behavior and apply closure intact. Audit other connections actually reached by this bounded route, including the immutable private archive reader at `1147`, for the same declared VFS/configuration requirement. Add a coherent same-lock owned-snapshot completion regression that observes **every relevant private writable open**, checks its fixed URI/configuration, and requires ATTACH, unsafe mmap/mode/page-limit/extension changes to refuse on that connection. Retain the current original-writer test. Full production P rehearsal remains future work; fixing this existing writer is K work.

Reproducer, from the exact candidate (existing interpreter; E1 OS/host model only):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PY'
import runpy, tempfile, sqlite3
from pathlib import Path
import pytest
m = runpy.run_path('tests/service/test_recovery.py')
r = m['recovery']
with tempfile.TemporaryDirectory(prefix='review-k-') as d, pytest.MonkeyPatch.context() as mp:
    fixture = m['installation'].__wrapped__(Path(d))
    roots, scope, _, _ = fixture
    intent, _, _ = m['core_fixture'](fixture)
    m['_lotk_model'](mp, roots, scope, intent)
    original = sqlite3.connect
    def operation(session):
        for sql in ("ATTACH ':memory:' AS extra_db", 'PRAGMA mmap_size=1048576'):
            try:
                session.db.execute(sql)
            except sqlite3.DatabaseError:
                pass
            else:
                raise AssertionError('original connection escape')
        owned = session.capture(intent['plan_id'], intent['plan_digest'])
        seen = []
        class Audited(sqlite3.Connection):
            checked = False
            def execute(self, sql, *args, **kwargs):
                if not self.checked and sql.startswith('INSERT INTO p1_recoveries '):
                    self.checked = True
                    super().execute("ATTACH ':memory:' AS extra_db")
                    super().execute('DETACH extra_db')
                    mmap = super().execute('PRAGMA mmap_size=1048576').fetchone()
                    seen.append(dict(at_first_insert=True, attach_allowed=True, mmap=mmap))
                return super().execute(sql, *args, **kwargs)
        def connect(path, *args, **kwargs):
            if (isinstance(path, Path) and path.name == 'snapshot.sqlite'
                    and path.parent.name.startswith('p1-evidence-')):
                assert not kwargs.get('uri', False)
                kwargs['factory'] = Audited
            return original(path, *args, **kwargs)
        sqlite3.connect = connect
        result = session.continue_kernel(intent, prepared=owned)
        assert result['status'] == 'complete' and len(seen) == 1, seen
        return seen
    print(r._run_bounded_maintenance(roots, scope, operation))
PY
```

The first diagnostic attempt used reserved SQL alias `escape` and failed with a syntax error; it is not defect evidence. Corrected alias `extra_db` produced the successful behavioral receipt above. No real limit was lowered in the reviewer/pytest/shell or probe child: `_lotk_model` explicitly replaces the Linux/limit/host boundary. No actual Linux enforcement claim follows.

## Preserved and usable portions of the foundation

The new seam is nonvacuous: it captures an actual committed original snapshot containing the immutable current plan, passes the grounded row/archive oracle, retains its exact hash and durable bundle, and executes the existing five-step kernel under the owning session. It does not supply a production P builder or P's required production-generated positive. Existing semantic-fixture helpers remain internal and cannot enable public apply.

Process setup is confined to the newly forked one-shot child. It deliberately ignores SIGXFSZ so extending I/O reports EFBIG, installs both fixed soft/hard `RLIMIT_FSIZE=4_194_304`, and checks exact readback, PID/child identity and single-thread conditions. No caller limit/environment/capability option, raise/restore path or live kernel retry was found. Both locks span capsule/prewriter checks, original open, capture/continuation and close/checkpoint. Trusted host freeze and the limit are separate prerequisites. DB/WAL/SHM ownership/regular/single-link/size checks and unknown-sidecar refusal precede original writable open and repeat on unfinished invocation; larger files are not truncated. Original connection and backup source/destination explicitly select Unix VFS. Original WAL is required rather than converted; FULL, trusted_schema OFF, memory temp store and mmap zero are set. Its authorizer denies the assessed escape operations. F1 concerns the additional private writer.

The supported OS proof remains conditional on local Linux/ext4, standard Unix VFS regular-file write/pwrite/truncate/ftruncate behavior, no mmap EOF extension, no external/child writer, unknown VFS, ATTACH/extension/trigger escape or privileged limit-raising launcher. C1/C2 must qualify the actual build/kernel/mount/UID/capabilities/entrypoint and freeze. A cap failure can leave accepted recovery pending/quarantined; progress is not guaranteed.

Snapshot ownership is checked by exact object, session, single consumption, plan identity and fixed SHA-256. Current plan/source/history and actual snapshot rows are revalidated by the shared kernel/oracle. Supplied frozen bytes no longer cause a recapture merely because initial acceptance is new. The bounded initial route requires its owned capture; continuation without it reads exact owned accepted bytes and bundle. Receipt acceptance/fence and durable original bundle precede Core SQL acceptance. No restore-over-Core exists. Kernel exceptions burn the session; ambiguous COMMIT is inspected by a fresh invocation, not retried live. Actual bundle/Core-intent deaths and fresh-interpreter owned-input resume are in the preservation population, with Linux/host boundaries explicitly modeled.

## Independent accounting and namespace audit

An executable byte comparison confirmed actual ROOT `service/resource_store.py` equals baseline `8ef0b0c`. Against that original, the **entire bounded measurement loop**, complete ordinary ResourceStore prefix, and recovery-scan suffix from job/staging eligibility through max/high-water, all demand charges, quota/free reserve and exact accepted-output equality are byte-identical. This independently audits the retained extraction rather than accepting `59c4a7` or the handoff's assertion.

`M` counts descendant directories and all measured file `st_size`, including sparse logical length; it excludes each measured root directory's own size. Static assets/prior archives/staging remain included. The three mutable original files are DB/WAL/SHM. Only flat snapshot and manifest are added under control/core before scan; the flat certificate follows scan/Core COMMIT. Receipts/bundle/descriptors/journal/replacement temporaries are outside these measured roots and retain their independent bounds. No counted descendant directory receives a new entry and staging is unchanged; counted directory-growth delta is zero under the supported freeze/namespace.

The private calculation is exactly `U_all_early=M+3*4_194_304+1_048_576+2*65_536`; it requires `U_all_early<=H`, `H+ALL retained disk charges<=unchanged budget`, finite arithmetic, current reserve and five-entry headroom within 2,048. The original scan repeats actual reserve/equality checks. Snapshot <=1 MiB, manifest/control certificate <=64 KiB and original aggregate 2,048/16 MiB/journal/receipt-certificate bounds remain. No H inflation, budget automation, deletion credit or changed scan equality was found. This metric is file length, not physical allocated blocks or host-disk consumption. P must still invoke the early guard before its future schema/plan writes and perform its concrete capacity/semantic rehearsal.

## Fresh verification

Existing interpreter only; no installs, bytecode or pytest cache writes. Focused and full campaigns are sequential on unchanged exact candidate source/tests:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q -rs
```

Fresh K-specific command adds `tests/service/test_recovery.py tests/service/test_admission.py -k lotk -q -rs`: **34 passed, 1 skipped, 796 deselected in 17.32s**, exit 0. It briefly overlapped the focused run on separate disposable roots. Focused result: **884 passed, 1 skipped in 558.21s (9:18)**, exit 0. Full service/release ran afterward on the same unchanged source/tests: **1,036 passed, 19 skipped in 600.79s (10:00)**, exit 0. All claimed final populations are independently confirmed; passing suites do not close F1. The exact embedded first-INSERT report reproducer was also extracted/compiled in memory and rerun successfully, exit 0. The coherent old-prefix forbidden-recapture RED and pre-handoff default-VFS RED, interrupted focused exit -15 before full started, and corrected frozen sequence are historical implementation receipts, not reviewer executions.

## Remaining limits and write boundary

All 19 platform skips remain unqualified: 18 inherited Linux peer-credential cases (10 authenticated exchange, 1 forged control proof, 1 replayed hello, 6 wrong execute), plus the real Linux RLIMIT_FSIZE child/SQLite test. Darwin E1 OS/host models are not actual Linux cap evidence or actual-host authentication. No Linux qualification workaround ran.

Every apply entrypoint remains unconditionally unavailable and enrolled activation closed. P/B2.3b production preparation and its production-generated positive, B2.3c, B3, C1/C2, parent integration and final Astra remain pending. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no cause or waiver. Physical ENOSPC/power-loss/daemon/VM/controller durability, actual installed operation, clean host, audio/hardware/engines and public/unattended acceptance remain unqualified.

Final HEAD is still exactly `cc3d790baead83d79117e372461668bfbda34076`; candidate status is empty. `git diff --check 8ef0b0c HEAD` passes. Frozen-to-final remains handoff-only, with no source/test delta. The exact five-path scope remains unchanged. No source/tests changed during or between the sequential focused/full suites.

Only this root report is written as a repository artifact. No candidate/source/test edit, commit, subagent, install, Docker/network/other-user-proc/mount/audio/model/paid/E2E runner, push/deploy, prune or user cleanup. Disposable local probe/test roots only; unrelated research/worktrees are preserved.
