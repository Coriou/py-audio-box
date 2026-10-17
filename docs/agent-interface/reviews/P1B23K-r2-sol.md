# P1B2.3 lot K — independent ROUND 2 review

2026-10-17 · gpt-6.1-sol / low. **GO for lot K's bounded source/E1 storage-and-snapshot foundation at exact candidate `6456ff42ed8d588713d5c8d5530c86f70b9a174a`. F1 is closed; no new concrete blocker found.** This is not P production preparation, apply, actual Linux/host, installed or release acceptance. Parent integration and final Astra remain required.

## Identity and reviewed scope

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-38eac3cc`, exact HEAD `6456ff42ed8d588713d5c8d5530c86f70b9a174a`, frozen source/tests `6518c9488715e52faa609d2965fec87f262845d6`, actual root baseline `88463f8`. Entry candidate status is empty. Frozen-to-final changes only RECOVERY-P1B2; source/tests are identical. Baseline-to-candidate changes exactly recovery.py, resource_store.py, recovery tests, admission tests and RECOVERY-P1B2. Correction from imported `0d3441e` changes only recovery.py, recovery tests and the handoff. No protected-path expansion.

Read the full P1B23K-r1-sol first, including its FIRST INSERT reproducer, minimal fix and OS/measurement audit; full accounting Astra assumptions/formula/K exits/snapshot handoff; kernel and original boundary Astra; full candidate RECOVERY-P1B2 contracts, tranches, correction responses and latest connection inventory; orchestration protocol and R1 fault-authority context. Inspected the complete actual delta, reached SQLite call sites, process/session lifetime, lock ownership, snapshot/bundle validators, SQL failure ordering, measurement and tests. No applicable ancestor/scoped AGENTS.md found. Memory registry lookup returned no relevant K authority; no memory-derived acceptance is used.

## F1 — closed in source/E1

`service/recovery.py:1852–1853` now opens the prospective private writer with `mode=rw&vfs=unix`, invokes shared maintenance hardening before its FIRST production INSERT and explicitly closes it via `closing`, including exceptions. `_archive_database:1147–1152` also pins `mode=ro&immutable=1&vfs=unix`, hardens before yielding and closes even on configuration failure. `_maintenance_connection:249–276` retains the escape-denying authorizer; the added `query_only=ON` allowance is restrictive only. OFF remains denied. No ordinary B1 offline/host-state/registration connection was changed to evade its source-preservation contract.

Independently adapted the R1 exact coherent owned-snapshot setup and FIRST INSERT interceptor to the corrected URI and required refusal. It ran through the actual bounded-child seam with explicit Linux/limit/host models. At FIRST `INSERT INTO p1_recoveries`, ATTACH, DETACH, unsafe mmap, DELETE journal mode, page-limit mutation, query_only OFF and SQL extension loading all raised DatabaseError; trusted_schema/temp_store/mmap read back 0/2/0. The invocation completed all five steps and printed `[{'at_first_insert': True, 'escapes_blocked': True}]`, exit 0. Actual parent RLIMIT_FSIZE was unchanged after monkeypatch restoration. No constructor-time assertion supplies this evidence.

The relevant connection inventory was independently traced against all sqlite3.connect sites:

| Reached role | Fixed URI / actual-use check |
| --- | --- |
| Prewriter WAL-copy reader, line 402 | ro + unix; hardening before WAL/registration reads |
| Original writer, line 412 | rw + unix; hardening, existing WAL/FULL before session use |
| Capture source and destination, line 289 | ro / rwc + unix; both hardened before backup |
| Prospective private writer, line 1852 | rw + unix; hardened before FIRST INSERT |
| Every immutable archive reader, line 1147 | ro + immutable + unix; hardened before schema/data reads |

The added repository actual-use inventory regression runs before prewriter/session entry and checks readers, backup source/destination and FIRST INSERT, fixed URI/settings, escapes and explicit close. I separately executed that coherent test in memory with query_only OFF added to every connection's escape audit: exit 0, `expanded actual-use inventory and explicit-close audit passed; query_only OFF also refused`. The original writer escape/VFS tests remain. No second actually reachable bounded-route connection omission was found. Ordinary offline, enrollment-gate and host-state readers are separate call paths, not hidden bounded-route opens.

Reproducible FIRST INSERT correctness probe (run from candidate cwd; exit 0 and the receipt above were independently obtained):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PY'
import runpy,tempfile,sqlite3,resource,urllib.parse
from pathlib import Path
import pytest
m=runpy.run_path('tests/service/test_recovery.py'); r=m['recovery']
parent=resource.getrlimit(resource.RLIMIT_FSIZE)
with tempfile.TemporaryDirectory(prefix='review-k-r2-') as d, pytest.MonkeyPatch.context() as mp:
    f=m['installation'].__wrapped__(Path(d)); roots,scope,_,_=f
    intent,_,_=m['core_fixture'](f); m['_lotk_model'](mp,roots,scope,intent)
    original=sqlite3.connect
    def operation(session):
        owned=session.capture(intent['plan_id'],intent['plan_digest']); seen=[]
        class Audit(sqlite3.Connection):
            checked=False
            def execute(self,sql,*a,**kw):
                if not self.checked and sql.startswith('INSERT INTO p1_recoveries '):
                    self.checked=True
                    for escape in ("ATTACH ':memory:' AS extra_db",'DETACH extra_db',
                            'PRAGMA mmap_size=1048576','PRAGMA journal_mode=DELETE',
                            'PRAGMA max_page_count=10','PRAGMA query_only=OFF',
                            "SELECT load_extension('foreign')"):
                        try:super().execute(escape)
                        except sqlite3.DatabaseError:pass
                        else:raise AssertionError(escape)
                    for key,value in [('trusted_schema',0),('temp_store',2),('mmap_size',0)]:
                        assert super().execute('PRAGMA '+key).fetchone()==(value,)
                    seen.append({'at_first_insert':True,'escapes_blocked':True})
                return super().execute(sql,*a,**kw)
        def connect(database,*a,**kw):
            name=urllib.parse.urlsplit(str(database)).path
            if (Path(name).name=='snapshot.sqlite' and
                    Path(name).parent.name.startswith('p1-evidence-') and
                    'immutable=1' not in str(database)):
                assert kw.get('uri') and 'mode=rw&vfs=unix' in str(database)
                kw['factory']=Audit
            return original(database,*a,**kw)
        sqlite3.connect=connect
        result=session.continue_kernel(intent,prepared=owned)
        assert result['status']=='complete' and len(seen)==1
        return seen
    print(r._run_bounded_maintenance(roots,scope,operation))
assert resource.getrlimit(resource.RLIMIT_FSIZE)==parent
print('parent RLIMIT unchanged; E1 Linux/host model only')
PY
```

## Process, ownership, snapshot and accounting audit

The one-shot child sets fixed soft AND hard RLIMIT_FSIZE=4,194,304 before SQLite opens, deliberately ignores SIGXFSZ to obtain ordinary I/O errors, then verifies exact readback, PID/private-child identity, native task count and Python thread count. No caller/environment/capability-selected larger limit, raise/restore or unbounded production fallback was found. Separate in-memory modeled probes varied PID, private child identity, native task count and Python thread count and each refused; actual parent RLIMIT remained unchanged. These probes did not call setrlimit.

Both normal locks span trusted capsule/prewriter checks, original open, capture/continuation and original close/checkpoint. Trusted host origin/freeze is a separate prerequisite from the process limit. Original DB/WAL/SHM ownership, regular-file/single-link/size <=L and unknown-sidecar refusal occur before writable open and repeat on fresh unfinished invocation; no truncate-to-fit. Original requires existing WAL, FULL, trusted_schema OFF, memory temp and mmap zero. Unix VFS/build correctness and no privileged hard-limit-raising launcher remain qualification assumptions.

Capture reads the actual committed original after its plan and before accepted SQL. Exact object/session, plan identity, frozen SHA-256 and one consumption bind the handoff; actual archive schema/rows/current plan and current source/history/media are revalidated under the same locks. No recapture/reminted digest occurs on the owned initial path. Accepted document/fence and exact durable bundle precede Core intent SQL. Fresh accepted resume uses owned accepted bytes/bundle and original proof/TTL rules, not caller snapshot RAM. No restore-over-Core. Kernel exceptions burn the session; ambiguous COMMIT ends that attempt and requires fresh state inspection. Targeted actual-child-death/fresh-interpreter bundle/Core-intent resume, tamper, no-remint and failure/burn cases pass with explicit OS/host models.

Executable byte comparison independently established actual ROOT resource_store.py equals `git show 88463f8:service/resource_store.py`. Against that original, the complete ordinary ResourceStore prefix, full bounded measurement loop and scan suffix from job/staging eligibility through max/high-water, all demand charges, quota/free-reserve and exact accepted-output equality are byte-identical. No prior preparation candidate is adopted as authority.

M counts all measured st_size, including sparse logical file length and descendant directories, excluding each measured root directory itself. Static assets, prior archives and staging remain included. Only flat snapshot/manifest are added under control/core before scan; the flat certificate follows scan/Core COMMIT. Receipts/bundle/descriptors/journal/temporaries are outside these measured roots and retain their independent budgets. No counted descendant directory gains an entry and staging is untouched, so counted directory delta is zero under the supported freeze/namespace.

The private guard remains `M + 3*4_194_304 + 1_048_576 + 2*65_536 <= OLD H`, then `H + ALL retained charges <= unchanged enrolled budget`, finite arithmetic, current free reserve and five-entry headroom within 2,048. Actual accepted scan repeats real reserve and unchanged exact equality. Snapshot <=1 MiB, manifest/control certificate <=64 KiB and retained 2,048/16 MiB aggregate/receipt-certificate limits remain. No H inflation, budget automation, deletion credit or relaxed eligibility/equality. The metric is length, not allocated blocks or host-disk consumption. P must still call the early guard before its future schema/plan writes and supply production capacity/semantic rehearsal and a production-generated positive.

## Fresh verification

All pytest commands use existing `/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, no installs and disposable local roots. Targeted command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py -k lotk -q -rs
```

**35 passed, 1 skipped, 796 deselected in 18.20s**, exit 0. A preceding same-command invocation's terminal handle was not retained by the orchestration call, so its final receipt is not claimed; this collected rerun supplies the result. Targeted/probe work briefly overlapped focused preservation on separate disposable roots. No source/test edits occurred.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q -rs
```

Focused: **885 passed, 1 skipped in 559.22s (9:19)**, exit 0, `/tmp/p1b23k-r2-focused.log`. Full service/release ran sequentially afterward on unchanged exact candidate: **1,037 passed, 19 skipped in 598.83s (9:58)**, exit 0, `/tmp/p1b23k-r2-full.log`. These independently confirm the claimed final populations. No source/tests changed during or between suites; no frozen preservation failure. The sole repeated targeted campaign above recovered an uncollected terminal receipt, not a hidden correction or broad repeat.

## Qualification and write boundary

Host is Darwin. E1 modeled Linux/host boundaries are not actual local-Linux/ext4/Unix-VFS enforcement or actual-host authentication. The real Linux RLIMIT_FSIZE child/SQLite test and 18 inherited peer-credential cases remain unqualified when skipped (10 authenticated exchange, 1 forged control proof, 1 replayed hello, 6 wrong execute). No Docker/Linux workaround or reviewer/pytest/shell limit lowering.

The selected proof assumes local Linux/ext4, standard Unix VFS regular-file write/pwrite/truncate/ftruncate behavior, SHM extension through limited file operations, no mmap EOF extension, external/background/helper writer, unknown VFS/trigger/extension/ATTACH escape, or privileged launcher raising the hard limit. C1/C2 must qualify actual kernel/build/mount/VFS/UID/capabilities/entrypoint and host freeze. A cap hit may leave accepted maintenance pending/quarantined; completion is not guaranteed. Physical ENOSPC, allocation metadata, power loss, daemon/VM/controller durability, installed/clean-host, audio/hardware/engines remain unqualified.

All apply remains unconditionally unavailable; enrolled activation remains closed. K is not production P/B2.3b completion. P production builder/positive, B2.3c, B3, C1/C2, parent integration, final Astra and public/unattended acceptance remain pending. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no cause or waiver.

Final HEAD remains exactly `6456ff42ed8d588713d5c8d5530c86f70b9a174a`; candidate status is empty. `git diff --check 88463f8 HEAD` passes. Final five-path scope, three-path correction-only scope and handoff-only frozen-to-final delta are independently rechecked. Root has no tracked diff; this new report is the only authored repository artifact. The report's embedded reproducer compiles and Markdown fences are balanced.

Only this root report is written as a repository artifact. No candidate/source/test edit, commit, subagent, install, Docker/network/audio/model/paid/E2E runner, mount operation, push/deploy, prune or user cleanup. Research and other worktrees are preserved. Bounded K GO does not waive any pending qualification above.
