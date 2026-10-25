# r385 — bounded r384 daemon wiring adversarial review

2026-10-25. **BOUNDED SOURCE NO-GO: one blocking source-preservation/bounded-reader defect.** The injected regression suite passes, but the new initial-history reader opens the enrolled WAL database directly and can mutate its source namespace. Full r378 family SOURCE NO-GO; production installed export UNAVAILABLE; release NO-GO; legacy CLOSED. No adoption, merge, activation or qualification authority is granted.

## Authority and exact pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-1f5ee68c` verified clean before and after tests/probes at HEAD `6fdb79625d4e49637dbb8c23c7d86d54e3caa7c3`. HEAD is report-only (sole changed file r384), directly atop source `c6e6fcc2883cb5f9ad6a5206976a3f1088011fc9`, directly atop reviewed report `359e8aeb2bcfab20f9fb8c5d3fd77997ce28f6db`. Verified remaining ancestry: worker source `8749e75519f41d9052978ed43a7127e637006c2f`, r380 `ff9a4d01eb53daad9ec8987b4b6d375f39ab1ae1`, initialization source `c9b391f9e231e7f0b92b455fac6f226970b92d5a`, r379 `c73e109272a3e27a038824ca72fb698a4e6578da`, primitives `3ac457d8532d6fe62764ae760b621a698e30e393`, required base `1bec52ea8e1fedd8411099c0afc5001ff1c3031a`. Diff whitespace check passed.

No applicable AGENTS.md found in candidate descendants, candidate/root ancestor chains or review destination directories. Read full root r378 contract and r381/r383 reviews, candidate r380/r382/r384 reports, complete r384 source/test diff and relevant reader, initializer, writer, subprocess custody, configuration and hello seams. Root input SHA-256:

- r378 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`
- r381 `7fbbe8988b37fa414b09b3a0c5f91d7649c3f2fd0943b168eddf2c933a6e9e9a`
- r383 `4617c07c962b8a9a5ff06aa7f36392d9262f45d837d3f25ece23a74c22554651`

Memory registry search returned no relevant hits; conclusions use current source and execution. Sole persistent write is this root report, mode 0600. No source/helper-script/log writes, commit, merge, delegation or self-adoption. Existing unrelated root evidence was untouched. Only authorized inert temporary fixtures used their existing disposal lifecycle.

## Blocking finding

**P1 — new initial-history snapshot bypasses the bounded, source-preserving private SQLite copy.** `service/daemon_helper.py:402-419`, specifically the direct `sqlite3.connect(path.as_uri()+'?mode=ro', ...)` at line 406.

`mode=ro` and `PRAGMA query_only=ON` do not prevent SQLite from creating/updating WAL shared-memory sidecars. The existing v1 registration snapshot (`service/host_install.py:1148-1154`) opens only a `recovery.private_database` copy; that helper (`service/recovery.py:74-114`) bounds DB/WAL/SHM bytes, captures their identities/bytes and the namespace, and checks source stability around inspection. The new branch instead checks the main DB inode/metadata only. `fresh_namespace` checks existing file privacy and empty media directories, but neither copies the source nor bounds DB/sidecar bytes. `ex.database`'s 2048-row/16 MiB scalar and SQL-progress limits apply after SQLite has opened the source; they do not replace a bounded source copy or constrain unused pages/WAL bytes before opening.

**Independent concrete reproduction:** initialize the existing tiny ExportDaemonTests fixture, read its public enrollment, record `control/core` names, then call `s.helper.registration_snapshot(public)` directly. No other SQLite connection is opened by this minimal probe. It returns protocol 2 successfully and leaves two new source files:

- `core.sqlite-shm`: 32768 bytes
- `core.sqlite-wal`: 0 bytes

The initial namespace had neither sidecar. This proves source mutation despite accepted read-only inspection; the main-file-only postcheck does not detect it. The main fixture DB was 229376 bytes in the instrumented probe. Ownership is injected; SQLite and filesystem side effects are real. The absence of a pre-open byte cap is source-inspected, not tested with large allocation.

Minimal reproduction body after the same pre-import audit hook/path/cwd setup as the authorized runner:

```python
from tests.service.test_export_daemon import ExportDaemonTests
import json, os
case = ExportDaemonTests()
try:
    s = case.initialized()
    public = json.loads((s.roots['receipts']/'p1-init.json').read_bytes())['public']
    root = s.roots['control']/'core'
    before = set(os.listdir(root))
    result = s.helper.registration_snapshot(public)
    print(result['protocol'], sorted(set(os.listdir(root))-before))
finally:
    case.doCleanups()
```

Observed: `2 ['core.sqlite-shm', 'core.sqlite-wal']`; forbidden attempts `[]`.

**Required correction:** inspect a bounded stable WAL-aware private copy under the existing control exclusion, using the existing private_database discipline with SPLIT ownership checks. Preserve exact selected-family/enrollment/v2 validation, the fixed initial rows except registrations, and all empty namespace checks. Add a focused assertion that successful/refused snapshots leave original DB/WAL/SHM bytes, identities and namespace unchanged, and exercise source-byte refusal with finite fixtures/counters. Do not broaden this reader into retained history by removing empty checks, and do not raise global bounds. Re-review the corrected exact pin before bounded GO.

## Other reviewed boundaries

No additional blocking defect found in the inspected slice; this is not acceptance of the full family.

- `service/daemon_client.py:403-424,481-486` explicitly selects the singleton through private export_main; closed config and actual supplied credential production-write/read principal coverage precede Docker construction. Helper import repeats credential corroboration (`daemon_helper.py:286-297`). Credential bytes/digests are excluded from authority records and child stderr/protocol errors are suppressed. Imported config is checked against canonical admitted config. Ordinary selection remains v1.
- `daemon_volumes.py:64-149,185-207` retains eight enrolled volumes and fixed commands, adding only RO/no-copy control-volume `core/assets` at `/references/assets` for the worker. Requested Subpath/NoCopy and boolean types, actual source/name/destination/RO and exact inventory are checked; no whole-control fallback. Unsupported create propagates once. Assets inode cross-view includes helper-private control assets (`daemon_helper.py:380-384`) and both runtime views. Actual Docker mount semantics remain UNRUN.
- `install_child.py:242-325` carries explicit FAMILY through fixed Core/writer/hello callbacks; descriptor closure, permanent drop and capability verification precede callbacks. Existing synchronous collect/reap retains exclusions. No arbitrary action/SQL/command protocol or worker credential/control-root mount was added. Fake drop observations do not prove kernel UID isolation.
- `daemon_helper.py:28-57,371-453` selects strict v2 gate/new evidence, checks canonical config/enrollment and full seal, running image, command, lane/boot, container start times and local roots. Observations bracket writer and notification; non-registration projection comparison remains. Old gate/hello selection rejects new shapes. Initial-history fixed/empty checks remain intentionally narrower than retained history; the blocking issue is how the source is opened.
- `Dockerfile.release:33-57` adds a separate worker-export target with explicit package versions/executable SHA-256 and no unversioned worker apt inheritance. Original existing target/default order is preserved. Executable hash and non-symlink/non-writable checks exist; direct CMD refuses. Complete resolved immutable dependency closure, transitive package provenance and finished image identity are still acquisition/review gates, not proven by source recipe inspection.
- `entrypoint.py:601-603` refuses owner activation; `daemon_helper.py:467-477` refuses ready and runtime up/status/stop before runtime create/start. Sealed registration tests deliberately inject a runtime fixture to reach the bounded seam. They do not enable ordinary new-family production or establish owner notification consumption.

## Independent execution and limits

Inspected `/private/tmp/r384-tests.py` and both candidate conftests before execution; no worker conftest exists. No pytest collection/listener fixture. Ran the exact supplied runner **once**:

`/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r384-tests.py`

CPython **3.14.7**; **28 methods PASS in 19.236 s; forbidden-event attempts `[]`**. This is independent of r384's reported 18.834 s. Composition: 9 ExportDaemonTests, 2 selected v1 DaemonTests, 2 selected ExportInitialization, 1 selected v1 Registration, 12 ExportSealTests and 2 AdapterBoundsTests. Audit hook installed before application/test imports blocks subprocess.Popen, os.system, socket.bind and socket.connect. Fake Popen/Docker/UID/pipes and tiny SQLite/files/flocks are injected evidence only.

Two additional finite inline audited invocations inspected only the snapshot seam. First instrumented SQLite connection targets and source namespace (and queried journal mode); it showed the direct source URI and sidecars. Second used the minimal reproduction above without any additional source connection and confirmed successful snapshot creates both sidecars. Both forbidden-attempt lists were empty. No new helper or log artifact was written. No suite rerun or broad regression campaign.

No actual Docker/network/listener/application child, FFmpeg, ML/hardware, HELD/STT content, install/migration/build/acquisition/download, resource cleanup or heavy waiver. Seven real-child reference-publication tests remain BLOCKED/UNRUN. Actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification all UNRUN. CUDA and native Metal remain mandatory, unqualified/UNRUN.

Owner activation, all retained-history/readers and media accounting, truthful capability, public HTTP/CLI/MCP >1 MiB upload/export/download/reopen/replay/generated-asset journey, integrated cancellation/timeout/output/reap/publication/ACK/capacity evidence and final complete-family Astra review remain parent-owned. Their absence is not the reason for this bounded rejection. **Bounded SOURCE NO-GO for the reproduced reader defect; full-family SOURCE NO-GO; production unavailable; release NO-GO; legacy CLOSED.**
