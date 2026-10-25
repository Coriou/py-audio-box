# r387 — independent r385 P1 snapshot correction review

2026-10-25. **BOUNDED SOURCE GO for the corrected daemon initialization/sealed-registration snapshot boundary.** The r385 P1 direct-source SQLite defect is corrected at the reviewed pin. No remaining blocking finding in this narrow correction. Full r378 family SOURCE NO-GO; new-family installed export UNAVAILABLE; release NO-GO; legacy CLOSED. No adoption, merge, activation or qualification authority is granted.

## Authority and pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-80bf03cf` verified clean at HEAD `a40d1ca2798273e0b812dea9ab304947ab73c499`. Exact direct chain verified: HEAD (report only, sole r386 report) -> source `184a3d5ff1cd4d6f87b94e482eae8a8e7d67a45b` -> r384 report `6fdb79625d4e49637dbb8c23c7d86d54e3caa7c3` -> daemon wiring `c6e6fcc2883cb5f9ad6a5206976a3f1088011fc9`. Required base `1bec52ea8e1fedd8411099c0afc5001ff1c3031a` is an ancestor. Full base-to-HEAD whitespace check passed. Correction changes only service/daemon_helper.py and tests/service/test_export_daemon.py; the subsequent commit adds only r386.

Read FULL explicit root inputs (not substituted with candidate copies):

- `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-DAEMON-WIRING-REVIEW-r385-sol.md`, SHA-256 `eee99057a4ecf96526376351c9b9e50d6379c60179a83c2b09e0dfc3e3de1597`.
- `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md`, SHA-256 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`.

Read full candidate r386 and r384 reports, complete correction diff, private-copy/read/role primitives, exact export database/fresh namespace checks, registration lock and dropped child seams. No applicable AGENTS.md found in candidate descendants, candidate/root ancestor chains or destination directories. Memory registry quick search supplied no relevant evidence; conclusions use current source and execution. Sole persistent write is this root report, mode 0600. No source/helper/log writes, commit, merge, delegation or self-adoption.

## Source decision and evidence

**r385 P1 resolved:** `service/daemon_helper.py:398-415` now passes the enrolled path to `recovery.private_database(path, policy=ip.SPLIT)` and connects SQLite only to the yielded private URI. No enrolled source SQLite connection remains in this branch. This fixes the reproduced creation of original WAL/SHM sidecars rather than merely detecting them afterward.

- `service/recovery.py:74-114` captures DB/WAL/SHM bytes and device/inode/mode/UID/GID/link/size/mtime/ctime plus the source directory namespace, enforces cumulative 16 MiB and finite namespace/deadline bounds before yielding, copies into a private temporary namespace, and compares the source again before opening and after inspection, including exceptional exit. Real SQLite sidecar changes occur only in that private copy. This is stability under the existing cooperative owner lock plus detected changes, not a claim to defeat arbitrary hostile concurrent mutation.
- `service/persistent_init.py:22-25` forwards the selected policy to `install_policy.read_private`; `service/install_policy.py:93-119` performs no-follow descriptor reads, validates fixed control role ownership/mode/link count, bounds size before reading and checks descriptor/path identities and ancestor traversal. SPLIT remains an identity-selected singleton (`install_policy.py:38-72`), with no caller-selected UID map or relaxed owner checks. The copy helper is not a dropped-role substitute: child Core/writer callbacks still follow descriptor closure, permanent drop and capability verification (`install_child.py:291-321`). Kernel UID/capability behavior remains unqualified.
- Existing control exclusion encloses both registration snapshots and the intervening writer (`daemon_helper.py:420-447`); the correction adds neither a second lock nor a coordinator. Exact family/enrolled control root checks precede copying. Exact v2/schema/enrollment/canonical document validation and two-second SQL progress deadline remain in `installed_export.py:162-218`. Initial fixed rows except registrations remain at `daemon_helper.py:408-413`; empty private media/staging checks remain in `installed_export.py:253-269`. No retained-history broadening, global bound increase, fallback, repair or source-sidecar removal was introduced.

The focused tests at `tests/service/test_export_daemon.py:215-313` corroborate the critical behavior: successful snapshots without sidecars and with real live committed WAL/SHM preserve original bytes, recorded identities and namespace; a WAL-only application_id is observed in the private copy; initial-history SQL refusal also preserves source state. Patched tiny byte budgets refuse main DB, DB+WAL and DB+WAL+SHM before any SQLite connection, and a finite namespace cap refuses pre-open. Injected source namespace changes during copy and after query refuse with `snapshot_source_changed` while retaining the external marker and original files. The marker changes simulate an uncooperative source, not real concurrent-process or kernel exclusion qualification.

## Independent execution receipt

Inspected exact `/private/tmp/r386-tests.py` before execution; SHA-256 `b33d367fe427bdbff2f1eaf6ecb54fbaa44aefb8765c73cd06d72fac851a5e9b`. Exact cwd/sys.path point at the candidate. Read both candidate conftests before imports; no worker conftest exists. The service conftest contains a listener/uvicorn fixture, but this unittest runner does not collect or invoke it. The audit hook is installed after importing only sys and before application/test imports, blocking subprocess.Popen, os.system, socket.bind and socket.connect.

Ran exactly once:

`/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r386-tests.py`

Independent result: **CPython 3.14.7; 32 methods PASS in 21.062 s; exit 0; forbidden attempts `[]`.** Receipt retained in this review turn's tool output; no separate log written. Composition: 9 ExportDaemonTests, 4 ExportSnapshotTests, 2 selected v1 DaemonTests, 2 selected ExportInitialization, 1 selected v1 Registration, 12 ExportSealTests and 2 selected AdapterBoundsTests. No rerun, additional probe or broad campaign was needed. This independently corroborates, rather than copies, r386's reported 21.493 s receipt. Candidate clean status rechecked after execution.

## Remaining limits

SOURCE/injected evidence only: tiny SQLite/files/flocks are real; UID, Docker, subprocess and runtime observations are injected. Existing fixture disposal only. No actual Docker, network, listener, application child, FFmpeg, ML/hardware, HELD/STT, installation/migration/build/acquisition/download, operational resource cleanup or heavy waiver. Seven real-child reference-publication tests remain BLOCKED/UNRUN.

This GO closes only the r385 blocker and accepts the corrected daemon boundary. Owner activation/notification consumption, retained-history readers and canonical export joins/media accounting, truthful capability, public HTTP/CLI/MCP upload/export >1 MiB/download/range/reopen/replay/generated-asset journey and integrated cancellation/timeout/reap/publication/ACK/capacity evidence remain parent-owned. Final complete-family Astra review remains required; the new family stays unavailable until complete. Actual Linux CPU, Intel Mac Docker Desktop and OrbStack are all UNRUN. CUDA and native Metal remain mandatory, unqualified/UNRUN. **Full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.**
