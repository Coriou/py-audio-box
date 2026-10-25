# r386 — bounded r385 P1 export snapshot correction candidate

2026-10-25. **Candidate awaiting fresh independent Sol review; no self-adoption.**
Full r378 family SOURCE NO-GO; new-family installed export UNAVAILABLE; release
NO-GO; legacy CLOSED. Owner activation, retained history and the public journey
remain later parent-owned work. Final complete-family Astra review remains later.

## Authority and exact pins

Own isolated checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-80bf03cf`
verified clean at `1bec52ea8e1fedd8411099c0afc5001ff1c3031a`, then fast-forwarded
ONLY this checkout to `6fdb79625d4e49637dbb8c23c7d86d54e3caa7c3`.
Read-only source checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-1f5ee68c`
was clean and still at that exact pin on final inspection. Source candidate:
**`184a3d5ff1cd4d6f87b94e482eae8a8e7d67a45b`**, direct child of 6fdb796.
This report is a subsequent documentation commit. No applicable AGENTS.md found
in this checkout/ancestor chain. No root/original project writes, merge, push,
delegation, review commissioning or adoption.

Read FULL root reports at the explicit read-only main-project paths:

- `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-DAEMON-WIRING-REVIEW-r385-sol.md`
  SHA-256 `eee99057a4ecf96526376351c9b9e50d6379c60179a83c2b09e0dfc3e3de1597`.
- `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md`
  SHA-256 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`.

Also read full candidate `H3-EXPORT-DAEMON-WIRING-r384-sol.md`, both conftests,
and relevant helper, private-copy, ownership, export database and fixture seams.
Memory registry search yielded no relevant hits; conclusions use current inputs.

## Narrow correction

The new-family branch of `Helper.registration_snapshot` now enters existing
`recovery.private_database(path, policy=ip.SPLIT)` and opens ONLY its private URI.
It no longer connects SQLite to the enrolled original. Existing callers retain
the enclosing control exclusion; no second lock or new coordinator is added.
The existing helper supplies no-follow split-role reads, cumulative 16 MiB
DB/WAL/SHM byte bounds before SQLite opens, bounded namespace/deadline, private
0700 temporary namespace and byte/identity/namespace stability checks before and
after inspection. SQLite may create/change sidecars only in that disposable copy.

Exact explicit family and enrolled root, exact v2 schema/enrollment/documents,
fixed fresh SQL rows except registrations, empty private media/staging namespaces,
SQL scalar/row limits and query deadline remain intact. No global bounds increase,
retained-history expansion, automatic repair or removal of existing sidecars.
The production new-family runtime remains unavailable before create/start/notify;
no owner activation, history or public capability was implemented here.

## Focused inert evidence

Runner `/private/tmp/r386-tests.py` is the existing r384 28-method selection with
exact cwd/sys.path changed to this isolated checkout plus four new
`ExportSnapshotTests`. SHA-256:
`b33d367fe427bdbff2f1eaf6ecb54fbaa44aefb8765c73cd06d72fac851a5e9b`.
Both conftests were inspected before imports; no pytest collection or server
fixture was used. Before application/test imports, the runner imports only sys
and installs an audit hook blocking real `subprocess.Popen`, `os.system`,
`socket.bind` and `socket.connect`.

Final command:
`/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r386-tests.py`

Retained final log `/private/tmp/r386-final-tests-2.log`: **32 methods PASS in
21.493 s**, CPython **3.14.7**, forbidden-event attempts **`[]`**. Composition:
9 ExportDaemonTests, 2 selected v1 DaemonTests, 2 selected ExportInitialization,
1 selected v1 Registration, all 12 ExportSealTests, 2 selected AdapterBoundsTests,
and 4 new snapshot tests. `git diff --check` passed.

New tests use tiny initialized SQLite/files with injected UID/Docker and inspect
actual SQLite connection targets. They check:

- Successful snapshot without original sidecars leaves bytes, device/inode,
  mode/UID/GID/link count/size/mtime/ctime and namespace unchanged.
- Live committed WAL plus SHM are preserved identically. A WAL-only application_id
  value is visible in the private connection, corroborating WAL-aware copying.
- SQL initial-history refusal preserves the source after the private file opens.
- Small patched byte budgets refuse before any SQLite connection for the main DB,
  cumulative DB+WAL and DB+WAL+SHM; a small namespace cap also refuses pre-open.
  No huge allocation or global policy change is used.
- External fixture namespace changes during copy and after query refuse with
  `snapshot_source_changed`; external marker bytes are retained and all original
  file bytes/identities remain unchanged. These injected changes are under the
  test's control lock, simulating an uncooperative source change, not a real
  concurrent process or proof of kernel writer exclusion.

The first 32-method invocation is retained unchanged at
`/private/tmp/r386-final-tests.log`: 30 passed, two test assertions failed in
22.870 s, forbidden attempts `[]`. Assertions incorrectly expected one total
SQLite connection; existing detached validation also opens three in-memory schema
check databases. Source preservation had passed. Corrected only the test target
accounting/application_id assertion to distinguish `:memory:` from the private
file, then ran the one final 32-method selection above. No other test campaign.

## Limits and remaining gates

This is SOURCE/injected evidence only. Tiny SQLite/files/flocks are real;
Docker, UID and subprocess observations are injected. No actual Docker, network,
listener, application child, FFmpeg, ML/hardware, HELD/STT, actual installation,
migration, build, acquisition/download, operational cleanup or heavy waiver.
Temporary fixture disposal uses the existing test lifecycle; no actual resources
were cleaned up. Seven real-child reference-publication tests remain
**BLOCKED/UNRUN**.

All actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification remains
**UNRUN**. CUDA and native Metal remain **mandatory, unqualified/UNRUN**.
Fresh independent Sol review of the exact candidate is next, commissioned by the
parent. Full r378 remains SOURCE NO-GO pending owner activation, all retained
history/readers/accounting, public HTTP/CLI/MCP journey and integrated failure,
publication/ACK/capacity evidence; final Astra review comes later. **New family
UNAVAILABLE; release NO-GO; legacy CLOSED.**
