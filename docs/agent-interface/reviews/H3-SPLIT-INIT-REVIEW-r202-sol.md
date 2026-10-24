# H3 split initializer r202 Sol — independent combined source review

2026-10-24. **NO-GO for adoption of the bounded internal split-initialization unit.** One blocking integration defect remains in the production dropped-child database path. The carried r199/r200 foundation was included in this review; this is not a review of r201 alone. Host actual observer, dropped registration writer, packaged launch and activation remain CLOSED, as intended. Their absence is not a new finding or a new design gate. No full Lot A or installed acceptance is granted.

## Provenance and combined delta

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/r201-split-init`, branch `impl/r201-split-init`, clean on entry and after review. Verified linear parent chain:

- Base `55d5e9f1675b5bb27dc786dbbe54c17e4d996311`.
- Carried r199 `57a258f87effe125533c24f5d980458d1bb54448`.
- Carried r200 `4341092f8d1e53d233f70ac45249dc225de935b3`.
- New source/tests `e4f8414833de90feab885d1b32bc980c0320c9d5`.
- Documentation HEAD `8aa9a5d800ab9da01a29e9bae2db2a945e6e17cb`.

Stable patch IDs match original r199 `eaccc205b70faa84bb9aeca80ebb6ae2e3109b69` to carried r199 (`3a980787dcafde40570419188a01f2719b70a9d6`) and original r200 `45d69668b478f5c9f509f1f3d231420c4d7ddf2e` to carried r200 (`64d6dd9c1d5bded832ac059385aa1fa1a5ccc599`). No duplicate foundation patch. Reviewed base-to-HEAD combined delta: 14 files, 2,105 insertions / 585 deletions, including documentation. Host extraction differs from the base operator implementation only by removal of its shebang and CLI path bootstrap; the compatible script supplies the bootstrap and calls the extracted main.

Read full normative root r198, root r199/r200 checkpoints and candidate r201 handoff. Inspected all changed source/test paths and relevant Core, admission/resource, execution/configuration, journal/transport and recovery dependencies. No applicable ancestor or candidate AGENTS.md found. Bounded memory-registry search had no matching premise. No delegation.

## Blocking finding

**R202-1 — P1: the fd-based child path is handed to SQLite, which canonicalizes it back to the inaccessible host pathname.** `service/persistent_init.py:266` calls `_split_core` with `/proc/self/fd/<control_fd>`. `_split_core` constructs ordinary Core at that path (`service/persistent_init.py:230`); Core then calls ordinary `sqlite3.connect(self.root / 'core.sqlite', ...)` at `service/core.py:230`. There is no SQLite VFS or other adapter preserving directory-fd-relative opens.

The Unix SQLite path canonicalization resolves symlink components, so the inherited-fd spelling does not preserve the intended access bypass at the SQLite boundary. For the supported host layout with root-only Docker ancestors, the permanently dropped 10001 child can create the Core directories and empty database through the fd path, but SQLite's subsequent open of the canonical host path requires traversal of those ancestors. It fails rather than completing fresh enrollment. Parent retains `p1-core-intent`, receives child failure and correctly quarantines; explicit completion cannot reconstruct that partial Core. Thus the authorized ordinary fresh-init case can be stranded by the implementation itself.

Evidence boundary: an inert local SQLite probe created `real/`, a directory symlink `alias -> real`, and connected through `alias/probe.sqlite`; `PRAGMA database_list` reported the canonical `real/probe.sqlite` pathname. This confirms the overlooked canonicalization in the available SQLite adapter. The Linux/root-only-ancestor consequence is source analysis, not an executed Linux permission result. No actual dropped child, UID transition or Docker host was used.

The 32 new tests do not cover this integration: `tests/service/test_split_init.py:87` passes ordinary fixture paths from its fd map to `_split_core`, bypassing both the production `/proc/self/fd` spelling and the real child exec. The successful injected ownership observations therefore cannot close this defect. Resolve the SQLite pathname integration while preserving the existing no-root-writer/no-permission-relaxation requirements, and add a focused test that exercises the actual path adapter rather than substituting ordinary paths. This is a correction within r198's existing unit, not a request to implement host registration or activation.

## Reviewed properties and residual limits

No second blocking finding established in this bounded inspection:

- Explicit singleton policy selection remains required; retained metadata does not select a UID exception. Configuration and installation/private records use shared closed parsers, fixed roles, local volume identities, configuration digest and context/profile/bundle/image/mount bindings. Persistence stores the validated scope. Reconciliation, snapshot and startup/gate parsers were updated together; startup/activation still refuse the installed scope.
- Fresh directory descriptors are locked control first, then receipts/IPC/staging. Distinct inode and lexical non-nesting checks, no-follow traversal, authority callback brackets and role rechecks constrain the fresh transaction. The callback is trusted external authority supplied by a future coordinator, not actual host inspection implemented here.
- The private record precedes role transitions. New file creation is exclusive, checks parent and new inode identities, sets only new-inode ownership/mode, writes completely and syncs. Deterministic output intent precedes creation. Missing attempted output, partial bytes, ambiguous Core and enrolled completion refuse; there is no remint or rollback repair path.
- The production launcher uses isolated Python, empty environment, bounded public stdin, closed inherited descriptors and no receipts fd/seed. Child permanently sets all UID/GID slots to 10001, clears groups and sets umask before request interpretation. Parent synchronously waits/reaps, retaining exclusions; failure cannot publish enrollment. These are inspected source properties, not OS privilege evidence.
- Parent uses held Core exclusion and a bounded private WAL-aware SQLite copy for completed schema, singleton/empty-table and reconciliation checks, followed by retained-byte/authority checks and exclusive enrollment-last. No parent original SQLite connection or root sidecar creation was found in this path. The production child defect above prevents treating this as a working supported-host transaction.
- Carried offline readers retain control-before-lane exclusion, fixed owner/group/mode/link checks, stable reads, private query-only copies and teardown error propagation. Reconciliation applies control ownership without changing worker attempt allocation. Detached journal inspection reuses the full validator without constructing a writable Journal. Ordinary worker startup remains legacy; installed startup validation is deferred explicitly.
- Legacy B2 writer routes retain prewrite scope fences, with private-copy DB scope verification before ordinary writable connections. Split `offline(write=True)` refuses immediately. Host observer/registration extraction preserves historical behavior; no installed provenance or registration writer is silently enabled. The fd hello primitive retains peer authentication and hello-only wire behavior; real helper spawn/reap integration remains deferred.

This is not exhaustive proof against hostile host-root mutation, process crash timing or OS permission behavior. Actual authority, descriptor/permission/setgid/peer behavior, competing real processes and Linux child failure/reap qualification remain parent-owned evidence obligations already defined by r198.

## Independent execution coverage

Inspected both conftests and the selected tests/fixture helpers before execution. Ran only `/private/tmp/pab-0c-venv/bin/python -I -B` with explicit candidate checkout insertion and named unittest modules:

- `tests.service.test_split_init`: 32 tests.
- `tests.service.test_install_ownership`: 14 tests.
- `tests.service.test_host_install`: 5 tests.

**51 tests PASS, exit 0, 5.308 seconds.** Real disposable files, descriptors, locks, SQLite, links, fsync and injected identity/child/hello observations only. The new suite covers creation/completion/quarantine/configuration/identity/exclusion negatives; retained cases cover role readers/private copies, journal/reconciliation, B2 scope refusal and extracted adapters. No actual setuid/setgid/chown was executed. Normal test-owned temporary cleanup only.

The implementer's reported **82 = 32 new + 19 retained + 31 direct legacy invocations** is not an independently rerun 82-test result here. No retained direct legacy harness was located or executed; inspected changed recovery test imports and relevant production legacy boundaries instead. `git diff --check 55d5e9f..HEAD` passed. The additional inert canonicalization probe passed and removed only its own temporary fixture.

No pytest runner/guard, full suite, HELD operation, maintenance kernel, actual Linux child, Docker, network/listener, model/ML/hardware or production runtime execution. No source edits, commits, source adoption or historical cleanup. Sole persistent write is this root review report.

## Handoff

Return the bounded initializer unit for R202-1 correction and focused independent re-review. Preserve carried source and retained partial-state semantics. Registration/observer implementation and activation remain separate closed work; this NO-GO does not reopen r198's topology decision or add a new acceptance gate.
