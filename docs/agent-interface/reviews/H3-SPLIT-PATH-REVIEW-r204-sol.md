# H3 r204 Sol — independent round 2 combined split-init adoption review

2026-10-24. **GO for source adoption of the bounded combined r199/r200/r201 initializer unit with r203's path correction. R202-1 is closed at source level.** No additional blocking finding established. This is not Linux actual qualification, full Lot A acceptance, installed acceptance or activation authority. Host observation/registration, dropped registration writer, packaged launch and activation remain CLOSED and parent-owned. Their absence is not a new design gate.

## Candidate and chain

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/r203-path-fix`, branch `impl/r203-path-fix`, clean on entry and after checks. Verified exact linear parent chain:

- Base `55d5e9f1675b5bb27dc786dbbe54c17e4d996311`.
- Carried r199 `f71671fe34077883a879ecb0b5d5cc70239c0b8a`.
- Carried r200 `f8b5682828d96ba563a3fe68842a7e56d5ac6a10`.
- Carried r201 `3de61af50ee7098940dc2db881c15f117bd828d0`.
- Correction source `0b02b138de239753b831698fb6987cd8c83c23cf`.
- Documentation HEAD `d9feecc3124dba3553f2f25cf3dbe57e01ac1f84`.

Independently recomputed stable patch IDs for originals and carried copies: r199 original `eaccc205b70faa84bb9aeca80ebb6ae2e3109b69` and carried copy both `3a980787dcafde40570419188a01f2719b70a9d6`; r200 original `45d69668b478f5c9f509f1f3d231420c4d7ddf2e` and copy both `64d6dd9c1d5bded832ac059385aa1fa1a5ccc599`; r201 original `e4f8414833de90feab885d1b32bc980c0320c9d5` and copy both `b5a729470300b4ff4b05767f8b4404d97c17d726`. Foundation patches occur once in the verified chain. Combined base-to-HEAD delta is 16 files, 2,380 insertions / 585 deletions. The correction changes only the helper, child entrypoint and two focused test files; its documentation is separate.

Read root r198 normative design, r201 handoff, r202 combined NO-GO and r203 fix report. Carried r202's combined assessment of r199/r200/r201, with source rechecks of transaction, readers, writer fences, parser/configuration and activation boundaries; independently reviewed the entire r203 source/test delta. No applicable ancestor/candidate AGENTS.md found. Bounded memory-registry search returned no relevant premise. No delegation.

## R202-1 closure and privileged helper review

The old production fd spelling no longer reaches SQLite. `service/persistent_init.py:252` prepares the mount view before dropping identity, and line 259 passes the returned ordinary paths to `_split_core`. `service/install_child.py:12-14,74-85` constructs fixed `/mnt/pab-install/control` and `/mnt/pab-install/staging` paths backed by the inherited directories. Ordinary Core still opens `control/core/core.sqlite` through ordinary SQLite (`service/persistent_init.py:230`, `service/core.py:230`). These paths have no fd symlink component for SQLite to canonicalize back through Docker ancestors. This resolves the specific source integration defect without root DB construction, host permission relaxation, custom VFS or chroot.

The newly privileged portion remains bounded:

- Root request parsing is now intentional and necessary for setup. Stdin is bounded to 65,536 bytes, the outer shape is closed, and the existing installation/configuration validators run before namespace operations (`service/persistent_init.py:246-252`, `service/install_child.py:53-68`). Invalid JSON/types/records fail the helper. The public record supplies no executable, SQL, mount target, flags, filesystem, mount options or arbitrary bind source path. Fds must be distinct exact integers at least 3; both fixed roles, ownership/mode/type and public device/inode bindings are checked before mounting.
- `LinuxMounts` requires Linux/root, declares libc argument/result types and propagates syscall errors (`service/install_child.py:17-44`). `/mnt` must already be a root-owned non-symlink directory without group/world write. No host directory is created. After `CLONE_NEWNS`, recursive `MS_PRIVATE` on `/` precedes tmpfs and both binds. Any failure aborts; there is no fallback to the old fd path or shared mounts (`service/install_child.py:69-85`). The tmpfs has fixed nosuid/nodev flags, 64 KiB and 16-inode limits. Binds are nonrecursive and derive solely from verified inherited descriptors; each resulting directory identity/type is checked before drop.
- Setup is confined to the fresh synchronous child. Trusted interpreter/source imports are preserved rather than relocating the process root. `-I -B`, an empty environment, cwd `/`, `close_fds=True` and exactly two passed role fds remain in the launcher (`service/persistent_init.py:275-280`). No receipts fd, seed or host callback crosses the exec boundary. The new helper adds only stdlib imports and the existing closed validator. This assumes trusted installed interpreter/dependencies/source; it does not implement their future packaging/host authority checks.
- Groups are cleared, all real/effective/saved GID and UID slots become 10001, umask becomes 077, and slots/groups are verified before Core (`service/persistent_init.py:253-259`). `_split_core` independently rejects root or wrong current IDs/groups before construction (lines 224-230). Standard Linux setuid capability-clearing semantics are relied on; there is no keepcaps, ambient-capability grant or privilege restoration path in this source. Actual capability state was not measured, and arbitrary inherited securebits/capability configurations are not qualified by this review.
- The child creates no subprocess/namespace keeper. Partial setup has no host rollback/deletion path; namespace mounts are released with the sole helper's process lifetime under normal Linux semantics. The existing launcher waits via `communicate`, kills on exceptional completion/timeout and waits for reap before returning the failure (`service/persistent_init.py:281-291`). Parent exclusions remain held. Setup/drop/Core failure cannot publish `p1-core-done` or enrollment. Kill/reap failures cannot become success; actual interruption/kernel-cleanup behavior still needs Linux qualification.

The fixed view requires a safe `/mnt`, working `/proc/self/fd`, libc namespace/mount support, tmpfs and sufficient setup privilege (normally CAP_SYS_ADMIN). Hiding `/mnt` can also hide dependencies placed there; deployment must keep required trusted source/libraries accessible. Missing prerequisites fail closed and retain the attempted Core transaction for quarantine. This is a deployment constraint, not evidence of successful operation on the target.

## Combined assessment retained

No regression established in the carried bounded unit:

- Explicit `SPLIT` selection and shared closed configuration/public/private parsers remain required. Fixed role expectations are not inferred from retained metadata or caller UID. Exact local role identities, scope/configuration digest, CPU profile/context/bundle/image/mount bindings remain corroborated (`service/install_policy.py:25-61,193-292`).
- Root descriptors are acquired and locked control first; distinct/non-nested identity and authority observations bracket the transaction. Protected private intent precedes fixed ownership transitions; exclusive fd-checked deterministic creation retains intent-before-output semantics. Missing attempted output, partial Core, mismatched retained bytes and enrolled completion quarantine rather than remint/repair (`service/persistent_init.py:335-471`).
- Original Core is constructed only in the dropped child. After synchronous reap the parent holds Core exclusion and validates schema, fresh singleton/empty tables and reconciliation on a bounded private WAL-aware copy. It rechecks retained bytes/roles/authority and writes enrollment last (`service/persistent_init.py:448-505`). No root original SQLite connection or sidecar writer was introduced.
- Role-aware no-follow reads, fixed lock identities, control-before-lane ordering, stable private-copy inspection and teardown-error propagation remain (`service/install_policy.py:65-185`, `service/recovery.py:74-237`). Reconciliation's control-role checks preserve existing worker attempt ownership; detached journal inspection reuses the validator without creating a writable Journal (`service/reconciliation.py:104-113,512-575`, `worker/runtime_journal.py:110-131`).
- Split `offline(write=True)` refuses before filesystem entry. Legacy writer paths retain private-record and copied-DB scope fences before writable original SQLite (`service/recovery.py:163-237,463-495`). Shared parsers do not authorize installed startup: `enrollment_gate` and `startup_read` still explicitly require synthetic scope (`service/recovery.py:1065-1067,1202-1204`). Extracted host adapters and hello-only fd primitive do not implement or enable installed registration/activation.

## Independent checks and limits

Inspected both conftests and selected test/fixture code. Ran only the authorized isolated interpreter with explicit candidate/test path insertion and named unittest modules, from the candidate cwd:

`/private/tmp/pab-0c-venv/bin/python -I -B -c ...`

**59 tests PASS, exit 0, 6.160 seconds**: `tests.service.test_split_init` (38), `test_install_child` (2), `test_install_ownership` (14), `test_host_install` (5). `git diff --check 55d5e9f HEAD` passed. Candidate tracked/untracked status remained clean.

The tests exercise real disposable files, descriptors, locks, fsync, SQLite and canonical DB/WAL/SHM paths, with injected ownership/drop/namespace observations. Positive split tests invoke actual `install_child.prepare`; the namespace fake relocates directories while preserving inode identity. It is not a real bind mount or multi-UID proof. Negative cases cover setup stages, unsafe mountpoint, invalid/changed/duplicate fds, bind mismatch, setup failure before drop/Core, quarantine/completion, record/configuration changes, exclusion and reader/writer refusal. Entrypoint order is checked with injected OS functions. The two direct helper tests include the old symlink canonicalization witness and non-Linux refusal.

The implementer's 31 direct legacy invocations remain reported evidence, not an independently reproduced 90-test total. No retained r201/r203 legacy harness was located in the temporary locations searched. A broad filename search encountered an unrelated permission-denied entry; no claim of exhaustive temporary-file discovery is made. Relevant legacy fences and changed recovery test imports were inspected instead. The reported red/green source swap was not repeated.

No actual mount/unshare/setuid/setgid/chown, Linux privileged child, Docker, network/listener, model/ML/hardware, pytest runner/guard, full suite, maintenance kernel or HELD operation. No source edit, adoption, commit, delegation or unrelated cleanup. Only ordinary test-owned temporary fixture cleanup occurred. Sole persistent review write is this report.

## Parent handoff

The parent may adopt this exact combined source unit on this bounded GO. Linux actual qualification must still establish bind/path traversal under inaccessible Docker ancestors, actual permanent identity/capability and sidecar ownership, propagation isolation and failure cleanup/reap, role/setgid permissions and competing-process behavior. Remaining host observation/registration and packaged activation must still be implemented and reviewed under r198. This report closes R202-1's source correction; it grants no operational or release GO.
