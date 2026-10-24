# H3 split initializer path correction r203 Sol

2026-10-24. **R202-1 source correction implemented; independent review next. No self-adoption or Linux permission proof.** Host registration and installed activation remain CLOSED.

## Provenance

Fresh own worktree `/Users/ben/.txcode/worktrees/py-audio-box/r203-path-fix`, branch `impl/r203-path-fix`, base `55d5e9f1675b5bb27dc786dbbe54c17e4d996311`. Cherry-picked exactly these source commits, in order:

- r199 `eaccc205b70faa84bb9aeca80ebb6ae2e3109b69` → `f71671fe34077883a879ecb0b5d5cc70239c0b8a`; stable patch ID `3a980787dcafde40570419188a01f2719b70a9d6`.
- r200 `45d69668b478f5c9f509f1f3d231420c4d7ddf2e` → `f8b5682828d96ba563a3fe68842a7e56d5ac6a10`; stable patch ID `64d6dd9c1d5bded832ac059385aa1fa1a5ccc599`.
- r201 `e4f8414833de90feab885d1b32bc980c0320c9d5` → `3de61af50ee7098940dc2db881c15f117bd828d0`; stable patch ID `b5a729470300b4ff4b05767f8b4404d97c17d726`.

Source/tests correction commit **`0b02b138de239753b831698fb6987cd8c83c23cf`** changes only `service/persistent_init.py`, new `service/install_child.py`, `tests/service/test_split_init.py`, and new `tests/service/test_install_child.py`. No recovery/reconciliation/record change. No historical checkout edited. Attached entry checkout was `6ce2cbf`; root observed at `b9d894f060b618590e1176ef86c1deab913c5840` before report copy, with pre-existing unrelated untracked STT evidence left alone. Sole root write is an identical copy of this report; no root source adoption/commit/reset.

Read normative r198 in full, r201 handoff and root `H3-SPLIT-INIT-REVIEW-r202-sol.md`. No applicable AGENTS.md found. Bounded memory lookup returned no matching premise. No delegation.

## Selected design and privilege boundary

SQLite's ordinary Unix VFS canonicalizes symlink components. A `/proc/self/fd/N/core/core.sqlite` spelling therefore cannot preserve access through inaccessible Docker ancestors. The fix supplies ordinary canonical paths within a private mount namespace, retaining the ordinary SQLite VFS and original writer UID 10001.

The existing isolated fresh Python helper now reads/validates the bounded closed public request before its permanent UID drop. New small `install_child.prepare` corroborates both inherited fds against the public fixed role policy and device/inode identities, rejects duplicate/invalid descriptors and aliases, and refuses an absent, symlink, non-directory, non-root-owned or group/world-writable `/mnt`. It then:

1. Unshares only `CLONE_NEWNS`.
2. Makes `/` recursively `MS_PRIVATE` before any new mount.
3. Mounts a private nosuid/nodev tmpfs at `/mnt`, with fixed `mode=0755,size=64k,nr_inodes=16`.
4. Creates `/mnt/pab-install/control` and `/mnt/pab-install/staging` in that tmpfs and non-recursively binds exactly `/proc/self/fd/<validated control fd>` and `/proc/self/fd/<validated staging fd>` there. Each resulting directory's device/inode and type are checked.
5. Clears supplementary groups, permanently sets all GID and UID slots to 10001, sets umask 077 and verifies identity before calling the unchanged `_split_core` at those fixed namespace paths.

No request value supplies a mount target, flags, filesystem, mount data or arbitrary source path. Only validated integer inherited fds form the bind sources. No receipts fd is inherited. The same synchronous launcher still retains parent exclusions and checks exit/reap; any setup/drop/Core failure exits the helper without publishing enrollment. There is no helper subprocess tree or namespace keeper; mounts are tied to the sole helper's namespace lifetime, including partial setup failure. No host mkdir, global mount, host ancestor chmod, root Core, post-DB chown, custom VFS or permission bypass is added. Existing create-only quarantine/completion and activation fences are preserved.

This deliberately does not chroot: existing interpreter, SQLite extension, dependent libraries and trusted source remain at their original paths. Mounting `/mnt` is confined to the child namespace. The supported host must provide safe `/mnt`, Linux libc `unshare`/`mount`, namespace/mount authority (normally trusted root with CAP_SYS_ADMIN), tmpfs and `/proc/self/fd`; missing support refuses. The helper uses only Python stdlib ctypes and existing dependencies. These are source prerequisites, not measured host capabilities.

## Verification

Inspected both conftests before execution. Used only `/private/tmp/pab-0c-venv/bin/python -I -B`, explicit checkout/test paths and unittest; no pytest runner, guard/autouse fixture execution or full suite.

**Final focused run: 59 tests PASS, exit 0, 6.802 seconds**: 38 split-init cases, 2 direct helper/canonicalization cases, 14 retained ownership cases and 5 retained host-adapter cases. This includes the existing 51 split/ownership/host cases.

All split-init injected child positives now call the actual `install_child.prepare` path constructor and validators. The injected OS adapter represents the namespace root by a test-owned temporary prefix and emulates bind roots by relocating real fixture directories, preserving device/inode identity; it restores them in normal cleanup. It never uses a symlink as a fake successful bind. Real Core, SQLite, locks, descriptors, fsync and ordinary filesystem behavior run against the constructed namespace paths. The direct SQLite test verifies `PRAGMA database_list` equals the constructed namespace DB pathname and observes real DB/WAL/SHM files there. A separate real symlink probe demonstrates the old fd-style canonicalization back to the backing pathname.

New negatives cover namespace/private-propagation/tmpfs/each-bind failure, changed/duplicate/invalid fds, unsafe mountpoint, bind identity mismatch, non-Linux refusal, and helper setup failure before UID drop/Core. An injected entrypoint test verifies setup precedes cleared groups, permanent GID/UID transition and umask, and that Core receives the namespace paths.

**Red/green:** temporarily loaded the carried `persistent_init.py` in this own worktree under the fully injected boundary test, restoring the corrected source in `finally`. The new entrypoint regression failed as expected (exit 1): carried code passed `/proc/self/fd/3` and `/proc/self/fd/4` to Core instead of prepared namespace paths. Corrected implementation passes. No privileged operation occurred in this red run. Initial development run had two fixture failures (namespace fixture reuse and old path-based connection classification); both were corrected before the passing runs.

**Legacy: 31 direct unittest FunctionTestCase invocations PASS, exit 0, 2.100 seconds.** Fresh ordinary TemporaryDirectory fixtures call the inspected `installation.__wrapped__` directly: synthetic creation/inspection; immutable plan/verify; schema recognition; injected actual-host activation gate and Docker projection; missing journal; oversized record; closed activation; unknown inventory; seven explicit completion boundaries; five old-data preservation cases; seven monitor/journal/registration/scope/consumer/key/volume mutations; both locks; missing-key no-remint. No maintenance child, kernel, HELD action or wait operation was invoked. `git diff --check` passed.

## Evidence limits and handoff

No actual mount/unshare/setuid/setgid/chown operation was executed. The inherited source fixtures still inject ownership observations and fchown, and emulate special mode bits on macOS. Namespace relocation tests establish the real SQLite pathname semantics of the prepared layout; they cannot establish Linux bind-mount behavior, root-only ancestor traversal, actual UID-owned sidecars, capability dropping, kernel mount cleanup, setgid inheritance, peer credentials or competing real processes. Those remain explicitly authorized actual Linux qualification work for the parent. No Docker/network/listener/model/ML/hardware/production execution. Test fixtures received only normal own cleanup.

Independent combined review is next. No Linux permission GO, installed acceptance, registration implementation or activation authority is claimed by this report.
