# H3 r209 Sol — installed registration transaction implementation

2026-10-24. **Bounded registration source unit implemented; independent review/adoption remains parent-owned.** This candidate returns a retained registration ID only. No activation notification, installed startup gate, launch implementation, operational qualification or release acceptance is supplied.

## Exact candidate and authority

Own checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-30d18996`, branch `t3/installed-registration-writer`. Entry was clean at `b4b83b3`; fast-forwarded without conflict to current root `ad0002aadd1d9a1fd9400afbcce25ad45ce14e95`. Verified ancestry includes r207 report `d0fe258fb5bb40b5909bd7bab2d39b49bc7d3b48` and initializer correction `0b02b138de239753b831698fb6987cd8c83c23cf`. Root remained at the same base when preparing this handoff.

Source commit: **`4fd3d5af7469024eba9ed1dd97293985b58a1502`**, direct child of the base. Exactly four source/test files changed: `service/host_install.py`, `service/install_child.py`, `service/supervisor_runtime.py`, `tests/service/test_installed_registration.py`. This report is a separate documentation commit. Only this report is copied to `/Users/ben/Projects/py-audio-box`; no root source adoption or root commit is performed. Existing root untracked STT evidence is untouched.

Read the complete normative r198 design, including all six registration/launch steps and restart/failure boundaries, and r204/r208 adoption reviews. Confirmed the root r198 file is byte-identical to the candidate copy. Inspected the integrated host observer, fixed policy, private readers/snapshot, initializer/mount adapter, persistence and authenticated runtime interfaces. No applicable ancestor/worktree/service/test AGENTS.md was found. Bounded memory-registry searches returned no relevant premise. No delegation or self-adoption.

## Implemented path

`installed_launch_authority(...).register(owners)` is the installed entrypoint. The compatible operator adapter exposes the explicit `installed-register` command with a closed request containing only `configuration`, `scope`, and `owners`. Configuration/host policy and source roots still derive from the protected fixed authority directory; the caller cannot submit an observation or an asserted actual-host receipt. Historical `register` and synthetic paths retain their implementations.

The protected launch exclusion encloses the complete transaction. Registration takes only the existing control lock, leaving the live supervisor's lane lock intact. Parent SQLite inspection always uses bounded WAL-aware private copies and the existing full split-policy snapshot/reconciliation validator. It corroborates retained enrollment, configuration and role identities, refuses faults, incoherent allocations, nonterminal journal records, exhausted hello allocation budget and recovery obligations, and derives the complete owner union from **all DB registrations plus all protected provenance-only interrupted attempts**. Unknown authority entries, missing provenance, ambiguous incarnations and finite row/byte/owner budget overflow refuse.

The existing closed installed observer performs host/source/process enumeration; it is not replaced by a permissive observer. Two pre-insert host observations bracket authenticated hello helpers, and a fresh private snapshot immediately before insertion must equal the original source digest. The chosen live pair must contain exactly one control and one worker. Hello must retain its lane/boot, healthy state and launch count throughout the bracket. Existing socket server/client peer identities remain unchanged.

A fresh hello exec inherits only the validated IPC directory fd and a bounded public fd request. It permanently drops all real/effective/saved UID/GID slots to 10001, clears supplementary groups and applies umask 077, then invokes the existing hello-only HMAC/session and peer-authentication primitive. It returns only the bounded public hello. The `/proc/self/fd/<fd>/lifecycle.sock` spelling belongs to this helper's own proc view; AF_UNIX connect uses the held directory without SQLite-style canonicalization through inaccessible Docker ancestors. Key and socket metadata/path-to-fd checks remain. This is source reasoning and injected coverage, not actual proc/permission qualification.

Protected provenance is retained **before** the DB write, under a distinct installed record kind and fixed per-install authority directory. It binds the registration, complete closed installed observation, enrollment/configuration/host policy digests, IPC inode identity and authenticated hello. Creation is exclusive, mode/UID/GID/link checked and fsynced. Existing records are reusable only with identical canonical bytes; all retained provenance bytes and captured file identities are rechecked before insertion and at completion. No replacement/repair/reminting occurs. Historical B2 provenance/capsule readers are unchanged.

The writer exec inherits exactly control/staging directory fds and the parent's held control lock. Its closed public request has installation, registration and exact registration-row prestate, never SQL, arbitrary actions, seed or keys. Before permanent identity drop it reuses `install_child.prepare` and the reviewed **same fixed namespace paths** `/mnt/pab-install/control` and `/mnt/pab-install/staging`; SQLite never receives a proc-fd alias. No new general mount/path adapter is introduced.

After drop, the single-purpose writer verifies control directory/lock/DB/sidecar identity, opens the **existing** DB with SQLite `mode=rw`, sets trusted-schema off and synchronous FULL, and starts `BEGIN IMMEDIATE`. It checks the exact installation row/digest and exact ordered registration prestate, inserts only the canonical validated registration or accepts an identical existing row, then commits and closes. It creates no schema, Core or other original DB writer. Parent never connects to original SQLite. Original writes and SQLite-created sidecars occur only in this control-identity helper; post-close sidecar/root checks and directory fsync precede child success.

After reap, parent checks a fresh complete private snapshot, exact expected DB row set, otherwise unchanged evidence projection, closed host/process observation, identical healthy hello and all provenance bytes/identities. Only then does it return the registration ID. Pre-insert, post-insert and post-bracket interruptions retain their provenance/row obligations and cannot return success. Worker boot or control process changes during the transaction refuse. Existing ambiguous changed container incarnations refuse rather than erase historical obligations.

Each exec has a 10-second communication wait followed, on failure, by kill and a 2-second reap wait. A reaped timeout still fails. An uncertain reap retains the child and duplicate control/launch exclusion descriptors for the coordinator process lifetime; no retry/adoption API or success is exposed. Normal completion explicitly closes the communication pipe before releasing parent exclusions. Exceptional pipe-close errors annotate the original failure. The inherited control-lock descriptor independently remains with a live writer. This is finite source coordination; it does not claim actual kernel cleanup or indefinite exclusion after coordinator death.

## Verification

Both conftests were inspected before execution. Only isolated unittest loading was used, through `/private/tmp/pab-0c-venv/bin/python -I -B`, with explicit candidate and candidate/tests paths inserted into `sys.path`. No pytest runner or guard was invoked.

Focused suite: **112 tests PASS, exit 0, 18.476 seconds** across:

- `tests.service.test_installed_registration` (27 new tests).
- `tests.service.test_installed_host_policy`.
- `tests.service.test_host_install`.
- `tests.service.test_install_ownership`.
- `tests.service.test_split_init`.
- `tests.service.test_install_child`.

The 112-test run covers the final functional changes. A subsequent error-preservation-only adjustment to communication-pipe cleanup was followed by **2 ChildTransport tests PASS, exit 0, 0.006 seconds** at the exact source commit. `git diff --check` passed; source checkout was clean after commit.

Reproduction from this checkout:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/t3-30d18996","/Users/ben/.txcode/worktrees/py-audio-box/t3-30d18996/tests"];s=unittest.defaultTestLoader.loadTestsFromNames(["tests.service.test_installed_registration","tests.service.test_installed_host_policy","tests.service.test_host_install","tests.service.test_install_ownership","tests.service.test_split_init","tests.service.test_install_child"]);r=unittest.TextTestRunner().run(s);sys.exit(not r.wasSuccessful())'
```

New tests exercise real temporary SQLite, transactions, fd/lock exclusion, private copies, canonical ordinary DB paths, file identities and fsync. Writer tests invoke the real namespace adapter with injected namespace operations preserving inode identity, and injected UID observations. One complete positive/retry runs the **real closed installed observer and provenance validator** with injected Docker/proc observations and real sealed input files. Narrow transaction tests separately inject observer content to isolate interruption/union behavior; those are not independent host-observer evidence.

Coverage includes exact retry, provenance-only and committed interruption obligations, preinsert/inserted/precommit rollback and committed failure, complete orphan-owner union, stale writer prestate, wrong installation/enrollment/candidate, missing original DB, worker boot/control process changes, live Core lock exclusion, canonical-path writer-only original connections, control-owned WAL/SHM observation seam, provenance tampering and identical-byte inode replacement, unknown authority entries, fixed-entrypoint prepare-before-drop order, and finite timeout/kill/reap failure retention. The inherited focused suites cover wrong scope/configuration, host policy/resources/mounts, source tampering, enrollment/init and historical synthetic adapters. No claim is made that every negative was newly or independently reproduced outside those suites.

## Evidence limits and next gate

No actual mount/unshare, setuid/setgid/chown, Docker invocation, Linux proc qualification, socket listener/network, model/ML/hardware, pytest guard/full suite, maintenance/apply or HELD operation was performed. UID/GID and sidecar-owner observations are injected; real OS permission, capability clearing, root-only Docker ancestor traversal, peer credentials, mount isolation, competing processes and uncertain kernel reap still require separately authorized Linux qualification. Only ordinary test-owned fixture cleanup occurred. No broad source or unrelated cleanup.

This source unit deliberately emits no activation file/notification and changes no Core, Dispatcher, worker protocol, installed B2 writer/apply or activation gate. Persistence/DDL needed no change: the validated registration remains in the existing JSON row format with its closed legacy-compatible parser. Independent source review/adoption is next; packaged waiting-control launch, notification and explicit activation remain **Lot B** under r198. No installed, speech, release or operational GO is claimed.
