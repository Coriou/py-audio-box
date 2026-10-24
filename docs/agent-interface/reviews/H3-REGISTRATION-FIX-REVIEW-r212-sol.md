# H3 r212 Sol — independent round 2 registration source-adoption review

2026-10-24. **GO for bounded source adoption of the pinned r211 correction and carried registration unit. R210-1 is closed; no additional blocking finding established.** Parent adoption remains parent-owned. This is no installed activation, actual Linux, speech, operational or release GO. No broad new gate is introduced.

## Candidate and chain

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/r211-identity-fix`, branch `t3/r211-identity-fix`, clean before and after verification. Verified direct chain:

- Base `ad0002aadd1d9a1fd9400afbcce25ad45ce14e95`.
- Carried source `dfd1222c0572dbbc36fb6db7be23601d842053e1`.
- Fix source `dd526fa950418336f7cbe0d802018716d89764ea`.
- Documentation HEAD `339f09465b4d7aede9441b77d7f09d63f44f704e`.

Complete-tree comparison of original `4fd3d5af7469024eba9ed1dd97293985b58a1502` and carried `dfd1222` returned no difference (exit 0). The carried delta from base contains exactly `service/host_install.py`, `service/install_child.py`, `service/supervisor_runtime.py`, and `tests/service/test_installed_registration.py`. Fix changes only host_install and its registration tests; documentation changes only r211. Read complete root normative r198, r209 handoff, r210 NO-GO and r211 correction report. Root r198 is byte-identical to candidate r198. No applicable AGENTS.md found. Bounded memory-registry search returned no relevant hit and supplied no premise. No delegation.

## R210-1 closure

Candidate line references below refer to the pinned worktree.

`service/host_install.py:972–985` now returns `(raw, identity)` from the same opened, metadata-validated descriptor. Its bounded no-follow/nonblocking read verifies size, post-read descriptor identity and no-follow path identity against the initial descriptor stat. Identity includes device/inode, mode, UID/GID, link count, size and nanosecond mtime/ctime (`service/install_policy.py:50–53`). There is no independent later path stat used to establish the retained baseline.

Enumeration saves that exact pair (`host_install.py:1064–1072`). A replacement before opening is subject to the full reader validation; the baseline is the validated enumeration descriptor, not a historical inode absent from the persisted format. Replacement during read/path corroboration refuses; replacement after reader return cannot redefine the saved descriptor identity. Both production callers of this private reader were updated: enumeration unpacks the pair, and retained validation compares the entire pair. Repository search found no remaining bytes-only production caller.

Exact retry at `1103–1104` checks identical canonical bytes and does not assign a new identity. The only new-record assignment is inside exclusive creation: after write, flush and file fsync, `1113` captures the still-open created descriptor identity. Directory fsync at `1114` cannot redefine that baseline. A replacement during creation/close/directory fsync is caught by common path corroboration against the created descriptor pair at `1116–1121`. Missing, symlinked, invalid-metadata or byte-identical replacement paths cannot become a replacement baseline.

`retained_exact()` checks every retained record before writer invocation (`1121`, writer at `1128`) and again after the DB/host/hello completion bracket (`1142`, return at `1144`). It compares against the originally captured pairs for existing records and the created descriptor pair for the new record. This closes the retry reset and separated read/stat windows identified by r210. These are bracketed continuity checks under protected authority and exclusion, not a claim of atomic immunity to arbitrary privileged mutation between every instruction.

Four new regressions at `tests/service/test_installed_registration.py:198–256` deterministically replace a file with identical bytes and assert an actual inode change: retry first hello, retry first observation, immediately after the first validated reader returns, and new-file authority-directory fsync. Each requires refusal before writer, no returned ID and unchanged registration rows. The existing writer-time identical replacement negative remains at `258` onward; exact retry positive remains at `84` onward.

## Carried transaction and regression assessment

No regression established in the carried authority, transaction or failure boundaries:

- Protected fixed authority and launch exclusion remain at `host_install.py:847–927`; callers cannot provide an observation receipt or arbitrary authority/SQL/child operation. Control exclusion at `1042` spans inspection, children/reap and completion; registration does not take the live supervisor's lane lock.
- Private WAL-aware snapshots retain full split-policy reconciliation and unhealthy-state refusal (`988–1000`). Complete DB registrations plus provenance-only interrupted attempts feed the bounded owner union (`1057–1084`); missing provenance, unknown entries and ambiguous incarnations refuse. Identity correction applies to all enumerated records, not only the selected retry.
- Closed host observation and healthy authenticated hello remain bracketed before provenance/write (`1085–1099`). Provenance is durable first; fresh source digest must equal prestate before writer (`1100–1128`). Completion checks exact expected registration rows, otherwise unchanged evidence projection, host/process observation and hello before returning (`1129–1144`). Interrupted writes retain obligations without success.
- Dropped hello and writer entrypoints retain cleared groups and permanent UID/GID 10001 transitions (`install_child.py:88–95,169–197`). The hello helper retains bounded key/socket validation, HMAC/session and server peer authentication (`supervisor_runtime.py:85–151`); existing transport peer checks remain (`worker/transport.py:84–89,107` onward).
- The writer retains fixed namespace preparation before drop, existing SQLite `mode=rw`, exact installation and ordered registration prestate checks inside `BEGIN IMMEDIATE`, narrow insert/exact-row reuse, rollback on precommit failure, close/identity checks and fsync (`install_child.py:98–166,195–197`). No root original-DB connection, Core construction, DDL or generic SQL interface was added.
- Finite communication/kill/reap failure handling and uncertain-child duplicate exclusions remain unchanged (`host_install.py:938–969,1145–1147`). Lock closure does not explicitly unlock the shared description (`install_policy.locked_file`). No kernel reap or exclusion after coordinator death is inferred.

Fix delta leaves schemas, legacy registration, protocol, authority selection and activation gates unchanged. No activation notification or packaged launch is introduced.

## Independent verification

Read both conftests before execution. Used only `/private/tmp/pab-0c-venv/bin/python -I -B`, explicit absolute candidate and candidate/tests import paths, and isolated unittest loading. No pytest runner, conftest fixture or maintenance guard was invoked.

**Exact candidate green: 116 tests PASS, exit 0, 19.521 seconds**, across exactly `tests.service.test_installed_registration`, `tests.service.test_installed_host_policy`, `tests.service.test_host_install`, `tests.service.test_install_ownership`, `tests.service.test_split_init`, and `tests.service.test_install_child`.

**Independent red: four regressions FAIL as expected on carried source, 4 failures / 0 errors, 1.576 seconds.** Each failed because the expected refusal was absent. Loaded carried host_install source directly from `git show dfd1222:service/host_install.py` into the imported module in an isolated Python process; ran the four final candidate regression methods. Expected-red wrapper exited 0 after verifying exactly those counts. No source file was edited or restored, and no persistent repro file was created. Together with candidate green, this independently corroborates the reported red/green distinction.

The focused suite also covers exact retry, real closed observer with injected external observations, orphan-owner union, interruption/rollback/committed retention, stale writer prestate, wrong enrollment/installation, control exclusion, private-copy parent connections, sidecar ownership observation seam, postwriter replacement and finite child failure retention. `git diff --check ad0002a HEAD` passed. Candidate HEAD and clean status were rechecked after tests.

## Evidence limits and handoff

Tests exercise real temporary files, descriptor identities, SQLite/private copies, transactions and locks with injected UID/GID, namespace and host observations. Narrow identity regressions inject provenance content validation; the separate closed-observer positive does not turn these into actual host qualification.

No actual mount/unshare, UID/setuid/setgid/chown, Docker, Linux proc qualification, network/listener, ML/model/hardware, pytest guard/full suite, maintenance/apply or HELD operation was performed. Actual OS permissions/capabilities, peer credentials, mount traversal/isolation, competing actual processes and kernel reap remain unqualified. Only normal test-owned fixture cleanup occurred. No source edits, commits, adoption, delegation or unrelated cleanup. Sole persistent write is this root r212 report.

Parent may adopt this bounded source unit. Existing actual Linux ownership/peer/mount qualification and Lot B packaged waiting-control launch/notification/activation remain pending under r198; their scope is unchanged.
