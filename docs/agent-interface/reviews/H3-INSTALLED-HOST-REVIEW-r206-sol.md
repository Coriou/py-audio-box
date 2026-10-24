# H3 r206 Sol — independent first-round installed HOST observer review

2026-10-24. **NO-GO for source adoption of the exact r205 observer candidate: R206-1 remains blocking.** The isolated tests pass, but model-directory enumeration does not enforce its advertised entry/time bounds while collecting entries. This is a bounded source decision, not actual Linux qualification, installed acceptance, speech acceptance or release authority. Registration writing and launch/activation are explicitly outside this unit; their absence is not an additional gate.

## Candidate and authority

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-042cb51a`, branch `t3/installed-host-observer`, clean on entry and after checks. Verified the exact direct-parent chain:

- Base `1291f2e7829fa341509bf3394fc1190161525141`.
- Source `4f282794b2b81e62d16ac7bf705b19db8f9c3dd5`, whose sole parent is that base.
- Documentation HEAD `778947777ca02895fc7cd4f04f75c40f6c0d297b`, whose sole parent is that source.

The base-to-source delta is exactly five files: `service/host_install.py`, `service/install_policy.py`, `service/recovery.py`, `tests/service/test_host_install.py`, `tests/service/test_installed_host_policy.py`. Base-to-documentation HEAD additionally contains only the r205 report. Read root r198 full policy, r204 review and r205 handoff; reviewed the full source/test delta and relevant unchanged Docker, proc, mountinfo, topology, role-reader and initializer interfaces. Applied the parent's clarification: explicit CPU allocations per role, exact finite tmpfs, fixed container destinations, explicit protected host inputs and canonical configuration binding; no automatic migration. No applicable ancestor/candidate AGENTS.md was located. The bounded memory-registry search returned no relevant hit and supplied no review premise. No delegation.

## Blocking finding

**R206-1 — P2: enforce model-directory limits during enumeration, before materialization/sorting.** `service/host_install.py:603` executes `names=sorted(e.name for e in entries)` over the entire `os.scandir` iterator. Only after that completes do lines 604–605 check 4,096 entries and the deadline. A stale, malformed or concurrently enlarged protected model directory can therefore consume arbitrary directory entries, memory and sorting time while the root host launch exclusion is held. Even an already-expired deadline does not prevent this work. The eventual refusal does not make the preceding host I/O bounded. Sealed directory modes do not establish a cardinality limit, and the frozen tree hash is checked only after traversal.

Independent inert reproduction used the existing `InstalledHost` disposable fixture and its injected ownership observations. Patched only `os.scandir` for the `_model_tree` call with a context-managed iterator yielding 10,000 synthetic names, and passed `time.monotonic()-1` as the deadline. Observed: `expired deadline; entries consumed: 10000 ; result: installed_source_budget`. No large actual tree, Docker, proc qualification, privileged operation or persistent fixture was used. This demonstrates the misplaced check; it is not a timing benchmark or actual host denial-of-service execution.

Required correction: check the shared deadline before scanning and during each iterator step, enforce the remaining total entry budget before appending each name, then sort only that bounded list. Preserve the aggregate file/tree limits and final identity rechecks. Add focused injected cases for an oversized/lazy iterator and an expired deadline, asserting early termination rather than merely eventual ValueError. Re-review the corrected source pin. This finding alone determines NO-GO; no additional blocking finding was established in this round.

## Other source assessment

The closed policy binds explicit per-role nanocpus, finite tmpfs bytes/inodes, fixed input destinations, root source identities, host scope and canonical configuration digest. Worker allocation covers the enrolled thread count. Closed enrollment corroboration binds the fixed ownership policy, configuration and every authority root inode. Missing/extra projected fields refuse through the closed validators; there is no metadata-selected ownership map, legacy migration or synthetic fallback.

The observer's Docker projection excludes environment and credential contents and checks the configured user, capabilities, security flags, namespaces, groups/devices, images, restart/network/port settings and explicit RAM/swap/PID/CPU/tmpfs settings. Exact authority-volume roles/access and protected input mounts are checked before projecting into the unchanged historical topology validator. Complete all-container enumeration includes stopped identities, with unknown/missing owners refusing. The trusted future coordinator remains responsible for supplying the complete retained owner union; the API does not itself acquire registrations.

Source traversal uses literal no-follow descriptor ancestry, regular single-link file checks, fixed ownership/modes and stat identity tuples. Config bytes are hash-bound and parsed against the expected configuration. Model files are hashed and checked against enrolled manifests; extra files refuse. Credential contents are not read or hashed: `_source_stat` does open the credential read-only to fstat it, so r205's phrase “no open/read/hash” is stronger than the implementation; the actual no-content-read requirement is preserved. The seven-source mount proof requires one observed mount/device, distinct inodes and disjoint ancestry, refusing nested/intervening mounts and another mount of the same device. Unrelated ambiguous bind/volume storage still reaches the historical refusal. These conservative source checks are not proof of actual mount equivalence or OS permission behavior.

Protected authority requires existing root-owned trusted ancestry, private leaf, exact existing lock/policy modes and closed policy data. Nonblocking flock is held across callbacks, checks bracket their results, and saved callbacks refuse after context close. Exit rechecks and cleanup failures propagate. Docker uses the existing finite subprocess transport with output limits and kill/reap handling; proc/cgroup and mountinfo retain their existing bounded adapters. R206-1 concerns the newly added filesystem enumeration despite those other bounds.

The default synthetic topology path is unchanged unless an explicit installed policy/configuration pair is supplied. Installed content has its own closed shape and digest bindings and does not mint actual-host provenance. Historical capsule/provenance consumers retain their incompatible closed shapes. No original SQLite writer, Core, persistence, Dispatcher, supervisor activation or split initializer implementation changed in this five-file unit. The existing installed activation/writer fences remain.

## Independent verification and limits

Read `tests/conftest.py` and `tests/service/conftest.py` before running tests. From the candidate cwd, ran `/private/tmp/pab-0c-venv/bin/python -I -B` with explicit absolute candidate and candidate/tests paths inserted into `sys.path`, loading only:

- `tests.service.test_installed_host_policy`
- `tests.service.test_host_install`
- `tests.service.test_install_ownership`
- `tests.service.test_split_init`
- `tests.service.test_install_child`

**81 unittest tests PASS, exit 0, 9.946 seconds.** The 37 historical injected cases invoked inside host tests are nested coverage, not 37 additional unittest tests or an 81+37 total. These tests use real temporary files, descriptors, hashes, locks and SQLite with injected external identity/Docker/proc/namespace boundaries. Their green result does not cover R206-1's early-termination obligation. The separate inert reproduction above establishes that gap. `git diff --check` over the exact base-to-documentation chain passed. Candidate tracked/untracked status remained clean.

No actual Docker/proc qualification command, privileged mount/unshare/UID operation, network/listener, ML/hardware, pytest runner/guard/full suite, HELD operation, source edit, commit, delegation or unrelated cleanup. Only test-owned disposable fixture cleanup occurred. Sole persistent write: `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-INSTALLED-HOST-REVIEW-r206-sol.md`.

Parent-owned remainder: correct and independently re-review R206-1 before adopting this observer unit. Protected authority provisioning, retained-registration acquisition, authenticated dropped-UID hello, protected provenance, control-UID registration writing and explicit launch/activation remain outside this review. Actual host permissions/setgid behavior, mount equivalence, peer credentials, namespaces/cgroup association, shutdown/reap/restart and CPU Kokoro speech remain pending separately authorized qualification. This report grants none of those outcomes.
