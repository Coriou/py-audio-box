# H3 r205 Sol — bounded installed HOST observer implementation

2026-10-24. Implemented source unit; independent review/adoption remains parent-owned. Installed registration writing, launch/activation and actual Linux qualification remain CLOSED. No self-adoption or operational GO.

## Pins and scope

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-042cb51a`, branch `t3/installed-host-observer`. Entry HEAD `caf5229cefc890ad259bbb136c9aed765d24ec07`, clean tracked/untracked state. Fast-forwarded to current root `1291f2e7829fa341509bf3394fc1190161525141` and verified clean before edits. Integrated split initializer correction `0b02b138de239753b831698fb6987cd8c83c23cf` is an ancestor. Read full r198 normative design and r204 combined review, initializer callback and actual host/operator/recovery/installation/context interfaces. No applicable repository/ancestor AGENTS.md found; bounded memory registry lookup found no relevant premise.

Source commit: `4f282794b2b81e62d16ac7bf705b19db8f9c3dd5`. Only source/test changes: `service/host_install.py`, `service/install_policy.py`, `service/recovery.py`, `tests/service/test_host_install.py`, new `tests/service/test_installed_host_policy.py`. Historical operator CLI needs no change. This report is committed separately and copied as the sole root write; source is not integrated into root by this task.

## Closed protected authority and entrypoints

Parent clarified the missing policy dependency during implementation: explicit operator-supplied CPU per role, exact finite tmpfs per role, fixed container destinations, private explicit host input paths, and binding to the canonical enrolled configuration digest. No one-CPU default or invented pre-existing record.

New fixed host policy kind `installed-local-docker-v1`, version integer 1, has exactly:

- `version kind configuration_digest host roots inputs cpu tmpfs control_port`.
- `host`: exact existing daemon/context/four named-volume creation identities.
- `roots`: exactly control/staging/receipts/ipc; each has `source device inode`.
- `inputs`: exactly models/config/credential. Each has `source device inode mode uid gid size mtime_ns ctime_ns`; models/config additionally have `sha256`. Credential hashes or extra credential fields refuse.
- `cpu`: exactly control/worker, integer nanocpus in 1..1,024,000,000,000. Worker allocation must be at least the maximum enrolled execution-thread count times 1,000,000,000. This is an allocation constraint, not a measured performance claim.
- `tmpfs`: exactly control/worker, each with positive integer `bytes inodes`; bytes at most the enrolled RAM budget, inodes at most 65,536. Each container has exactly `/tmp` with canonical `rw,noexec,nosuid,nodev,size=<bytes>,nr_inodes=<inodes>,mode=1777` options. Tmpfs uses that container's memory limit; no separate unbounded allocation is inferred.
- `control_port`: explicit integer 1..65,535, publishing only container `8000/tcp` on `127.0.0.1`.

`installed_launch_authority(config, scope)` selects only the fixed authority base plus `installed/<digest(scope)>`. It requires Linux and real/effective root, root-owned non-writable no-follow ancestors and a private 0700 leaf. Existing root:root 0600 single-link `launch.lock` is acquired with nonblocking exclusive flock. Existing root:root 0600 `host-policy.json` is read bounded/no-follow, checked against the expected closed config and scope, and frozen by file identity and bytes. No alternate policy filename, arbitrary UID map, creation/repair/migration, or fallback authority exists. Missing authority/policy/lock refuses; provisioning these protected inputs is a later parent-owned task. The context rechecks authority path, lock and policy around callbacks and on exit; retained callbacks refuse after close. Cleanup/check failures propagate rather than grant success.

Use the held object's `initialize_authority` as `initialize_split(..., authority=held.initialize_authority, policy=install_policy.SPLIT)`. It freshly verifies Docker/local filesystem/inputs and the complete empty-consumer inventory, returning only the initializer's existing closed `host roots consumers` shape. This is now tested through actual `initialize_split`, with its synchronous child boundary injected. Root inode role transitions and completion remain the existing initializer's responsibility.

For observation after enrollment, `held.observe_owners(owners, enrollment)` requires the complete expected owner union supplied by the trusted coordinator. Its closed retained enrollment must match the exact configuration, protected config digest, host scope, fixed policy and every root inode. Inventory compares every supplied historical/current owner to every actual consumer, including stopped containers; missing/unknown/changed containers refuse. The future registrar must derive that expected union from **all** retained registrations using the existing union validator, rather than a caller-selected subset. This unit neither writes registration rows nor introduces an alternate retained-record store.

## Observer behavior and narrow storage proof

The production adapter uses the existing bounded read-only Docker subprocess transport (fixed operations, no shell, bounded output/deadline, kill/reap on failures), existing bounded Linux proc/cgroup-v2 identity adapter, and bounded mountinfo reader. New inspect projection deliberately excludes Config.Env, arbitrary command output and credential bytes. It observes exact configured users, cap drop/add, security options, privileged flag, groups/devices/device rules/requests, PID/cgroup/IPC/UTS/user namespaces, image IDs, read-only rootfs, network/restart/ports, RAM/swap/PID/CPU controls, tmpfs and actual mounts.

Control is exactly 10001:10001, drop ALL/no added capabilities. Worker is exactly 0:10001, drop ALL and exactly SETUID/SETGID/KILL. Both require no-new-privileges, private cgroup and IPC namespaces, ordinary private PID/UTS and no user remap, no extra groups/devices, restart no, read-only image and exact pinned image. Worker network is none; control is bridge with only the exact loopback port binding. CPU nanocpus, RAM and PID settings must equal protected/enrolled allocations; alternate quota/cpuset settings and unlimited swap refuse. Owned roots are additionally checked against fixed split role modes when observing owners.

The authority volume roles/destinations/access remain exact. Extra allowed mounts are only the identical sealed `/models` tree RO in both, exact nonsecret config RO at `/etc/speech-box/install.json` in both, and exact root:10001 0640 credential RO at `/etc/speech-box/control.key` in control alone. Config is root:10001 0440. These strings are container destinations, not global host-source paths. Sources come only from protected per-install authority. Credential bytes are never read, hashed, returned or logged; its frozen inode/mode/size/timestamps are corroborated. All input leaf and ancestor traversal is literal/no-follow; files must be regular and single-link. Config bytes are hash-bound and parsed equal to the expected canonical closed configuration.

Models are a bounded sealed 10001:10001 tree, directories 0555/files 0444, no symlink/hardlink files. Every file is hashed, every enrolled manifest is checked against its enrolled hash/checkpoint/revision and exact declared file hashes/sizes, and extra files refuse. The protected model-tree SHA is the canonical digest of the sorted directory/file inventory, including full stat identity tuples and file hashes/sizes. It is intentionally host/inode-specific. Limits: 4,096 entries, 512 MiB per file, 2 GiB aggregate and the collection deadline. Descriptor reads and final tree rechecks detect exchanges/mutation. No download, inference or provider call occurs.

Narrow mount proof requires all seven source roots on the same observed ordinary host filesystem mount, matching stat device and distinct inodes, disjoint canonical ancestry, no source/intervening/nested mount, and no second mount of that device anywhere in observed mountinfo. Different devices or realpath alone never prove disjointness. This deliberately conservative rule can refuse otherwise legitimate hosts. Only exact protected external inputs receive this narrow exception; unrelated bind/volume storage still reaches the unchanged historical `_mount_touches` refusal. No general resolver, alias waiver, implicit pull/remove/prune or Docker mutation exists.

Inventory brackets list-all container IDs, exact daemon/volume identity, mountinfo and complete input identities/hashes. Owner observation additionally brackets process start ticks, private PID namespace, recognized cgroup-v2 association and IPC identity through the existing production proc adapter. Returned closed content includes configuration, host-policy and enrollment digests, topology and process records. `validate_installed_observation` validates those bindings and process coverage; it does not authenticate caller JSON or assert actual qualification. No `actual-host-inspection` label is manufactured by this branch. Protected retention/hello/registration transaction remains the next unit.

## Compatibility and tests

Historical `Docker.observe`, synthetic `host_inventory`, topology/provenance/union/capsule shapes, host recovery/freeze and compatible operator CLI remain reachable unchanged. `recovery.validate_host_topology` selects the installed content adapter only with explicit protected policy plus expected config arguments; its default is the original closed synthetic validator. Old B2 capsule/provenance consumers do not accept installed observation shapes. No split-init, persistence, Core, Dispatcher, worker, startup activation or original SQLite writer edit.

Read both conftests before testing. Ran only the authorized isolated interpreter, from the candidate directory, with explicit source/test path insertion:

```
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=[".","tests"];s=unittest.defaultTestLoader.loadTestsFromNames(["tests.service.test_installed_host_policy","tests.service.test_host_install","tests.service.test_install_ownership","tests.service.test_split_init","tests.service.test_install_child"]);r=unittest.TextTestRunner().run(s);sys.exit(not r.wasSuccessful())'
```

Final result **81 tests PASS, exit 0, 9.520 seconds**: installed observer 20, host install 7, ownership 14, split init 38, install child 2. Two host unittest cases directly invoke 37 named historical injected cases: capsule positive plus 18 negative variants, seven bounded Docker transport cases, eleven fake-proc tree cases. These are nested cases, not 37 additional unittest tests. The guard/pytest runner was not invoked. `git diff --check` passed.

New tests use actual disposable files, descriptors, modes, hashes, locks, SQLite and the real initializer with injected Linux/UID/Docker/proc/mount namespace boundaries. Positive observations cannot be presented as actual host qualification. Negative coverage includes wrong users/caps/groups/devices/namespaces/remap/resources/network/images/config, protected-policy unknown/missing fields, finite CPU/thread and tmpfs constraints, changed input bytes/identity, secret metadata change/no secret read, symlink ancestor/leaf, hardlinks/mode/overlap, mount alias/nested mounts/device mismatch, unknown/missing owners, Desktop/rootless/remote/daemon mismatch, container-list/process races, protected policy replacement, non-root/wrong lock mode, competing lock acquisition and callback use after exclusion close. Historical tests retain capsule/provenance refusal and sanitized bounded subprocess behavior.

During test construction, assertions exposed a canonical temporary path mismatch in the direct legacy harness and an incorrect expectation that initialize_split returned scope. Both were test-harness mistakes corrected before the final green run. No production fallback/waiver was introduced.

No actual Docker/proc qualification command, privileged mount/unshare/chown/setuid, network/listener, ML/hardware, pytest runner/guard/full suite, maintenance kernel, HELD operation or delegation. Only ordinary fixture cleanup. No source adoption into root.

## Parent-owned remainder

Independent review of this exact source pin; protected authority provisioning/packaging; complete retained-registration input acquisition; authenticated dropped-UID hello; protected provenance retention; UID 10001 original-DB registration insertion under exclusion; post-insert brackets; finite waiting-container launch/activation notification and installed startup gate. Existing synthetic writer/activation fences remain. Actual Linux admission, setgid inheritance, peer credentials, namespaces/cgroup association, mount equivalence, shutdown/reap, restart behavior and CPU Kokoro operation still require separately authorized qualification. This report grants none of those outcomes.
