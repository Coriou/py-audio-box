# r357 Sol — container-managed Kokoro CPU installation candidate

2026-10-25. Implemented the r356 single-lane SOURCE candidate. **Fresh independent
Sol review and parent adoption remain required; this report does not self-adopt
or grant SOURCE/release GO.** Actual Linux, THIS Intel Mac Docker Desktop and
THIS Intel Mac OrbStack qualification are separate **UNRUN** rows. Metal
UNQUALIFIED; release NO-GO; legacy CLOSED. Accepted native freshness/source epochs
and renewal remain unchanged.

## Authority and exact scope

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-b6d61621`, branch
`t3/astra-container-install-client`, initially clean, base HEAD
`45ce583b1f5c61e14c0218911a76ea68a6e928d9`. Read FULL normative r356 before design
or code, r351 (including commission 2), r354/r355, PRODUCT, CPU-DOCKER,
HEADLESS-INSTALL and current installation/activation/packaging seams. The normative
r356 file SHA256 is
`a59710f5ac746974730c91f9144f6bb1a990083b1a6be312ce7daf3109e960da`.
Bounded repository discovery and explicit ancestor checks found no AGENTS.md.
A memory registry search returned no relevant technical premise. No delegation,
root-checkout file writes, merge, push, self-adoption or operational cleanup.

Only permitted source work and tiny temporary filesystem/SQLite/ASGI/fake-Docker
fixtures were performed. No Docker command, build, pull, load, setup, daemon probe,
network/listener, actual application child, install/download, ML, hardware, host
crash, HELD or STT operation was executed. Direct Python test processes and
in-process Dispatcher threads are not installed child/kernel evidence. No heavy
operation was attempted; the supplied below-6-GiB gate was not measured or waived.

## Delivered source

The deterministic distribution now dispatches `speech-box docker install/up/status/stop`
to a stdlib-only Mac/Linux client. It resolves an explicitly named local Unix
context, freezes endpoint and daemon ID, strips ambient Docker overrides, checks
rootful/no-remapping Linux/amd64/cgroup-v2 metadata, and uses only the Docker CLI.
The helper runs in the pinned control image with real installed finite entrypoints.
No Mac root, VM SSH, client Linux wheelhouse or operator host-path/stat records are
part of this new surface. Inputs and immutable images must already be supplied;
there is no acquisition, pull, model substitution or qualification minting.

`daemon_volumes` defines the distinct `installed-daemon-volumes-v1` launch binding
and `daemon-volumes-inspection-v1` observation variant. Enrollment retains the
existing fixed Kokoro split ownership and carries the explicit new launch binding.
Trusted startup selection is passed by the packaged `daemon_main`; disk metadata
and public requests cannot select it. The new projection has Docker inspections,
container-local facts and authenticated hello, rather than host process fields.
State.Pid is never locally dereferenced or equated with SO_PEERCRED PID. Current
local namespace/start facts are bound to exact Docker StartedAt and authenticated
boot; namespace inode uniqueness across historical restarts is not asserted.

Eight unpredictable installation-scoped ordinary local volumes replace runtime
host binds. The separate authority root remains root:root 0700, its records and
launch lock 0600; neither runtime mounts it. Runtime mount/UID/security/resource
vectors retain narrow control/worker roles, private namespaces, read-only rootfs,
no-new-privileges, no wrapper init, restart=no and pull=never. Both requested mounts
(including VolumeOptions.NoCopy) and actual mounts are inspected. Custom drivers,
options, bind aliases, extra mounts and unexplained consumers refuse. Daemon
Mountpoints are opaque client-side bindings, not paths the Mac must stat.

Install imports and Core initialization are separate helper phases. The Core/writer
helper never mounts models or credentials. Its fixed-view child validates inherited
control/staging identities, closes other descriptors, permanently drops real,
effective and saved UID/GID to 10001, clears supplementary groups and verifies zero
inheritable/permitted/effective/ambient capabilities under no-new-privileges before
calling the existing `_split_core` or registration writer. It has no unshare,
mount, chroot or SYS_ADMIN operation. Parent exclusions remain held through the
synchronous child result; unknown completion remains pending and cannot enroll.
There is no copied Core schema, second database or scheduler.

Install/init helpers have exactly CHOWN/FOWNER/DAC_OVERRIDE/SETUID/SETGID/KILL.
Later helpers drop CHOWN/FOWNER and retain DAC_OVERRIDE/SETUID/SETGID/KILL: DAC
bypass is needed for finite locked reads of UID10001-private Core roots/files.
This is confined to trusted narrow-volume helpers. Runtime workers retain only
SETUID/SETGID/KILL; control drops all capabilities. Actual enforcement is UNRUN.

The helper holds authority-directory and protected launch-file exclusions through
bounded nonce/phase/sequence exchanges, client Docker actions, registration,
notification and final checks. Client handles retain intent before volume/helper
creation; protected helper authority retains runtime creation intent and exact IDs
before start. Partial state is never replaced or guessed from names/labels.
Current and bounded recorded terminal helper consumers are explicitly inspected
with exact ID/image/mounts/entrypoint/StartedAt/exited state. Unknown or failed
helpers refuse; at most 128 transactions and bounded provenance are admitted.

Input streaming copies only enumerated regular single-link files, with relative
closed paths, byte/count/time limits and exact public hashes. There is no archive
extractor. Existing Linux model-tree validation independently checks and seals
inputs. Credential bytes travel only in private attached stdin and their sealed
file; there is no credential hash in manifest/handle/provenance, argv/env token,
or secret-bearing helper stdout/error. The client avoids Docker-info invocations
for each input chunk; exact phase snapshots bracket the stream.

Launch order is control waiting without active Core/public listener, final worker
PID1, authenticated startup/hello and exact view/image/security brackets, dropped
writer under Core exclusion, durable protected provenance, notification of the
same control ID/current local incarnation, owner activation and final bracket.
Normal stop sends TERM to control, checks owner exit and existing terminal/reap/ACK/
publication history, then TERM/waits worker and verifies stopped history. Clean up
reuses the same IDs and storage. No Docker stop KILL escalation, force, rm/prune,
rollback, reset or repair path exists. New-lane partial phases remain retained and
refuse; the existing strict host exact-completion command remains unchanged.

Matching composition reference: `service/compose.daemon-volumes.yml`; Compose
up/down remains unsupported. `DOCKER-VOLUMES.md` documents the reviewed request,
preprovided inputs, few-command journey and concrete public preset/generation/read
schemas. Existing CPU/host instructions point to the distinct new lane.

## Evidence-consumer audit

- `install_policy`: closed explicit new enrollment binding; fixed execution identity
  unchanged. `host_enrollment` explicitly refuses daemon-volume enrollment.
- `persistent_init`: one `initialize_split`, existing ordered exclusions, seed,
  creation intents, completion digest and ambiguous-Core quarantine; explicit new
  binding and fixed-container child adapter only. Strict host launcher is retained.
- `install_child.registration_writer`: exact new-vs-host evidence binding derives
  from validated explicit enrollment; fixed container writer is separate from the
  unchanged host mount-namespace entrypoint.
- `host_install`: host observer/proc observation, provenance/history/handoff and
  `actual-host-inspection` validators remain strict. New authority adapter reuses
  private snapshot, existing history checks and dropped writer, not host fiction.
- `recovery`: shared closed registration parser admits the new variant; legacy
  generic writer rejects it. Snapshot/detached/startup readers bind evidence to
  enrollment and explicit trusted startup selection. Host union/observation gates
  still require their original evidence. No terminal-reap/publication fence is
  replaced by Docker health or a local observation.
- `entrypoint`: closed new notification variant and explicit trusted daemon factory;
  pre-owner and post-Core checks revalidate registration/configuration/boot/current
  local incarnation. The old projection retains its original 16-KiB budget and
  required host fields; new private gates have a separate 64-KiB ceiling.
- `package_install` and distribution: real client dispatch, exact allowlist entries,
  existing image COPY allowlist admits the new Python modules; deterministic archive
  entries/bytes are checked in process. Public HTTP/CLI/MCP trusted-config schemas
  are unchanged. Native/Metal/CUDA/STT/UI routing is not added.

## Finite verification receipts

Read both conftests before direct unittest. Every test invocation used
`/private/tmp/pab-0c-venv/bin/python -I -B`, asserted **CPython 3.14.7**, and inserted
explicit absolute root, tests and tests/service paths. No pytest/listener fixture,
archive extracted-child test or real Docker/application subprocess was run.

The final broad selection ran **once: 239 tests in 89.700s, 236 PASS, one failure
and two errors**, zero skips. Receipt `/private/tmp/r357-final-tests.txt`, SHA256
`b417f0e9b6b6b10fbdffece918d75cfd33c4f82e3c973823b933e3a363fa598d`.
The selected existing intentional close-failure test also emitted its injected
`injected_close_unknown` shielded-future diagnostic; it passed its assertions.

The failure found a real new-adapter gap: clean history also needed explicit healthy
resource-monitor/protocol/integrity checks. Added those checks before clean-history
acceptance and registration. The two errors were test defects: reading authority
through a helper descriptor after context exit, and inserting a publication-work
row whose job key already existed. The fixture now reads its retained public file
and corrupts the existing publication binding for the negative case.

Only the three affected cases plus two affected positive journeys were then run:
**5 PASS in 11.557s**, no failures/errors. Receipt `/private/tmp/r357-corrections.txt`,
SHA256 `e05482d844933adbcbffe39adba57dbd2241875bf7fcf4c2383ab381aac5d618`.
**There is no all-green 239-test receipt at the final corrected candidate.** No
broad unchanged repeat was performed. `git diff --check` passed.

Final broad names (explicit selections, not discovery): new DaemonTests,
ProtocolTests, ChildAndClientTests, DaemonPublicTests, FailureAndPackagingTests;
SplitInit; install-child Canonicalization; InstallOwnership; InstalledHost;
Registration and ChildTransport; release GateTests; ActivationTests and
PrivateInputs; old InstalledPresetTests; Restart, TerminalRestart and ClosedRestart;
HostInstall; PublicJobControlTests; package CoordinatorTests,
CLIResultPublicationTests, InstalledCommandTests, GeneratedTopologyTests; and only
ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change plus
ArchiveTests.test_release_context_and_cpu_base. Full qualified names and verbose
results are in the retained receipt.

The correction selection was exactly:

```
tests.service.test_daemon_volumes.DaemonTests.test_local_facts_no_host_pid_equality_or_global_namespace_uniqueness
tests.service.test_daemon_volumes.DaemonPublicTests.test_unknown_unreaped_unacked_and_partial_publication_history_refuse
tests.service.test_daemon_volumes.DaemonPublicTests.test_stale_registration_and_unsafe_history_refuse
tests.service.test_daemon_volumes.DaemonPublicTests.test_empty_public_journey_same_root_reopen_and_frozen_job
tests.service.test_daemon_volumes.DaemonTests.test_install_up_status_stop_same_id_up
```

Earlier focused development receipts, retained in this thread, were separate runs:
12 run/one error (tuple-vs-JSON stopped-history comparison); 10 run/one failure and
two errors (fixture secret-generator interception/missing lane and cleanup error);
8 run/one error (client stable-file comparison included mutable access time);
2 run/one failure (new journal fixture directory ownership); one run/one error
and a separate one-run diagnostic/error (fixture omitted installed 0077 umask,
causing inventory `unavailable`); then one PASS of the corrected complete public
journey. They are not combined into a fictional single green run. Product fixes
included JSON normalization and stable file identity excluding atime. Other listed
issues were fixture adaptations; no shared validator was weakened to admit them.

The coherent new-policy journey uses the same helper-created Core/enrollment and
registration for authenticated empty discovery, preset registration, generation,
events/bounded wait, artifact bytes/hash, control reopen, immutable version/old-key
replay, a new key, clean modeled worker stop/up and another replay/new-key generation.
It models UID20000+ journal allocation/terminal/reap/ACK with the existing validator;
no OS engine child or real PID1 reap occurs. Fake daemon metadata/actual entrypoint
source and wire tests are distinct from actual Docker transport/kernel evidence.
Negative fixtures cover policy/context/security/mounts, source links/owners/secrets,
partial import/create/child/writer/notify/final phases, lock contention, frame replay,
changed hello/views, mixed evidence, stale startup and unsafe/unacked/unreaped/
partial-publication history. This is a finite selection, not exhaustive qualification.

## Final source SHA256 pins

```
343b5e7d74d0347a336cd7f196f456bbf420a6d87d5ed91aa1fc2a776eed358e  service/daemon_client.py
ce28bb77277c067f6724160160a0b79fc6104ed34247a6400449b13b9e2b7652  service/daemon_helper.py
f810fa52f82d1517ecdc2ee381965b51e241935fb3215f1b0f908b13c8e5fcff  service/daemon_runtime.py
48c77ce62879dfed2e8120df799aaa0630532b947ca653a8128922268988b6bf  service/daemon_volumes.py
11ceec54cc2441d5d1e6b586754d35f9a9dd9cab34f873a2122a1bd483ba9045  service/install_policy.py
0d8ee9aa8644d53851df30e7bb64d9f848950103c2a9be3c6a7a8f002232c6fd  service/persistent_init.py
17d7fe76d2c6a84dc900934644b594e6c01e4a8fc4e99af56723b9a920003925  service/install_child.py
4edfe6d24d3265dab6d29cabcfd52e0f1898dee8ae64bd6bbdbf97193bf1ebff  service/recovery.py
cdf08da80559695262635fc42fa44574214eefdab87bdf542b723382b5e29dc4  service/entrypoint.py
4a7d89b730cf1b6d4cc8f32a00504b7fb4ecb1228362ad553492109844b2685d  service/package_install.py
93cb6d60abcc4e1b5bba13752839e2f823040ff94e384177c5b7fdc87d4a9f49  service/supervisor_runtime.py
5baacac2c1ed5823f1fdefe98a5ed900ecd30e99a64cd8960816918632600abc  worker/runtime_protocol.py
b8fc09455aef1b0274bd58dc6d655fbb78c3b2c16e4c55c85e74b16d2e88f097  worker/runtime_supervisor.py
5853c15c43b9b7404876ea443bbd26909af235362bbce3623b399903e4359f5a  tests/service/test_daemon_volumes.py
c538273eee4607ad2e13402124cdce5122bbe47fda64442ff88195148304f379  headless-dist-manifest.json
4873fcefa13ba31685999f253ecf5e4a3c2b859de105ced789252268307969f9  service/compose.daemon-volumes.yml
ffc2c54b3cc3ca10581e277fbc457a2e965753d6e70f1ea0fb2ff29b895c7da0  docs/agent-interface/DOCKER-VOLUMES.md
74ec673bbd01e35befa8bf7e5ff0f76bd19acff319babfbcce19c88cc03c4b55  docs/agent-interface/CPU-DOCKER.md
a83cc38361701c2e26ebfeb5ef580270b6abe01478ce75dc3b559f67032c8ba2  docs/agent-interface/HEADLESS-INSTALL.md
```

The source and this report are committed together in this worktree; the parent
must independently resolve and review that returned commit. No self-referential
commit hash is embedded here.

## Remaining parent-owned gates

Fresh independent Sol review of the exact final candidate, including its retained
failed broad receipt and focused correction receipt, precedes adoption. This is
not an accepted release or a claim of exhaustive r356 clearance.

Separately authorized actual rootful Linux/amd64, THIS Intel Mac Desktop, and THIS
Intel Mac OrbStack qualification must establish local client transport, exact
cross-view named-volume identity, helper mount/capability/UID boundaries, permanent
child drop/no key or Core access by engines, final worker PID1/descendant reap,
denied peers/layouts, loopback forwarding, first authenticated audible Kokoro
speech and artifact bytes, clean same-volume completed-work replay/restart and
retained refusal on dirty/partial paths. Actual packaged Python 3.11/dependency/image
execution, ABI and measured cold/warm resources remain UNRUN. VM reset/restore or
changed retained inode identities refuse; desktop VM/daemon restart identity
behavior requires measurement, not silent re-enrollment. Finite history/helper/
provenance capacity and no automatic repair remain explicit limitations.

CUDA and native Metal compatible release rows, real audio/listening and relevant
installed cancellation/reap/renewal evidence remain separate mandatory gates.
No below-6-GiB waiver, operational authority or new target is supplied. Release
NO-GO; Metal UNQUALIFIED; legacy CLOSED; accepted native freshness/renewal retained.
