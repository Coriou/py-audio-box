# r355 Sol — independent r354 CPU Docker SOURCE review

2026-10-25. **Bounded SOURCE GO for r351 commission 2's configuration-generation/install integration and public first-task fixture. No blocking changed-seam defect found. Actual installation and release remain NO-GO.** Parent adoption is separate; this report does not adopt, merge or authorize operations. Metal UNQUALIFIED, legacy CLOSED, native freshness/renewal unchanged.

## Authority and pins

Read-only candidate: `/Users/ben/.txcode/worktrees/py-audio-box/t3-47fca048`.
Base `069e64dc570cced6c54873c047293aaf882869bc`; source/report HEAD
`999cf7ef35f9e7e228656ca3f83ea1fad34e2999`. Both resolve exactly. Candidate status was empty before and after the one test execution; HEAD unchanged. `git diff --check 069e64dc..HEAD` passed. No AGENTS.md was present at the candidate or checked explicit ancestors.

Read FULL r351 milestone (including commission 2), r354 report, PRODUCT, CPU-DOCKER and HEADLESS-INSTALL. Inspected the actual eight-file base-to-HEAD diff, setup, package coordinator/CLI/archive, host observer, install policy/ownership, split initializer and permanent identity drop, credential factory, installed activation/public fixture and archive consumers. Memory skill supplied authority/evidence discipline only, not current technical facts.

Independently reproduced SHA256 pins:

```
service/cpu_setup.py fbac1f3a9036bc3c73173f4555ed1aa5095de5cadc77071fe59d81935abc284f
service/package_install.py 74b85327e42f5a8d1e662216ba5685238078525892d734ecc17c6fd67bba809a
tests/service/test_cpu_setup.py 4d9bb895e8701c3ef326e8f6c7f9c95230e4533ef21baf30d699729e0d1c0e87
tests/service/test_installed_preset.py 8f38bb66e1ff47d3c12b55da4143685320b90a1321739486ec9e4c235b6d376e
headless-dist-manifest.json 49604058ea64939e29cabb86ee720c0e2cea6ff71c0afa8c0d42b298732633f8
docs/agent-interface/CPU-DOCKER.md 3e23369f737920e3ca7fc0855127dbfc69d69102f64347227f833cd77b654d4c
```

Sole lasting write is this root-checkout report. No candidate edits, commit, merge, push, delegation or operations. No Docker invocation/build/pull/setup, listener/network, install/download, model acquisition/ML, actual child process, hardware probe, crash, HELD/STT or operational cleanup. Only authorized inert temporary test files/SQLite/ASGI/fake metadata were used; their fixture lifecycle is not operational cleanup. No additional adversarial execution probes or test reruns were needed.

## Adversarial source findings

**No blocking finding in the bounded exit.** The useful change is concrete: a root operator inside the actual Linux daemon filesystem can derive the previously hand-authored stat/hash host policy and feed the existing finite installer through one bundle. This is a configuration generator for already provisioned reviewed inputs, not turnkey acquisition or an automatic usable-service claim.

`cpu_setup.generate` requires Linux real/effective root, explicit closed request/scope, exact Kokoro configuration matching the protected config bytes, and refuses Pocket substitution. DockerRootDir must equal the requested canonical absolute non-root path, contain the selected volume sources, and accompany x86_64/amd64 metadata. `_host_scope` checks the exact local socket/context and daemon ID/Linux OSType; volume metadata is reconciled against scope by the observer/topology validator. Local driver/options, source identity, same ordinary filesystem, alias/nested mounts, overlapping inputs and unknown consumers retain their existing refusals. Rootless, remapping, cgroup-v1, remote endpoint and Desktop denial were not removed.

Fresh roots require root:root 0700 and emptiness. The generated policy carries exact source device/inode and input ownership/mode/size/time metadata; config and sealed model bytes are hashed, credential contents are neither read nor hashed. Exact immutable local image IDs are inspected without pull. Two complete observer calls bracket inputs, mounts, consumers and scope; daemon root/architecture are checked again at return. Integer-only CPU/tmpfs/port policy and model enumeration/byte/time budgets remain finite. The setup transport shares the existing absolute Docker collection deadline. Actual large-model timing usability remains unmeasured.

The generated bundle is **trusted operator input**, not an authenticated proof or public configuration API. Request/bundle readers use bounded regular-file reads with leaf O_NOFOLLOW/O_NONBLOCK; they do not establish protected ancestry/ownership or immutable read snapshots for these operator documents. Output uses O_EXCL/O_NOFOLLOW, mode 0600, flush/fsync and refuses replacement, but does not pin its parent directory or fsync that directory. These are acceptable under the documented protected-path/trusted administrator premise, not a hostile-writable-directory security guarantee. Underlying operational input paths use literal descriptor ancestry and observer identity/hash corroboration. Root/daemon concurrency is trusted; setup is not a global lock against administrator mutation. Init independently locks/checks roots and emptiness before transitions, so a configuration bundle cannot bypass fresh initialization by racing setup.

Validation failure precedes bundle creation. Output I/O failure can retain a partial file, with no success message and no automatic deletion. Setup error publication admits only restricted category strings; package errors are fixed/redacted. Operational results publish after held-authority context exit, including final stability checks. The existing partial-create/unknown shutdown/quarantine refusals stay intact.

For `init/complete/up/status/stop`, `--setup-bundle` excludes configuration/scope/host-policy arguments, parses the closed configuration and scope, and uses the existing protected authority. Init validates and creates that authority; later commands take policy from the protected record, not a replacement policy in the public bundle. `layout` is descriptive intent, not a Mac attestation; it is not revalidated as a runtime support grant. The only new bundle-specific CLI test is status with injected held authority. Other command integration is supported by the common source path and existing coordinator tests, not five newly executed bundle end-to-end installations.

Actual initialization still calls `initialize_split`, retains ordered root/lane/Core exclusions and uses the synchronous permanently dropped UID10001 child for one Core. No new DB or scheduler. Worker root supervisor/UID20000+ engine isolation, credential-only-control mount, read-only narrow models/config mounts, networkless worker and capability restrictions remain unchanged. Neither runtime container receives Docker socket, host-root mount, host PID namespace or protected host authority. Public authentication remains the existing credential/principal/scopes boundary.

The first-task fixture is substantive within SOURCE limits: empty catalogue, configured discovery, CLI preset registration, MCP/HTTP replay, immutable version, real Dispatcher/Core SQLite generation/publication, CLI durable events, MCP bounded zero-second wait, asset bytes/hash, same-root control reopen and same-key replay with no new runtime authority. New-key generation reuses the version. Runtime/reap/backend observations and media proposal are injected; ASGI is in-process. The wait addition asserts a non-error result after completion, not a newly demonstrated live blocking wait; existing public-job-control tests cover bounded waiting/disconnect/cursors. This is not audible Kokoro or actual Docker restart evidence.

Distribution adds setup module and CPU documentation to the exact allowlist. In-process archive tests compare all entries/bytes and deterministic hashes; runtime build context already admits service Python. Extracted-child dependency closure and Python 3.11 execution were deliberately not run here.

## One independent execution receipt

Exactly one finite unittest selection, exit 0: **73 tests, 73 PASS, 10.444s**. Interpreter reported `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`, executable `/private/tmp/pab-0c-venv/bin/python`, flags `-I -B`. Current count was asserted equal to 73 before running. No Python 3.11 or pytest execution.

Reproduction from the candidate cwd, using that interpreter with a stdin script:

```python
import sys, unittest
root = '/Users/ben/.txcode/worktrees/py-audio-box/t3-47fca048'
sys.path[:0] = [root, root+'/tests', root+'/tests/service']
names = [
 'tests.service.test_cpu_setup.SetupTests',
 'tests.service.test_installed_host_policy.InstalledHost',
 'tests.service.test_install_ownership.InstallOwnership',
 'tests.service.test_host_install.HostInstall',
 'tests.service.test_installed_preset.InstalledPresetTests',
 'tests.service.test_public_job_control.PublicJobControlTests',
 'tests.release.test_release_package.CoordinatorTests',
 'tests.release.test_release_package.CLIResultPublicationTests',
 'tests.release.test_release_package.GeneratedTopologyTests',
 'tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change',
 'tests.release.test_release_package.ArchiveTests.test_release_context_and_cpu_base',
]
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
assert suite.countTestCases() == 73
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
```

Class counts: 7 + 20 + 14 + 7 + 3 + 4 + 13 + 2 + 1 + 1 + 1 = 73. Read both conftests and selected source names before execution. Explicit root/tests/tests-service paths prevent the historical `_maintenance_wait_guard` import errors. HostInstall's historical adapter calls use injected Popen/proc data. Excluded `ArchiveTests.test_deterministic_extracted_help_without_checkout` and `test_pocket_archive_bytes_and_isolated_dependency_closure`, which execute actual subprocesses; selected only its two static methods. Setup's archive check remains in-process. No listener conftest was invoked.

Seven current setup names: `test_archive_allowlist_bytes_and_determinism_without_child`, `test_bundle_cli_status_reuses_held_authority`, `test_exclusive_result_does_not_replace`, `test_mac_client_refuses_before_transport`, `test_real_observer_setup_and_denials`, `test_two_layouts_exact_generated_policy_without_credentials`, `test_wrong_root_arch_layout_owner_credential_and_nonempty_refuse`.

This is a new coherent all-green selection at the final HEAD. It does not rewrite r354's historical 71-test receipt (69 PASS, 2 import errors), corrected four-test receipt or final seven-test receipt into a historical single pass. Raw verbose output is retained in this review thread's tool receipt; no extra receipt file was written.

## Parent-owned remaining gates

Parent may accept this bounded SOURCE exit. Linux-local means x86 Linux on the actual daemon host. Intel Mac means an operator-accessible x86 Linux VM with its own ordinary rootful daemon and all authority/package/input/volume facts inside it. Default Docker Desktop/LinuxKit remains explicitly denied; default OrbStack support is not delivered. The layout selector proves neither the Mac host nor VM usability. This is useful bounded progress, with substantial reviewed-config/offline provisioning prerequisites; it is not convenient default Mac Docker installation.

Separately authorized actual offline dependency/image/model provisioning, Python 3.11 compatibility, clean Linux and Intel Mac VM install, kernel mount/UID/peer enforcement, authenticated first audible speech/artifact replay, and clean/failed stop/reap/restart remain UNRUN. CUDA installed routing, native Metal/ABI, real listening and cold/warm resource qualification remain mandatory. Below-6-GiB heavy-operation gate is unchanged and was not measured or waived. No renewal/native freshness/legacy acceptance is reopened by this change. **Actual release NO-GO; Metal UNQUALIFIED.**
