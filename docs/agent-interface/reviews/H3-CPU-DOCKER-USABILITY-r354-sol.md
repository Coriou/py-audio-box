# r354 Sol — CPU Docker usability implementation

2026-10-25. Implementation candidate for r351 commission 2; independent fresh Sol
review and parent adoption remain separate. **No self-adoption or release GO.**
Actual clean Linux/Intel-Mac-VM install and audible speech remain UNRUN. Metal
UNQUALIFIED; legacy CLOSED; native freshness/renewal unchanged.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-47fca048`, clean initial
status, base HEAD `069e64dc570cced6c54873c047293aaf882869bc`. Read r351 commission
and surrounding milestone, r353 independent PASS, PRODUCT, existing installation
policy/host observer, split initializer, package/entrypoint, credential, CLI/MCP and
public preset/reopen fixture. No AGENTS.md found in worktree-parent discovery.
The memory registry supplied authority-boundary discipline only, not a current
technical premise. No root-checkout writes, merge, push, delegation or adoption.

Source tree before adding this report:
`8b75cd61b93904d7e9addf52539168c5e5e2cd15`.

SHA256 pins:

```
service/cpu_setup.py fbac1f3a9036bc3c73173f4555ed1aa5095de5cadc77071fe59d81935abc284f
service/package_install.py 74b85327e42f5a8d1e662216ba5685238078525892d734ecc17c6fd67bba809a
tests/service/test_cpu_setup.py 4d9bb895e8701c3ef326e8f6c7f9c95230e4533ef21baf30d699729e0d1c0e87
tests/service/test_installed_preset.py 8f38bb66e1ff47d3c12b55da4143685320b90a1321739486ec9e4c235b6d376e
headless-dist-manifest.json 49604058ea64939e29cabb86ee720c0e2cea6ff71c0afa8c0d42b298732633f8
docs/agent-interface/CPU-DOCKER.md 3e23369f737920e3ca7fc0855127dbfc69d69102f64347227f833cd77b654d4c
```

## Delivered behavior

`speech-box setup --request FILE --output NEWFILE` is a finite daemon-side
configuration/introspection operation. Explicit request inputs bind one existing
Kokoro CPU configuration, exact scope/volume identities, daemon root, protected
input paths, finite CPU/tmpfs budgets and port. The generator reads no credential
contents and performs no acquisition, model selection or substitution. It generates
the exact host policy stat identities and public model/config hashes instead of
requiring operators to hand-author them. Output contains configuration, scope and
policy and says `configuration-only`, never installed/ready. `init`, `complete`,
`up`, `status` and `stop` accept the generated `--setup-bundle` exclusively with
respect to the original configuration/scope/policy arguments.

Generation requires Linux root on the daemon host/VM, x86_64 Linux daemon metadata,
exact DockerRootDir containing selected volume roots, the existing local socket
and exact daemon ID/volume metadata. Fresh roots must be empty root:root 0700.
The existing observer checks rootful/cgroup-v2/no-remapping policy, exact models,
config/credential metadata, mountinfo and disjointness/consumers twice. Images
must already exist at exact local immutable IDs. Root/architecture are bracketed
again before returning. Invalid observations produce no successful output bundle;
exclusive output refuses replacement. An output I/O failure may retain a partial
file, without success, for operator review. No rollback or cleanup is inferred.

The actual initializer and one Core, permanent identity-drop child, existing
protected host authority, registration and finite launcher remain canonical.
Control/worker UID, credentials, capabilities and mounts are unchanged. There is
no daemon socket in either runtime container, blanket privileged mount,
initializer container, scheduler, persistence engine or public trusted-config API.
Pocket remains the explicitly selected separate installation and is refused by
this Kokoro setup command; native Qwen voices never fall back to it.

`CPU-DOCKER.md` specifies `linux-local` and `intel-mac-linux-vm`: the latter means
an operator-accessible Linux VM with its own ordinary rootful Docker engine and
all package/input/volume/authority facts inside that VM. It does **not** claim
Docker Desktop's default LinuxKit daemon support; the existing Desktop denial
remains. OrbStack's default service is also not qualified. An OrbStack Linux
machine with its own daemon is only a candidate when its actual metadata,
mount/proc and ownership checks satisfy the same policy. Layout names are operator
intent, not evidence that the machine is a Mac or a product was qualified.

The authenticated public preset fixture now covers CLI durable events, MCP bounded
wait, artifact access and exact same-key generation replay after control-only
reopen, asserting no runtime authority/launch on replay. Existing discovery,
CLI/MCP preset registration, generation, byte/hash retrieval, immutable versions,
auth denials and new-key generation remain in that real-SQLite/fake-runtime path.

## Finite verification receipts

Only direct unittest under `/private/tmp/pab-0c-venv/bin/python -I -B`, reported
**CPython 3.14.7**, with explicit absolute worktree paths inserted into `sys.path`.
Read both conftests first; their listener fixture was not invoked. No pytest,
actual subprocess, listener/network, Docker execution/build/pull/setup, model/ML,
hardware probe, install/download, host crash, HELD/STT or operational cleanup.
Disposable test fixture file lifecycle is the only cleanup.

Initial finite selection: **71 run, 69 PASS, 2 import errors** in 8.396s.
Selected new SetupTests; InstalledHost; install ownership; host install; installed
preset; public job control; package CoordinatorTests, CLIResultPublicationTests,
GeneratedTopologyTests and manifest/release-context checks. The two historical
HostInstall adapters import `_maintenance_wait_guard`; isolated sys.path lacked
its explicit `tests` directory. No implementation failure was reported. Receipt
`/private/tmp/r354-tests.txt`, SHA256
`bf87050d5ea7f22b7f603360e8a4ad118f00e1db52a8357fce8f4b2f7ebc8d90`.

Corrected explicit root **and tests** paths: only the two import-error tests plus
two newly added real-observer/bundle-CLI tests: **4 PASS**, 1.826s. Receipt
`/private/tmp/r354-focused.txt`, SHA256
`b7f524b4481d58300b6d32e1f8913771483e6d0f869d9a5e7cd40a3af4014b6d`.

After bounded request-read and final daemon metadata bracket changes, final finite
SetupTests only: **7 PASS**, 0.500s. Receipt
`/private/tmp/r354-final-setup.txt`, SHA256
`72f1776acd8af64d97b7d823a831f38f569bb6ac5ca6ab1d51f4e12d49433d3f`.
No broad unchanged rerun. These are separate receipts, not an invented single
all-green 71-test run. `git diff --check` passed.

The final suite includes both layout names, real observer with fake Docker
metadata, wrong root/architecture/layout/owner/credential/nonempty denials,
remote socket and wrong daemon, unchanged exclusive output, bundle CLI and
archive exact allowlist bytes/determinism without executing extracted children.
Existing selected guards cover rootless/Desktop/remapping/cgroup, mount/owner/
credential mutations and daemon mismatch; public tests cover CLI/MCP/auth/reopen.
Archive subprocess-based tests were deliberately excluded; the new archive test
checks exact entries and bytes in-process. Installed CPython 3.11 compatibility
is not established by this receipt.

## Remaining parent-owned gates

Fresh independent Sol review of the exact candidate and adoption are required.
Actual separately authorized offline artifact/dependency provisioning, clean
install, denied-peer/mount/UID/process enforcement, authenticated first audible
speech, artifact replay and clean/failed restart on Linux and Intel Mac Linux VM
remain UNRUN. This lane generates configuration from already provisioned exact
reviewed configuration/assets; it does not turn arbitrary models into enrollments
or provide acquisition. Docker Desktop default daemon support is not delivered.

Installed native Docker-host routing/peer isolation, CUDA, native Metal and ABI,
real audio/listening and cold/warm resource qualification remain mandatory separate
release work. The below-6-GiB heavy gate has not been measured or waived. Release
NO-GO remains; source/config fixtures are not installed or acoustic authority.
