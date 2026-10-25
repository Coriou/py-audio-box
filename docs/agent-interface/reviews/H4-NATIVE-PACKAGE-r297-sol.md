# Native package and private control consumer — r297 Sol

2026-10-25. **Partial Lot B SOURCE delivery only; NO-GO for complete B acceptance,
installed native activation, dependency ABI resolution or Metal speech.** This
implements a usable native source archive/launcher, strict offline package
preflight and a private control configuration consumer. It does not implement
the complete Docker/native enrollment/history policy or public first-use preset
registration. Parent independent Sol review remains required; no self-acceptance.

## Exact authority and pins

Implementation worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-0d06d538`.
Clean initial full HEAD `df0fafe2fe6a06925c46be4ae57a8c98af131ab4`.
SOURCE commit **`dba2d33033c25f081a709c557fbbafa56fd49af8`**. This report is a
separate successor documentation commit; identify that commit from the handoff.

Read the entire r277 contract, including B/C; full r295 and r296 reports. r296's
96 PASS is parent historical SOURCE evidence, not a new execution here. Inspected
actual NativeSupervisor CLI/registration/launch/serve, private protocol and
NativeRuntime consumers/import fences, NativeQwenContexts, installed package and
policy, host policy/history and Linux recovery refusal seams. No claim of a new
full audit of Lot A. No applicable AGENTS.md found by repository/parent discovery.
Both conftests were read before direct unittest execution; pytest collection,
maintenance guard and HTTP listener fixtures were not activated. A bounded memory
registry search for r277/r295/r296/Lot B returned no hit; no memory technical
facts were used.

The user's follow-up confirmed no reviewed Darwin arm64 Python or wheelhouse
exists in this session and authorized exact operator-supplied inputs with strict
validation and truthful unverified state. No further question/delegation.
Observed free storage 3,164,320 KiB, below 6 GiB. Only small inert source ZIPs,
metadata-only fake wheels, tiny existing model metadata/WAV fixtures and fake
TLS/process observations were used. No actual socket/listener/process launch,
ML/hardware, Docker, network/acquisition/download/install/model installation,
crash experiment, operational cleanup or HELD operation. Disposable unittest
fixture teardown only. Production headroom gates unchanged.

Preserved pins:

- r277 SHA256 `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`.
- lib/tts.py SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
- service/native_package.py SHA256 `ccb8611e2b1b227fb7ace1c9898d57314e5e18e5ceb90542528a1c0ef885f848`.
- native-dist-manifest.json SHA256 `5c631d9263ba04a3ed88906c3513688d0ad38499f8a6faa980b4fc0016dfcb83`.
- tests/release/test_native_package.py SHA256 `133a938dad127e7cc82a66f953b8e8851becee95659b6089bbf86a029339f1f4`.

## Delivered behavior

Five source/package files changed:

- `service/native_package.py`: `archive`, `check`, `serve`, plus the private
  `enrolled_runtime` control consumer. No import-time acquisition or ML.
- `native-dist-manifest.json`: separate named native source allowlist. No Python
  binary, dependency wheels, Linux locks, weights, private configs or keys.
- `headless-dist-manifest.json`: explicitly archives the new module/manifest/docs;
  existing Linux platform declaration and policies remain intact.
- `docs/agent-interface/NATIVE-PACKAGE.md`: invocation, exact manifest/config
  shapes, exposure/credential instructions and concrete unavailable gates.
- `tests/release/test_native_package.py`: 15 inert unittest methods.

The extracted `native-worker.py` runs with an operator-selected **absolute**
managed interpreter using `-I -B`. Preflight requires exact current executable,
full Python version/build/implementation, Darwin/arm64 and executable hash. It
refuses symlink, foreign-owner, writable-by-group/other or nonregular artifacts.
The launcher consumes the checked config snapshot directly through actual
NativeSupervisor/serve classes, avoiding a second unchecked config read. Duplicate
registration bindings refuse before supervisor construction. Lot A modules are
unchanged, including their minimal child environment/FD and launch fences.

The canonical runtime manifest records the whole exact installed distribution
set, every distribution's selected offline wheel/hash, retained source archive
hash, model manifest/revision/license/access metadata. Requires-Dist edges and
markers must close against exact installed metadata. URL/extra dependencies and
local-version locks (including Linux +cpu) refuse. One wheel per distribution;
filename name/version/platform tags and bounded WHEEL/METADATA headers must
agree with the current interpreter. Missing or incompatible wheels refuse. No
pins, official wheel hashes or compatible native ABI are invented.

Hashes and metadata do **not** prove those wheel bytes were installed, verify
installed binary contents, establish binary loader/ABI success, or establish an
upstream no-extra-process premise. No Torch/Qwen import is used as a proxy for
qualification. Operator license/access strings are recorded provenance, not a
legal determination. Existing complete Qwen bundle byte/capability verification
is reused for each exact configured model/descriptor mapping.

Config/trust reads walk absolute paths without following symlinks and require
fixed current owner, 0600 single-link regular files, private immediate 0700
roots, bounded stable contents. Canonical JSON rejects duplicates/nonfinite
values. Native bind is numeric, nonwildcard/nonmulticast; loopback requires an
explicit proven host-network declaration. Package docs require explicit reachable
Docker bridge interface/port and restrictive host firewall, numeric resolved
control endpoint, stopped/reconciled credential rotation, and client-only mounts
into Docker control. This portion neither creates mounts nor proves firewall or
routing. No native server key is provided to Docker or the inference child.

`enrolled_runtime` accepts a **distinct closed native control configuration**,
validates private client files, creates actual TLS1.3/pinned NativeRuntime and
requires hello to match all five configured deployment/config/lane/boot/package
identities. Quarantine or retained import uncertainty refuses. It does not learn
boot from hello or promote a profile. Tests exercise actual NativeRuntime and
actual supervisor hello via the existing fake TLS connector; the factory's
optional connector is an internal test seam, not a public operation.

A successful check prints only package prepared / dependency metadata verified /
ABI unrun / qualification unqualified / speech unavailable. It writes no readiness
receipt. A package prepared state is not installed activation.

## Execution receipts

All project interpreters were `/private/tmp/pab-0c-venv/bin/python -I -B`, with
explicit sys.path entries for this worktree and its `tests/service`; direct
`unittest.defaultTestLoader.loadTestsFromNames` and
`TextTestRunner(verbosity=2, failfast=True)`. No pytest collection.

Chronological observations (overlap is intentional; do not sum):

1. Initial package selection: **11 PASS / 1.043s**.
2. Package + unchanged Linux host policy + protocol: **39 PASS / 2.445s**.
   Names: `tests.release.test_native_package`,
   `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`,
   `tests.service.test_installed_host_policy.InstalledHost`,
   `tests.worker.test_native_protocol`.
3. After adding wheel validation: one **13-method failfast invocation failed**
   at `test_wheel_platform_name_version_and_metadata_refuse`. The production
   refusal occurred but its specific reason was masked by a broad ValueError
   catch (Refused subclasses ValueError). Fixed by preserving Refused before
   translating malformed metadata errors. No candidate acceptance from that run.
4. After correction: **19 PASS / 3.567s**: package selection plus
   `tests.service.test_installed_pocket.InstalledPocketJourney.test_closed_policy_enrollment_and_reference_free_factory`,
   `...test_exact_readonly_mount_and_launcher_policy`,
   `tests.service.test_installed_pocket.DiscoveryTests`,
   `tests.worker.test_native_protocol.NativeProtocolTests.test_receipt_requires_trusted_full_binding_and_owned_wait_shape`,
   `tests.release.test_headless_install.GateTests.test_wrong_config`.
5. **Final source package selection: 15 PASS / 1.395s**, after checked-snapshot
   serve handoff and strict integer-version changes. Explicit module
   `tests.release.test_native_package`. Covers extracted module/import and actual
   launcher help in-process, exact membership/secret exclusion, mutated source,
   wrong platform/build/executable/hash, wheel metadata/platform mismatch,
   missing closure/version/+cpu/artifact coverage, duplicate/nonfinite JSON,
   wrong owner/key mode/symlink/hardlink, pin/topology/runtime/model mismatch,
   actual bundle verifier, actual fake-TLS NativeRuntime hello with stale boot
   and wrong peer refusal, Linux policy refusal and injected no-listener serve
   handoff. Fake wheels contain no importable package/binary and are explicitly
   nonqualifying. Tests patch native platform/build observations; no real arm64
   interpreter is qualified.

`git diff --check df0fafe2fe6a06925c46be4ae57a8c98af131ab4..SOURCE`: PASS.
Clean source worktree after commit. Prior Linux/Pocket checks predate the last
native-only handoff change; final focused package checks cover that change.
No new broad 96-method Lot A run was necessary or claimed. Known legacy digest
RED remains unwaived/unregenerated/unexecuted.

## Explicit remaining B and C gates

**Not complete B.** The private native configuration consumer is a concrete
package seam, not a completed installed enrollment system. Remaining work:

1. Docker control native variant and mount/topology validation, durable native
   enrollment/provenance transaction, current boot/config/hello bracket under
   owner acquisition, and native installed startup/completed-history readers.
   Current Linux recovery still rejects native at its explicit guards; no broad
   history waiver was added. No native DDL/queue/recovery framework added.
2. Public first-use Qwen preset bootstrap via owner VoiceStore/idempotency/CAS
   from an empty catalogue, exact bundle/voice authorization, then generation
   through actual native classes and same completed native/control history
   reopen. **Not implemented/tested here.** Existing seeded Lot A journey is not
   a substitute and was not relabeled. No Core seeding was added.
3. Installed discovery prepared/qualified/unqualified integration. No speech or
   readiness promotion is supplied. The package CLI reports unavailability.
4. Actual selected Darwin arm64 build/wheel acquisition under separate authority,
   installation provenance, dependency imports/ABI and model compatibility.
   Metadata/wheel tags are insufficient; all remain **UNRUN**.
5. Actual TLS peers/host bridge/firewall, Darwin child/FD/resource/descendant
   observations, Metal probe/load/generate/cancel/restart, acoustic/performance
   qualification and release. Native Metal remains **UNAVAILABLE / UNQUALIFIED**.
   Clone nonempty conditioning/cancellation limits remain. CPU and CUDA release
   rows remain mandatory and separate.

The parent should independently review this exact partial source commit and
commission the remaining bounded installed/onboarding work before accepting B.
