# Native authenticated public launch implementation — r317 Sol

2026-10-25. SOURCE candidate for fresh independent Sol review; **no self-adoption,
merge, operational or release acceptance**. Implemented an explicit native installed
public HTTP lifespan/command consuming the reviewed native owner factory. Final
relevant selection ran once: **106 PASS / 65.503s / exit 0**. Native Metal remains
**UNAVAILABLE / UNQUALIFIED**. This does not complete the installed journey.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-4bc474cd`, branch
`t3/native-public-launch`, clean at entry. Base
`87c4908730d4672dfccf3c5f7ddb7d2ee8db8591` integrates reviewed native retained
restart after r316. SOURCE/tests/operator usage commit:
`29d6628e87379e7aae2b42c5dc127808454ae084`. This subsequent report-only commit
does not change tested source. No root-checkout writes, merge, push or delegation.

Read PRODUCT.md, NATIVE-PACKAGE.md, normative r277 and r316. Inspected actual
entrypoint, package/host launcher, CLI/MCP/HTTP, auth, native package/enrollment,
NativeActivation, Dispatcher and relevant fixtures/conftests. Repository/ancestor
AGENTS.md discovery found none. Bounded memory-registry lookup had no relevant
native-owner result; no technical conclusion relies on memory.

Tested source SHA256:

- service/native_launch.py: `77f1cb0a1fb25d4a3a3b4ecc68e2a51c6b55dff249aedd1bc3259e3f481ea08e`.
- service/entrypoint.py: `743e7fba5d2e9509300c63ebce5d5091374cf7292b7a553521708f830a1b5671`.
- tests/release/test_native_launch.py: `4f97fa97bf93b5cae10078780549c27550ac0664c2bb39db0d756a15069b4122`.
- Unchanged lib/tts.py: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

## Implemented behavior

`service/native_launch.py:create_native_app` accepts explicit durable enrollment,
control configuration, native templates, owner/staging and credential paths. It
uses `create_native_owner`, never a public runtime/config object or connector.
NativeRuntime remains constructed on the Dispatcher control-writer thread;
existing enrollment/hello/trust brackets, Core lifetime exclusion, retained-history
proof and import/publication/admission policies remain in force. NativeActivation,
native runtime, Core, scheduler, native history and public contracts are unchanged.

The existing Linux installed lifespan was extracted into `_owned_app` without
replacing its thread-owned Future, shielded acquisition, cleanup advancement or
strong process-lifetime registry. Both variants retain cancelled eventual
acquisitions; failed/unknown close is retained and not automatically retried.
The native wrapper waits for existing reconciliation/persistent activation before
mounting the existing authenticated `create_voice_app`. An app reused through a
second lifespan replaces its previous mount so it cannot address a closed owner.
Normal process restart still constructs a fresh app and owner.

`auth.decode_credentials` shares the existing closed installed credential decoder;
Linux root/group/mode file policy remains unchanged. The separate native consumer
uses existing canonical native no-follow reads, owner 0600 single-link file and
0700 credential directory. Principals must occur in templates; existing read and
production-write scopes, token limits and constant-time resolver remain unchanged.
Credential bytes and canonical template bytes are checked before acquisition and
before mounting. Immutable dedicated-account inputs/no hostile same-UID writers
remain premises, not a new hostile-writer isolation claim. No hot reload.

`NativeQwenContexts.capabilities` describes registered native principal metadata;
Dispatcher substitutes its current configured profile as before. This prevents
public native discovery falling back to synthetic fixture profiles. No profile,
inventory timestamp, model or qualification metadata is promoted. Native status
separates persistent activation from speech: installed native owner, non-synthetic,
activation_ready from owner status, ready_for_speech false and explicit
UNAVAILABLE / UNQUALIFIED. Existing request validation, errors, ownership,
production resolution, auth middleware and client operations are reused.

`scripts/native-installed.py serve` is a separate explicit command, not a Linux
installer mode. Every input path, loopback host and port is required; only
127.0.0.1/::1 and ports 1..65535 are allowed. No environment-derived service
configuration, wildcard bind, enrollment, boot learning, trust renewal, install,
download or model discovery. It uses existing Uvicorn and installed TERM/INT drain
policy, no access log, critical server logging and a fixed redacted launch failure.
Success requires a started server and successfully closed lifecycle. No listener
was exercised here. Loopback is the control host's loopback, not a Docker routing
or mount proof. Docker remains the intended control/DB owner topology; this source
command does not supply or validate a native Docker deployment.

Both exact archive manifests add only `service/native_launch.py` and
`scripts/native-installed.py`; changed existing files were already allowlisted.
Native archive remains source-only, not an interpreter/dependency bundle. Operator
usage, private credential schema, existing CLI/MCP API use and remaining gates
are documented in NATIVE-PACKAGE.md. Existing Linux command/routes/install policy
and source lock metadata are unchanged.

## Verification and development evidence

Read tests/conftest.py and tests/service/conftest.py first. Direct unittest avoids
pytest autouse maintenance and subprocess/listener fixtures. Injected native TLS
uses the existing in-memory NativeFixture; command test injects the server;
archive help uses in-process runpy. Selected native-child lifecycle paths use fake
backend/process/receiver evidence, not actual child execution. CLI/MCP parity uses
local ASGI transport with existing clients and authentication/errors.

Development runs preceded the frozen selection. The first run loaded the launch
module (also collecting its imported ten enrollment cases) and failed at the
second public lifespan because the old mount retained the closed owner. Fixed by
replacing the prior mount. A subsequent four-case run found a test orchestration
error: cancelling startup once waits for shielded cleanup, so awaiting it before
releasing the fake acquisition deadlocked until the test's four-second guard.
Corrected the test to repeat cancellation, matching existing cancellation tests.
Four-case and then six-case development selections passed. Added credential-policy
and mocked successful CLI command tests before freezing. No failed final selection
or final retry. These were tiny inert development checks, not operational attempts.

Final invocation: `/private/tmp/pab-0c-venv/bin/python -I -B -` from the worktree,
explicit `sys.path[:0]` entries for the absolute worktree and its `tests/service`.
Printed and asserted Python **3.14.7**, full runtime
`3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`.
Used `defaultTestLoader.loadTestsFromNames`, printed count **106** and full name
selection, `TextTestRunner(verbosity=2, failfast=True)`, nonzero on failure.
Session **4247**, initial chunk **6d7665**, terminal chunk **bc55e6**:
**Ran 106 tests in 65.503s / OK / exit 0**.

Exact final selection (class entries mean all tests in that class):

- tests.release.test_native_launch.NativeLaunchTests — 8.
- tests.release.test_native_history.NativeHistoryTests — 44.
- tests.release.test_native_activation.NativeActivationTests — 8.
- tests.release.test_native_enrollment.NativeEnrollmentTests — 10.
- tests.service.test_installed_activation.ActivationTests — 22.
- tests.service.test_installed_activation.PrivateInputs — 3.
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- tests.release.test_native_package.NativePackageTests.test_archive_exact_members_secrets_excluded_extracted_help.
- Same NativePackageTests.test_archive_deterministic_bytes.
- Same NativePackageTests.test_extra_archive_member_refuses.
- tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change.
- tests.release.test_release_package.InstalledCommandTests — 2.
- tests.release.test_release_package.SignalPolicyTests — 1.
- tests.service.test_native_lifecycle.NativeLifecycleTests.test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job.
- Same NativeLifecycleTests.test_stale_future_native_registration_and_historical_decode.
- Same NativeLifecycleTests.test_cancel_retains_reservation_until_actual_owned_receipt.

Expected injected failed-close tests emitted Python 3.14 shielded-future diagnostics
(`close_unknown` / `injected_close_unknown`); their ownership/no-retry assertions
passed. CLI refusal tests emitted argparse usage for missing inputs/wildcard bind.
These contain inert fixture errors, not real credentials or operational failures.
`git diff --check 87c4908..HEAD` passed and tested source stayed clean.
Observed available disk 2,845,072 KiB, below 6 GiB. No network/listener, Docker,
install/download, actual child, ML, crash, HELD/STT or operational cleanup. Only
fixture-owned temporary teardown; HEAVY ACTUAL admission not exercised or waived.

## Remaining gates and limits

Fresh independent Sol review and parent adoption remain required. The native
history language is unchanged: zero jobs or fewer than 64 clean succeeded,
single-attempt non-clone jobs under the same immutable configured identity/boot/
trust, with existing row/scalar/import/receipt/publication bounds. Unsupported
failed/cancelled/retried/clone/crash/unfinished retained histories still refuse.
Public native generation from a newly onboarded voice is not demonstrated by the
new public-read tests; existing seeded native lifecycle and Linux authenticated
job/artifact tests retain their own source-only scopes.

Fresh owned native voice onboarding, explicit Docker topology/mount/firewall and
reachability validation, and stopped/reconciled boot/trust renewal remain next
slices. No reviewed arm64 wheelhouse or actual compatible installed dependency/
interpreter ABI exists here. Hardware/model/MPS, actual TLS/denied peers, process/
FD/death/resource behavior, generate/cancel/restart, clone lifetime, audio and
performance/release qualification remain unrun. CPU/CUDA qualification remains
mandatory. Known legacy lib/tts digest RED is unchanged/unwaived; no legacy digest
regeneration or digest test. No source/inert result authorizes an installed,
hardware, acoustic or release readiness claim.
