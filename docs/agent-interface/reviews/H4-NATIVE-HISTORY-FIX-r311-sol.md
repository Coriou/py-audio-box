# Native retained history corrections — r311 Sol

2026-10-25. SOURCE candidate for fresh independent review; no self-adoption.
All five r310 invalid acceptances now have explicit refusal checks and passing
regressions. Final finite selection: **58 PASS in 39.073s, exit 0**. This includes
the prior 40-test selection and 18 new refusal regressions; existing passes alone
are not evidence that the reproduced gaps were fixed.

## Authority and exact pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-e6631dc9`, branch
`t3/native-history-acceptance-fixes`, clean at entry. Fresh parent HEAD was
`2e0f82aec6a9aff2f82d1a27fdc2e14f6e5f3b35`, without r309 implementation.
Required cherry-picks, in order, without conflicts:

- `9409178dfb4ee4fec25f679a3f4db82145e9f4a1` → `12dc00ff23b3a8025b21326f75fe53e9bfa35169`.
- `ee682f633ab81f091d3d5f2d2082cce8050f2793` → `ab4bc72248f08c0b6880f1e99fe55238a3dbe259`.
- `8f875ae686da6daf0f30570c7066b28404182d65` → `01d76f03870459e87545b5f3f5dd0a95b4cff35a`.

Final SOURCE/tests: `243a2fb7bcbaf1036fc889029b9939aa4ba25c66`.
This report is a subsequent documentation-only commit.

Read full root r310 exact reproductions, r309, normative r277, PRODUCT.md and
NATIVE-PACKAGE.md; inspected actual Authority, frozen execution/profile rules,
Admission/ResourceStore, DispatchStore/Dispatcher, Core duplicate/retry/publication
and preinitialization ordering, shared detached validators and test fixture seams.
No AGENTS.md found in repository discovery or checked ancestors. A bounded memory
registry search returned no relevant r309/r310/native-history entry; no memory
technical claim is used. No root-checkout writes, merge, push or delegation.

## Findings closure

**F1 P1 — retained resource demand.** The reader closes the demand key set, requires
exact integer dimensions (excluding bool/float), 0..2**53-1 finite bounds, and positive
slots/RAM/PIDs/disk/attempt disk. It corroborates the entire canonical demand against
`service.admission.native_demand` derived from the frozen native execution and profile:
one slot, descriptor RAM, zero dedicated VRAM, one PID, profile disk for both disk
fields and the canonical execution configuration hash. This pure calculation is
also the live Admission producer, after its existing authorization and current
hardware checks. Historical validation does not expire frozen inventory, reserve,
repair or mutate any charge. One attempt implies no replacement disk increment.
The native control transfer disk floor remains enforced by the shared rule.

Regression reproduces all five negative dimensions at -1 together, plus float,
bool, zero, overflow, nonfinite, unknown-key and positive-but-wrong reservation
mutations. Each refuses. No negative charge reaches ordinary admission startup.

**F2 P2 — retry lineage.** Require `retry_of IS NULL` explicitly for this finite
nonretry slice. The exact self-reference reproduction refuses; a single attempt
is no longer treated as proof of absent lineage. Core's failed/cancelled predecessor
retry rules and all general retry behavior are unchanged.

**F3 P2 — fingerprint.** Recompute with the existing `contracts.validate.fingerprint`
producer, exactly as Core.submit_execution and Dispatcher duplicate lookup do,
using retained principal/operation and the Authority-frozen plan request. Shared
document validation already binds principal/operation/key/id to the retained job;
canonical context comparison binds that request to the frozen Authority. Wrong
persisted fingerprint refuses. Positive completed-history test replays the original
production request/key after each of two reopens, returns the same job ID and
still observes exactly one inference fixture call.

**F4 P2 — command obligations.** Close the Dispatcher journal namespace to
prepare/start/observe/stop/collect/ack, with only returned commands accepted in a
completed history. Foreign-key validation binds each command to its dispatch.
Order by durable command sequence; require one prepare then one start, any
observe/stop commands before collect, collect before ACK, and only ACKs thereafter.
Require the completed dispatch's runtime and qwen fixture relationship. This is
the Dispatcher command journal, not the larger native RPC body/transfer namespace.
Unknown returned operation, unknown state, missing start and invalid ordering
regressions refuse. No unknown obligation is deleted or ACKed.

**F5 P2 — exact immutable joins.** Compare canonical bytes for frozen context,
configured native identity, frozen job, retained outcome and artifact metadata.
Normalize only the completed attempt's reaped field back to false when comparing
the frozen running job; exclude only intended mutable state/stage/artifact_ids.
Require succeeded/completed, exact True reaped, and the existing exact artifact ID
relationship. Job/outcome generation int→float substitutions, context descriptor
numeric substitution and artifact metadata generation substitution all refuse.
Other new history joins were inspected: Authority snapshot/binding, receipt,
proposal/output, snapshot/import marker bytes already compare canonical or raw
bytes; remaining IDs, hashes, names and classifications are string identities.
No accepted document is rewritten.

All refusal tests now snapshot the logical SQLite dump and spool/staging/assets
bytes after mutation, and verify they remain unchanged after refused startup,
with no extra ACK. Where startup constructed a NativeRuntime, they also assert
its import closure remains populated and import_completed remains empty.
The clean zero-job activation now reopens twice; completed-job reopen twice,
artifact retrieval and reservation retention continue to pass.

## Finite execution receipt

Inspected tests/conftest.py and tests/service/conftest.py before execution. Used
direct unittest, avoiding pytest maintenance and socket/uvicorn fixtures. Tiny
fixture audio/filesystem/model/ZIP metadata and injected TLS/process/backend
observations only; no real child/ML or listener. Observed free disk 2,950,112 KiB,
below 6 GiB; user explicitly authorized this tiny inert work. No HEAVY ACTUAL
admission was exercised or waived. No network, Docker, download/install, crash,
HELD/STT or operational cleanup. Only fixture-owned temporary teardown occurred.

Both invocations used `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit
absolute worktree and worktree/tests/service sys.path entries, asserted interpreter
Python **3.14.7**, printed Clang 17.0.0 (not target Python 3.11), asserted selection
count 58, defaultTestLoader.loadTestsFromNames and TextTestRunner(verbosity=2,
failfast=True), nonzero exit on failure.

Initial invocation stopped after one test with a test-helper AttributeError:
Dispatcher construction raises before assignment, so the new refusal helper could
not access self.dispatcher.installed_activation. **1 test / 1.057s / exit 1**,
session 95297, chunks 475ccc/cf5235. Corrected only the test helper to retain the
NativeActivation before constructing Dispatcher; no production correction or
weakening of assertions. This failure is retained here, not counted as a pass.

After that final edit, the finite relevant selection ran once:
**58 tests / 39.073s / OK / exit 0**, session 77277, chunks
675581/8de82b/3d4ff9/f0dcaf/ec32c1. Exact selection:

- tests.release.test_native_history.NativeHistoryTests — all 34.
- tests.release.test_native_activation.NativeActivationTests — all 8.
- tests.release.test_native_enrollment.NativeEnrollmentTests — all 10.
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- tests.service.test_installed_activation.ActivationTests.test_preowner_refusal_never_constructs_core.
- Same ActivationTests.test_post_core_private_reader_under_lifetime_lock.
- Same ActivationTests.test_changed_post_acquire_hello_refuses_and_releases_core.
- Same ActivationTests.test_changed_post_acquire_second_hello_refuses.
- tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change.

`git diff --check` passed. Changes after inherited r309 are limited to admission.py,
native_activation.py and the two release test files, plus this report. Core's
lifetime/schema-preinit ordering, owner-thread runtime construction, publication
intent and import fences remain intact. recovery.py, reconciliation.py,
native_runtime.py, package/trust inputs and Linux policies are unchanged.

SHA256 at final SOURCE:

- service/native_activation.py `4a2c34bf4df4454bb1c6f4c15904487a8b8ac614b2caded280727cbbd76c022f`.
- service/admission.py `d72b8edafdf696d0bf07811da140744d302a363d92cd9aaf011ff2af1ad2d164`.
- tests/release/test_native_history.py `017c5b5e178eda84225cbca24eb04ec774a247a264c29f52b055f6001618bd70`.
- tests/release/test_native_activation.py `d6885afa050ca81d467b881225cfc3ed262069364ee29850c02646b6f6bd0d5e`.
- Unchanged lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

## Remaining limits

Fresh independent review and parent adoption remain required. No broader policy,
recovery, scheduler, readiness promotion or boot learning is introduced. Deferred
public authenticated lifespan/credentials, topology/mounts, onboarding and stopped
renewal are not implemented here. No reviewed arm64 wheelhouse exists; actual
Metal remains UNAVAILABLE / UNQUALIFIED. Actual dependency/install/import/ABI,
TLS/topology, process/FD/death/resource, model/MPS, audio/performance and release
qualification remain unrun and parent-owned; CPU/CUDA release rows remain mandatory.
Known legacy lib/tts digest RED is unchanged/unwaived; no digest regeneration or
legacy digest test. SOURCE/inert test receipts confer no hardware/release authority.
