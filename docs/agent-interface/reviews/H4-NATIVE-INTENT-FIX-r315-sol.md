# Native publication intent identity — r315 Sol

2026-10-25. SOURCE candidate for fresh independent review; no self-adoption.
The exact r314 F1 intent-generation substitution now refuses. Final relevant
selection ran once: **71 PASS / 51.905s / exit 0**. SOURCE/tiny inert evidence only.

## Authority and exact pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-79171b09`, branch
`t3/native-intent-canonical-identity`, clean at entry. Fresh parent
`7799c5204d5ddbbb60a585413c9950388e69f91c` lacked the implementation.
Verified the requested exclusive range
`55ad442c9badb8eb02c749d6b4ea0a4146a2ca9d..77dc43f3d6fbd7ac970629de61d8a1c7af26abf5`
contains exactly seven chronological commits matching r313/r314's chain.
Cherry-picked all seven in order, without conflict:

- `1798e42341337eab4de38ea05b98d8726f816927` -> `5c700d13163b5b0c183814c25808c04ea00ddc92`.
- `da4d09ad69af8e25390f89c5fffcd43e66751219` -> `35763ea2c2ed004f2a41391c286a63c61337f8df`.
- `719823a1fdbf279e811341bad9d65b14fff3c732` -> `cd436255da146bd02bbf6a873f029f67cc7afcb8`.
- `f0e85fa68264df3f260cdeca0a370fc23b5d131f` -> `395c709eddcca52859340ce3caa03fc06a0bc6d9`.
- `b9e28ce7fedef1b51caa67157e4443afc8da9e56` -> `daaf78c4849b191dda6ece8597cc27ec5eb1fc2e`.
- `f1c434eb5cc4e318bc6d9be09491a299b3208fab` -> `62065c8d430fa047aff3fb539d8ecb3432edfe67`.
- `77dc43f3d6fbd7ac970629de61d8a1c7af26abf5` -> `73d4d95a52313b3717c8da01ac3968fe36f9be9e`.

Final SOURCE/tests commit: `eaeaa42eb6acdaad1d40c3d996e17eb157d604fd`.
This report follows in a documentation-only commit.

Read full r314, r313, r312, r311, r310 and r309 reports. Inspected native retained
reader, actual Core publication producer/barrier/commit, publication_intent and
publication_record, detached document validator, Reconciliation publication work
and job-owner consumers, publication oracle, Core preinitialization hook and
native fixture seams. No AGENTS.md found in worktree discovery or checked ancestors.
Bounded memory registry search supplied no relevant native-history guidance; no
memory-derived technical conclusion is used.

## Narrow correction and same-class joins

The shared persistence helper deliberately remains unchanged. After its existing
validation, the native reader canonically corroborates each intent artifact with
the retained publication artifact. A versioned intent's manifest also canonically
matches the retained publication manifest. This is read-only corroboration; it
cannot normalize, repair or overwrite a retained copy.

Inspection of all publication intent/record/job joins used by this reader found
the same primitive-equality class in oracle.publication's artifact-to-frozen-job,
identity/revision-to-plan and manifest-to-artifact owner comparisons. Merely making
all artifact copies equal would leave coordinated float substitutions accepted
at those joins. The native reader therefore canonically compares manifest owner
fields to the frozen Authority owner tuple, and the full artifact to the existing
Core producer's representation: manifest artifact ID/output IDs/hash, frozen job
owner tuple, frozen resolved identity and integer project_revision=0. The supported
native non-clone preset/design producer emits revision zero; no new operation or
publication shape is admitted. Existing schemas/oracle checks still run.

The remaining joins use strings/IDs/hashes, explicit integer byte-length validation,
canonical artifact metadata, canonical current/retained proposal (including complete
manifest/message) and output, canonical frozen context/job/outcome, or raw immutable
snapshot/import bytes. Versioned prepared-reference pairs remain outside the
non-clone/single-output native slice; the canonical manifest-copy check does not
expand their acceptance. No shared Linux helper or refusal policy changed.

Four new tests mutate completed fixtures before ordinary startup:

- Exact r314 reproduction: only publication_work.binding generation int -> float.
- Neighbor intent-only project_revision int -> float.
- Generation int -> float in publication artifact, artifacts metadata and intent
  together, isolating the join to the unchanged frozen job.
- Project revision int -> float in those same three copies, isolating the plan/
  producer identity join.

All use the retained refusal helper: compare post-mutation logical SQLite dump
and spool/staging/assets bytes against refused startup, require no extra ACK,
and require pending import closure with no import completion when runtime exists.
Clean completed reopen twice, exact artifact access and production-key replay
remain positive with one inference fixture call. Zero-job reopen and actual
producer command progression remain in the unchanged relevant selection.

Core lifetime exclusion and schema-before-initialization ordering, writer-thread
ownership, frozen Authority/resource proof, native terminal receipt, same configured
boot/trust, current/retained collect and import-intent proof remain mandatory.
No repair, ACK, readiness promotion, boot learning or history rewrite was added.

## Once-final finite receipt

Inspected tests/conftest.py and tests/service/conftest.py before execution. Direct
unittest avoids pytest autouse maintenance and socket/uvicorn server fixtures.
Inspected fake TLS, owned process/backend observations and inert guard seams.

Executed once from this worktree using
`/private/tmp/pab-0c-venv/bin/python -I -B -`, with explicit absolute worktree and
worktree/tests/service sys.path entries. Printed/asserted **Python 3.14.7 / Clang
17.0.0**, not target Python 3.11. Used defaultTestLoader.loadTestsFromNames,
asserted count 71, TextTestRunner(verbosity=2, failfast=True), nonzero on failure.
Exact selection is r314's 67 plus the four tests above:

- tests.release.test_native_history.NativeHistoryTests — all 44.
- tests.release.test_native_activation.NativeActivationTests — all 8.
- tests.release.test_native_enrollment.NativeEnrollmentTests — all 10.
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt.
- tests.service.test_installed_activation.ActivationTests.test_preowner_refusal_never_constructs_core.
- Same ActivationTests.test_post_core_private_reader_under_lifetime_lock.
- Same ActivationTests.test_changed_post_acquire_hello_refuses_and_releases_core.
- Same ActivationTests.test_changed_post_acquire_second_hello_refuses.
- tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change.
- tests.service.test_native_lifecycle.NativeLifecycleTests.test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job.
- Same NativeLifecycleTests.test_stale_future_native_registration_and_historical_decode.
- Same NativeLifecycleTests.test_cancel_retains_reservation_until_actual_owned_receipt.

**71 tests / 51.905s / OK / exit 0**, session 23039; chunks
852ce4/e79b29/4ccf0e/cafaa3/973c6d/92c624. No failed test invocation, separate
probe, retry or broad unchanged repetition. The exact reproduction is a refusal
regression, not a new pre-fix acceptance probe. git diff --check passed over the
entire candidate relative to fresh parent.

SHA256 at final SOURCE:

- service/native_activation.py `4549ba923ef4667ec5321cc6a8e13815c8657f120255c23c41f18dd8035d0a5e`.
- tests/release/test_native_history.py `17906dc923a2d6157ac5a69f2be0779dfbc418cbfb66aca7b028bd762ad90f43`.
- Unchanged lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

Observed available disk 2,889,716 KiB, below 6 GiB. Only authorized tiny inert
TLS/filesystem/audio/ZIP/process observations ran; historical test names do not
mean actual children executed. No network/listener, Docker, install/download,
actual child, ML, crash, HELD/STT or operational cleanup. Fixture-owned temporary
teardown only. Heavy ACTUAL admission was neither exercised nor waived.

## Remaining parent-owned limits

Fresh independent review and parent adoption remain required. No root writes,
merge, push, subdelegation or self-adoption. Accepted history remains fewer than
64 clean succeeded single-attempt non-clone jobs, 2048 SQL rows/16 MiB retained
scalar budget, same immutable configured identity/boot/trust and existing runtime
limits. These are materialization budgets, not hostile-database scan-time bounds;
dedicated account/immutable inputs/no hostile same-UID writers remain premises.
Failed/cancelled/retried/clone/crash/unfinished histories remain unsupported.

Legacy lib/tts digest RED remains unchanged/unwaived; no regeneration or legacy
digest test. No arm64 wheelhouse; Native Metal remains UNAVAILABLE / UNQUALIFIED.
Actual dependency/install/import/ABI, TLS/topology/denied peers, process/FD/death/
resource, hardware/model/MPS, generate/cancel/restart, clone lifetime, audio/
performance and release qualification remain unrun and parent-owned. CPU/CUDA
release rows remain mandatory. Public authenticated lifespan/credentials,
topology/mounts, fresh onboarding and stopped boot/trust renewal remain separate.
This SOURCE/inert receipt grants no operational, hardware or release acceptance.
