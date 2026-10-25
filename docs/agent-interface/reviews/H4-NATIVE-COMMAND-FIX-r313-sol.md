# Native completed command progression — r313 Sol

2026-10-25. SOURCE candidate for fresh independent review; no self-adoption.
The two exact r312 F1 command mutations now refuse. Once-final selection:
**67 PASS / 49.131s / exit 0**. This is source and tiny inert evidence only.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-57da2e53`, clean at entry,
fresh parent `55ad442c9badb8eb02c749d6b4ea0a4146a2ca9d` without the implementation.
Required cherry-picks applied in order, without conflict:

- `12dc00ff23b3a8025b21326f75fe53e9bfa35169` → `1798e42341337eab4de38ea05b98d8726f816927`.
- `ab4bc72248f08c0b6880f1e99fe55238a3dbe259` → `da4d09ad69af8e25390f89c5fffcd43e66751219`.
- `01d76f03870459e87545b5f3f5dd0a95b4cff35a` → `719823a1fdbf279e811341bad9d65b14fff3c732`.
- `243a2fb7bcbaf1036fc889029b9939aa4ba25c66` → `f0e85fa68264df3f260cdeca0a370fc23b5d131f`.
- `4b0712934d57a90b9f50ef221de77a07a9ee35bf` → `b9e28ce7fedef1b51caa67157e4443afc8da9e56`.

Final SOURCE/tests commit: `f1c434eb5cc4e318bc6d9be09491a299b3208fab`.
This report follows in a documentation-only commit. Branch at source commit:
`t3/native-history-sequence-fix`.

Read full r312/r310 review reports and r311/r309 implementation reports in this
worktree; inspected actual Dispatcher `_launch`, `_tick`, `_terminal`,
`_service_stop`, DispatchStore command/complete/current, native runtime
observe/terminate/collect and the retained validator. Inspected both conftests,
native lifecycle/runtime fixture seams and inert spawn guard before execution.
No AGENTS.md found by repository discovery or checked ancestors. Bounded memory
registry search had no relevant hit; no memory-derived technical claim is used.

## Narrow F1 correction

Each completed dispatch must have the following finite command progression, in
durable sequence order, with every command in state `returned`:

`prepare, start, (observe | stop)*, observe, collect, ack+`

There is exactly one prepare, start and collect. At least one observe is required,
and the command immediately preceding collect must be observe. Nothing except
ACK may follow collect, and at least one ACK must exist. This closes the missing
observe and consecutive successful collect cases without assuming arbitrary
journal RPC idempotence. Existing closed operation/state namespace and detached
SQL foreign-key checks bind commands to actual dispatches; runtime, fixture,
acknowledged phase, ACK flag, exact completed outcome and terminal receipt checks
continue to bind the progression to the completed native dispatch.

Producer basis: `_tick` consumes returned prepare then launches start. Its polling
loop observes, with stop requests permitted before further observation. Only a
returned STOPPED observe launches collect. Successful collect calls `_terminal`,
which records outcome; the next journal operation is ACK. Repeated observations
and stop followed by further observation remain in the accepted language. The
ACK-only suffix remains replay-compatible, without allowing new observations or
collects after terminal collection. Existing all-returned policy still excludes
uncertain commands; this change does not broaden crash/recovery acceptance.

The journal does not retain observe response states. The reader therefore checks
the supported operation progression and existing independently retained/current
native terminal proofs; it does not fabricate a historical STOPPED state or
require nonexistent response details. Direct retained-reader collect RPCs still
do not create Dispatcher command rows.

## Regression evidence

Six new tests, all in NativeHistoryTests:

- Exact r312 deletion of every observe: refuses.
- Exact r312 ordered `prepare,start,observe,collect,collect,ack`: refuses.
- Nearby `prepare,start,observe,stop,collect,ack`: refuses.
- Nearby `prepare,start,observe,collect,observe,ack`: refuses.
- Missing ACK command despite acknowledged dispatch: refuses.
- Real Dispatcher produces three observations before one collect and ACK; ordinary
  NativeActivation reopen succeeds. The fixture delays its injected process-exit
  observation until the third poll; it does not rewrite a positive journal or
  fabricate historical observation rows/states.

Each refusal uses the existing helper: snapshot logical SQLite dump and all
spool/staging/assets file bytes after mutation, require startup exception, compare
unchanged evidence, require no extra ACK, and where runtime was constructed require
nonempty import_closed and empty import_completed. The exact two reproductions
reach this runtime import fence. Stop/observe and ACK replay are preserved by the
finite language and inspected producer transitions; no synthetic positive stop
or ACK replay history is presented as live execution evidence.

## Once-final test receipt

One invocation only, with no failed probe or test retry. Executed
`/private/tmp/pab-0c-venv/bin/python -I -B -` from this worktree. Explicit absolute
worktree and worktree/tests/service sys.path entries; printed and asserted Python
**3.14.7**, Clang 17.0.0 (not target Python 3.11). Used
defaultTestLoader.loadTestsFromNames, asserted count 67, and
TextTestRunner(verbosity=2, failfast=True), nonzero exit on failure.

Exact selection is the original r312 58 plus six new tests and its three live
Admission lifecycle regressions:

- tests.release.test_native_history.NativeHistoryTests — all 40.
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

**67 tests / 49.131s / OK / exit 0**. Tool session 21488; initial chunk 20b8bd,
intermediate f3c6d2/cd3729/335242/0af38b/2638cd, final chunk 1731f7.
Despite existing test names, these use injected TLS/process/backend observations,
not an actual child, listener or hardware. Direct unittest avoids pytest autouse
maintenance and server fixtures. `git diff --check` passed for the entire inherited
candidate plus this fix. Available disk observed 2,913,024 KiB, below 6 GiB;
only explicitly authorized tiny inert tests ran. No heavy ACTUAL gate was exercised
or waived. No network/listener, Docker, install/download, actual child, ML, crash,
HELD/STT or operational cleanup. Fixture-owned temporary teardown only.

SHA256 at final SOURCE:

- service/native_activation.py `c3366709336b0a768cb682bd70b28b8b8066a7d446e7b4f39e48b841a1aa4003`.
- tests/release/test_native_history.py `4f3586cb0ce34d01a7c1c6cd65e41827fe929da5c35b5e60e24be10768ecc83c`.
- Unchanged lib/tts.py `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.

## Remaining limits and parent gates

Fresh independent review and parent adoption remain required. Source edits after
the five inherited commits touch only native_activation.py and its history tests.
Frozen Authority, exact primitive identity, native terminal/publication/import and
resource proofs, Core preinitialization exclusion and Linux refusal remain intact.
No framework, repair, scheduler, boot learning or readiness promotion was added.
No root-checkout writes, merge, push, subdelegation or self-adoption occurred.

Legacy lib/tts digest RED remains unchanged/unwaived; no regeneration or legacy
digest test. No arm64 wheelhouse; Native Metal remains UNAVAILABLE / UNQUALIFIED.
Actual dependency/install/import/ABI, TLS/topology/denied peers, process/FD/death/
resource, hardware/model/MPS, generate/cancel/restart, clone lifetime, audio and
performance/release qualification remain unrun and parent-owned. CPU/CUDA release
rows remain mandatory. This source/inert receipt grants no operational, hardware
or release acceptance.
