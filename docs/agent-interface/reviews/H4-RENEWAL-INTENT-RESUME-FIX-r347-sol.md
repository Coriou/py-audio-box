# r347 Sol — exact completed intent-stage resume repair

2026-10-25. SOURCE repair candidate for fresh independent review, not Lot A
adoption or operator authorization. Lot B public voice changes remain unimplemented
in this lane. Legacy reconciliation CLOSED; Metal UNQUALIFIED; release NO-GO.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-b4756d64`.
Initial clean HEAD `e8362953b172489fca12f005d7c3ae524bd672de` lacked the
implementation. Cherry-picked, in the requested order:

- `eb2ef8bf25874ebd306d15bb4f59301dea8c57c8` -> local
  `07fa4de0de5ec22638691a23e85e43e63133d3bf`.
- `08e03baed189939cfb0f099a74256d703a65beb2` -> local
  `029580f972069fde3b2fe1de2a8bd18b17022576` (repair parent).

Read the full root r346 review and exact probe, normative r343 contract and
r344/r345 reports. Inspected actual namespace/begin/selected_source/tip/publish,
Core maintenance ownership, startup/activation/reconciliation consumers and all
retire/seal/provision/commit entry paths. No AGENTS.md found in worktree discovery
or checked ancestors. Bounded memory registry search returned no relevant hits;
no memory-derived technical authority was used.

Only this worktree's source/tests/reports and temporary inert drivers/logs were
written. No root checkout writes, merge/push, subdelegation or self-adoption.
Both conftests were inspected before direct unittest; pytest listener/process
fixtures were not activated. Tiny files/SQLite, fake TLS/process/backend
observations and fixture fsync/rename faults only. No actual host crash,
network/listener, Docker, install/download, actual child, ML, HELD/STT or
operational cleanup. Normal fixture teardown is unchanged. The 6 GiB heavy gate
was neither exercised nor waived.

## Repair and systematic phase assessment

The completed stage is visible only to explicit `begin`, inside the existing
Core registry/control.lock maintenance lease and read-only SQL transaction.
The private `_begin` namespace mode requires explicit resume, validates the
closed intent and ordinary chain dependencies and rejects another file at the
same or a later epoch alongside the staged intent. It presents that intent only
as a candidate for exact request comparison. Normal namespace/tip/selected_source,
Core startup and activation retain their unconditional stage refusal.

Before publication, begin compares request ID, exact supplied successor snapshot,
predecessor record, live owner path/device/inode/UID, predecessor enrollment/trust
snapshot, closed compatibility, current stopped membership and stopped SQL
fingerprint. The existing publisher compares canonical bytes, re-fsyncs the
stage, renames only into an absent final, then fsyncs final and directory. Existing
final acknowledgement replay also re-fsyncs final and directory. Partial, changed,
foreign, conflicting or forked evidence is retained; no deletion, rollback or
startup repair is introduced. Committed begin replay still permits later legitimate
SQL extension; pending intent replay must match the stopped fingerprint.

Phase-entry inspection, beyond publisher helpers:

- Begin was the missing route: omitted stage -> no replay -> selected_source ->
  normal tip -> stage refusal. It now reaches the explicit publisher after binding.
- Retire reads its explicit final intent and independently validates stopped host
  evidence under native.lock. Its finite retired namespace admits marker/export
  stage names only for exact publication comparison; it does not call tip.
- Seal reads explicit intent/export, verifies the final intent and stopped bracket,
  and supplies validated pending RetiredEvidence to EpochEvidence. A staged export
  need not be returned by namespace: pending evidence supplies that exact proof.
  Export and seal publishers are reached explicitly after full history checks.
- Provision calls seal and publishes exact cumulative predecessor authorization;
  its new host namespace permits only predecessor/final stage. No startup selector
  precedes that publisher.
- Commit reads explicit intent/export, uses maintenance namespace resume and
  pending offline evidence, republishes dependencies and enrollment, then checks
  snapshots/hello before commit publication. A staged enrollment/commit does not
  need to become an activation tip. Existing integration phase tests exercise
  enrollment/commit rename and seal/commit final-fsync acknowledgement loss.
- Normal activation, reconciliation and preallocation continue to use the normal
  selector; they cannot repair a stage or acquire authority from this candidate.

No history/chain/capacity, owner exclusion, retirement, hello, ACK, lock or public
voice behavior was relaxed. No scheduler, cap reset, automatic ACK or rollback
was added. The r345 report's trailing blank EOF was removed. Its historical
statement that diff-check passed remains historical; r346 correctly identified
that the committed original diff was not clean.

## Reproduction and final execution

Pre-fix reproduction: executed the exact r346 rename boundary against the
cherry-picked pre-repair module loaded from `git show 029580f:service/native_renewal.py`
into an isolated in-memory module. One explicit Probe instance, no inherited
discovery. The complete fsynced stage remained byte-identical and final absent;
identical retry raised `native_renewal_partial_or_conflicting` as expected.
**1 test, 1.168 s, OK** means successful defect reproduction, not acceptance.
Driver `/tmp/r347-probe.py`, SHA-256
`af220db9ca3e9d4f23a80f3c2c87186fba630564d362d3526fd3dd22402e066e`.

Final driver retains the complete r346 current 110 selection, with no expensive
rows omitted, and adds eight directly defined RenewalCompletion cases:

- first/later intent rename interruption;
- first/later complete-stage fsync acknowledgement interruption;
- first/later visible-final fsync acknowledgement interruption;
- first/later staged intent changed-input retention tests.

Each positive case exercises actual begin/maintenance/selection/publication paths,
checks normal selector and Core refusal before resume, compares retained bytes,
and observes explicit final and parent fsync. Negative cases compare every owner
file byte before/after refusal for changed request, supplied boot, staged payload,
partial JSON, foreign request, forked epoch and changed stopped fingerprint.
Fixture inputs are explicitly restored by the test to check the original exact
request still resumes; production code does not restore or delete evidence.

Invocation from this exact worktree:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B /tmp/r347_tests.py > /tmp/r347-final-tests.log 2>&1
```

The driver inserts this cwd and explicit `tests/service` path. It preserves all
six broad selected classes and all explicit negative/failed history, cancelled/
failed two-renewal, managed-clone, native lifecycle, supervisor and package rows
from `/tmp/r345_tests.py`, selecting completion methods defined directly on the
class without inherited duplication. Interpreter printed CPython **3.14.7**,
Clang 17.0.0, not 3.11. Driver SHA-256:
`b5af6473149ec6452adfc48a8ee476d23a131968bc21f8d9f3ae531d0e8d2a9a`.

Exact existing selection: all methods of RenewalFilesTests, RenewalHistoryTests,
NativeTerminalHistoryTests, NativeCancelledHistoryTests, InstallOwnership and
NativeActivationTests; directly defined methods of RenewalCompletionTests;
`test_stopped_offline_success_seal_and_closed_commit` from both
RenewalNegativeHistoryTests and RenewalFailedHistoryTests;
`test_two_epochs_offline_old_history_new_allocation_and_stale_path` from both
RenewalCompletionCancelledTests and RenewalCompletionFailedTests; and:

- ManagedCloneTests: `test_saved_clone_cancel_reopen_and_new_job`,
  `test_saved_clone_failure_reopen_and_new_job`,
  `test_reopen_refuses_tampered_preparation_pair_without_repair`.
- NativeLifecycleTests:
  `test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job`.
- NativeSupervisorTests: `test_prepare_replay_tuple_session_and_stop_ack_restart`,
  `test_restart_unknown_never_relaunch`,
  `test_launch_intent_before_factory_replay_wait_and_receiver`,
  `test_actual_child_result_terminal_output_and_oldboot_replay`,
  `test_storage_gate_and_quota_refuse_before_reservation`.
- NativePackageTests: `test_archive_deterministic_bytes`,
  `test_native_configuration_does_not_enter_linux_policy`.

Methods named actual_child use fake channels/handles/backend observations.
Archive installation/subprocess-help and all real installed/hardware rows remain
unrun; no existing row from the current 110 selection was omitted.

Final coherent selection executed **once after source/test changes**:
**118 tests in 156.057 s, OK, exit 0**, no failures/errors/skips. No source or test
changes followed this run. Raw log `/tmp/r347-final-tests.log`, SHA-256
`fc6cb1a9ba7bd962550dcadf74b2caf8049ef76a5909a60af406969b5ae6f2e6`.
The separate pre-fix probe is a defect receipt, not a 119th acceptance case.
Both `git diff --check` and `git diff --check e8362953b172489fca12f005d7c3ae524bd672de`
passed after the repair, including the r345 EOF correction.

## Changed source fingerprints and handoff

- `service/native_renewal.py` SHA-256:
  `a8d2ab284e060cd8258551f65d19bf75af97e26c3f96dcdd552ed872f74d4449`.
- `tests/release/test_native_renewal.py` SHA-256:
  `02c22aabca678eb8df04b986b7c479ceffff108529a2ce8392443eefcebf92e6`.

The repair commit contains these files, this report and only the r345 report's
trailing-blank correction; its exact parent is recorded above. All other imported
implementation bytes remain those of the two pinned cherry-picks.

Fresh independent review of this committed repair is required before parent
Lot A adoption. No self-acceptance, merge or operational authorization follows
from 118 inert tests. Lot B public origin/voice journey is untouched. Installed
TLS/process/topology, package/ABI, retained real artifacts, audio/performance and
CPU/CUDA/Metal qualification remain separately authorized and UNRUN here.
Legacy CLOSED; Metal UNQUALIFIED; release NO-GO.
