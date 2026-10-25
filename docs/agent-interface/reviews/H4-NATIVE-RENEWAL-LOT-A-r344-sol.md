# r344 Sol — Lot A preparation candidate; end-to-end exit incomplete

2026-10-25. **NO-GO for Lot A adoption or operator use.** This commit implements a
bounded stopped-intent/retirement/offline-history preparation slice, not the
commissioned complete stopped epoch transaction. The successor seam explicitly
refuses rather than claiming authority. Do not begin/retire a real installation
with this candidate: there is no implemented route back to admission. Lot A must
be completed and independently reviewed before Lot B. Metal remains
**UNQUALIFIED**, release **NO-GO**; legacy reconciliation remains CLOSED.

## Authority and source

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-e321f26b`, clean initial
HEAD `374855445b0a4ec8ca57de483bf4339ee7a09236`. Its parent is the r343 normative
source pin `974d1e659ac6480f93a5dac591dfff8ce0243b41`. Read the entire r343 contract
before design/code, r341/r342/r334 and NATIVE-PACKAGE; inspected current
activation/history, enrollment/configuration, Core lifetime exclusion, native
journal/runtime, reconciliation, owned media and affected fixtures. No applicable
AGENTS.md was found in worktree discovery or checked ancestors. A bounded memory
registry search had no relevant hits; no memory-derived technical fact is used.

No root checkout writes, merge, push, subdelegation, independent review or
self-adoption. No network/listener, Docker, install/download, actual child/ML,
crash experiment, HELD/STT operation or operational cleanup. Tiny fixture
filesystem/SQLite and injected TLS/process/backend observations only; write,
fsync and rename faults are fixture injections. The 6 GiB heavy gate was neither
exercised nor waived. Both conftests were inspected before direct unittest; their
pytest listener/subprocess fixtures were not invoked. Test methods containing
“actual_child” exercise the existing child codec via fake channels/backend, not
an actual OS child.

## Implemented preparation

- `native_maintenance` reserves the existing Core process/thread registry slot,
  binds owner inode identity and takes existing `control.lock`. It opens existing
  SQLite with `mode=ro` and query-only SQL, without constructing another Core,
  initializing schema, recovery or scheduling. Existing registry/cleanup logic
  remains responsible for failed-close retention.
- `begin` compares canonical fixed-version compatibility, binds explicit supplied
  successor input snapshot, predecessor enrollment, stopped SQL fingerprint and
  exact dispatched membership. The immutable epoch-zero enrollment is untouched.
  Intent publication blocks native startup and generic Core startup before schema
  recovery; a stale enrollment path cannot bypass this preparation namespace.
- Host `retire` takes existing `native.lock`, uses the extracted shared journal
  validator without creating Journal, requires exact Core membership, terminal
  ACKs, one owned process/receiver receipt, exact retained files/output hashes and
  proposal validation. It writes an exclusive retirement marker and bounded
  export without secret/media duplication. Original host bytes remain retained.
  Partial marker publication blocks Journal construction, save, dispatch,
  prepare and direct start. A live native lock refuses without a marker.
- `RetiredEvidence` is an explicit validated private offline source. There is no
  generic collect callback, skip flag or fallback after live proof failure.
  `completed_history` retains its complete r341 SQL/schema/Authority/receipt/
  publication/reservation/reference/import/media predicates. Only its collect
  source is explicitly selected for this stopped single-epoch preparation.
  Old private key/TLS availability is not needed by offline sealing.
- Preparation seal/export are accepted only through an exact finite
  `native-renewal` namespace in read-only reconciliation. Arbitrary descendants,
  unknown/staging files and missing dependencies refuse. The preparation seal
  does not grant admission or replace media/history. Its stopped SQL fingerprint
  is a temporary transaction bracket, not a permanent whole-SQLite seal.
- Deterministic transaction-owned `.stage` publication uses exclusive create,
  write/fsync, rename and parent fsync. An exact completed stage resumes only on
  explicit request. Exact visible finals are re-fsynced on explicit resume.
  Conflicting/truncated/foreign evidence is retained and refused.
- The private installed command parser has explicit renewal-begin/retire/commit
  entries, and the new module is in the distribution allowlist. There is no
  public HTTP/MCP provisioning route, new framework or scheduler. Operator docs
  explicitly mark the commands unavailable for installation use.

## Exact missing work and refusal boundary

`commit` currently validates/persists offline preparation evidence and then raises
`native_renewal_successor_unavailable`. It does **not** publish an activation
commit or authorize a new owner. This is an explicit implementation gap, not a
completed Lot A or a user design question. In particular, these required pieces
are absent:

1. Immutable successor envelope/final commit as the sole activation point,
   dependency bracketing, exact new pinned hello (including zero epoch launches),
   supplied boot/trust comparison and second-boot refusal around commit.
2. Append-only numbered chain through up to eight total epochs, unique committed
   tip selection, committed namespace reconciliation, historical/current
   partitioning and normal reopening with unavailable old TLS. The present
   preparation format recognizes only epoch 001 and never selects an epoch.
3. Cumulative host registration/journal/bytes carried into a successor root and
   independently enforced there, cumulative control preallocation admission and
   prospective compatible-generation headroom. The existing counters are not
   reset or raised; no successor is admitted. Current preparation checks only
   stopped history count/rows/bytes, template/allocation count and metadata disk
   headroom. Export charges are preparatory data, not trusted future capacity.
4. Complete immutable per-binding historical seal revalidation across later
   legitimate catalogue/cursor changes. The current full-history validation is
   stopped single-epoch evidence only. It is not an extensible permanent seal.
5. Trusted successor-origin mapping format and host provisioning integration.
   The compatibility helper accepts only unchanged fixed voice/version semantics
   with the closed native tuple/trust/timestamp differences. Lot B's stable
   origin derivation and public existing-preset/clone generation are not present.
6. Full r343 two-renewal/interruption/capacity/new-hello/end-to-end fixtures. There
   is no public new-boot renewal journey receipt and no supported installed
   renewal. Fresh independent Sol review has not occurred and is parent-owned.

The commissioned coherent end-to-end Lot A is therefore **not delivered** by
this commit. No actual release or installed lifecycle claim follows from the
preparation tests. Do not advance directly to Lot B or adopt these commands.

## Execution receipts

All drivers used `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit worktree
`sys.path.insert`, `unittest.TestSuite` and `TextTestRunner(verbosity=2)`.
Interpreter printed **CPython 3.14.7**, Clang 17.0.0, never Python 3.11.

- Initial two-test smoke: **2 tests, 1.744 s, 1 error**. Offline success sealing
  passed. The standalone temporary owner test used `/var/...`, an unresolved
  symlink alias; the intentional no-follow path check refused. The fixture now
  resolves its temporary root explicitly. The failed receipt is retained here.
- Finite affected selection, once: **95 tests, 67.761 s, 1 error** (94 passed).
  Only the symlink test assertion was wrong: no-follow file open raised
  `OSError(ELOOP)`, not `Refused`. Production refusal was correct and unchanged.
- Corrected assertion only, once: **1 test, 0.003 s, OK**. It now asserts exact
  ELOOP, retained symlink and untouched source bytes, then checks hardlink refusal.
  No broad selection was repeated; there is no claim of a single 95/95 green run.

The 95-test selection loaded exactly these classes (no whole-module import
loader): `RenewalFilesTests`, `RenewalHistoryTests`,
`NativeTerminalHistoryTests`, `NativeCancelledHistoryTests`, `InstallOwnership`,
`NativeActivationTests`. It then explicitly selected:

- `RenewalNegativeHistoryTests.test_stopped_offline_success_seal_and_closed_commit`
- `RenewalFailedHistoryTests.test_stopped_offline_success_seal_and_closed_commit`
- `ManagedCloneTests.test_saved_clone_cancel_reopen_and_new_job`
- `ManagedCloneTests.test_saved_clone_failure_reopen_and_new_job`
- `ManagedCloneTests.test_reopen_refuses_tampered_preparation_pair_without_repair`
- `NativeLifecycleTests.test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job`
- `NativeSupervisorTests.test_prepare_replay_tuple_session_and_stop_ack_restart`
- `NativeSupervisorTests.test_restart_unknown_never_relaunch`
- `NativeSupervisorTests.test_launch_intent_before_factory_replay_wait_and_receiver`
- `NativeSupervisorTests.test_actual_child_result_terminal_output_and_oldboot_replay`
- `NativeSupervisorTests.test_storage_gate_and_quota_refuse_before_reservation`
- `NativePackageTests.test_archive_deterministic_bytes`
- `NativePackageTests.test_native_configuration_does_not_enter_linux_policy`

The final narrow invocation selected only
`RenewalFilesTests.test_symlink_hardlink_and_duplicate_keys`.

New fixtures establish offline success/cancel/failure proofs with old TLS/key
construction unavailable, exact-request replay, live native lock refusal, changed
request/ACK/membership/media/unknown evidence refusals, read-only shared Core
ownership, stale startup denial, partial retirement denial, canonical primitive
and immutable-profile/config comparisons, and injected write/fsync/rename states.
Existing same-boot negative history, success/pair public clone and Linux ownership
fences passed in the finite selection. Archive determinism passed; no install or
archive subprocess-help test was run. `git diff --check` passed.

## Final source pins

Hashes below describe the source after the assertion correction. The report
itself is excluded to avoid a self-referential hash. The commit records the full
candidate; its parent is the exact initial HEAD above.

- `service/native_renewal.py`: `072ccea0ce7be4177faec8d72f5766a773b51db1be74ff82c809c56a491210e9`
- `service/core.py`: `6d9c85a1edb8bb33cfdb88c0b69334cf531346739bf351b548b6b8ee4f0df820`
- `service/native_activation.py`: `d4be207fa8963120c6baefcb85d6c5adc857d2da37676787488ea039c3116cd4`
- `service/native_launch.py`: `1284bb0b67168450efc5efb7aab120c5bfd32a9c35c6039693f99666ea29f994`
- `service/native_package.py`: `f2ac110851a448bd676327eea74e2e96afd3f754572d90aef8698895d1f05df1`
- `service/reconciliation.py`: `3c02c90e66b416d0057c5210cf955c51ae3f2b30ad7ac79cfcf077e6f88d3850`
- `worker/native_supervisor.py`: `802add7844098c4b402df78ff1800a8877a40fbd87b7c53c646e5dcad142477f`
- `native-dist-manifest.json`: `70c931cb7c7e3c14a42ef1324f587ebb0cc39b8422723db2e7ccecbfe8cf2b00`
- `tests/release/test_native_renewal.py`: `19631c483cae59f86305ccb9ec089cf22d0c21c3f8acf51b5b77a6616a24fd78`
- `docs/agent-interface/NATIVE-PACKAGE.md`: `ee1b6e7843b3b0afea4af3ad8ff213dff14d1146e4d700b261b114dec08be140`
