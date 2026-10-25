# r342 Sol — independent adversarial review of r341

2026-10-25. **GO for the bounded same-boot, dispatched single-attempt negative-terminal SOURCE slice. No blocking finding established.** This is an independent review recommendation to the parent, not adoption. r334 delivery 2 remains incomplete; Metal remains **UNQUALIFIED** and actual release **NO-GO**.

## Authority, pins and evidence

Read-only candidate: `/Users/ben/.txcode/worktrees/py-audio-box/t3-37b3f731`. Base `63d47fc67d29d13d831665676996f3df367615e1`; HEAD `385976f8d7d8ebb406a6906488352ed6cf228dca`. Both resolved exactly; base is an ancestor. Candidate status was empty before review and after all executions. `git diff --check BASE HEAD` passed. No applicable AGENTS.md was found in candidate discovery or checked ancestors.

Read full r341, r334, PRODUCT and current NATIVE-PACKAGE; inspected the actual six-file diff, Core ownership/cancellation, Dispatcher terminal/outcome/ACK progression, Authority, native protocol/runtime/import accounting, supervisor decision/wait/receiver/conditioning, detached history validation, reconciliation consumers and public journey fixtures. Production code changes are confined to `service/native_activation.py`; no Core/scheduler change is concealed in the diff. The legacy reconciliation ledger is preserved CLOSED; r334's historical RED text is not a new current objection.

Sole lasting write is this root report. No candidate edits, commit, merge, push, delegation, network/listener, Docker, install/download, actual child, ML, crash, HELD/STT operation or operational cleanup. Only authorized tiny temporary filesystem/SQLite, in-process ASGI and injected TLS/process/backend observations ran. Inspected both conftests before direct unittest: root's maintenance guard is pytest-only; service's server fixture would bind/spawn, and was not invoked. The 6 GiB heavy gate remains unchanged and was not exercised or waived. Memory supplied evidence-boundary conventions only; technical conclusions use the pinned source and fresh receipts.

SHA256 values independently match r341:

| File | SHA256 |
| --- | --- |
| service/native_activation.py | `1c36e5fd6af669471b37b3c453682aea017ef2d094ef6ac55b03eadd19a611dd` |
| tests/release/test_native_history.py | `ffbc9df6191f4bd69e2de0c336d567bd8c225d961ada709e2ff39b71d15410ed` |
| tests/release/test_native_managed_clone.py | `90c346b99cd1996e9cb04f13da19a7fe8f106d9c51ac7a22717b951ea5c10104` |
| tests/service/test_native_lifecycle.py | `663bbcd1e99a79a6fef3480ec24f0762d6212f1738e6dc1983c10521e8c4a924` |
| docs/agent-interface/NATIVE-PACKAGE.md | `acd5357ab190587d96422db5a1d19eccff7a69fef224576213fcfd87cdd80ce1` |

## Adversarial assessment

1. **History joins remain closed.** Validation occurs under Core lifetime exclusion before schema initialization/recovery/scheduling. Exact sqlite_master identity, column order, detached SQL constraints/foreign keys, closed documents and finite row/byte budgets precede history use. Jobs, unique dispatches and reservations have equal coverage; publications and publication_work have equal coverage within jobs. Success requires its publication; negative history requires publication/work absent. Unknown commands, nonreturned commands, missing/duplicate collect, invalid progression and absent ACK refuse. The accepted sequence is prepare/start, observe/stop loop ending in observe, one collect, then only ACKs. This corroborates the producer without inventing unjournalled historical observe values.
2. **Terminal authority is narrow.** Frozen canonical Authority/context/native identity, request fingerprint, no retry link, one reaped attempt, exact durable outcome and acknowledged phase must match. Only completion state/stage/artifact fields are excluded from the immutable comparison. Negative jobs require `deadline_or_reaped`, empty artifact IDs, `timely:false` and exactly one process. The shared receipt validator checks native runtime/binding/deadline/boot, owned reap method, receiver closure, canonical process types and tuple shape. A read-only native collect must return the identical receipt and no proposal/output. Startup performs no ACK or allocation to manufacture proof. Hello must retain configured boot/trust, no quarantine and an integer launch count equal to finite dispatched history; enrollment brackets remain intact.
3. **The receipt producer closes the required lifecycle.** Supervisor signal/poll does not itself establish receipt authority. `finish_tick` requires a matching owned `wait` result, joined receiver with closed channel, and empty clone conditioning directory removed before terminal persistence. Uncertain persistence cannot grant a receipt. A zero-exit process without a valid result can legitimately produce a negative receipt; the policy correctly does not infer failure solely from exit code. Late/result-bearing or cancellation-race histories with retained output are conservatively unsupported here.
4. **References and uncertainty stay retained.** Negative spool membership is exactly snapshot plus input for clone; snapshot equals frozen bytes. Core reference ownership/metadata, exact length/hash/PCM and detached bytes are checked before either terminal branch. Negative staging must exist and be empty; import closure/reopened-import entries must be absent. Runtime accounting rejects partial/foreign spool evidence; the final read-only reconciliation scan rejects unknown namespaces, unavailable media and unresolved evidence. Reconciliation's acknowledged-dispatch shortcut is not relied on for negative proof: activation already checks those dispatches itself. Failure of any proof precedes in-memory discharge of success import closure.
5. **Existing success and public journeys are preserved.** Publication/intent/artifact canonical joins, actual hashes, pair/reference semantics and exact import markers remain on the success branch. Fresh public clone upload → prepare/results/save → generation → controlled failure or cancel → confirmed receipt/ACK → close/reopen preserves immutable version and old jobs/artifact reads, then admits a fresh successful compatible job without per-voice enrollment edits. Clean preset and managed-clone journeys, cancelled preparation reference retention and tampered preparation-pair refusal also passed independently.

No repair, auto-ACK, history deletion, capacity raise, reference reclamation, readiness promotion or alternate scheduler is introduced. Never-started cancellation remains explicitly unsupported; native retry remains disabled without RPC allocation. New-boot renewal, topology and dependency packaging are future delivery-2 parts, not blockers to this commissioned same-boot slice.

## Fresh execution receipts and reproduction

All three invocations used `/private/tmp/pab-0c-venv/bin/python -I -B -` from the candidate cwd. Printed interpreter: **CPython 3.14.7**, `3.14.7 (main, Aug  5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`; no Python 3.11 invocation. Each stdin driver explicitly prepended candidate root and candidate `tests/service` to sys.path, imported named classes, built a `unittest.TestSuite`, used `TextTestRunner(verbosity=2)` and exited nonzero unless `wasSuccessful()`. No whole-module loader was used, so imported classes and inherited tests outside the explicit selection were not accidentally run.

| Invocation | Receipt | Exit |
| --- | --- | --- |
| Exact r341 affected selection | **43 tests, 51.567 s, OK** | 0 |
| Focused regressions plus negative-history command/type/resource probes | **62 tests, 50.007 s, OK** | 0 |
| Independently authored negative-history mutations | **20 tests, 19.434 s, OK** | 0 |

Total **125 PASS**, zero failures/errors/skips. Each selection ran once. These are SOURCE/inert receipts, not hardware or installed process proof.

Exact 43 reproduction selection:

```python
from tests.release.test_native_history import NativeTerminalHistoryTests, NativeCancelledHistoryTests
from tests.release.test_native_managed_clone import ManagedCloneTests
from tests.service.test_native_lifecycle import NativeLifecycleTests
l = unittest.TestLoader()
s = unittest.TestSuite([
    l.loadTestsFromTestCase(NativeTerminalHistoryTests),
    l.loadTestsFromTestCase(NativeCancelledHistoryTests),
    ManagedCloneTests('test_saved_clone_cancel_reopen_and_new_job'),
    ManagedCloneTests('test_saved_clone_failure_reopen_and_new_job'),
    NativeLifecycleTests('test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job'),
])
```

The private lifecycle method explicitly reenrolls its fake new boot and original template store. Its PASS does not implement installed renewal. The final one-line fixture correction refreshes `_preset_templates`; the public terminal fixtures use fresh fake handles and correctly count preparation, initial generation, negative execution and subsequent generation. These fixes are coherent with the fixture model.

The 62 selection consists of these four explicit methods, all four `PublicJobControlTests` methods and all fourteen `InstallOwnership` methods, plus twenty explicitly selected existing refusal methods applied separately to each negative class (40 probes):

```text
NativeOnboardingTests.test_fresh_public_registration_generation_artifact_restart_replay
ManagedCloneTests.test_public_upload_prepare_results_save_generate_reopen_replay
ManagedCloneTests.test_reopen_refuses_tampered_preparation_pair_without_repair
ManagedCloneTests.test_cancelled_preparation_retains_reference_and_reopens
```

For exact reproduction of the 40 additional probes, import `NativeHistoryTests`, create an in-memory subclass of each negative class with the following methods copied from `NativeHistoryTests`, and add only `Class(name)` for each listed name, rather than loading that subclass wholesale:

```text
test_wrong_native_terminal_receipt_refuses
test_unknown_staging_evidence_refuses
test_wrong_request_fingerprint_refuses
test_unknown_returned_command_refuses
test_unknown_command_state_refuses
test_missing_start_command_refuses
test_collect_without_observe_refuses
test_duplicate_returned_collect_refuses
test_stop_between_terminal_observe_and_collect_refuses
test_observe_after_collect_refuses
test_command_order_refuses
test_job_generation_float_refuses
test_outcome_generation_float_refuses
test_retained_float_dimension_refuses
test_retained_bool_dimension_refuses
test_retained_zero_dimension_refuses
test_retained_overflow_dimension_refuses
test_retained_nonfinite_dimension_refuses
test_retained_positive_wrong_reservation_refuses
test_retained_unknown_dimension_refuses
```

The independent 20 selection used the same explicit in-memory subclass technique, ten cases for each negative class. Each case invoked the existing `refusal` helper (or `change_document`, which delegates to it). Reproductions are the following mutations after `prepare()` closes its owner:

| Case | Mutation |
| --- | --- |
| launch | Fake hello `launches += 1` |
| launch_bool | Fake hello `launches = True` |
| quarantine | Fake hello `quarantined = True` |
| remote | Fake collect receipt's sole process `pid = 124`, leaving local receipt unchanged |
| proposal | Fake collect `proposal = {}` |
| output | Fake collect `output = {bytes:46, sha256:'a'*64}` |
| outcome | Change only dispatch outcome state to the other negative terminal state |
| stage | Change only stored job stage to `verification_failed` |
| spool | Write `b'RETAIN'` to retained binding directory's `output` |
| inventory | Write `b'RETAIN'` to owner root's `unknown-evidence` |

Fake RPC response mutation is installed through `self.f.mutate(q,v)`, editing `v['value']` only for the selected op and returning v. SQL document mutations use `change_document('dispatches','outcome',...)` or `change_document('jobs','job',...)`. Filesystem mutations use `refusal(lambda c: ...)`. Every case refused reopening. The helper compares retained spool/staging/Core asset file bytes, SQL dumps and ACK counts before/after rejection. The independent root-unknown marker lies outside that helper's byte inventory; source inspection establishes no deletion path, and the probe establishes startup refusal, not an additional comprehensive filesystem-preservation oracle. No refusal was recast as success.

r341's historical initial **174 tests / 181.146 s / 31 failures + 1 error (142 passed)** remains a failed receipt, as documented. It was not rerun wholesale or rewritten. Fresh 43 PASS verifies the final affected selection independently; focused additional coverage above is separately counted.

## Parent-owned remainder

Parent decides adoption/integration of the pinned candidate. This review does not merge or self-adopt it. Delivery 2 still requires stopped/reconciled new-boot and trust renewal, bounded capacity renewal with retained history, Docker/native topology and exact mount/peer validation, and real arm64 dependency/ABI packaging. The private new-boot fixture is not that implementation. Never-started cancellation, native retry, terminal output/intent uncertainty and crash recovery are not newly supported.

Separately authorized actual packaged installation, authenticated speech/saved-clone reuse, cancellation/reap/restart/renewal, old artifact verification, listening, resource/performance and CPU/CUDA/Metal qualification remain open. No cleanup, cap bypass, history deletion or promotion remedies these gates. Preserve CLOSED legacy reconciliation and the deferred UI/public-STT boundary. **Bounded SOURCE GO; delivery 2 incomplete; Metal UNQUALIFIED; release NO-GO.**
