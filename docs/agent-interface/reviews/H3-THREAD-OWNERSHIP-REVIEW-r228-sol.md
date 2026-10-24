# H3 r228 Sol — independent round 3 activation source review

2026-10-24. **GO for bounded source adoption of the pinned candidate below.** No remaining blocking source finding in this review. R226-1 is corrected; R224-2 remains closed. Parent owns adoption. This is not installed-runtime, shutdown/reap, packaging, audible speech or release acceptance; those already separate qualifications remain pending.

## Pins and scope

Reviewed clean `/private/tmp/pab-r227-sol`, branch `fix/r227-thread-ownership`, HEAD documentation **`d5bbabe7c2827e60958fed8e7267cf9bbcd24eed`**. Verified the exact direct-parent chain:

- Base `ff42548f54ca272bc763675cebba6e7a417fa692`.
- Carry `c96330142d7d3ef84c9481c81e328cb6d8b38d8a`, whose service/tests/worker trees have zero difference from original `d692a3d715dc8d5cca6c2fc21f17150bd3f8d44d`.
- Carry `4b2d9731497388acbf595f4b4ad86c7750e74bf8`, whose service/tests/worker trees have zero difference from original `20f401833bb725db0dc15f2f683d27186577a97f`.
- Correction `7dfbd63953f620571e9fa1c8d546c1b97c5471a0`, changing only `service/entrypoint.py` and `tests/service/test_installed_activation.py`.
- Documentation HEAD adds only the r227 report. Dispatcher has zero difference from the carried r225 source.

Read root r198, including activation and shutdown requirements, r223, r224, r225, r226 NO-GO and r227; inspected the correction, carried activation/API/credential/context/worker changes, Dispatcher gate/close paths, regression fixtures and both conftests. No applicable AGENTS.md found in scoped/ancestor checks. Bounded memory registry search returned no relevant hit and supplied no premise. No skill, delegation, source edit, commit or adoption. Sole persistent write is this root report; pre-existing unrelated evidence is untouched.

## R226-1 closed within the source boundary

`service/entrypoint.py:27–52` records the callable's result or exception in a retained concurrent Future **inside the executor callable**, before scheduling completion observation. The asyncio wrapper and each observer await are shielded from cancelling that result latch. The executor submission is separate from those observer tasks. Cancelling a wrapper therefore cannot turn an unfinished acquisition into a completed operation or erase the eventual returned owner.

`entrypoint.py:325–329,358–361` retain the lifecycle and its operations on app.state and in the process-global registry. The executor callable/operation retain the in-flight result responsibility; after acquisition returns, its result latch retains the owner even when the wrapper was cancelled. Cleanup assigns that owner to the record and app.state at `375–378`. Notification close and registry removal occur only at `392–398`, after the relevant operation completes and any acquired owner's close succeeds.

The restart decision at `331–356` is serialized on the event loop. It refuses to replace an active observer or to restart while its recorded thread operation is unfinished. If completion's `advance` callback runs before the cancelled observer finishes, that observer's done callback rechecks the completed latch; if the observer finishes first, thread completion supplies the later recheck. Completed-before-await results are observed through the same latch. Operation registration occurs synchronously before the loop can run the queued callback. There is no cancellation-catching polling loop. Cleanup creates and records close only when no close operation exists (`379–386`); restarting an observer reuses that submission, preserving one owner-close invocation.

Cancellation at `369–371` propagates while the observer is being cancelled or its operation remains unfinished. A completed preparation/acquisition thread-originated `CancelledError` is handled as that operation's failure instead. A close failure, including thread-originated `CancelledError`, retains owner, notification, result and error: `344–347` prevents automatic close retry after the cancelled observer finishes. Ordinary failed cleanup tasks are likewise not restarted (`337–338`); wrapper/submission and cleanup callbacks retrieve task exceptions (`45–48,351–355`). Notification-close failure records its error and retains the lifecycle (`392–396`).

Rejected completion scheduling on a closed loop (`40–44`) does not erase the completed latch, owner or lifecycle. It also does not establish cleanup success. Recovery on a live loop is demonstrated below; **abrupt loop/process death remains unqualified**, with no claim of owner close, descendant death or reap. Retention is an unresolved obligation, not a shutdown receipt.

## Prior activation and ordering checks retained

R224-2 remains closed: `service/dispatcher.py:170–183` requires installed postcheck, reconciliation and healthy quarantine/orphan/resource/persistence/integrity state before queued-cancellation release. The final installed hello precedes that release at `426–437`; late-hello refusal preserves queued accounting. Internal startup reconciliation remains able to run at `362` while legacy enrolled owners remain inspect-only. Both the healthy release positive and combined late-hello negative passed.

The original activation brackets remain at `entrypoint.py:215–231,246–295`: pre-Core waiting/private-copy/configuration/registration/incarnation/hello verification, one-use runtime/root binding, and repeated checks under Core lifetime exclusion. Reconciliation and persistent readiness still precede API mounting (`410–417`). Explicit private credentials, fixed configured model mapping/policy, truthful installed status, resource/integrity/orphan guards and receipt-aware shutdown remain in the carried unit. The shutdown regression preserves executing capacity after stop send and permits owner exit only after its injected receipt. No packaging omission is introduced as a new source gate.

## Independent checks

Used only `/private/tmp/pab-0c-venv/bin/python -I -B`, with explicit sys.path entries `/private/tmp/pab-r227-sol`, `/private/tmp/pab-r227-sol/tests` and `/private/tmp/pab-r227-sol/tests/service`, running unittest. The seven authorized modules passed: **80 tests, 33.326 seconds, exit 0**:

- `tests.release.test_headless_install`
- `tests.service.test_installed_activation`
- `tests.worker.test_engine_lifecycle`
- `tests.service.test_production_api`
- `tests.service.test_production`
- `tests.service.test_production_admission`
- `tests.clients.test_production_clients`

The five r227 additions (`test_installed_activation.py:327–340`, helper `218–316`) exercise direct cleanup cancellation during preparation, acquisition and close, concurrent acquisition-wrapper cancellation, and rejected completion callback retention. They assert no premature notification close/registry removal, no observer replacement while the thread is paused, eventual live-loop cleanup and one close. The callback rejection is injected with a live test loop, not actual loop death. Existing repeated caller cancellation and close-failure retention also pass. Output includes the intentionally injected `injected_close_unknown` shielded-future diagnostic and worker refusal diagnostic; these are not evidence of successful unknown shutdown. The reported three RED-on-old-source failures were read as implementer evidence, not independently replayed through source mutation.

Two additional independent stdin-only inert probes passed (**2 tests, 1.268 seconds, exit 0**) using the real disposable ActivationTests/Core/Dispatcher fixture:

- Owner close raises thread-originated `asyncio.CancelledError`: after loop observation, exactly one close invocation, recorded CancelledError, cancelled observer, retained global/app owner, open notification, `closed=False`, no retry.
- Preparation raises thread-originated `asyncio.CancelledError`: completed failure distinguished from pending observer cancellation; no acquisition submitted, notification closed only after completion, lifecycle removed with recorded preparation error.

Only captured fixture owners received normal explicit close afterward; only those fixtures' retention was removed. No operational state was inspected or repaired. `git diff --check ff42548 HEAD` passes. All eight changed Python files parse with `ast.parse(feature_version=(3,11))`. Interpreter is Python 3.14.7: grammar evidence only, not Python 3.11 runtime/dependency/package compatibility. Candidate remains clean at the pinned HEAD.

## Limits and handoff

No pytest/maintenance guard/full suite, Docker, actual proc/mount/UID qualification, network/listener, model download/inference, ML/hardware, HELD operation or production-state mutation. Files, SQLite, locks and writer threads are disposable fixtures; identity, host, hello, reap and backend observations are injected. No actual Linux installation, packaging, audible generation, descendant reap or release acceptance is established. Parent may adopt this bounded source unit; the existing actual qualification and packaging successors remain parent-owned, without a broad new gate.
