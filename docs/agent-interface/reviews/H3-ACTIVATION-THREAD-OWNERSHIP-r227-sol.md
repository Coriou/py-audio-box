# H3 r227 Sol — thread result responsibility correction

2026-10-24. R226-1 corrected in the bounded source unit for independent review. No self-adoption, actual shutdown/reap, installed activation, packaging or release acceptance is claimed. R224-2 remains closed; Dispatcher is unchanged by this correction.

## Exact pins and scope

Fresh own worktree `/private/tmp/pab-r227-sol`, branch `fix/r227-thread-ownership`, created from current project root **`ff42548f54ca272bc763675cebba6e7a417fa692`**. Imported only requested `d692a3d715dc8d5cca6c2fc21f17150bd3f8d44d` then `20f401833bb725db0dc15f2f683d27186577a97f`, as direct successive commits **`c96330142d7d3ef84c9481c81e328cb6d8b38d8a`** and **`4b2d9731497388acbf595f4b4ad86c7750e74bf8`**. Correction source/tests commit **`7dfbd63953f620571e9fa1c8d546c1b97c5471a0`** directly follows those imports. This report is committed separately afterward. Only this report is copied to project root; no source adoption or root commit.

Read complete root r198 (including shutdown), r223, r224, r225 and r226, including the direct cleanup-cancellation probe. Inspected entrypoint, installed activation fixtures and both conftests. No applicable AGENTS.md found in scoped/ancestor checks. Bounded memory-registry search supplied no relevant premise. No skill, delegation or self-adoption. Correction manifest is exactly `service/entrypoint.py` and `tests/service/test_installed_activation.py`; this report is the only other correction write. Imported candidate changes are distinct from this correction. No packaging or Dispatcher edit.

## Responsibility change

Each preparation, acquisition and close submission now has a strongly retained `_ThreadResult` containing a concurrent Future. The executor callable stores its eventual value or exception itself, before its finally schedules loop observation. Neither an asyncio wrapper cancellation nor cancellation of cleanup can cancel that result latch or the executor submission. The app-visible/process-global lifecycle record retains these operations, their notification and any acquired owner. While an acquisition return is outstanding, the thread stack owns its local owner; after return, the independently retained result owns it even if no asyncio wrapper delivered it.

Cleanup cancellation propagates as cancellation; it never means the operation completed. A cancelled observer remains retained while the outstanding thread operation runs. Completion callbacks advance cleanup on the live loop, obtaining the acquisition result and putting the owner on app.state. Close has one recorded submission and one close intent; resuming observation never calls close again. Notification close and registry removal occur only after completed preparation/acquisition and successful owner close. Failed/unknown close retains its result/exception, owner and notification without retry. A thread-originated CancelledError is distinguished from cancellation of a still-running observer; close failure is retained rather than automatically retried.

Callback rejection by a closed loop leaves the completed result and obligation retained. No callback error erases the record or reports successful cleanup. This cannot establish successful close after abrupt loop/process death. No automatic loop revival, cancellation-catching busy loop, polling framework or unbounded close retry was added. The existing bounded readiness loop is unchanged.

## Verification

Only `/private/tmp/pab-0c-venv/bin/python -I -B` was used for tests, with explicit sys.path entries `/private/tmp/pab-r227-sol`, `/private/tmp/pab-r227-sol/tests`, `/private/tmp/pab-r227-sol/tests/service`. Used unittest, not pytest, its maintenance guard, listener fixture or full suite.

RED: retained new tests and restored only entrypoint to imported r225 bytes. The three direct cleanup-cancellation regressions all fail: acquisition and preparation close notification prematurely; close-thread cancellation never completes lifecycle cleanup after the actual thread returns (bounded source-wait failure). Final RED: **3 tests, 3 failures, 13.550s**, `/private/tmp/r227-red-final.log`. Captured owners received normal fixture cleanup. An earlier RED development run also exposed test access to a new-only record key; the final RED removes that artifact and fails on behavior. Corrected source restored byte-for-byte afterward.

GREEN: **80 tests PASS, 33.348s, exit 0**, `/private/tmp/r227-green.log`, using exactly:

- `tests.release.test_headless_install`
- `tests.service.test_installed_activation`
- `tests.worker.test_engine_lifecycle`
- `tests.service.test_production_api`
- `tests.service.test_production`
- `tests.service.test_production_admission`
- `tests.clients.test_production_clients`

This preserves all 75 previous tests and adds five: direct cleanup cancellation during acquisition, preparation and close; direct acquisition-wrapper plus cleanup cancellation; and rejected completion callback retaining the eventual owner as an unresolved obligation. Thread events bound the injected pauses. Tests repeat cancellation of the retained pending observer and assert no replacement/busy loop while the thread is unfinished, no premature dependency close or registry removal, eventual normal cleanup on the live loop and exactly one close. Callback rejection asserts the real owner remains in the retained thread result, notification stays open, and no successful-close marker appears. Existing repeated-caller cancellation and injected close-failure tests remain green. Callback-closed behavior is injected while the test loop remains alive; it is not actual abrupt-loop-death qualification.

The activation-only development run passed 25 tests before the callback rejection test was added. Output includes existing SQLite ResourceWarnings, the intentionally injected close-failure asyncio diagnostic and worker refusal diagnostic; none is reported as successful actual shutdown. No filters were added. Initial editing command found no `python` alias and made no edit; source editing used available `python3`, while every test used the exact requested interpreter.

`git diff --check` passes. Dispatcher has zero difference from the carried r225 source. Both correction Python files parse with `ast.parse(feature_version=(3,11))`. Interpreter is Python **3.14.7**; this establishes Python 3.11 grammar compatibility only, not Python 3.11 runtime/dependency/package compatibility.

## Limits and handoff

No Docker, actual proc/mount/UID qualification, network/listener, model download/inference, ML/hardware, pytest guard/full suite, HELD/maintenance operation, production-state mutation, delegation or self-adoption. Files/SQLite/locks/writer threads are ordinary disposable owned fixtures; host, identity, hello, reap and backend observations remain injected. Only captured fixture owners and fixture retention receive normal cleanup. Retention is an unresolved obligation, not death/reap evidence.

Independent review is next. Parent owns adoption and the separate actual runtime, installation, permission, peer/mount, shutdown/reap, speech and release gates. This remains the bounded R226-1 correction, not packaging work.
