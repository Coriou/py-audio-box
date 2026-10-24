# H3 r225 Sol — activation lifecycle corrections

2026-10-24. R224-1/R224-2 source corrections implemented for independent review. No self-adoption or actual installed/runtime/release approval.

## Pins and scope

Implementation checkout `/private/tmp/pab-r225-sol`, branch `fix/r225-activation-lifecycle`. Final base is current project root `4aa6c5d1a1f9eddb713e7655ee7d8fd46b50b0d1`. Only requested source `d692a3d715dc8d5cca6c2fc21f17150bd3f8d44d` (original parent `663af006833bbbe8304928b5c49aa168d1268283`) was imported, as `591c0f15b42c6239f714ffbe54089f39795a9a87`. Correction source/tests commit is **`20f401833bb725db0dc15f2f683d27186577a97f`**, directly following that import. Report is a separate subsequent commit; only this report is copied to project root, without source adoption or root commit.

The fresh checkout initially used supplied thread checkout HEAD `e13eaca5c0c314e398f3c967b3f7fe7e67b749b5`. Final pin verification found project root had advanced to `4aa6c5d`; the two implementation commits were rebased onto that current root. Its intervening delta is documentation only (r224 report and execution ledger), with no source difference. Final named verification was repeated after pin correction.

Read r198 activation/shutdown requirements, r223 handoff and full r224 review including both corrected probe descriptions; inspected real entrypoint/Dispatcher and fixture code and both conftests. No applicable AGENTS.md found in checked ancestor/scoped paths. Bounded memory registry search returned no relevant hit and supplied no premise. No delegation or skill. Correction allowlist: `service/entrypoint.py`, `service/dispatcher.py`, `tests/service/test_installed_activation.py`, and this report. Original eight-file candidate import is distinct from this three-file correction. No packaging or check weakening.

## R224-1

Lifespan keeps a strong process-global responsibility record, also exposed on app.state, from preparation until cleanup succeeds. Preparation and acquisition tasks are initially shielded. Finally creates one retained cleanup task that waits for their eventual results independently of caller cancellation, assigns any acquired owner to both the record and app.state, and issues owner.close exactly once. Every caller await of cleanup is shielded. Further cancellation propagates without cancelling acquisition/preparation or the cleanup task; there is no cancellation-catching busy loop.

The notification closes only after preparation has ceased and acquired owner close succeeds. Failed/unknown/timeout close retains notification, owner, error and cleanup task in app.state and the process-global registry. The background completion callback retrieves errors; it does not erase obligations or retry close. Successful cleanup removes global retention. A process/event-loop teardown is not established by these source tests as a successful shutdown; retained objects do not constitute reap proof.

Thread-event regressions construct a real temporary Dispatcher/Core and pause the acquisition return, cancel lifespan twice before releasing it, then verify dependency remains open, owner is tracked, one close occurs and the writer stops. Analogous preparation test pauses the real completed preparation before its return and proves no owner construction and no premature notification close. Injected close failure verifies owner/app/global retention and notification preservation. Fixture cleanup explicitly closes captured live owners and removes only fixture retention.

## R224-2

Installed queued cancellation release moved after the final reconciliation hello and reconciled/quarantine decision. The common activation predicate now requires installed postcheck plus reconciliation and healthy quarantine/orphan/resource/persistence/integrity state. Internal tick startup still runs the hello/reconciliation machinery without waiting on its own public activation predicate; legacy enrolled remains inspect-only. Installed release runs only through the full healthy predicate, preserving accounting on refusal. Command gates use that same closed predicate.

A real temporary Core/Admission/Production reservation is cancelled before startup; hello 5 changes boot. Reconciliation refuses and the reservation stays `queued` and accounted. The existing healthy positive still releases to `retained`. The late hello check is unchanged.

## Verification and evidence limits

No pytest invocation, maintenance guard or full suite. Interpreter command is `/private/tmp/pab-0c-venv/bin/python -I -B`, with explicit sys.path entries `/private/tmp/pab-r225-sol`, `/private/tmp/pab-r225-sol/tests`, `/private/tmp/pab-r225-sol/tests/service`, running unittest modules:

- `tests.release.test_headless_install`
- `tests.service.test_installed_activation`
- `tests.worker.test_engine_lifecycle`
- `tests.service.test_production_api`
- `tests.service.test_production`
- `tests.service.test_production_admission`
- `tests.clients.test_production_clients`

RED: swapped only the two allowlisted implementation files back to original imported source while retaining new regressions; all three defect regressions failed in 9.558s: accounting was retained instead of queued; notification closed prematurely during acquisition and preparation. Captured owners received normal fixture cleanup. Restored correction bytes afterward. Close-failure retention is an additional GREEN obligation test.

GREEN: seven modules, **75 tests PASS**, including 71 original plus four new tests. Final exact-source run after base correction: 75 tests, 29.616s, exit 0. Initial development run caught an extra fixture close on an already stopped owner; corrected fixture cleanup checks thread liveness. Logs include existing SQLite ResourceWarnings and an asyncio diagnostic for the intentionally injected close failure; those are not reported as successful real shutdown or hidden by filters.

`git diff --check` passes. Three correction Python files parse with `ast.parse(feature_version=(3,11))`; interpreter is Python 3.14.7. Grammar compatibility only, no Python 3.11 runtime/dependency claim. RED log `/private/tmp/r225-red.log`, final GREEN log `/private/tmp/r225-final.log` are disposable local logs, not production evidence.

No Docker, actual proc/mount/UID qualification, network/listener, actual model/ML/hardware, HELD operation, production-state mutation, packaging, delegation or self-adoption. Real files/SQLite/locks are disposable fixtures; identity, host, hello, reap and backend observations remain injected. Only normal fixture cleanup occurred. Independent review is next; parent owns source adoption and all actual qualification gates.
