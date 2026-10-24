# H4 Pocket guard-only independent review — r250 Sol

2026-10-24. **Bounded SOURCE GO for the guarded fallback only.** No new material defect found in the three-file source delta. **Pocket post-reap repair and restart readiness remain NO-GO:** the originally requested repair was not delivered. This review accepts only the explicit r248 reachability fallback, not complete lifecycle acceptance, activation, actual execution, installation or release. Parent owns adoption.

## Exact pins and scope

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-f0529861`, clean HEAD/report **`80a3458f515d7255bec4115728e0ee17aa4cece1`**. Base **`54b5fb5a5a76ae9f82f041fb687805c39a782956`**; source **`ca0f9fa0740bcd00e005bc3d813a21e2c2a691b8`**. Verified direct chain base → source → report; base's parent is `b94ffa8057e3d78d88ea90e0371fddec84c9e20b`. Source delta: `service/admission.py`, `worker/runtime_supervisor.py`, `tests/service/test_pocket_lifecycle.py`, 44 insertions/9 deletions. Source-to-report changes only r249 documentation.

Read r248 FULL (selected helper design and authorized guard fallback), r249 FULL and r244–r247 prior reports. Inspected complete source delta and relevant unchanged Core submission, Production, descriptor/dispatch validation and supervisor lifecycle paths. Inspected root/service conftests before running tests. No applicable AGENTS.md found in checked ancestors or candidate tree; checked shared-root report ancestors too. Bounded memory registry search returned no relevant match and supplied no premise. No delegation, source edits or commits. Sole lasting write is this shared-root report; pre-existing root evidence remains untouched. Candidate remained clean after execution.

## Findings and guard ordering

No new blocking finding within the requested guard-only scope.

- `Admission.demand` validates the closed descriptor, then unconditionally refuses `spec.engine == 'pocket-tts'` with `pocket_restart_repair_required`. Core calls demand before `_preflight`, the submission transaction, job/event insertion and resource reservation; `reserve_tx` also calls demand before its store mutation. Production may perform existing owned catalogue/reference reads first, but the guard creates no job, reservation or staging entry. Existing idempotent job lookup remains available; returning an old job does not create a new reservation.
- Supervisor `operation('prepare')` validates snapshot binding, stored context, descriptor and fixture/dispatch consistency, then independently refuses Pocket before mount/input verification, journal access/allocation, monotonic cutoff installation or active-lane mutation. Tests use an uninitialized supervisor with no journal and exercise missing and complete reference mappings; input verification is asserted uncalled. A malformed/incorrect fixture refuses earlier rather than bypassing the guard. Existing records are not made unreachable: observe/stop/collect/ACK branches do not pass through prepare. This is a new-prepare fence, not revocation of an already prepared record's existing start semantics.
- Neither production guard reads environment state or exposes a public bypass. Descriptor parsing, trusted context/catalogue reads and existing lifecycle handling are unchanged. Kokoro's parsed engine does not match the new condition, and its existing admission, generation, output and dispatch regressions pass. No helper, deletion, new receipt/schema/phase, retained-namespace exception, recovery waiver or historical latch reset appears in this delta.
- The test-only patch validates the actual descriptor, substitutes an inert engine label solely for the original resource-demand call, and retains qualification/resource checks. It is scoped with its own cleanup; the dedicated product guard test temporarily restores the original method and verifies refusal even with `POCKET_RESTART_READY=1`, followed by empty jobs/reservations/staging assertions. These patched adapter successes cannot count as product readiness. The old supervisor mapping-verification-through-prepare coverage is deliberately superseded by the earlier fence; r249 discloses that loss accurately.

R245's incompatible retained-asset representation remains removed by r246/r247. The retained empty/nonempty conditioning regression still carries actual persisted UNKNOWN rows to the real diagnostic eligibility consumer and asserts refusal. No new durable producer/reader mismatch is introduced by this guard delta.

## Independent verification

Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B`, **Python 3.14.7**; `PYTHONDONTWRITEBYTECODE=1`, candidate cwd and explicit absolute candidate/candidate-tests-service import paths. Used unittest imports only: no pytest collection, autouse maintenance guard or listener fixture.

Exact named suite:

```python
names = (
    'tests.service.test_pocket_lifecycle',
    'tests.service.test_engine_admission',
    'tests.worker.test_engine_lifecycle',
    'tests.worker.test_engine_pocket',
    'tests.worker.test_engine_kokoro',
    'tests.service.test_production',
    'tests.service.test_production_admission',
    'tests.service.test_production_reconciliation',
    'tests.service.test_production_api',
    'tests.clients.test_production_clients',
)
```

Loaded each name with `unittest.defaultTestLoader.loadTestsFromName`, combined in `unittest.TestSuite`, ran with `TextTestRunner(verbosity=1)` and exited nonzero unless successful. **63 tests PASS, 13.126 s, exit 0.** Existing injected `lifecycle supervisor unavailable; lane quarantined` diagnostic appeared with overall OK. HTTP client/API tests use in-process ASGI transport. Lazy-import tests use inert interpreter probes with heavy imports forbidden; no actual engine/model process runs.

Additional independent in-memory supervisor probe: an existing Pocket started record remains observable, stop calls a mocked `stop_tree`, and nonterminal collect returns no receipt; after an injected terminal transition, collect returns the original receipt and ACK preserves it while marking acknowledged and clearing proposal through a mocked persistence seam. **PASS, exit 0**, no process or filesystem operations. This checks branch reachability, not actual death, transport, journal durability or release.

`git diff --check 54b5fb5a5a76ae9f82f041fb687805c39a782956 HEAD`: PASS. Candidate status remained clean. `/private/tmp` observation: 3,877,892 KiB available, below the heavy-operation admission threshold; only authorized small inert fixtures and ordinary owned temporary-fixture teardown ran. No actual ML, Docker, network/download, listener, process crash, operational cleanup or HELD operation. Installed-history suites were not rerun in this focused guard review; prior reports' results are not represented as current execution.

## Remaining parent-owned gate

R249 honestly records incomplete delivery: all r248 full-repair acceptance groups remain unfulfilled. The selected supervisor-owned post-ECHILD helper is still required, with exact allocation/UID/GID/FD binding, kernel empty-only rmdir, fsync and stable identity validation, bounded helper lifetime and quarantine on uncertainty. Coordinate watchdog/lane ownership and preserve original engine reap/deadline/receipt meaning; no successful terminalization, release or ACK on cleanup uncertainty. Remove both readiness guards only with the complete independently reviewed repair.

Required evidence still includes ordering and failure tests, cross-UID policy/credential arguments, filesystem negatives, original timing/count preservation, journal failure after removal, and real same-root Dispatcher success/cancel reopen with coherent closed history. The cancellation fixture still skips conditioning creation and cannot prove empty-at-reap cancellation repair. Interrupted named WAV/YAML leftovers must remain UNKNOWN and blocked. Actual packaged permissions/mounts, containment, forced reap, audio/listening, reference preparation/audition and installed activation remain separate qualification work. No extra design gate is imposed here; the remaining implementation is the selected r248 helper and its acceptance tests.
