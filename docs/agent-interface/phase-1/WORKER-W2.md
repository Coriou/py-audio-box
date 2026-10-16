# W2a — durable owner fencing seam (candidate, not accepted)

## Frozen interface before implementation

Baseline 9254776; branch t3/w2-worker-supervision; app-owned worktree
/Users/ben/.txcode/worktrees/py-audio-box/t3-001f240a. Initially clean.
Contracts remain 1.3.0. W1 remains dormant and unchanged.

This is the first coherent W2 sublot, not completed W2. Owned implementation:
worker/lifecycle.py, service/owner_store.py, service/worker_supervisor.py,
tests/service/test_worker_lifecycle.py and this report/evidence directory.
No core hook, migration, dispatcher, transport change or publication call.

Interface: one serialized control owner holds an exclusive store lock. SQLite
FULL transactions persist an immutable dispatch tuple (job, attempt, generation,
owner, input hash, staging, absolute deadline, W1 authority binding) BEFORE any
spawn callback. A runtime adapter supplies an opaque immutable runtime identity
(e.g. container ID plus daemon identity), and independently observes that exact
identity. This seam does not implement a Docker adapter. JSON, PID, session and
lock release are never termination evidence.

State machine: intent -> bound -> alive | unknown | stopped-and-reaped.
Intent without runtime identity is unknown after interruption, including the
spawn/registration gap. Unknown retains the lane and forbids replacement,
cleanup and publication. A later independent exact-runtime observation can
resolve unknown. Stopped-and-reaped is absorbing. Runtime identity cannot change.
Replacement requires the prior exact runtime's confirmed termination. No automatic
retry. Reconnect returns the original tuple/deadline without invoking spawn.
Repeated dispatch of that tuple is idempotent; a different tuple is refused.

Stop asks the runtime adapter for bounded graceful stop, then bounded kill,
then independently observes. A successful signal is not termination. Adapter
errors yield unknown. Only explicit stopped-and-reaped evidence releases the
lane. The adapter is trusted control infrastructure, never worker input.

Deadline remains an immutable absolute value. At/after it, control requests stop;
this sublot does NOT enforce deadlines while control is absent. Therefore it
must not execute a worker or claim restart-safe deadline enforcement. Publication
remains exclusively the serialized core owner and is not connected here.

## Remaining W2 gates

Next sublot must implement the standalone synthetic runtime adapter/companion,
W1 reconnect with cached same-attempt result and no duplicate execute, independent
worker deadline watchdog, bounded stop/escalation and exact runtime proof across
control restart. Then deterministic pre/post spawn/send/reap/verification faults,
lost ACK, stale session, cancellation/result/deadline races and actual separate
network-none container receipts (source/image/topology E1/E2). No W2 or Phase 1A
acceptance follows from this seam. Independent Sol-low review is next; no self-GO.

## Candidate evidence and handoff

Owned paths also include scripts/test-worker-w2.sh. No W1 files changed.
The store requires a private, trusted local directory; it is not a hostile-store
repair tool. The lock serializes control writers only, never proves worker death.
Runtime adapter time bounds are a protocol obligation, not enforced against an
arbitrary blocking Python adapter in this sublot.

Checks (2026-10-16):

- Host `python3 -m pytest tests/service/test_worker_lifecycle.py -q -p no:cacheprovider`
  failed collection: host lacks jsonschema. No host installation attempted.
- Offline pinned dependency-image overlay: `python -m pytest tests/service tests/release
  -q -p no:cacheprovider`: **76 passed in 32.34s**. Full command/source map and
  termination receipt: w2-evidence/regression.json. This first run copied current
  source; its generated source-verifier file was not collected. Do not describe
  that run alone as in-container hash verification.
- `scripts/test-worker-w2.sh`: **9 passed in 6.91s**, comprising eight lifecycle
  cases plus in-container SHA256 verification of every supplied source file.
  Receipt w2-evidence/local.json. The runner subsequently gained the shared W1
  lane lock and moved temporary-path initialization ahead of cleanup; `bash -n`
  passes. Those harness-only edits have not repeated container execution.
- `git diff --check` and `bash -n scripts/test-worker-w2.sh`: passed.

Both runs used dependency image
sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e,
no image build, network none, no host mounts, no remote service or engine.
Each random owned container was inspected exited before removal; previous
container inventory was preserved. No volumes were created. These are **E1
source-overlay receipts**, not fresh-image, separate-control/worker E2, real
engine, power-loss, or complete W2 acceptance evidence. There was no active
pytest/build container in the initial local inventory; no unrelated workload
was stopped. The root 68-test result is not re-certified here.

Review this W2a candidate independently before expanding it. The remaining W2
gates listed above are still required. No production activation, cleanup of
staging, admission, registry, rendering or public API behavior was added.
