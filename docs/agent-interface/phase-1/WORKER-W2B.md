# W2b operational synthetic continuation — review candidate

2026-10-16. Implementation handoff, **awaiting independent Sol-low review**.
No self-GO, Phase 1A acceptance, production dispatcher activation or deployment.
Baseline `a008ea4256ec8f293e04c6bfae5b5c0355b74885`; initially clean.
Branch `t3/implement-w2b-runtime`; assigned worktree
`/Users/ben/.txcode/worktrees/py-audio-box/t3-8466bed9`; no extra checkout.
Candidate SHA is supplied in the commit handoff (this report is part of it).
Contracts/fixtures and canonical wire messages remain 1.3.0.

## Outcome and interface

This adds a functioning separate-container fixed synthetic runtime, not another
standalone evidence protocol. `worker.supervised_synthetic` listens on the private
UDS, mutually authenticates each connection using accepted W1 UID/GID plus
challenge/HMAC allocation binding, validates execute against the frozen context,
and computes one proposal. It caches the complete proposal, including allocated
output/artifact IDs, before attempting its physical send. A lost response can be
retrieved by a new authenticated control process without reexecuting or changing
the deadline. An exclusive execution marker fails closed if generation fails or
someone attempts to reuse staging. No worker DB/service import was added.

The existing accepted W1 fixed byte generator is reused through a bounded internal
socketpair/thread. This avoids introducing a second synthetic media implementation;
the internal exchange uses the same authenticated wire checks. The only supported
operation remains the known one-variant preset silence worker. It starts no child
processes. Threads are contained by the process-wide deadline watchdog. There is
**no general descendant/process-tree or engine qualification claim**.

`Supervisor.dispatch(authority, prepare, start, fault=...)` commits intent before
calling trusted infrastructure. `prepare` must create a never-started runtime;
the supervisor binds its immutable identity before calling `start`. Thus physical
spawn has no unregistered gap. Failure before binding leaves unbound unknown;
failure after binding retains the exact identity. Repeated allocation lookup
never calls prepare/start, including after lost start acknowledgement. Unknown
requires operator evidence/recovery; no automatic retry, cleanup or replacement.
A created-but-unstarted container is unknown, not falsely stopped-and-reaped.

`Supervisor.exchange` gates accepted W1 exchange on the persisted exact live
allocation. Narrow W1 transport hooks bracket the actual execute-frame send;
default behavior and wire/session format are unchanged. `Supervisor.verify`
requires a proposal for the stored tuple and independently confirmed termination,
then invokes only the serialized control owner's verifier callback. The store now
explicitly checks its owning thread in addition to SQLite affinity and lifetime
exclusive control lock. There is no new adoption/publication implementation and
no Core or public route hook. Existing publication remains on its control owner.

The worker anchors the original absolute deadline to monotonic time once and
installs a process-wide watchdog before accepting connections. No control process,
reconnect, missing ACK, blocked read or generation thread can reset that cutoff.
The synthetic harness also caps runtime at 30 seconds. A restart cannot authorize
another execution: runtime.start requires Docker's never-started `created` state;
changed start time/restart evidence becomes unknown and the fixed worker's
existing socket/staging fences reruns. Worker restart is fail-closed interruption,
not persistent cache recovery. Cache lifetime is exactly the original process.

## Trusted harness operator, not a deployed adapter

`DockerHarnessRuntime` runs **on the test host**, outside both containers. It
bounds daemon/inspect/wait/signal subprocesses. Identity binds daemon ID, immutable
container ID and image ID; the operator retains the first observed StartedAt after
its start operation. Every observation checks these and restart policy/count.
Alive requires daemon running evidence. Stopped-and-reaped requires exited,
Pid=0, FinishedAt, no restart, and independent matching `docker wait` exit status.
PID absence, JSON owner, lock acquisition, signal return or missing container
alone never suffices. TERM is followed by observation; KILL escalation is followed
by observation. Signals refuse changed or unobserved start identity.

The operator is a harness capability, **not a deployed control capability**.
The container control test receives typed observation values from the trusted
host operator through its command environment, never from the worker. Stop uses
the real Supervisor with a harness-only command proxy to this OwnerStore; it
does not duplicate the stop state machine. All DB access stays inside control. It has no
Docker socket or daemon credentials. Losing/restarting the operator loses its
start-observation cache and yields unknown; it cannot reconstruct proof from a
store/PID. No operational companion service or general production dispatcher is
claimed. A deployed companion with durable exact-incarnation evidence remains a
separate integration choice. Out-of-band daemon administrators remain trusted;
the supervisor does not protect against an administrator restarting an already
retired container outside this operator's start API.

## Deterministic coverage and actual container proof

E1 fixtures exercise committed intent surviving control death, reopen after
pre-spawn/pre-registration/pre-start/post-spawn faults, no repeated physical-start
callback, presend/postsend/observation-reap/preverification interrupts, unknown
publication refusal, stale proposal refusal, observer failures, signal-vs-death,
immutable deadlines, writer exclusion, restart/start-time mismatch and loss of
operator observation state. These are deterministic policy fixtures, not claims
that every one is a separate-container daemon crash experiment.

The E2 runner executes two real separate-container scenarios sequentially:

- Deadline with control absent: prepare commits control-only intent; external
  operator creates a stopped container; fresh control binds it; operator starts
  it and waits for an explicit ready barrier. Wrong session is refused. Control
  exits 17 immediately after sending execute, before reading/journaling result.
  Fresh control processes reconnect twice and compare exactly the cached proposal
  IDs/manifest/bytes. All controls then remain exited while the worker watchdog
  ends the original runtime. Independent daemon wait/inspect precedes a fresh
  control verifier. This covers actual control restart/lost ACK and timely result
  winning collection after deadline termination without extending execution.
- Bounded cancellation/stop after a result: the synthetic fault fixture ignores
  TERM. The actual Supervisor.stop policy drives the host operator through a
  harness-only serialized control-store command proxy. It requests TERM,
  independently observes it still alive, then sends KILL and independently confirms exited/reaped before control verification.
  The previously received valid result remains verifiable (stop is not itself a
  cancellation/publication verdict). Manual worker-container restart is detected
  as unknown by the operator, never rebound or used to dispatch again. The
  harness independently waits for the restarted test container before removing it.

These scenarios use real OS/container evidence; fake runtime states are confined
to the explicitly labeled E1 fixtures. No timing-only race selects a verdict:
ready uses a file barrier, lost ACK uses an exact send hook, stop uses daemon
observation, and deadline uses the worker watchdog plus daemon wait. Deadline
budgets bound waiting; they are not performance measurements. The exhaustive
production dispatcher/cancellation/publication durability matrix remains outside
this synthetic continuation.

## Packaging, topology and evidence

`service/Dockerfile.w2` builds offline from the existing pinned dependency image
`sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`
and the pinned existing Python stage used by W1. There is no fresh install claim.
The runner compares the actual control image's complete source map under the six
supplied roots against this checkout, and the worker's supplied subset against
those same hashes; verifies absence of service source in worker; records immutable
image IDs and hashes of runner/Compose/Dockerfile/ignore. This is source-overlay
E1/E2, not clean dependency installation, power-loss or hardware acceptance.

Actual topology is asserted and recorded: network none, distinct nonroot users,
private PID namespaces, read-only roots, dropped worker capabilities, no host
bind mounts. Worker sees only disposable private IPC and attempt staging volumes.
Its server requires writable IPC (unlike W1's connecting worker); it sees no
control-store/publication/credential/Docker-socket volume. Control sees its own
separate store volume. Both use private session provisioning; secrets are neither
printed nor recorded in receipts. No user data/audio/model mounts are used.

The runner takes the shared nonblocking W1/W2 Docker lane, with bounded commands,
random owned resource names and prior inventories. Initial read-only inventory
showed no active root test/build container. It never invokes the legacy remote
E2E script, prunes resources or stops unrelated workloads. It removes owned
containers only after wait/inspect confirms termination, then their volumes;
images are retained. Prior container/volume inventories are checked preserved.

Raw sanitized receipts are under `w2b-evidence/`. `failure-1.json` records the
unsupported Compose `create --no-deps` option. `failure-2.json` records the
one-shot-container volume-reference cleanup bug (`FileExistsError` on the next
scenario, followed by an empty-inspect cleanup `IndexError`).
`failed-run-cleanup.json` records later cleanup of those exact owned resources.
Intermediate successful receipts remain attributed to their own source hashes.
`run.json` is the final-source run; see the final verification section below.

## Ownership and remaining gates

Owned changes: worker/supervised_synthetic.py; service/runtime_adapter.py,
worker_supervisor.py, owner_store.py; narrow service/worker_transport.py send
hooks; tests/service/test_worker_lifecycle.py;
tests/release/test_worker_lifecycle_boundary.py; service/compose.w2.yml,
Dockerfile.w2, Dockerfile.w2.dockerignore; scripts/test-worker-w2b.sh; this report
and w2b-evidence. worker/lifecycle.py needed no change. No core, contract, client,
studio, legacy, ledger or review edits. No new DB access by workers.

Next gate is independent Sol-low source/diff/runtime review of the exact candidate,
then orchestrator integration verification. No subagents, push, PR, deployment,
remote E2E, models/private audio, cloud, STT or Voicebox operations occurred.
Resource admission, P1 persistence/publication durability, registry, rendering,
full production dispatcher, deployed companion, full legacy suite, fresh install,
real engines/hardware and final Astra/Phase 1A acceptance remain open.

## Final verification

- `W2_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-worker-w2b.sh` — exit 0.
  87 passed in 25.29s. Both E2 scenarios passed, including
  exact send-hook control death, same cached manifest, deadline without control,
  actual Supervisor TERM/KILL escalation, independent wait/reap and restart quarantine.
- `bash -n scripts/test-worker-w2b.sh` and `git diff --check` — passed.
- Final audit rehashed every supplied control source and the exact harness/packaging
  bytes against the successful run. Zero owned containers/volumes remain; prior
  inventories preserved. See `w2b-evidence/final-audit.json`.

Final immutable images:

- Worker: `sha256:c2d09af707f490a5ca635d2b405d0c44c9e2abb28a8e80f77817e38afa2f05c8`.
- Control: `sha256:ec0194ea33ec7ce9af1e0459e3e06a1c7feb1360e62d7ddaef1294c4bc1b93c7`.

These results are candidate evidence, not independent acceptance.
