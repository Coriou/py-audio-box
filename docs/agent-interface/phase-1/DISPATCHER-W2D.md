# W2d dispatcher candidate

Base 53429a0df626cbee2adfa34476caeaccadb4976e; assigned worktree
/Users/ben/.txcode/worktrees/py-audio-box/t3-efa81fcc,
branch t3/implement-w2d-dispatcher. Canonical contracts remain 1.3.0.
Implementation candidate only; independent Sol-low review must follow.

## Pre-code interface freeze

One Dispatcher thread constructs and exclusively owns Core and its SQLite
connection/lock. A bounded 64-command queue returns detached results via futures;
only named internal synthetic submit, read, cancel and status commands are allowed.
No public route or client plan admission is added. Resource and persistent
activation remain false; R1/P1 are mandatory before unattended activation.

A dispatch journal in the same Core SQLite connection retains immutable Authority
snapshot/binding, fixture, runtime identity, phase, receipt, terminal outcome and
ack state for every allocation. Allocation and running transition commit together.
Unique attempt/owner/staging IDs and generation old+1 are control allocated.
Request idempotency and explicit retry lineage remain Core policy. No automatic
retry or replacement of uncertain attempts. Confirmed terminal failure permits an
explicit internal replacement with fresh IDs and generation+1, preserving hash.

Intent -> prepare command -> bound -> start command -> observe/collect -> durable
terminal outcome -> ack pending -> acknowledged. Commands persist before UDS I/O.
Exactly one bounded asynchronous management call is outstanding; its thread never
uses SQLite. No socket receive or child wait runs in the writer/event loop.
Restart first reconciles all unresolved rows; admission waits for that first pass.
Unbound intent is quarantined (never prepare replay). Bound reconnect observes and
collects the original identity; it never replays start or resets cutoff. Unknown
retains lane and staging, prevents dispatch/replacement/ack/cleanup. All stages
are retained. No Docker observations enter runtime policy.

Only a bound durable PID-1 receipt with exact original deadline and reap evidence
can reach terminal policy. Timely verified completion may win cancel; otherwise
confirmed reap yields cancelled for cancel_requested or failed. Late/unknown
receipts cannot publish. Actual bytes/media/manifest verification and adoption use
existing Core publication machinery, with explicit supervisor-controlled stage
permissions and receipt engine UID. Publication intent recovery is receipt-fenced.
Outcome is durable before ack; lost ack retries the same identity idempotently.
Old local execution remains available only via an explicit historical test switch;
the active dispatcher cannot invoke it. No readiness mutation or public activation.

## Implementation and activation boundary

The shipped entrypoint is `python -m service.dispatcher`, an internal JSON-lines
session with only named fixed-fixture commands; it opens no network listener.
The separate worker remains the accepted W2c PID-1 runtime. Core's default
`start`, `finish` and inherited-lock `reconcile` now reject with
`dispatcher_required`. Existing foundation tests explicitly opt into
`historical_local=True`; no installed dispatcher selects that switch.
The worker has only IPC/staging/receipt volumes; control alone has Core SQLite
and managed assets. Stage media is verified against the receipt engine UID and
control-owned 0770 directory, then copied through existing adoption machinery.
No worker DB access, Docker socket, host-observation runtime or new dependency.

E1 tests use explicitly fake receipts for deterministic policy interruption only.
E2 runs actual Core/SQLite on isolated named volumes, installed lifecycle UDS and
real PID-1 child/reap receipts. Synthetic profiles are private diagnostic inputs;
the public catalogue/readiness and resource/persistent activation remain closed.
The bounded runtime executor has one outstanding call, at most two seconds;
queue capacity is 64. Caller timeout does not retract a queued command: retry
submission with the original idempotency key. Close retains every journal/stage,
never signals worker children, and waits only for the bounded outstanding call.
The worker continues original deadline enforcement when all controls are gone.

Full R1 admission, P1 filesystem/orphan/power-loss/operator recovery, long-term
receipt retention, real engines, residency, registry, ingest, render, clients and
release acceptance are outside this lot. No stage or orphan cleanup is added.

## Recovery decisions and retained failures

Admission stores fixed diagnostic settings in the same SQLite transaction as the
job and initial event. A real SQLite trigger failure proves rollback of all three.
Duplicate requests retain their original job/settings. Historical active jobs
without dispatch journal authority quarantine startup rather than obtaining death
proof from a new local lock. Explicit replacement requires an acknowledged,
confirmed-reaped failed attempt and refuses a retained publication intent pending
reconciliation. It keeps the same frozen plan and increments generation with
fresh attempt/owner/staging IDs. Explicit retry instead creates a new job and
persists `retry_of`, preserving the existing canonical equality checks.

Publication recovery is gated by the exact bound, timely supervisor receipt and
matching attempt/generation/owner/staging in the control publication intent.
Before-adoption process death fails explicitly and retains the intent/staging;
after-adoption recovery verifies adopted bytes before success. This is bounded
process-interruption recovery, not P1's complete orphan or power-loss matrix.

`failure-1.json` retains the initial packaged runner's wrong expectation that
before-adoption recovery would succeed. Actual Core correctly returned failed
without artifacts. The test expectation was corrected; no failed receipt is
promoted to final-source proof. `intermediate-2.json` is a passing 28-scenario run
of its recorded earlier source maps, before atomic diagnostic-setting admission
and the additional installed cases. It is not final-candidate evidence.

E1 uses the whole affected service/release suite, including inherited W1/W2/W2c
framing, lifecycle, fence/receipt failures, start/send interruptions and watchdog
checks, plus dispatcher-specific deterministic interruption and SQLite tests.
The explicit historical Core helper tests do not establish active authority.
E2 new scenarios operate the actual default Dispatcher and Core against the
unchanged W2c supervisor. The installed stale-proposal case deliberately corrupts
a proposal after real authenticated collection, then proves Core rejects it; it
does not claim the authenticated supervisor emitted a stale proposal. Lost ACK
injection discards a real successful supervisor ACK response; the durable outcome
precedes it and a second real ACK is idempotent. The inherited W2c packaged matrix
remains attributed to its accepted source/images; it is not silently relabelled as
a fresh W2d run. Full W2 acceptance requires independent review and orchestrator
integration verification of the complete matrix. No self-GO is given.

`intermediate-3.json` records a passing 32-scenario version before the final
queued-cancellation atomicity correction. Final queued cancellation records
cancel_requested and cancelled in one transaction; a real SQLite trigger failure
at the terminal event proves both state changes roll back to queued. Running
cancellation still waits for exact supervisor receipt/reap. No success/failure
state is rewritten by the queued-only optimization.

## Owned paths and reproducible commands

- `service/core.py`: historical-executor guard, atomic diagnostic admission/cancel,
  and explicit receipt-UID/staging arguments to existing verifier/publication.
- `service/dispatcher.py`, `service/dispatch_store.py`: sole queued writer,
  asynchronous installed lifecycle and durable allocator/command/outcome journal.
- `service/compose.dispatcher.yml`, `service/Dockerfile.dispatcher`,
  `service/Dockerfile.dispatcher.dockerignore`: isolated diagnostic packaging.
- `tests/service/test_core.py`, `tests/service/test_dispatcher.py`,
  `tests/release/test_dispatcher.py`, `scripts/test-worker-w2d.sh`.
- This report and `w2d-evidence/**` only.

Commands:

```
/tmp/pab-0c-venv/bin/python -m pytest tests/service/test_dispatcher.py tests/service/test_core.py -q
W2D_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-worker-w2d.sh
python3 -m py_compile service/core.py service/dispatcher.py service/dispatch_store.py tests/service/test_core.py tests/service/test_dispatcher.py tests/release/test_dispatcher.py
bash -n scripts/test-worker-w2d.sh
git diff --check
```

The focused run before the final queued-cancel test was **55 passed in 27.02s**;
the final Linux suite below supersedes it. The runner holds the shared
`/tmp/pab-worker-w1-docker-lane.lock` for the entire offline build/test/install/
fault/cleanup sequence, asserts the existing dependency image SHA before build,
and mounts only task-owned named volumes. Existing dependencies are overlaid;
there is no install/download or fresh-host reproducibility claim.

Retained command history, stage files and failed publication intents have no
long-term quota/retention policy in W2d. R1/P1 must close disk admission and
operator reconciliation before activation. Repeated unknown observations never
release ownership or authorize cleanup, even after the nominal deadline. P1
operator death-proof recovery is intentionally absent. No architectural blocker
has been demonstrated for this fixed diagnostic dispatcher, but this handoff is
not independent acceptance or a full-W2/Phase-1A completion claim.

`intermediate-4.json` records the passing 122-test/32-scenario version before the
final idempotent-lookup ordering correction. In the final dispatcher a repeated
fixed request key returns the existing durable job even during startup/unknown
quarantine. Only fresh admission is gated. Both E1 and the actual PID-1 restart
E2 case assert this distinction; neither path re-resolves a plan or starts work.

## Final-source handoff

Final frozen command exited **0**: **122 passed in 38.64s**, then all **32**
installed scenario steps passed. Read/events/cancel under the live descendant
fixture took **0.003133945050649345 seconds**, below the declared one-second
bounded diagnostic ceiling. This is not a universal latency guarantee.

Final worker image: `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b`
(the identical accepted W2c worker). Final control image:
`sha256:49e338a4a376bd3f9612c6d6ace24fd52e44fade12ae143d1abee73b157699db`.
Existing dependency image: `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.

`run.json` contains exact commands/results, source/image/packaging identities,
Core job states, generation/attempt tuples, receipt-backed outcomes and fault
results. `final-audit.json` rehashes every supplied source and packaging file:
zero mismatches. All prior container/volume IDs were preserved; zero owned
containers or volumes remain. Images are retained for independent review.
The static/syntax checks and `git diff --check` also pass. Protected contracts,
studio/clients, legacy paths, ledger, reviews and historical evidence are unchanged.

The local commit containing this report is the review candidate; exact SHA is
returned with the handoff. **Implemented and locally tested; awaiting separate
Sol-low review. No self-GO, full-W2 integration acceptance, R1/P1 activation,
Phase 1A completion, power-loss, fresh-install, engine/hardware or acoustic claim.**
No subagents, extra checkout, model/private-audio/cloud/STT/Voicebox access,
remote E2E, push, PR or deployment. `scripts/run-e2e-test.sh` was never run.
