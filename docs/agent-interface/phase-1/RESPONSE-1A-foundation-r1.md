# Foundation functional corrections — F1/F2 response

2026-10-16. Implementation candidate awaiting independent Sol-low review; no self-GO.
The earlier provider-failed review produced no report. The complete R1b Sol report
is the functional NO-GO being answered; its two findings are retained here.

## Identity and scope

Initial caller checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-8ee9a294`
was clean at `eb74643`. Created a clean isolated worktree with:

```sh
git worktree add -b fix/1a-foundation-f1-f2 /tmp/pab-1a-f1-f2 HEAD
git -C /tmp/pab-1a-f1-f2 status --porcelain
git -C /tmp/pab-1a-f1-f2 cherry-pick a1a46685f7d2cc5be22e84ca5693e04bb5d30779
```

Empty status before cherry-pick; resulting foundation commit
`f6386b75edf7a9f125a7b78493fe3faca5323708`. All corrections/tests run in that tree.
The corrective candidate SHA is the commit containing this response (reported
exactly in the final handoff; `git rev-parse HEAD` after checkout).

Correction inventory: `service/core.py`, `tests/service/test_core.py`,
`docs/agent-interface/phase-1/CORE-1A.md`, and this response. No contract, domain,
worker output/protocol, legacy, client, studio, ledger, review, dependency or
packaged wiring changes. Contract pack remains 1.3.0.

Read PRODUCT, implementation start, orchestration protocol, canonical plan shared
contracts/Phase 1A, CONTRACTS 1.3.0, CORE-1A and full R1b review. Inspected
`scripts/run-unit-tests.sh`, `scripts/test-skeleton-0c.sh` and
`service/Dockerfile.release` before choosing bounded commands.

## F1 — control-observed deadline independent of collection

Added one control-owned completion observer per spawned process. It starts before
execute transfer and waits/reaps independently of `finish()`. A monotonic cutoff
is anchored to the persisted deadline. Only completion observed strictly before
cutoff is eligible for publication; at/after cutoff is expired, even if wait
returns immediately for an already-exited child. A live expired worker is killed
and waited for. `finish()` joins the observer and refuses unconfirmed reap while
retaining the active lane. SQLite/publication remain on the control caller thread.
Cancellation still requests stop and awaits reap; a timely verified completed
result may win cancellation. Existing cancellation/publication/recovery tests pass.

The control observation is conservative: scheduler delay across the cutoff can
reject an earlier physical exit for which control lacks timely evidence. No child
timestamp or output existence proves timely completion. Evidence is not persisted:
control death before publication intent still recovers as interrupted failure,
never adopts staged output on the strength of an inferred completion time.
Durable publication-intent recovery is unchanged.

Added functional tests for delay=2/deadline=1 with no finish call until the observer
has killed/reaped; an actual already-exited delay=1.2/deadline=1 worker with a
deliberately delayed observer; demonstrably timely completion collected after the
deadline; deterministic strictly-before/equal/after control-observation edges; and
restart after timely observation but before publication, which fails closed.
Existing immediate deadline kill/reap, live-owner restart quarantine and all three
publication-death hooks remain covered. Synthetic silence/local temporary data only.

## F2 — exact canonical retry context and stored hash

Retry now compares canonical serialized entire contexts and requires the prior
stored job input hash to equal both the prior stored plan hash and submitted plan
hash. Python numeric equality no longer authorizes changed canonical preimages.

The actual domain test copies the failed/cancelled job context and converts its
integer queue limit to float. The existing oracle accepts its frozen-plan
relations, Python equality remains true, and plan hashes differ; retry rejects
`invalid_retry` without changing the old job/hash or inserting a job. An exact
copied context retry creates a distinct job, preserves input hash and durable
`retry_of`. A separate synthetic inconsistent-store test proves the old stored
hash is checked even when contexts match. Original cancellation/retry test passes.

## Commands and results

Host (existing local venv, no dependency installation):

```sh
/tmp/pab-0c-venv/bin/python -m pytest tests/service/test_core.py -q
/tmp/pab-0c-venv/bin/python -m pytest tests/service tests/release -q
git diff --check
```

Final focused: **34 passed in 11.25s**. Full service/release: **38 passed in
15.91s**. Earlier focused intermediate run: 33 passed in 9.93s, before the
already-exited late-worker test was added. Whitespace check passed.

Attempted an offline release test image rebuild under a 600-second subprocess
bound:

```sh
docker build --network=none --memory 2g -f service/Dockerfile.release --target tests -t speech-core-1a-tests:f1-f2 .
```

Exit 1: an uncached `pip install` layer could not resolve PyPI with networking
disabled (`annotated-doc==0.0.5`). No dependency/release Dockerfile changes were
made. This is not a successful fresh release-image build.

Instead built a source-overlay image, bounded to 120 seconds:

```sh
docker build --network=none --memory 2g -f /tmp/pab-1a-f1-f2.Dockerfile -t speech-core-1a-tests:f1-f2 .
```

Temporary Dockerfile exactly:

```dockerfile
FROM speech-core-1a-tests:21d5add2
USER root
COPY service ./service
COPY worker ./worker
COPY contracts ./contracts
COPY clients ./clients
COPY tests/service ./tests/service
COPY tests/release ./tests/release
USER 10001:10001
```

Existing dependency base image:
`sha256:eee41d4a98bd845d9857e3628cfcce08c898ab2f95dc421d6aede92379e2c44f`.
Final copied-source image:
`sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.
Build exit 0, no package installation/download. Sequential sole container lane.

`python3 /tmp/pab-1a-f1-f2-container.py` computed SHA256 maps for every file under
service, worker, contracts, clients, tests/service and tests/release, excluding
bytecode. An identically bounded `docker run --name pab-1a-f1-f2-verify -i ...
python -` computed the image-side map; exact dictionary comparison: **53 files,
zero missing/extra/hash mismatches**. No source or user-data mounts.

Final test command (120-second outer subprocess bound):

```sh
docker run --rm --network none --cpus 2 --memory 1g --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,size=64m --name pab-1a-f1-f2-tests speech-core-1a-tests:f1-f2
```

Inherited CMD: `python -m pytest tests/service tests/release -q -p no:cacheprovider`.
**38 passed in 21.91s**, exit 0. Verification container bound 30 seconds. Both
names inspected absent afterward; finally cleanup would remove only those owned
containers if needed. No task volumes or host/user-data mounts; no prune or
unrelated-container cleanup. Source verification covers executable/test source;
subsequent handoff-document edits do not change the tested bytes.

## Retained limits and next step

This is still the partial trusted synthetic single-preset subprocess foundation.
No full Phase 1 expansion, isolated worker-container execution, complete
admission/resource policy, registry, render, mutation idempotency/full history,
persistent deployment wiring, domain error transport mapping or four-client
integration is established. Spawn/reap/result-verification crash windows and full
disk reconciliation remain unfinished. The observer is not an arbitrary process
tree supervisor. Synthetic readiness bypass is not real admission. Existing
isolation/admission/render/registry gates remain mandatory, as do installed,
legacy, hardware, browser and acoustic acceptance. No performance improvement or
resource benchmark claim; only test wall times above.

No models, cloud, private audio, STT, Voicebox, Compose, push, PR, deployment,
subagents, independent security/external probing, or `scripts/run-e2e-test.sh`.
No fresh legacy suite or full release build acceptance is claimed. Independent
Sol-low functional review of the exact committed candidate and these responses
is next, followed by integration verification under the parent orchestrator.
