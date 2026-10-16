# 1A foundation functional review — R1b Sol

2026-10-16. **NO-GO for this foundation candidate pending the two functional corrections below.** This is not a verdict on completed Phase 1A: full 1A remains incomplete regardless of these corrections.

Reviewed SHA: `a1a46685f7d2cc5be22e84ca5693e04bb5d30779`.
Read-only candidate: `/Users/ben/.txcode/worktrees/py-audio-box/t3-21d5add2` (clean before and after inspection/tests). Reviewed the five-file commit against parent `9cca8a59c53810fb9c3e49ba6a9582374cb0ef25`: `service/core.py`, `service/domain.py`, `worker/synthetic.py`, `tests/service/test_core.py`, and `docs/agent-interface/phase-1/CORE-1A.md`. Read PRODUCT, IMPLEMENTATION-START, canonical IMPLEMENTATION-PLAN including Phase 1A/shared invariants, ORCHESTRATION-PROTOCOL, and phase-0/CONTRACTS; inspected relevant contract validators and test entry points. Reviewer: Codex gpt-6.1-sol, low effort. No source repairs, commits, ledger edits, or additional agents.

## Blocking functional findings

### F1 — P2: Deadline enforcement depends on when finish is called

Location: `service/core.py:185`, `service/core.py:218-240`; `worker/synthetic.py:33-45`.

`start()` persists a deadline but installs no deadline observer. The child sleeps and produces its result without checking the deadline. `finish()` initializes `expired=False`, then only sets it when `process.wait(timeout=remaining)` raises. An already-exited child returns immediately even when the persisted deadline is in the past. Its result therefore reaches `_publish()` and can succeed.

Concrete ordinary call sequence: use the existing fixture with deadline_seconds=1, submit, start(delay=2), let the control caller do other work for more than two seconds, then finish. The child completes after the one-second deadline; finish sees an exited child, never sets expired, and publishes it. Worker-message/publication validation checks the bound deadline value, not a trustworthy completion time. This contradicts the handoff's deadline-expiry-fails guarantee and makes execution ceilings depend on caller scheduling. This is a source-derived example, not an additional executed test.

Required correction: enforce expiry independently of delayed result collection, or retain trustworthy control-observed completion/deadline evidence and reject late completion. Preserve successful results demonstrably completed within deadline if collected later, and preserve stop/reap semantics. Verification should cover both late completion collected later and timely completion collected later. Existing `tests/service/test_core.py:222-227` immediately calls finish while the child is alive, so it exercises only the wait-timeout branch.

### F2 — P2: Explicit retry accepts a canonically changed frozen context

Location: `service/core.py:164-169`; canonical-number rule in `docs/agent-interface/phase-0/CONTRACTS.md` and `contracts/validate.py:345-347`.

Retry identity uses `previous == context`, Python dictionary equality. Canonical JSON preimages distinguish 1 from 1.0; Python equality does not. For example, after a failed/cancelled job, copy its context and replace an integer-valued profile limit such as queue=32 with queue=32.0. JSON Schema integer validation accepts an integral JSON number, the profile relationships remain equal, and the retry comparison passes. Yet sorted compact JSON and the full plan hash differ. The new retry job receives that different input_sha256 at line 169 while retaining retry_of lineage to the old job. This violates the foundation's identical-context retry guarantee and the explicit canonical-number rule. This is a reasoned source-level example; no custom probe was run.

Required correction: compare canonical serialized contexts (or equivalent exact canonical identities), including the prior job's frozen plan hash, rather than Python value equality. Verification should reject numerically equal but canonically different contexts and preserve the existing identical-context retry case (`tests/service/test_core.py:67-76`). No contract/schema change is needed.

## Tests actually run

Inspected `scripts/run-unit-tests.sh`, `scripts/test-skeleton-0c.sh`, and `service/Dockerfile.release`; did not invoke either build/smoke script. Reused existing test image `speech-core-1a-tests:21d5add2`, image SHA `sha256:eee41d4a98bd845d9857e3628cfcce08c898ab2f95dc421d6aede92379e2c44f`.

Before testing, a network-disabled, read-only, mount-free container compared SHA256 of all 53 candidate files under service, worker, contracts, clients, tests/service and tests/release (excluding bytecode) with the corresponding copied image files: **zero mismatches**. No image build/download was performed.

Executed, with a 120-second outer subprocess timeout and named-container cleanup:

```sh
docker run --name pab-review-1a-r1b-sol --rm --network none --cpus 2 --memory 1g --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,size=64m speech-core-1a-tests:21d5add2
```

Image CMD: `python -m pytest tests/service tests/release -q -p no:cacheprovider`.
Result: **29 passed in 21.69s**, container exit 0. Named container was absent afterward. No host/user-data mounts. Also ran `git diff --check HEAD^ HEAD`: passed. No other test suites or custom fault/security probes were executed.

## Coverage assessment and retained limits

The existing suite supplies useful bounded evidence for SQLite persistence, principal/key duplicate handling, changed-request conflicts, stale-revision rejection and historical snapshots, frozen input copies, one owner/lane, queued and active cancellation, verified completion winning cancellation, explicit retry lineage, stale results, media verification, ENOSPC refusal, publication-boundary process death, and live-owner restart quarantine. Source inspection confirms short immediate CAS transactions, separately persisted contexts, durable publication intent followed by file adoption and DB reference/success commit, and managed-ID retrieval checks. Those positives do not close F1/F2.

CAS coverage is sequential stale-revision rejection, not concurrent independent service clients. Publication recovery covers the three explicit hooks; ENOSPC is injected at adoption-file open, not every filesystem/SQLite operation. Spawn/reap/result-verification crash windows and complete disk reconciliation are explicitly unfinished in the handoff. No independent security, exploitation, external-access, or additional fault probing was attempted; this report does not claim full adversarial coverage. The prior provider-flagged review supplied no findings to validate or discard.

This candidate remains a trusted synthetic single-preset subprocess foundation. It does not establish isolated worker-container execution, complete resource policy, voice/version registry, project mutation idempotency/full history workflows, render, persistent deployment wiring, transport domain-error mapping, four-client integration, installed/browser acceptance, legacy regression acceptance, real model/hardware qualification, or acoustic acceptance. Synthetic readiness bypass is not real readiness. All original product and full Phase 1A acceptance requirements remain in force, with no waiver or self-implementation. No remote services, models, private audio, cloud, deployment, PRs, or scripts/run-e2e-test.sh were used.
