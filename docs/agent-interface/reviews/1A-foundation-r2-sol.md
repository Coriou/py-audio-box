# 1A foundation functional re-review — R2 Sol

2026-10-16. **GO for the bounded trusted synthetic foundation at `873dae231a5eacf663a2d36c0fe23de62a2db720`. F1 and F2 are closed.** No additional blocking ordinary functional finding was identified within this review. This is not full Phase 1A acceptance, a security/adversarial approval, or final Astra global approval.

## Identity and scope

Reviewer: independent Codex gpt-6.1-sol, low effort. Candidate read-only location: `/tmp/pab-1a-f1-f2`, branch `fix/1a-foundation-f1-f2`; HEAD verified before and after tests. This is the implementer's manually created temporary worktree, not the initially assigned T3 worktree. No new worktree or branch was created by this reviewer.

Read PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, canonical IMPLEMENTATION-PLAN shared contracts and Phase 1A, CONTRACTS 1.3.0, CORE-1A, RESPONSE-1A-foundation-r1 and the complete reviews/1A-foundation-r1b-sol.md. Inspected the actual correction against cherry-picked foundation `f6386b75edf7a9f125a7b78493fe3faca5323708` and the original reviewed foundation `a1a46685f7d2cc5be22e84ca5693e04bb5d30779`, plus current core, synthetic worker, domain export, contract hashing/message/publication validators, callers and existing tests. Correction changes four files: core, core tests and two handoff documents. Worker/model/protocol/contracts/domain export remain unchanged. The original foundation was partial and trusted; this review preserves that scope.

Source and candidate remained unmodified. The only review deliverable is this file in the root checkout; no source, ledger, commit, branch or research changes.

## F1 — closed

`service/core.py:42-74,224-238,253-281`: start anchors the persisted wall deadline once to a monotonic cutoff and installs the observer before execute transfer. The observer independently waits, treats observation at or after cutoff as expired even for an already-exited child, kills a still-live expired child and confirms reap. finish joins it before reading/publication. The join establishes completed observer state before use; no SQLite or publication work occurs on the observer thread. Later wall-clock changes do not change the established cutoff. No child timestamp supplies completion authority.

A delayed finish can publish a timely observed result; it cannot convert late completion into success. If collect cannot establish a finished observer and reap, it raises before finish's ownership-clearing finally, preserving the active lane. An unexpected observer failure likewise leaves reaped false rather than creating eligible publication evidence. Cancellation still requests termination and shares the same Popen lifecycle; the observer owns completion evidence, while control serializes domain transitions. A timely verified completed result can still win cancel_requested. This does not claim concurrent multithreaded Core API support: SQLite/domain ownership remains on the control caller thread.

The executed tests meaningfully cover the original failure: `tests/service/test_core.py:230-242` lets the independent observer expire/kill/reap without finish; `:328-344` deliberately delays observation until a real late worker has already exited. `:245-257` proves timely observation followed by collection after deadline succeeds. The before/equal/after boundary test at `:260-270` is a deterministic observer unit test with a stub process, not standalone process-timing proof. The real subprocess cases complement it. `:304-325` proves loss of in-memory timely evidence on control death fails closed without publication intent. Existing cancel/reap, completion-winning-cancel, live-owner quarantine and all three publication-death tests pass.

Recovery still commits only the existing verified durable publication intent and adopted bytes; otherwise it fails interrupted work after owner-death evidence. Conservative scheduling can reject an earlier physical exit that was not observed before cutoff; this is explicit policy, not an inferred trustworthy exit timestamp. Wall-clock adjustment during initial deadline creation/anchor sampling was not independently simulated. No general process-tree supervision or complete spawn/reap crash-window qualification is implied.

## F2 — closed

`service/core.py:201-205`: retry compares canonical serialization of the entire stored/submitted context and verifies the old input hash against both plans. encode uses the same sorted-key, compact, finite UTF-8 JSON convention as canonical_hash. Thus integer 32 and float 32.0 cannot become a changed-hash retry under Python value equality. Duplicate idempotency lookup still returns the original job before re-resolution; explicit retry remains a distinct job/key with durable lineage.

`tests/service/test_core.py:273-289` reaches actual domain admission: it first demonstrates Python equality, successful frozen-plan validation and unequal plan hashes, then expects invalid_retry, verifies no insertion/old-job mutation and checks an exact-context retry retains hash and retry_of. It does not pass merely because an unrelated schema rejects the float. `:292-301` separately exercises inconsistent prior stored hash rejection. The existing real cancellation/new-retry/success lifecycle also passes. No exception shortcut replaces these functional checks.

## Checks actually run

Before container tests, inspected the existing `/tmp/pab-1a-f1-f2-container.py` verifier and reused its logic in memory with review-owned container names. Verified the image ID first, then compared SHA256 dictionaries for all copied files under service, worker, contracts, clients, tests/service and tests/release, excluding bytecode: **53 files, zero missing/extra/hash mismatches**.

Exact image: `speech-core-1a-tests:f1-f2`, `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`. No image rebuild, download, dependency install or host/user-data mount.

Commands (candidate cwd `/tmp/pab-1a-f1-f2`):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest tests/service/test_core.py -q -p no:cacheprovider
docker run --rm --network none --cpus 2 --memory 1g --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,size=64m --name pab-review-1a-r2-sol-tests speech-core-1a-tests:f1-f2
git diff --check a1a4668 HEAD
```

Host focused: **34 passed in 12.03s**, exit 0. Container inherited CMD `python -m pytest tests/service tests/release -q -p no:cacheprovider`: **38 passed in 24.30s**, exit 0. Whitespace check passed. Verification used the same docker bounds with `--name pab-review-1a-r2-sol-verify -i IMAGE python -`, a 30-second outer timeout; test outer timeout was 120 seconds. Both review-owned containers were confirmed absent afterward. Candidate status remained clean.

The implementer's fresh offline release-build attempt failed at an uncached pip dependency layer with network disabled. That is a dependency/cache/network limitation, not a hardware failure or a successful fresh release build. The reused image is an exact-source overlay on existing pinned dependencies, not new release-build acceptance. Those earlier build claims were read from the response; no rebuild was repeated here.

## Evidence limits and remaining gates

E1: source/contract inspection, canonical retry and boundary tests, existing fixture/adapter checks. E2: existing local synthetic SQLite/socket/subprocess lifecycle, cancellation, deadline and publication/restart tests, rerun in the verified Linux test image. These do not demonstrate separately isolated worker-container execution or persistent-volume/power-loss durability. CAS tests use sequential stale-revision conflicts, not concurrent independent service clients. ENOSPC and publication death cover their existing explicit injection points, not all filesystem/SQLite/spawn/reap/verification windows. No new custom fault, security, exploitation or external-access probe was run.

Whole 1A remains incomplete: isolated execution/peer binding/reconnect, complete admission/resource/disk policy and reconciliation, registry/version workflows, project mutation idempotency/full history, render, persistent deployment wiring, bounded domain-error transport mapping and four-client integration remain outside this foundation approval. Synthetic readiness bypass does not establish actual admission. No fresh full legacy suite, E3 real model/release/hardware qualification, E4 browser/agent-host workflow, E5 acoustic acceptance or performance improvement is established. Required remaining adversarial coverage is not waived by this functional GO. Integration verification and separate final Astra global review remain required.

No models, cloud, private audio, STT, Voicebox, Compose, push, PR, deployment, subagents or scripts/run-e2e-test.sh were used. Root research and existing unrelated artifacts were preserved.
