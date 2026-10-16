# W1 independent Sol-low review — R1

2026-10-16. **GO for W1 only**, exact candidate `30afe884ebfa509113cc2b9872b044b11f1b1cf9`, branch `t3/private-uds-worker-exchange`, base `f5d0b3b98218fef187138d779a6505143707d6d2`. Inspected read-only at `/Users/ben/.txcode/worktrees/py-audio-box/t3-2ae2fcb5`. No blocking functional finding identified. No prior W1 findings were supplied.

This approves the bounded isolated synthetic execute/result exchange for integration consideration. **No completed Phase 1A, production executor, fresh installation, real engine/hardware, browser, acoustic or global Astra acceptance.** W2 remains unopened; lifecycle/reconnect, durable dispatch, admission, registry, render and publication durability are not requirements silently added to W1.

## Inspection and correctness assessment

Read PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, canonical plan scope/shared architecture/Phase 1 and qualification requirements, CONTRACTS 1.3.0, the full `reviews/1A-foundation-astra-gate.md` including its W1 brief, and candidate `phase-1/WORKER-W1.md`. Inspected all added transport/worker/control source, both test files, runner, Dockerfile/ignore/Compose, complete changed-file inventory, canonical validator paths, successful evidence and retained first-run harness failure. The old usage failure supplies no passing evidence.

- `service/worker_transport.py:14-47`: authority serializes the complete allocation/context/allowlist/profile snapshot into immutable bytes. Accessors deserialize detached values, including nested profile/context data. The binding hashes those bytes; caller/accessor mutation does not retarget the session. Canonical 1.3.0 validators consume full plans and enforce tuple, identity, hash, deadline, required inputs and current synthetic admission state. Shared contracts are unchanged.
- `worker/transport.py:84-127`, `service/worker_transport.py:67-72`, `worker/isolated_synthetic.py:21-27`: both endpoints check OS UID/GID and prove the allocation capability using role-separated HMAC and fresh nonces. Control exposes peer evidence only after authentication returns. Passing the allocated owner to the oracle is justified by the authenticated capability binding, rather than a received JSON owner claim. PID zero across container namespaces is diagnostic, not lifecycle proof. This relies on trusted control provisioning and its private IPC volume; it does not claim resistance to compromised control/root/daemon or qualify reconnect identity.
- `worker/transport.py:24-65`: bounded length prefix precedes allocation/read; zero/oversized lengths fail, partial reads and EOF remain bounded, duplicate keys and malformed/recursive JSON fail closed. Each read/send uses the same absolute monotonic cutoff. `service/worker_transport.py:87-94` checks the cutoff again after result validation and returns uncertainty for disconnect/timeout without inventing cancellation, completion, replacement or retry. Parser/protocol refusals carry no accepted message.
- `service/worker_transport.py:73-90`: a result is a bounded proposal with canonical message/manifest checks and allocation relations, not byte verification or publication. The control harness at `tests/release/test_worker_boundary.py:33-62` independently opens staged bytes without following the output symlink, checks type/link count/owner/size, measures WAV samples and hash/provenance, then calls the canonical publication oracle. It performs no DB mutation or adoption. Worker `isolated_synthetic.py:30-63` is restricted to fixed synthetic preset output and exclusive attempt staging.
- `service/Dockerfile.w1:4-12` and `service/compose.w1.yml:35-50`: worker image contains worker/contracts rather than service code; actual filesystem hash checks confirmed that exclusion. Runtime inspection confirmed distinct nonroot identities, network none, read-only root, dropped capabilities, read-only IPC and one private attempt staging volume. No host/user data, DB/publication root, credential or Docker socket mount exists; required input set is empty. The session capability is intentionally provisioned execution authority, not an application credential. Only the disposable init performs ownership setup.

The adapter is dormant: source search found no production/Core caller. Existing foundation/legacy/core/contracts/lock files have no delta from the supplied base. No reviewer fixes were made.

## Independently executed evidence

Candidate HEAD matched the requested SHA and candidate status was clean before and after checks. `git diff --check f5d0b3b HEAD` passed. Independently recomputed every recorded control-source hash and all nine owned code/packaging/runner hashes against checkout bytes before runtime work.

Existing images were present and inspected:

- Control `sha256:e32aba8060da69b34bf32f19321b481be33840d853b148c95fd6b0f34330749d`.
- Worker `sha256:91c9171b80168275509b644be0e8b165c85040e160c61544caed15587ee1cc76`.
- Runner verified dependency image `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.

To preserve tracked evidence and avoid a needless build/download, exported the committed candidate into a disposable diagnostic directory and ran its inspected existing runner with only diagnostic-copy changes: use these exact existing image IDs and skip the build loop. All source-map comparisons, tests, scenarios, topology inspections, lane lock and termination/cleanup checks remained enabled. Command: `W1_PYTHON=/tmp/pab-0c-venv/bin/python bash <diagnostic-copy>/scripts/test-worker-w1.sh`. Exit **0**.

Actual Linux command: `python -m pytest tests/service tests/release -q -p no:cacheprovider` in the exact-source control image, network none/read-only root, bounded CPU/RAM/PIDs/tmpfs: **68 passed in 23.61s**. Separate-container success, wrong-session and authenticated wrong-execute-tuple scenarios all passed. Success verified 9644 synthetic WAV bytes, SHA256 `639dad0ac2923f5fe9e9ccfb99aa9b3084048e2e53317d1903e6e899a4f6296a`. All three scenarios confirmed termination before volume removal. No owned containers/volumes remained; both pre-existing inventories were preserved. No user data mounts, package installation, model/audio acquisition, external service, paid/remote E2E, prune or unrelated resource cleanup.

Diagnostic runner and full independent receipt: `/tmp/pab-w1-sol-review-_5887v8z/scripts/test-worker-w1.sh` and `/tmp/pab-w1-sol-review-_5887v8z/w1-evidence/run.json`. The diagnostic export was initially created under the host temporary-directory default and relocated to this `/tmp` path after execution; candidate files/evidence were untouched.

The handoff's host **50 passed / 18 Linux skips** is attributed, not rerun here. Its supplemental contract suite missing numpy/soundfile is **failed evidence, not a passing suite**; no fresh supplemental contract pass is claimed. Canonical contract bytes are unchanged and the independently passing affected Linux suite exercises those validators. This is E1/E2 dependency-overlay evidence, not a fresh dependency install/reproducibility or broad legacy regression qualification.

Only this review report was written in the root repository. No source, ledger, contracts, lock, commit, PR, push, deployment or subagent action. Final integrated verification and later global Astra review remain separate.
