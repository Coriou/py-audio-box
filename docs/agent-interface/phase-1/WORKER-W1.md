# W1 isolated synthetic execute/result candidate

2026-10-16. Implementation handoff, awaiting independent Sol-low review; no self-GO.
Base `f5d0b3b98218fef187138d779a6505143707d6d2`, app-owned checkout
`/Users/ben/.txcode/worktrees/py-audio-box/t3-2ae2fcb5`, branch
`t3/private-uds-worker-exchange`. Candidate is the commit containing this report.
Foundation remains PARTIAL. No durable dispatcher is connected.

## Interface and authority

Canonical worker messages/manifest/oracle remain **1.3.0**, unchanged. Internal
session interface is `w1-session/1`. `Authority(job, context, allowed_assets,
profiles)` stores canonical immutable bytes and returns detached snapshots.
Control supplies the frozen plan and job/attempt/generation/hash/owner/staging/
deadline tuple. A binding hashes the complete allocation, allowlist and profiles.
`Session(secret, binding, peer_uid, peer_gid)` pairs that allocation with a fresh
256-bit ephemeral session capability, provisioned by control on a private IPC
volume. The secret is never logged or included in canonical worker messages.

Both sides check Linux `SO_PEERCRED` UID/GID, then mutually prove possession of
the allocation capability with role-separated HMAC and fresh challenge nonces.
UID/GID checks and session proofs jointly establish the binding; JSON owner IDs
and filesystem permissions alone confer no authority. PID is diagnostic only
(and zero across these PID namespaces). This is a trusted local provisioning
boundary, not protection against a compromised control, root or Docker daemon.
No reconnect/replacement identity or production authentication is implied.

`exchange(connection, authority, session, cutoff)` returns `TransportObservation`
with `proposal|refused|uncertain`, authenticated peer evidence, bounded received
message/manifest and local monotonic receipt time. A proposal is receipt only.
It never writes DB, adopts/publishes, cancels, retries or establishes worker reap.
EOF/disconnect/timeout is uncertainty. Caller verifies actual staged bytes and
canonical publication relations separately, with no DB mutation.

Framing is four-byte unsigned network-order length plus 1..65536 JSON bytes.
Duplicate keys, nonfinite numbers, invalid JSON, oversized/zero/incomplete frames
are rejected or time out. One absolute monotonic cutoff covers authentication,
execute and result; no per-read budget reset. Closed canonical schemas validate
execute/result and manifest. Session metadata stays outside canonical messages.

## Isolated harness and scope

Only fixed synthetic preset speech, one variant, reuse=never, no required input
assets is supported. It writes 4800 mono PCM16 silence frames at 48 kHz. Fixture
Qwen identities and mock qualified profiles are explicit synthetic authority,
never real Qwen output, hardware readiness or engine qualification. The required
read-only input set is empty, so there is no asset mount.

Separate nonroot control/worker containers use network none, read-only roots,
dropped capabilities and no-new-privileges. Worker image copies only worker and
contract code, not service code. A dedicated private volume contains exactly one
allocated attempt stage; a separate IPC volume is read-only for worker. Distinct
UIDs share a private group, stage mode 0770, output mode 0640. No host/user-data,
DB, publication, Docker socket or application credential mounts exist. A one-shot
root init only sets ownership/modes on the two disposable volumes.

The runner serializes its Docker lane via a nonblocking local advisory lock,
uses random owned project/container names, snapshots existing container/volume
IDs, builds offline against an exact pre-existing dependency image, compares
actual image source hashes to checkout bytes, runs affected tests, and inspects
actual container mounts/users/network/images. It runs separate-container success,
wrong-session and authenticated wrong-execute-tuple cases. Only confirmed stopped
owned containers/volumes are removed; prior resources are preserved. Images are
retained for review; no prune. This dependency-overlay build is not a fresh
installation/reproducibility claim.

## Preservation and continuation

Salvaged only the nine enumerated code/packaging/test files from the failed
uncommitted checkout at `t3-ba67e485` (HEAD `03b0544`), read-only. No prior test
or candidate success was assumed. The original checkout/evidence was untouched.
This candidate additionally freezes authority against nested mutation, handles
parser recursion refusal, and adds actual separate-container refusal scenarios.
The first local run passed Linux tests and success exchange, then failed a harness
exit expectation on wrong-session disconnect. Its sanitized receipt is retained;
the corrected run has its own evidence. The prior usage-limit failure is not a
code finding.

Owned paths: `worker/transport.py`, `worker/isolated_synthetic.py`,
`service/worker_transport.py`, `tests/service/test_worker_transport.py`,
`tests/release/test_worker_boundary.py`, `service/compose.w1.yml`,
`service/Dockerfile.w1`, necessary `service/Dockerfile.w1.dockerignore`,
`scripts/test-worker-w1.sh`, this report and `w1-evidence/**`.
No shared core/migration/contracts/client/studio/legacy/ledger/review edits.
No wire gap required a schema fork for this slice.

W2 remains unopened: lifecycle/reconnect/quarantine, persistent deadline and
reap authority belong there. Resource admission, registry, renderer, real engines,
publication durability, complete Phase 1A and release acceptance remain open.
No models, private audio, remote paid E2E, cloud, STT, Voicebox, subagents, push,
PR or deployment. E1/E2 here cannot establish E3 hardware/engine, E4 browser/agent
workflow or E5 acoustic/product acceptance. Independent Sol-low review is next.

## Exact verification and limitations

- E1 host: `/tmp/pab-0c-venv/bin/python -m pytest tests/service/test_worker_transport.py tests/release/test_worker_boundary.py -q` — 12 passed, 18 Linux-only skipped (before final uncertainty-reporting correction).
- E1 host, final code: `/tmp/pab-0c-venv/bin/python -m pytest tests/service tests/release -q` — 50 passed, 18 Linux-only skipped, 24.67s.
- E1/E2 Linux, final code: `W1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-worker-w1.sh` — exit 0; 68 service/release tests passed, 37.40s; all three separate-container scenarios passed. Exact child commands, source maps, topology, termination and inventory checks are in `w1-evidence/run.json`.
- Supplemental host contract check: full `tests/contracts -q` stopped collection for missing `soundfile`; explicit `tests/contracts/test_contracts.py tests/contracts/test_corrections.py tests/contracts/test_frozen_boundary.py -q` produced 108 passed, 6 failed due to missing `numpy`, including generator subprocess. No dependency installation or shared-contract edits followed. Git verified no tracked changes outside owned paths. This supplemental suite is **not passing evidence**; the W1 and affected service/release gates above passed.
- `git diff --check` passed. Current checkout bytes equal the successful image source map; packaging/runner hashes are additionally in `w1-evidence/owned-source-sha256.json`.

Successful control image: `sha256:e32aba8060da69b34bf32f19321b481be33840d853b148c95fd6b0f34330749d`.
Successful worker image: `sha256:91c9171b80168275509b644be0e8b165c85040e160c61544caed15587ee1cc76`.
Pinned existing dependency image: `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.
All successful scenarios confirmed owned container termination before volume removal;
zero owned containers/volumes remain and both prior inventories were preserved.
Images remain available for independent review. Synthetic output is 9644 WAV bytes,
SHA256 `639dad0ac2923f5fe9e9ccfb99aa9b3084048e2e53317d1903e6e899a4f6296a`.
