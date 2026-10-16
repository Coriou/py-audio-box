# W2c packaged synthetic lifecycle — implementation candidate

Base 8b64f70ef5dd0483b8b8430a976945b52d1cceeb; assigned worktree
/Users/ben/.txcode/worktrees/py-audio-box/t3-785fa36e,
branch t3/implement-w2c-runtime. Contracts remain 1.3.0. No self-approval.

## Pre-code interface freeze

Private lifecycle-v1 uses bounded W1 length framing (65536 bytes), Linux mutual
UID/GID checks and nonce HMAC, with a derived domain-separated lifecycle key.
Only hello/prepare/start/observe/stop/collect/ack exist. Each request is exactly
{version,op,body}. Allocation calls bind the immutable W1 snapshot/digest;
subsequent calls carry supervisor-generated installation/lane/boot/nonce identity.
No client PID, observation, signal, path or argv is accepted. Each connection has
an independent two-second communication budget; collection never extends execution.

Control commits OwnerStore intent before prepare, binds before start. The trusted
PID-1 main loop reserves and fsyncs before prepare ACK, fsyncs start fence before
fork, anchors the original absolute deadline before launch, and owns allocation
forks, signals and waitpid-reap. A dedicated watchdog thread
performs TERM/KILL independently of management/journal I/O; the main PID-1 loop
reaps and commits receipts. Request handlers cannot run child computation or defer
the deadline loop. TERM grace is 200ms, then repeated namespace-wide KILL; only
ECHILD from PID1 after drain permits a durable terminal receipt. Process groups
are not namespaces. There are no unrelated resident processes in this worker.

Journal: supervisor-only /receipts (0700 root), exclusive lane flock, checksummed
envelope, atomic
replace + file fsync + directory fsync, complete state bounded to 1 MiB. At most
8 unacknowledged allocations, 64 total identities; refuse capacity rather than
evict unresolved evidence. Ack requires a durable terminal receipt, removes its
proposal only, retains an identity tombstone and never deletes staging. Journal
missing/corrupt after initialization fails closed; unresolved prior boots remain
unknown. A fresh volume must have init-created journal. Old durable terminal
receipts replay only for their exact identities.

IPC /ipc is root:10001 0750; lifecycle key root:10001 0640, socket 0660.
Engine UID/GID are allocation-specific (20000+index). A short trusted setup child
creates stage owned by control UID 10001 and allocation GID, mode 0770, under
root-owned 01733 staging; engine output is 0644 inside this protected directory.
Other attempts have different groups. Engine gets no supplementary groups, zero
capabilities/no-new-privileges, closed descriptors except its internal socket.
Its ephemeral internal execution key is separate from the lifecycle capability.
Neither DB nor publication volumes are mounted in worker. Supervisor imports no
service or engine modules. Synthetic generation runs in the fixed child entrypoint.

Receipt records exact authority binding/runtime identity, actual reap, original
proposal, and timely only when both full proposal and ECHILD precede cutoff.
Receipt failure leaves unknown and blocks admission. Stop ACK is not cancellation.
Fixed diagnostic fixtures only; no public dispatcher or publication integration.

## Evidence and limits

Implementation and required matrix results will be recorded below. This lot cannot
claim fresh installation, power-loss durability, hostile-plugin isolation, universal
latency, engines/hardware/audio quality, W2d/R1/P1 or production activation.

## Implemented interface details

`/ipc/lifecycle.sock` serves only lifecycle-v1. `/ipc/worker.sock` preserves the
accepted W1 execute/result framing and 1.3.0 messages, using a distinct execution
key and per-allocation derivation. It returns the original proposal; it cannot
launch a process or authorize lifecycle operations. Its original monotonic cutoff
is retained in the same boot. Following restart, old results are available only
through receipt-bound lifecycle collection. The child has only an ephemeral
internal W1 channel; neither installed capability reaches it.

Staging parent mode is **01733** (sticky), not plain 0733: engine children cannot
rename other control-owned attempt directories. Stages are 0770 control-owned,
with unique engine group; output is 0644 within that protected directory. The
permissions fixture checks both cross-attempt read and rename denial, both keys,
receipt/DB/publication exclusions, management connection denial, signal denial,
zero permitted/effective capabilities, no-new-privileges and descriptor closure.

Shipped command: `python -m service.lifecycle_cli` with typed actions
`hello`, `dispatch`, `start`, `execute`, `observe`, `stop`, `collect`, `ack`.
`dispatch` creates only the fixed synthetic authority, and requires the previous
receipt to have been durably recorded and acknowledged before a new allocation.
`start` is explicit diagnostic continuation of a bound, same-boot reservation,
never automatic restart recovery. `execute` requests the cached W1 proposal;
`collect` retrieves its protected receipt under a new communication budget.
Fixed fixtures are silence, permissions, busy and descendants. Typed diagnostic
interrupt points exit 17; no arbitrary plan, argv, signal or path is accepted.

The supervisor alone owns deadline signals and waitpid. A request handler cannot
extend its cutoff. An unreaped stop becomes unknown after one second while the
parent continues draining; this is not a finite-kill guarantee. ECHILD plus a
durable terminal receipt are both required for STOPPED. A receipt-write failure
leaves the allocation unknown even if the actual child has been reaped.

## Qualification command and evidence levels

`W2C_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-worker-w2c.sh`
serializes the existing Docker lane, builds offline from the existing pinned
image, verifies supplied source/image identities, runs service/release regressions,
and drives the installed CLI and UDS runtime. No host observer supplies runtime
state to control. Host Docker calls only install, inject faults, audit, and clean
owned resources after termination. No normal service has a Docker executable,
socket, credential or host observation environment.

The ENOSPC fence case is deterministic **E1 injection at the real journal open**;
it proves no Popen after failed fence persistence. Packaged **E2** uses actual
filesystem permission failure for start-fence and terminal-receipt writes. These
are distinct tests, not a claim that the Docker host filesystem was filled.
Process kills/restarts are not power-loss evidence. `failure-*.json` retain bounded summaries of failed
runs; `intermediate-*.json` belong only to their recorded source/image hashes.
Only the final `run.json` and final source audit apply to the final candidate.

## Exact evidence coverage

E1 deterministic checks cover closed/domain-separated frames, missing/corrupt and
checksummed/oversized journal refusal, exclusive writer, intent/prepare/bind/start
interrupt points, same reservation replay without cutoff reset, new-boot quarantine,
ENOSPC at start-fence persistence without launch, receipt capacity without eviction,
ack tombstones, unreaped-stop unknown, and watchdog TERM/KILL independent of request
I/O. Existing W1/W2 transport, lifecycle and release regressions run in the same
network-disabled exact-source control image. The inherited presend/postsend tests
remain E1 canonical-execute interruption evidence, not separate Docker crashes.

The packaged E2 runner covers:

- Two distinct sequential silence/permissions allocations; real W1 UDS execute,
  lifecycle collect/verify/ack; identical proposal hashes and bytes on replay,
  including collection after cutoff. Old IDs cannot execute again.
- Actual control exit 17 after start ACK and immediately after start-frame send;
  new controls observe/collect the same allocation with exactly one physical child.
- Busy and TERM-ignoring double-fork/setsid children, independently audited live
  in their actual sessions before the test proceeds. With controls absent, original
  deadline enforcement kills them; trusted PID-1 receipts count every reap.
- Blocked client plus busy descendants, bounded readiness response, explicit
  TERM/grace/KILL stop, and lost stop response followed by actual observed reap.
- PID-1 KILL/container restart during execution: changed boot, original unknown,
  zero replayed launches, fresh admission refused. Restart after durable terminal
  receipt replays that exact old receipt/proposal only.
- Actual UID/capability/descriptor and directory probes; separate PID/net/mount
  namespaces, worker/control mount exclusions, no Docker executable; wrong tuple,
  wrong session/domain, oversized and incomplete management frames.
- Actual unwritable start fence (zero launches), unwritable terminal receipt
  (unknown, ack refused), and actual missing/corrupt journal restart refusal.
- Exact supplied source/image/topology hashes and confirmed owned-resource cleanup;
  prior inventory preservation checked by IDs, recorded only as counts/digests.

These do not test an uninterruptible kernel process or impose a universal latency
SLA. The declared readiness ceiling is five seconds for the bounded synthetic
scenario, with the measured value in run.json. The independent watchdog signals
without journal I/O; blocked OS reaping or persistence still withholds STOPPED.
No inference/hardware, acoustic, browser, production API or power-loss claim.

## Failed runs retained

- failure-1: harness expected CAP_ALL spelling; Docker returned ALL.
- failure-2: runtime ESRCH after leader reap escaped stop_tree; fixed by accepting
  ESRCH only for signalling and still requiring ECHILD plus durable receipt.
- failure-3: small tmpfs receipt-volume experiment lost initialization contents
  when the last init mount closed; reverted to ordinary isolated named volumes.
- failure-4: incomplete-frame client kept the authentication socket timeout and
  raced the server close; corrected audit receive window remains bounded.
- failure-5: harness used dispatch for a completed, unacknowledged allocation;
  the new ack guard correctly refused it. Reconnect now uses observe.
- failure-6: an in-progress run mixed the earlier image with a changed Compose
  journal initializer during reset; invalidated, retained, then rerun from frozen
  matching source/packaging. It supplies no final passing evidence.

No failure above is hidden or promoted to final-source acceptance. No additional
file allowance was needed; existing core/API/OwnerStore/contracts/plan/ledger,
legacy artifacts, historical evidence and reviews are unchanged.

## Final-source verification and handoff

Final command above exited **0**: **100 passed in 25.48s**, then all **15 packaged
E2 scenarios passed**. Readiness response under the blocked-client/descendant
fixture was **0.8987852509599179 seconds**, below the declared five-second ceiling.
Live audits independently found three TERM-ignoring processes across two sessions;
the deadline descendant receipt records **three PID-1 waitpid reaps**, followed by
ECHILD. Two sequential normal proposals independently verified the original 9644
WAV bytes, SHA256
`639dad0ac2923f5fe9e9ccfb99aa9b3084048e2e53317d1903e6e899a4f6296a`.

Final images (retained for independent review):

- Worker `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b`.
- Control `sha256:62b9d27609a63975ffba5d0c7d321ecd6df085e1dfd843bce162f2ebdb34e9b2`.
- Existing dependency `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.

`run.json` records commands, exact source maps, image IDs, namespace/topology
assertions, live-tree audits, scenarios and cleanup. `final-audit.json` rehashes
all supplied sources and packaging against the final run. All prior container and
volume IDs were preserved; zero owned containers/volumes remain. No prune or
unrelated-resource stop/removal occurred. Historical receipts are sanitized and
attributed to their own source/image versions; no capability values or audio
payloads are recorded.

`python3 -m py_compile worker/runtime*.py service/lifecycle_cli.py
service/supervisor_runtime.py tests/service/test_packaged_lifecycle.py
tests/release/test_packaged_lifecycle.py`, `bash -n scripts/test-worker-w2c.sh`,
and staged `git diff --check` pass. The local commit containing this report is the
candidate; its exact SHA is returned in the handoff. This is **implemented and
locally tested, awaiting independent Sol-low review**, with no self-GO.

Changed implementation paths are exactly the four worker runtime modules, two
service modules, three lifecycle packaging files, two packaged-lifecycle test
files, the W2c script, this report and w2c-evidence. No existing source file or
schema needed modification. Private lifecycle-v1/journal are new internal
interfaces; canonical execute/result/manifest remain 1.3.0.

No architectural blocker remains demonstrated by this bounded fixture. Full W2
still waits for W2d; R1/P1, operational quarantine recovery, resident mode, real
engines, accelerator qualification, registry/render and production activation
remain outside this candidate. The one-row OwnerStore is not an allocator or
history. Retention refuses admission after eight unacknowledged receipts, 64 total
identities or the 1 MiB journal bound; it does not invent a long-term cleanup policy.
No complete legacy/real-model suite, fresh install or power-loss claim is made.
