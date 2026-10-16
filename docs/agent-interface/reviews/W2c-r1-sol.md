# W2c independent Sol-low functional/reliability review — R1

2026-10-16. **GO for W2c packaged fixed synthetic lifecycle only**, exact candidate `b166665f3415b45bd509994a23245a8beddd4ca4`, base `8b64f70ef5dd0483b8b8430a976945b52d1cceeb`, branch `t3/implement-w2c-runtime`, inspected read-only at `/Users/ben/.txcode/worktrees/py-audio-box/t3-785fa36e`. No blocking functional/reliability finding identified. No prior W2c findings were supplied. Reviewer: independent gpt-6.1-sol, low effort; no subagents.

**This is not full W2, Phase 1A, persistent/public execution activation or final global Astra acceptance.** W2d dispatcher/allocator integration and R1/P1 remain required. The runtime remains a synthetic diagnostic, with no production Core/API activation.

## Inspection and preservation

Read PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, canonical plan shared architecture/Phase 1A and evidence boundaries, CONTRACTS 1.3.0, the full foundation Astra gate and DEPLOYED-LIFECYCLE-ASTRA decision/mandatory exit matrix, W1/W2a/W2b reviews, and complete WORKER-W2C. Inspected all six new implementation modules, packaging, runner and both test files, inherited transport/Authority/OwnerStore/Supervisor/generator/verifier call paths, base-to-candidate inventory, final evidence/audit and retained failure history. Failed/intermediate receipts supply no final passing proof. Canonical contracts and existing source remain unchanged in this delta.

Candidate HEAD matched the requested SHA; its worktree was clean before and after execution. `git diff --check 8b64f70 b166665` passed. Independently rehashed the candidate's recorded 82 control source files, 18 worker source files, four packaging files and final audit map: zero mismatches. Memory supplied only the general bounded-review/evidence-tier convention; all candidate conclusions below use current source and fresh execution.

Only this report was written in the root repository. No candidate source/evidence, ledger, contract, lock, commit or unrelated root research was changed. No remote E2E, external exploitation, models, private audio, cloud, STT, Voicebox, push, PR or deployment. The legacy remote E2E script was never run.

## Mandatory Astra matrix closure

| Requirement | Independent assessment and evidence |
| --- | --- |
| Static boot and two sequential allocations | Closed at packaged synthetic E2 scope. Real installed CLI, lifecycle UDS and canonical W1 execute return verified 9644-byte WAV output. Silence then permissions allocations have distinct runtime IDs; old starts do not launch. Actual image entrypoints, users, mounts, namespace separation and capabilities are asserted by Docker inspect and live probes. |
| Intent/reservation/binding/start fencing and interruptions | Closed at declared E1/E2 split. OwnerStore intent precedes prepare; reservation is persisted before prepare ACK; binding precedes start; `runtime_supervisor.py:135-162` persists the start fence before setup/engine fork. Same reservation retains its original cutoff; started/old identities do not fork again. E1 ordering hooks cover prespawn/preregistration/prestart/postspawn, inherited presend/postsend and unknown observation fencing. E2 exits actual controls with code 17 after start ACK and start-frame send, then reconnects to exactly one physical launch. These are not E2 crashes at every E1 hook. |
| Lost result ACK and post-deadline collection | Closed. Repeated collection returns the same original proposal/manifest and independently verified bytes, including after cutoff. Collection uses a fresh communication budget without executing again. Terminal old-boot receipt replay is separately exercised. Ack intentionally retires the proposal; it retains the identity/receipt tombstone and never deletes staging. |
| Deadline with controls absent, blocked client and busy child | Closed for the fixture. Original absolute deadline is anchored once to monotonic time before start. The watchdog performs namespace TERM/KILL independently of handlers/journal I/O (`runtime_supervisor.py:202-222`). Actual controls are absent during expiry. E2 busy/descendant runs terminate; reconnect/start does not reset the cutoff. Child timestamps are never termination or timeliness authority. |
| TERM-ignore/double-fork/setsid and exact reap | Closed. Live independent audits establish three TERM-ignoring processes across two sessions. PID 1 signals the namespace, repeatedly escalates after 200 ms, drains `waitpid(-1, WNOHANG)` and requires ECHILD, not just leader exit (`:238-268`). Fresh deadline receipt records three reaps. Unreaped stop becomes unknown while ownership remains held; no finite-kill/uninterruptible-kernel guarantee is claimed. |
| Actual PID-1/container restart and journal uncertainty | Closed. E2 kills PID 1 and restarts the actual container: boot changes, old active allocation is unknown, launches remain zero and fresh prepare is refused. Old durable terminal receipt/proposal replay survives restart only for its bound identity. Missing/corrupt journal restart fails closed; old session/domain and wrong identities are refused. A new lock/boot is not used as death proof. |
| Child authority, permissions and capabilities | Closed for trusted packaged engines. Engine runs at allocation-specific UID/GID, no supplementary groups, zero permitted/effective capabilities, no-new-privileges and only its internal socket descriptor. E2 tests deny keys/receipts/management socket/PID-1 signalling and other-attempt read/rename. Worker lacks DB/publication mounts and service source; control lacks Docker authority. Wrong tuple/domain, oversized and partial frames fail within bounded exchanges. Required input set is empty, so no real input-bundle permission qualification is implied. |
| Start/terminal persistence failure, lost stop and failed observation | Closed at explicit evidence tiers. E1 injects ENOSPC at real journal open before fence completion and verifies no Popen. E2 makes the receipt directory actually unwritable: failed fence causes zero launches; failed terminal persistence yields unknown, refuses ack, and remains unknown after restart. Lost stop response still requires subsequent actual observed reap. Existing failed-observation tests preserve unknown. No signal/EOF alone releases the lane. This does not prove every filesystem failure syscall or a physically full host filesystem. |
| Readiness during blocked work | Closed for the declared synthetic ceiling. Fresh blocked-client plus busy-descendant readiness measured **0.8800446740351617 s**, below **5 s**; stop/escalation also meets the runner's bounded five-second check. This is not a universal latency SLA or production API responsiveness proof. |
| Affected regressions, identity and owned cleanup | Closed. Fresh Linux affected suite: **100 passed in 25.57 s**; all **15 packaged scenarios** passed. Exact supplied image source maps and topology checks passed. Shared heavy-lane lock used; termination checked before removal; previous resource IDs preserved, zero owned containers/volumes remain. |

## Reliability reasoning

`runtime_supervisor.py:63-72,118-129,260-273` makes terminal state depend on a journal row installed only after ECHILD and successful persistence. Receipt-write failure sets quarantine and cannot advertise STOPPED. Timely evidence requires both receipt-authority proposal arrival and observed complete reap strictly before the unchanged cutoff; delayed scheduling can conservatively reject success, but no engine timestamp can manufacture timely completion. Receiver completion is awaited before terminal persistence.

`runtime_journal.py:77-92` writes a bounded checksummed envelope, fsyncs the temporary file, atomically replaces the journal and fsyncs its directory before updating in-memory authority. Startup validates identity/binding/terminal relations and refuses missing/corrupt/oversized state. Exclusive lifetime flock serializes writers. Unknown old allocations block admission; terminal replay does not reconstruct execution. Eight unacknowledged allocations, 64 total identities and 1 MiB are refusal bounds, not an eviction or general disk-retention policy.

Lifecycle authentication is domain-separated from the execution capability; the child receives only an ephemeral internal key. The private protocol accepts no caller PID, observed state, generic signal, argv or path. Request handlers enqueue bounded calls; the PID-1 owner performs mutations/forks/reaping, with the separate watchdog retaining enforcement during management I/O. File persistence and OS scheduling still have no universal completion bound; withholding STOPPED preserves the required conservative behavior.

The installed control command durably records receipt outcome before bound ack (`lifecycle_cli.py:139-155`), and requires prior acknowledgement before a new diagnostic allocation. It is a one-row diagnostic caller, not an allocator/history or integrated terminal cancellation/publication policy. Future W2d must retain these receipt and ownership fences while replacing the older execution path.

## Fresh execution and exact identities

Exported the exact commit with `git archive` into `/tmp/w2c-sol-review.XKNx6N`, then ran the **unchanged** inspected runner:

`W2C_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-worker-w2c.sh`

Exit **0**. Offline builds reproduced the candidate's image IDs:

- Worker: `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b`.
- Control: `sha256:62b9d27609a63975ffba5d0c7d321ecd6df085e1dfd843bce162f2ebdb34e9b2`.
- Existing dependency: `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.

Fresh source/packaging rehash against the candidate after execution again found zero mismatches. No diagnostic source/runner delta or skipped build lane. Worker entrypoint is `python -m worker.runtime_supervisor`, PID 1 root:10001 with only SETUID/SETGID/KILL; actual engine probes establish privilege drop. Control is 10001:10001 and uses `service.lifecycle_cli`. Both are network-none/read-only-root with private PID/net/mount namespaces. Worker mounts only isolated named IPC/staging/receipt volumes; control mounts IPC/staging read-only plus private control storage. No host data bind, daemon socket/CLI or host-injected runtime observation enters the application path.

Fresh receipt: `/tmp/w2c-sol-review.XKNx6N/docs/agent-interface/phase-1/w2c-evidence/run.json`, SHA256 `f8bf7e091ee2fdb9abc5b2e3252249efb3a8b0efa26753497e562bad114a5243`. Runner log: `/tmp/w2c-sol-review.XKNx6N-run.log`. Project `pab-w2c-397b10183e` preserved all 15 prior containers and 140 prior volumes with identical inventory digests. Cleanup recorded zero owned resources, independently confirmed with project-label queries. Built images and temporary diagnostic evidence remain available.

This is exact-source dependency-overlay E1/E2 synthetic evidence. It establishes no fresh-install reproducibility, power-loss durability, real engine/residency, hardware, acoustic, browser or production acceptance. W2d, R1/P1/operator quarantine recovery and separate final Astra remain open as required by the architecture decision.
