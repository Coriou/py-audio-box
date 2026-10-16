# W2a independent Sol-low functional review

2026-10-16. **GO for the dormant W2a durable owner-state seam only.** No blocking functional defect identified within this explicitly narrowed boundary. **NO-GO for completed W2, persistent execution activation or Phase 1A acceptance.** Final global Astra review remains separate.

Candidate: `1194814141073be76c78fde9d32a9439f737987b`, branch `t3/w2-worker-supervision`, inspected at `/Users/ben/.txcode/worktrees/py-audio-box/t3-001f240a`. Base: `925477626e7758922c6edb05a4877b0f0b2bf94c`. Candidate clean before review. Contract pack remains 1.3.0. Reviewer runtime: gpt-6.1-sol, low effort; independent read-only review, no child agents.

## Scope and evidence inspected

Read PRODUCT, IMPLEMENTATION-START, ORCHESTRATION-PROTOCOL, canonical IMPLEMENTATION-PLAN scope/shared contracts/Phase 1A/evidence requirements, CONTRACTS 1.3.0, the complete 1A-foundation-astra-gate report including its original W2 brief, and candidate WORKER-W2. Inspected the complete eight-file base-to-candidate delta, owner store, supervisor, lifecycle protocol, lifecycle tests, runner, supplied receipts, W1 Authority implementation and callers. Search confirms the new modules are consumed only by their synthetic tests; no Core/transport dispatcher hook is added.

The original W2 brief requires actual isolated reconnect, stop, deadline and fault coverage. W2a deliberately supplies only private durable policy primitives. This review accepts that useful sublot without waiving the original gates. No source fix, commit, ledger edit, model/audio/cloud/STT/Voicebox operation, remote E2E, push, PR or deployment occurred. The only workspace write is this report; root research and candidate evidence remain untouched.

## Functional assessment and claim closure

- **Serialized durable intent: closed at primitive scope.** `service/owner_store.py:17-29,41-54` acquires the exclusive lifetime flock before opening SQLite and commits WAL/FULL intent before returning. SQLite thread affinity plus the trusted single-owner call graph supports serialized access; this is not a general threaded command queue. Validation precedes mutation. A failed transaction propagates and cannot authorize dispatch. The process-death test exits after committed intent and reopens it as unknown. There is no spawn callback in this candidate: ordering intent before a future callback, and bounded handling of callback exceptions, are caller/integration obligations, not demonstrated behavior.
- **Immutable allocation and binding: closed for the current lane.** `owner_store.py:44-69,84-88` stores the canonical complete execute message plus W1 authority binding. W1 Authority snapshots its source objects and returns detached accessors. Same-current-allocation intent returns the persisted record; changed allocation is fenced while live/unknown. Binding requires matching authority and a nonempty identity and cannot overwrite an existing binding. Repeated bind deliberately refuses rather than retargets. Dispatch idempotence here means current intent lookup, not implemented exactly-once physical execution or a historical dispatch ledger.
- **Restart/reconnect: closed for state lookup, not transport reconnect.** `worker_supervisor.py:11-24` returns unbound intent unchanged; bound reconnect observes the persisted exact runtime and never invokes spawn or modifies authority/deadline. Observer exceptions, wrong runtime identity and invalid state become unknown. The existing test closes/reopens the DB and performs repeated observations of the same allocation. No reconnect send, ACK recovery or cached W1 result exists yet.
- **Unknown quarantine and replacement: closed at store scope.** `owner_store.py:48-53,75-80` permits a different allocation only after STOPPED. Missing identity, observation failure and unknown never release the lane. There is no cleanup/publication API to bypass that rule. STOPPED is absorbing for the current binding; later alive/unknown evidence is refused. Different-runtime evidence cannot stop or revive it. The future authoritative allocator still owns unique attempts/staging and monotonic generations; this single-row store is not an allocator or audit history.
- **Stop/escalation: closed as trusted-adapter policy.** `worker_supervisor.py:26-39` first observes, requests terminate, observes, and if necessary requests kill and observes again. Signal return values never establish STOPPED. Action exceptions do not skip independent subsequent observation. Already-stopped and unbound records cause no signal. Only exact typed STOPPED evidence releases the lane. Existing parameterized tests cover graceful success, escalation, observer errors and persistent unknown. Action-exception branches were source-reviewed, not separately fault-injected in this review.
- **Termination and bounds: explicitly unproven.** `worker/lifecycle.py:18-27` correctly assigns independent exact-runtime proof and bounded calls to the trusted adapter. A dataclass/enum is an evidence carrier, not OS termination proof. Passing a timeout argument cannot bound a blocking Python implementation. No concrete adapter is present, so this GO cannot establish actual process-tree/container death, reap or responsiveness.
- **Deadline: preserved, not enforced here.** The absolute deadline is part of immutable stored authority. Neither supervisor method compares it with the clock or schedules expiry. Even “at/after deadline, control requests stop” in WORKER-W2 is a future caller obligation, not current automatic behavior. Control-absent enforcement, deadline/cancel/result races and no-reset actual execution remain pending.

No concrete in-scope blocking defect or failing reproduction was found. Conservative exception/quarantine behavior is appropriate for this dormant private seam; it must not be presented as recovery availability or general runtime qualification.

## Tests actually run and identities

No host dependency installation or image build. The reported host missing-jsonschema collection failure is an environment limitation, not a code finding; it was not rerun.

To avoid changing candidate receipts, copied the source roots and exact final runner into a private temporary review directory, excluding Python caches. The unchanged `bash scripts/test-worker-w2.sh` ran there using its shared `/tmp/pab-worker-w1-docker-lane.lock`. Runner SHA256: `83b6207041f4de19a7b54e08d9e61a2fe09d641774103af21cafdfe293213969`.

- Exact final runner: **9 passed in 2.81s**, exit 0. Eight lifecycle cases plus the collected in-container source verifier. This closes the candidate's stated unrerun harness-only edits for this successful execution path.
- Proportional regression: same runner machinery in the temporary directory, changing only pytest selection to `tests/service tests/release`: **77 passed in 36.71s**, exit 0. This is the 76 service/release cases plus the source verifier, not 77 distinct product regressions.
- `git diff --check 9254776 HEAD` and `bash -n scripts/test-worker-w2.sh`: passed.

Both used dependency image `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`, working directory `/opt/speech`, network none, 2 CPUs, 1 GiB memory, 128 PID limit, no host mounts and no volumes. The runner copies service, worker, service/release tests, contracts and clients; a collected verifier checked all 65 supplied source files. Canonical sorted compact JSON source-map SHA256: `e155b5f3fefa5e56a9e7650b2dc971575b1678245910e3f4f77d276fceb5540b`.

Key verified source hashes:

- `service/owner_store.py`: `f140ac519d29aa60b38e916897ec6372f888e54f0c13553760a846f56b14060d`
- `service/worker_supervisor.py`: `c275bf7d2bf4e3c2b1225b0a0ce163f34355ebd5ae979965245e390f33bf0714`
- `worker/lifecycle.py`: `573bd2ed4c1da4a709f787c6cb57c408fed00046373d3ac5c701f3c5562e68a6`
- `tests/service/test_worker_lifecycle.py`: `fa21e059b20582a65ee80e5f6dc1d34940809abbd72284f29995a8ed2be39cf7`

Owned containers `pab-w2a-11c8c5424694` and `pab-w2a-c7f7fae6acec` were each inspected Running=false, Status=exited, ExitCode=0 before removal. Both runner receipts assert prior container inventory preserved. No existing Docker resource was removed or stopped. Temporary receipts remain under `/var/folders/gk/6rj578d95ls341449hb55pj80000gn/T/pab-w2a-review-o3yqodvn/docs/agent-interface/phase-1/w2-evidence/` (`focused.json` and `local.json`); they are transient supporting evidence, not committed artifacts.

These are E1 source-overlay checks in an existing dependency image. Hash verification covers supplied files, not absence of additional image files or a fresh packaged-image build. No complete legacy suite, fresh install, persistent-volume/power-loss, separate control/worker E2, real engine/hardware, browser or acoustic qualification follows. Candidate's earlier 76/9 receipts are supporting history; the results above are independently rerun.

## W2b continuation and bounded verdict

This is a sound small foundation for W2b: persistent uncertainty, immutable authority/runtime binding, conservative replacement and independent-observation stop policy are reusable constraints. It is not itself a functioning supervisor. W2b must supply the actual adapter/companion and integrate these primitives with the accepted W1 transport and serialized core owner.

Before full W2 acceptance, demonstrate separate network-none worker runtime identity and actual stop/reap; same-attempt reconnect with cached result/no duplicate execute; independent deadline enforcement while control is absent; durable pre-callback ordering and exception unwind; deterministic pre-spawn, post-spawn/pre-registration, pre/post-send, observation/reap and pre-verification interruptions; control/worker restart, lost ACK, stale session and cancellation/result/deadline races. Preserve unknown quarantine, no cleanup/reuse before exact confirmed termination, unchanged deadline, unique replacement allocation and control-only publication. Obtain exact source/image/topology receipts and independent review of that implementation, then integrated verification. **W2a GO grants none of those remaining gates and no final Astra approval.**
