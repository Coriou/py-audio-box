# P1B23 Core ownership r1 — independent Sol-low review

2026-10-23. **Bounded source/local-matrix GO for exact `e5d4df14deb49013ed7cc9f3435444c23eae1b39`, frozen source/tests `5619270d5cdce0d305a17fd60be58f53c53f550a`. No blocking owning-contract defect established. Full combined validation/integration remains NO-GO pending guard qualification, the separately authorized old-test adaptation, combined-source review and remaining gates.** No native-cause clearance, T2 or release acceptance.

## Identity and scope

Reviewed ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-0727f054`. Exact HEAD and empty porcelain verified at entry and after execution. Frozen-to-HEAD changes only RECOVERY-P1B2 documentation; executable/test diff empty across `service tests worker scripts clients contracts`. Baseline `c898f74aec2e859783b97126e6be7536be051239` independently equals reviewed T1 `7b9cafdc2073eba52149ba46fa9253706ace3015` across those groups. All eleven handoff replay pairs, oldest first from b94dd802→7ba1e39a through f0bfcc1f→e5b90d6d, independently have empty scoped diffs. Exact argv/full pins are retained in `/tmp/pab-core-review-r1/commands.jsonl`; no replay was performed by this reviewer.

Historical worktree HEAD inventory before execution and final HEAD/porcelain checks agree: original WIP `f0bfcc1f750c17ed126fdcf6c3ac620ce71f18cf`, F2 `b16a41f718fc83ee0561c4621017ff9ae8342c4e`, F1 `c8e94cf42dbe7724e6a9b27b5cc08e0195c924aa`, original lifetime `4259415daaf5acecfe53ed9d44f9895207a00d1b`, C `1d741cb328228e67f99a096e9d6a9c1acd1f01b6`, guard `4ff625669024c33b0eeb3ecad40f26a2a71bfc99`. All final porcelains empty. Historical cleanliness at entry is handoff-attributed; this review independently captured their initial HEADs and final cleanliness. No historical imports/edits or C/new guard import. No applicable AGENTS.md found in candidate ancestry/scoped tree. Root research/untracked artifacts preserved.

Read the entire authoritative LIFETIME-RETENTION-ASTRA (64 full-lifetime slots, pre-fork empty gate, Dispatcher outcome, all required matrix rows), original LIFETIME r1/r2/r3 findings, RUNTIME-LIFETIME-ASTRA L1–L5 and matrix, T1-r3 limits, canonical orchestration protocol, relevant ledger chronology and candidate RECOVERY-P1B2 **2626–2897/EOF entire**, including all 93/128/38+HOLD receipts and compatibility inventory. Inspected complete production delta, owning appended tests, authorized double-failure amendment and production callers. Memory supplied scope/preservation guidance only, not acceptance evidence.

AST comparison to baseline confirms changed existing methods/functions only: Core.__init__/close; Dispatcher.__init__/close/_run; recovery._run_bounded_maintenance/enrollment_gate/startup_read. Added ownership/bookkeeping/view/property definitions and `_startup_read_lock`; no removed definitions. No original baseline test definition changed. Relative to the replayed WIP, no existing owning test definition changes; relative to F2, the double-failure amendment is the selected narrow exception. Conftest is historical replay content, not a newly authored guard correction. No worker/scripts/clients/contracts delta. `git diff --check` passes.

## Owning-contract assessment

**R1 F1 and R2 F2 remain closed; R3 F1/F2 are addressed by the adopted design and candidate.**

- Core admission (`service/core.py:79–197`): fixed 64-element registry, no overflow/eviction/configuration. Capacity and normalized absolute root-key exclusion occur atomically under a nonblocking mutex before mkdir/owner/DB effects. A validated directory dev/inode binding catches aliases before owner open. Constructing, live, closing and uncertain records consume the same slot; successful construction does not release it. Existing actual flock remains authoritative across processes. Failed competing flock closes only its own resource; a raising close retains its reserved slot conservatively.
- Identity and close (`core.py:112–154`, `:198–285`): PID is checked before mutex access; release requires the identical Thread object, not numeric ident. Exact record identity prevents stale slot removal. Active Core refusal precedes permission for ordinary cleanup; private record.close refuses live ownership and has no caller-supplied bypass flag. DB close must return successfully before its record reference is cleared and owner close attempted. Before/after-close exceptions retain uncertainty; owner-only recovery does not repeat a proven DB close. No blind retry or cross-thread recovery. Resource calls occur outside the bookkeeping mutex. Failed close makes normal DB work terminal; explicit owning-thread continuation is distinct. Clean repeated close cannot remove a successor. Records store bounded state/resource/Thread identity, no Core, exception, traceback or Future field. This is cardinality, not RSS/whole-process FD proof.
- Constructor/L1: encompassing BaseException cleanup preserves primary traceback and bounded secondary notes, including early initialization and final sync phases. Both private readers retain transaction-context behavior and explicitly close before private namespace teardown on success, early return and failure. Source instability remains refusal; no recapture or ignored SHM. A raising reader close is not proof of SQLite release.
- Dispatcher (`dispatcher.py:35`, `:46–50`, `:56–82`, `:134–154`): existing five-second constructor wait and four-second join retained. Validated existing nonblocking control lock precedes startup copy/enrollment; independent Core acquisition and repeated deployment/completion/exact hello checks cover the gap. Startup primary publication follows both cleanup attempts. Pool failure cannot skip Core cleanup. Cleanup failure marker is set by either failure, checked before queueing and after join, so a cleanup-failed writer cannot yield successful close. Races retain the existing bounded queue/Future behavior; no cleanup retry or claim of bounded kernel/pool shutdown.
- Maintenance (`recovery.py:505–548`): Linux/single-thread check first; entire registry empty lease before pipe/fork/session/prewrite. All roots/states refuse, including healthy/constructing and owner-only uncertainty. Mutex is held only through fork, then released by parent on successful fork and pipe/fork/identity setup exceptions. Child never exits/unlocks/clears/rebinds inherited lease and registry; registry entry rejects inherited PID before locked mutex. Parent transport ownership/descriptor/exact-child cleanup body is otherwise preserved. No new source/session authority is conferred by admission.

The fixed retained-resource/dead-thread policy is explicit: abandoned records consume capacity and block maintenance until process exit. It is not eventual cleanup or a promise that a raising owner close still holds its actual flock. Empty Core registry is necessary only; raw/private/offline SQLite handles and fixture transaction-context leaks are outside its accounting. No whole-process SQLite/fork-safety proof or native-cause claim follows.

Production callers inspected: Dispatcher is the only production startup_read/enrollment_gate caller and holds the read lock; persistent_init owns Core with finally close. Recovery has the sole maintenance fork seam; worker runtime forks are separate process code. No legitimate >64 simultaneous-Core production caller found. This is source inventory, not full integration compatibility qualification.

## Matrix and preservation evidence

The complete required matrix is represented in candidate tests/handoff: capacity/last-slot race; nested/path/inode alias accounting; early initialization phases; healthy sequential lifetime; before/after DB and owner faults with retained/dropped traceback; wrong PID/Thread/reentry/stale identity; same/fresh-process exclusion; isolated dead writer; active/private-close refusal; Dispatcher startup/normal-stop independent faults; ten same/different-root maintenance state refusals; lease exceptions/parent release/child refusal; L1 readers/readlock/binding and old double-failure.

Independently read and SHA-256 verified frozen implementer logs:

| Attributed frozen execution | Outcome |
| --- | --- |
| Selected matrix | 93 passed, 1721 deselected, 14.77s |
| Complete Core/Dispatcher | 128 passed, 53.67s |
| Recovery preservation | 38 passed, then 1 failure, 1493 deselected, 17.43s — HOLD |

The last is exactly `test_t1_actual_pipe_transport[backpressure]`: PID 384, raw status 18944, exit 74 without semantic ValueError phase witness. It is not a proven native signal or proven semantic refusal. Remaining selected preservation cases did not run. No blocked broad rerun, guard waiver or generic exit-74 acceptance here. Separate G1 precise nofollow/OSError witnessing and G2 fresh-helper owner binding/transport-fault witnessing require their own qualified review and later combination.

The approved double-failure branch retains all seven parameters and primary identity/traceback/secondary-note/closed-DB assertions. Other branches retain flock reacquisition. Double-failure proves retained exclusion, switches off only its own fault in finally, invokes normal same-thread record cleanup, then proves record removal/owner closure/reacquisition. No base-class close, registry clear, skip or xfail.

`test_b22c_denied_fault_update_inherited_checking_new_epoch` (`test_recovery.py:1915–1944`) still constructs Core in a fork-only child and is incompatible with the new PID refusal. It remains unchanged. Per current parent authorization, its later narrow fresh-exec/bounded-JSON adaptation must retain the SAME query_only/checking/transient-growth body, exit 73 and every parent persistence/resume assertion. This pending gate is not a defect in the new ownership contract itself. No other old-test rewrite is approved or made by this review. Inventory of other fork/helper paths remains source-level evidence, not all unmanaged-connection safety.

## Fresh independent execution

One serialized Docker lane, atomically acquired `/tmp/pab-worker-w1-docker-lane.lock`; no wait/poll or guard conflict. Reused existing exact candidate-derived image `sha256:533fef42e66afa7686c4491f8e574644e97a44fe25839d21dab04f7aa5d3bd2b`, descended from pinned comparison image `sha256:3a921bd0951289f29c6697dba7271adf42788b19628eac0213b856c196fdbda1` / pinned Python 3.14.8 base index `sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83`. No image/package installation or source mutation. Before imports, all **188 tracked executable/test-group file hashes inside /candidate** matched this clean candidate; host hashes matched again after execution.

Owned name `pab-core-review-r1-20261023`; CPU2/memory2GiB/PID256, network none, read-only root, tmpfs /tmp 512MiB, user10001, cap-drop ALL/no-new-privileges, bytecode disabled, tokenizer cache fixed. No user mounts or published ports. PID1 lifetime 300s; each probe/test exec at most 60s. Exact commands, exits and elapsed times in `/tmp/pab-core-review-r1/commands.jsonl`; driver/probe/audit retained under /tmp.

1. Independent `/tmp/pab-core-review-r1-probe.py`, fresh Linux interpreter, **exit 0**. Real SQLite ordinary Core with injected pre-close failure remains SELECT-live on owning thread, retains actual flock against a fresh interpreter, refuses normal jobs and maintenance; explicit same-thread cleanup restores empty registry. Separately 64 actual healthy Cores retain 64 slots and the 65th refuses before root creation; all close normally. Actual empty-registry lease fork rejects child Core before inherited locked mutex, exact child exits 0/reaped, parent admits/closes afterward. This was a small bounded local probe, not an OS-limit or native-cause experiment; no inherited SQLite handle was live in its fork case.
2. `docker exec -w /candidate pab-core-review-r1-20261023 python -m pytest -p no:cacheprovider` with nodes below plus `-q -rs -x`: **12 passed, 12.65s, exit 0**, no skips/xfails:
   - `tests/service/test_core.py::test_bounded_core_atomic_last_slot`
   - `tests/service/test_dispatcher.py::test_lifetime_actual_timeout_excludes_next_reader` (both actual five-second cases, same/fresh-process exclusion)
   - `tests/service/test_dispatcher.py::test_bounded_core_dispatcher_cleanup_outcome` (six startup/initialized pool/Core/both cases)
   - `tests/service/test_recovery.py::test_bounded_core_prefork_lease_failure` (pipe/fork)
   - `tests/service/test_recovery.py::test_bounded_core_lease_parent_release_child_pid_refusal` (actual modeled clean transport)

These are fresh reviewer executions; the 93/128/38 receipts above remain implementer-attributed. No complete runtime/recovery campaign rerun. Actual five-second waiting is exercised, not shortened to a mock wait. The custom boundary probe supplements private-reservation tests with actual successful Core lifetimes. No RLIMIT or peer-credential qualification was selected.

Logs captured before exact-owned stop/inspect/rm (no -v). Final container ExitCode 0, OOMKilled false, Running false; removed; lane removed only by this owner. Probe-owned SQLite, temp roots, subprocess/thread resources and exact fork child cleaned. No image/foreign container/volume cleanup.

Evidence SHA-256:

- source manifest: `60c5a2207d80b2ea6cf8eb8b1281799289fe8f2d9da188207426dec9773bd251`
- selected stdout: `9fb9a7b5cac39d1010d14f202102198a3c68d328a2954249efd26e523bcc9952`
- independent stdout: `2465b8ff076d9805fc482db713a8b92ad79625fa5b2c0c35a2cc273e1bff8375`
- command ledger: `08a2faf2df79d240b4d14ea17b039e7545071db7ac92ba8e42b83711bf3f8711`
- AST inventory: `e30b703cd7ea2439b3f24cdc871ab5a3040ecbf28bdf65b0b161f16af717efad`
- preserved pins/final status: `04f0eaa0fc8f5e102b6351cc6ca600303d0ea62928349f402d38142dea6bb7ba`

## Remaining gates and sole write

Retain L=4,194,304; T=1,048,576 canonical result; original 1MiB snapshot/64KiB documents/2,048 and 16MiB inspection; WAL/FULL/Unix-VFS/mmap restrictions, both locks, strict complete current-source/history joins, accepted/no-attempt exact authority, retained H/charges (`U_all_early<=H`, `H+C<=B`), once-only capture/consumption and completed observational behavior. T1 transport finite 64-round cleanup, exact-child/single-reaper and kernel-latency limits remain. No schema/persistence/accounting/cap change beyond selected internal ownership admission.

All **19 Linux mechanisms remain unqualified** by this review; running on Linux alone does not qualify them. C1 mandatory `b2_pinned_launcher_context_unavailable`, actual Linux 4MiB/ext4/VFS/SHM/peer credentials, B3/C1/C2/install/durability, T2, native V1/V2/V3 and later fresh-helper/collection failures, final global Astra, hardware/browser/acoustic/activation remain pending. A1 inventory drift remains unexplained; Apple Silicon documentation-only qualification retained. Linux greens do not clear Darwin native causes.

Parent still owns fresh-exec old-test implementation/review, independently qualified test-only guard, new combined source/guard candidate and exact combined review/validation before integration. This bounded GO grants none of those gates.

Sole repository write: `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23-CORE-OWNERSHIP-r1-sol.md`. No fixes, commits, delegation, other-worktree edits, imports from historical candidates, installs, GC/allocator/runtime/fork-architecture workaround, native crash reproduction, parent-limit lowering, models/audio/paid E2E, push/deploy/prune/userdata changes or self-integration. Temporary evidence retained. Candidate remains exact/clean after execution; root preexisting research preserved.
