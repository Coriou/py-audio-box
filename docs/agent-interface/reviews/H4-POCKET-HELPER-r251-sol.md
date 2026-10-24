# H4 Pocket post-reap helper — r251 Sol implementation

2026-10-24. **Implemented source repair, submitted for parent independent Sol review.** Empty-at-reap Pocket success/cancellation now removes the exact conditioning directory before durable terminal persistence. Real same-root Dispatcher reopen, complete detached/clean-history readers, and next-job admission pass with inert fixtures. Nonempty interrupted WAV/YAML namespaces remain UNKNOWN. This is not actual installed/cross-UID/kernel, audio, activation or release acceptance; no self-adoption or additional Astra gate.

## Exact pins and ownership

Owned checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-3133b78f`, branch `t3/pocket-helper-r251`, clean entry **54b5fb5a5a76ae9f82f041fb687805c39a782956** (the requested root HEAD). Cherry-picked guard **ca0f9fa0740bcd00e005bc3d813a21e2c2a691b8** as **82339f9164605cf593cefff85d7df8fa39492755**, then report **80a3458** as **18735de3ed870de0d2aa7e81dccc74804e7b354a**. Implementation source commit **1979f60bef60b5463a0ca9c5b5affb0d111406f7**; this report is a separate subsequent commit.

Read FULL r248 normative original brief, r244–r247, and original r249 using `git show 80a3458:docs/agent-interface/reviews/H4-POCKET-POSTREAP-r249-sol.md`. Inspected supervisor/child/protocol/journal, admission, reconciliation/recovery readers and existing test support. No applicable AGENTS.md in the owned tree or checked ancestors. Bounded memory registry search returned no relevant hit and supplied no premise. No delegation or shared-root/historical-root writes.

Source ownership: `worker/runtime_supervisor.py`, new `worker/runtime_conditioning.py`, `tests/worker/test_runtime_conditioning.py`, `tests/service/test_pocket_lifecycle.py`, removal of the admission guard, and the parent's explicitly expanded **single configuration-digest correction** in `service/recovery.py`. Six source/test files, 569 insertions/53 deletions relative to the cherry-picked report. Child unchanged. No journal/schema/new durable type, phase or receipt; no recovery redesign, installer framework, replay, sweep or old-latch reset.

## Helper and lifecycle

After the existing `waitpid(-1,WNOHANG)` loop reaches ECHILD and the started allocation's receiver finishes, tick drains its final proposal and freezes the original receipt: allocation/runtime binding, child UID, deadline, timely flag, reap method and engine reap count. It marks the still-occupied active allocation `finalizing` under `tree_lock`. Stop/watchdog signalling skips that allocation; the lock is released before launching or waiting for housekeeping. No later prepare/start can acquire the lane during this interval.

Pocket housekeeping launches only `python -m worker.runtime_conditioning <exact allocated stage> <allocated UID>`, as UID10001, primary GID=child UID, supplementary groups `[]`, closed FDs and DEVNULL stdio. It waits that exact Popen handle for one second. Timeout or wait error triggers explicit kill and another bounded exact wait; unconfirmed death keeps the active handle occupied/quarantined. Refusal, nonzero exit, timeout, fsync error or journal failure leaves the existing nonterminal record without receipt/terminal transition. The failed lane cannot admit another allocation; collect provides no terminal receipt and ACK refuses. Next tick cannot reinterpret housekeeping as engine reap or terminalize the failed allocation. Stopped prepared allocations have an explicit no-start/no-stage-setup branch; a missing stage after start is refused.

The helper opens fixed `/staging` using Linux `O_PATH|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, validating root UID/GID0 and mode1733. It opens only the validated single-component stage readably, descriptor-relative/no-follow, requiring UID10001, child GID, mode0770, same device and stable root/name/FD identity. Literal `conditioning`, if present, must be a real directory, child UID/GID, mode0700 and same device, with unchanged identity. It never opens/enumerates that private directory. Its only removal is `os.rmdir('conditioning',dir_fd=stage_fd)`: the kernel rejects nonempty contents. After removal it fsyncs the stage and revalidates the bound identity. Verified absence is accepted only within an otherwise verified stage. There is no file unlink, recursion, chmod/chown, new capability or generic cleanup RPC. Trusted control-owner/dead-exclusive-child writer exclusion is the premise, not resistance to concurrent privileged host mutation.

Only helper success permits the ordinary terminal journal save, followed by existing Dispatcher collect/validation/release/ACK. Helper elapsed time and wait status do not alter the frozen engine receipt. Both readiness guards and their test-only admission bypass were removed with the implemented source gate; mapping/byte verification before supervisor allocation is exercised again. Installed configuration remains Kokoro-only.

## Concrete reader blocker discovered and narrowly corrected

The original additional real `installed_clean_history` probe failed with **`ValueError('installed_history_unclean')`**, `allocation_coherent=False`, no faults/inventory reasons, a retained reservation, succeeded/reaped job and acknowledged dispatch. The complete `_installed_detached_source` reader already passed. This failure was retained as code evidence and reported to the parent, not waived or hidden by rewriting fixture demand.

Original `recovery._allocation_coherent` compared every reservation configuration against `digest(context.plan.resolved.identity.effective)`. Actual typed admission stores `digest(context.execution)`. The parent explicitly authorized the exact correction and associated regressions. Recovery now invokes the existing closed `worker.runtime_protocol.execution` decoder and selects **one** expected value: full execution for a typed context, otherwise the legacy effective identity. It does not accept either hash loosely or add compatibility fallback. Equivalent configuration checks were searched: Admission and live `reconciliation.typed_references` already implement these same semantics; no other reservation-configuration check required a change. Unrelated installation/configuration hashes remain untouched.

The positive real clean-history tests fail on the original check and pass with this correction: Pocket success/cancellation and Kokoro typed execution; legacy synthetic effective-identity history remains accepted. Negative tests alter demand to zero or the formerly accepted effective-identity hash for typed execution. Another mutation changes a semantically valid descriptor budget, rebuilds the full Authority/context/binding, matching receipt and journal digests, but retains the old reservation configuration. `_allocation_coherent` rejects it, and `installed_clean_history` rejects the coherently rehashed evidence at snapshot binding even when its claimed coherent flag remains true. No schema/history redesign was needed.

## Source acceptance coverage

- Success and cancellation each close/reopen the **same** real Dispatcher DB/staging roots. Cancellation creates mode0700 conditioning before cancel intent, delays confirmed reap, verifies the directory and executing reservation remain, then calls actual helper logic before returning the terminal receipt. Completion waits for retained reservation and ACK/idle. Reopen checks healthy integrity, original terminal job/artifact, saved-version equality and foreign-principal refusal, directory absence, and a different admitted job reaching executing state.
- Test-only enrollment support copies only the small completed owned fixture DB and explicitly named asset/output files into an inert installed-history image. Real Reconciliation emits and persists rows; each row equals inventory persistence/latest history. Actual `snapshot`, complete `_installed_detached_source`, `installed_clean_history` and `_validate_diagnostic_eligibility` consume that image. This neither changes the original Dispatcher DB nor fabricates Pocket installed configuration. No AST extraction or reader monkeypatch is used.
- Supervisor seams prove leader exit without ECHILD and receiver-pending ECHILD do not launch housekeeping; exact helper credentials; occupied lane/watchdog coordination/no lock held across wait; original receipt binding/deadline/timely/count, including helper elapsed time beyond the original deadline; refusal/timeout/unknown death; no terminal/ACK on failure or next tick; and journal failure after actual same-UID empty rmdir.
- FD/filesystem tests cover kernel empty removal and verified absence, sentinel-bearing nonempty directories, symlink/file, UID/GID/mode/device policy, stage symlink/substitution, child identity change, post-fsync stage substitution, rmdir refusal and fsync failure. Output/sentinels survive. Root O_PATH/no-follow flags and exactly two directory opens are asserted; there is no descendant traversal. Ownership is injected policy over real same-UID temporary filesystem operations, **not cross-UID kernel evidence**.
- Wrapper WAV/backend WAV/translated YAML leftovers explicitly refuse helper removal and remain UNKNOWN/blocking through real reconciliation. Existing empty/nonempty retained legacy UNKNOWN persistence/diagnostic-reader regression remains. No historical latch is cleared.

## Exact verification and limits

Inspected `tests/conftest.py` and `tests/service/conftest.py` before running. Used only unittest imports with **Python 3.14.7**, `PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B`, cwd owned checkout, and explicit absolute `sys.path[:0]` entries for that checkout and its `tests/service`. Runner:

```python
r = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(n) for n in names))
sys.exit(not r.wasSuccessful())
```

Final named `names`:

```python
("tests.worker.test_runtime_conditioning", "tests.service.test_pocket_lifecycle",
 "tests.service.test_engine_admission", "tests.worker.test_engine_lifecycle",
 "tests.worker.test_engine_pocket", "tests.worker.test_engine_kokoro",
 "tests.service.test_production", "tests.service.test_production_admission",
 "tests.service.test_production_reconciliation", "tests.service.test_production_api",
 "tests.clients.test_production_clients")
```

**82 methods PASS, 17.766 s, exit0**, after the recovery correction. The existing injected `lifecycle supervisor unavailable; lane quarantined` line appeared with overall OK. Final assertion-only refinements then ran exact names `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests` and `tests.service.test_pocket_lifecycle.PocketLifecycleTests.test_dispatcher_generate_artifact_and_cancel_reap`: **3 PASS, 4.917 s, exit0**. Final helper-only run after strengthening leader-exit-without-ECHILD assertion: `tests.worker.test_runtime_conditioning`, **17 PASS, 0.047 s, exit0**. No production change followed these checks.

Installed regression `names=("tests.service.test_installed_restart","tests.service.test_installed_registration","tests.service.test_installed_host_policy")`: **87 PASS, 62.150 s, exit0** after the recovery correction. Earlier unchanged-reader run was 87 PASS/61.112 s; the repeat was justified by the newly authorized recovery change. No global recovery campaign was collected.

Preserved intermediate source-test evidence: before reader correction, clean-history failures above were real failing acceptance probes. During final assertion refinement one three-case focused run hit the existing five-second Dispatcher fixture wait (3 methods, 7.269 s, one failure); it did not retain terminal status for diagnosis. Terminal-failure diagnostic reporting was improved without extending any deadline, and the final exact three-case run passed. This isolated fixture wait was not converted into a production fallback or claimed to have an established root cause.

`git diff --check` PASS. Six changed Python files parse with Python3.11 grammar (grammar only, not Python3.11 execution). Observed `/private/tmp` available **3,873,736 KiB**; only small inert owned fixtures/normal fixture teardown. No pytest collection, maintenance guard, listener, actual ML/model loading, Docker, network/download, process crash, operational/historical cleanup, HELD operation or delegation. Existing lazy-import tests may perform inert interpreter probes. No PR or merge.

Parent independent Sol source review/adoption remains. Actual installed caps/mounts, UID switching, kernel permission enforcement, PID1 reap/transport/resources and audio/listening remain unqualified. Nonempty snapshot interruption and mid-flight restart remain conservative UNKNOWN; this repair promises the empty-at-reap completed-finalization path only.
