# P1B23 S driver global SOURCE + actual HOST review r13

2026-10-23. **NO-GO for construction readiness: one required HOST closure observation is masked by harness teardown.** F1–F4 are addressed in the final driver source. The retained HOST matrix is substantial and its latest84 distinct cases really pass, but it does not yet independently demonstrate driver-owned process-pipe closure. This is a bounded harness/evidence objection, not a demonstrated final-driver descriptor leak, a demand for a new product campaign, or permission to launch.

## Scope and provenance

Sole write is this root report; no commit. Candidate inspected READONLY at `/Users/ben/.txcode/worktrees/py-audio-box/s-driver-global-r13-20261023`, clean HEAD `559b0d333d495da38d281943af04a098586f22da`; executable/harness freeze `1fa414bcd8b49ce7bea6bd8891b023ffa3c346e2`. HEAD differs from that freeze only in `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. Baseline executable `b968691a50bde1cc7d4165bb3fa36e8ade1612ce`; application pin `bd34b74cc0634c6e3b65cabb373ecbad3ccc2521`.

Read the entire root `P1B23-S-DRIVER-GLOBAL-ASTRA-r13.md`, all460 final driver lines and538 harness lines, the complete new handoff from4112 onward and original owning brief at2830, all158 recipe lines, the Dockerfile and complete construction script. Reviewed repair diffs through both successor freezes and raw retained HOST receipts/results, not merely the consolidated audit's assertions. No applicable AGENTS.md was found in scoped inspection. A memory registry search supplied no relevant S-driver fact; no memory-derived conclusion is used.

Only read-only text/Git/stat inspection and host stdlib parsing/hashing were used. No candidate module import/execution, harness rerun, pytest, Docker query, SQLite/DB, product/recipe execution, source fix, delegation, push, cleanup or resource mutation occurred. One reviewer-owned parsing snippet had a syntax error before execution and was corrected; it did not run candidate code or write evidence.

Independently verified current SHA256s:

- Driver: `35de402a8dd4bd9484692663d54c57b625c6fd19fc11cd06553f568cb2ae1299`.
- Harness: `f99945d98e11bdc30ad0a0326e0b2502878a38a7a87d6b53daba9bbdc66f862c`.
- Dockerfile: `d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2`.
- Recipe: `1cdab0c25cfac3446fa233dc57d243f06a6ebfbf0dcd40b59326b49cef980b74`.
- Literal list: `5bd05d4221cf1f3a3e6badd36a8a683a32196b1f9122b23b0c44b6079675aec6`.
- Construction script: `3cbdfbb37a2c80013530c235c4f4f9c76439c2d5911d32ac33b725593de0b8f4`.

Pin-to-HEAD changes are only added driver/Dockerfile/harness and the inherited handoff. Thus application, guard, controls, schema, lock, list, recipe and construction code remain unchanged at Git level. Independently compared all263 selected current bytes against both `git show bd34b74:<path>` and the retained archive fixture, then recalculated263 entries/29390 bytes/manifest SHA256 `66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46`. This is frozen-input verification, not an actual recipe copy. Read-only diff --check passed.

## Blocking H1 — teardown makes process-pipe closure checks non-discriminating

Harness `wrapper`:409–410 runs `p.close()` in finally. `PipeProcess.close`:95–96 explicitly closes any still-open stdout/stderr pipe. Only afterward does :417 assert `p.stdout.closed and p.stderr.closed`. Likewise route orchestration :326 calls `boundary.close()` before returning; the case FD census at :525 occurs after this teardown. Its actual before/after `[0,1,2]` receipts therefore prove no residual descriptors **after harness rescue**, not that `call` closed its own process pipes.

Concrete static counterexample: omit the driver's process-pipe closes while leaving capture closes, group kill and reap intact. The wrapper's final closed assertion still becomes true because `p.close()` supplies the missing closes, and the outer census can still match. The route census is masked in the same way. No mutation or execution of that counterexample was performed. This objection does not apply to every descriptor: the stable-reader/bootstrap checks and capture-file behavior provide separate useful evidence. It specifically affects the process pipes whose closure was part of Astra group3 and the final pipe-close repair.

The final driver visibly attempts both pipe closes before its failure/group-kill decision, and the final `wrapper/pipe-close` receipt demonstrates that an injected close exception is latched and causes one group kill. Those source/control-flow facts do not repair the missing observation of normal driver-owned closure. The handoff's census claim is numerically correct; attributing it to the driver alone would exceed the retained evidence.

**Required bounded response:** capture/assert each fake process's pipe-closed state immediately after actual `call`/`run` returns or raises, before `PipeProcess.close`/`Boundary.close`. Preserve unconditional teardown in finally so a failed check does not leak test resources. Retain the pre-teardown observation in the result detail. For injected close failure, distinguish a close that actually closed then raised (the existing fixture) from unresolved closure; keep the error/kill/reap assertions. Do not delete cleanup or weaken assertions to obtain green.

This needs only a harness change and fresh scoped HOST proof of the affected wrapper lifecycle cases plus full route success and a failure route on the unchanged final driver. Because the new observation applies across the wrapper variants, cover those variants rather than presenting one unrelated green as closure. Keep MAX100/5s/300s and all historical REDs. No real Docker, pinned recipe, product/native process, broad matrix repeat or driver change is implied. Fresh independent review of that concrete response remains required; this review performs none of it.

## F1–F4 and P1 responses

**F1 closed in source and relevant HOST behavior.** Both `private_publish`:36–38 and capture opening:155 use built-in open with exclusive mode and explicit os.open0600 creator. They no longer pass opener to Path.open. The real wrapper opening interceptor checks0600 on the opened fd before capture writes; private publication, existing regular/symlink rejection and scoped022 are exercised. Checkout alone gets022 and finally restores077, including nonzero/pending/completed/second-open failure. The later snapshot's Path.open has no unsupported opener argument.

**F2 closed in source, subject to H1's closure-proof gap.** All latched timeout/overflow/read/select/write/close failures after spawn trigger one group kill independently of poll. ProcessLookupError is tolerated; other kill failures do not skip the bounded wait. A final wait failure triggers the kill if none occurred, records unreaped status and fails. Final wait may be the failed reap attempt before that kill; no later successful reap is invented. Raw timeout-exited and wait receipts show one kill and bounded wait5, with rc unresolved for wait failure. Docker wrapper failure appends uncertainty and blocks destructive cleanup/release. No escaped-session/grandchild/daemon completion claim is supported or made.

**F3 closed for the requested narrow admission.** Explicit module guard precedes acquisition; explicit bootstrap guard precedes reading/compiling the recipe. Harmless subprocess fixtures exercise optimized refusal. Identity JSON carries debug/optimize, and host-side explicit admission precedes manifest/construction, repeated after. Existing base inspection is admitted before build; built-image inspection before create; created config/resource checks before start. Config rejection fixtures cover entrypoint/onbuild/healthcheck/volume/startup overrides at base and a created startup override. Shared admission source supplies the other created fields; fixtures do not prove current image truth. The created command is explicit. Memory-reserve executes before identity but is a stdlib diagnostic under already admitted startup config, not product execution.

**F4 closed in source and retained failures.** final-state declares pending even when failed=false. Completion requires valid complete inventory plus actual raw host exit0 and DONE. Inventory includes final-state, excludes checkout and itself, and has readback. State/inventory publication and state/inventory read failures reject; structured stderr and best-effort separate failure file retain owner/ID/uncertainty/lifetime. No disk-failure reporting guarantee or self-hash is claimed. Route host.exit files are in-process return codes, not OS raw exits; the CLI maps run through sys.exit, and the future actual capture must supply the real exit.

**STOP collision repair is within scope.** `DRIVER-STOP.json` avoids the case-insensitive STOP.json/stop.json collision without overwriting either failure or command receipts. The five rejected-output/nonzero initial REDs and scoped successful repairs support the diagnosis. This grants no new permission.

**P1 accepted unchanged.** A known fatal prelaunch issue means STOP before invocation, retain diagnosis and return to repair. One authorization is not a requirement to launch a deterministically invalid candidate. r12 stays consumed: one historical host invocation, zero external dispatches/construction,0/52 records; no replenishment or automatic retry.

## Actual HOST evidence and freeze continuity

Read all three freeze/summary files and99 individual result files; each result agrees with its summary. Parsed all immediate evidence JSON receipts (23598/4140/1198), read case stdout/stderr and route host stdout/stderr/exit, and checked retained capture sizes against recorded caps. Independently rehashed every entry of every retained evidence-hashes inventory (45/7/1 inventories respectively); all matched. These inventory counts exclude terminal cases where complete inventory is intentionally unavailable.

All proof roots are currently0700. Inspected private proof leaves outside the deliberately mode-varied fixtures/checkout/context were0600. Fixture `/private/tmp/pab-s-driver-global-r13-fixture-20261023/app.tar` is0400 in0700, SHA256 `4e0c0d958fcabbd367ae9bcea4253e4dbb0786a9b9cd7364303afff59fef751c`.

Proof roots share `/private/tmp/pab-s-driver-global-r13-host-` and suffix `-20261023`:

- `matrix-01`:83 cases,76 PASS/7 retained RED,100.954565s. Freeze hash `715386c97e53496d53c9666760b07d3c0824a2592e0b501bf2fac3ac2f31908f`; summary `4e8dc4076b4458c68b109e113af250bc5ba1a20c55e58d475b87f2802d552b39`.
- `repair-02`:9 selected PASS,21.284689s. Freeze `9cfe52142592ecd3ec52a2219dc7fba8549efc308c5a12e47de683818f383add`; summary `dc1045f0d0d358e18703237a4cf134c5096a196b467375714ee744aed8ed0dcf`.
- `close-03`:7 selected PASS,5.986722s. Freeze `e6db4d39d6bd36421b9ff8f6663348b469aa0658c21c64cf8035bd1076a1270b`; summary `654c20e24293ea21576b876b6b6ce5294a3558d3559a37e13aedf8d98f97b319`.

Their driver/harness hashes independently match Git snapshots99efde11/e0eeca93/1fa414bc. All freeze identities report `/usr/local/opt/python@3.14/bin/python3.14`, CPython3.14.7, optimize0/debugtrue; this is retained run identity, not a Python probe by this reviewer. Budgets were100 max cases,5s per case,300s whole run. Every case was below5s; maximum4.507197s. Total99 executions/128.225976s. All post-teardown FD censuses match `[0,1,2]`, with H1's attribution limit.

Latest distinct count84/all PASS is independently confirmed; it is not84 cases on the final freeze. Initial REDs are wait, four row corruptions, construction-nonzero and oversized-tar producer. Repair adds final-wait kill, scoped022 fixture correction, collision-safe STOP name and valid small oversized-header fixture, rerunning those failures plus success. Final change moves pipe closure ahead of the kill decision and adds pipe-close coverage; seven directly affected cases include success, exited-leader timeout, capture-close, pipe-close, wait, terminal inventory and full success. That scoped selection is causally justified and does not mix unrelated green to hide the final change. H1 explains why its closure observation still falls short.

Consolidated audit `/private/tmp/pab-s-driver-global-r13-source-audit-20261023/audit.json` was read as secondary evidence; its tar0664/Git0644 comparison correction is disclosed. The independent byte/manifest/hash checks above do not depend on accepting that audit's conclusion.

## Seven required groups and global boundary assessment

1. Private IO/checkout: meaningful real open/exclusive/mode and restoration observations, including publication/open failures. No chmod-after-creation substitute at the F1 seams.
2. Stable reads/bootstrap: real binary IO, mode/cap, leaf/ancestor symlink and metadata/length mismatch rejection; unlinked0400 read-only fd, shared-offset reset, exact bytes/argv/cwd/pass_fds and harmless literal bootstrap. Fake recipe materialization is explicitly not execution of the pinned recipe.
3. Capture lifecycle: actual nested call, real pipes/selectors, binary streams, exact cap/+1, EOF, alive/exited-leader timeout and spawn/read/select/write/capture-close/pipe-close/wait failures. Kill/reap/error receipts are useful. **Driver pipe closure remains unobserved before teardown: blocking H1.**
4. Startup/config: explicit optimize guards, debug/optimize JSON admission and narrow pre-build/pre-start fixture rejections; no current Docker/image qualification.
5. Complete route/output: final success has289 logical dispatches and one synthetic52 construction, traversing checkout/snapshot/copy/lane/build/create/binding/start/archive/LIVE/postchecks/full-ID cleanup/absence/release/finalization. Row removal/reorder/actual disagreement/restoration, cap and nonzero reject. Synthetic operands are deliberately not independently meaningful product operands; parser reliance on unchanged assertion-enabled pinned inner code remains essential.
6. Ownership/failure: early/copy/lane/partial owner, tag/build/create/start/exec uncertainty, malformed ID/foreign token/image, archive and stop/remove/absence/owner failures retain appropriate STOP/HOLD. Single create reconciliation, no second construction/archive, no foreign cleanup. Release requires exact lane inode/owner and cleanup postconditions. Finite lifetime/volatile evidence are retained honestly.
7. Retention/terminal: binary valid/duplicate/normalized/escape/link/UID/truncated/LIVE mismatch fixtures and inventory failure/readback checks. Oversized-header fixture rejects but may do so on tar truncation before size assertion; this is honestly scoped malformed-archive rejection, not direct64MiB/1GiB capacity proof. No large allocation/native durability claim is required here.

The external boundary has no real Git/Docker fallback: actual run receives the fake Popen/kill adapters, unexpected dispatches are latched as harness errors, and allowed argv/cwd/fds/masks/call counts are checked. The only real subprocesses are bounded harmless bootstrap/optimized-startup Python checks. The adapter derives literal inner program fixtures from driver source, so it verifies routing, not their independent correctness. Actual private IO, regular reader, snapshot machinery, JSON, tar/output audits and orchestration remain live host behavior.

CLI defaults still select real Popen/killpg/fixed canonical lane; no CLI/env dry-run bypass exists. Optional blob caching is keyed by admitted immutable OID while every filesystem read/mode/metadata/byte comparison and repeated HEAD/status stays fresh. Git hooks are disabled per outer invocation and filter config is recorded; recipe-internal Git/environment assumptions remain disclosed. No broad new Git policy barrier is invented.

Global source inspection found no additional demonstrated fatal API/control-flow/sequencing defect beyond the required HOST observation gap. Preserve the established limits: per-command wrapper clamps to60s plus bounded reap, not a total-driver1800s budget; whole-file inventory can consume significant cooperative host RAM/IO; remote completion is not host CLI completion; regular LIVE map is not a concurrent whole-filesystem snapshot; inherited image layers and installed package/startup truth need actual evidence. These are not newly qualified by fake success.

## Next gate and known-fatal STOP protocol

Return the single H1 objection with its required bounded response. Do not launch on this NO-GO. After corrected scoped HOST evidence and independent review close H1, the parent may assess readiness to freeze exact driver/Dockerfile/review/argv/auth/private canonical paths for ONE construction. This report neither grants that authorization nor self-launches.

Before any future launch, STOP on known incompatible APIs, missing required HOST seam, mismatched freeze/hash/review/authorization, optimized or wrong host interpreter, unsafe/symlinked or occupied private paths/lane, unresolved ownership, inadequate declared resources, or known incompatible image/startup config. Distinguish prelaunch zero invocations from an already consumed actual attempt. After dispatch, any failure/uncertainty is STOP/HOLD with actual residual state, no blind retry, false cleanup, reconstructed receipt or product acceptance.

A future independent actual review must inspect real263 bytes/modes/manifest, Linux3.14.8/42-lock/decoder receipts before and after, actual raw inner exit0 and52 complete ordered records, all-/tmp binary archive/LIVE equality, actual OS driver exit/DONE/inventory, private integrity and exact-owned stop/remove/absence/lane release. HOST fake success supplies none of those Docker/pinned-recipe/product receipts.

Historical4 passes/1 failure/250 unrun and Sa;Ss; without Sf; RED remain unchanged. S/all255/native unwinding/FULL, actual acceptance, H/P/overlap/grouped/integration/maintenance, blocked-pipe/storage STOP, installed/platform/release/hardware/browser/acoustic/activation remain separate. No historical tree, evidence or resource was modified, cleaned, accepted or released.
