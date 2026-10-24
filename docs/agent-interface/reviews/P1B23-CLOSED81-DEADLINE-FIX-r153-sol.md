# P1B23 CLOSED81 deadline checker fix — r153 Sol

2026-10-24. Bounded checker-only SOURCE/inert F1 correction complete, pending parent fresh independent Sol review before adoption/private preparation. No self-GO. Source unit still81 UNRUN.

## Identity and scope

Read FULL r151 and root r152. Own initially tracked-clean checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-ee030f4e`, branch `t3/fix-closed81-deadline-checker`, initially `c8bd933f63bba7e9363768f3970dc174642df52f`; reset only this own branch to authorized candidate docs `185021cbf3496c3d72a800993d18198cf0f3dd65`, source parent `fc9f987b746a7d36592498764475419600a3c147`. Historical `/t3-c19701a1` untouched. No applicable AGENTS.md found by checkout/nearby search. Memory registry search had no relevant hit; no memory premise used.

Checker commit `86c7c744fd4c0cf38caeaf51ae77cde9209c37df` changes ONLY `diagnostics/b-controls/check_closed81.py`. This report is committed separately; only report copied to root. Runner/HOST/PLAN/manifest/guard/controls/product unchanged. No additional owning defect found; no scope expansion.

## Corrected evidence

Old operands are retained as `old-deadline`: outer0, result pytest_exit1/accepted false, passed phases and exact selection, clock exactly deadline. Actual extracted owning ordered loop catches `STOP B result receipt`, receipt `RuntimeError: STOP B result receipt`. This reproduces why the old generic STOP assertion falsely claimed deadline coverage.

Corrected `deadline-correct-source` supplies outer0, result pytest_exit0/accepted true/collected true, exact singleton selection before fixtures and setup/call/teardown all passed, then advances fake clock EXACTLY to deadline. Both caught exception and receipt failure must equal respectively `STOP B node completed after deadline` and `RuntimeError: STOP B node completed after deadline`. Assertions require empty successful prefix, one attempt019 and no020 directory. Separate result-exit1 asserts exact result-receipt reason, empty prefix/one attempt/no020. Existing exhausted campaign deadline case remains separate; first/middle/last outer-exit1 and all81 fake success checks remain.

Mutation counterproof reparses actual runner AST and extracts the actual owning loop, removing ONLY its unique `require(clock() < deadline, 'node completed after deadline')` expression from its try body. Same deadline fixture/oracle is applied. Mutant accepts019 and dispatches020; capture raises `AssertionError: deadline guard escaped: dispatched020`, which is required by the sensitivity check. Correct source stops019 with exact deadline reason. Retained mutant `deadline/019` PASS and attempted020 distinguish this from canned PASS or cloned predicates. No owning source changed and no whole helper imported.

## Verification and provenance

ONE affected checker execution: external stdlib `subprocess.run(['python3','-B','diagnostics/b-controls/check_closed81.py'])`, real redirected stdout/stderr and raw returncode0. No checker-selfwritten top-level stdout/stderr remains. No operational helper used. Seven coverage groups PASS, including all197 mapped current byte hashes (193+4); manifest itself and all mapping/provenance content unchanged. `git diff --check` passed. No unrelated checker or pytest collection run.

Private evidence parent `/var/folders/gk/6rj578d95ls341449hb55pj80000gn/T`:

- `pab-closed81-inert-4wwl7zmi`: actual extracted-function fixtures/source captures; receipt SHA256 `5a2dd9b698ec9c9f3de094594663990f37d75fa02157c931a997911a78469c63`.
- `pab-r153-os-capture-j2q4ymkj`: external process stdout, empty stderr, `os-receipt.json` with raw returncode0 and stream digests; receipt SHA256 `eb403085ae581093a195c6fadcf920dcd6e7e78e3ab24d7782558e54e9bdbe9e`.
- Mutation counterproof SHA256 `f8b13a6b1780f9b764d6e79900d2f0023b1d45361336adb8538ce00c2518fd9e`.

Fresh directories0700 and leaves0600, currentUID, regular single-link/no symlinks verified for both trees; bounded O_NOFOLLOW reads and exclusive O_NOFOLLOW creations. No cleanup. Old retained fixtures/receipts untouched. As r152 establishes, r151's old OS0 claim remains author-reported; its self-written streams are not an external OS receipt. New external receipt applies only to this corrected checker execution.

Pins SHA256:

```text
runner   f60ede0e96dff3732e0dfd303bdd54bd38f6748ddb329a0dff6c67ca162f27ab
HOST     d1d7efaa2911130fd02d20f38ddc1ecff95fd5b1ff83ecdfebb33d2150a44cfc
checker  f0228dd2f2857b4af1522e71f7ec6819ec6463ae5a3358991b0a1f89ed6416dd
manifest c2c0a7843888be0dc6197381d60a709fb3120d0e56fd35785251d588d317441c
```

## Parent-owned gates

No Docker/build/lane/runtime/private operative bundle/product guard/helper imports/delegation or resource probes. Inert81 success is not runtime qualification. Parent fresh Sol review next, before adoption/private preparation; no operative bundle authority. Capacity hard6GiB admission remains future, native cause UNKNOWN, C/T2 unintegrated, r135 diagnostic campaign CLOSED, historical17 PASS/018 FAIL/81 UNRUN and separate017/current018 PASS unchanged. Lane/resources/proof remain HELD under owner `c9b6506a-0f20-4440-8ddd-be41f35bb82b`, carried from reports without live probe. No cleanup authority.
