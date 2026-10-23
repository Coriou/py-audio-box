# P1B23 dummy build actual receipt audit r20 — bounded decision

2026-10-23. Independent Sol low READONLY audit. **NO-GO for persistence qualification or product dispatch. GO for parent acceptance of this attempt as a conclusively failed build and for a separately assigned exact-owner reconciliation/lane release plus cohesive SOURCE+HOST recipe repair.** The failed solve is terminal; an indefinite unresolved-build HOLD is not supported by this concrete receipt. This review executes neither reconciliation nor repair. The ONE actual authorization was spent.

## Scope and verified pins

Sole write is this report in the parent root, no commit. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/p-one-storage-r18-20261023` is clean at documentation `7df2dab65c710730dd87c1e36cae310fb82fb211`, executable `0cc06400d77323796e20345b864b01873c0c9c96`. Parent root HEAD `dcdce100fe9d7695855d0fa35ee3957112e0971e` has unrelated untracked STT artifacts, left untouched. Checked ancestor and scoped AGENTS inventories for parent/candidate; none found. Read FULL current ACTUAL-r19, GLOBAL-SOL-r19, Astra STORAGE-PROPORTIONALITY-r16 and candidate FREEZE/HANDOFF/QUALIFICATION; focused reads recovered truncated combined outputs. Inspected actual preparation/build/validation/journal/finalization source and affected HOST recipe checks. A lightweight memory registry search supplied no relevant evidence.

Only text/Git and stdlib file/JSON/stat/hash inspection; no candidate/product imports, pytest, build, preparation/execution rerun, DB open, cleanup, tagging, process signals or delegation. **Zero new Docker queries**, within the allowed maximum three. No historical proof/worktree changed. This is an actual receipt audit plus targeted source repair advice, not a renewed global source review.

Private proof: `/private/tmp/pab-p-one-r16-576d34be-5824-4ec7-b716-6fdf17b1c04f`.
Private OS provenance: `/private/tmp/pab-dummy-actual-r19-os-gf5qgbrm`.
Plan SHA256: `fb8fd52762b698d636b4d04746ae6bbf3c01338400ee44a661f01fcbe4ec1044`.

Independently rehashed every file-table entry in both r19 reports, all plan input files, original manifest193+2 payloads, generated context193+3 payloads and exact context source file set. Rechecked all19 evidence-hashes entries, all post-execution-audit proof/OS file entries, both slot payload checksums, authority bindings and existing independent-report digest. Both private trees remain directories0700/leaves0600 without symlinks. The supplemental audit itself hashes to `f9b3f3384e72191c67b6e4ca1fe05895afcd0983d98ae4efc2438b6394b35ae3`. No image.id exists. Source hash/provenance statements in r19 are supported. The stale source/documentation constants in the owner identify preserved prior inputs, not the new executable; current plan hashes separately bind actual program bytes.

## Exact actual execution and terminal evidence

OS records bind candidate cwd, `PYTHONDONTWRITEBYTECODE=1`, interpreter argv `python3.14 -B diagnostics/p-one-node/qualify.py`, and complete waits:

- Preparation suffix `--prepare`: PID71467, 21:10:28.996586–21:10:29.252050 UTC, exit0, 0.255517s.
- Execution suffix `--execute-plan <PROOF> --plan-sha256 fb8fd52762b698d636b4d04746ae6bbf3c01338400ee44a661f01fcbe4ec1044 --review <OS>/independent-review.json --authorization <OS>/parent-authorization.json`: PID71944, 21:11:05.757072–21:11:08.212287 UTC, exit1, 2.455313s.

The external JSON records honestly bind parent adoption of existing independent r19 review and conditional ONE-DUMMY-ONLY authority; they do not impersonate a new independent concrete-plan reviewer. Review SHA `59630aff9ab8e7fda1ed9602d2ba2bb7525af63f8e7c6bcc8f2fe937f4cc5e76`, authority SHA `3ec59e695a23646ad31ed993729c04071f9ecb74e86363f62415cedd88b984b5`, exclusive once-marker SHA `ea5eee452e881222b4cd59c9cac7d416023e1143674e0d8e0c0cbd9f48f9e9df` all match.

Exactly three captured Docker CLIs, all COMPLETE/leader_reaped/descriptors_closed, none group-killed:

1. `docker image inspect sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb`, exit0; base Linux/amd64/full ID admission.
2. `docker build --pull=false --network=none --label pab.owner=98319dd5d545bba225c960a94cba0ea07b45374a21f34dab --label pab.invocation=576d34be-5824-4ec7-b716-6fdf17b1c04f --iidfile <PROOF>/image.id <PROOF>/context`, exit1; stdout0/stderr1157 bytes.
3. `docker image ls --no-trunc --filter label=pab.owner=98319dd5d545bba225c960a94cba0ea07b45374a21f34dab --filter label=pab.invocation=576d34be-5824-4ec7-b716-6fdf17b1c04f`, exit0; headings only49 bytes.

Build stderr SHA `a67d9f2c6e1619a218c5b708646db5d2b2f01d85d6d7c87d6e543515ff7f025c`; build completion SHA `519f864e9533f31fdebc6ecfec7598ad6f7871610889978b08a273890e6d2017`. Raw terminal diagnostic:

```text
#0 building with "orbstack" instance using docker driver
#2 [internal] load metadata for docker.io/library/sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb
#2 ERROR: pull access denied, repository does not exist or may require authorization: server message: insufficient_scope: authorization failed
ERROR: failed to build: failed to solve: sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb: failed to resolve source metadata for docker.io/library/sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb: pull access denied, repository does not exist or may require authorization: server message: insufficient_scope: authorization failed
```

**This is a returned fatal solve result, not a lost acknowledgement, timeout, killed transport or EOF without a result.** Together with exit1/reap/closed streams it establishes completion of this build request as FAILED at source metadata resolution. The generic `remote_completion=false` field records that the capture layer does not infer daemon completion; it does not invalidate the explicit BuildKit terminal error. No image export or recipe RUN was reached. Backend/cache bookkeeping may remain and is intentionally retained; that is not evidence of an active dummy/product workload. No assertion of zero backend bytes is made.

The empty label query alone proves neither completion nor absence. Here it merely corroborates the terminal fatal result and missing IID. The frozen journal remains `SENT-UNKNOWN` because qualify.py catches every build exception identically, including a completely received nonzero fatal solve. Preserve that historical fact; supplement its interpretation rather than rewrite it. Do not generalize this finding into “any exit1 means terminal” or weaken genuine lost-response HOLDs.

Supervisor COMPLETE/worker_exit1, acceptance=false; final slot STOP-OR-UNKNOWN-HOLD, retained=false, lane_released=false, objects={}; handback FINALIZATION-PENDING. These are failed-run records, not persistence success. No volume-create, container-create/start, dummy, product, runtime identity program, readers or removals were dispatched in the reviewed path/captures. There is no dummy volume or container to retain and no acknowledged derivative image. This is source-and-receipt evidence for this invocation, not a refreshed whole-daemon inventory.

## Smallest bounded reconciliation and lane release — recommendation only

The lane currently exists at `/private/tmp/pab-worker-w1-docker-lane.lock`, inode **[16777223,1007090824]**. Its sole entry is owner.json, equal to the complete plan owner (including token, invocation, PID71467 and old source/documentation pins). This audit left it held.

Parent can assign one exact-owner finalization action, without retrying the plan:

1. Revalidate these retained receipt/plan hashes and the exact lane owner/inode immediately before action. Publish a small supplemental private reconciliation receipt that classifies the sole sent build as terminal FAILED from the explicit fatal solve, lists no dispatched volume/container operations, missing IID, preserved proof/context and intentionally unassessed cache overhead. Keep original slots, journal, once-marker and r19 reports unchanged.
2. If a current corroborating inventory is desired, use one bounded exact two-label image lookup only; inspect only returned full IDs if any. An unexpected result is a concrete discrepancy to retain/report, never permission to delete. No repeated empty lookups, global inventory, events watch, builder restart or cache investigation is needed to establish this already-returned failed solve. Do not make a new empty lookup the completion proof.
3. Record exact-owner release intent and, with owner equality and inode rechecked, remove only that owner.json and empty lane directory using the existing owner-bound release discipline. Record completion/absence in the supplemental handback. Refuse symlink, inode/owner change or extra entries; no takeover, force, proof deletion or other cleanup. An append-only supplemental receipt is the reconciliation of historical SENT-UNKNOWN, not retroactive success.

Retention for this failed-before-volume attempt is retention of all actual host evidence, which is verified; requiring nonexistent volume/archive retention would create a false permanent gate. Parent acceptance of this audit supplies the concrete terminal decision needed for release. No additional generic actual-review round is needed just to repeat this conclusion. Release is not performed or authorized by this reviewer and does not authorize a new build.

## Shared defect and cohesive repair — SOURCE + HOST next

Concrete blocker: `launcher.py:364` generates `FROM ` + BASE, where BASE is a raw local config/image ID. `qualify.py:75` reuses that context and `qualify.py:308` independently requires the same raw-ID recipe. Thus both dummy qualification and the product launcher's preparation proposal share the defect. The product launcher does not itself dispatch a build; a separately executed build of its proposal would hit the same bad reference. `host-check.py:702` currently asserts `FROM ` + L.BASE, so its green fixture codifies the defect; the qualification fake accepts build without Docker reference semantics.

Recommend one fresh successor SOURCE+HOST assignment across launcher, qualify, both affected HOST harnesses, manifest/hash bindings and FREEZE/HANDOFF/QUALIFICATION as necessary; preserve all historical trees and all193 product payloads. Fix once through a shared recipe helper used by preparation and validation:

- Keep full raw BASE ID for exact inspect comparison; **never emit raw `sha256:<64hex>` as FROM**. Select a deterministic invocation-owned local repository:tag (valid Docker reference), frozen in proposal/plan together with expected full BASE ID. Do not substitute an unverified existing tag or confuse image ID with registry manifest digest.
- In a later explicitly authorized actual path, inspect the pinned base first; require the chosen fresh owned tag absent, journal its creation, tag only the inspected full BASE ID, then inspect the tag and require exact Id/Linux/amd64/startup admission before build. Reverify immediately before the sole build under the lane. Freeze tag argv, reference, ID and labels in the plan; mismatch/lost tag response stops with exact reconciliation, never overwrite/retry. Preserve the tag under the same no-tag-deletion rule; report it as an owned retained artifact. No tag/build operation belongs in SOURCE+HOST preparation.
- Retain --pull=false/--network=none and the metadata-only recipe; these flags did not prevent registry source resolution in r19. State that native local-tag usability remains the actual obligation. Retain derivative IID/owner-label verification and full image-ID container binding. No floating base, pull/install, runtime root setup or product assertion change.
- HOST must execute the actual shared recipe/context/plan validation functions for BOTH qualification and product preparation. Assert exact owned tag FROM and separately pinned expected full ID; explicitly reject raw-ID FROM even when the base inspect fixture is valid. Replace the old assertion that blesses it. Do not merely search for a new literal.
- Exercise actual adapter ordering with finite fake externals: base inspect -> tag absence -> exact-ID tag -> tag inspect -> sole build; wrong tag ID/startup, collision and lost tag acknowledgement must refuse build before dispatch and preserve state. Plan/recipe/tag tampering must fail validation. Cover complete fatal-solve response separately from timeout/lost/truncated response: only the former may produce terminal FAILED; the latter keeps UNKNOWN and no retry. Assert no volume/product/readers follow either build failure. Retain existing meaningful reader/export/ownership coverage without rerunning unchanged campaigns.

This is one concrete repair with affected HOST regressions, not a new proof framework or quota prerequisite. After its ordinary independent complete source/HOST review, parent may authorize a new bounded dummy attempt under the existing r16 mechanism. It must establish actual initialization, stopped persistence and two-reader equality before the one product diagnosis; this failed attempt supplies none of those results. Once that receipt is green and independently examined, proceed to the intended bounded product work under parent scope, without another construction campaign or optional microreview loop.

**Disposition:** failed build conclusively ended; historical journal conservative; exact lane still held and releasable through the proposed owner-bound finalization. Repair shared base-reference construction before any next actual. Persistence, product diagnosis/acceptance, H/S, native/platform/ext4/durability/T2/release/activation remain unqualified; P47 expected RED and historical PID229 UNKNOWN remain unchanged. No cleanup, mutation, retry or runtime authority is exercised here.
