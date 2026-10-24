# P1B23 CLOSED81 — independent SOURCE / inert-retained review r152 Sol

2026-10-24. **NO-GO for acceptance of the complete r151 SOURCE + inert verification checkpoint: one bounded checker-coverage defect (F1).** No closed81 runner selection/budget defect found. Correct the node-deadline fixture and its coverage assertion before a fresh independent review. This is not an operational GO; 81 remains UNRUN.

## Identity and permitted method

Independent gpt-6.1-sol, low. Sole write: this root report. Read full r151/r147/r148/r146/r150, recovering truncated sections, the complete candidate runner/checker and full five-path source diff. Read the orchestration protocol. No applicable AGENTS.md found in the relevant checkout search; unrelated project instructions do not apply. Memory registry lookup returned no relevant match; no memory premise used.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-c19701a1`, branch `t3/closed81-source-selector`, remains clean at docs `185021cbf3496c3d72a800993d18198cf0f3dd65`, whose sole source parent is `fc9f987b746a7d36592498764475419600a3c147`. Baseline is integrated root `aaafb7f100db17b5bb28fbea8e75bafa8e649192`. The retained reflog records initial `ec78661`, a same-HEAD reset (no historical movement), rename, fast-forward to aaafb7f, source commit, report commit. Graph/diff agree with preservation rather than historical reconstruction; no claim of proving absence of every historical command globally.

Baseline→candidate changes exactly `diagnostics/b-controls/{runner.py,HOST.py,PLAN.md,check_closed81.py}`, `diagnostics/p-one-node/source-manifest.json`, and the r151 report. Root observed at `c8bd933f63bba7e9363768f3970dc174642df52f`; its advance from aaafb7f changes only the ledger and r151 report. Root was tracked-clean before this sole write. Existing unrelated untracked STT material was neither opened nor changed.

Read-only Git/text/hash, stdlib AST/literal analysis and source compilation only. No candidate/helper imports or function execution, checker rerun, pytest/collection, Docker/build/lane/resource probe, private operative bundle access/edit, cleanup, integration, commit or delegation. Two review-analysis assumptions about the older fixture layout failed (it has no top-level stdout and uses `exit1-019` names); subsequent read-only inspection corrected those assumptions. No checker was invoked.

## Blocking finding F1 — deadline fixture stops at result validation

Location: candidate `diagnostics/b-controls/check_closed81.py:158–176`, specifically line168. For `fail == 'deadline'`, `failed_index` is19, so the fake result is `pytest_exit=1`, `accepted=false`, despite outer exit0 and three passed phases. The fake clock is then advanced to the deadline. Actual `runner.ordered` validates the result before its `clock() < deadline` guard, so this case stops at the result predicate and never exercises node-deadline rejection. The catch asserts generic STOP only, allowing the incorrect coverage claim at line184.

Retained final `deadline/019/result.json` confirms exactly those operands. Its `receipt.slot` says `failure: RuntimeError: STOP B result receipt`, not `STOP B node completed after deadline`. Thus this is a demonstrated oracle/coverage defect, not a hypothetical source failure. Removing the actual node-deadline guard would still leave this purported deadline case green. The exhausted-campaign case separately exercises pre-dispatch budget refusal; it does not close this gap.

Bounded fix: make the deadline case supply outer exit0, result pytest_exit0/accepted true, exact collected selection and passed setup/call/teardown, then set the fake clock exactly to its deadline. Assert the specific node-deadline failure as well as STOP, empty successful prefix and absence of020. Keep result-exit1 a distinct case and assert its result-receipt reason. Retain a new inert receipt under separately authorized implementation verification; update r151's evidence claim/pins as appropriate and submit the changed checker/evidence for review. No runner/guard/product repair is requested by F1. No correction or rerun occurred here.

## Independently verified source behavior

All four requested SHA256 pins match:

```text
runner   f60ede0e96dff3732e0dfd303bdd54bd38f6748ddb329a0dff6c67ca162f27ab
HOST     d1d7efaa2911130fd02d20f38ddc1ecff95fd5b1ff83ecdfebb33d2150a44cfc
checker  544e495d070d2f82790285545eca8c8a82fa45d7b32f582243caf37e9c5fe906
manifest c2c0a7843888be0dc6197381d60a709fb3120d0e56fd35785251d588d317441c
```

Independent literal expansion of actual GROUPS/control parameters and inspection of ID lambdas reproduces the complete100 PLAN table and canonical JSON digest `94c4b4aa2c91245277548f0c03e89c3ac83ecf08ae9e6cb0b5f9dda25111a992`. The new selector first authenticates complete controls bytes and full100 AST, then that full100 digest; its literal tuple is exactly019…099,81 unique nodes,2135 seconds, identity `original100-minus000-through018-v1`. No arbitrary range/resume/skip fallback exists.

CLI `closed81` propagates explicitly through command/create/inspect/ordered/campaign. Mutual exclusion guards prevent conflicting modes; unknown CLI modes cannot fall back to campaign. Expected81, selected indices and successful ordered prefix reach intent/terminal state. Existing inventory, directed/complement selectors, environment, node argv, pytest argv, Selection and node owning ASTs equal baseline exactly. Full100/closed99/directed018 retain caps2375/2340/35 and totals2600/2565/260.

Actual PID1 consumers use phase_budget endpoints60/2195/2225/2285/2360 for precheck, dispatch observation, reap, postcheck and publication/alarm. Actual campaign remains statically inspected only. HOST's added pure function correctly yields writer2378, endpoints60/2438/2513/2558/2633/2670/2700, capture2715 and supervisor2730; it implements no operative private adapter.

Outer acceptance still requires node exit0, exact singleton before fixtures, result pytest_exit0/collected/accepted, exactly passed setup/call/teardown, matching retained stream hashes and completion before deadline. Internal refusals remain outer PASS obligations; outer exit1 cannot aggregate as success. Exceptions preserve STOP and abort the loop before later dispatch. The deadline source guard is present; F1 concerns its claimed inert exercise.

Manifest comparison after replacing only the old runner digest equals baseline exactly:193+4=197 mappings, membership and all historical metadata/commit/launcher provenance unchanged. Independently rehashed every mapped current file. Checker/HOST/PLAN are not newly mapped. Three Python sources parse/compile without execution; read-only diff whitespace check passes. Entire five-path delta leaves guard, controls, product, target and current018 decoder unchanged. Consequently049's natural raw18944/E/EMPTY/AE/no F/T premise and six fresh B-C current018 natural74 baselines remain mandatory; no raw9 waiver or r145 receipt substitution is introduced.

## Retained inert evidence and limits

Final retained directory `pab-closed81-inert-rzwm46bt` under the supplied temporary parent has receipt SHA256 `93b69f0c3f75d0741c944fea041669c218a9334831a805f0d100fe5403b9fbf1`. Raw runner/HOST/checker captures equal current pinned bytes. Inspected retained tree ownership/modes: currentUID, directories0700, regular single-link leaves0600, no symlink encountered. Bounded no-follow file reads were used for the final tree. Verified retained binary stream sizes/digests against receipts and captured node argv/indices/deadlines. All81 fake receipts PASS; first/middle/last outer-exit1 cases retain1/45/81 attempted nodes respectively, exact successful prefixes and final STOP; result-exit1 stops at019. No downstream fixture directory exists after each early failure. F1's misleading deadline STOP is separately identified above.

Earlier successful `pab-closed81-inert-p9zmsy3d` remains present, receipt SHA256 `77063b24c70293d490cbea43cce6fb5457ece4cc729cea7090ebe38bbc6e25ff`. Its captured source pins match its own receipt, not the final pins. Verified all81 and `exit1-019/063/099` retained receipt/stream hashes. Compared raw earlier/final sources: runner only removes unused node_budget assignment; checker adds result/deadline cases, argv captures and self-written top-level stdout/stderr. No failed prior checker run was established by these retained artifacts.

The checker extracts actual named owning functions/constants, rather than cloning the selector/ordered loop, and uses the pre-existing full100 PLAN table as independent oracle. Extra/duplicate/reordered/missing/cap/excluded-row and full-source drift negatives, command/create/inspect and old-mode/budget checks are substantive source-inspected checks. Campaign assertions are explicitly static. Final top-level stdout equals receipt and stderr is empty, but both are written by the checker itself before print: they are not independent process-supervisor stdout/stderr or an OS exit receipt. r151's OS0 statement is author-reported; retained PASS proves the checker reached receipt creation, within this evidence tier. This distinction does not independently block narrow SOURCE acceptance. No actual81 runtime or terminal/census/quiescence qualification follows.

## Parent handoff and unchanged gates

Close F1 in the standalone checker, then obtain independent acceptance of the corrected exact checkpoint. Parent owns adoption; this report performs none. Later cohesive private preparation must first freeze exact successor paths/hashes for bootstrap/entry/diagnostic-host/launch-boundary, binding-transform/bind-data, inventory/snapshot/manifest/context, namespace/paths/build-owner/bindings/plan/launch DATA and actual adapter/phase/receipt/PAX/external authority consumers. Unnamed dependencies remain STOP. Replace singleton expected1/index18/sole018 directory/diagnostic assumptions with exact81 ordered per-node phases/receipts and failure prefixes; preserve current018 decoder qualification only where applicable. Propagate writer2378/HOST2700/capture2715/supervisor2730 through actual consumers, test missing/new authority and source/image mismatch there. Old ONE018/authorize_one_directed018 authority must refuse81.

Existing018 image does not cover the mapped runner delta. No operative bundle, build/image decision, launch authority or capacity admission is accepted here. Fresh later hard6GiB admission remains required; stale58.5MiB margin is not admission and permits no cleanup. C/T2 unintegrated, native cause UNKNOWN, historical17 PASS/018 FAIL/81 UNRUN and separate017/current018 PASS remain distinct. r135 diagnostic campaign CLOSED; no automatic instrumentation. All resources/lane/proof remain HELD under owner `c9b6506a-0f20-4440-8ddd-be41f35bb82b`, carried from accepted reports without live probing.
