# Owning P/K nonempty successor — independent Sol-low review

2026-10-22 · gpt-6.1-sol / low. **NO-GO / owning blocker CONFIRMED** on exact candidate `a27f78b5e2e09a98940b9c543474337b511ce616`, frozen executable source/tests `e88ee066521b2ded66b611ac073bb6106cd67086`. C remains unintegrated blocked WIP. This review establishes a production-preparation coherence defect under explicit E1 host/OS models, not actual host authorization or full C acceptance.

## Identity and preservation

Reviewed ABS `/Users/ben/.txcode/worktrees/py-audio-box/t3-1e1cddba`, exact requested HEAD, empty `git status --porcelain` before and after tests/probes. Baseline `3dd545f70ee1f7c48451ddeba4834b628040a3b2`. Frozen-to-candidate delta is only 77 appended handoff lines. Baseline-to-candidate paths are exactly recovery.py, p1-operator.py, dispatcher.py, runtime_supervisor.py, recovery tests and RECOVERY-P1B2. `git diff --check 3dd545f70ee1f7c48451ddeba4834b628040a3b2 HEAD` passes.

Independent byte checks confirm baseline recovery/tests/handoff are exact prefixes; recovery appends **187 lines**, tests 868, handoff 238. All five executable/test paths equal frozen bytes. Thus every integrated K/P definition, including the defective coherence predicate, is unchanged. Historical C's three other executable files equal candidate bytes. Historical worktrees remain clean at `58e3eb9f40a8db5ca162fe8f64922efb13f224c4` (t3-adbf212d) and `6cfe3973f41db92679531e38660d88cf72ce27eb` (t3-4a8727d4). No applicable AGENTS.md was found.

SHA-256: recovery `27edd68c115f11b1697b007afe6f99beb52ae1f61f79bb2c3e766a55eb92d073`; recovery tests `825aae87574e513e363007e43b07855e8c380553d7145cb152ceb0c689d194ea`; handoff `402c2988509f76a5b38148501035f6c760f3658f39c590928eb8c766a85d4147`.

Read canonical ORCHESTRATION-PROTOCOL, execution-ledger context, KERNEL-ASTRA, C-ACCEPTANCE-ASTRA, C-PREFIFTH, K-PREFIFTH and K-PREFIFTH-ENV-20261022 reports, and the entire newest candidate RECOVERY-P1B2 section from 1717 through EOF. Inspected production preparation/session/snapshot/coherence, canonical Core output and evidence/archive validators, resolution consumers, typed-reference projection, actual C/death/second-generation setup and the RED. Memory supplied only read-only/exact-checkout/evidence-tier guidance, never acceptance evidence.

## Fresh execution and independent proof

From the exact candidate, executed:

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTEST_ADDOPTS -u PYTEST_PLUGINS PYTHONDONTWRITEBYTECODE=1 TIKTOKEN_CACHE_DIR=/tmp/pab-k-prefifth-env-20261022/tokenizer-cache /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py -k lotc_red_nonempty_successor_preparation_owning_p -q -rs > /tmp/pab-successor-review-r1-red.log 2>&1
```

**4 failed, 1929 deselected in 10.25s; pytest exit 1.** No skips/xfails. The shell display command subsequently tails the log and exits 0; that is not the pytest result. All failures are the required B positive: C routes fail their death-hook-exists assertion at tests:5375; direct P routes fail with `b2_maintenance_child_failed` at recovery:469. A setup, fifth, exact Core oracle and inventory assertions at tests:5963–5980 passed in all four.

Independently inspected all four newly created roots, not merely supplied JSON, under `/private/var/folders/gk/6rj578d95ls341449hb55pj80000gn/T/pytest-of-ben/pytest-23/test_lotc_red_nonempty_success{0,1,2,3}`. Command form: `env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PY' > /tmp/pab-successor-review-r1-probe.json`, using roots from those paths and scope from protected `p1-init.json`. The stdin probe used `recovery.offline`, `snapshot`, `validate_b2_database`, `_verify_core_rows`, `_completed_prefix_receipts` and `_verify_finished_receipt_evidence`; asserted exact original dispatch tuple equality, `_resolution_outputs` equality and original bundle SHA-256; then independently invoked `_coherent_b2`. Exit 0. Every root has:

- Exact Core after-image, canonical history/archive and retained receipt validators PASS.
- Inventory reasons empty, exactly three recovery-evidence entries.
- One failed job, one retained reservation, original dispatch unchanged with receipt=null, outcome=null, ACK=0.
- One recovery, five steps, one plan (A only), two registrations; original frozen snapshot hash unchanged.
- `_coherent_b2` raises `incoherent_b2_predecessor`.

The supplied `/tmp/pab-c-resume-20261022/owning-p-proof.json` agrees but is supporting historical evidence, not this review's execution receipt. No B plan/recovery/epoch was persisted. No fake receipt, source/hash repair or caller-authored expected rows was supplied.

Probe development had two errors, neither suppressed: an unused import of nonexistent `service._util` failed; a first probe used a direct SQLite connection before `offline` without explicitly closing it and its final source-stability guard raised `snapshot_source_changed`. The successful probe removed that original SQLite open entirely and read scope from protected JSON. Neither failed probe is counted as proof; only disposable roots were involved.

A second short independent command with the same interpreter/environment traced only `_allocation_coherent` line events against the successfully read actual data. `/tmp/pab-successor-review-r1-branch.json`, exit 0: `result=false`, last lines `[1454,1455,1456,1457,1458,1459,1460,1461]`. **The supplied diagnosis is incomplete: first rejection is line 1461, not 1467.**

## F1 — exact recovered dispatch is rejected by preparation provenance

Blocking locations: `service/recovery.py:1437–1475`, specifically **1461 and 1467**; caller `_coherent_b2` :1478–1483 and `_preparation_prestate` :2773–2804, especially :2788. Reproducer tests:5963–5986; actual new-generation setup :5587–5611.

`_lotp_population` (:3678 onward) uses real Core submit/Admission/DispatchStore allocation to create nonempty running state. It declares retained high-water/checking prestate before preparation, as predecessor accepted P tests do; this is not preparation-time inflation. Actual P preparation produces A's plan/intent/snapshot, C confirms and owning K completes the five-step transition. The second setup uses actual registration and host-capsule producer paths, with explicitly modeled host observations covering both retained owner generations. Direct P routes bypass C request/resume logic and still fail. Existing common validators accept A's prescribed unsuccessful transition; the defect is not a failed A setup or a host-coverage rejection.

At line 1461, unacknowledged dispatch + retained reservation + original null outcome fails the ordinary reservation/outcome correspondence. Even if that were corrected alone, line 1467 rejects failed job + original null receipt. Both are intentional consequences of operator recovery, which preserves original dispatch/outcome/ACK/receipt evidence and records separate resolution authority. `_validate_core_output` (:1015–1061) explicitly permits only the unsuccessful state/latest reap change and retained reservation, not fabrication of a PID-1 worker receipt. The predicate ignores completed resolutions altogether.

This is an availability/required-history blocker: valid completed nonempty A cannot prepare B. It does not establish authorization bypass or production host readiness. It belongs to the shared owning P/K validator lane because integrated source is an exact prefix and direct P reproduces. A C-only guard, an empty A fixture or editing the RED's expected result cannot close it.

## Smallest safe owning correction, no implementation supplied

Limit implementation to `service/recovery.py`, appended behavioral tests and appended RECOVERY-P1B2 handoff. Retain schema, commands, budgets, identity, resources, original evidence and existing host/session boundaries.

Reuse/factor existing strict validated joins, rather than inventing a second permissive resolution test:

- `validate_b2_database` :1269–1392 validates canonical records/digests, exact plan/intent/deployment/proof/source bindings, epoch/step history, resolution target/binding/original-dispatch digest, complete expected resolution sets and retained historical rows. `_validate_archives` :1169–1220 binds actual snapshot schema and every source table, actual current plan, manifest/certificate and frozen hash. Do not consume a raw resolution row merely because its key matches.
- Reconciliation :132 and :337–353 already consumes completed-resolution identities only after strict current authority/context/generation/deadline/attempt, demand and publication joins; only the finite old reasons `unbound_dispatch`, `reservation_outcome_mismatch`, `terminal_receipt_missing` can be consumed, with the **entire exact current job row/context and reservation row equal to the accepted recovered after-image**. Any other reason remains a defect. Persistence :242–248 exposes validated completed dispatch resolutions; it is not by itself a complete current recovered-tuple oracle.
- Adapt coherence through a small shared read-only helper/internal validated context so only the exact bound original dispatch, accepted canonical source and completed recovered tuple can discharge the two ordinary mismatches. Preserve all ordinary authority, reservation coverage/category/configuration/demand, publication and receipt validation. Completed means the existing validated Core-completion contract; preparation must also retain its existing all-history/fifth/receipt checks, not downgrade them to row existence. Require proof of completed resolution, not pending intent/resolution insertion.
- A nonempty production B can resolve the same retained dispatch again: resolution selection must validate every relevant retained history and exact current recovered tuple without ambiguity or broad history exclusion. Preserve complete resolution-set checks. Do not require the current whole database to equal an old A output after legitimate later activity; compare the exact protected tuple relevant to the resolution and retain existing current source/history checks for new preparation.
- Snapshot :625–667 preserves schema/SQL/history/media/typed references and excludes only exact current proposal/prefix labels; :615 and :638 verify output rows before projection. Keep that complete projection and its bounds. Do not erase new typed-reference, command/event/media/history defects or let a new diagnostic borrow an old decision. Reconciliation's diagnostic matching :503–520 is exact reason/content matching, not general ACK authority.

No fabricated receipt/outcome/ACK, evidence deletion, resource reset, reservation release, high-water/budget change, replacement proof/hash/identity or new command/ack-away route. Blanket terminal/no-receipt exemptions are unsafe. Fixing only line 1467 does not fix the observed first rejection.

This is straightforward consistency with the existing Astra contract: exact completed resolutions already justify these recovered joins in Reconciliation, and KERNEL-ASTRA explicitly preserves original evidence and requires nonempty production histories. No new product/architectural waiver is needed. Escalate narrowly to Astra only if the implementer proposes changing the meaning of completion, resolution composition or recovered tuple eligibility; retain NO-GO meanwhile. Independent review and final integrated Astra remain required.

## Required exit matrix

1. Replace the diagnostic `allocation_coherent is False` assertion in the owning RED with the desired positive, without deleting it or changing A to empty. Actual nonempty running P → C → five-step K A must pass coherence and production B preparation through both C and direct P, two independent setups. Prove B's authoritative plan/intent verify/rehearse, B completes once and exact retained A/B evidence remains. No fixture intents or hash rewriting.
2. Fresh strict refusals twice, with writer traps and unchanged logical SQL/protocol evidence, for pending/uncompleted resolution; missing/wrong/extra resolution sets; original dispatch/binding/runtime/receipt/outcome/ACK tamper; recovered job state/reap/attempt/context/generation/deadline or reservation/category/demand/configuration tamper; new fault/integrity defect; changed source/history/archive/typed reference/media. Include terminal histories with genuine valid receipt, original terminal missing receipt without a resolution, and malformed existing receipt even with an old resolution. Never let old authority consume a new mismatch.
3. Preserve full current-source verification before ANY unfinished write at all reached fourth/certificate/finalization/pre-fifth prefixes. Original stopped-owner restart/new-registration refusal and actual-stage creation prerequisites remain. Valid unchanged/accepted-after-TTL positives complete once; attempted corrupt output stays quarantine. Retain original accepted-parent and four C/direct-K pre-fifth refusals, exact plan/proof/hash and fifth absence for refusals.
4. Preserve F1 original-owner checks, F2 actual receipt-only/SQL successor stages, F3 predecessor versus own receipt ordering, F4 every-completed-history descriptor/certificate validation in both directions. Completed A replay with valid B at every reached prefix stays observational: no B scan/capture/continuation/write and no demand for B's future output. Explicit unfinished B still validates full current source before writers. Genuine completed replay tolerates legitimate later current activity without restoring old journal/fence/resources.
5. After owning correction, independent review and integration/equality proof, run the complete current C nonempty A/B/durable/history matrix. Freeze and run target, four B2 suites and full service/release sequentially. This review ran none of those broad campaigns and grants no full preservation/C/release GO.

## Evidence limits and boundary receipt

The 187 selected C checks, original accepted-parent/four pre-fifth green rows and earlier K locked full receipts are attributed prerequisites, not freshly rerun here. Full C matrix, four B2 and full service/release **NOT RUN / no GO claim**. Production `_pinned_maintenance_host_recheck` remains mandatory `b2_pinned_launcher_context_unavailable`; OS/host substitutes prove E1 control flow only, not authentication.

All 19 inherited Linux cap/peer-credential mechanisms remain UNQUALIFIED. Actual Linux 4 MiB/ext4/Unix-VFS/SHM, actual pinned launcher/provenance/C1/C2 installed/clean-host, B3/global Astra, engines/hardware/browser/listening/acoustic and physical ENOSPC/power-loss/daemon/VM/controller durability remain pending. A1 Docker-inventory drift remains unexplained/unqualified; no cause/waiver inferred. Apple Silicon remains documentation-only scope. Freeze remains through host-observed fifth/matching outputs; completed history grants no current freeze release. Enrolled/public activation stays closed.

Only repository write: `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23-SUCCESSOR-r1-sol.md`. No source/test repair, commit, delegation, other-worktree mutation, install, Docker/network/model/audio/paid E2E, cleanup or limit change. Temporary logs/roots retained. Probe inherited RLIMIT_FSIZE was `(9223372036854775807, 9223372036854775807)`; the existing tests model child limits and do not qualify Linux enforcement. Parent preexisting untracked STT research was not changed. Return this confirmed blocker to the owning validator implementer; preserve both historical C worktrees.
