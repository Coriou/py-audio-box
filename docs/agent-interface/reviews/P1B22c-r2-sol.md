# P1B2.2c independent adversarial review — round 2

2026-10-16 · gpt-6.1-sol / low. **GO for bounded B2.2c source/E1 acceptance only** at exact HEAD `3f41113535c3a686e7d281aafe308ef82f1fa354`. Round-1 F1 is closed within the source and independent evidence below. No new blocking finding. This approves the private continuation through atomic Core completion and immutable Core certificate, stopping **receipt PENDING**. It does not approve receipt completion, full B2/P1, integration, installed operation, actual-host authority or activation.

## Identity and review basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-0f37e098` read-only. Initial and final candidate HEAD match exactly; status is empty. Frozen source/tests are `45f8cd7778b5d423c7536eb059e5d70dbed45e71`; the remaining HEAD delta is handoff-only. Actual baseline is `ac92b4175df53c03ea97672a894af17733dd4ac4`. Baseline-to-HEAD contains exactly eight authorized paths: recovery, persistence, resource_store, dispatch_store, dispatcher, reconciliation, recovery tests and RECOVERY-P1B2. Correction-only `4e53399..HEAD` contains exactly reconciliation, recovery tests and handoff. Imported `4e53399` service/worker/service-tests match the original reviewed `c5173055bce21101621f25fb4d868495a7b7e2c3` exactly. Protected Core/initializer/worker/contracts/reviews/ledger comparison against baseline is empty. `git diff --check ac92b41 HEAD` passes. No applicable ancestor/scoped AGENTS.md was found, including report destination ancestry.

Read full P1B22c-r1-sol, including its exact reproducer and every independent probe; full P1B2-BOUNDARY-ASTRA architecture, closed records, resume/authority/reconciliation amendment and exits; full candidate RECOVERY-P1B2 original contracts, every tranche/correction and latest F1 response. Also read orchestration protocol, P1B22b-r1 and R1 fault-authority/B1 round-2 context. Inspected complete production/test delta, immutable validation and projection, recovery scan/transaction/certificate paths, current dispatch filtering/allocation/replacement and A2 diagnostics/reference callers. Memory supplied general checkout/evidence-boundary guidance only; no old acceptance result is used as current evidence.

## F1 — high — closed

`service/reconciliation.py:297-356` now accumulates independent join failures. The three permitted old reasons are removed only if there is no strict current defect. The check at lines 343-349 compares **every SQL job-row column**, including complete job/context and principal/operation/key/fingerprint/retry identity, plus the complete reservation against accepted Core output. This is not a text-field blacklist. Existing Authority/binding, latest attempt, generation, context, deadline, reservation configuration/charge/state, receipt and publication joins contribute independently. Malformed identity remains a defect. Stable inventory diagnostics prefer a strict reason; summary retains the accumulated reasons.

Resolution eligibility is independently bounded: `service/recovery.py:990-1010` binds the original complete dispatch tuple/digest and exact completed resolution set; pending records do not consume it. A later change to original dispatch runtime/binding/receipt cannot borrow its old resolution. `_resolved_diagnostic` (`reconciliation.py:503-521`) requires completed identity, original reason and complete diagnostic content; raw history, typed references and unavailable records remain retained. Unknown schema/reference and missing-origin checks remain blocking. Ordinary inherited checking is unchanged.

Fresh independent production-writer probes used disposable roots, real Admission/DispatchStore allocation and `_pending_through_core`, followed by a single current-field mutation and **two separately opened Core/A2 sessions**. All 16 variants passed their expected assertions: unchanged control clean twice; text-only original F1, deadline, attempt staging, latest reap bit, SQL operation/key/fingerprint/retry_of, reservation category/configuration/state, dispatch binding/runtime/receipt, and unknown context field all block twice. The original text negative reports `dispatch_owner_mismatch` and `recovered_job_reservation_mismatch` alongside the retained missing/outcome/unbound reasons; its second startup also retains unacknowledged integrity. Dispatch tuple mutations refuse at Core construction with `b2_b2_dispatch_binding`. The seven candidate regressions additionally cover bundles, SQL principal, input hash, owner and generation, and assert unchanged original ACK/receipt/outcome.

For reproducibility, use the R1 report's exact stdin reproducer on this candidate: it now satisfies its required `assert result['blocked']`; repeat the same body with a fresh disposable installation per variant. The independent extension mutates one named field only after production completion, then invokes `Core(root)` / `Reconciliation(...).reconcile_startup()` twice. SQL-row variants use `UPDATE jobs SET <column>=? WHERE id=?`; reservation configuration changes only the decoded demand's configuration; original dispatch variants change one SQL field. No accepted record, resolution, certificate or digest was rewritten to make these tests pass.

No additional fix is requested for F1.

## Preservation and independent continuation checks

Both ordinary locks still span the continuation. Exact accepted prestate/proof/scope/all-owner coverage and original history/media checks precede transitions. Evidence/journal prefix precedes the distinct bounded recovery scan; no normal observation, borrowed token or constructor heals inherited checking. High-water is max(old, actual), cumulative charges/demand stay retained, free reserve/quota failures stay pending, and budget changes remain unsupported.

One Core transaction writes only prescribed unsuccessful job JSON/latest reap, retained reservations, accounting/new monitor/A2 state, exact append-only resolutions/retirement and the fourth step/new epoch. Terminal rows, events, artifacts, idempotency and original dispatch/commands/receipt/outcome/ACK stay retained. COMMIT precedes certificate creation. Ambiguous failures unwind/close; new invocations inspect actual state. Attempted missing/partial/wrong certificate output quarantines, exact output is re-barriered. No source or fixture relaxation was introduced by the F1 correction.

Independent post-COMMIT/pre-certificate probes stopped the real writer at `core-complete-commit`, then separately changed the owned bundle, validly rehashed prior B1 plan, or original receipt journal archive. Each refused twice after TTL before a certificate appeared (`b2_prefix_snapshot_changed`, `b2_before_history_binding`, `b2_prefix_quarantine`). Unchanged uncertified resume succeeded twice; subsequent certified replay twice had both `_recovery_scan` and `Reconciliation._collect` replaced with assertion failures and retained certificate bytes/inode. Four variants passed.

The current-data mutation matrix also replayed the completed private continuation twice before A2 for all non-dispatch variants: it returned the existing epoch despite current job/reservation changes. Thus A2 eligibility checks were not silently imposed on immutable completed replay. A separate real same-principal retry job was submitted and allocated after completion through Core/Admission/DispatchStore. Two completed replays, with both scans forbidden, returned the original result and preserved jobs, dispatches, events, reservations, monitor/accounting, recovery steps and resolutions exactly. Ordinary new activity does not replay old mutations or acquire a new recovery epoch.

Probe setup errors are not candidate failures: the first broader matrix stopped when its probe tried numeric addition to an ISO timestamp; corrected to a valid changed timestamp and all 16 cases reran successfully. The first retry-activity probe deliberately used a different principal by mistake and hit existing `invalid_retry`; the corrected same-principal retry passed. No candidate edits resulted.

The focused suite freshly exercises actual fork death/fresh-interpreter resume, denied fault persistence and inherited checking, SQL authorizer rollback, certificate barriers/no-remint, uncertified restart evidence, exact old diagnostics/degraded availability/strict retrieval, terminal retention/permanent publication retirement, pending filters/completed resolution sets and both supported startup modes. These preserve the relevant prior R1 campaigns; I did not redundantly repeat every earlier independent death/media/quota campaign outside the suites. Original R1 probe receipts remain historical evidence, not fresh results of this review.

## Fresh verification

Existing interpreter only; no installs, bytecode/cache disabled. Focused and full suites ran sequentially on unchanged candidate source/tests. Independent probes overlapped only local disposable-root focused/full work; no Docker/heavy lane ran.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**509 passed in 237.86s, exit 0.**

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**661 passed, 18 skipped in 278.71s, exit 0.** All skips remain unqualified. Independent corrected probes above exited 0. Passing counts are supporting evidence; closure rests also on the exact F1 reproduction, broader mutations and source joins.

Final candidate HEAD remains `3f41113535c3a686e7d281aafe308ef82f1fa354`, status empty, diff check clean. Public 1.3.0, Core 1, R1 protocol 2 and private P1 revision 1 are unchanged. Apply stays unconditionally unavailable; enrolled activation remains closed. Receipt completion is a later tranche, not a requirement invented for this review.

## Limits and write boundary

E1 local source/SQLite/filesystem/process and mocked host documents only. E1-mock does not authenticate actual host authority. No physical power-loss/ENOSPC/daemon/VM/controller, installed lifecycle, engine/audio/hardware or clean-host qualification. **A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.** Receipt completion, B2.3 trusted transport, B3, C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending.

Only `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B22c-r2-sol.md` was written as a repository artifact. No candidate/source/test edits, commit, subagents, push/deploy, installs, Docker/network/model/audio/paid/E2E runner, prune or user cleanup. Disposable test/probe roots only; preexisting root research and other worktrees preserved.
