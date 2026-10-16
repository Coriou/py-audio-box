# P1B2.2c independent adversarial review — round 1

2026-10-16 · gpt-6.1-sol / low. **NO-GO for bounded B2.2c source/E1 acceptance** at exact HEAD `c5173055bce21101621f25fb4d868495a7b7e2c3`. One high-priority blocker: completed dispatch resolution can suppress a new stored-context/authority mismatch during A2 startup. Passing suites do not waive this counterexample. No source correction was made.

## Identity, scope and basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-bb9261f6` read-only. Initial HEAD is exact and status empty. Source/tests match `d1c732d9ead927e7705e559a775df75ceaf7bbf9`; production matches `766dc11d2ee736ddf70d2b755bc0c577163bf95a`. Actual baseline is `e52df2fe28aaafbff381fe3d650d92b7aa6b08ae`. Its delta contains exactly eight authorized paths: recovery, persistence, resource_store, dispatch_store, dispatcher, reconciliation, recovery tests and RECOVERY-P1B2. Protected paths are unchanged. `git diff --check e52df2f HEAD` passes. No applicable ancestor/scoped AGENTS.md was found, including report destination ancestry.

Read the full P1B2-BOUNDARY-ASTRA architecture, closed records, authority/reconciliation amendment, resume rules and exits; full candidate RECOVERY-P1B2 contracts/tranches/corrections, including the interrupted full run, uncertified-window correction and failed full-suite fixture-lifetime correction; P1B22b-r1, P1B22a-r2, P1B21-r3, orchestration protocol, original R1 fault-authority decision and original P1 design/B2/exit matrix. Inspected the complete production/test delta and actual allocation, startup, publication, archive and reconciliation callers. No inherited independent B2.2c findings were supplied. The uncertified-window defect was implementer-discovered and corrected before this review. Memory supplied only general exact-checkout/evidence-tier guidance, not current acceptance evidence.

## F1 — high — new context mismatch is acknowledged away by an old resolution

Location: `service/reconciliation.py:337-343`, interacting with the reason overwrites at lines 304-328.

After a real allocated job is abandoned through the production Core continuation, change only `jobs.context.plan.request.text`. Keep its job JSON, reservation, original dispatch/Authority/receipt/outcome/ACK and all immutable recovery records unchanged. A2 detects `a.context != contexts[job_id]` and initially sets `dispatch_owner_mismatch` at lines 304-308. The expected absence of an old receipt then overwrites that reason with `terminal_receipt_missing` at lines 327-328. The new completed-resolution branch checks only job JSON (`job_row[6]`) and reservation, so it clears the reason despite the changed context.

Observed on two separately opened Core sessions: `complete=True`, `defect_latched=False`, `blocked=False`, `reasons=[]`; no dispatch diagnostic exists. The unchanged control also passes twice. Original dispatch fields remain `(acked, receipt, outcome) == (0, None, None)`. Thus this is not a digest-tampering test or a fixture-installed completion: the actual writer committed the recovery before a single later current-data change.

This violates the exact reconciliation amendment: a completed recovery may account for its old known missing-receipt/outcome mismatch, but a changed/new owner/context defect must remain unresolved. It also contradicts the handoff's claim that new owner/binding defects remain unresolved. Public activation is still closed, so this is a source/E1 integrity-gate defect, not evidence of a live deployment incident. It is nevertheless within this tranche's required A2 consumption behavior.

Reproducer, run from the candidate with the existing interpreter; disposable roots only:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import closing
import json, sqlite3
from tests.service.test_recovery import installation, core_fixture
from service import recovery
from service.core import Core, encode
from service.reconciliation import Reconciliation
with TemporaryDirectory() as td:
    i = installation.__wrapped__(Path(td))
    roots, scope, _, _ = i
    intent, observation, jid = core_fixture(i, 'running')
    recovery._pending_through_core(roots, scope, intent, observation, now=102)
    root = roots['control'] / 'core'
    with closing(sqlite3.connect(root / 'core.sqlite')) as db, db:
        value = json.loads(db.execute(
            'SELECT context FROM jobs WHERE id=?', (jid,)).fetchone()[0])
        value['plan']['request']['text'] = 'changed after recovery'
        db.execute('UPDATE jobs SET context=? WHERE id=?', (encode(value), jid))
    for _ in range(2):
        core = Core(root)
        try:
            scanner = Reconciliation(core, roots['staging'])
            result = scanner.reconcile_startup()
            print(result)
            assert result['blocked'], 'new context mismatch was suppressed'
        finally:
            core.close()
PY
```

The diagnostic probe exited 0 after asserting the observed incorrect clean result twice. A separate expected-correctness RED collected both startup results, then asserted both must block: **exit 1**, `AssertionError: new context mismatch suppressed on both startups`. Both results were exactly `complete=True, blocked=False, defect_latched=False, reasons=[]`. The shorter reproducer shown above fails on its first scan. No candidate test/probe file was written.

Minimal fix: consume only the specifically permitted old mismatches after independently proving all current authority/context joins still match. Compare the full required recovered job row/context to the accepted output, and avoid a later allowed reason erasing an earlier unrelated defect (accumulated reasons or separate strict join predicates suffice). Add a production-writer regression with an unchanged positive and this single-field context negative over two startups; retain new receipt/publication/owner negatives and pending-resolution behavior. Do not make immutable completed replay reject ordinary unrelated new activity or replay old mutations to fix this.

## Other assessed behavior and independent probes

Both normal locks span the continuation. Closed accepted intent/proof/all-owner coverage, original source/history/media and exact plan checks remain before authority transitions. The journal/evidence prefix precedes the fresh recovery-specific scan; no normal Core/ResourceStore construction, borrowed token or ordinary `observe` closes inherited authority. The Core transaction updates the prescribed unsuccessful JSON/latest reap, retained reservations, monotonic accounting and new monitor/A2 state together with exact append-only resolutions and the fourth step. Original dispatch/commands/receipt/outcome/ACK, events, terminal history, artifacts and idempotency remain retained. Core COMMIT precedes certificate creation; exception paths unwind rather than retry a live ambiguous transaction.

The uncertified path now verifies current owned bundle/descriptor/original journal evidence before first certificate creation. Its original-projection reconstruction accounts for declared after-images and current recovery references rather than omitting all history. Once certificate creation intent exists, missing/partial/wrong output quarantines; exact output is re-barriered. Completed replay validates immutable evidence and resolution sets without a new resource/inventory scan or epoch. Receipt fence remains pending; supported worker and both control startup modes remain closed; direct Core publication wrappers retain permanent retirement checks. Apply remains unconditionally unavailable and enrolled activation has no new toggle.

Independent stdin probes, using fixture builders only to construct disposable installations/accepted input and the real production continuation:

- Actual fork death at `core-complete-commit` and separately `core-complete-parent`. Parent verified exit 73, discarded intent/observation, then launched a fresh interpreter supplied only roots/scope/recovery ID/expected epoch. Each resumed twice after TTL and verified exactly one epoch, one recovery, one resolution, exactly four named steps and unchanged `(0,None,None)` ACK/receipt/outcome. Both passed.
- Stopped after durable Core COMMIT before any certificate, then independently changed a validly rehashed old B1 plan, terminal asset bytes, a predecessor receipt journal, or current bundle descriptor. Each refused twice before any certificate appeared. Errors included `b2_before_history_binding`, `b2_prefix_prestate_changed`, missing predecessor archive and `b2_prefix_quarantine`. Passed. This extends the candidate's four uncertified-bundle/archive regressions to old logical history/media and predecessor evidence.
- With old high-water zero, actual filesystem population exceeded the accepted expected high-water; two calls refused `b2_scan_output_changed`, retaining exactly three prefix steps and original monitor, with no resolution. A separate real allocated installation had old high-water 67,108,864 plus unchanged 2,097,152 disk charge over its finite quota; two calls refused `disk_quota` with the same pending/no-partial-health result. No mocked byte measurement or free-space result was used in these probes.

The focused suite additionally executes actual SQL authorizer denial, pre/post-COMMIT fault boundaries, certificate file/parent barriers/no-remint, denied fault write plus immediate death/inherited checking, A2 token archival, terminal history/permanent retirement, exact degraded/unavailable diagnostics, changed/new media/reference defects, missing/corrupt journals, pending filters, completed resolution deletion and startup refusal. Inspected fixtures: `core_fixture` raises old high-water to keep successful examples above evidence-file growth; actual-low-high-water refusal is separately tested. `b2_intent(..., prepare_only=True)` builds input but does not install the recovery result. New B2.2c completion cases call the production writer. No blanket validator relaxation or accept-either crash assertion was found. Existing tests do not cover F1's later context-only mutation.

Budget changes are deliberately rejected in this tranche, including accepted-record reading. This is narrower than the original B2 design's future explicit increase procedure, but matches this review's explicit unsupported-budget-change scope; no activation or effective-budget feature is demanded here. It remains a limitation, not a waiver of accounting checks.

## Fresh verification

Existing environment only; bytecode and pytest cache disabled, no installs. Commands run sequentially on unchanged candidate source/tests.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_admission.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**502 passed in 226.91s, exit 0.**

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**654 passed, 18 skipped in 266.91s, exit 0.** Sequential after the focused suite on unchanged candidate source/tests. All 18 skips remain unqualified. Independent positive/refusal probes exited 0; the explicit F1 expected-correctness RED exited 1 as recorded above. These passing suites reproduce the claimed counts but do not cover or waive F1.

Final candidate HEAD remains exactly `c5173055bce21101621f25fb4d868495a7b7e2c3`; candidate status remains empty and `git diff --check e52df2f HEAD` passes. Frozen-source-to-HEAD remains handoff-only; production comparison remains empty.

The implementer's earlier interrupted run is not a full receipt. The recorded 1-failed/653-passed/18-skipped run is a failure, superseded only by its later corrected-fixture campaign. Inspected `766dc11..d1c732d`: the only test change explicitly closes the newly introduced SQLite source handle before private inspection; production validation is unchanged. This does not explain or qualify the separate historical A1 inventory drift.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked host documents only. E1-mock does not authenticate actual host death. Skips do not qualify installed behavior. Physical power-loss/ENOSPC/daemon/VM/controller durability, actual-host authority, installed lifecycle, audio/engine/hardware and clean-host operation remain unqualified. **A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.**

Receipt completion, B2.3 trusted transport, B3, C1/C2 installed proof, parent integration, final Astra and public/unattended acceptance remain pending. They are not additional prerequisites invented for this bounded review. F1 requires a correction and fresh independent review of its exact final candidate before B2.2c acceptance.

Only this designated root report is written as a repository artifact. No source/test edit, commit, subagent, push/deploy, installs, Docker, external network/model/audio/paid/E2E-runner work, prune or user cleanup. Test/probe-owned temporary roots only; unrelated research/worktrees preserved.
