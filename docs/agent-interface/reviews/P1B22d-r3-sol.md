# P1B2.2d independent adversarial review — round 3

2026-10-16–17 · gpt-6.1-sol / low. **NO-GO for bounded B2.2d source/E1 acceptance** at exact HEAD `3eddd2662ebbbb08a728fce6e23d22e3784e18af`. F1 remains closed. The exact R2 F2 Core-committed/no-attempt counterexample is corrected, but the same completed-replay separation requirement still fails for a legitimate receipt-only successor. F3 is closed within the finite-order assessment below. No candidate correction was made.

## Identity and basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-42ef127a` read-only. Initial HEAD matches exactly and status is empty. Frozen source/tests are `8dbfdca1da5b79ce1714984594a06e86bce3ead4`; frozen-to-HEAD changes only RECOVERY-P1B2. Actual baseline `3692f17428b5ad58d0fdaddb0eba9e541a4347e2` to HEAD changes exactly recovery.py, recovery tests and RECOVERY-P1B2. Correction-only `899c25c..8dbfdca` has the same path set. Baseline diff check passes. No applicable ancestor/scoped AGENTS.md was found.

Read full P1B22d-r1/r2, their findings/reproducers/closures and evidence limits; full boundary architecture/records/resume/B2.2; complete RECOVERY-P1B2 original contracts, tranches, corrections and latest R2 response; orchestration protocol, P1B22c-r1/r2 and R1 fault-authority/B1 round-2 WAL/no-remint context. Inspected actual complete production delta, retained-stage/database/archive validators, prefix/final writers and test additions. Memory supplied general exact-checkout/evidence-tier guidance only; no historical acceptance is current proof.

## F1 — high — remains closed

`service/recovery.py:1282-1291` retains the actual fifth-step distinction. Unfinished certified-Core receipt continuation compares the entire bounded canonical registration table to the accepted before-image and checks the unchanged registrations against original all-owner death coverage before remaining file/SQL transitions. Truly fifth-completed replay does not impose this current-eligibility requirement. No proof substitution or new lane/epoch is introduced. Original source reconstruction/check_consumers remains on the uncertified path. Both ordinary locks span each invocation. Existing production hello/register same-container and new-container stale-proof regressions, unchanged stopped-owner control and later ordinary-activity replay cases remain in the freshly run suites.

## F2 — medium — original R2 window corrected; completed replay still unavailable during receipt-only successor

**Blocking location:** `service/recovery.py:1270`, calling `_prefix_receipt_namespace(receipts, data['p1_recoveries'])`; the allowlist is derived only from SQL-accepted histories at `1195-1211`. The subsequent successor loop likewise sees only those SQL histories.

Complete A using the production writer. Prepare B with the existing prepare-only input builder. Run production B until `bundle-parent`: its full accepted receipt document, pending fence, exact bundle descriptor and exact original bundle are durable, but B's Core accepted row has not yet been inserted. This is the explicitly supported receipt-first acceptance/restart window, not corruption or a future output missing after an attempt. Replay A twice after TTL. Both calls raise `ValueError('b2_prefix_receipt_namespace')` because B's legitimate fixed receipt names are absent from the SQL-derived allowlist. A's own immutable certificate, five-step chain and retained evidence remain intact.

Independently reproduced twice at each of `accepted-parent`, `fence-replace-parent`, and `bundle-parent`. Each hook was asserted to fire. All logical SQL and retained regular-file bytes stayed unchanged. A strengthened bundle-parent reproduction forbade `snapshot`, `Reconciliation._collect`, `_recovery_scan`, `_prefix_file` and `_prefix_transaction`: both false refusals still occurred, with identical logical/file state. After restoring those hooks, B resumed successfully after TTL with its original accepted identity and no supplied snapshot bytes; A then returned its original result. This control establishes that the refused successor state is actually resumable by the supported production continuation.

An actual fork child also exited 73 at bundle-parent. Parent discarded A/B intent/observation RAM and launched a fresh interpreter with only roots/scope/A recovery ID. It reloaded A's protected accepted document and reproduced both false refusals with scans/writers forbidden and unchanged logical/file bytes. Logical capture preceded the physical-byte baseline, so no byte-preservation claim is made across initial opening of crash-left SQLite.

Consequence: immutable completed A depends on B advancing into SQL or completing, despite the required separation from current recovery eligibility/progress. This is a bounded availability failure; no authority escalation, mutation or live incident was demonstrated. It remains within the original completed-no-op/later-recovery requirement. B2.3 host authentication is not demanded to diagnose these locally owned inputs.

Minimal fix: recognize and validate the bounded protected receipt-only successor against its existing plan, accepted document, previous completed binding and actual finite durable prefix before namespace validation. Permit only its derived names and actually reached outputs; return A's original immutable result without scans, SQL/file writers, finishing B or demanding B's future Core row/certificates. Keep malformed/unknown/unrecorded/attempted-missing/partial/wrong evidence quarantined. Do not simply wildcard recovery filenames, omit receipt history, or require B to finish. Add accepted-document, pending-fence and bundle-durable pre-Core cases, including actual death/fresh interpreter, two A replays, forbidden scans/writers and complete state preservation.

Compact diagnostic reproducer (asserts the observed incorrect refusal; disposable roots only):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PYPROBE'
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from tests.service import test_recovery as t
from service import recovery as r, resource_store
from service.reconciliation import Reconciliation
with TemporaryDirectory() as td:
    i = t.installation.__wrapped__(Path(td))
    roots, scope, _, _ = i
    a, observation, _ = t.core_fixture(i)
    expected = r._complete_receipts(roots, scope, a, observation, now=102)
    b = t.b2_intent(i, prepare_only=True,
                    recovery_id='7'*32, archive_id='6'*32)
    fired = []
    def stop(point):
        if point == 'bundle-parent':
            fired.append(point)
            raise OSError('review-stop')
    try:
        r._complete_receipts(roots, scope, b, b['proof']['observation'],
                             now=102, fault=stop)
    except OSError as error:
        assert str(error) == 'review-stop'
    assert fired == ['bundle-parent']
    before = t.logical(roots)
    files = t.retained_bytes(roots)
    def forbidden(*a, **k):
        raise AssertionError('completed replay scan or write')
    with patch.object(r, 'snapshot', forbidden), \
         patch.object(Reconciliation, '_collect', forbidden), \
         patch.object(resource_store, '_recovery_scan', forbidden), \
         patch.object(r, '_prefix_file', forbidden), \
         patch.object(r, '_prefix_transaction', forbidden):
        for _ in range(2):
            try:
                r._complete_receipts(roots, scope, a, observation, now=1000)
            except ValueError as error:
                assert str(error) == 'b2_prefix_receipt_namespace'
            else:
                raise AssertionError('expected observed false refusal')
            assert t.logical(roots) == before
            assert t.retained_bytes(roots) == files
    assert r._complete_receipts(roots, scope, b, b['proof']['observation'],
                                now=1000)['status'] == 'complete'
    assert r._complete_receipts(roots, scope, a, observation, now=1000) == expected
PYPROBE
```

The exact R2 Core-COMMIT counterexample now returns A's original result twice with scans/writers forbidden and B unchanged. `_validate_archives:837-845` reserves a future certificate only for the latest unfinished history with neither creation attempt nor output; `_retained_pending_receipts:1598-1606` independently permits this no-attempt prefix. Default database/startup readers retain their certificate requirement. Once an attempt exists, exact descriptor/output remain mandatory. The new 27-case suite covers Core COMMIT/parent actual death and attempted missing/partial/wrong outputs. Those meaningful corrections do not cover the earlier receipt-only state above; the claimed valid-prefix matrix begins at Core intent, after the missed boundary.

## F3 — medium — closed within assessed finite order

`service/recovery.py:1577-1595` adds cross-stage prerequisites before exact artifact validation. Snapshot attempt requires pending SQL receipt; manifest and original-journal attempts require preceding attempts; each present attempt's exact output is then required by `_completed_prefix_receipts`. Evidence SQL requires all three exact creation records/outputs. Rotation attempt requires evidence SQL; journal SQL requires rotation evidence. Existing contiguous SQL validation requires all prior named SQL receipts for Core/fifth completion. Core-certificate attempt requires fourth receipt; final receipt-certificate attempt requires that Core attempt/output; completion-journal/fence attempts require their preceding final attempts/outputs. This distinguishes a missing OWN receipt from a missing PREDECESSOR receipt without demanding future outputs.

Independently reran exact journal-replace/delete-only-evidence-row reproduction and snapshot-parent/delete-only-pending-row extension. Both refuse twice with `b2_retained_creation_prerequisite`, scans/writers forbidden and all retained bytes/logical rows unchanged. Valid Core-COMMIT/no-attempt control returns A twice. Existing new tests exercise valid exact outputs before their own SQL receipts, preceding-descriptor deletion and attempted missing/partial/wrong Core output. No additional F3 blocker was found in the SQL/attempt/output finite order. The receipt-only omission above is recorded under F2, not waived by F3 closure.

## Preservation assessment

Production sequence remains actual Core COMMIT/fourth step/protected Core certificate → exact four-step receipt certificate → bound selected journal → durable complete fence → fifth SQL receipt. The fifth step does not replace/rehash Core's certificate or alter jobs/resources/events/resolutions/ACK. No cross-volume atomicity claim. Both locks, original all-owner accepted proof, bounded closed records, fixed derived names, exact 0600/no-follow/owner/regular/single-link reads, file/parent barriers, pre-attempt creation records and no-remint quarantine remain. Completed predecessors retain actual grounded control archives, production bundles/rotation/original journals and final descriptors; no whole history/table exclusion was added.

Protected source is unchanged against baseline: Core/initializer, persistence, Dispatcher/supervisor/worker, host, reconciliation/resources/admission/dispatch stores, contracts/dependencies/packaging/ledger/reviews. B2.2c strict context/resolution joins, B1 WAL/no-remint and R1 inherited uncertainty remain. All apply stays unconditionally unavailable and enrolled activation remains closed without flags/autolaunch.

## Fresh verification

Existing interpreter only, no installs, bytecode/cache disabled. Focused and full suites run sequentially on unchanged candidate source/tests. Independent stdin probes overlap only disposable local-root suite work.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

**633 passed in 424.16 seconds, exit 0.** Includes all 27 new R2 cases and inherited process-death, ordering, F1 registration, attempted-output quarantine, startup, context and no-remint cases. Full preservation ran sequentially afterward on the same unchanged source/tests. Implementer RED/GREEN receipts are historical claims, not independently rerun pre-correction results. Passing counts do not waive the F2 counterexample.

All independent diagnostic/control probes described above exited zero. The exact compact reproducer was extracted from this report in memory and executed: exit 0, including the after-TTL B-resume/A-restoration control. This diagnostic asserts observed false refusal; it is not a correctness pass.

Full preservation command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**785 passed, 18 skipped in 459.26 seconds, exit 0.** All 18 skips remain unqualified. Both independently rerun populations match the candidate claims. No source/tests changed during or between these sequential suites.

Final candidate HEAD remains exactly `3eddd2662ebbbb08a728fce6e23d22e3784e18af`, status empty. `git diff --check 3692f17 HEAD` passes; frozen-to-HEAD remains handoff-only; protected-path baseline comparison is empty. Only the authorized root review report was written. The F2 receipt-only counterexample requires correction and fresh independent review before bounded B2.2d acceptance.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked host documents only. E1-mock is not actual-host proof. Skips are not qualification. Physical power-loss/ENOSPC/daemon/VM/controller durability, installed lifecycle, engine/audio/hardware and clean-host operation remain unqualified. **A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.** B2.3/B3/C1/C2, parent integration, final Astra and public/unattended acceptance remain pending. Later B2.3 authentication is not an extra prerequisite invented for this review.

Only this designated root report is written as a repository artifact. No candidate/source/test edit, commit, subagent, push/deploy, install, Docker/network/model/audio/paid/E2E runner, prune or user cleanup. Only disposable test/probe roots were used; research and other worktrees are preserved.
