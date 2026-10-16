# P1B2.2d independent adversarial review — round 2

2026-10-16 · gpt-6.1-sol / low. **NO-GO for bounded B2.2d source/E1 acceptance** at exact HEAD `6d84699a9fa79ba1ef601e75bf1c06e0695bc7e9`. R1 F1 is closed. R1 F2 is improved but remains open at the legitimate committed-but-uncertified successor window. A new medium blocker accepts an inconsistent retained successor stage chain. No candidate correction was made.

## Identity and basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-ec6f294d` read-only. Initial HEAD is exact and status empty. Frozen source/tests are `b3e4fec940b5cfb8e1afe70bee7c76fccada9f96`; frozen-to-HEAD changes only RECOVERY-P1B2. Actual baseline `2bced6ec2cec67e9953f07cba1bea989af06f6b3` to candidate changes exactly recovery.py, recovery tests and RECOVERY-P1B2. Correction-only `c6ffb06..b3e4fec` changes the same paths. Protected source/worker/contracts/dependencies/packaging/ledger/reviews are unchanged. Baseline diff check passes. No applicable ancestor/scoped AGENTS.md was found.

Read the full P1B22d-r1-sol findings, reproducer, minimal fixes, ordering and evidence limits; full P1B2-BOUNDARY-ASTRA architecture/closed records/resume/B2.2; candidate RECOVERY-P1B2 contracts, tranches, corrections and latest response; orchestration protocol, P1B22c-r1/r2, P1B22b-r1, R1 fault-authority and B1 round-2 WAL/no-remint context. Inspected the production/test delta, current prefix/retained-stage validators, archive grounding, fixed writers, Journal/supervisor startup and apply/activation closure. Memory supplied only general exact-checkout/evidence-tier guidance; no earlier acceptance is used as current proof.

## R1 F1 — high — closed

`service/recovery.py:1274-1283` distinguishes actual fifth-step completion from unfinished receipt continuation. For every certified-Core unfinished `_complete_receipts` invocation, the entire bounded canonical current registration population must equal the accepted before-image before remaining receipt writes or fifth insertion. Unchanged registrations are individually validated and checked against the original all-consumer death observation. This catches arbitrary new retained rows and changed boot/lane/container/start identities, rather than special-casing the two regression examples. The earlier uncertified path still reconstructs and compares the complete source projection and checks consumers before writes. Both locks span either path; no new proof is substituted.

The suite exercises real Journal/production supervisor hello/production register with same-container and different-container new boot/lane histories, two stale-proof refusals, forbidden file/SQL writer hooks, unchanged SQL/file bytes and zero fifth rows. An unchanged stopped-owner control completes. Independently injected an additional valid canonical retained registration with a new boot at `core-certificate-parent`, `receipt-certificate-parent`, `completion-journal-replace` and `completion-fence-replace`. All four variants refused twice with `b2_prefix_registrations_changed`, preserving every retained file byte. These earlier-window injections use direct fixture SQL because ordinary register cannot pass a pending worker fence; they are retained-input validation probes, not claims of a supported live registration in those windows.

Truly fifth-completed replay skips this current-registration comparison. The later ordinary submission/allocation and fresh-registration inspect-only startup cases remain present. Current startup eligibility is separate. No additional fix requested for F1.

## R1 F2 — medium — partially corrected, still open

Locations: `service/recovery.py:1243`, `_validate_archives:833-840`, and the new successor check at `1570-1573`.

Stable pending successors at evidence, journal and certified Core stages now allow A's completed replay without requiring B's future receipt certificate. The new regressions cover all three twice, with scans/writers forbidden and state bytes unchanged, plus exact post-journal-replace/pre-step and complete-fence/pre-fifth positives. Bundle/original-journal/stage-descriptor/current-journal/fence damage refuses. These are meaningful improvements over R1.

However, complete A; prepare B; run production `_pending_through_core(B)` with a fault at `core-complete-commit`. B has its durable fourth SQL step, but **neither a Core certificate creation intent nor a Core certificate exists yet**. This is an explicitly supported resume window, not a missing output after a creation attempt. Replay A twice: both raise FileNotFoundError for B's control `p1-recovery-66666666666666666666666666666666.certificate.json`. All bytes remain unchanged in the exception-fault reproduction. Resuming only B through its Core certificate, without completing its receipts, makes both A replays succeed.

The first database validation exempts only requested A from the certificate requirement, so B is rejected before the stage-aware helper. That helper would also require B's descriptor/output whenever `core-complete` is present, even when no creation attempt exists. Thus fixing only the first call is insufficient.

An actual fork child also exited 73 immediately after B's Core COMMIT; A falsely refused twice with the same missing B certificate and unchanged logical state. In an initial fork probe, an immediate physical-byte equality assertion failed after opening the crash-left SQLite state; that probe is not a byte-preservation receipt. The diagnostic rerun captured the logical state before its byte baseline, then reproduced both refusals with no subsequent changed file names/bytes and identical logical state. The exception-fault version independently establishes the refusal without that crash-state capture issue.

Consequence: immutable completed A remains unavailable during a healthy, resumable later maintenance state. This is the same F2 separation failure at a missing boundary, not a demand to finish B or add B2.3 authentication.

Minimal fix: validate the latest unfinished successor's actual durable prefix, including the legitimate Core-committed/no-creation-attempt window, in both the initial database/archive validation and retained-input validator. Preserve exact before/after grounding and all completed predecessor obligations. Do not exempt whole histories, silently create B's certificate while replaying A, or relax missing/partial/wrong output quarantine once B's creation attempt exists. Add two no-scan/no-writer A replays at Core COMMIT and Core-parent boundaries, with unchanged B state; retain the attempted-missing-output negatives.

## F3 — medium — new retained-stage validator accepts a broken predecessor order

Locations: `service/recovery.py:1542-1543`, `1553-1559`, `1595-1597`. Production ordering is explicit at `1462-1477`: the evidence receipt commits before rotation intent and journal replacement.

Complete A. Run B's production writer until `journal-replace`, after the exact new journal replaces the old one but before B's journal SQL receipt. Verify B has pending/evidence rows. Delete **only B's evidence-step SQL row**. Retain all accepted documents, bundle, snapshot/manifest/original journal, creation descriptors, exact rotation intent and exact replacement journal. Replay A twice: both return A's original complete result, preserving the corrupted state bytes. A separate repeat forbade `snapshot`, `Reconciliation._collect`, `_recovery_scan`, `_prefix_file` and `_prefix_transaction`; both incorrect successful returns and byte preservation still reproduced, exit 0.

The new validator treats B's remaining `pending` SQL prefix and independently present exact files as sufficient. It never requires the committed evidence step that must precede a retained rotation attempt. The general SQL chain validator cannot catch this because B has no journal SQL row yet. This differs from a valid output with its own missing step receipt: a later durable creation attempt proves its predecessor receipt had to exist. Deleting that predecessor is retained-history corruption, not an interrupted unrecorded journal output.

Consequence: A's newly added successor validation reports success while hiding an inconsistent retained production chain. No authority mutation or live incident was demonstrated; the defect is false acceptance of retained corruption within the required immutable-history validation.

Minimal fix: enforce cross-stage prerequisites between SQL steps and persisted creation attempts, as well as exact bytes. In particular rotation intent/output requires the durable evidence receipt; preserve the valid journal-replace/pre-journal-step case. Apply the same finite-order reasoning to evidence creation sequence and Core/final receipt attempts. Add this one-row deletion negative over two A replays with scans/writers forbidden, retaining the unchanged postreplace positive. Do not fix by requiring B's future journal receipt or completion certificate.

## Compact independent reproducer for F2 and F3

Run from the exact candidate with the existing interpreter. This diagnostic asserts the observed defects and exits zero when both reproduce. Disposable roots only; builders construct inputs and all transitions use production writers.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -u - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import closing
import sqlite3
from tests.service import test_recovery as t
from service import recovery as r
for case in ('uncertified-successor', 'missing-predecessor-step'):
    with TemporaryDirectory() as td:
        installation = t.installation.__wrapped__(Path(td))
        roots, scope, _, _ = installation
        a, observation, _ = t.core_fixture(installation)
        expected = r._complete_receipts(roots, scope, a, observation, now=102)
        b = t.b2_intent(installation, prepare_only=True,
                        recovery_id='7'*32, archive_id='6'*32)
        point = ('core-complete-commit' if case == 'uncertified-successor'
                 else 'journal-replace')
        def stop(where):
            if where == point:
                raise OSError('review-stop')
        try:
            r._complete_receipts(roots, scope, b, b['proof']['observation'],
                                 now=102, fault=stop)
        except OSError as error:
            assert str(error) == 'review-stop'
        else:
            raise AssertionError('fault boundary not reached')
        if case == 'uncertified-successor':
            assert not (roots['receipts'] / ('p1-recovery-'+b['id']+
                        '.core-certificate.intent.json')).exists()
        else:
            with closing(sqlite3.connect(roots['control']/'core'/'core.sqlite')) as db, db:
                assert db.execute('DELETE FROM p1_recovery_steps '
                                  'WHERE recovery_id=? AND step=?',
                                  (b['id'], 'evidence')).rowcount == 1
        before = t.retained_bytes(roots)
        for _ in range(2):
            if case == 'uncertified-successor':
                try:
                    r._complete_receipts(roots, scope, a, observation, now=1000)
                except FileNotFoundError as error:
                    assert '66666666666666666666666666666666.certificate.json' in str(error)
                else:
                    raise AssertionError('expected observed false refusal')
            else:
                assert r._complete_receipts(roots, scope, a, observation, now=1000) == expected
            assert t.retained_bytes(roots) == before
        print(case, 'reproduced twice')
PY
```

## Preservation assessment and limits

The final writer still follows actual Core COMMIT/fourth step/Core certificate → exact four-step receipt certificate → bound journal → durable complete fence → fifth SQL receipt. Fifth insertion changes no jobs/resources/events/resolutions/ACK or lane/epoch. No cross-volume atomicity is claimed. Both normal locks, closed bounded records, all-owner accepted proof, fixed derived names, exact 0600/no-follow/single-link owned-file reads, file/parent barriers and pre-attempt creation intents remain. Missing/partial/wrong attempted outputs quarantine; exact durable outputs re-barrier and record missing receipts. Prior completed bundles/rotation/final descriptors and grounded archives remain checked. No remint or cleanup path is introduced.

B2.2c strict full job/context/reservation resolution checks and R1/B1 WAL/no-remint source remain unchanged. All apply remains unconditionally unavailable and enrolled activation closed without flags. B2.3 host authentication is later work, not a prerequisite invented for this review.

E1 local source/SQLite/filesystem/process and mocked host documents only. E1-mock is not actual-host proof. Skips do not qualify installed behavior. Physical power-loss/ENOSPC/daemon/VM/controller durability, installed lifecycle, engine/audio/hardware and clean-host operation remain unqualified. **A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.** B2.3, B3, C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending.

Only this designated root report is written as a repository artifact. No candidate/source/test edits, commits, subagents, push/deploy, installs, Docker/network/model/audio/paid/E2E runner, prune or user cleanup. Disposable test/probe roots only; unrelated research/worktrees preserved.

## Fresh verification

Existing interpreter only; no installs, bytecode/cache disabled. Focused and full suites run sequentially on unchanged candidate source/tests.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

**606 passed in 361.48 seconds, exit 0.** Includes all 23 correction cases and inherited real-process-death, denied I/O/SQL, ordering, retained evidence, startup, strict context and no-remint cases. Passing counts do not waive F2/F3.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**758 passed, 18 skipped in 404.87 seconds, exit 0.** Sequential after the 606-pass focused run on the same unchanged source/tests. All 18 skips remain unqualified. Both fresh populations match the claimed final counts; they do not cover or waive the two independent counterexamples.

The exact compact report reproducer above was extracted in memory and executed successfully: both cases reproduced twice, exit 0. The independent four-window F1 probe and the F2 certificate-only restoration control also exited 0. Exploratory probes named `pending-commit`/`evidence-commit` did not target the actual `*-receipt-commit` hooks and are excluded as interruption evidence; the final report reproducer explicitly asserts that its chosen hook fires. No candidate test failure or candidate edit occurred.

Final HEAD remains exactly `6d84699a9fa79ba1ef601e75bf1c06e0695bc7e9`; candidate status is empty, baseline diff check passes, and frozen-to-HEAD remains handoff-only with no service/worker/test delta. No source/test changes occurred during or between either suite. Only `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B22d-r2-sol.md` was written as a repository artifact. F2/F3 need correction and fresh independent review of the resulting exact candidate before bounded B2.2d acceptance.
