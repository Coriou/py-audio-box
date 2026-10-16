# P1B2.2d independent adversarial review — round 1

2026-10-16 · gpt-6.1-sol / low. **NO-GO for bounded B2.2d source/E1 acceptance** at exact candidate `fccb0391db4d45d52f56dfbdc33a63a2aa6cb6a0`. Two concrete blockers: unfinished finalization accepts an obsolete stopped-owner proof after a new worker registration; replay of an already completed recovery incorrectly requires a later pending recovery's not-yet-created certificate. No candidate correction was made.

## Identity and basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-11ad9433` read-only. Initial HEAD is exact and status empty. Frozen source/tests are `1ceec7726f799c80d586ac9ed5e5f835e8374b78`; frozen-to-HEAD is handoff-only. Actual baseline `50ee0b0eeed91cd1ef3836da4a27457a19180692` to candidate changes exactly `service/recovery.py`, `tests/service/test_recovery.py`, and `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. Protected paths, versions and schemas are unchanged. Baseline diff check passes. No applicable ancestor or scoped AGENTS.md was found.

Read the full boundary architecture/records/resume/authority/completion handshake/B2.2, candidate recovery handoff including original contracts, corrections and tranches, P1B22c-r1/r2, P1B22b-r1, orchestration protocol and R1/B1 fault-authority/WAL/no-remint context. Inspected the complete production/test delta and actual Journal, supervisor hello, Dispatcher startup, Core completion/publication, proof, archive and projection readers. No prior independent B2.2d findings were consulted. Memory supplied general exact-checkout/evidence-tier guidance only; no historical acceptance is used as current proof.

## F1 — high — unfinished finalization bypasses current retained-owner coverage

Location: `service/recovery.py:1256-1271`, entering `_finish_receipts` at `1630-1714` without the current-state consumer validation at `1327-1328`.

The fast branch is selected by **Core completion**, not by the fifth receipt-complete row. It correctly validates the immutable accepted proof against the registrations embedded in its old before-image, but skips validating the current registration population before performing remaining receipt writes/SQL completion. That separation is appropriate for a truly completed no-op; it is insufficient for unfinished maintenance.

Reproduced with the real production writer: stop at `completion-fence-dirsync`, with complete receipt fence and durable Core certificate but no fifth SQL row. Open a real Journal, obtain the supervisor's production hello, close its lane lock, and append a fresh new-lane/new-boot registration using production `register`. A fresh registered Dispatcher can then start inspect-only and close. Resume twice after TTL using the original accepted observation/proof: both calls return `status='complete'`; the first appends the fifth SQL row despite the newly retained registration absent from its stopped-owner proof. The added registration deliberately uses the existing synthetic `actual-host-inspection` document path, as the candidate's own startup test does; this is E1 provenance, not authenticated actual-host evidence.

A separate stronger coverage variant retained the original E1-mock provenance and registered a different worker container ID (`e` repeated 64 times) with `started='restarted-worker'`, as well as the new lane/boot. The accepted proof omits that owner entirely. Finalization still succeeded twice. Thus this is not merely a mismatch between two evidence labels; the current retained owner union exceeds the accepted stopped-owner coverage.

The failure is not a claim that the worker must read the fifth SQL row or that cross-volume completion is atomic. In this window the matching Core certificate really exists. The worker can hello and appropriately registered control may inspect; activation remains closed. The failure is that subsequent *unfinished maintenance* silently accepts stale freeze assumptions even after a supported durable registration exposes the restart. Both locks being free after shutdown does not prove that incarnation stayed stopped throughout maintenance. The brief requires stopped owners/both locks on resume and explicitly forbids silently re-proving a restarted consumer. B2.3 authentication is not demanded here: the changed retained input is already locally detectable.

Minimal fix: distinguish a fifth-step-completed immutable replay from an unfinished receipt continuation. Before any remaining receipt transition or fifth-row insertion, compare the current bounded registration/owner history to the accepted coverage and reject new/changed incarnations, preserving the original accepted proof. Do not substitute a fresh proof, mint a new identity, run ordinary resource scans, or impose current eligibility on truly completed no-ops. Add a two-resume regression at the complete-fence/pre-fifth window with a production worker hello and new retained registration; assert refusal and no fifth row. Keep the unchanged stopped-owner crash-resume positive.

## F2 — medium — completed replay depends on a later pending recovery's future certificate

Location: `service/recovery.py:1263-1267`, especially changing the prior-history loop from stopping at the requested recovery to traversing every other recovery.

Complete recovery A using `_complete_receipts`. Construct coherent recovery B with the existing prepare-only builder and execute production `_pending_through_evidence` for B. Replaying completed A twice after TTL raises `FileNotFoundError` for `p1-recovery-<B>.certificate.json`. That file correctly does not exist yet: B is still pending. A's immutable certificate, five-step chain and retained creation inputs remain intact. Repeating with B fully completed succeeds twice, confirming the defect is the pending successor requirement.

This conflates A's immutable completion with the current recovery's eligibility/progress. The brief requires completed replay to remain a no-op even after later recovery, with current eligibility separate. Requiring a future certificate for B makes the old completed result unavailable during an otherwise supported later maintenance state. Independent extensions also reproduced the same false refusal twice with B at journal and certified Core-complete stages. SQL dumps stayed unchanged in all three cases. No corruption repair or new mutation was observed; the bug is false refusal, not authority escalation.

Minimal fix: validate every retained history according to its actual declared durable stage, without deriving a completed receipt artifact requirement for a pending successor. Preserve validation of all genuine completed predecessors and their retained production creation/bundle/rotation evidence. Add A-completed/B-pending replay cases at evidence, journal and Core-complete stages, returning A's original result twice with scans forbidden and all current state unchanged. Do not ignore whole history tables or require B to finish first.

## Compact reproducer for both findings

Run from the exact candidate; only disposable roots are used. This diagnostic asserts the observed incorrect outcomes, so it exits zero when reproducing both defects. Replace the last assertions with the required refusal/completed-result expectations for RED regression tests.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
import json, sqlite3
from tests.service import test_recovery as t
from service import recovery as r
from worker.runtime_journal import Journal
for case in ('restarted-before-fifth', 'later-pending'):
    with TemporaryDirectory() as td:
        installation = t.installation.__wrapped__(Path(td))
        roots, scope, _, _ = installation
        intent, observation, _ = t.core_fixture(installation)
        if case == 'restarted-before-fifth':
            def fault(point):
                if point == 'completion-fence-dirsync': raise OSError('stop')
            try:
                r._complete_receipts(roots, scope, intent, observation, now=102, fault=fault)
            except OSError:
                pass
            journal = Journal(roots['receipts'])
            lane = journal.value['lane']; journal.lock.close()
            registration = json.loads(intent['before']['source']['data']['p1_registrations'][0][1])
            registration.update(lane=lane, boot='e'*32)
            registration['containers'][1].update(id='e'*64, started='restarted-worker')
            r.register(roots, scope, registration)
            for _ in range(2):
                assert r._complete_receipts(roots, scope, intent, observation, now=1000)['status'] == 'complete'
        else:
            expected = r._complete_receipts(roots, scope, intent, observation, now=102)
            successor = t.b2_intent(installation, prepare_only=True,
                                   recovery_id='7'*32, archive_id='6'*32)
            r._pending_through_evidence(roots, scope, successor,
                                       successor['proof']['observation'], now=102)
            for _ in range(2):
                try:
                    r._complete_receipts(roots, scope, intent, observation, now=1000)
                except FileNotFoundError as error:
                    assert '77777777777777777777777777777777.certificate.json' in str(error)
                else:
                    raise AssertionError('expected observed false refusal')
        print(case, 'reproduced twice')
PY
```

## Assessed ordering and preservation

The normal sequence is sound within its bounded E1 claim: actual Core COMMIT/fourth step/protected Core certificate precede the receipt certificate containing exactly four steps; the selected journal lane/version/records only gain the exact maintenance binding; the complete fence follows the bound journal; only then does the fifth SQL receipt append. The fifth step does not replace or rehash the original Core certificate. No cross-volume atomicity is claimed.

Creation descriptors precede their attempts. Derived names, exclusive no-follow writes, exact 0600 checks, owner/regular/single-link bounded reads, file and parent barriers, immutable missing/partial/wrong-output quarantine and exact-output re-barrier/re-record behavior remain present. No remint, overwrite, cleanup, new lane/epoch/events/resolutions or broad journal/history exclusion was introduced. Both locks span the private continuation. Truly completed replay retains production bundle/rotation/creation inputs and previous completed archives, and avoids ordinary resource/inventory scans. F2 is specifically the incorrect additional obligation on a pending successor.

The B2.2c strict context/current authority classification is unchanged; its production-writer two-startup regressions remain in the focused suite. Canonical frozen job output, grounded snapshots/history, original dispatch/receipt/outcome/ACK retention, B1 private WAL/no-remint and R1 inherited authority are preserved by unchanged source and suite coverage. Public/programmatic/CLI apply remains unconditionally unavailable. Enrolled submit/replace/cancel/dispatch remain closed; no activation toggle or autolaunch is added.

Independent actual-process probes called `os._exit(73)` before/after fence replacement and before/after fifth-step COMMIT. Each parent discarded accepted intent/observation and launched a fresh interpreter with only owned roots/scope/recovery ID. Two resumes after TTL quarantined the pre-replace attempt or completed exact durable outputs, retaining one accepted recovery, one lane/epoch, one resolution set and exactly four or five steps as appropriate. All four probes passed.

Independent complete-fence/pre-fifth startup probe used real Journal and supervisor `operation('hello', {})`: omitted/old registration refused; fresh explicit registration permitted inspect-only Dispatcher startup against the actual Core certificate. Submit/replace/cancel refused, persistent activation stayed false, runtime calls were hello-only and launches stayed zero. Closing these owners then exposed F1 on both stale-proof resumes. This is an E1 local seam test, not actual host authentication or a running installed supervisor/transport.

## Fresh verification

Existing interpreter only, no installs, bytecode/cache disabled. Commands run sequentially on unchanged candidate source/tests. Passing suites do not waive F1/F2.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

**583 passed in 318.13 seconds, exit 0.** Includes the 74 added receipt-completion cases, actual I/O/SQLite denials, both half-state negatives, retained evidence and later ordinary-activity no-op tests. Independent probe results above supplement rather than merely restate these tests.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**735 passed, 18 skipped in 358.82 seconds, exit 0.** Sequential after the focused run, on the same unchanged source/tests. All 18 skips remain unqualified. Both independently rerun counts match the candidate's claimed 583/735/18 populations; this does not waive the two missing behavioral cases.

The report's exact compact reproducer was executed successfully, including after strengthening the new-container coverage variant: both defects reproduced twice, diagnostic exit 0. Independent process-death and startup probes also exited 0. No probe setup failure or candidate test failure occurred in this review.

Final candidate HEAD remains `fccb0391db4d45d52f56dfbdc33a63a2aa6cb6a0`, status empty; baseline diff check remains clean. Frozen-source-to-HEAD remains handoff-only. No candidate edits occurred during or between verification runs. F1/F2 require correction and fresh independent review of the resulting exact candidate before bounded B2.2d acceptance.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked host documents only. E1-mock is not actual-host proof. Skips do not qualify installed behavior. No physical power-loss/ENOSPC/daemon/VM/controller durability, installed lifecycle, real engine/audio/hardware or clean-host qualification. **A1 global Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.** B2.3 trusted host transport, B3, C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending. These later obligations are not additional requirements invented for this tranche.

Only this designated root report is written as a repository artifact. No candidate/source/test edit, commit, subagent, push/deploy, install, Docker/network/model/audio/paid/E2E runner, prune or user cleanup. Disposable test/probe roots only; unrelated research/worktrees preserved.
