# P1B2.2d independent adversarial review — round 4

2026-10-17 · gpt-6.1-sol / low. **NO-GO for bounded B2.2d source/E1 acceptance** at exact candidate `e3d58452de53de114feb71b4bbc7c7c5d3821305`. Original F1 remains closed; F2 is now closed within the assessed SQL and receipt-only interruption windows; the original F3 SQL/creation-prerequisite cases remain closed. One additional medium retained-history blocker is confirmed below: completed replay validates the Core-certificate creation descriptor only for the requested recovery, omitting other completed histories. No candidate correction was made.

## Identity, scope and basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-108eec6e` read-only. Initial HEAD matches exactly and status is empty. Frozen source/tests are `8d6ccd5308aa6eee5413f203ab9c3296260d3dd0`; frozen-to-HEAD changes only RECOVERY-P1B2. Actual baseline `657ff8f` to HEAD changes exactly `service/recovery.py`, `tests/service/test_recovery.py`, and `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. Protected-path comparison excluding those three paths is empty and baseline diff check passes. No applicable ancestor/scoped AGENTS.md was found, including report destination ancestry.

Read full P1B22d-r1/r2/r3 findings, reproducers, closures and limits; full P1B2-BOUNDARY-ASTRA architecture/closed records/resume/B2.2; full candidate RECOVERY-P1B2 original contracts, tranches, corrections and latest R3 response; orchestration protocol, P1B22c-r1/r2 and R1 fault-authority/B1 round-2 WAL/no-remint context. Inspected the actual production delta, new discovery and archive grounding, pending-stage and completion validators, fixed writers and regression tests. Memory registry search provided no relevant B2-specific guidance; no historical acceptance is used as current proof.

## F4 — medium — another completed history's Core-certificate creation descriptor is not validated

Locations: `service/recovery.py:1368-1371`, `1380-1390`, `1402-1408`, and `_verify_finished_receipt_evidence` at `1819-1829`. The exact Core descriptor check at line 1404 covers only the requested intent. The shared completed-history validators cover the bundle, rotation, evidence descriptors and three final descriptors, but omit `.core-certificate.intent.json`. Namespace membership merely permits that name and reads present bytes; it does not require presence, exact canonical descriptor contents or 0600 mode.

Complete A and B through the real production `_complete_receipts` writer. Delete only A's Core-certificate creation descriptor. Replay completed B twice after TTL with snapshot, Reconciliation collection, recovery scan and production file/SQL writers forbidden: both calls return B's original `status='complete'`, with all logical rows and retained file bytes unchanged. Separate fresh installations replacing only that descriptor with `b'wrong'`, or changing only its mode to 0644, also return B successfully twice. A's actual Core certificate, receipt certificate and five-step SQL chain remain intact in every variant.

The asymmetry is concrete: requesting A itself rejects the same missing descriptor twice, but requesting B accepts it. Conversely, deleting B's descriptor makes B reject twice while A incorrectly succeeds twice. Thus this is an omitted retained obligation, not a future artifact requirement or initial-acceptance proof problem. No accepted document/digest was rewritten to force the counterexample. The existing prior-completed test matrix at `tests/service/test_recovery.py:2240-2253` covers bundle, rotation and the three final descriptors, but omits this Core descriptor.

Consequence: completed replay can report success while hiding missing, wrong or non-private retained production creation evidence from another completed recovery. This violates the required validation of all prior archives/creation inputs and the finite sequence in which receipt-certificate creation requires the exact Core-certificate attempt/output. It demonstrates false acceptance of retained corruption, not live activation, authority mutation or protection against a malicious administrator.

Minimal fix: require the exact derived Core-certificate creation descriptor for **every fifth-completed history** in the common immutable evidence validator, alongside its actual Core certificate. Preserve the legitimate latest unfinished Core-COMMIT/no-attempt exception and stage-aware pending validation. Do not demand future outputs, scan current resources, remint/repair evidence or apply current startup eligibility to completed replay. Add two-replay negatives for missing/wrong/mode-changed Core descriptor in a completed predecessor and a completed successor, with scans/writers forbidden and unchanged logical/file state; retain the unchanged two-history positive.

Compact independent diagnostic reproducer (asserts the observed incorrect acceptance, so exit zero reproduces the blocker):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python - <<'PYPROBE'
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from tests.service import test_recovery as t
from service import recovery as r, resource_store
from service.reconciliation import Reconciliation
for damage in ('missing', 'wrong', 'mode'):
    with TemporaryDirectory() as td:
        i = t.installation.__wrapped__(Path(td))
        roots, scope, _, _ = i
        a, observation, _ = t.core_fixture(i)
        r._complete_receipts(roots, scope, a, observation, now=102)
        b = t.b2_intent(i, prepare_only=True,
                        recovery_id='7'*32, archive_id='6'*32)
        expected = r._complete_receipts(
            roots, scope, b, b['proof']['observation'], now=102)
        path = roots['receipts'] / (
            'p1-recovery-' + a['id'] + '.core-certificate.intent.json')
        if damage == 'missing': path.unlink()
        elif damage == 'wrong': path.write_bytes(b'wrong')
        else: path.chmod(0o644)
        before = t.logical(roots)
        files = t.retained_bytes(roots)
        def forbidden(*a, **k): raise AssertionError('scan/write')
        with patch.object(r, 'snapshot', forbidden), \
             patch.object(Reconciliation, '_collect', forbidden), \
             patch.object(resource_store, '_recovery_scan', forbidden), \
             patch.object(r, '_prefix_file', forbidden), \
             patch.object(r, '_prefix_transaction', forbidden):
            for _ in range(2):
                assert r._complete_receipts(
                    roots, scope, b, b['proof']['observation'], now=1000) == expected
                assert t.logical(roots) == before
                assert t.retained_bytes(roots) == files
        print(damage, 'incorrectly accepted twice')
PYPROBE
```

## Prior findings and independent correction checks

**F1 — remains closed.** Lines 1391-1400 retain the actual fifth-step distinction. Unfinished certified-Core receipt continuation compares the entire bounded canonical current registration table with the accepted before-image and checks original all-owner stopped coverage before remaining writes/fifth insertion. The uncertified path still reconstructs the source and checks consumers. Both normal locks cover each invocation. Truly fifth-completed replay stays separate from current eligibility and later ordinary activity. No substituted proof, new lane or epoch was introduced. Fresh suites include production Journal/hello/register stale-proof refusals and unchanged stopped-owner controls.

**F2 — closed within assessed windows.** New `_receipt_only_successor` at lines 1218-1321 runs before namespace validation. Filename matching selects at most one candidate; it is not authority. The canonical closed accepted document binds an existing SQL plan, exact latest fifth-completed predecessor, full SQL schema/before-image, retained histories, accepted all-owner proof/scope, installation/volume/key identities and exact journal/fence. Only its own plan row is excluded. Embedded histories and discovery consume finite population/byte/deadline limits. Multiple candidates refuse; no wildcard recovery filename allowance or history-table omission was found.

Before bundle creation, the prior complete fence or exact successor pending fence is accepted according to reached state. A bundle attempt requires pending fence, exact descriptor, actual bounded bytes/hash and the existing private-copy archive oracle against canonical source rows and actual plan. Only the accepted name and actually reached bundle names are added. Unknown/premature artifacts and attempted missing/partial/wrong outputs remain quarantine. A returns directly without finishing B, writing owned outputs, taking a resource/inventory scan or demanding future Core SQL/certificates.

Independently repeated the full original R3 production-boundary reproducer with correctness assertions at `accepted-parent`, `fence-replace-parent` and `bundle-parent`: every hook fired; A returned its original result twice after TTL, with scans/writers forbidden and identical logical/file state. Restoring hooks then resumed B with its original identity and no snapshot argument, followed by successful A replay. Related valid `core-complete-commit` and `journal-replace` cases also passed the same two-replay/preservation/resume controls.

An independent actual child called `os._exit(73)` at `bundle-parent`; its parent verified that exit, discarded A/B intent/observation/snapshot variables and launched a fresh interpreter receiving only roots/scope/A ID. That interpreter reloaded A from owned evidence, replayed it twice with scans/writers forbidden, then discovered B's accepted document, resumed B after TTL with unchanged recovery/lane/epoch and replayed A. Exit zero. Logical capture preceded physical-byte baseline because initial opening of crash-left SQLite can recover/checkpoint WAL; no byte-preservation claim spans that first open.

**F3 — original pending SQL/creation-prerequisite cases remain closed.** The finite prerequisite validator distinguishes a valid output missing its own receipt from a missing predecessor. Independently stopped B at actual `journal-replace`, deleted only B's evidence SQL row, and replayed A twice: both refused `b2_retained_creation_prerequisite` with scans/writers forbidden and unchanged state. The unchanged postreplace control succeeded twice. Core-COMMIT/no-certificate-attempt remains valid. F4 above is an additional hole in completed-history creation evidence; it does not invalidate these narrower original reproducer closures or waive the all-history obligation.

## Preservation assessment

The production writer retains actual Core COMMIT/fourth step/protected Core certificate → exact four-step receipt certificate → bound selected journal → complete fence → fifth SQL receipt. The fifth append does not rehash/replace Core's certificate or change jobs/resources/events/resolutions/ACK. Both locks, exact original owner-death proof, bounded closed records, derived filenames, owner/regular/single-link/no-follow reads, exact 0600 validation where invoked, file/parent barriers and creation-before-attempt/no-remint remain. F4 identifies where one required invocation is missing; no blanket claim of complete retained evidence validation is made.

Core/initializer/persistence/Dispatcher/supervisor/worker/host/reconciliation/resources/contracts/dependencies/packaging/ledger/reviews are unchanged against the actual baseline. B2.2c strict context/resolution joins and R1 inherited uncertainty/B1 private WAL/no-remint remain. All apply entrypoints remain unconditionally unavailable; enrolled activation remains closed without flags/autolaunch.

## Fresh verification

Existing interpreter only, no installs; bytecode and pytest cache disabled. Focused and full suites ran sequentially on unchanged candidate source/tests. Independent stdin probes used separate disposable roots and overlapped only local suite work.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py tests/service/test_admission.py tests/service/test_dispatcher.py -q
```

**685 passed in 522.35 seconds (8:42), exit 0.** Includes all 52 new R3 cases and inherited F1 registration, F2 SQL pending/journal/Core uncertified, F3 prerequisite, attempted-output quarantine, process-death, startup, strict-context and no-remint cases.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**837 passed, 18 skipped in 556.80 seconds (9:16), exit 0.** Sequential after the focused suite on the same unchanged source/tests. All 18 skips remain unqualified. Both populations match the candidate claims; passing counts do not cover or waive F4. The implementer's three-RED and standalone 108-target receipts are historical claims, not independently rerun pre-correction results or a separate 108-case run by this reviewer. No redundant additional broad campaign ran.

All independent probes described above exited zero with their stated correctness/refusal or observed-defect assertions. The exact F4 compact reproducer was extracted from this report in memory and executed as written: missing, wrong and mode variants each incorrectly accepted twice, diagnostic exit zero. No candidate test failure or source/test edit occurred during or between the sequential suites.

Final HEAD remains exactly `e3d58452de53de114feb71b4bbc7c7c5d3821305`, candidate status empty. `git diff --check 657ff8f HEAD` passes; frozen-to-HEAD remains handoff-only; protected-path comparison excluding the three allowed paths is empty. Only the authorized root report was written. F4 requires correction and fresh independent review of the resulting exact candidate before bounded B2.2d acceptance.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked host inputs only. E1-mock is not actual-host proof. Skips are not qualification. No physical power-loss/ENOSPC/daemon/VM/controller durability, installed lifecycle, engine/audio/hardware or clean-host qualification. **A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.** B2.3/B3/C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending. These later gates are not new requirements invented for this review.

Only this designated root report is written as a repository artifact. No candidate/source/test edits, commit, subagents, push/deploy, installs, Docker/network/models/audio/paid/E2E runner, prune or user cleanup. Disposable test/probe roots only; research and other worktrees are preserved.
