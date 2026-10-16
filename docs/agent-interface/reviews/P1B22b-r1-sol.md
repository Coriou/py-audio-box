# P1B2.2b independent adversarial review — round 1

2026-10-16 · gpt-6.1-sol / low. **GO for bounded B2.2b source/E1 acceptance only** at exact HEAD `5290e50f30b76be1275e84c51e6a64ad56c040b3`, frozen production/tests `8864b6fa50d6418a2a468cb040a50eb47e90c4e0`, baseline `a6451ccb8172e99213b9d6841613b0ab050f89d1`. No concrete blocking finding. This approves only private journal rotation and its step receipt after the reviewed evidence prefix, stopping PENDING before Core/job/resource/resolution/epoch completion or receipt completion. It is not full B2.2, integration, installed, actual-host or activation approval.

## Identity, scope and review basis

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-7d5b1e82` read-only. Initial and final HEAD match exactly; candidate status is empty before and after execution. Baseline-to-candidate changes are exactly `service/recovery.py`, `worker/runtime_journal.py`, `tests/service/test_recovery.py` and `docs/agent-interface/phase-1/RECOVERY-P1B2.md`. Frozen-source-to-HEAD is handoff-only. `git diff --check a6451cc HEAD` passes. No applicable ancestor/scoped AGENTS.md found, including report destination ancestry. Protected source, contracts, dependencies, packaging, ledger and prior reviews are unchanged.

Read the complete P1B2-BOUNDARY-ASTRA architecture, closed records, plan/resume rules and B2.2 requirements; complete candidate RECOVERY-P1B2 original contracts, corrections and B2.2a/B2.2b handoffs; root ledger split/integration entries; P1B22a-r1/r2, P1B21-r3, orchestration protocol and R1 authority/review context. Inspected the complete production/test delta, actual protected readers, projection, grounded archives, SQL step validators and startup/publication seams. No prior B2.2b findings were consulted. Memory registry search returned no relevant B2 entry; no memory-derived acceptance evidence is used.

## Ordering, immutable intent and resume assessment

Both existing nonblocking control/lane locks cover every invocation (`service/recovery.py:1199`, `offline:118-149`). Accepted record validation, exact plan/proof/deployment binding, all potential-owner coverage, coherent allocation, original logical source/history/media/affected-set checks and initial-expiry versus accepted-resume handling remain present (`1195-1270`). Every completed predecessor's control archives are grounded by the database validator; receipt journal/certificate bytes are compared against each predecessor's immutable accepted record on every invocation (`1223-1232`), even after the current fence supersedes the old fence.

The B2.2a marker/full accepted document/pending fence, protected exact original snapshot bundle, Core accepted intent/pending receipt, snapshot/manifest/original-journal archives and verified evidence receipt precede rotation (`1274-1387`). Snapshot evidence is never restored over Core. Earlier no-remint quarantine and durable restart input remain intact.

The fixed rotation descriptor binds the full accepted-intent digest, exact expected journal step, original journal status/bytes and new canonical envelope SHA-256 (`1250-1254`). Exclusive creation, file fsync and parent sync finish before replacement (`1390-1398`). Lane and epoch come from the accepted record; the continuation generates neither. Receipt namespace/budget checks precede creation; unknown temporaries refuse rather than being removed.

`worker/runtime_journal.py:111-140` uses the factored ordinary durable path: write fixed temporary, flush/fsync, replace fixed journal, fsync parent. Maintenance adds exclusive temporary creation and pending-fence/new-lane binding. Exact protected new bytes are then verified and file/parent barriers repeated before the journal SQL receipt BEGIN/write/COMMIT (`recovery.py:1399-1401`, `record:1354-1364`). Transaction errors unwind/close; no live retry assumes an ambiguous COMMIT rolled back.

Resume requires the exact descriptor and exact expected current new journal before substituting only the recorded old journal bytes/status into the original projection (`1255-1268`, `snapshot:200-224`). Current accepted/step rows and exact derived archive/reference labels are validated separately; other history, inventory, SQL/media and original journal semantics remain bound. This is not a blanket journal/history omission. Missing/old/partial/wrong output after a recorded attempt quarantines. Exact post-replace output is re-barriered and its missing SQL step recorded without overwrite, duplicate lane or new identity. Return/fence remain pending; no completion certificate is produced.

Missing/corrupt old journals retain exact status/bytes and require coherent control/all-owner coverage. UNKNOWN dispatches do not become clean receipts. No PID-1 receipt, timeliness, reaped-children count or ACK is fabricated. Jobs, attempts, events, dispatch receipt/outcome/ACK, reservations/demand/charges/high-water, monitor and faults remain unchanged. B2.1 structural allocation, canonical output-delta/grounded evidence, startup/publication fences and B1 WAL-source/no-remint behavior remain preserved by source inspection and fresh regressions.

Ordinary `Journal.save` keeps its prior truncating temporary behavior, canonical envelope, barriers and assignment to `self.value` only after success. Maintenance exclusivity is not imposed on ordinary saves. The added short-write check is conservative. Public/programmatic/host apply remains unconditionally unavailable; enrolled activation stays closed with no new flag. Internal E1-mock proof validation is not actual-host authentication.

## Independent adversarial probes

Ran stdin probes with existing fixture builders and disposable roots; no candidate probe/test file was written.

- After successful rotation, independently restored the exact original journal image, deleted the rotation descriptor, changed only its recorded before-status, introduced `journal.tmp`, or changed new-journal mode to 0644. Each of five variants refused twice after TTL, preserved journal bytes and retained exactly three SQL steps. Errors respectively included `b2_prefix_quarantine`, `b2_missing_rotation_intent`, `b2_prefix_receipt_namespace` and `b2_prefix_permissions`.
- Actual fork children called `os._exit(73)` separately at `journal-replace` (before directory barrier/step receipt) and `journal-receipt-commit`. Parent verified exit 73, discarded intent/observation/snapshot variables, and launched a fresh interpreter with only owned root paths, scope and recovery ID. It reloaded the owned accepted document and resumed twice at `now=1000`. Both normal Core/Journal constructors refused pending before each resume. Both cases passed: unchanged new-journal inode/bytes, original lane identity, one accepted recovery, exactly pending/evidence/journal steps and one manifest/archive identity. Qualifying resume cannot depend on inherited Python RAM.
- Separate ordinary-Journal probes denied actual file fsync, replace or directory fsync. Each failure preserved the old in-memory value; ordinary retry succeeded and a new Journal reloaded the expected value. All three passed.

The focused suite additionally exercises every named rotation-intent/write/fsync/replace/SQL interruption, actual I/O denials, changed new journal/hash/lane/symlink/hardlink/archive/manifest/media/history/proof/extra entry, initial expiry/both locks, retained predecessor receipt evidence, coherent missing/corrupt/UNKNOWN positives and incomplete/live coverage negatives. Coherent allocated cancel-requested work and terminal media/history preservation run through both prefix stops. B2.2a actual process-death/staging and B2.1 semantic/grounded evidence regressions remain green. No minimal fix is requested.

## Fresh verification

Existing interpreter only; bytecode and pytest cache disabled, no installs. Commands ran sequentially on unchanged candidate source/tests:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py -q
```

**325 passed in 107.18s, exit 0**, `/tmp/P1B22b-r1-focused.log`.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**595 passed, 18 skipped in 213.51s, exit 0**, `/tmp/P1B22b-r1-full.log`. Skips remain unqualified. Independent probes above also exited 0. Final candidate HEAD/status and diff check remain exact/clean.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked host projections only. Process death and successful barriers do not qualify physical power-loss/ENOSPC/daemon/VM/controller durability, authenticated actual-host authority, installed lifecycle, engine/audio/hardware or clean-host operation. **A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver.** Remaining atomic Core/resource/resolution/epoch and receipt completion, B2.3 trusted host transport, B3, C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending. These later obligations are not demanded as additions to this bounded tranche.

Only this designated root review report was written as a repository artifact. Candidate source/tests stayed read-only. No commits, subagents, push/PR/deploy, installs, Docker/network/models/audio/paid/E2E runner, prune or user cleanup. Disposable test/probe roots and local test logs only; unrelated research/worktrees preserved.
