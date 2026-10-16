# P1B2.2a independent adversarial review — round 1

2026-10-16 · gpt-6.1-sol / low. **NO-GO for bounded B2.2a source/E1 acceptance** at exact candidate `903b44367bbbd07724b17d24627d693dfc1c308f`, frozen source/tests `b999c129c8cea52d637d270c977995cf11dca32b`, baseline `c8d3fa0f53f6fcead11467ab96ca81c59fb09276`. Two blocking crash-resume/evidence findings below. Pending authority remains closed; neither finding demonstrates an activation bypass. Later journal/resource/job/resolution transitions and trusted host transport are not requested here.

## Identity, scope and review basis

Inspected `/Users/ben/.txcode/worktrees/py-audio-box/t3-fa982d42` read-only. Initial HEAD matched and status was empty. Baseline-to-candidate changes are exactly `service/recovery.py`, `tests/service/test_recovery.py` and `docs/agent-interface/phase-1/RECOVERY-P1B2.md`; frozen-source-to-HEAD is handoff-only. Protected paths, ledger and prior reviews are unchanged. Baseline diff check passes. No applicable ancestor/scoped AGENTS.md found.

Read the full canonical P1B2-BOUNDARY-ASTRA, including architecture, accepted records, plan/resume rules and B2.2 exits; latest ledger split; complete candidate RECOVERY-P1B2 original contracts, corrections and B2.2a handoff; all three P1B21 reviews; orchestration protocol and relevant original P1/R1/B1 decisions. Inspected actual diff, source projection, storage ordering, archive validators, startup readers and tests. Memory supplied only general exact-candidate/evidence-tier guidance, not B2 authority or fresh evidence.

The ledger narrows B2.2 to an accepted-intent/evidence prefix; it does not waive that prefix's original durable restart rules. Astra section 3 requires persisted expected inputs, exact same-plan resume after acceptance and refusal on changed archive/history. B2.2 requires interruption/resume at SQL and evidence boundaries. Deliberate quarantine after a recorded artifact creation intent with missing/partial/wrong output remains required; the first finding concerns an earlier window with **no snapshot creation intent or artifact**.

## F1 — high: committed acceptance loses the only original snapshot bytes on process death

Locations: `service/recovery.py:1236-1240, 1244-1273, 1290-1305, 1315-1317`; crash tests at `tests/service/test_recovery.py:1090-1105`.

Initial capture makes the exact snapshot in a TemporaryDirectory and retains its bytes only in `snapshot_raw`. That directory is removed before receipt acceptance/Core commit. The receipt accepted document and Core accepted row retain the original logical projection and physical snapshot SHA-256, but not the exact snapshot bytes. Core acceptance and the pending step commit before the managed snapshot is created. After a crash in that interval, `existing is not None` prevents recapture, and absent caller `snapshot_bytes` reaches `b2_prefix_missing_snapshot_quarantine`.

This is a concrete functional defect against the required durable prefix resume, despite the handoff's candid limitation. The evidence-prefix operation cannot finish from its owned persisted state after its own ordinary successful Core commit. No archive creation has yet been attempted, so required no-remint quarantine is not the cause. Later transport cannot recover volatile bytes unless this lot establishes a durable source bundle/interface. It is correct not to substitute a new backup of the now-mutated pending DB: its physical bytes do not satisfy the accepted hash. The stored logical before-image also is not an exact physical snapshot reconstruction recipe.

Independent stdin probes used the existing `installation.__wrapped__` and `prefix_fixture` on separate disposable roots. First invocation omitted `snapshot_bytes`, matching the normal success test. Faults at these existing boundaries produced:

- `fence-parent`: zero Core intents/steps; reload receipt intent from disk, resume at `now=1000` succeeds with the original identity.
- `core-intent-commit`: one Core intent, no steps; same durable-document resume refuses `b2_prefix_missing_snapshot_quarantine`.
- `pending-receipt-commit`: one Core intent and pending step; same refusal.
- `snapshot-parent`: one Core intent/pending step and exact archive; durable-document resume succeeds without supplied bytes.

For the two middle exception probes, supplying the original externally retained RAM bytes makes resume succeed, isolating the missing durable input. The candidate's crash-resume matrix always passes `snapshot_bytes=evidence` on both invocations and therefore does not exercise caller/process state loss in these windows.

A separate real child-process probe called `os._exit(73)` exactly at `core-intent-commit`, without unwinding Python contexts. Parent observed exit 73, reloaded the full receipt intent from disk and resumed twice after TTL. Both returned `b2_prefix_missing_snapshot_quarantine`; no control archive existed. Direct historical Core refused `b2_b2_recovery_pending` and Journal refused `maintenance_pending` on each restart. Locks were released by process death. This demonstrates safe quarantine but failed required restart completion, not unsafe remint or host/power-loss qualification.

Minimal fix: durably retain the exact original bounded snapshot bytes in a protected, fixed, accepted-identity/digest-bound staging bundle before committing Core acceptance, then consume/verify those bytes for the final evidence archive after restart. Preserve receipt-first pending ordering, no restoration over Core, all bounds/barriers and no-remint handling for recorded creation attempts. Alternatively provide an equally concrete durable input protocol before crossing that commit; requiring the caller to retain RAM is insufficient. Add actual process-death tests at Core-intent and pending-step commit that discard caller snapshot state and resume using only owned durable inputs, preserving one recovery/lane/epoch/archive identity after TTL.

## F2 — medium: current pending resume ignores changed earlier receipt archives

Locations: `service/recovery.py:1156-1173` (`_prefix_receipt_namespace`), `1211-1220` (previous fence checked only while replacing it), `1227-1229` (source projection), `1324-1328` (final validation/return).

The receipt namespace helper permits prior recovery `.journal` and `.certificate.json` names and reads them for file safety/byte limits, but does not bind their contents. The logical/source inventory and `_validate_archives` cover the control-side archives, not these receipt-side archives. Initial replacement of a prior complete fence calls `read_maintenance` and validates the previous receipt archive. Once the new pending fence is installed, exact same-intent resume skips that previous-completion check and returns success even if the prior receipt journal or certificate changed.

Independent coherent two-history probes:

1. Build first completed history with existing `b2_intent(..., complete=True)` and `b2_receipt(..., complete=True)`.
2. Build a second prepare-only intent with recovery ID `7` repeated 32 times and archive ID `6` repeated 32 times. Run the production prefix successfully without external snapshot bytes.
3. In separate fresh roots, replace only the first byte of the **first recovery's** receipt `.journal`, or receipt `.certificate.json`, with `X`, preserving file length, mode, path and the second recovery's inputs. SQL and all control archives remain unchanged.
4. Resume second intent at `now=1000`, without snapshot bytes.

Both damaged variants return the identical `pending-through-evidence` result (`True` equality). No digest rehash or broad fixture corruption is involved. The same prior receipt files were valid and required for the previous completed worker startup. This violates the explicit changed-archive refusal rule; it is not a claim that the damaged old file opens startup or changes jobs. Existing prior-history coverage damages a control-side archive, so does not detect this receipt-side gap.

Minimal fix: on every prefix invocation, validate retained receipt journal/certificate contents for each completed predecessor against its immutable accepted record/step chain (using the fixed derived names and bounded protected readers), in addition to control-side archives. Include receipt evidence in exact resume binding or explicitly recompute its expected binding. Add separate changed/missing prior receipt journal and certificate negatives after the new pending fence is durable, plus unchanged multi-history positives. Do not drop history or rewrite damaged evidence.

## Preserved boundaries and storage assessment

Both normal locks are held across the helper. Exact plan/proof/source/schema/media/affected-set checks precede acceptance; initial expiry and accepted expiry are distinguished. Receipt marker/full accepted intent/fence precede Core intent/pending step. File creation is exclusive/no-follow/0600 with file and parent barriers. Exact durable existing outputs are verified and re-barriered rather than overwritten; recorded creation intent with missing/partial/wrong artifact quarantines. Transaction failures unwind/close the offline connection without live retry of an ambiguous COMMIT; commit-boundary probes reopen for inspection.

Current-row/reference/archive exclusions are explicit, not blanket history-table exclusions; older Core plans/history and control evidence remain bound. Structural allocation provenance and grounded archives from B2.1 F1/F2, canonical nonterminal before/output deltas, B1 no-remint/self-only projection/private WAL-aware source preservation, R1 uncertainty fences and publication retirement remain present. Missing/corrupt journal mode preserves status/bytes and requires coherent control/all-registration coverage; it does not infer clean death receipts. No journal rotation, jobs/attempts/events/reservations/monitor/fault/high-water/ACK/resolution transitions occur here. Snapshot remains evidence only.

Programmatic/CLI/host apply remains unconditionally unavailable; enrolled activation stays closed. E1-mock inputs are internal validator evidence, not authenticated actual-host proof. Source review found no added free death Boolean, arbitrary SQL/path writer or activation toggle.

## Fresh verification

Existing interpreter only; no installs, bytecode or pytest cache. Focused/full runs are sequential on unchanged source/tests. Inline probes above use disposable roots and stdin, with no candidate test-file writes.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_packaged_lifecycle.py -q
```

**252 passed in 73.79s, exit 0.** Includes 65 prefix cases and predecessor regressions. Passing tests do not cover the two independently reproduced gaps as described above.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**522 passed, 18 skipped in 177.99s, exit 0.** Sequential after the focused suite, on unchanged source/tests. Skips remain unqualified. The independent exception/process-death/two-history probes also exited 0 with the observations above. Final candidate status is empty; HEAD remains exactly `903b44367bbbd07724b17d24627d693dfc1c308f`; `git diff --check c8d3fa0 HEAD` passes. Only this root review report was written; candidate source/tests remained frozen throughout.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked host projections only. Skips are not qualifications. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver. Remaining B2.2/B2.3, B3, C1/C2, parent integration, final Astra and public/unattended acceptance remain pending. No physical power-loss/ENOSPC/daemon/VM/controller, actual-host, engine/audio/hardware or clean-host claim.

Only this designated root report is written. No candidate/source edit, commit, subagents, push/PR/deploy, installs, Docker/network/model/audio/paid/E2E campaign, prune or user cleanup. Unrelated research/worktrees preserved. Test-owned disposable roots only.
