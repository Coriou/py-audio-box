# P1B2.1 independent adversarial review — round 2

2026-10-16 · gpt-6.1-sol / low. **NO-GO for bounded B2.1 source/E1 acceptance** at exact HEAD `d4290c6e0c31e132b6ae6039312c5292e88fb12b`, frozen source/test commit `aae8919cca644b6197c2fb60267ace292a9dfe9d`. Original F1's reproduced hole is closed. F2 is substantially improved but remains blocking: nonterminal Core output documents accept destroyed attempt history and unknown nested job fields, even after independent full-chain rehashing and certificate replacement. No B2.2 kernel or B2.3 host-proof implementation is requested by this verdict.

## Identity, authority and scope

Inspected actual candidate read-only at `/Users/ben/.txcode/worktrees/py-audio-box/t3-361ad50c`. Initial HEAD matches and status is clean. Compared against committed correction baseline `d5b9bb9`: exactly the ten authorized paths change. The correction source delta from imported candidate `ae54dcc` to frozen source is five authorized source/test paths; HEAD after the frozen source changes only RECOVERY-P1B2. Core, initializer, resource/admission/dispatch stores, host script, contracts, dependencies, packaging, legacy, ledger and reviews are unchanged. `git diff --check d5b9bb9 HEAD` passes. No applicable ancestor or scoped AGENTS.md was found.

Read the full P1B2-BOUNDARY-ASTRA architecture, records, plan binding, briefs and exits; original P1 decision/invariants/exit matrix; PRODUCT, implementation start/protocol, relevant canonical shared/Phase 1/evidence guidance, R1 fault authority and B1 round-2 review/handoff. Read the full original P1B21-r1-sol findings/reproducers and the complete candidate RECOVERY-P1B2 original handoff plus correction response, including development failures and superseded receipts. Inspected actual changed production code, relevant diff/tests and Core/DispatchStore/Admission/worker callers. A quick memory registry search supplied no B2-specific authority or acceptance evidence; no old runtime result is used as current proof.

The source chain is `ae54dcc` → `141810a` (grounded evidence/allocation correction) → `9dda031` (accounting/retained-state hardening) → `aae8919` (cancel-request action correction) → report-only `d4290c6`. The handoff candidly distinguishes invalid initial F1 RED fixtures, private SQLite-copy development failures, overlapping superseded runs and the final cancellation correction. Its 193/388/423 receipts remain implementer evidence; independent results are below.

## F1 — high — original reproducer closed

`service/recovery.py:1017-1062` now checks allocation provenance directly rather than depending on an incomplete inventory-reason blacklist. In particular, an executing reservation must have an attempt and an unacknowledged dispatch. Snapshot computes this predicate; `plan_abandon` checks it before first extension DDL and again afterward, and `verify_abandon_plan` checks it on verification. Frozen Authority/context/generation/deadline/latest-attempt, reservation configuration/charge, receipt/outcome and publication ownership joins are present.

Independently reran the exact queued-job/only-reservation-to-executing negative through the candidate fixture, both before first installation and with B2 already installed. Both planning calls refuse `incoherent_b2`; complete logical/schema dumps are unchanged. The installed case also refuses verification with no mutation. Independently reran real allocation positives with an unbound prepare and with an old-lane UNKNOWN runtime: both plan and immediately verify. These cases are also in the fresh 193-test focused run. The reported missing-provenance hole is closed without rejecting those legitimate uncertain populations. This is bounded closure of the original finding, not a universal proof of every possible historical allocation population.

## F2 — high — still open: nonterminal output JSON is not a closed, preserved job contract

Location: `service/recovery.py:645-668`, especially `660-665`; consumed by `_validate_evidence` at `746-749`, then completion/startup readers. The new archive/private-snapshot checks and transitive hashes genuinely improve the old implementation: arbitrary before/monitor/fault hashes, missing archives and changed snapshot logical rows now refuse. The remaining issue is the semantic content accepted inside the hashed output.

`_validate_core_output` treats the job column as a scalar string, decodes it, and for nonterminal actions checks only final state, ID, generation and deadline. It neither validates the closed job schema nor preserves the original attempt list/ownership and other frozen job fields. Consequently a fully rehashed certificate can describe an allocated job becoming failed with **all attempts erased**, or contain an **unknown arbitrary nested job field**. Neither change is a legitimate abandonment output. The terminal branch correctly requires exact old/new row equality; the existing fully rehashed `core-output` regression tests only that terminal branch and therefore misses this nonterminal branch.

Independent disposable-root probes used a real Admission/DispatchStore allocation, then `b2_intent(..., complete=True)` with actual retained snapshot/manifest/certificate. Each probe varied only one semantic output field:

1. Replace `expected[3].output.rows.jobs[0][6]`'s decoded `attempts` with `[]` (original length was one).
2. In a separate fresh fixture, add `invented_field = {'arbitrary': ['unvalidated']}` to that decoded job.

For each, re-encode that job, recompute every expected descriptor's input/output chain, recompute the receipt-complete Core-output binding, update accepted/step SQL records and their canonical digests, and replace the derived control certificate with the corresponding recomputed certificate. The before-image, source snapshot, proof, plan, manifest and original allocation remain unchanged. **Both independent variants are accepted by `completion_binding`, `startup_read`, and `Core(root, historical_local=True)`.** An initial combined variant also reproduced the same acceptance; the separate variants establish that neither semantic change depends on the other.

Reproducer core, after creating the real allocated fixture as in `test_b2_cancel_requested_action_uses_frozen_job_state` (omit cancellation):

```python
r = b2_intent(installation, complete=True)
row = r['expected'][3]['output']['rows']['jobs'][0]
job = json.loads(row[6])
job['attempts'] = []  # Separate variant: add only invented_field instead.
row[6] = encode(job)
previous = r['before_digest']
for descriptor in r['expected']:
    if descriptor['step'] == 'receipt-complete':
        descriptor['output']['core_output_digest'] = previous
    descriptor['input_digest'] = previous
    descriptor['output_digest'] = digest(dict(
        step=descriptor['step'], input_digest=previous,
        output=descriptor['output']))
    previous = descriptor['output_digest']
with closing(sqlite3.connect(root / 'core.sqlite')) as db, db:
    db.execute('UPDATE p1_recoveries SET record=?,digest=?',
               (encode(r), digest(r)))
    for descriptor in r['expected'][:4]:
        step = dict(recovery_id=r['id'], **descriptor,
                    lane=r['new_lane'], epoch=r['new_epoch'])
        db.execute('UPDATE p1_recovery_steps SET record=?,digest=? WHERE step=?',
                   (encode(step), digest(step), step['step']))
name = next(n for n, k in recovery.archive_names(r).items()
            if k == 'certificate')
(root / name).write_text(encode(recovery._certificate(r, step)))
with closing(sqlite3.connect(root / 'core.sqlite')) as db:
    assert recovery.completion_binding(db) is not None
assert recovery.startup_read(root) is not None
core = Core(root, historical_local=True)
core.close()
```

Consequence: the frozen B2.1 reader contract still certifies unsupported caller-authored state inside an otherwise consistent evidence chain. This is not a claim of a public SQL writer, malicious-admin protection, or an executing mutation kernel. Hashes now bind actual documents, but these particular documents are insufficiently validated. The requirement for closed usable records and preserved attempt provenance belongs to this lot; deferring all semantic validation would again freeze an unusable completion contract.

Minimal fix: validate output job JSON with the canonical closed job/context rules and constrain its delta against the frozen before-image to the explicitly allowed unsuccessful transition. Preserve attempt count, identities, generations, owner/staging bindings and prior attempts; allow only specifically defined reap/outcome fields supported by separate operator evidence. Reject unknown job fields and unplanned changes to other frozen fields. Add separate fully rehashed nonterminal negatives for deleted/changed attempts and unknown nested fields, through completion/pre-Core/historical readers, alongside a coherent allocated fail/cancel-requested positive. Keep apply unavailable. No Core source expansion, transition writer, host authentication or new scan is needed for this validator correction.

## Preserved boundaries and independent verification

Source inspection confirms separate exact all-or-none optional B2 DDL, explicit offline-only installation after coherent B1, validate-only constructors, typed references and retained namespace without resolution consumption. Source/history projection excludes only the current proposal and its exact reference label. Pending/partial constructor refusal and both decorated publication checks precede method bodies, including terminal fast return. WAL-aware inspection still opens private copies. Worker checks maintenance under lane lock before journal/boot/listen; actual receipt certificate/original-journal checks are present. Default and explicit enrolled startup use pre-Core reads/authenticated hello, repeat before Admission/A2/R1, add no mount or bypass option, and retain activation closure. Programmatic, packaged CLI and host apply remain unconditionally unavailable. Public 1.3.0, Core 1, R1 protocol 2 and private P1 revision 1 remain unchanged.

Commands use the existing `/tmp/pab-0c-venv/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, no installs and `-p no:cacheprovider`:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**193 passed in 67.97 seconds, exit 0.** Includes original F1 negatives, grounded evidence/archive negatives, cancellation actions, prior completed histories, B1 no-remint/WAL preservation, publication fences and cross-volume startup regressions. The inline F1/provenance positives and F2 adversarial probes described above are additional, disposable-root checks. Their scripts were passed on stdin; no candidate test file was created.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**423 passed, 18 skipped in 140.53 seconds, exit 0.** The focused and full pytest runs were sequential on unchanged source/tests. The separate 388-test affected command was not redundantly rerun. Final candidate status is empty, HEAD remains exactly `d4290c6e0c31e132b6ae6039312c5292e88fb12b`, and the final baseline diff check passes. Passing preservation tests do not close the independently reproduced F2 contract gap.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked Docker/host projections only. Skips are not qualification. A1 whole-Docker-inventory drift remains unexplained and unqualified; no cause or waiver is inferred. B2.2/B2.3, B3, C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending. No physical power-loss/ENOSPC/controller/VM, engine/audio/hardware or clean-host qualification is claimed.

Only this designated root review report is written. No candidate/source changes, commit, subagents, push/PR/deploy, installs, heavy Docker, network/models/private audio/paid operation, remote E2E runner, prune or user cleanup. Disposable test-owned temporary roots only; unrelated research/worktrees are preserved.
