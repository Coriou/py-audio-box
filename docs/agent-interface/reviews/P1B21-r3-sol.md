# P1B2.1 independent adversarial review — round 3

2026-10-16 · gpt-6.1-sol / low. **GO for bounded B2.1 source/E1 acceptance only** at exact HEAD `5959d8fb9b577c5e6b4cf5f30276f712a4ee4fa1`, frozen source/test commit `0d976dac304db7eccdfb60d2078defda07026fdf`. Original F1 remains closed; original F2 and its round-2 nonterminal-output remainder are closed within the evidence below. No new blocking finding. This does not approve integrated P1, an abandonment writer, trusted host apply, installed operation or activation. Parent integration verification and final Astra remain separate gates.

## Identity, scope and review basis

Inspected candidate read-only at `/Users/ben/.txcode/worktrees/py-audio-box/t3-b05a2844`. Initial HEAD matches and status is empty. Relative to fresh-root baseline `361deab2a9269800925cf9f7095fde033f98c680`, exactly the original ten authorized paths change. HEAD differs from frozen source only in RECOVERY-P1B2. Protected Core, initializer, resource/admission/dispatch stores, host operator, contracts, dependencies, packaging, legacy, ledger and prior reviews are unchanged. `git diff --check 361deab HEAD` passes. No applicable ancestor or scoped AGENTS.md found.

Read the complete P1B2-BOUNDARY-ASTRA architecture, records, plan binding, briefs and exits; both complete P1B21 reviews and reproducers; and the entire candidate RECOVERY-P1B2 including original handoff, all corrections, development failures, superseded receipts and latest response. Consulted PRODUCT, implementation start/protocol, canonical shared/Phase 1/evidence guidance, original P1 invariants/exit matrix, R1 authority, both B1 reviews and B1 handoff/corrections. Inspected actual production source/diff, canonical job/context oracle, allocation callers, publication wrappers and worker/control startup ordering. Memory lookup supplied no B2-specific authority or acceptance evidence; no prior runtime receipt is reused as fresh proof.

## F1 — original high finding remains closed

`service/recovery.py:1036-1084` retains structural allocation joins. An executing reservation needs an attempt and unacknowledged dispatch; frozen Authority/context/generation/deadline/attempt, demand/charges, receipt/outcome and publication ownership are checked. `plan_abandon` checks coherence before optional DDL at lines 1010-1012 and again after installation; verification repeats it at line 1094.

Separately invoked the exact original one-field queued-reservation-to-executing regression in fresh disposable roots, before installation and with B2 installed. Both refuse and preserve the complete logical/schema dump; installed verification also refuses without mutation. Separately invoked real allocated unbound-prepare and old-lane UNKNOWN positives: both plan and immediately self-verify. All four probes exit 0 and also occur in the focused suite. No relaxation of this earlier fix was found. No further fix requested for F1.

## F2 — original evidence finding and round-2 remainder closed

The grounded before/source/proof documents, actual snapshot/manifest/certificate checks, transitive step hashes, prior completion/history checks and receipt journal archives remain present (`service/recovery.py:425-482, 689-829, 991-1000`; `worker/runtime_journal.py:145-218`). Readers still require actual evidence rather than matching digest labels. Full-chain rehashing does not excuse changed snapshot logical rows or terminal output; retained prior evidence is checked again for later completion.

The targeted correction is `service/recovery.py:645-686`, especially 661-685. Both before/output job documents pass the canonical closed job validator; stored context has exactly `plan bundles`, and the canonical stored-plan/input relation is checked for both jobs. Active-attempt validation enforces ordered unique generations, current generation and reaped prior attempts. Every SQL job-row column outside job JSON stays equal. Terminal rows remain exactly equal. Nonterminal output must equal a deep copy of the validated before-job with only the prescribed unsuccessful state and latest attempt's `reaped=True`. Thus attempt count/order/identity/ownership/staging, prior attempts, stage, artifacts, input, principal/idempotency, generation and deadline remain frozen. Action choice still uses actual `cancel_requested`; never-started queued cancellation retains no attempts. This is a conservative closed output contract, not a transition writer or fabricated PID-1 receipt.

Independent stdin probes rebuilt real Admission/DispatchStore allocations in three separate disposable installations. One unchanged allocated fail witness and two separate mutations were tested: (a) erase its one attempt; (b) add only `invented_field={'arbitrary':['unvalidated']}`. For each fixture, re-encoded the output job, recomputed every expected descriptor's input/output chain and receipt-complete binding, rewrote accepted/step SQL records with canonical digests, and replaced the derived control certificate. Before/source/proof/snapshot/manifest/allocation stayed unchanged.

Results through **each** of `completion_binding`, `startup_read` and `Core(..., historical_local=True)`:

- Unchanged allocated fail: accepted.
- Attempt erasure: refused `b2_b2_output_job_outcome`.
- Unknown nested job field: refused `b2_b2_output_job_contract`.

The independent script exited 0. This directly rechecks the separate round-2 reproductions, rather than relying on an un-rehashed digest rejection. The focused suite additionally passes all 34 fully rehashed allocated fail/cancel cases, including positive witnesses and changes to attempts, nested fields, context and other frozen fields through all three readers. No further fix requested for F2; no remaining concrete blocker identified in the reviewed contract.

## Preserved boundaries

Separate exact all-or-none B2 DDL remains explicit offline-only after coherent enrolled B1; constructors validate without installation. SQL/logical evidence and inventory remain bounded to 2,048 entries / 16 MiB with finite deadlines; journal ceilings remain 64 records / 1 MiB. History consumes bounds rather than being evicted. Source projection excludes only the current proposal and its exact typed reference label, retaining all prior plans/history. Reconciliation changes remain schema/namespace/typed-reference adapters without resolution consumption.

Pending/partial state refuses direct Core, including historical opt-in. `storage_session` checks before both publication bodies, including terminal fast return; exact permanent retirement and unchanged terminal SQL/media are exercised. Private WAL-aware copies preserve original inspection namespace/bytes. Journal checks maintenance under lane lock before loading/boot/watchdog/listen. Both default and explicit enrolled Dispatcher startup perform pre-Core validation/authenticated hello, repeat completion/hello under Core ownership before Admission/A2/R1, and retain activation closure. No new mount or bypass flag is introduced. Programmatic, packaged CLI and host apply remain unconditionally unavailable. Public 1.3.0, Core 1, R1 protocol 2 and private P1 revision 1 remain unchanged.

## Fresh verification

Existing interpreter only, no installs; bytecode and pytest cache disabled. Focused and full runs are sequential on frozen source/tests. The separate affected-full campaign was not redundantly rerun. Independent disposable-root probes above are additional; the short F1 fixture invocations overlapped part of the full E1 run.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**227 passed in 94.02 seconds, exit 0.** Includes prior F1, grounded evidence, history/archive, startup/publication, B1 no-remint/WAL and new output-contract regressions.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**457 passed, 18 skipped in 163.26 seconds, exit 0.** This followed the focused run on unchanged source/tests. Skips remain unqualified. Final candidate status is empty, HEAD remains exactly `5959d8fb9b577c5e6b4cf5f30276f712a4ee4fa1`, and `git diff --check 361deab HEAD` passes. The frozen-source-to-HEAD delta remains report-only.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked host projections only. Installed/platform skips are not qualification. A1 whole-Docker-inventory drift remains unexplained and unqualified, with no inferred cause or waiver. B2.2 kernel/resolution/resume, B2.3 trusted host proof/apply, B3, C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending. No physical power-loss/ENOSPC/controller/VM, engine/audio/hardware or clean-host claim.

Only this designated root review file is written. No candidate/source edit, commit, subagent, push/PR/deploy, installs, Docker/audio/model/network/paid campaign, remote E2E runner, prune or user cleanup. Test-owned disposable roots only; unrelated research and worktrees preserved.
