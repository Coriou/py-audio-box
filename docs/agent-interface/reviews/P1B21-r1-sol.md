# P1B2.1 independent adversarial review — round 1

2026-10-16 · gpt-6.1-sol / low. **NO-GO for bounded B2.1 acceptance** at `ccf3c07c93f504056a538d2f318bdbaa35bb4dd3`, baseline `8ede631c0f1c5428edf58d581f3e1700b5e4432c`. Two blocking validation/interface findings below. This is the first B2.1 review; no previous B2.1 findings were supplied or consulted. B2.2/B2.3 implementation is not required by this verdict.

## Scope and authority

Candidate inspected read-only at `/Users/ben/.txcode/worktrees/py-audio-box/t3-15922103`; HEAD matches and status is clean. The actual baseline diff changes exactly the ten allowed paths. Core, initializer, resource/admission/dispatch stores, host operator, contracts/fixtures, dependencies, packaging, legacy, ledger and prior reviews are unchanged. `git diff --check 8ede631..HEAD` passes. No applicable AGENTS.md was found in candidate ancestors or inspected source/test/docs paths.

Read the complete root P1B2-BOUNDARY-ASTRA decision, including actual call paths, cross-volume protocol, finite records, plan/resume rules, interfaces and exact B2.1 exit matrix; original P1 implementation decision and B2/invariants/exit matrix; PRODUCT, implementation start/protocol, R1 fault-authority decision, both B1 reviews and B1 handoff/corrections. Read the candidate's full RECOVERY-P1B2 response and inspected actual source/diff/callers/tests. A lightweight memory registry search supplied no B2-specific authority or acceptance evidence.

The earlier monolithic stops establish no implementation or Core dependency. The selected initializer and both existing storage-session wrappers are sufficient seams; no Core expansion is requested here. B2.2 transition/resolution/resume semantics and B2.3 authenticated host proof remain later work.

## F1 — high: incoherent executing authority is accepted before optional schema installation

Location: `service/recovery.py:765-770`, called before DDL at `737-740` and again at `750`, and by verification at `782`.

`_coherent_b2` rejects a selected list of inventory reasons but omits `unexplained_executing_reservation`. The jobs/reservation-ID equality check does not establish allocation provenance. A queued job whose reservation alone claims executing, with no dispatch or attempt, passes this gate. This is exactly an incoherent predecessor that the authoritative brief excludes; it is not a legitimate UNKNOWN worker or unbound prepare.

Independent disposable-root reproduction used the existing initialization and Admission fixtures:

1. Create a coherent enrolled installation, instantiate Admission/DispatchStore, submit one queued synthetic job, close Core. No allocation is made.
2. Vary only `resource_reservations.state` from `queued` to `executing` with a test-only SQL UPDATE.
3. Register the unchanged synthetic container/lane coverage and create the ordinary unavailable B1 proposal through `enrolled_plan`.
4. Inspection reports exactly `['unexplained_executing_reservation']`.
5. Call `plan_abandon(..., proof_digest='3'*64, profiles=profiles, budget=DEFAULT_BUDGET, now=100)`, then `verify_abandon_plan` under `offline` ownership at `now=101`.

Observed: **B2 schema and plan are installed, and immediate verification succeeds**. The plan lists the job/reservation but has no dispatch. This is a validator/predecessor failure, not an allegation of a shipped arbitrary-SQL interface. The single SQL mutation constructs the negative fixture, as in the candidate tests.

Consequence: explicit planning can extend an unsupported authority population and freeze a proposal without a coherent allocation origin. Both the pre-installation gate and later verification share the hole. Closed apply currently prevents execution of that proposal; it does not satisfy B2.1's requirement to validate the coherent enrolled B1 predecessor before installing the extension.

Minimal fix: explicitly reject unexplained executing reservations and audit the structural authority reasons against the actual reconciliation joins, distinguishing supported unbound/UNKNOWN dispatches from missing allocation provenance. Add this one-field negative before first B2 installation and after installation; assert no extension/plan mutation in the former and refusal in both. Preserve legitimate queued and coherent executing/unbound positives.

An exploratory second malformed-dispatch probe stopped earlier with IndexError in existing typed traversal; it supplies no additional blocking finding or successful qualification claim.

## F2 — high: frozen completion records validate hash-shaped claims rather than bound step evidence

Locations: `service/recovery.py:507-510` (accepted-record keys), `583-604` (prestate/expected descriptors), `689-695` (step validation), `443-453` (completion authority); candidate fixtures `tests/service/test_recovery.py:471-488`.

The frozen accepted record contains only `before_digest`, `monitor_digest`, `faults_digest` and generic step input/output hashes for the relevant evidence. Validators check hash syntax and equality between caller-authored descriptors/receipts, but never recompute these hashes from a closed before-image/step-output document or validate a binding to such a document. The same generic six-field record represents every step. The completion certificate hashes that generic Core-complete receipt, so agreeing strings become startup completion authority without establishing what was completed.

The candidate's own “complete” fixture illustrates the gap: before/monitor/fault/action digests and expected outputs are repeated-character constants, actions are always `retain-terminal`, and no evidence archive is produced. This is not a coherent completed recovery fixture for the mandatory cryptographic-binding checks. Its startup positive therefore proves agreement of labels, not agreement with retained recovery evidence.

Independent reproduction:

```python
r = b2_intent(installation, complete=True)
r['before_digest'] = 'a' * 64
r['monitor_digest'] = 'b' * 64
r['faults_digest'] = 'c' * 64
# Replace only that record and recompute its ordinary canonical row digest.
db.execute('UPDATE p1_recoveries SET record=?,digest=?', (encode(r), digest(r)))
assert recovery.completion_binding(db) is not None
b2_receipt(roots, r, complete=True)
core = Core(roots['control'] / 'core')
assert list(core.root.glob('p1-recovery-*')) == []
core.close()
```

Observed: arbitrary prestate hashes accepted; completion returned; Core opened with zero evidence archives. The already-passing explicit-startup positive uses this same ungrounded completion construction. No production writer or malicious-administrator security bypass is claimed: the finding is that the B2.1 frozen validators and required coherent fixtures do not establish the advertised completion binding.

Minimal fix: freeze bounded, closed evidence/before-image and per-step output descriptions (or exact typed references to bounded immutable documents), including the required prior authority/proof/expected output bindings, and recompute their digests. Completion readers must validate the certificate's evidence chain rather than only equal hash strings. Build a coherent test-only completed fixture from those documents; vary each bound prestate/output/archive/certificate field independently and require refusal. Keep production apply unavailable. This asks for the usable B2.1 record/reader contract required by the brief, not implementation of the B2.2 mutation kernel, host authentication, scans or resume operations.

## Preserved boundaries and verification

Positive source/test observations: separate all-or-none B2 schema with exact DDL and finite bounds; constructor validates without installing; pre-body pending/retirement checks in both existing decorated publication methods including terminal fast return; source-preserving private WAL copies retained; distinct self-excluding B2 plans with prior-history binding; lane-lock maintenance read before journal/boot/watchdog/listen; explicit and default enrolled pre-Core startup checks and repeated hello before Admission; no new mounts; activation remains closed. Programmatic apply, packaged CLI apply and host CLI apply remain unconditionally unavailable. Reconciliation changes are confined to schema/typed references/recognized evidence namespace and do not consume resolutions yet.

Fresh independent commands used the existing interpreter with no installs, bytecode or pytest cache:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service/test_recovery.py tests/service/test_dispatcher.py tests/service/test_packaged_lifecycle.py -q
```

**149 passed in 43.64 seconds, exit 0.** Includes B1 regressions, nonempty immediate verification/history staleness, publication pre-body fences, worker refusal and cross-volume/default/explicit startup checks. Independent inline probes above are additional to these passing assertions.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -p no:cacheprovider tests/service tests/release -q
```

**379 passed, 18 skipped in 114.45 seconds, exit 0.** The full run began while the focused run was finishing; both use disposable local synthetic roots. No heavy resource lane ran. No source/test edits occurred. These results independently confirm the candidate's full-suite count, but do not close the two adversarial findings. The implementer's separate 344-test affected command was not duplicated; its reported result remains attributed to the handoff.

## Evidence limits and write boundary

E1 local source/SQLite/filesystem/process and mocked Docker projections only. Installed/platform skips are not qualifications. A1 whole-Docker-inventory EXIT 1 remains unexplained and unqualified, with no cause or waiver inferred. B2.2/B2.3, B3, C1/C2, parent integration verification, final Astra and public/unattended acceptance remain pending. No physical power-loss/ENOSPC/controller/VM, engine/audio, hardware or clean-host claim.

Only this designated root review report was written. No candidate/source edits, commit, subagents, push/PR/deploy, installs, Docker/model/audio/network/paid campaign, run-e2e-test.sh, prune or user cleanup. Test-owned disposable temporary roots only; unrelated research/worktrees preserved.
