# P1A2 independent review — round 2

2026-10-16. **NO-GO for bounded A2 source acceptance** at exact candidate `43e6c8d51a3b4998084ee5ee74e1b1e05c42ee08`, correction source `76e693f47d232e0ce3be590dfa97cdbb3da83d7c`, branch `t3/typed-integrity-fence`, inspected in `/Users/ben/.txcode/worktrees/py-audio-box/t3-ae3b1e13`. One independently reproduced P1 blocker remains in the replacement F1 adapter. F2's bounded local correction is closed. This is neither final global Astra acceptance nor P1 activation.

## Blocking finding

### F1b — P1: valid nullable object branches permanently quarantine coherent data

Location: `service/reconciliation.py:131–146`, especially branch selection at 134–139 and unknown-field classification at 145–146. Canonical examples: `voice_version.design_lineage` (`contracts/v1/schema.json:221`) and `resolved_execution.reuse_identity/reuse_settings/reuse_configuration` (3561 onward).

The walker selects union branches solely by property constants. For `anyOf: [{$ref: ...}, {type: null}]`, both branches pass that selection for a non-null object. It correctly traverses the referenced object, then traverses the same value against the null branch. That branch has no properties, so every legitimate object key becomes `unknown_reference_schema`. A complete scan commits `unresolved`; subsequent coherent restarts cannot clear the false defect. This is valid contract data, not an unknown schema, a malformed test row, or missing media.

Independent real-Core reproduction on the final candidate, with only an owned temporary root:

```python
x = json.loads(Path('contracts/fixtures/v1/transcript-voice_design.json').read_text())
p = json.loads(Path('contracts/fixtures/v1/transcript-project_render.json').read_text())['context']['plan']['project']
v = next(e['response'] for e in x['exchanges'] if e['operation'] == 'voice_version_get')
a = next(e['response'] for e in x['exchanges'] if e['operation'] == 'asset_get')
a['sha256'] = hashlib.sha256(b'x').hexdigest()
for key in ('speakers', 'blocks', 'takes', 'tracks', 'clips'):
    p[key] = []
p['voices'] = [dict(id=v['voice_id'], aliases=[], version_ids=[v['id']], archived=False, revision=1)]
p['voice_versions'], p['assets'] = [v], [a]
oracle.validate('project', p)  # passes structural and relational oracle
c = Core(root)
c.create_project(p)          # real supported API validates it again
c.db.execute('INSERT INTO assets VALUES(?,?)', (a['id'], json.dumps(a)))
(c.root/'assets'/a['id']).write_bytes(b'x')
summary = Reconciliation(c).reconcile_startup()
```

Observed: `complete=True`, `blocked=True`, `defect_latched=True`, counts exactly `{'committed_referenced': 1}`, reasons exactly `['unknown_reference_schema']`. The scanner verifies the present file against its metadata; there is no orphan/missing/hash defect. After close/reopen, counts remain committed-referenced and reasons become `['unacknowledged_integrity', 'unknown_reference_schema']`. A separate untouched VoiceDesign fixture, checked by `oracle.stored_context` and stored in the real jobs shape, also produces `unknown_reference_schema`.

Required correction: select applicable union branches using their actual schema/type/discriminator semantics, including `$ref` and nullable objects; do not suppress unknown-field refusal globally. Add a coherent valid project with a non-null design lineage and present matching media, plus valid prepare/design resolved snapshots. Assert absence of unknown-schema refusal and preserve genuine opaque media references and unknown-schema negatives. The new description/instruction test currently checks only that prose names are absent; it ignores its summary, so this false refusal passes the test. Existing render tests deliberately expect blocked missing dependencies and likewise cannot establish absence of accidental schema refusal.

## Prior findings and A2 preservation

- **Original F1 scalar-prefix reproducer closed:** independently submitted `asset-management is useful.` through real `Core.submit_synthetic`, with `asset-principal` and `asset-key`. Three scans across coherent reopenings returned complete, healthy, empty reasons/counts; duplicate submission returned the original job. Prefix inference is removed. Explicit table/column adapters cover assets/artifacts, publications/work, jobs/frozen contexts, all revisions and dispatch authority/outcome. Opaque non-prefix positive regressions remain passing. The replacement adapter is nevertheless not accepted because of F1b. The singular `reference_asset_id` is not in the adapter's reference-key set; valid saved voice versions also carry its oracle-required equal `reference_asset_ids`, but prospective reuse configurations deserve explicit reference coverage in the correction.
- **F2 closed within E1:** `_begin` commits a separate checking token before `_collect`; failed begin performs no collection. Completion and diagnostic/history writes share one transaction with same-token conditional completion. Rollback leaves checking, and inherited checking cannot be sampled healthy. Missing authority beside populated diagnostics starts unresolved. Completed unresolved state is carried through reinspection. Constructor initialization performs schema/diagnostic-presence checks, not the defect-bearing namespace inventory; Dispatcher inventories before cancellation release and receipt/publication replay. No R1 health attestation/reset was added.
- Independent F2 probe began from a clean real Core scan, created a one-byte owned orphan, independently read committed `checking` from another SQLite connection at collection entry, ran actual collection and asserted the orphan was observed, then enabled `query_only` **after observation**. Diagnostic completion failed with `inventory_persistence_failed`; the child immediately used `os._exit(29)` without close/retry. The owning parent removed only that transient name. On each of two reopenings the authority remained checking, summary was blocked/latched with `unacknowledged_integrity`, and an observation sentinel was never called. Child timeout was 10 seconds and the parent waited/reaped it.
- Fresh full-suite execution includes all three open-epoch Dispatcher cases with the original valid runtime receipt over two restarts: receipt/binding retained, reservation executing, zero publication/ACK/new launch, fresh key refused. Also retained: exact clean receipt recovery, original conflicting receipt, pending lifecycle/page-fault stop handling, dedup/read paths, queued release refusal, and R1's independent authority tests. Source review found the integrity checks before terminal publication/release and outcome ACK; pending lifecycle processing remains ahead of new-work refusal.
- Page revalidation now uses the same precommitted epoch, and fingerprint mismatch participates in completion. Page completion denial and immediate death leave checking; successful invalidation remains unresolved after the changed name disappears. Ordinary clean coherent restart remains available. Death overlapping a healthy inventory deliberately costs availability; a later healthy sample or exact reap receipt cannot clear it. This limitation is explicitly reported, not unchanged W2 availability.
- Finite entry/file/metadata budgets, owner-thread API, bounded opaque pagination, descriptor-relative NOFOLLOW traversal, per-name hardlink charging, retained orphan/unknown identity/history and unavailable-media history remain present. Historical object/scan/history serialized bytes are summed before diagnostic fetch/JSON decode. Aggregate SQL work, busy handling and diagnostic transactions remain additional to the observation deadline; no hard realtime claim is accepted. No cleanup/recovery authority exists in A2.

## Actual verification

Existing venv; no install; bytecode/cache disabled. Fresh final-candidate full local suite:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service tests/release
```

**269 passed, 18 skipped in 105.19 s, exit 0.** Log: `/tmp/p1a2-r2-sol-full.log`. This includes the affected Core/Persistence/Dispatcher/Admission suites and new process-death/receipt regressions. Passing tests do not cover or negate F1b. Independent inline probes above were additional checks, not source/test modifications. `git diff --check b6d16ef HEAD` passed; candidate worktree remained clean.

Inspected PRODUCT, implementation start/protocol, canonical plan shared/preservation/Phase 1/evidence boundaries, CONTRACTS 1.3.0, P1 Astra A2/invariants/private format/lot and exit matrix, R1 fault-authority lesson, prior P1A2-r1 review, PERSISTENCE-P1A2 and RESPONSE-P1A2-r1; actual correction diff and original A2 Core/Dispatcher seams, scanner/initializer/transaction/dispatch-store call paths and affected tests. No applicable AGENTS.md was found. The memory registry supplied no task-specific authority or verdict.

## Remaining gates and write boundary

B1/B2/B3/C1/C2 remain pending, including actual installed C2. A1 global inventory drift remains unverified; no Docker campaign or global preservation waiver was inferred. Parent integration verification and final global Astra remain separate. P1/public/unattended activation remains closed.

Evidence is bounded local source, SQLite and synthetic process/runtime behavior. It does not establish installed PID-1, physical ENOSPC, power-loss/controller/VM durability, real models, audio quality or hardware acceptance. No source fix, worker/contracts/dependency/legacy/ledger edit, commit, subagent, private audio/model/cloud/STT/Voicebox access, external/security exploitation, remote E2E, prohibited `scripts/run-e2e-test.sh`, push/PR/deploy or unrelated cleanup. Only this report was written in the root repository; existing research and the clean candidate were preserved.
