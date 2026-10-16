# P1A2 independent functional/reliability review — round 1

2026-10-16. **NO-GO for bounded A2 source acceptance** at candidate `54730c631f8708dd577be5a794215df205c3df7d`, source `801cfbba590f28f9a999d810f3c4b50099c3b039`, base `b6d16ef31ef478356a2937fb4739da9b0fd0adfc`. Two independently reproduced blockers below. This is not final global Astra or P1 activation acceptance.

## Scope and inspection

Reviewed the clean candidate in the assigned worktree/branch, the complete seven-file diff, complete PERSISTENCE-P1A2 report, PRODUCT, start/protocol, canonical implementation plan's shared boundaries/evidence/Phase 1, CONTRACTS 1.3.0, complete P1-IMPLEMENTATION-ASTRA decision (A2, invariants, private format and exit matrix), R1 fault-authority decision and accepted A1 review. Inspected actual Core, reconciliation, Dispatcher, DispatchStore, Persistence, and affected tests/call paths. Candidate versus source SHA differs only by the handoff report. No applicable repository AGENTS.md was found. Memory registry search returned no relevant entries; no prior memory fact supplies this verdict.

## Blocking findings

### F1 — P1: ordinary authored text is treated as a missing asset and permanently quarantines valid work

Location: `service/reconciliation.py:125–135`, followed by missing-reference classification at `300–306`.

The recursive walker treats **any string value** beginning `asset-` as a media reference, including request text and other scalar fields. This is independent of its typed field. The resulting fake reference becomes `missing_hash_mismatched_terminal`, `unavailable`, and a durable defect latch. This is a normal supported input, not an unknown schema or corrupt installation.

Independent disposable-root repro on final source:

```python
ctx = json.loads(Path('contracts/fixtures/v1/transcript-speech_generate.json').read_text())['context']
ctx['plan']['request']['text'] = 'asset-management is useful.'
c = Core(root)
job = c.submit_synthetic('p', 'key', ctx)  # accepted, queued
summary = Reconciliation(c).reconcile_startup()
```

Observed `complete=True`, `blocked=True`, `defect_latched=True`, counts `{'missing_hash_mismatched_terminal': 1}`, reasons `['unavailable']`; the diagnostic name is the entire sentence. No asset was required by this preset speech input. A later Dispatcher startup therefore refuses valid queued/fresh work, and the retained diagnostic prevents ordinary recovery even after changing unrelated current data.

Required correction: use explicit schema-aware reference locations across all required tables/revisions/contexts, preserving true non-prefixed managed IDs. Do not infer references from arbitrary prose, principal/key values or descriptions. Add ordinary valid `asset-` text/key/history negatives alongside true-reference positives, including restart behavior.

### F2 — P1: an observed integrity defect loses its latch across restart when its diagnostic write fails

Location: `service/reconciliation.py:311–355`, especially observation before the first diagnostic transaction at `345`, and restart reconstruction exclusively from successfully persisted prior rows at `327–337`.

The live summary fails closed after a diagnostic write denial, but no durable uncertainty precedes observation. A defect can be observed, the diagnostic commit denied, and the transient name disappear before reopen. Restart then reports healthy and permits progress without operator accounting. This violates A2's detected-defect-until-explicit-recovery rule and the report's general restart-latch claim. It is distinct from inherited R1 resource authority; no resource-health inference or R1 reset is involved.

Independent disposable-root repro on final source:

1. Create Core/Reconciliation and commit a clean inventory.
2. Create `assets/asset-orphan` containing one byte.
3. Set the owner connection `PRAGMA query_only=ON`; call `reconcile_startup()`.
4. Observe `complete=False`, `blocked=True`, `defect_latched=True`, orphan count 1 and `inventory_persistence_failed`.
5. Restore query_only OFF, remove that synthetic name, close and reopen Core/Reconciliation.
6. Inventory returns `complete=True`, `blocked=False`, `defect_latched=False`, empty reasons/counts, with no acknowledgement.

The orphan removal models a transient local writer; the reviewer did not remove user data. An abrupt exit after step 4 instead of close has the same durable-state problem: the refused transaction wrote no witness. Rechecking current bytes cannot establish that no earlier defect was observed. The handoff's sentence that later reopen inspects actual durable state accurately describes this limitation, but does not close the requested latch behavior.

Required correction: establish a durable uncertainty boundary before the observation that may latch integrity, or obtain an explicit revised authority decision. Failed precommit must prevent observation/dependent progress; inherited incomplete observation must remain inspect-only. Do not invent a second always-writable store, silently reinterpret a healthy resource monitor, or claim a memory latch survives death. Test write denial plus transient disappearance/restoration and two reopenings, as well as clean coherent restart and stop responsiveness.

## A2 closure assessment

- Owner-thread API, idle explicit scan, 20-entry opaque generation cursors, finite entry/file/metadata observation budgets and incomplete refusal are implemented. Descriptor-relative NOFOLLOW traversal, per-name hardlink charging, conservative unexplained namespace handling and no cleanup permission are present. Pages revalidate fingerprints; database/filesystem changes refuse a generation.
- All recognized current non-diagnostic tables, all project revisions and frozen contexts are traversed; metadata itself remains retained. Unknown tables/column sets/reference kinds refuse. **Reference correctness is not closed: F1.** The generic recursive heuristic is not an adequate substitute for typed adapters.
- Orphan/unknown/pending/missing classes, private identity/owner/reference records, first observation and append-only history exist. Missing/corrupt terminal assets keep job/event history and strict retrieval. **Restart persistence of observed defects is not closed: F2.** Page invalidation also has only a live failure latch until subsequently persisted; do not generalize it into crash-durable evidence.
- Startup checks unresolved dispatch authority, job attempt/context/deadline, receipt/runtime/reap, publication tuple and reservations; terminality alone grants no ACK/release. Existing exact normal receipt recovery and conflicting-original-receipt retention are covered by final-source tests. No demand reconstruction was added.
- New integrity gate is separate from resource authority, preserves metadata/idempotent reads and blocks new allocation/publication/release/ACK. Pending lifecycle handling precedes admission refusal; live invalidation requests bounded stop. No resource measurement/healthy attestation was added by inventory and no R1 fence reset was found.
- Bounds are not a hard realtime guarantee: SQLite busy timeout, historical diagnostic reads/JSON decoding and diagnostic writes occur outside the observation deadline. In particular `327–337` limits historical record count but does not charge their aggregate serialized bytes before materialization. This remains an explicit reliability limitation to resolve/qualify; no hostile-input performance claim was tested here.

## Fresh final-source verification

Existing isolated venv, bytecode/cache disabled, no installs:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_persistence.py tests/service/test_dispatcher.py tests/service/test_core.py tests/service/test_admission.py
```

**205 passed in 90.72 seconds, exit 0**, log `/tmp/p1a2-r1-sol-tests.log`. This is fresh affected-suite evidence on the final source; passing existing tests does not cover or negate F1/F2.

The two additional repros above used the same interpreter and TemporaryDirectory-owned synthetic roots, and independently produced the stated bad outcomes. They were standalone local checks, not changes to candidate tests. `git diff --check b6d16ef HEAD` passed. The implementer's 203-target and 249-pass/18-skip broader results precede its final refinements; its final 30 focused passes are supporting historical evidence only. This review does not label any earlier broader run a final-source full suite.

## Remaining gates and boundaries

B1 deployment/volume enrollment and recovery planning, B2 proof-bound acknowledgement/abandonment, B3 cleanup, C1 persistent topology and **C2 actual installed gate remain pending**. No operator acknowledgement or cleanup authorization exists in A2. A1's global inventory drift remains unresolved; neither global preservation nor a green whole Docker runner is inferred or waived. No new Docker campaign was justified or run. Final global Astra and parent integration verification remain separate.

Evidence is bounded local synthetic source/process verification. It does not prove installed PID-1 authority, physical ENOSPC, clean-host installation, power-loss/VM/controller durability, real engines, acoustic quality or hardware qualification. No source fixes, dependencies, worker/contracts/legacy/ledger edits, commits, subagents, user/private audio, models, cloud/STT/Voicebox, remote E2E, prohibited run-e2e script, push/PR/deploy or unrelated process cleanup. Only this report was written in the root repository; existing research was preserved and candidate sources were not modified.
