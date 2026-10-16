# P1A2 independent review — round 3

2026-10-16. **GO for bounded A2 source acceptance only** at exact candidate `6723528fbb557c751b4fe69e4ccda5dd470c2687`, correction source `1b7d111c206d55c36e2d9b62f5cc0e4a3f645aeb`, branch `t3/fix-persistence-schema-refs`, inspected in `/Users/ben/.txcode/worktrees/py-audio-box/t3-d36d52bb`. F1b is closed; original F1 and F2 closure is preserved within the local evidence boundary. No new blocking finding. Parent integration verification and final global Astra remain separate. B/C and public/unattended activation remain closed.

## Inspection and exact delta

Read PRODUCT, implementation start/protocol, canonical plan shared architecture/preservation/Phase 1/evidence sections, CONTRACTS 1.3.0, P1 Astra A2/invariants/private format/lot ordering/exit matrix, R1 fault-authority decision, both complete P1A2-r1-sol and P1A2-r2-sol reviews including their reproducers, PERSISTENCE-P1A2 and both correction responses. Inspected actual reconciliation, persistence initialization/epoch transactions, Core call paths, Dispatcher startup/pending/stop/terminal/release/ACK paths and affected tests. No applicable AGENTS.md found. Memory registry search supplied no task-specific decision or evidence for this verdict.

Candidate began and ended clean at the exact SHA. Compared the complete narrow correction against imported `4bc9693`: only reconciliation, persistence tests and the two handoff documents changed. Source/test changes versus `1b7d111` are empty; the final candidate adds only documentation. Persistence, Core, Dispatcher and Dispatcher tests are unchanged by this correction. No worker/contracts/dependency/public-format changes.

## F1b closure and independent reproductions

`service/reconciliation.py:115–162` now constructs a Draft202012Validator with the complete canonical document and evolves it to each candidate branch. Local `$ref` resolution retains the root definitions. Applicability uses actual branch validity, including type/nullability and discriminator constraints, rather than only property constants. Non-null lineage/reuse objects no longer traverse the null branch. Scalar and reference-field type checks refuse invalid values; declared properties still bound traversal, and arbitrary scalar spellings never become references. Singular `reference_asset_id` is explicitly retained.

Independently recreated the previous coherent real-Core project reproducer, without calling the new test helper: saved VoiceDesign version with non-null design lineage, oracle-valid project, real `Core.create_project`, matching retained asset metadata and owned one-byte content. Three scans across two close/reopen boundaries returned `complete=True`, `blocked=False`, `defect_latched=False`, empty reasons and exactly `{'committed_referenced': 1}`. Exact project content survived. This reproduces the original baseline with all dependencies present, so missing media cannot mask a schema refusal.

Additional independent inline probes globally replaced each prepare/design fixture's prospective reference ID with the valid opaque non-prefix ID `opaque.prospective:1`, rehashed the plan and checked `oracle.stored_context` before storage. For each operation, installed all actual dependencies and optionally the prospective output with matching metadata/hash. Each of the four cases was scanned across two reopenings:

- All dependencies present: complete, healthy, exact empty reasons, zero missing diagnostics.
- Prospective reference absent: complete and blocked, exactly one missing diagnostic and exactly `['unavailable']`; subsequent scans had exactly `['unacknowledged_integrity', 'unavailable']`.

No unknown-schema reason appeared. Prepare reuse identity remains null as required by `contracts/validate.py:285`; prepare configuration/settings and design configuration/settings/identity retain their valid non-null objects. An invented non-null prepare identity would be an invalid oracle baseline and was not used.

The added tests at `tests/service/test_persistence.py:724–789` validate project/stored-context baselines, exercise healthy complete summaries and exact reference sources across reopenings, and reject unknown discriminators, nested fields, wrong container types and invalid scalar null. The repaired design-prose test now asserts its exact complete/unavailable summary instead of merely checking prose exclusion. Existing prefix/key/principal/history/dedup, historical revisions and non-prefix reference regressions remain green.

Canonical-schema inspection counted 24 anyOf and 6 oneOf nodes, no allOf and no semantic siblings alongside union/$ref nodes. oneOf cardinality and allOf failure refusal are explicit in source. This acceptance covers the current closed canonical definitions; it is not qualification of arbitrary future JSON Schema composition or a replacement for Core/oracle structural and relational validation. Future schema shapes still need adapter review.

## F2 and lifecycle preservation

Rechecked the unchanged durable boundary: `_begin` commits checking before collection; begin denial performs no observation; only the same uninterrupted token completes diagnostics/history and healthy/unresolved authority atomically. Completion denial/rollback leaves checking; inherited checking never collects or becomes healthy from a fresh sample. Completed unresolved state remains latched. Missing authority next to populated diagnostics starts unresolved. Page fingerprint mismatch is included in the durable completion path. No separate witness store, resource-health attestation or R1 reset was introduced.

Fresh tests include the actual process-death matrix with independently read committed checking, real observed transient orphan followed by query_only diagnostic denial, immediate os._exit without close/retry, parent-owned transient removal, and two blocked reopenings with zero collection. Healthy interrupted/completion-denied and page-denied epochs remain blocked too. Successful page invalidation remains durable after the changed name disappears.

Actual Dispatcher tests retain the original valid runtime receipt and executing reservation across two restarts for all three open-epoch cases, with zero publication/ACK/new launch and fresh admission refused. Conflicting original receipt retention, normal clean receipt recovery, strict unavailable retrieval/history, queued-release refusal and live page-failure stop responsiveness remain green. Source ordering still consumes pending lifecycle replies and services bounded stop before new-work refusal; integrity gates precede publication/release/ACK. No terminal history rewrite or reservation reconstruction was found.

## Fresh exact-candidate verification

Existing `/tmp/pab-0c-venv`, no installs; bytecode/cache disabled. Local synthetic roots/processes only.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_persistence.py tests/service/test_dispatcher.py -k 'inventory or integrity_open_epoch'
```

**39 passed, 79 deselected in 7.85 s, exit 0.** Log `/tmp/p1a2-r3-sol-focused.log`.

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service tests/release
```

**279 passed, 18 skipped in 104.87 s, exit 0.** Log `/tmp/p1a2-r3-sol-full.log`. Includes affected Persistence/Core/Dispatcher/Admission and local release tests. Skips are not installed acceptance. Both runs used the frozen candidate; no source/test modification. The independent original-project and non-prefix prospective probes above also passed. `git diff --check b6d16ef HEAD` passed.

## Remaining boundaries

B1/B2/B3/C1/C2 remain pending, including the actual installed C2 fault gate. A1 global inventory drift remains unresolved; no whole-runner preservation waiver or Docker campaign was inferred. Finite observation/history budgets remain conservative refusal boundaries; SQLite busy handling, aggregate SQL and diagnostic transactions add time outside the observation deadline. Interrupted healthy epochs deliberately cost availability until explicit later recovery. No recovery/acknowledgement/cleanup authority exists in A2.

Evidence is local source/SQLite/synthetic process and application behavior only. It does not establish installed PID-1, physical ENOSPC, power-loss/controller/VM durability, clean-host installation, real engines/audio or hardware qualification. No fixes, source/worker/contracts/dependencies/legacy/ledger changes, commits, subagents, models/private audio/cloud/STT/Voicebox, external/security exploitation, remote E2E, `scripts/run-e2e-test.sh`, push/PR/deploy or unrelated cleanup. Only this report was written in the root repository; its preexisting research artifacts were preserved. Candidate remained clean.
