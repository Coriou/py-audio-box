# P1A2 round-2 narrow correction candidate

2026-10-16. Implementation handoff for independent Sol-low re-review. No self-GO, parent integration acceptance, final Astra acceptance or P1 activation.

## Identity and imported history

Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-d36d52bb`, branch `t3/fix-persistence-schema-refs`, initially clean at `b4896ce`. No extra checkout or subagent. Inspected `git log --reverse --oneline b6d16ef..43e6c8d51a3b4998084ee5ee74e1b1e05c42ee08`: the three intervening parent documentation commits were already ancestors; imported ALL four original A2 and correction commits oldest first:

- `55d8adb73e14d1a9c487ab857b13ab874f07fa6f` → `b0ca4d6` (original source).
- `840af6022f01cb88ed3ee262d407364cbfde3815` → `a84ee73` (original report).
- `76e693f47d232e0ce3be590dfa97cdbb3da83d7c` → `099c416` (round-1 source).
- `43e6c8d51a3b4998084ee5ee74e1b1e05c42ee08` → `4bc9693` (round-1 response/report).

`git diff 43e6c8d HEAD -- service tests` was empty before correction. Read PRODUCT, implementation start/protocol, canonical plan shared/Phase 1/evidence boundaries, CONTRACTS 1.3.0, P1 Astra A2/invariants/private-format/lot/exit matrix, PERSISTENCE-P1A2, RESPONSE-P1A2-r1 and both complete P1A2-r1-sol/P1A2-r2-sol reviews. No applicable AGENTS.md found. Memory registry search found no task-specific guidance.

Correction source/tests: **`1b7d111c206d55c36e2d9b62f5cc0e4a3f645aeb`**. Final documentation-only handoff commit is supplied separately. Corrective delta is only `service/reconciliation.py`, `tests/service/test_persistence.py`, this response and the current-candidate pointer in `PERSISTENCE-P1A2.md`. No further allowlist expansion was needed. Imported A2 retains its original Core/Dispatcher/private schema changes; this round changes none of those semantics, integrity epochs or schema versions. No contracts/fixtures, worker, dependency/packaging, legacy, ledger or review changes.

## Finding response and preserved boundaries

F1b: union applicability now uses the already installed `jsonschema.Draft202012Validator` against the canonical document, resolving `$ref` through its root context. Nullable object branches, declared types and oneOf discriminators are selected by actual schema validity. oneOf requires exactly one applicable branch; allOf requires every branch. The walker still limits traversal to declared properties and explicit table/column adapters. Unknown properties/variants and invalid scalar/reference/container types refuse; no blanket unknown-schema suppression. Null is accepted only at a declared nullable location. This is a reference adapter, not a replacement contract/relational validator; Core/oracle retain validation ownership.

Added singular `reference_asset_id` to the declared media-reference fields beside plural fields. Valid prepare/design resolved configurations therefore retain prospective output references even before a saved version duplicates the reference in a plural list. IDs remain opaque; no scalar prefix inference returns. Missing prospective references produce exactly `unavailable` (plus durable `unacknowledged_integrity` on later scans), without unknown-schema reasons.

The request for a non-null prepare reuse identity conflicts with the existing oracle: `contracts/validate.py:frozen_plan` explicitly requires `reuse_identity is None` for every operation except voice_design. Tests use non-null prepare reuse configuration/settings, null prepare reuse identity, and non-null design reuse configuration/settings/identity, and call `oracle.stored_context` before storage. No contract amendment was made. Null design_lineage on a valid saved prepare version and non-null design_lineage on a valid saved design version are both exercised through `oracle.validate('project', ...)` and real `Core.create_project`.

Original F1 prefix regressions, non-prefix references, authored history, exact original key/principal/job and dedup remain. The existing design-description/instruction test now asserts its complete summary and exact one unavailable dependency, instead of checking prose exclusion alone. F2 begin/completion/page/restart/death authority, Dispatcher receipt/ACK/reservation/stop behavior and R1 authority code are untouched.

## Regression matrix

| Boundary | Final-source assertion |
| --- | --- |
| Real coherent project with saved VoiceDesign non-null lineage and matching present metadata/bytes | Oracle and Core accept; complete healthy scan with exactly one committed-referenced asset, empty reasons; three scans across reopenings retain exact project. |
| Valid saved prepare version with null lineage | Same healthy real-project/restart assertions. |
| Valid prepare/design stored snapshots, all actual input and prospective media present | Oracle stored_context accepts; complete healthy scans across two reopenings; exact diagnostic names and source references. |
| Singular prospective reference missing | Exactly one missing diagnostic and exact unavailable reasons; no unknown-schema reason, including two reopenings. |
| Nullable reuse fields | Prepare identity null and design identity object pass; existing preset speech null reuse config/settings/identity stay healthy. |
| Nested refs, all historical revisions, opaque non-prefix IDs | Existing positive reference tests retained. |
| Unknown discriminator, nested field, wrong container type, invalid scalar null | Blocked with unknown schema/kind, no ignore path. |
| Original prefix/prose/key/principal/history/dedup | Existing tests retained; design prose now checks actual refusal summary. |
| F2 precommit denial/immediate death/page failure/two restarts | Original process barriers and zero-observation/inherited-checking tests retained. |
| Dispatcher pending stop, original exact/conflicting receipts, reservation and ACK gates | Existing suite retained; no Dispatcher change. |

## Commands, results and failures

Existing `/tmp/pab-0c-venv`; no installs. All pytest commands use `PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider`. Local owned temporary synthetic roots/processes only.

RED: temporarily loaded the exact imported `4bc9693:service/reconciliation.py` in this worktree, ran `tests/service/test_persistence.py -k 'valid_design_lineage or valid_resolved_snapshot'`, then restored correction source in a finally block. **5 failed, 1 passed, 76 deselected / 0.59 s**, exit 1, `/tmp/p1a2-r2fix-red-final.log`. The null-lineage prepare project already passed; the coherent non-null design project and all four resolved snapshot cases failed on original source.

GREEN: `tests/service/test_persistence.py tests/service/test_dispatcher.py -k 'inventory or integrity_open_epoch'`: **39 passed, 79 deselected / 8.03 s**, exit 0, `/tmp/p1a2-r2fix-green-final.log`.

Final-source full local suite (once after source freeze): `tests/service tests/release`. **279 passed, 18 skipped / 107.66 s**, exit 0. Skips remain platform/installed boundaries, not passed installed tests. Log `/tmp/p1a2-r2fix-full.log`. No source/test edits after launching this run. `git diff --check` passed before the run; final check also passed after the run.

Development failures: first focused run 31 passed/3 failed because new assertions incorrectly demanded non-null prepare identity and omitted metadata's own reference source. Corrected to oracle-valid prepare semantics and exact reference-source lists. A subsequent focused run was 34 passed. Expanding project coverage to null prepare lineage initially selected its input asset instead of its saved version's reference asset (38 passed/1 failed); corrected selection by the version's declared reference ID. Intermediate red run included that setup failure; only the final RED above is defect evidence. These failures were not waived.

## Limits and next gate

E1 local source/SQLite/synthetic process evidence only. No installed PID-1, physical ENOSPC, power-loss/controller/VM durability, hardware/acoustic or clean-host claim. A1 global inventory drift remains unresolved; no Docker campaign was needed or run. C2 actual installed gate and B1/B2/B3/C1 remain pending; no recovery/cleanup authority, migration, unattended/public activation or global preservation waiver.

No models/private audio/cloud/STT/Voicebox, external/security exploitation, remote E2E, `scripts/run-e2e-test.sh`, push/PR/deploy or unrelated process/data cleanup. Next: independent Sol-low re-review of exact source and full corrective diff, carrying both previous reviews and this response; parent integration verification follows. No self-approval.
