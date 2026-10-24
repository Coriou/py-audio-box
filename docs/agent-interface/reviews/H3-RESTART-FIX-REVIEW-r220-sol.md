# H3 r220 Sol — independent round 2 restart source-adoption review

2026-10-24. **NO-GO for source adoption at this pin.** R218-2 has a substantive bounded graph correction, and the original R218-1 examples now refuse. However, R218-1 is not fully closed: malformed nested job documents, impossible duplicate source rows, and fabricated schema entries still pass the complete registration retry and handoff. These are retained-payload validation defects within the existing requested correction, not a new design or operational gate.

## Exact candidate and scope

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-7733b4d1`, branch `t3/restart-history-validation`, clean tracked/untracked status before and after checks. Verified linear parent chain:

- Base `2ba7f1276a41bb29596048be57e27867042d7116`.
- Carry `492c94cd92ff9e7f287de858081d0e31315c3f95`, direct parent base.
- Fix source/tests `33fc3097e111700417c950017eb834d71b2bfe8f`, direct parent carry.
- Documentation HEAD `037c32371f5d8783005fc8925e0ea080fea58f60`, direct parent source.

Original `fbd0aff770213024aff27c344ba0a2fa8db65d18` and carry have identical stable patch ID `d1473dddbb6a32673fbe22661ee462b47be40402`. Carry changes the five advertised files. Fix changes only `service/host_install.py`, `service/recovery.py`, `service/reconciliation.py`, `worker/runtime_journal.py`, and `tests/service/test_installed_restart.py`. Source-to-HEAD changes only r219 documentation. `git diff --check 2ba7f12 HEAD` passes. `070ae98419bc1f4a4fd48c08e32076ccc805280b` is an ancestor of candidate HEAD (exit 0), corroborating the correction to r217.

Read root normative r198, r217, r218 and r219; inspected changed source, typed reference/scanner phases, journal/Authority/allocation dependencies and both test conftests. No applicable AGENTS.md found. Bounded memory-registry search supplied no relevant premise. No delegation, source edit, commit, adoption or unrelated cleanup. Sole persistent write is this root report.

## R220-1 — blocking: shared reference extraction is not complete nested validation

At candidate `service/recovery.py:2936–2945`, detached source acceptance relies on `typed_references` producing no diagnostics. The extracted walker at `service/reconciliation.py:454–473` checks object type, unknown supplied keys and supplied property values, but never validates required properties of an ordinary object. Its full-node validation at line 457 applies only to scalars. Thus AST equivalence to the old reference phase does not establish complete contract validation.

Independent inert reproduction used the actual closed fixture (`HistoricalValidation.setUp`, then `history_fixture()`, with terminal retained job), with production enrollment/provenance validators enabled. Starting separately from the valid original second-registration provenance, parse `history.state.evidence_projection.data.jobs[0][6]`, delete **one** of `operation`, `principal_id`, or `stage`, re-encode that job, recompute `state.source_digest`, and write only that disposable provenance file. Each field is required by `contracts/v1/schema.json`'s job definition. No job-summary update is necessary because these fields are omitted from the summary projection.

For each independent mutation, `case.f.run_registration()` returned the original successful registration ID, and `_installed_handoff(..., notify=collector)` delivered exactly one notification. Observed: `missing operation ACCEPTED True 1`, `missing principal_id ACCEPTED True 1`, `missing stage ACCEPTED True 1`. The current DB and real fixture journal remained unchanged.

Required correction: enforce complete applicable contract/document validation at the detached boundary, including required nested fields, while preserving the authorized pure-extraction equivalence of legacy/current behavior. A reference enumerator is not a complete schema validator. Add full retry/handoff negatives for these missing-required examples and representative nested required-field omissions, with no writer/notification on refusal.

## R220-2 — blocking: detached rows need SQLite identity/constraint coherence

`service/recovery.py:1345–1353` validates source row widths/scalars and budgets. The in-memory schema pass at `2883–2912` does not insert or otherwise validate the retained data against row identity constraints. `2937–2943` converts rows to dictionaries, and the shared typed phase plus `_allocation_coherent` subsequently builds maps keyed by job ID, silently collapsing duplicates. Summary rederivation does not prevent this when both summaries and source rows contain the same duplicate.

Independent full retry/handoff reproduction: start from the same valid original provenance, append an exact copy of `source.data.jobs[0]`, append a copy of `state.jobs[0]`, append `state.affected.jobs[0]` to that list, recompute `state.source_digest`, and write only disposable provenance. Result: `duplicate_jobs ACCEPTED retry= True handoff= 1`.

This cannot be a valid snapshot of its own retained jobs schema: `service/core.py:240–243` declares both `seq INTEGER PRIMARY KEY` and `id TEXT UNIQUE NOT NULL`, as well as the principal/operation/key uniqueness constraint. It is not merely a different historical filesystem observation. Required correction: validate retained row identities/constraints and relevant cross-row joins without executing retained triggers or opening historical roots. Add this duplicate-job case through full retry/handoff; do not rely on dictionary construction to establish uniqueness.

## R220-3 — blocking: executed SQL is not bound to its declared schema entry

At `service/recovery.py:2893–2912`, each row's declared type/name/table is checked superficially, but arbitrary authorizer-permitted SQL is executed without checking that it creates the declared object. Afterwards only table coverage and column names are compared; exact catalogue identity covers catalogue objects only (`2913–2917`).

Independent reproduction appends `['index','invented','jobs','SELECT 1']` to the otherwise valid historical schema, recomputes only `state.source_digest`, and writes the disposable provenance. Result: `schema_select ACCEPTED retry= True handoff= 1`. This is not a possible sqlite_master entry for an index. The allowed SELECT leaves all existing table/column checks satisfied while the fabricated schema claim is accepted.

Required correction: bind every declared schema row to the actual admitted schema object/DDL identity and reject non-DDL or mismatched statements. Preserve the in-memory/no-external-effects boundary. This is closure of the requested malformed-schema validation, not a request for a schema migration or broad redesign.

## Corroborated corrections and extraction checks

Independently verified, using AST comparison against exact carry bytes:

- Journal validation block is AST-identical after substituting `self.value`/`self.maintenance` with parameters; `_load` delegates to it. Other journal methods/writers are unchanged.
- Extracted snapshot journal loop is AST-identical to the carry's original loop.
- Typed reference phase is AST-identical after moving contract-document loading to the caller and passing it explicitly.
- Scanner filesystem remainder is byte-identical, excluding trailing whitespace at the extraction boundary.
- Legacy `host_registration_union` source span is exactly identical.

The original unknown-record, invalid allocation/foreign boot, false schema/pending maintenance examples now have full retry/handoff refusal coverage. Shared journal validation and snapshot Authority/registration/dispatch joins are called for detached journals. Enrollment is bound back to exact current public enrollment. Summary rederivation materially improves fault/monitor/integrity/inventory/registration/version coverage; the findings above explain why the complete-validation claim remains too strong.

The SQLite connection is explicitly `:memory:` with a progress deadline and restrictive authorizer. Additional inert retained-SQL probes for `ATTACH DATABASE ':memory:'`, `SELECT sqlite_version()`, and an injected CREATE TRIGGER all refused with `ValueError installed_history_schema`. Inspection confirms retained recognized trigger text is compared and skipped, not executed; function, attach, view and virtual-table authorizer actions are not allowed. No external path or application database was used by these probes. R220-3 is false-schema acceptance, not demonstrated external SQL execution.

Graph inspection at `service/host_install.py:1088–1144` confirms bounded iterative topological elimination, complete pre-insert closure, branch/cycle refusal, exact digest-keyed retained registrations, monotonic incarnation times, lane/role/image/volume checks, changed-incarnation/worker-boot correspondence, process start-tick/namespace reuse checks and predecessor journal-prefix/schema checks. It runs for full reads and before prospective registration insertion (`1238–1240`). Focused tests exercise two-/three-node cycles, acyclic future prestate/replay and positive three-incarnation immutable retry/handoff. No remaining R218-2 blocker was demonstrated in this bounded review. Exact old retry bytes and legacy behavior remain preserved by the inspected paths and passing focused tests.

## Independent verification and limits

Exact candidate eight-module unittest run: **175 tests PASS, exit 0, 65.643 seconds**. Used `/private/tmp/pab-0c-venv/bin/python -I -B`, explicit candidate and candidate/tests import paths, and these modules: `tests.service.test_installed_restart`, `tests.service.test_installed_registration`, `tests.release.test_headless_install`, `tests.service.test_installed_host_policy`, `tests.service.test_host_install`, `tests.service.test_install_ownership`, `tests.service.test_split_init`, `tests.service.test_install_child`. Tool yield was 1 second with bounded polling; no blocking wait over 60 seconds. Additional reproductions ran as isolated `-I -B` inline unittest-fixture scripts and left candidate clean. No reproduction source file was written.

Source/disposable-fixture evidence only. Injected host/hello/UID/process/reap claims remain injected; no historical filesystem reobservation or independently proven actual death. No actual Docker/proc/mount/UID transition, runtime construction, network/listener, ML/model/hardware, pytest guard/full suite, HELD operation, production-evidence mutation, package/build or activation operation. Ordinary test-owned fixture cleanup only. Python 3.14 source tests do not establish Python 3.11 packaging qualification.

Parent owns remediation and a further independent review before source adoption. Package/activation, actual supported Linux behavior, speech/user journeys and release acceptance remain pending. Do not reinterpret passing 175 tests or retained hashes as adoption or actual-runtime approval.
