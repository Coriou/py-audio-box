# H3 r222 Sol — independent round 3 restart source-adoption review

2026-10-24. **GO for bounded source adoption of the exact candidate below.** R220-1/2/3 are closed in the inspected correction; no remaining blocker was demonstrated in this review. This is retained-claim/source validation, not actual Linux, installation, activation, speech or release approval. Parent owns adoption and the already outstanding operational qualification; no new broad gate is introduced.

## Exact pins and scope

Candidate `/private/tmp/pab-r221-sol`, branch `fix/r221-history-coherence`, clean before and after verification. Independently verified the direct-parent chain:

- Base `0c831e1e16a07638e3685ac0022af91a168c9992`.
- Carry `7c4944e71aeb5f02d34be331ff1ca431f1f16d73`, original `fbd0aff770213024aff27c344ba0a2fa8db65d18`; identical stable patch ID `d1473dddbb6a32673fbe22661ee462b47be40402`.
- Carry `27508c37ac065a5e030a9ac32e44925800d4664d`, original `33fc3097e111700417c950017eb834d71b2bfe8f`; identical stable patch ID `c1dee030be7d788ecf576aa866a284df41144962`.
- Correction source/tests `51133296a974e4ac198e967a61120da0ee99effc`.
- Documentation HEAD `fbd63fc02787a86d04e4fdf56f7aa46ec46c81e6`.

Correction changes only `service/recovery.py` and `tests/service/test_installed_restart.py`; source-to-HEAD adds only r221 documentation. `git diff --check 0c831e1 HEAD` passes. Read full normative r198, r218, r219, r220 NO-GO and r221; their candidate copies are byte-identical to root. Inspected the contract oracle, canonical schema initializers, execution-context validator, detached/current snapshot and allocation joins, journal/reference extraction, graph and retry/handoff paths, and test guard/conftest. No applicable AGENTS.md was found. Memory-registry search returned no relevant premise.

No delegation, source edit, commit, adoption or unrelated cleanup. Sole persistent write is this report. Existing unrelated root STT evidence was left untouched.

## R220-1 — closed: complete documents before reference extraction

`service/recovery.py:2948–2976` calls the existing full contract oracle for job, project, asset, artifact, publication manifest/message, worker plan, bundle entry, dispatch authority job/profile and outcome documents. Contexts have a closed outer shape, require a bundle list, validate each nested plan/bundle and call the existing private execution validator. The unchanged typed phase additionally enforces its execution-required/context rules and catalogue/request semantics. Job row ID/principal/operation/key are bound to their document at lines 2966–2970.

The full document pass runs before the shared reference walker (`3015–3024`). Thus the walker no longer stands in for validation of required nested fields. The new retry/handoff test at `tests/service/test_installed_restart.py:299` covers each prior missing operation/principal/stage example plus required attempt and plan/request omissions. Its shared refusal helper (`258–277`) forbids the writer, asserts no notification, and compares the complete fixture SQL dump and registration rows. These tests pass independently.

## R220-2/3 — closed: canonical schema identity and all-row constraints

`service/recovery.py:2868–2892` captures 23 ordinary table declarations. Independently parsed string literals from the actual Core, persistence, ResourceStore and DispatchStore initializers and compared every captured declaration by the same token identity, normalizing only initializer `IF NOT EXISTS`: **all 23 match**. This covers primary/composite keys, UNIQUE, NOT NULL, CHECK, REFERENCES, AUTOINCREMENT and dispatch defaults. Catalogue table/trigger declarations and optional B2 declarations are imported from their existing owners rather than recopied.

The complete retained sqlite_master multiset is compared against the reference catalogue (`2895–2920`), including type/name/table/token DDL and implicit indexes. It does not merely compare table names/columns or accept an arbitrary statement that happens to execute. Missing/extra/duplicate/misnamed objects, changed constraints and fabricated SELECT/index entries refuse. Recognized immutable catalogue trigger declarations are compared but never installed.

Every ordinary retained row is loaded with bound parameters into canonical in-memory tables before maps (`2929–2938`). SQL identity constraints apply before duplicate collapse; serialized readback equality detects affinity coercion, and post-load foreign_key_check covers declared joins independent of load order. sqlite_sequence is separately checked for unique known names, integer/nonnegative values, populated jobs/events coverage and high-water bounds (`2924–2943`). The private/resource versions and absence of maintenance are checked at `2991–2993`, catalogue singleton revision at `2995`, installation revision/identity at `2996–3002`, and integrity revision by the unchanged allocation checker. Core user_version is checked on current capture (`781`); the detached projection carries the fixed supported version summary and exact canonical schema, not a separate historical PRAGMA receipt.

The new duplicate-row, independent job-key/foreign-key, and schema identity tests (`314–364`) pass. Full source table closure remains in `_validate_source`; partial B2 table sets cannot evade that closure, and populated B2 maintenance is refused. Existing journal Authority/boot/runtime/dispatch, registration/enrollment, typed owner/reference, reservation/allocation, publication and summary joins remain in the detached path. The foreign-key pass complements those application joins rather than replacing them.

No retained SQL is executed. Only trusted declarations and internally constructed fixed-table INSERT/SELECT statements reach SQLite. The connection is explicitly `:memory:`, trusted_schema is off, the authorizer denies functions/attach/view/virtual-table/trigger operations, and a progress deadline remains (`2901–2915`). The retained SQL is tokenized only for equality. This review establishes validity checks for malformed retained claims, not a signature/authenticity scheme for a hostile database or independent observation of historical files/processes.

The captured DDL is an explicit supported-v1 reference. Future initializer/storage-format changes must update that reference deliberately; current declaration equality and passing real fixture captures establish this pin, not perpetual drift immunity. No current mismatch was found.

## Prior closures preserved

Independently verified complete byte equality against exact second carry for `service/host_install.py`, `service/reconciliation.py`, and `worker/runtime_journal.py`. Therefore the R218 graph correction, reference walker/scanner extraction and journal loader extraction have no new delta. AST equality also holds for recovery `_validate_snapshot_journal`, `_allocation_coherent` and legacy `host_registration_union`.

Reinspected the bounded iterative graph at `service/host_install.py:1088–1144`: complete digest-keyed predecessor closure, cycle/branch/future/replay refusal, monotonic incarnation/boot and process-reuse checks, retained schema/journal-prefix binding remain. Passing tests preserve two-/three-node cycles, future/replay negatives, exact retry bytes and positive three-incarnation immutable retry/handoff. No new writer, reset, acknowledgement, reservation release, reminting or historical repair path was introduced.

## Independent checks and limits

Exact candidate eight-module run: **179 tests PASS, exit 0, 83.395 seconds**, Python **3.14.7**. Used `/private/tmp/pab-0c-venv/bin/python -I -B`, explicit candidate and candidate/tests import paths, and only these modules:

`tests.service.test_installed_restart`, `tests.service.test_installed_registration`, `tests.release.test_headless_install`, `tests.service.test_installed_host_policy`, `tests.service.test_host_install`, `tests.service.test_install_ownership`, `tests.service.test_split_init`, `tests.service.test_install_child`.

Additional independent isolated inline probes used the real `HistoricalValidation` closed fixture and production retained readers. Separately changed only disposable retained provenance, recomputed its source digest, then checked full retry and handoff: missing dispatch_settings schema object; integer-to-text accounting affinity claim; jobs sequence below high-water; duplicate sequence row; fabricated function SELECT; ATTACH; and trigger SQL. **All seven refused with no writer, no notification and unchanged complete fixture database.** No probe source file was written. Only ordinary test-owned temporary cleanup occurred.

Runs yielded asynchronously at one second and used bounded polling; no blocking wait exceeded 60 seconds. No pytest guard/full suite, actual Docker/proc/mount/UID transition, network/listener, ML/model/hardware, HELD operation, production-evidence mutation, package/build or activation operation occurred. Host/hello/process/UID/reap evidence in fixtures remains injected; passing source tests do not independently prove actual descendant death. Python 3.14 source checks do not qualify Python 3.11 packaging.

Parent may adopt this exact bounded source correction. Actual supported Linux permission/peer/mount/process/restart behavior, Python 3.11 packaging, packaged activation, audible speech/user journeys and release acceptance remain pending under the existing plan.
