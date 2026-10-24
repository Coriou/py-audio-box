# H3 r221 Sol — retained restart coherence correction

2026-10-24. Implementation candidate for independent review. Addresses exactly R220-1/2/3; no self-adoption or operational GO. Parent owns review and adoption.

## Pins and scope

Fresh own worktree `/private/tmp/pab-r221-sol`, branch `fix/r221-history-coherence`, from current root `/Users/ben/Projects/py-audio-box` at `0c831e1e16a07638e3685ac0022af91a168c9992`. Supplied thread checkout was `15c39686acdbdab38d0942ec358311286a585955`; it was not edited. Cherry-picked only the two requested original source commits, in order:

- Original `fbd0aff770213024aff27c344ba0a2fa8db65d18` -> carry `7c4944e71aeb5f02d34be331ff1ca431f1f16d73`, direct parent root base. Both stable patch IDs `d1473dddbb6a32673fbe22661ee462b47be40402`.
- Original `33fc3097e111700417c950017eb834d71b2bfe8f` -> carry `27508c37ac065a5e030a9ac32e44925800d4664d`, direct parent first carry. Both stable patch IDs `c1dee030be7d788ecf576aa866a284df41144962`.
- Final correction source/tests **`51133296a974e4ac198e967a61120da0ee99effc`**, direct parent second carry.

Read complete root r198, r218, r219 and r220 reports and both conftests; no release conftest or applicable AGENTS.md found. Bounded memory-registry search supplied no relevant premise. No delegation. Correction changes only `service/recovery.py` and `tests/service/test_installed_restart.py`. No additional dependency authorization was needed. This report is a separate documentation commit; root receives only an identical report copy, without source adoption or a root commit.

## R220-1: full contract validation at the detached boundary

Before legacy reference enumeration, detached documents now use the existing complete `contracts.validate.validate` oracle for jobs, projects, assets, artifacts, publication manifests/messages, worker plans, bundle entries, dispatch authority jobs/profiles and outcomes. Nested required properties are therefore checked by the actual closed contracts, rather than inferred from the reference walker. Private contexts retain their closed shape and use the existing execution validator. Job row ID/principal/operation/key must match the validated job document. Existing catalogue/request validation remains shared through the unchanged typed phase.

No parallel contract parser or change to legacy extraction semantics was introduced. `service/reconciliation.py`, `worker/runtime_journal.py` and the entire `service/host_install.py` are byte-identical to second carry. The R218 graph correction is thus unchanged. `host_registration_union` is source-span identical as well.

## R220-2/3: canonical schema and row coherence before maps

Retained SQL is no longer executed. A bounded in-memory reference is created solely from supported canonical v1 table DDL (captured from the existing Core/persistence/ResourceStore/DispatchStore initializers), plus existing shared catalogue and optional B2 declarations. The complete retained sqlite_master multiset must match its type/name/table/normalized token DDL identity, including implicit constraint indexes. Unknown objects, non-DDL, changed constraints, false names/table claims, omitted objects and duplicates refuse. Known immutable catalogue triggers are compared with their shared declarations, never installed or executed.

Every retained ordinary table row is inserted with bound parameters into those canonical tables before any detached dictionary/map construction. Canonical primary keys, unique constraints, NOT NULL and CHECK constraints apply across the complete retained inventory. Post-insert values must match the claims without affinity coercion. `foreign_key_check` validates all declared cross-table joins after loading, independent of insertion order. Existing typed/journal/allocation joins remain in place. The unconstrained sqlite_sequence table receives explicit unique known-name, integer/nonnegative, coverage and high-water checks; catalogue format remains exactly revision 1.

SQLite remains `:memory:` only, trusted-schema off, with a one-second progress deadline and restricted authorizer. No functions, ATTACH, views, virtual tables or retained triggers are authorized. No retained SQL, arbitrary external path, historical filesystem root, migration, writer or proof framework is introduced. Existing byte/row/provenance budgets remain.

## Verification

Meaningful RED: the three new full retry/handoff regression methods ran against exact second-carry implementation before the fix. **3 tests, 14 expected assertion failures, exit 1, 16.998 seconds.** Accepted cases reached the writer guard: missing job operation/principal/stage; duplicate dispatch settings/events/jobs/accounting/sequence/catalogue format; and invented SELECT/index, mismatched name/table/constraint DDL and implicit-index claims. Log `/private/tmp/pab-r221-red.log`, SHA-256 `ef6e73a9cad0a003610fff7992083f646891d40ec301eec57bc982acbb7b9737`.

Four new test methods add representative nested required-field omissions, duplicate populated-table rows, independent jobs seq/id/principal-operation-key constraints, foreign event/command/retry joins, mismatched schema identity and non-DDL. Full closed retry and handoff negatives assert no writer, no notification and unchanged complete fixture database SQL dump, in addition to retained registration rows. Existing positive multi-restart tests retain immutable provenance and exercise exact retry/handoff. Existing R218 cycle/future/replay tests remain unchanged.

Intermediate verification found and corrected an empty command-table assumption in the new probe, then a test-only dump connection with delayed close causing fixture WAL disappearance; the connection now closes explicitly. Earlier implementation checks also corrected a denied aggregate-function query and the missing catalogue singleton check. These failures were not waived. The final run below supersedes them.

Final pinned run at source `51133296a974e4ac198e967a61120da0ee99effc`: **179 tests PASS (175 existing + 4 new), exit 0, 78.940 seconds**. Log `/private/tmp/pab-r221-final-exact.log`, SHA-256 `6f62bbc84d85717b552f0a68c26f45512f1fbbb6191adedec53e94d7b3d64b55`.

Exact isolated eight-module command, from the own checkout (no pytest fixture activation):

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/private/tmp/pab-r221-sol","/private/tmp/pab-r221-sol/tests"];names=["tests.service.test_installed_restart","tests.service.test_installed_registration","tests.release.test_headless_install","tests.service.test_installed_host_policy","tests.service.test_host_install","tests.service.test_install_ownership","tests.service.test_split_init","tests.service.test_install_child"];r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(names));sys.exit(not r.wasSuccessful())'
```

Tool yields were one second with bounded asynchronous polling, never a blocking wait over 60 seconds. Python 3.14.7. Stable patch-chain verification, unchanged graph/walker/journal/legacy-union comparisons and `git diff --check` passed. Only ordinary test-owned temporary cleanup occurred; scratch generation helpers were removed, and verification logs retained outside root.

## Evidence limits and handoff

Source/disposable-fixture evidence only. Host/UID/hello/process/PID1 reap claims remain injected; retained claims are validated without independent actual-death proof or old-filesystem reobservation. No actual Docker/proc/mount/UID transition, network/listener, ML/model/hardware, pytest guard/full suite, HELD operation, production-evidence mutation, package/build or installed activation operation. Python 3.14 tests do not qualify Python 3.11 packaging.

Independent review of the exact source pin remains parent-owned before adoption. Supported Linux permission/peer/mount/process/restart qualification, packaged launch/activation, audible speech/user journeys and release acceptance remain outstanding. No reset, compaction, reminting, acknowledgement manufacture, reservation release or historical provenance repair was added.
