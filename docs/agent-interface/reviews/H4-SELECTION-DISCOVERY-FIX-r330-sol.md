# r330 Sol — truthful selection discovery implementation

2026-10-25. Implementation candidate only; **not an all-green final test receipt and not self-adopted**. Fresh independent Sol review, then parent-owned Astra judgment remain required. r329's orphan-measurement defect is addressed in source; the one authorized combined invocation ran 38 tests: 37 PASS and one new assertion FAIL, corrected afterward without rerun.

## Authority and exact pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-ec6c788d`, branch `t3/truthful-discovery`. Clean starting HEAD `a9f858332aeaeb0eef3cc0d99090c71373bac567`. Requested prerequisite `e6690c83885239310b6a58e731bdcd53656e0d3d` cherry-picked cleanly as `e254197cd55ae706385fb7912e6065774a2c2fce`. This report and the following fix are relative to that prerequisite. No applicable AGENTS.md found in worktree/ancestor discovery. Memory registry search found no relevant r328/r329 entry; current source and reviews determined the work.

Read full root r329 review, full r328 report, full r323 including Lane 2, PRODUCT.md, actual production_selection/discovery/choose, Qwen candidate construction/resolution, profiles/select, selected integration fixtures and both conftests. The old local runner named the previous candidate worktree; a separate explicit-path runner was written for this worktree.

## Change and scope

Discovery previously accepted any receipt value with the policy workload. It now joins configured candidate names to their own receipts, then joins the current configured profile snapshot and canonical hash of the complete execution mapping, and requires the applicable nonempty policy workload. Orphan keys, other profiles, stale hashes and wrong workloads cannot establish strict measured readiness.

QwenContexts exposes `selection_metadata()` from its already configured serialized templates only. It returns detached profile snapshots and execution hashes without invoking resolution, reading bundles/references, resolving owned media, importing ML or provisioning. Providers without this metadata seam yield actionable `measurement_binding_verification_required` under strict policy; known metadata without bound evidence yields `measurement_required`. Explicit allow_unmeasured policy retains its separate policy meaning, not a performance claim. Metadata readiness describes possible configured execution, not successful resolution of every owned voice/request.

Inspected the same-class measurement joins: production choose already uses candidate-key lookup and rejects changed whole-execution hashes; pure select already requires matching nonempty workload. Neither was changed. Qualification, installation, inventory freshness, owner/runtime readiness, memory and concurrency discovery checks remain. Selection, frozen replay and independent runtime authorization remain unchanged. No public contracts, generator or committed fixtures changed in this fix.

Four new authenticated ASGI regression tests cover orphan key versus submission with no allocation; profile-scoped evidence and configured-bound positive discovery; stale execution receipts versus refused submission; unknown metadata, wrong workload and explicit unmeasured policy. Discovery calls patch Qwen resolution to fail if invoked. Refusal helper now also checks HTTP 503, existing jobs unchanged and no reservations.

## Single executable receipt

Inspected pytest conftests; direct unittest bypassed their autouse machinery and listener/subprocess fixture. Inspected selected inert lifecycle doubles: in-process ASGI, fabricated TLS/connector/backend/process handles, tiny temporary sealed files/ZIPs and child function execution, not actual child launch or ML.

Exactly one invocation:

```
/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/pab-r330-finite-tests.py
```

Runner explicitly inserts this worktree plus tests/service and tests/clients. It loads the same nine names recorded in r329: test_production_selection, test_profiles, ProductionAPITests, ProductionClientTests, public Pocket preparation journey, installed Pocket journey, installed Pocket DiscoveryTests, native preset onboarding journey and native managed clone journey. The original 34 tests plus four new tests ran together once.

Observed CPython **3.14.7**, Clang 17.0.0, not 3.11. **Ran 38 tests in 29.428s; FAILED (failures=1), exit 1.** All original 34 passed; three new tests passed. The fourth reached correct positive discovery, correct other-profile not-ready discovery, and HTTP 503 on the other-profile submission, then failed because it expected `override_unavailable` while the resolver refused earlier with `candidate_identity_or_readiness_refused`. Changed only that expected reason after execution. Its later no-allocation assertions and positive submission tail were not reached in that invocation. Existing measured ranking/override tests independently passed positive submission. The final assertion correction is unexecuted; no retry or second selection was run under the once-only instruction. Parent/reviewer must obtain any newly authorized final green receipt; this report does not claim one.

Raw receipt `/private/tmp/pab-r330-sol-unittest-receipt.txt`, SHA256 `b14eb56aa5fec92a44f9090bb0d20cd62d377564e90708c9fe3f4ad73ae18172`.
Runner SHA256 `dcbe261d0b825587479879b2c0b4f016afa154dacf3ba030812cb1239d770753`.

Final source pins:

```
15584b88b4dfbb1fa24e5d6d6c1fa746927813ec87073327e29fdae7277d29c9  service/production_selection.py
32496405c8921591403e1d034d501644dcb03a15a7f0f9e8f63903d735ad045d  service/execution_context.py
188a63a506eb6b7a18af7214a511494a89307846699f50be0455f1b4b9d29558  tests/service/test_production_selection.py
cd3010036de67d7168d30e035f7d8733541bbd0cf9eac16863983eec978d72f1  contracts/generate.py
2a12772220231d50d869e74a43e8db10bbf974594a99af16e0ad175214f83e20  contracts/v1/schema.json
```

The generator filename/byte determinism and committed-fixture identity test passed, as did all generator request/exchange consumers, public Pocket/preset/clone and CLI/MCP regression selections. `git diff --check` passed.

## Limits and handoff

No root checkout writes, merge, push, delegation, self-adoption, network/listener, Docker, installation/download, actual child launch, ML, crash experiment, HELD access, STT or operational cleanup. Tiny inert fixture lifecycle teardown only. No heavy ACTUAL activity; no 6 GiB gate waiver.

This is SOURCE/inert evidence only. Installed composition remains the existing single configured Linux CPU context or separate enrolled native registry; no universal hardware auto-selection, multi-engine installed deployment, cross-host routing, hardware/performance/acoustic qualification or release approval. Trusted operator measurement metadata is not independently acquired performance evidence. Existing qualification/inventory/runtime and native lifetime/topology limits remain.

Metal remains **UNQUALIFIED**, no authorized arm64 wheelhouse. Legacy digest RED remains unchanged/unwaived: lib/tts.py SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`, versus recorded control `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`. Release remains NO-GO. Parent owns fresh independent Sol review, any authorized corrective test receipt, adoption/integration, then Astra milestone review. No subdelegation was performed.
