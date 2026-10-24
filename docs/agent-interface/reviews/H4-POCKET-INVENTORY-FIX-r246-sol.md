# H4 Pocket inventory correction — r246 Sol implementation

2026-10-24. **PARTIAL functional result: premature retained-namespace exception removed. No complete Pocket lifecycle/restart adoption claim.** R245-1's incompatible new producer row is no longer emitted. Empty as well as nonempty conditioning directories remain UNKNOWN/blocking. Parent independent new review is required; this report grants no source adoption, actual, installed or release GO.

## Exact pins

Owned root `/Users/ben/.txcode/worktrees/py-audio-box/t3-24816055`, branch `t3/pocket-namespace-reconciliation`; clean entry `55cda0e827f4175c30b993fb9d0dcc1d6f7584a8`, containing base `d803efb80451f0035d1e4c380f3fccac488e9709` plus root r245 review docs. In the brief, `803efb` resolves only as the intended `d803efb` spelling, not an independently resolvable Git revision.

Before edits, cherry-picked exact source `39a26b32b8b8d4100ad9245eb312a3b1e2184c3e` as `b80e0c5`, then exact report `191c92a7920884022e737b23c0b0d6ca1cecd5e4` as `c8329a4`. Correction source commit **`5a91e41fd952e86e0203de01b4989aade10c9476`**, two files only: `service/reconciliation.py` and `tests/service/test_pocket_lifecycle.py`. This report is a separate subsequent commit.

Historical candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-b646dab9` remains clean at `191c92a7920884022e737b23c0b0d6ca1cecd5e4`; no write there. Read full r244, r245 and r188 including assignment 3. No applicable AGENTS.md found in checked ancestors/tree. Bounded memory registry search returned no relevant hit and supplied no premise.

## Decision and behavior

Used the parent's explicit fallback: remove the premature exception rather than manufacture durable proof. Existing inventory identity contains stat fields and an optional content digest; a directory stat does not encode an observed empty child set. The r244 exception's live emptiness observation was not persisted as a closed typed fact, nor did the retained asset classification bind a directory to the exact Pocket attempt/context. A complete new trusted namespace attestation contract is not implemented here. This is a conservative scope decision, not a claim that a future typed contract is impossible.

Removed the entire special recognition block. Existing no-follow descriptor-relative traversal records conditioning as `unknown_foreign` / `unknown_namespace`, retains the directory and any opaque contents, and latches/blockades reconciliation. No conditioning descendant is opened or enumerated by the removed exception. No directory becomes an asset or synthetic transient. Actual child UID20000+ directories remain UNKNOWN; no descendant death, deletion permission or reap inference follows. Even same-UID empty Pocket restart is now explicitly incomplete.

Inspected all source consumers found by classification/reason/inventory/history searches: reconciliation row creation, persistence, prior-history latch, summaries/export; recovery inspection/projection, bounded `_validate_source`, diagnostic decisions/eligibility, B1/B2 preparation gates, detached documents/source and installed clean history; host-install clean-state checks; persistent-init/persistence column/schema adapters. Existing UNKNOWN is already represented in the inventory schema and bounded evidence format and refused by recovery/clean-history gates. No new enum/reason/schema/export/history acceptance is introduced. `service/recovery.py`, old source/history fixtures and all DDL remain unchanged. Historical r244 documents remain historical evidence, superseded by this partial result.

New focused test generates through the inert Pocket backend, carries the actual emitted row through both persisted inventory objects and inventory history, checks exact equality, then gives the persisted row to the real pure `_validate_diagnostic_eligibility` consumer and verifies `b2_unresolved_diagnostic`. It covers empty then nonempty retention and repeated blocked scans. No AST fragment executes. Typed-record mutation/acceptance tests are inapplicable because no retained namespace type is accepted. Existing closed-history mutation tests remain unchanged.

The existing injected Dispatcher success/cancel test now gives each scenario an independent database and staging root. This preserves success/cancel/reap coverage without pretending that reopening a root with retained conditioning succeeds. Initial named run exposed precisely that previously shared staging assumption (61 passed, one failed); corrected fixture isolation, then reran the named suite successfully. Existing reference/catalogue reopen coverage is distinct from a claim of healthy installed conditioning restart.

## Verification

Inspected `tests/conftest.py` and `tests/service/conftest.py`; used unittest imports only, no pytest collection, maintenance guard or listener fixture. Interpreter `/private/tmp/pab-0c-venv/bin/python -I -B` (the same source-test interpreter used by r244/r245). Each command used `PYTHONDONTWRITEBYTECODE=1`, cwd owned root, and explicit imports:

```python
import sys, unittest
sys.path[:0] = ["/Users/ben/.txcode/worktrees/py-audio-box/t3-24816055",
                "/Users/ben/.txcode/worktrees/py-audio-box/t3-24816055/tests/service"]
r = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(
    unittest.defaultTestLoader.loadTestsFromName(n) for n in names))
sys.exit(not r.wasSuccessful())
```

Named r244/r245 suite, `names` exactly:

```python
("tests.service.test_pocket_lifecycle", "tests.service.test_engine_admission",
 "tests.worker.test_engine_lifecycle", "tests.worker.test_engine_pocket",
 "tests.worker.test_engine_kokoro", "tests.service.test_production",
 "tests.service.test_production_admission", "tests.service.test_production_reconciliation",
 "tests.service.test_production_api", "tests.clients.test_production_clients")
```

**62 methods PASS, 14.165 s, exit 0** (original 61 plus new regression). Existing injected `lifecycle supervisor unavailable; lane quarantined` diagnostic printed with overall OK. After adding the real recovery-consumer assertion to the new method, reran that exact focused method separately: **1 PASS, 0.545 s, exit 0**. No production code changed after the complete named run.

Additional existing source unittest modules, `names` exactly `("tests.service.test_installed_restart", "tests.service.test_installed_registration", "tests.service.test_installed_host_policy")`: **87 methods PASS, 62.590 s, exit 0**. Includes unchanged actual closed detached-history reader regressions, synthetic/terminal history and mutated historical evidence; host, UID and transport observations are injected, not operational installation. `tests/service/test_recovery.py` is pytest-function based and was inspected, not collected or claimed as executed.

`git diff --check` PASS. Disk observation `/private/tmp`: 3,841,888 KiB available, below 6 GiB actual gate. Only small inert owned fixtures and ordinary temporary-fixture teardown. No delegation, Docker, network/download, model loading/inference, integration install, listener, process crash, operational cleanup or HELD access.

## Parent handoff

Review this separate correction independently. It removes the producer/reader incompatibility by declining the premature retained classification; it does **not** deliver the preferred typed retained-namespace roundtrip. Pocket lifecycle/restart remains partial and blocked on retained conditioning, including actual cross-UID restart. No broader lifecycle/recovery redesign, installer/provisioning or cleanup was attempted. Actual clone/listening, resource/containment, mounted-reference and cancel/reap qualification remain separate activation gates.
