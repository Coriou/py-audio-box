# H3 r231 Sol — R230-1 CLI result publication fix

2026-10-24. Bounded implementation delivered for independent review. No self-adoption, runtime qualification or release GO.

## Pins and change

Fresh own worktree `/private/tmp/pab-r231-sol`, branch `fix/r231-cli-publication`, starts at current task root `1ff8d654b28be7f936460157b89e425180227341`. Only requested source `e0f9ecaa16b613ab3132e0a33b7c5956ba29eb5e` was cherry-picked, producing `45686fbfa7b8aa05a90556925489f8d146000195`. Fix source commit: `7dfffadf0ebcbca7b503073de3408129314631de`. Report committed separately. Source changes are exactly `service/package_install.py` and `tests/release/test_release_package.py`.

Read root r229 package report, r230 review and full normative r198, including final launch validation, no false coordinator success and retained failure obligations. Inspected authority exit's stable path/policy checks and descriptor cleanup. No applicable AGENTS.md found. Bounded memory registry query returned no relevant hit and supplied no premise. No skill or delegation used.

`main` computes its result under held authority, exits the complete context successfully, then prints one JSON record. Init/complete now wrap their result inside the context and reach the same post-context print; their in-context early return is removed. Final stability, policy and cleanup failures therefore reach the existing fixed redacted refusal with empty stdout. No exception handler, authority validation or retained-state cleanup policy changed.

## Verification

Two new unittest methods cover ten command/scenario combinations using real valid disposable configuration/scope/policy JSON files and the real CLI input parsing/dispatch. Python version observation, held authority, provisioning and operations are injected. Every `status/up/stop/init/complete` case checks computation while held, final exit refusal with empty stdout and exact redacted error, retained fixture bytes unchanged, and positive single JSON publication only after the successful exit event. Docker transport is forbidden by the fixture.

Meaningful RED on old cherry-picked implementation: 2 methods, 10 subtest failures, exit 1; all five commands emitted premature JSON on exit failure and all five positive event-order assertions failed. Log `/private/tmp/pab-r231-red.log`. GREEN package module: 30 tests PASS, 2.411 seconds, exit 0; log `/private/tmp/pab-r231-package-green.log`.

Final combined run: **212 tests PASS (210 existing + 2 new methods / 10 command-scenario subtests), 56.594 seconds, exit 0**, across the exact 13 modules below. Log `/private/tmp/pab-r231-final.log`. Existing intentional `injected_close_unknown` shielded-future and worker-refusal diagnostics appear in passing output; these are refusal/retention coverage, not clean-shutdown receipts.

Exact final command:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; p="/private/tmp/pab-r231-sol"; sys.path[:0]=[p,p+"/tests",p+"/tests/service"]; names=["tests.release.test_release_package","tests.release.test_headless_install","tests.service.test_installed_host_policy","tests.service.test_installed_registration","tests.service.test_installed_activation","tests.service.test_split_init","tests.service.test_install_child","tests.service.test_host_install","tests.worker.test_engine_lifecycle","tests.service.test_production_api","tests.service.test_production","tests.service.test_production_admission","tests.clients.test_production_clients"]; r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names)); sys.exit(not r.wasSuccessful())'
```

Both conftests inspected; isolated unittest avoids their pytest operational fixtures/guard. Interpreter is Python 3.14.7, not target Python 3.11. Changed Python files parse with Python 3.11 grammar; grammar evidence only. `git diff --check` passes.

Archive tests regenerate two deterministic standalone archives, byte-compare every manifest member with current source, extract and execute help and transitive imports outside checkout. Manifest is a path allowlist, not a frozen source-hash table, so no manifest update is necessary. Retained regenerated archive `/private/tmp/pab-r231-source.zip`, SHA256 `eb3277d51eac873203aec36e0fb7c2f1b16374e96cbd8771edbe8c2711f251e4`. Unchanged manifest SHA256 `3db3de9a16ae339f9c12ea3a0bc175ed387f6d9bc5c0dadb0b7ce63f46f0e0c3`.

Exact changed source SHA256:

- `service/package_install.py`: `8c8eff62eb6d6232ef34b7f1fb33084e422e4e566f967dde3d8c591539c1cd37`
- `tests/release/test_release_package.py`: `ebed1961e6663afbbd629d77c314f861b3c90338e567c742bdd791fc08ffe60f`

## Limits and handoff

No resolver/network, dependency installation, Docker build/run/exec/pull/stop, operational proc qualification, mount/UID transition, listener, model/ML/hardware, pytest guard/full suite or HELD operation. Existing Docker/HTTP/signal observations remain inert injections; ordinary test-owned file/SQLite/archive cleanup only. Source/archive tests do not qualify target Linux Python 3.11 images, real permissions/peers/process/mount authority, user journeys, audible CPU Kokoro, retained artifacts, descendant reap or restart behavior.

This report is the sole copy to root `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-LAUNCH-RESULT-FIX-r231-sol.md`; no root source adoption or root commit. Independent bounded review of R230-1 correction is next, followed by parent-owned adoption/actual qualification decisions.
