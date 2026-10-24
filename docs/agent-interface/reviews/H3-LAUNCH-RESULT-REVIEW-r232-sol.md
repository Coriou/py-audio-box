# H3 r232 Sol — independent round-two package/CLI source review

2026-10-24. **GO for bounded package/CLI source adoption. R230-1 is closed; no new blocking finding established.** This is source/injected-test acceptance only. Actual Linux Python 3.11 image, installed authority, listener/user-journey and speech qualification remain the existing parent-owned work; no new broad gate is proposed.

## Exact candidate and carry

Reviewed clean `/private/tmp/pab-r231-sol`, branch `fix/r231-cli-publication`, HEAD `493a56e3d29904ff65717a3f4ce100b063afa9ea`. Verified direct chain:

- Base `1ff8d654b28be7f936460157b89e425180227341`.
- Carry `45686fbfa7b8aa05a90556925489f8d146000195`, original `e0f9ecaa16b613ab3132e0a33b7c5956ba29eb5e`.
- Fix source `7dfffadf0ebcbca7b503073de3408129314631de`.
- Separate report `493a56e3d29904ff65717a3f4ce100b063afa9ea`.

Original and carry stable patch IDs both equal `a1c4e8f26b05d0a39ecc56286cf7f1898adb2d61`. Direct original-versus-carry comparison of all 13 package files is empty. Carry-to-fix changes exactly `service/package_install.py` and `tests/release/test_release_package.py`; documentation HEAD adds only r231's report. `git diff --check base..HEAD` passes. No source adoption or commit performed.

Read normative r198, r229 package report, root r230 NO-GO and r231 fix report. Reviewed the complete CLI, coordinator result paths, authority final checks/descriptor cleanup, regression/archive tests and release image/context definitions. No applicable repository/document-directory AGENTS.md found. A bounded memory-registry query supplied no relevant premise. No skill or delegation used.

## Closure and failure-path assessment

`service/package_install.py:260-272` now computes all five operational results under authority, leaves the complete context, then evaluates `json.dumps` and prints. Init/complete wrap their result at line 267 and no longer print/return inside the context. Up, stop and status share the same sole publication at line 272. Thus ready, stopped, initialized and retained-status results all cross the same final barrier.

`service/host_install.py:1030-1044` still performs final lock/path/policy stability checks and descriptor cleanup before context exit succeeds. Any ordinary operation, final validation or close exception unwinds past publication into the unchanged fixed refusal at `package_install.py:273-275`; KeyboardInterrupt is also redacted. An exception in serialization occurs before `print` receives a string, so emits no JSON. Actual results are bounded primitive dictionaries from initialization or coordinator paths, rather than lazy objects using authority after exit. Output-device errors can occur after a partial write, as with ordinary CLI output; this does not reinstate publication before mandatory authority checks and is not an exit-code success claim. Parser help/errors and the standalone archive digest path remain separate from operational coordinator success. Archive closes its ZIP and reads its digest before printing (`207-237`).

The change adds no rollback, deletion, registration writer, authority relaxation or retained-state cleanup. Existing creation intent/per-role IDs/pair remain retained (`141-174`); stop still requires ready state, exact control TERM/exit, clean-history verification, exact worker TERM/exit and final preflight (`193-204`). Ready corroboration and incarnation binding remain unchanged (`116-128`). The fixed refusal exposes no backend exception text. Exceptional unreaped-child handling in the authority still retains duplicated descriptors (`1031-1033`); moving publication does not waive that obligation. Primary-error replacement by a failing final check can still occur in the existing context implementation, but both paths refuse without success output; this patch does not change its diagnostics or retention policy.

`tests/release/test_release_package.py:174-241` exercises real CLI parsing and disposable valid public inputs, injected operations/authority and forbidden Docker transport. Two methods cover ten command/scenario subtests: five exit failures with empty stdout/exact redacted refusal and retained bytes unchanged, and five positives publishing one JSON line only after successful exit. The retained sentinel is fixture evidence, not proof of actual host retention. Independent in-memory replay against carried pre-fix source reproduced **2 methods / 10 failures / 0 errors**. An initial replay used a separately compiled Launcher class outside the test patches (7 failures/3 errors); corrected class binding produced the meaningful ten failures. No source was edited for replay.

## Full package assessment carried forward

The r230 assessment of the complete 13-file package remains applicable: exact final container IDs; local images and `--pull=never`; waiting control then worker/hello/register/notify/readiness; closed policy/topology/credential isolation; retained failure and clean-restart obligations; authenticated installed factory and lifecycle closure; standalone archive; pinned CPU image/context/locks; and unchanged legacy metadata. The only source delta since that assessed package is the publication correction and tests above.

Independent passing archive coverage (`test_release_package.py:244-282`) rebuilds twice deterministically, compares every manifest member byte-for-byte with current source, extracts outside checkout, executes isolated help and imports transitive modules from the extraction. The 93-file manifest plus generated root command includes source/data/locks; the package remains source-only, requiring a separately provisioned trusted interpreter and offline dependencies. Release targets retain the pinned Python 3.11 CPU base and combined service/Kokoro worker locks, with excluded-by-default context. Existing r230 lock/version/hash and metadata assessment is carried forward through byte-identical files, not represented as a fresh resolver or installation receipt.

## Independent verification and limits

Both conftests were inspected. Ran only the authorized isolated unittest modules with explicit paths:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; p="/private/tmp/pab-r231-sol"; sys.path[:0]=[p,p+"/tests",p+"/tests/service"]; names=["tests.release.test_release_package","tests.release.test_headless_install","tests.service.test_installed_host_policy","tests.service.test_installed_registration","tests.service.test_installed_activation","tests.service.test_split_init","tests.service.test_install_child","tests.service.test_host_install","tests.worker.test_engine_lifecycle","tests.service.test_production_api","tests.service.test_production","tests.service.test_production_admission","tests.clients.test_production_clients"]; r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names)); sys.exit(not r.wasSuccessful())'
```

**212 tests PASS, 55.850 seconds, exit 0.** Interpreter is Python 3.14.7. Both changed Python files also parse with Python 3.11 grammar; neither fact proves Python 3.11 execution. Passing output includes intentional `injected_close_unknown` shielded-future and worker quarantine diagnostics; these are negative lifecycle coverage, not successful shutdown receipts. Candidate remained clean after checks.

No resolver/network/install, Docker build/run/pull/exec/stop, operational proc qualification, privileged mount/UID transition, listener, model/ML/hardware, pytest guard/full suite, HELD operation, delegation or unrelated cleanup. Only disposable test-owned files/SQLite/archives and injected observations were exercised. A broad parent-directory AGENTS search encountered an unrelated permission-denied entry; no action was taken there.

Parent may adopt the reviewed source unit. Existing actual qualification remains pending: trusted Linux Python 3.11 dependencies/images, role/setgid/peer/mount/process authority, authenticated HTTP/CLI/MCP journeys, audible CPU Kokoro, retained voice/artifact reuse, cancellation with confirmed descendant reap and clean/failed restart. This review grants no runtime/release GO. Sole persistent write is this root r232 report.
