# H3 r230 Sol — independent first-round release-package source review

2026-10-24. **NO-GO for bounded source adoption: R230-1 remains open.** One reproduced command-result publication defect; no source correction or adoption performed. The 210 authorized tests pass. This decision does not add a runtime gate: actual Linux Python 3.11 images and audible CPU Kokoro qualification remain the existing parent-owned acceptance work.

## Pins and review boundary

Clean candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-9941da0e`, branch `t3/headless-launch-package`. Verified direct-parent chain: base `acb2d37b53d8fd96d68557a3916308866ba9fb75`, source `e0f9ecaa16b613ab3132e0a33b7c5956ba29eb5e`, documentation HEAD `3b3680e7199c2d31bbc0930d74f741f4c55b2c2e`. All **13 r229 file SHA256 pins match**, with zero mismatches. Manifest contains 93 distinct files; the generated root command is additional. Documentation HEAD adds the r229 report separately.

Read root normative r198, including six launch steps, Lot B, restart/failure rules; r213 partial checkpoint, r228 activation review, r229 report and current installation instructions. Inspected package coordinator, protected authority/history changes, entrypoint/signal/readiness additions, worker command and model mapping, split-init callback ordering, release Dockerfile/context/Compose, manifest and dependency/legacy metadata. No applicable AGENTS.md found. Bounded memory-registry search returned no relevant hit and supplied no premise. No skill or delegation used.

## R230-1 — publish success only after final authority validation (P2, adoption blocking)

`service/package_install.py:260-269` prints the operational result **inside** `installed_launch_authority`. Init/complete additionally return from inside that context at line 267. Context exit still performs mandatory authority/path/policy stability checks at `service/host_install.py:1034-1044`, any of which can fail. Consequently a failed final authority bracket emits a success-shaped JSON record before the command exits with refusal. This affects `ready`, `stopped`, and `initialized` publication, not just the instantaneous status payload used in the repro.

Independent inert repro called the real `main(['status', '--configuration', ..., '--scope', ...])` with real disposable valid public fixture JSON files, the Python-version observation patched to `(3,11)`, `Launcher.status` returning a fixed ready result, and an injected held-authority context yielding valid policy then raising `ValueError('installed_host_policy_changed')` on exit. No Docker, listener, privilege transition or host contents were accessed. Observed:

```text
stdout='{"phase": "ready", "registration_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n'
SystemExit: speech-box refused; retained state requires review
```

The process does refuse and does not exit zero; this is **not** a demonstrated exit-code bypass, leaked secret, or actual runtime incident. It is contradictory success output before a required final check, contrary to the requested no-false-coordinator-success boundary. A human or streaming JSON consumer can receive `ready` while the operation's own final authority validation has not succeeded. The passing tests do not exercise CLI result publication against context-exit failure.

Required bounded correction: calculate and retain the result while authority is held, successfully leave the authority context, then print the single result. Apply the same structure to init/complete, without their early in-context print/return. Preserve fixed redacted failure output and retained obligations. Add an inert CLI regression for authority-exit failure asserting **no success JSON on stdout**, plus the successful context-exit positive. No Docker qualification or new broad design round is needed to fix/review this finding.

## Other source conclusions within this boundary

- `package_install.py:15-45,138-173` wires exact final IDs, local image inspection, create with `--pull=never`, retained intent/per-role IDs/pair, waiting control before worker, dropped hello, existing register-and-notify and final readiness. Partial creation and failed/unknown operations retain state; no automatic recreate/rollback is introduced. Already-live pairs are corroborated rather than concurrently registered. `host_install.py:1168-1193,1297-1303` closes launcher documents and binds registration to the retained pair.
- `package_install.py:95-122,176-205` checks state, bounded exec phase, registration and incarnation, then corroborates protected provenance/current observation/hello. Stop uses exact-ID TERM and exit checks, existing clean-history validation before worker TERM, then stopped preflight. `host_install.py:927-968` preserves Core-before-lane offline lock order and uses stable journal/hello brackets while the live worker owns its lane lock. TERM send or CLI timeout is not represented as confirmed remote death; no Docker stop escalation, reset, remove or prune path was found.
- `entrypoint.py:492-549` changes installed Uvicorn TERM/INT into ordinary drain requests without forced-exit escalation or default signal re-raise; successful process completion requires retained lifecycle closure without error. Startup result is incarnation/registration bound and the exec reader checks the authenticated persistent API. Prior activation ownership, post-Core checks and retained cleanup responsibilities remain in place. Injected signal/server tests provide source behavior evidence only.
- Fixed scope/policy, models RO, control-only credential bind, fixed resources/caps/namespaces and closed topology remain explicit. Host source inspection validates config/model content and credential identity without reading credential bytes. No new generic host exec, credential logging, SQL writer or recovery authority was introduced.
- Deterministic archive construction and extracted help/transitive imports outside checkout passed through the authorized archive tests. Manifest includes source/data/locks and its own metadata. Dockerfile targets use the existing pinned Python 3.11 CPU base, installed factory/worker commands and combined service/Kokoro locks; context starts excluded and admits the runtime source/data families. Default legacy Docker/Compose and existing Poetry sections are unchanged. The distribution honestly requires a separately provisioned trusted interpreter and matching offline dependencies; it is not an interpreter/wheelhouse bundle.

These are bounded source/injected-test conclusions, not a declaration that all real filesystem, Docker or process timing behaviors have been qualified. R230-1 is the only blocking finding established in this round.

## Independent checks and exact test invocation

Both conftests inspected before execution; no release conftest exists. Used only isolated unittest loading, with explicit paths, avoiding pytest and its operational fixtures/guard. **210 tests PASS, 53.504 seconds, exit 0**, using this exact command:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; p="/Users/ben/.txcode/worktrees/py-audio-box/t3-9941da0e"; sys.path[:0]=[p,p+"/tests",p+"/tests/service"]; names=["tests.release.test_release_package","tests.release.test_headless_install","tests.service.test_installed_host_policy","tests.service.test_installed_registration","tests.service.test_installed_activation","tests.service.test_split_init","tests.service.test_install_child","tests.service.test_host_install","tests.worker.test_engine_lifecycle","tests.service.test_production_api","tests.service.test_production","tests.service.test_production_admission","tests.clients.test_production_clients"]; r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names)); sys.exit(not r.wasSuccessful())'
```

Interpreter is the authorized Python 3.14.7 environment, not target Python 3.11. Passing output includes the intentional `injected_close_unknown` shielded-future diagnostic and worker refusal message; neither is a clean-shutdown receipt. Archive tests build twice, compare deterministic hashes, extract, execute help and import transitive modules outside checkout using isolated Python. No separate release artifact was published by this review.

Independent lock comparison confirms all **32 service versions and all artifact hashes are substantively identical to base** (only comments/header differ). All 32 matching installed distribution metadata records in the test environment admit Python 3.11; the four service/Kokoro overlapping pins (`attrs`, `cffi`, `pycparser`, `typing-extensions`) match exactly. This independently supports the recorded metadata claim, not full cross-platform dependency execution or fresh resolver provenance. No resolver/network call or installation was performed; PyPI resolution remains the implementer's r229 receipt. `git diff --check acb2d37..HEAD` passes. Candidate remains clean.

The first attempted entirely mocked input-reader repro refused before emitting output and did not establish the finding. The subsequent real disposable public-input-file repro above reached the actual result-print/context-exit sequence and established R230-1. No source file or test was edited for either repro.

## Handoff and limits

Parent owns correction and another bounded source review before adoption. Actual trusted Linux Python 3.11 dependency/image execution, role/setgid/peer/mount/process checks, authenticated user journeys, audible CPU Kokoro speech, retained voice/artifact reuse, cancellation with confirmed descendant reap and clean/failed restart remain required as already specified by r198/r229. No new broad gate is proposed.

No install, network resolver, Docker build/run/pull/exec/stop, operational proc qualification, listener, privileged mount/UID transition, model/ML/hardware operation, pytest guard/full suite, HELD operation, commit, delegation, source adoption or unrelated cleanup was performed. Temporary files/SQLite/archive fixtures and mocked Docker/HTTP/signal observations are the only execution evidence. Sole persistent write: `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H3-RELEASE-PACKAGE-REVIEW-r230-sol.md`.
