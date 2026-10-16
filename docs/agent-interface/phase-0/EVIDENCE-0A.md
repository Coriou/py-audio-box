# Lot 0A implementation evidence

Base: `a25169bd674b38c232e3476a23ede251b99d20e4`; worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-0fe72d9f`; task branch initially `t3/0fe72d9f`, renamed by the orchestration environment to `t3/voice-contract-pack` while HEAD remained at the base. No integration-branch checkout or mutation was performed by this implementer. Baseline status was clean. PRODUCT, implementation entry point/protocol, shared contracts/Phase 0 and full lot 0A ticket were read; no applicable AGENTS.md was found in the worktree/parent search.

Contract, fixture and worker protocol versions: 1.0.0. All generated media is synthetic; no private audio, weights or real inference. No app/legacy defaults, production Compose, ticket or execution ledger changed. The remote e2e script was not executed. No PR/push/deploy or phase approval.

## Reproduction

Lightweight isolated validator environment (temporary, outside worktree):

```sh
python3 -m venv /tmp/pab-0a-contracts-venv
/tmp/pab-0a-contracts-venv/bin/pip install -r tests/contracts/requirements.txt
/tmp/pab-0a-contracts-venv/bin/python -m pytest tests/contracts -q
```

Final result: **14 passed** (1.07s). Host interpreter was Python 3.14; the test image below uses Python 3.11. Requirements pin the validator, its dependencies and pytest dependencies without modifying the application lock. Initial suite passed 10 tests, then 12/13 as cases were added; no test/build failure occurred. Installation was repeated after pinning all harness dependencies and passed.

Inspected `Dockerfile.tests` and `scripts/run-unit-tests.sh` before execution. Only validator dependency and copies needed for contract/launcher snapshot collection were added. Built the existing test Dockerfile, then used equivalent bounded network-disabled invocations rather than running the full offline suite:

```sh
docker build -f Dockerfile.tests -t py-audio-box-0a-contract-tests .
docker run --rm --network none --cpus 2 --memory 3g py-audio-box-0a-contract-tests python -m pytest tests -m 'not integration and not redis_integration' --collect-only -q
docker run --rm --network none --cpus 2 --memory 3g py-audio-box-0a-contract-tests python -m pytest tests/contracts tests/test_capabilities_and_metadata.py tests/test_clone_prompt_compatibility.py tests/test_clone_prompt_identity.py tests/test_customvoice_feature_flag.py tests/test_voice_cpu_batch.py tests/test_voice_synth_selection.py -m 'not integration and not redis_integration' -q
```

Final build: success, image `sha256:5b5cae63a569e685c95c9c3069af813f5de0ba1c59baa7cfd66507e93a434b1c`. Final collection: **320 tests collected in 2.24s**, no collection errors. Final selected suite: **85 passed in 19.02s**, exit 0. Prior image iteration passed 84 selected tests in 18.31s before the final non-interference test was added. No concurrent heavy test jobs were run; container executions were sequential, CPU/RAM bounded, network disabled, without user-data mounts, and `--rm` cleanup.

Local ephemeral logs: `/tmp/pab-0a-install-final.log`, `/tmp/pab-0a-docker-build-final.log`, `/tmp/pab-0a-collection-final.log`, `/tmp/pab-0a-regressions-final.log`. These are not required inputs for reproduction. `git diff --check` passed before commit.

## Evidence limits and next work

All results are **E1 static/schema/offline unit evidence**, including tests executed in the existing image. They do not establish E2 integrated service lifecycle behavior, E3 installed worker/hardware qualification, E4 browser/agent-host acceptance or E5 acoustic acceptance. No measured RAM, latency or support claims are made. CPU/CUDA/Metal fixtures are unqualified and not-ready; no remote host specs were inferred.

The oracle checks supplied snapshots and trusted verification inputs; future service consumers must implement actual transactions, owner fencing/reap, hash/media verification, filesystem reconciliation, request authorization, SSRF/decoder isolation and profile admission. Renderer consumers must implement the normative sample rules and measure preview agreement. Migration is policy/report validation only, never applied to a real registry. No new public STT/dictation/TxCode contract exists.

0B consumes fixture/schema 1.0.0 and CONTRACTS.md for editor adapter/browser evidence. 0C consumes individual operations/request definitions, errors, worker protocol and profile topology; it owns actual framework/client/package pins and token measurement. MIGRATION.md and COMPATIBILITY.md constrain later adapters. Consumer proposals that change these contracts must return to the contract owner with updated fixtures/tests rather than fork schemas. Candidate awaits separate adversarial review; no phase GO or self-approval is claimed.

## Exact changed paths

```text
Dockerfile.tests
contracts/__init__.py
contracts/fixtures/v1/artifact.json
contracts/fixtures/v1/capabilities.json
contracts/fixtures/v1/expected.json
contracts/fixtures/v1/impulse.wav
contracts/fixtures/v1/job.json
contracts/fixtures/v1/long.wav
contracts/fixtures/v1/negative.json
contracts/fixtures/v1/operation-requests.json
contracts/fixtures/v1/production-100.json
contracts/fixtures/v1/production-30.json
contracts/fixtures/v1/short.wav
contracts/fixtures/v1/sine.wav
contracts/generate.py
contracts/v1/errors.json
contracts/v1/legacy-controls.json
contracts/v1/operations.json
contracts/v1/schema.json
contracts/validate.py
docs/agent-interface/phase-0/COMPATIBILITY.md
docs/agent-interface/phase-0/CONTRACTS.md
docs/agent-interface/phase-0/EVIDENCE-0A.md
docs/agent-interface/phase-0/MIGRATION.md
tests/contracts/requirements.txt
tests/contracts/test_contracts.py
```
