# H2 upload corrections — r178

2026-10-24. Bounded implementation complete for R1/R2; parent-owned independent review and adoption remain open. No self-adoption or release GO.

## Pins and isolation

Fresh own worktree `/private/tmp/pab-upload-r178`, branch `fix/upload-r178`, created clean from current project root commit `1b741eeb9e465f3973993b05023593164c6bfe83`. This includes integrated hardware `93b4823` and hardware report `deeb611164963fc48387ffcb5e90a18f518058bf` (initial orchestration checkout pin). Project root had pre-existing untracked evaluation evidence, left untouched. No reset or historical-worktree mutation.

Only upload source `ae104e3bfc932d117047f6df31976c3f3a9dd0fc` (original base `b40f14accd857addedfc2d31d20e4322d47611c1`) was cherry-picked, cleanly, as `a90cb5d`. Correction source commit: `2963abc89c81c1fb560f6f284d55b0b3170a7c6d`. Report committed separately. Read root r173 and r176; r173 matches the worktree copy. No applicable AGENTS.md found. Memory registry search returned no relevant hit and supplied no implementation fact. No delegation.

## Corrections and scope

R1: removed the semantic 64 KiB ASGI message refusal. HTTP receive batching is server-controlled. Total 24 MiB counting before buffer writes, declared-length checks, finite body deadline, pre-buffer single admission, immutable owner payload and fail-closed unknown outcome behavior remain. No server/upstream allocation bound is claimed: the server may already have allocated an arbitrarily large receive message before application inspection. BytesIO/immutable payload and owner retention behavior remain as disclosed in r173; no RSS or lifecycle qualification follows.

Regression exercises a real valid 24 MiB RIFF preserving ancillary bytes/hash, delivered as one ASGI message, 131073-byte pieces, and original 65536-byte pieces. Repeated uploads get distinct asset IDs. Both original multi-message counter overflow and a single 24 MiB+1 message refuse with owner call patched to fail if invoked. Existing auth/media/length/deadline/disconnect/admission and owner tests remain.

R2: the contract oracle now treats the binary schema as raw immutable Python bytes rather than a JSON string or base64 envelope, enforcing nonempty/24 MiB bounds. Generic JSON Schema validation remains for all other definitions; binary wire representation necessarily uses the explicit raw-body adapter in validate(). Upload idempotency checks explicitly reject with non_idempotent before JSON fingerprinting. The generic fingerprint utility remains unchanged. Media validity remains owner-ingest responsibility; this transport oracle does not pretend to validate WAVs, auth, deadlines or receipts.

Migrated the two stale tests to actual raw semantics: exact byte/hash download reconstruction; maximum size and empty/over-total refusal; rejection of text, mutable bytes, memoryviews and legacy JSON; explicit non-idempotency declaration/replay rejection. Old claimed-size/hash/base64 assertions describe the removed public envelope and were replaced by actual byte boundaries/hash and legacy-envelope refusal, not suppressed. Public JSON upload was not restored.

rg across contracts/tests/clients/service found no other upload-oracle callers beyond tests/contracts/test_corrections.py; no additional contract tests required edits. Reported scope before edits. Only six allowlisted source paths changed: service/api.py, tests/service/test_upload_api.py, contracts/validate.py, tests/contracts/test_corrections.py, contracts/v1/schema.json and operations.json. The two declaration edits remove obsolete per-message limits while preserving raw binary/non-idempotent semantics. All other carried upload files unchanged from the cherry-pick.

## Red-green evidence

Interpreter `/private/tmp/pab-0c-venv/bin/python`. Read tests/conftest.py and tests/service/conftest.py before execution; no contracts conftest exists. No pytest collection, guard activation or listener fixtures executed.

Before implementation, modified upload regression ran six methods with one failure: maximum single-message upload returned 429 resource_limit, `Transport chunks must be at most 65536 bytes.` Both migrated raw oracle functions failed with jsonschema.ValidationError against the old helper. These are failing regression observations, not successful validation. During migration a test used the incorrect metadata key idempotent; corrected to existing idempotency before final green runs.

Final isolated unittest discovery, `python -m unittest discover -s DIRECTORY -p PATTERN -v`:

| Directory | Pattern | Methods | Result |
| --- | --- | --- | --- |
| tests/service | test_upload_api.py | 6 | PASS, 1.391s final rerun |
| tests/service | test_voice_api.py | 4 | PASS, 1.043s |
| tests/service | test_voice_dispatcher.py | 2 | PASS, 0.166s |
| tests/clients | test_voice_clients.py | 1 | PASS, 0.147s |
| tests/clients | test_upload_clients.py | 2 | PASS, 0.182s |

All 15 focused methods pass, exit 0. Last upload rerun includes the restored original counter-overflow case, all three receive batchings and distinct-ID assertions.

Standalone contract import/unittest wrapper (PYTHONPATH=., test module import path tests/contracts), without pytest collection: 35 cases PASS, exit 0, 2.085s. Exact coverage: all nine functions in test_corrections.py, expanding intrinsic identity over 3 kinds x 5 changes (15), immutable edits over 3 operations x 3 groups (9), block mutation over 3 field/value pairs (3), and six remaining unparameterized functions (6); plus unchanged test_contracts.test_schema_and_all_fixtures and test_revision_idempotency (2). Original assertions and pytest.raises contexts execute directly; pytest decorators are imported but no collection/fixtures/hooks run. Schema/fixture coherence and unrelated JSON idempotency remain checked. No determinism/generator subprocess test ran. Wrapper retained at `/private/tmp/r178-contract-check.py`; local logs `/private/tmp/r178-contract-green.log`, `/private/tmp/r178-red-upload.log`, and `/private/tmp/r178-test_*.py.log` are supplemental host evidence, not committed fixtures.

`git diff --check` passed. Tests use real temporary Core/SQLite/WAV/Dispatcher with in-process ASGI routing and substituted synchronous client transport. Normal fixture cleanup only.

## Remaining gates

No actual listener, network, Docker, full suite, pytest guard, ML, hardware inference, HELD-resource or production operation ran. Installed HTTP/server batching and lifecycle/security/memory behavior require a separately authorized lane. Existing per-app admission/deployment assumptions, owner timeout uncertainty and conservative closure remain. Hardware code was carried unchanged, not qualified here. Parent owns independent successor review, integration and any installed transport test. Release remains NO-GO.
