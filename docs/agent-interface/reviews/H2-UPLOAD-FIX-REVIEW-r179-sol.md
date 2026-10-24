# H2 upload independent round-two review — r179

2026-10-24. **NO-GO for adoption.** R1 is closed; R2's direct upload helper and two migrated tests are corrected, but a retained canonical contract consumer/fixture remains broken (P2 below). All 15 permitted focused unittest methods and 35 retained contract-wrapper cases pass. This is source/local-ASGI evidence only; parent owns remediation authorization and adoption. Installed HTTP qualification remains separate; no release GO.

## Verified pins and carried scope

Candidate `/private/tmp/pab-upload-r178`, branch `fix/upload-r178`, clean before and after checks. Verified exact linear ancestry:

- Base `1b741eeb9e465f3973993b05023593164c6bfe83`.
- Upload cherry-pick `a90cb5d7b51a354e27b2c8878cf53205c54cc474`.
- Correction source `2963abc89c81c1fb560f6f284d55b0b3170a7c6d`.
- HEAD/docs `616723b4074cf6403bc76feac45a00bc289f2077`; its only delta from source is r178 report.

Stable patch ID of base→cherry-pick equals original `ae104e3bfc932d117047f6df31976c3f3a9dd0fc`: `21fe1fe8d0171d10b71e00aa08fbc8ce5eeb4448`. Correction touches exactly the six reported paths: service/api.py, tests/service/test_upload_api.py, contracts/validate.py, tests/contracts/test_corrections.py, contracts/v1/schema.json, contracts/v1/operations.json. The complete base→HEAD file list contains upload/contract/client/tests and r178 documentation only; inherited hardware source is unchanged, not requalified. Root reports r173/r176/r178 were read. No applicable AGENTS.md found. Memory search returned no relevant hit and supplied no review fact. No delegation, source edit, commit or cleanup beyond normal test-owned fixtures. Sole authored output is this report; root's pre-existing untracked evaluation evidence was untouched.

## Finding — R2 remains partially open, P2: canonical operation samples still contradict binary upload and catalogue

`tests/contracts/test_contracts.py:119-123` loads the canonical `operation-requests.json`, requires its keys to equal all declared operation IDs, then passes each sample to `contracts.validate.validate`. The fixture has no `voice_versions_list` entry although the carried upload delta adds that operation (`contracts/v1/operations.json:1015`). Thus line 122 deterministically fails. Independently, `contracts/fixtures/v1/operation-requests.json:16-20` still supplies the removed upload dictionary with content_sha256/bytes/source/content_base64. The corrected immutable-bytes adapter (`contracts/validate.py:20-25`) necessarily rejects that dictionary as invalid_bytes. Merely adding the missing version-list key cannot close the upload failure.

The generator also needs an explicit binary-sample boundary: `contracts/generate.py:59-74` generically generates JSON requests, falling back to the text 'example' for the binary schema. Regenerating alone cannot produce the Python bytes required by the adapter and would still fail upload validation. The retained determinism test (`tests/contracts/test_contracts.py:57-60`) also expects regeneration not to change fixtures. Generator execution was not performed.

This is an existing executable consumer of the changed schema, not a request for a new API or broader feature. r178's statement that no other upload-oracle callers required migration overlooks the indirect per-operation validation loop. The 35-case wrapper deliberately selects only two functions from test_contracts.py and does not include this function or generator determinism; its green result cannot establish their coherence. Findings here are static and concrete, not claimed executed failures.

Parent should authorize the bounded remaining migration of canonical samples/generator and their consumer to an explicit raw-binary fixture representation/adapter, while preserving complete operation coverage, exact raw bytes and rejection of the old public JSON envelope. Include a valid voice_versions_list sample. Do not restore JSON upload, skip upload validation, weaken the key-set assertion or regenerate unrelated fixtures blindly. Correction paths beyond the six authorized here require parent scope ownership; this review makes no edits.

## Closure and carried behavior

R1 closed: `service/api.py:180-187` counts total bytes and refuses total overflow before writing the offending message; it no longer refuses server-controlled message sizes. `tests/service/test_upload_api.py:135-157` passes actual valid 24 MiB RIFF delivered in one message, 131073-byte pieces and 65536-byte pieces, checks exact hash and distinct IDs, and refuses single-message/multi-message total+1 with owner dispatch patched to fail. This establishes local-ASGI semantics, not installed server behavior or upstream allocation/RSS bounds.

R2 direct correction is sound within its boundary: `contracts/validate.py:20-25,47-49,414-420` accepts exact immutable bytes, enforces nonempty/24 MiB bounds and rejects upload idempotency before JSON fingerprinting. Migrated tests preserve byte reconstruction/hash, boundary/type/legacy-envelope refusal and explicit non-idempotency. Media validation stays owner-ingest responsibility. The remaining finding is consumer/fixture integration, not a defect in those direct checks.

No additional adoption blocker found in inspected carried auth/path/FD/owner/client behavior. Upload resolves read plus production-write, accepts no caller principal/server path, completes finite transport admission before sending immutable bytes to explicit owner-thread ingest, and retains total/deadline/single-app admission checks (`service/api.py:160-202`, service/auth.py, service/dispatcher.py). Principal ownership precedes Core media verification; content hashes and reads a bounded range from the same no-follow descriptor and closes it before detached response (`service/domain.py:94-140`). CLI/MCP walk client-host components with no-follow descriptors and nonblocking leaf open; MCP requires explicit root configuration, rejects absolute/traversal inputs, and remains undiscovered by default. Client error/asset parity and compact history owner commands remain intact. Declared readiness/media/auth errors and sanitized failures were inspected and focused tests pass.

Authorization is an admission snapshot. Per-app admission closes on not_ready/unknown/cancellation and may close on definite queue/storage failures; one factory per owner is a deployment assumption. Owner timeout does not cancel mutation or guarantee rollback. Repeated full-file hashing and conservative availability behavior remain disclosed limits, not new blockers. Historical assets without direct principal metadata remain unavailable. No stronger FD attack, revocation-during-upload, global admission or throughput guarantee is inferred.

## Executed checks and evidence limits

Read tests/conftest.py and tests/service/conftest.py, plus the retained wrapper and selected imported test code, before execution. Used only `/private/tmp/pab-0c-venv/bin/python`, with PYTHONDONTWRITEBYTECODE=1, in the pinned candidate:

| Command scope | Result |
| --- | --- |
| unittest discover tests/service, test_upload_api.py | 6 PASS, 1.305s |
| unittest discover tests/service, test_voice_api.py | 4 PASS, 0.741s |
| unittest discover tests/service, test_voice_dispatcher.py | 2 PASS, 0.081s |
| unittest discover tests/clients, test_voice_clients.py | 1 PASS, 0.100s |
| unittest discover tests/clients, test_upload_clients.py | 2 PASS, 0.132s |
| PYTHONPATH=. python /private/tmp/r178-contract-check.py | 35 PASS, 1.828s |

Every command exited 0. Wrapper imports pytest assertion contexts/decorators but performs unittest calls, not pytest collection/hooks/fixtures. `git diff --check 1b741ee HEAD` passes. Candidate remains clean. Tests use temporary real Core/SQLite/WAV/Dispatcher and in-process ASGI/substituted client transport.

No additional contract function, generator, pytest collection/guard/full suite, Docker, listener/network, ML, hardware, production or HELD-resource operation ran. No installed transport, lifecycle/security/durability, RSS/load or hardware qualification is claimed. Parent-owned next gate is bounded completion of the remaining contract migration and independent successor review; installed HTTP qualification remains separately authorized work.
