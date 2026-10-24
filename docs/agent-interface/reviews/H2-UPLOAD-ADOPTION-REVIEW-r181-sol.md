# H2 upload independent round-three adoption review — r181

2026-10-24. **GO for bounded source adoption of the pinned candidate.** R176 R1 remains closed and R2's remaining canonical fixture/generator integration defect is closed. No new adoption blocker found. All 91 authorized focused checks pass, including actual generator determinism. Parent owns adoption; this is source/isolated synthetic/local-ASGI evidence, not installed transport/lifecycle or release hardware qualification. No release GO.

## Verified provenance and scope

Candidate `/private/tmp/pab-upload-r180`, branch `fix/upload-r180`, clean before and after execution. Verified exact linear parent chain:

- Base `84f5b63167390c67903464b598f863db96be243c`.
- Carried upload `0b7b9ace59387c4a0ecc327620731ebbadb0e733`.
- Carried correction `e458d10ed59112d9fb83a8fd26ec57f8f3b3ebdf`.
- New source `ba82cd44d04d54e09f8c27ef519ef5bc6e1333d1`.
- HEAD/docs `26b011459ceb2f24c9caf5f47ff7c55e2e6d784d`; its only delta is r180 documentation.

Stable patch IDs match prior reviewed cherry-picks: upload equals a90cb5d (`21fe1fe8d0171d10b71e00aa08fbc8ce5eeb4448`); correction equals 2963abc (`203f5c5ea97be2a3d47f1cd9ee53b5f9b32a55a8`). New source changes exactly contracts/generate.py, contracts/validate.py, contracts/fixtures/v1/operation-requests.json, tests/contracts/test_contracts.py, and the sole authorized fixture loop in tests/service/test_voice_store.py. No new service/client/runtime delta.

Read root r173/r176/r178/r179/r180 reports, verified byte equality with candidate copies, inspected source changes, consumers, generator, wrapper, both conftests, and carried auth/owner/path/content seams. No AGENTS.md found in either repository. Memory registry search returned no relevant hit and supplied no review fact. No delegation, source edits, commits, package installation or network. Sole authored file is this root report. Root's pre-existing untracked evaluation evidence was left untouched.

## Closure evidence

**R1 remains closed.** Total-byte accounting precedes buffer writes; no semantic cap remains on server-controlled ASGI message size. Existing regression retains valid 24 MiB WAV delivery as one message, 131073-byte messages and 65536-byte messages, exact hash and distinct upload IDs, plus total+1 refusal before dispatch. Focused tests pass. This establishes local-ASGI behavior, not installed server buffering or RSS.

**R2 is closed without restoring public JSON upload.** `fixture_request` is an explicit fixture boundary: exact dictionary type and sole reserved `$fixture_binary_base64` key; exact string type; encoded-length bound before decoding; strict base64 decoding; canonical re-encoding equality; then unchanged public raw-byte validation for nonempty/24 MiB bounds. Invalid/noncanonical/non-ASCII encoding becomes ContractError (ValueError includes the decoder's binascii.Error). Public validate/upload still require exact immutable bytes and reject fixture objects, legacy envelopes, strings, bytearray and memoryview. No production service/client calls the decoder. This does not claim fixture decoding validates WAV media; owner ingest retains that responsibility.

The generator serializes exact newly generated short.wav bytes and supplies voice_versions_list. Both complete operation consumers decode only at the fixture boundary and then call validate for every operation. Exact sample-keyset equality remains in both loops; all 42 declared operations remain covered. No upload skip, public bypass or weakened prior assertion was introduced. The voice-store change is confined to its authorized fixture-loop expression. Raw bounds, byte reconstruction/hash, immutable type, legacy refusal and non-idempotency assertions remain.

Full executable consumer inventory from repository search: canonical writer contracts/generate.py; complete loops tests/contracts/test_contracts.py and tests/service/test_voice_store.py; selected block_text request in test_corrections.py; transcript consumers in test_frozen_boundary.py and test_output_boundary.py; contracts/scenarios.py::transcript copies only its selected request. The generator invokes only voice_prepare, voice_design, speech_generate and project_render transcripts; those consumers never forward serialized upload. No other Python operation-requests reader or production fixture decoder call was found. Historical prose/hash inventories do not execute requests and were not rewritten.

Actual determinism ran the original subprocess `[sys.executable, '-m', 'contracts.generate']`, without substitution. The strengthened assertion compares the complete pre/post fixture filename set and bytes, including detection of added outputs. Generator source writes only the 16 authorized outputs: artifact.json, capabilities.json, expected.json, impulse.wav, job.json, long.wav, negative.json, operation-requests.json, production-100.json, production-30.json, short.wav, sine.wav, and the four transcript JSON files. Existing voice-catalogue.json is included in equality but not written. All generated outputs remained byte-identical and candidate stayed clean. Relative to carried e458d10, only operation-requests.json changes among fixtures: reserved upload representation and added version-list sample. No unrelated fixture churn.

## Carried safety and behavior

No introduced regression found in the previously reviewed seam. HTTP requires read plus production-write, resolves trusted principal, accepts raw audio/wav without public principal/server-path fields, and finishes body/length/deadline/single-app admission before immutable bytes enter explicit owner-thread asset_ingest. No stream reaches the DB owner. Client HTTP streams bounded local reads; CLI/MCP no-follow component traversal and nonblocking regular-file admission remain. MCP requires explicit absolute operator root configuration, relative confined paths, and remains disabled/undiscovered by default. Tests retain traversal/symlink/nonregular refusal and HTTP/CLI/MCP parity.

Authorized asset lookup checks direct principal ownership before media verification. Content hashes and reads its bounded range from the same no-follow descriptor, checks stable metadata, and closes descriptors before detached response. Foreign/tampered and ownerless assets remain refused. Compact voice/version owner commands avoid full get; output budgets, declared sanitized errors and readiness mappings remain covered. Real temporary Core/SQLite/WAV/Dispatcher execution and restart checks pass without model work.

Existing limits carry unchanged: admission authorization snapshot; one app factory per owner assumption; timeout does not cancel or prove rollback; unknown outcome/cancellation retains admission, and definite queue/storage failures may conservatively close it. Upstream server allocations are outside the application body bound. Repeated full-file hashing is bounded but costly. Historical assets without direct principal metadata remain unavailable. No stronger FD attack, throughput, revocation-during-upload or global-admission proof is inferred.

## Executed verification and limits

Inspected `/private/tmp/r180-contract-check.py` and imported test code before executing with `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=.`, and `/private/tmp/pab-0c-venv/bin/python`, in the pinned candidate. Wrapper directly executes original assertions/pytest.raises contexts without pytest collection/hooks/fixtures: **56 PASS, 4.530s**, exit 0. Coverage is all 33 expanded correction cases, all 15 test_contracts functions (including actual determinism, complete operation loop and strict binary fixture boundary), four catalogue transcript cases and four nondefault output-boundary cases.

Authorized unittest discovery with the same interpreter and bytecode disabled: voice_store **20 PASS (1.178s)**; upload_api **6 PASS (1.701s)**; voice_api **4 PASS (1.127s)**; voice_dispatcher **2 PASS (0.198s)**; voice_clients **1 PASS (0.253s)**; upload_clients **2 PASS (0.295s)**. All exit 0; total **91 cases/methods PASS**. `git diff --check 84f5b631 HEAD` passes. Candidate remains clean after generator and tests.

No pytest collection/guard/full suite, new package/network access, Docker, listener, ML/hardware, production or HELD operation ran. Generator used already installed authorized NumPy/soundfile; no environment repair was needed. No cleanup beyond normal own test fixtures. Installed transport/lifecycle and release hardware remain separate existing parent-owned gates; this review adds no broad gate. Parent may adopt this exact source candidate within the original bounded upload scope.
