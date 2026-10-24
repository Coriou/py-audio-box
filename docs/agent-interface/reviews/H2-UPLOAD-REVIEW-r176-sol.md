# H2 upload independent first-round review — r176

2026-10-24. **NO-GO for adoption of this candidate.** Two concrete defects below; 15 permitted focused unittest methods pass. This is a bounded source/local-ASGI review, not installed transport, security, durability, hardware or release qualification. Parent owns remediation scope and adoption.

## Verified pins and scope

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-5c57fb62`, branch `t3/authenticated-upload-compact-reads`, clean before and after review tests. HEAD/docs `c8d418f45fcdebc5751679b6e5ae950ccd055a94` has sole parent/source `ae104e3bfc932d117047f6df31976c3f3a9dd0fc`; source has sole parent/base `b40f14accd857addedfc2d31d20e4322d47611c1`. Fourteen changed source/test paths match r173. Read r173, canonical IMPLEMENTATION-PLAN H0/H1/H2 and resource/evidence boundaries, CURRENT-PLAN pointer, changed source/contracts/tests and relevant Core/ingest/VoiceStore/auth/client code. No applicable AGENTS.md found. Memory registry search returned no relevant hit and supplied no review fact. No delegation, commits or source edits. Sole authored output is this report.

## Findings

### R1 — P1: valid uploads fail according to server receive batching

`service/api.py:181-186`, especially line 184, rejects any ASGI body message larger than 65,536 bytes even when the complete WAV is valid and below 24 MiB. HTTP does not give callers control over ASGI message boundaries. In particular, `clients/http.py:48-59` yielding 64 KiB pieces with Content-Length does not preserve those pieces across TCP and server buffering. A valid larger upload can therefore pass in-process and fail through the production transport with resource_limit.

This is supported by the installed server source, without starting a listener: `/private/tmp/pab-0c-venv/lib/python3.14/site-packages/uvicorn/protocols/http/h11_impl.py:265-268` appends Data events and pauses reading only AFTER the buffer exceeds its high water limit; lines 538-539 return the accumulated body in one ASGI message. `httptools_impl.py:316-319,579-580` has the same accumulation behavior. The 65,536 high water mark is not a maximum returned chunk size. The new tests preserve supplied chunk boundaries through ASGITransport; the test that rejects 65,537 bytes confirms the branch but does not establish compatibility with actual HTTP.

r173 discloses this correctly, but disclosure does not make it an acceptable restriction for a cohesive HTTP/CLI/MCP upload: compliant clients cannot satisfy it reliably. Remove the per-message semantic refusal and bound application buffering by total bytes/admission/deadline, processing bounded slices as needed; separately document/qualify upstream buffering, or supply an actual server adapter with a guaranteed receive bound. Do not claim application rejection prevents an upstream allocation already made. A permitted future transport qualification should verify a valid multi-chunk WAV through the actual server; no listener/network test was authorized here.

### R2 — P2: canonical upload schema change breaks the retained contract oracle/tests

`contracts/v1/schema.json:1168-1171` replaces the JSON request with a binary string, but `contracts/validate.py:408-415` still validates against that definition and then indexes `content_base64`, `bytes` and `content_sha256` on the request. Old dictionaries now fail schema validation; schema-valid strings cannot execute the helper's dictionary access. The helper has no working input under its new schema.

Concrete affected existing tests: `tests/contracts/test_corrections.py:11-24` expects a valid dictionary round trip, while lines 115-122 expect malformed base64 to reach ContractError after schema validation. Both are deterministically inconsistent with the new string schema. These are static findings; those tests were not executed because pytest/other test lanes were forbidden. The 15 focused methods do not cover this integration regression.

Parent should authorize a bounded contract-oracle/test migration to the raw non-idempotent upload semantics, or explicitly retain a separate legacy JSON schema for the legacy oracle with an unambiguous boundary. Do not restore the old public JSON API or weaken assertions merely to obtain green tests. Binary transport documentation alone does not reconcile executable contract consumers.

## Other reviewed behavior and practical limits

No additional blocker found in the inspected principal/owner seam: HTTP resolves trusted credentials, upload requires read plus production-write, no public principal/path enters ingest, and immutable bytes alone enter explicit owner asset_ingest after body completion. Revoked credentials fail subsequent resolution; authorization is an admission snapshot, with no continuous revocation/rollback guarantee during a body or queued owner operation. No such stronger guarantee is claimed here.

Body counter, declared length, finite body deadline, pre-buffer single admission and partial disconnect refusal are present. Owner work has its separate three-second Dispatcher wait; timeout does not cancel a queued/running mutation. Retaining admission on unknown outcome bounds repeated retained payloads. Closing admission even on definite queue-full/storage errors is conservative availability behavior disclosed by r173, not a hidden success/retry defect. Cancellation can leave the app unavailable until operator lifecycle recovery; no automatic retry/idempotency/rollback is promised. One app instance per owner remains a deployment assumption, not a global process/owner semaphore proof.

Asset reads check direct ownership before media verification, use bounded full hashing and detached range bytes, and close descriptors before network response. Core verification plus content verification means repeated full-file work (indeed two verifications on content requests); costly small ranges are a throughput limitation, not an unbounded-allocation defect. Existing CLI downloads use 64 KiB ranges, complete SHA256 verification and exclusive publication, so the new range cap does not break that path. Historical assets without direct principal metadata deliberately remain unavailable in this bounded slice.

Client root traversal uses no-follow directory descriptors and nonblocking leaf open, with explicit MCP root permission and disabled-by-default discovery. Compact list/history owner commands avoid full VoiceStore.get/history allocation; page output can still exceed 2 KiB and require a smaller limit. Full detail allocations and indexed-query performance are not qualified by this slice. Discovery, typed error membership and client parity were inspected; no clone/job implementation is required by this commission.

## Executed evidence and limits

Read both `tests/conftest.py` and `tests/service/conftest.py` before testing. Ran only `/private/tmp/pab-0c-venv/bin/python -m unittest discover -s DIRECTORY -p PATTERN -v` in the pinned candidate:

| Directory | Pattern | Result |
| --- | --- | --- |
| tests/service | test_upload_api.py | 6 PASS, exit 0, 1.009 s |
| tests/service | test_voice_api.py | 4 PASS, exit 0, 0.700 s |
| tests/service | test_voice_dispatcher.py | 2 PASS, exit 0, 0.089 s |
| tests/clients | test_voice_clients.py | 1 PASS, exit 0, 0.099 s |
| tests/clients | test_upload_clients.py | 2 PASS, exit 0, 0.130 s |

`git diff --check b40f14a ae104e3` passed. Candidate remained clean. Tests used their own temporary Core/SQLite/WAV roots and normal cleanup. No pytest/full suite/guard/Docker/listener/network/ML/hardware/HELD-resource action occurred. Server behavior evidence is installed-source inspection, not an actual network reproduction. No installed FD/lifecycle attack campaign, RSS/load benchmark, real storage/queue fault or production security claim follows.

Next gate: parent-owned bounded fixes for R1/R2 and independent successor review, followed separately by installed transport/lifecycle qualification. Release remains NO-GO.
