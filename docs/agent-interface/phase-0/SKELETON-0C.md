# Lot 0C — integration skeleton candidate

Implementation only, awaiting independent Sol-low review. No self-GO, Phase 0 exit, installed-product acceptance or final Astra approval.

Base: `a9163ad66daf500d405e984330ef03dbd0815da6` (integrated contract 1.3.0 and accepted 0B). Branch `t3/build-isolated-0c-skeleton`; worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-e6cca322`. Initially clean. Candidate identity is the commit containing this packet, supplied exactly in the handoff. No shared schemas, legacy sources/defaults/Poetry lock, studio/design, root ledger/tickets/reviews were edited.

## Final paths and ownership

- `service/catalogue.py`, `domain.py`, `api.py`: ML-free, stateless canonical fixture reads and separate skeleton diagnostics. No SQLite, job admission, durable executor, scheduler or migration.
- `clients/http.py`, `cli.py`, `mcp.py`: HTTP transport, named CLI and maintained SDK stdio MCP. No private client datastore or arbitrary execution bus.
- `worker/boundary.py`: deliberately tiny private UDS connectivity probe, **not** a fake durable executor. Future execution must consume the unchanged canonical worker protocol 1.3.0. `worker/README.md` retains the distinct optional Metal companion boundary.
- `service/Dockerfile.release`, its dedicated allowlist `.dockerignore`, `service/compose.release.yml` and explicit `compose.local-ui.yml` override: dedicated packaging, independent of every legacy image/default Compose. Only allowlisted source, shared contracts and studio are sent to the builder; no audio caches, private data or local dependencies.
- `service/requirements.in`/`.lock` and `tests/release/requirements.in`/`.lock`: exact dependencies and transitive hashes; separate from Poetry and legacy tests.
- `tests/service/`: real loopback HTTP process, CLI subprocess and MCP SDK client/stdio subprocess parity, compact failures, no-ML guard, byte transfer. `tests/release/`: deterministic context measurements, guardrails and owned container-boundary harness.
- `scripts/dev-skeleton-0c.sh`, `scripts/test-skeleton-0c.sh`: loopback development and sequential bounded build/test/smoke lane. Inspect before running; no legacy remote e2e invocation.
- This document, `ACCEPTANCE-0C.md` and `skeleton-0c/`: local candidate evidence. Shared contract changes still belong to the contract/integration owner.

## Implemented and unavailable interfaces

Canonical `discover` (`GET /v1/capabilities`), `voices_list` (`GET /v1/voices`), `asset_get` and explicit `asset_content` are implemented **only as synthetic fixture reads**. Voices are empty; assets are exactly the four checked-in synthetic WAV IDs. They are not a user workspace, legacy registry or model scan. No arbitrary filesystem path is accepted. Content uses the unchanged JSON/end-exclusive range grammar, full-file hash and HTTP 200; there is no invented HTTP Range/206 contract.

`GET /skeleton` reports readiness false, no store, limitations and detail paths. `/skeleton/operations/{operation_id}` reads **one** original operation plus reachable named request/response definitions and implemented_fixture/unavailable status. Every future operation is unavailable, absent from MCP tools/list and absent as an HTTP route; attempts receive the canonical not_found error. These diagnostics are not additional domain operations. There is no generic dispatcher endpoint.

Default canonical discovery has no installed engines or qualified profiles and lists only implemented fixture operations. `/skeleton/profiles/{profile_id}` returns an explicit canonical unqualified example from the frozen fixture; CPU/CUDA/Metal are all not-ready. `checked_at` is the fixed fixture snapshot date, not a live probe. Discovery imports no ML/legacy code, enumerates no model directories, and makes no network lookup/download.

**Canonical limitation to the shared owner:** capabilities.source_types is a required const enumerating four *intended* sources. It cannot express that acquisition is currently unavailable. We preserve that schema, return no acquisition operation, and explain this explicitly in `/skeleton`, MCP fixture descriptions and this packet. Do not interpret source_types as installed importers. A later availability redesign requires the shared owner; no schema fork was introduced here.

The six eager MCP tools are four individually named canonical reads, compact skeleton status and operation detail. The latter can only inspect schemas; it cannot execute an operation. Unknown fields and unavailable tools fail with the canonical error object, isError=true and identical text/structured content. An MCP transport outage maps to not_ready; CLI exits 0/2/3 follow the contract categories. No authentication is implemented: this is a public-synthetic-data loopback fixture, not a deployable trusted-team service. Credentials/session/CSRF/media authorization remain 1B/1C gates.

Direct `speech_asset_content` caps each decoded range at **3072 bytes** before making HTTP calls. Unknown asset_upload remains unavailable, so no inline upload of any size is accepted. Ordinary discover/status/list/errors contain no audio/base64/logs. Larger retrieval is the host path:

```sh
python -m clients.cli --url http://127.0.0.1:8780 asset content impulse --output /tmp/unique-fixture.wav
```

The CLI uses bounded HTTP chunks, verifies range/total/full SHA256 and exact byte count, writes a private temporary file, then exclusively publishes the complete file. It refuses existing destinations and removes its temporary file on failure. Only compact managed-ID/hash/byte-count success reaches stdout; file bytes stay outside model context. This proves synthetic download only, not uploads or durable artifact publication.

## Toolchain freeze and primary sources

Checked 2026-10-16. Release Python **3.14.8** and Node **24.22.0 LTS** use exact official-image index digests in the Dockerfile; architecture-specific resolved image IDs are recorded in evidence. Python 3.14 is a maintained bugfix line ([official lifecycle](https://devguide.python.org/versions/), [release](https://www.python.org/downloads/release/python-3148/)). [Node's official release table](https://nodejs.org/en/about/previous-releases) marks 24 LTS and 25 EOL. Thus packaging deliberately uses maintained Node 24, while preserving the original 0B package's Node25/npm engine declarations and lock. npm's expected engine warning is an evaluation limitation, not a claim that 0B changed its pins. The actual release build/compiler results are recorded separately.

Direct pins: FastAPI **0.143.2**, Pydantic **2.14.1**, HTTPX **0.28.1**, Uvicorn **0.54.0**, MCP **1.30.0**, jsonschema **4.26.0**; test-only pytest **9.1.1**, tiktoken **0.14.0**. Maintainer-published release metadata: [FastAPI](https://pypi.org/project/fastapi/), [Pydantic](https://pypi.org/project/pydantic/), [HTTPX](https://pypi.org/project/httpx/), [Uvicorn](https://pypi.org/project/uvicorn/), [MCP](https://pypi.org/project/mcp/1.30.0/), [jsonschema](https://pypi.org/project/jsonschema/), [pytest](https://pypi.org/project/pytest/), [tiktoken](https://pypi.org/project/tiktoken/). MCP 2 is current; 1.30.0 is a deliberate maintained v1 pin, not a claim of latest major. [Official SDK v1 documentation](https://py.sdk.modelcontextprotocol.io/v1/) explicitly documents that maintenance line; this lot needs only ordinary stable stdio tools, not experimental tasks. [FastAPI query-model documentation](https://fastapi.tiangolo.com/tutorial/query-param-models/) supports closed Pydantic transport fields. Canonical jsonschema/intrinsic validation remains authoritative rather than duplicating domain schemas in Pydantic.

Lock regeneration used `uv pip compile … --generate-hashes`; uv is an authoring tool, not an installed runtime prerequisite. Initial local checks used Python 3.14.7 because local uv could not provision 3.14.8; authoritative packaged checks use pinned 3.14.8. Local and container results are not conflated.

## Packaging and installation boundary

Build/install from this candidate:

```sh
# One owner of the expensive lane; sequential control, worker, tests builds.
scripts/test-skeleton-0c.sh
# Leave the synthetic evaluation running only when deliberately requested:
docker compose -f service/compose.release.yml up -d --no-build
# Offline default: HTTP/CLI inside the control container.
docker compose -f service/compose.release.yml exec control python -m clients.cli --url http://127.0.0.1:8780 discover
# Optional browser evaluation: explicit control bridge networking (worker stays none).
docker compose -f service/compose.release.yml -f service/compose.local-ui.yml up -d --no-build
# Intended http://127.0.0.1:8780 with this override; host/browser acceptance pending.
# Optional port override SPEECH_0C_PORT.
docker compose -f service/compose.release.yml down
```

Build subprocesses have 600-second deadlines; test/measurement/Compose processes have bounded deadlines and owned cleanup. The smoke command uses its own random Compose project, fresh named IPC/assets/staging volumes, loopback port, no host data mounts and finally removes **only its own** volumes/containers/network. It checks unrelated pre-existing volumes survive. It performs no prune. Test containers use --network none, bounded CPU/RAM/PIDs, read-only root and temporary tmpfs. Build dependency fetches are explicit build-time activity. Runtime worker is network_mode:none; control uses an internal Docker network with no external egress and a declared loopback published port. On the current OrbStack daemon the internal network suppresses host publication: the control is healthy and serves HTTP/compiled HTML internally, but default host-loopback access is **not demonstrated**. `compose.local-ui.yml` explicitly opts control into a standard bridge, which permits outbound connectivity; it is a configuration option, not offline runtime evidence, and was not started in this lot. Worker stays network_mode:none. The default smoke preserves containment, tests internal HTTP and reports host reachability separately rather than failing it into a false installed-UI claim. Both run UID/GID10001, drop all capabilities, use no-new-privileges and bounded memory/PIDs. Worker receives read-only assets and writable staging/IPC only: no DB, Docker socket, credentials or service code. [Official Compose service reference](https://docs.docker.com/reference/compose-file/services/) documents these settings. The UDS ping proves this daemon's basic connectivity, not peer-authenticated job execution, reconnection, cancellation/reap or publication semantics.

Limits (control384MiB/1CPU, worker64MiB/.25CPU, tests1GiB/2CPU) are **fixture harness ceilings**, not measured ML resource requirements or supported hardware promises. No model, decoder or inference is installed. The private staging boundary is not yet a job/attempt namespace; 1A must implement ownership/fencing/reconciliation before any real work.

The unchanged studio is compiled into the control image at `/evaluation/`; `/` explicitly labels it synthetic/local/no API integration. Its existing Saved locally · no service label remains. No service data comes from localStorage. The original evaluation limitations remain: 60-second synthetic preview, tiny PCM reader, full-snapshot local history, nontransactional cross-tab CAS, no faded split or multi-split take replacement, no general pointer editing/production recovery. Browser service save/reload, installed first task and full studio integration are **pending 1C**, not demonstrated by fetching compiled HTML.

## Measurements and handoff

Run `python -m tests.release.measure` in the test image with --network none; tokenizer ranks are cached during build. `skeleton-0c/measurements.json` records compact UTF-8 bytes and o200k_base tokens for SDK tools/list, discovery/status, and explicitly offline canonical receipt/job/list fixtures, including the SDK text+structuredContent duplication. Raw payload counts are separate. No agent model or real host was exercised; host system prompts, duplication beyond these two copies and actual task/retry costs remain unknown. The 2500-token initial guardrail is checked only for this named serialized fixture. Receipt/status numbers do not mean those job endpoints exist. There is no speed/memory improvement claim.

See [acceptance matrix](ACCEPTANCE-0C.md) and [execution evidence](skeleton-0c/results.md). Independent Sol-low review is next; integration owner then verifies the accepted candidate, with final Astra global review separate. No push, PR, deployment, subagent, real models, cloud/private audio, STT/Voicebox, remote e2e or production action occurred.
