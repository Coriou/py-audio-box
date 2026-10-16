# Agent contract research and architecture proposal

Research date: 2026-10-15. Scope: planning for a single person or trusted team running a self-hosted speech production studio, with HTTP, a small CLI, MCP, scripts, and timeline editing. This is a proposal, not implemented behavior or runtime acceptance. Repository inspection was read-only; no models, inference, deployments, credentials, or callbacks were exercised. Only this planning document was added. No applicable `AGENTS.md` was found in the checkout or its ancestors.

## Recommendation

Build one versioned domain service owning projects, exact script text, voice revisions, timeline revisions, durable jobs, and immutable artifacts. Make HTTP the canonical external contract. The human UI, CLI, and MCP adapter must call that same service. Workers execute immutable plans; adapters must not each grow their own queue, settings defaults, retries, or timeline rules.

An agent should be able to say: “Create this project, use this voice, generate these three lines, move this clip, regenerate line 2, render revision 14, notify my runner.” It should not need to learn Docker internals, pick model checkpoints, copy an entire timeline back into context, or poll through repeated model turns.

Timeline editing is mandatory in v1. Implement it in stages, but preserve project/script/timeline identity in the first API. A synthesis-only API followed by an unrelated timeline system would create incompatible agent and human workflows.

## Repository evidence and implications

These are source observations at inspection time, not an exhaustive security audit.

| Existing evidence | Architectural implication |
| --- | --- |
| `lib/tts_backends/contract.py:7-45` already has typed backend errors, capability descriptions, and validated audio results. | Preserve this engine boundary underneath the domain service; transport-specific response shapes belong above it. |
| `lib/jobqueue.py:37-55,185-236` defines synthesis fields, content hashes, request IDs, attempts, and `callback_data`. | Reuse useful concepts, but generalize from a synthesis job to typed production operations. Separate correlation metadata from executable callbacks. |
| `lib/jobqueue.py:73-89,381-406` atomically deduplicates enqueue by request ID. It does not compare the supplied payload against the earlier payload in that Lua script. | New idempotency semantics must reject same-key/different-request reuse, rather than silently treating every repeated key as equivalent. |
| `lib/jobqueue.py:408-453` leases Redis Stream jobs and reclaims stale leases. `apps/job-watcher/job-watcher.py:883-909` stores scheduled retry and ACK together. | Existing work is a useful execution adapter. Delivery/reclaim is not proof of exactly-once synthesis or publication. |
| `apps/job-watcher/job-watcher.py:1031-1058` stores results and ACKs in a Redis transaction, echoing `callback_data`. | Existing correlation data is not an implemented webhook or agent wakeup facility. Add a durable completion outbox explicitly. |
| `apps/job-watcher/job-watcher.py:1062-1064` assembles a local package after the done record and ACK. | Define precisely whether “succeeded” means raw audio ready or the complete requested package ready. A studio render job must not become successful before its required deliverables exist. |
| `apps/voice-synth/voice-synth.py:738-860` builds a machine-readable capability payload; by default its speaker probe can call `_load_owned_model` at line 794. | New cheap discovery must not wrap this command blindly. Static catalog discovery, cached worker readiness, and an explicit expensive runtime verification are separate operations. |
| `apps/voice-synth/voice-synth.py:1228-1288,1738-1761` defines and writes structured synthesis results. | Promote result schemas into reusable domain DTOs; do not scrape console output. |
| `lib/voices.py:805-870` imports bounded WAV/JSON exchanges, rejects pickle caches and unsafe archive entries, stages before promotion. | Preserve these restrictions in HTTP/MCP import. Never expose an arbitrary model pickle upload endpoint. |
| `apps/job-watcher/README.md:3,34-40` describes Vast provisioning, manifest execution, uploads and queue completion. | Keep remote GPU execution behind a worker interface; it is not a required control-plane dependency for every self-hosted install. |

The current content hash is deliberately a subset of synthesis fields. A new cache key must also bind immutable voice/reference identity, engine/checkpoint revision, resolved defaults, output settings, and pipeline version. Existing hashes should not be advertised as complete production reproducibility identities.

## Canonical domain model

Use opaque stable IDs, explicit revisions, and immutable binary artifacts. Keep one local durable metadata store for the initial trusted deployment, with a transactional jobs/outbox table. SQLite with one API owner is a reasonable v1 choice; a worker can still be a separate process. Redis remains an optional dispatch backend or bridge to the existing watcher. If SQLite and Redis are combined, insert an outbox record in the database transaction and make dispatch idempotent; do not pretend two independent writes form one transaction.

| Object | Required identity and useful contents |
| --- | --- |
| Project | ID, revision, title, script IDs, timeline IDs, voice assignments, render presets, compact summary, change cursor. |
| Script | ID/revision, ordered stable block IDs, exact authored text, speaker/voice assignment, optional pronunciation references. Editing text creates a revision; it does not rewrite accepted text elsewhere. |
| Voice | ID/revision, display name, kind, supported capabilities, immutable reference asset IDs, model preparation status. Mutable labels do not change historic renders. |
| Timeline | ID/revision, tracks, stable clip IDs, source artifact/take IDs, in/out points, positions, gain, fades, mute and clip/block association. |
| Take | ID, block/text revision, voice revision, resolved generation plan, source job, measurements and artifact references. Selection is separate from generation. |
| Job | ID, typed operation, immutable resolved plan, project/base revision, attempts, state, timestamps, limits, result/error, notification target ID. |
| Artifact | ID, digest, MIME type, byte length, audio duration/rate/channels where relevant, parent job, authorization and retention information. |

Timeline time should use integer ticks under an explicit timebase, with conversion into render sample positions defined once. Avoid floating point seconds as the canonical edit format. Choose and document one rounding rule for resampling and end-exclusive trims. Retain original media as immutable assets; edits are non-destructive references.

### Small canonical API surface

Endpoint names below are proposed, not present today.

| API | Purpose |
| --- | --- |
| `GET /v1/catalog` | Compact operation names, schema versions, defaults/preset IDs and capability revision. No model load. `?operation=…` returns one operation schema. |
| `GET /v1/voices?query=…&limit=…` | Filtered compact catalog; a detail endpoint returns one voice and readiness. |
| `POST /v1/media` | Bounded authenticated uploads. Optional remote import is a separate typed job. |
| `POST /v1/projects`; `GET /v1/projects/{id}` | Create/read production state. Reads accept a view, stable IDs, time range, or change cursor. |
| `POST /v1/projects/{id}/edits` | Atomic bounded batch of typed script/timeline edits against `base_revision`; idempotency key required. |
| `POST /v1/jobs` | Submit a typed operation with idempotency and budgets; return after durable acceptance. |
| `GET /v1/jobs/{id}`; `GET /v1/jobs?ids=…` | Compact status/result lookup, including bulk lookup without N model turns. |
| `POST /v1/jobs/{id}/cancel`; `POST /v1/jobs/{id}/retry` | Explicit cancellation and a new linked retry job, with distinct idempotency. |
| `GET /v1/artifacts/{id}` | Authorized bytes or a short-lived download grant; metadata remains separate. |
| `GET /v1/events?after=…` | Durable event page; optional SSE representation over the same stored events. |

Supported operation schemas should cover synthesis, built-in voice registration, clone/reference preparation, voice design, media extraction/splitting, transcription/alignment/QA as exposed by engines, audio export, and project rendering. Advertise actual supported engine/mode combinations. A “full studio” does not justify promising every engine supports every language, direction control, timing, or streaming mode.

Registry changes such as rename/delete/import have explicit authorization and revision checks. Keep them distinct from audio generation even if exposed through one compact `voice_manage` adapter tool. Scripts and timeline edits are cheap control-plane commands; model work and rendering are durable jobs.

### Project editing without loading everything

Provide `view=outline`, `view=selection`, `view=changes`, and an explicit full export. Default outline returns block/track/clip counts, IDs, revision, readiness and conflicts, not all text and waveforms. Support reading selected blocks or clips, a track time range, and changed IDs since a revision. Return an explicit cursor/limit and `truncated` marker when bounded.

Typed edits should include `script.add_block`, `script.replace_text`, `script.assign_voice`, `timeline.add_clip`, `timeline.move_clip`, `timeline.trim_clip`, `timeline.set_gain`, `timeline.set_fade`, `timeline.select_take`, and `timeline.remove_clip`. Stable IDs prevent index shifts from editing the wrong object. An atomic edit batch either validates completely or changes nothing. Retain exact submitted text; whitespace normalization is not an invisible convenience.

Require `base_revision` (and HTTP `If-Match` where used). A human editing revision 12 makes an agent's revision-11 edit fail with a bounded conflict report containing changed IDs and the current revision. Never overwrite the whole project to resolve this automatically. New generated takes can attach to the project library, but automatic replacement of an edited timeline clip needs the expected clip revision. Stale synthesis completion is not authority to replace a human choice.

Regenerate by block/clip IDs plus expected text/voice revisions. Build dependencies server-side: text or voice change invalidates the affected takes; a clip move invalidates the mix render, not the source speech; an export format change reuses the mix master where applicable. Return the changed dependency summary rather than a full graph.

Render a pinned project revision. Store its manifest, source digests, selected takes, mix/export configuration and tool versions. Re-rendering existing immutable takes should be reproducible under a pinned rendering environment; bit-for-bit identity across arbitrary toolchains/architectures is a separate tested claim. Neural regeneration may remain nondeterministic even with a seed. Never equate a seed with a reproducible model output guarantee.

## Durable job semantics

Recommended states: `queued → running → succeeded | failed | cancelled`, with `retry_wait` and `cancel_requested` as visible non-terminal states. Keep approval/input requirements in the plan before accepting expensive work whenever possible. Retries preserve the immutable plan and create attempt records; a user-requested new attempt after a terminal job creates a new linked job ID.

1. Validate schema, references, requested mode and admission limits before acceptance. Resolve presets/defaults and immutable asset/voice/project revisions once. Record both requested and resolved settings.
2. Bind idempotency to the caller and operation. Same key and same canonical request returns the original job; same key and different request returns `409 idempotency_conflict`. Publish the key retention period. After retention expires, never imply deduplication still holds. Content caching is separate and must be opt-in for stochastic creative work.
3. Commit job and dispatch event durably before returning. HTTP submission can use `202 Accepted`, `Location` and a server-recommended poll delay. HTTP 202 alone promises neither completion nor later delivery; the job resource supplies that contract. [HTTP semantics, §15.3.3](https://httpwg.org/specs/rfc9110.html#status.202).
4. Give each worker claim a lease and monotonically increasing generation/fencing token. Only the current claim may publish. Heartbeats extend ownership; recovery after a crash retries or fails according to policy. A dead worker does not get to overwrite a newer result.
5. Generate under an attempt namespace. Verify required outputs, then publish an immutable artifact manifest and terminal state atomically within the metadata transaction. External object storage requires reconciliation for staged/orphaned objects; a Redis ACK cannot make an object-store write atomic.
6. Cancellation is idempotent intent. Acknowledge `cancel_requested`; mark `cancelled` only after stop/reap or a reliable worker acknowledgement. A completion race may return `succeeded`. A disconnected HTTP or MCP stream must not accidentally cancel a durable job.
7. Retry only classified transient failures with bounded jittered backoff and an overall deadline. Do not silently retry bad requests, missing/incompatible voices, corrupt assets, policy failures or permanent capacity/configuration failures. A timeout can mean unknown execution outcome; reconcile the job and its publication fence before running again.
8. Make `max_attempts`, job deadline, maximum input text/media duration, variants, concurrency and optional compute/cost ceiling explicit. Estimates carry uncertainty; a hard admission/usage limit is a separate enforced value.

Compact failure example, mapped to HTTP `application/problem+json` and to MCP tool error content:

```json
{"type":"urn:pab:error:voice_not_ready","title":"Voice is not ready","status":409,"code":"voice_not_ready","retryable":false,"field":"voice_id","action":"prepare_voice","request_id":"req_7"}
```

RFC 9457 provides a standard machine-readable HTTP problem format with extension members. The proposed `code`, `retryable`, `field` and `action` fields are application extensions. Do not put stack traces, internal paths, private URLs, text bodies or credentials into default errors. Job failure remains a successful HTTP read of a failed job resource; it is not necessarily HTTP 500. [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457).

## MCP, discovery and context cost

### Protocol facts checked on 2026-10-15

The released MCP core revision reported by the maintainers is **2026-07-28**. It has a stateless per-request metadata model, optional `server/discover`, cacheable lists and an extensions framework. Do not design a new implementation solely from old session/initialize examples. Backward compatibility must be tested for the selected clients and SDK versions. [Release announcement, 2026-07-28](https://blog.modelcontextprotocol.io/posts/2026-07-28/).

MCP tools support discovery, typed input/output schemas, structured results and resource links. Return the compact job/result envelope in `structuredContent`; support compact serialized text for compatibility as recommended by the specification. Avoid duplicating large results. Resource links are useful artifact references, but client download/playback behavior still needs testing. [MCP tools, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/tools).

The `io.modelcontextprotocol/tasks` extension is optional. Its 2026-07-28 shape uses `tasks/get`, `tasks/update`, `tasks/cancel`, and optional `notifications/tasks` through an opted-in `subscriptions/listen`. `tasks/get` includes the final result. Cancellation is cooperative and completion can win a race. The extension requires durable creation before a handle is returned. Do not mix these names with 2025-11-25 experimental task APIs (`tasks/result`, `notifications/tasks/status`). Confirm the exact extension schema and client support at implementation time; the task docs have separate stable and draft schemas. [Tasks specification](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks/), [extension overview/schema links](https://tasks.extensions.modelcontextprotocol.io/).

### Adapter recommendation

Ship a baseline MCP adapter using ordinary, quickly completing tools that return the canonical job handle. This works without native MCP Tasks. Add native task mapping only for tested clients that declare the right extension; keep one underlying job and authorization model. Do not keep a synthesis `tools/call` open for minutes as the only way to obtain its result.

In that optional bridge, application generation failures belong in a completed tool result with `isError: true`; the extension's `failed` status is reserved for JSON-RPC failures. Keep domain job state available in the result instead of assuming transport status and production status are identical. [Tasks specification](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks/).

Start with approximately nine focused tool families: `voice_discover`, `voice_search`, `media_import`, `voice_manage`, `audio_generate`, `job_get`, `job_control`, `studio_read`, and `studio_edit`. Keep common inputs short and bounded. For rendering, expose a typed project-render operation through the job adapter. Do not turn `studio_edit` into arbitrary code, shell commands, or opaque untyped JSON. Split an ambiguous family when evaluation shows selection errors; nine is an initial hypothesis, not a token target that overrides usability.

The catalog exposes named profiles and operation-specific details on demand. Typical speech needs voice, text, language if needed, output preset, idempotency key and notification target. Most agents should never supply sampling internals. Do not pre-load all engine parameters, voices, model cards, examples and help text into every conversation. A one-page integration guide should show submission, wait/status, download and one project edit.

Host-level lazy tool loading can help, but it is not universal MCP behavior. Anthropic's documented tool search supports deferred loading and recommends retaining a few frequent tools eagerly. Its current docs distinguish model-context loading from sending tool definitions in the API request. Measure actual context and wire costs independently. [Claude tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool). Programmatic orchestration can also keep intermediate data outside model context; vendor-reported savings are not a benchmark for this product. [Anthropic advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use).

Default responses should exclude echoed source text, logs, waveforms, transcripts, embedded audio/base64, and entire manifests. Return opaque IDs, status, next recommended action and artifact metadata. Provide explicit detail/export views. Stable schema ordering, bounded pagination and cached catalogs preserve prompt prefixes. Short cryptic field names are a poor trade if they cause agent mistakes; readable keys with fewer irrelevant fields are preferable.

## Waiting, completion notifications and waking an agent

Support three independent paths:

| Path | Appropriate use | Guarantee boundary |
| --- | --- | --- |
| `pab jobs wait job_… --json` or SDK `await job.wait()` | Shell agents, scripts and local runners. It waits/polls in ordinary code. | No repeated LLM turns are required; process interruption does not lose the server job. |
| HTTP event pages and SSE | Human UI progress, a connected orchestration process. | Persist events, include cursor/sequence, replay after reconnect, document retention/gaps. SSE alone is not durable storage. |
| Signed completion webhook to a registered destination | A runner that can resume/start the correct workflow. | Delivery is at-least-once with retry/dead-letter state; “delivered” does not prove agent execution. |

WHATWG SSE defines event IDs and reconnect `Last-Event-ID` behavior. Build replay from stored events, not just an in-memory stream. On an expired cursor, return an explicit resync instruction and current job state. Use a same-origin session for browser EventSource or a fetch-based stream for bearer authentication; never place long-lived bearer tokens in query strings. [HTML SSE standard](https://html.spec.whatwg.org/multipage/server-sent-events.html).

Create a terminal event and notification outbox entry in the same transaction as terminal job state. Callback delivery failure must not turn successful audio into failed synthesis or trigger regeneration. Events include `event_id`, `job_id`, terminal state and correlation ID; the consumer retrieves the authorized result. Retry delivery independently with jitter and a bounded horizon; expose attempt count, last error, next retry and dead-letter/manual replay in the UI.

Register callback targets out of band and refer to a target ID in jobs. Sign raw event payloads and include a message ID and timestamp; verify before parsing for action, deduplicate IDs, and enforce a replay window. Standard Webhooks specifies these signing/replay conventions and key rotation. [Standard Webhooks specification](https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md).

**A notification cannot universally wake a dormant agent.** The receiving host must offer and authorize a resume/start entry point, store the run/workflow mapping, and handle duplicates. MCP notifications indicate a protocol event to a participating client; they do not guarantee a new model turn or reopen a closed conversation. Do not emulate waking by writing to terminals, posting into chats or reusing session credentials. Offer one small local runner recipe first; label each further host adapter with tested client/version and behavior.

## Security proportional to a trusted self-hosted v1

Use a local-only default bind and a clearly configured remote HTTPS mode. Keep model/compute-provider credentials server-side. The HTTP and CLI path can use a scoped per-installation token; remote MCP authorization must follow its applicable OAuth/resource-server contract for clients that require it, rather than assuming every host accepts an arbitrary bearer setting. Pin the chosen MCP authorization revision and validate issuer/audience/scopes. [MCP authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization).

At minimum separate read, production-write and administrative permissions. A trusted team still benefits from attribution and a recoverable edit history. Require authorization for artifact reads and downloads; random IDs are not access control. Keep cookies same-site with CSRF protection on UI writes. Explicit destructive voice/project deletion belongs outside routine generation and retains recoverable history according to a documented policy.

Treat inbound media and outbound callbacks as separate SSRF surfaces. Prefer uploads; if remote import is enabled, allow only configured schemes/hosts, prohibit URL credentials, cap redirects/download size/time and decoded duration, check every redirect and DNS resolution against private/loopback/link-local/metadata destinations, and constrain network egress. Admin-configured local targets are a deliberate exception, not a per-job bypass. Disable callback redirects. Use sandboxed unprivileged decoders with CPU/memory limits, validate content, and preserve bounded archive validation. These are proposed controls informed by [OWASP SSRF prevention](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html).

Imported transcripts, filenames, metadata and model outputs are data; they must not become instructions to an agent. The completion event should never contain executable commands or an instruction prompt. Avoid placing secrets, signed URLs, raw reference audio or full authored text in default logs, traces or tool results. Keep retention and deletion explicit for voices, inputs, takes, exports and stale attempts.

## Concrete happy paths

### Give a shell agent a voice

Illustrative proposed CLI, not commands available today:

```sh
pab generate --voice newsroom --text-file reply.txt --preset speech --idempotency reply-42 --json
# {"job_id":"job_42","state":"queued","poll_after_ms":2000}
pab jobs wait job_42 --json
# {"job_id":"job_42","state":"succeeded","artifacts":[{"id":"art_42","type":"audio/wav"}]}
pab artifacts download art_42 --out reply.wav
```

The agent submits once, waits in a process, and receives one compact final result. A runtime without a long-lived process stores the job ID and uses polling in its runner or a registered callback.

### Agent creates, human edits, agent regenerates one line

1. Create project `prj_9`; atomically add blocks `b1,b2,b3`, voice assignment and timeline clips. The response contains revision 1 and the created IDs.
2. Submit `generate_blocks` for those IDs at revision 1. The plan resolves exact text and voice references; results attach new takes without changing a later human selection.
3. Human trims clip `c2`, changes the second block and selects a take in the UI. Project becomes revision 4.
4. Agent reads `view=changes&since_revision=1`, then only block `b2` and clip `c2`. It receives the new exact text, selected take, trim and revision.
5. Agent requests regeneration for `b2` at its expected revision. It chooses a new take using an atomic `timeline.select_take` edit against revision 4. If the human edits again, this fails with a bounded conflict instead of replacing their work.
6. Submit `render_project` pinned to revision 5 and preset `podcast_stereo`. The worker reuses unaffected takes and produces a manifest and downloadable master/export. A registered completion target gets an authenticated event; its runner may resume the agent if configured.

### Callback is unavailable

Audio finishes and artifacts remain available. Outbox delivery retries without regenerating speech. After restart, the runner retrieves the job ID or replays stored events, verifies current state and processes the event once. Callback delivery and agent-run execution have separate statuses.

## Instrumentation and proposed acceptance budgets

These are initial design targets to validate, not measured current performance. Run protocol/control-plane tests with deterministic fake workers; model and listening tests are a separate budget. Collect both HTTP/CLI and real target MCP-host traces using a named tokenizer/client version.

| Area | Initial acceptance target |
| --- | --- |
| Cold agent discovery | Core tools plus minimal guide ≤2,500 model-visible tokens; compact catalog ≤800 tokens. No model load or network model download. Measure eager and host-lazy paths separately. |
| Common speech workflow | Known voice: one submission and one terminal result exposed to the model; ≤700 product-generated result tokens total, excluding user source text and unavoidable host protocol wrappers. Record actual tool definitions and duplicated content too. |
| Project scale | 100 blocks/200 clips: outline ≤1,000 tokens; targeted single-block edit/read ≤1,500 result tokens across the operation. No full project round-trip required for changing one clip. |
| Status calls | Small status ≤1 KiB, bounded bulk lookup, poll delay exposed. Runner polling creates zero additional LLM turns. |
| Control-plane latency | Local warm-host catalog/status p95 ≤100 ms and durable submit p95 ≤250 ms under 20 concurrent clients, measured separately from inference and large uploads. Publish hardware/database settings. |
| Reliability | Repeated same-key submits yield one logical job; changed payload conflicts. Inject crashes before/after dispatch, publish, completion/outbox commit and notification acknowledgement. No stale attempt may replace newer artifacts or project edits. |
| Cancellation | Deterministic fake worker stops within 5 s; real backend stop/reap latency measured and published independently. No “cancelled” state while an unfenced worker can still publish. |
| Revision safety | Concurrent UI/agent edit produces one winner and a bounded conflict. Moving a clip invokes no TTS. Regenerating one block invokes no unrelated block generation. |
| Render | Pinned fixture manifest yields stable sample counts, trims, placement/gain/fades and source mapping. Verify source text and selected take identity as well as output file existence. Bit-identical expectation only under a declared render environment. |
| Completion | Replay/dedupe tests cover dropped SSE connections, duplicate/out-of-order webhooks, consumer outage and service restart. A notification outage creates no extra synthesis jobs. |
| Agent usability | A small task suite: direct speech, clone setup, missing capability, variant selection, project creation, targeted regeneration, concurrent human edit, render, cancellation and resume after disconnect. Compare success, correction turns, wall time, input/output tokens and compute use against current CLI. |

Trace correlated IDs across API request, job, attempt, project revision, worker claim and artifact. Record queue delay, load/conditioning/generation/render/upload durations, cache hit reason, audio duration/RTF, peak memory where obtainable, attempts and notification latency. Keep estimates distinct from measured usage and genuine billing. Avoid high-cardinality raw text/URL labels. Track “waiting with no model turns” as a product efficiency metric alongside token counts.

## Alternatives and staging

1. **HTTP domain API + compact CLI + thin MCP, recommended.** Shared contracts, conventional operational tooling, easy agent scripting, same studio state for humans. Costs: an API lifecycle, schema/version discipline and conformance tests.
2. **CLI-only wrapper.** Excellent early local execution path; weak remote/browser and cross-client consistency if each wrapper invents state. Retain it as an adapter, not the studio's sole source of truth.
3. **MCP-only service with native Tasks.** Convenient for some hosts, but host/version/extension behavior controls progress and resume. It does not solve human timeline state or provide universal wakeup. Defer making Tasks mandatory.
4. **Full workflow platform immediately.** Can provide mature schedules/retries, but creates installation and operational cost for a single user. Begin with transactional jobs/outbox and fenced workers; adopt a workflow engine when measured orchestration complexity warrants it.

Suggested staged v1: (A) shared domain/store, script blocks, minimal timeline edits and immutable render plan; (B) durable synthesis/render workers, artifacts, revisions, CLI/UI/MCP parity; (C) completion outbox and one tested runner; (D) expanded voice/media operations and hardening benchmarks. Every stage keeps the same IDs and schemas. Timeline functionality remains part of the v1 acceptance gate.

Defer multi-tenant billing, collaborative CRDT editing, arbitrary executable workflows, a public model marketplace, universal host wake adapters, native MCP Tasks dependency, and real-time duplex conversation. A real-time streaming audio path has different latency/session semantics and should not be confused with long-form production jobs.

All external sources above are primary specifications, maintainers' announcements or vendor documentation, retrieved on 2026-10-15. Protocol and client support can change; version-pin and recheck immediately before implementation.
