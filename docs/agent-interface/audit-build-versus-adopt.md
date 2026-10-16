# Whole-product reuse audit: build, adapt or adopt

2026-10-15. Planning/source review only. Public upstream snapshots were fetched into a temporary research directory; no upstream code was executed, no packages installed, no models or user audio used. Sources are pinned below. Search initially surfaced downstream forks; conclusions use canonical upstreams. Feature documentation is evidence of a documented surface, not runtime acceptance.

## Why this changes the decision process

Voicebox and VoiceStudio overlap materially with the requested product. A custom API plus studio is not, by itself, a unique proposition. Before committing to a new full editor, compare a bounded adaptation of an existing product with a focused service using reusable editor components. The project's value must be demonstrated through efficient complete agent workflows, reliable production edits, local resource control and the specific existing voice behavior worth preserving.

This does not select a new upstream dependency or change the user's self-hosted/full-studio scope. It adds a real build-versus-adapt gate to the contract/prototype stage.

## Pinned evidence

- [Voicebox upstream](https://github.com/jamiepine/voicebox), source snapshot `8af7efe62fab8d33e2a5dfbedabaa45f68a5184f`.
- [VoiceStudio upstream](https://github.com/debpalash/VoiceStudio), source snapshot `8beed7e3df1983776658d03e7e4108e49ef385b1`.
- [Voicebox license](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/LICENSE) identifies MIT; [VoiceStudio license](https://github.com/debpalash/VoiceStudio/blob/8beed7e3df1983776658d03e7e4108e49ef385b1/LICENSE) identifies AGPL-3.0. These application files do not establish all dependency, model, voice or asset terms. No legal compatibility conclusion is made here.

## Voicebox: strongest whole-studio adaptation candidate in this pass

Its upstream README documents local voice cloning/design/presets, a stories timeline, versioned outputs, REST and four MCP tools, per-agent voice bindings, capture and speech playback. Docker packaging is present. Those are close enough to justify a source-level comparison rather than assuming a new studio is necessary. Native desktop/dictation/personality features also broaden its purpose beyond our proposed headless production service. [Product README](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/README.md).

### Useful concrete patterns

- Voice resolution has explicit argument → per-client binding → global default precedence. Save a voice binding once, then a routine agent call can contain only text. Keep authenticated principal identity separate from a user-controlled client label. [Resolver](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/mcp_server/resolve.py).
- `voicebox_speak` returns a generation identifier and publishes user playback activity instead of returning large audio bytes. Its tool makes playback side effects explicit. Our service needs distinct generate/download and authorized client playback operations, because a Docker server's audio device is not necessarily the listener's. [MCP tools](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/mcp_server/tools.py).
- Profile samples are separate from profile identity. Story items can pin a generation version, hold trim/position/track/gain and preserve source media references. This is useful evidence for take lineage and non-destructive arrangement. [Database models](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/database/models.py).

### Gaps against our reviewed contract

The generation queue stores coroutine work, pending IDs and running tasks in process memory using `asyncio.Queue`. Cancellation cancels an asyncio task. That is a useful serialization mechanism, but not a demonstrated durable executable job store or proof that native inference has stopped/reaped. Our job fencing and process ownership would require substantive integration, not an API shim. This is a comparison with our desired contract, not an assertion that its desktop design is defective. [Queue source](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/services/task_queue.py).

Its inspected speaking events are bounded in-memory pub/sub with dropped events for lagging subscribers; this is an activity indicator, not our durable completion outbox/replay contract. [Speaking events](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/mcp_server/events.py).

The inspected story export loads clips into arrays, skips missing/unreadable assets, computes placement in milliseconds and normalizes the whole mix when its peak exceeds one. Our proposed renderer instead rejects missing sources, uses an explicit timebase and records headroom policy. A source-reuse spike must test and reconcile these differences rather than replacing our contract implicitly. Long-project memory also needs measurement. [Story export](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/services/stories.py#L964).

The Compose file builds from source and puts the application on an ordinary bridge network with persisted data/cache and a loopback port. That is not the same deployment contract as selected pinned release images plus separated networkless inference. [Compose](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/docker-compose.yml).

The sampled models/services do not establish our project-wide compare-and-swap revisions, immutable voice snapshots or complete agent coverage of studio edits. Absence from the inspected paths is not proof of absence everywhere. These are targeted follow-up checks before adoption.

**Disposition:** shortlist for bounded whole-studio adaptation; borrow explicit per-agent binding and playback/generation separation now as contract ideas. Do not choose it as the new service authority without an integration and lifetime-cost comparison.

## VoiceStudio: unusually broad agent/headless surface, different constraints

Current official documentation describes a headless Docker backend serving a browser UI, HTTP/WebSocket speech endpoints, capability discovery, MCP and dependency-light client entrypoints. It also distinguishes edge microphone capture from remote compute. Those are relevant product patterns. Broad language/hardware claims have not been tested in this audit. [Speech platform guide](https://github.com/debpalash/VoiceStudio/blob/8beed7e3df1983776658d03e7e4108e49ef385b1/docs/speech-platform.md), [Docker guide](https://github.com/debpalash/VoiceStudio/blob/8beed7e3df1983776658d03e7e4108e49ef385b1/docs/install/docker.md).

Its MCP guide documents voice creation/design and per-agent voices, but inline WAV base64 remains the default output mode; an opt-in files mode provides URLs and confined paths. Our artifact-reference default should remain. Paths are namespace-dependent: a path returned by a container is not automatically usable by a remote agent. The lesson is to expose portable artifact handles and capability-specific transfer adapters explicitly. [MCP guide](https://github.com/debpalash/VoiceStudio/blob/8beed7e3df1983776658d03e7e4108e49ef385b1/docs/mcp.md).

The source distinguishes metadata persistence from work persistence: `job_store.py` persists job/events but explicitly retains in-memory work dispatch; restart marks interrupted work failed. `job_queue.py` serializes callable work in memory with cooperative cancellation. This is an excellent example of why restart-queryable status and durable resumption must be scored separately. [Job store](https://github.com/debpalash/VoiceStudio/blob/8beed7e3df1983776658d03e7e4108e49ef385b1/backend/core/job_store.py), [Job queue](https://github.com/debpalash/VoiceStudio/blob/8beed7e3df1983776658d03e7e4108e49ef385b1/backend/core/job_queue.py).

The longform guide documents chapter caching/progress, selective auditions, explicit interrupted-render resume and chapter cue sheets. It also describes browser/IndexedDB draft/project persistence in the Electron migration context. Thus a shared server-owned project revision contract cannot be inferred merely from the presence of both UI and API. Recheck current renderer code during a focused reuse spike. [Longform guide](https://github.com/debpalash/VoiceStudio/blob/8beed7e3df1983776658d03e7e4108e49ef385b1/docs/electron-longform.md).

**Disposition:** useful comparison for discovery, agent setup, longform preview/resume and portable exports; secondary whole-product candidate. Its application license and different persistence/ownership boundaries must be evaluated explicitly before selecting code reuse. Do not silently copy it into a differently licensed service.

## Proposed bounded decision exercise

Compare three paths using the same acceptance fixtures and preserve identical user requirements:

1. Adapt Voicebox as the human studio around a service boundary.
2. Build the shared service and use a maintained editor/component for arrangement.
3. Adopt an existing serving backend such as LocalAI for a subset of engine operations, retaining our production/job authority above it. The self-hosted audit supplies that assessment; do not add another authoritative queue.

This is not permission to build three competing products. The next contract stage produces a paper integration map and small synthetic/fake-worker prototypes for the top two credible paths, with effort ceilings and a decision record. No heavy inference is needed for initial renderer/contract checks.

Score: complete agent task success and total context; exact text/default compatibility; full studio controls; revision and lineage compatibility; restart/publication/cancellation behavior; resource isolation; installation without a source checkout; ability to retain our tested worker ownership; accessibility/long-project behavior; license/dependency compatibility; upstream changes carried; adapter complexity and ongoing maintenance.

A reusable waveform renderer is not automatically a full timeline model. An OpenAI-compatible speech route is not a production API. An MCP tool is not proof of context efficiency. SSE is not proof of durable replay or agent wakeup. A model-count list is not a quality/compatibility benchmark. Each gets separate evidence in the matrix.

Decision default until these checks: preserve the reviewed domain requirements, keep implementation selection provisional, and do not start a bespoke timeline or fork a whole studio solely on the strength of a feature list.
