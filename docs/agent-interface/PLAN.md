# Give your agents a voice — product and delivery plan

**Earlier engineering proposal:** the canonical phased handoff is now [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md), with [a short agent entry point](IMPLEMENTATION-START.md). Use its scope, working defaults and exit gates over conflicting details below. This document remains research/design rationale, not an additional mandatory checklist.

Status: [fresh global Astra review](review-fresh-product-astra.md) complete — GO to write the thorough product plan now; no further prototype, model campaign or questionnaire is a prerequisite. Latest user decision defers STT/contextual dictation from v1; the user will test it in TxCode first. V1 remains TTS, voices, full studio, HTTP/CLI/MCP and Docker. Voicebox is a research candidate only, not an adopted dependency. See [current scope and recap](CURRENT-PLAN.md). Earlier detailed engineering proposals below are superseded by the fresh review where they conflict; they are not additional user requirements.
Date: 2026-10-15. Baseline: ae67cd6a572e7105b7d41365aac0be4059d79584.
This is a planning artifact, not implementation or release acceptance.

## Confirmed direction

The first release serves a person or trusted team, self-hosted locally or on a private server. HTTP is the canonical integration, with a small CLI and MCP adapter. A complete speech production studio with scripts and timeline editing is part of v1. Agents must be able to discover and use the product cheaply without reading its implementation. Docker installation, voice acquisition, cloning/design, synthesis, assembly, queues, and optional completion integration are product requirements. STT and contextual dictation are deferred to a future upgrade, with the user testing them in TxCode first. CPU-only installation and qualified acceleration remain TTS requirements; STT quality evaluation is not a v1 prerequisite.

## Product thesis

A voice is reusable identity; a production is editable work; a job is execution; an artifact is an immutable result. People and agents operate on these same objects. The studio is an effective workspace for recording, auditioning, scripting, timing, and rendering. The agent surface expresses intent with concise typed operations and exposes detail only when needed.

The first successful task should require connecting, discovering usable voices/capabilities, submitting speech, and receiving an artifact. No repository exploration, model identifiers, mount paths, Redis access, or log parsing should be required.

## Recommended architecture

The baseline custom-build proposal uses one Python application/domain service for validation, voice/project changes, immutable execution snapshots, and result access. The fresh review recommends building on the existing backend with selected editor components. Voicebox remains source inspiration/conditional reuse, not an adopted dependency or a mandatory competing prototype; a whole-product adaptation slice needs a credible paper savings case. Waveform Playlist is a component candidate, not selected. No concrete v1 serving gap currently justifies LocalAI replacement. Preserve one authoritative production/voice/job contract in every path, and measure adapter/fork maintenance rather than assuming a bespoke system is cheaper. HTTP is its public contract; CLI, MCP, and studio are clients. Keep inference in supervised workers, separate from lightweight API/discovery. Reuse backend adapters and existing audio processing; incrementally extract logic from large CLI scripts rather than translate shell invocations into the permanent API.

Recommended architecture after Astra source review: SQLite in WAL mode with synchronous=FULL is the single authority for new service projects, jobs, idempotency records, leases, events, and completion outbox. One control owner writes a database on a local persistent volume; workers use authenticated RPC and never open that database. This requires a new bounded durable executor, not merely an HTTP wrapper around the current queue. No Redis dispatch/bridge for new v1 service jobs. Preserve existing CLI/Redis/Vast workflows as a separate legacy mode; import voices and completed packages without bidirectional synchronization or shared live ownership. Qualify SQLite crash/disk/backup behavior explicitly. Remote/cloud service execution is a later integration, not an installation dependency.

## Shared production model

Proposed v1 scope: projects, ordered script blocks with stable IDs, speaker assignments, alternative takes, selected take per block, audio clips referencing immutable assets, tracks, trims, explicit timing/gaps, gain/fades, waveform preview, playback/scrub/zoom, undo/history, and asynchronous render/export. Allow imported audio so productions can include existing recordings and supporting audio. This is a proposed speech production scope; unrestricted DAW/plugin/MIDI/video functionality is outside the recommendation.

Use a fixed integer timebase of 48,000 ticks/second and immutable 48 kHz editor/render derivatives, retaining originals and recording explicit ingest channel/resampling policy. Positions are nonnegative, ranges end-exclusive, conversions round half up, and values remain browser-safe integers. Derive playback and render from the same composition contract. A project revision identifies script, timeline, voice version, selected takes, and render settings. Editing text marks affected takes stale without destroying playable originals. Regenerate selected blocks; adopting a new take must not silently alter later explicit placements. Specify ripple versus fixed placement as explicit operations.

Agents request summaries, selected blocks/ranges, and semantic mutations rather than receiving a whole timeline. Revision preconditions prevent agents from overwriting concurrent human edits. Rendering captures an immutable revision; newer changes do not mutate a running render. Dependencies and invalidation must be explicit so retrying or changing one line does not regenerate an entire production.

## Agent interface

Proposed resource families: capabilities, voices, assets/imports, projects/scripts/timelines, jobs, artifacts, and event subscriptions. STT/provider transcription hooks and transcript retrieval are deferred-upgrade candidates, not v1 operations. Use a small set of task-oriented MCP tools and matching CLI commands; preserve typed schemas rather than a generic arbitrary-command executor. Return compact summaries with IDs, state, usable next action, and artifact references. Paginate lists, offer field projections and scoped detail, cap logs/errors, never return base64 audio or full transcripts by default. Prefer file/stdin input for long scripts.

Discovery must be cheap: static supported operations, cached installed/ready state, supported controls per backend, and explicit provisioning requirements. Discovery and liveness must not load weights, run inference, fetch models, or enumerate remote assets. Separate supported, installed, configured, ready, and verified states. Unsupported controls fail before queueing rather than silently falling back.

Submission returns a job receipt. An idempotency key is scoped to caller/operation and compared with a normalized request fingerprint: same key and same request returns the same job; changed request is a conflict. Repeating identical text intentionally with a new seed/key remains possible. Artifact identity and cache reuse are separate policies.

Persist queued/running/succeeded/failed/cancel-requested/cancelled states, attempts, bounded retry policy, leases/generations, and deadlines. Terminal publication is atomic and fenced against stale workers. Cancellation acknowledges intent separately from confirmed worker termination; cleanup follows confirmed reap. Preserve successful partial results and make batch failure policy explicit. Disk capacity, queue length, input size/duration, concurrency, and inference budgets are enforced before work can grow without bound.

Events report state transitions and meaningful stage changes, with cursors and replay. HTTP bounded wait or SSE reduces repeated status calls. Completion webhooks require explicit destination registration, signing, retry/deduplication semantics, and delivery status separate from job status. MCP notifications/tasks are optional negotiated adapters; completing work does not guarantee any client wakes its model. Durable status/artifact retrieval is always available.

## Voice management and acquisition

Use the existing named registry as migration input and preserve voice identity, prompts, source provenance, and exact reference hashes. Expose readiness by capability/backend instead of assuming all voices work everywhere. Presets, reference clones, and designed voices remain distinct to prevent misleading controls.

Microphone capture happens in the browser with permission and secure-context requirements; headless agents upload or register recordings. Local files are uploaded/ingested as assets, not interpreted as arbitrary server filesystem paths. Remote acquisition is a bounded import job with explicit supported source types, restricted network destinations/redirects, limits, timeouts, and validation. No remote pickle/prompt execution. Voice records retain source and authorization metadata without pretending a checkbox verifies rights.

## Studio journeys

First run: service is healthy → choose/provision a supported engine deliberately → connect agent or create a voice → audition a short take → start a project. Show model download, compute, storage, and readiness separately. Keep advanced engine controls available on demand.

Primary workspace: script and speaker assignments next to an audio timeline, with project-level playback and render. Voice library offers search, preview, import/record/design, readiness, and version history. Jobs show current activity, queue, clear failure/action, cancellation, and completion delivery. Settings cover connection credentials, storage/cache policy, engine readiness, and resource ceilings.

No visual style, framework, final layout, or mockup has been selected yet. They follow the approved interaction scope, with accessibility and efficient handling of large scripts/waveforms treated as implementation criteria.

## Delivery and verification

Stage 0: compare existing-product adaptation/component/backend reuse using bounded synthetic/fake-worker slices, record the implementation decision, freeze contracts and map every existing public feature into supported v1 operations or an explicit advanced compatibility path. Define representative journeys and baseline agent context/compute measurements. Preserve existing CLIs/default Qwen behavior.

Stage 1: headless service, durable TTS jobs/artifacts, versioned voices, local workers, compact discovery, HTTP and CLI, installable Docker CPU profile. Demonstrate restart/retry/cancel correctness and a first utterance without repository docs.

Stage 2: shared projects, scripts, selected takes, imported audio, composition model and deterministic render; agent operations and conflict/invalidation behavior. Existing clips/voices import through non-destructive migration.

Stage 3: full speech studio using those same contracts, microphone/import, waveform/timeline editing, audition, undo/history, export and job management. Stage 3 is required for v1, not an optional follow-up.

Stage 4: MCP adapter and one bounded provider-hook recipe, negotiated completion integration, final installation/upgrade/backups, TTS CPU/GPU profile qualification, performance/security/accessibility and real audio acceptance. Publish qualified platform/ISA/accelerator profiles; no universal GPU support is implied. MCP can be developed alongside earlier stages; release requires all confirmed surfaces.

Acceptance targets must be calibrated and approved, not presented as measured results. Candidate targets: no model load/download on connect/discovery; compact initial agent instructions and schemas with recorded tokenizer cost; bounded routine receipts/status payloads; one completion subscription/bounded wait instead of tight polling; targeted regeneration only for changed blocks; one controlled worker per resource lane; explicit cold/warm benchmarks and idle unload policy. Validate on a declared CPU/GPU/architecture matrix rather than promising universal Docker acceleration.

Required scenario tests: duplicate submission and changed-payload conflict; restart during execution/publication; cancel during load/render; expired lease with late result; callback failure with successful audio; human-agent concurrent edits; stale takes after script changes; selected-take replacement preserving timing; missing asset; low disk; malformed/oversized media; blocked remote destination; clean install, upgrade and restore; actual listening and rendered playback equivalence.

## Resolved technical defaults after Astra R1

These are engineering recommendations, not additional user-confirmed requirements. The full rationale and inspected evidence are in [Astra R1](review-astra-r1.md).

### Storage, migration and publication

SQLite uses foreign keys, bounded busy handling and short transactions in addition to WAL/FULL. No shared/network filesystem database and no multi-control-owner topology in v1. Workers cannot write canonical artifacts or metadata. The control owner validates attempt-owned staging, adopts immutable content on the same filesystem, then transactionally commits references, success and completion with a current-generation fence. Recovery reconciles unreferenced files; success cannot reference absent required deliverables. Database/filesystem writes are not jointly atomic.

Migrate through dry-run/report/apply into a distinct service data root, preserving original bytes, metadata, source hashes and legacy aliases. Never run legacy tools and the service against one mutable registry. Pending legacy jobs stay in legacy mode and appear in migration reports. Backups combine a consistent DB snapshot with referenced immutable files; test restoration on a fresh data root. Do not promise provenance that historical packages never recorded.

### Packaged roles and resource policy

Ship a small control image serving compiled UI/HTTP/state/events; a selected inference worker image with lightweight supervisor and model-owning child; and a narrowly scoped networked fetch/provision helper. No source mount, Docker socket or host development runtime in the release path. The local inference profile uses `network_mode: none`, private Unix-socket IPC on a dedicated volume, read-only model/reference inputs and writable attempt staging; no DB or provider credential mount. The helper enables the UI's explicit provisioning/import actions through typed IPC and validated staging, without giving the UI Docker control. Decode/inspect fetched media without networking.

Start with one host-wide inference slot and one resident backend configuration, on-demand load and configurable idle unload (five minutes is a prototype starting value). Compatible voices reuse model weights while conditioning caches stay keyed by exact voice/reference identity. Timeouts invalidate child generation; replacement follows confirmed reap, and unreapable owners quarantine the lane. Fair admission must prevent model-affinity starvation. Render/decode/provision lanes obey a combined host budget. Qualify Linux x86_64 CPU first, then one named Linux/NVIDIA profile; ARM/macOS GPU/cloud support requires separate evidence.

### V1 studio boundary and editing rules

Include basic uploaded speech/music/SFX beds, multiple tracks, intentional overlaps, clip and track mute, track solo, non-destructive split/trim/move/reorder, explicit pauses, gain and linear fades, play/scrub/zoom, keyboard timing edits, autosave and persistent revision history. No MIDI, plugins, video, time stretching or effects/automation graph is required. Splitting media creates references, never synthesis jobs.

Fixed placement is the default. A script reorder does not silently rearrange an edited timeline. Explicit first assembly can lay out blocks sequentially with a chosen gap. New takes are candidates; adoption preserves timing/trim only when valid and requires an explicit adjustment for a shorter incompatible take. Ripple is a separate revision-checked operation naming track/range/anchor and affected IDs. Undo creates a new revision, preserving history.

Final render rejects missing assets and unresolved placeholders. Partial render names its selection and reports omitted blocks. Stale selected takes require an explicit policy acknowledged in the render plan. Preview and offline render use the same timing, trim, gain/fade, overlap and channel contract, within declared preview tolerances. Output presets specify clipping/headroom with no hidden ducking or normalization. PCM24 WAV is the default master; additional codecs need qualification. Preserve exact script text and record synthesis-only transformations separately.

### Access and completion

Bind to loopback by default; document remote HTTPS via reverse proxy. A single-use expiring operator setup code establishes the admin credential. People use same-origin sessions with HttpOnly cookies, Secure under HTTPS, SameSite and CSRF protection; validate Host/Origin including local access. Agents get named attributable scoped tokens (read, production-write, admin), shown once and stored as verifiers through maintained libraries. Keep secrets out of URLs, browser local storage and logs.

Baseline MCP is a small stdio adapter calling the same HTTP API with a configured scoped credential; it can target a private HTTPS server. Direct remote MCP is an optional separately tested authorization profile. Neither MCP Tasks nor any subscriptions extension is mandatory. The baseline completion experience is durable handles, bounded/CLI wait, replayable events and optional signed registered webhooks. First wake adapter: a runner that maps completion IDs into its own authorized resumable workflow; host-specific wake behavior needs end-to-end evidence.

### One provisional efficiency budget

Prototype the agent-contract report's nine focused tool families, without an arbitrary permanent count cap. Preserve typed routes, project creation/rendering, artifact retrieval and bulk status. No generic command executor or untyped payload escape hatch. Essential eager catalogue plus onboarding: at most 2,500 model-visible tokens; compact status: at most 1 KiB; receipt: at most 2 KiB; default lists: at most 20 entries. Measure real tokenizer/client behavior, schema fetches and duplicate structured/text content. Task success, recovery and total cost decide any budget adjustment. Other research targets remain hypotheses until calibrated. See ACCEPTANCE.md for complete workflow measurement.

## Existing-solutions audit additions

See [AUDIT.md](AUDIT.md) and its four research tracks. Whole-studio, serving-backend and editor-component choices are separate decisions. An OpenAI speech endpoint, MCP tools, async responses, event history and a timeline do not independently establish contract fit. No dependency, model/default change, paid provider or fork is selected yet. Prior SQLite/networkless-worker choices remain the custom-build baseline; candidates must demonstrate equivalent ownership, durability/isolation and correctness or explicitly document an accepted tradeoff before selection.

### Required v1 contract refinements

- **Pronunciation and text interpretation:** retain exact authored text; version explicit pronunciation dictionaries and synthesis-only normalization policy, including locale/matching precedence and supported alias/phoneme subset. Record derived spoken text/transform identity and alignment mapping when available. Authored script, synthesis text and any ASR transcript are separate layers; a take whose source text differs from the active block is visibly stale. Speaker/character roles are separate from the voice versions assigned to them. Include these identities in take/cache plans. A new dictionary version does not mutate projects pinned to an older version; adopting it invalidates the affected dependency closure, conservatively reported when exact matching cannot be established. Expose backend-internal normalization limits rather than claiming fully known spoken behavior. Unsupported controls fail before work; no silent Markdown/header/emoji stripping or auto-polish.
- **Per-agent defaults:** bind default voice/preset/language/output settings to authenticated principal identity. Resolve precedence explicitly (call, project, principal, installation), pin immutable voice/settings identity in the accepted plan, and fail missing/incompatible bindings. A caller-supplied client name is not authorization.
- **Preflight and limits:** provide a metadata-only view of affected/locked/stale blocks, reusable results, transformations/context, new work and uncertain estimates. Execution binds an immutable resolved plan hash/revision. Simple one-shot submission derives and validates that plan internally under existing ceilings; preflight inspection is available for bulk/edit work, not a mandatory extra agent round-trip. Enforce input/storage, variant/attempt counts, queue/concurrency, container resource limits and deadlines with declared interruption margins, including QA/retries; atomically admit competing work against configured bounds. A prediction is not a CPU/VRAM guarantee. Requests within existing ceilings need no extra confirmation. A financial/per-token billing ledger is outside v1.
- **Review and selection protection:** auditions and A/B candidate history use exact take IDs; reviewed/approved state and selection locks attach to the precise revision and define invalidation. Locks guard bulk adoption, not candidate creation or a new approval ritual. Explicit revision-checked changes can alter selections; job completion cannot silently replace approved work. Technical QA is separate from human acoustic acceptance.
- **Generation, retrieval and playback:** creating audio, retrieving it and playing existing audio are distinct intents. No automatic regeneration to satisfy playback. Keep immutable service artifact IDs separate from caller-local paths and expiring grants. Playback requires an explicitly authorized client target; a server/container speaker is never presumed to be the listener.
- **Portable production handoff:** v1 exports a versioned project manifest plus selected referenced audio/source mapping and the actual rendered output and supports checked re-import into a fresh data root. Define whether originals/alternate takes are included. Engine caches/pickles are excluded. Additional captions, chapters, stems and interchange codecs need a supported-format matrix rather than an all-formats promise.

### Backend-dependent and deferred features

Represent supported neighboring text/audio context and performance direction as explicit capability-specific fields, pinning neighbor identities and invalidation scope where used. Do not promise native phonemes, audio infill, acting-reference transfer, emotional controls or duration-constrained replacement for an engine that lacks them. Invalidate the exact declared dependency closure; independent blocks remain unchanged and no invalidation automatically queues regeneration. Real-time duplex voice, global native dictation, telephony, video/dubbing, local personality rewriting, effects/plugin graphs and paid-provider dependencies remain deferred.

### Bounded reuse decision gate

Use one synthetic production and fake worker to compare whole-studio adaptation with a shared service plus maintained editor primitives. Separately compare LocalAI or an appropriate focused server against existing supervisors only for required serving gaps. Record release/source/license/asset pins, exact feature mapping, one-authority integration, unsupported controls, durability/cancellation/publication, renderer semantics, agent task/token cost, long-project/a11y, deploy/restore and upstream maintenance burden. No need to install every candidate or run models for the first decision slice. Existing voice identities/defaults and recent lifecycle guarantees remain preservation gates.

## Deferred STT research and shared portability lessons

Latest scope: all STT/transcript/contextual-dictation contracts in this section are future research, not v1 requirements. The user will test STT in TxCode first. General installation/resource lessons still inform TTS, but no new public STT-specific framework or engine is required now. Preserve existing clone-preparation, optional QA and alignment internals where current workflows depend on them.

See [STT-AND-PORTABILITY.md](STT-AND-PORTABILITY.md) and its research/evaluation receipts. The research found that core STT has direct agent value and shares ingest/assets/jobs/events with TTS; a Wispr-like global input method remains a separate native-client decision. Qualify OpenWhispr/Handy integration before creating a sibling. Prefer scoped credential support in the client; a credential-holding native shim must be transcribe-only, protect Host/Origin/browser requests, and forbid arbitrary URL/server-path forwarding. Loopback alone is not authentication. No sibling repository, always-listening client, hidden cleanup LLM or desktop-control subsystem is selected.

### Baseline STT contract

Accept uploaded/managed media or a bounded voice-message recording, probe/select an audio track/channel/range, and transcribe using a pinned declared language/model/context profile. Return a durable handle and transcript artifact with segment timing, source-range map, raw recognizer output, model/options/VAD/context identities and coverage/uncertainty. The old internal Qwen-language fallback must not turn unknown languages into English. Requested hints must propagate or fail explicitly; mixed-language capability is independently qualified, not inferred from separate FR/EN support.

Raw recognizer output, user correction and any cleaned/translated text are separate revisions. ASR may internally normalize language; “raw” means as emitted, not guaranteed human-verbatim truth. Segment timing is estimated recognition evidence; optional word timestamps are not forced alignment of an approved script. Empty recognition, pipeline-detected non-speech, no audio stream, failed decode and missing coverage are distinct outcomes. No-speech diagnostics cannot guarantee absence of quiet speech.

Bound context packs to language/vocabulary/project hints with explicit source, version and recognition-engine limits. They are data, not privileged agent instructions. Do not collect foreground apps, screenshots or clipboard by default. No speculative inference/download on context/discovery/preflight. Retain context separately from an optional cleanup stage and measure wrong-hint insertion as well as recognition gains.

A future STT release advertises a tested maximum media duration/size and refuses unsupported long inputs before inference; no hour-scale support is implied. Where an accepted profile requires multiple windows, use bounded decode/chunk artifacts, original-time mapping through VAD and joins, generation-fenced checkpoint publication and explicit covered/missing ranges. Full hour-scale incremental recovery is a later requirement unless that input duration is actually advertised. Resume reuses exact committed input/plan identities. Retain repeated words and uncertainty at joins; no global duplicate-word cleanup. Large transcripts are file/range/search-addressable and do not flood ordinary tool results. Summaries are optional derived evidence with segment references, not baseline recognition.

### Contextual dictation as a first-class workflow

The user clarified that successful speech input must understand the work context well enough to avoid repeated manual rewriting. [TxCode issue #8](https://github.com/Coriou/txcode/issues/8), inspected 2026-10-15, is the concrete client precedent: short composer dictation, small draft/thread terminology context, constrained correction, review before send, cancellation and revision-safe insertion. Its hosted-provider proposal is not a backend choice for this Docker-first project or a verified provider/model/cost contract.

Keep two explicit intents: faithful media transcription and contextual dictation. The second may combine supported recognizer hints with a bounded context-resolution stage. A recognizer hint is not proof of contextual accuracy, and raw ASR preservation is not the user-facing success criterion. The product must demonstrate useful final wording with low correction effort while retaining raw evidence and a reversible edit mapping.

Accept a small versioned context pack assembled by the client: draft near the caret, selected project/thread candidate names, symbols/paths and language preferences. The core service has no implicit right to collect the whole thread, repository, screen or clipboard. Preserve automatic language detection and French/English switching when requested; never translate to match the context's language. Record requested/effective hints. If the requested contextual policy is unavailable, reject it or require an explicit caller-authorized downgrade; never label blind recognition as contextual dictation.

Separate acoustic recognition, terminology resolution and generative rewriting. Context resolution may repair a supported obvious proper-name/technical-term match; it must not invent a task, remove negation, change an amount or silently replace valid unexpected speech with what the thread predicts. Return uncertainty when ambiguous. Optional rewriting remains a different explicit intent. A local capable model, constrained resolver or caller-supplied context processor can implement the stage; do not assume the same ASR model can do it or require another heavyweight LLM/cloud provider on every host.

Make the ordinary workflow available through one composed request/handle rather than forcing an agent to stitch multiple prompts. Return the useful candidate text, raw transcript reference, changed spans/reasons, bounded uncertainty and pinned context/processor identity. Cache identity includes context and correction policy. Dictation supports ephemeral capture with declared audio, raw-text, derived-text and context retention; provenance follows that policy rather than forcing permanent audio storage. Cover caches, logs, backups and cancellation/expiry deletion explicitly. A failed contextual stage reports partial/failed status; it cannot trigger automatic insertion or cloud/model substitution. A client's existing context-aware model can perform the text stage without resending all context or audio, but that adapter must be explicitly configured and measured.

TxCode owns microphone permission, recording, caret/focus, cancellation and draft revision. Capturing or transcribing does not submit an agent turn. A late result cannot overwrite a changed draft; cancellation discards insertion. Other clients can use the same service intent without becoming desktop automation. This is a strong integration target before any Wispr-like sibling product.

Qualification uses real contextual FR/EN utterances with project names, code vocabulary, paths, numbers and negation. Measure user correction effort and critical-token accuracy, plus false corrections under irrelevant, stale, ambiguous or malicious context. Blind recognition, hints only and composed resolution must be compared on the same held-out audio. No claim about Claude/Codex internal speech pipelines follows from the user's reported experience; that experience defines the desired outcome.

### Resource and installation policy

Start with one host-wide heavy inference lane covering TTS/STT and explicit memory admission, bounded model-affinity batches/fairness, no per-backend claim on the entire host budget. Record load/switch/queue/conditioning/decode costs separately. Preview/UI activity cannot silently start or evict a model. Interactive jobs must not starve behind unbounded bulk transcription, and priority changes do not make process preemption free.

Whisper-family CPU profiles are the initial qualification path; faster-whisper small/turbo INT8 measurements are narrow evidence, with whisper.cpp remaining a slim-runtime candidate before final selection. Better hardware does not automatically mean a larger model is more accurate or faster. Profile selection preserves language/capability and reports the actual device/compute type; fallback must be declared, never silent engine/model/quantization substitution.

The convenient-agent install goal becomes a tested Docker OS/architecture/ISA/profile matrix, a no-download doctor/preflight and explicit provisioning of selected pinned assets. This Mac's CPU/Linux-VM path is the first concrete target; arm64 CPU, older SSE4.1 images and accelerator profiles need their own full-stack qualification. Native Apple GPU support, Docker-managed host inference and GPU access inside a Linux container are distinct. Docker Model Runner's Metal support alone does not establish STT compatibility. No unsupported universal-hardware claim, source checkout or host ML installation belongs in the basic release path.

### Capability-specific follow-ups

Live streaming audio input, diarization, biometric speaker matching, forced alignment, translation, summary and LLM cleanup remain separate optional/later capabilities. A file-upload SSE response is not live input support. If later implemented, partial hypotheses replace previous text; utterance-final and session-completed are distinct, preventing duplicate insertion. No full transcript-based media editor or native OS dictation is added to v1 by implication.

## Next stage and outstanding human decisions

Next is the bounded contract/prototype stage described in [CURRENT-PLAN.md](CURRENT-PLAN.md). STT package/performance diagnosis produced working research profiles; further STT qualification is deferred and no STT starter/default is selected. The fresh review recommends writing the thorough plan now, recording justified engineering defaults and bounded qualification within delivery milestones. Proposed implementation work then produces versioned schemas for IDs/revisions, typed operations, state transitions/errors and composition; capability/default matrix; measured API/MCP catalogue fixture; complete legacy feature mapping; install/storage/network design; one fake-worker submit/restart/claim/publish/event slice; one synthetic timeline render and browser-preview slice. Stage 0 must resolve these outputs before implementation expands. This plan does not assert that those contracts already exist.

All three initial Sol reports and Astra R1 are complete. The subsequent four-track [existing-solutions audit](AUDIT.md) and [Astra audit review](review-solutions-astra.md) are also complete and supersede the earlier assumption of a bespoke implementation. Technical defaults above resolve R1's five planning objections; [Astra R2](review-astra-r2.md) verified incorporation and returned GO with no remaining blockers for the contract/prototype stage. The right approval boundary is readiness for contract/prototype work, not a frozen full implementation specification or runtime/security/visual/acoustic acceptance.

Product name/visual preferences, actual hardware/agent hosts available for certification, and final human workflow/listening approval remain human decisions. They do not block the next stage. No additional product questions are required for the current plan. The existing-solutions audit reopens implementation selection, not the confirmed product scope.
