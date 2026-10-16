# Fresh global product review

2026-10-15. Repository baseline: `ae67cd6a572e7105b7d41365aac0be4059d79584`. Read-only review of application source and the research packet, with this report as the sole new file. No implementation, dependency installation, inference, benchmark, account action or runtime change. The parent is updating scope documents concurrently; the user's latest direction takes precedence over earlier research recommendations.

## Verdict

**GO to write the thorough product and delivery plan now.** No further research campaign, model comparison, prototype or human questionnaire is a prerequisite to that plan. The plan can choose sensible engineering defaults, identify the few choices requiring later evidence, and put their qualification inside delivery milestones. It must not present an unbuilt service, an unqualified editor or a hypothetical hardware profile as accepted.

The product is a private, self-hosted speech workspace: agents obtain and use reusable voices cheaply; people and agents create, revise, arrange and export the same productions. HTTP, a small CLI, MCP, voice acquisition/creation/cloning/design, manageable jobs/results, and a complete speech-production studio are the core. STT and contextual dictation are a **future upgrade**. The user will first try them in TxCode. That experiment neither blocks v1 nor authorizes a TxCode implementation here.

My recommended direction is **a focused service built on the existing voice/audio work, with selectively reused studio components**. Keep Voicebox as a source of product ideas and a conditional reuse candidate. Do not make a Voicebox fork or a two-product prototype race part of the default schedule. Keep Qwen's existing defaults and capability distinctions; qualify explicit alternative profiles without silent substitution.

The strongest prior recommendations remain sound at the outcome level: cheap discovery, coherent identities, recoverable work, bounded compute, shared human/agent edits and trustworthy exports. Earlier reports sometimes turned implementation preferences and interesting market features into mandatory gates. The new plan should correct that expansion.

## Evidence used and its limits

Read: [PRODUCT.md](../../PRODUCT.md), [CURRENT-PLAN.md](CURRENT-PLAN.md), [PLAN.md](PLAN.md), [ACCEPTANCE.md](ACCEPTANCE.md), [AUDIT.md](AUDIT.md), the [feature inventory](research-feature-audit.md), [agent-contract research](research-agent-contract.md), [product/install research](research-product-install.md), [whole-product audit](audit-build-versus-adopt.md), [serving/adoption integration map](research-serving-adoption-comparison.md), [Docker-portability research](research-docker-portability.md), the earlier R1 architecture review, and [final STT debugging synthesis](evaluation/stt/DEBUGGING.md).

I independently inspected the current registry, queue and execution boundaries, Qwen process owner, discovery imports, Docker files and CPU launcher. I also inspected the already downloaded Voicebox snapshot `8af7efe62fab8d33e2a5dfbedabaa45f68a5184f`, particularly its queue, story exporter, data model and packaging. This is fresh inspection of pinned source, not a claim about a newer upstream release. The recorded editor and Voicebox export probes were read, not rerun.

Source establishes available code and integration differences. Existing probe receipts establish their named function/headless observations. Neither establishes real browser usability, complete runtime durability, acoustic quality, release installation or measured agent savings. [TTS_MEMORY.md](../TTS_MEMORY.md) contains bounded Kokoro runtime evidence; no numerical result was independently remeasured in this review.

## What matters most for the product

The useful distinction is not the count of speech engines or the existence of an MCP endpoint. It is completing work with little supervision: an agent can find a usable voice, generate speech, receive a recoverable result, create an editable production, and repair one line without losing the person's arrangement or regenerating everything.

The studio is a first-release requirement, not an optional dashboard after an API launch. Conversely, an early working service is a legitimate delivery milestone without being advertised as the complete v1. Installation, voice preparation and editing deserve equal attention to synthesis: if any requires reading source, knowing Redis, identifying model paths or repeatedly correcting hidden defaults, the product promise is unmet.

Five journeys should organize the plan and its acceptance:

1. Install a declared profile; discover readiness; connect an agent; generate and retrieve the first short utterance.
2. Acquire a reference by microphone, file or supported remote source; audition/select a segment; prepare a voice; explicitly generate a test.
3. Describe a designed voice; audition candidates; save a reusable identity; distinguish this from cloning and built-in presets.
4. Create a multi-speaker script; generate/select takes; arrange, listen and export; reopen the same work.
5. Let a person and an agent alternate edits, regenerate only the changed line, recover a failed job, and preserve the selected audio and timeline.

## Voicebox: why it was raised, and what to do with it

It was raised by the earlier reuse audit because it visibly overlaps with the requested product: voices, generation versions, stories/timeline, agent bindings and playback. That was a legitimate way to challenge an assumption that everything must be built here. **It was not selected by the user, adopted as a dependency or chosen as the foundation.** The new plan should say that directly.

The inspected source supports real reuse value. Story items reference generation versions, carry positioning and trims, and expose gain; the application already has voice, player and story UI. Per-agent voice defaults and explicit generation versus playback are useful patterns. Reusing an entire mature workflow can save more than extracting a few components, so a whole-product option is not inherently unreasonable.

However, the concrete integration surface is substantial:

| Observed evidence | Consequence for this product |
| --- | --- |
| `backend/services/task_queue.py` stores coroutine jobs in an in-memory `asyncio.Queue`; cancellation cancels the task. | Durable request/attempt ownership and demonstrated native-worker stop/reap need integration. Existing generation history does not itself restore executable queued work. |
| `backend/services/stories.py` resolves selected versions, but skips missing/unreadable media and peak-normalizes an over-range mix. | Useful take lineage, different export behavior. Strict missing-media failures and explicit output treatment need changes. |
| The recorded isolated exporter probe reproduces those policies and a zero-gain issue. | These are bounded function observations, not proof that the whole product is unsuitable. The gain fix alone is not a fork/no-fork decision. |
| Inspected story models/services do not establish the intended project-wide revision preconditions. | Agent/person editing needs a specific integration map and verification; do not infer absence everywhere from selected files. |
| Its pinned Compose builds from source; the image includes multiple model families and a web build. | A small pinned release with explicit selected model provisioning remains packaging work. It has a real web build, so calling it desktop-only would be inaccurate. |
| The desired canonical API, existing voice identities/defaults and worker guarantees belong to this repository's product. | A fork may require changes through backend, frontend state and release lifecycle together, reducing the benefit of ordinary upstream updates. |

Primary pinned source: [queue](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/services/task_queue.py), [stories](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/services/stories.py), [models](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/database/models.py), [Compose](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/docker-compose.yml). Probe boundary: [receipt](evaluation/voicebox/README.md).

**Decision recommendation:** use source inspiration and narrowly justified component reuse by default. A short reuse decision note inside the first implementation milestone should inventory the concrete UI work saved, the state/API seams changed, dependency/license boundaries and ongoing patch ownership. Only reopen a whole-Voicebox thin slice if that map presents a credible net reduction in work. Do not require equal prototype effort or build two studios to find out. This is a preference based on observed integration risk, not a measured assertion that custom development is cheaper.

Waveform Playlist is also a candidate, not an adoption decision. Its [headless probe](evaluation/editor/README.md) found overlap clamping, refresh-cleared undo and client-generated split IDs. Those are relevant adapter questions. The first browser slice must show the intended overlap, keyboard edits, reload, selected-take replacement and preview/export behavior. Rejecting a component after this small slice should not force redesign of canonical project data. A waveform display alone is insufficient reuse for a full editor.

LocalAI is a separate serving decision. The existing supervisors already implement meaningful mode validation and process cleanup. No concrete v1 gap currently justifies replacing all of them with another server. Evaluate a serving adapter later only for a named need—an unavailable hardware backend, materially better supported lifecycle, or a measured integration saving. Remove STT-serving and TTS/STT-switch prototypes from the v1 path.

## Architecture: requirements versus recommended means

### State, queue and execution

**Required outcomes:** accepted work and finished results survive restart; duplicate delivery does not create unintended jobs; job inputs pin voice/text/settings; late or canceled work cannot overwrite accepted results; edits cannot silently overwrite another client; recovery remains understandable. These outcomes do not require a distributed scheduler or automatic resumption of every partially completed inference.

**Recommended default:** a Python domain service with SQLite metadata on local persistent storage, immutable managed media, one local execution lane initially, and supervised model processes. HTTP is canonical; CLI, MCP and studio call the same operations. SQLite is a proportionate way to keep projects, requests, attempts and optional event delivery in one transaction domain for one installation. WAL/FULL, foreign keys, bounded busy handling and short transactions are reasonable starting settings to verify. They are engineering choices, not user commitments or guarantees that a host power loss cannot lose data.

The existing Redis code must be credited accurately. [`lib/jobqueue.py`](../../lib/jobqueue.py) already has leases/reclaim and owner-checked coordination; the initial audit's “no queue rewrite” and the later “new SQLite executor” are different recommendations. SQLite is preferable here because the new project/edit state needs persistence anyway, and adding Redis would create another installed service and a metadata/dispatch consistency boundary. This does **not** prove a new executor is cheaper or safer. Carry over useful state-transition ideas and failure fixtures; preserve the legacy runner intact. Document what is reused and what is new before implementation. Do not undertake a general job-platform rewrite.

Start with the minimum execution contract: durable queued request, owned attempt, bounded execution, staged result, verified publication, terminal result and explicit retry. After restart, an interrupted attempt may be reported as interrupted/failed with a safe retry action once the old owner is fenced/stopped. Automatic retry is warranted only for classified cases and within declared limits. Exactly-once physical inference is not promised; one logical request and one authoritative published result are the practical invariants.

File adoption and a database commit are not a joint atomic transaction. Use attempt-owned staging, verify required files, adopt immutable media, then commit current-generation references and success. Recovery must distinguish orphan files from committed deliverables. A returned success cannot point to absent required media. Avoid an event-sourcing framework: ordinary revision records and a small event/outbox table, when delivery is implemented, suffice.

### Model ownership and resource control

Keep API discovery responsive while inference loads or fails. Reuse [`QwenBackend`](../../lib/tts_backends/qwen.py) and the CPU worker supervision contracts; timeout/error must not lead to a second active owner while the old one remains unreaped. Network cancellation alone is not proof that computation stopped.

One expensive model job at a time is a safe initial scheduling policy, not a permanent product limit or a reason to block cheap reads, uploads and playback. Include render/decode/provision contention in resource controls; a second category is not free because it is not called inference. Start with straightforward admission and fairness. Defer sophisticated model-affinity scheduling, predictive RAM reservations and remote-worker orchestration until measurements warrant them.

Warm model reuse across service jobs is worth implementing because the current manifest runner launches a process per job while existing backend owners/batches already support residency within their lifetime. Qualify the extension rather than assuming current queue batching keeps Qwen warm. Idle unload policy is an operator setting; “five minutes” should remain a prototype value, not a requirement.

### Packaged processes and network access

Recommend one lightweight control/UI image plus the selected worker image. Serve a compiled frontend. No source checkout, writable source mount, Docker socket, Redis credential or host ML environment should be needed by a normal release install. Existing [`docker-compose.yml`](../../docker-compose.yml) is a developer toolbox; it is not the release path.

Preserve explicit provisioning and offline inference where supplied today by [`run-cpu`](../../run-cpu). A networked downloader/importer and a networkless inference worker are distinct permissions. They do not require three independent permanent services at the beginning. The implementation can use a small typed helper or explicit operator command, with the UI provisioning flow added through that boundary. Do not hide model downloads in discovery or broaden ordinary inference egress for convenience.

A private Unix socket in a Docker-managed Linux volume is a reasonable local transport candidate. The prior Mac ping/pong proves a primitive, not production authentication, reconnect/cancel behavior or Windows portability. Qualify the packaged arrangement on advertised targets. If a qualified internal TCP profile is necessary, describe its actual network boundary rather than claiming it is networkless. The API/process boundary matters more than a prematurely universal IPC promise.

### Storage, migration and access

Use a separate service data root and one owner for each mutable registry. Import existing voices and completed packages with a dry run, byte/hash checks, legacy aliases and explicit missing provenance. Keep pending legacy jobs with their current runner. Do not invent bidirectional synchronization or automatically rewrite the existing cache. Versioned voice identity is necessary; filesystem slug rename is not stable API identity.

Keep originals, references, selected takes and project revisions separate from rebuildable peaks/caches/model bundles. State retention and deletion plainly; unlimited immutable history eventually fills a disk. A conservative first policy can retain production material until explicit deletion and clean only demonstrably unreferenced transient data. A backup must include a consistent database and referenced files; a restore on a fresh data root is a release gate.

For one person/trusted team: loopback by default, a documented private HTTPS deployment, attributable revocable agent credentials, browser sessions and protection against cross-origin writes. Maintain authorization for media downloads. No multi-tenant IAM or public signup is implied. Remote source acquisition remains required but should support an explicit source set with bounded redirects, size, duration and fetch destination rules. A trusted team does not make arbitrary URL fetches or imported media trusted code.

## Full studio: retain the real scope, avoid an accidental DAW

The minimum coherent studio includes reusable voices; exact editable script blocks and speaker roles; generated alternatives and selected takes; imported audio; multiple tracks; explicit placement, overlap and pauses; non-destructive trim/split/move; gain/fades; audition/playback/scrub/zoom; keyboard timing controls; save/reload, undo/history; and queued render/export. Basic music/SFX beds fit this model. This is enough to complete a real production, without a plugin graph, MIDI, video, mastering suite or automatic dubbing.

The central workflow rule should be visible: changing text makes relevant takes stale; it does not erase audio or immediately run inference. A new take is a candidate. Adopting it preserves explicit placements where valid and asks for a concrete adjustment when an old trim no longer fits. Reordering the script and rearranging an already edited timeline are distinct actions. A late generation cannot silently select itself over a newer human choice.

Revision preconditions and a recoverable history are justified. A separate formal approval/lock workflow, elaborate dependency graph, universal phoneme dictionary and dedicated review bureaucracy are not established user requirements. Start with exact text, explicit transformations/settings and selected-take lineage. Add pronunciation aliases if supported and useful to actual v1 work; unsupported phonemes must fail rather than disappear. Do not make a full dictionary subsystem or speculative cross-utterance continuity schema a first-release blocker.

The 48,000-tick integer arrangement is a defensible implementation convention, not a product benefit. Keep originals unchanged and make any renderer derivative/resampling policy explicit. A fixed timebase does not require eagerly transcoding every imported file. Derivatives can be created when needed and cached. Preview and render must implement the same clip semantics; exact sample rendering and perceptual browser playback are separate checks.

Choose one master format and a small useful delivery set, qualified in real workflows. A portable manifest plus selected media and rendered mix is a useful proposed v1 handoff; captions, chapters, stems and broad interchange formats are optional additions. Operational backup/restore is mandatory regardless of portable-project feature scope. Do not require all export ideas before users can finish speech.

## TTS quality, hardware and installation

The new plan needs a **capability-by-profile matrix**, not one green “CPU works” badge. Preset speech, reference preparation, clone synthesis, CustomVoice direction and VoiceDesign have different requirements. Qwen Base does not become an instruction-driven voice merely because the UI has an instruction field; Kokoro does not replace cloning; VoiceDesign entails its own model and conversion path. These distinctions are in [the existing adapter contract](../../lib/tts_backends/qwen.py) and [feature research](research-feature-audit.md).

Preserve existing Qwen defaults. Offer explicit CPU alternatives where they actually support the requested capability. Installation can succeed on an older Intel machine while a demanding design profile remains impractical or unavailable; tell the user before starting work. Do not silently relabel that condition as successful equivalent fallback. The full product must demonstrate cloning/design on at least a declared capable profile, but every host need not pretend to offer every engine mode.

TTS qualification is needed during delivery, not before writing the plan. Reuse existing receipts for what they establish and run only the gaps for the shipped worker/service combination: first/cold and repeat/warm utterance, reference preparation, designed-voice preview/save, cancellation/restart, longer script, and final human listening. Record exact images/assets/settings, model load count, queue/load/generation time, output duration, measured memory scope, failures and source identity. Compute improvements must preserve the intended voice and controls. No broad engine bake-off is necessary absent a specific unresolved product problem.

The [STT debugging work](evaluation/stt/DEBUGGING.md) contributes general release lessons: generic architecture labels can conceal unsupported instructions; zero exit code can still mean help rather than an output; downloads need realistic supervised deadlines; CPU placement can materially affect execution. It does not validate a TTS profile or require changing every TTS worker's affinity. Retain the precise receipts as future STT research.

Deferring public STT does not remove internal reference transcription, optional QA or alignment already used by existing voice workflows. Preserve those behaviors and make optional costs explicit. New transcript resources, context packs, provider hooks, held-out contextual-dictation campaigns and TxCode integration are deferred. Generic assets/jobs/artifacts and versioned operations already leave sufficient extension room; speculative STT abstractions are unnecessary now.

Installation should be measured as an actual user task: prerequisites, commands, manual decisions, download size, disk use, elapsed time to first usable voice, failed steps and recovery. Start with the available Intel/Linux-container CPU profile, then qualify named additional CPU/GPU targets as hardware permits. Distinguish CPU support on a Mac from Apple GPU access. No universal Docker/GPU promise follows from a model's upstream support list. Broad hardware compatibility is a direction; the release promise is the tested matrix.

## Agent efficiency and completion

Short schemas are useful but not the objective. Measure the total successful task: onboarding/tool definitions, discovery, request/results, corrections, failed attempts, status calls and model turns. Compare current CLI and new interfaces using the same requested work and engine/profile, separating first connection from returning use. Count host-injected and duplicated tool content where observable. Byte budgets are engineering guardrails, not tokenizer or billing measurements.

Keep obvious operations typed, readable and compact. Known-voice speech should need one submit and one terminal result exposed to the language model; waiting occurs in the CLI/runner. Do not force a metadata preflight for every utterance. Use bulk submission/status and scoped project/block reads to make larger work economical. Tool count is a consequence of task usability; neither eight nor nine is a product requirement. Ordinary responses should reference artifacts rather than embed audio, long logs or a whole timeline.

For routine speech, propose an explicit economical preset: one take, no optional QA/alignment, resolved voice/settings and existing resource ceilings. Preserve legacy defaults; a versioned new preset is different from silently changing them. Track inference/model loads and regenerated blocks as well as tokens: a tiny response can conceal expensive repeated synthesis.

Baseline completion is a durable job handle plus CLI/bounded wait and later retrieval. Replayable progress supports the UI. Optional registered completion delivery can add a small transactional outbox with bounded retries when implemented; notification failure must never regenerate audio or change success into synthesis failure. Test one concrete runner/host adapter when its host is available. Do not build universal wake adapters, require native MCP Tasks, or imply a server notification can wake an arbitrary dormant model.

## Decision register for the new plan

“Confirmed” denotes user direction or explicit preservation scope. “Proposed” is the recommended engineering/product default to carry in the plan. “Open” needs a bounded delivery decision, not necessarily a question to the user. “Deferred” is outside v1.

| Decision | Status | Recommended disposition |
| --- | --- | --- |
| One person/trusted team; local/private Docker | Confirmed | Plan one installation and a tested support matrix. |
| HTTP, small CLI, MCP; agents first | Confirmed | Shared domain handlers and compact typed operations. |
| Voice acquisition, cloning/design, synthesis; full script/timeline studio | Confirmed | Keep complete v1 journeys; early service milestones are not the full release. |
| Qwen defaults and existing work preserved | Confirmed preservation | Explicit optional profiles; no migration/default changes in planning. |
| STT/contextual dictation/TxCode first experiment | Deferred from this v1 | Preserve research and existing internal voice-processing dependencies. |
| Core service plus reused editor parts | Proposed | Default route; qualify components through the actual workflow. |
| Voicebox foundation/fork | Not selected; conditional open option | Source inspiration now; reopen whole-product prototype only after a credible savings map. |
| LocalAI/new serving platform | Deferred unless a concrete gap emerges | Reuse existing supervisors first. |
| SQLite metadata and small local executor | Proposed | One service authority; credit Redis knowledge and preserve legacy mode. |
| WAL/FULL, local volume, staged artifact adoption | Proposed | Test crash/disk/restore; do not claim joint filesystem/database atomicity. |
| Lightweight control plus selected worker | Proposed | No source mounts or Docker socket in release; explicit provisioning boundary. |
| Networkless local worker transport | Proposed, qualification open | Start from named-volume UDS; verify actual supported daemons. |
| One heavy inference lane, configurable unload | Proposed initial policy | Keep cheap operations responsive; broaden only from measured need. |
| Minimal editing contract and canonical timing | Proposed | Fixed placement; explicit ripple; preserve originals and selected-take lineage. |
| Exact component, frontend framework, detailed layout | Open engineering decisions | Select in first browser slice; do not ask the user to choose libraries. |
| Published model/device/capability profiles | Open qualification | TTS runtime and listening evidence for what ships; no STT prerequisite. |
| Token/payload budgets and target MCP host | Proposed budgets / open client evidence | Calibrate with end-to-end tasks and the available host. |
| Completion/wake integrations | Optional | Durable retrieval first; one qualified adapter; no universal guarantee. |
| Branding, name, redistribution terms | Open human decisions | Can use a working name now; resolve terms before public distribution. |
| Extra codecs, dictionaries, stems, captions, formal approval locks | Open additions | Add for demonstrated workflow need; not automatic release gates. |
| Remote GPU orchestration, native dictation, realtime duplex, SaaS | Deferred | Do not build infrastructure for them in v1. |

## Practical delivery stages after the plan

These are milestones for subsequent authorized implementation. They are not work to execute during this review, and not prerequisites to writing the plan.

| Stage | Concrete outcome | Decision/evidence required to proceed |
| --- | --- | --- |
| 0. Product plan and bounded technical decisions | Consolidated scope, journeys, requirement/choice register, legacy feature mapping, first release matrix, and acceptance tiers. | Complete now from existing evidence. Record defaults and open qualifications; no new broad research or model run. |
| 1. Release-shaped walking slice | Lightweight packaged service, fake worker, persistent job/artifact, compact HTTP/CLI/MCP fixture; one browser script/timeline round trip with synthetic media. | Decide SQLite executor boundary and editor component against restart, conflict, missing-media, overlap and reload cases. Review Voicebox on paper only unless a specific saving justifies a conditional slice. |
| 2. Real voice work through the shared service | Existing voice import; one real Qwen path; explicit worker provisioning; microphone/file/remote acquisition; clone preparation; built-in/design flows; candidates and reusable voice versions. | Capability preservation, staging/publication and cancel/reap evidence; bounded TTS profile gaps and human audition. Temporary typed CLI subprocess adapters are acceptable bridges; replace expensive orchestration where residency or structured errors require it. |
| 3. Complete speech production | Script/speaker/take workflows, imported beds, full bounded timeline, history/conflicts, targeted regeneration, preview, asynchronous render/export. | End-to-end human/agent handoff; edits cause no unrequested synthesis; exact render fixtures and real browser/listening acceptance. An early browser slice from stage 1 prevents postponing editor risk until this stage. |
| 4. Product qualification and release | Clean install, resource/readiness UX, supported profiles, upgrade/restore, access/media boundaries, compact agent journeys, optional completion adapter and practical handoff/export. | Measured first/returning agent success and cost, real install recovery, supported-host evidence, visual/accessibility and acoustic approval. Publish only qualified promises. |

Use small vertical lots rather than separate large backend, UI and packaging projects. Packaging and basic UI appear early; richer workflows and qualification mature together. Avoid waiting for every TTS model or every hardware family before testing the shared product. Do not release the early service slice as fulfillment of the promised full studio.

## Main risks and necessary human input

The main execution risks are editor integration and long-project browser behavior; extracting reusable orchestration from large CLI modules without regressing voice semantics; realistic cloning/design performance on supported hardware; reliable publication/recovery; install/provision disk and dependency footprint; and scope growth from every researched feature becoming mandatory. Name them in the plan with the milestone that resolves them.

No unresolved human question blocks planning. Engineers can choose the initial database, process layout, framework, component evaluation and conservative scheduling defaults with stated reasons. Later, bundle only the decisions evidence cannot answer: product/name preferences when visual work starts; which physical hosts and agent clients are available for certification; acceptable perceived voice/workflow quality on actual examples; and redistribution terms before distribution. Hardware unavailable for testing stays unqualified rather than blocking all progress or provoking a speculative questionnaire.

**Final disposition:** write the full product plan now around TTS and the shared production studio. Carry a clear recommended implementation direction with bounded qualification milestones. Do not require Voicebox adoption, competing studio prototypes, STT qualification or TxCode integration to begin or complete that plan.
