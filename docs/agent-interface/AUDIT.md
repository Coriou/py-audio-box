# Existing-solutions audit and implications

Reviewed synthesis, 2026-10-15. Primary-source research plus selected public-source inspection; no product account sessions, audio generation, model downloads or runtime comparison. Astra source-audit review: GO for bounded build-versus-adapt comparison and contract/prototype; no adoption or runtime certification. See [review-solutions-astra.md](review-solutions-astra.md).

## Question and evidence

Can an existing product meet this project's confirmed self-hosted, agent-first, full-speech-studio requirements, or supply capabilities/components worth adopting? Separate documented behavior, selected source evidence and untested claims. No numerical quality/latency/cost ranking is valid without matched runtime evidence.

Research tracks:

- [Whole-product build/adapt/adopt assessment](audit-build-versus-adopt.md), with pinned Voicebox and VoiceStudio source snapshots.
- [Self-hosted serving and voice management](audit-self-hosted-solutions.md).
- [Commercial voice and agent APIs](audit-commercial-agent-solutions.md).
- [Studios and reusable editor components](audit-studios-editor-components.md).

## Comparison at the decision boundary

| Family / candidate | What it can teach or supply | Decision for this project |
| --- | --- | --- |
| Voicebox | Whole local studio, takes/timeline, agent defaults, explicit playback | Shortlist whole-studio adaptation; qualify durable work, revision semantics and renderer behavior. |
| VoiceStudio | Headless discovery, voice tools, longform preview/resume, export | Secondary whole-product/reference candidate; different license and project ownership need evaluation. |
| LocalAI | Serving, managed voice profiles, discovery, model lifecycle | Shortlist backend reuse; fail-fast controls and one voice authority required. |
| Kokoro-FastAPI / Speaches | Focused warm speech serving, compatibility endpoints | Conditional focused-lane comparison, not a whole production replacement. |
| VoiceMode | Voice conversations and practical MCP integration | Client/integration reference, not the production authority. |
| ElevenLabs / Cartesia / Hume | Pronunciation, continuity, candidate/acting controls, artifact and agent ergonomics | Adopt supported contract patterns; no paid-provider dependency or quality ranking. |
| Azure / MiniMax | Async longform handles, sidecars, task/artifact distinction | Learn batch/transfer semantics without requiring their hosting. |
| Descript and studio products | Real agent/API handoff plus useful production workflows | Learn selective editing; verify typed edit/export parity and publication requirements. |
| Waveform Playlist / wavesurfer | Headless arrangement primitives versus narrower waveform/audition tools | Qualify Waveform Playlist core/engine first; keep server-render/CAS contracts independent. |

## Findings and disposition

1. Whole-product overlap is substantial. Voicebox is a plausible studio adaptation candidate, not merely a synthesis server. VoiceStudio offers further headless/discovery/agent patterns. Assess them before committing to a bespoke editor.
2. Voice bindings per agent can remove repeated voice selection. Bindings must attach to authenticated caller identity and resolve explicitly in each immutable plan; client labels are not credentials.
3. Generate, download, and play are distinct intents. Headless/local/private-server deployment must never assume that audio playback on the server reaches the user. A client playback adapter requires an explicit authorized target.
4. Queryable history, serialized execution, durable executable work, fenced publication, event replay and agent resumption are different capabilities. Inspected queues illustrate why a combined 'async/SSE' checkmark would be misleading.
5. Adopt useful patterns without accepting hidden behavior: the inspected Voicebox story exporter skips missing audio and may peak-normalize the mix, which conflict with our strict render/preset contract.

## Incorporated plan additions

Make pronunciation rules and synthesis-only text normalization explicit, versioned and capability-aware. Preserve exact authored text. A dictionary change invalidates only affected takes; unsupported phonemes/controls must not silently disappear.

Represent longform continuity context in generation plans where supported, binding neighboring text/take IDs and exposing invalidation scope. Do not imply that all current engines support request stitching or emotion controls.

Make reference sources, voice samples/versions, audition candidates, selected takes and human approvals distinct. Approval attaches to an exact take/revision; it does not imply that a voice is legally verified or every future generation sounds correct.

Add a cheap preflight plan with resolved inputs, readiness, estimated work, uncertainty and enforceable ceilings. Text/chunk expansion, optional alignment/QA, variants and retries must count. Estimates are not guarantees or billing facts.

Add clear portability contracts for project manifests, selected audio/stems and optional timing/chapter/subtitle outputs, without promising every format in v1. Keep stable artifact IDs separate from expiring access grants and remote filesystem paths.

## Build-versus-adapt gate

Before large implementation, compare the strongest whole-studio adaptation with a focused shared service plus a reusable timeline component. Include serving-backend reuse where it reduces work without creating another authority. Use the same fake-worker/synthetic-audio fixtures, preserve existing defaults/voice identity, and compare integration effort, license/dependencies, lifetime upstream burden, context cost, shared-state semantics and resource isolation. No fork or library choice is made by this document.

## Scope discipline

This audit is not authority to add paid speech providers, switch default engines, adopt models, introduce a local text-rewriting LLM, implement global OS dictation, real-time duplex calls, a plugin effects graph, dubbing/video or multi-tenant SaaS. Useful market features may become explicit deferrals. The original motto describes a goal, not proven differentiation.

## Final scope triage after Astra

**Contract-stage additions with direct v1 value:** explicit per-principal voice defaults; preflight and enforceable work ceilings; versioned pronunciation/normalization policy; exact take review/selection locks and provenance; explicit generate/download/playback distinction; canonical capability and artifact-transfer semantics. These should be shared UI/API operations, not transport-specific features.

**Represent now, activate only for a verified backend:** neighboring-text/audio continuity inputs, phoneme markup, emotional/performance controls and speech-to-speech reference performance. Unsupported controls fail explicitly. Current backend coverage must be audited before any v1 implementation promise.

**Cheap portability to evaluate for v1:** a project manifest plus referenced selected audio and source mapping; existing alignment/timing export where explicitly requested. Caption/chapter/stem interchange formats require a small supported-format matrix, not every export format at launch.

**Defer:** universal real-time voice conversations, native global dictation, telephony, dubbing/video, LLM personality rewriting, effects/plugin ecosystem and paid-provider integrations. Preserve extension points without loading these schemas by default.

**Decision gate rather than a new dependency:** compare adapting Voicebox with a service-plus-component approach using the same fake-worker/synthetic production. LocalAI belongs in the backend comparison. No complete-product adoption or bespoke-editor commitment is justified by this source audit alone.

## Review disposition

[Astra review](review-solutions-astra.md) found this audit sufficient for the bounded decision, with no new human-only question needed. Its refinements are incorporated into PLAN.md and ACCEPTANCE.md: orthogonal serving choice; no mandatory billing ledger/preflight round-trip; version-pinned dictionary adoption; dependency-closure invalidation; playback/transfer distinction; minimal manifest/media/mix portability. Independent studio suggestions for mandatory stems/captions/comments are narrowed to a qualified format matrix and later workflow work.

The audit changes our next step: compare actual adaptation paths before committing to a custom studio. It does not establish acoustic superiority, measured token savings, runtime adoption readiness or a new default backend. No purchase, fork, dependency installation or provider integration was performed.

## Subsequent STT and portability research

The user expanded the comparison to CPU-first STT, contextual transcription and a possible Wispr-like sibling. See [STT-AND-PORTABILITY.md](STT-AND-PORTABILITY.md) for that synthesis, five source lanes and bounded empirical receipts. The prior source-audit GO does not certify the new runtime candidates. It remains valid for the contract/prototype boundary while actual CPU feasibility and native-client integration are separately evaluated.

[Final STT/portability Astra review](review-stt-portability-astra.md) keeps the contract/prototype GO, supports bounded core STT, and withholds CPU starter, universal-hardware, sibling-product and adoption promotion.
