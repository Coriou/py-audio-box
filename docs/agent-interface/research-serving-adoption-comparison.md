# Serving reuse and whole-product adoption: integration map

Evidence date: **2026-10-15**. Bounded source review; no models, inference, application installs, private audio or runtime serving tests. This lane wrote only this report in the repository and downloaded a small pinned LocalAI source subset into a task-owned temporary directory. Existing editor probe results are attributed to the parent run, not rerun here.

## Decision at this boundary

Compare **Voicebox adaptation** with **one service plus a reusable editor component** using the same production fixtures. Treat **LocalAI or another inference server as an independent worker choice**. Neither a whole studio nor a model server removes the need to choose one authority for voice identity, immutable assets, production revisions and durable work.

LocalAI is a credible serving-reuse candidate: inspected code implements actual reference profiles, Qwen clone/design/custom modes, STT request controls and model lifecycle management. It is not an already qualified replacement for the repository's request validation, process supervision or proposed durable production contract. Begin qualification with a narrow STT adapter or one explicit TTS mode; do not replace every existing backend at once.

No product, serving backend, model or default is selected by this report. Source parity, runtime compatibility and audio quality remain different evidence tiers.

## Source identities

* LocalAI: `1d0a3227d7c48ba099f49c341dda39d73b583e99`. Read-only public subset saved under `/var/folders/gk/6rj578d95ls341449hb55pj80000gn/T/pab-localai-source-6sppzvam`. Ten files, about 142 KiB; no build/install scripts executed.
* Voicebox: `8af7efe62fab8d33e2a5dfbedabaa45f68a5184f`, supplied snapshot `/tmp/py-audio-solutions-audit.6vhA7k/voicebox`.
* VoiceStudio: `8beed7e3df1983776658d03e7e4108e49ef385b1`; its relevant STT/queue evidence is detailed in [STT contract research](research-stt-agent-contract.md).
* Editor upstream source: `772328046f7f3f191e82e43b56a4b3025ae417a0`. Parent probe actually imported published `@waveform-playlist/engine@13.6.0` and `@waveform-playlist/core@12.6.1`; distribution hashes/integrity are separate identities. [Probe receipt](evaluation/editor/README.md), [results](evaluation/editor/result.json).

## What LocalAI implements, and what remains above it

| Concern | Inspected implementation | Integration consequence |
| --- | --- | --- |
| TTS input | `core/http/endpoints/localai/tts.go` consumes a structured speech request, resolves model config, language, voice, instructions and backend parameters. | Common fields exist; backend-specific parameters still require an allowlisted typed domain schema. |
| Reference voices | The same route parses profile references, checks cloning support, leases reference audio and injects transcript as `ref_text`; multiple references are selectively supported by backend. | Real reuse opportunity. Choose canonical profile ownership; do not keep two independently editable voice libraries. |
| TTS result | Streaming writes WAV chunks; nonstreaming generates a file, optionally resamples/converts it and returns an attachment. | This route's result is bytes, not our durable job/take/artifact manifest. Output conversion must be explicitly planned. |
| Voice discovery | `tts_voices.go` advertises named voices from installed model configurations, filtered by model/authorization. | Catalog metadata is not proof of loaded readiness or checkpoint-compatible controls. Add domain capability/status projection. |
| STT | `openai/transcription.go` reads upload, language, translation, prompt, temperature, timing granularity and streaming; creates a temporary source and invokes the backend. | Useful request-level adapter, not independently durable transcription/resume. Return/store evidence in our transcript model. |
| Runtime | `pkg/model/loader.go` and `watchdog.go` track/load/evict processes and requests. | Serving lifecycle can be delegated, while durable attempts, reservations, fencing and terminal publication stay with the control owner. |

Primary source: [TTS route](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/core/http/endpoints/localai/tts.go), [voice discovery](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/core/http/endpoints/localai/tts_voices.go), [STT route](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/core/http/endpoints/openai/transcription.go), [loader](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/pkg/model/loader.go), [watchdog](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/pkg/model/watchdog.go).

### Defaults and semantics an adapter must freeze

The inspected Qwen Python backend implements `generate_voice_clone`, `generate_voice_design` and `generate_custom_voice`. It strips input text; resolves per-request instruction with YAML fallback; selects mode from model type/configuration; and uses a default CustomVoice speaker if none is provided. Generation overrides arrive as string values and are coerced. Reference cache code stores pickle-based runtime caches alongside reference paths. These are implementation facts, not a remote-code-execution claim. Our exchange format must remain WAV/JSON only, with private runtime caches outside portable voice assets. [Qwen backend, especially lines 350-384 and 690-940](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/backend/python/qwen-tts/backend.py).

Separate Qwen backends are not drop-in aliases: the inspected `faster-qwen3-tts` implementation is clone-oriented and defaults missing language to English. The C++ backend forwards language, instruction, speaker/reference and sampling controls, but this review did not qualify its checkpoint/mode equivalence or CPU quality. Pin the backend implementation as well as the model. [Faster Qwen backend](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/backend/python/faster-qwen3-tts/backend.py), [C++ request resolver](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/backend/go/qwen3-tts-cpp/goqwen3ttscpp.go).

STT defaults also matter: the route derives `diarize` from whether the form value is literally `false`; missing/malformed temperature is tolerated; language/translate resolve against request/config defaults. An adapter must explicitly send and validate its intended policy, preserve the selected backend/result identity and reject unavailable timing/context controls. Compatible field names do not prove recognition semantics. [Pinned STT implementation](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/core/http/endpoints/openai/transcription.go).

Current memory documentation describes LRU, concurrency groups and idle/busy watchdogs. It also states that pinned models can defeat group conflicts and that exhausted group-eviction retries may proceed with a warning. These mechanisms must not be treated as a hard combined-memory admission limit. Keep canonical reservations/serialization above serving, and test the pinned implementation. [Lifecycle contract and caveats](https://localai.io/docs/advanced/vram-management/index.html).

## Comparison with existing py-audio-box supervision

`lib/tts_backends/qwen.py` already checks checkpoint mode/direction, owns a spawned model process, bounds calls and performs terminate/kill/join, disabling a backend after failed reap. `apps/voice-cpu/voice-cpu.py` owns homogeneous batch workers, includes first load in deadlines, advances worker generations and publishes staged outputs. These are valuable behaviors to preserve. They are not yet the complete durable service, but replacing them with a network timeout alone would regress ownership guarantees.

The current Qwen contract rejects unsupported direction rather than guessing; for example, it distinguishes clone/Base, design and CustomVoice checkpoint capabilities. Keep that validation above LocalAI and record resolved model/backend/instruction/defaults before admission. Preserve reference hashes and exact supplied text even if a backend's synthesis transform trims it. See `lib/tts_backends/qwen.py:17-46,150-288` and `apps/voice-cpu/voice-cpu.py:296-429`.

If LocalAI owns a model process, the domain worker should not also spawn a second owner for that same request. Instead it owns a bounded request attempt, tracks the serving instance and backend identity, stages returned audio, and commits only under its current fence. On timeout, verify serving cancellation/termination or quarantine the resource lane before replacement. Late bytes never authorize publication. An HTTP disconnect alone proves neither model stop nor safe retry.

## One authority map for either whole-product path

| Canonical object | Voicebox adaptation path | Service + component path | LocalAI role, if selected |
| --- | --- | --- | --- |
| Voice ID/revision | Migrate/extend Voicebox profile identity into the chosen domain contract; keep its UI editing through that owner. | Import the existing registry non-destructively into service IDs/revisions. | Derived backend profile/reference mapping. No independent user edits or identity reassignment. |
| Source/audio identity | Content-addressed domain asset store; bridge existing paths/history through an import manifest. | Same domain asset store. | Read-only staged references and attempt outputs; serving filenames are not canonical asset IDs. |
| Project/script/timeline revisions | Add revision/CAS semantics to existing story mutations and front-end operations. | Build the domain model and adapt component commands. | None. |
| Job/attempt/reservation | Replace/bypass in-process callable dispatch with the domain executor; keep one public status authority. | Implement the planned durable executor. | Executes one bounded backend request. Runtime activity is not the production job ledger. |
| Runtime model residency | Qualified existing backend or server behind scheduler. | Existing supervisors or qualified server behind scheduler. | Sole process/model owner for requests delegated to it; expose observed status only. |
| Transcript/raw/derived versions | Project/library transcript objects and source maps must be added or reconciled. | Shared transcript objects as specified in STT report. | Returns recognition/diagnostics; does not decide editorial cleanup or identity labels. |
| Takes and selection | Preserve existing generation versions; explicit adoption must respect project revision/locks. | Domain take library plus explicit selection. | Supplies candidate audio, not a selected take decision. |
| Renderer/export | Replace or reconcile current exporter to canonical composition/presets. | Canonical renderer; component previews the same contract. | None: a speech endpoint does not mix a timeline. |
| Agent defaults/events | Preserve useful client bindings, but route all mutations/status through owner. | Add bindings and adapters over shared API. | Private service credential and backend refs; no separate agent authorization authority. |

Alternative: make LocalAI's voice store canonical and project immutable versions around it. That would be a different architecture requiring explicit migration, source identity and version semantics. It is not a harmless convenience toggle. For the current trusted-team plan, derived backend mappings are the smaller qualification surface.

## Whole-product path A: adapt Voicebox

**What it saves:** existing voice/profile workflows, generation history, story/timeline UI, capture/agent integrations and practical playback behavior. The pinned story service already resolves pinned generation versions; preserve that useful work. [Story source](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/services/stories.py), [prior product audit](audit-build-versus-adopt.md).

**Required integration:** introduce/extend the canonical domain owner inside or behind its backend; adapt front-end calls to revision-aware commands; migrate existing profiles/assets/version relations; replace callable queue semantics with durable attempts; expose the compact API/provider adapters; and isolate model ownership. Extracting a few React components is a different route from adopting the product and must be costed separately.

**Observed contract differences:** its queue is in-process work; speaking events are bounded activity pub/sub. Story export uses millisecond placement, skips unavailable/unreadable clips and peak-normalizes an over-range mix. Those conflict with our strict missing-source and declared-headroom rules. They are observable source differences, not evidence that the product fails its own intended desktop contract. [Queue](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/services/task_queue.py), [events](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/mcp_server/events.py), [export around lines 964-1100](https://github.com/jamiepine/voicebox/blob/8af7efe62fab8d33e2a5dfbedabaa45f68a5184f/backend/services/stories.py#L964).

The parent's isolated export-function probe now provides execution evidence beyond inspection: the actual pinned function body, unchanged after AST extraction, ran with mocked database/media inputs and synthetic PCM in a network-disabled 1-CPU/512-MiB container. It reproduced missing-clip skipping and peak normalization; zero gain behaved as default gain, while half gain worked. These are narrow function-level observations, not installed-app/API/browser or acoustic results. Fixing zero gain is a small patch and is not by itself a reason to reject adaptation. [Probe scope and receipt](evaluation/voicebox/README.md), [recorded results](evaluation/voicebox/result.json).

**Not established:** server-wide optimistic revisions, restart-safe executable resumption, late-result fencing, preview/export acoustic equivalence, low-token complete agent workflows, and old-Intel performance. The inspected PyTorch Whisper path is not a substitute for CPU int8/quantized engine evaluation. A friendly desktop flow does not settle unattended Docker lifecycle.

**Upstream burden:** keep a focused delta around domain ownership, queue/worker boundaries and render policy; document rejected upstream defaults. If upstream frequently changes those same seams, adaptation becomes ongoing product-fork maintenance. Do not estimate savings from screen count alone.

## Whole-product path B: service plus editor component

**What it saves:** reusable timeline mechanics, waveform rendering and audio UI building blocks while keeping canonical backend semantics explicit. **What remains:** the actual script/voice/take workflows, command adapter, persistent undo/history, project state, accessible UI, asset loading and qualified render/preview parity.

The parent's published-package headless probe ran 11 checks successfully, some intentionally confirming differences: same-track move clamps overlap although ingestion accepts it; `setTracks` clears local undo; `getState` shares clip objects; splitting generates client IDs. These are actual metadata-engine observations. They are not browser, decoded-audio, performance or acoustic checks. [Exact probe evidence](evaluation/editor/README.md).

The adapter must therefore validate desired edits against the server before committing; map split IDs; clone/immutably handle boundary state; and rebuild client history around canonical revisions. Never accept a server overlap while the browser silently clamps it. If a component's edit policy cannot express intended overlaps, change the adapter/component or choose another component; do not silently change production semantics.

**Upstream burden:** fewer product-wide dependencies, but ownership of the editor integration and all missing production UX. Pin distribution versions separately from upstream source. A small engine is not a whole editor, and headless tests do not prove playback.

## Smallest real qualification slices

Use one public/synthetic project fixture and mock worker protocol for both UI paths; do not run two model stacks concurrently. The designated CPU benchmark owner supplies separate inference receipts.

1. **One authority round trip:** import one legacy voice/reference and two takes with unchanged digests; create a project, have agent and UI race a revision; one edit conflicts; select a take and retain exact script/voice identity. Restart and reload. Both paths must show the same IDs and state.
2. **One durable job slice:** mock submit → claim → staged WAV → fenced publish → completion outbox. Crash before/after publication, time out one worker and deliver its result late. Verify one accepted result and no second mutation from an expired attempt. LocalAI candidate uses a mocked request adapter at this stage.
3. **One composition slice:** synthetic impulses/tones with trim/split/fades/overlap. Reject missing asset, render pinned revision, verify sample positions/headroom and clip source mapping. Compare browser preview separately. For Voicebox, disable/replace implicit skip/normalization behavior; for component path, prove no overlap clamp or ID drift.
4. **One STT serving slice:** permitted public FR/EN/mixed audio with explicit language/context/timing controls; inspect actual sent request and returned backend identity. Compare current direct engine with LocalAI adapter on the same declared engine/model when possible. Test unavailable model and unknown controls before inference, silent/no-audio inputs and interrupted request. A different engine is a separate comparison, not “server overhead.”
5. **Three Qwen capability slices:** CustomVoice with supported instruction; Base clone with exact reference/transcript; VoiceDesign preview plus deliberate save. Include unsupported instruction/mode and mismatched checkpoint rejection. Mock propagation first, then one bounded authorized acoustic run each. Verify no default speaker/language/mode substitution and no untrusted cache import.
6. **One switch/lifetime slice:** TTS → STT → TTS under the actual Docker RAM cap, measured load/idle/switch costs and cancellation. Serving policy must not start a second heavy owner because eviction retries expired. No performance forecast precedes this test.
7. **One agent journey:** incoming voice through a provider hook without extra model tool turns; deliberate transcript range/search; targeted line regeneration; pinned render and completion retrieval. Measure context bytes/tokens and recovery, not only endpoint success.

Each slice produces input IDs/digests, software/model identity, requested/resolved settings, source/static versus runtime versus acoustic result, unresolved differences and approximate integration delta. Use those receipts to choose the next bounded implementation lot. No feature-count score overrides a violated ownership or evidence contract.

## Practical recommendation

Continue the two-path studio comparison; keep the custom service contract as the comparison fixture rather than a foregone implementation. Qualify LocalAI as a replaceable serving adapter, initially without replacing the existing Qwen defaults/supervisor. Its broader backend catalog is useful only where it reduces measured integration/maintenance work while preserving declared controls and isolation.

An old Intel Mac can evaluate CPU serving paths, but the current Docker VM's memory and actual engine builds decide fit. GPU acceleration is a separate optional profile on a demonstrated capable host. Avoid an all-in-one image that downloads models on discovery or silently changes execution backend; provision selected pinned profiles deliberately and report actual readiness.

This source review establishes concrete integration seams and observed contract differences. It does not establish runtime/acoustic parity, final effort, a winning renderer, or readiness to fork/adopt a whole product.
