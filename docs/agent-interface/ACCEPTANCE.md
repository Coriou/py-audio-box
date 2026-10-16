# Agent and studio acceptance scenarios

Scope update (2026-10-15): the phase/release gates in [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) are authoritative. This earlier catalogue contains candidate engineering/detail checks, not an extra mandatory release list. STT, transcription, contextual dictation and TxCode integration checks are deferred-upgrade research. V1 remains TTS/voice management/full speech studio through HTTP, CLI and MCP. The [fresh review](review-fresh-product-astra.md) distinguishes core requirements from optional additions.

Proposed release gates, 2026-10-15. Not measured results. These scenarios turn the product plan into reviewable outcomes; budgets are candidates for contract-stage calibration.

## Measure the whole agent task

Use the same frozen request, voice assets, engine, hardware, and agent model for baseline CLI and proposed interface runs. Record initial instructions/schema tokens, total request/response tokens, number of calls, active model turns, recovery turns, wall time, inference starts, model loads, generated audio duration, CPU/GPU memory, and final success. Count retries, discovery, and completion handling; excluding setup/schema costs would hide the problem this product is meant to solve.

Measure at least: first connection to first speech; returning agent using a known voice; clone from an uploaded reference; creating a designed voice; a multi-speaker production; correcting one line; failed job recovery; a long job using an available completion integration. Test both a minimal-context agent and an agent with an established session. A schema being short does not establish task success or low total usage.

Proposed control-plane budgets: zero inference imports/model loads/downloads for liveness, basic discovery, voice listing, and job status; a receipt at most 2 KiB; default voice list page at most 20 items; no inline audio or unrequested logs; capped error detail; no mandatory status polling loop. Establish a total exposed-tool-schema token budget using a named tokenizer and actual client behavior before choosing tool count. Verify limits for non-ASCII IDs/messages too.

## Voice and speech

A fresh agent lists ready voices, generates a short utterance, and retrieves audio using only documented public operations and a compact onboarding instruction. It never needs repository files, internal model paths, or Redis credentials.

A user uploads or records a reference, auditions candidate segments, selects one, prepares a voice, and synthesizes a test. Preparation and optional test synthesis are separate explicit costs. An agent can continue the same draft by ID. Reference hashes and provenance survive reload/export/migration. Unsupported engine controls fail before queueing with a short corrective action.

## Studio and agent handoff

An agent creates a project with three speakers and 30 script blocks. A person opens the same revision, auditions/selects takes, adjusts a clip and pause, then saves. The agent edits one block with a current revision precondition. For independent generation, exactly that block becomes stale; when explicit pronunciation/context dependencies exist, mark only their exact dependency closure stale. Existing audio remains playable; invalidation does not automatically start regeneration. A stale edit request receives a conflict and scoped change information, rather than overwriting human work.

A person's clip move, trim, gain/fade adjustment, or take selection creates zero inference calls. Fixed and ripple edits have explicit semantics. A new take longer than its previous take cannot silently push later fixed clips. The UI exposes overlaps/stale clips and allows recovery through undo/history.

## Render correctness

Render an immutable project revision while another client edits the draft. The resulting artifact and manifest refer only to the captured revision. Synthetic impulses and sine waves verify offsets, trims, fades, gain, overlap, channel conversion, silence, and exact output sample count. Browser preview and exported render follow the same composition rules. Real listening separately verifies cuts, intelligibility, identity and delivery suitability.

## Durable work

Submit the same operation twice with one idempotency key: one logical job. Reuse the key with changed content: conflict. Request another take intentionally with a new request/seed: a distinct operation. A job captures voice revision and engine asset identity so later registry changes cannot alter queued work.

Exercise restart during model load, synthesis, render, and final publication; expired leases; late worker replies; cancellation at each stage; disk full; malformed media; lost client connections; and partial batch failure. At most one artifact set becomes authoritative, originals remain intact, worker cleanup follows confirmed termination, and successful partial outputs remain accessible.

## Completion and integration

With no completion target, no outbound notification request occurs. With a registered target, terminal events are durable, signed, deduplicable, retried within bounds, and visible in delivery history. A failed callback never makes successful audio fail or reruns synthesis. Reconnect with an event cursor replays missed transitions. A client without wakeup support can use bounded wait or durable status retrieval. Verify actual wakeup behavior for each advertised integration; server delivery alone is insufficient evidence.

## Installation and operation

On each declared supported platform: follow only the release installation guide, start the pinned images, provision required assets explicitly, connect through HTTP/CLI/MCP, create a voice/project, render, restart, upgrade, back up and restore. No source checkout, host Python, or developer build is needed for the release path. Network-isolated inference is tested separately from successful provisioning. Record image digest, asset identity, architecture, runtime, device and storage location.

Measure lightweight API idle usage independently of worker/model memory. Test cold and warm execution, explicit idle unload and resource ceilings without asserting quality or universal speed. A missing model, incompatible GPU, unavailable disk or inaccessible remote import produces a useful readiness state and remediation rather than a generic healthy indicator.

## Human acceptance

Use keyboard and assistive technology for script editing, take choice, job actions and clip timing controls; essential operations cannot require drag alone. Test large scripts and long recordings with bounded waveform data and rendering. Verify secure-context microphone behavior on the intended local/private-server deployment. Human approval of the workflow, visual design, and acoustic results remains separate from protocol/unit/Docker checks.

## Astra R1 additions

Fault-inject the sole SQLite owner's submission, claim, adoption, commit and outbox boundaries; reconcile orphan content without missing successful deliverables. Test dry-run/apply migration preserving legacy aliases/bytes and rejecting concurrent mutation of one registry. Restore a consistent DB-plus-referenced-files backup.

Verify networkless inference and socket IPC in the actual Compose profile, worker denial of DB/canonical-publish access, typed fetch-helper requests, failed provisioning preserving the previous active bundle, and admission limits across decode/render/provision/inference. Demonstrate model reuse across jobs without model-affinity starvation.

Use 48,000-tick end-exclusive fixtures covering split, explicit ripple, shorter-take adoption, stale selections, unresolved placeholders, missing sources and partial-render omission reports. Verify fixed placement, explicit ingest transformations, preset headroom and preview/render timing tolerances. Store undo as a new revision.

Exercise expiring one-use bootstrap, token scopes/revocation, browser CSRF/Host/Origin checks, authorized artifact reads, callback address policy, and the stdio-to-HTTP MCP path. Direct remote MCP and every advertised wake adapter need separate client-specific evidence. Reconcile provisional budgets to PLAN.md: catalogue plus onboarding ≤2,500 tokens, status ≤1 KiB, receipt ≤2 KiB, default lists ≤20 entries. These remain proposed gates, not results.

## Existing-solutions audit additions

Compare candidate adaptation/component/backend paths on the same synthetic production and fake worker; record actual adapter/upstream burden and fail contract mismatches explicitly. Pin canonical repositories/releases/licenses separately from model/assets. A feature not verified by inspected sources is not marked absent.

Preserve source whitespace, Markdown, emoji, French/English names, numbers/dates/currency, acronyms and literal instruction-like brackets. Test opt-in dictionary/normalization versions, unsupported phonemes, deterministic matching priority and affected-only invalidation. Alignment must reference the actual spoken-text transformation, not assume authored character indices still apply.

Preflight a bulk correction, show changed/locked/reused work, then alter project/defaults before submission: stale plan fails. Concurrent jobs cannot each consume the same configured admission allowance. Variants, QA and retries count against explicit limits; stop before another attempt exceeds the bound. Estimates stay labelled uncertain, and no financial ledger is required.

Bind two authenticated agents to different voices and test resolution/override without spoofable client-label authority. Generate candidates beside a locked selected take, then finish a stale job: selection remains. Distinguish creating a new take from playing existing audio; headless playback requires an explicit client. A portable bundle roundtrip retains revisions/selected media/source mapping without engine pickle caches. Test artifact IDs independently of download-grant expiry and remote filesystem namespaces.

One-shot known-voice speech requires no mandatory preflight tool call: server-side plan derivation/validation applies existing ceilings. Explicit bulk preflight stays inspectable. Distinguish authored script, synthesis transformation and ASR transcript; selecting a take for older text exposes the mismatch and stale policy. Speaker roles survive reassignment to another voice version. Component qualification checks headless edit/CAS mapping, browser memory, keyboard alternatives and server-render parity rather than just waveform display. Stems, captions and transcript-editing are not implied mandatory exports.

Creating a dictionary revision leaves projects pinned to older versions unchanged; explicit adoption invalidates a reported dependency closure, conservatively when needed. Distinguish backend-internal normalization uncertainty from application transformations. Portable exports include the actual rendered mix and fail clearly on missing required media or unsupported manifest versions. Reuse evaluation uses at most two small synthetic/fake-worker slices with a declared effort ceiling and one selected-path decision record; component browser state never becomes a second canonical project store.

## CPU-first STT and provider integration

Qualify French, English and genuine mixed-language speech on declared profiles; a concatenation of monolingual clips is a mechanics fixture, not natural code-switch evidence. Keep public/consented reference transcripts independent of model output. Report raw/normalized WER/CER and critical name/number/negation errors by stratum, plus no-speech insertions; tiny read-speech tests do not certify dictation or choose a default.

Test no audio stream, corrupt decode, empty recognition, quiet speech, silence/noise and partial failed ranges separately. Bounded language/vocabulary/context must propagate to the backend or fail visibly; test unsupported/dropped hints and false glossary insertions. Preserve raw-ASR and derived correction provenance, native language codes and estimated source-time segment maps through VAD. ASR timestamps never claim forced alignment of authored script.

For advertised multi-window profiles, restart transcription after committed chunks: unchanged ranges are reused; gaps are explicit; no duplicated boundary words, silently removed real repetitions or compressed timestamps after silence filtering. Search/range/file transcript access stays within the same token budgets as other resources. Context text cannot grant tools/permissions to the service or overwrite an agent's privileged instructions.

Use a provider-hook fixture for incoming voice: host submits before the language-model invocation without requiring an extra model tool turn; deliberate MCP transcription remains separately available. A desktop client uses scoped authentication and an explicit capture/insertion target; no server mic assumption, hidden cloud cleanup or silent command execution.

Measure TTS → STT → TTS load/switch/evict costs and shared host resource admission/fairness under the Docker VM cap. Do not launch one full-budget owner per backend. Record requested/effective model, compute type, device, threads, cold/warm timing, RTF, container/process memory method and failures. Validate CPU-only install first, then each declared architecture/ISA/GPU profile with pinned artifacts, clean provisioning/offline inference, no-download doctor and restore. Current Mac UDS smoke proves only its named-volume transport, not Windows or production IPC auth.

A bounded first release may refuse media beyond its measured maximum duration/size rather than implement hour-scale checkpointing. Test the advertised cap before model work; long-file mechanics are required only for accepted profiles. A native compatibility shim must retain scoped authentication, transcribe-only routing, Host/Origin protections and no arbitrary URL/path forwarding; localhost binding alone is not its authorization model.

## Contextual dictation and TxCode integration target

Use [TxCode issue #8](https://github.com/Coriou/txcode/issues/8) as a concrete client requirement, not a chosen hosted backend. Test short auto-language FR/EN and genuine code-switch dictation with exact project names, symbols, paths, acronyms, negation and values. Compare identical audio under no context, recognizer hints and composed terminology resolution. Record critical-token accuracy, user corrections/keystrokes, time to reviewable draft, false candidate substitutions and processor/recognizer resource cost. Raw WER alone does not define useful dictation.

Provide correct, irrelevant, stale, ambiguous and adversarial context packs. Preserve raw ASR and derived spans, but evaluate the final proposed text. No translation, invented task, altered negation/amount or forced replacement with an expected thread term. Unsupported hints are visible; uncertainty remains unresolved rather than confidently invented. A processor's better wording is not evidence that the recognizer heard it.

The composed API works through one job without extra mandatory model tool turns or full-thread/audio forwarding to a text processor. TxCode/client tests: release microphone, stop/cancel, review-before-send, caret selection and current draft revision; a late response never overwrites a changed draft or submits a turn. Bound context scope and record its revision/source. Establish the CPU/runtime path before claiming local contextual-dictation latency. No native-client or hosted-provider integration has been executed in this research pass.

Contextual failure tests reject unsupported policy before admission or return explicit partial/failed stages; raw-only fallback requires caller opt-in and cannot auto-insert or silently substitute a cloud/model. Verify ephemeral retention independently for audio, raw/resolved text and context, including cache/log/backup handling and cancellation/expiry deletion. Provenance must respect retention rather than force archival storage.

## CPU setup lessons from hands-on debugging

The [debugging follow-up](evaluation/stt/DEBUGGING.md) supplies a local CT2 placement workaround and a repaired AVX2 whisper.cpp build, not release qualification. A CPU profile must declare its instruction-set requirements, avoid unintended `-march=native` in distributed binaries, and pass a model-load plus transcription-artifact smoke on the actual target. Architecture labels and exit code zero alone are insufficient. Verify supported CLI flags, bounded client/container supervision, stopped receipts before cleanup, and actual worker placement/throughput. Pull deadlines must account for measured transfer and canceled-layer behavior. Keep proper-name/contextual accuracy gates separate from successful execution or low latency.
