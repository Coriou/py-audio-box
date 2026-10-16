# Shared speech input, portable installation and dictation boundary

Reviewed research synthesis, 2026-10-15. The user requested a dedicated STT lane, deeper existing-product comparison, and Docker installation on CPU-only and accelerated machines. French, English and mixed-language speech lead evaluation. Larger-model/GPU evaluation is authorized where available. No sibling application or upstream adoption has been selected.

## Confirmed requirements and boundaries

The convenient repository an agent can install and use locally should provide speech input as well as output when the research supports it. CPU-only Intel machines are a first-class target, not an error fallback. More capable hardware should permit explicitly selected larger/faster models. Hardware alone does not establish model quality, supported language or accuracy. Retain the full speech studio requirement and the existing feature/default preservation gates.

An “any box” installation goal must become a published OS/architecture/CPU-feature/runtime support matrix and actionable preflight. It cannot truthfully mean every historical CPU, every GPU and every OS version. A CPU deployment must work without installing a GPU stack; acceleration is an explicitly qualified profile. No model downloads on discovery; selected model provisioning is visible and bounded.

## Current machine evidence

Read-only checks on 2026-10-15 found Darwin x86_64, Intel Core i7-8700B at 3.20 GHz, six physical/twelve logical cores, 64 GiB RAM, AVX2/FMA/F16C. Docker client/server 29.4.0 is running; its Linux x86_64 VM exposes twelve CPUs and 16,843,739,136 bytes memory (~15.7 GiB). These are this session's host/VM facts, not a universal minimum or an inference benchmark. No usable inference GPU has been verified on this host.

## Product hypothesis to test

STT belongs in the shared service: agents receive voice messages, process recordings/media, prepare clone reference transcripts, inspect generated speech and retrieve timestamped evidence without becoming desktop applications. Native dictation belongs at a separate interface boundary: hotkey, microphone permission, foreground-app context, clipboard/insertion and desktop feedback. Existing open-source clients may already supply that boundary; qualify reuse before creating a new sibling product.

A sibling would consume the same transcription/context APIs rather than duplicate models, caches, queue state or quality policy. Do not embed a global desktop automation system into the Docker service. Browser capture is useful but not equivalent to system-wide dictation.

## Required distinctions

- Transcription is not translation, diarization, forced alignment, summarization or LLM cleanup. Each has separate capability, cost, provenance and job semantics.
- Raw recognizer output, corrected transcript and final dictated text are separate versioned artifacts. Preserve raw output even if the recognizer itself applies punctuation/normalization. Do not claim byte-verbatim reconstruction of speech.
- Language hints, vocabulary/context, uncertain words and speaker labels are data with explicit limits. Never silently turn unsupported language into English or identify a diarized speaker as a known person.
- Audio capture stays at the client edge. A private server/container does not implicitly own the user's microphone or speakers.
- GPU support in a native runtime, in a Linux container and in a Docker-managed host service are different claims. Docker Model Runner documents native-host engines on macOS/Windows and Apple Silicon Metal, but its documented engines alone do not establish a usable STT path. [DMR architecture](https://docs.docker.com/ai/model-runner/), [engine scope](https://docs.docker.com/ai/model-runner/inference-engines/).

## Research/evaluation status

- [CPU-first runtime/model audit](research-stt-runtimes.md): complete source research; runtime smoke is separate.
- [Shared STT agent contracts](research-stt-agent-contract.md): complete.
- [Dictation/native-client boundary](research-dictation-product-boundary.md): complete.
- [Docker portability](research-docker-portability.md) and [serving/adoption comparison](research-serving-adoption-comparison.md): complete source research.
- [Actual CPU model evaluation](evaluation/stt/RESULTS.md): completed with two timeouts, two English samples and a blocked different-runtime diagnostic; no starter profile qualified. Sole benchmark owner, serialized, task-owned public corpus/assets only. No private voice/audio access. Resource and transfer ceilings, pinned identities, all failures and measurement limitations belong in the receipt. GPU results require an actual qualified GPU; availability is not inferred from a model library's support table.

Final [Astra review](review-stt-portability-astra.md): GO for bounded core STT and contract/prototype comparison; NO-GO for CPU starter/runtime promotion, universal hardware claims, product adoption or a new dictation sibling. This is not runtime, multilingual accuracy or acoustic acceptance.

## Preliminary contract recommendation

Keep a small shared STT surface in v1: ingest/probe supported media, choose language/profile and bounded vocabulary/context, submit transcription, wait/recover, retrieve transcript files or selected timestamped segments. Provider hooks can perform obligatory voice-message transcription before a language-model turn; MCP remains available for deliberate transcription/search. Translation, speaker clustering, forced alignment, cleanup and summaries are distinct optional capabilities, never silently enabled stages.

Treat raw ASR text as fallible evidence, retain source audio/ranges and recognizer output, and record corrections as derived revisions. Do not reuse the internal TTS language mapping that falls back to English for unknown ISO codes. ASR timestamps are not forced alignment of an approved script. Large transcripts should be downloadable/searchable/range-addressable instead of inserted into every model response.

“Smart context” begins with explicit language/vocabulary/project context packs, bounded to the selected recognizer, with source and scope. It does not require screenshots, foreground-app scraping or another LLM. Cleanup can be done by a caller's agent or explicit optional transformation while retaining the original. Context capture and rewrite policy must be visible and independently controllable.

The first desktop qualification should connect an existing OpenWhispr or Handy client to the shared service, not start a new sibling repository. OpenWhispr's documented self-hosted mode has a missing-credential constraint and cleanup-on default; use a scoped authenticated loopback adapter/client correction and disable extra processing during baseline evaluation. Never weaken the service's authentication to make a client work. A sibling remains a later decision if real UX/integration gaps justify its maintenance cost.

## Actual non-model comparison evidence

The [published editor probe](evaluation/editor/README.md) ran 11 headless checks using exact package versions and recorded distribution hashes. Results establish reusable editing mechanics and identify overlap/history/state-ownership differences. They do not establish browser playback, accessibility, memory, RPC/CAS or FFmpeg parity. Whole-studio versus component selection remains provisional until the same production fixture passes both integration paths.

## Scope guardrails from Astra review

A bounded file-STT v1 is sufficient: publish a tested duration/size cap and reject unsupported inputs before work. Hour-scale checkpoint/join recovery is required only when advertised. Context hints and segment evidence belong baseline; live input, diarization, forced alignment, transcript editor and LLM cleanup do not become mandatory simply because an upstream supports them.

Prefer authenticated client support over a shim. If needed, a local native shim is transcribe-only, holds a scoped credential, validates Host/Origin/browser requests and rejects arbitrary URL/server-path forwarding. Loopback is a transport boundary, not authentication.

The networkless [two-container socket probe](evaluation/ipc/README.md) passed on this Intel Mac's Docker Linux VM; production IPC permissions/protocol/reconnect and Windows/WSL2 remain unqualified.

Runtime identity clarification: the observed Linux VM is OrbStack (`7.0.14-orbstack-00380-ga7e0a2dc9535`) accessed through the Docker CLI. Actual local probes do not certify Docker Desktop, Windows/WSL2 or another hypervisor.

## Actual CPU evaluation outcome

The public-data [CPU evaluation results](evaluation/stt/RESULTS.md) are final for this research pass. Both faster-whisper runs used four CPU threads/four CPU quota, a 6 GiB container limit, networking disabled, beam 5, batch 1 and a 128-generated-token cap, with effective `int8_float32`. They ran in this Mac's OrbStack Linux VM, not a certified Docker Desktop profile.

- Small loaded in 3.17 seconds but hit its 295-second deadline before completing its first 8.94-second clip. No WER is available.
- Large-v3-turbo loaded in 12.82 seconds and completed two English clips in 135.22 and 99.67 seconds (RTF 15.12 and 16.95), then hit its deadline. Neither run was OOM-killed.
- No French, constructed mixed-language, VAD, silence/noise, low-amplitude or warm-repeat case completed. Those strata are unmeasured, not passing or zero-error.
- The final whisper.cpp tiny-q5_1 model downloaded/verified, but its pinned container-image pull stalled before inference. Its intended 120-second pull watchdog failed; the exact task client was manually stopped after 147 seconds. This is an image-delivery/supervision failure, not a model-quality result.

No efficient CPU starter, default model, multilingual accuracy, warm latency or GPU performance is qualified by this run. AVX2 is present and sampled CPU-quota throttling was negligible; the poor faster-whisper throughput cause remains unverified. Do not generalize from this runtime/VM configuration to all Intel CPUs or all Whisper runtimes.

The two completed English samples have normalizer-sensitive WER: “twentieth” versus “20th” counts as an edit under the pinned digit-preserving normalizer, without an incorrect numeric value. Their narrow scores are not a dictation-quality ranking. Exact identities, source licenses, failed/partial receipts, memory measurement limits and reproduction harness belong in the evaluation report.

Next CPU lot: diagnose image delivery and runtime performance on one fixed short FR/EN pair, with bounded supervision proven first; compare a small whisper.cpp candidate and a known-good faster-whisper configuration under matched resource/input controls, using the same checkpoint/decoding policy where possible and explicitly separating quantization differences. Expand to the held-out French, English, natural code-switch and difficult-audio corpus only after one viable profile is demonstrated. No fourth configuration is attempted in this research pass.

## Contextual usability clarification

The user's TxCode example makes contextual usefulness a release criterion for dictation, not a cosmetic cleanup feature. [Issue #8](https://github.com/Coriou/txcode/issues/8) was found and read on 2026-10-15. Its small candidate-context and draft-safe composer flow is a concrete first integration target; its hosted-provider proposal is not automatically adopted here.

The shared service should offer contextual dictation alongside faithful media transcription: supported recognizer hints plus bounded terminology resolution when needed, a useful reviewable result through one job, and raw/derived provenance. Measure real correction effort and false context-induced edits on held-out FR/EN technical speech. Preserve unexpected valid speech, language switching, negation and numbers; do not silently rewrite intent. The client owns recording/caret/context selection. A sibling desktop product is not required to prove this value in TxCode.

No new model test, hosted-provider request, TxCode modification or issue comment was made for this clarification. CPU feasibility remains unqualified as recorded above.
