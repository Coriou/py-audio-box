# Existing self-hosted voice solutions

Research date: **2026-10-15**. Local planning baseline:
`ae67cd6a572e7105b7d41365aac0be4059d79584`.
Scope: primary official documentation, upstream source and public GitHub metadata.
No installation, model download, inference, private audio or secret access.
Only this report was written. Every external URL below was inspected on this date;
moving documentation/default branches may differ from their latest published release.

## Decision

**Do not approve a bespoke inference server or complete studio implementation
before an existing-product comparison.** LocalAI already supplies more of the
proposed voice/discovery/control plane than the initial inventory assumed.
Kokoro-FastAPI supplies an inexpensive focused serving alternative; VoiceMode
supplies a working agent conversation surface. These are meaningful reuse options.
They do not, by themselves, establish the requested shared script/timeline studio.

The [whole-product audit](audit-build-versus-adopt.md) separately investigates the closer candidates
[Voicebox](https://github.com/jamiepine/voicebox) and
[VoiceStudio](https://github.com/debpalash/VoiceStudio). Their coverage must be
combined with this report before deciding to build a production/editor domain.
Those two candidates were not independently audited here. Its findings about
in-memory execution queues, restart recovery and render semantics must remain
explicit adoption work, rather than being hidden by a feature checklist.

The most defensible alternatives to compare are:

1. Adapt a whole studio product if its production model and agent contracts pass.
2. Build only the missing production layer, using LocalAI or a focused server for
   model/voice serving, while retaining existing py-audio-box workflows as adapters.
3. Extend py-audio-box's supervisors only where a measured compatibility,
   reproducibility or lifecycle gap makes an external server unsuitable.

## Evidence levels

- **D — documented:** upstream describes the behavior. This is not execution proof.
- **S — source inspected:** current routing or lifecycle code supports the claim.
- **M — metadata:** live GitHub repository/commit/release fields; activity is a
  maintenance signal, not proof of support, security, quality or stability.
- **U — unverified:** no runtime/browser/acoustic test performed, or behavior was
  not established by the inspected material. “Not established” does not mean absent.

## Strongest server candidates

### LocalAI — shortlist for serving and voice management

**D:** Broad self-hosted engine platform, MIT core, Docker recommended, isolated
backend services installed as needed; deployment choices span CPU and several
GPU families. This is a product server, not an individual voice model.
[Overview](https://localai.io/docs/overview/index.html),
[repository](https://github.com/mudler/LocalAI).

**D:** Managed cloning library supports upload/record, transcripts, opaque profile
URIs, previews and per-model compatibility. Its reference list can hold multiple
clips, with backend-specific handling. API includes `/api/voice-profiles`,
`/v1/audio/voices` and cloning-capability filtering. It covers Qwen/Pocket/Fish and
other backend families. Instructions may be ignored by unsupported backends;
streaming can fall back to a complete response. Therefore its advertised API
compatibility does not satisfy strict control semantics automatically.
[TTS and Voice Library](https://localai.io/docs/features/text-to-audio/).

**D/S:** `/.well-known/localai.json`, focused instructions and model-capability
queries already implement agent discovery. Source also registers MCP list/create/
delete voice-profile tools; creation currently carries base64 WAV in the tool
request. Reuse discovery patterns, but keep large audio outside model context in
our adapter.
[Discovery](https://localai.io/docs/features/api-discovery/),
[MCP voice tools](https://github.com/mudler/LocalAI/blob/1d0a3227d7c48ba099f49c341dda39d73b583e99/pkg/mcp/localaitools/tools_voice_profiles.go).

**D:** Model watchdog/idle and resource management exist, so a new model manager
must justify its cost. Artifact configuration can resolve selected Hub files to
an immutable revision. Neither feature proves our timeout/reap/publication contract
for each speech backend.
[Memory management](https://localai.io/docs/advanced/vram-management/),
[model artifacts](https://localai.io/docs/advanced/model-configuration/).

**Recommendation — adapt/evaluate:** strongest broad server contender. Test the
required Qwen variant controls, exact text, profile identity, audio ingest and
idle reuse. No project-revision/timeline-render or restart-safe synthesis-job
completion guarantee was established by the inspected speech routes/docs. Do not
confuse LocalAI's agent-job system with durable audio-production jobs. Keep one
authoritative profile store or define explicit immutable mirroring; two mutable
voice libraries would increase identity risk.

### Speaches — shortlist for focused Kokoro/Piper serving

**D:** MIT, Docker/Compose CPU/GPU, OpenAI-style audio APIs, local model/voice
discovery, downloads and experimental loaded-model controls. Current documented
TTS engines are Kokoro/Piper; speaker embeddings are a different operation and
do not establish reference cloning or voice design.
[Repository](https://github.com/speaches-ai/speaches),
[API](https://speaches.ai/api/),
[installation](https://github.com/speaches-ai/speaches/blob/master/docs/installation.md).

**S:** Speech supports audio/SSE output, speed and sample-rate conversion. The
route strips emojis and Markdown emphasis before synthesis. This is a concrete
exact-script compatibility problem unless changed or explicitly surfaced. SSE
audio uses base64 and token usage is currently filled with zeros; these are not
cost measurements or compact job receipts.
[Speech route](https://github.com/speaches-ai/speaches/blob/993994f7984bf3fe9655b267448328cf66fccb42/src/speaches/routers/speech.py).

**S:** Reference-counted model ownership and idle TTL support keep-warm, immediate
unload or indefinite retention. This is an in-process lifecycle, not evidence of
isolated worker termination/reap or a durable job queue.
[Model manager](https://github.com/speaches-ai/speaches/blob/993994f7984bf3fe9655b267448328cf66fccb42/src/speaches/executors/shared/base_model_manager.py).

**Recommendation — adapt/defer:** good constrained API backend and lifecycle
reference; weaker than LocalAI for registry/cloning. Evaluate only if its simpler
operational footprint beats both LocalAI and our existing CPU adapter. Published
API docs identify 0.8.3 while the latest-release API returns `v0.9.0-rc.3`; pin and
test the exact release rather than combining documentation assumptions.

### Kokoro-FastAPI / FastKoko — shortlist for a dedicated speech lane

**D:** Apache-2.0 server with prebuilt CPU/CUDA multiarch images, experimental
ROCm, optional readalong UI, voice combinations/aliases, inline speakers, SSML,
captions and phoneme endpoints. Native MPS is distinct from Docker CPU operation.
Custom voicepacks come through a separate tuning project; do not equate that
with arbitrary reference-audio cloning.
[Repository and deployment](https://github.com/remsky/Kokoro-FastAPI).

**S:** Singleton model manager has startup warmup, active-request tracking,
reload and idle-unload handling. Speech routes validate aliases/tags before
streaming and expose controls beyond the common OpenAI subset.
[Model manager](https://github.com/remsky/Kokoro-FastAPI/blob/ebd4ec9cc97b258b25afc39444a5e1701055db50/api/src/inference/model_manager.py),
[speech route](https://github.com/remsky/Kokoro-FastAPI/blob/ebd4ec9cc97b258b25afc39444a5e1701055db50/api/src/routers/openai_compatible.py).

**Recommendation — adapt/evaluate:** strongest focused alternative if a warm
Kokoro service is useful. Test against our existing ONNX CPU path rather than
assuming PyTorch serving improves memory, latency or quality. No durable
production/job store, restart replay, immutable timeline or completion outbox
was established. Avoid importing custom `.pt` voicepacks into our trusted registry
without a separate validation/trust contract.

## Agent adapters and human frontends

### VoiceMode — reuse integration concepts, evaluate as a client

**D:** MIT MCP/CLI conversational voice product: speak/listen turns, silence
detection, local Whisper/Kokoro services and configurable OpenAI-compatible
providers. Its installation targets access to host microphone/speakers rather
than a headless Docker speech studio.
[Repository](https://github.com/mbailey/voicemode),
[authoritative converse parameters](https://github.com/mbailey/voicemode/blob/e03b5d91a984908032b848d9798ecb34a31a7491/voice_mode/resources/docs/parameters.md).

**S:** It now has filesystem clone profiles with WAV/transcript, expression/tone
selection, group resolution and model/endpoint routing, including MLX/Qwen.
Remote use may require mirrored reference paths. This is more than a preset-only
adapter, but not the opaque uploaded asset model proposed for the service.
[Profile resolver](https://github.com/mbailey/voicemode/blob/e03b5d91a984908032b848d9798ecb34a31a7491/voice_mode/voice_profiles.py).

**Recommendation — reuse/adapt:** test VoiceMode against our chosen backend for
“give agents a voice now”; do not rebuild live conversation merely to expose
synthesis. Maintain a separate production API for durable takes and timeline
work. Listening for a response is not an asynchronous completion wakeup for a
detached agent.

### Open WebUI — useful interoperability target, not the studio foundation

**D:** Read-aloud/voice conversations configure an OpenAI-compatible TTS base URL,
model and voice. This makes it a ready client for compatible servers.
[Audio overview](https://docs.openwebui.com/features/chat-conversations/audio/),
[TTS integration](https://docs.openwebui.com/features/chat-conversations/audio/text-to-speech/openai-tts-integration/).
No shared editable narration/timeline production workflow was established from
these audio features. Current license has branding conditions and exceptions;
it should not be called plain MIT/BSD for a new fork.
[License](https://github.com/open-webui/open-webui/blob/main/LICENSE).

**Recommendation — reuse as a compatibility client; defer a fork.** A successful
read-aloud demonstration proves endpoint integration, not the required studio.

### AllTalk v2 — substantial bulk production reference, adoption has friction

**D:** v2 is recommended over v1; Gradio engine controls, cloning-related voices,
RVC/narrator flows and multi-engine request queue are documented. Docker remains
beta. Maintainer explicitly reports limited available time.
[v2 README](https://github.com/erew123/alltalk_tts/blob/alltalkbeta/README.md),
[Docker](https://github.com/erew123/alltalk_tts/wiki/Docker-Support-(Beta)),
[multi-engine manager](https://github.com/erew123/alltalk_tts/wiki/Multi-Engine-Manager).

**D:** Bulk generator edits/regenerates chunks, persists a JSON catalogue, exports
combined WAV/SRT and offers Whisper comparison. JSON depends on separately kept
audio; stopping playback does not stop generation. These are useful production
UX precedents, not proof of our revisioned timeline or durable cancellation.
[Generator](https://github.com/erew123/alltalk_tts/wiki/TTS-Generator).

**D:** OpenAI speech adapter ignores `model`, remaps six voice names and cannot
select language. Standard API has richer controls and status/voice/settings
discovery. Streaming excludes narration and RVC.
[OpenAI endpoint](https://github.com/erew123/alltalk_tts/wiki/API-%E2%80%90-OpenAI-V1-Speech-Compatible-Endpoint),
[status](https://github.com/erew123/alltalk_tts/wiki/API-%E2%80%90-Server-%26-TTS-Engine-Status-API),
[standard API](https://github.com/erew123/alltalk_tts/wiki/API-%E2%80%90-Standard-TTS-Generation-API),
[streaming](https://github.com/erew123/alltalk_tts/wiki/API-%E2%80%90-Streaming-TTS-Generation-API).

**Recommendation — reuse patterns; defer adopting/forking as default.** AGPL-3.0
server, engine/model-specific licenses, beta packaging and older v2 tip require
deliberate ownership. Do not mistake its request queue for restart-safe jobs.

### openedai-speech — defer new adoption

**D/M:** OpenAI speech wrapper around XTTS/Piper, Docker and configured reference
voices, AGPL-3.0. The repository is archived and its main tip is 2025-02-02.
[Repository](https://github.com/matatonic/openedai-speech),
[metadata](https://api.github.com/repos/matatonic/openedai-speech).
Open WebUI still documents integration, which does not establish active upstream
maintenance. Retain only as a compatibility reference, not a new core dependency.

## Engines and serving runtimes are separate choices

| Candidate | Verified scope / gap | Recommendation |
| --- | --- | --- |
| Qwen3-TTS upstream | Apache-2.0 engine family with Base cloning, CustomVoice and VoiceDesign; library/demo is not a production registry/job/editor. [Upstream](https://github.com/QwenLM/Qwen3-TTS). | Reuse current adapter and compare serving alternatives before rewriting it. |
| vLLM-Omni | Apache-2.0 serving runtime. Qwen variants, streaming and voice-upload/list routes documented; runtime upload/delete is disabled with multiple API frontends because registry state is process-local. [Serving](https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/examples/online_serving/text_to_speech/), [repo](https://github.com/vllm-project/vllm-omni). | Evaluate GPU lane when hardware/concurrency justify it; not a durable production store. |
| MLX Audio | MIT Apple MLX library/server, OpenAI-style speech route with reference parameters. [Repo](https://github.com/Blaizzy/mlx-audio), [server](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/server.py). | Optional native Apple serving adapter; does not satisfy a Docker-only Apple GPU promise. |
| Chatterbox upstream | MIT engine with reference conditioning, multilingual/Turbo variants and watermarking. [Upstream](https://github.com/resemble-ai/chatterbox). | Candidate engine, subject to listening/resource tests; not a whole studio. |
| Chatterbox TTS API | Independent AGPL-3.0 wrapper documents Docker, uploaded named voices, React UI/status/chunking. README still warns about non-CUDA compatibility and pending Turbo support. [Wrapper](https://github.com/travisvn/chatterbox-tts-api). | Defer as default versus broader LocalAI; inspect exact engine pin before evaluation. |
| Fish Speech | Current server source has `/v1/tts`, streaming, defaults and reference add/list/delete/rename, not just a model script. [Routes](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/tools/server/views.py), [inference](https://github.com/fishaudio/fish-speech/blob/main/docs/en/inference.md). | Optional backend after license/version decision; no studio/job guarantees established. |
| Coqui community fork / XTTS | Maintained MPL-2.0 toolkit, voice cache and server utilities; XTTS-v2 weights use separate CPML, explicitly non-commercial for model/output. [Fork](https://github.com/idiap/coqui-ai-TTS), [XTTS license](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt). | Defer default production engine; code license does not grant model/output permission. |

Fish needs particular care with stale assumptions: current code and Drama3 are
under **Fish Audio Community License v1.0, dated 2026-10-13**; S2-Pro remains
under separate research terms. The current license includes commercial thresholds,
hosted-service restrictions and attribution. This is not an Apache/MIT-only choice.
Freeze the exact code/weights/licenses and establish eligibility for the intended
internal/team/product use before adopting; this report does not decide eligibility.
[Current upstream notice](https://github.com/fishaudio/fish-speech),
[license text](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/LICENSE).

None of these engines is an editor component. Waveform/timeline libraries also
do not automatically provide script revisions, take selection, sample-accurate
server rendering or jobs; assess editor components separately from server adoption.

## Maintenance snapshot and reproducibility

**M:** Mainline commit date is shown separately from repository `pushed_at`:
other branches/tags can make a repository look recently updated. Commit pins below
come from the official `repos/<owner>/<repo>/commits/<branch>` GitHub API. SPDX
labels are code-license indicators only; model/dependency terms remain separate.

| Repository/branch | Inspected tip, commit date UTC | Code license / release signal |
| --- | --- | --- |
| [LocalAI/master](https://api.github.com/repos/mudler/LocalAI/commits/master) | `1d0a3227d7c4`, 2026-10-15 | MIT; [v4.11.0](https://github.com/mudler/LocalAI/releases/tag/v4.11.0), 2026-10-02 |
| [Speaches/master](https://api.github.com/repos/speaches-ai/speaches/commits/master) | `993994f7984b`, 2026-04-18 | MIT; [v0.9.0-rc.3](https://github.com/speaches-ai/speaches/releases/tag/v0.9.0-rc.3), 2025-12-27; API marks this non-prerelease despite tag name |
| [VoiceMode/master](https://api.github.com/repos/mbailey/voicemode/commits/master) | `e03b5d91a984`, 2026-10-04 | MIT; [v8.12.0](https://github.com/mbailey/voicemode/releases/tag/v8.12.0), 2026-07-21 |
| [AllTalk/alltalkbeta](https://api.github.com/repos/erew123/alltalk_tts/commits/alltalkbeta) | `f16117e95b54`, 2026-01-09 | AGPL-3.0; v2 branch differs from default main |
| [openedai-speech/main](https://api.github.com/repos/matatonic/openedai-speech/commits/main) | `31f033595a81`, 2025-02-02 | AGPL-3.0; archived |
| [Kokoro-FastAPI/master](https://api.github.com/repos/remsky/Kokoro-FastAPI/commits/master) | `ebd4ec9cc97b`, 2026-10-14 | Apache-2.0; [v0.9.0](https://github.com/remsky/Kokoro-FastAPI/releases/tag/v0.9.0), 2026-09-10 |
| [Chatterbox/master](https://api.github.com/repos/resemble-ai/chatterbox/commits/master) | `5de7a54aa4e5`, 2026-07-21 | MIT |
| [Fish/main](https://api.github.com/repos/fishaudio/fish-speech/commits/main) | `541566932abb`, 2026-10-14 | Custom community/research terms by artifact |
| [Qwen3-TTS/main](https://api.github.com/repos/QwenLM/Qwen3-TTS/commits/main) | `022e286b98fb`, 2026-03-17 | Apache-2.0 |
| [Coqui fork/dev](https://api.github.com/repos/idiap/coqui-ai-TTS/commits/dev) | `aeb064d41b5d`, 2026-10-02 | MPL-2.0; [v0.27.5](https://github.com/idiap/coqui-ai-TTS/releases/tag/v0.27.5), 2026-01-26 |

## Evidence required before reuse

Use a bounded comparison rather than installing everything. Initially compare
one whole-product finalist, LocalAI and the existing py-audio-box path. Add
Kokoro-FastAPI only for a focused CPU-lane decision, and VoiceMode as the agent
client. Heavy tests require a later authorized evaluation; none ran here.

| Gate | Required evidence |
| --- | --- |
| Contract fit | Exact feature mapping for upload/reference transcript, clone/design/preset distinction, tones/instructions, language and unsupported-control rejection; no silent engine/voice fallback. |
| Preservation | Synthetic text including whitespace, Markdown, accents, numbers and emoji compared at ingest/submission/metadata boundaries; original reference and selected model/voice version hashed. |
| Agent cost | Actual tool schemas/tokenizer counts, paginated voices, compact receipts/status, zero base64/audio in normal tool responses; compare first-utterance call count. |
| Warm lifecycle | Cold and sequential warm load counts, idle eviction/reload, CPU/RAM/VRAM, queue wait and TTFA/complete latency with pinned image/assets/hardware. Upstream benchmark claims are not local results. |
| Job correctness | Restart during synthesis/publication, duplicate-key conflict, cancel during model load/inference, late result after expired generation, partial batch failure, disk exhaustion and isolated callback failure. An `async def` handler, stream or in-memory queue does not pass this gate. |
| Voice store | One authority, stable IDs/revisions, upload/rename/delete/export/import, missing/corrupt assets, concurrent edit and migration/rollback tests. No remote pickle execution. |
| Studio fit | Required script/timeline edit/save/reload, changed-block-only regeneration, chosen-take adoption preserving placement, deterministic pinned render and browser/server playback equivalence. |
| Deployment | Pin release and image digest; check declared target architecture and accelerator; clean install/upgrade/backup/restore without host ML installation or private credentials in images. |
| Adoption cost | Count adapter changes and missing domain features; compare upgrade/fork burden against custom implementation. A two-day integration spike is a proposed budget, not evidence that adoption is easy. |

The plan should therefore reserve custom work for validated gaps. Preserve the
current CLI/registry identities and recent worker/reference safety guarantees
while making external serving an explicit adapter option. “OpenAI-compatible”
is a useful transport baseline; it is not a voice registry, durable jobs, an
agent wakeup guarantee or an editable production contract.
