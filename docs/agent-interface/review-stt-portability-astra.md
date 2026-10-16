# Astra review — shared STT, CPU portability and dictation scope

15 October 2026. Planning and evidence review in `py-audio-box`; no implementation,
model run, install, private recording, GPU or desktop client was executed by this
reviewer. Reviewed the five STT/portability/serving reports, working synthesis,
PLAN/ACCEPTANCE, prior adoption review and the actual evaluation harnesses and
receipts. Upstream product claims remain attributed to the pinned source audits;
this review does not independently certify every upstream feature.

Source packet: [runtime/model audit](research-stt-runtimes.md),
[agent contract](research-stt-agent-contract.md),
[native-client boundary](research-dictation-product-boundary.md),
[Docker portability](research-docker-portability.md),
[serving/adoption comparison](research-serving-adoption-comparison.md).
Execution evidence: [final CPU results](evaluation/stt/RESULTS.md),
[CPU harness and scope](evaluation/stt/README.md),
[editor](evaluation/editor/README.md), [Voicebox export](evaluation/voicebox/README.md),
[Mac IPC](evaluation/ipc/README.md).

**GO — bounded contract/prototype and build-versus-adapt comparison.** The expanded
research supports first-class bounded core STT, CPU-first qualification and one
existing native-client handoff later. Final PLAN/ACCEPTANCE incorporation and the
synthesis's actual CPU outcome were reread; no unresolved blocker remains for
that planning stage. The full speech studio remains in scope.

**NO-GO — promotion of either tested CPU profile to the recommended starter,
runtime/model selection, universal Docker support, a new dictation sibling or
full-product adoption/release.** The final empirical receipt preserves two timed-
out inference runs and a blocked image pull with a failed watchdog. Core STT's
value is supported; usable CPU execution, multilingual accuracy and effortless
installation are not yet qualified. This is an evidence boundary, not a verdict
that Intel CPUs or any runtime family are inherently unsuitable.

## Product disposition

First-class STT belongs in the shared speech service. Received voice messages,
recording/media import, clone-reference preparation, generated-speech QA and
timestamped retrieval give agents concrete uses without a native desktop shell.
Share assets, jobs, immutable plans, resource admission and transcript artifacts
with the production service. Provider hooks handle obligatory incoming-audio
transcription before a language-model turn; deliberate HTTP/CLI/MCP operations
handle transcription, retrieval and search chosen by the agent.

A Wispr Flow-style client is a separate native interaction problem: global
hotkeys, capture permissions, foreground focus, clipboard/insertion and optional
app context. Qualifying one existing client is justified. Creating a sibling
product now would expand the work beyond what the present evidence supports.
OpenWhispr is the first adapter candidate from the source audit; Handy is the
simpler local interaction reference. Neither has been installed/qualified here.
The shared full script/timeline studio remains required; a dictation client does
not replace it or require the server to own the host microphone.

| Include in bounded core STT | Qualify later or only when advertised |
| --- | --- |
| Tested media ingest/probe, explicit track/channel/range, immutable source and bounded duration/bytes/decode | Arbitrary remote sites, every codec/DRM source, unbounded recordings |
| Explicit multilingual profile/language behavior, FR/EN and genuine mixed-language qualification | Universal accuracy or an automatic language-repair guarantee |
| Raw recognizer output, source-linked segment timing when supported, stable transcript IDs/revisions, TXT/JSON and bounded range/search retrieval | Word-perfect timing, forced alignment, speaker diarization/identification, full transcript-editing UI |
| Explicit bounded vocabulary/context with supported/dropped/ignored semantics and recorded effective identity | App scraping, screenshots, automatic context collection, mandatory rewriting LLM |
| Durable bounded file jobs, cancellation/recovery policy, complete/partial/failed coverage and saved evidence | Live-input streaming, near-real-time latency guarantee, conversational duplex |
| Shared TTS/STT resource policy and one authority for results | A second native/server model cache and independent job ledger by accident |

An initial duration cap is acceptable. Advertised long-media support requires
bounded decoding, source-preserving chunk manifests, overlap/join rules,
checkpoint/recovery and exact missing ranges. The illustrative one-hour request
in research does not create a one-hour release promise. ASR segment/word estimates
are not forced alignment of an authored script. Diarization, alignment, translation,
cleanup and summary stay separately declared capabilities rather than silent
stages. Caption exports are conditional on qualified timing/format semantics.

## Context, evidence and client boundaries

“Raw” means the engine's emitted result, which may contain normalization and
recognition errors. Preserve it with audio/range, model/runtime, language,
precision, decoding/VAD, selected context and timing-method identities. A user
correction or cleanup result creates a derived revision; it does not overwrite
what the recognizer emitted. Do not copy the current reference helper's fallback
to English into general STT. Recognizer scores are diagnostics, not calibrated
accuracy percentages.

Start smart context with explicit, versioned user/project vocabulary and a small
topic/language pack. Resolve backend token limits; record selected/dropped terms
and reject strict unsupported requests before inference. Test absent hotwords
alongside intended names and numbers. Never expose a field that the selected
backend silently ignores. Recognition, substitutions/snippets and LLM rewrite
have different effects and provenance. No local conversational model is required
when the caller already provides its own agent.

Spoken text, media metadata and context are untrusted content. Provider hooks
deliver them as sourced attachments, not privileged instructions. Desktop app
context requires an explicit client-side policy and separate host permissions.
Insertion must handle focus changes/clipboard races and must not submit/execute
terminal text as an incidental paste operation.

An OpenWhispr compatibility experiment must keep cleanup/cloud/context capture
disabled for the baseline and prove its actual request path. Prefer scoped
authenticated client support. If a native loopback shim is needed, loopback alone
is not the authentication policy: narrow its operations and credentials, protect
Host/Origin/browser-request boundaries, bound uploads/wait/retries and prohibit
arbitrary server paths or URL forwarding. The shim must not make the canonical
API unauthenticated or claim that an HTTP disconnect stopped inference.

## Portability and resource disposition

Qualify Linux/amd64 CPU first on this identified Intel Mac's OrbStack Linux VM.
This is Docker-compatible runtime evidence, not a Docker Desktop certification.
Its AVX2/FMA/
F16C-capable i7 and approximately 15.7 GiB VM budget do not represent every older
Intel Mac. The conservative AVX2 profile is a product starting point; it does not
mean CTranslate2 universally requires AVX2. SSE4.1-only, native arm64, Linux Engine,
Windows/WSL2 and each accelerator tuple need their own complete dependency/runtime
qualification. A multi-architecture image manifest is insufficient.

CPU installation must require no GPU stack. Provision the chosen pinned engine/
model explicitly; discover capabilities without imports, downloads or inference.
Record requested/effective precision/device, image/model/tokenizer identities,
daemon architecture and actual VM/container limits. Missing GPU or incompatible
precision cannot silently select another model, English-only variant or cloud
endpoint. Native macOS acceleration is a separate worker profile, not Metal in a
Linux container. No GPU inference evidence exists on this reviewed host.

The Mac IPC receipt establishes a useful transport primitive: two networkless
Linux containers exchanged ping/pong over a Linux named volume. It does not prove
non-root identity mapping, authentication, restart/stale-socket behavior,
generation fencing, Windows support or complete worker isolation. Those remain
profile qualification gates; metadata-only doctor must not create test containers.

Keep one host-wide heavy inference lane initially, shared by STT, TTS, reference
ASR and QA, with bounded affinity and age/quantum fairness. Admission also accounts
for decode/render/provision work; model load/eviction is a measured cost. A long
transcription cannot reserve unlimited priority, while small tasks must not force
constant model swapping. On timeout/cancellation, fence publication and confirm
owner termination before replacement; quarantine an unreapable lane. Resource
estimates and sampled memory are not hard peak guarantees.

LocalAI remains an optional serving adapter beneath either studio architecture.
Its broad backend catalog or watchdog does not become the canonical voice/job/
project authority or establish combined-memory admission. First qualify one
explicit backend/model operation with the same immutable plan and lifecycle
boundary; preserve current Qwen behavior and legacy CLI defaults meanwhile.

## Runtime evidence review

The actual CPU harness uses pinned public FLEURS FR/EN inputs and pinned converted
models, serialized `small` and `large-v3-turbo` CPU INT8 runs, four threads/four
Docker CPUs, a 6 GiB container limit, network disabled and a bounded whole-run
deadline. It retains raw text, normalization/metric definitions, package/device/
precision details and separate process RSS/cgroup observations. This is a sound
bounded smoke design, not a representative quality benchmark.

Its six natural clips use forced language and total approximately 51 seconds.
The automatic-language FR→EN concatenation is a constructed control, not natural
code switching. Silence/noise, one warm repetition, VAD pairs and a reduced-volume
English clip address a few failure modes. Decoder beam 5, 128-token output cap,
temperature 0 and no word timestamps are part of the tested configuration.
First-decode/cold-model timing does not imply cold OS caches. One warm repeat does
not establish p95 latency. Sequential model order and tiny sample count prohibit
a generalized quality/speed ranking.

The first two terminal receipts are now available:

| Configuration | Actual completed evidence | Limit |
| --- | --- | --- |
| faster-whisper small, CPU requested INT8/effective `int8_float32` | Loaded in 3.17 s; no completed sample before exit 124 at the 295 s deadline; `OOMKilled=false` | No transcript, WER, RTF or usable-profile result exists |
| faster-whisper large-v3-turbo, same runtime/limits/effective type | Loaded in 12.82 s; two English clips completed in 135.22 s / 8.94 s audio and 99.67 s / 5.88 s audio; exit 124; `OOMKilled=false` | No French, mixed, negative, VAD, low-volume or repeated-warm results completed; not a recommended CPU profile |

Sources: [small records](evaluation/stt/results/small.jsonl),
[small terminal state](evaluation/stt/results/small.container.json),
[turbo records](evaluation/stt/results/turbo.jsonl),
[turbo terminal state](evaluation/stt/results/turbo.container.json).

The first turbo transcript exactly matches its public reference. The second
substitutes written `20th` for `Twentieth`, plus punctuation. The documented
normalizer therefore counts one word edit; this is not evidence of a wrong
numeric value. Tiny aggregate WER on these two English clips cannot rank models.
Small completed no sample, so there is no matched small/turbo quality comparison.
The observed decode RTFs are approximately 15.1 and 17.0, which demand diagnosis
before a useful interactive or efficient file-STT promise.

The separately authorized tiny quantized whisper.cpp diagnostic never reached
inference. Its model hash was verified, but the pinned GHCR image pull produced
no available local image and the pull client was ultimately killed (exit 137).
This is an image-acquisition/supervision failure, not a whisper.cpp inference or
quality result. The owner's terminal account reports 147 s against the intended
120 s pull cap: the attempted alarm failed to stop the Docker client, requiring
verified-PID termination. That 27 s cap overrun must remain visible; pull
supervision is unqualified. The two 295 s inference deadlines did hold.

Known corpus/model/image payload sizes stayed below the conservative 4 GiB plan;
actual registry wire bytes and partial transfers were not measured. Do not turn
that estimate into a measured transfer total or say every cap held. Even if the
tiny diagnostic had run, changing runtime, model, quantization and language/
decoding policy would not isolate a runtime effect or rank quality against
small/turbo. No further runtime/corpus experiment is requested by this review.

Timeouts, partial runs and absent Python finalization records stay visible;
external sampling is separately scoped and can include observer overhead.
Near-zero cgroup quota throttling in a sample does not explain the slowdown or
prove its cause. No OOM is different from a safe or efficient model profile.
A successful read-speech smoke establishes execution, not dictation/code-switch/
long-file qualification. sherpa, native clients and GPU profiles remain untested.

## Bounded next-stage lots

1. **One contract/integration fixture shared by the two studio candidates.** Add
   one STT job/result to the existing asset/job/revision fixture; test context
   propagation, unsupported controls, silent versus no-audio outcomes, bounded
   retrieval, coverage and late/cancelled publication. Keep at most two whole-
   studio paths: Voicebox adaptation and service plus editor component. LocalAI
   can be a replaceable worker candidate in the same fixture, not a third studio.
2. **One CPU/package diagnosis, then qualification before choosing a release profile.**
   Resolve image acquisition/pull supervision and the unexpectedly slow/censored
   faster-whisper runs first with a small
   explicit effort ceiling and matched settings where a runtime effect is being
   investigated. Do not start a broad model sweep or enlarge the corpus while
   the execution path remains unexplained. Then use a fixed public/authorized
   FR/EN set with real mixed utterances, critical names/
   numbers, silence/noise and one representative longer recording. Compare only
   the retained configurations; include one pinned whisper.cpp alternative if
   packaging/latency remains a live decision. Measure matched quality, warm/cold
   latency and TTS→STT→TTS switching under the actual VM budget. Set the duration/
   latency promises from evidence instead of requiring every runtime candidate.
3. **One native client handoff only after the core request works.** Reuse an
   existing client with literal transcription, explicit credentials and no extra
   cleanup. Qualify capture → service → review/insertion, cancellation, unavailable
   server and focus/permission failures. Decide whether a sibling is justified
   only from observed integration/UX gaps; do not create one in this planning lot.

The fake-worker studio/contract comparison can proceed independently of CPU
diagnosis. Expansion of the recognition corpus and native-client qualification
should follow a functioning, explainable CPU execution/package path. A failed
field probe does not invalidate core STT's agent value or authorize three product
implementations.

The editor's 11 headless checks, four Voicebox export-function fixtures and Mac
IPC smoke are actual narrow execution evidence. They do not certify an installed
studio, browser audio/accessibility, durable service, native dictation or a release.
Voicebox's missing-source/normalization/zero-gain findings are adaptation work;
component overlap/undo/shared-object/split-ID findings are adapter work. Neither
set alone determines the less costly product path.

## Required incorporation and final gate

The updated PLAN and ACCEPTANCE were reread after incorporation. They explicitly
include bounded STT, transcript/context resource families and CPU-first installation
in confirmed direction and delivery stages; cover truthful mixed-language claims,
context identity/support, source timing, partial coverage, shared resource
lifetimes, provider-hook versus deliberate-tool journeys and actual platform
evidence tiers; and keep later capabilities conditional while retaining the full
speech studio. No source-stage blocker remains in those dispositions.

The synthesis's completed CPU-outcome section accurately records the failures,
OrbStack boundary, absence of FR/mixed evidence and diagnostic prerequisite. Its
introductory working-status label can be closed with this review and the final
receipt links as the research handoff is assembled.
No benchmark winner, all-hardware support, installed-client success, full-product
adoption or release readiness follows from a planning GO.

The source report's missing project redistribution license is a publication
prerequisite for a distributable release. It does not block this research or the
bounded contract/prototype stage; choose terms and inspect the actual shipped
dependency/model closure before publication.
