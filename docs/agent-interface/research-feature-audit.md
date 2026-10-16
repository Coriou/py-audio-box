# Existing feature and execution audit

Research snapshot: 2026-10-15, commit `ae67cd6a572e7105b7d41365aac0be4059d79584`.
The tracked tree was clean at inspection. This document is planning only: no
inference, model downloads, Docker builds, private audio or secret access. No
applicable `AGENTS.md` was found in the repository or its ancestor directories.
Evidence below is static source inspection; existing test files are coverage
pointers, not a claim that tests were rerun for this research.

## Main finding

The repository already contains most voice operations and a substantial queue.
The confirmed v1 is a self-hosted speech production studio for one person or a
trusted team, with HTTP, a small CLI and MCP. Scripts and timeline editing are
required v1 capabilities. The main work is exposing operations through one inexpensive contract, separating
control operations from model ownership, and connecting the isolated CPU engines
to the existing execution architecture. A web API, agent tool server, human web
interface, generic assembly job and completion delivery are new work. A queue
rewrite is unnecessary, but the current queue schema and execution adapter need
an additive migration.

## Inventory

| Area | Existing implementation and evidence | Classification for an agent service |
| --- | --- | --- |
| Discovery | `run` discovers `apps/<name>/<name>.py`; `voice-synth` offers JSON voices, speakers, capabilities. `apps/voice-synth/voice-synth.py:360`, `:738`, `:916`, `:1021`. | Reuse output data; refactor cheap discovery out of heavy CLI imports. |
| Voice registry | Named slugs, source/ref WAVs, clone prompt variants, built-in profiles, tones, generation defaults, rename/delete/export/import and schema migration. `lib/voices.py:160`, `:353`, `:399`, `:573`, `:738`, `:769`, `:805`, `:892`. | Reuse canonical registry, expose typed summaries and operations. |
| Reference extraction | URL/local audio, source time ranges, raw VAD, candidate windows, Demucs vocal separation, precise VAD, scored registry clip and interactive choice. `apps/voice-split/voice-split.py:140`, `:183`, `:232`, `:401`, `:449`. | Reuse pipeline; turn interactive selections into explicit draft/candidate/choose calls. |
| Clone preparation | `prepare-ref`, `synth`, self-test; normalize, VAD candidate selection, Whisper transcription/scoring and prompt creation. `apps/voice-clone/voice-clone.py`, app README. | Reuse stages; separate registration/preparation from sample synthesis so a human need not buy an unwanted take. |
| One-shot registration | `voice-register` chains split and clone through subprocesses; registration normally also synthesizes supplied text. `apps/voice-register/voice-register.py`. | Retain CLI; add orchestration adapter with structured progress/errors. |
| Qwen synthesis | Cached clone/designed voices, named or direct CustomVoice speakers, language selection, tones, profiles, sampling controls, seed/variants, QA/ranking, chunking, word timestamps. `apps/voice-synth/voice-synth.py:1289`, `:2039`; `lib/tts.py:259`, `:942`, `:961`. | Reuse; move orchestration from argparse/print/exit handlers into callable operations. |
| Voice design | VoiceDesign description → sample → reusable Base clone prompt, with registration. `apps/voice-synth/voice-synth.py:1772`. | Reuse as an expensive background operation, with distinct sample/registration state. |
| CPU engines | Kokoro presets and Pocket reference cloning have isolated images, hash-verified provisioned assets, offline Docker synthesis, metadata and supervised homogeneous batches. `run-cpu`, `Dockerfile.cpu`, `apps/voice-cpu/voice-cpu.py:168`, `:357`. | Reuse supervisors; add registry voice records and queue adapters. They currently bypass `voice-synth` and `JobSpec`. |
| Stitching | Exact-span chunk planning and strict rate/channel-aware WAV concatenation with boundary pauses. `lib/fidelity.py:66`, `:103`; internal synthesis concatenation at `apps/voice-synth/voice-synth.py:1547`. | Reuse primitive; generic ordered multi-asset/multi-voice assembly is missing as a public operation/job type. |
| Export | Separate PCM24 WAV export, actual FFmpeg resampling, 1/2 channels and optional single-pass loudness target; originals preserved, atomic no-replace publication. `apps/audio-export/audio-export.py:10`. | Reuse as export job; advanced mixing, crossfades, codec delivery and measured loudness compliance are separate scope. |
| Queue producer | YAML/defaults/beatsheet ingestion, plan, compile, local run, idempotent enqueue, status/history/result/report/retry/flush; deterministic job ordering and argv manifests. `apps/job-runner/job-runner.py:257`, `:336`, `:1414`. | Reuse primitives; introduce typed operations beyond synthesis and safer public endpoints. |
| Queue consumer | Redis Streams leases, stale reclaim, owner-checked renewable global lock, delayed retries, terminal records, local/Vast executors and local/rclone sinks, package manifests and stream retention. `lib/jobqueue.py:351`; `apps/job-watcher/job-watcher.py:162`, `:731`, `:1403`, `:1540`. | Reuse coordination, decouple executor/sink/plugin boundaries. |
| Human management | Terminal menus and registry commands; audio exists on disk for external listening. No human web UI found in `apps`, `lib` or project dependencies. | Missing: voice library, upload/reference selection, previews, profile editing and job/artifact inspection. |
| Agent transports | Existing machine-readable CLI contracts; no HTTP/OpenAPI/MCP server found in source/dependencies. | Missing: thin transports over one operation layer, compact discovery and stable error/result envelopes. |
| Completion wakeups | `callback_data` is carried into queue/package records, not executed or delivered. `apps/job-watcher/job-watcher.py:1030`, `:1078`, `:1197`. | Missing: durable completion events/outbox and optional registered delivery adapters. |

## Required shared production domain

The beatsheet importer is a useful script bridge: `topicId`, `title`, numeric beat
IDs, narration, project TTS defaults and per-beat overrides become independent
jobs. It preserves topic/beat/source metadata in `callback_data`
(`apps/job-runner/job-runner.py:521`). The local sink aggregates per-beat takes,
selected take, status and failures into `job.json`, ordered by beat ID
(`apps/job-watcher/job-watcher.py:1065`). This is package aggregation, not a mixed
audio render or a saved editable timeline. The importer also strips narration
edge whitespace; a new script model should retain authored source exactly and
record any submission transform explicitly.

Use a versioned production model shared by human UI and agent transports:
production → script revision → utterance/segment → generated takes → chosen take
→ timeline revision → render/export. A timeline clip references an immutable
asset/take and carries track, start, source-in/out, gain, fades and mute state.
Script edits invalidate only changed utterances; selection and timeline edits
must not regenerate speech. A render pins script/timeline/voice revisions and
source hashes, while the editor can continue changing the draft.

For v1, deliver script editing with per-segment voice/settings/take selection,
ordered and timed speech clips, explicit pauses, trim/reorder/move, gain/fades,
playback/scrubbing, and render/export through the shared production contract.
Basic imported bed/music/SFX clips can use the same asset/track model. Sample
editing, MIDI, plugins and an arbitrary effects graph are outside this bounded
speech-studio model. Define the precise track/overlap envelope in the contract
before implementation; timeline editing itself is part of v1.

Existing mix work is experimental and separate from the voice application:
`experiments/sfx-bank-pipeline/render_mix.py:87`, `:98`, `:177` implements
sample-aligned 48 kHz mono overlap-add, fades, gain, peak guard, tempo handling,
two-pass normalization and video review muxing. It is coupled to SFX bank plans,
oracle analysis, benchmark paths/ledgers and an optional remote generation
caller. Treat it as algorithm/reference evidence, not a ready production mixer;
extract a pure renderer only after synthetic deterministic tests and dependency
review. Do not enable its remote generation while importing timeline support.

## Execution and cost facts

- A watcher batch keeps a Vast machine alive across batches; it does **not** keep
  the Qwen model warm across individual jobs. `execute_manifest_data` starts one
  `subprocess.run` per job (`apps/job-runner/job-runner.py:336`). Sorting by source
  is insufficient for model reuse. Local watcher mode uses the same execution
  path (`apps/job-watcher/job-watcher.py:1434`).
- Qwen already owns a resident spawned model worker within one operation, with
  serialized calls, CPU prompt DTOs, deadlines and bounded stop/reap
  (`lib/tts_backends/qwen.py:83`, `:155`). Reusing that owner across service jobs
  requires an explicit lifetime, idle eviction, identity key and resource budget.
- CPU `batch` already provides actual warm reuse, sequential homogeneous jobs,
  first-job initialization timing and worker generation/restart tracking. Limits
  are 1,000 jobs/16 MiB JSON, with upfront collision checks and no overwrite
  (`apps/voice-cpu/voice-cpu.py:168`, `:220`, `:296`, `:357`). Different backend,
  language or voice requires a different batch.
- Queue normalization enables `align_words` by default; argv generation adds
  Whisper alignment to jobs unless explicitly disabled
  (`lib/jobqueue.py:305`, `:549`). A low-cost speaking preset should explicitly
  request one take and no QA/alignment; preserve the existing CLI/queue defaults
  until a versioned contract or explicit preset selects the new behavior.
- `voice-synth` imports Torch before registry listing; capabilities probes a
  runtime CustomVoice model unless speaker probing is skipped
  (`apps/voice-synth/voice-synth.py:45`, `:738`). Cheap discovery must not call
  that default probe. Split static compatibility, installed assets and live
  readiness, with cached timestamps and an explicit expensive probe.
- No audio streaming transport exists. Complete files/results are the current
  unit. Avoid claiming upstream streaming capabilities as repository features.

## Engine contracts to preserve

| Engine/mode | Supported behavior in current adapters | Explicit limits |
| --- | --- | --- |
| Qwen Base | Reference cloning, local prompt caching, 0.6B/1.7B checkpoints. | Text direction rejected; clone tone means a different reference prompt. |
| Qwen CustomVoice | Built-in speakers and named profiles. | Text direction only on 1.7B; 0.6B rejects it. Feature flag remains authoritative. |
| Qwen VoiceDesign | Description-driven design using 1.7B, then conversion into a clone. | Additional model/lifecycle cost; not ordinary Base direction. |
| Kokoro | Explicit French/English compatible presets, CPU ONNX. | Rejects reference cloning, instructions and seed. |
| Pocket | Explicit per-language provisioned bundle, authorized WAV reference and seed. | Rejects presets/instructions; real-weight availability and listening acceptance must be separately established. |

Evidence: `lib/tts_backends/qwen.py:16`, `:37`,
`lib/tts_backends/kokoro.py:8`, `:42`,
`lib/tts_backends/pocket_tts.py:12`, `:85`.
Qwen remains the default. A lower compute route must be explicit or selected by
an approved preset, with no silent engine, language, identity or control fallback.

## Migration and reliability risks

1. **Identity and reproducibility.** Queue hashes contain the voice slug and
   synthesis parameters, but not a pinned voice revision, reference hash,
   checkpoint revision, backend or asset manifest (`lib/jobqueue.py:39`). A
   mutable voice can change after enqueue. Add a resolved immutable voice/engine
   snapshot at submission; preserve existing IDs and queue records.
2. **Idempotency conflict.** Current Lua deduplicates by `request_id` existence,
   without comparing the submitted content hash (`lib/jobqueue.py:76`). A new API
   should return the same operation for the same key/hash and an explicit
   conflict for the same key/different hash, rather than silently returning an
   unrelated prior result.
3. **Schema coverage.** `JobSpec` is synthesis-only. It lacks CPU backend/assets,
   seed, checkpoint/revision, chunk pause controls, registration, design,
   assembly and export operations (`lib/jobqueue.py:201`). Add an operation
   envelope/version and adapters; do not reinterpret old payloads in place.
4. **Normalization surprises.** `build_speak_argv` silently drops tone in direct
   speaker mode and gives instruction-style precedence (`lib/jobqueue.py:549`).
   Preserve legacy behavior through its adapter; new tools should validate
   unsupported/conflicting controls before queueing and return resolved settings.
5. **Publication differences.** CPU/export paths publish without replacement;
   Qwen `--out-exact` reuses a directory and writes fixed take/result filenames
   (`apps/voice-synth/voice-synth.py:1198`, `:1523`, `:1722`). Use operation-owned
   staging, immutable artifact IDs and a final publication boundary for new jobs.
6. **Prompt trust and portability.** Keep hash/metadata checks before local
   pickle use; never expose arbitrary pickle upload/path execution through an
   API. ZIP exchange intentionally contains portable WAV/JSON and strips derived
   clone caches/tones (`lib/voices.py:300`, `:769`, `:805`). Existing slugs,
   legacy prompt IDs and backup/recovery semantics must remain compatible.
7. **Worker ownership.** Error/timeout invalidates the worker; never publish a
   late result. Pocket's reference/config namespace belongs to the parent,
   is private mode 0700, and is removed only after confirmed reap; unreapable
   workers keep it and cannot be reused (`apps/voice-cpu/voice-cpu.py:88`, `:296`;
   `lib/tts_backends/pocket_tts.py:12`). Do not replace this with child finalizers.
8. **Watcher is broader than its README.** Source supports local executor,
   local sink and mode-dependent credentials. README still describes Vast/S3
   requirements as universal. Concurrency above one is explicitly rejected by
   config. Liveness healthcheck does not establish Redis/model readiness.
   (`apps/job-watcher/job-watcher.py:202`, `:271`; `docker-compose.watcher.yml`).
9. **Public surface boundaries.** Existing manifest execution accepts argv and
   producer commands include destructive flush/delete. New remote tools should
   accept typed operation inputs and managed artifact IDs, not expose generic
   subprocess execution, filesystem paths or queue administration by default.

## Bounded delivery sequence and acceptance gates

| Stage | Concrete delivery | Measurable acceptance |
| --- | --- | --- |
| 0 — Contract freeze | Operation inventory/schema, production/script/segment/take/timeline/render domain, legacy beatsheet/package adapter mapping, synthetic fixtures and voice identity snapshot design. | Every existing command/control maps to an operation or explicit advanced escape hatch; script and timeline revisions/asset references have deterministic serialization; unchanged legacy fixtures retain IDs/fields/defaults; documented unsupported-control errors. |
| 1 — Cheap control plane | Callable registry/capability/artifact/job layer; compact CLI JSON and HTTP transport; thin agent tool adapter using the same handlers. | Discovery/list/get imports no inference runtime and loads no models; routine agent requests have zero audio/base64 and no logs; proposed budgets: default list page ≤20 entries, compact receipt ≤2 KiB, at most 8 core agent tools, verified by serialized payload measurements. |
| 2 — Durable synthesis | Versioned queue envelope, existing Redis adapter, local Qwen worker pool of one, CPU batch adapter, cancellation/deadlines and exclusive publication. | Same key/hash creates one operation; changed hash conflicts; crash/reclaim and timeout tests produce one terminal state and no late published WAV; successful homogeneous jobs share model generation; forced failure starts a new generation; legacy queue payloads still execute. |
| 3 — Required studio editor | Voice library/reference audition/registration and script/timeline editing over the shared production model; per-segment voice/settings, take selection, clip trim/reorder/move, pauses, gain/fades, playback/scrubbing and job status. | Synthetic-fixture browser flow completes script → voices → segment synthesis → take selection → timeline edit → saved/reloaded draft; changing one utterance queues only that utterance; moving/selecting clips queues zero inference jobs; revisions prevent overwriting newer edits; engine-specific controls cannot submit invalid combinations. |
| 4 — Required timeline render/export | Deterministic timeline renderer, ordered clips/overlaps/pauses and source manifests, explicit resampling/export options. | Impulse/sine fixtures verify clip offsets, trims, gain/fades/overlap and total sample duration; rate/channel mismatches fail unless explicit conversion requested; originals remain byte-identical; render pins timeline/script/voice revision and every source hash; a draft edit during render cannot change its result. |
| 5 — Optional wakeups/operations | Completion event outbox, opt-in registered wake target, retries/delivery history, health/readiness and cost telemetry. | No target means zero network delivery; duplicate terminal processing yields one logical completion event; retry failure never reopens synthesis; delivery retries are bounded; receiver event IDs support deduplication; offline contract tests cover success/failure/restart. |
| 6 — Runtime qualification | Explicitly authorized image/runtime tests, real assets, paired cold/warm jobs and human listening. | Report exact image/assets/hardware, model-load count, cold/warm latency, RTF, peak memory and timeout cleanup separately; acoustic acceptance recorded by a person. No speed/quality/default-engine promotion from unit tests. |

The payload/tool-count limits above are proposed product budgets, not measured
current performance. Freeze them against representative agent workflows before
implementation. Keep heavy synthesis behind queue jobs; status defaults to a
compact summary and expands only when explicitly requested. Completion events
can reduce polling/context use without requiring any specific agent framework.

Relevant existing regression suites include `tests/test_registry_security.py`,
`tests/test_voice_registration_atomicity.py`, `tests/test_clone_prompt_identity.py`,
`tests/test_clone_prompt_compatibility.py`, `tests/test_fidelity.py`,
`tests/test_audio_export.py`, `tests/test_jobqueue.py`,
`tests/integration/test_jobqueue_redis_integration.py`,
`tests/test_job_runner.py`, `tests/test_job_watcher.py`,
`tests/test_voice_cpu_batch.py`, `tests/test_cpu_output_publication.py` and
`tests/test_qwen_worker_diagnostics.py`. Historical remediation/benchmark
documents explain constraints but do not replace fresh runtime or listening
qualification: `docs/TTS_AUDIT_REMEDIATION.md`, `docs/TTS_PERFORMANCE.md`,
`docs/TTS_MEMORY.md`, `docs/TTS_BENCHMARK.md`.
