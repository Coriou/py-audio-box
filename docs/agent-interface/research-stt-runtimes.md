# CPU-first transcription: runtimes, hardware and evaluation

Research date: **2026-10-15**. Source baseline:
`ae67cd6a572e7105b7d41365aac0be4059d79584`. Read alongside
[PLAN.md](PLAN.md), [AUDIT.md](AUDIT.md), and the
[existing serving audit](audit-self-hosted-solutions.md).
The user subsequently confirmed French, English and mixed-language speech as
primary and authorized bounded public-data CPU/larger-model evaluation. STT is
now a requested capability; the concrete runtime/profile choices below remain
recommendations, not an adopted implementation or a model promotion.

The subsequent bounded [CPU probe](evaluation/stt/RESULTS.md) did **not** qualify
either tested faster-whisper profile: both timed out, with only two English
turbo clips completed. A whisper.cpp tiny diagnostic was blocked before inference
by an image pull. These failures and the pull-cap overrun are explicit in the
receipt; source support must not be presented as a working CPU starter.

## Recommendation

Start with one multilingual Whisper-family runtime and explicit quality profiles.
Compare the existing faster-whisper `small` INT8 path with `large-v3-turbo` INT8
on the actual Intel host. Qualify whisper.cpp as the slim C++/quantized alternative
before choosing the release worker; do not ship two Whisper implementations
without a demonstrated reason. The best initial challenger is sherpa-onnx with
Parakeet v3 INT8, if a later matched test shows a useful FR/EN tradeoff. Keep
Moonshine/Vosk as specialized choices, rather than universal fallback engines.

A CPU fallback must preserve the selected model/language/operation, or fail with
an actionable resource message. It must not quietly select an English-only model,
smaller checkpoint, different quantization or lossy speech filtering. Better
hardware can select a qualified device profile through the same public contract;
an upgrade in hardware is not evidence of higher transcript accuracy.

## Concrete hardware boundary

Parent-agent host inspection established Darwin x86_64, Intel i7-8700B at 3.20 GHz,
6 physical/12 logical cores, 64 GiB host RAM, AVX2/FMA/F16C and no reported AVX512.
Docker 29.4 runs an x86_64 Linux VM with 12 vCPUs and about 15.7 GiB RAM. These
are the acceptance target, not a claim about all older Intel Macs. The VM memory
allocation, host contention and Docker overhead matter even with 64 GiB host RAM.

The CPU release target is a Linux inference container running in that VM. It
cannot use macOS Metal inside its Linux process. Native macOS whisper.cpp/MLX
workers are separate host adapters. Docker Model Runner can use host-side GPU
inference on supported platforms; its engines run outside containers on macOS
and Windows. No STT route was established here for Model Runner, so its existence
does not qualify this transcription worker for Metal.
[Docker GPU scope](https://docs.docker.com/desktop/features/gpu/),
[Model Runner architecture](https://docs.docker.com/ai/model-runner/).

An old Mac also needs a supported Docker/OS combination; current Docker Desktop
supports the current and two previous major macOS releases and lists a 4 GB RAM
minimum. That installer minimum is not an ASR capacity claim.
[Mac installation requirements](https://docs.docker.com/desktop/setup/install/mac-install/).

## Existing repository paths

| Path | Source evidence | Reuse / required change |
| --- | --- | --- |
| `lib/tts.py:83`, `:567` | Default Whisper model is `small`; CPU compute is INT8; faster-whisper model loader derives device through Torch/CUDA. | Reuse knowledge and controls; extract into lightweight ASR worker rather than loading TTS dependencies for discovery. |
| `lib/tts.py:580`, `:611` | Reference transcription uses beam 5 and `vad_filter=True`, returning joined text, mean log probability, detected language mapped into the Qwen language list. Unknown mapping defaults to English. | Reference-selection contract only. General STT must retain native language codes/unknown/mixed evidence; no English coercion. |
| `apps/voice-clone/voice-clone.py:530` onward | Clean reference candidate scoring, transcription cache, explicit transcript/reference-language override and quality gate. | Reuse immutable media/reference identity concepts; reference quality thresholds are not calibrated transcript acceptance/confidence. |
| `apps/voice-synth/voice-synth.py:238` | One cached ASR model, up to 16 transcript results keyed by path/mtime/size; beam 2, VAD, word timestamps. | Useful warm model/result reuse but service cache must key exact audio/model/options/context hashes and have explicit memory lifetime. |
| `apps/voice-synth/voice-synth.py:269`, `:282` | QA compares ASR output to authored text; word alignment is explicitly `transcription_timestamps`. | Preserve QA; these timestamps do not force-align approved script words. Forced alignment needs its own capability/job and source-text identity. |
| `lib/vad.py:28`, `:67` | Silero loads via Torch Hub with `trust_repo=True`, then merges/filters spans using caller thresholds. | Pin/provision code+weights; preserve media offsets and original audio. Extraction thresholds must not silently discard STT speech. |
| `lib/audio.py`, `apps/voice-split/voice-split.py` | ffmpeg decode/normalize/trim and source extraction; split pipeline includes VAD/Demucs. | Reuse explicit decode/channel/range stages, not mandatory vocal separation. Arbitrary media STT should not require Demucs or clone registration. |
| `lib/jobqueue.py:201`, `apps/job-runner` | Legacy jobs synthesize speech, not transcribe general media. | Add typed STT operation to the new durable service contract; no competing Redis queue for new service jobs. |

No generic STT CLI/API with media ingest, durable progress, exact transcript
artifacts, context revision and retrieval was found. Current reference/QA reuse
does not establish that service. No private registry or work audio was inspected.

## Runtime and hardware matrix

**Evidence:** D = official documentation/model card; S = selected source inspected;
U = not qualified by an actual run. Support in source, a binary wheel and a
published container manifest are different claims. Unless the evaluation receipt
says otherwise, every cell is D/S, not local runtime proof.

| Runtime | x86_64 / arm64 CPU | Linux NVIDIA container | macOS native GPU | AMD / other accelerators | Product implication |
| --- | --- | --- | --- | --- | --- |
| whisper.cpp | C++ CPU and quantized models; official CPU images document linux/amd64 + linux/arm64. CPU ISA is build-specific. | CUDA image documented for amd64; driver/toolkit/model compatibility still required. | Metal/Core ML support documented, strongest Apple-specific route is native. | HIP/ROCm, Vulkan and OpenVINO code paths documented; qualify exact build/device, not all GPUs. | Strong slim CPU alternative; CLI/server are building blocks, not durable production state. |
| faster-whisper / CTranslate2 | Prebuilt CT2 CPU supports x86_64 with at least SSE4.1 and AArch64; ISA dispatch selects AVX/AVX2/AVX512. | CUDA/cuDNN version compatibility required. A CT2 image is not automatically a configured faster-whisper server. | No Metal backend in this CT2 path; CPU execution is distinct from native Apple acceleration. | ROCm not established in the inspected CT2 backend; do not promise it from generic Torch support. | Lowest migration cost; explicit INT8 effective type and bounded batch/worker count. |
| sherpa-onnx | CPU Linux/macOS and embedded/arm64 support; ONNX export/model/operator combinations matter. | CUDA provider where built and available; exact package/image/architecture must be qualified. | Native CoreML provider exists; not Metal inside Linux Docker. | Provider support is selected at build/runtime; universal ROCm support not established. | One C++/ONNX runtime can serve many model families, offline and streaming, but each capability needs a model-specific contract. |
| Vosk | CPU-oriented Kaldi toolkit, desktop/embedded platforms; small and large language-specific models. | Specialized GPU builds are not the default install contract. | CPU/native support; no established Metal route. | Not established for proposed release. | Useful constrained streaming/vocabulary niche, not automatic FR/EN mixed-media replacement. |
| Moonshine Voice | Linux x86_64/arm64 prebuilt libraries and on-device CPU focus; separate streaming/legacy model families. | No qualified CUDA container route established. | Native platform APIs exist; CPU is the baseline here. | Not established for proposed release. | Strong live/English niche, but current official model catalogue has **no French entry**. |
| Qwen3-ASR Transformers / vLLM | CPU experimentation possible with appropriate runtime; official quickstarts are GPU-oriented and do not establish old-Intel performance. | Official Docker and serving paths documented, with variant/runtime constraints. | Not established by upstream GPU examples; separate implementations require qualification. | Not established from upstream example alone. | Larger/context-capable candidate, not the minimal CPU dependency. |

Primary hardware/runtime sources:
[whisper.cpp](https://github.com/ggml-org/whisper.cpp),
[CT2 hardware](https://opennmt.net/CTranslate2/hardware_support.html),
[CT2 install](https://opennmt.net/CTranslate2/installation.html),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx),
[sherpa provider source](https://github.com/k2-fsa/sherpa-onnx/blob/fa29eb91c80335fc70af869a7bb459578f90e5ad/sherpa-onnx/csrc/session.cc),
[Vosk install](https://alphacephei.com/vosk/install),
[Moonshine quickstart](https://moonshine-voice.readthedocs.io/en/latest/quickstart/),
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).

whisper.cpp exposes `GGML_NATIVE`, AVX/AVX2/FMA/F16C and backend-variant build
options. A binary optimized on a newer build host is not evidence it runs on an
older target; pin the release image and verify CPU features inside the VM.
Consider a declared portable baseline plus runtime-dispatched variants rather
than uncontrolled native compilation. Quantization is an exact asset/profile
choice, not a universal speed or accuracy improvement.
[Build options](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/ggml/CMakeLists.txt).

CT2 can implicitly convert unsupported compute types; its CPU INT8 route may be
`int8_float32`. Record requested and effective types and use capability queries;
CPU FP16/BF16 labels do not prove optimized arithmetic. Do not infer supported
GPU architecture solely from the generic minimum in hardware docs; the selected
CUDA runtime/wheel and shipped kernels determine actual execution.
[Compute-type support and conversions](https://opennmt.net/CTranslate2/quantization.html).
Linux GPU deployment additionally needs the host driver and supported container
runtime integration.
[NVIDIA container prerequisites](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Model and language choices

| Model/family | Language/task fit | Code versus model terms | Recommendation |
| --- | --- | --- | --- |
| Whisper multilingual `base` / `small` | FR/EN, language identification and transcription; `.en` variants are English-only. Larger checkpoints remain the same general family. | Whisper code and weights MIT; conversions/runtimes have their own provenance. [Official Whisper](https://github.com/openai/whisper). | `small` INT8 or equivalent quantized variant as first baseline; `base` only as an explicit economy profile. |
| Whisper `large-v3-turbo` | Multilingual transcription, separate tradeoff from full large-v3. | Pin exact converted checkpoint/tokenizer, not only a marketing alias. [Official model family](https://github.com/openai/whisper), [conversion used for probe](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo). | Larger CPU comparator; upgrade only on measured FR/EN/mixed quality and bounded resource fit. |
| Parakeet TDT 0.6B v3 | 25 European languages including FR/EN; automatic language handling. NeMo/PyTorch and CPU ONNX are separate runtime paths. | NVIDIA model card CC-BY-4.0; runtime/export licenses separate. [Model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). | First modern challenger. Sherpa documents CPU INT8 export/use; VAD-simulated streaming is distinct from native streaming. [CPU example](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html). |
| Canary 1B v2 | Multilingual ASR/translation with explicit task/language handling; NVIDIA-optimized deployment. | Model card CC-BY-4.0. [Card](https://huggingface.co/nvidia/canary-1b-v2). | Defer initial old-Intel comparison; evaluate only for a demonstrated task/language gap. |
| Qwen3-ASR 0.6B / 1.7B | 30 languages plus 22 Chinese dialects, FR/EN; separate forced-aligner model. | Apache-2.0 model cards/repo; runtime and conversions separately pinned. [0.6B card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [1.7B HF card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf). | Optional modern challenger. Sherpa has Qwen3-ASR integration, so GPU-oriented upstream examples do not mean the family is GPU-only. [Sherpa integration](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/index.html). |
| Moonshine streaming | Language-specific streaming models; English supported, French absent from inspected catalogue. | Current streaming models/code default MIT; specified legacy non-English models use separate community terms. [Catalogue](https://moonshine-voice.readthedocs.io/en/latest/models/available-models/), [license](https://github.com/moonshine-ai/moonshine/blob/234f60faa0eb388b01cdf7e60aca232af37aefda/LICENSE). | Optional English live transcription, not common FR/EN/mixed baseline. |
| Vosk language models | Small FR/EN models and larger language-specific alternatives, configurable vocabulary. | API Apache-2.0; model catalogue licenses vary by exact model. [API](https://github.com/alphacep/vosk-api), [model table](https://alphacephei.com/vosk/models). | Optional low-footprint/grammar-constrained mode; do not auto-route arbitrary mixed-language material into one monolingual model. |

No upstream leaderboard score is used here to rank accuracy on the user's audio.
Language lists do not establish natural code-switch competence. A transducer's
token timestamps, Whisper's word estimates, forced alignment, diarization and
speaker identification remain distinct capabilities with separate costs.

## Smart context and transcript fidelity

Use bounded declared input context: expected language(s), domain terms/names,
acronyms, optional brief topic and a project glossary revision. Preserve context
exactly in the accepted plan; record which fields the backend actually uses.
Do not use a local text-rewriting LLM by default. The transcript must remain
separate from a later edited/cleaned document and from the authored TTS script.

- faster-whisper exposes `initial_prompt`, hotwords, language detection controls,
  previous-text conditioning and hallucination-related thresholds. They affect
  decoding and can introduce bias/repetition; evaluate false insertions as well
  as term recall.
  [Transcribe source](https://github.com/SYSTRAN/faster-whisper/blob/a52b026ac844c02117865e61968547639b60a6f8/faster_whisper/transcribe.py).
- Sherpa hotwords require supported transducer decoding and
  `modified_beam_search`; greedy decoding and other model classes cannot be
  assumed to honor them. Report unsupported controls before queueing.
  [Contextual biasing](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html).
- Qwen3-ASR's package accepts context and explicit language alongside audio;
  carry those identities into chunks. Its streaming prefix/rollback behavior is
  not equivalent to independently transcribing arbitrary fragments.
  [Inference source](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/qwen3_asr.py).
- Context should never override clear contrary audio without leaving observable
  raw/derived layers. Score correct terms and deliberately absent glossary terms;
  a context system that simply inserts the glossary is a failure.

VAD is an optimization and segmentation policy, not a fact oracle. Low speech,
whispers, short words, far-field voices, overlap and music can be filtered out.
Store original media, decoded audio identity, complete kept/dropped spans,
padding/overlap and offset mapping. Compare VAD-on/off on the same clips; prefer
recoverable metadata over destructive pre-trimming. For long audio, preserve
segment continuity and chunk checkpoints, and identify overlaps/deduplication
without rewriting speech. Endpoint streaming and file-job progress are separate.

## Minimal selection and resource policy

1. Discover installed profiles and readiness without loading models. Resolve
   requested languages/task/context/timestamps and immutable source range.
2. Select one installed, qualified profile from explicit quality/device ceilings.
   Proposed initial profiles: multilingual CPU `small` and a separately installed
   larger multilingual profile. A faster draft profile is opt-in and visibly
   identified. No automatic English-only, smaller-model or VAD downgrade.
3. Pin runtime/image/model/tokenizer/precision/decoder/VAD/context identities in
   the job. If a chosen accelerated path fails, only an explicitly approved
   same-model CPU fallback may run, under a new bounded attempt with recorded
   reason. Otherwise return failure/action; do not hide it behind “auto”.
4. Start with one host-wide inference slot shared with TTS. Fair admission and
   idle unload avoid starving speech generation. Keep transcripts/artifacts
   paginated and downloadable; ordinary agent status must not dump long text.
5. Enforce compressed bytes, decoded duration/channels/rate, temporary storage,
   queue size, resident-model memory, CPU threads, container memory and wall-time.
   Decode/provision budgets count alongside inference. Large inputs are chunked
   and checkpointed under one logical job, not loaded into an unbounded array.

For the actual Intel target, the evaluation uses 4 CPU threads, 4 Docker CPUs,
6 GiB container memory and one worker. These are test ceilings, not promised
requirements or product throughput. Keep headroom for the 15.7 GiB VM and its
other workloads. Admission estimates are not an OOM guarantee; timeouts must
terminate/reap the worker and fence late publication. An unreapable worker
quarantines the slot instead of spawning overlapping inference.

## Cheap reproducible evaluation design

The first probe is deliberately small: pinned licensed FR/EN public read speech,
synthetic silence/noise, constructed FR→EN concatenation clearly labeled as
**not natural code switching**, a low-amplitude derivative, paired VAD settings
and repeated warm decoding. It can expose basic failures; it cannot certify
arbitrary-media accuracy or choose the final default. Public audio/transcripts
come from [Google FLEURS](https://huggingface.co/datasets/google/fleurs), CC-BY-4.0.
Exact rows, files, revisions and hashes belong in its receipt.

After the initial probe, extend only the two strongest configurations to a
fixed approximately 20-minute evaluation set:

| Stratum | Proposed inexpensive qualification set |
| --- | --- |
| Clean FR/EN | 16 short clips per language, speaker-balanced and held out from tuning. |
| Names/jargon/numbers | 8 consented/public human clips with proper nouns, acronyms, dates, amounts and negation; score exact critical tokens. |
| Natural code switching | 4 genuine FR/EN mixed utterances with manual language spans; constructed concatenations stay a separate stratum. |
| Acoustic difficulty | A few lawful noisy/far-field/quiet/overlap/codec variants of the same human clips; document SNR/transform/channel policy. |
| Long continuity | One genuine ~10-minute segment with manually checked reference and boundary/offset annotations. A repeated/concatenated short fixture tests mechanics only. |
| No speech | Digital silence, room noise and non-speech/music with known rights; count hallucinated words and false speech durations. |

Human/source references must be fixed independently of engine output. Do not use
TTS-generated speech as the principal accuracy corpus, tune on the held-out set,
or compare models on different clipping/language/normalization settings.

Measure and retain:

- Raw transcript; raw WER/CER; normalized WER/CER with a pinned, accent/digit-
  preserving normalizer. Report corpus-weighted edits/word count plus per-language
  and challenge strata. Empty-reference WER is undefined: report insertion counts.
- Critical name/number/acronym/negation recall and false glossary insertions;
  language detection and code-switch omissions separately from aggregate WER.
- VAD dropped-speech duration/word omissions, with and without padding; no-speech
  hallucinations; chunk boundary duplication/deletion and timestamp drift.
- Runtime imports, model load, first decode and warm decode separately; RTF is
  processing seconds/audio seconds. Report end-to-end decode/queue/export wall
  time separately. Streaming first partial/final latency applies only to streaming.
- Requested/effective compute type and provider, load count, thread/batch settings,
  image/model hashes, host/VM resource constraints, CPU contention and failures.
- Process maximum RSS plus cgroup memory peak/sample method; a sampled cgroup
  peak includes cache/container processes and can miss spikes. Repeated warm
  decoding does not imply the OS page cache was cold at first load.

Acceptance should be based on matched quality/resource tradeoffs, not an arbitrary
unmeasured accuracy target. Publish exact failures/timeouts and successful partial
results. A faster model may be accepted only if its permitted quality tradeoff is
visible and supported languages/context/timestamps remain truthful.

## Source snapshot and evaluation handoff

Official GitHub commit endpoints were checked 2026-10-15:

| Runtime | Inspected revision | Code license |
| --- | --- | --- |
| [whisper.cpp](https://api.github.com/repos/ggml-org/whisper.cpp/commits/master) | `0c7be6963710`, 2026-10-12 | MIT |
| [faster-whisper](https://api.github.com/repos/SYSTRAN/faster-whisper/commits/master) | `a52b026ac844`, 2026-10-10 | MIT |
| [CTranslate2](https://api.github.com/repos/OpenNMT/CTranslate2/commits/master) | `8d91aa082250`, 2026-10-13 | MIT |
| [sherpa-onnx](https://api.github.com/repos/k2-fsa/sherpa-onnx/commits/master) | `fa29eb91c803`, 2026-10-15 | Apache-2.0 |
| [Vosk API](https://api.github.com/repos/alphacep/vosk-api/commits/master) | `4ab9821510b4`, 2026-10-15 | Apache-2.0 |
| [Moonshine](https://api.github.com/repos/moonshine-ai/moonshine/commits/main) | `234f60faa0eb`, 2026-08-24 | MIT core; model/third-party exceptions listed upstream |
| [Qwen3-ASR](https://api.github.com/repos/QwenLM/Qwen3-ASR/commits/main) | `7c6daf77a242`, 2026-06-26 | Apache-2.0 |

Selected-source claims do not mean the entire runtime was audited. The evaluation
uses pinned **installed versions**, which may differ from source tips above.
Actual results and reproduction instructions are recorded under
[evaluation/stt](evaluation/stt/); they must remain separate from this source
matrix and from future product/runtime acceptance.
