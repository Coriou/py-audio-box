# Native Apple cloning: source implementation, runtime unvalidated

Cloning model downloads and runtime tests are outside this work's scope. This
page describes source contracts, not successful Apple cloning or performance.
Kokoro is a preset engine and cannot replace reference cloning.

| Path | Source state | Qualification |
| --- | --- | --- |
| Qwen 12Hz 0.6B Base, EN/FR, MPS FP32 SDPA | Implemented native prepare/save/version/alias-to-generation path; Mac descriptors added | Runtime unvalidated; installed 30 s generation budget may be insufficient |
| Qwen 0.6B CustomVoice, Ryan | Separate preset path, not Base cloning | Diagnostic FP32 WAV only; about 74 s generation for 4 s audio on the tested runner |
| Qwen FP16 | Explicit diagnostic selector only | Pretrained CustomVoice sampling failed with invalid probabilities; origin within the model is not established; not a production default |
| Pocket native CPU | No native activation/adapter/lock | Not implemented; existing Linux adapter is not Mac evidence |
| MLX or Pocket Metal | No adapter | Not implemented |

The EN/FR `qwen-0.6b-base-macos-*.expected.json` descriptors retain the existing
Base revision `5d83992436eae1d760afd27aff78a71d676296fc`, file origins, sizes and
hashes. They use the existing Mac dependency lock and declare only MPS/FP32/SDPA.
No precision fallback, unsafe half cast, watermark bypass, CPU substitution or
checkpoint discovery is added. The [pinned Base config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/5d83992436eae1d760afd27aff78a71d676296fc/config.json)
identifies Base/0b6/12Hz and includes English and French. Descriptors are inputs
to managed provisioning; they grant no qualification or admission authority.

The [official Qwen instructions](https://github.com/QwenLM/Qwen3-TTS/blob/main/README.md)
distinguish Base cloning from CustomVoice presets. The pinned `qwen-tts==0.1.1`
wheel's `qwen_tts/inference/qwen3_tts_model.py` confirms Base-only clone prompts,
ICL reference transcript requirements and reuse through `voice_clone_prompt`.
The native backend constructs this prompt once per request and passes it to
generation once; it does not retain a loaded model or prompt across jobs.
Single-load-per-job behavior remains explicit. Weights retain their separate
Apache-2.0 license; project source remains AGPL-3.0-only. The audited local wheel
modification/provenance remains described in [native dependencies](NATIVE-DEPENDENCY.md).

The public source path requires an owned canonical mono PCM16 WAV and explicit
authored reference transcript, a separate audition text and full-range tick
selection. Prepared reference and audition remain distinct. Save creates a
managed immutable version; an alias (including a `default.*` convention) resolves
to that version before generation. Native reconstruction binds the asset hash,
transcript, language, checkpoint and managed version to an original trusted
template. Supplied versions cannot substitute a checkpoint, family, device or
reference. Archive affects catalogue visibility, not frozen job authority.
Existing ownership, cancellation, original deadlines, private input transport,
confirmed reaping and archive membership checks remain required.

Native startup retains sealed storage ownership checks. Full model/reference
hashing and preparation selection execute once in the shared engine before
backend construction, avoiding duplicate multi-GB reads. The backend additionally
checks actual upstream model dtype as well as device. These are source-level
efficiency/correctness changes, not measured latency improvements. Existing
30 s installed request and separate 120 s startup/150 s job ceilings remain;
the diagnostic preset measurement does not make those budgets usable for Base.

[Pocket v3.3.0 documentation](https://github.com/kyutai-labs/pocket-tts/blob/v3.3.0/README.md)
describes a CPU-oriented engine and separate language models. This repository's
pinned Pocket French bundle is French-specific; multilingual upstream support
does not make that bundle English-capable. No gated model terms are accepted,
credentials read, or new engine family silently installed by this source audit.
