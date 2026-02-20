# voice-clone

Clone your voice from a short reference recording and synthesise new speech,
running entirely on CPU.

**Engine:** [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
**Transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (int8 / CPU)

---

## Quick start

```bash
# End-to-end smoke test (downloads a demo ref clip, runs the full pipeline)
./run voice-clone self-test

# Stage 1–3 only: normalise → VAD-trim → transcribe a ref clip
./run voice-clone prepare-ref --ref-audio /work/myvoice.wav

# Full clone + synthesis
./run voice-clone synth \
  --ref-audio /work/myvoice.wav \
  --text "Hello, this is my cloned voice speaking."
```

Outputs land in `/work` by default:

- `voice_clone_YYYYMMDD_HHMMSS.wav` — synthesised audio
- `voice_clone_YYYYMMDD_HHMMSS.meta.json` — full provenance (timings, RTF, transcript, model)

---

## Reference audio tips

- **Best cloning quality:** a **clean, noise-free, 5–10 s** mono recording of your
  voice with no music or background noise.
- The pipeline auto-selects the best 3–12 s segment using Silero VAD.
- Override with `--ref-start / --ref-end` if the auto-selection is wrong.
- A reference clip produced by `voice-split` is ideal input.

---

## Sub-commands

### `synth` — full pipeline

```bash
./run voice-clone synth \
    --ref-audio /work/myvoice.wav \
    --text "Hello, this is my cloned voice speaking."
```

**Required**

| Flag               | Description                                  |
| ------------------ | -------------------------------------------- |
| `--ref-audio PATH` | Reference voice recording (WAV / MP3 / etc.) |

One of `--text` or `--text-file` must be provided.

**Input text**

| Flag               | Type   | Default | Description                                        |
| ------------------ | ------ | ------- | -------------------------------------------------- |
| `--text TEXT`      | `str`  | —       | Text to synthesise                                 |
| `--text-file FILE` | `path` | —       | Read synthesis text from a file                    |
| `--ref-text TEXT`  | `str`  | —       | Transcript of ref audio — skips auto-transcription |

**Reference segment**

| Flag                  | Type     | Default | Description                                                   |
| --------------------- | -------- | ------- | ------------------------------------------------------------- |
| `--ref-start SEC`     | `float`  | —       | Manual segment start (seconds)                                |
| `--ref-end SEC`       | `float`  | —       | Manual segment end (seconds)                                  |
| `--ref-language LANG` | `choice` | `Auto`  | Language of the reference audio; `Auto` = whisper auto-detect |
| `--force-bad-ref`     | flag     | off     | Bypass the transcript quality gate (low avg_logprob warning)  |

**Language & synthesis**

| Flag                   | Type     | Default | Description                                                                     |
| ---------------------- | -------- | ------- | ------------------------------------------------------------------------------- |
| `--language LANG`      | `choice` | `Auto`  | Target synthesis language; `Auto` detects from text (langid), then ref language |
| `--style PRESET`       | `str`    | —       | Style preset from `styles.yaml` (e.g. `serious_doc`, `nature_doc`, `excited`)   |
| `--prompt-prefix TEXT` | `str`    | —       | Text prepended to synthesis text (after style expansion)                        |
| `--prompt-suffix TEXT` | `str`    | —       | Text appended to synthesis text (after style expansion)                         |

**Generation knobs** (passed to Transformers `model.generate`)

| Flag                         | Type    | Default | Description              |
| ---------------------------- | ------- | ------- | ------------------------ |
| `--temperature FLOAT`        | `float` | —       | Sampling temperature     |
| `--top-p FLOAT`              | `float` | —       | Top-p nucleus sampling   |
| `--repetition-penalty FLOAT` | `float` | —       | Repetition penalty       |
| `--max-new-tokens INT`       | `int`   | —       | Maximum generated tokens |

**Infrastructure**

| Flag              | Type   | Default                         | Description                                                                             |
| ----------------- | ------ | ------------------------------- | --------------------------------------------------------------------------------------- |
| `--model ID`      | `str`  | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3-TTS model (HF repo ID or local path)                                              |
| `--whisper-model` | `str`  | `small`                         | faster-whisper model for ref transcription                                              |
| `--x-vector-only` | flag   | off                             | Build voice prompt from speaker embedding only — skips ref_text (faster, lower quality) |
| `--threads N`     | `int`  | `8`                             | CPU threads for torch + whisper                                                         |
| `--seed N`        | `int`  | —                               | Random seed for reproducible synthesis                                                  |
| `--force`         | flag   | off                             | Ignore all cached results and recompute from scratch                                    |
| `--out DIR`       | `path` | `/work`                         | Output directory for WAV + meta.json                                                    |

### `prepare-ref` — inspect pipeline stages 1–3

Runs normalise → VAD → score candidates → transcribe and prints the cached
paths + transcript. Useful for verifying the ref clip before a long synthesis run.

```bash
./run voice-clone prepare-ref --ref-audio /work/myvoice.wav
```

Accepts all the same flags as `synth` except `--text*`, `--style`, `--prompt-*`,
generation knobs, and `--out`.

### `self-test` — smoke test

Downloads the Qwen3-TTS demo reference clip (once, cached), runs the full
pipeline on a short sentence, and asserts basic sanity (duration, no NaNs,
peak amplitude).

```
./run voice-clone self-test
```

---

## Language detection

Two independent language flags exist because the reference and synthesis languages
may differ (e.g. a Spanish-spoken reference cloning English speech):

- `--ref-language` — language _of the reference recording_. `Auto` means
  let faster-whisper detect it. Set explicitly if whisper misidentifies your accent.

- `--language` — language _of the synthesis text_. `Auto` (default):
  1. Run langid on the target text (≥ 3 words)
  2. Fall back to the detected ref language
  3. Default to `English`

Supported languages: `Chinese`, `English`, `Japanese`, `Korean`, `German`,
`French`, `Russian`, `Portuguese`, `Spanish`, `Italian`.

---

## Transcript quality gate

After transcribing the reference segment, the pipeline checks the faster-whisper
`avg_logprob` confidence score. If it falls below the threshold (`-1.2`), synthesis
is blocked with a clear diagnostic:

```
⚠  LOW TRANSCRIPT CONFIDENCE: avg_logprob=-1.45 (threshold -1.2).
This may result in poor cloning quality. Try:
  • --ref-start / --ref-end to select a cleaner segment
  • a higher-quality reference recording
  • --force-bad-ref to proceed anyway
```

Use `--force-bad-ref` to bypass the gate and synthesise anyway (useful for
exploring edge cases or when you know the recording is acceptable).

---

## Cache layout

All intermediate artefacts are stored under `/cache/voice-clone/`:

```
/cache/voice-clone/
  refs/
    <sha256>/
      ref_normalized.wav           # 16 kHz mono loudnorm
      candidates/
        candidate_00.wav           # top-N VAD candidates (trimmed)
        candidate_01.wav
        scores.json                # per-candidate whisper scores
      best_segment.json            # winning segment bounds + score
      ref_segment.wav              # 24 kHz final trimmed clip (ready for Qwen3)
      ref_transcript.json          # faster-whisper transcript + confidence
      pipeline_state.json          # lightweight run summary
  prompts/
    <hash>_<model>_full_v1.pkl     # cached voice clone prompt tensors
    <hash>_<model>_full_v1.meta.json  # provenance sidecar (readable without unpickling)
    <hash>_<model>_xvec_v1.pkl     # x-vector-only variant
    <hash>_<model>_xvec_v1.meta.json
  self-test/
    self_test_ref.wav              # demo ref clip (downloaded once)
```

Re-running `synth` on the same reference is fast: stages 1–4 are all cache
hits and only synthesis itself runs.

---

## Models

| Model                           | Size  | Notes                             |
| ------------------------------- | ----- | --------------------------------- |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6 B | **Default** — CPU practical       |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7 B | Higher quality, ~3× slower on CPU |

Model weights are downloaded once by `qwen-tts` and cached under
`$XDG_CACHE_HOME/huggingface` (→ `/cache/huggingface` inside the container).

---

## CPU performance expectations

Measured on a 16-core Mac Mini (Apple Silicon, `--threads 8`, 0.6B model):

| Stage           | First run (cold) | Subsequent runs  |
| --------------- | ---------------- | ---------------- |
| Model download  | ~5 min (one-off) | skipped          |
| Model load      | ~5 min           | ~5 min           |
| Voice prompt    | ~45 s            | **cached → 0 s** |
| Synthesis (RTF) | ~30–35×          | ~30–35×          |

So synthesising 5 s of speech takes ~2–3 min of synthesis time once the model
is loaded. On a **warm second run** (same reference audio), stages 1–4 are all
cache hits — only model load + synthesis run.

Use `--threads 12` or `--threads 14` to speed up; higher thread counts give
decreasing returns above physical core count. The 1.7B model is ~3× slower.

---

## `--x-vector-only` mode

Skips the faster-whisper transcription stage and builds the voice prompt from
speaker embedding only. Useful when you cannot transcribe the reference (e.g.
unknown language) or want to iterate quickly. Quality is typically lower.

---

## Expressive synthesis tips

Qwen3-TTS is steered by the text itself — punctuation and wording matter:

- Use `...` for natural pauses.
- Use `!` / `?` for emphasis / questions.
- Use `--style` for one-word delivery presets (`serious_doc`, `nature_doc`,
  `excited`, `whisper`, `audiobook`, …). See `styles.yaml` for the full list.
- Use `--prompt-prefix` for custom style instructions, e.g.
  `--prompt-prefix "Speak slowly and thoughtfully."`
