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

```
./run voice-clone synth \
  --ref-audio PATH   # your voice recording (WAV/MP3/etc.)
  --text TEXT        # what to say  (or --text-file FILE)
  [--ref-text TEXT]  # transcript of ref audio (skips auto-transcription)
  [--ref-start SEC]  # manual segment start
  [--ref-end   SEC]  # manual segment end
  [--prompt-prefix TEXT]  # prepend to text (style steering)
  [--prompt-suffix TEXT]  # append  to text (style steering)
  [--out DIR]        # output directory (default: /work)
  [--model ID]       # Qwen3-TTS HF repo id or local path
  [--language LANG]  # English (default), Chinese, French …
  [--x-vector-only]  # skip transcription; lower quality
  [--threads N]      # CPU thread count (default: 8)
  [--seed N]         # reproducible generation
  [--force]          # ignore all caches
```

### `prepare-ref` — inspect pipeline stages 1–3

Runs normalise → VAD → transcribe and prints the cached paths + transcript.
Useful for verifying the ref clip before a long synthesis run.

```
./run voice-clone prepare-ref --ref-audio /work/myvoice.wav
```

### `self-test` — smoke test

Downloads the Qwen3-TTS demo reference clip (once, cached), runs the full
pipeline on a short sentence, and asserts basic sanity (duration, no NaNs,
peak amplitude).

```
./run voice-clone self-test
```

---

## Cache layout

All intermediate artefacts are stored under `/cache/voice-clone/`:

```
/cache/voice-clone/
  refs/
    <sha256>/
      ref_normalized.wav       # 16 kHz mono loudnorm
      ref_segment.wav          # 24 kHz 3–12 s trimmed clip
      vad_best_segment.json    # VAD-selected bounds
      ref_transcript.json      # faster-whisper transcript + confidence
      pipeline_state.json      # lightweight run summary
  prompts/
    <hash>_<model>_full.pkl    # cached voice clone prompt tensors
    <hash>_<model>_xvec.pkl    # x-vector-only variant
  self-test/
    self_test_ref.wav          # demo ref clip (downloaded once)
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

Qwen3-TTS understands punctuation and semantic context:

- Use `...` for natural pauses.
- Use `!` / `?` for emphasis / questions.
- Use `--prompt-prefix` for style instructions, e.g.
  `--prompt-prefix "Speak slowly and thoughtfully."`
