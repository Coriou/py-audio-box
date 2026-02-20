# voice-synth

DevX-first synthesis rig built on cached voice-clone prompts.

Once a voice has been prepared with `voice-clone`, `voice-synth` lets you iterate
fast — no VAD, no Whisper, no prompt extraction on repeated runs.
Just load the cached prompt and generate.

**Clone engine:** [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
**Design engine:** [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (`design-voice` only)
**QA transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (int8 / CPU, optional)

---

## Quick start

```bash
# See all voices you have cached (named + legacy)
./run voice-synth list-voices

# Synthesise using a named voice (created by voice-split/voice-clone --voice-name)
./run voice-synth speak \
    --voice david-attenborough \
    --text "Hello, this is my cloned voice."

# Also works with a legacy hex ID from list-voices
./run voice-synth speak \
    --voice 3fa8c1 \
    --text "Hello from a legacy prompt."

# Generate 4 takes with different seeds and print an intelligibility scoreboard
./run voice-synth speak \
    --voice david-attenborough \
    --text "The forest was silent except for the distant call of birds." \
    --variants 4 --qa

# Apply a style preset
./run voice-synth speak \
    --voice david-attenborough \
    --text "Welcome to the natural history of the Earth." \
    --style nature_doc

# Design a brand-new voice and register it by name, then use it immediately
./run voice-synth design-voice \
    --instruct "Calm male narrator, mid-40s, warm and unhurried" \
    --ref-text  "The forest was silent except for the distant call of birds." \
    --voice-name forest-narrator
./run voice-synth speak --voice forest-narrator --text "Hello from a designed voice."
```

Outputs land in `/work/voice_synth_<timestamp>/`:

- `take_01.wav` … `take_N.wav` — synthesised takes
- `takes.meta.json` — voice ID, model, language, seeds, per-take timings and optional QA

---

## How it connects to the voice registry

```
voice-split --voice-name <slug>     ───┬─── writes source_clip.wav
voice-clone synth --voice <slug>    ───┼─── writes ref.wav + .pkl prompt
voice-synth design-voice --voice-name ──┼─── writes designed .pkl prompt
                                       │
               /cache/voices/<slug>/ ──┴───
                 voice.json
                 source_clip.wav
                 ref.wav
                 prompts/<hash>.pkl
                       │
voice-synth speak --voice <slug>  (fast: load prompt → generate → write)
```

Legacy prompts in `/cache/voice-clone/prompts/` are still fully supported —
`list-voices` and `speak` work with both.

---

## Sub-commands

### `list-voices`

List all cached voice prompts: named voices first (from `/cache/voices/`), then
legacy prompts (from `/cache/voice-clone/prompts/`).

```bash
./run voice-synth list-voices [--cache DIR]
```

Example output:

```
NAMED VOICES  (2)
  NAME                          LANG          DUR  PROMPTS  STATUS
  -----------------------------------------------------------------------
  david-attenborough            English      9.8s        1  ready
    Once a year, they must all return to the sea to breed…
  forest-narrator               English      6.1s        1  ready
    The forest was silent except for the distant call of birds.

LEGACY PROMPTS  (1)  (no name — tip: run voice-clone synth --voice-name <slug>)
  ID (truncated)                                      LANG          DUR
  -----------------------------------------------------------------------
  3fa8c1b2…_qwen3tts_full                            ?            0.0s

3 voice(s): 2 named, 1 legacy
```

Named voices are shown first. Prompts already registered under a named voice
are deduplicated and do not appear in the legacy section.

| Flag      | Type   | Default  | Description           |
| --------- | ------ | -------- | --------------------- |
| `--cache` | `path` | `/cache` | Persistent cache root |

---

### `speak`

Synthesise text using a cached voice prompt, with optional multi-take generation,
auto-chunking, and whisper-based QA scoring.

```bash
./run voice-synth speak \
    --voice <id> \
    --text "..." \
    [options]
```

**Required**

| Flag         | Description                                                                        |
| ------------ | ---------------------------------------------------------------------------------- |
| `--voice ID` | Named voice slug (e.g. `david-attenborough`), legacy hex prefix, or `/path/to.pkl` |

One of `--text` or `--text-file` must be provided.

**Input text**

| Flag               | Type   | Default | Description                     |
| ------------------ | ------ | ------- | ------------------------------- |
| `--text TEXT`      | `str`  | —       | Text to synthesise              |
| `--text-file FILE` | `path` | —       | Read synthesis text from a file |

**Style & language**

| Flag              | Type     | Default | Description                                                                      |
| ----------------- | -------- | ------- | -------------------------------------------------------------------------------- |
| `--style PRESET`  | `str`    | —       | Style preset from `styles.yaml` (see below)                                      |
| `--language LANG` | `choice` | `Auto`  | Synthesis language; `Auto` detects from text then falls back to the ref language |

**Multiple takes**

| Flag           | Type  | Default | Description                                               |
| -------------- | ----- | ------- | --------------------------------------------------------- |
| `--variants N` | `int` | `1`     | Generate N takes with seeds `base_seed + 0 … N-1`         |
| `--seed INT`   | `int` | —       | Base seed for reproducible generation (random if omitted) |

**Long text**

| Flag      | Description                                                        |
| --------- | ------------------------------------------------------------------ |
| `--chunk` | Auto-split text at sentence boundaries and concatenate output WAVs |

**Generation knobs** (passed to Transformers `model.generate`)

| Flag                         | Type    | Default | Description              |
| ---------------------------- | ------- | ------- | ------------------------ |
| `--temperature FLOAT`        | `float` | —       | Sampling temperature     |
| `--top-p FLOAT`              | `float` | —       | Top-p nucleus sampling   |
| `--repetition-penalty FLOAT` | `float` | —       | Repetition penalty       |
| `--max-new-tokens INT`       | `int`   | —       | Maximum generated tokens |

**QA**

| Flag              | Type  | Default | Description                                                       |
| ----------------- | ----- | ------- | ----------------------------------------------------------------- |
| `--qa`            | flag  | off     | Run faster-whisper on each take; print intelligibility scoreboard |
| `--whisper-model` | `str` | `small` | faster-whisper model for QA transcription                         |

**Infrastructure**

| Flag          | Type   | Default                         | Description                                     |
| ------------- | ------ | ------------------------------- | ----------------------------------------------- |
| `--model ID`  | `str`  | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3-TTS Base model (HF repo ID or local path) |
| `--threads N` | `int`  | `8`                             | CPU threads for torch + whisper                 |
| `--out DIR`   | `path` | `/work`                         | Output directory                                |
| `--cache DIR` | `path` | `/cache`                        | Persistent cache root                           |

---

### `design-voice` _(slow on CPU)_

Create a reusable voice from a natural-language description using the
Qwen3-TTS VoiceDesign → Clone workflow.

```bash
./run voice-synth design-voice \
    --instruct "Calm male narrator, mid-40s, warm and unhurried" \
    --ref-text  "The forest was silent except for the distant call of birds."
```

Steps performed:

1. **Load VoiceDesign model** (1.7 B) — generates a short reference clip in the described style
2. **Cache designed reference** to `/cache/voice-clone/designed_refs/<hash>/`
3. **Build clone prompt** from the designed clip using the Base model
4. **Write** `.pkl` + `.meta.json` to the shared prompts directory

The printed voice ID is immediately usable with `speak`.

| Flag                | Type     | Default                                | Description                                                         |
| ------------------- | -------- | -------------------------------------- | ------------------------------------------------------------------- |
| `--instruct TEXT`   | `str`    | _(required)_                           | Natural-language description of the voice persona, style and timbre |
| `--ref-text TEXT`   | `str`    | _(required)_                           | Short script spoken in the designed voice (1–2 sentences)           |
| `--voice-name SLUG` | `str`    | —                                      | Register the result as a named voice in `/cache/voices/`            |
| `--design-model ID` | `str`    | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign model                                                   |
| `--clone-model ID`  | `str`    | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`        | Base clone model for prompt extraction                              |
| `--language LANG`   | `choice` | `Auto` → `English`                     | Language for design synthesis                                       |
| `--threads N`       | `int`    | `8`                                    | CPU threads                                                         |
| `--out DIR`         | `path`   | `/work`                                | Working directory                                                   |
| `--cache DIR`       | `path`   | `/cache`                               | Persistent cache root                                               |

---

## Style presets

Presets are defined in `styles.yaml` alongside `voice-synth.py`.
Each wraps the synthesis text with an instructional prefix and/or suffix.

| Preset        | Effect                             |
| ------------- | ---------------------------------- |
| `serious_doc` | Calm, steady documentary narration |
| `nature_doc`  | Quiet, reverent, measured delivery |
| `excited`     | High energy, upbeat, smiling       |
| `energetic`   | Lively and expressive              |
| `warm`        | Inviting, intimate, friendly       |
| `sad`         | Heavy-hearted, slow, mournful      |
| `melancholic` | Resigned, quietly sorrowful        |
| `whisper`     | Soft, close-mic whisper            |
| `calm`        | Slow, deliberate, unhurried        |
| `formal`      | Professional, authoritative        |
| `audiobook`   | Clear, expressive but measured     |

Use `--style` with `speak` to apply any preset. Presets compose with
`--prompt-prefix` / `--prompt-suffix` on `voice-clone synth`.

---

## Cache layout

```
/cache/voices/
  <slug>/
    voice.json           # identity, source, ref metadata, prompt index
    source_clip.wav      # raw clip from voice-split (44.1 kHz)
    ref.wav              # processed 24 kHz segment
    prompts/
      <hash>_<model>_full_v1.pkl
      <hash>_<model>_full_v1.meta.json

/cache/voice-clone/
  prompts/
    <id>.pkl             # legacy: pickled voice-clone prompt tensor dict
    <id>.meta.json       # legacy: model, language, transcript, timestamps
  designed_refs/
    <hash>/
      design_ref.wav     # VoiceDesign-synthesised reference clip
      design_meta.json   # instruct, ref_text, language, model, timings
```

`voice-synth speak` reads `.pkl` files only. Named voice prompts and legacy
prompts are both resolved transparently by `--voice`.

---

## CPU performance expectations

Measured on a 16-core Mac Mini (Apple Silicon, `--threads 8`, 0.6 B model):

| Stage           | Duration         | Notes                          |
| --------------- | ---------------- | ------------------------------ |
| Model download  | ~5 min (one-off) | Cached in `/cache/huggingface` |
| Model load      | ~5 min           | Every run — not skippable      |
| Prompt load     | < 1 s            | Pickle load only — fast        |
| Synthesis (RTF) | ~30–35×          | ~2–3 min for 5 s of audio      |

The key advantage over `voice-clone synth` is that VAD, Whisper and prompt
extraction are **never paid again** on re-runs — only model load + generation.

Use `--threads 12` or `--threads 14` for modest speed gains; returns diminish
above physical core count. The 1.7 B clone model is ~3× slower.

---

## Models

| Model                                  | Size  | Used by                                      |
| -------------------------------------- | ----- | -------------------------------------------- |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base`        | 0.6 B | `speak` (default), `design-voice` clone step |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base`        | 1.7 B | `speak` (higher quality, slower)             |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7 B | `design-voice` generation step               |

Model weights are downloaded once and cached at `$XDG_CACHE_HOME/huggingface`
(→ `/cache/huggingface` inside the container).
