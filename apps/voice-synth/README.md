# voice-synth

DevX-first synthesis rig for both cached clone prompts and built-in Qwen3 speakers.

Once a voice has been prepared with `voice-clone`, `voice-synth` lets you iterate
fast — no VAD, no Whisper, no prompt extraction on repeated runs.
Just load the cached prompt and generate.

**Clone engine:** [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
**Built-in speaker engine:** [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
**Design engine:** [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (`design-voice` only)
**QA transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (int8 / CPU, optional)

---

## Quick start

```bash
# See all voices you have cached (named + legacy)
./run voice-synth list-voices

# See built-in Qwen3 CustomVoice speakers
./run voice-synth list-speakers

# Probe runtime/model capabilities (agent + CI friendly)
./run voice-synth capabilities --json --strict

# Rollout guard (default: enabled)
QWEN3_ENABLE_CUSTOMVOICE=0 ./run voice-synth capabilities --json

# Register a built-in speaker as a reusable named voice
./run voice-synth register-builtin \
    --voice-name newsroom \
    --speaker Ryan \
    --instruct-default "Calm, clear and warm delivery"

# Same, using an instruction template from lib/styles.yaml
./run voice-synth register-builtin \
    --voice-name newsroom \
    --speaker Ryan \
    --instruct-default-style serious_doc

# Synthesise using a named voice (created by voice-split/voice-clone --voice-name)
./run voice-synth speak \
    --voice david-attenborough \
    --text "Hello, this is my cloned voice."

# Named built-in voices use the same --voice flow
./run voice-synth speak \
    --voice newsroom \
    --text "Top story tonight."

# Also works with a legacy hex ID from list-voices
./run voice-synth speak \
    --voice 3fa8c1 \
    --text "Hello from a legacy prompt."

# Generate 4 takes with different seeds and print an intelligibility scoreboard
./run voice-synth speak \
    --voice david-attenborough \
    --text "The forest was silent except for the distant call of birds." \
    --variants 4 --qa

# Deterministic best-take mode with weighted ranking
./run voice-synth speak \
    --voice david-attenborough \
    --text "The forest was silent except for the distant call of birds." \
    --variants 4 --select-best --seed 7

# Persist preferred generation profile for a named voice
./run voice-synth speak \
    --voice david-attenborough \
    --profile stable --save-profile-default \
    --text "Welcome back."

# Select a tone (speaks with the delivery of whichever ref clip was used for that tone)
./run voice-synth speak \
    --voice david-attenborough \
    --tone sad \
    --text "Hey, Max, I've been captured."

# Use built-in speaker mode directly (no clone prompt needed)
./run voice-synth speak \
    --speaker Ryan \
    --instruct "Warm, calm, slightly upbeat delivery" \
    --text "Tonight's headline: container-native voice tooling at scale."

# Direct speaker mode with an instruction template
./run voice-synth speak \
    --speaker Ryan \
    --instruct-style warm \
    --text "Tonight's headline: container-native voice tooling at scale."

# Design a brand-new voice and register it by name, then use it immediately
./run voice-synth design-voice \
    --instruct "Calm male narrator, mid-40s, warm and unhurried" \
    --ref-text  "The forest was silent except for the distant call of birds." \
    --voice-name forest-narrator
./run voice-synth speak --voice forest-narrator --text "Hello from a designed voice."
```

Outputs land in:

- named voice mode (clone + registered built-in): `/work/<voice-id>/<timestamp>/`
- built-in speaker mode: `/work/customvoice/<speaker-slug>/<timestamp>/`
  (`<speaker-slug>` is a filesystem-safe lowercase slug, e.g. `Ryan` -> `ryan`)

- `take_01.wav` … `take_N.wav` — synthesised takes
- `best.wav` — written when `--select-best` is enabled
- `takes.meta.json` — voice ID, model, language, seeds, per-take timings and optional QA

---

## How it connects to the voice registry

```
voice-split --voice-name <slug>     ───┬─── writes source_clip.wav
voice-clone synth --voice <slug>    ───┼─── writes ref.wav + .pkl prompt
voice-synth design-voice --voice-name ──┼─── writes designed .pkl prompt
voice-synth register-builtin         ───┼─── writes CustomVoice profile
                                       │
               /cache/voices/<slug>/ ──┴───
                 voice.json
                 source_clip.wav
                 ref.wav
                 prompts/<hash>.pkl
                       │
voice-synth speak --voice <slug>  (fast: load prompt → generate → write)
voice-synth speak --speaker <name> (direct CustomVoice mode; no named profile required)
```

Legacy prompts in `/cache/voice-clone/prompts/` are still fully supported —
`list-voices` and `speak` work with both.

---

## Sub-commands

### `list-voices`

List all named voices (clone and built-in) from `/cache/voices/`, then legacy
clone prompts from `/cache/voice-clone/prompts/`.

```bash
./run voice-synth list-voices [--cache DIR] [--json]
```

Example output:

```
NAMED VOICES  (2)
  NAME                          ENGINE          LANG          DUR  PROMPTS  STATUS
  ------------------------------------------------------------------------------------------
  newsroom                      custom_voice    English      0.0s        0  ready
    speaker: Ryan  model: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
    default instruct: Calm, clear and warm delivery
    custom tones: promo
  david-attenborough            clone_prompt    English      9.8s        1  ready
    Once a year, they must all return to the sea to breed…
  forest-narrator               designed_clone  English      6.1s        1  ready
    The forest was silent except for the distant call of birds.

LEGACY PROMPTS  (1)  (no name — tip: run voice-clone synth --voice-name <slug>)
  ID (truncated)                                      LANG          DUR
  -----------------------------------------------------------------------
  3fa8c1b2…_qwen3tts_full                            ?            0.0s

4 voice(s): 3 named, 1 legacy
```

Named voices are shown first. Prompts already registered under a named voice
are deduplicated and do not appear in the legacy section.

| Flag      | Type   | Default  | Description                       |
| --------- | ------ | -------- | --------------------------------- |
| `--cache` | `path` | `/cache` | Persistent cache root             |
| `--json`  | `flag` | off      | Emit machine-readable JSON output |

JSON output shape:

```json
{
  "cache": "/cache",
  "registry_dir": "/cache/voices",
  "legacy_dir": "/cache/voice-clone/prompts",
  "total": 3,
  "named_count": 2,
  "legacy_count": 1,
  "named": [ ... ],
  "legacy": [ ... ]
}
```

---

### `list-speakers`

List built-in speakers supported by a Qwen3 CustomVoice model.
Speaker matching in `speak --speaker ...` is case-insensitive.
Requires `QWEN3_ENABLE_CUSTOMVOICE=1` (default).

```bash
./run voice-synth list-speakers [--json]
./run voice-synth list-speakers --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice [--json]
```

| Flag        | Type   | Default                                      | Description                                  |
| ----------- | ------ | -------------------------------------------- | -------------------------------------------- |
| `--model`   | `str`  | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`       | CustomVoice model repo ID or local path      |
| `--threads` | `int`  | `8`                                          | CPU threads for torch                        |
| `--dtype`   | choice | `auto`                                       | Weight dtype                                 |
| `--cache`   | `path` | `/cache`                                     | Persistent cache root                        |
| `--json`    | `flag` | off                                          | Emit machine-readable JSON output            |

JSON output shape:

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "count": 9,
  "speakers": ["Aiden", "Dylan", "Eric", "Ono_Anna", "Ryan", "Serena", "Sohee", "Uncle_Fu", "Vivian"]
}
```

---

### `capabilities`

Probe runtime/device/model capabilities for CI and autonomous agents.
This command reports device mode, qwen-tts package version, API compatibility
checks, known model families, and CustomVoice speaker availability.

```bash
./run voice-synth capabilities --json
./run voice-synth capabilities --json --strict --require-runtime-speakers
./run voice-synth capabilities --skip-speaker-probe
```

| Flag                        | Type   | Default                                      | Description |
| --------------------------- | ------ | -------------------------------------------- | ----------- |
| `--model`                   | `str`  | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`       | CustomVoice model to probe for speaker availability |
| `--threads`                 | `int`  | `8`                                          | CPU threads used when model probing is enabled |
| `--dtype`                   | choice | `auto`                                       | Weight dtype for runtime speaker probe |
| `--skip-speaker-probe`      | `flag` | off                                          | Skip model load and return static fallback speaker list |
| `--require-runtime-speakers`| `flag` | off                                          | In strict mode, fail if speakers are not obtained from live model probing |
| `--strict`                  | `flag` | off                                          | Exit non-zero when API compatibility/speaker probing checks fail |
| `--json`                    | `flag` | off                                          | Emit machine-readable JSON output |

JSON output includes:

- `feature_flags`: rollout state (`QWEN3_ENABLE_CUSTOMVOICE`) for CustomVoice mode
- `runtime`: python/torch versions, device mode, CUDA info, resolved dtype
- `packages`: installed qwen-tts version
- `available_models` + `default_models`: known Qwen3 families used in this repo
- `api_compatibility`: required method matrix + missing methods (if any)
- `custom_voice`: probed speakers, probe mode, and probe errors

---

### `register-builtin`

Create or update a named built-in CustomVoice profile in `/cache/voices/<slug>/`.
This lets you use `speak --voice <slug>` for built-in speakers, just like cloned voices.
Requires `QWEN3_ENABLE_CUSTOMVOICE=1` (default).

```bash
./run voice-synth register-builtin \
  --voice-name newsroom \
  --speaker Ryan \
  --instruct-default "Calm, clear and warm delivery"

# Add/update a tone-specific instruction preset
./run voice-synth register-builtin \
  --voice-name newsroom \
  --speaker Ryan \
  --tone promo \
  --tone-instruct "Energetic, upbeat promo read"

# Template-based defaults and tone presets
./run voice-synth register-builtin \
  --voice-name newsroom \
  --speaker Ryan \
  --instruct-default-style serious_doc \
  --tone promo \
  --tone-instruct-style energetic
```

| Flag                 | Type     | Default                                | Description |
| -------------------- | -------- | -------------------------------------- | ----------- |
| `--voice-name SLUG`  | `str`    | _(required)_                           | Named voice slug in `/cache/voices/<slug>/` |
| `--speaker NAME`     | `str`    | _(required)_                           | Built-in CustomVoice speaker |
| `--instruct-default` | `str`    | —                                      | Default instruction used when speaking this named voice without `--tone` |
| `--instruct-default-style` | `str` | —                                   | Named instruction template for default delivery (mutually exclusive with `--instruct-default`) |
| `--tone NAME`        | `str`    | —                                      | Optional tone label (e.g. `neutral`, `promo`) |
| `--tone-instruct`    | `str`    | —                                      | Instruction preset for `--tone` |
| `--tone-instruct-style` | `str` | —                                     | Named instruction template for `--tone` (requires `--tone`; mutually exclusive with `--tone-instruct`) |
| `--language-default` | `choice` | `English`                              | Default language when `speak --language Auto` is used |
| `--display-name`     | `str`    | —                                      | Optional display name in `voice.json` |
| `--description`      | `str`    | —                                      | Optional description in `voice.json` |
| `--model`            | `str`    | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice model repo ID or local path |
| `--threads`          | `int`    | `8`                                    | CPU threads for torch |
| `--dtype`            | `choice` | `auto`                                 | Weight dtype |
| `--cache`            | `path`   | `/cache`                               | Persistent cache root |

---

### `speak`

Synthesise text from either:

1. a named voice (`--voice` mode), or
2. a built-in Qwen3 CustomVoice speaker (`--speaker` mode).

Supports optional multi-take generation, deterministic best-take ranking,
auto-chunking, and whisper-based QA scoring.

```bash
./run voice-synth speak \
    --voice <id> | --speaker <name> \
    --text "..." \
    [options]
```

**Required**

| Flag              | Description                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------- |
| `--voice ID`      | Named voice slug (clone/designed/built-in), legacy hex prefix, or `/path/to.pkl` |
| `--speaker NAME`  | Built-in CustomVoice speaker (direct mode) |

One of `--text` or `--text-file` must be provided.

**Input text**

| Flag               | Type   | Default | Description                     |
| ------------------ | ------ | ------- | ------------------------------- |
| `--text TEXT`      | `str`  | —       | Text to synthesise              |
| `--text-file FILE` | `path` | —       | Read synthesis text from a file |

**Tone & language**

| Flag              | Type     | Default | Description |
| ----------------- | -------- | ------- | ----------- |
| `--tone NAME`     | `str`    | —       | `--voice` mode only. Clone voices: selects tone-labeled prompt variant. Named built-in voices: selects stored tone instruction preset. |
| `--instruct TEXT` | `str`    | —       | `--speaker` mode only. Natural-language style/delivery instruction for direct CustomVoice runs. |
| `--instruct-style NAME` | `str` | —      | Named instruction template. Supported in `--speaker` mode and for named CustomVoice voices in `--voice` mode (template override). |
| `--language LANG` | `choice` | `Auto`  | Synthesis language; `Auto` detects from target text then falls back to named-voice default/ref language or English. |

**Multiple takes**

| Flag            | Type  | Default | Description                                               |
| --------------- | ----- | ------- | --------------------------------------------------------- |
| `--variants N`  | `int` | `1`     | Generate N takes with seeds `base_seed + 0 … N-1`         |
| `--seed INT`    | `int` | —       | Base seed for reproducible generation (random if omitted) |
| `--select-best` | flag  | off     | Rank variants and copy best output to `best.wav`         |

**Generation profiles**

| Flag                     | Type     | Default | Description |
| ------------------------ | -------- | ------- | ----------- |
| `--profile NAME`         | `choice` | voice default or `balanced` | Deterministic generation profile: `stable`, `balanced`, `expressive` |
| `--save-profile-default` | flag     | off     | `--voice` mode only. Persist resolved profile + sampling defaults to `voice.json["generation_defaults"]` |

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

| Flag          | Type   | Default | Description |
| ------------- | ------ | ------- | ----------- |
| `--model ID`  | `str`  | mode-dependent | `--voice` clone/designed: Base model. `--voice` built-in: model stored in voice profile. `--speaker`: CustomVoice model. |
| `--threads N` | `int`  | `8`     | CPU threads for torch + whisper |
| `--out DIR`   | `path` | `/work` | Output directory |
| `--cache DIR` | `path` | `/cache`| Persistent cache root |

---

### `design-voice` _(slow on CPU)_

Create a reusable voice from a natural-language description using the
Qwen3-TTS VoiceDesign → Clone workflow.

```bash
./run voice-synth design-voice \
    --instruct "Calm male narrator, mid-40s, warm and unhurried" \
    --ref-text  "The forest was silent except for the distant call of birds."

# Template-based voice design instruction
./run voice-synth design-voice \
    --instruct-style warm \
    --ref-text "The forest was silent except for the distant call of birds."
```

Steps performed:

1. **Load VoiceDesign model** (1.7 B) — generates a short reference clip in the described style
2. **Cache designed reference** to `/cache/voice-clone/designed_refs/<hash>/`
3. **Build clone prompt** from the designed clip using the Base model
4. **Write** `.pkl` + `.meta.json` to the shared prompts directory

The printed voice ID is immediately usable with `speak`.

| Flag                | Type     | Default                                | Description                                                         |
| ------------------- | -------- | -------------------------------------- | ------------------------------------------------------------------- |
| `--instruct TEXT`   | `str`    | one required                           | Natural-language description of the voice persona, style and timbre (mutually exclusive with `--instruct-style`) |
| `--instruct-style NAME` | `str` | one required                           | Named instruction template from `lib/styles.yaml` (mutually exclusive with `--instruct`) |
| `--ref-text TEXT`   | `str`    | _(required)_                           | Short script spoken in the designed voice (1–2 sentences)           |
| `--voice-name SLUG` | `str`    | —                                      | Register the result as a named voice in `/cache/voices/`            |
| `--design-model ID` | `str`    | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign model                                                   |
| `--clone-model ID`  | `str`    | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`        | Base clone model for prompt extraction                              |
| `--language LANG`   | `choice` | `Auto` → `English`                     | Language for design synthesis                                       |
| `--threads N`       | `int`    | `8`                                    | CPU threads                                                         |
| `--out DIR`         | `path`   | `/work`                                | Working directory                                                   |
| `--cache DIR`       | `path`   | `/cache`                               | Persistent cache root                                               |

---

## Tones

For **direct built-in speaker mode** (`--speaker`), style and delivery are controlled
with `--instruct` or `--instruct-style`, for example:

```bash
./run voice-synth speak \
  --speaker Ryan \
  --instruct "Calm, authoritative, evening-news delivery" \
  --text "Good evening. Here are tonight's headlines."

./run voice-synth speak \
  --speaker Ryan \
  --instruct-style serious_doc \
  --text "Good evening. Here are tonight's headlines."
```

For voice cloning, **delivery style comes from the reference audio** used to build
the prompt, not from text instructions. `generate_voice_clone` has no `instruct=`
parameter — prepending `[Sad]` or similar text cues causes the model to _speak
them aloud_, not to adopt that style.

For **named built-in voices** (`--voice <slug>` where `engine=custom_voice`), use
`register-builtin` tone presets:

```bash
# Add a tone instruction preset to an existing named built-in voice
./run voice-synth register-builtin \
  --voice-name newsroom \
  --speaker Ryan \
  --tone breaking \
  --tone-instruct "Urgent, high-energy breaking-news delivery"

# Use that tone later via --voice
./run voice-synth speak --voice newsroom --tone breaking --text "Breaking update..."
```

The tone system maps a short label (e.g. `neutral`, `sad`, `excited`) to a
specific `.pkl` prompt built from a reference clip that already sounds that way:

```bash
# Build and label a tone (use a ref clip that already sounds "sad")
./run voice-clone synth \
    --voice david-attenborough-sad-ref \
    --voice-name david-attenborough \
    --tone sad \
    --text "A sample sentence."

# Select the tone when speaking
./run voice-synth speak --voice david-attenborough --tone sad --text "..."

# See available tones per voice
./run voice-synth list-voices
```

If `--tone` is omitted, the most recently built prompt is used.

Text punctuation and rhythm still influence delivery:

- `...` for natural pauses
- `!` / `?` for emphasis / questions

For **designed voices** (`design-voice`), style is controlled via `--instruct`
or `--instruct-style` (both map to the VoiceDesign model's `instruct=` parameter).

---

## Voice management

### `rename-voice`

Rename a voice slug in the registry (moves the directory and updates `voice.json`).

```bash
./run voice-synth rename-voice <old-slug> <new-slug>
```

### `delete-voice`

Permanently delete a voice and all its files (source clip, ref, prompts).

```bash
./run voice-synth delete-voice <slug> [--yes]
```

Omitting `--yes` shows a confirmation prompt.

### `export-voice`

Pack a voice into a portable zip archive for sharing or backup.

```bash
./run voice-synth export-voice <slug> [--dest /work/my-voice.zip]
```

Default output: `/work/<slug>.zip`. The archive contains the full voice directory
(source clip, ref audio, prompts) and can be imported on any machine running this
toolbox.

### `import-voice`

Unpack a zip created by `export-voice` into the local registry.

```bash
./run voice-synth import-voice --zip /work/my-voice.zip [--force]
```

`--force` overwrites an existing voice with the same slug.

---

## Cache layout

```
/cache/voices/
  <slug>/
    voice.json           # identity + engine profile (+ optional generation defaults)
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

`voice-synth speak` supports three synthesis sources:

- clone/designed mode via `--voice` (named/legacy `.pkl` prompts)
- named built-in mode via `--voice` (`engine=custom_voice` in `voice.json`)
- direct built-in mode via `--speaker`

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
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base`        | 0.6 B | `speak --voice` for clone/designed voices (default), `design-voice` clone step |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base`        | 1.7 B | `speak --voice` for clone/designed voices (higher quality, slower) |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6 B | `register-builtin`, `list-speakers`, `speak --speaker`, and `speak --voice` for named built-ins |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7 B | Same as above, higher quality/slower         |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7 B | `design-voice` generation step               |

Model weights are downloaded once and cached at `$XDG_CACHE_HOME/huggingface`
(→ `/cache/huggingface` inside the container).
