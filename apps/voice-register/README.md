# voice-register

One-shot pipeline to register a new voice: obtain audio → Demucs split → clone-prompt build → synthesis test.

Works with a **YouTube URL** or a **local audio file** — both are first-class sources.
Internally chains **voice-split** → **voice-clone synth** so you never have to
copy-paste the two-step boilerplate again. Each step is fully cached — re-runs
skip every stage that has already been computed.

---

## What it does

| Step | App         | What runs                                                                                                                                                                                             |
| ---- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1/2  | voice-split | Download (URL) or load (local file) audio, optionally trim, run Demucs on speech-dense windows, VAD-select clean segments, export clips, register best clip to `/cache/voices/<slug>/source_clip.wav` |
| 2/2  | voice-clone | Normalise ref, VAD + whisper-score best segment, build Qwen3-TTS clone prompt, run a test synthesis to confirm quality                                                                                |

After both steps the voice is immediately usable with `voice-synth speak`.

---

## Quick start

### From a YouTube URL

```bash
./run voice-register \
    --url "https://www.youtube.com/watch?v=XXXX" \
    --voice-name david-attenborough \
    --text "Nature is the greatest artist of all."

# Target a specific segment
./run voice-register \
    --url "https://www.youtube.com/watch?v=XXXX" \
    --start 1:23 --end 5:00 \
    --voice-name david-attenborough \
    --text "Nature is the greatest artist of all."
```

### From a local audio file

Files in `./work/` on your host are available inside the container as `/work/<file>`.

```bash
./run voice-register \
    --audio /work/my-recording.wav \
    --voice-name my-voice \
    --text "Hello, this is a test."

# Trim before processing
./run voice-register \
    --audio /work/interview.mp3 \
    --start 0:45 --end 3:30 \
    --voice-name interviewee \
    --text "Hello, this is a test."
```

Then use the registered voice:

```bash
./run voice-synth speak \
    --voice david-attenborough \
    --text "Welcome to the natural history of our planet."
```

---

## CLI flags

### Audio source (exactly one required)

| Flag           | Description                                         |
| -------------- | --------------------------------------------------- |
| `--url URL`    | YouTube (or any yt-dlp-supported) URL               |
| `--audio PATH` | Path to a local audio file (WAV, MP3, M4A, FLAC, …) |

### Timestamp trim (optional — works with both `--url` and `--audio`)

| Flag                | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `--start TIMESTAMP` | Start time: raw seconds (`90`), `MM:SS` (`1:30`), or `HH:MM:SS` |
| `--end TIMESTAMP`   | End time: same formats as `--start`                             |

### Required

| Flag                | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| `--voice-name SLUG` | Voice slug — lowercase, hyphens only (e.g. `david-attenborough`) |
| `--text TEXT`       | Text for the test synthesis that confirms the voice is working   |

### voice-split options

| Flag                      | Default        | Description                                                               |
| ------------------------- | -------------- | ------------------------------------------------------------------------- |
| `--clips N`               | `3`            | Random preview clips to export alongside the registry clip                |
| `--length SECS`           | `30`           | Target clip length in seconds                                             |
| `--max-scan-seconds SECS` | _(full audio)_ | Limit VAD scan to first N seconds — useful for quick tests on long videos |
| `--cookies FILE`          | —              | Netscape `cookies.txt` for authenticated downloads                        |

### voice-clone options

| Flag                   | Default                         | Description                                                                                           |
| ---------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `--language LANG`      | `Auto`                          | Synthesis language for the test output. `Auto` detects from `--text` then from the ref audio          |
| `--ref-language LANG`  | `Auto`                          | Language of the reference audio; set explicitly if whisper detection is unreliable                    |
| `--tone NAME`          | —                               | Tone label (e.g. `neutral`, `warm`). Stored in the voice registry for `voice-synth speak --tone NAME` |
| `--whisper-model SIZE` | `small`                         | faster-whisper model for ref transcription (`tiny` / `small` / `medium`)                              |
| `--model REPO`         | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3-TTS model                                                                                       |
| `--dtype TYPE`         | `auto`                          | Weight dtype: `auto` picks `float32` on CPU, `float16` on CUDA, `bfloat16` on Apple Silicon           |
| `--seed N`             | —                               | Random seed for reproducible synthesis                                                                |
| `--force-bad-ref`      | off                             | Bypass transcript quality gate (low avg_logprob warning)                                              |

### Common

| Flag           | Default  | Description                                                               |
| -------------- | -------- | ------------------------------------------------------------------------- |
| `--out DIR`    | `/work`  | Output directory for clips + synthesis WAV                                |
| `--cache DIR`  | `/cache` | Persistent cache directory                                                |
| `--force`      | off      | Recompute all stages (ignores cache)                                      |
| `--skip-synth` | off      | Build the clone prompt but skip the test synthesis — voice is still ready |

---

## Output

```
/work/
  clip_01_from_NNs.wav        ← random preview clips (if --clips > 0)
  clip_02_from_NNs.wav
  clip_ref_from_NNs.wav       ← deterministic registry clip
  <voice_name>/
    voice_clone_TIMESTAMP.wav ← test synthesis output
    voice_clone_TIMESTAMP.meta.json

/cache/voices/<slug>/
  voice.json                  ← registry metadata
  source_clip.wav             ← best extracted vocal clip
  ref.wav                     ← processed 24 kHz ref segment
  prompts/
    <hash>_<model>_full_v1.pkl
    <hash>_<model>_full_v1.meta.json
```

---

## Re-runs and caching

All heavy steps write to `/cache/` and skip if already done:

| Cache entry             | Trigger to re-run                                         |
| ----------------------- | --------------------------------------------------------- |
| Downloaded audio        | Delete `cache/downloads/<video_id>.m4a`                   |
| Demucs separated vocals | Delete `cache/separated/htdemucs/<chunk_id>/`             |
| VAD segments            | Delete the matching `cache/*_segments.json`               |
| Normalised ref          | Delete `cache/voice-clone/refs/<hash>/ref_normalized.wav` |
| Best segment selection  | Delete `cache/voice-clone/refs/<hash>/best_segment.json`  |
| Clone prompt            | Delete `cache/voice-clone/prompts/<stem>.pkl`             |

Use `--force` to recompute everything in one go.

---

## Skipping the test synthesis

If you want to build the voice without running a test synthesis (saves ~1 min per voice):

```bash
./run voice-register \
    --url "https://www.youtube.com/watch?v=XXXX" \
    --voice-name my-narrator \
    --text "unused" \
    --skip-synth
```

The voice is still fully registered and ready for `voice-synth speak`.
