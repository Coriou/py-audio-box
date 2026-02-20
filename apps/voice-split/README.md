# voice-split

Extract clean, voice-only WAV clips from a YouTube video **or a local audio file**.

## What it does

1. Obtains audio (download from URL, or use a local file directly)
2. Optionally trims to a timestamp range (`--start` / `--end`)
3. Fast VAD scan with Silero to find speech-dense regions (~10s even for a 1hr video)
4. Picks the speech-densest windows and runs Demucs on those only
5. Second-pass VAD on the separated vocals
6. Randomly picks N clips from the clean-speech pool and exports them

Everything is cached — re-runs with different `--clips`/`--length` values are nearly instant.

---

## Usage — YouTube source

```bash
# Basic: 5 clips of 30s each from a YouTube video
./run voice-split --url "https://www.youtube.com/watch?v=XXXX" --clips 5 --length 30

# Target a specific segment of a long video (skip the rest)
./run voice-split --url "https://www.youtube.com/watch?v=XXXX" \
    --start 1:23 --end 5:00 \
    --clips 3

# Timestamps also accept raw seconds or HH:MM:SS
./run voice-split --url "https://www.youtube.com/watch?v=XXXX" --start 83 --end 300

# Register best clip as a named voice
./run voice-split \
    --url "https://www.youtube.com/watch?v=XXXX" \
    --voice-name david-attenborough

# Authenticated download (age-gated videos)
./run voice-split --url "..." --cookies /work/cookies.txt --clips 5
```

## Usage — local audio file

Mount your file into the container (via the `./run` script or `docker compose run -v`):

```bash
# Full file
./run voice-split --audio /work/my-recording.wav --clips 5 --length 30

# Trim to a segment before processing
./run voice-split --audio /work/interview.mp3 \
    --start 0:45 --end 3:30 \
    --clips 3

# Register as a named voice from a local file
./run voice-split \
    --audio /work/my-recording.wav \
    --voice-name my-voice
```

The file path must be accessible inside the container. The default `./run` script mounts `./work` as `/work`, so files in your `work/` directory are available as `/work/<file>`.

## All options

```
# Source (exactly one required)
--url URL               YouTube (or any yt-dlp-supported) URL
--audio PATH            Path to a local audio file (WAV, MP3, M4A, FLAC, …)

# Timestamp trim (optional, works with both --url and --audio)
--start TIMESTAMP       Trim start: seconds (90 / 1:30.5), MM:SS, or HH:MM:SS
--end TIMESTAMP         Trim end:   same formats

# Clip extraction
--clips N               Number of clips to produce               [default: 10]
--length SECS           Target clip length in seconds            [default: 30]
--window-len SECS       Demucs chunk size (default: 4× --length)
--candidates N          Candidate windows for Demucs (default: 2× --clips)
--seed N                Random seed for reproducibility

# I/O
--out DIR               Output dir for WAV clips                 [default: /work]
--cache DIR             Persistent cache dir                     [default: /cache]
--max-scan-seconds N    Limit VAD scan to first N seconds (quick tests)

# YouTube-only
--cookies FILE          Netscape cookies.txt for age-gated videos

# Voice registry
--voice-name SLUG       Register best clip to the named voice registry
```

Timestamp examples:

| Input     | Parsed as         |
| --------- | ----------------- |
| `90`      | 90 seconds        |
| `1:30`    | 1 min 30 s = 90 s |
| `1:30.5`  | 90.5 s            |
| `1:02:30` | 1 hr 2 min 30 s   |

Slugs must be lowercase, alphanumeric with hyphens, e.g. `david-attenborough`.

---

## Output format

`clip_NN_from_XXXs.wav` — 44.1 kHz, 16-bit PCM, mono, with voice EQ applied:

- High-pass at 80 Hz (removes rumble)
- Low-pass at 8 kHz (removes harshness)
- Noise reduction via afftdn
- Light compression

---

## Named voice registry

Pass `--voice-name <slug>` to automatically register the **best** extracted clip
under `/cache/voices/<slug>/` so the rest of the toolbox can reference it by name.

**Best-clip selection** is deterministic: the pool segment with the longest
continuous speech run is chosen, regardless of `--seed`.

### Full pipeline example (YouTube)

```bash
# 1. Extract clips and register the voice
./run voice-split \
    --url "https://www.youtube.com/watch?v=XXXX" \
    --voice-name david-attenborough

# 2. Process ref audio + build clone prompt (done once, cached)
./run voice-clone synth \
    --voice david-attenborough \
    --text "Nature is the greatest artist."

# 3. Fast iteration — no re-processing on subsequent runs
./run voice-synth speak \
    --voice david-attenborough \
    --text "Welcome to the natural history of the Earth."
```

### Full pipeline example (local file)

```bash
# 1. Extract clips from a local recording and register
./run voice-split \
    --audio /work/interview.wav \
    --start 0:15 --end 2:00 \
    --voice-name interviewee

# 2. Build clone prompt
./run voice-clone synth \
    --voice interviewee \
    --text "Hello, this is a test."

# 3. Synthesise
./run voice-synth speak \
    --voice interviewee \
    --text "Your generated text here."
```

Or use `voice-register` to run steps 1–3 in a single command (see `apps/voice-register/`).

---

## Dependencies

All handled by the shared toolbox image:

- `demucs` (htdemucs model, ~80MB, cached on first run)
- `silero-vad` (torch hub, ~5MB, cached on first run)
- `yt-dlp`, `ffmpeg`, `torch`, `torchaudio`
