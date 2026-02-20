# voice-split

Extract clean, voice-only WAV clips from any YouTube video (or any URL yt-dlp supports).

## What it does

1. Downloads audio (cached per video)
2. Fast VAD scan with Silero to find speech-dense regions (~10s even for 1hr video)
3. Picks the speech-densest windows and runs Demucs on those only
4. Second-pass VAD on the separated vocals
5. Randomly picks N clips from the clean-speech pool and exports them

Everything is cached — re-runs with different `--clips`/`--length` values are nearly instant.

## Usage

From the project root:

```bash
# Quick: 5 clips of 30s each
./run voice-split --url "https://www.youtube.com/watch?v=XXXX" --clips 5 --length 30

# More clips, shorter
./run voice-split --url "https://www.youtube.com/watch?v=XXXX" --clips 20 --length 10

# Reproducible run
./run voice-split --url "..." --clips 10 --length 30 --seed 42

# Scan only the first 10 min (fast test)
./run voice-split --url "..." --clips 5 --length 30 --max-scan-seconds 600

# Custom output dir
./run voice-split --url "..." --out /work/my-project --clips 10
```

Output WAVs land in `./work/` on your host (or whatever `--out` you pass).

## All options

```
--url URL               YouTube (or any yt-dlp) URL              [required]
--out DIR               Output dir for WAV clips                 [default: /work]
--clips N               Number of clips to produce               [default: 10]
--length SECS           Target clip length in seconds            [default: 30]
--window-len SECS       Demucs chunk size (default: 4× --length)
--candidates N          Candidate windows for Demucs (default: 2× --clips)
--cache DIR             Persistent cache dir                     [default: /cache]
--seed N                Random seed for reproducibility
--max-scan-seconds N    Limit VAD scan to first N seconds
--cookies FILE          Netscape cookies.txt for age-gated videos
--voice-name SLUG       Register best clip to the named voice registry [optional]
```

Slugs must be lowercase, alphanumeric with hyphens, e.g. `david-attenborough`.

Note: `--out` is optional. All intermediate work and named clips land under
`/work` by default.

## Output format

`clip_NN_from_XXXs.wav` — 44.1 kHz, 16-bit PCM, mono, with voice EQ applied:

- High-pass at 80 Hz (removes rumble)
- Low-pass at 8 kHz (removes harshness)
- Noise reduction via afftdn
- Light compression

## Named voice registry

Pass `--voice-name <slug>` to automatically register the **best** extracted clip
under `/cache/voices/<slug>/` so the rest of the toolbox can reference it by name.

The slug is lowercase ASCII with hyphens — it becomes the identity of the voice
across every app.

**Best-clip selection** is deterministic: the pool segment with the longest
continuous speech run (highest speech density) is chosen, regardless of `--seed`.
It is written as `clip_ref_from_<start>s.wav` in the output directory and
registered as `source_clip.wav` in the voice registry.

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

`voice-split` writes `source_clip.wav` and a `voice.json` manifest.  
`voice-clone synth` adds `ref.wav` + a `.pkl` prompt.  
`voice-synth list-voices` shows all named voices with their current pipeline status.

---

## Dependencies

All handled by the shared toolbox image:

- `demucs` (htdemucs model, ~80MB, cached on first run)
- `silero-vad` (torch hub, ~5MB, cached on first run)
- `yt-dlp`, `ffmpeg`, `torch`, `torchaudio`
