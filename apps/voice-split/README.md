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
```

## Output format

`clip_NN_from_XXXs.wav` — 44.1 kHz, 16-bit PCM, mono, with voice EQ applied:

- High-pass at 80 Hz (removes rumble)
- Low-pass at 8 kHz (removes harshness)
- Noise reduction via afftdn
- Light compression

## Dependencies

All handled by the shared toolbox image:

- `demucs` (htdemucs model, ~80MB, cached on first run)
- `silero-vad` (torch hub, ~5MB, cached on first run)
- `yt-dlp`, `ffmpeg`, `torch`, `torchaudio`
