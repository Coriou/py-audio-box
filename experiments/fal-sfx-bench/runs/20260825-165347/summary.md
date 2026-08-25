# fal-sfx-bench report — 20260825-165347

Generated: 2026-08-25T17:20:27.807272+00:00
Run dir: `/Users/ben/Projects/py-audio-box/work/sfx-bench/20260825-165347`
Manifest: `/Users/ben/Projects/py-audio-box/work/sfx-bench/20260825-165347/manifest.json` (30 cases, seed 20260825)

## Run totals

| metric | value |
|---|---:
| cases | 30 |
| succeeded | 30 |
| failed | 0 |
| estimated spend | $1.7582 |
| total wall time | 3709.75 s |
| total output audio | 287.04 s |
| ledger events | 30 completed, 30 submitted |

## Per-model aggregates

| model | endpoint | ok | failed | mean wall | median wall | est $ |
|---|---|---:|---:|---:|---:|---:|
| cassette-t2s | cassetteai/sound-effects-generator | 5 | 0 | 25.91 s | 24.48 s | $0.0500 |
| cassette-v2s | cassetteai/video-sound-effects-generator | 5 | 0 | 257.57 s | 293.00 s | $0.1512 |
| hunyuan | fal-ai/hunyuan-video-foley | 5 | 0 | 174.18 s | 198.96 s | $0.4766 |
| kling-v2a | fal-ai/kling-video/video-to-audio | 5 | 0 | 113.79 s | 76.36 s | $0.1750 |
| mirelo16 | mirelo-ai/sfx1.6/video-to-video | 5 | 0 | 111.73 s | 109.45 s | $0.4766 |
| sonilo11 | sonilo/v1.1/video-to-sound-effects | 5 | 0 | 58.77 s | 46.36 s | $0.4289 |

## Pricing

Price specs source: `models.py`.

| model | unit | per second | flat | published | source | schema checked |
|---|---|---:|---:|---|---|---|
| cassette-t2s | flat_per_request | — | $0.0100 | $0.01 per generation | https://fal.ai/models/cassetteai/sound-effects-generator | 2026-08-25 |
| cassette-v2s | per_second_of_output | $0.0033/s | — | $0.2 per minute of video processed | https://fal.ai/models/cassetteai/video-sound-effects-generator | 2026-08-25 |
| hunyuan | per_second_of_output | $0.0100/s | — | $0.1 per 10 seconds | https://fal.ai/models/fal-ai/hunyuan-video-foley | 2026-08-25 |
| kling-v2a | flat_per_request | — | $0.0350 | $0.035 per video | https://fal.ai/models/fal-ai/kling-video/video-to-audio | 2026-08-25 |
| mirelo16 | per_second_of_output | $0.0100/s | — | $0.01 per second x duration x num_samples | https://fal.ai/models/mirelo-ai/sfx1.6/video-to-video | 2026-08-25 |
| sonilo11 | per_second_of_output | $0.0090/s | — | $0.009 per second of output and per sample | https://fal.ai/models/sonilo/v1.1/video-to-sound-effects | 2026-08-25 |

## Failures

None.

## Case data

Long-format per-case data lives in `summary.csv` (case, model, clip, status, timings, estimated cost).
