# fal-sfx-bench — multi-model SFX/Foley benchmark

Hosted SFX/Foley models on fal.ai, benchmarked on an identical clip set with
auditable per-request cost/latency. Built as the successor to the single-model
`experiments/fal-hunyuan-foley-eval/` sidecar: same fal queue flow, extended to
a model registry, a cost ledger, a normalized review set, and billing
reconciliation against the Fal dashboard CSV export.

## Models (schemas verified 2026-08-25)

| key | endpoint | input | output | published price |
|---|---|---|---|---|
| `mirelo16` | `mirelo-ai/sfx1.6/video-to-video` | video + `text_prompt`, duration, seed | mp4 with baked soundtrack | $0.01 / s × samples (forced 1) |
| `kling-v2a` | `fal-ai/kling-video/video-to-audio` | video + `sound_effect_prompt` (≤200 ch) | dubbed mp4 + mp3 | $0.035 flat / video |
| `sonilo11` | `sonilo/v1.1/video-to-sound-effects` | video + `prompt` | audio files (wav forced) | $0.009 / s of output |
| `hunyuan` | `fal-ai/hunyuan-video-foley` | video + `text_prompt` | muxed mp4 | $0.1 / 10 s |
| `cassette-v2s` | `cassetteai/video-sound-effects-generator` | video only (no prompt — auto-SFX) | mp4 with SFX track | $0.2 / min of video |
| `cassette-t2s` | `cassetteai/sound-effects-generator` | text + duration (no video) | wav | $0.01 flat / generation |
| `controlfoley` | `fal-ai/controlfoley` | video + `prompt`, seed | dubbed mp4 + wav, ~8 s cap | $0.002 / s generated audio |

Quirks encoded in `models.py` (see each entry's `notes`): Kling's schema default
`background_music_prompt="intense car race"` is overridden to `""`; Mirelo's
`num_samples` default 2 is forced to 1; prompt key differs per model
(`text_prompt` vs `prompt` vs `sound_effect_prompt`); Kling accepts only
.mp4/.mov, 3–20 s, ≤100 MB. `controlfoley` is UNLISTED on fal (catalog API
public, playground 404): schema verified live — seed honored, prompt consumed,
output truncated to ~8 s; Xiaomi attribution unconfirmed by fal. `audio_url`
is accepted but its content is ignored (verified: two different refs →
byte-identical output at the same seed) — no AC-V2A via fal; upstream
ControlFoley's reference-audio conditioning (AC-V2A) requires the local
weights or the official HF Space.

## Methodology

- **Identical bytes per case.** `prepare_inputs.py` builds five silent h264
  clips (480p proxies; ocean-long at 720p) — every model receives the same
  file. Sources: `walk1` (hike, 10.0–19.4 s), `walk2` (barefoot grass,
  18.0–27.4 s of the portrait .webm — the sibling .mp4 is corrupt, never use
  it), `man` (paved overlook, 9.4 s), `ocean-long` (aerial waves, 10–20 s of
  the Pexels clip, source audio stripped), `ocean-short` (prior eval's
  `ocean-9.4s.mp4`, copied verbatim).
- **Canonical prompts.** One prompt string per clip, identical across models
  (per-model key mapping happens in the registry). Footstep prompts demand
  synchronized footsteps at every visible contact and explicitly exclude
  speech/music.
- **Fixed seed** 20260825 where the schema supports one.
- **Review fairness.** `build_review.py` extracts each result's audio, applies
  two-pass `loudnorm I=-16:TP=-1.5:LRA=11` (48 kHz stereo AAC), and muxes it
  onto the ORIGINAL input frames (`-c:v copy`) — so A/B is identical frames +
  equal loudness. Raw model outputs are kept untouched under `raw/`.
- **Cost audit.** Every submission appends `request_id` to `ledger.jsonl`
  (fsynced). `estimated_cost_usd` uses published unit prices; `reconcile.py`
  joins the ledger against the Fal dashboard CSV export for billed truth.

## Run it

```bash
# one-time
uv venv /tmp/pab-sfx-bench --python 3.12
uv pip install --python /tmp/pab-sfx-bench/bin/python fal-client
echo 'FAL_KEY=...' >> .env   # or export FAL_KEY

RUN_ID=$(date -u +%Y%m%d-%H%M%S)
PY=/tmp/pab-sfx-bench/bin/python

$PY experiments/fal-sfx-bench/prepare_inputs.py --out-dir work/sfx-bench/$RUN_ID/inputs
$PY experiments/fal-sfx-bench/make_manifest.py  --run-dir work/sfx-bench/$RUN_ID
$PY experiments/fal-sfx-bench/run_benchmark.py  --run-dir work/sfx-bench/$RUN_ID --max-workers 4
$PY experiments/fal-sfx-bench/build_review.py   --run-dir work/sfx-bench/$RUN_ID
$PY experiments/fal-sfx-bench/report.py         --run-dir work/sfx-bench/$RUN_ID
# after exporting the Fal billing CSV:
$PY experiments/fal-sfx-bench/reconcile.py      --run-dir work/sfx-bench/$RUN_ID --csv fal-usage.csv
```

Useful flags: `--models mirelo16,kling-v2a`, `--cases walk1` (substring),
`--force` (ignore ledger/review reuse). The runner is resumable: completed
request_ids in the ledger are skipped on re-run.

## Run directory layout

```
work/sfx-bench/<run_id>/
  manifest.json               35 cases (7 models × 5 clips), prompts + args + est cost
  inputs/                     standardized silent clips + inputs-index.json (argv audit)
  runs/<run_id>/
    ledger.jsonl              append-only request audit (request_id per submission)
    records/<case>.json       full per-case record (timings, statuses, probes, sha256s)
    raw/<case>/               untouched model outputs
    review/<case>.mp4         loudness-normalized, muxed on original frames
    review/index.html         side-by-side players per clip + stats
    report/summary.md|.csv    matrix + per-model aggregates + pricing table
    report/pricing.json       machine-readable price specs (source: models.py)
    report/reconciliation.md  billed vs estimated (needs Fal CSV)
```

Open `review/index.html` in a browser to judge quality; `report/summary.md`
for latency/cost; `report/reconciliation.md` once the billing CSV exists.

## First run — 20260825-165347 (cost audit vs real Fal billing)

30/30 cases succeeded in the first sweep; `controlfoley` was added later
(5 cases, seed-pinned, ~$0.08 est) for 35 total. Real billed (credential
`bfe2a3a9…`): **$2.6615** vs
$1.7582 estimated. The export is an aggregated daily rollup per endpoint +
credential — no request ids — so reconciliation joins per endpoint and
filters rows by credential. A foreign-credential hunyuan row ($0.6586,
`92cc21dc…`) is excluded automatically.

| model | billed | estimated | delta | billing behavior (verified) |
|---|---:|---:|---:|---|
| cassette-v2s | $1.0000 | $0.1588 | +$0.8412 | video time **ceiled to whole minutes per request** — a 10 s clip costs $0.20 |
| sonilo11 | $0.4500 | $0.4289 | +$0.0211 | output seconds ceiled per request (9.4 s → 10 s) |
| mirelo16 | $0.5100 | $0.5100 | 0 | billed seconds = ceil per request (matches our `duration=ceil` arg) |
| hunyuan | $0.4765 | $0.4766 | ~0 | fractional units (seconds/10), no rounding |
| kling-v2a | $0.1750 | $0.1750 | 0 | flat per video, exact |
| cassette-t2s | $0.0500 | $0.0500 | 0 | flat per generation, exact |

Takeaway: per-second list prices mislead at short clip lengths. For ≤10 s
clips the true cost order is cassette-t2s ($0.01, text-only) < Kling
($0.035) < cassette-v2s ($0.20!) < sonilo ($0.045–0.09) ≤ mirelo/hunyuan
($0.10). Rounding rules are recorded per model in `models.py` notes.

## Adding a model

1. Verify the endpoint on fal.ai (model page + `…/openapi.json?endpoint_id=<id>`).
2. Add one `ModelSpec` entry to `MODELS` in `models.py`: endpoint id, argument
   builder (mind key names and schema defaults that must be overridden),
   output extractor, `PriceSpec` with the exact published text + source URL,
   `schema_checked_at`.
3. Re-run `make_manifest.py` — new cases appear automatically.

## Prior art

`experiments/fal-hunyuan-foley-eval/` — single-model latency harness
(2026-08-25) that established the trim windows, prompts, and record shape
reused here.
