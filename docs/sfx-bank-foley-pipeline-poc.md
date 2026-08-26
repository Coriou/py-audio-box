# POC brief — timing-intelligence × SFX-bank foley pipeline

**Status**: handoff brief for the NEXT orchestrator session. Written 2026-08-26
from the fal-sfx-bench session (7-model hosted benchmark, commits `b388d2f` →
`ded145a`). Nothing of the architecture below is built yet — that is the next
session's job.

## 1. Product idea (decided direction)

Separate **timing intelligence** from **sound identity** instead of waiting for
one magic end-to-end `video + SFX bank → audio` model:

```
VIDEO ──► foley analysis + timing model ──► events + onset timings
                                              │
              ┌───────────────────────────────┘
              ▼
        SFX BANK RETRIEVAL  ◄── audio embeddings
              ▼
        best candidates
              ▼
        AUDIO ADAPTATION (pitch / duration / transient / intensity /
                           reverb / spectral shape)
              ▼
        FRAME-LOCKED MIX
```

Key ideas, verbatim from the product owner:

- **Use Mirelo 1.6 as the *timing oracle*, not the final audio.** Generate a
  track, read off event detection / onsets / durations / intensity / approximate
  semantics, then use that as a *reference temporal envelope* — and throw the
  samples away.
- Retrieve matching sounds from **the user's own SFX bank** (pristine
  professional recordings keep the production's sonic character).
- Adapt locally: align the **transient** frame-exact (untouched), warp the
  **body** moderately, stretch/shorten the **tail** aggressively. Never
  time-stretch a whole foley sample to "fit 430 ms" — that destroys foley.
- Physical heuristics: small object → slightly higher pitch, large → lower;
  fast movement → sharper transient, slow → softer attack.
- **Tiered strategy** (economics: most events cost $0):
  | Situation | Strategy |
  |---|---|
  | Exact library sound fits | use directly |
  | Fits but needs adaptation | local DSP/time/pitch/reverb |
  | Right identity, wrong variation | Stable Audio 3 audio→audio ($0.024/gen) |
  | Nothing suitable in bank | Mirelo generative fallback |
- **"Sound Palette"** (longer-term product): index a studio's library
  (impacts/footsteps/UI/vehicles/cloth/ambience), persistent sonic identity per
  project ("Blade Runner-ish industrial Foley, ACME Library #14"). Same video
  re-sounded with **identical event timing** across palettes. UI concept: a
  "Choose Sound Library" picker.
- POC to prove first: **automated event → library retrieval → frame-accurate
  placement**, with Mirelo as baseline AND timing guide. Reference-audio
  generative models become enhancements, not dependencies.

## 2. Verified building blocks (all prices/behavior verified against fal, 2026-08-25/26)

| Piece | Facts | Source of truth |
|---|---|---|
| Mirelo 1.6 `mirelo-ai/sfx1.6/video-to-video` | $0.01/s × ceil-per-request seconds; `text_prompt`+`duration`(≤60, default 10; >10 s sliding-window)+`num_samples`(force 1)+`seed`; returns mp4 w/ baked soundtrack. Still the newest Mirelo on fal. | `experiments/fal-sfx-bench/models.py` notes |
| Stable Audio 3 SFX a2a `fal-ai/stable-audio-3/small/sfx/audio-to-audio` | audio→audio SFX variation, 459M, **commercial use**. Inputs: `prompt`, `audio_url`, `negative_prompt`, `duration` (**default 30 — set per-event!**), `num_inference_steps` (8, distilled), `guidance_scale`, `seed`, `output_format` (**default mp3 — set wav**), `bitrate`. Price **$0.024/generation** (owner-cited; confirm on first call). | fal /api page (schema read 2026-08-26) |
| Stable Audio 3 trainer `fal-ai/stable-audio-3-trainer` | LoRA fine-tune, **commercial use**. `audio_data_url` = **zip of audio files with sibling `.txt` captions**; `model` enum incl. **`small-sfx-base`**; steps default 1000; adapters (dora-rows default); $3/1,000 steps (owner-cited). This is the "Sound Palette" training path. | fal /api page (schema read 2026-08-26) |
| ControlFoley fal lane `fal-ai/controlfoley` | UNLISTED, $0.002/s (floor-rounded, ~8 s cap). `{video_url, prompt, seed}` — seed honored, prompt consumed. **`audio_url` accepted but content IGNORED** (footsteps vs ocean refs → byte-identical output) ⇒ **no AC-V2A via fal**. Vendor attribution unverified. | `models.py` notes; probes in session transcript |
| ControlFoley upstream | Weights released (HF `YJX-Xiaomi/ControlFoley`, 16.7 GB), code Apache-2.0, **weights CC-BY-NC-4.0**. **AC-V2A = video + 2–4 s reference audio → timbre-guided synced foley** (the feature the owner wants most) — local/vast or HF Space only, ~17–22 GB VRAM fp32, torch 2.5.1 pinned env. | session scout (transcript) |
| ThinkSound `fal-ai/thinksound` | $0.001/compute-second, video+prompt. Never benchmarked — cheap candidate. | fal catalog scout |
| AC-Foley / MultiFoley | Reference-audio-guided V2A candidates (arXiv 2603.15597 / IFICL MultiFoley). License + weights availability **not yet scouted**. | owner's brief |
| Billing reality check | Aggregated per-endpoint CSV exports (no request ids); Sonilo ceils seconds, cassette-v2s ceils whole minutes per request ($0.20 for a 10 s clip!), Mirelo ceils, Hunyuan/SA3-family fractional/flat. Always reconcile. | `runs/20260825-165347/reconciliation.md` |

## 3. Existing assets to reuse (do NOT rebuild)

- **Benchmark framework**: `experiments/fal-sfx-bench/` — model registry
  (`models.py`), manifest generator, resumable runner with fsynced request
  ledger, review builder (two-pass loudnorm −16 LUFS muxed on original frames),
  report + billing reconciliation. README documents the 7-model run.
- **Run artifacts**: `work/sfx-bench/20260825-165347/` — 5 standardized silent
  h264 clips (`inputs/`, same bytes fed to every model), 35 case records with
  request_ids, raw outputs, `runs/<id>/review/index.html` (side-by-side
  players). Mirelo raw outputs for the 5 clips are the **timing-oracle
  starting material**.
- **Timing ground truth**: `experiments/auto-sfx/` already has detector/
  association/metrics tooling and `work/auto-sfx/runs/real-walk*/events.json`
  footstep-contact events + `work/auto-sfx/fixtures/*_contacts.json`. Reuse for
  measuring oracle timing fidelity and final placement error — do not invent a
  new metrics stack.
- **Clips**: `work/auto-sfx/clips/` — walk1 (35.6 s hike), walk2.webm (45 s
  portrait, barefoot grass; **sibling walk2.mp4 is corrupt, never use**), man
  (9.4 s), 3571264…mp4 (33 s aerial ocean, has source audio). Standardized
  9.4–10 s versions in the run's `inputs/`.
- **Runtime**: `uv venv /tmp/pab-sfx-bench --python 3.12 && uv pip install
  --python /tmp/pab-sfx-bench/bin/python fal-client` (recreate if /tmp wiped).
  `FAL_KEY` lives in repo `.env` (key id `bfe2a3a9…`). ffmpeg/ffprobe on host.
- **SFX bank**: owner will provide template SFX for testing (location TBC —
  ask; suggest `work/sfx-bank/<palette>/` with category subdirs). NOTE: the
  SA3 trainer needs sibling `.txt` captions per wav — if the bank lacks
  captions, plan an auto-captioning step (CLAP zero-shot or similar) before
  palette training.

## 4. POC scope for the next session

Prove the loop end-to-end on the existing 5 clips + owner's template bank:

1. **Oracle**: run Mirelo on each clip (or reuse existing raw outputs);
   extract event list (onset s, duration, intensity, coarse semantics) from the
   generated audio via onset detection + embedding clustering.
2. **Oracle validation**: compare extracted footstep onsets against
   `events.json` contact times (auto-sfx tooling) → report median/percentile
   timing error. Gate: median error small enough to be perceptually
   frame-locked (propose ≤50 ms for footsteps; owner sets final bar).
3. **Retrieval**: index the bank (embeddings per file); match each detected
   event (semantic label + Mirelo snippet embedding) → top-k candidates with
   scores. Owner judges retrieval sanity on the review page.
4. **Placement + adaptation**: transient-aligned placement (frame-lock),
   tiered strategy (direct → DSP → SA3 a2a → Mirelo fallback), render final
   mixes.
5. **Deliverable**: review/index.html-style A/B per clip — original Mirelo vs
   bank-based render (+ per-event retrieval choices + timing-error table) —
   plus cost per clip (target ≈ $0.01 oracle + ~$0 retrieval/DSP; SA3 only
   where needed) and a written verdict on whether the architecture holds.

Budget guidance: ≤ $5 total fal spend for the POC (Mirelo re-runs ≈ $0.05,
SA3 a2a $0.024/variation — use sparingly). Reuse the ledger/reconcile
discipline from fal-sfx-bench.

## 5. Open questions for the owner

- Where is the template SFX bank / how many files / categories? Any captions?
- Timing-error tolerance per event class (footsteps vs doors vs ambience).
- Does the oracle need to be Mirelo specifically, or "any good V2A" (Kling/
  Sonilo raw outputs from the same run are also available as oracle candidates
  at zero extra cost — they're already downloaded)?
- Palette training (SA3 trainer) in- or out-of-scope for the first POC?

## 6. Paste-ready prompt for the next orchestrator

```text
Read docs/sfx-bank-foley-pipeline-poc.md first — it is the authoritative
brief for this session (architecture, verified endpoints/prices, reusable
assets, POC scope, open questions). Also read experiments/fal-sfx-bench/
README.md for the benchmark framework conventions you must reuse (registry
pattern, ledger discipline, review-set builder, reconciliation).

Goal: prove the timing-intelligence × SFX-bank foley pipeline end-to-end on
the existing 5-clip set: Mirelo (or existing raw outputs) as timing oracle →
event extraction → retrieval from my SFX bank (I will tell you where it is;
ask if the brief's location is stale) → transient-aligned placement with the
tiered direct/DSP/Stable-Audio-3-a2a/Mirelo-fallback strategy → frame-locked
mix per clip.

Non-goals: no new benchmark of hosted models; no palette LoRA training unless
I approve it mid-session; no product UI.

Deliverables: (1) per-clip A/B review page (Mirelo original vs bank render,
same loudnorm/mux conventions as fal-sfx-bench), (2) timing-error metrics vs
the auto-sfx contact events, (3) per-clip cost accounting via the ledger
pattern, (4) honest verdict on whether retrieval+placement quality is
production-interesting, (5) committed code under experiments/ following repo
conventions. Keep total fal spend under $5 and reconcile it against my Fal
CSV export before finishing.
```

## 7. Sources

- fal model/API pages read 2026-08-25/26: Mirelo 1.6, Kling video-to-audio,
  Sonilo v1.1 SFX, HunyuanVideo-Foley, CassetteAI ×2, controlfoley (catalog +
  live probes), stable-audio-3/small/sfx/audio-to-audio, stable-audio-3-trainer.
- Session transcript artifacts: ControlFoley repo/paper deep-dive, hosted-
  availability scout, billing CSV reconciliations (v1, v2).
- Owner's architecture brief (this document §1) — 2026-08-26.
