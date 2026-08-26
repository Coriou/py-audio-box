# sfx-bank-pipeline — timing-intelligence × SFX-bank foley POC

End-to-end proof of the architecture in
`docs/sfx-bank-foley-pipeline-poc.md`: separate **timing intelligence** from
**sound identity** by detecting events on a video, retrieving matches from a
studio SFX bank, adapting them locally (transient-aligned), and mixing a
frame-locked soundtrack. Built as the successor to `experiments/fal-sfx-bench/`
and reuses its conventions: standalone scripts, fsynced request ledgers,
two-pass −16 LUFS review muxes, aggregate billing reconciliation.

## Findings first (2026-08-26 run)

1. **No hosted V2A is a frame-lock timing oracle.** Racing all four benchmarked
   models' raw outputs against auto-sfx foot contacts (`validation/timing.md`):
   best case ControlFoley on walk1 = 50% contact coverage at 68 ms median error;
   Kling on walk2 = 33% at 57 ms. Mirelo 1.6 — the brief's proposed oracle —
   matched **0/4** walk1 contacts within ±250 ms. Gate (≤50 ms median AND full
   coverage) **FAILS**. Generated tracks sound plausible but their onsets are
   tens-to-hundreds of ms away from visible contacts.
2. Therefore the pipeline's event clock defaults to **vision** where ground
   truth exists (auto-sfx contacts) and uses V2A output only as a semantic /
   intensity reference plus event source for clips without contacts. This
   answers the brief's open question §5-3: the oracle need not be Mirelo —
   hosted V2A should not be the timing layer at all.
3. The tiered strategy works mechanically end-to-end (direct / DSP /
   Stable-Audio-3 a2a / $0 oracle-crop fallback); see the tier tables in
   `report/summary.md` and the A/B page `review/index.html`.

## Pipeline

```
inputs (bench clips) ──► oracle_events.py   V2A raws → SuperFlux onset/intensity events
                          │
auto-sfx contacts ──► validate_oracle.py    race V2A oracles vs contacts → winner + gate
                          │
SFX bank ─────────► index_bank.py           CLAP embeddings, onsets, zero-shot labels
                          │
                          ▼
                      retrieve.py            per-event snippet embedding → top-k candidates
                          │                  tier: direct | dsp | sa3 | mirelo-fallback
                          ▼
                      render_mix.py          transient-aligned placement, rubberband tail DSP,
                          │                  ledgered SA3 calls, loudnorm −16 LUFS mux
                          ▼
                      build_ab_page.py       A/B players + per-event decision tables
                      report.py              timing/tier/cost summary
                      reconcile.py           ledger vs Fal CSV export
```

Stage 2 result drives stage 4: clips with contact ground truth place events at
**contact times** (vision clock); footstep-free clips fall back to the winning
V2A model's onsets; surf clips become a single bed event filled from a bank
ambience loop.

### Placement rules (the core of the POC)

- The candidate's **first transient is aligned sample-exactly** onto the event
  time; the pre-transient head keeps its natural length.
- The attack is never stretched. Tail handling: trim with an 8 ms fade, or
  compress the post-transient region up to 1.35× via ffmpeg `rubberband`
  (pitch preserved) when the next event needs room.
- Intensity matched by RMS gain into the event's measured dBFS (clamped ±12 dB).
- Tiers, decided per event in `retrieve.py`:
  | tier | condition | cost |
  |---|---|---|
  | `direct` | fits gap & intensity within 3 dB | $0 |
  | `dsp` | gain ≤12 dB, tail trim or ≤1.35× tempo | $0 |
  | `sa3` | identity ≥0.45 but timing impossible locally; ≤4 intents/run | $0.024/gen |
  | `mirelo-fallback` | best score <0.30 floor → crop the V2A track itself | $0 |

Retrieval score: `0.6·cos(snippet, file) + 0.4·cos("the sound of <label>", file)`
with CLAP (`laion/clap-htsat-fused`, CPU). Labels are zero-shot over a fixed
vocabulary (`--labels-file` overrides).

## Run it

```bash
PY=/tmp/pab-sfx-bench/bin/python        # numpy/scipy/torch-CPU/transformers<5/fal-client
RUN=work/sfx-pipeline/$(date -u +%Y%m%d-%H%M%S)

# 1+2. oracle events + validation race (no fal spend)
$PY experiments/sfx-bank-pipeline/oracle_events.py   --out-dir $RUN
$PY experiments/sfx-bank-pipeline/validate_oracle.py --run-dir $RUN

# 3–5. bank → plans → renders → page (drop files in work/sfx-bank/template first)
$PY experiments/sfx-bank-pipeline/index_bank.py   --run-dir $RUN --force
$PY experiments/sfx-bank-pipeline/retrieve.py     --run-dir $RUN          # --no-sa3 to disable fal tier
$PY experiments/sfx-bank-pipeline/render_mix.py   --run-dir $RUN
$PY experiments/sfx-bank-pipeline/build_ab_page.py --run-dir $RUN
$PY experiments/sfx-bank-pipeline/report.py       --run-dir $RUN

# after exporting the Fal usage CSV:
$PY experiments/sfx-bank-pipeline/reconcile.py --run-dir $RUN --csv fal-usage.csv
```

Every stage is resumable (`--force` re-computes); plans/records are JSON with
sha256 audit fields; every fal submission appends a fsynced `ledger.jsonl` line.

## Run directory layout

```
work/sfx-pipeline/<run_id>/
  oracle/<clip>.<model>.wav|.events.json   per-model onset/intensity analysis
  validation/timing.md|.json|per-event.json  oracle race + gate verdict
  bank/index.json|embeddings.npz|audio/      indexed SFX bank
  plan/<clip>.placement.json                 per-event candidates, tier decision, ops
  mix/<clip>.bed.wav                         placed timeline (pre-loudnorm)
  records/, runs/ledger.jsonl, raw/sa3__*/    fal audit trail
  review/<clip>.mirelo16.mp4 | <clip>.bank.mp4 | index.html   A/B set
  report/summary.md | reconciliation.md
```

## Quirks discovered (verify before reuse)

- `fal-ai/stable-audio-3/small/sfx/audio-to-audio`: schema defaults
  `duration=30` and `output_format=mp3`. Both must be set explicitly per call;
  `wav` IS honored when requested (verified 2026-08-26,
  `records/sa3__*.json`), seed is honored.
- torch on macOS x86_64 ends at 2.2.2 ⇒ transformers must be pinned `<5`
  (5.x demands torch≥2.5) and numpy `<2`; scipy `1.13.x` completes the pin set.
- librosa/numba have no macOS x86_64 wheels anymore — the SuperFlux detector
  here is a self-contained numpy/scipy implementation (`oracle_events.py`),
  geometry documented in each events.json `detector` block.
- Mirelo mixes footsteps *into* an ambience bed: flux onsets rise only ~1–6 dB
  over local texture, so pure spectral-flux peak picking over-detects ~10×.
  The energy-jump gate (`ENERGY_JUMP_DB`) does the perceptual work.

## Prior art

- `experiments/fal-sfx-bench/` — 7-model benchmark whose raw outputs are the
  oracle inputs here; loudnorm/mux/ledger/reconcile conventions reused.
- `experiments/auto-sfx/` — pose-based foot-contact detector supplying vision
  timing ground truth (`runs/real-walk*/events.json`).

## Phase-2 additions (2026-08-26)

- **Event model v2** (`events_v2.py`): every plan/oracle event now carries an
  additive `temporal_type` (`impulse|continuous|repetitive|bed|hybrid`, `None`
  for unclassified detector output), optional `subclass`, `sequence_id`,
  `source`/`physical`/`visual` placeholders and a `schema_event_v2` marker.
  Type inference is a documented keyword heuristic; refine later, not here.
- **Repetitive diversity**: inside one clip, events classified as a repetitive
  sequence share a `sequence_id`; re-picking the same bank entry costs
  ×0.85 per prior use in ranking (CTO §4C). Tier logic unchanged.
- **Per-event stems**: `render_mix.py --run-dir …` writes
  `stems/<clip>/evt_<event_id>.wav` (post-gain/post-fade placed audio) and
  `stems/<clip>/ambience.wav` (bed contribution incl. its gain) at 48 kHz,
  alongside the existing mix. Peak-guard stays on the summed bed only.
- **Loudness policy** (unchanged but now explicit): per-event RMS→intensity
  gain is the local match; global −16 LUFS happens ONLY on the review m4a/mp4
  encode path; internal renders are lossless 48 kHz WAV end-to-end.
- **registry.yaml**: provider/license rows as data — quality status is kept
  separate from commercial status. Consult it before wiring any new provider.
