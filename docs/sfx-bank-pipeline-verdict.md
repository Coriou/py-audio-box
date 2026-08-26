# Verdict — timing-intelligence × SFX-bank foley POC

Run: `work/sfx-pipeline/20260826-poc` · code: `experiments/sfx-bank-pipeline/`
(commit `2bc817a`) · date: 2026-08-26 · fal spend: **$0.048** (budget $5,
reconciled delta **$0.0000** against the 2026-08 usage CSV)

## TL;DR

The pipeline runs end-to-end and every mechanical piece works. But the
benchmark's central premise — use a hosted V2A model as the *timing oracle* —
is **falsified**: none of the four models comes close to frame-lock accuracy.
The good news buried in that: the fix is already in the repo. Vision-derived
contacts (auto-sfx) are a perfect clock, V2A output demotes gracefully to a
semantic/intensity reference, and the retrieval→placement→mix half of the
architecture is exactly what makes an owned SFX bank valuable. What blocks a
production verdict today is not the algorithm — it is that the template bank
contains one file.

## 1. Timing oracle race (the decisive result)

Greedy matching of detected onsets vs auto-sfx foot contacts, ±250 ms window
(`validation/timing.md`):

| clip | best oracle | matched | median abs err | gate (≤50 ms + full coverage) |
|---|---|---|---:|---|
| walk1 | controlfoley | 2/4 | 68.2 ms | FAIL |
| walk1 | kling-v2a | 2/4 | 94.0 ms | FAIL |
| walk1 | mirelo16 | **0/4** | — | FAIL |
| walk2 | kling-v2a | 3/9 | 57.3 ms | FAIL |
| walk2 | mirelo16 | 2/9 | 94.9 ms | FAIL |

Even at maximum detection sensitivity (3× more candidates accepted), errors
scatter ±50–220 ms with no tight clustering: the models generate plausible
* rhythms*, not synchronized ones. Detection is not the bottleneck — the
generated audio genuinely does not lock to visible contacts.

**Consequence adopted in the pipeline:** event times come from vision where
ground truth exists (walk1/walk2), from the winning V2A's onsets otherwise
(`man`), and surf clips become a bed event filled from the bank. This answers
the brief's open question §5-3: no hosted V2A qualifies as "the" oracle;
Mirelo certainly doesn't.

## 2. Retrieval + placement mechanics (proven, with one caveat)

- Transient alignment is sample-exact: candidate attack placed onto the event
  time, pre-transient head preserved, tail trimmed (8 ms fade) or compressed
  ≤1.35× with ffmpeg `rubberband` **post-transient only** — the attack is
  never stretched. This implements the brief's "never time-stretch a whole
  foley sample" rule literally.
- Tier economics behaved exactly as designed on today's run: **20/20 events
  cost $0** (all fell to the oracle-crop fallback — see caveat below), and the
  Stable-Audio-3 a2a lane was separately verified live end-to-end
  (seed honored, `output_format=wav` honored once explicitly set — schema
  defaults are duration=30/mp3; ledgered request ids in `runs/ledger.jsonl`,
  billed $0.048 = 2 × $0.024).
- CLAP zero-shot labels are sane per clip (gravel footsteps on the walks,
  rain-like surf texture on the aerials), so retrieval decisions will be
  interpretable on the review page.

## 3. The caveat: the template bank has one file

The single provided file (car gear-shift metal clank) is semantically distant
from every event in the five clips. Best combined similarity across all 20
events: **0.26** (walk2) down to **0.06** (ocean-short), all below the 0.30
floor → correct rejection, correct fallback. That proves *discrimination*
(no false matches from wrong sounds) but match *quality* — the actual
production-interesting question — cannot be judged until the bank has real
footstep/impact/water content. The A/B page (`review/index.html`) shows the
mechanism per event either way.

## 4. Is this production-interesting?

- **What's already valuable:** frame-locked placement machinery + tiered
  economics ($0 for most events, generative spend only by exception) +
  auditable ledgers. Re-cutting Mirelo's own audio onto true contact times
  already produces visibly better-synced walk clips than any raw model output.
- **What the POC could not show:** whether bank retrieval picks sounds a
  sound designer endorses. Needs a populated library (even ~30 files across
  footsteps/impacts/water) indexed and judged via `review/index.html`.
- **Recommended next step:** populate `work/sfx-bank/template/`, re-run the
  six commands in `experiments/sfx-bank-pipeline/README.md` (≈10 min CPU),
  judge top-3 retrieval sanity, then decide on the Sound-Palette LoRA path
  with real captions.

## 5. Artifacts

- A/B review: `work/sfx-pipeline/20260826-poc/review/index.html`
- Timing matrix: `…/validation/timing.md` · summary: `…/report/summary.md`
- Reconciliation: `…/report/reconciliation.md` (billed == estimated)
- Fal audit: `…/runs/ledger.jsonl`, `…/records/sa3__*.json`
