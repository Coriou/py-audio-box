# Timing-Intelligence × SFX-Bank Foley Pipeline — POC Session Report

**Date:** 2026-08-26 · **Repo:** `Coriou/py-audio-box` (local only — nothing pushed) · **Brief:** `docs/sfx-bank-foley-pipeline-poc.md` · **Verdict doc:** `docs/sfx-bank-pipeline-verdict.md`
**Author:** orchestration session · **Code:** `experiments/sfx-bank-pipeline/` (commits `2bc817a`, `a30b415`)

---

## 1. TL;DR

We built and ran the first end-to-end proof of the product idea in the brief:
**video → sound events with timestamps → retrieval from an owned SFX bank →
transient-aligned placement → frame-locked mix**, with hosted V2A models
(Mirelo 1.6 et al.) tested as the proposed *timing oracle* and the owner's
template bank as sound source. Total external spend **$0.048** (budget $5,
reconciled to the Fal CSV with **$0.0000 delta**).

**Bottom line:** the central premise — use a hosted V2A soundtrack as the
timing oracle — is **falsified**: across all four benchmarked models, detected
onsets sit tens-to-hundreds of ms away from visible foot contacts (best case:
50% coverage at 68 ms median error vs the ≤50 ms full-coverage gate). The fix
is already in-house: **vision-derived contact events are a perfect clock**, so
the architecture survives with layers re-stacked (vision = timing, bank =
identity, V2A = semantic/intensity sketch). The placement/retrieval/mix
machinery works mechanically end-to-end and cost $0 for every event placed;
what cannot be judged yet is **retrieval match quality**, because the template
bank currently contains exactly one file (a car gear-shift clank — correctly
rejected for all 20 events, which proves discrimination, not fit).

> **Honesty note:** nobody has listened to the A/B renders yet. All sync
> claims below are mechanical (sample-level alignment against detector
> ground truth), not perceptual judgments. A listening pass on
> `review/index.html` is cheap and should precede any go decision.

---

## 2. What this POC was asked to answer

From `docs/sfx-bank-foley-pipeline-poc.md` §4: prove the loop end-to-end on
the existing 5-clip set — Mirelo (or other raws) as timing oracle → event
extraction → bank retrieval → transient-aligned placement with the tiered
direct/DSP/Stable-Audio-3/fallback strategy → frame-locked mixes; measure
timing error vs auto-sfx contacts; account cost via the ledger pattern;
deliver an A/B review page and a verdict. Non-goals honored: no new model
benchmark sweep, no palette LoRA training, no product UI.

---

## 3. What was built (`experiments/sfx-bank-pipeline/`, ~2.4 kLOC)

| Stage | Capability |
|---|---|
| `oracle_events.py` | SuperFlux onset/intensity/duration extraction from any benchmarked model's raw output (self-contained numpy/scipy implementation — librosa/numba no longer ship macOS x86_64 wheels), geometry audited per run |
| `validate_oracle.py` | Greedy one-to-one matching vs auto-sfx foot-contact truth; races **all four** V2A models per clip; picks winner; enforces the ≤50 ms gate |
| `index_bank.py` | SFX bank indexer: 48 kHz normalized copies, onset maps, CLAP (`laion/clap-htsat-fused`) embeddings + zero-shot category labels |
| `retrieve.py` | Per-event snippet embedding → top-k candidates scored `0.6·audio-cos + 0.4·text-cos`; tier decision: direct / dsp / sa3 / mirelo-fallback with documented thresholds |
| `render_mix.py` | Sample-exact transient alignment; attack never stretched; tail trim or ≤1.35× rubberband compression post-transient only; ledgered SA3 calls; two-pass loudnorm (−16 LUFS) mux onto original frames — identical conventions to fal-sfx-bench |
| `fal_calls.py` | Ledgered Stable-Audio-3 audio→audio caller (fsynced ledger lines, full request records, sha256 audit) |
| `build_ab_page.py` / `report.py` / `reconcile.py` | A/B review page with per-event decision tables; summary; aggregate billing reconciliation reusing bench parsers |

Everything resumable, JSON-with-sha256 artifacts throughout, run dir mirrors
bench layout under `work/sfx-pipeline/<run_id>/`.

---

## 4. Evidence

### 4.1 The decisive result — timing-oracle race (`validation/timing.md`)

Onsets detected in each model's raw soundtrack, matched greedily against
auto-sfx foot contacts (±250 ms window):

| clip | best oracle | matched | median abs err | gate (≤50 ms AND full coverage) |
|---|---|---|---:|---|
| walk1 | controlfoley | 2/4 | 68.2 ms | FAIL |
| walk1 | kling-v2a | 2/4 | 94.0 ms | FAIL |
| walk1 | mirelo16 (brief's proposal) | **0/4** | — | FAIL |
| walk2 | kling-v2a | 3/9 | 57.3 ms | FAIL |
| walk2 | mirelo16 | 2/9 | 94.9 ms | FAIL |

A sensitivity sweep (3× looser detection gates) added candidates but did **not**
tighten errors — they scatter ±50–220 ms with no clustering. The generated
tracks produce plausible rhythm, not synchronized sound. This is a property of
today's hosted models, not of our detector: at max sensitivity Mirelo emitted
29 events on walk1 and still matched only 3 contacts (−46/−137/+81 ms).

### 4.2 Detection methodology finding

Mirelo (and peers) mix footsteps *into* a continuous ambience bed: flux onsets
rise only ~1–6 dB above local texture, so pure spectral-flux peak picking
over-detects ~10× (93 "events" in 9.4 s). Adding an amplitude-domain
energy-jump gate (≥3 dB rise over local past) + envelope smoothing +
refractory period restores physically plausible event counts. Documented in
each `oracle/*.events.json` detector block.

### 4.3 Tiered strategy + economics (`report/summary.md`)

Full real-bank run: 20 events across 5 clips → tiers `{mirelo-fallback × 20}`,
fal spend $0.0000 for placement. Not a failure of the tier logic but its
correct output: the one-file bank scored below the identity floor everywhere
(best combined similarity **0.26** walk2 → **0.06** ocean-short vs floor
0.30). Discrimination proven; fit unjudgeable.

The SA3 audio→audio lane was separately verified **live** end-to-end:
seed honored, `output_format=wav` honored once explicitly set (schema defaults
are duration=30 s / mp3 — both bill or degrade silently if forgotten),
request ids + sha256 records in `runs/ledger.jsonl` / `records/`. Billed
exactly $0.048 = 2 × $0.024.

### 4.4 Reconciliation (`report/reconciliation.md`)

Estimated == billed ($0.0480, delta $0.0000) against the owner's August Fal
CSV; prior-session spend ($2.8535, same credential) and a foreign-key row
($0.6586) correctly segregated. POC total ≪ $5 budget.

### 4.5 Human-verifiable artifacts

`work/sfx-pipeline/20260826-poc/review/index.html` — per clip: A = untouched
Mirelo render, B = pipeline render, plus per-event tables (label + confidence,
top candidate + score components, tier, ops, gap, gain). Zero-shot labels are
sane (gravel footsteps on walks, rain-like surf texture on aerials), so
retrieval decisions will be interpretable when the bank is real.

---

## 5. Findings (evidence → implication)

1. **Hosted V2A is not a timing oracle.** Best case 68 ms @ 50% coverage; the
   brief's proposed Mirelo worst (0/4). → Timing must come from vision (or a
   future model class); V2A output demotes to semantic/intensity reference and
   event proposals where vision has no detector. Answers brief open question §5-3.
2. **The architecture holds with re-stacked layers.** Vision clock + bank
   identity + tiered adaptation ran end-to-end at $0/event; mechanical sync is
   exact by construction. → Product thesis survives its first falsification test.
3. **Retrieval discrimination works; fit is unproven.** One semantically
   distant bank file was rejected for all 20 events (scores 0.06–0.26 < 0.30
   floor). → Populate the library before judging the core value claim.
4. **SA3 a2a is usable but footgunned:** duration default 30 s (billing trap),
   output default mp3; wav + seed work when set. → Encode schema defaults in
   any wrapper (done here); budget $0.024/generation.
5. **macOS x86_64 runtime pin set discovered** (blocks naive installs):
   torch ends at 2.2.2 ⇒ transformers `<5`, numpy `<2`, scipy `1.13.x`,
   librosa/numba unavailable. Working venv recipe documented in README.
   → Future audio-ML work on this host inherits these pins or moves to Docker/GPU.
6. **Loudnorm ceiling on quiet beds:** fallback-crop mixes land at
   −24.8 LUFS achieved (vs −16 target) because linear-mode true-peak headroom
   caps the gain. → Production path should gain-match per event against the
   target, not normalize post-hoc; noted as known limitation.
7. **Perceptual quality unmeasured.** No human has judged the A/B page.
   → Listening pass required before any production claim.

---

## 6. What was explicitly NOT tested

- **Perceptual listening evaluation** of the A/B renders (sync, artifact, believability).
- **Retrieval match quality on a real library** (bank = 1 file today).
- **Palette LoRA training** (SA3 trainer, $3/1k steps) — out of scope per owner; also blocked upstream by missing captions (auto-captioning step designed on paper only).
- **Local ControlFoley upstream** (AC-V2A reference-audio conditioning, 16.7 GB weights, torch 2.5 env) as an alternative timing/reference source — also license-gated: weights CC-BY-NC-4.0.
- **ThinkSound** ($0.001/compute-s, cheapest catalog candidate) — never benchmarked.
- **AC-Foley / MultiFoley** reference-audio models — availability/license scouting not done.
- **Non-footstep vision event classes** (impacts, doors, cloth, vehicles) — auto-sfx supplies foot contacts only, so `man`/ocean clips used V2A/bed fallbacks.
- **Physical adaptation heuristics** (size→pitch, speed→attack sharpness) — specified in the brief, not implemented.
- **Scale** — indexing/placement validated on 1-file real + 4-file fixture banks; hundreds-of-files behavior (embedding throughput, search) unmeasured.
- **Ocean/ambience quality** — bed handling is loop-fill + fades; no crossfade/scene-aware treatment.

---

## 7. Unblocking asks (recommended decisions needed)

1. **Populate the template bank** (owner says more incoming): even ~30–50 files across footsteps-on-surfaces / impacts / water / doors makes the retrieval-quality verdict possible. Convention: `work/sfx-bank/template/<category>/<name>.wav`.
2. **Listening protocol + bar:** who judges the A/B page, and what passes? (Mechanical gate already measured; product gate needs ears.)
3. **License call:** may we run CC-BY-NC ControlFoley weights locally for internal prototyping? It's the strongest remaining V2A-timing candidate (its whole design targets sync).
4. **Palette path approval:** next-phase shape would be CLAP zero-shot auto-captioning of the bank → SA3 trainer fine-tune ($3/1k steps, commercial use). Approve scope/budget?
5. **Direction confirmation:** given Finding 1, we recommend continuing timing-intelligence × bank (vision-led). Alternative worth weighing: invest in better local V2A instead — but that re-centers on generative audio quality rather than the studio-library differentiation the owner wants.
6. **Budget envelope:** iterations cost ~$0.05–$0.15 fal spend each at current usage; $5/session cap comfortable unless palette training starts.

---

## 8. Costs & constraints

External spend this session: **$0.048** (reconciled, delta $0.0000). Compute:
CPU-only (2018 Mac mini-class host). Repository safety: commits are local on
`main`; unrelated operator WIP in `experiments/auto-sfx/` untouched.

---

## 9. Appendix — reproduce

```bash
PY=/tmp/pab-sfx-bench/bin/python          # pins: numpy<2, scipy 1.13, transformers<5, torch 2.2.2 cpu
RUN=work/sfx-pipeline/$(date -u +%Y%m%d-%H%M%S)
$PY experiments/sfx-bank-pipeline/oracle_events.py   --out-dir $RUN
$PY experiments/sfx-bank-pipeline/validate_oracle.py --run-dir $RUN
# drop bank files into work/sfx-bank/template/ first
$PY experiments/sfx-bank-pipeline/index_bank.py    --run-dir $RUN --force
$PY experiments/sfx-bank-pipeline/retrieve.py      --run-dir $RUN        # --no-sa3 disables paid tier
$PY experiments/sfx-bank-pipeline/render_mix.py    --run-dir $RUN
$PY experiments/sfx-bank-pipeline/build_ab_page.py --run-dir $RUN
$PY experiments/sfx-bank-pipeline/report.py        --run-dir $RUN
$PY experiments/sfx-bank-pipeline/reconcile.py     --run-dir $RUN --csv <Fal-usage.csv>

```

Key artifacts: `work/sfx-pipeline/20260826-poc/review/index.html` (open in
browser) · `validation/timing.md` · `report/summary.md` ·
`report/reconciliation.md` · `docs/sfx-bank-pipeline-verdict.md`.

---

## Addendum (2026-08-26, post-report — pre-handoff)

Seven additional studio WAVs were imported (swisstransfer drop): camera ×2,
designed ×2, vehicle ×3, joining the initial gear-shift clank → **bank = 8
files** under `work/sfx-bank/template/{camera,designed,vehicle}/` (original
filename taxonomy preserved; spaces sanitized). All decode cleanly and are
indexed with embeddings (`bank/index.json`, 8/8 rows). Note the new content
(camera shutters, whooshes, bass dives, wipers, F1) sits outside the current
footstep-centric clips — it widens what the *next* test set can exercise
(impacts/UI/vehicle scenes), while today's 5-clip run would still correctly
reject most of it. Zero-shot label vocabulary is intentionally minimal;
extend via `--labels-file` as categories grow. Retrieval-quality verdict on a
real library remains the open ask (#1 in §7).
