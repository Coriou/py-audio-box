# Phase-2 evaluation-first session — verdict

**Date:** 2026-08-26 · **Scope:** brief `docs/sfx-bank-foley-phase2-eval-brief.md`
§3 (Q1–Q3 + prep for Q4). Directive context: CTO report is authoritative and
was read in full. Every build below reused the POC machinery; nothing rebuilt.

## Q1 — blinded listening harness: BUILT, awaiting owner ratings

`experiments/sfx-bank-pipeline/build_blind_review.py` generated
`work/sfx-pipeline/20260826-poc/review/blind/`
(page + `assignment.json` + `RATING_GUIDE.md`) over the EXISTING renders:
5 clips × {mirelo16, bank} × 20 events. No generation was performed.
Guarantees verified headlessly: zero plaintext system names before reveal
(base64 config blob), gate closed at 0/160, counter reaches 160/160,
serialization emits annotator/cfg_b64/ratings, reveal decodes the true
mapping. Owner protocol lives in `experiments/sfx-bank-pipeline/phase2-handoff/HANDOFF.md`.
**Q1 statistical answer pending owner's listening pass** — cannot be closed
by an agent by design.

## Q2 — gold annotation round-trip + bake-off scorer: BUILT

- `experiments/auto-sfx/gold_annotator.html`: fps-aware frame transport
  (`requestVideoFrameCallback`, ±1/±10 frame stepping, J/K/L), CTO §6 family
  taxonomy editor with start/contact/end capture buttons, acceptable window
  auto `[contact−1, contact+1]`, per-annotator JSON downloads matching the
  cross-session contract schema exactly.
- `experiments/auto-sfx/scripts/build_gold_session.py`: ffprobe-verified
  demo session over walk1(30 fps)/man(29.97)/ocean-short(30).
- `experiments/sfx-bank-pipeline/timing_bakeoff.py`: recall@±33/±50/±100 ms,
  FP/min, median+p95 (never median alone; CTO §7 exact set) plus acceptable-
  window coverage secondary metric. Known-answer selftest PASSES (boundary
  ±33 ms inclusive, FP/FN accounting, tightest-tolerance error stats);
  dry-run mode reproduces detector self-consistency numbers.

Annotator IDs A/B will be two independent passes by the owner (their call),
so inter-rater disagreement data will exist even without a second human.
**Timing bake-off vs auto-sfx runs the moment ≥3 impulse clips carry
contact_frame values; scorer is already green on fixtures.**

## Q3 — retrieval top-5 harness + overlap manifest: BUILT

`retrieval_eval.py`: recomputes retrieve.py scoring (sync-documented formula)
with the tier floor bypassed → forced top-5 per event, browser-taggable page +
audio copies for all 8 indexed bank files across all 20 plan events
(`work/sfx-pipeline/20260826-poc/retrieval-eval/`). tally aggregates
top-1/top-3 acceptable, chosen==human-best, wrong-class %, failure
localization histogram (bank-lacks-sound | search-miss | scoring-wrong |
adaptation-damage) — selftest PASSES with hand-checked fixture numbers.
Overlap manifest seeded at
`experiments/sfx-bank-pipeline/phase2-handoff/bank_overlap.manifest.json`:
ocean and footstep rows pre-seeded from existing assets; 5 needs_clip
placeholders match the owner's shoot-phone-clips decision (wipers, gear-shift,
camera close-up, whoosh, bass-dive). **Statistical retrieval answer waits for
the 10–20 variant/class bank expansion — unchanged per CTO §28 "don't scale
the library past proving retrieval" — now proven-in-principle via harness.**

## Event model v2 + stems + registry: COMMITTED under this experiment

- `events_v2.py` canonical additive fields incl. documented keyword-heuristic
  temporal_type inference; plans bumped to schema_version 2; v1 keys intact.
- Repetitive sequences share sequence_id and incur ×0.85 candidate-reuse rank
  penalty (CTO §4C diversity pressure); decide_tier flow otherwise untouched.
- render_mix persists `stems/<clip>/evt_<id>.wav` + `ambience.wav` at 48 kHz;
  internal mixes remain lossless WAV; global −16 LUFS confirmed review-path
  only (this was already true — now explicit in README §Phase-2 additions).
- Latent POC bug found+fixed en route: bed events lost their `kind` marker in
  plans, so surf clips silently degraded to fallback-crop audio instead of the
  documented ambience loop-fill; beds now tier `direct` with a chosen bank
  candidate and produce a proper ambience stem.
- `registry.yaml` seeds QUALITY-vs-COMMERCIAL rows verbatim from CTO §25 plus
  ThinkSound/Synchformer/FoleyCrafter-onset/GreatestHits/MultiFoley/FoleyGenEx
  watchlist entries.

End-to-end proof run `work/sfx-pipeline/20260826-154059-v2check/`: full
retrieve (all clips) + targeted render ($0 spend, SA3 unused, ledger empty).

## Honest verification matrix

| Item | Method | Result |
|---|---|---|
| All python entrypoints | py_compile ×10 | green |
| blind page logic | bun-run of extracted JS w/ DOM stub | gate/collate/reveal verified |
| annotator/top5 pages | HTTP 200 + inline-JS syntax check | pass |
| in-browser interaction | NOT VERIFIABLE this session — Chromium spawn failed 3× | residual risk |
| bake-off math | known-answer selftest incl. inclusive-boundary assertion | PASS |
| v2 chain end-to-end | $0 rerun, stems probed pcm_f32le@48 kHz, v2 fields asserted | pass |
| fal ledger discipline | zero calls; no network writes anywhere | clean |

## Decisions recorded (owner, 2026-08-26)

Second annotator = owner twice as ids A/B · clip source = shot phone clips ·
no VLM key ⇒ Stage-1 proposal prototype stays CLOSED (CTO P2 stretch row) ·
SA3 authorized ≤$0.12 for the future Q4 probe (still $0 used this session).

## What would close Q1–Q3 next session

1. Owner completes the blinded rating pass (~15 min) → Q1 answered with named
   failure examples pulled from reveal tallies.
2. Owner finishes ≥3 clip annotations (two passes) → `timing_bakeoff` produces
   the real recall/p95 table vs auto-sfx → partial Q2 answer; adding isolated
   VideoOnsetNet requires its checkpoint-license verification first.
3. Shots land + expected-bank rows filled → tag the top-5 page once per event
   → failure-localization histogram answers whether retrieval search, scoring,
   or bank coverage is the binding constraint (Q3 partial; full answer after
   bank expansion per CTO §19).
