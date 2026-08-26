# Phase-2 brief — disentangle timing, retrieval, adaptation (evaluation-first)

**Status:** handoff brief for the NEXT orchestrator session. Written 2026-08-26,
after the sfx-bank-pipeline POC (`experiments/sfx-bank-pipeline/`, commits
`2bc817a`→`4bb36ea`) and the CTO review
(`experiments/sfx-bank-pipeline/reports/CTO-REPORT-timing-oracle-guidance.md` —
**read it in full; it is authoritative**). Nothing below is built yet except
where noted.

## 0. Where things stand (one paragraph)

POC proved the machinery (bank indexing/CLAP retrieval, tiered
direct/DSP/SA3/fallback strategy, sample-exact transient placement, ledgered
fal calls, A/B rendering) and falsified one premise: hosted V2A soundtracks
(Mirelo/Kling/Sonilo/ControlFoley raws) are **not** usable as the timing clock
(best 68 ms @ 50% coverage vs ≤50 ms gate; detection sensitivity is not the
bottleneck). CTO accepted that, retained the architecture hypothesis
("vision = when/what · bank = identity · DSP = cheap adaptation · SA3 =
repair/variation · pure V2A = fallback/baseline"), and redirected the next
phase: **stop building the stack; benchmark each layer independently against
human ground truth, then recombine only what wins.**

## 1. CTO directives that shape everything (digest — full text is authoritative)

- **Do NOT claim vision is a perfect clock.** Our "ground truth" so far is
  another detector (auto-sfx). Phase-2 needs **human frame-level annotations**
  (two annotators where practical, frame-based, acceptable-window e.g.
  `{contact_frame: 143, acceptable_frames: [142,144]}`, disagreements kept).
- **Nobody has listened yet.** Highest-priority gap. Blind the systems on the
  review page (randomized order, labels revealed after rating). Metric that
  matters: *"Would you keep this result without manually replacing the sound?"*
- **Event taxonomy is now mandatory** — stop treating every event as
  label+timestamp: A impulsive (contact ts + intensity) · B continuous gesture
  (start/end + envelope) · C repetitive sequence (family id + ordered ts) · D
  beds (scene interval + envelope + environment) · E hybrid structured objects
  (door = creak body + latch transient + tail). Canonical event schema in CTO §5.
- **Retrieval:** CLAP cosine is an internal feature, not a quality metric.
  Upgrade path = semantic filter → CLAP/text → acoustic-descriptor rerank
  (duration/transient strength/decay/centroid/rolloff/noisiness…), scored
  per event class — no global 0.30 threshold. Log top-5 candidates and have
  humans tag best/acceptable/wrong so failures localize (no-sound vs bad-search
  vs bad-scoring vs damaged-by-DSP).
- **Timing metrics:** recall@±33/±50/±100 ms, false-positives/min, median AND
  p95 error (never median alone). Provisional gates: ≥90% recall@±50 ms,
  median ≤33 ms, p95 ≤~80 ms; retrieval top-3-acceptable ≥80%, top-1 ≥65–70%,
  wrong-class <3%; hybrid beats/ties pure Mirelo in ≥60% of blinded comparisons.
- **Vision timing is hierarchical:** Stage 1 VLM semantic proposal ("what +
  roughly when"; Gemini custom-FPS sampling / Qwen3-VL temporal grounding are
  candidates) → Stage 2 native-frame specialist refinement on proposed windows
  only (pose/keypoints for footsteps; object tracking/collision for impacts;
  geometry velocity-stop for mechanisms; optical-flow envelopes for continuous).
- **Benchmark FoleyCrafter's `VideoOnsetNet` in isolation** before building our
  own onset specialist — verify the *checkpoint's* license separately from the
  Apache repo (Auffusion dependency is NC; do not assume).
- **Greatest Hits dataset (CC-BY-4.0)** = future path for owning a small timing
  model. Do NOT train it yet.
- **Reference-conditioned V2A** (ControlFoley/AC-Foley, both CC-BY-NC weights):
  research benchmarks only, never production deps without written clearance.
  Question they answer: "how much does joint timing+timbre buy?"
- **Mirelo stays** as (a) mandatory baseline ("just give the video to Mirelo"),
  (b) missing-event fallback, (c) semantic/intensity sketch — never the clock.
  Fal endpoints current: `mirelo-ai/sfx1.6/video-to-video` $0.01/s;
  `mirelo-ai/sfx-v1/video-to-audio` ~$0.007/s.
- **ThinkSound**: quality benchmark yes; production only after Fal confirms in
  writing upstream NC doesn't apply. Synchformer: QC/global-offset only, never
  the contact clock.
- **Mixing:** persist per-event STEMS (`stems/evt_001.wav`, `ambience.wav`);
  local/scene gain matching instead of early global loudnorm; lossless 48 kHz
  internally, encode only for review assets; add perspective metadata (distance/
  screen position/indoor-outdoor) eventually.
- **Physical adaptation heuristics:** implement conservatively (velocity→gain,
  size→pitch/spectrum, distance→level/HF), always A/B against unmodified source.
- **License/provider registry** as data (YAML: provider, commercial_status,
  reference_audio, production_allowed) separating QUALITY from COMMERCIAL status.
- **What NOT to do** (§28, near-verbatim): no palette fine-tuning yet; no more
  SuperFlux-vs-Mirelo tuning; don't scale library past 50–100 sounds before
  retrieval is proven; no V2A model tours without hypotheses; never equate
  mechanical alignment with perceptual sync; never call auto-sfx ground truth;
  no NC checkpoints in shipping paths; no NLE integration; no infra optimization;
  no product UI. Heavy ML experimentation belongs in containers/GPU, not the
  Intel-host venv (pins: numpy<2, scipy 1.13.x, transformers<5, torch 2.2.2-cpu).

## 2. Reusable assets (do NOT rebuild)

| Asset | Where |
|---|---|
| Pipeline stages (oracle/validation/index/retrieve/render/A-B page/report/reconcile) | `experiments/sfx-bank-pipeline/*.py`, conventions in its README |
| Prior run artifacts (Mirelo+3 raws, plans, mixes, review page, ledgers, reconciliations) | `work/sfx-pipeline/20260826-poc/`; bench raws `work/sfx-bench/20260825-165347/` |
| SFX bank (8 files: camera×2, designed×2, vehicle×3, gear-shift; indexed w/ CLAP embeddings) | `work/sfx-bank/template/{camera,designed,vehicle}/` |
| Foot-contact detector + annotation helper + adjudicated draft GT | `experiments/auto-sfx/` (`run annotate`, `gt/`, `gt-adjudication/`, `scripts/build_agent_draft_gt.py`) |
| Bench clips + trim mapping + prompts | `prepare_inputs.py::CLIPS`, `common.py::CLIP_TRIMS/CANONICAL_PROMPTS` (walk2.mp4 sibling is CORRUPT — use webm) |
| Runtime | `/tmp/pab-sfx-bench` venv (recreate per README pins), `FAL_KEY` in `.env`, ffmpeg+ffprobe host-side, rubberband filter available |
| Billing discipline | fsynced `Ledger`, aggregate reconcile vs Fal CSV (owner exports on request; last export covered 2026-08 month) |

## 3. Scope for the next session (answers CTO §29 Q1–Q3; Q4 stretch)

Ordered by CTO priority. Human-in-the-loop steps are unavoidable — the
orchestrator's job is to make each human step take minutes, not meetings.

1. **Blinded listening harness (Q1).** Extend the review builder with a blind
   mode: randomized system ordering (A/B/C…), labels withheld until ratings
   submitted, per-event + per-clip 1–5 ratings on sync / semantic correctness /
   physical believability / audio quality / palette consistency / artifacts /
   overall preference + the binary keep-replace question. Ratings stored as
   JSON next to the run. Then hand it to the owner to score the EXISTING
   20260826-poc renders (zero new generation). Deliver scored results + named
   failure examples before touching any pipeline code.
2. **Human-gold timing annotations (Q2 prerequisite).** Select 8–12 short clips
   spanning the CTO family table (footsteps hard/gravel/grass/stairs/run,
   impacts, mechanisms incl. camera shutters, vehicles incl. wipers/gear,
   cloth/body, continuous scrape/pour, natural water/wind, multi-event).
   Source: owner-shot phone clips and/or clearly-licensed stock (record
   provenance!). Build the annotation round-trip: frame-accurate player with
   scrub/step, minimal UI writing the CTO §6 schema
   (class/start/contact/end/acceptable_window/confidence, two-annotator fields),
   reusing auto-sfx annotation machinery where possible.
3. **Timing bake-off vs human gold (Q2, no audio involved).** Once ≥3 annotated
   clips exist: score auto-sfx (and, license-check permitting, isolated
   FoleyCrafter `VideoOnsetNet`) against human frames with the CTO §7 metric
   set. Optional stretch if a Gemini/Qwen3-VL key is available: VLM proposal +
   refinement prototype, proposals logged even if refinement is manual.
4. **Bank↔video overlap set (Q3 prerequisite).** For each new test clip, list
   which bank file SHOULD win; expand bank toward 10–20 variants per important
   one-shot class (owner provides; log provenance/licenses in the registry).
   Build the retrieval-eval harness: forced top-5 display, human
   best/acceptable/wrong tags, failure-localization tally.
5. **Event model v2 + stems (enables Q4 later).** Implement the canonical event
   schema (temporal_type-aware) across oracle/retrieve/render; persist
   `stems/evt_*.wav` + ambience stem; switch to local gain matching (global
   loudnorm only as final review encode); lossless internal renders. Registry
   YAML for providers/licenses (seed with known rows from CTO §25).
6. **Stretch (only if 1–5 land early): adaptation isolation probe (Q4/P3)** —
   on events with human-gold timestamps, compare direct vs DSP vs SA3-A2A vs
   Mirelo crops, same placement, blinded; SA3 capped ≈$0.12, Mirelo re-runs
   only where a fallback crop doesn't already exist.

## 4. Non-goals (CTO §28 applies verbatim)

No palette training; no SuperFlux/Mirelo clock tuning; no library scaling past
proving retrieval; no new V2A benchmarks without a hypothesis; no claims of
mechanical=perceptual sync; auto-sfx ≠ ground truth; no NC weights in shipping
paths; no NLE integration; no infra optimization; no product UI. Keep heavy ML
containerized/GPU-ready.

## 5. Deliverables & budget

- Blinded-listening package + owner-completed ratings (Q1 answer).
- Annotation tooling + first human-gold set (≥3 clips fully annotated by ≥1
  human, 2 where practical) + timing bake-off report vs auto-sfx (Q2 partial
  answer; full answer may need another session of annotation).
- Retrieval-eval harness + overlap manifest (Q3 ready; statistical answer waits
  for bank expansion).
- Event-model-v2 + stems rendering, committed under
  `experiments/sfx-bank-pipeline/` (or a sibling experiment dir) following repo
  conventions; license/provider registry checked in.
- Ledger every fal request (fsynced JSONL) and reconcile against a fresh Fal
  CSV before finishing. Budget ≤ $5; expected spend ≈ $0 unless stretch runs.

## 6. Open questions for the owner (ask early, don't stall)

- Who is the second annotator (CTO wants two people where practical)?
- Preferred source for bank-overlap test clips: shoot vs licensed stock vs both?
- Is a Gemini or Qwen3-VL API key available for the proposal-stage prototype?
- Confirm SA3 spend authorization for the Q4 stretch (~$0.12 cap).

## 7. Paste-ready prompt for the next orchestrator

```text
Read these three docs first — they are authoritative for this session:
1. experiments/sfx-bank-pipeline/reports/CTO-REPORT-timing-oracle-guidance.md
   (CTO review: decisions, event taxonomy, gates, priorities P0–P4, what NOT to do)
2. docs/sfx-bank-foley-phase2-eval-brief.md (this phase's scoped plan, assets,
   runtime facts, deliverables)
3. experiments/sfx-bank-pipeline/README.md (pipeline stages, quirks, how to run)

Context: the POC proved the bank-retrieval/placement/mixing machinery and
falsified hosted-V2A-as-timing-clock. The CTO directed an evaluation-first
phase: separate timing, retrieval and adaptation; measure each against HUMAN
ground truth; recombine only winners. Do not rebuild existing stages; reuse
work/sfx-pipeline/20260826-poc/, the bench raws, and the indexed 8-file bank.

Goal, in CTO priority order (details in the phase brief §3):
Q1 blinded listening harness + owner scores the EXISTING renders (before any
   code changes);
Q2 human-gold frame annotations (schema in CTO §6) + timing bake-off vs
   auto-sfx with recall@±33/50/100ms, FP/min, median+p95 — no audio generation;
Q3 bank↔video overlap set + retrieval top-5 harness with human
   best/acceptable/wrong tags;
then event-model-v2 (temporal_type-aware) + per-event stems + local gain
matching + license/provider registry YAML; Q4 adaptation-isolation probe ONLY
as stretch.

Non-goals (CTO §28): palette training, SuperFlux-vs-Mirelo tuning, library
scaling, unhypothesized V2A benchmarks, NC checkpoints anywhere near shipping,
NLE/UI/infra work. Auto-sfx is a detector, not ground truth; mechanical
alignment ≠ perceptual sync.

Constraints: budget ≤ $5 total fal spend (expected ≈ $0; SA3 stretch cap
≈ $0.12); fsynced ledger + fresh-CSV reconciliation before finishing; commits
under experiments/ following repo conventions; heavy ML containerized/GPU-ready,
host venv pins numpy<2 / scipy 1.13.x / transformers<5 / torch 2.2.2-cpu;
walk2.mp4 is corrupt (use webm); FAL_KEY in .env; human-in-the-loop steps must
be reduced to minutes for the owner (blinded player + rating forms + annotation
UI), and blocking on the owner is expected at rating/annotation points — build
everything else while waiting.

Deliverables: blinded listening package + recorded ratings; annotation tooling
+ first human-gold clips; timing bake-off report; retrieval-eval harness +
overlap manifest; event-model-v2/stems/registry committed; honest session
verdict appended to reports/ answering whatever subset of Q1–Q3 closed.
```

## 8. Sources

- CTO review: `experiments/sfx-bank-pipeline/reports/CTO-REPORT-timing-oracle-guidance.md` (2026-08-26)
- POC session report: `experiments/sfx-bank-pipeline/reports/CTO-REPORT-timing-oracle-sfx-bank-poc.md`
- Verdict doc: `docs/sfx-bank-pipeline-verdict.md`
- Original POC brief: `docs/sfx-bank-foley-pipeline-poc.md`
