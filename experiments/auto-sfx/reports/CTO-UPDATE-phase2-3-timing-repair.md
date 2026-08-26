# Auto-SFX Phase 2 — CTO Update #3 (timing repair)

**Date:** 2026-08-26
**Scope:** Phase 2B identity tracking, contact-state reasoning, pose/smoothing
ablations, foot-pixel ROI fork, frozen evaluation, and real review artifacts.
**Execution:** local-only Docker/Linux CPU; no push, PR, release, or remote
mutation. `hybrid-v0` remains the default.

## 1. Outcome first: better mechanisms, failed exit gate

`contact-state-v1` is materially better than `hybrid-v0` on the frozen marked
subset, but it is not accurate enough to become the default. Across the frozen
walk1 + walk2 shards it improves strict side-agnostic precision/recall from
**0.120/0.059 to 0.273/0.176**, diagnostic median/p90 timing from **5/8 to 3/6
displayed frames**, and >3-frame pairs from **81.0% to 35.5%**. It removes the
two frozen simultaneous L/R pairs and eliminates multi-second `CANDIDATE`
runs because the new state vocabulary has no `CANDIDATE` state.

That is a real repair, not a pass. The required frozen bar is 0.90/0.90,
median ≤1 frame, p90 ≤3, ≤5% beyond 3 frames, and known-side accuracy ≥0.95.
The final candidate misses every numeric criterion. Of 36 frozen contacts
labelled `clear`, it strictly matches 8: clear-contact recall is **0.222**. A
conservative clear precision lower bound that retains all 33 predictions in
exhaustively marked ranges is **0.242**; unmatched predictions are not assigned
an invented visibility label. Anatomical side is **0/5** on strict known-side
matches. The current pose and ROI inputs therefore cannot meet this regression
gate. This is evidence about these two clips and implementations, not a claim
about the general population or every monocular method.

### Real frozen before/after

| Frozen subset | Method | strict P / R | diagnostic median / p90 | pairs >3f | known-side | simultaneous L/R |
|---|---|---:|---:|---:|---:|---:|
| walk1, 21 marks | `hybrid-v0` | .143 / .048 | 8 / 8.6 f | 80.0% | n/a: all draft sides uncertain | 1 |
| walk1, 21 marks | `contact-state-v1` | .150 / .143 | 3 / 5 f | 31.6% | n/a | 0 |
| walk2, 30 marks | `hybrid-v0` | .111 / .067 | 5 / 8 f | 81.3% | .000 | 1 |
| walk2, 30 marks | `contact-state-v1` | .462 / .200 | 2 / 7.8 f | 41.7% | .000 | 0 |
| **combined, 51 marks** | **`hybrid-v0`** | **.120 / .059** | **5 / 8 f** | **81.0%** | **.000 (2/2 flipped)** | **2** |
| **combined, 51 marks** | **`contact-state-v1`** | **.273 / .176** | **3 / 6 f** | **35.5%** | **.000 (5/5 flipped)** | **0** |

The wider diagnostic association does not inflate P/R. A prediction outside
the 34 ms strict tolerance remains an existence miss + false positive while
still contributing its signed timing error.

## 2. Ground-truth provenance and frozen protocol

No operator GT appeared. Every mark used here is explicitly **agent-draft;
NOT operator verified**. Draft evidence can guide experiments but is not
operator-verified product evidence.

- `walk1`: one raw-only pass was too sparse; another raw-only pass contained
  mixed gait phases. An isolated raw-frame semantic audit adjudicated every
  proposed row and replaced toe-off/push-off marks 548/598/676/1038 and swing
  marks 122/200/280 with defensible IC frames. The resulting 41 contacts carry
  ±1-frame windows and `foot: uncertain`; anatomical side was not inferred.
- `walk2`: two independent passes inspected raw frames without predictions,
  traces, event times, or overlays. Disagreements were adjudicated on raw-frame
  sheets. The resulting 54 contacts span frontal, rear, sagittal, and uncertain
  views; 20 left, 21 right, and 13 uncertain, with uncertainty retained.
- The development/validation allocation in
  [`phase2b-split.json`](../gt/phase2b-split.json) was frozen from temporal and
  shot/viewpoint ranges before usable contact tables existed. A symmetric
  two-frame guard matches the maximum draft uncertainty at shard boundaries;
  it does not reassign a range or mark.
- Canonical draft inputs are
  [`walk1.agent-draft.gt.json`](../gt/walk1.agent-draft.gt.json) and
  [`walk2.agent-draft.gt.json`](../gt/walk2.agent-draft.gt.json). The raw
  walk2 adjudication table is
  [`walk2.agent-draft.adjudication.csv`](../gt/walk2.agent-draft.adjudication.csv).

Compact operator-validation sheet, deliberately raw and prediction-free:

![Walk2 agent-draft operator-validation sheet](../../../work/auto-sfx/gt-adjudication/phase2b-walk2/walk2-agent-draft-operator-validation.jpg)

Walk1 sheets are under
`work/auto-sfx/gt-adjudication/walk1_recovery_adjudication_*.jpg`; raw pass and
semantic-audit records are under `work/auto-sfx/gt-drafts/`. These runtime
artifacts were preserved and not committed as operator files.

## 3. Accepted mechanisms

The accepted outcome is an opt-in research architecture and regression
fixtures, not a validated detector default.

1. **Continuity-aware feet before contact reasoning.**
   [`tracking.py`](../src/auto_sfx/tracking.py) separates temporal track
   identity from anatomical left/right attribution. A permutation is corrected
   only when the direct assignment violates a physical 14 body-scale/s bound,
   the crossed assignment is plausible, and the exchange is bounded. The
   high-confidence MediaPipe swap at frame 364 is isolated: raw speeds L
   20.615/R 18.262 become L 4.974/R 2.440 body-scale/s. Long exchanges,
   implausible jumps, pose loss, and crop loss abstain. Interpolated samples
   remain invalid contact evidence.
2. **Bounded local/cycle statistics.** No historical maximum survives for the
   rest of a shot. Thresholds are local and clipped; shot/pose segments reset
   state. The historical 6.53/6.39 thresholds and 678/695-frame `CANDIDATE`
   runs cannot occur in `contact-state-v1`.
3. **Physical state/change-point model.**
   [`state_contacts.py`](../src/auto_sfx/state_contacts.py) retains heel and toe
   separately; computes velocity components, acceleration, support height,
   persistence, confidence, and body-relative phase; and looks retrospectively
   for forward swing → locally advanced apex → sustained rearward supported
   motion. Heel/toe onset disagreement becomes an uncertainty window. Toe-off
   is inferred internally to bound grounded duration.
4. **Honest raw evidence.** Public events retain `raw_layer:true`. Pose loss,
   cropping, discontinuity, and unobservable phase emit explicit abstentions.
   No decoder creates or silently removes contacts because physical phase has
   not passed validation.
5. **Evaluation semantics.** Metrics aggregation now reports the correct
   known-side denominator and a conservative clear-visible view. Development
   and validation use identical frozen shards for every ablation.

## 4. Rejected mechanisms and negative results

| Mechanism / run | Development evidence | Decision |
|---|---|---|
| Assignment-cost-only tracking | 96 MediaPipe-video, 193 image-mode, and 24 RTMPose "swaps" by latching onto normal crossings | Rejected; continuity cost alone is not label-exchange evidence. |
| Bounds 8/10 body-scale/s | Latched crossed tracks around frames 355–366 | Rejected; below legitimate fast motion. |
| Corrected identity into `hybrid-v0` | Events 25→39, simultaneous pairs 5→6, longest `CANDIDATE` still 102 frames | Rejected; identity cannot repair threshold lifecycle/state semantics. |
| MediaPipe image | Walk1 P/R .368/.350, median/p90 2/5 f; 20 discontinuities | Rejected; no existence gain and worse tail. |
| RTMPose video | Walk1 P/R .263/.250, 2/5 f; production license not cleared | Rejected; worse accuracy, research-only. |
| Light robust smoothing | All 40 event times identical to raw | Rejected; no benefit. |
| Offline zero-phase smoothing | 18/40 timestamps changed; P/R .316/.300; large 429→415 and 921→932 shifts | Rejected; plot smoothness did not improve timing. |
| ROI retained full 1080p clip | 6,515.7 MB peak RSS | Rejected resource shape. |
| `phase2b-roi-walk1-dev1` | P/R .368/.350, median/p90 2/6.2 f; shifts −7…+5 | Rejected; no existence gain, worse timing. |
| `phase2b-roi-walk1-dev2` | Same P/R, median/p90 2/7 f | Rejected; edge isolation did not localise touchdown. |
| `phase2b-roi-walk1-dev3` | q90/downward-flow variant ran >9 minutes without output | Rejected CPU cost; code restored to the completed q75 prototype. |

The ROI fork was constrained by Kobayashi and Nakazawa's 2025
[heel-strike/toe-off paper](https://www.jstage.jst.go.jp/article/transjsme/91/945/91_25-00049/_article/-char/en): foot-region optical flow, acceleration,
static-region, and edge evidence are relevant ideas. Its reported results are
not imported. Their setting used a fixed low camera, foot-only controlled
walking course, 640×360/30 fps, 13 young students, manual frame truth, and a
±3-frame correctness band; ours has moving/edited cameras, 1080p or portrait
multi-shot crops, unknown subjects, and stricter existence/timing rules. The
paper is CC BY-NC-ND 4.0, so the implementation is an independently written,
smaller prototype rather than copied material.

Both completed ROI variants regressed development, so they were correctly
rejected before frozen validation. A walk1-only gain was not manufactured.

## 5. Representative visual failures and corrections

At 18.100 s, `hybrid-v0` is trapped in a long low-speed candidate and misses
the draft contact. Yellow is the draft uncertainty window.

![Hybrid baseline miss at 18.100 s](../../../work/auto-sfx/runs/phase2b-baseline-walk1-agent-draft/forensics/gt_18100ms.png)

On the identical pose input, the body-relative state transition emits at
18.067 s: one displayed frame early and inside the defensible window. The
bottom panel now exposes only `SWING`/`GROUNDED`, never multi-second
`CANDIDATE`.

![State correction at 18.067 s](../../../work/auto-sfx/runs/phase2b-state-final-walk1/forensics/evt_021.png)

Honesty also means missing a contact when evidence is broken. At 20.767 s,
frames 623–628 contain a discontinuity/pose gap. The state method does not
bridge it into a synthetic contact; the blank trajectory interval and explicit
miss are visible.

![Explicit discontinuity abstention at 20.767 s](../../../work/auto-sfx/runs/phase2b-state-final-walk1/forensics/gt_20767ms.png)

The unresolved multi-view failure is equally visible. At walk2 10.867 s the
pose trajectory jumps around the shot/view geometry and the right draft
contact is missed; this is why the frontal/rear validation result cannot be
replaced by the cleaner profile result.

![Walk2 missed contact at 10.867 s](../../../work/auto-sfx/runs/phase2b-state-final-walk2/forensics/gt_10867ms.png)

## 6. Acceptance matrix

| Criterion on frozen subset | Final evidence | Status |
|---|---|---|
| clear-visible side-agnostic P/R ≥.90/.90 | conservative P lower bound .242; clear recall .222 (8/36) | **Fail** |
| median absolute timing ≤1 frame | 3 frames | **Fail** |
| p90 absolute timing ≤3 frames | 6 frames | **Fail** |
| ≤5% diagnostic pairs >3 frames | 11/31 = 35.5% | **Fail** |
| anatomical side ≥.95 where known | 0/5 correct | **Fail** |
| no unexplained simultaneous L/R | 0 frozen; one full-clip pair at 27.100/27.133 s is retained and explicitly diagnosed | Pass for frozen; raw contradiction remains visible |
| no multi-second `CANDIDATE` during observable walking | state set is `SWING | UNCERTAIN | GROUNDED`; no `CANDIDATE` token in frame states | **Pass** |
| pose loss/crop → explicit abstention | final walk2: 5 phase-abstention spans, 14 pose-lost and 33 feet-out-of-frame diagnostics | **Pass** |
| neither clip/viewpoint materially regresses | walk1 and walk2 frozen existence/timing both improve over baseline; walk2 remains poor | Pass relative-to-baseline, not absolute |
| CPU/resources compatible locally | state contact stage 2.72 s walk1 / 25.81 s walk2; 167/188 MB peak including render | **Pass** |

The criterion set as a whole fails. `contact-state-v1` is not accepted and is
not the default.

## 7. T3, forensics, previews, and resources

Final diagnostic runs:

- `phase2b-state-final-walk1`: 40 raw events, 19 left/21 right; 35.567 s,
  1920×1080. Cached pose, contacts 2.72 s, render 80.58 s, total 94.20 s,
  peak RSS 167.0 MB.
- `phase2b-state-final-walk2`: 35 raw events, 15 left/20 right; 44.966 s VFR,
  720×1280. Cached pose, contacts 25.81 s, render 56.46 s, total 94.36 s,
  peak RSS 188.3 MB.
- Both use Docker/Linux CPU-only and six-thread limits. Rendering, not the
  walk1 contact method, dominates final wall time. Completed ROI contact
  stages took 103.34/114.09 s and 364.3/401.8 MB, without accuracy benefit.

Each final state run contains `annotated.mp4`, `marker-preview.mp4`,
`contact-trace.jsonl`, tracked pose, Metrics-v2 shard outputs, T3 JSON/CSV, and
per-event/per-miss forensics. These are diagnostic candidates, not accepted
artifacts—the acceptance rule requires videos for every accepted candidate,
and there is no accepted candidate.

T3 full-clip counts:

| Run | T3 counts |
|---|---|
| walk1 | `NONE 10 · DUPLICATE_CONTACT 1 · MISSED_CONTACT 2 · UNKNOWN 29` |
| walk2 | `NONE 8 · EARLY_PHASE 3 · LATE_STANCE 5 · MISSED_CONTACT 30 · WRONG_SIDE 8 · SIMULTANEOUS_LR 2 · DUPLICATE_CONTACT 1 · UNKNOWN 8` |

The two walk2 `SIMULTANEOUS_LR` rows are the same raw 33 ms pair. No decoder
deleted them. Sequence consistency remains diagnostic-only until physical
phase works.

## 8. Tests, commits, and integrity

Docker test command: `make auto-sfx-test` → **284 passed in 17.72 s**. The
starting baseline was 272; 12 Phase 2B behavioral tests cover one-frame
high-confidence L/R swaps, implausible jumps, quarantine/interpolation,
legitimate crossings, bounded threshold recovery after a velocity spike,
invalid-frame exclusion, camera-translation invariance, explicit unobservable
phase abstention, camera-motion subtraction in ROI, ROI change-point selection,
ROI abstention, and known-side aggregation.

Local commits, oldest first:

- `728b6a5` — continuity tracker, versioned state/ROI forks, CLI wiring,
  forensics/metrics integration, first regression fixtures.
- `f517c0a` — agent-draft canonical GT, frozen shards, converters/aggregator,
  corrected known-side denominator.
- `e624e8d` — body-relative contact-state repair, completed ROI ablations,
  behavioral tests, ongoing negative-result log.

Interleaved operator/unrelated commits and untracked files were preserved. No
remote state changed. Runtime outputs remain under `work/auto-sfx/`; no media
or draft operator-validation sheets were staged as source changes.

## 9. Recommended architecture and next decision

Keep this layered architecture:

`pose observations → continuity tracks → per-foot physical state/evidence →
raw IC + uncertainty/abstention → optional pixel refiner → bounded decoder`

with three hard boundaries:

- temporal track IDs are not anatomical side labels;
- a refiner may use corrected pose for identity/ROI/coarse windows but camera
  motion is never contact evidence;
- decoded events must remain separate from raw physical evidence and may not
  fabricate unsupported contacts.

For the repository now:

- keep `hybrid-v0` as default for compatibility;
- retain `contact-state-v1` and `contact-roi-v1` as opt-in experimental methods;
- do not add a decoder yet;
- do not claim validation from agent-draft truth;
- obtain operator-verified frame windows before any product claim.

The next credible accuracy branch is not another threshold or smoother. It is
an input upgrade: a pose/backend with calibrated anatomical identity and
better foot landmarks across edited frontal/rear shots, or a learned
foot-pixel change-point model trained/validated on licensed footage with
operator IC truth. With the current MediaPipe/RTMPose traces and the measured
OpenCV ROI evidence, further local parameter tuning would be validation
leakage, not engineering progress.
