# Auto-SFX Phase 2C Batch 1 — identity/sparse negative closeout

Date: 2026-08-27

Scope: opt-in full-leg temporal identity/anatomical attribution and a new
sparse stabilized heel/toe pixel mechanism on opened development diagnostics.
No operator truth, new pose inference, future holdout, global timing offset,
dense-flow rerun, default change, remote mutation, or accuracy claim.

## Outcome first

Both mechanisms are rejected for detector integration in their present form.

- The full-leg prototype repairs the known opened `walk1` frame-364 mixed-chain
  fault and passes its synthetic/PTS/determinism/resource checks, but its own
  audit marks 1,172 of 2,416 opened frames unsafe and rejects a general
  whole-chain claim on both traces.
- The sparse pixel prototype passes deterministic camera/edit/texture/VFR
  counterfactuals and exact decoded-PTS checks, but produces zero evidence-
  bearing suggestions on 19 opened `walk1` development candidates. Eighteen
  abstain for weak adjacent-ground tracks and one for weak foot tracks.

`hybrid-v0` remains the default. `contact-state-v1` remains the opt-in research
candidate. The established bounded foot-centre tracker remains in that
candidate; neither Batch 1 prototype is wired into the detector or CLI.

## Batch question and decisions

Do full-leg-chain temporal assignment and sparse stabilized foot/ground pixels
pass their mechanism counterfactuals and provide usable opened-development
evidence without global retuning?

| Experiment | Mechanical result | Decision |
|---|---|---|
| `phase2c-b1-identity-paired-opened-v2-a1` | f364 repaired; neighbours, PTS, deterministic repeat, wall and memory pass; 1,172/2,416 unsafe frames and two rejected whole-chain claims | **Rejected** for general identity/anatomy integration |
| `phase2c-b1-sparse-paired-walk1-opened-v1-a1` | Exact PTS, explicit disposition, wall and memory pass; 0/19 suggestions, coverage 0% | **Rejected** for pixel refinement |

This is a substantial negative mechanism marker, not an accuracy marker. The
operator-gold challenge gate remains entirely unmeasured.

## Full-leg identity evidence

The first positional full-chain Viterbi passed synthetic label swaps but left
the real opened frame 364 `direct`; it was not accepted. Inspection showed
that first-order position/shape cost lacked the bounded velocity innovation
which isolated the original Phase 2B fault. The repaired opt-in prototype uses
separate proximal hip/knee and distal ankle/heel/toe second-order Viterbi
decoders, bounded constant-velocity innovation, explicit per-joint source
labels, and component disagreement/quarantine.

On the already-opened frame 364, proximal evidence favors direct assignment
with cost 0.025642 versus 0.411105, while distal evidence favors crossed
assignment with cost 0.015744 versus 0.214165. The output is therefore
`distal_swapped` / `incoherent_mixed`, uses left-labelled proximal and
right-labelled distal observations for temporal track 0, and is explicitly
ineligible as contact evidence. Frames 363 and 365 remain direct. Because this
example drove the repair, it is a regression fixture, never independent
validation.

The negative coverage result dominates that correction:

| Opened trace | Frames | Accepted reassigned frames | Unsafe whole-chain frames | Candidate whole-chain claim |
|---|---:|---:|---:|---|
| `walk1` | 1,067 | 1 distal | 633, including 632 distal quarantines | rejected |
| `walk2` | 1,349 | 22 distal | 539 | rejected |
| Combined | 2,416 | 23 distal | 1,172 (48.51%) | rejected on both traces |

The paired run took 2.203630 seconds for the candidate versus 0.978443 seconds
for the current tracker, with an internally measured 88.043 MiB peak RSS.
There were zero known-frame, neighbour, source-PTS, or deterministic-repeat
failures. Those passes do not override the unsafe-coverage failure. Anatomical
posterior output remains uncalibrated and cannot establish known-side accuracy.

## Sparse pixel evidence

The new primitive is independent of rejected `contact-roi-v1`. It uses sparse
pyramidal LK tracks only, robust affine background stabilization outside a
mandatory actor exclusion, pose-guided heel/toe and adjacent-ground regions,
and relative deceleration/staticness evidence. It fails closed on weak track
support, invalid geometry/cropping, edits, timestamps, or decoder identity.
Camera motion is never contact evidence.

An initial prediction-free host smoke proved that OpenCV frame seeking could
not establish presentation-frame identity: all candidates abstained before
scoring. That non-manifest smoke was discarded rather than weakening the
check. The final video path uses PyAV source PTS for every decoded frame and
requires exact equality with the ffprobe/pose-trace mapping before pixel
analysis.

On 19 coarse `contact-state-v1` candidates wholly inside the opened `walk1`
development ranges:

- suggestions: 0/19 (0% coverage; required ≥25%);
- abstentions: 18 `weak_ground_tracks`, 1 `weak_foot_tracks`;
- source-PTS membership failures: 0;
- candidate wall: 15.090877 seconds (budget 35.57 seconds);
- internally measured peak RSS: 199.211 MiB (budget 320 MiB).

The cached rejected dense-ROI output was used only for a shift/resource
reference; dense flow was not rerun. No distance to agent-draft marks was
computed. The `walk2` VFR real-pixel stress run was not opened after `walk1`
failed the registered coverage gate; raw-array VFR timing remains covered by a
behavioral counterfactual.

## Run control and reproducibility

Both attempts completed under the atomic runner and were explicitly decided
`rejected`:

- identity attempt `7288c198-6544-404b-8bf3-b8949d57a26e`, artifact commitment
  `4374beda9a96489461bf6a76544feaed1cf4cfa34122f1f1bdecd385a985f75f`;
- sparse attempt `4970cce4-71d9-482a-a925-9c6326250981`, artifact commitment
  `d73e1ebc562a6d2df1d83bfdc3d31231d2862e8d7048aceea00989f97044d60b`.

The compact public report is `phase2c-batch1-run-report.json`; it contains no
input paths, split contents, argv, or private metrics. Each scientific command
used a pinned read-only Docker image, no network, exact read-only input mounts,
a dedicated writable output directory, six CPUs, a 1 GiB memory cap, and
declared thread variables. No new model or license-sensitive checkpoint was
used.

## Verification and resources

- Focused host integration: 38 passed.
- Focused Docker/Linux integration: 38 passed.
- Full rebuilt Docker/Linux reference suite: 370 passed in 14.80 seconds.
- Final image: `sha256:1df6b36c57fbbe82a4402700c7db65175f1d5be5ddd0c03e8bf87e5aeb3ef874`,
  2,182,408,336 bytes.
- No Auto-SFX process remained after the run.

## Decision boundary and continuation

Do not wire either prototype into `contact-state-v1`, make it a default, or
retune its thresholds on the opened material. A new identity attempt needs a
new mechanism that resolves component observability without long-run latching;
a new classical pixel attempt needs a materially different ground/foot support
mechanism, not a lower minimum-track threshold.

The next safe development branch is the queued `phase2c-state-groundplane`:
shot-local support geometry and retrospective bounded-duration physical state,
with no registered global offset. Operator work remains the only route to the
next scientific accuracy gate: rights-cleared source reservation, two immutable
prediction-blind passes, and append-only adjudication. A learned foot-patch
classifier remains blocked until rights-cleared operator-labelled development
windows exist.

Exact verification command:

```bash
make auto-sfx-test
```
