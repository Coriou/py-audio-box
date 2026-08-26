# Auto-SFX Phase 2B — ongoing experiment log

Date: 2026-08-25. Local-only, Docker/CPU. Ground-truth provenance and the
development/frozen split are recorded before any GT-guided tuning below.

Frozen split: `experiments/auto-sfx/gt/phase2b-split.json`. It was written
from raw-video temporal blocks and shot/viewpoint ranges while both blind
contact tables were still unavailable. No prediction, trace, or event time
was used to allocate a range. Validation ranges remain frozen below.

## Pre-GT invariants and rejected mechanisms

| Run / check | Change | Evidence | Decision |
|---|---|---|---|
| `phase2b-baseline-preflight` | Unchanged local HEAD `320d450` | Docker suite: 272 passed. Cached `gate2-walk1-smoke` remains the unchanged 25-event `hybrid-v0` reference. | Keep as baseline. |
| `phase2b-track-greedy-cost` | Assign temporal feet to the lower-cost permutation whenever it wins by a margin. | Cached walk1 audit: MediaPipe video 96 swapped frames; image mode 193; RTMPose 24. The tracker latched onto ordinary crossing trajectories. | **Rejected.** Lower assignment cost alone is not evidence of a backend label exchange. |
| `phase2b-track-bound-8` / `-10` | Require direct assignment to violate an 8 or 10 body-scale/s physical bound and crossed assignment to remain below it. | MediaPipe video still latched crossed over frames 355–363 and 365–366 around the known frame-364 fault. | **Rejected.** Bound below observed legitimate fast motion. |
| `phase2b-track-bound-12` / `-14` | Same two-hypothesis gate at 12/14 body-scale/s. | Both isolate frame 364 and restore corrected f364 speeds to L 4.974 / R 2.440 body-scale/s (raw L 20.615 / R 18.262). RTMPose remains a zero-swap negative control. Bound 14 is least invasive. | Carry 14 forward; long swap episodes still require abstention. |
| `phase2b-track-bound-14-hybrid-v0` | Feed bound-14 corrected identity to the old lifetime-threshold velocity machine (diagnostic ablation, no GT claim). | Maximum speed 20.61→14.32 and maximum historical threshold 6.53→5.01, but events 25→39, simultaneous pairs 5→6, and longest CANDIDATE still 102 frames. | **Rejected as a detector repair.** Identity correction cannot rescue the threshold lifecycle/state semantics. |

The first implementation checkpoint adds a distinct `contact-state-v1`; it
does not retune or replace `hybrid-v0`. It uses separate heel/toe evidence,
bounded local statistics, retrospective sustained-grounded change points,
internal toe-off transitions, explicit uncertainty, and discontinuity
abstention. Accuracy decisions wait for the independently drafted GT split.

## ROI literature constraint (read before prototype)

Kobayashi & Nakazawa (2025), DOI `10.1299/transjsme.25-00049`, is an
open-access Japanese paper licensed **CC BY-NC-ND 4.0**. It does not validate
our footage or authorize copying a derivative implementation. Its experiment
used a fixed low-mounted Logitech C615 RGB camera (0.5 m high, aimed down), a
2 m camera-to-course-centre distance, 640×360 at 30 fps, a controlled 6 m
walking lane, camera/course angles 90/75/60/45 degrees, normal walking, and 13
students in their twenties (three trials per direction). Ground truth was
manual visual frame selection. A prediction within ±3 frames was counted
correct, and each evaluated clip window contained two heel strikes and two
toe-offs. Reported RMSE was computed only over correct predictions.

Those conditions differ materially from our 1080p/portrait multi-shot stock
footage, moving or edited cameras, arbitrary crops, unknown subjects, varied
pace, and stricter existence/timing reporting. The paper's headline F/RMSE is
therefore **not imported**. The constrained design ideas are: centred
previous/current/next optical flow; dynamic edge evidence; radial acceleration
near the forefoot for heel strike; persistence/disappearance of static edge
regions for toe-off; and a coarse gait geometry window. Our prototype must add
explicit local/global motion compensation and use corrected pose only for
identity/ROI/coarse timing.

## Pre-GT ROI resource smokes

| Run | Implementation | Output (accuracy not yet scored) | CPU/resource result | Decision |
|---|---|---|---|---|
| `phase2b-roi-pre-gt` | Retained all 1,067 decoded 1080p BGR frames, then evaluated camera-compensated Farneback evidence in pose windows. | Refined 24/24 pose candidates. | contacts 45.06 s; total 53.94 s; **peak RSS 6,515.7 MB**. | **Rejected resource shape.** |
| `phase2b-roi-streaming` | Stream decode; retain only downsampled grayscale frames needed by candidate windows. Same physical evidence and candidates. | Refined 24/24 pose candidates. | contacts 21.46 s; total 31.11 s (RTF 0.87×); peak RSS **306.4 MB**. | Carry forward for frozen-GT comparison; not accepted on event count. |

## Independent draft truth and frozen split

No operator GT appeared. All marks below are labelled **agent-draft; NOT
operator verified** and are experiment guidance only.

- `walk1`: the first raw-only pass was too sparse to be exhaustive and the
  second raw-only pass mixed initial contact with toe-off/terminal swing. A
  third isolated raw-frame semantic audit adjudicated every proposed row and
  produced the 41-contact replacement table under
  `work/auto-sfx/gt-drafts/recovery/`. In particular, frames 548/598/676/1038
  were toe-off/push-off and 122/200/280 were swing; the replacement IC frames
  are 543/593/673/1033 and 128/207/285. Anatomical side remains `uncertain`
  because it cannot be defended from the raw view.
- `walk2`: independent raw-only passes A and B were compared before any
  prediction was shown. Raw-frame sheets adjudicated disagreements and retain
  ±1/±2-frame uncertainty. The canonical draft contains 54 contacts over the
  useful shot/viewpoint sequences; 20 left, 21 right, 13 uncertain.
- Canonical drafts: `gt/walk1.agent-draft.gt.json` and
  `gt/walk2.agent-draft.gt.json`. The `overhead` observer description is not a
  schema-v2 enum, so the mechanical converter preserves it in notes and maps
  the schema value to `uncertain`; no contact time or side changed.
- Operator-validation sheets:
  `work/auto-sfx/gt-adjudication/walk1_recovery_adjudication_*.jpg` and
  `work/auto-sfx/gt-adjudication/phase2b-walk2/walk2-agent-draft-operator-validation.jpg`.
- `gt/phase2b-split.json` was frozen before either usable mark table existed.
  It alternates temporal blocks in walk1 and shot/viewpoint sequences in
  walk2. The symmetric two-frame boundary guard equals the maximum draft
  uncertainty and prevents events outside one fold from becoming boundary
  false positives in the other; assignments were not changed.

## T8 signal, backend, tracking, and smoothing fork

| Run / check | Result | Decision |
|---|---|---|
| MediaPipe video identity audit | High-confidence L/R exchange at frame 364. Raw one-frame speeds L 20.615 / R 18.262 body-scale/s; continuity correction L 4.974 / R 2.440. The bounded tracker also quarantines the long 623–628 discontinuity rather than interpolating it. | **Accepted mechanism:** temporal track identity is separate from anatomical labels; swap only when direct assignment is physically impossible and crossed assignment is plausible. Interpolated samples remain invalid contact evidence. |
| MediaPipe image, `phase2b-state-mp-image-walk1-dev1` | 37 events, 20 discontinuities; development P/R 0.368/0.350, median/p90 2/5 frames; 32.06 s, 88.2 MB. | **Rejected:** no existence gain and worse tail timing/discontinuity load. |
| RTMPose video, `phase2b-state-rtmpose-walk1-dev1` | 40 events, stable negative-control trace; development P/R 0.263/0.250, median/p90 2/5 frames; 16.36 s, 75.9 MB. Its production license is also not cleared. | **Rejected:** worse measured accuracy; research-only backend. |
| Light robust smoothing, `phase2b-state-mp-video-light-walk1-dev1` | All 40 event times identical to raw. Development P/R 0.368/0.350, median/p90 2/4.2 frames. | **Rejected:** no benefit. |
| Offline zero-phase smoothing, `phase2b-state-mp-video-zero-walk1-dev1` | 18/40 timestamps changed, including large 429→415 and 921→932 shifts. Development P/R fell to 0.316/0.300; median/p90 2/4 frames. | **Rejected:** cleaner-looking motion was not better timing. |

## T5 contact-state iterations (development only)

`contact-state-v1` was rewritten around heel and toe body-relative trajectories:
forward swing, a locally advanced apex, sustained rearward support motion,
bounded local speed/support evidence, separate heel/toe onset windows, and an
internal toe-off. It has only `SWING | UNCERTAIN | GROUNDED`; it cannot enter
the historical `CANDIDATE` state. One-frame gaps may be interpolated for
derivatives, but a candidate needs an unbroken 11-frame evidence window and
cannot use quarantined/interpolated frames.

| Run | Change/result | Development decision |
|---|---|---|
| `phase2b-state-relative-dev1` | Replaced absolute image velocity with pelvis-relative phase. Recovered continuous profile cadence but emitted four short jitter reversals. | Continue; physical signal found, dedup required. |
| `phase2b-state-relative-dev2` | Bounded same-foot local-cycle dedup. 41 events on walk1. | Continue; one contact still bridged the 623–628 pose gap. |
| `phase2b-state-relative-dev3` | Future-evidence gate and bounded one-frame interpolation only. 40 events; explicit abstention over 623–628. | **Locked development candidate.** |
| `phase2b-state-walk2-relative-dev1` | Same locked parameters on all walk2 shots. Development P/R 0.500/0.318; diagnostic median/p90 1/2 frames; >3-frame 9.1%; known-side 0.857; no simultaneous contacts. | Does not meet bar. Continue through ROI before opening validation. |

Corrected walk1 development results on identical shards:

| Method | Precision | Recall | diagnostic median / p90 | >3f | simultaneous L/R |
|---|---:|---:|---:|---:|---:|
| `hybrid-v0` baseline | 0.111 | 0.100 | 5.5 / 9.5 f | 60.0% | 4 |
| locked raw state | 0.368 | 0.350 | 2 / 4.2 f | 21.1% | 0 |
| image state | 0.368 | 0.350 | 2 / 5 f | 31.6% | 0 |
| RTMPose state | 0.263 | 0.250 | 2 / 5 f | 31.6% | 0 |
| zero-phase state | 0.316 | 0.300 | 2 / 4 f | 26.3% | 0 |

## Foot-pixel ROI branch (development only)

The OpenCV prototype uses corrected pose only for identity/ROI/coarse windows.
It streams downsampled grayscale frames; subtracts local ground motion from
foot-edge optical flow; keeps heel and toe regions separate; scores a
moving-to-static change, vertical deceleration, static-pixel proportion, and
edge persistence; and explicitly excludes camera translation from contact
evidence.

| Run | Result | CPU/resource | Decision |
|---|---|---|---|
| `phase2b-roi-walk1-dev1` | P/R 0.368/0.350, median/p90 2/6.2 f, 31.6% >3 f. Shifts ranged −7…+5 frames. | 122.16 s; 364.3 MB. | **Rejected:** no existence gain, worse timing. |
| `phase2b-roi-walk1-dev2` | Edge-ellipse isolation. Same P/R, median/p90 2/7 f, 36.8% >3 f. | 126.79 s; 401.8 MB. | **Rejected:** texture isolation did not localise touchdown. |
| `phase2b-roi-walk1-dev3` | Stricter q90/downward-flow evidence. | Terminated after >9 min at ~353 MB before output. | **Rejected:** incompatible CPU cost. Source restored to the last completed q75 prototype. |

Because both completed ROI variants regressed development, neither was allowed
to see frozen validation. This exhausts the required pixel branch without
turning a failed walk1-only result into a validation claim.

## Candidate lock and frozen validation (opened once)

The lock decision was recorded after the development tables above:
MediaPipe-video + continuity tracking + raw `contact-state-v1`. No offset,
view-specific correction, decoder, or ROI parameter was selected using frozen
results.

| Frozen subset | Method | P | R | median / p90 | >3f | known-side | simultaneous |
|---|---|---:|---:|---:|---:|---:|---:|
| walk1 | baseline | .143 | .048 | 8 / 8.6 f | 80.0% | n/a | 1 |
| walk1 | state | .150 | .143 | 3 / 5 f | 31.6% | n/a | 0 |
| walk2 | baseline | .111 | .067 | 5 / 8 f | 81.3% | .000 | 1 |
| walk2 | state | .462 | .200 | 2 / 7.8 f | 41.7% | .000 | 0 |
| **combined** | **baseline** | **.120** | **.059** | **5 / 8 f** | **81.0%** | **.000 (2/2 flipped)** | **2** |
| **combined** | **state** | **.273** | **.176** | **3 / 6 f** | **35.5%** | **.000 (5/5 flipped)** | **0** |

The state fork improves every combined existence/timing measure and removes
frozen simultaneous pairs, but fails every numeric exit criterion except the
simultaneous-contact check. Walk2 validation also exposes the unresolved
distinction between temporal track identity and anatomical side attribution.
No decoder was added: physical phase is not correct, and a sequence model must
not fabricate or silently delete contacts. The full-clip T3 artifact surfaces
the raw 27.100/27.133 s simultaneous pair and the detector records it as a
contradiction; it is outside the frozen marked ranges and is not hidden.

## Final diagnostic artifacts and resources

- `phase2b-state-final-walk1`: 40 raw contacts; final `annotated.mp4`,
  `marker-preview.mp4`, Metrics-v2 shards, T3, and all forensics windows.
  Cached-pose CPU run: contacts 2.72 s, render 80.58 s, total 94.20 s,
  peak RSS 167.0 MB.
- `phase2b-state-final-walk2`: 35 raw contacts; the same artifact set. Five
  explicit phase-abstention spans, 14 pose-lost and 33 feet-out-of-frame
  diagnostics. Cached-pose CPU run: contacts 25.81 s, render 56.46 s,
  total 94.36 s, peak RSS 188.3 MB.
- Both ran in Docker/Linux CPU-only with six-thread limits. The detector itself
  is compatible with the local system; full video rendering dominates wall
  time. The ROI branch is not compatible enough to justify its accuracy.

Disposition: keep `hybrid-v0` as the default; ship neither experimental method
as validated. Preserve `contact-state-v1` as an opt-in research/diagnostic path
because it fixes the threshold lifecycle, state semantics, swap handling, and
abstention mechanisms even though the current monocular pose/pixel inputs do
not support the requested frame-accurate acceptance bar on this regression
set.
