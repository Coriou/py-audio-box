# Auto-SFX Phase 2B — ongoing experiment log

Date: 2026-08-25. Local-only, Docker/CPU. Ground-truth provenance and the
development/frozen split are recorded before any GT-guided tuning below.

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
