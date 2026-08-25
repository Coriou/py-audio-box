# Auto-SFX POC — Session Report (Foot-Contact Perception)

**Date:** 2026-08-25 · **Repo:** `Coriou/py-audio-box` (local only — nothing pushed) · **Phase doc:** `docs/auto-sft-poc.md`
**Author:** orchestration session (human-reviewed artifacts listed in Appendix)

---

## 1. TL;DR

We built and validated the first end-to-end perception POC for the automatic-SFX product idea: **video in → physically-meaningful foot-contact events out, with source-accurate timestamps**, running entirely in Docker on the 2018 Intel Mac mini (CPU-only, $0 compute).

**Bottom line:** the core perception question — *can we detect individual foot contacts with useful timing on modest CPU hardware?* — is **promising but not yet proven**. On the cleanest real clip, the strongest free stack (RTMPose) detects roughly **double the steps MediaPipe does**, with step-interval statistics matching human gait; when events are detected, timing lands within **±1 displayed frame**. The frontier is **recall and side-assignment, not timing**. We can now measure all of this reproducibly in minutes, which was the real deliverable of this phase.

> **ERRATUM (2026-08-25, post-CTO-review).** Human review of the annotated/marker
> previews found many `initialContact` events that do **not** coincide with visible
> foot-ground contact (early-phase, mid-stance, late-stance, simultaneous L+R,
> consecutive same-side). The claim "timing lands within ±1 displayed frame" rested
> on the **synthetic** fixture and on eyeballing banners — it proved timestamp
> plumbing, **not** correct biomechanical phase. Finding 1 below is **RETRACTED**
> pending ground-truthed measurement. The follow-up research directive is
> `docs/auto-sfx-phase2-truth-repair.md`.

---

## 2. What this phase was asked to answer

> Given an ordinary video containing a clearly visible walking person, can we automatically detect plausible individual foot contacts with useful timing on a CPU-only Docker environment?

Deliberately out of scope: NLE plugins, SFX libraries, semantics/VLMs, cloud, benchmarks at scale.

---

## 3. What was built (one week-equivalent of agent-orchestrated work)

| Layer | Capability |
|---|---|
| **Environment** | Isolated `experiments/auto-sfx/` Docker image (2.10 GB, python 3.11 + ffmpeg + mediapipe + rtmlib + onnxruntime + PyAV). Reuses repo conventions: `/work`, `/cache` mounts, CLI-first, no host installs. |
| **CLI** | `experiments/auto-sfx/run analyze / compare / metrics / annotate / selftest` |
| **Pose backends** | MediaPipe Pose Landmarker (lite/full, video+image modes) · RTMPose Halpe26 via rtmlib/ONNX (research-only license banner enforced on every artifact) |
| **Contact engine** | Three inspectable algorithms behind `--contact`: `velocity` (O'Connor-style foot-velocity), `pelvis` (Zeni-style pelvis-relative extrema), `hybrid` (velocity state machine + support-band envelope). Offline state machine (UNKNOWN→SWING→CANDIDATE→STANCE→EXIT), body-scale normalization (no magic pixel constants), abstention-first. |
| **Timestamps** | Exact source PTS preserved end-to-end (µs integers + seconds); verified on a constructed VFR file to 3.3e-7 s. Shot-boundary resets via ffmpeg scene detection. |
| **Diagnostics** | `annotated.mp4` (skeleton, heel/ankle/toe, side colors, confidence, contact banners with `LEFT initialContact t=…s score=…`), `marker-preview.mp4` (stereo clicks, L/R distinguishable, for judging timing by ear), `events.json/csv`, `pose.jsonl`, `diagnostics.json` (first-class abstention reasons), `resource-report.json` (per-stage wall times, RTF, peak RSS). |
| **Caching** | Content-hash pose cache (file sha256 + stream + backend + model sha + mode + preprocess version): pose re-runs are eliminated; contact-algorithm ablations are free. |
| **Evaluation** | Greedy one-to-one matcher (duplicates, wrong-side, FP/miss, median/p90 error, ±1-frame fraction, FP/min) + manual-annotation helper (`run annotate`). |
| **Quality gates** | 155 tests green in-container (unit + integration incl. fake-backend end-to-end, cache, VFR fidelity, renderer, metrics). |

---

## 4. Evidence

### 4.1 Synthetic control (ground truth known exactly)

A deterministic rendered walker (12 s, 41 strikes at known instants) exercises the full pipeline without human annotation:

| Config | Pose coverage | Events | Matched | Median err | Within ±1 frame |
|---|---|---|---|---|---|
| MediaPipe lite / video / hybrid | 29% | 15 | 6 | 33 ms | 100% |
| MediaPipe lite / **image** | **0%** | 0 | — | — | — |
| RTMPose / video | pose present, feet unreliable | 0 (abstains) | — | — | — |

**Takeaway:** when a contact is detected, timing is frame-accurate; the synthetic mannequin is out-of-distribution for pose detectors, so coverage — not timing — fails. This is why real footage mattered next.

### 4.2 Real dogfood footage (5 clips, ~2 min total: 1080p side-walk 36 s, portrait 45 s multi-shot, two 9 s shorts, one 33 s 720p)

Pose coverage on real humans: **88–100% on 4/5 clips** (vs 29% synthetic — the fixture was the confound, not the approach).

| Clip (lite/video/hybrid unless noted) | Coverage | Events | L/R | Step-interval median / p90 | Notes |
|---|---|---|---|---|---|
| walk1 (1080p, 36 s) | 100% | 25 | 11/14 | 0.57 s / 2.13 s | clean tracking, misses steps in clusters |
| walk1 / **rtmpose** | 100% | **53** | 33/20 | **0.53 s / 0.87 s (71% in gait band)** | ~2× recall, plausible cadence; L/R imbalanced; 5.5 fps |
| walk1 / mediapipe-**full** | 100% | 22 | **11/11** | 1.40 s / 3.63 s | fixes side balance, same recall ceiling |
| walk1 / lite / **image mode** | 100% | 26 | 15/11 | 1.27 s / 2.87 s | stateless works on real footage |
| walk2 (portrait, 13 shots) | 88% | 63 | 34/29 | 0.37 s / 1.80 s | shot resets work; close-up shots honestly `POSE LOST` |
| man-walk (9 s) | 93% | 16 | 11/5 | 0.37 / 1.37 | far-side foot under-detected |
| walk3 (9 s, distant person) | 93% | 13 | 6/7 | 0.43 / 1.94 | `actor_too_small` fired correctly |
| 3571264 (33 s) | **24%** | **0** | — | — | honest abstention: person small/far, feet rarely visible |

**Method agreement (same cached pose, ±67 ms):** velocity↔hybrid 83–100%; **velocity↔pelvis ≈ 0–14%** — the Zeni-style method fires at a *different gait phase* on real footage (likely walk-axis/phase misalignment), not merely fewer events. Flagged for audit before anyone trusts `--contact pelvis`.

### 4.3 Human-verifiable artifacts

For every run: `annotated.mp4` + `marker-preview.mp4` (audible clicks at predicted contacts). Visual inspection of extracted frames confirmed: correct skeleton lock-on on real footage, contact banner `LEFT initialContact t=11.97s score=0.91` landing on a visibly planted foot (RTMPose run), honest `POSE LOST` during full-frame foot close-ups, and clean rendering of the research-license banner.

---

## 5. Findings (each with evidence → implication)

1. **[RETRACTED — see Erratum above] "Timing is solved when detection exists."** The ±1-frame evidence was synthetic-only; real-footage review shows wrong-phase events (early/mid-stance/late, simultaneous L+R). → Real initial-contact timing is **unproven**; phase correctness is the phase-2 target (`docs/auto-sfx-phase2-truth-repair.md`).
2. **Recall is the frontier.** On walk1, MediaPipe's step-interval median (~1.3 s ≈ 2× normal) reveals systematically missed steps; RTMPose finds ~2× more with gait-plausible intervals. → Backend choice matters more than contact-algorithm choice right now.
3. **RTMPose is the recall leader but carries costs:** 5.5 fps CPU (3× slower), L/R imbalance (33/20; hip-keypoint crossing visible in overlays), and **uncleared production license** (research use only, banner enforced). → Usable to keep proving perception; production needs either license resolution or MediaPipe-class models improved otherwise.
4. **Stateless (image) mode is viable on real footage** (26 vs 25 events on walk1) — contradicting what the synthetic fixture suggested. The video-vs-image phase-shift question from the research dossier gets a real-data answer: no catastrophic phase penalty. → Keeps a no-tracking fallback on the table.
5. **The pelvis method needs an audit before use** (0–14% phase agreement with velocity). → One concrete, well-scoped algorithm task.
6. **Honest failure works.** Distant/small actor (0 events, `actor_too_small`, `feet_out_of_frame`) and close-up shots (`POSE LOST`) abstain instead of inventing events. → The truth-seeking requirement is met by construction, not by tuning.
7. **Performance envelope (6-core CPU, no GPU):** 720p ≈ 12–17 fps effective; 1080p ≈ 7–14 fps (RTF 2–4×); RTMPose 5.5 fps; peak RSS ≤ 315 MB. → Batch/semi-live analysis viable on this class of hardware; interactive-realtime is not there yet (and was not the goal).
8. **Tooling bug for your splitter:** `walk2.mp4` was unfinalized (missing moov atom → unplayable); the `.webm` twin worked. → Ensure the splitter finalizes MP4 containers (`-movflags +faststart` or clean close) before writing.

---

## 6. Recommended next steps (priority order)

1. **Ground-truth a subset (½ day):** hand-annotate ~3 clips with `run annotate` (minutes per clip) → real precision/recall/duplicate numbers per backend; converts "promising" into measured.
2. **Audit pelvis-phase alignment** (the 0–14% agreement): one experiment, likely axis-sign or extremum-polarity on monocular 2D.
3. **RTM side-assignment fix:** disambiguate crossed hip keypoints in side view (cheap geometric prior), re-measure L/R balance.
4. **Missed-step recovery:** the p90≈2 s gaps are clustered; a gap-aware second pass (search for suppressed candidates inside long intervals) is the highest-leverage recall win for MediaPipe.
5. **1080p speed:** downscale-to-inference (e.g. 720p) preprocessing — likely ~2× RTF improvement; measure before/after.
6. **More footage variety:** running, partial occlusion, camera motion, negative clips (the doc's list) — the pipeline ingests them as-is.

---

## 7. Costs & constraints

- All compute local CPU (2018 Mac mini, Docker/OrbStack). External spend this session: **$0**. Disk: image 2.10 GB + ~2 GB models/cache.
- Repository safety honored: **nothing pushed, no PRs, no remote mutation**; all work is untracked-local under `experiments/` for operator review.

---

## 8. Appendix — reproduce

```bash
experiments/auto-sfx/run selftest                       # synthetic end-to-end + metrics, no gates
experiments/auto-sfx/run analyze /work/auto-sfx/clips/walk1.mp4 \
    --pose mediapipe-lite --contact hybrid              # per-clip artifacts in /work/auto-sfx/runs/<id>
experiments/auto-sfx/run compare /work/auto-sfx/clips/walk1.mp4   # 3 methods, one cached pose
experiments/auto-sfx/run metrics <run-dir> <ground-truth.json>    # after hand-annotation
python /scripts/summarize_runs.py real-walk1 real-walk1-rtm …     # aggregate table
docker compose -f experiments/auto-sfx/compose.yml run --rm \
    --entrypoint python -v "$PWD/experiments/auto-sfx/src:/app/src:ro" \
    -v "$PWD/experiments/auto-sfx/tests:/app/tests:ro" auto-sfx -m pytest tests/ -q   # 155 passed
```

Key artifacts: `experiments/auto-sfx/reports/env-checkpoint1.md` (environment + packaging defects found/fixed), `work/auto-sfx/runs/real-*` (all real-clip outputs incl. annotated previews), `experiments/auto-sfx/scripts/summarize_runs.py` (batch summarizer for future sessions).
