# Auto-SFX POC — Environment & Checkpoint 1/2 Report

Date: 2026-08-25 · Branch: `main` (local only, nothing pushed) · Mission: docs/auto-sft-poc.md

## Checkpoint 1 — Docker POC builds, Pose Landmarker runs on video

### Environment (measured inside `pab-auto-sfx:latest`)

| Item | Value |
|---|---|
| Host | 2018 Intel Mac mini, i7-8700B (6c/12t), 64 GiB RAM, macOS, OrbStack Docker |
| Docker VM | ~16.8 GiB RAM visible, 12 CPUs |
| Image | `pab-auto-sfx:latest`, **2.10 GB** (`SIZE=2100896215`) |
| Build time | 13 s warm (all layers cached incl. pip); cold build not re-measured this session |
| Base | `python:3.11-slim-bookworm` + ffmpeg (Debian 5.1.9), libgl1, libglib2.0 |
| Python | 3.11.16 |
| Key deps | numpy 1.26.4 · opencv-contrib 4.10.0.84 · mediapipe 0.10.21 · onnxruntime 1.19.2 · rtmlib 0.0.x · pyav 14.2.0 (added this session) · pytest 8.3.3 |
| Threads | `OMP_NUM_THREADS=6`, `ORT_NUM_THREADS=6`, `AUTO_SFX_THREADS=6` (compose; avoids 12-way oversubscription) |
| Execution providers | mediapipe: XNNPACK CPU delegate (GPU/EGL unavailable as expected); ORT: CPU |
| Models | `pose_landmarker_lite.task` sha256 `59929e1d…d574a` (5,642 kB), `pose_landmarker_full.task` sha256 `4eaa5eb7…c4ad` (9,177 kB), cached at `/cache/auto-sfx/models` |
| RTMPose models | yolox_tiny (18 MB) + rtmpose-s halpe26 (20.4 MB) from download.openmmlab.com, cached under `/root/.cache/rtmlib` in-container. Notice printed and rendered on every artifact: `RESEARCH POC ONLY - PRODUCTION LICENSE NOT CLEARED` |

### Packaging problems found & fixed this session

1. `mediapipe_backend.run_pose` referenced undefined `bgr` after the PTS-injectable-frames change (Contract B) — crashed on first real inference. Fixed.
2. `FrameReader` ffmpeg pipe lacked `-fps_mode passthrough` → **VFR inputs were silently re-timed to CFR with duplicated frames** (360-frame VFR file decoded as >360 raw frames, positional PTS pairing desynced). Fixed; VFR fixture now decodes exactly 360 frames with true PTS.
3. `FrameReader` yielded **read-only** numpy frames (`np.frombuffer` view) → renderer's first in-place draw raised `cv2.error … readonly`. Fixed with a single copy per frame (~1 MiB @ 480p, negligible vs pose cost).
4. Debian ffmpeg 5.1.9 ignores concat-demuxer `duration` for bare stills AND evaluates `setpts=N*TB` as constant zero → VFR fixture was silently flat CFR 25 fps. Fixture now written via PyAV with explicit per-frame PTS (time base 1/90000); decoded PTS match the analytic timeline to 3.3e-7 s.

## Checkpoint 2 — First foot-contact predictions + annotated preview

### What exists now

- CLI: `experiments/auto-sfx/run analyze|compare|metrics|annotate|selftest` (entrypoint `python -m auto_sfx`).
- Outputs per run under `/work/auto-sfx/runs/<run-id>/`: `events.json`, `events.csv`, `pose.jsonl`, `diagnostics.json`, `resource-report.json`, `annotated.mp4`, `marker-preview.mp4` (+`markers.wav`), `compare-summary.json` for compare.
- Pose cache: content-hash keyed (file sha256 + stream + backend + model sha + mode + preprocess version); verified HIT on second run (pose stage 0.03 s vs ~10 s MISS).
- Contact methods: `velocity` (O'Connor-style), `pelvis` (Zeni-style, principal-axis projection), `hybrid` (velocity state machine + support-band envelope), offline state machine, abstention-first diagnostics (`pose_lost`, `feet_out_of_frame`, `heel/toe_low_confidence`, `cadence_violation`, `camera_motion_high`, `shot_boundary`, …).
- Tests: **155 passed, 0 failed** inside the container (unit + integration incl. fake-backend end-to-end, cache, VFR fidelity, renderer, RTM mapping).

### Measured on the synthetic fixture (12 s, 720×480@30, silhouette walker, exact analytic GT strikes at t=k·0.6 L / +0.3 R, 41 strikes)

| Config | pose found | events | matched | FP | median err (matched) | within ±1 frame | RTF | peak RSS |
|---|---|---|---|---|---|---|---|---|
| mediapipe-lite / video / hybrid | 104/360 frames, stretches | 15 | 6 | 8 | 33 ms | 100% | 1.47× | 260 MB |
| mediapipe-lite / video / hybrid (VFR file) | stretches | 17 | — | — | — | — | 0.87× | 381 MB |
| mediapipe-full / video / hybrid | sparser (5 lost intervals) | 2 | 1 | 1 | 33 ms | 100% | 0.92× | 262 MB |
| mediapipe-lite / **image** mode | **0 frames — pose lost entire clip** | 0 | 0 | 0 | — | — | 0.88× | 236 MB |
| rtmpose-lightweight / video | pose present whole clip, score ~0.34; heel vis mean 0.44 | 0 (abstains: feet unreliable) | 0 | 0 | — | — | **4.9×** (6.1 fps) | 215 MB |
| compare (one cached trace, 3 methods) | velocity 15 / pelvis 10 / hybrid 15 events | — | — | — | — | — | cache HIT | — |

### Honest failure analysis (synthetic fixture)

1. **Coverage, not timing, is the failure.** Whenever BlazePose (video mode) holds the figure, matched contacts land within ±1 displayed frame. But the detector only holds the stylized silhouette intermittently (~29% of frames), so most strikes are never scored. Image mode (stateless) finds **nothing** — tracking is doing all the work.
2. **Side assignment is unreliable** (5 wrong-side of 15) when the silhouette's legs cross — consistent with pose jitter, not the contact logic.
3. **RTMPose keeps the person but not the feet** on this figure (heel/toe visibility ≈0.44 mean): its YOLOX detector is stronger, its foot landmarks weaker here. At 6.1 fps CPU it is 3× slower than MediaPipe lite.
4. The fixture itself is the confound: a flat-shaded silhouette is out-of-distribution for both pose models. These numbers measure "pose models on a mannequin", not "the pipeline on real footage". Per the mission's truth-seeking rule they are reported raw, not tuned away.

### Sample artifacts (local, untracked)

- `work/auto-sfx/runs/selftest-cfr-20260825T050105Z/` — annotated.mp4, marker-preview.mp4 (12.0 s stereo click track, L/R distinguishable), events/pose/diagnostics/resource JSON+CSV.
- `work/auto-sfx/runs/rtm-video2/` — RTMPose annotated render.
- Visual inspection performed on extracted annotated frames: skeleton, HUD (backend/mode/t/frame/score), per-foot state chips, heel-speed scrolling graph, contact banners, ASCII-safe license banner all render correctly.

## Next-step recommendation

Get real dogfood footage before any further algorithm work: the pipeline, caching, VFR handling, diagnostics and evaluation loop are ready and the synthetic-fixture ceiling is understood.
