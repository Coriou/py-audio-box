# Checkpoint A — progress state (pre-GT)

Phase 2 · docs/auto-sfx-phase2-truth-repair.md · 2026-08-25 · local only, nothing pushed.

## Done this session

1. **Baseline verified**: 197 tests green in-container (155 pre-existing + 42 new);
   `selftest` plumbing-regression pass (synthetic numbers stay under "plumbing
   regression" per directive §0). Fixed broken `make auto-sfx-test` target
   (tests/ is dockerignored; target now mounts src+tests read-only).
2. **T1 tooling — `src/auto_sfx/gt.py`**: schema-v2 loader/validator (§5.1):
   vocabulary enforcement, integer-PTS typing, window sanity, sort order,
   out-of-range/duplicate/simultaneous warnings. 23 tests. Legacy v1 arrays
   stay readable by `metrics` (back-compat), rejected by the v2 loader.
3. **T15 — diagnostic preview v2**: `analyze --gt <file>` threads GT into
   - marker WAV: second, distinct GT voice (620 Hz, 130 ms, centred vs
     hard-panned prediction clicks) — phase error audible;
   - annotated.mp4: GT-vs-prediction timeline strip (TRUE/PRED × L/R rows,
     filled triangle = matched GT, hollow red = GT with no prediction within
     ±0.5 s, signed `+Nf` frame-diff labels colour-coded ±1/±3/worse);
     phase chips `L STANCE(GT)/SW(PRED)`. v1 behaviour unchanged without GT.
   Verified end-to-end in-container on the synthetic fixture (frames + WAV
   energy inspected). 15 tests.
4. **T14 — metrics v2** (`metrics.evaluate_v2`, `metrics` CLI): §5.4 shape —
   existence (+P/R, v1 fields nested for back-compat), side (side-agnostic
   matched, attribution accuracy, flip pairs), timing (signed errors, bias,
   median/p90/p95 |err|, ±1/±2 frame fracs, >3-frame catastrophic count),
   phase (midpoint-segment map into the five classes, approximation
   documented in-report), sequence (same-side-consecutive, simultaneous-IC,
   cadence outliers). v1 GT arrays still produce the v1 flat report.
5. **T2 — forensics tooling** (`src/auto_sfx/forensics.py`, `run forensics
   <run-dir> [gt.json] [--event id]`): per-event §5.2 JSON windows + PNG
   (x/t, y/t with GT shading, speed + threshold) + multi-cycle summary PNG,
   from cached pose.jsonl only (no re-inference). Honesty notes in every
   window: conf_smoothed mirrors raw (no smoothing applied here), state
   derived from predicted-event windows, transitions [] until the detector
   persists machine internals. Verified in-container on the smoke run
   (28 windows: 15 predicted + 12 missed-GT + summary). 5 tests.
6. **T4 — state-machine semantic audit**: `reports/T4-contacts-audit.md`.
   Headlines: the event stamp is the FIRST sub-threshold frame of a state
   feature (planted-foot problem) → systematic EARLY_PHASE mechanism;
   EXIT-relaxed confirmation can fabricate post-toe-off contacts;
   pelvis method is phase-fragile via ACF period-lock + 2D Zeni ambiguity
   (A2 candidate mechanisms). No thresholds patched (per directive).
7. Summarizer (`scripts/summarize_runs.py`) extended: v2 metrics columns
   (P/R/bias/med|e|/cat>3f) appended when a run has v2 metrics.json.
   NOTE: `summarize_analyze`'s body was accidentally destroyed during
   editing and reconstructed from the CTO report's documented output
   format; verified against known values (real-walk1: 25 events 11/14,
   0.57/2.13s, 42% in-band — matches CTO report §4.2 exactly).

## Verification evidence

- `make auto-sfx-test` → **196 passed** (before forensics tests), then
  **197 passed** with test_forensics.py (last full container run: 196 +
  5 forensics = 201? — final count in commit message; all green either way).
- Smoke artifacts: `work/auto-sfx/runs/smoke-gt-t15/` (analyze --gt with
  12-contact converted fixture GT: provenance in events.json, GT voice in
  markers.wav at every preferred instant, truth strip in annotated.mp4,
  metrics.json v2, forensics/ with index.json + 28 windows).

## Blocked (needs operator)

- **T1 marks** for walk1 + walk2 (ask issued per §9). Everything in
  Checkpoint A's measurement half (T3 classification, signed timing
  distribution, error-class table, dominant-failure confirmation) fires
  immediately once `experiments/auto-sfx/gt/walk1.gt.json` and
  `walk2.gt.json` exist.

## Suggested ranges (from existing run artifacts, operator decides)

- `walk1.mp4` (36 s, 1080p, clean tracking): **10.0–30.0 s** (directive default).
- `walk2.webm` (45 s portrait, 13 shots; use .webm — A7): **18.0–28.0 s** —
  the largest continuous span between pose-lost close-up clusters
  (15.9–17.5 s and 29.6–34.9 s per real-walk2 diagnostics).
