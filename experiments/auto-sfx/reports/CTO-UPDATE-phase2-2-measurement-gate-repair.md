# Auto-SFX Phase 2 — CTO Update #2 (measurement-gate repair)

**Date:** 2026-08-25 · **Repo:** `Coriou/py-audio-box`, local commits `20c59a8…8392bef` on top of `264b358` — nothing pushed (directive §0 honored)
**Directive:** `docs/auto-sfx-phase2-truth-repair.md` · **Predecessor:** `CTO-UPDATE-phase2-1-tooling-audit.md`
**Scope:** repair association + Metrics v2, complete behavior-neutral detector tracing (T2), honest T3 classification. **No algorithm changes, no threshold retuning.**

---

## 1. TL;DR

The Checkpoint-A gate could not have measured the known 2–6-frame wrong-phase failures: the
metrics v1 tolerance (~66 ms ≈ 2 frames @30 fps) governed BOTH existence and timing, so a
+4-frame prediction was scored miss/FP **and vanished from every timing statistic** —
`catastrophic_gt3frame` was structurally unable to fire on exactly the failure band this phase
exists to measure. The gate is now repaired: strict existence semantics are unchanged, while a
wider **diagnostic** association (monotone, one-to-one, 0.5 s horizon) feeds signed timing,
phase diagnosis, forensics, preview labels and T3. The four-frame reproducer demonstrates the
repair end-to-end (§5). Detector internals are persisted per frame (`contact-trace.jsonl`) with
a proven behavior-neutrality regression; forensics now shows **exact** machine states,
transitions and active thresholds instead of reconstructions; T3 assigns only mechanically
defensible classes and says `UNKNOWN` + evidence otherwise.

**Gate status: BLOCKED on operator ground-truth marks.** No real operator GT exists in
`work/auto-sfx/gt/` (only the two synthetic fixture conversions). Checkpoint A cannot be
produced without them, and per directive nothing was fabricated. The verbatim ASK is in §8.

---

## 2. Discrepancies found in the prior dossier (update #1)

| # | Claim in update #1 | Reality found |
|---|---|---|
| D1 | "T14 metrics v2 … >3-frame catastrophic count" | `catastrophic_gt3frame` could never exceed 0 on real wrong-phase errors: timing was computed over strict matches only (tol ≈ 2 frames). The 2–6-frame band the T4 audit predicts was invisible by construction. |
| D2 | "T15 strip … signed `+Nf` labels" | Labels came from a **private nearest-match** (`render.nearest_gt_diff_frames`): not one-to-one (one GT could "match" several predictions), not monotone, side-filtered — a prediction between two GTs could be labelled against either. |
| D3 | "T2 forensics … state derived from event windows … `transitions: []`" | Confirmed and worse: states were `PRED_WINDOW` strings derived from the event's ±1-frame uncertainty window — i.e. the artifact **assumed the conclusion** (event ⇒ stance). Thresholds shown were defaults with a note, not the run's active config. |
| D4 | "load_gt_any (v2 windows survive)" | Only `preferred/earliest/latest_s` survived. `visibility`, `viewpoint`, source PTS and notes were silently dropped — yet the directive requires per-view breakdowns (T11) and the ASK asks operators for visibility/viewpoint. |
| D5 | "phase (midpoint-segment map)" | Phase was computed against the **global** timeline mixing both feet: alternating L/R contacts were treated as one stride, so a left prediction halfway between left marks was classified against a right mark 0.3 s away. |
| D6 | Preview predicted chips | Derived from `e.earliest_s <= t <= e.latest_s` — the event's one-frame window, not detector state. |
| D7 | `uncertain` foot | `metrics._normalise_annotation` **rejected** `foot:"uncertain"` with a ValueError — the directive's own vocabulary was unloadable through the metrics path. |

---

## 3. Exact fixes (all inside `experiments/auto-sfx/`)

### Phase 0 — shared association + Metrics v2 (commits `20c59a8`)

- **NEW `src/auto_sfx/association.py`** — the one matcher, two semantics:
  - `associate_strict(preds, gts, tol_s, fps)` — the v1 greedy nearest-first algorithm,
    verbatim; drives precision/recall/duplicates/misses only.
  - `associate_diagnostic(preds, gts, horizon_s=0.5, fps)` — **monotone, strictly
    one-to-one in both directions, local horizon ≤ 0.5 s**, computed as a dynamic-programming
    alignment that maximises pair count (tie-break: smaller total |dt|). A prediction can never
    match two GT contacts; a GT can never absorb two predictions; order is preserved.
  - Side never blocks a match; each pair reports `side_ok` (`None` when either side is
    unknown/`uncertain`), `err_frames`, and `inside_gt_window` (earliest ≤ pred ≤ latest).
- `metrics.evaluate` (v1) rewired onto `associate_strict` — output schema byte-compatible
  (asserted by tests); `matched_pairs`/`unmatched_*` shapes unchanged.
- `metrics.evaluate_v2`:
  - existence = strict only; **timing/phase = diagnostic** — a +4-frame prediction remains an
    existence miss/FP *and* appears in `timing.diagnostic_pairs` with `err_frames:+4`,
    `inside_gt_window:false`, incrementing `catastrophic_gt3frame`.
  - `load_gt_any` now preserves `gt_id` (`<stem>#NNN`), `visibility`, `viewpoint`,
    `preferred/earliest/latest_pts_us`, `notes`; v1 arrays degrade to point records.
  - `uncertain` foot accepted everywhere (fixes D7); included in existence/timing, excluded
    from attribution denominators.
  - **Per-foot phase**: `classify_phase_for_foot` phases each prediction against its OWN
    foot's GT timeline only (fixes D5); predictions without same-side marks get explicit
    `UNKNOWN_NO_SAME_FOOT_GT` classes instead of invented bands.
  - `phase.per_prediction` rows carry `pred_id, foot, phase_class, same_foot_gt_count,
    assoc_gt_id, assoc_kind (strict|diagnostic|None), err_frames, inside_gt_window, side_ok`.
  - **View-stratified results**: `report["views"][viewpoint]` (n_gt/matched/misses/P/R/
    timing median+bias/`catastrophic_gt3frame`/visibility counts); pairs are attributed to the
    stratum of their GT contact (global association computed once, filtered — no double
    counting); FPs stay top-level (stated in the docstring, not hidden).
  - `report["identification"]` = `{backend, method, displayed_fps, viewpoints}`; the `metrics`
    CLI passes backend/method from `events.json`.
- `cli.evaluate_run_document` threads backend/method through.

### Phase 1 — behavior-neutral tracing + T2 (commits `0d33510`, `140f703`, `789a411`)

- **`contacts.py`**: `detect_contacts(..., record_trace=True)` returns `frame_trace` +
  `trace_header`. Per frame and foot (velocity/hybrid): segment/frame/PTS, state before/after,
  embedded `transition {from,to}`, speed + **velocity components** (vx, vy), active threshold,
  swing peak, hybrid support-band height, foot reference x/y, landmark visibilities,
  `run_start_frame`, `relaxed_run`. Guard decisions (`below_confirm_frames`, `short_duration`,
  `ascent_veto`, `no_pre_run_sample_for_ascent_veto`, `empty_envelope_window`,
  `envelope_height`) and confirmed candidates (with full `evidence` + run provenance) are
  recorded as separate lines. Pelvis method records raw/sway-nulled offset rows and peak
  reject reasons. Recording is append-only; **events and diagnostics are bit-identical with
  tracing on/off — enforced by `tests/test_tracing.py::TestNeutrality`** (hybrid, velocity,
  pelvis, and with config overrides).
- **`cli.py`**: every analyze run now writes `contact-trace.jsonl` (header = exact active
  config incl. shot boundaries + overrides, backend, body scale) and `detector-config.json`
  (`source: "contact-trace.jsonl#header"`).
- **`forensics.py`** (rework): consumes the persisted trace. With a trace: `win["trace"] =
  {"source":"contact-trace.jsonl","exact":true}`, real per-frame machine states, real
  transitions inside the window, thresholds from the header config + per-frame `thr_bs_s`
  arrays, per-side support-band values, detector reference speeds; plot gains machine-state
  ribbons. Without a trace (old runs): reconstruction kept but explicitly flagged
  `exact:false` with a note that states are approximated from event windows and thresholds are
  defaults. Backend-internal unsmoothed landmarks are **not** claimed: the honesty note states
  `conf_smoothed` mirrors raw because no backend-internal smoothing is exposed.
  GT overlays and missed-GT selection now run through `associate_diagnostic`; every predicted
  event and every (diagnostic ⊇ strict) GT miss has a window; `index.json` entries carry
  `gt_id` + `assoc_err_frames`.
- **`render.py`** (preview): truth-strip pairing computed ONCE via
  `strip_pair_labels()` over the shared diagnostic association (filled ▲ ⇔ partner exists);
  side-flipped predictions still get their partner's signed `+Nf` label.
  **Predicted STANCE/SWING chips come from `pred_states`** — the persisted state timeline
  (`states_to_timeline` + `state_at`, 1 s max age) — never from event windows; without a
  trace the chip is `--` (honest unknown; verified visually, §6). `_render_videos` loads
  `contact-trace.jsonl` and threads the timeline through.

### Phase 2 — honest T3 (commit `15a3951`)

- **NEW `src/auto_sfx/classify.py`** + `run classify <run-dir> <gt.json>` (writes
  `classify.json` + `classify.csv`). Table columns: `event_id, gt_id, prediction_time_s,
  gt_preferred_s, error_frames, error_ms, error_class, backend, method, viewpoint,
  visibility, notes`.
- Auto-assigned **only** where arithmetic forces it: `NONE` (inside defensible window),
  `WRONG_SIDE` (opposite-foot partner **inside** its window — good timing, wrong label),
  `EARLY_PHASE` / `LATE_STANCE` / `TOE_OFF_CONFUSED_AS_CONTACT` (same-foot phase map; when the
  side-agnostic partner is the opposite foot, the prediction is re-anchored to its OWN foot's
  nearest mark and judged by signed frames + stride bands — T7 doctrine), `DUPLICATE_CONTACT`
  (unpaired pred whose nearest GT is claimed), `SIMULTANEOUS_LR` (opposite known feet ≤ 1
  displayed frame), `MISSED_CONTACT`, `POSE_LOSS` (window fully inside a recorded POSE_LOST
  diagnostics span), `UNKNOWN` + evidence. `POSE_JITTER / LANDMARK_SWAP / CAMERA_MOTION /
  VIEW_GEOMETRY / CROPPING` are **never** auto-assigned (asserted by test).

### GT protocol decision (per directive)

- `man-walk-no-audio.mp4` is **not** added; it duplicates the distant sagittal `walk3` scene.
- Marking targets: `walk1.mp4` 10–30 s (sagittal) **plus** shot-contiguous frontal/rear/diagonal
  sequences of `walk2.webm` (18–28 s is the largest pose-clean span per `real-walk2`
  diagnostics; extend to adjacent shots where feet are visible).
- **Sufficiency = clear contacts per viewpoint** (enough to give each view class a signed-error
  distribution), not elapsed seconds.
- `walk2.webm` remains canonical through Checkpoint B; the regenerated MP4 is **not** used
  (different duration/timebase would invalidate PTS marks).

---

## 4. Association semantics (normative summary)

| | STRICT | DIAGNOSTIC |
|---|---|---|
| Question | "right contact?" | "what did the detector do near this contact?" |
| Horizon | `max(66 ms, 1.5/fps)` (~2 frames) | ≤ 0.5 s (directive ceiling), configurable |
| One-to-one | yes (greedy, nearest-first) | yes (DP alignment, structural) |
| Monotone | no guarantee | **yes** (structural) |
| Drives | precision/recall/duplicates/misses | signed timing, `catastrophic_gt3frame`, phase, forensics links, strip labels, T3 |
| Existence impact | none of the wider band counts as match | **none** — wide pairs never become matches |

Consumers: `metrics.evaluate`/`evaluate_v2`, `forensics` (overlays + missed-GT selection),
`render.strip_pair_labels`, `classify`. One implementation, no private matchers left.

---

## 5. Four-frame reproducer — before/after

`experiments/auto-sfx/scripts/repro_four_frame.py` (tracked; runnable in-container).
Scenario: one GT contact @10.0 s (window ±1 frame), prediction exactly +4 frames late @30 fps.

**BEFORE (HEAD `264b358` code, in Docker):**
```
existence.matched            0
existence.misses             1
existence.false_positives    1
timing.n                     0
timing.signed_errors_s       []
timing.catastrophic_gt3frame None
phase.per_prediction[0]      pred_id=e1, assoc_gt_id=None, phase_class=None
BEFORE-state reproduced: +4f prediction is an existence miss/FP AND absent from timing
```

**AFTER (this branch, in Docker):**
```
existence.misses             1
existence.false_positives    1          <- strict semantics unchanged
timing.n                     1
timing.signed_errors_s       [0.133333]
timing.catastrophic_gt3frame 1
timing.diagnostic_pairs[0]   pred_id=e1, gt_id=g1, err_frames=4
phase.per_prediction[0]      pred_id=e1, assoc_gt_id=g1, phase_class=mid_stance
AFTER-state verified
```
The former failure is reproduced and fixed: existence still refuses the match, while the
signed error and the catastrophic counter now see exactly what operators saw.

---

## 6. Verification & test evidence

| Check | Command | Result |
|---|---|---|
| Full Docker suite (the gate) | `make auto-sfx-test` | **272 passed** (was 196; +76 new/extended) |
| Per-area counts | pytest --collect-only | association 14 · metrics 50 · classify 15 · tracing 14 · render/markers 46 · forensics 14 · gt 23 |
| Behavior neutrality | `tests/test_tracing.py::TestNeutrality` | events+diagnostics identical, tracing on/off, all three methods + overrides |
| Reproducer | `scripts/repro_four_frame.py [--after]` | §5 outputs |
| Real-clip smoke | `./run analyze /work/auto-sfx/clips/walk1.mp4 --run-id gate2-walk1-smoke` | 25 events; `contact-trace.jsonl`: 2134 frame rows (2×1067), 212 transitions, 33 guard decisions, 26 candidates; state distribution UNKNOWN 35 / SWING 26 / CANDIDATE 1373 / STANCE 637 / EXIT 63; `./run forensics` (no GT fallback path) OK |
| Exact-trace forensics | `./run forensics runs/gate2-synth-smoke gt/walk_synthetic.smoke.gt.json` | 27 windows (15 predicted + 12 missed_gt); window JSON: `trace.exact=true`, real states, 22 real transitions, per-frame `thr_bs_s`, `thresholds.note="exact active config"`; missed_gt entries carry `gt_id`/`visibility`/`viewpoint` |
| T3 on real run | `./run classify runs/gate2-synth-smoke …smoke.gt.json` | `MISSED_CONTACT 12 · SIMULTANEOUS_LR 4 · UNKNOWN 11` — honest: most predictions sit >0.5 s from the GT range end, so UNKNOWN+evidence, not invented classes |
| Preview visual | `runs/gate2-synth-render/annotated.mp4` frame inspection | chips `L STANCE(GT)/--(PRED)` during pose-loss (machine genuinely UNKNOWN — old code would have shown a misleading SW from event windows); `L --(GT)/SW(PRED)` at t=3.6 s from persisted SWING state; hollow-red GT triangles where no prediction came near |
| Host-suite note | host pytest | 8 failures in `test_ingest`/`test_shots`/`test_pose_cache` are **host-environmental** (macOS cv2 lacks the fixture mp4 encoder; heavy-deps import check) — files untouched by this branch; Docker is authoritative and green |

---

## 7. T2 / T3 acceptance checklist

**T2 (forensic traces):**
- [x] Separate per-frame trace (`contact-trace.jsonl`), events.json unbloated
- [x] Per frame+foot: segment/frame/PTS, heel/ankle/toe coords+conf (via pose.jsonl windows), foot reference, velocity components+magnitude, support-band value, active thresholds, swing peak, state before/after, run provenance, relaxed-run flag, transitions, guard accept/reject reasons
- [x] Exact active config persisted (`detector-config.json`, `source` provenance)
- [x] Behavior-neutrality regression (TestNeutrality)
- [x] Forensics consumes the trace; exact vs reconstructed is machine-checkable (`win["trace"]["exact"]`), never mislabelled
- [x] No invented "raw": backend-internal smoothing explicitly declared unavailable in notes
- [x] Every predicted event and every GT miss has an inspectable artifact with linked `gt_id`/`assoc_err_frames`
- [x] Preview on shared association; chips from persisted state (visually verified)
- [x] PNG per window + summary plot (matplotlib, in-image)

**T3 (classification):**
- [x] Directive table produced (`classify.json`/`.csv`), columns incl. backend/method/viewpoint/visibility
- [x] Only mechanically defensible classes auto-assigned; UNKNOWN always carries evidence
- [x] Human-only classes asserted never auto-assigned
- [x] GT protocol decided (§3); no media substitution

---

## 8. GT status — BLOCKED, with the ASK

`work/auto-sfx/gt/` contains only `walk_synthetic.{smoke,aligned}.gt.json` (fixture
conversions — plumbing regression only, never real-contact evidence). **No operator marks
exist for walk1 or walk2.** Per the directive, Checkpoint A is therefore not produced; the
tooling fires the moment marks land (pose caches make re-runs ~free).

```
ASK (ground-truth marks)
Clips & ranges: work/auto-sfx/clips/walk1.mp4   10.0–30.0 s   (sagittal)
                work/auto-sfx/clips/walk2.webm  18.0–28.0 s   (+ adjacent
                shot-contiguous frontal/rear/diagonal segments where feet are
                visible; keep walk2.webm — do NOT use the regenerated MP4)
For every visible initial contact, give: foot (left/right/uncertain),
earliest / preferred / latest frame (use: experiments/auto-sfx/run annotate
<clip> --at <t> --frames 7  → prints frame/PTS, saves PNGs to scrub),
visibility (clear/partial/occluded), viewpoint if known.
Sufficiency = clear contacts per viewpoint (enough per view class for a
signed-error distribution), not elapsed seconds.
Format: fill experiments/auto-sfx/gt/<clip-stem>.gt.json (schema §5.1;
worked example: work/auto-sfx/gt/walk_synthetic.smoke.gt.json).
Partial marks welcome; uncertain foot is a valid value.
```

Once marks land, the unchanged-baseline Checkpoint A run is:
```bash
./experiments/auto-sfx/run analyze  /work/auto-sfx/clips/walk1.mp4  --gt experiments/auto-sfx/gt/walk1.gt.json
./experiments/auto-sfx/run analyze  /work/auto-sfx/clips/walk2.webm --gt experiments/auto-sfx/gt/walk2.gt.json
./experiments/auto-sfx/run metrics   <run-dir> <gt.json>   # Metrics v2 + views
./experiments/auto-sfx/run forensics <run-dir> <gt.json>   # exact windows + missed-GT
./experiments/auto-sfx/run classify  <run-dir> <gt.json>   # T3 table
```

## 9. Remaining blockers / known limitations

1. **Operator marks** (the only hard blocker for Checkpoint A).
2. Forensics heel/ankle/toe coordinate windows still read `pose.jsonl` (backend-exported
   landmarks); backend-internal *unsmoothed* values remain unavailable — stated in-window,
   not worked around.
3. `walk2.mp4` unfinalized (A7): upstream splitter fix deferred per prior Q4 — no change this
   session; `walk2.webm` stays canonical.
4. Host-macOS cv2 cannot encode the test fixtures (8 host-only failures); Docker is the
   reference environment.
5. `summarize_runs.py` v2 columns read keys that all still exist; not yet extended for the new
   `views`/classify artifacts (cosmetic, non-blocking).

## 10. Integrity

- Commits (local only, oldest→newest): `20c59a8` association+metrics v2 · `0d33510` tracing ·
  `140f703` cli wiring fix · `15a3951` T3 classify · `789a411` forensics+preview rework ·
  `8392bef` test cleanup.
- Dirty-worktree state: tracked files clean; untracked = operator's own docs
  (`docs/*.md`), `experiments/fal-hunyuan-foley-eval/`, and gitignored runtime outputs
  (`work/`, smoke runs referenced above live under `work/auto-sfx/runs/gate2-*`).
- No pushes, no PRs, no releases, no remote mutation. No thresholds touched; detector
  decision logic unchanged (neutrality-tested).

## 11. Exact next authorized action

**Awaiting operator GT marks (§8).** On receipt: run the unchanged baseline on the exact
marked sources, generate Metrics v2 + forensics + T3 + previews, deliver Checkpoint A, and
stop before any T5/T8 algorithm work. If the marks contradict the T4 stamp-policy hypothesis,
the repaired instruments will show it — that is the point.
