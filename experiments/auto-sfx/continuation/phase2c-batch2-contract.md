# Phase 2C Batch 2 contract — `phase2c-state-groundplane`

Orchestrator: 2026-08-27. Status: closed — rejected.

## Falsifiable question

Does shot-local support geometry plus retrospective bounded-duration
physical state (SWING → COLLISION → SUPPORT) reduce the pose-apex early
bias of `contact-state-v1` on opened Phase 2B **development** shards
without a registered global timing offset?

This is a mechanism experiment, not an accuracy marker. Agent-draft GT may
be used for regression and signed-error diagnosis only. No operator-gold
claim. No default change. No future locked holdout. No pose inference.
No dense-flow rerun. No threshold retune of rejected identity/sparse
prototypes.

## Hard exclusions

Do not edit, delete, or restyle:

- `experiments/fal-hunyuan-foley-eval/`
- `experiments/sfx-bank-pipeline/`
- unrelated untracked docs
- `src/auto_sfx/state_contacts.py`
- `src/auto_sfx/contacts.py`
- `src/auto_sfx/types.py` (`CONTACT_METHODS` stays unchanged)
- `src/auto_sfx/cli.py`
- `src/auto_sfx/tracking.py`
- `src/auto_sfx/leg_tracking.py`
- `src/auto_sfx/sparse_roi_refiner.py`
- `src/auto_sfx/roi_refiner.py`
- operator files under `work/auto-sfx/gt-drafts/` and `gt-adjudication/`

Workers must not commit, push, open a PR, upload media, or mutate remote
state. Root owns integration and the local checkpoint commit.

## Frozen public API

### `auto_sfx.support_geometry`

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SupportGeometry:
    kind: str                    # "line" | "abstain"
    a: float                     # line a*x + b*y + c = 0
    b: float
    c: float
    residual_rms_bs: float       # inlier RMS in body-scale; nan if abstain
    inlier_count: int
    reason: str | None           # required when kind == "abstain"
    shot_start_frame: int
    shot_end_frame: int          # inclusive

    def height_bs(self, x: float, y: float, scale: float) -> float:
        """Signed height in body-scale. Air (image-up / smaller y) is > 0.
        Returns nan when kind == "abstain" or scale is not finite/positive.
        """
```

```python
def estimate_shot_support(
    xs: np.ndarray,              # (N,) image x, nan allowed
    ys: np.ndarray,              # (N,) image y, y-down
    speeds_bs: np.ndarray,       # (N,) finite when the point is usable
    times: np.ndarray,           # (N,) seconds
    shot_indices: np.ndarray,    # integer frame indices in this shot
    body_scale_px: float,
    *,
    min_points: int = 8,
    max_residual_rms_bs: float = 0.08,
    min_speed_quantile: float = 0.40,
    ground_y_quantile: float = 0.50,
) -> SupportGeometry:
    """One constant support line for the shot. Never a rolling floor.

    Seed observations: finite (x, y, speed) inside shot_indices whose speed
    is at or below the shot's min_speed_quantile AND whose y is at or above
    the shot's ground_y_quantile (lower in the image). Fit a robust line
    (RANSAC or IRLS). Orient the normal so smaller y (air) has positive
    height. Abstain when seeds < min_points, the line is within 15 degrees
    of vertical, residual RMS exceeds max_residual_rms_bs, or body_scale
    is invalid.
    """
```

Also export `DEFAULT_SUPPORT_GEOMETRY_CONFIG` dict with the keyword defaults
above. Unknown keys to any public config helper must raise `ValueError`.

Forbidden: a `global_offset` / `timing_offset` / `frame_offset` key anywhere.

### `auto_sfx.state_groundplane`

```python
GROUNDPLANE_METHOD = "contact-groundplane-v1"

GROUNDPLANE_CONFIG = {
    # tracking — copy contact-state-v1 defaults, do not invent new identity
    "min_step_interval_s": 0.25,
    "max_step_interval_s": 2.0,
    "min_landmark_visibility": 0.5,
    "shot_boundaries": [],
    "actor_too_small_px": 24.0,
    "camera_motion_std_px": 12.0,
    "track_policy": "interpolate",
    "track_jump_speed_bs_s": 14.0,
    "track_swap_margin_bs": 0.08,
    "track_max_interp_gap": 1,
    "track_max_swap_run_frames": 2,
    "smoothing": "raw",
    # geometry
    "geometry_mode": "shot_local",          # shot_local | rolling_quantile_control
    "support_height_frac": 0.09,
    "min_support_points": 8,
    "max_support_residual_rms_bs": 0.08,
    # semi-Markov
    "state_mode": "semi_markov",            # semi_markov | phase_apex_control
    "min_grounded_frames": 3,
    "min_collision_frames": 1,
    "max_collision_frames": 4,
    "min_same_foot_cycle_s": 0.45,
    "phase_min_velocity_bs_s": 0.08,
    "phase_peak_radius_frames": 3,
    "local_window_s": 1.2,
}

def default_groundplane_config() -> dict: ...
def validate_groundplane_config(config: dict | None) -> dict: ...

def detect_groundplane_contacts(trace, config: dict | None = None,
                                record_trace: bool = False):
    """Return contacts.ContactMethodResult. Do not go through detect_contacts.

    method on events: GROUNDPLANE_METHOD
    detector: f"{trace.backend}-contact-groundplane-v1"
    frame states: SWING | COLLISION | SUPPORT | UNCERTAIN only
    interpolated/quarantined frames are invalid as COLLISION or SUPPORT
    """
```

Ablation modes (registered, not tuned after seeing opened GT):

| geometry_mode | state_mode | Meaning |
|---|---|---|
| `rolling_quantile_control` | `phase_apex_control` | Control that should closely track `contact-state-v1` (sanity) |
| `shot_local` | `phase_apex_control` | Geometry only |
| `rolling_quantile_control` | `semi_markov` | State only |
| `shot_local` | `semi_markov` | Full candidate (default) |

`phase_apex_control` may reuse the body-relative apex logic of
`state_contacts._relative_phase_candidates` by importing that private
function, or by a faithful local copy. Do not edit `state_contacts.py`.
`rolling_quantile_control` is the existing per-landmark `_rolling_quantile(y, 0.92)`
floor. `shot_local` uses `estimate_shot_support` once per `_segments()` shot
and evaluates heel/toe height against that constant line.

Semi-Markov (interpretable Viterbi with duration bounds, numpy/scipy only):

- Emissions from heel/toe height, vertical velocity, body-relative vx,
  and speed. Missing observations emit UNCERTAIN. A 1-frame hole may be
  bridged for derivatives only; it is never contact evidence.
- Transitions: SWING → COLLISION → SUPPORT → SWING. UNCERTAIN from any;
  a gap > 1 frame resets to UNCERTAIN.
- COLLISION duration in `[min_collision_frames, max_collision_frames]`.
- SUPPORT duration ≥ `min_grounded_frames`.
- Initial contact = first COLLISION frame of a path that reaches SUPPORT.
- `earliest_s` = first heel or toe collision-band entry; `latest_s` =
  first SUPPORT frame; `time_s` is chosen inside that interval from
  heel-vs-toe collision evidence (earlier of the two if both exist;
  heel preferred when heel collision precedes toe and both are valid).
  Never average away a heel/toe disagreement. Never add a constant N.
- Record `evidence["apex_frame"]` (nearest body-relative forward apex
  within `phase_peak_radius_frames` of the collision, or null),
  `evidence["collision_frame"]`, `evidence["support_frame"]`,
  `evidence["geometry_kind"]`, `evidence["state_change"]`.
- Forbid keys `global_offset`, `timing_offset`, `frame_offset`.

Reuse `track_feet` with the tracking keys above. Reuse
`contacts._segments`, `_body_scale`, `_extract`, observability / shot /
camera diagnostics, `_in_too_small_span`, `_sort_diags` exactly as
`detect_state_contacts` does. Keep `raw_layer: true`. Cadence
contradictions are diagnostics only; do not invent or delete contacts.

## Worker file ownership (disjoint)

### Worker A — identity/state engineer

Owns only:

- `experiments/auto-sfx/src/auto_sfx/support_geometry.py`
- `experiments/auto-sfx/src/auto_sfx/state_groundplane.py`
- `experiments/auto-sfx/tests/test_support_geometry.py`
- `experiments/auto-sfx/tests/test_state_groundplane.py`

Required tests (synthetic; no real clips, no GT files):

1. Horizontal ground: planted points on y=constant, swinging points above;
   `kind=="line"`, heights of planted ≈ 0, swinging > 0.
2. Tilted camera: planted points on a sloped line; rolling-y floor would
   mis-rank uphill vs downhill plants; shot-local heights of plants are
   all near 0.
3. Vertical-line / too few points / huge residual → `kind=="abstain"`.
4. Early-apex fixture: body-relative x peaks 2–3 frames before the foot
   reaches support. Full candidate IC is at collision, not the apex.
   `time_s > apex_time`. Control `phase_apex_control` may fire at the apex.
5. Missing observation / interpolated quarantine frame cannot be IC.
6. VFR timestamps: source PTS membership of events equals trace frames.
7. `validate_groundplane_config({"global_offset": 2})` raises.
8. Frame states ⊆ {SWING, COLLISION, SUPPORT, UNCERTAIN}; no CANDIDATE.
9. Deterministic: two calls, identical events and PTS.
10. Tracing on/off does not change events.

Use `tests/test_contacts.py` helpers (`build_walk`, `uniform_times`,
`make_meta`) where useful. Do not add the method to `CONTACT_METHODS`.

Run: from `experiments/auto-sfx`,
`PYTHONPATH=src python -m pytest tests/test_support_geometry.py tests/test_state_groundplane.py -q`

Write `experiments/auto-sfx/continuation/batch2-worker-state-report.md`
with files changed, test command/output, remaining risks. Do not commit.

### Worker B — truth/evaluation steward

Owns only:

- `experiments/auto-sfx/scripts/run_phase2c_batch2_groundplane.py`
- `experiments/auto-sfx/configs/phase2c-b2-state-groundplane-opened-dev-v1.json`
- `experiments/auto-sfx/tests/test_phase2c_batch2_groundplane.py`

Driver responsibilities:

- Load cached pose traces only. No MediaPipe/RTMPose inference.
- Split fold **must** be `development`. Refuse `validation`, `challenge`,
  `holdout`, or mixed folds.
- Load GT only from
  `experiments/auto-sfx/gt/phase2b-shards/{walk1,walk2}.development.*.gt.json`.
  Refuse to open validation shards.
- Apply the split's 2-frame boundary guard; contacts in the censored band
  are not decision evidence.
- Compare four registered variants plus the live `contact-state-v1`
  baseline on the same traces. Variants are the four geometry/state
  combinations above. Default candidate is shot_local + semi_markov.
- Metrics: `auto_sfx.metrics.evaluate_v2` only. Report strict P/R, clear
  recall if visibility is present, diagnostic median/p90 abs frames,
  fraction >3 frames, **signed** median frames (prediction − GT;
  negative = early), known-side if any known, simultaneous L/R count.
- Offset-equivalence test: per matched event, `candidate_frame - baseline_frame`.
  A result is `offset_like` when ≥80% of those shifts equal the same
  integer (the mode) and that mode ≠ 0. That fails the question.
- Do not compute or print validation-shard metrics anywhere in the
  private or public report.
- Resource: wall seconds and peak RSS (Darwin KiB→MiB like batch 1).
- Atomic JSON writes; refuse overwrite.
- Public metrics schema must not include input paths, argv, or split
  membership lists.

Registered decision gates (write them in the config **before** any run):

```text
accept if ALL:
  - candidate signed median frames is closer to 0 than baseline
    AND candidate signed median > baseline signed median
    (baseline is expected early / negative)
  - not offset_like
  - candidate diagnostic median abs frames <= baseline
  - candidate strict recall >= baseline strict recall - 0.05
  - simultaneous L/R == 0 for candidate
  - geometry abstention is explicit (not silent floor fallback) when
    kind==abstain
  - wall and RSS within config budgets
reject if:
  - offset_like, or signed bias not improved, or recall drop > 0.05,
    or tests fail, or a global offset key exists
inconclusive if mixed and not offset_like
```

Suggested budgets: candidate wall ≤ 8 s combined on both traces;
peak RSS ≤ 256 MiB internally measured.

CLI: argparse with `--config`, `--walk1-pose`, `--walk2-pose`,
`--split`, `--gt-dir`, `--output-dir`. Exit 0 after writing artifacts
even if the scientific decision is reject; the runner records
completion, root writes the decision.

Config experiment_id: `phase2c-b2-state-groundplane-opened-dev-v1-a1`.

Driver tests (no real media required):

1. Tracked config keys are exact; unknown keys fail closed.
2. `fold` other than `development` raises.
3. Validation shard paths in `--gt-dir` are not loaded (temp dir fixture
   with fake `*.validation.*.gt.json` plus development files).
4. `global_offset` in candidate_config raises.
5. Offset-like helper: constant +2 → True; mixed shifts → False.
6. Atomic write refuses overwrite.

If `state_groundplane` is not importable yet, keep detector imports
inside the run function so config tests still collect.

Run: `PYTHONPATH=src python -m pytest tests/test_phase2c_batch2_groundplane.py -q`

Write `experiments/auto-sfx/continuation/batch2-worker-eval-report.md`.
Do not commit. Do not execute the real opened-dev run; root will.

## Evidence boundary sentence (copy into reports)

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.

## After workers return

Root integrates, runs focused + Docker suite, executes the opened-dev
driver under the atomic experiment runner in pinned Docker/CPU, decides
accepted/rejected/inconclusive, and commits the coherent batch.
