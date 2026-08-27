# Phase 2C Batch 3 contract — `phase2c-interval-calibrate`

Orchestrator: 2026-08-27. Status: closed — rejected. HEAD at start: `8cfc87e`.

## Why this is the next mechanism

Batch 2 showed the rolling image-y floor is not the early-bias source
(`geometry_only` == `contact-state-v1`) and that replacing apex detection
with semi-Markov collision destroys existence (0/42 strict). Heel/toe
disagreement windows on these traces are almost empty (61/77 width 0
frames; 72/77 preferred == earliest), so calibrating *inside the already
emitted interval* cannot move timing.

This batch keeps `contact-state-v1` **existence** and constructs a
**new per-event interval** `[apex, first support crossing]` bounded by
`max_forward_frames` (default 4). Preferred IC is the support crossing
when it exists, otherwise the original apex. That is not a clip-wide +N
and is not a re-tune of rejected semi-Markov.

## Falsifiable question

Does shifting each `contact-state-v1` event's preferred time to the first
shot-local support crossing inside a bounded forward window reduce signed
early bias on opened development without dropping existence and without
being `offset_like`?

## Hard exclusions

Do not edit: `state_contacts.py`, `contacts.py`, `types.py` CONTACT_METHODS,
`cli.py`, `tracking.py`, `leg_tracking.py`, `sparse_roi_refiner.py`,
`roi_refiner.py`, foley/sfx-bank trees, operator files.

Do not commit. Do not retune Batch 2 semi-Markov. Do not add a
`global_offset` key. Do not score validation shards. Do not run pose
inference.

Reuse `support_geometry.estimate_shot_support` as-is (do not edit it).

## Frozen API

`auto_sfx.state_interval`

```python
INTERVAL_METHOD = "contact-interval-v1"

INTERVAL_CONFIG = {
    # tracking/state keys must match contact-state-v1 defaults when passed through
    "geometry_mode": "shot_local",          # shot_local | rolling_quantile_control
    "max_forward_frames": 4,
    "support_height_frac": 0.09,
    "min_support_points": 8,
    "max_support_residual_rms_bs": 0.08,
}

def detect_interval_contacts(trace, config=None, record_trace=False):
    """ContactMethodResult.

    1. Call detect_state_contacts (existence layer). Do not drop or add events.
    2. Per shot, estimate support (shot_local or rolling quantile floor).
    3. For each event, search frames (preferred_idx, preferred_idx + max_forward_frames]
       for the first heel-or-toe height <= support_height_frac (finite, not
       interpolated/quarantined). If found, time_s = that frame, latest_s =
       max(original latest, that time), earliest_s = original earliest/apex.
       evidence must record apex_frame, support_frame, shift_frames, geometry_kind.
    4. If geometry abstains or no crossing, leave the event times unchanged
       and record shift_frames=0, geometry_kind, reason.
    5. method = INTERVAL_METHOD. No CANDIDATE state. No offset keys.
    """
```

Forbidden: changing which events exist; averaging L/R; clip-wide shift.

## Worker A — state engineer

Owns:

- `experiments/auto-sfx/src/auto_sfx/state_interval.py`
- `experiments/auto-sfx/tests/test_state_interval.py`

Tests (synthetic only):

1. Early-apex fixture (reuse Batch 2 geometry): candidate time_s > apex,
   same event count as `contact-state-v1` on that trace.
2. Existence identity: same number of events, same feet, same order as
   `detect_contacts(..., "contact-state-v1")` on `build_walk`.
3. `max_forward_frames=0` leaves times unchanged.
4. Geometry abstain / missing height → shift_frames 0, original time.
5. `validate` rejects `global_offset`.
6. Interpolated frame cannot be the support crossing.
7. Deterministic; tracing on/off does not change events.

Run: `PYTHONPATH=src python3 -m pytest tests/test_state_interval.py -q`

Report: `experiments/auto-sfx/continuation/batch3-worker-state-report.md`

## Worker B — eval steward

Owns:

- `experiments/auto-sfx/scripts/run_phase2c_batch3_interval.py`
- `experiments/auto-sfx/configs/phase2c-b3-interval-opened-dev-v1.json`
- `experiments/auto-sfx/tests/test_phase2c_batch3_interval.py`

Clone Batch 2 driver contracts: exact config keys, development fold only,
validation shards never opened, 2-frame guard, offset_like (≥80% same
nonzero integer shift), atomic JSON, AUTO_SFX_IMAGE_ID, lazy imports,
exit 0.

Variants (frozen):

- baseline: `contact-state-v1`
- rolling_control: interval + `rolling_quantile_control`
- candidate: interval + `shot_local`

Gates (accept if ALL):

- candidate n_pred == baseline n_pred (existence preserved)
- candidate strict recall >= baseline − 0.02
- candidate signed median closer to 0 than baseline AND greater than baseline
- not offset_like
- candidate diagnostic median abs <= baseline
- simultaneous L/R == 0
- wall ≤ 8 s combined, RSS ≤ 256 MiB

experiment_id: `phase2c-b3-interval-opened-dev-v1-a1`

Do not run real traces. Report:
`experiments/auto-sfx/continuation/batch3-worker-eval-report.md`

## Evidence boundary (copy)

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.
