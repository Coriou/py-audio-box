# Phase 2C Batch 4 contract — `phase2c-post-apex-settle`

Orchestrator: 2026-08-27. Status: closed — accepted (opt-in, not default).
HEAD at start: `6b6d58a`. Attempt
`4db22081-d136-4342-a35b-5ffbaf1b3e92`.

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.

## Why this is the next mechanism

Batch 3 showed pose landmark height does not separate apex from contact
on these traces: 72/77 full-clip events are already in-band at the apex,
so a forward support-height search is a no-op and opened-development
timing is identical to `contact-state-v1`.

This batch keeps `contact-state-v1` **existence** and constructs a
**new per-event search** on downward heel/toe `vy`, not height. Preferred
IC is the first frame in `(apex, apex+4]` where downward `vy` falls below
a bound, **even if support height is already in-band**. That is not a
clip-wide +N, not a height-window lengthening, and not a revival of
semi-Markov.

If the apex itself is already settled (downward `vy` already ≤ bound),
there is no *forward* settle: keep the apex (`shift_frames=0`). Searching
past an already-settled apex would manufacture a systematic +1 and fail
`offset_like`.

Image `y` is down. `_kinematics` in `state_contacts.py` defines
`vy[i] = (y[i]-y[i-1])/dt/scale`, so **positive `vy` is downward**.

## Falsifiable question

Does shifting each `contact-state-v1` event's preferred time to the first
post-apex frame where downward heel/toe `vy` settles, even when height is
already in-band, reduce signed early bias on opened development without
dropping existence and without being `offset_like`?

## Hard exclusions

Do not edit: `state_contacts.py`, `state_interval.py`,
`state_groundplane.py`, `support_geometry.py`, `contacts.py`,
`types.py` CONTACT_METHODS, `cli.py`, `tracking.py`, `leg_tracking.py`,
`sparse_roi_refiner.py`, `roi_refiner.py`, foley/sfx-bank trees,
classification experiments, operator files, unrelated docs.

Do not commit. Do not retune Batch 2/3. Do not lengthen
`max_forward_frames`. Do not add a `global_offset` / `timing_offset` /
`frame_offset` key. Do not score validation shards. Do not run pose
inference. Do not revive semi-Markov. Do not use support height as a
gate on the candidate search.

Batch 2 first-attempt `0b53ad08-2577-42a9-b005-bd0ebb091043` is already
handled (failed-finalization, superseded by complete a2). Do not resume
it.

## Frozen API

`auto_sfx.state_settle`

```python
SETTLE_METHOD = "contact-settle-v1"

SETTLE_CONFIG = {
    # tracking/state keys must match contact-state-v1 defaults when passed through
    "max_forward_frames": 4,
    "settle_vy_bs_s": 0.08,          # downward-vy bound; same magnitude as phase_min_velocity_bs_s
    "settle_speed_bs_s": 0.16,       # isotropic-speed control bound; matches speed_thr floor
    "settle_signal": "downward_vy",  # downward_vy | speed
}

def detect_settle_contacts(trace, config=None, record_trace=False):
    """ContactMethodResult.

    1. Call detect_state_contacts (existence layer). Do not drop or add events.
    2. Reuse the contact-state diagnostic_trace / contact_abstention_frames.
       Interpolated and quarantined frames cannot be settle_frame.
    3. Per event, apex_frame is the original preferred index.
       downward_vy(i) = max of finite {heel.vy[i], toe.vy[i]}.
       speed(i) = max of finite {heel.speed[i], toe.speed[i]}.
       A frame is settled iff it is not abstained and the chosen signal is
       finite and <= the matching bound (settle_vy_bs_s or settle_speed_bs_s).
    4. If the apex is already settled, leave times unchanged (shift_frames=0).
       This is required: already-settled must not become a systematic +1.
    5. Else search frames (apex, apex + max_forward_frames] for the first
       settled sample. Height MUST NOT participate. If found, time_s = that
       frame, latest_s = max(original latest, that time), earliest_s =
       original earliest/apex. evidence records apex_frame, settle_frame,
       shift_frames, settle_signal, reason=None.
    6. If no settle in the window, or signal missing on the usable frames,
       leave times unchanged and record shift_frames=0 plus a reason
       (already_settled | no_settle | missing_signal | event_unindexed).
    7. method = SETTLE_METHOD. No CANDIDATE state. No offset keys.
       No geometry / support-height keys in SETTLE_CONFIG.
    """
```

Forbidden: changing which events exist; averaging L/R; clip-wide shift;
using height to accept or reject a settle; lengthening the window after
seeing the fold.

## Worker A — state engineer

Owns:

- `experiments/auto-sfx/src/auto_sfx/state_settle.py`
- `experiments/auto-sfx/tests/test_state_settle.py`

Clone the *structure* of `state_interval.py` (existence wrap, `replace`,
PTS binding, tracing-neutral) but **do not import or call** interval /
groundplane / support-geometry. Use `_kinematics` from `state_contacts`
for `vy`/`speed`. Write tests first (TDD). Synthetic only.

Tests:

1. **Height-in-band late-settle.** At the body-relative apex the foot is
   already in-band on height, but downward `vy` is still above
   `settle_vy_bs_s`; the first settled frame is two source frames later.
   Candidate `time_s` > apex, same event count / feet / order as
   `contact-state-v1`. Evidence: `shift_frames=2`, `settle_signal=downward_vy`.
   Suggested fixture: reuse Batch 3 `rel_x` apex at frame 40; set left-foot
   `y` already near ground at the apex (in-band) with remaining +y motion
   that reaches `vy<=0.08` at frame 42, then stays. Hip-ground scale ~100 px
   so 0.09 body-scale ≈ 9 px.
2. Existence identity on `build_walk` (count, feet, order).
3. `max_forward_frames=0` leaves times unchanged.
4. **Already-settled at apex** (y planted, `vy` below bound at apex and
   after): `shift_frames=0`. Later settled frames must not pull time forward.
5. `validate_settle_config` rejects `global_offset` / `timing_offset` /
   `frame_offset`.
6. Interpolated/quarantined frames cannot be `settle_frame`.
7. Deterministic; tracing on/off does not change events.
8. VFR timestamps: `time_s` is an existing source-frame timestamp.

Run: `PYTHONPATH=src python3 -m pytest tests/test_state_settle.py -q`

Report: `experiments/auto-sfx/continuation/batch4-worker-state-report.md`

Do not commit. Do not run real walk1/walk2 traces.

## Worker B — eval steward

Owns:

- `experiments/auto-sfx/scripts/run_phase2c_batch4_settle.py`
- `experiments/auto-sfx/configs/phase2c-b4-settle-opened-dev-v1.json`
- `experiments/auto-sfx/tests/test_phase2c_batch4_settle.py`

Clone Batch 3 driver contracts: exact config keys, development fold only,
validation shards never opened, 2-frame guard, offset_like (≥80% same
nonzero integer shift), atomic JSON, AUTO_SFX_IMAGE_ID, lazy imports of
`detect_settle_contacts` / `detect_contacts` inside `run()`, exit 0.

Variants (frozen):

- baseline: `contact-state-v1`
- speed_control: settle + `settle_signal=speed`
- candidate: settle + `settle_signal=downward_vy`

Gates (accept if ALL):

- candidate n_pred == baseline n_pred (existence preserved)
- candidate strict recall >= baseline − 0.02
- candidate signed median closer to 0 than baseline AND greater than baseline
- not offset_like
- candidate diagnostic median abs <= baseline
- simultaneous L/R == 0
- wall ≤ 8 s combined, RSS ≤ 256 MiB

experiment_id: `phase2c-b4-settle-opened-dev-v1-a1`

Do not run real traces. Report:
`experiments/auto-sfx/continuation/batch4-worker-eval-report.md`

Public `metrics.json` must be top-level JSON scalars (no nested
`variants` object). Flatten `baseline_*` / `speed_control_*` /
`candidate_*` counts, recalls, signed medians, plus
`recommended_decision`, `offset_like`, `existence_preserved`.

## Evidence boundary (copy)

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.

## Root after workers

1. Focused host tests for both new files.
2. Hash dirty Batch 4 sources/config.
3. Queue+execute via `run_phase2c_experiment.py` in the same pinned
   Docker/CPU envelope as Batch 3 (image
   `sha256:1df6b36c57fbbe82a4402700c7db65175f1d5be5ddd0c03e8bf87e5aeb3ef874`,
   6 CPU, 1 GiB, no network, read-only, traces
   `work/auto-sfx/runs/phase2b-state-final-walk{1,2}/pose.jsonl`).
4. Decide from frozen gates. Do not retune.
5. Compact public report, CTO update, START-HERE / state.json / ledger.
6. `make auto-sfx-test`. Local commit only.
