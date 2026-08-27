# Phase 2C Batch 4 — Worker A state report

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.

## Files written

- `experiments/auto-sfx/src/auto_sfx/state_settle.py`
- `experiments/auto-sfx/tests/test_state_settle.py`
- `experiments/auto-sfx/continuation/batch4-worker-state-report.md`

Not edited: `state_contacts.py`, `state_interval.py`,
`state_groundplane.py`, `support_geometry.py`, `contacts.py`, `types.py`
(`CONTACT_METHODS` unchanged), `cli.py`, `tracking.py`. No
`global_offset` / `timing_offset` / `frame_offset` config key exists.
No geometry / support-height *search* keys exist in `SETTLE_CONFIG`.
`support_height_frac` is present only as a `STATE_CONFIG` passthrough to
`detect_state_contacts` and is not used to gate settle. `contact-settle-v1`
is opt-in via `detect_settle_contacts`; it is not registered on
`detect_contacts`. Interval / groundplane / support-geometry are not
imported or called. Existence is exactly `detect_state_contacts`.

## Tests

```text
cd /Users/ben/Projects/py-audio-box/experiments/auto-sfx
PYTHONPATH=src python3 -m pytest tests/test_state_settle.py -q
```

Summary: `8 passed in 1.45s`.

Required cases: height-in-band late-settle (`time_s` after the
body-relative apex, same count/feet/order as `contact-state-v1`,
`shift_frames=2`, `settle_signal=downward_vy`); existence identity on
`build_walk`; `max_forward_frames=0` leaves times unchanged;
already-settled at apex keeps `shift_frames=0` even if later frames are
also settled; `validate_settle_config` rejects offset keys (and unknown
geometry keys fail closed); interpolated/quarantined frames cannot be
`settle_frame`; deterministic events; tracing on/off does not change
events; VFR `time_s` is an existing source-frame timestamp.

## How preferred time is chosen (not +N)

Existence is exactly `detect_state_contacts`. No event is added or dropped.
`time_s` is a source-frame timestamp already present on the pose trace.
Height does not participate in accept/reject.

`downward_vy(i)` is the max of finite `{heel.vy[i], toe.vy[i]}` from
`state_contacts._kinematics` (image y down; positive `vy` is downward).
`speed(i)` is the max of finite `{heel.speed[i], toe.speed[i]}`. A frame
is settled iff it is not in `contact_abstention_frames` and the chosen
signal is finite and `<=` `settle_vy_bs_s` (default 0.08) or
`settle_speed_bs_s` (default 0.16).

1. Per `contact-state-v1` event, `apex_frame` is the original preferred
   index.
2. If that apex frame is already settled, times stay at the apex
   (`shift_frames=0`, `reason=already_settled`). The search does **not**
   take `apex+1`. Searching past an already-settled apex would manufacture
   a systematic +1 and fail `offset_like`.
3. Else search `(apex_frame, apex_frame + max_forward_frames]` (default 4)
   for the first settled sample. Interpolated and quarantined frames are
   skipped and cannot be `settle_frame`.
4. If a later settle exists, `time_s` becomes that frame,
   `latest_s = max(original latest, that time)`, `earliest_s` stays the
   original. Evidence records `apex_frame`, `settle_frame`,
   `shift_frames`, `settle_signal`, `reason=None`.
5. If no settle in the window, the signal is missing on usable frames, or
   the event cannot be indexed, times are unchanged and evidence records
   `shift_frames=0` plus a reason in `{already_settled, no_settle,
   missing_signal, event_unindexed}`.

On the height-in-band fixture the body-relative forward apex is frame 40
(`t=1.333333s`). Left-foot support height at the apex is `0.038` body-scale
(in-band vs `support_height_frac=0.09`); Batch 3 would therefore keep the
apex. Downward `vy` is `1.14` at frames 40 and 41 (not settled) and `0.0`
at frame 42. Settle fires at `t=1.400s` (frame 42): `apex_frame=40`,
`settle_frame=42`, `shift_frames=2`, `settle_signal=downward_vy`. That is
two source frames later because that is the first settled sample, not
because a constant N was applied. `max_forward_frames=0` cannot search
forward, so the original apex is kept. A planted-y variant is already
settled at the apex (`vy=0`); later planted frames are also settled and
still leave `shift_frames=0`.

`validate_settle_config({"global_offset": 2})` raises.

## Remaining risks

- A foot whose downward `vy` sits just above `settle_vy_bs_s` at the apex
  and just below it at `apex+1` will shift by one frame. That is per-event
  kinematics, but a clip whose apexes sit systematically on that edge can
  look offset-like without an offset key.
- Interpolating the would-be settle frame also poisons
  `contact-state-v1`'s `valid[i-5:i+6]` window, so existence of *that*
  event is dropped at the wrapped layer rather than retargeted to a later
  settled frame. Remaining events still refuse the abstained index.
- Negative (upward) `vy` is `<=` bound, so an apex that is already moving
  up counts as settled and is not shifted. The signal is max of finite
  heel/toe `vy`, so one still-descending landmark blocks settle.
- `support_height_frac` still affects *existence* inside
  `detect_state_contacts`. It is not a settle accept/reject gate.
- Worker A did not run opened walk1/walk2 traces.

No commit, push, or PR.
