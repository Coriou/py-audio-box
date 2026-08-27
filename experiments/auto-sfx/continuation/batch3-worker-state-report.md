# Phase 2C Batch 3 — Worker A state report

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.

## Files written

- `experiments/auto-sfx/src/auto_sfx/state_interval.py`
- `experiments/auto-sfx/tests/test_state_interval.py`
- `experiments/auto-sfx/continuation/batch3-worker-state-report.md`

Not edited: `state_contacts.py`, `support_geometry.py`, `contacts.py`,
`types.py` (`CONTACT_METHODS` unchanged), `cli.py`, `tracking.py`. No
`global_offset` / `timing_offset` / `frame_offset` config key exists.
`contact-interval-v1` is opt-in via `detect_interval_contacts`; it is not
registered on `detect_contacts`. Batch 2 modules were not edited. The
early-apex fixture was copied into the interval test file.

## Tests

```text
cd /Users/ben/Projects/py-audio-box/experiments/auto-sfx
PYTHONPATH=src python3 -m pytest tests/test_state_interval.py -q
```

Summary: `7 passed in 2.06s`.

Required cases: early-apex `time_s` after the body-relative apex with the
same event count as `contact-state-v1`; existence identity on `build_walk`
(count, feet, order); `max_forward_frames=0` leaves times unchanged;
geometry abstain / no in-band height in the window leave original time
(`shift_frames=0`); `validate_interval_config` rejects offset keys;
interpolated/quarantined frames cannot be `support_frame`; deterministic
events; tracing on/off does not change events.

## How preferred time is chosen (not +N)

Existence is exactly `detect_state_contacts`. No event is added or dropped.
`time_s` is a source-frame timestamp already present on the pose trace.

1. Per `_segments()` shot, fit one constant support line
   (`estimate_shot_support`) or, in the control, the existing rolling-y
   floor. Heights are signed body-scale distances; air (smaller image y)
   is positive. Geometry abstention is explicit (`kind=="abstain"` plus
   `shift_frames=0` and `reason`); events still exist.
2. For each `contact-state-v1` event, `apex_frame` is the original
   preferred index. Search `(apex_frame, apex_frame + max_forward_frames]`
   (default 4) for the first finite heel-or-toe height `<=
   support_height_frac`. Interpolated and quarantined frames are skipped
   and cannot be the crossing.
3. If the apex is already in-band, there is no *forward* crossing:
   times stay at the apex (`shift_frames=0`).
4. If a later crossing exists, `time_s` becomes that frame,
   `latest_s = max(original latest, that time)`, `earliest_s` stays the
   original apex. Evidence records `apex_frame`, `support_frame`,
   `shift_frames`, `geometry_kind`.
5. If geometry abstains, height is missing, or no in-band frame is found,
   times are unchanged and evidence records `shift_frames=0`,
   `geometry_kind`, and `reason`.

On the early-apex fixture the body-relative forward apex is frame 40
(`t=1.333333s`, height `0.389` body-scale). Interval `shot_local` fires at
`t=1.400s` (frame 42): `apex_frame=40`, `support_frame=42`,
`shift_frames=2`, `geometry_kind=line`. That is two source frames later
because that is the first in-band sample, not because a constant N was
applied. `max_forward_frames=1` cannot reach frame 42, so the original
apex is kept.

`validate_interval_config({"global_offset": 2})` raises.

## Remaining risks

- A foot whose apex is just above `support_height_frac` (here ~0.090–0.091
  on the first half of the synthetic walker, from a slightly tilted
  shot-local line) will shift by one frame. That is per-event geometry,
  but a clip whose apexes sit systematically on that edge can look
  offset-like without an offset key.
- Interpolating the would-be crossing frame also poisons
  `contact-state-v1`'s `valid[i-5:i+6]` window, so existence of *that*
  event is dropped at the wrapped layer rather than retargeted to a later
  in-band frame. Remaining events still refuse the abstained index.
- Shot-local RANSAC abstains on near-vertical structure or sparse slow
  seeds. Abstention does not drop recall versus `contact-state-v1`, but it
  also cannot move those events.
- Worker A did not run opened walk1/walk2 traces.

No commit, push, or PR.
