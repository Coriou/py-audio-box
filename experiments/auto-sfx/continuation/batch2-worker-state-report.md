# Phase 2C Batch 2 — Worker A state report

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.

## Files written

- `experiments/auto-sfx/src/auto_sfx/support_geometry.py`
- `experiments/auto-sfx/src/auto_sfx/state_groundplane.py`
- `experiments/auto-sfx/tests/test_support_geometry.py`
- `experiments/auto-sfx/tests/test_state_groundplane.py`
- `experiments/auto-sfx/continuation/batch2-worker-state-report.md`

Not edited: `state_contacts.py`, `contacts.py`, `types.py` (`CONTACT_METHODS`
unchanged), `cli.py`, `tracking.py`. No `global_offset` / `timing_offset` /
`frame_offset` config key exists. `contact-groundplane-v1` is opt-in via
`detect_groundplane_contacts`; it is not registered on `detect_contacts`.

## Tests

```text
cd /Users/ben/Projects/py-audio-box/experiments/auto-sfx
PYTHONPATH=src python -m pytest tests/test_support_geometry.py tests/test_state_groundplane.py -q
```

Summary: `11 passed in 0.95s` (this host: `python3`).

Required cases: horizontal and tilted support lines, abstention
(vertical / too few / residual / invalid scale), early-apex IC after the
body-relative apex, interpolated frame not IC, VFR PTS membership, offset
keys rejected, frame states ⊆ {SWING, COLLISION, SUPPORT, UNCERTAIN},
deterministic events, tracing-neutral events.

## How initial contact is chosen (not +N)

`time_s` is a source-frame timestamp already present on the pose trace. The
path is:

1. Per `_segments()` shot, fit one constant support line
   (`estimate_shot_support`) or, in the control, the existing rolling-y
   floor. Heights are signed body-scale distances; air (smaller image y)
   is positive. Geometry abstention is explicit (`kind=="abstain"` plus
   `AMBIGUOUS_SUPPORT_PHASE`); there is no silent rolling fallback.
2. Semi-Markov Viterbi labels each valid frame SWING / COLLISION / SUPPORT
   / UNCERTAIN with duration bounds. Interpolated and quarantined frames
   cannot be COLLISION or SUPPORT evidence. Missing observations emit
   UNCERTAIN. A 1-frame hole may be bridged for derivatives only.
3. A contact exists only for a path that actually does
   SWING → COLLISION → SUPPORT (support run ≥ `min_grounded_frames`).
4. `collision_frame` is the first COLLISION of that path.
   `support_frame` is the first SUPPORT frame (`latest_s`).
   `earliest_s` is the first heel or toe collision-band entry.
5. `time_s` is the earlier of the heel/toe collision-band entries inside
   `[earliest_s, latest_s]` (heel preferred when heel precedes toe). The
   index is clamped to that interval of real frames. Heel/toe disagreement
   is never averaged. Nothing is added.

On the early-apex fixture the body-relative forward apex is frame 40
(`t=1.333s`). Control `phase_apex_control` fires at `1.333s` (the apex).
Full candidate (`shot_local` + `semi_markov`) fires at `t=1.400s`
(frame 42): `apex_frame=40`, `collision_frame=42`, `support_frame=44`,
`geometry_kind=line`, `state_change=swing_to_collision_to_support`.
That is two source frames later because collision-band evidence is there,
not because a constant N was applied.

`validate_groundplane_config({"global_offset": 2})` raises.

## Remaining risks

- Emission scores are hand-set on synthetic kinematics. Opened walk1/walk2
  traces (follow-cam, landmark jitter) are the actual mechanism test;
  Worker A did not run them.
- Shot-local RANSAC abstains on near-vertical structure or sparse slow
  seeds. That is explicit, but it can drop recall versus `contact-state-v1`
  on a shot with no usable floor line.
- `phase_apex_control` + `shot_local` reuses `_relative_phase_candidates`
  by encoding shot-local height into image-y. Only the geometry-only
  ablation uses that encoding; the default candidate does not.
- Viterbi is O(T²) per foot per shot. Fine for POC lengths; not free if a
  shot is thousands of frames.
- Collision-band width is `support_height_frac` (copied, not retuned).
  Wrong scale on a real actor could slide IC by a frame without being an
  offset key.

No commit, push, or PR.
