# Phase 2C Batch 3 — Worker B eval report

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.

Worker B implemented the opened-development driver only. No commit, no
push, no pose inference, and **no real walk1/walk2 pose-trace run**.

## Files

Created (owned paths only):

- `experiments/auto-sfx/scripts/run_phase2c_batch3_interval.py`
- `experiments/auto-sfx/configs/phase2c-b3-interval-opened-dev-v1.json`
- `experiments/auto-sfx/tests/test_phase2c_batch3_interval.py`
- `experiments/auto-sfx/continuation/batch3-worker-eval-report.md`

CLI: `--config --walk1-pose --walk2-pose --split --gt-dir --output-dir`.
Writes `artifact.json` and `metrics.json` under `--output-dir`. Exit 0
after a successful write even when `recommended_decision` is `reject`.
`auto_sfx.state_interval.detect_interval_contacts` and
`auto_sfx.contacts.detect_contacts` are imported inside `run()` so
config-unit tests collect without Worker A's modules.
`AUTO_SFX_IMAGE_ID` is required at `run()` time.

Registered variants (frozen in the hashed config; not extended after
numbers):

| name | detector |
|---|---|
| `baseline` | `detect_contacts(..., "contact-state-v1")` |
| `rolling_control` | interval + `geometry_mode=rolling_quantile_control` |
| `candidate` | interval + `geometry_mode=shot_local` |

Decision gates are copied into the config as structured JSON and hashed
with it. Metrics use `auto_sfx.metrics.evaluate_v2` / `load_gt_any` only.

Public `metrics.json` is top-level JSON scalars (no nested `variants`
object). Nested per-variant rows and gate predicates live in the private
artifact. Flattened `baseline_*` / `rolling_control_*` / `candidate_*`
counts, recalls, signed medians, plus `recommended_decision`,
`offset_like`, and `existence_preserved`, are top-level so a later
runner wrap can list them as `--public-metric` names.

## Tests

```text
cd experiments/auto-sfx
PYTHONPATH=src python3 -m pytest tests/test_phase2c_batch3_interval.py -q
...........                                                              [100%]
11 passed in 0.12s
```

Covered:

1. Tracked config keys are exact; unknown keys fail closed.
2. `fold` other than `development` raises.
3. `*.validation.*.gt.json` (and challenge/holdout) in `--gt-dir` are
   never opened (garbage JSON in those names would fail if loaded).
4. `global_offset` / `timing_offset` / `frame_offset` raise.
5. Offset-like helper: constant +2 → true; mixed / zero mode → false.
6. Atomic JSON refuses overwrite.
7. `recommended_decision` is `reject` when `offset_like` is true.
8. Source-PTS membership and canonical JSON hashing.
9. Loaded clip-ranges that overlap split validation ranges are refused.
10. Split 2-frame boundary guard drops edge contacts from decision GT.
11. Existence-preservation helper: equal `n_pred` → accept predicate
    true; unequal `n_pred` → `recommended_decision` is `reject`.

## How validation is excluded

- Config `fold` must be `development`; `validation` / `challenge` /
  `holdout` fail closed.
- GT filenames matching `walk{1,2}.{validation,challenge,holdout}.*.gt.json`
  are skipped by name and never passed to `load_gt_any`.
- Only `walk{1,2}.development.*.gt.json` are opened.
- Each loaded file's `range_s` and preferred frames are checked against
  the split's **validation** ranges; overlap raises.
- The split's `boundary_guard_frames` must be 2. Contacts whose preferred
  displayed frame falls in the two-frame censored band of a development
  range are dropped before Metrics v2.
- Public metrics record `validation_shards_loaded: 0` and do not contain
  input paths, argv, split membership lists, or GT preferred times.

## How `offset_like` is computed

Per clip, candidate events are greedily matched one-to-one to baseline
events within a 0.5 s horizon. Each pair contributes
`candidate_frame - baseline_frame` using source-PTS membership on the
cached trace. Across both traces, the result is `offset_like` when ≥80%
of those integer shifts equal the same mode and that mode is not 0.
That predicate is a hard reject in the frozen `decision_gates`.

## How existence is gated

`_existence_preserved(candidate_n_pred, baseline_n_pred)` is true iff
both values are non-bool integers and equal. Equal counts set
`candidate_n_pred_eq_baseline` (accept-required). Unequal counts set
`existence_not_preserved` (hard reject). Recall slack is 0.02.

## Confirmation: no real traces

This worker did not read or score `work/auto-sfx/runs` pose traces, did
not invoke `run_phase2c_experiment.py`, and did not load validation
shards from `gt/phase2b-shards/`. Tests used only synthetic GT JSON in
temporary directories plus the frozen split file for range membership.

## Remaining risks

- The live detector path is unexecuted here; root must run the opened-dev
  driver after Worker A lands `state_interval`.
- JSON `null` (empty diagnostic pairs / undefined recall) is not a valid
  experiment-runner `--public-metric` value. A wrap should list the
  always-present scalars (`recommended_decision`, `offset_like`,
  `existence_preserved`, `n_pred` counts) and the flattened recalls /
  signed medians after a successful run that populated them.
- `AUTO_SFX_IMAGE_ID` is required at `run()` time, matching Batch 1/2
  drivers. Root must pass the pinned image ID when wrapping this command.
