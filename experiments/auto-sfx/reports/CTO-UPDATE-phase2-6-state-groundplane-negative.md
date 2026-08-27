# Auto-SFX Phase 2C Batch 2 — state/groundplane negative closeout

Date: 2026-08-27

Scope: opt-in shot-local support geometry plus retrospective bounded-duration
physical state (`SWING → COLLISION → SUPPORT`) on opened Phase 2B
**development** shards only. No operator truth, pose inference, future
holdout, global timing offset, dense-flow rerun, default change, remote
mutation, or accuracy claim.

## Outcome first

The combined candidate is rejected for detector integration.

- Shot-local support geometry as a drop-in for the rolling image-y floor
  **does not move** `contact-state-v1` events: `geometry_only` and
  `control_apex_rolling` match the live baseline exactly on this fold
  (33 predictions, 14 strict matches, signed median −1.5 frames).
- Replacing the body-relative apex timestamp with semi-Markov collision
  **collapses existence**: full candidate strict P/R **0.000/0.000**
  (0/42), 14 predictions, diagnostic median/p90 **6/10.8** frames,
  signed median **+3.0** (late, farther from zero than baseline −1.5).
- The result is **not** a registered global offset (`offset_like` false;
  mode fraction 0.219). Synthetic early-apex still fires after the apex.
  Real opened-development coverage fails the registered gates.

`hybrid-v0` remains the default. `contact-state-v1` remains the opt-in
research candidate. The new modules stay opt-in evidence only; they are
not wired into `detect_contacts` or the CLI.

## Batch question and decisions

Does shot-local support geometry plus retrospective bounded-duration
physical state reduce the pose-apex early bias of `contact-state-v1`
without a global timing offset?

| Variant | Mechanical result | Decision |
|---|---|---|
| baseline `contact-state-v1` | strict P/R 0.424/0.333; median/p90 2/4.1 f; signed −1.5 f; 14/42 | reference |
| `control_apex_rolling` | identical to baseline | control passed |
| `geometry_only` | identical to baseline | **no timing movement** |
| `state_only` | P/R 0.040/0.024; signed −4.5 f; 1/42 | **rejected** |
| full candidate `shot_local`+`semi_markov` | P/R 0.000/0.000; median/p90 6/10.8 f; signed +3.0 f; 0/42 | **rejected** |

This is a substantial negative mechanism marker, not an accuracy marker.
The operator-gold challenge gate remains entirely unmeasured.

All Phase 2B shards are opened diagnostics with agent-draft truth. No
challenge/locked corpus exists, and no contact accuracy is claimed.
Development-only comparison diagnoses mechanism bias; it is not the
operator-gold gate.

## Ablation reading

The pose-apex early bias is the timestamp of the body-relative forward
apex, not the rolling image-y floor. Swapping that floor for a shot-local
RANSAC line left every baseline event in place on these two clips.

Semi-Markov collision/support is a different detector, not a calibration
inside the existing heel/toe uncertainty interval. On opened development
it under-fires and the pairs it does form are late. Lowering collision
duration or support height after seeing these numbers would not be a new
mechanism.

Attempt `0b53ad08-2577-42a9-b005-bd0ebb091043` ran the same scientific
command to exit 0 but failed runner finalization (non-scalar public
metrics). It is retained as rejected. Complete attempt
`212cb12b-1342-4292-829e-e09e0a1cfd92` supersedes that control-plane
record with the same scientific gates.

## Run control and reproducibility

- Complete attempt `212cb12b-1342-4292-829e-e09e0a1cfd92`, artifact
  commitment `8a6f7690ce9ddf44415796a1dcc5e163175f86ca083f664a2eb6e3c47a20f054`.
- Compact public report: `phase2c-batch2-run-report.json`.
- Pinned read-only Docker image
  `sha256:1df6b36c57fbbe82a4402700c7db65175f1d5be5ddd0c03e8bf87e5aeb3ef874`,
  no network, six CPUs, 1 GiB cap. Live `src` and `scripts` bind-mounted
  over the image. Candidate wall 4.670 s, internally measured peak RSS
  216.621 MiB. PTS and deterministic-repeat failures: 0.
- Validation shards loaded: 0. Pose inference: no.

## Verification

- Focused host: 43 passed in 3.44 s.
- Full Docker/Linux reference suite: **391 passed** in 18.65 s
  (370 inherited + 21 new).
- Final image unchanged from Batch 1.

## Decision boundary and continuation

Do not wire `contact-groundplane-v1` into `contact-state-v1`, make it a
default, or retune collision/support thresholds on the opened material.

The next safe development branch is **interval calibration on the existing
apex detector**: keep `contact-state-v1` existence, and choose `time_s`
inside the already-emitted heel/toe `[earliest, latest]` window using
shot-local support crossing. That is the original D1 calibration clause,
isolated from the rejected full state replacement. Do not inspect a
future locked holdout.

Operator work remains the only route to the next scientific accuracy
gate.

Exact verification command:

```bash
make auto-sfx-test
```
