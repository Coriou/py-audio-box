# Auto-SFX Phase 2C Batch 3 — interval-calibrate negative closeout

Date: 2026-08-27

Scope: keep `contact-state-v1` existence and shift preferred IC to the
first shot-local support crossing inside a bounded forward window
(`max_forward_frames=4`). Opened development only. No operator truth,
pose inference, holdout, global offset, default change, or accuracy claim.

## Outcome first

Rejected. Existence is preserved (33=33 predictions, strict P/R unchanged
at 0.424/0.333) but **timing does not move** on the decision fold: signed
median stays **−1.5 frames**, diagnostic median/p90 stay **2/4.1**. The
candidate is event-metric-identical to baseline.

On the full clips, only **5/77** events shift (2–3 frames); **72** are
already inside the support-height band at the apex, so the search
`(apex, apex+4]` never fires. Pose landmark height on this footage does
not distinguish the body-relative apex from contact.

`hybrid-v0` remains default. `contact-state-v1` remains the research
candidate. `contact-interval-v1` is retained as opt-in negative evidence
and is not wired into the CLI.

## Batch question

Does shifting each `contact-state-v1` event to the first shot-local
support crossing inside a bounded forward window reduce signed early
bias without dropping existence and without being `offset_like`?

| Variant | n_pred | strict P/R | signed median | median/p90 | Decision |
|---|---:|---:|---:|---:|---|
| baseline | 33 | 0.424/0.333 | −1.5 | 2/4.1 | reference |
| rolling_control | 33 | identical | −1.5 | 2/4.1 | no movement |
| candidate shot_local | 33 | identical | −1.5 | 2/4.1 | **rejected** |

`offset_like` false. Validation shards loaded: 0. Candidate wall 4.120 s,
133.078 MiB. Attempt `f7c3e0a3-ca30-48d0-9fbb-629307abebf0`, artifact
`a0bd85ca29f0bd4bc11624bbd1ac2b632e178ab56f30fde14f3d3b275939b021`.

Synthetic early-apex still shifts 40→42. That fixture plants the apex in
the air; real MediaPipe feet are already in-band at the apex.

Do not lengthen `max_forward_frames` on this fold: the miss is in-band
at t0, not a short window. Do not return to semi-Markov (Batch 2).

## Continuation

Next pose-only branch: **post-apex vertical settling** — first frame in
`(apex, +4]` where downward `vy` falls below a bound, *even if height is
already in-band*. If that also no-ops or is `offset_like`, treat 2D pose
height/apex as a measured ceiling on these clips and wait on operator
gold plus a different observable (pixels remain rejected; learned
foot-patch remains blocked).

```bash
make auto-sfx-test
```
