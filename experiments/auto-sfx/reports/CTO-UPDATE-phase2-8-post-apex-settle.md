# Auto-SFX Phase 2C Batch 4 — post-apex-settle closeout

Date: 2026-08-27

Scope: keep `contact-state-v1` existence and shift preferred IC to the
first frame in `(apex, apex+4]` where downward heel/toe `vy` settles,
even if support height is already in-band. Opened development only. No
operator truth, pose inference, holdout, global offset, default change,
or accuracy claim.

## Outcome first

Accepted as an opt-in timing wrapper, not as a default and not as an
accuracy marker.

Existence is preserved (33=33 predictions, strict P/R unchanged at
0.424/0.333). Signed median moves **−1.5 → −1.0** frames (later and
closer to zero). Diagnostic median stays **2.0**; p90 **4.1 → 4.0**.
Not `offset_like`: the shift mode is **0** on **65/77** full-clip pairs
(84.4%). The isotropic-speed control is event-metric-identical to
`contact-state-v1`. Height-in-band no longer blocks a vy settle, which
is why this batch moves timing where Batch 3 did not.

Known-side matches **7 → 6** (0.857 → 0.833). That is not a frozen gate
and does not reverse the accept; it is recorded as remaining risk.

`hybrid-v0` remains default. `contact-settle-v1` becomes the opt-in
research candidate and is not wired into the CLI. `contact-state-v1`
remains the existence layer.

## Batch question

Does shifting each `contact-state-v1` event to the first post-apex frame
where downward heel/toe `vy` settles, even when height is already
in-band, reduce signed early bias without dropping existence and without
being `offset_like`?

| Variant | n_pred | strict P/R | signed median | median/p90 | Decision |
|---|---:|---:|---:|---:|---|
| baseline | 33 | 0.424/0.333 | −1.5 | 2/4.1 | reference |
| speed_control | 33 | identical | −1.5 | 2/4.1 | no movement |
| candidate downward_vy | 33 | 0.424/0.333 | **−1.0** | 2/4.0 | **accepted** |

`offset_like` false. Validation shards loaded: 0. Candidate wall 3.286 s,
131.484 MiB. Attempt `4db22081-d136-4342-a35b-5ffbaf1b3e92`, artifact
`86b0775e9cc762e1368aaf0c90e81ca29af5f1c37a6183fd1928dae429a7b204`.

Synthetic height-in-band late-settle still shifts 40→42. Already-settled
apexes keep `shift_frames=0`, so the search cannot manufacture a
systematic +1.

Do not retune `settle_vy_bs_s` or lengthen `max_forward_frames` on this
fold. Do not return to semi-Markov or support-height interval. Remaining
signed bias (−1.0) is not a new pose-height mechanism.

## Continuation

Operator-gold corpus, two prediction-blind passes, and adjudication remain
the next substantial marker. Learned foot-patch stays blocked until those
labels exist. Sparse pixels and full-leg identity stay rejected.

```bash
make auto-sfx-test
```
