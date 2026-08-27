# Auto-SFX Phase 2C Batch 5 — operator-pass reconciliation control plane

Date: 2026-08-27

Scope: close the gap between two prediction-blind foot-contact exports and a
durable adjudicated schema-v2 artifact. No media acquisition, human marking,
detector run, opened-fold retune, validation scoring, default change, or
accuracy claim.

## Outcome first

Accepted as an operator-truth mechanism. The repository can now:

1. validate and hash-lock a complete raw pass without overwriting it;
2. detect later source, PTS-map, session, or annotation drift;
3. compare two frozen source-only passes only after pass/order isolation checks;
4. keep existence, window, side, visibility, viewpoint, actor, observability,
   cropping, and audible-onset disagreements explicit;
5. append resolved or unresolved decisions in a superseding SHA-256 chain; and
6. seal only a complete, schema-valid adjudication into canonical per-clip
   operator-gold JSON plus a receipt binding both raw pass IDs.

This removes a real control-plane blocker. It does not remove the evidence
blocker: zero operator exports and zero real rights-cleared corpus reservations
exist. `hybrid-v0` remains default; `contact-settle-v1` remains the opt-in
research candidate.

## Batch question

Can two prediction-blind operator passes become hash-locked, disagreement-
auditable, append-only adjudicated truth without loading predictions or
silently changing source PTS/windows?

**Yes for the mechanism; operator execution remains pending.**

## Integrity behavior

- Finalization requires exactly one canonical export per session clip, one
  stable annotator ID per pass, declared actor IDs, matching session/pass/source
  provenance, and unchanged source/PTS bytes.
- Finalization and sealed artifacts are no-overwrite. Verification recomputes
  every pass commitment rather than trusting filenames or manifest claims.
- Multi-clip comparison requires distinct pass IDs, distinct recorded seeds,
  and different *realized* clip orders. Different seeds that collide on the
  same order fail closed.
- Comparison is recomputed from both frozen raw passes during package
  verification. Canonically rewriting a summary or row does not make a changed
  comparison valid.
- The pairing policy is recorded as
  `actor-scoped-monotonic-nearest-v1` with a default six-displayed-frame
  worklist window. This is not a detector tolerance or evaluation match rule.
- Pass-local contact IDs and prose notes are retained but excluded from physical
  equality. Every structured physical/provenance field remains comparison
  evidence.
- Adjudication stores complete zero/one/many final contact objects per row.
  `resolved` with zero contacts is an explicit no-contact decision;
  `unresolved` cannot assert contacts.
- Corrections append a new unique decision that explicitly supersedes the
  active decision. Old JSONL records are never rewritten, and every line binds
  the prior line and the exact comparison SHA-256.
- `ready_to_seal` validates the fully assembled per-clip annotations. Sealing
  produces canonical schema-v2 files and a receipt with comparison/log hashes,
  source pass IDs, contact counts, and clear-contact counts.

The strict annotation loader remains the prediction-field firewall. The
comparison policy records `prediction_fields_loaded: false`; no detector
artifact path is accepted by the source-session or annotation schemas.

## Behavioral evidence

New focused tests cover:

- complete-pass finalization, no-overwrite behavior, and mutation detection;
- missing clips and undeclared actors;
- VFR/nonuniform source-PTS alignment with timing and side disagreement;
- same-seed and different-seed/same-realized-order rejection;
- recomputation after canonical comparison tampering;
- unresolved-to-resolved append-only supersession;
- decision-chain tampering;
- schema-valid sealed output; and
- exact physical agreement despite pass-local ID/note differences.

Focused reconciliation tests: **9 passed**. Related truth/session/reconciliation
tests: **24 passed**. Pinned Docker/Linux CPU suite: **437 passed in 25.24 s**.
Image `sha256:1df6b36c57fbbe82a4402700c7db65175f1d5be5ddd0c03e8bf87e5aeb3ef874`,
2,182,408,336 bytes.

## Artifacts

- reconciliation module SHA-256
  `5ae44fe30e403b83d13bfeef5c799f3ba5c13441abf31e7df6e49a98e2bfcf60`;
- CLI SHA-256
  `415f8566d0bb283455aa1fe5bd1ec618aeb9869ad9623d5aacf1436e12c52f73`;
- behavioral tests SHA-256
  `e19ae5f39487cc37d2d737c6eeef2b728d4a3a47d54232e934fc65f878ff9766`;
- unchanged prediction-blind annotator SHA-256
  `0fa91846a100adc67e068e20839510eeeffd19cb21ef3f2b9a45d15e3d725085`.

Operator commands and decision shape are documented in
`FOOT-CONTACT-ANNOTATION-GUIDE.md`. Corpus handoff is documented in
`corpus/README.md`.

## Remaining boundary

No tool can create the missing rights, clips, operator observations, reviewed
hard-negative spans, or human independence. The reconciliation receipt does
not fabricate annotated/hard-negative duration; those remain explicit operator
manifest inputs.

The next substantial marker is unchanged: reserve a new rights-cleared corpus
by source group before predictions, run two prediction-blind passes, adjudicate,
and score the declared challenge gate. Do not retune `settle_vy_bs_s`, lengthen
the four-frame settle window, or treat this tooling result as accuracy.

```bash
make auto-sfx-test
```
