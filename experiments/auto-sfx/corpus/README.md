# Auto-SFX corpus integrity and blind split reservation

No challenge corpus has been supplied in this repository. The checked-in
`source-manifest.empty.json` is deliberately valid-but-empty and cannot be
reserved. It does not assert rights, provenance, source membership, or a real
split.

The full ready manifest, rights evidence, reservation secret, media, and
operator mapping are private operator inputs. Keep them outside the repository
and outside paths exposed to ordinary engineering agents. Only the generated
engineering manifest and public receipt may enter the normal experiment path.

## Source inventory contract

Copy `source-manifest.empty.json` to private operator storage, change
`manifest_status` to `ready`, and supply a real `corpus_id`, UTC creation time,
operator identity, and sources sorted by `clip_id`. The normative shape is
`schemas/source-manifest-v1.schema.json`; semantic validation is stricter than
JSON Schema alone.

For every source, the operator must record, never infer:

- a normalized relative media path (never an operator-private absolute path),
  byte count, SHA-256, and duration in integer microseconds;
- rights status `verified`, the legal/licence basis, restrictions, the explicit
  allowed use `auto-sfx-local-research-evaluation`, verifier/time, and a
  SHA-256-addressed evidence reference;
- the original source reference, creator/owner, acquisition operator and time;
- sorted leakage keys with at least `source:`, `actor:`, `session:`, and
  `camera:` keys;
- observed viewpoint(s), pseudonymous actor ID(s), and camera ID(s); actor and
  camera IDs must exactly match their leakage keys;
- `gold_evidence: null` until operator-gold annotation evidence exists.

Leakage keys are not cosmetic labels. The tool takes their transitive closure:
clips sharing any source, actor, session, camera, near-duplicate, capture, room, or
other declared key are one indivisible source group. This also handles chains
such as A sharing an actor with B and B sharing a session with C. Exact media
duplicates are rejected rather than split.

Once every clip has operator-gold evidence, replace `gold_evidence: null` with
its annotation artifact SHA-256, clear-contact count, annotated and
hard-negative durations, and pass IDs. The readiness gate is then computed
from actual supplied evidence. Missing evidence is `pending`; it is never
silently treated as zero or a pass.

Use `scripts/reconcile_foot_contact_passes.py` and the procedure in
`../FOOT-CONTACT-ANNOTATION-GUIDE.md` to freeze the two raw passes, create the
prediction-free disagreement package, append adjudication decisions, and seal
canonical per-clip annotations. The resulting `adjudication-receipt.json`
supplies the annotation SHA-256, clear-contact count, and both source pass IDs.
It deliberately does **not** invent annotated or hard-negative duration: the
operator must record those from the declared reviewed spans before populating
`gold_evidence`.

## Deterministic operator procedure

Run from `experiments/auto-sfx` with its container image, or with a Python
environment whose `PYTHONPATH` includes `src`. The commands below use symbolic
operator paths on purpose; they are not claims that a corpus exists.

1. Hash and inventory the real files outside this tool, fill the private
   manifest, then validate bytes, rights, grouping, and readiness:

   ```bash
   PYTHONPATH=src python3 scripts/reserve_source_split.py validate \
     /operator/private/auto-sfx/source-manifest.json
   ```

   `--skip-media-hash` is schema-only inspection and explicitly reports that
   media hashes were not verified. Never use it for a reservation.

2. Before any prediction is run on this corpus, place an operator-held secret
   of at least 32 bytes in private storage. Do not pass the secret on the
   command line, commit it, print it, or share it with engineering agents.
   Record the explicit reservation timestamp; the splitter never reads random
   or clock state.

3. Reserve development, challenge, and locked-holdout source groups. Put the
   operator mapping in a mode-0700 private directory outside the repository:

   ```bash
   umask 077
   PYTHONPATH=src python3 scripts/reserve_source_split.py reserve \
     /operator/private/auto-sfx/source-manifest.json \
     --secret-file /operator/private/auto-sfx/reservation.secret \
     --reserved-at-utc 2026-08-26T12:00:00Z \
     --engineering-output /operator/release/auto-sfx/engineering-corpus.json \
     --receipt-output /operator/release/auto-sfx/split-receipt.json \
     --private-output /operator/private/auto-sfx/locked-split.json
   ```

   Given the same manifest bytes, secret, timestamp, and fractions, membership
   and serialized outputs are identical. The splitter refuses to overwrite any
   existing reservation output. The private mapping is written mode 0600.

4. Give ordinary engineering only `engineering-corpus.json` and
   `split-receipt.json`. The former contains development/challenge entries; the
   latter contains aggregate counts, full-manifest SHA-256, and keyed
   commitments. Neither contains a locked clip ID, source-group ID, media path,
   or media content hash. Do not expose the full source manifest or infer the
   holdout as the complement of a separately shared full inventory.
   Stage only the released development/challenge media beneath the engineering
   manifest's release directory, preserving each relative `media.path`; never
   expose or directory-mount the private all-source media root.

5. As an operator-only check, reconstruct and audit the frozen reservation:

   ```bash
   PYTHONPATH=src python3 scripts/reserve_source_split.py audit \
     /operator/private/auto-sfx/source-manifest.json \
     --secret-file /operator/private/auto-sfx/reservation.secret \
     --engineering /operator/release/auto-sfx/engineering-corpus.json \
     --receipt /operator/release/auto-sfx/split-receipt.json \
     --private /operator/private/auto-sfx/locked-split.json
   ```

   The audit rehashes media, rebuilds every assignment, checks byte-for-byte
   document equality, checks source-group/content separation, and scans
   ordinary output for locked identifiers and content hashes. Its stdout is
   aggregate-only.

## Gate interpretation

The public receipt carries the current corpus gate without claiming that a
reservation passed the scientific marker. Readiness requires at least 12
transitively distinct source groups, multiple actors and cameras, 200
operator-gold clear contacts,
sagittal/diagonal/frontal/rear coverage (and at least two distinct known
views), and at least 25% hard-negative annotated duration. Annotation-dependent
criteria remain `pending` until every source provides operator-gold evidence.
Reservation before annotation is allowed and expected; predictions do not turn
a pending or failed gate into a pass.
