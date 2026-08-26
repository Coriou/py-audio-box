# Auto-SFX Phase 2C — operator-gold control-plane closeout

Date: 2026-08-27

Scope: prediction-blind foot-contact annotation, rights/source corpus integrity,
blind source-group reservation, and resumable experiment manifests. Local-only;
no prediction run, remote mutation, media upload, default change, or claim of
operator-gold accuracy.

## Outcome first

The project now has a durable technical path to replace agent-draft/opened
evaluation, but it has not replaced the evidence yet. The foot-contact truth
contract, blind corpus reservation, and crash-safe experiment runner all pass
their behavioral tests. No rights-cleared challenge corpus or operator export
was supplied, so the checked-in corpus remains deliberately empty, no real
split was fabricated, and `operator_verified_available` remains false.

`hybrid-v0` remains the default. `contact-state-v1` remains an opt-in research
candidate. This batch did not run either detector or revisit any Foley model,
classification, RTMPose, smoothing, dense-flow ROI, or other experiment.

## Batch question and decisions

Can agent-draft/opened evaluation be replaced by a durable operator-gold and
resumable experiment path?

| Experiment | Evidence | Mechanism decision | Scientific disposition |
|---|---|---|---|
| `phase2c-gold-contract` | Genuine per-frame source PTS, enriched schema-v2 export, deterministic round trip, HTTP UI, VFR/nonzero-PTS tests, two-pass guide | **Accepted** | Pending two immutable operator passes and adjudication |
| `phase2c-corpus-split` | Rights/source schema, transitive leakage grouping, deterministic HMAC reservation, private locked mapping, public commitments, leakage audit | **Accepted** | Pending a private rights-cleared corpus; no real split exists |
| `phase2c-run-manifest` | Atomic lifecycle, hard-kill/signal reconciliation, pre/post provenance verification, privacy-safe deterministic reports | **Accepted** | Ready for development/challenge experiments |

The answer is therefore “yes” for the audited infrastructure and “not yet” for
the replacement dataset. Accuracy evaluation remains blocked on operator work,
not on another threshold or tooling change.

## Operator truth contract

The dedicated `foot_contact_annotator.html` path does not reuse or reinterpret
generic Foley marks. Session construction probes
`ffprobe.best_effort_timestamp_time` for every presentation-order frame and
fails on absent, duplicated, or non-monotonic timestamps. Source PTS remains
canonical; displayed frame is a zero-based derived index. Browser seeking alone
normalizes by the first source PTS, including nonzero and negative starts.

Each contact records earliest/preferred/latest physical-contact PTS and frames,
anatomical side or uncertainty, visibility, viewpoint, actor, observability,
cropping, notes, and optional audible onset as a separate field. Document-level
provenance includes annotator, pass, session, tool/version, source SHA-256, PTS
map SHA-256, and frame count. Unknown fields and contradictory observation
states fail closed. Import/export is deterministic, and the enriched document
remains readable by the existing schema-v2 GT loader.

The UI isolates local drafts by session, pass, clip, and annotator; changing
identity reloads the correct draft, and stale asynchronous clip loads are
discarded. Separate non-empty pass directories are rejected. The guide requires
two prediction-blind passes, immutable raw exports, and later append-only
adjudication. Two passes by one separated operator must not be described as two
independent annotators.

## Corpus and locked-holdout boundary

The source manifest requires verified local-research rights assertions with
evidence hashes, provenance, media bytes/SHA-256, and leakage keys for source,
actor, session, and camera. Optional room, capture, and near-duplicate keys join
the same transitive leakage component. Exact media duplicates are rejected.
Unknown fields cannot carry predictions or traces into engineering output.

Reservation is deterministic from the validated manifest, an operator-held
secret, explicit timestamp, and fold fractions. Engineering receives only
development/challenge sources. The public receipt contains aggregate locked
counts and keyed commitments, never locked IDs, groups, paths, or content
hashes. The full manifest, secret, and mode-0600 locked map must stay in a
mode-0700 directory outside the workspace. The operator audit reconstructs the
assignment and scans ordinary outputs for leakage.

The checked-in `source-manifest.empty.json` is valid but unreservable. Its gate
is `pending`, with zero clips/groups and no claim that media hashes were
verified. A real gate additionally requires at least two unique annotation pass
IDs per gold source, 12 source groups, 200 clear contacts, all required views,
multiple actors/cameras, and at least 25% hard-negative annotated duration.

## Resumable experiment manifests

`run_phase2c_experiment.py` records unique attempts with atomic
`queued → running → complete | failed | cancelled` transitions. It stores Git
HEAD and an explicit dirty declaration/fingerprint, exact `shell=False` argv,
controlled environment/thread budget, inputs/split/config/models/licenses,
logs, metrics, resources, artifact hashes, decisions, reasons, and both
supersession directions.

Recorded provenance is rehashed before spawn and after exit. Changed inputs
fail before execution; command-time mutation prevents completion. Exit zero is
insufficient without every declared artifact and valid metrics. A hard-killed
runner leaves `running`, never `complete`; explicit reconciliation first proves
the runner PID and child process group are gone, then retains the attempt as
`cancelled`. There is no automatic scientific retry. Deterministic reports use
completed attempts only and omit private inputs, split, command, environment,
paths, and non-public metrics.

## Verification and resources

- Host focused integration: 48 passed in 11.93 seconds.
- Rebuilt Docker/Linux CPU reference image: 2.182 GB, with Git and the new
  runtime assets present.
- Full reference suite: 332 passed in 39.64 seconds.
- Real prediction-free annotation smoke: 1,067 genuine source timestamps from
  local `walk1.mp4`; generated PTS/session assets and all three HTTP resources
  returned 200. Temporary smoke output was removed to Trash.
- Embedded annotator JavaScript parsed successfully with Node.
- No Auto-SFX process remains after verification.

Runner resource figures use `getrusage`; the combined `run` command's peak RSS
can inherit an earlier child high-water mark. Thread budgets are environment
enforced rather than cgroup-enforced, and arbitrary experiment commands are not
sandboxed by the runner.

## Integrity boundary and exact continuation

Unrelated untracked docs and `experiments/fal-hunyuan-foley-eval/` were never
staged or modified. No `annotations/*.gt.json` operator export exists at
closeout. No remote state changed.

Operator execution follows:

- `experiments/auto-sfx/FOOT-CONTACT-ANNOTATION-GUIDE.md`;
- `experiments/auto-sfx/corpus/README.md`.

Until private corpus release and marking are available, engineering may work
only on development data: full-leg-chain temporal identity/anatomical
attribution and the new sparse stabilized heel/toe pixel mechanism. Do not
retune an offset or open a future locked holdout. The immediate verification
command remains:

```bash
make auto-sfx-test
```
