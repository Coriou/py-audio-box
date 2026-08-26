# Auto-SFX step-sync — orchestrator handoff

This is the entrypoint for the next implementation orchestrator. The goal is
another substantial, audited accuracy marker—not another tooling checkpoint.

## Resume contract

Work locally in `/Users/ben/Projects/py-audio-box`. Begin from the current local
HEAD. Phase 2C Batch 0 was built from
`76dc2843b507b18531e795cb0efebde91d54252e`, after the Phase 2B commits.
Preserve all operator files and unrelated work. Do not push, open a PR, release,
upload media, or mutate remote state without explicit authorization.

Read, in order:

1. this file;
2. `state.json` beside it;
3. `plans/2026-08-26-auto-sfx-step-sync-continuation.md`;
4. `../reports/CTO-UPDATE-phase2-4-operator-gold-control-plane.md`;
5. `../reports/CTO-UPDATE-phase2-3-timing-repair.md`;
6. `../reports/phase2b-experiment-log.md`.

Do not load every old report or runtime trace into context. Read source and
tests only for the current batch.

## First commands

```bash
cd /Users/ben/Projects/py-audio-box
git status --short
git log --oneline -12
git merge-base --is-ancestor 728b6a5 HEAD
git merge-base --is-ancestor f517c0a HEAD
git merge-base --is-ancestor e624e8d HEAD
git merge-base --is-ancestor 71c0e10 HEAD
make auto-sfx-test
find work -type f -path '*/annotations/*.gt.json' -print
```

The last verified suite was 332 passed in the rebuilt Docker/Linux CPU
reference image. Newer unrelated commits exist after the Phase 2B report; do
not reset or rewrite them.

## Current verdict

- `hybrid-v0` is still the default.
- `contact-state-v1` is the opt-in research candidate.
- `contact-roi-v1` is an opt-in rejected pixel prototype, retained for evidence.
- Candidate stack: MediaPipe lite VIDEO, raw smoothing, bounded continuity
  tracking with interpolation used only for trajectories and never as contact
  evidence.
- Phase 2B improved combined opened-frozen strict P/R 0.120/0.059 →
  0.273/0.176 and timing 5/8 → 3/6 frames, but clear recall is 0.222 and
  known-side is 0/5.
- No multi-second `CANDIDATE` state remains; abstentions are explicit.
- Pose image mode, current RTMPose, smoothing, and dense-flow ROI variants were
  measured and rejected. Do not rerun them without a new mechanism.

## Truth warning

The existing Phase 2B GT is agent-draft and the old validation has been opened.
It may be used for regression and failure analysis, never as the next final
holdout.

The dedicated, prediction-blind foot-contact path now exists at
`../foot_contact_annotator.html`, with its operating procedure in
`../FOOT-CONTACT-ANNOTATION-GUIDE.md`. It records genuine source PTS, contact
windows, side/uncertainty, visibility, viewpoint, actor, observability,
cropping, pass/session/tool/source provenance, and separate optional audible
onset. Its lossless/schema behavior is tested. Generic Foley marks remain
insufficient and must not be converted into missing anatomical or timing
claims.

No operator foot-contact exports existed at the Phase 2C Batch 0 closeout.
Check again first. The rights/source manifest and blind source-group reservation
mechanism also exist, but the checked-in manifest is deliberately empty and no
real split has been fabricated. A private rights-cleared corpus, deterministic
reservation before predictions, two immutable prediction-blind passes, and
append-only adjudication are still required before evaluation can become
operator-gold.

## Next substantial marker

On a new operator-gold challenge corpus with at least 12 source-group-distinct
clips, 200 clear contacts, multiple views, and ≥25% hard-negative duration:

- clear precision and recall ≥0.75;
- median ≤2 displayed frames, p90 ≤4;
- ≤15% diagnostic pairs beyond 3 frames;
- known-side ≥0.90 where known;
- zero unexplained simultaneous L/R;
- explicit abstention and no supported-view regression;
- CPU/resource envelope recorded.

The ultimate exit gate remains 0.90/0.90, median ≤1, p90 ≤3, ≤5% beyond 3,
and known-side ≥0.95. Do not relax it or make a new default early.

## Phase 2C Batch 0 closeout

Answer one question: can the project replace agent-draft/opened evaluation with
a durable operator-gold and resumable experiment path?

The audited answer is **yes for the mechanism, not yet for the evidence**:

1. The foot-contact-specific source-PTS annotator, strict schema, two-pass guide,
   and round-trip tests are complete.
2. The rights/source manifest, transitive leakage grouping, deterministic blind
   reservation, private locked map, and public receipt are complete. A real
   private corpus and split are still pending.
3. The atomic experiment runner now records queued/running/complete/failed/
   cancelled lifecycle, provenance and artifact hashes, resources, decisions,
   hard-kill reconciliation, and privacy-safe public reports.
4. Zero operator exports and zero real reserved sources existed at closeout, so
   operator-gold accuracy has not been measured and no default changed.

The next engineering batch, while operator work is pending, is development-only:
full-leg-chain temporal identity with separately calibrated anatomical
attribution, plus a new sparse stabilized heel/toe pixel change-point refiner.
Use the manifest runner for experiments. Do not retune a global offset on old
validation, repeat dense Farneback, or inspect a future locked holdout.

## Orchestrated batch shape

With four total slots, the root may assign at most three disjoint workers:

- truth/evaluation steward;
- identity/state engineer;
- pixel/model engineer or independent auditor.

The root owns integration and commits. Workers share a filesystem, must receive
disjoint file/run ownership, and should not commit. Every batch answers one
falsifiable question with 2–4 experiments and closes with accepted/rejected
evidence, tests, resource figures, and a coherent local commit.

## Rotation and rate-limit closeout

Before rotating orchestrators or when rate pressure appears:

1. finish or terminate processes and record their state;
2. update `state.json` and the experiment ledger;
3. record input/split/config/artifact hashes and exact next command;
4. run proportional tests and the full Docker suite when code changed;
5. commit the coherent batch locally;
6. leave a short closeout naming HEAD, result, blocker, and next experiment.

The next orchestrator continues the unresolved batch; it does not restart the
project. Runtime session IDs are not assumed to survive rotation—inspect
processes and atomic run status instead.

## Copy/paste kickoff for the next orchestrator

```text
You are the implementation orchestrator for Auto-SFX step-sync continuation.
Work locally from the current HEAD in /Users/ben/Projects/py-audio-box.

Read first:
  experiments/auto-sfx/continuation/START-HERE.md
  experiments/auto-sfx/continuation/state.json
  plans/2026-08-26-auto-sfx-step-sync-continuation.md
  experiments/auto-sfx/reports/CTO-UPDATE-phase2-4-operator-gold-control-plane.md
  experiments/auto-sfx/reports/CTO-UPDATE-phase2-3-timing-repair.md
  experiments/auto-sfx/reports/phase2b-experiment-log.md

Follow the resume contract, truth boundaries, orchestration/rotation protocol,
and next substantial progress gate exactly. Preserve unrelated/operator files.
No push, PR, release, upload, or remote mutation. Docker/CPU remains the
reference unless explicitly expanded. Continue through negative results; do
not stop at a tooling checkpoint, do not tune on the opened Phase 2B holdout,
and do not make an experimental method default before the ultimate gate passes.
```
