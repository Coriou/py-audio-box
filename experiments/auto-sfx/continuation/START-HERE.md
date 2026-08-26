# Auto-SFX step-sync — orchestrator handoff

This is the entrypoint for the next implementation orchestrator. The goal is
another substantial, audited accuracy marker—not another tooling checkpoint.

## Resume contract

Work locally in `/Users/ben/Projects/py-audio-box`. Begin from the current local
HEAD; this handoff was drafted at `a470db9`, after the Phase 2B commits. Preserve
all operator files and unrelated work. Do not push, open a PR, release, upload
media, or mutate remote state without explicit authorization.

Read, in order:

1. this file;
2. `state.json` beside it;
3. `plans/2026-08-26-auto-sfx-step-sync-continuation.md`;
4. `../reports/CTO-UPDATE-phase2-3-timing-repair.md`;
5. `../reports/phase2b-experiment-log.md`.

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

The last verified suite was 284 passed. Newer unrelated commits exist after
the Phase 2B report; do not reset or rewrite them.

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

No operator foot-contact exports existed when this handoff was written. Check
again first. `gold_annotator.html` and `build_gold_session.py` were added by a
later parallel effort, but their generic Foley schema is not automatically
auto-sfx schema v2. Do not convert generic marks into anatomical side,
visibility, viewpoint, or source-PTS claims without an explicit lossless
foot-contact mode and tests.

The first durable task is an operator-gold foot-contact path plus a new
source-group split reserved before predictions.

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

## Immediate batch

Answer one question: can the project replace agent-draft/opened evaluation with
a durable operator-gold and resumable experiment path?

1. Build a foot-contact-specific annotation mode or converter with source PTS,
   earliest/preferred/latest, side/uncertain, visibility, viewpoint, actor,
   observability, annotator/pass provenance, and optional separate audible
   onset.
2. Add round-trip/schema tests and a prediction-free two-pass guide.
3. Create a rights/source manifest and reserve a new source-level holdout before
   running any predictions on it.
4. Add a small atomic experiment manifest runner (`queued/running/complete/
   failed`) that records HEAD, dirty state, hashes, command/config, metrics,
   resources, and decision.
5. While operator marking is pending, work only on development: full-leg-chain
   identity/anatomical attribution and a sparse stabilized heel/toe pixel
   change-point refiner. Do not retune a global offset on old validation.

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
  experiments/auto-sfx/reports/CTO-UPDATE-phase2-3-timing-repair.md
  experiments/auto-sfx/reports/phase2b-experiment-log.md

Follow the resume contract, truth boundaries, orchestration/rotation protocol,
and next substantial progress gate exactly. Preserve unrelated/operator files.
No push, PR, release, upload, or remote mutation. Docker/CPU remains the
reference unless explicitly expanded. Continue through negative results; do
not stop at a tooling checkpoint, do not tune on the opened Phase 2B holdout,
and do not make an experimental method default before the ultimate gate passes.
```
