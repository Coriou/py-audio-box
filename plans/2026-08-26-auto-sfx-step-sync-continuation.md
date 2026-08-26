# Auto-SFX step-sync continuation plan

Status: active

Created: 2026-08-26

Supersedes: none

Machine-readable state: `experiments/auto-sfx/continuation/state.json`

## Mission

Continue the foot-contact timing vertical slice until either:

1. the ultimate operator-gold exit gate is met on a new source-level locked
   holdout; or
2. every credible pose, identity, pixel, and small learned timing branch has
   been measured and evidence supports a narrower product scope or an input
   upgrade.

This is a product-quality effort for video creators, not a demo-tuning loop.
The product promise should eventually be “extremely accurate inside a declared
supported envelope, and honest outside it,” not “a step on every walking shot.”

## Non-negotiable evidence rules

- Never use predictions, event times, traces, overlays, generated audio, or
  model outputs to manufacture truth.
- The Phase 2B frozen subset has been opened. It is now an audit/regression set,
  not future final validation.
- Freeze the next holdout by source/actor/session before predictions. Do not
  split adjacent frames or near-duplicate takes across folds.
- Agent-draft marks remain labelled agent-draft. Only human marks become
  operator-gold, with annotator identity and independent-pass provenance.
- Keep visible physical contact, audible onset, and editorial placement as
  distinct fields. This project currently optimizes visible physical IC.
- Report strict existence separately from broad diagnostic timing. Large
  timing errors must remain measurable; a wider diagnostic pair is never a
  strict match.
- Reject improvements that regress any supported viewpoint or the locked
  challenge set. Preserve negative results.
- Do not make a method default until the ultimate gate passes.
- Do not add a cadence decoder until raw physical phase is good. A decoder may
  flag/re-rank evidence but cannot invent or silently remove contacts.
- Local work only unless the owner explicitly authorizes remote mutation.

## What Phase 2B established

Accepted mechanisms:

- bounded continuity tracking isolates the high-confidence frame-364 L/R swap;
- temporal track identity is separate from anatomical attribution;
- interpolated/quarantined frames cannot be contact evidence;
- bounded local statistics replace lifetime historical maxima;
- `contact-state-v1` uses heel/toe-specific body-relative state and internal
  toe-off, with `SWING | UNCERTAIN | GROUNDED` only;
- discontinuity, crop loss, pose loss, and unobservable phase abstain;
- raw events, diagnostic contradictions, uncertainty, Metrics v2, T3,
  forensics, and previews remain inspectable.

Rejected branches:

- greedy assignment-cost identity, bounds 8/10, corrected identity fed into
  `hybrid-v0`, MediaPipe image mode, RTMPose as currently cached/licensed,
  light smoothing, zero-phase smoothing, full-frame dense-flow ROI, two
  completed Farneback ROI variants, and the >9-minute strict ROI variant.

Measured Phase 2B result on opened agent-draft validation:

- strict P/R 0.273/0.176;
- clear recall 0.222;
- median/p90 3/6 frames;
- 35.5% diagnostic pairs beyond 3 frames;
- known-side 0/5;
- no frozen simultaneous pairs and no `CANDIDATE` state.

Read `experiments/auto-sfx/reports/CTO-UPDATE-phase2-3-timing-repair.md`
for the complete evidence and visuals.

## The next substantial progress marker

Do not claim another marker from the two opened clips alone. The next marker
requires operator-gold challenge evidence:

- at least 12 source-group-distinct clips and 200 clear contacts;
- multiple actors/cameras and side, diagonal, frontal, and rear views;
- at least 25% hard-negative duration;
- source-level development/challenge/locked-holdout split reserved before
  prediction;
- challenge clear P/R ≥0.75/0.75;
- median ≤2 frames, p90 ≤4, and ≤15% beyond 3 frames;
- known-side accuracy ≥0.90 where annotated;
- zero unexplained simultaneous contacts;
- explicit abstention on unsupported/cropped/lost spans;
- no material supported-view regression;
- CPU/resource envelope and correction burden recorded.

This is deliberately below the final 0.90/0.90, 1/3-frame gate. It is a
substantial evidence-backed marker, not a relaxed shipping standard. Open the
new final holdout only for a candidate that first passes the challenge marker.

## Workstream A — operator truth and corpus integrity (first priority)

### A1. Foot-contact-specific annotation path

The current `gold_annotator.html` is useful infrastructure but its generic
Foley schema is not automatically auto-sfx schema v2. Build a foot-contact
mode or lossless converter that requires:

- source PTS plus derived displayed frame;
- earliest/preferred/latest physical-contact window;
- anatomical side `left | right | uncertain`;
- visibility `clear | partial | occluded`;
- viewpoint `sagittal | diagonal | frontal | rear | uncertain`;
- actor ID when more than one person is visible;
- contact observability/cropping flags;
- optional audible onset stored separately;
- annotator, pass ID, notes, and tool version.

Add schema and round-trip tests. The UI must work over HTTP from repo root and
must never show predictions during marking.

### A2. Independent passes and adjudication

- Two human passes per marked sequence; randomize clip order between passes.
- Neither pass sees the other, predictions, traces, or overlays.
- Generate disagreement sheets only after both exports are immutable.
- Adjudicate windows, not just preferred points; preserve unresolved
  `uncertain` side/visibility.
- Keep the raw exports and an append-only adjudication table.

If only one operator is available, two separated passes by that operator are
still useful but must not be described as independent annotators.

### A3. New corpus and split

Create a rights/provenance manifest. Start with 12–20 clips for the next marker,
then grow toward the 40–50-clip diagnostic corpus in the research brief.
Include:

- fixed and moving cameras;
- clear side/diagonal plus deliberately difficult frontal/rear views;
- ordinary walk, run, shuffle, barefoot/shoes, hard/soft surfaces;
- cuts, VFR, blur, compression, crop loss, stationary people, leg motion that
  is not walking, camera motion with no step, and long negative spans;
- multiple source groups, actors, rooms, and cameras.

Reserve folds by source group. Record the manifest and split SHA-256 in
`continuation/state.json`. The final holdout should be inaccessible to
engineering agents until the challenge gate passes.

## Workstream B — resumable experiment control plane

Before another large sweep, add a small local runner (suggested:
`scripts/run_phase2c_experiment.py`) that writes one manifest per run:

```text
run_id
status: queued | running | complete | failed | cancelled
created/started/completed timestamps
git HEAD and dirty-state declaration
input clip hashes and split hash
method/backend/model artifact hashes and licenses
full command/config/environment/thread budget
stdout/stderr path and exit code
metrics/resource/forensics artifact paths
decision: accepted | rejected | inconclusive
reason and supersedes/superseded_by
```

Use atomic status updates so a rate limit or orchestrator rotation cannot make
a half-run look complete. Never auto-retry a failed scientific result; retries
must create a new attempt ID and retain the first failure.

Add a deterministic report generator over completed manifests. Keep media and
large traces in `work/`; commit compact manifests/tables or a tracked ledger.

## Workstream C — identity and observability v2

The next identity model should track the full leg chain rather than only foot
centres:

- hip, knee, ankle, heel, and toe observations per temporal leg track;
- shot-local two-state assignment solved across time (Viterbi/dynamic
  programming), with switch cost and robust innovation likelihood;
- separate posterior for anatomical left/right attribution;
- torso orientation/view direction as evidence, never a hard image-x rule;
- `uncertain` attribution when posterior margin is insufficient;
- observability from foot size, truncation, blur, visibility, track covariance,
  and discontinuity—not pose confidence alone.

Required fixtures:

- the real frame-364 swap;
- one-frame and short-run synthetic swaps;
- legitimate fast crossings;
- frontal/rear view inversion;
- cut reset, re-entry, crop loss, and multi-frame occlusion;
- same-side output caused by identity drift.

Report event accuracy conditioned on ID switches and attribution uncertainty.
Anatomical side must not block side-agnostic timing experiments, but it must not
be silently claimed either.

## Workstream D — timing signal v2

Run two complementary branches on identical development/challenge truth.

### D1. Better physical pose change point

- Estimate shot-local ground/support geometry rather than a rolling image-y
  floor alone.
- Retain separate heel and toe x/y velocity, acceleration, curvature,
  support height, leg extension, and foot pitch.
- Model swing → collision/settling → support as a retrospective semi-Markov
  state sequence with bounded durations and explicit missing observations.
- Calibrate preferred IC inside the heel/toe uncertainty interval on
  development data; do not apply an unregistered global frame offset.
- Emit a calibrated interval and abstention probability.

Start with interpretable change-point/linear or tree baselines. Only use a
temporal neural model if it beats information-matched baselines on source-level
challenge data.

### D2. Sparse stabilized foot-pixel refiner

Do not repeat full-frame Farneback. Prototype:

- decode each candidate window once;
- robust global/background affine or homography from non-actor features;
- pose-guided heel/toe masks with GrabCut/edge or another cheap foreground
  constraint;
- pyramidal LK/sparse feature tracks for shoe/foot and adjacent ground;
- relative vertical deceleration, contact-patch shear/staticness, edge
  persistence, and support-line intersection;
- explicit rejection when background compensation or foot texture is weak;
- camera/edit counterfactuals in which camera motion cannot score as contact.

If classical pixels remain inadequate, train a tiny foot-patch temporal
classifier on rights-cleared operator-labelled windows. MediaPipe may propose
identity/ROI, but the target is physical contact state. Split by source/actor,
retain calibrated abstention, export to ONNX, and compare at matched compute.

## Workstream E — backend and artifact strategy

MediaPipe's foot geometry and anatomical labels are now a measured ceiling on
multi-view footage. Evaluate alternatives only with exact artifact/license
records:

- a commercially cleared whole-body/feet pose checkpoint, if available;
- a custom small foot keypoint/contact network trained on owned or explicitly
  licensed footage;
- high-frame-rate synchronized captures for timing calibration where possible.

Do not spend another batch on RTMPose without resolving the exact checkpoint
grant. Do not infer production readiness from a code repository license.

## Workstream F — product envelope and workflow measurement

Extremely high accuracy may first be achievable on a narrow supported subset.
Measure and declare the gate, for example:

- one dominant actor;
- both feet visible above a minimum pixel size;
- side/diagonal view;
- ordinary walking/running;
- no cut or severe camera warp within the candidate window;
- 24–60 fps with valid source PTS.

Evaluate both coverage and conditional accuracy. A 0.95-precision detector that
honestly covers 55% of creator footage may be a better first product than a
0.70-precision detector that claims everything.

Once timing passes, run a small editor workflow study: accept unchanged,
move/relabel, reject, seconds saved per accepted contact, and trust after false
positives. Keep physical IC and editorial sound offset separately adjustable.

## Orchestration model

The root orchestrator owns scope, truth boundaries, integration, commits, and
the candidate decision. With four total agent slots, use at most three bounded
workers per batch:

1. **truth/evaluation steward** — manifests, annotation conversion,
   split/leak audits, metrics; never sees predictions while marking truth;
2. **identity/state engineer** — full-leg continuity and pose change point;
3. **pixel/model engineer or independent auditor** — refiner/model branch,
   counterfactuals, resource and license review.

Workers share the filesystem. Assign disjoint files/run IDs, forbid worker
commits, and let the root integrate and commit. Do not parallelize two agents
editing the same detector or report.

Each batch should answer one falsifiable question with 2–4 experiments. Close
the batch when the question is answered, even if the answer is negative.

## Context and orchestrator rotation protocol

Rotate the orchestrator after a large coherent batch, before context quality
degrades. A batch closeout must:

1. finish or explicitly terminate running processes;
2. update `continuation/state.json` and the experiment ledger;
3. record candidate, split, input, config, and artifact hashes;
4. append accepted/rejected evidence and exact next command to `START-HERE.md`;
5. run targeted tests plus the full Docker suite when code changed;
6. commit a coherent local checkpoint;
7. leave unrelated/operator files untouched;
8. write a short final message that names the new HEAD and remaining blocker.

The next orchestrator should load only:

1. `continuation/START-HERE.md`;
2. `continuation/state.json`;
3. this plan;
4. the latest CTO report;
5. only the source/tests required for the next queued task.

This bounded read order prevents old reports and huge traces from consuming
context. Use scripts/JQ summaries rather than pasting event arrays or logs.

## Rate-limit and interruption behavior

- Persist the run manifest and `running` status before an expensive command.
- Use resumable process/session IDs only inside one orchestrator turn; after a
  rotation, inspect process state and artifacts rather than assuming a session
  survives.
- On rate limit, stop at a recoverable boundary: no partial source edit, exact
  process/run status recorded, latest coherent commit named.
- A new orchestrator continues the current batch if its falsifiable question
  is unresolved; it does not restart the entire project.
- Never call a checkpoint report “done” while a safe in-scope branch remains.
- Never use rate pressure as a reason to open the locked holdout early.

## Experiment decision template

Every row in the ongoing ledger should contain:

| Field | Required content |
|---|---|
| Question | One falsifiable mechanism |
| Run IDs | Stable `phase2c-*` IDs |
| Inputs | Clip/source group, fold, manifest hash |
| Change | One isolated mechanism or registered bundle |
| Development | Metrics v2, T3, resources, counterfactuals |
| Challenge | Same table, only after development decision |
| Visuals | Representative correction and new failure |
| Decision | accepted/rejected/inconclusive and why |
| Tests | Behavioral fixture and full-suite result |
| Provenance | GT/backend/model/license declarations |

## Immediate first batch

Question: can we replace agent-draft/opened validation with a durable
operator-gold evaluation path before further tuning?

1. Verify current HEAD, required ancestors, dirty state, and `make auto-sfx-test`.
2. Inspect whether new operator exports have appeared under `work/`; do not
   assume the generic demo session contains footstep truth.
3. Add foot-contact annotation mode/conversion plus tests and a precise guide.
4. Build a session manifest for current audit clips and a separate manifest for
   new source-group footage; keep marking UI prediction-free.
5. Add the resumable experiment manifest/status runner.
6. Produce a compact owner ASK: exact URL, clips/ranges, expected minutes,
   two-pass protocol, output destination, and what engineering continues while
   marking is pending.
7. Commit the infrastructure checkpoint, update state, then start identity and
   sparse-pixel development without opening the new holdout.

## Stop conditions

Stop and ask the owner only when:

- operator-gold or new footage is the sole blocker to the next gate;
- a license decision requires owner/legal authority;
- a hardware purchase/upload/hosted service would materially expand scope;
- two product envelopes have similar evidence but imply meaningfully different
  market promises.

Otherwise continue through negative experiments, document them, and rotate
orchestrators cleanly until the next substantial marker or ultimate exit gate.
