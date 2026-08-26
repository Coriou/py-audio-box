# Auto-SFX continuation experiment ledger

This ledger is append-only at the experiment-row level. Correct mistakes with
a superseding row; do not erase rejected work. Large artifacts live in
`work/auto-sfx/`, while compact evidence and decisions remain here or in a
versioned CTO report.

## Inherited reference

| Batch | Question | Runs/evidence | Decision |
|---|---|---|---|
| Phase 2B | Can continuity tracking, physical contact state, pose alternatives, smoothing, or camera-compensated dense ROI reach the original frozen gate? | `CTO-UPDATE-phase2-3-timing-repair.md`; `phase2b-experiment-log.md`; final runs `phase2b-state-final-walk1/2` | Mechanisms materially improved, gate failed. Preserve opt-in `contact-state-v1`; keep `hybrid-v0` default. Old validation is now opened. |

## Queue

| Priority | ID | Falsifiable question | Required output | Status |
|---:|---|---|---|---|
| 0 | `phase2c-gold-contract` | Can two prediction-blind human passes round-trip into auto-sfx schema v2 without losing source PTS, windows, side, visibility, viewpoint, actor, or provenance? | UI/converter, guide, fixtures/tests, sample session | queued |
| 0 | `phase2c-run-manifest` | Can an interrupted/rate-limited run resume without ambiguity about command, hashes, status, outputs, or decision? | atomic manifest runner and tests | queued |
| 0 | `phase2c-corpus-split` | Can a source-group split prevent near-duplicate/actor/session leakage and keep a new holdout prediction-blind? | rights manifest, split audit, hashes | blocked on new footage/owner input after tooling is ready |
| 1 | `phase2c-track-legchain` | Does full-leg-chain temporal assignment fix multi-view identity without latching legitimate crossings? | fixtures, dev/challenge Metrics v2, ID-switch-conditioned table | queued after gold contract |
| 1 | `phase2c-state-groundplane` | Does shot-local support geometry plus retrospective semi-Markov state reduce the pose-apex early bias without a global offset? | identical-fold ablation, calibration coverage, resources | queued after gold contract |
| 1 | `phase2c-roi-sparse` | Does sparse foot-vs-ground tracking with robust stabilization localise IC better than pose-only at compatible CPU cost? | counterfactuals, identical-fold ablation, resources | queued after manifest runner |
| 2 | `phase2c-footpatch-model` | If classical pixels fail, does a tiny rights-cleared temporal foot-patch classifier add challenge accuracy at calibrated abstention? | source-group CV, ONNX CPU report, license/data manifest | conditional |
| 3 | `phase2c-decoder` | Does bounded sequence reasoning add value without fabricating/deleting unsupported raw contacts? | raw/decoded parallel metrics and contradictions | blocked until raw physical gate is strong |

## Append-row template

| Field | Value |
|---|---|
| Date / orchestrator | |
| Experiment ID / run IDs | |
| Question | |
| Created-at HEAD / dirty state | |
| Input manifest and split hash | |
| GT provenance | |
| Backend/model/artifact/license | |
| Isolated change | |
| Development metrics | |
| Challenge metrics | not opened / values |
| Counterfactuals and resources | |
| Representative correction | |
| Representative new failure | |
| Behavioral/full tests | |
| Decision | accepted / rejected / inconclusive |
| Reason | |
| Supersedes / superseded by | |
| Exact next command | |
