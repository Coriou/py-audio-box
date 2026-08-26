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
| 0 | `phase2c-gold-contract` | Can two prediction-blind human passes round-trip into auto-sfx schema v2 without losing source PTS, windows, side, visibility, viewpoint, actor, or provenance? | UI/converter, guide, fixtures/tests, sample session | mechanism complete; two operator passes pending |
| 0 | `phase2c-run-manifest` | Can an interrupted/rate-limited run resume without ambiguity about command, hashes, status, outputs, or decision? | atomic manifest runner and tests | complete — accepted |
| 0 | `phase2c-corpus-split` | Can a source-group split prevent near-duplicate/actor/session leakage and keep a new holdout prediction-blind? | rights manifest, split audit, hashes | mechanism complete; real private corpus/split pending |
| 1 | `phase2c-track-legchain` | Does full-leg-chain temporal assignment fix multi-view identity without latching legitimate crossings? | fixtures, dev/challenge Metrics v2, ID-switch-conditioned table | queued after gold contract |
| 1 | `phase2c-state-groundplane` | Does shot-local support geometry plus retrospective semi-Markov state reduce the pose-apex early bias without a global offset? | identical-fold ablation, calibration coverage, resources | queued after gold contract |
| 1 | `phase2c-roi-sparse` | Does sparse foot-vs-ground tracking with robust stabilization localise IC better than pose-only at compatible CPU cost? | counterfactuals, identical-fold ablation, resources | queued after manifest runner |
| 2 | `phase2c-footpatch-model` | If classical pixels fail, does a tiny rights-cleared temporal foot-patch classifier add challenge accuracy at calibrated abstention? | source-group CV, ONNX CPU report, license/data manifest | conditional |
| 3 | `phase2c-decoder` | Does bounded sequence reasoning add value without fabricating/deleting unsupported raw contacts? | raw/decoded parallel metrics and contradictions | blocked until raw physical gate is strong |

## Phase 2C Batch 0 — durable truth/control plane

| Field | Value |
|---|---|
| Date / orchestrator | 2026-08-27 / root with three disjoint workers |
| Experiment IDs | `phase2c-gold-contract`; `phase2c-corpus-split`; `phase2c-run-manifest` |
| Question | Can agent-draft/opened evaluation be replaced by a durable operator-gold and resumable experiment path? |
| Created-at HEAD / dirty state | `76dc2843b507b18531e795cb0efebde91d54252e`; unrelated untracked docs and `experiments/fal-hunyuan-foley-eval/` declared and preserved |
| Input manifest and split hash | No real corpus/split. Empty template file SHA-256 `d1548711a17a0b5a3542c12c68eab89f8bb10624c6f84febbd3787ba435e9341`; semantic SHA-256 `eba1fe925c810d47af5bb2196d873e5984f082a574b538b57616f37c15b42952` |
| GT provenance | Existing truth remains agent-draft/opened. Zero operator exports found. No prediction supplied to truth tooling. |
| Backend/model/artifact/license | No detector/model run. Rights are operator declarations backed by evidence hashes; no rights claim was invented. |
| Isolated change | Dedicated source-PTS foot-contact UI/schema/guide; rights-aware blind split reservation; atomic run manifests. Generic Foley annotator and detector defaults untouched. |
| Development/challenge metrics | Not run; no accuracy claim and future locked holdout not opened. |
| Counterfactuals and resources | VFR/nonzero/negative PTS, relabelling, prediction-field injection, leakage chains, duplicate content, hard kill, SIGINT, stale process, provenance mutation, report privacy. Docker image 2,182,088,721 bytes; full suite 39.64 s. |
| Representative correction | Linux signal cancellation no longer waits re-entrantly inside a `Popen.wait` signal handler; Docker regression added. |
| Representative remaining blocker | No private rights-cleared corpus, immutable two-pass exports, adjudication, or real reserved split. |
| Behavioral/full tests | Focused 48 passed in 11.93 s; `make auto-sfx-test` 332 passed in 39.64 s; image `sha256:3bd74c1618aebbf7c923898be3ed8e4dc2808fee050363c41b86cdcf90982efd`. |
| Decision | All three mechanisms accepted. Scientific replacement remains inconclusive/pending operator data; no default change. |
| Reason | Tests establish lossless truth, blind grouping, and resumability. They cannot manufacture human marks, rights, corpus membership, or accuracy. |
| Supersedes / superseded by | Generic Foley marks are explicitly not Auto-SFX foot-contact truth. No scientific run superseded. |
| Exact next command | `make auto-sfx-test`; operator procedures are in `FOOT-CONTACT-ANNOTATION-GUIDE.md` and `corpus/README.md`. |
| Detailed evidence | `reports/CTO-UPDATE-phase2-4-operator-gold-control-plane.md` |

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
