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
| 1 | `phase2c-track-legchain` | Does full-leg-chain temporal assignment fix multi-view identity without latching legitimate crossings? | fixtures, dev/challenge Metrics v2, ID-switch-conditioned table | rejected — f364 fixed, but 1,172/2,416 opened frames unsafe |
| 1 | `phase2c-state-groundplane` | Does shot-local support geometry plus retrospective semi-Markov state reduce the pose-apex early bias without a global offset? | identical-fold ablation, calibration coverage, resources | rejected — geometry did not move apex; semi-Markov 0/42 strict |
| 1 | `phase2c-interval-calibrate` | Does a bounded post-apex support-height crossing reduce signed early bias without dropping existence? | identical-fold ablation vs contact-state-v1, offset_like, resources | rejected — 72/77 already in-band at apex; development metrics identical |
| 1 | `phase2c-post-apex-settle` | Does first post-apex downward-vy settling in (apex, +4] reduce signed early bias without dropping existence or becoming offset_like? | identical-fold ablation, offset_like, resources | accepted — signed −1.5→−1.0; 65/77 unshifted; not default |
| 1 | `phase2c-roi-sparse` | Does sparse foot-vs-ground tracking with robust stabilization localise IC better than pose-only at compatible CPU cost? | counterfactuals, identical-fold ablation, resources | rejected — 0/19 evidence-bearing opened suggestions |
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

## Phase 2C Batch 1 — full-leg and sparse-pixel mechanisms

| Field | Value |
|---|---|
| Date / orchestrator | 2026-08-27 / root with two disjoint implementation workers and one read-only auditor |
| Experiment IDs / run IDs | `phase2c-b1-identity-paired-opened-v2-a1` / attempt `7288c198-6544-404b-8bf3-b8949d57a26e`; `phase2c-b1-sparse-paired-walk1-opened-v1-a1` / attempt `4970cce4-71d9-482a-a925-9c6326250981` |
| Question | Do full-leg temporal assignment and sparse stabilized foot/ground pixels pass mechanism counterfactuals and provide usable opened-development evidence without global retuning? |
| Created-at HEAD / dirty state | `c8505c07c9653cc294eb03bf5de31681b1129ce7`; Batch 1 files explicitly declared dirty and exact source/driver/config inputs hashed; unrelated untracked work preserved |
| Input manifest and split hash | Opened Phase 2B traces/media only; split SHA-256 `20829632c0e854336537397839468eec6e2a287a8d74b419a4831432071a297e`. No engineering/challenge/locked corpus exists. |
| GT provenance | Agent-draft/opened only. No GT proximity was computed for sparse pixels and no identity accuracy was claimed. Zero operator exports. |
| Backend/model/artifact/license | Existing cached MediaPipe-lite VIDEO traces; no inference or new model. Repository code plus OpenCV Apache-2.0 declaration. Final reference image `sha256:1df6b36c57fbbe82a4402700c7db65175f1d5be5ddd0c03e8bf87e5aeb3ef874`. |
| Isolated change | Opt-in full-leg second-order component tracking; opt-in sparse LK/affine candidate analyzer; exact PyAV decoded-PTS binding; least-privilege paired drivers. No detector/CLI/default wiring. |
| Development metrics | Identity: repaired opened f364 and neighbours, but 1,172/2,416 unsafe frames and both whole-chain claims rejected; 2.204 s, 88.043 MiB. Sparse: 0/19 suggestions, 18 weak-ground and 1 weak-foot abstention; 15.091 s, 199.211 MiB. |
| Challenge metrics | Not opened; no real challenge/locked split exists. |
| Counterfactuals and resources | One/short swaps, physical crossings, frontal/rear ambiguity, cut/re-entry, crop/occlusion, camera pan/affine, edit, low texture, VFR and exact decoded PTS. Pinned read-only containers, no network, 6 CPUs, 1 GiB cap. |
| Representative correction | Positional full-chain v1 missed opened f364. Bounded second-order velocity isolates distal swap while proximal stays direct; mixed output is contact-ineligible. |
| Representative new failure | Identity quarantines long component runs (48.51% combined unsafe). Sparse adjacent-ground LK supplies no usable real opened candidates. |
| Behavioral/full tests | Focused host/Docker 38 passed; `make auto-sfx-test` 370 passed in 14.80 s. |
| Decision | Both mechanisms **rejected** for integration. Preserve as opt-in evidence only; no default change. |
| Reason | Local counterfactual success does not overcome failed real opened coverage. Lowering reassignment/track thresholds post hoc would not be a new mechanism. |
| Supersedes / superseded by | Identity v2 supersedes the unmanifested positional v1 diagnostic. Neither scientific result supersedes `track_feet` or `contact-state-v1`. |
| Artifact commitments | Identity `4374beda9a96489461bf6a76544feaed1cf4cfa34122f1f1bdecd385a985f75f`; sparse `d73e1ebc562a6d2df1d83bfdc3d31231d2862e8d7048aceea00989f97044d60b`; compact public report file SHA-256 `2894dcb29846cfc5f862415428627c01616319ce6af52f29e2f07b1d9ceda542`. |
| Exact next command | `make auto-sfx-test`; next experiment is `phase2c-state-groundplane` on opened development only. |
| Detailed evidence | `reports/CTO-UPDATE-phase2-5-identity-sparse-negative.md`; `reports/phase2c-batch1-run-report.json` |

## Phase 2C Batch 2 — shot-local geometry and semi-Markov state

| Field | Value |
|---|---|
| Date / orchestrator | 2026-08-27 / root with two disjoint workers (state engineer, eval steward) |
| Experiment IDs / run IDs | `phase2c-b2-state-groundplane-opened-dev-v1-a1` / complete attempt `212cb12b-1342-4292-829e-e09e0a1cfd92` (supersedes failed-finalization `0b53ad08-2577-42a9-b005-bd0ebb091043`) |
| Question | Does shot-local support geometry plus retrospective semi-Markov state reduce the pose-apex early bias without a global offset? |
| Created-at HEAD / dirty state | `167e1706e469d757a52aed41a9c3a8dfe5fdbd86`; Batch 2 files declared dirty; unrelated untracked foley/sfx-bank docs preserved |
| Input manifest and split hash | Opened Phase 2B traces + development shards only; split SHA-256 `20829632c0e854336537397839468eec6e2a287a8d74b419a4831432071a297e`. Validation shards loaded: 0. |
| GT provenance | Agent-draft/opened development only. No operator exports. No accuracy claim. |
| Backend/model/artifact/license | Cached MediaPipe-lite VIDEO traces; no inference. Repository code. Image `sha256:1df6b36c57fbbe82a4402700c7db65175f1d5be5ddd0c03e8bf87e5aeb3ef874`. |
| Isolated change | Opt-in `support_geometry` + `state_groundplane` (`contact-groundplane-v1`); opened-dev driver with frozen ablation variants and decision gates. No detector/CLI/default wiring. |
| Development metrics | Baseline P/R 0.424/0.333, signed −1.5 f, median/p90 2/4.1. Geometry-only identical. Candidate P/R 0.000/0.000, signed +3.0 f, median/p90 6/10.8; 14 predictions; `offset_like` false. Candidate wall 4.670 s, 216.621 MiB. |
| Challenge metrics | Not opened; no real challenge/locked split exists. |
| Counterfactuals and resources | Synthetic early-apex (candidate after apex), tilt/vertical/sparse abstain, VFR PTS, no offset keys, tracing-neutral, Docker 6 CPU / 1 GiB / no network. |
| Representative correction | Synthetic fixture timestamps collision two frames after the relative-x apex. |
| Representative new failure | Semi-Markov under-fires on real opened development (0 strict matches) and is late; shot-local geometry does not shift apex events. |
| Behavioral/full tests | Focused host 43 passed; `make auto-sfx-test` 391 passed in 18.65 s. |
| Decision | Combined mechanism **rejected** for integration. Preserve as opt-in evidence only; no default change. |
| Reason | Local synthetic success and non-offset shifts do not overcome failed real opened existence/timing gates. Retuning collision/support after seeing the fold is not a new mechanism. |
| Supersedes / superseded by | Complete a2 supersedes failed-finalization a1. Neither supersedes `contact-state-v1`. |
| Artifact commitments | Complete analysis `8a6f7690ce9ddf44415796a1dcc5e163175f86ca083f664a2eb6e3c47a20f054`; compact public report file SHA-256 `3b853e6be2249e9be4c52cf73ed5a2397eb83c533d0d0c91777868b8b3a7da0f`. |
| Exact next command | `make auto-sfx-test`; next experiment is `phase2c-interval-calibrate` on opened development only. |
| Detailed evidence | `reports/CTO-UPDATE-phase2-6-state-groundplane-negative.md`; `reports/phase2c-batch2-run-report.json` |

## Phase 2C Batch 3 — bounded post-apex support-height interval

| Field | Value |
|---|---|
| Date / orchestrator | 2026-08-27 / root with two disjoint workers |
| Experiment IDs / run IDs | `phase2c-b3-interval-opened-dev-v1-a1` / `f7c3e0a3-ca30-48d0-9fbb-629307abebf0` |
| Question | Does a bounded post-apex support-height crossing reduce signed early bias without dropping existence? |
| Created-at HEAD / dirty state | `8cfc87e`; Batch 3 files declared dirty; unrelated foley/sfx-bank preserved |
| Input manifest and split hash | Same opened traces + development shards; split `20829632c0e854336537397839468eec6e2a287a8d74b419a4831432071a297e`. Validation loaded: 0. |
| GT provenance | Agent-draft/opened development only. No operator exports. |
| Isolated change | Opt-in `contact-interval-v1` wraps `contact-state-v1` and may delay `time_s` to the first in-band crossing in `(apex, apex+4]`. No events added/dropped. |
| Development metrics | Candidate identical to baseline: n_pred 33, P/R 0.424/0.333, signed −1.5, median/p90 2/4.1. Existence preserved. offset_like false. Full-clip shifts 5/77. Wall 4.120 s, 133.078 MiB. |
| Decision | **Rejected**. Height-after-apex is not a timing signal on these pose traces. |
| Reason | Real feet are already in-band at the kinematic apex. Lengthening the window would not fix in-band-at-t0. |
| Artifact commitments | Analysis `a0bd85ca29f0bd4bc11624bbd1ac2b632e178ab56f30fde14f3d3b275939b021`; public report `c5a030f54b468a353622417ce557fb47d0c255dc6e86211c69a4e19fbdbf0598`. |
| Exact next command | `make auto-sfx-test`; next experiment is `phase2c-post-apex-settle`. |
| Detailed evidence | `reports/CTO-UPDATE-phase2-7-interval-calibrate-negative.md`; `reports/phase2c-batch3-run-report.json` |

## Phase 2C Batch 4 — post-apex downward-vy settle

| Field | Value |
|---|---|
| Date / orchestrator | 2026-08-27 / root with two disjoint workers |
| Experiment IDs / run IDs | `phase2c-b4-settle-opened-dev-v1-a1` / `4db22081-d136-4342-a35b-5ffbaf1b3e92` |
| Question | Does first post-apex downward-vy settling in (apex, +4] reduce signed early bias without dropping existence or becoming offset_like? |
| Created-at HEAD / dirty state | `6b6d58a`; Batch 4 files declared dirty; unrelated foley/sfx-bank/classification preserved |
| Input manifest and split hash | Same opened traces + development shards; split `20829632c0e854336537397839468eec6e2a287a8d74b419a4831432071a297e`. Validation loaded: 0. |
| GT provenance | Agent-draft/opened development only. No operator exports. |
| Isolated change | Opt-in `contact-settle-v1` wraps `contact-state-v1` and may delay `time_s` to the first settled downward-vy frame in `(apex, apex+4]`, even if height is already in-band. Already-settled apexes keep `shift_frames=0`. No events added/dropped. |
| Development metrics | Candidate n_pred 33=33, P/R 0.424/0.333, signed −1.0 vs baseline −1.5, median/p90 2/4.0 vs 2/4.1. Speed-control identical to baseline. offset_like false (mode 0, 65/77). Known-side 6/7. Wall 3.286 s, 131.484 MiB. |
| Challenge metrics | Not opened; no real challenge/locked split exists. |
| Counterfactuals and resources | Height-in-band late-settle, already-settled apex, max_forward=0, interpolated/quarantined skip, VFR PTS, no offset keys, tracing-neutral. Docker 6 CPU / 1 GiB / no network. |
| Behavioral/full tests | Focused host 19 passed; `make auto-sfx-test` 428 passed in 19.70 s. |
| Decision | **Accepted** as opt-in research candidate. No default change. No accuracy claim. |
| Reason | Frozen gates all passed. Height-in-band no longer blocks a vy settle. Isotropic speed does not move events. Remaining −1.0 signed bias is not a new pose-height mechanism. |
| Artifact commitments | Analysis `86b0775e9cc762e1368aaf0c90e81ca29af5f1c37a6183fd1928dae429a7b204`; public report `5d32615a1b7dc657f780a352491af394ba08e9760704f430f467dd17f0c10003`. |
| Exact next command | `make auto-sfx-test`; operator gold remains first; do not retune `settle_vy_bs_s`. |
| Detailed evidence | `reports/CTO-UPDATE-phase2-8-post-apex-settle.md`; `reports/phase2c-batch4-run-report.json` |

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
