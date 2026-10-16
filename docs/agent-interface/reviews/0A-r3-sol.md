# Lot 0A independent adversarial review — round 3

**NO-GO for 0A** at exact candidate `ca0ad376e237f6bf75777cc6196b34688af90a2c`, branch `t3/implement-0a-contracts`, read-only worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-01ab530d`. Contract/schema/fixture/worker revision: **1.2.0**. Review date: 2026-10-16; independent Sol-low reviewer.

The original R2 worker fallback, simultaneous artifact identity substitution, profile drift and omitted-reference counterexamples now reject semantically in valid detached envelopes. The correction is substantial. Two remaining output-boundary defects prevent the complete typed configuration → frozen plan → trusted attempt → reusable/published result handoff required by Astra. Neither requires a store, worker, inference or runtime implementation.

## Required corrections

### R3-1 — P1: reusable versions still expose an unverifiable settings digest without its executable preimage

Locations: `contracts/v1/schema.json:173–240` (voice_version), `contracts/validate.py:142–157` (candidate publication), `:270` (speech settings equality), `contracts/scenarios.py:35–40,60`; `docs/agent-interface/phase-0/CONTRACTS.md:82–86`.

A saved version contains `settings_sha256` but no effective settings object or typed retrievable settings resource. Speech requests likewise have no settings field. The speech plan must contain an exact settings object whose canonical hash equals that saved digest. For prepare, the original plan contains preprocessing settings; for design, it contains VoiceDesign generation settings. Neither supplies the future Base clone speech settings whose digest the candidate claims. A consumer cannot reconstruct language, seed and sampling values from SHA256. An unspecified settings lookup/default convention would recreate the missing executable-input problem in F2.

Candidate publication does not validate this digest against any preimage. From each passing prepare/design transcript, independently JSON-detach all objects, change only `manifest.candidates[0].version.settings_sha256` to `'f'*64` and recompute the artifact's manifest digest. `worker_message` and `publication` both still pass with the original stored plan, job, verified output bytes/hash and ownership tuple. No acoustic input, ACL or output bytes are changed. This publishes a purported reusable version with no supplied or verified settings preimage. A later speech plan cannot honestly recover those settings from the version-get response.

The transcript generator conceals this missing relation: line 60 manufactures the clone digest from the Python helper `qwen_settings('qwen_clone')`, which is not a typed preceding response or declared reusable-settings fixture. Lines 37–40 then choose those defaults again and **overwrite the input version's settings digest** when building speech. The reuse test happens to compare equal for the one matching helper default; it does not establish general exact-version reuse. Even a legitimate non-default seed/settings version needs an explicit typed preimage, not a hash-to-default guess.

Bounded correction: freeze the reusable speech settings in an immutable typed version/configuration (or an explicitly retrievable immutable typed resource), verify its canonical digest at version/publication and containing entry points, and bind the prospective output configuration/settings to the approved prepare/design contract. Keep preparation settings, design-generation settings and reuse settings distinct. Build reuse from the exact version-get response without rewriting it. Add publication negatives for arbitrary/mismatched reuse digests and positives with two non-default reuse settings; run the resulting exact version through speech worker/publication. This is the adjacent settings/reuse requirement still open, not a reopening of the closed selector check.

### R3-2 — P2: verified output metadata is not checked against the operation's required result

Locations: `contracts/validate.py:130–157` (publication), `:307` (speech requires WAV reference); `contracts/scenarios.py:57`; `contracts/fixtures/v1/transcript-project_render.json`.

Publication verifies attempt tuple, manifest digest, result/output ID sets and byte hashes, but not the already-known output media constraints. A truthful verified hash does not establish that the output satisfies the frozen operation.

Executed counterexamples, each following a passing baseline and using detached wire objects:

- For **both prepare and design**, replace the output with an actual synthetic valid **FLAC** (4,800 mono frames, 48 kHz, PCM16; generated in memory by soundfile), supplying its true SHA256, truthful `media_format='flac'`, unchanged candidate reference ID and recomputed manifest digest. Publication accepts. Reusing that exact published version with the truthful output metadata fails `reference_format_mismatch`, because the admitted clone configuration requires a WAV. Prepare also explicitly promised `output_format='wav_pcm16'`. This is valid decoded media with correct hashes, not a false byte-verification receipt.
- For **render**, first supply a true synthetic 146,400-frame stereo PCM24 WAV matching the frozen policy and prove publication passes. Replace only the output with a true 4,800-frame mono PCM24 WAV, updating its truthful metadata/hash and manifest digest. Publication still passes under the original frozen revision/policy. The checked-in render transcript already exhibits this relation failure: policy length is 146,400 and channels=2, while its output is the copied 4,800-frame mono impulse asset. It is not a valid complete render result merely because all catalogue envelopes validate.

Bounded correction: enforce operation-specific result metadata against frozen policy at publication: reusable prepare/design realization must have the supported WAV representation; render frames/sample rate/channels/format must match its frozen policy. Supply enough explicit measured format metadata or a typed trusted verification receipt for PCM subtype checks; actual decoding remains the later runtime's responsibility. Correct the synthetic render transcript and add truthful verified-media positive/negative publication cases. Do not require inference, a renderer or listening in 0A.

## Prior findings and Astra closure matrix

No prior objection was waived. “Closed” below means E1 contract relation only.

| Boundary | Disposition and evidence |
| --- | --- |
| R2-1 / preset selector part of F2 | **Closed.** Closed Qwen/Kokoro configuration selectors, exact bundle/mode catalogue membership and selector-language checks. Two selectors survive typed version-get → frozen plan → message → publication with different hashes. Independent unknown/other-engine selector mutations reach worker entry and reject `unknown_selector`. Missing/empty fields have separate schema coverage. Overall F2 remains open through R3-1. |
| R2-2 / resolution and policy | **Closed for original cases.** Stored complete plan hash includes approved resolution, settings and profile snapshot. Detached worker fallback/profile changes and simultaneous requested/effective artifact model/bundle changes reject `resolution_mismatch`. Caller/operator forbidden fallback and missing reason independently reject through worker entry. Received/rehashed plans cannot update the authoritative job. |
| R2-3 / dependencies and authorization | **Closed.** `required_asset_ids` derives exact prepare, clone/design and render inputs; worker requires duplicate-free exact set and trusted ACL coverage. Original clone/design empty-list-plus-empty-ACL bypass rejects `asset_dependencies_mismatch`; retained list without grant rejects `unmanaged_asset`. Prepare/render omissions and unauthorized inputs likewise reject. Muted beds count; unused history grants no transfer. Scoped content hash/attempt/ACL tests pass. |
| Engine settings and digest | Closed engine variants reject Kokoro seed/instruction/sampling, Pocket presets and unsupported language/conditioning. Plan settings canonical hash and speech version equality are enforced. **Partial overall: prospective/saved version settings preimage/publication remains R3-1.** |
| Clone/design reuse | Selected single reference/transcript/conditioning, Base versus VoiceDesign identity and design lineage are explicit; wrong lineage/realization mutations reject. **Partial overall: reusable settings and actual output representation remain R3-1/R3-2.** |
| Profile admission | CPU/CUDA/Metal share exact immutable snapshot/compatibility checks. Independently exercised execute entry for each: unqualified fixtures reject, explicitly labelled mock qualification passes, same-ID changed thread limit rejects `profile_mismatch`. Suite covers auto, topology/config revision/device/compatibility drift and unknown profiles. No actual hardware qualification. |
| Frozen authority/deadline/retry | **Closed.** Detached text/selector/received-plan changes reject; deadline changes reject `deadline_mismatch`. Stored plan hash is compared to job. Reap/replacement tests preserve acoustic hash with new authorized attempt/deadline/owner; unreaped or stale owner rejects. |
| Publication identity/revision/manifest | Original authority relations **closed**: exact frozen resolution and render revision/non-project zero, current attempt tuple, canonical manifest, result sets and true byte-hash evidence. Independent wrong revision/digest/identity tests reject across operations. **Operation-specific output conformance remains R3-2.** Control must supply a verified current-attempt manifest; worker JSON/global file presence alone is expressly not that authority. |
| F1 bytes | **Preserved closed.** Strict base64 actual-byte upload and authenticated bounded content ranges round-trip in the executed suite. Host/CLI transfer outside model context and direct-text MCP cap are documented. No transport redesign requested. |
| F3 fencing | **Preserved closed.** Rogue attempt/wrong job/owner/staging/input/generation and cancellation/reap tests retained. Independent detached wrong tuple and authenticated-owner cases reject semantically. |
| F4 intrinsic identity | **Preserved closed.** Recursive resource/message resolution checks and all existing substitution/unresolved/forbidden fallback regressions pass. Cross-object full resolution is now additionally bound. |
| F5 immutable edits | **Preserved closed.** Existing take/asset/version preservation tests pass; independently replayed ordinary block edit plus old take rewrite rejects `immutable_identity` after a valid edit baseline. |
| F6 ripple bounds | **Preserved closed.** Safe-maximum positive and atomic overflow tests pass; independently replayed valid maximum-end clip plus one-tick shift rejects `invalid_range`. |
| Legacy/migration | Exact legacy source/defaults retained; independent AST reconstruction matches **8 apps / 296 complete control records**, not just counts. Distinct-root non-destructive migration policy, pending legacy-job retention, unknown provenance and backup/image rollback distinctions remain explicit. No migration executed. |

## Inspection and checks actually performed

Read PRODUCT, IMPLEMENTATION-START, canonical scope/shared contracts/Phase 0, ORCHESTRATION-PROTOCOL, lot 0A, full R1/R2 Sol reports, authoritative targeted Astra report, both candidate responses and current normative CONTRACTS. Inspected cumulative pack/change scope from `38c5b4a8cc6352eaddbd8c825387ca93577f5219`, correction delta from `449dc64`, schema/catalogue, validator, generator/scenarios, all four transcripts, tests, legacy ledger and compatibility/migration/evidence documents. Checked actual Qwen CustomVoice/clone/checkpoint, Kokoro preset/language/seed and Pocket reference/language/seed backend mappings, plus legacy VoiceDesign → realized reference → Base clone reuse. No applicable AGENTS.md found in checked ancestry/relevant paths.

Verified candidate HEAD before/after. Candidate source status remained clean. Independently inspected image ID **`sha256:322ce89410c82841862177a8fed41458fefe2de49e8c43fb0c0bd914f490f868`** and compared SHA256 of every Dockerfile COPY-root file (excluding ignored bytecode): **137 host files / 137 image files, zero mismatches**. No rebuild was needed. The following sole selected suite was run after that match:

```sh
docker run --rm --network none --cpus 2 --memory 3g \
  py-audio-box-0a-fix-r2-tests python -m pytest \
  tests/contracts tests/test_capabilities_and_metadata.py \
  tests/test_clone_prompt_compatibility.py tests/test_clone_prompt_identity.py \
  tests/test_customvoice_feature_flag.py tests/test_voice_cpu_batch.py \
  tests/test_voice_synth_selection.py \
  -m 'not integration and not redis_integration' -q
```

**185 passed in 33.64s**, exit 0: 114 contracts + 71 selected legacy tests. This includes deterministic regeneration only inside the disposable writable container. No source mount or user-data mount.

Independent diagnostics: `/tmp/pab-0a-r3-review.py` and `/tmp/pab-0a-r3-matrix.py`, using the existing temporary validator venv with `PYTHONDONTWRITEBYTECODE=1`; the first diagnostic was also replayed through stdin in the exact verified image, followed by true synthetic FLAC output/reuse probes. A separate bounded image diagnostic exercised true exact versus truncated render WAV metadata/bytes. All containers used `--rm --network none --cpus 2 --memory 3g`, no mounts, sequentially after the suite. Every negative began from a passing baseline and detached JSON/deepcopy objects; observed semantic diagnostics are not TypeErrors/schema errors relabelled as authority closure. `git diff --check 38c5b4a HEAD` passed. No full-suite or collection result is claimed.

Only this root review report was written; candidate source, root ledger/ticket and existing untracked research were preserved. No commits, integration, subagents, models, private audio, cloud/STT/Voicebox, remote e2e script, push/deploy or PR work.

**Final bounded verdict: NO-GO for 0A only.** Correct R3-1/R3-2 in the contract owner lane and independently review the exact new candidate. All evidence here is **E1**. Durable storage/process races, actual credentials/reap, real engine/hardware identity, browser/agent-host acceptance, acoustic approval and final Astra global product review remain separate gates.
