# Lot 0A correction round 2 — candidate response

Implementation response only; independent Sol-low re-review required. No self GO, integration, Phase 0 exit, hardware or acoustic approval.

## Baseline and scope

Task worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-01ab530d`, branch `t3/implement-0a-contracts`, clean start `38c5b4a8cc6352eaddbd8c825387ca93577f5219`. Cherry-picked only `9f1027b95229ed30f12d918b27bba989fbfe5d9d` then `aa9a61a183545df65c96b5ccc5df071aa464c714`, producing `eafdb12` then correction base `449dc64`. Final candidate is the commit containing this response; exact SHA is in the handoff and obtainable with `git rev-parse HEAD`.

Read PRODUCT, IMPLEMENTATION-START, shared contracts/Phase 0, orchestration protocol, lot 0A, both Sol reports and full authoritative Astra targeted report. No AGENTS.md found in task ancestry/relevant tree. Only contracts, contract tests and phase-0 consumer documentation changed. No app/server/UI/scheduler/DB/provision/runtime/Compose changes, private assets, inference, model download, remote e2e, cloud, push/PR/deploy, ledger/ticket/review edit or delegated agent.

Current schema/catalogue/fixture/worker revision is **1.2.0**, API/directory namespace remains v1. This unintegrated pack has no deployed migration obligation. Legacy inventory remains its original 1.0.0 evidence revision, byte-exact; all source/default hashes remain checked. Historical r1/evidence documents are explicitly marked historical. CONTRACTS.md is the single current normative consumer specification.

## Finding response and entry points

| Finding / adjacent requirement | Implemented boundary | Regression evidence |
| --- | --- | --- |
| R2-1 / original F2 | Closed immutable voice configuration, exact selectors, control bundle catalogue membership/languages; exact version snapshot in speech plan; no ID-to-selector convention. | Two selectors survive version-get → worker → publication and have different hashes; unknown/other-engine selector and mode/engine mismatch reject. Missing fields reject schema. |
| R2-2 | Complete resolved execution/profile/settings/policy hashed inside plan; worker compares received plan and resolution to stored control plan and job hash; publication compares the same resolution. | Detached simultaneous requested/effective model/bundle changes, device/precision/profile changes, caller/operator fallback widening, changed/rehashed received inputs, profile topology/limits/revision/compatibility drift. |
| R2-3 | Shared required_asset_ids derives prepare source, selected clone/design reference and all render clips including muted clips. Exact duplicate-free metadata and message dependency sets; trusted ACL independent. | Empty/partial/extra/duplicate IDs and unauthorized dependencies reject; historical render references grant no transfer; wrong bytes reject scoped transfer hash verification. |
| Engine-specific settings | Closed Qwen, Kokoro, Pocket, prepare and render variants; canonical settings digest verified; no silent filtering/default lookup. | Kokoro seed/instruction/sampling schema rejection; incompatible language/settings digest semantic rejection; Qwen clone instruction and conditioning reject; Pocket preset/accelerator invention rejects. |
| Designed reuse | Historical VoiceDesign lineage separate from frozen Base reuse identity and realized WAV; design plan explicitly selects reuse identity. Candidate must retain this identity and ICL realization. | Prepare/design save/version-get then speech over exact realization; mutated realization/origin/description rejects. No reroll. |
| Admission | Config snapshot binds exact operation/mode/engine/model/hash/precision/device/topology/limits/config revision; execute messages invoke admission against current trusted profiles. | CPU/CUDA/Metal mocked positives, auto resolution, absent/unqualified/profile-drift negatives. Synthetic fixture profiles themselves remain unqualified/absent/not-ready. |
| Deadline/reconnect/retry | Deadline matches trusted active job/attempt; same hash/config survives valid reap/replacement with new authorized deadline/owner. | Changed worker deadline, unrelated tuples and unreaped replacement reject; valid replacement keeps input hash. |
| Publication/revision/manifest | Control-verified current result message and canonical manifest required; exact output ID/hash set, artifact linkage, attempt tuple, frozen resolution and revision (0 for non-project). | Global unrelated asset, changed result set/hash/metadata/manifest/revision/attempt and late owner/generation/state reject; valid cancellation race retained. |
| F1/F3/F4/F5/F6 preservation | Existing byte JSON/ranges, intrinsic resolution, immutable edit, safe-max ripple, ownership/reap checks remain active at new revision. | Existing correction/contract suites retained and updated for valid envelopes, plus selected unchanged legacy suites. |

Every authority mutation starts from a passing baseline and detached JSON/deepcopy objects. Semantic relations assert ContractError diagnostics, never count TypeError/signature errors as closure. Unsupported fields and missing required fields separately exercise standard JSON Schema rejection. `stored_context`, configuration/settings/dependencies and publication checks are reached through worker/publication entry points. Admission is required by execute messages; result completion uses stored context without requiring the completing profile to be idle-ready.

## Engine/mode support table (E1 mapping, never installed qualification)

| Origin / execution mode | Exact source mapping | Admitted boundary / limits |
| --- | --- | --- |
| preset / qwen_preset | `lib/tts.py:synthesise_custom_voice`: speaker, language, optional instruct, seed, generation kwargs; `lib/tts_backends/qwen.py:checkpoint_capabilities` | Control-approved CustomVoice bundle and speaker catalogue. Instruction support is an explicit trusted capability (1.7B supports direction, 0.6B does not). CPU/CUDA/Metal require exact qualified profile; no real accelerator claim. |
| preset / kokoro_preset | `lib/tts_backends/kokoro.py`: explicit available preset; fr→fr-fr/f prefix, en→en-us/a prefix; CPUExecutionProvider | CPU float32 only in bounded pack. No seed/instruction/reference/sampling. Selector/language catalogue must reflect actual bundle voices; fixtures synthetic. |
| clone / qwen_clone | `lib/tts.py:synthesise_clone` and Qwen `create_voice_clone_prompt`; Base required by checkpoint validator | Exactly one managed WAV, exact transcript and ICL (nonempty transcript) or x-vector (empty transcript). No instruction. No remote pickle. Local prompt construction is future adapter work. |
| clone / pocket_clone | `lib/tts_backends/pocket_tts.py`: bundle language, authorized WAV reference, optional manual seed | CPU float32 only; exactly one WAV; bundle language must match. No preset/instruction/Qwen sampling. |
| design / qwen_clone reuse | `apps/voice-synth/voice-synth.py:cmd_design_voice`: VoiceDesign audition saved as WAV, then Base clone with `ref_text`, `x_vector_only_mode=False` | Frozen realized WAV/ICL and separate historical VoiceDesign identity. Speech uses Base; it never invokes VoiceDesign. Other designed reuse combinations reject. |
| voice_prepare | Selected source/trim/transcript plus explicit conditioning | Qwen Base ICL/x-vector or Pocket reference preparation. Output managed WAV, not pickle. Heavy preparation execution remains future work; no unrelated project settings. |
| voice_design | Actual VoiceDesign generation over exact description/reference text with explicit Qwen settings | Frozen generation identity plus explicitly frozen Base reuse identity; one candidate/reference output. No fallback by omission. |
| project_render | Frozen composition/policy and `pcm-render`/`pcm24-v1` operation identity | CPU float32 initial renderer contract, one output. Per-take TTS identities retained as provenance. No fabricated single synthesis identity for mixed imported media. |

Catalogue model IDs are opaque pinned control keys, not filesystem paths; backend/checkpoint mapping is control provisioning metadata. The fixture names reflect checked backend modes but hashes/selectors remain synthetic. No new unsupported backend mode, automatic model/device/precision switch or hardware qualification is inferred.

## Inspectable transcripts and canonical preimages

`contracts/fixtures/v1/transcript-{voice_prepare,voice_design,speech_generate,project_render}.json` include catalogue requests/responses, trusted synthetic bundle/profile/ACL fixtures, authoritative plan/job, detached message, artifact, verified manifest and actual synthetic fixture byte hashes. `contracts/scenarios.py` deterministically constructs them. Tests validate every exchange using operation catalogue request/response definitions, then pass the complete worker/publication boundary. Prepare/design candidates are saved/read by version ID and used in a subsequent speech plan with the exact reference and origin. Render obtains the project/revision from project-get. No hidden lookup or selector convention is needed.

Canonical preimages are explicitly documented in CONTRACTS.md and implemented by `canonical_hash`: settings object; complete validated plan; complete result_manifest. Request idempotency remains separately principal/operation/payload scoped. Existing job lookup must return the stored job before new resolution. Manifest outputs identify this attempt's generated assets; input IDs are not output IDs. Runtime must verify bytes/media before supplying trusted evidence.

Bytes remain authenticated explicit base64 JSON ranges. Ordinary replies stay metadata-only. Host/CLI adapters transfer/decode outside model context. A direct text-only MCP lane must cap inline decoded payloads at 3072 bytes and send larger work through that host/CLI path. Real client payload/token behavior is a later 0C/1B gate; no transport platform redesign or model-context acceptance is claimed.

## Verification and limits

All evidence here is **E1**. No durable store/process race, actual OS credential/reap, renderer sample fidelity, engine inference, installed hardware, acoustic identity, browser or agent-host acceptance is established. Trusted catalogues/profile qualification/ACLs/result verification must come from control; worker JSON is never their authority. The oracle verifies relations, not the truth of independently supplied hardware receipts. Busy/disconnected/quarantined profiles prevent dispatch; they never permit re-resolution under an old hash. Runtime admission must enforce live concurrency/RAM/disk/deadline ceilings beyond the static text/variant/decoded-duration checks here.

Initial local lightweight contract run: 9 failed / 91 passed, then 100 passed after correcting receipts, stale immutable-test expectations, empty-selector semantic ordering and a render test helper. Raw short traceback output retained in FAILURES-0A-r2-initial.txt.gz. Additional candidate lineage regression initially expected a later diagnostic but rejected earlier as incompatible_bundle: 1 failed / 111 passed; raw output retained in FAILURES-0A-r2-third.txt.gz, then test changed to isolate description lineage. No failure was treated as hardware or runtime evidence. Final exact-source image commands/results and identities follow below.

Final source revision also added an engine/device negative: initially 1 failed / 113 passed because the changed device rejected earlier at profile_mismatch. Raw output is FAILURES-0A-r2-fourth.txt.gz. The final test makes the mocked profile device consistent, isolating unsupported engine execution. All raw failures are retained; no TypeError is treated as evidence.

### Final reproducible lane

Inspected Dockerfile.tests and scripts/run-unit-tests.sh before execution; Dockerfile wiring needed no change. Built **once** after final executable-source edits:

```sh
docker build -f Dockerfile.tests -t py-audio-box-0a-fix-r2-tests .
docker run --rm --network none --cpus 2 --memory 3g \
  py-audio-box-0a-fix-r2-tests python -m pytest \
  tests/contracts tests/test_capabilities_and_metadata.py \
  tests/test_clone_prompt_compatibility.py tests/test_clone_prompt_identity.py \
  tests/test_customvoice_feature_flag.py tests/test_voice_cpu_batch.py \
  tests/test_voice_synth_selection.py \
  -m 'not integration and not redis_integration' -q
```

Build exit 0. Selected final suite exit 0: **185 passed in 29.59s** (114 contract tests plus 71 affected legacy tests). Deterministic regeneration is `test_determinism`, run in the disposable writable container; no user mounts. Selected legacy tests ran once on the final exact executable source; no full offline suite or remote e2e was run. Image Python 3.11.17, pytest 8.4.2, jsonschema 4.26.0. Small development-only contract diagnostics used the existing temporary Python 3.14 validator venv; those are not substituted for the final image result.

Final image: `sha256:322ce89410c82841862177a8fed41458fefe2de49e8c43fb0c0bd914f490f868`. Independently enumerated Dockerfile COPY roots (`contracts`, `lib`, `apps`, `tests`, `benchmarks`, `scripts`, five launcher/config files), excluding ignored Python bytecode: **137 host files / 137 image files, zero SHA256 mismatches**. Canonical sorted/indented path-to-hash inventory byte digest: `ca7c5a8b7012f8b792eac8259badfb11326b1e8caba2b8ee1853e4a98ec27bb6`. Documentation is not copied by Dockerfile.tests; later evidence-only documentation edits do not change the tested executable image. Fresh verification container used the same bounded flags, no mounts, and removed itself. `git diff --check` passed.

Independent Sol-low re-review must inspect the actual final commit, backend mappings/schema/oracle/diff and all original counterexamples with new valid detached envelopes. R2-1/2/3 and adjacent settings/deadline/revision/manifest relations each require their own closure; none is waived by this implementation response. No self approval. Integration and E2–E5 gates remain parent-owned.

### Exact correction file inventory

```text
contracts/fixtures/v1/artifact.json
contracts/fixtures/v1/capabilities.json
contracts/fixtures/v1/expected.json
contracts/fixtures/v1/operation-requests.json
contracts/fixtures/v1/production-100.json
contracts/fixtures/v1/production-30.json
contracts/fixtures/v1/transcript-project_render.json
contracts/fixtures/v1/transcript-speech_generate.json
contracts/fixtures/v1/transcript-voice_design.json
contracts/fixtures/v1/transcript-voice_prepare.json
contracts/generate.py
contracts/scenarios.py
contracts/v1/errors.json
contracts/v1/operations.json
contracts/v1/schema.json
contracts/validate.py
docs/agent-interface/phase-0/CONTRACTS.md
docs/agent-interface/phase-0/EVIDENCE-0A.md
docs/agent-interface/phase-0/FAILURES-0A-r2-fourth.txt.gz
docs/agent-interface/phase-0/FAILURES-0A-r2-initial.txt.gz
docs/agent-interface/phase-0/FAILURES-0A-r2-third.txt.gz
docs/agent-interface/phase-0/RESPONSE-0A-r1.md
docs/agent-interface/phase-0/RESPONSE-0A-r2.md
docs/agent-interface/phase-0/TESTS-0A-r2-final.txt
tests/contracts/test_corrections.py
tests/contracts/test_frozen_boundary.py
```

The final staged whitespace check flagged trailing spaces emitted by pytest in the raw initial failure log. Failure logs are stored as deterministic gzip files, preserving exact decompressed bytes; source/docs checks then pass without rewriting raw evidence. No executable-source change or test rerun was needed.
