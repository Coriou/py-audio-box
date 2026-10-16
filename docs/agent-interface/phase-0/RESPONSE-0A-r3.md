# Lot 0A R3 correction response

Implementation candidate only; **independent Sol-low re-review required**, no self-GO. All evidence is E1. R3-1 and R3-2 are the bounded brief; no earlier objection is waived. The R2/Astra selector, frozen authority/profile/fallback, dependencies/ACLs, deadline/retry, lineage, manifest/attempt/revision and F1/F3/F4/F5/F6 checks remain in place and in the selected suite.

## Source and scope

Fresh clean start: `99d49a26c5e5c300e873ec7656f8fbe855038c71`, worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-66ad7ec3`. Initial branch observed `t3/66ad7ec3`; app-owned branch subsequently observed `t3/speech-output-boundary-fix` (no branch rename command issued here). Exactly the requested commits were cherry-picked in order:

- `9f1027b95229ed30f12d918b27bba989fbfe5d9d` → `029d643050a8048a211c31851e34b0d644a0fd24`.
- `aa9a61a183545df65c96b5ccc5df071aa464c714` → `50a7d4ab28541db418565783ef22c4a9c4b22d7e`.
- `ca0ad376e237f6bf75777cc6196b34688af90a2c` → `673130bc3f3f848ac9ea43e63da74275bdf629d9` (correction base).

Read PRODUCT, entry point, canonical shared/Phase-0 plan, protocol, ticket, R1/R2/R3 Sol and targeted Astra reviews, prior responses and CONTRACTS. No applicable AGENTS.md found in ancestry/relevant tree. Ownership stays contracts, contract tests and phase-0 documents. Dockerfile.tests/script inspected; wiring unchanged. No application/legacy, root ledger/ticket/review, runtime/DB/Compose, private audio, cloud, model, STT/Voicebox, PR/push/deploy or subagent work. Remote e2e script never run.

Current schema/catalogue/fixtures/worker protocol: **1.3.0**, API/directory namespace v1. Legacy compatibility ledger and historical reports/responses remain historical and unchanged. CONTRACTS.md is current consumer guidance; R2 response's backend support table remains applicable. No deployed migration is claimed.

## Exact finding mapping

| Finding | Implementation | Entry-point evidence |
| --- | --- | --- |
| R3-1 missing reusable preimage | Required closed speech `settings` in each immutable version; canonical digest and mode verified recursively in containing resources. Speech uses the exact version-get object/settings; generator no longer rewrites digest or guesses backend defaults. | `test_containing_resources_verify_settings_preimage`, `test_speech_builder_never_repairs_an_invalid_version_digest`, prior complete catalogue transcript tests. |
| R3-1 approved prospective configuration | Hashed prepare/design plan freezes separate `reuse_configuration` and `reuse_settings`; preparation/VoiceDesign generation settings stay distinct. Candidate publication compares complete configuration/settings and retains identity/transcript/conditioning/lineage checks. Other operation reuse fields are null. | Detached arbitrary digest, mismatched preimage, rehashed changed settings and simultaneous output/configuration ID replacement tests reach valid worker envelope then reject semantically at publication. |
| R3-1 exact non-default reuse | Two explicit non-default settings sets (English seed 37 / temperature .4; French seed 901 / temperature 1.2, both top_k 17), each through prepare and design, typed save/version-get, exact snapshot speech worker/publication. | `test_nondefault_exact_save_version_speech_publication` (four cases); original full transcripts retained. |
| R3-2 reusable representation | Required measured subtype in typed asset metadata; current-attempt verified manifest is control-owned decoded-media evidence. Prepare/design publication requires WAV PCM_16. | True tiny WAV positive, true FLAC PCM16 and WAV PCM24/FLOAT negatives for both operations; rejected FLAC also reproduces unusable reference on reuse. |
| R3-2 render policy | Publication requires WAV PCM_24, exact frozen frames/rate/channels. Frames use exact rational ticks relation; no hidden rounding. Render transcript corrected to true 146400-frame stereo PCM24 silence/hash/metadata. | Passing true exact WAV baseline, then detached true 4800-frame mono render, wrong frame count/rate/channels/container/subtype all reject `output_policy_mismatch`. |

`contracts.media` only constructs synthetic silence bytes with soundfile and measures them. Tests independently reopen bytes with soundfile and verify their hash/metadata. It is not a renderer/inference implementation. Runtime must actually verify bytes and decoded properties before supplying the trusted manifest; the oracle cannot establish truth from arbitrary worker JSON. `present_hashes` is trusted hash evidence, not a decoder. Immutable save semantics preserve all candidate content except allocated version ID.

## Development diagnostics

Development-only local contract checks used the existing `/tmp/pab-0a-contracts-venv` (Python 3.14) with temporary numpy/soundfile dependencies; final acceptance below uses the pinned Docker image. The initial generation lacked local numpy; installed it in that temporary environment. Initial suite: 20 failed / 94 passed (incomplete generation, old helper assumptions and version constant). Next: 3 failed / 111 passed (earlier semantic rejection diagnostics). Next: 2 failed / 140 passed (new configuration negative correctly rejected as reference_mismatch before expected diagnostic). Exact raw logs retained in deterministic gzip files `FAILURES-0A-r3-dev{,2,3}.txt.gz`. Focused development follow-up: 95 passed, followed by final targeted checks after final regression/source adjustments. No TypeError or schema-only rejection is counted as publication closure.

## Final isolated verification

Inspected Dockerfile.tests and scripts/run-unit-tests.sh. Built the final executable source once; no Dockerfile change required:

```sh
docker build -f Dockerfile.tests -t py-audio-box-0a-fix-r3-tests .
docker run --rm --network none --cpus 2 --memory 3g \
  py-audio-box-0a-fix-r3-tests python -m pytest \
  tests/contracts tests/test_capabilities_and_metadata.py \
  tests/test_clone_prompt_compatibility.py tests/test_clone_prompt_identity.py \
  tests/test_customvoice_feature_flag.py tests/test_voice_cpu_batch.py \
  tests/test_voice_synth_selection.py \
  -m 'not integration and not redis_integration' -q
```

Build exit 0; suite exit 0: **214 passed in 42.20s** (**143 contract + 71 selected legacy tests**). Raw final output: TESTS-0A-r3-final.txt. This includes deterministic regeneration in the disposable writable container, with no mounts. Focused and selected legacy checks ran together once on final source. No full suite, real engine or remote e2e result is claimed. Earlier focused development follow-ups each passed 95 tests; the final source adds the invalid-digest no-repair regression.

Image: **`sha256:062f4e93f561a0181c94fbcfd34d266790b3634ab4f5a623639f054b604cef5a`**, tag **`py-audio-box-0a-fix-r3-tests`**. After the suite, a separate sequential verification container used the same `--rm --network none --cpus 2 --memory 3g`, with stdin only and no mounts. Compared every file under Docker COPY roots (`contracts`, `lib`, `apps`, `tests`, `benchmarks`, `scripts`, `run-cpu`, `run`, `run-direct`, `run.ps1`, `pyproject.toml`), excluding Docker-ignored bytecode/DS_Store. **139 host files / 139 image files, zero SHA256 mismatches**. Sorted/indented path-to-hash inventory byte SHA256: `97d5666d822f63853256b1fa4e67b288241f03980c70805c657fff6ea6ba8c34`. Verification compared both key sets and each digest, not only counts. All containers exited/removed. Documentation/evidence is not copied into this test image; only documentation was written after that comparison. `git diff --check` passed.

## Remaining gates

Independent Sol-low must re-review the exact committed candidate, original R3 counterexamples and earlier closures using valid detached envelopes. No self-GO, Phase 0 exit or integration approval. Actual decoding/adoption races, durable store/credential/reap behavior, inference/hardware identity, agent/browser acceptance and acoustic approval remain E2–E5 gates. The fixture media proves truthful metadata/byte relations only. No objection was waived or converted into a runtime claim.

## Exact R3 correction file inventory

The three requested precursor commits are separate from this correction inventory. Final commit SHA is supplied in the handoff (a commit cannot contain its own hash).

```text
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
contracts/media.py
contracts/scenarios.py
contracts/v1/errors.json
contracts/v1/operations.json
contracts/v1/schema.json
contracts/validate.py
docs/agent-interface/phase-0/CONTRACTS.md
docs/agent-interface/phase-0/FAILURES-0A-r3-dev.txt.gz
docs/agent-interface/phase-0/FAILURES-0A-r3-dev2.txt.gz
docs/agent-interface/phase-0/FAILURES-0A-r3-dev3.txt.gz
docs/agent-interface/phase-0/RESPONSE-0A-r3.md
docs/agent-interface/phase-0/TESTS-0A-r3-final.txt
tests/contracts/requirements.txt
tests/contracts/test_corrections.py
tests/contracts/test_frozen_boundary.py
tests/contracts/test_output_boundary.py
```
