# Lot 0A independent adversarial review — round 4

**GO for 0A only**, exact candidate `26f8e08ed9db4935dad1f5c4916e885d9b392645`, branch `t3/speech-output-boundary-fix`, inspected read-only in `/Users/ben/.txcode/worktrees/py-audio-box/t3-66ad7ec3`. Contract/schema/fixture/worker revision **1.3.0**. Independent Sol-low review, 2026-10-16. No blocking finding remains from this bounded review; no earlier objection is waived.

This is local candidate E1 approval of the executable contract pack. It is not integration, Phase 0 exit, runtime, hardware, acoustic or final Astra global approval.

## R3 disposition

**R3-1 closed at E1.** `contracts/v1/schema.json` requires a closed speech-settings preimage on immutable versions. `contracts/validate.py:205` verifies its canonical digest and mode; recursive intrinsic validation reaches versions inside candidates/results/projects/plans. `frozen_plan` freezes separate prospective reuse configuration/settings for prepare/design, validates their backend configuration, and publication compares them exactly with the candidate. Preparation settings and VoiceDesign generation settings remain distinct. Speech copies the version's supplied settings and requires equality with resolved settings; `contracts/scenarios.py` no longer replaces a supplied version digest with guessed defaults.

Independent detached-wire probes started from passing checked-in prepare/design transcripts. Arbitrary digest and changed preimage rejected `settings_mismatch`; changing the preimage and recomputing its digest rejected `reuse_configuration_mismatch`. The worker result envelope still passed before publication rejection. Standalone voice_version and containing voice_candidate, voice_results, project and worker_plan validation independently rejected corrupted digests with `settings_mismatch`. The suite additionally exercises simultaneous output/reference configuration replacement against the approved plan and the no-repair builder behavior.

Independently tested **two different non-default settings versions through each of prepare and design**: English seed 47 / temperature .31 and French seed 823 / temperature 1.17, both top_k 19. Each passed publication, exact typed version-get response → speech plan → worker result → publication. The original version object remained byte-equivalent as JSON, its digest was unchanged, and the speech plan contained the exact snapshot/settings. The selected suite also passes its separate seeds 37/901 cases, including all typed save/read exchanges. Save is a contract transcript, not an implemented persistence operation.

**R3-2 closed at E1.** Asset metadata now requires measured subtype. Publication requires prepare/design WAV PCM_16 and render WAV PCM_24 with frozen channels/rate and exact rational frame relation (`frames*48000 == length_ticks*sample_rate`). No rounding/truncation allowance is inferred.

Independent soundfile-generated, reopened and hashed truthful media reproduced the old counterexamples in detached valid envelopes. For both prepare and design, FLAC PCM_16, WAV PCM_24 and WAV FLOAT now reject `output_format_mismatch`. A true 146400-frame stereo 48 kHz PCM24 render passes; true truncated 4800-frame mono, 146399-frame stereo, 44100 Hz and mono-only replacements reject `output_policy_mismatch`. The suite also covers wrong render container/subtype. These were semantic ContractError rejections, not TypeErrors or structural-validation failures counted as fixes.

All four checked-in transcripts pass worker/publication and typed catalogue exchanges in the suite. Independently recovered their actual synthetic output bytes: prepare/design/speech match impulse.wav hash and decoded PCM16 metadata; render matches independently generated 146400-frame stereo PCM24 silence hash and decoded metadata. The corrected render fixture satisfies its frozen output policy. These bytes demonstrate synthetic output conformance, not speech generation or actual rendering.

## Preserved prior findings and Astra matrix

Inspected the actual cumulative pack and R3 correction delta, schema, catalogue, validators, scenarios/generator, fixtures and tests, plus backend selector/reference/settings mappings in lib/tts.py, Qwen, Kokoro and Pocket paths. Read PRODUCT, IMPLEMENTATION-START, canonical shared contracts and Phase 0, ORCHESTRATION-PROTOCOL, lot 0A, full R1/R2/R3 Sol reports, authoritative targeted Astra review, all three responses and current CONTRACTS. No applicable AGENTS.md was found in the relevant tree/checked ancestry.

The full original Astra boundary regression matrix remains required and passed in the executed contract suite; no matrix obligation was discarded:

- F1 actual byte upload/range retrieval remains closed: strict base64/hash/size, authorization, explicit bounded end-exclusive ranges and exact reconstruction.
- F2 missing executable workflow inputs is now closed at E1, including R2-1 selectors and the remaining R3-1 saved settings preimage. Revisions/candidates/version/history reads and typed inline frozen plans remain available.
- F3 full attempt ownership/fencing, authenticated owner, generation, staging, hash, cancellation/reap and replacement checks remain closed. Death/reap/peer evidence is trusted input, not proven runtime behavior.
- F4 recursive intrinsic resolution remains closed; cross-object resolution is additionally bound to stored authority. F5 immutable takes/assets/versions across ordinary edits and F6 exact maximum/atomic ripple overflow remain closed.
- R2-2 stored resolution/policy/profile binding and R2-3 exact derived dependency/ACL authority remain closed. Detached simultaneous identity changes, received rehashed plans, fallback widening, missing reason, explicit precision, effective auto, profile drift, deadline/retry, omitted/unauthorized/extra/duplicate assets, muted render beds and scoped byte transfer remain covered.
- Engine-specific Qwen/Kokoro/Pocket settings/selectors, clone conditioning/checkpoint/reference, design lineage and Base realization reuse remain explicit. Publication still binds manifest/result IDs, bytes, attempt, neutral/render revision and frozen resolution. R3 adds result representation/settings conformance without weakening those checks.
- Legacy source/default hash checks and compatibility coverage pass; no application source changed in this correction. Distinct-root, non-destructive migration policy and pending legacy-job/provenance/rollback boundaries remain intact. No migration executed.

## Verification and reproducibility

Verified candidate HEAD and clean status before/after tests. Inspected Dockerfile.tests, .dockerignore and scripts/run-unit-tests.sh before execution. Independently inspected image ID **`sha256:062f4e93f561a0181c94fbcfd34d266790b3634ab4f5a623639f054b604cef5a`**, then compared both path sets and SHA256 for every Docker COPY-root file, excluding ignored bytecode/DS_Store: **139 source files / 139 image files, zero mismatches**. Documentation is outside the image COPY roots. Reuse did not rely on the implementer's test claim.

Executed the sole selected test lane:

```sh
docker run --rm --network none --cpus 2 --memory 3g \
  py-audio-box-0a-fix-r3-tests python -m pytest \
  tests/contracts tests/test_capabilities_and_metadata.py \
  tests/test_clone_prompt_compatibility.py tests/test_clone_prompt_identity.py \
  tests/test_customvoice_feature_flag.py tests/test_voice_cpu_batch.py \
  tests/test_voice_synth_selection.py \
  -m 'not integration and not redis_integration' -q
```

**214 passed in 42.49s**, exit 0 (143 contracts + 71 selected legacy). Includes deterministic regeneration only in the disposable writable container. No user/source mounts, no rebuild or full-suite claim. Independent diagnostics `/tmp/pab-r4-inventory.py`, `/tmp/pab-r4-probes.py` and `/tmp/pab-r4-extra.py` used sequential disposable containers with the same bounds; diagnostic Python entered via stdin, no mounts. All completed successfully. `git diff --check` passed. No detached alias mutation or exception-type shortcut was used as closure evidence.

Only this report was written in the root checkout. Candidate source, root research, ledger and ticket were preserved. No source repair, commit, integration, subagent, inference/models, private audio, STT/Voicebox/app change, paid remote e2e, cloud, push, deploy or PR action occurred. `scripts/run-e2e-test.sh` was never run.

## Limits and handoff

All conclusions are **E1 static/schema/offline**. Control must actually decode/measure and hash bytes before supplying the trusted current-attempt manifest; a worker JSON claim or present_hashes alone is not media verification. Durable publication/adoption/recovery, credentials, OS ownership/reap and races remain E2/runtime work. Real engines/release images/hardware qualification remain E3, browser/agent-host/token behavior E4, listening/production acceptance E5. Synthetic CPU/CUDA/Metal fixtures remain unqualified. No memory, speed, audio quality or universal installation claim follows.

**GO for this exact 0A candidate at E1**, with R3-1/R3-2 closed and prior closures preserved. Integration owner must verify the integrated SHA and affected consumers. Final Astra global review remains separate and required; this report does not approve the whole phase or product.
