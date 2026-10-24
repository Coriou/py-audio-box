# Managed Qwen provisioning — r264 implementation handoff

2026-10-24. SOURCE candidate implemented; fresh parent-owned Sol review required before adoption. No actual model availability, hardware, acoustic, installation, readiness or release qualification.

## Pins and scope

Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-008ea39f`, branch `t3/qwen-provisioning`. Clean entry HEAD verified exactly **`6b9f81a1a7c972aa26227ccbf975efa7c89839ea`** before editing. Source commit **`f0b0217cf8575f8ed0a1d94018538a529269dc8f`**. This report is committed separately.

Read r188 assignments 2/4 and milestone context, full r259/r260/r261/r262 and r255/r256, current provisioning/assets/script, Qwen execution/backend/checkpoint capability code and shared Pocket file/reference verification. No applicable worktree/checked ancestor AGENTS.md found. A bounded memory registry lookup returned no relevant hit and supplied no premise. Inspected root/service conftests; direct unittest avoids their maintenance and listener fixtures.

Authorization explicitly includes extraction of pure local verification in worker/engines/qwen.py. Changed only that file, service/provisioning.py, lib/tts_backends/assets.py, scripts/provision-tts.py and focused new tests/service/test_qwen_provisioning.py. No new shared module or distribution manifest entry is needed: all production modules are already allowlisted. No lifecycle/Core/recovery/installer activation or delegation.

## Implemented contract

Managed Qwen uses the existing lock, wait/transfer budgets, cancellation, stage, partial discard, complete-file resume, expected hash/size verification, seal, immutable directory and atomic active-pointer replacement. Nested snapshot directories extend this same path for speech_tokenizer; sealing covers every descendant. Failures before pointer replacement preserve the previous active pointer and bundle. Trusted callback downloaders retain their existing responsibility for transfer bounds. The production downloader receives Qwen's allowed HTTPS authorities and uses the existing redirect policy and exact-size stream cap. No second acquisition framework or implicit discovery/download was added.

The descriptor is closed: schema_version/backend/language/checkpoint/revision/license/access/versions/files plus exactly allowed_origins and execution. Each file has exactly path/bytes/sha256/origin/revision. Revision is one operator-provided lowercase 40-hex immutable revision, equal on every file. HTTPS origins are explicit authorities; credentials/query/fragment and unapproved redirects refuse. Paths are bounded relative components with .json/.safetensors/.txt/.model suffixes; traversal, duplicate files, file/directory conflicts and executable/pickle suffixes refuse. Limits: 256 files, 8 GiB/file, 16 GiB total, JSON assets at most 16 MiB each, descriptor/raw manifest at most 65536 bytes. Small inert tests do not allocate these maxima.

The supported managed layout requires config.json, generation_config.json, tokenizer_config.json, tokenizer.json, root safetensors and speech_tokenizer/config.json, speech_tokenizer/preprocessor_config.json, speech_tokenizer safetensors. Additional tokenizer vocabulary, merges, special-token files, JSON indices and safetensor shards can be explicitly listed. This is a deliberate bounded local snapshot layout, not an automatic hub snapshot discovery implementation or a promise to accept every upstream packaging variant. JSON indices must reference listed safetensors. All config/tokenizer bytes are downloaded verbatim under their independently supplied expected size/hash; no Qwen model configuration is generated, rewritten or inferred from its checkpoint name.

Runtime versions must name exactly qwen-tts, torch, transformers, tokenizers and safetensors, with exact equality to the trusted runtime_versions input. No real upstream pin or version value was invented. License/access are explicit nonempty operator metadata, not inferred consent or proof of rights. Full installed dependency closure remains a qualification obligation.

Qwen's deliberate language contract uses **English/French**, without translation or fallback. Managed Qwen refuses en/fr; existing Kokoro/Pocket remain en/fr. lib/tts_backends/assets.verified_bundle dispatches Qwen to the Qwen-specific verifier rather than globally widening the old language validator. Declaring English/French expresses trusted operator selection; these fixtures establish neither model language truth nor audio quality.

Execution declarations contain exactly sample_rate/languages/presets/devices. Lists are bounded, typed and unique. Device tuples are explicit cpu/mps/cuda:N with concrete dtype and attention; impossible CPU precision, MPS bf16 and non-CUDA flash attention declarations refuse without probing hardware. The existing hashed config capability helper remains authoritative for checkpoint task and instruction permission. Managed non-CustomVoice checkpoints cannot declare presets. Present talker lang_id/spk_id and sample_rate declarations must agree with requested capabilities; absence does not manufacture an upstream declaration. Bounded JSON parsing refuses remote URL/config file references, custom code maps/pipelines and trust_remote_code activation; explicit file/path references must name listed local assets. These checks do not implement upstream architecture or tensor-semantic validation; an operator's inconsistent architecture may still be refused by the actual offline loader.

## Shared verification and preserved execution gates

worker/engines/qwen.verify_bundle is the extracted pure local verifier, called by BOTH provisioning and actual verify_inputs, as well as the Qwen branch of verified_bundle. It authenticates the raw manifest hash and identity, owned no-follow files, exact complete sizes/streaming hashes/stable metadata, required config/safetensors, and no unlisted files/symlinks. Managed JSON/declaration consistency is checked before active replacement. It imports no Torch or upstream model and takes no HardwareInventory or reference mapping.

verify_inputs still requires the actual typed execution, trusted HardwareInventory, available/supported indexed device with probe provenance, topology checks, authorized bundle mapping and exact selected capability tuple. Clone still requires the existing exact authorized voice/revision/checkpoint/revision/transcript binding and authenticated complete PCM16 reference. execute still revalidates before backend construction and uses existing named routes, explicit device/precision/attention/offline forwarding, measured output bounds and close. No fabricated inventory or reference is used to activate a bundle. Tests inject a clearly labelled trusted test inventory only when entering execution.

Storage bundle ID hashes the FULL provisioning descriptor, including origin policy. Runtime manifest strips only allowed_origins (and Pocket's existing descriptor config). active.json manifest_sha256 hashes the exact serialized runtime manifest bytes. Tests show a policy-only change creates a distinct bundle directory while preserving the runtime manifest digest. Neither identity is a qualification receipt; existing actual-receipt readiness matching remains unchanged.

## Executed checks

Every Python invocation used `/private/tmp/pab-0c-venv/bin/python -I -B`, with explicit source/test paths. No pytest collection.

Final functional command:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B -c "import sys,unittest; sys.path.insert(0,'/Users/ben/.txcode/worktrees/py-audio-box/t3-008ea39f'); s=unittest.defaultTestLoader.loadTestsFromNames(['tests.service.test_qwen_provisioning','tests.service.test_provisioning','tests.worker.test_engine_qwen','tests.worker.test_engine_pocket','tests.worker.test_engine_kokoro','tests.service.test_provisioned_engine']); r=unittest.TextTestRunner().run(s); sys.exit(not r.wasSuccessful())"
```

**67 PASS (0.905 s)**: 8 new Qwen provisioning cases plus 59 existing provisioning/engine regressions. New coverage includes both exact languages; provision → actual shared verifier → actual verify_inputs → injected backend/AudioResult; unchanged config bytes, raw-manifest/full-descriptor identities, seal/idempotent reuse, missing hardware and unsupported language refusal, interrupted resume, seal retry, bad transfer/runtime mismatch preserving old active, descriptor paths/origins/revisions/bounds, hashed task/config/capability contradictions before activation, independent config pin refusal, and existing redirect/exact-stream bounds. Existing Qwen tests retain clone reference and task/direction/selected-device/output negatives; Pocket/Kokoro checks pass.

Separate fresh-interpreter command:

```sh
/private/tmp/pab-0c-venv/bin/python -I -B tests/worker/test_qwen_import_seam.py -v
```

**3 PASS (0.239 s)**: complete real worker/loader under root-only, legacy and extracted archive contexts, with injected upstream/ML and exact device forwarding. Total final evidence: **70 selected cases PASS across these two commands**. git diff --check PASS.

An earlier combined 70-case command had 67 passing functional checks and three import-seam assertion failures: their fresh-import precondition rejects NumPy already imported by prior audio tests. Running the unchanged import-seam file in a fresh interpreter passed all three. No production workaround or test weakening was added. Diff inspection also caught and corrected an overbroad intermediate text replacement in Pocket's required-assets condition before the final regression run.

Disk observed 3,588,264 KiB free, below 6 GiB heavy-work gate. Only tiny inert fixture bytes and the existing small source archive were used. Fixture revision is forty a characters, versions are fixture, authority is fixture.invalid, and safetensor bytes are deliberately inert text; none is a real model pin or semantic tensor fixture. No actual ML/hardware, external network/download, dependency install, Docker, listener, process-crash campaign, operational cleanup, HELD action or delegation. Existing bounded downloader tests use disposable inert children; no inference process starts. Only test-owned temporary teardown occurred.

## Parent-owned gates

Fresh independent Sol source review precedes adoption. Supply reviewed real snapshot pins, complete compatible config/tokenizer/model bytes, license/access acceptance and real runtime closure separately. Actual offline loader compatibility, English/French support and listening, model-child/helper descendant ownership and confirmed reap, installed provisioning/HTTP/MCP journeys, constrained CPU/CUDA/native Metal resource/performance/cancel/restart evidence and integrated release approval remain unrun. This source candidate does not wire Qwen into production lifecycle or activate an installation/readiness profile.
