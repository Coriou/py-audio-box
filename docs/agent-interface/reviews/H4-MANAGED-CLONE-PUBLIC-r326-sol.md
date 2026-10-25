# Managed native clone public journey — r326 Sol implementation

2026-10-25. **The bounded r323 Lane 1 public SOURCE journey is implemented and passes the finite selection below. Fresh independent Sol review remains required; no self-adoption, installed/hardware qualification or release GO is issued.** This round delivers onboarding rather than another history-only prerequisite.

## Pins and authority

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-91254c6f`, branch `t3/managed-native-clone-onboarding`. Clean entry/base `9ee09a2e76d90b3215247e420330b60fbe4ad375`. SOURCE/tests commit **`f8bc0085221158290f67ad01b262166d04a054cc`**. This report is the following documentation-only commit.

Read full r323, r324 and r325, PRODUCT, and current typed preparation/results/save, Pocket owned-resolution, native protocol/context/supervisor/runtime/production/VoiceStore, admission, publication, retained-history and public client consumers. No applicable AGENTS.md appeared in worktree/parent discovery or checked ancestors through `/`. Memory registry search had no relevant hit; no memory-derived technical conclusion. No root writes, merge, push, subdelegation or self-adoption.

Only authorized tiny inert ASGI/fake TLS/filesystem/SQLite/audio/fake-process/child-function observations ran. Disk observation: 2,782,576 KiB available, below 6 GiB. Fixture free-space and hardware declarations are injected SOURCE evidence, never an actual gate waiver. No network/listener, Docker, install/download, actual child/ML, crash, HELD, STT or operational cleanup occurred. Fixture-owned temporary teardown only. Tests bearing historical `actual_child`/`actual_tls` names use the inspected child-function/fake-channel/backend seams here, not an actual native child, real TLS peer or model load.

## Implemented product path

One provisioned original native **clone** template for the authenticated principal, with exactly matching explicit `voice_prepare`/`prepare` profile compatibility, enables the managed path. Preset templates acquire no clone permission. Missing or ambiguous clone capability refuses and discovery omits preparation. This uses the existing enrollment/configuration boundary and one `qwen-mps` profile; it neither provisions a model nor qualifies Metal.

The supported preparation mode is deliberately finite: a complete canonical mono PCM16 WAV, at most 1 MiB and 60 seconds, with `in_tick=0` and exact integral `out_tick=frames*48000/sample_rate`. Supply the exact nonblank user transcript as `reference_text` and a separately authored nonblank `audition_text`. Authored strings, including surrounding whitespace/newlines, are preserved. Fractional tick boundaries, noncanonical headers, non-mono/non-PCM16 inputs, partial segments, absent transcript/audition and x-vector preparation refuse. The canonical media upload remains its existing wider bounded operation; preparation has this narrower supported mode. No transcription, normalization, resampling or inferred transcript is introduced.

Canonical usage, using the already shipped HTTP/CLI/MCP operations:

1. Read `/v1/capabilities` and `/skeleton` configured capability; upload PCM through `POST /v1/assets` (thin client `HTTP.upload_file`, existing CLI/MCP upload helpers).
2. Call `voice_prepare` with the owned opaque asset ID, complete segment ticks, exact transcript, separate audition and the configured identity. Use a fresh idempotency key.
3. Read the job and `voice_results`. The candidate has distinct verified `prepared_reference` and `generated_audition` roles. Reference provenance binds original upload ID/hash, exact segment and source transcript; audition provenance binds typed execution and separate spoken text.
4. Call `voice_save` with candidate/job IDs, target voice ID, CAS revision and idempotency key. Core stores the immutable clone version and reference hash through the existing VoiceStore.
5. Generate with the returned immutable version ID. Retrieve the artifact and bounded asset content. Under the same supervisor boot/trust, cleanly close/reopen and replay the original preparation/save/generation keys without another backend launch or per-voice enrollment edit.

No new public request schema was necessary: existing closed preparation requests already contain transcript and audition fields, and existing role contracts cover the pair. The missing private variation is `QwenPreparation`, a closed typed native descriptor. Its audition reuses the existing Qwen generation relation through an internal derived validation view; it does not create a synthetic job or success row. Existing pure candidate and role-metadata builders are reused. Linux Qwen preparation and VoiceDesign remain unavailable.

## Trust, publication and lifetime

The Core writer resolves owned uploads/saved versions and verifies exact bounded bytes before admission. `owned_clone_context` copies model, bundle, settings, profile, hardware and native boot trust exclusively from the original trusted clone template, varying only managed reference metadata, authored transcript and immutable catalogue identity. Reuse checks reference hash, configuration, transcript, settings, model/identity and provenance against the owned reference and original settings.

Native `register_snapshot` independently reconstructs preparation and saved-clone execution from `_preset_templates`, including deterministic output-role IDs. It never uses `registrations` working allocations as derivation authority. Native transfer verifies reference size/hash/PCM; start independently verifies sealed model inputs and exact reference/transcript mapping before backend construction. Child-function verification preserves the complete canonical reference and synthesizes only the explicit audition text.

The native wire still transfers one bounded audition WAV. Its proposal binds the complete pair, candidate, attempt and exact provenance. Control materializes `reference.wav` from its already verified immutable detached input; it does not trust reference bytes supplied as a second worker result. Both fixed stage members are handed off under the existing exclusive import intent/completion fence. The new `import-reference` temporary has captured file identity and no-replace owner publication. Partial handoff, substituted reference identity or incomplete import leaves uncertainty closed; no repair, publication or reclamation is inferred from a visible file. Core independently measures both files, compares the reference to the original bytes, and uses its existing pair adoption/publication transaction. Preparation charges four reference copies plus the existing audition/control allowance; generation charges are unchanged.

Clean completed preparation history adds only the pair variation to existing retained schema/Authority/admission/command/receipt/import/publication/inventory checks. It verifies original ownership and detached reference, exact retained stage reference, both output digests and the audition transport. Clean saved-clone generation uses r324's existing clone history support. Failed/cancelled/unknown histories, changed boot/trust, uncertain import, unknown evidence, unsupported schema and existing capacity limits remain refusals. No reference garbage collection or renewal policy is added; references remain retained before and after confirmed reap.

## Final direct unittest receipt

Read `tests/conftest.py` and `tests/service/conftest.py`; direct unittest avoids pytest autouse maintenance/listener fixtures. Every test observation used `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit absolute worktree plus `tests/service` sys.path, version assertion `(3,14,7)`, and `TextTestRunner(verbosity=2,failfast=True)` with failing results mapped to exit 1.

Actual runtime: `3.14.7 (main, Aug 5 2026, 10:29:49) [Clang 17.0.0 (clang-1700.6.4.2)]`.

**One final finite selection: 102 PASS / 50.855s / exit 0 / 0 failures / 0 errors / 0 skips.** Session `90376`; initial chunk `7c16cc`; intermediate chunks `0beda3`, `6b8a31`; terminal chunk `6f170d`. Per-block and total counts were asserted before execution, and `testsRun == 102` was required for exit 0. No source/test edits followed this run; no broad unchanged repeat followed it. `git diff --check` passed.

Selection (module names below have the `tests.` prefix):

- `release.test_native_managed_clone`: all **11** new tests (3 public journey/history/cancellation; 8 derivation/proposal/ownership/byte/import/capability tests).
- `release.test_native_onboarding.NativeOnboardingTests`: all **6**.
- `service.test_native_runtime.NativeRuntimeTests`: all **26**.
- `worker.test_native_qwen.NativeQwenTests`: all **4**.
- `worker.test_native_protocol.NativeProtocolTests`: all **5**.
- `release.test_native_history.NativeHistoryTests`: **11** named methods: `test_completed_job_reopens_twice_and_retains_exact_artifact`, `test_changed_boot_refuses_without_learning`, `test_unknown_schema_refuses`, `test_changed_schema_refuses`, `test_unconfirmed_ack_refuses`, `test_wrong_native_terminal_receipt_refuses`, `test_changed_authority_refuses`, `test_missing_import_completion_refuses`, `test_changed_publication_refuses`, `test_unknown_staging_evidence_refuses`, `test_context_primitive_substitution_refuses`.
- `release.test_native_history.NativeCloneHistoryTests`: **5** named methods only: `test_clone_keeps_exact_reference_and_transcript_across_replay`, `test_changed_owned_reference_refuses`, `test_changed_detached_reference_refuses`, `test_missing_owned_reference_refuses`, `test_foreign_owned_reference_refuses`. Inherited history methods were not selected.
- `service.test_pocket_preparation`: all **13** module-loaded tests.
- `service.test_installed_pocket.DiscoveryTests`: **1**; `service.test_installed_preset.InstalledPresetTests`: all **3**.
- `service.test_qwen_lifecycle.QwenLifecycleTests`: **6** named methods: `test_saved_reference_exact_forwarding_publication_and_restart`, `test_refusals_before_allocation`, `test_stale_probe_refuses_resolution_and_dispatch_without_expiring_history`, `test_supervisor_independent_registration_and_prejournal_verification`, `test_limits_before_child_backend_and_exact_cuda_reservation`, `test_http_cli_mcp_saved_reference_generation_and_authorization`.
- `clients.test_production_clients.ProductionClientTests`: all **3**; `service.test_production_api.ProductionAPITests`: all **5**.
- `worker.test_native_supervisor.NativeSupervisorTests`: **3** named methods: `test_input_complete_replay_conflict_gap_and_hash`, `test_clone_conditioning_nonempty_blocks_terminal`, `test_input_wrong_complete_hash_quarantines`.

The new public success test starts with an empty catalogue, uploads through authenticated ASGI, prepares through CLI, replays through MCP/HTTP, checks exact reference versus audition provenance, saves through CLI/MCP, generates and hashes downloaded artifact bytes, then reopens the real owner and replays both jobs and save. Exactly two fake backend launches occur, one audition and one reuse. No Core/catalogue DB seed or per-voice enrollment mutation is used. Backend call assertions distinguish both authored texts from the exact reference transcript. New public cancellation evidence retains owned/control/native reference bytes with an executing reservation before confirmed reap, then refuses cancelled history on reopen. New tampered-pair history compares retained files before/after refusal and observes no additional ACK. New import-reference identity failure preserves uncertainty across runtime reconstruction.

Development observations preceded the final run and are not represented as clean passes: initial fixture upload incorrectly called a nonexistent generic client operation (session `27368`); the first preparation exposed an unhandled empty-version preset derivation (`26049`); the next exposed Pocket-only resource fields (`50608`, diagnostic `79836`); the next exposed the old single-output native response predicate (`30638`, diagnostic `41365`). Refusal paths in the two unresolved collection probes ended with fixture control-close timeouts; no actual child existed. These source gaps were corrected before the final selection. Corrected journey/module observation `12000` passed 17 tests because unittest also discovered the two imported fixture TestCase classes; subsequent code imports those modules instead, and the final selection explicitly asserts the intended counts. Derivation run `15604` had 3 passes and one test-oracle error because the tampered plan hash was evaluated outside the expected-refusal block; moving that evaluation inside the assertion corrected the test. Affected follow-ups: `48335` 4 PASS, `17232` 2 PASS, `52562` 1 PASS. No unrelated broad retry was used.

Selected SOURCE SHA256:

- `worker/engines/qwen_prepare.py`: `fb6fc15d0c90a8edf505840d2d95c3e83633c6f1eb1d73297ba09e0e01902c58`
- `worker/native_protocol.py`: `087e1e7de7e1c04f4f62fb222cb6eb1479280a0b0cce31c1e579805e9032c3c3`
- `worker/native_supervisor.py`: `2f0d9cd7cf34c4e6fc469b561d8a2ac2c2d800eb9b33aee0f5e6bcd8a9b2c03c`
- `service/native_runtime.py`: `32170a54644775c854f05b9230c383fa5dfee62f6c216a13f842b53d757d801c`
- `tests/release/test_native_managed_clone.py`: `a3efee7f1814d929518fe91841ca17b7cc1e61d7534b1036f687a5a2c9de8c81`

## Remaining parent gates

Fresh independent Sol review owns adoption. This is SOURCE/inert evidence only: no actual ML, acoustic intelligibility/similarity, cold/warm latency/memory/throughput, native kernel/process containment, installation or release qualification. **Metal remains UNQUALIFIED with no reviewed arm64 wheelhouse. CPU and CUDA actual qualification remain mandatory.** Native topology/mount/firewall/pinned real TLS, compatible ABI/models, hardware selection, terminal-history renewal/reboot/fault handling and broader FULL API/CLI/MCP deliveries remain parent-owned. This slice supports same-boot clean succeeded histories only and retains existing finite capacity; it is not durable general service/reboot acceptance.

Legacy `lib/tts.py` SHA256 remains `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`, versus control `53520fff300f575f80eb492738f228d53ebe11ba81da9ecdfce4fdad80cc76cd`: **RED unchanged/unwaived**. No legacy reconciliation or metadata regeneration occurred. UI, STT, VoiceDesign, selector/topology/framework expansion and actual operational work were not added.
