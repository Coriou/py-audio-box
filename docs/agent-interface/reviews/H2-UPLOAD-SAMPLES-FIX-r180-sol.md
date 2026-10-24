# H2 canonical upload samples correction — r180

2026-10-24. Bounded R2 implementation and isolated checks PASS; ready for independent review. Parent adoption remains pending. No self-adoption or release GO.

## Pins and isolation

Fresh owned worktree `/private/tmp/pab-upload-r180`, branch `fix/upload-r180`, created clean at current project-root commit `84f5b63167390c67903464b598f863db96be243c`. Relative to r178 base `1b741eeb9e465f3973993b05023593164c6bfe83`, root adds only execution-ledger/r178/r179 documentation; inherited hardware is unchanged. Ordered clean cherry-picks:

- `a90cb5d7b51a354e27b2c8878cf53205c54cc474` → `0b7b9ace59387c4a0ecc327620731ebbadb0e733`.
- `2963abc89c81c1fb560f6f284d55b0b3170a7c6d` → `e458d10ed59112d9fb83a8fd26ec57f8f3b3ebdf`.
- R180 source: `ba82cd44d04d54e09f8c27ef519ef5bc6e1333d1`.

The initial clean T3 checkout was `6a0db52c7feb45fb5487b35d5fca431cc75c37be`; its initial cherry-picks were superseded by the fresh current-root checkout above. No historical checkout or project-root source was edited. Root's existing untracked evaluation evidence was left untouched. Sole root write is a copy of this report. Report committed separately from source; its commit is supplied in handoff. Root r173/r176/r178/r179 reports and both test conftests were read. No applicable AGENTS.md found. Memory registry search had no relevant hit and supplied no implementation fact. No delegation.

## Changes and authorization

Exactly five source/test/fixture paths change beyond the two carried commits: contracts/generate.py, contracts/fixtures/v1/operation-requests.json, contracts/validate.py, tests/contracts/test_contracts.py and tests/service/test_voice_store.py. Parent explicitly extended scope during this lane to the latter file solely for its test_contract_fixtures validation loop; no other voice-store code changes.

Canonical JSON fixtures represent binary bodies with the reserved fixture-only object `{"$fixture_binary_base64": "..."}`. The generator uses the exact synthetic short.wav bytes. `fixture_request(kind, value)` decodes strict canonical base64 to exact immutable bytes at the test/oracle boundary, bounds encoded size, rejects extra keys/nontext/noncanonical/invalid encodings, and applies the unchanged nonempty/24 MiB upload validator. Other request kinds pass through unchanged. Both all-operation consumers explicitly invoke this decoder before validate(). Public validate()/upload() still require exact bytes and reject both the fixture object and the old public hash/size/source/base64 envelope. No service/client adapter calls the fixture decoder. Media validity remains owner-ingest responsibility.

The generator supplies voice_versions_list with cursor='', limit=1, voice_id='voice.0'. Exact canonical sample keyset still equals all 42 operation IDs; no upload bypass or skipped operation. Determinism now compares the entire post-generation fixture filename set as well as bytes, detecting added outputs too. New regression asserts exact short.wav byte equality, immutable type, public rejection, strict fixture decoding and empty-body rejection. Existing raw-size/type/non-idempotency/reconstruction tests remain intact.

## Complete canonical operation-consumer inventory

Repository search covered operation-requests filename references and indirect generator/scenario/fixture iteration, rather than only upload-helper calls:

- contracts/generate.py: sole canonical writer; generic schema samples, explicit binary/version-list samples; passes request map to contracts/scenarios.py::transcript.
- tests/contracts/test_contracts.py: exact all-operation keyset and request validation (migrated); binary boundary regression; actual generator subprocess/determinism.
- tests/service/test_voice_store.py::test_contract_fixtures: second exact all-operation keyset/validation loop (migrated under explicit extension).
- tests/contracts/test_corrections.py::test_voice_history_and_frozen_plan_reconnect_workflows: selects block_text only; unchanged, executed.
- tests/contracts/test_frozen_boundary.py::scenario and test_complete_catalogue_transcripts: select speech_generate/voice_prepare/voice_design/project_render, with reuse of speech_generate; unchanged. Four complete catalogue transcript cases executed.
- tests/contracts/test_output_boundary.py: selects the same transcript operations and speech reuse; unchanged. Four nondefault exact save/version/speech-publication cases executed. Other mutation cases were inspected, not run.
- contracts/scenarios.py::transcript: deep-copies only requests[operation], for those four non-upload operations; no iteration over or forwarding of serialized upload. Unchanged.
- phase-0/CONTRACTS.md and historical RESPONSE/EVIDENCE reports describe examples/generation; retained historical P1B23 report contains an old operation-fixture hash. These are documentation/historical evidence, not executable request consumers; not rewritten.

No other canonical operation-request readers were found. Generic release source hash inventories are unrelated snapshot readers, not public request consumers.

## Red-green and actual generation

All Python checks use `/private/tmp/pab-0c-venv/bin/python` and PYTHONDONTWRITEBYTECODE=1; wrapper also uses `PYTHONPATH=.`. No pytest collection, hooks, guard or fixture machinery executed. Wrapper directly calls original functions/assertions, supplying fresh production-30 data for p arguments and expanding explicit parameter cases.

Before edits, observed exact-keyset AssertionError from test_operation_requests_jobs_artifacts_and_error_mapping; independently, existing asset_upload sample raises invalid_bytes. These are actual red observations. A staged generator preflight emitted only operation-requests.json as changed, then stopped at the pre-existing render-transcript dependency import: ModuleNotFoundError: numpy. The first expanded actual wrapper ran 48 cases: 47 passed, test_determinism errored on that generator subprocess. This was not claimed green.

Parent explicitly amended the network prohibition solely to install NumPy/soundfile and necessary dependencies into the authorized venv using compatible binary wheels; no compiler/global install/unrelated upgrades. Initial `.../python -m pip install --only-binary=:all: numpy soundfile` failed immediately because pip is absent. Successful exact command:

`uv pip install --python /private/tmp/pab-0c-venv/bin/python --only-binary :all: numpy soundfile`

Installed only numpy 2.5.4 and soundfile 0.14.0. Existing required cffi 2.1.1 and pycparser 3.11 were reused. No models/ML frameworks installed. This package-index access is the only network exception used.

Complete staged generation then ran against temporary copies of schema/operations, with all generated outputs compared byte-for-byte before in-worktree regeneration. Sixteen named output paths under contracts/fixtures/v1 were inspected/authorized for generator writes:

artifact.json, capabilities.json, expected.json, impulse.wav, job.json, long.wav, negative.json, operation-requests.json, production-100.json, production-30.json, short.wav, sine.wav, transcript-project_render.json, transcript-speech_generate.json, transcript-voice_design.json, transcript-voice_prepare.json.

Complete staged outputs matched the corrected owned fixtures exactly. Actual test_determinism retains its subprocess invocation of `[sys.executable, '-m', 'contracts.generate']` in the owned worktree; it passes byte/set equality across all canonical fixture files. Final generator output diff versus carried source: ONLY operation-requests.json, 6 added/4 removed lines, confined to binary upload representation and new version-list sample. No unrelated fixture churn. The generator was not stubbed or substituted; NumPy/soundfile render-silence generation executed normally.

Final wrapper `/private/tmp/r180-contract-check.py`: **56 cases PASS**, exit 0, 4.534s; log `/private/tmp/r180-contract-final.log`. Exact coverage:

- All nine test_corrections functions: test_exact_byte_upload_download_transcript; test_authenticated_attempt_reconnect_cancel_and_replacement; test_intrinsic_identity_at_resource_entrypoints (3 kinds × 5 changes =15); test_edit_preserves_existing_immutable_resources (3 operations × 3 groups =9); test_ripple_exact_max_and_atomic_overflow; test_voice_history_and_frozen_plan_reconnect_workflows; test_block_edit_cannot_rewrite_old_take (3 mutations); test_upload_raw_bounds_and_nonidempotency; test_all_frozen_operation_plans_have_executable_inputs. Total 33 cases.
- All fifteen test_contracts functions, one case each: test_schema_and_all_fixtures; test_purposeful_negative_fixtures; test_structural_rejections; test_exact_text_ids_defaults_and_source_control_coverage; test_determinism; test_revision_idempotency; test_lifecycle_cancel_fence_publication; test_stale_selection_render_and_fixed_timing; test_device_policy_and_readiness; test_tick_rounding_and_immutability; test_replacement_duration_does_not_ripple; test_migration_and_worker_boundaries; test_operation_requests_jobs_artifacts_and_error_mapping; test_late_candidates_and_text_edits_cannot_rearrange_or_select; test_binary_fixture_boundary.
- test_frozen_boundary::test_complete_catalogue_transcripts across all four operations (4 cases).
- test_output_boundary::test_nondefault_exact_save_version_speech_publication across prepare/design × two nondefault settings tuples (4 cases).

Isolated unittest discovery additionally passed:

- tests/service/test_voice_store.py: 20 methods, 0.858s, including the migrated canonical loop.
- tests/service/test_upload_api.py: 6 methods, 1.635s.
- tests/service/test_voice_api.py: 4 methods, 0.949s.
- tests/service/test_voice_dispatcher.py: 2 methods, 0.161s.
- tests/clients/test_voice_clients.py: 1 method, 0.225s.
- tests/clients/test_upload_clients.py: 2 methods, 0.264s.

Discovery uses `python -m unittest discover -s DIRECTORY -p PATTERN -v`. Logs retained at /private/tmp/r180-voice-store.log and /private/tmp/r180-test_*.py.log. Total final executed cases: 91. git diff --check passes. Supplemental host wrappers/logs are retained, not committed fixtures.

## Evidence boundary and next gate

This is source/isolated synthetic/local-ASGI evidence. Real temporary Core/SQLite/WAV/Dispatcher runs and substituted in-process client transport remain the same focused tier as r178/r179. No full suite, pytest collection/guard, Docker, listener/application network, ML/model execution, hardware, production, lifecycle/security/durability campaign or HELD-resource operation ran. Only specifically authorized package-index network access occurred. Inherited hardware was not requalified. Installed server batching, upstream memory/RSS and unknown owner outcomes retain r178/r179 limitations. Parent commissions independent review next and owns adoption; no release GO is implied.
