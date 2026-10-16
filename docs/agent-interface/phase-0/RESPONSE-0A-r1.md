# Lot 0A round-1 correction response

Candidate for independent Sol-low re-review; no self GO or phase acceptance. All six findings remain subject to that review. No objections waived. Evidence is E1 only.

## Source and ownership

Starting clean HEAD: `e8aa9a92f6e00778d113fa2d6a141fe5fbf16683`, initially reported branch `t3/032fdaf1`, worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-032fdaf1`. App-owned branch was subsequently named `t3/fix-lot-0a-contracts`. Rejected candidate `9f1027b95229ed30f12d918b27bba989fbfe5d9d` cherry-picked here as `dd5b869bb82a8ade68c031e41e06b074c7946537`. Original/root/reviewer trees were not modified. Final correction SHA is the commit containing this response (use `git rev-parse HEAD`); it cannot be embedded in its own content.

Read PRODUCT, implementation start/protocol, shared contracts/Phase 0, lot ticket and full authoritative r1 report. Changed only contracts, contract tests and phase-0 documentation. Root ledger, ticket, review, application code, legacy defaults, research and production configuration remain untouched.

## Finding responses and regression evidence

- **F1:** Actual bounded base64 byte upload plus authenticated explicit end-exclusive content ranges, mapped HTTP/CLI/MCP in catalogue and CONTRACTS. `upload` verifies decoded size/hash; `content` validates authorization/range and returns exact bytes plus full hash. `test_exact_byte_upload_download_transcript` sends synthetic WAV bytes, reconstructs header/body ranges byte-for-byte, rejects wrong hash/length/range and denied access. `test_upload_oversize_and_byte_idempotency` covers maximum ceiling, invalid encoding and changed-byte conflict. This deliberately uses JSON range responses, not unspecified HTTP Range/206 semantics.
- **F2:** Voice reads expose revision; job voice-results expose immutable candidate IDs, prospective versions and audition assets; version-get retrieves saved versions. History/revision-get returns numeric revisions, typed recorded edits and frozen snapshots. Worker plans carry the complete typed heavy-operation request, exact project/voice/asset snapshots and explicit synthesis settings. Plan hash binds those bytes; semantic checks reject unavailable voice/reference inputs, mismatched voice identity, invalid prepare trims and render revisions. `test_voice_history_and_frozen_plan_reconnect_workflows` exercises candidate/save/version/history/restore inputs and stale reconnect conflicts; `test_all_frozen_operation_plans_have_executable_inputs` covers speech/prepare/design/render and frozen snapshots. Durable save allocation, pagination, undo and transport remain normative contracts, not an implemented store.
- **F3:** Worker/publication helpers consume authoritative full job/attempt state and authenticated transport-owner identity. Compare full tuple, current attempt membership, strictly increasing history and prior reap. Replacement requires death plus reap and exactly next generation with fresh attempt/staging IDs. `test_authenticated_attempt_reconnect_cancel_and_replacement` covers same-generation wrong job/attempt/owner/hash/stage, wrong transport peer, rogue publication, unreaped/death-unknown replacement, reconnect and late results after replacement/cancel. Completion may win cancel_requested; terminal cancellation rejects publication. Caller must supply trusted death/reap/peer evidence and serialize durable commits in later runtime work.
- **F4:** Recursive intrinsic identity checks run at every public validation entry point. Fifteen parameterized standalone artifact/take/message cases reject model/bundle substitution, unresolved device/precision and prohibited fallback. Existing resolution/readiness tests remain.
- **F5:** All edit oracle calls preserve existing take/asset/voice-version records, while accepting explicit new candidate IDs. Nine operation/resource combinations plus three take-field cases reject ordinary-edit rewrites, including block edit text/settings hash/voice identity. New candidates and unchanged timeline/selection remain tested.
- **F6:** Ripple validates resulting end as well as start, builds and validates a copy, and leaves input unchanged on failure. Exact safe maximum passes; maximum plus one fails atomically in `test_ripple_exact_max_and_atomic_overflow`.

## Versions, verification and limits

Operation catalogue/worker protocol correction revision **1.1.0**; persisted project schema/fixture namespace remains **1.0.0**, with voice revision fields added and fixtures deterministically regenerated. This is a deliberate pre-consumer contract correction. No application runtime consumes these contracts yet.

Inspected Dockerfile.tests and scripts/run-unit-tests.sh before execution; never ran run-e2e-test.sh. Built exact-source tag `py-audio-box-0a-fix-r1-tests` with Dockerfile.tests (cached dependencies, no model downloads). Test containers use `--rm --network none --cpus 2 --memory 3g`, no mounts. Selected execution and collection are serialized; a bounded hash-only source inspection overlapped the tail of an earlier focused/collection command, with no second test or model lane.

Commands:

```sh
docker build -f Dockerfile.tests -t py-audio-box-0a-fix-r1-tests .
docker run --rm --network none --cpus 2 --memory 3g py-audio-box-0a-fix-r1-tests python -m pytest tests/contracts tests/test_capabilities_and_metadata.py tests/test_clone_prompt_compatibility.py tests/test_clone_prompt_identity.py tests/test_customvoice_feature_flag.py tests/test_voice_cpu_batch.py tests/test_voice_synth_selection.py -m 'not integration and not redis_integration' -q
docker run --rm --network none --cpus 2 --memory 3g py-audio-box-0a-fix-r1-tests python -m pytest tests/contracts -q
docker run --rm --network none --cpus 2 --memory 3g py-audio-box-0a-fix-r1-tests python -m pytest tests -m 'not integration and not redis_integration' --collect-only -q
```

Selected contracts + legacy: **113 passed in 41.51s** before five additional correction regressions and explicit worker settings were added. Final focused/collection/image results are recorded below. Selected legacy sources/dependencies were unchanged thereafter; no unnecessary full legacy repeat. Determinism and unchanged legacy parser/source hashes are included in focused tests.

No E2 store/process/crash proof, E3 hardware/inference qualification, E4 runtime/UI/agent acceptance or E5 listening/production evidence. No real audio/private data, inference, public STT, Voicebox, cloud, Compose, push/PR/deploy or agent spawning. Independent Sol-low must inspect exact final commit against the full original findings; root owns integration and escalation after two unsuccessful rounds on the same issue.

Final exact-source image: `sha256:bf5e59d5c3a8703ee6bbb2a4eea1a91e133a2946863e43ada9420bf68c43f865`. Compared SHA256 for **131** copied contract/app/lib/test/benchmark/script/launcher/config files against this checkout: **zero mismatches**. Final focused suite: **47 passed in 8.51s**. Final offline collection: **353 collected in 3.75s**; collection is not execution. `git diff --check` passed. Disposable test containers exited and were removed. Working tree was committed clean at handoff; final SHA is supplied in the implementer handoff.
