# H2 read error contract fix — r169 implementation

2026-10-24. **R167-1 implemented; focused source/local-ASGI checks PASS. Ready for fresh independent Sol closure review, then parent H2 adoption.** No independent review, H2 adoption, production activation or release GO is claimed.

## Pins and authority

Own checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-e04748bb`, branch `t3/voice-read-error-contract-fix`, initially clean at `6bbb0de30bff1df4c63f6c59eca463450e9b2710`. Reset only this own branch to authorized H2 candidate/report `2bc39a39417278a71dc5d670c6632bc801d78656`, whose source is `718a404288f17d33bb74927ff9daa877aa8eb7b4`. Historical `t3-d5b026af` was not modified. Read full r165 and parent-root r167. No applicable AGENTS.md found. Memory keyword lookup had no relevant hit; no memory-derived fact is used.

Fix commit: **`54262e3eaf28faa78365352251cf33f700a1e4ae`**. Exactly two source/test paths changed: `contracts/v1/operations.json` and `tests/service/test_voice_api.py`. This report is committed separately; only this report is copied to parent root, without staging or committing parent work.

## Change and regression

Added existing `not_ready`, then `resource_limit`, after the existing unchanged `invalid_request`, `forbidden`, `not_found` sequence for only `voices_list`, `voice_get`, and `voice_version_get`. This follows the existing relative code order in other declarations. No new operation/code/schema, route, scope, request, response or production implementation changed.

The API regression fetches authenticated `/skeleton/operations/{operation}` details and asserts all three exact error lists. For every read it injects OSError, queue.Full and TimeoutError at the borrowed dispatcher call boundary, executes the actual ASGI route, validates the emitted error with existing pure `contracts.validate.validate`, and checks code membership in that same operation's advertised errors, status 503, bounded body and sanitized exception text. These are exception translation tests, not actual queue saturation or elapsed timeout experiments.

The existing real oversized aliases fixture remains: close test Dispatcher, seed aliases via real temporary Core, close Core, start new test Dispatcher, request actual voice_get. It now checks status 429 and `resource_limit` membership in advertised voice_get errors, canonical error validation and bounded response. No canned metadata return, truncation, relaxed bound or failure relabeling was introduced. Resource-limit execution is demonstrated for real voice_get metadata; declarations and readiness are checked for all three reads. This does not claim oversized real list/version fixtures were executed.

Existing tests retain pagination/cursor/limit checks, successful immutable version/detail values, principal isolation, auth/revocation, fixture separation and cheap discovery. Unchanged client and dispatcher regressions also pass.

## Executed evidence

Only installed `/private/tmp/pab-0c-venv/bin/python` was used for tests:

- `-m unittest discover -s tests/service -p test_voice_api.py -v`: final **4 PASS**, exit 0, 0.742 s.
- `-m unittest discover -s tests/service -p test_voice_dispatcher.py -v`: **1 PASS**, exit 0, 0.044 s.
- `-m unittest discover -s tests/clients -p test_voice_clients.py -v`: **1 PASS**, exit 0, 0.100 s.

Before editing the contract, the ordinary source fixture regression failed as expected: first run detected actual voices_list not_ready outside its operation declaration and all three missing declarations. Removed the redundant earlier OSError check (now covered by the all-operation matrix), allowing a second red run to reach real oversized aliases: actual voice_get resource_limit failed same-operation membership, with all three declaration checks also red. After the six memberships were added, all four tests passed. These were intentional red/green source checks, not unexpected fatal retries. No fatal occurred.

A bounded structural comparison against the pinned candidate removed exactly the appended two codes from each of these three operations and proved the complete catalogue otherwise equal. Existing pure error validator accepted both existing error codes. `git diff --check` passed. No pytest import/collection or broader suite ran.

Tests use real temporary Core/SQLite/media and Dispatcher plus in-process ASGITransport, no listener. Cleanup is limited to the tests' own temporary roots. No API/domain/auth/dispatcher/client production/store/Core/ingest/global-code/schema/recovery changes; no Docker, model, network, lane, native-fault, delegation or deployment action. No HELD resources/proofs were inspected or changed. Parallel H1 work was not incorporated.

## Handoff

Parent owns fresh focused Sol review of R167-1 closure on the pinned fix plus report, then H2 adoption. User explicitly prohibited delegation in this implementation lane, so no review agent was launched. All prior operational and production qualification gates remain outside this source/local-ASGI result.
