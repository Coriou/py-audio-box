# H2 read error contract — independent r171 Sol review

2026-10-24. **GO for R167-1 closure and complete commissioned H2A voice-read slice SOURCE/local-ASGI adoption readiness.** No new blocking finding. This carries r167's owner/authentication/principal-isolation review because those production files are unchanged. It is not whole-H2 completion, installed activation or release acceptance.

## Pins and scope

Reviewed candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-e04748bb`, branch `t3/voice-read-error-contract-fix`, clean HEAD/docs `68504a21c9f8b48b88086e5d70b755df3025c378`, fix `54262e3eaf28faa78365352251cf33f700a1e4ae`, base `2bc39a39417278a71dc5d670c6632bc801d78656`. Read full r169, parent-root r167 and r165. No applicable AGENTS.md found; memory registry search returned no relevant hit and supplies no conclusion.

Base-to-HEAD changes exactly `contracts/v1/operations.json`, `tests/service/test_voice_api.py`, and r169. Fix commit changes exactly the first two. API/domain/auth/dispatcher/client/Core/store production source is identical to the r167 candidate. Sole write is this report in the parent workspace; no source/test edit, commit or delegation.

## R167-1 closure

The complete catalogue diff appends exactly existing `not_ready`, then `resource_limit`, to only `voices_list`, `voice_get`, `voice_version_get`, preserving the prior `invalid_request`, `forbidden`, `not_found` sequence. No other catalogue value, operation, route, scope, request/response reference, schema or global error code changes. No bound is relaxed, metadata truncated, or failure relabeled.

Actual public read paths map request/response validation and cursor failures to invalid_request, credentials to forbidden, absent principal-scoped records to not_found, oversized serialized results to resource_limit, and storage/queue/timeout or other mapped readiness ValueErrors to not_ready. These codes are now declared by each read. The adapter's retained voice_archived mapping belongs to trusted resolve/mutation failures: the public branches call list/get/version, whose source does not reject archived records with that code; public alias resolution is absent. It does not require an additional read code.

The new test obtains errors from the same authenticated production `/skeleton/operations/{operation}` detail and asserts each exact five-code list. For all three actual ASGI routes, OSError, queue.Full and TimeoutError are injected only at Dispatcher.call. Each of nine responses must validate against the canonical global error schema, belong to its own advertised operation, return not_ready/503, stay below 2048 bytes, and omit the injected private-path text. Responses are produced by the real route/adapter, not canned error bodies.

The retained oversized-metadata case closes the test owner, seeds thirty long aliases through real temporary Core, closes Core and starts a replacement test owner. Actual voice_get must return resource_limit/429, canonical error shape, same-operation membership and bounded body. Existing successful detail/version, pagination, principal/auth/revocation and discovery checks remain. Real oversized list/version responses are not separately executed.

The before-fix red reason is correct by source comparison: the prior declarations omit both codes, so actual readiness responses and the real oversized voice_get response fail membership, independently of global schema validity. R169 records intentional red runs; this review did not reproduce them by editing or swapping the catalogue. The candidate tests' exact-list assertion can fail before entering the readiness matrix against the old catalogue; the retained separate oversized case still reaches the real resource_limit membership failure. No claim of independently rerun historical red output is made.

## Independent executed evidence

Ran each authorized command once using installed `/private/tmp/pab-0c-venv/bin/python`, in the pinned candidate:

- `-m unittest discover -s tests/service -p test_voice_api.py -v`: **4 PASS**, exit 0, 0.764 s.
- `-m unittest discover -s tests/service -p test_voice_dispatcher.py -v`: **1 PASS**, exit 0, 0.047 s.
- `-m unittest discover -s tests/clients -p test_voice_clients.py -v`: **1 PASS**, exit 0, 0.105 s.

`git diff --check 2bc39a39417278a71dc5d670c6632bc801d78656 HEAD` passed. Candidate was clean before checks; no unexpected fatal or retry occurred. Fixtures use real test-owned temporary Core/SQLite/media and a real Dispatcher owner, with their own cleanup. Client tests substitute the synchronous HTTP I/O boundary with actual in-process ASGI requests. These checks establish local routing/translation/contract behavior, not real queue saturation, elapsed timeouts, sockets, TLS or installed deployment behavior.

## Adoption boundary and next owner

R167's no-owner-thread-escape and no-cross-principal-disclosure findings carry unchanged; this focused review does not claim a new exhaustive owner/auth audit. Closing its sole P2 contract finding makes the complete commissioned H2A read slice ready for parent SOURCE/local-ASGI adoption. The 2048-byte output cap still allocates/serializes before measuring; it is not an allocation/work bound. H1 compact-store integration remains pending. Public mutations, uploads, jobs/media and later H2 work remain outside this slice and are not newly invented blockers.

Parent may integrate H1 after this green review, checking H2/H1 file disjointness and running meaningful combined tests of the integrated behavior. This review does not certify that future merge. Production lifecycle/security, runtime/provider and release gates remain separate.

No pytest guard/full-suite collection, listener, Docker, model/network, lane, activation, operational cleanup or resource/proof inspection ran. Operational markers/resources/proofs and STT were not changed; no live state or operational-readiness claim is made.
