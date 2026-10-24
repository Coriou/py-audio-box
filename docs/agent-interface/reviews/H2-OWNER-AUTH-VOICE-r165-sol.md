# H2 owner-thread authenticated voice reads — r165 implementation

2026-10-24. **Implemented and focused local tests PASS; ready for independent Sol source/in-process review as one feature lot.** No independent review, parent adoption, installed activation or release GO is claimed. This is the approved H2A read slice, not complete H2.

## Pins and scope

Checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-d5b026af`, branch `t3/voice-dispatch-auth-clients`. Supplied clean base `e72f08278bc72871a4239a3f1d7d0874734edd7d`; verified root had advanced only to the parent authorization/document commit `22154af`, and fast-forwarded this clean checkout to that commit without reset. Source commit **`718a404288f17d33bb74927ff9daa877aa8eb7b4`**. Prior r164 source was not cherry-picked: it was report-only. Read full r164, r162 and canonical IMPLEMENTATION-PLAN, actual store/Core/dispatcher, contracts, clients and both conftests. No applicable AGENTS.md was found. Memory registry keyword lookup returned no relevant hit; no prior memory fact is used.

Exactly eleven source/test files changed: `service/{api,auth,catalogue,domain,dispatcher}.py`, `clients/{http,cli,mcp}.py`, `tests/service/{test_voice_api,test_voice_dispatcher}.py`, `tests/clients/test_voice_clients.py`. No Core/store/ingest/persistence/contracts/worker/recovery/guard/UI changes. This report is a separate commit; only this report is copied to root, without staging or committing root work.

## Concrete interfaces and behavior

`Dispatcher.call` now explicitly admits `voice_list`, `voice_get`, `voice_version`, `voice_resolve`. Four explicit `_command` branches call the corresponding `self.core.voices.list/get/version/resolve` methods. Existing bounded Future queue, argument/result deepcopy, exception propagation, owner construction/close and persistent activation fences are unchanged. No callback bus or generic attribute dispatch is introduced for voice commands. `voice_resolve` is trusted-only and has no public route/tool.

`service.api.create_voice_app(dispatcher, *, credentials)` borrows an already initialized Dispatcher. It does not construct, initialize, enroll or close one; caller owns its lifecycle. The default module `app` remains the separate synthetic fixture host. Production has no fixture asset fallback, evaluation UI, fixture header or speech execution path. The existing fixture list route now supplies the canonical empty cursor by default, correcting its previous missing-required-field validation failure. `service.domain.DurableCore` remains available as a lazy historical export; ordinary API/discovery import no longer imports Core.

`StaticCredentials([Credential(token, principal, frozenset(scopes)), ...])` is explicit operator configuration, with no default secret. `replace` atomically installs configuration; `revoke(token)` removes it. Bearer matching uses `hmac.compare_digest` on all configured entries under a lock. `resolve(Authorization)` supplies principal only from credential configuration and requires `read`; missing/incorrect/revoked credentials and wrong scope return existing `forbidden`/403. Revocation affects subsequent authorization checks; an already authorized in-flight read is not cancelled. A custom trusted resolver may supply the same `.resolve` interface; that resolver owns its credential policy. Request query/path cannot select principal. Store principal predicates produce cross-principal `not_found`, without fallback.

Production routes use existing declarations only:

- `GET /v1/capabilities`: cheap authenticated discovery, no profiles/engines asserted ready; advertises discover and the three implemented voice reads.
- `GET /v1/voices?limit=1..20&cursor=...`: `voices_list`, IDs only, numeric continuation derived from store sequence; full pages may yield a final empty page. Cursor is not a snapshot or opaque principal-bound token; every query remains filtered by authenticated principal.
- `GET /v1/voices/{voice_id}`: `voice_get`, existing voice metadata contract.
- `GET /v1/voice-versions/{voice_version_id}`: `voice_version_get`, existing immutable version contract.
- `GET /skeleton/operations/{operation_id}`: authenticated opt-in schema detail for these reads/discovery, marked `fixture=false`; retained client detail path reused without inventing a public operation contract.

`voice_read` validates canonical requests/responses, translates known domain failures and queue/timeouts/storage failures to bounded structured errors, and caps each serialized voice list/detail/version response at **2048 UTF-8 bytes**. Oversize returns user-facing `resource_limit`, never truncation. H1 compact methods are absent: list still materializes store metadata/history before reducing to IDs, and detail/version serialization allocates before measuring. This is a strict transport-output cap, not a store-allocation/work cap. Oversized details remain unavailable until compact domain interfaces exist. No authored content/history is silently discarded. Error mapping does not expose storage paths, payloads or exception text.

`HTTP(base_url=None, *, credential=None)` explicitly carries the bearer header and adds only the three declared reads to its finite operation set. CLI accepts `--credential` or explicit `SPEECH_CREDENTIAL`; MCP reads `SPEECH_CREDENTIAL`. CLI names are `voices list --limit --cursor`, `voice get ID`, `voice version get ID`; MCP names are `speech_voices_list`, `speech_voice_get`, `speech_voice_version_get`. All call HTTP and validate the same contracts. Existing fixture asset/status/download clients remain; production rejects unavailable fixture routes. Schemas remain opt-in details, audio remains explicit bounded fixture content/file transfer, with no default inline audio. Supplying CLI credentials through environment avoids command-line process-list exposure; no credentials are included in response/error bodies.

## Focused verification

Authorized interpreter `/private/tmp/pab-0c-venv/bin/python`; dependencies were already installed, with no installation/network. Final commands:

1. `-m unittest discover -s tests/service -p test_voice_dispatcher.py -v` — **1 PASS**, 0.092 s.
2. `-m unittest discover -s tests/service -p test_voice_api.py -v` — **3 PASS**, 0.630 s.
3. `-m unittest discover -s tests/clients -p test_voice_clients.py -v` — **1 PASS**, 0.102 s.

All exit 0; `git diff --check` passed. Real temporary Core/SQLite/media and a real Dispatcher owner run only under test-owned `pab-h2-voice-*` roots. Data is seeded via existing trusted Core methods while no Dispatcher owns the root; oversized metadata test closes its owner before trusted seeding and starts a new local test owner. No fake/unimplemented voice path is used. httpx ASGITransport executes without a listener; the client test substitutes only the sync HTTP I/O boundary with actual in-process ASGI requests, preserving HTTP wrapper headers/routes/validation and real owner/store behavior. It is not socket/TLS evidence.

Coverage: all four dispatcher reads; observed execution on `dispatcher.thread.ident`, distinct from request thread; detached returns; cross-principal refusal; unsupported command rejection before queueing; missing/bad/wrong-scope credentials, revocation; list page/cursor/limit and unexpected principal field; real immutable version/detail; actual oversized metadata resource_limit; sanitized OSError/not_ready translation; fixture/production separation; discovery makes no owner call; fresh import subprocess asserts no torch/transformers/Core/Dispatcher import; CLI/HTTP/MCP list/get/version parity and auth errors. Queue saturation, timeout, custom resolver policies and socket/TLS protections are source paths rather than separately executed negative cases.

Earlier runs were not passes: first fixture omitted required expected_revision; next compared registration input to the store-enriched immutable version and exposed the fixture route missing cursor default. Fixed fixtures use the actual registered immutable version; fixed default route supplies cursor. Subsequent runs passed; test inheritance was then refactored to avoid redundant collection, followed by the final five-case pass above. No native fatal, unexpected signal or blind operational retry occurred.

No pytest/conftest collection, full suite, installed listener, Docker, external network, model/ML download/import/inference, maintenance/diagnostic/lane operation, delegation, deployment or persistent enrollment ran. Discovery's subprocess only imports the API and inspects module names. Cleanup touches only exclusively test-owned temporary roots. Historical/HELD resources and STT were not accessed or modified; no live resource status claim follows.

## Deferred and next owner

Public alias resolution, version history, mutations/save/archive/import/export await declared shared contracts and approved owner interfaces. No new unauthorized error code was added. H1 compact reads, ingest/ownership/production journey, full H2 jobs/media/client coverage and installed security/lifecycle evidence remain open. All CPU/CUDA/packaged native Metal qualification and listening, durability/upgrade/restore/fault gates remain required; UI/STT remain deferred as commissioned. Next owner is the parent: commission an independent Sol review of source `718a404` plus this report as one cohesive owner/auth/client lot, then decide adoption. Release remains NO-GO.
