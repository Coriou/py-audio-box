# H2 authenticated production control r186

Status: IMPLEMENTED FOR INDEPENDENT SOL REVIEW. No self-adoption, integration GO, installed activation, release approval or actual engine/cloning-quality claim.

## Pins and authority

Sole successor to terminated r185; no predecessor work resumed. Own checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-640cf27f`, branch `t3/h2-production-control`. Clean entry/base `5379d314642f7e36a717e0399c72769731b5f2d1`. Source commit `48bcba3324c1e16af982171c739ccfb80864284f` has that sole parent. Source was committed with clean status. Report is a separate successor commit; its exact SHA is returned in the handoff. Parent root report destination was explicitly confirmed as `/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/H2-PRODUCTION-CONTROL-r186-sol.md`; historical `t3-2e84fcbe` was not written.

Read PRODUCT.md, canonical plan H1/H2/H4, integrated r183/r184, actual contracts/schema/error declarations and generator/scenarios, API/auth/catalogue/domain, Core/Production/VoiceStore, admission/dispatch store/dispatcher/Authority and synthetic worker source. Read both root/service conftests without executing or collecting them. No applicable AGENTS.md found. Memory registry quick search had no relevant H2 hit; no memory-derived implementation premise was used.

Original authority explicitly allowed narrow synthetic reference admission and exact worker Authority dependencies. Two additional concrete dependencies were presented to parent and explicitly approved during this task:

1. `service/production.py`: solely named private `_admission`, `_profiles`, `_dispatch_settings` forwarding from trusted owner to existing Core.submit_synthetic. Existing direct-domain defaults remain unchanged. No arbitrary kwargs or public context forwarding.
2. `service/reconciliation.py`: exact integrated H0 catalogue schema/reference adapter and focused tests. Existing inventory rejected these already-created Core tables as unknown schema on a fresh owner. The fix uses VoiceStore's actual DDL/schema identity, including constraint indexes and immutable-version triggers, and refuses unsupported format/schema/rows. No generic table acceptance or bypass of integrity blocking.

## Result and trust boundary

The borrowed-owner `create_voice_app` factory exposes authenticated upload, catalogue/asset reads plus prepare, results, explicit candidate save, generation, job get/cancel and artifact get. All three clients traverse the same canonical routes, IDs and schemas. Read plus production-write is required for mutations; reads require read. Existing StaticCredentials already implements both-scope enforcement and revocation, so auth.py needed no change. The default fixture app remains separate; this does not install a listener.

`SyntheticContexts` accepts explicit operator snapshots mapped by authenticated principal and operation, serializes them into immutable detached bytes, and returns fresh copies after request validation. No public field can supply plans, assets, bundles, settings, readiness or arbitrary profiles. Only the fixed CPU synthetic Qwen Base clone/prepare context is accepted; incompatible identity, variants, fallback/reuse and unsupported configuration fail closed. Unconfigured provider means not_ready. Discovery and status advertise no qualified engines/profiles or actual speech readiness. Test profile readiness receipts are explicitly fabricated E1 synthetic evidence and are never returned as qualified engine discovery.

Explicit Dispatcher production owner commands validate typed requests and run Production/Core/SQLite on the owner thread. There is no public generic bus, new scheduler, alternate worker or Core semantics change. Prepare/generate use Core's existing atomic submission plus reservation and dispatch settings. Duplicate lookup precedes new resolution and admission, preserves frozen identity, and does not double-reserve. Admission refusal and injected reservation failure leave no accepted job/event/settings/reservation. Existing allocation, attempts/generation, cancellation, resource accounting, result validation and publication barriers remain in use.

Admission now permits exact declared reference dependencies for prepare and single-result speech; it does not allow arbitrary caller asset lists. DispatchStore rechecks the frozen plan's required inputs through Core._input_assets, including principal/media/exact metadata, before resource acquisition and attempt allocation. Authority freezes exactly those IDs, excludes unrelated managed assets and remains detached from caller mutation. Missing, foreign and altered references fail before advancing generation or acquiring execution resources.

Results read only committed owned prepare publications through Production. Save selects a candidate from a named prepare job and uses existing VoiceStore revision CAS, immutable version and durable idempotency behavior. Generation resolves the saved version and freezes exact authored text/settings/identity. Published audition/generation assets use the existing VoiceStore ownership oracle for public asset retrieval, including inherited producing-job ownership; explicit upload ownership is checked before media verification to preserve foreign-caller 404 behavior even for tampered media.

Cancellation invokes the existing owner cancel command. A durable intent in the existing catalogue request journal binds principal/key to the exact job before cancellation; an interrupted intent can only resume that cancellation. No new database table was introduced. Cancel returns the observed typed job, so repeated cancellation may show later terminal state. Queued cancellation releases the unstarted reservation through existing machinery; active cancellation retains existing receipt/publication fences.

Public JSON mutations have a 16 KiB body limit, finite receive deadline, duplicate-key rejection, one validated Idempotency-Key, and sanitized declared errors. Production typed responses have a 12 KiB limit. Existing media range/MCP byte budgets remain explicit. Clients perform no automatic retries. CLI uses named commands and canonical JSON request objects for prepare/save/generate; MCP adds a typed transport idempotency_key, removes it before the canonical request, and passes it only as the HTTP header.

## Contract delta and exact files

`voice_save_request` now includes job_id, removing ambiguous global candidate lookup. Its response is the existing synchronous `voice_register_result`, rather than a fictitious queued job. `job_cancel` returns the existing `job` schema. Relevant per-operation errors include readiness/resource/media/control errors. Generator scenarios, both prepare/design transcript consumers, operation request fixtures, the direct correction fixture consumer and existing adapter tool expectations were updated together. The generator is deterministic and all generated requests/exchanges validate.

Source commit contains exactly these 23 files:

- clients/cli.py
- clients/http.py
- clients/mcp.py
- contracts/fixtures/v1/operation-requests.json
- contracts/fixtures/v1/transcript-voice_design.json
- contracts/fixtures/v1/transcript-voice_prepare.json
- contracts/scenarios.py
- contracts/v1/operations.json
- contracts/v1/schema.json
- service/admission.py
- service/api.py
- service/catalogue.py
- service/dispatch_store.py
- service/dispatcher.py
- service/domain.py
- service/production.py
- service/reconciliation.py
- tests/clients/test_production_clients.py
- tests/contracts/test_corrections.py
- tests/service/test_adapters.py
- tests/service/test_production_admission.py
- tests/service/test_production_api.py
- tests/service/test_production_reconciliation.py

No Core, persistence, VoiceStore, worker, engine, transport, deployment or legacy source was edited. Reconciliation checks exact catalogue schemas and known stored result types, validates version/configuration and owned reference hashes, joins inherited published ownership to producing jobs, inventories saved references and rejects unknown record kinds. Dynamic reference_asset_sha256 keys are followed only at the typed voice_version location. No deletion, repair or migration was added or executed.

## Executed verification

All Python execution used `/private/tmp/pab-0c-venv/bin/python`; tests used PYTHONDONTWRITEBYTECODE=1 and isolated unittest discovery. Final passing commands/checks:

- `-m unittest discover -s tests/service -p 'test_production*.py' -v`: 13 PASS (4 existing direct-domain, 3 admission, 4 API, 2 reconciliation).
- `-m unittest discover -s tests/service -p test_voice_api.py -v`: 4 PASS.
- `-m unittest discover -s tests/service -p test_upload_api.py -v`: 6 PASS.
- `-m unittest discover -s tests/service -p test_voice_dispatcher.py -v`: 2 PASS.
- `-m unittest discover -s tests/clients -p '*clients.py' -v`: 4 PASS (production plus existing upload/voice clients).
- `-m unittest discover -s tests/worker -p test_production_synthetic.py -v`: 1 PASS.
- Existing `test_adapters.test_discovery_without_ml_scan_or_network` directly imported/invoked: PASS; no pytest collection. Its subprocess forbids ML imports and network/host scans.
- Canonical generator executed, then rerun with complete fixture byte comparison: deterministic PASS. New unittest validates every operation request and all four generator transcript exchange request/response consumers against current declarations.
- `git diff --check`: PASS before source commit.

Total focused unittest coverage: 30 PASS. Representative actual in-process ASGI + real temporary owner/Core/SQLite/files + synthetic worker adapter journey covers reference upload, prepare publication, owner-only audition retrieval, candidate selection, save, duplicate replay/CAS conflict, frozen generation, job/artifact/media retrieval, both-scope auth, missing key, public context rejection, unsupported model, no provider readiness, explicit cancel and client parity. Restart preserves saved immutable voice/media and admits generation. Reconciliation negatives cover unknown tables, changed owned triggers, unsupported format, malformed version, valid foreign-owned reference and unknown request operation. Authority tests cover exactness, detached mutation and preallocation failures. Test-owned synthetic runtime fabricates lifecycle receipts and calls the existing candidate builder; it is not supervisor authentication or containment evidence.

Intermediate failures were corrected without relaxing gates: catalogue inventory unknown schema (approved dependency); test-only Core.jobs shape assumptions; test synthetic provenance not matching existing fixed silence policy; test API retaining a closed pre-restart dispatcher; client fixture path crossing macOS /var symlink (resolved fixture path); saved version hash-map inventory support; existing foreign tampered-upload 404 ordering regression. Focused affected checks then passed.

## Evidence limits and next gate

No pytest collection/conftest/maintenance guard, broad suite, listener, network, Docker, ML/model inference, hardware test, Linux sandbox on Mac, HELD resource operation, delegation, self-adoption or historical cleanup. Only normal own temporary fixtures were cleaned. The existing listener/stdio integration cases in test_adapters.py were updated for the new tool surface but not run; actual client requests here use in-process ASGI transport, not a socket listener or installed MCP host. Existing direct-domain tests do use their historical-local synthetic subprocess; no installed supervisor/runtime qualification is implied.

This implements the commissioned H2 prepare/results/save/generate/job read/cancel slice and necessary artifact/media retrieval, not all H2 scripts/stitch/retry/events/alias-management delivery or the complete release plan. Default app/deployment activation remains parent-owned. Independent Sol review of the exact source commit is next; parent owns integration and any later installed/H4 actual qualification.
