# H2 production adapters — independent first-round source review r187

Verdict: **bounded GO** for adoption of the commissioned authenticated synthetic production adapter source slice. No blocking finding identified. This is not complete H2 delivery, installed transport/engine qualification, activation, runtime containment evidence or release approval. Parent owns integration/adoption and subsequent installed/H4 gates.

## Pins, scope and manifest

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-640cf27f`, branch `t3/h2-production-control`, clean HEAD `6ce9e378dd11bed0e01d8c1c4e840fc3f07b69ee`. Verified source `48bcba3324c1e16af982171c739ccfb80864284f` has sole parent/base `5379d314642f7e36a717e0399c72769731b5f2d1`; docs HEAD has sole parent source and changes only r186. Source diff is exactly 23 files; base-to-docs diff is 24 files. Candidate r186 and root r186 compare byte-identical.

Reviewed PRODUCT.md, canonical IMPLEMENTATION-PLAN H1/H2/H4 and integration boundaries, r186, integrated r183/r184, complete source diff and relevant existing auth/Core/VoiceStore/admission/dispatch/Authority/reconciliation/contract/client consumers. Read tests/conftest.py and tests/service/conftest.py without executing or collecting them. No AGENTS.md found in candidate tree, candidate worktree parent tree or checked ancestor paths. Memory registry search returned no relevant H2 entry; no memory-derived premise used. No delegation.

Exact source manifest:

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

The brief explicitly authorizes the narrow admission/dispatch-store reference changes, private Production forwarding and exact H0 reconciliation dependency. No Core or worker source change is present.

## Source assessment

No actionable blocking finding in this slice. The following conclusions are bounded to the reviewed managed-store and trusted-operator interfaces:

- `service/api.py:120-126,223-250` authenticates all factory routes and requires production-write in addition to read for mutations through existing StaticCredentials. Principal derives from credentials, never JSON. Mutation parsing enforces JSON, one Idempotency-Key, a 16 KiB streamed body, finite receive deadline and duplicate-field rejection. Closed schemas reject arbitrary context/profile/settings fields. Errors are sanitized and constrained by declared operations (`service/domain.py:215-249`). Production results have a 12 KiB response budget; existing explicit media/MCP range budgets remain.
- `service/domain.py:174-212` keeps detached serialized operator snapshots keyed by principal/operation. Fixed CPU synthetic Qwen Base identity and prepare/clone settings kinds are enforced; request identity must match configured requested identity and generation rejects fallback/reuse/variants outside the synthetic seam. Core frozen-plan and profile admission remain the further consistency checks. This is operator configuration, not a public context resolver or evidence of real readiness. Discovery/status report no qualified engines/profiles or speech readiness.
- `service/dispatcher.py:159-210` supplies named owner-thread commands and ownership checks for jobs, prepare results and artifacts. Prepare/generate duplicate lookup checks principal/operation/key fingerprint before context resolution or admission, preserving frozen identity and avoiding another reservation. New submissions use existing Production/Core atomic admission, settings and resource reservation. `service/production.py:16-25,49-58` forwards only the explicitly named private admission/profile/settings parameters. No alternative scheduler or publication path appears.
- `service/admission.py:30-56` admits exact declared dependencies for prepare and single-result synthetic speech. `service/dispatch_store.py:55-65` invokes existing Core._input_assets before acquiring execution resources or allocating an attempt and binds the returned exact IDs into Authority. Existing Core dependency verification checks managed ownership, bytes and exact metadata. Tests establish detached Authority input membership and missing/foreign/altered dependency refusal before generation advances. Existing attempt, cancellation, result and publication fences remain unchanged.
- `service/dispatcher.py:183-196` binds cancellation key/fingerprint to the owned job in the existing catalogue journal before invoking the existing cancel command. An interrupted intent can resume only the same cancellation. Repeated cancellation returns observed job state, consistently with the changed contract; it need not freeze a historical response. Queued cancellation releases the existing unstarted reservation. Save selects a candidate from an explicit owned published job and delegates immutable version storage, durable replay and revision CAS to the existing Production/VoiceStore path. It does not accept caller-authored version metadata.
- `service/domain.py:102-119` preserves explicit-upload foreign-owner rejection before media verification and uses the existing VoiceStore ownership oracle for inherited published asset ownership. Published audition/generation retrieval remains tied to producing jobs. Compatibility ownership scans retain their existing cap; no unlimited new public ownership scan was added.
- `service/reconciliation.py:35-63,118-123,195-199,206-227,241-274` recognizes the exact H0 DDL identity, including constraint indexes and immutable-version triggers, and exact format revision. Catalogue versions and request result types are validated; unknown operations/schema fail closed. Version references bind principal ownership and retained metadata hashes; published ownership is joined through artifact/producing-job records. Reference IDs and typed dynamic hash-map keys are retained through the existing inventory machinery. This is an adapter for trusted persisted records, not authentication of arbitrary coordinated hostile database rewrites. It adds no repair, deletion or generic unknown-table bypass.
- `contracts/v1/schema.json:1261`, operation declarations, scenarios, both prepare/design transcripts and the direct correction consumer agree on explicit save job_id and synchronous voice_register_result. Cancel returns the existing job schema. HTTP/CLI/MCP use canonical IDs/routes/request schemas; MCP separates its typed idempotency_key argument from canonical HTTP JSON and sends the header. No automatic retry was introduced. Existing listener/stdio adapter expectations were updated, but those installed transports were not exercised here.

## Independently executed verification

All Python execution used `/private/tmp/pab-0c-venv/bin/python` with `PYTHONDONTWRITEBYTECODE=1`. Only these authorized isolated unittest discoveries ran:

- `-m unittest discover -s tests/service -p 'test_production*.py' -v`: **13 PASS**.
- `-m unittest discover -s tests/service -p test_voice_api.py -v`: **4 PASS**.
- `-m unittest discover -s tests/service -p test_upload_api.py -v`: **6 PASS**.
- `-m unittest discover -s tests/service -p test_voice_dispatcher.py -v`: **2 PASS**.
- `-m unittest discover -s tests/clients -p '*clients.py' -v`: **4 PASS**.
- `-m unittest discover -s tests/worker -p test_production_synthetic.py -v`: **1 PASS**.

Total: **30 PASS**. Direct import/invocation of `test_adapters.test_discovery_without_ml_scan_or_network`: **PASS**, including its fresh-interpreter forbidden-import/network/scan checks. No pytest collection.

Called canonical `contracts.generate.generate()` twice and compared complete fixture byte maps before, after first generation and after second: **unchanged and deterministic PASS, 17 files**. The targeted production test also validates all operation request fixtures and every request/response exchange in the four generator transcripts. `git diff --check 5379d314..HEAD`: **PASS**. Candidate status remained clean after all checks. No extra defect reproduction was needed.

Coverage includes real temporary SQLite/files and owner dispatch with in-process ASGI: upload→prepare→results→explicit selection/save→generation; duplicate/reservation behavior; ownership/auth/scope failures; revision/idempotency conflict; exact authored text; job/artifact/media retrieval; queued/active cancellation; client parity; saved-reference retention and restart admission; unknown catalogue table/trigger/format/version/ownership/request rejection. Existing direct Production tests also use their historical-local synthetic subprocess.

## Evidence limits and remaining owner gates

SyntheticContexts and test profile readiness are fixed fixture configuration. The fake test runtime fabricates lifecycle/reap receipts and emits synthetic silence; passing tests establish source/contract behavior, not supervisor authentication, containment, installed readiness, inference or clone quality. Carried installed transport and actual engine/hardware qualification remain separate and unadvanced.

No pytest collection, maintenance guard, full suite, listener, network operation, Docker, ML/model inference, hardware, Linux sandbox on Mac, HELD operation, source edit, commit, delegation or historical cleanup. Only ordinary test-owned temporary fixtures were cleaned. Generator writes were explicitly authorized named fixture regeneration and left candidate bytes/status unchanged. Sole persistent deliverable is this root review report.

Parent may adopt this exact source slice subject to its integration authority. Default app/deployment activation and broader H2 scripts/stitch/retry/events/alias-management completion remain parent-owned; this bounded GO does not assert the canonical full H2 exit. Existing installed transport/runtime and H4 actual three-family engine acceptance gates remain in place.
