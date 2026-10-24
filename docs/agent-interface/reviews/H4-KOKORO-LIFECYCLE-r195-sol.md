# H4 Kokoro authenticated lifecycle — r195 Sol implementation

2026-10-24. Source implementation complete for independent review. No self-adoption, actual engine qualification, installed readiness, audible acceptance or release approval. No clone capability added.

## Exact pins and scope

Own assigned fresh worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-b3a7929e`, branch `t3/kokoro-lifecycle-integration`. Clean entry `66ec1357737ed140416e71b97cc899faf2314f4c`; fast-forwarded before edits to current root `2f047d51c4e4f4985a8825dd2fcd05226c459068`. Verified ancestors: reviewed engine `85eb3d41ce3bea6642ef680c7aa277eb4d6bde09` and provisioning fix `c826c0c6f33e7acec00c7e88bc6484023ca0ae5f`. Source commit **`f04a3a16af055ab21c32c0d5554b17aff1c4a148`**. Root later advanced to `e782f42565fdc9b10e6d32cf14259c0b6d86b0a8`; its delta from this base is only EXECUTION-LEDGER documentation. No source copied back; only this report is copied to root, and the report is committed separately here.

Read r188 milestone including assignments/blockers; r189/r191 engine and r190/r192/r193/r194 provisioning reports; relevant actual Core submission/verification/publication, Dispatcher, admission/reservations, dispatch journal, Authority/session, supervisor/child/protocol, domain and provisioning implementations/tests. No applicable AGENTS.md found in checked ancestors or owned subtrees. Bounded memory-registry search returned no relevant hit and supplied no premise. No delegation.

Exact source files (16):

- `service/execution_context.py` (new), `service/production.py`, `service/dispatcher.py`, `service/admission.py`, `service/dispatch_store.py`, `service/core.py`, `service/reconciliation.py`, `service/worker_transport.py`.
- `worker/runtime_protocol.py`, `worker/runtime_supervisor.py`, `worker/runtime_child.py`, `worker/engines/kokoro.py`.
- `contracts/v1/schema.json`.
- `tests/service/test_engine_admission.py` (new), `tests/worker/test_engine_lifecycle.py` (new), `tests/worker/test_engine_kokoro.py`.

No API/client/install/provision-network implementation changed. Existing `SupervisorRuntime.prepare` already transports the entire Authority snapshot; no adapter change or public execute message change was needed. No dispatch DDL changed.

## Closed integration boundary

Operator constructs a KokoroContexts principal template from an explicitly selected provisioned immutable bundle and closed Execution descriptor. Local admission checks owner, single-link regular files, sealed nonwritable bundle/files, nonsymlink absolute ancestry, bounded exact raw manifest digest, verified asset hashes, checkpoint/revision/language. The immutable provisioning path is retained; active.json is never reread during generation. Trusted operator-owned storage and absence of hostile concurrent mutation remain prerequisites.

Public request data supplies no descriptor, bundle path, runtime factory or provenance source. The private context adds exactly `execution: {descriptor, bundles}` beside existing `plan,bundles`. Descriptor validation is reused from Execution; shared `worker.runtime_protocol.execution` binds exact CPU/float32 Kokoro identity, owned preset configuration/language, version ID, authored request identity, forbid fallback, single variant/no reuse, empty reference assets, profile `kokoro-cpu`, thread/memory and finite output/text/disk limits. Model revision and raw manifest digest remain descriptor fields; no synthetic catalogue identity is renamed into a real one.

Production reloads the actual principal-owned immutable version before submission. Core's existing single transaction is named `submit_execution`; `submit_synthetic` is a compatibility forwarding entry point to that same transaction. Real contexts require admission/profiles/Kokoro dispatch settings. Duplicate lookup, owned input validation, job/event/reservation/settings insertion, allocation generation and exact dependency checks remain in place. No second scheduler or direct worker database access exists.

The four-element Authority snapshot format is unchanged. Its context now includes the descriptor and mount mapping, so the existing SHA256/session binding freezes these together with job, attempt, generation, owner, staging, deadline, plan/input and profiles. Detached accessors prevent ordinary caller mutation. Execution/public messages retain v1.3.0; lifecycle frames retain lifecycle-v1. Supervisor requires fixture/descriptor agreement and an independently configured exact owned mapping before prepare. It defaults to no authorized real mounts.

Runtime mapping is deliberately fixed to `{bundle_id: '/models/bundles/'+bundle_id}`. Control uses a separately supplied sealed local path solely for preverification. Worker never receives that arbitrary host path. Installation must bind the identical immutable bundle read-only at the specified runtime path and explicitly initialize RuntimeSupervisor with this mapping. This lane does not add an installation/config-file route or qualify host/VM mount equivalence. Default installed topology remains closed.

Kokoro child uses existing authenticated session and exact worker-message oracle, invokes the reviewed callable with frozen authored text and mapping, then writes measured WAV with exclusive/no-follow staging, fsync and the existing result manifest. Child sets finite address-space, file-size and process-count RLIMIT ceilings plus explicit numerical-library thread environment before real backend import. These are source controls, not measured process-memory/containment evidence; actual engine allocation can fail below its needed budget. Backend materializes samples before wrapper frame checks. Existing networkless Linux/container/supervisor topology remains an installation prerequisite, untested here.

Admission replaces fixed synthetic demands/output checks only for the closed Kokoro profile, reserving descriptor memory and profile disk through ResourceStore. It requires explicit ready/installed/qualified trusted current profile and measured verification with a nonempty nonsynthetic receipt. Those trusted assertions are not independently established qualification. Tests deliberately use clearly marked injected-only assertion strings; no production readiness or actual receipt is created. Existing synthetic policy/path remains unchanged.

Core's minimal real-output dependency measures actual bytes/hash/frames/rate/channels and requires bounded 24-kHz mono PCM16, exact ordinary WAV serialized size and descriptor-derived provenance. Existing stage ownership, attempt/deadline/oracle identity, storage transaction, adoption and publication barrier remain authoritative. Publication alone cannot release executing capacity. Dispatcher terminal handling still requires exact confirmed reap receipt before reservation release/ACK. Cancellation remains durable intent with existing result/cancellation race semantics; confirmed cancelled/stale attempts cannot publish, and stop request alone cannot release capacity.

R191 metadata observation fixed: mutable AudioResult sample_rate/channels now require Integral and reject bool and numerically equal float metadata, before format equality. Backend close behavior is preserved.

## Explicitly authorized dependencies

Parent authorized minimal contract enum addition after the concrete interface need was identified: **only `generated` added to `$defs.provenance.properties.kind.enum` in `contracts/v1/schema.json`**, retaining all six prior values and their order. Generated `source_id` is `kokoro-execution-sha256:` plus canonical SHA256 of the frozen full Execution descriptor. Source SHA256 is measured output bytes; existing job/attempt/input/publication relations provide the remaining execution identity. No public caller source string is accepted by real publication. This denotes output origin, not evidence that an injected backend performed real speech.

Enum consumer inventory: central contract validation and typed reconciliation walk consume the shared schema; generator examples choose the first enum member (still synthetic); synthetic worker/scenarios/generator explicitly retain synthetic provenance; ingest explicitly retains upload provenance; lifecycle_cli remains its intentionally synthetic oracle; capabilities source_types describes acquisition sources and is unchanged. No parallel generated enum or client-specific provenance switch was found in the inspected client/service/worker/contract consumers. Isolated generator verification reproduced existing fixtures byte-for-byte; no fixture or operations contract changed.

Restart regression exposed the existing exact private-context reader and synthetic reservation digest assumption. Parent explicitly authorized the narrow `service/reconciliation.py` extension: accept only exact plan/bundles/execution keys with shared closed Kokoro validation, reject missing execution for kokoro-cpu, and reconstruct its configuration digest from canonical full execution snapshot. Existing plan/bundles context reading and synthetic effective-identity digest remain unchanged. Unknown keys, malformed descriptor/mapping and reservation mismatch stay blocking. No generic context bypass, schema migration, repair or cleanup was introduced.

## Verification

Inspected `tests/conftest.py` and `tests/service/conftest.py`; used isolated unittest only with `/private/tmp/pab-0c-venv/bin/python`. Final focused command:

`/private/tmp/pab-0c-venv/bin/python -m unittest tests.service.test_engine_admission tests.worker.test_engine_lifecycle tests.worker.test_engine_kokoro tests.service.test_provisioned_engine tests.service.test_provisioning tests.service.test_production -q`

**46 distinct methods PASS**. Includes provisioned local fixture to injected engine, strict metadata, immutable/detached Authority, identity/deadline/input substitution refusals, missing authority, closed mount and wrong bundle/revision, real-profile synthetic-qualification refusal, atomic reservation-trigger rollback, actual frames/hash/PCM bytes, output limits before staging, schema acceptance of generated/all prior provenance values, unknown kind refusal, tampered generated source/hash/frames/stale attempt refusal, cancelled stale result refusal, reservation retention before reap, dispatcher generation/artifact and cancellation through existing owner loop, reopen of the same data root, and malformed snapshot/reservation reconciliation negatives. Existing synthetic production lifecycle still passes.

Additional regression command:

`PYTHONPATH=.:tests/service /private/tmp/pab-0c-venv/bin/python -m unittest tests.service.test_production_admission tests.service.test_production_reconciliation tests.service.test_production_api tests.clients.test_production_clients -q`

**10 methods PASS**. The earlier command mistakenly named the client module under tests.service; corrected its path and reran the complete named regression group successfully. Early test runs also exposed test expectation/mocking issues and the concrete reconciliation dependency; final commands above pass. No test failures remain.

JSON schema meta-validation PASS. Ran contracts.generate in a separate owned temporary copy and compared every fixture byte-for-byte: no differences. `git diff --check` PASS. Fixtures clean only their own temporary storage, including restoring permissions on their sealed fixture bundle before removal.

Kokoro lifecycle tests replace socket authenticate/receive/send with inert mocks and assert the supplied session binding equals Authority binding; existing authentication implementation is reused unchanged. Their injected engine seam redirects the asserted fixed runtime mapping to the separately provisioned test directory, without creating `/models` or claiming a real mounted bundle. Dispatcher runtime/receipts are bounded fakes, explicitly source evidence. No new actual UDS peer authentication, PID1 launch/reap, RLIMIT enforcement, installed mount or containment behavior is claimed from these tests.

## Remaining parent-owned gates

Independent source review/adoption is next; include both the exact enum delta and exact reconciliation context/digest delta. Real expected pins/runtime versions, installed read-only mount topology and worker import dependencies, Linux resource/network/process containment, real audible speech, actual resource/performance/cancel/restart and installation/enrollment remain pending. No real backend/model import or inference, ML install, real network/download, Docker, listener, Linux sandbox-on-Mac, pytest/autouse guard/full suite, hardware or HELD operation occurred. No old bundle deletion, unrelated cleanup, delegation or source adoption occurred. Source completion does not waive any actual/install gate.
