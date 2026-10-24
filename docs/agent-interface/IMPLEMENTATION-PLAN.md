# Headless release implementation plan

2026-10-24 · **Canonical release scope, r157.** The new user priority supersedes the earlier full-studio release plan and diagnostic-first sequencing in r146/r147. [PRODUCT.md](../../PRODUCT.md) defines the three binding priorities; [IMPLEMENTATION-START.md](IMPLEMENTATION-START.md) is the entry point. [EXECUTION-LEDGER.md](EXECUTION-LEDGER.md) and historical reviews remain evidence records, not extra feature prerequisites. This is a source/docs replan, not operational authority or release acceptance.

## 1. Required outcome and explicit deferrals

Ship existing useful fixes and audio functionality through a complete HTTP API, small CLI and MCP; deliver working qualified CPU, CUDA and Apple Silicon native Metal profiles; provide clean voice cloning and management. The future web client consumes this same public API.

Required headless journeys:

1. Install, authenticate, discover actual readiness, explicitly provision, generate with a known voice, wait and retrieve verified audio.
2. Upload/import a reference, inspect/select a segment, prepare a clone, explicitly generate an audition, save/reuse its immutable version and manage its name/aliases/archive/export/import.
3. Register built-in voices or design candidates where the declared engine supports them, audition and explicitly select/save; preserve provenance and compatibility.
4. Submit a script/beatsheet with speakers, generate/select takes, stitch/assemble/export audio, revise one line without losing other takes and retrieve the frozen result.
5. Disconnect/reconnect, cancel or recover an interrupted job, restart/upgrade/restore the installation, retain voices and finished artifacts, and refuse stale/foreign publication.

**Deferred:** all shipping web UI, studio, browser accessibility, visual timeline/drag/keyboard editor, browser microphone capture and browser acceptance; public STT/dictation/TxCode; remote cloud orchestration, multi-tenant SaaS, training and arbitrary executable workflows. Existing file/URL acquisition, script/stitch/audio operations and internal transcription/QA/alignment are retained. External completion webhooks/host wakeups are optional; durable job events and retrieval are required. No UI work is on the critical path. Preserve existing evaluation UI files as history; the headless release must not require building or serving them.

## 2. Verified starting point and integration status

Read-only source baseline: root `feat/agent-voice-workspace`, HEAD `9b39f6061abebd6fdfd90885ee25d4aa75cdb28c`. Recheck before a future implementation commission; documentation may advance independently.

- `service/api.py` explicitly identifies itself as a synthetic 0C fixture with no auth, mutation, DB or ML runtime. Its four public reads are capabilities, voices, asset metadata and asset content. Skeleton diagnostics and evaluation UI are not production operations.
- `clients/http.py`, `cli.py`, `mcp.py` provide bounded fixture reads through HTTP. They are reusable transport/client scaffolding, not full service coverage.
- `contracts/v1/{schema,operations,errors,legacy-controls}.json` and fixtures describe a broader intended interface; declaring an operation does not implement it. `service/core.py` already has jobs/events, project revisions, synthetic submission, cancellation, publication and reconciliation. `service/{persistence,recovery,dispatcher,admission,resource_store}.py` and `worker/` contain useful foundation work. Reuse it; do not build a parallel job/store system.
- `service/compose.release.yml` is still a synthetic control/fake-worker topology. Its filename does not prove a production install or persisted service DB.
- Existing source in `apps/voice-{synth,clone,register,split,cpu}/`, `apps/audio-export/`, `apps/job-{runner,watcher}/`, `lib/{voices,tts,audio,fidelity,jobqueue}.py` and `lib/tts_backends/` is the compatibility inventory, including beatsheet workflows. Verify concrete commands and dependency closure during H0, not by importing/loading engines during planning.

Ledger-carried status: accepted source checkpoint was adopted into root at `ec78661fcb6fb9cbf4a569ea65cd3e446786702d` after r149/r150, retaining the 33 candidate blobs and root documents. Preserve W2/R1/A1/A2/B1/B2 bounded accepted fixes and lifetime/ownership/publication corrections at their stated evidence tiers. CLOSED81 source `86c7c744` (docs `c6354dd0`) is source/inert accepted through r154 but **not root adopted**. At entry r155 private preparation was in r156 review. The parent concurrently recorded r156 bounded SOURCE/private-closure/retained-inert GO; this is not build/runtime acceptance or source adoption. Separate C/T2 source `1d741cb` is not silently integrated. No operations are known running from the current handoff; no live probe was performed. All historical resources/proofs and the lane remain **HELD**, including untracked STT work.

Historical campaign remains **17 PASS /018 FAIL /81 UNRUN**; separate017 and accepted extended018 are separate results. No all100, combined T2, native-cause or release claim follows. r146/r147 remain useful accounting references; their mandatory studio and “finish diagnostics before product source” ordering is superseded here.

## 3. Public contract and retained safety invariants

One Python domain/control owner, SQLite metadata on persistent local storage, immutable managed media, typed supervised worker protocol. Workers do not open the DB. HTTP is the authoritative interface; CLI/MCP have no private business state or direct storage access. Keep legacy Redis workflows independent; no silent bidirectional registry synchronization. Import legacy identities through a previewable, explicit migration with conflict handling.

Required API groups: cheap capabilities/readiness and profile detail; voices/immutable versions/aliases/archive/import/export; managed media upload/import/segment selection/content; prepare/design/audition/generate; scripts/speakers/blocks/takes/assembly/stitch/export; jobs/list/status/wait/cancel/retry/events; artifacts/download/manifests; explicit provisioning and scoped administrative lifecycle/migration operations. Every shipped operation has equivalent CLI/MCP access. A typed operation endpoint is acceptable; arbitrary argv, filesystem paths or generic executable command buses are not.

H0 freezes a finite operation coverage map against existing source and contracts. Mark each entry implemented, planned required, deferred editor-only or preserved legacy compatibility, with reason and owner. Full coverage means every in-scope audio/control operation, not a claim that all old editor-only schemas already execute. Do not silently discard an existing legacy capability: retain its command and implement the corresponding typed API bridge for release; unsupported engine controls return explicit capability/errors. Pure editor-only gesture/history scaffolding may be deferred; headless exact script revision, take selection and assembly remain required.

Voice versions bind exact source/reference bytes, origin, model/engine compatibility, conditioning and settings. Rename/alias/archive cannot change completed jobs; deletion is explicit and dependency-aware. Preparation and audible test are distinct operations. No remote pickle or unsafe prompt deserialization. Record exact authored text; any requested normalization is a separate value.

Mutations use revision preconditions and principal/operation-scoped idempotency. Same key/same canonical request returns the same result; changed input conflicts. Retain deduplication for the job lifetime and disclose expiry. Intentional alternatives use new requests. Errors have stable code, field, corrective action and bounded detail across all clients. No silent overwrite or automatic selection of a late take.

Durable job states distinguish queued/running/cancel_requested from terminal success/failure/cancelled. A cancellation receipt is not proof of process death. Fence stale incarnations; confirm ownership and stop/reap before releasing capacity or deleting staging. Unknown owner state quarantines the lane. Publish verified bytes and hashes before committing success; reconcile the filesystem/SQLite crash gap, missing media and orphans without false success. Preserve unexpected-signal detection outside expected-refusal handlers. Default heavy retries remain one attempt; explicit retry freezes inputs and links attempts.

Persist ordered cursor-based events with retention/gap semantics. Bounded wait/long-poll returns terminal state or a resumable cursor; client disconnect never cancels work. CLI/MCP handle waiting outside model turns, propagate deadlines and cancellation, and return artifact references rather than inline audio/logs. Initial compactness budgets: discovery/onboarding ≤2,500 model-visible tokens, receipt ≤2 KiB, status ≤1 KiB, list page ≤20, with host duplication counted. Adjust only from measured successful tasks.

Loopback default and authenticated named revocable credentials with read/production-write/admin scopes; documented HTTPS for trusted-team remote access. Protect artifact/range reads and reject secrets in URLs/logs/errors. No unauthenticated non-loopback mode. Browser sessions/CSRF implementation waits for the future browser client; credential, host/network and media protections do not. Remote acquisition allows bounded declared HTTPS media/importers, validates redirects/DNS/private destinations, caps bytes/time/decode, isolates decoding and never exposes a request-level SSRF bypass.

## 4. Hardware selection, provisioning and performance

All three families are mandatory release acceptance rows:

- **CPU:** constrained Intel Mac Docker/VM and an actual small Linux CPU host, with named ISA/RAM/VM/disk/threads. No inferred host specification or STT-to-TTS extrapolation.
- **CUDA:** available NVIDIA Linux host with compatible driver/runtime, pinned worker image/model/precision and real inference. Windows/WSL2 or additional GPUs are only supported after separate qualification; one selected CUDA profile must work for release.
- **Apple Silicon:** macOS arm64 packaged native Metal worker plus Docker control plane. CPU-only arm64 Docker is useful but insufficient for this row. Bundle/manage the native runtime and dependencies; no undocumented manual ML setup. Implement authenticated scoped local transport, explicit asset transfer/path mapping, child ownership/cancel/reap and control-owned publication. Native isolation limits must be stated honestly; Linux namespace claims do not apply to the native worker.

Discovery distinguishes supported, installed, provisioned, available, ready and qualified, with reasons and qualification timestamp. Select requested voice/version/model first; filter by engine/operator/device/precision compatibility and provisioned assets; then choose a qualified measured profile within memory/resource constraints. Explicit override wins if valid, otherwise fail with a corrective action. An unavailable accelerator may fall back to CPU only within an explicit compatible policy preserving identity/model/settings and disclosing the selected backend; otherwise request an explicit alternative. Never download or silently choose another voice/model to make a request succeed. Unmeasured profiles are labelled unqualified, not “best”.

Provision deliberately: pinned dependencies/models, license/redistribution check, resumable staging, checksum verification, atomic activation, disk failure preserving the previous ready bundle. Ordinary connection/list/generate does not provision. Control image stays lightweight; worker has no service credentials, DB or Docker socket. Reuse static supervisor/typed transport where compatible, with private UDS/networkless Docker worker as the baseline and explicitly qualified native transport.

Constrained CPU starts with one expensive slot and bounded threads/RAM/disk; cache compatible resident weights and exact-reference conditioning. Stronger hardware uses measured model-specific concurrency, memory admission and independent lightweight request responsiveness, not a global permanent CPU cap. Benchmark cold/warm load, queue, generation/export latency, real-time factor, throughput, memory scope/peak limitations, model-load count and failures at concurrency 1 and a bounded higher profile. Compare identical text/voice/model/settings; model alternatives require explicit quality/identity comparison. Human listening checks intelligibility, clone similarity, cuts and final output. No “best possible” or universal speed claim.

If CUDA/Apple hardware is unavailable: implement/package adapters and synthetic protocol coverage now; record exact missing host/runtime/model/provision/access row, finite smoke/benchmark/listening work remaining, and qualification NO-GO. Do not downgrade these families or rent cloud resources. Release remains blocked until all three actual rows close.

## 5. Finite phases and implementer allowlists

These are prospective package boundaries, not permission to execute them in this docs assignment. Parent pins the actual base and exact changed-file subset before each commission. New filenames below are planned, not existing commands. No diagnostics/private-bundle writes are implied. Each package uses Sol-low implementation then a separate Sol-low review; Astra reviews material integrated milestones or architectural disputes only.

### H0 — Inventory and first durable voice slice (NEXT)

Allowlist: `contracts/v1/{schema,operations,errors,legacy-controls}.json`, `contracts/fixtures/v1/`, new `service/voice_store.py`, existing `service/{core,persistence}.py` only for migration/store integration, new `tests/service/test_voice_store.py`, and `docs/agent-interface/IMPLEMENTATION-PLAN.md` for the finite operation map. Do not edit recovery/guards/diagnostics/worker or legacy registries in this unit.

Deliver the bounded existing-command→API/CLI/MCP map and one durable voice catalogue slice: register metadata pointing only to validated managed assets, list/get immutable versions, aliases and archive with conflict/dependency checks. Reuse the current DB transaction/migration boundary and existing voice semantics; no model call, file acquisition or new job runner. Refuse absent/unmanaged assets. Keep recovery-sensitive activation dormant pending H3/H5; repository schema changes are not a deployment.

Exit: restart retains versions/aliases; duplicate/revision races are deterministic; archive does not break a prior reference; migration preserves existing records. Focused store tests with temporary DB/media and existing affected core/persistence tests under separately authorized test work. No full suite, Docker, audio model or 81 campaign required for this source unit. This is useful implementation feasible now, not another research/preparation ladder.

#### H0 finite command and operation coverage (r159)

`contracts/v1/legacy-controls.json` retains the exact existing control/default inventory and adds the single finite `headless_operation_coverage` map (every current catalogue operation exactly once), `headless_legacy_bridges` and `headless_missing_contracts`. Based on r158 and the actual argument parsers; no command execution or model probing. All legacy commands remain unchanged compatibility entry points. Required bridges have API/CLI/MCP parity owned by H2 against H1 domain and H4 engine capabilities.

| Actual legacy surface | Required headless bridge / owner |
|---|---|
| voice-synth list-voices/list-speakers/capabilities/register-builtin/speak/design-voice/rename-voice/delete-voice/export-voice/import-voice | Discovery/profile detail, catalogue register/save/version/alias/archive/exchange, prepare/design and speech; H1/H2/H4. Rename changes aliases; archive replaces destructive delete while preserving references. |
| voice-clone migrate-registry/synth/prepare-ref/self-test; voice-register one-shot; voice-split one-shot | Explicit migration; managed upload/acquire/segment; clean-reference prepare, optional VAD/separation/transcription, explicit audition/save. Self-test stays a legacy smoke; H1/H2/H4. |
| voice-cpu speak/batch | Typed generation with explicit compatible CPU profile and homogeneous resident reuse; H1/H2/H4. |
| audio-export one-shot; existing concatenation helpers | Managed-ID ordered stitch/assembly/export, pauses, sample/rate/channel validation, PCM24/resampling/loudness; H1/H2. |
| job-runner plan/enqueue/enqueue-beatsheet/compile/run/execute-manifest/status/result/report/history/retry/flush/package-status; job-watcher daemon | Scripts/blocks/takes/assembly; durable jobs/list/wait/events/cancel/retry/artifacts and scoped administration; H1/H2/H3. Preserve Redis commands/consumer independently; no public argv, daemon runner or silent queue migration. |

Catalogue coverage is finite: four current synthetic reads (`discover`, `voices_list`, `asset_get`, `asset_content`) retain fixture status; voice/media/production/control groups in the JSON map are planned required. Only `clip_put`, `clip_split`, `clip_delete`, `track_put`, `track_delete`, `project_ripple`, `project_restore`, `project_history` are deferred editor-only scaffolding. Headless script revisions, take selection, ordered assembly/render and revision retrieval remain required. Missing typed contracts are explicitly voice aliases/version listing/import/export, segment selection, audio export, job listing/wait/events and profile detail; H2 owns additions, H1 supplies domain. No listed name is claimed callable HTTP.

H0 implements trusted `Core.voices` methods: `register(principal,key,version,expected_revision=...)`, `get`, `version`, `versions`, `list`, `resolve`, `aliases`, `archive`. Register appends immutable metadata and verified asset digests; revision zero creates an identity, subsequent versions require the current revision. `voice_save` still requires H1 candidate selection, then calls register; it is not redefined as arbitrary metadata HTTP input. Principal-local aliases reserve canonical IDs and archived names. Archive hides new name resolution but keeps direct immutable version reads and all dependencies; there is no hard delete. H2 must authenticate principals and use explicit adapters. H1 ingest writes optional asset `principal_id` only after trusted managed publication; historical published assets derive ownership from artifact/job records (compatibility scan capped at 1,024, otherwise `resource_limit`). No acquisition/model validation or readiness is implied. Versions freeze identity/configuration/settings/provenance/reference hashes for future workers; actual profile compatibility and reference media preparation remain H1/H4 gates.

### H1 — Coherent media, voice and production domain

Allowlist: `service/voice_store.py`, new `service/{ingest,production}.py`, `service/{core,persistence}.py`, existing `lib/{voices,audio,fidelity}.py` only for reviewed extraction, relevant `contracts/v1/` and `contracts/fixtures/v1/`, `tests/service/` with named ingest/production/voice tests and affected existing legacy tests. Shared schema changes stay with one integration owner.

Implement managed local upload/ingest and bounded remote acquisition; preserve originals/provenance and explicit segment selection. Add prepare/design/generate trusted plans with frozen settings, immutable results and explicit voice save/test. Provide script/beatsheet mapping, speakers/blocks, take selection, stitch/assembly/export using existing audio behavior. Use synthetic adapters until H4 engines land. Preserve Qwen defaults, CPU alternatives, voice management and Redis CLI behavior.

Exit: one reference→voice-version→synthetic generation→stitch artifact journey survives restart; exact text/revisions, duplicate requests, stale takes, invalid media, limits, unsafe remote input and missing assets are covered. Synthetic impulses/sines verify sample count, trim/gain/silence/overlap for shipped operations. No browser editing suite. Only affected legacy regressions during development.

### H2 — Complete HTTP plus thin CLI/MCP

Allowlist: `service/{api,catalogue,domain}.py`, new `service/auth.py`, `clients/{http,cli,mcp}.py`, owner-approved contract fixtures, `tests/service/test_adapters.py` and new `tests/service/test_api.py`, `tests/clients/`. Integrate via H0/H1 domain interfaces, not duplicated storage logic.

Wire every required operation to real domain methods; add credential transport, uploads/downloads, structured errors, pagination, bounded wait, cancel/retry and durable event cursor resume. Keep MCP typed and compact with on-demand detail; cover all required operations without sending audio/base64 or verbose schemas by default. No skeleton receipt masquerades as production success.

Exit: HTTP/CLI/MCP traverse the same IDs/jobs/artifacts; a contract coverage assertion catches missing adapters, and representative end-to-end cases cover clone preparation, generation, stitch, cancellation and retrieval. Test auth/scopes/media, conflict/idempotency, disconnect/resume, event gaps, transport failures and bounded output. Use synthetic service integration first; measure actual agent first/returning tasks at H5.

### H3 — Install, persistence and trusted operation

Allowlist: `service/Dockerfile.release`, `service/Dockerfile.release.dockerignore`, `service/compose.release.yml`, `service/{persistent_init,persistence,reconciliation,recovery}.py`; `service/{worker_supervisor,supervisor_runtime,worker_transport}.py`, `worker/` only named lifecycle/transport files, dependency locks, `tests/service/{test_persistence,test_recovery,test_packaged_lifecycle,test_worker_transport}.py`, new `tests/release/`, deployment documentation. Parent must enumerate literal paths before dispatch.

Ship clean headless images/entrypoints and persistent metadata/assets/voices/models/staging volumes without source checkout, writable source mounts or Docker socket. Preserve accepted recovery fixes. Finish necessary bounded transient cleanup policy (never execute against HELD resources), installed topology, same-volume restart, backup/restore and upgrade/rollback. Reconcile separate C/T2 source through reviewed dependency comparison, not blind import. Provisioning and native packaging land with H4. H3 first proves the installed persistence boundary using the synthetic worker; final provision/offline-inference acceptance joins H4, so packaging source does not wait for real-engine qualification.

Exit: clean install, explicit provision, offline inference where models permit, repeated restart, upgrade and fresh-root restore preserve voices/jobs/media; disk pressure fails admission safely. Product fault matrix in section 6 passes on the installed topology. Model weights need not be redownloaded for every test. No browser assets/build gate. Existing prototype Compose evidence is not this exit.

### H4 — Real engines and three hardware families

Allowlist: new `worker/engines/`, new `worker/native/`, `worker/{runtime_protocol,runtime_child,runtime_supervisor}.py`, new `service/{profiles,provisioning}.py`, `lib/tts_backends/` and `lib/tts.py` only named adapter extraction changes, packaging files explicitly pinned by parent, `tests/service/` named profile/provisioning tests, new `tests/worker/`, `tests/release/`, profile/install docs. Do not rewrite legacy app defaults. Shared protocol changes merge before consumers.

Wire existing Qwen/Kokoro/Pocket capabilities as applicable; provision exact bundles, implement compatible auto/override selection, resident reuse, constrained and stronger hardware admission. Package CPU/CUDA and native Metal implementations against one typed protocol. Establish exact supported feature/profile matrix; preset-only support is not advertised as cloning. Clone/design need actual capable-profile evidence, and each required family needs real speech plus its advertised voice operations.

Exit: deterministic selection/override/incompatible-voice/provision-failure tests plus actual install/generate/cancel/restart on each required family; CPU responsiveness, bounded higher-concurrency measurements and human listening. Missing hardware leaves only actual qualification blocked, not source work. No private harness or new paid host.

### H5 — Integrated release evidence

Allowlist: integration conflict resolutions limited to reviewed H0–H4 files, `tests/release/`, `docs/agent-interface/` release/profile/acceptance records and release packaging pins. Semantic integration deltas receive separate review.

Run the complete offline regression suite once on the integrated candidate under separately bounded authority, plus relevant legacy Redis tests if affected, installed product fault matrix, all headless journeys and three real hardware rows. Existing `scripts/run-unit-tests.sh` builds Docker; inspect before commissioning. Do not run remote/Vast or legacy `scripts/run-e2e-test.sh` as an ordinary local gate. Add/reuse only the minimum public product smoke runner needed; no new private proof framework. Record exact source/image/model/hardware and command outcomes; finish with integrated Astra review.

Exit: every required operation has HTTP/CLI/MCP coverage, installed durability/security and migration evidence, three qualified hardware families, measured agent/performance behavior and human listening. Unrun required checks remain blockers. No release-complete claim from source or fake-worker evidence alone.

## 6. Risk-based replacement for diagnostic-first gates

The parent may change engineering test selection under the new priority without asking for a scope referendum. This plan removes CLOSED81/native-diagnostic completion as a prerequisite to independent feature source work. It does **not** waive production safety or convert historical failures to passes. Before release, record a finite old-risk→replacement-test map on the integrated candidate; independent Sol review must verify dependency closure and uncovered risks. Until that map has actual passing evidence, the associated production gate is open.

- **81 diagnostic controls:** parser/code-authority/trace/witness/guard mutations and natural74 scheduling certify the diagnostic harness. Defer the blanket 81 campaign as an internal diagnostic obligation; retain exact source acceptance, 81 UNRUN and r155/r156 preparation unchanged. No automatic adoption/build/run, new instrumentation or private successor. If any harness-dependent acceptance is reused, its required negative controls remain prerequisites for that claim. Do not label deferred controls passed or silently change their oracle.
- **Production timeout/ownership:** retain bounded deadline, unmodified resource limits, correct child ownership, stop/reap, foreign/stale refusal, no late publication and quarantine-on-unknown. Replacement product tests exercise the actual installed supervisor and blocked child, normal completion, timeout/cancel, stale incarnation, missing/mismatched receipt and unexpected signal, with wall-clock bounds. A child need not outrun cleanup to emit a diagnostic natural74 witness; product success must still require real confirmed output. Separate this assertion from witness certification explicitly.
- **T2/C durability and recovery:** required before release wherever the shipped path uses it. Cover positive startup and two restarts on the same volumes; adoption/fsync/DB ambiguity before/after commit; orphan/missing media; owner/receipt mismatch and inherited checking; incomplete/cross-volume recovery; death-barrier/descriptor/reader lifetime refusal; preservation of existing global inventory (A1 gap); bounded cleanup only after reference/grace/identity checks. Select representative boundary cases plus changed-path regressions, not every combinatorial diagnostic permutation. A reduced matrix needs an explicit risk-equivalence review, never a bare waiver of 108/636/216 historical families.
- **Native V1:** predecessor setup failure and snapshot/GC/collection SIGSEGV observations remain distinct and undiagnosed. Required replacement is real positive startup/snapshot where shipped, repeated job/cancel/restart and direct unexpected-signal detection on the selected runtime. A crash or ambiguous ownership is a failed gate, never caught as an expected refusal.
- **Native V2:** SQLite UTF8 failure did not reach the intended half-startup hook. Require installed control DB integrity and interrupted receipt/commit recovery with verified content across restart; it is not covered by a clean inference smoke.
- **Native V3:** missing SHM capture did not prove reader lifetime/refusal. Require real reader/owner lifetime and consistent WAL backup/restore behavior on the shipped DB topology. A private capture-file assumption may be deferred if absent from production; the integrity/refusal property cannot.

Docker control on macOS uses Linux SQLite/recovery, while the packaged native worker must not own/open that DB. Therefore reproducing every historical Darwin control-DB test is not automatically a Metal release gate. Parent must demonstrate this deployment boundary and separately qualify native worker authentication, asset handling, ownership, stop/reap and signal behavior. If native code actually uses the failing snapshot/reader path, its relevant failure is release-blocking until resolved and validated. Linux success never diagnoses or clears Darwin crashes.

Evidence tiers: static/source and inert tests; integrated synthetic/process tests; installed real runtime/hardware; actual agent workflows; human listening. Keep each separate. Historical browser evidence remains historical only. Tests for prose are unnecessary. No harness count, AST equality or source review substitutes for product measurements.

## 7. Parallel delivery within four slots

After the H0 contract/store seam is reviewed, use **two independent feature lanes**, with the parent as integration/schema owner. At most parent + two implementers + one rotating reviewer = four slots. Each lane gets a separate Sol-low review after implementation; the same agent never reviews its own lot. Pause/finish one implementer when an integrated Astra milestone needs a slot. No delegation is performed by this planning assignment.

- **Lane A — voice/media/domain:** H1 on the reviewed H0 store/contract seam; owns voice_store/ingest/production and its tests. Provides frozen domain fixtures to B. Does not edit API/client or worker packaging files.
- **Lane B — HTTP/CLI/MCP:** H2 against those frozen interfaces and synthetic fixtures; owns API/auth/client files and adapter tests. Does not edit domain DB/recovery or engines. Shared schema changes return to the parent, then merge first.

After A completes, reuse that lane for H4 engine/profile/native work; B completes clients then takes H3 packaging/persistence with explicit ownership transfer. Engine protocol and packaging collisions are sequenced by the parent. Hardware qualification can wait for a named host while source lanes proceed. Astra milestones: integrated headless synthetic journey with durable install; then real three-family release candidate. No Astra report after every unchanged preparation.

## 8. Remaining unknowns and handoff discipline

Finite unknowns: exact existing-command coverage gaps (H0); current hardware/access and model/operator compatibility (H4); C/T2 integration and installed fault evidence (H3); measured usable limits and acoustic acceptance (H4/H5); any later operational use of the now source-reviewed r155/r156 preparation (off the feature critical path). None requires re-opening studio scope or a research ladder.

Every commission states base SHA, literal file allowlist, preserved defaults/resources, interfaces, meaningful checks and their authority/cost, source versus runtime evidence, integration order and one next owner. Preserve all historical worktrees/manifests and untracked STT. No new private bundles, lane actions, cleanup, cloud spend or operational continuation follows from this plan. Current NEXT is H0 source implementation, not an 81 build. Release remains NO-GO until H5 evidence closes.
