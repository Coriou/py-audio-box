# H1 Pocket managed preparation and explicit audition — r257 Sol

2026-10-24. Implementation candidate for parent fresh independent Sol review. Source journey implemented and tested with inert injected audio; no self-adoption, actual speech/listening, installed Pocket activation or release qualification.

## Pins and authority

Owning worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-fb2cfaa6`, branch `t3/pocket-preparation-audition`, entered clean at exactly requested base **`fa2a524fe609cc7d6f42e31b3ec01f609743227e`**; verified HEAD and base ancestry before edits. Source commit **`89366614abfb566155e8aecb304fcd08918d4ff0`**. This report is a separate subsequent commit. No root/historical checkout write or delegation.

Read complete r188 including assignment 3, r235–r239 reference/publication contracts, r240–r243 adapter correction, r244–r254 lifecycle/post-reap/distribution limits, r255/r256 provisioning, and relevant actual source: contract/oracle, managed ingest/catalogue, production/contexts, admission/dispatch/Authority, Core/persistence, reconciliation/shared detached reader, child/supervisor/helper/journal, Pocket adapter/backend/provisioning and explicit package builder. Checked worktree and ancestors for AGENTS.md; none found. Bounded memory-registry search supplied no implementation premise. Inspected both conftests; used explicit unittest/direct calls, never pytest collection or its maintenance/listener fixtures.

Parent clarified that r257 retains the reviewed empty-transcript Pocket contract: nonempty `reference_text` must refuse before reservation, rather than be discarded. Parent explicitly authorized two evidence-driven shared dependencies during implementation: (1) narrow typed planned-output reference classification in `service/reconciliation.py`, and (2) narrow resource-monitor handling of legitimate concurrent empty conditioning-directory disappearance. No contract, DDL, queue, recovery framework, installer or provisioning edit.

Exact source delta (14 files):

- `worker/engines/pocket_prepare.py` (new), `worker/engines/pocket.py`, `worker/runtime_protocol.py`, `worker/runtime_child.py`.
- `service/execution_context.py`, `service/production.py`, `service/core.py`, `service/admission.py`, `service/domain.py`, `service/reconciliation.py`, `service/resource_store.py`.
- `headless-dist-manifest.json`, `tests/service/test_pocket_preparation.py` (new), `tests/release/test_release_package.py`.

`runtime_supervisor.py`, conditioning helper, journal, Dispatcher, persistence/recovery, synthetic workers, Kokoro adapter, backend and public request schemas remain unchanged. Supervisor's existing independent byte verifier accepts the exact new preparation type through the shared verifier; dispatch remains the existing `pocket` mode, so the reviewed post-reap helper ordering applies without another lifecycle mode.

## Prepared reference and spoken audition

`PocketPreparation` is a separate closed frozen descriptor, extending Pocket's model/profile/resource/original-reference binding with exact selected ticks and literal phase/transform `prepare` / `pcm16-mono-exact-segment-v1`. Here `reference_*` binds the complete managed original, while `sample_rate` binds the generated audition. No public paths, descriptor/control injection, seed, model fallback, cleaning claim or transcript interpretation is added.

The supported transformation is explicitly bounded **classic PCM16 mono WAV segment extraction**. Both 48-kHz ticks must map exactly to original frames (`tick * original_rate % 48000 == 0`); partial-frame selections refuse. Full original ID, SHA-256, size, frames, rate, channels, WAV subtype and exact segment are bound against managed owner metadata and strict RIFF validation before reservation. Original duration remains limited to the reviewed Pocket verifier's sixty seconds / 192 kHz. The cohesive preparation context additionally caps original bytes and audition maximum output bytes to **1,048,576 each**, preserving the existing pair publication intent's per-file ceiling rather than widening persistence. Profile upload/decoded/text/disk/RAM/thread limits are independently checked; reserved disk covers original/snapshot/output headroom. This bounded capability may refuse larger uploads even when generic ingest supports them.

The worker writes a canonical PCM16 mono WAV header at the **original sample rate**, containing every selected original PCM byte unchanged. Ancillary container chunks are not copied. There is no denoise, normalization, resampling, STT or acoustic quality claim. Core independently reloads the managed original and reconstructs this exact selected WAV during output verification; a coherently rehashed fabricated reference still refuses.

Production requires an explicit nonempty trusted `reference_roles.audition_text` for real preparation; it cannot fall back to the synthetic fixture phrase. The reviewed Pocket adapter's existing bounded synthesis body is shared through a private helper. It receives the selected prepared WAV snapshot and **exact** frozen audition text, with explicit local bundle/language/threads/private namespace. Its existing AudioResult validation, actual rate/frame measurement, finite samples, PCM16 serialization, hashing, close/unwind and private snapshots remain in force. Model imports stay worker-only/lazy. No new inference implementation or backend defaults are introduced.

Child stages only the fixed ordered `reference.wav` and `audition.wav`, after both engine results exist. Candidate roles bind the full original, exact ticks, original provenance and exact authorized empty transcript separately from generated audition text. Generated audition provenance binds the preparation descriptor; reference provenance retains original ID/hash/kind. Core measures both actual files, verifies their role/order/hash/format semantics and passes both hashes through the existing publication oracle and atomic pair adoption barrier. Existing failure/intent/quarantine behavior is retained; no half-pair metadata transaction exists. A failed audition returns no proposal and stages neither output. Partial staging/adoption remains evidence, never a published half voice.

Nonempty Pocket reference text returns `unsupported_parameter` before reservation and is exposed through the existing public `unsupported_control` error with an explicit `reference_text` explanation. No contract change. Empty transcript is retained exactly through candidate/save/generation; Pocket is never represented as consuming it.

Save still selects a committed candidate and registers the immutable prepared-reference bytes/hash/configuration/provenance, never the generated audition. Full original/segment association remains in the durable preparation manifest/context. Generation reloads the owned immutable saved version and uses its prepared reference. Trusted `PocketContexts` can now resolve either configured preparation or generation templates. The public fixture explicitly installs trusted templates after managed upload/save; this is not an installer or automatic readiness resolver. All authored requests still traverse existing HTTP/CLI/MCP → Dispatcher → Production → Core paths.

## Narrow shared-reader correction

The cancellation/reopen fixture reproduced two missing assets: the old generic typed walk interpreted frozen prepare role output IDs and resolved future reuse reference ID as already-existing input dependencies even when nothing had been published.

Only a schema-validated, oracle-validated frozen `voice_prepare` plan with exact `prepared_reference` roles enables special handling. The walker tracks schema paths and treats just `reference_roles.{reference_asset_id,audition_asset_id}` and `resolved.reuse_configuration.reference_asset_id` as output declarations. It does not subtract arbitrary IDs or alter original input, version, catalogue, artifact, manifest or publication-intent traversal. Published outputs remain independently required through committed documents. Missing original and missing committed output regressions remain blocking; failed/cancelled unpublished plans may lack their future outputs. Legacy absent-role and synthetic pair publication paths retain their established representations.

Inspected the equivalent detached/history reader: `service/recovery.py` imports this same `typed_references` after `_installed_detached_documents`. There is no independent incompatible walker to patch. Tests carry actual failed-job rows through both shared functions; existing complete installed-history tests also run. No recovery source or decoder waiver was added. Incomplete publication intents continue through the existing shared intent/barrier/quarantine logic.

## Narrow resource-monitor correction

Repeated public journeys exposed a concrete race in `ResourceStore.measure`: its path-based walk could enumerate `conditioning` immediately before the existing post-reap helper removed it, then latch a resource fault on FileNotFoundError. This was not fixed by deadline tuning or ignoring errors globally.

Only staging traversal now uses directory FDs, no-follow stat/open and before/after directory identity checks, bounded to 2,048 entries / one second (matching the existing bounded inventory scale). Core-root measurement and accounting transactions remain unchanged. Disappearance is accepted only after a no-follow directory observation for literal `conditioning` immediately below the exact current executing Pocket attempt: validated stored context/descriptor, single executing reservation, matching current stage ID, unchanged control-owned mode0770 stage on the bound root device, and verified continued absence. Unknown paths/files, symlinks, changed/reappeared identities, missing roots, nonempty/permission errors and uncertain observations still fault. There is no private-byte skip: existing directories are traversed and measured, and permission failures still refuse.

An absent empty directory occupies no current named bytes. Any size already observed remains counted; highwater never decreases and full reservation charges remain until the original confirmed-reap release path. This change grants no cleanup/death/ACK authority and does not waive helper failure. The trusted exclusive writer/supervisor premise remains; no hostile same-UID/concurrent privileged writer guarantee is claimed. If a filesystem cannot establish the no-follow directory observation before disappearance, uncertainty still refuses rather than inventing proof.

A deterministic test invokes the actual empty-dir helper filesystem logic between directory observation and open, producing the formerly fatal FileNotFoundError. Monitor remains healthy, highwater conservative and reservation executing; release without reap still refuses. Negative interleavings cover unexpected disappearance, real ENOTEMPTY with opaque bytes retained, PermissionError, directory substitution, wrong stage mode, missing root and symlink. Public fixture ownership injection is restricted to its helper thread so concurrent owner measurement sees real fixture metadata.

## Executed verification

All commands use **`PYTHONDONTWRITEBYTECODE=1 /private/tmp/pab-0c-venv/bin/python -I -B`**, actual **Python 3.14.7**, owning cwd and explicit absolute sys.path entries for this checkout and `tests/service`. Direct resource wrapper additionally includes checkout `tests` to resolve existing test-only imports. No pytest collection, listener or maintenance guard activation.

Final focused suite: **142 tests PASS**, 34.196 s, exit0; `/private/tmp/r257-final-focused.log`. Modules:

```
tests.service.test_pocket_preparation
tests.worker.test_runtime_conditioning
tests.service.test_pocket_lifecycle
tests.service.test_engine_admission
tests.worker.test_engine_lifecycle
tests.worker.test_engine_pocket
tests.worker.test_engine_kokoro
tests.service.test_production
tests.service.test_production_admission
tests.service.test_production_reconciliation
tests.service.test_production_api
tests.service.test_reference_publication
tests.worker.test_production_synthetic
tests.contracts.test_reference_roles
tests.clients.test_production_clients
tests.release.test_release_package
```

New preparation module contributes **12 methods**, including exact source PCM/ticks/transcript/provenance, full distinct-output semantic bindings, exact audition text/conditioning bytes, measured 24-kHz reference versus 22,050-Hz audition, partial ticks/formats/foreign/tampered/qualification/mapping/cap refusals before reservation, supervisor verification-before-journal, frozen plan/Authority text binding, failure atomicity, cancelled/stale output fences, opaque interruption UNKNOWN, immutable save/reopen/generation, failed/cancelled unpublished references, missing original/committed output and deterministic resource interleavings.

The public journey uses real in-process ASGI, HTTP client, CLI and MCP adapters, owner Dispatcher, Core/SQLite/media/oracle and immutable catalogue. It uploads managed PCM, prepares/results/saves/generates, reopens the same database/staging root, generates again, cancels preparation with an existing namespace and delayed injected reap, verifies reservation retention and no half metadata, reopens the same root again and generates from the saved prepared reference. Backend and transport/death observations are injected. The actual exact-empty helper filesystem logic runs only on owned tiny fixture directories with injected identity policy; no actual UID switching/PID1/authentication/containment evidence follows.

Release suite builds/extracts the actual explicit archive, checks exact source bytes/membership and isolated local import closure including the new preparation module, with heavy imports forbidden. Existing synthetic pair and Kokoro paths pass. Expected injected `lifecycle supervisor unavailable; lane quarantined` diagnostic occurred with overall PASS.

Additional direct existing resource regressions: **17 cases PASS**, `/private/tmp/r257-resource-regressions.py` and `.log`. Explicit allowlist covers quota/free/growth/root disappearance, reserve rollback, monitor/fence persistence, nonrollbackable snapshots, unresolved cancel/acquire/release, inherited protocol damage, acquisition rollback and observation outside SQLite transactions. No crash/death-test function is selected; importing test modules does not execute them.

Final installed restart/registration/host-policy suite: **87 tests PASS**, 63.010 s, exit0; `/private/tmp/r257-final-installed.log`. Exact modules: `tests.service.test_installed_restart`, `tests.service.test_installed_registration`, `tests.service.test_installed_host_policy`, with the same isolated interpreter and absolute checkout/tests-service paths. Final total: **246 passing tests/direct cases** (142 + 87 + 17), with additional mutation/interleaving loops inside methods. Earlier 137-method run passed before the added resource tests; it is superseded by the final 142. Early failures were corrected: missing fixture `wav_pcm16` setting; direct fixture ACK/job-list shape; double-close test cleanup; planned output reader mismatch; a frozen-text negative initially expected a new Authority despite stale input hash (now asserts refusal, then changed binding after valid rehash); shadowing unittest.fail with a fixture flag; and the real concurrent-removal race above. Initial FD-scan tests exposed inconsistent raw DirEntry/lstat versus injected stat/fstat metadata; using descriptor-relative no-follow `os.stat` consistently corrected the fixture seam without relaxing production identity checks. A direct wrapper initially omitted checkout/tests import path and failed before any case ran, then passed with the explicit path. These failures are not represented as clean first-run evidence.

`git diff --check` / staged whitespace PASS. All **13 changed Python files parse with Python 3.11 grammar**; no Python 3.11 runtime execution. Available `/private/tmp` space observed **3,726,552 KiB**, below the 6-GiB actual gate. Only small inert fixture media, SQLite and source archives were used, with ordinary owned temporary teardown. Existing synthetic workers and inert import/help interpreter probes may run; no actual model/ML, network/download, Docker, install, listener, process-crash exercise, operational cleanup, HELD access or delegation.

## Parent handoff and limits

Review source `89366614abfb566155e8aecb304fcd08918d4ff0` freshly with Sol before adoption, especially the two expressly authorized shared-reader/resource dependencies. No self-GO is claimed. Actual operator Pocket pins/runtime/config availability, installed mappings/caps/UIDs/dependencies, real authentication/reap/resources, spoken-text intelligibility and clone similarity/listening remain later separately authorized gates. Inert waveform samples establish forwarding/measurement/source behavior, not speech correctness.

Post-reap namespace policy is unchanged: clean completed empty-namespace success/cancellation may reopen; interrupted nonempty WAV/YAML snapshots and mid-flight restart remain conservative UNKNOWN. No recursive cleanup, old-state repair or proof from leader exit was added. UI, STT, Qwen and hardware-family qualification remain outside this lot.
