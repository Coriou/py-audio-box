# H0 durable voice catalogue — r159 Sol implementation

2026-10-24. **Implemented source with focused isolated unit PASS; independent Sol adversarial review and parent adoption remain pending. Release/production activation NO-GO.** No public HTTP mutation or complete speech API is claimed.

## Pins and scope

Supplied fresh worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-e1b44469` started clean at `b66264b3925ae17e736273e20b8161ebfb2870e4`, equal to latest root `/Users/ben/Projects/py-audio-box` at entry and pre-commit recheck. No reset, historical checkout modification or diagnostic branch adoption. Own branch is `t3/durable-voice-catalogue` (initial thread branch name `t3/e1b44469`). Source commit: **`a6702296dcfd0a54612dafa123007edbe6286eb3`**. This report is committed separately afterward; only this report is copied to root.

Read PRODUCT, IMPLEMENTATION-START, canonical IMPLEMENTATION-PLAN sections 1–5, r157/r158, actual core/persistence/contracts and legacy parser declarations. Preserved latest integrated fixes and docs. No delegation, Docker/build, worker/model/network invocation, provisioning, deployment or push. No recovery/guard/diagnostic/worker/lib/legacy-registry edits or held-resource access. No dependency on CLOSED81/private bundles. `service/persistence.py` needed no edit: the implementation reuses `Core.transaction` and its existing P1/inspect-only boundary.

Source changed exactly nine allowed paths: `contracts/v1/{schema,operations,errors,legacy-controls}.json`, new `contracts/fixtures/v1/voice-catalogue.json`, new `service/voice_store.py`, `service/core.py` (three integration lines), new `tests/service/test_voice_store.py`, and IMPLEMENTATION-PLAN (finite coverage map only). No API/client implementation.

## Delivered behavior

`Core.voices` is a trusted owning-thread domain object. Additive SQLite schema installation runs atomically through existing BEGIN IMMEDIATE/commit/rollback handling, retains core user_version 1 and existing tables/records, refuses partial/unsupported catalogue schemas, and does nothing to pre-P1 roots. It does not enroll persistent service operation or remove activation fences.

Registration appends contract-validated metadata; it acquires no files and performs no inference. Existing IDs, exact authored reference text, origin, configuration, conditioning, model/engine/bundle hash, settings/hash and provenance are copied canonically without substitutions. Every reference must exist in managed asset metadata and pass actual `Core.asset` DB/schema/no-follow/inode/size/SHA verification. An ID-shaped string or a loose file is insufficient. Ownership is checked against trusted optional asset `principal_id`; older published assets derive their principal from artifact/job records. Ownerless, foreign, absent, unmanaged, missing, symlinked and tampered references fail. Historical ownership fallback is capped at 1,024 artifact rows, otherwise `resource_limit`; H1 ingest should write principal ownership directly. This is existing managed-byte validation, not new audio decoding or model compatibility qualification.

Stored versions add `reference_asset_sha256` and cannot be UPDATEd or DELETEd through normal SQL because of immutable-row triggers. There is no version mutation or hard-delete method. Alias changes reserve principal-local canonical IDs, detect collisions and use optimistic revisions. Archive hides name resolution/new use through aliases, reserves names and retains direct version reads and every previous reference. No dependency is deleted or rewritten. Version selection for completed/frozen work remains by immutable version ID, including after archive; H1 must use active name resolution for new selections.

Mutations atomically persist principal/operation/key fingerprints and original results without expiry. Same key and canonical request returns the original result even after restart or revision advance; changed valid input conflicts. A new key attempting an existing version ID gets `version_conflict`. Revision-zero registration creates a voice; later registrations/alias/archive changes require the current revision. Alias set order is canonicalized. No silent overwrite or default/model selection. Returned JSON is detached from storage.

The schema evolves additively: optional asset principal and version reference hashes, domain register/alias requests and register result, plus stable alias/version/archive conflict codes. Existing operation routes remain unchanged and explicitly describe fixture status. `voice_save` keeps candidate selection semantics; H0 registration is not advertised as its HTTP implementation. The detailed legacy-control inventory is retained, with a finite command bridge and an exactly-once current operation coverage map appended. Deferred editor scaffolding is explicit; script/take/assembly/export and all existing audio controls remain required.

## Exact H1/H2 seam

All calls run on the Core owner, never directly from an untrusted HTTP payload or a separate DB connection:

- `register(principal, key, version, *, expected_revision)` returns `{voice, version}`. Input is the existing full `voice_version` contract (optional supplied reference hashes must match verified assets); output freezes verified hashes. H1 performs candidate preparation/selection then calls this for `voice_save`.
- `get(principal, voice_id)` returns the existing `{id, aliases, version_ids, archived, revision}` voice, including archived metadata.
- `version(principal, version_id)` returns detached immutable version metadata, including archived references. `versions(principal, voice_id, *, after=0, limit=20)` returns ordered `{cursor, version}` pages.
- `list(principal, *, after=0, limit=20, include_archived=False)` returns `{cursor, voice}` pages; limits are 1–20 and cursors are nonnegative integers. Voice records retain the existing version_ids contract; page count is bounded, not a fixed serialized-byte budget.
- `resolve(principal, name)` resolves canonical ID or alias only for an active voice.
- `aliases(principal, key, voice_id, aliases, *, expected_revision)` atomically replaces aliases (at most 32), increments revision and returns voice metadata.
- `archive(principal, key, voice_id, *, expected_revision)` archives without deletion, increments revision and returns voice metadata.

H1 trusted ingest owns validated publication of managed assets and principal metadata; callers cannot register ownership through this store. H1 production must retain frozen version IDs/hashes/settings in plans. H2 authenticates/revokes credentials, derives principal, enforces scopes and translates schema/domain/storage errors via explicit adapters. H2 owns missing alias/version-list/import/export, segment/export, job-list/wait/events and profile-detail transport contracts; the finite map identifies these as planned requirements, not callable routes. H0 adds no dispatcher method or product activation.

## Executed verification and limits

Inspected `tests/conftest.py` before execution: its autouse service fixture imports `_maintenance_wait_guard`; `tests/service/conftest.py` also has optional HTTP process fixtures. Therefore **no pytest collection/execution**, full suite, existing core/persistence test module, snapshot/native-fault path or maintenance guard import was run. Used existing `/private/tmp/pab-0c-venv/bin/python` (Python 3.14 environment, jsonschema 4.26.0), with stdlib unittest discovery restricted to the new file:

`/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_voice_store.py -v`

Final result: **13 tests PASS in 0.502 s**. Actual functions, Core SQLite/WAL and media verification run only against exclusively owned temporary `pab-h0-voice-*` directories. Normal test cleanup removes only those directories. No native fatal or unexpected signal occurred. Initial run was 9/10 PASS: the idempotency test used an intrinsically invalid preset mutation and received `reference_mismatch`; corrected to a valid settings/hash mutation, then expanded coverage. Intermediate 13-test run passed; final run added backward-compatible contract checks and passed. This was ordinary test correction, not a native-failure retry.

Coverage: restart retains multiple versions/aliases/exact settings; idempotency survives restart and precedes stale revision handling; duplicate-version conflict; sequential competing stale revisions through the existing serialized owner; alias collision rollback/canonical-name reservation; archive retains immutable references/names; SQL update/delete refusal and detached returns; managed ownership/hash persistence and tamper rejection; absent/unmanaged/foreign/ownerless references; existing publication ownership inheritance; symlink/missing managed files; additive migration preserving an old revision row, authorizer-denied schema creation rollback, repeated opens and unchanged user_version; partial-schema refusal and inspect-only fence; bounded list pages/principal filtering and blocked heavy imports; schema validity, all existing operation request fixtures and error-code/status consistency. `git diff --check` PASS.

The revision test is serialized competing requests, not a parallel cross-process race campaign. Migration rollback is an isolated SQLite authorizer denial, not a power-loss/native-fault test. Existing core/persistence regression modules remain unrun under the guard restriction. No fixtures were regenerated and no full contract test suite was run. Installed upgrade/rollback, hardware speech, acoustic validation, performance/listening and agent-client journeys remain unqualified.

## Parent handoff

Next: independent Sol adversarial review of the source commit and report before parent adoption. Material limitations for review: legacy ownership scan cap; existing 1 MiB Core asset verification limit; voice version_ids can grow within a page; trusted asset principal assignment remains H1; archive intentionally preserves rather than removes dependencies; actual profile/bundle provisioning and worker use remain H1/H4. No self-adoption or release GO.

H1/H2 implement the managed-ingest/candidate/production and authenticated adapter journeys against this seam. H3 installed durability/recovery acceptance remains required. H4/H5 must actually qualify constrained CPU, NVIDIA CUDA and packaged Apple Silicon native Metal, including real speech and listening; GPU/Apple qualification is mandatory, not optional. All inherited unresolved production safety gates and historical 81/native evidence limits remain parent-owned and unchanged.
