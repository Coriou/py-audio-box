# H1A managed LOCAL ingest — r163 Sol implementation

2026-10-24. **Implemented source and focused isolated unittest PASS; independent Sol review and parent adoption pending. NO-GO for release/activation or installed durability claims.** This is bounded H1A, not the complete H1 production journey.

## Pins and scope

Fresh supplied worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-f62009a3` began clean at `e72f08278bc72871a4239a3f1d7d0874734edd7d`, verified equal to root `/Users/ben/Projects/py-audio-box`. This includes reviewed H0 adoption (`aa3dfb5`). Initial branch `t3/f62009a3`, current own branch `t3/managed-local-ingest`. Source commit **`b1dcda182716e48fcb0750d8d772dc813e77647c`**. Root advanced concurrently to `22154af34a938d050a77c637f183053e0c6a526b` at precommit inspection; no reset/rebase or copying parallel source. Report committed separately; only this report copied to root.

Read canonical IMPLEMENTATION-PLAN, r157/r158/r159/r160/r162, Core initialization/transaction/asset verification, existing persistence publication/write/fsync/verification/session behavior, voice store and focused tests, both conftests and relevant contract definitions. Source changed exactly five allowlisted paths: new `service/ingest.py`, `service/voice_store.py`, `service/core.py` only asset verification, new `tests/service/test_ingest.py`, `tests/service/test_voice_store.py`. Sixth path is this report. No API/domain/catalogue/clients/auth/contracts/persistence/dispatcher/worker/recovery/guard/legacy edits, delegation, Docker, model, network, remote URL, lane, deployment or historical/HELD path operation. No new runner or activation-fence removal.

## Exact interfaces for H2

Calls must execute on the Core owning thread through H2's explicit owning-Dispatcher allowlist. Existing `Core.voices` remains unchanged as an integration object; this lane adds no Dispatcher methods or public routes.

`service.ingest.ingest_local(core, principal, source) -> asset metadata` takes immutable `bytes` or a trusted blocking binary stream implementing `read(size) -> bytes`. It leaves input streams open. It accepts no source path, filename, URL, caller asset ID, ownership metadata or provenance claims. Principal must already be derived/authenticated by the owning transport; this function validates ID syntax, not credentials. H2 must wrap client bodies with its independent transport byte/time limits; an arbitrary trusted blocking stream can block in read, so never run it as an unbounded network reader on the owner. No upload idempotency is promised.

The result uses the **existing asset contract**, with optional `principal_id`. An opaque random `upload-<uuid>` logical ID identifies the managed file and provenance source ID. SHA256 covers every original byte and is also provenance source_sha256; kind is `upload`, missing_fields is empty. No transformations, extension trust, normalization, model inference or microphone capture occur. A client may later upload captured bytes; no microphone-origin claim is inferred here. Identical content always receives distinct IDs, even for one principal. There is no foreign-content adoption/deduplication or ownership scan.

`Core.asset(id)` remains the existing metadata/read-verification interface; `Core.voices.register` verifies and freezes these asset hashes as before. The test includes an actual ingested reference larger than 1 MiB registered and read after restart.

New compact methods on `Core.voices`:

- `summary(principal, voice_id)` returns `{id,revision,archived,version_count}` without materializing aliases, version IDs or authored records.
- `summaries(principal, *, after=0, limit=20, include_archived=False)` returns ordered `{cursor,voice: summary}` pages.
- `version_summaries(principal, voice_id, *, after=0, limit=20)` returns `{cursor,id,voice_id}` pages without decoding version records.
- Existing `versions(...)` returns the same full-version page shape but now checks existence with SELECT 1 instead of calling full-history get.

Cursors are nonnegative integers, page sizes 1–20; booleans are rejected as integers. Principal filters and archived direct-read behavior remain. Existing get/list/resolve/register/aliases/archive return shapes and semantics remain; full get/list still expose legacy complete history. New summaries do not call get. Counts/query work may still scale with existing rows; no index/migration/backfill overlaps H0/H2 here. No claim of constant-time database work.

## Explicit finite media and metadata budgets

Supported local ingest: WAV integer PCM16/24/32, mono/stereo, 8,000–48,000 Hz inclusive, nonempty and at most **60 seconds**, encoded file at most **24 MiB (25,165,824 bytes)**. Worst supported payload is 60 * 48,000 * 2 * 4 = 23,040,000 bytes, leaving finite header/container overhead. Compressed WAV, float/double WAV (including NaN/Inf), PCM8, other formats, unsupported rates/channels, empty and truncated sample payloads refuse. Integer PCM samples are intrinsically finite; no floating-point decoder or ML dependency is imported. wave parses and reads bounded chunks; it does not allocate declared untrusted payload lengths. This is media structure/length validation, not acoustic quality, loudness, consent or clone suitability qualification.

Input reads are requested at at most 64 KiB and include a one-byte over-limit probe. A stream returning too many bytes or non-bytes refuses. Decode reads at most 8,192 frames at a time. Before staging, fstatvfs requires 24 MiB plus 1 MiB reserve available; this deliberately conservative admission can refuse a small file on low disk. Actual I/O errors remain possible and propagate. No quota reservation or concurrent-process disk guarantee is claimed.

Core preserves its original **1 MiB** cap for legacy/other assets. Only existing-contract metadata with a direct principal, upload provenance, WAV PCM16/24/32, supported rate and positive <=60-second frames gets the 24 MiB verifier budget. DB metadata is trusted managed state, not accepted transport input. SHA verification streams at <=64 KiB, checks a one-byte overflow, retains regular/current-uid/single-link requirements, checks actual length and SHA, and compares inode/size/mtime/ctime before/after hashing. Directory and leaf opens use no-follow directory descriptors. Artifact calls still invoke this verification; publication_work and synthetic artifact limits are unchanged.

New registrations additionally require canonical input version metadata <=**65,536 UTF-8 bytes** and every string (including authored text and nested configuration/provenance/settings) <=**16,384 UTF-8 bytes**, after existing contract validation. These are domain limits, not silent text truncation. Exact authored content is retained. Limits apply inside new-registration application after successful idempotency lookup: historical records/read paths and valid prior replay results are not retroactively rejected or rewritten. A new oversized registration of legacy content refuses resource_limit; export/import/transport compatibility must disclose this policy. Stored derived reference hashes add bounded-by-input reference entries beyond the input metadata budget; no claim that every old full read is <=64 KiB. Historical idempotency/history growth remains unbounded over time; compact page count alone does not make full legacy responses bounded.

## Publication, rollback and orphan policy

Reuse Core's owner/thread/P1/transaction contract and SQLite FULL WAL durability. Ingest refuses pre-P1 or nesting in an existing transaction before files. Private staging/assets directories open O_DIRECTORY|O_NOFOLLOW and must belong to current uid with no group/other permissions. Staging uses random O_EXCL|O_NOFOLLOW 0600 files. Partial writes are looped, flushed, media-validated and fsynced. Atomic hard-link publication refuses destination overwrite; staging link is removed before metadata insertion, restoring the existing one-link invariant. Both directory namespaces are fsynced before inserting metadata through Core.transaction. No success is returned before COMMIT succeeds.

Known errors before COMMIT remove only this call's random staging/final names and fsync the affected directories. No directory-wide cleanup or adoption occurs. Insert denial rolls back and removes the final file. Once COMMIT is attempted, an error can be ambiguous: retain the complete published file even if a tested rollback makes it an orphan. A crash before DB commit can likewise leave a complete orphan (or an earlier temporary/link state). No startup cleanup, orphan adoption, diagnostic migration or recovery algorithm is added. Without an assets DB row, Core.asset returns not_found; a later same-content call gets a new ID. This conservative retention avoids deleting bytes potentially referenced by a committed row. H3 owns installed crash/upgrade/restore/orphan reconciliation qualification. Ordinary fsync sequencing is not power-loss qualification.

Errors: ValueError invalid_request for principal/source/stream shape or nested call; resource_limit for bytes/duration/disk admission/metadata budgets; invalid_audio for unsupported/invalid decoded structure; foreign_media for namespace/inode/size verification; missing_media for hash/length/change mismatch; not_found for absent metadata. Existing pre_p1_inspect_only, forbidden and contract validation failures remain. Native OSError (including missing/symlink, ENOSPC/EIO), SQLite errors and ambiguous commit failures propagate; H2 must translate them with bounded messages and no private paths. Cleanup failure also propagates and may retain files; no blanket success/cleanup guarantee or live retry is added.

## Executed evidence

Read both conftests first; used unittest only, without pytest collection/maintenance fixture. Actual Core SQLite/WAL and real generated PCM WAVs ran exclusively inside owned temporary `pab-h1-ingest-*` and `pab-h0-voice-*` roots with normal test cleanup.

`/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_ingest.py -v`

Final **13 PASS, exit 0, 0.413 s**. Earlier initial 10-test run also passed; added post-publication sync, commit refusal and PCM24/32/short-read cases and reran. No failing assertion, unexpected fatal or native signal.

`/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_voice_store.py -v`

Final **20 PASS, exit 0, 0.744 s**. Earlier 19-test pass preceded addition of actual >1 MiB ingest→registration→restart coverage. New compact/history and text-budget tests pass alongside retained H0 migration/schema/refusal/ownership/idempotency/archive tests. `git diff --check` passed.

Coverage includes original bytes/hash/provenance, private mode/link count, restart, duplicate cross-principal isolation, maximum-duration stereo PCM32 verification and tamper, invalid/truncated/float-NaN media, byte/duration/channel/rate/width limits, path-like input refusal, oversized/mid-failing/short reads, low-disk injection, atomic publish I/O failure, staging fsync failure, post-publish directory fsync failure, DB insert denial cleanup, COMMIT denial conservative orphan retention/reopen/nonadoption, symlink directory/leaf refusal, old asset cap, inspect-only and nested-transaction fences, compact no-full-get paging and exact text limits. Old missing-file/tamper/publication-owner compatibility remains covered by H0 tests.

No full suite, affected existing Core/persistence regression module, pytest guards, native faults, power interruption, installed process/upgrade/restore, production, audio listening or hardware inference ran. Low disk is injected fstatvfs, not a filled disk; SQLite denial is not a native storage fault. No concurrency/race/crash campaign claim.

## Remaining owners and next gate

Independent Sol review next; no self-adoption. Parent owns any shared contract additions for compact summaries, version-list transport or upload envelopes/error translation. Existing contracts were not edited. H2 may continue existing Core.voices adapters and explicitly add these interfaces after coordination; never expose arbitrary getattr or accept asset metadata/principal from the client.

Remote acquisition remains unavailable (SSRF/redirect/DNS/address/time/credential boundaries needed). Client microphone capture, segmentation/clean-reference processing, prepared-candidate selection and voice_save orchestration, engine jobs/frozen plans/speech/stitch/export, provisioning, authenticated API/CLI/MCP parity and indexed legacy ownership/backfill remain subsequent work. No new engine job behavior is claimed. H3 installed durability/fault acceptance and later actual CPU, CUDA and packaged Apple Silicon native Metal qualification/listening remain mandatory. UI stays deferred. Historical native UNKNOWN and HELD-resource gates are unchanged.
