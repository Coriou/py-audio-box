# H0 schema refusal fix — r161 Sol implementation

2026-10-24. **R160-1 implemented; focused isolated unittest PASS. Parent fresh independent Sol review remains required before adoption. No self-GO, installed durability or release claim.**

## Authority, pins and scope

Own fresh worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-91c6d46e`, branch `t3/voice-schema-refusal`, was clean at entry `7ae9df6fc43314a1a5351ab4db0f717cfc051b9c`. Under explicit own-branch reset authority, reset only this branch/worktree to candidate docs **`c32744a538f88b76031126aeeb98a672b3c542b3`**, containing source **`a6702296dcfd0a54612dafa123007edbe6286eb3`**. That is the implementation base. Original `/t3-e1b44469` was read only and unchanged by this task.

Read full r159 and root r160, full canonical new headless IMPLEMENTATION-PLAN and r157 replan; inspected current initializer, Core constructor/transaction boundary, complete focused test source and both conftests. Source fix commit: **`cb9d8d4b40776d65865b3e69cd1d1378cad534a8`**. This report is committed separately afterward. Only this report is copied into root `/Users/ben/Projects/py-audio-box`; parent owns adoption and ledger changes.

Changed paths are exactly `service/voice_store.py`, `tests/service/test_voice_store.py`, and this report. No Core/persistence/recovery/guard/contracts/worker/API/legacy edits, delegation, push, Docker, model, network, lane or operational-resource action. Historical proofs, STT and HELD resources were not touched. Cleanup was limited to normal test-owned temporary directories.

## Fix and supported schema boundary

A single declaration of the original five table definitions and two immutable-version trigger definitions now drives both installation and validation. A temporary in-memory SQLite reference expands that DDL into the expected table/index/trigger object set, including SQLite's implicit PK/unique indexes. The actual owned object set and SQL token identities must match. This covers declared column order/types/nullability, PK/unique/FK constraints, any CHECK clauses, table options, index definitions and trigger bodies across all owned tables. Extra indexes/triggers on owned tables also refuse. Unrelated Core objects are excluded.

This is conservative supported-DDL identity, not a general SQL semantic-equivalence parser or whitespace deletion. Tokens retain boundaries, punctuation and complete quoted strings/identifiers; only whitespace between tokens and unquoted keyword/identifier case normalize. Alternate equivalent constraint spellings, quoting, comments or extra indexes are not promised compatible. Literal changes cannot normalize away. H0 declares no CHECK constraints; added checks are rejected, with no new domain constraint introduced.

Presence, full schema identity and the exact format row `(1,)` are checked within the same existing `Core.transaction` BEGIN IMMEDIATE boundary. Fresh creation, format insertion and post-creation validation commit atomically. Existing layouts receive no CREATE/repair, index replacement, table replacement or row rewrite. Missing triggers now refuse rather than being silently recreated. Partial schemas and unsupported format rows still refuse. Pre-P1 returns without schema reads/writes; core user_version remains 1. Store exposure remains after initialization in the unchanged Core constructor.

No registration/version/principal/idempotency/alias/archive method changed. Immutability and existing domain semantics remain intact.

## Actual focused verification

Authorized command only, executed in this worktree:

`/private/tmp/pab-0c-venv/bin/python -m unittest discover -s tests/service -p test_voice_store.py -v`

Before-fix evidence: temporarily restored only the original candidate `voice_store.py`, ran the new tests against it, then restored the fix. **16 tests, exit 1, 0.544 s: four assertion failures and one test-fixture error.** The R160 counterexample demonstrably registered a voice with NULL seq and returned an empty list; original initialize then failed to raise. Both same-named no-op triggers and an extra unique index likewise failed the refusal assertions. The matrix fixture initially assumed one FK whitespace layout and errored during construction; that did not invalidate the independently executed counterexample/trigger failures.

First fixed run: **16 tests, exit 1, 0.645 s**, with only a fixture snapshot error: Python SQLite `iterdump` invokes foreign_key_check, which itself refuses the deliberately missing parent uniqueness. Replaced that matrix snapshot with explicit schema/row snapshots, preserving its assertions rather than weakening the required refusal. Added token/literal and missing-trigger coverage.

Final run: **17 tests PASS, exit 0, 0.655 s**. No native fatal, unexpected signal or native retry occurred. An initial edit command used unavailable `python`; it exited 127 without executing code, then edits used `python3`. Actual tests always used the authorized interpreter above.

New coverage includes 19 malformed table layouts: wrong type and removed NOT NULL on each of five owned tables; missing integer PK/unique on voices and versions; missing composite PK on names/requests; missing FK on names/versions; added CHECK. Each refuses without a transaction left open and preserves the exact prior schema, all retained catalogue rows and a prior Core revision row. Parent-unique negatives retain parent/request/format/core rows and omit child inserts that SQLite cannot execute on that invalid layout. The matrix restores only its test fixture afterward and successfully reopens a valid schema.

The explicit NULL-seq counterexample checks the invisibility behavior, refusal with unchanged data and constructor refusal on reopen. Both no-op immutable triggers refuse with unchanged dumps; extra unique index, changed RAISE literal and missing trigger refuse. Whitespace/keyword-case variation accepts. Existing 13 tests still cover fresh migration rollback under authorizer denial, repeated valid opens/user_version, partial schema refusal, inspect-only fence, immutable versions, assets, idempotency, aliases/archive and principal isolation. `git diff --check` PASS; source commit contains only the two authorized source/test files.

All DB/media activity used actual Core/SQLite in exclusively owned `pab-h0-voice-*` temporary roots plus the memory-only reference connection. No pytest collection/conftest import, maintenance guard import, full suite, native fault campaign or installed upgrade test. The old-root test still flips the in-memory P1 flag; it is not historical-root acceptance. Transaction rollback testing is not power-loss durability. No installed durability, concurrency campaign, hardware/audio or production evidence is claimed.

## Parent handoff and next domain scope

Fresh independent Sol review of source `cb9d8d4b40776d65865b3e69cd1d1378cad534a8` and this report is the next gate, before parent adoption. The candidate remains unadopted by this implementer.

R160's H1/H2 seams remain deliberately unchanged: existing Core 1 MiB reference verification cap; metadata/token/history growth and version paging existence check; historical ownership fallback cap/indexed compatibility; trusted ingest ownership and prepared-candidate selection; bounded authenticated transport/error adapters. These require the next domain/contract/adapter scope, not expansion of this schema fix. H3 installed upgrade/restore/fault acceptance, H4/H5 CPU/CUDA/native Metal, speech/listening and integrated headless journeys remain required. All historical native UNKNOWN and HELD-resource gates remain parent-owned.
