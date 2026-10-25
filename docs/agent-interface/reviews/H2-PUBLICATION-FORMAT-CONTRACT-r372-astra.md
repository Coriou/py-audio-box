# r372 — finite publication-format contract for r370 F2

2026-10-25. **DESIGN GO WITH THE REVISIONS BELOW; F2 implementation/adoption remains SOURCE NO-GO.** This approves a finite source implementation choice, not execution of a migration. Release NO-GO; legacy CLOSED.

## Evidence and authority

Source-readonly candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-312bb29e` verified at `381064e9772871e180a0c178b064b4972a6d27cf`, clean on initial inspection; base `96a1b249c1b6a505e1d32c500adca10355115458` verified ancestor. Root checkout verified at that base. Read full root `docs/agent-interface/reviews/H2-AUDIO-EXPORT-GATE-r370-astra.md` and full updated r371 report in `/Users/ben/.txcode/worktrees/py-audio-box/t3-b413d79c`. Inspected relevant Core, persistence, admission, dispatcher, reconciliation, recovery, native activation/runtime, persistent initialization, installation policy and dropped registration writer. No applicable ancestor AGENTS found. Memory supplied only the canonical numeric-identity caution; conclusions below use current source.

r371's attributed inert reproduction confirms 1,152,044 output bytes -> PublicationQuarantine in receipt phase, running job, executing reservation, persistence fault/quarantine. It is not rerun here. Its prepared tests/report are uncommitted work, not a production correction. This report is the sole write, in the ROOT checkout, mode 0600. No application imports, tests, code edits, commits, merges, delegation, actual migration, Docker, network/listener, application child, FFmpeg, ML/hardware, HELD/STT or cleanup.

## Approved choice and deliberate limits

Use two exact named publication store formats. Do not modify the v1 reference in place. There is no generic migration registry, new store, recovery action or caller-selected SQL/cap.

| Named format | PRAGMA user_version | Sole p1_format row | publication_work CHECK |
| --- | --- | --- | --- |
| publication-v1 | 1 | integer 1, text synthetic-core | byte_length BETWEEN 1 AND 1048576 |
| publication-v2 | 2 | integer 2, text synthetic-core | byte_length BETWEEN 1 AND 37748736 |

**Revise the proposed user_version=1/P1=2 combination: it is NOT supported.** The Core-level version identifies an incompatible SQL store even though only the publication table changes. Enrollment revision, resource_protocol `(1,2)`, voice catalogue revision, manifest versions and native/runtime protocol revisions do not change. All cross-pairs, unknown markers, duplicate rows, partial private formats, widened-under-v1 and narrowed-under-v2 SQL refuse. Do not use Python equality to accept float/bool versions in detached documents.

New ordinary unenrolled local Core roots use v2. Existing exact v1 roots open as v1 without rewriting and retain their old reads/publication recovery and speech/prepare semantics. Pre-P1 roots retain their existing query-only treatment; they are not migration candidates. No constructor, Dispatcher startup, installer completion/resume, reconciliation or retry performs a migration.

**Revise blanket 'all fresh roots v2': installed-family and native activation roots stay v1 in this lot.** Both existing and fresh installed/native formats and enrollment are preserved. Existing trusted initializer paths (`persistent_init` Core calls at 175/230) must explicitly select v1 only for creation; native Dispatcher construction must select v1 from the already checked NativeActivation path. Use a closed internal creation choice, not an HTTP/CLI/MCP option or a value inferred from untrusted rows. The ordinary default is v2. A creation choice cannot restamp or reinterpret an existing DB. Installed/native startup must explicitly refuse a v2 root before any writer/activation is invoked in this lot. Their export capability remains unavailable.

Their shared pure validators should recognize exact v1/v2 publication models when used by generic detached consumers, but role-specific installed/native activation continues to require v1. This is an intentional gate, not a claim that installed/native v2 is qualified or enrollable. Do not alter the native 1 MiB file/accounting rules, Linux role policies or registration writer. Broadening these writers is a separate parent-owned product step.

## Admission and publication contract

Simplest finite rule: **every NEW audio_export requires v2**, including small exports. This avoids output-size-dependent storage capability and a second export compatibility mode. Existing idempotency lookup remains FIRST: same key/input returns its retained job without converter readiness, format migration or new reservation; conflicting key retains existing conflict behavior. A retry creating a new job is new admission and requires v2.

Enforce the v1 refusal at the trusted Core submission boundary (`submit_execution:430-475`), immediately after duplicate lookup and before native preallocation, resource preflight/observation, transaction/job/event creation, dispatch settings or reservation. The Dispatcher/public route must expose a stable normal `publication_format_migration_required` refusal, not PublicationQuarantine, storage fault or resource fault. Map it explicitly through the existing public error contract (409 conflict, safe code/message); CLI/MCP consume HTTP normally. No migration command appears in public voice APIs. Capability/readiness must truthfully report the v1 restriction. Existing installed unavailable-converter gates retain precedence where applicable. Recheck the version at admission using trusted opened-store format; no user-provided 'v2' flag.

A v1 job already admitted by the defective candidate is not repaired by this feature. In particular r371's running/executing failure is NOT an eligible clean migration prestate. Retain it; do not mark it terminal, release capacity, synthesize ACK or replay publication to make migration succeed.

For v2, only an authenticated audio_export producer receives the 37,748,736-byte ceiling, further bounded by the frozen export descriptor/profile/media rules. Speech and each prepare file remain <=1,048,576 bytes. A public artifact's provenance tag, a caller cap, SQL marker alone, or the presence of an audio_export field is insufficient authority. Bind operation, exact frozen plan/execution, retained job/attempt, manifest, publication, intent and public artifact as currently required by F1; use canonical representation comparisons, not numeric-tolerant dict equality. Literal single-output and prepared-reference-adoption-v1 formats remain unchanged.

Thread trusted format and corroborated operation through `publication_intent`, `publication_record`, `public_artifact`, Core `_publish`, `_publication_barrier`, and every verify_file call (links=2 destination and alias, links=1 final). The barrier currently calls the intent parser without job/context at 677: load them before selecting the v2 cap. Detached record readers must derive format from their already validated source model, never default missing format to v2. Preserve strict integer lengths, bounded reads, no-follow/owner/inode/link/hash/fsync checks and the intent-before-files-before-public-commit order. Streaming the hash instead of accumulating the full file is appropriate; no unbounded reads. SQL's wider CHECK is storage capacity, not permission to enlarge speech/prepare.

## The sole migration entrypoint approved for source implementation

A private local operator CLI, for example `python -m service.publication_format migrate-v1-to-v2 --root ABSOLUTE_CORE_ROOT --expect-v1`, with a single fixed transition. The spelling may follow repository conventions; the semantic surface is fixed: explicit existing root and explicit v1 expectation, no target version selector, force, SQL, repair, ownership override, arbitrary policy or public API/MCP exposure. Running this CLI against actual roots is NOT authorized by this design commission.

Supported writer scope is deliberately **clean, same-UID, unenrolled local Core only**. Refuse Linux enrollment tables/receipts, installed activation, native activation/renewal state, any B1/B2 recovery/enrollment tables or maintenance artifacts, and any foreign/ambiguous namespace. No root privilege exception. Do not invoke `recovery.offline(write=True)` for an unenrolled root: it requires owned-synthetic enrollment. Do not reopen legacy owned-synthetic migration/recovery authority either. All enrolled legacy, split installed and native roots return scope-unsupported unchanged. This bounded migration does NOT promise conversion of every existing deployment.

There is a further explicit quiescence limit: **no retained dispatches or dispatch commands** in an eligible root. If those tables exist they must be exact and empty. Refuse even apparently terminal/acked dispatch history; SQL alone does not prove external journal quiescence. This avoids inventing a journal authority for unenrolled local roots. Such roots remain readable v1 and exports remain migration-required; parent may later commission a stopped-journal migration adapter. Do not delete history to qualify.

Eligible retained direct-Core jobs must be terminal with all attempts reaped; no queued/running/cancel_requested jobs, active owners, pending publication, adoption aliases, executing/queued reservations, resource faults, unhealthy monitor/integrity, or unknown names. Completed retained publication_work rows are allowed and must remain intact. A valid but incomplete publication is a busy prestate, not corrupt data: refuse without touching its intent/files. Empty roots with no jobs and completed direct-Core speech/prepare history are the finite acceptance cases. Resource tables, if present, may retain only fully corroborated terminal reservations; never release or recompute them.

### Ownership and exclusion

1. Take the existing process-wide `_core_maintenance_lease` before opening the owner DB. It excludes live and uncertain-cleanup Core ownership. Do not hold it and then call `_reserve_core`/Core constructor (nested bookkeeping can deadlock). Use maintenance-owned cleanup while holding the lease until SQLite and locks are conclusively closed.
2. Validate existing private root/ancestors, DB, lock and sidecars with no-follow ownership/mode/link/inode checks. Require an existing control.lock; take its existing nonblocking exclusive flock. No mkdir, chmod, chown, lock replacement or missing-DB creation. Revalidate directory/DB/lock identities at boundaries. Native startup/renewal markers refuse; no bypass through direct SQL.
3. For any topology with a receipts lane, BOTH control then lane locks and validated journal/maintenance state would be required. This lot refuses those topologies before mutation; absence of dispatch/journal authority is not 'journal quiescent'. No change to `_locked_roots` ordering, `install_policy.locked_file`, `offline(policy!=None,write=True)` refusal, or install_child registration_writer's explicit No DDL/Core boundary.
4. Inspect a bounded source-preserving private SQLite copy including WAL before opening the original mode=rw, following existing recovery private-copy principles. Never open retained SQL text as executable code. Validate exact predecessor schema/rows/documents and quiescence. Then open original with trusted_schema=OFF, foreign_keys=ON, synchronous=FULL, timeout=0, existing WAL mode and finite SQL deadline. Revalidate under BEGIN IMMEDIATE before changing anything. No WAL truncation, checkpoint prerequisite, VACUUM, backup replacement or journal-mode conversion.

### Exact predecessor and transaction

Use closed canonical module DDL, not columns-only inspection. The finite allowed local shapes are the existing Core+P1+catalogue tables, optionally extended by the COMPLETE existing inventory and/or resource/dispatch initializer groups. A partial group, extra table/view/index/trigger, new FK referencing publication_work, missing constraint, changed SQL token or unrecognized namespace refuses. Validate full sqlite_master identity including implicit indexes and existing immutable catalogue triggers using the established `_sql_identity`/model comparison technique. Generate models from trusted constants only. Bound inspection to 2048 rows and 16 MiB scalar data, with a finite deadline; exhaustion refuses. SQL affinity must not launder malformed detached values. Require integrity/foreign-key checks and canonical retained intent/publication/artifact joins before any DDL.

Within ONE explicit transaction, after revalidation:

- Rename old publication_work to a fixed private transition name, guaranteed absent in the exact predecessor. No accepted incoming FKs/views/triggers refer to it.
- CREATE the new publication_work using the exact canonical v2 SQL below. INSERT original explicit rowid and all six columns from the old table, ordered by rowid. Preserve SQL types, raw binding strings, destinations, temporary names, hashes and lengths; never deserialize/re-encode retained rows.
- Drop the renamed old table. Its old autoindexes disappear; the canonical CREATE supplies the exact original names/constraints for the new table. Do not create ad hoc indexes/triggers or rebuild other tables.
- Change the sole p1_format revision from integer 1 to integer 2 with exact precondition; set PRAGMA user_version=2 in the SAME transaction.
- Compare the entire post-schema to the exact expected v2 model. Verify rowid/typed-row equality of publication_work, all other table rows/SQL, sqlite_sequence, FK integrity, and retained documents. Only the named table DDL/its regenerated indexes and the two version values may differ logically. COMMIT once, then sync required directory namespaces and verify the committed pair while exclusion remains held.

Canonical v2 declaration (v1 reference retained separately):

```sql
CREATE TABLE publication_work(
          job_id TEXT PRIMARY KEY REFERENCES publications(job_id),
          binding TEXT NOT NULL, destination TEXT NOT NULL UNIQUE,
          temporary TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,
          byte_length INTEGER NOT NULL CHECK(byte_length BETWEEN 1 AND 37748736))
```

Do not use executescript across the transaction: its implicit commit behavior can split the change. Do not disable foreign keys or use writable_schema. Renaming/recreating uses trusted literal statements, not the retained SQL. Compare normalized token identity after final recreation; never bless an arbitrary equivalent-looking CHECK. Fresh v2 creation must also commit publication DDL and the coherent version pair together; a crash during broader fresh initialization must remain incomplete/quarantined, never advertised usable.

## Finite exits and retention

| Condition | Exit/result | Required retained state |
| --- | --- | --- |
| Exact eligible v1, validation and commit complete | 0 / migrated | coherent v2, original logical rows and files preserved |
| Exact eligible already-v2 | 0 / already-v2 | validate then no mutation; supports operator reinspection after lost reply |
| Wrong scope, active/uncertain owner, dispatch history, pending intent, journal/maintenance, budget exhausted | 2 / refused with specific reason | no schema/data mutation, no cleanup/ACK/reap/reservation changes |
| Mixed/unknown format, malformed schema/row/identity | 2 / invalid-prestate | retain everything; no repair, migration or downgrade |
| Error before COMMIT with confirmed rollback | 3 / failed-rolled-back | coherent v1 logical store; no success claim |
| COMMIT may have happened, fsync/verification/close uncertain | 4 / outcome-unknown, reopen required | retain resources/sidecars and ownership obligation if close uncertain; no retry/rollback-after-commit/downgrade |

SQLite may change journal/sidecar/page bytes even for a rolled-back write. 'Preserved' means original logical state and user media/intent evidence, not a false promise of byte-identical DB files. Interruption during a transaction is handled by SQLite recovery to a coherent v1 or v2 state. A later explicit inspection may classify outcome; do not auto-run the transition twice or report success merely because an exception is absent. No new durable cross-store journal is needed because this migration changes only one SQLite store and writes no receipt/media state.

## Exact consumer obligations and compatibility traps

- `core.py:275-295`, `persistence.py:13-41`: accept only named coherent live pairs with exact publication DDL; preserve pre-P1 behavior. Do not infer version from the CHECK alone.
- `persistence.py:259-372`, `core.py:637-710,765-779`, `reconciliation.py:553-557,647-651,679-685`: shared authenticated cap selection and F1 canonical joins. Reconciliation `_collect:118-145` needs publication format/schema validation in addition to its existing exact catalogue schema.
- `recovery.py:781,2874-2945,2977-3002`: keep `_INSTALLED_V1_TABLE_DDL` immutable; add a separate exact v2 model differing only in publication_work. Detached source envelopes currently carry schema/data but NOT user_version (`_validate_source:1340`). Do not invent a historical user_version claim or add an optional envelope key silently: select the detached named model from exact marker+DDL; every actual SQLite snapshot/open must check user_version coherency separately. Role-specific installed acceptance stays gated v1.
- `recovery.py:1572`: B2 archive snapshots have their OWN historical version. Validate them against their own marker+DDL, never the current root's version; retain existing v1 archives byte-for-byte. Migration refuses B2 roots, so no B2 writer/certificate/abandon change is authorized.
- `native_activation.py:124-152`: two explicit v1 model constructions and v1 marker gate remain correct because native roots stay v1. If shared helper signatures change, pass v1 explicitly. Do not claim these gates accept v2 or enlarge native reads at 277 / native_runtime's 1 MiB limits.
- `persistent_init.py:175,230,473-502`: keep installed fresh v1 selection and exact v1 fresh validation aligned. Installer resume is never migration. `host_install`, `daemon_helper`, `package_install` must not route a v2 installed root around validation; reject it before mutation. `install_child.py:98-167` remains registration-only.
- `dispatcher.py:319-330,348`, `core.submit_execution`, production retry/public error mapping: enforce new-export v2 gate without interfering with replay, speech/prepare or installed unavailable-converter reporting.

**Additional bounded-size dependency:** reconciliation currently charges media, including each hardlink name, against 16 MiB (`reconciliation.py:75-97,169-178`), with a 64 MiB constructor maximum. A >1 MiB fixture passes that budget but does not prove the whole export envelope. Approve a v2 LOCAL export inventory byte budget of **160 MiB**, selected internally from the validated v2 format, retaining 2048 entries and the existing finite time bound. Arithmetic for one maximum attempt: 24 MiB source + 3 x 36 MiB (stage/final/adoption alias) + 16 MiB metadata allowance = 148 MiB. Keep the SQL/JSON scalar inspection allowance at 16 MiB separately; do not raise it with the media budget. v1/installed/native keep their existing budgets. Charge all names, no hardlink discount; over-budget accumulated history and time exhaustion remain fail-closed. This is a bounded single-attempt capacity, not unlimited retained history or a latency qualification. Include >16 MiB scalar/accounting boundary coverage without large media allocation; the finite live inert journey stays ~1.1 MiB. No general cap increase to recovery, worker speech, native or installation limits.

### Old-binary refusal: verified source facts, not a universal claim

At candidate 381064e9, old Core explicitly rejects user_version=2 at `core.py:276-277` before persistence initialization. Old persistence also rejects P1=2 at `persistence.py:20-21`. Old recovery actual snapshot rejects at 781; old installed detached/native validators reject the changed DDL/marker. These are source-verified control flows, NOT executed old-binary tests. Old Core can open SQLite/set WAL/FULL and touch namespace metadata before the version check; do not claim zero filesystem effects.

Crucially **old install_child.registration_writer does NOT check user_version or P1 revision**: it validates enrollment/prestate then INSERTs registration at 134-153. Therefore 'all old binaries refuse v2' is FALSE for a hypothetically installed v2 root. This is an additional reason installed/native v2 creation/migration stays gated. The approved local v2 root has no installed enrollment and that writer cannot pass its required enrollment check. Do not retrofit old binaries by assertion or weaken the boundary to make the broad claim true.

## Finite implementer exit gate

Parent can forward this contract to a new Sol implementation round; this report does not contact or restart Sol. Implement only the finite source change above. Keep r371's existing evidence attributed. Required focused evidence after implementation:

1. Fresh local v2 and explicitly migrated eligible v1 direct-Core history; retained bindings/rows/rowids and public speech/prepare reads preserved; >1 MiB export succeeds with verified public range download, reopen/idempotent replay/reconciliation, normal ACK/reap/capacity behavior using the inert runtime.
2. Exact v1 legacy reads/pending-intent handling unchanged; every new v1 export/retry refuses before allocation/fault while replay works. Pending/running r371-style state and dispatch-bearing migration roots refuse unchanged.
3. Both exact named models; all mixed marker/user_version/DDL, extra index/trigger, partial groups, invalid row affinity/FK/canonical copies refuse. Scalars at 1 MiB, 1 MiB+1, 36 MiB and 36 MiB+1 exercise policy without near-ceiling allocations; speech/prepare never gain the export cap.
4. Fixed local migration: process lease/control lock exclusion, wrong owner/scope, unknown sidecars, SQL failure after each transition step, confirmed rollback, commit ambiguity, already-v2 reinspection. No live-root migration or fault injection on real evidence.
5. Fresh/resumed installed v1 and native v1 exact model compatibility; explicit installed/native v2 gate; old pinned Core rejection of a disposable v2 DB when test authority permits. Preserve old registration-writer limitation in reporting. No real child/hardware qualification inferred.

No broad new framework/campaign is needed. If implementation cannot meet one of these boundaries, return that concrete gap to the parent; do not silently open installed migration, delete history, keep Core version 1, or enlarge every cap. Actual conversion/image enrollment UNRUN; Linux/amd64, Intel Mac Docker Desktop and OrbStack UNRUN; CUDA/Metal mandatory and unqualified. Headless HTTP/CLI/MCP voice product only; no UI. Last supplied disk evidence remains below the 6 GiB heavy gate, not refreshed here and no waiver. **Design choice bounded GO; F2 source closure pending implementation and independent gate; release NO-GO; legacy CLOSED.**
