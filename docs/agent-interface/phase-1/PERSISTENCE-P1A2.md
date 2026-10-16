# P1A2 bounded inventory and startup integrity candidate

2026-10-16. Implementation handoff for separate Sol-low review. No self-GO, integration acceptance, installed P1 gate or activation claim.

Round-2 correction uses the existing Draft 2020-12 validator to select applicable reference-schema union branches and includes singular `reference_asset_id`. See [RESPONSE-P1A2-r2.md](RESPONSE-P1A2-r2.md) for the current candidate and verification. Round-1 [response](RESPONSE-P1A2-r1.md) remains the integrity-epoch/availability-cost record. Earlier command results and candidate identities below remain historical.

## Identity and exact scope

Assigned worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-9d9cebfc`, branch `t3/p1a2-reconciliation-inventory`, initially clean. Base `b6d16ef31ef478356a2937fb4739da9b0fd0adfc`. A1 integrated predecessor `2b914b0`; independently checked its Core/Persistence/Dispatcher diff against reviewed `29fc22c`: empty. The base ledger records the parent's 103-test root integration run as still running; this assignment does not claim its completion. The separate A1 review's 103 passing tests are predecessor evidence.

Source candidate: `801cfbba590f28f9a999d810f3c4b50099c3b039`. Final report-only handoff SHA is supplied in the response.

Exact changed paths:

- `service/persistence.py`: additive private inventory schema initializer.
- New `service/reconciliation.py`: bounded owner-thread inventory, reference traversal, startup joins, private evidence and pagination.
- `service/core.py`: integrity reads only; asset retrieval validates metadata and exact managed ID before opening bytes.
- `service/dispatcher.py`: startup/status/intent reconciliation and integrity gate; preserve lifecycle stop scheduling.
- `tests/service/test_persistence.py`, `tests/service/test_dispatcher.py`.
- This report.

No additional checkout, subagent, dependencies, contracts/fixtures, worker, packaging, legacy, ledger, reviews or evidence files changed. Read the P1 Astra decision including A2/invariants/private format/full exit matrix, PRODUCT/start/protocol/canonical shared plan/CONTRACTS 1.3.0, R1 authority decision and rounds 1–3, A1 review/report; inspected actual Core, Dispatcher, DispatchStore, Admission, ResourceStore, Authority and worker staging ownership. No applicable AGENTS.md found.

Public contracts/worker protocol remain 1.3.0, Core user_version remains 1, resource protocol remains 2, private P1 format remains revision 1. No populated pre-P1 root enrollment or migration. B1 deployment identity/volume-role enrollment remains unavailable.

## Private schema and interface

The A1 initializer remains unchanged. `initialize_inventory(core)` requires A1 `core.p1`; it creates all three diagnostic tables in one Core transaction, or checks the exact existing column sets. Partial/unsupported schemas refuse mutation.

| Table | Columns and retention |
| --- | --- |
| `p1_inventory_scans` | `id, observed, complete, fingerprint, summary`; scan generations and bounded summaries retained. |
| `p1_inventory_objects` | `id, first_observed, last_observed, scan_id, record`; stable namespace/name digest; first observation preserved, latest verified identity/owner/reason/reference sources stored privately. |
| `p1_inventory_history` | `seq, scan_id, object_id, observed, record`; append-only observations. Old records remain when names disappear or identity/classification changes. |

Private records retain namespace/name, classification, reason, device/inode/type/UID/GID/link count/size/mtime/ctime/hash where read, full known job/attempt/generation/owner/staging/input tuple, and reference-source IDs. Reading atime is excluded from stability comparisons. File identities are observations, never deletion permission. Missing media reports include asset ID and publication/job/artifact reference-source IDs; metadata and terminal job/events/attempts stay unchanged.

`Reconciliation(core, configured_staging).reconcile_startup()` is owner-thread only and refuses an active transaction. It returns `scan_id`, `complete`, `blocked`, `defect_latched`, bounded `counts`/`reasons`, and `cleanup_allowed=False`. Defaults: **2,048 charged entries, 16 MiB combined serialized metadata plus per-name file bytes, one second observation deadline**. Schema, metadata rows, directories and names consume entries; hardlinked names each consume their full bytes. SQL progress interruption, row-size prechecks, bounded reads and descriptor-relative NOFOLLOW descent enforce refusal on incomplete/changing scans. SQLite's existing bounded busy timeout, aggregate historical size queries and begin/completion diagnostic transaction cost are additional to the observation deadline; this is not a hard realtime guarantee.

Only exact control root names, assets/adoption namespace, internal staging/owners and the configured worker staging root are inspected. Only recorded attempt directories are descended, one level; unexplained directories, symlinks and mount crossings are retained/refused. Unknown control names are reported without traversal. No host/legacy discovery, recursive arbitrary path input, media replacement or asset deletion.

`inventory_page(scan_id, cursor=None, limit=20)` returns at most 20 object IDs/classes/reasons/known owner tuples and at most 20 reference-source identifiers per entry, with an opaque random cursor tied to the live scan generation. Pages omit paths and full private identity records. Invalid/changed cursors, a changed database/filesystem fingerprint, an incomplete scan, foreign thread or invalid limit refuse. Page reads repeat a bounded fingerprint verification; changed observations invalidate the generation. Cursor storage reuses offsets and is finite. A restart needs a new scan ID.

Dispatcher marshals both calls to the Core owner. Status adds only the compact integrity summary and `integrity_fault`. Explicit re-inventory requires an idle lane. Integrity defects latch across scans/restarts through retained diagnostic summaries/history; A2 implements no acknowledgement. Replacing bytes/removing a name cannot reopen dispatch. Before every scan or page fingerprint observation, a separate revision-1 `p1_integrity` row commits a random checking token. Diagnostic completion and the same-token healthy/unresolved transition commit atomically. Begin denial prevents observation; completion denial or death leaves checking. Inherited checking is never scanned healthy. Completed unresolved diagnostics may be re-inspected, but never cleared. A death during a healthy scan therefore costs availability until later explicit B2 recovery. Missing authority beside populated diagnostics transitions explicitly to unresolved.

Reference traversal covers every current non-diagnostic Core table and every stored row, including assets/artifacts, original publications/work, jobs/frozen contexts, all project revisions, dispatch snapshots/receipts/outcomes and resource records. It recognizes typed asset reference fields and media metadata IDs, including valid IDs without an `asset-` prefix; references are derived only from explicit table/contract-schema locations; arbitrary scalar prefixes grant no reference meaning. Unknown tables (including future cleanup/recovery tables), column shapes or asset-reference kinds refuse until their owning lot adds an explicit adapter. Existing diagnostic snapshots remain private retained history; A2 never offers cleanup eligibility, even for a complete scan.

## State and fault behavior

| State/fault | Result |
| --- | --- |
| Committed referenced asset | Strict hash/regular-file/owner/single-link availability check; report metadata retention obligation. |
| Pending adoption/known temporary | Retain provenance and identity; absent destination remains incomplete adoption for ordinary exact-reap A1 handling. A2 performs no adoption/unlink. |
| Owned synthetic transient | Associate only with a recorded attempt tuple; retain identity/link count. Classification alone is no ownership/death/grace/cleanup proof. |
| Adopted orphan | Inventory and retain; no inferred publication/success. Blocks admission. |
| Unknown namespace, symlink, unexpected hardlink, mount, unknown schema/reference kind | Retain/refuse; do not descend unexplained names or claim hardlink accounting credit. |
| Missing/hash-changed terminal media | `unavailable`, asset/job/artifact references; strict asset/artifact retrieval fails, job stays succeeded with original history/events. No resynthesis or replacement bytes. |
| Budget/deadline/read failure or changed scan | Incomplete/blocked; no skipped tail represented as complete, no cleanup or new dispatch. Reads/stop processing remain serviceable. |
| Inventory diagnostic write denial | Blocked bounded summary even if evidence cannot be committed; no live automatic retry-to-healthy. |
| Dispatch authority/attempt/deadline/context/publication tuple mismatch, missing/bad reservation, unbound runtime | Unresolved owner/receipt/reservation diagnostic; no publication, execution release, ACK, replacement or fresh key. No demand/reservation reconstruction. |
| Stored receipt/runtime/binding/deadline/reap mismatch, or terminal without exact receipt | Retain original conflicting receipt and reservation; terminality alone grants nothing. Fresh exact receipt can fill an empty slot but cannot clear the integrity gate. |
| Stored outcome with executing reservation | Reject inconsistent atomic outcome/release state; withhold ACK. |
| Exact normal stored receipt/outcome and resolved R1 authority | Ordinary original release/ACK recovery; two restarts preserve one launch/ACK and one terminal event. |
| Inherited R1 checking/invalid authority | Original unresolved fence remains; inventory never calls measure/statvfs/observe or attests resource health. Existing known-fault and F2 stop obligations remain separate. |
| Live page failure with pending lifecycle | Latch integrity/stop obligation, retain responsiveness, service one bounded stop and exact receipt; no publication/release/ACK. |
| Integrity gate and queued cancellation | Refuse combined cancellation/release; retain atomic state. Running cancellation intent remains available. |

Startup reconstructs R1 authority first, inventories before queued cancellation-release reconciliation, then follows the existing asynchronous lifecycle path. The new gate does not return early ahead of pending lifecycle/stop processing. No resource measurement was moved outside R1 fencing, and no profile-derived missing reservation is invented. Disk charges/high-water never shrink. All assets and metadata remain retained. No B1/B2/B3 reset/recovery/cleanup, initializer, UI or public activation is implemented.

## Commands and evidence

Existing isolated interpreter, no dependency install, bytecode/cache disabled; disposable synthetic roots only:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_persistence.py tests/service/test_dispatcher.py tests/service/test_core.py tests/service/test_admission.py
```

**203 passed / 89.32 s**, log `/tmp/p1a2-final-target.log`. This target preceded the last receipt-retention/owner-diagnostic refinements. The final-source focused check below covers those changes; no full-suite final-source equivalence is claimed.

Final-source focused check:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service/test_persistence.py tests/service/test_dispatcher.py -k 'inventory or startup or dispatcher_storage'
```

**30 passed, 62 deselected / 18.68 s**, log `/tmp/p1a2-final-candidate2.log`. Includes actual frozen render context/all project revisions, managed legacy-form IDs, two-restart conflicting-original-receipt preservation, normal receipt idempotence, missing/corrupt media/history, page invalidation, budget/refusal and live stop responsiveness. Separate final receipt/reference checks passed **2 / 2.15 s**, log `/tmp/p1a2-final-preservation2.log`; their first run had one fixture-filename error (one passed/one failed), corrected to the existing `transcript-project_render.json` without changing fixtures.

Broader local service/release command:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/pab-0c-venv/bin/python -m pytest -q -p no:cacheprovider tests/service tests/release
```

**249 passed, 18 skipped / 99.25 s**, log `/tmp/p1a2-final-e1.log`. This broader run preceded the final reference-ID and receipt-retention refinements; final-source affected verification is stated above/below. Skips are platform-specific installed/Linux boundaries, not passed installed tests. No full installed acceptance is inferred.

Development evidence: initial A1 target (before Dispatcher wiring) 68 passed / 30.87 s. Initial wired development exposed stale identity from read-atime changes and recovery blocked by incomplete-adoption classification; corrected with stable stat fields and pending-adoption handling. Focused correction 13 passed/1 failed (empty-root entry budget test), then directory/schema charging corrected that boundary. Next Core/Dispatcher/Persistence target 118 passed / 52.80 s; expanded Admission target 200 passed / 90.33 s. A later owner-tuple enrichment briefly changed object names (28 passed/2 failed); keeping names independent of owner tuples corrected it, and the final 30-test rerun above passed. Failed/intermediate logs remain under `/tmp/p1a2-*.log`; they are not acceptance receipts. Final-source checks supersede them. `git diff --check` passed.

No Docker campaign was launched: the A1 global inventory drift remains unresolved and its runner was not weakened or repeated blindly. No heavy resources, user-data mounts, models/private audio, cloud/STT/Voicebox, remote E2E, prohibited `scripts/run-e2e-test.sh`, prune, push, PR or deployment. Local child processes belong to existing synthetic tests and are reaped by their harnesses.

## Remaining gates and practical limits

These are E1/local synthetic process checks, with in-process E1 runtime receipts for new Dispatcher cases. They do not prove installed PID-1 authority, physical ENOSPC, filesystem/controller/VM/power-loss durability, clean-host installation or acoustic/hardware acceptance. **C2 actual installed gate remains pending.** B1 owns deployment enrollment; B2 owns proof-bound operator acknowledgement/abandonment; B3 owns cleanup; C1/C2 own persistent topology and exact named-volume fault evidence. R1's inherited-checking availability cost remains unchanged.

The closed finite scanner deliberately stops admitting work once its metadata/history/file workload no longer fits its budget; it does not advertise indefinitely unattended throughput. Sampling is detection, not a filesystem quota or exclusion of concurrent writers. Descriptor-relative traversal refuses observed races but is not a general hostile-host security claim. No new deletion path exists.

Next: independent Sol-low review of exact source/diff/schema/state joins and these evidence limits; then parent integration verification. This report makes no GO decision.
