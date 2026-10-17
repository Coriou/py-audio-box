# B2.3b accounting arbitration — enforce a finite maintenance footprint

2026-10-17. **Design decision: retain the existing accounting equation and add a small, independently reviewed kernel storage envelope before finishing preparation. No implementation GO and no integration of `59c4a7373f899f1397277ba4a5a290390c3201bf`.** The candidate's always-refusing preparation is incomplete, not an impossibility proof. Do not send the same unbounded physical-growth question back to the implementer.

Select a dedicated Linux maintenance process with a fixed **4 MiB hard regular-file size limit**, enforced before opening original SQLite for writing and on every unfinished continuation. Combine that absolute limit for DB/WAL/SHM with the actual finite file namespace and bounded exact archive bytes. This provides a conservative footprint bound without predicting SQLite page allocation or WAL length. Keep `max(actual, old_highwater) == expected_highwater`, with the expected value equal to the unchanged old high-water. No ordinary R1 check heals inherited uncertainty.

This is an explicit kernel-envelope amendment, not something authorized by the previous measurement-factoring-only B2.3b allowlist. It introduces a deliberately narrower maintenance capacity and a corresponding possible storage-limit failure. Separate implementation/review/integration of lot K below precedes lot P. All apply entrypoints remain unavailable throughout both. No product question is needed.

## Basis and attribution

Root observed at `568a3afc11290e53fb4aae376023cca396e41a8b`; `git diff ad61e0c HEAD -- service worker scripts tests` is empty. Root therefore retains integrated B2.3a source, without the preparation candidate. Candidate worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-277fefd5` is at exact `59c4a7373f899f1397277ba4a5a290390c3201bf`. Inspected its complete `d57d594..59c4a7` diff: recovery, resource_store, their two tests, and RECOVERY-P1B2 only. `a7d92dc..59c4a7` is handoff-only. No tracked root edits existed at entry; unrelated untracked research is preserved.

Read the **full candidate RECOVERY-P1B2**, including original contracts, all correction rounds and B2.3b calculation/refusal report; full P1B2-KERNEL-ASTRA and P1B2-BOUNDARY-ASTRA; original P1 implementation architecture/exit matrix; R1-FAULT-AUTHORITY-ASTRA; P1B23a-r2-sol and relevant ledger entries. Inspected actual offline/private-copy paths, source projection, plan writer/verifier, accepted/output/archive validators, fixed prefix writers, SQLite transactions, Core scan/completion and the test-only intent builder. No applicable ancestor/scoped AGENTS.md was found. Memory supplied only general exact-scope/evidence-tier guidance, not a B2 result.

**No tests, application probes, SQLite experiments, Docker or network operations ran here.** Candidate reports 29 targeted passes and 1028 passes/18 skips on frozen `a7d92dc10de2240368fedf927c52e51130223121`; these are implementer claims, not fresh Astra verification or B2.3b acceptance. P1B23a R2 reports its independent 850 focused/1002 full passes and 18 skips; the ledger attributes 157 host tests/14.50s to integration. None qualifies the new process/filesystem envelope proposed here.

## What the current code actually bounds

The scan in root `service/resource_store.py:204–246` counts **`st_size`**, including sparse logical length and measured descendant directory sizes. It is not allocated blocks, filesystem journal space, reflink/COW overhead or total host-disk consumption. The report's word “physical” must not turn this into a different metric.

The complete growing namespace through `_complete_core`'s scan is small:

- Mutable SQLite files: `control/core/core.sqlite`, `core.sqlite-wal`, `core.sqlite-shm`.
- New control evidence: the one derived flat `p1-recovery-<archive_id>.sqlite` and `.manifest.json` from `archive_names()`.
- Existing assets, internal subdirectories, prior archives and all staging entries remain counted and unchanged under the freeze. The prefix neither publishes media nor writes staging.
- All creation descriptors, accepted intent, snapshot bundle, original journal and journal replacements are on **receipts**, outside these two measured roots. They still consume the existing receipt namespace/byte limits and real free space.
- The control `.certificate.json` is created **after** the scan and Core COMMIT. Bound it separately if promising the whole completion footprint; do not place it before the actual scan in the protocol.

**The missing directory-growth term is zero in this exact namespace.** `os.walk(root)` adds `dirs + files` beneath the root; it does not add the root's own `st_size`. Every new control file is directly beneath `control/core`; its directory growth is therefore not counted. Staging is untouched; no counted descendant directory receives a new entry. This is a source-derived fact conditional on freeze/no extra writer, not an ext4 directory-size estimate. Adding a recovery subdirectory or a temporary under a counted descendant would invalidate it and must fail a namespace regression. Do not change the scanner to manufacture this result: it already behaves this way.

The SQLite term really is absent from the old proof. `page_count * page_size` is not an upper bound on the WAL or SHM, and a logical row/JSON ceiling is not one either. The candidate's `2 * MAX_BYTES` archive allowance also exceeds its 20 MiB fixture H before SQLite is considered. Use actual generated bytes or explicit smaller admission ceilings, not that universal allowance.

Actual original write order is optional B2 schema installation, plan insertion, then receipt acceptance/fence/bundle, Core intent transaction, pending step transaction, evidence files/evidence step, journal replacement/journal step, **scan**, Core transaction/certificate, receipt finalization/fifth step. `_prefix_transaction` stops after an ambiguous COMMIT. A retry inspects retained state; it is not permission to replay SQL blindly. The bound below is independent of transaction count, checkpoints and retry count.

## Selected supported envelope and its precise availability cost

Use an internal constant `L = 4_194_304` bytes. No request/environment/plan parameter may choose a larger limit. This is a maximum file **length**, not a reservation, budget increase, preallocation or instruction to write padding.

The supported model for K/P is:

1. Local Linux, ordinary regular-file semantics on a declared local ext4 filesystem, standard SQLite Unix VFS and WAL mode. No network/FUSE filesystem, overlay-backed authority root, alternate/custom VFS, externally modified file or unqualified mmap/file-growth implementation. C1/C2 must later identify actual kernel/filesystem/VFS/build/mounts. E1 can model those declared assumptions now; macOS local tests do not establish the Linux mechanism.
2. A dedicated, single-threaded one-shot maintenance process, not an ordinary control/worker or the invoking host shell. Before any writable SQLite open, set both soft and hard `RLIMIT_FSIZE` to L; read back and require equality. No path raises/restores it, including error handling and resume. Unsupported platform, failed set/readback, or missing fixed-process entry refuses. Treat SIGXFSZ/EFBIG as failure, never success; use a deliberate signal disposition so ordinary short-write/error handling remains effective. A fatal signal is also an interrupted invocation, not a completed recovery.
3. Under the declared Linux regular-file contract, any extending write/pwrite/truncate/ftruncate by that process cannot produce file length greater than L. SHM growth must pass through that same limited truncate path; shared mappings cannot extend EOF by themselves. SQLite database mmap is disabled for this envelope. No external writer, background SQLite owner, child helper with another limit, attached database, extension, user trigger or writable VFS escape is supported. C1 must exclude privilege/launcher paths that raise the hard limit; hashes and a proof-shaped JSON value do not establish this process property.
4. Acquire the existing control lock then lane lock and validate the existing trusted host capsule/full owner freeze. Locks alone are insufficient. Retain both locks through each owned operation. No live readers/writers may checkpoint, extend or replace the original SQLite files behind this process. An administrator changing storage or restarting owners violates the freeze; detected drift refuses. No protection against a malicious administrator is claimed.
5. Before the original writable open, require each present DB/WAL/SHM file to be a supported owned regular file with size <= L, and reject unknown SQLite sidecars. Never truncate a larger file to make it eligible. Check the same constraints on unfinished resume. Static unrelated measured files may be larger than L: they are not rewritten and are fully included in M. If copying retained evidence under the bounded process would exceed L, refuse that population rather than rewriting it.
6. Keep WAL/FULL, `trusted_schema=OFF`, original schema/version and exact namespace. Query and require existing `journal_mode=wal`; do not convert an old journal mode as part of preparation. Private maintenance connections use `temp_store=MEMORY`, `mmap_size=0`; prohibit ATTACH/extensions/unknown schema. A connection may use `cache_spill=OFF` and disabled auto-checkpoint for predictable small-workload behavior, but neither is the size proof. All original-file openings, checkpoints, backup destinations and closes in this path run inside the limited process. No VACUUM, manual WAL deletion, recovery-root preallocation or persistent database page-limit change is required.
7. For this first preparation envelope, admit only an actual original snapshot of at most **1 MiB**, exact canonical manifest at most **64 KiB**, and control certificate at most **64 KiB**. Every other newly written file must also fit L and its existing smaller per-document limit. Keep all original 2,048/16 MiB aggregate limits and the 64 KiB receipt-certificate ceiling. These are additional conservative refusals, never enlarged historical limits. Small valid, missing and bounded unreadable journal populations fit; a large journal may legitimately refuse.

This mechanism bounds length even if SQLite spills more pages, retains an old WAL, fails a checkpoint or repeats an interrupted attempt. It does **not** prove unlimited progress below a logical budget. Hitting L can leave accepted recovery pending/quarantined, just like a failed storage write. Do not catch EFBIG/SQLITE_FULL, raise L and retry; do not call that physical host ENOSPC evidence.

**Explicit amendment to the earlier gate:** preparation proves a footprint invariant under an enforced storage envelope, rather than proving that every future SQLite write will fit without a new storage-limit failure. This is the minimum availability concession selected here; accounting semantics do not change. Before returning an intent, reject known-too-large documents and run a bounded private capacity/semantic rehearsal of the concrete SQL sequence with the same limits. That rehearsal detects predictable capacity failures; it is not used as the physical-growth proof or as a promise of future I/O success. The required actual nonempty production-generated positive below prevents this from becoming another vacuous refusal design. If a review requires guaranteed completion for every accepted population despite storage failures, that is a different and stronger requirement than this decision.

No new storage authority file or second accounting database is needed. A process-capability object is internal plumbing, not caller authorization. The trusted proof mount still supplies host origin; the file limit supplies a separate storage property. Neither substitutes for the other.

## Finite footprint formula and nonempty numerical witness

Use the **same factored measurement** for preparation and scan. At a stable locked preparation observation, define:

```text
M     = total current measured st_size across control/core + staging
D0    = sum of current lengths of the three exact DB/WAL/SHM files
H     = retained resource_accounting.highwater, read from validated source
C     = sum of every retained reservation demand.disk_bytes
B     = unchanged enrolled disk budget
S     = exact original snapshot byte length, <= 1_048_576
J     = exact canonical manifest byte length, <= 65_536
K     = exact control certificate byte length, <= 65_536
L     = 4_194_304, hard per-file process ceiling

U_scan_exact = (M - D0) + 3*L + S + J
U_all_exact  = U_scan_exact + K

# Simpler early guard, before a plan or snapshot has been written:
U_scan_early = M + 3*L + 1_048_576 + 65_536
U_all_early  = U_scan_early + 65_536

require U_all_early <= H and H + C <= B
require current free-reserve check; repeat the existing real check at scan
expected core highwater = H; expected accounting rows = [[1, H]]
```

Use the deliberately looser early guard in version one, so capacity refusal does not require writing a plan first. D0 subtraction is optional later precision, not necessary for a positive. Never subtract old archives, demand charges or a file because it is sparse. An exact-size calculation must include every static measured entry; no blanket SQLite/archive exclusion from M. Check finite integer arithmetic before encoding.

Proof: all static entries contribute at most M (or exactly M-D0 in the tighter form); three mutable SQLite regular files each have length <= L at every reachable supported point; only the two named control evidence files are added before the scan, with exact bounded lengths S and J. Counted directory delta and staging delta are both zero. Thus actual scan <= U_scan <= H and `max(actual,H)=H`. Core/fifth transactions are already covered by the same absolute SQLite caps; adding K also covers the later control certificate. This is a bound on the existing metric, not on allocation metadata or free-space consumption.

For the reported coherent fixture **only as a numerical prestate**, M=188,608, H=20,971,520, C=2,097,152, B=67,108,864:

```text
U_scan_early = 13_885_632
U_all_early  = 13_951_168 < 20_971_520
H + C       = 23_068_672 < 67_108_864
```

Therefore the policy is not arithmetically empty and the reported 46-page/4096-byte database is not disqualified by the initial size ceiling. This is **not** a fresh execution or a claim that the unimplemented full preparation passes all other validators. The required green witness must produce the real intent/snapshot and complete the existing private kernel under the envelope. H=20 MiB remains test prestate; production reads it and never creates it. No host disk filling, synthetic padding, historical H UPDATE, user-data mutation or budget change is a way to obtain eligibility.

Entry counts also need an early check: reserve up to two absent SQLite sidecars plus two new control archives, and one later certificate, within the existing scanner/namespace ceilings. Continue enforcing SQL/evidence populations, receipt-file aggregate sizes and deadlines separately. Byte arithmetic alone cannot qualify an oversized history.

## Proof-first preparation and acyclic snapshot binding

The candidate `_prepare_abandon_locked` currently checks prerequisites then raises. It is not a source/history verifier. Its read-only `offline` handle cannot commit an authoritative plan. Do not merely remove the final raise or call the test `b2_intent` builder.

Use these dependencies in lot P, with locked seams supplied by K where needed:

1. In the dedicated bounded process, read the production fixed-mount capsule using the OS clock. Require pre-plan purpose, exact scope/deployment/volume identities and the complete retained registration union. Build/freeze the kernel proof **first**, from that capsule, and compare its prior epoch/old lane to the validated owned state. The proof's hash is final before the plan exists. Transport origin is not a `mock_auth`/E1 flag or an `actual-host` string.
2. Hold both locks. Read a source-preserving private WAL-aware copy for early validation, ordinary authority coherence, journal bytes/status, all history/media and the early footprint/quota/capacity gates. Keep B1 strict valid-journal inspection unchanged. Factor a narrowly typed B2 observation mode for valid/missing/unreadable journals rather than inventing a half-built intent to bypass snapshot validation. Missing/corrupt bytes are not clean receipt evidence.
3. Validate complete diagnostic coverage before a plan: every required degraded-media/orphan decision must match the exact object/reason/content/availability; unknown schema, owner/reservation origin and foreign namespace remain unacknowledgeable. Reuse `_complete_core`'s eligibility rules as a pure validator so known refusals are caught before acceptance. No operator-authored actions or expected rows.
4. Open the original writer only after process/size checks, while retaining both locks and rechecking the locked logical prestate. If needed, install the complete optional B2 schema atomically, using its existing DDL, after predecessor validation. A private shadow may validate that prospective schema first; it must not be mistaken for original durable state. Compute the canonical source **after** schema installation, including every prior plan/history/reference and actual inventory.
5. Choose plan/recovery/archive IDs and new lane/epoch once. Construct the plan with final proof digest, observation digest, source digest, unchanged budgets, journal state/digest and exact diagnostic decisions. Insert that final plan once. No UPDATE of a published plan or digest. Immediately run the production plan verifier under the same locks, excluding **only** the exact current proposal row and typed reference label. Existing `verify_abandon_plan` needs the same typed B2 journal observation path for damaged-journal plans. Later plan appends must still stale older unaccepted plans.
6. Capture the actual bounded SQLite backup **after that plan is committed and before current accepted-intent/step writes**. It includes the exact current plan and all older rows. Record its immutable bytes and SHA-256 (the `snapshotHash` binding is `expected[evidence].output.snapshot_digest`). The before/source projection excludes only the current plan; `_validate_archives` must continue adding that exact plan back when checking the snapshot's actual tables. Do not hash a shadow database and claim it is the original snapshot.
7. Build canonical before/proof/actions/output rows from validated owned state. Preserve every terminal job tuple and context. For nonterminal jobs change only the permitted unsuccessful state and latest attempt reap bit; preserve events, dispatch/command/receipt/outcome/ACK, demands and historical attempts. Reservations become retained; monitor/fault/accounting output is exactly the existing closed contract with H unchanged. Integrity settlement and resolution rows remain the actual kernel's prescribed transaction, not arbitrary caller patches.
8. Build pending output, evidence output using the captured snapshot hash, new-journal output and Core output. Construct `archive_manifest` from its existing nonrecursive fields (source/before/proof/snapshot/journal), then its digest; build the five descriptor hashes in order from `before_digest`. The fifth output binds the fourth output digest; the certificate binds the fourth step. Finally compute intent digest. The snapshot contains the plan, **not the intent that hashes that snapshot**. No fixed point, digest rewriting or zero-placeholder value may escape as a final document.
9. Validate the result with `validate_b2_record`, actual `_validate_archives` on private protected copies, full database/history validators and bounded prospective output/certificate/receipt budgets. Rehearse the concrete production SQL operations on a disposable private copy with the same storage envelope; share the actual prescribed SQL writer where factoring is needed, never introduce an alternate recovery implementation. Rehearsal cannot mutate original authority, grant a healthy epoch or supply a replacement source fingerprint. Failures return no applicable intent.
10. Return the four immutable public-to-this-private-layer fields `{plan, plan_digest, intent, intent_digest}`. Retain exact snapshot bytes only in a private same-lock preparation session for the later kernel acceptance seam. Preparation alone writes no receipt marker/fence/accepted intent/bundle and no healthy authority. A failed final check may leave an immutable **unaccepted** plan/schema installed; report this explicitly. Never delete or rewrite it on failure. Early low-H/quota/unsupported-envelope negatives must refuse before those writes.

There is a real snapshot handoff issue to fix, not wish away: current `_pending_prefix` recaptures a backup in its `if not accepted ...` branch even when a `snapshot_bytes` argument exists. K must factor capture/validation so a same-lock, production-owned prepared snapshot can be consumed exactly once without recapture after its hash is frozen. Validate its actual rows/plan/hash again. The accepted document/fence and **owned durable bundle** still precede Core accepted SQL; no snapshot is ever restored over Core.

The four-field preparation result is not durable restart authority by itself. If the preparation process exits before receipt acceptance, no new authority has been accepted. A later initial invocation must either recapture bytes matching the frozen hash exactly and revalidate the entire plan, or refuse before acceptance; it must not rewrite the hash to match a new backup. Once acceptance/bundle exists, restart loads only the owned accepted document and bundle under the original proof, after-TTL rules and no-remint policy. Same-lock preparation-to-acceptance is the primary positive path for later B2.3c; K/P do not expose it through apply.

## Sequential finite Sol-low briefs

### K — bounded maintenance process and exact snapshot handoff

Predecessor: root records this decision; no automatic adoption of candidate `59c4a7`. Exact implementation allowlist:

- `service/recovery.py`: fixed process-envelope setup/verification, private maintenance connection configuration, shared lock-held session seam, prepared-snapshot capture/validation/handoff, and only necessary factoring of the existing prescribed SQL operations for private rehearsal. Preserve existing public apply refusal and all history/resume/order checks.
- `service/resource_store.py`: extract/reuse the candidate's unchanged bounded measurement loop and expose a private calculation using fixed envelope constants; keep ordinary ResourceStore and `_recovery_scan` eligibility/arithmetic/equality unchanged.
- `tests/service/test_recovery.py`, `tests/service/test_admission.py`: K-specific bounded-process, namespace, snapshot and preservation tests.
- `docs/agent-interface/phase-1/RECOVERY-P1B2.md`: append K assumptions, exact finite namespace/formula, refusals, actual results and limits.

No Core/persistence schema, admission, dispatcher, worker, host script, packaging, dependency, contract, ledger or existing review edit. The process helper is an internal one-shot seam, not a new enabled CLI operation or flag. Ordinary B1 inspect and ordinary Core must not inherit the reduced process limit. Keep the existing supplied-document private kernel tests as internal semantic tests; they do not qualify the bounded production route. New production-route tests may not select an unbounded branch.

Required RED/GREEN and exits:

1. Before implementation, demonstrate the missing bounded-session/snapshot handoff behavior with coherent tests, not merely an absent-name assertion. Green: no original writable open before a verified envelope; busy locks/unsupported platform/limit mismatch leave authority untouched. Raising a caller limit or supplying caller capabilities cannot enable the route.
2. In disposable child processes on a supported E1 platform, test extending write, pwrite and truncate at L and L+1 with short-write/error/termination handling and file length never > L. Exercise real SQLite DB/WAL/SHM creation and checkpoint/close under that limit. Use small owned files, not disk exhaustion. Where Linux is unavailable, explicitly model the contract and mark actual Linux cap evidence pending; do not count skips as proof.
3. Measurement test proves root-directory size is excluded and descendant directory size included; adding flat control archives changes measured bytes by their file lengths, not root directory growth. Trace the actual prefix's created paths: no counted descendant directory/staging mutation, no unexpected SQLite file. Test source sidecar > L refusal without truncation. Hard cap remains across fresh unfinished resume.
4. A captured snapshot's hash and logical contents survive exact same-lock kernel acceptance and bundle persistence; neither the snapshot nor plan digest is reminted. Tamper one byte, plan row, source row or old archive and require refusal. Actual process death after bundle/Core-intent boundaries resumes from durable bytes only. Preserve missing-attempt/partial-output quarantine and all F1–F4 history cases.
5. Limit/SQLite/file-write failure at each relevant acceptance/transaction boundary never leaves partial healthy resource/integrity authority. Ambiguous COMMIT closes the session; fresh invocation verifies retained state. No auto-raise, live SQL retry, checkpoint-as-proof, deletion or cleanup.

K exit is a separately reviewed storage/snapshot foundation, **not B2.3b completion**. Run targeted regressions and the existing four B2 suites/full service-release preservation sequentially on frozen source/tests; retain attributed failures/skips. Independent Sol-low review must audit the process/OS assumptions and the unchanged scan equality, then root verifies exact integrated source. No approval by this report of code that does not yet exist.

### P — complete production B2.3b preparation under K

Predecessor: K independently reviewed/integrated, including any retained measurement factor. Exact allowlist: `service/recovery.py`, `service/resource_store.py` **measurement/calculation reuse only; no new accounting changes**, `tests/service/test_recovery.py`, `tests/service/test_admission.py`, and append `docs/agent-interface/phase-1/RECOVERY-P1B2.md`.

Implement the ten-step preparation above; no source expansion into C1/C2 or apply wiring. `_prepare_abandon(roots, scope, *, profiles, budget, diagnostic_decisions)` remains the closed private request shape. `_prepare_abandon_locked` operates on the owning bounded session, never acquires the same locks again and never begins a nested transaction. Caller cannot supply actions, before/expected rows, snapshot/proof bytes, clock, H, limit, capability flag or budget-after. Share production validators, not fixture builders.

Required RED/GREEN and exits:

- **Decisive nonempty green:** coherent enrolled real queued/allocated authority with retained H and unchanged budget; use the real preparation function and trusted-reader path with explicitly modeled host/OS transport only. It returns a nonempty immediately self-verifying plan and immutable intent, passes every semantic/archive validator and executes the existing private kernel to five-step completion under K. Assert exact row deltas, archive bytes, `max(actual,H)==H`, unchanged charges/budget, one lane/epoch and both startup directions. No import/call of `b2_intent`, supplied expected documents, post-hoc digest UPDATE or production high-water UPDATE. Fixture H may be seeded as declared test prestate only, before preparation begins.
- RED on the current always-refusing candidate must fail at the expected successful preparation assertion on that coherent population. The same test must pass in P; a reclassified refusal is not green. Report actual M/H/C/B/S/J/K and observed scan, plus the calculated bound.
- Low H below the early bound, H+C over budget, missing/free reserve, unknown filesystem/VFS/process envelope, larger-than-cap DB/WAL/SHM, oversize snapshot/manifest/certificate/history and private capacity failure all refuse **before receipt acceptance**. Early bound/quota/envelope negatives leave plan/schema/authority unchanged. No healthy epoch or charge adjustment on any refusal.
- Valid, small unreadable and missing journal positives retain exact original bytes/absence, full owner coverage and coherent allocation authority. Omitted owner/live owner, unknown namespace or reservation origin refuse. Large unsupported damaged-journal population refuses honestly, without increasing a ceiling.
- Exact required degraded decisions produce a valid intent; missing/extra/changed decision content, new defect, prior history deletion/change, later plan append, media drift and source drift refuse. Existing terminal success and unavailable retrieval are retained. Ordinary inherited checking remains blocked on two restarts.
- Trace immutable proof creation before the single plan INSERT; disallow plan UPDATE/digest rewrite in production tests. Verify snapshot contains the actual current plan but no current accepted intent; before/source excludes only its exact row/reference. Rehashing a tampered expected row must still fail the semantic oracle.
- Kill after kernel acceptance/bundle/Core-intent using the production-generated intent, discard all caller RAM, and resume twice in a fresh process from owned files. Also cover preaccept preparation exit, hash-mismatching recapture, and no-remint behavior. Existing kernel fixture crash tests alone do not satisfy this row.

Exit report gives exact candidate/frozen SHA, source diff, full successful call chain, unchanged budgets/accounting, every unsupported population and actual failed/passed/skipped checks. Separate independent Sol-low review and root integration verification are mandatory before B2.3c. A passing refusal suite alone never qualifies P.

## Candidate reuse, alternatives and remaining gates

The candidate's measurement extraction and focused tests are useful material. If retained, independently review that exact factor against root's original loop, then integrate only the reviewed delta through K; do not cherry-pick an always-refusing implementation and mark preparation complete. Its `_PreparationUnavailable` diagnostic object can remain a refusal result, but an unconditional `prefix_upper_bound=null` is superseded for the supported K envelope. The candidate has not received independent acceptance in this arbitration.

A page-limit-only design was considered and is not selected: `max_page_count` alone does not cap WAL/SHM; `journal_size_limit` is not a peak-write bound. A cache/spill/transaction-count proof would additionally need pinned SQLite/VFS write/sector-padding, restart and checkpoint reasoning. The hard file-length envelope avoids that dependency. A private dry-run supplies capacity/semantic evidence, not a future physical-size oracle. Closed preallocation would need new evidence slots, admission/migration and crash rules; it is larger than the fixed-process constraint. Deterministic logical accounting would change scan semantics and multiple readers and is unnecessary here. None of those alternatives is silently authorized.

Unchanged-budget-only remains selected. Explicit budget increase is future work with its existing full configuration-consumption obligations. Low-H populations remain unsupported; no historical-counter inflation, credits for deletion or user-data manipulation. No host-disk filling or storage provisioning is authorized by this report.

B2.3c still owns trusted closed apply/resume wiring, fresh host incarnation checks and same-lock acceptance; C1 owns the actual pinned process/image/entrypoint, hard-limit setup, UID/capability/filesystem/VFS model, proof mount and exact maintenance-consumer exception. C2 supplies installed evidence. Modelled transport/rlimit checks are E1 assumptions, **not actual host authentication or installed enforcement**. Keep host-origin authentication distinct from content hashes and storage constraints. Normal services get no proof mount or recovery bypass.

**A1 global Docker-inventory drift remains unexplained and unqualified, with no cause or waiver.** Physical ENOSPC/power loss/daemon/VM/controller durability, hardware/audio/engines, clean-host operation and public/unattended acceptance remain unqualified. B2.3b/c, B3, C1/C2 and final global Astra remain pending. This source-only decision supplies concrete K/P work, not implementation acceptance, activation or release authority.

Only `docs/agent-interface/reviews/P1B23B-ACCOUNTING-ASTRA.md` is written. No source/test edits, tests/probes, commits, subagents, Docker/network/audio/models, push/deploy or cleanup.
