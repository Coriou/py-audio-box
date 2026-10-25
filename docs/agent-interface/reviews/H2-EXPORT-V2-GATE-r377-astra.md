# r377 — Astra overall bounded SOURCE gate: export + publication v2

2026-10-25. **SOURCE GO for parent adoption of the exact candidate below.**
No blocking functional contract break found in the bounded managed PCM24 export
vertical against r372, original F1/F2 and the r374 retained-lock finding.
F1 canonical public provenance, F2 legitimate >1 MiB publication and R374-F1
private owner-lock mode are closed at SOURCE tier. Parent alone decides and
performs merge/adoption. **Release NO-GO; legacy CLOSED.** Installed export
remains unavailable; this is not operational migration or runtime acceptance.

## Verified authority and pins

Reviewed source-readonly candidate
`/Users/ben/.txcode/worktrees/py-audio-box/t3-9c5f400f` at
`531b3eff72ff6b9c8eee6da44f41055bf58fe179`. Verified consecutive parent chain:

- `531b3eff72ff6b9c8eee6da44f41055bf58fe179`: r375 report only.
- `87f33eb1367cce6ed84ca3b73d10ec4146b9b207`: lock correction, only
  `service/publication_format.py` and its focused test module.
- `5fa085a6401e21170b786114523b2c801c5a6831`: r373 report only.
- `052abeac6a352e5c6a844098f58342195f09bd3f`: publication-v2 implementation.
- `381064e9772871e180a0c178b064b4972a6d27cf`: r368 report atop
  `d6f66076fb2c8a5400a1c44e393d8e989e29df56` provenance correction, atop
  `23a682e7eb16e26fc11d121237870d0c577ca324` report, atop
  `fb1e8fe5473db70676aa25f794b9df33c7b3f214` export implementation.
- Ancestor/base and current root HEAD:
  `96a1b249c1b6a505e1d32c500adca10355115458`.

Candidate status empty before and at final source inspection; base-to-HEAD
`git diff --check` passed. Applicable root/candidate ancestor AGENTS checks,
whole candidate AGENTS inventory and root docs inventory found none. Root has
pre-existing untracked evidence/reviews; only unrelated status filenames were
observed, never STT/HELD contents. A lightweight memory registry search returned
no relevant export-review entry; conclusions use current source and named reports.

Read FULL root normative r372, r370, r374, r376, r369 and r365; candidate r373,
r375, r366, r368 and AUDIO-EXPORT. Inspected affected diffs, exact format/migration
implementation, publication and retained readers, admission/retry, HTTP/CLI/MCP,
worker descriptor/child/supervisor, schema/reference/distribution contracts,
installed/native and historical B2 consumers, and focused test assertions.
All source locations below refer to the exact candidate, not root-base code.

Sole write: this ROOT report, mode **0600**. No application imports, tests,
generators, code writes, commit, merge, delegation, actual-root migration, Docker,
network/listener, application child, FFmpeg, ML/hardware, HELD/STT access or cleanup.
No fresh execution evidence is claimed. Shell/git reads and report creation only.

## Source closure and r372 conformance

**F1 remains closed.** `service/persistence.py:410-453` reconstructs the expected
public artifact from retained job/latest attempt, frozen plan, publication manifest
and intent. Public, publication and intent copies must match canonical bytes,
including export source/hash/settings/converter and numeric representation.
`publication_intent:360-386` retains canonical producer/manifest corroboration.
`service/core.py:798-812` invokes the shared reader; `:822-840` additionally binds
an export asset read to its retained producer before enlarging the byte budget.
`service/reconciliation.py:564-571` blocks inconsistent public copies. Existing
prepare results/save use `Core.artifact` (`service/production.py:60-74`). Literal
single-output and explicit two-file prepare intents remain distinct. Missing or
contradictory retained evidence refuses; no fallback/repair or authenticity claim
against coordinated replacement of every copy is introduced.

**F2 is closed across the bounded supported source path.**
`service/persistence.py:13-84` defines exact v1 (1,1; 1 MiB) and v2 (2,2; 36 MiB),
checks marker SQL type and exact publication DDL/index identity, and corroborates
live user_version. Detached selection uses marker plus schema, without inventing
a historical user_version field. Core fresh publication DDL and markers commit
together (`service/core.py:295-318`); existing v1 is not restamped, pre-P1 remains
query-only, and incomplete v2 catalogue initialization refuses (`:325-330`).

`Core.submit_execution:467-470` performs duplicate lookup before the new-export
format gate and before preallocation, resource observation, transaction, job/event,
dispatch settings or reservation. Dispatcher submission/retry retains replay first
and unavailable-converter precedence (`service/dispatcher.py:237-278,324-354`).
`service/domain.py:249-254`, errors.json and operation/schema declarations map
migration-required to a safe HTTP 409, also consumed by thin CLI/MCP. Every NEW
export/retry needs v2; retained keys do not require migration or converter readiness.

`service/persistence.py:302-319` grants the larger ceiling only to a validated
frozen export producer, bounded further by its descriptor. Speech and each prepare
file stay <=1 MiB. Intent/record/public readers receive trusted format and producer
context. `service/core.py:681,709,729-742` carries those through initial intent,
barrier, both two-link checks and final one-link verification. The barrier loads
job/context before cap selection. `verify_file:109-129` hashes bounded chunks and
preserves ownership/link/length/hash/fsync checks. SQL capacity is not authority
for an arbitrary provenance tag. The original 1,152,044-byte case therefore no
longer hits an unconditional 1 MiB intent/SQL/verifier ceiling.

`service/reconciliation.py:93-103,128-137,180-183` selects the separate 160 MiB
media-name allowance only for validated unenrolled local v2, retains 16 MiB scalar
accounting, bounded entries/time, and charges every hardlink name. No general
native/recovery/speech cap increase. This is bounded single-attempt capacity,
not maximum-envelope performance or unlimited retained-history qualification.

**Migration remains the fixed private local transition approved by r372.**
`service/publication_format.py:48-197` requires complete canonical schema groups,
full sqlite_master identity and integrity/FK checks; bounds inspection; validates
raw typed rows, rowids, sequence, documents and canonical retained joins; refuses
enrollment/native/B1/B2/unknown state, dispatch history, active/unreaped work,
pending publication/aliases and unhealthy resources/integrity. Direct terminal
speech/prepare history is the supported retained-history case. No history is
deleted to make a root eligible, and r371's failed running/executing root is not
made eligible or repaired.

At `:273-315`, maintenance lease precedes DB access; existing private namespace,
control lock and retained owner locks are checked, with a bounded WAL-aware private
copy before original mode=rw. Original validation repeats under BEGIN IMMEDIATE.
At `:319-340`, trusted literal rename/create/raw rowid copy/drop and both marker
updates share one transaction; full expected schema and typed rows are rechecked,
COMMIT occurs once, then namespaces and pair are verified under exclusion.
No retained SQL execution, executescript across transition, Core construction
under the lease, checkpoint/VACUUM, generic migration registry or public API.
`:341-364` separates refusal, confirmed rollback and commit/close uncertainty;
unconfirmed close retains the process ownership fence and resources. CLI is only
fixed transition + explicit absolute root + expect-v1 (`:370-378`).

**R374-F1 is closed.** `_owner_info:236-241` requires exact 0600, regular type,
same UID, one link and zero size. `_hold_owners:255-270` applies it before and
after nonblocking flock and matches pathname device/inode to the descriptor.
`_same:244-252` revalidates held descriptor and retained path; calls at 307, 315,
333 and 339 cover pre-writer, pre-DDL, pre-COMMIT and committed verification.
Unsupported modes/identity changes refuse without chmod, replacement or deletion.
Locks survive through DB/private-copy closure. These finite checks do not promise
continuous exclusion of a malicious same-UID namespace actor between boundaries.

**Installed/native and historical consumers stay bounded.** Both persistent
initializer Core sites request v1 (`service/persistent_init.py:177,232`), and
resumed initialization checks v1. Dispatcher supplies the restricted choice;
Core checks a private copy before its writer/native retained activation
(`service/core.py:239-247,279-295`). Native keeps its two exact v1 model
constructions and marker gate (`service/native_activation.py:124-152`). Shared
detached SQL models recognize exact v1/v2 (`service/recovery.py:2901-2955`);
role-specific installed history remains v1 (`:3007,3034-3037`). Historical B2
archives check their own version/schema (`:1572-1575`), not a live root's v2.

Pinned old Core at 381064e9 rejects user_version 2; old persistence rejects P1=2.
This was source-inspected, not executed. Old Core can touch WAL/namespace metadata
before refusal. Unchanged `install_child.registration_writer:98-167` has **no
universal publication-version check** and remains registration-only. Eligible
local v2 lacks its required enrollment. Installed/native v2 stays CLOSED; no claim
that every old writer universally rejects hypothetical enrolled v2 is justified.

The public vertical retains owned managed input, frozen settings/converter,
canonical HTTP and thin CLI/MCP, closed supervised FFmpeg argv and independent
Core PCM24 verification. Worker supervisor checks independent converter/profile/
principal and image/source mappings (`worker/runtime_supervisor.py:335-357`).
Existing attempt/Authority, descendant-reap receipt, publication, ACK and capacity
fences remain. No second scheduler, caller-trusted executable or synthetic success
fallback. Distribution manifests contain the added modules/fixture; the ledger
truthfully says implemented_source_unqualified. These are source seams, not an
installed converter: current installed entrypoints do not enroll it.

## Evidence attribution and limits

No tests ran in r377. Read test code and relied on the following explicitly
attributed prior review evidence, without relabelling it fresh:

- r369: F1 independent 69 PASS / 23.937 s, nine contracts, in-memory deterministic
  generation and prepare/public-reopen checks. Its initial ASGI binding failure
  remains recorded separately from its corrected probe.
- r374: independent 97 PASS / 33.643 s, nine contracts, deterministic in-memory
  generation (20 retained / 19 generated / zero fixture writes). Fresh and
  explicitly migrated empty local roots published/downloaded the 1,152,044-byte
  inert master, preserved source, reconciled, reopened and replayed. The reviewed
  test assertions are in `tests/service/test_export_publication_size.py:49-108`.
  r373's 97 / 34.794 s and later focused 28 / 7.830 s remain implementer evidence.
- r376: independent 26 focused tests PASS / 6.311 s, including retained-mode and
  held-descriptor-change refusal/preservation. Its 4096-combination probe accepted
  only 0600; that is descriptor-predicate evidence, not 4096 migrations.
- The r376 extra pathname-replacement command **exited 1**: it expected
  `owner-changed` but observed earlier `(2, 'refused:local-identity-changed')`.
  Its later SQL/path-preservation and final audit-counter assertions did NOT run.
  This is not an overall command pass. Observed pre-writer refusal plus current
  source supports that boundary; the completed 26 tests separately support
  preservation. No new rerun or additional proof campaign is required to accept
  this narrowly understood label mismatch as non-blocking for SOURCE closure.

All seven real-child reference-publication methods remain **BLOCKED / UNRUN**.
Injected PCM/process/reap receipts prove neither FFmpeg conversion/loudness nor
actual descendant containment, crash/fsync behavior, installation or hardware.

## Parent merge exit and one practical next step

**The fresh overall bounded SOURCE gate is satisfied for 531b3eff. Parent may
adopt/merge this exact chain after its normal destination/pin check.** No further
source correction or broad unchanged-test campaign is requested by this review.
This reviewer neither merges nor authorizes an actual-root migration, installed
v2 activation, converter acquisition or resource-gate waiver.

Next commission ONE fresh-installed CPU export lane for one selected Linux/amd64
family, with an explicit installed publication-v2 creation/activation and writer
compatibility boundary (fresh root only; no existing-installed migration), a real
immutable converter image/runtime closure corroborated by the trusted supervisor,
and sealed registration/enrollment. Merely injecting AudioExports or a digest
into the present v1 installation cannot enable usable export. Keep its finite
acceptance journey: clean install/start, managed upload or existing generated
speech -> >1 MiB PCM24 master -> verified public download -> reopen/key replay,
plus cancellation/descendant reap and no-publication failure; use existing
HTTP/CLI/MCP job and asset controls. This completes the current vertical toward
usable installed CPU rather than adding another feature or proof framework.

Before heavy execution, parent must satisfy the unchanged **6 GiB DISK gate**.
Last supplied disk evidence is about 5 GiB, below gate; not refreshed here.
No RAM observation or waiver is inferred, and no cleanup is authorized here.
Actual converter pin/corroboration/enrollment/conversion/loudness and installed
Linux/amd64, THIS Intel Mac Docker Desktop and OrbStack remain **UNRUN**.
CUDA and native Metal remain **mandatory, unqualified/UNRUN**, with no GPU or
Apple Silicon execution authority supplied. CPU acceptance would not qualify
those platforms or erase the seven blocked real-child tests. **Release NO-GO;
legacy CLOSED.**
