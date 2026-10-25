# r374 — independent Sol review of publication v2 / F2

2026-10-25. **SOURCE NO-GO for adoption of this exact candidate: one P2 private-migration prestate gap.** The original r370 F2 size failure is closed in the exercised fresh/migrated local paths; F1 canonical provenance remains closed. The migration nevertheless accepts a non-private retained owner lock contrary to r372's ownership/mode boundary. Fix the narrow check and obtain the parent-owned fresh Astra gate. No adoption, merge or release authority is exercised. **Release NO-GO; legacy CLOSED.**

## Pins, authority and cleanliness

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-b413d79c` resolved and verified:

- HEAD/report-only `5fa085a6401e21170b786114523b2c801c5a6831`.
- Implementation `052abeac6a352e5c6a844098f58342195f09bd3f`.
- Prior F1 report `381064e9772871e180a0c178b064b4972a6d27cf`.
- Base/root HEAD `96a1b249c1b6a505e1d32c500adca10355115458`.

Verified ancestry, HEAD's report-only diff, and clean candidate status before and after execution. Base-to-HEAD `git diff --check` passed. Root has pre-existing untracked STT evidence and review reports; only their status names were inventoried, without reading STT content. No applicable ancestor or relevant nested AGENTS found. Read full root normative r372, r370 and r369; candidate r373 and retained r371/r366/r368; implementation diff and current publication, admission/retry, public read, reconciliation, detached/recovery, B2, initialization/native and registration consumers. Memory supplied the canonical numeric-identity caution only; conclusions use current pinned source and fresh evidence.

Sole persistent write is this root report, created mode 0600. No source/test/fixture edits, new runner/log files, commits, merge, delegation, self-adoption or actual-root migration. Existing inspected temporary runners were executed directly. Disposable inert fixture writes/teardown only.

## Finding R374-F1 — P2: retained owner lock mode is not validated before migration

**Location:** `service/publication_format.py:250-257`, especially the predicate at **253**, reached before the original DB is opened writable at 296.

`_hold_owners` checks that each known retained owner lock is a same-UID regular, single-link, zero-length file, then obtains its flock. It never checks `stat.S_IMODE(info.st_mode)`. `_namespace` validates only the top-level owners directory, not these files. The local reconciliation owner branch (`service/reconciliation.py:214-218`) classifies a known regular owner file as an owned transient without checking its mode either. Consequently the two `_validate` calls do not supply the missing check.

Independent guarded inert reproduction on two separate physical temporary roots:

1. Create exact v1 with `Core(root, _publication_creation=1)`.
2. Use inspected `tests/service/test_publication_format.py:direct_history` to publish a tiny direct-Core speech job with a terminal/reaped attempt and no dispatch history.
3. Create `root/owners/<retained attempt owner_id>` as an empty regular file, first mode 0600 (control), then on a separate fixture chmod it to 0666. Close Core.
4. Call `migrate(root, expect_v1=True)` and inspect the committed format and retained mode.

Fresh receipts:

```text
OWNER_MODE 0o600 RESULT (0, 'migrated') VERSION 2 RETAINED_MODE 0o600
OWNER_MODE 0o666 RESULT (0, 'migrated') VERSION 2 RETAINED_MODE 0o666
```

Expected for the second fixture: exit 2 / refused before original writer/DDL, preserving logical v1 and the evidence file unchanged. Actual: migration succeeds and commits both v2 markers while retaining the non-private lock. This violates the approved private ownership/mode prestate; it does not reproduce the old export quarantine. The root remains 0700 in this probe, so this is not a demonstrated cross-user exploit or lost-lock event.

**Finite correction:** validate the supported private mode of retained owner lock descriptors alongside the existing UID/type/link/length checks, without chmod/repair/deletion. Add a disposable mode-refusal case that checks unchanged logical v1 and retained lock evidence, retaining the valid 0600 success and active-lock refusal. No new migration framework or operational root test is needed.

## F2 closure and consumer assessment

The coherent named formats are implemented: v1 `(user_version=1, P1 integer 1, exact 1 MiB SQL)` and v2 `(2, integer 2, exact 36 MiB SQL)`. `persistence.store_format` corroborates live pairs and exact publication DDL/index identity; detached selection uses exact marker plus model rather than inventing historical user_version. Mixed pairs, missing markers and altered/extra publication schema refuse. Ordinary fresh local creation commits publication DDL and both markers together; incomplete v2 catalogue initialization refuses. Existing exact v1 stays v1, and pre-P1 query-only behavior is retained.

`Core.submit_execution:470` gates every new export immediately after duplicate lookup and before native preallocation, resource observation/reservation, settings/jobs/events or transaction. Dispatcher submission and retry preserve replay first and installed/unavailable-converter precedence. The safe public migration-required error is HTTP 409. Fresh regression confirms unchanged v1 reservation state/no persistence fault, retained-key replay, new retry refusal and truthful operation detail.

The authenticated export ceiling passes through intent, publication/public-artifact readers, Core publish/barrier, both links=2 alias/destination verifications and final links=1 verification. The barrier loads producer job/context before cap selection. `publication_limit` binds frozen export plan/input hash/execution; a provenance label alone does not grant 36 MiB. F1 producer/publication/manifest/intent comparisons remain canonical, including numeric representation. Speech and each prepare file retain 1 MiB. The verifier streams bounded hash chunks and retains ownership/link/size/hash/fsync checks. Direct export asset reads now require the retained producer/public artifact join too.

Fresh and explicitly migrated empty v1 roots both complete the inert **1,152,044-byte PCM24** ASGI publication, verified CLI range/hash download, unchanged source, healthy retained reconciliation, reopen and same-key replay without converter registration. The allowlist checks succeeded/idle state, no executing reservation, quarantine or persistence fault. These are injected PCM/process receipts, not FFmpeg or descendant-containment proof.

Local v2 inventory selects 160 MiB media-name accounting separately from 16 MiB scalar accounting, retains 2048 entries and the finite scan deadline, and charges each alias name. Scalar policy probes cover 1 MiB, +1, 36 MiB, +1 without ceiling media allocations; speech/prepare retain their cap. Accounting tests use scalar charges rather than large files. This establishes bounded logic, not maximum-envelope latency or unlimited history.

The migration otherwise has the finite intended surface: explicit fixed transition/expectation, maintenance lease before DB, existing control lock, bounded WAL-aware private copy before mode=rw original, complete canonical schema groups, typed/raw rowid preservation, terminal/reaped direct history, dispatch/enrollment/native/B1/B2/unknown/pending refusal, and one explicit transaction for table replacement plus both markers. It does not execute retained SQL, use executescript across transition, construct Core under the lease, disable FKs, checkpoint/VACUUM/replace the DB, or repair retained state. Fault tests exercise rollback boundaries, post-COMMIT ambiguity, eligible already-v2 inspection and uncertain-close ownership retention. The finding above prevents accepting its entire prestate boundary as complete.

## Installed/native, recovery and old-version limits

Both trusted persistent initializer Core creation sites explicitly request v1. Native/installed Dispatcher construction requests v1, and restricted existing Core construction checks a source-preserving private copy before its writer/retained activation. A separate fresh guarded probe created a disposable local v2 root and invoked `Core(root, _publication_creation=1)`:

```text
RESTRICTED_V2_REFUSED installed_native_publication_v2_unsupported
RESTRICTED_V2_LOGICAL_ROWS_UNCHANGED
```

This tests the restricted constructor, not actual installation or enrollment. Shared detached validators recognize exact v1/v2 models; role-specific installed/native acceptance remains v1. `_INSTALLED_V1_TABLE_DDL` is retained separately; native's two v1 model constructions and 1 MiB media rules remain. B2 archive validation still checks its own historical v1 snapshot version/model, not the current root's version. No B2 writer/certificate/abandon migration is introduced. Current host/package/daemon initialization paths retain their role checks; no installed v2 enrollment is authorized.

Pinned old Core source at 381064e9 rejects user_version 2, and old persistence rejects P1=2; old snapshot/native/installed models reject the changed pair/DDL. This is source inspection, not an executed old-binary trial. Old Core may touch WAL/namespace metadata before refusing. The unchanged old `install_child.registration_writer` has **no publication-version check**: do not claim universal old-binary refusal. It remains registration-only and requires enrollment absent from eligible local v2 roots. Installed/native v2 remains closed.

## Fresh execution receipts

All application/test execution used `/private/tmp/pab-0c-venv/bin/python -I -B`, asserting CPython **3.14.7**. Inspected `/tmp/pab-r373-check.py`, `/tmp/pab-r373-contracts.py`, both conftests, new migration/size test code and fixture helpers before execution. The runners and independent inline probe install the audit hook **before application/test imports**, refusing `subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect`. No broad pytest or conftest activation.

Exactly **one 97-test allowlist run: PASS / 33.643 s**, with inspected existing check runner and these exact candidate-relative paths in order:

```text
tests/worker/test_audio_export.py
tests/service/test_audio_export.py
tests/service/test_production_api.py
tests/clients/test_production_clients.py
tests/service/test_production_admission.py
tests/service/test_production_reconciliation.py
tests/worker/test_production_synthetic.py
tests/worker/test_engine_lifecycle.py
tests/service/test_engine_admission.py
tests/contracts/test_reference_roles.py
tests/contracts/test_legacy_controls.py
tests/service/test_export_publication_size.py
tests/service/test_publication_format.py
```

Nine explicit contract functions PASS: schema/all fixtures; exact IDs/defaults/source coverage; revision/idempotency; lifecycle/cancel/publication; device/readiness; ticks/immutability; migration/worker boundaries; operations/jobs/artifacts/errors; binary fixture boundary. Two intercepted in-memory generations PASS: **20 retained / 19 generated / zero fixture writes**, identical complete maps/name sets and unchanged disk fixture map. Fresh output is in tool receipts; no new persistent log was written.

The independent owner-mode boundary probe produced the unexpected successful migration documented above; it was not tuned or rerun as a pass. The 97-test suite was not repeated. All seven real-child reference-publication methods remain **BLOCKED / UNRUN**, never relabelled passing. Fake CPU/MPS/CUDA labels and injected reap observations are source fixtures only. r373's historical 97/34.794s and label-only 28/7.830s receipts remain attributed historical evidence, distinct from this fresh run; its earlier failures remain in the retained report.

## Disposition and remaining gates

**Original F2 functional size correction: bounded source GO. Overall exact candidate adoption: SOURCE NO-GO pending R374-F1.** No additional blocker was identified within this finite review. Parent owns the narrow correction and fresh Astra gate, then any adoption decision. This report does not approve an actual migration.

Actual conversion/loudness, immutable converter/image/runtime corroboration, sealed registration and installed-family enrollment: **UNRUN**. Actual Linux/amd64, Intel Mac Docker Desktop and OrbStack installation: **UNRUN**. CUDA and native Metal: **mandatory, unqualified/UNRUN**. No Docker/build/pull/load, network/listener, application child, FFmpeg, ML/hardware, HELD/STT access, operational cleanup, install/acquisition/download or actual-root migration occurred. No heavy 6 GiB waiver; prior below-gate disk evidence was not refreshed. HTTP/CLI/MCP source evidence does not establish installed product qualification. **Release NO-GO; legacy CLOSED.**
