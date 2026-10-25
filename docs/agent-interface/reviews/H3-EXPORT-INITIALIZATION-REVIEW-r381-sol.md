# r381 — bounded initialization/writer adversarial review

2026-10-25. **BOUNDED SOURCE GO** for the pure new-family primitives and explicit private initialization/registration-writer slice at the exact pins below. No blocking defect found in this slice. **Full r378 family SOURCE NO-GO; production installed export UNAVAILABLE; release NO-GO; legacy CLOSED.** This review grants no merge, adoption, installation or activation authority. Parent keeps the candidate unmerged while completing subsequent slices; one final Astra review belongs after the complete implementation.

## Pins, authority and inspection

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-1b97856c` verified clean before and after execution. Exact direct-parent chain:

- `ff9a4d01eb53daad9ec8987b4b6d375f39ab1ae1` — r380 report.
- `c9b391f9e231e7f0b92b455fac6f226970b92d5a` — initialization/writer source.
- `c73e109272a3e27a038824ca72fb698a4e6578da` — r379 gap report.
- `3ac457d8532d6fe62764ae760b621a698e30e393` — pure family primitives.
- `1bec52ea8e1fedd8411099c0afc5001ff1c3031a` — required base.

No applicable AGENTS.md found in candidate descendants or its ancestor chain, or root review destination ancestors. Read the full root r378 normative contract, candidate r372 normative contract, r379 and r380 reports, complete changed source diff, new tests and relevant Core/persistence/recovery/initializer/writer context. Input SHA-256: r378 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`; r372 `a41dc92e7380e791478897300e42df98a8663b46876df04e4858f0ef6185f5c1`. Candidate diff check passed. Root already contained unrelated untracked evidence; it was left untouched. Historical memory supplied split-role cautions only; current source and execution control this decision.

Sole persistent write is this root report, mode 0600. No source edits, helper-script writes, commits, merge, delegation or self-adoption. The authorized tests/probes created only disposable inert fixture files through their existing fixture lifecycle.

## Boundary findings

No blocking findings for this bounded slice.

1. **Explicit family and exact identity.** `service/installed_export.py:34-62,125-158,237-240` keeps strict version 2, closed configuration, required converter/image/profile/principal inputs and separate singleton family selection. Legacy configuration/enrollment parsers remain closed. Enrollment revision stays 1. Canonical configuration, launch and host binding prevent numeric-tolerant configuration comparisons from conferring family authority. Context/config substitution with stale binding refuses. Credential presence, actual context resolver authentication and activation sealing remain later integration responsibilities; these pure validators do not confer them.
2. **Core gates exceed a creation-version argument.** `service/core.py:229-260,302-303,486-487` requires the checked internal expectation, verifies enrolled control-role identity, inspects a source-preserving private copy before owner mutation, rechecks the opened store before WAL/initializers and checks again after duplicate lookup before new admission. `_publication_creation=2` alone is still ordinary local creation, not installed-family authority. Expectation bytes are detached from caller mutation. Old installed/native creation choices retain explicit v1 rejection. This establishes the requested private Core seam, not a public activated owner.
3. **Exact SQL and fresh completion.** `service/installed_export.py:160-218` checks coherent live markers, complete canonical table/index/trigger identity through the existing trusted model, row affinity, foreign keys, canonical enrollment and registrations, scalar/row caps and finite SQL progress deadlines. Retained SQL text is compared, never executed. Fresh postcheck requires the exact fixed rows and otherwise empty tables; `fresh_namespace` separately rejects unexpected or populated media namespaces. `persistent_init.py:460-488` allows lost-reply completion only with matching private intent and exact fresh DB under control exclusion; partial DDL/enrollment is retained and refused. No migration or reconstruction path is added. The 160 MiB constant is not wired retained-history capacity.
4. **Writer is an INSERT boundary.** `service/install_child.py:98-205` validates explicit family/enrollment/evidence/images, inherited control-lock inode and actual exclusion, private DB/sidecars, exact v2 schema/enrollment and canonical prestate. It repeats validation under BEGIN IMMEDIATE and around insertion/commit. Target statements perform no DDL, construct no Core and access no receipts. Existing retry is exact; stale prestate refuses. Tests compare every non-registration table before/after and verify rollback. The old branch explicitly requires coherent v1. Changed-function legacy parser fixtures establish source refusal of new records; they are not an executed archived binary. The historical old writer's absent universal format check is not retroactively repaired.
5. **Finite bounds and regression scope.** SQL uses 2048 total rows and 16 MiB scalar accounting with progress handlers; trusted in-memory constraint reconstruction has its own finite deadline. Repeated validations install fresh per-phase deadlines, so no claim of a single two-second end-to-end writer limit is made. Configuration/converter envelope limits remain unchanged. Existing v1 split initialization and registration selections passed; local v2 default, native v1 activation gates, publication transaction and migration code were source-inspected, not operationally qualified or broadly regression-tested here.

## Independent execution

Inspected `/private/tmp/r380-tests.py` before use: its inserted path is exactly the candidate above. Read both candidate conftests before imports; their real listener/server fixture was not invoked. Ran **exactly once**:

`/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r380-tests.py`

CPython **3.14.7**. **30 methods PASS in 6.316 s**, comprising 13 ExportInitialization + 4 selected SplitInit + 3 selected Registration + all 10 pure InstalledExportContractTests. This is the independent result, separate from r380's attributed 6.498 s. Audit hook installed before application/test imports blocked real subprocess.Popen, os.system, socket.bind and socket.connect; forbidden attempts **[]**. No broad pytest collection or seven real-child publication tests ran.

One additional inline audited interpreter invocation ran **seven adversarial refusal cases**, all PASS:

- Configuration version `2.0` and `True` separately refuse `installed_export_family_required`.
- Foreign family token refuses `explicit_export_family_required`.
- Configured context descriptor revision substitution with retained old binding refuses `installed_bundle_binding`.
- 2049 tiny retained inventory-history rows refuse `export_scalar_budget` before detached constraint admission.
- Live user_version 3 refuses `publication_version_pair` before owner mutation; retained fixture bytes compare unchanged.
- Registration worker image differing from enrolled image refuses `export_registration_image_binding`; registration table remains empty.

Reproduction: initialize the existing ExportInitialization fixture with `run_init()`, use its `public()`, `identity()` and tiny `db()` helpers, apply each listed change, and invoke respectively `configuration`, `expectation`, `database`, checked-family `Core`, or `registration_writer` under the existing real control flock. The row probe inserts 2049 five-column inventory-history rows then removes only those disposable fixture rows; it does not inspect or migrate a real installation. No additional test framework was created. Additional invocation forbidden attempts **[]**.

Ownership/UID/GID/drop observations are mocked; SQLite, tiny files and flock exclusion are real. **Mocked UID drop is not kernel identity proof.** Converter bytes are inert. Native/B2 namespace negatives are fixtures, not actual operational deployments. Scalar byte-ceiling exhaustion and time-expiry were inspected in source, not separately exhausted at full size; row exhaustion was exercised. These limits constrain the evidence, not permission to broaden this campaign.

## Parent-owned remaining gates

Production launcher and dropped private entrypoint wiring, credential-principal corroboration, authenticated extended hello/seal, supervisor mappings and full frozen-profile checks, owner activation/AudioExports and post-owner brackets, explicit-pin image target and complete immutable closure, named-volume RO assets subpath and inode corroboration, dropped managed-reference descriptor path, truthful measured capability, all installed retained-history consumers and 160 MiB media-name accounting, public >1 MiB HTTP/CLI/MCP upload/export/download/reopen/replay, generated-asset export, failure/cancel/timeout/reap/ACK/capacity integration remain unwired/unavailable. Their absence is not a blocker to this expressly bounded slice and prevents any overall GO. Same-family private SQL reopen tested here must not be called completed production restart/history support.

No Docker/network/listener/application child, FFmpeg, ML/hardware, HELD/STT content, actual migration/install/download, heavy-gate waiver or actual pin acquisition. No real evidence cleanup. Actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification all **UNRUN**. CUDA and native Metal remain **mandatory, unqualified/UNRUN**. Seven real-child reference-publication tests remain **BLOCKED/UNRUN**. **Bounded slice SOURCE GO only; full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.**
