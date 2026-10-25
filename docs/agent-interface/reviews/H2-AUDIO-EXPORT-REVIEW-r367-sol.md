# r367 — independent adversarial managed audio export review

2026-10-25. **SOURCE NO-GO pending F1; product/release NO-GO; legacy CLOSED.**
One reproduced retained-provenance integrity defect blocks source adoption. The
closed export execution path otherwise has coherent bounded source evidence.
This review neither adopts the candidate nor qualifies any actual conversion,
installation, process containment, hardware or complete product vertical.

## Authority, exact pins and cleanliness

Reviewed candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-6c69f9c1`:

- HEAD/report: `23a682e7eb16e26fc11d121237870d0c577ca324`.
- Implementation: `fb1e8fe5473db70676aa25f794b9df33c7b3f214`.
- Base: `96a1b249c1b6a505e1d32c500adca10355115458`.

These are consecutive commits. HEAD minus implementation contains only the
236-line r366 report. Candidate status was empty before and after review;
`git diff --check 96a1b24 HEAD` passed. No applicable ancestor/nested AGENTS
was found. Read the complete candidate r366, candidate AUDIO-EXPORT.md, root
r365 and candidate PRODUCT.md, current operation/schema/validation contracts,
legacy coverage ledger, distribution entries and source diff/consumers.

Sole persistent write is this root-checkout report, mode 0600. No source edits,
commit, merge, delegation or self-adoption. Root's unrelated pre-existing
untracked evidence was observed as status filenames and left untouched.
Converter authority was decided by the **ORCHESTRATOR**, not clarified by the
user. r366 lines 18-21 misattribute that decision; this report corrects its
provenance without editing the candidate. No real converter pin was selected.

## F1 — high: public export provenance can diverge from frozen publication without quarantine

Candidate introduces public `artifact.audio_export` at
`contracts/v1/schema.json:793` and constructs it at `service/core.py:630`.
Publication initially binds it correctly in `contracts/validate.py:160-165`.
But the existing public reader `service/core.py:765-770` subsequently reads the
independent `artifacts.metadata` row and verifies only its referenced asset
bytes. `service/dispatcher.py:293-296` returns that row after checking job owner.
It does not corroborate the artifact with its producing frozen plan, retained
publication artifact/manifest, or publication intent.

The reconciliation adapter at `service/reconciliation.py:539-552` walks the
artifact and publication as separate schema-valid reference documents. The
cross-document helper `service/persistence.py:317-324` corroborates publication
intent against publications/job/context, but does not inspect the independently
returned `artifacts.metadata` copy. Thus adding export provenance to the schema
and successful initial publication is insufficient to preserve that provenance
on public reads/reopen.

**Actual inert repro on this exact HEAD, under the audit guard:**

1. Use the inspected `tests/service/test_audio_export.py` fixture to upload a
   small WAV, submit export and await injected successful publication.
2. Close Dispatcher. Open its private temporary Core on the current thread.
3. Read artifact A. Change ONLY
   `A['audio_export']['converter']['reported_build']='TAMPERED BUILD CLAIM'`.
   Execute `UPDATE artifacts SET metadata=? WHERE id=?` using `encode(A)`.
   Leave publications, manifest hash, frozen context, intent and audio untouched.
4. Run `Reconciliation(core, stage).reconcile_startup()`, close Core, reopen
   Dispatcher without a converter registration and GET `/v1/artifacts/A_ID`.

Observed exact receipt:

```text
reconcile True False []             # complete, blocked, reasons
publication unchanged True
HTTP 200 public build TAMPERED BUILD CLAIM
integrity_fault False
```

The earlier exploratory probe expected this mutation to block and failed its
assertion (`ARTIFACT MUTATION False []`); the separate public repro confirmed
rather than relabelled that failure. No actual worker/converter ran.

Impact: agents receive a plausible but false converter provenance claim for a
successful durable master after restart, despite contradictory retained trusted
copies. This is a generic pre-existing reader/corroboration weakness newly exposed
by export's new converter/settings provenance, not evidence that public callers
can execute arbitrary SQL. It violates this slice's retained transform provenance
and reconciliation integrity requirements.

**Required narrow correction:** corroborate the publicly returned artifact copy
against the retained publication/intent and frozen job/context using canonical
identity, both in retained-history reconciliation and public artifact reads;
refuse/quarantine divergence. Cover single-copy mutation of export converter,
settings/source provenance and missing export provenance without mutating all
trusted copies together. Preserve existing speech/prepare contracts. No new
framework, scheduler or generic command bus is needed. Independent fresh review
must assess the correction; this report does not authorize its implementation.

## Source audit observations

- Required owned asset/rate/channels/loudness request is closed; API is canonical,
  CLI adds `audio export`, MCP uses the existing typed operation transport.
  Principal/write scope and managed asset ownership remain enforced.
- `service/dispatcher.py:319-330` looks up duplicate execution before volatile
  activation/readiness/source resolution. Frozen input hash, changed-payload
  conflict, reopen replay and existing retry context fences remain in Core.
  `_input_assets` at `service/core.py:510-516` checks current owned metadata and
  bytes against frozen dependencies before allocation/publication.
- Dedicated export profile has no speech compatibility entries. Admission,
  reservations, queue, Authority and durable publication remain the existing
  mechanisms. Oversize refusal leaves no jobs/reservations in the tested case.
- Core's export branch uses complete RIFF/PCM24/header/frame/hash checks and
  operation-specific transform provenance. Existing mono PCM16 speech and prepare
  branches were retained, not globally relaxed. Reference-walker discriminated
  manifest branches and explicit source_asset_id reference are consistent with
  its closed traversal; clean two-export retained history reconciled successfully.
- Worker has closed FFmpeg argv, forced WAV, inherited no-follow source descriptor,
  private exclusive output, timeout/size/threads/resource ceilings and no new
  detached FFmpeg session. Supervisor independently allowlists converter by
  principal/profile, checks trusted image digest, executable hash and closure
  paths, and binds the managed reference root. Image immutability/corroboration
  remains deployment-owned; registration metadata alone is not such proof.
- Proposal is not death proof. Existing PID1 descendant drain, durable reap receipt,
  ACK, cancellation and quarantine/reservation mechanics remain authoritative.
  Injected reap receipts establish only exercised source control flow.
- Current installed profiles have no converter and explicit installed-family
  refusal. Discovery omits unavailable export and detail says
  unavailable_converter without converter/media probing. AudioExports and explicit
  RuntimeSupervisor arguments are source integration seams, not completed
  installed enrollment. No fake voice, second scheduler, alternate store,
  acquisition, UI, public STT or stitch work was introduced.
- Both distribution manifests include service/audio_export.py,
  worker/audio_export.py and transcript-audio_export.json. Contracts retain closed
  identities and operation declarations; legacy ledger marks export
  implemented_source_unqualified and retains the installed gate. Existing legacy
  command sources/default behavior are unchanged.

## Fresh coherent current evidence

Interpreter: `/private/tmp/pab-0c-venv/bin/python -I -B`, verified **CPython 3.14.7**.
A Python audit hook was installed before application/test imports and refused
`subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect` with
`FORBIDDEN_REAL_OPERATION:<event>`. Exact candidate root and module paths were
inserted explicitly; unittest loaded only the named modules. No broad pytest or
conftest activation. Inspected top-level/service conftests and test imports,
including inert provisioning fixture dependencies. The listener/uvicorn fixture
was not used. Fake backend log labels CPU/MPS/CUDA are not hardware execution.

**60 tests PASS in one current run, 15.927 seconds**, after both final tightenings:

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
```

Same guarded process directly called nine inspected contract functions from
`tests/contracts/test_contracts.py`, all PASS:
`test_schema_and_all_fixtures`,
`test_exact_text_ids_defaults_and_source_control_coverage`,
`test_revision_idempotency`, `test_lifecycle_cancel_fence_publication`,
`test_device_policy_and_readiness`, `test_tick_rounding_and_immutability`,
`test_migration_and_worker_boundaries`,
`test_operation_requests_jobs_artifacts_and_error_mapping`,
`test_binary_fixture_boundary`.

Generator checked twice in memory: intercept fixture wave writes, Path.write_text,
Path.read_bytes and mkdir; compare complete virtual directory filename/byte map
with disk, then compare second generation and generated-name set with first.
**PASS: 20 retained files, 19 generated; voice-catalogue.json retained unchanged;
zero fixture writes.** Initial harness incorrectly demanded the generator rewrite
every existing fixture and asserted on voice-catalogue.json; corrected the harness
to compare whole-directory before/after semantics, without changing source or
weakening the audit block. This was a harness assumption failure, not a generator
regression.

Additional inert consumer probe: export an exported PCM24 asset to a second master
at 24 kHz mono; **succeeded**. Close and reconcile the two-export retained history:
**healthy** before mutation. F1's two mutation observations are failures of the
candidate's integrity expectation, not passing negative tests. Tool outputs retain
the receipts; no runner/log files were written by this review.

r366's 60 PASS preceded its final two tightenings and its separate 8+8 PASS are
historical receipts; the fresh coherent run above supersedes that coverage gap.
Its earlier reference_publication selection (33 tests, 2 failures / 20 errors)
is not a pass. The two non-audit fixes are covered by allowed regressions; all
seven real-child publication methods remain **BLOCKED**, unexecuted here.

## Parent-owned gates and disposition

Fix F1 and correct r366's authority attribution before source adoption. Obtain
fresh Astra independent review after this review/correction; no delegation was
performed in this sole-write lane. Passing bounded tests do not close F1.

Actual immutable converter pin/acquisition, deployment image corroboration,
sealed registration, installed-family enrollment and real short FFmpeg
conversion/loudness/replay/process-descendant qualification remain **UNRUN**.
Linux/amd64, this Intel Mac Docker Desktop and OrbStack actual installation are
**UNRUN**. CUDA and native Metal remain mandatory and **unqualified/UNRUN**.
Full HTTP/CLI/MCP, CPU/CUDA/Metal and coherent clean-voice release obligations
remain; UI and public STT are deferred. This is not complete product acceptance.

No Docker/build/pull/load, network/listener, application child, actual FFmpeg,
ML/hardware, HELD/STT, operational cleanup, installation or real download was
executed. Temporary inert fixture files/SQLite/fake process outputs were confined
to their disposable test directories. Observed disk available: 5,314,904 KiB,
below the unchanged 6 GiB heavy gate; no waiver. **SOURCE NO-GO pending F1;
release NO-GO; legacy CLOSED.**
