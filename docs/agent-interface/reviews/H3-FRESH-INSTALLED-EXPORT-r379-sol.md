# r379 — incomplete fresh-installed export implementation

2026-10-25. **SOURCE NO-GO / INCOMPLETE.** This candidate does not implement
the full r378 contract and must not be adopted as installed export enablement.
Installed usability is not established. Release NO-GO; legacy CLOSED.

## Authority and exact pins

Isolated checkout: `/Users/ben/.txcode/worktrees/py-audio-box/t3-03c7234c`.
Initial HEAD was exactly `1bec52ea8e1fedd8411099c0afc5001ff1c3031a`, with empty
`git status --short`. No applicable AGENTS.md was found in the checkout or the
checked ancestor chain. No delegates, root-project writes, merge, push or
self-adoption. The implementation is deliberately returned incomplete under
the brief's precise-gap exit, not represented as a completed source gate.

Read FULL root r378, r372 and r377 before edits. SHA-256 of those inputs:

- r378: `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`.
- r372: `a41dc92e7380e791478897300e42df98a8663b46876df04e4858f0ef6185f5c1`.
- r377: `c3e007a76a5c3d2e5ba4e51fab50e515248f63080d84ebdaa6a0d3b7efb417c5`.

Source candidate commit: `3ac457d8532d6fe62764ae760b621a698e30e393`, directly
on the required base. This report is a subsequent documentation-only commit;
the containing commit is the report pin, not a different source candidate.

## Actual changes

Exactly three source-candidate files changed:

- `service/installed_export.py`: 115 lines, one pure new-family contract module.
  Defines the four exact new discriminators, strict integer version 2, 16/160 MiB
  constants; closed <=48,000-byte configuration with mandatory closed export
  object; existing <=4096-byte converter parser and exact worker-image binding;
  CPU/networkless export profile and sorted unique configured principals;
  detached combined profiles; canonical expected seal and strict seal matching;
  no-execution local converter hash/immutable declared-closure validation.
- `tests/service/test_installed_export.py`: 163 lines, ten injected unit methods
  for those primitives. Temporary inert executable/library bytes are never run.
- `headless-dist-manifest.json`: includes the new source module. Existing Docker
  release ignore rules already include service Python sources.

The new module is **not called by any production entrypoint**. A local projection
uses the unchanged v1 Kokoro field validator, but does not persist or return a
v1 configuration. Existing v1 parsers/defaults are untouched and still reject
the new closed shape. There is no public flag or export activation shortcut.
The expected seal function is not an authenticated hello or a qualification
receipt. Fixture pins remain explicitly inert/unqualified.

These primitives alone do not validate credential presence, finished-image
packaging, complete runtime closure, live RO mounts, enrollment or publication
DDL. Closure descriptors are inspected and closed; this does not independently
provide continuous namespace immutability. The intended checked immutable image
and live Docker brackets have not been wired. Do not infer operational security
or readiness from this module's successful return.

## Focused execution evidence and failure record

Interpreter: `/private/tmp/pab-0c-venv/bin/python -I -B`, verified CPython 3.14.7.
Read both `tests/conftest.py` and `tests/service/conftest.py` before execution.
No pytest collection: those conftests include a real-listener/server fixture,
which was not invoked. Used unittest with this exact checkout inserted in
`sys.path` and selected only
`tests.service.test_installed_export.InstalledExportContractTests`.

Before importing any application/test module, installed a `sys.addaudithook`
which raises on `subprocess.Popen`, `os.system`, `socket.bind` and
`socket.connect`, recording every attempted forbidden event. Imported only
stdlib sys before installing it; unittest/application imports followed it.

- First invocation: **10 methods, FAILED (8 subtest errors), 0.258 s**. Missing
  converter fields and invalid topology correctly refused via jsonschema's
  `ValidationError`; tests incorrectly expected only ValueError. No acceptance
  defect was observed in this invocation. Forbidden-event attempts: `[]`.
- Corrected those test expectations to explicitly include `ValidationError`.
  Second invocation: **10 methods PASS, 0.239 s**. Forbidden-event attempts: `[]`.
- Source staged `git diff --cached --check`: PASS before source commit.

No broad unchanged-test campaign, old-binary fixture execution, API journey,
SQLite installed initialization or real application child ran. These ten tests
cannot substitute for any of r378's five integrated finite exits.

## Binding five-condition gap matrix

| r378 finite exit | Current result | Concrete remaining work |
| --- | --- | --- |
| 1. Fresh exact v2, same-intent completion, same-family reopen; old v1 invariants | NOT IMPLEMENTED / NOT TESTED | `persistent_init._split_core:232` still selects publication 1. Add trusted family expectation through pre-Core reads/Core/postcheck; exact v2 enrollment/DDL creation and partial-state refusal/completion. No migration. |
| 2. New dropped registration writer, old writer boundary, brackets | NOT IMPLEMENTED / NOT TESTED | `install_child.registration_writer:98` is unchanged and has no universal publication-format check. Add exact v2/enrollment/prestate validation under transaction, retaining no-DDL/no-Core/no-receipt boundary. Test exact predecessor parser rejection and all bracket changes. |
| 3. Seal/mount/closure/pin/activation capability refusals | PRIMITIVES ONLY / EXIT UNSATISFIED | Wire private family selection, config/credential readers, extended hello, independent supervisor maps/full frozen profile, Docker inspection and role/assets cross-view, owner AudioExports activation. Add exact RO `core/assets` named-volume subpath/noCopy mount and dropped managed-reference helper. Add explicit-pin build target/refusal wiring; none exists yet. |
| 4. Installed injected upload -> >1 MiB public export/download/reopen/replay and generated asset | NOT IMPLEMENTED / NOT RUN | Dispatcher still has all three installed export blanket gates (`259`, `327`, `352`); owner constructs no installed AudioExports. Implement exact checked-family gates with replay first, then the ~1.1 MiB installed ASGI/CLI/MCP journey. Existing local tests are not installed evidence. |
| 5. Failure/reap/ACK/capacity plus retained joins and accounting | NOT IMPLEMENTED / NOT TESTED | `recovery._installed_detached_source:3007` still requires P1=1; scanner at `3036` remains v1/16 MiB. Thread trusted exact new-family 160 MiB media-name/16 MiB scalar accounting through every installed history consumer, canonical export joins and finite deadlines; exercise cancel/timeout/invalid output/unknown reap and boundary counters. |

Additional concrete unchanged seams: `daemon_volumes.command` still passes the
v1 worker policy; `worker.runtime_supervisor.main:537` accepts only existing v1
policies and does not register export mappings; `verify_export:335` still directly
opens private assets as root instead of the required dropped helper and compares
profile ID rather than the full frozen profile. `host_install._registration_hello`
and InstalledActivation still use the old hello shape. No selector or marker-only
workaround has been added. New constants alone authorize none of these paths.

This is an implementation shortfall, **not** a request for another design round
or an assertion that missing actual pins prevent source wiring. r378 already
resolves the choices and explicitly permits injected missing-pin refusal work.
Full implementation and its finite integrated fixtures remain required before
the parent commissions fresh Sol adversarial then Astra source review. This
agent did not commission those reviews or self-accept the partial candidate.

## Operational limits

No Docker/build/pull/load/network/listener/application child, FFmpeg, ML,
hardware, install/download, actual-root migration, HELD/STT access or operational
cleanup. Only disposable tiny fixture files were created/removed by unittest.
No disk refresh, RAM claim, heavy 6 GiB gate waiver or real pin acquisition.
Actual pins/closure completeness/qualification **UNRUN**. Linux CPU, Intel Mac
Docker Desktop and OrbStack **UNRUN**. CUDA and native Metal remain mandatory,
unqualified/**UNRUN**. All seven real-child reference-publication tests remain
**BLOCKED / UNRUN**. Existing frozen-input/Authority/reap/publication/ACK fences
were not changed. **SOURCE NO-GO; release NO-GO; legacy CLOSED.**
