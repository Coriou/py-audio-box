# r382 — bounded installed-export worker seal candidate

2026-10-25. Executable worker subslice candidate, awaiting fresh independent Sol
review. **No self-acceptance. Full r378 SOURCE NO-GO; production installed export
UNAVAILABLE; release NO-GO; legacy CLOSED.**

## Authority and pins

Own isolated checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-e320873b`
started clean at `1bec52ea8e1fedd8411099c0afc5001ff1c3031a`. Verified ancestry and
fast-forwarded ONLY this checkout to
`ff9a4d01eb53daad9ec8987b4b6d375f39ab1ae1`. Source worktree
`/Users/ben/.txcode/worktrees/py-audio-box/t3-1b97856c` was read only.

Source candidate **`8749e75519f41d9052978ed43a7127e637006c2f`**, direct child of
that ff9a4d0 pin. This report is a subsequent documentation commit. No root or
original source writes, parent adoption, merge beyond the authorized own-worktree
fast-forward, push, delegation or review commissioning. No applicable AGENTS.md
was found in the worktree/ancestor inspection.

Read full root r378 contract and r381 review, and candidate r379/r380 reports.
Root reports are absent from the candidate/source worktree and were read at
`/Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/`:

- r378 SHA-256 `17123fc564f5b6dfb2010e61d22279edd0b8dbd83d800fabc9c2fd016203e9da`.
- r381 SHA-256 `7fbbe8988b37fa414b09b3a0c5f91d7649c3f2fd0943b168eddf2c933a6e9e9a`.

## Implemented boundary

`worker/runtime_supervisor.py` now accepts the explicit private installed policy
`docker-kokoro-export-split-v2`, closed configuration file and mandatory separate
`--expected-worker-image`. No operational image default exists. The expected
image is trusted startup authority to be supplied by later Docker brackets; it
is not a claim that the process has inspected its own Docker image. Configuration
uses the existing pure new-family validator; before exposing any hello, startup
checks the expected image and actual immutable executable/hash/declared closure
with `installed_export.verify_local_converter`. Existing Kokoro bundle checks
run through a detached legacy field projection, without changing the v1 parser.
The projection is not persisted as a legacy installation.

The constructor independently detaches the checked configuration, derives the
fixed principal/reference mapping and converter image from it, and refuses
parallel caller converter/image/reference mappings or installed Pocket selection.
It never obtains authority from public AudioExports. The existing local export
mapping path remains unchanged. Worker hello adds mandatory `export_seal` only
for this explicit family, using the existing canonical six-field seal, including
strict integer publication format 2. Old hello construction remains unchanged.
An unqualified configuration can yield an assertion of its inputs; this does not
make export ready. New work also requires qualified/installed/valid/ready,
measured profile status and a nonempty receipt, independently in the worker.
Injected receipt strings in tests are not operational qualification.

At each existing prepare/start/source-acquisition call, new-family verification
compares canonical converter registration and the **entire frozen profile**
(id, config revision, device, topology, all limits, compatibility) against the
configured profile snapshot, plus configured principal membership. Numeric
representation substitutions such as integer 2 to float 2.0 cannot pass this
comparison. The hello profile digest covers the full configured profile,
including readiness/qualification inputs. Local converter immutability/hash is
rechecked before source acquisition. No public request image claim is trusted.

New-family source acquisition uses existing `worker.managed_reference.acquire`:
UID10001 helper, fixed `/references/assets`, one O_RDONLY FD, conclusive helper
wait before return. Parent rechecks FD owner/group, mode 0600, regular type,
single link and access mode through `check_fd`, then positive size <=24 MiB,
complete bounded PCM/hash metadata and stable file identity/times/size across
reading. It rewinds and returns the retained FD to the existing isolated child
handoff. Existing runtime_child FD plumbing, process ownership, descendant reap,
publication proposal, ACK/capacity fences and scheduler are unchanged. Any acquire
failure quarantines the supervisor; the existing helper retains an unreaped
process handle on unknown wait. No DAC override or permission relaxation exists.

Necessary transport touches are limited to `service/host_install.py` and
`service/supervisor_runtime.py`: exact explicitly selected new-family hello/seal
validation in the pure host parser, lifecycle adapter and dropped helper reader.
The old parser still rejects an extended hello; no optional key was added to it.
No daemon/helper command caller selects this family yet. These seams are needed
to consume the new authenticated contract later, and confer no owner activation.
No extra module, image recipe, mount or distribution-manifest change was needed.

## Focused inert evidence

Read `tests/conftest.py` and `tests/service/conftest.py`; there is no worker
conftest. No pytest collection or server fixture was invoked. Exact runner:

`/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r382-tests.py`

Runner inserts only this checkout and changes cwd to this exact checkout.
Before application/test imports it imports only sys, installs an audit hook
raising on subprocess.Popen, os.system, socket.bind and socket.connect, then
imports unittest/tests. CPython **3.14.7**. Final retained local log
`/private/tmp/r382-final-tests.log`: **42 methods PASS in 0.655 s; forbidden-event
attempts `[]`**. Runner/log are scratch evidence, not distributed executables.

- 12 new ExportSealTests: explicit startup/image/hash/immutable closure;
  exact new/old hello, seal fields, strict float/bool version/closed-key refusals,
  quarantine readiness refusal; full frozen profile (including float numeric
  substitution), principal, caller converter/image mismatches before helper;
  unqualified work refusal; verified readonly/rewound FD and bad source hash;
  helper uncertainty quarantine; explicit main selection/constructor handoff;
  transport matching seal and detached config; parallel caller-map refusal;
  closure recheck; actual acquire implementation with fake Popen and unknown
  waits, retained fake process, descriptor closure and UID10001 arguments.
- Existing AudioExportTests and ExportWorkerBoundaryTests (8 methods): local
  converter/PCM/closed limits, fake process/child authority and staging behavior,
  supervisor independent old mapping refusal.
- Existing ManagedReferenceTests (5): private RO FD, metadata/path/link/symlink
  refusals, exact dropped-helper arguments/reap, failure FD closure.
- Existing ConditioningSupervisorTests (7): existing reaper/uncertainty,
  housekeeping/terminalization and watchdog fences.
- Existing InstalledExportContractTests (10): closed new inputs, legacy parser,
  strict versions/seals, immutable executable/closure/hash/alias refusals.

The first development run had one fixture error: host temporary-file group did
not equal the injected owner expected by check_fd. Corrected only the fixture's
UID/GID observation. It ran 38 methods, one error, forbidden attempts `[]`.
Next run passed 40 methods; after adding constructor/closure negatives the final
42-method run above passed. A preliminary shell edit invocation found no `python`
command; all actual edit-script/test interpreter calls used the exact interpreter
path above. `git diff --check` passed before source commit.

Tiny files, pipes/socketpair descriptors and fake process/UID observations are
inert source evidence. No real subprocess or UID drop occurs. The existing
fake converter test writes PCM bytes without FFmpeg. This is not Docker image,
DAC, kernel cross-UID, installed launch, engine, hardware or operational evidence.
No broad unchanged regression campaign or seven real-child publication tests.

## Remaining parent-owned scope

The daemon selector and exact launch arguments, actual image pin acquisition and
complete closure review, explicit-pin image target, named-volume RO assets
subpath and assets inode cross-view, inspected image/config/seal/boot brackets,
credential corroboration, owner activation/AudioExports, truthful public
capability, retained history/readers and 160 MiB accounting, public >1 MiB
HTTP/CLI/MCP upload/export/download/reopen/replay and generated-asset journey,
integrated cancel/timeout/output/reap/ACK/capacity cases remain later work.
Existing daemon launches and owner activation still do not enable this family.
This slice does not complete r378 or any complete installed product journey.

A fresh separate Sol adversarial review of this exact candidate is parent-owned;
full Astra review follows the completed family later. No review was delegated or
self-adopted here. No Docker/network/listener/application child, FFmpeg/ML,
hardware, HELD/STT, actual migration/install/download, heavy waiver, actual pin
acquisition or operational resource cleanup. All actual Linux CPU, Intel Mac
Docker Desktop and OrbStack qualification **UNRUN**. CUDA and native Metal remain
**mandatory, unqualified/UNRUN**. Seven real-child reference-publication tests
remain **BLOCKED/UNRUN**. **Full-family SOURCE NO-GO; release NO-GO; legacy CLOSED.**
