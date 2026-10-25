# r366 — managed PCM24 audio export implementation candidate

2026-10-25. **SOURCE CANDIDATE; parent adoption/release NO-GO; legacy CLOSED.**
Executable closed worker logic and public HTTP/CLI/MCP job lifecycle are implemented.
Only inert source evidence was authorized and obtained. This is not a conversion,
installation, hardware, independent review or release acceptance report.

## Authority and pins

- Worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-6c69f9c1`.
- Branch: `t3/managed-pcm24-export`.
- Starting parent verified exactly: `96a1b249c1b6a505e1d32c500adca10355115458`; initially clean.
- Source candidate commit: `fb1e8fe5473db70676aa25f794b9df33c7b3f214`.
- Read the root r365 brief in full before implementation. Checked ancestor AGENTS
  through the worktree chain, root project/Projects paths, and nested worktree
  AGENTS inventory; none applies here. No root checkout file writes, merge, push,
  self-adoption or delegation. Independent Sol/Astra review remains parent-owned.
- User clarified converter authority during implementation: immutable converter
  artifact bound to trusted worker image digest, exact executable path/hash,
  reported build/version and runtime closure from the same image. No invented
  approved pin, acquisition, build, converter execution or installed-family claim.
- Legacy `apps/audio-export/audio-export.py` inspected and unchanged. Its PCM24,
  resampling/channel conversion and optional single-pass loudnorm semantics are
  represented by closed argv, rather than exposing its path-based CLI.

## Implemented behavior and fences

All four request fields are required: managed `asset_id`, enumerated rate/channels,
finite bounded loudness or null. One idempotency key and production-write scope.
HTTP remains canonical; CLI and MCP reuse existing typed HTTP operation transport.

Owner resolves owned/hash-verified WAV metadata; freezes source, settings, distinct
converter identity and bounded profile into the ordinary input hash. No fake voice,
model or generated-speech identity. Replay precedes readiness/source resolution;
changed payload conflicts, including after reopen. Existing retry validates current
trusted registration without replacing the old frozen plan.

Dedicated `audio-export-*` CPU profile snapshot has no speech compatibility entries.
Existing admission/reservation/queue/Authority/ACK/reap/publication are reused.
No new scheduler, command bus, DB worker or artifact store. Core independently
verifies complete classic/extensible PCM24 RIFF, output hash, rate/channels,
nonzero duration within one frame of the requested resampling ratio, provenance
and attempt bindings. Existing mono PCM16 speech/prepare policies remain separate.
The artifact read includes source hash/settings/converter metadata; output assets
and CLI download use existing read/hash APIs. Source bytes remain unchanged.

Worker validates an independent supervisor converter allowlist and trusted image
digest, no-follow executable hash/closure files, and exact owned managed source
mapping. The existing parent opens the source descriptor and passes it to the
existing supervised child. FFmpeg uses only closed descriptors/argv, forced WAV
input, bounded filters/threads/time/file size, and private attempt `output.wav`.
It does not start a detached session. Converter failure retains staging and yields
no proposal. A proposal alone releases no capacity: PID1 must drain descendants,
produce the exact durable receipt, and Core must verify/publicly commit it. Unknown
ownership remains quarantined and charged; acknowledgement is not proof of death.

Bounds: managed integer PCM WAV inputs <=24 MiB / 60 seconds; output <=36 MiB,
rate <=96 kHz, <=2 channels, PCM24; full registration canonical JSON <=4096 bytes;
context <=48000 bytes; current supervisor deadline <=25 seconds. Output duration
has a one-frame rounding tolerance. Loudness is a single-pass transform and does
not certify delivery LUFS. Arbitrary paths, executable selectors, filter text,
URLs, filenames, voice substitution and provisioning are absent from public input.

## Installed availability and finite parent prerequisites

`AudioExports` and the explicit RuntimeSupervisor converter arguments are trusted
source integration seams. They are not an alternate enrollment API. Existing
installed entrypoints provide no converter registration and owner admission
explicitly refuses their export. Discovery reads metadata only, omits unavailable
export, and operation detail says `unavailable_converter`; no version process or
media/converter probing occurs during discovery.

The Kokoro worker recipe currently apt-installs unversioned FFmpeg; the Pocket
worker recipe does not install it. Neither is an admitted export artifact. No
recipe or existing installed identity was silently changed. An executable SHA on
an arbitrary dynamic bundle is insufficient; runtime closure belongs to the
registered immutable image. Actual image identity must be corroborated by the
parent's trusted deployment/enrollment mechanism, not supplied by a public caller
or inferred by this module from an untrusted plan.

Before actual public installation: parent selects/captures the real immutable
converter image/artifact, seals and registers the exact converter/runtime closure,
reviews and enrolls a narrow installed-family extension, and qualifies the actual
short FFmpeg journey plus process containment. This is a finite, explicit remaining
integration gate; the source candidate deliberately cannot enable existing installed
families merely by constructing an HTTP payload. No substantial unrelated engine
or persistence contract redesign was introduced.

Actual Linux/amd64 install: **UNRUN**. This Intel Mac Docker Desktop: **UNRUN**.
OrbStack: **UNRUN**. CUDA and native Metal: mandatory, **UNRUN/unqualified**.
Actual FFmpeg conversion/loudness/audio journey: **UNRUN**. Fresh independent Sol
review and Astra review: **UNRUN / parent-owned**. No operational cleanup, Docker,
build/pull/load, network/listener, application child, FFmpeg, ML/hardware, HELD,
STT, install, download or provisioning execution was performed. Disk observation:
5,327,712 KiB available, below the unchanged 6 GiB heavy gate; no waiver.

## Verification performed

Interpreter throughout: `/private/tmp/pab-0c-venv/bin/python -I -B` (Python 3.14.7).
An audit hook refused real `subprocess.Popen`, `os.system`, `socket.bind` and
`socket.connect`. Tests were loaded by explicit paths through unittest, without
pytest collection/conftest activation. Read both service and top-level conftests:
the service fixture would create a listener/uvicorn child and was not used.

Final selected regression set: **60 tests PASS** (15.852 seconds):

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

Command: `/private/tmp/pab-0c-venv/bin/python -I -B /tmp/pab-r366-check.py` followed
by those exact paths in that order. Raw final log: `/tmp/pab-r366-tests-final.log`.
After two final focused changes (bounded no-follow owner admission read and
registration metadata byte cap), the affected service export set passed **8 tests**
(4.667 s), and worker export set passed **8 tests** (0.134 s). Logs:
`/tmp/pab-r366-admission-final.log`, `/tmp/pab-r366-worker-final.log`.

Nine explicit existing contract functions PASS: `test_schema_and_all_fixtures`,
`test_exact_text_ids_defaults_and_source_control_coverage`,
`test_revision_idempotency`, `test_lifecycle_cancel_fence_publication`,
`test_device_policy_and_readiness`, `test_tick_rounding_and_immutability`,
`test_migration_and_worker_boundaries`,
`test_operation_requests_jobs_artifacts_and_error_mapping`,
`test_binary_fixture_boundary`, from `tests/contracts/test_contracts.py`.
Called directly under the same audit hook; no broad pytest. Full fixture directory
filename-plus-byte equality before/after an in-process `contracts.generate.generate()`
passed. `git diff --check` passed.

The export tests cover real local SQLite/ASGI upload, durable receipt/wait,
artifact/format/hash and CLI verified download, unchanged source, reopen key replay,
changed payload conflict, absent converter, invalid/missing/foreign inputs, scope,
oversize atomic refusal, mutation after admission, converter failure, malformed
PCM24, stale proposal, cancel/reap, unknown ownership/reservation refusal, and
HTTP/CLI/MCP parity against the same IDs. Worker tests exercise the executable
argv/descriptor logic with an injected process, authenticated child protocol with
inert transport, and independent registration refusal. Injected bytes/reap receipts
are **not evidence of conversion, descendant containment or installation**.

Earlier selection attempted `tests/service/test_reference_publication.py`:
audit guard blocked its real synthetic child launches with raw error
`RuntimeError: FORBIDDEN_REAL_OPERATION:subprocess.Popen`. That 33-test selection
reported 2 failures / 20 error instances; it is NOT a passing qualification.
The two non-audit failures (discovery expectation and schema reference traversal)
were fixed and their allowed regressions passed. The seven child-dependent
publication test methods remain **BLOCKED under this lane**, not passed. No retry
with weaker audit or real process execution occurred. Speech/prepare publication
is covered here by the allowed in-process production API and injected engine
lifecycle regressions. Actual runtime publication/containment remains separate.

The schema shape was changed to explicit discriminated result-manifest branches
because the existing bounded reference walker does not treat an object plus
conditional allOf as ordinary object properties. Its narrow source_asset_id
reference location was added explicitly. Existing voice persistence/reopen checks
pass; unknown reference kinds still refuse rather than being guessed.

## Exact source candidate files

```text
clients/cli.py
contracts/fixtures/v1/operation-requests.json
contracts/fixtures/v1/transcript-audio_export.json
contracts/generate.py
contracts/v1/legacy-controls.json
contracts/v1/operations.json
contracts/v1/schema.json
contracts/validate.py
docs/agent-interface/AUDIO-EXPORT.md
headless-dist-manifest.json
native-dist-manifest.json
service/admission.py
service/api.py
service/audio_export.py
service/catalogue.py
service/core.py
service/dispatch_store.py
service/dispatcher.py
service/domain.py
service/production.py
service/reconciliation.py
tests/clients/test_production_clients.py
tests/service/test_audio_export.py
tests/worker/test_audio_export.py
worker/audio_export.py
worker/runtime_child.py
worker/runtime_protocol.py
worker/runtime_supervisor.py
```

This report is the only subsequent report-commit file. Public journey:
`docs/agent-interface/AUDIO-EXPORT.md`. Finite coverage status is in
`contracts/v1/legacy-controls.json`; legacy command hashes/behavior are unchanged.
Both distribution manifests name the actual new service/worker modules and fixture.
No changes to errors.json were needed: existing stable error codes/statuses suffice;
new operation errors are explicitly listed in operations.json.

## Audit runner retained for reproduction

The temporary runner used for the exact-path checks is reproduced below. For
contract function checks, the same setup/audit prefix was used before directly
calling the nine names listed above and in-process regeneration. This runner is
not a product CLI or a worker/scheduler.

```python
import sys
ROOT='/Users/ben/.txcode/worktrees/py-audio-box/t3-6c69f9c1'
sys.path.insert(0,ROOT)
import os
os.chdir(ROOT)
def audit(event,args):
    if event in ('subprocess.Popen','os.system','socket.bind','socket.connect'):
        raise RuntimeError('FORBIDDEN_REAL_OPERATION:'+event)
sys.addaudithook(audit)
if sys.argv[1]=='generate':
    from contracts.generate import generate
    generate()
else:
    import importlib.util,unittest
    suite=unittest.TestSuite()
    for i,path in enumerate(sys.argv[1:]):
        sys.path.insert(0,str(__import__('pathlib').Path(path).resolve().parent))
        spec=importlib.util.spec_from_file_location('r366_test_'+str(i),path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
```
