# Native stage correction independent review — r291 Sol

2026-10-25. **NO-GO for complete native Lot A SOURCE adoption.** F9's stage-mode visibility race is corrected in the reviewed owner/executor arrangement. A separate **F10 (P2): private import-alias removal races ResourceStore enumeration**, remains material. The independent one-shot 87-method selection ran **86 PASS / 1 FAIL**; the same-root lifecycle method failed its first generation with `verification_failed`. A single additional deterministic inert probe proves a durable resource fault while actual native collect succeeds. No failing selection was repeated. Parent adoption remains pending; native packaging, actual TLS deployment, Metal and release remain unqualified.

## Authority and exact pins

Review only; sole lasting write is this report under `/Users/ben/Projects/py-audio-box`. No source changes, commit, delegation, pytest collection/maintenance/listener, actual socket/process/signal, ML/hardware, Docker/network/download/install, crash, operational cleanup or HELD action. Existing fixture teardown touched only disposable fixture resources. Root's pre-existing untracked STT evidence was listed only, not opened or modified. A memory-registry pass supplied general retention/evidence-boundary conventions, not technical authority; this explicit task authorizes the executions below.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-63bbaed6`, branch `t3/native-stage-visibility-race`, clean at entry and final verification:

- Parent/root: `0805c19530668430fb72ec67623283a4f77345aa`.
- Base: `bd85e1c4c41e1b1927d6f4cc2385041ac357158d`.
- Integration: `0c33ffc26ae2765a3be577c86f69b407f17614ed`.
- r286 documentation: `c1a4493786149acf1b0c0bed7560b9f3f08c5290`.
- F6–F8: `01e7325f06cd3eecc8ffeb06923c10818ede14cf`.
- r288 documentation: `8ec14dca611b1f814c714944d4ac283462af80fe`.
- F9 correction: `f368f31c1fdd7c158f31f38ad311bcbf942957a6`.
- Reviewed report HEAD: **`1277b8fe791ad1747934c1d7fb5fdbbd08234b09`**.

Read full r277, r285, r286, r287, r288, r289 and r290. Inspected complete NativeRuntime, Dispatcher and ResourceStore, their actual tests and owner/executor handoff; followed Authority, Core verification/publication, private import and replay, native registration/terminal decision, generated provenance, shared receipt and detached/install-policy consumers. No AGENTS.md found in repository discovery or checked candidate ancestors. Read both conftests before execution: the root test maintenance autouse hook and service HTTP process/listener fixture were neither collected nor invoked.

Disk available was 3,270,028 KiB, below 6 GiB. Only tiny fixture/archive bytes with injected capacity ran. Production native 6 GiB-plus-demand and RAM gates were unchanged. Printed CPU/CUDA/MPS labels are synthetic.

## F10 — P2: ordinary import-alias retirement durably faults private-spool monitoring

Material locations on reviewed HEAD:

- `service/admission.py:20–22` includes private spool as the third ResourceStore root.
- `service/native_runtime.py:388–395` creates private `import`, no-replace links it into the held stage FD, removes the alias, then syncs both directories.
- `service/resource_store.py:126–136` uses ordinary `os.walk` plus pathname `lstat` for this third root. The staging-specific FD scan/finite Pocket disappearance exception does not apply.
- `service/resource_store.py:69–85` latches and durably records observation failure.
- `service/dispatcher.py:404–407,511–534` continues monitoring while lifecycle I/O is pending and maps an already-latched resource fault to `verification_failed` during terminal processing.

A scan can enumerate `import`, then collect can remove that exact temporary alias before the scan's `lstat`. Actual `lstat` raises ENOENT; the monitor commits `fault` even though output import succeeded, with one complete 48-byte output, nlink=1, owned final-mode stage, timely receipt and valid proposal. This requires no hostile substitution, mode change, late result, hardware or invalid media. It is a normal trusted writer transition. Final-mode stage allocation cannot close this separate private-root race.

The fresh 87-method invocation failed `NativeLifecycleTests.test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job` at line 72, before reopen. An observational wrapper around Dispatcher._fail printed the active caught exception and original terminal locals, then delegated unchanged to the original method. No runtime/validator result was replaced:

```text
R291_VERIFICATION_TRACE ValueError('resource_violation')
service/dispatcher.py:512, _terminal:
    if getattr(self,'resource_fault',False): raise ValueError('resource_violation')
job_id=job-5099dfbab2fe45399a95a0b20cc5eb0c
attempt_id=attempt-783f57811b614f7392670011310b8f6b
staging_id=stage-5072151b3b7940fc8f541130e450d1a9
binding=0436f140d4c46a10128e12dd83272bc98280a2c7c204485941fd4a94c05d95e1
runtime_id=3ebc4bc8ad2fc7068896b37b0b423ea5/d4937246cac1b99ac3365eebff6e1fd3/1f538638d00d8d19e5d0b4443b21d086
original_deadline=2026-10-24T23:36:05.261339+00:00
timely=True, reaped=True, receiver_closed=True, model exit_code=0
local_import_uid=501
stage: mode=16888 (0770 directory), inode=1008341555, uid=501
output.wav: mode=33152 (0600 regular), inode=1008341565,
            uid=501, nlink=1, size=48
output_sha256=e39615880f20c22dfacc48c154b9ff7e2e190b12ccaf3070b01d02ce30e7c530
source_id=qwen-execution-sha256:6f6455eba990ea3e4e5f105b4b32c79f219a81b3f67bf18b097c81b5397c2388
final job: state=failed, stage=verification_failed, artifact_ids=[], reaped=True
Ran 87 tests in 12.246s; FAILED (failures=1)
```

This trace establishes the resource-fault branch, not the exact earlier exception that latched it: the wrapper did not read that fault row. The independent probe below proves the concrete alias race. It is a strong explanation for this failure class, but neither this first-generation trace nor the probe proves the exact cause of every historical r290 second-generation failure. Those historical failures remain unwaived; no green repeat is used to dismiss them.

## Single deterministic additional probe and retained raw state

One additional stdin interpreter, same isolated Python and explicit candidate paths. Actual NativeFixture runtime/supervisor/codecs/child-function/fake ML/owned handle completed a tiny output. Actual Core/DB/Admission/ResourceStore came from PocketResourceRaceTests' setup. The probe assigned native staging to that fixture's owned 0770 staging root and appended native private spool as the third monitor root, matching production Admission's root arrangement. It did not substitute scanner, collect, receipt or validator results. There were no reservations in this separate scanner fixture; the full lifecycle invocation independently demonstrates job failure.

The only scheduling hooks pause actual `Path.unlink` for the exact `import` alias and pause `os.walk` after it really enumerated that alias. All waits/join are bounded at three seconds. Scanner owner remains the setup thread; actual collect runs in one disposable thread. After enumeration the hook releases collect, waits for actual collect to finish, then returns the original unchanged file list to production ResourceStore. Its actual next `lstat` fails. No file is artificially removed by the scanner hook; native collect removes its own alias.

Reproduction core (after the setup described above and `rid=f.launch(); b.admission.check()`):

```python
alias = f.spool / f.a.binding / 'import'
real_unlink, real_walk = Path.unlink, os.walk
entered, release, done = threading.Event(), threading.Event(), threading.Event()
def unlink(path, *args, **kwargs):
    if path == alias:
        entered.set()
        assert release.wait(3)
    return real_unlink(path, *args, **kwargs)
def walk(root, *args, **kwargs):
    for directory, dirs, files in real_walk(root, *args, **kwargs):
        if Path(directory) == alias.parent and 'import' in files:
            release.set()
            assert done.wait(3)
        yield directory, dirs, files
def collect():
    try: collected.append(f.runtime.collect(f.a, rid))
    finally: done.set()
# Patch exact functions above; start collect thread; assert entered.wait(3).
# Owner calls b.admission.check(); record exception and DB rows before teardown.
# Finally release and join(3); assert thread is no longer alive.
```

Raw diagnostic output retained here, before normal disposable fixture teardown:

```text
PROBE_SCAN_EXCEPTION FileNotFoundError(2, 'No such file or directory')
service/admission.py:86 -> resource_store.py:70 -> admission.py:93
 -> resource_store.py:217 -> resource_store.py:133 -> Path.lstat
missing=/private/var/folders/gk/6rj578d95ls341449hb55pj80000gn/T/tmp0gifvajt/control-spool/c0390040caac697749fd75060891b838770669abe9ca28b51de33f2383d9f6be/import
LISTED files=['snapshot', 'complete', 'output', 'import']
COLLECT_ERRORS []
RAW_DB resource_monitor [(1, 'fault')]
RAW_DB resource_faults [("[Errno 2] No such file or directory: '/private/var/folders/gk/6rj578d95ls341449hb55pj80000gn/T/tmp0gifvajt/control-spool/c0390040caac697749fd75060891b838770669a",)]
RAW_DB resource_accounting [(1, 356087)]
RAW_DB resource_reservations []
PROBE_ASSERTIONS {'monitor': 'fault', 'scanner_failed': True,
 'alias_exists': False, 'output_bytes': 48, 'receipt_timely': True,
 'runtime_admission': True}
```

The fault reason is truncated by the actual production `str(exc)[:160]`; the full missing pathname above comes from the exception traceback. Probe receipt (unaltered wire value):

```json
{"binding":"c0390040caac697749fd75060891b838770669abe9ca28b51de33f2383d9f6be","boot":"1f1ec1be13c79b921dcd2c2e90a02cfe","configuration_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","deployment_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","lane":"ec9dce0c4ac4fc9c224ae1ac2b64553a","original_deadline":"2026-10-24T23:37:09.673487+00:00","processes":[{"exit_code":0,"pgid":123,"pid":123,"role":"model","spawn_token":"b1305e9be3467e1d113f9adc89696351","start_identity":null}],"reap_method":"darwin-owned-exec-v1","reaped":true,"receiver_closed":true,"runtime_id":"ec9dce0c4ac4fc9c224ae1ac2b64553a/1f1ec1be13c79b921dcd2c2e90a02cfe/c32a19a5ea457e9d2fb9599a5555cc4d","timely":true,"version":"native-receipt-v1"}
```

Private snapshot was 5,543 bytes and its exact binding was `c0390040caac697749fd75060891b838770669abe9ca28b51de33f2383d9f6be`; job `job-speech`, attempt `attempt-speech`, owner `worker1`, stage `stage-speech`, generation 1, input hash `a725fb34e8d0ec8dd70a5783b2788a0ddcf9d9b74b0ec7709d0c0b933a0b0564`. Full snapshot and proposal were emitted to the tool transcript before teardown. Complete marker's raw contents were `{"bytes":48,"sha256":"e39615880f20c22dfacc48c154b9ff7e2e190b12ccaf3070b01d02ce30e7c530"}`. Private output inode 1008342912 and final output inode 1008342914 each had uid501/nlink1/size48; stage inode1008342903 had mode0770. Both output files had these exact bytes:

```text
524946462800000057415645666d74201000000001000100c05d000080bb0000020010006461746104000000cc0c34f3
```

Required correction: make private temporary-name retirement coherent with actual owner resource observation, keeping conservative accounting, exact ownership/substitution/partial-import fences and no-replace publication. Add an actual private-root barrier regression covering this enumeration-to-stat removal. Do not suppress arbitrary ENOENT, disable monitoring, broaden the Pocket exception or delete retained uncertainty. This report does not prescribe or approve an unreviewed implementation.

## F9 correction and preserved integration

F9 closed in bounded SOURCE scope: `NativeRuntime.detach:175–185` authenticates clone bytes on the sole Core owner, then synchronously allocates the stage. `allocate_stage:187–203` validates root, exclusively creates, opens no-follow, fchmods the held directory FD to0770, checks owner/mode/FD-path identity and fsyncs stage and parent before Dispatcher `_prepare:355–360` submits executor work. A resource observation on that same owner cannot interleave the intermediate0700 mode. Collisions refuse; uncertain creation remains retained. No global production umask change or monitoring suspension exists.

Collect at `native_runtime.py:370–401` only verifies stage; it samples original identity, holds a no-follow FD, compares original/FD/path identities before and after import, links to the directory FD and fsyncs it. No executor mkdir/chmod remains. The three new correction methods passed, including actual ResourceStore0077 scan/import with chmod forbidden, conservative highwater, retained fsync uncertainty/collision and wrong-mode/substitution negatives. Existing four Pocket resource negatives/removal cases also passed. Their staging-only scope does not cover the private-spool F10 transition.

F6–F8 remain preserved by source inspection and the relevant fresh selected methods. NativeSupervisor.prepare extends only a pre-enrolled matching principal/context/profile template, rechecks typed execution/admission/inventory, and retains finite64-registration/64-record/eight-unacked/64MiB bounds. A refused prepare can consume retained registration capacity; no automatic eviction is claimed. No public self-enrollment operation was added. Connector accepts only a numeric absolute endpoint, carries one absolute RPC cutoff across TCP/explicit TLS handshake/framing, closes owned descriptors and retains TLS1.3/exact pin/CERT_REQUIRED. Strict Linux receipt shape including truthful fixture reaped_children=0 remains unchanged.

Core `_verified:491–551` still independently validates owner/nlink/media/hash/provenance/manifest/Authority, with trusted local UID rather than native UID. The private import source and final inode are separate copies, and transient link alias removal precedes collect return; F10 concerns observing that removal. Original output/complete bytes stay retained and exact local replay checks equality without new overwrite. Stage identity failures refuse rather than repairing. No Core/SQLite callback enters executor work.

Clock/source/receipt ordering was explicitly traced: Authority holds canonical detached frozen bytes; native and common generated_source hash the same typed execution identity, not mutable boot/time metadata. Child proposal and control validation use frozen context; receipt binds original boot/runtime/binding/deadline. Receiver close/exact wait and output validation precede the locked terminal timely decision; durable receipt precedes import/publication. First import checks cutoff; exact completed local replay remains historical; Dispatcher checks first-publication UTC cutoff after resource check, while an existing durable publication intent is handled first. In this fresh failure, resource_violation precedes the deadline and Core media/provenance checks. There is no evidence here of a source digest or clock discrepancy causing it. Historical second-job causes remain unproven.

Dispatcher stores validated receipt, performs resource verification and Core durable outcome, then reservation transition and ACK. Storage ambiguity retains reservation/no ACK until same-root replay. Shared live/detached receipt consumers select by frozen lane; Linux runtime/install-policy paths continue refusing native activation. F1–F5 supervisor implementation is unchanged by F9; its selected inert terminal/watchdog regressions pass. These observations do not prove exhaustive concurrency, real kernel identity, crash durability or actual TLS behavior.

## Independent executions and carried evidence

All project/test interpreters: `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd, explicit paths:

```python
sys.path[:0] = [
 '/Users/ben/.txcode/worktrees/py-audio-box/t3-63bbaed6',
 '/Users/ben/.txcode/worktrees/py-audio-box/t3-63bbaed6/tests/service',
]
```

1. **87 run, 86 PASS / 1 FAIL, 12.246s**, once: `tests.service.test_native_runtime`, `tests.service.test_native_lifecycle`, `tests.worker.test_native_supervisor`, `tests.worker.test_native_protocol`, `tests.worker.test_native_qwen.NativeQwenTests`, `tests.worker.test_engine_qwen`, `tests.service.test_pocket_preparation.PocketResourceRaceTests`. Direct unittest.main(module=None, argv=['unittest', *these separate dotted selections], verbosity=0), with the observational _fail wrapper described above. No source file was edited to instrument it.
2. **6 PASS, 0.392s**, separate fresh interpreter, once: `tests.worker.test_qwen_import_seam`, `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`. Fake upstream/root/legacy/extracted archive evidence only.
3. **One deterministic additional inert probe**, above; not counted as unittest passes. No failing test or suite rerun.

Fresh distinct selected coverage: **93 methods, 92 passing / one failing**. r290's reported172 passing methods remain historical, not this review's outcome.

The 47- and32-method regression selections are carried from r289's independent47 PASS/45.212s and32 PASS/20.717s, corroborated historically by r290's47/42.779s and32/18.189s. Checked diff from01e7325 to reviewed HEAD: production changes only NativeRuntime, test changes only test_native_runtime, plus documentation. Their Linux runtime, Core, Dispatcher, Admission/ResourceStore, fixtures and engine/receipt code are unchanged. These broader Linux selections do not exercise the native private alias and cannot resolve F10; no blind repetition was justified. Five direct dispatcher cases are likewise historical only. New native F10 requires corrected native/private-monitor coverage and fresh review, not unchanged Linux repetition.

`git diff --check bd85e1c..HEAD`: PASS; candidate remains clean. r277 SHA256 `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`; unchanged lib/tts.py SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known broader legacy digest RED is unwaived, unregenerated and unexecuted.

## Parent-owned gates

Correct F10, preserve F9/F6–F8 and obtain fresh independent review before complete SOURCE adoption. Keep earlier intermittent failures visible until their exact causes are established; this report supplies a concrete failing resource-monitor interleaving rather than claiming final green makes them disappear. No adoption occurred here.

Lot B native packaging/launcher/dependency closure, private configuration/credential enrollment and rotation, numeric endpoint/bridge/firewall topology and supported owned-version onboarding remain open. Actual TLS deployment and native installed activation are unqualified. Lot C actual Apple Silicon Metal probe/load/generate/cancel/restart, process/FD/resource/crash durability, acoustic/performance and release qualification remain open with unchanged headroom. Metal remains UNAVAILABLE / UNQUALIFIED. Nonempty clone conditioning and arbitrary cancellation-point readiness remain unresolved. CPU Docker and CUDA remain separate mandatory release rows. This partial actual-function/inert SOURCE evidence grants none of those authorities.
