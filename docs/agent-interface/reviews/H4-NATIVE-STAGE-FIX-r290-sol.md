# Native stage visibility correction — r290 Sol

2026-10-25. F9 implementation candidate complete for fresh parent Sol review.
No self-acceptance or adoption. Final selected invocations pass; earlier intermittent
same-root second-job verification failures remain unexplained and require review.
Native packaging B and actual Metal C remain CLOSED / UNQUALIFIED.

## Authority and exact pins

Owned worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-63bbaed6`, branch
`t3/native-stage-visibility-race`. Entry clean at
`0805c19` (root bd85e1c plus review-document foundation). Historical
`t3-fa09f5b2` was not changed. Requested cherry-picks, in order:

- `434e26747736e303e116bdde0b413282755b1f30` → `0c33ffc`.
- `e187bf2061ec61a458aafee82d68472ef9097086` → `c1a4493`.
- `9ed05f744ac3342dacb114e301e33fe4cfbc254b` → `01e7325`.
- `5ba3c66ace2c5853101424c44c73be863c924854` →
  `8ec14dca611b1f814c714944d4ac283462af80fe`.

Correction source: **`f368f31c1fdd7c158f31f38ad311bcbf942957a6`**.
This report is a separate subsequent commit. Read r289, r287, r288 and normative
r277. Inspected both test conftests before execution. No applicable AGENTS.md
found. Bounded memory registry search returned no relevant hit; no memory-derived
technical authority was used. No delegation occurred.

## Implementation

Only `service/native_runtime.py` and `tests/service/test_native_runtime.py` change.
Existing Dispatcher `_prepare` invokes `detach` on its sole owner loop before
submitting asynchronous prepare. Native detach now authenticates clone input and
calls `allocate_stage` there. That method validates the staging root, exclusively
creates the exact stage, opens it no-follow, sets final 0770 via its FD, validates
owner/mode and FD/path identity, fsyncs the directory and parent, and only then
returns. Owner resource observation cannot run concurrently with this owner
operation. No Core/SQLite object or callback moves into the executor.

Collisions refuse rather than adopting or overwriting an existing directory.
Creation/fsync uncertainty retains the directory; a later allocation attempt at
that name refuses. There is no repair, deletion or ambient-umask assumption.
A restart in an incomplete pre-submit window may therefore conservatively retain
uncertainty; this is not automatic stage recovery authority.

Executor collect no longer mkdir/chmods a stage. It requires the precreated owned
0770 directory, samples its identity before no-follow open, holds the FD through
import, and compares original/FD/path identities before and after import. The
no-replace output link targets that directory FD; directory fsync uses that FD.
Exact existing-byte replay still works after the active cutoff, while new import
checks cancellation/deadline before linking and after syncing. Trusted local UID,
original native receipt, durable Core publication-intent replay, reservation/reap
and ACK ordering remain existing code. No ResourceStore suppression, Pocket
exception widening, monitoring suspension, Core publication bypass, DDL, protocol,
DNS/deadline or framework change was made. F6–F8 corrections are retained.

The new deterministic 0077 regression uses an actual ResourceStore with real
Core/DB/accounting from the tiny Pocket resource fixture. Native owner detach
observes the restrictive 0700 intermediate mode at exact-stage fchmod, completes
0770 before scanning, then actual NativeRuntime collect runs at the scanner's
exact stage-open hook, after scanner stat and before real open. Chmod/fchmod are
forbidden during that scan/import. Actual native codecs, supervisor, fake child
ML/owned handle and output import run. Monitor stays healthy, fault table empty,
highwater nondecreasing and measured staging includes output bytes. Test-only
umask is restored in finally; production never changes process umask.

Two further tests cover retained owner fsync uncertainty/collision/no overwrite
and executor wrong-mode/substitution refusal. Existing stage-symlink refusal,
output conflicts and Pocket namespace fault cases remain. Runtime fixtures now
explicitly perform the owner stage allocation before executor lifecycle work;
cancel/late assertions expect a retained empty stage rather than no stage.

## Executed evidence

All project/test runs used `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate
cwd, and explicit absolute candidate plus candidate `/tests/service` sys.path
entries. No pytest collection, maintenance hook, listener, actual socket/process/
signal, ML/hardware, Docker, network/download/install, crash, operational cleanup
or HELD action. Fixture teardown only touched disposable fixture resources.
Available disk was 3,265,188 KiB, below 6 GiB; only tiny fixtures/archive bytes
with injected capacity ran. Production headroom gates were unchanged. Printed
CPU/CUDA/MPS loader labels are synthetic.

Command shape (each dotted selection is a separate argv item):

```python
import sys, unittest
sys.path[:0] = [
    '/Users/ben/.txcode/worktrees/py-audio-box/t3-63bbaed6',
    '/Users/ben/.txcode/worktrees/py-audio-box/t3-63bbaed6/tests/service',
]
unittest.main(module=None, argv=['unittest', *selections], verbosity=0)
```

Final invocations:

1. **87 PASS, 14.074 s**, final source, without diagnostic wrappers:
   `tests.service.test_native_runtime`, `tests.service.test_native_lifecycle`,
   `tests.worker.test_native_supervisor`, `tests.worker.test_native_protocol`,
   `tests.worker.test_native_qwen.NativeQwenTests`, `tests.worker.test_engine_qwen`,
   `tests.service.test_pocket_preparation.PocketResourceRaceTests`.
   This is original native 80 + three new methods + four existing resource race
   methods. Resource negatives include missing root, symlink, unexpected removal,
   permission, nonempty conditioning, substitution and foreign-stage mode.
2. **47 PASS, 42.779 s**:
   `tests.service.test_pocket_preparation.PocketPreparationTests`,
   `tests.service.test_pocket_preparation.PocketPreparationInventoryTests`,
   `tests.service.test_pocket_preparation.PocketPublicJourneyTests`,
   `tests.service.test_installed_pocket.InstalledPocketJourney`,
   `tests.service.test_installed_activation`,
   `tests.service.test_installed_restart.TerminalRestart`,
   `tests.worker.test_engine_lifecycle.DispatcherEngineTests`.
3. **32 PASS, 18.189 s**:
   `tests.service.test_qwen_lifecycle.QwenLifecycleTests`,
   `tests.service.test_pocket_lifecycle.PocketLifecycleTests`,
   `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`,
   `tests.service.test_production_reconciliation`, plus
   `tests.worker.test_engine_kokoro.EngineTests` methods
   `test_exact_arguments_and_detached_identity`,
   `test_closed_descriptor_and_unsupported_controls`, `test_limits_close_backend`,
   `test_mutated_invalid_audio`, `test_mutated_numeric_metadata_refuses`,
   `test_bundle_binding_before_backend` (fully qualified selections).
4. **6 PASS, 0.465 s**, fresh interpreter:
   `tests.worker.test_qwen_import_seam`,
   `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`.

Total final selected coverage: **172 distinct passing methods**. Selections 2–4
ran before the final narrow additional original-stage-identity comparison; final
87 exercised that change. SQLite ResourceWarnings and expected injected_close_unknown
diagnostic were not suppressed. No five direct dispatcher cases were rerun because
Dispatcher/DispatchStore were unchanged.

Earlier executions are not erased by final green: initial 83 ran 82 pass/one
failure in second generation after completed same-root reopen (`verification_failed`).
A targeted diagnostic rerun passed (one test, 1.969 s). An intermediate 87 had
that same failure plus one new regression fixture error: its staging root had
wrong mode. A subsequent 20-method diagnostic selection had only that fixture
error and passed the reopen method. The fixture now explicitly establishes owned
0770 staging before its 0077 experiment. A diagnostic 87 then passed in 13.617 s;
its only `_fail` trace was the expected ordinary failure path, no verification
exception. Final original-stage FD comparison prompted the final uninstrumented
87 above. The intermittent prior second-job failures were not explained or
claimed corrected by the fixture repair. Parent review must retain this limitation
and investigate if it recurs; no complete race-free claim follows from final green.

`git diff --check`: PASS. r277 SHA256 remains
`0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`;
`lib/tts.py` remains
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
Known broader legacy digest RED remains unwaived, unregenerated and unexecuted.

## Parent gates and limits

Fresh independent Sol review of exact source and this report before adoption,
including the unexplained earlier reopen failures. No actual TLS deployment,
installed activation, native packaging/enrollment B, actual Apple Silicon Metal C,
process/FD/crash durability, acoustic/performance or release authority is granted.
Native Metal remains UNAVAILABLE / UNQUALIFIED. Nonempty clone conditioning and
arbitrary cancellation-point readiness remain unresolved. CPU Docker and CUDA
remain separate mandatory release rows.
