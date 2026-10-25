# Native owner import uncertainty correction — r294 Sol

2026-10-25. Implementation candidate for fresh parent Sol review. F11's live
fsync-failure replay is closed in the focused inert checks; no self-adoption.
Reopen is deliberately conservative, including a visible COMPLETE whose final
fsync may not have returned. Historical intermittent causes remain unproven.
Native B/C, actual Metal, installed readiness and release remain CLOSED /
UNQUALIFIED.

## Authority and exact pins

Owned worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-f76594f2`, branch
`t3/latch-import-fsync-failure`. Entry clean at
`313991689993dd550fd948d50231357f41838e3e`: fresh bd85e1c integration plus review
documents, with requested correction sources still unmerged. Historical branch
`t3/fix-private-import-race` remains at
`1bbc63b887d2c941947c93615a690ea6f7d9d60e`; its worktree was not edited.

Requested cherry-picks, in order, with resulting local pins:

- `434e26747736e303e116bdde0b413282755b1f30` -> `b29384938e7a1ea1412af98644564716dd529dae`.
- `e187bf2061ec61a458aafee82d68472ef9097086` -> `54850fa472ac020c58dd0e2a821ee92509cdaa4f`.
- `9ed05f744ac3342dacb114e301e33fe4cfbc254b` -> `6d239c25d20748b84d29f9c85fc87c6a6e9d56f4`.
- `5ba3c66ace2c5853101424c44c73be863c924854` -> `61ffd2f60bbcb32980f4f708c6ed62de4190ed03`.
- `f368f31c1fdd7c158f31f38ad311bcbf942957a6` -> `f1b4eb6c7c41d496f97daf756662fa35152187f6`.
- `1277b8fe791ad1747934c1d7fb5fdbbd08234b09` -> `a10dc1a6486fea59c9c7a29d1ca0d74ee98b485c`.
- `2a8b8dacb1d82fc44b7b1cba1f0897c72465ee5f` -> `15b6d723a5d60325035743986dc534e7d4a1a625`.
- `1bbc63b887d2c941947c93615a690ea6f7d9d60e` -> `f4ebb15e28a574cfbbe18d3433f0c20eabcc4c92`.

Correction source: **`afa4aaa6aff9778963fddb6d9f46aecd50e328e5`**. This report is a
separate subsequent commit. Read full r293/r291/r292/r277 and relevant r285,
recovering combined-output truncation with bounded reads. Inspected actual
NativeRuntime/Dispatcher handoff, accounting, selected tests/conftests and Core
publication/receipt joins. No applicable AGENTS.md found in worktree discovery
or checked ancestors. Bounded memory-registry search returned no relevant hit;
no memory-derived technical authority. No delegation or independent review here.

## Change and conservative closed state

Production edits are only `service/native_runtime.py` and
`service/dispatcher.py`; focused tests are the two existing native service files.
No DDL, Core/public contract/oracle, ResourceStore, receipt, identity, deadline,
resource or cancellation policy was weakened.

NativeRuntime now keeps per-binding closed/completed import sets. Owner
finalization failure closes the binding and marks the runtime failed. Collect
storage errors do likewise; retained output validation/storage refusals cannot
later become a receipt-only handoff just because stop or deadline follows.
Ordinary collect, stale PendingImport finalization, start/input/output mutation
and ACK cannot regain authority for the closed binding. Observe and stop still
use the real native supervisor, retaining its exact process accounting. A
STOPPED observation remains truthful but cannot clear import quarantine or
provide an eligible handoff. Dispatcher separately retains quarantine on such
observations and guards terminal publication/release before consuming a closed
binding. Reservations and evidence remain retained; no automatic repair or
failure-to-success retry was introduced.

Two bounded private, immutable, stable O_EXCL files extend the existing spool:
`import-intent` and `import-complete`. Each contains the closed version
`native-import-v1`, exact Authority binding/runtime ID and already authenticated
output tuple. INTENT is written/file-fsynced/directory-fsynced before private
import creation and owner link/unlink. COMPLETE is written only after import
link/unlink, private-directory and held-stage-FD fsync, active checks, exact
output comparison and stage identity verification. Its own file and directory
fsync must return before the live completed set is advanced or an eligible
TransportObservation is returned. Marker failures never overwrite/retry a
visible marker. There is no marker rename/replace or retirement: the tested
post-creation directory-sync failure is the corresponding final marker
visibility uncertainty, rather than an os.replace seam.

A bare on-disk COMPLETE does **not** prove that its final fsync returned. On
reopen, INTENT-only, completed-but-unhanded-off, markerless output, and even
INTENT+COMPLETE all close admission/import replay initially. Partial/malformed,
conflicting, foreign-owner/mode/type/link or unknown-name files remain refused
or closed by the strict existing account reader. Every allowed filename is
accounted; worst-case reservation, count64 and 64MiB limits remain. Markers fit
within the existing metadata allowance.

One finite owner-startup hook reads the existing Core database, without writing
or repairing files. It recognizes only the exact stored Authority/runtime,
validated original timely native receipt, and matching existing Core publication
intent's job/generation/attempt/owner/staging identity. Such an intent can only
follow a prior successful finalized handoff in this source path. Normal Core
publication validation/recovery still performs its independent checks. Only
that subsequent durable Core evidence permits completed historical replay and
admission; legacy markerless completed history uses the same proof. A new-format
INTENT without COMPLETE is never restored by this hook. The failed live session
has no such reopened candidates and cannot self-recover.

This intentionally retains even a genuinely completed import if control died
before the durable Core publication intent. Filesystem visibility alone cannot
distinguish that case from a completion-marker fsync that raised. No operator
repair, eviction or new general recovery framework is supplied. Historical
valid Core-completed import replay and durable Core-intent recovery remain
distinct from incomplete first import.

F10 ordering is preserved: executor only creates stable private filenames; only
the serialized resource owner removes `import`, after executor completion and
between resource observations. No executor rename/unlink was added. Owner
finalization performs bounded filesystem work/read comparisons, no RPC, output
hash or PCM work; fsync has no hard real-time latency guarantee. Existing F9
stage ownership and F6–F8 authority/deadline/Linux receipt paths remain intact.

## Focused regressions and execution record

Both conftests were inspected first. All project/test interpreters were
`/private/tmp/pab-0c-venv/bin/python -I -B`, with explicit absolute sys.path
entries for this worktree and its `tests/service`; direct Dispatcher cases also
included its `tests`. Unittest used `module=None`, explicit selections,
`verbosity=1`, `failfast=True`. No pytest collection/maintenance hook or HTTP
listener fixture was invoked.

Seven added methods cover:

- Actual Dispatcher/Core/SQLite/Admission/ResourceStore/native-runtime journeys,
  separately injecting exactly one private-directory `_sync` failure and one
  exact held-stage-FD fsync failure after real link/unlink. Fault hooks delegate
  all other calls. Six additional automatic observe cycles, repeated explicit
  observe/stop/collect/ACK attempts and reopened control retain no artifacts,
  executing reservation and no wire ACK. Native STOPPED remains observable;
  runtime failed/binding uncertainty remains. Core has no publication intent at
  failure. These use fake TLS/ML/owned native handles, not replacement receipt,
  validator, scanner or publication results.
- Prelink failure, completion-write failure, and final completion-directory-sync
  failure (including a fully visible COMPLETE). Each retains identical evidence
  across five repeated observe/stop/collect/finalize/ACK attempts, blocks reopened
  admission/import, and emits no wire ACK.
- Actual output-file fsync failure before import retains partial output, blocks
  replay/reopen and leaves no stage output.
- Completed live replay succeeds, while completed-but-unhanded-off reopen stays
  closed without subsequent durable Core evidence.

The existing publication-storage-uncertainty positive now constructs a fresh
NativeRuntime at reopen: initially closed COMPLETE is recognized only from the
existing valid Core intent; publication completes, reservation transitions and
ACK follows. The existing completed same-root/native old-boot replay and next
job, seeded HTTP/CLI/MCP journeys, F9/F10 real scanner barriers and F1–F8
regressions also pass in the selected coverage. The stage-mode/substitution
negative uses separate fixtures because the first owner failure now correctly
latches that binding rather than authorizing a second finalization attempt.

Execution results, with interrupted runs retained rather than reported green:

1. First native selection stopped **14 PASS / 1 ERROR**, 15 methods /3.862s.
   Existing symlink negative expected OSError; the new conservative check raised
   Refused before reaching its filesystem check. Corrected validation order and
   latched collect OSError. This was an implementation error, not the historical
   second-job intermittent.
2. Next native selection stopped **26 PASS / 1 ERROR**, 27 methods /6.678s.
   All 26 runtime methods passed. A test-only edit incorrectly indexed historical
   PendingImport as a tuple; restored its original `.receipt` assertion. No
   production change for this error. These 26 passing runtime methods remain
   the runtime portion of the required selection.
3. Remaining native selection **70 PASS /14.671s**, once after that assertion
   correction: `tests.service.test_native_lifecycle`,
   `tests.worker.test_native_supervisor`, `tests.worker.test_native_protocol`,
   `tests.worker.test_native_qwen.NativeQwenTests`, `tests.worker.test_engine_qwen`,
   `tests.service.test_pocket_preparation.PocketResourceRaceTests`.
   With the preceding 26 runtime passes, this covers **89 original + seven new =
   96 distinct methods**. This was not one uninterrupted all-green invocation.
4. Fresh import/archive interpreter: **6 PASS /0.400s**,
   `tests.worker.test_qwen_import_seam` and
   `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`.
   Fake upstream/root/legacy/extracted archive bytes only; selected archive method
   has no subprocess.
5. Linux Dispatcher tuple/publication/cancel/history selection, once:
   **25 PASS /17.245s**: `tests.worker.test_engine_lifecycle.DispatcherEngineTests`,
   `tests.service.test_qwen_lifecycle.QwenLifecycleTests`,
   `tests.service.test_pocket_lifecycle.PocketLifecycleTests`,
   `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`.
   Existing SQLite ResourceWarnings remained visible.
6. Five direct Dispatcher cases **PASS**, once, separate TemporaryDirectory and
   `pair.__wrapped__` closed in finally for each: serialized cancel/unknown/retry;
   timely completion/lost ACK; late/stale `(False,False)` and `(True,True)`;
   confirmed-terminal replacement generation.
7. Final source inspection added the narrow retained-Refused guard described
   above and extended the existing bad-output negative through repeated stop.
   Only its affected branches were rerun: **3 PASS /1.111s**, exact methods
   `NativeRuntimeTests.test_receipt_linux_substitution_and_bad_output_never_import`,
   `test_stop_and_cancel_during_output_import_are_not_death_proof`, and
   `test_first_late_import_fails_but_completed_local_replay_does_not_expire`.
   Earlier complete selections precede this final narrow guard; they were not
   rerun or represented as a new all-green final full suite.

Distinct selected coverage: **127 unittest methods plus five direct cases**;
repetitions are not added to that count. No historical intermittent was retried,
waived or assigned a proved cause. One ancillary shell check accidentally used
zsh's reserved `path` variable; git/shasum were not found in that invocation.
Fresh-shell diff/hash checks subsequently succeeded; no project execution or
mutation resulted from that failed ancillary check.

Available disk at entry check: 3,204,456KiB, below6GiB; final 3,213,068KiB. Only
tiny fixtures/archive bytes with injected capacity ran. Production 6GiB-plus-
demand/RAM gates are unchanged. No actual socket/process/signal, ML/hardware,
Docker/network/download/install, listener, crash, operational cleanup or HELD
action. Existing fixture teardown touched only disposable fixture resources.
Printed CPU/CUDA/MPS labels are synthetic.

`git diff --check`: PASS. r277 SHA256 remains
`0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`;
lib/tts.py remains
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
Known broader legacy digest RED is unwaived, unregenerated and unexecuted.

## Parent-owned gates

Fresh independent Sol review of the exact source/report before adoption.
Historical intermittent causes and exhaustive concurrency/crash durability are
unproven. This is bounded SOURCE/inert evidence, not runtime, installed,
hardware, acoustic or release authority.

Lot B native packaging/enrollment/dependency closure, actual TLS deployment and
installed activation remain closed. Lot C actual Apple Silicon Metal
probe/load/generate/cancel/restart, process/FD/resource/durability,
acoustic/performance and release qualification remain closed. Actual Metal
remains UNAVAILABLE / UNQUALIFIED. Nonempty clone conditioning/arbitrary
cancellation readiness remain unresolved. CPU Docker and CUDA remain separate
mandatory release rows.
