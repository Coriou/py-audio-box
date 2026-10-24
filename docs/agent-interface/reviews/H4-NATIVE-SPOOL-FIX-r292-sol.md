# Native private spool correction — r292 Sol

2026-10-25. Implementation candidate complete for fresh parent Sol review;
no self-adoption. F10's ordinary private import-alias retirement now runs on the
resource owner after executor completion. Historical r290 intermittent failures
remain unproven as to exact cause. Native B/C remain CLOSED / UNQUALIFIED.

## Authority and pins

Owned worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-3309c0e6`, branch
`t3/fix-private-import-race`. Entry clean at
`cc5c740a55d7efa922ca515579d3440d25e770e2`: bd85e1c supervisor integration plus
review documents only. Historical `t3-63bbaed6` was not modified. Requested
cherry-picks applied in order:

- `434e26747736e303e116bdde0b413282755b1f30` -> `329d7cb`.
- `e187bf2061ec61a458aafee82d68472ef9097086` -> `e021dc4`.
- `9ed05f744ac3342dacb114e301e33fe4cfbc254b` -> `d5699dd`.
- `5ba3c66ace2c5853101424c44c73be863c924854` -> `4ee3a5c`.
- `f368f31c1fdd7c158f31f38ad311bcbf942957a6` -> `a1822ec`.
- `1277b8fe791ad1747934c1d7fb5fdbbd08234b09` -> `0d1c55c`.

Correction source commit: **`2a8b8dacb1d82fc44b7b1cba1f0897c72465ee5f`**.
This report is a separate subsequent commit. Read full r291, r289, r290 and
normative r277, using bounded reads to recover truncated combined output.
No applicable AGENTS.md found. A bounded memory registry search had no relevant
hit; no memory-derived technical authority was used. No delegation occurred.
Both conftests were inspected before test execution; no pytest collection,
maintenance hook or HTTP listener fixture ran.

## Correction and complete control-spool writer audit

Production changes are limited to NativeRuntime and Dispatcher. ResourceStore
is unchanged: no ENOENT suppression, Pocket exception extension, monitor bypass,
scan lock or uncertainty deletion.

NativeRuntime executor collect still authenticates the receipt, output tuple,
proposal, full hash/PCM/oracle relations, downloads bounded bytes, fsyncs them,
and creates a separate O_EXCL private `import` copy. It now returns a detached
PendingImport, not a tuple containing an eligible TransportObservation. That
object contains validated bytes/proposal/receipt and the private import's exact
file identity. No Core/DB object or callback enters executor work.

Dispatcher transfers adapter ownership on owner startup, including reopen, and
invokes finalize_collect only after the collect future is done and before
completing that command or entering terminal verification. Its resource check
at the start of the tick finishes before finalization; terminal publication
resource observation follows finalization. Thus the sole owner cannot enumerate
the private name and then retire it before that same scan's lstat. Other runtime
adapters retain their existing collect tuple path without either optional hook.

Owner finalization verifies the held stage FD/path identity, owner and 0770 mode,
checks the prevalidated private inode identity, performs no-replace link,
unlinks only that alias, fsyncs private directory and held stage FD, and checks
cancellation/deadline locally before/after the new import. It performs no RPC,
download, hash or PCM validation. Exact file comparison reads are capped at
1 MiB. Filesystem calls including fsync are finite operations, not a claimed
hard real-time storage latency bound. It returns TransportObservation only after
these fences. O_EXCL copy creation remains executor-side; final stage nlink=1
and trusted local UID checks remain mandatory. Stage mode/substitution,
symlink/conflicting output and uncertain partial-import conditions still refuse.

The complete NativeRuntime control-spool mutation graph is small:

- prepare exclusively creates a binding directory and creates `snapshot` and
  optional `input` directly at their stable names through _write. No inputpart
  rename or input-name retirement exists in this implementation.
- collect exclusively creates `output` directly at its stable name, streams and
  validates it, then creates `complete` directly at its stable name. No
  outputpart rename or temporary complete-marker retirement exists. Failed or
  interrupted files stay retained. Other scans can conservatively see growing
  files; prior highwater and full reservations do not decrease.
- collect creates `import` directly and leaves it present across executor return.
  Only owner finalization retires that name. There are no other NativeRuntime
  unlink, rename, replace, rmdir or automatic spool cleanup operations.
- stage allocation remains the F9 owner operation before executor submission;
  output linking now also belongs to that owner. Core's later publication is
  already owner-owned. Native supervisor journal/conditioning is a separate
  native-host root, outside the control ResourceStore third root.

Restart _account still charges declared worst case and rejects/blocks partial
output, retained import, conflicts, unknown names, wrong ownership/mode/type or
hardlinks. A finalization failure retains evidence and the dispatcher unknown
phase/reservation rather than retrying as new work. No automatic recovery or
capacity release was added. Receipt validation, terminal durable outcome,
publication intent/storage uncertainty, reservation transition and ACK ordering
remain the existing terminal path. F6-F8 supervisor authorization, client cutoff
and strict Linux receipt corrections are unchanged.

## Regressions and execution record

New actual private-root regression uses NativeFixture's actual runtime, codecs,
supervisor, child inference function with fake ML/channel/owned handle, and
PocketResourceRaceTests' actual Core/SQLite/Admission/ResourceStore. Native spool
is the third scan root. A hook pauses after actual import creation; real os.walk
then enumerates `import`, releases executor collect, and waits for its actual
completion before returning the original file list. The alias remains present
and final stage output remains absent through that enumeration-to-lstat window.
After the scan completes, actual owner finalization retires the alias; another
actual DB-backed scan stays healthy with no fault rows and nondecreasing
highwater. All scheduling waits/join are bounded at three seconds. No scanner,
collect, validator or receipt result is fabricated.

A second new method checks nonowner finalization refusal, retained import causing
restart admission closure, cancellation between executor and owner finalization,
no owner network call, retained alias and absent output. The stage substitution
negative now retains one PendingImport across wrong-mode refusal and actual
stage-open substitution, so it still reaches the intended identity fence.
Existing F9 0077 stage barrier remains. Existing required-root/removal,
permission/symlink and Pocket resource negatives are unchanged and selected.
Existing native partial/conflict/unknown restart checks remain. No new
permission or arbitrary-missing-name exception exists.

All test interpreters were `/private/tmp/pab-0c-venv/bin/python -I -B`, with exact
absolute sys.path entries for this worktree and its `tests/service`; the direct
Dispatcher interpreter additionally used its `tests`. No pytest invocation.
Available disk: 3,248,908 KiB, below 6 GiB. Only tiny inert fixture/archive bytes
ran, with existing injected capacity; production disk/RAM gates were unchanged.
No actual socket/process/signal, ML/hardware, Docker, network/download/install,
listener, crash, operational cleanup or HELD action. Existing fixture teardown
only handled disposable fixture resources. Printed device labels are synthetic.

Final selected checks:

1. **89 PASS / 13.665s**, final source, once: `tests.service.test_native_runtime`,
   `tests.service.test_native_lifecycle`, `tests.worker.test_native_supervisor`,
   `tests.worker.test_native_protocol`, `tests.worker.test_native_qwen.NativeQwenTests`,
   `tests.worker.test_engine_qwen`,
   `tests.service.test_pocket_preparation.PocketResourceRaceTests`.
   This is the original 87 plus two new methods. Includes full seeded public
   HTTP/CLI/MCP generation/job/artifact and completed same-root reopen plus next
   job, each selected once in this final invocation. The historical completed
   receipt probe now reads the detached pending receipt without claiming a
   second off-owner artifact import.
2. **6 PASS / 0.370s**, fresh interpreter, final source:
   `tests.worker.test_qwen_import_seam` and
   `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`.
3. **25 PASS / 18.763s**: `tests.worker.test_engine_lifecycle.DispatcherEngineTests`,
   `tests.service.test_qwen_lifecycle.QwenLifecycleTests`,
   `tests.service.test_pocket_lifecycle.PocketLifecycleTests`,
   `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`.
   These affected Linux Dispatcher tuple/publication/cancel/history seams ran
   after finalization handoff, before the final native-only attach_owner startup
   hook. Linux adapters expose neither new hook. Existing SQLite ResourceWarnings
   were visible, not suppressed. No blind unchanged 47/32 selections were run.
4. **Five direct Dispatcher cases PASS**, final source: serialized cancel/unknown/
   retry; timely completion/lost ACK; late proposal `(False,False)`; corrupt
   proposal `(True,True)`; confirmed-terminal replacement generation. Each used
   a separate TemporaryDirectory and pair.__wrapped__ closed in finally.

There were no failing invocations in this implementation turn. Earlier
implementation checks are retained distinctly: initial 88 PASS /14.434s before
adding owner cancellation/restart coverage and removing owner stop-RPC possibility;
then 89 PASS /13.540s before explicit owner attachment on reopen. Six import/archive
checks also passed /0.404s before that last startup hook; the final fresh six above
covers final archive bytes. Repetition followed source/coverage changes; no failed
run was rerun away. Final distinct selected coverage is 120 unittest methods plus
five direct cases, not the sum of repeated invocations.

`git diff --check`: PASS. r277 SHA256 remains
`0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`;
lib/tts.py remains
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
Known broader legacy digest RED is unwaived, unregenerated and unexecuted.

## Parent-owned gates

Fresh independent Sol review of exact source and report before adoption. This
fix addresses a proved ordinary writer/observer interleaving; it does not prove
the exact causes of every historical r290 intermittent second-generation failure,
exhaustive concurrency safety, real kernel identity or crash durability.

Lot B native packaging/enrollment/dependency closure, actual TLS deployment and
installed activation remain closed. Lot C actual Apple Silicon Metal probe,
load/generate/cancel/restart, process/FD/resource/durability, acoustic/performance
and release qualification remain closed. Actual Metal remains UNAVAILABLE /
UNQUALIFIED. Nonempty clone conditioning and arbitrary cancellation readiness
remain unresolved. CPU Docker and CUDA are separate mandatory release rows.
