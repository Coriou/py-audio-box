# Native retained owner restart — r309 Sol implementation

2026-10-25. SOURCE candidate for independent review; no self-adoption. Supports
zero-job close/reopen and clean completed native publication histories under the
same immutable configured boot/trust. No installed public service, hardware,
Metal or release qualification is claimed.

## Pins and scope

Isolated worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-6ebb26e1`, branch
`t3/native-owner-restart`, clean at entry. Parent/base:
`0375aae7437536156b2aa4f1a5bfeb04cb7f53da`.
Initial source/docs/tests: `9409178dfb4ee4fec25f679a3f4db82145e9f4a1`.
Final SOURCE, including pre-initialization schema ordering correction:
`ee682f633ab81f091d3d5f2d2082cce8050f2793`.
This report is a subsequent documentation-only commit.

Read PRODUCT.md, NATIVE-PACKAGE.md, normative r277, r296/r303/r305/r306/r307/r308;
inspected actual package/activation/runtime/Core/Dispatcher/entrypoint,
recovery/reconciliation, publication, Authority, admission and fixture seams.
No applicable AGENTS.md discovered. Memory registry supplied evidence-boundary
conventions only; technical conclusions use current source. No root-checkout
writes, merge, push or subdelegation.

Final SHA256:

- service/native_activation.py: `9c13dc03a6818d3d0590880f98431fd04a9fe3f2e7a074c6c676c5b1cdafa673`
- service/native_package.py: `879106b73fe379de93e5f069fc4c56648ac8cb70c2b3d5d396134eb37dbed152`
- service/core.py: `13ef9dca789b82b68d3798af76fdfe3eed07c285b05654af28d11bef35a4a5ee`
- service/dispatcher.py: `ffe39b41ccea9ebc057b4e8426c866f1990ed3d1f1ade9dacd837d7a7b7ef20a`
- tests/release/test_native_history.py: `349c894495eb184ed57428b10460b2d52af0b950a336dd35797068841597d194`
- Unchanged lib/tts.py: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
  Known legacy digest RED remains unchanged/unwaived; no regeneration or digest test.

## Product behavior and boundaries

NativeActivation consumes the existing exact canonical durable enrollment on the
Dispatcher control writer. Its private preparation mode permits constructing a
runtime with pending completed imports, while admission remains closed. Ordinary
enrolled_runtime/native_startup_inputs consumers retain their refusal by default.
No pending-import flag grants history, scheduling or readiness authority.

Core receives only an exact-type private NativeActivation hook. After lifetime
flock acquisition and opening its sole SQLite connection, before WAL configuration
or schema/catalogue initialization, it invokes the native retained validator.
This prevents Core's ordinary additive initialization from concealing a missing
entire catalogue. Fresh initialization still requires the prechecked empty root;
a competitor initialized in the acquisition gap refuses. Native postcheck repeats
canonical enrollment/trust/owner/record binding around exact configured hello.
No boot learning or renewal is added.

The native-only reader compares exact canonical SQLite schema/index/trigger DDL,
checks finite row/byte budgets and SQL constraints via the existing detached
in-memory validator, and reuses closed contract documents, immutable Authority,
terminal_receipt, publication_record/publication_intent and oracle.publication.
The shared detached SQL constraint helper receives empty Linux enrollment tables
only in its memory model after actual native schema equality is established; no
Linux enrollment/history acceptance is invoked or weakened. recovery.py and its
explicit native refusal route are unchanged.

Accepted completed jobs must be succeeded, single-attempt, non-clone native jobs,
with exact configured five-field native identity/boot, reaped attempt, confirmed
outcome/ACK and retained reservation. Job/Authority/context/runtime/receipt joins,
frozen execution reservation configuration, exact artifact/publication/intent,
actual media hashes and retained staging bytes must agree. Current authenticated
supervisor collect must return the exact retained native receipt, proposal and
output metadata. Exact private snapshot/output/complete/import-intent/import-complete
membership and identity are required. Host launch count must equal completed jobs;
zero-job owner requires zero launches. Healthy hello and prior/current clean
inventory are required. Unknown namespaces and prior unresolved diagnostics refuse.
Only after all history proofs does the reader clear in-memory import closure;
it writes no history, deletes nothing, repairs nothing and sends no ACK.

Core/SQLite never cross into TLS/executor callbacks. The synchronous validator
uses detached Authority RPC and keeps SQLite on its control owner thread; no SQL
transaction is held while calling TLS. Existing Dispatcher admission, scheduling,
publication, reservation, stop/ACK and cleanup ownership remain authoritative.
No scheduler, store, DDL, readiness promotion or universal recovery bypass is added.
Existing distribution manifests already include all changed modules.

Supported history is deliberately finite: fewer than 64 jobs, at most 2048 total
SQL rows and 16 MiB retained scalar bytes, plus existing runtime/inventory limits.
Failed/cancelled/retried/clone histories, unfinished work/commands, uncertain or
legacy imports without exact intent/completion, maintenance and Linux-enrolled
roots remain unsupported. Changed-boot/trust renewal is separate. Dedicated-account,
immutable operator-managed input/no hostile same-UID writer premise still applies;
repeated snapshots are not atomic hostile-writer/ABA isolation.

## Execution receipts

Inspected tests/conftest.py and tests/service/conftest.py. Direct unittest avoids
pytest guards and socket/uvicorn fixtures. All selected native helpers use tiny
filesystem metadata/audio fixtures and injected TLS/process/backend observations;
no actual child or ML execution. Available disk was 2,990,640 KiB, below 6 GiB;
only explicitly permitted tiny inert source work occurred. No heavy ACTUAL gate,
network/listener, Docker, install/download, crash, HELD/STT or operational cleanup.

Both invocations used `/private/tmp/pab-0c-venv/bin/python -I -B -`, explicit absolute
worktree and worktree/tests/service sys.path entries, asserted Python 3.14.7,
`defaultTestLoader.loadTestsFromNames` and `TextTestRunner(verbosity=2,failfast=True)`.
Observed Python 3.14.7 / Clang 17.0.0, not target Python 3.11.

Initial finite selection on source 9409178: **37 PASS / 18.553s / exit 0**.
Tool session 44088, chunks 8f8720/a21f4f/600fc8 retain verbose output. Exact selection:

- tests.release.test_native_activation.NativeActivationTests (8)
- tests.release.test_native_history.NativeHistoryTests (15 at that pin)
- tests.release.test_native_enrollment.NativeEnrollmentTests (10)
- tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt
- tests.service.test_installed_activation.ActivationTests.test_post_core_private_reader_under_lifetime_lock
- tests.service.test_installed_activation.ActivationTests.test_changed_post_acquire_hello_refuses_and_releases_core
- tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change

Final source inspection then identified the missing-entire-catalogue initialization
ordering issue described above. Corrected it and added a regression; no failed-test
retry or broad unchanged rerun. The final focused selection ran once after those
changes: **6 PASS / 3.581s / exit 0**, session 34366, chunks cbbd43/c6c49e:

- NativeHistoryTests.test_missing_entire_catalogue_refuses_before_core_can_recreate_it
- NativeHistoryTests.test_completed_job_reopens_twice_and_retains_exact_artifact
- NativeActivationTests.test_owner_thread_first_start_reconcile_close_and_clean_restart
- NativeActivationTests.test_gap_competitor_initialization_refuses_after_acquisition
- NativeActivationTests.test_gap_trust_change_refuses_and_closes_core
- tests.service.test_installed_activation.ActivationTests.test_post_core_private_reader_under_lifetime_lock

Native classes above are in tests.release.test_native_history and
 tests.release.test_native_activation respectively. The final regression verifies
missing catalogue tables remain missing after refusal. Completed-job success
reopens twice, returns the retained exact job/artifact, retains reservation, and
observes one inference fixture call total. Refusal tests cover malformed job,
schema/table/index, Authority binding, native receipt substitution, remote receipt,
publication, import intent/completion, unknown staging, changed boot, missing trust,
unfinished reservation and unconfirmed ACK. They assert private evidence and ACK
count remain unchanged. Enrollment regressions cover canonical primitive-type,
identity, trust/config/template mutations and malformed records. `git diff --check`
passed; recovery.py/reconciliation.py/native_runtime.py remain unchanged.

## Remaining parent-owned gates

Independent source review and adoption are still required. This is private owner
restart functionality, not authenticated installed public lifespan/credential or
Docker/native topology/mount completion, fresh owned voice onboarding or renewal.
No reviewed arm64 wheelhouse exists. Actual Darwin install/import/ABI, TLS/topology,
process/FD/death/resource, hardware/model/MPS, speech/audio/performance and clone
lifetime qualification remain unrun. Actual Metal remains UNAVAILABLE / UNQUALIFIED;
CPU and CUDA remain mandatory separate release rows. Source/inert receipts do not
waive any actual gate or the legacy digest RED.
