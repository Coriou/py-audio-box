# Independent native first-start owner review — r308 Sol

2026-10-25. **Bounded GO for parent consideration/adoption of r307 SOURCE within the explicitly authorized finite EMPTY-HISTORY ONLY alternative. No material finding remains in that scope.** Independent final selection: **18 PASS / 7.287s, exit 0**, once. This does not accept full native history, installed public service activation, Metal readiness or release. No self-adoption.

## Authority and exact pins

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-59e53153`, branch `t3/native-startup-activation`. Verified clean at entry and after execution. Exact base `5a734b15302084fceed9efcfdfad87dde3f7bfd2`; SOURCE `68581f8470d20061383780aac5e75316a1f7fda7`; report HEAD resolves `b1c3b87` to `b1c3b8741b18b83a686506253d387625dcb63271`. The only two commits after base are SOURCE followed by the r307 report. `git diff --check base..HEAD` passed.

Verified SHA256:

- `service/native_activation.py`: `b6cf067d9af2c621de5f4039959771bc86a6c781db63ba6846b9184f21622a32`.
- `service/native_package.py`: `2acee59ab9a92a8aa708932b4ef4cef55685cfa44c0319a3b5e07a760885fc2e`.
- `tests/release/test_native_activation.py`: `a31b6c1a2644fb7135077a0e693bb549c7a0c5a13832b51fac094f60ff9ab6ca`.
- Unchanged `lib/tts.py`: `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known legacy digest RED remains unwaived/unmodified; no regeneration or legacy digest test.

Read PRODUCT.md, NATIVE-PACKAGE.md, normative r277 and r296/r303/r305/r306/r307 reports. Inspected the complete SOURCE diff, new activation/tests, actual entrypoint/Core/Dispatcher seams, enrollment snapshots/context construction, NativeRuntime transport/import ownership, recovery native refusal and reconciliation receipt consumers. No applicable AGENTS.md found in candidate discovery or checked ancestors. Memory supplied evidence-boundary conventions only; technical conclusions use the pinned candidate. Sole lasting write is this root-checkout report. No candidate edits, commits, merge, push, delegation or cleanup.

## Findings and adversarial assessment

**No material SOURCE defect found within finite first start.** `create_native_owner` supplies explicit enrollment/configuration/templates/root/staging inputs to the existing Dispatcher. NativeActivation.precheck runs on its control-writer thread and calls actual native_startup_inputs → enrolled_runtime → NativeRuntime. NativeRuntime captures that thread as owner; recover_imports/finalization stay on the Core owner, while detached lifecycle I/O stays in the existing executor. No Core/SQLite callback is passed to TLS. The public factory itself is thin; its production wiring was inspected, while the new tests exercise the same activation through Dispatcher with fake TLS injection.

Precheck validates the private enrolled root, refuses any existing entry before TLS, brackets canonical enrollment inputs around actual runtime construction/healthy exact-boot hello, and checks staging identity plus transport emptiness. NativeRuntime separately rejects overlapping spool/staging; NativeActivation rejects either overlapping the Core root in either direction. Postcheck executes after Core acquires lifetime exclusion, requires Core's fresh_initialized flag, compares canonical reconstructed snapshot and reread record bytes before and after another healthy hello, requires the exact configured five-field native identity, strict integer zero launches and no quarantine, and repeats empty spool/staging checks. Canonical comparisons preserve the r305 true/1/1.0 identity correction. A Core initialized and closed in the acquisition gap is rejected; an outstanding competitor is subject to existing ownership exclusion.

All retained Core roots refuse, including a prior clean zero-job start. Any spool/staging entry refuses, including completed import evidence. Failed post-acquisition startup may leave a newly initialized root that permanently refuses this first-start API; no repair or cleanup is performed. This is the chosen authorized alternative, not a missing-history blocker. No Linux history reader grants native activation: the installed native-policy refusals in recovery remain unchanged. Native receipt substitution is independently exercised by the selected Linux-runtime regression.

Checked native owners are explicitly treated as enrolled, so they use the existing closed activation predicate rather than the unenrolled foundation path. Reconciliation, quarantine/orphan, resource, persistence, integrity and transport import-admission gates remain active; _ready additionally checks configured profile readiness fields. Activation does not promote profiles or mint qualification. Zero launches is a startup/reconciliation constraint, not a new scheduler or a per-job limit. Existing publication, frozen Authority/plan, cancellation, reservation/ACK and import-uncertainty implementations are unchanged. Selected cancellation and publication-storage regressions passed through the actual existing native lifecycle with injected process/transport/backend observations.

NativeActivation registers a strong Dispatcher reference before potentially blocking startup work. Constructor timeout does not discard that responsibility: the owner continues to be retained in _INSTALLED_LIFECYCLES. Successful refusal cleanup or successful close removes it; failed pool/Core close preserves it and _cleanup_failed prevents retry. The identity guard in finished prevents a rejected second Dispatcher from deleting the first owner's record. Both native reuse and failed-close tests passed. The constructor-timeout path is supported by source inspection, not a separate wall-clock timeout experiment in this finite receipt. The private caller still owns eventual close/drain; this patch does not claim automatic public lifespan supervision or automatic timeout cleanup. Retention is an in-process obligation, not durable crash recovery.

The operator-controlled immutable-input/no hostile same-UID writer premise remains relevant. Repeated snapshots are not an atomic transaction, FD-pinned TLS trust loading or hostile-writer/ABA isolation. These stated limits do not contradict the bounded implementation claim.

Both explicit distribution manifests include native_activation.py; selected archive membership/legacy metadata checks passed. NativeRuntime/recovery/reconciliation/host policy source implementations are unchanged by this candidate. No new DDL, enrollment bypass, readiness record, scheduler or public configuration surface was introduced.

## Independent one-shot execution receipt

Inspected tests/conftest.py and tests/service/conftest.py before execution: direct unittest avoids the maintenance autouse guard and socket/uvicorn fixtures. Inspected selected setup and helper seams: tiny disposable filesystem/model metadata/ZIP fixtures, fake TLS/certificates, injected host/process/reap/backend observations. No actual child, listener or model execution is required. Archive selection is inert.

Exactly one invocation from candidate cwd: `/private/tmp/pab-0c-venv/bin/python -I -B -`. The stdin script explicitly prepended absolute candidate root and candidate `tests/service` paths, printed and asserted interpreter version, used defaultTestLoader.loadTestsFromNames, asserted 18 cases, and ran TextTestRunner(verbosity=2, failfast=True), exiting nonzero on failure. Observed **Python 3.14.7 (main, Aug 5 2026, 10:29:49), Clang 17.0.0 (clang-1700.6.4.2)**, not target Python 3.11.

Exact selection:

1. `tests.release.test_native_activation.NativeActivationTests` — all 8 methods.
2. `tests.service.test_installed_activation.ActivationTests` — `test_post_core_private_reader_under_lifetime_lock`, `test_changed_post_acquire_hello_refuses_and_releases_core`, `test_late_reconciliation_hello_change_stays_closed`, `test_close_requests_stop_but_waits_for_receipt` (the four r307 methods).
3. Same Linux ActivationTests — `test_repeated_acquisition_cancellation_retains_owner`, `test_cancelled_acquisition_close_failure_retains_obligations`. Added because shared Dispatcher startup/cleanup and lifecycle registry integration are touched.
4. `tests.service.test_native_runtime.NativeRuntimeTests.test_linux_runtime_rejects_native_lane_and_preserves_exact_linux_receipt`.
5. `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`.
6. `tests.service.test_native_lifecycle.NativeLifecycleTests.test_cancel_retains_reservation_until_actual_owned_receipt` and `test_native_publication_storage_uncertainty_retains_reservation_until_reopen`. Added to verify preserved native cancellation/publication obligations through shared Dispatcher changes.

Receipt: **Ran 18 tests in 7.287s; OK; exit 0**. Tool session 41408, chunks b800ae/1b9259 retain raw verbose output. The failed-close Linux regression printed `RuntimeError exception in shielded future` with expected `RuntimeError: injected_close_unknown`; that test passed its explicit retention assertions. No retry, second selection, development execution or actual timeout/crash probe. Candidate remained clean at exact report HEAD afterward.

Tiny inert fixtures were explicitly authorized below 6 GiB; no heavy ACTUAL gate waiver is implied. Only fixture-owned teardown occurred. No network/listener, Docker, download/install, actual child/ML/crash, HELD/STT action or operational cleanup.

## Parent-owned gates and disposition limits

Parent may separately adopt exact reviewed SOURCE. Full native completed-history/publication/import-intent/schema validation, stopped boot/trust renewal, authenticated public entrypoint/lifespan/credentials, Docker-control topology/mount policy and supported fresh owned-voice onboarding remain deferred Lot B work, not blockers to this explicitly bounded first-start alternative. Do not advertise a complete installed native journey or retained-root restart from this factory.

No reviewed arm64 wheelhouse is available. Actual Darwin dependency provenance/install/import/ABI, real authenticated TLS/reachability/firewall/denied peers, hardware/model/MPS, process/FD/death/resource observations, generate/cancel/restart, clone lifetime, audio and performance qualification remain unrun. **Actual Metal remains UNAVAILABLE / UNQUALIFIED.** CPU and CUDA release rows remain separate mandatory gates. Source/inert PASS does not close these gates, waive legacy digest RED, authorize heavy ACTUAL work or constitute parent adoption.
