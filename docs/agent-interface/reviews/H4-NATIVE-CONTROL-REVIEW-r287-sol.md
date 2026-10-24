# Native control SOURCE vertical slice independent review — r287 Sol

2026-10-25. **NO-GO for accepting r286 as the complete r277 Lot A SOURCE vertical slice.** The injected, pre-registered path is substantial and its 74 focused tests pass, but ordinary future allocations still require a test-only trusted provisioning call. A bounded connector deadline defect and an additional Pocket public-prepare regression also remain. This is not a rejection of the previously reviewed F1–F5 supervisor corrections. Native installed activation, packaging, actual Metal and final release remain CLOSED.

## Authority, pins and evidence

Review-only; sole lasting write is this root report. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-ef4727f6`, branch `t3/native-runtime-integration`, clean at entry and after execution.

- Base: `bd85e1c4c41e1b1927d6f4cc2385041ac357158d`.
- Source: `434e26747736e303e116bdde0b413282755b1f30`.
- Docs HEAD: `e187bf2061ec61a458aafee82d68472ef9097086`.
- Root HEAD: `bd85e1c4c41e1b1927d6f4cc2385041ac357158d`. Existing untracked STT evidence was listed only, not read or changed.

Read normative r277 and r280–r286, including all normative sections and F1–F5 history. Inspected the complete new NativeRuntime, changed source and consumers: admission, Dispatcher, NativeQwenContexts, reconciliation/recovery/SupervisorRuntime, common Qwen relation decoder, native protocol/receipt and supervisor registration addition, manifest and affected fixtures. Followed Authority, native child/backend/encoder, owned-process terminal handling, ResourceStore and Core publication seams. r277 SHA256 remains `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`. Source-to-docs delta contains only r286. No AGENTS.md found in candidate discovery or checked ancestors/root. A bounded memory-registry search returned no relevant entry; no memory-derived technical authority was used.

Both conftests were read before execution. Root pytest maintenance hooks and service HTTP listener fixtures were neither collected nor executed. All Python invocations used `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd and explicit candidate plus candidate/tests/service paths. No delegation, source edit, commit, actual socket/listener, process launch/signal, real ML/hardware, Docker/network, download/install, crash experiment, operational cleanup or HELD action occurred. Existing fixture teardown touched only its owned disposable fixtures.

Available storage observed: 3,370,884 KiB, below 6 GiB. Only tiny fixtures/archive contents ran with injected native capacity. The actual 6 GiB plus attempt-demand admission gate was not lowered or qualified. Printed CPU/CUDA/MPS load labels are synthetic declarations.

## Findings

### F6 — P1: future jobs cannot traverse the production runtime without out-of-band test registration

`worker/native_supervisor.py:186–188,207–232,315–323`; `service/native_runtime.py:212–235`; `service/dispatcher.py:355–360`; `tests/service/test_native_lifecycle.py:49–56`.

NativeSupervisor.prepare still requires an exact binding/snapshot already present in `registrations`. The new internal `register_snapshot` safely restricts extension to an existing principal/context/profile template, preserving execution, bundle mappings and inventory, and validates Authority/admission. It is absent from wire OPS, as required. However, no production caller invokes it: NativeRuntime sends prepare directly; Dispatcher only detaches input and submits prepare; the native CLI loads a fixed exact-snapshot registration map. The only caller for newly allocated public jobs is the lifecycle fixture's `before_prepare` fault callback, which reaches directly into the native supervisor object from the control owner. Actual separated control/native processes do not have that object reference or a corresponding trusted provisioning route.

This is a functional source integration gap, not evidence of public self-enrollment or a trust bypass. Fixed pre-authorized snapshots work. Public Core allocation creates a fresh job/attempt/binding, so the registered principal/version/context alone cannot authorize the first ordinary new job or the next job. Deferring installed enrollment to Lot B does not establish the missing source path for future allocations promised by Lot A.

Executed inert reproduction: construct `NativeLifecycleTests` with its existing setup and explicitly seeded owned version; patch only `case.f.s.register_snapshot` to return None, removing the fixture-only side effect; call `case.open()`, then actual `production_generate`; wait for phase unknown. Actual Authority → NativeRuntime → canonical fake TLS routing → actual supervisor produced:

```text
wire_ops=['hello','prepare']
phase='unknown', quarantined=True
job_state='running', reservation='executing'
launches=0, native_records=0
```

No runtime/supervisor operation was replaced. The failure is conservative: no model launch/artifact/release. It nevertheless disproves a complete ordinary future-job vertical slice. The reported same-fixture restart/next-job/public HTTP/CLI/MCP successes remain valid only with explicit per-allocation trusted fixture provisioning; they are neither onboarding nor a production registration integration.

Required correction: connect allocation authorization to the already trusted native principal/profile/version/execution/inventory authority through a concrete bounded production source path. Keep register_snapshot internal and independently validate the incoming exact allocation against the pre-existing trusted template; do not add a public enrollment operation or let a request establish new trusted metadata. Exercise first and next actual Core allocations without a fault hook that mutates supervisor registrations. Parent must review the chosen seam and exact scope.

### F7 — P2: the production connector refreshes the timeout between TCP and TLS

`service/native_runtime.py:102–107,177–195`.

_rpc computes an absolute command/original-attempt cutoff, but passes only a remaining duration to _connect. socket.create_connection can consume that duration; _connect then assigns the original duration again to the socket before the blocking TLS handshake. Address resolution and multiple address attempts in create_connection are also not bounded by this single absolute cutoff. The later framing `remaining` check refuses an expired operation, but cannot retroactively bound connector blocking. This can occupy the sole lifecycle executor past the advertised command/attempt communication bound, including delaying control stop work. The independent native watchdog remains intact; no late publication was demonstrated.

Executed deterministic inert probe: use NativeFixture's actual runtime with its production _connect, replacing socket.create_connection and SSL wrapping with context-manager fakes only. Inject monotonic time 100.0; TCP entry consumes 1.5 seconds and TLS entry another 1.5. Call hello with timeout 2.0. Observed connect timeout 2.0, handshake socket timeout 2.0, elapsed 3.0, then `Uncertain` at the frame deadline check, with zero wire requests. No real wait/socket/TLS exchange occurred. The source permits each phase to consume nearly the whole supplied timeout; hostname resolution is an additional unqualified bound.

Required correction: carry the absolute cutoff through address resolution/connect/handshake, recompute remaining time before each blocking phase and refuse expiration before TLS. Provide a bounded strategy for the configured host-gateway hostname rather than assuming socket timeout bounds DNS. Add an inert production-connector deadline regression; preserve TLS 1.3, CERT_REQUIRED, exact server pin and no fallback.

### F8 — P2: shared Linux receipt validation leaves the Pocket public prepare regression red

`worker/native_protocol.py:320–325`; `service/dispatcher.py:495–499`; `tests/service/test_pocket_preparation.py:344–363,431`.

The stricter shared validator correctly requires the exact Linux receipt field `reaped_children`. r286 updates one production API fixture but misses the existing Pocket public prepare fixture, whose success and cancel receipts both omit that field (lines 348 and 362). The public prepare/save/generate/reopen test now quarantines its first prepare result instead of completing. This is a fixture-consumer regression caused by the integration; it is not evidence that actual Linux supervisor receipts omit the field, and it does not justify weakening production receipt validation.

Independent additional regression invocation failed at line 431 waiting for succeeded. Status: reconciled true, quarantined true, phase unknown, runtime `inert/1`, error `Refused`, executing inference reservation, no resource/persistence/integrity fault. The omitted field and new closed-shape validator explain the failure directly. No retry or fixture repair was performed.

Required correction: make affected synthetic Linux receipts match the existing exact Linux journal contract, covering success and cancel, and run the Pocket public prepare journey. Audit indirect/inherited fixture consumers of the shared receipt seam. Keep actual Linux enrollment and native receipt separation closed.

## Bounded positive assessment

NativeRuntime uses actual canonical framed TLS client operations with random per-request nonce, exact response envelope checks and fresh JSON decoding. Server certificate pin and TLS 1.3 are checked for send/receive; response tuple, op, nonce, boot/configuration/lane/runtime/binding and chunk digest/offset/size are closed. Three lifecycle slots and one transfer slot are finite. Accepted empty stop and committing/terminal variants are understood as observations, with no stop reply converted into reap proof. Malformed replies/exceptions fail closed into retained ownership; F7 concerns elapsed blocking, not authentication or fabricated death.

The native CLI checks current-owner mode0600 regular, non-hardlinked trust files and symlink components before OpenSSL loads paths. It does not hold credential FDs through loading; the declared immutable/private-directory/no-hostile-same-UID-writer premise remains essential. Control client key enrollment/loading ownership is still Lot B, not established by accepting a supplied SSLContext. No real TLS or bridge/firewall claim is made.

Dispatcher calls detach on the Core owner before executor submission. Managed clone bytes are authenticated there; the executor receives bytes and immutable Authority, with no Core/SQLite callback. Durable cancellation interrupts an already bound transfer; cancellation received during prepare is checked after binding and before start. Runtime checks interruption between bounded chunks and before staging completion; Core retains its independent cancellation/publication fence. First late import cannot publish, and completed exact local replay remains usable historically. The existing durable publication-intent path precedes the new first-publication deadline check, preserving replay semantics.

The private control spool stays outside scanned staging, with 0700 directories, 0600 exclusive files, no-follow/non-hardlink checks, identity checks during reads and exact byte replay. It accounts declared worst-case retained bytes, limits 64 records/64 MiB and contributes to ResourceStore disk measurement. Partial/conflicting output, missing/tampered clone input and retained import alias block reopening/admission; files are retained rather than repaired or evicted. Output goes through complete hash/PCM/provenance validation, fsync and no-replace linking into the fixed 0770 stage, with private alias removal and directory fsync before return. Trusted local_import_uid must equal control UID; native UID is never a local output-owner credential. Core publication/oracle implementation remains unchanged.

NativeQwenContexts compares the actual owned immutable version, fixes trusted principal/profile/version metadata and checks native inventory freshness at resolution. Admission uses descriptor RAM, one inference PID, zero dedicated VRAM, with independent native disk/RAM/bundle checks retained. Native relation selection requires MPS/float32/sdpa, public metal and packaged_native. Linux CPU/CUDA rules and Linux hardware validation remain separate. Historical relation decoding does not age out probe timestamps. No new public contracts, DDL, scheduler, installer or readiness claim appears.

The shared receipt validator selects from the frozen profile and binds expected configuration, lane, original boot, allocation binding/runtime and deadline. Native receipts require receiver closure and a finite owned-process ledger; timely requires one zero-exit model entry. Linux substitutions are rejected by both live and detached consumers. Stored receipts stay unchanged; local UID comes separately from trusted runtime configuration. Resource release still follows validated reap and durable Core terminal outcome in the reservation transaction, then ACK. Storage uncertainty after durable adoption retains reservation/no ACK until same-root publication-intent replay. Existing Linux installed recovery explicitly rejects native history in its allocation/journal policy; installed activation is not loosened to admit native.

The only supervisor delta is the internal registration helper. It acquires the state lock, performs no handle-lock mutation and leaves F1–F5 terminal/watchdog/persistence paths unchanged. Focused tests retain independent TERM/KILL despite failed/blocked historical ACK, UNKNOWN after journal uncertainty with no retry promotion, original-boot terminal download, and cancellation/terminal-decision barriers. Wait/receiver closure precede terminal persistence; committing remains no receipt. This supports the corrected stage under fake handles/clocks/channels, not exhaustive kernel/concurrency/crash qualification. Nonempty clone conditioning still retains UNKNOWN.

## Independently executed checks

Common command prefix, all from the candidate cwd:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/t3-ef4727f6","/Users/ben/.txcode/worktrees/py-audio-box/t3-ef4727f6/tests/service"];unittest.main(module=None,argv=["unittest", <selections below>],verbosity=0)'
```

Each selection below is a separate quoted argv item; four independent invocations, each once:

1. **74 PASS, 11.529 seconds:** `tests.service.test_native_runtime`, `tests.service.test_native_lifecycle`, `tests.worker.test_native_supervisor`, `tests.worker.test_native_protocol`, `tests.worker.test_native_qwen.NativeQwenTests`, `tests.worker.test_engine_qwen`. This single final invocation includes the additional storage-intent method. r286's historical 73+1 split is accurately distinguished from this new 74-method execution.
2. **32 PASS, 16.683 seconds:** `tests.service.test_qwen_lifecycle.QwenLifecycleTests`, `tests.service.test_pocket_lifecycle.PocketLifecycleTests`, `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`, `tests.service.test_production_reconciliation`, and `tests.worker.test_engine_kokoro.EngineTests` methods `test_exact_arguments_and_detached_identity`, `test_closed_descriptor_and_unsupported_controls`, `test_limits_close_backend`, `test_mutated_invalid_audio`, `test_mutated_numeric_metadata_refuses`, `test_bundle_binding_before_backend` (each passed as its full dotted selection).
3. **6 PASS, 0.380 seconds, separate fresh interpreter:** `tests.worker.test_qwen_import_seam`, `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`. Root/legacy/extracted archive complete loader seams use fake upstream; no actual dependency/ABI claim.
4. **14 run: 13 PASS, 1 FAIL, 10.867 seconds:** `tests.service.test_installed_restart.TerminalRestart`, `tests.service.test_pocket_preparation.PocketPreparationTests`, `tests.service.test_pocket_preparation.PocketPreparationInventoryTests`, `tests.service.test_pocket_preparation.PocketPublicJourneyTests`. Five installed terminal-history and eight Pocket preparation/inventory methods pass; the sole public journey method fails as F8 describes. No unchanged-suite rerun.

Total: **126 distinct selected unittest methods, 125 passing and one failing.** The reported r286 112-method selection independently passes, but the broader appropriate checks are not all green. Linux fixtures emitted ResourceWarning for unclosed SQLite test connections; warnings were not suppressed. Two additional stdin scripts ran the observational inert F6/F7 scenarios above; they are not extra unittest passes. Each used the same interpreter/paths and existing owned fixture teardown.

`git diff --check bd85e1c..HEAD`: PASS. Candidate remained clean. `lib/tts.py` SHA256 is still `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known broader legacy digest RED remains unwaived and was neither regenerated nor claimed green.

## Parent-owned gates

Correct F6/F7 and F8's affected fixture consumers, then obtain fresh independent review of the exact correction. Existing source components can be retained as a bounded pre-registered integration foundation; no complete future-job/native-control acceptance is granted here. No public self-enrollment should be introduced to satisfy the fixture gap.

Lot B must still supply native package/launcher, real dependency closure, control/native private configuration and credentials, explicit bridge/firewall topology, native enrollment and supported owned-version onboarding. Lot C requires separately authorized actual Apple Silicon probe/load/generate/cancel/restart and resource/process/FD/durability qualification with unchanged headroom gates. Actual Metal remains UNAVAILABLE / UNQUALIFIED. Clone arbitrary cancellation with nonempty conditioning remains unresolved. CPU Docker and CUDA remain mandatory separate release rows. No installed/native activation, acoustic/performance, hardware or final release claim follows from this review.
