# Native supervisor stage independent review — r281 Sol

2026-10-25. **NO-GO for acceptance of the implemented r280 supervisor stage.** Four material findings below concern implemented handlers, not deferred client/Core integration. The partial foundation remains usable source; this review does not supersede r279's bounded foundation GO. Full Lot A, native activation/readiness, actual Metal and release qualification remain NO-GO.

## Authority and pins

Review-only. Sole lasting write: this root report, mode 0600. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-a5d44039`, branch `t3/native-supervisor-child`, clean before/after execution.

- Base `326c4d28ecbb935b96cb3b50e35342b076a66ac4`.
- Foundation cherry-picks `5e626a919a4d0d83a98d503bf29adf83fefe115f` and `5da25c605899ff85598ca55ef7f83113ec70b42c`.
- New source `de6c07be80d702c19f7dcabe113e16b02e7dcae4`.
- Docs HEAD `b16694f5654de69b9536912d97b07b83c74c7309`.
- Root HEAD at review `5ae10fdba90637a8ca378963a5c232e77bbfb4a7`; existing untracked STT evidence was listed only, not opened or changed.

Read full normative r277, historical partial r278, root r279 and candidate r280. r279 is absent from the candidate and was read at its explicit root location. r277 SHA256 verified `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`. Inspected actual five-file delta (supervisor, child, protocol, supervisor tests, archive manifest), backend/spawn guard, engine/reference/encoder paths, transport and Linux Dispatcher/reconciliation/recovery consumers. No applicable AGENTS.md found in candidate discovery or checked ancestors. Memory keyword lookup supplied no relevant technical authority; findings derive from current source and executed probes.

Entry available storage was 3,400,956 KiB, below 6 GiB. Only tiny disposable inert fixtures and existing archive fixtures ran; injected capacity never established actual-work admission. No pytest collection, actual socket/listener, process launch, actual ML, hardware, Docker, network, acquisition/install, real UID qualification, process crash, operational cleanup, HELD action, delegation, candidate/source edit or commit. Existing fixture teardown removed only its own disposable fixtures.

## Material findings

### F1 — P1: storage quarantine disables termination of a still-owned child

`worker/native_supervisor.py:174–180, 227–231, 383–397`.

`persist` sets `self.failed=True` for any journal failure. The watchdog's TERM/KILL branch requires `not self.failed`; public stop is also rejected by the quarantine guard. Thus an unrelated historical ACK journal failure while a new child is loading/running permanently prevents cutoff/cancel/shutdown escalation, despite the exact live handle still being held. Receipt authority must remain UNKNOWN, but that does not justify leaving the owned model running indefinitely. The same suppression affects escalation after a storage fault between TERM and KILL. This is distinct from an identity mismatch, where signalling must refuse.

Executed inert reproduction: stop a first allocation to terminal; retain its ACK request; register/prepare/start a second job using the actual handlers with fake Popen/channel; inject `OSError('ACK fsync failure')` into `journal.save` for the first ACK; restore save; advance fake monotonic clock 26 seconds and tick. Observed `failed=True`, `signals=[]`, active receipt `None`, exact fake handle retained. A separate direct failed-state probe produced the same absence of signals. No actual process was launched or signalled.

Required correction: preserve quarantine/no-receipt/no-relaunch fences while independently continuing safe termination of the exact owned live handle. Add a historical-ACK/storage-fault-during-active-load test, including TERM-to-KILL continuation.

### F2 — P1: watchdog shares a lock held across journal I/O and expensive lifecycle work

`worker/native_supervisor.py:214–229, 262–289, 333–360, 368–369`.

Every watchdog tick acquires `self.lock`. Dispatch holds that same lock through ACK persistence, including file/directory fsync; prepare also performs registration, full bundle verification and journal persistence under it. Start holds it across launch-intent persistence and Popen. In particular, ACK of a previous terminal record is allowed while the current child is active, so a stalled historical ACK fsync blocks cutoff TERM/KILL for that child. The watchdog avoiding `self.transfer` does not establish independence from journal I/O required by r277. The implementation report's trusted-system-call limitation does not remove this shared-lock dependency.

Executed inert lock probe: start fake active child, hold `self.lock` from a second Python thread, move fake time past cutoff, and call tick from another thread. After 50 ms, tick had not completed and signals remained empty; after releasing the lock, it issued TERM. Source inspection identifies the actual journal/ACK path holding that lock; no slow disk syscall or kernel timing claim was tested.

Required correction: separate minimal handle/identity signalling synchronization from durable state/verification I/O. Demonstrate that an injected blocked historical ACK save cannot block stop or watchdog escalation, while preserving handle-registration/signalling serialization.

### F3 — P1: a later watchdog tick can expose a receipt after durable-write uncertainty

`worker/native_supervisor.py:174–180, 225–229, 364–377`.

For a prepared allocation, stop adds cancellation before attempting terminal persistence. If that write fails, `failed=True` and the immediate collect correctly returns no receipt. But the no-active branch of tick ignores `self.failed` and retries terminal persistence for the cancelled prepared record. Once the injected failure disappears, it writes a terminal receipt; observe reports stopped-and-reaped, collect exposes the receipt and ACK can persist even though the supervisor remains quarantined. This contradicts the explicit r277 durable-write-failure UNKNOWN fence and r280's claim that journal uncertainty cannot become terminal. A directory-fsync failure after replacement is especially not resolved by assuming the old in-memory phase proves a safe retry.

Executed reproduction: prepare; inject one save failure during stop; restore save; collect; tick; collect again. Observed `before=None`, then `after_reaped=True`, with `failed=True` throughout. No process was needed. The public collect path returns that stored receipt without a quarantine check.

Required correction: latch uncertain allocations and prohibit automatic terminal promotion/receipt exposure after persistence uncertainty. Keep original evidence/ownership for parent recovery. Test repeated tick/stop/collect/start/ACK after both pre-replace and post-replace/directory-sync failures.

### F4 — P2: completed old-boot output cannot be downloaded after restart

`worker/native_supervisor.py:223–224, 248–253`.

The historical-boot exception permits observe/collect/ACK but excludes `get_output`. A completed old record therefore returns its proposal/output descriptor through collect, yet every attempt to retrieve its bytes is rejected: original boot fails the current-boot check; current boot fails the record tuple check. Output is retained and revalidated on journal reopen, so this is a handler authorization error, not missing NativeRuntime work. It breaks same-output redownload after a lost collect reply/restart and the promised readable completed history.

Executed reproduction reused the actual child-entry/fake-ML/encoder/parent-receiver/fake-wait fixture, reopened its completed journal and ACKed it as the existing test does. Historical collect returned a 48-byte output; `get_output` with its exact original runtime/binding/boot/digest returned `native_session`.

Required correction: permit immutable terminal output reads under the exact original allocation tuple after authenticating the current deployment/configuration; keep old-boot mutation/start refusal. Extend the restart fixture through actual get_output, not only collect/ACK.

## Independently executed validation and reproduction accounting

Both conftests were inspected before execution: pytest maintenance hooks and the service fixture can activate real listeners; neither was collected. All Python execution used `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd and explicit `sys.path[:0]=['.','tests/service']`.

1. Direct unittest selection: `tests.worker.test_native_supervisor`, `tests.worker.test_native_protocol`, `tests.worker.test_native_qwen.NativeQwenTests`, `tests.worker.test_engine_qwen`: **44 PASS**, 2.615 seconds.
2. Separate fresh interpreter: `tests.worker.test_qwen_import_seam` and `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`: **6 PASS**, 0.422 seconds. Complete loader/root/extracted-archive/legacy import checks use fake upstream; no dependency ABI claim.
3. Two additional direct stdin scripts executed **six observational inert probe scenarios**, not six additional unittest passes: (1) failed active cutoff; (2) failed prepared terminal persistence then tick; (3) socketpair-factory failure; (4) lock-blocked watchdog; (5) completed old-boot output; (6) historical ACK failure during active second job. Expected adverse observations occurred without script errors. Probe 3 showed direct dispatch leaving launch-intent/alive after pair factory failure; the production serve_connection generic exception handler quarantines this case, so it is not elevated to a separate production finding. Probe 1 is independently strengthened by the actual handler-triggered fault in probe 6.
4. `git diff --check 326c4d28 HEAD`: PASS. Candidate remained clean. No retries of failed test suites or source corrections were performed.

The probe recipes above use `NativeSupervisorTests` setup/cleanup, its fake `Handle`/`Channel`, `patch('worker.native_child.parent_exchange',return_value=None)` and explicit fake-clock advancement. F4 uses `test_actual_child_result_terminal_output_and_oldboot_replay` followed by historical get_output. The fixtures are tiny and their inventories/readiness labels are synthetic declarations. Printed MPS/CPU/CUDA load messages establish no hardware result. Existing tests all passing does not cover these failure combinations.

## Bounded positive assessment and limits

The new child script explicitly inserts the archive root from its fixed __file__ before worker imports, addressing Python -I script-path isolation at source level. The production argv is absolute interpreter, -I/-B, fixed child file and only inherited socket FD; close_fds/pass_fds, DEVNULL, offline environment and new session are explicit. No real exec or architecture/dependency qualification occurred.

The child validates bounded canonical frames, token/binding and native execution, installs the accidental Python spawn guard before lazy ML, and uses the in-process backend and existing encoder. Parent exchange requires chunk completion/hash and EOF by the original cutoff. Terminal normally requires exact fake wait, joined receiver and verified parent FD closure. Group signalling is serialized with handle selection and skips it after observed owned exit; default null start identity is only the trusted exact-handle premise, not Darwin identity API qualification. No Linux ECHILD/native UID substitution is introduced. This is explicitly a trusted bundled runtime, not a malicious native-extension/direct-syscall sandbox.

TLS codecs require TLS 1.3 and exact peer certificate digest. CLI checks current-owner mode-0600 regular non-hardlinked trust files and refuses symlink components before SSL loads them; it does not hold no-follow credential FDs through OpenSSL loading. Under the stated immutable/private-directory/no-hostile-same-UID-writer premise this inspection found no additional material credential issue. Enrollment, real certificate exchange, concurrent listener availability and kernel FD/process behavior remain untested.

Journal source has checksum, 1 MiB cap, flock, exclusive temporary file, fsync/replace/directory-sync and restart refusal for unexplained entries. Nonterminal old records block restart and are not PID-adopted. Input/output path derivation, exact offset/replay bytes, PCM/hash caps, retained-spool reservation and 64-lifetime/eight-unacked limits are present. ACK intentionally retains bounded transfer evidence. These checks do not waive F1–F4 or establish an exhaustive crash/fsync/concurrency matrix. Clone nonempty conditioning remains UNKNOWN; only literal empty owned conditioning is removed after exit. No broad cleanup was added.

## Parent-owned remaining gates

Correct F1–F4 and independently re-review this stage before adoption as an accepted supervisor implementation. Missing NativeRuntime/client/Core publication and native receipt-consumer wiring are acknowledged partial scope, not new defects. They still block full Lot A: detached immutable input/output import, trusted local control UID, native registration/admission/resolution, shared frozen-lane receipt selection across live/history/recovery, cancellation/publication-intent/reservation/ACK ordering and the full Authority-to-Core restart/replay slice. Existing Linux consumers retain pid1-waitpid-echild and engine_uid requirements; this stage does not activate native receipts there.

Lot B native package/enrollment/dependency closure and Lot C actual Apple Silicon probe/load/generate/cancel/restart remain separate. Actual Metal is unavailable/unqualified; no actual Metal gate ran. CPU Docker and CUDA qualification remain mandatory separate release rows. `lib/tts.py` SHA256 remains `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`; the historical broader digest RED is not waived. No activation, cleanup or release authority is granted by this report.
