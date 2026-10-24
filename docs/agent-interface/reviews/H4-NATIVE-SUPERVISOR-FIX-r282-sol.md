# Native supervisor F1–F4 corrections — r282 Sol

2026-10-25. Implemented SOURCE corrections for all four r281 findings. Pending
parent fresh independent Sol review before adoption. Full native Lot A remains
PARTIAL/CLOSED to acceptance; installed readiness, actual Metal and release
qualification remain NO-GO.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-4d9ec80c`, branch
`t3/fix-native-supervisor-races`. Clean entry
`8ac77dd7e0e53daa66516cde4a9bff1761d79ead`. Root foundation was already present;
no foundation cherry-pick was repeated. Only the requested supervisor source
`de6c07be80d702c19f7dcabe113e16b02e7dcae4` was cherry-picked as
`128a189f2597917677598e1f1a1316b6db02597f`, then report
`b16694f5654de69b9536912d97b07b83c74c7309` as
`60dae5082a3b94cdbe995074c83c414b67dfb512`.
Historical `t3-a5d44039` was not changed.

Correction source commit: `e9c17128f65d36d0e97befc3bf38e7a5227183f8`.
This report is a separate subsequent commit. Only `worker/native_supervisor.py`
and `tests/worker/test_native_supervisor.py` changed in the correction source.
No protocol/schema, service/Core, foundation, installed policy or manifest change.

Read full root r281, normative r277 and r280. r277 SHA256:
`0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`.
No applicable AGENTS.md found in repository/checked ancestors. Memory registry
keyword search had no relevant hit and supplied no technical authority.
Both test conftests inspected; neither was collected or executed.

## Corrections

F1: Quarantine no longer disables exact-owned-handle TERM/KILL. Public stop
checks deployment/configuration/lane/original runtime/binding/boot before taking
the minimal handle lock and setting cancellation, including while state I/O is
blocked. Signalling checks poll and retained start identity on every live pass;
observed exit permanently stops numeric group signalling. Failed storage still
blocks new launch and terminal promotion, retaining the handle and UNKNOWN.
An exact wait can complete under quarantine without creating a receipt.

F2: `handle_lock` serializes active registration/retirement, live poll, identity
and signalling. No journal, prepare verification/hash, Popen, receiver join or
blocking wait occurs under it. Popen runs with a reserved active slot whose
handle is initially absent; stop can latch cancellation there, and publication
of the returned exact handle is serialized before later signalling. Tick signals
first, then tries the state lock without waiting. Journal/state operations remain
serialized; persist now takes that state lock even from input transfer. No
state-lock acquisition occurs while holding handle_lock. Shutdown cancellation
uses the same handle synchronization. Trusted poll/identity/signal calls remain
platform operations, not a hard realtime guarantee.

F3: Failed persistence latches the affected allocation ID in `uncertain` and
quarantines the supervisor. Prepared tick cannot promote anything after failure;
terminal has its own failure fence. Uncertain collect returns nulls, observe is
UNKNOWN, ACK cannot retry the uncertain history, and start/stop mutation cannot
retry terminal persistence. Immutable previously valid history is not globally
waived or discarded. Both pre-write and real post-replace directory-fsync fault
injections preserve evidence and perform no later save. The in-memory uncertainty
latch is not a new durable recovery protocol; no restart after an uncertain write
was authorized or claimed. Parent recovery must retain the original evidence.

F4: Authenticated terminal get_output accepts the exact original allocation tuple
across boot changes and after ACK, with the same stored digest/range and full-file
hash validation. Old-boot start/mutation still refuses. The actual child/encoder/
parent-receiver fixture now compares actual downloaded bytes before restart,
after restart and after ACK, and rejects wrong boot/binding/configuration/digest.

All self.failed and supervisor lock uses were inspected. Exact wait, joined
receiver, verified FD closure and durable terminal remain mandatory. Failed wait
is attempted once and retained UNKNOWN; no blind wait retry or receipt waiver.

## Executed inert verification

Entry free space 3,444,604 KiB, below 6 GiB. Only tiny existing disposable fixture
and archive contents were used, with injected admission capacity. Actual-work
6 GiB plus attempt demand gate is unchanged. No real ML, native process, signal,
hardware, Docker, network, download/install, listener, process crash, operational
cleanup, HELD action or delegation. Existing fixture teardown handles only its
own disposable fixtures. Printed CPU/CUDA/MPS names are fake fixture declarations.

Final focused invocation:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=[".","tests/service"]; unittest.main(module=None,argv=["unittest","tests.worker.test_native_supervisor","tests.worker.test_native_protocol","tests.worker.test_native_qwen.NativeQwenTests","tests.worker.test_engine_qwen"],verbosity=0)'
```

52 PASS, 2.601 seconds: original 44 plus eight new regression methods. Six review
probe regressions cover historical ACK failure during a second active attempt,
blocked historical ACK cutoff, blocked ACK public stop, prepared pre-replace
uncertainty, post-replace directory-sync uncertainty and pair-factory failure.
The existing actual-byte old-boot fixture is extended for F4. Two additional new
methods cover permanent storage failure between TERM/KILL with receiver closure
unknown, and blocked Popen/stop/registration plus changed identity and observed
exit. All use actual handlers with fake Popen/channels/clocks; thread barriers
bound blocked I/O deterministically. No actual socketpair factory is entered.

Separate fresh interpreter:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=[".","tests/service"]; unittest.main(module=None,argv=["unittest","tests.worker.test_qwen_import_seam","tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change"],verbosity=1)'
```

6 PASS, 0.389 seconds, including root/extracted archive fake-upstream imports.
This ran before the final wait-once/state-lock refinements, which affect only
supervisor lifecycle and were covered by the subsequent final focused run.
Earlier focused runs: 50 PASS then 52 PASS; no failing test run occurred.
`git diff --check`: PASS. `lib/tts.py` unchanged, SHA256
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
Historical broader legacy digest RED remains unwaived.

## Parent gates and limits

Fresh independent Sol review of this exact correction is required before
supervisor adoption. No independent review or acceptance was self-issued here.
NativeRuntime/client, immutable control-side input/output import, shared native
receipt consumers, Dispatcher/Core cancellation/publication/reservation/ACK
ordering and the complete Authority-to-Core restart/replay slice remain absent.
Full Lot A stays partial. Package/enrollment/dependency closure and actual Apple
Silicon probe/load/generate/cancel/restart remain later gates. Clone nonempty
conditioning remains UNKNOWN; no cleanup framework was added. CPU Docker and
CUDA remain mandatory separate qualification rows. These injected source tests
establish no kernel FD/process/identity, TLS deployment, actual hardware, acoustic,
performance, installed readiness or release authority.
