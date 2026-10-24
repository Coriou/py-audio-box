# Native control correction independent review — r289 Sol

2026-10-25. **NO-GO for accepting the complete corrected native SOURCE vertical slice.** F6–F8 are corrected within their bounded SOURCE scope, but fresh execution exposed **F9 (P2): native output import races the resource scanner's directory identity checks and durably faults an ordinary public generation journey**. A separate deterministic inert interleaving reproduces it with actual production classes. Parent adoption remains pending. Native packaging/enrollment, installed activation, real TLS deployment, actual Metal and release remain CLOSED / UNQUALIFIED.

## Authority and pins

Review only. Sole lasting write: this report in `/Users/ben/Projects/py-audio-box`. No source edits, commit, delegation, maintenance/listener execution, actual socket/process/signal, real ML/hardware, Docker/network/download/install, crash experiment, operational cleanup or HELD action. Existing fixture teardown touched only its own disposable resources. Root's pre-existing untracked STT evidence was listed only, not opened or modified.

Candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-fa09f5b2`, branch `t3/native-control-integration`, clean at entry and after checks:

- Parent/root HEAD: `fdc70aa18f95f745cf09a37ccae8149791e6e1e9`.
- Base: `bd85e1c4c41e1b1927d6f4cc2385041ac357158d`.
- Source cherry-pick: `ceecf828b5aba035b627565e32f92c88992a67dc`.
- Integration report: `d319205afa81ecbe8efe604de62b45c9b230dfa5`.
- Correction: `9ed05f744ac3342dacb114e301e33fe4cfbc254b`.
- Docs HEAD: `5ba3c66ace2c5853101424c44c73be863c924854`.

Read full normative r277 and r285/r286/r287/r288. Inspected correction delta, complete NativeRuntime, supervisor authorization/prepare and terminal/watchdog seams, common Qwen relations, Authority/oracle, native contexts/admission, shared live/history receipt consumers, Dispatcher publication/resource/ACK ordering and Core owner checks. No applicable AGENTS.md found in candidate discovery or checked ancestors. Bounded memory-registry search returned no relevant hit; no memory-derived technical authority was used.

Read `tests/conftest.py` and `tests/service/conftest.py` before execution. The maintenance autouse hook and HTTP listener fixture were not collected or invoked. Explicit unittest selections and five direct dispatcher cases only. All project/test interpreters used `/private/tmp/pab-0c-venv/bin/python -I -B`, candidate cwd and explicit absolute import paths. Available storage was 3,309,716 KiB, below 6 GiB; only tiny inert fixtures/archive bytes ran with injected native capacity. Production headroom gates were neither lowered nor qualified. Printed CPU/CUDA/MPS load labels are synthetic.

## F9 — P2: native stage mkdir/chmod races the resource monitor

Material locations: `service/native_runtime.py:356–358`; `service/resource_store.py:151–152,185–204` (identity includes mode), `service/resource_store.py:69–85` (durable fault); `service/dispatcher.py:395–398,415,515–516`; public regression `tests/service/test_native_lifecycle.py:163–213`, failing wait at line 201.

The lifecycle executor's actual `_collect` exposes the new stage through `stage.mkdir(mode=0o770)` and then separately calls `stage.chmod(0o770)`. Normal restrictive umasks make the first visible mode 0750 or 0700. The owner continues `_resource_check` while executor I/O is pending. ResourceStore samples the directory's mode with `stat`, opens it and checks its FD identity, then checks again after scanning. A scan spanning the chmod sees a changed mode and raises `resource_namespace_changed`. This is an expected writer transition being interpreted as namespace substitution. It does not require a hostile writer, invalid model result or actual native hardware.

The fresh 80-method invocation failed only `test_seeded_http_cli_mcp_generation_job_artifact_and_owner`:

```text
AssertionError: bounded source fixture wait expired
reconciled=True, quarantined=True, phase='idle'
error='resource_namespace_changed', resource_fault=True
persistence_fault=False, stop_obligation=False
reservation state='retained', integrity_fault=False
wire operations: hello, prepare, start, observe, collect, get_output, ack
```

The test failed waiting for job success; it was not rerun or relabelled green. Its final status had no current runtime, so native reap/ACK alone did not clear the persistent resource fault. This is an availability/product completion defect, not demonstrated late publication or fabricated death.

Independent deterministic reproduction used a separate NativeFixture: actual `runtime.prepare/start`, fake owned Popen/channel/ML, actual child encoder/receiver and terminal receipt; then actual `runtime.collect` in a thread and actual `ResourceStore._measure_staging` on the owner. Within this disposable interpreter only, set umask 0077 (restored in finally). Delay only `os.chmod` for the new exact staging directory with Events. The scanner observes its pre-chmod stat; an `os.open` wrapper at that exact directory-entry open releases chmod and waits for its completion before calling the real open. Production scan then raises exactly `resource_namespace_changed`; actual collect still completes successfully. Both waits and thread join are bounded at three seconds. No scanner/runtime return value or identity field was fabricated. The standalone scanner used `ResourceStore.__new__` because this non-removal path needs no DB/Core; the full public test independently demonstrates the durable deployment fault.

Reproduction recipe, after constructing `f=NativeFixture(unittest.TestCase())` and `rid=f.launch()` with the normal explicit paths:

```python
stage = f.stage / f.a.job['attempts'][-1]['staging_id']
# In the collect thread's exact-stage os.chmod wrapper:
entered.set(); assert release.wait(3)
try: return real_chmod(path, mode, *args, **kwargs)
finally: changed.set()
# In the scan thread's os.open wrapper, only for stage.name with dir_fd:
release.set(); assert changed.wait(3)
return real_open(path, flags, *args, **kwargs)
# Start collect, await entered, then:
ResourceStore.__new__(ResourceStore)._measure_staging(f.stage)
# ValueError('resource_namespace_changed'); collect joins and returns normally.
```

Required correction: make final-mode stage visibility and resource observation coherent under the actual owner/executor architecture. For example, perform validated staging-directory creation on the owner before asynchronous import, or introduce a narrowly reviewed atomic private-to-stage transition with appropriate identity/durability/accounting fences. Do not suppress arbitrary directory mode/substitution faults, widen the Pocket removal exception, depend on ambient umask, or disable monitoring during transfers. Add an inert barrier regression for this exact transition, then rerun the affected public/native selection and obtain fresh review. No implementation choice is approved here.

## F6–F8 correction assessment

**F6 closed in bounded SOURCE scope:** `worker/native_supervisor.py:186–233,315–343`. Prepare now invokes internal `register_snapshot` for a missing binding. Authority validates frozen plan and typed execution first. Template equality fixes principal, complete context except authored request, and complete profile inventory. Thus resolved settings, owned voice/version metadata, descriptors, model/revision/maps, references, hardware, native deployment metadata and capabilities cannot be enrolled by a request. Candidate registration is independently checked before insertion. Admission and execute-message validation precede journal creation; bundle/headroom/deadline checks still precede allocation persistence. No new wire enrollment operation exists.

The authored text may change as intended, but closed request/worker-plan schemas, frozen-plan relations and `qwen_relation` retain the version/identity/fallback/variants/reuse/text-character/text-byte bounds. The actual child consumes that validated request text; it does not merge caller fields into resolved settings or descriptor controls (`worker/runtime_protocol.py:171–213`; `contracts/validate.py:316–419`; `service/worker_transport.py:18–49`). No request-established new model/settings/reference/probe authority was found. New negative tests recompute hashes where relevant and refuse altered metadata with zero journal/exact registration creation.

Fresh first/next/same-root-restart lifecycle test passes without a prepare callback mutating registrations (`tests/service/test_native_lifecycle.py:49–110`). Its explicit initial owned-version/template seed and trusted boot refresh remain SOURCE configuration, not public onboarding. The separate public HTTP/CLI/MCP journey reached prepare/start/output without such a hook but fails F9. Additional independent no-hook probe changed job ID and authored text, recomputed plan hash and called actual NativeRuntime prepare through canonical fake TLS into actual supervisor: accepted exact registration, one journal record, zero launches.

Growth is bounded, not transactional rollback of every refused prepare. Constructor and insertion cap registrations at 64; journal caps 64 lifetime records/eight unacked, one active; transfer reservation stays 64 MiB. A valid new registration is inserted before disk/RAM/bundle/deadline rejection and remains on refusal. Independent probe sent 65 distinct otherwise-valid IDs with injected free_bytes=0: registration count stopped at 64, journal records=0, launches=0. Thus rejected trusted-principal traffic can exhaust the finite retained registration capacity; no unbounded growth or automatic eviction/rotation occurs. Operators must treat this retained-exhaustion behavior honestly in later provisioning/reconciliation, rather than claim 64 successful future jobs per template.

**F7 closed for the production client:** `service/native_runtime.py:85–129,199–220`; `worker/native_protocol.py:180–218`; `worker/transport.py:29–43`. Trusted endpoint must parse as numeric IPv4/IPv6; hostnames and scoped IPv6 refuse. `_connect` uses one numeric sockaddr and no resolver/multi-address retry. `_rpc` supplies its absolute two-second/original-attempt cutoff; TCP, pre-wrap, explicit TLS handshake, post-handshake and framing use the remaining same budget. Owned TLS/raw descriptors close on success/expiry/exception. TLS1.3 min=max, CERT_REQUIRED and per-frame exact certificate pin remain; there is no fallback. Injected connector is constructor-only trusted wiring and receives the absolute cutoff, not a public configuration seam.

Independent actual `_rpc` + production `_connect` probe injected socket/SSL/clock only. TCP consumed 1.5 seconds of a two-second command. Handshake consumed .25: observed timeouts raw 2.0, raw .5, TLS .5, send .25, frame header .15, body .14; one canonical hello, both descriptors closed. Handshake consuming the remaining .5 refused `native_deadline` before any wire request, both descriptors closed. getaddrinfo was patched to raise if used. Existing new tests additionally cover pre-TCP/pre-handshake expiration and hostname refusal.

Timeout audit: active prepare/input/start/output RPCs intersect original cutoff; stop/observe/collect/ACK have fresh finite communication bounds for post-cutoff reap/recovery. Upload/download loops recheck cancellation and original cutoff. Historical completed local replay and durable publication-intent replay retain their separate recovery semantics. Native listener still separately bounds TLS handshake and handler framing at two seconds each (`worker/native_supervisor.py:521–563`); the client correction does not establish a single combined server accept-to-handler budget. Server synchronous local verification/fsync/dispatch is not made preemptible by socket timeouts. Independent watchdog ownership remains separate. These existing source limitations are not a real TLS/runtime scheduling qualification; no new server elapsed-time guarantee is claimed.

**F8 closed:** exact Linux receipt validator remains strict and unchanged by correction (`worker/native_protocol.py:320–340`). Five injected zero-child receipts now truthfully supply `reaped_children=0`: Pocket success/cancel, dispatcher, installed activation and worker DispatcherEngine. No actual supervisor counts were relabelled. `Mock(spool=None)` is confined to SelectedRestart; production resource accounting gets no Mock/fallback workaround. The complete affected 46 plus worker DispatcherEngine invocation passes, as do five direct affected dispatcher cases.

## Remaining integrated SOURCE assessment

Private spool stays outside staging with 0700/0600, finite declared accounting, no-follow/non-hardlink/identity/hash checks, retained uncertainty and no automatic repair/eviction. ResourceStore includes it. Owner detach authenticates clone bytes before executor work; no Core/SQLite callback crosses that boundary. Output validates full hash/PCM/provenance/proposal and uses no-replace linking, alias removal/fsync and trusted control-local import UID; Core independently requires local owner/nlink/media/manifest. F9 concerns directory visibility, not a waiver of these file checks.

Native contexts compare the actual owned immutable version and enrolled inventory, with freshness at resolution/admission/start. Native accounting is one inference process, descriptor RAM and zero dedicated VRAM; native independent bundle/RAM/disk checks remain. Shared receipt selection is frozen-lane based and binds native configuration/lane/original boot/runtime/binding/deadline with receiver closure and owned-process ledger; both live and detached consumers reject Linux/native substitution. Stored receipts remain unchanged. Dispatcher checks cancellation after binding/before start and interrupts transfers; first late publication refuses, while existing publication-intent recovery precedes the first-publication deadline check. Durable Core outcome and reservation transition precede ACK. Storage ambiguity retains reservation/no ACK until exact same-root replay.

F1–F5 supervisor terminal/watchdog/persistence code is unchanged by the correction; its selected inert regressions pass, including independent TERM/KILL, blocked/failed ACK, UNKNOWN fences, original-boot output and cancel/decision barriers. This is not exhaustive race, kernel identity or crash durability proof. Existing installed Linux recovery rejects native activation. No public schema/DDL/Core publishing/legacy-loader change was introduced to satisfy these fixes.

## Independently executed evidence

All selections below ran once on exact docs HEAD, with no unchanged failing-suite retry. Common command form:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest;sys.path[:0]=["/Users/ben/.txcode/worktrees/py-audio-box/t3-fa09f5b2","/Users/ben/.txcode/worktrees/py-audio-box/t3-fa09f5b2/tests/service"];unittest.main(module=None,argv=["unittest", <separate quoted selections>],verbosity=0)'
```

1. **80 run, 79 PASS / 1 FAIL, 17.934 seconds:** `tests.service.test_native_runtime`, `tests.service.test_native_lifecycle`, `tests.worker.test_native_supervisor`, `tests.worker.test_native_protocol`, `tests.worker.test_native_qwen.NativeQwenTests`, `tests.worker.test_engine_qwen`. Sole failure is F9 above.
2. **6 PASS, 0.475 seconds, separate fresh interpreter:** `tests.worker.test_qwen_import_seam`, `tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change`. Fake upstream root/legacy/extracted-archive loader evidence only.
3. **32 PASS, 20.717 seconds:** `tests.service.test_qwen_lifecycle.QwenLifecycleTests`, `tests.service.test_pocket_lifecycle.PocketLifecycleTests`, `tests.service.test_pocket_lifecycle.ExecutionCleanHistoryTests`, `tests.service.test_production_reconciliation`, plus `tests.worker.test_engine_kokoro.EngineTests` methods `test_exact_arguments_and_detached_identity`, `test_closed_descriptor_and_unsupported_controls`, `test_limits_close_backend`, `test_mutated_invalid_audio`, `test_mutated_numeric_metadata_refuses`, `test_bundle_binding_before_backend` (each passed fully qualified).
4. **47 PASS, 45.212 seconds, one all-green affected invocation:** `tests.service.test_pocket_preparation.PocketPreparationTests`, `tests.service.test_pocket_preparation.PocketPreparationInventoryTests`, `tests.service.test_pocket_preparation.PocketPublicJourneyTests`, `tests.service.test_installed_pocket.InstalledPocketJourney`, `tests.service.test_installed_activation`, `tests.service.test_installed_restart.TerminalRestart`, `tests.worker.test_engine_lifecycle.DispatcherEngineTests`.
5. **Five direct dispatcher cases PASS:** `test_serialized_responsive_cancel_unknown_and_retry`, `test_timely_completion_wins_cancel_and_lost_ack`, `test_late_or_stale_proposal_cannot_publish(False,False)` and `(True,True)`, `test_replacement_only_confirmed_terminal_increments_generation`. Each used its own TemporaryDirectory and `pair.__wrapped__` generator closed in finally. This interpreter also explicitly added candidate `/tests` to sys.path; no pytest collection/maintenance activation occurred.

Total **165 distinct unittest methods: 164 passing, one failing; plus five direct dispatcher cases passing.** Additional F6 no-hook/growth, F7 budget and F9 barrier observations are not added as unittest passes. The independent test interpreters overlapped in wall time; each fixture had its own temporary roots. The deterministic F9 probe subsequently ran separately, so the finding is not inferred merely from concurrent-suite timing. SQLite ResourceWarnings and the expected injected_close_unknown lifecycle diagnostic were not suppressed. r288's historical 80+32+6, mixed 46 (45 pass/one fixture error), then targeted two and five direct passes remain historical evidence, accurately distinguished from this fresh result.

`git diff --check bd85e1c..HEAD`: PASS. Candidate remains clean. r277 SHA256 `0d3f9110ee320f8bb1eecff3322ff1057de2a2765972e248acbf71f4dd1139c6`; `lib/tts.py` SHA256 `fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`. Known broader legacy digest RED remains unwaived, unregenerated and unexecuted here.

## Parent gates

Correct F9 and obtain fresh independent review before adopting the complete SOURCE slice. F6–F8 corrections and passing bounded components can be retained without asserting the failed journey is green. No source adoption was performed here.

Lot B remains native packaging/launcher/dependency closure, dedicated-account/private-directory trust, credential enrollment/rotation/loading, numeric endpoint provisioning, explicit bridge/firewall topology and owned-version onboarding. Lot C remains separately authorized actual Apple Silicon probe/load/generate/cancel/restart, process/FD/resource/durability, acoustic/performance and release qualification with unchanged headroom. Actual Metal remains UNAVAILABLE / UNQUALIFIED. Nonempty clone conditioning and arbitrary cancellation-point readiness remain UNKNOWN/unresolved. CPU Docker and CUDA remain separate mandatory release rows. No actual TLS, installed native activation, hardware or release authority follows from this SOURCE review.
