# r395 — independent composed helper uncertainty review

2026-10-25. **Bounded SOURCE GO: R393-E1 closed.** No new acceptance-blocking finding or demonstrated production defect in this test-only delta. Full-family r378 acceptance remains pending parent-owned final Astra review. Release **NO-GO**; legacy **CLOSED**. No adoption, merge, activation or operational qualification authority.

## Authority and exact candidate

Read-only candidate `/Users/ben/.txcode/worktrees/py-audio-box/t3-7652fe00` verified clean at `14f5fd4bd4cdb2ff5d15085809a6046d4cd1de67`. Direct chain verified:

- HEAD report-only r394 -> `820e2d7f3fe28cea2a59965a5173e4e1fdd4ec20`.
- 820e2d7 test-only -> `37dd1d44fe1d18e4d2eb4d5b9f7570b4ff7de5ed` r392 report.
- Complete linear parent chain through owner candidate `48da311dfce93f3a5a6ad09dd1448ee251e3153c` to base `1bec52ea8e1fedd8411099c0afc5001ff1c3031a` inspected. Base-to-HEAD diff whitespace check passed.

Read FULL absolute ROOT r393 `H3-EXPORT-ACTIVATION-REVIEW-r393-sol.md` and normative `H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md`; read FULL candidate r394 and r392. Inspected complete test diff and actual acquire, verify_export, Dispatcher prepare-failure/unknown/close loop, owned-app lifecycle and thread completion/disposal paths. No applicable AGENTS.md found in candidate descendants, checked candidate/ROOT ancestor paths or ROOT docs destination tree. Memory quick search supplied no task-specific evidence; no historical memory fact controls this decision.

Sole persistent write is this ROOT report, mode **0600**. No source/runner/log writes, source changes, commit, merge, delegation or self-adoption.

## R393-E1 disposition

**Closed at the authorized inert SOURCE evidence tier.** `tests/service/test_export_owner_activation.py:298-429` composes the previously separate seams. The `actual_reference=True` option at lines 71 and 118 allows real managed_reference.acquire; it does not substitute a helper refusal. The runtime adapter calls actual sealed RuntimeSupervisor.verify_export before its fake runtime allocation. Public authenticated ASGI upload/export, checked fresh Core/owner acquisition and ordinary installed-app lifespan shutdown run against real tiny SQLite and actual owner threads.

The helper socket endpoints and Popen are mocks, but the transferred source FD is a real O_RDONLY descriptor. Actual acquire receives ancillary FD bytes, encounters two injected unknown waits, invokes kill once, retains the fake process and closes the FD in finally. Synchronous close instrumentation observes EBADF before descriptor-number reuse. Both mock endpoints receive close calls. Real verify_export catches the acquire failure and sets supervisor failure/quarantine. It does not reach source use or allocate a worker runtime. The FD check after confirmed helper reap is deliberately unreachable in this unknown-wait case; this fixture does not newly prove successful helper validation.

The real Dispatcher consumes prepare failure and retains SQL `(phase='unknown', runtime=None, receipt=None, outcome=None, acked=0)`. Assertions cover executing reservation, empty public artifact IDs, zero publication rows, zero ACKs, no fake runtime authorities/completions, unchanged worker journal, retained fake handle with returncode None and quarantined worker hello. The dispatch Authority already exists; no runtime ID, journal allocation or productive completion is invented. The adapter is not actual RuntimeSupervisor.prepare/transport, but inspection corroborates the same relevant ordering: actual supervisor verifies export before allocating journal identity. This is the requested composed source path, not real process qualification.

Ordinary lifespan exit invokes actual Dispatcher.close via actual _ThreadResult. With the unknown row still present, close raises `control_close_timeout`; the fixture checks the retained lifecycle registry record, exact owner/notification identity, error, completed failed close operation, cleanup_requested, closed=False, notification not closed, owner thread alive and stop obligation. Competing Core construction refuses specifically ValueError/control_busy, demonstrating retained Core lifetime exclusion. The full uncertainty assertions run again after timeout. No UNKNOWN clearing, fake receipt/reap, ACK, reservation discharge, automatic close retry or success marking enables these assertions. The composed case covers timeout; the selected existing direct-close cancellation test separately covers lifecycle cancellation, not cancellation of this same helper-unknown scenario. R393-E1 does not require another fault campaign.

## Disposal: what it proves and does not prove

Only after retention assertions, the Event-controlled patched _tick sets owner.stopping and returns. That bypass deliberately ends the test owner loop despite its unresolved durable row. Actual Dispatcher finally shuts down its pool and closes Core; the fixture joins owner, checks pool threads dead and lifecycle operations done, removes the failed registry record explicitly, and closes notification explicitly. These are test disposal actions, not successful product close, safe operational release, recovery or a supported way to clear uncertainty. In particular, Core lifetime exclusion ends during disposal; its earlier retained state is the evidence under review. SQL UNKNOWN, unacked work and helper uncertainty are not rewritten into success to justify close.

The fake helper remains in the test-local retained list with unknown returncode after disposal, never reaped. It owns no actual process or OS handle. Mock socket close calls establish branch execution only, not actual IPC teardown. The real transferred FD is closed. asyncio.run drains its default executor/wakeup socketpairs; existing Session cleanup disposes inert temporary files. Independent post-suite enumeration found **only MainThread**, corroborating no surviving test thread. This does not establish operational resource reclamation under a real helper fault. No durable post-disposal recovery claim is made or needed.

## One coherent independent execution

Read `tests/conftest.py` and `tests/service/conftest.py` before imports. The pytest maintenance guard and actual listener/uvicorn fixture were neither collected nor invoked. Exactly one stdin direct-unittest execution used:

`/private/tmp/pab-0c-venv/bin/python -I -B -`

Before application/test imports: import sys; install audit hook recording and raising on subprocess.Popen, os.system, socket.bind and socket.connect; assert CPython `(3,14,7)`. Then set cwd to exact candidate and prepend only candidate root, tests/service and tests. Assert selection count four; failfast runner. No retries, probes, broad pytest or unchanged 37-method rerun.

Exact allowlist:

- tests.service.test_export_owner_activation.ExportOwnerTests.test_public_helper_unknown_retains_owned_app_close
- tests.service.test_export_owner_activation.ExportOwnerTests.test_unknown_descendant_retains_capacity_until_cancel_reaped
- tests.service.test_export_owner_activation.ExportOwnerTests.test_factory_uses_same_lifespan_and_checked_configuration
- tests.service.test_installed_activation.ActivationTests.test_direct_cleanup_cancel_close_thread

**4 PASS, 12.452 s, zero failures/errors, exit 0; forbidden attempts `[]`; CPython 3.14.7. Remaining threads: `[('MainThread', False)]`.** Raw receipt retained in review-turn tool output; no disk log/runner. This is a new coherent current four-test result. Historical r394 four PASS/12.383 s followed by tightened final one PASS/6.894 s remain distinct; neither is relabelled as a coherent final four-test receipt.

## Remaining parent gates

R393's bounded owner SOURCE GO remains bounded. This review accepts the one missing E1 fixture and finds no need for production changes. Final complete-family Astra review and adoption remain parent-owned after findings resolution; not commissioned here. No full-family/release GO is granted by these four tests.

No actual Docker/network/listener/application child/FFmpeg/ML/hardware, HELD/STT access, installation/migration/build/download/acquisition, operational cleanup or heavy waiver. No fresh disk/RAM observation. Seven real-child reference-publication tests remain **BLOCKED/UNRUN**. Actual converter/image pins, complete alias-free runtime closure, measured profiles, installed CPU/runtime and resource qualification remain **UNRUN**. Linux CPU, Intel Mac Docker Desktop and OrbStack remain **UNRUN**. CUDA and native Metal remain **mandatory, unqualified/UNRUN**. **R393-E1 SOURCE GO/closed; final family review pending; release NO-GO; legacy CLOSED.**
