# r394 — public helper uncertainty and retained owner close candidate

2026-10-25. Implementation evidence candidate for ONE R393-E1 gap. No production
change or demonstrated production defect. Independent Sol review, then final
complete-family Astra review remain parent-owned. No self-adoption. Release
**NO-GO**; legacy **CLOSED**.

## Authority and pins

Own isolated checkout `/Users/ben/.txcode/worktrees/py-audio-box/t3-7652fe00`
was clean at exact base `1bec52ea8e1fedd8411099c0afc5001ff1c3031a`, then
fast-forwarded only this checkout to `37dd1d44fe1d18e4d2eb4d5b9f7570b4ff7de5ed`.
Source candidate **`820e2d7`** is its direct child; this report follows separately.
Read-only source `/Users/ben/.txcode/worktrees/py-audio-box/t3-a145b721` was
rechecked clean at 37dd1d4. No applicable AGENTS.md found. Memory registry search
returned no relevant hits; no historical fact controls this candidate.

Read FULL absolute ROOT `H3-EXPORT-ACTIVATION-REVIEW-r393-sol.md` and
`H3-FRESH-INSTALLED-EXPORT-CONTRACT-r378-astra.md`, and candidate
`H3-EXPORT-OWNER-ACTIVATION-r392-sol.md`. No ROOT/original writes, merge adoption,
push, delegation, review commissioning, or operational cleanup.

## Finite composition

Only `tests/service/test_export_owner_activation.py` changed. Existing owner
fixture gains an explicit test-only choice to use actual managed-reference acquire.
The new `ExportOwnerTests.test_public_helper_unknown_retains_owned_app_close`:

- Creates a fresh checked family with real tiny SQLite initialization, fake Docker,
  UID and immutable converter facts, then uses actual create_installed_app and its
  `_owned_app` lifespan to acquire the checked owner. Model mount/context resolution
  remains explicitly injected, as in the existing factory fixture.
- Uploads 1,152,044-byte managed PCM through authenticated public ASGI and submits
  public audio_export. Actual RuntimeSupervisor.verify_export calls actual
  managed_reference.acquire. Fake Popen and socket endpoints transfer a real
  read-only source FD; both waits raise unknown, and the fake handle remains
  retained with returncode None. No actual child or listener exists.
- Checks descriptor closure at the actual os.close call before the FD number can
  be reused; fake endpoints are closed. The socketpair injection is limited by
  worker-thread identity so asyncio retains its real local wakeup socketpairs.
  The existing worker fixture's test-only MSG_CMSG_CLOEXEC portability injection
  is retained for this host. Real acquire/verification are never replaced.
- Asserts worker quarantine, SQL dispatch `(unknown, runtime=None, receipt=None,
  outcome=None, acked=0)`, no runtime authorities/completions, unchanged worker
  journal, no publications/public artifact IDs, zero ACK and executing reservation.
  The preexisting dispatch Authority binding is retained; no runtime binding is
  invented and no worker journal allocation occurs.
- Requests ordinary lifespan shutdown with uncertainty still present. Actual
  Dispatcher.close times out with `control_close_timeout` through the real
  `_ThreadResult` cleanup operation. Owner, notification, error, close completion,
  lifecycle registry record and stop obligation remain retained. Owner thread is
  alive, notification is not closed, lifecycle is not marked closed, and competing
  Core acquisition specifically refuses `control_busy`. All uncertainty assertions
  are repeated after timeout. No UNKNOWN clearing, reap/success injection, close
  retry or production repair/unquarantine path is used.

## Retained test resource disposition

Only after those assertions a finite Event-controlled test-only `_tick` seam exits
this fixture's owner loop. It changes no SQL uncertainty/reservation/publication or
worker state. The owner thread joins within three seconds; every bounded-lifecycle
executor thread is asserted dead and every lifecycle operation result complete.
The failed lifecycle record is explicitly removed from the test registry, without
marking it successful or rerunning product close. The real notification is then
closed for fixture disposal. asyncio.run drains its default executor/wakeup
socketpairs. Session's existing temporary fixture cleanup disposes inert files.
Fake helper remains in the test-local retained list, unknown and never reaped;
it owns no OS process/thread. Fake socket handles were closed; transferred FD
closure is observed synchronously. No never-ending test thread is retained.

## Execution and failures

Read both conftests before application imports. No broad pytest or listener fixture.
Every run used `/private/tmp/pab-0c-venv/bin/python -I -B -` from stdin, asserting
CPython 3.14.7, setting exact own cwd and adding only own root/tests/service/tests.
Before application imports the audit hook blocked subprocess.Popen, os.system,
socket.bind and socket.connect. No disk runner/log was created. Raw receipts are
in this task's tool output. All counter-bearing runs reported forbidden events
`[]`; the diagnostic run without a counter had the same blocking hook.

Development failures are retained, not relabelled:

1. One-test ERROR, 2.077 s: global fake socketpair intercepted asyncio wakeup setup;
   corrected to worker-thread-only injection. No owner lifespan was started.
2. One-test FAIL, 5.445 s: factory uses its actual fixture credential, not `writer`;
   corrected credential and failure-path lifespan disposal. That run also emitted
   async-generator close timeout diagnostics during teardown.
3. One-test FAIL, 6.746 s: FD assertion initially inspected after other opens.
4. Five-test selection, 13.938 s: 4 PASS / new fixture FAIL at FD assertion.
   Passing affected tests were unknown-descendant, factory lifespan, direct close
   cancellation, and cancelled-acquisition close failure. The last emitted its
   existing injected_close_unknown shielded-future diagnostic, while passing.
5. One-test FAIL, 6.708 s, and diagnostic one-test FAIL, 6.740 s: close-call evidence
   showed no transferred FD had been reached. Captured prepare error correctly
   remained helper_unreaped; host lacks MSG_CMSG_CLOEXEC, so acquire had failed
   before recvmsg. Added the existing worker test's portability injection.
6. New fixture **1 PASS, 6.765 s**, forbidden `[]`.
7. Coherent focused selection **4 PASS, 12.383 s**, forbidden `[]`: new fixture;
   ExportOwnerTests.test_unknown_descendant_retains_capacity_until_cancel_reaped;
   ExportOwnerTests.test_factory_uses_same_lifespan_and_checked_configuration;
   ActivationTests.test_direct_cleanup_cancel_close_thread.
8. Tightened lifetime exclusion assertion from broad Exception to exact ValueError
   control_busy, then final new fixture **1 PASS, 6.894 s**, forbidden `[]`.

Final diff whitespace check passed. No unchanged 37-method rerun. No production
fix was justified by these fixture-development errors. Candidate supplies the
requested finite evidence; independent acceptance remains pending.

## Parent gates

Fresh independent Sol review of exact source candidate, then complete-family
Astra are parent-owned. All actual Docker/network/listener/application child,
FFmpeg/ML/hardware, installation/migration/build/download/acquisition, operational
cleanup, HELD/STT and heavy waiver work remained UNRUN. Seven real-child
publication tests remain BLOCKED/UNRUN. No fresh resource qualification claimed.
Actual Linux CPU, Intel Mac Docker Desktop and OrbStack qualification UNRUN;
CUDA and native Metal mandatory, unqualified/UNRUN. Release **NO-GO**;
legacy **CLOSED**.
