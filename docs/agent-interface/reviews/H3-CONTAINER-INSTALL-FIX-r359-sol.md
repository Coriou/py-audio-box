# r359 Sol — narrow r358 F1–F3 implementation candidate

2026-10-25. Corrections delivered for fresh independent review and Astra global
assessment. No self-adoption or SOURCE/release GO. Actual rootful Linux/amd64,
THIS Intel Mac Docker Desktop and THIS Intel Mac OrbStack remain separate UNRUN
rows. CUDA and native Metal remain mandatory and UNQUALIFIED; release NO-GO;
legacy CLOSED. Accepted native freshness/source epochs remain unchanged.

## Authority and exact pins

Implementation worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-1e21c6da`.
Verified clean base `45ce583b1f5c61e14c0218911a76ea68a6e928d9`, verified candidate
`bc075ccc93510564f25c9788348c2ac0aa88df54` has exactly that direct parent, then
fast-forwarded only this worktree using `git merge --ff-only` to that candidate.
The final commit containing this report has that candidate as its direct parent;
resolve its exact ID from the returned commit, avoiding a self-referential hash.
Original candidate/root were never mutated. No push, adoption, delegation or
additional scheduler/store. Bounded repository and explicit ancestor checks found
no applicable AGENTS.md. A memory registry search supplied no relevant premise.

Read full root r358 and normative candidate r356/r357. SHA256:

```
2304f26c6547e50368b1706a22b82b99e2b0a47ed590402963cdb0d9663fe4a3  root H3-CONTAINER-INSTALL-REVIEW-r358-sol.md
a59710f5ac746974730c91f9144f6bb1a990083b1a6be312ce7daf3109e960da  DOCKER-CONVENIENCE-CONTRACT-r356-astra.md
71a19a9d8f10463b7b2a0ca4bc1c54ff695cc304a86b5e7ff243099930b2b89a  H3-CONTAINER-MANAGED-INSTALL-r357-sol.md
```

The unchanged r357 is r358 NO-GO despite its independent reported 239 PASS in
89.949s. That historical receipt is not a test run of this corrected candidate.

## Responses and limits

**F1:** `daemon_helper.child` uses synchronous pipe collection shared with the
client. Timeout/overflow become sticky failure; output is discarded thereafter,
stdin closes, stdout drains, and actual `wait` must return an integer exit status
before the adapter can raise through the helper/initializer exclusions. Unknown
wait results and timeout transitions do not release ownership. No child kill,
replacement, further phase, success publication or intent clearing is introduced.
The same adapter serves Core, hello and registration writer. Existing Core,
intent, lane/Core/root lock ordering and quarantine remain unchanged.

The user explicitly resolved the finite-return tension in this session: retain
helper and locks until actual exit/reap, with bounded resources and no further
work. Thus exceptional retention has **unbounded duration**, even after the
30-second child collection deadline. One child remains owned by that helper;
selector and wait calls use 0.1-second intervals, not a busy wait. A caller may
separately time out; this is not helper completion. Existing exact helper IDs,
client receipts, helper authority and incomplete intent remain the retained
identity/refusal mechanism. Manual intervention may be needed; no automated
intervention or reset is supplied. Helper PID1 death is not substituted for reap.

**F2:** Both actual adapters use PIPE stdout and no temporary output file. The
Docker call collector retains at most 1,048,576 payload bytes; helper child retains
at most 65,536 bytes. One read chunk is at most 65,536 bytes; request input is
already bounded (child encoded request at most 65,536 bytes). Python object/
allocator overhead, fixed OS pipe capacity and selector state are additional
bounded overhead, not claimed to fit inside the payload limits. Overflow clears
the collected payload, closes input and continues discard/drain through EOF and
reap. Timeout does the same. On adapter/interruption faults, closing the output
reader gives a writer EPIPE rather than an undrained full pipe, then ownership
still waits for reap. Failed/unknown children cannot accumulate across calls:
there is no return or replacement until reap; completed Docker.call references
are removed. No stderr or child output is included in public failure text.
Docker call's success window is at most 15 seconds and the remaining 120-second
client operation deadline. The exceptional ownership wait can exceed both.

**F3:** After the single complete frame, `Docker.exchange` closes input and calls
`Wire.finish` before exit/container/identity success checks. Pending bytes reject
immediately; otherwise a one-byte read must establish exact EOF within 10 seconds
and the remaining operation deadline. Any byte, including another valid frame,
rejects. EOF timeout rejects. Existing nonce/sequence/closed-field checks remain.
This is a finite terminal protocol check, not runtime readiness qualification.

Only `daemon_client`, `daemon_helper`, their focused tests and the new-lane user
instructions change, plus this report. Strict old-host routing, shared Core/
publication/reap validators, public authority schemas and packaging entries do not
change. Deterministic packaging tests still inspect the changed module bytes.

## Verification receipts

Both conftests/import seams were inspected; direct explicit unittest selections
avoid pytest/conftest activation and extracted archive subprocess tests. Every
test runner used `/private/tmp/pab-0c-venv/bin/python -I -B`, asserted CPython
3.14.7 and inserted these explicit absolute import paths: this worktree, its
`tests`, and `tests/service`. Audit hooks reject actual `subprocess.Popen`,
`os.system`, socket bind and socket connect. Only inert pipes, fake Popen/wait,
controlled producer threads, tiny files/SQLite and existing in-process ASGI
fixtures were used. No Docker/build/pull/load/network/listener, real application
child, ML/provider/hardware, HELD/STT or operational cleanup occurred.

Final selection: **38 PASS in 22.817s; 0 failures/errors/skips**. Exact classes:

```
tests.service.test_daemon_volumes.AdapterBoundsTests
tests.service.test_daemon_volumes.DaemonTests
tests.service.test_daemon_volumes.ProtocolTests
tests.service.test_daemon_volumes.ChildAndClientTests
tests.service.test_daemon_volumes.DaemonPublicTests
tests.service.test_daemon_volumes.FailureAndPackagingTests
```

AdapterBoundsTests has seven new methods and three inherited protocol methods.
It demonstrates timeout at the actual helper adapter while nonblocking flock
attempts on authority/control/receipts/staging/IPC all refuse at each of three
fake waits (two TimeoutExpired then verified reap); locks are acquirable only
after return and the Core intent remains. Actual client timeout similarly waits
through controlled transitions. Both actual adapters receive more than their
limits from pipe producers: capped bytearray instrumentation checks retained
payload, producers finish, and wait records reap before error. Good outputs and
nonzero exits are covered. Setup faults and a nonterminal wait result cannot
unwind ownership. Actual exchange tests cover pending junk, pending second frame,
unread trailing junk (forced separate OS read), and exact good EOF. A live writer
without EOF triggers the bounded terminal timeout. These fake waits do not prove
OS process or kernel behavior.

Final receipt `/private/tmp/r359-final-focused-2.txt`, SHA256
`8869a84ab337c6303e39d709d9537a6d99f81b80f4fb46db3e20d2f697158b46`.
`git diff --check` passed. The full r358 239 allowlist was not rerun; no new
239-green receipt is claimed.

Earlier local attempts remain distinct and retained:

- `/private/tmp/r359-focused.txt`: incomplete after two PASS. A fake-wait assertion
  used the wrong intent root (control instead of receipts), so ownership correctly
  stayed parked. Stopped only that test-runner process with TERM, then corrected
  the fixture. No application child existed. SHA256
  `d95ae384e0e738d2bb3db6debcc7a49124fd1748229f4918bf0cb0f6ffa52a1e`.
- `/private/tmp/r359-focused-2.txt`: 8 PASS in 0.297s, before subsequent collector
  setup/unknown-wait hardening and two further methods. SHA256
  `ded92e1d56b3eb6ce1234e07d4d9cb4ec3860b672fbbbd3c69f726758b167ef2`.
- `/private/tmp/r359-final-focused.txt`: incomplete after one PASS. The overflow
  fake wait set returncode but omitted the real wait API's integer return, so the
  hardened adapter correctly retained unknown completion. Stopped only that
  test-runner process with TERM and corrected the fake return. SHA256
  `f9bc28fbd512cd010d9eb4f002b9a3f97900ac9a732a7730493f9ed6acce7440`.

## Final changed-source SHA256

```
f5ce42d6231ee35734449c56816b45ad977828c78837d93437ca91d67b845bbb  service/daemon_client.py
9c445b7c0f9dea22bbcb31a7ac5c6685109c1bcf60dea0ba37f06226e390b9ec  service/daemon_helper.py
1931c87a151c1473c58c7ee10fcd951ad45ac63b68e0692316281f4ae62f7bc4  tests/service/test_daemon_volumes.py
61f24049d106546bbc97745369099af35af2513f79578c8573228b827ca4aedd  docs/agent-interface/DOCKER-VOLUMES.md
```

## Remaining gates

Fresh independent review and Astra global assessment must judge these corrections
and the explicit exceptional unbounded ownership duration before parent adoption.
This implementation does not self-accept F1–F3 as independently cleared. Manual
intervention limitations, actual OS reap/pipe/UID/capability behavior, packaged
Python 3.11/image/dependencies and Docker transport remain unqualified. No install,
audio, VM restart, volume identity, engine isolation, measured resources or real
runtime cancellation/reap/publication claim follows from these SOURCE fixtures.
Actual Linux/Desktop/OrbStack rows remain UNRUN; CUDA/Metal mandatory unqualified;
release NO-GO, legacy CLOSED. No operational permission or heavy-operation gate
waiver is supplied.
