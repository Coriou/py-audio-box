# r360 Sol — fresh independent adversarial fix review

2026-10-25. **NO-GO for source adoption of unchanged e29af7892e6200fdaea5c1388c9e80bcf0fd227c.** The ordinary timeout path now retains exclusions through reap, and live output bounds and terminal EOF validation are implemented. Two ownership gaps remain: an interruption in the exceptional wait cadence still unwinds helper locks before reap; the new terminal rejection path leaves its attach child and pipe unmanaged. Independent final regression selection: **249 PASS in 91.746s, zero failures/errors/skips**. Green regression tests do not cover the defects below.

## Authority, pins and write scope

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-1e21c6da`: HEAD exactly `e29af7892e6200fdaea5c1388c9e80bcf0fd227c`, direct parent exactly `bc075ccc93510564f25c9788348c2ac0aa88df54` (r357), next ancestor exactly `45ce583b1f5c61e14c0218911a76ea68a6e928d9`. Candidate tracked/untracked status was empty before review and after tests. `git diff --check HEAD^ HEAD` passed. Candidate discovery and explicit ancestor checks found no applicable AGENTS.md.

Read full normative r356, candidate r357/r359 and root r358 reports. Verified normative SHA256 `a59710f5ac746974730c91f9144f6bb1a990083b1a6be312ce7daf3109e960da`, r357 `71a19a9d8f10463b7b2a0ca4bc1c54ff695cc304a86b5e7ff243099930b2b89a`, r359 `680f1a1f578f7f188f2d6dde3e4b88b829442242d782332bee727da57c975afa`. All four changed-source SHA256 pins in r359 match. Fifteen of nineteen r357 pins remain identical; only client/helper/tests/DOCKER-VOLUMES changed, as expected. Diff adds those four changes and r359 report only.

Sole persistent write: this mode-0600 report. No source edits, commits, merge, self-adoption or delegation. Memory registry search returned no relevant premise. No Docker, network/listener, actual application child, model/provider/hardware, HELD/STT, extracted-archive child, operational cleanup or heavy operation occurred. No 6-GiB waiver was inferred. Tiny fixture filesystem/SQLite/ASGI, inert pipes, fake processes and controlled threads only. Test fixture teardown is distinct from operational cleanup.

## Blocking findings

### R1 — P1: exceptional wait cadence can still unwind unreaped helper ownership (r358 F1 not fully closed)

`service/daemon_client.py:109-121`, especially the sleeps at 115 and 121, are outside a surrounding interruption-retention guard. The `except BaseException` catches an exception from `p.wait`, then calls `time.sleep(.1)` inside its handler. A KeyboardInterrupt during that cadence escapes immediately. The noninteger-wait sleep has the same issue: its first interruption is caught, but the handler's sleep is unguarded. Cleanup closes at 105/108 are also outside an independent ownership-finalization guard; this review's concrete reproduction uses the wait cadence.

Independent inert reproduction at the actual helper boundary:

1. Construct `test_daemon_volumes.Session`, run `install`, set `s.child=daemon_helper.child`.
2. Use `AdapterBoundsTests.process()` for real inert pipes/fake process; set `p.wait=Mock(side_effect=OSError('inert wait error'))` and leave `p.returncode=None`.
3. Patch helper Popen to that process and `daemon_client.time.sleep` to raise `KeyboardInterrupt('inert cadence interruption')`; run `s.run('init')`.
4. KeyboardInterrupt escapes after exactly **one wait call**. Independent nonblocking flock attempts immediately acquire **authority, control, receipts, staging and IPC**. `p.returncode` remains **None** and receipts `p1-core-intent` remains **present**.

The lock-level probe passed in **0.265s** by demonstrating this defect. A separate direct collector probe also observed one wait call and unknown completion. No real process/signal was used; these are explicit fault injections of the adapter's documented exceptional paths. This does not establish that ordinary wait always fails, or that a second Core runs. It establishes that the claimed interruption-safe ownership fence is incomplete. Durable intent remains useful but does not replace parent exclusion through actual exit/reap.

Retain the same one child and exclusions across interruption of every ownership-wait/cleanup transition. Preserve cadence, bounded resources, no KILL/replacement/further work/success/intent clearing. The orchestrator permits indefinite parking; this finding does not object to its duration. Add a regression at this exact helper boundary, including interruption during the exceptional cadence.

### R2 — P2: terminal rejection leaves live attach process/pipe without cleanup or reap

`service/daemon_client.py:246-280` appends the Docker attach process to `self.retained` but has no enclosing exception/finalization path. The newly added `wire.finish()` at 268 can raise before the only `p.wait` at 269. Pending/unread junk and terminal EOF timeout therefore return an error while leaving stdout open, no wait/reap and a retained process reference. Even success never removes the exchange reference or explicitly closes stdout. `Docker.call` now has cleanup/removal, but exchange does not use it.

Independent inert reproduction: create a Session configuration and valid running helper inspection; preload real pipe stdout with matching nonce/phase/step-2 complete frame followed by `JUNK`; patch Popen to a fake process with `returncode=None` and mocked wait; call actual `Docker.exchange`. It correctly raises `daemon_trailing_protocol`, but afterward **wait calls=0, stdout.closed=False, len(d.retained)=1, returncode=None**. The probe passed. Existing test fixtures close these streams in test teardown, hiding production ownership cleanup.

A real attach producer can remain alive or block on its now-undrained stdout after rejection. In an embedding caller, repeated refused exchanges can retain processes/descriptors; a standalone client's eventual OS teardown does not implement explicit adapter ownership. This is a client attach-process issue, not proof that helper volume locks are released or that another Core exists. Do not stop/kill the helper as a workaround. Close protocol input, arrange bounded drain/discard or safe output-reader closure, and retain responsibility for the exact attach process through verified exit/reap on every exception. Preserve incomplete operation receipts and helper authority; do not create replacement work or success. Test pending bytes, subsequent bytes, EOF timeout, send/receive/setup faults and successful reference/stream release.

## r358 finding disposition and bounded source assessment

- **F1: PARTIAL / still blocking via R1.** Ordinary timeout and overflow now remain sticky, drain and wait for integer status while all five exclusions remain held in the new regression. Setup failure/noninteger wait coverage exists. The exceptional cadence remains an escape path.
- **F2: original defect corrected at SOURCE.** Both adapters use live PIPE collection, no temporary stdout spool. Retained payload is capped at 1,048,576 bytes for Docker calls and 65,536 for helper children, with at most a 65,536-byte read chunk. Input slice, output conversion/decoding, allocator/selector state and kernel pipe buffers add bounded overhead; the limit is not a total-memory ceiling. Failure clears payload and discards subsequent output. stderr is DEVNULL. No unbounded disk sink was found. Lifetime obligations still need R1/R2 correction.
- **F3: original trailing acceptance corrected at SOURCE.** Pending bytes, a second frame and subsequent bytes reject; exact EOF is required before wait/final inspection/success. A live writer without EOF times out. Parsing no longer silently accepts the demonstrated trailing output. R2 prevents accepting the complete adapter lifecycle as finished.

No change in this five-file diff weakens strict old-host policy, existing Core/persistence/intent/ACK/reap/publication validators or explicit eight-volume topology. Fifteen unchanged source pins and broad negative regression coverage support that bounded conclusion; this is not a new exhaustive audit of all historical source. No actual second Core, false success, secret disclosure or publication bypass was demonstrated. Shared collector setup faults close readers and wait in the covered path; ordinary selector timeouts have cadence and one owned child. A continuously productive drain performs I/O work, rather than accumulating payload. Unknown wait may park indefinitely under the accepted decision; exceptional ownership must remain intact throughout.

## Provenance correction

`H3-CONTAINER-INSTALL-FIX-r359-sol.md:43-44` says “The user explicitly resolved” the finite-return tension. The current commission explicitly identifies this as an **orchestrator-resolved decision**. Correct that attribution. Its technical content is accepted for this review: ONE child, bounded resources/cadence, no busy spin/replacement/further work/success/intent clear, locks retained until actual integer wait/reap, no force KILL or rollback. Caller timeout is not helper completion. No new user design decision is requested here.

## Independent verification receipts

Every runner used `/private/tmp/pab-0c-venv/bin/python -I -B`, asserted CPython **3.14.7**, and inserted absolute candidate root, tests and tests/service paths. Audit hooks were installed before application/test imports and reject actual `subprocess.Popen`, `os.system`, `socket.bind`, `socket.connect`. Both conftests and import/subprocess/socket seams were inspected. Direct unittest selection avoids pytest/conftest activation and its listener fixture. Static archive assertions only; no extracted application child. No audit rejection occurred.

One coherent final run: **249 tests in 91.746 seconds, 249 PASS, 0 failures, 0 errors, 0 skips**. This includes the focused ten-method AdapterBoundsTests (seven new, three inherited) plus the exact r358 broad allowlist. No separate duplicate focused rerun is claimed. The intentional `injected_close_unknown` shielded-future diagnostic appeared and its test passed; it does not signify successful close.

Exact selections:

```text
tests.service.test_daemon_volumes.AdapterBoundsTests
tests.service.test_daemon_volumes.DaemonTests
tests.service.test_daemon_volumes.ProtocolTests
tests.service.test_daemon_volumes.ChildAndClientTests
tests.service.test_daemon_volumes.DaemonPublicTests
tests.service.test_daemon_volumes.FailureAndPackagingTests
tests.service.test_split_init.SplitInit
tests.service.test_install_child.Canonicalization
tests.service.test_install_ownership.InstallOwnership
tests.service.test_installed_host_policy.InstalledHost
tests.service.test_installed_registration.Registration
tests.service.test_installed_registration.ChildTransport
tests.release.test_headless_install.GateTests
tests.service.test_installed_activation.ActivationTests
tests.service.test_installed_activation.PrivateInputs
tests.service.test_installed_preset.InstalledPresetTests
tests.service.test_installed_restart.Restart
tests.service.test_installed_restart.TerminalRestart
tests.service.test_installed_restart.ClosedRestart
tests.service.test_host_install.HostInstall
tests.service.test_public_job_control.PublicJobControlTests
tests.release.test_release_package.CoordinatorTests
tests.release.test_release_package.CLIResultPublicationTests
tests.release.test_release_package.InstalledCommandTests
tests.release.test_release_package.GeneratedTopologyTests
tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change
tests.release.test_release_package.ArchiveTests.test_release_context_and_cpu_base
```

Additional independent defect-demonstration receipts, separate from the 249 regression tests: **2 PASS in 0.019s** (exchange leak and direct cadence interruption), then **1 PASS in 0.265s** (actual helper/initializer lock boundary). Probe code ran on stdin; no script or extra receipt file was persisted. Raw counts/observations are in this thread's tool outputs. No failed runner preparation, retry or corrective source mutation occurred during this review.

## Remaining parent-owned gates

Correct R1/R2 in a newly pinned candidate and obtain fresh independent review. Fresh Astra assessment and parent adoption remain parent-owned; this report adopts nothing. The r359 historical 38-PASS receipt remains separate; this review supplies the new 249-green receipt without rewriting earlier history.

Actual rootful Linux/amd64, THIS Intel Mac Docker Desktop and THIS Intel Mac OrbStack installations remain separately **UNRUN**. Python 3.11/package/image/dependency execution, real Docker transport, actual pipe/process reap, volume/mount/inode continuity, helper/child UID/capability enforcement, denied peers, engine isolation, final worker PID1 descendant reap, authenticated audible Kokoro/artifact bytes, replay/restart, loopback forwarding and measured resources remain unqualified. CUDA and native Metal compatible rows remain mandatory and unqualified. **Release NO-GO; legacy CLOSED.** Accepted native freshness/source epochs remain unchanged. No operational qualification follows from these SOURCE fixtures.
