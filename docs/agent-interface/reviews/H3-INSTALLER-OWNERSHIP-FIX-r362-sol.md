# r362 Sol — finite local installer ownership correction

2026-10-25. Candidate for fresh independent review, then Astra assessment and parent adoption. No self-adoption or SOURCE/release GO. Actual Linux/amd64, THIS Intel Mac Docker Desktop and OrbStack remain separate UNRUN rows; CUDA/Metal mandatory and unqualified; release NO-GO; legacy CLOSED.

## Authority and pins

Only own isolated worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-ad4c73be`, branch `t3/h3-installer-ownership`, was changed. Initial status was clean at `45ce583b1f5c61e14c0218911a76ea68a6e928d9`; verified source worktree HEAD `e29af7892e6200fdaea5c1388c9e80bcf0fd227c`, then fast-forwarded this worktree to it. Verified ancestry e29af789 → `bc075ccc93510564f25c9788348c2ac0aa88df54` → 45ce583b. The commit containing this report has e29af789 as direct parent; its exact hash is supplied with the handoff, avoiding a self-referential hash.

Read full root r361/r360/r358 and normative r356, candidate r357/r359. Root r361 SHA256 `f2edf188a7cd03cdc3a343c094b6ad9568f7a1a0d485da7981da07b08858790b`; root r360 `5a46e4289fb8dd8cd690c7580a2b715a018de4d872eadf9512864cd0fbdb054c`. Repository and explicit ancestor AGENTS checks found none. Memory registry keyword lookup returned no relevant premise. No root/source-candidate writes, push, delegation, adoption or operational cleanup. The authorized initial fast-forward was the only merge operation.

## Bounded correction

`daemon_client.Owner` provides synchronous local Popen custody for call, exchange, notify and helper child (the last through collect). Ordinary deadlines are computed before spawning where practical. Reference insertion is guarded; setup, protocol, selector/stream cleanup, clocks, wait and cadence faults become sticky failure. The first exception is retained without its traceback/context/cause; repeated faults do not append exceptions or logs. No per-retry process, payload or traceback accumulation is introduced.

Each stream is resolved independently through its object. Close-before-completion faults retry; close-then-raise checks the object's closed state on the next pass, without closing a remembered numeric descriptor. Selectors remain owned until close completes. Reap and resource disposition are separate obligations, and retained references are removed only after both. Wait uses the normal synchronous Popen contract, with no competing reaper; a production wait integer is the collected status, including signaled exits, which fail. No synthetic returncode, waitpid replacement, signals or background reaper is added.

Immediate wait errors/noninteger results and cleanup retries pass through guarded monotonic cadence. A cadence interruption resumes its remaining interval and cannot unwind exclusions. Ordinary deadline expiry stays a failure even if eventual exit is zero. Exceptional ownership duration remains unbounded under the **orchestrator-resolved** decision; r359 attribution is corrected without rewriting its historical test receipts. Finite progress-ending fault sequences are the acceptance target, not SIGKILL/crash/os._exit or infinite adversarial bytecode interruption.

Collection retains the existing live PIPE payload limits (client 1 MiB/helper 64 KiB), fixed 64-KiB read chunks and no spool; overflow/timeout discard subsequent output. Other adapter faults close readers safely and still wait. Notify now uses that collector with bounded input and absent stdout, rather than a separate communicate-timeout escape. Nested action ownership finalizes before enclosing attach ownership unwinds.

Exchange encloses Wire construction, send/receive/actions, stdin closure and exact terminal EOF in local attach custody. Success also closes streams and releases the reference. Final helper inspection/check runs after attach finalization. Exact nonce/phase/frame/EOF, same helper ID/image/mount/incarnation and daemon checks remain. Reaping the local Docker CLI does not establish helper completion; helper-running/nonzero final states still refuse. No helper kill, replacement, rollback, intent clearing or synthetic terminal receipt is introduced. Core persistence/intents/ACK/reap/publication, lock order and strict old-host validators are unchanged.

## Finite regressions

Five new explicitly selected methods cover:

1. Actual helper init wait error or noninteger wait → cadence KeyboardInterrupt → wait timeout → modeled reap. All authority/control/receipts/staging/IPC roots remain contested at every pre-reap observation and become acquirable afterward; one spawn, retained intent, no enrollment/next phase.
2. Independent stdin and stdout close failures before completion and after closing, another resource serviced, retained reference through reap, sticky first error, and controlled-clock remaining cadence (.1 then .06 seconds).
3. Exchange Wire construction/send/receive/action/finish failures, actual terminal EOF timeout, final inspection error, helper still running/nonzero, attach nonzero/signaled and full success. Assertions run before fixture teardown. Existing pending junk/second frame/separately read trailing-byte cases now assert reap, closed streams and empty references too.
4. Notify success/deadline timeout/interruption through the common collector nested under an attach owner; action finalizes before attach, both references release, no helper operation/replacement.
5. Selector close-then-raise retry and reference append-then-raise finalization. Existing setup failure and all live overflow/exact EOF/ordinary timeout/good-output tests remain selected.

Both conftests and selected import/subprocess/socket seams were inspected. Direct unittest avoids pytest listener/maintenance fixture activation. Every runner used `/private/tmp/pab-0c-venv/bin/python -I -B`, asserted CPython 3.14.7, inserted absolute candidate/tests/tests-service paths, and installed audit rejection for subprocess.Popen, os.system, socket.bind/connect before application imports. Tiny inert pipes/fake processes/controlled threads/files/SQLite/in-process ASGI only. No actual Docker/build/pull/load/network/listener/application child/ML/hardware/HELD/STT or heavy-operation waiver.

Development receipts are separate: initial 15 PASS in 2.132s; after selector hardening, one runner import failed with SyntaxError (zero tests loaded), corrected locally; then 15 PASS in 2.212s; after extending both-stream closure and exchange EOF-timeout coverage, 15 PASS in 2.234s. Retained `/private/tmp/r362-focused.txt`, `r362-focused-2.txt`, `r362-focused-3.txt`, `r362-focused-4.txt`. No test-runner termination or audit rejection occurred.

Final coherent selection ran **once: 254 tests in 92.026s, 254 PASS, zero failures/errors/skips**: exact r360 249 allowlist below plus the five new OwnershipTests methods listed above (selected individually, without inherited duplication). Receipt `/private/tmp/r362-final.txt`, SHA256 `14087eb27881648a43f09b2f0c6ec6655930851cad4c5877c08255f197c354d4`. Runner `/private/tmp/r362-run.py`, SHA256 `90ced5e711b121f4ab02c8fc9f62cac8c016a1879e1e1fd49c2f647a01a1a1b2`. The intentional injected_close_unknown shielded-future diagnostic appeared in its passing existing test; it is not a successful-close receipt. `git diff --check` passed. No source edits followed this final run.

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

Final source SHA256:

```
6dad24cb20dbc3f0d5878e69b305cd5bb1d7607ee1e6ea0b005551a52e012844  service/daemon_client.py
369e50864de1c434b819adefb4eb52a3a83da6312eb5f227e1ec0fa18fa8709e  service/daemon_helper.py
b8b7812483410be5853a3077deb8e5b0d2fd3b155e8ed8dd03effdf4eb1c2161  tests/service/test_daemon_volumes.py
8742d6746d5addbbd40dfa6933c85c0ccfde64a75102f62104063aefdffaa525  docs/agent-interface/reviews/H3-CONTAINER-INSTALL-FIX-r359-sol.md
```

## Remaining parent-owned gates

Fresh independent review must judge the exact candidate and finite r361 conditions; fresh Astra assessment and adoption remain parent-owned. These fixtures do not establish actual OS process/pipe/reap/UID/capability enforcement, packaged Python 3.11/dependencies/images, Docker transport, volume identity/restart, audible Kokoro/artifact bytes, runtime descendant isolation or measured resources. Linux/amd64, THIS Intel Mac Desktop and OrbStack remain separately UNRUN. CUDA and native Metal remain mandatory unqualified release rows. Accepted native freshness/source epochs remain unchanged; headless HTTP/CLI/MCP scope retained. Release NO-GO; legacy CLOSED. No cleanup, installation or heavy-operation authority follows from this report.
