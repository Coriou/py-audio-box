# r358 Sol — independent adversarial container-install review

2026-10-25. **NO-GO for source adoption of the unchanged candidate.** Three concrete gaps remain below. The coherent final allowlist is now **239 PASS, 0 failures, 0 errors, 0 skips**, but does not cover these gaps. Three additional inert adversarial probes passed by demonstrating the defects. No adoption, merge or source correction was performed.

## Exact authority and pins

Reviewed clean worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-b6d61621`, HEAD `bc075ccc93510564f25c9788348c2ac0aa88df54`; its direct parent is exactly `45ce583b1f5c61e14c0218911a76ea68a6e928d9`. Verified these before review. All 19 final file SHA256 pins in r357 match candidate bytes. Normative r356 SHA256 is `a59710f5ac746974730c91f9144f6bb1a990083b1a6be312ce7daf3109e960da`; r357 report SHA256 is `71a19a9d8f10463b7b2a0ca4bc1c54ff695cc304a86b5e7ff243099930b2b89a`. The prior r358 provider-limit attempt supplied no findings; none were inherited.

Read the full normative r356 and candidate r357 reports, PRODUCT, Docker-volume journey, strict CPU/HEADLESS install documentation, changed code/diff, helper/client/runtime and affected initialization/registration/activation/history/package seams. No applicable AGENTS.md was found in candidate discovery and explicit ancestor checks. Memory was used only for prior evidence/ownership discipline; current source and this commission govern this review.

Sole persistent write: this mode-0600 report. No candidate/root source mutation, commit, merge, delegation, Docker, network, listener, real application child, ML/provider/hardware, HELD/STT access or operational cleanup. Tiny disposable fixture files/SQLite and in-process ASGI/fake child/daemon functions only. No heavy operation or 6-GiB gate waiver. No pytest run or conftest fixture activation. Both conftests were inspected first; the listener fixture was excluded.

## Blocking findings

### F1 — P1: timeout releases parent exclusions before child exit/reap

`service/daemon_helper.py:56-60` raises immediately when `communicate` times out. It neither waits/reaps the child nor retains the enclosing helper transaction/exclusions until that obligation is discharged. The exception unwinds `initialize_split`'s ExitStack (including root/lane exclusions) and `Helper.held` (`service/daemon_helper.py:214`), before `main` exits. The same child adapter serves the registration writer.

Reproduction executed without a child: create `test_daemon_volumes.Session`, run `install`, set `s.child = daemon_helper.child`, patch `daemon_helper.subprocess.Popen` to a Mock whose `returncode=None` and `communicate` raises `TimeoutExpired`, then run `init`. It raises `daemon_child_timeout_retained`. The mock receives **zero wait calls and zero poll calls**. Immediately afterward, independent descriptors can acquire nonblocking exclusive flocks on all five authority/control/receipts/staging/IPC roots. `p1-core-intent` remains. Probe passed in this review.

This specifically violates r356's synchronous parent-exclusion-through-exit/reap fence. Durable incomplete intent correctly prevents ordinary successor activation; this finding does **not** claim a demonstrated second Core or false completion. Linux may kill namespace children when helper PID1 finally exits, but that does not establish synchronous exit/reap before the locks unwind. Preserve the incomplete intent while retaining responsibility and exclusions through verified child exit/reap; do not substitute eventual PID1 teardown for this fence. Add an injected timeout/unreaped test at the actual adapter boundary.

### F2 — P2: claimed stream bounds are only post-write size checks

`service/daemon_client.py:83-92` passes a normal unbounded temporary file directly as Docker stdout, waits for process exit, and only then checks its size. `service/daemon_helper.py:52-63` repeats this for child output (64-KiB result ceiling). Neither implementation imposes an output storage bound while the producer runs; on timeout, the producer is also left running with its output descriptor. A wall-clock timeout is not a byte bound.

Reproduction executed with fake Popen: in fake `wait`, write `LIMIT+1` bytes into the supplied stdout file, record its fstat size, and return exit 0. `Docker.call` allocates **1,048,577 bytes** before raising `daemon_output_budget`; there is no intervening size observation or bounded write path. This tiny probe deliberately stopped at one byte over the advertised ceiling. Inspection shows the producer can continue writing arbitrarily until exit/timeout; no large-output test was run.

This matters on the expressly constrained client and for bounded helper tmpfs: rejection after storage exhaustion does not enforce the contract. Drain through a byte-capped mechanism with a finite retained-process policy; cover overflow and timeout while retaining/reaping obligations safely. The helper's child output needs the same treatment.

### F3 — P2: `complete` accepts trailing protocol output

`service/daemon_client.py:199-214` stops reading at the first `complete` frame and returns success after process/container exit checks. It never verifies stdout EOF or rejects `Wire.pending`/unread trailing bytes. Thus exact nonce/sequence/closed-field validation is bypassed for anything following the apparent completion.

Reproduction executed with real inert pipes and fake Popen/container inspections: preload stdout with the correctly bound status `complete` frame (version 1, matching nonce/phase, step 2, result `{"phase":"initialized"}`), followed by `UNEXPECTED-TRAILING-PROTOCOL\n`. Fake inspections show the same exact helper running then exited 0. `Docker.exchange` returns `{"phase":"initialized"}` successfully. No subprocess or Docker ran. Probe passed.

Require bounded EOF after the single terminal frame and reject pending/unread bytes or a second frame before publishing success. Test both bytes already buffered in Wire and bytes read subsequently. This is a finite-protocol correctness defect, not a claim that an untrusted public request can replace the pinned helper.

## Coherent final verification

Every Python invocation used `/private/tmp/pab-0c-venv/bin/python -I -B`, asserted CPython **3.14.7**, and inserted explicit absolute candidate, tests and tests/service paths. The runner installed audit hooks rejecting actual `subprocess.Popen`, `os.system`, socket bind and socket connect events; injected functions bypassed no real operation. Read test implementations/import seams and subprocess/socket sites before selection. Archive tests were only static/in-process; no extracted archive child ran.

One runner preparation attempt failed parsing the old receipt because it selected the bracket in the interpreter's Clang version string. It loaded/executed **zero tests** and was corrected to select the newline-delimited JSON array. The subsequent coherent suite ran **239 tests in 89.949 seconds: 239 PASS, 0 failures/errors/skips**. The existing intentional `injected_close_unknown` test emitted its shielded-future diagnostic and passed. No claim that this diagnostic proves successful close.

Exact allowlist (class selections include inherited unittest methods):

```text
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

Independent additional probes: **2 PASS in 0.239s** (F1/F2), then **1 PASS in 0.018s** (F3). These are defect-demonstration tests, separate from the 239-test regression receipt; no combined fictional suite is claimed. Probe code was supplied on stdin and left no persistent script/receipt. Tool outputs retain the raw counts and observations. `git diff --check` against the parent passed. Candidate tracked/untracked status remained clean after verification.

The supplied r357 historical receipt remains **236 PASS, 1 failure, 2 errors / 239** before correction, followed by **5 focused PASS**. The new 239-green receipt applies to the exact corrected candidate and does not rewrite those historical results.

## Bounded audit conclusions and remaining gates

The source implements the requested separate explicit `installed-daemon-volumes-v1` enrollment and `daemon-volumes-inspection-v1` observation route. The stdlib client dispatch/package entries exist; deterministic archive bytes/entries passed in process. New fixed-directory config/credential mounts and eight distinct named-volume roles are represented. Authority is separate from control and absent from both runtime mount vectors. Authority records use exclusive, checked, fsynced root-owned writes; pending operations and unexpected consumers refuse. Client endpoint/daemon/image/volume checks and helper/local cross-view checks remain explicit. Credentials are omitted from public hashes/manifest records and travel through attached stdin; no credential disclosure was found in inspected argv/handle/error/provenance paths.

Fixed-view Core/writer entrypoints validate role descriptors, close other descriptors, permanently drop IDs/groups and check capabilities before existing Core/writer functions. They do not introduce mount/unshare/SYS_ADMIN, copied schema or a second scheduler. Source fixtures only model those OS boundaries. Strict host enrollment/startup evidence stays separately checked; public schemas do not gain installer authority. The regression allowlist exercises old host activation, persistence, cancellation, ACK/reap/publication and clean restart refusals. No additional concrete weakening of those shared validators was established by this review.

Launch ordering, registration brackets and same-ID notification are implemented. Stop uses TERM control, bounded clean-history checks, then TERM worker; there is no Docker-stop KILL escalation. Same-ID clean stop/up and completed-work replay are covered with synthetic runtime/journal/daemon facts, including negative history tests. Those green fixtures do not resolve F1-F3 or prove installed kernel behavior.

Parent-owned next steps: correct F1-F3 in a new pinned candidate, add their focused regression tests, and obtain fresh independent review before source adoption. **Do not adopt this unchanged commit.** This report does not self-adopt anything.

Actual rootful Linux/amd64, THIS Intel Mac Docker Desktop and THIS Intel Mac OrbStack installs remain separate **UNRUN** rows. Packaged Python 3.11/dependency/image execution, local Docker transport, mount/inode continuity, actual UID/capability enforcement and denied peers, engine isolation, final worker PID1 descendant reap, loopback exposure, authenticated audible Kokoro/artifact bytes, completed-work replay/restart and measured resources remain unqualified here. VM/daemon restart identity behavior and changed retained inode refusal require actual measurement; no reset/restore/rebind permission is supplied. CUDA/native Metal compatible rows remain mandatory later. **Metal UNQUALIFIED; release NO-GO; legacy CLOSED.** Accepted native freshness/source epochs and renewal remain unchanged.
