# r364 Astra — final bounded container-install source gate

2026-10-25. **GO for parent SOURCE adoption of exactly f8c3861664e608b5aaca28d08ec90bee39634bc7.** No blocking concrete functional contract break was established in this bounded r356 plus r361 assessment. r358 F1–F3 and r360 R1/R2 are closed at SOURCE under the accepted finite ownership criteria. This recommendation does not perform adoption or qualify installation. **Release NO-GO; legacy CLOSED.**

## Authority, pins and checks

Reviewed source-readonly `/Users/ben/.txcode/worktrees/py-audio-box/t3-ad4c73be`; HEAD and direct-parent chain verified:

```
f8c3861664e608b5aaca28d08ec90bee39634bc7
e29af7892e6200fdaea5c1388c9e80bcf0fd227c
bc075ccc93510564f25c9788348c2ac0aa88df54
45ce583b1f5c61e14c0218911a76ea68a6e928d9
```

Root remains at 45ce583b1f5c61e14c0218911a76ea68a6e928d9, with no tracked edits. Candidate tracked/untracked status was empty. Applicable repository/explicit ancestor AGENTS checks found none. Read full candidate normative r356 and r357/r359/r362, full root r358/r360/r361/r363; inspected source, relevant base-to-candidate changes, finite ownership fixtures, distribution dispatch, topology reference, public journey documentation and PRODUCT. Memory keyword lookup supplied no technical premise for this decision.

`git diff --check 45ce583b HEAD` passed. The final correction changes only client/helper ownership, focused tests, historical attribution and its report. Core, Dispatcher, host_install and runtime_journal bytes are unchanged from root base; the broader candidate's explicit enrollment/startup additions were inspected separately rather than described as an entirely unchanged recovery module.

Verified SHA256 pins:

```
6dad24cb20dbc3f0d5878e69b305cd5bb1d7607ee1e6ea0b005551a52e012844  service/daemon_client.py
369e50864de1c434b819adefb4eb52a3a83da6312eb5f227e1ec0fa18fa8709e  service/daemon_helper.py
b8b7812483410be5853a3077deb8e5b0d2fd3b155e8ed8dd03effdf4eb1c2161  tests/service/test_daemon_volumes.py
8742d6746d5addbbd40dfa6933c85c0ccfde64a75102f62104063aefdffaa525  candidate r359
 a59710f5ac746974730c91f9144f6bb1a990083b1a6be312ce7daf3109e960da  candidate r356
f2edf188a7cd03cdc3a343c094b6ad9568f7a1a0d485da7981da07b08858790b  root r361
5a46e4289fb8dd8cd690c7580a2b715a018de4d872eadf9512864cd0fbdb054c  root r360
```

Sole write is this root report, mode 0600. No application imports, tests, probes, source edits, commit, merge, delegation, Docker, network/listener, application child, ML/hardware, HELD/STT content access or cleanup. Unrelated untracked evidence was observed only as status names and left untouched. No hardware/resource remeasurement or gate waiver.

## Ownership correction is adequate for the accepted exit

`service/daemon_client.py:65-133` owns the same local Popen through independent stream/selector disposition and integer wait completion. Setup/reference insertion, protocol, cleanup, wait and cadence failures remain sticky. Close-then-raise is handled through the stream object's closed state, without an unsafe remembered-fd close. Cadence (`81-90`) catches interruptions and resumes the remaining monotonic interval; finalization (`100-130`) cannot ordinarily unwind on the previously demonstrated cleanup/wait/cadence errors. References release only after reap and resource disposition. Later zero status cannot erase failure; negative exit cannot pass.

All four sites are covered: Docker.call (`197-201`) through collect, notify (`263-272`) through the same collector, exchange (`293-315`) through enclosing Owner, and helper Core/hello/writer (`service/daemon_helper.py:46-58`) through synchronous collect. Helper initialization (`276-294`) and registration (`353-388`) keep their existing enclosing exclusions while their child remains unresolved. Nested action CLI ownership finalizes before enclosing attach ownership unwinds.

Live payload bounds remain at client 1 MiB/helper 64 KiB with fixed 64-KiB reads, bounded overhead and no disk spool (`collect:136-176`). Terminal pending/trailing data and missing EOF still refuse (`Wire.finish:354-363`). Exchange final inspection follows local finalization (`316-319`); attach zero alone cannot accept a running/nonzero helper. The finite fixtures at `tests/service/test_daemon_volumes.py:741-883` cover lock contention before modeled reap, interrupted cadence, independent close failures, selector/reference faults, exchange failures/success and notify nesting. These source assertions align with r363's independent receipt; they are not actual process evidence.

Three identities remain distinct: the client-owned local Docker attach CLI; the daemon-owned retained helper container; and the helper-owned Core/hello/writer child. Local CLI reap does not establish helper completion, and helper exit does not replace the helper's child-reap fence. Worker PID1 engine-tree reap remains the existing, separate publication prerequisite.

**Bounded resources with exceptionally unbounded duration is the accepted ORCHESTRATOR decision**, explicitly corrected in r359. Ordinary deadlines bound a success attempt; unknown completion may park the same owner and exclusions indefinitely. No force KILL, rollback, replacement/new work, success or intent clearing follows. Ordinary startup retry of an already finalized transient hello is distinct from replacing an unresolved child. Normal OS-backed synchronous Popen ownership is the premise; SIGKILL, interpreter/VM death and unlimited adversarial bytecode interruption are not promised away. Durable pending intent on parent death remains refusal evidence, not a reap receipt. This acceptance does not authorize manual intervention or automatic installation qualification.

## Overall r356 contract assessment

The source delivers the intended architectural adaptation: stdlib operator client and installed helper entrypoints, explicit local Unix endpoint/daemon binding, eight distinct named volumes, separate protected authority, finite recorded helper history and exclusive creation receipts. `daemon_client.py:402-466` dispatches actual install/up/status/stop coordination; `daemon_volumes.py:41-184` defines admission, exact narrow mount/security/resource vectors and inspection. No client VM shell, host stat fiction, Docker socket mount, implicit pull or generic command bus is introduced.

`daemon_helper.py:154-220,224-294` holds authority, corroborates mounted roots, imports bounded enumerated inputs, seals them and calls the existing initializer. The separate fixed-view child in `install_child.py:205-281` validates inherited roots, closes other descriptors, permanently drops IDs/groups and verifies capabilities before existing Core/writer work. Its helper lacks model/credential mounts. Kernel enforcement remains unproved here.

The new launch/evidence variants are explicitly bound through enrollment, registration and trusted startup selection. Strict host enrollment rejects the new launch binding; recovery's installed readers bind the selected evidence to enrollment. Runtime local facts describe actual container PID1, not the helper or Docker host PID. Registration, notification and readiness remain bracketed (`daemon_helper.py:341-400`); startup rechecks before and after Core ownership. The source preserves existing persistence, frozen inputs, attempt Authority, ACK/reap and publication fences rather than treating Docker health as terminal proof.

Stop and restart (`daemon_helper.py:402-463`) retain exact IDs, TERM control before worker, require clean history and matching stop receipts, and refuse partial/dirty state. Authority/history limits and retained failure refusal are explicit product limitations. Concrete public preset/generation/artifact schemas are documented. No additional blocker was found by these targeted checks; this is not an exhaustive audit or evidence that ordinary Desktop/OrbStack execution already works.

## Evidence and parent-owned next product step

No suite was rerun. r363 reports independent **254 PASS in 92.908s**, once on this clean candidate; r362 reports its separate **254 PASS in 92.026s**. These are distinct source/inert receipts, not a combined run or this review's execution. Historical r358 239-green and r360 249-green did not close their demonstrated defects; their objections are resolved by the inspected correction and finite regression evidence, not by counts alone.

Parent may now adopt the exact pin under its own merge authority and finish this source gate. No further unchanged broad suite or indefinite ownership-proof framework is required by this recommendation.

Next PRODUCT priority is a separately authorized practical CPU installation qualification: packaged Python 3.11/images/dependencies and preprovided inputs, ordinary client install/up, authenticated discovery/preset/generation, audible Kokoro and verified artifact bytes, same-volume reopen/replay and clean stop/up, with real UID/capability/mount boundaries, worker PID1 descendant reap, denied access, loopback exposure and measured resources. Linux/amd64, THIS Intel Mac Docker Desktop and OrbStack remain separate **UNRUN** rows; qualifying one does not qualify the others or VM-reset/restore continuity.

The supplied last-known free memory is approximately 2.7 GiB, below the unchanged 6 GiB heavy-operation gate; this is not a fresh measurement. No suitable GPU/Apple Silicon target is authorized and no arm64 wheelhouse is available under the supplied constraints. Therefore actual work remains blocked unless parent obtains an admitted, authorized target and prerequisites; this report grants neither cleanup nor acquisition/rental nor a waiver.

If actual CPU work remains blocked, take one finite public product slice from the existing delivery plan: close a concrete uncovered cleanvoice/production operation through canonical HTTP and thin CLI/MCP, with a bounded journey covering immutable voice reuse, durable jobs and artifact retrieval. Inspect existing coverage first to avoid reimplementing accepted paths; exit on that slice's source review, without calling it installed qualification. Voice acquisition/cloning/preparation/audition/management and scripts/takes/assembly coverage remain broader product obligations, not fulfilled merely by a Kokoro preset journey. GPU packaging/compatible profile selection and a managed native macOS arm64 runtime/wheelhouse remain separate delivery gaps; source progress can continue without inventing hardware results.

Headless full HTTP/CLI/MCP and cleanvoice remain the goal; no UI or public STT is commissioned. CUDA and native Metal are mandatory and unqualified, not optional substitutes. Accepted native freshness/source epochs remain unchanged. **Exit: bounded SOURCE GO for the exact candidate; parent adoption next; actual installation unqualified; release NO-GO; legacy CLOSED.**
