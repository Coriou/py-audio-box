# r363 Sol — fresh independent finite ownership review

2026-10-25. **Bounded GO for SOURCE adoption of f8c3861664e608b5aaca28d08ec90bee39634bc7 under r361's finite acceptance criteria.** r360 R1 and R2 close at SOURCE; no remaining blocking finding was established in the commissioned ownership slice. Independent coherent verification ran once: **254 PASS in 92.908s, zero failures/errors/skips**. This is a review recommendation, not adoption. Fresh parent Astra assessment and parent adoption remain separate. **Release NO-GO; legacy CLOSED.**

## Authority and verified pins

Reviewed `/Users/ben/.txcode/worktrees/py-audio-box/t3-ad4c73be`. Before review and after verification, tracked/untracked status was empty. Verified exact chain:

```text
f8c3861664e608b5aaca28d08ec90bee39634bc7
e29af7892e6200fdaea5c1388c9e80bcf0fd227c
bc075ccc93510564f25c9788348c2ac0aa88df54
45ce583b1f5c61e14c0218911a76ea68a6e928d9
```

Read normative full root r361, root r360/r358, candidate r356 contract and r357/r359/r362 reports; inspected the ownership diff, actual helper boundary, Wire/client/notify seams, focused tests, both conftests and selected import/subprocess/socket seams. Candidate/ancestor AGENTS discovery found none. Memory registry search supplied no relevant technical premise. Root r361 SHA256 matches `f2edf188a7cd03cdc3a343c094b6ad9568f7a1a0d485da7981da07b08858790b`; root r360 matches `5a46e4289fb8dd8cd690c7580a2b715a018de4d872eadf9512864cd0fbdb054c`. All four r362 final SHA256 pins match candidate bytes:

```text
6dad24cb20dbc3f0d5878e69b305cd5bb1d7607ee1e6ea0b005551a52e012844  service/daemon_client.py
369e50864de1c434b819adefb4eb52a3a83da6312eb5f227e1ec0fa18fa8709e  service/daemon_helper.py
b8b7812483410be5853a3077deb8e5b0d2fd3b155e8ed8dd03effdf4eb1c2161  tests/service/test_daemon_volumes.py
8742d6746d5addbbd40dfa6933c85c0ccfde64a75102f62104063aefdffaa525  docs/agent-interface/reviews/H3-CONTAINER-INSTALL-FIX-r359-sol.md
```

`git diff --check HEAD^ HEAD` passed. The five-file diff changes client ownership, helper deadline placement, focused tests, corrected historical attribution, and r362 report only. Strict old-host code, Core/persistence/ACK/reap/publication validators and the eight-volume contract are unchanged by this correction.

Sole persistent review write: this mode-0600 root report. No source edits, application imports outside inspected tests, commit, merge, adoption or delegation. No Docker/network/listener/application child/ML/hardware/HELD/STT, extracted-archive child, operational cleanup or heavy-operation waiver. Tiny fixture files/SQLite/in-process ASGI, inert pipes/fake processes and controlled threads only; fixture teardown is not operational cleanup. Existing unrelated root untracked evidence was left untouched.

## R1 closure and four-site coverage

**R1 CLOSED at SOURCE.** `service/daemon_client.py:65-133` makes setup/reference/protocol/cleanup/wait/cadence errors sticky while preserving synchronous local ownership. Stream closures are independent (`104-110`): a failing stdin close does not skip stdout or reap; closed state is checked through the stream object on retry, including close-then-raise. There is no remembered-fd fallback that could close a reused descriptor. Selector close retries remain owned (`111-114`). Wait/clock faults and unknown status pass through guarded monotonic cadence (`81-90`, `115-123`); interruption resumes the remaining interval. Reference removal occurs after reap and resource disposition (`124-130`). Only the first exception is retained, without accumulating per-retry traces/logs/payloads. Negative exit status fails; no invented returncode, signal escalation or replacement exists.

All four new-lane Popen sites use this owner:

- `Docker.call`, client `197-201`, immediately delegates to `collect`, whose owner begins at `138`, before selector/stream setup and decoding.
- Notify, client `263-272`, prepares bounded encoded input/deadline before spawning and uses the same collector with absent stdout; the old communicate-timeout exception path is removed.
- Exchange, client `293-315`, enters Owner immediately after attach acquisition, enclosing Wire construction, protocol and nested actions.
- Helper child, `service/daemon_helper.py:46-58`, computes its deadline before spawning and calls the shared collector synchronously inside existing exclusions. Core, hello and writer use the same seam.

The normal OS-backed synchronous Popen wait contract is the source premise, as specified by r361. The helper remains on the same stack inside authority/control/receipts/staging/IPC exclusions until actual modeled reap. Ordinary deadlines bound success, not exceptional ownership duration. Orchestrator-approved indefinite ONE-helper-child parking remains bounded in retained resources/cadence, without KILL/rollback/replacement/success/intent clearing. This does not claim SIGKILL/crash or unlimited adversarial bytecode-interruption resistance.

## R2 closure and preserved stream/protocol bounds

**R2 CLOSED at SOURCE.** Exchange's enclosing Owner covers pending junk/second frame, separately read trailing data, terminal EOF timeout, Wire setup/send/receive/action/finish exceptions and successful completion. It closes both streams and reaps the exact local attach process before removing its retained reference or returning/raising. `Wire.selector` (`329-340`) registers selectors with the owner so failed teardown is retried during finalization. Final helper inspection and daemon check (`316-319`) occur only after attach finalization. Attach zero is insufficient: helper running/nonzero final states still refuse. A final inspection error cannot leak the already finalized attach owner.

Attach CLI reap is explicitly separate from helper completion and from the helper's own child reap. Nested action ownership completes before enclosing attach finalization. No helper stop/kill, synthetic terminal receipt, changed intent or public success follows from attach failure.

r358 F2/F3 remain corrected at SOURCE. `collect` (`136-176`) retains the live 1-MiB client / 64-KiB helper payload limits, 64-KiB chunks, bounded input slices and no disk spool; failures discard output or safely close readers before waiting. These are payload caps plus bounded object/pipe overhead, not total-memory measurements. `Wire.finish` (`354-363`) still rejects pending/trailing bytes and requires bounded exact EOF. EOF never substitutes for wait or helper identity checks. r359's provenance now correctly says orchestrator-resolved exceptional parking (`43`), without changing its historical receipts.

## Finite acceptance and independent verification

The six r361 conditions are satisfied by the inspected source and final selection:

1. `OwnershipTests.test_helper_wait_cadence_interruptions_hold_locks` (`test_daemon_volumes.py:741`) exercises actual helper init with wait error or noninteger status, cadence KeyboardInterrupt, later timeout and modeled reap. All five roots are contested before every reap observation and acquirable afterward; one spawn, retained intent and no enrollment.
2. Independent before-close/close-then-raise faults on both streams, controlled .1/.06 remaining cadence, selector teardown and append-then-raise reference setup are covered at `773-800` and `859-883`; existing setup/unknown-wait coverage remains selected. Source independently guards clock/wait/cleanup retry paths.
3. Actual pending junk/second frame/separately read trailing-byte tests (`684-705`) assert closed streams/reap/empty references before teardown. Exchange fault tests (`803-829`) cover construction/send/receive/action/finish and real EOF timeout.
4. The same exchange matrix covers success, attach nonzero/signaled exits, failed final inspection and helper running/nonzero. Only full success returns; local resources release before final checks.
5. Notify success/timeout/interruption through the common collector and nested owner order are asserted at `831-857`, without helper kill/replacement.
6. Existing live overflow, ordinary timeout, good-output and exact EOF fixtures remain in the coherent allowlist.

No additional ad hoc probe or correction run was necessary. One final run used the already present, inspected `/private/tmp/r362-run.py` (SHA256 `90ced5e711b121f4ab02c8fc9f62cac8c016a1879e1e1fd49c2f647a01a1a1b2`):

```text
/private/tmp/pab-0c-venv/bin/python -I -B /private/tmp/r362-run.py final
Ran 254 tests in 92.908s
OK
```

The runner asserts CPython 3.14.7; inserts absolute candidate root, tests and tests/service paths; installs audit rejection of real subprocess.Popen, os.system, socket.bind and socket.connect before application/test imports. Direct unittest avoids pytest conftest/listener activation. Exact selection is r360's `Exact selections` 249-test allowlist plus these five individually named methods, without inherited duplication:

```text
tests.service.test_daemon_volumes.OwnershipTests.test_helper_wait_cadence_interruptions_hold_locks
tests.service.test_daemon_volumes.OwnershipTests.test_independent_close_retry_and_cadence_remaining_interval
tests.service.test_daemon_volumes.OwnershipTests.test_exchange_faults_and_exit_predicates_finalize
tests.service.test_daemon_volumes.OwnershipTests.test_notify_common_owner_and_nested_order
tests.service.test_daemon_volumes.OwnershipTests.test_selector_teardown_and_reference_registration_faults
```

The existing intentional `injected_close_unknown` shielded-future diagnostic appeared in its passing activation test; it is not evidence of successful close. No audit rejection, failed preparation, retry, runner termination, skipped test or corrective source mutation occurred. Raw run output is retained in the review tool transcript, not an additional persistent receipt file. r362's prior 254 PASS/92.026s remains its separate implementation receipt.

## Parent-owned disposition

No remaining concrete finding in this bounded ownership review blocks SOURCE adoption of the exact pinned candidate. Parent Astra assessment/adoption remain required; this report performs neither. The GO does not qualify installation or runtime behavior.

Actual rootful Linux/amd64, THIS Intel Mac Docker Desktop and OrbStack remain separate **UNRUN** rows. Actual OS process/pipe/reap, UID/capability/mount enforcement, Docker transport, packaged Python 3.11/dependency/images, volume/inode/restart continuity, supervisor PID1 descendant isolation/reap, loopback exposure, authenticated audible Kokoro/artifact bytes and measured resources remain unqualified. CUDA and native Metal remain mandatory and unqualified. Accepted native freshness/source epochs, headless HTTP/CLI/MCP scope and legacy CLOSED remain unchanged. **Release NO-GO.** No operational cleanup, installation or heavy-operation authorization follows.
