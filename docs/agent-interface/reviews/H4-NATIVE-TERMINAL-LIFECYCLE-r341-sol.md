# r341 Sol — native terminal lifecycle implementation candidate

2026-10-25. First same-boot SOURCE slice of r334 delivery 2; pending fresh independent Sol review and parent adoption. Delivery 2 is incomplete. Metal remains UNQUALIFIED; actual release remains NO-GO.

## Authority and pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-37b3f731`, initial clean HEAD `63d47fc67d29d13d831665676996f3df367615e1` (integrated public controls). Read r334, PRODUCT, NATIVE-PACKAGE, current activation/history, runtime import/receipt paths, supervisor terminal decision, Core cancellation/publication/reconciliation/ownership fences, public controls and affected inert fixtures. No applicable AGENTS.md found in worktree discovery or checked ancestors. The closed legacy ledger was not edited.

Only source, affected fixture/tests, operator documentation and this report changed. No root checkout writes, merge, push, delegation or self-adoption. No network/listener, Docker, installation/download, actual child, ML/model execution, hardware qualification, crash, HELD/STT or operational cleanup. Direct unittest only after inspecting both conftests; pytest fixtures (including server subprocess/listener) were not invoked. Tiny temporary filesystem/SQLite and in-process ASGI, fake TLS/process/backend only; no heavy operation or 6 GiB gate waiver.

## Result and finite policy

`completed_history` retains its exact canonical schema/SQL/document validation under Core lifetime exclusion before initialization/recovery/scheduling. It now permits single-attempt `failed`/`cancelled` jobs at `deadline_or_reaped` only with exact frozen Authority/native boot/trust, reaped attempt, durable matching outcome and ACK journal, retained exact native demand, and a native terminal receipt identical to the supervisor's read-only collect response. The canonical native receipt primitive still establishes exact runtime/binding/deadline, owned reap method, receiver closure and process tuple. This slice requires one process and `timely:false`, with no proposal/output.

Publication/work must both be absent for these negative histories. Spool must contain exactly snapshot plus frozen clone input when applicable; staging must be empty; import closure/reopened import evidence must be absent. Clone input and owned Core reference are checked with the existing ownership, metadata, hash and PCM primitives before either terminal branch. The negative supervisor receipt follows owned wait, receiver join and conditioning removal. No reference reclamation, history rewrite, repair, automatic ACK, cap increase, new Core/scheduler or generic recovery was added. Failed verification with retained output, unknown/unreaped/unacknowledged and partial publication/pair/import all continue to refuse without allocation or evidence mutation. Success publication/intent/import validation is preserved.

Never-started cancellation is explicitly unsupported: this policy requires the dispatched Authority and one owned process receipt. Empty-process receipts and never-started history refuse. Native retry remains unavailable through the existing public guard; tests assert retry refusal without RPC allocation. New compatible submissions after proved terminal history are supported.

The operator document consolidates current owner history and managed-clone instructions, removing obsolete first-start-only/no-clean-clone and stale legacy-RED claims. It continues to distinguish packaged SOURCE from actual installed evidence.

## Test receipts

Interpreter: `/private/tmp/pab-0c-venv/bin/python -I -B`, CPython **3.14.7**, not 3.11. Test imports explicitly insert this worktree. Initial finite selection loaded these explicit paths with `unittest.TestLoader.loadTestsFromName(path[:-3].replace('/','.'))`:

- `tests/release/test_native_history.py`
- `tests/release/test_native_managed_clone.py`
- `tests/release/test_native_onboarding.py`
- `tests/service/test_public_job_control.py`
- `tests/service/test_install_ownership.py`

Initial receipt: **174 tests, 181.146 s, 31 failures + 1 error** (142 passed). Twenty-nine failures were the success-only helper's erroneous assumption that a rejected receipt-only history must have a pending import; the histories themselves refused. Public cancellation reused the previous fake handle's exit code; public failure counted three process starts instead of four (preparation, success, failure, subsequent success). An existing imported private new-boot lifecycle fixture omitted refreshing its explicit original enrolled templates after rebuilding the fake supervisor. Corrected these fixture/assertion issues, including fresh fake handles. Production policy did not change after that run.

The final affected selection is the two terminal-history classes, the two public saved-clone cancel/failure restart methods, and the corrected existing private lifecycle method. It includes new Core exclusion and explicit never-started refusal. No broad unchanged repeat was performed. Final receipt: **43 tests in 50.087 s — OK, zero failures/errors/skips**. The exact selected IDs were:

- all tests in `tests.release.test_native_history.NativeTerminalHistoryTests` and `NativeCancelledHistoryTests`;
- `tests.release.test_native_managed_clone.ManagedCloneTests.test_saved_clone_cancel_reopen_and_new_job`;
- `tests.release.test_native_managed_clone.ManagedCloneTests.test_saved_clone_failure_reopen_and_new_job`;
- `tests.service.test_native_lifecycle.NativeLifecycleTests.test_actual_authority_native_child_dispatcher_core_completed_reopen_and_next_job`.

Invocation used the pinned interpreter with `-I -B -`, explicit worktree `sys.path.insert`, `loadTestsFromTestCase` for those two classes and explicit `Class(method)` instances for those three methods, then `TextTestRunner(verbosity=2)`. Its terminal receipt was `Ran 43 tests in 50.087s` / `OK`. The initial failed receipt remains historical above; it is not rewritten as PASS.

Public saved-clone cases cover upload/prepare/results/save/generation, controlled failure or cancellation during work, confirmed terminal receipt/ACK, close/reopen, terminal public wait, immutable voice and previous jobs/artifacts, and new successful compatible generation without per-voice enrollment edits. Existing clean preset/clone journeys and public wait/cancel/cursor/disconnect controls passed in the initial selection. Refusal tests compare retained filesystem bytes, SQL dumps and ACK counts; foreign/old-boot receipt, missing reap, receiver open, unACKed/reservation, partial pair/import, changed reference and pre-schema catalogue refusal are included. Core ownership excludes another reader before activation. These are SOURCE/inert results only.

## Remaining parent-owned gates

Fresh independent Sol review and parent integration are required. New-boot/stopped-reconciled trust renewal, bounded capacity renewal, Docker/native topology and mount/peer validation, and real arm64 dependency packaging are next required delivery-2 parts. The private fixture that explicitly reenrolls a new boot does not implement installed reauthorization. Never-started and terminal output/intent uncertainty remain unsupported, as does native retry. No history deletion or cap bypass is a remedy. Actual packaged install, cancellation/reap/restart/renewal, retained artifact verification, audio/listening, resource/performance and CPU/CUDA/Metal qualification remain separately authorized gates.

## Tested source SHA256 pins

- `service/native_activation.py`: `1c36e5fd6af669471b37b3c453682aea017ef2d094ef6ac55b03eadd19a611dd`
- `tests/release/test_native_history.py`: `ffbc9df6191f4bd69e2de0c336d567bd8c225d961ada709e2ff39b71d15410ed`
- `tests/release/test_native_managed_clone.py`: `90c346b99cd1996e9cb04f13da19a7fe8f106d9c51ac7a22717b951ea5c10104`
- `tests/service/test_native_lifecycle.py`: `663bbcd1e99a79a6fef3480ec24f0762d6212f1738e6dc1983c10521e8c4a924`
- `docs/agent-interface/NATIVE-PACKAGE.md`: `acd5357ab190587d96422db5a1d19eccff7a69fef224576213fcfd87cdd80ce1`
