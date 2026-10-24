# Native terminal/cancel linearization correction — r284 Sol

2026-10-25. SOURCE implementation ready for parent fresh independent Sol review;
no self-acceptance. Full native Lot A remains PARTIAL and closed to acceptance.
Native installation/readiness, actual Metal and release qualification remain NO-GO.

## Authority and exact pins

Own worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-85b1b50f`, branch
`t3/fix-native-terminal-cancel-r284`. Clean root-foundation/review-doc entry:
`ab460e5` (no foundation reapplication). Requested commits applied in order:

- Supervisor `de6c07be80d702c19f7dcabe113e16b02e7dcae4` → `36e329d6a291189483a963d907defffea70edcc2`.
- Report `b16694f5654de69b9536912d97b07b83c74c7309` → `9d8113dc5dff4b6b51316c052aa1569f3746e838`.
- F1–F4 source `e9c17128f65d36d0e97befc3bf38e7a5227183f8` → `f91b8e44abcbf4ef45ab3c1be740cf10cf7b6bc4`.
- Report `30a7a09ce19029104d2db5feb5554860a8538c30` → `02e82ba56b27cfa2b8800c12fbae47cd3672d358`.

New correction source: **`2bc93f5152eb942d54788867b18ed6f223c65e31`**.
This report is a separate subsequent commit. Historical candidate
`/Users/ben/.txcode/worktrees/py-audio-box/t3-4d9ec80c` was untouched; final read-only
verification found it clean at `30a7a09ce19029104d2db5feb5554860a8538c30`.
Read full normative r277 and r280–r283. No AGENTS.md found in repository/checked
ancestors. Memory keyword search returned no relevant technical entry. Both
conftests were inspected; neither pytest collection nor its maintenance guard or
service listener fixture was invoked. No delegation occurred.

## Decision and stop semantics

Only supervisor, supervisor tests and minimal private protocol response validation
changed. Validation, output write/hash/file and directory fsync, receiver join and
exact wait occur outside handle_lock. After those prerequisites, handle_lock
serializes one immutable in-memory terminal decision with accepted cancellation.
Deadline eligibility is also sampled at this decision. A stop that wins this lock
first latches cancellation; the decision then forces timely=false and null
proposal/output. A stop after this decision cannot latch cancellation.

The private stop response retains `{}` for accepted cancellation. After a frozen
decision it returns the closed `{state:'committing',timely:bool}` variant; after
durable terminal it returns `{state:'terminal',timely:bool}`. These responses
observe a decision rather than acknowledge a new cancellation. `committing` is
explicitly not a receipt, durability proof, STOPPED proof or release authority.
Unknown fields/states and nonboolean timely refuse. Request/receipt/public schema
shapes are unchanged. A future NativeRuntime client must understand these finite
stop variants; no client/Core integration is claimed here.

Terminal persistence follows the decision outside handle_lock. The active handle
remains owned until persistence returns successfully; retirement uses handle_lock.
Shutdown does not overwrite a frozen decision with a cancellation latch and still
requires active accounting to complete. Journal failure latches uncertainty,
retains the active handle/decision, yields UNKNOWN/null collect and refuses later
stop/start/ACK authority without retrying persistence. No in-memory decision can
repair an uncertain journal. The existing watchdog still signals independently of
state-lock/journal work and storage quarantine; no hash/fsync/journal/wait was
added under handle_lock. Original F1–F4 regression tests/properties are retained.

## Executed checks

All commands used `/private/tmp/pab-0c-venv/bin/python -I -B` with explicit paths
`sys.path[:0]=['.','tests/service']` and explicit unittest selections.

Final focused invocation:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=[".","tests/service"]; unittest.main(module=None,argv=["unittest","tests.worker.test_native_supervisor","tests.worker.test_native_protocol","tests.worker.test_native_qwen.NativeQwenTests","tests.worker.test_engine_qwen"],verbosity=0)'
```

**58 PASS, 4.050 seconds**: existing 52 plus six new methods. Actual handlers with
fake handles/channels/clocks and valid tiny encoded results enter deterministic
thread barriers at validation, actual output fsync, terminal persistence and
post-persistence/preretirement. Predecision stop returns empty ACK and produces
false/null terminal evidence. Postdecision stop returns committing/terminal without
setting cancel. Repeated stop/start/collect/ACK/download stays coherent and launch
count stays one. Persistence failure retains UNKNOWN with no later save. Exact
wait count one and receiver FD closure are asserted. Response negatives cover
extra fields, invalid state and bool-as-int. Existing blocked historical ACK
cutoff/stop TERM→KILL and storage-quarantine tests remain passing.

Separate fresh interpreter after the final source refinement:

```text
/private/tmp/pab-0c-venv/bin/python -I -B -c 'import sys,unittest; sys.path[:0]=[".","tests/service"]; unittest.main(module=None,argv=["unittest","tests.worker.test_qwen_import_seam","tests.release.test_release_package.ArchiveTests.test_manifest_exact_files_and_no_legacy_metadata_change"],verbosity=1)'
```

**6 PASS, 0.377 seconds**. Root/extracted archive/legacy imports use fake upstream.
Earlier focused invocation passed 58 in 3.107 seconds, before moving deadline
sampling into the decision lock and removing an unused field; no failing suite.
`git diff --check`: PASS. `lib/tts.py` remains unchanged with SHA256
`fb8b65eab87e8286db56644af71b2ce547a29e9a73ee52235fe48aa6ca663528`.
Historical broader legacy digest RED is not waived.

Available storage observed 3,414,452 KiB, below 6 GiB. Only tiny disposable fixtures
and archive contents ran, with injected admission capacity. The production 6 GiB
plus declared attempt-demand gate is unchanged. No actual process/socket/listener,
ML, hardware, Docker, network, download/install, crash experiment, operational
cleanup or HELD action occurred. Existing fixture teardown touched only its owned
disposable resources. Printed MPS/CPU/CUDA names are synthetic fixture declarations.

## Remaining authority gates

Parent fresh independent Sol review of this exact source is required before
adoption. This is focused source concurrency evidence, not an exhaustive crash,
platform scheduling, kernel PID/FD or durability matrix. In-memory uncertainty is
not restart recovery authority. Nonempty clone conditioning remains UNKNOWN.
NativeRuntime/client, detached immutable control transfers/local import UID,
shared frozen-lane receipt consumers, native registration/admission/resolution,
Dispatcher/Core cancellation/publication-intent/reservation/ACK ordering and the
complete Authority-to-Core restart/replay slice remain parent-owned. Full Lot A
is partial. Native packaging/enrollment/dependency closure and actual Apple Silicon
probe/load/generate/cancel/restart remain later gates. Metal stays unavailable and
unqualified; CPU Docker and CUDA remain mandatory separate release rows.
