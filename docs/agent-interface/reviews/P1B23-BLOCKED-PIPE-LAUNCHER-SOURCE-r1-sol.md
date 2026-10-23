# Blocked-pipe ONE launcher — independent source review r1

2026-10-23. **SOURCE NO-GO for the proposed ONE diagnosis. Four blocking launcher findings below. Do not execute this freeze to test these findings.** The prior observer SOURCE GO remains observational readiness only. No runtime result, P acceptance, full211 acceptance, attestor approval or historical PID229 cause is established.

## Authority, provenance and review boundary

Independent Sol-low SOURCEONLY ABSREADONLY review. Candidate `/Users/ben/.txcode/worktrees/py-audio-box/p-one-node-launcher-20261023` is clean at documentation HEAD `f93b7fef6cc461b756cd9e13ff14e88e243359a4`; executable launcher freeze resolves to `e941c7aee7f53348905659904b8299d5500d1250`. Freeze→HEAD changes only the HANDOFF append. Observer executable `d5f493e63909d31e00680c81e8aefbcf3fcd5162`, documentation `8d5aba5fc702719843d055b7d3f693c407deb0e9`. Sole write is this root report; no commit.

Read FULL root BLOCKED-PIPE-OBSERVER-SOURCE-r1-sol and P-R5-ACTUAL-RECEIPT-AUDIT-r1-sol, candidate HANDOFF/FREEZE, complete launcher, runtime identity, entry, observer and manifest; checked relevant guard/target/runner interaction. Read original brief and prior findings/responses preserved in the handoff and input chain, including F1–F3, R2-1, R3-1 and R4-1 dispositions. Large combined historical appendix output truncated; focused reads recovered the relevant findings/responses. This is not a new audit of every historical appendix or every unchanged production function. No prior objection is waived. No applicable AGENTS.md appeared in checked ancestry/scoped source inventories; a broader read-only search was interrupted without completing, so no global absence claim is made. Memory registry search supplied no relevant launcher evidence.

Only text/Git/hash and host AST parsing/compile without execution or pyc were used. No candidate import/execution, pytest/collection, Docker call/build/inspect, SQLite/database open, native/runtime probe, fix, delegation, push, integration or resource cleanup occurred. No private runtime proof was changed or extracted. No lane, image, container or HOLD was acquired. Existing root STT evidence remains untouched.

## L1 — copied source is unreadable by the declared runtime user

**Blocking before test.** `diagnostics/p-one-node/launcher.py:55–58,71,85,99`.

The staging recipe creates source directories with0700 and every copied file with0600. The generated Dockerfile uses plain `COPY source/ /app/`, with no ownership conversion. Docker COPY creates these source entries owned by root while preserving their restrictive permissions. The container is explicitly run as10001:10001. Thus UID10001 cannot traverse the copied nested directories or read the copied files. In particular it cannot read `/app/diagnostics/p-one-node/entry.py`; the earlier manifest-before command cannot read its root-owned0600 `manifest.json` either. An inherited world-readable `/app` does not make the nested0700 source directories traversable.

The private host proof permissions are appropriate, but this recipe does not establish a usable UID10001 image. The proposed launcher cannot reach the authorized diagnosis as written. This is a source-derived launch failure, not an observed Docker result. Preserve private host evidence while explicitly defining usable container source ownership/permissions in a new frozen recipe; do not repair or run this candidate in this review.

## L2 — HOLD does not preserve unexported proof across finite PID1 expiry

**Blocking evidence-loss path.** Launcher:71,103,113–123; FREEZE retention/HOLD claim.

PID1 sleeps1800 seconds and then exits independently of the host launcher. All diagnostic proof lives in the container's `/tmp` tmpfs. If archive export, LIVE hash collection/comparison or an earlier operation fails, `retained` remains false and the intended response is a LIVE-PROOF-HOLD file with the container left running. Nothing changes or coordinates PID1's remaining lifetime. Its eventual exit stops the container and discards the tmpfs contents, including any unexported evidence. Keeping the stopped container and lane cannot recover those bytes.

This contradicts the required preservation-on-retention-failure policy even when the HOLD marker is successfully written. Near expiry, `prestop` can instead see not-running and raise before writing the marker. The finite process lifetime and HOLD policy need an explicit compatible preservation design before execution. Extending an inner target deadline or merely claiming a live HOLD is insufficient. No proof-loss experiment is authorized here.

## L3 — command timeouts do not bound remote work or guarantee retention/cleanup time

**Blocking deadline/process interaction.** Launcher:5–10,12–17,39–54,100–127.

`subprocess.run(..., timeout=...)` kills/waits for the local Docker CLI on timeout. It does not establish termination or quiescence of a process started by `docker exec`. A timed-out pytest exec can still be writing records/tmp files when `audit()` immediately exports them and computes LIVE hashes; a timed-out tar/hash exec can likewise remain active. No exact remote-process completion/termination receipt or quiescence check precedes retention or subsequent owned stop. A failed/partial archive remains STOP, but this is not a proven finite process/evidence lifecycle.

The admission assertion reserves1450 seconds, then permits1200 for pytest, leaving just over250 at the boundary. The two retention commands alone each allow180 seconds, followed by post manifests/identity and multiple cleanup commands. `call()` caps each command to the remaining global budget; it provides no protected cleanup budget. Once the budget is exhausted, the first `call('prestop', ...)` in finally raises before HOLD publication. Later cleanup failures also abort finally without a structured remaining-resource/HOLD receipt. No retry is present, which is appropriate, but the failure state is not safely handed back.

The outer1800 bound is additionally cooperative rather than comprehensive: host Git subprocesses in `hostcheck` have no timeout; hashing/copying/tar parsing and serialization have no outer supervisor. Archive member-count bounds are checked after `getmembers()` has already read the member list. Static source therefore cannot certify the advertised finite outer lifecycle. A revised launcher must account for exact remote process completion, finite retention and failure handback within its budget without altering inner deadlines.

## L4 — uncertain resource creation can release the lane without inventorying its container

**Blocking ownership/exception path.** Launcher:58–59,71–73,113–126.

`created` becomes true only after `call('create', args)` returns0. If Docker creates the container but the CLI times out or the result is lost, `call` returnsNone (or raises), the assertion fails, and `created` remains false. Finally skips every container inspection and HOLD branch, then releases the owner lane. An actually created named container can remain while lane-released.json claims release. There is no reconciliation of the uncertain create result against exact name/token/ID before releasing ownership. This counterexample needs no foreign mutation or test execution.

There is also an uncovered acquisition window: lane mkdir and owner-file writes occur before the encompassing try. Failure after mkdir can strand an incomplete lane without a structured acquisition-failure receipt. A start failure after a confirmed create takes the `created` branch, but its unconditional running assertion fails before recording an owned stopped-container hold. These paths prevent certification of all-path resource accounting.

On the nominal path the launcher checks the full create ID and owner token before stop, uses rm without-v, verifies anchored name absence and compares the complete lane owner before release. Those are useful controls. They do not close uncertain-create/acquisition failures. Destructive commands also use the mutable name after inspection rather than the captured immutable ID; the source does not establish ownership continuously across that interval. Never reconcile uncertainty by removing an unverified/foreign resource.

## Preserved source and bounded positive observations

Independently matched all193 manifest entries to current candidate bytes, with no mismatch. Entire guard, observer and entry are byte-identical to observer executable d5f493e6. Observer/entry acceptance, raw wait semantics, enrollment/reaper/latch, clock/deadline/caps and original target assertions have no launcher delta. Source manifest points to observer documentation8d5aba5, rather than claiming that launcher freeze is the copied application pin. Freeze→documentation is HANDOFF-only.

The declared image recipe inspects exact existing base `sha256:69bd9216869374d87fa14702ef380b4124a12f03f5f88a2681957ea81ee274cb`, requires the owned local base tag absent before tagging, verifies its ID, uses that tag in FROM and COPY only, and builds with --pull=false/--network=none. No pull/install or runtime alternate is selected. These are source intentions; no image availability/build or tag-race behavior was exercised.

The create argv declares CPU2, RAM2GiB, PID256, networknone, read-only root, UID10001, cap-dropALL, no-new-privileges and private tmpfs1GiB. Inspect checks cover those controls, binds/devices/ports and mounts. Host disk reserve6GiB, in-container available-memory/disk4GiB and tmpfs900MiB pretest checks are present. No host socket/secret/audio/service operation or port publication is selected. No actual admission/resource qualification follows from reading these checks.

Identity source checks CPython3.14.8, all42 locked versions, SQLite3.40.1, exact Debian libsqlite3-0 package/architecture and source ID, linked-library digest `2e6eef9a727f081f0d453b4e5e6cbd8b9ef8b6f86cbf7681cbad444d3b0b55c8`; ldd and compile options are recorded, with identical pre/post stdout required. Linux/amd64 provenance derives from the exact retained base and package expectations; no fresh host/image fact is certified. The identity helper's in-memory SQLite opens are future in-container operations only; none was performed by this review.

Private host proof uses umask077,0700 directories and0600 files. Sentinel includes NUL/FF/80 and requires byte equality. Export is direct binary tar with hard-dereference over ALL/tmp, followed by an independently collected LIVE regular-file SHA256 map and exact map equality. Inventory validates relative unique names/types, hashes regular files and treats symlinks as metadata; no extraction, docker cp or DB interpretation is selected. These nominal controls do not resolve L2/L3.

Record validation requires four47-integer rows,512-byte rows/16KiB total, child1 and owner2/3/4, owner/current PID/PPID/enrolled-child/raw18944 and empty phase; expected child timeout shape is checked. Unknown or changed exception/phase fails, rather than supplying acceptance. Actual offsets, type interpretation, exact selection/guard/report binding and cleanup remain independent receipt-review obligations. The unconditional final STOP explicitly withholds automatic cause acceptance. Expected pytest exit1 remains RED; diagnostic retention cannot attest unreached target assertions or turn the target PASS.

## Proposed command, required proof and remaining gate

After correction, re-freeze and fresh independent source review, separate parent authority would still be required for ONE future diagnosis only. No current runnable authorization is given. The proposed in-container command (cwd `/app`) remains:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/diagnostics/p-one-node python /app/diagnostics/p-one-node/entry.py --one-node
```

The corresponding host proposal is `python3 /Users/ben/.txcode/worktrees/py-audio-box/p-one-node-launcher-20261023/diagnostics/p-one-node/launcher.py`; **do not run this reviewed freeze**. Entry invokes pytest once with `-q -x --tb=short` and exactly `tests/service/test_recovery.py::test_t1_real_blocked_pipe_backpressure_deadline`, asserting singleton equality before fixtures. No separate collection,211-prefix, retry or tuning is authorized.

Required future proof includes frozen launcher/source/image/manifest and exact argv; pre-import/post source and runtime pins; exact selected-node/owner/token/PID binding; complete raw waits, empty phase, actual-EAGAIN marker, expected RED call/teardown reports and explicit unreached assertions; complete private47-field child/owner records with independently checked actual caller/runner offsets and active/original exception relation; binary sentinel, safe ALL/tmp inventory/archive and independent LIVE map equality; finite process completion and all failure/HOLD receipts; exact owned stop/inspect/rm-without-v/absence and lane release. Unknown, partial, missing, extra or changed-phase proof is STOP, never generic exit74 acceptance.

Historical actual-receipt audit remains146 PASS,1 FAIL,1 teardown ERROR,64 NOT RUN of211. That report and the observer review establish no historical active-final-exception identity for PID229. Even a usable future timeout observation would identify only that future run. No P/211, H/S/shared/grouped, installed/native/platform/durability, T2/C1295, full-service/release or activation gate is cleared.

## Static verification and handback

AST parse/compile without executing candidate code or writing pyc passed for launcher, runtime identity, entry and observer; the generated LIVE hash program also parses. Git diff --check passed. Recomputed hashes match FREEZE:

- launcher `ea419c21e81966861f79cb6af81c724bad5ebf78618de4c31fd56f554bdf0ee6`
- runtime identity `81acbdbb70df0a9c2a5db1f9daf86373cb22ea3b92371d7dddd924420a60237c`
- entry `5e7eb11bf5b93f7dd7dabe19e3d6dcdc9f16f40ec88a898f011bac63238bf600`
- observer `ee20c60ff9d5890ae6d9ceacf892443a4aea88dedf7a3952baf5bc98797de582`

Candidate and history remain unchanged. Sole write is this report, no commit. No runtime resource or cleanup obligation was acquired. L1–L4 require correction and independent source re-review before any execution decision; this review performs no repair.
