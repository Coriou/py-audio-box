# R1 correction round 2 implementation response

2026-10-16. Candidate for independent Sol-low review; no self-GO. R1 integration,
P1/public/unattended persistent activation, full Phase 1A and final Astra remain
closed. This implements the complete frozen authority decision, preserving F2's
bounded lifecycle semantics and explicitly qualifying W2 availability.

## Checkout and scope

Worktree `/Users/ben/.txcode/worktrees/py-audio-box/t3-f53dbe4c`, branch
`t3/r1-fault-authority-round2`, initially clean at
`bb60c7a` (parent planning plus W2). Only this checkout cherry-picked, in order:

- `22e75ca2964d51b5b80ee91e9a14c7776c682d15` → `b6368cb`.
- `0aaa70780a979d501eae959147f01c0f7475ebf3` → `04deec4`.

Both applied cleanly. The final implementation SHA is in the handoff; this report
cannot contain its own commit hash. Read PRODUCT, start/protocol, canonical
resource/shared/CPU-GPU/Phase 1 plan, CONTRACTS 1.3.0, RESOURCE-R1, response r1,
both full Sol reviews and the new Astra authority decision before design.
No extra worktree or subagent. No worker/contracts/packaging/dependencies/client/
studio/legacy/ledger/review changes. The allowlisted files changed are
`service/{resource_store,admission,dispatcher,dispatch_store,core}.py`,
`tests/service/test_admission.py`, `tests/release/{test_admission,test_dispatcher}.py`,
RESOURCE-R1, this response and new authority-prefixed JSON receipts.

## Finding response and protocol

F1: checking now commits BEFORE every R1 measurement, output/decoded check and
statvfs. Healthy completion requires the same live owner token and commits its
accounting. Fault UPDATE/reason INSERT are atomic and cannot roll back the earlier
begin. Local fault/quarantine/stop latch precedes fallible fault writes. Begin
failure performs no observation; completion failure leaves checking. A later
process cannot finish or clear an earlier process's epoch. Protocol revision 2
refuses old, missing or invalid populated monitor authority. Constructors do no
resource observation; startup reconstructs before release, publication and ACK.

Measurement is separated from reservation/acquisition arithmetic via private
preflight scope and immediate transaction serial. Allocation transactions retain
atomic job/events/settings/reservation and attempt/Authority changes. They do no
filesystem reads. Demand refusal does not create a resource fault. Direct trusted
allocator tests use the same scope. A call-site audit covers ResourceStore.measure,
check_disk/statvfs, Admission.check, startup/periodic/prepublication and both Core
submission/DispatchStore ordinary-or-replacement allocation. Other Authority.check
calls are unchanged worker envelope validation, not resource observations.

Unresolved authority blocks admission/acquisition/replacement/publication/intent
recovery/new ACK/execution release even after exact reap. Reap can record failed
or cancelled while retaining charges. Stored outcomes and already-terminal jobs
cannot bypass ACK fencing; earlier successes/releases remain intact. Known
committed ordinary faults preserve failed/cancelled exact-reap release/ACK.

F2: unresolved/known faults skip new resource scans, allowing pending lifecycle
replies, bind-then-stop, stop service, observation and collection to proceed.
One stop per bound runtime per session; failed/UNKNOWN stop is not reap. The full
pending prepare/start/observe × cancellation × stop success/UNKNOWN matrix remains,
including failed pending calls and no stop storm. After the first committed fault
new scans are deliberately skipped; repeated check requests cannot clear it.

## Regression evidence and availability

The disposable subprocess regression creates the real 10,000-byte output only
after independently reading committed checking. The original Admission check
raises resource_output_bytes; only then owner query_only is enabled. SQL tracing
requires the fault UPDATE itself to be attempted and denied. The immediate branch
calls os._exit before any close, finally, later loop or retry. The interval branch
restores writability/removes pressure and requests healthy checks while latched,
then dies; no new scan attests that old epoch. Both reopen writable, pressure-free
roots twice with valid timely fake receipts and assert no publication/admission/
replacement/ACK/release. The exact old INSERT trigger is retained, now with and
without cancellation.

Other deterministic regressions cover begin UPDATE and commit denial (zero
observations), completion UPDATE and commit denial (checking survives), healthy
pre-scan/mid-scan/post-return deaths, atomic reservation/acquisition rollback,
private/stale snapshots, no constructor scan, direct allocation and dedup,
old/missing protocol, queued unresolved cancellation, publication-intent recovery,
already committed success plus pending ACK, and failed-begin unchanged-healthy
restart. Owner barriers and persisted state establish ordering; polling only waits
for outcomes, never decides which crash result is acceptable.

The installed suite keeps the complete previous R1/W2 family list and adds actual
supervisor stop/reap and two writable restarts after the same denied-UPDATE
injection. Injection is E1 policy/SQLite evidence; real UDS lifecycle is E2.
Clean start-sent uses an owner barrier before the next scan and inspects persisted
healthy at actual send/death. A separate open-fence start-sent barrier inspects
checking and requires quarantine through two restarts. Clean recovery still
requires one physical execution, original binding/generation/deadline, publication,
ACK and release. Death during a healthy scan deliberately sacrifices availability;
no unconditional W2 reconnect claim remains for that overlap.

## Commands, results and limits

Development used the existing `/tmp/pab-0c-venv/bin/python`, no installation.
Initial focused failures exposed direct allocator tests needing private preflight,
tests that cleared old faults, and repeated-scan expectations incompatible with
latched scan suppression; these were corrected without weakening the required
terminal/charge/stop assertions. Query_only initially denied BEGIN IMMEDIATE;
the final test traces the fault UPDATE using the same-writer deferred fault
transaction. Historical receipts are preserved; only authority-prefixed receipts
are added.

```sh
/tmp/pab-0c-venv/bin/python -m pytest tests/service/test_admission.py tests/service/test_dispatcher.py -q -p no:cacheprovider
R1_EVIDENCE_PREFIX=authority- R1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-resource-r1.sh
git diff --check
```

Final results and exact source/image/cleanup audit follow after verification.
The complete affected suite is `tests/service` plus `tests/release` in the bounded
read-only network-none installed image, followed by installed scenarios. This is
not a legacy real-model campaign or fresh-host installation. The sole heavy lane
uses the existing nonblocking lock, random owned named resources, no user-data
mounts, prior-inventory digests and confirmed termination before owned removal.
No prune, models/private audio/cloud/STT/Voicebox, remote E2E, prohibited
`scripts/run-e2e-test.sh`, push/PR/deploy or operator cleanup/recovery was used.

P1 power-loss/fsync/partial-write/persistent-volume/operator recovery, D1–D5,
real adapters/residency/hardware, public clients/browser/acoustic acceptance remain
separate. The monitor is sampled control evidence, not a continuous filesystem
quota. Independent Sol-low review must assess the exact committed candidate;
this implementer supplies evidence, not finding closure or phase acceptance.

Two preliminary packaged runs are retained in authority-preliminary*.json and
failure-*.json, with complete cleanup in the preliminary receipts. The first
passed 191 E1 tests, the second 199; each passed the existing clean W2 crash
families and open-fence restart negative before failing the new installed
injection setup. Diagnostics proved staging is correctly read-only in control.
The final installed case therefore closes a clean setup control, has the test
operator write exactly 10,000 bytes through the owned worker staging volume, and
lets a newly constructed real control fence/observe it before query_only denies
its fault UPDATE. The operator removes only those injected bytes before the two
writable control restarts. No production mount or worker code changes. The
healthy-interval-before-death variant is an E1 subprocess test; the installed
immediate-death variant uses actual staged bytes and real supervisor stop/reap.
The policy/write-denial injection is E1, not physical disk exhaustion evidence.

## Final frozen-source result

Final command `R1_EVIDENCE_PREFIX=authority- R1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-resource-r1.sh`
exited **0**: **200 E1 tests passed in 82.95 seconds; all 43 installed E2 steps
passed**. Steps include setup/crash/recovery pairs, not 43 independent fault
families. Includes clean start-sent success, separately mandatory open-fence
quarantine, real 10,000-byte observation followed by denied fault UPDATE and
immediate death, two writable pressure-free restarts, and all inherited
R1/W2 refusal/cancel/replacement/stale/lost-ACK/publication/deadline/UNKNOWN families.

No service/test/packaging bytes changed after that run began. Post-run audit
matched all **96 control, 18 worker and four packaging entries**, zero mismatches;
actual Docker image IDs were rechecked. Worker/contracts/clients/packaging/reviews
and every inherited R1 evidence file were compared to correction base `04deec4`:
unchanged. `git diff --check` passed. The runner's nonblocking heavy-lane lock
covered the builds, complete suite and installed cases. It confirmed termination
before cleanup, preserved all **15 prior containers and 140 prior volumes** by
inventory digest, and left **zero owned containers or volumes**. Images and
non-sensitive evidence remain; no prune.

- Dependency: `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.
- Worker: `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b`.
- Control: `sha256:4cc00cca231ad920710b9e28aa36ea0e22838d87a14920e774a6b70774e7ec8a`.
- authority-run.json SHA256: `6ffb8bb3052acc45d5334885910fd8d90b4bb49c6a885cba4d1a9efa736342d2`.

`authority-run.json` contains the exact commands/results/source manifests and
runtime topology. `authority-audit.json` rechecks identities, unchanged inherited
evidence, results and cleanup; `authority-matrix.json` maps every required case to
its regression. The two authority-preliminary receipts contain completed cleanup
for the failed installed-injection attempts; authority-failure receipts retain
their original failures. The final run supersedes those candidates.

Candidate is ready for separate Sol-low adversarial review, not accepted by this
implementer. R1/P1/full Phase 1A and public/unattended persistent activation remain
closed. The explicit healthy-check-death availability cost is preserved.
