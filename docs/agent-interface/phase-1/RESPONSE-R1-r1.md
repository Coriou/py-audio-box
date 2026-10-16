# R1 review response — correction round 1

2026-10-16. Implementation candidate, **awaiting independent Sol-low re-review**.
R1 remains unaccepted; P1/public/unattended persistent/full Phase 1A activation and
final global Astra acceptance remain closed. W2's separately bounded acceptance
is not expanded by this correction.

## Identity and ownership

- Worktree: `/Users/ben/.txcode/worktrees/py-audio-box/t3-f6dcdb48`.
- Branch: `t3/r1-fail-closed-stop`.
- Initial clean parent: `946966ec4abc8cf83544d7b8e2576e364bb3a2fa`.
- Authorized R1 cherry-pick: `22e75ca2964d51b5b80ee91e9a14c7776c682d15`
  → local `3f816ea96d021eed23d2593da0753d6c707e799a`, without conflicts.
- Correction commit SHA is supplied by the implementation handoff; a committed
  report cannot contain its own commit hash.
- Correction files: `service/dispatcher.py`, `service/resource_store.py`,
  `tests/service/test_admission.py`, `tests/release/test_admission.py`,
  `docs/agent-interface/phase-1/{RESOURCE-R1.md,RESPONSE-R1-r1.md}` and
  `r1-evidence/correction-*.json`. No Admission/Core/DispatchStore hook changes
  were necessary. All other inherited R1/W2 files remain unchanged.

Read PRODUCT, implementation start/protocol, canonical plan's shared/resource,
CPU/GPU and Phase 1A gates, contracts 1.3.0, RESOURCE-R1 and the full original
`reviews/R1-r1-sol.md`, including both mandatory reproducers. No extra checkout or
subagent was created.

## F1 — failed fault INSERT loses authority

Corrected: memory fault authority and stop obligation precede the fallible INSERT.
Every later observation, ACK and error path retains that authority; submission and
publication explicitly consult it. A separate committed monitor witness, written after observing the fault and
before its fault INSERT, leaves `checking` when fault persistence fails. It cannot be cleared by a later
healthy scan. Restart exposes `persistence_fault=true`, quarantines, refuses fresh
keys/publication and retains execution reservations, even after confirmed reap.
The shared store guards reserve/acquire/release as well. No automatic repair,
unlatch or pretend recovery is provided. Persistent normal faults remain
irreversible too, but can release execution after the validated receipt/outcome.

The deterministic Dispatcher regression installs the exact review SQL trigger in
the owner-thread prespawn hook, parks observe, creates 10,000 output bytes, waits
for IntegrityError, removes the output, then permits healthy ALIVE observations.
It checks stop, continued quarantine, fresh-key refusal, no publication, no early
reservation release, no ACK after failed persistence, and conservative restart
with an empty resource_faults table and unresolved monitor witness.

## F2 — pending I/O loses transient stop

Corrected: stop obligation survives healthy scans and pending prepare/start/observe
completion or failure. A newly returned prepare is bound, then stopped without
starting the engine if pressure was already detected. Bound start/observe replies
return to stop service before another observation is launched. Stop failure/UNKNOWN
retains the obligation and execution charge; only the exact validated reap receipt
discharges it. One stop per bound runtime per session avoids a retry storm; the
existing independent supervisor deadline remains in force. Unknown unbound prepare
failure never creates another execution. Cancellation does not erase the resource
obligation or relax publication fencing.

The deterministic matrix exercises all three pending operations with/without a
cancel race and with successful/unknown stop, plus failed pending calls and repeated
faults. It checks the old reservation before receipt and after confirmed reap,
start/prepare/stop counts, publication refusal and fresh-key refusal after ACK and
later healthy scans. The installed resource-transient case uses the real packaged
supervisor/worker and UDS stop/reap/receipt, with one E1 policy-failure injection
while observe is pending. It asserts one physical launch and one stop; no physical
disk exhaustion or external probing is claimed.

## Validation and evidence

`correction-red.json` records baseline service/runtime bytes restored from local
cherry-pick `3f816ea` for two disposable test processes, then the correction bytes
restored in a finally block. This was not another checkout. Existing baseline
service/release tests: 137 passed, 18 permission-dependent skipped, 17 new tests
deselected. Both mandatory regression tests then failed for the reported behavior:
F1 status returned quarantined=false; F2 had no stop after transient pressure with
cancel=false. An earlier probe included cancellation and thereby masked F2 through
the independent cancel stop; the retained final red evidence separates that race.

Host development checks used the existing `/tmp/pab-0c-venv/bin/python`, with no
installation. The expanded full host suite passed 155 with 18 permission-dependent
skips before the final small receipt/monitor-state refinements. The final-source
container run below is authoritative for those refinements and includes those
permission-dependent cases.

The first full correction container pass passed all 173 E1 tests and the new
installed transient case, then failed inherited `crash-start-sent` → `recover`
with installed_timeout. Root cause: the initial witness design marked every
healthy scan `checking`; a pool-thread crash could interrupt that scan and cause
an unnecessary permanent quarantine. This was an actual correction regression,
not an accepted recovery change. The final design writes the witness only after
observing a fault, before the fault INSERT. Healthy scans do not touch it.
`correction-failure-1.json` preserves the failure; `correction-failed-cleanup-1.json`
adds completed cleanup, zero owned resources, and preservation of all 15 prior
containers/140 volumes. Focused checks after narrowing the witness passed 72 tests.
The subsequent final-source runner below repeats the full affected suite and
installed cases against the narrowed design; no failed run is called passing.

Commands (from the assigned tree, existing interpreter only):

```sh
/tmp/pab-0c-venv/bin/python -m pytest tests/service/test_admission.py tests/service/test_dispatcher.py -q -p no:cacheprovider
/tmp/pab-0c-venv/bin/python -m pytest tests/service tests/release -q -p no:cacheprovider
R1_EVIDENCE_PREFIX=correction- R1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-resource-r1.sh
git diff --check
```

The baseline negative-test subprocess commands/output are in correction-red.json;
the full container build/source-check/pytest/Compose commands/output are in each
runner receipt. Final results are recorded after completion below.

## Limits and next step

Only E1 deterministic policy/SQLite/concurrency and E2 fixed-synthetic installed
runtime evidence are claimed. Builds reuse the exact pinned dependency overlay;
this is static source/image verification, not a fresh-host installation. No model,
cloud, private audio, STT/Voicebox, remote E2E, prohibited run-e2e-test.sh, push, PR,
deploy, prune or user-data mount. The sole heavy lane uses a nonblocking lock,
network-none bounded containers, random owned resources and confirmed termination
before owned cleanup. Historical evidence and unrelated resources are preserved.

The new monitor deliberately sacrifices availability after interrupted or failed
fault persistence. It provides no operator recovery, cleanup or complete
SQLite/filesystem/power-loss guarantees. P1's full failure/fsync/missing-media and
persistent-volume acceptance matrix, D1–D5, real engine/residency profiles, hardware,
client/browser and acoustic qualification remain open. Next: independent Sol-low
read-only re-review of the exact committed candidate, then parent integration
verification only if accepted. This implementer makes no GO decision.

## Final corrected-source verification

Final command `R1_EVIDENCE_PREFIX=correction- R1_PYTHON=/tmp/pab-0c-venv/bin/python bash scripts/test-resource-r1.sh`
exited **0**: **173 E1 tests passed in 63.87s; all 36 installed E2 steps passed**.
The installed steps include crash/recovery pairs, not 36 independent fault
families. In particular, the previously regressed start-sent recovery passed,
as did the real-supervisor transient-stop case and all inherited W2/R1 cases.
No service/test/packaging source changed after this final run began.

Post-run static verification matched all 96 control, 18 worker and four packaging
manifest entries to current bytes, with zero mismatches. Docker image IDs were
rechecked against the final receipt. This verifies exact source/image identity,
not fresh installation, physical disk pressure or model/hardware qualification.
`git diff --check` passed. All 15 prior containers and 140 prior volumes were
preserved; zero owned containers/volumes remain after confirmed termination and
cleanup. Images and bounded evidence are retained; no prune occurred.

- Dependency overlay: `sha256:85141b253c04344134252c0878750756aef6bae3d4d4d083301004db7f43ce0e`.
- Worker: `sha256:7101cd00bbd2e49544233d62bf439d9483e5612e1254cbb750591ce8f14dbb9b` (unchanged W2 worker).
- Final control: `sha256:384bce5a1596e443f9fa195678374434fc5e42b95a469026f09617dea0e4b612`.
- Final correction-run.json SHA256: `14b2a8d3d70fa6bc63110d058c447e3fcad1cb112ae62f9b6dad07a920730c20`.

`correction-audit.json` records the exact correction paths, current branch/tree,
manifest checks, image identities, original-evidence hashes, final test output
and cleanup/preservation receipts. `correction-red.json`,
`correction-failure-1.json` and `correction-failed-cleanup-1.json` retain the
negative baseline and failed correction history. Original R1 evidence is byte
identical to the cherry-picked candidate.
